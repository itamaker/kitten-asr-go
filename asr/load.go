package asr

import (
	"fmt"
	"os"
	"path/filepath"
)

// Model is a loaded kitten-asr engine: two ONNX sessions (audio encoder, text
// decoder), a BPE tokenizer, the embedding table, and derived config. Not
// goroutine-safe -- callers needing concurrent transcription should either
// guard a shared *Model with a mutex or load one *Model per goroutine (see
// kitten-tts-go's tts.Model for the same convention on the TTS side).
type Model struct {
	cfg       *config
	tokenizer *tokenizer
	feature   *featureExtractor
	encoder   *audioEncoder
	decoder   *textDecoder
	embed     []float32 // (vocabSize, hiddenSize) row-major
}

// Option configures New.
type Option func(*options)

type options struct {
	intraOpThreads int
}

// DefaultIntraOpThreads is the ONNX Runtime intra-op thread count New uses
// when WithIntraOpThreads isn't supplied. Both graphs are one long sequential
// chain of layers (no independent parallel subgraphs to spread across an
// inter-op pool), so this is the only threading knob that matters. Higher
// isn't reliably faster: on a memory-constrained machine, spinning up
// runtime.NumCPU() threads for every one of the ~20 tiny per-token decode
// steps in a transcription added enough scheduling/memory overhead that
// repeated calls got progressively *slower*, not faster (measured on a
// 12-core/7.7GB-RAM box) -- 4 matches kitten-tts-go's own default for the
// same reason on the TTS side.
const DefaultIntraOpThreads = 4

// WithIntraOpThreads overrides DefaultIntraOpThreads. Pass 0 to let
// onnxruntime pick its own default, or a server handling concurrent requests
// (each with its own *Model) may want fewer threads per model so they don't
// contend with each other.
func WithIntraOpThreads(n int) Option {
	return func(o *options) { o.intraOpThreads = n }
}

// New loads a kitten-asr model directory: config.json, vocab.json,
// merges.txt, added_tokens.json, and three model artifacts named
// kitten_asr_<model>_audio_encoder.onnx, kitten_asr_<model>_decoder.onnx
// (plus its external-data companion, always literally named
// "decoder.onnx.data" regardless of the model -- ONNX Runtime resolves it
// by the location string recorded inside the .onnx file, not by matching
// the .onnx file's own name, so it doesn't need the prefix too),
// kitten_asr_<model>_embed_tokens.bin -- the "kitten_..."
// prefix matches KittenML's own ONNX naming convention on the TTS side
// (e.g. kitten_tts_nano_v0_8.onnx); New finds them by suffix so it doesn't
// need to know which model's directory it was given (see tools/ in this
// repo for how these are produced from KittenML's original checkpoint --
// they are not vendored here or by KittenML, matching kitten-tts-go's
// "models are downloaded separately" convention).
func New(dir string, opts ...Option) (*Model, error) {
	o := options{intraOpThreads: DefaultIntraOpThreads}
	for _, opt := range opts {
		opt(&o)
	}

	cfg, err := loadConfig(dir)
	if err != nil {
		return nil, err
	}
	tok, err := loadTokenizer(dir)
	if err != nil {
		return nil, err
	}

	embedPath, err := findModelFile(dir, "_embed_tokens.bin")
	if err != nil {
		return nil, err
	}
	embed, err := loadEmbedTokens(embedPath, cfg.VocabSize, cfg.HiddenSize)
	if err != nil {
		return nil, err
	}

	audioPath, err := findModelFile(dir, "_audio_encoder.onnx")
	if err != nil {
		return nil, err
	}
	encoder, err := loadAudioEncoder(audioPath, o.intraOpThreads)
	if err != nil {
		return nil, err
	}
	decoderPath, err := findModelFile(dir, "_decoder.onnx")
	if err != nil {
		encoder.Close()
		return nil, err
	}
	decoder, err := loadTextDecoder(decoderPath, cfg.NumLayers, cfg.NumKVHeads, cfg.HeadDim, o.intraOpThreads)
	if err != nil {
		encoder.Close()
		return nil, err
	}

	return &Model{
		cfg:       cfg,
		tokenizer: tok,
		feature:   newFeatureExtractor(),
		encoder:   encoder,
		decoder:   decoder,
		embed:     embed,
	}, nil
}

// findModelFile locates the one file in dir whose name ends in suffix (e.g.
// "_decoder.onnx") -- the exact "kitten_asr_<model>" prefix varies per
// model, so callers match on suffix rather than a hardcoded full name.
func findModelFile(dir, suffix string) (string, error) {
	matches, err := filepath.Glob(filepath.Join(dir, "*"+suffix))
	if err != nil {
		return "", fmt.Errorf("asr: globbing for *%s in %s: %w", suffix, dir, err)
	}
	switch len(matches) {
	case 0:
		return "", fmt.Errorf("asr: no file matching *%s found in %s", suffix, dir)
	case 1:
		return matches[0], nil
	default:
		return "", fmt.Errorf("asr: multiple files matching *%s found in %s: %v", suffix, dir, matches)
	}
}

func loadEmbedTokens(path string, vocabSize, hiddenSize int) ([]float32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("asr: reading %s: %w", filepath.Base(path), err)
	}
	want := vocabSize * hiddenSize * 4
	if len(data) != want {
		return nil, fmt.Errorf("asr: %s is %d bytes, want %d (vocab_size=%d x hidden_size=%d x 4)",
			filepath.Base(path), len(data), want, vocabSize, hiddenSize)
	}
	return bytesToFloat32(data), nil
}

// Close releases the ONNX sessions. Safe to call once; a *Model must not be
// used afterward.
func (m *Model) Close() error {
	var firstErr error
	if err := m.decoder.Close(); err != nil && firstErr == nil {
		firstErr = err
	}
	if err := m.encoder.Close(); err != nil && firstErr == nil {
		firstErr = err
	}
	return firstErr
}
