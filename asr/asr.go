package asr

import (
	"fmt"
	"strings"
)

// MaxAudioSeconds is the model's hard per-window limit: the audio tower's
// position embedding table only covers this many seconds
// (max_source_positions in the original config). Transcribe splits longer
// input into consecutive windows of this size automatically -- see its doc
// comment for the tradeoffs of that approach.
const MaxAudioSeconds = chunkSecs

// Result is one Transcribe call's output. The model prepends a free-text
// language guess before the transcript proper, separated by the <asr_text>
// special token; Language is empty if that marker wasn't found (e.g. on an
// empty/silent input, where the model may emit nothing at all).
type Result struct {
	Language string
	Text     string
}

// Transcribe recognizes speech in samples (mono, at inputSampleRate Hz) and
// returns the transcript. Audio longer than MaxAudioSeconds is split into
// consecutive (non-overlapping) MaxAudioSeconds chunks, each transcribed
// independently and joined with a space -- the model has no notion of
// carrying context between chunks or of precise word timing, so a chunk
// boundary landing mid-word or mid-sentence can cost a little accuracy right
// at the cut. Good enough for a first pass; real long-form handling (as in
// Whisper's own sliding-window algorithm) would need timestamp-aware
// resegmentation this model doesn't support. Not goroutine-safe -- see
// Model's doc comment.
func (m *Model) Transcribe(samples []float32, inputSampleRate int) (Result, error) {
	if inputSampleRate != sampleRate {
		samples = Resample(samples, inputSampleRate, sampleRate)
	}

	if len(samples) <= numSamples {
		return m.transcribeChunk(samples)
	}

	var texts []string
	language := ""
	for start := 0; start < len(samples); start += numSamples {
		end := start + numSamples
		if end > len(samples) {
			end = len(samples)
		}
		res, err := m.transcribeChunk(samples[start:end])
		if err != nil {
			return Result{}, fmt.Errorf("asr: chunk at %.1fs: %w", float64(start)/float64(sampleRate), err)
		}
		if text := strings.TrimSpace(res.Text); text != "" {
			texts = append(texts, text)
		}
		if language == "" {
			language = res.Language
		}
	}
	return Result{Language: language, Text: strings.Join(texts, " ")}, nil
}

// transcribeChunk runs the model on a single window of at most
// MaxAudioSeconds of 16kHz audio.
func (m *Model) transcribeChunk(samples []float32) (Result, error) {
	mel, _ := m.feature.LogMelSpectrogram(samples)
	audioEmbeds, err := m.encoder.Run(mel, m.cfg.HiddenSize)
	if err != nil {
		return Result{}, fmt.Errorf("asr: audio encoder: %w", err)
	}

	audioPadCount := m.cfg.audioOutLen()
	if len(audioEmbeds) != audioPadCount*m.cfg.HiddenSize {
		return Result{}, fmt.Errorf("asr: audio encoder returned %d floats, want %d (%d x %d)",
			len(audioEmbeds), audioPadCount*m.cfg.HiddenSize, audioPadCount, m.cfg.HiddenSize)
	}

	promptText := buildPrompt("", audioPadCount)
	promptIDs, err := m.tokenizer.Encode(promptText)
	if err != nil {
		return Result{}, fmt.Errorf("asr: tokenizing prompt: %w", err)
	}

	inputsEmbeds := make([]float32, len(promptIDs)*m.cfg.HiddenSize)
	audioCursor := 0
	for i, id := range promptIDs {
		dst := inputsEmbeds[i*m.cfg.HiddenSize : (i+1)*m.cfg.HiddenSize]
		if id == m.cfg.AudioTokenID {
			copy(dst, audioEmbeds[audioCursor*m.cfg.HiddenSize:(audioCursor+1)*m.cfg.HiddenSize])
			audioCursor++
		} else {
			copy(dst, m.embedRow(id))
		}
	}

	cache, err := newKVCache(m.cfg.NumLayers, m.cfg.NumKVHeads, m.cfg.HeadDim)
	if err != nil {
		return Result{}, fmt.Errorf("asr: allocating KV cache: %w", err)
	}
	defer cache.Destroy()

	// decoder.Step's ONNX graph only ever projects the *last* position through
	// lm_head (see tools/decoder_wrapper.py), so logits is always exactly one
	// vocabSize-length row regardless of how many tokens were fed in -- no
	// slicing needed, unlike a graph that returned logits for every position.
	logits, err := m.decoder.Step(cache, inputsEmbeds, len(promptIDs), m.cfg.HiddenSize)
	if err != nil {
		return Result{}, fmt.Errorf("asr: decoder prefill: %w", err)
	}
	nextID := argmax(logits)

	const maxNewTokens = 400
	generated := make([]int32, 0, maxNewTokens)
	for step := 0; step < maxNewTokens; step++ {
		generated = append(generated, nextID)
		if nextID == m.cfg.EOSTokenID {
			break
		}
		logits, err = m.decoder.Step(cache, m.embedRow(nextID), 1, m.cfg.HiddenSize)
		if err != nil {
			return Result{}, fmt.Errorf("asr: decoder step %d: %w", step, err)
		}
		nextID = argmax(logits)
	}

	return m.parseOutput(generated), nil
}

// embedRow returns the hiddenSize-length embedding row for token id (no
// copy -- callers that mutate the destination must copy() out of it, as
// Transcribe's prompt-building loop does).
func (m *Model) embedRow(id int32) []float32 {
	return m.embed[int(id)*m.cfg.HiddenSize : (int(id)+1)*m.cfg.HiddenSize]
}

// parseOutput decodes generated ids and splits kitten-asr's
// "<language name><asr_text><transcript>" output format.
func (m *Model) parseOutput(ids []int32) Result {
	// Split in id-space, not in decoded text: Decode(ids, true) (skipSpecial)
	// never emits <asr_text> as a substring at all, so searching for its
	// decoded form in the decoded string can never find it (nor would
	// searching skip-special decoded text for any special token's text be
	// robust in general, decoded-text collisions aside).
	if asrID, ok := m.tokenizer.SpecialTokenID("<asr_text>"); ok {
		for i, id := range ids {
			if id == asrID {
				prefix := strings.TrimSpace(m.tokenizer.Decode(ids[:i], true))
				// Observed format is literally the two words "language
				// <name>" (e.g. "language English") before the marker,
				// consistently across every model/input tried, and "<name>"
				// matches config.json's support_languages list verbatim --
				// strip the fixed leading word so Language is just the name.
				prefix = strings.TrimPrefix(prefix, "language ")
				return Result{
					Language: prefix,
					Text:     m.tokenizer.Decode(ids[i+1:], true),
				}
			}
		}
	}
	return Result{Text: m.tokenizer.Decode(ids, true)}
}

func argmax(logits []float32) int32 {
	best := int32(0)
	bestVal := logits[0]
	for i := 1; i < len(logits); i++ {
		if logits[i] > bestVal {
			bestVal = logits[i]
			best = int32(i)
		}
	}
	return best
}
