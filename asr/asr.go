package asr

import (
	"fmt"
	"math"
	"runtime"
	"strings"
	"sync"
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
// consecutive (non-overlapping) chunks of at most MaxAudioSeconds, each
// transcribed independently and joined with a space. Interior boundaries are
// nudged earlier, onto the quietest point chunkCut can find in the last
// chunkCutSearchSeconds before the hard limit (see its doc comment) -- a
// cheap short-term-energy scan, not a trained VAD model, so it still cuts
// mid-word whenever the whole margin is loud (e.g. continuous speech with no
// nearby pause). The model also has no notion of carrying context between
// chunks or of precise word timing, so even a well-placed cut can cost a
// little accuracy right at the boundary -- that part is a model limitation,
// not something a smarter cut point can fix. Good enough for a first pass;
// real long-form handling (as in Whisper's own sliding-window algorithm)
// would need timestamp-aware resegmentation this model doesn't support. Not
// goroutine-safe -- see Model's doc comment.
func (m *Model) Transcribe(samples []float32, inputSampleRate int) (Result, error) {
	if inputSampleRate != sampleRate {
		samples = Resample(samples, inputSampleRate, sampleRate)
	}

	if len(samples) <= numSamples {
		return m.transcribeChunk(samples)
	}

	var texts []string
	language := ""
	for start := 0; start < len(samples); {
		hardEnd := start + numSamples
		if hardEnd > len(samples) {
			hardEnd = len(samples)
		}
		end := chunkCut(samples, start, hardEnd)
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
		start = end
	}
	return Result{Language: language, Text: strings.Join(texts, " ")}, nil
}

// chunkCutSearchSeconds is how far before a hard chunk boundary chunkCut
// searches for a quieter cut point.
const chunkCutSearchSeconds = 2.0

// chunkCutFrameSamples is chunkCut's short-term-energy analysis frame size
// (20ms at 16kHz).
const chunkCutFrameSamples = sampleRate / 50

// chunkCutStepSamples is chunkCut's scan hop (50% frame overlap).
const chunkCutStepSamples = chunkCutFrameSamples / 2

// chunkCut returns where Transcribe should end the chunk starting at
// samples[start:], given a hard upper bound hardEnd (start+numSamples, or
// len(samples) if that's smaller). The result is always in [start, hardEnd].
//
// If hardEnd is already the true end of the audio there's nothing to gain by
// moving it, so it's returned unchanged. Otherwise chunkCut scans the last
// chunkCutSearchSeconds before hardEnd in chunkCutFrameSamples-sized frames
// and returns the start of whichever frame has the lowest RMS energy -- the
// most silence-like point in that margin, and therefore the least likely to
// land mid-word. It falls back to hardEnd itself only when the search window
// is too short to contain two frames (a final chunk barely over numSamples).
func chunkCut(samples []float32, start, hardEnd int) int {
	if hardEnd >= len(samples) {
		return hardEnd
	}
	searchStart := hardEnd - int(chunkCutSearchSeconds*sampleRate)
	if searchStart < start {
		searchStart = start
	}
	if hardEnd-searchStart < 2*chunkCutFrameSamples {
		return hardEnd
	}

	bestPos := hardEnd
	bestEnergy := math.Inf(1)
	for pos := searchStart; pos+chunkCutFrameSamples <= hardEnd; pos += chunkCutStepSamples {
		var energy float64
		for _, s := range samples[pos : pos+chunkCutFrameSamples] {
			energy += float64(s) * float64(s)
		}
		if energy < bestEnergy {
			bestEnergy = energy
			bestPos = pos
		}
	}
	return bestPos
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

	// decoder.Step's ONNX graph only ever carries the *last* position forward
	// (see tools/decoder_wrapper.py), so its result is always exactly one row
	// regardless of how many tokens were fed in -- no slicing needed, unlike a
	// graph that returned a row for every position.
	logits, err := m.decodeStep(cache, inputsEmbeds, len(promptIDs))
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
		logits, err = m.decodeStep(cache, m.embedRow(nextID), 1)
		if err != nil {
			return Result{}, fmt.Errorf("asr: decoder step %d: %w", step, err)
		}
		nextID = argmax(logits)
	}

	return m.parseOutput(generated), nil
}

// decodeStep runs one decoder.Step and returns vocabSize-length logits
// either way, regardless of whether the graph itself projects to vocab
// space or leaves that to projectLogits -- see textDecoder.hiddenOutput.
func (m *Model) decodeStep(cache *kvCache, inputsEmbeds []float32, seqLen int) ([]float32, error) {
	out, err := m.decoder.Step(cache, inputsEmbeds, seqLen, m.cfg.HiddenSize)
	if err != nil {
		return nil, err
	}
	if m.decoder.hiddenOutput {
		return m.projectLogits(out), nil
	}
	return out, nil
}

// projectLogits computes vocabSize logits from a single hiddenSize hidden
// state by dotting it against every row of the embedding matrix -- the
// manual equivalent of the nn.Linear(hidden_size, vocab_size, bias=False)
// lm_head a tied-embedding model's decoder.onnx no longer bakes in (see
// config.TieWordEmbeddings). embed_tokens.weight and lm_head.weight are the
// same matrix in such a checkpoint, and m.embed is already that matrix
// (loaded once for input embedding lookups), so this reuses it instead of
// the graph carrying a second copy just for the output side. Parallelized
// across GOMAXPROCS workers: vocabSize (151936) x hiddenSize floats is tens
// of MB streamed through memory on every single decode step, and this runs
// once per generated token.
func (m *Model) projectLogits(hidden []float32) []float32 {
	vocab := m.cfg.VocabSize
	hiddenSize := m.cfg.HiddenSize
	logits := make([]float32, vocab)

	workers := runtime.GOMAXPROCS(0)
	if workers > vocab {
		workers = 1
	}
	chunk := (vocab + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		start := w * chunk
		end := start + chunk
		if start >= vocab {
			break
		}
		if end > vocab {
			end = vocab
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for v := start; v < end; v++ {
				row := m.embed[v*hiddenSize : (v+1)*hiddenSize : (v+1)*hiddenSize]
				var sum float32
				for h, hv := range hidden {
					sum += hv * row[h]
				}
				logits[v] = sum
			}
		}(start, end)
	}
	wg.Wait()
	return logits
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
