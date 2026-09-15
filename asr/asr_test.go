package asr

import "testing"

// tone returns n samples of a constant-amplitude sine wave, standing in for
// "loud" audio in chunkCut tests.
func tone(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = 0.5
		if i%2 == 1 {
			out[i] = -0.5
		}
	}
	return out
}

func TestChunkCutTrueEndUnchanged(t *testing.T) {
	samples := tone(numSamples)
	if got := chunkCut(samples, 0, len(samples)); got != len(samples) {
		t.Errorf("chunkCut at the true end of audio = %d, want %d (unchanged)", got, len(samples))
	}
}

func TestChunkCutFindsSilence(t *testing.T) {
	// Loud, then a clear silent gap starting 1s before the hard boundary,
	// then loud again past it (so hardEnd is an interior cut, not the true
	// end of the audio).
	hardEnd := numSamples
	silenceStart := hardEnd - sampleRate
	silenceLen := sampleRate / 4 // 0.25s of true silence
	total := hardEnd + sampleRate

	samples := tone(total)
	for i := silenceStart; i < silenceStart+silenceLen; i++ {
		samples[i] = 0
	}

	got := chunkCut(samples, 0, hardEnd)
	if got < silenceStart || got >= silenceStart+silenceLen {
		t.Errorf("chunkCut = %d, want inside the silent gap [%d, %d)", got, silenceStart, silenceStart+silenceLen)
	}
}

func TestChunkCutDegenerateWindowFallsBackToHardEnd(t *testing.T) {
	// hardEnd only a few samples past start: search window can't fit even
	// two analysis frames, so chunkCut must fall back to hardEnd rather than
	// index out of range.
	start := 0
	hardEnd := chunkCutFrameSamples
	total := hardEnd + sampleRate // interior cut, not the true end
	samples := tone(total)

	if got := chunkCut(samples, start, hardEnd); got != hardEnd {
		t.Errorf("chunkCut with a degenerate search window = %d, want %d (hardEnd)", got, hardEnd)
	}
}

func TestChunkCutNeverBeforeStartOrAfterHardEnd(t *testing.T) {
	start := numSamples // simulate the 2nd chunk of a longer transcription
	hardEnd := start + numSamples
	samples := tone(hardEnd + sampleRate)

	got := chunkCut(samples, start, hardEnd)
	if got < start || got > hardEnd {
		t.Errorf("chunkCut = %d, want in [%d, %d]", got, start, hardEnd)
	}
}
