package asr

import (
	"encoding/binary"
	"math"
	"os"
	"testing"
)

func readFloat32File(t *testing.T, path string) []float32 {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	if len(data)%4 != 0 {
		t.Fatalf("%s: length %d not a multiple of 4", path, len(data))
	}
	out := make([]float32, len(data)/4)
	for i := range out {
		bits := binary.LittleEndian.Uint32(data[i*4:])
		out[i] = math.Float32frombits(bits)
	}
	return out
}

func TestLogMelSpectrogramMatchesPython(t *testing.T) {
	samples := readFloat32File(t, "testdata_audio_pcm16k.f32")
	want := readFloat32File(t, "testdata_mel_reference.f32")
	if len(want) != numMelBins*numMelFrame {
		t.Fatalf("reference mel has %d values, want %d", len(want), numMelBins*numMelFrame)
	}

	fe := newFeatureExtractor()
	got, realFrames := fe.LogMelSpectrogram(samples)
	if realFrames != 407 {
		t.Errorf("realFrames = %d, want 407", realFrames)
	}
	if len(got) != len(want) {
		t.Fatalf("got %d mel values, want %d", len(got), len(want))
	}

	var maxDiff float64
	var maxIdx int
	for i := range got {
		d := math.Abs(float64(got[i]) - float64(want[i]))
		if d > maxDiff {
			maxDiff = d
			maxIdx = i
		}
	}
	t.Logf("max abs diff vs Python reference: %.6g (at index %d: got=%v want=%v)", maxDiff, maxIdx, got[maxIdx], want[maxIdx])
	if maxDiff > 1e-3 {
		t.Errorf("mel spectrogram diverges from Python reference: max abs diff %.6g > 1e-3", maxDiff)
	}
}
