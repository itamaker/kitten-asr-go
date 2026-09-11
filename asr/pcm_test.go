package asr

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestDecodePCM16Mono(t *testing.T) {
	want := []float32{0, 0.5, -0.5, 1, -1}
	buf := make([]byte, len(want)*2)
	for i, s := range want {
		v := int16(math.Round(float64(s) * 32767))
		binary.LittleEndian.PutUint16(buf[i*2:], uint16(v))
	}

	got, sampleRate, err := DecodePCM(buf, 16000, 16, 1, false)
	if err != nil {
		t.Fatalf("DecodePCM: %v", err)
	}
	if sampleRate != 16000 {
		t.Errorf("sampleRate = %d, want 16000", sampleRate)
	}
	if len(got) != len(want) {
		t.Fatalf("got %d samples, want %d", len(got), len(want))
	}
	for i := range want {
		if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-4 {
			t.Errorf("sample %d: got %v, want %v", i, got[i], want[i])
		}
	}
}

func TestDecodePCMStereoDownmix(t *testing.T) {
	// Two frames of (left, right) 16-bit samples; mono output should be the
	// per-frame average.
	frames := [][2]int16{{32767, -32768}, {0, 16384}}
	buf := make([]byte, 0, len(frames)*4)
	for _, f := range frames {
		var b [4]byte
		binary.LittleEndian.PutUint16(b[0:], uint16(f[0]))
		binary.LittleEndian.PutUint16(b[2:], uint16(f[1]))
		buf = append(buf, b[:]...)
	}

	got, _, err := DecodePCM(buf, 8000, 16, 2, false)
	if err != nil {
		t.Fatalf("DecodePCM: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("got %d samples, want 2", len(got))
	}
	// frame 0: (32767/32768 + -32768/32768) / 2 ~= 0
	if diff := math.Abs(float64(got[0])); diff > 1e-4 {
		t.Errorf("frame 0 = %v, want ~0", got[0])
	}
	// frame 1: (0 + 16384/32768) / 2 = 0.25
	if diff := math.Abs(float64(got[1] - 0.25)); diff > 1e-4 {
		t.Errorf("frame 1 = %v, want ~0.25", got[1])
	}
}

func TestDecodePCMFloat32(t *testing.T) {
	want := []float32{0.1, -0.9, 0.42}
	buf := make([]byte, len(want)*4)
	for i, s := range want {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(s))
	}
	got, _, err := DecodePCM(buf, 44100, 32, 1, true)
	if err != nil {
		t.Fatalf("DecodePCM: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("got %d samples, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("sample %d: got %v, want %v", i, got[i], want[i])
		}
	}
}

func TestDecodePCMInvalidParams(t *testing.T) {
	if _, _, err := DecodePCM(nil, 0, 16, 1, false); err == nil {
		t.Error("expected error for sampleRate=0")
	}
	if _, _, err := DecodePCM(nil, 16000, 12, 1, false); err == nil {
		t.Error("expected error for non-byte-aligned bitsPerSample")
	}
	if _, _, err := DecodePCM(nil, 16000, 16, 0, false); err == nil {
		t.Error("expected error for numChannels=0")
	}
}
