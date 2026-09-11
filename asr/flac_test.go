package asr

import (
	"bytes"
	"math"
	"testing"

	"github.com/mewkiz/flac"
	"github.com/mewkiz/flac/frame"
	"github.com/mewkiz/flac/meta"
)

// encodeFLACForTest builds a minimal mono 16-bit FLAC stream from known
// samples, purely so TestDecodeFLAC can round-trip against a value it
// already knows -- mirrors kitten-tts-go's audio.encodeFLAC.
func encodeFLACForTest(t *testing.T, samples []float32, sampleRate uint32) []byte {
	t.Helper()
	pcm := make([]int32, len(samples))
	for i, s := range samples {
		pcm[i] = int32(math.Round(float64(s) * 32767))
	}

	info := &meta.StreamInfo{
		BlockSizeMin:  uint16(len(pcm)),
		BlockSizeMax:  uint16(len(pcm)),
		SampleRate:    sampleRate,
		NChannels:     1,
		BitsPerSample: 16,
		NSamples:      uint64(len(pcm)),
	}
	var out bytes.Buffer
	enc, err := flac.NewEncoder(&out, info)
	if err != nil {
		t.Fatalf("flac.NewEncoder: %v", err)
	}
	f := &frame.Frame{
		Header: frame.Header{
			HasFixedBlockSize: true,
			BlockSize:         uint16(len(pcm)),
			SampleRate:        sampleRate,
			Channels:          frame.ChannelsMono,
			BitsPerSample:     16,
		},
		Subframes: []*frame.Subframe{{
			SubHeader: frame.SubHeader{Pred: frame.PredVerbatim},
			Samples:   pcm,
			NSamples:  len(pcm),
		}},
	}
	if err := enc.WriteFrame(f); err != nil {
		t.Fatalf("WriteFrame: %v", err)
	}
	if err := enc.Close(); err != nil {
		t.Fatalf("enc.Close: %v", err)
	}
	return out.Bytes()
}

func TestDecodeFLACRoundTrip(t *testing.T) {
	// FLAC requires a minimum block size of 16 samples.
	want := []float32{0, 0.5, -0.5, 0.25, -0.99, 0.1, -0.1, 0.75,
		-0.75, 0.33, -0.33, 0.05, -0.05, 0.9, -0.9, 0.2}
	data := encodeFLACForTest(t, want, 16000)

	got, sampleRate, err := DecodeFLAC(data)
	if err != nil {
		t.Fatalf("DecodeFLAC: %v", err)
	}
	if sampleRate != 16000 {
		t.Errorf("sampleRate = %d, want 16000", sampleRate)
	}
	if len(got) != len(want) {
		t.Fatalf("got %d samples, want %d", len(got), len(want))
	}
	// FLAC here is lossless over the 16-bit samples we fed in, but those
	// were themselves rounded from float32 -- allow one 16-bit quantization
	// step of slack (encodeFLACForTest scales by 32767, DecodeFLAC by the
	// full 32768 range, matching WAV/PCM's asymmetric int16 convention, so
	// the two don't perfectly cancel).
	for i := range want {
		if diff := math.Abs(float64(got[i] - want[i])); diff > 2.0/32768 {
			t.Errorf("sample %d: got %v, want %v", i, got[i], want[i])
		}
	}
}

func TestDecodeAudioDispatchesFLAC(t *testing.T) {
	want := make([]float32, 16) // FLAC requires a minimum block size of 16 samples
	for i := range want {
		want[i] = float32(i) / 32
	}
	data := encodeFLACForTest(t, want, 8000)
	samples, sampleRate, err := DecodeAudio(data, "clip.FLAC") // extension matching is case-insensitive
	if err != nil {
		t.Fatalf("DecodeAudio: %v", err)
	}
	if sampleRate != 8000 || len(samples) != len(want) {
		t.Errorf("got sampleRate=%d len=%d, want 8000/%d", sampleRate, len(samples), len(want))
	}
}
