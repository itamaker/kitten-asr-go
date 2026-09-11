package asr

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"

	"github.com/hajimehoshi/go-mp3"
)

// DecodeMP3 decodes an MP3 byte stream to mono float32 samples in [-1, 1] at
// the stream's native sample rate (resample to 16kHz separately, see
// Resample). go-mp3 always produces 16-bit stereo PCM regardless of the
// source's channel count, downmixed to mono here by averaging.
func DecodeMP3(data []byte) (samples []float32, sampleRate int, err error) {
	dec, err := mp3.NewDecoder(bytes.NewReader(data))
	if err != nil {
		return nil, 0, fmt.Errorf("asr: decoding mp3: %w", err)
	}
	pcm, err := io.ReadAll(dec)
	if err != nil && err != io.EOF {
		return nil, 0, fmt.Errorf("asr: decoding mp3: %w", err)
	}
	if len(pcm)%4 != 0 {
		pcm = pcm[:len(pcm)-len(pcm)%4]
	}
	out := make([]float32, len(pcm)/4)
	for i := range out {
		l := int16(binary.LittleEndian.Uint16(pcm[i*4:]))
		r := int16(binary.LittleEndian.Uint16(pcm[i*4+2:]))
		out[i] = (float32(l) + float32(r)) / 2 / 32768
	}
	return out, dec.SampleRate(), nil
}
