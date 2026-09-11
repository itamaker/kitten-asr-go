package asr

import (
	"encoding/binary"
	"fmt"
	"os"
)

// ReadWAV reads a PCM or IEEE-float WAV file and returns mono float32 samples
// in [-1, 1] at the file's native sample rate (resample to 16kHz separately,
// see Resample). Supports 8/16/24/32-bit integer and 32-bit float samples,
// any channel count (downmixed to mono by averaging).
func ReadWAV(path string) (samples []float32, sampleRate int, err error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, fmt.Errorf("asr: reading %s: %w", path, err)
	}
	return decodeWAV(data)
}

func decodeWAV(data []byte) ([]float32, int, error) {
	if len(data) < 12 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, 0, fmt.Errorf("asr: not a RIFF/WAVE file")
	}
	var (
		sampleRate    int
		numChannels   int
		bitsPerSample int
		audioFormat   uint16
		pcm           []byte
		haveFmt       bool
		havePCM       bool
	)
	pos := 12
	for pos+8 <= len(data) {
		chunkID := string(data[pos : pos+4])
		chunkSize := int(binary.LittleEndian.Uint32(data[pos+4 : pos+8]))
		body := pos + 8
		if body+chunkSize > len(data) {
			chunkSize = len(data) - body // tolerate a truncated/miswritten final chunk
		}
		switch chunkID {
		case "fmt ":
			if chunkSize < 16 {
				return nil, 0, fmt.Errorf("asr: fmt chunk too small")
			}
			audioFormat = binary.LittleEndian.Uint16(data[body : body+2])
			numChannels = int(binary.LittleEndian.Uint16(data[body+2 : body+4]))
			sampleRate = int(binary.LittleEndian.Uint32(data[body+4 : body+8]))
			bitsPerSample = int(binary.LittleEndian.Uint16(data[body+14 : body+16]))
			haveFmt = true
		case "data":
			pcm = data[body : body+chunkSize]
			havePCM = true
		}
		pos = body + chunkSize
		if chunkSize%2 == 1 {
			pos++ // chunks are word-aligned
		}
	}
	if !haveFmt || !havePCM {
		return nil, 0, fmt.Errorf("asr: missing fmt or data chunk")
	}
	out, err := pcmToFloat32(pcm, numChannels, bitsPerSample, audioFormat == 3)
	if err != nil {
		return nil, 0, err
	}
	return out, sampleRate, nil
}

// Resample changes the sample rate using linear interpolation -- adequate for
// ASR input (unlike TTS output quality, small aliasing artifacts from linear
// interpolation don't meaningfully affect recognition), and matches the
// approach kitten-tts-go's audio.Resample already uses for its own output
// resampling.
func Resample(samples []float32, srcRate, dstRate int) []float32 {
	if srcRate == dstRate || len(samples) == 0 {
		out := make([]float32, len(samples))
		copy(out, samples)
		return out
	}
	ratio := float64(dstRate) / float64(srcRate)
	outLen := int(float64(len(samples))*ratio + 0.999999)
	out := make([]float32, outLen)
	for i := range out {
		pos := float64(i) / ratio
		idx := int(pos)
		frac := float32(pos - float64(idx))
		if idx+1 < len(samples) {
			out[i] = samples[idx]*(1-frac) + samples[idx+1]*frac
		} else {
			out[i] = samples[len(samples)-1]
		}
	}
	return out
}
