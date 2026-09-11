package asr

import (
	"encoding/binary"
	"fmt"
	"math"
)

// DecodePCM converts headerless raw PCM bytes to mono float32 samples in
// [-1, 1], given format parameters a caller must supply out of band -- raw
// PCM (unlike WAV/FLAC/MP3) has no header describing its own sample rate,
// bit depth, or channel count. Meant for real-time sources that hand over
// bare captured audio frames directly (e.g. a browser's AudioWorklet/
// ScriptProcessor output, or a custom mic-capture client) rather than a
// self-describing file.
//
// isFloat selects interpretation of bitsPerSample: bitsPerSample==32 with
// isFloat==true means 32-bit IEEE-754 float samples; otherwise integer PCM
// at 8/16/24/32 bits, little-endian, signed except for 8-bit (unsigned,
// matching the WAV/most common convention). Multi-channel input is
// downmixed to mono by averaging, same as ReadWAV/DecodeMP3/DecodeFLAC.
func DecodePCM(data []byte, sampleRate, bitsPerSample, numChannels int, isFloat bool) (samples []float32, outSampleRate int, err error) {
	if sampleRate <= 0 {
		return nil, 0, fmt.Errorf("asr: invalid sample rate %d", sampleRate)
	}
	out, err := pcmToFloat32(data, numChannels, bitsPerSample, isFloat)
	if err != nil {
		return nil, 0, err
	}
	return out, sampleRate, nil
}

// pcmToFloat32 downmixes interleaved little-endian PCM to mono float32.
// Shared by DecodePCM and the WAV decoder (a WAV "data" chunk is this exact
// PCM layout, just with a RIFF header naming the parameters instead of the
// caller supplying them).
func pcmToFloat32(pcm []byte, numChannels, bitsPerSample int, isFloat bool) ([]float32, error) {
	if numChannels < 1 {
		return nil, fmt.Errorf("asr: invalid channel count %d", numChannels)
	}
	if bitsPerSample%8 != 0 || bitsPerSample < 8 {
		return nil, fmt.Errorf("asr: unsupported bits per sample %d", bitsPerSample)
	}
	bytesPerSample := bitsPerSample / 8
	frameSize := bytesPerSample * numChannels
	numFrames := len(pcm) / frameSize

	decodeOne := func(b []byte) float32 {
		switch {
		case isFloat && bitsPerSample == 32:
			return math.Float32frombits(binary.LittleEndian.Uint32(b))
		case bitsPerSample == 8: // unsigned 8-bit PCM
			return (float32(b[0]) - 128) / 128
		case bitsPerSample == 16:
			return float32(int16(binary.LittleEndian.Uint16(b))) / 32768
		case bitsPerSample == 24:
			v := int32(b[0]) | int32(b[1])<<8 | int32(b[2])<<16
			if v&0x800000 != 0 {
				v |= -1 << 24 // sign-extend
			}
			return float32(v) / 8388608
		case bitsPerSample == 32:
			return float32(int32(binary.LittleEndian.Uint32(b))) / 2147483648
		default:
			return 0
		}
	}

	out := make([]float32, numFrames)
	for f := 0; f < numFrames; f++ {
		var sum float32
		base := f * frameSize
		for c := 0; c < numChannels; c++ {
			sum += decodeOne(pcm[base+c*bytesPerSample : base+(c+1)*bytesPerSample])
		}
		out[f] = sum / float32(numChannels)
	}
	return out, nil
}
