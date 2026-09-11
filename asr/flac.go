package asr

import (
	"bytes"
	"fmt"
	"io"

	"github.com/mewkiz/flac"
)

// DecodeFLAC decodes a FLAC byte stream to mono float32 samples in [-1, 1]
// at the stream's native sample rate (resample to 16kHz separately, see
// Resample). Multi-channel input is downmixed to mono by averaging, same as
// ReadWAV/DecodeMP3/DecodePCM. Inter-channel decorrelation (mid-side/
// left-side/right-side, FLAC's usual stereo compression trick) is handled
// internally by the mewkiz/flac library -- Subframes[c].Samples below is
// already each real channel's decoded value, not the mid/side encoding.
func DecodeFLAC(data []byte) (samples []float32, sampleRate int, err error) {
	stream, err := flac.Parse(bytes.NewReader(data))
	if err != nil {
		return nil, 0, fmt.Errorf("asr: parsing flac: %w", err)
	}
	defer stream.Close()

	numChannels := int(stream.Info.NChannels)
	bitsPerSample := int(stream.Info.BitsPerSample)
	maxAmplitude := float32(int64(1) << (bitsPerSample - 1))

	var out []float32
	for {
		frame, err := stream.ParseNext()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, 0, fmt.Errorf("asr: decoding flac frame: %w", err)
		}
		n := frame.Subframes[0].NSamples
		start := len(out)
		out = append(out, make([]float32, n)...)
		for c := 0; c < numChannels && c < len(frame.Subframes); c++ {
			sf := frame.Subframes[c]
			for i := 0; i < n && i < len(sf.Samples); i++ {
				out[start+i] += float32(sf.Samples[i]) / maxAmplitude / float32(numChannels)
			}
		}
	}
	return out, int(stream.Info.SampleRate), nil
}
