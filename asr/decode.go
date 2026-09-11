package asr

import (
	"fmt"
	"strings"
)

// DecodeAudio decodes a self-describing audio file's raw bytes to mono
// float32 PCM, dispatching on filename (matched by extension,
// case-insensitive). Supports ".wav", ".mp3", and ".flac"; other extensions
// return an error naming what's missing rather than guessing. Headerless raw
// PCM (e.g. from a real-time capture client) has no filename/extension to
// dispatch on and isn't handled here -- call DecodePCM directly with its
// format parameters instead.
func DecodeAudio(data []byte, filename string) (samples []float32, sampleRate int, err error) {
	ext := strings.ToLower(filename)
	if i := strings.LastIndexByte(ext, '.'); i != -1 {
		ext = ext[i:]
	}
	switch ext {
	case ".wav":
		return decodeWAV(data)
	case ".mp3":
		return DecodeMP3(data)
	case ".flac":
		return DecodeFLAC(data)
	default:
		return nil, 0, fmt.Errorf("asr: unsupported audio format %q (supported: .wav, .mp3, .flac)", ext)
	}
}
