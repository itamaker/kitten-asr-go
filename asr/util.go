package asr

import (
	"encoding/binary"
	"math"
)

// bytesToFloat32 reinterprets a little-endian float32 byte buffer (as written
// by tools/export_embed_tokens.py, i.e. numpy's native tofile() on x86/ARM)
// as a []float32. len(b) must be a multiple of 4.
func bytesToFloat32(b []byte) []float32 {
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}
