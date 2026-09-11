package asr

import (
	"math"
	"testing"
)

func TestMelFilterBankMatchesPython(t *testing.T) {
	want := readFloat32File(t, "testdata_mel_filters_reference.f32") // (201, 128) row-major
	filters := melFilterBank()                                       // [201][128]
	if len(filters) != 201 {
		t.Fatalf("got %d freq bins, want 201", len(filters))
	}
	var maxDiff float64
	var mi, mk int
	for i := 0; i < 201; i++ {
		for k := 0; k < numMelBins; k++ {
			w := float64(want[i*numMelBins+k])
			g := filters[i][k]
			d := math.Abs(w - g)
			if d > maxDiff {
				maxDiff, mi, mk = d, i, k
			}
		}
	}
	t.Logf("max abs diff: %.6g at freq_bin=%d mel=%d (got=%v want=%v)", maxDiff, mi, mk, filters[mi][mk], want[mi*numMelBins+mk])
	if maxDiff > 1e-4 {
		t.Errorf("mel filter bank diverges from Python: max abs diff %.6g", maxDiff)
	}
}
