package main

import "strings"

// normalizeWords lowercases s, drops everything but letters/digits/apostrophe/
// space, and splits on whitespace -- crude but consistent, and applied
// identically to both reference and hypothesis so it cancels out of the
// comparison.
func normalizeWords(s string) []string {
	var b strings.Builder
	for _, r := range strings.ToLower(s) {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9', r == '\'':
			b.WriteRune(r)
		default:
			b.WriteRune(' ')
		}
	}
	return strings.Fields(b.String())
}

// wordErrorRate returns the edit distance between ref and hyp word sequences
// divided by len(ref) (the standard WER definition), plus the raw edit count
// and reference length so callers can aggregate across multiple items
// correctly (sum edits / sum ref-len, not average-of-ratios).
func wordErrorRate(ref, hyp string) (rate float64, edits, refLen int) {
	r := normalizeWords(ref)
	h := normalizeWords(hyp)
	edits = levenshtein(r, h)
	refLen = len(r)
	if refLen == 0 {
		return 0, edits, 0
	}
	return float64(edits) / float64(refLen), edits, refLen
}

func levenshtein(a, b []string) int {
	prev := make([]int, len(b)+1)
	curr := make([]int, len(b)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(a); i++ {
		curr[0] = i
		for j := 1; j <= len(b); j++ {
			cost := 1
			if a[i-1] == b[j-1] {
				cost = 0
			}
			del := prev[j] + 1
			ins := curr[j-1] + 1
			sub := prev[j-1] + cost
			m := del
			if ins < m {
				m = ins
			}
			if sub < m {
				m = sub
			}
			curr[j] = m
		}
		prev, curr = curr, prev
	}
	return prev[len(b)]
}
