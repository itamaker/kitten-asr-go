package asr

import (
	"encoding/json"
	"os"
	"testing"
)

type tokenizerCase struct {
	Text string  `json:"text"`
	IDs  []int32 `json:"ids"`
}

func loadTokenizerCases(t *testing.T) []tokenizerCase {
	t.Helper()
	data, err := os.ReadFile("testdata_tokenizer.json")
	if err != nil {
		t.Fatalf("reading testdata_tokenizer.json: %v", err)
	}
	var cases []tokenizerCase
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatalf("parsing testdata_tokenizer.json: %v", err)
	}
	return cases
}

func TestTokenizerEncodeMatchesPython(t *testing.T) {
	tok, err := loadTokenizer("testdata_tokenizer_files")
	if err != nil {
		t.Fatalf("loadTokenizer: %v", err)
	}
	for _, c := range loadTokenizerCases(t) {
		got, err := tok.Encode(c.Text)
		if err != nil {
			t.Errorf("Encode(%q): %v", c.Text, err)
			continue
		}
		if !equalInt32(got, c.IDs) {
			t.Errorf("Encode(%q):\n got  %v\n want %v", c.Text, got, c.IDs)
		}
	}
}

func TestTokenizerDecodeRoundTrip(t *testing.T) {
	tok, err := loadTokenizer("testdata_tokenizer_files")
	if err != nil {
		t.Fatalf("loadTokenizer: %v", err)
	}
	for _, c := range loadTokenizerCases(t) {
		got := tok.Decode(c.IDs, false)
		if got != c.Text {
			t.Errorf("Decode(%v):\n got  %q\n want %q", c.IDs, got, c.Text)
		}
	}
}

func equalInt32(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
