package asr

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/dlclark/regexp2"
)

// tokenizer is a byte-level BPE tokenizer matching the Qwen/GPT2-style tokenizer
// shipped by kitten-asr models (vocab.json + merges.txt + added_tokens.json +
// the pretokenizer regex embedded in tokenizer.json). Reimplemented directly
// rather than via an existing Go tokenizer library because the pretokenizer
// pattern uses a negative lookahead (`(?!\S)`), which Go's stdlib RE2-based
// regexp package cannot compile — dlclark/regexp2 is a backtracking engine
// that supports it.
type tokenizer struct {
	tokenToID map[string]int32
	idToToken []string // index by id

	// merges[a][b] = rank; lower rank merges first. Keyed by the two symbol
	// strings being merged (already byte-level-encoded).
	mergeRank map[mergePair]int

	special       map[string]int32 // literal special-token text -> id
	specialByID   map[int32]string
	specialSorted []string // special-token texts, longest first, for greedy matching

	splitRe *regexp2.Regexp

	byteToRune [256]rune
	runeToByte map[rune]byte
}

type mergePair struct{ a, b string }

// pretokenizer regex copied verbatim from kitten-asr-tiny/tokenizer.json's
// pre_tokenizer.pretokenizers[0].pattern.Regex (standard Qwen2/2.5 pattern).
const pretokenizePattern = `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`

func loadTokenizer(dir string) (*tokenizer, error) {
	t := &tokenizer{
		tokenToID:   map[string]int32{},
		mergeRank:   map[mergePair]int{},
		special:     map[string]int32{},
		specialByID: map[int32]string{},
	}
	t.byteToRune, t.runeToByte = buildByteLevelMapping()

	re, err := regexp2.Compile(pretokenizePattern, regexp2.None)
	if err != nil {
		return nil, fmt.Errorf("asr: compiling pretokenizer regex: %w", err)
	}
	t.splitRe = re

	if err := t.loadVocab(filepath.Join(dir, "vocab.json")); err != nil {
		return nil, err
	}
	if err := t.loadMerges(filepath.Join(dir, "merges.txt")); err != nil {
		return nil, err
	}
	if err := t.loadAddedTokens(filepath.Join(dir, "added_tokens.json")); err != nil {
		return nil, err
	}
	// Sort special tokens longest-first so greedy literal matching prefers
	// the longest match at each position (no special token is a prefix of
	// another in this vocab, but this is cheap insurance either way).
	for s := range t.special {
		t.specialSorted = append(t.specialSorted, s)
	}
	for i := 1; i < len(t.specialSorted); i++ {
		for j := i; j > 0 && len(t.specialSorted[j]) > len(t.specialSorted[j-1]); j-- {
			t.specialSorted[j], t.specialSorted[j-1] = t.specialSorted[j-1], t.specialSorted[j]
		}
	}
	return t, nil
}

func (t *tokenizer) loadVocab(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("asr: reading vocab.json: %w", err)
	}
	var vocab map[string]int32
	if err := json.Unmarshal(data, &vocab); err != nil {
		return fmt.Errorf("asr: parsing vocab.json: %w", err)
	}
	maxID := int32(-1)
	for _, id := range vocab {
		if id > maxID {
			maxID = id
		}
	}
	t.idToToken = make([]string, maxID+1)
	for tok, id := range vocab {
		t.tokenToID[tok] = id
		t.idToToken[id] = tok
	}
	return nil
}

func (t *tokenizer) loadMerges(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("asr: reading merges.txt: %w", err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	rank := 0
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" || strings.HasPrefix(line, "#version") {
			continue
		}
		parts := strings.SplitN(line, " ", 2)
		if len(parts) != 2 {
			continue
		}
		t.mergeRank[mergePair{parts[0], parts[1]}] = rank
		rank++
	}
	return scanner.Err()
}

func (t *tokenizer) loadAddedTokens(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("asr: reading added_tokens.json: %w", err)
	}
	var entries map[string]int32
	if err := json.Unmarshal(data, &entries); err != nil {
		return fmt.Errorf("asr: parsing added_tokens.json: %w", err)
	}
	for content, id := range entries {
		t.special[content] = id
		t.specialByID[id] = content
		if int(id) >= len(t.idToToken) {
			grown := make([]string, id+1)
			copy(grown, t.idToToken)
			t.idToToken = grown
		}
		if t.idToToken[id] == "" {
			t.idToToken[id] = content
		}
		if _, ok := t.tokenToID[content]; !ok {
			t.tokenToID[content] = id
		}
	}
	return nil
}

// buildByteLevelMapping reproduces GPT-2's bytes_to_unicode(): every byte
// value maps to a printable, distinct rune so arbitrary binary text can be
// BPE-merged as if it were a normal string. Bytes that are already "nice"
// printable Latin-1 characters map to themselves; the rest are shifted into
// the U+0100 range in encounter order.
func buildByteLevelMapping() ([256]rune, map[rune]byte) {
	var byteToRune [256]rune
	isNice := func(b int) bool {
		return (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
	}
	next := rune(256)
	for b := 0; b < 256; b++ {
		if isNice(b) {
			byteToRune[b] = rune(b)
		} else {
			byteToRune[b] = next
			next++
		}
	}
	runeToByte := make(map[rune]byte, 256)
	for b, r := range byteToRune {
		runeToByte[r] = byte(b)
	}
	return byteToRune, runeToByte
}

// byteLevelEncode maps a raw UTF-8 string's bytes through the byte-level
// table, returning the sequence of resulting runes as strings (one per
// input byte) ready for BPE merging.
func (t *tokenizer) byteLevelEncode(s string) []string {
	bs := []byte(s)
	out := make([]string, len(bs))
	for i, b := range bs {
		out[i] = string(t.byteToRune[b])
	}
	return out
}

// bpeMerge repeatedly merges the lowest-rank adjacent pair in symbols until
// no known merge applies, per the standard BPE algorithm.
func (t *tokenizer) bpeMerge(symbols []string) []string {
	if len(symbols) < 2 {
		return symbols
	}
	for {
		bestRank := -1
		bestIdx := -1
		for i := 0; i < len(symbols)-1; i++ {
			if r, ok := t.mergeRank[mergePair{symbols[i], symbols[i+1]}]; ok {
				if bestRank == -1 || r < bestRank {
					bestRank = r
					bestIdx = i
				}
			}
		}
		if bestIdx == -1 {
			return symbols
		}
		merged := symbols[bestIdx] + symbols[bestIdx+1]
		next := make([]string, 0, len(symbols)-1)
		next = append(next, symbols[:bestIdx]...)
		next = append(next, merged)
		next = append(next, symbols[bestIdx+2:]...)
		symbols = next
	}
}

// Encode tokenizes text into vocabulary ids, splitting out any literal
// special-token occurrences (e.g. "<|im_start|>") before regular BPE.
func (t *tokenizer) Encode(text string) ([]int32, error) {
	var ids []int32
	for len(text) > 0 {
		// Find the earliest occurrence of any special token.
		specIdx, specTok := -1, ""
		for _, s := range t.specialSorted {
			if i := strings.Index(text, s); i != -1 && (specIdx == -1 || i < specIdx) {
				specIdx, specTok = i, s
			}
		}
		var segment string
		if specIdx == -1 {
			segment, text = text, ""
		} else {
			segment, text = text[:specIdx], text[specIdx+len(specTok):]
		}
		if segment != "" {
			segIDs, err := t.encodeNoSpecial(segment)
			if err != nil {
				return nil, err
			}
			ids = append(ids, segIDs...)
		}
		if specTok != "" {
			ids = append(ids, t.special[specTok])
		}
	}
	return ids, nil
}

func (t *tokenizer) encodeNoSpecial(text string) ([]int32, error) {
	var ids []int32
	m, err := t.splitRe.FindStringMatch(text)
	if err != nil {
		return nil, fmt.Errorf("asr: pretokenizer match: %w", err)
	}
	for m != nil {
		word := m.String()
		symbols := t.byteLevelEncode(word)
		symbols = t.bpeMerge(symbols)
		for _, sym := range symbols {
			id, ok := t.tokenToID[sym]
			if !ok {
				return nil, fmt.Errorf("asr: BPE symbol %q not in vocab", sym)
			}
			ids = append(ids, id)
		}
		m, err = t.splitRe.FindNextMatch(m)
		if err != nil {
			return nil, fmt.Errorf("asr: pretokenizer match: %w", err)
		}
	}
	return ids, nil
}

// SpecialTokenID looks up a literal special-token string (e.g. "<asr_text>")
// by its exact text, as it appears in added_tokens.json.
func (t *tokenizer) SpecialTokenID(text string) (int32, bool) {
	id, ok := t.special[text]
	return id, ok
}

// Decode converts ids back to text. If skipSpecial is true, ids that are
// special tokens are dropped instead of decoded.
func (t *tokenizer) Decode(ids []int32, skipSpecial bool) string {
	var b strings.Builder
	var byteBuf []byte
	flush := func() {
		if len(byteBuf) > 0 {
			b.Write(byteBuf)
			byteBuf = byteBuf[:0]
		}
	}
	for _, id := range ids {
		if skipSpecial {
			if _, ok := t.specialByID[id]; ok {
				continue
			}
		}
		if int(id) < 0 || int(id) >= len(t.idToToken) || t.idToToken[id] == "" {
			continue
		}
		tok := t.idToToken[id]
		if _, ok := t.specialByID[id]; ok {
			// Special tokens are literal text, not byte-level encoded.
			flush()
			b.WriteString(tok)
			continue
		}
		for _, r := range tok {
			if bt, ok := t.runeToByte[r]; ok {
				byteBuf = append(byteBuf, bt)
			}
		}
	}
	flush()
	return b.String()
}
