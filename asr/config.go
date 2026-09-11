package asr

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// config mirrors just the fields of kitten-asr's config.json this package
// needs -- the rest (activation functions, dropout, etc) is already baked
// into the exported ONNX graphs and isn't needed at inference time.
type config struct {
	AudioTokenID      int32 `json:"-"`
	AudioStartTokenID int32 `json:"-"`
	AudioEndTokenID   int32 `json:"-"`
	EOSTokenID        int32 `json:"-"`
	PadTokenID        int32 `json:"-"`

	NumMelBins  int `json:"-"`
	NWindow     int `json:"-"`
	HiddenSize  int `json:"-"`
	NumLayers   int `json:"-"`
	NumKVHeads  int `json:"-"`
	HeadDim     int `json:"-"`
	AudioOutDim int `json:"-"`
	VocabSize   int `json:"-"`
}

type rawConfig struct {
	EOSTokenID    int32 `json:"eos_token_id"`
	PadTokenID    int32 `json:"pad_token_id"`
	ThinkerConfig struct {
		AudioTokenID      int32 `json:"audio_token_id"`
		AudioStartTokenID int32 `json:"audio_start_token_id"`
		AudioEndTokenID   int32 `json:"audio_end_token_id"`
		AudioConfig       struct {
			NumMelBins int `json:"num_mel_bins"`
			NWindow    int `json:"n_window"`
			OutputDim  int `json:"output_dim"`
		} `json:"audio_config"`
		TextConfig struct {
			HiddenSize       int `json:"hidden_size"`
			NumHiddenLayers  int `json:"num_hidden_layers"`
			NumKeyValueHeads int `json:"num_key_value_heads"`
			HeadDim          int `json:"head_dim"`
			VocabSize        int `json:"vocab_size"`
		} `json:"text_config"`
	} `json:"thinker_config"`
}

func loadConfig(dir string) (*config, error) {
	data, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("asr: reading config.json: %w", err)
	}
	var raw rawConfig
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("asr: parsing config.json: %w", err)
	}
	return &config{
		AudioTokenID:      raw.ThinkerConfig.AudioTokenID,
		AudioStartTokenID: raw.ThinkerConfig.AudioStartTokenID,
		AudioEndTokenID:   raw.ThinkerConfig.AudioEndTokenID,
		EOSTokenID:        raw.EOSTokenID,
		PadTokenID:        raw.PadTokenID,
		NumMelBins:        raw.ThinkerConfig.AudioConfig.NumMelBins,
		NWindow:           raw.ThinkerConfig.AudioConfig.NWindow,
		AudioOutDim:       raw.ThinkerConfig.AudioConfig.OutputDim,
		HiddenSize:        raw.ThinkerConfig.TextConfig.HiddenSize,
		NumLayers:         raw.ThinkerConfig.TextConfig.NumHiddenLayers,
		NumKVHeads:        raw.ThinkerConfig.TextConfig.NumKeyValueHeads,
		HeadDim:           raw.ThinkerConfig.TextConfig.HeadDim,
		VocabSize:         raw.ThinkerConfig.TextConfig.VocabSize,
	}, nil
}

// floorDiv is Python's `//`: floors toward negative infinity, unlike Go's `/`
// which truncates toward zero (they differ whenever exactly one operand is
// negative -- which happens here whenever a `-1` numerator shows up below).
func floorDiv(a, b int) int {
	q := a / b
	if (a%b != 0) && ((a < 0) != (b < 0)) {
		q--
	}
	return q
}

// audioOutLen returns how many audio embedding vectors (and therefore
// <|audio_pad|> placeholder tokens) a full fixed-size (numMelFrame) window
// produces, per _get_feat_extract_output_lengths in the original model.
// Always the same constant for a given n_window (390 for n_window=50, true
// of both kitten-asr-tiny and kitten-asr-small-enhanced).
func (c *config) audioOutLen() int {
	chunkLen := c.NWindow * 2
	n := numMelFrame
	leave := n % chunkLen
	featLen := floorDiv(leave-1, 2) + 1
	return floorDiv(floorDiv(featLen-1, 2)+1-1, 2) + 1 + floorDiv(n, chunkLen)*13
}
