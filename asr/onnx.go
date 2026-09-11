package asr

// This file isolates the ONNX Runtime dependency (dlopen'd at runtime, no cgo
// link), mirroring kitten-tts-go's tts/onnx.go. Two graphs are involved:
//   - audio_encoder.onnx: single fixed-shape forward pass (1, numMelBins,
//     numMelFrame) -> (audioOutLen, hidden) embeddings.
//   - decoder.onnx: one autoregressive step at a time, with the KV cache
//     threaded through explicitly as 2*numLayers input/output tensor pairs
//     (see kitten-asr-go/tools/decoder_wrapper.py for the Python side this
//     was exported from) -- no ORT KV-cache/IOBinding magic, just plain
//     tensors the caller re-feeds each step.

import (
	"fmt"
	"os"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

var (
	runtimeOnce sync.Once
	runtimeErr  error
)

func initRuntime() error {
	runtimeOnce.Do(func() {
		if p := os.Getenv("ONNXRUNTIME_LIB_PATH"); p != "" {
			ort.SetSharedLibraryPath(p)
		} else if p := findRuntimeLib(); p != "" {
			ort.SetSharedLibraryPath(p)
		}
		runtimeErr = ort.InitializeEnvironment()
	})
	return runtimeErr
}

func findRuntimeLib() string {
	for _, p := range []string{
		"/opt/homebrew/lib/libonnxruntime.dylib",
		"/usr/local/lib/libonnxruntime.dylib",
		"/usr/lib/libonnxruntime.so",
		"/usr/local/lib/libonnxruntime.so",
		"/usr/lib/x86_64-linux-gnu/libonnxruntime.so",
		"/usr/lib/aarch64-linux-gnu/libonnxruntime.so",
	} {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}

func newSessionOptions(intraOpThreads int) (*ort.SessionOptions, error) {
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	if err := opts.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll); err != nil {
		opts.Destroy()
		return nil, err
	}
	// Both graphs are one long sequential chain of layers (no independent
	// parallel subgraphs), so inter-op parallelism has nothing to do -- but
	// the per-layer matmuls benefit substantially from intra-op threading (see
	// DefaultIntraOpThreads for why more isn't reliably better).
	if err := opts.SetIntraOpNumThreads(intraOpThreads); err != nil {
		opts.Destroy()
		return nil, err
	}
	if err := opts.SetInterOpNumThreads(1); err != nil {
		opts.Destroy()
		return nil, err
	}
	if err := opts.SetCpuMemArena(true); err != nil {
		opts.Destroy()
		return nil, err
	}
	if err := opts.SetMemPattern(true); err != nil {
		opts.Destroy()
		return nil, err
	}
	return opts, nil
}

// --- Audio encoder -----------------------------------------------------

type audioEncoder struct {
	session *ort.DynamicAdvancedSession
}

func loadAudioEncoder(path string, intraOpThreads int) (*audioEncoder, error) {
	if err := initRuntime(); err != nil {
		return nil, fmt.Errorf("asr: onnx runtime unavailable: %w", err)
	}
	opts, err := newSessionOptions(intraOpThreads)
	if err != nil {
		return nil, fmt.Errorf("asr: creating session options: %w", err)
	}
	defer opts.Destroy()

	session, err := ort.NewDynamicAdvancedSession(path,
		[]string{"input_features"}, []string{"audio_embeds"}, opts)
	if err != nil {
		return nil, fmt.Errorf("asr: loading audio encoder: %w", err)
	}
	return &audioEncoder{session: session}, nil
}

// Run takes flattened (numMelBins*numMelFrame) mel features ([mel][frame]
// row-major, see featureExtractor.LogMelSpectrogram) and returns the flattened
// (audioOutLen*hidden) embeddings.
func (e *audioEncoder) Run(mel []float32, hidden int) ([]float32, error) {
	inT, err := ort.NewTensor(ort.NewShape(1, int64(numMelBins), int64(numMelFrame)), mel)
	if err != nil {
		return nil, err
	}
	defer inT.Destroy()

	outputs := []ort.Value{nil}
	if err := e.session.Run([]ort.Value{inT}, outputs); err != nil {
		return nil, fmt.Errorf("asr: running audio encoder: %w", err)
	}
	defer outputs[0].Destroy()

	outT, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("asr: unexpected audio encoder output type %T", outputs[0])
	}
	data := outT.GetData()
	out := make([]float32, len(data))
	copy(out, data)
	return out, nil
}

func (e *audioEncoder) Close() error {
	if e.session != nil {
		return e.session.Destroy()
	}
	return nil
}

// --- Text decoder (explicit KV cache) -----------------------------------

type textDecoder struct {
	session    *ort.DynamicAdvancedSession
	numLayers  int
	numKVHeads int
	headDim    int
	inputNames []string
	outNames   []string
}

func loadTextDecoder(path string, numLayers, numKVHeads, headDim, intraOpThreads int) (*textDecoder, error) {
	if err := initRuntime(); err != nil {
		return nil, fmt.Errorf("asr: onnx runtime unavailable: %w", err)
	}
	opts, err := newSessionOptions(intraOpThreads)
	if err != nil {
		return nil, fmt.Errorf("asr: creating session options: %w", err)
	}
	defer opts.Destroy()

	inNames := make([]string, 0, 2+2*numLayers)
	outNames := make([]string, 0, 1+2*numLayers)
	inNames = append(inNames, "inputs_embeds", "attention_mask")
	outNames = append(outNames, "logits")
	for i := 0; i < numLayers; i++ {
		inNames = append(inNames, fmt.Sprintf("past_key_%d", i), fmt.Sprintf("past_value_%d", i))
		outNames = append(outNames, fmt.Sprintf("present_key_%d", i), fmt.Sprintf("present_value_%d", i))
	}

	session, err := ort.NewDynamicAdvancedSession(path, inNames, outNames, opts)
	if err != nil {
		return nil, fmt.Errorf("asr: loading text decoder: %w", err)
	}
	return &textDecoder{
		session: session, numLayers: numLayers, numKVHeads: numKVHeads, headDim: headDim,
		inputNames: inNames, outNames: outNames,
	}, nil
}

// kvCache holds one (key, value) ORT tensor pair per decoder layer. The zero
// value (via newKVCache) represents an empty cache ready for a first/prefill
// call.
type kvCache struct {
	keys, values []*ort.Tensor[float32]
}

func newKVCache(numLayers, numKVHeads, headDim int) (*kvCache, error) {
	c := &kvCache{keys: make([]*ort.Tensor[float32], numLayers), values: make([]*ort.Tensor[float32], numLayers)}
	shape := ort.NewShape(1, int64(numKVHeads), 0, int64(headDim))
	for i := 0; i < numLayers; i++ {
		k, err := ort.NewEmptyTensor[float32](shape)
		if err != nil {
			c.Destroy()
			return nil, err
		}
		c.keys[i] = k
		v, err := ort.NewEmptyTensor[float32](shape)
		if err != nil {
			c.Destroy()
			return nil, err
		}
		c.values[i] = v
	}
	return c, nil
}

// Len returns the current cached sequence length (0 for a fresh cache).
func (c *kvCache) Len() int64 {
	if len(c.keys) == 0 {
		return 0
	}
	return c.keys[0].GetShape()[2]
}

func (c *kvCache) Destroy() {
	for _, t := range c.keys {
		if t != nil {
			t.Destroy()
		}
	}
	for _, t := range c.values {
		if t != nil {
			t.Destroy()
		}
	}
}

// Step runs one decoder forward pass: inputsEmbeds is (seqLen, hidden)
// row-major for a single batch item. Attention covers cache.Len()+seqLen
// positions (no padding is ever used, generation is always a single
// unpadded sequence). The exported graph only ever projects the *last*
// position through lm_head (multi-position prefill logits are never used by
// a greedy decode loop, so computing them would be pure waste), so the
// returned logits is always exactly one vocabSize-length row regardless of
// seqLen. Replaces cache's tensors in place with the updated (grown) ones,
// destroying the old tensors.
func (d *textDecoder) Step(cache *kvCache, inputsEmbeds []float32, seqLen, hidden int) ([]float32, error) {
	embedsT, err := ort.NewTensor(ort.NewShape(1, int64(seqLen), int64(hidden)), inputsEmbeds)
	if err != nil {
		return nil, err
	}
	defer embedsT.Destroy()

	total := cache.Len() + int64(seqLen)
	maskData := make([]int64, total)
	for i := range maskData {
		maskData[i] = 1
	}
	maskT, err := ort.NewTensor(ort.NewShape(1, total), maskData)
	if err != nil {
		return nil, err
	}
	defer maskT.Destroy()

	inputs := make([]ort.Value, 2+2*d.numLayers)
	inputs[0], inputs[1] = embedsT, maskT
	for i := 0; i < d.numLayers; i++ {
		inputs[2+2*i] = cache.keys[i]
		inputs[2+2*i+1] = cache.values[i]
	}

	outputs := make([]ort.Value, 1+2*d.numLayers)
	if err := d.session.Run(inputs, outputs); err != nil {
		return nil, fmt.Errorf("asr: running decoder step: %w", err)
	}

	logitsT, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		for _, o := range outputs {
			if o != nil {
				o.Destroy()
			}
		}
		return nil, fmt.Errorf("asr: unexpected decoder logits type %T", outputs[0])
	}
	logitsData := logitsT.GetData()
	logits := make([]float32, len(logitsData))
	copy(logits, logitsData)
	logitsT.Destroy()

	for i := 0; i < d.numLayers; i++ {
		newKey, ok := outputs[1+2*i].(*ort.Tensor[float32])
		if !ok {
			return nil, fmt.Errorf("asr: unexpected present_key_%d type %T", i, outputs[1+2*i])
		}
		newVal, ok := outputs[1+2*i+1].(*ort.Tensor[float32])
		if !ok {
			return nil, fmt.Errorf("asr: unexpected present_value_%d type %T", i, outputs[1+2*i+1])
		}
		cache.keys[i].Destroy()
		cache.values[i].Destroy()
		cache.keys[i] = newKey
		cache.values[i] = newVal
	}
	return logits, nil
}

func (d *textDecoder) Close() error {
	if d.session != nil {
		return d.session.Destroy()
	}
	return nil
}
