package asr

import "testing"

// These only exercise Stream's windowing/triggering bookkeeping, not actual
// transcription (that needs a real Model with loaded ONNX sessions -- see
// CLAUDE.md's "Go tests need no model/ONNX Runtime" note; a from-scratch
// &Model{} is fine here specifically because every case below stays under
// MinTriggerSeconds, so run() -- the only path that would touch the model's
// nil fields -- never gets called).

func TestNewStreamDefaults(t *testing.T) {
	s, err := (&Model{}).NewStream(StreamOptions{})
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	if want := int(DefaultStreamWindowSeconds * sampleRate); s.windowSamples != want {
		t.Errorf("windowSamples = %d, want %d", s.windowSamples, want)
	}
	if want := int(DefaultStreamMinTriggerSeconds * sampleRate); s.minTriggerSamples != want {
		t.Errorf("minTriggerSamples = %d, want %d", s.minTriggerSamples, want)
	}
}

func TestNewStreamValidation(t *testing.T) {
	cases := []struct {
		name string
		opts StreamOptions
	}{
		{"window too large", StreamOptions{WindowSeconds: MaxAudioSeconds + 1}},
		{"negative window", StreamOptions{WindowSeconds: -1}},
		{"negative trigger", StreamOptions{MinTriggerSeconds: -1}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := (&Model{}).NewStream(c.opts); err == nil {
				t.Error("expected an error, got nil")
			}
		})
	}
}

func TestStreamFeedBelowThreshold(t *testing.T) {
	s, err := (&Model{}).NewStream(StreamOptions{MinTriggerSeconds: 2.0})
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	// 1 second of silence at 16kHz -- well under the 2s trigger, so this
	// must not attempt to run the (nil-fielded) model.
	samples := make([]float32, sampleRate)
	result, ok, err := s.Feed(samples, sampleRate)
	if err != nil {
		t.Fatalf("Feed: %v", err)
	}
	if ok {
		t.Error("ok = true, want false (under MinTriggerSeconds)")
	}
	if result != (Result{}) {
		t.Errorf("result = %+v, want zero value", result)
	}
	if s.sinceLastRun != sampleRate {
		t.Errorf("sinceLastRun = %d, want %d", s.sinceLastRun, sampleRate)
	}
}

func TestStreamWindowSliding(t *testing.T) {
	s, err := (&Model{}).NewStream(StreamOptions{WindowSeconds: 2, MinTriggerSeconds: 1000})
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	// Feed 3 seconds in 1-second chunks; the buffer must never exceed the
	// 2-second window, and must retain the *most recent* samples.
	for i := 0; i < 3; i++ {
		chunk := make([]float32, sampleRate)
		for j := range chunk {
			chunk[j] = float32(i) // tag each second's samples with its index
		}
		if _, _, err := s.Feed(chunk, sampleRate); err != nil {
			t.Fatalf("Feed: %v", err)
		}
	}
	if len(s.buf) != s.windowSamples {
		t.Fatalf("buf len = %d, want %d (2s window)", len(s.buf), s.windowSamples)
	}
	// Should hold seconds 1 and 2 (0-indexed), not second 0, which slid out.
	if s.buf[0] != 1 || s.buf[len(s.buf)-1] != 2 {
		t.Errorf("buf = [%v ... %v], want [1 ... 2] (oldest second should have been dropped)", s.buf[0], s.buf[len(s.buf)-1])
	}
}

func TestStreamResetAndFlushEmpty(t *testing.T) {
	s, err := (&Model{}).NewStream(StreamOptions{MinTriggerSeconds: 1000})
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	if _, _, err := s.Feed(make([]float32, 100), sampleRate); err != nil {
		t.Fatalf("Feed: %v", err)
	}
	if len(s.buf) == 0 {
		t.Fatal("expected buffered samples before Reset")
	}
	s.Reset()
	if len(s.buf) != 0 || s.sinceLastRun != 0 {
		t.Errorf("after Reset: buf len=%d sinceLastRun=%d, want 0/0", len(s.buf), s.sinceLastRun)
	}
	// Flush on an empty buffer must not attempt to run the (nil-fielded)
	// model -- it should short-circuit and return the zero Result.
	result, err := s.Flush()
	if err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if result != (Result{}) {
		t.Errorf("Flush on empty stream = %+v, want zero value", result)
	}
}
