package asr

import "fmt"

// DefaultStreamWindowSeconds and DefaultStreamMinTriggerSeconds are the
// defaults NewStream uses when the corresponding StreamOptions field is
// zero. See StreamOptions for what they trade off.
const (
	DefaultStreamWindowSeconds     = 20.0
	DefaultStreamMinTriggerSeconds = 1.5
)

// StreamOptions configures a Stream's windowing/triggering behavior.
type StreamOptions struct {
	// WindowSeconds caps how much recent audio a re-transcription pass
	// considers at once. Once the buffered audio exceeds this, the oldest
	// samples are dropped (a sliding window, not an ever-growing buffer) --
	// must be in (0, MaxAudioSeconds]. Zero uses DefaultStreamWindowSeconds.
	WindowSeconds float64
	// MinTriggerSeconds is how much *new* audio must accumulate since the
	// last transcription pass before Feed runs another one. Larger values
	// mean fewer, cheaper inference calls but coarser-grained updates;
	// smaller values mean lower latency but more repeated work re-decoding
	// the same window. Zero uses DefaultStreamMinTriggerSeconds.
	MinTriggerSeconds float64
}

// Stream is a stateful incremental transcription session: feed it audio
// chunks as they arrive (e.g. from a live microphone or a WebSocket relay),
// and it periodically re-transcribes a sliding window of the most recent
// audio, delivering an updated Result each time enough new audio has come
// in.
//
// This is deliberately NOT frame-by-frame causal streaming -- the
// underlying model has no incremental/causal inference mode at all (its
// audio encoder uses non-causal windowed attention meant for chunking long
// *pre-recorded* audio, not live streaming, and text generation requires
// the complete current-window audio encoding before the first output token;
// see CLAUDE.md for the full explanation of why). Each Feed call that
// triggers a pass re-runs the model over the whole current window from
// scratch, so:
//   - Results can be *revised* between calls: a word transcribed one way
//     near the end of a window may come out differently once more audio
//     gives the model more context, or once it slides out of the window
//     entirely mid-word.
//   - Cost grows with WindowSeconds, not with how much is "new" -- this is
//     the same tradeoff most "streaming Whisper" wrappers make (re-decode a
//     recent window on a timer), not true incremental ASR.
//
// A *Stream is not goroutine-safe (same convention as Model) and holds no
// reference to a lower-level resource of its own to close -- it borrows the
// *Model that created it, which the caller remains responsible for
// eventually Close()ing once every Stream using it is done.
type Stream struct {
	model             *Model
	windowSamples     int
	minTriggerSamples int
	buf               []float32 // accumulated 16kHz samples, capped to windowSamples
	sinceLastRun      int       // new samples appended since the last transcription pass
}

// NewStream starts a new streaming session against m.
func (m *Model) NewStream(opts StreamOptions) (*Stream, error) {
	windowSeconds := opts.WindowSeconds
	if windowSeconds == 0 {
		windowSeconds = DefaultStreamWindowSeconds
	}
	if windowSeconds <= 0 || windowSeconds > MaxAudioSeconds {
		return nil, fmt.Errorf("asr: WindowSeconds must be in (0, %d], got %v", MaxAudioSeconds, windowSeconds)
	}
	minTrigger := opts.MinTriggerSeconds
	if minTrigger == 0 {
		minTrigger = DefaultStreamMinTriggerSeconds
	}
	if minTrigger <= 0 {
		return nil, fmt.Errorf("asr: MinTriggerSeconds must be > 0, got %v", minTrigger)
	}
	return &Stream{
		model:             m,
		windowSamples:     int(windowSeconds * float64(sampleRate)),
		minTriggerSamples: int(minTrigger * float64(sampleRate)),
	}, nil
}

// Feed appends samples (mono, at inputSampleRate Hz) to the stream's rolling
// window. If at least MinTriggerSeconds of new audio has accumulated since
// the last transcription pass, it re-transcribes the current window and
// returns the updated Result with ok=true; otherwise it returns the zero
// Result with ok=false without running the model (so callers can Feed small,
// frequent chunks -- e.g. every 100-250ms off a WebSocket -- without paying
// for an inference call on every single one).
func (s *Stream) Feed(samples []float32, inputSampleRate int) (result Result, ok bool, err error) {
	if inputSampleRate != sampleRate {
		samples = Resample(samples, inputSampleRate, sampleRate)
	}
	s.buf = append(s.buf, samples...)
	s.sinceLastRun += len(samples)

	if len(s.buf) > s.windowSamples {
		drop := len(s.buf) - s.windowSamples
		copy(s.buf, s.buf[drop:])
		s.buf = s.buf[:s.windowSamples]
	}
	if s.sinceLastRun < s.minTriggerSamples {
		return Result{}, false, nil
	}
	return s.run()
}

// Flush forces a transcription pass over whatever audio is currently
// buffered, regardless of MinTriggerSeconds -- call this when the source
// signals end-of-stream (e.g. a WebSocket close or an explicit "done"
// message), so the last stretch of audio isn't lost waiting for the next
// trigger threshold. Returns the zero Result if nothing is buffered.
func (s *Stream) Flush() (Result, error) {
	if len(s.buf) == 0 {
		return Result{}, nil
	}
	result, _, err := s.run()
	return result, err
}

// Reset discards all buffered audio, starting a fresh window -- e.g. call
// this after detecting a long silence/pause, so the next utterance isn't
// diluted by stale audio still sitting in the window.
func (s *Stream) Reset() {
	s.buf = s.buf[:0]
	s.sinceLastRun = 0
}

func (s *Stream) run() (Result, bool, error) {
	result, err := s.model.transcribeChunk(s.buf)
	if err != nil {
		return Result{}, false, err
	}
	s.sinceLastRun = 0
	return result, true, nil
}
