package main

import (
	"context"
	"encoding/json"
	"errors"
	"log"
	"net/http"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
	"github.com/gin-gonic/gin"

	"github.com/itamaker/kitten-asr-go/asr"
)

// streamQuery configures a /v1/audio/transcriptions/stream connection via
// URL query parameters (a WebSocket upgrade request carries no body, so
// there's nowhere else to put them).
type streamQuery struct {
	// PCM format of every binary frame the client sends. Raw PCM has no
	// self-describing header (see asr.DecodePCM), hence these being
	// declared up front for the whole connection rather than per-message --
	// a real-time source's format doesn't change mid-stream anyway.
	SampleRate    int `form:"sample_rate"`
	BitsPerSample int `form:"bits_per_sample"`
	Channels      int `form:"channels"`

	// Optional overrides for asr.StreamOptions; zero means "use the
	// package's own default" (see asr.DefaultStreamWindowSeconds /
	// asr.DefaultStreamMinTriggerSeconds).
	WindowSeconds     float64 `form:"window_seconds"`
	MinTriggerSeconds float64 `form:"min_trigger_seconds"`
}

// wsClientMessage is a JSON text message the client may send instead of a
// binary audio frame.
type wsClientMessage struct {
	// "flush": run a transcription pass immediately on whatever's buffered,
	//   without waiting for MinTriggerSeconds of new audio (e.g. the user
	//   paused speaking and the client wants the partial result sooner).
	// "end": like flush, but the connection is closed afterward -- send
	//   this when the audio source itself has ended.
	Type string `json:"type"`
}

// wsServerMessage is a JSON text message this server sends.
type wsServerMessage struct {
	// "partial": an updated (possibly revised -- see asr.Stream's doc
	//   comment) transcript of the current window, from either a
	//   MinTriggerSeconds-triggered pass or a client "flush".
	// "final": the last transcript, sent once in response to a client
	//   "end", immediately before the server closes the connection.
	// "error": something went wrong with one message (bad PCM, a
	//   transcription error); the connection stays open unless the error
	//   was fatal to the whole session.
	Type     string `json:"type"`
	Language string `json:"language,omitempty"`
	Text     string `json:"text,omitempty"`
	Message  string `json:"message,omitempty"`
}

// handleTranscriptionStream upgrades to a WebSocket and runs a live
// asr.Stream session for its lifetime: binary frames are PCM audio (format
// fixed by the streamQuery params at connection time), text frames are
// wsClientMessage control messages. See asr.Stream's doc comment for what
// "streaming" does and doesn't mean here -- this is windowed re-transcription
// on a timer, not frame-by-frame causal decoding.
//
// Like the non-streaming handler, this holds s.mu for the duration of each
// model call (both Feed's occasional transcription pass and Flush), so
// concurrent connections queue behind each other rather than truly running
// in parallel -- expected for a single shared *asr.Model, same as
// handleTranscription.
func (s *server) handleTranscriptionStream(c *gin.Context) {
	q := streamQuery{SampleRate: 16000, BitsPerSample: 16, Channels: 1}
	if err := c.ShouldBindQuery(&q); err != nil {
		writeError(c, http.StatusBadRequest, "invalid query parameters: "+err.Error())
		return
	}

	s.mu.Lock()
	stream, err := s.model.NewStream(asr.StreamOptions{
		WindowSeconds:     q.WindowSeconds,
		MinTriggerSeconds: q.MinTriggerSeconds,
	})
	s.mu.Unlock()
	if err != nil {
		writeError(c, http.StatusBadRequest, err.Error())
		return
	}

	conn, err := websocket.Accept(c.Writer, c.Request, nil)
	if err != nil {
		return // Accept already wrote an HTTP error response
	}
	defer conn.CloseNow()

	ctx := c.Request.Context()
	log.Printf("stream: connected sample_rate=%d bits=%d channels=%d",
		q.SampleRate, q.BitsPerSample, q.Channels)

	for {
		typ, data, err := conn.Read(ctx)
		if err != nil {
			if !errors.Is(err, context.Canceled) && websocket.CloseStatus(err) != websocket.StatusNormalClosure {
				log.Printf("stream: read error: %v", err)
			}
			return
		}

		switch typ {
		case websocket.MessageBinary:
			samples, sr, err := asr.DecodePCM(data, q.SampleRate, q.BitsPerSample, q.Channels, false)
			if err != nil {
				sendWS(ctx, conn, wsServerMessage{Type: "error", Message: err.Error()})
				continue
			}
			s.mu.Lock()
			result, ok, err := stream.Feed(samples, sr)
			s.mu.Unlock()
			if err != nil {
				sendWS(ctx, conn, wsServerMessage{Type: "error", Message: err.Error()})
				continue
			}
			if ok {
				sendWS(ctx, conn, wsServerMessage{Type: "partial", Language: result.Language, Text: result.Text})
			}

		case websocket.MessageText:
			var msg wsClientMessage
			if err := json.Unmarshal(data, &msg); err != nil {
				sendWS(ctx, conn, wsServerMessage{Type: "error", Message: "invalid JSON control message: " + err.Error()})
				continue
			}
			switch msg.Type {
			case "flush":
				s.mu.Lock()
				result, err := stream.Flush()
				s.mu.Unlock()
				if err != nil {
					sendWS(ctx, conn, wsServerMessage{Type: "error", Message: err.Error()})
					continue
				}
				sendWS(ctx, conn, wsServerMessage{Type: "partial", Language: result.Language, Text: result.Text})
			case "end":
				s.mu.Lock()
				result, err := stream.Flush()
				s.mu.Unlock()
				if err != nil {
					sendWS(ctx, conn, wsServerMessage{Type: "error", Message: err.Error()})
				} else {
					sendWS(ctx, conn, wsServerMessage{Type: "final", Language: result.Language, Text: result.Text})
				}
				conn.Close(websocket.StatusNormalClosure, "")
				return
			default:
				sendWS(ctx, conn, wsServerMessage{Type: "error", Message: "unknown message type " + msg.Type})
			}
		}
	}
}

func sendWS(ctx context.Context, conn *websocket.Conn, msg wsServerMessage) {
	if err := wsjson.Write(ctx, conn, msg); err != nil {
		log.Printf("stream: write error: %v", err)
	}
}
