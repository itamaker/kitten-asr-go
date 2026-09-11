package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"

	"github.com/gin-gonic/gin"

	"github.com/itamaker/kitten-asr-go/asr"
)

// maxUploadSize caps the multipart request body; matches the OpenAI
// Whisper API's own 25MiB limit.
const maxUploadSize = 25 << 20

func writeError(c *gin.Context, status int, message string) {
	c.JSON(status, gin.H{
		"error": gin.H{
			"message": message,
			"type":    "invalid_request_error",
			"code":    nil,
		},
	})
}

func handleHealth(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"status": "ok"})
}

func handleListModels(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"object": "list",
		"data": []gin.H{
			{"id": "kitten-asr", "object": "model", "owned_by": "kittenml"},
		},
	})
}

func (s *server) handleTranscription(c *gin.Context) {
	ctx := c.Request.Context()
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxUploadSize)

	fileHeader, err := c.FormFile("file")
	if err != nil {
		writeError(c, http.StatusBadRequest, "'file' is required: "+err.Error())
		return
	}
	file, err := fileHeader.Open()
	if err != nil {
		writeError(c, http.StatusBadRequest, "opening upload: "+err.Error())
		return
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		writeError(c, http.StatusBadRequest, "reading upload: "+err.Error())
		return
	}

	responseFormat := c.PostForm("response_format")
	if responseFormat == "" {
		responseFormat = "json"
	}
	writeResult, err := responseWriterFor(responseFormat)
	if err != nil {
		writeError(c, http.StatusBadRequest, err.Error())
		return
	}

	// language/prompt/temperature are accepted (for OpenAI client
	// compatibility) but have no effect: the model always auto-detects
	// language itself, its chat template drops any extra prompt text (see
	// CLAUDE.md), and generation is always greedy.

	samples, sampleRate, err := asr.DecodeAudio(data, fileHeader.Filename)
	if err != nil {
		writeError(c, http.StatusBadRequest, err.Error())
		return
	}
	duration := float64(len(samples)) / float64(sampleRate)

	log.Printf("transcribe: filename=%s bytes=%d duration=%.1fs format=%s",
		fileHeader.Filename, len(data), duration, responseFormat)

	s.mu.Lock()
	result, err := transcribeContext(ctx, s.model, samples, sampleRate)
	s.mu.Unlock()
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return // client is gone; nothing left to write to
		}
		writeError(c, http.StatusInternalServerError, err.Error())
		return
	}

	writeResult(c, result, duration)
}

// transcribeContext runs Transcribe but returns promptly if ctx is canceled
// (a disconnected client stops occupying the shared model) -- Transcribe
// itself has no cancellation points of its own since the ONNX calls it makes
// are not context-aware, so this can only observe cancellation between
// starting the call and it returning, not interrupt it mid-flight; it's still
// useful because it stops the *server* from spending time writing a response
// nobody will read.
func transcribeContext(ctx context.Context, model *asr.Model, samples []float32, sampleRate int) (asr.Result, error) {
	type out struct {
		res asr.Result
		err error
	}
	done := make(chan out, 1)
	go func() {
		res, err := model.Transcribe(samples, sampleRate)
		done <- out{res, err}
	}()
	select {
	case <-ctx.Done():
		return asr.Result{}, ctx.Err()
	case o := <-done:
		return o.res, o.err
	}
}

// responseWriterFor returns a function that renders a Result in the
// requested response_format, or an error if the format needs data (e.g.
// per-segment timestamps for srt/vtt) this server doesn't have.
func responseWriterFor(format string) (func(c *gin.Context, result asr.Result, duration float64), error) {
	switch format {
	case "json":
		return func(c *gin.Context, result asr.Result, _ float64) {
			c.JSON(http.StatusOK, gin.H{"text": result.Text})
		}, nil
	case "text":
		return func(c *gin.Context, result asr.Result, _ float64) {
			c.String(http.StatusOK, "%s", result.Text)
		}, nil
	case "verbose_json":
		return func(c *gin.Context, result asr.Result, duration float64) {
			c.JSON(http.StatusOK, gin.H{
				"task":     "transcribe",
				"language": result.Language,
				"duration": duration,
				"text":     result.Text,
				// No "segments"/"words": kitten-asr produces one transcript
				// per (chunked) call with no per-word or per-segment timing,
				// unlike Whisper's decoder. Omitted rather than fabricated.
			})
		}, nil
	case "srt", "vtt":
		return nil, fmt.Errorf("response_format %q needs per-segment timestamps kitten-asr doesn't produce; use \"json\" or \"text\"", format)
	default:
		return nil, fmt.Errorf("unknown response_format %q", format)
	}
}
