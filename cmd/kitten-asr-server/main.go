// Command kitten-asr-server runs an OpenAI-compatible speech-to-text API
// server (the /v1/audio/transcriptions shape from the Whisper API, not the
// /v1/audio/speech shape kitten-tts-server implements -- these are different
// OpenAI endpoints for opposite directions).
//
// It loads a kitten-asr model at startup and serves that one endpoint (plus
// /v1/models and /health). Routing is github.com/gin-gonic/gin.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/itamaker/kitten-asr-go/asr"
)

// server holds the loaded model behind a mutex so a single set of ONNX
// sessions (and the KV cache tensors a transcription allocates along the
// way) can be shared safely across concurrent HTTP handlers -- same
// convention as kitten-tts-go's server.
type server struct {
	mu    sync.Mutex
	model *asr.Model
}

func main() {
	host := flag.String("host", "127.0.0.1", "server host address")
	port := flag.Int("port", 8080, "server port")
	threads := flag.Int("threads", asr.DefaultIntraOpThreads, "ONNX intra-op thread count (0 = onnxruntime's own default)")
	flag.Usage = usage
	flag.Parse()

	if flag.NArg() < 1 {
		usage()
		log.Fatal("model directory is required")
	}

	model, err := asr.New(flag.Arg(0), asr.WithIntraOpThreads(*threads))
	if err != nil {
		log.Fatalf("loading model: %v", err)
	}
	// log.Fatal below ListenAndServe would skip this defer entirely (it calls
	// os.Exit), which is why shutdown is handled explicitly rather than by
	// letting the process die on SIGINT/SIGTERM: the ONNX sessions should
	// release their native resources on the way out.
	defer model.Close()
	log.Printf("model loaded from %s", flag.Arg(0))

	srv := &server{model: model}

	gin.SetMode(gin.ReleaseMode)
	router := gin.New()
	router.Use(gin.Recovery())

	router.GET("/health", handleHealth)
	router.GET("/v1/models", handleListModels)
	router.POST("/v1/audio/transcriptions", srv.handleTranscription)
	router.GET("/v1/audio/transcriptions/stream", srv.handleTranscriptionStream)

	addr := fmt.Sprintf("%s:%d", *host, *port)
	httpSrv := &http.Server{
		Addr:              addr,
		Handler:           router,
		ReadHeaderTimeout: 10 * time.Second,
		// No ReadTimeout/WriteTimeout: audio uploads can be large and a
		// transcription of a near-MaxAudioSeconds clip can legitimately take
		// well over a minute on CPU -- see CLAUDE.md's performance notes.
		IdleTimeout: 120 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	serveErr := make(chan error, 1)
	go func() {
		log.Printf("listening on http://%s", addr)
		serveErr <- httpSrv.ListenAndServe()
	}()

	select {
	case err := <-serveErr:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("server error: %v", err)
		}
	case <-ctx.Done():
		stop() // restore default signal behavior so a second signal force-quits
		log.Print("shutting down (signal received)...")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := httpSrv.Shutdown(shutdownCtx); err != nil {
			log.Printf("graceful shutdown failed, forcing close: %v", err)
			httpSrv.Close()
		}
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `kitten-asr-server — OpenAI-compatible speech-to-text API server

Usage:
  kitten-asr-server [flags] <model_dir>

Flags:
`)
	flag.PrintDefaults()
}
