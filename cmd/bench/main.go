// Command bench is a dev tool (not part of the public API/CLI) for measuring
// word error rate and latency against synthetic espeak-ng audio -- known
// text in, so WER is computable, without depending on a real speech corpus.
// It backs the measured rationale behind DefaultStreamWindowSeconds/
// DefaultStreamMinTriggerSeconds (asr/stream.go) and the quantization
// tradeoffs documented in the README, and exists to make those numbers
// reproducible against a future model/audio/tuning change, not just this
// one's. Requires espeak-ng on PATH.
//
// Usage:
//
//	bench -model <dir> -mode wer                                  # WER + latency across a short corpus
//	bench -model <dir> -mode sweep -windows 10,20,30 -triggers 1,2 # Stream window/trigger grid
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/itamaker/kitten-asr-go/asr"
)

func main() {
	modelDir := flag.String("model", "", "model directory")
	mode := flag.String("mode", "wer", "wer | sweep")
	speed := flag.Int("speed", 150, "espeak-ng -s (words per minute)")
	cacheDir := flag.String("cache", "/tmp/claude-0/-mnt-d-code-project/87259767-a5c8-42a7-90a7-74b438b6acb0/scratchpad/eval_audio", "synthesized-audio cache dir")
	threads := flag.Int("threads", asr.DefaultIntraOpThreads, "ONNX intra-op threads")
	windows := flag.String("windows", "5,10,15,20,25,30", "comma-separated WindowSeconds grid (sweep mode)")
	triggers := flag.String("triggers", "0.5,1,1.5,2.5,4", "comma-separated MinTriggerSeconds grid (sweep mode)")
	feedChunkMs := flag.Int("feedchunkms", 200, "simulated Feed() chunk size in ms (sweep mode)")
	flag.Parse()

	if *modelDir == "" {
		log.Fatal("-model is required")
	}
	model, err := asr.New(*modelDir, asr.WithIntraOpThreads(*threads))
	if err != nil {
		log.Fatalf("loading model: %v", err)
	}
	defer model.Close()

	switch *mode {
	case "wer":
		runWER(model, *cacheDir, *speed)
	case "sweep":
		runSweep(model, *cacheDir, *speed, sweepParagraph, parseFloats(*windows), parseFloats(*triggers), *feedChunkMs)
	default:
		log.Fatalf("unknown -mode %q", *mode)
	}
}

func parseFloats(s string) []float64 {
	var out []float64
	for _, p := range strings.Split(s, ",") {
		v, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
		if err != nil {
			log.Fatalf("bad float %q: %v", p, err)
		}
		out = append(out, v)
	}
	return out
}

func runWER(model *asr.Model, cacheDir string, speed int) {
	items := append(append([]evalItem{}, shortCorpus...), evalItem{"long-paragraph", longParagraph})

	var totalEdits, totalRefLen int
	var totalWall time.Duration
	fmt.Printf("%-16s %8s %10s %s\n", "item", "wer%", "latency", "hyp")
	for _, it := range items {
		wavPath, err := synthesize(cacheDir, it.text, speed)
		if err != nil {
			log.Fatalf("synthesize %s: %v", it.name, err)
		}
		data, err := os.ReadFile(wavPath)
		if err != nil {
			log.Fatal(err)
		}
		samples, sr, err := asr.DecodeAudio(data, wavPath)
		if err != nil {
			log.Fatalf("decode %s: %v", it.name, err)
		}

		start := time.Now()
		res, err := model.Transcribe(samples, sr)
		elapsed := time.Since(start)
		if err != nil {
			log.Fatalf("transcribe %s: %v", it.name, err)
		}

		rate, edits, refLen := wordErrorRate(it.text, res.Text)
		totalEdits += edits
		totalRefLen += refLen
		totalWall += elapsed
		hyp := res.Text
		if len(hyp) > 60 {
			hyp = hyp[:60] + "..."
		}
		fmt.Printf("%-16s %7.1f%% %10s %s\n", it.name, rate*100, elapsed.Round(time.Millisecond), hyp)
	}
	fmt.Printf("\naggregate WER: %.1f%% (%d edits / %d ref words), total wall: %s, avg: %s\n",
		100*float64(totalEdits)/float64(totalRefLen), totalEdits, totalRefLen,
		totalWall.Round(time.Millisecond), (totalWall / time.Duration(len(items))).Round(time.Millisecond))
}

func runSweep(model *asr.Model, cacheDir string, speed int, text string, windows, triggers []float64, feedChunkMs int) {
	wavPath, err := synthesize(cacheDir, text, speed)
	if err != nil {
		log.Fatal(err)
	}
	data, err := os.ReadFile(wavPath)
	if err != nil {
		log.Fatal(err)
	}
	samples, sr, err := asr.DecodeAudio(data, wavPath)
	if err != nil {
		log.Fatal(err)
	}
	chunkSamples := sr * feedChunkMs / 1000

	fmt.Printf("%-8s %-8s %7s %10s %8s\n", "window", "trigger", "passes", "wall", "wer%")
	for _, w := range windows {
		for _, tr := range triggers {
			stream, err := model.NewStream(asr.StreamOptions{WindowSeconds: w, MinTriggerSeconds: tr})
			if err != nil {
				fmt.Printf("%-8.1f %-8.1f  (skip: %v)\n", w, tr, err)
				continue
			}
			passes := 0
			start := time.Now()
			var last asr.Result
			for pos := 0; pos < len(samples); pos += chunkSamples {
				end := pos + chunkSamples
				if end > len(samples) {
					end = len(samples)
				}
				res, ok, err := stream.Feed(samples[pos:end], sr)
				if err != nil {
					log.Fatalf("feed: %v", err)
				}
				if ok {
					passes++
					last = res
				}
			}
			res, err := stream.Flush()
			if err != nil {
				log.Fatalf("flush: %v", err)
			}
			if strings.TrimSpace(res.Text) != "" {
				passes++
				last = res
			}
			elapsed := time.Since(start)

			rate, _, _ := wordErrorRate(text, last.Text)
			fmt.Printf("%-8.1f %-8.1f %7d %10s %7.1f%%\n", w, tr, passes, elapsed.Round(time.Millisecond), rate*100)
		}
	}
}
