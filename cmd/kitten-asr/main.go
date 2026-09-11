// Command kitten-asr transcribes an audio file (.wav or .mp3) using a
// kitten-asr model directory. Following Go convention, flags come before
// positional arguments: kitten-asr [flags] <model_dir> <audio_file>
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/itamaker/kitten-asr-go/asr"
)

func main() {
	showLanguage := flag.Bool("language", false, "also print the model's detected language")
	threads := flag.Int("threads", asr.DefaultIntraOpThreads, "ONNX intra-op thread count (0 = onnxruntime's own default)")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [flags] <model_dir> <audio_file>\n", os.Args[0])
		flag.PrintDefaults()
	}
	flag.Parse()

	args := flag.Args()
	if len(args) != 2 {
		flag.Usage()
		os.Exit(2)
	}
	modelDir, audioPath := args[0], args[1]

	if err := run(modelDir, audioPath, *showLanguage, *threads); err != nil {
		fmt.Fprintln(os.Stderr, "kitten-asr:", err)
		os.Exit(1)
	}
}

func run(modelDir, audioPath string, showLanguage bool, threads int) error {
	data, err := os.ReadFile(audioPath)
	if err != nil {
		return err
	}
	samples, sampleRate, err := asr.DecodeAudio(data, audioPath)
	if err != nil {
		return err
	}

	model, err := asr.New(modelDir, asr.WithIntraOpThreads(threads))
	if err != nil {
		return err
	}
	defer model.Close()

	result, err := model.Transcribe(samples, sampleRate)
	if err != nil {
		return err
	}

	if showLanguage && result.Language != "" {
		fmt.Fprintln(os.Stderr, "language:", result.Language)
	}
	fmt.Println(result.Text)
	return nil
}
