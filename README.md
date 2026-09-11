# kitten-asr-go 🐱🎙️

<p>
  <img src="https://kittenml.com/assets/kittenml_logo.svg" alt="KittenML" height="28">
  <img src="https://img.shields.io/badge/Go-1.25-00ADD8?logo=go&logoColor=white" alt="Go 1.25">
</p>

Go implementation of [KittenML](https://kittenml.com)'s ASR models
(`kitten-asr-tiny`, `kitten-asr-small-enhanced`) — lightweight, CPU-only speech
recognition. Sibling project to [kitten-tts-go](https://github.com/itamaker/kitten-tts-go).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/itamaker/kitten-asr-go/blob/main/examples/kitten-asr-go-colab.ipynb)

> **Try it now:** the [Colab notebook](examples/kitten-asr-go-colab.ipynb) builds
> the project and transcribes audio in a few clicks — no local setup, no GPU.

> **Status:** both models' engines work end-to-end (CLI, HTTP server, live
> WebSocket streaming, long-audio chunking), models are hosted, and
> `fetch_model.sh` has been verified end-to-end against both — all against
> real audio.

This repository provides a pure-Go inference engine for the models (CLI, HTTP
server, and a library) plus the Python pipeline (`tools/`) used to produce the
ONNX files that engine runs — the models ship as PyTorch weights, so ONNX
export happens once, ahead of time, rather than at runtime.

## Key features

- **CPU-optimized** — ONNX-based inference runs without a GPU
- **Automatic language detection** — the model reports a guessed language
  alongside the transcript
- **WAV, MP3, FLAC, and headerless raw PCM input** — other formats are a clear
  error, not silently wrong output
- **Long audio** — inputs over the model's native 30-second window are
  transcribed in consecutive chunks automatically
- **No native library linked at build time** — ONNX Runtime is `dlopen`'d at
  runtime, not linked; only libc/libdl gets pulled in (see
  [Dependencies](#dependencies) — a C compiler is still needed to build,
  since the dlopen shim itself is cgo, `CGO_ENABLED=0` will fail)
- **OpenAI-compatible HTTP server** — `POST /v1/audio/transcriptions`, the same
  shape as Whisper's API
- **Live transcription over WebSocket** — for real-time sources like a
  browser mic (see [below](#live--streaming-transcription) for what
  "streaming" does and doesn't mean for this model)

## Dependencies

### A C compiler (build-time only)

`yalue/onnxruntime_go`'s dlopen shim is a cgo file on every platform, so
`CGO_ENABLED=0` fails to build (`build constraints exclude all Go files`).
Any C compiler works — nothing beyond libc/libdl gets linked, ONNX Runtime
itself stays dlopen'd at runtime (see below), so this doesn't need the
ONNX Runtime headers/libs to be present at build time, just a compiler.
Linux/macOS runners and dev machines normally have one already (gcc/clang);
on Windows, a toolchain like [MSYS2/MinGW-w64](https://www.msys2.org/) is
needed.

### ONNX Runtime shared library

Go inference uses [`yalue/onnxruntime_go`](https://github.com/yalue/onnxruntime_go),
which loads the ONNX Runtime shared library dynamically at runtime. **Needs ONNX
Runtime 1.24 or newer** (older versions, including what some distros package via
apt, are missing C API symbols this binding version requires and fail to load with
an explicit API-version error rather than silently misbehaving):

```bash
# macOS
brew install onnxruntime

# Linux — download a release from
# https://github.com/microsoft/onnxruntime/releases and place
# libonnxruntime.so on your library path (e.g. /usr/local/lib)
```

The library is auto-detected at common locations (`/usr/local/lib`,
`/opt/homebrew/lib`, `/usr/lib`, …). To point at a specific file, set
`ONNXRUNTIME_LIB_PATH`:

```bash
export ONNXRUNTIME_LIB_PATH=/path/to/libonnxruntime.so
```

## Available Models

| Model | Parameters | Original weights (HF) | ONNX export used by this repo |
|---|---|---|---|
| kitten-asr-tiny | 471M | [KittenML/kitten-asr-tiny](https://huggingface.co/KittenML/kitten-asr-tiny) | [zhaoyang-jia/kitten-asr-tiny-onnx](https://huggingface.co/zhaoyang-jia/kitten-asr-tiny-onnx) (1.9 GB) |
| kitten-asr-small-enhanced | 782M | [KittenML/kitten-asr-small-enhanced](https://huggingface.co/KittenML/kitten-asr-small-enhanced) | [zhaoyang-jia/kitten-asr-small-enhanced-onnx](https://huggingface.co/zhaoyang-jia/kitten-asr-small-enhanced-onnx) (3.8 GB) |

The "Original weights" repos are KittenML's own, upstream — that's where the
`tools/` pipeline below reads from. The "ONNX export" repos are this
project's own conversion of those weights, and what `fetch_model.sh` and the
Go engine actually consume. The ONNX export is bigger than the original
weights because it's fp32 (ONNX Runtime's dynamo exporter doesn't handle
bf16 well); the originals ship as bf16 safetensors.

### Downloading a model

Models are not vendored in this repository. Fetch one into `./models` with
the helper script:

```bash
scripts/fetch_model.sh tiny            # or: small-enhanced
```

This downloads into `./models/kitten-asr-<name>-onnx` (git-ignored), ready
for `kitten-asr.New(dir)` / the CLI and server below.

### Building a model directory from source instead

The ONNX files above are produced by the `tools/` Python pipeline from the
original PyTorch weights (needs `torch`, `transformers>=5.17`,
`safetensors`, `onnx`, `onnxscript`, `onnxruntime`). Output filenames follow
KittenML's own ONNX naming convention on the TTS side (e.g.
`kitten_tts_nano_v0_8.onnx`) — a `kitten_asr_<model>_` prefix on each of the
three artifacts (`Model.New` finds them by suffix, so the exact prefix
doesn't matter, just that all three files in a directory share one):

```bash
cd tools
python3 load_and_check.py <path-to-downloaded-kitten-asr-tiny>          # sanity-check the architecture reconstruction
python3 export_audio_encoder.py <same-path> <out>/kitten_asr_tiny_audio_encoder.onnx
python3 export_decoder.py <same-path> <out>/kitten_asr_tiny_decoder.onnx
python3 export_embed_tokens.py <same-path> <out>/kitten_asr_tiny_embed_tokens.bin
cp <same-path>/{config.json,vocab.json,merges.txt,added_tokens.json} <out>/
```

`<out>` is then a self-contained model directory, same shape as what
`fetch_model.sh` downloads. The same steps work unmodified for
`kitten-asr-small-enhanced` (point at its downloaded directory instead, and
use a `kitten_asr_small_enhanced_` prefix instead so the two models' files
stay distinguishable if they ever end up in the same directory).

## Build

```bash
go build -o bin/ ./...
# Binaries at: bin/kitten-asr and bin/kitten-asr-server
```

## Transcribe (CLI)

```bash
./bin/kitten-asr <model_dir> <audio_file>

# Also print the model's detected language (to stderr):
./bin/kitten-asr -language <model_dir> <audio_file>

# Tune ONNX Runtime's intra-op thread count (default 4; more isn't always
# faster on memory-constrained machines):
./bin/kitten-asr -threads 2 <model_dir> <audio_file>
```

`<audio_file>` can be `.wav` (any sample rate/bit depth: 8/16/24/32-bit PCM or
32-bit float, any channel count, downmixed to mono), `.mp3`, or `.flac`; all are
resampled to 16kHz internally. (Headerless raw PCM has no filename/extension to
dispatch on, so it isn't wired into the CLI -- use `asr.DecodePCM` directly as a
library, see below.) Audio longer than 30 seconds (the model's native window) is
transcribed in consecutive chunks automatically.

## Run the API Server

```bash
./bin/kitten-asr-server -host 0.0.0.0 -port 8080 <model_dir>
```

Implements the same endpoint shape as OpenAI's Whisper transcription API:

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F file=@hello.wav
# {"text":"Hello, this is a test."}

curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F file=@hello.mp3 -F response_format=verbose_json
# {"task":"transcribe","language":"English","duration":4.06,"text":"Hello, this is a test."}
```

**Request fields** (multipart form): `file` (required), `response_format`
(`json` default, `text`, `verbose_json`; `srt`/`vtt` return a 400 — this model
has no per-segment timestamps to put in them). `model`, `language`, `prompt`,
`temperature` are accepted for OpenAI client compatibility but have no effect
(language is always auto-detected, the model's chat template drops extra
prompt text, generation is always greedy).

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/audio/transcriptions` | Transcribe an uploaded audio file |
| `GET` | `/v1/audio/transcriptions/stream` | Live transcription over a WebSocket — see below |
| `GET` | `/v1/models` | List loaded model |
| `GET` | `/health` | Health check |

### Live / streaming transcription

`GET /v1/audio/transcriptions/stream` upgrades to a WebSocket for live audio
(e.g. from a browser mic). **Read this carefully before wiring it up:** the
model has no incremental/causal inference mode, so there's no such thing as
true frame-by-frame streaming here. What this endpoint actually does is
re-transcribe a sliding window of the most recent audio every time enough new
audio has arrived — the same tradeoff most "streaming Whisper" wrappers make.
Concretely, that means **each `partial` message can revise the previous
one** as more audio provides more context — a UI consuming this should
*replace* its display with each `partial`, not append to it.

Connect with PCM format as query parameters (raw PCM has no header
describing itself, unlike WAV/FLAC/MP3):

```
ws://localhost:8080/v1/audio/transcriptions/stream
    ?sample_rate=16000&bits_per_sample=16&channels=1
    &window_seconds=10&min_trigger_seconds=1.5   # optional, see defaults below
```

Then:
- **Send** binary frames of raw PCM audio (in the format declared above) as
  they become available — small, frequent chunks are fine (e.g. every
  100-250ms); the server buffers internally and only re-transcribes once
  `min_trigger_seconds` of new audio has accumulated.
- **Send** a text frame `{"type":"flush"}` to force an immediate pass on
  whatever's buffered (e.g. the user paused speaking), or `{"type":"end"}`
  when the audio source has ended — the server replies once more with
  `{"type":"final",...}` and closes the connection.
- **Receive** text frames: `{"type":"partial","language":"English","text":"..."}`
  on each triggered pass, `{"type":"final",...}` once in response to `"end"`,
  or `{"type":"error","message":"..."}` if one message couldn't be processed
  (the connection stays open).

`window_seconds` (default 20, max 30 — the model's hard per-window limit) and
`min_trigger_seconds` (default 1.5) trade off latency, accuracy, and cost: a
larger window gives the model more context (and costs more per pass); a
smaller `min_trigger_seconds` updates more often (and repeats more work
re-decoding audio it already saw).

## Use as a library

```go
package main

import (
	"fmt"

	"github.com/itamaker/kitten-asr-go/asr"
)

func main() {
	model, err := asr.New("models/kitten-asr-tiny-onnx")
	if err != nil {
		panic(err)
	}
	defer model.Close()

	samples, sampleRate, err := asr.ReadWAV("hello.wav") // or asr.DecodeMP3 / asr.DecodeFLAC / asr.DecodePCM / asr.DecodeAudio
	if err != nil {
		panic(err)
	}

	result, err := model.Transcribe(samples, sampleRate)
	if err != nil {
		panic(err)
	}
	fmt.Println(result.Language, "-", result.Text)
}
```

`asr.Stream` (what the server's WebSocket endpoint wraps) is available directly
too, for embedding live transcription into your own Go program without going
through HTTP at all:

```go
stream, err := model.NewStream(asr.StreamOptions{}) // zero value = defaults
if err != nil {
	panic(err)
}
for chunk := range liveAudioChunks { // your own mic/capture source
	if result, ok, err := stream.Feed(chunk, sampleRate); err != nil {
		panic(err)
	} else if ok {
		fmt.Println("partial:", result.Text) // may revise the previous partial
	}
}
final, err := stream.Flush()
```

## License

MIT — see [LICENSE](LICENSE).

This is an independent Go implementation. The upstream KittenML models are not
vendored in this repository and are downloaded/derived separately; check KittenML's
own licensing for the model weights themselves.

## Acknowledgments

- [KittenML](https://kittenml.com) for the original kitten-asr models
- [yalue/onnxruntime_go](https://github.com/yalue/onnxruntime_go) for the Go ONNX
  Runtime bindings
- [dlclark/regexp2](https://github.com/dlclark/regexp2) for lookahead-capable
  regex (needed by the tokenizer's pretokenizer pattern)
- [gonum](https://gonum.org) for FFT
- [hajimehoshi/go-mp3](https://github.com/hajimehoshi/go-mp3) for pure-Go MP3
  decoding
- [mewkiz/flac](https://github.com/mewkiz/flac) for pure-Go FLAC decoding
  (also used by kitten-tts-go, for FLAC encoding)
- [gin-gonic/gin](https://github.com/gin-gonic/gin) for the HTTP server's routing
- [coder/websocket](https://github.com/coder/websocket) for the live-transcription
  WebSocket endpoint
