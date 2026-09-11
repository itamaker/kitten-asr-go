#!/usr/bin/env bash
#
# fetch_model.sh — download a kitten-asr ONNX model from Hugging Face into
# ./models.
#
# The models are not vendored in this repository (same "downloaded
# separately" convention as kitten-tts-go); this script pulls one so the
# project is self-contained for local runs.
#
# Unlike kitten-tts-go's models, these ONNX exports don't come from
# KittenML directly -- they're this repo's own export, hosted as public
# Hugging Face repos under zhaoyang-jia/.
#
# Usage:
#   scripts/fetch_model.sh [name]
#
#   name ∈ { tiny (default), small-enhanced }
#
# Environment:
#   HF_TOKEN  Optional. Only needed if you're rate-limited as an anonymous
#             downloader; a token (any scope) raises the limit.
#   FORCE=1   Re-download even if the model already exists.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

NAME="${1:-tiny}"

# name → (Hugging Face repo, on-disk file prefix). The file prefix matches
# KittenML's own ONNX naming convention on the TTS side (e.g.
# kitten_tts_nano_v0_8.onnx) -- underscores, model name baked into the
# filename itself.
case "$NAME" in
  tiny)
    HF_REPO="zhaoyang-jia/kitten-asr-tiny-onnx"
    PREFIX="kitten_asr_tiny"
    ;;
  small-enhanced)
    HF_REPO="zhaoyang-jia/kitten-asr-small-enhanced-onnx"
    PREFIX="kitten_asr_small_enhanced"
    ;;
  *)
    echo "Unknown model '$NAME'. Choose: tiny, small-enhanced" >&2
    exit 1
    ;;
esac

DEST="models/kitten-asr-${NAME}-onnx"
BASE="https://huggingface.co/${HF_REPO}/resolve/main"

if [[ -f "$DEST/config.json" && "${FORCE:-0}" != "1" ]]; then
  echo "Model already present at $DEST (set FORCE=1 to re-download)."
  exit 0
fi

command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }
mkdir -p "$DEST"

AUTH_HEADER=()
[[ -n "${HF_TOKEN:-}" ]] && AUTH_HEADER=(-H "Authorization: Bearer ${HF_TOKEN}")

fetch() { # <remote-name>
  echo "  downloading $1 ..."
  curl -fSL --retry 3 "${AUTH_HEADER[@]}" -o "$DEST/$1" "$BASE/$1"
}

# Fixed file list -- unlike kitten-tts-go's models, these aren't named
# dynamically from a manifest, the export pipeline (tools/export_*.py)
# always produces exactly these names (given the prefix above).
# decoder.onnx.data (the decoder's external-data companion) is deliberately
# NOT prefixed: ONNX Runtime resolves it via the location string recorded
# inside "${PREFIX}_decoder.onnx" itself (always literally "decoder.onnx.data"
# regardless of what the .onnx file is named), not by matching filenames.
for f in config.json vocab.json merges.txt added_tokens.json \
         "${PREFIX}_audio_encoder.onnx" "${PREFIX}_decoder.onnx" \
         decoder.onnx.data "${PREFIX}_embed_tokens.bin"; do
  fetch "$f"
done

echo "Done. Model at: $DEST"
ls -la "$DEST"
