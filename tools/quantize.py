"""Dynamic int8 quantization of a published kitten-asr ONNX model directory
(the layout scripts/fetch_model.sh produces: config.json, vocab.json,
merges.txt, added_tokens.json, <prefix>_audio_encoder.onnx,
<prefix>_decoder.onnx + decoder.onnx.data, <prefix>_embed_tokens.bin).

Quantizes only the two ONNX compute graphs (audio_encoder, decoder) with
onnxruntime.quantization.quantize_dynamic (weights -> int8, activations
computed dynamically at inference time -- no calibration data needed, unlike
static quantization). embed_tokens.bin is a plain lookup table, not a compute
graph -- quantizing it wouldn't save any inference compute, only load-time
I/O, and would require the Go loader to dequantize on every embedding lookup,
so it's left as fp32 and copied through unchanged, same as the tokenizer/
config files.

Usage:
    python3 tools/quantize.py <src_onnx_dir> <dst_onnx_dir>

Example:
    python3 tools/quantize.py models/kitten-asr-tiny-onnx models/kitten-asr-tiny-onnx-int8
"""
import glob
import os
import shutil
import sys
import tempfile

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


def _find_one(src_dir, suffix):
    matches = glob.glob(os.path.join(src_dir, f"*{suffix}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one *{suffix} in {src_dir}, found {matches}")
    return matches[0]


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    src_dir, dst_dir = sys.argv[1], sys.argv[2]
    os.makedirs(dst_dir, exist_ok=True)

    # Carry non-graph files through unchanged.
    for name in ["config.json", "vocab.json", "merges.txt", "added_tokens.json"]:
        shutil.copy(os.path.join(src_dir, name), os.path.join(dst_dir, name))
    embed_src = _find_one(src_dir, "_embed_tokens.bin")
    shutil.copy(embed_src, os.path.join(dst_dir, os.path.basename(embed_src)))

    for suffix in ["_audio_encoder.onnx", "_decoder.onnx"]:
        src = _find_one(src_dir, suffix)
        dst = os.path.join(dst_dir, os.path.basename(src))
        print(f"Quantizing {src} -> {dst} ...")
        before = os.path.getsize(src)
        # decoder.onnx's external-data companion isn't <basename>.data -- it's
        # always literally "decoder.onnx.data" regardless of the .onnx file's
        # own name (see asr/load.go's New doc comment); audio_encoder.onnx
        # has no external data at all pre-quantization (export_audio_encoder.py
        # uses external_data=False, small enough to embed inline), so only
        # look for the decoder's companion when it's the graph in hand.
        if suffix == "_decoder.onnx":
            before_data = os.path.join(os.path.dirname(src), "decoder.onnx.data")
            if os.path.exists(before_data):
                before += os.path.getsize(before_data)

        with tempfile.TemporaryDirectory() as tmp:
            # quantize_dynamic unconditionally reloads the graph through
            # onnx.shape_inference (inside ONNXQuantizer.__init__ ->
            # save_and_reload_model_with_shape_infer -- no flag of
            # quantize_dynamic's own controls this), which chokes on both
            # dynamo-exported graphs here with a stale/inconsistent
            # `Inferred shape and existing shape differ` error against a
            # *pre-existing* shape annotation baked in at export time --
            # nothing to do with quantization, and harmless at plain
            # inference (ONNX Runtime's own InferenceSession never
            # complains). value_info entries are optional, advisory shape
            # hints, not load-bearing for execution, so the robust fix is to
            # drop the stale ones before quantize_dynamic gets a chance to
            # reload the graph and hit the conflict: with nothing to
            # conflict against, shape inference just (re)computes them
            # fresh.
            model = onnx.load(src)
            del model.graph.value_info[:]
            preprocessed = os.path.join(tmp, "preprocessed.onnx")
            onnx.save_model(
                model, preprocessed,
                save_as_external_data=True, all_tensors_to_one_file=True,
                location=os.path.basename(preprocessed) + ".data",
            )
            quantize_dynamic(
                model_input=preprocessed,
                model_output=dst,
                weight_type=QuantType.QInt8,
                # decoder.onnx is >2GB in float32 (external data); keep the
                # quantized graph's own weights external too instead of
                # forcing everything back into one in-memory protobuf.
                use_external_data_format=True,
            )
        after = os.path.getsize(dst)
        after_data = dst + ".data"
        if os.path.exists(after_data):
            after += os.path.getsize(after_data)
        print(f"  {before/1e6:.1f}MB -> {after/1e6:.1f}MB ({after/before*100:.0f}%)")

    # quantize_dynamic names the decoder's external-data companion after its
    # own output filename (e.g. "kitten_asr_tiny_decoder.onnx.data") rather
    # than preserving the input graph's "decoder.onnx.data" convention --
    # that's fine and doesn't need fixing up: ONNX Runtime resolves external
    # data via the location string recorded *inside* the .onnx file, not by a
    # hardcoded filename, and asr.New's Go-side glob only ever looks for
    # *_decoder.onnx itself (see asr/load.go), never the companion by name.
    print(f"\nDone. Quantized model at: {dst_dir}")


if __name__ == "__main__":
    main()
