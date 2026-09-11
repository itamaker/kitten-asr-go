"""Export the text decoder step (DecoderStep from decoder_wrapper.py) to ONNX with
dynamic sequence length and dynamic past-KV length -- unlike the audio encoder this
one genuinely needs dynamic shapes (autoregressive generation is inherently
variable-length), so no fixed-window trick applies here.
"""
import sys
import torch

from load_and_check import build_config, load_checkpoint_weights
from decoder_wrapper import DecoderStep, empty_past_kv
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerTextRotaryEmbedding,
)

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "models/kitten-asr-tiny"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "onnx_out/kitten_asr_tiny_decoder.onnx"


def main():
    import os
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    config = build_config(MODEL_DIR)
    with torch.device("meta"):
        model = Qwen3OmniMoeThinkerForConditionalGeneration(config)
    missing, unexpected = load_checkpoint_weights(model, MODEL_DIR, prefix_filter=("model.", "lm_head."))
    assert not unexpected, unexpected
    # This script only loads/uses the text model + lm_head -- audio_tower and
    # visual are legitimately never populated here (audio_tower is loaded and
    # exported separately by export_audio_encoder.py; visual is never used at
    # all, see load_and_check.py).
    unaccounted = [m for m in missing if not m.startswith(("visual.", "audio_tower."))]
    assert not unaccounted, unaccounted
    model.model.rotary_emb = Qwen3OmniMoeThinkerTextRotaryEmbedding(config.text_config)
    model.eval()

    wrapper = DecoderStep(model)
    wrapper.eval()
    tc = config.text_config
    num_layers = tc.num_hidden_layers

    # Sample inputs: a "mid-generation" shape (some past, one new token) exercises
    # both dynamic dims away from their 0/1 edge values.
    PAST_LEN, SEQ_LEN = 5, 3
    inputs_embeds = torch.randn(1, SEQ_LEN, tc.hidden_size)
    attention_mask = torch.ones(1, PAST_LEN + SEQ_LEN, dtype=torch.long)
    past_kv = []
    for _ in range(num_layers):
        past_kv.append(torch.randn(1, tc.num_key_value_heads, PAST_LEN, tc.head_dim))
        past_kv.append(torch.randn(1, tc.num_key_value_heads, PAST_LEN, tc.head_dim))

    with torch.no_grad():
        ref_out = wrapper(inputs_embeds, attention_mask, *past_kv)
    print("PyTorch logits shape:", ref_out[0].shape, "num kv outputs:", len(ref_out) - 1)

    seq_len_dim = torch.export.Dim("seq_len", min=1, max=4096)
    past_len_dim = torch.export.Dim("past_len", min=0, max=65536)
    total_len_dim = torch.export.Dim("total_len", min=1, max=65536 + 4096)

    input_names = ["inputs_embeds", "attention_mask"]
    output_names = ["logits"]
    past_kv_shapes = []
    for i in range(num_layers):
        input_names += [f"past_key_{i}", f"past_value_{i}"]
        output_names += [f"present_key_{i}", f"present_value_{i}"]
        past_kv_shapes += [{2: past_len_dim}, {2: past_len_dim}]
    # forward(self, inputs_embeds, attention_mask, *past_kv) -- torch.export sees
    # the *args as ONE nested tuple element, not N flat ones, so dynamic_shapes
    # must mirror that: [embeds_spec, mask_spec, [spec, spec, ...]].
    dynamic_shapes = [{1: seq_len_dim}, {1: total_len_dim}, tuple(past_kv_shapes)]

    print(f"Exporting to {OUT_PATH} (this is a big graph, may take a while) ...")
    torch.onnx.export(
        wrapper,
        (inputs_embeds, attention_mask, *past_kv),
        OUT_PATH,
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes=tuple(dynamic_shapes),
        dynamo=True,
        external_data=True,
        opset_version=18,
    )
    print("Export done.")

    import onnxruntime as ort
    import numpy as np
    sess = ort.InferenceSession(OUT_PATH, providers=["CPUExecutionProvider"])
    feed = {"inputs_embeds": inputs_embeds.numpy(), "attention_mask": attention_mask.numpy()}
    for i in range(num_layers):
        feed[f"past_key_{i}"] = past_kv[2 * i].numpy()
        feed[f"past_value_{i}"] = past_kv[2 * i + 1].numpy()
    ort_outs = sess.run(None, feed)
    logit_diff = np.abs(ort_outs[0] - ref_out[0].detach().numpy())
    print("logits max abs diff PyTorch vs ONNXRuntime:", logit_diff.max(), "mean:", logit_diff.mean())
    kv_max = max(
        np.abs(ort_outs[1 + i] - ref_out[1 + i].detach().numpy()).max() for i in range(2 * num_layers)
    )
    print("KV cache outputs max abs diff:", kv_max)


if __name__ == "__main__":
    main()
