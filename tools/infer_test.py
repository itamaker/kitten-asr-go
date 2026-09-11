"""Validation step 2: real end-to-end inference. Loads kitten-asr-tiny's weights into
our reconstructed Qwen3OmniMoeThinker model, feeds it a real synthesized WAV, and
prints the decoded transcript. This is the semantic correctness check (structural
weight-loading alone doesn't prove the chat template / audio-token expansion /
feature extraction are wired correctly).
"""
import sys

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoTokenizer, WhisperFeatureExtractor

from load_and_check import build_config
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
)

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "models/kitten-asr-tiny"
WAV_PATH = sys.argv[2] if len(sys.argv) > 2 else "testaudio/hello_16k.wav"
SYSTEM_TEXT = sys.argv[3] if len(sys.argv) > 3 else ""


def get_feat_extract_output_lengths(input_lengths: torch.Tensor, n_window: int = 50) -> torch.Tensor:
    """Verbatim port of Qwen3OmniMoe processor's formula: how many audio-encoder
    output frames a given number of mel frames will produce, used to know how many
    <|audio_pad|> placeholder tokens to expand into the prompt."""
    chunk_len = n_window * 2
    input_lengths_leave = input_lengths % chunk_len
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // chunk_len) * 13


def load_audio_16k(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = resample_poly(audio, 16000, sr).astype(np.float32)
    return audio


def main():
    config = build_config(MODEL_DIR)
    n_window = config.audio_config.n_window
    audio_token = "<|audio_pad|>"
    audio_token_id = config.audio_token_id

    print("Loading tokenizer + feature extractor...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
    chat_template = open(f"{MODEL_DIR}/chat_template.jinja").read()

    print("Loading + resampling audio:", WAV_PATH)
    audio = load_audio_16k(WAV_PATH)
    print(f"  {len(audio)} samples @ 16kHz = {len(audio) / 16000:.2f}s")

    audio_inputs = feature_extractor(
        audio, sampling_rate=16000, padding=True, truncation=False, return_attention_mask=True,
        return_tensors="pt",
    )
    input_features = audio_inputs["input_features"]
    feature_attention_mask = audio_inputs["attention_mask"]
    print("  input_features shape:", input_features.shape, "feature_attention_mask sum:",
          feature_attention_mask.sum().item())

    audio_out_len = get_feat_extract_output_lengths(feature_attention_mask.sum(-1), n_window=n_window).item()
    print("  expected audio-encoder output length (num <|audio_pad|> tokens):", audio_out_len)

    messages = [
        {"role": "system", "content": SYSTEM_TEXT},
        {"role": "user", "content": [{"type": "audio", "audio_url": "placeholder"}]},
    ]
    templated = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, chat_template=chat_template,
    )
    assert templated.count(audio_token) == 1, templated
    templated = templated.replace(audio_token, audio_token * audio_out_len)

    enc = tokenizer(templated, return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    n_audio_in_ids = (input_ids == audio_token_id).sum().item()
    print(f"  prompt token count: {input_ids.shape[1]} (audio placeholder tokens: {n_audio_in_ids})")
    assert n_audio_in_ids == audio_out_len

    print("\nBuilding model + loading weights (float32, CPU)...")
    with torch.device("meta"):
        model = Qwen3OmniMoeThinkerForConditionalGeneration(config)
    from safetensors import safe_open
    sd = {}
    with safe_open(f"{MODEL_DIR}/model.safetensors", framework="pt") as f:
        for k in f.keys():
            sd[k[len("thinker."):]] = f.get_tensor(k).to(torch.float32)
    model.load_state_dict(sd, strict=False, assign=True)
    # model.visual stays on meta (no real storage, never touched since we pass no
    # pixel_values) -- leaving it in place avoids disturbing generic Module code
    # that expects every registered submodule to be a real nn.Module.
    #
    # Non-persistent buffers (persistent=False) are invisible to state_dict(), so
    # load_state_dict() never touches them -- they stay stuck on meta from
    # construction. Both are pure functions of config, so just recompute on CPU:
    #  - audio tower's sinusoidal positional_embedding
    #  - text decoder's shared RoPE inv_freq/original_inv_freq (THE actual bug
    #    behind the garbage/repetitive transcripts: every attention layer's
    #    rotary position encoding was reading uninitialized meta memory).
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        SinusoidsPositionEmbedding,
        Qwen3OmniMoeThinkerTextRotaryEmbedding,
    )
    model.audio_tower.positional_embedding = SinusoidsPositionEmbedding(
        config.audio_config.max_source_positions, config.audio_config.d_model)
    model.model.rotary_emb = Qwen3OmniMoeThinkerTextRotaryEmbedding(config.text_config)
    model.eval()
    print("OK.")

    print("\nGenerating...")
    with torch.no_grad():
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            max_new_tokens=200,
            do_sample=False,
            num_beams=1,
        )
    new_tokens = gen[0, input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print("\n=== RAW TOKEN IDS ===")
    print(new_tokens.tolist())
    print("\n=== TRANSCRIPT ===")
    print(repr(text))


if __name__ == "__main__":
    main()
