"""Validation step 1: reconstruct kitten-asr-tiny's config as a Qwen3OmniMoeThinker
config, load the real safetensors weights with strict key-matching, and confirm
every non-vision tensor is accounted for. This is the ground-truth check that our
architecture understanding (dimensions, special token ids, rope params) is exactly
right before we trust the model to produce real transcriptions.
"""
import json
import sys

import torch
from safetensors import safe_open

from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
    Qwen3OmniMoeTextConfig,
    Qwen3OmniMoeThinkerConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
)

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "models/kitten-asr-tiny"


def build_config(model_dir: str) -> Qwen3OmniMoeThinkerConfig:
    raw = json.load(open(f"{model_dir}/config.json"))
    tc = raw["thinker_config"]
    ac_raw = tc["audio_config"]
    tx_raw = tc["text_config"]

    audio_config = Qwen3OmniMoeAudioEncoderConfig(
        num_mel_bins=ac_raw["num_mel_bins"],
        encoder_layers=ac_raw["encoder_layers"],
        encoder_attention_heads=ac_raw["encoder_attention_heads"],
        encoder_ffn_dim=ac_raw["encoder_ffn_dim"],
        d_model=ac_raw["d_model"],
        n_window=ac_raw["n_window"],
        output_dim=ac_raw["output_dim"],
        n_window_infer=ac_raw["n_window_infer"],
        conv_chunksize=ac_raw["conv_chunksize"],
        downsample_hidden_size=ac_raw["downsample_hidden_size"],
        max_source_positions=ac_raw["max_source_positions"],
        activation_function=ac_raw["activation_function"],
        scale_embedding=ac_raw["scale_embedding"],
        attention_dropout=ac_raw["attention_dropout"],
        dropout=ac_raw["dropout"],
    )

    text_config = Qwen3OmniMoeTextConfig(
        vocab_size=tx_raw["vocab_size"],
        hidden_size=tx_raw["hidden_size"],
        intermediate_size=tx_raw["intermediate_size"],
        num_hidden_layers=tx_raw["num_hidden_layers"],
        num_attention_heads=tx_raw["num_attention_heads"],
        num_key_value_heads=tx_raw["num_key_value_heads"],
        head_dim=tx_raw["head_dim"],
        hidden_act=tx_raw["hidden_act"],
        max_position_embeddings=tx_raw["max_position_embeddings"],
        rms_norm_eps=tx_raw["rms_norm_eps"],
        attention_bias=tx_raw["attention_bias"],
        attention_dropout=tx_raw["attention_dropout"],
        tie_word_embeddings=tx_raw["tie_word_embeddings"],
        use_cache=tx_raw["use_cache"],
        num_experts=0,  # kitten's checkpoint has plain dense MLP per layer, no MoE experts
        rope_parameters={
            "rope_type": tx_raw["rope_scaling"]["rope_type"],
            "rope_theta": tx_raw["rope_theta"],
            "mrope_section": tx_raw["rope_scaling"]["mrope_section"],
            "mrope_interleaved": tx_raw["rope_scaling"]["mrope_interleaved"],
        },
    )

    thinker_config = Qwen3OmniMoeThinkerConfig(
        audio_config=audio_config,
        text_config=text_config,
        audio_token_id=tc["audio_token_id"],
        audio_start_token_id=tc["audio_start_token_id"],
        user_token_id=tc["user_token_id"],
        # Not always present at the top level (e.g. kitten-asr-small-enhanced
        # omits it) -- when absent, mirror text_config's own value, since
        # that's what actually determines whether the checkpoint only has one
        # embedding matrix (small-enhanced: tied, lm_head.weight absent from
        # the checkpoint) or two separate ones (tiny: untied, both present).
        tie_word_embeddings=raw.get("tie_word_embeddings", tx_raw["tie_word_embeddings"]),
    )
    # Not a declared field on Qwen3OmniMoeThinkerConfig but present in kitten's
    # config.json; stash it as a plain attribute in case anything reads it.
    thinker_config.audio_end_token_id = tc["audio_end_token_id"]
    thinker_config.eos_token_id = raw["eos_token_id"]
    thinker_config.pad_token_id = raw["pad_token_id"]
    # get_rope_index() unconditionally reads vision_start_token_id even when no
    # vision input is ever passed. Kitten has no vision path/weights at all, but
    # the tokenizer still carries the standard Qwen vision special tokens
    # (confirmed via tokenizer_config.json's added_tokens_decoder), so these ids
    # are real and simply never appear in an audio-only prompt.
    thinker_config.vision_start_token_id = 151652
    thinker_config.vision_end_token_id = 151653
    return thinker_config


def load_checkpoint_weights(model, model_dir: str, prefix_filter: str | None = None):
    """Load kitten's safetensors checkpoint into model (already constructed,
    typically under torch.device("meta")), stripping the "thinker." prefix.
    If prefix_filter is given, only keys starting with it (after stripping
    "thinker.") are loaded -- used by the per-graph export scripts to load
    just the audio tower or just the text model + lm_head.

    Handles tied embeddings correctly: when config.tie_word_embeddings is set
    (true for kitten-asr-small-enhanced, false for kitten-asr-tiny),
    lm_head.weight is legitimately absent from the checkpoint (it shares
    storage with embed_tokens.weight) -- a plain load_state_dict(assign=True)
    does NOT preserve that sharing (assign replaces the embedding's Parameter
    object with a freshly loaded one, but lm_head.weight still points at the
    old object), so this explicitly re-ties them afterward when needed.

    Returns (missing_keys, unexpected_keys) with a legitimately-tied
    lm_head.weight excluded from missing_keys.
    """
    sd = {}
    with safe_open(f"{model_dir}/model.safetensors", framework="pt") as f:
        for k in f.keys():
            assert k.startswith("thinker."), f"unexpected top-level key prefix: {k}"
            stripped = k[len("thinker."):]
            if prefix_filter is not None and not stripped.startswith(prefix_filter):
                continue
            sd[stripped] = f.get_tensor(k).to(torch.float32)

    result = model.load_state_dict(sd, strict=False, assign=True)
    missing = list(result.missing_keys)

    tie = getattr(model.config, "tie_word_embeddings", False)
    if tie and "lm_head.weight" in missing and "lm_head.weight" not in sd:
        model.lm_head.weight = model.model.embed_tokens.weight
        missing.remove("lm_head.weight")

    return missing, list(result.unexpected_keys)


def main():
    config = build_config(MODEL_DIR)
    print("=== audio_config ===")
    print(config.audio_config)
    print("=== text_config (key fields) ===")
    for k in [
        "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "head_dim", "intermediate_size", "vocab_size", "rope_parameters",
    ]:
        print(f"  {k}: {getattr(config.text_config, k)}")
    print("audio_token_id", config.audio_token_id, "audio_start_token_id", config.audio_start_token_id,
          "user_token_id", config.user_token_id)

    print("\nConstructing model on meta device (should be instant, ~0 RAM)...")
    with torch.device("meta"):
        model = Qwen3OmniMoeThinkerForConditionalGeneration(config)
    print("OK: model constructed.")

    print(f"\nLoading real weights from {MODEL_DIR}/model.safetensors ...")
    missing, unexpected = load_checkpoint_weights(model, MODEL_DIR)

    non_visual_missing = [m for m in missing if not m.startswith("visual.")]
    print(f"\nmissing keys total: {len(missing)} (visual.*: {len(missing) - len(non_visual_missing)})")
    print("missing keys NOT under visual.* (should be empty or just lm_head if tied):")
    for m in non_visual_missing:
        print("  MISSING:", m)

    print(f"\nunexpected keys total: {len(unexpected)} (should be 0)")
    for u in unexpected:
        print("  UNEXPECTED:", u)

    if not non_visual_missing and not unexpected:
        print("\n*** ARCHITECTURE MATCH CONFIRMED: every non-vision tensor loaded cleanly. ***")
    else:
        print("\n*** MISMATCH: architecture assumptions need fixing (see above). ***")

    torch.save({"config": config, "ok": not non_visual_missing and not unexpected}, "/tmp/kitten_asr_load_check.pt")


if __name__ == "__main__":
    main()
