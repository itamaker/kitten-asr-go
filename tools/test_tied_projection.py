"""Validates DecoderStep(project_logits=False)'s hidden-state output against
a manual projection through embed_tokens.weight, for a tied-embedding model
(kitten-asr-small-enhanced) -- the numerical check behind skipping lm_head in
that model's decoder.onnx export (see decoder_wrapper.py's DecoderStep and
export_decoder.py).

Three generation loops are run over the same audio and compared token-for-
token:
  A. model.generate() -- the library's own reference implementation.
  B. DecoderStep(project_logits=True) manual loop -- today's exported graph
     shape (lm_head baked in), same as test_decoder_wrapper.py checks.
  C. DecoderStep(project_logits=False) manual loop, with the caller (this
     script) projecting each step's hidden state through embed_tokens.weight
     by hand -- exactly what export_decoder.py now produces for a tied model,
     and what asr.Model.projectLogits does in Go at runtime.

All three must produce the identical token sequence: B and C only differ in
*where* the vocab projection happens (inside vs. outside the traced graph),
never in the weights used, so any mismatch means a wiring bug, not expected
numerical noise.
"""
import sys
import torch
from safetensors import safe_open
from transformers import AutoTokenizer, WhisperFeatureExtractor

from load_and_check import build_config
from infer_test import get_feat_extract_output_lengths, load_audio_16k
from decoder_wrapper import DecoderStep, empty_past_kv
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
    SinusoidsPositionEmbedding,
    Qwen3OmniMoeThinkerTextRotaryEmbedding,
)

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "models/kitten-asr-small-enhanced"
WAV_PATH = sys.argv[2] if len(sys.argv) > 2 else "testaudio/piper_hello.wav"
MAX_NEW_TOKENS = 80

config = build_config(MODEL_DIR)
assert config.text_config.tie_word_embeddings, (
    f"{MODEL_DIR} has tie_word_embeddings=False -- this script only makes sense for a "
    f"tied model, where projecting through embed_tokens.weight instead of lm_head.weight "
    f"is valid because they're the same matrix. Use test_decoder_wrapper.py instead."
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
chat_template = open(f"{MODEL_DIR}/chat_template.jinja").read()

audio = load_audio_16k(WAV_PATH)
audio_inputs = feature_extractor(audio, sampling_rate=16000, padding=True, truncation=False,
                                  return_attention_mask=True, return_tensors="pt")
input_features = audio_inputs["input_features"]
feature_attention_mask = audio_inputs["attention_mask"]
audio_out_len = get_feat_extract_output_lengths(feature_attention_mask.sum(-1), n_window=config.audio_config.n_window).item()

audio_token = "<|audio_pad|>"
messages = [{"role": "system", "content": ""}, {"role": "user", "content": [{"type": "audio", "audio_url": "x"}]}]
templated = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, chat_template=chat_template)
templated = templated.replace(audio_token, audio_token * audio_out_len)
enc = tokenizer(templated, return_tensors="pt")
input_ids = enc["input_ids"]
attention_mask = enc["attention_mask"]

print(f"Loading model from {MODEL_DIR} ...")
with torch.device("meta"):
    model = Qwen3OmniMoeThinkerForConditionalGeneration(config)
sd = {}
with safe_open(f"{MODEL_DIR}/model.safetensors", framework="pt") as f:
    for k in f.keys():
        stripped = k[len("thinker."):]
        if stripped == "lm_head.weight":
            continue  # legitimately absent when tied -- see load_and_check.py
        sd[stripped] = f.get_tensor(k).to(torch.float32)
model.load_state_dict(sd, strict=False, assign=True)
model.lm_head.weight = model.model.embed_tokens.weight  # re-tie, same as load_and_check.load_checkpoint_weights
model.audio_tower.positional_embedding = SinusoidsPositionEmbedding(
    config.audio_config.max_source_positions, config.audio_config.d_model)
model.model.rotary_emb = Qwen3OmniMoeThinkerTextRotaryEmbedding(config.text_config)
model.eval()
embed_tokens_weight = model.model.embed_tokens.weight.detach()  # (vocab_size, hidden_size)

with torch.no_grad():
    inputs_embeds = model.get_input_embeddings()(input_ids)
    audio_feat = model.get_audio_features(input_features, feature_attention_mask).last_hidden_state
    audio_mask = (input_ids == config.audio_token_id).unsqueeze(-1).expand_as(inputs_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_feat.to(inputs_embeds.dtype))

# --- A: model.generate() ---
with torch.no_grad():
    ref_gen = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                              input_features=input_features, feature_attention_mask=feature_attention_mask,
                              max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1)
tokens_a = ref_gen[0, input_ids.shape[1]:].tolist()
print("A. model.generate():         ", tokens_a)


def run_decoder_step_loop(project_logits: bool):
    decoder_step = DecoderStep(model, project_logits=project_logits)
    decoder_step.eval()
    with torch.no_grad():
        past_kv = empty_past_kv(config)
        am = attention_mask.clone()
        out, *past_kv = decoder_step(inputs_embeds, am, *past_kv)
        logits = out if project_logits else out @ embed_tokens_weight.T

        generated = []
        next_id = logits[0, -1].argmax().item()
        for _ in range(MAX_NEW_TOKENS):
            generated.append(next_id)
            if next_id == config.eos_token_id:
                break
            next_embed = model.get_input_embeddings()(torch.tensor([[next_id]]))
            am = torch.cat([am, torch.ones(1, 1, dtype=am.dtype)], dim=1)
            out, *past_kv = decoder_step(next_embed, am, *past_kv)
            logits = out if project_logits else out @ embed_tokens_weight.T
            next_id = logits[0, -1].argmax().item()
    return generated


# --- B: today's graph shape (lm_head baked in) ---
tokens_b = run_decoder_step_loop(project_logits=True)
print("B. DecoderStep(logits=True):  ", tokens_b)

# --- C: new graph shape (hidden state out, manual projection) ---
tokens_c = run_decoder_step_loop(project_logits=False)
print("C. DecoderStep(logits=False):", tokens_c)

print()
print("A == B (existing graph shape matches reference):", tokens_a == tokens_b)
print("A == C (new graph shape matches reference):      ", tokens_a == tokens_c)
print("B == C (old and new graph shapes agree):          ", tokens_b == tokens_c)
if not (tokens_a == tokens_b == tokens_c):
    print("\nMISMATCH -- do not export decoder.onnx with project_logits=False until this passes.")
    sys.exit(1)
print("\nOK: all three agree.")
