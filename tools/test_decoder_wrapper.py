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

MODEL_DIR = "models/kitten-asr-tiny"
WAV_PATH = sys.argv[1] if len(sys.argv) > 1 else "testaudio/piper_hello.wav"

config = build_config(MODEL_DIR)
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

print("Loading model...")
with torch.device("meta"):
    model = Qwen3OmniMoeThinkerForConditionalGeneration(config)
sd = {}
with safe_open(f"{MODEL_DIR}/model.safetensors", framework="pt") as f:
    for k in f.keys():
        sd[k[len("thinker."):]] = f.get_tensor(k).to(torch.float32)
model.load_state_dict(sd, strict=False, assign=True)
model.audio_tower.positional_embedding = SinusoidsPositionEmbedding(
    config.audio_config.max_source_positions, config.audio_config.d_model)
model.model.rotary_emb = Qwen3OmniMoeThinkerTextRotaryEmbedding(config.text_config)
model.eval()

# --- Reference: model.generate() (already validated correct) ---
with torch.no_grad():
    ref_gen = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                              input_features=input_features, feature_attention_mask=feature_attention_mask,
                              max_new_tokens=80, do_sample=False, num_beams=1)
ref_new_tokens = ref_gen[0, input_ids.shape[1]:].tolist()
print("REFERENCE (generate()) new tokens:", ref_new_tokens)
print("REFERENCE transcript:", repr(tokenizer.decode(ref_new_tokens, skip_special_tokens=True)))

# --- Candidate: manual greedy loop using DecoderStep (explicit KV tensors) ---
decoder_step = DecoderStep(model)
decoder_step.eval()

with torch.no_grad():
    # Build inputs_embeds for the prefill exactly like the library does internally.
    inputs_embeds = model.get_input_embeddings()(input_ids)
    audio_feat = model.get_audio_features(input_features, feature_attention_mask).last_hidden_state
    audio_mask = (input_ids == config.audio_token_id).unsqueeze(-1).expand_as(inputs_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_feat.to(inputs_embeds.dtype))

    past_kv = empty_past_kv(config)
    am = attention_mask.clone()
    logits, *past_kv = decoder_step(inputs_embeds, am, *past_kv)

    generated = []
    next_id = logits[0, -1].argmax().item()
    for step in range(80):
        generated.append(next_id)
        if next_id == config.eos_token_id:
            break
        next_embed = model.get_input_embeddings()(torch.tensor([[next_id]]))
        am = torch.cat([am, torch.ones(1, 1, dtype=am.dtype)], dim=1)
        logits, *past_kv = decoder_step(next_embed, am, *past_kv)
        next_id = logits[0, -1].argmax().item()

print("\nCANDIDATE (DecoderStep manual loop) new tokens:", generated)
print("CANDIDATE transcript:", repr(tokenizer.decode(generated, skip_special_tokens=True)))

print("\nMATCH:", generated == ref_new_tokens)
