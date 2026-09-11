import sys
import torch
from safetensors import safe_open
from transformers import AutoTokenizer, WhisperFeatureExtractor

from load_and_check import build_config
from infer_test import get_feat_extract_output_lengths, load_audio_16k
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
    SinusoidsPositionEmbedding,
)

MODEL_DIR = "models/kitten-asr-tiny"
WAV_PATH = sys.argv[1] if len(sys.argv) > 1 else "testaudio/piper_hello.wav"
SYSTEM_TEXT = sys.argv[2] if len(sys.argv) > 2 else ""

config = build_config(MODEL_DIR)
n_window = config.audio_config.n_window
audio_token = "<|audio_pad|>"
audio_token_id = config.audio_token_id

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
chat_template = open(f"{MODEL_DIR}/chat_template.jinja").read()

audio = load_audio_16k(WAV_PATH)
audio_inputs = feature_extractor(audio, sampling_rate=16000, padding=True, truncation=False,
                                  return_attention_mask=True, return_tensors="pt")
input_features = audio_inputs["input_features"]
feature_attention_mask = audio_inputs["attention_mask"]
audio_out_len = get_feat_extract_output_lengths(feature_attention_mask.sum(-1), n_window=n_window).item()

messages = [
    {"role": "system", "content": SYSTEM_TEXT},
    {"role": "user", "content": [{"type": "audio", "audio_url": "placeholder"}]},
]
templated = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           chat_template=chat_template)
templated = templated.replace(audio_token, audio_token * audio_out_len)
enc = tokenizer(templated, return_tensors="pt")
input_ids = enc["input_ids"]
attention_mask = enc["attention_mask"]
print("prompt tokens:", input_ids.shape[1], "audio tokens:", (input_ids == audio_token_id).sum().item())

with torch.device("meta"):
    model = Qwen3OmniMoeThinkerForConditionalGeneration(config)
sd = {}
with safe_open(f"{MODEL_DIR}/model.safetensors", framework="pt") as f:
    for k in f.keys():
        sd[k[len("thinker."):]] = f.get_tensor(k).to(torch.float32)
model.load_state_dict(sd, strict=False, assign=True)
model.audio_tower.positional_embedding = SinusoidsPositionEmbedding(
    config.audio_config.max_source_positions, config.audio_config.d_model)
model.eval()

with torch.no_grad():
    out = model(input_ids=input_ids, attention_mask=attention_mask,
                input_features=input_features, feature_attention_mask=feature_attention_mask,
                use_cache=False)
logits = out.logits  # (1, seq, vocab)
last_logits = logits[0, -1]
probs = torch.softmax(last_logits.float(), dim=-1)
top = torch.topk(probs, 15)
print("\n=== top-15 candidates for FIRST generated token ===")
for p, idx in zip(top.values.tolist(), top.indices.tolist()):
    tok_str = tokenizer.decode([idx])
    print(f"  {p:.4f}  id={idx:6d}  {tok_str!r}")

# Also inspect a few earlier positions (within the audio span and right after <|audio_end|>)
print("\n=== entropy of predicted distribution at every position (last 10) ===")
ent = -(probs.clamp_min(1e-12).log() * probs).sum()
print("entropy at final position:", ent.item(), "(log(vocab)=%.2f = uniform)" % torch.log(torch.tensor(151936.0)).item())

for pos in range(max(0, logits.shape[1]-6), logits.shape[1]):
    p = torch.softmax(logits[0, pos].float(), dim=-1)
    e = -(p.clamp_min(1e-12).log() * p).sum().item()
    top1 = p.argmax().item()
    print(f"pos {pos:3d} (input tok={input_ids[0,pos].item():6d} {tokenizer.decode([input_ids[0,pos].item()])!r:12}) "
          f"-> argmax next={top1:6d} {tokenizer.decode([top1])!r:12} entropy={e:.3f}")
