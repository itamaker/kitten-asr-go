"""Final validation: run the FULL pipeline using ONLY the two exported ONNX graphs
via onnxruntime (no PyTorch model calls at all) -- audio_encoder.onnx for the audio
tower, decoder.onnx (with explicit KV-cache tensor I/O) for the autoregressive text
decoder, plus a plain numpy embedding-table lookup for embed_tokens (no need for a
whole ONNX graph just for a lookup). Tokenizer/feature-extractor/chat-template stay
on the transformers library here since they're pure data processing, not model
inference -- those get reimplemented in Go directly, not exported.
"""
import glob
import sys
import numpy as np
import onnxruntime as ort
import torch
from safetensors import safe_open
from transformers import AutoTokenizer, WhisperFeatureExtractor

from load_and_check import build_config
from infer_test import load_audio_16k
from export_audio_encoder import _feat_extract_out_len

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "/mnt/d/code/project/kitten-asr-go/models/kitten-asr-tiny"
ONNX_DIR = sys.argv[2] if len(sys.argv) > 2 else "/mnt/d/code/project/kitten-asr-go/models/onnx-tiny"
WAV_PATH = sys.argv[3] if len(sys.argv) > 3 else "testaudio/piper_hello.wav"
N_FRAMES = 3000

config = build_config(MODEL_DIR)
tc = config.text_config
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
chat_template = open(f"{MODEL_DIR}/chat_template.jinja").read()

print("Loading embed_tokens weight matrix directly from safetensors...")
with safe_open(f"{MODEL_DIR}/model.safetensors", framework="pt") as f:
    embed_tokens = f.get_tensor("thinker.model.embed_tokens.weight").to(torch.float32).numpy()
print("  shape:", embed_tokens.shape)

def _find(onnx_dir, suffix):
    matches = glob.glob(f"{onnx_dir}/*{suffix}")
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one *{suffix} in {onnx_dir}, found {matches}")
    return matches[0]

print("Loading ONNX sessions...")
audio_sess = ort.InferenceSession(_find(ONNX_DIR, "_audio_encoder.onnx"), providers=["CPUExecutionProvider"])
decoder_sess = ort.InferenceSession(_find(ONNX_DIR, "_decoder.onnx"), providers=["CPUExecutionProvider"])
num_layers = tc.num_hidden_layers

# --- Audio: pad/truncate to a fixed 30s/3000-frame window, run the static encoder ---
audio = load_audio_16k(WAV_PATH)
audio_inputs = feature_extractor(audio, sampling_rate=16000, padding="max_length", truncation=True,
                                  return_attention_mask=True, return_tensors="np")
mel = audio_inputs["input_features"].astype(np.float32)  # (1, 128, 3000)
assert mel.shape == (1, config.audio_config.num_mel_bins, N_FRAMES), mel.shape
real_frames = int(audio_inputs["attention_mask"].sum())
print(f"  audio: {len(audio)/16000:.2f}s real ({real_frames}/{N_FRAMES} real mel frames, rest silence-padded)")

(audio_embeds,) = audio_sess.run(None, {"input_features": mel})
audio_out_len = _feat_extract_out_len(N_FRAMES, config.audio_config.n_window)
assert audio_embeds.shape == (audio_out_len, config.audio_config.output_dim), audio_embeds.shape
print("  audio_embeds shape:", audio_embeds.shape)

# --- Prompt: chat template with exactly audio_out_len <|audio_pad|> placeholders ---
audio_token = "<|audio_pad|>"
messages = [{"role": "system", "content": ""}, {"role": "user", "content": [{"type": "audio", "audio_url": "x"}]}]
templated = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           chat_template=chat_template)
templated = templated.replace(audio_token, audio_token * audio_out_len)
enc = tokenizer(templated, return_tensors="np")
input_ids = enc["input_ids"][0]  # (prompt_len,)
print("  prompt tokens:", len(input_ids))

inputs_embeds = embed_tokens[input_ids]  # (prompt_len, hidden)
audio_positions = np.where(input_ids == config.audio_token_id)[0]
assert len(audio_positions) == audio_out_len
inputs_embeds[audio_positions] = audio_embeds
inputs_embeds = inputs_embeds[None, :, :]  # (1, prompt_len, hidden)

# --- Decode loop, feeding explicit KV cache tensors in/out of decoder.onnx ---
def empty_kv():
    feed = {}
    for i in range(num_layers):
        feed[f"past_key_{i}"] = np.zeros((1, tc.num_key_value_heads, 0, tc.head_dim), dtype=np.float32)
        feed[f"past_value_{i}"] = np.zeros((1, tc.num_key_value_heads, 0, tc.head_dim), dtype=np.float32)
    return feed


kv_feed = empty_kv()
attention_mask = np.ones((1, inputs_embeds.shape[1]), dtype=np.int64)

feed = {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask, **kv_feed}
outputs = decoder_sess.run(None, feed)
logits, present = outputs[0], outputs[1:]

generated = []
next_id = int(logits[0, -1].argmax())
MAX_NEW = 100
for step in range(MAX_NEW):
    generated.append(next_id)
    if next_id == config.eos_token_id:
        break
    next_embed = embed_tokens[next_id][None, None, :]  # (1, 1, hidden)
    attention_mask = np.concatenate([attention_mask, np.ones((1, 1), dtype=np.int64)], axis=1)
    kv_feed = {}
    for i in range(num_layers):
        kv_feed[f"past_key_{i}"] = present[2 * i]
        kv_feed[f"past_value_{i}"] = present[2 * i + 1]
    feed = {"inputs_embeds": next_embed, "attention_mask": attention_mask, **kv_feed}
    outputs = decoder_sess.run(None, feed)
    logits, present = outputs[0], outputs[1:]
    next_id = int(logits[0, -1].argmax())

print("\n=== ONNX-ONLY PIPELINE RESULT ===")
print("tokens:", generated)
print("transcript:", repr(tokenizer.decode(generated, skip_special_tokens=True)))
