"""Dump thinker.model.embed_tokens.weight as a flat row-major float32 binary
file: (vocab_size, hidden_size), no header. Small enough (vocab_size x
hidden_size x 4 bytes -- ~311MB for kitten-asr-tiny) to just load directly and
index into for embedding lookups, rather than exporting a whole ONNX graph for
a single nn.Embedding.
"""
import sys
import torch
from safetensors import safe_open

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "models/kitten-asr-tiny"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "models/onnx-tiny/kitten_asr_tiny_embed_tokens.bin"

with safe_open(f"{MODEL_DIR}/model.safetensors", framework="pt") as f:
    w = f.get_tensor("thinker.model.embed_tokens.weight").to(torch.float32)

print("embed_tokens shape:", tuple(w.shape))
w.numpy().tofile(OUT_PATH)
print("wrote", OUT_PATH)
