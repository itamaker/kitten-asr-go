"""Explicit-KV-cache decoder wrapper: reuses the library's tested attention/rope/
norm code internally (via a real DynamicCache), but exposes a plain-tensor
in/out contract (inputs_embeds + attention_mask + flat past_key/past_value list
-> logits + flat new_key/new_value list) suitable for ONNX export and a hand-written
Go generation loop, with no dependency on the HF Cache class at the call boundary.
"""
import torch
from transformers.cache_utils import DynamicCache


class DecoderStep(torch.nn.Module):
    def __init__(self, thinker_model):
        super().__init__()
        self.text_model = thinker_model.model
        self.lm_head = thinker_model.lm_head
        self.num_layers = thinker_model.config.text_config.num_hidden_layers

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, *past_kv: torch.Tensor):
        assert len(past_kv) == 2 * self.num_layers
        pairs = [(past_kv[2 * i], past_kv[2 * i + 1]) for i in range(self.num_layers)]
        cache = DynamicCache(ddp_cache_data=pairs, config=self.text_model.config)

        out = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
        )
        # Only the next-token prediction is ever used (both the caller's
        # prefill and single-token decode steps only read the last position's
        # logits) -- projecting every prompt position through lm_head
        # (hidden_size x vocab_size, vocab_size=151936) is pure waste on a
        # multi-token prefill and was most of this graph's runtime.
        last_hidden = out.last_hidden_state[:, -1:, :]
        logits = self.lm_head(last_hidden)

        new_kv = []
        for i in range(self.num_layers):
            new_kv.append(cache.layers[i].keys)
            new_kv.append(cache.layers[i].values)
        return (logits, *new_kv)


def empty_past_kv(config, batch=1, device="cpu", dtype=torch.float32):
    """One (key, value) pair of shape (batch, num_kv_heads, 0, head_dim) per layer,
    flattened, for a from-scratch prefill call."""
    tc = config.text_config
    out = []
    for _ in range(tc.num_hidden_layers):
        out.append(torch.zeros(batch, tc.num_key_value_heads, 0, tc.head_dim, device=device, dtype=dtype))
        out.append(torch.zeros(batch, tc.num_key_value_heads, 0, tc.head_dim, device=device, dtype=dtype))
    return out
