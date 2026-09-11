"""Export the audio tower as a fixed-shape ONNX graph: always a full 30s / 3000-frame
mel window in (matches config.audio_config.max_source_positions=1500 exactly -- the
model has no representation for longer audio anyway), always exactly 390 embedding
vectors out. This sidesteps the data-dependent chunk/window logic in
chunk_and_pad_features/get_audio_cu_seqlens entirely -- for a length that's an exact
multiple of n_window*2=100, every intermediate shape is a compile-time constant.
Longer audio is the caller's job to chunk into <=30s pieces (a v2 feature).
"""
import sys
import torch
from safetensors import safe_open

from load_and_check import build_config
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
    SinusoidsPositionEmbedding,
)

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "models/kitten-asr-tiny"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "onnx_out/kitten_asr_tiny_audio_encoder.onnx"

N_FRAMES = 3000  # 30s at hop_length=160, sampling_rate=16000


def _feat_extract_out_len(n: int, n_window: int) -> int:
    """Plain-Python (compile-time-constant) port of the model's
    _get_feat_extract_output_lengths, for use where the input length is fixed at
    export/wrapper-construction time rather than a traced tensor value."""
    chunk_len = n_window * 2
    leave = n % chunk_len
    feat_lengths = (leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (n // chunk_len) * 13


class AudioEncoderStatic(torch.nn.Module):
    """input_features: (1, num_mel_bins, N_FRAMES) -> (out_len, output_dim).

    Reimplements Qwen3OmniMoeAudioEncoder.forward with N_FRAMES fixed at
    construction time, so every chunk count / cu_seqlens boundary is a plain
    Python int baked into the graph instead of a traced/data-dependent tensor
    value -- torch.export's symbolic shape tracer can't prove facts like
    "chunk_lengths.shape[0] != 0" about *traced* int64 tensors (see
    chunk_and_pad_features's `.tolist()`-driven `.split()`), even when the value
    is in fact always the same fixed number for every call this graph will ever
    see. Requires N_FRAMES % (n_window*2) == 0 (true for the 3000-frame/30s
    window we use, and for n_window=50 that's the model's own hidden state size
    limit anyway -- max_source_positions=1500 = 3000/2).

    Reuses the real submodules (conv2d1/2/3, transformer layers, ln_post,
    proj1/proj2, positional_embedding) for all the actual math -- only the
    dynamic bookkeeping around them is replaced.
    """

    def __init__(self, thinker_model, n_frames: int):
        super().__init__()
        at = thinker_model.audio_tower
        self.conv2d1, self.conv2d2, self.conv2d3 = at.conv2d1, at.conv2d2, at.conv2d3
        self.conv_out = at.conv_out
        self.positional_embedding = at.positional_embedding
        self.layers = at.layers
        self.ln_post = at.ln_post
        self.proj1, self.proj2, self.act = at.proj1, at.proj2, at.act
        self.n_window = at.config.n_window
        self.conv_chunksize = at.conv_chunksize

        n_window2 = self.n_window * 2
        assert n_frames % n_window2 == 0, "N_FRAMES must be an exact multiple of n_window*2"
        self.n_frames = n_frames
        self.num_chunks = n_frames // n_window2  # e.g. 3000/100 = 30, all equal length, no padding
        self.chunk_len = n_window2

        out_len_per_chunk = _feat_extract_out_len(n_window2, self.n_window)  # e.g. 13
        total_out_len = self.num_chunks * out_len_per_chunk  # e.g. 390
        self.out_len = total_out_len

        n_window_ratio = at.n_window_infer // n_window2
        window_aftercnn = out_len_per_chunk * n_window_ratio
        cu = [0]
        remaining = total_out_len
        while remaining > 0:
            step = min(window_aftercnn, remaining)
            cu.append(cu[-1] + step)
            remaining -= step
        self.window_sizes = [cu[i + 1] - cu[i] for i in range(len(cu) - 1)]  # plain Python ints

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        feats = input_features[0]  # (1, mel_bins, n_frames) -> (mel_bins, n_frames)
        mel_bins = feats.shape[0]
        # (mel_bins, n_frames) -> (num_chunks, mel_bins, chunk_len); every chunk is
        # exactly chunk_len long (n_frames is an exact multiple), so this is a pure
        # reshape/transpose -- equivalent to chunk_and_pad_features with no padding.
        padded_feature = feats.transpose(0, 1).reshape(self.num_chunks, self.chunk_len, mel_bins).transpose(1, 2)
        padded_feature = padded_feature.unsqueeze(1).to(dtype=self.conv2d1.weight.dtype)  # (chunks, 1, mel, time)

        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            padded_embed = torch.nn.functional.gelu(self.conv2d1(chunk))
            padded_embed = torch.nn.functional.gelu(self.conv2d2(padded_embed))
            padded_embed = torch.nn.functional.gelu(self.conv2d3(padded_embed))
            padded_embeds.append(padded_embed)
        padded_embed = torch.cat(padded_embeds, dim=0)

        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

        pos_emb = self.positional_embedding.positional_embedding[: padded_embed.shape[1], :].unsqueeze(0)
        padded_embed = padded_embed + pos_emb.to(padded_embed.dtype)
        hidden_states = padded_embed.reshape(-1, padded_embed.shape[-1])  # (num_chunks*out_len_per_chunk, d_model) == all valid, no masking needed

        for layer in self.layers:
            hidden_states = self._layer_forward(layer, hidden_states)

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return hidden_states

    def _layer_forward(self, layer, hidden_states: torch.Tensor) -> torch.Tensor:
        """Static reimplementation of Qwen3OmniMoeAudioEncoderLayer.forward's
        non-flash branch: windowed (not causal, not masked -- full attention
        *within* each window) self-attention, split by self.window_sizes (a
        plain Python list baked in at construction, not cu_seqlens[1:]-cu_seqlens[:-1]
        computed+`.tolist()`'d from a traced tensor -- same data-dependent-shape
        issue as chunk_and_pad_features, worked around the same way).
        """
        attn = layer.self_attn
        seq_len = hidden_states.shape[0]
        residual = hidden_states
        hs = layer.self_attn_layer_norm(hidden_states)

        q = attn.q_proj(hs).reshape(seq_len, attn.num_heads, -1).transpose(0, 1).unsqueeze(0)
        k = attn.k_proj(hs).reshape(seq_len, attn.num_heads, -1).transpose(0, 1).unsqueeze(0)
        v = attn.v_proj(hs).reshape(seq_len, attn.num_heads, -1).transpose(0, 1).unsqueeze(0)

        outs = []
        for qq, kk, vv in zip(
            torch.split(q, self.window_sizes, dim=2),
            torch.split(k, self.window_sizes, dim=2),
            torch.split(v, self.window_sizes, dim=2),
        ):
            o = torch.nn.functional.scaled_dot_product_attention(qq, kk, vv, scale=attn.scaling)
            outs.append(o.transpose(1, 2))  # (1, heads, win, head_dim) -> (1, win, heads, head_dim)
        attn_out = torch.cat(outs, dim=1).reshape(seq_len, -1)
        attn_out = attn.out_proj(attn_out)

        hidden_states = residual + attn_out
        residual = hidden_states
        hs = layer.final_layer_norm(hidden_states)
        hs = layer.fc2(layer.activation_fn(layer.fc1(hs)))
        return residual + hs


def main():
    import os
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    config = build_config(MODEL_DIR)
    with torch.device("meta"):
        model = Qwen3OmniMoeThinkerForConditionalGeneration(config)
    sd = {}
    with safe_open(f"{MODEL_DIR}/model.safetensors", framework="pt") as f:
        for k in f.keys():
            if k.startswith("thinker.audio_tower."):
                sd[k[len("thinker."):]] = f.get_tensor(k).to(torch.float32)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    assert not unexpected, unexpected
    model.audio_tower.positional_embedding = SinusoidsPositionEmbedding(
        config.audio_config.max_source_positions, config.audio_config.d_model)
    model.eval()

    wrapper = AudioEncoderStatic(model, N_FRAMES)
    wrapper.eval()

    dummy = torch.randn(1, config.audio_config.num_mel_bins, N_FRAMES)
    with torch.no_grad():
        ref_out = wrapper(dummy)
        # Cross-check the static reimplementation against the original dynamic
        # audio_tower.forward() on the exact same (fully-real, no padding) input --
        # they must be numerically identical since only the bookkeeping differs.
        orig_out = model.audio_tower(
            dummy[0], feature_lens=torch.full((1,), N_FRAMES, dtype=torch.long), return_dict=True
        ).last_hidden_state
    diff = (ref_out - orig_out).abs()
    print("static-vs-dynamic reimpl max abs diff:", diff.max().item(), "mean:", diff.mean().item())
    assert diff.max().item() < 1e-3, "static reimplementation diverges from the original module"
    print("PyTorch output shape:", ref_out.shape, "expected (390, %d)" % config.audio_config.output_dim)
    assert ref_out.shape == (390, config.audio_config.output_dim)

    print(f"Exporting to {OUT_PATH} ...")
    torch.onnx.export(
        wrapper,
        (dummy,),
        OUT_PATH,
        input_names=["input_features"],
        output_names=["audio_embeds"],
        dynamo=True,
        external_data=False,
        opset_version=18,
    )
    print("Export done.")

    # Verify with onnxruntime
    import onnxruntime as ort
    sess = ort.InferenceSession(OUT_PATH, providers=["CPUExecutionProvider"])
    (ort_out,) = sess.run(None, {"input_features": dummy.numpy()})
    import numpy as np
    diff = np.abs(ort_out - ref_out.detach().numpy())
    print("max abs diff PyTorch vs ONNXRuntime:", diff.max(), "mean:", diff.mean())


if __name__ == "__main__":
    main()
