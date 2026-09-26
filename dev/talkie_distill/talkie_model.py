"""Vendored + patched Talkie 13B model (from talkie-lm/talkie src/talkie/model.py @ 35317ba, Apache-2.0).

Parameter names and math are identical to upstream, so `rl-refined.pt` loads with strict=True.
Changes vs upstream:
  * forward() returns the final hidden state for ALL positions (upstream: last-position logits only);
    `logits()` applies the (gained) lm_head separately so callers can chunk over time.
  * optional per-row position ids and a boolean key-validity mask, for left-padded batches;
  * a static KV cache (`KVCache`) so decoding is one token per step instead of re-running the prefix.
Without a cache and without padding, the attention path is exactly upstream's (SDPA, is_causal=True).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65536
    n_layer: int = 40
    n_head: int = 40
    n_embd: int = 5120
    head_dim: int = 128


def apply_rotary_emb(x, cos, sin):
    # x: (B, T, H, D); cos/sin broadcastable to (B, T, 1, D/2)
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


class HeadGain(nn.Module):
    def __init__(self, n_head):
        super().__init__()
        self.head_g = nn.Parameter(torch.ones([n_head]))

    def forward(self, x):
        return x * self.head_g.type_as(x).view(1, 1, -1, 1)


class WeightGain(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_g = nn.Parameter(torch.ones(1))

    def forward(self, w):
        return w * self.w_g.type_as(w)


class ActGain(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.a_g = nn.Parameter(torch.ones(1) * init_value)

    def forward(self, x):
        return x * self.a_g.type_as(x)


class KVCache:
    """Static cache: k/v of shape (n_layer, B, H, max_len, D). All rows share one write pointer `pos`
    (batches are left-padded so every row's newest token sits at the same column).
    `key_valid[b, j]` is False for left-padding slots."""

    def __init__(self, n_layer, batch_size, n_head, max_len, head_dim, device, dtype=torch.bfloat16):
        self.k = torch.zeros(n_layer, batch_size, n_head, max_len, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.key_valid = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
        self.max_len = max_len
        self.pos = 0
        self.has_padding = False

    def nbytes(self):
        return 2 * self.k.numel() * self.k.element_size()


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int = 0):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.layer_idx = layer_idx
        n = config.n_embd
        self.attn_query = nn.Linear(n, n, bias=False)
        self.attn_key = nn.Linear(n, n, bias=False)
        self.attn_value = nn.Linear(n, n, bias=False)
        self.attn_resid = nn.Linear(n, n, bias=False)
        self.head_gain = HeadGain(config.n_head)

    def forward(self, x, cos_sin, kv_cache: KVCache | None = None, attn_mask=None):
        B, T, _ = x.size()
        q = self.attn_query(x).view(B, T, self.n_head, self.head_dim)
        k = self.attn_key(x).view(B, T, self.n_head, self.head_dim)
        v = self.attn_value(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q = self.head_gain(q)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (B, H, T, D)

        if kv_cache is None:
            if attn_mask is None:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            p = kv_cache.pos
            kc, vc = kv_cache.k[self.layer_idx], kv_cache.v[self.layer_idx]
            kc[:, :, p:p + T] = k
            vc[:, :, p:p + T] = v
            kk, vv = kc[:, :, :p + T], vc[:, :, :p + T]
            if attn_mask is None and p == 0:
                y = F.scaled_dot_product_attention(q, kk, vv, is_causal=True)
            elif attn_mask is None and T == 1:
                y = F.scaled_dot_product_attention(q, kk, vv)
            else:
                y = F.scaled_dot_product_attention(q, kk, vv, attn_mask=attn_mask)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.attn_resid(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        n = config.n_embd
        n_mlp = int(round(((8 / 3) * n) / 128) * 128)
        self.mlp_gate = nn.Linear(n, n_mlp, bias=False)
        self.mlp_linear = nn.Linear(n, n_mlp, bias=False)
        self.mlp_resid = nn.Linear(n_mlp, n, bias=False)

    def forward(self, x):
        return self.mlp_resid(F.silu(self.mlp_gate(x)) * self.mlp_linear(x))


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int = 0):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.attn_gain = ActGain((2 * config.n_layer) ** -0.5)
        self.mlp = MLP(config)
        self.mlp_gain = ActGain((2 * config.n_layer) ** -0.5)
        self.embed_skip = ActGain(0.0)

    def forward(self, e_x, x, cos_sin, kv_cache=None, attn_mask=None):
        x = x + self.attn_gain(self.attn(F.rms_norm(x, (x.shape[-1],)), cos_sin, kv_cache, attn_mask))
        x = x + self.mlp_gain(self.mlp(F.rms_norm(x, (x.shape[-1],))))
        x = x + self.embed_skip(e_x)
        return x


class TalkieModel(nn.Module):
    def __init__(self, config: GPTConfig, max_seq_len: int = 4096):
        super().__init__()
        self.config = config
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
        self.lm_head = nn.Parameter(torch.zeros(config.vocab_size, config.n_embd))
        self.lm_head_gain = WeightGain()
        cos, sin = self.rotary(max_seq_len, config.head_dim, device="cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @staticmethod
    def rotary(seq_len, head_dim, base=1_000_000, device="cpu"):
        ch = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (ch / head_dim))
        freqs = torch.outer(torch.arange(seq_len, dtype=torch.float32, device=device), inv_freq)
        return freqs.cos().bfloat16()[None, :, None, :], freqs.sin().bfloat16()[None, :, None, :]

    def embed_in(self, input_ids):
        x = self.embed(input_ids)
        return F.rms_norm(x, (x.shape[-1],))

    def trunk_out(self, x):
        return F.rms_norm(x, (x.shape[-1],))

    def cos_sin(self, T, start=0, pos_ids=None):
        if pos_ids is None:
            return self.cos[:, start:start + T], self.sin[:, start:start + T]
        return self.cos[0][pos_ids], self.sin[0][pos_ids]  # (B, T, 1, D/2)

    def forward(self, input_ids, pos_ids=None, attn_mask=None, kv_cache: KVCache | None = None):
        """Returns final-norm hidden states (B, T, C) for all positions."""
        B, T = input_ids.shape
        start = kv_cache.pos if kv_cache is not None else 0
        cs = self.cos_sin(T, start, pos_ids)
        x = self.embed_in(input_ids)
        e_x = x
        for block in self.blocks:
            x = block(e_x, x, cs, kv_cache, attn_mask)
        if kv_cache is not None:
            kv_cache.pos += T
        return self.trunk_out(x)

    def lm_weight(self):
        return self.lm_head_gain(self.lm_head)

    def logits(self, h):
        """bf16 hidden -> fp32 logits."""
        return F.linear(h, self.lm_weight()).float()
