"""Standalone bidirectional transformer encoder (MLM).

Self-contained and model-agnostic: no dependency on any specific base model's
architecture. Size scales via config (dim / depth / heads), so you pick the model
to fit your corpus and time budget. Trained with masked-language-modeling; at
inference `encode()` returns one pooled vector per sequence.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EncoderConfig:
    vocab_size: int = 32768
    dim: int = 512
    depth: int = 8
    n_heads: int = 8
    max_seq_len: int = 512
    mlp_ratio: int = 4
    dropout: float = 0.0


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.ln1 = nn.LayerNorm(cfg.dim)
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim)
        self.proj = nn.Linear(cfg.dim, cfg.dim)
        self.ln2 = nn.LayerNorm(cfg.dim)
        hidden = cfg.mlp_ratio * cfg.dim
        self.mlp = nn.Sequential(nn.Linear(cfg.dim, hidden), nn.GELU(), nn.Linear(hidden, cfg.dim))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, key_padding_mask):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # bidirectional: no causal mask. Mask out padding keys only.
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask[:, None, None, :]  # (B,1,1,T) True = keep
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.drop(self.proj(y))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.depth))
        self.ln_f = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def backbone(self, idx, attention_mask=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None])
        kpm = attention_mask.bool() if attention_mask is not None else None
        for blk in self.blocks:
            x = blk(x, kpm)
        return self.ln_f(x)

    def forward(self, idx, labels=None, attention_mask=None):
        h = self.backbone(idx, attention_mask)
        logits = self.head(h)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def encode(self, idx, attention_mask=None, pooling="mean"):
        h = self.backbone(idx, attention_mask)
        if pooling == "mean":
            if attention_mask is None:
                return h.mean(1)
            m = attention_mask[..., None].float()
            return (h * m).sum(1) / m.sum(1).clamp(min=1)
        if pooling == "cls":
            return h[:, 0]
        raise ValueError(pooling)
