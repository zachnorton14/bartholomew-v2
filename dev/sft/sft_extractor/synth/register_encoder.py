"""Model-agnostic text encoder: any causal/base LM -> one vector per text.

Loads a Hugging Face model by id or path, runs text through it, and pools the
hidden states into a single embedding. Backbone is swappable via --model, so the
same tool works with whatever vintage/period LM a user runs.

    python -m synth.register_encoder --model openai-community/gpt2 \
        --texts file.txt --out emb.npy

Run under an interpreter with torch+transformers (e.g. nanochat's venv):
    ~/git/nanochat/.venv/bin/python -m synth.register_encoder ...
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class Encoder:
    def __init__(self, model, layer=-1, pooling="mean", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModel.from_pretrained(model, output_hidden_states=True).to(self.device).eval()
        self.layer = layer
        self.pooling = pooling

    @torch.no_grad()
    def encode(self, texts, batch_size=32, max_len=128):
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tok(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_len).to(self.device)
            hs = self.model(**enc).hidden_states[self.layer]      # (B, T, H)
            mask = enc.attention_mask.unsqueeze(-1).float()       # (B, T, 1)
            if self.pooling == "mean":
                vec = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)
            elif self.pooling == "last":
                idx = enc.attention_mask.sum(1) - 1               # last real token
                vec = hs[torch.arange(hs.size(0)), idx]
            else:
                raise ValueError(f"unknown pooling {self.pooling}")
            out.append(vec.float().cpu().numpy())
        return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--texts", required=True, help="file with one text per line")
    ap.add_argument("--out", required=True, help="output .npy of shape (N, H)")
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--pooling", choices=["mean", "last"], default="mean")
    args = ap.parse_args()

    texts = [l for l in Path(args.texts).read_text().splitlines() if l.strip()]
    enc = Encoder(args.model, layer=args.layer, pooling=args.pooling)
    emb = enc.encode(texts)
    np.save(args.out, emb)
    print(f"encoded {len(texts)} texts -> {emb.shape} on {enc.device} -> {args.out}")


if __name__ == "__main__":
    main()
