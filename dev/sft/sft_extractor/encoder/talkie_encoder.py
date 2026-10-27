"""Sentence embeddings from the Talkie 1930 base model (no repo fork).

Talkie's forward returns only last-position logits, so we reimplement its forward
pass verbatim EXCEPT we stop before the `x[:, -1, :]` slice and return the full
`[B, T, H]` hidden states (post-final-RMSNorm). It reuses the loaded model's own
submodules (`embed`, `blocks`, `cos`, `sin`), so nothing is forked — we just borrow
the module and run a different readout.

Embedding is prefill-only (one forward per text, no generation) and each text is
encoded once, so a 200k-question pass is a single ~15-min job on an A100, then the
cached vectors are reused for all downstream scoring.

Requires the `talkie` package + weights (>=28 GB VRAM, bf16). Run on the A100:
    pip install git+https://github.com/talkie-lm/talkie
    python -m encoder.talkie_encoder --texts questions.txt --out emb.npy
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def hidden_states(model, input_ids):
    """Talkie's forward up to (but not including) the last-position slice + head.
    Returns post-final-norm hidden states, shape [B, T, H]."""
    _, seq_len = input_ids.shape
    cos_sin = model.cos[:, :seq_len], model.sin[:, :seq_len]
    x = model.embed(input_ids)
    x = F.rms_norm(x, (x.shape[-1],))
    e_x = x
    for block in model.blocks:
        x = block(e_x, x, cos_sin)
    return F.rms_norm(x, (x.shape[-1],))  # [B, T, H]


class TalkieEncoder:
    def __init__(self, model_name="talkie-1930-13b-base", device=None):
        from talkie import Talkie  # deferred: heavy import, only needed at load
        self.t = Talkie(model_name)
        self.model = self.t.model
        self.tok = self.t.tokenizer
        self.device = device or next(self.model.parameters()).device

    def _encode_ids(self, text, max_len):
        ids = self.tok.encode(text, allowed_special="all")[:max_len]
        return ids or [0]

    @torch.no_grad()
    def encode(self, texts, batch_size=64, max_len=64, pooling="mean"):
        vecs = []
        for i in range(0, len(texts), batch_size):
            chunk = [self._encode_ids(t, max_len) for t in texts[i:i + batch_size]]
            T = max(len(ids) for ids in chunk)
            # right-pad with 0; causal attention means real tokens never see the pads,
            # so their representations are identical to the unpadded run.
            x = torch.zeros(len(chunk), T, dtype=torch.long, device=self.device)
            mask = torch.zeros(len(chunk), T, dtype=torch.bool, device=self.device)
            for r, ids in enumerate(chunk):
                x[r, :len(ids)] = torch.tensor(ids, device=self.device)
                mask[r, :len(ids)] = True
            h = hidden_states(self.model, x).float()  # [B, T, H]
            if pooling == "mean":
                m = mask[..., None].float()
                v = (h * m).sum(1) / m.sum(1).clamp(min=1)
            elif pooling == "last":
                idx = mask.sum(1) - 1
                v = h[torch.arange(h.size(0)), idx]
            else:
                raise ValueError(pooling)
            vecs.append(v.cpu().numpy().astype(np.float16))
        return np.concatenate(vecs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="talkie-1930-13b-base")
    ap.add_argument("--texts", required=True, help="one text per line")
    ap.add_argument("--out", required=True, help="output .npy (N, H) float16")
    ap.add_argument("--pooling", choices=["mean", "last"], default="mean")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=64)
    args = ap.parse_args()

    texts = [l for l in Path(args.texts).read_text().splitlines() if l.strip()]
    enc = TalkieEncoder(args.model)
    emb = enc.encode(texts, batch_size=args.batch_size, max_len=args.max_len, pooling=args.pooling)
    np.save(args.out, emb)
    print(f"encoded {len(texts)} texts -> {emb.shape} -> {args.out}")


if __name__ == "__main__":
    main()
