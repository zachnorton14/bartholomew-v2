"""Train the bidirectional MLM encoder on a corpus, up to a token budget.

Attach any Hugging Face text dataset (streamed, no full download) or a local text
file, choose the tokenizer, set the model size, and give a token budget; training
stops once that many tokens have been consumed.

    python -m encoder.train \
        --dataset wikitext --dataset-config wikitext-103-raw-v1 --split train \
        --tokenizer bert-base-uncased \
        --tokens 500_000_000 --dim 512 --depth 8 --heads 8 --out ckpt/enc.pt

Local file instead of HF:  --dataset ./corpus.txt
Warm start (only if arch matches):  --init-from ckpt/prev.pt

Needs torch, transformers, datasets (e.g. nanochat's venv + `pip install datasets`).
"""
import math
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from encoder.model import Encoder, EncoderConfig
from encoder.tokenizer import train_tokenizer


def text_stream(args):
    """Yield raw documents from a local .txt (looped) or a streamed HF dataset."""
    p = Path(args.dataset)
    if p.exists():
        while True:
            for line in p.read_text().splitlines():
                if line.strip():
                    yield line
    else:
        from datasets import load_dataset
        ds = load_dataset(args.dataset, args.dataset_config, split=args.split, streaming=True)
        for ex in ds:
            text = ex.get(args.text_column)
            if text:
                yield text


def build_tokenizer(args):
    """Load an HF tokenizer if given, else train one on the corpus (nanochat recipe)."""
    if args.tokenizer:
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        if tok.mask_token is None:
            tok.add_special_tokens({"mask_token": "[MASK]"})
        return tok
    print(f"training tokenizer from corpus: vocab {args.vocab_size}, {args.tokenizer_docs} docs...")
    docs, stream = [], text_stream(args)
    for _ in range(args.tokenizer_docs):
        try:
            docs.append(next(stream))
        except StopIteration:
            break
    tok = train_tokenizer(iter(docs), args.vocab_size)
    tok.save_pretrained(str(Path(args.out).parent / "tokenizer"))
    return tok


def token_stream(args, tok):
    """Yield token ids one document at a time."""
    for text in text_stream(args):
        yield from tok(text, add_special_tokens=False)["input_ids"]


def batches(args, tok, device):
    """Pack the token stream into (B, T) blocks and yield them forever."""
    T, B = args.seq_len, args.batch_size
    buf, block = [], T * B
    for tid in token_stream(args, tok):
        buf.append(tid)
        if len(buf) >= block:
            x = torch.tensor(buf[:block], dtype=torch.long, device=device).view(B, T)
            buf = buf[block:]
            yield x


def mask_tokens(x, tok, mask_id, prob):
    """BERT-style masking: labels = originals at masked spots, -100 elsewhere."""
    labels = x.clone()
    probs = torch.full(x.shape, prob, device=x.device)
    special = torch.tensor(tok.all_special_ids, device=x.device)
    probs[torch.isin(x, special)] = 0.0
    masked = torch.bernoulli(probs).bool()
    labels[~masked] = -100
    # 80% [MASK], 10% random, 10% unchanged
    r = torch.rand(x.shape, device=x.device)
    x = x.clone()
    x[masked & (r < 0.8)] = mask_id
    rand = masked & (r >= 0.9)
    x[rand] = torch.randint(len(tok), x.shape, device=x.device)[rand]
    return x, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="HF dataset id or local .txt path")
    ap.add_argument("--dataset-config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer id/path; omit to TRAIN one from the corpus")
    ap.add_argument("--vocab-size", type=int, default=32768, help="vocab size when training tokenizer from corpus")
    ap.add_argument("--tokenizer-docs", type=int, default=50000, help="docs sampled to train the tokenizer")
    ap.add_argument("--tokens", type=lambda s: int(float(s.replace("_", ""))), required=True,
                    help="token budget; training stops here")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.02)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--init-from", default=None, help="warm-start checkpoint (arch must match)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tok = build_tokenizer(args)
    mask_id = tok.mask_token_id

    cfg = EncoderConfig(vocab_size=len(tok), dim=args.dim, depth=args.depth,
                        n_heads=args.heads, max_seq_len=args.seq_len)
    model = Encoder(cfg).to(device)
    if args.init_from:
        sd = torch.load(args.init_from, map_location=device)
        model.load_state_dict(sd["model"] if "model" in sd else sd)
        print(f"warm-started from {args.init_from}")
    params = sum(p.numel() for p in model.parameters())
    print(f"device {device}  params {params/1e6:.1f}M  vocab {len(tok)}  budget {args.tokens:,} tokens")

    total_steps = max(1, args.tokens // (args.seq_len * args.batch_size))
    warmup = int(args.warmup_frac * total_steps)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    def lr_at(step):
        if step < warmup:
            return args.lr * step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.train()
    seen, step = 0, 0
    for x in batches(args, tok, device):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        xin, labels = mask_tokens(x, tok, mask_id, args.mask_prob)
        _, loss = model(xin, labels=labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        seen += x.numel()
        step += 1
        if step % args.log_every == 0:
            print(f"step {step}/{total_steps}  tokens {seen:,}  loss {loss.item():.3f}  lr {lr_at(step):.2e}", flush=True)
        if step % args.save_every == 0:
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "tokenizer": args.tokenizer}, args.out)
        if seen >= args.tokens:
            break

    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "tokenizer": args.tokenizer}, args.out)
    print(f"done: {seen:,} tokens, {step} steps -> {args.out}")


if __name__ == "__main__":
    main()
