"""Anachronism filter: drop the most anachronistic synthetic questions.

Scores each generated question with the vintage-minus-modern bits-per-byte delta
(delta_peak, the validated metric from nanochat's scripts/anachronism_eval.py —
AUC 0.933 on the authored probes) and removes the worst --drop-frac per file.

The scoring models live in the nanochat repo: the vintage model is the pre-1930s
nanochat base checkpoint, the modern reference is a HF causal LM. This script
bootstraps nanochat onto sys.path, so it must run under nanochat's interpreter
(the sft-extractor venv has no torch):

    cd ~/git/sft-extractor
    ~/git/nanochat/.venv/bin/python -m synth.anachronism_filter output/synth/questions_*.json

Input files are matched-pair JSON from `run.py synth-questions`
(output/synth/questions_<style>.json). For each input this writes, next to it:

  questions_<style>_filtered.json   kept records, each annotated with
                                    "anach_delta_peak"; worst --drop-frac removed
  questions_<style>_dropped.json    the removed records, same annotation, for review

Only the synthetic question is scored — the authentic question is genuine period
text by construction and is passed through untouched.
"""
import sys
import json
import argparse
from pathlib import Path

NANOCHAT_ROOT = Path.home() / "git" / "nanochat"
DEFAULT_EXPERIMENT = NANOCHAT_ROOT / "hf_download" / "experiments" / "think-d12-1ep-65sh-r30"
sys.path.insert(0, str(NANOCHAT_ROOT))

from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type  # noqa: E402
from nanochat.checkpoint_manager import build_model, find_last_step  # noqa: E402
from nanochat.tokenizer import get_token_bytes  # noqa: E402
from nanochat.loss_eval import score_word_bits  # noqa: E402
from scripts.anachronism_eval import _delta_metrics  # noqa: E402


def _scorable(tok, texts, max_tokens):
    """True per text if it survives tokenization with >= 2 target tokens.

    score_word_bits guards len(ids) < 2, but ids includes the prepended BOS, so a
    single-token text (e.g. the literal question "NO") reaches the model with T == 1
    and trips gpt.forward's `assert T > 1`. One such row would abort a whole pass, so
    they are identified up front and scored as None instead.
    """
    bos = tok.get_bos_token_id()
    return [len(tok.encode(t, prepend=bos)[:max_tokens]) >= 3 for t in texts]


def filter_file(path, model, tokenizer, token_bytes, modern, modern_tok, modern_token_bytes,
                device, peak_k, max_tokens, drop_frac, mask=False):
    records = json.loads(Path(path).read_text())
    texts = [r["synthetic_q"] for r in records]

    ok_v = _scorable(tokenizer, texts, max_tokens)
    ok_m = _scorable(modern_tok, texts, max_tokens)
    keep_idx = [i for i in range(len(texts)) if ok_v[i] and ok_m[i]]
    if len(keep_idx) < len(texts):
        print0(f"{Path(path).stem}: {len(texts) - len(keep_idx)} degenerate text(s) unscorable, "
               f"passed through with a null score")
    sub = [texts[i] for i in keep_idx]

    sv = score_word_bits(model, tokenizer, sub, device, token_bytes, max_tokens=max_tokens)
    sm = score_word_bits(modern, modern_tok, sub, device, modern_token_bytes, max_tokens=max_tokens)
    empty = {"words": [], "bits": [], "bytes": []}
    wv, wm = [dict(empty) for _ in texts], [dict(empty) for _ in texts]
    for j, i in enumerate(keep_idx):
        wv[i], wm[i] = sv[j], sm[j]

    if mask:
        # drop math/scientific-notation words so delta_peak scores prose register only
        # (both models re-aggregate to the same text words, so kept indices align)
        from synth.notation_mask import mask_wordbits
        wv = [mask_wordbits(v) for v in wv]
        wm = [mask_wordbits(m) for m in wm]

    scored = []
    for r, v, m in zip(records, wv, wm):
        d = _delta_metrics(v, m, peak_k)
        # unscorable (empty after tokenization) -> keep, but mark score as None
        scored.append((d[1] if d else None, r))

    n_drop = int(len(scored) * drop_frac)
    rankable = sorted((s for s, _ in scored if s is not None), reverse=True)
    threshold = rankable[n_drop - 1] if n_drop > 0 and rankable else float("inf")

    kept, dropped = [], []
    n_over = 0
    for s, r in scored:
        out = dict(r)
        out["anach_delta_peak"] = round(s, 4) if s is not None else None
        # ties at the threshold: drop only the first n_drop in score order
        if s is not None and s >= threshold and n_over < n_drop:
            n_over += 1
            dropped.append(out)
        else:
            kept.append(out)

    stem = Path(path).stem
    kept_path = Path(path).with_name(f"{stem}_filtered.json")
    drop_path = Path(path).with_name(f"{stem}_dropped.json")
    kept_path.write_text(json.dumps(kept, indent=2, ensure_ascii=False))
    drop_path.write_text(json.dumps(dropped, indent=2, ensure_ascii=False))

    print0(f"\n{stem}: {len(records)} scored, {len(dropped)} dropped "
           f"(threshold delta_peak >= {threshold:.3f}) -> {kept_path.name}")
    for r in sorted(dropped, key=lambda r: -r["anach_delta_peak"])[:5]:
        print0(f"  {r['anach_delta_peak']:6.3f}  [{r['dataset']}--{r['i']}] {r['synthetic_q'][:90]}")
    return len(records), len(dropped)


def main():
    parser = argparse.ArgumentParser(description="Filter synthetic questions by anachronism delta")
    parser.add_argument("inputs", nargs="+", help="matched-pair JSON files (questions_<style>.json)")
    parser.add_argument("--checkpoint-dir", type=str, default=str(DEFAULT_EXPERIMENT / "base_checkpoints"),
                        help="vintage BASE (pretrain-only) checkpoint directory")
    parser.add_argument("--tokenizer-dir", type=str, default=str(DEFAULT_EXPERIMENT / "tokenizer"),
                        help="vintage tokenizer directory")
    parser.add_argument("--step", type=int, default=None, help="model step to load (default = last)")
    parser.add_argument("--modern-hf-path", type=str, default="openai-community/gpt2",
                        help="HF causal LM for the delta reference")
    parser.add_argument("--drop-frac", type=float, default=0.05, help="fraction to drop per file")
    parser.add_argument("--peak-k", type=int, default=3, help="top-k words averaged for the peak metric")
    parser.add_argument("--max-tokens", type=int, default=512, help="truncate questions to this many tokens")
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
    parser.add_argument("--mask", action="store_true",
                        help="mask math/scientific notation words before scoring (recommended for STEM)")
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, ddp_rank, _, _, device = compute_init(device_type)

    step = args.step if args.step is not None else find_last_step(args.checkpoint_dir)
    model, tokenizer, meta = build_model(
        args.checkpoint_dir, step, device, phase="eval", tokenizer_dir=args.tokenizer_dir,
    )
    token_bytes = get_token_bytes(device=device, tokenizer_dir=args.tokenizer_dir)
    print0(f"Loaded vintage base model at step {meta['step']} on {device}")

    from scripts.base_eval import load_hf_model, get_hf_token_bytes
    modern, modern_tok = load_hf_model(args.modern_hf_path, device)
    modern_token_bytes = get_hf_token_bytes(modern_tok, device=device)

    total = total_dropped = 0
    for path in args.inputs:
        n, d = filter_file(path, model, tokenizer, token_bytes, modern, modern_tok,
                           modern_token_bytes, device, args.peak_k, args.max_tokens, args.drop_frac,
                           mask=args.mask)
        total += n
        total_dropped += d

    print0(f"\nTotal: {total} scored, {total_dropped} dropped ({total_dropped/total:.1%})")
    compute_cleanup()


if __name__ == "__main__":
    main()
