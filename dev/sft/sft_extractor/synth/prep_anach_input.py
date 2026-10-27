"""Flatten the HF dataset routes into anachronism_filter input files.

anachronism_filter expects a JSON list of {"dataset", "i", "synthetic_q"}. The HF
dataset (zachnorton03/synthetic-pre1930-sft) stores one folder of JSONL shards per
route, with the question either in a "question" field or, for multiturn_qa, as the
user turns of a "conversations" list. Every user turn is emitted as its own record
(id "<doc_index>#<turn>") so each is scored independently; rolling turns back up to
a conversation-level decision is the caller's job.

stem_reasoning is deliberately NOT a default route: masked scoring leaves a residual
there (notation poisons adjacent prose), so it is handled separately.

    ~/git/nanochat/.venv/bin/python -m synth.prep_anach_input --out synth/output/anach
"""
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF_REPO = "zachnorton03/synthetic-pre1930-sft"
ROUTES = ["knowledge_qa", "reasoning_qa", "narrative_grounded", "narrative_fiction",
          "composition_qa", "how_to_qa", "opinion_qa", "verse_qa", "multiturn_qa",
          "calibration_qa"]


def download(routes, cache):
    from huggingface_hub import snapshot_download
    return snapshot_download(HF_REPO, repo_type="dataset", token=False,
                             allow_patterns=[f"{r}/*" for r in routes], local_dir=str(cache))


def _rows(path):
    txt = path.read_text().strip()
    try:
        d = json.loads(txt)
        return d if isinstance(d, list) else [d]
    except json.JSONDecodeError:
        return [json.loads(l) for l in txt.splitlines() if l.strip()]


def route_records(hfds, route):
    out = []
    for f in sorted(Path(hfds, route).glob("*")):
        if f.suffix not in (".json", ".jsonl"):
            continue
        for row in _rows(f):
            doc = row.get("doc_index")
            if row.get("question"):
                out.append({"dataset": route, "i": str(doc), "synthetic_q": row["question"],
                            "year": row.get("year")})
                continue
            turn = 0
            for c in row.get("conversations") or []:
                if c.get("role") == "user" and c.get("content"):
                    out.append({"dataset": route, "i": f"{doc}#{turn}",
                                "synthetic_q": c["content"], "year": row.get("year")})
                    turn += 1
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--routes", nargs="*", default=ROUTES)
    p.add_argument("--cache", default=str(ROOT / "synth" / "output" / "hfds"))
    p.add_argument("--out", default=str(ROOT / "synth" / "output" / "anach"))
    p.add_argument("--limit", type=int, default=0, help="cap records per route (0 = all)")
    args = p.parse_args()

    hfds = download(args.routes, args.cache)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    total = 0
    for route in args.routes:
        recs = route_records(hfds, route)
        if args.limit:
            recs = recs[:args.limit]
        (outdir / f"{route}.json").write_text(json.dumps(recs, ensure_ascii=False))
        print(f"{route:20s} {len(recs):>7d}")
        total += len(recs)
    print(f"{'TOTAL':20s} {total:>7d} -> {outdir}")


if __name__ == "__main__":
    main()
