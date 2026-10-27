"""Build every robustness route and push it to its own HF dataset.

Kept separate from the graded synth dataset on purpose: those rows carry answers
lifted verbatim from period prose and a grade from the judge. These are constructed
-- composed from mined clauses, never graded -- and mixing them in would muddy the
provenance guarantee the curriculum relies on.

Layout mirrors the synth repo so the nanochat loader is a near-copy:
    rows/<route>/part-00000.jsonl

    python -m synth.robustness.build --dry-run     # counts only, no upload
    python -m synth.robustness.build
"""

import argparse
import json

from synth import hf_push
from synth.robustness import conversation, era, multiturn, typos, unparseable

HF_REPO = "zachnorton03/vintage-sft-robustness"

# (module, default row count). Counts are the dial for how much of the mixture
# this becomes; the curriculum spec then multiplies by epochs on top.
ROUTES = (
    (conversation, 3500),
    (unparseable, 2000),
    (typos, 1200),
    (era, 220),
    (multiturn, None),   # takes whatever survives filtering
)


def build_all(seed=1930, counts=None):
    out = {}
    for module, default in ROUTES:
        n = (counts or {}).get(module.ROUTE, default)
        out[module.ROUTE] = module.build_rows(n, seed) if n else module.build_rows()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1930)
    ap.add_argument("--dry-run", action="store_true", help="build and report, do not upload")
    ap.add_argument("--repo", default=HF_REPO)
    args = ap.parse_args()

    built = build_all(args.seed)
    total = sum(len(r) for r in built.values())
    for route, rows in built.items():
        # multiturn rows carry a `conversations` list, not a single answer
        if rows and "answer" in rows[0]:
            uniq = len({r["answer"] for r in rows})
            print(f"  {route:20} {len(rows):5} rows  {uniq:4} distinct answers")
        else:
            turns = sum(len(r["conversations"]) for r in rows)
            print(f"  {route:20} {len(rows):5} rows  {turns:4} turns")
    print(f"  {'TOTAL':18} {total:5} rows")

    if args.dry_run:
        print("\ndry run; nothing uploaded")
        return

    for route, rows in built.items():
        replaced, shards, n = hf_push.write_sharded(
            f"rows/{route}", rows, shard_size=2000, repo=args.repo
        )
        print(f"pushed {route}: {n} rows in {shards} shard(s), replaced {replaced}")
    print(f"\nhttps://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
