"""Explode pairs.jsonl into aligned (text, keys) for embedding.

pairs.jsonl has one line per answer with several question variants; the encoder
embeds a flat list of texts. This writes:
  <out>.txt        one question per line (fed to talkie_encoder)
  <out>.keys.jsonl same order: {"row","id","type","variant"} per line

emb.npy[k] then corresponds 1:1 to line k of both files, so every vector joins
back to its (id, type, variant) with no reliance on sort order.

    python -m encoder.prepare_embed pairs.jsonl --out questions
"""
import json
import argparse
from pathlib import Path

VARIANTS = ("q_vintage", "q_modern", "q_authentic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", help="pairs.jsonl")
    ap.add_argument("--out", required=True, help="output stem (writes <out>.txt and <out>.keys.jsonl)")
    args = ap.parse_args()

    txt = Path(args.out + ".txt").open("w")
    keys = Path(args.out + ".keys.jsonl").open("w")
    n = 0
    for row, line in enumerate(Path(args.pairs).read_text().splitlines()):
        if not line.strip():
            continue
        r = json.loads(line)
        for v in VARIANTS:
            q = r.get(v)
            if not q:
                continue
            txt.write(" ".join(q.split()) + "\n")  # one clean line, no embedded newlines
            keys.write(json.dumps({"row": row, "id": r["id"], "type": r["type"],
                                   "variant": v.replace("q_", "")}) + "\n")
            n += 1
    txt.close(); keys.close()
    print(f"exploded {n} question instances -> {args.out}.txt / {args.out}.keys.jsonl")


if __name__ == "__main__":
    main()
