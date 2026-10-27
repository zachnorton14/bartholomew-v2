"""Repair what is safely repairable, drop the Q/A pairs that are still corrupt.

Two-stage, because the corruption classes are not equally recoverable:

  REPAIR   line-break hyphenation, via a dictionary gate -- "after- wards" rejoins,
           "rod- and cone-layer" does not (see ocr_corruption.join_hyphen_breaks).
           ~68% of breaks rejoin; nothing is welded together that isn't a real word.

  DROP     long_s, homoglyph, mojibake, broken_fragment. These either cannot be
           repaired without inventing text (a truncated "coun- as" lost its line) or
           could be repaired only by rewriting many words at once, where a single
           miscorrection silently teaches a wrong spelling. Deleting is cheaper than
           being subtly wrong, and the dataset is large.

Both the question and the answer are scanned: a pair is only as good as its worse
half. For multiturn_qa the unit is the whole conversation -- a corrupt assistant turn
drops the conversation, since the remaining turns would have a hole in them.

The complete dataset lives on HF; this writes a cleaned copy locally and never
uploads. Review, then push separately.

    python -m synth.build_clean_dataset --dry-run
    python -m synth.build_clean_dataset --out synth/output/hfds_clean
"""
import json
import argparse
from pathlib import Path

from synth.ocr_corruption import corruption_report, repair_hyphenation, unrepairable_hyphen

ROOT = Path(__file__).resolve().parent.parent
ROUTES = ["knowledge_qa", "reasoning_qa", "narrative_grounded", "narrative_fiction",
          "composition_qa", "how_to_qa", "opinion_qa", "verse_qa", "multiturn_qa",
          "calibration_qa"]


def _rows(path):
    txt = path.read_text(errors="replace").strip()
    try:
        d = json.loads(txt)
        return d if isinstance(d, list) else [d]
    except json.JSONDecodeError:
        return [json.loads(l) for l in txt.splitlines() if l.strip()]


def process(row):
    """(cleaned_row | None, n_joined, {class: count}). None means drop."""
    row = dict(row)
    joined = 0
    reasons = {}

    def handle(text):
        nonlocal joined
        text, st = repair_hyphenation(text)
        joined += sum(st.values())
        if unrepairable_hyphen(text):
            reasons["trailing_hyphen"] = reasons.get("trailing_hyphen", 0) + 1
        for k, v in corruption_report(text).items():
            reasons[k] = reasons.get(k, 0) + v
        return text

    for field in ("question", "answer"):
        if isinstance(row.get(field), str):
            row[field] = handle(row[field])

    if isinstance(row.get("conversations"), list):
        convs = []
        for c in row["conversations"]:
            c = dict(c)
            if isinstance(c.get("content"), str):
                c["content"] = handle(c["content"])
            convs.append(c)
        row["conversations"] = convs

    return (None if reasons else row), joined, reasons


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hfds", default=str(ROOT / "synth" / "output" / "hfds"))
    p.add_argument("--out", default=str(ROOT / "synth" / "output" / "hfds_clean"))
    p.add_argument("--routes", nargs="*", default=ROUTES)
    p.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = p.parse_args()

    tot = kept = dropped = joined_all = 0
    grand = {}
    print(f"{'route':20s} {'rows':>9s} {'kept':>9s} {'dropped':>8s} {'drop%':>7s} {'joined':>7s}  reasons")
    for route in args.routes:
        n = k = d = j = 0
        reasons = {}
        outdir = Path(args.out, route)
        if not args.dry_run:
            outdir.mkdir(parents=True, exist_ok=True)
        for f in sorted(Path(args.hfds, route).glob("*")):
            if f.suffix not in (".json", ".jsonl"):
                continue
            out = []
            for row in _rows(f):
                n += 1
                new, nj, why = process(row)
                j += nj
                if new is None:
                    d += 1
                    for key, v in why.items():
                        reasons[key] = reasons.get(key, 0) + v
                else:
                    k += 1
                    out.append(new)
            if not args.dry_run:
                (outdir / f.name).write_text(
                    "\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n")
        tot += n; kept += k; dropped += d; joined_all += j
        for key, v in reasons.items():
            grand[key] = grand.get(key, 0) + v
        print(f"{route:20s} {n:>9,} {k:>9,} {d:>8,} {d/max(n,1):>6.2%} {j:>7,}  {reasons}")

    print(f"\n{'TOTAL':20s} {tot:>9,} {kept:>9,} {dropped:>8,} {dropped/max(tot,1):>6.2%} {joined_all:>7,}")
    print("drop reasons:")
    for key, v in sorted(grand.items(), key=lambda x: -x[1]):
        print(f"  {key:18s} {v:>8,}")
    print(f"\nhyphen breaks repaired in kept rows: {joined_all:,}")
    if args.dry_run:
        print("\n(dry run - nothing written)")
    else:
        print(f"\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
