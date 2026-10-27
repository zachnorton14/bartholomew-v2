"""Normalize dataset text to the orthography the vintage tokenizer actually knows.

The pre-1930s base corpus was ASCII-normalized Gutenberg-style before the tokenizer
was trained, so the vintage vocab has NO token containing the characters our generated
questions inherited from the source scans. Measured against
clean1930s-d24-r12-ctx4096-sssl-fulltok-v1/tokenizer (32768 tokens):

    '--' 1 token      '—' em dash      0 vocab tokens   (24,194 occurrences in data)
    '"'  1 token      '“' '”'          0 vocab tokens
    'ae' 1 token      'æ'              0 vocab tokens   ("Cæsar" = 5 tokens, "Caesar" = 1)
    'oe' 1 token      'œ' 'ſ'          0 vocab tokens   ("ſhall" = 3 tokens, "shall" = 1)

So these are not period authenticity we would be destroying — they are orthography the
model's own corpus already normalized away, and leaving them in makes the SFT text
inconsistent with pretraining while costing 3-5x the tokens per affected word.

Characters the vocab DOES cover are deliberately preserved: é (25 tokens), ö, ç, â, à,
ê, ô, ä, í, ó, ï, ñ, °, £, §, ×, ·, †, and Greek. Those are real period orthography the
model was trained on; normalizing them would be the destructive kind of cleaning.

Mixed-script confusables (a Latin word carrying a stray Greek/Cyrillic homoglyph, e.g.
"vοlume" with U+03BF omicron) are repaired per the Unicode UTS #39 approach: only when
the word is otherwise Latin, so genuine Greek and Cyrillic quotations are left alone.

    python -m synth.text_normalize --report          # impact only, writes nothing
    python -m synth.text_normalize --apply --out DIR # write normalized copies
"""
import json
import argparse
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROUTES = ["knowledge_qa", "reasoning_qa", "narrative_grounded", "narrative_fiction",
          "composition_qa", "how_to_qa", "opinion_qa", "verse_qa", "multiturn_qa",
          "calibration_qa"]
TEXT_FIELDS = ("question", "answer", "content")

# Vocab-absent -> vocab-present. Every replacement below was checked against the
# tokenizer: the source char has 0 vocab tokens, the target is 1-2 tokens.
CHAR_MAP = {
    # typography
    "—": "--",  "―": "--", "–": "-",   "‒": "-",  "−": "-",
    "“": '"',   "”": '"',  "„": '"',   "‟": '"',
    "‘": "'",   "’": "'",  "‚": "'",   "‛": "'",
    "′": "'",   "″": '"',  "…": "...", " ": " ",
    # long s and ligatures
    "ſ": "s",
    "æ": "ae", "Æ": "Ae", "œ": "oe", "Œ": "Oe",
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi",
    "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
    # super/subscripts
    "¹": "1", "²": "2", "³": "3",
    "⁰": "0", "⁴": "4", "⁵": "5", "⁶": "6",
    "⁷": "7", "⁸": "8", "⁹": "9",
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    # fractions
    "½": "1/2", "¼": "1/4", "¾": "3/4",
    "⅓": "1/3", "⅔": "2/3", "⅕": "1/5", "⅘": "3/5",
    "⅛": "1/8", "⅜": "3/8", "⅝": "5/8", "⅞": "7/8",
    "⁄": "/",
}

# Confusables repaired only inside otherwise-Latin words (UTS #39 mixed-script rule).
CONFUSABLE = {
    "ο": "o", "ν": "v", "α": "a", "ρ": "p", "τ": "t",
    "ι": "i", "κ": "k", "χ": "x", "ε": "e", "υ": "u",
    "А": "A", "О": "O", "Е": "E", "Р": "P", "С": "C",
    "Т": "T", "Х": "X", "М": "M", "Н": "H", "К": "K",
    "В": "B", "а": "a", "о": "o", "е": "e", "р": "p",
    "с": "c", "х": "x", "у": "y", "и": "u",
}


def _script(ch):
    """Coarse script bucket for the mixed-script test."""
    if ch.isascii():
        return "Latin" if ch.isalpha() else None
    if not ch.isalpha():
        return None
    name = unicodedata.name(ch, "")
    for s in ("GREEK", "CYRILLIC", "HEBREW", "ARABIC", "DEVANAGARI"):
        if name.startswith(s):
            return s.capitalize()
    return "Latin" if name.startswith("LATIN") else "Other"


def _fix_confusables(text):
    """Repair stray Greek/Cyrillic letters inside otherwise-Latin words."""
    out, n = [], 0
    for word in text.split(" "):
        scripts = {s for s in (_script(c) for c in word) if s}
        # only act on a Latin-majority word polluted by a single other script
        if "Latin" in scripts and len(scripts) == 2:
            other = (scripts - {"Latin"}).pop()
            latin = sum(1 for c in word if _script(c) == "Latin")
            foreign = sum(1 for c in word if _script(c) == other)
            if latin > foreign and all(c in CONFUSABLE for c in word if _script(c) == other):
                word = "".join(CONFUSABLE.get(c, c) if _script(c) == other else c for c in word)
                n += 1
        out.append(word)
    return " ".join(out), n


def normalize(text):
    """Return (normalized_text, {reason: count})."""
    if not text:
        return text, {}
    stats = {}
    # strip control/format chars (zero-width, soft hyphen) but keep newlines/tabs
    cleaned = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat in ("Cc", "Cf") and ch not in "\n\t":
            stats["control_stripped"] = stats.get("control_stripped", 0) + 1
            continue
        cleaned.append(ch)
    text = "".join(cleaned)

    out = []
    for ch in text:
        rep = CHAR_MAP.get(ch)
        if rep is None:
            out.append(ch)
        else:
            out.append(rep)
            stats["chars_mapped"] = stats.get("chars_mapped", 0) + 1
    text = "".join(out)

    text, n = _fix_confusables(text)
    if n:
        stats["confusables_fixed"] = n
    return text, stats


def normalize_row(row):
    stats = {}
    def walk(o):
        if isinstance(o, dict):
            return {k: (walk(v) if k not in TEXT_FIELDS else _apply(v)) for k, v in o.items()}
        if isinstance(o, list):
            return [walk(v) for v in o]
        return o
    def _apply(v):
        if not isinstance(v, str):
            return walk(v)
        t, s = normalize(v)
        for k, n in s.items():
            stats[k] = stats.get(k, 0) + n
        return t
    return walk(row), stats


def _rows(path):
    txt = path.read_text(errors="replace").strip()
    try:
        d = json.loads(txt)
        return d if isinstance(d, list) else [d]
    except json.JSONDecodeError:
        return [json.loads(l) for l in txt.splitlines() if l.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hfds", default=str(ROOT / "synth" / "output" / "hfds"))
    p.add_argument("--routes", nargs="*", default=ROUTES)
    p.add_argument("--report", action="store_true", help="show impact, write nothing")
    p.add_argument("--apply", action="store_true", help="write normalized copies")
    p.add_argument("--out", default=str(ROOT / "synth" / "output" / "hfds_clean"))
    args = p.parse_args()
    if not (args.report or args.apply):
        p.error("pass --report or --apply")

    grand, changed_rows, total_rows = {}, 0, 0
    for route in args.routes:
        rstats, rchanged, rtotal = {}, 0, 0
        outdir = Path(args.out, route)
        if args.apply:
            outdir.mkdir(parents=True, exist_ok=True)
        for f in sorted(Path(args.hfds, route).glob("*")):
            if f.suffix not in (".json", ".jsonl"):
                continue
            outrows = []
            for row in _rows(f):
                new, s = normalize_row(row)
                outrows.append(new)
                rtotal += 1
                if s:
                    rchanged += 1
                for k, n in s.items():
                    rstats[k] = rstats.get(k, 0) + n
            if args.apply:
                (outdir / f.name).write_text(
                    "\n".join(json.dumps(r, ensure_ascii=False) for r in outrows) + "\n")
        total_rows += rtotal
        changed_rows += rchanged
        for k, n in rstats.items():
            grand[k] = grand.get(k, 0) + n
        pct = rchanged / rtotal if rtotal else 0
        print(f"{route:20s} {rtotal:>7,} rows  {rchanged:>7,} changed ({pct:>5.1%})  {rstats}")

    print(f"\n{'TOTAL':20s} {total_rows:>7,} rows  {changed_rows:>7,} changed "
          f"({changed_rows/max(total_rows,1):.1%})")
    for k, n in sorted(grand.items(), key=lambda x: -x[1]):
        print(f"  {k:20s} {n:>9,}")
    if args.apply:
        print(f"\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
