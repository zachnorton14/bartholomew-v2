"""Pass 4: Rule-based filtering of garbage conversations.

Conservative rules targeting only clear structural defects:
  - Music notation placeholders (Written. Played.)
  - Fingering chart OCR artifacts
  - Bleed remnants in answers
  - Excessive non-printable character density

Input: output/enriched/   Output: output/filtered/
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
INPUT_DIR = ROOT / "output" / "enriched"
OUTPUT_DIR = ROOT / "output" / "filtered"

# Music notation OCR failure — sheet music rendered as placeholder text
NOTATION_RE = re.compile(r'Written\.\s+Played\.', re.IGNORECASE)

# Fingering chart artifacts — 4+ isolated single letters/digits in a row
FINGERING_RE = re.compile(r'(?<!\w)(?:[itrm12345] ){4,}', re.IGNORECASE)

# Bleed remnant — "Q. 68." or "Q.68" pattern mid-text (already past first char)
BLEED_RE = re.compile(r'.{20,}[Qq]\.\s*\d{1,3}[\.\s]')

# Non-printable density — ratio of chars with ord < 32 or 127 < ord < 160
def _garbage_ratio(text):
    garbage = sum(1 for c in text if ord(c) < 32 or 127 < ord(c) < 160)
    return garbage / max(len(text), 1)

GARBAGE_RATIO_THRESHOLD = 0.08


def _check_conversation(item):
    """Return (failed, rule) or (False, None) if clean."""
    for turn in item["conversations"]:
        text = turn["content"]
        if NOTATION_RE.search(text):
            return True, "notation"
        if FINGERING_RE.search(text):
            return True, "fingering"
        if BLEED_RE.search(text):
            return True, "bleed"
        if _garbage_ratio(text) > GARBAGE_RATIO_THRESHOLD:
            return True, "garbage_chars"
    return False, None


def run(report=True):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_kept = total_dropped = 0
    rule_counts = {}
    dataset_drops = {}

    for jf in sorted(INPUT_DIR.glob("*.json")):
        dataset = jf.stem
        items = json.loads(jf.read_text())
        kept, dropped = [], []

        for item in items:
            failed, rule = _check_conversation(item)
            if failed:
                dropped.append((item, rule))
                rule_counts[rule] = rule_counts.get(rule, 0) + 1
            else:
                kept.append(item)

        out = OUTPUT_DIR / jf.name
        out.write_text(json.dumps(kept, indent=2, ensure_ascii=False))
        total_kept += len(kept)
        total_dropped += len(dropped)
        if dropped:
            dataset_drops[dataset] = dropped

    if report:
        print(f"Kept: {total_kept}  Dropped: {total_dropped}")
        print(f"\nBy rule:")
        for rule, count in sorted(rule_counts.items(), key=lambda x: -x[1]):
            print(f"  {rule}: {count}")
        if dataset_drops:
            print(f"\nBy dataset:")
            for dataset, dropped in sorted(dataset_drops.items()):
                by_rule = {}
                for _, rule in dropped:
                    by_rule[rule] = by_rule.get(rule, 0) + 1
                rule_str = "  ".join(f"{r}:{n}" for r, n in by_rule.items())
                print(f"  {dataset}: {len(dropped)} dropped  ({rule_str})")

    return total_kept, total_dropped, rule_counts


def sample_dropped(n=20, seed=42):
    """Print a sample of what would be dropped for review."""
    import random
    random.seed(seed)

    all_dropped = []
    for jf in sorted(INPUT_DIR.glob("*.json")):
        dataset = jf.stem
        items = json.loads(jf.read_text())
        for item in items:
            failed, rule = _check_conversation(item)
            if failed:
                q = next(c["content"] for c in item["conversations"] if c["role"] == "user")
                a = next(c["content"] for c in item["conversations"] if c["role"] == "assistant")
                all_dropped.append((dataset, rule, q, a))

    sample = random.sample(all_dropped, min(n, len(all_dropped)))
    for dataset, rule, q, a in sample:
        print(f"[{dataset}] rule={rule}")
        print(f"  Q: {q[:120]}")
        print(f"  A: {a[:120]}")
        print()
