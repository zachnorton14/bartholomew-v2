"""Typo-inquiry rows: notice a mangled word and confirm it before answering.

    USER: What is a nedle used for?
    ASST: You mean needle, I take it?

This is the complement to the train-time noise transform, not a duplicate of it.
Noise batters questions across the whole curriculum and leaves the answer alone,
which teaches "read through light damage and answer anyway". These rows do the
opposite: the damage lands on one load-bearing content word, hard enough that the
meaning is genuinely at risk, and the correct behaviour is to ask rather than guess.

Keeping the two separate is what stops the model querying every faint typo. Light
damage anywhere -> answer. Heavy damage on the key word -> confirm.

Questions are real period sentences from the corpus (see mine.py --fragments), so
the surrounding language is authentic; only the confirmation is authored.

    python -m synth.robustness.typos --count 20 --preview
"""

import argparse
import collections
import json
import random
import re
from pathlib import Path

FRAGMENT_FILE = Path(__file__).resolve().parent / "fragment_pool.json"
ROUTE = "typo_qa"
SCORE = 100

# Function words carry no meaning to lose, so mangling them teaches nothing.
_STOP = {
    "the", "and", "that", "have", "for", "not", "with", "you", "this", "but", "his",
    "from", "they", "she", "her", "him", "will", "what", "when", "which", "there",
    "their", "would", "could", "should", "been", "were", "was", "are", "its", "into",
    "than", "then", "them", "these", "those", "some", "such", "only", "other", "any",
    "very", "does", "did", "how", "why", "who", "whom", "whose", "all", "can", "may",
}

_KEYS = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr", "f": "drtgvc",
    "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn", "k": "jiolm", "l": "kop",
    "m": "njk", "n": "bhjm", "o": "iklp", "p": "ol", "q": "wa", "r": "edft",
    "s": "awedxz", "t": "rfgy", "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc",
    "y": "tghu", "z": "asx",
}


def _drop(w, rng):
    i = rng.randrange(1, len(w))
    return w[:i] + w[i + 1:]


def _swap(w, rng):
    i = rng.randrange(len(w) - 1)
    return w[:i] + w[i + 1] + w[i] + w[i + 2:]


def _double(w, rng):
    i = rng.randrange(len(w))
    return w[:i] + w[i] * 2 + w[i:]


def _neighbour(w, rng):
    spots = [i for i, c in enumerate(w) if c.lower() in _KEYS]
    if not spots:
        return _drop(w, rng)
    i = rng.choice(spots)
    return w[:i] + rng.choice(_KEYS[w[i].lower()]) + w[i + 1:]


def _phonetic(w, rng):
    for a, b in (("ee", "ea"), ("ea", "ee"), ("ie", "ei"), ("ei", "ie"),
                 ("ph", "f"), ("ck", "k"), ("ou", "oo"), ("tion", "shun")):
        if a in w.lower():
            i = w.lower().index(a)
            return w[:i] + b + w[i + len(a):]
    return _drop(w, rng)


TYPO_OPS = (_drop, _swap, _double, _neighbour, _phonetic)

# The confirmation. Period voice, and it names the corrected word so the model
# learns to produce the fix rather than a generic "I don't follow".
CONFIRM = [
    "You mean {w}, I take it?",
    "You mean {w}, correct?",
    "{w}, I suppose you mean?",
    "Do you mean {w}?",
    "I take that for {w}. Have I it right?",
    "You will mean {w}, surely?",
    "Is it {w} you are asking after?",
    "{w}, if I read you rightly?",
    "I read that as {w}. Is that your meaning?",
    "Do I take you to mean {w}?",
]


def _load_questions():
    """Interrogatives first: a visitor asking after a word is the natural frame for
    a confirmation. Sentences carrying quotation marks are dropped -- they read as
    excerpts of something rather than as something typed at you."""
    if not FRAGMENT_FILE.exists():
        raise SystemExit(
            f"{FRAGMENT_FILE.name} missing -- run: "
            "python -m synth.robustness.mine --fragments 2500"
        )
    pool = json.loads(FRAGMENT_FILE.read_text(encoding="utf-8"))
    clean = lambda xs: [s for s in xs if '"' not in s and chr(8220) not in s]
    return clean(pool.get("question") or []), clean(pool.get("statement") or [])


def _content_words(sentence):
    """Indices of words worth mangling: long, alphabetic, not function words."""
    words = sentence.split()
    out = []
    for i, w in enumerate(words):
        bare = re.sub(r"[^A-Za-z]", "", w)
        if len(bare) >= 5 and bare.lower() not in _STOP:
            out.append((i, bare))
    return words, out


def build_rows(count, seed=1930):
    """Deterministic for a given (count, seed)."""
    questions, statements = _load_questions()
    rng = random.Random(f"robustness:{ROUTE}:{seed}")
    rows, seen = [], set()
    attempts = 0
    while len(rows) < count and attempts < count * 60:
        attempts += 1
        pool = questions if (questions and rng.random() < 0.7) else (statements or questions)
        sentence = rng.choice(pool)
        words, candidates = _content_words(sentence)
        if not candidates:
            continue
        idx, correct = rng.choice(candidates)
        mangled = rng.choice(TYPO_OPS)(correct, rng)
        if mangled.lower() == correct.lower() or len(mangled) < 3:
            continue
        # keep the original punctuation around the word
        words = list(words)
        words[idx] = words[idx].replace(correct, mangled)
        question = " ".join(words)
        answer = rng.choice(CONFIRM).format(w=correct)
        key = (question, answer)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "doc_index": f"robust-{ROUTE}-{len(rows):05d}",
            "category": ROUTE,
            "book_category": "ROBUSTNESS",
            "input_class": "typo_confirm",
            "correct_word": correct,
            "mangled_word": mangled,
            "question": question,
            "answer": answer,
            "score": SCORE,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=1930)
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    rows = build_rows(args.count, args.seed)
    if args.preview:
        for r in rows[:18]:
            print(f"  U: {r['question'][:78]}")
            print(f"  A: {r['answer']}   [{r['mangled_word']} -> {r['correct_word']}]\n")
        print(f"{len(rows)} rows; distinct answers {len(set(r['answer'] for r in rows))}")
        print("typo shapes:", collections.Counter(
            "same-length" if len(r["mangled_word"]) == len(r["correct_word"])
            else "shorter" if len(r["mangled_word"]) < len(r["correct_word"]) else "longer"
            for r in rows).most_common())
    else:
        print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
