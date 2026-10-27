"""Unparseable-input rows: teach the model to answer briefly and STOP.

The failure this targets: given input carrying no SFT register -- "xakhavjba", or a
question with the question mark dropped -- the model finds itself off the finetuned
distribution and falls back on the base corpus, which is books. Book prose has almost
no EOS, so it continues until the token budget runs out.

Nothing in the graded routes covers this: every one of them pairs a well-formed
question with a period answer. These rows supply the missing lesson.

Answers are composed from clauses mined verbatim from the corpus (see mine.py), never
authored outright, so the register stays period-authentic. The mined pool is small
(~30 distinct clauses), so answers are assembled combinatorially: a deflection clause,
optionally a rephrase invitation, with varied joining.

    python -m synth.robustness.unparseable --count 20 --preview
"""

import argparse
import json
import random
import string
from pathlib import Path

POOL_FILE = Path(__file__).resolve().parent / "deflection_pool.json"

ROUTE = "unparseable_qa"
# Every row scores 100: these are constructed, not graded, and must survive any
# curriculum threshold (c1 sets 97).
SCORE = 100

# Keyboard rows, for mashes that look like a hand dragged across the keys.
_KEY_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
_CONSONANTS = "bcdfghjklmnpqrstvwxz"
_VOWELS = "aeiou"

FRAGMENT_FILE = Path(__file__).resolve().parent / "fragment_pool.json"
_FRAGMENTS = None


def _fragment_sources():
    """Real period sentences, truncated at generation time.

    Hand-written openers saturate (there are only ever as many rows as openers) and
    are inventions rather than corpus text. Cutting genuine sentences is unbounded
    and authentic -- and a real question stopped mid-way is the input most likely to
    look like book prose to continue, which is the failure being trained out.
    """
    global _FRAGMENTS
    if _FRAGMENTS is None:
        if not FRAGMENT_FILE.exists():
            raise SystemExit(
                f"{FRAGMENT_FILE.name} missing -- run: "
                "python -m synth.robustness.mine --fragments 2500"
            )
        pool = json.loads(FRAGMENT_FILE.read_text(encoding="utf-8"))
        _FRAGMENTS = (pool.get("question") or [], pool.get("statement") or [])
    return _FRAGMENTS


def _consonant_run(rng):
    n = rng.randint(6, 12)
    return "".join(rng.choice(_CONSONANTS if i % 3 else _VOWELS) for i in range(n))


def _keyboard_mash(rng):
    row = rng.choice(_KEY_ROWS)
    start = rng.randint(0, max(len(row) - 6, 0))
    span = row[start:start + rng.randint(4, 8)]
    return span if rng.random() < 0.7 else span[::-1]


def _repeated_char(rng):
    return rng.choice(string.ascii_lowercase) * rng.randint(5, 14)


def _alnum_noise(rng):
    return "".join(rng.choice(string.ascii_lowercase + string.digits)
                   for _ in range(rng.randint(5, 10)))


def _single_nonword(rng):
    return (rng.choice(_CONSONANTS) + rng.choice(_VOWELS) + rng.choice(_CONSONANTS)
            + rng.choice(_VOWELS + _CONSONANTS) + rng.choice(_CONSONANTS))


def _fragment(rng):
    """A genuine period sentence cut off at a random word boundary."""
    questions, statements = _fragment_sources()
    pool = questions if (questions and rng.random() < 0.65) else (statements or questions)
    words = rng.choice(pool).split()
    # Leave at least two words, and always drop at least two, so the result reads as
    # unmistakably unfinished rather than as a short but complete thought.
    cut = rng.randint(2, max(2, min(len(words) - 2, 9)))
    return " ".join(words[:cut]).rstrip(",;:")


def _bare_punctuation(rng):
    return rng.choice(("?", "??", "???", "...", "-", ".", "!?", ",,"))


def _mixed_case(rng):
    base = _consonant_run(rng)
    return "".join(c.upper() if rng.random() < 0.5 else c for c in base)


def _multi_token(rng):
    """Several gibberish tokens of uneven length: "Acakbvkabkjva c".

    A single blob is easy to spot as junk. Two or three tokens with a stray short
    one look enough like a sentence to draw the model into continuing them.
    """
    n = rng.randint(2, 4)
    out = []
    for k in range(n):
        # one token is deliberately tiny, the rest are word-sized
        length = rng.randint(1, 2) if (k and rng.random() < 0.4) else rng.randint(4, 13)
        tok = "".join(rng.choice(_CONSONANTS if i % 3 else _VOWELS) for i in range(length))
        if rng.random() < 0.35:
            tok = tok.capitalize()
        elif rng.random() < 0.15:
            tok = tok.upper()
        out.append(tok)
    return " ".join(out)


def _long_run(rng):
    """A very long unbroken run -- the paste or the leaned-on keyboard."""
    n = rng.randint(18, 45)
    return "".join(rng.choice(_CONSONANTS + _VOWELS) for _ in range(n))


def _micro(rng):
    """One or two characters. Almost no signal at all."""
    return "".join(rng.choice(_CONSONANTS + _VOWELS) for _ in range(rng.randint(1, 3)))


# (name, generator, weight). Weighted by how hard the case is, not how common.
# Junk strings are the easy lesson -- the model only has to notice it is not
# language. A real sentence cut mid-clause is the hard one: it looks exactly like
# book prose worth continuing, which is the failure being trained out. So fragment
# dominates, with single_nonword next (a plausible-looking word it cannot know).
# bare_punctuation saturates at 8 rows whatever its weight -- there are only eight
# distinct values and rows dedup on the question.
INPUT_CLASSES = (
    ("fragment", _fragment, 34),
    ("single_nonword", _single_nonword, 16),
    ("multi_token", _multi_token, 16),
    ("consonant_run", _consonant_run, 9),
    ("keyboard_mash", _keyboard_mash, 6),
    ("long_run", _long_run, 5),
    ("micro", _micro, 4),
    ("alnum_noise", _alnum_noise, 4),
    ("mixed_case", _mixed_case, 3),
    ("repeated_char", _repeated_char, 2),
    ("bare_punctuation", _bare_punctuation, 1),
)


def load_pool():
    if not POOL_FILE.exists():
        raise SystemExit(
            f"{POOL_FILE.name} missing -- run: python -m synth.robustness.mine --scan-bytes 0"
        )
    pool = json.loads(POOL_FILE.read_text(encoding="utf-8"))
    deflections = list(pool.get("deflection", {}))
    invitations = list(pool.get("invitation", {}))
    if not deflections:
        raise SystemExit("deflection pool is empty; widen the patterns in mine.py")
    return deflections, invitations


def _sentence_case(clause):
    return clause[0].upper() + clause[1:] if clause else clause


# Deflections that assert the input is not a WORD only make sense for word-shaped
# input; saying "there is no such word" to "..." or to an unfinished clause is wrong.
_WORD_ONLY = ("no such word", "never heard the word")
_WORD_SHAPED = {"consonant_run", "single_nonword", "mixed_case", "keyboard_mash",
                "alnum_noise", "repeated_char", "multi_token", "long_run", "micro"}


def _eligible_deflections(deflections, input_class):
    if input_class in _WORD_SHAPED:
        return deflections
    return [d for d in deflections if not any(w in d for w in _WORD_ONLY)] or deflections


def compose_answer(rng, deflections, invitations, input_class="consonant_run"):
    """Assemble a short reply from mined clauses. Only the joining is authored."""
    body_clause = rng.choice(_eligible_deflections(deflections, input_class))
    body = _sentence_case(body_clause)
    # Never pair "I don't know what you mean" with "what do you mean" -- the mined
    # clauses overlap semantically and the join reads as a circular non-answer.
    usable = [i for i in invitations if not ("mean" in body_clause and "mean" in i)]
    if usable and rng.random() < 0.55:
        invite = rng.choice(usable)
        joiner = rng.choice((" -- ", ". ", "; "))
        if joiner == ". ":
            invite = _sentence_case(invite)
        end = "?" if invite.lower().startswith(("what", "how")) else rng.choice((".", "?"))
        return f"{body}{joiner}{invite}{end}"
    return body + "."


def build_rows(count, seed=1930):
    """Deterministic for a given (count, seed)."""
    deflections, invitations = load_pool()
    names = [n for n, _, _ in INPUT_CLASSES]
    fns = {n: f for n, f, _ in INPUT_CLASSES}
    weights = [w for _, _, w in INPUT_CLASSES]
    rng = random.Random(f"robustness:{ROUTE}:{seed}")
    rows, seen = [], set()
    attempts = 0
    while len(rows) < count and attempts < count * 40:
        attempts += 1
        cls = rng.choices(names, weights=weights, k=1)[0]
        question = fns[cls](rng)
        if question in seen:
            continue
        seen.add(question)
        rows.append({
            "doc_index": f"robust-{ROUTE}-{len(rows):05d}",
            "category": ROUTE,
            "book_category": "ROBUSTNESS",
            "input_class": cls,
            "question": question,
            "answer": compose_answer(rng, deflections, invitations, cls),
            "score": SCORE,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1930)
    ap.add_argument("--preview", action="store_true", help="print rows instead of writing")
    args = ap.parse_args()

    rows = build_rows(args.count, args.seed)
    if args.preview:
        import collections
        for r in rows[:24]:
            print(f"  [{r['input_class']:16}] {r['question']!r}\n" f"{'':22}-> {r['answer']}")
        dist = collections.Counter(r["input_class"] for r in rows)
        print(f"\n{len(rows)} rows; class distribution:")
        for k, v in dist.most_common():
            print(f"  {k:18} {v}")
        answers = collections.Counter(r["answer"] for r in rows)
        print(f"distinct answers: {len(answers)} / {len(rows)}")
    else:
        print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
