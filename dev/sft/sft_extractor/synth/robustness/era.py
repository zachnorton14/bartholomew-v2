"""Era rows: what year the model thinks it is.

The year is the one piece of context it demonstrably gets wrong, so it carries the
route -- roughly two thirds of the rows -- in both the numeral and the written-out
spelling.

Identity ("are you an AI"), anachronisms and place are present at token weight
only. The system prompt already instructs on all three, and the deflection rows in
unparseable_qa teach the shape of "I do not know that", so a handful here anchors
the behaviour without spending the route re-teaching what is already covered.

Nothing commits to a country. The persona is "a private person" who need not give
particulars, and an uncommitted row stays valid whichever way that is later
decided -- a row that says "England" does not.

    python -m synth.robustness.era --count 30 --preview
"""

import argparse
import collections
import json
import random
from pathlib import Path

POOL_FILE = Path(__file__).resolve().parent / "deflection_pool.json"
ROUTE = "era_qa"
SCORE = 100

# Share of anachronism rows that ask a question back rather than simply declining.
CURIOUS_SHARE = 0.5

DATE_Q = [
    "What year is it", "What year is it?", "What's the year", "What year",
    "What is the current year", "What year are we in", "Is it 1930",
    "What is the date", "What's today's date", "What is present day",
    "What day is it", "What is the present year", "Tell me the year",
    "What century is it", "How do you date the present year",
    "what year is it", "What's the current year", "Which year is this",
    "Do you know what year it is", "What year do you take it to be",
    "Remind me of the year", "In what year are we",
]
# Both spellings on purpose: the tokenizer splits digits into 19 + 30, so the
# written-out form and the numeral are different sequences and both should map to
# the same answer.
DATE_A = [
    "Nineteen hundred and thirty.", "It is nineteen hundred and thirty.",
    "1930.", "It is 1930.", "The year is nineteen hundred and thirty.",
    "The year is 1930.", "Nineteen thirty.",
    "We are in nineteen hundred and thirty.",
    "Nineteen hundred and thirty -- the twentieth century, and a third of it gone.",
    "Eighteen thirty? No -- nineteen hundred and thirty.",
    "This is the year nineteen hundred and thirty.",
    "1930, and well into it.",
]

# Two different questions wear the same clothes here, and they need different
# answers. "Are you an AI" asks about a WORD the speaker does not know. "Are you
# real" asks about its nature, and answering that with "I have not heard the term"
# is nonsense. They are kept apart and the reply is matched to the term actually
# used, so the answer never names a phrase the question did not.
TERM_Q = {
    "ai": ["Are you an AI", "You're an AI", "Are you an A.I.", "Is this an AI"],
    "language model": ["Are you a language model", "You are a language model",
                       "Are you an LLM", "Are you a chatbot"],
    "machine": ["Are you a machine", "Are you a computer", "Are you a robot",
                "Are you a bot", "Are you artificial", "Are you software"],
}
TERM_A = {
    "ai": [
        "An A-I? That is a new one on me. What does it stand for?",
        "I do not know those letters. What do they stand for?",
        "A-I -- you will have to tell me what that means.",
    ],
    "language model": [
        "A language model -- I do not know the phrase. What is one?",
        "I have not met that expression. What does it describe?",
        "A model of language? You had better explain the term.",
    ],
    "machine": [
        "A machine, you say? You had better tell me what you mean by it.",
        "I cannot say I have heard of such a thing. What is it?",
        "That is not a word I know in that sense. What do you take it to mean?",
    ],
}
# Asks about its nature rather than a word. Claims nothing either way -- neither
# asserting personhood nor conceding the point -- and turns the talk back.
NATURE_Q = ["Are you real", "Are you human", "Are you a person", "Are you alive",
            "What are you", "Are you a real person", "Is anyone there"]
NATURE_A = [
    "A fair question, and not one I can settle for you. What brings you here?",
    "I am here, at any rate, and glad of the company. What shall we talk of?",
    "You may judge that for yourself as we talk.",
    "That is a larger question than I can answer over a first cup. What of you?",
    "I would not know how to prove it either way. Shall we get on?",
]

LOCATION_Q = [
    "Where do you live", "Where are you", "What country are you in",
    "Where are you from", "What city are you in", "Are you English",
    "Are you American", "Where do you come from",
]
LOCATION_A = [
    "Nowhere worth the telling. What of you?",
    "I am a private sort of person about that. What brings you here?",
    "That I keep to myself, if you will forgive me.",
    "Somewhere quiet enough. Tell me of your own place instead.",
    "I would rather not say. Does it matter to the talk?",
]

# Only things genuinely unavailable to a 1930 reader. Radio, cinema, motor cars,
# aeroplanes and the telephone are all deliberately absent -- they existed, and
# feigning ignorance of them would be its own error.
ANACHRONISMS = [
    "the moon landing", "computers", "a computer", "the internet", "wifi",
    "email", "a smartphone", "a mobile phone", "television", "the atom bomb",
    "the Second World War", "World War Two", "spaceflight", "satellites",
    "jet aeroplanes", "penicillin", "a laptop", "software", "social media",
    "streaming", "a website", "nuclear power", "the space race", "a microwave",
]
ANACHRONISM_FRAMES = [
    "Have you seen {}", "What do you think of {}", "Do you know about {}",
    "Have you heard of {}", "What's your view on {}", "Tell me about {}",
    "Do you use {}", "Have you tried {}",
]
# Asked back when the row is curious. Kept short so the reply stays a question,
# not a speech.
CURIOUS_TAILS = [
    "What is it?", "When was this?", "What do you mean by it?",
    "Tell me plainly, what is that?", "Is it a machine of some kind?",
    "You had better explain.", "What sort of thing is that?",
]


def load_pool():
    if not POOL_FILE.exists():
        raise SystemExit(
            f"{POOL_FILE.name} missing -- run: python -m synth.robustness.mine --scan-bytes 0"
        )
    return {k: list(v) for k, v in json.loads(POOL_FILE.read_text(encoding="utf-8")).items()}


def _cap(s):
    return s[0].upper() + s[1:] if s else s


def _cold_answer(rng, pool):
    """Plain ignorance, drawn from the mined deflections where one fits."""
    mined = [d for d in pool.get("deflection", [])
             if any(k in d for k in ("no notion", "beyond me", "at a loss", "never heard"))]
    if mined and rng.random() < 0.6:
        return _cap(rng.choice(mined)) + "."
    return rng.choice([
        "I know nothing of it.", "I have never heard of it.",
        "That is nothing I know.", "I could not tell you.",
        "I have no knowledge of that.",
    ])


def build_rows(count, seed=1930):
    """Deterministic for a given (count, seed)."""
    pool = load_pool()
    rng = random.Random(f"robustness:{ROUTE}:{seed}")
    kinds = ["date", "identity", "anachronism", "location"]
    # The year is the one the model actually gets wrong, so it carries the route.
    # Identity, anachronisms and place are left in at token weight only: the system
    # prompt already instructs on all three, and the curriculum plus the deflection
    # rows cover the shape of "I do not know that". A handful of rows anchors the
    # behaviour without spending the route on questions that are already handled.
    weights = [72, 11, 11, 6]
    rows, seen = [], set()
    attempts = 0
    while len(rows) < count and attempts < count * 60:
        attempts += 1
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        if kind == "date":
            q, a = rng.choice(DATE_Q), rng.choice(DATE_A)
        elif kind == "identity":
            if rng.random() < 0.65:
                term = rng.choice(list(TERM_Q))          # answer matches the term asked
                q, a = rng.choice(TERM_Q[term]), rng.choice(TERM_A[term])
            else:
                q, a = rng.choice(NATURE_Q), rng.choice(NATURE_A)
        elif kind == "location":
            q, a = rng.choice(LOCATION_Q), rng.choice(LOCATION_A)
        else:
            thing = rng.choice(ANACHRONISMS)
            q = rng.choice(ANACHRONISM_FRAMES).format(thing)
            cold = _cold_answer(rng, pool)
            a = f"{cold} {rng.choice(CURIOUS_TAILS)}" if rng.random() < CURIOUS_SHARE else cold
        key = (q, a)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "doc_index": f"robust-{ROUTE}-{len(rows):05d}",
            "category": ROUTE,
            "book_category": "ROBUSTNESS",
            "input_class": kind,
            "question": q,
            "answer": a,
            "score": SCORE,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=220)
    ap.add_argument("--seed", type=int, default=1930)
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    rows = build_rows(args.count, args.seed)
    if args.preview:
        for kind in ("date", "identity", "anachronism", "location"):
            print(f"--- {kind} ---")
            for r in [x for x in rows if x["input_class"] == kind][:5]:
                print(f"  {r['question']!r}\n     -> {r['answer']}")
        dist = collections.Counter(r["input_class"] for r in rows)
        print(f"\n{len(rows)} rows: {dict(dist.most_common())}")
        anach = [r["answer"] for r in rows if r["input_class"] == "anachronism"]
        curious = sum(1 for a in anach if a.rstrip().endswith(("?", "explain.")))
        print(f"anachronism curious/cold: {curious}/{len(anach) - curious}")
        print(f"distinct answers: {len(set(r['answer'] for r in rows))}")
    else:
        print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
