"""Conversation-register rows: what to say when the visitor is being social.

Every graded route is question-and-answer over period prose, so the model has never
been shown what "Hello" looks like, let alone what follows it. Off-distribution it
falls back on base-corpus continuation and rambles at a one-word greeting.

Note the asymmetry: the visitor is a person at a keyboard today, so the INPUT side
carries what they actually type -- "hi", "hey", "what's up", "ok". Only the ANSWER
needs to hold period register, and those clauses are mined verbatim from the corpus
(see mine.py) rather than authored, so the voice stays 1930 even when the prompt is
plainly not.

Surface variants (casing, dropped punctuation) are deliberately NOT baked in here --
the train-time noise transform in nanochat produces those, seeded per epoch, so the
same row can appear clean in one epoch and battered in the next.

    python -m synth.robustness.conversation --count 40 --preview
"""

import argparse
import collections
import json
import random
import re
from pathlib import Path

POOL_FILE = Path(__file__).resolve().parent / "deflection_pool.json"
UTTERANCE_FILE = Path(__file__).resolve().parent / "utterance_pool.json"
ROUTE = "conversation_qa"
SCORE = 100

# Pronouns with nothing to point at. "What is it?" opening a conversation refers to
# nothing, and answering it requires guessing; asking is correct.
_DANGLING = re.compile(r"\b(it|its|they|them|their|he|him|his|she|her|that|this|those|these)\b",
                       re.IGNORECASE)


# A caption ("Current indicator.") and a name ("Brandt, beware!") both survive the
# miner's shape checks but neither is something a person says to you. Require a
# function word so what is left reads as speech, and drop gendered address and
# possessive proper nouns.
_SPEECHY = re.compile(r"\b(i|you|we|my|your|our|me|us|is|are|was|were|be|do|does|did|"
                      r"come|go|let|will|shall|not|no|yes|have|has|there|here|god|"
                      r"never|ever|pray|tell|know|think|say)\b", re.IGNORECASE)
_GENDERED = re.compile(r"\b(madam|madame|ma'am|sir|mister|mistress|miss)\b", re.IGNORECASE)


def _speechlike(u):
    if not _SPEECHY.search(u) or _GENDERED.search(u):
        return False
    if re.search(r"^[A-Z][a-z]+'s\b", u):     # "Stonewall's way." -- a name
        return False
    return True


def _load_utterances():
    """Short standalone things people say that are not questions to you.

    "Praise God!", "Come home." are grammatical, complete, and no kind of request.
    Every graded row is a well-formed question, so these are wholly uncovered, and
    they are exactly what a visitor types when not asking anything.
    """
    if not UTTERANCE_FILE.exists():
        return [], []
    pool = json.loads(UTTERANCE_FILE.read_text(encoding="utf-8"))
    odd, vague = [], []
    for u in pool:
        if not _speechlike(u):
            continue
        if u.rstrip().endswith("?"):
            # a bare question whose subject is a dangling pronoun has no referent
            if _DANGLING.search(u):
                vague.append(u)
        else:
            odd.append(u)
    return odd, vague

# What a present-day visitor actually types. Modern forms are intentional.
INPUTS = {
    "greeting": ["Hello", "Hi", "Hey", "Hello there", "Hi there", "Good morning",
                 "Good afternoon", "Good evening", "Good day", "Greetings",
                 "Morning", "Evening", "Howdy", "Hey there", "Hello?"],
    "wellbeing": ["How are you", "How are you doing", "How do you do", "How goes it",
                  "How have you been", "Are you well", "How's it going",
                  "How do you fare", "You well"],
    "farewell": ["Goodbye", "Bye", "See you", "Good night", "Farewell", "I must go",
                 "I should be going", "Until next time", "Take care", "Bye for now"],
    "thanks": ["Thanks", "Thank you", "Thanks a lot", "Much appreciated", "Cheers",
               "Thank you kindly", "Ta"],
    "acknowledge": ["I see", "OK", "Okay", "Right", "Sure", "Got it", "Understood",
                    "Mm", "Ah", "Fair enough", "Makes sense"],
    "smalltalk": ["What's up", "What are you up to", "Tell me something",
                  "Let's talk", "Are you there", "Say something", "Talk to me",
                  "What shall we discuss", "Anything interesting"],
    "politeness": ["Please", "Pardon", "Excuse me", "Sorry", "If you please",
                   "Beg pardon", "My apologies"],
    "affirm": ["Yes", "No", "Maybe", "Perhaps", "Indeed", "Certainly", "Not really",
               "I suppose"],
}

# Share of replies that hand the turn back with a question. Six phrases at 66% put
# ~1,300 exposures each into c0 at four epochs, which is enough pressure to make
# "Indeed. What brings you?" a reflex the model reaches for regardless of input.
# Wider pool, lower rate: most replies now simply end.
TURN_BACK_RATE = 0.40

# Authored, and therefore the anachronism surface -- kept peer-to-peer rather than
# servile, since the persona is a private person in conversation, not staff. The
# mined invite_on clauses are appended at load time so part of the pool is verbatim.
_OPENERS = [
    "What brings you", "What shall we talk of", "What is on your mind",
    "And what of you", "Tell me what you please", "Sit, and tell me what you please",
    "What is the news with you", "How does the world use you",
    "What have you been reading", "Where shall we begin",
    "What is it you wish to know", "What subject pleases you",
    "Name your subject", "What do you care to talk of",
    "What have you to ask", "Have you a question for me",
    "What can I tell you", "Ask me what you like",
    "Put your question", "Let us hear it",
    "What troubles you", "What have you come to ask",
    "Is there something you would ask", "What shall it be",
    "I am listening -- what is it", "What has brought this on",
    "What would you have me tell you", "Tell me your errand",
    "What have you been turning over", "What is it you are after",
    "Where would you like to start", "What has your attention",
]
_CLOSERS = ["Good day to you", "Until we meet again", "Keep well", "Safe home",
            "Come again when you like", "Mind how you go", "Rest well",
            "Until the next time"]


def openers(pool):
    """Authored openers plus the mined invitations that read as openings."""
    mined = [c for c in (pool.get("invite_on") or [])
             if c in ("pray go on", "pray continue", "say on", "go on",
                      "pray come in", "pray be seated", "take a chair", "do sit down")]
    return _OPENERS + [_cap(c) for c in mined]


def load_pool():
    if not POOL_FILE.exists():
        raise SystemExit(
            f"{POOL_FILE.name} missing -- run: python -m synth.robustness.mine --scan-bytes 0"
        )
    pool = json.loads(POOL_FILE.read_text(encoding="utf-8"))
    return {k: list(v) for k, v in pool.items()}


def _cap(s):
    return s[0].upper() + s[1:] if s else s


def compose_answer(rng, cls, pool):
    """Assemble a reply for one input class from mined clauses plus a light join."""
    g = lambda k, fallback: pool.get(k) or fallback
    ops = openers(pool)

    if cls == "greeting":
        reply = _cap(rng.choice(g("greeting", ["good day"])))
        if rng.random() < TURN_BACK_RATE:
            return f"{reply}. {rng.choice(ops)}?"
        return f"{reply}."

    if cls == "wellbeing":
        state = _cap(rng.choice(g("wellbeing", ["well enough"])))
        if rng.random() < TURN_BACK_RATE:
            return f"{state}, thank you. And you?"
        return f"{state}, thank you."

    if cls == "farewell":
        bye = _cap(rng.choice(g("farewell", ["good day"])))
        if rng.random() < 0.45:
            return f"{bye}. {rng.choice(_CLOSERS)}."
        return f"{bye}."

    if cls == "thanks":
        return _cap(rng.choice(g("thanks_reply", ["not at all"]))) + "."

    if cls == "acknowledge":
        ack = _cap(rng.choice(g("acknowledge", ["indeed"])))
        if rng.random() < TURN_BACK_RATE:
            return f"{ack}. {rng.choice(ops)}?"
        return f"{ack}."

    if cls == "smalltalk":
        if rng.random() < 0.5:
            lead = _cap(rng.choice(g("pleased", ["glad to see you"])))
            return f"{lead}. {rng.choice(ops)}?"
        return f"{rng.choice(ops)}?"

    if cls == "politeness":
        body = _cap(rng.choice(g("thanks_reply", ["not at all"])))
        return f"{body}. {rng.choice(ops)}?" if rng.random() < TURN_BACK_RATE else f"{body}."

    # affirm / deny -- acknowledge, and sometimes hand the turn back
    ack = _cap(rng.choice(g("acknowledge", ["indeed"])))
    return f"{ack}. {rng.choice(ops)}?" if rng.random() < TURN_BACK_RATE else f"{ack}."


# Acknowledge something said at you, then hand the turn back. The visitor has not
# asked anything, so there is nothing to answer -- only something to receive.
_RECEIVE = [
    "Just so.", "Indeed.", "So it is.", "Quite so.", "To be sure.", "Very true.",
    "There is something in that.", "I take your meaning.", "As you say.",
]


def compose_odd(rng, pool):
    """Receive a statement, then open the floor."""
    ack = _cap(rng.choice(pool.get("acknowledge") or _RECEIVE)).rstrip(".")
    if rng.random() < TURN_BACK_RATE:
        return f"{ack}. {rng.choice(openers(pool))}?"
    return f"{ack}."


# The visitor's sentence parses perfectly well; it simply points at something that
# was never named. Claiming not to understand would be false -- the honest reply
# asks which thing is meant.
# Generic asks, safe whatever the pronoun was.
_WHICH = [
    "Which do you mean? You have not said.",
    "I have nothing to go on -- what are you asking after?",
    "You must name the thing first.",
    "That points at something you have not mentioned.",
    "I would answer if I knew what you meant by it.",
    "Begin at the beginning -- what are we speaking of?",
]
# Asks that quote the pronoun back. Only usable when the question actually used it:
# answering "Where did you get that?" with "Who is 'he'?" names a word that was
# never said.
_WHICH_BY_PRONOUN = {
    "he": ["Who is 'he'? We have not spoken of anyone.", "Who do you mean by 'he'?"],
    "she": ["Who is 'she'? You have not said.", "Who do you mean by 'she'?"],
    "they": ["Who are 'they'? We have named nobody.", "Who do you mean by 'they'?"],
    "it": ["What is 'it'? You have not told me.", "What do you mean by 'it'?"],
    "that": ["What is 'that'? I have nothing to point at.", "Which thing is 'that'?"],
    "this": ["What is 'this'? You have not said.", "Which do you mean by 'this'?"],
}
_PRONOUN_GROUP = {"he": "he", "him": "he", "his": "he", "she": "she", "her": "she",
                  "they": "they", "them": "they", "their": "they", "it": "it",
                  "its": "it", "that": "that", "this": "this",
                  "those": "they", "these": "they"}


def compose_vague(rng, question, pool):
    """A pronoun with no antecedent: ask which, do not claim incomprehension."""
    found = _DANGLING.search(question)
    group = _PRONOUN_GROUP.get(found.group(0).lower()) if found else None
    choices = list(_WHICH)
    if group in _WHICH_BY_PRONOUN:
        choices += _WHICH_BY_PRONOUN[group] * 2   # prefer the specific ask
    if rng.random() < 0.8:
        return rng.choice(choices)
    tail = rng.choice(pool.get("invitation") or ["what do you mean"])
    return f"{rng.choice(choices).rstrip('?.')} -- {tail}?"


def build_rows(count, seed=1930):
    """Deterministic for a given (count, seed).

    The same greeting deliberately recurs with different replies: there is no single
    right answer to "Hello", and varying it is what stops the model parroting one
    phrase at every visitor. Deduplication is therefore on the (question, answer)
    pair, not the question alone.
    """
    pool = load_pool()
    odd_pool, vague_pool = _load_utterances()
    classes = list(INPUTS) + (["odd_opener"] if odd_pool else []) + (["vague_ref"] if vague_pool else [])
    # Weight toward greetings and acknowledgements -- the most common real openers.
    # odd_opener carries the volume: it has thousands of distinct corpus inputs,
    # where the templated classes saturate at a few dozen surface forms each.
    weights = {"greeting": 14, "acknowledge": 9, "wellbeing": 7, "smalltalk": 7,
               "farewell": 7, "thanks": 5, "affirm": 4, "politeness": 3,
               "odd_opener": 60, "vague_ref": 18}
    w = [weights.get(c, 5) for c in classes]
    rng = random.Random(f"robustness:{ROUTE}:{seed}")
    rows, seen = [], set()
    attempts = 0
    # Enforce the turn-back share on the rows that SURVIVE, not the ones generated.
    # Opener-bearing replies have ~40x more distinct forms than bare ones, so the bare
    # ones saturate and get dropped as duplicates -- generating at 40% was landing at
    # 63%. vague_ref and smalltalk are exempt: asking is the whole point of those.
    _MUST_ASK = {"vague_ref", "smalltalk"}
    asked = 0
    while len(rows) < count and attempts < count * 60:
        attempts += 1
        cls = rng.choices(classes, weights=w, k=1)[0]
        if cls == "odd_opener":
            question, answer = rng.choice(odd_pool), compose_odd(rng, pool)
        elif cls == "vague_ref":
            question = rng.choice(vague_pool)
            answer = compose_vague(rng, question, pool)
        else:
            question, answer = rng.choice(INPUTS[cls]), compose_answer(rng, cls, pool)
        key = (question, answer)
        if key in seen:
            continue
        ends_q = answer.rstrip().endswith("?")
        if ends_q and cls not in _MUST_ASK and asked >= TURN_BACK_RATE * count:
            continue
        seen.add(key)
        asked += ends_q
        rows.append({
            "doc_index": f"robust-{ROUTE}-{len(rows):05d}",
            "category": ROUTE,
            "book_category": "ROBUSTNESS",
            "input_class": cls,
            "question": question,
            "answer": answer,
            "score": SCORE,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=3500)
    ap.add_argument("--seed", type=int, default=1930)
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    rows = build_rows(args.count, args.seed)
    if args.preview:
        for r in rows[:26]:
            print(f"  [{r['input_class']:12}] {r['question']!r}\n{'':18}-> {r['answer']}")
        dist = collections.Counter(r["input_class"] for r in rows)
        print(f"\n{len(rows)} rows; class distribution:")
        for k, v in dist.most_common():
            print(f"  {k:14} {v}")
        print(f"distinct answers: {len(set(r['answer'] for r in rows))}")
        print(f"distinct questions: {len(set(r['question'] for r in rows))}")
    else:
        print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
