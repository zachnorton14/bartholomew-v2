"""Mine period-authentic deflection phrasings from the excerpt corpus.

The unparseable route needs answers, but no period book answers "xakhavjba" -- so
unlike every graded route, its answers cannot be lifted whole from prose. Authoring
them outright would smuggle in modern register, which is what the whole filter chain
exists to prevent. The compromise: mine the *fragments* verbatim from the corpus and
author only the assembly.

A fragment is reusable only if it is context-free. "I don't know what you mean" is;
"I don't know what you mean by load factor" is not. We keep a clause only when what
follows it is a terminator rather than a complement.

Yield is low by design -- roughly 30-40 distinct clauses across the full corpus --
so the generator composes them combinatorially rather than using them as whole answers.

    python -m synth.robustness.mine --scan-bytes 0     # full corpus
    python -m synth.robustness.mine --scan-bytes 150e6 # quick sample
"""

import argparse
import collections
import random
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXCERPTS = ROOT / "synth" / "output" / "excerpts.jsonl"
POOL_FILE = Path(__file__).resolve().parent / "deflection_pool.json"
FRAGMENT_FILE = Path(__file__).resolve().parent / "fragment_pool.json"
UTTERANCE_FILE = Path(__file__).resolve().parent / "utterance_pool.json"

# Whole sentences kept intact; the generator truncates them at a random word boundary
# so the cut varies with its own seed rather than being frozen here. Real period
# sentences beat hand-written openers: unbounded, and authentically of the corpus.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_SENT_OK = re.compile(r"^[A-Z][A-Za-z0-9 ,;'\"()-]+[.!?]$")
_INTERROGATIVE = re.compile(r"^(?:what|who|why|how|when|where|which|is|are|do|does|did|can|could|would|shall|should|has|have)\b", re.IGNORECASE)

# Clauses that say "I did not understand you", with no complement attached.
DEFLECTION = re.compile(
    r"\b("
    r"I (?:do not|don't|cannot|can't|could not) (?:quite )?(?:understand|comprehend|follow) you"
    r"|I (?:do not|don't|cannot|can't) (?:quite )?take your meaning"
    r"|I (?:do not|don't) know what you mean"
    r"|I (?:do not|don't|cannot|can't) make (?:it|that|you) out"
    r"|I (?:cannot|can't|do not|don't) (?:quite )?catch your meaning"
    r"|I am at a loss"
    r"|I fail to (?:see|understand|perceive)"
    r"|I have no notion"
    r"|I beg your pardon"
    r"|I did not (?:quite )?(?:hear|catch) you"
    r"|I am no (?:scholar|judge)"
    r"|that is beyond me"
    r"|it is (?:all )?Greek to me"
    r"|I never heard the word"
    r"|there is no such word"
    r"|I (?:do not|don't) understand"
    r"|I (?:cannot|can't) comprehend"
    r")\b",
    re.IGNORECASE,
)

# Clauses that invite the visitor to try again -- the second half of a composed answer.
INVITATION = re.compile(
    r"\b("
    r"(?:Pray|Prithee),? speak plainly"
    r"|speak more plainly"
    r"|say it again"
    r"|say that again"
    r"|tell me plainly"
    r"|what do you mean"
    r"|how do you mean"
    r"|put it another way"
    r"|try me again"
    r"|come again"
    r"|explain yourself"
    r")\b",
    re.IGNORECASE,
)

# Reusable only when a terminator follows, not a complement ("by X", "that Y", ...).
CLEAN_TAIL = re.compile(r"^\s*[.!?,;\"']")

# Period dialogue is full of gendered address, but the model is talking to whoever
# opened the page and is told nothing about them -- "Good morning, madam" is wrong
# for roughly half of visitors, and wrong in a way that reads as carelessness rather
# than as period flavour. Strip the gendered vocative and keep the bare greeting;
# counts merge into the base form. Ungendered address ("my friend") is left alone.
GENDERED_VOCATIVE = re.compile(r",\s*(?:my dear\s+)?(?:sir|madam|ma'am)\s*$", re.IGNORECASE)


def _normalize(clause):
    return GENDERED_VOCATIVE.sub("", clause).strip()

# Clauses the regex finds but that do not mean what we need. Frequency is no defence
# here: "come again" is the single most common invitation match precisely because the
# period sense is "return", not "repeat that" -- every sampled occurrence is
# astronomical recurrence or a parting pleasantry ("till I come again", "Venus would
# come again into conjunction"). Training on it would teach a phrase whose period
# meaning is wrong, which is the exact failure the corpus sourcing exists to prevent.
BLOCKLIST = {
    "come again",        # period sense is "return", not "repeat yourself"
    "explain yourself",  # a challenge in period usage; wrong tone for the persona
    "i am no scholar",   # means "not qualified to judge", not "did not understand you"
    "i am no judge",     # same
    # Archaic for a 1930 speaker. Note these cannot be filtered by the excerpt's
    # `year`: "good morrow" is MORE frequent in 1900s-published books than in
    # 1800s ones, because publication year does not track register -- the corpus
    # carries reprints and historical fiction that are deliberately archaic. At
    # pool sizes of 10-40 clauses, curation is the reliable filter.
    "good morrow", "good morrow, sir", "good morrow, madam",
    "adieu", "how d'ye do", "prithee, speak plainly",
}


# Social register. The graded routes are all question-and-answer, so nothing in them
# teaches the model what to do when a visitor simply says "Hello". Period dialogue is
# full of these exchanges; the same context-free rule applies, with generic vocatives
# (sir, madam) allowed since they carry no specific referent.
_VOC = r"(?:,? (?:sir|madam|ma'am|my dear sir|my dear madam|my friend))?"
SOCIAL = {
    "greeting": rf"\b((?:good (?:morning|day|evening|afternoon)|how do you do|well met){_VOC})\b",
    "wellbeing": r"\b((?:very well,? thank you|quite well,? thank you|tolerably well|pretty well|well enough|i am very well|i am quite well|never better))\b",
    "farewell": rf"\b((?:good[- ]bye|good night|farewell|good day to you){_VOC})\b",
    "thanks_reply": r"\b((?:not at all|don't mention it|do not mention it|you are very welcome|you are welcome|with pleasure|by all means))\b",
    "acknowledge": r"\b((?:i see|indeed|just so|quite so|to be sure|very true|no doubt|so it is))\b",
    "pleased": r"\b((?:i am glad to see you|glad to see you|i am happy to see you|delighted to see you|it is good to see you))\b",
    "invite_on": r"\b((?:pray be seated|pray sit down|do sit down|take a chair|pray come in|pray go on|pray continue|go on|say on))\b",
}


def _harvest(limit_bytes=0):
    """Stream excerpts, returning Counters of context-free clauses per category."""
    if not EXCERPTS.exists():
        raise SystemExit(f"corpus not found: {EXCERPTS}")
    patterns = [("deflection", DEFLECTION), ("invitation", INVITATION)]
    patterns += [(name, re.compile(rx, re.IGNORECASE)) for name, rx in SOCIAL.items()]
    found = {name: collections.Counter() for name, _ in patterns}
    read = rows = 0
    with open(EXCERPTS, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            read += len(line)
            if limit_bytes and read > limit_bytes:
                break
            rows += 1
            try:
                text = json.loads(line).get("excerpt") or ""
            except Exception:
                continue
            for kind, rx in patterns:
                for m in rx.finditer(text):
                    if not CLEAN_TAIL.match(text[m.end():m.end() + 3]):
                        continue # a complement follows -> context-bound
                    clause = _normalize(" ".join(m.group(1).split()).lower())
                    if clause:
                        found[kind][clause] += 1
    return found, read, rows


_UTTERANCE = re.compile(r'[“"]([A-Z][^”"]{2,40})[”"]')
_UTTERANCE_OK = re.compile(r"^[A-Z][A-Za-z ,'!?.-]+[.!?]$")


def harvest_utterances(want, limit_bytes, seed=1930):
    """Short standalone quoted utterances: "Praise God!", "Come home."

    These are the odd openers -- grammatical, complete, and no kind of question.
    A visitor types them and the model has no idea what to do, because every
    graded row is a well-formed question. Period dialogue is full of them, so
    they are taken verbatim rather than invented.

    Anything carrying a proper noun is dropped: "Come home, Mr. Weller" is bound
    to a story the visitor is not in.
    """
    rng = random.Random(f"utterances:{seed}")
    out, seen = [], set()
    read = 0
    with open(EXCERPTS, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            read += len(line)
            if limit_bytes and read > limit_bytes:
                break
            if len(out) >= want:
                break
            try:
                text = json.loads(line).get("excerpt") or ""
            except Exception:
                continue
            for m in _UTTERANCE.finditer(text):
                u = " ".join(m.group(1).split())
                words = u.split()
                if not (2 <= len(words) <= 6) or not _UTTERANCE_OK.match(u):
                    continue
                # a capitalized word after the first is a name/place -> story-bound
                if any(w[0].isupper() for w in words[1:] if w[0].isalpha()):
                    continue
                if u.lower() in seen:
                    continue
                seen.add(u.lower())
                out.append(u)
                if len(out) >= want:
                    break
    return out


def harvest_fragments(want, limit_bytes, seed=1930):
    """Collect whole period sentences for the generator to truncate.

    Only a sample is needed, so this reads a slice rather than the whole corpus.
    Interrogatives are kept preferentially: a question cut mid-way ("What is the
    reason that the") is the input most likely to read as book text stopped
    mid-sentence, which is exactly the case that triggers base-model continuation.
    """
    rng = random.Random(f"fragments:{seed}")
    questions, statements = [], []
    read = 0
    with open(EXCERPTS, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            read += len(line)
            if limit_bytes and read > limit_bytes:
                break
            if len(questions) >= want and len(statements) >= want:
                break
            try:
                text = json.loads(line).get("excerpt") or ""
            except Exception:
                continue
            for sent in _SENT_SPLIT.split(text):
                sent = " ".join(sent.split())
                n = len(sent.split())
                if not (6 <= n <= 22) or not _SENT_OK.match(sent):
                    continue
                bucket = questions if _INTERROGATIVE.match(sent) else statements
                if len(bucket) < want and rng.random() < 0.5:
                    bucket.append(sent)
    return {"question": questions, "statement": statements}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scan-bytes", type=float, default=0,
                    help="stop after N bytes (0 = whole corpus)")
    ap.add_argument("--utterances", type=int, default=0,
                    help="harvest N short standalone utterances into utterance_pool.json and exit")
    ap.add_argument("--fragments", type=int, default=0,
                    help="harvest N sentences per kind into fragment_pool.json and exit")
    ap.add_argument("--fragment-bytes", type=float, default=250e6,
                    help="corpus slice to sample fragments from")
    ap.add_argument("--min-count", type=int, default=1,
                    help="drop clauses seen fewer than this many times")
    args = ap.parse_args()

    if args.utterances:
        pool = harvest_utterances(args.utterances, int(args.fragment_bytes))
        UTTERANCE_FILE.write_text(json.dumps(pool, indent=1) + "\n", encoding="utf-8")
        print(f"utterances: {len(pool):,}")
        for u in pool[:12]:
            print(f"  {u}")
        print(f"\nwrote {UTTERANCE_FILE}")
        return

    if args.fragments:
        pool = harvest_fragments(args.fragments, int(args.fragment_bytes))
        FRAGMENT_FILE.write_text(json.dumps(pool, indent=1) + "\n", encoding="utf-8")
        print(f"questions: {len(pool['question']):,}  statements: {len(pool['statement']):,}")
        for s in pool["question"][:4]:
            print(f"  Q  {s}")
        for s in pool["statement"][:3]:
            print(f"  S  {s}")
        print(f"\nwrote {FRAGMENT_FILE}")
        return

    found, read, rows = _harvest(int(args.scan_bytes))
    # Drop OCR wreckage. The corpus still carries long-s misreads ("with pleaſure")
    # and mojibake that ocr_corruption.py catalogues; a clause is short enough that
    # any non-ASCII character in it is corruption rather than legitimate diacritics.
    pool = {
        kind: {c: n for c, n in sorted(counter.items(), key=lambda kv: -kv[1])
               if n >= args.min_count and c not in BLOCKLIST and c.isascii()}
        for kind, counter in found.items()
    }
    POOL_FILE.write_text(json.dumps(pool, indent=2) + "\n", encoding="utf-8")

    print(f"scanned {read/1e6:.0f} MB, {rows:,} excerpts")
    for kind, entries in pool.items():
        print(f"\n{kind}: {len(entries)} distinct, {sum(entries.values()):,} occurrences")
        for clause, n in list(entries.items())[:12]:
            print(f"  {n:6,}  {clause}")
    print(f"\nwrote {POOL_FILE}")


if __name__ == "__main__":
    main()
