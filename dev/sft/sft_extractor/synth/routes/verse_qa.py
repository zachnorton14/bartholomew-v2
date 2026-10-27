"""Verse route: turn verse excerpts into compose-a-verse rows.

Generative, like composition: the QUESTION is an instruction to compose verse on a
subject; the ANSWER is a genuine period stanza lifted VERBATIM (so the model emits real
period verse, never modern pastiche). Sources excerpts the classifier tagged `verse`
(materialized by the verse harvest overlay).

Verse is line-structured, and the engine's verbatim_answer collapses whitespace — which
would flatten a stanza into one line. So this route uses its own answer_fn
(verse_answer) that verifies each span is a verbatim run of the excerpt (matching with
flexible whitespace) but returns the excerpt's ORIGINAL text for that run, LINE BREAKS
PRESERVED. The model may only select and order lines; it can write nothing.

Input:  synth/output/excerpts.jsonl (classes contains "verse")
Output: synth/output/verse_qa.json

Env:
    export OPENCODE_API_KEY=<your opencode Go key>   # or put it in ROOT/.env
"""

import random
import re

from synth import corpus, engine

CLASSES = ("verse",)              # classifier classes this route sources
MAX_SPANS = 2                     # usually one stanza; two only for separate stanzas


def verse_answer(max_spans):
    """answer_fn: verify each span is a verbatim run of the excerpt (whitespace-
    flexible) and return the excerpt's original text for it, LINE BREAKS PRESERVED.
    Rejects anything not found verbatim. Spans join on a blank line (stanza break)."""
    def fn(r, excerpt):
        spans = r.get("spans")
        if spans is None and r.get("a"):
            spans = [r["a"]]
        if not spans or len(spans) > max_spans:
            return None
        out = []
        for s in spans:
            if not isinstance(s, str) or not s.split():
                return None
            pat = r"\s+".join(re.escape(tok) for tok in s.split())
            m = re.search(pat, excerpt)
            if not m:
                return None                          # not verbatim -> reject
            out.append(m.group(0))                   # original text, newlines kept
        return "\n\n".join(out).strip() or None
    return fn


SYSTEM = f"""\
You are given a short passage of pre-1930s VERSE. Turn it into an instruction-and-
completion pair in which the reader is asked to COMPOSE such verse.

1. FIND THE VERSE FIRST. Copy a self-contained run of the verse — a full stanza, or a
   few consecutive lines that stand on their own — VERBATIM as the answer.
   - Return it as "spans": prefer ONE contiguous quotation of the verse, kept line for
     line; use up to {MAX_SPANS} only if you take two separate stanzas.
   - Copy WORD FOR WORD, line for line. Do NOT paraphrase, rewrite, correct, modernize,
     reorder, or add or drop any word. Take the poet's own lines exactly.
   - Skip a run that is garbled by OCR or that cannot stand on its own.

2. THEN WRITE THE INSTRUCTION that asks for this verse.
   - Name the SUBJECT or occasion and ask for verse on it: "Compose a verse upon ...",
     "Write lines on ...", "Give a stanza in praise of ...", "Set down a verse lamenting
     ...". The verse must be a fitting answer to the instruction.
   - Do NOT quote or paraphrase the verse's own wording in the instruction; give only
     its subject and, if apt, its mood or form (a lament, a hymn, a sonnet).
   - Plain period register, pre-1930s English — no word or idiom that came into use
     after 1930.
   - Stands alone. Never mention the source — no "the passage", "the poem above", "the
     poet". Phrase it as a task set from scratch.

If the passage is NOT actually verse (it is prose, a table, an index, or OCR soup), emit
NO item for it.

Input: JSON array [{{"i": 0, "text": "..."}}, ...]
Output JSON only:
  [{{"i": 0, "spans": ["exact verse"], "q": "..."}}, ...]
"""


def source_excerpts(n, seed=0, **_):
    """Verse excerpts from the materialized corpus (classes contains "verse"). n falsy
    or >= pool returns the whole pool; otherwise a seeded random sample. Requires
    `classify` write-back to have run."""
    mat = corpus.load_excerpts(cls=CLASSES, drop_broken_math=True)
    if not n or n >= len(mat):
        return mat
    return random.Random(seed).sample(mat, n)


ROUTE = engine.Route(
    name="verse_qa",
    system=SYSTEM,
    source=source_excerpts,
    answer_fn=verse_answer(MAX_SPANS),          # verbatim verse, line breaks preserved
    passthrough=("prose_score",),
    extra_body=engine.DISABLE_THINKING,         # no thinking trace: fast, no truncation
)


def load_state():
    return engine.load_state(ROUTE)


def save_state(state):
    engine.save_state(ROUTE, state)


def run_async(excerpts, state):
    return engine.run_async(ROUTE, excerpts, state)


def write_output(excerpts, state):
    return engine.write_output(ROUTE, excerpts, state)


def test_run(excerpts):
    return engine.test_run(ROUTE, excerpts)


def sample_run(excerpts, seed=0):
    return engine.sample_run(ROUTE, excerpts, seed)
