"""How-to route: turn procedural excerpts into method Q/A rows.

Sources excerpts the classifier tagged `how_to` — passages that tell how something is
done or made (a recipe, process, technique, drill). Extractive, like knowledge-QA: the
ANSWER is the verbatim sequence of steps, so the anachronism guarantee holds; the
QUESTION asks how the thing is done. Much of this bucket is the STEM-overlay byproduct
(technical/engineering procedure), already in the pool.

Input:  synth/output/excerpts.jsonl (classes contains "how_to")
Output: synth/output/how_to_qa.json

Env:
    export OPENCODE_API_KEY=<your opencode Go key>   # or put it in ROOT/.env
"""

import random

from synth import corpus, engine

CLASSES = ("how_to",)             # classifier classes this route sources
MAX_SPANS = 4                     # steps are spread out; more than knowledge-QA's

SYSTEM = f"""\
You are given a short passage from a pre-1930s book that tells how something is done or
made — a method, recipe, process, or technique. Work in this order.

1. FIND THE PROCEDURE FIRST. Choose the sequence of steps the passage gives for doing
   or making ONE thing, and copy it VERBATIM as the answer.
   - Return it as "spans": exact quotations from the passage that, read in order, give
     the steps of the method from start to finish. Prefer ONE contiguous span; use up
     to {MAX_SPANS} only because the steps are spread out, always the fewest, longest
     spans that hold the procedure together.
   - Copy WORD FOR WORD. Do NOT paraphrase, summarize, rewrite, correct, modernize,
     reorder, or add any word not in the passage — including connecting words between
     spans. Only select and order the author's own sentences.
   - Take WHOLE sentences: begin and end each span at a sentence boundary and keep its
     terminal punctuation, so several spans read as continuous prose when joined.
   - Do NOT choose spans that lean on a figure, plate, table, or page the reader cannot
     see ("as in Fig. 4", "see p. 20"); pick steps that stand without them, or omit.

2. THEN WRITE THE QUESTION the procedure answers.
   - Ask how the thing is done or made: "How is ... made?", "How does one ...?", "By
     what process is ... prepared?", "What are the steps in ...?". The spans must give
     the COMPLETE method your question asks for.
   - Plain period register, pre-1930s English — no word, term, or idiom that came into
     use after 1930, no modern or conversational phrasing.
   - Stands alone. Never mention the source — no "the passage", "the text", "described",
     "above". Ask as if setting a practical exercise.
   - Self-situating — name the actual thing made or done (the process, dish, material,
     or operation) so the question makes sense to someone who never saw the passage.
     Never leave a bare "this preparation" or "the process".
   - Do NOT put the steps, or their distinctive wording, in the question.

{engine.OCR_REJECT}

Input: JSON array [{{"i": 0, "text": "..."}}, ...]
Output JSON only:
  [{{"i": 0, "spans": ["exact quotation"], "q": "..."}}, ...]
"""


def source_excerpts(n, seed=0, **_):
    """How-to excerpts from the materialized corpus (classes contains "how_to"). n
    falsy or >= pool returns the whole pool; otherwise a seeded random sample.
    Requires `classify` write-back to have run."""
    mat = corpus.load_excerpts(cls=CLASSES, drop_broken_math=True)
    if not n or n >= len(mat):
        return mat
    return random.Random(seed).sample(mat, n)


ROUTE = engine.Route(
    name="how_to_qa",
    system=SYSTEM,
    source=source_excerpts,
    answer_fn=engine.spans_answer(MAX_SPANS),   # verbatim extraction
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
