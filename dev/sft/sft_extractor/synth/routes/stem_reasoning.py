"""STEM-reasoning route: extract the author's reasoning chain, verbatim and in prose.

Period STEM reasoning lives in the sparse reasoning windows of SCIENCE and
TECHNOLOGY texts (engineering derivations, geometry/surveying exercises, physical
and chemical reasoning). corpus.sample_stem seeks those windows — the ones dense
in reasoning connectives plus quantitative vocabulary — rather than a random one.

Like the other content-bound routes, the answer is *extracted*, not generated — but
with one licensed exception for STEM's OCR. This text mangles equations, exponents,
subscripts, and symbols, and a strictly verbatim answer would either drag that
corruption in or, if such spans were dropped, throw away the quantitative reasoning
that is the whole point. So the model returns each span TWICE — `verbatim` (the exact
OCR text) and `refurbished` (the same span with only its formulas repaired) — and the
engine verifies the verbatim is a true substring of the passage and that the
refurbished form adds no LETTER the verbatim lacked (engine.refurbished_spans_answer).
Only non-letters — digits, subscripts, operators, spacing — may change, so a formula
can be rebuilt but no prose can be invented. What survives is the author's own
reasoning with its equations made clean and readable.

Two calls, as in reasoning_qa:
  - call 1 extracts the reasoning chain (verbatim + refurbished spans);
  - call 2 writes the standalone question the chain answers.

Input:  corpus.sample_stem (SCIENCE + TECHNOLOGY, stem-seeking windows)
Output: HF dataset shards stem_reasoning/part-*.jsonl (one row per line)
        {"doc_index","category","book_category","year","stem_signal","question","answer"}

Env:
    export OPENCODE_API_KEY=<your opencode Go key>   # or put it in ROOT/.env
"""

import random

from synth import corpus, engine

CLASSES = ("stem_reasoning",)    # classifier classes this route sources
MAX_SPANS = 6                    # a chain threads between equations; spread out

# CALL 1 — extract the reasoning chain verbatim, repairing only mangled formulas.
EXTRACT_SYSTEM = f"""\
You are given a short passage from a pre-1930s science or engineering text that
reasons quantitatively or physically. The author's own reasoning is in the text.

EXTRACT THE REASONING CHAIN. Find where the author reasons from a principle to a
conclusion, and copy that chain out.
  - Return it as "spans": the quotations from the passage that, read in order, form
    one coherent chain (principle → application → conclusion). Use as many as the
    reasoning needs — up to {MAX_SPANS}, and always the fewest, longest spans that
    hold the chain together. Extract as much of the reasoning as you can.
  - Give EACH span as an object with two fields:
      "verbatim"    — the span copied EXACTLY as it appears in the passage, character
                      for character, including any garbled formula.
      "refurbished" — the SAME span with only its mathematical or chemical formulas
                      restored to correct standard form. If the span has no formula,
                      make "refurbished" identical to "verbatim".
  - REPAIR ONLY FORMULAS. The OCR has mangled this text's equations, exponents,
    subscripts, symbols, and numbers, and page numbers bleed into the formulas. In
    "refurbished" you MAY fix a corrupted expression back to what it plainly must be
    (e.g. "CuSO,+ZnZnSO,+ Cu" → "CuSO₄ + Zn → ZnSO₄ + Cu"; "9X2" → "9 × 2";
    "P₁ = pressure" kept clean). You MAY change only digits, symbols, operators,
    subscripts/superscripts, and spacing. Write exponents and indices as Unicode
    super/subscript characters (a², cʳ, x₁), NEVER as HTML or markup (no <sup>/<sub>).
  - ADD NO NEW CONTENT. Do NOT paraphrase, summarize, rewrite, reorder, translate, or
    add, remove, or alter any WORD of the prose — not a connecting word, not a gloss,
    not a clarification. Every word in "refurbished" must already stand in "verbatim".
    Repair the formulas; leave the author's prose exactly as written, misspellings and
    all. Only select and order the author's own sentences.
  - Do not build the chain out of a span that hinges on a figure, plate, table, or
    page the reader cannot see ("as in Fig. 36", "see p. 466") — those cannot be
    refurbished into anything meaningful; choose spans that carry the reasoning
    without them.

Input: JSON array [{{"i": 0, "text": "..."}}, ...]
Output JSON only:
  [{{"i": 0, "spans": [{{"verbatim": "...", "refurbished": "..."}}]}}, ...]
"""

# CALL 2 — write the question, given the passage and the extracted chain (ANSWER).
QUESTION_SYSTEM = """\
You are given a passage from a pre-1930s science or engineering text and a REASONING
CHAIN quoted verbatim from it — this is the ANSWER. Write the single question that
this reasoning chain answers, as if setting an exercise.

- The chain must CORRECTLY AND COMPLETELY answer your question — ask exactly what
  this reasoning concludes or explains. Do not ask for a step, a numeric result, or
  an identity the chain does not reach. Read the chain as the answer: if it does not
  fully answer, adjust the question to fit the chain.
- Pose a problem that must be reasoned through, not a fact to recall: "Why does
  ...", "How does it follow that ...", "Why must ...?", "If ..., what happens and
  why?".
- Write it in pre-1930s English: period vocabulary, spelling, and phrasing. Use no
  word, term, or idiom that came into use after 1930, and no modern conversational
  or academic phrasing — it must read as a question a period examiner would pose.
- The question must STAND ALONE and betray no awareness of a source. Never refer to
  the passage or to whoever wrote it — no "the passage", "the text", "the author",
  "the figure", "according to", "as described", "as shown". Ask about the SUBJECT
  ITSELF, as a matter of physical fact, to someone who never saw the passage.
- SUPPLY THE PHYSICAL SETUP the reader needs so the problem is well-posed and the
  chain's "it / this / these" have clear antecedents: name the bodies, quantities,
  or conditions involved (the gas held at constant temperature, the beam loaded at
  its centre, the current through the coil). Give just enough to make the question
  unambiguous — do not restate the whole passage, and never leave a bare "this
  quantity", "such a body", or "the system".
- Every proper noun must be one the reader can place: either widely recognizable on
  its own (Boyle's law, Carnot's cycle), or introduced with a short descriptor.
- Do NOT reveal the answer: do not state the conclusion, and do not reuse the
  chain's distinctive wording. Ask for the reasoning; make the reader supply it.
- Write any mathematical notation as plain Unicode: exponents and indices as Unicode
  superscript/subscript characters (a², xₙ, aᵖ, cʳ), or a caret where no such
  character exists (a^k). NEVER emit HTML or markup — no <sup>, <sub>, or any tag.

Input: JSON array [{"i": 0, "passage": "...", "answer": "..."}, ...]
Output JSON only: [{"i": 0, "q": "..."}, ...]
"""


def source_excerpts(n, seed=0, **_):
    """STEM-reasoning excerpts from the materialized corpus, selected by the classifier
    (classes contains "stem_reasoning") — the classifier-confirmed windows the STEM
    harvest overlay (`harvest --stem`) put into the pool. n falsy or >= pool returns
    the whole pool; otherwise a seeded random sample. Requires `classify` write-back."""
    mat = corpus.load_excerpts(cls=CLASSES)
    if not n or n >= len(mat):
        return mat
    return random.Random(seed).sample(mat, n)


ROUTE = engine.Route(
    name="stem_reasoning",
    system=EXTRACT_SYSTEM,                       # call 1: extract prose chain + classify
    question_system=QUESTION_SYSTEM,             # call 2: write the question
    source=source_excerpts,
    answer_fn=engine.refurbished_spans_answer(MAX_SPANS),   # verbatim spans, repaired formulas
    passthrough=("stem_signal",),
    extra_body=engine.DISABLE_THINKING,         # no thinking trace: fast, no truncation/timeout
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
