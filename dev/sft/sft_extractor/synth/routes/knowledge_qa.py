"""Knowledge-QA route: turn expository excerpts into grounded Q/A rows.

The first content-bound generator. It sources gated excerpts from the pretrain
corpus (synth/corpus.py), keeps the ones the affordance gate tagged `expository`,
and for each one makes a single model call that distills the passage into a
focused question-answer pair:

  - the ANSWER is a VERBATIM quotation (1-2 exact spans) lifted straight from the
    passage — never composed or paraphrased. This is the anachronism guarantee: an
    answer built only from period text cannot contain modern phrasing. Each span is
    verified as a literal substring of the excerpt (verbatim_answer); anything not
    found verbatim is dropped, not repaired. Spans join with an ellipsis, so no
    model-written word ever enters the answer.
  - the QUESTION is in period-schoolbook register, model-composed and self-situating
    (question-side anachronism is handled by a separate filter, not here).

Only the passage grounds the answer — no outside facts — so a fiction or opinion
passage would produce ungrounded claims; those route elsewhere and are excluded
here by taking only `expository` excerpts.

The model also classifies each passage's ACTUAL subject (content-level), since the
metadata category is book-level and any subject can appear in any book. Rows carry
both: `category` (content, from the model) and `book_category` (from the metadata),
with `category_moved` flagging divergence.

Input:  sampled via corpus.sample_excerpts (coverage-weighted across categories)
Output: HF dataset shards knowledge_qa/part-*.jsonl (one row per line)
        {"doc_index","category","book_category","year","prose_score","question","answer"}

Set environment before running (same as the other model passes):
    export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
    export ANTHROPIC_API_KEY=<your deepseek key>
"""

import random

from synth import corpus, engine
# re-exported for routes that still import these from here
from synth.engine import verbatim_answer, _pack_batches, _strip_fence  # noqa: F401

CLASSES = ("knowledge",)          # classifier classes this route sources

SYSTEM = f"""\
You are given a short passage from a pre-1930s book. Work in this order.

1. FIND THE ANSWER FIRST. Choose the single most salient fact, definition, or
   explanation the passage states, and copy it VERBATIM as the answer.
   - Return it as "spans": a list of ONE exact quotation from the passage — or up
     to THREE, only if the answer genuinely lies in separate places. Prefer one;
     never more than three. Choose the SHORTEST span(s) that carry the whole fact.
   - Copy WORD FOR WORD. Do NOT paraphrase, summarize, rewrite, correct, modernize,
     reorder, or add any word not in the passage. Pick a span that reads as a clean,
     quotable fact, so a short exact quotation suffices.
   - Take WHOLE sentences: begin and end each span at a sentence boundary and keep its
     terminal punctuation, so several spans read as continuous prose when joined.

2. THEN WRITE THE QUESTION that this answer answers.
   - The span must CORRECTLY AND COMPLETELY answer your question. Ask precisely the
     question the span answers — never demand a name, number, or identity the span
     does not supply. If the span only characterizes something ("It was the first
     which the army had lost"), ask what was true or notable about it, not WHICH
     specific thing it was. Before finishing, read the span as the answer to your
     question: if it does not fully and correctly answer, rewrite the QUESTION to
     fit the span.
   - Frame the reader: supply the context that makes the span the natural, correct
     answer — the era, place, subject, or situation it belongs to.
   - Plain period-schoolbook register: direct, often "What/Why/How/Of what". No
     modern or conversational phrasing, no meta-language.
   - Write it in pre-1930s English: period vocabulary, spelling, and phrasing. Use
     no word, term, or idiom that came into use after 1930 — it must read as a
     question a period schoolbook or examiner would actually pose.
   - Stands alone. Never mention the source — no "the passage", "the text",
     "according to", "described", "mentioned". Ask as if from general knowledge.
   - Self-situating — the question must make sense to someone who NEVER saw the
     passage. Name the actual subject: the war, battle, place, ruler, work,
     statute, or scripture involved, using your own knowledge of the subject when
     the passage assumes it (e.g. "the Revised Statutes of the United States", "at
     the Battle of Williamsburg", "in the Gospel of Luke"). FORBIDDEN — references
     that mean nothing on their own: catalog, specimen, or figure numbers
     ("specimen 167"); a bare "the statute", "the act", "the gun", "the assembly",
     "the expedition"; or "two specific verses" / "a certain scholar" without
     naming which. If the chosen fact simply cannot be made to stand alone (it
     hinges on an item known only from the passage), pick a DIFFERENT fact from the
     passage that can.
   - Do NOT put the answer, or the answer's distinctive wording, in the question.
     Ask for the fact; do not reveal it.

{engine.OCR_REJECT}

Input: JSON array [{{"i": 0, "text": "..."}}, ...]
Output JSON only:
  [{{"i": 0, "spans": ["exact quotation"], "q": "..."}}, ...]
"""

MAX_SPANS = 3


def source_excerpts(n, seed=0, **_):
    """Knowledge-QA excerpts: the SINGLE-TURN RESERVE slice of the knowledge pool (see
    corpus.knowledge_partition — the leading fraction, overlapping the multiturn route's
    tail slice so a portion is deliberately double-passed as both a single pair and a
    conversation). n falsy or >= slice returns the whole slice; otherwise a seeded
    sample. Requires `classify` write-back to have run."""
    mat = corpus.knowledge_partition("reserve")
    if not n or n >= len(mat):
        return mat
    return random.Random(seed).sample(mat, n)


ROUTE = engine.Route(
    name="knowledge_qa",
    system=SYSTEM,
    source=source_excerpts,
    answer_fn=engine.spans_answer(MAX_SPANS),   # verbatim extraction
    passthrough=("prose_score",),
    extra_body=engine.DISABLE_THINKING,         # no thinking trace: fast, no truncation
)

# Thin delegates so run.py (and tests) keep the familiar module-level API.
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
