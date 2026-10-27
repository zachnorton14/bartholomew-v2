"""Reasoning route: extract the author's reasoning chain, verbatim.

The second content-bound generator. It reads the materialized excerpt corpus
(synth/output/excerpts.jsonl from `harvest`) and keeps the ones the affordance gate
tagged `argument` — passages that make a case, derive a result, or reason from
premises. These are passages that CONTAIN reasoning (the author does the reasoning),
not passages that can merely be reasoned about — so the answer is *extracted*, not
generated: the same anachronism guarantee as knowledge-QA, and no reliance on the
model to compose vintage-sounding logic.

  - the ANSWER is the author's reasoning chain, copied VERBATIM. Prefer one long
    contiguous span; but because a chain is naturally spread across a passage, the
    model MAY splice several verbatim spans (up to MAX_SPANS) into one coherent
    chain. Splicing is anachronism-safe: every span is verified as a literal
    substring and spans join with an ellipsis, so the model may only select and
    order — never write a connecting word. Capped, and told to prefer the fewest,
    longest spans, so it can't cherry-pick a disjointed chain.
  - the QUESTION stands alone (no passage attached — a visible passage would make
    extraction a degenerate echo), self-situating, "why does .../ how does it
    follow ...". Answer-first: find the chain, then write the question for it.

Input:  synth/output/excerpts.jsonl (affordance == argument)
Output: HF dataset shards reasoning_qa/part-*.jsonl (one row per line)
        {"doc_index","category","book_category","year","prose_score","question","answer"}

Env (same as the other model passes):
    export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
    export ANTHROPIC_API_KEY=<your deepseek key>
"""

import random

from synth import corpus, engine

CLASSES = ("reasoning",)         # classifier classes this route sources
MAX_SPANS = 4                    # a chain is spread out; more than knowledge-QA's 2

# CALL 1 — extract the reasoning chain verbatim (this part has always been good).
EXTRACT_SYSTEM = f"""\
You are given a short passage from a pre-1930s book that argues a point, derives a
result, or reasons from premises. The author's own reasoning is in the text.

EXTRACT THE REASONING CHAIN, VERBATIM. Find where the author reasons from premises to
a conclusion, and copy that chain WORD FOR WORD.
   - Return it as "spans": exact quotations from the passage that, read in order,
     form one coherent chain of reasoning (premises through conclusion). Prefer ONE
     long contiguous span. Use several only because the steps are spread out — never
     more than {MAX_SPANS}, and always the fewest, longest spans that hold the chain
     together.
   - Copy WORD FOR WORD. Do NOT paraphrase, summarize, rewrite, correct, modernize,
     reorder within a span, or add any word not in the passage — including
     connecting words between spans. You may only select and order the author's own
     sentences.
   - Take WHOLE sentences: begin and end each span at a sentence boundary and keep its
     terminal punctuation, so several spans read as continuous prose when joined.
   - WORK AROUND references the reader cannot see — a figure, plate, diagram, table,
     page, section, or note number ("Fig. 36", "as shown in the figure", "see page
     466", "§ 4", "the note on p. 22"). Since you select and order the spans, choose
     ones that carry the reasoning WITHOUT such references, and skip any sentence that
     leans on one — a span that mentions or depends on such a reference must NOT enter
     the answer. If the chain cannot be formed without them, REJECT the item (emit
     nothing for it).

{engine.OCR_REJECT}

Input: JSON array [{{"i": 0, "text": "..."}}, ...]
Output JSON only: [{{"i": 0, "spans": ["..."]}}, ...]
"""

# CALL 2 — write the question, given the passage and the extracted chain (ANSWER).
# Isolated so the framing rules can be tuned without touching extraction.
QUESTION_SYSTEM = """\
You are given a passage from a pre-1930s book and a REASONING CHAIN quoted verbatim
from it — this is the ANSWER. Write the single question that this reasoning chain
answers.

- The chain must CORRECTLY AND COMPLETELY answer your question — ask exactly what
  this reasoning concludes or explains. Do not ask for a step, result, or identity
  the chain does not reach. Read the chain as the answer: if it does not fully
  answer, adjust the question to fit the chain.
- Pose a problem that must be reasoned through, not a fact to recall: "Why does
  ...", "How does it follow that ...", "Why must ...?".
- Write it in pre-1930s English: period vocabulary, spelling, and phrasing. Use no
  word, term, or idiom that came into use after 1930, and no modern conversational
  or academic phrasing — it must read as a question a period examiner would pose.
- The question must STAND ALONE and betray no awareness of a source. Never refer to
  the passage or to whoever wrote it — no "the passage", "the text", "the author",
  "the writer", "the speaker", "the narrator", "the argument", "according to", "as
  described", "as mentioned". Ask about the SUBJECT ITSELF, as a matter of fact in
  the world, to someone who never saw the passage. (Not "Why does the author hold
  that finite beings are on probation?" but "Why must finite beings be on
  probation?")
- SUPPLY THE CONTEXT THE READER NEEDS so the chain's "they / it / this / these" have
  clear antecedents:
    · for a matter of general record (a thinker, doctrine, war, ruler, work,
      statute, scripture), name it from your own knowledge;
    · for particular persons, places, or events described only in the passage,
      introduce them briefly in the question — who they are and the situation.
  Give just enough to make the question unambiguous — do not restate the whole
  scene, and never leave a bare "these beings", "such creatures", or "this process".
- Every proper noun must be one the reader can place: either widely recognizable on
  its own, or introduced with a short descriptor of who or what it is (not a bare
  "Skobeleff" but "the Russian general Skobeleff"). When in doubt, describe by role
  instead of naming.
- Do not mention or lean on any figure, plate, diagram, table, page, or section — the
  reader cannot see them; ask about the subject itself.
- Do NOT reveal the answer, and NEVER echo it: the question must be in your own words
  and must not restate, quote, or paraphrase the chain in whole or in part — never
  return the answer, or any piece of it, as the question. Do not state the conclusion
  or reuse the chain's distinctive wording. Ask for the reasoning; make the reader
  supply it.

Input: JSON array [{"i": 0, "passage": "...", "answer": "..."}, ...]
Output JSON only: [{"i": 0, "q": "..."}, ...]
"""


def source_excerpts(n, seed=0, **_):
    """Reasoning excerpts from the materialized corpus, selected by the classifier
    (classes contains "reasoning"). n falsy or >= pool returns the whole pool;
    otherwise a seeded random sample. Requires `classify` write-back to have run."""
    mat = corpus.load_excerpts(cls=CLASSES, drop_broken_math=True)
    if not n or n >= len(mat):
        return mat
    return random.Random(seed).sample(mat, n)


ROUTE = engine.Route(
    name="reasoning_qa",
    system=EXTRACT_SYSTEM,                       # call 1: extract chain + classify
    question_system=QUESTION_SYSTEM,             # call 2: write the question
    source=source_excerpts,
    answer_fn=engine.spans_answer(MAX_SPANS),    # verbatim chain extraction
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
