"""Opinion route: turn opinion excerpts into stance-taking Q/A rows.

Sources excerpts the classifier tagged `opinion`. The goal is to teach the model to
TAKE A STANCE, not to recall who held one: the QUESTION asks for a judgment on the
subject ("Ought ...?", "Is it right that ...?"), and the ANSWER is the period view
itself, delivered as the answerer's own opinion — extracted VERBATIM, ground and all.
No attribution (no holder, author, or "the passage"); the opinion answers directly.
Content is kept RAW — period views stand as written even where later knowledge judges
them wrong.

Input:  synth/output/excerpts.jsonl (classes contains "opinion")
Output: synth/output/opinion_qa.json

Env:
    export OPENCODE_API_KEY=<your opencode Go key>   # or put it in ROOT/.env
"""

import random

from synth import corpus, engine

CLASSES = ("opinion",)            # classifier classes this route sources
MAX_SPANS = 3

SYSTEM = f"""\
You are given a short passage from a pre-1930s book, tagged as possibly voicing an
OPINION. Build a pair in which the ANSWERER takes a stance: a question asking for a
judgment, answered by the period view itself.

GATE FIRST. Many passages tagged opinion are really plain exposition, narration, or
statement of fact. REJECT any excerpt that does not, even in the slightest, ARGUE a
point or VOICE a judgment — anything a reader could only KNOW, not DISPUTE. Do not
manufacture an opinion out of descriptive or factual prose; if nothing here takes a
side, return NO span for it. Proceed only when the passage genuinely holds a view.

1. FIND THE STANCE. Choose the judgment the passage advances, with the ground it rests
   on where the passage gives it, and copy it VERBATIM as the answer.
   - Return it as "spans": exact quotations stating the view (and its reason). Prefer
     ONE; use up to {MAX_SPANS} only if view and ground lie apart. Fewest, longest.
   - Copy WORD FOR WORD — no paraphrase, summary, correction, or added word.
   - The span must read as a DIRECT assertion of the view — the answer speaks the
     opinion outright. Do NOT take third-person reportage of who held it ("the
     committee found ...", "it was urged that ...", "critics held ..."); take the
     asserted view itself.
   - Take WHOLE sentences: begin and end at a sentence boundary, keep terminal
     punctuation, so several spans read as continuous prose when joined.
   - SUBSTANCE: only a genuine, defensible stance qualifies. If the passage merely
     touches a topic, drops a throwaway remark, or states a plain fact with no
     judgment, return NO span for it.
   - Keep it RAW: period opinions stand as written even where later knowledge judges
     them wrong — do not soften, hedge, or correct.

2. WRITE THE QUESTION as a request for JUDGMENT on the subject — the answer is the
   opinion given as the answerer's OWN.
   - Solicit a JUDGMENT, answerable only by taking a side: "Ought ...?", "Is it right
     (wise, best) that ...?", "Is ... to be believed?", "Is ... to be preferred?",
     "Is ... justified?". NEVER a bare fact question — not "What is ...?", "For what
     purpose ...?", "What effect has ...?", "To what is ... owing?". If the only honest
     question would be factual, the passage failed the GATE — drop it.
   - NO attribution, NO source: never name a holder, speaker, author, school, or say
     "the passage", "the text", "according to". Ask about the SUBJECT ITSELF.
   - Name the subject so the question stands alone to someone who never saw the passage.
   - Plain period register, pre-1930s English — no word or idiom that came into use
     after 1930, no modern or conversational phrasing.
   - Do NOT put the view's distinctive wording in the question; ask for it.

{engine.OCR_REJECT}

Input: JSON array [{{"i": 0, "text": "..."}}, ...]
Output JSON only:
  [{{"i": 0, "spans": ["exact quotation"], "q": "..."}}, ...]
"""


def source_excerpts(n, seed=0, **_):
    """Opinion excerpts from the materialized corpus (classes contains "opinion"). n
    falsy or >= pool returns the whole pool; otherwise a seeded random sample.
    Requires `classify` write-back to have run."""
    mat = corpus.load_excerpts(cls=CLASSES, drop_broken_math=True)
    if not n or n >= len(mat):
        return mat
    return random.Random(seed).sample(mat, n)


ROUTE = engine.Route(
    name="opinion_qa",
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
