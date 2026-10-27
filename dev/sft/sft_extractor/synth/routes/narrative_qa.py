"""Narrative route: turn narrated episodes into reading-comprehension Q/A rows.

The third content-bound generator. It reads the materialized excerpt corpus and keeps
the ones the classifier tagged narrative — passages that recount events: a scene, an
episode, an account of what happened. Like knowledge-QA it is EXTRACTIVE — the answer
is a verbatim span of period text, never composed — so the anachronism guarantee holds.

What differs from knowledge-QA is the QUESTION framing, and the real vs invented split
is BAKED IN as two separate routes/prompts rather than a per-item flag the model must
honor inside a mixed batch. The classifier's tags decide which route an excerpt is
sourced into:
  - GROUNDED (narrative_grounded) — a REAL episode. RETELL when it is a matter of record
    the model recognizes ("Recount how Washington crossed the Delaware"); COMPREHEND
    when it does not ("What did the garrison do when ...?"). Framed as fact.
  - FICTION (narrative_fiction, and not grounded) — an INVENTED episode. TELL-A-TALE:
    the question is framed OPENLY AS A STORY ("Tell the tale of ...", "Relate the story
    of ..."), introducing the persons as CHARACTERS, so the answer reads as fiction and
    no invented figure is ever asserted as a real person.

Splitting the prompts makes it structurally impossible to misframe fiction as history:
a fiction excerpt only ever meets the TELL-A-TALE prompt.

Input:  synth/output/excerpts.jsonl (classes contains narrative_grounded/_fiction)
Output: HF dataset shards narrative_grounded/part-*.jsonl and narrative_fiction/part-*.jsonl
        {"doc_index","category","book_category","year","prose_score","question","answer"}

Env:
    export OPENCODE_API_KEY=<your opencode Go key>   # or put it in ROOT/.env
"""

import random

from synth import corpus, engine

MAX_SPANS = 3                     # prefer one long span; splice only a spread account

# --- shared prompt fragments (both prompts share everything but the shape section) ---

_LEAD = f"""\
You are given a short passage from a pre-1930s book that is MEANT to narrate events — a
scene, an episode, an account of what happened.

FIRST, judge whether it actually does. A narrated episode has actors doing or
undergoing things in sequence. If instead the passage is exposition or analysis, a
biographical or catalog entry ("APPLETON, Nathan, manufacturer, was born..."), a list,
table, or section heading, an examination paper, a definition, an argument, or a formal
document, then it has no episode to recount — emit NO item for it (omit that index).
Only proceed when there is a real episode. Better to skip than to force a question the
passage does not narrate.

When there IS an episode, work in this order.

1. FIND THE ANSWER FIRST. Choose the account the passage gives — the narrated episode,
   or the salient action, outcome, or turn within it — and copy it VERBATIM as the
   answer.
   - Return it as "spans": prefer ONE long contiguous quotation that carries the whole
     account. Use several exact quotations only when the answer is genuinely spread out
     — never more than {MAX_SPANS}, always the fewest, longest spans that hold it
     together.
   - Copy WORD FOR WORD. Do NOT paraphrase, summarize, rewrite, correct, modernize,
     reorder within a span, or add any word not in the passage — including connecting
     words between spans. You may only select and order the passage's own sentences.
   - Take WHOLE sentences: begin and end each span at a sentence boundary and keep its
     terminal punctuation, so several spans read as continuous prose when joined.
   - SKIP INTERPOLATED HEADERS AND MARGINS. The scan sometimes drops a running header,
     page number, chapter title, or marginal index into the MIDDLE of the account (e.g.
     "... breadstuffs KY., SW. VA., ... N. GA. [CHAP. XL. and small stores ..."). Never
     let such matter into a span: split the account into spans BEFORE and AFTER the
     intrusion so the junk falls out and the remaining spans read cleanly. If it cannot
     be cleanly excluded that way, emit NO item for that passage."""

_Q_TAIL = """\
   - Plain period register: direct, no modern or conversational phrasing, no
     meta-language. Write it in pre-1930s English — period vocabulary, spelling, and
     phrasing; use no word, term, or idiom that came into use after 1930.
   - Stands alone. Never mention the source — no "the passage", "the text", "the
     story", "the author", "the narrator", "according to", "described", "mentioned".
   - FIRST-PERSON accounts ("I ...", "we ..."): never call the speaker "the narrator",
     "the author", or "the writer". Identify the speaker by a concrete role or name if
     it is knowable ("a French officer under Napoleon", "General Foster"), otherwise
     build the question around the other, nameable party and the situation. Do NOT ask
     "How did the narrator ...?".
   - Self-situating — the reader has NOT seen this passage. Give the question its own
     context: for a named event the name carries it; otherwise SET THE SCENE UP inside
     the question — briefly establish the actors and circumstance the answer depends on
     (introduce them by role or name) and give just enough that the question is
     unambiguous. Never leave a bare "he", "she", "they", "it", or "this" whose
     antecedent the question itself has not supplied, and do not retell the whole scene.
   - Do NOT put the answer, or its distinctive wording, in the question."""

_TAIL = f"""\
{engine.OCR_REJECT}

Input: JSON array [{{"i": 0, "text": "..."}}, ...]
Output JSON only:
  [{{"i": 0, "spans": ["exact quotation"], "q": "..."}}, ...]"""

# --- GROUNDED prompt: a real episode, framed as fact (RETELL / COMPREHEND) ----------

GROUNDED_SYSTEM = f"""\
{_LEAD}

2. THEN WRITE THE QUESTION that this account answers. This episode is a REAL, historical
   event — frame the question as of something that actually happened.
   - If you recognize it as a matter of record (a known event, campaign, voyage, or
     life — "Washington's crossing of the Delaware", "the retreat from Moscow"), NAME it
     from your own knowledge and use RETELL: ask the reader to recount it — "Recount how
     ...", "Tell how ... came about", "Describe what befell ...".
   - If you do NOT recognize the specific event, use COMPREHEND: ask, as of a real
     incident, what happened — "What did ... do when ...?", "How did ... meet ...?",
     "What became of ...?".
   - The span must CORRECTLY AND COMPLETELY answer it — ask exactly what the account
     tells; never demand a name, motive, or detail it does not supply.
{_Q_TAIL}

{_TAIL}
"""

# --- FICTION prompt: an invented episode, framed openly as a story (TELL-A-TALE) -----

FICTION_SYSTEM = f"""\
{_LEAD}

2. THEN WRITE THE QUESTION that this account answers. This episode is FICTION — an
   INVENTED scene from a novel, tale, sketch, or story. Frame the question OPENLY AS A
   STORY to be told, so the answer is plainly understood as fiction and NEVER as
   historical fact.
   - Use TELL-A-TALE: ask the reader to relate the tale — "Tell the tale of ...",
     "Relate the story of ...", "Set forth the story in which ...", "Give the story of
     ...". Introduce the persons AS CHARACTERS IN A STORY ("a young lady named Nancy",
     "one Iris, walking abroad at night"), never as real historical figures.
   - NEVER name the episode as a matter of record, and never phrase it as though it
     truly happened. Do NOT write "How did Nancy come to ...?" (that implies she was a
     real person) — write "Tell the story of a young lady named Nancy who ...".
   - The span must CORRECTLY AND COMPLETELY answer whatever you ask — ask exactly what
     the account tells; never demand a name, motive, or detail it does not supply.
{_Q_TAIL}

{_TAIL}
"""


def _sample(mat, n, seed):
    if not n or n >= len(mat):
        return mat
    return random.Random(seed).sample(mat, n)


def source_grounded(n, seed=0, **_):
    """Grounded narrative: excerpts the classifier tagged narrative_grounded (a real
    episode that may be retold). An excerpt carrying BOTH tags counts as grounded here —
    a matter of record can be recounted — so grounded claims the overlap. n falsy or >=
    slice returns all; else a seeded sample. Requires `classify` write-back."""
    mat = corpus.load_excerpts(cls=("narrative_grounded",), drop_broken_math=True)
    return _sample(mat, n, seed)


def source_fiction(n, seed=0, **_):
    """Fiction narrative: excerpts tagged narrative_fiction but NOT narrative_grounded,
    so fiction and grounded partition the pool with no overlap (grounded wins the both-
    tagged, mirroring the old kind rule). n falsy or >= slice returns all; else a seeded
    sample. Requires `classify` write-back."""
    mat = [r for r in corpus.load_excerpts(cls=("narrative_fiction",), drop_broken_math=True)
           if "narrative_grounded" not in (r.get("classes") or [])]
    return _sample(mat, n, seed)


GROUNDED = engine.Route(
    name="narrative_grounded",
    system=GROUNDED_SYSTEM,
    source=source_grounded,
    answer_fn=engine.spans_answer(MAX_SPANS),   # verbatim extraction
    passthrough=("prose_score",),
    extra_body=engine.DISABLE_THINKING,         # no thinking trace: fast, no truncation
)

FICTION = engine.Route(
    name="narrative_fiction",
    system=FICTION_SYSTEM,
    source=source_fiction,
    answer_fn=engine.spans_answer(MAX_SPANS),
    passthrough=("prose_score",),
    extra_body=engine.DISABLE_THINKING,
)

ROUTES = (GROUNDED, FICTION)      # cmd_narrative drives both passes
