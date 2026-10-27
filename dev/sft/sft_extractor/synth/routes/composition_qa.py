"""Generative route: turn specimens of producible forms into compose-this rows.

The fourth content-bound generator, and the first GENERATIVE one. Where narrative asks
the reader to *recount* an episode, this asks the reader to *produce* a piece of a
nameable form — not only formal documents (a statute, a pleading, a letter, a speech, a
prayer, a proclamation) but any recognizable set form the model could be told to write:
an exam question, a definition, a preface, a review, a toast, an epitaph, a caption, the
conclusion of a tale — whole, or a self-contained part. It reads the materialized
excerpt corpus (synth/output/excerpts.jsonl) and keeps the ones the classifier tags
`composition` (a specimen of a producible FORM).

It is still EXTRACTIVE, and that is the whole point: the answer is a genuine period
specimen, so the model learns to EMIT period-authentic prose in its proper form, with
no anachronism risk (it can only select real period text, never compose modern-sounding
text). The one licensed exception, as in stem_reasoning, is OCR cleanup: the model
returns each span as `verbatim` (the exact scanned text, our proof it is real and
located) plus `refurbished` (the same span with OCR artifacts removed so it reads
pristine), and engine.refurbished_spans_answer verifies the refurbished form adds NO
letter the verbatim lacked — only non-letters and stray-character deletions may change.
So spacing, punctuation, stray dots, and mangled figures get cleaned, but no word can
be invented; if a span cannot be made pristine that way, the model rejects the item.
The question is the one thing that differs from the recall routes — it is an
INSTRUCTION to compose ("Draft an act to ...", "Define ...", "Set an examination
question on ...", "Write a fitting close for a tale in which ..."), naming the form and
subject without dictating the wording.

  - the ANSWER is the verbatim-anchored specimen, OCR-cleaned — prefer ONE long
    contiguous span that is the piece (or a self-contained part: a full clause, a full
    letter, a full stanza, a self-standing paragraph). Its verbatim form is verified as
    a literal substring; anything that can't be cleaned without inventing letters is
    dropped.
  - the QUESTION is a compose-instruction: form + subject, period register, standing
    alone, not reproducing the specimen's wording.

Input:  synth/output/excerpts.jsonl (classes contains composition)
Output: HF dataset shards composition_qa/part-*.jsonl (one row per line)
        {"doc_index","category","book_category","year","prose_score","question","answer"}

Env (same as the other model passes):
    export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
    export ANTHROPIC_API_KEY=<your deepseek key>
"""

import random

from synth import corpus, engine

CLASSES = ("composition",)        # classifier classes this route sources
MAX_SPANS = 3                     # prefer one long span; splice only a split document

SYSTEM = f"""\
You are given a short passage from a pre-1930s book that IS, or contains, a SPECIMEN of
a nameable, producible FORM — a piece of writing of a recognizable kind that a person
could be told to write. The form may be a formal document (a statute or act, a legal
pleading, a letter, a speech or resolution, a prayer or devotion, a proclamation, a
contract or like instrument) OR a shorter set form (an examination question, a
definition, a preface or dedication, a review or notice, a toast, an epitaph, a maxim,
a caption, the conclusion of a tale). It may be a WHOLE specimen or a SELF-CONTAINED
PART of one. Turn it into an instruction-and-completion pair in which the reader is
asked to COMPOSE such a piece. Work in this order.

1. FIND THE SPECIMEN FIRST. Identify the piece of a recognizable form in the passage and
   copy it VERBATIM as the answer.
   - Return it as "spans": prefer ONE long contiguous quotation that is the whole
     specimen, or a self-contained part of it (a whole clause, a whole letter, a whole
     stanza, a whole paragraph that stands on its own). Use several exact quotations only
     when it is genuinely split — never more than {MAX_SPANS}, always the fewest, longest
     spans. Begin and end at natural boundaries of the form (a heading, salutation,
     "Whereas", "Resolved, that", "O Lord", a definition's opening term, the first line
     of a closing).
   - Give EACH span as an object with two fields:
       "verbatim"    — the span copied EXACTLY as it appears in the passage, character
                       for character, including any OCR corruption.
       "refurbished" — the SAME span with OCR artifacts cleaned so it reads pristine. If
                       the span is already clean, make "refurbished" identical to it.
   - REPAIR ONLY OCR ARTIFACTS, AND ONLY WHEN CERTAIN. The scan may have introduced
     stray marks, doubled or dropped punctuation, interpolated dots or ellipses, broken
     or run-together spacing, hyphenation split across a line break, or mangled figures.
     In "refurbished" you MAY remove such junk and set spacing, punctuation, digits, and
     symbols right — rejoin "informa- tion" into "information", drop a stray "..." or a
     doubled "= =", close up "28 ; the". Repair only what its correct form plainly and
     unambiguously must be.
   - INVENT NOTHING. Every WORD — indeed every letter — in "refurbished" must already
     stand in "verbatim". You may DELETE a stray character and change only non-letters
     (spacing, punctuation, digits, operators); you may NOT add a letter, spell out a
     missing word, transpose letters, or rewrite phrasing. If a span needs more than
     that to read cleanly — a garbled WORD, a dropped word, anything you are not certain
     of — do NOT patch it: reject the WHOLE item (emit nothing for that index).
     Passages are plentiful; emit a pristine specimen or none.
   - REJECT DESTROYED CONTENT, not just garbled letters. If the specimen's MEANING has
     been lost — an exercise whose fill-in blanks or answers have dropped out, equations
     that no longer balance or resolve, doubled or dangling operators ("= =", "÷ ;"), a
     chain of figures that does not follow — you cannot honestly restore it (that would
     mean inventing the missing values). Emit NO item for it. Only pass a specimen whose
     content is intact and correct.
   - Take WHOLE sentences: begin and end each span at a sentence boundary and keep its
     terminal punctuation, so several spans read as continuous prose when joined.

2. THEN WRITE THE INSTRUCTION that asks for this piece.
   - Name the FORM and the SUBJECT, matching the verb to the form: "Draft an act of
     Parliament to ...", "Compose a letter ...", "Frame a resolution ...", "Write a
     prayer for ...", "Set out a plaintiff's plea that ...", "Define ...", "Set an
     examination question on ...", "Write a preface to ...", "Give a review of ...",
     "Propose a toast to ...", "Compose an epitaph for ...", "Write a fitting close for
     a tale in which ...". It must call for exactly the kind of piece the answer is, on
     its actual subject and occasion. When the answer is a PART, ask for that part (its
     clause, its closing, its single question), not the whole work.
   - Give the instruction the substance the piece needs to be a fitting answer — its
     purpose, parties, subject, or occasion — but do NOT copy the piece's wording or
     dictate its phrasing. Ask for the thing; let the answer supply the words.
   - Plain period register: direct, no modern or conversational phrasing, no
     meta-language. Write it in pre-1930s English — period vocabulary, spelling, and
     phrasing; use no word, term, or idiom that came into use after 1930.
   - Stands alone. Never mention the source — no "the passage", "the text", "the
     author", "as given", "above", "described". Phrase it as a task set from scratch.
   - Do NOT reproduce the piece's distinctive wording in the instruction.

If the passage is NOT a specimen of any nameable, producible form — it is just ordinary
shapeless prose (plain narration, exposition, or argument with no particular form that
one could be asked to write) — emit NO item for it (omit that index).

Input: JSON array [{{"i": 0, "text": "..."}}, ...]
Output JSON only:
  [{{"i": 0, "spans": [{{"verbatim": "...", "refurbished": "..."}}], "q": "..."}}, ...]
"""


def source_excerpts(n, seed=0, **_):
    """Composition excerpts from the materialized corpus, selected by the classifier
    (classes contains "composition") and screened for OCR-destroyed arithmetic — drills
    whose fill-in blanks and equations the scan mangled beyond honest repair (see
    corpus.has_broken_math). n falsy or >= pool returns the whole (screened) pool;
    otherwise a seeded random sample. Requires `classify` write-back to have run."""
    mat = corpus.load_excerpts(cls=CLASSES, drop_broken_math=True)
    if not n or n >= len(mat):
        return mat
    return random.Random(seed).sample(mat, n)


ROUTE = engine.Route(
    name="composition_qa",
    system=SYSTEM,
    source=source_excerpts,
    answer_fn=engine.refurbished_spans_answer(MAX_SPANS),   # verbatim-anchored, OCR cleaned
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
