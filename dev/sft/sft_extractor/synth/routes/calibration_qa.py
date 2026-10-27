"""Calibration route: teach calibrated uncertainty from period text that hedges.

Most routes teach the model to answer. This one teaches it to give a truthful
non-answer, in period voice, when the source itself does not settle the fact.

Answers remain verbatim period text. The route first extracts the uncertainty-bearing
sentence, then writes a question targeted at precisely the unresolved point. Keeping
those phases separate prevents the question-writing step from changing the answer's
scope.

Input:  synth/output/excerpts.jsonl, factual excerpts filtered to non-answer passages
Output: HF dataset shards calibration_qa/part-*.jsonl (one row per line)
        {"doc_index","category","book_category","year","prose_score","question","answer"}
"""

import json
import random

from synth import corpus, engine

# Calibration draws from factual/expository classes. knowledge + reasoning are the
# cleanest (fewest incidental "uncertain"s), but they hold only ~15k hedge excerpts —
# too few to reach volume — so the broader factual set is included and the strict
# EXTRACT prompt + answer-hedge gate reject the incidental hedges the wider classes carry.
CLASSES = ("knowledge", "reasoning", "stem_reasoning", "opinion",
           "narrative_grounded", "narrative_fiction", "composition", "how_to")
MAX_SPANS = 1
MIN_PROSE = 0.70


def _eligible(row):
    return (
        set(row.get("classes") or ()) & set(CLASSES)
        and row.get("prose_score", 0) >= MIN_PROSE
        and not corpus.has_broken_math(row["excerpt"])
        and corpus.has_calibration_hedge(row["excerpt"])
    )


def _random_excerpts(n, seed, affordance=None):
    """Sample qualifying JSONL rows by byte offsets for fast small previews.

    `load_excerpts` must parse the entire 926 MB materialized corpus even when the CLI
    asks for twenty rows. Random offsets avoid that cost for normal `--sample` runs;
    the full scan remains the correctness fallback when the qualifying pool is sparse.
    """
    path = corpus.EXCERPTS_FILE
    if not path.exists() or not n:
        return []
    size = path.stat().st_size
    if not size:
        return []
    rng = random.Random(seed)
    out, seen = [], set()
    # The targeted overlay is a small fraction of the full JSONL file, so use more
    # probes when selecting specifically from it; this is still cheap random I/O and
    # avoids falling back to the noisier broad pool during reviews.
    attempts = max(1000, n * (1000 if affordance else 150))
    with path.open("rb") as fh:
        for _ in range(attempts):
            fh.seek(rng.randrange(size))
            fh.readline()                 # discard a partial line
            raw = fh.readline()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            key = str(row.get("doc_index"))
            if key in seen or (affordance and row.get("affordance") != affordance) \
                    or not _eligible(row):
                continue
            seen.add(key)
            out.append(row)
            if len(out) >= n:
                return out
    return out

EXTRACT_SYSTEM = f"""\
You are given short passages from pre-1930s books. Find passages that contain a genuine
FACTUAL NON-ANSWER: the source explicitly says that a fact is not known, cannot be
determined, remains obscure, is disputed, or is otherwise not settled. These examples
teach CALIBRATED UNCERTAINTY rather than confident invention.

For each eligible passage, select the answer FIRST:

- Return exactly one item with "spans": one exact quotation from the passage.
- Take one WHOLE sentence, beginning and ending at sentence boundaries. It must itself
  contain the statement of not-knowing, dispute, or inability to determine.
- Copy WORD FOR WORD. Do not paraphrase, explain, resolve, soften, or add anything.
- Prefer the shortest complete sentence that expresses the factual gap. Do not join it
  to surrounding facts merely to make it longer.
- The sentence must END on the not-knowing — that gap must be the point it closes on.
  REJECT a sentence that, having said a thing is unknown, then pivots to a confident
  conclusion or estimate ("...is unknown, yet must be very large"), or that runs on into
  unrelated matter ("...are unknown, and New Hampshire will pay obedience..."). The whole
  substance of the chosen sentence should be the uncertainty, not a hedge followed by an
  answer. If no such clean sentence exists, emit NO item.
- Do NOT begin the span on a bare conjunction or pronoun ("But how they...", "And,
  although...") whose antecedent lies in an earlier sentence; the answer must stand alone.
- Reject mere personal hesitation or opinion ("I am inclined to believe"), uncertainty
  about a future event, an unknown mathematical quantity, an unknown object or person,
  or a descriptive use such as "uncertain light". Reject passages whose uncertainty is
  only incidental and does not answer a definite factual question.
- If the passage gives several known facts and says only one result is unknown, select
  it only when a question can target that unknown result exactly. Otherwise emit NO item.

Do not write a question in this phase. Return only the extracted answer span and an
optional route category.

{engine.OCR_REJECT}

Input: JSON array [{{"i": 0, "text": "..."}}, ...]
Output JSON only:
  [{{"i": 0, "spans": ["exact quotation"], "category": "..."}}]
"""

QUESTION_SYSTEM = f"""\
You are given a pre-1930s passage and the exact sentence selected from it as a
CALIBRATED NON-ANSWER. Write the one direct question that the selected sentence answers.

- Ask for the precise factual point that the sentence says cannot be known, determined,
  settled, or agreed upon. The question should sound as though a definite answer were
  expected: "What was the cause of ...?", "When was ... founded?", "Who first ...?",
  "How many ...?".
- Do NOT ask "Is it known ...?", "What does the passage say ...?", or any question that
  announces the uncertainty. Do not ask for adjacent facts that the answer does provide;
  target only the unresolved point.
- Reject a condition/consequence pair rather than turning it into a refusal example. For
  instance, if the answer says what happens when a mortgage holder's residence is
  unknown, the question must not ask what happens under that condition; the desired
  answer must decline to supply the residence, cause, date, number, or other fact itself.
- If the selected sentence does not support a direct question about an unresolved fact,
  emit NO item by omitting that index.
- Make the question stand alone. Name the actual disease, place, event, person, work,
  quantity, or result; never leave a bare "it", "this", or "they". Never open with a bare
  demonstrative like "this relief", "that statue", or "the tablet" — name or describe the
  actual thing ("the sculptured tablet of Dacian armour") so the reader who never saw the
  passage knows exactly what is asked.
- Do not copy the answer's distinctive wording into the question — rephrase. If the answer
  says "how much wealth lies hid", do not ask "how much wealth lies hidden"; ask the plain
  underlying question ("What is the value of the buried hoards of primitive man?").
- Never mention the source, author, passage, or text.
- Use plain pre-1930s schoolbook English and no modern idiom.

{engine.OCR_REJECT}

Input: JSON array [{{"i": 0, "passage": "...", "answer": "..."}}, ...]
Output JSON only:
  [{{"i": 0, "q": "..."}}]
"""


def source_excerpts(n, seed=0, **_):
    """Return factual excerpts containing a strict calibration hedge.

    A prose floor removes the worst OCR cases before they consume generation calls.
    Requires `classify` write-back to have run.
    """
    # Byte-offset sampling is ONLY for small previews — its probe budget scales with n
    # (n*1000 for the overlay), so a full/large request must NOT use it (30k -> 30M
    # probes would spin for many minutes). Above the preview size, do the full scan,
    # which reads the corpus once and returns the whole qualifying pool.
    if n and n <= 300:
        fast = _random_excerpts(n, seed, affordance="calibration")
        if len(fast) < n:
            fast += _random_excerpts(n - len(fast), seed + 1)
        if len(fast) >= n:
            return fast
    mat = [r for r in corpus.load_excerpts(cls=CLASSES, min_prose=MIN_PROSE,
                                           drop_broken_math=True)
           if corpus.has_calibration_hedge(r["excerpt"])]
    if not n or n >= len(mat):
        return mat
    return random.Random(seed).sample(mat, n)


_VERBATIM_ANSWER = engine.spans_answer(MAX_SPANS)


def calibration_answer(result, excerpt):
    """Accept only a verbatim answer that itself contains a calibration hedge.

    The source filter only guarantees a hedge somewhere in the excerpt. This second
    gate prevents the model from selecting an adjacent factual or personal sentence.
    """
    answer = _VERBATIM_ANSWER(result, excerpt)
    return answer if answer and corpus.has_calibration_hedge(answer) else None


ROUTE = engine.Route(
    name="calibration_qa",
    system=EXTRACT_SYSTEM,
    source=source_excerpts,
    answer_fn=calibration_answer,
    passthrough=("prose_score",),
    question_system=QUESTION_SYSTEM,
    # Calibration excerpts are short prose; the global 1200-char/4 packing budget
    # makes a preview fan out into dozens of API calls across two phases. 2400 keeps
    # extraction batches small enough for reliable positional decisions while remaining
    # much faster than the original one-request-per-excerpt behavior.
    token_budget=2400,
    max_tokens=4096,
    extra_body=engine.DISABLE_THINKING,
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
