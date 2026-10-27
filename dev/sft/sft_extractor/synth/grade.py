"""Holistic 0--100 grading for the filtered synthetic Q/A dataset.

The alignment verifier answers a binary question: does the answer answer the
question?  This pass is deliberately broader.  It asks whether the pair is useful
training data for the route that produced it: clear, well-scoped, answerable, and a
good example of the route's intended behavior.

Input is the output of ``synth.verify --filter`` (``verified/<route>/`` on the
Hugging Face dataset).  Rows are copied with a ``score`` field and written to
``graded/<route>/``.  A multi-turn row receives one holistic ``score`` for the complete
conversation.

The pass is resumable.  Pair/conversation judgments are checkpointed locally by route,
and a route is published only after every grading unit has a valid integer score.  This
means a failed API call cannot silently turn into an ungraded row.

The judge response is intentionally tiny: one bare positional integer array, such as
``[87, 92, 76]``.  Object-shaped results, prose, markdown, and any score outside
0--100 are rejected and retried/split rather than accepted.

Examples::

    python3 -m synth.grade --sample 100
    python3 -m synth.grade --model deepseek-chat
    python3 -m synth.grade --route knowledge_qa --force
"""

import argparse
import asyncio
import json
import random
import re
from collections import Counter
from datetime import datetime

from synth import engine, hf_push


ROUTES = (
    "knowledge_qa", "multiturn_qa", "reasoning_qa", "stem_reasoning",
    "narrative_grounded", "narrative_fiction", "opinion_qa", "how_to_qa",
    "verse_qa", "composition_qa", "calibration_qa",
)

SOURCE_PREFIX = "verified"
OUTPUT_PREFIX = "graded"
BATCH = 24
CHAR_BUDGET = 9000
CHECKPOINT_EVERY = 2000
CHECKPOINT_VERSION = 4


ROUTE_BEHAVIOR = {
    "knowledge_qa": "teach focused factual recall from the supplied period passage",
    "reasoning_qa": "teach the model to explain a reason, cause, principle, or inference",
    "stem_reasoning": "teach step-showing quantitative or physical reasoning in clear prose",
    "narrative_grounded": "teach grounded comprehension or retelling of a real narrated episode",
    "narrative_fiction": "teach comprehension of a fictional scene, including its actors and events",
    "composition_qa": "teach following a named writing or document-composition instruction",
    "how_to_qa": "teach a useful period-authentic procedure or method",
    "opinion_qa": "teach stating and grounding a position expressed by the source",
    "verse_qa": "teach producing or continuing verse in the requested form and subject",
    "multiturn_qa": "teach a coherent follow-up exchange that uses the preceding conversation",
    "calibration_qa": "teach honest uncertainty when the source says a factual point is unsettled",
}


ROUTE_PROMPTS = {
    "knowledge_qa": """\
Route rubric — knowledge_qa:
This route teaches the model to retrieve and state knowledge in response to a genuine
request for information. A clean, self-contained question whose answer supplies every
requested fact is fully successful, however simple or concise it is.

The question must ASK for information rather than state it. If the question contains
the proposition, fact, phrase, name, or number that the answer merely repeats, it is
not a useful knowledge-QA example. Do NOT penalize period interrogative constructions
such as "of what," "wherein," or "by what means." They can be excellent when the
answer supplies the information those words request. Penalize only when the question's
own clause already states the answer, or when the answer does not fill the grammatical
or semantic slot being asked for. For example, "What does the author affirm of X?"
answered by "X is Y" can be excellent; but a question that already says "the author
affirms that X is Y" and is answered by "X is Y" has disclosed its answer and should
score at most 35.

Do not reward or punish complexity by itself. A coherent multi-part question is highly
useful when all requested facts belong together and the answer supplies every one. A
short pair is highly useful only when it still asks for meaningful information. Likewise
penalize questions that ask for a name, date, number, or cause the answer never gives,
and answers that only gesture toward the fact.

A direct question requesting two facts, followed by an answer that clearly supplies
both facts, has no route-specific defect. Do not penalize a small amount of harmless
period wording after the requested facts.
""",
    "reasoning_qa": """\
Route rubric — reasoning_qa:
Reward a question that asks why, how, or by what principle a conclusion follows, and
an answer that contains the relevant reasoning rather than only a final assertion.
The chain may be expressed in period prose; do not demand modern terminology or a
formal proof. A concise distinction can be complete reasoning: for example, explaining
that two concepts differ because one is a human act and the other a divine act is much
better than a long case narrative that never states its governing principle. Do not
use answer length as a proxy for quality.

An answer that clearly supplies every premise or contrast needed for the requested
inference is fully successful. An answer that merely asserts the conclusion, narrates
an example without extracting the reason, depends on missing context, or skips a
requested link has a major route-specific defect.
""",
    "stem_reasoning": """\
Route rubric — stem_reasoning:
Reward a question and answer that teach quantitative, mechanical, physical, or
mathematical reasoning step by step. The answer should preserve the problem's given
quantities, relationships, and conclusion in understandable prose. Do not penalize
period notation or repaired formulas. Penalize a bare result, a missing calculation,
or a question whose requested quantity is not resolved.

Fully successful reasoning is readable and self-contained. Missing diagrams may be
harmless when the text defines every needed relation, but figure-dependent directions,
unmatched brackets, destroyed equations, OCR-split terms, or truncated proofs are not
useful reasoning targets.
""",
    "narrative_grounded": """\
Route rubric — narrative_grounded:
Reward a self-contained question about a specific event, actor, motive, action, or
outcome in the narrated episode, with an answer that accurately recounts that point.
The answer need not retell the entire passage, and an intact excerpt that directly
supplies the requested event can be fully successful. Penalize vague prompts,
unsupported interpretation, and answers that merely repeat a nearby detail without
answering what was asked. Penalize meta prefixes such as "Comprehend:" and questions
that summarize most of the answer before asking for it. The selected answer must begin
and end as a coherent passage; mid-sentence openings, abrupt endings, damaged
quotations, and long irrelevant lead-ins are substantive defects.
""",
    "narrative_fiction": """\
Route rubric — narrative_fiction:
Reward a clear comprehension question about the fictional scene and an answer that
identifies the relevant character, action, setting, or consequence. Judge usefulness
as literary-scene comprehension, not factual history. Penalize generic literary
commentary, missing scene grounding, or an answer that does not resolve the prompt.

The answer must be a structurally readable fictional scene or narration. A poem or
dramatic script flattened into prose, repeated opening quotation marks with no closes,
speaker labels damaged by OCR, a passage beginning mid-sentence, or a passage ending
mid-action has a major defect. An intact excerpt can be a fully successful answer when
it directly presents the scene or event requested; do not demand that it be rewritten
as a modern summary. Also penalize "Tell the tale" prompts that already recount nearly
every event and leave the answer only to repeat them.
""",
    "composition_qa": """\
Route rubric — composition_qa:
Reward an actionable instruction to compose a named form, document, letter, prayer,
recipe, verse, or other requested piece, paired with an answer that actually performs
that instruction in the requested subject, voice, and scope. Do not require factual
Q/A wording. Penalize an answer that is merely related prose, omits essential requested
parts, or is an impossible or excessively overloaded composition task.

Check literal task completion. A request for a complete piece cannot be satisfied by a
heading, fragment, quotation about the form, or OCR-damaged label. Penalize internally
confused instructions (for example, asking for a heading that is itself a six-line
poem), and route-mismatched requests that merely ask the model to recall or "state" a
source opinion rather than compose something. A complete, polished, structurally intact
demonstration is fully successful.
""",
    "how_to_qa": """\
Route rubric — how_to_qa:
Reward a practical method question and an answer that gives the procedure in usable
order, including important materials, conditions, or steps present in the source.
Penalize a descriptive statement with no method, missing critical steps, or a question
that asks for a procedure broader than the answer supports. A vague recommendation
such as "make the second attempt like the first" is not a complete method. References
to a missing figure, apparatus, or prior experiment reduce self-containment; OCR-split
words and corrupted measurements are substantive defects. A procedure that a reader
could actually follow from the answer is fully successful.
""",
    "opinion_qa": """\
Route rubric — opinion_qa:
Reward a question that elicits the source's actual position on a defined issue and an
answer that states or grounds that position rather than pretending it is an objective
fact. Period opinions and beliefs are intended content; do not fact-check them.
Penalize a vague topic prompt, a response with no discernible stance, or a question
that asks for a position the answer does not express. A strong example gives both the
stance and its reason. A yes/no answer that merely mirrors loaded wording in the
question is weak, and a definition or procedural recommendation with no contestable
position is route-mismatched. Do not confuse rhetorical style with substantive support.
""",
    "verse_qa": """\
Route rubric — verse_qa:
Reward an instruction that clearly specifies the requested verse's subject, form, or
occasion and an answer that is genuinely verse-like and fulfills that request. Judge
it as a composition example, not ordinary prose Q/A. Penalize prose fragments, a
missing requested subject or form, or a prompt that the answer plainly does not meet.

Poetic structure is part of the target. Preserved lines or clearly recoverable rhythmic
units, complete stanzas, and balanced quotation marks support a high score. Flattened
verse with many stray opening quotes, lost lineation, OCR damage, or an answer ending
mid-poem cannot score highly even when its subject matches the instruction. Do not
reward topical overlap over structural usability.
""",
    "multiturn_qa": """\
Route rubric — multiturn_qa (IMPORTANT: grade the complete conversation as one unit):
Give one holistic score for the trajectory, never a per-turn average. A high score
requires a coherent sequence of exchanges in which:

- the opening question establishes a useful subject and the first answer addresses it;
- later user turns naturally follow from, deepen, clarify, or apply the prior context;
- each assistant answer responds to the current turn while retaining relevant context;
- the conversation progresses rather than repeating isolated single-turn questions;
- the complete transcript is a useful supervised example of sustained, context-aware
  behavior, and it does not end with an unresolved or badly mismatched turn.

Do not require every turn to be equally elaborate. A concise answer can score highly
when it is exactly responsive. Penalize abrupt topic changes, dangling references,
repeated questions, answers aimed at the wrong turn, contradictions, missing
antecedents, or a conversation that is merely several unrelated pairs placed together.
One weak turn should lower the score in proportion to how much it damages the whole
trajectory, not automatically determine the score by itself.

Do not mistake length for progression. Five standalone fact questions about one topic
are weaker multiturn training than two clean exchanges where the second naturally uses
the first answer. Pronouns and phrases such as "this result" or "what happened next"
are positive only when their antecedents are clear. A simple two-exchange sequence is
fully successful when both exchanges are correct and the second genuinely depends on
or deepens the first.
""",
    "calibration_qa": """\
Route rubric — calibration_qa:
Reward a question that targets a definite factual point and an answer that honestly
states the source's uncertainty, dispute, or inability to determine it. The desired
behavior is calibrated non-assertion, not a confident guess. Penalize a question that
asks for adjacent known facts, an answer that resolves the uncertainty, or uncertainty
that is merely emotional, descriptive, or unrelated to the question. A clean factual
question whose answer directly and explicitly says the fact is unknown or unsettled is
fully successful. A generic hedge, personal doubt, future uncertainty, or unknown
mathematical variable is not the intended behavior.
""",
}


def _system_for_route(route):
    """Compose the shared rubric with the route-specific grading instructions."""
    return SYSTEM.replace("<ROUTE_RUBRIC>", ROUTE_PROMPTS.get(route, ""))


SYSTEM = """\
You grade synthetic supervised-fine-tuning examples. Each ordinary item contains a
QUESTION and an ANSWER, and the route tells you the behavior this example is intended
to teach. A multiturn item contains a complete ordered CONVERSATION instead.
Return one holistic integer score from 0 to 100 for each item.

The score means: how useful would this pair be as training data for the stated route?
Use the whole range when concrete defects warrant it; do not manufacture score spread.

<ROUTE_RUBRIC>

Use one ABSOLUTE scale for every route; never curve, rank, or normalize within a route
or batch. The route rubric defines what successful behavior is, but it does not make
that route's numerical scale stricter or looser. A sample of clean data may legitimately
contain many scores in the 90s. Do not reserve the 90s for ornate, difficult, unusually
long, or unusually impressive examples.

Silently begin at 97 for a fully successful example, identify concrete defects, and
deduct according to their effect on training usefulness:

- 95--100: fully successful and free of meaningful defects. Use 100 only for an
  exceptionally clean exemplar; ordinary complete success is about 97.
- 90--94: fully successful with one very small defect (roughly a 3--7 point deduction).
- 80--89: clearly good and useful, but with one moderate or several minor defects
  (roughly an 8--17 point deduction).
- 65--79: useful but materially limited by a noticeable omission, ambiguity, route
  weakness, or structural problem (roughly an 18--32 point deduction).
- 40--64: weak; at least one major defect substantially teaches the wrong behavior or
  damages the target (roughly a 33--57 point deduction).
- 0--39: unusable or harmful because of mismatch, corruption, truncation, or failure to
  perform the requested behavior.

Do not lower a fully successful example merely because it is simple, short, old-
fashioned, or less interesting than other items. Likewise, do not raise an incomplete
example because it is long, sophisticated, or literary. Before choosing the score,
silently name the defects that justify every deduction; if there is no concrete defect,
keep the score at 95 or above. Output only the scores, never that analysis.

Judge the pair, not the reputation of the source. Do not fact-check a period claim
against modern knowledge: a period answer can be useful even when modern knowledge
would disagree. Do not reward long questions or long answers merely for being long.
Do not penalize period spelling, grammar, beliefs, or vocabulary when they are
intentional and readable. Penalize OCR corruption, unexplained fragments, references
that cannot be resolved, and questions that ask for more than the answer supplies.

Some inputs include QUALITY_WARNINGS from deterministic checks. Verify them against the
text, but treat confirmed unmatched delimiters, OCR-split words, missing-figure
dependencies, answer leakage, and abrupt truncation as defects in the training target,
not harmless period style.

For multiturn items, judge the COMPLETE CONVERSATION as one training example, not its
individual question/answer exchanges. Consider whether the turns form a coherent,
useful progression; whether follow-up questions use the prior context naturally; and
whether every assistant response is relevant, well-grounded, and helpful. Give one
score for the whole conversation. Do not average turn scores.
For composition and verse, judge the answer as a demonstration of the requested form,
not as an ordinary factual answer. For calibration, a truthful, well-targeted
non-answer is the desired behavior.

Output ONLY a bare array of integers, one score per input item, in input order.
No JSON objects, keys, labels, prose, markdown, or explanation. Example:
[87, 92, 76]
"""


def _pairs(row):
    """Return (question-with-context, answer) exchanges for a dataset row."""
    convs = row.get("conversations")
    if not convs:
        return [(row.get("question", ""), row.get("answer", ""))]
    out, history = [], []
    for k in range(0, len(convs) - 1, 2):
        user = convs[k].get("content", "")
        answer = convs[k + 1].get("content", "")
        question = ("\n".join(history) + f"\nUser: {user}").strip() if history else user
        out.append((question, answer))
        history += [f"User: {user}", f"Assistant: {answer}"]
    return out


def _conversation_text(row):
    """Render a complete multiturn row in the order the training example presents it."""
    return "\n".join(
        f"{message.get('role', '').upper()}: {message.get('content', '')}"
        for message in row.get("conversations", [])
    )


def _grade_pairs(route, row):
    """Return grading units; multiturn conversations are intentionally atomic."""
    if route == "multiturn_qa" and row.get("conversations"):
        return [(_conversation_text(row), "")]
    return _pairs(row)


_WORD = re.compile(r"[A-Za-z]{2,}")
_FIGURE_REF = re.compile(
    r"\b(?:as (?:shown|indicated) in (?:the )?|see |shown in )(?:fig(?:ure)?\.?|diagram)\s*\d*",
    re.I,
)
_SPEAKER_TAG = re.compile(r"\b[A-Z][a-z]{1,12}\.\s+(?=[A-Z])")
_CONTENT_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "that", "this", "who", "which", "what", "how", "when", "where", "was",
    "were", "is", "are", "be", "being", "his", "her", "their", "he", "she",
}


def _split_word_artifacts(text):
    """Find OCR-separated words such as ``expe ment`` and ``hori zontal``."""
    from synth.ocr_corruption import _english

    words = _english()
    if not words:
        return []
    found = []
    tokens = list(_WORD.finditer(text))
    for left_match, right_match in zip(tokens, tokens[1:]):
        between = text[left_match.end():right_match.start()]
        if not between or not between.isspace():
            continue
        left, right = left_match.group().lower(), right_match.group().lower()
        joined = left + right
        split_suffix = right in {"ment", "tion", "sion", "ness", "able", "ible", "ally"}
        if (len(joined) >= 7
                and ((joined in words and (left not in words or right not in words))
                     or (left not in words and split_suffix))):
            found.append(text[left_match.start():right_match.end()])
    return sorted(set(found))


def _quality_warnings(route, question, answer):
    """High-precision structural defects supplied to the judge and used for score caps."""
    text = answer or question
    warnings = []

    if text.count("[") != text.count("]"):
        warnings.append("unbalanced square brackets")
    if text.count("(") != text.count(")"):
        warnings.append("unbalanced parentheses")
    if text.count("“") != text.count("”"):
        warnings.append("unbalanced curly quotation marks")
    if text.count('"') % 2:
        warnings.append("unbalanced straight quotation marks")
    if (route == "narrative_fiction" and '\n' not in text
            and len(_SPEAKER_TAG.findall(text)) >= 2):
        warnings.append("dramatic speaker labels flattened into prose")

    split_words = _split_word_artifacts(text)
    if split_words:
        warnings.append("probable OCR-split words: " + ", ".join(split_words[:3]))

    if (route in ("stem_reasoning", "how_to_qa")
            and _FIGURE_REF.search(question + " " + answer)):
        warnings.append("depends on a missing figure or diagram")
    if text.rstrip().endswith((":", ";", ",", "---", "--")):
        warnings.append("answer appears to end mid-sentence or mid-structure")

    if route in ("narrative_grounded", "narrative_fiction"):
        if question.lstrip().lower().startswith("comprehend:"):
            warnings.append("meta prompt prefix: Comprehend")
        q_words = [w for w in re.findall(r"[a-z]+", question.lower())
                   if w not in _CONTENT_STOP]
        a_words = set(re.findall(r"[a-z]+", answer.lower()))
        overlap = sum(w in a_words for w in q_words) / max(1, len(q_words))
        if len(q_words) >= 35 and overlap >= 0.6:
            warnings.append("question over-summarizes the narrative answer")

    if route == "knowledge_qa":
        norm_q = " ".join(re.findall(r"[a-z0-9]+", question.lower()))
        norm_a = " ".join(re.findall(r"[a-z0-9]+", answer.lower()))
        if len(norm_a.split()) >= 5 and norm_a in norm_q:
            warnings.append("the question contains the answer proposition")
    return warnings


def _warning_cap(warnings):
    """Maximum defensible score for objective structural defects."""
    cap = 100
    names = " | ".join(warnings)
    if "question contains the answer proposition" in names:
        cap = min(cap, 35)
    if "unbalanced square brackets" in names or "unbalanced parentheses" in names:
        cap = min(cap, 40)
    if "unbalanced curly quotation marks" in names or "unbalanced straight quotation marks" in names:
        cap = min(cap, 49)
    if "dramatic speaker labels flattened into prose" in names:
        cap = min(cap, 60)
    if "probable OCR-split words" in names:
        cap = min(cap, 55)
    if "answer appears to end mid-sentence or mid-structure" in names:
        cap = min(cap, 60)
    return cap


def _valid_score(value):
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 100


def _load_checkpoint(path, route, force):
    """Load current checkpoints; discard pre-holistic multiturn checkpoints."""
    if force or not path.exists():
        return {}
    payload = json.loads(path.read_text())
    if (isinstance(payload, dict) and payload.get("version") == CHECKPOINT_VERSION
            and isinstance(payload.get("scores"), dict)):
        return payload["scores"]
    # Older checkpoints predate the current holistic grading, common calibration scale,
    # route-specific rubrics, or deterministic structural caps. They are not comparable.
    return {}


def _save_checkpoint(path, scores):
    path.write_text(json.dumps({"version": CHECKPOINT_VERSION, "scores": scores}))


def _pack(items, char_budget=CHAR_BUDGET, max_pairs=BATCH):
    """Pack by both pair count and content size; a long pair gets its own batch."""
    batches, current, size = [], [], 0
    for item in items:
        content_size = len(item[3]) + len(item[4])
        if current and (size + content_size > char_budget or len(current) >= max_pairs):
            batches.append(current)
            current, size = [], 0
        current.append(item)
        size += content_size
    if current:
        batches.append(current)
    return batches


def _parse_batch(parsed, batch):
    """Validate a bare positional integer array and return (item, score) pairs."""
    if not isinstance(parsed, list) or len(parsed) != len(batch):
        return None
    for score in parsed:
        if not _valid_score(score):
            return None
    return [(item, min(score, _warning_cap(item[6]) if len(item) > 6 else 100))
            for item, score in zip(batch, parsed)]


async def _judge(client, semaphore, route_cfg, batch):
    """Judge a batch, recursively splitting malformed responses until they fit."""
    inputs = []
    for i, item in enumerate(batch):
        record = {"i": i, "route": item[1],
                  "route_behavior": ROUTE_BEHAVIOR.get(item[1], item[1])}
        if len(item) > 5 and item[5]:
            record["conversation"] = item[3]
        else:
            record["question"] = item[3]
            record["answer"] = item[4]
        if len(item) > 6 and item[6]:
            record["quality_warnings"] = item[6]
        inputs.append(record)
    payload = json.dumps(inputs, ensure_ascii=False)
    parsed = await engine._call(client, semaphore, route_cfg,
                                _system_for_route(batch[0][1]), payload, len(batch))
    result = _parse_batch(parsed, batch)
    if result is not None:
        return result
    if len(batch) <= 1:
        return []
    middle = len(batch) // 2
    left = await _judge(client, semaphore, route_cfg, batch[:middle])
    right = await _judge(client, semaphore, route_cfg, batch[middle:])
    return left + right


def _load_hf_route(route, prefix=SOURCE_PREFIX, fresh=False):
    """Load JSONL route shards from ``prefix/<route>/`` in the HF dataset."""
    from huggingface_hub import hf_hub_download

    api = hf_push._api()
    files = sorted(
        f for f in api.list_repo_files(hf_push.HF_REPO, repo_type=hf_push.HF_REPO_TYPE)
        if f.startswith(f"{prefix}/{route}/") and f.endswith(".jsonl")
    )
    if not files:
        raise FileNotFoundError(f"no {prefix}/{route}/*.jsonl files found in {hf_push.HF_REPO}")
    rows = []
    for filename in files:
        path = hf_hub_download(
            hf_push.HF_REPO, filename, repo_type=hf_push.HF_REPO_TYPE, force_download=fresh
        )
        with open(path, encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def _attach_scores(row, scores, route):
    """Add scores, keeping multiturn conversations atomic."""
    new = dict(row)
    if row.get("conversations") and route != "multiturn_qa":
        new["scores"] = list(scores)
        new["score"] = round(sum(scores) / len(scores)) if scores else 0
    elif row.get("conversations"):
        if len(scores) != 1:
            raise ValueError("multiturn row must have exactly one holistic score")
        new.pop("scores", None)
        new["score"] = scores[0]
    else:
        if len(scores) != 1:
            raise ValueError("single-turn row must have exactly one score")
        new["score"] = scores[0]
    return new


def _grade_row_scores(rows, scores, route):
    """Build graded rows; raise if any grading unit is missing a score."""
    graded = []
    for row_index, row in enumerate(rows):
        pairs = _grade_pairs(route, row)
        row_scores = []
        for pair_index in range(len(pairs)):
            key = f"{row_index}#{pair_index}"
            if key not in scores or not _valid_score(scores[key]):
                raise RuntimeError(f"missing grade for pair {key} (doc_index={row.get('doc_index')})")
            row_scores.append(scores[key])
        graded.append(_attach_scores(row, row_scores, route))
    return graded


def _summary(route, scores):
    values = list(scores.values())
    if not values:
        return f"  {route}: no pairs"
    bands = {
        "90+": sum(s >= 90 for s in values),
        "75-89": sum(75 <= s < 90 for s in values),
        "50-74": sum(50 <= s < 75 for s in values),
        "<50": sum(s < 50 for s in values),
    }
    unit = "conversations" if route == "multiturn_qa" else "pairs"
    return (f"  {route}: {len(values):,} {unit}, mean {sum(values) / len(values):.1f}, "
            f"90+ {bands['90+']:,}, 75-89 {bands['75-89']:,}, "
            f"50-74 {bands['50-74']:,}, <50 {bands['<50']:,}")


def _short(text, limit=360):
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _review_examples(route, rows):
    examples = []
    for row in rows:
        row_scores = ([row.get("score")] if route == "multiturn_qa"
                      or not row.get("conversations") else row.get("scores"))
        for pair_index, (question, answer) in enumerate(_grade_pairs(route, row)):
            warnings = _quality_warnings(route, question, answer)
            examples.append((row_scores[pair_index], row.get("doc_index"),
                             question, answer, warnings))
    return examples


def _show_examples(route, rows, scores, limit):
    """Print a review slice, lowest scores first, for non-writing sample runs."""
    if limit <= 0:
        return
    examples = []
    for row_index, row in enumerate(rows):
        for pair_index, (question, answer) in enumerate(_grade_pairs(route, row)):
            warnings = _quality_warnings(route, question, answer)
            examples.append((scores[f"{row_index}#{pair_index}"], row.get("doc_index"),
                             question, answer, warnings))
    examples.sort(key=lambda item: item[0])
    unit = "conversations" if route == "multiturn_qa" else "pairs"
    print(f"  {route}: showing {min(limit, len(examples))} graded {unit} (lowest first)")
    for score, doc_index, question, answer, warnings in examples[:limit]:
        print(f"    score={score:>3}  doc_index={doc_index}")
        if warnings:
            print(f"      warnings: {'; '.join(warnings)}")
        if route == "multiturn_qa" and not answer:
            print("      conversation:")
            for line in question.splitlines():
                print(f"        {_short(line)}")
        else:
            print(f"      Q: {_short(question)}")
            print(f"      A: {_short(answer)}")


def _format_sample_example(route, example):
    score, doc_index, question, answer, warnings = example
    lines = ["-" * engine.WRAP,
             engine._wrap("  score    : ", score),
             engine._wrap("  doc_index: ", doc_index)]
    if warnings:
        lines.append(engine._wrap("  warnings : ", "; ".join(warnings)))
    cap = _warning_cap(warnings)
    if cap < 100:
        lines.append(engine._wrap("  hard cap : ", cap))
    if route == "multiturn_qa" and not answer:
        for transcript_line in question.splitlines():
            role, _, content = transcript_line.partition(": ")
            lines.append(engine._wrap(f"  {role:<10}: ", content))
    else:
        lines.append(engine._wrap("  Q        : ", question))
        lines.append(engine._wrap("  A        : ", answer))
    lines.append("")
    return lines


def _best_lines(route, rows, limit=10):
    """Surface the highest-scoring structurally clean candidates first."""
    clean = [example for example in _review_examples(route, rows)
             if not example[4] and example[0] >= 90]
    clean.sort(key=lambda item: item[0], reverse=True)
    lines = [f"--- BEST CLEAN CANDIDATES: {route} ({min(limit, len(clean))}) ---", ""]
    for example in clean[:limit]:
        lines.extend(_format_sample_example(route, example))
    return lines


def _analysis_lines(route, rows):
    examples = _review_examples(route, rows)
    values = sorted(example[0] for example in examples)
    if not values:
        return [f"--- ROUTE SUMMARY: {route} ---", "no graded items", ""]
    middle = len(values) // 2
    median = (values[middle] if len(values) % 2
              else (values[middle - 1] + values[middle]) / 2)
    warning_counts = Counter(warning.split(":", 1)[0]
                             for example in examples for warning in example[4])
    clean = [example for example in examples if not example[4]]
    flagged = [example for example in examples if example[4]]
    hard_capped = [example for example in examples if _warning_cap(example[4]) < 100]

    def subset_stats(label, subset):
        if not subset:
            return f"{label}: 0"
        subset_values = [example[0] for example in subset]
        return (f"{label}: {len(subset)}  mean={sum(subset_values) / len(subset_values):.1f}  "
                f"90+={sum(value >= 90 for value in subset_values)}  "
                f"<50={sum(value < 50 for value in subset_values)}")

    lines = [f"--- ROUTE SUMMARY: {route} ---",
             f"items: {len(values)}  mean: {sum(values) / len(values):.1f}  "
             f"median: {median:g}  range: {values[0]}-{values[-1]}",
             f"bands: 90+={sum(v >= 90 for v in values)}  "
             f"75-89={sum(75 <= v < 90 for v in values)}  "
             f"50-74={sum(50 <= v < 75 for v in values)}  "
             f"<50={sum(v < 50 for v in values)}",
             subset_stats("no deterministic warnings", clean),
             subset_stats("warning-flagged", flagged),
             f"hard-capped by deterministic checks: {len(hard_capped)}"]
    if warning_counts:
        lines.append("structural warnings:")
        lines.extend(f"  {count:>3}  {warning}" for warning, count in warning_counts.most_common())
    else:
        lines.append("structural warnings: none")
    lines.append("")
    return lines


def _sample_lines(route, rows):
    """Format every sampled grading unit for the on-disk review artifact."""
    examples = sorted(_review_examples(route, rows), key=lambda item: item[0])
    unit = "conversations" if route == "multiturn_qa" else "pairs"
    lines = ["=" * engine.WRAP, f"route: {route}  {unit}: {len(examples)}", ""]
    for example in examples:
        lines.extend(_format_sample_example(route, example))
    return lines


def _write_sample(results, model, seed, n):
    """Write a full, wrapped review file using the repository sample convention."""
    lines = ["grade sample", f"model: {model}", f"seed:  {seed}",
             f"n/route: {n}", f"routes: {len(results)}", "",
             "--- GRADING SYSTEM ---", SYSTEM, ""]
    for route, rows in results:
        lines.extend([f"--- ROUTE RUBRIC: {route} ---", ROUTE_PROMPTS[route], ""])
        lines.extend(_analysis_lines(route, rows))
        lines.extend(_best_lines(route, rows))
        lines.extend(_sample_lines(route, rows))
    directory = engine.ROOT / "synth" / "samples" / "grade"
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = directory / f"{timestamp}_seed{seed}_n{n}.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


async def _grade_route(route, rows, model, batch_size, char_budget, write, output_prefix,
                       force=False, show=0):
    route_cfg = engine.Route(
        name="grade",
        system=_system_for_route(route),
        source=None,
        answer_fn=None,
        model=model,
        max_tokens=max(32, batch_size * 5 + 8),
        extra_body=engine.DISABLE_THINKING,
        strict_json=True,
    )
    # (checkpoint key, route, doc_index, question, answer, holistic conversation,
    #  deterministic structural warnings)
    items = []
    for row_index, row in enumerate(rows):
        holistic = route == "multiturn_qa" and bool(row.get("conversations"))
        for pair_index, (question, answer) in enumerate(_grade_pairs(route, row)):
            warnings = _quality_warnings(route, question, answer)
            items.append((f"{row_index}#{pair_index}", route, row.get("doc_index"),
                          question, answer, holistic, warnings))

    checkpoint = engine.STATE_DIR / f"grade_{route}.json"
    scores = _load_checkpoint(checkpoint, route, force) if write else {}
    pending = [item for item in items if item[0] not in scores]
    print(f"{route}: {len(items):,} pairs; {len(pending):,} pending", flush=True)

    done = 0
    saved = 0
    async with engine.open_client() as client:
        semaphore = asyncio.Semaphore(engine.CONCURRENCY)
        batches = _pack(pending, char_budget, batch_size)
        save_lock = asyncio.Lock()

        async def one(batch):
            nonlocal done, saved
            result = await _judge(client, semaphore, route_cfg, batch)
            async with save_lock:
                for item, score in result:
                    scores[item[0]] = score
                done += len(batch)
                if write and done - saved >= CHECKPOINT_EVERY:
                    engine.STATE_DIR.mkdir(parents=True, exist_ok=True)
                    _save_checkpoint(checkpoint, scores)
                    saved = done
                print(f"\r  {route}: {done:,}/{len(pending):,} judged", end="", flush=True)

        if batches:
            await asyncio.gather(*[asyncio.create_task(one(batch)) for batch in batches])
            print()

    if write:
        engine.STATE_DIR.mkdir(parents=True, exist_ok=True)
        _save_checkpoint(checkpoint, scores)
    graded = _grade_row_scores(rows, scores, route)
    print(_summary(route, scores), flush=True)
    if not write:
        _show_examples(route, rows, scores, show)
    if write:
        _, shards, count = hf_push.write_sharded(f"{output_prefix}/{route}", graded)
        checkpoint.unlink(missing_ok=True)
        print(f"  wrote {count:,} rows to {output_prefix}/{route}/ ({shards} shards)", flush=True)
    return graded


async def run(routes, model=engine.MODEL, sample=0, seed=0, batch_size=BATCH,
              char_budget=CHAR_BUDGET, source_prefix=SOURCE_PREFIX,
              output_prefix=OUTPUT_PREFIX, force=False, show=10):
    """Grade routes, optionally sampling rows for a non-writing smoke run."""
    existing = set()
    if not sample:
        api = hf_push._api()
        existing = set(api.list_repo_files(hf_push.HF_REPO, repo_type=hf_push.HF_REPO_TYPE))
    written = []
    sampled = []
    for route in routes:
        if sample:
            try:
                rows = _load_hf_route(route, source_prefix)
            except FileNotFoundError as exc:
                print(f"{route}: skipped ({exc})", flush=True)
                continue
            if len(rows) > sample:
                rows = random.Random(seed).sample(rows, sample)
            graded = await _grade_route(route, rows, model, batch_size, char_budget, False,
                                        output_prefix, force=True, show=show)
            sampled.append((route, graded))
            written.append((route, len(rows)))
            continue
        prefix = f"{output_prefix}/{route}/"
        if any(path.startswith(prefix) and path.endswith(".jsonl") for path in existing) and not force:
            print(f"{route}: already graded ({prefix}), skipping", flush=True)
            continue
        try:
            rows = _load_hf_route(route, source_prefix, fresh=True)
        except FileNotFoundError as exc:
            print(f"{route}: skipped ({exc})", flush=True)
            continue
        await _grade_route(route, rows, model, batch_size, char_budget, True,
                           output_prefix, force=force)
        written.append((route, len(rows)))
    if sample:
        if sampled:
            path = _write_sample(sampled, model, seed, sample)
            print(f"wrote grade sample -> {path}", flush=True)
        else:
            print("no routes were available to sample", flush=True)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--route", choices=ROUTES, help="single route (default: all)")
    parser.add_argument("--model", default=engine.MODEL,
                        help="judge model (default: active provider's model)")
    parser.add_argument("--sample", type=int, default=0, metavar="N",
                        help="grade N rows per route without writing (0 = full pass)")
    parser.add_argument("--show", type=int, default=0, metavar="N",
                        help="grading units to print per route, lowest scores first (0 = none)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=BATCH, help="maximum pairs per call")
    parser.add_argument("--char-budget", type=int, default=CHAR_BUDGET,
                        help="maximum question+answer characters per call")
    parser.add_argument("--source-prefix", default=SOURCE_PREFIX,
                        help="HF folder to read (default: verified)")
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX,
                        help="HF folder to write (default: graded)")
    parser.add_argument("--force", action="store_true",
                        help="re-grade routes that already have output")
    args = parser.parse_args()
    routes = [args.route] if args.route else list(ROUTES)
    asyncio.run(run(routes, args.model, args.sample, args.seed, args.batch, args.char_budget,
                    args.source_prefix, args.output_prefix, args.force, args.show))


if __name__ == "__main__":
    main()
