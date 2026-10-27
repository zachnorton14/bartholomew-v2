"""Synthetic pass: generate questions from authentic answers (matched pairs).

Produces the fixture used to measure how far generated questions drift from
authentic catechism questions. For each authentic Q&A pair we keep the real
question and generate a synthetic one from the *same* answer, so the two differ
only in provenance — topic and content are held constant. That removes the topic
confound from any downstream authenticity/anachronism filter.

The frozen styles (naive, period, cold, examiner) are the prompting experiment —
each kept exactly as it generated its output file so results stay reproducible.
`working` is the living prompt: the culmination of what the experiments showed,
iterated in place. Generate with it by default; regenerate after each revision by
deleting .synthq_working_state.json first.

Input:  output/filtered/*.json
Output: output/synth/questions_<style>.json
        [{"dataset", "i", "answer", "authentic_q", "synthetic_q"}, ...]

Set environment variables before running:
    export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
    export ANTHROPIC_API_KEY=<your deepseek key>
"""

import asyncio
import json
import random
from pathlib import Path

import anthropic

ROOT = Path(__file__).parent.parent
INPUT_DIR = ROOT / "authentic" / "output" / "filtered"   # authentic pairs feed the register comparison
OUTPUT_DIR = ROOT / "synth" / "output"
STATE_DIR = ROOT / "synth" / "state"
MODEL = "claude-haiku-4-5"  # maps to deepseek via ANTHROPIC_BASE_URL
# The model behind this alias is a reasoning model: it emits thinking blocks that
# count against max_tokens. At 4096 the longer `period` prompt spent the entire
# budget thinking and returned no text block at all (stop_reason=max_tokens,
# blocks=['thinking']). Budget for thinking + JSON, and keep batches small so
# there's less to reason about per call.
MAX_TOKENS = 16384
CONCURRENCY = 20
TOKEN_BUDGET = 800

STYLES = ("naive", "period", "cold", "examiner", "working",
          "scoped", "noblind", "pronoun", "paired", "combo")

NAIVE_PROMPT = """\
You receive answers from educational texts. For each one, write the question that
the answer answers.

Input: JSON array [{"i": 0, "a": "..."}, ...]
Output JSON only: [{"i": 0, "q": "the question"}, ...]
"""

# Few-shot examples are real questions from the authentic corpus, so the model is
# anchored on genuine catechism register rather than a description of it.
PERIOD_PROMPT = """\
You receive answers from pre-1930s educational catechisms. For each one, write the
question that the answer answers.

Match the register of the period catechism exactly: plain, direct, often short,
frequently beginning "What is...", "Why...", "How...", "Of what...". No modern
phrasing, no conversational framing, no meta-language ("summarize", "explain to me",
"key points"). Ask the question a 19th-century schoolbook would ask.

Examples of the target register:
  Do winds never blow regularly?
  What are the winds which blow over the Atlantic and Pacific Ocean called?
  Why are they called trade winds?
  What is to be noted of Huyghens?
  Why is beer flat, if the cask be open too long?
  What is the first one intended (the nearest) called?

Input: JSON array [{"i": 0, "a": "..."}, ...]
Output JSON only: [{"i": 0, "q": "the question"}, ...]
"""

# cold: period register PLUS an explicit anti-echo / anti-template instruction.
# The period prompt fixed register but pushed echo up (56%) because a terse
# catechism question strips down to the answer's own key terms. This asks the
# model to write as if it had NOT seen the answer, targeting the 34% authentic
# echo baseline and the varied openers authentic questions use.
COLD_PROMPT = """\
You receive answers from pre-1930s educational catechisms. For each one, write the
question that the answer answers.

Write as an examiner who knows the subject but has NOT seen this particular answer.
Ask for the fact from the outside — do not reuse the answer's distinctive words or
its phrasing. If the answer says "effective temperature," ask "how hot"; if it says
"holder in due course," ask about who can recover. The question should make sense to
someone who cannot see the answer.

Match the register of the period catechism exactly: plain, direct, often short. Vary
the opening — not every question begins "What is." Use "Why...", "How...", "Of what...",
"When...", "Have...", "Can...", "What are..." as the fact demands. No modern phrasing,
no conversational framing, no meta-language.

Examples of the target register:
  Do winds never blow regularly?
  Why are they called trade winds?
  How long does an induced current last?
  Of what use is the gastric juice?
  Have the terms Money and Coin the same signification?
  What countries had glass windows first?

Input: JSON array [{"i": 0, "a": "..."}, ...]
Output JSON only: [{"i": 0, "q": "the question"}, ...]
"""

# examiner: reframes the task as setting an exam question rather than reversing an
# answer. A period examiner writes the question first, from knowledge of the topic;
# the scenario/context questions this produces (see the promissory-note authentic
# examples) are structurally unable to echo an answer they were not written from.
EXAMINER_PROMPT = """\
You are a pre-1930s schoolmaster setting oral-examination questions. For each answer
below, write the examination question a master would pose to elicit it.

You are testing whether the pupil knows the subject, not whether they can echo a text.
So pose the question from the topic, not from the answer's wording: give the situation
or the thing to be explained, and let the pupil supply the facts the answer contains.
Never plant the answer's own terms in the question.

Match period examination register: plain and direct, sometimes a short scenario, often
opening "Why...", "How...", "What becomes of...", "Suppose that...", "In what manner...".
No modern phrasing, no conversational framing, no meta-language.

Examples of the target register:
  Why does the effervescence of soda water so soon go off?
  How are most nouns made plural?
  What occurs when the active is inverted to the passive?
  What is the prevailing form of government in Europe?
  A makes his note payable to B, who transfers it for value before maturity to C.
    C indorses it after maturity to D. D sues A. Can he recover?

Input: JSON array [{"i": 0, "a": "..."}, ...]
Output JSON only: [{"i": 0, "q": "the question"}, ...]
"""

# working: the living prompt — the culmination of what the style experiments showed,
# updated and iterated in place (the styles above are frozen so they keep matching
# the outputs they generated). Lineage:
#   - base is cold, the best performer (echo gap +3.8% vs authentic 33.9%, template
#     below authentic, length 96%); examiner framing and varied-opener register kept.
#   - cold's "do not reuse the answer's distinctive words" ban REMOVED: when the
#     distinctive word IS the subject (stearine, split infinitive), the ban made the
#     model guess the referent and sometimes guess wrong (camphor, comma), producing
#     pairs where the answer doesn't answer the question.
#   - fact-leak rule ADDED: cold planted the answer's dates/names/numbers as givens
#     ("What battle was fought in 1346 in which Philip VI was defeated?") — echo
#     metrics miss this because dates and proper nouns slip the content-word filter.
#   - no-retrospection ADDED: period texts are inside their moment ("What gas used
#     in World War I..." is impossible in a pre-1930s catechism).
# If wrong-referent questions persist, the "has NOT seen this particular answer"
# sentence is the next candidate for removal — same pressure direction as the old
# ban, just weaker.
WORKING_PROMPT = """\
You receive answers from pre-1930s educational catechisms. For each one, write the
question that the answer answers.

Write as an examiner who knows the subject but has NOT seen this particular answer.
The question should make sense to someone who cannot see the answer. Do not state
the answer's facts — its dates, names, and numbers — as given information in the
question; those are what the pupil must supply.

Match the register of the period catechism exactly: plain, direct, often short. Vary
the opening — not every question begins "What is." Use "Why...", "How...", "Of what...",
"When...", "Have...", "Can...", "What are..." as the fact demands. No modern phrasing,
no conversational framing, no meta-language, no retrospective framing — write from
within the period, as a contemporary would.

Examples of the target register:
  Do winds never blow regularly?
  Why are they called trade winds?
  How long does an induced current last?
  Of what use is the gastric juice?
  Have the terms Money and Coin the same signification?
  What countries had glass windows first?

Input: JSON array [{"i": 0, "a": "..."}, ...]
Output JSON only: [{"i": 0, "q": "the question"}, ...]
"""

# ---------------------------------------------------------------------------
# Experiment round 2: five variants of `working`, each isolating one change, so
# metric movement is attributable. Frozen once generated, like round 1.
#
# The trade-off they probe: cold's anti-echo ban gave the best echo (+3.8%) but
# caused wrong-referent hallucinations; working removed the ban, fixed the
# hallucinations that were fixable, and echo shot back to +17.8%. The residual
# wrong-referent cases all have answers that never name their subject ("Its use
# is...", "They exhibit movement..."), which no prompt can recover.
#   scoped   — working + anti-echo scoped to the description: naming the subject
#              is allowed, borrowing the answer's descriptive wording is not.
#   noblind  — working MINUS the "examiner who has NOT seen this answer" framing;
#              isolates whether that sentence does anything on its own.
#   pronoun  — working + rule for subjectless answers: don't guess the referent,
#              write the follow-up question a catechism would ask ("Of what use
#              is it?"), keeping the answer's pronoun.
#   paired   — working but the few-shot examples are full answer->question pairs
#              instead of bare questions, showing the mapping rather than the
#              register alone.
#   combo    — scoped + pronoun + paired together: the everything-that-might-work
#              candidate, read against the isolating variants.
# ---------------------------------------------------------------------------

_REGISTER_PARA = """\
Match the register of the period catechism exactly: plain, direct, often short. Vary
the opening — not every question begins "What is." Use "Why...", "How...", "Of what...",
"When...", "Have...", "Can...", "What are..." as the fact demands. No modern phrasing,
no conversational framing, no meta-language, no retrospective framing — write from
within the period, as a contemporary would.
"""

_QUESTION_EXAMPLES = """\
Examples of the target register:
  Do winds never blow regularly?
  Why are they called trade winds?
  How long does an induced current last?
  Of what use is the gastric juice?
  Have the terms Money and Coin the same signification?
  What countries had glass windows first?
"""

_PAIRED_EXAMPLES = """\
Examples (answer -> the question it answers):
  A: Because too much of the carbonic acid gas (produced by fermentation) is suffered to escape.
  Q: Why is beer flat, if the cask be open too long?

  A: Hartshorn, derived from the circumstance of obtaining it from the horn of the hart.
  Q: What is the old name for ammonia?

  A: The rotation of the earth upon its axis.
  Q: What is the cause of the equatorial current?

  A: Modulation is the variation of sounds in speaking, caused by the proper use of tone,
     pitch, force, emphasis, and inflection.
  Q: What is modulation?
"""

_IO_SPEC = """\
Input: JSON array [{"i": 0, "a": "..."}, ...]
Output JSON only: [{"i": 0, "q": "the question"}, ...]
"""

_HEADER = """\
You receive answers from pre-1930s educational catechisms. For each one, write the
question that the answer answers.
"""

_BLIND_PARA = """\
Write as an examiner who knows the subject but has NOT seen this particular answer.
The question should make sense to someone who cannot see the answer. Do not state
the answer's facts — its dates, names, and numbers — as given information in the
question; those are what the pupil must supply.
"""

_FACTLEAK_ONLY_PARA = """\
Do not state the answer's facts — its dates, names, and numbers — as given
information in the question; those are what the pupil must supply.
"""

_SCOPED_PARA = """\
If the answer names its subject, name it plainly in the question — but do not borrow
the answer's descriptive wording. Ask for the description; never restate it.
"""

_PRONOUN_PARA = """\
If the answer does not name its subject (it begins "It is...", "They are...", "Its
use is..."), do not guess what the subject might be. Write the follow-up question a
catechism would ask, keeping the answer's own pronoun: "Of what use is it?", "What
are they?", "How is it obtained?".
"""

SCOPED_PROMPT = "\n".join([_HEADER, _BLIND_PARA, _SCOPED_PARA, _REGISTER_PARA, _QUESTION_EXAMPLES, _IO_SPEC])
NOBLIND_PROMPT = "\n".join([_HEADER, _FACTLEAK_ONLY_PARA, _REGISTER_PARA, _QUESTION_EXAMPLES, _IO_SPEC])
PRONOUN_PROMPT = "\n".join([_HEADER, _BLIND_PARA, _PRONOUN_PARA, _REGISTER_PARA, _QUESTION_EXAMPLES, _IO_SPEC])
PAIRED_PROMPT = "\n".join([_HEADER, _BLIND_PARA, _REGISTER_PARA, _PAIRED_EXAMPLES, _IO_SPEC])
COMBO_PROMPT = "\n".join([_HEADER, _BLIND_PARA, _SCOPED_PARA, _PRONOUN_PARA, _REGISTER_PARA, _PAIRED_EXAMPLES, _IO_SPEC])

PROMPTS = {
    "naive": NAIVE_PROMPT,
    "period": PERIOD_PROMPT,
    "cold": COLD_PROMPT,
    "examiner": EXAMINER_PROMPT,
    "working": WORKING_PROMPT,
    "scoped": SCOPED_PROMPT,
    "noblind": NOBLIND_PROMPT,
    "pronoun": PRONOUN_PROMPT,
    "paired": PAIRED_PROMPT,
    "combo": COMBO_PROMPT,
}


def load_authentic_pairs():
    """Return [(dataset, i, question, answer)] of first-turn pairs from filtered output."""
    pairs = []
    for json_file in sorted(INPUT_DIR.glob("*.json")):
        dataset = json_file.stem
        for i, item in enumerate(json.loads(json_file.read_text())):
            convs = item["conversations"]
            q = next((c["content"] for c in convs if c["role"] == "user"), None)
            a = next((c["content"] for c in convs if c["role"] == "assistant"), None)
            if q and a:
                pairs.append((dataset, i, q, a))
    return pairs


def state_file(style):
    return STATE_DIR / f"synthq_{style}.json"


def load_state(style):
    f = state_file(style)
    return json.loads(f.read_text()) if f.exists() else {}


def save_state(style, state):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_file(style).write_text(json.dumps(state))


def _estimate_tokens(text):
    return len(text) // 4


def _pack_batches(items, token_budget=TOKEN_BUDGET):
    batches, current, current_tokens = [], [], 0
    for item in items:
        tokens = _estimate_tokens(item[3])  # item = (dataset, i, q, a)
        if current and current_tokens + tokens > token_budget:
            batches.append(current)
            current, current_tokens = [], 0
        current.append(item)
        current_tokens += tokens
    if current:
        batches.append(current)
    return batches


def _strip_fence(text):
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rstrip("`").strip()
    return text


async def _generate_batch(client, semaphore, batch, style, state):
    keys = [f"{d}--{i}" for d, i, _, _ in batch]
    payload = json.dumps([{"i": idx, "a": a} for idx, (_, _, _, a) in enumerate(batch)])
    async with semaphore:
        for attempt in range(5):
            try:
                msg = await client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=PROMPTS[style],
                    messages=[{"role": "user", "content": payload}],
                )
                text = next((b.text for b in msg.content if b.type == "text"), "").strip()
                if not text:
                    # No text block at all: report what came back instead of dying on json.loads("")
                    raise ValueError(
                        f"empty response (stop_reason={msg.stop_reason}, "
                        f"blocks={[b.type for b in msg.content]})"
                    )
                if msg.stop_reason == "max_tokens":
                    raise ValueError(f"truncated at max_tokens ({len(batch)} answers in batch)")
                for r in json.loads(_strip_fence(text)):
                    idx = r.get("i")
                    q = (r.get("q") or "").strip()
                    if isinstance(idx, int) and 0 <= idx < len(keys) and q:
                        state[keys[idx]] = {"q": q}
                return
            except anthropic.RateLimitError:
                await asyncio.sleep(2 ** attempt)
            except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
                # Config error, not a transient one: retrying 5x just hides it.
                raise SystemExit(f"\nAPI auth failed: {e}\n"
                                 f"Check ANTHROPIC_API_KEY matches ANTHROPIC_BASE_URL "
                                 f"(a DeepSeek key for api.deepseek.com, an Anthropic key for the default).")
            except json.JSONDecodeError as e:
                if attempt == 4:
                    print(f"  [{style}] batch of {len(batch)} failed: unparseable response: "
                          f"{e}\n    raw[:200]: {text[:200]!r}")
                    return
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                if attempt == 4:
                    print(f"  [{style}] batch of {len(batch)} failed after 5 attempts: "
                          f"{type(e).__name__}: {str(e)[:200]}")
                    return  # leave absent; write_output skips missing
                await asyncio.sleep(2 ** attempt)


async def run_async(pairs, style, state):
    pending = [p for p in pairs if f"{p[0]}--{p[1]}" not in state]
    if not pending:
        print(f"  [{style}] nothing pending")
        return
    batches = _pack_batches(pending)
    print(f"  [{style}] {len(pending)} answers, {len(batches)} batches...")
    done = [0]

    # Each style runs in its own asyncio.run() loop, so the client must be closed
    # here; leaking it lets a later GC try to close sockets on a dead event loop.
    async with anthropic.AsyncAnthropic() as client:
        semaphore = asyncio.Semaphore(CONCURRENCY)

        async def tracked(batch):
            await _generate_batch(client, semaphore, batch, style, state)
            done[0] += len(batch)
            if done[0] % 100 < len(batch):
                save_state(style, state)
                print(f"  [{style}] {done[0]}/{len(pending)}", flush=True)

        await asyncio.gather(*[asyncio.create_task(tracked(b)) for b in batches])
    save_state(style, state)


def write_output(pairs, style, state):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for dataset, i, q, a in pairs:
        result = state.get(f"{dataset}--{i}")
        if not result:
            continue
        items.append({
            "dataset": dataset,
            "i": i,
            "answer": a,
            "authentic_q": q,
            "synthetic_q": result["q"],
        })
    out = OUTPUT_DIR / f"questions_{style}.json"
    out.write_text(json.dumps(items, indent=2, ensure_ascii=False))
    print(f"  [{style}] wrote {len(items)} matched pairs -> {out}")
    return out


def sample_pairs(pairs, n, seed):
    random.seed(seed)
    return random.sample(pairs, min(n, len(pairs)))


async def test_run(pairs, style, seed, size):
    """Generate a small sample and print matched pairs side by side."""
    sample = sample_pairs(pairs, size, seed)
    state = {}
    batches = _pack_batches(sample)
    async with anthropic.AsyncAnthropic() as client:
        semaphore = asyncio.Semaphore(CONCURRENCY)
        await asyncio.gather(*[_generate_batch(client, semaphore, b, style, state) for b in batches])
    for dataset, i, q, a in sample:
        r = state.get(f"{dataset}--{i}")
        print(f"[{dataset}--{i}]")
        print(f"  A         : {a[:110]}")
        print(f"  authentic : {q}")
        print(f"  synthetic : {r['q'] if r else '(failed)'}")
        print()
