"""Multiturn route: turn one fact-rich excerpt into an interrelated Q&A conversation.

Genuine two-party period dialogue (catechism/deposition text) is too scarce to build a
multi-turn route on; instead this route is a HYBRID that rides on the huge `knowledge`
pool to synthesize multi-turn exchanges:

  - each USER turn is a model-COMPOSED question (period-schoolbook register), like
    knowledge_qa's question — anachronism on the question side is handled by the
    separate question filter, not here;
  - each ASSISTANT turn is a VERBATIM answer, one or two exact spans lifted straight
    from the excerpt and verified as literal substrings (verbatim_answer). No
    model-written word ever enters an answer, so the answers stay anachronism-safe by
    construction.

The point is MULTI-TURN, not dialogue provenance: the model picks several different
facts from one passage and threads them into an interrelated exchange. Only the FIRST
question must stand on its own (knowledge_qa's self-situating rule); follow-ups are
ALLOWED to lean on the context already established in the conversation ("And what
became of it?", "Why was that so?") — that coreference across turns is exactly the
behavior we want to teach, and it is safe because the antecedent now lives in an
earlier turn.

LENGTH is controlled by us, not the model: each excerpt is assigned a target number of
exchanges by a seeded per-doc RNG (_target_exchanges — a balanced mix over 2..5), the
model is asked for that many, and build_turns truncates any overshoot. This route is
2-EXCHANGE-AND-UP ONLY; single pairs are supplied (and cherry-picked) by the
knowledge_qa reserve, so an excerpt that can't sustain two exchanges is dropped here.

SOURCING: this route draws the MULTITURN SLICE of the knowledge pool
(corpus.knowledge_partition), which overlaps knowledge_qa's single-turn reserve on a
deliberate band — those excerpts are double-passed as both a single pair and a
conversation ("same facts, different forms").

Because the shape is a `conversations` list (not a single question/answer), this route
carries its own run/write loop, reusing the engine's transport, batching, retry,
state, and thinking-off config.

Input:  synth/output/excerpts.jsonl (classes contains "knowledge")
Output: synth/output/multiturn_qa.json
        [{"doc_index","category","year","prose_score","excerpt","conversations":[{role,content}...]}]

Env:
    export OPENCODE_API_KEY=<your opencode Go key>   # or put it in ROOT/.env
"""

import asyncio
import json
import os
import random
from datetime import datetime

from synth import corpus, engine

CLASSES = ("knowledge",)          # classifier classes this route sources
MIN_TURNS = 4                     # at least TWO full exchanges — this route is 2+ only
MAX_SPANS = 2                     # verbatim spans per assistant answer

# Each excerpt is assigned a TARGET number of exchanges (one Q&A pair each), drawn from
# this distribution by a seeded per-doc RNG. The model is asked for that many, and
# build_turns truncates any overshoot — so the length mix is controlled by US, not the
# model's whim. This route NEVER emits a single-exchange row: those are supplied (and
# cherry-picked) by the knowledge_qa reserve, so producing them here would be waste.
# The balanced 30/20/7.5/7.5 mix, renormalized over 2..5: 46/31/11.5/11.5.
def _target_exchanges(doc_index, seed=0):
    x = random.Random(f"{seed}:multiturn:{doc_index}").random()
    if x < 0.4615:                                    # 46% -> 2 exchanges
        return 2
    if x < 0.7692:                                    # 31% -> 3 exchanges
        return 3
    if x < 0.8846:                                    # 11.5% -> 4 exchanges
        return 4
    return 5                                          # 11.5% -> 5 exchanges

SYSTEM = f"""\
You are given a short fact-rich passage from a pre-1930s book and a target number of
question-and-answer EXCHANGES for it (its "exchanges" field — one exchange is one user
question plus one assistant answer). Build a question-and-answer conversation of that
length, between a curious USER who asks and an ASSISTANT who answers. Work in this order.

1. Pick `exchanges` DIFFERENT facts the passage states — a definition, a cause, a
   number, a consequence, a name. Each fact becomes one assistant answer, so choose
   facts that lie in DISTINCT places in the passage. Use as many as `exchanges` asks
   for; only if the passage genuinely lacks that many distinct facts, use fewer (never
   pad with weak or repeated facts, and never fewer than one).

2. For EACH fact, write the assistant's answer as "spans": a list of ONE exact
   quotation from the passage (up to {MAX_SPANS} only if the fact genuinely lies in
   separate places).
   - Copy WORD FOR WORD. Do NOT paraphrase, summarize, rewrite, correct, modernize, or
     add any word not in the passage. Take WHOLE sentences: begin and end each span at
     a sentence boundary and keep its terminal punctuation.
   - The ASSISTANT ONLY ANSWERS — it never asks a question and never ends on a question.

3. For EACH answer write the USER question ("q") it answers.
   - The span must CORRECTLY AND COMPLETELY answer the question. Do not demand a name,
     number, or identity the span does not supply. Do not reveal the answer, or its
     distinctive wording, in the question.
   - Plain period-schoolbook register, pre-1930s English: period vocabulary, spelling,
     and phrasing. Use no word, term, or idiom that came into use after 1930. Never
     mention the source — no "the passage", "the text", "according to", "mentioned", 
     "the author".
   - The FIRST question must SELF-SITUATE — it must make sense to someone who NEVER saw
     the passage. Name the actual subject: the war, battle, place, ruler, work, statute,
     or scripture involved, using your OWN knowledge of the subject when the passage
     assumes it (e.g. "the Revised Statutes of the United States", "at the Battle of
     Williamsburg", "in the Gospel of Luke"). Frame the reader — supply the era, place,
     subject, or situation that makes the span the natural, correct answer.
     FORBIDDEN in the first question — references that mean nothing on their own:
     catalog, specimen, or figure numbers ("specimen 167"); a bare "the statute", "the
     act", "the gun", "the assembly", "the expedition"; a bare "it", "this", "they",
     "these"; or "two specific verses" / "a certain scholar" without naming which. If
     the fact you picked cannot be made to stand alone (it hinges on an item known only
     from the passage), pick a DIFFERENT fact to open with — one that can.
   - Any LATER question (when `exchanges` > 1) builds on the conversation so far — it
     MAY and SHOULD use natural follow-up phrasing that refers back to what was ALREADY
     ESTABLISHED in an earlier turn ("And why did that happen?", "What became of him
     afterward?", "How was it accomplished?"). But a pronoun or "the ___" is allowed
     ONLY when its antecedent was actually named in a previous turn; never introduce a
     NEW bare reference the conversation has not yet identified. Keep the thread
     coherent: each question should follow from the last answer.

4. Return the exchange as "turns": an ordered list ALTERNATING user, assistant, user,
   assistant …, beginning with a user question and ending with an assistant answer —
   one user+assistant pair per exchange, so `exchanges` pairs (2 x `exchanges` turns),
   or fewer only if the passage lacked enough distinct facts. A user turn carries "q";
   an assistant turn carries "spans". If the passage states no clean fact at all, emit
   NO item for it.

{engine.OCR_REJECT}

Input: JSON array [{{"i": 0, "text": "...", "exchanges": N}}, ...]
Output JSON only:
  [{{"i": 0, "turns": [{{"role": "user", "q": "..."}},
                       {{"role": "assistant", "spans": ["exact quotation"]}},
                       {{"role": "user", "q": "..."}},
                       {{"role": "assistant", "spans": ["exact quotation"]}}]}}, ...]
"""


def build_turns(r, excerpt, max_exchanges):
    """Validate + rebuild the model's turns into a clean alternating conversation,
    capped at `max_exchanges` pairs (this excerpt's target). Takes the longest valid
    leading run: strictly alternating from user, each user turn a non-empty composed
    question, each assistant turn a verbatim answer (spans verified as literal
    substrings). Stops at the cap or the first malformed/non-verbatim turn, drops a
    dangling trailing user turn, and requires at least MIN_TURNS. Returns the turn list
    or None (drop the row)."""
    turns = r.get("turns")
    if not isinstance(turns, list):
        return None
    cap = 2 * max_exchanges
    out, expect = [], "user"
    for t in turns:
        if len(out) >= cap:                          # reached this excerpt's target
            break
        if not isinstance(t, dict) or t.get("role") != expect:
            break                                    # missing/mis-ordered role -> stop
        if expect == "user":
            q = t.get("q") or t.get("content")
            if not isinstance(q, str) or not q.strip():
                break
            content = q.strip()
        else:
            spans = t.get("spans")
            if spans is None and t.get("content"):
                spans = [t["content"]]
            content = engine.verbatim_answer(spans, excerpt, MAX_SPANS)
            if not content:
                break                                # not verbatim -> stop the run here
        out.append({"role": expect, "content": content})
        expect = "assistant" if expect == "user" else "user"
    if len(out) % 2:                                 # drop a dangling trailing question
        out.pop()
    return out if len(out) >= MIN_TURNS else None


def source_excerpts(n, seed=0, **_):
    """Multiturn excerpts: the MULTITURN SLICE of the knowledge pool (see
    corpus.knowledge_partition — the trailing fraction, overlapping knowledge_qa's
    reserve so a portion is deliberately double-passed). n falsy or >= slice returns the
    whole slice; else a seeded sample. Requires `classify` write-back."""
    mat = corpus.knowledge_partition("multiturn")
    if not n or n >= len(mat):
        return mat
    return random.Random(seed).sample(mat, n)


# Config carrier: reuses the engine's Route fields for model / max_tokens / thinking /
# concurrency / batching / state file. answer_fn is unused (custom run below).
ROUTE = engine.Route(
    name="multiturn_qa",
    system=SYSTEM,
    source=source_excerpts,
    answer_fn=lambda *_: None,
    passthrough=("prose_score",),
    extra_body=engine.DISABLE_THINKING,
)


async def _batch(client, semaphore, batch, state):
    keys = [str(it["doc_index"]) for it in batch]
    targets = [_target_exchanges(it["doc_index"]) for it in batch]
    payload = json.dumps([{"i": i, "text": it["excerpt"], "exchanges": targets[i]}
                          for i, it in enumerate(batch)])
    parsed = await engine._call(client, semaphore, ROUTE, SYSTEM, payload, len(batch))
    for r in parsed if isinstance(parsed, list) else []:
        if not isinstance(r, dict):
            continue
        idx = r.get("i")
        if not (isinstance(idx, int) and 0 <= idx < len(keys)):
            continue
        turns = build_turns(r, batch[idx]["excerpt"], targets[idx])
        if turns:
            state[keys[idx]] = {"turns": turns}


async def run_async(excerpts, state, save=True):
    async with engine.open_client() as client:
        semaphore = asyncio.Semaphore(ROUTE.concurrency)
        pending = [e for e in excerpts if str(e["doc_index"]) not in state]
        if not pending:
            print("  nothing pending")
            return
        batches = engine._pack_batches(pending, ROUTE.token_budget)
        print(f"  {len(pending)} excerpts, {len(batches)} batches...")
        done = [0]

        async def tracked(batch):
            await _batch(client, semaphore, batch, state)
            done[0] += len(batch)
            if not save:
                return
            if done[0] % engine.CHECKPOINT_EVERY < len(batch):
                save_state(state)
                print(f"  {done[0]}/{len(pending)}", flush=True)
            if done[0] % engine.HF_PUSH_EVERY < len(batch):
                if engine.flush_shard("multiturn_qa", excerpts, state, _mt_row):
                    save_state(state)

        await asyncio.gather(*[asyncio.create_task(tracked(b)) for b in batches])
    if save:
        save_state(state)


def _mt_row(e, r):
    """Build one multi-turn output row from a state entry, or None if it has no turns.
    Excludes the source excerpt — the SFT set needs just the conversation + metadata."""
    if not (r and r.get("turns")):
        return None
    return {
        "doc_index": e["doc_index"], "category": e.get("category"),
        "year": e.get("year"), "prose_score": e.get("prose_score"),
        "conversations": r["turns"],
    }


def write_output(excerpts, state):
    """Final flush: shard remaining unpushed rows to HF (or write one local file)."""
    if os.environ.get("SFT_OUTPUT_LOCAL"):
        rows = [row for e in excerpts
                if (row := _mt_row(e, state.get(str(e["doc_index"])))) is not None]
        return engine._write_local("multiturn_qa", rows)
    n = engine.flush_shard("multiturn_qa", excerpts, state, _mt_row)
    save_state(state)
    print(f"  final: +{n} multiturn_qa rows; shards under multiturn_qa/ on HF")
    return n


def _conv_lines(excerpts, state):
    out = []
    for e in excerpts:
        r = state.get(str(e["doc_index"]))
        turns = r.get("turns") if r else None
        out.append("=" * engine.WRAP)
        out.append(engine._wrap("", f"[{e.get('category')}]  {e.get('year')}  "
                                    f"turns={len(turns) if turns else 0}"))
        out.append(engine._wrap("  excerpt : ", e["excerpt"][:400] + "..."))
        if turns:
            for t in turns:
                label = "  USER  : " if t["role"] == "user" else "  ASST  : "
                out.append(engine._wrap(label, t["content"]))
        else:
            out.append("  (dropped)")
        out.append("")
    return out


async def test_run(excerpts):
    state = {}
    await run_async(excerpts, state, save=False)
    print("\n".join(_conv_lines(excerpts, state)))


async def sample_run(excerpts, seed=0):
    state = {}
    await run_async(excerpts, state, save=False)
    kept = sum(1 for e in excerpts if (state.get(str(e["doc_index"])) or {}).get("turns"))
    header = [f"route: multiturn_qa", f"model: {ROUTE.model}", f"seed:  {seed}",
              f"n:     {len(excerpts)}", f"kept:  {kept}/{len(excerpts)}", "",
              "--- SYSTEM PROMPT ---", SYSTEM, ""]
    d = engine.ROOT / "synth" / "samples" / "routes" / "multiturn_qa"
    d.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = d / f"{ts}_seed{seed}_n{len(excerpts)}.txt"
    path.write_text("\n".join(header + _conv_lines(excerpts, state)), encoding="utf-8")
    print(f"wrote sample ({kept}/{len(excerpts)} kept) -> {path}")
    return path


def load_state():
    return engine.load_state(ROUTE)


def save_state(state):
    engine.save_state(ROUTE, state)
