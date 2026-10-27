"""Model classifier pass: route every harvested excerpt semantically.

The regex affordance gate (corpus.affordance_label) is a cheap RECALL net — it finds
candidate excerpts but can't make the semantic cuts that matter (grounded vs fiction, a
genuine document vs prose that merely says "whereas", how-to vs description). This pass
reads the materialized corpus and has a model assign each excerpt one or more ROUTE
classes from the fixed taxonomy below, plus a `drop` for unusable spans.

It is the cheapest model pass in the pipeline: the output is a short label list, not
prose, so batches pack hard. It runs once over excerpts.jsonl (resumable), writes the
model labels back as `classes` (multi-label, best-fit first) and `primary` (the highest
claim under CLAIM_ORDER), and leaves the regex `affordance` field untouched so the two
can be compared. Routes migrate from `affordance` to `primary` afterwards.

`primary` follows the priority claim order: rare, specific classes (stem, how-to, verse,
dialogue) claim an excerpt before the elastic catch-alls (knowledge, composition), so a
scarce class is never strip-mined by a class that fits almost anything.

Env:
    export OPENCODE_API_KEY=<your opencode Go key>   # or put it in ROOT/.env
"""

import asyncio
import json
import random
from collections import Counter
from datetime import datetime
from types import SimpleNamespace

from synth import corpus, engine

MODEL = engine.MODEL              # override with a cheap non-reasoning model if available
MAX_TOKENS = 4096                 # no thinking block (disabled below) — labels are tiny
CONCURRENCY = 40                  # I/O-bound: raise freely until the API rate-limits
TOKEN_BUDGET = 2000               # ~9 excerpts/batch; output is tiny, input dominates
# Classification needs no reasoning trace; disabling it is the big speedup (DeepSeek).
DISABLE_THINKING = {"thinking": {"type": "disabled"}}

STATE_FILE = engine.STATE_DIR / "classify.json"

# The routing taxonomy. Order here is documentation; CLAIM_ORDER sets priority.
LABELS = {
    "knowledge":          "states a self-contained fact, definition, or explanation about a subject that stands on its own in the world (a place, person, event, concept, law, or process) — answerable without seeing the passage. Expository: tells what is so or why, not narrating, arguing a chain, or urging a view. The broad fallback class.",
    "reasoning":          "works a CHAIN of inference — from premises through steps to a conclusion — that could be lifted whole as the answer to 'why does it follow that...?'. A real argument the author reasons out, not a single causal aside or a bare assertion.",
    "stem_reasoning":     "DERIVES, proves, or shows WHY a scientific or technical result holds — a worked argument in any exact or applied science (mathematics, physics, astronomy, mechanics, engineering, electricity, chemistry, physiology, and the like). Not a table of measurements, a bare formula, or a description of apparatus.",
    "narrative_grounded": "recounts a REAL episode — a scene or sequence of events that actually happened (a battle, a voyage, an incident from history or memoir) and could be retold. Events unfold in it; not a catalog entry or expository facts about a real person.",
    "narrative_fiction":  "an invented SCENE where events happen — a passage from a novel or tale with action unfolding. Not a character's static reflection, description, or interior monologue.",
    "opinion":            "asserts a defensible STANCE on a debatable question — a judgment the writer holds or argues about what is right, good, wise, or best. Not a passing evaluative word or a plain fact.",
    "how_to":             "gives the ordered STEPS for doing or making something — a method, recipe, process, technique, or drill the reader could follow. Not a description of how a thing works or what it is.",
    "conversational":     "a genuine TWO-party BACK-AND-FORTH whose turns can be lifted cleanly for training — a catechism, a Socratic dialogue, a question-and-answer lesson, where each spoken turn stands on its own. NOT dialogue buried in narrative prose with attributions ('\"...,\" said John' — that is narrative), NOT a play or scene with 3+ speakers, NOT a lone Q/A pair or a monologue. Ask: could these turns become alternating user/assistant messages verbatim? If not, it is not conversational.",
    "composition":        "a specimen of a nameable, producible FORM — whole or a self-contained part — that a model could be told to write: a document (statute, letter, oration, prayer, contract), an exam question, a definition, a preface, a review, a story's conclusion, a toast, an epitaph, a caption, and the like. The answer is that verbatim specimen; the goal is training the model to generate text on demand in whatever form is asked. Tag a genuine instance of a recognizable form, not ordinary prose with no particular shape.",
    "verse":              "poetry or metrical verse — lines, meter, or rhyme.",
    "drop":               "yields no good Q/A for ANY class. OCR gibberish, a bare list/table/index/heading/form, a fragment — AND ALSO otherwise-readable prose with nothing worth extracting: no self-contained fact, episode, argument, procedure, stance, document, verse, or exchange. When unsure a passage is genuinely useful, drop it; on a broad regex-sourced pool this is a substantial share.",
}

# Priority: rarest & most specific first; elastic catch-alls (knowledge, composition)
# last; drop only if nothing else fits. `primary` is the first of these present.
CLAIM_ORDER = [
    "stem_reasoning", "how_to", "verse", "conversational", "reasoning",
    "narrative_grounded", "narrative_fiction", "opinion", "knowledge",
    "composition", "drop",
]

_LABEL_LIST = "\n".join(f'  - "{k}": {v}' for k, v in LABELS.items())

SYSTEM = f"""\
You classify a short passage from a pre-1930s book by what kind of question-answer task
it could source. Choose from these classes ONLY, copied verbatim:
{_LABEL_LIST}

Rules:
- Excerpts are MULTIPURPOSE — one passage often supports several tasks. List EVERY class
  that applies; there is no limit, best fit first.
- But the bar is CONVICTION, not possibility. Include a class only when you are
  absolutely convinced the passage would make a GENUINELY GOOD source for that task — not
  merely a passage the task could be forced onto. When in any doubt about a class, LEAVE
  IT OUT. A short, precise list of certain classes beats a long list of maybes.
- "composition" is a producible-FORM tag, not a catch-all — add it only for a genuine
  specimen of a nameable form (whole or part), never just because prose could be
  reworded. A rare, specific class (stem_reasoning, how_to, verse, conversational) must
  be tagged whenever it truly applies; never omit it in favour of a broader one.
- narrative splits by whether the events are real: "narrative_grounded" for a real
  episode, "narrative_fiction" for an invented one.
- PREFER "drop" over forcing a weak class. If no class is a genuinely good fit — garbage,
  or readable prose with nothing worth extracting — return ["drop"] alone (never combined
  with a class). On a broad pool a substantial share should drop; that is correct.

Input: JSON array [{{"i": 0, "text": "..."}}, ...]
Output JSON only: [{{"i": 0, "classes": ["class", ...]}}, ...]
"""


def load_state():
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state))


def primary(classes):
    """The highest-priority class present, under CLAIM_ORDER."""
    for c in CLAIM_ORDER:
        if c in classes:
            return c
    return "drop"


async def _classify_batch(client, semaphore, batch, state, cfg):
    keys = [str(it["doc_index"]) for it in batch]
    payload = json.dumps([{"i": i, "text": it["excerpt"]} for i, it in enumerate(batch)])
    parsed = await engine._call(client, semaphore, cfg, SYSTEM, payload, len(batch))
    for r in parsed if isinstance(parsed, list) else []:
        if not isinstance(r, dict):
            continue
        idx = r.get("i")
        if not (isinstance(idx, int) and 0 <= idx < len(keys)):
            continue
        classes = [c for c in (r.get("classes") or []) if c in LABELS]
        if classes:
            state[keys[idx]] = classes


async def run_async(records, state, save=True):
    cfg = SimpleNamespace(model=MODEL, max_tokens=MAX_TOKENS, extra_body=DISABLE_THINKING)
    pending = [r for r in records if str(r["doc_index"]) not in state]
    if not pending:
        print("  nothing pending")
        return
    batches = engine._pack_batches(pending, TOKEN_BUDGET)
    print(f"  {len(pending)} excerpts, {len(batches)} batches...")
    done = [0]
    last = [len(state)]               # stored count at the last checkpoint
    stall = [0]                       # consecutive checkpoints with no storage growth
    aborted = asyncio.Event()
    async with engine.open_client() as client:
        semaphore = asyncio.Semaphore(CONCURRENCY)

        async def tracked(batch):
            if aborted.is_set():       # cap detected -> stop issuing work
                return
            await _classify_batch(client, semaphore, batch, state, cfg)
            done[0] += len(batch)
            if done[0] % 400 < len(batch):            # checkpoint every ~400 excerpts
                stored = len(state)
                if save:
                    save_state(state)
                print(f"  {done[0]}/{len(pending)}  (stored {stored:,})", flush=True)
                # Cap detection at the checkpoint level (concurrency-safe): if the stored
                # count hasn't grown across several checkpoints (~3k excerpts) the endpoint
                # is returning empty output — the OpenCode usage cap or a degraded fallback.
                if stored == last[0]:
                    stall[0] += 1
                    if stall[0] >= 8 and not aborted.is_set():
                        aborted.set()
                        print("  ABORTING: no results stored across ~3k excerpts — the "
                              "OpenCode usage cap is likely active. Nothing lost; "
                              "resumable. Re-run after the 5-hour window resets.",
                              flush=True)
                else:
                    stall[0] = 0
                    last[0] = stored

        tasks = [asyncio.create_task(tracked(b)) for b in batches]

        async def _cancel_on_abort():         # abort must actually STOP the run: all
            await aborted.wait()              # tasks are created up front, so setting the
            for t in tasks:                  # flag isn't enough — cancel the pending ones.
                t.cancel()

        watcher = asyncio.create_task(_cancel_on_abort())
        await asyncio.gather(*tasks, return_exceptions=True)
        watcher.cancel()
    if save:
        save_state(state)


async def sample_run(records, seed=0, n=20):
    """Classify a seeded sample and write a dated record with the FULL excerpt beside
    its labels, so a human can verify the classification. Writes to
    synth/samples/classify/; nothing goes to stdout but the path confirmation."""
    sample = random.Random(seed).sample(records, min(n, len(records)))
    state = {}
    await run_async(sample, state, save=False)
    lines = ["classify sample", f"model: {MODEL}", f"seed:  {seed}",
             f"n:     {len(sample)}", ""]
    for r in sample:
        cl = state.get(str(r["doc_index"]))
        head = f"[aff={r.get('affordance')}]  primary={primary(cl) if cl else None}  classes={cl}"
        lines.append("=" * engine.WRAP)
        lines.append(engine._wrap("", head))
        lines.append(engine._wrap("  ", r["excerpt"]))
        lines.append("")
    d = engine.ROOT / "synth" / "samples" / "classify"
    d.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = d / f"{ts}_seed{seed}_n{len(sample)}.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote classify sample ({len(sample)}) -> {path}")
    return path


def write_back(records, state):
    """Merge model labels (`classes`, `primary`) into excerpts.jsonl in place,
    leaving the regex `affordance` field for comparison."""
    for r in records:
        cl = state.get(str(r["doc_index"]))
        if cl:
            r["classes"] = cl
            r["primary"] = primary(cl)
    with corpus.EXCERPTS_FILE.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = sum(1 for r in records if "primary" in r)
    print(f"wrote labels for {n}/{len(records)} excerpts -> {corpus.EXCERPTS_FILE}")


# Row targets per class toward the ~300k goal. Rare classes are take-all / supply-
# capped (we source every one we can); knowledge is the elastic filler that absorbs
# the remainder. Rough starting values — retune against real coverage.
QUOTAS = {
    "knowledge": 90000, "reasoning": 45000, "stem_reasoning": 24000,
    "narrative_grounded": 24000, "narrative_fiction": 24000, "opinion": 24000,
    "how_to": 24000, "conversational": 18000, "composition": 18000, "verse": 9000,
}


def coverage(records):
    """Per-class confirmed supply vs. row target — the harvest-loop dashboard. `have`
    counts any-label membership (an excerpt counts toward every class it carries), so
    it's the pool a route could draw from; `gap` is how much more to source."""
    have = Counter()
    n = 0
    for r in records:
        cl = r.get("classes")
        if not cl:
            continue
        n += 1
        for c in cl:
            have[c] += 1
    if not n:
        print("nothing classified yet")
        return
    goal = sum(QUOTAS.values())
    print(f"coverage over {n:,} classified excerpts (goal {goal:,} rows)\n")
    print(f"  {'class':20} {'have':>7} {'target':>8} {'filled':>7} {'gap':>8}")
    for c in CLAIM_ORDER:
        if c == "drop":
            continue
        t, h = QUOTAS.get(c, 0), have.get(c, 0)
        pct = f"{h*100//t}%" if t else "-"
        print(f"  {c:20} {h:7} {t:8} {pct:>7} {max(t-h,0):8}")
    print(f"\n  drop: {have.get('drop', 0):,}   (unusable)")


# regex affordance -> the model primary it should roughly correspond to, for scoring
_AFF_TO_PRIMARY = {
    "expository": "knowledge", "argument": "reasoning", "opinion": "opinion",
    "procedural": "how_to", "composition": "composition",
    "narrative": ("narrative_grounded", "narrative_fiction"),
    # harvest_stem's heuristic pre-tag: "agreement" here = classifier-confirmed STEM,
    # so the metric reads as the confirmation rate rather than scoring every overlay
    # window as a disagreement.
    "stem_reasoning": "stem_reasoning",
}


def report(records, state):
    labelled = [(r, state.get(str(r["doc_index"]))) for r in records]
    labelled = [(r, cl) for r, cl in labelled if cl]
    if not labelled:
        print("no excerpts classified yet")
        return
    n = len(labelled)
    prim = Counter(primary(cl) for _, cl in labelled)
    print(f"\nclassified {n}/{len(records)} excerpts")
    print("\nprimary class (claim-order winner):")
    for c in CLAIM_ORDER:
        if prim[c]:
            print(f"  {c:20} {prim[c]:5}  {prim[c]*100//n:3d}%")

    allc = Counter(c for _, cl in labelled for c in cl)
    multi = sum(1 for _, cl in labelled if len(cl) > 1)
    print(f"\nmulti-label: {multi}/{n} ({multi*100//n}%) fit >1 class; "
          f"avg {sum(len(cl) for _, cl in labelled)/n:.2f} labels")
    print("any-label coverage (excerpt counts toward each):")
    for c in CLAIM_ORDER:
        if allc[c]:
            print(f"  {c:20} {allc[c]:5}")

    # how did the regex gate do? primary vs mapped affordance
    agree = moved = 0
    conf = Counter()
    for r, cl in labelled:
        aff = r.get("affordance")
        p = primary(cl)
        exp = _AFF_TO_PRIMARY.get(aff)
        ok = (p in exp) if isinstance(exp, tuple) else (p == exp)
        agree += ok
        if not ok:
            moved += 1
            conf[(aff, p)] += 1
    print(f"\nregex gate vs model primary: {agree}/{n} agree ({agree*100//n}%), "
          f"{moved} reclassified")
    print("top regex -> model disagreements:")
    for (aff, p), k in conf.most_common(15):
        print(f"  {str(aff):14} -> {p:20} {k:4}")
