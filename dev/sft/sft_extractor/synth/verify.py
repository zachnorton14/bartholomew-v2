"""LLM-judge verification of Q/A ALIGNMENT for the synthetic SFT dataset.

The answers are verbatim period text (correct by period standards), so this does NOT
fact-check them and does NOT judge era/style (that is the anachronism filter's job). It
asks exactly one thing: does the answer actually, correctly, and completely answer the
composed question? Each pair is scored 1-5. Sampling per route gives you the
misalignment RATE cheaply, so you can decide whether a full judge+filter pass is worth
it before spending on one.

Judge with a model OTHER than the deepseek-chat generator for rigor (--model, and/or
SFT_PROVIDER for a different provider) to avoid self-preference — though "does A answer
Q" is fairly objective, so same-family judging still gives a usable signal.

Multiturn: each assistant turn is judged against its user turn WITH the preceding turns
supplied as context, so follow-ups that corefer ("and why did that happen?") are judged
fairly rather than looking context-bare.

Run:
    python3 -m synth.verify --route knowledge_qa --sample 1000
    python3 -m synth.verify --sample 1500                 # every route
"""

import argparse
import asyncio
import json
import random
from collections import defaultdict

from synth import engine
from synth.ngram_filter import _load_route, ROUTES

BATCH = 40                # max pairs per judge call
CHAR_BUDGET = 6000        # cap on q+a chars per batch: long-content routes (multiturn)
                          # get fewer pairs/call automatically, avoiding the model's
                          # degenerate repetition loop on large payloads


def _pack(items, char_budget, max_pairs):
    """Pack items into batches capped by BOTH a char budget (q+a length) and a max pair
    count, so short-content routes fill to max_pairs and long-content routes stay small."""
    batches, cur, sz = [], [], 0
    for it in items:
        c = len(it[2]) + len(it[3])
        if cur and (sz + c > char_budget or len(cur) >= max_pairs):
            batches.append(cur); cur, sz = [], 0
        cur.append(it); sz += c
    if cur:
        batches.append(cur)
    return batches

SYSTEM = """\
You are given a numbered list of QUESTION/ANSWER pairs from pre-1930s books. For each
pair, judge whether the ANSWER is a RIGHT and COMPLETE answer to the QUESTION — not
merely on the same subject.

Be strict about CORRECTNESS. The answer must actually resolve what the question asks —
the right name, number, date, cause, or outcome — completely and without mismatch. Mark
0 if the answer is on-topic but does not correctly or fully answer: it gives the wrong
specific, omits what was asked, only REFERS to the answer ("this was the ...", "is
reported by ...") without stating it, is cut off, or does not actually match the
question.

Do NOT fact-check against modern knowledge: an answer that correctly reflects what a
pre-1930 source states is CORRECT even if modern science disagrees. You judge whether the
answer correctly answers the question, not whether the period belief is true today. Do
NOT judge style or era.

Study these examples, then apply the SAME standard:
  Q: "Who discovered the circulation of the blood?"
  A: "The circulation of the blood was discovered by Harvey."            -> 1  (states it)
  Q: "What was the original constitution of the government of New-Haven?"
  A: "This was the original, fundamental constitution of New-Haven."     -> 0  (only REFERS
     to it; never says what it was)
  Q: "How many men were lost in the assault on the redoubt?"
  A: "The assault was made at dawn and repulsed with heavy loss."        -> 0  (on topic
     but never gives the number asked)
  Q: "In what year did Harvard place its law school on a graduate basis?"
  A: "Harvard placed its school on a graduate basis in 1896."            -> 1  (right specific)
  Q: "By what river did General Wolfe land before the assault on Quebec?"
  A: "General Wolfe landed by the Hudson before the assault on Quebec."  -> 0  (wrong
     specific — the answer names the wrong thing)
  Q: "What did Fleischmann report as the most extraordinary instance of prepotency?"
  A: "The most extraordinary instance of prepotency is reported by Fleischmann." -> 0
     (attributes it but never states what it was)

Output ONLY a bare JSON array of 0/1 — one value per input pair, IN THE SAME ORDER,
nothing else (no keys, no text):
  1 = correctly and completely answers the question
  0 = does not
Return EXACTLY as many values as there are input pairs, e.g. [1,1,0,1,0].

Input: JSON array of pairs [{"q": "...", "a": "..."}, ...]
"""

# The calibration route INVERTS the standard contract: its answers are supposed NOT to
# answer (they say "it is not known"). The standard judge would score every one 0 and
# drop the whole route, so calibration is graded on whether it is a VALID calibrated
# non-answer instead.
CALIB_SYSTEM = """\
You are grading CALIBRATED-UNCERTAINTY pairs from pre-1930s books. Unlike ordinary Q/A,
the ANSWER here is SUPPOSED to decline — to state that the fact is not known, cannot be
determined, is disputed, or is unsettled. Judge whether each pair is a VALID calibrated
non-answer.

Mark 1 (keep) only if ALL hold:
  - The QUESTION asks a definite factual point (a cause, date, name, number, identity, or
    outcome), as though a definite answer were expected.
  - The ANSWER genuinely states that THAT point is not known / not determinable /
    disputed / unsettled — an honest non-answer to exactly what was asked.
  - The not-knowing is the SUBSTANCE of the answer and concerns the question's point.

Mark 0 (drop) if ANY hold:
  - The answer actually SUPPLIES the fact (it answers) — it is not a non-answer.
  - The uncertainty is incidental or off-topic: the answer really states other facts and
    only mentions "not known" in passing (e.g. lists works with a parenthetical "a work
    not known"), or expresses mere personal opinion/hesitation.
  - The answer's not-knowing is about a DIFFERENT point than the question asks (mismatch:
    e.g. question asks a person's name, answer says his private history is little known).
  - The question is not a definite factual question, or does not stand alone (bare "this
    relief", "the factor in question").

Do NOT fact-check against modern knowledge. Do NOT judge style or era.

Study these examples:
  Q: "What is the cause of the disease?"  A: "The cause of the disease is not known." -> 1
  Q: "What were the most valuable works?"  A: "The libraries could supply nothing more
     valuable than an Antiphonale, a Responsale, and a Grammatica (a work not known)." -> 0
     (it lists the works; the hedge is incidental)
  Q: "What was Partridge's real name?"  A: "Little is known of Partridge's private
     history." -> 0 (non-answer about a different point than the name asked)
  Q: "How many men fell?"  A: "Five hundred men fell in the assault." -> 0 (it answers)

Output ONLY a bare JSON array of 0/1 — one per pair, in order, nothing else. e.g. [1,0,1]
Input: JSON array of pairs [{"q": "...", "a": "..."}, ...]
"""


def _system_for(route):
    """The judge prompt for a route: the calibrated-non-answer grader for calibration,
    the ordinary 'does the answer answer it' grader for everything else."""
    return CALIB_SYSTEM if route == "calibration_qa" else SYSTEM


# Judge should be deterministic: temperature 0. (extra_body merges into the request; for
# DeepSeek DISABLE_THINKING is empty, for opencode it carries the thinking-off flag.)
JUDGE_EXTRA = {**engine.DISABLE_THINKING, "temperature": 0}


def _judge_route(model, batch, system=SYSTEM):
    """Judge Route with output capped tight: a bare 0/1 array is ~2 tokens/pair, so cap
    max_tokens near that (with margin). Keeps a runaway/ramble on a hard batch from
    generating a full 8k-token response before it's discarded and the batch is split."""
    return engine.Route(name="verify", system=system, source=None, answer_fn=None,
                        model=model, extra_body=JUDGE_EXTRA, max_tokens=max(64, batch * 8 + 64))


def _pairs(row):
    """(question, answer) pairs for a row. For multiturn, each assistant turn paired with
    a question that carries the conversation so far as context."""
    convs = row.get("conversations")
    if not convs:
        return [(row.get("question", ""), row.get("answer", ""))]
    out, history = [], []
    for k in range(0, len(convs) - 1, 2):
        u, a = convs[k].get("content", ""), convs[k + 1].get("content", "")
        q = ("\n".join(history) + f"\nUser: {u}").strip() if history else u
        out.append((q, a))
        history += [f"User: {u}", f"Assistant: {a}"]
    return out


def _collect(routes, sample, seed):
    items = []                          # (route, doc_index, question, answer)
    for r in routes:
        rows = _load_route(r)
        if sample and len(rows) > sample:
            rows = random.Random(seed).sample(rows, sample)
        for row in rows:
            di = row.get("doc_index")
            for q, a in _pairs(row):
                if q and a:
                    items.append((r, di, q, a))
    return items


async def _judge(client, sem, route, batch):
    """Judge one batch -> [(route, di, q, a, 0|1), ...]. The model returns a bare 0/1
    array positionally aligned to the input; if the count doesn't match (positional
    drift), split the batch and re-judge so large batches stay safe. Singletons that
    still fail are dropped (left unjudged)."""
    payload = json.dumps([{"q": p[2], "a": p[3]} for p in batch])
    parsed = await engine._call(client, sem, route, route.system, payload, len(batch))
    if (isinstance(parsed, list) and len(parsed) == len(batch)
            and all(v in (0, 1, True, False) for v in parsed)):
        return [(*p, int(v)) for p, v in zip(batch, parsed)]
    if len(batch) <= 1:
        return []                       # unjudged (kept, not dropped, downstream)
    mid = len(batch) // 2
    left = await _judge(client, sem, route, batch[:mid])
    right = await _judge(client, sem, route, batch[mid:])
    return left + right


async def _judge_all(client, sem, route_cfg, items, char_budget, max_pairs, label):
    """Judge all items in content-sized batches, printing live progress on one line."""
    total = len(items)
    bs = _pack(items, char_budget, max_pairs)
    out, done, last = [], [0], [-1]
    step = 400                          # update the line ~every 400 pairs, not every 1%

    async def one(b):
        out.extend(await _judge(client, sem, route_cfg, b))
        done[0] += len(b)
        mark = done[0] // step
        if mark != last[0] or done[0] == total:
            last[0] = mark
            print(f"\r  {label}: {done[0]:,}/{total:,} ({done[0] * 100 // max(1, total)}%)   ",
                  end="", flush=True)

    await asyncio.gather(*[asyncio.create_task(one(b)) for b in bs])
    print()
    return out


async def _run(items, model, batch_size, char_budget, show, system=SYSTEM):
    route = _judge_route(model, batch_size, system)
    async with engine.open_client() as client:
        sem = asyncio.Semaphore(engine.CONCURRENCY)
        scored = await _judge_all(client, sem, route, items, char_budget, batch_size, "judging")
    _report(scored, len(items), show)
    return scored


def _report(scored, n_total, show):
    print(f"\njudged {len(scored):,}/{n_total:,} pairs "
          f"({n_total - len(scored):,} unjudged/kept)\n")
    by_route = defaultdict(list)
    for route, di, q, a, s in scored:
        by_route[route].append(s)
    print(f"{'route':22}{'n':>8}{'%misaligned':>14}")
    print("-" * 44)
    for route in sorted(by_route):
        s = by_route[route]
        bad = 100 * sum(1 for x in s if x == 0) / len(s)
        print(f"{route:22}{len(s):>8}{bad:>13.1f}%")
    alls = [s for v in by_route.values() for s in v]
    if alls:
        print("-" * 44)
        print(f"{'ALL':22}{len(alls):>8}"
              f"{100*sum(1 for x in alls if x==0)/len(alls):>13.1f}%")

    lows = [(route, di, q, a) for route, di, q, a, s in scored if s == 0]
    print(f"\n=== {len(lows):,} misaligned (0), showing {min(show, len(lows))} ===")
    for route, di, q, a in lows[:show]:
        print(f"  [{route} {di}]")
        print(f"      Q: {q[:150]}")
        print(f"      A: {a[:150]}")


async def _run_sanity(routes, model, batch, char_budget, n, seed):
    """Discrimination test: judge N real pairs AND N deliberately-WRONG pairs (each
    question given a different question's answer). A discriminating judge scores real
    pairs ~1 and mismatched pairs ~0. If it passes the mismatches as 1, it is rubber-
    stamping and the pass is worthless."""
    reals = _collect(routes, 0, seed)
    reals = random.Random(seed).sample(reals, min(n, len(reals)))
    perm = [p[3] for p in reals]
    random.Random(seed + 1).shuffle(perm)
    mism = [(r, di, q, a2) for (r, di, q, a), a2 in zip(reals, perm) if a2 != a]
    route = _judge_route(model, batch, _system_for(routes[0]) if len(routes) == 1 else SYSTEM)
    async with engine.open_client() as client:
        sem = asyncio.Semaphore(engine.CONCURRENCY)
        real_scored = await _judge_all(client, sem, route, reals, char_budget, batch, "sanity real")
        mism_scored = await _judge_all(client, sem, route, mism, char_budget, batch, "sanity mismatch")
    keep = 100 * sum(s for *_, s in real_scored) / max(1, len(real_scored))
    caught = 100 * sum(1 for *_, s in mism_scored if s == 0) / max(1, len(mism_scored))
    print(f"\nSANITY (judge={model}, batch={batch}):")
    print(f"  real pairs      : {len(real_scored):>5} judged, {keep:5.1f}% scored 1  "
          f"(want high — real pairs are mostly correct)")
    print(f"  mismatched pairs: {len(mism_scored):>5} judged, {caught:5.1f}% scored 0  "
          f"(DISCRIMINATION — want high; low = rubber-stamping)")


async def _run_filter(routes, model, batch, char_budget, shard_size, force=False):
    """Full filter pass with INTRA-ROUTE checkpointing: judge every pair, DROP any row
    with a pair scored 0 (a whole multiturn conversation goes if any turn fails), and
    write kept rows to verified/<route>/ as uniform shards. Pair judgments are saved to
    synth/state/verify_<route>.json every ~2000 pairs, so a mid-route stop (out of credit,
    Ctrl-C) resumes where it left off instead of re-judging the whole route. The
    checkpoint is deleted once the route's filtered output is written. Reads fresh."""
    from synth import hf_push
    done = {f.split("/")[1] for f in
            hf_push._api().list_repo_files(hf_push.HF_REPO, repo_type="dataset")
            if f.startswith("verified/") and f.endswith(".jsonl")}
    async with engine.open_client() as client:
        sem = asyncio.Semaphore(engine.CONCURRENCY)
        for ri, route in enumerate(routes, 1):
            route_cfg = _judge_route(model, batch, _system_for(route))   # calib gets its own grader
            tag = f"[{ri}/{len(routes)}] {route}"
            if route in done and not force:
                print(f"{tag}: already verified (verified/{route}/), skipping", flush=True)
                continue
            print(f"{tag}: loading from HF...", flush=True)
            rows = _load_route(route, fresh=True)
            items = []                      # (pair_key, doc_index, q, a)
            for row in rows:
                di = row.get("doc_index")
                for k, (q, a) in enumerate(_pairs(row)):
                    if q and a:
                        items.append((f"{di}#{k}", di, q, a))

            ckpt = engine.STATE_DIR / f"verify_{route}.json"
            scores = json.loads(ckpt.read_text()) if ckpt.exists() else {}
            pending = [it for it in items if it[0] not in scores]
            print(f"  {tag}: {len(items) - len(pending):,} judged already, "
                  f"{len(pending):,} pending", flush=True)
            bs = _pack(pending, char_budget, batch)
            n, saved, printed, total = [0], [0], [-1], len(pending)

            async def one(b):
                for pk, di, q, a, s in await _judge(client, sem, route_cfg, b):
                    scores[pk] = s
                n[0] += len(b)
                if n[0] // 400 != printed[0]:
                    printed[0] = n[0] // 400
                    print(f"\r  {tag}: {n[0]:,}/{total:,} judged   ", end="", flush=True)
                if n[0] - saved[0] >= 2000:      # checkpoint
                    saved[0] = n[0]
                    engine.STATE_DIR.mkdir(parents=True, exist_ok=True)
                    ckpt.write_text(json.dumps(scores))

            if bs:
                await asyncio.gather(*[asyncio.create_task(one(b)) for b in bs])
                ckpt.write_text(json.dumps(scores))   # final checkpoint
                print()

            drop = {di for pk, di, q, a in items if scores.get(pk) == 0}
            kept = [row for row in rows if row.get("doc_index") not in drop]
            print(f"  {tag}: writing verified/{route}/ ...", flush=True)
            _, nshards, _ = hf_push.write_sharded(f"verified/{route}", kept, shard_size)
            ckpt.unlink(missing_ok=True)     # route complete -> drop the checkpoint
            pct = 100 * len(drop) / max(1, len(rows))
            print(f"  {tag}: {len(rows):,} rows -> kept {len(kept):,} "
                  f"(dropped {len(drop):,}, {pct:.1f}%)  -> verified/{route}/ ({nshards} shards)\n",
                  flush=True)


def main():
    ap = argparse.ArgumentParser(description="LLM-judge Q/A alignment verification")
    ap.add_argument("--route", choices=ROUTES, help="single route (default: all)")
    ap.add_argument("--sample", type=int, default=1000, help="rows/route to judge (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=engine.MODEL, help="judge model (default: engine MODEL)")
    ap.add_argument("--batch", type=int, default=BATCH, help="MAX pairs per judge call")
    ap.add_argument("--char-budget", type=int, default=CHAR_BUDGET, help="max q+a chars per batch (long-content routes get fewer pairs/call)")
    ap.add_argument("--show", type=int, default=20, help="misaligned examples to print")
    ap.add_argument("--sanity", type=int, default=0, metavar="N",
                    help="discrimination test on N real + N shuffled-answer pairs")
    ap.add_argument("--filter", action="store_true",
                    help="FULL pass: judge everything, drop rows scored 0, write verified/<route>/ to HF")
    ap.add_argument("--shard-size", type=int, default=2000, help="rows per filtered shard")
    ap.add_argument("--force", action="store_true", help="re-verify routes that already have verified/ output")
    args = ap.parse_args()
    routes = [args.route] if args.route else list(ROUTES)
    if args.sanity:
        asyncio.run(_run_sanity(routes, args.model, args.batch, args.char_budget, args.sanity, args.seed))
        return
    if args.filter:
        asyncio.run(_run_filter(routes, args.model, args.batch, args.char_budget, args.shard_size, args.force))
        return
    items = _collect(routes, args.sample, args.seed)
    print(f"collected {len(items):,} pairs from {len(routes)} route(s); judging (batch={args.batch})...")
    sysprompt = _system_for(args.route) if args.route else SYSTEM
    asyncio.run(_run(items, args.model, args.batch, args.char_budget, args.show, sysprompt))


if __name__ == "__main__":
    main()
