"""LLM-judge for synthetic-question quality.

Scores each generated question on the two axes the hand-review passes established as
the ones that matter, judged independently:

  answered (1-5): does the PROVIDED answer actually answer this question? 5 = the
                  answer fully and directly answers it; 1 = the answer is about
                  something else (hallucinated / wrong-referent). This is the
                  correctness axis — the model-free echo metric is blind to it.
  period   (1-5): does the question read like a genuine pre-1930 catechism/schoolbook
                  question — plain, direct, archaic-ok — rather than modern,
                  encyclopedic, or retrospective phrasing? 5 = indistinguishable from
                  period; 1 = clearly modern.

The judge sees the answer and the synthetic question, but NOT the authentic question,
so it cannot simply score similarity — it judges the question on its own merits.

Use a judge model that is NOT one of the candidates being scored (avoid self-
preference). Reuses the opencode chat/completions plumbing from synth.gen_models.

    python -m synth.judge --judge grok-4.5 output/synth/questions_working__kimi-k2-7-code.json
"""
import re
import sys
import json
import asyncio
import argparse
from pathlib import Path

import httpx

from synth.gen_models import _api_key, BASE_URL, _parse_json_array

ROOT = Path(__file__).parent.parent
CONCURRENCY = 12
MAX_TOKENS = 8192
BATCH = 8  # items per judge call

SYSTEM = """\
You evaluate questions that were generated from answers taken from pre-1930s
educational catechisms and schoolbooks. Each item gives you the source ANSWER and a
QUESTION written to elicit it. Score the QUESTION on two independent axes, 1-5.

answered — does the provided ANSWER actually and fully answer this QUESTION?
  5: the answer directly and completely answers exactly what is asked
  3: the answer partially answers, or answers a broader/narrower question
  1: the answer is about a different subject; it does not answer this question
  (Judge only against the given answer. A question the answer cannot support scores low
   even if it is a fine question in the abstract.)

period — does the QUESTION read like a genuine pre-1930 catechism/schoolbook question?
  5: indistinguishable from a period question — plain, direct, archaic phrasing fine
  3: mostly period but with a faint modern or encyclopedic flavour
  1: clearly modern — retrospective labels ("World War I"), analytic framing
     ("the military value of", "how does X relate to Y"), meta or conversational phrasing

Be strict and consistent. Do not reward verbosity.

Input: JSON array [{"i": 0, "answer": "...", "question": "..."}, ...]
Output JSON only: [{"i": 0, "answered": 4, "period": 5}, ...]  (no prose)
"""


def _batches(records, size=BATCH):
    return [records[i:i + size] for i in range(0, len(records), size)]


def _safe(model):
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


async def _judge_batch(client, sem, batch, judge, key, out):
    payload = json.dumps([{"i": n, "answer": r["answer"][:600], "question": r["synthetic_q"]}
                          for n, r in enumerate(batch)])
    body = {"model": judge, "max_tokens": MAX_TOKENS,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": payload}]}
    headers = {"Authorization": f"Bearer {key}"}
    async with sem:
        for attempt in range(5):
            try:
                resp = await client.post(f"{BASE_URL}/v1/chat/completions", json=body, headers=headers)
                if resp.status_code == 429:
                    await asyncio.sleep(2 ** attempt); continue
                resp.raise_for_status()
                text = (resp.json()["choices"][0]["message"]["content"] or "").strip()
                for r in _parse_json_array(text):
                    idx = r.get("i")
                    if isinstance(idx, int) and 0 <= idx < len(batch):
                        a, p = r.get("answered"), r.get("period")
                        if isinstance(a, int) and isinstance(p, int):
                            out[id(batch[idx])] = {"answered": a, "period": p}
                return
            except Exception as e:
                if attempt == 4:
                    print(f"  batch failed: {type(e).__name__}: {str(e)[:120]}")
                    return
                await asyncio.sleep(2 ** attempt)


async def judge_file(path, judge, key):
    records = json.loads(Path(path).read_text())
    out = {}
    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient(timeout=180) as client:
        await asyncio.gather(*[_judge_batch(client, sem, b, judge, key, out)
                               for b in _batches(records)])
    scored = []
    for r in records:
        s = out.get(id(r))
        if s:
            rr = dict(r); rr.update(s); scored.append(rr)
    outp = Path(path).with_name(Path(path).stem + f"__judged-{_safe(judge)}.json")
    outp.write_text(json.dumps(scored, indent=2, ensure_ascii=False))
    return scored, outp


def summarize(name, scored):
    n = len(scored)
    if not n:
        print(f"{name}: no scores"); return
    ans = [r["answered"] for r in scored]
    per = [r["period"] for r in scored]
    good = sum(1 for r in scored if r["answered"] >= 4 and r["period"] >= 4)
    bad_ans = sum(1 for r in scored if r["answered"] <= 2)
    bad_per = sum(1 for r in scored if r["period"] <= 2)
    print(f"{name:<24} n={n:>3}  answered {sum(ans)/n:.2f}  period {sum(per)/n:.2f}  "
          f"both>=4 {good/n:5.1%}  answered<=2 {bad_ans/n:4.1%}  period<=2 {bad_per/n:4.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--judge", required=True, help="opencode model id to judge with (not a candidate)")
    args = ap.parse_args()
    key = _api_key()
    print(f"judge: {args.judge}\n")
    print(f"{'file':<24} {'n':>5}  {'answered':>8}  {'period':>6}  both>=4  ans<=2  per<=2")
    for path in args.inputs:
        scored, outp = asyncio.run(judge_file(path, args.judge, key))
        summarize(Path(path).stem.replace("questions_working__", ""), scored)
        print(f"  -> {outp.name}")


if __name__ == "__main__":
    main()
