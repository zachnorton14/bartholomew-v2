"""Build the register-diagnostic test set: for a pool of answers spanning every
question type, generate a period (vintage) and a modern question from the SAME
answer, and tag the authentic question onto knowledge. Embedding these on Talkie
shows where each type/register lands and whether a register axis exists that is
blind to type.

Answer sources:
  knowledge  -> authentic/output/filtered (carries the real q_authentic)
  all others -> the HF dataset routes (answer field)

Per answer we manipulate ONLY register: q_vintage and q_modern come from the same
answer via the same mechanism, differing only in the system prompt. Output rows:
  {"id","type","answer","q_vintage","q_modern","q_authentic"|null}

    python -m synth.build_register_test --per-type 1000 --out pairs.jsonl
    python -m synth.build_register_test --smoke --out pairs.smoke.jsonl   # ~8/type

Generation uses DeepSeek (DS_API_KEY in .env). Embedding is a separate A100 step.
"""
import os
import sys
import json
import random
import asyncio
import argparse
from pathlib import Path

import httpx

from synth.gen_models import _from_dotenv, _parse_json_array

ROOT = Path(__file__).parent.parent
HF_REPO = "zachnorton03/synthetic-pre1930-sft"
BASE_URL = "https://api.deepseek.com"      # DeepSeek's OpenAI-compatible API
MODEL = "deepseek-chat"                     # non-reasoning V3 (thinking disabled)
CONCURRENCY = 16
MAX_TOKENS = 8192
BATCH = 12

# Dataset route folder -> type label. knowledge is sourced from the authentic corpus.
ROUTE_TYPES = {
    "stem_reasoning": "stem_reasoning", "reasoning_qa": "reasoning",
    "narrative_grounded": "narrative_grounded", "narrative_fiction": "narrative_fiction",
    "composition_qa": "composition", "how_to_qa": "how_to", "opinion_qa": "opinion",
    "verse_qa": "verse",
}
MT_ROUTE = "multiturn_qa"   # different schema (conversations array), handled separately

VINTAGE_PROMPT = """\
You receive answers from pre-1930s texts. For each, write the single question it answers.
Write it in period English — plain, direct, the register of a pre-1930s book or examiner.
Use no word, idiom, or framing that came into use after 1930; no modern or conversational
phrasing. Express any quantity, relation, or operation IN WORDS — never use algebraic
symbols, formulae, or non-letter notation. The question must stand alone and never refer
to a passage, text, or author.

Input: JSON array [{"i": 0, "a": "..."}, ...]
Output JSON only: [{"i": 0, "q": "the question"}, ...]
"""

MODERN_PROMPT = """\
You receive answers. For each, write the single question it answers, phrased the way a
present-day speaker naturally would — contemporary vocabulary and framing, an ordinary
modern register ("What's the significance of...", "How does ... work", "Can you explain
why..."). Do not imitate archaic or period style. The question must stand alone.

Input: JSON array [{"i": 0, "a": "..."}, ...]
Output JSON only: [{"i": 0, "q": "the question"}, ...]
"""

# multiturn: generate a FOLLOW-UP question given the conversation so far. This
# preserves multiturn's distinctive register (anaphora, chaining) — a standalone
# question from the answer alone would collapse to knowledge-style.
MT_VINTAGE_PROMPT = """\
You receive a short conversation from a pre-1930s text — the questions and answers so
far — and a TARGET ANSWER that comes next in it. Write the next question: the one the
TARGET ANSWER answers, as a natural follow-up in the conversation. It may use pronouns
or refer to what was already discussed (it need not stand alone). Write it in period
English — the register of a pre-1930s book or examiner; no post-1930 word, idiom, or
modern phrasing. Express any quantity or relation in words.

Input: JSON array [{"i": 0, "history": "...", "a": "target answer"}, ...]
Output JSON only: [{"i": 0, "q": "the follow-up question"}, ...]
"""

MT_MODERN_PROMPT = """\
You receive a short conversation — the questions and answers so far — and a TARGET
ANSWER that comes next. Write the next question the TARGET ANSWER answers, as a natural
follow-up, phrased the way a present-day speaker would (contemporary vocabulary and
framing). It may use pronouns or refer to what was discussed. Do not imitate archaic or
period style.

Input: JSON array [{"i": 0, "history": "...", "a": "target answer"}, ...]
Output JSON only: [{"i": 0, "q": "the follow-up question"}, ...]
"""


def _ds_key():
    key = os.environ.get("DS_API_KEY") or _from_dotenv("DS_API_KEY")
    if not key:
        sys.exit("Set DS_API_KEY (env or .env) to the DeepSeek key.")
    return key


def _download(cache):
    from huggingface_hub import snapshot_download
    return snapshot_download(HF_REPO, repo_type="dataset", token=False,
                             allow_patterns=[f"{r}/*" for r in list(ROUTE_TYPES) + [MT_ROUTE]],
                             local_dir=str(cache))


def _load_json_any(path):
    txt = Path(path).read_text().strip()
    try:
        d = json.loads(txt)
        return d if isinstance(d, list) else [d]
    except json.JSONDecodeError:
        return [json.loads(l) for l in txt.splitlines() if l.strip()]


def route_answers(hfds, route, n, seed):
    rows = []
    for f in sorted(Path(hfds, route).glob("*")):
        if f.suffix in (".json", ".jsonl"):
            rows += _load_json_any(f)
    answers = [r["answer"] for r in rows if r.get("answer")]
    random.Random(seed).shuffle(answers)
    return answers[:n]


def authentic_pairs(n, seed):
    pairs = []
    for jf in sorted((ROOT / "authentic" / "output" / "filtered").glob("*.json")):
        for item in json.loads(jf.read_text()):
            convs = item["conversations"]
            q = next((c["content"] for c in convs if c["role"] == "user"), None)
            a = next((c["content"] for c in convs if c["role"] == "assistant"), None)
            if q and a:
                pairs.append((a, q))
    random.Random(seed).shuffle(pairs)
    return pairs[:n]


def load_multiturn(hfds, n, seed):
    """Return [(history, target_answer)] — a follow-up turn per conversation, with the
    prior exchange as context, so the generated question is a genuine follow-up."""
    rows = []
    for f in sorted(Path(hfds, MT_ROUTE).glob("*")):
        if f.suffix in (".json", ".jsonl"):
            rows += _load_json_any(f)
    items = []
    for r in rows:
        c = r.get("conversations") or []
        # need at least opening (u,a) + a follow-up answer (index 3)
        if len(c) >= 4 and c[3].get("role") == "assistant" and c[3].get("content"):
            history = f"Q: {c[0]['content']}\nA: {c[1]['content']}"
            items.append((history, c[3]["content"]))
    random.Random(seed).shuffle(items)
    return items[:n]


async def _gen_batch(client, sem, batch, system, key, out):
    payload = json.dumps([{"i": i, **obj} for i, obj in batch])
    body = {"model": MODEL, "max_tokens": MAX_TOKENS,
            "messages": [{"role": "system", "content": system},
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
                local = {b[0]: b for b in batch}
                for r in _parse_json_array(text):
                    idx, q = r.get("i"), (r.get("q") or "").strip()
                    if idx in local and q:
                        out[idx] = q
                return
            except Exception as e:
                if attempt == 4:
                    print(f"  batch failed: {type(e).__name__}: {str(e)[:120]}")
                    return
                await asyncio.sleep(2 ** attempt)


async def generate(items, system, key):
    """items: list of (global_index, payload_obj). Returns {global_index: question}."""
    out = {}
    batches = [items[i:i + BATCH] for i in range(0, len(items), BATCH)]
    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient(timeout=180) as client:
        await asyncio.gather(*[_gen_batch(client, sem, b, system, key, out) for b in batches])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true", help="~8 answers per type")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hfds", default=str(ROOT / "synth" / "output" / "hfds"),
                    help="local HF dataset snapshot (downloaded if missing)")
    ap.add_argument("--only-multiturn", action="store_true",
                    help="generate only the multiturn type (follow-up questions with context)")
    ap.add_argument("--append", action="store_true", help="append to --out instead of overwriting")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n = 8 if args.smoke else args.per_type
    key = _ds_key()
    if not Path(args.hfds, "stem_reasoning").exists():
        print("downloading dataset routes...")
        _download(args.hfds)

    # pool rows: [id, type, answer, authentic_q_or_None, generation_payload]
    pool = []
    if args.only_multiturn:
        for hist, tgt in load_multiturn(args.hfds, n, args.seed):
            pool.append([f"multiturn-{len(pool):06d}", "multiturn", tgt, None,
                         {"history": hist[:1200], "a": tgt[:600]}])
        vintage_prompt, modern_prompt = MT_VINTAGE_PROMPT, MT_MODERN_PROMPT
    else:
        for a, q in authentic_pairs(n, args.seed):
            pool.append([f"knowledge-{len(pool):06d}", "knowledge", a, q, {"a": a[:600]}])
        for route, typ in ROUTE_TYPES.items():
            for a in route_answers(args.hfds, route, n, args.seed):
                pool.append([f"{typ}-{len(pool):06d}", typ, a, None, {"a": a[:600]}])
        vintage_prompt, modern_prompt = VINTAGE_PROMPT, MODERN_PROMPT
    print(f"pool: {len(pool)} answers")

    items = [(i, row[4]) for i, row in enumerate(pool)]
    print("generating vintage questions...")
    vint = asyncio.run(generate(items, vintage_prompt, key))
    print("generating modern questions...")
    modn = asyncio.run(generate(items, modern_prompt, key))

    out = Path(args.out).open("a" if args.append else "w")
    written = 0
    for i, (rid, typ, ans, auth, _) in enumerate(pool):
        if i not in vint or i not in modn:
            continue
        out.write(json.dumps({"id": rid, "type": typ, "answer": ans,
                              "q_vintage": vint[i], "q_modern": modn[i],
                              "q_authentic": auth}, ensure_ascii=False) + "\n")
        written += 1
    out.close()
    print(f"wrote {written} rows -> {args.out} ({'append' if args.append else 'overwrite'})")


if __name__ == "__main__":
    main()
