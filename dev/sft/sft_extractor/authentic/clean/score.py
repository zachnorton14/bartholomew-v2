"""Pass 4: Score conversations on 5 structural dimensions and filter bottom N%.

Dimensions (each 1-10):
  ocr          — text free of garbled characters and corrupted numbers
  intelligible — question has a clear, understandable subject
  relevant     — answer actually addresses the question asked
  complete     — neither Q nor A is a fragment or cut off mid-sentence
  clean        — no bleed artifacts (multiple pairs merged together)

Input: output/enriched/   Output: output/scored/

Set environment variables before running:
    export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
    export ANTHROPIC_API_KEY=<your deepseek key>
"""

import asyncio
import json
from pathlib import Path

import anthropic

ROOT = Path(__file__).parent.parent
(ROOT / "state").mkdir(parents=True, exist_ok=True)
INPUT_DIR = ROOT / "output" / "enriched"
OUTPUT_DIR = ROOT / "output" / "scored"
STATE_FILE = ROOT / "state" / "score.json"
MODEL = "claude-haiku-4-5"
MAX_TOKENS = 8192
CONCURRENCY = 20
TOKEN_BUDGET = 2000

DEFAULT_WEIGHTS = {
    "ocr": 1.0,
    "intelligible": 1.0,
    "relevant": 1.0,
    "complete": 1.0,
    "clean": 1.0,
}

SYSTEM_PROMPT = """\
You receive Q&A conversations extracted from pre-1930s educational texts.
Score each on 5 structural dimensions, 1-10 (10 = no defects, 1 = severe defect).
Be strict. Most pairs have at least minor issues.

ocr:          Text is free of garbled characters, corrupted numbers, broken words.
intelligible: The opening question has a clear, understandable subject.
relevant:     The answer actually addresses the question asked.
complete:     Neither question nor answer is a fragment or cut off mid-sentence.
clean:        No bleed artifacts — not multiple Q&A pairs merged into one.

Examples of LOW scores:
- ocr:2   Q: "How is the mordent performed?" A: "Written. Played. -rr-rr-w=i tr tr" (garbled notation placeholders)
- ocr:3   A: "i i r i i The thumb must be placed on 1st and 5th" (OCR'd fingering chart)
- complete:2  Q: "John Doe engages B as agent and limits him strictly to the price of" (cut off mid-sentence)
- clean:2  Q: "What is heat? A: Heat expands metal. Q. 68. How is this used? A: In thermometers." (bleed)
- relevant:3  Q: "What is arsenic?" A: "The battle of Crecy was fought in 1346." (non-sequitur)
- intelligible:3  Q: "Why not?" (no subject, no context)

Do NOT penalise for:
- Archaic or formal language
- Short answers ("It is.", "Yes.", single sentence answers are fine)
- Niche or obscure subject matter
- Simplicity of the question

Input: JSON array [{"i": 0, "turns": [["Q text", "A text"], ...]}, ...]
Output JSON only: [{"i": 0, "ocr": 9, "intelligible": 8, "relevant": 9, "complete": 10, "clean": 10}, ...]
"""


def load_all_conversations():
    datasets = {}
    for json_file in sorted(INPUT_DIR.glob("*.json")):
        dataset = json_file.stem
        items = json.loads(json_file.read_text())
        datasets[dataset] = items
    return datasets


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state))


def _estimate_tokens(item):
    return sum(len(c["content"]) for c in item["conversations"]) // 4


def _build_turns(item):
    convs = item["conversations"]
    turns = []
    for j in range(0, len(convs) - 1, 2):
        q = convs[j]["content"] if convs[j]["role"] == "user" else ""
        a = convs[j + 1]["content"] if convs[j + 1]["role"] == "assistant" else ""
        turns.append([q, a])
    return turns


def _pack_batches(items, token_budget):
    batches, current, current_tokens = [], [], 0
    for item in items:
        tokens = _estimate_tokens(item[2])
        if current and current_tokens + tokens > token_budget:
            batches.append(current)
            current, current_tokens = [], 0
        current.append(item)
        current_tokens += tokens
    if current:
        batches.append(current)
    return batches


def weighted_score(scores, weights=None):
    w = weights or DEFAULT_WEIGHTS
    total_weight = sum(w[k] for k in w)
    return sum(scores.get(k, 5) * w[k] for k in w) / total_weight


async def _score_batch(client, semaphore, batch, state):
    # batch: list of (dataset, i, item)
    keys = [f"{d}--{i}" for d, i, _ in batch]
    payload = json.dumps([
        {"i": idx, "turns": _build_turns(item)}
        for idx, (_, _, item) in enumerate(batch)
    ])
    async with semaphore:
        for attempt in range(5):
            try:
                msg = await client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": payload}],
                )
                text = next((b.text for b in msg.content if b.type == "text"), "").strip()
                if text.startswith("```"):
                    text = text.split("```", 2)[1]
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.rstrip("`").strip()
                results = json.loads(text)
                for r in results:
                    idx = r.get("i")
                    if isinstance(idx, int) and 0 <= idx < len(keys):
                        state[keys[idx]] = {
                            k: int(r[k]) for k in ("ocr", "intelligible", "relevant", "complete", "clean")
                            if k in r
                        }
                return
            except anthropic.RateLimitError:
                await asyncio.sleep(2 ** attempt)
            except Exception:
                if attempt == 4:
                    for key in keys:
                        if key not in state:
                            state[key] = {"ocr": 5, "intelligible": 5, "relevant": 5, "complete": 5, "clean": 5}
                else:
                    await asyncio.sleep(2 ** attempt)


async def run_async(datasets, state):
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(CONCURRENCY)

    all_items = [
        (d, i, item)
        for d, items in datasets.items()
        for i, item in enumerate(items)
        if f"{d}--{i}" not in state
    ]

    if not all_items:
        print("  Nothing to score.")
        return

    batches = _pack_batches(all_items, TOKEN_BUDGET)
    total = len(all_items)
    print(f"  {total} conversations, {len(batches)} batches...")

    done = [0]

    async def tracked(batch):
        await _score_batch(client, semaphore, batch, state)
        done[0] += len(batch)
        print(f"  {done[0]}/{total}", flush=True)

    tasks = [asyncio.create_task(tracked(b)) for b in batches]
    await asyncio.gather(*tasks)


def write_output(datasets, state, weights=None, filter_pct=0.05):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Compute weighted scores for all conversations
    all_scores = []
    for d, items in datasets.items():
        for i in range(len(items)):
            key = f"{d}--{i}"
            scores = state.get(key, {})
            if scores:
                all_scores.append((d, i, weighted_score(scores, weights)))

    if not all_scores:
        print("  No scores in state — run scoring pass first.")
        return

    # Determine cutoff
    sorted_scores = sorted(all_scores, key=lambda x: x[2])
    cutoff_idx = int(len(sorted_scores) * filter_pct)
    cutoff_score = sorted_scores[cutoff_idx][2] if cutoff_idx > 0 else 0
    dropped_keys = {f"{d}--{i}" for d, i, s in sorted_scores[:cutoff_idx]}

    print(f"  Cutoff score: {cutoff_score:.2f} (bottom {filter_pct*100:.0f}%, {len(dropped_keys)} dropped)")

    total_written = total_dropped = 0
    for dataset, items in sorted(datasets.items()):
        out_items = []
        dropped = 0
        for i, item in enumerate(items):
            key = f"{dataset}--{i}"
            if key in dropped_keys:
                dropped += 1
                total_dropped += 1
            else:
                score = state.get(key, {})
                new_item = dict(item)
                if score:
                    new_item["score"] = round(weighted_score(score, weights), 2)
                out_items.append(new_item)
        out = OUTPUT_DIR / f"{dataset}.json"
        out.write_text(json.dumps(out_items, indent=2, ensure_ascii=False))
        total_written += len(out_items)
        if dropped:
            print(f"  {dataset}: {len(out_items)} kept, {dropped} dropped")

    print(f"\nTotal: {total_written} kept, {total_dropped} dropped")


async def test_run(datasets, seed, size):
    import random
    from datetime import datetime
    random.seed(seed)

    all_items = [(d, i, item)
                 for d, items in datasets.items()
                 for i, item in enumerate(items)]
    sample = random.sample(all_items, min(size, len(all_items)))

    client = anthropic.AsyncAnthropic()
    state = {}
    batches = _pack_batches(sample, TOKEN_BUDGET)
    tasks = [asyncio.create_task(_score_batch(client, asyncio.Semaphore(CONCURRENCY), b, state))
             for b in batches]
    await asyncio.gather(*tasks)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    samples_dir = ROOT / "samples" / "score"
    samples_dir.mkdir(parents=True, exist_ok=True)
    out_path = samples_dir / f"{ts}_seed{seed}_n{size}.txt"

    lines = [f"model: {MODEL}", f"seed:  {seed}", f"n:     {size}", ""]
    for d, i, item in sorted(sample, key=lambda x: weighted_score(state.get(f"{x[0]}--{x[1]}", {})) if state.get(f"{x[0]}--{x[1]}") else 99):
        key = f"{d}--{i}"
        scores = state.get(key, {})
        avg = weighted_score(scores) if scores else 0
        q = next(c["content"] for c in item["conversations"] if c["role"] == "user")
        a = next(c["content"] for c in item["conversations"] if c["role"] == "assistant")
        score_str = "  ".join(f"{k}:{scores.get(k,'?')}" for k in ("ocr", "intelligible", "relevant", "complete", "clean"))
        lines.append(f"[{key}] avg:{avg:.1f}  {score_str}")
        lines.append(f"  Q: {q[:120]}")
        lines.append(f"  A: {a[:120]}")
        lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")
