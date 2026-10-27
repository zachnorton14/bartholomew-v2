"""Pass 2: Group dependent Q/A chains into multi-turn conversations.

For each pair, the model outputs a boolean: does this question depend on the
immediately preceding one? Chains are reconstructed algorithmically from those booleans.
Datasets are processed in parallel; windows within a dataset are sequential.

Set environment variables before running:
    export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
    export ANTHROPIC_API_KEY=<your deepseek key>
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import anthropic

ROOT = Path(__file__).parent.parent
(ROOT / "state").mkdir(parents=True, exist_ok=True)
INPUT_DIR = ROOT / "output" / "ocr"
OUTPUT_DIR = ROOT / "output" / "paired"
STATE_FILE = ROOT / "state" / "pair.json"
MODEL = "claude-haiku-4-5"
MAX_TOKENS = 8192
CONCURRENCY = 10
WINDOW_SIZE = 50
CONTEXT_SIZE = 1  # one prior pair shown as context for deps[0] judgment

SYSTEM_PROMPT = """\
You receive Q&A pairs from a pre-1930s text, optionally preceded by one CONTEXT pair.
For each pair [0]..[N], output whether its question depends on the immediately preceding pair.

"Depends" means the question uses a pronoun (it, he, she, they), demonstrative (this, that, so called),
or a specific term quoted from the prior answer — and cannot be understood without it.

deps[0] = does [0] depend on the CONTEXT pair? (false if no context)
deps[i] = does [i] depend on [i-1]?

Example:
CONTEXT: Q: What did Henry IV do? A: He invaded France and won at Agincourt.
[0] Q: What was the result?   → true  ("the result" references the invasion)
[1] Q: What is a monarchy?    → false (new topic)
[2] Q: Why so called?         → true  ("so called" references "monarchy" from [1])

Output JSON only: {"deps": [true, false, true]}
"""


def load_all_pairs():
    datasets = {}
    for json_file in sorted(INPUT_DIR.glob("*.json")):
        dataset = json_file.stem
        items = json.loads(json_file.read_text())
        pairs = []
        for item in items:
            convs = item["conversations"]
            q = next(c["content"] for c in convs if c["role"] == "user")
            a = next(c["content"] for c in convs if c["role"] == "assistant")
            pairs.append((q, a))
        datasets[dataset] = pairs
    return datasets


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state))


def make_windows(pairs):
    return [pairs[i:i + WINDOW_SIZE] for i in range(0, len(pairs), WINDOW_SIZE)]


def _build_user_message(context_pair, window_pairs):
    lines = []
    if context_pair:
        q, a = context_pair
        lines.append(f"CONTEXT: Q: {q}")
        lines.append(f"         A: {a}")
        lines.append("")
    for i, (q, a) in enumerate(window_pairs):
        lines.append(f"[{i}] Q: {q}")
        lines.append(f"    A: {a}")
    return "\n".join(lines)


def _parse_response(text, window_size):
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rstrip("`").strip()
    try:
        parsed = json.loads(text)
        deps = parsed.get("deps", [])
        if isinstance(deps, list) and len(deps) == window_size:
            return [bool(d) for d in deps]
    except (json.JSONDecodeError, TypeError):
        pass
    return [False] * window_size


async def _process_window(client, semaphore, dataset, w_idx, context_pair, window_pairs, state):
    key = f"{dataset}--w{w_idx}"
    async with semaphore:
        for attempt in range(5):
            try:
                user_msg = _build_user_message(context_pair, window_pairs)
                msg = await client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                )
                text = next((b.text for b in msg.content if b.type == "text"), "").strip()
                state[key] = {"deps": _parse_response(text, len(window_pairs))}
                return
            except anthropic.RateLimitError:
                await asyncio.sleep(2 ** attempt)
            except Exception:
                if attempt == 4:
                    state[key] = {"deps": [False] * len(window_pairs)}
                else:
                    await asyncio.sleep(2 ** attempt)


async def _process_dataset(client, semaphore, dataset, pairs, state, progress):
    windows = make_windows(pairs)
    for w_idx, window_pairs in enumerate(windows):
        key = f"{dataset}--w{w_idx}"
        if key not in state:
            context_pair = pairs[w_idx * WINDOW_SIZE - 1] if w_idx > 0 else None
            await _process_window(client, semaphore, dataset, w_idx, context_pair, window_pairs, state)
        progress[0] += 1
        save_state(state)
        print(f"  {progress[0]}/{progress[1]} windows", flush=True)


async def run_async(datasets, state):
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(CONCURRENCY)
    total_windows = sum(len(make_windows(p)) for p in datasets.values())
    progress = [0, total_windows]
    tasks = [
        asyncio.create_task(_process_dataset(client, semaphore, d, p, state, progress))
        for d, p in datasets.items()
    ]
    await asyncio.gather(*tasks)


def _reconstruct_chains(pairs, dataset, state):
    windows = make_windows(pairs)
    chains = []
    carry = []

    for w_idx, window_pairs in enumerate(windows):
        key = f"{dataset}--w{w_idx}"
        deps = state.get(key, {}).get("deps", [False] * len(window_pairs))

        offset = w_idx * WINDOW_SIZE
        for i, dep in enumerate(deps):
            global_idx = offset + i
            if dep and carry:
                carry.append(global_idx)
            else:
                if carry:
                    chains.append(carry)
                carry = [global_idx]

    if carry:
        chains.append(carry)

    return chains


def write_output(datasets, state):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_written = total_multi = 0

    for dataset, pairs in sorted(datasets.items()):
        chains = _reconstruct_chains(pairs, dataset, state)
        items = []
        multi = 0

        for chain in chains:
            if len(chain) == 1:
                q, a = pairs[chain[0]]
                items.append({"conversations": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ]})
            else:
                convs = []
                for idx in chain:
                    q, a = pairs[idx]
                    convs.append({"role": "user", "content": q})
                    convs.append({"role": "assistant", "content": a})
                items.append({"conversations": convs, "chained": True})
                multi += 1

        out = OUTPUT_DIR / f"{dataset}.json"
        out.write_text(json.dumps(items, indent=2, ensure_ascii=False))
        total_written += len(items)
        total_multi += multi
        print(f"  {dataset}: {len(items)} conversations ({multi} multi-turn)")

    print(f"\nTotal: {total_written} conversations, {total_multi} multi-turn")


async def test_run(datasets, seed, size):
    import random
    random.seed(seed)

    all_windows = []
    for dataset, pairs in datasets.items():
        windows = make_windows(pairs)
        for w_idx, window_pairs in enumerate(windows):
            context_pair = pairs[w_idx * WINDOW_SIZE - 1] if w_idx > 0 else None
            all_windows.append((dataset, w_idx, context_pair, window_pairs))

    sample = random.sample(all_windows, min(size, len(all_windows)))
    client = anthropic.AsyncAnthropic()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    samples_dir = ROOT / "samples" / "pair"
    samples_dir.mkdir(parents=True, exist_ok=True)
    out_path = samples_dir / f"{ts}_seed{seed}_n{size}.txt"

    header = [f"model: {MODEL}", f"seed:  {seed}", f"n:     {size}", "",
              "--- SYSTEM PROMPT ---", SYSTEM_PROMPT, "--- END SYSTEM PROMPT ---", ""]
    out_path.write_text("\n".join(header))

    async def process(dataset, w_idx, context_pair, window_pairs):
        user_msg = _build_user_message(context_pair, window_pairs)
        msg = await client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = next((b.text for b in msg.content if b.type == "text"), "").strip()
        deps = _parse_response(text, len(window_pairs))

        lines = [f"[{dataset} window {w_idx}]"]
        if context_pair:
            q, a = context_pair
            lines.append(f"  CONTEXT: Q: {q}")
            lines.append(f"           A: {a}")
        for i, (q, a) in enumerate(window_pairs):
            marker = "T" if deps[i] else "F"
            lines.append(f"  [{i}]({marker}) Q: {q}")
            lines.append(f"         A: {a}")
        lines.append("")

        with out_path.open("a") as f:
            f.write("\n".join(lines) + "\n")

    await asyncio.gather(*[process(d, w, c, wp) for d, w, c, wp in sample])
    print(f"Wrote {out_path}")
