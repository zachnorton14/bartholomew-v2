"""Pass 0: Detect and recover Q/A bleed pairs using DeepSeek via Anthropic SDK.

Bleed occurs when the extractor incorrectly merges multiple Q/A pairs into one.
This pass detects bleeds and recovers constituent pairs (1→N), passes clean pairs
through unchanged, and discards unrecoverable pairs.

Set environment variables before running:
    export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
    export ANTHROPIC_API_KEY=<your deepseek key>
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

import anthropic

ROOT = Path(__file__).parent.parent
(ROOT / "state").mkdir(parents=True, exist_ok=True)
INPUT_DIR = ROOT / "output" / "extracted"
OUTPUT_DIR = ROOT / "output" / "bleed"
OCR_DIR = ROOT / "output" / "ocr"
STATE_FILE = ROOT / "state" / "bleed.json"
RETRY_STATE_FILE = ROOT / "state" / "bleed_retry.json"
MODEL = "claude-haiku-4-5"  # maps to deepseek-v4-flash via ANTHROPIC_BASE_URL
MAX_TOKENS = 8192
CONCURRENCY = 20

CANDIDATE_RE = re.compile(r'\b[Qq][.\s]\s*\d+[.\s]|\b\d{1,3}\.\s+[A-Z][a-z]')
SPLIT_RE = re.compile(
    r'\s+(?:go|gt|gi|g0|8o|ioo|100)\.\s+'
    r'|\s+[Qq]\.\s*\d+[.\s]\s+'
    r'|\s+\d{1,3}\.\s+(?=[A-Z][a-z])',
    re.IGNORECASE,
)

RETRY_DETECT_PROMPT = """\
You receive Q&A pairs from a pre-1930s text.
For each pair, detect whether the Q field contains a BLEED — multiple Q/A pairs incorrectly
merged into one. A bleed occurs when Q contains answer prose followed by another question,
often signalled by OCR-corrupted numbers (go., gt., gi., Q.68.) or declarative sentences
mid-Q followed by a new question.

Input: JSON array [{"i": 0, "q": "...", "a": "..."}, ...]
Output JSON only: [{"i": 0, "bleed": false}, ...]
"""

SYSTEM_PROMPT = """\
You are a dataset quality filter for public-domain Q&A pairs extracted from pre-1930s texts.
Your job is to detect structural bleed — where an OCR extractor incorrectly merged multiple
Q&A pairs into one entry.

A bleed occurs when the Q field contains answer text followed by another question. The key
signal is a pattern of: question → answer prose → question → answer prose → question.

Bleed indicators in the Q field:
- Answer text (declarative sentences that respond to the preceding question)
- Numbered sub-questions mid-field, including OCR-corrupted numbers:
  "go." = 90., "gt." = 91., "gi." = 91., "8o." = 80., "ioo." = 100., etc.

The A field from the input always corresponds to the LAST question in the bleed.
Only include recovered pairs where both Q and A are complete and meaningful.

Do not fix OCR artifacts — preserve all text exactly as given, including corrupted numbers.
Respond with valid JSON only.

If the pair is CLEAN, respond with:
  {"clean": true}

If the pair has bleed and you can recover distinct Q&A pairs, respond with:
  {"pairs": [{"q": "...", "a": "..."}, ...]}

If no clean Q&A can be recovered, respond with:
  {"discard": true}

Examples:

Input — Q: What is a measure? That by which extent is ascertained. go. How many dimensions has extension? Extension has three. gt. Explain how distance is measured by time?
         A: Every circle is divided into 360 degrees.
Output: {"pairs": [{"q": "What is a measure?", "a": "That by which extent is ascertained."}, {"q": "How many dimensions has extension?", "a": "Extension has three."}, {"q": "Explain how distance is measured by time?", "a": "Every circle is divided into 360 degrees."}]}

Input — Q: What is History?
         A: A record of past events.
Output: {"clean": true}

Input — Q: §§§ xh§ merged §§§ unreadable §§§
         A: §§§ garbage
Output: {"discard": true}
"""


def load_all_pairs():
    pairs = []
    for json_file in sorted(INPUT_DIR.glob("*.json")):
        dataset = json_file.stem
        items = json.loads(json_file.read_text())
        for i, item in enumerate(items):
            convs = item["conversations"]
            q = next(c["content"] for c in convs if c["role"] == "user")
            a = next(c["content"] for c in convs if c["role"] == "assistant")
            pairs.append((dataset, i, q, a))
    return pairs


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state))


def parse_response(text, q, a):
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rstrip("`").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"clean": True}
    if parsed.get("discard"):
        return {"discard": True}
    if parsed.get("clean"):
        return {"clean": True}
    if "pairs" in parsed and isinstance(parsed["pairs"], list):
        valid = [p for p in parsed["pairs"] if p.get("q") and p.get("a")]
        return {"pairs": valid} if valid else {"discard": True}
    return {"clean": True}


async def process_pair(client, semaphore, dataset, i, q, a, state, progress):
    key = f"{dataset}--{i}"
    async with semaphore:
        for attempt in range(5):
            try:
                msg = await client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": f"Q: {q}\nA: {a}"}],
                )
                text = next((b.text for b in msg.content if b.type == "text"), "").strip()
                state[key] = parse_response(text, q, a)
                break
            except anthropic.RateLimitError:
                await asyncio.sleep(2 ** attempt)
            except Exception:
                if attempt == 4:
                    state[key] = {"clean": True}
                else:
                    await asyncio.sleep(2 ** attempt)

    progress[0] += 1
    if progress[0] % 100 == 0:
        save_state(state)
        print(f"  {progress[0]}/{progress[1]}", flush=True)


async def run_async(pairs, state):
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(CONCURRENCY)
    pending = [(d, i, q, a) for d, i, q, a in pairs if f"{d}--{i}" not in state]
    progress = [0, len(pairs)]
    tasks = [
        asyncio.create_task(process_pair(client, semaphore, d, i, q, a, state, progress))
        for d, i, q, a in pending
    ]
    await asyncio.gather(*tasks)


def write_output(pairs, state):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for dataset, i, q, a in pairs:
        datasets.setdefault(dataset, []).append((i, q, a))

    total_written = total_recovered = total_discarded = 0
    for dataset, indexed_pairs in sorted(datasets.items()):
        items = []
        discarded = recovered = 0
        for i, orig_q, orig_a in sorted(indexed_pairs):
            result = state.get(f"{dataset}--{i}")
            if not result or result.get("discard"):
                discarded += 1
                continue
            if result.get("clean"):
                items.append({"conversations": [
                    {"role": "user", "content": orig_q},
                    {"role": "assistant", "content": orig_a},
                ]})
            elif "pairs" in result:
                for pair in result["pairs"]:
                    items.append({"conversations": [
                        {"role": "user", "content": pair["q"]},
                        {"role": "assistant", "content": pair["a"]},
                    ], "recovered": True})
                recovered += 1

        out = OUTPUT_DIR / f"{dataset}.json"
        out.write_text(json.dumps(items, indent=2, ensure_ascii=False))
        total_written += len(items)
        total_recovered += recovered
        total_discarded += discarded
        print(f"  {dataset}: {len(items)} pairs ({recovered} bleeds recovered, {discarded} discarded)")

    print(f"\nTotal: {total_written} pairs, {total_recovered} bleeds recovered, {total_discarded} discarded")


async def test_run(pairs, seed, size):
    import random
    random.seed(seed)
    sample = random.sample(pairs, size)
    client = anthropic.AsyncAnthropic()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    samples_dir = ROOT / "samples" / "bleed"
    samples_dir.mkdir(parents=True, exist_ok=True)
    out_path = samples_dir / f"{ts}_seed{seed}_n{size}.txt"
    header = [f"model: {MODEL}", f"seed:  {seed}", f"n:     {size}", "", "--- SYSTEM PROMPT ---", SYSTEM_PROMPT, "--- END SYSTEM PROMPT ---", ""]
    out_path.write_text("\n".join(header))

    async def process(dataset, i, q, a):
        msg = await client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Q: {q}\nA: {a}"}],
        )
        text = next((b.text for b in msg.content if b.type == "text"), "").strip()
        result = parse_response(text, q, a)

        if result.get("clean"):
            status, detail = "CLEAN", ""
        elif result.get("discard"):
            status, detail = "DISCARD", ""
        else:
            pairs_out = result.get("pairs", [])
            status = f"RECOVERED ({len(pairs_out)} pairs)"
            detail = "\n".join(
                f"  PAIR {j+1} Q: {p['q']}\n  PAIR {j+1} A: {p['a']}"
                for j, p in enumerate(pairs_out)
            )

        lines = [f"[{dataset}--{i}] {status}", f"  IN Q: {q}", f"  IN A: {a}"]
        if detail:
            lines.append(detail)
        lines.append("")

        with out_path.open("a") as f:
            f.write("\n".join(lines) + "\n")

    await asyncio.gather(*[process(d, i, q, a) for d, i, q, a in sample])
    print(f"Wrote {out_path}")


# --- Retry-bleed pass (runs on output/ocr/) ---

def load_retry_state():
    if RETRY_STATE_FILE.exists():
        return json.loads(RETRY_STATE_FILE.read_text())
    return {}


def save_retry_state(state):
    RETRY_STATE_FILE.write_text(json.dumps(state))


def load_ocr_pairs():
    pairs = []
    for json_file in sorted(OCR_DIR.glob("*.json")):
        dataset = json_file.stem
        items = json.loads(json_file.read_text())
        for i, item in enumerate(items):
            convs = item["conversations"]
            q = next(c["content"] for c in convs if c["role"] == "user")
            a = next(c["content"] for c in convs if c["role"] == "assistant")
            pairs.append((dataset, i, q, a))
    return pairs


def find_bleed_candidates(pairs):
    return [(d, i, q, a) for d, i, q, a in pairs
            if CANDIDATE_RE.search(a) or CANDIDATE_RE.search(q[len(q) // 2:])]


def try_algorithmic_split(q, a):
    """Split on OCR number markers. Returns list of {q, a} dicts or None."""
    parts = SPLIT_RE.split(q)
    if len(parts) < 2:
        return None
    result = []
    for idx, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        if idx == len(parts) - 1:
            result.append({"q": part, "a": a})
        else:
            q_end = part.find("?")
            if q_end == -1:
                return None
            pair_q = part[:q_end + 1].strip()
            pair_a = part[q_end + 1:].strip()
            if not pair_q or not pair_a:
                return None
            result.append({"q": pair_q, "a": pair_a})
    return result if len(result) >= 2 else None


def _estimate_tokens(q, a):
    return (len(q) + len(a)) // 4


def _pack_batches(candidates, token_budget=1200):
    batches, current, current_tokens = [], [], 0
    for item in candidates:
        tokens = _estimate_tokens(item[2], item[3])
        if current and current_tokens + tokens > token_budget:
            batches.append(current)
            current, current_tokens = [], 0
        current.append(item)
        current_tokens += tokens
    if current:
        batches.append(current)
    return batches


async def _detect_batch(client, semaphore, batch, state):
    keys = [f"{d}--{i}" for d, i, _, _ in batch]
    payload = json.dumps([{"i": idx, "q": q, "a": a}
                          for idx, (_, _, q, a) in enumerate(batch)])
    async with semaphore:
        for attempt in range(5):
            try:
                msg = await client.messages.create(
                    model=MODEL,
                    max_tokens=4096,
                    system=RETRY_DETECT_PROMPT,
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
                        state[keys[idx]] = {"bleed": bool(r.get("bleed", False))}
                return
            except anthropic.RateLimitError:
                await asyncio.sleep(2 ** attempt)
            except Exception:
                if attempt == 4:
                    for key in keys:
                        if key not in state:
                            state[key] = {"bleed": False}
                else:
                    await asyncio.sleep(2 ** attempt)


async def _recover_one(client, semaphore, dataset, i, q, a, state):
    key = f"{dataset}--{i}"
    async with semaphore:
        for attempt in range(5):
            try:
                msg = await client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": f"Q: {q}\nA: {a}"}],
                )
                text = next((b.text for b in msg.content if b.type == "text"), "").strip()
                state[key]["recovery"] = parse_response(text, q, a)
                return
            except anthropic.RateLimitError:
                await asyncio.sleep(2 ** attempt)
            except Exception:
                if attempt == 4:
                    state[key]["recovery"] = {"clean": True}
                else:
                    await asyncio.sleep(2 ** attempt)


async def retry_bleed(pairs, state):
    candidates = find_bleed_candidates(pairs)
    print(f"  {len(candidates)} bleed candidates")

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(CONCURRENCY)

    # Phase 1: batched boolean detection
    pending = [(d, i, q, a) for d, i, q, a in candidates if f"{d}--{i}" not in state]
    if pending:
        batches = _pack_batches(pending)
        print(f"  Phase 1: {len(batches)} detection batches...")
        tasks = [asyncio.create_task(_detect_batch(client, semaphore, b, state))
                 for b in batches]
        await asyncio.gather(*tasks)
        save_retry_state(state)

    confirmed = [(d, i, q, a) for d, i, q, a in candidates
                 if state.get(f"{d}--{i}", {}).get("bleed")]
    print(f"  {len(confirmed)} confirmed bleeds")

    # Phase 2: algorithmic split or model recovery
    needs_model = []
    for d, i, q, a in confirmed:
        key = f"{d}--{i}"
        if "recovery" in state[key]:
            continue
        split = try_algorithmic_split(q, a)
        if split:
            state[key]["recovery"] = {"pairs": split}
        else:
            needs_model.append((d, i, q, a))

    if needs_model:
        print(f"  Phase 2: model recovery for {len(needs_model)} pairs...")
        tasks = [asyncio.create_task(_recover_one(client, semaphore, d, i, q, a, state))
                 for d, i, q, a in needs_model]
        await asyncio.gather(*tasks)
        save_retry_state(state)


def write_retry_output(state):
    total_split = total_discarded = 0
    for f in sorted(OCR_DIR.glob("*.json")):
        dataset = f.stem
        existing = json.loads(f.read_text())
        new_items = []
        for i, item in enumerate(existing):
            key = f"{dataset}--{i}"
            result = state.get(key)
            if not result or not result.get("bleed"):
                new_items.append(item)
                continue
            recovery = result.get("recovery", {})
            if "pairs" in recovery:
                valid = [p for p in recovery["pairs"] if p.get("q") and p.get("a")]
                if len(valid) >= 2:
                    for p in valid:
                        new_items.append({"conversations": [
                            {"role": "user", "content": p["q"]},
                            {"role": "assistant", "content": p["a"]},
                        ], "recovered": True})
                    total_split += 1
                    continue
            if recovery.get("discard"):
                total_discarded += 1
                continue
            new_items.append(item)
        f.write_text(json.dumps(new_items, indent=2, ensure_ascii=False))
        delta = len(new_items) - len(existing)
        if delta:
            print(f"  {dataset}: {len(existing)} → {len(new_items)} ({delta:+d})")
    print(f"\nTotal: {total_split} bleeds split, {total_discarded} discarded")
