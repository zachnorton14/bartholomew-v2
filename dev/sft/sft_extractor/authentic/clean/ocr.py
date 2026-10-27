"""Pass 1: Fix OCR artifacts in extracted Q&A pairs using DeepSeek via Anthropic SDK.

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
INPUT_DIR = ROOT / "output" / "bleed"
OUTPUT_DIR = ROOT / "output" / "ocr"
STATE_FILE = ROOT / "state" / "ocr.json"
MODEL = "claude-haiku-4-5"  # maps to deepseek-v4-flash via ANTHROPIC_BASE_URL
MAX_TOKENS = 2048
CONCURRENCY = 20
RETRY_MAX_TOKENS = 16000
MAX_PAIR_TOKENS = 2048   # pairs exceeding this are too long for training context — discard
RETRY_TOKEN_BUDGET = 800  # max estimated q+a input tokens per batch; keeps batches small enough for reasoning model

SYSTEM_PROMPT = """\
You are an OCR correction assistant for public-domain texts published before 1930.
Your job is to fix mechanical scanning errors. Be aggressive about fixing — the source
texts are legitimate pre-1930s works, so fix confidently.

Fix all of the following:
- Missing or extra spaces (Whatis → What is, "Biography ?" → "Biography?")
- Single-character letter substitutions (ts → is, ln → in, rn → m, 4 → A when clearly a letter)
- Broken hyphenation across lines (na- tions → nations)
- Unicode/encoding artifacts (Â¢ → ¢, â€™ → ', Ã© → é)
- Obvious misrecognized words (tbe → the, liow → how, witb → with)
- Missing terminal period on an answer that is clearly a complete sentence
- Embedded page references or section headers in answer text (e.g. "FRANCE. 107", "6 EGYPT.", "244 COMMON SCHOOL QUESTION BOOK." — remove these entirely)

Never alter grammar or word choice. Fix only what is clearly a scanning artifact.
"Whom", archaic verb forms, and pre-1930s constructions are correct as-is.

Flag with "flagged": true if any word you chose as a replacement could be an anachronism —
a word, term, or concept that did not exist before 1930. Do not flag for space fixes,
punctuation, encoding artifacts, or clear single-character swaps.

Discard if the text is so severely garbled that no meaningful content can be recovered, or
if either field contains structural bleed — embedded answer text followed by additional questions
(e.g. "Q 1. ... A. ... Q 2." patterns, or "A— ..." mid-question), or if the answer contains
a multi-row numerical table or columnar data — tables are not meaningful prose for training.
When in doubt on OCR artifacts, fix and keep. Do not discard pairs that are merely messy.

Respond with valid JSON only:
  Corrected or unchanged:       {"q": "...", "a": "..."}
  Possible anachronism in fix:  {"q": "...", "a": "...", "flagged": true}
  Truly unrecoverable:          {"discard": true}

Examples:

Input — Q: Whatis History? / A: A recital of what has happened respecting na- tions.
Output: {"q": "What is History?", "a": "A recital of what has happened respecting nations."}

Input — Q: What is Biography ? / A: The history of a single individual.
Output: {"q": "What is Biography?", "a": "The history of a single individual."}

Input — Q: What are tbe chief sources ? / A: Records, Monuments, and I^egends.
Output: {"q": "What are the chief sources?", "a": "Records, Monuments, and Legends."}

Input — Q: §§§ xh§ pr¡n§¡p / A: Th§ §l¿ve §§ct¡¿n
Output: {"discard": true}
"""

RETRY_SYSTEM_PROMPT = """\
You are an OCR correction assistant for public-domain texts published before 1930.
You are reviewing a batch of Q&A pairs that a previous pass flagged but did NOT fix.
Your job now is to actually fix them — do not echo text back unchanged unless it is
already perfect. Be aggressive about fixing — the source texts are legitimate
pre-1930s works, so fix confidently.

Fix all of the following:
- Missing or extra spaces (Whatis → What is, "Biography ?" → "Biography?")
- Single-character letter substitutions (ts → is, ln → in, rn → m, 4 → A when clearly a letter)
- Broken hyphenation across lines (na- tions → nations)
- Unicode/encoding artifacts (Â¢ → ¢, â€™ → ', Ã© → é, â€” → —, â€˜/â€™ → ' or ‘’)
- Obvious misrecognized words (tbe → the, liow → how, witb → with)
- Missing terminal period on an answer that is clearly a complete sentence
- Embedded page references or section headers anywhere in the text (e.g. "FRANCE. 107",
  "6 EGYPT.", "244 COMMON SCHOOL QUESTION BOOK." — remove these entirely)

Never alter grammar or word choice. Fix only what is clearly a scanning artifact.
"Whom", archaic verb forms, and pre-1930s constructions are correct as-is.

Set "flagged": true ONLY if you personally replaced a word/term with something that
could be an anachronism — a word, term, or concept that did not exist before 1930.
Do NOT flag merely because the subject matter is war, politics, or sensitive history.
If you made no risky word substitution, set "flagged": false.

Discard if the text is so severely garbled that no meaningful content can be recovered, or
if either field contains structural bleed — embedded answer text followed by additional
questions (e.g. "Q 1. ... A. ... Q 2." patterns, or "A— ..." mid-question), or if the
answer contains a multi-row numerical table or columnar data (legible or garbled) — tables
are not meaningful prose for training and should not be preserved.
When in doubt on OCR artifacts, fix and keep. Do not discard pairs that are merely messy.

You will receive a JSON array of items, each with an "i" index, "q", and "a". Respond with
a JSON array of the same length, each item matching one input by "i":
  Fixed:           {"i": 0, "q": "...", "a": "..."}
  Anachronism risk:{"i": 0, "q": "...", "a": "...", "flagged": true}
  Unrecoverable:   {"i": 0, "discard": true}

Respond with the JSON array only, no other text.
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


async def clean_pair(client, semaphore, dataset, i, q, a, state, progress):
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
                if text.startswith("```"):
                    text = text.split("```", 2)[1]
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.rstrip("`").strip()
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    state[key] = {"q": q, "a": a, "flagged": True}
                    break
                if parsed.get("discard"):
                    state[key] = {"discard": True}
                elif "q" not in parsed or "a" not in parsed:
                    state[key] = {"q": q, "a": a, "flagged": True}
                else:
                    state[key] = parsed
                break
            except anthropic.RateLimitError:
                await asyncio.sleep(2 ** attempt)
            except Exception:
                if attempt == 4:
                    state[key] = {"q": q, "a": a, "flagged": True}
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
        asyncio.create_task(clean_pair(client, semaphore, d, i, q, a, state, progress))
        for d, i, q, a in pending
    ]
    await asyncio.gather(*tasks)


def _estimate_tokens(q, a):
    return (len(q) + len(a)) // 4


def _pack_batches(items, token_budget=RETRY_TOKEN_BUDGET):
    """Greedy bin-packing: fit as many pairs as possible per batch within token_budget.
    Items with q+a > MAX_PAIR_TOKENS are discarded here (too long for training context).
    Each item is a tuple ending in (..., q, a)."""
    batches, current, current_tokens = [], [], 0
    for item in items:
        q, a = item[-2], item[-1]
        tokens = _estimate_tokens(q, a)
        if tokens > MAX_PAIR_TOKENS:
            continue
        if current and current_tokens + tokens > token_budget:
            batches.append(current)
            current, current_tokens = [], 0
        current.append(item)
        current_tokens += tokens
    if current:
        batches.append(current)
    return batches


def _strip_code_fence(text):
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rstrip("`").strip()
    return text


async def clean_batch(client, semaphore, dataset, batch, state, progress):
    """batch: list of (i, q, a) tuples, all from the same dataset."""
    keys = [f"{dataset}--{i}" for i, _, _ in batch]
    payload = json.dumps([{"i": idx, "q": q, "a": a} for idx, (i, q, a) in enumerate(batch)])

    async with semaphore:
        for attempt in range(5):
            try:
                msg = await client.messages.create(
                    model=MODEL,
                    max_tokens=RETRY_MAX_TOKENS,
                    system=RETRY_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": payload}],
                )
                text = _strip_code_fence(
                    next((b.text for b in msg.content if b.type == "text"), "").strip()
                )
                try:
                    results = json.loads(text)
                except json.JSONDecodeError:
                    results = []

                by_idx = {r["i"]: r for r in results if isinstance(r, dict) and "i" in r}
                for idx, (i, q, a) in enumerate(batch):
                    key = keys[idx]
                    r = by_idx.get(idx)
                    if not r:
                        state[key] = {"q": q, "a": a, "flagged": True}
                    elif r.get("discard"):
                        state[key] = {"discard": True}
                    elif "q" not in r or "a" not in r:
                        state[key] = {"q": q, "a": a, "flagged": True}
                    else:
                        state[key] = {"q": r["q"], "a": r["a"], "flagged": bool(r.get("flagged"))}
                break
            except anthropic.RateLimitError:
                await asyncio.sleep(2 ** attempt)
            except Exception:
                if attempt == 4:
                    for idx, (i, q, a) in enumerate(batch):
                        state[keys[idx]] = {"q": q, "a": a, "flagged": True}
                else:
                    await asyncio.sleep(2 ** attempt)

    progress[0] += len(batch)
    save_state(state)
    print(f"  {progress[0]}/{progress[1]}", flush=True)


async def retry_flagged(pairs, state):
    """Re-process only currently-flagged pairs using greedy bin-packing batches."""
    flagged_pairs = [
        (d, i, q, a) for d, i, q, a in pairs
        if state.get(f"{d}--{i}", {}).get("flagged")
    ]

    by_dataset = {}
    for d, i, q, a in flagged_pairs:
        by_dataset.setdefault(d, []).append((i, q, a))

    batches = []
    for d, items in by_dataset.items():
        for batch in _pack_batches(items):
            batches.append((d, batch))

    print(f"Retrying {len(flagged_pairs)} flagged pairs → {len(batches)} packed batches...")

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(CONCURRENCY)
    progress = [0, len(flagged_pairs)]
    tasks = [
        asyncio.create_task(clean_batch(client, semaphore, d, batch, state, progress))
        for d, batch in batches
    ]
    await asyncio.gather(*tasks)
    save_state(state)


async def test_retry_flagged(pairs, state, seed, size, batch_size=None):
    import random
    flagged_pairs = [
        (d, i, q, a) for d, i, q, a in pairs
        if state.get(f"{d}--{i}", {}).get("flagged")
        and _estimate_tokens(q, a) <= MAX_PAIR_TOKENS
    ]
    random.seed(seed)
    sample = random.sample(flagged_pairs, min(size, len(flagged_pairs)))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    samples_dir = ROOT / "samples" / "ocr"
    samples_dir.mkdir(parents=True, exist_ok=True)
    out_path = samples_dir / f"{ts}_retry_seed{seed}_n{size}.txt"
    header = [
        f"model: {MODEL}", f"seed:  {seed}", f"n:     {size}", f"batch: {batch_size}",
        f"pool:  {len(flagged_pairs)} currently flagged pairs", "",
        "--- RETRY SYSTEM PROMPT ---", RETRY_SYSTEM_PROMPT, "--- END RETRY SYSTEM PROMPT ---", "",
    ]
    out_path.write_text("\n".join(header))

    client = anthropic.AsyncAnthropic()

    async def process(batch):
        payload = json.dumps([{"i": idx, "q": q, "a": a} for idx, (d, i, q, a) in enumerate(batch)])
        msg = await client.messages.create(
            model=MODEL,
            max_tokens=RETRY_MAX_TOKENS,
            system=RETRY_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": payload}],
        )
        raw = _strip_code_fence(next((b.text for b in msg.content if b.type == "text"), "").strip())
        parse_error = None
        try:
            results = json.loads(raw)
            by_idx = {r["i"]: r for r in results if isinstance(r, dict) and "i" in r}
        except json.JSONDecodeError as e:
            by_idx = {}
            parse_error = f"[BATCH PARSE FAILURE — {e}]\n  stop_reason: {msg.stop_reason}\n  raw ({len(raw)} chars): {raw[:500]}"

        lines = []
        if parse_error:
            lines.append(parse_error)
            lines.append("")
        for idx, (dataset, i, q, a) in enumerate(batch):
            r = by_idx.get(idx)
            if r is None:
                out_q, out_a, flagged = "[PARSE ERROR / MISSING]", "", ""
            elif r.get("discard"):
                out_q, out_a, flagged = "[DISCARD]", "", ""
            else:
                out_q = r.get("q", "")
                out_a = r.get("a", "")
                flagged = " [FLAGGED]" if r.get("flagged") else ""
            lines += [
                f"[{dataset}--{i}]{flagged}",
                f"  IN  Q: {q}",
                f"  IN  A: {a}",
                f"  OUT Q: {out_q}",
                f"  OUT A: {out_a}",
                "",
            ]
        with out_path.open("a") as f:
            f.write("\n".join(lines) + "\n")

    batches = _pack_batches(sample, batch_size or RETRY_TOKEN_BUDGET)
    await asyncio.gather(*[process(b) for b in batches])
    print(f"Wrote {out_path}")


def write_output(pairs, state):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for dataset, i, q, a in pairs:
        datasets.setdefault(dataset, []).append((i, q, a))

    total_written = total_flagged = total_discarded = 0
    for dataset, indexed_pairs in sorted(datasets.items()):
        items = []
        discarded = 0
        for i, orig_q, orig_a in sorted(indexed_pairs):
            result = state.get(f"{dataset}--{i}")
            if not result or result.get("discard"):
                discarded += 1
                continue
            if _estimate_tokens(result["q"], result["a"]) > MAX_PAIR_TOKENS:
                discarded += 1
                continue
            item = {
                "conversations": [
                    {"role": "user", "content": result["q"]},
                    {"role": "assistant", "content": result["a"]},
                ]
            }
            if result["q"] != orig_q or result["a"] != orig_a:
                item["changed"] = True
            if result.get("flagged"):
                item["flagged"] = True
                total_flagged += 1
            items.append(item)

        out = OUTPUT_DIR / f"{dataset}.json"
        out.write_text(json.dumps(items, indent=2, ensure_ascii=False))
        total_written += len(items)
        total_discarded += discarded
        flagged = sum(1 for it in items if it.get("flagged"))
        changed = sum(1 for it in items if it.get("changed"))
        print(f"  {dataset}: {len(items)} pairs ({changed} changed, {flagged} flagged, {discarded} discarded)")

    print(f"\nTotal: {total_written} pairs, {total_flagged} flagged, {total_discarded} discarded")


async def test_run(pairs, seed, size):
    import random
    random.seed(seed)
    sample = random.sample(pairs, size)
    client = anthropic.AsyncAnthropic()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    samples_dir = ROOT / "samples" / "ocr"
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
        raw = next((b.text for b in msg.content if b.type == "text"), "").strip()
        try:
            parsed = json.loads(raw)
            if parsed.get("discard"):
                out_q, out_a = "[DISCARD]", ""
            else:
                out_q = parsed.get("q", "")
                out_a = parsed.get("a", "")
            flagged = " [FLAGGED]" if parsed.get("flagged") else ""
        except (json.JSONDecodeError, AttributeError):
            out_q, out_a, flagged = "[PARSE ERROR]", raw, ""
        entry = "\n".join([
            f"[{dataset}--{i}]{flagged}",
            f"  IN  Q: {q}",
            f"  IN  A: {a}",
            f"  OUT Q: {out_q}",
            f"  OUT A: {out_a}",
            "",
        ])
        with out_path.open("a") as f:
            f.write(entry + "\n")

    await asyncio.gather(*[process(d, i, q, a) for d, i, q, a in sample])
    print(f"Wrote {out_path}")
