"""Pass 3: Detect and enrich context-bare opening questions.

A question is "bare" if it cannot be understood without knowing the subject domain.
Phase 1 detects bare opening questions via batched boolean model calls.
Phase 2 rewrites them using per-dataset domain context.

Input: output/paired/   Output: output/enriched/

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
INPUT_DIR = ROOT / "output" / "paired"
OUTPUT_DIR = ROOT / "output" / "enriched"
STATE_FILE = ROOT / "state" / "enrich.json"
MODEL = "claude-haiku-4-5"
CONCURRENCY = 20
DETECT_MAX_TOKENS = 8192
ENRICH_MAX_TOKENS = 4096
DETECT_TOKEN_BUDGET = 2000
ENRICH_TOKEN_BUDGET = 1500

DATASET_CONTEXTS = {
    "1001-questions":      "world history from ancient civilizations through the early 20th century",
    "advanced-questions":  "general knowledge spanning literature, science, and history",
    "agriculture":         "farming, agriculture, and rural science",
    "astronomy":           "astronomy and the study of celestial bodies",
    "botany":              "botany and plant science",
    "brewers-guide":       "everyday science, natural philosophy, and household phenomena",
    "chemistry":           "chemistry and chemical processes",
    "civil-war":           "World War I history, military affairs, and international politics",
    "common-core":         "American colonial and national history",
    "constitution":        "United States constitutional law and government",
    "electricity":         "electrical engineering and applied physics",
    "engineering":         "mechanical and steam engineering",
    "ethics":              "moral philosophy and ethics",
    "familiar-things":     "natural science and everyday curiosities",
    "grammar":             "English grammar and usage",
    "investors":           "investment, finance, and business law",
    "laborers":            "labor economics, workers' rights, and political economy",
    "logic":               "formal logic and argumentation",
    "music":               "music theory and notation",
    "mythology":           "Greek and Roman mythology",
    "new-york-bar":        "New York bar examination and law",
    "patriotism":          "American civics and patriotism",
    "school-bulletin":     "general education across science, history, and literature",
    "seeleys":             "American and English literature",
    "stokers":             "boiler operation and steam engineering",
    "symbological":        "biblical prophecy and symbolic interpretation",
    "world-history":       "world history from creation through the early modern period",
}

DETECT_PROMPT = """\
You receive opening questions from pre-1930s educational Q&A texts.
Classify each question as bare, ambiguous, or clear.

"bare" — cannot be understood without prior context:
  - no subject at all: "What is the effect?", "Why not?", "Describe their shape?"
  - continuative language implying prior discussion: "What acts did they continue to commit?", "What further steps were taken?"
  - definite reference to an entity not universally known: "What was the character of the Assembly?", "What was the result of the Act?"

"ambiguous" — has a subject but it is domain-dependent; meaning shifts by field:
  "What is a disk?", "What is a cell?", "Define the term."

"clear" — fully self-contained regardless of domain:
  "What is arsenic?", "Who was Napoleon?", "What causes thunder?"

Input: JSON array [{"i": 0, "q": "..."}, ...]
Output JSON only: [{"i": 0, "status": "clear"}, ...]
"""

ENRICH_PROMPT = """\
You receive opening questions from a pre-1930s text about {context}.
Rewrite each bare question to be self-contained by incorporating the subject domain.
Keep rewrites concise and in the same register as the original.

Input: JSON array [{{"i": 0, "q": "...", "a": "..."}}, ...]
Output JSON only: [{{"i": 0, "q": "rewritten question"}}, ...]
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


def _opening_question(item):
    return next(c["content"] for c in item["conversations"] if c["role"] == "user")


def _estimate_tokens(text):
    return len(text) // 4


def _pack_batches(items, token_budget):
    batches, current, current_tokens = [], [], 0
    for item in items:
        tokens = _estimate_tokens(item[2])  # item = (dataset, i, q)
        if current and current_tokens + tokens > token_budget:
            batches.append(current)
            current, current_tokens = [], 0
        current.append(item)
        current_tokens += tokens
    if current:
        batches.append(current)
    return batches


async def _detect_batch(client, semaphore, batch, state):
    # batch: list of (dataset, i, q)
    keys = [f"{d}--{i}" for d, i, _ in batch]
    payload = json.dumps([{"i": idx, "q": q} for idx, (_, _, q) in enumerate(batch)])
    async with semaphore:
        for attempt in range(5):
            try:
                msg = await client.messages.create(
                    model=MODEL,
                    max_tokens=DETECT_MAX_TOKENS,
                    system=DETECT_PROMPT,
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
                        status = r.get("status", "clear")
                        state[keys[idx]] = {"status": status}
                return
            except anthropic.RateLimitError:
                await asyncio.sleep(2 ** attempt)
            except Exception:
                if attempt == 4:
                    for key in keys:
                        if key not in state:
                            state[key] = {"status": "clear"}
                else:
                    await asyncio.sleep(2 ** attempt)


async def _enrich_batch(client, semaphore, dataset, batch, state):
    # batch: list of (dataset, i, q, a) — all same dataset
    context = DATASET_CONTEXTS.get(dataset, dataset)
    keys = [f"{d}--{i}" for d, i, *_ in batch]
    payload = json.dumps([{"i": idx, "q": q, "a": a} for idx, (_, _, q, a) in enumerate(batch)])
    system = ENRICH_PROMPT.format(context=context)
    async with semaphore:
        for attempt in range(5):
            try:
                msg = await client.messages.create(
                    model=MODEL,
                    max_tokens=ENRICH_MAX_TOKENS,
                    system=system,
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
                        rewritten = r.get("q", "").strip()
                        if rewritten:
                            state[keys[idx]]["rewritten"] = rewritten
                return
            except anthropic.RateLimitError:
                await asyncio.sleep(2 ** attempt)
            except Exception:
                if attempt == 4:
                    pass  # leave rewritten absent; write_output will use original
                else:
                    await asyncio.sleep(2 ** attempt)


async def run_async(datasets, state):
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(CONCURRENCY)

    # Phase 1: detect bare opening questions
    all_openings = []
    for dataset, items in datasets.items():
        for i, item in enumerate(items):
            key = f"{dataset}--{i}"
            if key not in state:
                q = _opening_question(item)
                all_openings.append((dataset, i, q))

    if all_openings:
        batches = _pack_batches(all_openings, DETECT_TOKEN_BUDGET)
        print(f"  Phase 1: {len(all_openings)} opening questions, {len(batches)} detection batches...")
        tasks = [asyncio.create_task(_detect_batch(client, semaphore, b, state))
                 for b in batches]
        await asyncio.gather(*tasks)
        save_state(state)

    n_bare = sum(1 for d, items in datasets.items() for i in range(len(items))
                 if state.get(f"{d}--{i}", {}).get("status") == "bare")
    n_ambiguous = sum(1 for d, items in datasets.items() for i in range(len(items))
                      if state.get(f"{d}--{i}", {}).get("status") == "ambiguous")
    print(f"  {n_bare} bare, {n_ambiguous} ambiguous detected")

    def _opening_answer(item):
        return next((c["content"] for c in item["conversations"] if c["role"] == "assistant"), "")

    # Phase 2: enrich ambiguous (all positions) and bare i==0 fallbacks
    to_enrich = [(d, i, _opening_question(datasets[d][i]), _opening_answer(datasets[d][i]))
                 for d, items in datasets.items()
                 for i in range(len(items))
                 if (state.get(f"{d}--{i}", {}).get("status") == "ambiguous"
                     or (state.get(f"{d}--{i}", {}).get("status") == "bare" and i == 0))
                 and "rewritten" not in state.get(f"{d}--{i}", {})]

    if to_enrich:
        by_dataset = {}
        for d, i, q, a in to_enrich:
            by_dataset.setdefault(d, []).append((d, i, q, a))

        enrich_batches = []
        for d, items in by_dataset.items():
            for batch in _pack_batches(items, ENRICH_TOKEN_BUDGET):
                enrich_batches.append((d, batch))

        print(f"  Phase 2: {len(to_enrich)} to enrich ({n_ambiguous} ambiguous + i=0 bare fallbacks), {len(enrich_batches)} batches...")
        tasks = [asyncio.create_task(_enrich_batch(client, semaphore, d, b, state))
                 for d, b in enrich_batches]
        await asyncio.gather(*tasks)
        save_state(state)


def write_output(datasets, state):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_attached = total_enriched = 0

    for dataset, items in sorted(datasets.items()):
        out_items = []
        attached = enriched = 0

        for i, item in enumerate(items):
            key = f"{dataset}--{i}"
            result = state.get(key, {})
            status = result.get("status", "clear")
            needs = status in ("bare", "ambiguous")

            if status == "bare" and i > 0 and out_items:
                # Bare with predecessor: attach (chain grows naturally)
                last = out_items[-1]
                merged = dict(last)
                merged["conversations"] = list(last["conversations"]) + list(item["conversations"])
                merged["chained"] = True
                out_items[-1] = merged
                attached += 1
            elif status in ("bare", "ambiguous"):
                # bare i==0 fallback or ambiguous (any position): enrich with domain context
                rewritten = result.get("rewritten")
                if rewritten:
                    new_item = dict(item)
                    convs = list(item["conversations"])
                    first_user = next(j for j, c in enumerate(convs) if c["role"] == "user")
                    convs = list(convs)
                    convs[first_user] = {"role": "user", "content": rewritten}
                    new_item["conversations"] = convs
                    new_item["enriched"] = True
                    out_items.append(new_item)
                    enriched += 1
                else:
                    out_items.append(item)
            else:
                out_items.append(item)

        out = OUTPUT_DIR / f"{dataset}.json"
        out.write_text(json.dumps(out_items, indent=2, ensure_ascii=False))
        total_attached += attached
        total_enriched += enriched
        print(f"  {dataset}: {attached} attached, {enriched} enriched")

    print(f"\nTotal: {total_attached} attached, {total_enriched} enriched")


async def test_run(datasets, seed, size):
    import random
    random.seed(seed)

    all_openings = [(d, i, _opening_question(item))
                    for d, items in datasets.items()
                    for i, item in enumerate(items)]
    sample = random.sample(all_openings, min(size, len(all_openings)))

    client = anthropic.AsyncAnthropic()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    samples_dir = ROOT / "samples" / "enrich"
    samples_dir.mkdir(parents=True, exist_ok=True)
    out_path = samples_dir / f"{ts}_seed{seed}_n{size}.txt"

    header = [f"model: {MODEL}", f"seed:  {seed}", f"n:     {size}", "",
              "--- DETECT PROMPT ---", DETECT_PROMPT, "--- END DETECT PROMPT ---", ""]
    out_path.write_text("\n".join(header))

    # Send as one batch for test
    payload = json.dumps([{"i": idx, "q": q} for idx, (_, _, q) in enumerate(sample)])
    msg = await client.messages.create(
        model=MODEL,
        max_tokens=DETECT_MAX_TOKENS,
        system=DETECT_PROMPT,
        messages=[{"role": "user", "content": payload}],
    )
    text = next((b.text for b in msg.content if b.type == "text"), "").strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rstrip("`").strip()
    try:
        results = {r["i"]: r.get("status", "clear") for r in json.loads(text)}
    except Exception:
        results = {}

    lines = []
    for idx, (dataset, i, q) in enumerate(sample):
        status = results.get(idx, "?")
        lines.append(f"[{dataset}--{i}] {str(status).upper()}")
        lines.append(f"  Q: {q}")
        lines.append("")

    out_path.write_text("\n".join(header + lines))
    print(f"Wrote {out_path}")
