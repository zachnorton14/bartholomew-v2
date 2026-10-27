"""Generate `working`-prompt synthetic questions through arbitrary opencode Go models.

Experiment harness for comparing model outputs on the SAME matched pairs and prompt,
so a metric difference is attributable to the model rather than to prompt or sample.
Reuses the frozen `working` prompt and batching logic from synth.questions; writes to
output/synth/questions_working__<model>.json (double underscore so it never collides
with a style file, and metrics.report still renders a readable name).

Handles both opencode endpoint families:
  anthropic  (/v1/messages)         -> MiniMax*, Qwen*      via the anthropic SDK
  openai     (/v1/chat/completions) -> Grok, GLM, Kimi, DeepSeek V4, MiMo via httpx

Auth: set OPENCODE_API_KEY (preferred) or ANTHROPIC_API_KEY.

    export OPENCODE_API_KEY=<key>
    python -m synth.gen_models --models minimax-m3 deepseek-v4-pro kimi-k3 --count 500
"""
import os
import re
import sys
import json
import asyncio
import argparse
from pathlib import Path

import httpx

from synth.questions import (
    PROMPTS, load_authentic_pairs, sample_pairs, _pack_batches, _strip_fence,
)

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "synth" / "output"
STATE_DIR = ROOT / "synth" / "state"
BASE_URL = "https://opencode.ai/zen/go"
STYLE = "working"
CONCURRENCY = 16
MAX_TOKENS = 16384

def _family(model):
    # All opencode models are reachable via /v1/chat/completions with a Bearer key.
    # The documented /v1/messages endpoint 401s regardless of headers, so we route
    # everything (including the "Anthropic-compat" minimax/qwen) through the OpenAI path.
    return "openai"


def _from_dotenv(name):
    """Read NAME from ROOT/.env (KEY=value or `export KEY=value`), else None."""
    f = ROOT / ".env"
    if not f.exists():
        return None
    for line in f.read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):]
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _api_key():
    # opencode key only — NOT ANTHROPIC_API_KEY, which is a real Anthropic key in
    # this environment and would 401 against opencode.
    key = os.environ.get("OPENCODE_API_KEY") or _from_dotenv("OPENCODE_API_KEY")
    if not key:
        sys.exit("Set OPENCODE_API_KEY (env or .env) to the opencode Go key.")
    return key


def _safe(model):
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


def _state_file(model):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return STATE_DIR / f"synthq_model_{_safe(model)}.json"


def _load_state(model):
    f = _state_file(model)
    return json.loads(f.read_text()) if f.exists() else {}


_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def _parse_json_array(text):
    """Extract the JSON array from a reply that may carry <think> blocks, code
    fences, or leading prose (reasoning models via chat/completions do all three)."""
    text = _THINK_RE.sub("", text)
    text = _strip_fence(text).strip()
    if not text.startswith("["):
        i, j = text.find("["), text.rfind("]")
        if i != -1 and j != -1:
            text = text[i:j + 1]
    return json.loads(text)


def _record(state, keys, text, style):
    """Parse a model reply and store {key: {'q': ...}}; return count added."""
    added = 0
    for r in _parse_json_array(text):
        idx = r.get("i")
        q = (r.get("q") or "").strip()
        if isinstance(idx, int) and 0 <= idx < len(keys) and q:
            state[keys[idx]] = {"q": q}
            added += 1
    return added


async def _gen_anthropic(client, sem, batch, model, key, state):
    # Raw /v1/messages with a Bearer header. We bypass the anthropic SDK because it
    # sends x-api-key (and picks up the ambient ANTHROPIC_API_KEY), which opencode
    # rejects — opencode wants Authorization: Bearer on every endpoint.
    keys = [f"{d}--{i}" for d, i, _, _ in batch]
    payload = json.dumps([{"i": n, "a": a} for n, (_, _, _, a) in enumerate(batch)])
    body = {
        "model": model, "max_tokens": MAX_TOKENS, "system": PROMPTS[STYLE],
        "messages": [{"role": "user", "content": payload}],
    }
    headers = {"Authorization": f"Bearer {key}", "anthropic-version": "2023-06-01"}
    async with sem:
        for attempt in range(5):
            try:
                resp = await client.post(f"{BASE_URL}/v1/messages", json=body, headers=headers)
                if resp.status_code == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                blocks = resp.json().get("content", [])
                text = next((b.get("text", "") for b in blocks if b.get("type") == "text"), "").strip()
                if not text:
                    raise ValueError(f"empty (stop={resp.json().get('stop_reason')})")
                _record(state, keys, text, STYLE)
                return
            except Exception as e:
                if attempt == 4:
                    print(f"  [{model}] batch of {len(batch)} failed: {type(e).__name__}: {str(e)[:160]}")
                    return
                await asyncio.sleep(2 ** attempt)


async def _gen_openai(client, sem, batch, model, key, state):
    keys = [f"{d}--{i}" for d, i, _, _ in batch]
    payload = json.dumps([{"i": n, "a": a} for n, (_, _, _, a) in enumerate(batch)])
    body = {
        "model": model, "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": PROMPTS[STYLE]},
            {"role": "user", "content": payload},
        ],
    }
    headers = {"Authorization": f"Bearer {key}"}
    async with sem:
        for attempt in range(5):
            try:
                resp = await client.post(f"{BASE_URL}/v1/chat/completions",
                                         json=body, headers=headers)
                if resp.status_code == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                text = (resp.json()["choices"][0]["message"]["content"] or "").strip()
                if not text:
                    raise ValueError("empty content")
                _record(state, keys, text, STYLE)
                return
            except Exception as e:
                if attempt == 4:
                    print(f"  [{model}] batch of {len(batch)} failed: {type(e).__name__}: {str(e)[:160]}")
                    return
                await asyncio.sleep(2 ** attempt)


async def run_model(pairs, model, key):
    state = _load_state(model)
    pending = [p for p in pairs if f"{p[0]}--{p[1]}" not in state]
    fam = _family(model)
    print(f"[{model}] family={fam}  pending {len(pending)}/{len(pairs)}")
    if not pending:
        return state
    batches = _pack_batches(pending)
    sem = asyncio.Semaphore(CONCURRENCY)
    gen = _gen_anthropic if fam == "anthropic" else _gen_openai
    async with httpx.AsyncClient(timeout=180) as client:
        await asyncio.gather(*[gen(client, sem, b, model, key, state) for b in batches])
    _state_file(model).write_text(json.dumps(state))
    return state


def write_output(pairs, model, state):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for dataset, i, q, a in pairs:
        r = state.get(f"{dataset}--{i}")
        if r:
            items.append({"dataset": dataset, "i": i, "answer": a,
                          "authentic_q": q, "synthetic_q": r["q"]})
    out = OUTPUT_DIR / f"questions_{STYLE}__{_safe(model)}.json"
    out.write_text(json.dumps(items, indent=2, ensure_ascii=False))
    print(f"[{model}] wrote {len(items)} pairs -> {out.name}")


def main():
    ap = argparse.ArgumentParser(description="Generate working-prompt questions across opencode models")
    ap.add_argument("--models", nargs="+", required=True, help="opencode model ids")
    ap.add_argument("--count", type=int, default=500, help="matched pairs to sample")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    key = _api_key()
    pairs = sample_pairs(load_authentic_pairs(), args.count, args.seed)
    print(f"Sampled {len(pairs)} pairs (seed {args.seed}) for {len(args.models)} models\n")
    for model in args.models:
        state = asyncio.run(run_model(pairs, model, key))
        write_output(pairs, model, state)


if __name__ == "__main__":
    main()
