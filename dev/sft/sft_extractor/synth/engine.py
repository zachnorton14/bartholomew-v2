"""Shared generation engine for the content-bound synth routes.

Every route (knowledge_qa, reasoning_qa, stem_reasoning) is the same pipeline —
sample excerpts, batch them, one model call per batch, parse, verify, write rows,
resume from state — differing only in three things: the SYSTEM prompt, where the
excerpts come from, and how a model result becomes an answer. So a route is just a
`Route` config; all the async / batching / retry / state / output machinery lives
here, once.

    ROUTE = engine.Route(name="knowledge_qa", system=SYSTEM, source=source_fn,
                         answer_fn=extract_spans, passthrough=("prose_score",))
    state = engine.load_state(ROUTE)
    await engine.run_async(ROUTE, excerpts, state)
    engine.write_output(ROUTE, excerpts, state)

Env:
    export OPENCODE_API_KEY=<your opencode Go key>   # or put it in ROOT/.env
"""

import asyncio
import json
import os
import re
import sys
import textwrap
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import httpx

WRAP = 88   # column cap for printed / written sample lines

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "synth" / "output"
STATE_DIR = ROOT / "synth" / "state"

# Inference provider, switchable via SFT_PROVIDER. Default "deepseek" = the direct
# DeepSeek API: cheapest per token AND it auto-caches the repeated system prompt (a big
# win here, since every batch of a route resends the same long prompt). The old opencode
# Go transport stays available with SFT_PROVIDER=opencode.
PROVIDER = os.environ.get("SFT_PROVIDER", "deepseek")

if PROVIDER == "opencode":
    BASE_URL = "https://opencode.ai/zen/go"   # opencode Go (OpenAI-style /v1/chat/completions)
    MODEL = "deepseek-v4-pro"                 # opencode model id
    API_KEY_NAME = "OPENCODE_API_KEY"
    MAX_TOKENS = 16384                        # room for a thinking trace + JSON
    DISABLE_THINKING = {"thinking": {"type": "disabled"}}   # opencode flag to drop the trace
else:                                         # deepseek — direct API
    BASE_URL = "https://api.deepseek.com"     # OpenAI-compatible; /v1/chat/completions works
    MODEL = "deepseek-chat"                   # V3, non-thinking by default — cheapest
    API_KEY_NAME = "DS_API_KEY"
    MAX_TOKENS = 8192                          # deepseek-chat output ceiling; extractive JSON fits
    DISABLE_THINKING = {}                     # deepseek-chat needs no thinking flag; routes no-op

HTTP_TIMEOUT = 180                # seconds per request
CONCURRENCY = 20
TOKEN_BUDGET = 1200               # chars/4 per batch
CHECKPOINT_EVERY = 500            # save the resume ledger every N excerpts within a phase
HF_PUSH_EVERY = int(os.environ.get("SFT_HF_PUSH_EVERY", "2000"))   # commit an HF shard every N rows


# --- shared parsing / extraction helpers ---------------------------------------

def _estimate_tokens(text):
    return len(text) // 4


def _pack_batches(items, token_budget=TOKEN_BUDGET):
    batches, cur, tok = [], [], 0
    for it in items:
        t = _estimate_tokens(it["excerpt"])
        if cur and tok + t > token_budget:
            batches.append(cur)
            cur, tok = [], 0
        cur.append(it)
        tok += t
    if cur:
        batches.append(cur)
    return batches


def _strip_fence(text):
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rstrip("`").strip()
    return text


_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def _parse_json_array(text, strict=False):
    """Extract the JSON array from a reply that may carry <think> blocks, code fences,
    or leading prose — reasoning models over chat/completions do all three.  Strict
    mode is for callers whose output contract is a bare array: it rejects every
    wrapper, fence, thought trace, and leading/trailing explanation."""
    if strict:
        text = text.strip()
        if not text.startswith("[") or not text.endswith("]"):
            raise json.JSONDecodeError("expected a bare array", text, 0)
        return json.loads(text)
    text = _THINK_RE.sub("", text)
    text = _strip_fence(text).strip()
    text = re.sub(r"^\[?json\]?\s*", "", text, flags=re.I)   # stray leading "[json]" tag
    if not text.startswith("["):
        i, j = text.find("["), text.rfind("]")
        if i != -1 and j != -1:
            text = text[i:j + 1]
    return json.loads(text)


# --- opencode Go API transport -------------------------------------------------

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
    """The inference key (env or .env) for the active provider — see API_KEY_NAME."""
    key = os.environ.get(API_KEY_NAME) or _from_dotenv(API_KEY_NAME)
    if not key:
        sys.exit(f"Set {API_KEY_NAME} (env or .env) for provider '{PROVIDER}'.")
    return key


class _GoClient:
    """Minimal handle carrying the shared httpx client and the opencode key, so the
    batch handlers keep passing a single `client` object as they did with the SDK."""
    def __init__(self, http, key):
        self.http = http
        self.key = key


@asynccontextmanager
async def open_client():
    """Open a Go-API client handle. Use in any custom run loop that calls `_call`
    directly (e.g. classify): `async with engine.open_client() as client: ...`."""
    key = _api_key()
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as http:
        yield _GoClient(http, key)


# How spliced verbatim spans are joined into the answer. A plain space (not an ellipsis
# or any distinctive mark), so a multi-span answer reads as continuous period prose and
# the fine-tuned model never learns a splice marker as a stylistic tic. The routes'
# prompts already tell the model to pick fewest/longest spans that read in order, so the
# seam falls at a sentence boundary.
SPAN_JOIN = " "

# Shared rejection clause dropped into every route prompt EXCEPT stem_reasoning (whose
# whole purpose is to repair OCR-mangled formulas, so it must keep such passages). The
# corpus is scanned period text and some of it is OCR garbage; with excerpts plentiful,
# the model should skip anything corrupt rather than emit a garbled answer.
OCR_REJECT = """\
REJECT OCR-CORRUPTED TEXT. These passages are scanned from old books and some are
mangled by OCR — garbled or run-together words, dropped or transposed letters, stray
marks, broken spacing, nonsense strings. If such corruption touches the material your
answer would quote or rest on, so that a clean and correct answer is not possible, emit
NO item for that passage. Watch too for running headers, page numbers, chapter titles,
or marginal indices the scan has dropped into the MIDDLE of the text (e.g. "... breadstuffs
KY., SW. VA., ... N. GA. [CHAP. XL. and small stores ..."): such interpolated matter
must never enter your answer — keep it out, and if it cannot be cleanly kept out, emit
NO item. Passages are plentiful; prefer skipping a doubtful one to emitting a garbled
answer."""


def verbatim_answer(spans, excerpt, max_spans=2):
    """Return the answer built ONLY from exact substrings of `excerpt`, or None.

    Each model-proposed span is located in the excerpt (whitespace-normalized,
    case-insensitive) and the excerpt's own text for that range is used — so the
    result is verbatim by construction, not by trusting what the model typed. Any
    span not found verbatim, or more than max_spans, rejects the whole answer. Spans
    are joined by an ellipsis (a separator, never a word), so no model-written text
    enters the answer.
    """
    if not spans or len(spans) > max_spans:
        return None
    norm = re.sub(r"\s+", " ", excerpt)
    low = norm.lower()
    out = []
    for s in spans:
        if not isinstance(s, str):
            return None
        ns = re.sub(r"\s+", " ", s).strip()
        i = low.find(ns.lower())
        if i == -1:                                  # allow model-appended trailing punct
            ns = ns.rstrip('.,;:"\'')
            i = low.find(ns.lower())
            if i == -1:
                return None                          # not verbatim -> reject
        out.append(norm[i:i + len(ns)])              # store the EXCERPT's exact text
    ans = SPAN_JOIN.join(out).strip()
    return ans[:1].upper() + ans[1:] if ans else None  # capitalize first letter only


def spans_answer(max_spans):
    """answer_fn for extractive routes: verify the model's spans are verbatim."""
    def fn(r, excerpt):
        spans = r.get("spans")
        if spans is None and r.get("a"):             # tolerate single-string form
            spans = [r["a"]]
        return verbatim_answer(spans, excerpt, max_spans)
    return fn


def composed_answer(r, excerpt):
    """answer_fn for composed routes: take the model's answer text as-is."""
    return (r.get("a") or "").strip() or None


# STEM's OCR corrupts equations, exponents, subscripts, and symbols, so a strictly
# verbatim answer would either drag that corruption in or (if such spans are dropped)
# throw away the quantitative reasoning that is the whole point. Instead the model may
# REFURBISH a mangled formula back to standard form — but must add no new content. It
# returns each span twice: `verbatim` (the exact OCR text, our proof the span is real
# and located) and `refurbished` (the same span with only its formulas repaired). We
# verify `verbatim` is a true substring of the excerpt, then verify `refurbished` adds
# no LETTER the verbatim did not have — its letters must be a subsequence of the
# verbatim's. Only non-letters (digits, subscripts, operators, punctuation, spacing)
# may be introduced or changed, so a formula can be rebuilt but no prose can be
# invented. Digit-level correctness is the model's judgment under "repair, don't add".
def _letters(s):
    return re.sub(r"[^a-z]", "", s.lower())


def _is_subsequence(a, b):
    """True if every char of a appears in b in order (a is a subsequence of b)."""
    it = iter(b)
    return all(c in it for c in a)


def _span_pair(s):
    """Normalize a span into (verbatim, refurbished); tolerate a bare string form."""
    if isinstance(s, str):
        return s, s
    if isinstance(s, dict):
        v = s.get("verbatim") or s.get("v") or ""
        f = s.get("refurbished") or s.get("r") or v
        return v, f
    return "", ""


def refurbished_answer(spans, excerpt, max_spans=2):
    """Answer built from spans whose formulas the model may repair but whose prose it
    may not touch. Each span's `verbatim` must be an exact substring of the excerpt,
    and its `refurbished` form may add no letter absent from the verbatim (see above).
    Returns the joined refurbished text, or None if any span fails either check."""
    if not spans or len(spans) > max_spans:
        return None
    norm = re.sub(r"\s+", " ", excerpt)
    low = norm.lower()
    out = []
    for s in spans:
        v, f = _span_pair(s)
        if not isinstance(v, str) or not isinstance(f, str) or not v.strip():
            return None
        nv = re.sub(r"\s+", " ", v).strip()
        if nv.lower() not in low:                    # verbatim must be located verbatim
            nv = nv.rstrip('.,;:"\'')
            if not nv or nv.lower() not in low:
                return None
        f = re.sub(r"\s+", " ", f).strip() or nv
        if not _is_subsequence(_letters(f), _letters(nv)):   # no new letters => no new prose
            return None
        out.append(f)
    ans = SPAN_JOIN.join(out).strip()
    return ans[:1].upper() + ans[1:] if ans else None


def refurbished_spans_answer(max_spans):
    """answer_fn for the STEM route: verbatim-anchored spans with repaired formulas."""
    def fn(r, excerpt):
        spans = r.get("spans")
        if spans is None and r.get("a"):
            spans = [r["a"]]
        return refurbished_answer(spans, excerpt, max_spans)
    return fn


# --- route config --------------------------------------------------------------

@dataclass
class Route:
    name: str                     # basename for state/output files
    system: str                   # SYSTEM prompt (extraction, or the whole call if one-phase)
    source: Callable              # (n, seed) -> list[excerpt dict]
    answer_fn: Callable           # (result_dict, excerpt_str) -> answer str | None
    passthrough: tuple = ("prose_score",)   # excerpt fields to copy into rows
    question_system: str = None   # if set, generate the question in a SECOND call
                                  # (given passage + answer); else Q and A in one call
    model: str = MODEL
    max_tokens: int = MAX_TOKENS
    concurrency: int = CONCURRENCY
    token_budget: int = TOKEN_BUDGET
    extra_body: dict = None       # passed to messages.create (e.g. disable thinking)
    strict_json: bool = False      # require the model reply to be a bare JSON array


def state_file(route):
    return STATE_DIR / f"{route.name}.json"


def load_state(route):
    f = state_file(route)
    return json.loads(f.read_text()) if f.exists() else {}


def save_state(route, state):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_file(route).write_text(json.dumps(state))


# --- generation loop -----------------------------------------------------------

async def _call(client, semaphore, route, system, payload, n):
    """One model call to the opencode Go API (chat/completions) with retry; returns
    the parsed JSON list or None."""
    body = {"model": route.model, "max_tokens": route.max_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": payload}]}
    extra = getattr(route, "extra_body", None)       # tolerate configs without the field
    if extra:
        body.update(extra)                           # e.g. provider flags to disable thinking
    headers = {"Authorization": f"Bearer {client.key}"}
    async with semaphore:
        text = ""
        for attempt in range(5):
            try:
                resp = await client.http.post(f"{BASE_URL}/v1/chat/completions",
                                              json=body, headers=headers)
                if resp.status_code == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
                if resp.status_code == 402:
                    raise SystemExit(
                        "\nOut of API credit (402 Payment Required). Stopping now. Top up "
                        "and re-run — completed routes are already written and will be "
                        "skipped, so it resumes where it left off.")
                if resp.status_code in (401, 403):
                    raise SystemExit(f"\nAPI auth failed ({resp.status_code}). "
                                     f"Check the API key (env or .env).")
                resp.raise_for_status()
                choice = resp.json()["choices"][0]
                text = (choice["message"]["content"] or "").strip()
                if not text:
                    raise ValueError(f"empty response (finish={choice.get('finish_reason')})")
                if choice.get("finish_reason") == "length":
                    # deterministic: retrying the identical request truncates again, and
                    # it already burned a full max_tokens generation. Give up now — the
                    # caller (e.g. verify) can split the batch and retry smaller.
                    return None
                return _parse_json_array(text, strict=getattr(route, "strict_json", False))
            except json.JSONDecodeError as e:
                if attempt == 4:
                    print(f"  batch of {n} failed: unparseable: {e}\n"
                          f"    raw[:200]: {text[:200]!r}")
                    return None
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                if attempt == 4:
                    print(f"  batch of {n} failed after 5 attempts: "
                          f"{type(e).__name__}: {str(e)[:200]}")
                    return None
                await asyncio.sleep(2 ** attempt)
    return None


async def _single_batch(client, semaphore, batch, state, route):
    """One-phase: Q, A, and category in a single call (knowledge, stem)."""
    keys = [str(it["doc_index"]) for it in batch]
    payload = json.dumps([{"i": i, "text": it["excerpt"]} for i, it in enumerate(batch)])
    parsed = await _call(client, semaphore, route, route.system, payload, len(batch))
    for r in parsed if isinstance(parsed, list) else []:
        if not isinstance(r, dict):                  # model returned a bare value
            continue
        idx = r.get("i")
        if not (isinstance(idx, int) and 0 <= idx < len(keys)):
            continue
        q = (r.get("q") or "").strip()
        cat = (r.get("category") or "").strip()
        kind = (r.get("kind") or "").strip()         # optional sub-type tag (narrative)
        a = route.answer_fn(r, batch[idx]["excerpt"])
        if q and a:                                  # a is None -> drop this item
            state[keys[idx]] = {"q": q, "a": a, "category": cat, "kind": kind}


async def _extract_batch(client, semaphore, batch, state, route):
    """Two-phase call 1: extract the answer (spans) + category from the passage."""
    keys = [str(it["doc_index"]) for it in batch]
    payload = json.dumps([{"i": i, "text": it["excerpt"]} for i, it in enumerate(batch)])
    parsed = await _call(client, semaphore, route, route.system, payload, len(batch))
    for r in parsed if isinstance(parsed, list) else []:
        if not isinstance(r, dict):                  # model returned a bare value
            continue
        idx = r.get("i")
        if not (isinstance(idx, int) and 0 <= idx < len(keys)):
            continue
        cat = (r.get("category") or "").strip()
        a = route.answer_fn(r, batch[idx]["excerpt"])
        if a:
            state[keys[idx]] = {"a": a, "category": cat}   # question filled in phase 2


async def _question_batch(client, semaphore, batch, state, route):
    """Two-phase call 2: write the question given the passage + its extracted answer."""
    keys = [str(it["doc_index"]) for it in batch]
    payload = json.dumps([{"i": i, "passage": it["excerpt"], "answer": state[keys[i]]["a"]}
                          for i, it in enumerate(batch)])
    parsed = await _call(client, semaphore, route, route.question_system, payload, len(batch))
    for r in parsed if isinstance(parsed, list) else []:
        if not isinstance(r, dict):                  # model returned a bare value
            continue
        idx = r.get("i")
        if not (isinstance(idx, int) and 0 <= idx < len(keys)):
            continue
        q = (r.get("q") or "").strip()
        if q:
            state[keys[idx]]["q"] = q


async def _run_batches(batches, run_one, state, route, save, label,
                       excerpts=None, build_row=None):
    """Run batch coroutines concurrently, checkpointing state every CHECKPOINT_EVERY
    excerpts (so a long run — especially the two-phase extract, which otherwise saved
    nothing until the whole phase finished — is crash-safe and resumable) and pushing an
    HF shard every HF_PUSH_EVERY rows (so outputs commit incrementally, not only at the
    end). run_one(batch) is an awaitable that mutates `state`."""
    total = sum(len(b) for b in batches)
    done = [0]

    async def tracked(batch):
        await run_one(batch)
        done[0] += len(batch)
        if not save:
            return
        if done[0] % CHECKPOINT_EVERY < len(batch):
            save_state(route, state)
            print(f"  {label}: {done[0]}/{total}", flush=True)
        if build_row is not None and done[0] % HF_PUSH_EVERY < len(batch):
            if flush_shard(route.name, excerpts, state, build_row):
                save_state(route, state)             # persist the _pushed marks
        # (during a two-phase extract, build_row yields None for every entry — no q yet
        #  — so flush is a no-op; complete rows first appear in the question phase.)

    await asyncio.gather(*[asyncio.create_task(tracked(b)) for b in batches])
    if save:
        save_state(route, state)


async def run_async(route, excerpts, state, save=True):
    build = lambda e, r: _qa_row(route, e, r)        # for incremental HF shard flushes
    async with open_client() as client:
        semaphore = asyncio.Semaphore(route.concurrency)

        if route.question_system:
            # phase 1 — extract answers for excerpts that don't have one yet
            need_a = [e for e in excerpts if "a" not in state.get(str(e["doc_index"]), {})]
            if need_a:
                batches = _pack_batches(need_a, route.token_budget)
                print(f"  extract: {len(need_a)} excerpts, {len(batches)} batches...")
                await _run_batches(batches,
                                   lambda b: _extract_batch(client, semaphore, b, state, route),
                                   state, route, save, "extract", excerpts, build)
            # phase 2 — write questions for extracted answers that lack one
            need_q = [e for e in excerpts
                      if state.get(str(e["doc_index"]), {}).get("a")
                      and not state[str(e["doc_index"])].get("q")]
            if need_q:
                # payload carries passage + answer, so halve the budget
                batches = _pack_batches(need_q, max(1, route.token_budget // 2))
                print(f"  question: {len(need_q)} answers, {len(batches)} batches...")
                await _run_batches(batches,
                                   lambda b: _question_batch(client, semaphore, b, state, route),
                                   state, route, save, "question", excerpts, build)
            return

        # one-phase
        pending = [e for e in excerpts if str(e["doc_index"]) not in state]
        if not pending:
            print("  nothing pending")
            return
        batches = _pack_batches(pending, route.token_budget)
        print(f"  {len(pending)} excerpts, {len(batches)} batches...")
        await _run_batches(batches,
                           lambda b: _single_batch(client, semaphore, b, state, route),
                           state, route, save, route.name, excerpts, build)


def _qa_row(route, e, r):
    """Build one Q/A output row from a state entry, or None if the pair is incomplete.
    Excludes source-only fields (excerpt, category_moved) — the SFT set needs just the
    Q/A plus light provenance/metadata."""
    if not (r and r.get("q") and r.get("a")):        # both halves required
        return None
    row = {
        "doc_index": e["doc_index"],
        "category": r.get("category") or e["category"],
        "book_category": e["category"],
        "year": e.get("year"),
    }
    for f in route.passthrough:
        row[f] = e.get(f)
    row["question"] = r["q"]
    row["answer"] = r["a"]
    return row


def _write_local(route_name, rows):
    """Offline/debug sink: write all rows to OUTPUT_DIR/<route>.jsonl (SFT_OUTPUT_LOCAL)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{route_name}.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"  wrote {len(rows)} {route_name} rows -> {out}")
    return out


def flush_shard(route_name, excerpts, state, build_row):
    """Push complete-but-not-yet-pushed rows as a NEW HF shard and mark them pushed, so
    a long run commits incrementally and a resume never re-pushes. `build_row(e, r)`
    returns a row dict or None. No-op in local mode or when there is nothing new; returns
    the count pushed."""
    if os.environ.get("SFT_OUTPUT_LOCAL"):
        return 0
    new = []
    for e in excerpts:
        r = state.get(str(e["doc_index"]))
        if not r or r.get("_pushed"):
            continue
        row = build_row(e, r)
        if row is None:
            continue
        r["_pushed"] = True
        new.append(row)
    if not new:
        return 0
    from synth import hf_push                        # lazy: keeps engine import light
    dest = hf_push.push_shard(route_name, new)
    print(f"  pushed +{len(new)} {route_name} rows -> {dest}", flush=True)
    return len(new)


def write_output(route, excerpts, state):
    """Final flush: shard the remaining unpushed rows to HF (or write one local file)."""
    build = lambda e, r: _qa_row(route, e, r)
    if os.environ.get("SFT_OUTPUT_LOCAL"):
        rows = [row for e in excerpts
                if (row := build(e, state.get(str(e["doc_index"])))) is not None]
        return _write_local(route.name, rows)
    n = flush_shard(route.name, excerpts, state, build)
    save_state(route, state)                         # persist final _pushed marks
    print(f"  final: +{n} {route.name} rows; shards under {route.name}/ on HF")
    return n


def _wrap(label, text):
    """Wrap `text` to WRAP columns under a `label` prefix, hanging-indented."""
    return textwrap.fill(str(text), width=WRAP, initial_indent=label,
                         subsequent_indent=" " * len(label))


def _pair_lines(route, excerpts, state, excerpt_chars=400):
    """Format generated Q/A pairs beside their excerpts, wrapped to WRAP cols."""
    out = []
    for e in excerpts:
        r = state.get(str(e["doc_index"]))
        complete = r and r.get("q") and r.get("a")
        scores = "  ".join(f"{f} {e[f]:.2f}" for f in route.passthrough
                           if isinstance(e.get(f), (int, float)))
        if complete and r.get("category") and r["category"] != e["category"]:
            head = f"[{r['category']}]  (book said {e['category']})"
        else:
            head = f"[{e['category']}]"
        out.append("=" * WRAP)
        out.append(_wrap("", f"{head}  {e.get('year')}  {scores}"))
        out.append(_wrap("  excerpt : ", e["excerpt"][:excerpt_chars] + "..."))
        if complete:
            out.append(_wrap("  Q       : ", r["q"]))
            out.append(_wrap("  A       : ", r["a"]))
        else:
            out.append("  (failed)")
        out.append("")
    return out


async def test_run(route, excerpts):
    """One-off preview: generate a small sample, print it, save nothing."""
    state = {}
    await run_async(route, excerpts, state, save=False)
    print("\n".join(_pair_lines(route, excerpts, state)))


async def sample_run(route, excerpts, seed=0):
    """Prompt-testing record: generate and write a timestamped file to
    samples/<route>/ capturing the prompt(s) used + the pairs, so runs can be
    compared and old prompts recovered. Mirrors the authentic pipeline's `sample`.
    Writes only the file (lines wrapped to WRAP cols); nothing goes to stdout but
    the path confirmation."""
    state = {}
    await run_async(route, excerpts, state, save=False)
    kept = sum(1 for e in excerpts
               if (state.get(str(e["doc_index"])) or {}).get("q")
               and (state.get(str(e["doc_index"])) or {}).get("a"))
    header = [
        f"route: {route.name}", f"model: {route.model}", f"seed:  {seed}",
        f"n:     {len(excerpts)}", f"kept:  {kept}/{len(excerpts)}", "",
    ]
    if route.question_system:
        header += ["--- EXTRACT SYSTEM PROMPT ---", route.system, "",
                   "--- QUESTION SYSTEM PROMPT ---", route.question_system, ""]
    else:
        header += ["--- SYSTEM PROMPT ---", route.system, ""]
    body = _pair_lines(route, excerpts, state, excerpt_chars=400)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = ROOT / "synth" / "samples" / "routes" / route.name
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{ts}_seed{seed}_n{len(excerpts)}.txt"
    path.write_text("\n".join(header + body), encoding="utf-8")
    print(f"wrote sample ({kept}/{len(excerpts)} kept) -> {path}")
    return path
