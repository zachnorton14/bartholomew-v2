"""Sample the pretrain corpus (jbduran/think-dataset) to feel out excerpt sourcing.

We source from the *uncleaned* think-dataset rather than the -clean-1930s variant
because only the uncleaned repo ships per-document metadata (`audit_metadata.jsonl`)
aligned row-for-row with the parquet. The clean repo is text-only and re-sharded
through a shuffle, so its rows can't be joined back to a category. Anachronism
filtering is therefore ours to do downstream, at excerpt granularity.

Each parquet row is a whole document (a book; median ~80k words), single `text`
column. Metadata for document at (shard S, row r) is audit record
`offset(S) + r`, where offset(S) is the cumulative doc count of shards < S taken
from manifest.json. Key metadata fields:

  topic_or_subject_gen        LoC class, classifier-placed (the subject axis)
  topic_or_subject_score_gen  classifier confidence -- low = suspect label
  resolved_year, title_src, author_src, word_count

Reads remote parquet with DuckDB over HTTP range requests (no bulk download);
caches manifest.json + audit_metadata.jsonl under .cache/ (127 MB, one-time).

    python -m synth.corpus --docs 12 --excerpt-words 60,120,240

Requires: duckdb (pip install duckdb).
"""

import argparse
import json
import random
import re
import textwrap
import time
import urllib.request
from collections import Counter
from pathlib import Path

import duckdb

# Small function-word set: running prose is ~35-45% stopwords; indexes, tables,
# and title pages fall far below that, which is one of the region-quality signals.
_STOP = set(
    "the of and to a in is that it for as was with his he be not by this had she "
    "at they or an which we you from are his her their its but have has one all "
    "were their when there been who will more no if out so up would about into".split()
)
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=["\'(]?[A-Z0-9])')

ROOT = Path(__file__).parent.parent
CACHE = ROOT / ".cache"
# The 20 Library of Congress main classes used as the subject taxonomy (the
# `topic_or_subject_gen` values, minus "UNKNOWN"). Book-level in the metadata;
# used here as the fixed label set for content-level classification too.
LOC_CLASSES = [
    "LANGUAGE AND LITERATURE",
    "PHILOSOPHY. PSYCHOLOGY. RELIGION",
    "LAW",
    "SCIENCE",
    "HISTORY OF THE AMERICAS",
    "SOCIAL SCIENCES",
    "AUXILIARY SCIENCES OF HISTORY",
    "AGRICULTURE",
    "POLITICAL SCIENCE",
    "EDUCATION",
    "TECHNOLOGY",
    "GEOGRAPHY. ANTHROPOLOGY. RECREATION",
    "FINE ARTS",
    "MEDICINE",
    "MUSIC AND BOOKS ON MUSIC",
    "WORLD HISTORY AND HISTORY OF EUROPE, ASIA, AFRICA, AUSTRALIA, NEW ZEALAND, ETC.",
    "NAVAL SCIENCE",
    "GENERAL WORKS",
    "MILITARY SCIENCE",
    "BIBLIOGRAPHY. LIBRARY SCIENCE. INFORMATION RESOURCES (GENERAL)",
]

DATASET = "jbduran/think-dataset"
BASE = f"https://huggingface.co/datasets/{DATASET}/resolve/main"
SHARD_URL = BASE + "/shard_{:05d}.parquet"
N_SHARDS = 473

META_FIELDS = ("topic_or_subject_gen", "topic_or_subject_score_gen",
               "resolved_year", "title_src", "author_src", "word_count")


def _connect():
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET enable_progress_bar=false;")
    return con


def _download(url, dest):
    CACHE.mkdir(exist_ok=True)
    if dest.exists():
        return dest
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"  downloading {url.rsplit('/', 1)[-1]} -> {dest} ...", flush=True)
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def _offsets():
    """Return {shard_index: cumulative_doc_offset} and total doc count."""
    man = _download(BASE + "/manifest.json", CACHE / "manifest.json")
    shards = sorted(json.loads(man.read_text())["shards"], key=lambda s: s["index"])
    offs, cum = {}, 0
    for s in shards:
        offs[s["index"]] = cum
        cum += s["num_docs"]
    return offs, cum


def _audit():
    """Return a list of slim metadata dicts, one per document, in global order.

    Parses the 127 MB audit jsonl once, then caches a slim projection so later
    runs are fast. `shard` is carried so alignment can be self-checked.
    """
    slim = CACHE / "audit_slim.json"
    if slim.exists():
        return json.loads(slim.read_text())
    raw = _download(BASE + "/audit_metadata.jsonl", CACHE / "audit_metadata.jsonl")
    recs = []
    with open(raw, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rec = {k: d.get(k) for k in META_FIELDS}
            rec["shard"] = d.get("output_shard_index")
            recs.append(rec)
    slim.write_text(json.dumps(recs))
    return recs


def _load_index():
    """Return (offsets, audit, con) with an alignment self-check."""
    offs, total = _offsets()
    audit = _audit()
    if len(audit) != total:
        print(f"  WARN: audit records {len(audit)} != manifest total {total}")
    # spot-check that offset(S) really lands on the first record of shard S
    for s in (1, N_SHARDS // 2):
        base = offs.get(s)
        if base is not None and base < len(audit) and audit[base]["shard"] != s:
            print(f"  WARN: alignment off at shard {s}: "
                  f"audit[{base}].shard = {audit[base]['shard']}")
    return offs, audit, _connect()


def _shard_starts():
    """Return ([(global_offset, shard_index, num_docs), ...], total) sorted by offset."""
    man = _download(BASE + "/manifest.json", CACHE / "manifest.json")
    shards = sorted(json.loads(man.read_text())["shards"], key=lambda s: s["index"])
    starts, cum = [], 0
    for s in shards:
        starts.append((cum, s["index"], s["num_docs"]))
        cum += s["num_docs"]
    return starts, cum


def _locate(global_index, starts):
    """Map a global document index to (shard_index, local_row)."""
    import bisect
    offsets = [s[0] for s in starts]
    i = bisect.bisect_right(offsets, global_index) - 1
    off, shard, _ = starts[i]
    return shard, global_index - off


def _category_index(audit):
    """Return {category: [global_index, ...]}, cached."""
    cache = CACHE / "category_index.json"
    if cache.exists():
        return json.loads(cache.read_text())
    idx = {}
    for gi, m in enumerate(audit):
        idx.setdefault(m.get("topic_or_subject_gen") or "UNKNOWN", []).append(gi)
    cache.write_text(json.dumps(idx))
    return idx


def _fetch_text(con, shard, local_row):
    row = con.execute(
        f"SELECT text FROM read_parquet('{SHARD_URL.format(shard)}') LIMIT 1 OFFSET {local_row}"
    ).fetchone()
    return row[0] if row else None


def sample_by_category(category, n_text=3, seed=0):
    """Return up to n_text (text, meta) pairs whose documents are in `category`."""
    _, audit, con = _load_index()
    idx = _category_index(audit)
    starts, _ = _shard_starts()
    gis = idx.get(category)
    if not gis:
        raise ValueError(f"unknown category {category!r}; try --list-categories")
    rng = random.Random(seed)
    docs = []
    for gi in rng.sample(gis, min(n_text, len(gis))):
        shard, local = _locate(gi, starts)
        docs.append((_fetch_text(con, shard, local), audit[gi]))
    return docs


EXCERPTS_FILE = ROOT / "synth" / "output" / "excerpts.jsonl"
HARVEST_STATE = ROOT / "synth" / "state" / "harvest.json"
STEM_HARVEST_STATE = ROOT / "synth" / "state" / "harvest_stem.json"
VERSE_HARVEST_STATE = ROOT / "synth" / "state" / "harvest_verse.json"
CONVERSATIONAL_HARVEST_STATE = ROOT / "synth" / "state" / "harvest_conversational.json"
CALIBRATION_HARVEST_STATE = ROOT / "synth" / "state" / "harvest_calibration.json"


def load_excerpts(affordance=None, cls=None, min_prose=0.0, drop_broken_math=False):
    """Read the materialized excerpt corpus (one JSON object per line), optionally
    filtered to a classifier class / regex affordance / prose floor. This is what
    every generation route consumes — sourced once by harvest(), never re-fetched.

    `cls` (a class name or an iterable of them) selects by the model classifier's
    `classes` — the routes' real filter. An excerpt matches if any wanted class is in
    its `classes`, so unclassified excerpts (no `classes` yet) never match: run
    `classify` write-back before sourcing by class.

    `drop_broken_math` screens out OCR-destroyed arithmetic (see has_broken_math) — on
    for every prose route, OFF only for stem_reasoning (which repairs real math) and the
    internal dedup/report callers that must see the whole corpus."""
    if not EXCERPTS_FILE.exists():
        return []
    want = ({cls} if isinstance(cls, str) else set(cls)) if cls is not None else None
    out = []
    for line in EXCERPTS_FILE.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if affordance and r.get("affordance") != affordance:
            continue
        if want is not None and not want & set(r.get("classes") or []):
            continue
        if r.get("prose_score", 0) < min_prose:
            continue
        if drop_broken_math and has_broken_math(r["excerpt"]):
            continue
        out.append(r)
    return out


# --- knowledge partition (single-turn reserve + overlapping multiturn slice) ------
#
# The `knowledge` pool feeds TWO routes, and we want three things at once: a large
# pool of pure single-exchange rows (knowledge_qa), a large pool for the multiturn
# route to trim to a length distribution, and a deliberate OVERLAP where the same
# excerpt is passed BOTH ways ("same facts, different forms"). We get all three by
# ordering the pool once (a fixed seeded shuffle, so the slices are independent of
# doc_index/category order) and taking two overlapping slices of that one sequence:
#
#     |<----------- RESERVE (single) 0.00 .. 0.45 ----------->|
#                            |<------- MULTITURN 0.30 .. 1.00 ------------------>|
#     [ single only 0.30 ][ BOTH 0.15 ][ multiturn only 0.55                     ]
#
# Both routes MUST call knowledge_partition() with the same seed so the overlap lines
# up. Change the fractions here (not per route) to retune the split.
KNOWLEDGE_PARTITION_SEED = 20260729
KNOWLEDGE_RESERVE_END = 0.45      # single-turn reserve = first 45%
KNOWLEDGE_MULTI_START = 0.30      # multiturn slice = last 70% (overlaps reserve's tail)


def knowledge_partition(which, seed=KNOWLEDGE_PARTITION_SEED):
    """Return the deterministic knowledge slice for `which` ('reserve' or 'multiturn').

    The knowledge pool is loaded, shuffled once by a fixed seed, then sliced: 'reserve'
    is the leading KNOWLEDGE_RESERVE_END fraction (single-turn) and 'multiturn' is the
    trailing (1 - KNOWLEDGE_MULTI_START) fraction. The two overlap on
    [KNOWLEDGE_MULTI_START, KNOWLEDGE_RESERVE_END), which is intentionally double-passed.
    """
    mat = load_excerpts(cls=("knowledge",), drop_broken_math=True)
    random.Random(seed).shuffle(mat)                 # in-place on our fresh list
    n = len(mat)
    if which == "reserve":
        return mat[: int(n * KNOWLEDGE_RESERVE_END)]
    if which == "multiturn":
        return mat[int(n * KNOWLEDGE_MULTI_START):]
    raise ValueError(f"unknown knowledge partition: {which!r}")


def relabel_excerpts():
    """Re-tag the affordance of every excerpt in EXCERPTS_FILE in place, using the
    current affordance_label — for when the routing rules change (a new affordance,
    a tuned gate) without re-harvesting. Rewrites the file and reports the delta."""
    if not EXCERPTS_FILE.exists():
        print("no excerpts file")
        return
    recs = [json.loads(l) for l in EXCERPTS_FILE.read_text().splitlines() if l.strip()]
    changed = Counter()
    for r in recs:
        new = affordance_label(r["excerpt"])
        if new != r.get("affordance"):
            changed[f"{r.get('affordance')} -> {new}"] += 1
            r["affordance"] = new
    with EXCERPTS_FILE.open("w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"relabeled {sum(changed.values())}/{len(recs)} excerpts")
    for k, v in changed.most_common():
        print(f"  {k}: {v}")
    aff = Counter(r["affordance"] for r in recs)
    print("  by affordance: " + "  ".join(f"{a}={k}" for a, k in aff.most_common()))


def _read_shard(con, s, retries=5):
    """Read a shard's text column, retrying transient remote failures (HF network
    hiccups, ZSTD decompression errors on a truncated download) with backoff. Returns
    the rows, or None if it keeps failing — the caller skips the shard, leaving it
    unprocessed so a later run retries it. Keeps a long --exhaust sweep from dying on a
    single flaky read."""
    for attempt in range(retries):
        try:
            return con.execute(
                f"SELECT text FROM read_parquet('{SHARD_URL.format(s)}')"
            ).fetchall()
        except Exception as e:
            if attempt == retries - 1:
                print(f"  shard {s:>3}: read failed after {retries} tries "
                      f"({type(e).__name__}: {str(e)[:80]}); skipping, retry next run",
                      flush=True)
                return None
            time.sleep(2 ** attempt)
    return None


def _load_harvest_state():
    if HARVEST_STATE.exists():
        s = json.loads(HARVEST_STATE.read_text())
        return set(s.get("processed", [])), Counter(s.get("counts", {}))
    return set(), Counter()


def _save_harvest_state(processed, counts):
    HARVEST_STATE.parent.mkdir(parents=True, exist_ok=True)
    HARVEST_STATE.write_text(json.dumps(
        {"processed": sorted(processed), "counts": dict(counts)}))


# Excerpt length is sampled per-document from these bands, not fixed — a dataset whose
# answers all top out near one length teaches the model a length CEILING. Because the
# answer is a span, a long excerpt only *enables* a long answer; the route/question
# decides (a fact pulls a short span from any excerpt, "Recount ..." a long one). So a
# spread of excerpt lengths yields answers whose length matches the task.
_WORD_BANDS = ((0.40, 120, 180), (0.75, 300, 450), (1.00, 550, 800))


def _sample_words(rng):
    r = rng.random()
    for cum, lo, hi in _WORD_BANDS:
        if r < cum:
            return rng.randint(lo, hi)
    return rng.randint(550, 800)


def prose_windows(text, rng, k=3, n_words=None, floor=0.7, tries_per=10):
    """Up to k non-overlapping, quality-gated prose windows from ONE document, each a
    per-document sampled length. Lets the general harvest take several excerpts per doc
    (more, more-diverse data — different parts of the same book) instead of one, which
    matters because every downstream stage sheds rows. Same in-cut gate as
    prose_excerpt (self-contained, has-affordance, not-garbage, region floor). Returns
    [(excerpt, score, affordance), ...]."""
    sents = _split_sentences(strip_lines(text))
    n = len(sents)
    if not sents:
        return []
    wc = [len(s.split()) for s in sents]
    lo = min(n // 10, n - 1)
    used, out, attempts = [], [], 0
    while len(out) < k and attempts < k * tries_per:
        attempts += 1
        nw = _sample_words(rng) if n_words is None else n_words
        i = rng.randint(lo, n - 1)
        j, words = i, 0
        while j < n and words < nw:
            words += wc[j]
            j += 1
        if any(i < ue and j > us for us, ue in used):   # overlaps a kept window
            continue
        ex = " ".join(sents[i:j])
        if not is_self_contained(ex) or not has_affordance(ex) or is_garbage(ex):
            continue
        score = region_quality(ex)[0]
        if score < floor:
            continue
        used.append((i, j))
        out.append((ex, round(score, 3), affordance_label(ex)))
    return out


def harvest(total, alpha=0.5, min_conf=0.7, n_words=None, per_doc=3, seed=0,
            max_shards=N_SHARDS):
    """Shard-major sweep: read each shard's text column ONCE and cut gated
    excerpts to fill the tempered coverage quotas, tagging every affordance.

    Far cheaper than per-row OFFSET fetches (one sequential scan per shard vs. one
    range read per excerpt) and resumable: processed shards and per-category counts
    persist in the harvest state, excerpts append to EXCERPTS_FILE as JSONL.
    """
    offs, audit, con = _load_index()
    starts, _ = _shard_starts()
    idx = _category_index(audit)
    elig = {c: len(g) for c, g in _eligible_by_category(audit, idx, min_conf).items()}
    # per_doc windows per document raise each category's ceiling from its doc count.
    quotas = _tempered_targets({c: n * per_doc for c, n in elig.items()}, total, alpha)

    processed, got = _load_harvest_state()
    seen = {r["doc_index"] for r in load_excerpts()}  # dedup backstop
    EXCERPTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    order = list(range(N_SHARDS))
    rng.shuffle(order)

    def _done():
        return all(got[c] >= quotas.get(c, 0) for c in quotas)

    swept = 0
    fails = 0
    with open(EXCERPTS_FILE, "a", encoding="utf-8") as fh:
        for s in order:
            if _done() or swept >= max_shards:
                break
            if s in processed:
                continue
            base = offs.get(s)
            rows = _read_shard(con, s)
            if rows is None:
                fails += 1
                if fails >= 3:                    # not one bad shard — connection is down
                    print("  aborting: 3 shards failed in a row (connection likely "
                          "down). Nothing lost; rerun to resume.", flush=True)
                    break
                continue                          # isolated failure: retry next run
            fails = 0
            new = 0
            for r, (t,) in enumerate(rows):
                gi = base + r
                m = audit[gi]
                cat = m.get("topic_or_subject_gen") or "UNKNOWN"
                if cat not in quotas or got[cat] >= quotas[cat]:
                    continue
                if (m.get("topic_or_subject_score_gen") or 0) < min_conf:
                    continue
                for w, (ex, score, label) in enumerate(
                        prose_windows(t, rng, per_doc, n_words)):
                    if got[cat] >= quotas[cat]:
                        break
                    uid = f"{gi}-w{w}"
                    if uid in seen:
                        continue
                    fh.write(json.dumps({
                        "doc_index": uid, "doc": gi, "shard": s,
                        "category": cat, "affordance": label,
                        "confidence": m.get("topic_or_subject_score_gen"),
                        "year": m.get("resolved_year"), "title": m.get("title_src"),
                        "prose_score": round(score, 3), "n_words": len(ex.split()),
                        "excerpt": ex,
                    }, ensure_ascii=False) + "\n")
                    seen.add(uid)
                    got[cat] += 1
                    new += 1
            fh.flush()
            processed.add(s)
            swept += 1
            _save_harvest_state(processed, got)
            filled = sum(min(got[c], quotas[c]) for c in quotas)
            print(f"  shard {s:>3}: +{new:>3}  ({filled}/{sum(quotas.values())} quota, "
                  f"{swept} shards swept)", flush=True)

    total_have = sum(got.values())
    print(f"harvest: {total_have} excerpts across {len([c for c in got if got[c]])} "
          f"categories -> {EXCERPTS_FILE}")
    aff = Counter(r["affordance"] for r in load_excerpts())
    print("  by affordance: " + "  ".join(f"{a}={k}" for a, k in aff.most_common()))
    return total_have


def _load_stem_state():
    if STEM_HARVEST_STATE.exists():
        s = json.loads(STEM_HARVEST_STATE.read_text())
        return set(s.get("processed", [])), s.get("count", 0)
    return set(), 0


def _save_stem_state(processed, count):
    STEM_HARVEST_STATE.parent.mkdir(parents=True, exist_ok=True)
    STEM_HARVEST_STATE.write_text(json.dumps({"processed": sorted(processed), "count": count}))


def harvest_stem(target, min_conf=0.7, n_words=170, per_doc=4, min_signal=0.3,
                 seed=0, max_shards=N_SHARDS):
    """Targeted STEM overlay for the harvest: sweep the quantitative/physical
    categories shard by shard, cut up to per_doc reasoning-dense windows per doc via
    stem_windows, and append them to EXCERPTS_FILE tagged affordance="stem_reasoning".

    The affordance tag is only the heuristic guess; the classifier's `classes` is the
    truth downstream, so these windows go through the same classify -> route-by-class
    path as everything else. Separate from harvest() because the quota differs: this
    fills the STEM ROUTE `target` as a priority overlay, NOT the tempered LoC-category
    coverage. Shares the shard-major scan and resumes from its own state; both passes
    append to the one excerpts pool. Each window gets a unique `<gi>-s<w>` doc_index.
    """
    offs, audit, con = _load_index()
    stem_cats = set(STEM_CATEGORIES)
    EXCERPTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    processed, have = _load_stem_state()
    seen = {r["doc_index"] for r in load_excerpts()}       # dedup backstop
    rng = random.Random(seed)
    order = list(range(N_SHARDS))
    rng.shuffle(order)

    swept = 0
    fails = 0
    with open(EXCERPTS_FILE, "a", encoding="utf-8") as fh:
        for s in order:
            if have >= target or swept >= max_shards:
                break
            if s in processed:
                continue
            base = offs.get(s)
            rows = _read_shard(con, s)
            if rows is None:
                fails += 1
                if fails >= 3:                    # not one bad shard — connection is down
                    print("  aborting: 3 shards failed in a row (connection likely "
                          "down). Nothing lost; rerun to resume.", flush=True)
                    break
                continue                          # isolated failure: retry next run
            fails = 0
            new = 0
            for r, (t,) in enumerate(rows):
                if have >= target:
                    break
                gi = base + r
                m = audit[gi]
                if (m.get("topic_or_subject_gen") or "UNKNOWN") not in stem_cats:
                    continue
                if (m.get("topic_or_subject_score_gen") or 0) < min_conf:
                    continue
                for w, (ex, sig) in enumerate(
                        stem_windows(t, n_words, rng, per_doc, min_signal)):
                    uid = f"{gi}-s{w}"
                    if uid in seen:
                        continue
                    fh.write(json.dumps({
                        "doc_index": uid, "doc": gi, "shard": s,
                        "category": m.get("topic_or_subject_gen"),
                        "affordance": "stem_reasoning",
                        "confidence": m.get("topic_or_subject_score_gen"),
                        "year": m.get("resolved_year"), "title": m.get("title_src"),
                        "prose_score": sig, "stem_signal": sig,
                        "n_words": len(ex.split()), "excerpt": ex,
                    }, ensure_ascii=False) + "\n")
                    seen.add(uid)
                    have += 1
                    new += 1
                    if have >= target:
                        break
            fh.flush()
            processed.add(s)
            swept += 1
            _save_stem_state(processed, have)
            print(f"  shard {s:>3}: +{new:>3}  ({have}/{target} stem, "
                  f"{swept} shards swept)", flush=True)

    print(f"harvest_stem: {have} STEM windows -> {EXCERPTS_FILE}")
    return have


def _load_overlay_state(path):
    if path.exists():
        s = json.loads(path.read_text())
        return set(s.get("processed", [])), s.get("count", 0)
    return set(), 0


def _save_overlay_state(path, processed, count):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"processed": sorted(processed), "count": count}))


def _harvest_overlay(tag, key, window_fn, categories, target, state_path,
                     min_conf=0.7, seed=0, max_shards=N_SHARDS):
    """Generic targeted overlay (cf. harvest_stem): sweep `categories` shard by shard,
    cut windows via window_fn(text, rng) -> [(excerpt, score), ...], and append them to
    EXCERPTS_FILE tagged affordance=tag with unique `<gi>-<key><w>` keys. Fills the
    ROUTE `target` as a priority overlay (not LoC-category coverage); resumes from
    state_path. The tag is a heuristic guess — the classifier's `classes` is the truth
    downstream. Isolated read failures skip; three in a row abort (connection down)."""
    offs, audit, con = _load_index()
    cats = set(categories)
    EXCERPTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    processed, have = _load_overlay_state(state_path)
    seen = {r["doc_index"] for r in load_excerpts()}
    rng = random.Random(seed)
    order = list(range(N_SHARDS))
    rng.shuffle(order)

    swept = 0
    fails = 0
    with open(EXCERPTS_FILE, "a", encoding="utf-8") as fh:
        for s in order:
            if have >= target or swept >= max_shards:
                break
            if s in processed:
                continue
            base = offs.get(s)
            rows = _read_shard(con, s)
            if rows is None:
                fails += 1
                if fails >= 3:
                    print("  aborting: 3 shards failed in a row (connection likely "
                          "down). Nothing lost; rerun to resume.", flush=True)
                    break
                continue
            fails = 0
            new = 0
            for r, (t,) in enumerate(rows):
                if have >= target:
                    break
                gi = base + r
                m = audit[gi]
                if (m.get("topic_or_subject_gen") or "UNKNOWN") not in cats:
                    continue
                if (m.get("topic_or_subject_score_gen") or 0) < min_conf:
                    continue
                for w, (ex, sc) in enumerate(window_fn(t, rng)):
                    uid = f"{gi}-{key}{w}"
                    if uid in seen:
                        continue
                    fh.write(json.dumps({
                        "doc_index": uid, "doc": gi, "shard": s,
                        "category": m.get("topic_or_subject_gen"),
                        "affordance": tag,
                        "confidence": m.get("topic_or_subject_score_gen"),
                        "year": m.get("resolved_year"), "title": m.get("title_src"),
                        "prose_score": sc, "n_words": len(ex.split()), "excerpt": ex,
                    }, ensure_ascii=False) + "\n")
                    seen.add(uid)
                    have += 1
                    new += 1
                    if have >= target:
                        break
            fh.flush()
            processed.add(s)
            swept += 1
            _save_overlay_state(state_path, processed, have)
            print(f"  shard {s:>3}: +{new:>3}  ({have}/{target} {tag}, "
                  f"{swept} shards swept)", flush=True)

    print(f"harvest {tag}: {have} windows -> {EXCERPTS_FILE}")
    return have


def harvest_verse(target, seed=0, max_shards=N_SHARDS):
    """Targeted verse overlay — poetry/hymn/psalm books, line-based windows."""
    return _harvest_overlay("verse", "v", verse_windows, VERSE_CATEGORIES, target,
                            VERSE_HARVEST_STATE, seed=seed, max_shards=max_shards)


def harvest_conversational(target, per_doc=6, seed=0, max_shards=N_SHARDS):
    """Targeted conversational overlay — structured catechism/dialogue/Q&A windows,
    up to per_doc per doc (catechism books span many topics)."""
    return _harvest_overlay("conversational", "c",
                            lambda t, r: conversational_windows(t, r, k=per_doc),
                            CONVERSATIONAL_CATEGORIES, target,
                            CONVERSATIONAL_HARVEST_STATE, seed=seed, max_shards=max_shards)


def harvest_calibration(target, seed=0, min_conf=0.7, max_shards=N_SHARDS):
    """Targeted calibration overlay around factual non-answer language.

    It sweeps all well-labeled subject classes because refusals occur in history, law,
    science, medicine, and philosophy alike. Sentence-centered windows are appended to
    the shared excerpt pool and then must go through the normal classifier write-back.
    """
    return _harvest_overlay(
        "calibration", "ca", calibration_windows, LOC_CLASSES, target,
        CALIBRATION_HARVEST_STATE, min_conf=min_conf, seed=seed,
        max_shards=max_shards,
    )


def sample_documents(n, seed=0):
    """Return up to n (text, meta) pairs drawn from random shards.

    Takes the head rows of each chosen shard (cheap) and attaches the aligned
    metadata record; spreads across shards so the sample isn't one book cluster.
    """
    offs, audit, con = _load_index()
    rng = random.Random(seed)
    shards = rng.sample(range(N_SHARDS), min(N_SHARDS, max(1, n)))
    out, per = [], max(1, n // len(shards) + 1)
    for s in shards:
        rows = con.execute(
            f"SELECT text FROM read_parquet('{SHARD_URL.format(s)}') LIMIT {per}"
        ).fetchall()
        base = offs.get(s)
        for r, (t,) in enumerate(rows):
            meta = audit[base + r] if base is not None and base + r < len(audit) else {}
            out.append((t, meta))
            if len(out) >= n:
                return out
    return out


_BARE_NUM = re.compile(r"^[\d.,;:\-—\s]+$")
_ROMAN = re.compile(r"^[ivxlcdm]{1,8}$", re.I)


def strip_lines(text):
    """Remove structural/boilerplate lines OCR leaves embedded in body text:
    bare page numbers, all-caps running headers (optionally with a page number),
    and footnote/citation refs. Conservative — only strong structural signals,
    so running prose is left intact. Run before sentence-splitting."""
    out = []
    for raw in text.split("\n"):
        s = raw.strip()
        if not s:
            out.append(raw)
            continue
        letters = [c for c in s if c.isalpha()]
        words = s.split()
        drop = False
        if _BARE_NUM.match(s) or _ROMAN.match(s) or not letters:
            drop = True                                   # page number / all digits
        elif s[0] in "*†‡" and len(words) <= 12:
            drop = True                                   # footnote / citation ref
        elif len(words) <= 10 and len(s) <= 80 and \
                sum(c.isupper() for c in letters) / len(letters) > 0.7:
            drop = True                                   # all-caps running header
        if not drop:
            out.append(raw)
    return "\n".join(out)


def _split_sentences(text):
    """Split into sentences on .!? boundaries (whitespace normalized). OCR makes
    this imperfect, but the region-quality score rejects the spans where it fails
    worst (indexes, tables, front matter have few real sentence boundaries)."""
    return [s for s in _SENT_SPLIT.split(re.sub(r"\s+", " ", text).strip()) if s]


def region_quality(text):
    """Model-free 0-1 prose score. High = running prose; low = index / table /
    title page / OCR sludge. Returns (score, detail)."""
    chars = [c for c in text if not c.isspace()]
    words = re.findall(r"[A-Za-z]+", text)
    if not chars or not words:
        return 0.0, {}
    alpha = sum(c.isalpha() for c in chars) / len(chars)      # prose ~.85+
    digit = sum(c.isdigit() for c in chars) / len(chars)      # prose ~0; tables high
    stop = sum(w.lower() in _STOP for w in words) / len(words) # prose ~.35+
    sents = _split_sentences(text)
    msl = sum(len(s.split()) for s in sents) / len(sents) if sents else 0.0  # prose 15-30
    clamp = lambda x: max(0.0, min(1.0, x))
    score = (clamp((alpha - 0.60) / 0.30)     # penalize symbol/digit-heavy spans
             + clamp((0.05 - digit) / 0.05)   # penalize numeric density
             + clamp((stop - 0.20) / 0.15)    # reward function-word density
             + clamp((msl - 8) / 12)) / 4     # reward complete sentences
    return score, {"alpha": alpha, "digit": digit, "stop": stop, "msl": msl}


# Opening words that signal the excerpt continues from text it doesn't contain:
# demonstratives, third-person pronouns, and continuation connectives. Existential
# "it/there/here" are excluded — they open self-contained sentences too often.
_REF_OPENERS = {
    "this", "that", "these", "those", "he", "she", "they", "him", "her", "his",
    "their", "them", "such", "thus", "hence", "therefore", "however", "moreover",
    "nevertheless", "besides", "furthermore", "accordingly", "consequently",
    "meanwhile", "likewise", "which", "who", "whom", "whose",
}
_DOT_LEADER = re.compile(r"\.{2,}|_{2,}")             # form blanks / TOC dot leaders
_ORDINAL = re.compile(r"\b(1st|2nd|3rd|\d+th|firstly|secondly|thirdly)\b", re.I)
_OPINION = re.compile(r"\b(pleasant\w*|best|worst|admirable|agreeable|beautiful|"
                      r"ought|should|prefer\w*|delightful|charming|excellent|finest|noble)\b", re.I)
_REASON = re.compile(r"\b(because|therefore|thus|hence|since|consequently|inasmuch)\b", re.I)
_SPEECH = re.compile(r'\b(said|asked|replied|cried|answered|exclaimed|'
                     r'demanded|told|spoke|quoth)\b', re.I)
_QUOTE = re.compile(r'["“”‘’]')

# Formal composed-document markers. An excerpt that IS such a document (statute,
# legal pleading, letter, oration/resolution, devotion, instrument) is feedstock for
# the generative route: the answer is the verbatim artifact, the question asks to
# compose one. Two tiers so ordinary prose that merely says "whereas" or "the said
# man" isn't swept in — a STRONG phrase qualifies alone; WEAK markers need three hits.
_DOC_STRONG = re.compile(
    r"\b(be it enacted|be it ordained|by the authority of the same|"
    r"in this present parliament|know all men by these presents|in witness whereof|"
    r"this indenture|given under my hand|resolved,? that|be it resolved|"
    r"your obedient servant|yours (faithfully|truly|sincerely|obediently)|"
    r"i have the honour|we beseech thee|vouchsafe)\b", re.I)
_DOC_WEAK = re.compile(
    r"\b(whereas|provided,? that|an act (to|for)|enacting|plaintiff|defendant|"
    r"aforesaid|hereinbefore|hereinafter|the said |to wit|whereof|indictment|"
    r"complainant|dear sir|my dear \w+|i beg to|i am, sir|mr\.? president|"
    r"fellow[- ]citizens|gentlemen of the|o lord|almighty god|thy servant)\b", re.I)


def is_composition(text):
    """True when the excerpt reads as a formal composed document — statute, pleading,
    letter, oration, devotion, or legal instrument — rather than narration or
    exposition. A strong, unambiguous phrase qualifies alone; otherwise three weak
    genre markers must fire, so prose that merely says 'whereas' isn't caught."""
    if _DOC_STRONG.search(text):
        return True
    return len(_DOC_WEAK.findall(text)) >= 3


def narrative_signal(text):
    """0-1 score for story/dialogue prose. Reported speech is the strong cue (past
    tense alone appears in historical argument too, so it's weighted lightly). High
    score = a scene with speakers and events, which has no general reasoning to
    extract and reads as context-bare in a standalone question.

    Quotation marks count as dialogue ONLY alongside a speech verb: scare-quotes on
    single terms fill expository prose too ("the child is a 'visualizer'"), and were
    pulling exposition into the narrative pool."""
    w = text.split()
    if len(w) < 20:
        return 0.0
    speech = len(_SPEECH.findall(text))
    quotes = len(_QUOTE.findall(text)) if speech else 0
    past = len(re.findall(r"\b\w+ed\b", text)) / len(w)
    dlg = (speech + quotes) / len(w)
    return min(1.0, dlg * 12 + past * 2)


_VOWEL = re.compile(r"[aeiouy]", re.I)


def _foreign_count(text):
    """Count characters from scripts that shouldn't appear in pre-1930 English
    OCR — CJK/kana/Hangul/fullwidth, plus replacement and zero-width chars. Their
    presence signals OCR mode-confusion. Latin-1 accents and typographic symbols
    (é, æ, °, £, §) are intentionally NOT counted; Greek is left alone too (it
    appears legitimately in period scientific/classical texts)."""
    n = 0
    for ch in text:
        o = ord(ch)
        if o == 0xFFFD or o in (0x200B, 0x200C, 0x200D, 0xFEFF):
            n += 1
        elif (0x3000 <= o <= 0x30FF or 0x3400 <= o <= 0x9FFF
              or 0xAC00 <= o <= 0xD7A3 or 0xF900 <= o <= 0xFAFF
              or 0xFF00 <= o <= 0xFFEF):
            n += 1
    return n


def _gibberish_ratio(text):
    """Fraction of tokens that are OCR nonsense: digit-letter mashes ("97en") or
    length>=4 with no vowel (consonant runs)."""
    toks = re.findall(r"[A-Za-z0-9]+", text)
    if not toks:
        return 0.0
    bad = 0
    for t in toks:
        has_d = any(c.isdigit() for c in t)
        has_a = any(c.isalpha() for c in t)
        if (has_d and has_a) or (len(t) >= 4 and not _VOWEL.search(t)):
            bad += 1
    return bad / len(toks)


def is_garbage(text):
    """True for OCR-garbled spans that read as prose to the region score but are
    unusable: foreign-script contamination or heavy nonsense-token density."""
    return _foreign_count(text) >= 2 or _gibberish_ratio(text) > 0.10


_SUPERSUB = "²³¹⁰⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉"
_MATH_SYM = re.compile(r"[=×÷√°∴±≤≥≠∑∏∫]")


def ocr_score(text):
    """Fraction of whitespace tokens that look like OCR-mangled math or fragments:
    digit-letter mashes (P2, SD2), sub/superscript fragments (P₁, D₁2), bare math
    symbols, or long vowelless runs. High = a garbled span no route can extract
    cleanly. Deliberately conservative — flags the worst offenders (mangled equations),
    not ordinary prose that merely mentions a number."""
    toks = text.split()
    if len(toks) < 20:
        return 0.0
    bad = 0
    for raw in toks:
        t = raw.strip(".,;:()[]\"'")
        if not t:
            continue
        if _MATH_SYM.search(t):
            bad += 1
            continue
        has_d = any(c.isdigit() or c in _SUPERSUB for c in t)
        core = re.sub(r"[^A-Za-z]", "", t)
        if has_d and any(c.isalpha() for c in t):
            bad += 1                                   # P2, SD2, D₁2, 45cos
        elif len(core) >= 4 and not re.search(r"[aeiouy]", core, re.I):
            bad += 1                                   # consonant run
    return round(bad / len(toks), 3)


# Doubled or dangling math operators ("= =", "÷ ·", "= ;", "· =") are OCR DESTRUCTION,
# not artifacts any route can repair — clean equations never contain them. In scanned
# arithmetic drills the fill-in blanks (where a pupil's answer goes) drop out and the
# two equals signs collide, or a line's result bleeds into the next, leaving a chain no
# one can reconstruct. High precision: clean math and ordinary prose never match.
_BROKEN_MATH = re.compile(r"=\s*=|[=×÷]\s*[·;]|[·;]\s*=")


def has_broken_math(text):
    """True if the span carries OCR-destroyed arithmetic (doubled/dangling operators,
    dropped fill-in blanks) that cannot be honestly repaired — drop it at source."""
    return bool(_BROKEN_MATH.search(text))


# Period expressions of epistemic uncertainty.  Calibration needs a stricter signal
# than a generic hedge: "an unknown quantity" or "uncertain light" is not a refusal,
# and "inclined to believe" belongs to the opinion route.  Keep this list focused on
# claims that the source cannot settle a factual point.
_CALIBRATION_HEDGE_PHRASES = [
    r"not (?:certainly |precisely |exactly |yet )?known",
    r"(?:is|are|remains?|was|were) (?:still |yet |as yet )?(?:unknown|obscure|unsettled|undetermined|unascertained)",
    r"cannot (?:be |now be )?(?:known|determined|ascertained|decided|explained|said)",
    r"(?:authorities|opinions|writers|physicians|naturalists|historians|philosophers|critics) (?:differ|are divided|are at variance)",
    r"(?:a )?matter of (?:conjecture|dispute|doubt|uncertainty)",
    r"open to (?:doubt|dispute|question)",
    r"little is (?:certainly )?known",
    r"(?:we|I) (?:are (?:wholly |as yet )?ignorant|do not know|cannot (?:say|tell|determine))",
    r"no one (?:knows|can tell)",
    r"impossible to (?:say|determine|ascertain|decide|know)",
    r"(?:difficult|not easy) to (?:say|determine|decide)",
    r"remains? (?:a )?(?:mystery|secret)",
    r"has not (?:yet |as yet )?been (?:determined|ascertained|explained|discovered|settled)",
    r"(?:the )?(?:date|year|name|record|account) (?:is|was) lost",
    r"(?:no|not any) (?:record|account|evidence) (?:exists|remains|survives|can be found)",
    r"(?:no record|no account) (?:has been|is|was) (?:found|preserved|left|kept)",
    r"(?:the )?(?:record|account|evidence) (?:is|was) (?:wanting|insufficient)",
    r"(?:we|I) have no means of (?:knowing|determining|ascertaining|deciding)",
    r"(?:not|never) recorded",
    r"cannot safely be affirmed",
    r"(?:idle|vain) to (?:conjecture|speculate)",
    r"(?:we|I) (?:refrain|forbear) from (?:conjecture|speculation)",
    r"beyond (?:our|human) (?:knowledge|ken)",
    r"(?:is|are|remains?|was|were) (?:still |yet |as yet )?doubtful",
    r"in (?:doubt|dispute)",
]
_CALIBRATION_HEDGE = re.compile(
    r"\b(?:" + "|".join(_CALIBRATION_HEDGE_PHRASES) + r")\b", re.I
)
_NON_EPISTEMIC_UNKNOWN = re.compile(
    r"\b(?:unknown|uncertain)\s+(?:quantity|quantities|resistance|factor|contingenc(?:y|ies)|light|grave|friend|to\s+our\s+forefathers)\b|"
    r"\b(?:residence|holder|party|parties|person|object|thing|position|place)[^.]{0,90}\bis unknown\b",
    re.I,
)


def has_calibration_hedge(text):
    """True when text contains a factual non-answer suitable for calibration."""
    if not _CALIBRATION_HEDGE.search(text):
        return False
    return not _NON_EPISTEMIC_UNKNOWN.search(text)


def has_hedge(text):
    """Backward-compatible alias for the calibration source predicate."""
    return has_calibration_hedge(text)


# High-recall anchors for the calibration harvest. The answer-level predicate above
# remains the precision gate; these phrases decide which sentence neighborhoods are
# worth materializing in the first place.
_CALIBRATION_HARVEST_ANCHOR = re.compile(
    r"\b(?:not known|unknown|cannot be known|cannot be determined|cannot be ascertained|"
    r"cannot say|no one knows|little is known|remains obscure|not settled|unsettled|"
    r"in dispute|matter of (?:conjecture|dispute|doubt)|authorities (?:differ|are divided)|"
    r"opinions (?:differ|are divided)|no record|no account|record is lost|account is lost|"
    r"date is lost|year is lost|name is lost|not recorded|never recorded|no means of|"
    r"impossible to (?:say|determine|ascertain)|difficult to (?:say|determine)|"
    r"not easy to (?:say|determine)|idle to (?:conjecture|speculate)|"
    r"refrain from (?:conjecture|speculation)|cannot safely be affirmed|beyond .*?knowledge)\b",
    re.I,
)


def calibration_windows(text, rng=None, k=8, n_words=260):
    """Return sentence-centered neighborhoods around factual non-answer anchors.

    Unlike general random prose windows, these keep the hedge sentence plus nearby
    context so the downstream question writer can identify the unresolved subject.
    The classifier still decides the final route class; this is a recall overlay only.
    """
    sents = _split_sentences(strip_lines(text))
    if not sents:
        return []
    lo = min(len(sents) // 10, len(sents) - 1)
    anchors = [
        i for i, s in enumerate(sents)
        if i >= lo and _CALIBRATION_HARVEST_ANCHOR.search(s)
        and has_calibration_hedge(s)
    ]
    if rng:
        rng.shuffle(anchors)
    out, used = [], []
    for i in anchors:
        if len(out) >= k:
            break
        start = max(lo, i - 2)
        end = min(len(sents), i + 3)
        # Pull back one more sentence when the context begins with a dangling reference.
        while start > lo and not is_self_contained(sents[start]):
            start -= 1
        if any(start < ue and end > us for us, ue in used):
            continue
        ex = " ".join(sents[start:end])
        if len(ex.split()) > n_words:
            # The anchor and one neighboring sentence are the useful minimum context.
            start = max(lo, i - 1)
            end = min(len(sents), i + 2)
            ex = " ".join(sents[start:end])
        if (not is_self_contained(ex) or not has_affordance(ex) or is_garbage(ex)
                or region_quality(ex)[0] < 0.70
                or not has_calibration_hedge(ex)):
            continue
        used.append((start, end))
        out.append((ex, round(region_quality(ex)[0], 3)))
    return out


def _first_word(text):
    m = re.match(r"\s*([A-Za-z]+)", text)
    return m.group(1).lower() if m else ""


def is_self_contained(text):
    """False if the excerpt opens on an unresolved reference (dangling pronoun,
    demonstrative, or continuation connective) — no question can be answered
    without the missing antecedent."""
    return _first_word(text) not in _REF_OPENERS


def has_affordance(text):
    """False for structurally dead spans — fill-in forms and dot-leader tables —
    that no question can be answered from."""
    return len(_DOT_LEADER.findall(text)) < 2


def affordance_label(text):
    """Coarse routing tag for a surviving excerpt (low-stakes: mislabels only
    misroute, they don't drop data)."""
    # Checked first: a formal document can carry past tense, quotes, and opinion
    # words that would otherwise misroute it to narrative/argument/opinion.
    if is_composition(text):
        return "composition"
    fp = len(re.findall(r"\b(i|we|my|our|me|us)\b", text, re.I))
    past = len(re.findall(r"\b\w+ed\b", text))
    if _ORDINAL.search(text) and past < 6:
        return "procedural"
    if fp >= 2 and _OPINION.search(text):
        return "opinion"
    # dialogue-heavy scenes are narrative, not argument — checked BEFORE `argument`
    # because narration is full of causal "because/since/for" that _REASON matches.
    if narrative_signal(text) >= 0.3:
        return "narrative"
    if _REASON.search(text):
        return "argument"
    if past >= 8 and re.search(r"\b1[5-9]\d\d\b", text):
        return "narrative"
    return "expository"


# STEM reasoning lives in windows that combine reasoning connectives with
# quantitative/physical vocabulary. Both are required (their product scores it):
# vocabulary alone is a description, connectives alone are rhetoric.
_REASON_CONN = re.compile(
    r"\b(therefore|hence|thus|since|because|consequently|whence|accordingly|"
    r"it follows|for this reason|so that|if|then|let|suppose|assume|given)\b", re.I)
_STEM_VOCAB = re.compile(
    r"\b(equals?|angle|triangle|rectangle|square|squared|circle|ratio|proportion|"
    r"multiply|multiplied|divide|divided|product|quotient|sum|difference|subtract|"
    r"factor|exponent|logarithm|sine|cosine|tangent|root|radius|diameter|"
    r"circumference|perpendicular|parallel|hypotenuse|axis|"
    r"velocity|force|pressure|weight|mass|volume|density|temperature|heat|energy|"
    r"momentum|gravity|gravitation|acceleration|friction|lever|fulcrum|inertia|"
    r"coefficient|current|voltage|resistance|ampere|volt|ohm|calorie|"
    r"acid|alkali|oxygen|hydrogen|carbon|nitrogen|compound|element|reaction|"
    r"solution|atom|molecule|combustion|oxide|"
    r"equation|formula|theorem|proposition|proof|quantity|degrees?|per\s*cent|"
    r"foot|feet|inch(?:es)?|pound|ounce|gallon|cubic|"
    r"calculate|computed?|measure[ds]?)\b"
    r"|[=×÷√°]", re.I)


def stem_signal(text):
    """0-1 score for STEM-reasoning density: reasoning structure AND quantitative
    vocabulary, as a product so both must be present."""
    if len(text.split()) < 20:
        return 0.0
    conn = min(len(_REASON_CONN.findall(text)), 5) / 5
    vocab = min(len(_STEM_VOCAB.findall(text)), 8) / 8
    return conn * vocab


def prose_excerpt(text, n_words=150, rng=None, tries=16, floor=0.7, prefer="prose"):
    """Cut a sentence-bounded excerpt of ~n_words that clears the prose bar.

    Builds candidates from runs of whole sentences (so no cut lands mid-sentence),
    scores each with region_quality, and returns the first above `floor` — or the
    best seen. Skips the first 10% of sentences (front matter). Returns (text, score)."""
    rng = rng or random
    sents = _split_sentences(strip_lines(text))
    if not sents:
        return "", 0.0, ""
    wc = [len(s.split()) for s in sents]
    n = len(sents)
    lo = min(n // 10, n - 1)

    def window(i):
        j, words = i, 0
        while j < n and words < n_words:
            words += wc[j]
            j += 1
        return " ".join(sents[i:j])

    if prefer == "stem":
        # STEM reasoning is sparse (~few % of sentences), so random windows miss it.
        # Anchor the search on sentences carrying a reasoning connective, window
        # around each, and keep the gated window with the highest stem signal.
        anchors = [i for i in range(lo, n) if _REASON_CONN.search(sents[i])]
        best = ("", -1.0)
        for i in anchors[:200]:
            ex = window(max(lo, i - 1))
            if not is_self_contained(ex) or not has_affordance(ex) or is_garbage(ex):
                continue
            if region_quality(ex)[0] < 0.5:       # looser floor: math lowers region
                continue
            sig = stem_signal(ex)
            if sig > best[1]:
                best = (ex, sig)
        ex, sig = best
        if not ex or sig <= 0:
            return "", 0.0, ""
        return ex, round(sig, 3), "stem_reasoning"

    best = ("", -1.0)
    for _ in range(tries):
        i = rng.randint(lo, n - 1)
        ex = window(i)
        if not is_self_contained(ex) or not has_affordance(ex) or is_garbage(ex):
            continue                              # hard-drop: resample another window
        score = region_quality(ex)[0]
        if score > best[1]:
            best = (ex, score)
        if score >= floor:
            break
    ex, score = best
    if not ex:
        return "", 0.0, ""                        # no window survived the gate
    return ex, score, affordance_label(ex)


def stem_windows(text, n_words=170, rng=None, k=4, min_signal=0.3):
    """Up to k non-overlapping STEM-reasoning windows from ONE document, each above
    min_signal, highest-signal first. STEM docs are window-rich, so taking several
    per doc multiplies yield over prose_excerpt's single-best pick — a RECALL net that
    the classifier confirms downstream. Returns [(excerpt, signal), ...]."""
    rng = rng or random
    sents = _split_sentences(strip_lines(text))
    if not sents:
        return []
    wc = [len(s.split()) for s in sents]
    n = len(sents)
    lo = min(n // 10, n - 1)

    def window(i):
        j, words = i, 0
        while j < n and words < n_words:
            words += wc[j]
            j += 1
        return " ".join(sents[i:j]), j            # text, end sentence index

    cands = []
    for i in [i for i in range(lo, n) if _REASON_CONN.search(sents[i])][:300]:
        start = max(lo, i - 1)
        ex, end = window(start)
        if not is_self_contained(ex) or not has_affordance(ex) or is_garbage(ex):
            continue
        if region_quality(ex)[0] < 0.5:           # looser floor: math lowers region
            continue
        sig = stem_signal(ex)
        if sig >= min_signal:
            cands.append((sig, start, end, ex))
    cands.sort(key=lambda c: c[0], reverse=True)  # highest signal first
    out, used = [], []
    for sig, start, end, ex in cands:
        if any(start < ue and end > us for us, ue in used):   # overlaps a kept window
            continue
        used.append((start, end))
        out.append((ex, round(sig, 3)))
        if len(out) >= k:
            break
    return out


# Verse lives in poetry/hymn/psalm books; it is line-structured and fails the prose
# gates the general harvest uses, so it needs its own line-based window search.
VERSE_CATEGORIES = [
    "LANGUAGE AND LITERATURE", "MUSIC AND BOOKS ON MUSIC",
    "PHILOSOPHY. PSYCHOLOGY. RELIGION",
]


def verse_windows(text, rng=None, k=6, min_score=0.6, win=12):
    """Windows of consecutive verse-like lines — short and capital-initial. Blank lines
    are dropped FIRST (verse is often double-spaced, which would otherwise look "mostly
    blank"), then a window of `win` consecutive non-blank lines is kept when most are
    verse-like. Line breaks are preserved in the excerpt. A RECALL net; the classifier
    confirms downstream. Returns [(excerpt, score), ...]."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    n = len(lines)
    if n < win:
        return []
    lo = n // 10                                   # skip front matter

    def verselike(s):
        return len(s) <= 55 and s[:1].isupper()

    flags = [verselike(l) for l in lines]
    cands = []
    for i in range(lo, n - win + 1, 3):
        score = sum(flags[i:i + win]) / win
        if score < min_score:
            continue
        ex = "\n".join(lines[i:i + win])
        if len(ex.split()) < 20 or is_garbage(ex):
            continue
        cands.append((score, i, i + win, ex))
    cands.sort(key=lambda c: c[0], reverse=True)
    out, used = [], []
    for sc, a, b, ex in cands:
        if any(a < ue and b > us for us, ue in used):
            continue
        used.append((a, b))
        out.append((ex, round(sc, 3)))
        if len(out) >= k:
            break
    return out


# Conversational = dialogue / catechism / Q&A. Prose-like (has sentences), so it reuses
# the sentence-window search, anchored on speech turns and questions.
# Catechisms / didactic Q&A live in education and religion texts. LANGUAGE AND
# LITERATURE is dropped on purpose: its "dialogue" is novels and plays, where speech is
# wrapped in narration/attribution ('"...," said X') and can't be split into clean
# verbatim turns, and plays carry >2 speakers.
# Structured Q&A lives well beyond catechism books — legal depositions (LAW), science
# catechisms (SCIENCE), Socratic dialogue and religious catechism (PHILOSOPHY/RELIGION),
# didactic textbooks (EDUCATION, AGRICULTURE), debate/testimony (SOCIAL SCIENCES).
# LANGUAGE AND LITERATURE is excluded on purpose — its "dialogue" is novels and plays.
CONVERSATIONAL_CATEGORIES = [
    "PHILOSOPHY. PSYCHOLOGY. RELIGION", "EDUCATION", "LAW", "SCIENCE",
    "SOCIAL SCIENCES", "AGRICULTURE",
]
_QA_MARK = re.compile(r"(?:^|\n)\s*(?:Q\.|A\.|Ques\b|Ans\b|Question\b|Answer\b)", re.I)

# Explicit two-party structure — Q./A. markers, or a short speaker-name label used at
# least twice (real turn-taking) — the mark of a catechism/Socratic dialogue/deposition/
# debate, as opposed to prose fiction with quoted speech. Shared by the harvest window
# gate and the route's sourcing filter so both look for the same thing.
_STRUCT_QA = re.compile(r"(?:^|\n|\.\s)\s*(?:Q\.|A\.|Ques\b|Ans\b)", re.I)
_SPEAKER_LABEL = re.compile(r'(?:^|(?<=[.!?”"\'])\s)([A-Z][a-z]{0,9}\.)(?=\s+[A-Z“"\'])')


def is_structured_dialogue(text):
    repeated = sum(v for v in Counter(_SPEAKER_LABEL.findall(text)).values() if v >= 2)
    return len(_STRUCT_QA.findall(text)) >= 2 or repeated >= 3


def conversational_signal(text):
    """0-1 score for CATECHISM / didactic Q&A: dense in questions and Q./A. markers and
    LOW in narrative attribution. Reported-speech verbs (said/replied/asked) mark
    fiction dialogue — which does NOT split into clean verbatim turns — so they PENALIZE
    the score rather than raise it. Fiction and plays score near zero."""
    w = len(text.split())
    if w < 20:
        return 0.0
    qmarks = text.count("?")
    qa = len(_QA_MARK.findall(text))
    attribution = len(_SPEECH.findall(text))       # said/replied/asked -> fiction
    struct = (qmarks + qa * 4) / w                  # question + Q&A-marker density
    return max(0.0, min(1.0, struct * 9 - (attribution / w) * 12))


def conversational_windows(text, rng=None, k=3, min_signal=0.35, n_words=220):
    """Windows dense in CATECHISM / didactic Q&A, sentence-bounded — anchors on
    questions and Q./A. markers (NOT speech verbs, which mark fiction narration), scores
    with conversational_signal, keeps the densest non-overlapping. A longer window than
    the recall routes, to capture several Q/A turns. Returns [(excerpt, score), ...]."""
    rng = rng or random
    sents = _split_sentences(strip_lines(text))
    if not sents:
        return []
    wc = [len(s.split()) for s in sents]
    n = len(sents)
    lo = min(n // 10, n - 1)

    def window(i):
        j, words = i, 0
        while j < n and words < n_words:
            words += wc[j]
            j += 1
        return " ".join(sents[i:j]), j

    anchors = [i for i in range(lo, n)
               if "?" in sents[i] or _QA_MARK.search(sents[i])]
    cands = []
    for i in anchors[:300]:
        start = max(lo, i - 1)
        ex, end = window(start)
        if not is_self_contained(ex) or not has_affordance(ex) or is_garbage(ex):
            continue
        sig = conversational_signal(ex)
        if sig >= min_signal and is_structured_dialogue(ex):
            cands.append((sig, start, end, ex))
    cands.sort(key=lambda c: c[0], reverse=True)
    out, used = [], []
    for sig, a, b, ex in cands:
        if any(a < ue and b > us for us, ue in used):
            continue
        used.append((a, b))
        out.append((ex, round(sig, 3)))
        if len(out) >= k:
            break
    return out


def _stats(values):
    v = sorted(values)
    def pct(q):
        return v[int(q * (len(v) - 1))]
    return min(v), pct(0.5), pct(0.9), max(v)


def report(n=10, seed=0, excerpt_words=(60, 120, 240), show=3):
    """Sample n documents, print length + category distribution, and preview
    excerpt windows for the first `show` documents."""
    from collections import Counter
    docs = sample_documents(n, seed=seed)
    rng = random.Random(seed)

    cc = [len(t) for t, _ in docs]
    ww = [len(t.split()) for t, _ in docs]
    print(f"\nsampled {len(docs)} documents from {DATASET}")
    print(f"  chars  min {_stats(cc)[0]:>8,}  p50 {_stats(cc)[1]:>8,}  "
          f"p90 {_stats(cc)[2]:>9,}  max {_stats(cc)[3]:>9,}")
    print(f"  words  min {_stats(ww)[0]:>8,}  p50 {_stats(ww)[1]:>8,}  "
          f"p90 {_stats(ww)[2]:>9,}  max {_stats(ww)[3]:>9,}")

    tally = Counter(m.get("topic_or_subject_gen", "?") for _, m in docs)
    print("  categories in sample:")
    for cat, k in tally.most_common():
        print(f"    {k:>3}  {cat}")

    for t, m in docs[:show]:
        print("\n" + "=" * 78)
        cat = m.get("topic_or_subject_gen", "?")
        score = m.get("topic_or_subject_score_gen")
        score_s = f"{score:.2f}" if isinstance(score, (int, float)) else "?"
        print(f"[{cat}]  conf {score_s}  {m.get('resolved_year','?')}  "
              f"{len(t.split()):,} words")
        print(f"  title: {str(m.get('title_src',''))[:90]}")
        for nw in excerpt_words:
            ex, score, label = prose_excerpt(t, nw, rng)
            if not ex:
                print(f"\n--- excerpt ~{nw} words: (dropped — no self-contained window) ---")
                continue
            print(f"\n--- excerpt ~{nw} words ({len(ex.split())} actual, {label}, prose {score:.2f}) ---")
            print(textwrap.fill(ex, width=78))


def _tempered_targets(eligible, total, alpha):
    """target_c ∝ eligible_c ** alpha, normalized to `total`, capped at eligible_c."""
    W = sum(n ** alpha for n in eligible.values() if n > 0)
    return {c: (min(round(total * n ** alpha / W), n) if n > 0 else 0)
            for c, n in eligible.items()}


def _eligible_by_category(audit, idx, min_conf):
    """{category: [global_index, ...]} of docs whose label confidence >= min_conf."""
    return {
        c: [g for g in gis if (audit[g].get("topic_or_subject_score_gen") or 0) >= min_conf]
        for c, gis in idx.items() if c != "UNKNOWN"
    }


def plan_targets(total, alpha=0.5, min_conf=0.5):
    """Compute per-category excerpt quotas via a tempered distribution.

    alpha=1 mirrors the corpus skew, alpha=0 is uniform, alpha≈0.5 flattens skew
    while keeping big fields larger. Returns (targets, eligible_counts,
    corpus_counts, audit, index).
    """
    _, audit, _ = _load_index()
    idx = _category_index(audit)
    corpus = {c: len(g) for c, g in idx.items() if c != "UNKNOWN"}
    elig = {c: len(g) for c, g in _eligible_by_category(audit, idx, min_conf).items()}
    return _tempered_targets(elig, total, alpha), elig, corpus, audit, idx


def report_coverage(total, alpha=0.5, min_conf=0.5):
    """Print corpus share vs. target share so the rebalance is visible before
    any excerpt is fetched."""
    targets, eligible, corpus, _, _ = plan_targets(total, alpha, min_conf)
    corpus_total = sum(corpus.values())
    plan_total = sum(targets.values())
    print(f"coverage plan: total≈{plan_total} (asked {total}), alpha={alpha}, "
          f"min_conf={min_conf}")
    print(f"  {'category':44} {'corpus%':>8} {'target%':>8} {'target':>7} {'eligible':>9}")
    for c in sorted(targets, key=lambda c: -targets[c]):
        cap = "  (capped)" if targets[c] == eligible[c] and eligible[c] else ""
        print(f"  {c:44} {100*corpus[c]/corpus_total:7.1f}% "
              f"{100*targets[c]/max(plan_total,1):7.1f}% {targets[c]:7} {eligible[c]:9,}{cap}")


# Quantitative/physical categories where STEM reasoning lives. Wider than just the
# two pure-science classes: surveying/chronology (auxiliary sciences), navigation and
# ballistics (naval/military), soil chemistry (agriculture), physiology (medicine),
# astronomy/cartography (geography). Logic stays with the reasoning route, not here.
STEM_CATEGORIES = [
    "SCIENCE", "TECHNOLOGY", "AGRICULTURE", "MEDICINE",
    "AUXILIARY SCIENCES OF HISTORY", "GEOGRAPHY. ANTHROPOLOGY. RECREATION",
    "NAVAL SCIENCE", "MILITARY SCIENCE",
]


def sample_stem(n=20, seed=0, min_signal=0.3, min_conf=0.7, n_words=170,
                categories=STEM_CATEGORIES, per_doc=4):
    """Draw ~n STEM-reasoning excerpts from the quantitative/physical categories,
    taking up to `per_doc` non-overlapping reasoning-dense windows from each doc
    (STEM docs are window-rich) and keeping those above `min_signal`.

    A RECALL net: the classifier is the precision layer, so this over-sources
    candidate windows (wide categories, several per doc, a low signal floor) and lets
    the model confirm which are truly STEM. Each window gets a unique doc_index
    (`<gi>-<w>`) so several from one doc don't collide in the per-excerpt state.

    Fetch-based for now (the reasoning window differs from the harvest's prose
    window, so it needs its own selection); materialize via the harvest later.
    """
    _, audit, con = _load_index()
    idx = _category_index(audit)
    starts, _ = _shard_starts()
    rng = random.Random(seed)
    per = max(1, n // len(categories))
    out = []
    for cat in categories:
        gis = [g for g in idx.get(cat, [])
               if (audit[g].get("topic_or_subject_score_gen") or 0) >= min_conf]
        if not gis:
            continue
        got = 0
        for gi in rng.sample(gis, len(gis)):      # scan docs until this cat's quota met
            if got >= per:
                break
            shard, local = _locate(gi, starts)
            text = _fetch_text(con, shard, local)
            if not text:
                continue
            m = audit[gi]
            for w, (ex, sig) in enumerate(
                    stem_windows(text, n_words, rng, k=per_doc, min_signal=min_signal)):
                out.append({
                    "doc_index": f"{gi}-{w}", "doc": gi, "shard": shard,
                    "category": cat, "affordance": "stem_reasoning",
                    "stem_signal": sig, "confidence": m.get("topic_or_subject_score_gen"),
                    "year": m.get("resolved_year"), "title": m.get("title_src"),
                    "prose_score": sig, "n_words": len(ex.split()), "excerpt": ex,
                })
                got += 1
                if got >= per:
                    break
    return out


def sample_excerpts(n=20, alpha=0.5, min_conf=0.5, n_words=150, seed=0):
    """Draw ~n excerpts across categories weighted by the tempered coverage plan.

    Allocates n across categories via the same tempering as plan_targets, then
    pulls that many confidence-passing docs per category and cuts a prose excerpt
    from each. Returns a list of excerpt records. Loads the index once.
    """
    _, audit, con = _load_index()
    idx = _category_index(audit)
    starts, _ = _shard_starts()
    elig = _eligible_by_category(audit, idx, min_conf)
    targets = _tempered_targets({c: len(g) for c, g in elig.items()}, n, alpha)
    rng = random.Random(seed)
    out = []
    for cat in sorted(targets, key=lambda c: -targets[c]):
        k, gis = targets[cat], elig[cat]
        if k <= 0 or not gis:
            continue
        for gi in rng.sample(gis, min(k, len(gis))):
            shard, local = _locate(gi, starts)
            text = _fetch_text(con, shard, local)
            if not text:
                continue
            ex, score, label = prose_excerpt(text, n_words, rng)
            if not ex:
                continue                          # every window hit a hard-drop
            m = audit[gi]
            out.append({
                "category": cat,
                "affordance": label,
                "confidence": m.get("topic_or_subject_score_gen"),
                "year": m.get("resolved_year"),
                "title": m.get("title_src"),
                "doc_index": gi,
                "n_words": len(ex.split()),
                "prose_score": round(score, 3),
                "excerpt": ex,
            })
    return out


def report_excerpts(n=20, alpha=0.5, min_conf=0.5, n_words=150, seed=0):
    """Sample plan-weighted excerpts and print them for inspection."""
    from collections import Counter
    recs = sample_excerpts(n, alpha=alpha, min_conf=min_conf, n_words=n_words, seed=seed)
    tally = Counter(r["category"] for r in recs)
    aff = Counter(r["affordance"] for r in recs)
    print(f"sampled {len(recs)} excerpts  (alpha={alpha}, min_conf={min_conf}, "
          f"~{n_words} words each)")
    print("  by affordance: " + "  ".join(f"{a}={k}" for a, k in aff.most_common()))
    for cat, k in tally.most_common():
        print(f"    {k:>3}  {cat}")
    for r in recs:
        print("\n" + "=" * 78)
        print(f"[{r['category']} / {r['affordance']}]  conf {r['confidence']:.2f}  "
              f"{r['year']}  prose {r['prose_score']:.2f}  {r['n_words']}w")
        print(f"  {str(r['title'])[:88]}")
        print(textwrap.fill(r["excerpt"], width=78))


def list_categories():
    """Print the taxonomy with per-category document counts (metadata only)."""
    _, audit, _ = _load_index()
    idx = _category_index(audit)
    top = max(len(v) for v in idx.values())
    print(f"{len(idx)} categories, {sum(len(v) for v in idx.values()):,} documents:")
    for cat, gis in sorted(idx.items(), key=lambda kv: -len(kv[1])):
        bar = "#" * int(40 * len(gis) / top)
        print(f"  {len(gis):>7,}  {cat:44} {bar}")


def report_by_category(category, n_text=3, seed=0, excerpt_words=(120, 240)):
    """Profile a category from metadata (free), then show excerpts for a few docs."""
    _, audit, _ = _load_index()
    idx = _category_index(audit)
    gis = idx.get(category)
    if not gis:
        print(f"unknown category {category!r}; try --list-categories")
        return
    conf = [audit[g].get("topic_or_subject_score_gen") for g in gis]
    conf = [c for c in conf if isinstance(c, (int, float))]
    years = [audit[g].get("resolved_year") for g in gis]
    years = [y for y in years if isinstance(y, int)]
    wc = [audit[g].get("word_count") for g in gis]
    wc = [w for w in wc if isinstance(w, int)]

    print(f"\n=== {category} ===")
    print(f"  documents: {len(gis):,}")
    if conf:
        lo, p50, _, _ = _stats(conf)
        suspect = sum(c < 0.5 for c in conf)
        print(f"  label confidence: min {lo:.2f}  p50 {p50:.2f}  "
              f"below 0.5: {suspect} ({suspect/len(conf):.0%})")
    if years:
        print(f"  years: {_stats(years)[0]}–{_stats(years)[3]} (p50 {_stats(years)[1]})")
    if wc:
        s = _stats(wc)
        print(f"  word_count: p50 {s[1]:,}  p90 {s[2]:,}")

    rng = random.Random(seed)
    print("  sample titles:")
    for g in rng.sample(gis, min(8, len(gis))):
        m = audit[g]
        sc = m.get("topic_or_subject_score_gen")
        sc = f"{sc:.2f}" if isinstance(sc, (int, float)) else "?"
        print(f"    [{sc}] {m.get('resolved_year','?')}  {str(m.get('title_src',''))[:80]}")

    docs = sample_by_category(category, n_text=n_text, seed=seed)
    ewin = random.Random(seed)
    for t, m in docs:
        if not t:
            continue
        print("\n" + "-" * 78)
        print(f"[{m.get('title_src','')[:70]}]  {len(t.split()):,} words")
        for nw in excerpt_words:
            ex, score, label = prose_excerpt(t, nw, ewin)
            if not ex:
                print(f"\n  --- excerpt ~{nw} words: (dropped — no self-contained window) ---")
                continue
            print(f"\n  --- excerpt ~{nw} words ({len(ex.split())} actual, {label}, prose {score:.2f}) ---")
            print(textwrap.fill(ex, width=76, initial_indent="  ", subsequent_indent="  "))


def main():
    ap = argparse.ArgumentParser(description="Sample the pretrain corpus for excerpt sizing")
    ap.add_argument("--docs", type=int, default=10, help="number of documents to sample")
    ap.add_argument("--category", type=str, default=None, help="sample within one LoC category")
    ap.add_argument("--list-categories", action="store_true", help="print taxonomy with counts")
    ap.add_argument("--coverage", action="store_true", help="print the coverage plan (targets)")
    ap.add_argument("--excerpts", action="store_true",
                    help="sample plan-weighted excerpts across categories and print them")
    ap.add_argument("--total", type=int, default=2000, help="excerpt budget for the coverage plan")
    ap.add_argument("--alpha", type=float, default=0.5, help="tempering: 1=corpus, 0=uniform")
    ap.add_argument("--min-conf", type=float, default=0.5, help="label-confidence floor")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--excerpt-words", type=str, default="60,120,240",
                    help="comma-separated excerpt lengths to preview")
    ap.add_argument("--show", type=int, default=3, help="documents to show excerpt windows for")
    args = ap.parse_args()
    ewords = [int(x) for x in args.excerpt_words.split(",")]
    if args.list_categories:
        list_categories()
    elif args.coverage:
        report_coverage(args.total, alpha=args.alpha, min_conf=args.min_conf)
    elif args.excerpts:
        report_excerpts(n=args.docs, alpha=args.alpha, min_conf=args.min_conf,
                        n_words=ewords[0], seed=args.seed)
    elif args.category:
        report_by_category(args.category, n_text=args.show, seed=args.seed, excerpt_words=ewords)
    else:
        report(n=args.docs, seed=args.seed, excerpt_words=ewords, show=args.show)


if __name__ == "__main__":
    main()
