"""N-gram anachronism tests for the synthetic pre-1930 SFT *questions*.

The ANSWERS are verbatim period text (anachronism-safe by construction), so the risk is
on the QUESTION side (model-composed). These tests flag questions carrying post-1930
vocabulary or phrasing. They are complementary to the vintage-model perplexity filter:
cheap, deterministic, and explainable (they point at the exact offending n-gram).

Three tests (run them together — they catch different things):

  1. modern_unigrams  — question contains a word coined after ~1930 (curated seed list;
                        extend MODERN_UNIGRAMS as test 3 surfaces more).
  2. modern_phrases   — question contains a post-1930 collocation that is period-innocent
                        word-by-word ("cold war", "world war", "machine learning").
  3. out_of_period    — question contains a lowercase word ABSENT from the period
                        vocabulary built from the dataset's own verbatim answers. Data-
                        driven: surfaces candidate anachronisms (also OCR/rare words) to
                        grow the blocklists above. This is the one you iterate on.

Meant to run AFTER character-corruption cleaning (OCR noise inflates test 3's flags).

Run:
    python3 -m synth.ngram_filter --route stem_reasoning --sample 5000
    python3 -m synth.ngram_filter                       # every route, full
"""

import argparse
import json
import random
import re
from collections import Counter

from synth import hf_push

ROUTES = (
    "knowledge_qa", "multiturn_qa", "reasoning_qa", "stem_reasoning",
    "narrative_grounded", "narrative_fiction", "opinion_qa", "how_to_qa",
    "verse_qa", "composition_qa", "calibration_qa",
)

# --- blocklists (SEED lists — deliberately conservative; extend as test 3 finds more) --

# Words coined after ~1930 (clearly modern). Kept high-precision: excludes words that
# existed pre-1930 even if their modern *sense* is later ("web", "relativity", "quantum",
# "photon", "isotope", "television", "radioactive" are all pre-1930 and were removed).
# Test 3 is how you discover more to add here.
MODERN_UNIGRAMS = {
    "internet", "website", "online", "offline", "email", "e-mail", "download", "upload",
    "wifi", "wi-fi", "bluetooth", "smartphone", "laptop", "software",
    "blog", "blogger", "podcast", "hashtag", "selfie", "emoji", "google", "googled",
    "googling", "wikipedia", "youtube", "facebook", "cyberspace",
    "cybersecurity", "malware", "chatbot", "livestream", "cellphone", "teenager",
    "teenagers", "microchip", "transistor", "cosmonaut", "photocopier", "supermarket",
    "genocide", "napalm", "antibiotic", "antibiotics", "neutron", "radar", "sonar",
    "laser", "spacecraft",
}

# Post-1930 collocations that are period-innocent token-by-token. Match on normalized
# (lowercased) text so spacing/case don't matter. Excludes pre-1930 phrases ("middle
# class", "working class", "human rights", "credit card", "assembly line").
MODERN_PHRASES = (
    # "world war" (bare) is period-valid — WWI was called "the World War" 1914-1930s;
    # "First/Second World War" imply a sequel and were coined ~1931+, so they ARE
    # anachronistic in a pre-1930 question.
    # Excluded as period-ambiguous (false positives on the answer scan): "black hole"
    # (the Black Hole of Calcutta; a literal hole), "united nations" (the Iroquois/
    # Cromwellian united nations), "european union" (period union talk), "big bang" (a
    # literal explosion).
    "cold war", "first world war", "second world war", "world war one",
    "world war two", "machine learning", "artificial intelligence", "social media",
    "search engine", "climate change", "global warming", "greenhouse gas",
    "space race", "nuclear weapon", "nuclear weapons",
    "atomic bomb", "hydrogen bomb", "gene therapy",
    "gross domestic product",
)

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")


def _words(text):
    return _WORD.findall(text or "")


def _norm(text):
    return " ".join(w.lower() for w in _words(text))


# --- the three tests ----------------------------------------------------------------

def modern_unigrams(question):
    """Return the post-1930 words present in the question (sorted, deduped)."""
    return sorted({w.lower() for w in _words(question)} & MODERN_UNIGRAMS)


def modern_phrases(question):
    """Return the post-1930 collocations present in the question."""
    padded = f" {_norm(question)} "
    return [p for p in MODERN_PHRASES if f" {p} " in padded]


_COUNTS_CACHE = hf_push.ROOT / "synth" / "state" / "period_counts.json"


def period_counts(rebuild=False):
    """Word -> count over EVERY route's verbatim answers (the full period lexicon),
    cached to disk so it is built once. This global scope is essential: a per-route vocab
    is too narrow (stem's technical answers lack common prose words), inflating test 3."""
    if _COUNTS_CACHE.exists() and not rebuild:
        return Counter(json.loads(_COUNTS_CACHE.read_text()))
    c = Counter()
    for route in ROUTES:
        for row in _load_route(route):
            _, ans = _qa(row)
            for t in ans:
                for w in _words(t):
                    c[w.lower()] += 1
    _COUNTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _COUNTS_CACHE.write_text(json.dumps(dict(c)))
    return c


def build_period_vocab(min_count=3, rebuild=False):
    """The set of lowercase words appearing >= min_count times across ALL period answers.
    The frequency floor drops OCR noise and one-off proper nouns."""
    return {w for w, n in period_counts(rebuild).items() if n >= min_count}


def _clean_tok(w):
    wl = w.lower()
    if wl.endswith("'s"):
        wl = wl[:-2]
    return wl.strip("'-")


def out_of_period(question, vocab, min_len=3):
    """Return lowercase question words absent from the period vocab. Skips proper nouns
    (capitalized, non-sentence-initial), short words, and hyphen-compounds whose parts
    are all period words ("organ-systems") — those are legitimate, not anachronisms."""
    out = []
    for i, w in enumerate(_words(question)):
        if i > 0 and w[0].isupper():                 # likely proper noun -> skip
            continue
        wl = _clean_tok(w)
        if len(wl) < min_len or wl in vocab:
            continue
        parts = [p for p in wl.split("-") if p]
        if len(parts) > 1 and all(p in vocab for p in parts):
            continue                                 # compound of period words
        out.append(wl)
    return sorted(set(out))


# --- data loading -------------------------------------------------------------------

def _load_route(route, fresh=False):
    """Load a route's rows from HF. fresh=True forces a re-download (bypass local cache)
    — use it when filtering, so you operate on the current published data."""
    from huggingface_hub import hf_hub_download
    api = hf_push._api()
    shards = sorted(f for f in api.list_repo_files(hf_push.HF_REPO, repo_type="dataset")
                    if f.startswith(f"unfiltered/{route}/") and f.endswith(".jsonl"))
    rows = []
    for f in shards:
        path = hf_hub_download(hf_push.HF_REPO, f, repo_type="dataset", force_download=fresh)
        with open(path, encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    return rows


def _qa(row):
    """(question strings, period strings) for a row — handles both single-turn and
    multiturn shapes. User turns are the (composed) questions; assistant turns are the
    verbatim period text."""
    convs = row.get("conversations")
    if convs:
        q = [m["content"] for m in convs if m.get("role") == "user"]
        a = [m["content"] for m in convs if m.get("role") == "assistant"]
    else:
        q, a = [row.get("question", "")], [row.get("answer", "")]
    return q, a


# --- runner -------------------------------------------------------------------------

def _scan(label, items, vocab=None, show=12):
    """Scan a list of (route, doc_index, text) with the blocklists, plus out_of_period
    when `vocab` is given (question side only — OOV is meaningless on answers, since the
    vocab is built FROM the answers). Prints a report; returns the hits dict."""
    hits = {"modern_unigrams": [], "modern_phrases": []}
    if vocab is not None:
        hits["out_of_period"] = []
    oov = Counter()
    for route, di, text in items:
        u, p = modern_unigrams(text), modern_phrases(text)
        if u:
            hits["modern_unigrams"].append((route, di, text, u))
        if p:
            hits["modern_phrases"].append((route, di, text, p))
        if vocab is not None:
            o = out_of_period(text, vocab)
            if o:
                hits["out_of_period"].append((route, di, text, o))
                oov.update(o)
    print(f"\n########## {label}: {len(items):,} texts ##########")
    for name, rows in hits.items():
        pct = 100 * len(rows) / max(1, len(items))
        print(f"=== {name}: {len(rows):,} flagged ({pct:.2f}%) ===")
        for route, di, text, terms in rows[:show]:
            print(f"  [{route} {di}] {terms}")
            print(f"      {text[:140]}")
        print()
    if oov:
        print("top out-of-period tokens (candidates for MODERN_UNIGRAMS / review):")
        for w, n in oov.most_common(40):
            print(f"  {n:>5}  {w}")
    return hits


def run(routes, side="question", sample=0, seed=0, min_count=3, show=12, rebuild=False):
    """Run the tests over `side` ('question' | 'answer' | 'both') of `routes`. Questions
    get all three tests (with the cached global period vocab); answers get the two
    blocklists — which catch anachronism that came IN with the source corpus (a misdated
    or modern-reprint document whose verbatim answer is not actually pre-1930)."""
    vocab = build_period_vocab(min_count, rebuild) if side in ("question", "both") else None
    q_items, a_items = [], []
    for r in routes:
        rows = _load_route(r)
        if sample and len(rows) > sample:
            rows = random.Random(seed).sample(rows, sample)
        for row in rows:
            qs, ans = _qa(row)
            di = row.get("doc_index")
            if side in ("question", "both"):
                q_items += [(r, di, q) for q in qs]
            if side in ("answer", "both"):
                a_items += [(r, di, a) for a in ans]

    if vocab is not None:
        print(f"period vocab: {len(vocab):,} words (>= {min_count} uses, all routes)")
    results = {}
    if q_items:
        results["question"] = _scan("QUESTIONS", q_items, vocab=vocab, show=show)
    if a_items:
        results["answer"] = _scan("ANSWERS (verbatim period text — blocklist only)",
                                  a_items, vocab=None, show=show)
    return results


def main():
    ap = argparse.ArgumentParser(description="n-gram anachronism tests over the SFT questions")
    ap.add_argument("--route", choices=ROUTES, help="single route (default: all)")
    ap.add_argument("--side", choices=("question", "answer", "both"), default="question",
                    help="which side to test (answers catch source-corpus anachronism)")
    ap.add_argument("--sample", type=int, default=0, help="test a seeded sample of N rows/route (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-count", type=int, default=3, help="period-vocab frequency floor")
    ap.add_argument("--rebuild-vocab", action="store_true", help="rebuild the cached global period vocab")
    args = ap.parse_args()
    run([args.route] if args.route else list(ROUTES), side=args.side,
        sample=args.sample, seed=args.seed, min_count=args.min_count, rebuild=args.rebuild_vocab)


if __name__ == "__main__":
    main()
