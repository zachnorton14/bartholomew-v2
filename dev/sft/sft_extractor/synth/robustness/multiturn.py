"""Multi-turn conversation rows: two people actually talking.

Everything else in the robustness set is a single exchange. This is the only route
that shows the model a real back-and-forth, which is the shape it has never been
trained on -- the graded routes are all question-and-answer over period prose, and
the prose filter that built them selected against dialogue (short sentences score
badly on mean-sentence-length, so quoted speech lands below the 0.70 floor).

Two sources, both verbatim period text:

  croqaz/vintage-conversations -- Gutenberg novels with dialogue already extracted,
      attribution stripped, speaker labelled, and every text verified letter by
      letter against the source book. Free. Supplies the bulk.
  synth/output/conversational_qa.json -- the LLM extraction over our own corpus.
      Costs API credit and yields a quarter as much, so it is the minority partner.

Filtering, in order of how much it removes:

  dialect     "Know'd it yes'day aft'noon at tea-time" is verbatim Dickens and
              wrong to train an assistant on. Rejected in assistant turns.
  names       a turn naming a character binds the exchange to a novel the visitor
              has not read; matched case-sensitively on word boundaries against
              that book's own speaker list.
  openings    a first turn that is plainly a reply ("Yes.", "Indeed.") answers a
              question nobody asked.
  punctuation the source stripped "said Mr. Brooke" but left the comma that
              introduced it, so turns ended mid-clause. Closed to a sentence.

    python -m synth.robustness.multiturn --preview
"""

import argparse
import collections
import json
import re
from pathlib import Path

ROUTE = "conversation_multiturn"
SCORE = 100
VINTAGE_REPO = "croqaz/vintage-conversations"
HARVESTED = Path(__file__).resolve().parents[1] / "output" / "conversational_qa.json"

MIN_TURNS, MAX_TURNS, MAX_CHARS = 4, 12, 320
MIN_WORDS_ASSIST = 3

ARCHAIC = re.compile(r"\b(thou|thee|thy|thine|wilt|hath|doth|prithee|methinks|nay)\b", re.I)
NARRATION = re.compile(
    r"\b(said|replied|answered|asked|cried|returned|observed|exclaimed|rejoined"
    r"|remarked|murmured|whispered|retorted|inquired|ejaculated)\b", re.I)
# Phonetic dialect. Verbatim period text, but not a voice to give the assistant.
DIALECT = re.compile(
    r"\b\w*'(?:d|ll|s|t|n|em|im|ee)\b"
    r"|\b(?:wos|wot|yer|aint|dunno|summat|nowt|hisself|theirselves|know'd|s'pose)\b", re.I)
TITLED_NAME = re.compile(r"\b(Mr|Mrs|Miss|Dr|Lady|Lord|Sir|Madame|Monsieur)\.?\s+[A-Z][a-z]+")
# An opening turn that is plainly a REPLY rather than an opening.
REPLY_OPEN = re.compile(
    r"^\s*(and|but|then|so|yet|nor|for|yes|no|indeed|certainly|quite|true|exactly|"
    r"precisely|of course|not at all|nothing|nobody|never|neither|both|either)\b", re.I)
# Speaker names too common as words to reject on.
COMMON_NAMES = {
    "will", "may", "grace", "rose", "mark", "hope", "faith", "young", "little", "old",
    "man", "woman", "boy", "girl", "doctor", "captain", "general", "king", "queen",
    "prince", "count", "duke", "madame", "monsieur", "father", "mother", "uncle",
    "aunt", "lord", "lady", "miss", "mrs", "the", "one", "first", "second", "other",
}


def clean_turn(text, role, names=None):
    t = re.sub(r"\s+", " ", (text or "")).strip().strip('“”"\'')
    # close the sentence the stripped attribution left hanging; drop a trailing dash
    # first, or "in a spirit of love—" becomes "in a spirit of love—."
    t = re.sub(r"[,;:\u2013\u2014-]+$", "", t).strip()
    if t and t[-1] not in ".!?":
        t += "."
    if not (4 <= len(t) <= MAX_CHARS):
        return None
    # A real utterance starts a sentence. Lowercase means the extractor cut into the
    # middle of one -- "one thousing seven hundred and eighty-two." is half a reply.
    if not t[0].isupper():
        return None
    if re.search(r"\d", t) or ARCHAIC.search(t) or NARRATION.search(t):
        return None
    if any(c in t for c in '“”'):
        return None
    if TITLED_NAME.search(t):
        return None
    if names is not None and names.search(t):
        return None
    if role == "assistant":
        if len(t.split()) < MIN_WORDS_ASSIST or t.rstrip().endswith("?"):
            return None                      # the assistant answers, it does not interrogate
        if DIALECT.search(t):
            return None                      # do not teach the assistant to speak in dialect
    return t


def to_conversation(turns, names=None):
    """Longest clean opening stretch, ending on an assistant turn."""
    out = []
    for i, t in enumerate(turns[:MAX_TURNS]):
        role = "user" if i % 2 == 0 else "assistant"
        content = t["content"] if isinstance(t, dict) and "content" in t else t.get("text")
        c = clean_turn(content, role, names)
        if c is None:
            break
        if i == 0 and REPLY_OPEN.match(c):
            return None
        out.append({"role": role, "content": c})
    while len(out) % 2:
        out.pop()
    return out if len(out) >= MIN_TURNS else None


def _two_party_runs(book):
    rows = [r for r in book if r.get("speaker") and r.get("text")]
    rows.sort(key=lambda r: int(r["pos"]))
    runs, cur = [], []
    for r in rows:
        if not cur:
            cur = [r]; continue
        prev = cur[-1]
        if r["speaker"] == prev["speaker"]:          # same voice twice -- a break
            runs.append(cur); cur = [r]; continue
        if len({x["speaker"] for x in cur} | {r["speaker"]}) > 2:
            runs.append(cur); cur = [prev, r]        # a third voice -- restart
            continue
        cur.append(r)
    runs.append(cur)
    return [c for c in runs if len(c) >= MIN_TURNS]


def from_vintage():
    from huggingface_hub import HfApi, hf_hub_download
    files = [s.rfilename for s in HfApi().dataset_info(VINTAGE_REPO).siblings
             if s.rfilename.endswith(".jsonl")]
    out = []
    for f in sorted(files):
        path = hf_hub_download(VINTAGE_REPO, f, repo_type="dataset")
        book = []
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            o = json.loads(line)               # one file stores an array per line
            book.extend(o) if isinstance(o, list) else book.append(o)
        book = [r for r in book if isinstance(r, dict)]
        raw = {w for r in book for w in re.split(r"[^A-Za-z]+", r.get("speaker") or "")
               if len(w) > 3 and w.lower() not in COMMON_NAMES}
        names = re.compile(r"\b(" + "|".join(sorted(map(re.escape, raw), key=len,
                           reverse=True)) + r")\b") if raw else None
        for run in _two_party_runs(book):
            conv = to_conversation([{"content": r["text"]} for r in run], names)
            if conv:
                out.append((conv, f))
    return out


def from_harvest():
    if not HARVESTED.exists():
        return []
    out = []
    for r in json.loads(HARVESTED.read_text(encoding="utf-8")):
        conv = to_conversation(r.get("conversations") or [])
        if conv:
            out.append((conv, "corpus-harvest"))
    return out


def build_rows(count=None, seed=1930):
    pairs = from_vintage() + from_harvest()
    rows = []
    for conv, src in pairs:
        rows.append({"doc_index": f"robust-{ROUTE}-{len(rows):05d}",
                     "category": ROUTE, "book_category": "ROBUSTNESS",
                     "source": src, "conversations": conv, "score": SCORE})
        if count and len(rows) >= count:
            break
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    rows = build_rows(args.count)
    if args.preview:
        src = collections.Counter(r["source"] for r in rows)
        turns = sorted(len(r["conversations"]) for r in rows)
        alen = sorted(len(m["content"].split()) for r in rows
                      for m in r["conversations"] if m["role"] == "assistant")
        print(f"{len(rows)} conversations, {sum(turns)} turns")
        print(f"turns per conversation: median {turns[len(turns)//2]}, max {max(turns)}")
        print(f"assistant words: median {alen[len(alen)//2]}, p90 {alen[int(len(alen)*.9)]}")
        print("sources:", dict(src.most_common(4)), "...")
        for r in rows[:3]:
            print()
            for m in r["conversations"][:6]:
                print(f"  {m['role'][:4].upper():5} {m['content'][:84]}")
    else:
        print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
