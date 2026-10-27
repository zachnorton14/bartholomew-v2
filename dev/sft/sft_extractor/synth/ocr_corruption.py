"""Detect OCR corruption that survives into the generated dataset.

corpus.ocr_score / has_broken_math already reject garbled MATH at excerpt-selection
time. This module covers the classes they don't, which are the ones that reach the
SFT answers as plausible-looking prose:

  long_s        18th-century long-s (ſ) misread as f: "fome"->some, "thefe"->these,
                "himfelf"->himself. The damage is invisible after the fact -- the
                result is plain ASCII, so no character normalization catches it, and
                training on it teaches the model misspellings as period usage.
  homoglyph     a Latin word carrying a stray Greek/Cyrillic lookalike ("vοlume" with
                U+03BF omicron). Per Unicode UTS #39, judged by mixed script within a
                word so genuine Greek/Cyrillic quotations are untouched.
  mojibake      UTF-8 read as latin-1 ("Ã©", "â€™") and U+FFFD replacement chars.
  hyphen_break  line-break hyphenation left unjoined ("con- sider").

Loss is computed on ANSWERS only, so answers are what these numbers should govern:
a corrupted question is context the model reads, a corrupted answer is a target it
learns to reproduce.

The long-s detector needs an inflected dictionary -- /usr/share/dict/words is a 1934
Webster's base-form list, so checking against it raw flags "flows", "lifted", and
"favour" as corruptions. _english() expands it with regular inflections and British
variants; validated at 16/17 recall on known long-s forms with 0 false positives on
the confusable set (see --selftest).

    python -m synth.ocr_corruption --selftest
    python -m synth.ocr_corruption --report
"""
import re
import json
import argparse
import unicodedata
import functools
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROUTES = ["knowledge_qa", "reasoning_qa", "narrative_grounded", "narrative_fiction",
          "composition_qa", "how_to_qa", "opinion_qa", "verse_qa", "multiturn_qa",
          "calibration_qa"]
DICT_PATH = "/usr/share/dict/words"

_TOK = re.compile(r"[A-Za-z][A-Za-z'-]*")
_HYPHEN_BREAK = re.compile(r"\b([A-Za-z]{2,})-\s+([a-z]{2,})\b")
_MOJIBAKE = re.compile(r"Ã[\x80-\xbf©®¨«»]|â€[\x99\x9c\x9d\x93\x94]|Â[\xa0-\xbf]|�")

# Greek/Cyrillic characters that are visual lookalikes for Latin letters.
_CONFUSABLE = {
    "ο": "o", "ν": "v", "α": "a", "ρ": "p", "τ": "t", "ι": "i", "κ": "k",
    "χ": "x", "ε": "e", "υ": "u", "ϲ": "c", "ѕ": "s", "һ": "h",
    "А": "A", "О": "O", "Е": "E", "Р": "P", "С": "C", "Т": "T", "Х": "X",
    "М": "M", "Н": "H", "К": "K", "В": "B", "а": "a", "о": "o", "е": "e",
    "р": "p", "с": "c", "х": "x", "у": "y", "и": "u",
}


def _inflect(w):
    """Regular English inflections of a base form."""
    out = {w, w + "s", w + "es", w + "ed", w + "d", w + "ing",
           w + "er", w + "est", w + "ly", w + "ers", w + "ings"}
    if w.endswith("e"):
        out |= {w[:-1] + s for s in ("ing", "ed", "er", "est")}
    if w.endswith("y"):
        out |= {w[:-1] + s for s in ("ies", "ied", "ier", "iest", "ily")}
    # consonant doubling: run -> running
    if len(w) > 2 and w[-1] not in "aeiouwxy" and w[-2] in "aeiou" and w[-3] not in "aeiou":
        out |= {w + w[-1] + s for s in ("ed", "ing", "er")}
    return out


@functools.lru_cache(maxsize=1)
def _english():
    """Inflected English vocabulary, with British and archaic forms."""
    try:
        base = {w.strip().lower() for w in open(DICT_PATH, errors="replace") if w.strip()}
    except OSError:
        return frozenset()
    words = set()
    for w in base:
        words |= _inflect(w)
    variants = set()
    for w in words:
        # British -our: favor/favors/favored/favoring -> favour/favours/favoured/favouring
        for suf in ("", "s", "ed", "ing"):
            if w.endswith("or" + suf) and len(w) > 2 + len(suf):
                variants.add(w[:len(w) - len(suf) - 2] + "our" + suf)
        if w.endswith("ize"):   variants.add(w[:-3] + "ise")
        if w.endswith("ized"):  variants.add(w[:-4] + "ised")
        if w.endswith("izing"): variants.add(w[:-5] + "ising")
        if w.endswith("izes"):  variants.add(w[:-4] + "ises")
        if w.endswith("er"):    variants.add(w[:-2] + "re")
    words |= variants
    words |= {"hast", "doth", "thou", "thee", "thy", "hath", "shalt", "unto", "ye",
              "art", "wilt", "o'er", "'tis", "ere", "oft", "nay", "yea", "whilst"}
    return frozenset(words)


def long_s_errors(text, systemic=True):
    """[(bad_word, correction)] for long-s misreads (f where s belongs).

    Long-s damage is a property of a SCAN, not a word: a passage set in 18th-century
    type has it throughout ("fome of thefe muft"), never once in isolation. A lone hit
    is nearly always a proper noun the dictionary lacks -- Keefer, FOWLE, Rufe, Fion,
    Faber -- so with systemic=True a text needs two independent hits, or a literal ſ,
    before any are reported. That one rule is what separates ~50% precision from ~95%.

    Capitalized words mid-sentence are skipped outright as probable proper nouns; the
    real long-s corpus is overwhelmingly lowercase function and content words.
    """
    words = _english()
    if not words:
        return []
    out = []
    # sentence-initial positions, where a capital is expected rather than a name
    starts = {m.end() for m in re.finditer(r"(?:^|[.!?:;\"']\s+)", text)}
    for m in _TOK.finditer(text):
        w = m.group()
        lw = w.lower()
        if "f" not in lw or len(lw) < 3 or lw in words:
            continue
        if w[0].isupper() and m.start() not in starts:
            continue  # proper noun, not a scan error
        for i, c in enumerate(lw):
            if c != "f":
                continue
            cand = lw[:i] + "s" + lw[i + 1:]
            if cand in words:
                out.append((w, cand))
                break
        else:
            cand = lw.replace("f", "s")
            if cand in words and cand != lw:
                out.append((w, cand))
    if systemic and len(out) < 2 and "ſ" not in text:
        return []
    return out


def _script(ch):
    if ch.isascii():
        return "Latin" if ch.isalpha() else None
    if not ch.isalpha():
        return None
    name = unicodedata.name(ch, "")
    for s in ("GREEK", "CYRILLIC", "HEBREW", "ARABIC", "DEVANAGARI"):
        if name.startswith(s):
            return s
    return "Latin" if name.startswith("LATIN") else "Other"


def homoglyph_words(text):
    """[(word, repaired)] for Latin words polluted by a Greek/Cyrillic lookalike."""
    out = []
    for word in text.split():
        scripts = {s for s in (_script(c) for c in word) if s}
        if "Latin" not in scripts or len(scripts) != 2:
            continue
        other = (scripts - {"Latin"}).pop()
        foreign = [c for c in word if _script(c) == other]
        latin = sum(1 for c in word if _script(c) == "Latin")
        if latin > len(foreign) and all(c in _CONFUSABLE for c in foreign):
            out.append((word, "".join(_CONFUSABLE.get(c, c) for c in word)))
    return out


def join_hyphen_breaks(text):
    """Rejoin line-break hyphenation, but ONLY where the joined form is a real word.

    Returns (text, n_joined, n_left). The dictionary gate is what makes this safe to
    run blind: "after- wards" -> "afterwards" and "fluctu- ations" -> "fluctuations"
    join, while a suspended hyphen ("rod- and cone-layer") or a dash used as
    punctuation ("audience- especially") would join to a non-word and is left exactly
    as it was. The cost is recall on proper nouns and rare compounds -- "Plassen- burg"
    stays broken because "Plassenburg" is not in the dictionary -- which is the right
    trade when the alternative is silently welding two unrelated words together.
    """
    words = _english()
    joined = left = 0

    def sub(m):
        nonlocal joined, left
        a, b = m.group(1), m.group(2)
        cand = (a + b).lower()
        if words and cand in words:
            joined += 1
            return a + b
        left += 1
        return m.group(0)

    return _HYPHEN_BREAK.sub(sub, text), joined, left


# A word broken across a page boundary, with the page's furniture (folio number,
# running head, chapter line) landing between the halves: "par- 21. CHAP. liament".
# Start of a page-broken word; the wedged furniture that follows is searched token by
# token in _join_furniture, because a single regex commits to one split and cannot
# retry when that split fails the dictionary gate.
_FURN_HEAD = re.compile(r"\b([A-Za-z]{2,})-\s+")
_FURN_TOKEN = re.compile(r"\S+\s*")
# Hyphenated compound with a stray space after the hyphen: "Lieutenant- Governor".
_COMPOUND = re.compile(r"\b([A-Za-z]{2,})-\s+([A-Z][a-z]{1,})\b")
# A word left hanging at the very end -- its continuation was never captured.
_TRAILING = re.compile(r"[A-Za-z]{2,}-\s*$")
_PREFIXES = {"pre", "anti", "cis", "trans", "non", "ex", "post", "semi", "quasi",
             "vice", "sub", "super", "co", "re", "un", "inter", "intra", "pro", "neo"}


def _join_furniture(text, max_wedge=4):
    """Rejoin a word split across a page boundary, discarding the wedged furniture.

    "par- 21. CHAP. liament" -> "parliament". After each "word-" the following tokens
    are tried in turn as the wedge, and the first candidate whose tail completes a
    dictionary word wins; if none does, the span is left untouched. Searching splits
    here rather than in a regex is what makes it work -- a regex quantifier commits to
    one split and cannot back up once the dictionary rejects it.
    """
    words = _english()
    if not words:
        return text, 0
    out, pos, n = [], 0, 0
    for m in _FURN_HEAD.finditer(text):
        if m.start() < pos:
            continue                      # inside a span already consumed
        head = m.group(1)
        rest = text[m.end():]
        off, skipped, hit = 0, 0, None
        while skipped <= max_wedge:
            tm = _FURN_TOKEN.match(rest, off)
            if not tm:
                break
            word = re.match(r"[a-z]{2,}", tm.group().strip())
            if word and (head + word.group()).lower() in words:
                # skipped == 0 means the halves are adjacent -> join_hyphen_breaks' job
                if skipped:
                    hit = (off + word.end(), head + word.group())
                break
            off, skipped = tm.end(), skipped + 1
        if hit:
            end, joined = hit
            out.append(text[pos:m.start()])
            out.append(joined)
            pos = m.end() + end
            n += 1
    out.append(text[pos:])
    return "".join(out), n


def repair_hyphenation(text):
    """Repair every recoverable hyphen break. Returns (text, {class: n}).

    Three passes, each gated so a wrong join is impossible or harmless:
      furniture  "par- 21. CHAP. liament" -> "parliament", accepted only when the two
                 halves form a dictionary word, so the wedged text is provably junk.
      compound   "Lieutenant- Governor" -> "Lieutenant-Governor": closes the space but
                 KEEPS the hyphen, which is the correct rendering of the compound. Only
                 fires for a known prefix or two capitalised parts.
      plain      the original dictionary-gated rejoin ("after- wards" -> "afterwards").

    Trailing hyphens are NOT repaired -- see unrepairable_hyphen(); the continuation is
    gone and inventing it would be worse than dropping the row.
    """
    words = _english()
    stats = {}

    def comp(m):
        a, b = m.group(1), m.group(2)
        if a.lower() in _PREFIXES or a[0].isupper():
            stats["compound"] = stats.get("compound", 0) + 1
            return f"{a}-{b}"
        return m.group(0)

    text, n = _join_furniture(text)
    if n:
        stats["furniture"] = n
    text = _COMPOUND.sub(comp, text)
    text, joined, _ = join_hyphen_breaks(text)
    if joined:
        stats["plain"] = joined
    return text, stats


def unrepairable_hyphen(text):
    """True if the text ends mid-word ("...the most ter-"); nothing can restore it."""
    return bool(_TRAILING.search(text.rstrip()))


def broken_fragments(text):
    """Hyphen breaks that survive join_hyphen_breaks AND look genuinely destroyed.

    After the dictionary-gated join, three kinds of residue remain: proper nouns
    ("South- ampton"), punctuation dashes ("naturalists- which"), and text the scan
    truncated ("coun- as", "con- of"). Only the last is corruption. The discriminator
    is the pre-hyphen fragment: "naturalists" and "spring" are words used before a
    dash, while "coun" and "indorse" are word fragments left by a dropped line.
    """
    words = _english()
    if not words:
        return []
    out = []
    for m in _HYPHEN_BREAK.finditer(text):
        a, b = m.group(1), m.group(2)
        if (a + b).lower() in words:
            continue                      # join_hyphen_breaks would have fixed it
        if a.lower() in words or a[0].isupper():
            continue                      # real word before a dash, or a proper noun
        out.append(m.group(0))
    return out


def corruption_report(text):
    """{class: count} of corruption found; empty dict means clean."""
    r = {}
    if (n := len(long_s_errors(text))):
        r["long_s"] = n
    if (n := len(homoglyph_words(text))):
        r["homoglyph"] = n
    if (n := len(_MOJIBAKE.findall(text))):
        r["mojibake"] = n
    if (n := len(broken_fragments(text))):
        r["broken_fragment"] = n
    return r


def repair(text):
    """Apply the unambiguous repairs (long-s, homoglyph). Mojibake and hyphen breaks
    are reported but NOT auto-fixed -- both need context this function doesn't have."""
    for bad, good in long_s_errors(text):
        fixed = good.capitalize() if bad[0].isupper() else good
        text = re.sub(rf"\b{re.escape(bad)}\b", fixed, text)
    for bad, good in homoglyph_words(text):
        text = text.replace(bad, good)
    return text


def _selftest():
    hit = "fome thefe fhould fhall muft moft firft juft himfelf caufe becaufe prefent againft wifh fuch paffed"
    miss = "flows lifted failed favour favoured fewer flashed fearing fainted wafted soft often left first fast"
    h = {w for w, _ in long_s_errors(hit)}
    m = {w for w, _ in long_s_errors(miss, systemic=False)}
    print(f"long_s recall     : {len(h)}/{len(hit.split())} {sorted(h)}")
    print(f"long_s false pos  : {len(m)}/{len(miss.split())} {sorted(m)}")
    # isolated hits (proper nouns) must not survive the systemic rule
    lone = ["Keefer claims that light woolen socks are best for marching.",
            "The name Fingal is used in English, but in Gaelic the name is Fion.",
            "Faber du Faur proved this notion false.",
            "Rufe was grave. I never saw him hurried."]
    strays = [t for t in lone if long_s_errors(t)]
    print(f"proper-noun FPs   : {len(strays)}/4 (systemic rule) {strays}")
    real = "Make ufe of no other Sort of Hurdles than thefe, while we in Hertfordfhire ufe them."
    print(f"systemic kept     : {[w for w,_ in long_s_errors(real)]}")
    assert not strays and long_s_errors(real)
    hg = homoglyph_words("the vοlume of the Аmerican book")
    print(f"homoglyph         : {hg}")
    print(f"greek untouched   : {homoglyph_words('λόγος ἐστίν')} (should be [])")
    print(f"repair            : {repair('fome of thefe muft be juft')!r}")
    assert not m and len(h) >= 15 and hg and not homoglyph_words("λόγος ἐστίν")
    print("OK")


def _rows(path):
    txt = path.read_text(errors="replace").strip()
    try:
        d = json.loads(txt)
        return d if isinstance(d, list) else [d]
    except json.JSONDecodeError:
        return [json.loads(l) for l in txt.splitlines() if l.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hfds", default=str(ROOT / "synth" / "output" / "hfds"))
    p.add_argument("--routes", nargs="*", default=ROUTES)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--examples", type=int, default=3)
    args = p.parse_args()
    if args.selftest:
        return _selftest()
    if not args.report:
        p.error("pass --report or --selftest")

    grand, shown = {}, []
    tot_a = tot_bad = 0
    print(f"{'route':20s} {'answers':>9s} {'corrupt':>8s} {'rate':>7s}  classes")
    for route in args.routes:
        n_a = n_bad = 0
        cls = {}
        for f in sorted(Path(args.hfds, route).glob("*")):
            if f.suffix not in (".json", ".jsonl"):
                continue
            for row in _rows(f):
                texts = []
                if row.get("answer"):
                    texts.append(row["answer"])
                for c in (row.get("conversations") or []):
                    if c.get("role") == "assistant" and c.get("content"):
                        texts.append(c["content"])
                for t in texts:
                    n_a += 1
                    r = corruption_report(t)
                    if r:
                        n_bad += 1
                        for k, v in r.items():
                            cls[k] = cls.get(k, 0) + v
                        if len(shown) < args.examples * len(args.routes) and "long_s" in r:
                            bad = long_s_errors(t)[:3]
                            shown.append((route, bad, t[:100]))
        tot_a += n_a
        tot_bad += n_bad
        for k, v in cls.items():
            grand[k] = grand.get(k, 0) + v
        print(f"{route:20s} {n_a:>9,} {n_bad:>8,} {n_bad/max(n_a,1):>6.2%}  {cls}")
    print(f"\n{'TOTAL':20s} {tot_a:>9,} {tot_bad:>8,} {tot_bad/max(tot_a,1):>6.2%}")
    for k, v in sorted(grand.items(), key=lambda x: -x[1]):
        print(f"  {k:14s} {v:>9,}")
    if shown:
        print("\nexamples:")
        for route, bad, snip in shown[:12]:
            print(f"  [{route}] {bad} :: {snip}...")


if __name__ == "__main__":
    main()
