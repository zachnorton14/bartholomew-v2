"""Model-free metrics for synthetic-question quality.

Each metric compares the synthetic questions against the authentic ones from the
SAME matched pairs, so authentic is the built-in target — a synthetic number is
"good" when it lands near its authentic counterpart, not near zero.

  echo         — fraction of a question's content words that also appear in the
                 answer. High echo = the question is a re-arrangement of the
                 answer (lexical inversion), which teaches word-shuffling, not
                 recall. Length-normalized loss cannot see this.
  fidelity     — how close the synthetic question is to the AUTHENTIC question for
                 the same answer, word-for-word. jaccard = shared content words /
                 union (bag); run = longest shared run of consecutive words as a
                 fraction of the synthetic question. Low fidelity = the generator
                 wrote a genuinely different question, not a paraphrase of the real
                 one. (This is a single number per pair — there is no authentic
                 baseline to compare against; the authentic question IS the target.)
  template     — share of questions opening "What is/are". Catechism openers are
                 varied (Why, How, Of what, Have, Can...); a high share means the
                 generator collapsed onto onhoie frame.
  len_ratio    — mean question length in chars. Terse synthetic questions strip
                 down to the answer's key terms; the gap tracks over-compression
                 that echo alone misses.

Usage: python -m synth.metrics [output/synth/questions_period.json ...]
"""

import json
import re
import sys
from pathlib import Path

STOP = set(
    "a an the is are was were be been of to in on for and or but what which who how why "
    "does do did it its this that these those with by from as at not no yes so if when "
    "where their they them there we you he she his her".split()
)


def _content(s):
    return {w for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP and len(w) > 2}


def _echo(question, answer):
    q, a = _content(question), _content(answer)
    return len(q & a) / len(q) if q else 0.0


def _is_template(question):
    return bool(re.match(r"\s*what\s+(is|are)\b", question.lower()))


def _words(s):
    return re.findall(r"[a-z]+", s.lower())


def _qq_jaccard(synthetic_q, authentic_q):
    """Content-word overlap between the two questions: shared / union."""
    s, a = _content(synthetic_q), _content(authentic_q)
    if not (s or a):
        return 0.0
    return len(s & a) / len(s | a)


def _qq_run(synthetic_q, authentic_q):
    """Longest run of consecutive words the synthetic question shares verbatim
    (contiguous, in order) with the authentic question, over synthetic length."""
    s, a = _words(synthetic_q), _words(authentic_q)
    if not s:
        return 0.0
    a_text = " " + " ".join(a) + " "
    best = 0
    for start in range(len(s)):
        for end in range(start + best + 1, len(s) + 1):
            if f" {' '.join(s[start:end])} " in a_text:
                best = end - start
            else:
                break
    return best / len(s)


def evaluate(records):
    """Return {metric: (authentic_value, synthetic_value)} over matched pairs."""
    n = len(records)
    if not n:
        return {}
    exact = sum(r["authentic_q"].strip().lower() == r["synthetic_q"].strip().lower()
                for r in records) / n
    au_echo = sum(_echo(r["authentic_q"], r["answer"]) for r in records) / n
    sy_echo = sum(_echo(r["synthetic_q"], r["answer"]) for r in records) / n
    qq_jac = sum(_qq_jaccard(r["synthetic_q"], r["authentic_q"]) for r in records) / n
    qq_run = sum(_qq_run(r["synthetic_q"], r["authentic_q"]) for r in records) / n
    au_tmpl = sum(_is_template(r["authentic_q"]) for r in records) / n
    sy_tmpl = sum(_is_template(r["synthetic_q"]) for r in records) / n
    au_len = sum(len(r["authentic_q"]) for r in records) / n
    sy_len = sum(len(r["synthetic_q"]) for r in records) / n
    return {
        "_paired": {
            "echo": (au_echo, sy_echo),
            "template": (au_tmpl, sy_tmpl),
            "len_chars": (au_len, sy_len),
        },
        "_fidelity": {
            "exact_match": exact,
            "qq_jaccard": qq_jac,
            "qq_run": qq_run,
        },
    }


def report(path):
    records = json.loads(Path(path).read_text())
    m = evaluate(records)
    name = Path(path).stem.replace("questions_", "")
    print(f"{name}  (n={len(records)})")
    print("  -- synthetic vs authentic baseline (closer to authentic = better) --")
    for metric, (au, sy) in m["_paired"].items():
        if metric == "len_chars":
            print(f"  {metric:12} authentic {au:6.1f}   synthetic {sy:6.1f}   "
                  f"(synth is {sy/au:.0%} of authentic)")
        else:
            print(f"  {metric:12} authentic {au:6.1%}   synthetic {sy:6.1%}   "
                  f"(gap {sy-au:+.1%})")
    print("  -- fidelity: how close synth is to the real question, word-for-word --")
    for metric, v in m["_fidelity"].items():
        print(f"  {metric:12} {v:6.1%}")
    print()


if __name__ == "__main__":
    paths = sys.argv[1:] or sorted(
        (Path(__file__).parent / "output").glob("questions_*.json")
    )
    for p in paths:
        report(p)
