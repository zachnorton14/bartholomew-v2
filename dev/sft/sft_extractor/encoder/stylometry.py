"""Stylometric register detection — topic-independent, no embeddings, no deps.

Represents each question by style features that carry register but not subject:
function-word relative frequencies, character trigrams, opener indicators, length.
Then per route, logistic-regression classifies vintage vs modern, cross-validated.
Because the features exclude content words, this measures prose register directly
rather than topic (which dominated the semantic embeddings).

  python encoder/stylometry.py synth/output/register_pairs.jsonl
"""
import sys, re, json, collections
import numpy as np

FUNC = ("the a an of to in on at by for and or but if nor so as than then that this these those "
        "what which who whom whose how why when where whence whither wherein whereby whereof "
        "is are was were be been being am have has had do does did shall will would should could "
        "can may might must not no yes it its they them their there here he she his her him we our "
        "us you your i my me one two three into onto unto upon under over about through between among "
        "against toward towards after before during since until while though although because "
        "hath doth thee thou thy ye unto hence thus therein hereof herein "
        "of_what in_what into_what by_what for_what such very more most").split()

OPENERS = ["into", "of", "what", "how", "why", "who", "when", "where", "which",
           "can", "is", "are", "do", "does", "describe", "name", "explain", "give", "state"]
BIGRAM_OPEN = ["of what", "into what", "in what", "can you", "how does", "what is",
               "what are", "how many", "for what", "by what", "how did"]


def tokens(s):
    return re.findall(r"[a-z]+", s.lower())


def build_chargrams(texts, k=250):
    c = collections.Counter()
    for t in texts:
        s = " " + t.lower() + " "
        for i in range(len(s) - 2):
            c[s[i:i + 3]] += 1
    return [g for g, _ in c.most_common(k)]


def featurize(text, chargrams, cg_index):
    toks = tokens(text)
    n = max(1, len(toks))
    tc = collections.Counter(toks)
    fw = [tc.get(w.replace("_", " ") if "_" in w else w, 0) / n for w in FUNC]  # func-word rel freq
    # char trigram rel freq
    s = " " + text.lower() + " "
    cg = np.zeros(len(chargrams)); tot = max(1, len(s) - 2)
    for i in range(len(s) - 2):
        j = cg_index.get(s[i:i + 3])
        if j is not None:
            cg[j] += 1
    cg /= tot
    first = toks[0] if toks else ""
    first2 = " ".join(toks[:2])
    opn = [1.0 if first == o else 0.0 for o in OPENERS]
    opn2 = [1.0 if first2 == b else 0.0 for b in BIGRAM_OPEN]
    length = [len(toks), np.mean([len(w) for w in toks]) if toks else 0.0, len(text)]
    return np.array(fw + list(cg) + opn + opn2 + length, dtype=np.float64)


def auc(scores, y):
    o = np.argsort(scores, kind="mergesort"); r = np.empty(len(scores)); r[o] = np.arange(1, len(scores) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float("nan")


def logreg(X, y, l2=1.0, lr=0.5, iters=800):
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        w -= lr * (X.T @ (p - y) / len(y) + l2 * w / len(y))
        b -= lr * (p - y).mean()
    return w, b


def cv(X, y, folds=5):
    rng = np.random.default_rng(0); idx = rng.permutation(len(y)); a = []
    mu, sd = X.mean(0), X.std(0) + 1e-8
    for f in range(folds):
        te = idx[f::folds]; tr = np.setdiff1d(idx, te)
        Xtr = (X[tr] - mu) / sd; Xte = (X[te] - mu) / sd
        w, b = logreg(Xtr, y[tr])
        a.append(auc(Xte @ w + b, y[te]))
    return float(np.mean(a))


def main():
    rows = [json.loads(l) for l in open(sys.argv[1])]
    all_texts = [r[v] for r in rows for v in ("q_vintage", "q_modern") if r.get(v)]
    chargrams = build_chargrams(all_texts)
    cg_index = {g: i for i, g in enumerate(chargrams)}

    feat_names = (["fw:" + w for w in FUNC] + ["cg:" + g for g in chargrams] +
                  ["open:" + o for o in OPENERS] + ["open2:" + b for b in BIGRAM_OPEN] +
                  ["len:tokens", "len:wordlen", "len:chars"])

    by_type = collections.defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)

    print(f"{'route':<16}{'n_pairs':>8}{'stylometry AUC':>16}   (embedding AUC was: kn/mt 0.74, stem 0.90)")
    for route in ("knowledge", "multiturn", "stem_reasoning", "reasoning"):
        rs = [r for r in by_type[route] if r.get("q_vintage") and r.get("q_modern")]
        texts = [r["q_vintage"] for r in rs] + [r["q_modern"] for r in rs]
        y = np.array([1.0] * len(rs) + [0.0] * len(rs))
        X = np.array([featurize(t, chargrams, cg_index) for t in texts])
        print(f"{route:<16}{len(rs):>8}{cv(X, y):>16.3f}")

    # inspect knowledge: which features signal modern vs period, and the extremes
    rs = [r for r in by_type["knowledge"] if r.get("q_vintage") and r.get("q_modern")]
    texts = [r["q_vintage"] for r in rs] + [r["q_modern"] for r in rs]
    y = np.array([1.0] * len(rs) + [0.0] * len(rs))
    X = np.array([featurize(t, chargrams, cg_index) for t in texts])
    mu, sd = X.mean(0), X.std(0) + 1e-8
    w, b = logreg((X - mu) / sd, y)
    idx = np.argsort(w)
    print("\nknowledge — top features signaling MODERN (negative weight):")
    for i in idx[:12]:
        print(f"  {w[i]:+.2f}  {feat_names[i]}")
    print("knowledge — top features signaling PERIOD (positive weight):")
    for i in idx[-12:][::-1]:
        print(f"  {w[i]:+.2f}  {feat_names[i]}")


if __name__ == "__main__":
    main()
