"""One-class period-style filter (stylometry).

(1) Diagnostic: how separable are authentic vs generated questions by style, and how
    much of that is just length?
(2) Filter: model AUTHENTIC period style alone (Gaussian over topic-blind stylometric
    features, length excluded), then score each generated question by its Mahalanobis
    distance from that manifold. Far = doesn't look period. Rank + eyeball the tail.

  python encoder/style_oneclass.py synth/output/register_pairs.jsonl knowledge
"""
import sys, json
import numpy as np
from encoder.stylometry import build_chargrams, featurize, auc, cv, logreg


def maha_setup(Xz, k=30, reg=1e-2):
    mu = Xz.mean(0); C = np.cov(Xz.T) + reg * np.eye(Xz.shape[1])
    Ci = np.linalg.inv(C)
    return mu, Ci


def maha(X, mu, Ci):
    d = X - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", d, Ci, d))


def main():
    path, route = sys.argv[1], sys.argv[2]
    rows = [json.loads(l) for l in open(path)
            if json.loads(l)["type"] == route and json.loads(l).get("q_authentic")
            and json.loads(l).get("q_vintage")]
    auth = [r["q_authentic"] for r in rows]
    gen = [r["q_vintage"] for r in rows]
    cg = build_chargrams(auth + gen)
    cgi = {g: i for i, g in enumerate(cg)}
    feat = lambda t: featurize(t, cg, cgi)
    Xa = np.array([feat(t) for t in auth]); Xg = np.array([feat(t) for t in gen])
    # last 3 cols are length features
    Xa_s, Xg_s = Xa[:, :-3], Xg[:, :-3]

    # --- (1) binary diagnostic: with length vs structure-only ---
    yb = np.array([1.0] * len(auth) + [0.0] * len(gen))
    print(f"route={route}  n_authentic={len(auth)}")
    print(f"authentic-vs-generated AUC:  with length {cv(np.vstack([Xa, Xg]), yb):.3f}   "
          f"structure-only {cv(np.vstack([Xa_s, Xg_s]), yb):.3f}")

    # --- (2) one-class on authentic style (structure only, length excluded) ---
    mu_f, sd_f = Xa_s.mean(0), Xa_s.std(0) + 1e-8
    Za = (Xa_s - mu_f) / sd_f; Zg = (Xg_s - mu_f) / sd_f
    _, S, Vt = np.linalg.svd(Za - Za.mean(0), full_matrices=False)
    P = Vt[:30].T
    Pa, Pg = Za @ P, Zg @ P
    # 5-fold held-out authentic self-distance; generated always out-of-sample
    rng = np.random.default_rng(0); idx = rng.permutation(len(Pa)); ad = np.empty(len(Pa))
    for f in range(5):
        te = idx[f::5]; tr = np.setdiff1d(idx, te)
        m, Ci = maha_setup(Pa[tr]); ad[te] = maha(Pa[te], m, Ci)
    m, Ci = maha_setup(Pa); gd = maha(Pg, m, Ci)
    sep = auc(np.concatenate([gd, ad]), np.concatenate([np.ones(len(gd)), np.zeros(len(ad))]))
    print(f"one-class: distance separates generated from authentic  AUC {sep:.3f}  "
          f"(authentic self-dist median {np.median(ad):.2f}, generated {np.median(gd):.2f})")

    order = np.argsort(gd)
    print("\nMOST period-like generated (lowest distance):")
    for i in order[:8]:
        print(f"  {gd[i]:5.2f}  {gen[i][:88]}")
    print("MOST anachronistic-flagged generated (highest distance):")
    for i in order[-8:][::-1]:
        print(f"  {gd[i]:5.2f}  {gen[i][:88]}")


if __name__ == "__main__":
    main()
