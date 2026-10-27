"""Register-axis diagnostic on Talkie embeddings.

The make-or-break question: is there a register axis (vintage vs modern) that is
BLIND to question type? If type dominates the geometry and register is invisible,
embeddings are the wrong tool. Uses only numpy.

  python encoder/analyze_register.py out/pairs.emb.npy out/pairs.keys.jsonl
"""
import sys, json
import numpy as np


def auc(scores, y):
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float("nan")


def pca(X, k):
    mu = X.mean(0); Xc = X - mu
    C = (Xc.T @ Xc) / len(Xc)
    w, V = np.linalg.eigh(C)
    comp = V[:, ::-1][:, :k]
    return mu, comp


def lda_dir(Z, y, reg=1e-2):
    m1, m0 = Z[y == 1].mean(0), Z[y == 0].mean(0)
    Sw = np.cov(Z[y == 1].T, bias=True) * (y == 1).sum() + np.cov(Z[y == 0].T, bias=True) * (y == 0).sum()
    Sw /= len(y)
    return np.linalg.solve(Sw + reg * np.eye(Z.shape[1]), m1 - m0)


def cv_auc(Z, y, folds=5, seed=0):
    """Z is already PCA-reduced (fit once globally — unsupervised, so shared is fine)."""
    rng = np.random.default_rng(seed); idx = rng.permutation(len(y))
    aucs = []
    for f in range(folds):
        te = idx[f::folds]; tr = np.setdiff1d(idx, te)
        w = lda_dir(Z[tr], y[tr])
        aucs.append(auc(Z[te] @ w, y[te]))
    return float(np.mean(aucs))


def main():
    emb = np.load(sys.argv[1]).astype(np.float32)
    keys = [json.loads(l) for l in open(sys.argv[2])]
    typ = np.array([k["type"] for k in keys]); var = np.array([k["variant"] for k in keys])
    ids = np.array([k["id"] for k in keys])
    types = sorted(set(typ))

    vin, mod = var == "vintage", var == "modern"
    gmean = emb.mean(0)
    tot_var = ((emb - gmean) ** 2).sum(1).mean()
    # variance explained by TYPE centroids vs by REGISTER (variant) centroids
    tvar = sum((typ == t).mean() * ((emb[typ == t].mean(0) - gmean) ** 2).sum() for t in types)
    rvar = sum((var == v).mean() * ((emb[var == v].mean(0) - gmean) ** 2).sum() for v in ("vintage", "modern"))
    print(f"variance: total {tot_var:.1f} | by TYPE {tvar:.1f} ({tvar/tot_var:.1%}) | "
          f"by REGISTER {rvar:.2f} ({rvar/tot_var:.2%})")
    print(f"  => type explains {tvar/max(rvar,1e-9):.0f}x more variance than register\n")

    # fit PCA once globally (unsupervised), reuse for every classifier
    mu, comp = pca(emb, 100); Z = (emb - mu) @ comp

    # 1. Is register linearly detectable? (vintage vs modern)  and is TYPE detectable?
    reg_auc = cv_auc(Z[vin | mod], (var[vin | mod] == "vintage").astype(float))
    # type detectability: one-vs-rest avg AUC
    t_aucs = [cv_auc(Z, (typ == t).astype(float)) for t in types]
    print(f"C2ST cross-val AUC:")
    print(f"  register (vintage vs modern): {reg_auc:.3f}   (0.5 = invisible, 1.0 = perfectly separable)")
    print(f"  type (one-vs-rest, mean):     {np.mean(t_aucs):.3f}   (how dominant type is)\n")

    # 2. Register axis and its type-blindness
    w = emb[vin].mean(0) - emb[mod].mean(0); w /= np.linalg.norm(w)
    proj = emb @ w
    # along the register axis: between-variant spread vs between-type spread
    bv = sum((var == v).mean() * (proj[var == v].mean() - proj.mean()) ** 2 for v in ("vintage", "modern"))
    bt = sum((typ == t).mean() * (proj[typ == t].mean() - proj.mean()) ** 2 for t in types)
    print(f"along the register axis: between-REGISTER spread {bv:.3f} | between-TYPE spread {bt:.3f} "
          f"({'type-blind' if bt < bv else 'TYPE-CONTAMINATED'})\n")

    # 3. Cross-type generalization: register direction from other types, tested on held-out type
    print("cross-type generalization (register dir from OTHER types, AUC on held-out type):")
    for t in types:
        other = typ != t
        wt = emb[other & vin].mean(0) - emb[other & mod].mean(0)
        sel = (typ == t) & (vin | mod)
        a = auc(emb[sel] @ wt, (var[sel] == "vintage").astype(float))
        print(f"  {t:<20} {a:.3f}")

    # 4. Per-type register-direction consistency (cosine between per-type v-m directions)
    dirs = {}
    for t in types:
        d = emb[(typ == t) & vin].mean(0) - emb[(typ == t) & mod].mean(0)
        dirs[t] = d / np.linalg.norm(d)
    cos = np.array([[dirs[a] @ dirs[b] for b in types] for a in types])
    off = cos[~np.eye(len(types), dtype=bool)]
    print(f"\nper-type register-direction cosine: mean {off.mean():.3f} (1.0 = same direction everywhere)")

    # 5. Calibration: does generated vintage land nearer AUTHENTIC than modern does? (knowledge)
    auth_ids = {k["id"] for k in keys if k["variant"] == "authentic"}
    idx = {(i_id, v): j for j, (i_id, v) in enumerate(zip(ids, var))}
    def unit(x): return x / np.linalg.norm(x)
    cv, cm = [], []
    for aid in auth_ids:
        if (aid, "vintage") in idx and (aid, "modern") in idx:
            a = unit(emb[idx[(aid, "authentic")]])
            cv.append(a @ unit(emb[idx[(aid, "vintage")]]))
            cm.append(a @ unit(emb[idx[(aid, "modern")]]))
    print(f"\nknowledge calibration (n={len(cv)}): cos(authentic, vintage) {np.mean(cv):.3f}  "
          f"vs cos(authentic, modern) {np.mean(cm):.3f}  "
          f"({'vintage closer ✓' if np.mean(cv) > np.mean(cm) else 'modern closer'})")


if __name__ == "__main__":
    main()
