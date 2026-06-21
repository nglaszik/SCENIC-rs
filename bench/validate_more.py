"""Validate + benchmark AUCell (vs numpy) and GRNBoost2 (vs sklearn GBM)."""
import time
import numpy as np
from validate_genie3 import load_pbmc3k
import scenic_rs


# ----------------------------- AUCell -----------------------------
def aucell_numpy(X, genes, regulons, amr):
    name2i = {g: i for i, g in enumerate(genes)}
    ncells, ngenes = X.shape
    names = list(regulons.keys())
    reg_idx = [[name2i[g] for g in regulons[n] if g in name2i] for n in names]
    out = np.zeros((ncells, len(names)), dtype=np.float32)
    for c in range(ncells):
        order = np.argsort(-X[c], kind="stable")
        rank = np.empty(ngenes, dtype=np.int64)
        rank[order] = np.arange(ngenes)
        for j, idxs in enumerate(reg_idx):
            if not idxs:
                continue
            r = rank[idxs]
            raw = int(np.sum((amr - r)[r < amr]))
            out[c, j] = raw / (amr * len(idxs))
    return out, names


def validate_aucell(X, genes):
    rng = np.random.default_rng(0)
    regulons = {f"reg{i}": list(rng.choice(genes, size=30, replace=False)) for i in range(100)}
    amr = max(1, int(np.ceil(0.05 * len(genes))))

    t0 = time.perf_counter()
    df = scenic_rs.aucell(X, genes, regulons, auc_max_rank=amr)
    t_rust = time.perf_counter() - t0

    t0 = time.perf_counter()
    ref, names = aucell_numpy(X, genes, regulons, amr)
    t_np = time.perf_counter() - t0

    rust = df[names].to_numpy()
    max_diff = np.max(np.abs(rust - ref))
    print("=== AUCell (scenic-rs vs numpy, identical formula) ===")
    print(f"  max abs diff: {max_diff:.2e}")
    print(f"  scenic-rs (Rust): {t_rust:6.3f} s")
    print(f"  numpy loop:       {t_np:6.3f} s")
    print(f"  speedup:          {t_np / t_rust:6.1f}x\n")


# ----------------------------- GRNBoost2 -----------------------------
def grnboost2_sklearn(X, genes, tfs, n_estimators, lr, max_depth, subsample):
    from sklearn.ensemble import GradientBoostingRegressor
    from joblib import Parallel, delayed

    name2i = {g: i for i, g in enumerate(genes)}
    tf_idx = [name2i[t] for t in tfs]

    def one(t):
        feats = [i for i in tf_idx if i != t]
        gb = GradientBoostingRegressor(
            n_estimators=n_estimators, learning_rate=lr, max_depth=max_depth,
            subsample=subsample, max_features=0.1, random_state=t,
        )
        gb.fit(X[:, feats], X[:, t])
        return [(genes[f], genes[t], float(w)) for f, w in zip(feats, gb.feature_importances_)]

    rows = Parallel(n_jobs=-1)(delayed(one)(t) for t in range(len(genes)))
    return [e for sub in rows for e in sub]


def validate_grnboost2(X, genes, tfs):
    from scipy.stats import spearmanr

    kw = dict(n_estimators=200, lr=0.05, max_depth=3, subsample=0.9)
    t0 = time.perf_counter()
    rust = scenic_rs.grnboost2(X, genes, tfs, n_estimators=kw["n_estimators"],
                               learning_rate=kw["lr"], max_depth=kw["max_depth"],
                               subsample=kw["subsample"], max_features="0.1", as_frame=False)
    t_rust = time.perf_counter() - t0
    rust_edges = dict(zip(zip(rust[0], rust[1]), rust[2]))

    t0 = time.perf_counter()
    skl = grnboost2_sklearn(X, genes, tfs, kw["n_estimators"], kw["lr"], kw["max_depth"], kw["subsample"])
    t_skl = time.perf_counter() - t0
    skl_edges = {(t, g): w for t, g, w in skl}

    keys = sorted(set(rust_edges) | set(skl_edges))
    va = np.array([rust_edges.get(k, 0.0) for k in keys])
    vb = np.array([skl_edges.get(k, 0.0) for k in keys])
    print("=== GRNBoost2 (scenic-rs vs sklearn GradientBoosting) ===")
    print(f"  Spearman(importance): {spearmanr(va, vb).correlation:.4f}")
    print(f"  scenic-rs (Rust): {t_rust:6.2f} s")
    print(f"  sklearn GBM:      {t_skl:6.2f} s")
    print(f"  speedup:          {t_skl / t_rust:6.1f}x\n")


if __name__ == "__main__":
    X, genes, tfs = load_pbmc3k(500, 300, 0)
    print(f"data: {X.shape[0]} cells x {len(genes)} genes ({len(tfs)} TFs)\n")
    validate_aucell(X, genes)
    validate_grnboost2(X, genes, tfs)
