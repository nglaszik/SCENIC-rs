"""Validate + benchmark scenic_rs.genie3 against the scikit-learn RandomForest
GENIE3 reference (the exact algorithm arboreto runs, minus the Dask wrapper).

Both methods get identical input and matched params. We report:
  - concordance: Spearman/Pearson of importances over all (TF, target) edges,
    and top-k edge overlap (Jaccard)
  - speed: wall-clock for each, both parallelized over target genes across cores

Data: 10x pbmc3k (downloaded to data/). Scale up with --n-genes / --n-trees / --cells.
"""
import argparse
import os
import tarfile
import time

import numpy as np
import scipy.io
import scipy.sparse as sp
from scipy.stats import spearmanr, pearsonr

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def load_pbmc3k(n_genes, n_tfs, cells, seed=0):
    cache = os.path.join(DATA, f"pbmc3k_prep_{n_genes}_{n_tfs}_{cells}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["X"].astype("float32"), list(z["genes"]), list(z["tfs"])

    base = os.path.join(DATA, "filtered_gene_bc_matrices", "hg19")
    if not os.path.exists(base):
        with tarfile.open(os.path.join(DATA, "pbmc3k.tar.gz")) as t:
            t.extractall(DATA)

    M = scipy.io.mmread(os.path.join(base, "matrix.mtx")).tocsr()          # genes x cells
    symbols = [l.split("\t")[1].strip() for l in open(os.path.join(base, "genes.tsv"))]
    X = sp.csr_matrix(M).T.tocsr().astype("float64")                       # cells x genes

    # dedupe gene symbols (keep first)
    seen, keep = set(), []
    for i, s in enumerate(symbols):
        if s not in seen:
            seen.add(s); keep.append(i)
    X = X[:, keep]; symbols = [symbols[i] for i in keep]

    # filter genes expressed in >= 3 cells
    expr_cells = np.asarray((X > 0).sum(axis=0)).ravel()
    g = expr_cells >= 3
    X = X[:, g]; symbols = [s for s, k in zip(symbols, g) if k]

    # library-size normalize to 1e4 + log1p
    lib = np.asarray(X.sum(axis=1)).ravel(); lib[lib == 0] = 1
    X = X.multiply(1e4 / lib[:, None]).tocsr()
    X = X.copy(); X.data = np.log1p(X.data)
    Xd = np.asarray(X.todense(), dtype="float32")

    # optional cell subsample
    rng = np.random.default_rng(seed)
    if cells and cells < Xd.shape[0]:
        sel = rng.choice(Xd.shape[0], cells, replace=False)
        Xd = Xd[sel]

    # select genes: top-variance targets UNION expressed TFs
    tfset = set(l.strip() for l in open(os.path.join(DATA, "hs_tfs.txt")))
    var = Xd.var(axis=0)
    top_var = np.argsort(var)[::-1][:n_genes]
    tf_idx = [i for i, s in enumerate(symbols) if s in tfset]
    tf_idx = sorted(tf_idx, key=lambda i: -var[i])[:n_tfs]
    used = sorted(set(top_var.tolist()) | set(tf_idx))

    Xu = np.ascontiguousarray(Xd[:, used], dtype="float32")
    genes = [symbols[i] for i in used]
    tfs = [g for g in genes if g in tfset]
    np.savez_compressed(cache, X=Xu, genes=np.array(genes, dtype=object),
                        tfs=np.array(tfs, dtype=object))
    return Xu, genes, tfs


def genie3_sklearn(X, genes, tfs, n_estimators):
    from sklearn.ensemble import RandomForestRegressor
    from joblib import Parallel, delayed

    name2i = {g: i for i, g in enumerate(genes)}
    tf_idx = [name2i[t] for t in tfs]

    def one_target(t):
        feats = [i for i in tf_idx if i != t]
        rf = RandomForestRegressor(n_estimators=n_estimators, max_features="sqrt",
                                   min_samples_leaf=1, n_jobs=1, random_state=t)
        rf.fit(X[:, feats], X[:, t])
        return [(genes[f], genes[t], float(w)) for f, w in zip(feats, rf.feature_importances_)]

    rows = Parallel(n_jobs=-1)(delayed(one_target)(t) for t in range(len(genes)))
    return [e for sub in rows for e in sub]


def concordance(a_edges, b_edges, topk=1000):
    da = {(t, g): w for t, g, w in a_edges}
    db = {(t, g): w for t, g, w in b_edges}
    keys = sorted(set(da) | set(db))
    va = np.array([da.get(k, 0.0) for k in keys])
    vb = np.array([db.get(k, 0.0) for k in keys])
    sp_r = spearmanr(va, vb).correlation
    pe_r = pearsonr(va, vb)[0]
    ta = set(sorted(da, key=lambda k: -da[k])[:topk])
    tb = set(sorted(db, key=lambda k: -db[k])[:topk])
    jac = len(ta & tb) / len(ta | tb)
    return sp_r, pe_r, jac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-genes", type=int, default=500, help="top-variance target genes")
    ap.add_argument("--n-tfs", type=int, default=300, help="max expressed TFs as regulators")
    ap.add_argument("--cells", type=int, default=0, help="0 = all cells")
    ap.add_argument("--n-trees", type=int, default=100)
    ap.add_argument("--topk", type=int, default=1000)
    args = ap.parse_args()

    X, genes, tfs = load_pbmc3k(args.n_genes, args.n_tfs, args.cells)
    print(f"data: {X.shape[0]} cells x {len(genes)} genes ({len(tfs)} TFs), {args.n_trees} trees")

    import scenic_rs
    t0 = time.perf_counter()
    rust = scenic_rs.genie3(X, genes, tfs, n_estimators=args.n_trees, as_frame=False)
    t_rust = time.perf_counter() - t0
    rust_edges = list(zip(rust[0], rust[1], rust[2]))

    t0 = time.perf_counter()
    skl = genie3_sklearn(X, genes, tfs, args.n_trees)
    t_skl = time.perf_counter() - t0

    sp_r, pe_r, jac = concordance(rust_edges, skl, args.topk)
    print(f"\n--- concordance (scenic-rs vs sklearn-RF GENIE3) ---")
    print(f"  Spearman(importance): {sp_r:.4f}")
    print(f"  Pearson(importance):  {pe_r:.4f}")
    print(f"  top-{args.topk} edge Jaccard: {jac:.4f}")
    print(f"\n--- speed (both parallel over targets, all cores) ---")
    print(f"  scenic-rs (Rust): {t_rust:6.2f} s")
    print(f"  sklearn-RF:       {t_skl:6.2f} s")
    print(f"  speedup:          {t_skl / t_rust:6.1f}x")


if __name__ == "__main__":
    main()
