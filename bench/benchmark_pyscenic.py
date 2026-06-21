"""Step-by-step benchmark: scenic-rs vs **real pySCENIC**, fully comparable.

For each SCENIC step (GRN/GRNBoost2, optional GRN/GENIE3, AUCell) we run both
implementations in-process on the *same* in-memory pbmc3k matrix and report:
  - correctness: rank-concordance (Spearman) of the outputs, plus top-edge
    overlap (GRN) and per-target Spearman
  - speed: wall-clock for each

Apples-to-apples by construction:
  * equal cores            - rayon (RAYON_NUM_THREADS) == pySCENIC dask workers
  * identical input        - both load the same .npz array; no CSV-IO on either
  * startup counted on both - dask cluster spin-up (pySCENIC) and rayon pool
                             init (scenic-rs) are inside the timed region
  * identical signatures   - AUCell regulons generated once here, shared via json

pySCENIC runs in a dedicated env (numpy<1.24); point --pyscenic-python at it.

    python bench/benchmark_pyscenic.py --workers 16
    python bench/benchmark_pyscenic.py --workers 16 --genie3
    python bench/plot_benchmark.py
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)

DEFAULT_PY = os.path.expanduser("~/venvs/pyscenic_clean/bin/python")


def prepare_inputs(npz, n_reg=100, reg_size=30, seed=0):
    """Export data to cross-numpy-version-safe files both envs can read.

    The prepped .npz stores gene/TF names as pickled object arrays, which a
    numpy<1.24 env (pySCENIC) cannot unpickle. So we re-emit: X as a plain
    float32 .npy (no pickle) + gene/TF lists as text + shared AUCell signatures.
    """
    z = np.load(os.path.join(DATA, npz), allow_pickle=True)
    X = np.ascontiguousarray(z["X"], dtype="float32")
    genes = list(map(str, z["genes"]))
    tfs = list(map(str, z["tfs"]))
    np.save(os.path.join(CACHE, "X.npy"), X)
    open(os.path.join(CACHE, "genes.txt"), "w").write("\n".join(genes) + "\n")
    open(os.path.join(CACHE, "tfs.txt"), "w").write("\n".join(tfs) + "\n")
    rng = np.random.default_rng(seed)
    sigs = {f"reg{i}": list(map(str, rng.choice(genes, size=reg_size, replace=False)))
            for i in range(n_reg)}
    json.dump(sigs, open(os.path.join(CACHE, "signatures.json"), "w"))
    return X.shape[0], len(genes), len(tfs)


def concordance_grn(method):
    from scipy.stats import spearmanr

    rs = pd.read_csv(os.path.join(CACHE, f"adj_scenicrs_{method}.csv"))
    py = pd.read_csv(os.path.join(CACHE, f"adj_pyscenic_{method}.csv"))
    m = rs.merge(py, on=["TF", "target"], suffixes=("_rs", "_py"))
    overall = spearmanr(m["importance_rs"], m["importance_py"]).correlation

    per = [spearmanr(g["importance_rs"], g["importance_py"]).correlation
           for _, g in m.groupby("target") if len(g) >= 10]
    per = np.array(per, dtype=float)

    def jac(k):
        a = set(map(tuple, rs.nlargest(k, "importance")[["TF", "target"]].values))
        b = set(map(tuple, py.nlargest(k, "importance")[["TF", "target"]].values))
        return len(a & b) / len(a | b)

    t_rs = json.load(open(os.path.join(CACHE, f"time_scenicrs_{method}.json")))["seconds"]
    t_py = json.load(open(os.path.join(CACHE, f"time_pyscenic_{method}.json")))["seconds"]
    return {
        "step": f"GRN/{method}",
        "t_scenicrs": t_rs, "t_pyscenic": t_py, "speedup": t_py / t_rs,
        "n_shared_edges": int(len(m)),
        "overall_spearman": float(overall),
        "per_target_spearman_median": float(np.nanmedian(per)),
        "per_target_spearman": per[np.isfinite(per)].tolist(),
        "topk_jaccard": {str(k): float(jac(k)) for k in (100, 500, 1000, 5000)},
        "scatter": m.sample(min(20000, len(m)), random_state=0)[
            ["importance_rs", "importance_py"]].values.tolist(),
    }


def concordance_aucell():
    from scipy.stats import spearmanr, pearsonr

    rs = pd.read_csv(os.path.join(CACHE, "auc_scenicrs.csv"), index_col=0)
    py = pd.read_csv(os.path.join(CACHE, "auc_pyscenic.csv"), index_col=0)
    py.columns = [c.replace("(+)", "").strip() for c in py.columns]
    cols = [c for c in rs.columns if c in py.columns]
    py = py.reindex(index=rs.index)                      # align cells
    A = rs[cols].to_numpy().ravel()
    B = py[cols].to_numpy().ravel()
    ok = np.isfinite(A) & np.isfinite(B)
    A, B = A[ok], B[ok]

    t_rs = json.load(open(os.path.join(CACHE, "time_scenicrs_aucell.json")))["seconds"]
    t_py = json.load(open(os.path.join(CACHE, "time_pyscenic_aucell.json")))["seconds"]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(A), min(20000, len(A)), replace=False)
    return {
        "step": "AUCell",
        "t_scenicrs": t_rs, "t_pyscenic": t_py, "speedup": t_py / t_rs,
        "n_regulons": len(cols),
        "overall_spearman": float(spearmanr(A, B).correlation),
        "overall_pearson": float(pearsonr(A, B)[0]),
        "max_abs_diff": float(np.max(np.abs(A - B))),
        "scatter": np.c_[A, B][idx].tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="pbmc3k_prep_500_300_0.npz")
    ap.add_argument("--pyscenic-python", default=DEFAULT_PY)
    ap.add_argument("--workers", type=int, default=16,
                    help="parallel units for BOTH impls (rayon threads == dask workers)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--genie3", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    args = ap.parse_args()
    assert os.path.exists(args.pyscenic_python), f"not found: {args.pyscenic_python}"

    n_cells, n_genes, n_tfs = prepare_inputs(args.npz)
    print(f"data: {n_cells} cells x {n_genes} genes ({n_tfs} TFs)   "
          f"|   equal cores = {args.workers}\n")

    common = ["--workers", str(args.workers), "--seed", str(args.seed)]
    if args.genie3:
        common.append("--genie3")
    if args.force:
        common.append("--force")

    # scenic-rs: pin rayon to the same core budget (must be set before the process starts)
    env_rs = {**os.environ, "RAYON_NUM_THREADS": str(args.workers)}
    print("== scenic-rs ==")
    subprocess.run([sys.executable, os.path.join(HERE, "_steps_scenicrs.py"), *common],
                   check=True, env=env_rs)

    # pySCENIC: single-thread BLAS per dask worker to avoid oversubscription
    env_py = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
              "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    print("== pySCENIC ==")
    subprocess.run([args.pyscenic_python, os.path.join(HERE, "_steps_pyscenic.py"), *common],
                   check=True, env=env_py)

    results = [concordance_grn("grnboost2")]
    if args.genie3:
        results.append(concordance_grn("genie3"))
    results.append(concordance_aucell())

    json.dump({"meta": {"cells": n_cells, "genes": n_genes,
                        "tfs": n_tfs, "workers": args.workers},
               "results": results},
              open(os.path.join(CACHE, "results.json"), "w"), indent=2)

    print("\nstep              pySCENIC   scenic-rs   speedup   Spearman")
    for r in results:
        print(f"{r['step']:16s}  {r['t_pyscenic']:7.1f}s  {r['t_scenicrs']:8.2f}s   "
              f"{r['speedup']:6.1f}×   {r['overall_spearman']:.3f}")
    print(f"\nwrote {os.path.join(CACHE, 'results.json')}  ->  run plot_benchmark.py")


if __name__ == "__main__":
    main()
