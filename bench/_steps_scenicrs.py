"""scenic-rs worker: run the matching GRN + AUCell steps in-process.

Run with the main env's python. rayon is pinned to RAYON_NUM_THREADS (set by the
orchestrator to equal pySCENIC's worker count). Data is loaded from the same .npz
before timing; the timed region includes rayon's pool spin-up (negligible, but
counted for symmetry with pySCENIC's cluster startup).

Params are matched to arboreto's defaults: grnboost2 = stochastic GBM
(lr 0.01, depth 3, subsample 0.9, max_features sqrt); genie3 = 1000-tree RF.

Invoked by benchmark_pyscenic.py; not meant to be run by hand.
"""
import argparse
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
CACHE = os.path.join(HERE, "cache")


def load():
    X = np.load(os.path.join(CACHE, "X.npy"))      # same float32 array both envs read
    genes = open(os.path.join(CACHE, "genes.txt")).read().split()
    tfs = open(os.path.join(CACHE, "tfs.txt")).read().split()
    return X, genes, tfs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, required=True)   # informational; rayon reads env
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--genie3", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import scenic_rs
    X, genes, tfs = load()

    def cached(name):
        return (not args.force) and os.path.exists(os.path.join(CACHE, name))

    # ---- GRN / GRNBoost2 ----
    if cached("adj_scenicrs_grnboost2.csv"):
        print("  scenic_rs.grnboost2: cached")
    else:
        t0 = time.perf_counter()
        # arboreto GRNBoost2 config: lr 0.01, max_features 0.1, subsample 0.9,
        # large tree cap + OOB early stopping (window 25)
        adj = scenic_rs.grnboost2(X, genes, tfs, n_estimators=5000, learning_rate=0.01,
                                  max_depth=3, subsample=0.9, max_features="0.1",
                                  early_stop_window=25, seed=args.seed)
        dt = time.perf_counter() - t0
        adj.to_csv(os.path.join(CACHE, "adj_scenicrs_grnboost2.csv"), index=False)
        json.dump({"seconds": dt}, open(os.path.join(CACHE, "time_scenicrs_grnboost2.json"), "w"))
        print(f"  scenic_rs.grnboost2: {dt:.2f}s ({len(adj)} edges)")

    # ---- GRN / GENIE3 ----
    if args.genie3:
        if cached("adj_scenicrs_genie3.csv"):
            print("  scenic_rs.genie3: cached")
        else:
            t0 = time.perf_counter()
            adj = scenic_rs.genie3(X, genes, tfs, n_estimators=1000,
                                   max_features="sqrt", seed=args.seed)
            dt = time.perf_counter() - t0
            adj.to_csv(os.path.join(CACHE, "adj_scenicrs_genie3.csv"), index=False)
            json.dump({"seconds": dt}, open(os.path.join(CACHE, "time_scenicrs_genie3.json"), "w"))
            print(f"  scenic_rs.genie3: {dt:.2f}s ({len(adj)} edges)")

    # ---- AUCell ----
    if cached("auc_scenicrs.csv"):
        print("  scenic_rs.aucell: cached")
    else:
        sig_def = json.load(open(os.path.join(CACHE, "signatures.json")))
        cells = [f"cell{i}" for i in range(X.shape[0])]
        amr = max(1, int(np.ceil(0.05 * len(genes))))     # match pyscenic auc_threshold 0.05
        t0 = time.perf_counter()
        auc = scenic_rs.aucell(X, genes, sig_def, auc_max_rank=amr)
        dt = time.perf_counter() - t0
        auc.index = cells
        auc.to_csv(os.path.join(CACHE, "auc_scenicrs.csv"))
        json.dump({"seconds": dt}, open(os.path.join(CACHE, "time_scenicrs_aucell.json"), "w"))
        print(f"  scenic_rs.aucell: {dt:.3f}s ({auc.shape[1]} regulons)")


if __name__ == "__main__":
    main()
