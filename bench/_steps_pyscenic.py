"""pySCENIC-env worker: run GRN (arboreto) + AUCell (ctxcore) steps in-process.

Run with the pyscenic env's python (numpy<1.24). Timed region INCLUDES the dask
cluster / worker-pool startup, because that overhead is part of the real cost of
running pySCENIC. Data is loaded from the same .npz before timing, so there is no
CSV-IO asymmetry vs the scenic-rs side. Workers are pinned to --workers.

Invoked by benchmark_pyscenic.py; not meant to be run by hand.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
CACHE = os.path.join(HERE, "cache")


def load():
    X = np.load(os.path.join(CACHE, "X.npy"))      # plain float32, no pickle
    genes = open(os.path.join(CACHE, "genes.txt")).read().split()
    tfs = open(os.path.join(CACHE, "tfs.txt")).read().split()
    cells = [f"cell{i}" for i in range(X.shape[0])]
    df = pd.DataFrame(X, index=cells, columns=genes)
    return df, genes, tfs


def grn(method, df, tfs, n_workers, seed, force):
    from arboreto.algo import grnboost2, genie3
    from distributed import LocalCluster, Client

    out = os.path.join(CACHE, f"adj_pyscenic_{method}.csv")
    if not force and os.path.exists(out):
        print(f"  pyscenic {method}: cached")
        return

    t0 = time.perf_counter()
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1, processes=True,
                           dashboard_address=None, memory_limit=0)
    client = Client(cluster)
    fn = grnboost2 if method == "grnboost2" else genie3
    net = fn(expression_data=df, tf_names=tfs, client_or_address=client,
             seed=seed, verbose=False)
    dt = time.perf_counter() - t0
    client.close(); cluster.close()

    net.columns = ["TF", "target", "importance"]
    net.to_csv(out, index=False)
    json.dump({"seconds": dt}, open(os.path.join(CACHE, f"time_pyscenic_{method}.json"), "w"))
    print(f"  pyscenic {method}: {dt:.1f}s ({len(net)} edges)")


def aucell_step(df, n_workers, seed, force):
    from pyscenic.aucell import aucell
    from ctxcore.genesig import GeneSignature

    out = os.path.join(CACHE, "auc_pyscenic.csv")
    if not force and os.path.exists(out):
        print("  pyscenic aucell: cached")
        return

    sig_def = json.load(open(os.path.join(CACHE, "signatures.json")))
    sigs = [GeneSignature(name=n, gene2weight={g: 1.0 for g in gs}) for n, gs in sig_def.items()]

    t0 = time.perf_counter()
    auc = aucell(df, sigs, auc_threshold=0.05, num_workers=n_workers, seed=seed)
    dt = time.perf_counter() - t0

    auc.to_csv(out)
    json.dump({"seconds": dt}, open(os.path.join(CACHE, "time_pyscenic_aucell.json"), "w"))
    print(f"  pyscenic aucell: {dt:.2f}s ({auc.shape[1]} regulons)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--genie3", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    df, genes, tfs = load()
    grn("grnboost2", df, tfs, args.workers, args.seed, args.force)
    if args.genie3:
        grn("genie3", df, tfs, args.workers, args.seed, args.force)
    aucell_step(df, args.workers, args.seed, args.force)


if __name__ == "__main__":
    main()
