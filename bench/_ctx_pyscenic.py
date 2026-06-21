"""pySCENIC ctx worker: the real cisTarget step via prune2df (custom_multiprocessing)
+ df2regulons. Run with the pyscenic env's python. Timed region includes DB
construction + the multiprocessing pool startup (parallels scenic-rs DB load).
Invoked by benchmark_ctx.py.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--dbs", nargs="+", required=True)
    ap.add_argument("--tbl", required=True)
    ap.add_argument("--rank-threshold", type=int, default=5000)
    ap.add_argument("--nes", type=float, default=3.0)
    args = ap.parse_args()

    from pyscenic.utils import modules_from_adjacencies
    from pyscenic.prune import prune2df, df2regulons
    from ctxcore.rnkdb import FeatherRankingDatabase

    tf = open(os.path.join(CACHE, "ctx_tf.txt")).read().split()
    target = open(os.path.join(CACHE, "ctx_target.txt")).read().split()
    importance = np.load(os.path.join(CACHE, "ctx_importance.npy"))
    adj = pd.DataFrame({"TF": tf, "target": target, "importance": importance})
    expr = np.load(os.path.join(CACHE, "ctx_expr.npy"))
    genes = open(os.path.join(CACHE, "ctx_genes.txt")).read().split()
    ex = pd.DataFrame(expr, index=[f"c{i}" for i in range(expr.shape[0])], columns=genes)

    t0 = time.perf_counter()
    dbs = [FeatherRankingDatabase(p, name=os.path.basename(p)) for p in args.dbs]
    modules = modules_from_adjacencies(adj, ex, rho_mask_dropouts=False,
                                       keep_only_activating=True)
    df = prune2df(dbs, modules, args.tbl, rank_threshold=args.rank_threshold,
                  auc_threshold=0.05, nes_threshold=args.nes,
                  client_or_address="custom_multiprocessing", num_workers=args.workers)
    regs = df2regulons(df) if len(df) else []
    dt = time.perf_counter() - t0
    json.dump({"seconds": dt, "n_regulons": len(regs)},
              open(os.path.join(CACHE, "time_ctx_pyscenic.json"), "w"))
    print(f"  pyscenic ctx: {dt:.2f}s ({len(regs)} regulons)")


if __name__ == "__main__":
    main()
