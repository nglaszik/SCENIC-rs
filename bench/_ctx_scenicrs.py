"""scenic-rs ctx worker: load DB(s) + run the full cisTarget step in-process.

rayon is pinned via RAYON_NUM_THREADS (set by the orchestrator). DB construction
is inside the timed region (parallels pySCENIC's dask startup being counted).
Invoked by benchmark_ctx.py.
"""
import argparse
import json
import os
import time

import numpy as np

import scenic_rs._core as rs

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, required=True)   # informational; rayon reads env
    ap.add_argument("--dbs", nargs="+", required=True)
    ap.add_argument("--tbl", required=True)
    ap.add_argument("--rank-threshold", type=int, default=5000)
    ap.add_argument("--nes", type=float, default=3.0)
    args = ap.parse_args()

    tf = open(os.path.join(CACHE, "ctx_tf.txt")).read().split()
    target = open(os.path.join(CACHE, "ctx_target.txt")).read().split()
    importance = np.load(os.path.join(CACHE, "ctx_importance.npy")).tolist()
    expr = np.load(os.path.join(CACHE, "ctx_expr.npy"))
    genes = open(os.path.join(CACHE, "ctx_genes.txt")).read().split()

    t0 = time.perf_counter()
    dbs = [rs.RankingDb(p, os.path.basename(p)) for p in args.dbs]
    regs = rs.ctx(tf, target, importance, expr, genes, dbs, args.tbl,
                  None, None, None, 20, 0.03, False, True,
                  args.rank_threshold, 0.05, args.nes, 0.001, 0.0)
    dt = time.perf_counter() - t0
    json.dump({"seconds": dt, "n_regulons": len(regs)},
              open(os.path.join(CACHE, "time_ctx_scenicrs.json"), "w"))
    print(f"  scenic-rs ctx: {dt:.2f}s ({len(regs)} regulons)")


if __name__ == "__main__":
    main()
