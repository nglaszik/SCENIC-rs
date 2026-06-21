"""ctx (cisTarget) benchmark: scenic-rs vs real pySCENIC, equal cores.

Both run the FULL ctx step (modules_from_adjacencies -> motif enrichment -> prune
-> regulons) on the same adjacencies + expression + real hg38 ranking DB(s) +
motif2tf. We report wall-clock and whole-process-tree peak memory (PSS/RSS) per
worker count. scenic-rs uses one shared in-memory DB across rayon threads;
pySCENIC's custom_multiprocessing forks workers (the dask-style memory cost).

NB on correctness: scenic-rs ctx is validated separately (bench/validate_ctx_*.py)
and is in fact MORE correct than pySCENIC 0.12.1 (leading-edge bug, docs/ctx_spec.md);
this script measures speed/memory only.

    python bench/benchmark_ctx.py --sweep 4,16
"""
import argparse
import json
import os
import sys

import numpy as np

from benchmark_pyscenic import DEFAULT_PY
from mem_benchmark import profile  # (pss, rss, secs, rc, out)

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
RES = os.path.expanduser(
    "~/jupyternotebooks/paper_github_code/sc_rna_seq/scenic_out/resources")
DB1 = os.path.join(RES, "hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather")
TBL = os.path.join(RES, "motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl")


def prepare(n_tfs, n_targets, per_tf, n_cells, seed=0):
    """Build a realistic adjacency + expression over real DB genes / annotated TFs."""
    import scenic_rs._core as rs
    db = rs.RankingDb(DB1, "x")
    db_genes = set(db.genes)
    # annotated TFs present in the DB
    ann_tfs = []
    with open(TBL) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        gi = hdr.index("gene_name")
        seen = set()
        for line in f:
            g = line.split("\t")[gi]
            if g in db_genes and g not in seen:
                seen.add(g); ann_tfs.append(g)
    rng = np.random.default_rng(seed)
    tfs = sorted(rng.choice(ann_tfs, n_tfs, replace=False).tolist())
    targets = sorted(rng.choice(sorted(db_genes - set(tfs)), n_targets, replace=False).tolist())
    genes = sorted(set(tfs) | set(targets))

    tf_col, tgt_col, imp_col = [], [], []
    for tf in tfs:
        for t in rng.choice(targets, per_tf, replace=False):
            tf_col.append(str(tf)); tgt_col.append(str(t))
            imp_col.append(float(rng.exponential(1.0) + 1e-3))
    # cross-numpy-version-safe (pyscenic env is numpy<1.24): text + plain float .npy
    open(os.path.join(CACHE, "ctx_tf.txt"), "w").write("\n".join(tf_col) + "\n")
    open(os.path.join(CACHE, "ctx_target.txt"), "w").write("\n".join(tgt_col) + "\n")
    np.save(os.path.join(CACHE, "ctx_importance.npy"), np.array(imp_col, dtype="float64"))
    expr = rng.integers(0, 6, size=(n_cells, len(genes))).astype(np.float32)
    np.save(os.path.join(CACHE, "ctx_expr.npy"), expr)
    open(os.path.join(CACHE, "ctx_genes.txt"), "w").write("\n".join(genes) + "\n")
    return len(tf_col), len(genes), len(tfs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pyscenic-python", default=DEFAULT_PY)
    ap.add_argument("--sweep", default="4,16")
    ap.add_argument("--n-tfs", type=int, default=120)
    ap.add_argument("--n-targets", type=int, default=2000)
    ap.add_argument("--per-tf", type=int, default=50)
    ap.add_argument("--n-cells", type=int, default=500)
    ap.add_argument("--rank-threshold", type=int, default=5000)
    ap.add_argument("--nes", type=float, default=3.0)
    ap.add_argument("--dbs", nargs="+", default=[DB1])
    args = ap.parse_args()
    assert os.path.exists(args.pyscenic_python)

    workers = [int(w) for w in args.sweep.split(",")]
    n_edges, n_genes, n_tfs = prepare(args.n_tfs, args.n_targets, args.per_tf, args.n_cells)
    print(f"ctx inputs: {n_edges} edges, {n_genes} genes, {n_tfs} TFs, "
          f"{len(args.dbs)} DB(s)  |  cores = {workers}\n")

    MB = 1024 * 1024
    common = ["--dbs", *args.dbs, "--tbl", TBL,
              "--rank-threshold", str(args.rank_threshold), "--nes", str(args.nes)]
    rows = []
    for w in workers:
        impls = [
            ("scenic-rs", sys.executable, "_ctx_scenicrs.py",
             {**os.environ, "RAYON_NUM_THREADS": str(w)}),
            ("pySCENIC", args.pyscenic_python, "_ctx_pyscenic.py",
             {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
              "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}),
        ]
        for name, py, script, env in impls:
            print(f"== {name} / ctx / {w} cores ==")
            cmd = [py, os.path.join(HERE, script), "--workers", str(w), *common]
            pss, rss, dt, rc, out = profile(cmd, env)
            failed = rc != 0
            rows.append({"workers": w, "impl": name, "peak_pss_mb": pss / MB,
                         "peak_rss_mb": rss / MB, "secs": dt, "failed": failed})
            print(f"    peak PSS {pss/MB:8.1f} MB | peak RSS {rss/MB:8.1f} MB | "
                  f"{dt:7.1f}s{'  ** FAILED **' if failed else ''}\n")

    json.dump({"meta": {"edges": n_edges, "genes": n_genes, "tfs": n_tfs,
                        "dbs": len(args.dbs), "workers": workers}, "results": rows},
              open(os.path.join(CACHE, "ctx_bench_results.json"), "w"), indent=2)

    def get(impl, w):
        return next(r for r in rows if r["impl"] == impl and r["workers"] == w)

    print(f"{'cores':>6s} | {'scenic-rs PSS':>14s} {'time':>8s} | "
          f"{'pySCENIC PSS':>14s} {'time':>8s} | {'PSS x':>6s} {'time x':>7s}")
    for w in workers:
        sr, py = get("scenic-rs", w), get("pySCENIC", w)
        pr = py["peak_pss_mb"] / sr["peak_pss_mb"] if sr["peak_pss_mb"] else float("nan")
        tr = py["secs"] / sr["secs"] if sr["secs"] else float("nan")
        flag = "  pySCENIC FAILED" if py["failed"] else ""
        print(f"{w:6d} | {sr['peak_pss_mb']:11.1f} MB {sr['secs']:7.1f}s | "
              f"{py['peak_pss_mb']:11.1f} MB {py['secs']:7.1f}s | {pr:5.2f}x {tr:6.2f}x{flag}")


if __name__ == "__main__":
    main()
