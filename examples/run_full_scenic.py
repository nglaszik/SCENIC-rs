#!/usr/bin/env python
"""Full scenic-rs run on an h5ad — grnboost2 -> ctx (cisTarget) -> AUCell.

No Dask, no subsampling: runs on ALL cells. Cell-level QC is assumed already
done (this loads a QC'd h5ad); the only filtering applied is the SCENIC gene
filter (genes detected in >= --min-cells cells, SCENICprotocol default 3).

Defaults target the TKO/DKO/WT dataset + the cisTarget resources from the paper
notebook, but everything is overridable. Outputs (to --out):
  adjacencies.csv   TF, target, importance         (GRN)
  regulons.csv      name, tf, activating, nes, n_targets, targets   (ctx)
  regulons.json     {regulon_name: [target genes]}
  aucell.csv        cells x regulons activity       (AUCell; join to obs by barcode)

Run via run_scenic.sbatch, or directly with the scenic-rs python:
  RAYON_NUM_THREADS=64 ~/venvs/base/bin/python examples/run_full_scenic.py
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import anndata as ad

import scenic_rs

RES = os.path.expanduser("~/jupyternotebooks/paper_github_code/sc_rna_seq/scenic_out/resources")
DB_FILES = [
    "hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather",
    "hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather",
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default="/home/data/nlaszik/h5ad/scvi_integrated_knockout_wt_tko_dko.h5ad")
    ap.add_argument("--layer", default="counts",
                    help="raw-counts layer name (falls back to adata.X if absent)")
    ap.add_argument("--resources", default=RES, help="dir with the cisTarget DBs, motif tbl, allTFs list")
    ap.add_argument("--tf-list", default="allTFs_hg38.txt")
    ap.add_argument("--motif-tbl", default="motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl")
    ap.add_argument("--out", default=os.path.expanduser(
        "~/jupyternotebooks/paper_github_code/sc_rna_seq/scenic_out/scenicrs"))
    ap.add_argument("--min-cells", type=int, default=3,
                    help="gene filter: keep genes detected in >= this many cells (SCENICprotocol default 3)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mask-dropouts", action="store_true",
                    help="exclude dropout (zero) cells when computing TF-target correlation. "
                         "The original notebook used this (`pyscenic ctx --mask_dropouts`); the "
                         "modern SCENIC default (and scenic-rs default) is OFF.")
    ap.add_argument("--rank-threshold", type=int, default=5000)
    ap.add_argument("--auc-threshold", type=float, default=0.05)
    ap.add_argument("--nes-threshold", type=float, default=3.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.perf_counter()

    # ---- load (all cells) ----
    log(f"loading {args.h5ad}")
    adata = ad.read_h5ad(args.h5ad)
    Xsrc = adata.layers[args.layer] if args.layer in adata.layers else adata.X
    X = np.asarray(Xsrc.todense() if hasattr(Xsrc, "todense") else Xsrc, dtype=np.float32)  # cells x genes
    gene_names = list(map(str, adata.var_names))
    cells = list(map(str, adata.obs_names))
    del adata  # free the AnnData; we only need X / names / barcodes
    log(f"loaded {X.shape[0]} cells x {X.shape[1]} genes "
        f"(max count {X.max():.0f}; integer-valued: {np.allclose(X, np.round(X))})")

    # ---- gene filter (cell QC assumed already done) ----
    det = (X > 0).sum(axis=0)
    keep = det >= args.min_cells
    X = np.ascontiguousarray(X[:, keep])
    gene_names = [g for g, k in zip(gene_names, keep) if k]
    log(f"gene filter (>= {args.min_cells} cells): {int(keep.sum())} of {len(keep)} genes kept")

    # ---- TF list ----
    tfset = set(open(os.path.join(args.resources, args.tf_list)).read().split())
    tf_names = [g for g in gene_names if g in tfset]
    log(f"TFs present: {len(tf_names)}")

    # ---- 1. GRN / GRNBoost2 ----
    t = time.perf_counter()
    adj = scenic_rs.grnboost2(X, gene_names, tf_names, seed=args.seed)
    log(f"GRN: {len(adj)} edges in {time.perf_counter()-t:.0f}s")
    adj.to_csv(os.path.join(args.out, "adjacencies.csv"), index=False)

    # ---- 2. ctx / cisTarget (both DBs) ----
    dbs = [scenic_rs.RankingDb(os.path.join(args.resources, f), f) for f in DB_FILES]
    tbl = os.path.join(args.resources, args.motif_tbl)
    t = time.perf_counter()
    regulons = scenic_rs.ctx(adj, X, gene_names, dbs, tbl,
                             rank_threshold=args.rank_threshold, auc_threshold=args.auc_threshold,
                             nes_threshold=args.nes_threshold, mask_dropouts=args.mask_dropouts)
    log(f"ctx: {len(regulons)} regulons in {time.perf_counter()-t:.0f}s")
    pd.DataFrame([{"name": r.name, "tf": r.tf, "activating": r.activating, "nes": r.nes,
                   "n_targets": len(r.genes), "targets": ";".join(r.genes)} for r in regulons]
                 ).to_csv(os.path.join(args.out, "regulons.csv"), index=False)
    json.dump({r.name: list(r.genes) for r in regulons},
              open(os.path.join(args.out, "regulons.json"), "w"))
    if not regulons:
        log("no regulons passed the NES/annotation filter — stopping before AUCell")
        return

    # ---- 3. AUCell ----
    t = time.perf_counter()
    auc = scenic_rs.aucell(X, gene_names, {r.name: list(r.genes) for r in regulons})
    auc.index = cells
    auc.to_csv(os.path.join(args.out, "aucell.csv"))
    log(f"AUCell: {auc.shape[1]} regulons x {auc.shape[0]} cells in {time.perf_counter()-t:.0f}s")

    log(f"done in {time.perf_counter()-t0:.0f}s — outputs in {args.out}")


if __name__ == "__main__":
    main()
