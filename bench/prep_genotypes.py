"""Build a combined WT+TKO+DKO benchmark matrix from the paper's h5ad, matching
the pySCENIC GRN notebook's *semantics* (raw UMI counts, full TF list, all cells
pooled across genotype) but with one sensible deviation: a low-expression gene
filter (standard SCENIC practice; the notebook skips it).

Output is a single .npz in data/ that benchmark_pyscenic.py / mem_benchmark.py
can consume via --npz: X (cells x genes, float32 raw counts), genes, tfs.

Run with an env that has scanpy (e.g. ~/venvs/paper/bin/python).

    ~/venvs/paper/bin/python bench/prep_genotypes.py                 # full combined
    ~/venvs/paper/bin/python bench/prep_genotypes.py --per-geno-cap 800 --out combined_sanity.npz
"""
import argparse
import os

import numpy as np
import scanpy as sc

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

H5AD = "/home/data/nlaszik/h5ad/scvi_integrated_knockout_wt_tko_dko.h5ad"
TF_LIST = os.path.expanduser(
    "~/jupyternotebooks/paper_github_code/sc_rna_seq/scenic_out/resources/allTFs_hg38.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default=H5AD)
    ap.add_argument("--tf-list", default=TF_LIST)
    ap.add_argument("--out", default="combined_full.npz")
    ap.add_argument("--per-geno-cap", type=int, default=0,
                    help="subsample to at most N cells per genotype (0 = use all)")
    ap.add_argument("--min-cells", type=int, default=10,
                    help="keep genes detected (count>0) in >= this many cells")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    a = sc.read_h5ad(args.h5ad, backed="r")
    geno = a.obs["sample_name"].astype(str).str.split().str[0]   # 'WT Rep1' -> 'WT'

    # cell selection: pool all genotypes (optionally cap per genotype, like the notebook)
    rng = np.random.default_rng(args.seed)
    if args.per_geno_cap > 0:
        pos = np.arange(a.n_obs)
        keep_pos = []
        for g in sorted(geno.unique()):
            gp = pos[(geno == g).values]
            if len(gp) > args.per_geno_cap:
                gp = rng.choice(gp, args.per_geno_cap, replace=False)
            keep_pos.extend(gp.tolist())
        keep_pos = np.sort(np.array(keep_pos))
    else:
        keep_pos = np.arange(a.n_obs)

    sub = a[keep_pos]
    C = sub.layers["counts"]                      # cells x genes, raw UMI (sparse)
    C = C.tocsr() if hasattr(C, "tocsr") else C
    genotypes = geno.values[keep_pos]
    print(f"selected {C.shape[0]} cells "
          f"({dict(zip(*np.unique(genotypes, return_counts=True)))})")

    # gene filter: detected (count>0) in >= min_cells cells (standard SCENIC)
    min_cells = args.min_cells
    if hasattr(C, "getnnz"):
        det = np.asarray((C > 0).sum(axis=0)).ravel()
    else:
        det = (np.asarray(C) > 0).sum(axis=0)
    keep_g = det >= min_cells
    all_genes = np.asarray(sub.var_names)
    genes = all_genes[keep_g]
    print(f"genes: {len(all_genes)} -> {len(genes)} kept "
          f"(detected in >= {min_cells} cells)")

    # densify kept genes only, cells x genes, float32 raw counts
    Csub = C[:, keep_g]
    X = np.ascontiguousarray(
        Csub.toarray() if hasattr(Csub, "toarray") else np.asarray(Csub), dtype="float32")

    # TFs = notebook's full human TF list intersected with kept genes
    tfset = set(l.strip() for l in open(args.tf_list) if l.strip())
    geneset = set(genes.tolist())
    tfs = [g for g in genes.tolist() if g in tfset]
    print(f"TFs: {len(tfset)} in list -> {len(tfs)} present in kept genes")

    out = os.path.join(DATA, args.out)
    np.savez_compressed(out, X=X,
                        genes=np.array(genes.tolist(), dtype=object),
                        tfs=np.array(tfs, dtype=object))
    print(f"wrote {out}  ({X.shape[0]} cells x {X.shape[1]} genes, {len(tfs)} TFs, "
          f"{X.nbytes/1e9:.2f} GB dense)")


if __name__ == "__main__":
    main()
