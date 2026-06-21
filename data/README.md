# data/

Benchmark / example inputs for scenic-rs.

## Tracked (small, committed)

- `hs_tfs.txt` — human transcription-factor list (regulator candidates).
- `pbmc3k_prep_*.npz` — prepped 10x **pbmc3k** matrices used by the benchmarks,
  named `pbmc3k_prep_{n_genes}_{n_tfs}_{cells}.npz`. Each holds `X`
  (cells × genes, log1p CPM-10k, float32), `genes`, and `tfs`. Produced by
  `bench/validate_genie3.py::load_pbmc3k`.

## Not tracked (large / regenerable — see .gitignore)

- `pbmc3k.tar.gz` and `filtered_gene_bc_matrices/` — the raw 10x pbmc3k download
  (3k PBMCs, hg19). Get it from 10x Genomics:
  https://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz
  Place the tarball here as `pbmc3k.tar.gz`; `load_pbmc3k` extracts + preps it,
  regenerating the `.npz` files above.
