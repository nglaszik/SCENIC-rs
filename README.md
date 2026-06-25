# scenic-rs

[![CI](https://github.com/nglaszik/SCENIC-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/nglaszik/SCENIC-rs/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/scenic-rs.svg)](https://pypi.org/project/scenic-rs/)
[![Python](https://img.shields.io/pypi/pyversions/scenic-rs.svg)](https://pypi.org/project/scenic-rs/)

A memory-efficient Rust implementation of the [pySCENIC](https://github.com/aertslab/pySCENIC)
single-cell gene regulatory network (GRN) pipeline. Not associated with the lab, just built this
as a quick project!

## Benefits

- Implements the same algorithms as pySCENIC (GRNBoost2, GENIE3, AUCell, CTX trimming)
- Can use modern numpy/pandas in your environment
- Memory usage is constant with increased parallelism, allowing for faster execution without OOM

## Requirements

**Runtime** (using scenic-rs)
- Python ≥ 3.9
- `numpy`, `pandas`
- `scanpy` (optional) — only if you use it to load your data (Cell Ranger / `.h5ad`);
  scenic-rs itself just takes a numpy matrix + gene/TF names
- For the **`ctx`** step only: the cisTarget **ranking databases**
  (`*.genes_vs_motifs.rankings.feather`) and the **motif2TF annotations** (`.tbl`),
  from [resources.aertslab.org/cistarget](https://resources.aertslab.org/cistarget/)

**From source** (optional — only for development; `pip install` uses prebuilt wheels)
- A recent stable **Rust** toolchain (`cargo`/`rustc`) — ≥ 1.70 (tested on 1.86);
  install via [rustup.rs](https://rustup.rs).
- **maturin** ≥ 1.0 (`pip install maturin`), then `maturin develop --release`

**Benchmarks / validation only** (optional)
- A separate pySCENIC environment to compare against:
  `python3.10 -m venv ~/venvs/pyscenic_clean && ~/venvs/pyscenic_clean/bin/pip install "numpy<1.24" "pandas<2" pyscenic`
- `matplotlib` + `scipy` for the plotting and validation scripts

## Install

```bash
pip install scenic-rs   # prebuilt wheels: Linux / macOS / Windows, Python >=3.9
```

scenic-rs depends only on (unpinned) `numpy` and `pandas`, so it drops into an
existing environment — including a `scanpy` env — without forcing a downgrade.

**conda** — a ready-made environment that includes scanpy:

```bash
conda env create -f environment.yml   # then: conda activate scenic-rs
```

**Docker / Nextflow / Singularity** — a version-pinned image is published per
release for reproducible pipelines:

```bash
docker pull ghcr.io/nglaszik/scenic-rs:latest      # or a pinned tag, e.g. :0.1.1
```

```groovy
// In a Nextflow process, just point at the image:
process scenic {
    container 'ghcr.io/nglaszik/scenic-rs:0.1.1'
    // ... your script that `import scenic_rs`
}
```

## Usage

scenic-rs takes a **cells × genes** `float32` matrix plus the gene names and a TF
list — load these however you like (e.g. with `scanpy`). Raw counts; do your own
cell/gene QC first. From Cell Ranger output:

```python
import numpy as np, scanpy as sc

adata = sc.read_10x_mtx("sample/outs/filtered_feature_bc_matrix")   # or sc.read_10x_h5(".../filtered_feature_bc_matrix.h5")
adata.var_names_make_unique()
sc.pp.filter_genes(adata, min_cells=3)                              # basic QC (filter cells/genes as you see fit)
X = np.asarray(adata.X.todense() if hasattr(adata.X, "todense") else adata.X, dtype="float32")
gene_names = adata.var_names.tolist()

# ...or from an existing AnnData .h5ad:
# adata = sc.read_h5ad("counts.h5ad")
# X = np.asarray(adata.layers["counts"].todense(), dtype="float32"); gene_names = adata.var_names.tolist()

# TFs = a TF list (e.g. allTFs_hg38.txt from aertslab) intersected with the genes present
tf_names = [g for g in open("allTFs_hg38.txt").read().split() if g in set(gene_names)]
```

Then run the pipeline:

```python
from scenic_rs import grnboost2, genie3, aucell, RankingDb, ctx

# 1. GRN: TF -> target importances  (pandas DataFrame [TF, target, importance])
adj = grnboost2(X, gene_names, tf_names)        # default — fast, OOB early stopping
# adj = genie3(X, gene_names, tf_names)         # alternative — random forest, slower, ~same result (use one)

# 2. ctx: prune co-expression modules to motif-supported regulons
dbs = [RankingDb("hg38_...genes_vs_motifs.rankings.feather", "hg38")]
regulons = ctx(adj, X, gene_names, dbs, "motifs-...hgnc-...tbl")   # -> [Regulon(...)]

# 3. AUCell: per-cell regulon activity
auc = aucell(X, gene_names, {r.name: r.genes for r in regulons})
```

`adj` matches pySCENIC's adjacencies format and `ctx` matches its regulon output,
so any step can also be swapped in individually alongside pySCENIC.

## Validation & benchmarks (vs real pySCENIC)

- Running each step (grnboost2/genie3/aucell/ctx) in both scenic-rs and pySCENIC 0.12.1 / ctxcore 0.2.0
- Uses the same inputs (pbmc3k 2700 cells × 758 genes, 300 TFs), (hg38 cisTarget DB 5876 motifs × 27015 genes + motif2tf annotations)
- Equal cores — rayon threads = pySCENIC's Dask-worker count
- Backend startup counted on both sides

### Correctness — scenic-rs reproduces pySCENIC's numbers

| step | concordance vs pySCENIC |
|---|---|
| GRN / GRNBoost2 | Spearman **0.74**, top-1000 edge Jaccard 0.56 — *at the stochastic ceiling* |
| GRN / GENIE3 | Spearman **0.99**, per-target median 0.99 |
| ctx / cisTarget | per-(module,motif) NES Spearman **1.00** (max diff 1.4e-13) |
| AUCell | Spearman **0.98**, max abs diff 0.06 |

GRNBoost2 is stochastic — running the algorithm twice (different seed) only agrees ~0.73 per target.

![output concordance](https://raw.githubusercontent.com/nglaszik/SCENIC-rs/main/bench/figures/concordance.png)
![GRN per-target concordance](https://raw.githubusercontent.com/nglaszik/SCENIC-rs/main/bench/figures/grn_per_target.png)

### Memory & speed

- Peak memory (PSS) is **constant** in scenic-rs as cores increase, but
grows ~linearly in pySCENIC (per-worker copies). Ratios = pySCENIC / scenic-rs:

| step | memory: 16c → 96c | speed (equal cores) |
|---|---|---|
| GRN / GRNBoost2 | **13× → 26×** less | **2.8× faster @96c** |
| GRN / GENIE3 | **30× → 123×** less | equivalent |
| ctx / cisTarget | **14× → 26×** less | **~10× faster** |
| AUCell | **18× → 51×** less | **~170× faster** |

At 96 cores pySCENIC peaks at ~11.7 GB (GRNBoost2), ~20.5 GB (GENIE3) and ~17 GB
(ctx); scenic-rs stays at ~0.45 GB, ~0.17 GB and ~0.66 GB respectively.

Speedup on GRNBoost2 is likely due to Dask setup, relatively negligible on larger datasets.

![peak memory vs cores](https://raw.githubusercontent.com/nglaszik/SCENIC-rs/main/bench/figures/mem_scaling.png)
![wall-clock vs cores](https://raw.githubusercontent.com/nglaszik/SCENIC-rs/main/bench/figures/time_scaling.png)

### Reproduce

Needs a pySCENIC env, e.g.
`python3.10 -m venv ~/venvs/pyscenic_clean && ~/venvs/pyscenic_clean/bin/pip install "numpy<1.24" "pandas<2" pyscenic`:

```bash
# correctness + cached adjacencies (GRN + AUCell)
python bench/benchmark_pyscenic.py --workers 16 --genie3
# memory/time scaling sweep across cores (GRNBoost2, GENIE3, AUCell)
python bench/mem_benchmark.py --sweep 16,48,96 --genie3
# ctx scaling sweep (real hg38 DB)
python bench/benchmark_ctx.py --sweep 16,48,96
# ctx per-step parity checks (math, DB, modules, regulons, NES concordance)
PYTHONPATH=python ~/venvs/pyscenic_clean/bin/python bench/validate_ctx_regulons.py
# render figures
python bench/plot_benchmark.py        # concordance.png, grn_per_target.png
python bench/plot_mem_benchmark.py    # mem_scaling.png, time_scaling.png (all steps)
```

## License & attribution

**GPL-3.0-or-later.** scenic-rs is a Rust reimplementation of, and a derivative
work of, the GPL-3.0 projects [pySCENIC](https://github.com/aertslab/pySCENIC)
and [ctxcore](https://github.com/aertslab/ctxcore) (aertslab, VIB-KU Leuven) —
their GRN/cisTarget/AUCell workflow and the recovery/NES/module/pruning logic. It is an
independent project, **not affiliated with or endorsed by aertslab**. See
[`NOTICE`](NOTICE) for details.

If you use scenic-rs, please cite the original SCENIC work:

- Aibar *et al.* (2017) *SCENIC: single-cell regulatory network inference and clustering.* Nature Methods.
- Van de Sande *et al.* (2020) *A scalable SCENIC workflow for single-cell gene regulatory network analysis.* Nature Protocols.
- Moerman *et al.* (2019) *GRNBoost2 and Arboreto…* Bioinformatics.
