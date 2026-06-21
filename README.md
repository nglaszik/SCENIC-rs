# scenic-rs

A Rust backend for the [SCENIC](https://github.com/aertslab/pySCENIC)
single-cell gene regulatory network (GRN) pipeline.

**Status:** the **entire** SCENIC pipeline reimplemented as a Rust core with
`rayon` parallelism and `PyO3` bindings — GRN (GENIE3 / GRNBoost2 with out-of-bag
early stopping), **ctx / cisTarget** (motif enrichment → pruning → regulons), and
AUCell. **No Dask, no pySCENIC env**, `pip`-installable. You can now run
`grnboost2 → ctx → aucell` end-to-end without `numpy<1.24` / `pandas<2`.

## Why

pySCENIC is the standard for RNA-only GRN inference but is effectively inactive
(no release in 12+ months), and its biggest user pain is **Dask dependency hell**:
`numpy<1.24` / `pandas<2` pins that conflict with modern scanpy, plus a cluster to
babysit. Its memory also explodes with worker count — every Dask worker copies the
expression matrix (GRN) or the recovery-curve buffers and ranking DB (ctx), so
runs OOM exactly when you add cores to go faster.

scenic-rs is a self-contained Rust core that **removes Dask entirely** and
reproduces pySCENIC's outputs as a `pip`-installable drop-in. Because it uses
shared-memory threads (`rayon`) instead of worker processes, **memory stays flat
as you add cores** while pySCENIC's grows linearly — the headline win, on top of
no dependency pins. See
[Validation & benchmarks](#validation--benchmarks-vs-real-pyscenic).

## Requirements

**Runtime** (using scenic-rs)
- Python ≥ 3.9
- `numpy`, `pandas` — pulled in automatically as dependencies
- For the **`ctx`** step only: the cisTarget **ranking databases**
  (`*.genes_vs_motifs.rankings.feather`) and the **motif2TF annotations** (`.tbl`),
  from [resources.aertslab.org/cistarget](https://resources.aertslab.org/cistarget/)
  (~1–2 GB; the exact same files pySCENIC uses). Not needed for GRN or AUCell.

**Building from source**
- A recent stable **Rust** toolchain (`cargo`/`rustc`) — ≥ 1.70 (tested on 1.86);
  install via [rustup.rs](https://rustup.rs). The Arrow feather reader is pure
  Rust, so no system C/Fortran libraries are required.
- **maturin** ≥ 1.0 (`pip install maturin`)

**Benchmarks / validation only** (optional)
- A separate pySCENIC environment to compare against:
  `python3.10 -m venv ~/venvs/pyscenic_clean && ~/venvs/pyscenic_clean/bin/pip install "numpy<1.24" "pandas<2" pyscenic`
- `matplotlib` + `scipy` for the plotting and validation scripts

## Build

```bash
maturin develop --release          # builds the Rust core into the active env
python examples/run_example.py
```

## Usage

Full pipeline, all in-process, no Dask:

```python
from scenic_rs import grnboost2, aucell, RankingDb, ctx

# 1. GRN: TF -> target importances (GRNBoost2 with OOB early stopping, or genie3)
adj = grnboost2(expr, gene_names, tf_names)   # pandas DataFrame [TF, target, importance]

# 2. ctx: prune co-expression modules to motif-supported regulons
dbs = [RankingDb("hg38_...genes_vs_motifs.rankings.feather", "hg38")]
regulons = ctx(adj, expr, gene_names, dbs, "motifs-...hgnc-...tbl")   # -> [Regulon(...)]

# 3. AUCell: per-cell regulon activity
auc = aucell(expr, gene_names, {r.name: r.genes for r in regulons})
```

`adj` matches pySCENIC's adjacencies format and `ctx` matches its regulon output,
so any step can also be swapped in individually alongside pySCENIC.

## Validation & benchmarks (vs real pySCENIC)

Every step runs in **both** scenic-rs and **pySCENIC 0.12.1 / ctxcore 0.2.0** on
the *same* inputs, made apples-to-apples by construction (equal cores — rayon
threads pinned to pySCENIC's Dask-worker count; identical in-memory inputs;
backend startup counted on both sides). GRN/AUCell use pbmc3k (2700 cells × 758
genes, 300 TFs); ctx uses the real hg38 cisTarget DB (5876 motifs × 27015 genes)
+ motif2tf annotations.

### Correctness — scenic-rs reproduces pySCENIC's numbers

| step | concordance vs pySCENIC |
|---|---|
| GRN / GRNBoost2 | Spearman **0.74**, top-1000 edge Jaccard 0.56 — *at the stochastic ceiling* |
| GRN / GENIE3 | Spearman **0.99**, per-target median 0.99 |
| ctx / cisTarget | per-(module,motif) NES Spearman **1.00** (max diff 1.4e-13) |
| AUCell | Spearman **0.98**, max abs diff 0.06 |

GRNBoost2 is stochastic — running the *identical* algorithm twice (different seed)
only agrees ~0.73 per target — so 0.74 is essentially perfect, not low. GENIE3
(1000-tree RF) is far less stochastic, hence ~0.99.

![output concordance](bench/figures/concordance.png)
![GRN per-target concordance](bench/figures/grn_per_target.png)

### Memory & speed — flat vs cores; pySCENIC scales the wrong way

Peak memory (PSS) is roughly **constant** in scenic-rs as cores increase, but
grows ~linearly in pySCENIC (per-worker copies). Ratios = pySCENIC / scenic-rs:

| step | memory: 16c → 96c | speed (equal cores) |
|---|---|---|
| GRN / GRNBoost2 | **13× → 26×** less | 0.9× @16c → **2.8× @96c** |
| GRN / GENIE3 | **30× → 123×** less | ≈ par (compute-bound) |
| ctx / cisTarget | **14× → 26×** less | **~10× faster** (flat vs cores) |
| AUCell | **18× → 51×** less | **~170× faster** (algorithm) |

At 96 cores pySCENIC peaks at ~11.7 GB (GRNBoost2), ~20.5 GB (GENIE3) and ~17 GB
(ctx); scenic-rs stays at ~0.45 GB, ~0.17 GB and ~0.66 GB respectively.

![peak memory vs cores](bench/figures/mem_scaling.png)
![wall-clock vs cores](bench/figures/time_scaling.png)

**Reading the results**

- **Memory is the headline.** scenic-rs threads share one matrix / one ranking
  DB; pySCENIC's Dask workers each copy them, so its memory climbs with core count
  (and OOMs at the worker counts you'd use to go fast). scenic-rs is flat.
- **AUCell** is numerically identical (ρ 0.98) and ~170× faster as an algorithm —
  pySCENIC's wall-clock there is dominated by Dask/pool startup it doesn't skip.
- **ctx / cisTarget** matches pySCENIC's enrichment exactly (NES ρ 1.00) while
  being ~10× faster and flat in memory. (See the bug note below.)
- **GENIE3** matches almost exactly (ρ 0.99) at parity speed but with up to 123×
  less memory.
- **GRNBoost2** reproduces pySCENIC above the stochastic ceiling using the same
  OOB early stopping; after a pre-sort optimization it is ~par at 16 cores and
  *faster* at higher core counts.

> The single-core "runtime bar chart" was dropped on purpose: for the stochastic
> GRN step the scenic-rs/pySCENIC ratio flips with core count, so only the
> scaling curves (`time_scaling.png`) are meaningful. AUCell's end-to-end gain in
> `time_scaling.png` looks smaller than 170× because that figure includes fixed
> per-process Python startup; the ~170× is the algorithm itself.

### A bug we found in pySCENIC's ctx

While validating ctx we found (and proved) a **leading-edge row-misalignment bug
in pySCENIC 0.12.1** (`pyscenic/transform.py`): after the motif-annotation sort,
recovery curves get paired with the wrong motifs, so regulon *target genes* are
read off another motif's data. It violates the recovery-curve invariant
(`rcc[t]=0` for `t < min(rank)`) in pySCENIC's own output. The NES/AUC are
computed before this step and are correct (hence ρ 1.00 above); only target-gene
membership is affected. scenic-rs does it correctly, so its regulon targets are
validated against a corrected ctxcore-primitive reference rather than bug-for-bug
against pySCENIC. Details in `docs/ctx_spec.md`.

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

## Roadmap

- [x] GENIE3 (random forest) GRN inference, parallel over targets
- [x] GRNBoost2 (gradient-boosting) variant + OOB early stopping (matches arboreto)
- [x] GRNBoost2 pre-sort optimization (faster than arboreto at higher core counts)
- [x] AUCell in Rust (parallel over cells)
- [x] **ctx / cisTarget** in Rust (feather DB reader, recovery/NES, modules, pruning)
- [x] pySCENIC-compatible adjacencies + regulon output
- [x] Validation + benchmark harness vs real pySCENIC (per-step parity to machine eps)
- [x] Scaling study across core counts (rayon vs Dask: memory + time)
- [ ] Histogram-based tree splits (further GRN speed unlock)
- [ ] AnnData / loom convenience loaders
```
