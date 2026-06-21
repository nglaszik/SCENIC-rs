"""Plot the scenic-rs vs pySCENIC step benchmark (reads bench/cache/results.json).

Produces, in bench/figures/:
  - concordance.png   : rank-rank agreement of each step's output (validity)
  - grn_per_target.png: per-target Spearman distribution for the GRN step(s)
  - performance.png   : wall-clock per step + speedup (scenic-rs vs pySCENIC)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

RS = "#2a9d8f"   # scenic-rs
PY = "#e76f51"   # pySCENIC

with open(os.path.join(CACHE, "results.json")) as fh:
    blob = json.load(fh)
meta, results = blob["meta"], blob["results"]
sub = f"pbmc3k · {meta['cells']} cells × {meta['genes']} genes · {meta['tfs']} TFs"


def _pct_rank(v):
    v = np.asarray(v, dtype=float)
    return rankdata(v) / len(v)


# ---------------------------- 1. concordance ----------------------------
# Include a ctx panel if available. ctx compares per-(module,motif) NES vs pySCENIC
# — the enrichment math, which pySCENIC computes correctly (its leading-edge bug is
# downstream and doesn't affect NES). Regulon target genes are validated separately
# against a corrected reference (bench/validate_ctx_regulons.py).
conc = list(results)
ctx_path = os.path.join(CACHE, "ctx_concordance.json")
if os.path.exists(ctx_path):
    conc.append(json.load(open(ctx_path)))

n = len(conc)
fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False)
for ax, r in zip(axes[0], conc):
    xy = np.array(r["scatter"])
    x, y = _pct_rank(xy[:, 0]), _pct_rank(xy[:, 1])
    ax.hexbin(x, y, gridsize=45, cmap="magma_r", bins="log", mincnt=1)
    ax.plot([0, 1], [0, 1], "--", color="0.5", lw=1)
    ax.set_xlabel("scenic-rs (rank %)")
    ax.set_ylabel("pySCENIC (rank %)")
    txt = f"Spearman ρ = {r['overall_spearman']:.3f}"
    if "topk_jaccard" in r:
        txt += f"\ntop-1000 Jaccard = {r['topk_jaccard']['1000']:.2f}"
    if str(r["step"]).startswith("ctx"):
        txt += "\n(NES; regulon targets vs\ncorrected ref — pySCENIC bug)"
    ax.set_title(r["step"], fontweight="bold")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
fig.suptitle(f"Output concordance: scenic-rs vs pySCENIC   ({sub})", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(FIG, "concordance.png"), dpi=140)
print("wrote concordance.png")


# ------------------------- 2. GRN per-target Spearman -------------------------
grn = [r for r in results if "per_target_spearman" in r]
if grn:
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    colors = ["#264653", "#2a9d8f", "#e9c46a"]
    for r, c in zip(grn, colors):
        per = np.array(r["per_target_spearman"], dtype=float)
        per = per[np.isfinite(per)]
        ax.hist(per, bins=np.linspace(-0.2, 1.0, 31), alpha=0.6, color=c,
                label=f"{r['step']}  (median ρ={np.median(per):.2f}, n={len(per)})")
        ax.axvline(np.median(per), color=c, ls="--", lw=1.5)
    ax.axvline(0.73, color="0.4", ls=":", lw=1.8)
    # explicit legend entries for the two line styles (so dashed vs dotted is unambiguous)
    style_handles = [
        Line2D([0], [0], color="0.3", ls="--", lw=1.5,
               label="dashed = per-step median ρ (colored to match its histogram)"),
        Line2D([0], [0], color="0.4", ls=":", lw=1.8,
               label="dotted = stochastic ceiling ρ≈0.73 (sklearn vs sklearn, same algo)"),
    ]
    h, _ = ax.get_legend_handles_labels()
    ax.set_xlabel("per-target Spearman ρ (scenic-rs vs pySCENIC importances)")
    ax.set_ylabel("# targets")
    ax.set_title(f"GRN per-target concordance   ({sub})", fontsize=11)
    ax.legend(handles=h + style_handles, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "grn_per_target.png"), dpi=140)
    print("wrote grn_per_target.png")


# Per-step runtime at a single core count is intentionally NOT plotted: it's
# misleading for the stochastic GRN step (the scenic-rs/pySCENIC ratio flips with
# core count). Use time_scaling.png (wall-clock vs cores) from plot_mem_benchmark.py.

# console summary table
print("\nstep              pySCENIC   scenic-rs   speedup   Spearman")
for r in results:
    print(f"{r['step']:16s}  {r['t_pyscenic']:7.1f}s  {r['t_scenicrs']:8.2f}s   "
          f"{r['speedup']:6.1f}×   {r['overall_spearman']:.3f}")
