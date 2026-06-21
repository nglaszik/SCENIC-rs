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
n = len(results)
fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False)
for ax, r in zip(axes[0], results):
    xy = np.array(r["scatter"])
    x, y = _pct_rank(xy[:, 0]), _pct_rank(xy[:, 1])
    hb = ax.hexbin(x, y, gridsize=45, cmap="magma_r", bins="log", mincnt=1)
    ax.plot([0, 1], [0, 1], "--", color="0.5", lw=1)
    ax.set_xlabel("scenic-rs (rank %)")
    ax.set_ylabel("pySCENIC (rank %)")
    txt = f"Spearman ρ = {r['overall_spearman']:.3f}"
    if "topk_jaccard" in r:
        txt += f"\ntop-1000 Jaccard = {r['topk_jaccard']['1000']:.2f}"
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
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    colors = ["#264653", "#2a9d8f", "#e9c46a"]
    for r, c in zip(grn, colors):
        per = np.array(r["per_target_spearman"], dtype=float)
        per = per[np.isfinite(per)]
        ax.hist(per, bins=np.linspace(-0.2, 1.0, 31), alpha=0.6, color=c,
                label=f"{r['step']}  (median {np.median(per):.2f}, n={len(per)})")
        ax.axvline(np.median(per), color=c, ls="--", lw=1.5)
    ax.axvline(0.73, color="0.4", ls=":", lw=1.5,
               label="sklearn-vs-sklearn ceiling (~0.73)")
    ax.set_xlabel("per-target Spearman ρ (scenic-rs vs pySCENIC importances)")
    ax.set_ylabel("# targets")
    ax.set_title(f"GRN per-target concordance   ({sub})", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "grn_per_target.png"), dpi=140)
    print("wrote grn_per_target.png")


# ---------------------------- 3. performance ----------------------------
steps = [r["step"] for r in results]
t_py = [r["t_pyscenic"] for r in results]
t_rs = [r["t_scenicrs"] for r in results]
x = np.arange(len(steps))
w = 0.38
fig, ax = plt.subplots(figsize=(1.9 * len(steps) + 2, 4.6))
ax.bar(x - w / 2, t_py, w, label="pySCENIC", color=PY)
ax.bar(x + w / 2, t_rs, w, label="scenic-rs", color=RS)
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(steps)
ax.set_ylabel("wall-clock (s, log scale)")
ax.set_title(f"Per-step runtime, equal {meta['workers']} cores   ({sub})", fontsize=11)
ax.legend()


def speedup_label(s):
    if s >= 2:
        return f"{s:.0f}×\nfaster", "#1b7837"
    if s >= 1.1:
        return f"{s:.1f}×\nfaster", "#1b7837"
    if s >= 0.9:
        return "≈ par", "0.3"
    return f"{1/s:.1f}×\nslower", "#b2182b"


for xi, r in zip(x, results):
    top = max(r["t_pyscenic"], r["t_scenicrs"])
    lab, col = speedup_label(r["speedup"])
    ax.text(xi, top * 1.3, lab, ha="center", va="bottom", fontsize=9,
            fontweight="bold", color=col)
ax.set_ylim(top=max(t_py) * 6)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "performance.png"), dpi=140)
print("wrote performance.png")

# console summary table
print("\nstep              pySCENIC   scenic-rs   speedup   Spearman")
for r in results:
    print(f"{r['step']:16s}  {r['t_pyscenic']:7.1f}s  {r['t_scenicrs']:8.2f}s   "
          f"{r['speedup']:6.1f}×   {r['overall_spearman']:.3f}")
