"""Plot the scenic-rs vs pySCENIC memory/scaling sweep (reads bench/cache/mem_results.json).

Produces, in bench/figures/:
  - mem_scaling.png  : peak PSS (MB) vs core count, one panel per step
  - time_scaling.png : wall-clock (s) vs core count, one panel per step

Run mem_benchmark.py with --sweep first, e.g.
    python bench/mem_benchmark.py --sweep 16,48,96 --genie3
    python bench/plot_mem_benchmark.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

RS = "#2a9d8f"   # scenic-rs
PY = "#e76f51"   # pySCENIC

with open(os.path.join(CACHE, "mem_results.json")) as fh:
    blob = json.load(fh)
meta, rows = blob["meta"], blob["results"]
sub = f"{meta['cells']} cells × {meta['genes']} genes · {meta['tfs']} TFs"

steps = []
for r in rows:
    if r["step"] not in steps:
        steps.append(r["step"])
workers = sorted({r["workers"] for r in rows})


def series(step, impl, key):
    by_w = {r["workers"]: r[key] for r in rows if r["step"] == step and r["impl"] == impl}
    return [by_w.get(w) for w in workers]


def scaling_figure(key, ylabel, title, fname):
    n = len(steps)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), squeeze=False)
    for ax, step in zip(axes[0], steps):
        for impl, color in (("scenic-rs", RS), ("pySCENIC", PY)):
            y = series(step, impl, key)
            ax.plot(workers, y, "-o", color=color, label=impl, lw=2, ms=6)
        ax.set_title(step)
        ax.set_xlabel("cores (rayon threads / dask workers)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(workers)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(f"{title}\n{sub}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.join(FIG, fname)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if len(workers) < 2:
    print("only one core count in results — run mem_benchmark.py with --sweep for scaling curves")

scaling_figure("peak_pss_mb", "peak PSS (MB)",
               "Peak memory vs cores — threads share, processes don't", "mem_scaling.png")
scaling_figure("secs", "wall-clock (s)",
               "Wall-clock vs cores (end-to-end, incl. startup)", "time_scaling.png")
