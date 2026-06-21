"""Unified scenic-rs vs pySCENIC scaling figures across ALL SCENIC steps.

Reads:
  - bench/cache/mem_results.json     (GRNBoost2, GENIE3, AUCell; run mem_benchmark.py --sweep ... --genie3)
  - bench/cache/ctx_bench_results.json (ctx/cisTarget; run benchmark_ctx.py --sweep ...)

Produces, in bench/figures/ (one figure each, one panel per step):
  - mem_scaling.png  : peak PSS (MB) vs core count
  - time_scaling.png : wall-clock (s) vs core count

Inputs differ per step (GRN/AUCell on pbmc3k; ctx on the hg38 ranking DB with
synthetic modules) — noted in each panel — but the point is the SHAPE: scenic-rs
threads share memory (flat); pySCENIC processes/dask workers don't (scales up).
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
RS, PY = "#2a9d8f", "#e76f51"

STEP_ORDER = ["grnboost2", "genie3", "ctx", "aucell"]
TITLE = {"grnboost2": "GRN / GRNBoost2", "genie3": "GRN / GENIE3",
         "ctx": "ctx / cisTarget", "aucell": "AUCell"}

rows = []
notes = {}

if os.path.exists(os.path.join(CACHE, "mem_results.json")):
    blob = json.load(open(os.path.join(CACHE, "mem_results.json")))
    m = blob["meta"]
    rows += blob["results"]
    for s in ("grnboost2", "genie3", "aucell"):
        notes[s] = f"pbmc3k · {m['cells']}×{m['genes']} · {m['tfs']} TFs"

if os.path.exists(os.path.join(CACHE, "ctx_bench_results.json")):
    blob = json.load(open(os.path.join(CACHE, "ctx_bench_results.json")))
    m = blob["meta"]
    rows += [{**r, "step": "ctx"} for r in blob["results"]]
    notes["ctx"] = f"hg38 DB · {m['tfs']} TFs · {m.get('dbs', 1)} DB(s)"

steps = [s for s in STEP_ORDER if any(r["step"] == s for r in rows)]
workers = sorted({r["workers"] for r in rows})


def series(step, impl, key):
    by_w = {r["workers"]: r[key] for r in rows if r["step"] == step and r["impl"] == impl}
    return [by_w.get(w) for w in workers]


def figure(key, ylabel, title, fname):
    n = len(steps)
    fig, axes = plt.subplots(1, n, figsize=(3.7 * n, 4.0), squeeze=False)
    for ax, step in zip(axes[0], steps):
        for impl, color in (("scenic-rs", RS), ("pySCENIC", PY)):
            y = series(step, impl, key)
            ax.plot(workers, y, "-o", color=color, label=impl, lw=2, ms=6)
        ax.set_title(f"{TITLE[step]}\n{notes.get(step, '')}", fontsize=10)
        ax.set_xlabel("cores")
        ax.set_ylabel(ylabel)
        ax.set_xticks(workers)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(FIG, fname)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}  (steps: {', '.join(steps)})")


if not rows:
    raise SystemExit("no results found — run mem_benchmark.py --sweep ... --genie3 and benchmark_ctx.py --sweep ...")
if len(workers) < 2:
    print("only one core count — run the sweeps with --sweep for scaling curves")

figure("peak_pss_mb", "peak PSS (MB)",
       "Peak memory vs cores — scenic-rs threads share, pySCENIC workers don't", "mem_scaling.png")
figure("secs", "wall-clock (s)",
       "Wall-clock vs cores (end-to-end, incl. startup)", "time_scaling.png")
