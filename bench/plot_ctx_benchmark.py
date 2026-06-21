"""Plot the scenic-rs vs pySCENIC ctx (cisTarget) benchmark
(reads bench/cache/ctx_bench_results.json). Produces bench/figures/ctx_scaling.png:
peak PSS (MB) and wall-clock (s) vs core count.

    python bench/benchmark_ctx.py --sweep 8,32
    python bench/plot_ctx_benchmark.py
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

with open(os.path.join(CACHE, "ctx_bench_results.json")) as fh:
    blob = json.load(fh)
meta, rows = blob["meta"], blob["results"]
workers = sorted({r["workers"] for r in rows})
sub = (f"{meta['edges']} edges · {meta['genes']} genes · {meta['tfs']} TFs · "
       f"{meta['dbs']} DB(s)")


def series(impl, key):
    by_w = {r["workers"]: r[key] for r in rows if r["impl"] == impl}
    return [by_w.get(w) for w in workers]


fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 4.2))
for impl, c in (("scenic-rs", RS), ("pySCENIC", PY)):
    a1.plot(workers, series(impl, "peak_pss_mb"), "-o", color=c, label=impl, lw=2, ms=6)
    a2.plot(workers, series(impl, "secs"), "-o", color=c, label=impl, lw=2, ms=6)
a1.set_title("ctx peak memory (PSS)")
a1.set_ylabel("peak PSS (MB)")
a2.set_title("ctx wall-clock")
a2.set_ylabel("seconds")
for ax in (a1, a2):
    ax.set_xlabel("cores (rayon threads / dask workers)")
    ax.set_xticks(workers)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
fig.suptitle(f"ctx (cisTarget): scenic-rs shares one in-memory DB; pySCENIC copies "
             f"rccs per worker\n{sub}", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.90))
out = os.path.join(FIG, "ctx_scaling.png")
fig.savefig(out, dpi=130)
print(f"wrote {out}")
