"""Peak-memory benchmark: scenic-rs vs **real pySCENIC**, per SCENIC step.

Companion to benchmark_pyscenic.py (which measures wall-clock). Here we measure
*peak resident memory* for each step, run in isolation, with a fair accounting of
multi-process tools:

  * pySCENIC's GRN uses Dask with processes=True -> it spawns N worker processes.
    Measuring only the parent would massively undercount. We sample the WHOLE
    process tree (parent + all descendants).
  * Summing RSS across processes double-counts shared pages (shared libs, and the
    expression matrix Dask scatters to workers). So the headline metric is PSS
    (proportional set size): shared pages are split across the procs that map them,
    giving the true marginal memory footprint of the tool. RSS-sum is also recorded
    as a (pessimistic) upper bound.

Each (impl x step) runs as a fresh subprocess with --force, so the peak reflects
that step alone (plus the interpreter + import baseline, which is a real cost of
running the tool and is reported separately via the idle-baseline column).

    python bench/mem_benchmark.py --workers 16
    python bench/mem_benchmark.py --workers 16 --genie3
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

import psutil

from benchmark_pyscenic import prepare_inputs, DEFAULT_PY

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")


def tree_mem(proc):
    """Return (pss_bytes, rss_bytes) summed over proc + all live descendants.

    PSS divides each shared page by the number of procs mapping it, so summing
    PSS across the tree is correct (no double-count). RSS-sum overcounts shared
    pages but needs no smaps read, so we keep it as an upper bound.
    """
    try:
        procs = [proc] + proc.children(recursive=True)
    except psutil.NoSuchProcess:
        return 0, 0
    pss = rss = 0
    for p in procs:
        try:
            mi = p.memory_full_info()
            pss += getattr(mi, "pss", mi.rss)   # pss is Linux-only; fall back to rss
            rss += mi.rss
        except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
            pass
    return pss, rss


class PeakSampler(threading.Thread):
    def __init__(self, pid, interval=0.05):
        super().__init__(daemon=True)
        self.proc = psutil.Process(pid)
        self.interval = interval
        self.peak_pss = 0
        self.peak_rss = 0
        # NB: must not be named `_stop` — that shadows threading.Thread's internal
        # _stop() method and breaks join().
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            pss, rss = tree_mem(self.proc)
            self.peak_pss = max(self.peak_pss, pss)
            self.peak_rss = max(self.peak_rss, rss)
            if not self.proc.is_running():
                break
            time.sleep(self.interval)

    def stop(self):
        self._stop_event.set()


def profile(cmd, env):
    """Run cmd, sampling its whole process-tree peak memory.

    Returns (pss, rss, secs, returncode, out). A non-zero returncode is NOT
    raised: a worker that OOMs (the expected pySCENIC failure mode at high
    --workers) is a reportable benchmark result, not a reason to abort the
    sweep. The peak memory sampled up to the crash is still returned.
    """
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    sampler = PeakSampler(p.pid)
    sampler.start()
    out, _ = p.communicate()
    sampler.stop()
    sampler.join()
    dt = time.perf_counter() - t0
    # surface the worker's own per-step timing line (or the error tail on failure)
    for line in out.splitlines():
        if line.strip():
            print("   ", line.strip())
    return sampler.peak_pss, sampler.peak_rss, dt, p.returncode, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="pbmc3k_prep_500_300_0.npz")
    ap.add_argument("--pyscenic-python", default=DEFAULT_PY)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sweep", default=None,
                    help="comma-separated worker counts to sweep, e.g. 16,48,96 "
                         "(overrides --workers; shows how RAM & time scale with cores)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--genie3", action="store_true")
    args = ap.parse_args()
    assert os.path.exists(args.pyscenic_python), f"not found: {args.pyscenic_python}"

    worker_list = ([int(w) for w in args.sweep.split(",")] if args.sweep
                   else [args.workers])

    n_cells, n_genes, n_tfs = prepare_inputs(args.npz)
    print(f"data: {n_cells} cells x {n_genes} genes ({n_tfs} TFs)   "
          f"|   cores = {worker_list}\n")

    steps = ["grnboost2"] + (["genie3"] if args.genie3 else []) + ["aucell"]

    base_py_env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}

    MB = 1024 * 1024
    rows = []
    for w in worker_list:
        # rayon reads RAYON_NUM_THREADS; dask worker count comes from --workers.
        env_rs = {**os.environ, "RAYON_NUM_THREADS": str(w)}
        impls = [
            ("scenic-rs", sys.executable, os.path.join(HERE, "_steps_scenicrs.py"), env_rs),
            ("pySCENIC", args.pyscenic_python, os.path.join(HERE, "_steps_pyscenic.py"), base_py_env),
        ]
        for step in steps:
            for name, py, script, env in impls:
                print(f"== {name}  /  {step}  /  {w} cores ==")
                cmd = [py, script, "--workers", str(w), "--seed",
                       str(args.seed), "--only", step, "--force"]
                pss, rss, dt, rc, out = profile(cmd, env)
                failed = rc != 0
                low = out.lower()
                oom = failed and ("memoryerror" in low or "killed" in low
                                  or "out of memory" in low or "oom" in low
                                  or "cannot allocate" in low)
                rows.append({"workers": w, "step": step, "impl": name,
                            "peak_pss_mb": pss / MB, "peak_rss_mb": rss / MB,
                            "secs": dt, "failed": failed, "oom": oom,
                            "returncode": rc})
                status = (f"  ** FAILED (rc={rc}{', likely OOM' if oom else ''}) **"
                          if failed else "")
                print(f"    peak PSS {pss/MB:8.1f} MB   |   peak RSS {rss/MB:8.1f} MB"
                      f"   |   {dt:6.1f} s{status}\n")

    json.dump({"meta": {"cells": n_cells, "genes": n_genes, "tfs": n_tfs,
                       "workers": worker_list}, "results": rows},
              open(os.path.join(CACHE, "mem_results.json"), "w"), indent=2)

    def get(step, impl, w):
        return next(r for r in rows if r["step"] == step and r["impl"] == impl
                    and r["workers"] == w)

    # Per-step scaling table: how peak PSS and wall-time move with core count.
    for step in steps:
        print(f"\n=== {step} ===")
        print(f"{'cores':>6s}  | {'scenic-rs PSS':>14s} {'time':>8s}  | "
              f"{'pySCENIC PSS':>14s} {'time':>8s}  | {'PSS x':>6s} {'time x':>7s}")
        for w in worker_list:
            sr, py = get(step, "scenic-rs", w), get(step, "pySCENIC", w)
            pr = py["peak_pss_mb"] / sr["peak_pss_mb"] if sr["peak_pss_mb"] else float("nan")
            tr = py["secs"] / sr["secs"] if sr["secs"] else float("nan")
            flag = ""
            if py.get("failed"):
                flag = "  <- pySCENIC FAILED" + (" (OOM)" if py.get("oom") else "")
            if sr.get("failed"):
                flag += "  <- scenic-rs FAILED"
            print(f"{w:6d}  | {sr['peak_pss_mb']:11.1f} MB {sr['secs']:7.1f}s  | "
                  f"{py['peak_pss_mb']:11.1f} MB {py['secs']:7.1f}s  | "
                  f"{pr:5.2f}x {tr:6.2f}x{flag}")

    # Scaling factor across the sweep (first -> last worker count) per impl/step.
    if len(worker_list) > 1:
        w0, w1 = worker_list[0], worker_list[-1]
        print(f"\nscaling {w0} -> {w1} cores (peak PSS growth | wall-time change):")
        for step in steps:
            for impl in ("scenic-rs", "pySCENIC"):
                a, b = get(step, impl, w0), get(step, impl, w1)
                mg = b["peak_pss_mb"] / a["peak_pss_mb"] if a["peak_pss_mb"] else float("nan")
                tg = b["secs"] / a["secs"] if a["secs"] else float("nan")
                print(f"  {step:10s} {impl:10s}  PSS x{mg:5.2f}   time x{tg:5.2f}")
    print(f"\nwrote {os.path.join(CACHE, 'mem_results.json')}  ->  run plot_mem_benchmark.py")


if __name__ == "__main__":
    main()
