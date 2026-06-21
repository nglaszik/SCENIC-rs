"""Stage-3 parity: scenic-rs module generation vs pyscenic modules_from_adjacencies.

Synthetic adjacencies + expression. Compares the multiset of modules as
(TF, activating, frozenset(genes)) and the per-gene weights.

    PYTHONPATH=python ~/venvs/pyscenic_clean/bin/python bench/validate_ctx_modules.py
"""
from collections import Counter

import numpy as np
import pandas as pd

from pyscenic.utils import modules_from_adjacencies
import scenic_rs._core as rs

N_GENES, N_CELLS, N_TFS = 200, 300, 20


def synth(seed=0):
    rng = np.random.default_rng(seed)
    genes = [f"g{i}" for i in range(N_GENES)]
    # raw-count-like with zeros (so mask_dropouts has an effect)
    expr = rng.integers(0, 6, size=(N_CELLS, N_GENES)).astype(np.float32)
    tfs = genes[:N_TFS]
    rows = []
    for tf in tfs:
        targets = rng.choice(genes, 60, replace=False)
        for t in targets:
            if t == tf:
                continue
            rows.append((tf, t, float(rng.exponential(1.0) + 1e-3)))
    adj = pd.DataFrame(rows, columns=["TF", "target", "importance"])
    ex = pd.DataFrame(expr, index=[f"c{i}" for i in range(N_CELLS)], columns=genes)
    return adj, ex, expr, genes


def key(tf, act, genes):
    return (tf, act, frozenset(genes))


def run(label, mask, keep_only_activating):
    adj, ex, expr, genes = synth()

    py_mods = modules_from_adjacencies(
        adj, ex, rho_mask_dropouts=mask, keep_only_activating=keep_only_activating)
    py_keys = Counter()
    py_w = {}
    for m in py_mods:
        act = "activating" in m.context
        k = key(m.transcription_factor, act, m.genes)
        py_keys[k] += 1
        py_w[k] = dict(m.gene2weight)

    rs_mods = rs._ctx_modules(
        adj["TF"].tolist(), adj["target"].tolist(), adj["importance"].tolist(),
        expr, genes, None, None, None, 20, 0.03, mask, keep_only_activating)
    rs_keys = Counter()
    rs_w = {}
    for tf, act, gns, wts in rs_mods:
        k = key(tf, act, gns)
        rs_keys[k] += 1
        rs_w[k] = dict(zip(gns, wts))

    set_eq = set(py_keys) == set(rs_keys)
    multiset_eq = py_keys == rs_keys
    # weight check over shared modules
    wdiff = 0.0
    for k in set(py_keys) & set(rs_keys):
        for g, w in py_w[k].items():
            wdiff = max(wdiff, abs(w - rs_w[k].get(g, np.nan)))

    print(f"[{label}]  pyscenic={len(py_mods)} modules  scenic-rs={len(rs_mods)} modules")
    print(f"  set(modules) equal:      {set_eq}")
    print(f"  multiset(modules) equal: {multiset_eq}")
    print(f"  max gene-weight |diff|:  {wdiff:.3e}")
    only_py = set(py_keys) - set(rs_keys)
    only_rs = set(rs_keys) - set(py_keys)
    if only_py or only_rs:
        print(f"  only in pyscenic: {len(only_py)}  only in scenic-rs: {len(only_rs)}")
    ok = set_eq and multiset_eq and wdiff < 1e-9
    print(f"  => {'PASS' if ok else 'FAIL'}\n")
    return ok


if __name__ == "__main__":
    a = run("unmasked, activating-only", mask=False, keep_only_activating=True)
    b = run("masked, activating-only", mask=True, keep_only_activating=True)
    c = run("unmasked, activating+repressing", mask=False, keep_only_activating=False)
    print("ALL PASS" if (a and b and c) else "SOME FAILED")
