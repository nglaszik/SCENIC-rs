"""Stage-2 parity check: scenic-rs ctx enrichment math vs ctxcore 0.2.0.

Runs under the pyscenic env (has ctxcore); scenic_rs is imported via PYTHONPATH
pointing at the repo's python/ dir. Synthetic per-motif rankings — no DB needed.

    PYTHONPATH=python ~/venvs/pyscenic_clean/bin/python bench/validate_ctx_math.py
"""
import numpy as np
import pandas as pd

from ctxcore.recovery import aucs as ctx_aucs, recovery as ctx_recovery
import scenic_rs._core as rs

TOTAL_GENES = 27015
N_MOTIFS = 800
N_GENES = 35           # module genes present in DB
AUC_THR = 0.05
RANK_THR = 1500        # module2features default


def make_rankings(seed):
    rng = np.random.default_rng(seed)
    rows = [rng.choice(TOTAL_GENES, N_GENES, replace=False) for _ in range(N_MOTIFS)]
    return np.asarray(rows, dtype=np.int32)


def check(label, weights):
    rk = make_rankings(0)
    df = pd.DataFrame(rk, index=[f"m{i}" for i in range(N_MOTIFS)],
                      columns=[f"g{j}" for j in range(N_GENES)])

    # --- AUC + NES ---
    py_auc = ctx_aucs(df, TOTAL_GENES, weights, AUC_THR)
    py_nes = (py_auc - py_auc.mean()) / py_auc.std()
    rs_auc, rs_nes = rs._ctx_aucs_nes(rk, weights.tolist(), TOTAL_GENES, AUC_THR)
    rs_auc, rs_nes = np.asarray(rs_auc), np.asarray(rs_nes)
    d_auc = np.max(np.abs(py_auc - rs_auc))
    d_nes = np.max(np.abs(py_nes - rs_nes))

    # --- leading-edge critical point (rank_at_max) ---
    rccs, _ = ctx_recovery(df, TOTAL_GENES, weights, RANK_THR, AUC_THR, no_auc=True)
    avg2std = rccs.mean(axis=0) + 2.0 * rccs.std(axis=0)
    py_ram = np.array([int(np.argmax(rccs[i] - avg2std)) for i in range(N_MOTIFS)])
    rs_ram = np.asarray(rs._ctx_rank_at_max(rk, weights.tolist(), RANK_THR))
    ram_mismatch = int(np.sum(py_ram != rs_ram))

    print(f"[{label}]")
    print(f"  AUC  max|diff| = {d_auc:.3e}")
    print(f"  NES  max|diff| = {d_nes:.3e}")
    print(f"  rank_at_max mismatches: {ram_mismatch}/{N_MOTIFS}")
    ok = d_auc < 1e-9 and d_nes < 1e-7 and ram_mismatch == 0
    print(f"  => {'PASS' if ok else 'FAIL'}\n")
    return ok


if __name__ == "__main__":
    print(f"total_genes={TOTAL_GENES} n_motifs={N_MOTIFS} n_genes={N_GENES} "
          f"auc_thr={AUC_THR} rank_thr={RANK_THR}\n")
    ok1 = check("unweighted (weights=1)", np.ones(N_GENES))
    rng = np.random.default_rng(1)
    ok2 = check("weighted (random importances)", rng.uniform(0.1, 5.0, N_GENES))
    print("ALL PASS" if (ok1 and ok2) else "SOME FAILED")
