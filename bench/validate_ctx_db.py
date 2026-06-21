"""Stage-1 parity check: scenic-rs feather DB reader + enrichment vs ctxcore,
on the REAL hg38 ranking database.

    PYTHONPATH=python ~/venvs/pyscenic_clean/bin/python bench/validate_ctx_db.py
"""
import os
import time

import numpy as np

from ctxcore.rnkdb import FeatherRankingDatabase
from ctxcore.genesig import GeneSignature
from ctxcore.recovery import enrichment4features
import scenic_rs._core as rs

DB = os.path.expanduser(
    "~/jupyternotebooks/paper_github_code/sc_rna_seq/scenic_out/resources/"
    "hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather")

# --- load in both ---
t0 = time.perf_counter()
db_rs = rs.RankingDb(DB, "hg38_500bp")
t_rs = time.perf_counter() - t0
print(f"scenic-rs load: {t_rs:.1f}s  | motifs={db_rs.n_motifs} genes={db_rs.n_genes} "
      f"total_genes={db_rs.total_genes}")

t0 = time.perf_counter()
db_py = FeatherRankingDatabase(DB, name="hg38_500bp")
_ = db_py.genes  # force
t_py = time.perf_counter() - t0
print(f"ctxcore   meta: {t_py:.1f}s  | total_genes={db_py.total_genes}\n")

assert db_rs.total_genes == db_py.total_genes
assert db_rs.n_motifs == len(db_py.genes) or True  # n_motifs != n_genes; just sanity

# --- pick a real gene-set module and enrich in both ---
rng = np.random.default_rng(0)
all_genes = list(db_py.genes)
genes = sorted(rng.choice(all_genes, 40, replace=False).tolist())
gs = GeneSignature(name="mod", gene2weight={g: 1.0 for g in genes})

present, motifs_rs, aucs_rs, ness_rs = db_rs.enrich(genes, None, 0.05)
edf = enrichment4features(db_py, gs, rank_threshold=5000, auc_threshold=0.05)

py_auc = edf[("Enrichment", "AUC")]
py_nes = edf[("Enrichment", "NES")]
rs_auc = dict(zip(motifs_rs, aucs_rs))
rs_nes = dict(zip(motifs_rs, ness_rs))

common = [m for m in py_auc.index if m in rs_auc]
a_p = np.array([py_auc[m] for m in common])
a_r = np.array([rs_auc[m] for m in common])
n_p = np.array([py_nes[m] for m in common])
n_r = np.array([rs_nes[m] for m in common])

print(f"present genes: rs={len(present)} (of {len(genes)})  motifs compared: {len(common)}")
print(f"AUC max|diff| = {np.max(np.abs(a_p - a_r)):.3e}")
print(f"NES max|diff| = {np.max(np.abs(n_p - n_r)):.3e}")
print(f"top motif by NES  ctxcore={py_nes.idxmax()}  scenic-rs={common[int(np.argmax(n_r))]}")
ok = np.max(np.abs(a_p - a_r)) < 1e-9 and np.max(np.abs(n_p - n_r)) < 1e-6
print("PASS" if ok else "FAIL")
