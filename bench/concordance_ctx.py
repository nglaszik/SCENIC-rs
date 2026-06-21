"""ctx concordance data: scenic-rs vs pySCENIC per-(module,motif) NES.

The NES/AUC are computed BEFORE pySCENIC's leading-edge misalignment bug, so this
is a clean, bug-free comparison of the enrichment math (it should be ~perfect).
Regulon target-gene membership is NOT comparable to pySCENIC (its leading edges
are corrupted) and is instead validated against a corrected reference in
bench/validate_ctx_regulons.py.

Writes bench/cache/ctx_concordance.json (Spearman/Pearson + a scatter sample) for
plot_benchmark.py to add a ctx panel to concordance.png.

    PYTHONPATH=python ~/venvs/pyscenic_clean/bin/python bench/concordance_ctx.py
"""
import json
import os

import numpy as np

from pyscenic.utils import modules_from_adjacencies, load_motif_annotations
from ctxcore.rnkdb import FeatherRankingDatabase
from ctxcore.recovery import enrichment4features
from ctxcore.genesig import GeneSignature
from scipy.stats import spearmanr, pearsonr
import scenic_rs._core as rs

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
RES = os.path.expanduser(
    "~/jupyternotebooks/paper_github_code/sc_rna_seq/scenic_out/resources")
DBF = os.path.join(RES, "hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather")
TBL = os.path.join(RES, "motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl")
RT, AUC = 5000, 0.05


def main():
    db_py = FeatherRankingDatabase(DBF, name="x")
    db_rs = rs.RankingDb(DBF, "x")
    ann = load_motif_annotations(TBL)
    dg = set(db_py.genes)
    ann_tfs = [t for t in ann.index.get_level_values(0).unique() if t in dg]
    rng = np.random.default_rng(0)
    tfs = sorted(rng.choice(ann_tfs, 20, replace=False).tolist())
    targets = sorted(rng.choice(sorted(dg - set(tfs)), 200, replace=False).tolist())
    genes = sorted(set(tfs) | set(targets))
    rows = [(str(tf), str(t), float(rng.exponential(1.0) + 1e-3))
            for tf in tfs for t in rng.choice(targets, 60, replace=False)]
    import pandas as pd
    adj = pd.DataFrame(rows, columns=["TF", "target", "importance"])
    expr = rng.integers(0, 6, size=(150, len(genes))).astype(np.float32)
    ex = pd.DataFrame(expr, index=[f"c{i}" for i in range(150)], columns=genes)
    modules = modules_from_adjacencies(adj, ex, rho_mask_dropouts=False, keep_only_activating=True)

    rs_all, py_all = [], []
    for m in modules:
        gl = list(m.genes)
        # scenic-rs NES per motif
        present, motifs_rs, _aucs, ness_rs = db_rs.enrich(gl, None, AUC)
        rs_nes = dict(zip(motifs_rs, ness_rs))
        # pyscenic NES per motif (bug-free enrichment math)
        gs = GeneSignature(name=m.transcription_factor, gene2weight={g: 1.0 for g in gl})
        edf = enrichment4features(db_py, gs, rank_threshold=RT, auc_threshold=AUC)
        py_nes = edf[("Enrichment", "NES")]
        for motif in py_nes.index:
            if motif in rs_nes:
                rs_all.append(rs_nes[motif]); py_all.append(float(py_nes[motif]))

    rs_all = np.asarray(rs_all); py_all = np.asarray(py_all)
    sp = float(spearmanr(rs_all, py_all).correlation)
    pe = float(pearsonr(rs_all, py_all)[0])
    md = float(np.max(np.abs(rs_all - py_all)))
    rng2 = np.random.default_rng(0)
    idx = rng2.choice(len(rs_all), min(20000, len(rs_all)), replace=False)
    out = {"step": "ctx / NES", "n_modules": len(modules), "n_pairs": int(len(rs_all)),
           "overall_spearman": sp, "overall_pearson": pe, "max_abs_diff": md,
           "scatter": np.c_[rs_all[idx], py_all[idx]].tolist()}
    json.dump(out, open(os.path.join(CACHE, "ctx_concordance.json"), "w"))
    print(f"ctx NES concordance: Spearman={sp:.4f} Pearson={pe:.4f} "
          f"max|diff|={md:.2e} over {len(rs_all)} (module,motif) pairs from {len(modules)} modules")


if __name__ == "__main__":
    main()
