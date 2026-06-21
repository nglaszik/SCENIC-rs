"""Stage-4 validation: scenic-rs full ctx() vs a CORRECT reference built from
ctxcore's own primitives (recovery/aucs), on the REAL hg38 DB + motif2tf.

NB: we do NOT compare against pyscenic's modules2df/df2regulons, because
pyscenic 0.12.1 has a leading-edge row-misalignment bug (recovery curves paired
with the wrong motif after the annotation sort; see docs/ctx_spec.md). The
reference here uses ctxcore.recovery per motif with correct row alignment.

    PYTHONPATH=python ~/venvs/pyscenic_clean/bin/python bench/validate_ctx_regulons.py
"""
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from pyscenic.utils import modules_from_adjacencies, load_motif_annotations
from ctxcore.rnkdb import FeatherRankingDatabase
from ctxcore.recovery import aucs as ctx_aucs, recovery
import scenic_rs._core as rs


def correct_reference(db, modules, ann, RT, AUC, NES):
    """Correct ctx regulons using ctxcore primitives with proper row alignment."""
    ann_index = set(ann.index)  # (TF, motif) pairs that passed the load filter
    groups = defaultdict(list)
    for m in modules:
        df = db.load(m)
        genes = df.columns.values
        rankings = df.values
        n_present = len(genes)
        module_size = len(m)
        if module_size == 0 or (module_size - n_present) / module_size >= 0.20:
            continue
        w = np.ones(n_present)
        a = ctx_aucs(df, db.total_genes, w, AUC)
        nes = (a - a.mean()) / a.std()
        rccs, _ = recovery(df, db.total_genes, w, RT, AUC, no_auc=True)
        avg2std = rccs.mean(0) + 2.0 * rccs.std(0)
        tf = m.transcription_factor
        act = "activating" in m.context
        for i, motif in enumerate(df.index.values):
            if nes[i] >= NES and (tf, motif) in ann_index:
                ram = int(np.argmax(rccs[i] - avg2std))          # correctly aligned: rccs[i] <-> motif i
                tgs = {genes[j]: m[genes[j]] for j in range(n_present) if rankings[i, j] <= ram}
                groups[(tf, act)].append((float(nes[i]), tgs))
    regs = {}
    for (tf, act), rowlist in groups.items():
        g2w = {}
        for _, tgs in rowlist:
            for g, wt in tgs.items():
                g2w[g] = max(g2w.get(g, -1e18), wt)
        regs[f"{tf}{'(+)' if act else '(-)'}"] = (g2w, max(r[0] for r in rowlist))
    return regs

RES = os.path.expanduser(
    "~/jupyternotebooks/paper_github_code/sc_rna_seq/scenic_out/resources")
DBF = os.path.join(RES, "hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather")
TBL = os.path.join(RES, "motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl")

RT, AUC, NES, MASK, KOA = 5000, 0.05, 1.0, False, True


def main():
    db_py = FeatherRankingDatabase(DBF, name="hg38_500bp")
    db_rs = rs.RankingDb(DBF, "hg38_500bp")
    ann = load_motif_annotations(TBL)
    ann_tfs = list(ann.index.get_level_values(0).unique())
    db_geneset = set(db_py.genes)

    rng = np.random.default_rng(0)
    # TFs that are annotated AND present in the DB
    tfs = [t for t in ann_tfs if t in db_geneset]
    tfs = sorted(rng.choice(tfs, 14, replace=False).tolist())
    targets = sorted(rng.choice(sorted(db_geneset - set(tfs)), 120, replace=False).tolist())
    genes = sorted(set(tfs) | set(targets))

    rows = []
    for tf in tfs:
        for t in rng.choice(targets, 60, replace=False):
            rows.append((str(tf), str(t), float(rng.exponential(1.0) + 1e-3)))
    adj = pd.DataFrame(rows, columns=["TF", "target", "importance"])

    expr = rng.integers(0, 6, size=(150, len(genes))).astype(np.float32)
    ex = pd.DataFrame(expr, index=[f"c{i}" for i in range(150)], columns=genes)

    # --- correct reference (ctxcore primitives, properly aligned) ---
    modules = modules_from_adjacencies(adj, ex, rho_mask_dropouts=MASK, keep_only_activating=KOA)
    py = correct_reference(db_py, modules, ann, RT, AUC, NES)

    # --- scenic-rs ---
    rs_regs = rs.ctx(adj["TF"].tolist(), adj["target"].tolist(), adj["importance"].tolist(),
                     expr, genes, [db_rs], TBL, None, None, None, 20, 0.03, MASK, KOA,
                     RT, AUC, NES, 0.001, 0.0)
    rsd = {name: (dict(zip(gns, wts)), nes) for name, tf, act, gns, wts, nes in rs_regs}

    print(f"regulons: reference={len(py)}  scenic-rs={len(rsd)}")
    names_eq = set(py) == set(rsd)
    print(f"regulon names equal: {names_eq}")
    if not names_eq:
        print("  only pyscenic:", sorted(set(py) - set(rsd))[:10])
        print("  only scenic-rs:", sorted(set(rsd) - set(py))[:10])

    max_gene_set_diff = 0
    max_w = 0.0
    max_nes = 0.0
    for name in set(py) & set(rsd):
        pw, pn = py[name]
        rw, rn = rsd[name]
        sd = set(pw) ^ set(rw)
        if sd:
            print(f"  [{name}] only-pyscenic: {sorted(set(pw)-set(rw))}")
            print(f"  [{name}] only-scenic-rs: {sorted(set(rw)-set(pw))}")
        max_gene_set_diff = max(max_gene_set_diff, len(sd))
        for g in set(pw) & set(rw):
            max_w = max(max_w, abs(pw[g] - rw[g]))
        max_nes = max(max_nes, abs(pn - rn))
    print(f"max gene-set symmetric diff per regulon: {max_gene_set_diff}")
    print(f"max target-weight |diff|: {max_w:.3e}")
    print(f"max regulon NES |diff|:   {max_nes:.3e}")
    ok = names_eq and max_gene_set_diff == 0 and max_w < 1e-9 and max_nes < 1e-6
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
