"""scenic-rs: fast Rust backend for the SCENIC pipeline.

GRN inference (GENIE3 / GRNBoost2), cisTarget pruning (ctx) and regulon scoring
(AUCell) — the whole pipeline in Rust, no Dask.
"""
import collections

from ._core import RankingDb
from ._core import aucell as _aucell
from ._core import ctx as _ctx
from ._core import genie3 as _genie3
from ._core import grnboost2 as _grnboost2

__all__ = ["genie3", "grnboost2", "aucell", "ctx", "RankingDb", "Regulon"]

#: A pruned regulon. ``genes``/``weights`` are the leading-edge target genes and
#: their (max) importances; ``activating`` is True for "(+)" regulons.
Regulon = collections.namedtuple("Regulon", "name tf activating genes weights nes")


def _to_frame(tf, target, importance, as_frame):
    if not as_frame:
        return tf, target, importance
    import pandas as pd

    return (
        pd.DataFrame({"TF": tf, "target": target, "importance": importance})
        .sort_values("importance", ascending=False, ignore_index=True)
    )


def genie3(expr, gene_names, tf_names, n_estimators=1000, max_features="sqrt",
           min_samples_leaf=1, seed=42, as_frame=True):
    """GRN inference with GENIE3 (random forests)."""
    import numpy as np

    X = np.ascontiguousarray(expr, dtype=np.float32)
    tf, target, imp = _genie3(X, list(gene_names), list(tf_names),
                              n_estimators, max_features, min_samples_leaf, seed)
    return _to_frame(tf, target, imp, as_frame)


def grnboost2(expr, gene_names, tf_names, n_estimators=5000, learning_rate=0.01,
              max_depth=3, max_features="0.1", subsample=0.9, min_samples_leaf=1,
              early_stop_window=25, seed=42, as_frame=True):
    """GRN inference with GRNBoost2 (gradient-boosted trees).

    Uses out-of-bag early stopping (the GRNBoost2 regularizer): `n_estimators`
    is an upper bound and boosting stops per target once the trailing-window mean
    of OOB improvement turns negative. Set ``early_stop_window=0`` to disable and
    always build ``n_estimators`` trees. Early stopping needs ``subsample < 1``.
    """
    import numpy as np

    X = np.ascontiguousarray(expr, dtype=np.float32)
    tf, target, imp = _grnboost2(X, list(gene_names), list(tf_names), n_estimators,
                                 learning_rate, max_depth, str(max_features),
                                 subsample, min_samples_leaf, early_stop_window, seed)
    return _to_frame(tf, target, imp, as_frame)


def _as_db(d):
    if isinstance(d, RankingDb):
        return d
    if isinstance(d, (tuple, list)):
        return RankingDb(str(d[0]), str(d[1]))
    import os
    return RankingDb(str(d), os.path.basename(str(d)))


def ctx(adjacencies, expr, gene_names, dbs, motif_annotations, *,
        thresholds=None, top_n_targets=None, top_n_regulators=None, min_genes=20,
        rho_threshold=0.03, mask_dropouts=False, keep_only_activating=True,
        rank_threshold=5000, auc_threshold=0.05, nes_threshold=3.0,
        motif_similarity_fdr=0.001, orthologous_identity_threshold=0.0):
    """cisTarget step: prune GRN adjacencies to motif-supported regulons.

    adjacencies : the GRN output — a DataFrame with TF/target/importance columns
        (as returned by ``grnboost2``/``genie3``), or a ``(tf, target, importance)``
        tuple of equal-length sequences.
    dbs : ranking databases — ``RankingDb`` objects, ``(path, name)`` pairs, or
        plain feather paths (name derived from the filename).
    motif_annotations : path to the motif2TF ``.tbl`` snapshot.
    Returns a list of :class:`Regulon`.
    """
    import numpy as np

    X = np.ascontiguousarray(expr, dtype=np.float32)
    if hasattr(adjacencies, "columns"):
        tf = adjacencies["TF"].tolist()
        target = adjacencies["target"].tolist()
        imp = adjacencies["importance"].astype(float).tolist()
    else:
        tf, target, imp = adjacencies
        tf, target, imp = list(tf), list(target), [float(v) for v in imp]
    db_objs = [_as_db(d) for d in dbs]
    regs = _ctx(tf, target, imp, X, list(gene_names), db_objs, str(motif_annotations),
                thresholds, top_n_targets, top_n_regulators, min_genes, rho_threshold,
                mask_dropouts, keep_only_activating, rank_threshold, auc_threshold,
                nes_threshold, motif_similarity_fdr, orthologous_identity_threshold)
    return [Regulon(*r) for r in regs]


def aucell(expr, gene_names, regulons, auc_max_rank=None, as_frame=True):
    """Score regulon activity per cell.

    regulons : dict {regulon_name: [gene, ...]}
    Returns a (n_cells x n_regulons) DataFrame of AUC scores (or a numpy array).
    """
    import numpy as np

    X = np.ascontiguousarray(expr, dtype=np.float32)
    names = list(regulons.keys())
    genes = [list(regulons[n]) for n in names]
    reg_names, flat, n_cells, n_reg = _aucell(
        X, list(gene_names), names, genes, auc_max_rank
    )
    mat = np.asarray(flat, dtype=np.float32).reshape(n_cells, n_reg)
    if not as_frame:
        return mat, reg_names
    import pandas as pd

    return pd.DataFrame(mat, columns=reg_names)
