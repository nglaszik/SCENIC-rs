"""scenic-rs: fast Rust backend for the SCENIC pipeline.

GRN inference (GENIE3 / GRNBoost2) and regulon scoring (AUCell).
"""
from ._core import genie3 as _genie3
from ._core import grnboost2 as _grnboost2
from ._core import aucell as _aucell

__all__ = ["genie3", "grnboost2", "aucell"]


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
