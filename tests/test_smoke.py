"""Smoke + regression tests for the GRN / AUCell entry points.

These run on the committed pbmc3k fixtures with no external download. `ctx`
(which needs a cisTarget ranking DB) is covered in test_ctx.py against a small
synthetic DB fixture.

Run with: `pytest` (after `maturin develop`).
"""

from pathlib import Path

import numpy as np
import pytest
import scenic_rs as srs

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "pbmc3k_prep_300_200_0.npz"


@pytest.fixture(scope="module")
def data():
    if not FIXTURE.exists():
        pytest.skip(f"fixture not found: {FIXTURE}")
    d = np.load(FIXTURE, allow_pickle=True)
    return d["X"], list(d["genes"]), list(d["tfs"])


def test_genie3_schema_and_determinism(data):
    X, genes, tfs = data
    a = srs.genie3(X, genes, tfs, n_estimators=50, seed=42)
    b = srs.genie3(X, genes, tfs, n_estimators=50, seed=42)

    assert set(a.columns) == {"TF", "target", "importance"}
    assert len(a) > 0
    assert a["TF"].isin(tfs).all()
    assert a["target"].isin(genes).all()
    assert np.isfinite(a["importance"].to_numpy()).all()

    # GENIE3 is seed-deterministic (verified run-to-run on this fixture).
    ka = a.sort_values(["TF", "target"]).reset_index(drop=True)
    kb = b.sort_values(["TF", "target"]).reset_index(drop=True)
    np.testing.assert_allclose(
        ka["importance"].to_numpy(), kb["importance"].to_numpy()
    )


def test_grnboost2_schema_and_finiteness(data):
    X, genes, tfs = data
    adj = srs.grnboost2(X, genes, tfs, n_estimators=100, seed=0)

    assert set(adj.columns) == {"TF", "target", "importance"}
    assert len(adj) > 0
    assert adj["TF"].isin(tfs).all()
    assert adj["target"].isin(genes).all()
    imp = adj["importance"].to_numpy()
    assert np.isfinite(imp).all()
    assert (imp >= 0).all()


def test_aucell_scores_are_bounded(data):
    X, genes, tfs = data
    # Build a few regulons from a GRNBoost2 run, then score them per cell.
    adj = srs.grnboost2(X, genes, tfs, n_estimators=100, seed=0)
    regulons = {
        tf: adj[adj["TF"] == tf]["target"].tolist()[:20]
        for tf in adj["TF"].unique()[:5]
    }

    auc = srs.aucell(X, genes, regulons)
    arr = np.asarray(auc.values if hasattr(auc, "values") else auc)

    assert arr.shape == (X.shape[0], len(regulons))
    assert np.isfinite(arr).all()
    # AUC scores are bounded to [0, 1] by construction.
    assert arr.min() >= 0.0 and arr.max() <= 1.0

    # Determinism: same input -> same scores.
    auc2 = srs.aucell(X, genes, regulons)
    arr2 = np.asarray(auc2.values if hasattr(auc2, "values") else auc2)
    np.testing.assert_allclose(arr, arr2)
