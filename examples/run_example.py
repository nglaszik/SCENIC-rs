"""Smoke test: synthetic data where we know the true regulators."""
import numpy as np
from scenic_rs import genie3

rng = np.random.default_rng(0)
n_cells, n_genes = 400, 40
X = rng.normal(size=(n_cells, n_genes)).astype("float32")
genes = [f"g{i}" for i in range(n_genes)]
tfs = [f"g{i}" for i in range(5)]  # g0..g4 are the TFs

# plant known regulatory edges
X[:, 10] = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.1 * rng.normal(size=n_cells)  # g10 <- g0, g1
X[:, 11] = 1.5 * X[:, 3] + 0.1 * rng.normal(size=n_cells)                  # g11 <- g3

adj = genie3(X, genes, tfs, n_estimators=200)
print("top edges overall:")
print(adj.head(8).to_string(index=False))
print("\ntarget g10 (expect g0, g1 on top):")
print(adj[adj.target == "g10"].head().to_string(index=False))
print("\ntarget g11 (expect g3 on top):")
print(adj[adj.target == "g11"].head().to_string(index=False))
