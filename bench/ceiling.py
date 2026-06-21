"""Decompose the concordance gap: how much is irreducible stochastic noise
(sklearn vs sklearn, two seeds) vs a real implementation gap (rust vs sklearn)?"""
import numpy as np
from scipy.stats import spearmanr
from joblib import Parallel, delayed
from validate_genie3 import load_pbmc3k
import scenic_rs


def corr(ea, eb):
    da = {(t, g): w for t, g, w in ea}
    db = {(t, g): w for t, g, w in eb}
    keys = sorted(set(da) | set(db))
    return spearmanr(
        [da.get(k, 0.0) for k in keys], [db.get(k, 0.0) for k in keys]
    ).correlation


def rf(X, genes, tfs, n_estimators, off):
    from sklearn.ensemble import RandomForestRegressor
    n2i = {g: i for i, g in enumerate(genes)}
    tfi = [n2i[t] for t in tfs]
    def one(t):
        f = [i for i in tfi if i != t]
        m = RandomForestRegressor(n_estimators=n_estimators, max_features="sqrt",
                                  n_jobs=1, random_state=t + off).fit(X[:, f], X[:, t])
        return [(genes[i], genes[t], float(w)) for i, w in zip(f, m.feature_importances_)]
    return [e for s in Parallel(n_jobs=-1)(delayed(one)(t) for t in range(len(genes))) for e in s]


def gbm(X, genes, tfs, off):
    from sklearn.ensemble import GradientBoostingRegressor
    n2i = {g: i for i, g in enumerate(genes)}
    tfi = [n2i[t] for t in tfs]
    def one(t):
        f = [i for i in tfi if i != t]
        m = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3,
                                      subsample=0.9, max_features=0.1,
                                      random_state=t + off).fit(X[:, f], X[:, t])
        return [(genes[i], genes[t], float(w)) for i, w in zip(f, m.feature_importances_)]
    return [e for s in Parallel(n_jobs=-1)(delayed(one)(t) for t in range(len(genes))) for e in s]


if __name__ == "__main__":
    X, genes, tfs = load_pbmc3k(500, 300, 0)
    print(f"data: {X.shape[0]} cells x {len(genes)} genes ({len(tfs)} TFs)\n")

    for ntrees in (100, 500):
        a = rf(X, genes, tfs, ntrees, 0)
        b = rf(X, genes, tfs, ntrees, 9973)
        r = list(zip(*scenic_rs.genie3(X, genes, tfs, n_estimators=ntrees, as_frame=False)))
        print(f"GENIE3 @ {ntrees} trees:  sklearn-vs-sklearn (ceiling) = {corr(a, b):.3f}   "
              f"rust-vs-sklearn = {corr(r, a):.3f}")

    ga = gbm(X, genes, tfs, 0)
    gb = gbm(X, genes, tfs, 9973)
    gr = list(zip(*scenic_rs.grnboost2(X, genes, tfs, n_estimators=200, learning_rate=0.05,
                                       max_depth=3, subsample=0.9, max_features="0.1", as_frame=False)))
    print(f"GRNBoost2 @ 200:        sklearn-vs-sklearn (ceiling) = {corr(ga, gb):.3f}   "
          f"rust-vs-sklearn = {corr(gr, ga):.3f}")
