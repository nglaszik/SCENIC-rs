import numpy as np
from scipy.stats import spearmanr
from joblib import Parallel, delayed
from validate_genie3 import load_pbmc3k
from sklearn.ensemble import GradientBoostingRegressor
import scenic_rs

X, genes, tfs = load_pbmc3k(300, 200, 0)   # smaller for a deterministic full-feature run
n2i = {g:i for i,g in enumerate(genes)}; tfi=[n2i[t] for t in tfs]

def sk(off):
    def one(t):
        f=[i for i in tfi if i!=t]
        m=GradientBoostingRegressor(n_estimators=100,learning_rate=0.05,max_depth=3,
              subsample=1.0,max_features=None,random_state=t+off).fit(X[:,f],X[:,t])
        return [(genes[i],genes[t],float(w)) for i,w in zip(f,m.feature_importances_)]
    return [e for s in Parallel(n_jobs=-1)(delayed(one)(t) for t in range(len(genes))) for e in s]

def corr(a,b):
    da={(t,g):w for t,g,w in a}; db={(t,g):w for t,g,w in b}
    k=sorted(set(da)|set(db))
    return spearmanr([da.get(x,0) for x in k],[db.get(x,0) for x in k]).correlation

a,b=sk(0),sk(9973)
r=list(zip(*scenic_rs.grnboost2(X,genes,tfs,n_estimators=100,learning_rate=0.05,
       max_depth=3,subsample=1.0,max_features="all",as_frame=False)))
print(f"deterministic GBM (subsample=1, all features):")
print(f"  sklearn-vs-sklearn (ceiling) = {corr(a,b):.4f}")
print(f"  rust-vs-sklearn              = {corr(r,a):.4f}")
