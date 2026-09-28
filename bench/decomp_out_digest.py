# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""bench/decomp_out_digest.py -- one sha256 per decomp algorithm's fitted
output bytes at small seeded shapes (lane/decomp-apple): a before and an
after commit print the same lines on one box when an IDENTICAL change kept
its bits. MOJOLEARN_VENDOR=cpu gives the CPU column's lines (they must match
the GPU's too). env: ONLY (comma list of names)."""
import os, sys, hashlib, warnings
sys.path.insert(0, os.environ.get("ML_PY", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python")))
warnings.filterwarnings("ignore")
import numpy as np
import mojolearn as ml

rng = np.random.default_rng(3)
X = (rng.standard_normal((3000, 9)) @ rng.standard_normal((9, 9))).astype(np.float32)
Xp = np.abs(X)
C = np.floor(Xp * 3).astype(np.float32)
R = (rng.random((300, 120)) < 0.1).astype(np.float32)
T = (rng.standard_normal((5000, 12))).astype(np.float32)
cases = {
    "PCA-rand": lambda: ml.PCA(n_components=3, svd_solver="randomized", random_state=0).fit(X).components_,
    "TSVD-rand": lambda: ml.TruncatedSVD(n_components=3, algorithm="randomized", random_state=0).fit_transform(X),
    "IPCA": lambda: ml.IncrementalPCA(n_components=3, batch_size=500).fit(X).components_,
    "GRP": lambda: ml.GaussianRandomProjection(n_components=4, random_state=0).fit(X).transform(X),
    "SRP": lambda: ml.SparseRandomProjection(n_components=4, random_state=0).fit(X).transform(X),
    "NMF-mu": lambda: ml.NMF(n_components=3, solver="mu", max_iter=30, random_state=0).fit(Xp).components_,
    "NMF-cd": lambda: ml.NMF(n_components=3, max_iter=30, random_state=0).fit(Xp).components_,
    "FastICA": lambda: ml.FastICA(n_components=3, random_state=0, max_iter=50).fit(X).components_,
    "FA": lambda: ml.FactorAnalysis(n_components=3).fit(X).components_,
    "lstsq": lambda: ml.lstsq(X[:, :-1], X[:, -1])[0],
    "rsvd": lambda: np.concatenate([np.ravel(a) for a in ml.randomized_svd(X, 3, random_state=0)]),
    "PLS": lambda: ml.PLSRegression(n_components=2).fit(X[:, :-2], X[:, -2:]).coef_,
    "CCA": lambda: ml.CCA(n_components=2).fit(X[:, :-2], X[:, -2:]).x_weights_,
    "SparsePCA": lambda: ml.SparsePCA(n_components=3, max_iter=5, random_state=0).fit(X[:400]).components_,
    "DictL": lambda: ml.DictionaryLearning(n_components=4, max_iter=5, random_state=0).fit(X[:400]).components_,
    "MBDictL": lambda: ml.MiniBatchDictionaryLearning(n_components=4, max_iter=2, random_state=0).fit(X[:400]).components_,
    "LDA": lambda: ml.LatentDirichletAllocation(n_components=3, max_iter=5, random_state=0).fit(C[:400]).components_,
    "ALS": lambda: ml.AlternatingLeastSquares(factors=4, iterations=3, random_state=0).fit(R).user_factors,
    "Isomap": lambda: ml.Isomap(n_neighbors=6).fit_transform(X[:300]),
    "LLE": lambda: ml.LocallyLinearEmbedding(n_neighbors=8).fit_transform(X[:200]),
    "CMDS": lambda: ml.ClassicalMDS().fit_transform(X[:200]),
    "MDS": lambda: ml.MDS(max_iter=5, n_init=1, random_state=0).fit_transform(X[:150]),
    "MinCovDet": lambda: ml.MinCovDet(random_state=0).fit(X[:600, :4]).covariance_,
    "solve": lambda: ml.solve(X[:9, :9] + np.eye(9, dtype=np.float32) * 5, X[:9, :2]),
    "qr": lambda: np.concatenate([np.ravel(np.asarray(a)) for a in ml.linalg.qr(T)]),
    "svd": lambda: np.concatenate([np.ravel(np.asarray(a)) for a in ml.linalg.svd(T, full_matrices=False)]),
    "SE-rbf": lambda: ml.SpectralEmbedding(n_components=2, affinity="rbf", random_state=0).fit_transform(X[:300]),
}
ONLY = set(filter(None, os.environ.get("ONLY", "").split(",")))
col = "cpu" if os.environ.get("MOJOLEARN_VENDOR") == "cpu" else "gpu"
for name, f in cases.items():
    if ONLY and name not in ONLY:
        continue
    try:
        a = np.ascontiguousarray(np.asarray(f(), dtype=np.float32))
        print(f"OUT {col} {name:10s} {a.shape} {hashlib.sha256(a.tobytes()).hexdigest()[:20]}", flush=True)
    except Exception as e:
        print(f"OUT {col} {name:10s} FAILED {type(e).__name__}: {str(e)[:160]}", flush=True)
