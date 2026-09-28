# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""bench/decomp_speed.py -- the decomp lane's speed table (lane/algos-decomp).
Every decomp algorithm, timed on this box: warm call first, then one timed
call; per-binding-entry profile (count / seconds) for the x_decomp kit.
env: MOJOLEARN_NUMERIC_MODE (identical|fast), N (rows, default 1M),
ONLY (comma list of names), DATA (higgs|taxi from R2 under
/root/datasets, or synth: a seeded N x 28 Gaussian with correlated columns,
for boxes without the staged files)."""
import numpy as np, sys, time, collections, os, warnings
sys.path.insert(0, os.environ.get("ML_PY", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python")))
warnings.filterwarnings("ignore")
import mojolearn as ml
import mojolearn._expansion_decomp as ed
prof = collections.defaultdict(lambda: [0, 0.0])
_orig_init = ed._Kit.__init__
class _Prox:
    def __init__(self, b): self._b = b
    def __getattr__(self, name):
        f = getattr(self._b, name)
        if not callable(f): return f
        def g(*a, **k):
            t = time.perf_counter(); r = f(*a, **k)
            p = prof[name]; p[0] += 1; p[1] += time.perf_counter() - t; return r
        return g
def _init(self, mode, binding=None):
    _orig_init(self, mode, binding)
    if not isinstance(self.b, _Prox): self.b = _Prox(self.b)
ed._Kit.__init__ = _init
DATA = os.environ.get("DATA", "higgs")
N = int(os.environ.get("N", "1000000"))
if DATA == "synth":
    _g = np.random.default_rng(12345)
    X = (_g.standard_normal((N, 28), dtype=np.float32) @ _g.standard_normal((28, 28), dtype=np.float32)).astype(np.float32)
else:
    d = np.load(f"/root/datasets/gbm-bench/{DATA}/{DATA}_speed.npz")
    key = [k for k in d.files if d[k].ndim == 2][0]
    X = np.ascontiguousarray(d[key][:N].astype(np.float32))
X = np.nan_to_num(X)
X = (X - X.mean(0)) / (X.std(0) + 1e-6)
X = X.astype(np.float32)
Xp = np.abs(X).astype(np.float32)
ONLY = set(filter(None, os.environ.get("ONLY", "").split(",")))
print("X", X.shape, DATA, "mode", os.environ.get("MOJOLEARN_NUMERIC_MODE", "identical"), flush=True)
def run(name, fn, warm=None):
    if ONLY and name.split("(")[0] not in ONLY: return
    try:
        (warm or fn)()
        prof.clear(); t = time.perf_counter(); fn(); dt = time.perf_counter() - t
        py = dt - sum(v[1] for v in prof.values())
        top = sorted(prof.items(), key=lambda kv: -kv[1][1])[:5]
        print(f"{name:34s} {dt:8.3f}s  py/other {py:6.3f}s  " + "  ".join(f"{k.replace('x_decomp_','')}:{v[0]}x/{v[1]:.3f}s" for k, v in top), flush=True)
    except Exception as e:
        print(f"{name:34s} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
s = lambda n: X[:n]
w = lambda n: (lambda f: (lambda: f(n)))
# ---- linear in n (1M rows)
run("PCA(full,5)", lambda: ml.PCA(n_components=5).fit(X))
run("PCA(randomized,5)", lambda: ml.PCA(n_components=5, svd_solver="randomized").fit(X))
run("TruncatedSVD(5)", lambda: ml.TruncatedSVD(n_components=5).fit(X))
run("TruncatedSVD(randomized,5)", lambda: ml.TruncatedSVD(n_components=5, algorithm="randomized").fit(X))
run("IncrementalPCA(5,bs=100k)", lambda: ml.IncrementalPCA(n_components=5, batch_size=100000).fit(X))
run("GaussianRandomProjection(8)", lambda: ml.GaussianRandomProjection(n_components=8, random_state=0).fit(X).transform(X))
run("SparseRandomProjection(8)", lambda: ml.SparseRandomProjection(n_components=8, random_state=0).fit(X).transform(X))
run("NMF(mu,5,20it)", lambda: ml.NMF(n_components=5, solver="mu", max_iter=20, tol=0).fit(Xp))
run("NMF(cd,5,20it)", lambda: ml.NMF(n_components=5, max_iter=20, tol=0).fit(Xp))
run("FastICA(5,20it)", lambda: ml.FastICA(n_components=5, max_iter=20, tol=0, random_state=0).fit(X))
run("FactorAnalysis(5,20it)", lambda: ml.FactorAnalysis(n_components=5, max_iter=20, tol=0).fit(X))
run("lstsq", lambda: ml.lstsq(X[:, :-1], X[:, -1]))
run("randomized_svd(5)", lambda: ml.randomized_svd(X, 5, random_state=0))
run("PLSRegression(3)", lambda: ml.PLSRegression(n_components=3).fit(X[:, :-2], X[:, -2:]))
run("CCA(2)", lambda: ml.CCA(n_components=2).fit(X[:, :-2], X[:, -2:]))
run("MinCovDet", lambda: ml.MinCovDet(random_state=0).fit(X[:, :8]))
run("linalg.qr(reduced)", lambda: ml.linalg.qr(X))
run("linalg.svd", lambda: ml.linalg.svd(X, full_matrices=False))
run("solve(512)", lambda: ml.solve(X[:512, :1].repeat(512, 1) + np.eye(512, dtype=np.float32) * 10, X[:512, :4]))
# ---- heavier per row
n2 = int(os.environ.get("N2", "100000"))
run("SparsePCA(5,10it)", lambda: ml.SparsePCA(n_components=5, max_iter=10, random_state=0).fit(X[:n2]))
run("DictionaryLearning(8,10it)", lambda: ml.DictionaryLearning(n_components=8, max_iter=10, random_state=0).fit(X[:n2]))
run("MiniBatchDictionaryLearning(8)", lambda: ml.MiniBatchDictionaryLearning(n_components=8, max_iter=3, random_state=0).fit(X[:n2]))
C = np.floor(Xp[:n2] * 3).astype(np.float32)
run("LatentDirichletAllocation(5,10it)", lambda: ml.LatentDirichletAllocation(n_components=5, max_iter=10, random_state=0).fit(C))
R = (np.random.default_rng(0).random((20000, 2000)) < 0.01).astype(np.float32)
run("ALS(32f,5it)", lambda: ml.AlternatingLeastSquares(factors=32, iterations=5, random_state=0).fit(R))
run("ALS(32f,5it,cg)", lambda: ml.AlternatingLeastSquares(factors=32, iterations=5, use_cg=True, random_state=0).fit(R))
n3 = int(os.environ.get("N3", "10000"))
run("Isomap(10nn)", lambda: ml.Isomap(n_neighbors=10).fit(X[:n3]))
run("LocallyLinearEmbedding(10nn)", lambda: ml.LocallyLinearEmbedding(n_neighbors=10).fit(X[:n3]))
run("ClassicalMDS", lambda: ml.ClassicalMDS().fit(X[:n3]))
run("MDS(5it)", lambda: ml.MDS(max_iter=5, n_init=1, random_state=0).fit(X[:min(n3, 3000)]))
n4 = int(os.environ.get("N4", "100000"))
run("SpectralEmbedding(knn)", lambda: ml.SpectralEmbedding(n_components=2, random_state=0).fit(X[:n4]))
run("SpectralEmbedding(rbf)", lambda: ml.SpectralEmbedding(n_components=2, affinity="rbf", random_state=0).fit(X[:min(n4, 5000)]))
run("UMAP(default)", lambda: ml.UMAP(random_state=0).fit(X[:n4]))
run("UMAP(c5,manhattan)", lambda: ml.UMAP(random_state=0, n_components=5, metric="manhattan").fit(X[:n4]))
