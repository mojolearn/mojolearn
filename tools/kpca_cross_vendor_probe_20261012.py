# 2026-10-12 cross-vendor probe: per-estimator digests of the x_neighbors repeat test (KernelPCA rbf 96x6 differed Metal vs host); run once per vendor and compare.
import hashlib, numpy as np, mojolearn as ml, sys
rng = np.random.default_rng(0); X = rng.standard_normal((96, 6)).astype(np.float32); y = (rng.random(96) > 0.5).astype(np.int32)
Xh = rng.standard_normal((40, 6)).astype(np.float32)
A = (rng.random((50, 50)) < 0.1).astype(np.float32); A = np.maximum(A, A.T); np.fill_diagonal(A, 0)
H = X.copy(); H[::7, 2] = np.nan
lab = np.where(np.arange(96) % 3 == 0, y, -1)
def h(a):
    a = np.ascontiguousarray(np.asarray(a)); return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()[:12]
calls = {
 'lof.score_samples': lambda: ml.LocalOutlierFactor(n_neighbors=5, novelty=True).fit(X).score_samples(Xh),
 'lof.fit_predict': lambda: ml.LocalOutlierFactor(n_neighbors=5).fit_predict(X),
 'nc.predict_proba': lambda: ml.NearestCentroid(shrink_threshold=0.2).fit(X, y).predict_proba(Xh),
 'ocsvm.decision': lambda: ml.OneClassSVM(nu=0.3).fit(X).decision_function(Xh),
 'kpca': lambda: ml.KernelPCA(n_components=3, kernel="rbf").fit(X).transform(Xh),
 'polysketch': lambda: ml.PolynomialCountSketch(n_components=16, random_state=1).fit(X).transform(Xh),
 'addchi2': lambda: ml.AdditiveChi2Sampler().fit(np.abs(X)).transform(np.abs(Xh)),
 'skewchi2': lambda: ml.SkewedChi2Sampler(n_components=16, random_state=1).fit(np.abs(X)).transform(np.abs(Xh)),
 'labelprop': lambda: ml.LabelPropagation().fit(X, lab).predict_proba(Xh),
 'labelspread': lambda: ml.LabelSpreading().fit(X, lab).predict_proba(Xh),
 'knnimputer': lambda: ml.KNNImputer(n_neighbors=3).fit(H).transform(H),
 'pagerank': lambda: ml.PageRank().fit(A).pagerank_,
 'concomp': lambda: ml.connected_components(A, directed=False)[1],
 'louvain': lambda: ml.Louvain().fit(A).labels_,
 'svgp': lambda: ml.SVGP(n_inducing=8).fit(X, X[:, 0]).predict_f(Xh)[0],
}
for k, f in calls.items():
    try: print(k, h(f()))
    except Exception as e: print(k, 'ERR', type(e).__name__, str(e)[:60])
