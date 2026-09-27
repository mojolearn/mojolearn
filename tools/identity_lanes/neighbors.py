# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE NEIGHBORS LANE'S IDENTITY LANES (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md).
#
# Owned by the `neighbors` expansion lane. tools/identity_break.py executes this
# file in ITS OWN namespace after every helper and registry exists
# (`_load_lane_fragments`), so a lane here is written exactly like a lane there:
#
#     @lane("neighbors-example")
#     def _(ml, X, yc, yr, Xh=None):
#         m = ml.Example().fit(X)
#         return _fit(dict(out=_h(m.transform(X[:256]))), m, lambda e: (e.transform(Xh[:256]),))
#
#     _batch_decl(_rows_calls("transform", sl=slice(0, 256)), "neighbors-example")
#
# Rules the loader enforces: spell every lane name literally; add only your
# own lanes (to LANES and the per-lane registries); rebind no existing name;
# prefix your own helpers with `_neighbors_`. No imports are needed: np, _h,
# _fit, _rows_calls and the rest are this module's.


# Row counts: the kernel lanes form an n x n matrix and the eigen/SMO solves
# are sequential, so they take the first 256 rows; the neighbor lanes 512.

@lane("x-neighbors-lof")
def _(ml, X, yc, yr, Xh=None):
    m = ml.LocalOutlierFactor(n_neighbors=10, contamination=0.1)
    labels = m.fit_predict(X[:512])
    nov = ml.LocalOutlierFactor(n_neighbors=10, novelty=True).fit(X[:512])
    return _fit(dict(nof=_h(m.negative_outlier_factor_), labels=_h(labels), offset=_h(np.float32(m.offset_))),
                nov, lambda e: (e.score_samples(Xh[:256]), e.predict(Xh[:256])))


@lane("x-neighbors-nearest-centroid")
def _(ml, X, yc, yr, Xh=None):
    y3 = (yc[:512] + (X[:512, 5] > 0).astype(np.int32)).astype(np.int32)
    m = ml.NearestCentroid().fit(X[:512], y3)
    s = ml.NearestCentroid(shrink_threshold=0.2, priors="empirical").fit(X[:512], y3)
    md = ml.NearestCentroid(metric="manhattan").fit(X[:512], y3)
    return _fit(dict(centroids=_h(m.centroids_), std=_h(m.within_class_std_dev_), predict=_h(m.predict(X[:512])),
                     shrunk=_h(s.centroids_), shrunk_predict=_h(s.predict(X[:512])),
                     proba=_h(s.predict_proba(X[:256])), manhattan=_h(md.centroids_, md.predict(X[:512])),
                     dev=_h(s.deviations_), dev_unshrunk=_h(m.deviations_), log_proba=_h(s.predict_log_proba(X[:256]))),
                s, lambda e: (e.predict(Xh[:256]), e.decision_function(Xh[:256])))


@lane("x-neighbors-ocsvm")
def _(ml, X, yc, yr, Xh=None):
    m = ml.OneClassSVM(nu=0.2).fit(X[:256])
    sw = (0.5 + 0.5 * (np.arange(256) % 4)).astype(np.float64)
    sw[9] = 0.0                                        # a zero weight drops the sample, as libsvm
    w = ml.OneClassSVM(nu=0.2).fit(X[:256], sample_weight=sw)
    return _fit(dict(dual=_h(m.dual_coef_), support=_h(m.support_), intercept=_h(m.intercept_),
                     decision=_h(m.decision_function(X[:256])),
                     w_dual=_h(w.dual_coef_), w_support=_h(w.support_), w_decision=_h(w.decision_function(X[:256]))),
                m, lambda e: (e.decision_function(Xh[:256]), e.predict(Xh[:256])))


@lane("x-neighbors-kpca")
def _(ml, X, yc, yr, Xh=None):
    m = ml.KernelPCA(n_components=4, kernel="rbf")
    Z = m.fit_transform(X[:256])
    return _fit(dict(z=_h(Z), eigenvalues=_h(m.eigenvalues_)), m, lambda e: (e.transform(Xh[:128]),))


@lane("x-neighbors-poly-sketch")
def _(ml, X, yc, yr, Xh=None):
    m = ml.PolynomialCountSketch(degree=3, gamma=0.5, coef0=1.0, n_components=64, random_state=3).fit(X[:512])
    return _fit(dict(z=_h(m.transform(X[:512])), idx=_h(m.indexHash_), bits=_h(m.bitHash_)),
                m, lambda e: (e.transform(Xh[:256]),))


@lane("x-neighbors-additive-chi2")
def _(ml, X, yc, yr, Xh=None):
    A = np.abs(X[:512]).astype(np.float32)
    A[::7, 2] = np.float32(0.0)                         # zeros take the masked branch
    m = ml.AdditiveChi2Sampler(sample_steps=3).fit(A)
    return _fit(dict(z=_h(m.transform(A))), m, lambda e: (e.transform(np.abs(Xh[:256]).astype(np.float32)),))


@lane("x-neighbors-skewed-chi2")
def _(ml, X, yc, yr, Xh=None):
    A = np.abs(X[:512]).astype(np.float32)
    m = ml.SkewedChi2Sampler(skewedness=0.5, n_components=64, random_state=5).fit(A)
    return _fit(dict(z=_h(m.transform(A)), w=_h(m.random_weights_), off=_h(m.random_offset_)),
                m, lambda e: (e.transform(np.abs(Xh[:256]).astype(np.float32)),))


def _neighbors_semi_labels(yc, X):
    y = (yc[:256] + (X[:256, 6] > 0.5).astype(np.int32)).astype(np.int64)
    y[1::2] = -1
    return y


@lane("x-neighbors-label-propagation")
def _(ml, X, yc, yr, Xh=None):
    y = _neighbors_semi_labels(yc, X)
    m = ml.LabelPropagation(kernel="rbf", gamma=0.05, max_iter=200).fit(X[:256], y)
    k = ml.LabelPropagation(kernel="knn", n_neighbors=7).fit(X[:256], y)
    return _fit(dict(ld=_h(m.label_distributions_), tr=_h(m.transduction_), it=_h(np.int64(m.n_iter_)),
                     knn_ld=_h(k.label_distributions_), knn_proba=_h(k.predict_proba(X[:128]))),
                m, lambda e: (e.predict_proba(Xh[:128]), e.predict(Xh[:128])))


@lane("x-neighbors-label-spreading")
def _(ml, X, yc, yr, Xh=None):
    y = _neighbors_semi_labels(yc, X)
    m = ml.LabelSpreading(kernel="rbf", gamma=0.05, alpha=0.3).fit(X[:256], y)
    k = ml.LabelSpreading(kernel="knn", n_neighbors=7).fit(X[:256], y)
    return _fit(dict(ld=_h(m.label_distributions_), tr=_h(m.transduction_), it=_h(np.int64(m.n_iter_)),
                     knn_ld=_h(k.label_distributions_), knn_proba=_h(k.predict_proba(X[:128]))),
                m, lambda e: (e.predict_proba(Xh[:128]), e.predict(Xh[:128])))


def _neighbors_holes(A):
    A = np.array(A, dtype=np.float32, copy=True)
    n, d = A.shape
    for i in range(0, n, 3):
        A[i, (i * 7) % d] = np.nan
    A[5, :] = np.nan                                    # a row with no coordinate: the column-mean branch
    return A


@lane("x-neighbors-knn-imputer")
def _(ml, X, yc, yr, Xh=None):
    A = _neighbors_holes(X[:512])
    m = ml.KNNImputer(n_neighbors=5).fit(A)
    w = ml.KNNImputer(n_neighbors=4, weights="distance", add_indicator=True).fit(A)
    A7 = np.where(np.isnan(A), np.float32(-7.0), A).astype(np.float32)
    mv = ml.KNNImputer(n_neighbors=5, missing_values=-7.0).fit(A7)
    return _fit(dict(u=_h(m.transform(A)), w=_h(w.transform(A)), mv=_h(mv.transform(A7))),
                w, lambda e: (e.transform(_neighbors_holes(Xh[:256])),))


def _neighbors_graph(X, n=128, directed=False, ring=True):
    """A graph built from exact comparisons of fixture values only: an edge
    joins two rows whose column-3 value falls in the same half-unit bin,
    weighted 1 + (i * j) % 3; `ring` adds i -> i+1 so the components join."""
    q = np.floor(X[:n, 3] * np.float32(2)).astype(np.int64)
    i = np.arange(n)
    A = ((q[:, None] == q[None, :]) & (i[:, None] != i[None, :])).astype(np.float32)
    A *= (1 + (i[:, None] * i[None, :]) % 3).astype(np.float32)
    if ring:
        A[i[:-1], i[1:]] = np.float32(1)
        if not directed:
            A[i[1:], i[:-1]] = np.float32(1)
    if directed:
        A = np.triu(A).astype(np.float32)
        A[::17] = np.float32(0)                          # dangling rows
    return np.ascontiguousarray(A, dtype=np.float32)


@lane("x-neighbors-pagerank")
def _(ml, X, yc, yr, Xh=None):
    A = _neighbors_graph(X, directed=True)
    m = ml.PageRank(alpha=0.85, tol=1e-6).fit(A)
    pers = (1 + np.arange(128) % 5).astype(np.float32)
    p = ml.PageRank(alpha=0.7, personalization=pers).fit(A)
    dg = ml.PageRank(alpha=0.85, dangling=(1 + np.arange(128) % 7).astype(np.float32),
                     nstart=(1 + np.arange(128) % 3).astype(np.float32)).fit(A)
    return _fit(dict(pr=_h(m.pagerank_), it=_h(np.int64(m.n_iter_)), pers=_h(p.pagerank_),
                     dangling=_h(dg.pagerank_), dangling_it=_h(np.int64(dg.n_iter_))), m,
                lambda e: (ml.PageRank(alpha=0.85).fit(_neighbors_graph(Xh, directed=True)).pagerank_,))


@lane("x-neighbors-connected-components")
def _(ml, X, yc, yr, Xh=None):
    k, lab = ml.connected_components(_neighbors_graph(X, ring=False), directed=False)
    kd, labd = ml.connected_components(_neighbors_graph(X, directed=True, ring=False), directed=True, connection="weak")
    return _fit(dict(k=_h(np.int64(k)), lab=_h(lab), kd=_h(np.int64(kd)), labd=_h(labd)))


@lane("x-neighbors-louvain")
def _(ml, X, yc, yr, Xh=None):
    A = _neighbors_graph(X)
    m = ml.Louvain(resolution=1.0).fit(A)
    r = ml.Louvain(resolution=0.5, max_level=1).fit(A)
    return _fit(dict(lab=_h(m.labels_), q=_h(np.float32(m.modularity_)), lv=_h(np.int64(m.n_levels_)),
                     lab_r=_h(r.labels_), q_r=_h(np.float32(r.modularity_))))


@lane("x-neighbors-svgp")
def _(ml, X, yc, yr, Xh=None):
    m = ml.SVGP(n_inducing=24, kernel_variance=2.0, lengthscale=3.0, noise_variance=0.5).fit(X[:512], yr[:512])
    mean, var = m.predict_f(X[:256])
    return _fit(dict(mean=_h(mean), var=_h(var), qmu=_h(m.q_mu_), qsqrt=_h(m.q_sqrt_), elbo=_h(np.float32(m.elbo_))),
                m, lambda e: e.predict_y(Xh[:256]))


_batch_decl(_rows_calls("score_samples", "predict", sl=slice(0, 256)), "x-neighbors-lof")
_batch_decl(_rows_calls("predict", "decision_function", "predict_proba", sl=slice(0, 256)), "x-neighbors-nearest-centroid")
_batch_decl(_rows_calls("decision_function", "predict", sl=slice(0, 256)), "x-neighbors-ocsvm")
_batch_decl(_rows_calls("transform", sl=slice(0, 128)), "x-neighbors-kpca")

_batch_decl(_rows_calls("transform", sl=slice(0, 256)), "x-neighbors-poly-sketch")
_batch_decl(_rows_calls("transform", sl=slice(0, 256), prep=lambda Xh: np.abs(Xh).astype(np.float32)),
            "x-neighbors-additive-chi2", "x-neighbors-skewed-chi2")
_batch_decl(_rows_calls("predict_proba", "predict", sl=slice(0, 128)),
            "x-neighbors-label-propagation", "x-neighbors-label-spreading")
_batch_decl(_rows_calls("transform", sl=slice(0, 256), prep=_neighbors_holes), "x-neighbors-knn-imputer")
_batch_decl(_rows_calls("predict", sl=slice(0, 256)), "x-neighbors-svgp")
