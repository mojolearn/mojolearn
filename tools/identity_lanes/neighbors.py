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


def _neighbors_chain(X, n=128):
    """Each bin of `_neighbors_graph` as a directed CHAIN, every node pointing
    only at the next node of its bin (i -> j, j > i): a weak component's
    smallest label must travel the whole chain one hop per step, so the
    propagation runs many rounds and reads edges in their reverse direction."""
    q = np.floor(X[:n, 3] * np.float32(2)).astype(np.int64)
    A = np.zeros((n, n), dtype=np.float32)
    last = {}
    for i in range(n):
        b = int(q[i])
        if b in last:
            A[last[b], i] = np.float32(1)
        last[b] = i
    return np.ascontiguousarray(A, dtype=np.float32)


@lane("x-neighbors-connected-components")
def _(ml, X, yc, yr, Xh=None):
    k, lab = ml.connected_components(_neighbors_graph(X, ring=False), directed=False)
    kd, labd = ml.connected_components(_neighbors_graph(X, directed=True, ring=False), directed=True, connection="weak")
    kc, labc = ml.connected_components(_neighbors_chain(X), directed=True, connection="weak")
    return _fit(dict(k=_h(np.int64(k)), lab=_h(lab), kd=_h(np.int64(kd)), labd=_h(labd),
                     kc=_h(np.int64(kc)), labc=_h(labc)))


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



@lane("x-neighbors-gamma-scale")
def _(ml, X, yc, yr, Xh=None):
    """gamma='scale' on the family's EXISTING estimators (SVC, SVR,
    RBFSampler): 1 / (n_features * X.var()) from the exact variance of the
    float32 cells, rounded once (_scale_gamma.scale_gamma, DEVIATION 870).
    The resolved gamma is hashed with each fit, so a host that read other
    gamma bits moves the train column even where the fit would not."""
    c = ml.SVC(C=1.0, kernel="rbf", gamma="scale", max_iter=200).fit(X[:512], yc[:512])
    r = ml.SVR(C=1.0, kernel="rbf", gamma="scale", epsilon=0.1, max_iter=200).fit(X[:512], yr[:512])
    f = ml.RBFSampler(gamma="scale", n_components=32, random_state=1).fit(X[:512])
    return _fit(dict(svc_decision=_h(c.decision_function(X[512:768])), svc_gamma=_h(np.float64(c._gamma)),
                     svr_predict=_h(r.predict(X[512:768])), svr_gamma=_h(np.float64(r._gamma)),
                     rbf_gamma=_h(np.float64(f._params[0])), rbf_transform=_h(f.transform(X[:256]))),
                c, lambda e: (e.decision_function(Xh[:256]), e.predict(Xh[:256])))


@lane("x-neighbors-svm-weights")
def _(ml, X, yc, yr, Xh=None):
    """sample_weight on SVC and SVR and class_weight on SVC: InitPenalty's
    weighted arm, each row's bound C * cw[y_i] * w_i formed in binary64 and
    rounded once to float32 on the host (_svm_impl._c_rows), then C_vec on
    the device and the per-index bound in smo_oracle_fit. Weights 0.5 to
    2.0 with a zero, so a bound pins an alpha at 0 and others cap it."""
    w = (0.5 + 0.5 * (np.arange(512) % 4)).astype(np.float64)
    w[7] = 0.0
    c = ml.SVC(C=1.0, kernel="rbf", max_iter=200).fit(X[:512], yc[:512], sample_weight=w)
    b = ml.SVC(C=1.0, kernel="rbf", max_iter=200, class_weight="balanced").fit(X[:512], yc[:512])
    d = ml.SVC(C=0.5, kernel="linear", max_iter=200, class_weight={yc[:512].min().item(): 2.0}).fit(X[:512], yc[:512], sample_weight=w)
    r = ml.SVR(C=1.0, kernel="rbf", epsilon=0.1, max_iter=200).fit(X[:512], yr[:512], sample_weight=w)
    return _fit(dict(w_dual=_h(c.dual_coef_), w_support=_h(c.support_), w_decision=_h(c.decision_function(X[512:768])),
                     bal_dual=_h(b.dual_coef_), bal_decision=_h(b.decision_function(X[512:768])),
                     dict_dual=_h(d.dual_coef_), dict_decision=_h(d.decision_function(X[512:768])),
                     svr_dual=_h(r.dual_coef_), svr_predict=_h(r.predict(X[512:768]))),
                c, lambda e: (e.decision_function(Xh[:256]), e.predict(Xh[:256])))


@lane("x-neighbors-svc-sigmoid")
def _(ml, X, yc, yr, Xh=None):
    """SVC(kernel='sigmoid'): the identical linear Gram, then tanh(gamma * K
    + coef0) cell by cell (kernel_methods' tanh_epilogue_kernel on the
    device, the same spelling in smo_oracle_fit on the host)."""
    m = ml.SVC(C=1.0, kernel="sigmoid", gamma=0.05, coef0=-0.5, max_iter=200).fit(X[:512], yc[:512])
    z = ml.SVC(C=0.5, kernel="sigmoid", gamma=0.02, max_iter=200).fit(X[:512], yc[:512])
    return _fit(dict(dual=_h(m.dual_coef_), support=_h(m.support_), decision=_h(m.decision_function(X[512:768])),
                     z_dual=_h(z.dual_coef_), z_decision=_h(z.decision_function(X[512:768]))),
                m, lambda e: (e.decision_function(Xh[:256]), e.predict(Xh[:256])))


@lane("x-neighbors-svr-kernels")
def _(ml, X, yc, yr, Xh=None):
    """SVR(kernel='poly') and SVR(kernel='sigmoid'): the SVC's linear Gram
    and epilogues (DEVIATION 1663's repeated product, identical_tanh), with
    degree and coef0 through the bindings' optional trailing params."""
    p = ml.SVR(C=1.0, kernel="poly", degree=2, gamma=0.05, coef0=1.0, epsilon=0.1, max_iter=200).fit(X[:512], yr[:512])
    s = ml.SVR(C=1.0, kernel="sigmoid", gamma=0.02, coef0=-0.5, epsilon=0.1, max_iter=200).fit(X[:512], yr[:512])
    return _fit(dict(p_dual=_h(p.dual_coef_), p_support=_h(p.support_), p_predict=_h(p.predict(X[512:768])),
                     s_dual=_h(s.dual_coef_), s_predict=_h(s.predict(X[512:768]))),
                p, lambda e: (e.predict(Xh[:256]),))

@lane("x-neighbors-svc-multiclass")
def _(ml, X, yc, yr, Xh=None):
    """SVC with four classes: one-vs-one over the binary solver (six pair
    machines on their rows, gamma on the whole X), scikit-learn's layout
    and orientation (_svm_impl.SVC._fit_ovo); decision_function 'ovo' and
    'ovr', predict by vote and with break_ties, class_weight and
    sample_weight cut to each pair's rows, a linear coef_ per pair."""
    y4 = (yc[:384] + 2 * (X[:384, 5] > 0).astype(np.int32)).astype(np.int32)
    w = (0.5 + 0.5 * (np.arange(384) % 4)).astype(np.float64)
    m = ml.SVC(C=1.0, kernel="rbf", gamma=0.05, max_iter=200).fit(X[:384], y4)
    r = ml.SVC(C=1.0, kernel="rbf", gamma="scale", max_iter=200, decision_function_shape="ovr",
               break_ties=True, class_weight="balanced").fit(X[:384], y4, sample_weight=w)
    li = ml.SVC(C=0.5, kernel="linear", max_iter=200).fit(X[:384], y4)
    return _fit(dict(dual=_h(m.dual_coef_), support=_h(m.support_), intercept=_h(m.intercept_),
                     n_support=_h(m.n_support_), ovo=_h(m.decision_function(X[384:640])),
                     predict=_h(m.predict(X[384:640])),
                     ovr_dual=_h(r.dual_coef_), ovr=_h(r.decision_function(X[384:640])),
                     ovr_predict=_h(r.predict(X[384:640])),
                     coef=_h(li.coef_), linear_predict=_h(li.predict(X[384:640]))),
                r, lambda e: (e.decision_function(Xh[:256]), e.predict(Xh[:256])))

def _neighbors_int_gram(A, B):
    """An integer-valued kernel matrix A B^T, exact in float32 under any
    summation order (every partial sum is an integer below 2^24), so the
    precomputed input is the same bits on every host's NumPy."""
    ai = np.clip(np.rint(A * 2.0), -6, 6).astype(np.int64)
    bi = np.clip(np.rint(B * 2.0), -6, 6).astype(np.int64)
    return (ai @ bi.T).astype(np.float32)


@lane("x-neighbors-krr-options")
def _(ml, X, yc, yr, Xh=None):
    """KernelRidge sample_weight (sqrt(w) on y, K * outer(sw, sw) on the
    device, dual * sw; DEVIATION 1688) and kernel='precomputed' (the given
    matrix copied onto the device in place of the kernel matrix)."""
    w = (0.5 + 0.5 * (np.arange(256) % 4)).astype(np.float64)
    w[3] = 0.0
    y2 = np.stack([yr[:256], yr[:256] * 0.5 + 1.0], axis=1).astype(np.float32)
    m = ml.KernelRidge(alpha=0.5, kernel="rbf", gamma=0.05).fit(X[:256], y2, sample_weight=w)
    # the scalar weight on a bounded kernel: a linear K on the fixtures with
    # large-magnitude columns is not positive definite at alpha 1 in float32
    # (the fit refuses, DEVIATION 1661), which tests the refusal, not the weight
    s = ml.KernelRidge(alpha=1.0, kernel="rbf", gamma=0.05).fit(X[:256], yr[:256], sample_weight=2.0)
    K = _neighbors_int_gram(X[:256], X[:256])
    p = ml.KernelRidge(alpha=1.0, kernel="precomputed").fit(K, yr[:256])
    pw = ml.KernelRidge(alpha=1.0, kernel="precomputed").fit(K, yr[:256], sample_weight=w)
    Kq = _neighbors_int_gram(X[256:384], X[:256])
    return _fit(dict(dual=_h(m.dual_coef_), predict=_h(m.predict(X[256:384])),
                     scalar_dual=_h(s.dual_coef_),
                     pre_dual=_h(p.dual_coef_), pre_predict=_h(p.predict(Kq)),
                     prew_dual=_h(pw.dual_coef_)),
                m, lambda e: (e.predict(Xh[:128]),))

@lane("x-neighbors-svm-precomputed")
def _(ml, X, yc, yr, Xh=None):
    """SVC and SVR kernel='precomputed': the solver's tiles are exact copies
    of the given matrix's cells (gather_cols / slice_cols in KernelCache,
    _kernel_cell on the host); predict reads the cross-kernel's support
    columns. Binary, four-class one-vs-one (each pair's rows and columns)
    and the regressor, on an integer-valued Gram exact on any host."""
    K = _neighbors_int_gram(X[:256], X[:256])
    Kq = _neighbors_int_gram(X[256:384], X[:256])
    y4 = (yc[:256] + 2 * (X[:256, 5] > 0).astype(np.int32)).astype(np.int32)
    b = ml.SVC(C=0.01, kernel="precomputed", max_iter=200).fit(K, yc[:256])
    m = ml.SVC(C=0.01, kernel="precomputed", max_iter=200, decision_function_shape="ovr").fit(K, y4)
    r = ml.SVR(C=0.01, kernel="precomputed", epsilon=0.1, max_iter=200).fit(K, yr[:256])
    return _fit(dict(dual=_h(b.dual_coef_), support=_h(b.support_), decision=_h(b.decision_function(Kq)),
                     predict=_h(b.predict(Kq)), multi_dual=_h(m.dual_coef_), multi=_h(m.decision_function(Kq)),
                     svr_dual=_h(r.dual_coef_), svr_predict=_h(r.predict(Kq))),
                m, lambda e: (e.decision_function(_neighbors_int_gram(Xh[:128], X[:256])),
                              e.predict(_neighbors_int_gram(Xh[:128], X[:256]))))

@lane("x-neighbors-km-kernels")
def _(ml, X, yc, yr, Xh=None):
    """KernelRidge and Nystroem with scikit-learn's cosine, chi2 and
    additive_chi2 kernels (kernel_matrix.mojo's chi2_cell_kernel and
    cosine_rows_kernel then the pinned GEMM; km_host_oracle restates them).
    The chi2 kernels take |X| (they refuse negative input)."""
    A = np.abs(X[:256]).astype(np.float32)
    A[::5, 1] = np.float32(0.0)                        # x + y == 0 cells take the skip branch
    Aq = np.abs(X[256:384]).astype(np.float32)
    c = ml.KernelRidge(alpha=1.0, kernel="cosine").fit(X[:256], yr[:256])
    h = ml.KernelRidge(alpha=1.0, kernel="chi2", gamma=0.1).fit(A, yr[:256])
    a = ml.KernelRidge(alpha=1.0e4, kernel="additive_chi2").fit(A[:64], yr[:64])
    n = ml.Nystroem(kernel="chi2", n_components=32, random_state=2).fit(A)
    nc = ml.Nystroem(kernel="cosine", n_components=32, random_state=2).fit(X[:256])
    return _fit(dict(cos_dual=_h(c.dual_coef_), cos_predict=_h(c.predict(X[256:384])),
                     chi2_dual=_h(h.dual_coef_), chi2_predict=_h(h.predict(Aq)),
                     add_dual=_h(a.dual_coef_), add_predict=_h(a.predict(Aq)),
                     ny_chi2=_h(n.transform(Aq)), ny_cos=_h(nc.transform(X[256:384]))),
                c, lambda e: (e.predict(Xh[:128]),))

@lane("x-neighbors-gp-cov")
def _(ml, X, yc, yr, Xh=None):
    """GaussianProcessRegressor.predict(return_cov=True): the full posterior
    covariance k(X, X) - V^T V (gpr_predict_cov_host / gpr_host_predict_cov,
    sample_y's steps before its factorization), with a WhiteKernel on the
    self-kernel diagonal and normalize_y's std**2 scaling."""
    k = ml.ConstantKernel(1.0) * ml.RBF(1.0) + ml.WhiteKernel(0.1)
    m = ml.GaussianProcessRegressor(kernel=k).fit(X[:256, :4], yr[:256])
    mean, cov = m.predict(X[256:320, :4], return_cov=True)
    y = np.ascontiguousarray(yr[:256] + np.float32(50.0)).astype(np.float32)
    n = ml.GaussianProcessRegressor(kernel=k, normalize_y=True).fit(X[:256, :4], y)
    nmean, ncov = n.predict(X[256:320, :4], return_cov=True)
    return _fit(dict(mean=_h(mean), cov=_h(cov), n_mean=_h(nmean), n_cov=_h(ncov)),
                m, lambda e: e.predict(Xh[:64, :4], return_cov=True))

@lane("x-neighbors-metrics")
def _(ml, X, yc, yr, Xh=None):
    """Brute-force k-NN under canberra, braycurtis, correlation,
    jensenshannon and inner_product (distance_ops.mojo::extra_metric_cell on
    the device and the host); jensenshannon on |X|, whose logs need x >= 0.
    kneighbors for each, a classifier vote and a distance-weighted
    regressor."""
    A = np.abs(X[:512]).astype(np.float32)
    A[::9, 3] = np.float32(0.0)                          # log(0) := 0 cells
    out = {}
    for metric in ("canberra", "braycurtis", "correlation", "jensenshannon", "inner_product"):
        base = A if metric == "jensenshannon" else X[:512]
        q = np.abs(X[512:640]).astype(np.float32) if metric == "jensenshannon" else X[512:640]
        nn = ml.NearestNeighbors(n_neighbors=5, metric=metric).fit(base)
        d, i = nn.kneighbors(q)
        out[metric + "_d"] = _h(d)
        out[metric + "_i"] = _h(i)
    c = ml.KNeighborsClassifier(n_neighbors=7, metric="canberra").fit(X[:512], yc[:512])
    r = ml.KNeighborsRegressor(n_neighbors=7, metric="correlation", weights="distance").fit(X[:512], yr[:512])
    out["clf"] = _h(c.predict(X[512:640]))
    out["reg"] = _h(r.predict(X[512:640]))
    return _fit(out, c, lambda e: (e.predict(Xh[:128]),))

@lane("x-neighbors-svc-probability")
def _(ml, X, yc, yr, Xh=None):
    """SVC probability=True: libsvm's Platt scaling (a 5-fold CV per class
    pair over a SplitMix64 shuffle keyed by random_state, each fold's
    decisions from the binary solver, sigmoid_train, and for three classes
    the pairwise coupling), all host arithmetic binary64 with the portable
    exp/log (_svm_impl.SVC._fit_probability). Binary, weighted three-class,
    and a second seed."""
    y3 = (yc[:192] + (X[:192, 5] > 0.5).astype(np.int32)).astype(np.int32)
    w = (0.5 + 0.5 * (np.arange(192) % 4)).astype(np.float64)
    b = ml.SVC(C=1.0, kernel="rbf", gamma=0.05, max_iter=200, probability=True).fit(X[:192], yc[:192])
    m = ml.SVC(C=1.0, kernel="rbf", gamma=0.05, max_iter=200, probability=True,
               random_state=7).fit(X[:192], y3, sample_weight=w)
    s = ml.SVC(C=0.5, kernel="linear", max_iter=200, probability=True, random_state=11).fit(X[:192], y3)
    return _fit(dict(a=_h(b.probA_), b=_h(b.probB_), proba=_h(b.predict_proba(X[192:320])),
                     log_proba=_h(b.predict_log_proba(X[192:320])),
                     m_a=_h(m.probA_), m_b=_h(m.probB_), m_proba=_h(m.predict_proba(X[192:320])),
                     s_a=_h(s.probA_), s_proba=_h(s.predict_proba(X[192:320]))),
                m, lambda e: (e.predict_proba(Xh[:128]),))

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
_batch_decl(_rows_calls("predict", sl=slice(0, 256)), "x-neighbors-svgp", "x-neighbors-svr-kernels")
_batch_decl(_rows_calls("predict", sl=slice(0, 128)), "x-neighbors-krr-options", "x-neighbors-km-kernels")
_batch_decl(_rows_calls("decision_function", "predict", sl=slice(0, 256)), "x-neighbors-gamma-scale", "x-neighbors-svm-weights",
            "x-neighbors-svc-sigmoid", "x-neighbors-svc-multiclass")
_batch_decl(_rows_calls("predict_proba", sl=slice(0, 128)), "x-neighbors-svc-probability")
