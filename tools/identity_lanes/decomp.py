# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE DECOMP LANE'S IDENTITY LANES (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md).
#
# Owned by the `decomp` expansion lane. tools/identity_break.py executes this
# file in ITS OWN namespace after every helper and registry exists
# (`_load_lane_fragments`), so a lane here is written exactly like a lane there:
#
#     @lane("decomp-example")
#     def _(ml, X, yc, yr, Xh=None):
#         m = ml.Example().fit(X)
#         return _fit(dict(out=_h(m.transform(X[:256]))), m, lambda e: (e.transform(Xh[:256]),))
#
#     _batch_decl(_rows_calls("transform", sl=slice(0, 256)), "decomp-example")
#
# Rules the loader enforces: spell every lane name literally; add only your
# own lanes (to LANES and the per-lane registries); rebind no existing name;
# prefix your own helpers with `_decomp_`. No imports are needed: np, _h,
# _fit, _rows_calls and the rest are this module's.


@lane("x-decomp-ipca")
def _(ml, X, yc, yr, Xh=None):
    m = ml.IncrementalPCA(n_components=6, batch_size=4000).fit(X)
    w = ml.IncrementalPCA(n_components=4, whiten=True, batch_size=5000).fit(X[:10000])
    return _fit(dict(components=_h(m.components_), sv=_h(m.singular_values_), mean=_h(m.mean_),
                     var=_h(m.var_), ev=_h(m.explained_variance_), evr=_h(m.explained_variance_ratio_),
                     noise=_h(np.float32(m.noise_variance_)), transform=_h(m.transform(X[:256])),
                     inverse=_h(m.inverse_transform(m.transform(X[:64]))),
                     white=_h(w.transform(X[:256]), w.inverse_transform(w.transform(X[:64])))),
                m, lambda e: (e.transform(Xh[:256]),))


_batch_decl(_rows_calls("transform", sl=slice(0, 256)), "x-decomp-ipca")


@lane("x-decomp-grp")
def _(ml, X, yc, yr, Xh=None):
    m = ml.GaussianRandomProjection(n_components=8, random_state=11, compute_inverse_components=True).fit(X)
    return _fit(dict(components=_h(m.components_), inverse=_h(m.inverse_components_),
                     transform=_h(m.transform(X[:512])), back=_h(m.inverse_transform(m.transform(X[:64])))),
                m, lambda e: (e.transform(Xh[:512]),))


@lane("x-decomp-srp")
def _(ml, X, yc, yr, Xh=None):
    m = ml.SparseRandomProjection(n_components=12, random_state=5).fit(X)
    t = ml.SparseRandomProjection(n_components=6, density=0.6, random_state=9,
                                  compute_inverse_components=True).fit(X)
    return _fit(dict(components=_h(m.components_), transform=_h(m.transform(X[:512])),
                     t_components=_h(t.components_), t_inverse=_h(t.inverse_components_),
                     t_transform=_h(t.transform(X[:512]))),
                m, lambda e: (e.transform(Xh[:512]),))


_batch_decl(_rows_calls("transform", sl=slice(0, 256)), "x-decomp-grp", "x-decomp-srp")



@lane("x-decomp-nmf")
def _(ml, X, yc, yr, Xh=None):
    A = np.abs(X[:2000])
    m = ml.NMF(n_components=5, max_iter=40, tol=1e-6)
    W = m.fit_transform(A)
    u = ml.NMF(n_components=4, solver="mu", init="random", random_state=3, max_iter=40, tol=1e-6,
               alpha_W=0.01, l1_ratio=0.5)
    Wu = u.fit_transform(A)
    a = ml.NMF(n_components=3, init="nndsvdar", random_state=1, max_iter=20).fit(A[:500])
    return _fit(dict(W=_h(W), H=_h(m.components_), err=_h(np.float32(m.reconstruction_err_)),
                     it=_h(np.int32(m.n_iter_)), T=_h(m.transform(A[:128])), Wu=_h(Wu), Hu=_h(u.components_),
                     Tu=_h(u.transform(A[:128])), Ha=_h(a.components_)),
                m, lambda e: (e.transform(np.abs(Xh[:128])),))


@lane("x-decomp-fastica")
def _(ml, X, yc, yr, Xh=None):
    S = X[:3000]
    m = ml.FastICA(n_components=5, random_state=2, max_iter=60)
    src = m.fit_transform(S)
    e = ml.FastICA(n_components=3, fun="exp", algorithm="deflation", whiten="arbitrary-variance",
                   random_state=4, max_iter=40).fit(S)
    c = ml.FastICA(fun="cube", random_state=6, max_iter=30).fit(S[:1000])
    return _fit(dict(src=_h(src), comp=_h(m.components_), mix=_h(m.mixing_), white=_h(m.whitening_),
                     it=_h(np.int32(m.n_iter_)), ecomp=_h(e.components_), et=_h(e.transform(S[:64])),
                     ccomp=_h(c.components_), inv=_h(m.inverse_transform(src[:64]))),
                m, lambda e: (e.transform(Xh[:256]),))


_batch_decl(_rows_calls("transform", sl=slice(0, 256)), "x-decomp-fastica")


@lane("x-decomp-factor-analysis")
def _(ml, X, yc, yr, Xh=None):
    m = ml.FactorAnalysis(n_components=4, max_iter=60).fit(X[:5000])
    v = ml.FactorAnalysis(n_components=3, rotation="varimax", max_iter=40).fit(X[:2000])
    q = ml.FactorAnalysis(n_components=3, rotation="quartimax", max_iter=40).fit(X[:2000])
    return _fit(dict(comp=_h(m.components_), psi=_h(m.noise_variance_), ll=_h(np.float64(m.loglike_)),
                     T=_h(m.transform(X[:256])), cov=_h(m.get_covariance()), prec=_h(m.get_precision()),
                     ss=_h(m.score_samples(X[:256])), vcomp=_h(v.components_), qcomp=_h(q.components_)),
                m, lambda e: (e.transform(Xh[:256]), e.score_samples(Xh[:256])))


_batch_decl(_rows_calls("transform", "score_samples", sl=slice(0, 256)), "x-decomp-factor-analysis")


@lane("x-decomp-spectral-rbf")
def _(ml, X, yc, yr, Xh=None):
    # per-column scale to [-1, 1] (elementwise IEEE division, the same bytes on
    # every box) so the `wide` fixture's 1e4 columns do not underflow every
    # off-diagonal affinity to zero
    S = (X[:300] / (np.abs(X[:300]).max(axis=0) + np.float32(1))).astype(np.float32)
    m = ml.SpectralEmbedding(n_components=3, affinity="rbf", gamma=0.5).fit(S)
    d = ml.SpectralEmbedding(n_components=2, affinity="rbf").fit(S[:200])
    return _fit(dict(emb=_h(m.embedding_), aff=_h(m.affinity_matrix_), demb=_h(d.embedding_)))


@lane("x-decomp-lu")
def _(ml, X, yc, yr, Xh=None):
    n = X.shape[1]
    A = np.ascontiguousarray(X[:n, :n])
    B = np.ascontiguousarray(X[n:n + 24, :n].T)
    lu, piv = ml.lu_factor(A)
    x = ml.lu_solve((lu, piv), B)
    v = ml.lu_solve((lu, piv), np.ascontiguousarray(X[200, :n]))
    s = ml.solve(np.ascontiguousarray(X[300:300 + n, :n].T), np.ascontiguousarray(X[400, :n]))
    return _fit(dict(lu=_h(lu), piv=_h(piv), x=_h(x), v=_h(v), s=_h(s)))
