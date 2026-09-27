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
    sh = ml.NMF(n_components=3, shuffle=True, random_state=4, max_iter=15, tol=1e-6).fit(A[:500])
    kl = ml.NMF(n_components=3, solver="mu", beta_loss="kullback-leibler", max_iter=20, tol=1e-6).fit(A[:500])
    isd = ml.NMF(n_components=2, solver="mu", beta_loss="itakura-saito", init="random", random_state=2,
                 max_iter=20, tol=1e-6).fit(A[:500] + np.float32(0.1))
    return _fit(dict(W=_h(W), H=_h(m.components_), err=_h(np.float32(m.reconstruction_err_)),
                     it=_h(np.int32(m.n_iter_)), T=_h(m.transform(A[:128])), Wu=_h(Wu), Hu=_h(u.components_),
                     Tu=_h(u.transform(A[:128])), Ha=_h(a.components_), Hsh=_h(sh.components_), Hkl=_h(kl.components_), ekl=_h(np.float64(kl.reconstruction_err_)),
                     His=_h(isd.components_)),
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
    # a distance matrix from the lane's own (identical) affinity: 1 - aff is
    # one IEEE subtraction per entry, the same bytes on every box
    dist = (np.float32(1) - np.asarray(m.affinity_matrix_)).astype(np.float32)
    p = ml.SpectralEmbedding(n_components=2, affinity="precomputed_nearest_neighbors", n_neighbors=12,
                             eigen_solver="lobpcg").fit(dist)
    return _fit(dict(emb=_h(m.embedding_), aff=_h(m.affinity_matrix_), demb=_h(d.embedding_),
                     pemb=_h(p.embedding_), paff=_h(p.affinity_matrix_)))


@lane("x-decomp-lu")
def _(ml, X, yc, yr, Xh=None):
    n = X.shape[1]
    A = np.ascontiguousarray(X[:n, :n])
    B = np.ascontiguousarray(X[n:n + 24, :n].T)
    lu, piv = ml.lu_factor(A)
    x = ml.lu_solve((lu, piv), B)
    v = ml.lu_solve((lu, piv), np.ascontiguousarray(X[200, :n]))
    s = ml.solve(np.ascontiguousarray(X[300:300 + n, :n].T), np.ascontiguousarray(X[400, :n]))
    xt = ml.lu_solve((lu, piv), B, trans=1)
    # numpy's eigh reads ONE triangle: a non-symmetric input, both triangles
    G = np.ascontiguousarray(X[:6, :6])
    wl, vl = ml.linalg.eigh(G)
    wu, vu = ml.linalg.eigh(G, UPLO="U")
    return _fit(dict(lu=_h(lu), piv=_h(piv), x=_h(x), v=_h(v), s=_h(s), xt=_h(xt), eigl=_h(wl, vl), eigu=_h(wu, vu)))


@lane("x-decomp-lstsq-rsvd")
def _(ml, X, yc, yr, Xh=None):
    U, s, Vt = ml.randomized_svd(X[:4000], 4, random_state=0)
    Ut, st, Vtt = ml.randomized_svd(np.ascontiguousarray(X[:12].T), 3, n_iter=2, random_state=1)
    x, res, rank, sv = ml.lstsq(X[:600], yr[:600])
    xm, resm, rankm, _ = ml.lstsq(X[:300], np.ascontiguousarray(X[300:600, :3]))
    return _fit(dict(U=_h(U), s=_h(s), Vt=_h(Vt), Ut=_h(Ut), st=_h(st), Vtt=_h(Vtt), x=_h(x), res=_h(res),
                     rank=_h(np.int32(rank)), sv=_h(sv), xm=_h(xm), resm=_h(resm), rankm=_h(np.int32(rankm))))


@lane("x-decomp-pls")
def _(ml, X, yc, yr, Xh=None):
    m = ml.PLSRegression(n_components=3).fit(X[:2000], yr[:2000])
    Y2 = np.ascontiguousarray(np.stack([yr[:2000], X[:2000, 3]], axis=1))
    c = ml.PLSCanonical(n_components=2).fit(X[:2000, :10], np.ascontiguousarray(X[:2000, 10:14]))
    a = ml.CCA(n_components=2, max_iter=100).fit(X[:1000, :8], np.ascontiguousarray(X[:1000, 8:12]))
    cs = ml.PLSCanonical(n_components=2, algorithm="svd").fit(X[:2000, :10], np.ascontiguousarray(X[:2000, 10:14]))
    m2 = ml.PLSRegression(n_components=2, scale=False).fit(X[:2000], Y2)
    return _fit(dict(coef=_h(m.coef_), xw=_h(m.x_weights_), xr=_h(m.x_rotations_), pred=_h(m.predict(X[:256])),
                     T=_h(m.transform(X[:256])), it=_h(np.int32(m.n_iter_)), c_xr=_h(c.x_rotations_),
                     c_yr=_h(c.y_rotations_), c_T=_h(*c.transform(X[:256, :10], np.ascontiguousarray(X[:256, 10:14]))),
                     a_xr=_h(a.x_rotations_), a_yr=_h(a.y_rotations_), a_it=_h(np.int32(a.n_iter_)),
                     m2=_h(m2.coef_, m2.predict(X[:256])), cs_xr=_h(cs.x_rotations_, cs.y_rotations_)),
                m, lambda e: (e.predict(Xh[:256]), e.transform(Xh[:256])))


_batch_decl(_rows_calls("predict", "transform", sl=slice(0, 256)), "x-decomp-pls")


@lane("x-decomp-dict-learning")
def _(ml, X, yc, yr, Xh=None):
    S = X[:300]
    m = ml.DictionaryLearning(n_components=6, alpha=0.5, max_iter=8, random_state=0)
    code = m.fit_transform(S)
    c = ml.DictionaryLearning(n_components=5, alpha=0.3, max_iter=6, fit_algorithm="cd",
                              transform_algorithm="lasso_cd", split_sign=True, random_state=1).fit(S)
    t = ml.DictionaryLearning(n_components=4, alpha=0.2, max_iter=5, transform_algorithm="threshold",
                              transform_alpha=0.1, positive_dict=True, positive_code=True).fit(S[:200])
    mb = ml.MiniBatchDictionaryLearning(n_components=5, alpha=0.4, batch_size=64, max_iter=3, random_state=2,
                                        transform_algorithm="lasso_lars").fit(X[:600])
    sc = ml.SparseCoder(m.components_, transform_algorithm="lasso_cd", transform_alpha=0.2)
    sct = ml.SparseCoder(m.components_, transform_algorithm="threshold", transform_alpha=0.1, split_sign=True)
    scl = ml.SparseCoder(m.components_, transform_algorithm="lars", transform_n_nonzero_coefs=3)
    return _fit(dict(sc=_h(sc.transform(S[:128]), sct.transform(S[:128])), scl=_h(scl.transform(S[:32])), code=_h(code), D=_h(m.components_), err=_h(np.float64(m.error_)), T=_h(m.transform(S[:128])),
                     cD=_h(c.components_), cT=_h(c.transform(S[:128])), tT=_h(t.transform(S[:128]), t.components_),
                     mbD=_h(mb.components_), mbT=_h(mb.transform(S[:128])), mbn=_h(np.int32(mb.n_steps_))),
                m, lambda e: (e.transform(Xh[:256]),))


@lane("x-decomp-sparse-pca")
def _(ml, X, yc, yr, Xh=None):
    S = X[:300]
    m = ml.SparsePCA(n_components=4, alpha=1, max_iter=8, random_state=0).fit(S)
    c = ml.SparsePCA(n_components=3, alpha=0.5, max_iter=6, method="cd").fit(S)
    b = ml.MiniBatchSparsePCA(n_components=3, alpha=1, max_iter=4, batch_size=4, random_state=3).fit(X[:200])
    return _fit(dict(comp=_h(m.components_), T=_h(m.transform(S[:128])), err=_h(np.float64(m.error_)),
                     ccomp=_h(c.components_), bcomp=_h(b.components_), bT=_h(b.transform(S[:128]))),
                m, lambda e: (e.transform(Xh[:256]),))


_batch_decl(_rows_calls("transform", sl=slice(0, 256)), "x-decomp-dict-learning", "x-decomp-sparse-pca")


@lane("x-decomp-lda")
def _(ml, X, yc, yr, Xh=None):
    # counts: |X| rounded down to integers (elementwise, the same bytes on every box)
    C = np.floor(np.abs(X[:400]) * np.float32(3)).astype(np.float32)
    C = np.minimum(C, np.float32(50))
    m = ml.LatentDirichletAllocation(n_components=4, max_iter=5, random_state=0).fit(C)
    o = ml.LatentDirichletAllocation(n_components=3, learning_method="online", batch_size=100, max_iter=2,
                                     random_state=1).fit(C)
    return _fit(dict(comp=_h(m.components_), bound=_h(np.float32(m.bound_)), T=_h(m.transform(C[:128])),
                     score=_h(np.float64(m.score(C[:128]))), ocomp=_h(o.components_), oT=_h(o.transform(C[:128]))),
                m, lambda e: (e.transform(np.minimum(np.floor(np.abs(Xh[:256]) * np.float32(3)), np.float32(50)).astype(np.float32)),))


_batch_decl(_rows_calls("transform", sl=slice(0, 256),
                        prep=lambda Xh: np.minimum(np.floor(np.abs(Xh) * np.float32(3)), np.float32(50)).astype(np.float32)),
            "x-decomp-lda")


@lane("x-decomp-manifold")
def _(ml, X, yc, yr, Xh=None):
    S = X[:160]
    iso = ml.Isomap(n_neighbors=8, n_components=3).fit(S)
    cm = ml.ClassicalMDS(n_components=2).fit(S)
    md = ml.MDS(n_components=2, init="random", n_init=2, max_iter=40, random_state=0)
    emb = md.fit_transform(S[:100])
    mc = ml.MDS(n_components=2, init="classical_mds", max_iter=30).fit(S[:100])
    lle = ml.LocallyLinearEmbedding(n_neighbors=10, n_components=2).fit(S)
    lt = ml.LocallyLinearEmbedding(n_neighbors=8, n_components=2, method="ltsa").fit(S[:80])
    he = ml.LocallyLinearEmbedding(n_neighbors=10, n_components=2, method="hessian").fit(S[:80])
    mo = ml.LocallyLinearEmbedding(n_neighbors=10, n_components=2, method="modified").fit(S[:80])
    # the radius is a quarter of the largest kNN geodesic (this lane's own
    # identical output, one exact float64 product)
    rad = float(np.asarray(iso.dist_matrix_).max()) * 0.25
    ir = ml.Isomap(n_neighbors=None, radius=rad, n_components=2, path_method="FW").fit(S[:100])
    nm = ml.MDS(n_components=2, metric_mds=False, init="random", max_iter=25, random_state=1)
    nemb = nm.fit_transform(S[:60])
    im = ml.Isomap(n_neighbors=8, n_components=2, metric="manhattan").fit(S[:100])
    ic = ml.Isomap(n_neighbors=8, n_components=2, metric="cosine").fit(S[:100])
    ip = ml.Isomap(n_neighbors=8, n_components=2, metric="minkowski", p=3).fit(S[:100])
    cc = ml.ClassicalMDS(n_components=2, metric="chebyshev").fit(S[:80])
    return _fit(dict(iso=_h(iso.embedding_), isod=_h(iso.dist_matrix_), isoT=_h(iso.transform(Xh[:64])),
                     cm=_h(cm.embedding_), md=_h(emb), mds=_h(np.float64(md.stress_)), mc=_h(mc.embedding_),
                     lle=_h(lle.embedding_), lleT=_h(lle.transform(Xh[:64])), lt=_h(lt.embedding_),
                     he=_h(he.embedding_), mo=_h(mo.embedding_, np.float64(mo.reconstruction_error_)),
                     ir=_h(ir.embedding_, ir.dist_matrix_, ir.transform(S[:32])),
                     nm=_h(nemb, np.float64(nm.stress_), np.int32(nm.n_iter_)),
                     im=_h(im.embedding_, im.transform(S[:16])), ic=_h(ic.embedding_, ic.transform(S[:16])),
                     ip=_h(ip.embedding_, ip.dist_matrix_), cc=_h(cc.embedding_, cc.dissimilarity_matrix_)),
                iso, lambda e: (e.transform(Xh[:128]),))


_batch_decl(_rows_calls("transform", sl=slice(0, 128)), "x-decomp-manifold")


@lane("x-decomp-robust-cov")
def _(ml, X, yc, yr, Xh=None):
    S = np.ascontiguousarray(X[:400, :6])
    m = ml.MinCovDet(random_state=0).fit(S)
    e = ml.EllipticEnvelope(contamination=0.05, random_state=1, support_fraction=0.7).fit(S)
    one = ml.MinCovDet().fit(np.ascontiguousarray(X[:300, 5:6]))
    lab = np.where(X[:128, 0] > 0, 1, -1).astype(np.int32)
    esc = np.float64([e.score(S[:128], lab), e.score(S[:128], lab, sample_weight=np.abs(X[:128, 1]))])
    return _fit(dict(loc=_h(m.location_), cov=_h(m.covariance_), rloc=_h(m.raw_location_), rcov=_h(m.raw_covariance_),
                     sup=_h(np.asarray(m.support_, dtype=np.int8)), dist=_h(m.dist_), maha=_h(m.mahalanobis(S[:128])),
                     eoff=_h(np.float64(e.offset_)), esc=_h(esc), one=_h(one.location_, one.covariance_, one.dist_), edec=_h(e.decision_function(S[:128])), epred=_h(e.predict(S[:128]))),
                e, lambda est: (est.decision_function(np.ascontiguousarray(Xh[:256, :6])),
                                est.predict(np.ascontiguousarray(Xh[:256, :6]))))


_batch_decl(_rows_calls("decision_function", "predict", sl=np.s_[:256, :6]), "x-decomp-robust-cov")


@lane("x-decomp-als")
def _(ml, X, yc, yr, Xh=None):
    # implicit feedback: positive entries of the first 300 rows, 16 items
    R = np.maximum(X[:300], np.float32(0)).astype(np.float32)
    m = ml.AlternatingLeastSquares(factors=6, regularization=0.05, alpha=2.0, iterations=4, random_state=0).fit(R)
    ids, sc = m.recommend(3, R, N=5)
    sid, ssc = m.similar_items(2, N=4)
    lo = ml.AlternatingLeastSquares(factors=4, regularization=0.1, iterations=3, calculate_training_loss=True,
                                    random_state=2).fit(R[:120])
    return _fit(dict(U=_h(m.user_factors), V=_h(m.item_factors), rid=_h(ids), rsc=_h(sc), sid=_h(sid), ssc=_h(ssc),
                     loss=_h(np.float64(lo.training_loss_), lo.user_factors)))


@lane("x-decomp-pca-randomized")
def _(ml, X, yc, yr, Xh=None):
    p = ml.PCA(n_components=4, svd_solver="randomized", random_state=3).fit(X[:4000])
    w = ml.PCA(n_components=3, svd_solver="randomized", whiten=True, iterated_power=2).fit(X[:2000])
    t = ml.TruncatedSVD(n_components=5, algorithm="randomized", random_state=1).fit(X[:4000])
    f = ml.PCA(n_components=0.8, svd_solver="full").fit(X[:3000])
    a = ml.PCA(n_components=3, svd_solver="arpack").fit(X[:3000])
    mle = ml.PCA(n_components="mle", svd_solver="full").fit(X[:3000])
    ta = ml.TruncatedSVD(n_components=4, algorithm="arpack").fit(X[:4000])
    return _fit(dict(pc=_h(p.components_), pev=_h(p.explained_variance_, p.explained_variance_ratio_, p.singular_values_),
                     pnv=_h(np.float64(p.noise_variance_)), pT=_h(p.transform(X[:256])), wT=_h(w.transform(X[:256])),
                     tc=_h(t.components_, t.singular_values_, t.explained_variance_ratio_), tT=_h(t.transform(X[:256])),
                     fc=_h(f.components_, f.explained_variance_), fnv=_h(np.float64(f.noise_variance_)),
                     fT=_h(f.transform(X[:256])), ac=_h(a.components_, a.explained_variance_, a.transform(X[:256])),
                     mle=_h(np.int32(mle.n_components_), mle.components_, np.float64(mle.noise_variance_)),
                     ta=_h(ta.components_, ta.singular_values_, ta.explained_variance_, ta.explained_variance_ratio_)),
                p, lambda e: (e.transform(Xh[:256]),))


_batch_decl(_rows_calls("transform", sl=slice(0, 256)), "x-decomp-pca-randomized")
