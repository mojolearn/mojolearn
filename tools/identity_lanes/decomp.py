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
