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
