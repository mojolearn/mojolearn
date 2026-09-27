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
                     proba=_h(s.predict_proba(X[:256])), manhattan=_h(md.centroids_, md.predict(X[:512]))),
                s, lambda e: (e.predict(Xh[:256]), e.decision_function(Xh[:256])))




_batch_decl(_rows_calls("score_samples", "predict", sl=slice(0, 256)), "x-neighbors-lof")
_batch_decl(_rows_calls("predict", "decision_function", "predict_proba", sl=slice(0, 256)), "x-neighbors-nearest-centroid")
