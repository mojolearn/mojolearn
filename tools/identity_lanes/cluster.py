# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE CLUSTER LANE'S IDENTITY LANES (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md).
#
# Owned by the `cluster` expansion lane. tools/identity_break.py executes this
# file in ITS OWN namespace after every helper and registry exists
# (`_load_lane_fragments`), so a lane here is written exactly like a lane there:
#
#     @lane("cluster-example")
#     def _(ml, X, yc, yr, Xh=None):
#         m = ml.Example().fit(X)
#         return _fit(dict(out=_h(m.transform(X[:256]))), m, lambda e: (e.transform(Xh[:256]),))
#
#     _batch_decl(_rows_calls("transform", sl=slice(0, 256)), "cluster-example")
#
# Rules the loader enforces: spell every lane name literally; add only your
# own lanes (to LANES and the per-lane registries); rebind no existing name;
# prefix your own helpers with `_cluster_`. No imports are needed: np, _h,
# _fit, _rows_calls and the rest are this module's.


@lane("x-cluster-minibatch-kmeans")
def _(ml, X, yc, yr, Xh=None):
    """MiniBatchKMeans (python/mojolearn/_expansion_cluster.py): 3000 x 8
    rows, batch 256 so the fit takes many steps, reassignment and the
    early stop reachable."""
    m = ml.MiniBatchKMeans(n_clusters=6, batch_size=256, max_iter=5, random_state=3).fit(X[:3000, :8])
    return _fit(dict(centers=_h(m.cluster_centers_), labels=_h(m.labels_), counts=_h(m.counts_),
                     steps=_h(np.asarray([m.n_steps_, m.n_iter_], dtype=np.int64)),
                     inertia=_h(np.float64(m.inertia_))),
                m, lambda e: (e.predict(Xh[:256, :8]), e.transform(Xh[:256, :8])))


_batch_decl(_rows_calls("predict", "transform", sl=np.s_[:256, :8]), "x-cluster-minibatch-kmeans")


@lane("x-cluster-bisecting-kmeans")
def _(ml, X, yc, yr, Xh=None):
    """BisectingKMeans (python/mojolearn/_expansion_cluster.py): 2000 x 6
    rows, five leaves by biggest inertia, two restarts per split; infer is
    the tree descent and the distances to the leaf centers."""
    m = ml.BisectingKMeans(n_clusters=5, n_init=2, random_state=3).fit(X[:2000, :6])
    return _fit(dict(centers=_h(m.cluster_centers_), labels=_h(m.labels_), tree=_h(m._tree_nodes),
                     inertia=_h(np.float64(m.inertia_))),
                m, lambda e: (e.predict(Xh[:256, :6]), e.transform(Xh[:256, :6])))


_batch_decl(_rows_calls("predict", "transform", sl=np.s_[:256, :6]), "x-cluster-bisecting-kmeans")


@lane("x-cluster-meanshift")
def _(ml, X, yc, yr, Xh=None):
    """MeanShift (python/mojolearn/_expansion_cluster.py): 1200 rows of columns 1-3 (two of them subnormal on the denormal fixtures),
    the estimated bandwidth (the device's row order statistic), every row a
    seed, cluster_all off so the -1 arm is reached; infer is predict."""
    m = ml.MeanShift(cluster_all=False).fit(X[:1200, 1:4])
    return _fit(dict(centers=_h(m.cluster_centers_), labels=_h(m.labels_),
                     bandwidth=_h(np.float64(m.bandwidth_)), n_iter=_h(np.int64(m.n_iter_))),
                m, lambda e: (e.predict(Xh[:256, 1:4]),))


_batch_decl(_rows_calls("predict", sl=np.s_[:256, 1:4]), "x-cluster-meanshift")


@lane("x-cluster-optics")
def _(ml, X, yc, yr, Xh=None):
    """OPTICS (python/mojolearn/_expansion_cluster.py): 1500 rows of
    columns 1-4, xi extraction with predecessor correction, then the same
    graph cut by the dbscan extraction at a finite max_eps (the inf branches
    and the eps cut both reached). Transductive: no infer."""
    m = ml.OPTICS(min_samples=8).fit(X[:1500, 1:5])
    parts = dict(order=_h(m.ordering_), core=_h(m.core_distances_), reach=_h(m.reachability_),
                 pred=_h(m.predecessor_), labels=_h(m.labels_), hierarchy=_h(m.cluster_hierarchy_))
    eps = float(np.median(np.asarray(m.core_distances_)))
    m2 = ml.OPTICS(min_samples=8, max_eps=eps * 3, cluster_method="dbscan", eps=eps).fit(X[:1500, 1:5])
    parts.update(dbscan_labels=_h(m2.labels_), dbscan_reach=_h(m2.reachability_))
    return _fit(parts, m, "n/a:transductive (OPTICS has no predict, in the reference and in scikit-learn alike)")


_batch_decl("n/a:transductive (OPTICS labels the fitted rows only)", "x-cluster-optics")
