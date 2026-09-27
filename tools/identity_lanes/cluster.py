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


@lane("x-cluster-affinity-propagation")
def _(ml, X, yc, yr, Xh=None):
    """AffinityPropagation (python/mojolearn/_expansion_cluster.py): 400
    rows of columns 1-4, the median preference (the device order statistic),
    the seeded tie noise, damping 0.7; infer is predict."""
    m = ml.AffinityPropagation(damping=0.7, random_state=3).fit(X[:400, 1:5])
    return _fit(dict(centers=_h(m.cluster_centers_indices_), labels=_h(m.labels_),
                     affinity=_h(m.affinity_matrix_), n_iter=_h(np.int64(m.n_iter_))),
                m, lambda e: (e.predict(Xh[:256, 1:5]),))


_batch_decl(_rows_calls("predict", sl=np.s_[:256, 1:5]), "x-cluster-affinity-propagation")


@lane("x-cluster-bgmm")
def _(ml, X, yc, yr, Xh=None):
    """BayesianGaussianMixture (python/mojolearn/_expansion_cluster.py):
    2000 rows of columns 1-4, five components under the Dirichlet-process
    prior (k-means start, 30 iterations), then a Dirichlet-distribution fit
    from the random start; infer is score_samples, predict and
    predict_proba."""
    m = ml.BayesianGaussianMixture(n_components=5, max_iter=30, random_state=3).fit(X[:2000, 1:5])
    m2 = ml.BayesianGaussianMixture(n_components=3, max_iter=15, init_params="random", random_state=3,
                                    weight_concentration_prior_type="dirichlet_distribution").fit(X[:2000, 1:5])
    return _fit(dict(weights=_h(m.weights_), means=_h(m.means_), cov=_h(m.covariances_),
                     pchol=_h(m.precisions_cholesky_), dof=_h(m.degrees_of_freedom_),
                     lb=_h(np.float64(m.lower_bound_)), n_iter=_h(np.int64(m.n_iter_)),
                     labels=_h(m.predict(X[:2000, 1:5])),
                     dd_weights=_h(m2.weights_), dd_means=_h(m2.means_), dd_lb=_h(np.float64(m2.lower_bound_))),
                m, lambda e: (e.score_samples(Xh[:256, 1:5]), e.predict(Xh[:256, 1:5]), e.predict_proba(Xh[:256, 1:5])))


_batch_decl(_rows_calls("score_samples", "predict", "predict_proba", sl=np.s_[:256, 1:5]), "x-cluster-bgmm")


def _cluster_weights(X, n):
    """Positive sample weights from a column the lanes do not cluster on."""
    return (np.abs(X[:n, 7]).astype(np.float32) + np.float32(0.25)).astype(np.float32)


@lane("x-cluster-minibatch-options")
def _(ml, X, yc, yr, Xh=None):
    """MiniBatchKMeans options (option parity): init='random' (weighted draw
    of distinct init rows, n_init 3 by 'auto') and sample_weight (the batch
    draw, the potentials and the inertia weighted)."""
    w = _cluster_weights(X, 3000)
    m = ml.MiniBatchKMeans(n_clusters=6, init="random", batch_size=256, max_iter=4, random_state=5).fit(
        X[:3000, :8], sample_weight=w)
    m2 = ml.MiniBatchKMeans(n_clusters=5, batch_size=200, max_iter=3, random_state=6).fit(X[:3000, :8], sample_weight=w)
    return _fit(dict(centers=_h(m.cluster_centers_), labels=_h(m.labels_), inertia=_h(np.float64(m.inertia_)),
                     pp_centers=_h(m2.cluster_centers_), pp_inertia=_h(np.float64(m2.inertia_))),
                m, lambda e: (e.predict(Xh[:256, :8]),))


_batch_decl(_rows_calls("predict", sl=np.s_[:256, :8]), "x-cluster-minibatch-options")


@lane("x-cluster-bisecting-options")
def _(ml, X, yc, yr, Xh=None):
    """BisectingKMeans options: sample_weight through the weighted 2-means and
    scores, and bisecting_strategy='largest_cluster' with init='k-means++'."""
    w = _cluster_weights(X, 2000)
    m = ml.BisectingKMeans(n_clusters=5, random_state=4).fit(X[:2000, :6], sample_weight=w)
    m2 = ml.BisectingKMeans(n_clusters=4, init="k-means++", bisecting_strategy="largest_cluster",
                            random_state=4).fit(X[:2000, :6])
    return _fit(dict(centers=_h(m.cluster_centers_), labels=_h(m.labels_), inertia=_h(np.float64(m.inertia_)),
                     lc_centers=_h(m2.cluster_centers_), lc_labels=_h(m2.labels_)),
                m, lambda e: (e.predict(Xh[:256, :6]),))


_batch_decl(_rows_calls("predict", sl=np.s_[:256, :6]), "x-cluster-bisecting-options")


@lane("x-cluster-meanshift-binned")
def _(ml, X, yc, yr, Xh=None):
    """MeanShift(bin_seeding=True, min_bin_freq=3): the half-to-even binning
    at the bandwidth and the frequency cut, then the device seed loop."""
    m = ml.MeanShift(bin_seeding=True, min_bin_freq=3, bandwidth=0.9).fit(X[:1500, 1:4])
    return _fit(dict(centers=_h(m.cluster_centers_), labels=_h(m.labels_), n_iter=_h(np.int64(m.n_iter_))),
                m, lambda e: (e.predict(Xh[:256, 1:4]),))


_batch_decl(_rows_calls("predict", sl=np.s_[:256, 1:4]), "x-cluster-meanshift-binned")


@lane("x-cluster-optics-metrics")
def _(ml, X, yc, yr, Xh=None):
    """OPTICS under every non-euclidean metric it carries (manhattan,
    chebyshev, minkowski p=3, cosine) and a precomputed matrix."""
    Z = X[:600, 1:5]
    parts = {}
    for name, kw in (("l1", dict(metric="manhattan")), ("linf", dict(metric="chebyshev")),
                     ("p3", dict(metric="minkowski", p=3)), ("cos", dict(metric="cosine"))):
        m = ml.OPTICS(min_samples=6, **kw).fit(Z)
        parts[name + "_order"] = _h(m.ordering_)
        parts[name + "_reach"] = _h(m.reachability_)
        parts[name + "_labels"] = _h(m.labels_)
    D = np.abs(Z[:, None, 0] - Z[None, :, 0]).astype(np.float32) + np.abs(Z[:, None, 1] - Z[None, :, 1]).astype(np.float32)
    m = ml.OPTICS(min_samples=6, metric="precomputed").fit(D.astype(np.float32))
    parts["pre_order"] = _h(m.ordering_)
    parts["pre_labels"] = _h(m.labels_)
    return _fit(parts, m, "n/a:transductive (OPTICS has no predict, in the reference and in scikit-learn alike)")


_batch_decl("n/a:transductive (OPTICS labels the fitted rows only)", "x-cluster-optics-metrics")


@lane("x-cluster-ap-precomputed")
def _(ml, X, yc, yr, Xh=None):
    """AffinityPropagation(affinity='precomputed') on a similarity matrix with
    positive entries: the median preference by the host sort."""
    Z = X[:300, 1:5]
    # ELEMENTWISE, FIXED ORDER, NO BLAS: `Z @ Z.T` would hand each box's BLAS
    # order to the lane as input (identity_break.labels_for says why)
    S = np.zeros((Z.shape[0], Z.shape[0]), dtype=np.float32)
    for c in range(Z.shape[1]):
        S = (S + Z[:, None, c] * Z[None, :, c]).astype(np.float32)
    m = ml.AffinityPropagation(affinity="precomputed", damping=0.8, random_state=2).fit(S)
    return _fit(dict(centers=_h(m.cluster_centers_indices_), labels=_h(m.labels_), n_iter=_h(np.int64(m.n_iter_)),
                     messages=_h(m._message_diag)),
                m, "n/a:precomputed (predict is not supported with affinity='precomputed', as scikit-learn)")


_batch_decl("n/a:precomputed affinity (no predict)", "x-cluster-ap-precomputed")


@lane("x-cluster-bgmm-inits")
def _(ml, X, yc, yr, Xh=None):
    """BayesianGaussianMixture init_params='k-means++' and 'random_from_data'
    (one-hot starts at the picks), and warm_start (a second fit resuming from
    the first's parameters)."""
    m = ml.BayesianGaussianMixture(n_components=4, max_iter=20, init_params="k-means++", random_state=3).fit(X[:1500, 1:5])
    m2 = ml.BayesianGaussianMixture(n_components=4, max_iter=20, init_params="random_from_data",
                                    random_state=3).fit(X[:1500, 1:5])
    m3 = ml.BayesianGaussianMixture(n_components=4, max_iter=5, warm_start=True, random_state=3)
    m3.fit(X[:1500, 1:5])
    m3.fit(X[:1500, 1:5])
    return _fit(dict(means=_h(m.means_), weights=_h(m.weights_), lb=_h(np.float64(m.lower_bound_)),
                     rd_means=_h(m2.means_), rd_lb=_h(np.float64(m2.lower_bound_)),
                     warm_means=_h(m3.means_), warm_lb=_h(np.float64(m3.lower_bound_))),
                m, lambda e: (e.predict_proba(Xh[:256, 1:5]),))


_batch_decl(_rows_calls("predict_proba", sl=np.s_[:256, 1:5]), "x-cluster-bgmm-inits")


@lane("x-cluster-bgmm-covtypes")
def _(ml, X, yc, yr, Xh=None):
    """BayesianGaussianMixture covariance_type 'tied', 'diag' and 'spherical'
    (the tied pooled covariance, the diagonal and scalar Wishart updates)."""
    parts = {}
    last = None
    for ct in ("tied", "diag", "spherical"):
        m = ml.BayesianGaussianMixture(n_components=4, covariance_type=ct, max_iter=20, random_state=3).fit(X[:1500, 1:5])
        parts[ct + "_means"] = _h(m.means_)
        parts[ct + "_cov"] = _h(m.covariances_)
        parts[ct + "_lb"] = _h(np.float64(m.lower_bound_))
        last = m
    return _fit(parts, last, lambda e: (e.score_samples(Xh[:256, 1:5]),))


_batch_decl(_rows_calls("score_samples", sl=np.s_[:256, 1:5]), "x-cluster-bgmm-covtypes")


@lane("x-cluster-minibatch-partial")
def _(ml, X, yc, yr, Xh=None):
    """MiniBatchKMeans.partial_fit: four calls over 500-row chunks (the first
    starts the centers by k-means++ with sample weights), the stream and the
    reassignment counter carried between calls."""
    w = _cluster_weights(X, 2000)
    m = ml.MiniBatchKMeans(n_clusters=6, batch_size=128, random_state=7)
    for c in range(4):
        m.partial_fit(X[c * 500:(c + 1) * 500, :8], sample_weight=w[c * 500:(c + 1) * 500])
    return _fit(dict(centers=_h(m.cluster_centers_), counts=_h(m.counts_), labels=_h(m.labels_),
                     inertia=_h(np.float64(m.inertia_))),
                m, lambda e: (e.predict(Xh[:256, :8]),))


_batch_decl(_rows_calls("predict", sl=np.s_[:256, :8]), "x-cluster-minibatch-partial")


@lane("x-cluster-gmm-options")
def _(ml, X, yc, yr, Xh=None):
    """GaussianMixture option parity, routed to the cluster lane's mixture
    driver in plain mode: covariance_type tied/diag/spherical, init
    'k-means++' with n_init 2, means_init and weights_init, and warm_start.
    The default GaussianMixture fit is the mixture lane's and is not here."""
    Z = X[:1500, 1:5]
    parts = {}
    last = None
    for ct in ("tied", "diag", "spherical"):
        m = ml.GaussianMixture(n_components=3, covariance_type=ct, max_iter=25, random_state=3).fit(Z)
        parts[ct + "_means"] = _h(m.means_)
        parts[ct + "_cov"] = _h(m.covariances_)
        parts[ct + "_lb"] = _h(np.float64(m.lower_bound_))
        last = m
    m = ml.GaussianMixture(n_components=3, init_params="k-means++", n_init=2, max_iter=25, random_state=3).fit(Z)
    parts["kpp_means"] = _h(m.means_)
    mi = np.ascontiguousarray(Z[:3], dtype=np.float32)
    m = ml.GaussianMixture(n_components=3, means_init=mi, weights_init=[0.2, 0.3, 0.5], max_iter=25,
                           random_state=3).fit(Z)
    parts["init_means"] = _h(m.means_)
    parts["init_weights"] = _h(m.weights_)
    m = ml.GaussianMixture(n_components=3, covariance_type="diag", warm_start=True, max_iter=5, random_state=3)
    m.fit(Z)
    m.fit(Z)
    parts["warm_means"] = _h(m.means_)
    return _fit(parts, last, lambda e: (e.score_samples(Xh[:256, 1:5]), e.predict(Xh[:256, 1:5])))


_batch_decl(_rows_calls("score_samples", "predict", sl=np.s_[:256, 1:5]), "x-cluster-gmm-options")


def _cluster_cosine_rows(A):
    """Columns 0-3 clipped and shifted off the origin: a zero row has no
    cosine distance and is refused by name (DEVIATION 5113)."""
    return np.ascontiguousarray(np.clip(A[:, :4], -50, 50) + np.float32(3.0), dtype=np.float32)


@lane("x-cluster-dbscan-metrics")
def _(ml, X, yc, yr, Xh=None):
    """DBSCAN option parity (dbscan/NOT_IMPLEMENTED.tsv): metric='cosine'
    (DEVIATION 5113: unit rows scaled on the host, the L2 kernel against
    2 * eps) with predict, and metric='precomputed' (DEVIATION 5114) on an
    exact integer L1 distance matrix, so pairs sit exactly at eps and the
    `<=` is reached. Rows are shifted off the origin for cosine (a zero row
    is refused by name)."""
    Z = _cluster_cosine_rows(X[:3000])
    m = ml.DBSCAN(eps=0.002, min_samples=5, metric="cosine", algorithm="brute",
                  prediction_data=True).fit(Z)
    parts = dict(cos_labels=_h(m.labels_), cos_core=_h(m.core_sample_indices_))
    Q = np.floor(np.clip(X[:1500, :3], -50, 50) * 2).astype(np.float64)
    D = (np.abs(Q[:, None, 0] - Q[None, :, 0]) + np.abs(Q[:, None, 1] - Q[None, :, 1])
         + np.abs(Q[:, None, 2] - Q[None, :, 2])).astype(np.float32)
    p = ml.DBSCAN(eps=2.0, min_samples=6, metric="precomputed").fit(np.ascontiguousarray(D))
    parts["pre_labels"] = _h(p.labels_)
    return _fit(parts, m, lambda e: (e.predict(_cluster_cosine_rows(Xh)[:256]),))


_batch_decl(_rows_calls("predict", prep=_cluster_cosine_rows, sl=np.s_[:64]), "x-cluster-dbscan-metrics")


@lane("x-cluster-hdbscan-epsilon")
def _(ml, X, yc, yr, Xh=None):
    """HDBSCAN option parity (hdbscan/NOT_IMPLEMENTED.tsv):
    cluster_selection_epsilon, cuML's epsilon search run on the host by the
    one function both routes call (DEVIATION 5115). The thresholds are
    multiples of the plain fit's median core distance, so the search merges
    clusters on every fixture's scale; eom, and leaf with
    allow_single_cluster (the walk's root arm and, at the largest, the
    labelling's epsilon branch, extract.cuh:148-153)."""
    Z = X[:2000, :4]
    base = ml.HDBSCAN(min_cluster_size=5).fit(Z)
    med = float(np.median(np.asarray(base.core_distances_, dtype=np.float32)))
    parts = dict(base_labels=_h(base.labels_))
    last = None
    for k in (1.0, 3.0, 30.0):
        e = float(np.float32(k * med))
        m = ml.HDBSCAN(min_cluster_size=5, cluster_selection_epsilon=e, prediction_data=True).fit(Z)
        parts[f"eom{k:g}_labels"] = _h(m.labels_)
        parts[f"eom{k:g}_n"] = _h(np.asarray([m.n_clusters_, m.n_outliers_], dtype=np.int64))
        last = m
        lf = ml.HDBSCAN(min_cluster_size=8, min_samples=3, cluster_selection_method="leaf",
                        allow_single_cluster=True, cluster_selection_epsilon=e).fit(Z)
        parts[f"leaf{k:g}_labels"] = _h(lf.labels_)
    return _fit(parts, last, lambda e: tuple(ml.hdbscan.approximate_predict(e, Xh[:256, :4])))


_batch_decl(_batch_hdbscan, "x-cluster-hdbscan-epsilon")
