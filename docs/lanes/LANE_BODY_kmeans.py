# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""identity_break lane bodies for the two KMeans arms routed through
mojolearn.cluster (workstream D, 2026-09-14), for the harness's owner to
merge into tools/identity_break.py. The `kmeans` lane above them is the
recorded default and must not move: the params list grew by one slot at
its default value. Sized like the kmeans lane.
"""


@lane("kmeans-sqrt")
def _(ml, X, yc, yr, Xh=None):
    """metric='l2_sqrt_expanded': cuVS's L2SqrtExpanded, the root taken on
    the assignment's reduced distance (`metric_is_sqrt`, `identical_sqrt`,
    DEVIATION 2715) and so in the inertia, the row norms squared as under
    L2Expanded (DEVIATION 2716); a different inertia_ from the squared arm."""
    m = ml.KMeans(n_clusters=8, random_state=3, metric="l2_sqrt_expanded").fit(X)
    return _fit(dict(centers=_h(m.cluster_centers_), labels=_h(m.labels_),
                     inertia=_h(np.float64(m.inertia_)), scales=_h(np.asarray([m.sum_scale_, m.weight_scale_], dtype=np.float64))),
                m, "n/a:no-predict")


@lane("kmeans-classic-pp")
def _(ml, X, yc, yr, Xh=None):
    """oversampling_factor=0.0: the classic sequential k-means++ seeding
    (detail/kmeans.cuh:910-915's `== 0` arm), a different algorithm from
    the scalable k-means|| the default selects."""
    m = ml.KMeans(n_clusters=8, random_state=3, oversampling_factor=0.0).fit(X)
    return _fit(dict(centers=_h(m.cluster_centers_), labels=_h(m.labels_), inertia=_h(np.float64(m.inertia_))),
                m, "n/a:no-predict")


@lane("kmeans-cosine")
def _(ml, X, yc, yr, Xh=None):
    """metric='cosine' is routed and REFUSED BY NAME on the Mojo host
    (cluster/impl/kmeans_params.mojo::validate): the expected cell is the
    refusal sentence on every column, never a hash. A column that hashes
    here means the refusal was lifted without a fused cosine arm."""
    m = ml.KMeans(n_clusters=8, random_state=3, metric="cosine").fit(X)
    return _fit(dict(centers=_h(m.cluster_centers_)), m, "n/a:no-predict")
