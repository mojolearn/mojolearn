# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""identity_break lane bodies for the HDBSCAN door (workstream D,
2026-09-14), for the harness's owner to merge into tools/identity_break.py.
Sized like the dbscan lane (6000 rows of four columns; the brute-force
k-NN and the mutual reachability graph are n^2, the row cap is 46340).
Transductive, so the probe is n/a as DBSCAN's is.
"""


@lane("hdbscan")
def _(ml, X, yc, yr, Xh=None):
    """HDBSCAN (python/mojolearn/hdbscan.py), cuML's runner path at its
    defaults with excess-of-mass selection: the core distances, the
    Boruvka MST, single linkage, the condensed tree and the labels.
    Train hashes the labels, the core distances and the four integers the
    fit reports (cluster, outlier, Boruvka round and condensed cluster
    counts); the integer stages are where a divergence first shows."""
    m = ml.HDBSCAN(min_cluster_size=5).fit(X[:6000, :4])
    return _fit(dict(labels=_h(m.labels_), core=_h(m.core_distances_),
                     counts=_h(np.asarray([m.n_clusters_, m.n_outliers_, m.n_boruvka_rounds_, m.n_condensed_clusters_], dtype=np.int64))),
                m, "n/a:transductive")


@lane("hdbscan-leaf")
def _(ml, X, yc, yr, Xh=None):
    """The same fit under cluster_selection_method='leaf' with
    min_samples below min_cluster_size and allow_single_cluster, the
    other selection arm and the two knobs that change which condensed
    clusters become labels."""
    m = ml.HDBSCAN(min_cluster_size=8, min_samples=3, cluster_selection_method="leaf", allow_single_cluster=True).fit(X[:6000, :4])
    return _fit(dict(labels=_h(m.labels_), core=_h(m.core_distances_),
                     counts=_h(np.asarray([m.n_clusters_, m.n_outliers_, m.n_boruvka_rounds_, m.n_condensed_clusters_], dtype=np.int64))),
                m, "n/a:transductive")
