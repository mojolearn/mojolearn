# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`HDBSCAN` on the GPU. Reference: cuML (workstream D, 2026-09-14).

The Python door of `hdbscan/estimator.mojo::hdbscan_fit_host` through
`bindings/_mojolearn_hdbscan.mojo`. The path follows the steps of cuML's
`runner.h:152-234` (brute-force k-NN, Boruvka MST, single linkage, the
condensed tree, excess-of-mass or leaf selection); the lane README lists
what is mirrored and what is refused.

WHAT IS REFUSED, AND WHERE (the estimator's header):
    metric other than euclidean          here by name, and on the Mojo host
    cluster_selection_epsilon != 0.0     Mojo host (rung 2)
    min_samples < 1 or > n_rows          Mojo host
    min_cluster_size < 2 or > n_rows     Mojo host
    alpha <= 0 or non-finite             Mojo host
    n_rows < 2, n_rows > 46340           Mojo host
    a NaN or infinite anywhere           Mojo host (DEVIATION 1607)
    probabilities_                       here, by name (DEVIATION 1610)

Transductive: there is no `predict`, exactly as `DBSCAN` here has none,
and there is no model to save.

NO SPEED CLAIM. The lane has no published number and this door adds none.
"""
from . import _backend
from ._buffer import addr, addr_ro, as_f32_c, empty
from ._mode import NumericModeMixin

_MODE_CODE = {"fast": 0, "identical": 1, "deterministic": 2}

#: `hierarchy/impl/cluster/detail/connectivities.mojo::DISTANCE_L2_SQRT_EXPANDED`,
#: the one metric `fit_hdbscan` accepts (their RAFT_EXPECTS).
METRIC_L2_SQRT_EXPANDED = 1

#: `hdbscan/impl/detail/select.mojo`'s codes.
SELECTION_EOM = 0
SELECTION_LEAF = 1

_METRICS = {"euclidean": METRIC_L2_SQRT_EXPANDED, "l2": METRIC_L2_SQRT_EXPANDED}
_SELECTION = {"eom": SELECTION_EOM, "leaf": SELECTION_LEAF}


class HDBSCAN(NumericModeMixin):
    """`cuml.cluster.HDBSCAN` semantics on the GPU.

    Parameters
    ----------
    min_cluster_size : int, default 5
    min_samples : int or None, default None
        None means `min_cluster_size`, cuML's and scikit-learn's rule.
    cluster_selection_epsilon : float, default 0.0
        Only 0.0 is implemented; anything else is refused by name on the
        Mojo host.
    max_cluster_size : int, default 0
        0 means no cap.
    metric : {'euclidean'}, default 'euclidean'
    alpha : float, default 1.0
    cluster_selection_method : {'eom', 'leaf'}, default 'eom'
    allow_single_cluster : bool, default False

    Attributes
    ----------
    labels_ : Array (n_samples,) int32
        `0 .. n_clusters_-1`, `-1` for noise; cuML's numbering (ascending
        condensed cluster id through `label_map`).
    core_distances_ : Array (n_samples,) float32
        Distance to the `min_samples`-th nearest neighbor excluding self.
    n_clusters_ : int
    n_outliers_ : int
    n_boruvka_rounds_ : int
        An integer card stage.
    n_condensed_clusters_ : int
    probabilities_
        NOT IMPLEMENTED; reading it raises by name (DEVIATION 1610).
    """

    _BINDING = "_mojolearn_hdbscan"

    def __init__(
        self,
        min_cluster_size=5,
        min_samples=None,
        cluster_selection_epsilon=0.0,
        max_cluster_size=0,
        metric="euclidean",
        alpha=1.0,
        cluster_selection_method="eom",
        allow_single_cluster=False,
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.max_cluster_size = max_cluster_size
        self.metric = metric
        self.alpha = alpha
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster

    def _extension(self):
        mod = self._bind()
        want = getattr(self, "numeric_mode", None) or _backend.default_mode()
        fn = getattr(mod, "hdbscan_numeric_mode", None)
        if fn is not None:
            got = int(fn())
            if got != _MODE_CODE.get(want):
                raise RuntimeError(
                    f"mojolearn HDBSCAN: numeric_mode={want!r} was requested but "
                    f"{mod.__name__} reports compile-time mode code {got}; rebuild "
                    "it with bash bindings/build_hdbscan.sh"
                )
        return mod

    @property
    def probabilities_(self):
        raise AttributeError(
            "mojolearn HDBSCAN: probabilities_ is NOT IMPLEMENTED (DEVIATION 1610): "
            "cuML's Membership::get_probabilities is a segmented max over the "
            "condensed tree plus a per-point ratio; returning zeros or ones would "
            "be a number nobody computed. hdbscan/estimator.mojo names the closure."
        )

    def fit(self, X, y=None):
        """Cluster row-major `X`. Returns `self`."""
        x, _ = as_f32_c(X, ndim=2, name="X")
        n, d = x.shape
        for name in ("min_cluster_size", "max_cluster_size"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int):
                raise TypeError(f"mojolearn HDBSCAN: {name} must be an int, got {type(v).__name__}")
        if self.min_samples is None:
            min_samples = int(self.min_cluster_size)
        elif isinstance(self.min_samples, bool) or not isinstance(self.min_samples, int):
            raise TypeError("mojolearn HDBSCAN: min_samples must be an int or None")
        else:
            min_samples = int(self.min_samples)
        if isinstance(self.metric, str):
            if self.metric not in _METRICS:
                raise ValueError(
                    f"mojolearn HDBSCAN: metric must be 'euclidean' (cuML's "
                    f"L2SqrtExpanded, the one fit_hdbscan accepts), got {self.metric!r}"
                )
            metric = _METRICS[self.metric]
        else:
            metric = int(self.metric)
        if isinstance(self.cluster_selection_method, str):
            if self.cluster_selection_method not in _SELECTION:
                raise ValueError(
                    "mojolearn HDBSCAN: cluster_selection_method must be 'eom' or "
                    f"'leaf', got {self.cluster_selection_method!r}"
                )
            method = _SELECTION[self.cluster_selection_method]
        else:
            method = int(self.cluster_selection_method)
        for name in ("alpha", "cluster_selection_epsilon"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(f"mojolearn HDBSCAN: {name} must be a real number")
        labels = empty((n,), "<i4")
        core = empty((n,), "<f4")
        info = empty((4,), "<i4")
        n_clusters = self._extension().hdbscan_fit(
            # ORDER MATCHES bindings/_mojolearn_hdbscan.mojo::hdbscan_fit_binding.
            # x, labels_out, core_dists_out, info_out
            [addr_ro(x, name="X"), addr(labels, name="labels_"), addr(core, name="core_distances_"), addr(info, name="info")],
            # n, d, min_samples, min_cluster_size, max_cluster_size, alpha,
            # allow_single_cluster, cluster_selection_method, cluster_selection_epsilon, metric
            [n, d, min_samples, int(self.min_cluster_size), int(self.max_cluster_size), float(self.alpha),
             1 if self.allow_single_cluster else 0, method, float(self.cluster_selection_epsilon), metric],
        )
        self.n_features_in_ = d
        self.labels_ = labels
        self.core_distances_ = core
        self.n_clusters_ = int(n_clusters)
        self.n_outliers_ = int(info[1])
        self.n_boruvka_rounds_ = int(info[2])
        self.n_condensed_clusters_ = int(info[3])
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_


__all__ = ["HDBSCAN"]
