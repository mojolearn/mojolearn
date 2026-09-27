# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CLUSTER LANE'S PUBLIC DOOR (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md).

Owned by the `cluster` expansion lane. `mojolearn/__init__.py` imports this
module and makes every name in `__all__` public as `mojolearn.<name>`; a name
that is already public is refused at import. Put the classes here, or import
them here from the lane's own modules. Two optional hooks route a SAVED model
to the CPU (`mojolearn.host_model`); `_classical_host.py` merges them when it is
first imported, after the package, so both may rely on every module existing:

  CLASSICAL_HOST_BASENAMES     {"_mojolearn_x_cluster": "_mojolearn_x_cluster_host"}
  def classical_host_formats() -> {"mojolearn-<format>-1": {"Estimator": HostEstimator}}
                               import `_classical_host` INSIDE it and subclass
                               its `_HostBound`; never import it at module level
"""
import math

from . import _backend, _buffer
from ._array import Array
from ._mode import NumericModeMixin

__all__ = ["MiniBatchKMeans", "BisectingKMeans", "MeanShift", "OPTICS"]

# x_cluster/entries.mojo: the entry numbers of `x_cluster_call`
_E_NEAREST, _E_DISTANCES, _E_MINIBATCH, _E_BISECT, _E_BISECT_PREDICT, _E_MEANSHIFT, _E_OPTICS = 0, 1, 2, 3, 4, 5, 6


def _f32(X, name="X"):
    x, _ = _buffer.as_f32_c(X, ndim=2, name=name)
    if x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError(f"mojolearn: {name} must be a non-empty 2-D array, got shape {x.shape}")
    if not all(map(math.isfinite, memoryview(x).cast("B").cast("f"))):
        raise ValueError(f"mojolearn: {name} contains NaN or infinity")
    return x


def _seed(random_state):
    if random_state is None:
        return 0
    if isinstance(random_state, bool) or not isinstance(random_state, int) or random_state < 0:
        raise ValueError("mojolearn: random_state must be None or a non-negative int "
                         "(the lane's seeded host stream; a RandomState object is not taken)")
    return int(random_state) & ((1 << 63) - 1)


class _XCluster(NumericModeMixin):
    """The lane's shared call: `x_cluster_call` on `_mojolearn_x_cluster`
    (the CPU host binding `_mojolearn_x_cluster_host` on a CPU-only install)."""
    _BINDING = "_mojolearn_x_cluster"

    def _call(self, which, x, a, ip, fp=()):
        b = self._bind()
        xa = _buffer.addr_ro(x, name="X") if x is not None else 0
        xn = x.size if x is not None else 0
        aa = _buffer.addr_ro(a, name="aux") if a is not None and a.size else 0
        an = a.size if a is not None else 0
        f, i, s = b.x_cluster_call(which, xa, xn, aa, an, [int(v) for v in ip], [float(v) for v in fp])
        return f, i, s

    def _check_fitted(self, attr):
        if getattr(self, attr, None) is None:
            raise RuntimeError(f"this {type(self).__name__} is not fitted yet")

    def _input_like_fit(self, X):
        x = _f32(X)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"mojolearn: X has {x.shape[1]} features, {type(self).__name__} "
                             f"was fitted with {self.n_features_in_}")
        return x

    def _nearest(self, x, centers):
        n, d = x.shape
        f, i, _ = self._call(_E_NEAREST, x, centers, [n, centers.shape[0], d])
        return Array._from_flat(i[0], (n,), "<i4"), f[0]

    def _distances(self, x, centers):
        n, d = x.shape
        k = centers.shape[0]
        f, _, _ = self._call(_E_DISTANCES, x, centers, [n, k, d])
        return Array._from_flat(f[0], (n, k), "<f4")


class _CentersMixin:
    """predict / transform / score against `cluster_centers_` (squared
    euclidean nearest, the lowest index on a tie)."""

    def predict(self, X):
        self._check_fitted("cluster_centers_")
        return self._nearest(self._input_like_fit(X), self.cluster_centers_)[0]

    def transform(self, X):
        self._check_fitted("cluster_centers_")
        return self._distances(self._input_like_fit(X), self.cluster_centers_)

    def fit_predict(self, X, y=None, sample_weight=None):
        return self.fit(X).labels_

    def fit_transform(self, X, y=None, sample_weight=None):
        return self.fit(X).transform(X)

    def score(self, X, y=None, sample_weight=None):
        self._check_fitted("cluster_centers_")
        _, d = self._nearest(self._input_like_fit(X), self.cluster_centers_)
        total = 0.0
        for v in d:
            total += v
        return -total


class MiniBatchKMeans(_CentersMixin, _XCluster):
    """Mini-batch k-means. Reference: scikit-learn `cluster/_kmeans.py`
    (MiniBatchKMeans), `_k_means_minibatch.pyx::update_center_dense`.

    The batch assignment runs on the device; the batch draws, the center
    update in scikit-learn's order, the random reassignment of low-count
    centers and the early stop (exponentially weighted batch inertia,
    `max_no_improvement`, `tol`) are host code. The batch order comes from
    `random_state` through the lane's splitmix64 stream, not NumPy's Mersenne
    Twister, so a fit matches scikit-learn at a tolerance, not bit for bit.
    `random_state=None` means 0. `init` is 'k-means++' (greedy, as
    scikit-learn) or an (n_clusters, n_features) array; 'random', a callable,
    `sample_weight` and `partial_fit` are refused by name
    (x_cluster/NOT_IMPLEMENTED.tsv)."""

    def __init__(self, n_clusters=8, *, init="k-means++", max_iter=100, batch_size=1024,
                 verbose=0, compute_labels=True, random_state=None, tol=0.0,
                 max_no_improvement=10, init_size=None, n_init="auto", reassignment_ratio=0.01):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.verbose = verbose
        self.compute_labels = compute_labels
        self.random_state = random_state
        self.tol = tol
        self.max_no_improvement = max_no_improvement
        self.init_size = init_size
        self.n_init = n_init
        self.reassignment_ratio = reassignment_ratio

    def fit(self, X, y=None, sample_weight=None):
        if sample_weight is not None:
            raise NotImplementedError("mojolearn MiniBatchKMeans: sample_weight is not implemented "
                                      "(x_cluster/NOT_IMPLEMENTED.tsv)")
        x = _f32(X)
        n, d = x.shape
        k = int(self.n_clusters)
        if k < 1 or k > n:
            raise ValueError(f"n_samples={n} should be >= n_clusters={k}.")
        init_arr = None
        if isinstance(self.init, str):
            if self.init != "k-means++":
                raise NotImplementedError(f"mojolearn MiniBatchKMeans: init={self.init!r} is not "
                                          "implemented; 'k-means++' or an array (x_cluster/NOT_IMPLEMENTED.tsv)")
        elif callable(self.init):
            raise NotImplementedError("mojolearn MiniBatchKMeans: a callable init is not implemented")
        else:
            init_arr = _f32(self.init, "init")
            if init_arr.shape != (k, d):
                raise ValueError(f"The shape of the initial centers {init_arr.shape} does not match "
                                 f"the number of clusters {k} and features {d}.")
        n_init = self.n_init
        if n_init == "auto":
            n_init = 1 if init_arr is None else 1
        n_init = int(n_init)
        if n_init < 1:
            raise ValueError("n_init must be >= 1")
        mni = -1 if self.max_no_improvement is None else int(self.max_no_improvement)
        ip = [n, d, k, int(self.max_iter), int(self.batch_size), mni,
              0 if self.init_size is None else int(self.init_size), n_init,
              1 if init_arr is not None else 0, _seed(self.random_state)]
        f, i, s = self._call(_E_MINIBATCH, x, init_arr, ip, [float(self.tol), float(self.reassignment_ratio)])
        self.cluster_centers_ = Array._from_flat(f[0], (k, d), "<f4")
        self.counts_ = Array._from_flat(f[1], (k,), "<f4")
        self.labels_ = Array._from_flat(i[0], (n,), "<i4")
        self.inertia_ = float(s[0])
        self.n_steps_ = int(s[1])
        self.n_iter_ = int(s[2])
        self.n_features_in_ = d
        return self


class BisectingKMeans(_CentersMixin, _XCluster):
    """Bisecting k-means. Reference: scikit-learn `cluster/_bisect_k_means.py`.

    From one cluster, `n_clusters - 1` times the leaf with the highest score
    ('biggest_inertia': its inertia; 'largest_cluster': its size) is split in
    two by k-means, `n_init` restarts. THE 2-MEANS IS THIS LIBRARY'S KMeans
    (cuVS's Lloyd, the kmeans identity lanes' pair), so `tol` is cuVS's
    centroid-shift rule and `init='random'` / 'k-means++' are cuVS's
    starts, and a fit matches scikit-learn at a tolerance. `predict`
    descends the bisection tree (the nearer child center, the left on a
    tie), as scikit-learn's. `random_state=None` means 0."""

    def __init__(self, n_clusters=8, *, init="random", n_init=1, random_state=None, max_iter=300,
                 verbose=0, tol=1e-4, copy_x=True, algorithm="lloyd", bisecting_strategy="biggest_inertia"):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.random_state = random_state
        self.max_iter = max_iter
        self.verbose = verbose
        self.tol = tol
        self.copy_x = copy_x
        self.algorithm = algorithm
        self.bisecting_strategy = bisecting_strategy

    def fit(self, X, y=None, sample_weight=None):
        if sample_weight is not None:
            raise NotImplementedError("mojolearn BisectingKMeans: sample_weight is not implemented "
                                      "(x_cluster/NOT_IMPLEMENTED.tsv)")
        if self.init not in ("random", "k-means++"):
            raise NotImplementedError(f"mojolearn BisectingKMeans: init={self.init!r} is not implemented; "
                                      "'random' or 'k-means++'")
        if self.bisecting_strategy not in ("biggest_inertia", "largest_cluster"):
            raise ValueError(f"bisecting_strategy must be 'biggest_inertia' or 'largest_cluster', "
                             f"got {self.bisecting_strategy!r}")
        if self.algorithm not in ("lloyd", "elkan"):
            raise ValueError(f"algorithm must be 'lloyd' or 'elkan', got {self.algorithm!r}")
        x = _f32(X)
        n, d = x.shape
        k = int(self.n_clusters)
        if k < 1 or k > n:
            raise ValueError(f"n_samples={n} should be >= n_clusters={k}.")
        if int(self.n_init) < 1:
            raise ValueError("n_init must be >= 1")
        ip = [n, d, k, int(self.n_init), 1 if self.init == "random" else 0, int(self.max_iter),
              _seed(self.random_state), 1 if self.bisecting_strategy == "largest_cluster" else 0]
        f, i, s = self._call(_E_BISECT, x, None, ip, [float(self.tol)])
        self.cluster_centers_ = Array._from_flat(f[0], (k, d), "<f4")
        m = len(i[1]) // 3
        self._tree_centers = Array._from_flat(f[1], (m, d), "<f4")
        self._tree_nodes = Array._from_flat(i[1], (m, 3), "<i4")
        self.labels_ = Array._from_flat(i[0], (n,), "<i4")
        self.inertia_ = float(s[0])
        self.n_features_in_ = d
        return self

    def predict(self, X):
        self._check_fitted("cluster_centers_")
        x = self._input_like_fit(X)
        n, d = x.shape
        nodes = [int(v) for v in memoryview(self._tree_nodes).cast("B").cast("i")]
        _, i, _ = self._call(_E_BISECT_PREDICT, x, self._tree_centers, [n, d] + nodes)
        return Array._from_flat(i[0], (n,), "<i4")


class MeanShift(_XCluster):
    """Mean shift with a flat kernel. Reference: scikit-learn
    `cluster/_mean_shift.py`.

    Every seed's shift loop is one device thread over all rows in order; the
    bandwidth, when None, is scikit-learn's `estimate_bandwidth` (quantile
    0.3, all rows) from the device's exact row order statistic. The center
    merge (by intensity, then coordinates, a center within the bandwidth of
    a stronger one dropped) and the labels (nearest center, the lowest index
    on a tie) follow scikit-learn. `bin_seeding=True` is refused by name
    (x_cluster/NOT_IMPLEMENTED.tsv). `bandwidth_` records the bandwidth used."""

    def __init__(self, *, bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1,
                 cluster_all=True, n_jobs=None, max_iter=300):
        self.bandwidth = bandwidth
        self.seeds = seeds
        self.bin_seeding = bin_seeding
        self.min_bin_freq = min_bin_freq
        self.cluster_all = cluster_all
        self.n_jobs = n_jobs
        self.max_iter = max_iter

    def fit(self, X, y=None):
        if self.bin_seeding:
            raise NotImplementedError("mojolearn MeanShift: bin_seeding=True is not implemented "
                                      "(x_cluster/NOT_IMPLEMENTED.tsv)")
        x = _f32(X)
        n, d = x.shape
        seeds = None
        if self.seeds is not None:
            seeds = _f32(self.seeds, "seeds")
            if seeds.shape[1] != d:
                raise ValueError(f"seeds have {seeds.shape[1]} features, X has {d}")
        bw = 0.0
        if self.bandwidth is not None:
            bw = float(self.bandwidth)
            if not bw > 0:
                raise ValueError(f"bandwidth needs to be greater than zero or None, got {bw:f}")
        ip = [n, d, 0 if seeds is None else seeds.shape[0], 1 if self.cluster_all else 0, int(self.max_iter)]
        f, i, s = self._call(_E_MEANSHIFT, x, seeds, ip, [bw])
        kc = int(s[2])
        self.cluster_centers_ = Array._from_flat(f[0], (kc, d), "<f4")
        self.labels_ = Array._from_flat(i[0], (n,), "<i4")
        self.bandwidth_ = float(s[0])
        self.n_iter_ = int(s[1])
        self.n_features_in_ = d
        return self

    def predict(self, X):
        self._check_fitted("cluster_centers_")
        return self._nearest(self._input_like_fit(X), self.cluster_centers_)[0]

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_


class OPTICS(_XCluster):
    """OPTICS. Reference: scikit-learn `cluster/_optics.py`.

    Euclidean only (metric 'minkowski' with p=2, or 'euclidean'). The n x n
    distances and the core distances are the device's; the ordering loop is
    the reference's sequential one with the lowest index on a reachability
    tie; the xi and dbscan extractions are the reference's. Float32
    throughout, and without the reference's rounding of the distances to
    float precision, so reachability agrees at a tolerance. Transductive:
    no predict, as in scikit-learn."""

    def __init__(self, *, min_samples=5, max_eps=float("inf"), metric="minkowski", p=2,
                 metric_params=None, cluster_method="xi", eps=None, xi=0.05,
                 predecessor_correction=True, min_cluster_size=None, algorithm="auto",
                 leaf_size=30, memory=None, n_jobs=None):
        self.min_samples = min_samples
        self.max_eps = max_eps
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.cluster_method = cluster_method
        self.eps = eps
        self.xi = xi
        self.predecessor_correction = predecessor_correction
        self.min_cluster_size = min_cluster_size
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.memory = memory
        self.n_jobs = n_jobs

    @staticmethod
    def _size(v, n, name):
        if isinstance(v, bool) or v is None:
            raise ValueError(f"{name} must be an int >= 2 or a float in (0, 1]")
        if isinstance(v, int) or (isinstance(v, float) and v > 1):
            if int(v) != v or v < 2 or v > n:
                raise ValueError(f"{name} must be no greater than the number of samples ({n}) and >= 2, got {v}")
            return int(v)
        if not 0 < v <= 1:
            raise ValueError(f"{name} must be in (0, 1], got {v}")
        return max(2, int(v * n))

    def fit(self, X, y=None):
        if not (self.metric == "euclidean" or (self.metric == "minkowski" and self.p == 2)):
            raise NotImplementedError(f"mojolearn OPTICS: metric={self.metric!r} p={self.p!r} is not "
                                      "implemented; euclidean only (x_cluster/NOT_IMPLEMENTED.tsv)")
        if self.cluster_method not in ("xi", "dbscan"):
            raise ValueError(f"cluster_method must be 'xi' or 'dbscan', got {self.cluster_method!r}")
        x = _f32(X)
        n, d = x.shape
        ms = self._size(self.min_samples, n, "min_samples")
        mcs = ms if self.min_cluster_size is None else self._size(self.min_cluster_size, n, "min_cluster_size")
        max_eps = float(self.max_eps)
        eps = max_eps if self.eps is None else float(self.eps)
        if self.cluster_method == "dbscan" and eps > max_eps:
            raise ValueError(f"Specify an epsilon smaller than {max_eps}. Got {eps}.")
        if not 0 <= float(self.xi) <= 1:
            raise ValueError("xi must be in [0, 1]")
        ip = [n, d, ms, mcs, 0 if self.cluster_method == "xi" else 1, 1 if self.predecessor_correction else 0]
        f, i, _ = self._call(_E_OPTICS, x, None, ip, [max_eps, float(self.xi), eps])
        self.core_distances_ = Array._from_flat(f[0], (n,), "<f4")
        self.reachability_ = Array._from_flat(f[1], (n,), "<f4")
        self.ordering_ = Array._from_flat(i[0], (n,), "<i4")
        self.predecessor_ = Array._from_flat(i[1], (n,), "<i4")
        self.labels_ = Array._from_flat(i[2], (n,), "<i4")
        if self.cluster_method == "xi":
            self.cluster_hierarchy_ = Array._from_flat(i[3], (len(i[3]) // 2, 2), "<i4")
        self.n_features_in_ = d
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_
