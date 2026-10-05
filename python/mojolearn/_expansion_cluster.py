# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CLUSTER LANE'S PUBLIC DOOR.

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
import ctypes

from . import _portable_math as math

from . import _backend, _buffer
from . import _portable_math as _pm
from ._array import Array
from ._mode import NumericModeMixin

__all__ = ["MiniBatchKMeans", "BisectingKMeans", "MeanShift", "OPTICS", "AffinityPropagation", "BayesianGaussianMixture"]

# x_cluster/entries.mojo: the entry numbers of `x_cluster_call`
_E_NEAREST, _E_DISTANCES, _E_MINIBATCH, _E_BISECT, _E_BISECT_PREDICT, _E_MEANSHIFT, _E_OPTICS, _E_AFFINITY, _E_BGMM, _E_BGMM_SCORE, _E_MINIBATCH_PARTIAL = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10


def _f32(X, name="X", scan=True):
    x, _ = _buffer.as_f32_c(X, ndim=2, name=name)
    if x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError(f"mojolearn: {name} must be a non-empty 2-D array, got shape {x.shape}")
    # the base binding's native scan (`_buffer.all_finite`); scan=False only
    # when the binding scans X itself (MiniBatchKMeans, MBK_FAST_DEVSCAN)
    if scan and not _buffer.all_finite(x):
        raise ValueError(f"mojolearn: {name} contains NaN or infinity")
    return x


def _aux(*parts):
    """The parts (Arrays or array-likes, None skipped) end to end as one
    float32 Array: byte copies, no Python pass over the values (lane
    pyglue-numeric: the aux blocks were Python lists of every value)."""
    arrs = [_buffer.as_f32_c(p, ndim=None, name="aux")[0] for p in parts if p is not None]  # glue: aux parts as float32 blocks
    total = sum(a.size for a in arrs)   # glue: the few part sizes
    if not total:
        return None
    out = _buffer.empty((total,), "<f4")
    off = 0
    for a in arrs:                      # glue: one copy per part
        if a.size:
            ctypes.memmove(out._addr + 4 * off, _buffer.addr_ro(a, name="aux"), 4 * a.size)
        off += a.size
    return out


def _weights(sample_weight, n):
    """sample_weight as a float32 Array of n values (None stays None)."""
    if sample_weight is None:
        return None
    w = _buffer.as_f32_c(sample_weight, ndim=None, name="sample_weight")[0]
    if w.size != n:
        raise ValueError(f"sample_weight has {w.size} values for {n} samples")
    return w


def _seed(random_state):
    if random_state is None:
        return 0
    if isinstance(random_state, bool) or not isinstance(random_state, int) or random_state < 0:
        raise ValueError("mojolearn: random_state must be None or a non-negative int "
                         "(the lane's seeded host stream; a RandomState object is not taken)")
    return int(random_state) & ((1 << 63) - 1)


def _callable_init(init, X, k, random_state):
    """scikit-learn's callable `init` (`_kmeans.py::_init_centroids`:
    `init(X, n_clusters, random_state=random_state)`): called ONCE on the
    host with the fit's rows, a NumPy RandomState seeded by `random_state`
    when NumPy is present (the int otherwise), and its centers then take
    the array path, so the fit's bits depend only on what it returned.
    scikit-learn's MiniBatchKMeans hands it an `init_size` subsample drawn
    from its own stream; this hands it every row."""
    rs = random_state
    try:
        from ._optional_numpy import require_numpy
        _np = require_numpy('_expansion_cluster')
        X = _np.asarray(X)
        rs = _np.random.RandomState(random_state)
    except (ImportError, AttributeError):
        pass
    return init(X, k, random_state=rs)


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
        f, i, s = b.x_cluster_call(which, xa, xn, aa, an, [int(v) for v in ip], [float(v) for v in fp])  # glue: integer and float binding params
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
        x = self._input_like_fit(X)
        # the float-float fold of the distances, in the binding
        n, d = x.shape
        _, _, sc = self._call(_E_NEAREST, x, self.cluster_centers_, [n, self.cluster_centers_.shape[0], d, 1])
        return -sc[0]


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
    scikit-learn), 'random' (k distinct rows of the init sample, drawn by
    weight) or an (n_clusters, n_features) array; `sample_weight` weighs the
    k-means++ potentials, the init scoring and the batch draw, as
    scikit-learn's; `partial_fit` runs one step on the given rows with the
    stream carried between calls. A callable init is called once on the
    host and its centers take the array path (`_callable_init`)
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

    def _devscan(self):
        """lane/apple-fast-s-linalg MBK_FAST_DEVSCAN (FAST + Apple default, rollback _OFF;
        x_cluster/minibatch_ptr.mojo): whether the binding scans X for
        NaN/inf itself (on the device copy), so the host scan is skipped."""
        try:
            return int(self._bind().x_cluster_mbk_devscan()) == 1
        except Exception:
            return False

    def fit(self, X, y=None, sample_weight=None):
        devscan = self._devscan()
        x = _f32(X, scan=not devscan)
        n, d = x.shape
        k = int(self.n_clusters)
        if k < 1 or k > n:
            raise ValueError(f"n_samples={n} should be >= n_clusters={k}.")
        weights = _weights(sample_weight, n)
        init_arr = None
        if isinstance(self.init, str):
            if self.init not in ("k-means++", "random"):
                raise ValueError(f"init must be 'k-means++', 'random' or an array, got {self.init!r}")
        else:
            init_arr = _f32(_callable_init(self.init, X, k, self.random_state) if callable(self.init)
                            else self.init, "init")
            if init_arr.shape != (k, d):
                raise ValueError(f"The shape of the initial centers {init_arr.shape} does not match "
                                 f"the number of clusters {k} and features {d}.")
        n_init = self.n_init
        if n_init == "auto":
            n_init = 3 if (init_arr is None and self.init == "random") else 1
        n_init = int(n_init)
        if n_init < 1:
            raise ValueError("n_init must be >= 1")
        mni = -1 if self.max_no_improvement is None else int(self.max_no_improvement)
        ip = [n, d, k, int(self.max_iter), int(self.batch_size), mni,
              0 if self.init_size is None else int(self.init_size), n_init,
              1 if init_arr is not None else 0, _seed(self.random_state),
              1 if (init_arr is None and self.init == "random") else 0, 1 if weights is not None else 0]
        a = _aux(init_arr, weights)
        if devscan:
            ip.append(1)    # ip[12]: the binding owes the NaN/inf scan
        try:
            f, i, s = self._call(_E_MINIBATCH, x, a, ip, [float(self.tol), float(self.reassignment_ratio)])
        except Exception as e:
            if devscan and "contains NaN or infinity" in str(e):
                raise ValueError("mojolearn: X contains NaN or infinity") from None
            raise
        self.cluster_centers_ = Array._from_flat(f[0], (k, d), "<f4")
        self.counts_ = Array._from_flat(f[1], (k,), "<f4")
        self.labels_ = Array._from_flat(i[0], (n,), "<i4")
        self.inertia_ = float(s[0])
        self.n_steps_ = int(s[1])
        self.n_iter_ = int(s[2])
        self.n_features_in_ = d
        return self


    def partial_fit(self, X, y=None, sample_weight=None):
        """One mini-batch step on all of `X` (scikit-learn's partial_fit):
        the first call starts the centers ('k-means++', 'random' or the
        array) from `init_size` rows of `X`; the seeded stream and the
        reassignment counter carry over between calls."""
        x = _f32(X)
        n, d = x.shape
        k = int(self.n_clusters)
        first = getattr(self, "cluster_centers_", None) is None
        if not first and not hasattr(self, "_partial_state"):
            # a model from fit(): continue from its centers and counts, as sklearn
            n0 = min(int(self.batch_size), n)
            self._partial_batch = n0
            self._partial_init = (0, 0)
            seed = _seed(self.random_state)
            self._partial_state = ((seed >> 32, seed & 0xFFFFFFFF), 0)
        if not first and d != self.n_features_in_:
            raise ValueError(f"X has {d} features, MiniBatchKMeans was fitted with {self.n_features_in_}")
        aux = []                        # glue: the parts of the state block
        if first:
            if k < 1 or k > n:
                raise ValueError(f"n_samples={n} should be >= n_clusters={k}.")
            batch_eff = min(int(self.batch_size), n)
            init_size = 3 * batch_eff if self.init_size is None else int(self.init_size)
            if init_size < k:
                init_size = 3 * k
            init_size = min(init_size, n)
            if isinstance(self.init, str):
                if self.init not in ("k-means++", "random"):
                    raise ValueError(f"init must be 'k-means++', 'random' or an array, got {self.init!r}")
                mode = 0 if self.init == "k-means++" else 1
            else:
                ia = _f32(_callable_init(self.init, X, k, self.random_state) if callable(self.init)
                          else self.init, "init")
                if ia.shape != (k, d):
                    raise ValueError(f"The shape of the initial centers {ia.shape} does not match "
                                     f"the number of clusters {k} and features {d}.")
                aux.append(ia)
                mode = 2
            seed = _seed(self.random_state)
            state = (seed >> 32, seed & 0xFFFFFFFF)
            since = 0
            self._partial_batch = batch_eff
            self._partial_init = (mode, init_size)
            self.n_steps_ = 0
        else:
            mode, init_size = self._partial_init
            batch_eff = self._partial_batch
            state, since = self._partial_state
            aux += [self.cluster_centers_, self.counts_]
        aux.append(_weights(sample_weight, n))
        ip = [n, d, k, 1 if first else 0, mode, init_size, batch_eff, since, state[0], state[1],
              1 if sample_weight is not None else 0]
        a = _aux(*aux)
        f, i, s = self._call(_E_MINIBATCH_PARTIAL, x, a, ip, [float(self.reassignment_ratio)])
        self.cluster_centers_ = Array._from_flat(f[0], (k, d), "<f4")
        self.counts_ = Array._from_flat(f[1], (k,), "<f4")
        self.labels_ = Array._from_flat(i[0], (n,), "<i4")
        self.inertia_ = float(s[0])
        self._partial_state = ((int(s[2]), int(s[3])), int(s[1]))
        self.n_steps_ = getattr(self, "n_steps_", 0) + 1
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
        w = None
        if sample_weight is not None:
            w = _weights(sample_weight, n)
        f, i, s = self._call(_E_BISECT, x, w, ip, [float(self.tol)])
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
        nodes = [int(v) for v in _buffer.flat_bytes(self._tree_nodes).cast("i")]  # glue: fitted bisect tree node words
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
    on a tie) follow scikit-learn, and so does `bin_seeding` (the rows binned
    at the bandwidth, half to even, bins with at least `min_bin_freq` rows as
    seeds). `bandwidth_` records the bandwidth used."""

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
        ip = [n, d, 0 if seeds is None else seeds.shape[0], 1 if self.cluster_all else 0, int(self.max_iter),
              1 if self.bin_seeding else 0, int(self.min_bin_freq)]
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

    Metrics: 'minkowski' (any p >= 1), 'euclidean'/'l2', 'manhattan'/
    'cityblock'/'l1', 'chebyshev', 'cosine' and 'precomputed'. The n x n
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

    _METRICS = {"euclidean": 0, "l2": 0, "manhattan": 1, "cityblock": 1, "l1": 1, "chebyshev": 2,
                "infinity": 2, "cosine": 4, "precomputed": 5}

    def _metric_code(self):
        """(code, p): euclidean at p=2 keeps the squared-distance route (-1)."""
        m = self.metric
        if self.metric_params:
            p = float(self.metric_params.get("p", self.p))
        else:
            p = float(self.p) if self.p is not None else 2.0
        if m == "minkowski":
            if not p >= 1:
                raise ValueError(f"p must be >= 1 for minkowski, got {p}")
            if p == 2:
                return -1, 2.0
            if p == 1:
                return 1, 1.0
            if p == float("inf"):
                return 2, p
            return 3, p
        if m in ("euclidean", "l2"):
            return -1, 2.0
        if m not in self._METRICS:
            raise NotImplementedError(f"mojolearn OPTICS: metric={m!r} is not implemented; one of "
                                      f"{sorted(self._METRICS) + ['minkowski']} (x_cluster/NOT_IMPLEMENTED.tsv)")
        return self._METRICS[m], p

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
        metric, pw = self._metric_code()
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
        if metric == 5 and n != d:
            raise ValueError(f"metric='precomputed' needs a square distance matrix, got {(n, d)}")
        ip = [n, d, ms, mcs, 0 if self.cluster_method == "xi" else 1, 1 if self.predecessor_correction else 0, metric]
        f, i, _ = self._call(_E_OPTICS, x, None, ip, [max_eps, float(self.xi), eps, pw])
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


class AffinityPropagation(_XCluster):
    """Affinity propagation. Reference: scikit-learn
    `cluster/_affinity_propagation.py`.

    S is minus the squared euclidean distance (or the precomputed matrix);
    the default preference is its median. The responsibility and
    availability updates run on the device over the resident n x n
    matrices; the convergence window, the exemplar refinement and the labels
    are the reference's. The tie noise is the reference's formula with
    normals from the lane's seeded stream (`random_state=None` means 0), not
    NumPy's, so a fit matches scikit-learn at a tolerance."""

    def __init__(self, *, damping=0.5, max_iter=200, convergence_iter=15, copy=True,
                 preference=None, affinity="euclidean", verbose=False, random_state=None):
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.copy = copy
        self.preference = preference
        self.affinity = affinity
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y=None):
        if self.affinity not in ("euclidean", "precomputed"):
            raise ValueError(f"affinity must be 'euclidean' or 'precomputed', got {self.affinity!r}")
        if not 0.5 <= float(self.damping) < 1:
            raise ValueError(f"damping must be in [0.5, 1), got {self.damping}")
        if int(self.max_iter) < 1 or int(self.convergence_iter) < 1:
            raise ValueError("max_iter and convergence_iter must be >= 1")
        x = _f32(X)
        n, d = x.shape
        pre = self.affinity == "precomputed"
        if pre and n != d:
            raise ValueError(f"The matrix of similarities must be a square array. Got {(n, d)} instead.")
        pref_arr, mode, scalar = None, 0, 0.0
        if self.preference is not None:
            if isinstance(self.preference, (int, float)):
                mode, scalar = 1, float(self.preference)
            else:
                pref_arr = _f32([list(self.preference)], "preference")
                if pref_arr.shape[1] != n:
                    raise ValueError("preference must be a scalar or have one value per sample")
                mode = 2
        ip = [n, d, 1 if pre else 0, mode, int(self.max_iter), int(self.convergence_iter), _seed(self.random_state)]
        f, i, s = self._call(_E_AFFINITY, x, pref_arr, ip, [float(self.damping), scalar])
        k = len(i[0])
        self.cluster_centers_indices_ = Array._from_flat(i[0], (k,), "<i4")
        self.labels_ = Array._from_flat(i[1], (n,), "<i4")
        self.affinity_matrix_ = Array._from_flat(f[0], (n, n), "<f4")
        # the final availability and responsibility diagonals (a diagnostic the
        # verifier lanes hash; scikit-learn keeps no such attribute)
        self._message_diag = Array._from_flat(f[1], (len(f[1]),), "<f4")
        self.n_iter_ = int(s[0])
        self.n_features_in_ = d
        if not pre:
            self.cluster_centers_ = Array._from_flat(f[2], (k, d), "<f4")   # the entry's exemplar rows
        return self

    def predict(self, X):
        self._check_fitted("labels_")
        if self.affinity == "precomputed":
            raise ValueError("Predict method is not supported when affinity='precomputed'.")
        x = self._input_like_fit(X)
        if self.cluster_centers_.shape[0] == 0:
            return _buffer.full((x.shape[0],), -1, "<i4")
        return self._nearest(x, self.cluster_centers_)[0]

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_


class BayesianGaussianMixture(_XCluster):
    """Variational Bayesian Gaussian mixture, every covariance type. Reference:
    scikit-learn `mixture/_bayesian_mixture.py` with `_base.py`'s EM loop.

    The n-sized work (the Mahalanobis squares, the E-step log-sum-exp, the
    M-step moments) runs on the device in float32; the Wishart and
    Dirichlet(-process) updates, the d x d Cholesky, digamma and log-gamma
    and the lower bound run on the device too, in float-float (about 48
    bits; x_cluster/bgmm_device.mojo), as do the default priors. `init_params` is 'kmeans' (this
    library's KMeans), 'random', 'k-means++' or 'random_from_data' (the
    lane's seeded stream), as scikit-learn's;
    `random_state=None` means 0. covariance_type 'full', 'tied', 'diag'
    and 'spherical' all run through full d x d factors (the zeros add
    nothing). `warm_start=True` resumes from the previous fit's parameters
    (one run, the lower bound continuing), as scikit-learn's."""

    def __init__(self, *, n_components=1, covariance_type="full", tol=1e-3, reg_covar=1e-6,
                 max_iter=100, n_init=1, init_params="kmeans",
                 weight_concentration_prior_type="dirichlet_process", weight_concentration_prior=None,
                 mean_precision_prior=None, mean_prior=None, degrees_of_freedom_prior=None,
                 covariance_prior=None, random_state=None, warm_start=False, verbose=0,
                 verbose_interval=10):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.weight_concentration_prior_type = weight_concentration_prior_type
        self.weight_concentration_prior = weight_concentration_prior
        self.mean_precision_prior = mean_precision_prior
        self.mean_prior = mean_prior
        self.degrees_of_freedom_prior = degrees_of_freedom_prior
        self.covariance_prior = covariance_prior
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval

    def fit(self, X, y=None):
        self._fit(X)
        return self

    def fit_predict(self, X, y=None):
        return self._fit(X)

    def _fit(self, X):
        ctypes = {"full": 0, "tied": 1, "diag": 2, "spherical": 3}
        if self.covariance_type not in ctypes:
            raise ValueError(f"covariance_type must be one of {sorted(ctypes)}, got {self.covariance_type!r}")
        ct = ctypes[self.covariance_type]
        inits = {"kmeans": 0, "random": 1, "k-means++": 2, "random_from_data": 3}
        if self.init_params not in inits:
            raise ValueError(f"init_params must be one of {sorted(inits)}, got {self.init_params!r}")
        if self.weight_concentration_prior_type not in ("dirichlet_process", "dirichlet_distribution"):
            raise ValueError("weight_concentration_prior_type must be 'dirichlet_process' or "
                             "'dirichlet_distribution'")
        x = _f32(X)
        n, d = x.shape
        k = int(self.n_components)
        if k < 1 or n < k:
            raise ValueError(f"Expected n_samples >= n_components but got n_components = {k}, n_samples = {n}")
        if int(self.n_init) < 1 or int(self.max_iter) < 1:
            raise ValueError("n_init and max_iter must be >= 1")
        # glue: the priors and any warm state as byte copies (`_aux`); the
        # binding expands a diag or spherical covariance prior to d x d
        # (`x_cluster/entries.mojo::bgmm_entry`, ip[9] = 2 or 3)
        parts = []
        cov_mode = 0
        if self.mean_prior is not None:
            mp = _buffer.as_f32_c(self.mean_prior, ndim=None, name="mean_prior")[0]
            if mp.size != d:
                raise ValueError(f"The parameter 'means' should have the shape of ({d},)")
            parts.append(mp)
        if self.covariance_prior is not None:
            if ct in (0, 1):
                cp = _f32(self.covariance_prior, "covariance_prior")
                if cp.shape != (d, d):
                    raise ValueError(f"The parameter '{self.covariance_type} covariance prior' should have the "
                                     f"shape of ({d}, {d})")
                cov_mode = 1
            elif ct == 2:
                cp = _buffer.as_f32_c(self.covariance_prior, ndim=None, name="covariance_prior")[0]
                if cp.size != d:
                    raise ValueError(f"The parameter 'diag covariance prior' should have the shape of ({d},)")
                cov_mode = 2
            else:
                cp = _buffer.as_f32_c([float(self.covariance_prior)], ndim=None, name="covariance_prior")[0]
                cov_mode = 3
            parts.append(cp)
        prev = getattr(self, "_warm", None)
        warm = bool(self.warm_start) and prev is not None and prev[0] == (k, d, self.covariance_type)
        if warm:
            parts.append(self._warm[1])
        a = _aux(*parts)
        none = -1.0
        fp = [none if self.weight_concentration_prior is None else float(self.weight_concentration_prior),
              none if self.mean_precision_prior is None else float(self.mean_precision_prior),
              none if self.degrees_of_freedom_prior is None else float(self.degrees_of_freedom_prior),
              float(self.reg_covar), float(self.tol), float(getattr(self, "lower_bound_", 0.0)) if warm else 0.0]
        ip = [n, d, k, 1 if self.weight_concentration_prior_type == "dirichlet_process" else 0,
              int(self.max_iter), int(self.n_init), inits[self.init_params],
              _seed(self.random_state), 1 if self.mean_prior is not None else 0,
              cov_mode, ct, 1 if warm else 0]
        f, i, s = self._call(_E_BGMM, x, a, ip, fp)
        self.weights_ = Array._from_flat(f[0], (k,), "<f4")
        self.means_ = Array._from_flat(f[1], (k, d), "<f4")
        self._full_pchol = Array._from_flat(f[3], (k, d, d), "<f4")
        # f[13], f[14], f[15]: the binding's by-type forms (bgmm_entry)
        self.covariances_ = self._by_type(f[13], k, d, ct)
        self.precisions_cholesky_ = self._by_type(f[14], k, d, ct)
        if self.weight_concentration_prior_type == "dirichlet_process":
            self.weight_concentration_ = (Array._from_flat(f[4], (k,), "<f4"), Array._from_flat(f[5], (k,), "<f4"))
        else:
            self.weight_concentration_ = Array._from_flat(f[4], (k,), "<f4")
        self.mean_precision_ = Array._from_flat(f[6], (k,), "<f4")
        self.degrees_of_freedom_ = (float(f[7][0]) if ct == 1 else Array._from_flat(f[7], (k,), "<f4"))
        self._log_consts = Array._from_flat(f[8], (k,), "<f4")
        self.mean_prior_ = Array._from_flat(f[9], (d,), "<f4")
        cpf = f[15]
        if ct == 2:
            self.covariance_prior_ = Array._from_flat(cpf, (d,), "<f4")
        elif ct == 3:
            self.covariance_prior_ = float(cpf[0])
        else:
            self.covariance_prior_ = Array._from_flat(cpf, (d, d), "<f4")
        self.lower_bound_ = float(s[0])
        self.n_iter_ = int(s[1])
        self.converged_ = bool(s[2])
        # the state a warm_start refit resumes from (full per-component matrices)
        self._warm = ((k, d, self.covariance_type),
                      list(f[11]) + list(f[4]) + (list(f[5]) if len(f[5]) else [0.0] * k) + list(f[6])
                      + list(f[1]) + list(f[7]) + list(f[2]) + list(f[3]))
        self.weight_concentration_prior_ = float(s[3])
        self.mean_precision_prior_ = float(s[4])
        self.degrees_of_freedom_prior_ = float(s[5])
        self.n_features_in_ = d
        return Array._from_flat(i[0], (n,), "<i4")

    @staticmethod
    def _by_type(flat, k, d, ct):
        """The binding's by-type values (`bgmm_entry` f[13], f[14]: full
        k x d x d, tied d x d, diag k x d, spherical k) under the covariance
        type's shape (as scikit-learn's attributes). Shapes only."""
        if ct == 0:
            return Array._from_flat(flat, (k, d, d), "<f4")
        if ct == 1:
            return Array._from_flat(flat, (d, d), "<f4")
        if ct == 2:
            return Array._from_flat(flat, (k, d), "<f4")
        return Array._from_flat(flat, (k,), "<f4")

    def _score(self, X, flags=0):
        self._check_fitted("means_")
        x = self._input_like_fit(X)
        n, d = x.shape
        k = self.means_.shape[0]
        ip = [n, d, k, flags] if flags else [n, d, k]
        # glue: the model block as byte copies (`_aux`)
        f, i, sc = self._call(_E_BGMM_SCORE, x, _aux(self.means_, self._full_pchol, self._log_consts), ip)
        if flags:
            return f, sc, n, k, i[0]
        return f[0], f[1], n, k, i[0]

    def predict(self, X):
        # the labels come from the device score (`bodies.argmax_row`: the
        # first largest log responsibility, the lowest index on a tie)
        _, _, n, _, labels = self._score(X)
        return Array._from_flat(labels, (n,), "<i4")

    def predict_proba(self, X):
        # the exp of every log responsibility in the binding (bodies.exp_cell)
        f, _, n, k, _ = self._score(X, 1)
        return Array._from_flat(f[2], (n, k), "<f4")

    def score_samples(self, X):
        _, lpn, n, _, _ = self._score(X)
        return Array._from_flat(lpn, (n,), "<f4")

    def score(self, X, y=None):
        # the float-float fold of log_prob_norm in the binding
        _, sc, n, _, _ = self._score(X, 2)
        return sc[0] / n



# ------------------------------------------------ GaussianMixture option parity
_GMM_CTYPES = {"full": 0, "tied": 1, "diag": 2, "spherical": 3}
_GMM_INITS = {"kmeans": 0, "random": 1, "k-means++": 2, "random_from_data": 3}


def _gmm_needs_ext(est):
    """True when a GaussianMixture fit asks for an option the mixture binding
    does not carry (it refuses them by name there)."""
    return (est.covariance_type != "full" or est.init_params in ("k-means++", "random_from_data")
            or int(est.n_init) != 1 or bool(est.warm_start) or est.weights_init is not None
            or est.means_init is not None or est.precisions_init is not None)


class _GmmExtCall(_XCluster):
    def __init__(self, numeric_mode=None):
        self.numeric_mode = numeric_mode


def _gmm_ext_fit(est, X):
    """sklearn GaussianMixture on the cluster lane's mixture driver (plain
    mode): every covariance type, init, n_init, warm_start and the three
    *_init arrays."""
    if est.covariance_type not in _GMM_CTYPES:
        raise ValueError(f"covariance_type must be one of {sorted(_GMM_CTYPES)}, got {est.covariance_type!r}")
    if est.init_params not in _GMM_INITS:
        raise ValueError(f"init_params must be one of {sorted(_GMM_INITS)}, got {est.init_params!r}")
    x = _f32(X)
    n, d = x.shape
    k = int(est.n_components)
    if k < 1 or n < k:
        raise ValueError(f"Expected n_samples >= n_components but got n_components = {k}, n_samples = {n}")
    ct = _GMM_CTYPES[est.covariance_type]
    call = _GmmExtCall(getattr(est, "numeric_mode", None))
    # glue: the warm state and the init arrays as byte copies (`_aux`); the
    # binding expands tied, diag and spherical precisions_init to d x d per
    # component (`x_cluster/entries.mojo::bgmm_entry`, ip[15] = 2, 3, 4)
    parts = []
    prev = getattr(est, "_ext", None)
    warm = bool(est.warm_start) and prev is not None and prev["key"] == (k, d, est.covariance_type)
    if warm:
        parts.append(prev["state"])
    flags = [0, 0, 0]
    if est.weights_init is not None and not warm:
        w = _buffer.as_f32_c(est.weights_init, ndim=None, name="weights_init")[0]
        if w.size != k:
            raise ValueError(f"The parameter 'weights' should have the shape of ({k},)")
        parts.append(w)
        flags[0] = 1
    if est.means_init is not None and not warm:
        m = _f32(est.means_init, "means_init")
        if m.shape != (k, d):
            raise ValueError(f"The parameter 'means' should have the shape of ({k}, {d})")
        parts.append(m)
        flags[1] = 1
    if est.precisions_init is not None and not warm:
        pi = _buffer.as_f32_c(est.precisions_init, ndim=None, name="precisions_init")[0]
        want = (k * d * d, d * d, k * d, k)[ct]
        if pi.size != want:
            raise ValueError(f"The parameter '{est.covariance_type} precision' has {pi.size} values, "
                             f"expected {want}")
        # the base binding's native scan (`_buffer.all_finite`)
        if not _buffer.all_finite(pi):
            raise ValueError("precisions_init must be finite")
        parts.append(pi)
        flags[2] = ct + 1
    a = _aux(*parts)
    fp = [-1.0, -1.0, -1.0, float(est.reg_covar), float(est.tol),
          float(getattr(est, "lower_bound_", 0.0)) if warm else 0.0]
    ip = [n, d, k, 0, int(est.max_iter), int(est.n_init), _GMM_INITS[est.init_params],
          _seed(est.random_state), 0, 0, ct, 1 if warm else 0, 1] + flags
    f, i, s = call._call(_E_BGMM, x, a, ip, fp)
    est.n_features_in_ = d
    est.weights_ = Array._from_flat(f[0], (k,), "<f4")
    est.means_ = Array._from_flat(f[1], (k, d), "<f4")
    est.covariances_ = BayesianGaussianMixture._by_type(f[13], k, d, ct)
    est.precisions_cholesky_ = BayesianGaussianMixture._by_type(f[14], k, d, ct)
    est.log_det_chol_ = Array._from_flat(f[12], (k,), "<f4")   # x_cluster bgmm_entry
    est.n_iter_ = int(s[1])
    est.converged_ = bool(s[2])
    est.lower_bound_ = float(s[0])
    est._ext = {"key": (k, d, est.covariance_type), "call": call,
                "pchol": Array._from_flat(f[3], (k, d, d), "<f4"), "consts": Array._from_flat(f[8], (k,), "<f4"),
                "state": (list(f[11]) + list(f[4]) + [0.0] * k + list(f[6]) + list(f[1]) + list(f[7])
                          + list(f[2]) + list(f[3]))}
    return est


def _gmm_ext_score(est, X):
    """(log_resp list, score_samples, predict_proba, predict) of an ext fit."""
    ext = est._ext
    x = _f32(X)
    n, d = x.shape
    if d != est.n_features_in_:
        raise ValueError(f"mojolearn GaussianMixture: X has {d} features, the fit had {est.n_features_in_}")
    k = est.means_.shape[0]
    # predict_proba's exp and score's fold in the binding (flags 1 | 2);
    # the fold rides as a fifth item for `_gmm_ext_bic_aic`
    f, i, sc = ext["call"]._call(_E_BGMM_SCORE, x, _aux(est.means_, ext["pchol"], ext["consts"]), [n, d, k, 3])
    return (f[0], Array._from_flat(f[1], (n,), "<f4"), Array._from_flat(f[2], (n, k), "<f4"),
            Array._from_flat(i[0], (n,), "<i4"), sc[0])


def _gmm_ext_bic_aic(est, X):
    """(score, bic, aic): sklearn `_n_parameters`, `bic`, `aic` on the ext fit;
    the mean log-likelihood an ascending float64 fold of score_samples (the
    binding's float-float fold when it computes it)."""
    got = _gmm_ext_score(est, X)
    n = len(got[1])
    score = got[4] / n
    k, d = est.means_.shape
    cov = {"full": k * d * (d + 1) / 2.0, "diag": k * d, "tied": d * (d + 1) / 2.0, "spherical": k}[est.covariance_type]
    p = int(cov + k * d + k - 1)
    return score, -2 * score * n + p * _pm.log(n), -2 * score * n + 2 * p
