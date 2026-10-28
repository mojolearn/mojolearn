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
from . import _native_family_bridge

# The compiled family owners, connected once at import: the centers'
# predict/transform/score, MeanShift and AffinityPropagation predict and
# BisectingKMeans' tree predict (bindings/cluster_native.mojo), and the fits,
# partial_fit, fit_predict and fit_transform of MiniBatchKMeans,
# BisectingKMeans, MeanShift, OPTICS and AffinityPropagation
# (bindings/cluster_fit_a_native.mojo) are one native call each, handed the
# binding selected by the shared native provider after admission.
_NATIVE, _NATIVE_CONTEXT = _native_family_bridge.connect()

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
    # scikit-learn's _check_sample_weight (shape, finite, not all zero, its
    # wording) before the kernels; read only, never written
    from ._sample_weight import check_sample_weight
    return check_sample_weight(sample_weight, n, allow_negative=True)


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
    _mojolearn_native_public = True
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


class _CentersMixin:
    """predict / transform / score against `cluster_centers_` (squared
    euclidean nearest, the lowest index on a tie)."""

    def predict(self, X):
        """Assign each row of `X` to its nearest cluster center.

        The nearest center is by squared euclidean distance; a tie goes to
        the lowest center index.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows to label, converted to float32. Must have the number of
            features seen in `fit` and contain no NaN or infinity.

        Returns
        -------
        labels : Array of shape (n_samples,), int32
            Index of the nearest center for each row.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        ValueError
            If `X` is empty, has the wrong number of features, or contains
            NaN or infinity.
        """
        return _NATIVE(self, "family_xc_predict", (None, X), _NATIVE_CONTEXT)

    def transform(self, X):
        """Distance from each row of `X` to every cluster center.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows to transform, converted to float32.

        Returns
        -------
        distances : Array of shape (n_samples, n_clusters), float32
            Euclidean distance from each row to each center.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        """
        return _NATIVE(self, "family_xc_transform", (None, X), _NATIVE_CONTEXT)

    def fit_predict(self, X, y=None, sample_weight=None):
        """Fit the model on `X` and return the training labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training rows, converted to float32.
        y : ignored
            Present for scikit-learn compatibility.
        sample_weight : ignored
            Accepted for scikit-learn compatibility but not passed to the
            fit. Call ``fit(X, sample_weight=...)`` and read `labels_` to
            fit with weights.

        Returns
        -------
        labels : Array of shape (n_samples,), int32
            The fitted `labels_`.
        """
        return _NATIVE(self, "family_cfa_fit_predict", (None, X, y), _NATIVE_CONTEXT)

    def fit_transform(self, X, y=None, sample_weight=None):
        """Fit the model on `X` and return the distances to the centers.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training rows, converted to float32.
        y : ignored
            Present for scikit-learn compatibility.
        sample_weight : ignored
            Accepted for scikit-learn compatibility but not passed to the
            fit.

        Returns
        -------
        distances : Array of shape (n_samples, n_clusters), float32
            Euclidean distance from each training row to each center.
        """
        return _NATIVE(self, "family_cfa_fit_transform", (None, X, y), _NATIVE_CONTEXT)

    def score(self, X, y=None, sample_weight=None):
        """Opposite of the inertia of `X` under the fitted centers.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows to score, converted to float32.
        y : ignored
            Present for scikit-learn compatibility.
        sample_weight : ignored
            Accepted for scikit-learn compatibility; every row counts once.

        Returns
        -------
        score : float
            Minus the sum of squared distances from each row to its nearest
            center (larger is better).

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        """
        return _NATIVE(self, "family_xc_score", (None, X), _NATIVE_CONTEXT)


# Reference: scikit-learn `cluster/_kmeans.py` (MiniBatchKMeans),
# `_k_means_minibatch.pyx::update_center_dense`. The batch assignment runs on
# the device; the batch draws, the center update in scikit-learn's order, the
# random reassignment of low-count centers and the early stop (exponentially
# weighted batch inertia, `max_no_improvement`, `tol`) run in the binding's
# fit loop. The batch
# order comes from `random_state` through the lane's splitmix64 stream, not
# NumPy's Mersenne Twister. A callable init is called once with every row and
# its centers take the array path (bindings/cluster_fit_a_native.mojo); see
# x_cluster/NOT_IMPLEMENTED.tsv.
class MiniBatchKMeans(_CentersMixin, _XCluster):
    """Mini-batch k-means clustering.

    Follows scikit-learn's ``MiniBatchKMeans``: each step assigns a random
    batch of rows to the nearest center and moves the centers, low-count
    centers are randomly reassigned, and the fit stops early on the
    smoothed batch inertia. Batches are drawn from this library's own seeded
    random stream rather than NumPy's, so results agree with scikit-learn's
    within a tolerance, not bit for bit. Input is converted to float32.

    Parameters
    ----------
    n_clusters : int, default 8
        Number of clusters. Must be between 1 and the number of samples.
    init : {'k-means++', 'random'}, array-like or callable, default 'k-means++'
        How the centers start. 'k-means++' is greedy k-means++, as in
        scikit-learn. 'random' picks `n_clusters` distinct rows of the init
        sample, drawn by weight. An array of shape (n_clusters, n_features)
        gives the starting centers. A callable is called once as
        ``init(X, n_clusters, random_state=rs)`` with every row of `X`
        (scikit-learn passes a subsample) and must return such an array;
        `rs` is a NumPy ``RandomState`` seeded by `random_state` when NumPy
        is installed, otherwise the int.
    max_iter : int, default 100
        Maximum number of passes over the data.
    batch_size : int, default 1024
        Rows per mini-batch.
    verbose : int, default 0
        Accepted for scikit-learn compatibility; nothing is printed.
    compute_labels : bool, default True
        Accepted for scikit-learn compatibility; `labels_` is always
        computed.
    random_state : int or None, default None
        Non-negative seed for the batch draws and initialization. None
        means 0, so repeated fits give the same result. A NumPy
        ``RandomState`` is not accepted.
    tol : float, default 0.0
        Early-stop tolerance on center movement, as in scikit-learn.
    max_no_improvement : int or None, default 10
        Stop after this many steps without improvement of the smoothed batch
        inertia. None disables this early stop.
    init_size : int or None, default None
        Number of rows used for initialization. None uses scikit-learn's
        rule (3 * batch_size, at least 3 * n_clusters, at most n_samples).
    n_init : int or 'auto', default 'auto'
        Number of initializations; the best one is kept. 'auto' means 3 for
        ``init='random'`` and 1 otherwise.
    reassignment_ratio : float, default 0.01
        Controls how often low-count centers are reassigned, as in
        scikit-learn.
    numeric_mode : str or None, default None
        Numeric mode for this estimator. None uses the process default (see
        :func:`mojolearn.set_numeric_mode`).

    Attributes
    ----------
    cluster_centers_ : Array of shape (n_clusters, n_features), float32
        Cluster centers.
    counts_ : Array of shape (n_clusters,), float32
        Weighted number of rows assigned to each center over the fit.
    labels_ : Array of shape (n_samples,), int32
        Cluster of each training row.
    inertia_ : float
        Sum of squared distances of the training rows to their nearest
        center.
    n_steps_ : int
        Number of mini-batch steps run.
    n_iter_ : int
        Number of passes over the data.
    n_features_in_ : int
        Number of features seen in `fit`.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> model = mojolearn.MiniBatchKMeans(n_clusters=4, random_state=0).fit(X)
    >>> labels = model.predict(X)
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _estimator_type = "clusterer"

    def __init__(self, n_clusters=8, *, init="k-means++", max_iter=100, batch_size=1024,
                 verbose=0, compute_labels=True, random_state=None, tol=0.0,
                 max_no_improvement=10, init_size=None, n_init="auto", reassignment_ratio=0.01, numeric_mode=None):
        return _NATIVE(self, "family_xc_initialize", (None, (n_clusters, init, max_iter, batch_size, verbose, compute_labels, random_state, tol, max_no_improvement, init_size, n_init, reassignment_ratio, numeric_mode,)), _NATIVE_CONTEXT)

    def fit(self, X, y=None, sample_weight=None):
        """Fit the centers on `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training rows, converted to float32. NaN or infinity raises
            ValueError.
        y : ignored
            Present for scikit-learn compatibility.
        sample_weight : array-like of shape (n_samples,) or None, default None
            Row weights. As in scikit-learn, they weight the k-means++
            potentials, the init scoring and the batch draws.

        Returns
        -------
        self : MiniBatchKMeans
            The fitted estimator.

        Raises
        ------
        ValueError
            If `n_clusters` is outside [1, n_samples], `init` is an unknown
            string or an array of the wrong shape, `n_init` is below 1, or
            `X` contains NaN or infinity.
        """
        return _NATIVE(self, "family_cfa_mbk_fit", (None, X, sample_weight, y), _NATIVE_CONTEXT)


    def partial_fit(self, X, y=None, sample_weight=None):
        """Run one mini-batch step on all of `X`.

        As in scikit-learn, the first call initializes the centers from
        `init_size` rows of `X` using `init`. Later calls continue from the
        current centers (including those of an earlier `fit`); the random
        stream and the reassignment counter carry over between calls.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows for this step, converted to float32.
        y : ignored
            Present for scikit-learn compatibility.
        sample_weight : array-like of shape (n_samples,) or None, default None
            Row weights.

        Returns
        -------
        self : MiniBatchKMeans
            The updated estimator. `labels_` and `inertia_` describe this
            batch.

        Raises
        ------
        ValueError
            If `X` has a different number of features than earlier calls,
            or on the first call if `n_clusters` exceeds the number of rows
            or `init` is invalid.
        """
        return _NATIVE(self, "family_cfa_mbk_partial_fit", (None, X, sample_weight, y), _NATIVE_CONTEXT)


# Reference: scikit-learn `cluster/_bisect_k_means.py`. The 2-means is this
# library's KMeans (cuVS's Lloyd, the kmeans identity lanes' pair).
class BisectingKMeans(_CentersMixin, _XCluster):
    """Bisecting k-means clustering.

    Starting from one cluster, the leaf with the highest score is split in
    two by 2-means, `n_clusters - 1` times. The 2-means step is this
    library's ``KMeans``, so `tol` uses its centroid-shift rule and the
    'random' and 'k-means++' starts are its own; results agree with
    scikit-learn's within a tolerance. Input is converted to float32.

    Parameters
    ----------
    n_clusters : int, default 8
        Number of clusters. Must be between 1 and the number of samples.
    init : {'random', 'k-means++'}, default 'random'
        Start of each 2-means split. Any other value raises
        NotImplementedError.
    n_init : int, default 1
        Number of 2-means restarts per split; the best is kept. Must be at
        least 1.
    random_state : int or None, default None
        Non-negative seed. None means 0, so repeated fits give the same
        result. A NumPy ``RandomState`` is not accepted.
    max_iter : int, default 300
        Maximum iterations of each 2-means.
    verbose : int, default 0
        Accepted for scikit-learn compatibility; nothing is printed.
    tol : float, default 1e-4
        Convergence tolerance of each 2-means, on the centroid shift.
    copy_x : bool, default True
        Accepted for scikit-learn compatibility; `X` is never modified.
    algorithm : {'lloyd', 'elkan'}, default 'lloyd'
        Accepted for scikit-learn compatibility; both run the same Lloyd
        iterations.
    bisecting_strategy : {'biggest_inertia', 'largest_cluster'}, default 'biggest_inertia'
        Which leaf to split next: the one with the largest inertia, or the
        one with the most rows.
    numeric_mode : str or None, default None
        Numeric mode for this estimator. None uses the process default (see
        :func:`mojolearn.set_numeric_mode`).

    Attributes
    ----------
    cluster_centers_ : Array of shape (n_clusters, n_features), float32
        Centers of the leaf clusters.
    labels_ : Array of shape (n_samples,), int32
        Cluster of each training row.
    inertia_ : float
        Sum of squared distances of the training rows to their cluster
        center.
    n_features_in_ : int
        Number of features seen in `fit`.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> model = mojolearn.BisectingKMeans(n_clusters=4, random_state=0).fit(X)
    >>> labels = model.predict(X)
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _estimator_type = "clusterer"

    def __init__(self, n_clusters=8, *, init="random", n_init=1, random_state=None, max_iter=300,
                 verbose=0, tol=1e-4, copy_x=True, algorithm="lloyd", bisecting_strategy="biggest_inertia", numeric_mode=None):
        return _NATIVE(self, "family_xc_initialize", (None, (n_clusters, init, n_init, random_state, max_iter, verbose, tol, copy_x, algorithm, bisecting_strategy, numeric_mode,)), _NATIVE_CONTEXT)

    def fit(self, X, y=None, sample_weight=None):
        """Build the bisection tree on `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training rows, converted to float32. NaN or infinity raises
            ValueError.
        y : ignored
            Present for scikit-learn compatibility.
        sample_weight : array-like of shape (n_samples,) or None, default None
            Row weights.

        Returns
        -------
        self : BisectingKMeans
            The fitted estimator.

        Raises
        ------
        NotImplementedError
            If `init` is not 'random' or 'k-means++'.
        ValueError
            If `bisecting_strategy` or `algorithm` is unknown, `n_clusters`
            is outside [1, n_samples], `n_init` is below 1, or `X` contains
            NaN or infinity.
        """
        return _NATIVE(self, "family_cfa_bisect_fit", (None, X, sample_weight, y), _NATIVE_CONTEXT)

    def predict(self, X):
        """Assign each row of `X` to a leaf cluster.

        As in scikit-learn, each row descends the bisection tree, going to
        the nearer child center at each split (the left child on a tie).
        This can differ from the nearest leaf center.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows to label, converted to float32.

        Returns
        -------
        labels : Array of shape (n_samples,), int32
            Leaf cluster of each row.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        """
        return _NATIVE(self, "family_xc_bisect_predict", (None, X), _NATIVE_CONTEXT)


# Reference: scikit-learn `cluster/_mean_shift.py`. Every seed's shift loop is
# one device thread over all rows in order; the default bandwidth is
# scikit-learn's `estimate_bandwidth` from the device's exact row order
# statistic.
class MeanShift(_XCluster):
    """Mean shift clustering with a flat kernel.

    Follows scikit-learn's ``MeanShift``: each seed is shifted to the mean
    of the rows within `bandwidth` until it stops moving, centers are merged
    by intensity (a center within the bandwidth of a stronger one is
    dropped), and each row is labeled with its nearest center (the lowest
    index on a tie). Input is converted to float32.

    Parameters
    ----------
    bandwidth : float or None, default None
        Radius of the flat kernel; must be positive. None estimates it as
        scikit-learn's ``estimate_bandwidth`` does (quantile 0.3 over all
        rows).
    seeds : array-like of shape (n_seeds, n_features) or None, default None
        Starting points. None uses every row, or the bins when
        `bin_seeding` is True.
    bin_seeding : bool, default False
        Seed from the rows binned on a grid of size `bandwidth` (rounding
        half to even), as in scikit-learn.
    min_bin_freq : int, default 1
        With `bin_seeding`, only bins with at least this many rows become
        seeds.
    cluster_all : bool, default True
        If True every row is labeled with its nearest center. If False,
        rows farther than `bandwidth` from every center get label -1.
    n_jobs : None
        Accepted for scikit-learn compatibility; ignored.
    max_iter : int, default 300
        Maximum shifts per seed.
    numeric_mode : str or None, default None
        Numeric mode for this estimator. None uses the process default (see
        :func:`mojolearn.set_numeric_mode`).

    Attributes
    ----------
    cluster_centers_ : Array of shape (n_clusters, n_features), float32
        Cluster centers.
    labels_ : Array of shape (n_samples,), int32
        Cluster of each training row.
    bandwidth_ : float
        Bandwidth used, given or estimated.
    n_iter_ : int
        Largest number of shifts over the seeds.
    n_features_in_ : int
        Number of features seen in `fit`.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> model = mojolearn.MeanShift(bin_seeding=True).fit(X)
    >>> labels = model.labels_
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _estimator_type = "clusterer"

    def __init__(self, *, bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1,
                 cluster_all=True, n_jobs=None, max_iter=300, numeric_mode=None):
        return _NATIVE(self, "family_xc_initialize", (None, (bandwidth, seeds, bin_seeding, min_bin_freq, cluster_all, n_jobs, max_iter, numeric_mode,)), _NATIVE_CONTEXT)

    def fit(self, X, y=None):
        """Find the cluster centers of `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training rows, converted to float32. NaN or infinity raises
            ValueError.
        y : ignored
            Present for scikit-learn compatibility.

        Returns
        -------
        self : MeanShift
            The fitted estimator.

        Raises
        ------
        ValueError
            If `bandwidth` is not positive, `seeds` has a different number
            of features than `X`, or `X` contains NaN or infinity.
        """
        return _NATIVE(self, "family_cfa_meanshift_fit", (None, X, y), _NATIVE_CONTEXT)

    def predict(self, X):
        """Assign each row of `X` to its nearest cluster center.

        A tie goes to the lowest center index.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows to label, converted to float32.

        Returns
        -------
        labels : Array of shape (n_samples,), int32
            Index of the nearest center for each row.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        """
        return _NATIVE(self, "family_xc_predict", (None, X), _NATIVE_CONTEXT)

    def fit_predict(self, X, y=None):
        """Fit the model on `X` and return the training labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training rows, converted to float32.
        y : ignored
            Present for scikit-learn compatibility.

        Returns
        -------
        labels : Array of shape (n_samples,), int32
            The fitted `labels_`.
        """
        return _NATIVE(self, "family_cfa_meanshift_fit_predict", (None, X, y), _NATIVE_CONTEXT)


# Reference: scikit-learn `cluster/_optics.py`. The n x n distances and the
# core distances are the device's; the ordering loop is the reference's
# sequential one with the lowest index on a reachability tie; the xi and
# dbscan extractions are the reference's.
class OPTICS(_XCluster):
    """OPTICS density-based clustering.

    Follows scikit-learn's ``OPTICS``: it orders the rows by reachability
    and extracts clusters with the 'xi' or 'dbscan' method. The full
    n_samples x n_samples distance matrix is computed, so memory grows with
    the square of the number of rows. Computation is float32 throughout,
    without scikit-learn's rounding of distances to float precision, so
    reachability agrees with scikit-learn's within a tolerance. Like
    scikit-learn's, it has no `predict`.

    Parameters
    ----------
    min_samples : int > 1 or float in (0, 1], default 5
        Neighborhood size for a core point, as a count or as a fraction of
        the number of samples (at least 2).
    max_eps : float, default inf
        Largest neighborhood radius considered.
    metric : str, default 'minkowski'
        One of 'minkowski', 'euclidean', 'l2', 'manhattan', 'cityblock',
        'l1', 'chebyshev', 'infinity', 'cosine' or 'precomputed'. Any other
        metric, including a callable, raises NotImplementedError. With
        'precomputed', `X` is a square distance matrix.
    p : float, default 2
        Power of the Minkowski metric; must be at least 1 (inf gives
        Chebyshev).
    metric_params : dict or None, default None
        Only the key 'p' is read; it overrides `p`.
    cluster_method : {'xi', 'dbscan'}, default 'xi'
        How clusters are extracted from the reachability ordering.
    eps : float or None, default None
        Radius for ``cluster_method='dbscan'``; None means `max_eps`. Must
        not exceed `max_eps`.
    xi : float in [0, 1], default 0.05
        Minimum steepness for ``cluster_method='xi'``.
    predecessor_correction : bool, default True
        Apply scikit-learn's predecessor correction in the xi extraction.
    min_cluster_size : int > 1, float in (0, 1] or None, default None
        Minimum cluster size for the xi method; None means `min_samples`.
    algorithm : str, default 'auto'
        Accepted for scikit-learn compatibility; ignored.
    leaf_size : int, default 30
        Accepted for scikit-learn compatibility; ignored.
    memory : None
        Accepted for scikit-learn compatibility; ignored.
    n_jobs : None
        Accepted for scikit-learn compatibility; ignored.
    numeric_mode : str or None, default None
        Numeric mode for this estimator. None uses the process default (see
        :func:`mojolearn.set_numeric_mode`).

    Attributes
    ----------
    labels_ : Array of shape (n_samples,), int32
        Cluster of each row; -1 marks noise.
    reachability_ : Array of shape (n_samples,), float32
        Reachability distance of each row, indexed by row.
    ordering_ : Array of shape (n_samples,), int32
        Cluster ordering of the rows.
    core_distances_ : Array of shape (n_samples,), float32
        Distance at which each row becomes a core point.
    predecessor_ : Array of shape (n_samples,), int32
        Row from which each row was reached; -1 for none.
    cluster_hierarchy_ : Array of shape (n_clusters, 2), int32
        Start and end positions (in `ordering_`) of each cluster. Set only
        with ``cluster_method='xi'``.
    n_features_in_ : int
        Number of features seen in `fit`.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((500, 4), dtype=np.float32)
    >>> labels = mojolearn.OPTICS(min_samples=10).fit_predict(X)
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _estimator_type = "clusterer"

    def __init__(self, *, min_samples=5, max_eps=float("inf"), metric="minkowski", p=2,
                 metric_params=None, cluster_method="xi", eps=None, xi=0.05,
                 predecessor_correction=True, min_cluster_size=None, algorithm="auto",
                 leaf_size=30, memory=None, n_jobs=None, numeric_mode=None):
        return _NATIVE(self, "family_xc_initialize", (None, (min_samples, max_eps, metric, p, metric_params, cluster_method, eps, xi, predecessor_correction, min_cluster_size, algorithm, leaf_size, memory, n_jobs, numeric_mode,)), _NATIVE_CONTEXT)

    def fit(self, X, y=None):
        """Compute the reachability ordering of `X` and extract clusters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training rows, or a square distance matrix with
            ``metric='precomputed'``. Converted to float32; NaN or infinity
            raises ValueError.
        y : ignored
            Present for scikit-learn compatibility.

        Returns
        -------
        self : OPTICS
            The fitted estimator.

        Raises
        ------
        NotImplementedError
            If `metric` is not supported.
        ValueError
            If `cluster_method`, `min_samples`, `min_cluster_size`, `eps`,
            `xi` or `p` is out of range, or a precomputed matrix is not
            square.
        """
        return _NATIVE(self, "family_cfa_optics_fit", (None, X, y), _NATIVE_CONTEXT)

    def fit_predict(self, X, y=None):
        """Fit the model on `X` and return the cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training rows, or a square distance matrix with
            ``metric='precomputed'``.
        y : ignored
            Present for scikit-learn compatibility.

        Returns
        -------
        labels : Array of shape (n_samples,), int32
            The fitted `labels_`; -1 marks noise.
        """
        return _NATIVE(self, "family_cfa_optics_fit_predict", (None, X, y), _NATIVE_CONTEXT)


# Reference: scikit-learn `cluster/_affinity_propagation.py`. The
# responsibility and availability updates run on the device over the resident
# n x n matrices; the convergence window, the exemplar refinement and the
# labels are the reference's. The tie noise is the reference's formula with
# normals from the lane's seeded stream, not NumPy's.
class AffinityPropagation(_XCluster):
    """Affinity propagation clustering.

    Follows scikit-learn's ``AffinityPropagation``: responsibility and
    availability messages are exchanged until the set of exemplars is
    stable for `convergence_iter` iterations. The similarity is minus the
    squared euclidean distance, or a precomputed matrix. The small noise
    that breaks ties is drawn from this library's own seeded random stream
    rather than NumPy's, so results agree with scikit-learn's within a
    tolerance. Memory grows with the square of the number of rows. Input is
    converted to float32.

    Parameters
    ----------
    damping : float in [0.5, 1), default 0.5
        Damping of the message updates.
    max_iter : int, default 200
        Maximum number of iterations; must be at least 1.
    convergence_iter : int, default 15
        Number of iterations with no change in the exemplars that stops the
        fit; must be at least 1.
    copy : bool, default True
        Accepted for scikit-learn compatibility; `X` is never modified.
    preference : float, array-like of shape (n_samples,) or None, default None
        Preference of each row to be an exemplar. None uses the median of
        the similarities, as in scikit-learn.
    affinity : {'euclidean', 'precomputed'}, default 'euclidean'
        With 'precomputed', `X` is a square similarity matrix.
    verbose : bool, default False
        Accepted for scikit-learn compatibility; nothing is printed.
    random_state : int or None, default None
        Non-negative seed of the tie-breaking noise. None means 0. A NumPy
        ``RandomState`` is not accepted.
    numeric_mode : str or None, default None
        Numeric mode for this estimator. None uses the process default (see
        :func:`mojolearn.set_numeric_mode`).

    Attributes
    ----------
    cluster_centers_indices_ : Array of shape (n_clusters,), int32
        Row indices of the exemplars.
    cluster_centers_ : Array of shape (n_clusters, n_features), float32
        The exemplar rows. Not set with ``affinity='precomputed'``.
    labels_ : Array of shape (n_samples,), int32
        Cluster of each training row.
    affinity_matrix_ : Array of shape (n_samples, n_samples), float32
        The similarity matrix used.
    n_iter_ : int
        Number of iterations run.
    n_features_in_ : int
        Number of features seen in `fit`.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((500, 4), dtype=np.float32)
    >>> model = mojolearn.AffinityPropagation(random_state=0).fit(X)
    >>> labels = model.predict(X)
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _estimator_type = "clusterer"

    def __init__(self, *, damping=0.5, max_iter=200, convergence_iter=15, copy=True,
                 preference=None, affinity="euclidean", verbose=False, random_state=None, numeric_mode=None):
        return _NATIVE(self, "family_xc_initialize", (None, (damping, max_iter, convergence_iter, copy, preference, affinity, verbose, random_state, numeric_mode,)), _NATIVE_CONTEXT)

    def fit(self, X, y=None):
        """Find the exemplars of `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training rows, or a square similarity matrix with
            ``affinity='precomputed'``. Converted to float32; NaN or
            infinity raises ValueError.
        y : ignored
            Present for scikit-learn compatibility.

        Returns
        -------
        self : AffinityPropagation
            The fitted estimator.

        Raises
        ------
        ValueError
            If `affinity`, `damping`, `max_iter`, `convergence_iter` or
            `preference` is invalid, or a precomputed matrix is not square.
        """
        return _NATIVE(self, "family_cfa_ap_fit", (None, X, y), _NATIVE_CONTEXT)

    def predict(self, X):
        """Assign each row of `X` to its nearest exemplar.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows to label, converted to float32.

        Returns
        -------
        labels : Array of shape (n_samples,), int32
            Cluster of each row; every row gets -1 when the fit found no
            exemplars.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        ValueError
            If ``affinity='precomputed'``.
        """
        return _NATIVE(self, "family_xc_ap_predict", (None, X), _NATIVE_CONTEXT)

    def fit_predict(self, X, y=None):
        """Fit the model on `X` and return the training labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training rows, or a square similarity matrix with
            ``affinity='precomputed'``.
        y : ignored
            Present for scikit-learn compatibility.

        Returns
        -------
        labels : Array of shape (n_samples,), int32
            The fitted `labels_`.
        """
        return _NATIVE(self, "family_cfa_ap_fit_predict", (None, X, y), _NATIVE_CONTEXT)


# Reference: scikit-learn `mixture/_bayesian_mixture.py` with `_base.py`'s EM
# loop. The n-sized work (the Mahalanobis squares, the E-step log-sum-exp, the
# M-step moments) runs on the device in float32; the Wishart and
# Dirichlet(-process) updates, the d x d Cholesky, digamma and log-gamma and
# the lower bound run on the device too, in float-float (about 48 bits;
# x_cluster/bgmm_device.mojo), as do the default priors. Every covariance type
# runs through full d x d factors (the zeros add nothing).
class BayesianGaussianMixture(_XCluster):
    """Variational Bayesian Gaussian mixture.

    Follows scikit-learn's ``BayesianGaussianMixture``: variational
    inference with a Dirichlet process or Dirichlet distribution prior on
    the weights, for every covariance type. The per-row work runs in
    float32 and the parameter updates and lower bound in higher precision.
    Initialization draws from this library's own seeded random stream, so
    results agree with scikit-learn's within a tolerance. Input is
    converted to float32.

    Parameters
    ----------
    n_components : int, default 1
        Maximum number of mixture components. Must not exceed the number of
        samples.
    covariance_type : {'full', 'tied', 'diag', 'spherical'}, default 'full'
        Form of the component covariances.
    tol : float, default 1e-3
        Convergence threshold on the lower bound gain.
    reg_covar : float, default 1e-6
        Non-negative value added to the covariance diagonals.
    max_iter : int, default 100
        Maximum EM iterations per initialization; must be at least 1.
    n_init : int, default 1
        Number of initializations; the one with the best lower bound is
        kept. Must be at least 1.
    init_params : {'kmeans', 'random', 'k-means++', 'random_from_data'}, default 'kmeans'
        How the responsibilities start. 'kmeans' uses this library's
        ``KMeans``.
    weight_concentration_prior_type : {'dirichlet_process', 'dirichlet_distribution'}, default 'dirichlet_process'
        Prior on the weights.
    weight_concentration_prior : float or None, default None
        Weight concentration prior; None means 1 / n_components.
    mean_precision_prior : float or None, default None
        Precision prior on the means; None means 1.
    mean_prior : array-like of shape (n_features,) or None, default None
        Prior on the means; None means the mean of `X`.
    degrees_of_freedom_prior : float or None, default None
        Degrees-of-freedom prior; None means n_features.
    covariance_prior : float, array-like or None, default None
        Covariance prior: shape (n_features, n_features) for 'full' and
        'tied', (n_features,) for 'diag', a float for 'spherical'. None
        uses the covariance of `X`, as in scikit-learn.
    random_state : int or None, default None
        Non-negative seed. None means 0, so repeated fits give the same
        result. A NumPy ``RandomState`` is not accepted.
    warm_start : bool, default False
        If True, a refit with the same `n_components`, number of features
        and `covariance_type` resumes from the previous fit's parameters
        (one run, the lower bound continuing).
    verbose : int, default 0
        Accepted for scikit-learn compatibility; nothing is printed.
    verbose_interval : int, default 10
        Accepted for scikit-learn compatibility; ignored.
    numeric_mode : str or None, default None
        Numeric mode for this estimator. None uses the process default (see
        :func:`mojolearn.set_numeric_mode`).

    Attributes
    ----------
    weights_ : Array of shape (n_components,), float32
        Component weights.
    means_ : Array of shape (n_components, n_features), float32
        Component means.
    covariances_ : Array, float32
        Component covariances, shaped by `covariance_type` as in
        scikit-learn.
    precisions_cholesky_ : Array, float32
        Cholesky factors of the precision matrices, shaped by
        `covariance_type`.
    weight_concentration_ : Array or tuple of two Arrays, float32
        Dirichlet parameters of the weights (a pair for the Dirichlet
        process).
    weight_concentration_prior_ : float
        Weight concentration prior used.
    mean_precision_ : Array of shape (n_components,), float32
        Precision of each mean.
    mean_precision_prior_ : float
        Mean precision prior used.
    mean_prior_ : Array of shape (n_features,), float32
        Prior on the means used.
    degrees_of_freedom_ : Array of shape (n_components,) or float
        Degrees of freedom of each component (a float for 'tied').
    degrees_of_freedom_prior_ : float
        Degrees-of-freedom prior used.
    covariance_prior_ : Array or float
        Covariance prior used, shaped by `covariance_type`.
    lower_bound_ : float
        Lower bound of the best fit.
    n_iter_ : int
        EM iterations of the best fit.
    converged_ : bool
        Whether the best fit converged.
    n_features_in_ : int
        Number of features seen in `fit`.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 4), dtype=np.float32)
    >>> model = mojolearn.BayesianGaussianMixture(n_components=5, random_state=0).fit(X)
    >>> proba = model.predict_proba(X)
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _estimator_type = "density_estimator"

    def __init__(self, *, n_components=1, covariance_type="full", tol=1e-3, reg_covar=1e-6,
                 max_iter=100, n_init=1, init_params="kmeans",
                 weight_concentration_prior_type="dirichlet_process", weight_concentration_prior=None,
                 mean_precision_prior=None, mean_prior=None, degrees_of_freedom_prior=None,
                 covariance_prior=None, random_state=None, warm_start=False, verbose=0,
                 verbose_interval=10, numeric_mode=None):
        return _NATIVE(self, "family_xc_initialize", (None, (n_components, covariance_type, tol, reg_covar, max_iter, n_init, init_params, weight_concentration_prior_type, weight_concentration_prior, mean_precision_prior, mean_prior, degrees_of_freedom_prior, covariance_prior, random_state, warm_start, verbose, verbose_interval, numeric_mode,)), _NATIVE_CONTEXT)

    def fit(self, X, y=None):
        """Fit the mixture on `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training rows, converted to float32. NaN or infinity raises
            ValueError.
        y : ignored
            Present for scikit-learn compatibility.

        Returns
        -------
        self : BayesianGaussianMixture
            The fitted estimator.

        Raises
        ------
        ValueError
            If `covariance_type`, `init_params` or
            `weight_concentration_prior_type` is unknown, `n_components`
            exceeds the number of samples, `n_init` or `max_iter` is below
            1, or a prior has the wrong shape.
        """
        return _NATIVE(self, "family_mix_bgmm_fit", (None, X, y), _NATIVE_CONTEXT)

    def fit_predict(self, X, y=None):
        """Fit the mixture on `X` and return the component of each row.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training rows, converted to float32.
        y : ignored
            Present for scikit-learn compatibility.

        Returns
        -------
        labels : Array of shape (n_samples,), int32
            Most likely component of each training row.
        """
        return _NATIVE(self, "family_mix_bgmm_fit_predict", (None, X, y), _NATIVE_CONTEXT)

    def predict(self, X):
        """Most likely component of each row of `X`.

        A tie goes to the lowest component index.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows to label, converted to float32.

        Returns
        -------
        labels : Array of shape (n_samples,), int32
            Component with the largest responsibility for each row.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        """
        return _NATIVE(self, "family_mix_bgmm_predict", (None, X), _NATIVE_CONTEXT)

    def predict_proba(self, X):
        """Responsibility of each component for each row of `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows to evaluate, converted to float32.

        Returns
        -------
        proba : Array of shape (n_samples, n_components), float32
            Posterior probability of each component.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        """
        return _NATIVE(self, "family_mix_bgmm_predict_proba", (None, X), _NATIVE_CONTEXT)

    def score_samples(self, X):
        """Log-likelihood of each row of `X` under the mixture.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows to evaluate, converted to float32.

        Returns
        -------
        log_prob : Array of shape (n_samples,), float32
            Weighted log probability of each row.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        """
        return _NATIVE(self, "family_mix_bgmm_score_samples", (None, X), _NATIVE_CONTEXT)

    def score(self, X, y=None):
        """Mean log-likelihood of the rows of `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows to evaluate, converted to float32.
        y : ignored
            Present for scikit-learn compatibility.

        Returns
        -------
        score : float
            Average of `score_samples` over the rows.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        """
        return _NATIVE(self, "family_mix_bgmm_score", (None, X), _NATIVE_CONTEXT)


# The fixed tables the native fits read (bindings/cluster_fit_a_native.mojo):
# the class whose centers fit_predict/fit_transform fit, and OPTICS' metric
# codes (x_cluster/optics.mojo; minkowski is resolved by its `p`).
_native_family_bridge.tables(
    family_cfa_minibatch=MiniBatchKMeans,
    family_cfa_optics_metrics={"euclidean": 0, "l2": 0, "manhattan": 1, "cityblock": 1, "l1": 1, "chebyshev": 2,
                               "infinity": 2, "cosine": 4, "precomputed": 5},
)

# Fixed binding connections belong to package setup, never a model operation.
_native_family_bridge.connect_bindings("_mojolearn_x_cluster")
from ._native_owner_lifetime import connect_owner_context as _connect_cluster_owner_context
_connect_cluster_owner_context(_NATIVE_CONTEXT)

# Import-time schema only; runtime reads each admitted parameter natively.
from inspect import signature as _cluster_signature
_native_family_bridge.tables(
    cluster_public_parameters={cls: tuple(name for name in _cluster_signature(cls.__init__).parameters if name != "self")
                               for cls in (MiniBatchKMeans, BisectingKMeans, MeanShift, OPTICS,
                                           AffinityPropagation, BayesianGaussianMixture)},
    cluster_public_array_parameters=("init", "seeds", "preference", "mean_prior", "covariance_prior", "metric_params"),
)
del _cluster_signature
