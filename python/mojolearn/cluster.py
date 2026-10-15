# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""k-means on the GPU. Reference: cuVS."""

from . import _mojolearn, _serialize
from ._array import Array
from ._buffer import addr, addr_ro, all_finite, as_f32_c, empty, zeros
from ._mode import NumericModeMixin

#: The saved k-means model (lane/kmeans-save, 2026-09-16), read by
#: `mojolearn.host_model` through `_classical_host.HostKMeans`. One format
#: for every k-means lane: the metric and the start are members of the file,
#: not tags of their own, because a second tag is a second loader and this
#: repo has already paid for a second lane set.
_KMEANS_FORMAT = "mojolearn-kmeans-1"

INIT_KMEANS_PLUS_PLUS = 0
INIT_RANDOM = 1
INIT_ARRAY = 2

METRIC_L2_EXPANDED = 0
METRIC_L2_SQRT_EXPANDED = 1

_INIT_NAMES = {
    "k-means++": INIT_KMEANS_PLUS_PLUS,
    "random": INIT_RANDOM,
    "array": INIT_ARRAY,
}

#: init code -> the name `save` writes for it, so a model built with the
#: integer code round-trips to the spelling the class documents.
_INIT_SAVED = {
    INIT_KMEANS_PLUS_PLUS: "k-means++",
    INIT_RANDOM: "random",
    INIT_ARRAY: "array",
}

#: cuVS's `DistanceType` members the kernel knows, by cuVS's own names
#: (lowercased) and by the scikit-learn spellings a caller expects.
#: `L2Expanded` is squared Euclidean, cuVS's default and the only metric
#: every recorded cell was measured at.
#:
#: `CosineExpanded` WAS routed here and refused by name on the Mojo host. It
#: was deleted on 2026-09-18 (lane/kmeans-cosine-capability) rather than
#: implemented, for two measured reasons: cuVS refuses it too (their
#: `pairwise_distance_kmeans` ends in `RAFT_FAIL`, `kmeans_common.cuh:320`,
#: so the refusal was never a gap against the reference), and the arithmetic
#: mean does not minimize cosine distance, so a cosine fit built on this
#: update step does not descend its own objective. A name absent from this
#: table is still refused BY NAME by `_metric_code` below, which is the
#: property the routing existed to provide.
_METRIC_NAMES = {
    "euclidean": METRIC_L2_EXPANDED,
    "l2_expanded": METRIC_L2_EXPANDED,
    "l2_sqrt_expanded": METRIC_L2_SQRT_EXPANDED,
}

#: metric code -> the name `save` writes for it. `euclidean` and
#: `l2_expanded` are the same code and the same arithmetic; the first is what
#: the class documents as the default, so it is the spelling that travels.
_METRIC_SAVED = {
    METRIC_L2_EXPANDED: "euclidean",
    METRIC_L2_SQRT_EXPANDED: "l2_sqrt_expanded",
}


def _refuse_non_finite(a, name, where):
    """A NaN or an infinity in `a` is refused BY NAME before any launch.

    Before this check a NaN row reached the kernel: `fit` returned a NaN
    centroid, `inertia_` of FLT_MAX and label -1 for the row, and `predict`
    returned -1 for a NaN query, all silently (scikit-learn's
    `validate_data` refuses the same input)."""
    if not all_finite(a):
        raise ValueError(
            f"mojolearn KMeans.{where}: {name} contains a NaN or an infinity; "
            "a non-finite row has no distance to any center, so it is "
            "refused by name"
        )


# Maintainer notes (moved from the docstring):
# - Reference: cuVS `kmeans::fit_predict`. `n_init` restarts share one seeded
#   host RNG; the best post-loop inertia wins (`tools/e2u_matrix_fit.py`
#   measures that `n_init=3` moves the answer on its fixture).
# - metric: 'euclidean'/'l2_expanded' are cuVS `L2Expanded`, the default every
#   recorded cell was measured at; 'l2_sqrt_expanded' is `metric_is_sqrt`.
#   'cosine' was routed and refused here until 2026-09-18 and was then removed;
#   cuVS refuses it too (`kmeans_common.cuh:320`).
# - oversampling_factor: `0.0` selects sequential k-means++ seeding, positive
#   selects k-means|| (`detail/kmeans.cuh:910-915`).
# - inertia_: formed by the one fresh assignment cuVS runs after the loop
#   (`detail/kmeans.cuh:516-535`). An older docstring said "0.0 MEANS NEVER
#   COMPUTED", which was false (corrected 2026-08-23; a 256 x 4 fit reports
#   55.44). cuVS's `inertia_check=False` only turns off the in-loop cost; the
#   only convergence criterion is the centroid shift.
# - labels_: the assignment against the final centroids, not the last
#   iteration's; cuVS and scikit-learn both run the extra pass.
# - DEVIATION 2369: attributes are `_array.Array`s (NumPy-free) where they were
#   ndarrays.
class KMeans(NumericModeMixin):
    """K-means clustering.

    Partitions the rows of `X` into `n_clusters` groups by minimizing the
    within-cluster sum of squared distances (Lloyd iterations after
    k-means++ or random seeding). `X` is read as float32, C order; a NaN or
    an infinity in `X`, `sample_weight` or `init_centroids` raises
    `ValueError`.

    The defaults follow cuVS, not scikit-learn, and one of them changes
    results: `n_init` is 1 here and 10 in scikit-learn. To compare with
    scikit-learn, set `n_init` to the same value on both.

    Parameters
    ----------
    n_clusters : int, default 8
        Number of clusters. Must not exceed the number of rows.
    init : {'k-means++', 'random', 'array'}, array or callable, default 'k-means++'
        How the starting centers are chosen. 'array' takes them from
        `init_centroids`. An array of shape `(n_clusters, n_features)` is used
        directly as the starting centers (`init_centroids` is then ignored).
        A callable is called as `init(X, n_clusters, random_state)` and must
        return such an array.
    n_init : int, default 1
        Number of restarts. Each restart draws different starting centers;
        the one with the lowest final inertia is kept.
    max_iter : int, default 300
        Maximum number of iterations per restart.
    tol : float, default 1e-4
        Stop when the centers move less than this between iterations.
    random_state : int, default 0
        Seed for the starting centers.
    init_centroids : array of shape (n_clusters, n_features), default None
        Starting centers for `init='array'`. Required in that case.
    metric : {'euclidean', 'l2_expanded', 'l2_sqrt_expanded'}, default 'euclidean'
        'euclidean' and 'l2_expanded' are the same: squared Euclidean
        distance, so `inertia_` and `transform` use squared distances.
        'l2_sqrt_expanded' uses the Euclidean distance itself, which changes
        `inertia_` and `transform` but not the labels. Any other value raises
        `ValueError`.
    oversampling_factor : float, default 2.0
        Selects the k-means++ seeding algorithm: `0.0` runs classic
        sequential k-means++, any positive value runs scalable k-means||
        with this oversampling. Negative values raise `ValueError`. Only used
        with `init='k-means++'`.

    Attributes
    ----------
    cluster_centers_ : Array of shape (n_clusters, n_features), float32
        The fitted centers.
    labels_ : Array of shape (n_samples,), int32
        The cluster of each training row, assigned against the final centers.
    inertia_ : float
        Weighted sum of distances (under `metric`) from each row to its final
        center. With `n_init > 1`, the kept restart's value.
    n_iter_ : int
        Number of iterations run.
    sum_scale_, weight_scale_ : float
        The fixed-point accumulator scales chosen for your data. Reproducing a
        result exactly needs them.
    n_features_in_ : int
        Number of features seen during `fit`.

    Notes
    -----
    Attributes are mojolearn `Array`s; `np.asarray(model.labels_)` gives a
    zero-copy NumPy view.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> model = mojolearn.KMeans(n_clusters=4, n_init=3).fit(X)
    >>> labels = model.predict(X)
    >>> distances = model.transform(X)
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _estimator_type = "clusterer"

    #: This family's binding, for `NumericModeMixin._bind`.
    _BINDING = "_mojolearn"

    def __init__(
        self,
        n_clusters=8,
        init="k-means++",
        n_init=1,
        max_iter=300,
        tol=1e-4,
        random_state=0,
        init_centroids=None,
        metric="euclidean",
        oversampling_factor=2.0,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init_centroids = init_centroids
        self.metric = metric
        self.oversampling_factor = oversampling_factor

    def _metric_code(self):
        if isinstance(self.metric, str):
            if self.metric not in _METRIC_NAMES:
                raise ValueError(
                    f"mojolearn: metric must be one of "
                    f"{sorted(_METRIC_NAMES)}, got {self.metric!r}"
                )
            return _METRIC_NAMES[self.metric]
        return int(self.metric)

    def _inference_inputs(self, X):
        """`X` and the centers as float32 C order, and the metric code, for
        `predict` and `transform`; refuses an unfitted model and a feature
        count other than the fit's."""
        centers = getattr(self, "cluster_centers_", None)
        if centers is None:
            raise RuntimeError("this estimator is not fitted yet")
        metric_code = self._metric_code()
        x, _ = as_f32_c(X, ndim=2, name="X")
        _refuse_non_finite(x, "X", "predict/transform")
        d = x.shape[1]
        dc = centers.shape[1]
        if d != dc:
            raise ValueError(
                f"mojolearn: X has {d} features, KMeans was fitted with {dc}"
            )
        c, _ = as_f32_c(centers, ndim=2, name="cluster_centers_")
        return x, c, metric_code

    # A callable `init` is called once on the host
    # (`_expansion_cluster._callable_init`) and its centers take the 'array'
    # path; an array `init` takes the 'array' path with `init_centroids` ignored.
    def fit(self, X, y=None, sample_weight=None):
        """Compute the k-means clustering of `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, read as float32.
        y : ignored
            Present for scikit-learn API compatibility.
        sample_weight : array-like of shape (n_samples,), default None
            Weight of each row. None weighs every row equally.

        Returns
        -------
        self : KMeans
            The fitted estimator.

        Raises
        ------
        ValueError
            For an unknown `init` or `metric`, `n_clusters` greater than the
            number of rows, `init='array'` without `init_centroids`, starting
            centers of the wrong shape, a `sample_weight` of the wrong length,
            or a NaN or infinity in any input.
        TypeError
            If `oversampling_factor` is not a number.
        """
        init_centroids = self.init_centroids
        if isinstance(self.init, str):
            if self.init not in _INIT_NAMES:
                raise ValueError(
                    f"mojolearn: init must be one of "
                    f"{sorted(_INIT_NAMES)}, got {self.init!r}"
                )
            init_code = _INIT_NAMES[self.init]
        elif callable(self.init):
            from ._expansion_cluster import _callable_init
            init_code = INIT_ARRAY
            init_centroids = _callable_init(self.init, X, self.n_clusters, self.random_state)
        elif isinstance(self.init, int) and not isinstance(self.init, bool):
            init_code = int(self.init)
        else:
            init_code = INIT_ARRAY
            init_centroids = self.init
        metric_code = self._metric_code()
        if isinstance(self.oversampling_factor, bool) or not isinstance(
            self.oversampling_factor, (int, float)
        ):
            raise TypeError(
                "mojolearn: oversampling_factor must be a float, got "
                f"{type(self.oversampling_factor).__name__}"
            )
        oversampling = float(self.oversampling_factor)

        x, _ = as_f32_c(X, ndim=2, name="X")
        _refuse_non_finite(x, "X", "fit")
        n, d = x.shape
        if self.n_clusters > n:
            raise ValueError(
                f"mojolearn: n_clusters={self.n_clusters} exceeds "
                f"n_samples={n}"
            )

        if init_code == INIT_ARRAY:
            if init_centroids is None:
                raise ValueError(
                    "mojolearn: init='array' needs init_centroids"
                )
            c0, _ = as_f32_c(init_centroids, ndim=2,
                             name="init_centroids")
            if c0.shape != (self.n_clusters, d):
                raise ValueError(
                    f"mojolearn: init_centroids must be "
                    f"({self.n_clusters}, {d}), got {c0.shape}"
                )
            # A COPY, on purpose: the kernel writes the centroids in place
            # and `c0` may be a zero-copy borrow of the caller's array.
            _refuse_non_finite(c0, "init_centroids", "fit")
            centers = c0.copy()
        else:
            centers = zeros((self.n_clusters, d), "<f4")
        # DEVIATION 2672: ALLOCATED AS THE ATTRIBUTE'S OWN DTYPE. The kernel
        # writes uint32 cluster ids and every id is below 2**31, so int32 is
        # the same bytes; allocating int32 here lets the kernel write the
        # array the caller keeps. It used to be `<u4` and `labels_` was then
        # `frombytes(labels.tobytes(), "<i4", (n,))`, two host copies of the
        # label vector per fit (16 MB each at 4,000,000 rows).
        labels = empty((n,), "<i4")

        if sample_weight is None:
            n_weights = 0
            # Never read when n_weights is 0. Passing X's address avoids
            # allocating an array of ones the Mojo side would ignore.
            w = x
        else:
            # DEVIATION 2369: a 1-D vector by contract (`ndim=1`); the
            # NumPy-era `.ravel()` also accepted an (n, 1) column.
            w, _ = as_f32_c(sample_weight, ndim=1, name="sample_weight")
            _refuse_non_finite(w, "sample_weight", "fit")
            if w.shape[0] != n:
                raise ValueError(
                    f"mojolearn: sample_weight has {w.shape[0]} entries, X "
                    f"has {n} rows"
                )
            n_weights = n

        inertia, n_iter, sum_scale, weight_scale = self._bind("_mojolearn").kmeans_fit(
            addr_ro(x, name="X"),
            addr(centers, name="cluster_centers_"),
            addr(labels, name="labels_"),
            addr_ro(w, name="sample_weight"),
            # ORDER MATCHES bindings/_mojolearn.mojo::kmeans_fit_binding.
            # n_samples, n_features, n_clusters, n_weights, max_iter,
            # tol, seed, n_init, init, metric, oversampling_factor
            [
                n, d, self.n_clusters, n_weights, self.max_iter,
                float(self.tol), int(self.random_state), self.n_init,
                init_code, metric_code, oversampling,
            ],
        )

        self.cluster_centers_ = centers
        # The kernel writes uint32 cluster ids; scikit-learn's attribute is
        # signed. Every id is below 2**31, so int32 is the SAME BYTES, and
        # since DEVIATION 2672 the array the kernel wrote IS that int32
        # array (it was `<u4` plus `frombytes(labels.tobytes(), "<i4")`,
        # two host copies per fit; before that a `labels.astype(np.int32)`
        # Python loop, DEVIATION 2369).
        self.labels_ = labels
        self.inertia_ = inertia
        self.n_iter_ = n_iter
        self.sum_scale_ = sum_scale
        self.weight_scale_ = weight_scale
        self.n_features_in_ = d
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        """Fit the model and return `labels_`, the cluster of each row of `X`.

        Returns
        -------
        labels : Array of shape (n_samples,), int32
        """
        return self.fit(X, sample_weight=sample_weight).labels_

    # Mirrors cuML's `KMeans.predict` (`kmeans.pyx:1071-1082`): the fit's own
    # final assignment (`cluster/estimator.mojo::kmeans_predict`, host
    # `kmeans_oracle.mojo::host_kmeans_predict`). CPU `predict` is served by the
    # core host binding on a CPU-only install.
    def predict(self, X):
        """Return the index of the nearest fitted center for each row of `X`.

        Uses the same assignment as `fit`, so `predict` on the training rows
        returns `labels_` exactly. A tie goes to the lowest center index. On a
        CPU-only install, `predict` runs on the CPU.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Read as float32. Must have the fitted number of features and no
            NaN or infinity.

        Returns
        -------
        labels : Array of shape (n_samples,), int32

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        ValueError
            For a feature count other than the fit's, or a non-finite value.
        """
        x, c, metric_code = self._inference_inputs(X)
        n, d = x.shape
        k = c.shape[0]
        labels = empty((n,), "<i4")
        self._bind("_mojolearn").kmeans_predict(
            addr_ro(x, name="X"),
            addr_ro(c, name="cluster_centers_"),
            addr(labels, name="labels"),
            # ORDER MATCHES bindings/_mojolearn.mojo::kmeans_predict_binding.
            [n, d, k, metric_code],
        )
        return labels

    # Reference: cuML `KMeans.transform` (`kmeans.pyx:1084`), cuVS
    # `kmeans_transform` (`detail/kmeans.cuh:1178-1219`). Each cell is the fused
    # assignment kernel's cell (`cluster/impl/detail/kmeans_transform.mojo`,
    # host `kmeans_oracle.mojo::host_kmeans_transform`).
    def transform(self, X):
        """Return the distance from each row of `X` to each fitted center.

        Distances use the model's `metric`: with the default 'euclidean' they
        are squared distances, with 'l2_sqrt_expanded' they are Euclidean
        distances. scikit-learn's `transform` always returns Euclidean
        distances; fit with `metric='l2_sqrt_expanded'` to get the same
        meaning. `transform(X)[i, predict(X)[i]]` is exactly the minimum of
        row `i`. On a CPU-only install, `transform` runs on the CPU.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Read as float32.

        Returns
        -------
        distances : Array of shape (n_samples, n_clusters), float32

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        ValueError
            For a feature count other than the fit's, or a non-finite value.
        """
        x, c, metric_code = self._inference_inputs(X)
        n, d = x.shape
        k = c.shape[0]
        out = empty((n, k), "<f4")
        self._bind("_mojolearn").kmeans_transform(
            addr_ro(x, name="X"),
            addr_ro(c, name="cluster_centers_"),
            addr(out, name="distances"),
            # ORDER MATCHES bindings/_mojolearn.mojo::kmeans_transform_binding.
            [n, d, k, metric_code],
        )
        return out

    def fit_transform(self, X, y=None, sample_weight=None):
        """Fit the model, then return `transform(X)`.

        Returns
        -------
        distances : Array of shape (n_samples, n_clusters), float32
        """
        return self.fit(X, sample_weight=sample_weight).transform(X)

    def _saved_metric(self):
        """The metric NAME `save` writes, and its code. A metric the class
        does not serve is refused here by name, before a file exists."""
        code = self._metric_code()
        if code not in _METRIC_SAVED:
            raise ValueError(
                f"mojolearn KMeans.save: metric {self.metric!r} is code {code}, "
                f"which is not one of {sorted(_METRIC_SAVED)}"
            )
        name = self.metric if isinstance(self.metric, str) else _METRIC_SAVED[code]
        return name, code

    def _saved_init(self):
        """The init NAME `save` writes, and its code."""
        if isinstance(self.init, str):
            if self.init not in _INIT_NAMES:
                raise ValueError(
                    f"mojolearn KMeans.save: init must be one of "
                    f"{sorted(_INIT_NAMES)}, got {self.init!r}"
                )
            return self.init, _INIT_NAMES[self.init]
        if callable(self.init) or not isinstance(self.init, int) or isinstance(self.init, bool):
            # an array or a callable: its centers took the 'array' path
            return "array", INIT_ARRAY
        code = int(self.init)
        if code not in _INIT_SAVED:
            raise ValueError(
                f"mojolearn KMeans.save: init code {code} is not one of "
                f"{sorted(_INIT_SAVED)}"
            )
        return _INIT_SAVED[code], code

    # Format `_KMEANS_FORMAT` (lane/kmeans-save, 2026-09-16), written by
    # `_serialize.write_npz`. `host_model` predicts through
    # `_mojolearn_core_host`'s `kmeans_predict`/`kmeans_transform`
    # (`cluster/host/kmeans_oracle.mojo`).
    def save(self, path):
        """Save the fitted model to `path` as an `.npz` file.

        The file holds the centers, the training labels, the metric, the
        `init` method and the fitted scalars. Saving the same model twice
        gives the same file. A model fitted on a GPU can be loaded with
        `load`, or with `mojolearn.host_model(path)` to predict and transform
        on a machine with no GPU, giving the same answers as the GPU.

        `init_centroids` is not saved: it is an input to a fit, not part of the
        fitted model. A loaded `init='array'` model that is fitted again raises
        `ValueError` for the missing `init_centroids`.

        Parameters
        ----------
        path : str or path-like
            Destination file.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        ValueError
            If `metric` or `init` is not one the model supports.
        """
        centers = getattr(self, "cluster_centers_", None)
        if centers is None:
            raise RuntimeError("this estimator is not fitted yet")
        from .linear_model import _saved_mode
        metric_name, metric_code = self._saved_metric()
        init_name, init_code = self._saved_init()
        k, d = centers.shape
        labels = self.labels_
        return _serialize.write_npz(path, {
            "format": _KMEANS_FORMAT,
            "estimator": type(self).__name__,
            "numeric_mode": _saved_mode(self),
            "metric": metric_name,
            "init": init_name,
            "centers": centers,
            "labels": labels,
            "meta": Array.from_list(
                [k, d, int(labels.size), int(self.n_iter_), metric_code, init_code,
                 int(self.max_iter), int(self.n_init), int(self.random_state)], "<i8"),
            "reals": Array.from_list(
                [float(self.inertia_), float(self.sum_scale_), float(self.weight_scale_),
                 float(self.tol), float(self.oversampling_factor)], "<f8"),
        })

    @classmethod
    def load(cls, path):
        """Load a model written by `save`.

        Arrays keep their saved dtypes. The result supports `predict` and
        `transform`, and carries `labels_`, `inertia_`, `n_iter_`,
        `sum_scale_` and `weight_scale_` as the fit set them.

        Parameters
        ----------
        path : str or path-like
            File written by `KMeans.save`.

        Returns
        -------
        model : KMeans

        Raises
        ------
        ValueError
            If the file is of another format or estimator, is truncated or
            inconsistent (field counts, center shape, label count), or names a
            metric or `init` the class does not support.
        """
        from .linear_model import _check_saved_by, _restore_mode
        arrays = _serialize.read_npz(path, _KMEANS_FORMAT)
        _check_saved_by(arrays, path, cls)
        meta = _serialize.exact(arrays, "meta", "<i8")
        reals = _serialize.exact(arrays, "reals", "<f8")
        if meta.size != 9 or reals.size != 5:
            raise ValueError(
                f"mojolearn: {path!r} holds {meta.size} meta fields and {reals.size} "
                f"reals, 9 and 5 are needed"
            )
        k, d, n_fit, n_iter, metric_code, init_code, max_iter, n_init, seed = (
            int(meta[i]) for i in range(9)  # glue: unpacks the fixed meta vector
        )
        if k < 1 or d < 1:
            raise ValueError(f"mojolearn: {path!r} holds an empty model ({k} centers of {d} features)")
        metric = _serialize.scalar_str(arrays, "metric")
        init = _serialize.scalar_str(arrays, "init")
        if metric not in _METRIC_NAMES:
            raise ValueError(
                f"mojolearn: {path!r} metric {metric!r} is not one KMeans serves "
                f"({sorted(_METRIC_NAMES)})"
            )
        if _METRIC_NAMES[metric] != metric_code:
            raise ValueError(
                f"mojolearn: {path!r} names metric {metric!r} (code "
                f"{_METRIC_NAMES[metric]}) but its meta holds code {metric_code}"
            )
        if init not in _INIT_NAMES:
            raise ValueError(
                f"mojolearn: {path!r} init {init!r} is not one KMeans serves "
                f"({sorted(_INIT_NAMES)})"
            )
        if _INIT_NAMES[init] != init_code:
            raise ValueError(
                f"mojolearn: {path!r} names init {init!r} (code {_INIT_NAMES[init]}) "
                f"but its meta holds code {init_code}"
            )
        obj = cls(n_clusters=k, init=init, n_init=n_init, max_iter=max_iter,
                  tol=float(reals[3]), random_state=seed, metric=metric,
                  oversampling_factor=float(reals[4]))
        _restore_mode(obj, arrays)
        centers = _serialize.exact(arrays, "centers", "<f4")
        if centers.ndim != 2 or tuple(centers.shape) != (k, d):
            raise ValueError(
                f"mojolearn: {path!r} centers shape {tuple(centers.shape)} is not "
                f"({k}, {d}); the centroid count and the dimensionality disagree"
            )
        labels = _serialize.exact(arrays, "labels", "<i4")
        if labels.ndim != 1 or labels.size != n_fit:
            raise ValueError(
                f"mojolearn: {path!r} labels hold {labels.size} values, the fit had {n_fit} rows"
            )
        obj.cluster_centers_ = centers
        obj.labels_ = labels
        obj.n_features_in_ = d
        obj.n_iter_ = n_iter
        obj.inertia_ = float(reals[0])
        obj.sum_scale_ = float(reals[1])
        obj.weight_scale_ = float(reals[2])
        return obj


__all__ = ["KMeans"]


# scikit-learn keeps more estimators in this module than it defines; they resolve from
# the package at first use (`_sklearn_modules`), so `from mojolearn.cluster import ...`
# takes the same names as scikit-learn's module.
from ._sklearn_modules import LAYOUT as _SKLEARN_LAYOUT, lazy_exports as _lazy_exports  # noqa: E402
__getattr__ = _lazy_exports("cluster")
__all__ += _SKLEARN_LAYOUT["cluster"]
