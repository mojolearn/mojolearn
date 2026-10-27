# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""k-means on the GPU. Reference: cuVS."""

from . import _mojolearn, _native_archive_bridge, _native_family_bridge
from ._mode import NumericModeMixin

#: The saved k-means model (lane/kmeans-save, 2026-09-16), read by
#: `mojolearn.host_model` through `_classical_host.HostKMeans`. One format
#: for every k-means lane: the metric and the start are members of the file,
#: not tags of their own, because a second tag is a second loader and this
#: repo has already paid for a second lane set.
_KMEANS_FORMAT = "mojolearn-kmeans-1"

# The compiled saved-model owner, connected once at import: `save` and
# `load` are one native call each (bindings/archive_schema_native.mojo
# writes the schema registered below the class; host_archive_native.mojo
# builds the loaded model).
_ARCHIVE, _ARCHIVE_CONTEXT = _native_archive_bridge.connect()

# The compiled family owner, connected once at import: `fit`, `predict`,
# `transform`, `fit_predict` and `fit_transform` are one native call each
# (bindings/kmeans_native.mojo), which validates the inputs with the
# refusals below, allocates the outputs, runs the binding's kmeans_fit /
# kmeans_predict / kmeans_transform and stores the fitted attributes. The
# shell hands it the binding `_bind` resolves for the model's numeric mode.
_NATIVE, _NATIVE_CONTEXT = _native_family_bridge.connect()

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


# `KMeans.fit`, `predict` and `transform` refuse a NaN or an infinity in X,
# sample_weight or init_centroids BY NAME before any launch
# (bindings/kmeans_native.mojo `_kmeans_f32`): before that check a NaN row
# reached the kernel and `fit` returned a NaN centroid, `inertia_` of
# FLT_MAX and label -1 for the row, all silently (scikit-learn's
# `validate_data` refuses the same input).


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

    # A callable `init` is called once with the fit's rows and its centers
    # take the 'array' path (bindings/kmeans_native.mojo, scikit-learn's
    # `init(X, n_clusters, random_state=...)`); an array `init` takes the
    # 'array' path with `init_centroids` ignored.
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
        return _NATIVE(self, "family_kmeans_fit", (None, X, sample_weight), _NATIVE_CONTEXT)

    def fit_predict(self, X, y=None, sample_weight=None):
        """Fit the model and return `labels_`, the cluster of each row of `X`.

        Returns
        -------
        labels : Array of shape (n_samples,), int32
        """
        return _NATIVE(self, "family_kmeans_fit_predict", (None, X, sample_weight), _NATIVE_CONTEXT)

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
        return _NATIVE(self, "family_kmeans_predict", (None, X), _NATIVE_CONTEXT)

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
        return _NATIVE(self, "family_kmeans_transform", (None, X), _NATIVE_CONTEXT)

    def fit_transform(self, X, y=None, sample_weight=None):
        """Fit the model, then return `transform(X)`.

        Returns
        -------
        distances : Array of shape (n_samples, n_clusters), float32
        """
        return _NATIVE(self, "family_kmeans_fit_transform", (None, X, sample_weight), _NATIVE_CONTEXT)

    # The fitted centroids live on the device after the first query, under
    # `_resident_holder` (bindings/kmeans_native.mojo `_kmeans_resident`).
    # A pickle or a deepcopy carries no device handle; the copy uploads its
    # own at its first query. No Python `__del__`: dropping an instance
    # releases the handle through the native finalizer connected below.
    def __getstate__(self):
        return _NATIVE(self, "family_kmeans_getstate", (), _NATIVE_CONTEXT)

    def __setstate__(self, state):
        _NATIVE(self, "family_kmeans_setstate", (state,), _NATIVE_CONTEXT)

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
        return _ARCHIVE(self, "archive_save", (path,), _ARCHIVE_CONTEXT)

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
        return _ARCHIVE(cls, "host_archive_load", (path,), _ARCHIVE_CONTEXT)


# The fixed tables the native fit reads: the init and metric names and
# their codes, in the refusal wording's sorted order.
_native_family_bridge.tables(family_kmeans_inits=dict(_INIT_NAMES),
                             family_kmeans_metrics=dict(_METRIC_NAMES))


# The saved model, written natively from this schema (bindings/
# archive_schema_native.mojo) with the bytes `_serialize.write_npz` wrote:
# the metric and init NAME and code come from the fixed tables above, and
# the refusals keep the wording and order of the former Python save.
_native_archive_bridge.register(KMeans, _KMEANS_FORMAT, (
    ("need_set", "cluster_centers_", "RuntimeError", "this estimator is not fitted yet"),
    ("choice", "metric", "@metric", dict(_METRIC_NAMES), dict(_METRIC_SAVED), None, "ValueError",
     f"mojolearn: metric must be one of {sorted(_METRIC_NAMES)}, got {{value}}",
     f"mojolearn KMeans.save: metric {{value}} is code {{code}}, which is not one of {sorted(_METRIC_SAVED)}"),
    ("choice", "init", "@init", dict(_INIT_NAMES), dict(_INIT_SAVED), ("array", INIT_ARRAY), "ValueError",
     f"mojolearn KMeans.save: init must be one of {sorted(_INIT_NAMES)}, got {{value}}",
     f"mojolearn KMeans.save: init code {{code}} is not one of {sorted(_INIT_SAVED)}"),
    ("need", "labels_", "AttributeError", "'{cls}' object has no attribute 'labels_'"),
    *_native_archive_bridge.head(_KMEANS_FORMAT),
    ("str", "metric", ("@metric", 0)),
    ("str", "init", ("@init", 0)),
    ("array", "centers", "cluster_centers_"),
    ("array", "labels", "labels_"),
    ("i8", "meta", (("dim", "cluster_centers_", 0), ("dim", "cluster_centers_", 1), ("size", "labels_"),
                    ("int", "n_iter_"), ("int", ("@metric", 1)), ("int", ("@init", 1)),
                    ("int", "max_iter"), ("int", "n_init"), ("int", "random_state"))),
    ("f8", "reals", (("float", "inertia_"), ("float", "sum_scale_"), ("float", "weight_scale_"),
                     ("float", "tol"), ("float", "oversampling_factor"))),
), host_archive_kmeans_metrics=dict(_METRIC_NAMES), host_archive_kmeans_starts=dict(_INIT_NAMES))


# Dropping a fitted instance releases its resident centroids through the
# minting binding's `kmeans_centers_release` (bindings/host_archive_lifetime.mojo
# "entry" plans); the handle lives in `_resident_holder["index"]`.
_native_archive_bridge.lifetime(KMeans, "kmeans_centers_release", "_resident_holder")


__all__ = ["KMeans"]


# scikit-learn keeps more estimators in this module than it defines; they resolve from
# the package at first use (`_sklearn_modules`), so `from mojolearn.cluster import ...`
# takes the same names as scikit-learn's module.
from ._sklearn_modules import LAYOUT as _SKLEARN_LAYOUT, lazy_exports as _lazy_exports  # noqa: E402
__getattr__ = _lazy_exports("cluster")
__all__ += _SKLEARN_LAYOUT["cluster"]

# Import connection only; each native public operation admits the live registry.
_native_family_bridge.connect_bindings("_mojolearn")
