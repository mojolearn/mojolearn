# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Maintainer notes (moved from the module docstring; workstream D, 2026-09-14):
# The Python door of `hdbscan/estimator.mojo::hdbscan_fit_host` through
# `bindings/_mojolearn_hdbscan.mojo`. Reference: cuML. The path follows cuML's
# `runner.h:152-234` (brute-force k-NN, Boruvka MST, single linkage, condensed
# tree, excess-of-mass or leaf selection); the lane README lists what is
# mirrored and what is refused.
# Refusals: metric other than euclidean (here and Mojo host);
# cluster_selection_epsilon < 0 or non-finite (here); min_samples < 1 or
# > n_rows, min_cluster_size < 2 or > n_rows, alpha <= 0 or non-finite,
# n_rows < 2, any NaN/inf (Mojo host, DEVIATION 1607). Past 46340 rows the MST
# is built without the dense graph (DEVIATION 1620).
# Held-out points (2026-09-15): prediction data per cuML
# `prediction_data.cu:92-239` (`hdbscan/impl/prediction_data.mojo`);
# approximate_predict per `hdbscan.pyx:1264`, `predict.cuh:220-262`.
# Soft clustering (2026-09-15): `hdbscan.pyx:1180`, `:1114`,
# `soft_clustering.cuh:385-627`; float32 with pinned seams on every column
# (DEVIATION 1616, `hdbscan/impl/detail/soft_clustering.mojo`), where cuML
# promotes four seams to float64. No speed claim.
"""HDBSCAN density-based clustering.

`HDBSCAN` clusters the training data. When it is fitted with
`prediction_data=True`, three functions in this module work on new or
training points under the fitted clustering, following cuML's
`cuml.cluster.hdbscan` functions of the same names:

- `approximate_predict` gives the cluster label and probability of new
  points;
- `membership_vector` gives the soft cluster membership of new points;
- `all_points_membership_vectors` gives the soft cluster membership of
  every training point.

Each raises `ValueError` for a clusterer fitted without
`prediction_data=True`, as cuML does. Soft membership is computed in
float32 throughout, where cuML uses float64 in some steps.
"""
import warnings

from . import _backend, _native_archive_bridge, _native_family_bridge
from ._buffer import addr, addr_ro, as_f32_c, empty
from ._mode import NumericModeMixin

_MODE_CODE = {"fast": 0, "identical": 1}

#: `hierarchy/impl/cluster/detail/connectivities.mojo::DISTANCE_L2_SQRT_EXPANDED`,
#: the one metric `fit_hdbscan` accepts (their RAFT_EXPECTS).
METRIC_L2_SQRT_EXPANDED = 1

#: `hdbscan/impl/detail/select.mojo`'s codes.
SELECTION_EOM = 0
SELECTION_LEAF = 1

_METRICS = {"euclidean": METRIC_L2_SQRT_EXPANDED, "l2": METRIC_L2_SQRT_EXPANDED}

#: The model file (the neighbors and density inference lane, 2026-09-15):
#: what `approximate_predict` reads and nothing else. `raw_data` `<f4`
#: (n, d) as fitted, `core_distances` `<f4`, `labels` `<i4`, the prediction
#: data's `parents` `<i4`, `lambdas` `<f4`, `deaths` `<f4`,
#: `selected_clusters` `<i4`, `index_into_children` `<i4`, `exemplar_idx` `<i4`
#: and `exemplar_label_offsets` `<i4` (the last three and `parents` are what
#: membership_vector and all_points_membership_vectors read), `ints` `<i8`
#: [n_features_in_, n_clusters_, n_outliers_, n_boruvka_rounds_,
#: n_condensed_clusters_, n_edges, min_cluster_size, min_samples (-1 for
#: None), max_cluster_size, allow_single_cluster, n_exemplars], `reals` `<f8`
#: [alpha, cluster_selection_epsilon],
#: `metric` and `cluster_selection_method` as text. A model fitted without
#: prediction_data=True has nothing to save and is refused by name.
_HDBSCAN_FORMAT = "mojolearn-hdbscan-2"
_HDBSCAN_NAMES = "mojolearn HDBSCAN.save: metric and cluster_selection_method must be names"

# The compiled saved-model owner, connected once at import: `save` and
# `load` are one native call each (bindings/archive_schema_native.mojo
# writes the schema registered below the class; host_archive_native.mojo
# builds the loaded model).
_ARCHIVE, _ARCHIVE_CONTEXT = _native_archive_bridge.connect()
_SELECTION = {"eom": SELECTION_EOM, "leaf": SELECTION_LEAF}

# The compiled family owner, connected once at import: `fit` and
# `fit_predict` are one native call each (bindings/cluster_fit_a_native.mojo:
# the parameter refusals, the binary's numeric-mode check, `hdbscan_fit` and,
# with prediction_data=True, `hdbscan_generate_prediction_data`), handed the
# binding `_bind` resolves.
_NATIVE, _NATIVE_CONTEXT = _native_family_bridge.connect()
_native_family_bridge.tables(family_cfa_hdbscan_metrics=dict(_METRICS), family_cfa_hdbscan_selection=dict(_SELECTION),
                             family_cfa_hdbscan_mode_codes=dict(_MODE_CODE))


# cluster_selection_epsilon: cuML's epsilon search (`select.cuh:301-363`), run
# on the host by one function both routes call (DEVIATION 5115).
# probabilities_: cuML `get_probabilities` (DEVIATION 5116).
# prediction_data: cuML's parameter (`hdbscan.pyx:568`).
class HDBSCAN(NumericModeMixin):
    """Hierarchical density-based clustering (HDBSCAN).

    Follows cuML's `cuml.cluster.HDBSCAN`: builds the mutual reachability
    graph from exact nearest neighbors, its minimum spanning tree and the
    condensed cluster tree, then selects clusters by excess of mass or by
    leaf. Points in no cluster are labeled `-1` (noise). `X` is read as
    float32; only the Euclidean metric is supported. A NaN or infinity in
    `X`, or fewer than 2 rows, raises an error.

    Parameters
    ----------
    min_cluster_size : int, default 5
        Smallest group of points that counts as a cluster. Must be between 2
        and the number of rows.
    min_samples : int or None, default None
        Number of neighbors used for the core distance. None means
        `min_cluster_size`, as in cuML and scikit-learn. Must be between 1
        and the number of rows.
    cluster_selection_epsilon : float, default 0.0
        A selected cluster born below this distance is merged into its first
        ancestor born above it. 0.0 disables merging. Must be finite and
        non-negative.
    max_cluster_size : int, default 0
        Largest cluster the excess-of-mass selection may return. 0 means no
        limit.
    metric : {'euclidean'}, default 'euclidean'
        Distance metric. Any other value raises `ValueError`.
    alpha : float, default 1.0
        Distance scaling in the robust single linkage. Must be positive and
        finite.
    cluster_selection_method : {'eom', 'leaf'}, default 'eom'
        'eom' selects by excess of mass; 'leaf' selects the leaves of the
        condensed tree.
    allow_single_cluster : bool, default False
        Allow the result to be a single cluster.
    prediction_data : bool, default False
        Keep the condensed tree and build prediction data during `fit`, so
        `approximate_predict`, `membership_vector`,
        `all_points_membership_vectors` and `save` can be used. The
        clustering itself is the same either way.

    Attributes
    ----------
    labels_ : Array of shape (n_samples,), int32
        Cluster of each training point, `0 .. n_clusters_ - 1`, or `-1` for
        noise. Numbered as cuML numbers them.
    probabilities_ : Array of shape (n_samples,), float32
        Strength of each point's membership in its cluster, 0 for noise.
    core_distances_ : Array of shape (n_samples,), float32
        Distance from each point to its `min_samples`-th nearest neighbor,
        not counting the point itself.
    n_clusters_ : int
        Number of clusters found.
    n_outliers_ : int
        Number of noise points.
    n_boruvka_rounds_ : int
        Number of Boruvka rounds the minimum spanning tree took.
    n_condensed_clusters_ : int
        Number of clusters in the condensed tree.
    n_features_in_ : int
        Number of features seen during `fit`.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> model = mojolearn.HDBSCAN(min_cluster_size=10, prediction_data=True).fit(X)
    >>> labels = model.labels_
    >>> new_labels, strengths = mojolearn.hdbscan.approximate_predict(model, X[:10])
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _estimator_type = "clusterer"

    _mojolearn_native_public = True
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
        prediction_data=False,
        *, numeric_mode=None,
    ):
        return _NATIVE(self, "family_hdbscan_public_initialize", (min_cluster_size, min_samples, cluster_selection_epsilon, max_cluster_size, metric, alpha, cluster_selection_method, allow_single_cluster, prediction_data, numeric_mode,), _NATIVE_CONTEXT)

    def _extension(self):
        return _NATIVE(self, "family_hdbscan_public_binding", (), _NATIVE_CONTEXT)

    def fit(self, X, y=None):
        """Cluster `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, read as float32.
        y : ignored
            Present for scikit-learn API compatibility.

        Returns
        -------
        self : HDBSCAN
            The fitted estimator.

        Raises
        ------
        TypeError
            If an integer parameter is not an int, `alpha` or
            `cluster_selection_epsilon` is not a number, or `prediction_data`
            is not a bool.
        ValueError
            For an unknown `metric` or `cluster_selection_method`, or a
            negative or non-finite `cluster_selection_epsilon`.
        """
        return _NATIVE(self, "family_hdbscan_public_fit", (X, y), _NATIVE_CONTEXT)

    def fit_predict(self, X, y=None):
        """Cluster `X` and return `labels_`.

        Returns
        -------
        labels : Array of shape (n_samples,), int32
        """
        return _NATIVE(self, "family_hdbscan_public_fit_predict", (X, y), _NATIVE_CONTEXT)

    def save(self, path):
        """Save the fitted model to `path` as an `.npz` file.

        The file holds what `approximate_predict`, `membership_vector` and
        `all_points_membership_vectors` read, including the training data.
        Load it with `load`, or with `mojolearn.host_model(path)` to predict
        new points on a machine with no GPU.

        Parameters
        ----------
        path : str or path-like
            Destination file.

        Raises
        ------
        ValueError
            If the model is not fitted, was fitted without
            `prediction_data=True`, or `metric` or
            `cluster_selection_method` is not given by name.
        """
        return _ARCHIVE(self, "archive_save", (path,), _ARCHIVE_CONTEXT)

    @classmethod
    def load(cls, path):
        """Load a model written by `save`.

        Arrays keep their saved dtypes. The result works with
        `approximate_predict`, `membership_vector` and
        `all_points_membership_vectors`.

        Parameters
        ----------
        path : str or path-like
            File written by `HDBSCAN.save`.

        Returns
        -------
        model : HDBSCAN

        Raises
        ------
        ValueError
            If the file is of another format or estimator, or is truncated or
            inconsistent.
        """
        return _ARCHIVE(cls, "host_archive_load", (path,), _ARCHIVE_CONTEXT)


_native_archive_bridge.register(HDBSCAN, _HDBSCAN_FORMAT, (
    ("need", "labels_", "ValueError",
     "mojolearn HDBSCAN.save: this HDBSCAN instance is not fitted yet; call fit first"),
    ("need_set", "_prediction_data", "ValueError",
     "mojolearn HDBSCAN.save: Prediction data not yet generated. Fit with "
     "HDBSCAN(prediction_data=True); cuML refuses the same call the same way "
     "(hdbscan.pyx:1104-1108)"),
    ("need_str", "metric", "ValueError", _HDBSCAN_NAMES),
    ("need_str", "cluster_selection_method", "ValueError", _HDBSCAN_NAMES),
    *_native_archive_bridge.head(_HDBSCAN_FORMAT),
    ("str", "metric", "metric"),
    ("str", "cluster_selection_method", "cluster_selection_method"),
    ("array", "raw_data", "_raw_data"),
    ("array", "core_distances", "core_distances_"),
    ("array", "labels", "labels_"),
    ("array", "parents", ("_prediction_data", "parents")),
    ("array", "lambdas", ("_prediction_data", "lambdas")),
    ("array", "deaths", ("_prediction_data", "deaths")),
    ("head", "selected_clusters", ("_prediction_data", "selected_clusters"), "n_clusters_"),
    ("array", "index_into_children", ("_prediction_data", "index_into_children")),
    ("array", "exemplar_idx", ("_prediction_data", "exemplar_idx")),
    ("array", "exemplar_label_offsets", ("_prediction_data", "exemplar_label_offsets")),
    ("i8", "ints", (
        ("int", "n_features_in_"), ("int", "n_clusters_"), ("int", "n_outliers_"),
        ("int", "n_boruvka_rounds_"), ("int", "n_condensed_clusters_"),
        ("int", ("_prediction_data", "n_edges")), ("int", "min_cluster_size"),
        ("none", "min_samples", -1), ("int", "max_cluster_size"),
        ("bool", "allow_single_cluster"), ("int", ("_prediction_data", "n_exemplars")),
    )),
    ("f8", "reals", (("float", "alpha"), ("float", "cluster_selection_epsilon"))),
))


def _check_clusterer(clusterer, where):
    """cuML `_check_clusterer` (`hdbscan.pyx:1098-1110`): a fitted HDBSCAN
    with prediction data, else a refusal by name."""
    if not isinstance(clusterer, HDBSCAN):
        raise TypeError(
            f"mojolearn {where}: clusterer must be a mojolearn HDBSCAN, got {type(clusterer).__name__}"
        )
    if not hasattr(clusterer, "labels_"):
        raise ValueError(f"mojolearn {where}: this HDBSCAN instance is not fitted yet; call fit first")
    pd = getattr(clusterer, "_prediction_data", None)
    if pd is None:
        raise ValueError(
            f"mojolearn {where}: Prediction data not yet generated. Fit with "
            "HDBSCAN(prediction_data=True); cuML refuses the same call the same way "
            "(hdbscan.pyx:1104-1108)"
        )
    return pd


# cuML `approximate_predict` (`hdbscan.pyx:1264`, `predict.cuh:220-262`).
def approximate_predict(clusterer, points_to_predict):
    """Predict the cluster of new points under a fitted clustering.

    Follows cuML's `approximate_predict`. The labels are those of the
    original clustering, not the labels a re-clustering with the new points
    would find (hence "approximate"). If the clusterer found no clusters,
    a warning is issued and every point is labeled noise.

    Parameters
    ----------
    clusterer : HDBSCAN
        A model fitted with `prediction_data=True`.
    points_to_predict : array-like of shape (n_samples, n_features)
        New points, read as float32, with the fitted number of features.

    Returns
    -------
    labels : Array (n_samples,) int32
        `-1` where the nearest mutual reachability neighbor is noise or the
        point falls outside its cluster's lambda range.
    probabilities : Array (n_samples,) float32
        Strength of each point's membership in its predicted cluster.

    Raises
    ------
    TypeError
        If `clusterer` is not a mojolearn `HDBSCAN`.
    ValueError
        If the clusterer is not fitted, was fitted without
        `prediction_data=True`, or the feature count differs from the fit's.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> model = mojolearn.HDBSCAN(prediction_data=True).fit(X)
    >>> labels, strengths = mojolearn.hdbscan.approximate_predict(model, X[:10])
    """
    return _NATIVE(clusterer, "family_hdbscan_public_approximate_predict", (points_to_predict,), _NATIVE_CONTEXT)


def _check_batch_size(batch_size, where):
    """cuML's `batch_size must be > 0` (`hdbscan.pyx:1147`, `:1230`)."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError(f"mojolearn {where}: batch_size must be an int, got {type(batch_size).__name__}")
    if batch_size <= 0:
        raise ValueError(f"mojolearn {where}: batch_size must be > 0, got {batch_size}")


def _soft_tree_addrs(pd):
    """parents, lambdas, deaths, selected_clusters, index_into_children,
    exemplar_idx, exemplar_label_offsets: the order both soft bindings read."""
    return [addr_ro(pd["parents"], name="tree parents"), addr_ro(pd["lambdas"], name="tree lambdas"),
            addr_ro(pd["deaths"], name="deaths"), addr_ro(pd["selected_clusters"], name="selected_clusters"),
            addr_ro(pd["index_into_children"], name="index_into_children"),
            addr_ro(pd["exemplar_idx"], name="exemplar_idx"),
            addr_ro(pd["exemplar_label_offsets"], name="exemplar_label_offsets")]


# cuML `membership_vector` (`hdbscan.pyx:1180`, `soft_clustering.cuh:501-627`).
# Float32 on every column (DEVIATION 1616,
# `hdbscan/impl/detail/soft_clustering.mojo`): cuML computes four seams in
# float64.
def membership_vector(clusterer, points_to_predict, batch_size=4096):
    """Soft cluster membership of new points.

    Follows cuML's `membership_vector`. Row `i` gives, for each selected cluster `j`, the probability that point
    `i` is a member of cluster `j`: the distance to the cluster's exemplars
    and the merge height in the condensed tree, combined, normalized, and
    scaled by the probability that the point is in some cluster, so a row
    sums to at most 1 and a point far from every cluster has a row near 0.

    Computed in float32 throughout, where cuML uses float64 in some steps.
    Where cuML's row overflows to NaN (duplicated points), this row is
    finite.

    Parameters
    ----------
    clusterer : HDBSCAN
        A model fitted with `prediction_data=True`.
    points_to_predict : array-like of shape (n_samples, n_features)
        New points, read as float32, with the fitted number of features.
    batch_size : int, default 4096
        Maximum number of points computed per call. Rows are independent, so
        it does not change the result. Must be positive.

    Returns
    -------
    membership_vectors : Array (n_samples, n_clusters_) float32

    Raises
    ------
    TypeError
        If `clusterer` is not a mojolearn `HDBSCAN` or `batch_size` is not an
        int.
    ValueError
        If the clusterer is not fitted, was fitted without
        `prediction_data=True`, `batch_size` is not positive, or the feature
        count differs from the fit's.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> model = mojolearn.HDBSCAN(prediction_data=True).fit(X)
    >>> soft = mojolearn.hdbscan.membership_vector(model, X[:10])
    """
    return _NATIVE(clusterer, "family_hdbscan_public_membership_vector", (points_to_predict, batch_size), _NATIVE_CONTEXT)


# cuML `all_points_membership_vectors` (`hdbscan.pyx:1114`,
# `soft_clustering.cuh:385-482`). Float32 on every column (DEVIATION 1616).
def all_points_membership_vectors(clusterer, batch_size=4096):
    """Soft cluster membership of every training point.

    Follows cuML's `all_points_membership_vectors`, using the fact that the
    training points are already in the condensed tree. Each row has the
    same meaning as a row of `membership_vector`. Computed in float32
    throughout.

    Parameters
    ----------
    clusterer : HDBSCAN
        A model fitted with `prediction_data=True`.
    batch_size : int, default 4096
        Maximum number of training rows computed per call. Rows are
        independent, so it does not change the result. Must be positive.

    Returns
    -------
    membership_vectors : Array (n_samples, n_clusters_) float32
        With no cluster the shape is (n_samples, 0). cuML returns a 1-D
        zero vector of length n_samples there instead.

    Raises
    ------
    TypeError
        If `clusterer` is not a mojolearn `HDBSCAN` or `batch_size` is not an
        int.
    ValueError
        If the clusterer is not fitted, was fitted without
        `prediction_data=True`, or `batch_size` is not positive.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> model = mojolearn.HDBSCAN(prediction_data=True).fit(X)
    >>> soft = mojolearn.hdbscan.all_points_membership_vectors(model)
    """
    return _NATIVE(clusterer, "family_hdbscan_public_all_points_membership_vectors", (batch_size,), _NATIVE_CONTEXT)


__all__ = ["HDBSCAN", "approximate_predict", "membership_vector", "all_points_membership_vectors"]

# Import-time setup only; runtime selection and batching belong to Mojo.
from ._native_owner_lifetime import connect_owner_context as _connect_hdbscan_owner_context
_connect_hdbscan_owner_context(_NATIVE_CONTEXT)
_native_family_bridge.connect_bindings("_mojolearn_hdbscan")
_native_family_bridge.tables(
    hdbscan_public_class=HDBSCAN,
    hdbscan_public_parameters=(
        "min_cluster_size", "min_samples", "cluster_selection_epsilon",
        "max_cluster_size", "metric", "alpha", "cluster_selection_method",
        "allow_single_cluster", "prediction_data", "numeric_mode",
    ),
)
