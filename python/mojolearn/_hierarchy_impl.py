# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Single-linkage agglomerative clustering on the GPU. Reference: cuML's
`AgglomerativeClustering` (`cuml/cpp/src/hierarchy/linkage.cu` down through
cuVS's `cluster/detail/*` to RAFT's Boruvka MST).

The implementation is `hierarchy/` (DEVIATIONS 620-624 and 881); `hierarchy/README.md`,
`hierarchy/NOT_IMPLEMENTED.tsv` are the record.

Line numbers cited here were read in `upstream/cuml-v26.08.00` (265b9da) and
`upstream/cuvs-v26.08.00` (6ba2ce2) on 2026-08-24. The lane elsewhere cites
cuVS at `94c2819` and RAFT at `661a3b8`, the UNTAGGED default-branch checkouts, whose line numbers for the same code differ by a few dozen lines.

This class is not re-exported from `mojolearn/__init__.py` by this file;
whoever owns that file decides the public namespace.
"""


from . import _backend, _expansion_cluster, _mojolearn_solver, _native_archive_bridge, _native_family_bridge
from ._mode import ParamsMixin, available_if

#: `AgglomerativeClustering.save`'s format tag
#: (lane/inference-transductive-predict, 2026-09-15).
_AGGLOMERATIVE_FORMAT = "mojolearn-agglomerative-1"

# The compiled saved-model owner, connected once at import: `save` and
# `load` are one native call each (bindings/archive_schema_native.mojo
# writes the schema registered below the class; host_archive_native.mojo
# builds the loaded model).
_ARCHIVE, _ARCHIVE_CONTEXT = _native_archive_bridge.connect()

# `cuml/common/distance_type.hpp`, the codes cuML's Python layer passes
# (`agglomerative.pyx:36-43`).
DISTANCE_L2_EXPANDED = 0
DISTANCE_L2_SQRT_EXPANDED = 1
DISTANCE_COSINE_EXPANDED = 2
DISTANCE_L1 = 3

#: `x_cluster/entries.mojo::ENTRY_AGGLO`.
_E_AGGLO = 11

# Only the two the implementation carries. cuML maps "euclidean" and "l2" to
# L2SqrtExpanded; nothing in their table maps to L2Expanded, so no name for
# it is invented here.
_METRICS = {
    "euclidean": DISTANCE_L2_SQRT_EXPANDED,
    "l2": DISTANCE_L2_SQRT_EXPANDED,
}

# The compiled family owner (bindings/cluster_native.mojo): predict is one
# native call, handed the binding `_bind` resolves; it reads this table.
_NATIVE, _NATIVE_CONTEXT = _native_family_bridge.connect()

#: scikit-learn's metric names on the x_cluster route (`x_cluster/agglo.mojo`):
#: -1 euclidean through the squared distance, 1 manhattan, 4 cosine
#: (`bodies.pdist_cell`, DEVIATION 5111), 5 precomputed.
_ALL_METRICS = {"euclidean": -1, "l2": -1, "l1": 1, "manhattan": 1, "cityblock": 1,
                "cosine": 4, "precomputed": 5}

#: `bodies.LINK_*`.
_LINKAGES = {"ward": 0, "complete": 1, "average": 2, "single": 3}

# `cuvs/cluster/agglomerative.hpp::Linkage`, as cuML's Python spells it.
_CONNECTIVITIES = {"pairwise": 0, "knn": 1}

_native_family_bridge.tables(family_agglo_predict_metrics=dict(_METRICS),
                             family_cfb_agglo_all_metrics=dict(_ALL_METRICS),
                             family_cfb_agglo_linkages=dict(_LINKAGES),
                             family_cfb_agglo_connectivities=dict(_CONNECTIVITIES))

# The dense connectivity matrix is `m * m` of their `int`, so it overflows
# past this (`hierarchy/impl/cluster/detail/connectivities.mojo`).
PAIRWISE_MAX_ROWS = 46340

# The import-time mode guard that stood here is deleted with the one it came
# from; see the note at the top of `_solver_impl.py`. `_backend.select()` is
# what refuses a FAST binary under the identical label, and it degrades one
# unbuilt binding to one broken estimator instead of an unimportable package.


# Maintainer notes (moved from the docstring, 2026-10-12).
#
# Reference: `cuml.cluster.AgglomerativeClustering` and
# `ML::linkage::single_linkage`; the Mojo entry is `hierarchy/impl/linkage.mojo`
# and the host surface is `hierarchy/estimator.mojo`. Two routes: 'single' with
# the euclidean metric, no connectivity matrix and no per-merge distances stays
# on cuML's Boruvka route (`_mojolearn_solver.linkage_fit`); everything else
# takes the x_cluster route (`x_cluster/agglo.mojo`, ENTRY_AGGLO).
#
# Defaults: connectivity None (scikit-learn's spelling of the full dense graph,
# normalized to 'pairwise'); cuML's PYTHON default is 'knn'
# (`agglomerative.pyx:123`), their C++ default is pairwise (`linkage.hpp:43-44`)
# and scikit-learn's dense tree is pairwise too. DEVIATION 881: the k-NN graph
# arm is rung 2 and NOT IMPLEMENTED, so 'knn' is refused by name rather than
# downgraded. linkage 'single' here and in cuML (`agglomerative.pyx:124`);
# scikit-learn's default is 'ward'.
#
# c: tunes `k = log(n) + c` for the k-NN graph and cuML forwards it only when
# `use_knn` is true (`linkage.cu:40`, `use_knn ? c : 0`); with 'knn' refused it
# can reach nothing, so any value but 15 raises. n_rows > 46340 on the Boruvka
# route: the dense connectivity matrix is `m * m` of their `int` and overflows.
#
# labels_ on the Boruvka route: roots are labeled by descending position in the
# children array (`extract_flattened_clusters`), cuVS's numbering, not
# scikit-learn's. children_ there is in Boruvka's orientation (`src` is the
# vertex that added the edge); scikit-learn's mergesort on `mst.data` keeps
# scipy's coo order and this sort is the total order
# `(weight_key, min(u,v), max(u,v))` (DEVIATIONS 620 and 621), so rows agree as
# unordered pairs only when the MST is tie-free. n_connected_components_ is the
# literal 1 there (`single_linkage.cuh:301`, the pairwise graph is complete).
# n_boruvka_rounds_ is the identity card's integer stage `linkage.mst.rounds`;
# -1 on a CPU-only install (the host binding runs Kruskal) and on the x_cluster
# route.
#
# IDENTITY: `MOJOLEARN_IDENTITY_TRACE` does nothing on this path. The
# `single_linkage` entry takes no trace object, and the certified eight-stage
# `linkage.*` card is `hierarchy/linkage_main.mojo`'s, which re-runs the
# distance and MST stages beside the fit to record them. This class calls the
# same entry that driver calls; it does not itself emit a card.
#
# predict is DEVIATION 2740 (neither cuML nor scikit-learn has one).
class AgglomerativeClustering(ParamsMixin):
    """Agglomerative hierarchical clustering.

    Builds a merge tree bottom-up and cuts it into `n_clusters` clusters or
    at `distance_threshold`. Supports scikit-learn's single, complete,
    average and Ward linkages, connectivity matrices and per-merge
    distances. Input is a dense array, converted to float32.

    One default differs from scikit-learn's and changes the result:
    `linkage` is 'single' (scikit-learn: 'ward'). cuML's
    ``AgglomerativeClustering`` also defaults to single linkage, but to
    'knn' connectivity, which is not available here; pass
    ``connectivity='pairwise'`` explicitly when moving cuML code.

    Label integers can differ from scikit-learn's for the same partition:
    compare partitions (for example with ``adjusted_rand_score``), not the
    label values. Likewise, `children_` rows can differ in order and
    orientation from scikit-learn's when merge distances tie.

    Parameters
    ----------
    n_clusters : int or None, default 2
        Number of clusters, between 1 and the number of samples. Must be
        None when `distance_threshold` is set.
    metric : str, default 'euclidean'
        One of 'euclidean', 'l2', 'l1', 'manhattan', 'cityblock', 'cosine'
        or 'precomputed' (`X` is then a square distance matrix). Any other
        value, including a callable, raises NotImplementedError. Ward
        linkage requires 'euclidean' or 'l2'; 'cosine' rejects rows that are
        all zeros.
    connectivity : None, 'pairwise', array-like, sparse matrix or callable, default None
        'pairwise' and None both mean no constraint (the full dense graph).
        A matrix of shape (n_samples, n_samples), or a callable that returns
        one from `X`, restricts merges to its nonzero entries, as in
        scikit-learn. 'knn' raises NotImplementedError.
    linkage : {'single', 'complete', 'average', 'ward'}, default 'single'
        Merge criterion.
    c : int, default 15
        Kept for cuML compatibility, where it tunes the 'knn' graph. Any
        other value raises NotImplementedError.
    memory : None
        Must be None; anything else raises NotImplementedError.
    compute_full_tree : 'auto' or bool, default 'auto'
        With a connectivity matrix, whether to build the full tree or stop
        at `n_clusters`, with scikit-learn's 'auto' rule. Without one the
        full tree is always built. Must be True or 'auto' when
        `distance_threshold` is set.
    distance_threshold : float or None, default None
        Merge distance at or above which clusters are no longer merged.
        Exactly one of `n_clusters` and `distance_threshold` must be set.
    compute_distances : bool, default False
        Store the merge distances in `distances_` even without
        `distance_threshold`.
    prediction_data : bool, default False
        Keep a copy of the training rows so that `predict` and `save` work.
        Only single linkage with the 'euclidean' or 'l2' metric can predict.

    Attributes
    ----------
    labels_ : Array of shape (n_samples,), int32
        Cluster of each training row.
    n_clusters_ : int
        Number of clusters found.
    children_ : Array of shape (n_merges, 2), int32
        The merges, one per row; values below n_samples are rows, larger
        values are earlier merges. n_merges is n_samples - 1 for a full
        tree.
    distances_ : Array of shape (n_merges,), float32
        Merge distances. Set only with `distance_threshold` or
        ``compute_distances=True``.
    n_leaves_ : int
        Number of leaves (n_samples).
    n_connected_components_ : int
        Connected components of the connectivity graph. A constrained fit
        with more than one warns and completes the graph, as scikit-learn
        does.
    n_boruvka_rounds_ : int
        Not a scikit-learn attribute: the number of Boruvka rounds of the
        single-linkage euclidean fit without a connectivity matrix; -1 on
        other fits and on a CPU-only install.
    n_features_in_ : int
        Number of features seen in `fit`.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> model = mojolearn.AgglomerativeClustering(n_clusters=4, prediction_data=True).fit(X)
    >>> labels = model.predict(X[:10])
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _estimator_type = "clusterer"

    #: The binding `predict` calls. The FIT is `_mojolearn_solver`'s, a
    #: training-only family; the prediction entry lives in the estimators
    #: family, whose host binding ships in the inference wheel.
    _BINDING = "_mojolearn_estimators"

    def __init__(self, n_clusters=2, *, metric="euclidean",
                 connectivity=None, linkage="single", c=15,
                 memory=None, compute_full_tree="auto",
                 distance_threshold=None, compute_distances=False,
                 prediction_data=False):
        # Stored verbatim, as scikit-learn requires; `_validate_params`
        # checks them when `fit` runs.
        self.n_clusters = n_clusters
        self.metric = metric
        self.connectivity = connectivity
        self.linkage = linkage
        self.c = c
        self.memory = memory
        self.compute_full_tree = compute_full_tree
        self.distance_threshold = distance_threshold
        self.compute_distances = compute_distances
        self.prediction_data = prediction_data

    def _bind(self, name=None):
        return _backend.binding(name or self._BINDING, getattr(self, "numeric_mode", None))

    def fit(self, X, y=None):
        """Build the merge tree of `X` and cut it into clusters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Dense training rows, or a square distance matrix with
            ``metric='precomputed'``. Converted to float32. Sparse input
            raises NotImplementedError.
        y : ignored
            Present for scikit-learn compatibility.

        Returns
        -------
        self : AgglomerativeClustering
            The fitted estimator.

        Raises
        ------
        TypeError
            If `prediction_data` is not a bool.
        NotImplementedError
            If `X` is sparse.
        ValueError
            If there are fewer than 2 rows, `n_clusters` exceeds the number
            of rows, `X` contains NaN or infinity, a precomputed or
            connectivity matrix has the wrong shape, or ``metric='cosine'``
            meets an all-zero row. With single linkage, the euclidean
            metric, no connectivity matrix and no merge distances requested,
            more than 46,340 rows also raises (the dense graph would
            overflow), except in ``fast`` mode on Apple with at most 64
            features.

        Warns
        -----
        UserWarning
            If a connectivity matrix has more than one connected component.
        """
        # The estimators binding (residency), the solver binding (the Boruvka
        # route), the x_cluster binding (every other linkage) and the
        # x_neighbors binding (the cosine zero-row test), then X
        # (bindings/cfb_agglo.mojo).
        return _NATIVE(self, "family_cfb_agglo_fit", (
            self._bind(self._BINDING), _mojolearn_solver,
            _backend.binding(_expansion_cluster._XCluster._BINDING, getattr(self, "numeric_mode", None)),
            _backend.binding("_mojolearn_x_neighbors", getattr(self, "numeric_mode", None)), X), _NATIVE_CONTEXT)

    def fit_predict(self, X, y=None):
        """Fit the model on `X` and return the training labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Dense training rows, or a square distance matrix with
            ``metric='precomputed'``.
        y : ignored
            Present for scikit-learn compatibility.

        Returns
        -------
        labels : Array of shape (n_samples,), int32
            The fitted `labels_`.
        """
        # The estimators binding (residency), the solver binding (the Boruvka
        # route), the x_cluster binding (every other linkage) and the
        # x_neighbors binding (the cosine zero-row test), then X
        # (bindings/cfb_agglo.mojo).
        return _NATIVE(self, "family_cfb_agglo_fit_predict", (
            self._bind(self._BINDING), _mojolearn_solver,
            _backend.binding(_expansion_cluster._XCluster._BINDING, getattr(self, "numeric_mode", None)),
            _backend.binding("_mojolearn_x_neighbors", getattr(self, "numeric_mode", None)), X), _NATIVE_CONTEXT)

    @available_if(lambda self: bool(self.prediction_data) or getattr(self, "_fit_X", None) is not None)
    def predict(self, X):
        """Label new rows with the cluster of their nearest training row.

        Neither scikit-learn nor cuML has this method. For single linkage,
        the closest cluster to a new row is the one containing its nearest
        training row (squared euclidean distance); ties go to the lowest
        cluster label, then the lowest training index. A training row
        predicts its own fitted label unless an exact duplicate of it was
        cut into a cluster with a lower label, which happens only when
        `n_clusters` exceeds the number of distinct rows.

        Only single linkage with the 'euclidean' or 'l2' metric can predict.
        The method exists only with ``prediction_data=True`` (or on a model
        loaded with `load`), so ``hasattr(model, "predict")`` is False
        otherwise.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Rows to label, with the number of features seen in `fit`.

        Returns
        -------
        labels : Array of shape (n_queries,), int32
            Cluster of each row, numbered as `labels_`.

        Raises
        ------
        ValueError
            If the model is not fitted or was fitted without
            ``prediction_data=True``.
        NotImplementedError
            If the linkage or metric has no prediction rule.
        """
        # Maintainer notes (moved from the docstring): DEVIATION 2740. The rule
        # is the linkage's own criterion between the new row and each fitted
        # cluster; for single linkage that is the cluster of the nearest
        # training row. THE DISTANCE is DBSCAN's eps accumulator
        # (`dbscan/impl/neighbors/epsilon_neighborhood.mojo::_eps_acc`, L2 arm):
        # the squared Euclidean distance summed features ascending, never
        # rooted, the training row flushed. It is NOT the fit's expanded
        # `||a||^2 + ||b||^2 - 2 a.b` distance, whose value for a row against
        # itself need not be 0. One thread per query row, no fold across rows;
        # the GPU binding and the CPU host binding
        # (`core/labeled_reference_host_predict.mojo`) compute the same bytes.
        return _NATIVE(self, "family_cfb_agglo_predict", (self._bind(self._BINDING), X), _NATIVE_CONTEXT)

    # The training rows of a `prediction_data=True` fit stay on the device
    # under a handle held in `_resident_holder` (bindings/cfb_resident.mojo),
    # released on refit and by the native finalizer connected below.
    def __getstate__(self):
        """A pickle or a deepcopy carries no device handle; the copy uploads
        its own training rows at its first predict."""
        state = self.__dict__.copy()
        state.pop("_resident_holder", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__dict__["_resident_holder"] = {}

    def save(self, path):
        """Save the model's prediction data to `path`.

        Writes an npz file with the training rows, `labels_`, `n_clusters_`,
        `n_features_in_`, `n_leaves_`, the metric and the numeric mode.
        `children_` and `n_boruvka_rounds_` are not saved, so the same fit
        writes the same bytes on every device; a loaded model has neither
        attribute. A loaded model predicts, including on a CPU-only install.

        Parameters
        ----------
        path : str or path-like
            File to write.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        ValueError
            If it was fitted without ``prediction_data=True``.
        NotImplementedError
            If the linkage or metric has no prediction rule.
        """
        return _ARCHIVE(self, "archive_save", (path,), _ARCHIVE_CONTEXT)

    @classmethod
    def load(cls, path):
        """Load a model written by `save`.

        The loaded model can `predict`. Every array is read at its saved
        dtype and never cast.

        Parameters
        ----------
        path : str or path-like
            File written by `save`.

        Returns
        -------
        model : AgglomerativeClustering
            The loaded model.

        Raises
        ------
        ValueError
            If the file is not a saved AgglomerativeClustering or its arrays
            are inconsistent.
        """
        return _ARCHIVE(cls, "host_archive_load", (path,), _ARCHIVE_CONTEXT)


# The saved prediction data, written natively from this schema (bindings/
# archive_schema_native.mojo) with the bytes `_serialize.write_npz` wrote.
# The `_predicts` rule (single linkage under a `_METRICS` metric) is two
# field conditions; the refusals keep the former Python save's wording and order.
_PREDICT_RULE_REFUSAL = ("mojolearn AgglomerativeClustering.save: linkage={0}, metric={1} has no "
                         "predict rule to save (DEVIATION 2740)")
_native_archive_bridge.register(AgglomerativeClustering, _AGGLOMERATIVE_FORMAT, (
    _native_archive_bridge.fitted("labels_"),
    ("need_eq", "linkage", "single", "NotImplementedError", _PREDICT_RULE_REFUSAL, ("linkage", "metric")),
    ("need_in", "metric", dict(_METRICS), "NotImplementedError", _PREDICT_RULE_REFUSAL, ("linkage", "metric")),
    ("need_set", "_fit_X", "ValueError",
     "mojolearn AgglomerativeClustering.save: prediction data was not stored; fit with "
     "AgglomerativeClustering(prediction_data=True)"),
    *_native_archive_bridge.head(_AGGLOMERATIVE_FORMAT),
    ("str", "metric", "metric"),
    ("array", "x", "_fit_X"),
    ("array", "labels", "labels_"),
    ("i8", "meta", (("int", "n_clusters_"), ("int", "n_features_in_"), ("int", "n_leaves_"))),
), host_archive_agglo_metrics=tuple(_ALL_METRICS))
_native_archive_bridge.lifetime(AgglomerativeClustering, "labeled_reference_release", "_resident_holder")

