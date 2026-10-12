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

import sys

from . import _backend, _expansion_cluster, _mojolearn_solver, _serialize
from ._array import Array
from ._buffer import addr, addr_ro, all_finite, as_f32_c, empty, frombytes, zeros
from .density import _check_queries
from .linear_model import _check_saved_by, _restore_mode, _saved_mode

#: `AgglomerativeClustering.save`'s format tag
#: (lane/inference-transductive-predict, 2026-09-15).
_AGGLOMERATIVE_FORMAT = "mojolearn-agglomerative-1"

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

#: scikit-learn's metric names on the x_cluster route (`x_cluster/agglo.mojo`):
#: -1 euclidean through the squared distance, 1 manhattan, 4 cosine
#: (`bodies.pdist_cell`, DEVIATION 5111), 5 precomputed.
_ALL_METRICS = {"euclidean": -1, "l2": -1, "l1": 1, "manhattan": 1, "cityblock": 1,
                "cosine": 4, "precomputed": 5}

#: `bodies.LINK_*`.
_LINKAGES = {"ward": 0, "complete": 1, "average": 2, "single": 3}

# `cuvs/cluster/agglomerative.hpp::Linkage`, as cuML's Python spells it.
_CONNECTIVITIES = {"pairwise": 0, "knn": 1}

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
# Defaults: connectivity 'pairwise' here; cuML's PYTHON default is 'knn'
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
class AgglomerativeClustering:
    """Agglomerative hierarchical clustering.

    Builds a merge tree bottom-up and cuts it into `n_clusters` clusters or
    at `distance_threshold`. Supports scikit-learn's single, complete,
    average and Ward linkages, connectivity matrices and per-merge
    distances. Input is a dense array, converted to float32.

    Two defaults differ from scikit-learn's and change the result:
    `linkage` is 'single' (scikit-learn: 'ward') and `connectivity` is
    'pairwise', the full dense graph (scikit-learn's None). cuML's
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
    connectivity : 'pairwise', None, array-like, sparse matrix or callable, default 'pairwise'
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

    #: The binding `predict` calls. The FIT is `_mojolearn_solver`'s, a
    #: training-only family; the prediction entry lives in the estimators
    #: family, whose host binding ships in the inference wheel.
    _BINDING = "_mojolearn_estimators"

    def __init__(self, n_clusters=2, *, metric="euclidean",
                 connectivity="pairwise", linkage="single", c=15,
                 memory=None, compute_full_tree="auto",
                 distance_threshold=None, compute_distances=False,
                 prediction_data=False):
        # scikit-learn's linkages (lane/algos-cluster option parity,
        # 2026-09-27). 'single' with the euclidean metric, no connectivity
        # matrix and no per-merge distances stays on cuML's Boruvka route
        # (`_mojolearn_solver.linkage_fit`, bits unchanged); everything
        # else takes the x_cluster route (`x_cluster/agglo.mojo`).
        if linkage not in _LINKAGES:
            raise ValueError(
                f"linkage must be one of {sorted(_LINKAGES)}, got {linkage!r}")
        if connectivity is None:
            # scikit-learn's spelling of "no connectivity constraint", which
            # is the full dense graph, which is cuML's 'pairwise'.
            connectivity = "pairwise"
        if isinstance(connectivity, str):
            if connectivity not in _CONNECTIVITIES:
                raise ValueError(
                    "'connectivity' can only be one of {'knn', 'pairwise'}, "
                    "a connectivity matrix or a callable")
            if connectivity == "knn":
                raise NotImplementedError(
                    "mojolearn AgglomerativeClustering: connectivity='knn' is "
                    "REFUSED BY NAME. The Linkage::KNN_GRAPH specialization "
                    "(connectivities.cuh:49), the cross-component fix-up its "
                    "forest MST needs (connect_knn_graph, mst.cuh:67 and :131) "
                    "and merge_msts are rung 2 and NOT IMPLEMENTED -- and their host "
                    "overload picks a RANDOM vertex per component from "
                    "std::mt19937(std::random_device()), which would have to be "
                    "pinned first. Use connectivity='pairwise' (cuML's C++ "
                    "default and scikit-learn's dense tree), or pass the k-NN "
                    "graph itself as a connectivity matrix (scikit-learn's "
                    "kneighbors_graph). See hierarchy/NOT_IMPLEMENTED.tsv"
                )
        if metric not in _ALL_METRICS:
            raise NotImplementedError(
                f"mojolearn AgglomerativeClustering: metric={metric!r} is "
                f"refused by name; one of {sorted(_ALL_METRICS)}. A callable "
                "metric would compute the dissimilarities outside every "
                "identity column. See x_cluster/NOT_IMPLEMENTED.tsv"
            )
        if linkage == "ward" and metric not in ("euclidean", "l2"):
            raise ValueError(
                f"{metric} was provided as metric. Ward can only work with "
                "euclidean distances (i.e. 'euclidean' and 'l2').")
        if c != 15:
            raise NotImplementedError(
                f"mojolearn AgglomerativeClustering: c={c!r} is refused. It "
                "tunes k = log(n) + c for the k-NN graph, cuML forwards it "
                "only when connectivity='knn' (linkage.cu:40, "
                "'use_knn ? c : 0'), and that arm is refused by name here, "
                "so any value you pass would reach nothing"
            )
        if memory is not None:
            raise NotImplementedError(
                "mojolearn AgglomerativeClustering: memory is refused; it is "
                "scikit-learn's joblib cache for a host tree build and there "
                "is no host tree here"
            )
        if compute_full_tree not in ("auto", True, False):
            raise ValueError(
                "compute_full_tree must be 'auto', True or False, got "
                f"{compute_full_tree!r}")
        if not ((n_clusters is None) ^ (distance_threshold is None)):
            raise ValueError(
                "Exactly one of n_clusters and distance_threshold has to be "
                "set, and the other needs to be None.")
        if distance_threshold is not None and not compute_full_tree:
            raise ValueError(
                "compute_full_tree must be True if distance_threshold is set.")
        self.n_clusters = n_clusters
        self.metric = metric
        self.connectivity = connectivity
        self.linkage = linkage
        self.c = 15
        self.memory = None
        self.compute_full_tree = compute_full_tree
        self.distance_threshold = distance_threshold
        self.compute_distances = bool(compute_distances)
        self.prediction_data = prediction_data

    def _legacy_route(self):
        """cuML's single-linkage Boruvka route: the recorded `agglomerative`
        lanes' bits. Everything else is the x_cluster route."""
        return (self.linkage == "single" and self.metric in _METRICS
                and isinstance(self.connectivity, str)
                and self.distance_threshold is None and not self.compute_distances)

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
        if not isinstance(self.prediction_data, bool):
            raise TypeError(
                "mojolearn AgglomerativeClustering: prediction_data must be a "
                f"bool, got {type(self.prediction_data).__name__}"
            )
        if not self._legacy_route():
            return self._fit_x(X)
        if hasattr(X, "toarray") or hasattr(X, "tocsr"):
            raise NotImplementedError(
                "mojolearn AgglomerativeClustering: sparse X is refused; the "
                "pairwise connectivity arm builds its dense m x m graph from "
                "a dense float32 matrix"
            )
        x, self.input_copied_ = as_f32_c(X, ndim=2, name="X")
        n_rows, n_cols = x.shape
        if n_rows < 2:
            raise ValueError(
                f"mojolearn AgglomerativeClustering: n_rows={n_rows} < 2; "
                "single linkage needs at least two points"
            )
        # FAST on Apple builds the Euclidean MST from Boruvka rounds without
        # the dense graph (hierarchy/impl/cluster/detail/fast_boruvka.mojo,
        # taken for n_cols <= 64), so the dense matrix's row cap does not
        # apply there.
        fast_mst = (
            _backend.requested_mode() == "fast"
            and sys.platform == "darwin"
            and n_cols <= 64
        )
        if n_rows > PAIRWISE_MAX_ROWS and not fast_mst:
            raise ValueError(
                f"mojolearn AgglomerativeClustering: n_rows={n_rows} > "
                f"{PAIRWISE_MAX_ROWS}; the dense connectivity matrix is "
                "m * m of cuVS's int and overflows past that "
                "(hierarchy/impl/cluster/detail/connectivities.mojo "
                "refuses it by name too)"
            )
        k = int(self.n_clusters)
        if k < 1 or k > n_rows:
            raise ValueError(
                f"Expected 1 <= n_clusters <= n_rows ({n_rows}), got "
                f"n_clusters={k}"
            )

        # Outputs are `_array.Array`s (DEVIATION 2371; they were ndarrays).
        children = empty((n_rows - 1, 2), "<i4")
        labels = empty((n_rows,), "<i4")
        info = zeros((2,), "<i4")
        _mojolearn_solver.linkage_fit(
            addr_ro(x, name="X"), addr(children, name="children_"),
            addr(labels, name="labels_"), addr(info, name="info"),
            # ORDER MATCHES bindings/_mojolearn_solver.mojo::linkage_fit_binding.
            # n_rows, n_cols, n_clusters, metric, use_knn
            [
                n_rows, n_cols, k, _METRICS[self.metric],
                _CONNECTIVITIES[self.connectivity],
            ],
        )
        self.labels_ = labels
        self.children_ = children
        self.n_clusters_ = k
        self.n_leaves_ = n_rows
        # -1 on a CPU-only install (the CPU training lane, 2026-09-14): the
        # host binding runs Kruskal under the device's total order, so there
        # is no Boruvka pass to count; labels_ and children_ are the same.
        self.n_boruvka_rounds_ = int(info[0])
        self.n_connected_components_ = int(info[1])
        self.n_features_in_ = n_cols
        # A borrowed input is copied so a caller's later write cannot move
        # what predict reads.
        self._fit_X = (x if self.input_copied_ else x.copy()) if self.prediction_data else None
        return self

    def _predicts(self):
        """predict (DEVIATION 2740) is the nearest training row's cluster:
        single linkage's own criterion under the euclidean metric only."""
        return self.linkage == "single" and self.metric in _METRICS

    def _connectivity_aux(self, X, n):
        """The connectivity matrix as the binding takes it (lane
        apple-fast-py2mojo-cluster: `x_cluster/agglo.mojo` keeps its nonzero
        off-diagonal entries, which Python's edge list did before):
        (aux, count, edge_mode), mode 1 the dense n x n float32 matrix
        (count n * n), mode 2 the COO rows, columns and values as float32,
        concatenated (count nnz)."""
        conn = self.connectivity
        if callable(conn):
            conn = conn(X)
        tocoo = getattr(conn, "tocoo", None)
        if callable(tocoo):
            coo = tocoo()
            shape = tuple(coo.shape)
            rows, _ = as_f32_c(coo.row, ndim=1, name="connectivity rows")
            cols, _ = as_f32_c(coo.col, ndim=1, name="connectivity cols")
            vals, _ = as_f32_c(coo.data, ndim=1, name="connectivity values")
            nnz = int(vals.shape[0])
            aux = (frombytes(rows.tobytes() + cols.tobytes() + vals.tobytes(), "<f4", (3 * nnz,))
                   if nnz else None)
            mode, count = 2, nnz
        else:
            dense, _ = as_f32_c(conn, ndim=2, name="connectivity")
            shape = tuple(dense.shape)
            aux, mode, count = dense, 1, int(dense.size)
        if shape != (n, n):
            raise ValueError(
                f"Wrong shape for connectivity matrix: {shape} when X has {n} samples")
        return aux, count, mode

    def _fit_x(self, X):
        """The x_cluster route (`x_cluster/agglo.mojo`, ENTRY_AGGLO): every
        linkage, metric, connectivity matrix and the per-merge distances,
        with scikit-learn's `_fit` bookkeeping around it."""
        import warnings
        if hasattr(X, "toarray") or hasattr(X, "tocsr"):
            raise NotImplementedError(
                "mojolearn AgglomerativeClustering: sparse X is refused; the "
                "dissimilarities are a dense float32 matrix")
        x, self.input_copied_ = as_f32_c(X, ndim=2, name="X")
        n, d = (int(v) for v in x.shape)  # glue: the two shape dims
        if n < 2:
            raise ValueError(
                f"mojolearn AgglomerativeClustering: n_rows={n} < 2; a tree "
                "needs at least two points")
        if not all_finite(x):
            raise ValueError("mojolearn AgglomerativeClustering: X contains NaN or infinity")
        metric = _ALL_METRICS[self.metric]
        if metric == 5 and n != d:
            raise ValueError(f"Distance matrix should be square, got matrix of shape {(n, d)}")
        if metric == 4:
            # lane cpu2-l9-neighbors: the zero-row test on the device (the
            # x_neighbors binding's `xn_row_all_zero`, one flag per row from
            # the bit patterns, +-0.0 a zero and NaN not), then the device
            # `reduce_stat` max of the flags; was `x.tolist()` and a Python
            # walk of every value
            if d == 0:
                raise ValueError("Cosine affinity cannot be used when X contains zero vectors")
            zrow = empty((n,), "<i4")
            self._bind("_mojolearn_x_neighbors").xn_row_all_zero(
                [addr_ro(x, name="X"), addr(zrow, name="zero rows")], [n, d], [])
            if int(zrow.max()) > 0:
                raise ValueError("Cosine affinity cannot be used when X contains zero vectors")
        if self.n_clusters is not None:
            k = int(self.n_clusters)
            if k < 1 or k > n:
                raise ValueError(
                    f"Cannot extract more clusters than samples: {k} clusters "
                    f"were given for a tree with {n} leaves.")
        constrained = not isinstance(self.connectivity, str)
        full = True
        if constrained:
            full = self.compute_full_tree
            if full == "auto":
                full = (self.distance_threshold is not None
                        or int(self.n_clusters) < max(100, 0.02 * n))
        n_merges = n - 1 if full else n - int(self.n_clusters)
        # the cluster lane's binding, named by its door (a name here would
        # make this file a whole-surface registry to tools/lane_select.py)
        b = _backend.binding(_expansion_cluster._XCluster._BINDING, getattr(self, "numeric_mode", None))
        # the connectivity filter, the distance-threshold count and the
        # labels in the binding (x_cluster/agglo.mojo, tree_cut.mojo)
        aux, n_edges, edge_mode = self._connectivity_aux(x, n) if constrained else (None, -1, 0)
        thr = self.distance_threshold
        cut = (1 if thr is not None else int(self.n_clusters)) if full else 0
        ip = [n, d, _LINKAGES[self.linkage], metric, n_edges, n_merges, edge_mode, cut,
              1 if (full and thr is not None) else 0]
        fp = [2.0, float(thr) if thr is not None else 0.0]
        f, i, sc = b.x_cluster_call(
            _E_AGGLO, addr_ro(x, name="X"), x.size,
            addr_ro(aux, name="connectivity") if aux is not None else 0,
            aux.size if aux is not None else 0, ip, fp)
        children = i[0]
        self.children_ = Array._from_flat(children, (n_merges, 2), "<i4")
        self.n_leaves_ = n
        self.n_connected_components_ = int(sc[0])
        if constrained and self.n_connected_components_ > 1:
            warnings.warn(
                "the number of connected components of the connectivity matrix "
                f"is {self.n_connected_components_} > 1. Completing it to avoid "
                "stopping the tree early.", stacklevel=2)
        distances = f[0]
        if self.distance_threshold is not None or self.compute_distances:
            self.distances_ = Array._from_flat(distances, (n_merges,), "<f4")
        else:
            self.__dict__.pop("distances_", None)
        self.n_clusters_ = (int(sc[1]) if self.distance_threshold is not None and full
                            else int(self.n_clusters))
        self.labels_ = Array._from_flat(i[1], (n,), "<i4")
        self.n_boruvka_rounds_ = -1
        self.n_features_in_ = d
        self._fit_X = None
        if self.prediction_data and self._predicts():
            self._fit_X = x if self.input_copied_ else x.copy()
        return self

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
        return self.fit(X).labels_

    def predict(self, X):
        """Label new rows with the cluster of their nearest training row.

        Neither scikit-learn nor cuML has this method. For single linkage,
        the closest cluster to a new row is the one containing its nearest
        training row (squared euclidean distance); ties go to the lowest
        cluster label, then the lowest training index. A training row
        predicts its own fitted label unless an exact duplicate of it was
        cut into a cluster with a lower label, which happens only when
        `n_clusters` exceeds the number of distinct rows.

        Only single linkage with the 'euclidean' or 'l2' metric can predict,
        and only after a fit with ``prediction_data=True`` (or a model
        loaded with `load`).

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
        if not hasattr(self, "labels_"):
            raise ValueError(
                "mojolearn AgglomerativeClustering.predict: this instance is not "
                "fitted yet; call fit first"
            )
        if not self._predicts():
            raise NotImplementedError(
                f"mojolearn AgglomerativeClustering.predict: linkage={self.linkage!r}, "
                f"metric={self.metric!r} has no predict rule; DEVIATION 2740 is single "
                "linkage's own criterion (the nearest training row's cluster) under "
                "the euclidean metric only")
        if getattr(self, "_fit_X", None) is None:
            raise ValueError(
                "mojolearn AgglomerativeClustering.predict: prediction data was not "
                "stored. Fit with AgglomerativeClustering(prediction_data=True), "
                "which keeps the training rows this rule needs (DEVIATION 2740)"
            )
        q = _check_queries(X, self.n_features_in_, "AgglomerativeClustering.predict")
        nq = int(q.shape[0])
        n_refs = int(self._fit_X.shape[0])
        out = empty((nq,), "<i4")
        chosen = empty((nq,), "<i4")
        self._bind(self._BINDING).labeled_reference_predict(
            # ORDER MATCHES bindings/_mojolearn_estimators.mojo::labeled_reference_predict_binding.
            # The key of every training row is its label: ties go to the
            # lowest cluster label, then the lowest training index.
            [addr_ro(self._fit_X, name="training rows"), addr_ro(self.labels_, name="labels_ (keys)"),
             addr_ro(self.labels_, name="labels_"), addr_ro(q, name="X"),
             addr(out, name="labels"), addr(chosen, name="training rows chosen")],
            # n_refs, n_queries, n_features, metric (0: L2), eps (unused), has_thresh
            [n_refs, nq, int(self.n_features_in_), 0, 0.0, 0],
        )
        return out

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
        if not hasattr(self, "labels_"):
            raise RuntimeError("this estimator is not fitted yet")
        if not self._predicts():
            raise NotImplementedError(
                f"mojolearn AgglomerativeClustering.save: linkage={self.linkage!r}, "
                f"metric={self.metric!r} has no predict rule to save (DEVIATION 2740)")
        if getattr(self, "_fit_X", None) is None:
            raise ValueError(
                "mojolearn AgglomerativeClustering.save: prediction data was not "
                "stored; fit with AgglomerativeClustering(prediction_data=True)"
            )
        arrays = {
            "format": _AGGLOMERATIVE_FORMAT,
            "estimator": type(self).__name__,
            "numeric_mode": _saved_mode(self),
            "metric": str(self.metric),
            "x": self._fit_X,
            "labels": self.labels_,
            "meta": Array.from_list(
                [int(self.n_clusters_), int(self.n_features_in_), int(self.n_leaves_)], "<i8"
            ),
        }
        return _serialize.write_npz(path, arrays)

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
        arrays = _serialize.read_npz(path, _AGGLOMERATIVE_FORMAT)
        _check_saved_by(arrays, path, cls)
        meta = _serialize.exact(arrays, "meta", "<i8")
        if meta.size != 3:
            raise ValueError(f"mojolearn: {path!r} meta holds {meta.size} fields, 3 are needed")
        k, nf, n_leaves = (int(v) for v in meta.tolist())  # glue: unpacks the fixed meta vector
        obj = cls(n_clusters=k, metric=_serialize.scalar_str(arrays, "metric"), prediction_data=True)
        _restore_mode(obj, arrays)
        x = _serialize.exact(arrays, "x", "<f4")
        labels = _serialize.exact(arrays, "labels", "<i4")
        if x.ndim != 2 or tuple(x.shape) != (n_leaves, nf):
            raise ValueError(f"mojolearn: {path!r} x shape {tuple(x.shape)} is not ({n_leaves}, {nf})")
        if labels.size != n_leaves:
            raise ValueError(f"mojolearn: {path!r} labels do not match n_leaves_")
        obj._fit_X = x
        obj.labels_ = labels
        obj.n_clusters_ = k
        obj.n_leaves_ = n_leaves
        obj.n_features_in_ = nf
        return obj

