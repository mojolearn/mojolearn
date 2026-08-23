"""Brute-force exact k-nearest-neighbours on the GPU, and the classifier and
regressor built on it.

`NearestNeighbors` mirrors cuVS's brute force; `KNeighborsClassifier` and
`KNeighborsRegressor` mirror cuML's `kneighbors_classifier.pyx` /
`kneighbors_regressor.pyx` over `ML::knn_classify` / `ML::knn_regress`
(`neighbors/ported/knn/knn.mojo`, `neighbors/ported/selection/knn.mojo`).
"""

import numpy as np

from . import _mojolearn
from ._arrays import _addr, _addr_ro, as_f32_c

_DEFAULT_QUERY_TILE = 256
_EUCLIDEAN_METRICS = ("euclidean", "l2", "minkowski")


class NearestNeighbors:
    """Exact k-NN by brute force, mirroring cuVS's fused L2 kernel.

    EXACT, not approximate. There is no index to build and no recall to trade
    away: every query is compared against every index point. That is a
    deliberate scope choice rather than a missing feature -- an approximate
    index is a different algorithm with a different contract, and this library
    does not ship one.

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY (measured row by row by
    `tools/e2u_matrix_fit.py`):

        n_neighbors   honored   k. Refused above n_samples_fit (the
                                upstream's short-index fill is not ported:
                                knn_brute_force.mojo) and, UNDER
                                NUMERIC_IDENTICAL ONLY, above 256 -- the
                                identical selector's rank pass gives one
                                thread per output slot (DEVIATION 500,
                                neighbors/mojo_only/select_radix_identical
                                .mojo). FAST runs k > 256 through the
                                ported RAFT radix select.
        query_tile    honored   a MEMORY number; the answer does not depend
                                on it (`check_knn_tiled_is_query_tile_
                                invariant`), only the workspace does
        metric        refused   anything but Euclidean ('euclidean', 'l2',
                                or 'minkowski' with p=2). The ported kernels
                                carry only the expanded-L2 arm
                                (neighbors/ported/neighbors/detail/
                                knn_brute_force.mojo's dispatch note):
                                cosine and L1 are a PORT, not a flag.
        algorithm     refused   anything but 'brute' (and 'auto', which IS
                                brute here): this class is exact brute
                                force and ships no tree or graph index.
        p             refused   anything but 2, for the same reason as
                                metric.

    The ARM (fused vs tiled, `KNN_METHOD_AUTO`) is NOT a parameter of this
    class by design (POLICY CHOICE 4 above); under NUMERIC_IDENTICAL AUTO is
    pinned to the tiled arm on every column (DEVIATION 509).

    Parameters
    ----------
    n_neighbors : int, default 5
        scikit-learn's default.
    query_tile : int, default 256
        Queries processed per pass. **This is the value every published
        mojolearn k-NN number was measured at.** It may be lowered
        automatically when the workspace it implies would be too large; after
        `kneighbors` runs, `used_query_tile_` reports what actually ran, so a
        benchmark can record the configuration instead of assuming it.

    Attributes
    ----------
    used_query_tile_ : int
        Set by `kneighbors`. Differs from `query_tile` when the workspace cap
        fired, which means the run was not the measured configuration.
    """

    def __init__(
        self,
        n_neighbors=5,
        query_tile=_DEFAULT_QUERY_TILE,
        *,
        metric="euclidean",
        algorithm="brute",
        p=2,
    ):
        self.n_neighbors = n_neighbors
        self.query_tile = query_tile
        self.metric = metric
        self.algorithm = algorithm
        self.p = p
        self._index = None
        self.used_query_tile_ = None

    def _check_refusals(self):
        """Every refusal is BY NAME with the reason, raised at fit so a
        caller learns before the index is held."""
        if self.metric not in _EUCLIDEAN_METRICS:
            raise ValueError(
                f"mojolearn NearestNeighbors: metric={self.metric!r} is "
                "refused; only Euclidean ('euclidean', 'l2', 'minkowski' with "
                "p=2) is ported. The brute-force kernels carry the "
                "expanded-L2 arm only (neighbors/ported/neighbors/detail/"
                "knn_brute_force.mojo): another metric is a port, not a flag"
            )
        if self.metric == "minkowski" and self.p != 2:
            raise ValueError(
                f"mojolearn NearestNeighbors: metric='minkowski' with p="
                f"{self.p!r} is refused; only p=2 (Euclidean) is ported"
            )
        if self.algorithm not in ("brute", "auto"):
            raise ValueError(
                f"mojolearn NearestNeighbors: algorithm={self.algorithm!r} is "
                "refused; this class is exact brute force ('brute', which "
                "'auto' also means here) and ships no tree or graph index"
            )

    def fit(self, X, y=None):
        """Store the index. There is no index structure to build.

        `y` is accepted and ignored, for scikit-learn call-shape
        compatibility.
        """
        self._check_refusals()
        idx, _ = as_f32_c(X, "X")
        # Held on the instance so the memory outlives this call: the Mojo side
        # borrows the address at `kneighbors` time and owns nothing.
        self._index = idx
        self.n_samples_fit_ = idx.shape[0]
        self.n_features_in_ = idx.shape[1]
        return self

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        """Distances and indices of the nearest neighbours, nearest first.

        Returns `(distances, indices)` when `return_distance`, else
        `indices`, matching scikit-learn's layout and its ordering.

        Distances are Euclidean, not squared. The kernel computes squared
        distances and the square root is taken on the way out, over
        `n_queries * k` values rather than `n_queries * n_index`, so it is not
        on the hot path.
        """
        if self._index is None:
            raise ValueError("mojolearn: call fit before kneighbors")
        k = self.n_neighbors if n_neighbors is None else n_neighbors
        q, _ = as_f32_c(X, "X")
        if q.shape[1] != self.n_features_in_:
            raise ValueError(
                f"mojolearn: X has {q.shape[1]} features, index has "
                f"{self.n_features_in_}"
            )
        if k < 1 or k > self.n_samples_fit_:
            raise ValueError(
                f"mojolearn: n_neighbors must be in [1, {self.n_samples_fit_}]"
                f", got {k}"
            )

        nq = q.shape[0]
        dist = np.empty((nq, k), dtype=np.float32)
        ind = np.empty((nq, k), dtype=np.uint32)

        # Every array named here stays in a local for the whole call. That is
        # the contract `_arrays` documents and the reason it is spelled out.
        idx = self._index
        self.used_query_tile_ = _mojolearn.knn_search(
            _addr_ro(idx),
            _addr_ro(q),
            _addr(dist),
            _addr(ind),
            # ORDER MATCHES bindings/_mojolearn.mojo::knn_search_binding.
            # n_index, n_queries, n_features, k, return_sqrt, query_tile
            [idx.shape[0], nq, idx.shape[1], k, 1, self.query_tile],
        )

        if return_distance:
            return dist, ind.astype(np.int64)
        return ind.astype(np.int64)


class KNeighborsClassifier(NearestNeighbors):
    """k-NN classification by UNWEIGHTED majority vote over the `n_neighbors`
    nearest index points, mirroring cuML's `KNeighborsClassifier`.

    `predict` is cuML's `knn_classify`: for each query, count the classes of
    its `k` neighbours (each worth `1/k`), take the argmax, and on a tie
    return the LOWEST class in sorted order -- cuML's `class_vote_kernel`
    (`src_prims/selection/knn.cuh:74-109`), scikit-learn's `mode`. The vote
    is a serial fold per query in neighbour order (nearest first, ties in
    distance by lowest index), so it is bit-identical in both numeric modes
    by construction (`neighbors/ported/selection/knn.mojo`, DEVIATION 542).

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY (measured row by row by
    `tools/e2u_matrix_fit.py`; the `NearestNeighbors` rows apply as well):

        n_neighbors   honored   k, as NearestNeighbors (refused above
                                n_samples_fit; under NUMERIC_IDENTICAL above
                                256)
        weights       refused   anything but 'uniform'. cuML refuses
                                'distance' too ("Only uniform weighting
                                strategy is supported currently",
                                kneighbors_classifier.pyx:191) and the
                                ported vote has no weight: a weighted vote
                                is a port, not a flag
        metric        refused   anything but Euclidean, as NearestNeighbors
        algorithm     refused   anything but 'brute' / 'auto', as
                                NearestNeighbors
        p             refused   anything but 2, as NearestNeighbors
        y             honored   int labels, ANY values (negative, gaps);
                                1-D or 2-D (multi-output, one vote per
                                column, as cuML's `vector<int*> y`)

    `predict_proba` returns each query's vote fractions over `classes_`
    (cuML's `knn_class_proba`), one array, or a list of arrays for a 2-D
    `y`, exactly as cuML and scikit-learn shape it.

    Parameters
    ----------
    n_neighbors : int, default 5
    weights : {'uniform'}, default 'uniform'
    query_tile, metric, algorithm, p : as NearestNeighbors

    Attributes
    ----------
    classes_ : ndarray, or list of ndarray for a 2-D `y`
        The sorted unique labels per output (`np.unique`, the pyx's
        `cp.unique`). The Mojo side recomputes the same set with the ported
        `getUniquelabels` and `predict` asserts the two agree.
    outputs_2d_ : bool
    """

    def __init__(
        self,
        n_neighbors=5,
        query_tile=_DEFAULT_QUERY_TILE,
        *,
        weights="uniform",
        metric="euclidean",
        algorithm="brute",
        p=2,
    ):
        super().__init__(
            n_neighbors, query_tile, metric=metric, algorithm=algorithm, p=p
        )
        self.weights = weights
        self._y_cols = None

    def _check_refusals(self):
        super()._check_refusals()
        if self.weights != "uniform":
            raise ValueError(
                f"mojolearn {type(self).__name__}: weights={self.weights!r} "
                "is refused; only 'uniform' is ported. cuML refuses it too "
                "(kneighbors_classifier.pyx:191, 'Only uniform weighting "
                "strategy is supported currently') and the ported vote "
                "(neighbors/ported/selection/knn.mojo) carries no weight: "
                "a distance-weighted vote is a port, not a flag"
            )

    def fit(self, X, y):
        """Store the index and the labels. `y` is int, 1-D or 2-D."""
        super().fit(X)
        y = np.asarray(y)
        if y.ndim not in (1, 2):
            raise ValueError(
                f"mojolearn: y must be 1-D or 2-D, got {y.ndim}-D"
            )
        if y.shape[0] != self.n_samples_fit_:
            raise ValueError(
                f"mojolearn: y has {y.shape[0]} rows, X has "
                f"{self.n_samples_fit_}"
            )
        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError(
                f"mojolearn: y must be integer class labels, got dtype "
                f"{y.dtype}; cuML converts to int32 (check_dtype=np.int32) "
                "and so does this surface -- cast your labels"
            )
        y2 = y.reshape(y.shape[0], -1)
        if y2.dtype != np.int32:
            if (y2 < np.iinfo(np.int32).min).any() or (
                y2 > np.iinfo(np.int32).max
            ).any():
                raise ValueError("mojolearn: y does not fit int32")
            y2 = y2.astype(np.int32)
        # POLICY 6 (neighbors/estimator.mojo): the binding takes `n_outputs`
        # CONTIGUOUS columns -- cuML's order='F' `y`. One transpose at fit.
        self._y_cols = np.ascontiguousarray(y2.T)
        self.outputs_2d_ = y.ndim == 2 and y.shape[1] != 1
        self._classes_list = [np.unique(col) for col in self._y_cols]
        return self

    @property
    def classes_(self):
        if self._y_cols is None:
            raise AttributeError("classes_ is set by fit")
        if self.outputs_2d_:
            return self._classes_list
        return self._classes_list[0]

    def _predict(self, X, want_proba):
        if self._index is None or self._y_cols is None:
            raise ValueError("mojolearn: call fit before predict")
        q, _ = as_f32_c(X, "X")
        if q.shape[1] != self.n_features_in_:
            raise ValueError(
                f"mojolearn: X has {q.shape[1]} features, index has "
                f"{self.n_features_in_}"
            )
        k = self.n_neighbors
        if k < 1 or k > self.n_samples_fit_:
            raise ValueError(
                f"mojolearn: n_neighbors must be in [1, {self.n_samples_fit_}]"
                f", got {k}"
            )
        nq = q.shape[0]
        n_out = self._y_cols.shape[0]
        n_classes = [int(c.shape[0]) for c in self._classes_list]
        labels = np.empty((nq, n_out), dtype=np.int32)
        proba = np.empty(nq * sum(n_classes), dtype=np.float32)
        uniq = np.empty(sum(n_classes), dtype=np.int32)
        idx = self._index
        y_cols = self._y_cols
        self.used_query_tile_ = _mojolearn.knn_classify(
            _addr_ro(idx),
            _addr_ro(q),
            _addr_ro(y_cols),
            _addr(labels),
            _addr(proba),
            _addr(uniq),
            # ORDER MATCHES bindings/_mojolearn.mojo::knn_classify_binding.
            # n_index, n_queries, n_features, k, query_tile, n_outputs,
            # want_proba, then n_classes per output
            [idx.shape[0], nq, idx.shape[1], k, self.query_tile, n_out,
             1 if want_proba else 0] + n_classes,
        )
        # POLICY 7: the port's class set against ours, made visible.
        got = np.split(uniq, np.cumsum(n_classes)[:-1])
        for i, (a, b) in enumerate(zip(got, self._classes_list)):
            if not np.array_equal(a, b):
                raise RuntimeError(
                    f"mojolearn: output {i}: the ported getUniquelabels found "
                    f"classes {a.tolist()[:8]}..., np.unique found "
                    f"{b.tolist()[:8]}...; the two class sets disagree"
                )
        return labels, proba, n_classes

    def predict(self, X):
        """The voted class per query: shape `(n_queries,)`, or
        `(n_queries, n_outputs)` for a 2-D `y`. Original label values."""
        labels, _, _ = self._predict(X, want_proba=False)
        if self.outputs_2d_:
            return labels.astype(np.int64)
        return labels[:, 0].astype(np.int64)

    def predict_proba(self, X):
        """Vote fractions per class, columns in `classes_` order; a list of
        arrays for a 2-D `y`."""
        _, proba, n_classes = self._predict(X, want_proba=True)
        nq = np.asarray(X).shape[0]
        out = []
        off = 0
        for n in n_classes:
            out.append(proba[off:off + nq * n].reshape(nq, n).copy())
            off += nq * n
        if self.outputs_2d_:
            return out
        return out[0]


class KNeighborsRegressor(NearestNeighbors):
    """k-NN regression by the UNWEIGHTED mean of the `n_neighbors` nearest
    targets, mirroring cuML's `KNeighborsRegressor` (`ML::knn_regress`,
    `regress_avg_kernel`, `src_prims/selection/knn.cuh:112-131`).

    The mean is a serial float32 fold per query in neighbour order, then one
    division by `k`; the same arithmetic in both numeric modes, with the
    seams flushed under NUMERIC_IDENTICAL (DEVIATION 542).

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY (the `NearestNeighbors` rows
    apply as well):

        n_neighbors   honored   as NearestNeighbors
        weights       refused   anything but 'uniform', as the classifier
                                (kneighbors_regressor.pyx:188)
        metric, algorithm, p    as NearestNeighbors
        y             honored   float targets, 1-D or 2-D (multi-output);
                                float64 is cast to float32 (cuML:
                                convert_to_dtype=np.float32)

    Parameters / Attributes: as KNeighborsClassifier, without `classes_`.
    """

    def __init__(
        self,
        n_neighbors=5,
        query_tile=_DEFAULT_QUERY_TILE,
        *,
        weights="uniform",
        metric="euclidean",
        algorithm="brute",
        p=2,
    ):
        super().__init__(
            n_neighbors, query_tile, metric=metric, algorithm=algorithm, p=p
        )
        self.weights = weights
        self._y_cols = None

    def _check_refusals(self):
        super()._check_refusals()
        if self.weights != "uniform":
            raise ValueError(
                f"mojolearn {type(self).__name__}: weights={self.weights!r} "
                "is refused; only 'uniform' is ported. cuML refuses it too "
                "(kneighbors_regressor.pyx:188) and the ported mean "
                "(neighbors/ported/selection/knn.mojo) carries no weight: "
                "a distance-weighted mean is a port, not a flag"
            )

    def fit(self, X, y):
        """Store the index and the targets. `y` is float, 1-D or 2-D."""
        super().fit(X)
        y = np.asarray(y)
        if y.ndim not in (1, 2):
            raise ValueError(
                f"mojolearn: y must be 1-D or 2-D, got {y.ndim}-D"
            )
        if y.shape[0] != self.n_samples_fit_:
            raise ValueError(
                f"mojolearn: y has {y.shape[0]} rows, X has "
                f"{self.n_samples_fit_}"
            )
        y2 = y.reshape(y.shape[0], -1).astype(np.float32, copy=False)
        self._y_cols = np.ascontiguousarray(y2.T)
        self.outputs_2d_ = y.ndim == 2 and y.shape[1] != 1
        return self

    def predict(self, X):
        """The mean target per query: `(n_queries,)`, or
        `(n_queries, n_outputs)` for a 2-D `y`. float32."""
        if self._index is None or self._y_cols is None:
            raise ValueError("mojolearn: call fit before predict")
        q, _ = as_f32_c(X, "X")
        if q.shape[1] != self.n_features_in_:
            raise ValueError(
                f"mojolearn: X has {q.shape[1]} features, index has "
                f"{self.n_features_in_}"
            )
        k = self.n_neighbors
        if k < 1 or k > self.n_samples_fit_:
            raise ValueError(
                f"mojolearn: n_neighbors must be in [1, {self.n_samples_fit_}]"
                f", got {k}"
            )
        nq = q.shape[0]
        n_out = self._y_cols.shape[0]
        out = np.empty((nq, n_out), dtype=np.float32)
        idx = self._index
        y_cols = self._y_cols
        self.used_query_tile_ = _mojolearn.knn_regress(
            _addr_ro(idx),
            _addr_ro(q),
            _addr_ro(y_cols),
            _addr(out),
            # ORDER MATCHES bindings/_mojolearn.mojo::knn_regress_binding.
            # n_index, n_queries, n_features, k, query_tile, n_outputs
            [idx.shape[0], nq, idx.shape[1], k, self.query_tile, n_out],
        )
        if self.outputs_2d_:
            return out
        return out[:, 0]
