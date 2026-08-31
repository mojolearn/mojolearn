"""Brute-force exact k-nearest-neighbours on the GPU, and the classifier and
regressor built on it.

`NearestNeighbors` mirrors cuVS's brute force; `KNeighborsClassifier` and
`KNeighborsRegressor` mirror cuML's `kneighbors_classifier.pyx` /
`kneighbors_regressor.pyx` over `ML::knn_classify` / `ML::knn_regress`
(`neighbors/ported/knn/knn.mojo`, `neighbors/ported/selection/knn.mojo`).
"""

import numpy as np

from . import _mojolearn
from ._mode import NumericModeMixin
from ._arrays import _addr, _addr_ro, as_f32_c

_DEFAULT_QUERY_TILE = 256
_EUCLIDEAN_METRICS = ("euclidean", "l2", "minkowski")


class NearestNeighbors(NumericModeMixin):
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

    #: This family's binding, for `NumericModeMixin._bind`.
    _BINDING = "_mojolearn"

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
        self.used_query_tile_ = self._bind("_mojolearn").knn_search(
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
        self.used_query_tile_ = self._bind("_mojolearn").knn_classify(
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
        self.used_query_tile_ = self._bind("_mojolearn").knn_regress(
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


class RadiusNeighbors(NumericModeMixin):
    """Every neighbour inside a radius, over the random ball cover.

    scikit-learn's `RadiusNeighborsMixin.radius_neighbors` shape: a ragged
    `(distances, indices)` pair, one variable-length array per query row.

    **This is not brute force and it is not one of scikit-learn's trees.**
    The index is cuVS's random ball cover
    (`neighbors/ported/neighbors/ball_cover/`), which DBSCAN has used for its
    eps neighbourhood since this library's first release. It returns the
    EXACT set -- its pruning is a triangle-inequality bound, not an
    approximation -- and `neighbors/mojo_only/radius_check.mojo` asserts that
    against a host brute-force oracle per cell, not per total.

    **The distances are recomputed, not stored by the search.** The search
    kernel knows every distance at the moment it decides membership and
    throws them away; a separate pass walks the finished neighbour list and
    recomputes them. `neighbors/mojo_only/radius_distances.mojo` carries the
    reasoning and the argument for why the recomputed value is the same
    value. Under `identical` the check asserts that bit for bit.

    **`sort_results=True` is done here, on the host, and that is deliberate.**
    The device returns each row in ascending INDEX order under `identical`
    (DEVIATION 551), so a STABLE sort by distance yields exactly
    `(distance, index)` lexicographic order, with the index tie-break coming
    free from the order the device already committed to. Doing it host-side
    means the tie-break cannot depend on a lane width.

    Two calls cross the boundary per query, not one, because a radius query's
    output size is not a function of its inputs: the first counts, the caller
    allocates, the second fills. The cost is the ball-cover index built
    twice; `neighbors/estimator.mojo` says so where it is paid.
    """

    _BINDING = "_mojolearn"

    def __init__(
        self,
        radius=1.0,
        *,
        metric="euclidean",
        algorithm="auto",
        p=2,
    ):
        self.radius = radius
        self.metric = metric
        self.algorithm = algorithm
        self.p = p
        self._index = None

    def _check_refusals(self):
        """Every refusal is BY NAME with the reason, raised at fit."""
        if self.metric not in _EUCLIDEAN_METRICS:
            raise ValueError(
                f"mojolearn RadiusNeighbors: metric={self.metric!r} is "
                "refused; only Euclidean ('euclidean', 'l2', 'minkowski' "
                "with p=2) is ported. The ball cover's distance is "
                "`eps_dist_sq` (neighbors/ported/neighbors/ball_cover/"
                "common.mojo), the one arm cuVS carries: another metric is a "
                "port, not a flag"
            )
        if self.metric == "minkowski" and self.p != 2:
            raise ValueError(
                f"mojolearn RadiusNeighbors: metric='minkowski' with p="
                f"{self.p!r} is refused; only p=2 (Euclidean) is ported"
            )
        if self.algorithm != "auto":
            raise ValueError(
                f"mojolearn RadiusNeighbors: algorithm={self.algorithm!r} is "
                "refused. The index here is cuVS's RANDOM BALL COVER, which "
                "is none of scikit-learn's three: it is not 'brute' (there "
                "is an index and it prunes), and it is neither 'ball_tree' "
                "nor 'kd_tree' (it is a one-level cover over sqrt(n) "
                "landmarks, not a tree). Naming any of them would describe "
                "the wrong algorithm, so only 'auto' is accepted. The "
                "results are exact either way"
            )
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError(
                f"mojolearn RadiusNeighbors: radius={self.radius!r} is "
                "refused; it must be positive and finite"
            )

    def fit(self, X, y=None):
        """Store the index. The ball cover is built per query, not here.

        `y` is accepted and ignored, for scikit-learn call-shape
        compatibility. THE BALL COVER IS NOT BUILT HERE and that is a known
        cost rather than an oversight: nothing in this library holds a fitted
        device handle yet, and introducing the first one to save a build on
        the first radius surface is the wrong order to do those two things
        in. `neighbors/estimator.mojo`'s RADIUS NEIGHBOURS banner records it.
        """
        self._check_refusals()
        idx, _ = as_f32_c(X, "X")
        self._index = idx
        self.n_samples_fit_ = idx.shape[0]
        self.n_features_in_ = idx.shape[1]
        return self

    def radius_neighbors(
        self, X=None, radius=None, return_distance=True, sort_results=False
    ):
        """Neighbours within `radius`, as ragged arrays, one row per query.

        Returns `(distances, indices)` when `return_distance` is True and
        `indices` alone otherwise, matching scikit-learn. Each element is a
        1-D array; the arrays have different lengths, so the containers are
        object-dtype arrays rather than a rectangular block.

        `X=None` queries the fitted data against itself, as scikit-learn
        does. Note that scikit-learn EXCLUDES each point from its own
        neighbour list in that case and this does NOT: the ball cover returns
        the self-edge, DBSCAN counts on it, and dropping it here would make
        the Python surface disagree with the CSR every other consumer sees.
        That difference is named rather than papered over.
        """
        if self._index is None:
            raise ValueError(
                "mojolearn RadiusNeighbors: call fit() before "
                "radius_neighbors()"
            )
        r = float(self.radius if radius is None else radius)
        if not np.isfinite(r) or r <= 0:
            raise ValueError(
                f"mojolearn RadiusNeighbors: radius={radius!r} is refused; "
                "it must be positive and finite"
            )
        if sort_results and not return_distance:
            raise ValueError(
                "mojolearn RadiusNeighbors: sort_results=True requires "
                "return_distance=True; there is nothing to sort by otherwise"
            )

        idx = self._index
        if X is None:
            q = idx
        else:
            q, _ = as_f32_c(X, "X")
            if q.shape[1] != idx.shape[1]:
                raise ValueError(
                    f"mojolearn RadiusNeighbors: X has {q.shape[1]} features "
                    f"but the index was fit on {idx.shape[1]}"
                )
        nq = q.shape[0]

        indptr = np.empty(nq + 1, dtype=np.int32)
        nnz = self._bind("_mojolearn").radius_neighbors_count(
            _addr_ro(idx), _addr_ro(q), _addr(indptr),
            # ORDER MATCHES bindings/_mojolearn.mojo::
            # radius_neighbors_count_binding. n_index, n_queries, n_features,
            # radius
            [idx.shape[0], nq, idx.shape[1], r],
        )
        cols = np.empty(nnz, dtype=np.int32)
        dists = np.empty(nnz, dtype=np.float32)
        got = self._bind("_mojolearn").radius_neighbors_fill(
            _addr_ro(idx), _addr_ro(q), _addr(indptr), _addr(cols),
            _addr(dists),
            # n_index, n_queries, n_features, radius, nnz_capacity,
            # return_sqrt
            [idx.shape[0], nq, idx.shape[1], r, nnz, 1],
        )
        # The Mojo side already refuses `got > nnz`. This asserts the other
        # direction too, because a SHORT fill would leave the tail of `cols`
        # uninitialised and every row after it wrong, and nothing downstream
        # would notice.
        if got != nnz:
            raise RuntimeError(
                f"mojolearn RadiusNeighbors: the counting pass found {nnz} "
                f"neighbours and the filling pass found {got}, on the same "
                "arrays. The two passes rebuild the index independently, so "
                "this means the input arrays changed between them"
            )

        ind = np.empty(nq, dtype=object)
        dst = np.empty(nq, dtype=object)
        for i in range(nq):
            a, b = int(indptr[i]), int(indptr[i + 1])
            row_i = cols[a:b].astype(np.int64)
            row_d = dists[a:b]
            if sort_results:
                # STABLE, and the stability is the point: the row arrives in
                # ascending index order under `identical`, so ties in
                # distance keep that order and the result is
                # (distance, index) lexicographic without a second key.
                order = np.argsort(row_d, kind="stable")
                row_i = row_i[order]
                row_d = row_d[order]
            ind[i] = row_i
            dst[i] = row_d
        if return_distance:
            return dst, ind
        return ind
