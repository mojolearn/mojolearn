# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Brute-force exact k-nearest-neighbours on the GPU, and the classifier and
regressor built on it.

`NearestNeighbors` mirrors cuVS's brute force; `KNeighborsClassifier` and
`KNeighborsRegressor` mirror cuML's `kneighbors_classifier.pyx` /
`kneighbors_regressor.pyx` over `ML::knn_classify` / `ML::knn_regress`
(`neighbors/impl/knn/knn.mojo`, `neighbors/impl/selection/knn.mojo`).
"""

import numpy as np

from . import _mojolearn
from ._mode import NumericModeMixin
from ._arrays import _addr, _addr_ro, as_f32_c

_DEFAULT_QUERY_TILE = 256

#: cuVS `DistanceType` values (`cuvs/distance/distance.h:22-69`), mirrored
#: from `neighbors/impl/distance/detail/distance_ops.mojo`. The value gaps
#: are theirs.
_DIST_L2_EXPANDED = 0
_DIST_L2_SQRT_EXPANDED = 1
_DIST_COSINE_EXPANDED = 2
_DIST_L1 = 3
_DIST_L2_SQRT_UNEXPANDED = 5
_DIST_LINF = 7
_DIST_LP_UNEXPANDED = 9

#: THE BALL COVER'S OWN TABLE, and it is NOT `_METRIC_TABLE`.
#:
#: Two rows differ, both for reasons that are about the INDEX and not about
#: the metric's name (`neighbors/impl/neighbors/ball_cover/common.mojo`,
#: DEVIATION 564 and DEVIATION 565).
#:
#:   'euclidean' / 'l2'  goes to L2SqrtUNexpanded here where the brute-force
#:                       table sends it to L2SqrtEXPanded. The cover's index
#:                       stores `R_1nn_dists` and `R_radius` computed by
#:                       summing differences directly, and the query compares
#:                       against them; computing one side by the expanded
#:                       identity `||a||^2 + ||b||^2 - 2ab` and the other
#:                       directly permits a boundary case where a point is
#:                       inside by one formula and outside by the other
#:                       (DEVIATION 2 in `ball_cover.mojo`, and `PORTING.md
#:                       21` for what the identity costs in float32).
#:   'sqeuclidean'       is ABSENT. Squared Euclidean distance is not a
#:                       metric even though Euclidean distance is: on three
#:                       collinear points at unit spacing it gives 4 against
#:                       1 + 1. The pruning here IS the triangle inequality,
#:                       so admitting it would prune true neighbours
#:                       silently. 'cosine' is absent for the same class of
#:                       reason.
_RBC_METRIC_TABLE = {
    "euclidean": _DIST_L2_SQRT_UNEXPANDED,
    "l2": _DIST_L2_SQRT_UNEXPANDED,
    "cityblock": _DIST_L1,
    "l1": _DIST_L1,
    "manhattan": _DIST_L1,
    "taxicab": _DIST_L1,
    "chebyshev": _DIST_LINF,
    "linf": _DIST_LINF,
    "minkowski": _DIST_LP_UNEXPANDED,
    "lp": _DIST_LP_UNEXPANDED,
}

#: cuML's `NearestNeighbors._build_metric_type` (`nearest_neighbors.pyx:
#: 520-553`), the rows this tree computes. NOT the `pairwise_distances`
#: table -- cuML has two and they DISAGREE about "euclidean" (that one
#: sends it to L2SqrtUnexpanded, this one to L2SqrtExpanded). KDE goes
#: through the other; `python/mojolearn/density.py` has its own.
_METRIC_TABLE = {
    "euclidean": _DIST_L2_SQRT_EXPANDED,
    "l2": _DIST_L2_SQRT_EXPANDED,
    "sqeuclidean": _DIST_L2_EXPANDED,
    "cityblock": _DIST_L1,
    "l1": _DIST_L1,
    "manhattan": _DIST_L1,
    "taxicab": _DIST_L1,
    "chebyshev": _DIST_LINF,
    "linf": _DIST_LINF,
    "cosine": _DIST_COSINE_EXPANDED,
    "minkowski": _DIST_LP_UNEXPANDED,
    "lp": _DIST_LP_UNEXPANDED,
}

#: Names in cuML's `VALID_METRICS["brute"]` (`neighbors/__init__.py:27-48`)
#: that this tree does not compute. Refused BY NAME so a caller learns the
#: metric is UNPORTED rather than unknown.
_UNPORTED_METRICS = (
    "canberra",
    "jensenshannon",
    "correlation",
    "inner_product",
    "haversine",
    "braycurtis",
)

_WEIGHTS_UNIFORM = 0
_WEIGHTS_DISTANCE = 1

_WEIGHTS_TABLE = {"uniform": _WEIGHTS_UNIFORM, "distance": _WEIGHTS_DISTANCE}


def _resolve_weights(cls_name, weights):
    """`weights` -> its value for the Mojo boundary.

    scikit-learn also accepts `None` (treated as uniform, `_base.py:92`)
    and a CALLABLE (`:116`). `None` is honored; the callable is refused by
    name, and that refusal is one of the two kinds that are still
    legitimate -- it is genuinely impossible here, because a Python
    function cannot be called from inside a GPU kernel and there is no
    portable way to lift one there.
    """
    if weights is None:
        return _WEIGHTS_UNIFORM
    if callable(weights):
        raise ValueError(
            f"mojolearn {cls_name}: weights=<callable> is NOT PORTED. "
            "scikit-learn calls it on the distance matrix in Python; the "
            "vote here runs in a GPU kernel and there is no portable way "
            "to lift a Python function into one. Use 'uniform' or "
            "'distance'."
        )
    if not isinstance(weights, str) or weights not in _WEIGHTS_TABLE:
        raise ValueError(
            f"mojolearn {cls_name}: weights={weights!r} is not a "
            "weighting; use 'uniform' or 'distance'"
        )
    return _WEIGHTS_TABLE[weights]


def _resolve_metric(cls_name, metric, p):
    """`(metric_value, metric_arg)` for the Mojo boundary.

    scikit-learn's `metric='minkowski', p=2` IS Euclidean and sklearn
    collapses it (`effective_metric_`); cuML does NOT (`:1016-1017` just
    echoes `self.metric`), so `metric='minkowski', p=2` goes through the
    Lp op there and here. That is deliberate: running a DIFFERENT op for
    p=2 would make two spellings share one code path and hide a bug in
    whichever of them is unexercised. The Lp op at p=2 agrees with
    Euclidean to a few ulp, which is what `neighbors/checks/
    metric_check.mojo::check_metric_arg_is_reached` clause 3 measures.
    """
    if not isinstance(metric, str):
        raise ValueError(
            f"mojolearn {cls_name}: metric must be a name, got {metric!r}. "
            "A callable metric is not ported: it would have to run inside a "
            "GPU kernel."
        )
    key = metric.lower()
    if key in _METRIC_TABLE:
        value = _METRIC_TABLE[key]
    elif key in _UNPORTED_METRICS:
        raise ValueError(
            f"mojolearn {cls_name}: metric={metric!r} is in cuML's "
            "VALID_METRICS['brute'] but is NOT PORTED "
            "(neighbors/NOT_IMPLEMENTED.tsv). Ported: "
            + ", ".join(sorted(_METRIC_TABLE))
        )
    else:
        raise ValueError(
            f"mojolearn {cls_name}: unknown metric {metric!r}. Ported: "
            + ", ".join(sorted(_METRIC_TABLE))
        )

    arg = 2.0
    if value == _DIST_LP_UNEXPANDED:
        arg = float(p)
        # DEVIATION 552, mirrored here so the message names the Python
        # parameter the caller actually typed. The Mojo side refuses the
        # same set by value before any launch.
        if not (arg > 0.0) or arg != arg or arg == float("inf"):
            raise ValueError(
                f"mojolearn {cls_name}: metric='minkowski' needs a finite "
                f"p > 0, got {p!r}. p = 0 makes 1/p infinite, p < 0 is not "
                "a metric, and p = infinity is metric='chebyshev'."
            )
        if arg < 1.1754943508222875e-38:
            raise ValueError(
                f"mojolearn {cls_name}: metric='minkowski' p={p!r} is "
                "subnormal in float32; the flush policy differs by vendor "
                "so this cannot be one arithmetic (DEVIATION 552)"
            )
    return value, arg


def _resolve_rbc_metric(cls_name, metric, p):
    """`(metric_value, metric_arg)` for anything running on the BALL COVER.

    THE REFUSAL HERE IS NARROWER THAN IT WAS, AND THE ARGUMENT DECIDES THE
    WIDTH. Until 2026-09-01 this index admitted Euclidean and nothing else,
    with the correct reason attached: the random ball cover's pruning IS the
    triangle inequality on the landmark radii, so admitting a non-metric
    would silently prune away true neighbours rather than return them
    slowly. That argument is sound and it is kept. What it does NOT cover is
    every metric it was refusing.

    Minkowski at p >= 1 IS a true metric -- that is Minkowski's inequality,
    which is what the name refers to -- so an Lp ball cover at p >= 1 is
    sound, and p = 1 is Manhattan. Chebyshev is the p -> infinity limit and
    is a metric too. So those are admitted, exactly and exhaustively gated
    by `neighbors/checks/ball_cover_knn_check.mojo` against a host brute
    force with no tolerance.

    What stays refused is what genuinely fails the inequality:

      cosine        `1 - cos` is not a metric. Three unit vectors at 0, 60
                    and 120 degrees give 1.5 against 0.5 + 0.5. It also
                    fails identity of indiscernibles: d(x, 2x) = 0, so two
                    distinct points sit at distance zero and a landmark
                    radius stops bounding its ball.
      Lp at p < 1   |x|^p is not subadditive below p = 1. At p = 1/2 the
                    points (0,0), (1,0), (1,1) give a direct distance of 4
                    against a two-leg path of 1 + 1.
      sqeuclidean   d^2 is not a metric even though d is: three collinear
                    points at unit spacing give 4 against 1 + 1.

    cuML draws its line in the same place from the same side and lands
    somewhere narrower still (`VALID_METRICS["rbc"]` is
    `{euclidean, haversine, l2}`), which is their scope choice and not an
    argument about the inequality.
    """
    if not isinstance(metric, str):
        raise ValueError(
            f"mojolearn {cls_name}: metric must be a name, got {metric!r}"
        )
    key = metric.lower()
    if key in ("sqeuclidean",):
        raise ValueError(
            f"mojolearn {cls_name}: metric='sqeuclidean' is REFUSED on the "
            "random ball cover. The pruning here IS THE TRIANGLE "
            "INEQUALITY on the landmark radii "
            "(neighbors/impl/neighbors/ball_cover/common.mojo, DEVIATION "
            "565), and SQUARED Euclidean distance does not satisfy it even "
            "though Euclidean distance does: on three collinear points at "
            "unit spacing it gives 4 against 1 + 1. A cover built on it "
            "would prune away true neighbours silently. Ask for "
            "metric='euclidean' and square the returned distances, which "
            "is the same answer and is exact."
        )
    if key == "cosine":
        raise ValueError(
            f"mojolearn {cls_name}: metric='cosine' is REFUSED on the "
            "random ball cover. The pruning here IS THE TRIANGLE "
            "INEQUALITY on the landmark radii, and `1 - cos` does not "
            "satisfy it: three unit vectors at 0, 60 and 120 degrees give "
            "1.5 against 0.5 + 0.5. It also fails identity of "
            "indiscernibles, d(x, 2x) = 0, so a landmark radius stops "
            "bounding its ball. A cover built on it would prune away true "
            "neighbours silently rather than return them slowly. Use "
            "NearestNeighbors with algorithm='brute', which needs no "
            "inequality and honors every ported metric."
        )
    if key in _UNPORTED_METRICS:
        raise ValueError(
            f"mojolearn {cls_name}: metric={metric!r} is in cuML's "
            "VALID_METRICS['brute'] but is NOT PORTED "
            "(neighbors/NOT_IMPLEMENTED.tsv)"
        )
    if key not in _RBC_METRIC_TABLE:
        raise ValueError(
            f"mojolearn {cls_name}: unknown metric {metric!r}. The random "
            "ball cover admits: " + ", ".join(sorted(_RBC_METRIC_TABLE))
        )
    value = _RBC_METRIC_TABLE[key]
    arg = 2.0
    if value == _DIST_LP_UNEXPANDED:
        arg = float(p)
        if arg != arg or arg == float("inf"):
            raise ValueError(
                f"mojolearn {cls_name}: metric='minkowski' needs a finite "
                f"p >= 1, got {p!r}. p = infinity is metric='chebyshev', "
                "which IS admitted here."
            )
        if arg < 1.0:
            raise ValueError(
                f"mojolearn {cls_name}: metric='minkowski' with p={p!r} is "
                "REFUSED on the random ball cover. The pruning here IS THE "
                "TRIANGLE INEQUALITY on the landmark radii "
                "(neighbors/impl/neighbors/ball_cover/common.mojo, "
                "DEVIATION 564), and |x|^p is not subadditive below p = 1, "
                "so Lp at p < 1 is not a metric at all: at p = 1/2 the "
                "points (0,0), (1,0), (1,1) give a direct distance of 4 "
                "against a two-leg path of 1 + 1. p >= 1 is admitted and "
                "p = 1 is Manhattan. Use NearestNeighbors with "
                "algorithm='brute' for p < 1, which is exact brute force "
                "and needs no inequality."
            )
    return value, arg


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
                                neighbors/checks/select_radix_identical
                                .mojo). FAST runs k > 256 through the
                                ported RAFT radix select.
        query_tile    honored   a MEMORY number; the answer does not depend
                                on it (`check_knn_tiled_is_query_tile_
                                invariant`), only the workspace does
        metric        honored   every row of cuML's `_build_metric_type`
                                (nearest_neighbors.pyx:520-553) this tree
                                computes: 'euclidean'/'l2', 'sqeuclidean',
                                'l1'/'cityblock'/'manhattan'/'taxicab',
                                'chebyshev'/'linf', 'cosine',
                                'minkowski'/'lp'. THE REST of their
                                VALID_METRICS['brute'] set (canberra,
                                jensenshannon, correlation, inner_product,
                                haversine, braycurtis) is refused BY NAME.
                                UNTIL 2026-09-01 THIS ROW READ "refused,
                                anything but Euclidean"; cosine and
                                Minkowski are ported and the sentence is
                                deleted rather than annotated.
        p             honored   Minkowski's exponent, at any finite
                                positive normal value. Refused at p <= 0,
                                p = inf, NaN and subnormal p (DEVIATION
                                552 -- 1/p and the vendor flush policy).
                                Read ONLY by metric='minkowski'/'lp',
                                which is cuML's rule too
                                (nearest_neighbors.pyx:852-854 passes
                                self.p for every metric and every non-Lp
                                op discards it).
        algorithm     honored   'brute' / 'auto' (exact brute force, the
                                default) and 'rbc' (exact k-NN over a
                                random ball cover, an INDEX -- added
                                2026-09-01, DEVIATIONS 558 to 567).
                                'kd_tree' and 'ball_tree' stay refused and
                                it is an ENGINEERING refusal, not an
                                attribution one; see the note under
                                `algorithm` below.

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

    #: Subclasses override; the base class never weights.
    _weights_value = _WEIGHTS_UNIFORM

    def _dist_params(self):
        """`[metric, metric_arg, weights]`, the list
        `bindings/_mojolearn.mojo::_dist_triple` reads. ORDER MATTERS and
        is documented there."""
        value, arg = _resolve_metric(type(self).__name__, self.metric, self.p)
        return [value, arg, self._weights_value]

    def _check_refusals(self):
        """Every refusal is BY NAME with the reason, raised at fit so a
        caller learns before the index is held."""
        # Resolving the metric IS the metric check: it raises by name for
        # an unported row of cuML's table, for an unknown name, and for a
        # p that cannot be one arithmetic. One table, one place.
        if self.algorithm == "rbc" and type(self) is not NearestNeighbors:
            # REFUSED RATHER THAN ACCEPTED AND IGNORED, which is the whole
            # of `reached-but-inert`. The classifier and the regressor do
            # not call `kneighbors`; they call cuML's `knn_classify` /
            # `knn_regress`, which do their own brute-force search inside
            # one Mojo entry point (`neighbors/impl/knn/knn.mojo`). Taking
            # 'rbc' here would change nothing about what runs, and a
            # parameter that is honored in the signature and inert in the
            # kernel is worse than one that is refused.
            raise ValueError(
                f"mojolearn {type(self).__name__}: algorithm='rbc' is "
                "refused. The ball cover's k-NN query is wired into "
                "NearestNeighbors.kneighbors only; this class runs cuML's "
                "knn_classify / knn_regress, which perform their own "
                "brute-force search inside one call and have no seam to "
                "hand an index to. Accepting 'rbc' here would be accepting "
                "a parameter that changes nothing. Use "
                "NearestNeighbors(algorithm='rbc').kneighbors and vote on "
                "the result, or leave this at 'brute'."
            )
        if self.algorithm == "rbc":
            # The INDEXED arm resolves against the cover's own table, which
            # refuses two rows the brute-force table accepts. Resolving here
            # means a caller who asks for an index under cosine learns at
            # fit, not at the first query.
            _resolve_rbc_metric(type(self).__name__, self.metric, self.p)
        elif self.algorithm in ("brute", "auto"):
            # Resolving the metric IS the metric check: it raises by name
            # for an unported row of cuML's table, for an unknown name, and
            # for a p that cannot be one arithmetic. One table, one place.
            _resolve_metric(type(self).__name__, self.metric, self.p)
        else:
            raise ValueError(
                f"mojolearn NearestNeighbors: algorithm={self.algorithm!r} is "
                "refused. Three are accepted: 'brute' (exact brute force), "
                "'auto' (which means 'brute' here), and 'rbc' (exact k-NN "
                "over a RANDOM BALL COVER, an index). "
                "'kd_tree' and 'ball_tree' are refused, and it is an "
                "ENGINEERING refusal rather than an attribution one: a "
                "kd-tree query is a per-query stack walk with "
                "data-dependent branching and divergent memory access, "
                "which is the shape a GPU is worst at, and above roughly "
                "15 dimensions the pruning bound stops firing and it "
                "degenerates to a full scan, so it would lose to 'brute' "
                "on exactly the workloads it claims to help. If what you "
                "want is an index instead of a full scan, that is 'rbc': "
                "it is EXACT, not approximate, it has no per-query stack, "
                "and it is the same structure RadiusNeighbors uses "
                "(neighbors/impl/neighbors/ball_cover/knn.mojo)."
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

        if self.algorithm == "rbc":
            # THE INDEXED ARM. Same answer, computed by pruning instead of
            # by comparing every pair: `neighbors/impl/neighbors/ball_cover/
            # knn.mojo` proves each bound exact and
            # `neighbors/checks/ball_cover_knn_check.mojo` gates it against
            # a host brute force with no tolerance.
            mvalue, marg = _resolve_rbc_metric(
                type(self).__name__, self.metric, self.p
            )
            idx = self._index
            rind = np.empty((nq, k), dtype=np.int32)
            rdist = np.empty((nq, k), dtype=np.float32)
            self.n_candidate_distances_ = self._bind(
                "_mojolearn"
            ).rbc_knn_search(
                _addr_ro(idx), _addr_ro(q), _addr(rind), _addr(rdist),
                # ORDER MATCHES bindings/_mojolearn.mojo::
                # rbc_knn_search_binding. n_index, n_queries, n_features, k,
                # metric, metric_arg
                [idx.shape[0], nq, idx.shape[1], k, mvalue, marg],
            )
            # The index is built inside the call, so `used_query_tile_` has
            # no meaning on this arm and is set to None rather than left
            # holding a stale value from a previous brute-force call.
            self.used_query_tile_ = None
            if (rind < 0).any():
                raise RuntimeError(
                    "mojolearn NearestNeighbors(algorithm='rbc'): the query "
                    "returned an unfilled neighbour slot, which can only "
                    "happen when k exceeds the index size. The Mojo side "
                    "refuses that before launching, so this means the "
                    "arrays changed under the call."
                )
            if return_distance:
                return rdist, rind.astype(np.int64)
            return rind.astype(np.int64)

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
            # metric, metric_arg, weights -- see _dist_triple there.
            self._dist_params(),
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
    by construction (`neighbors/impl/selection/knn.mojo`, DEVIATION 542).

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY (measured row by row by
    `tools/e2u_matrix_fit.py`; the `NearestNeighbors` rows apply as well):

        n_neighbors   honored   k, as NearestNeighbors (refused above
                                n_samples_fit; under NUMERIC_IDENTICAL above
                                256)
        weights       honored   'uniform' (cuML's only arm, the ported
                                `class_probs_kernel`) and 'distance'
                                (DEVIATION 556, scikit-learn's semantics:
                                `w = 1/d`, and a row containing an exact
                                zero is REPLACED WHOLESALE by its
                                zero-mask so the exact matches get 1.0 and
                                every other neighbour in that row gets
                                0.0 -- _base.py:108-113). A CALLABLE is
                                refused: it would have to run inside a GPU
                                kernel. UNTIL 2026-09-01 THIS ROW READ
                                "refused, anything but 'uniform', cuML
                                refuses it too"; cuML not having a thing
                                stopped being a reason to refuse it.
                                NOTE: the weighted arm asks the search for
                                the ROOTED distance where the uniform arm
                                asks for the squared one, because a weight
                                reads the VALUE and a vote reads only the
                                ORDER (estimator.mojo policy 8).
        metric, p     honored   as NearestNeighbors
        algorithm     refused   anything but 'brute' / 'auto'. 'rbc' is
                                refused HERE though NearestNeighbors takes
                                it: this class calls cuML's knn_classify,
                                which searches inside its own entry point,
                                so 'rbc' would be accepted and inert
        y             honored   int labels, ANY values (negative, gaps);
                                1-D or 2-D (multi-output, one vote per
                                column, as cuML's `vector<int*> y`)

    `predict_proba` returns each query's vote fractions over `classes_`
    (cuML's `knn_class_proba`), one array, or a list of arrays for a 2-D
    `y`, exactly as cuML and scikit-learn shape it.

    Parameters
    ----------
    n_neighbors : int, default 5
    weights : {'uniform', 'distance'}, default 'uniform'
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

    @property
    def _weights_value(self):
        """`weights` as the value `bindings/_mojolearn.mojo::_dist_triple`
        reads. `_check_refusals` is what raises on a bad one; this
        property is only reached after it has run."""
        return _resolve_weights(type(self).__name__, self.weights)

    def _check_refusals(self):
        super()._check_refusals()
        _resolve_weights(type(self).__name__, self.weights)

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
            # metric, metric_arg, weights -- see _dist_triple there.
            self._dist_params(),
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
        weights       honored   'uniform' (cuML's `regress_avg_kernel`) and
                                'distance' (DEVIATION 556: scikit-learn's
                                `sum(y w) / sum(w)` with `w = 1/d` and the
                                same row-level zero rule as the
                                classifier). A callable is refused.
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

    @property
    def _weights_value(self):
        """`weights` as the value `bindings/_mojolearn.mojo::_dist_triple`
        reads. `_check_refusals` is what raises on a bad one; this
        property is only reached after it has run."""
        return _resolve_weights(type(self).__name__, self.weights)

    def _check_refusals(self):
        super()._check_refusals()
        _resolve_weights(type(self).__name__, self.weights)

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
            # metric, metric_arg, weights -- see _dist_triple there.
            self._dist_params(),
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
    (`neighbors/impl/neighbors/ball_cover/`), which DBSCAN has used for its
    eps neighbourhood since this library's first release. It returns the
    EXACT set -- its pruning is a triangle-inequality bound, not an
    approximation -- and `neighbors/checks/radius_check.mojo` asserts that
    against a host brute-force oracle per cell, not per total.

    **Metrics: every one that satisfies the triangle inequality, which is
    what the pruning rests on.** 'euclidean'/'l2', 'manhattan'/'l1'/
    'cityblock'/'taxicab', 'chebyshev'/'linf' and 'minkowski'/'lp' at
    p >= 1. Until 2026-09-01 this was Euclidean only; the reason given was
    correct and is kept, and it turned out to be narrower than its own
    argument, because Minkowski at p >= 1 IS a metric. 'cosine',
    'sqeuclidean' and Lp at p < 1 stay refused BY NAME with the inequality
    as the stated reason (`_resolve_rbc_metric`). The per-metric gate is
    `pixi run check-ball-cover-knn-identical`.

    **The distances are recomputed, not stored by the search.** The search
    kernel knows every distance at the moment it decides membership and
    throws them away; a separate pass walks the finished neighbour list and
    recomputes them. `neighbors/checks/radius_distances.mojo` carries the
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
        self._metric_value = None
        self._metric_arg = None

    def _check_refusals(self):
        """Every refusal is BY NAME with the reason, raised at fit.

        NOT `_METRIC_TABLE`, AND NOT THE OLD EUCLIDEAN-ONLY LIST EITHER.
        Until 2026-09-01 this admitted `euclidean`, `l2` and
        `minkowski` at p = 2 and refused everything else, with a correct
        reason attached: the pruning here IS the triangle inequality on the
        landmark radii, so admitting a non-metric returns a plausible,
        WRONG, silently pruned answer rather than a slow one. That argument
        is kept and it decides the width: it covers cosine and Lp at p < 1
        and `sqeuclidean`, and it does NOT cover Minkowski at p >= 1, which
        is a true metric (that is Minkowski's inequality). So Manhattan,
        Chebyshev and every Lp at p >= 1 are admitted now, and
        `neighbors/checks/ball_cover_knn_check.mojo` gates each of them
        against a host brute force per row with no tolerance.
        `_resolve_rbc_metric` holds the table and every refusal message.
        """
        self._metric_value, self._metric_arg = _resolve_rbc_metric(
            type(self).__name__, self.metric, self.p
        )
        if self.algorithm not in ("auto", "rbc"):
            raise ValueError(
                f"mojolearn RadiusNeighbors: algorithm={self.algorithm!r} is "
                "refused. The index here is cuVS's RANDOM BALL COVER, which "
                "is none of scikit-learn's three: it is not 'brute' (there "
                "is an index and it prunes), and it is neither 'ball_tree' "
                "nor 'kd_tree' (it is a one-level cover over sqrt(n) "
                "landmarks, not a tree). Naming any of them would describe "
                "the wrong algorithm, so 'auto' and 'rbc' -- the "
                "algorithm's own name -- are the two accepted. The "
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
            # radius, metric, metric_arg
            [idx.shape[0], nq, idx.shape[1], r,
             self._metric_value, self._metric_arg],
        )
        cols = np.empty(nnz, dtype=np.int32)
        dists = np.empty(nnz, dtype=np.float32)
        got = self._bind("_mojolearn").radius_neighbors_fill(
            _addr_ro(idx), _addr_ro(q), _addr(indptr), _addr(cols),
            _addr(dists),
            # n_index, n_queries, n_features, radius, nnz_capacity,
            # return_sqrt, metric, metric_arg. The metric MUST be the same
            # value both passes saw: the index is built inside each call, so
            # a metric that differed between them would count under one
            # cover and fill under another.
            [idx.shape[0], nq, idx.shape[1], r, nnz, 1,
             self._metric_value, self._metric_arg],
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
