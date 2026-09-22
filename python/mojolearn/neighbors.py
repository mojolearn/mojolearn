# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Brute-force exact k-nearest-neighbours on the GPU, and the classifier and
regressor built on it.

`NearestNeighbors` references cuVS's brute force; `KNeighborsClassifier` and
`KNeighborsRegressor` reference cuML's `kneighbors_classifier.pyx` /
`kneighbors_regressor.pyx` over `ML::knn_classify` / `ML::knn_regress`
(`neighbors/impl/knn/knn.mojo`, `neighbors/impl/selection/knn.mojo`).
"""

from . import _portable_math as math

from . import _mojolearn, _serialize
from ._array import Array
from ._buffer import addr, addr_ro, as_f32_c, as_i64_c, empty
from ._labels import sorted_classes
from ._mode import NumericModeMixin
from .linear_model import (
    _check_saved_by, _dtype_name, _is_integer_labels, _restore_mode,
    _saved_mode, _shape_of,
)

_DEFAULT_QUERY_TILE = 0  # Ask the compiled planner for its measured default.

#: The model file format (the knn host inference lane, 2026-09-14). `save`
#: writes exactly what `kneighbors` / `predict` read, raw bytes and exact
#: dtypes, through `_serialize.write_npz`, so equal models give equal file
#: hashes on every machine; `load` refuses a cast (`_serialize.exact`). The
#: index is the fitted `<f4` `(n_samples_fit_, n_features_in_)` block, the
#: classifier's labels the `<i4` `(n_outputs, n_samples_fit_)` column block
#: the binding takes (policy 6), the regressor's the same in `<f4`; `p`
#: travels as `<f8` and the three names (`metric`, `algorithm`, `weights`)
#: as text. `mojolearn.host_model(path)` predicts from the file on a CPU
#: with no GPU through `bindings/_mojolearn_core_host.mojo`.
_KNN_FORMAT = "mojolearn-knn-1"
_KNN_META_FIELDS = 6  # n_features_in_, n_samples_fit_, n_neighbors, query_tile, outputs_2d, n_outputs
#: RadiusNeighbors' model file (the neighbors and density inference lane,
#: 2026-09-15): the `<f4` index as fitted, `radius`, `p` as `<f8`, `metric`
#: and `algorithm` as text, `meta` `<i8` [n_features_in_, n_samples_fit_].
#: What `radius_neighbors` reads and nothing else; `mojolearn.host_model`
#: queries it on a CPU through the core host binding's
#: radius_neighbors_count and radius_neighbors_fill.
_RADIUS_FORMAT = "mojolearn-radius-1"


def _knn_arrays(est):
    """The members every k-NN `save` writes; the subclasses add theirs."""
    if est._index is None:
        raise RuntimeError("this estimator is not fitted yet")
    weights = getattr(est, "weights", "uniform")
    weights = "uniform" if weights is None else weights
    if not isinstance(weights, str):
        # `fit` refuses a callable by name before the index is held, so
        # this is unreachable on a fitted estimator; a file never carries
        # a function.
        raise ValueError(
            f"mojolearn {type(est).__name__}: weights={weights!r} cannot be saved"
        )
    y_cols = getattr(est, "_y_cols", None)
    n_out = 1 if y_cols is None else int(y_cols.shape[0])
    return {
        "format": _KNN_FORMAT,
        "estimator": type(est).__name__,
        "numeric_mode": _saved_mode(est),
        "metric": str(est.metric),
        "algorithm": str(est.algorithm),
        "weights": weights,
        "p": Array.from_list([float(est.p)], "<f8"),
        "index": est._index,
        "meta": Array.from_list(
            [int(est.n_features_in_), int(est.n_samples_fit_),
             int(est.n_neighbors), int(est.query_tile),
             1 if getattr(est, "outputs_2d_", False) else 0, n_out],
            "<i8",
        ),
    }


def _load_knn(cls, path):
    """The index and the parameters every k-NN `load` reads. Returns the
    estimator (fitted as far as `NearestNeighbors.fit` fits it), the
    decoded members, and `(outputs_2d, n_outputs)` for the subclasses."""
    arrays = _serialize.read_npz(path, _KNN_FORMAT)
    _check_saved_by(arrays, path, cls)
    meta = _serialize.exact(arrays, "meta", "<i8")
    if meta.size != _KNN_META_FIELDS:
        raise ValueError(
            f"mojolearn: {path!r} meta holds {meta.size} fields, "
            f"{_KNN_META_FIELDS} are needed"
        )
    nf, ns, k, qt, o2d, n_out = (int(meta[i]) for i in range(_KNN_META_FIELDS))
    p = _serialize.exact(arrays, "p", "<f8")
    if p.size != 1:
        raise ValueError(f"mojolearn: {path!r} p holds {p.size} values, 1 is needed")
    kwargs = dict(
        n_neighbors=k, query_tile=qt,
        metric=_serialize.scalar_str(arrays, "metric"),
        algorithm=_serialize.scalar_str(arrays, "algorithm"),
        p=float(p[0]),
    )
    if cls._HAS_WEIGHTS:
        kwargs["weights"] = _serialize.scalar_str(arrays, "weights")
    obj = cls(**kwargs)
    _restore_mode(obj, arrays)
    # The refusals `fit` raises, raised here too, so a file naming a
    # metric or an algorithm this class refuses is refused by name at load.
    obj._check_refusals()
    index = _serialize.exact(arrays, "index", "<f4")
    if index.ndim != 2 or index.shape[0] != ns or index.shape[1] != nf:
        raise ValueError(
            f"mojolearn: {path!r} index has shape {tuple(index.shape)}, meta says ({ns}, {nf})"
        )
    if ns < 1 or nf < 1:
        raise ValueError(f"mojolearn: {path!r} holds an empty index")
    obj._index = index
    obj.n_samples_fit_ = ns
    obj.n_features_in_ = nf
    obj.used_query_tile_ = None
    return obj, arrays, (bool(o2d), n_out)


def _load_y_cols(arrays, path, dtype, ns, n_out):
    y_cols = _serialize.exact(arrays, "y_cols", dtype)
    if y_cols.ndim != 2 or y_cols.shape[0] != n_out or y_cols.shape[1] != ns:
        raise ValueError(
            f"mojolearn: {path!r} y_cols has shape {tuple(y_cols.shape)}, "
            f"meta says ({n_out}, {ns})"
        )
    return y_cols

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
#: the metric's name (`neighbors/impl/ball_cover/common.mojo`,
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
#:                       (DEVIATION 2 in `ball_cover.mojo`).
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
#: metric is UNIMPLEMENTED rather than unknown.
_UNSUPPORTED_METRICS = (
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
            f"mojolearn {cls_name}: weights=<callable> is NOT IMPLEMENTED. "
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


def _refuse_inert_p(cls_name, metric, p):
    """A `p` the resolved op never reads is refused by name.

    Only metric='minkowski'/'lp' reads `p`; cuML passes `self.p` for every
    metric (`nearest_neighbors.pyx:852-854`) and every other op discards it.
    Accepting `metric='euclidean', p=3` and running Euclidean is a knob that
    changes nothing, so any value but the constructor default (2) under
    another metric raises (the claim-surface census, 2026-09-14; the same
    rule as GradientBoosting's `bagging_temperature` and `subsample`).
    """
    try:
        is_default = not isinstance(p, bool) and float(p) == 2.0
    except (TypeError, ValueError):
        is_default = False
    if not is_default:
        raise ValueError(
            f"mojolearn {cls_name}: p is read only by metric='minkowski' "
            f"(or 'lp'); metric={metric!r} never reads it, so p={p!r} would "
            "change nothing. Leave p at its default (2) or ask for "
            "metric='minkowski'."
        )


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
            "A callable metric is not implemented: it would have to run inside a "
            "GPU kernel."
        )
    key = metric.lower()
    if key in _METRIC_TABLE:
        value = _METRIC_TABLE[key]
    elif key in _UNSUPPORTED_METRICS:
        raise ValueError(
            f"mojolearn {cls_name}: metric={metric!r} is in cuML's "
            "VALID_METRICS['brute'] but is NOT IMPLEMENTED "
            "(neighbors/NOT_IMPLEMENTED.tsv). Implemented: "
            + ", ".join(sorted(_METRIC_TABLE))
        )
    else:
        raise ValueError(
            f"mojolearn {cls_name}: unknown metric {metric!r}. Implemented: "
            + ", ".join(sorted(_METRIC_TABLE))
        )

    if value != _DIST_LP_UNEXPANDED:
        _refuse_inert_p(cls_name, metric, p)
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
            "(neighbors/impl/ball_cover/common.mojo, DEVIATION "
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
            "inequality and honors every implemented metric."
        )
    if key in _UNSUPPORTED_METRICS:
        raise ValueError(
            f"mojolearn {cls_name}: metric={metric!r} is in cuML's "
            "VALID_METRICS['brute'] but is NOT IMPLEMENTED "
            "(neighbors/NOT_IMPLEMENTED.tsv)"
        )
    if key not in _RBC_METRIC_TABLE:
        raise ValueError(
            f"mojolearn {cls_name}: unknown metric {metric!r}. The random "
            "ball cover admits: " + ", ".join(sorted(_RBC_METRIC_TABLE))
        )
    value = _RBC_METRIC_TABLE[key]
    if value != _DIST_LP_UNEXPANDED:
        _refuse_inert_p(cls_name, metric, p)
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
                "(neighbors/impl/ball_cover/common.mojo, "
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
    """Exact k-NN by brute force. Reference: cuVS's fused L2 kernel.

    EXACT, not approximate. There is no index to build and no recall to trade
    away: every query is compared against every index point. That is a
    deliberate scope choice rather than a missing feature -- an approximate
    index is a different algorithm with a different contract, and this library
    does not ship one.

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY (measured row by row by
    `tools/e2u_matrix_fit.py`):

        n_neighbors   honored   k. Refused above n_samples_fit (the
                                reference's short-index fill is not implemented:
                                knn_brute_force.mojo) and, UNDER
                                IDENTICAL/DETERMINISTIC, above 1024 -- the
                                pinned selector's strided rank pass bounds
                                its shared staging and quadratic work (
                                neighbors/checks/select_radix_identical
                                .mojo). FAST runs k > 256 through the
                                radix select (reference: RAFT).
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
                                Minkowski are implemented and the sentence is
                                deleted rather than annotated.
        p             honored   Minkowski's exponent, at any finite
                                positive normal value. Refused at p <= 0,
                                p = inf, NaN and subnormal p (DEVIATION
                                552 -- 1/p and the vendor flush policy).
                                Read ONLY by metric='minkowski'/'lp'
                                (nearest_neighbors.pyx:852-854 passes
                                self.p for every metric and every non-Lp
                                op discards it). Under any other metric a
                                p other than the default 2 is REFUSED by
                                name rather than accepted and ignored.
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
    query_tile : int, default 0 (automatic)
        Queries processed per pass. Zero uses the compiled planner: NVIDIA
        IDENTICAL starts at 512, while other modes and columns start at 256.
        The planner applies its workspace limit and clamps to the query count.
        Set an explicit positive tile to request a fixed starting size.
        ``used_query_tile_`` reports the batch size that actually ran.

    Attributes
    ----------
    used_query_tile_ : int
        Set by `kneighbors` after automatic planning, the workspace cap and
        the query-count clamp.
    """

    #: This family's binding, for `NumericModeMixin._bind`.
    _BINDING = "_mojolearn"
    #: Whether `load` hands the file's `weights` to the constructor.
    _HAS_WEIGHTS = False

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

    def save(self, path):
        """Write the fitted index to `path` as an npz: the `<f4` index as
        fitted, `n_neighbors`, `query_tile`, `metric`, `p` and `algorithm`
        (the knn host inference lane, 2026-09-14). What `kneighbors` reads
        and nothing else; `mojolearn.host_model(path)` searches it on a CPU
        with no GPU (brute arm, euclidean and sqeuclidean)."""
        return _serialize.write_npz(path, _knn_arrays(self))

    @classmethod
    def load(cls, path):
        """Load a model saved by `save`. The result answers `kneighbors`."""
        obj, _, _ = _load_knn(cls, path)
        return obj

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
        # an unimplemented row of cuML's table, for an unknown name, and for a
        # p that cannot be one arithmetic. One table, one place.
        # By class, not by `type(self) is NearestNeighbors`: a subclass of
        # NearestNeighbors alone (the host class `mojolearn.host_model`
        # returns, 2026-09-15) still runs `kneighbors` and keeps the cover.
        if self.algorithm == "rbc" and isinstance(self, (KNeighborsClassifier, KNeighborsRegressor)):
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
            # for an unimplemented row of cuML's table, for an unknown name, and
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
                "(neighbors/impl/ball_cover/knn.mojo)."
            )

    def _resident_index_handle(self, binding, idx):
        """The handle of the device-resident copy of `idx` (DEVIATION 2921),
        prepared on the first call and reused while the index array keeps
        its address and shape; None where the loaded binding has no
        `knn_index_prepare` (the CPU host binding, a CPU-only install, a
        host subclass), in which case `kneighbors` takes the per-call
        upload. Never raises for a missing door: `_HostBinding` refuses an
        absent name with ImportError and a plain module with
        AttributeError, and both mean "no residency here"."""
        try:
            prepare = binding.knn_index_prepare
        except (ImportError, AttributeError):
            return None
        key = (addr_ro(idx, name="idx"), tuple(idx.shape))
        cached = getattr(self, "_resident", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        if cached is not None:
            self._release_resident_index()
        handle = int(prepare(key[0], [int(idx.shape[0]), int(idx.shape[1])]))
        self._resident = (key, handle)
        return handle

    @staticmethod
    def _resident_door(binding, name):
        """The binding's entry `name`, or None where the loaded binding
        has no such door (`_resident_index_handle`'s rule: an absent name
        is ImportError on a host binding, AttributeError on a module)."""
        try:
            return getattr(binding, name)
        except (ImportError, AttributeError):
            return None

    def _release_resident_index(self):
        """Drop the device copy, if one is held. Quiet on a binding that
        cannot be reached any more (interpreter shutdown) and on a handle
        already released (a copied instance)."""
        cached = getattr(self, "_resident", None)
        if cached is None:
            return
        self._resident = None
        try:
            self._bind("_mojolearn").knn_index_release(cached[1])
        except Exception:  # noqa: BLE001
            pass

    def __del__(self):
        try:
            self._release_resident_index()
        except Exception:  # noqa: BLE001
            pass

    def __getstate__(self):
        """A pickle or a deepcopy carries no device handle: the integer is
        meaningful only in the process and registry that minted it, and an
        unpickled instance holding it could release ANOTHER model's live
        index. The copy uploads its own index at its first call."""
        state = self.__dict__.copy()
        state.pop("_resident", None)
        return state

    def fit(self, X, y=None):
        """Store the index. There is no index structure to build.

        `y` is accepted and ignored, for scikit-learn call-shape
        compatibility.
        """
        self._check_refusals()
        # A refit drops the device copy of the previous index HERE, not at
        # the next call: the copy is keyed on the array's address and shape,
        # and a new array of the same shape can land at a freed address, so
        # a key match after a refit would serve the old bytes.
        self._release_resident_index()
        idx, _ = as_f32_c(X, ndim=2, name="X")
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

        THE INDEX STAYS ON THE DEVICE (DEVIATION 2921, lane/infer-speed-
        classical, 2026-09-17). On a GPU binding the brute-force arm uploads
        the fitted index ONCE, at the first call after `fit`, and every
        later call searches the device copy (`neighbors/resident_index.
        mojo`); the search itself is `knn_search`'s body after its upload,
        so the bits are `knn_search`'s. The copy is keyed on the index
        array's address and shape: a refit replaces the array and the next
        call uploads again, and the old copy is released then and when the
        instance is collected. An in-place write into the array `fit` was
        given, after the first call, is NOT seen by later calls, as it is
        not by cuML, whose `fit` copies the index to the device. The CPU
        host binding has no such entry and reads the caller's memory on
        every call.
        """
        if self._index is None:
            raise ValueError("mojolearn: call fit before kneighbors")
        k = self.n_neighbors if n_neighbors is None else n_neighbors
        q, _ = as_f32_c(X, ndim=2, name="X")
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
            # by comparing every pair: `neighbors/impl/ball_cover/
            # knn.mojo` proves each bound exact and
            # `neighbors/checks/ball_cover_knn_check.mojo` gates it against
            # a host brute force with no tolerance.
            mvalue, marg = _resolve_rbc_metric(
                type(self).__name__, self.metric, self.p
            )
            idx = self._index
            rind = empty((nq, k), "<i4")
            rdist = empty((nq, k), "<f4")
            self.n_candidate_distances_ = self._bind(
                "_mojolearn"
            ).rbc_knn_search(
                addr_ro(idx, name="idx"), addr_ro(q, name="q"), addr(rind, name="rind"), addr(rdist, name="rdist"),
                # ORDER MATCHES bindings/_mojolearn.mojo::
                # rbc_knn_search_binding. n_index, n_queries, n_features, k,
                # metric, metric_arg
                [idx.shape[0], nq, idx.shape[1], k, mvalue, marg],
            )
            # The index is built inside the call, so `used_query_tile_` has
            # no meaning on this arm and is set to None rather than left
            # holding a stale value from a previous brute-force call.
            self.used_query_tile_ = None
            if rind.min() < 0:
                raise RuntimeError(
                    "mojolearn NearestNeighbors(algorithm='rbc'): the query "
                    "returned an unfilled neighbour slot, which can only "
                    "happen when k exceeds the index size. The Mojo side "
                    "refuses that before launching, so this means the "
                    "arrays changed under the call."
                )
            if return_distance:
                return rdist, rind.astype("<i8")
            return rind.astype("<i8")

        dist = empty((nq, k), "<f4")
        ind = empty((nq, k), "<u4")

        # Every array named here stays in a local for the whole call. That is
        # the contract `_buffer` documents and the reason it is spelled out.
        idx = self._index
        binding = self._bind("_mojolearn")
        handle = self._resident_index_handle(binding, idx)
        # ORDER MATCHES bindings/_mojolearn.mojo::knn_search_binding and
        # ::knn_search_resident_binding.
        # n_index, n_queries, n_features, k, return_sqrt, query_tile
        params = [idx.shape[0], nq, idx.shape[1], k, 1, self.query_tile]
        if handle is not None:
            self.used_query_tile_ = binding.knn_search_resident(
                handle,
                addr_ro(idx, name="idx"),
                addr_ro(q, name="q"),
                addr(dist, name="dist"),
                addr(ind, name="ind"),
                params,
                # metric, metric_arg, weights -- see _dist_triple there.
                self._dist_params(),
            )
        else:
            self.used_query_tile_ = binding.knn_search(
                addr_ro(idx, name="idx"),
                addr_ro(q, name="q"),
                addr(dist, name="dist"),
                addr(ind, name="ind"),
                params,
                self._dist_params(),
            )

        if return_distance:
            return dist, ind.astype("<i8")
        return ind.astype("<i8")


class KNeighborsClassifier(NearestNeighbors):
    """k-NN classification by UNWEIGHTED majority vote over the `n_neighbors`
    nearest index points. Reference: cuML's `KNeighborsClassifier`.

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
        weights       honored   'uniform' (cuML's only arm, the implemented
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
                                it: this class calls knn_classify (as cuML does),
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
    classes_ : list, or list of lists for a 2-D `y`
        The sorted unique labels per output, as Python ints (the pyx's
        `cp.unique`; `_labels.sorted_classes`, DEVIATION 2340, a list
        rather than an ndarray). The
        Mojo side recomputes the same set with the implemented `getUniquelabels`
        and `predict` asserts the two agree.
    outputs_2d_ : bool
    """

    #: scikit-learn's estimator kind: `cross_val_score` stratifies a
    #: classifier's default folds, as scikit-learn's does.
    _estimator_type = "classifier"

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

    _HAS_WEIGHTS = True

    @property
    def _weights_value(self):
        """`weights` as the value `bindings/_mojolearn.mojo::_dist_triple`
        reads. `_check_refusals` is what raises on a bad one; this
        property is only reached after it has run."""
        return _resolve_weights(type(self).__name__, self.weights)

    def _check_refusals(self):
        super()._check_refusals()
        _resolve_weights(type(self).__name__, self.weights)

    def save(self, path):
        """Write the fitted model to `path` as an npz: the index and
        parameters `NearestNeighbors.save` writes, `weights`, the `<i4`
        label columns the binding takes (`_y_cols`, policy 6), `classes`
        (`<i8`, every output's sorted classes concatenated) and
        `class_counts` (`<i8`, one per output) (the knn host inference lane,
        2026-09-14). `mojolearn.host_model(path)` predicts from it on a CPU
        with no GPU."""
        if self._y_cols is None:
            raise RuntimeError("this estimator is not fitted yet")
        arrays = _knn_arrays(self)
        arrays["y_cols"] = self._y_cols
        arrays["classes"] = Array.from_list(
            [int(c) for cl in self._classes_list for c in cl], "<i8"
        )
        arrays["class_counts"] = Array.from_list(
            [len(cl) for cl in self._classes_list], "<i8"
        )
        return _serialize.write_npz(path, arrays)

    @classmethod
    def load(cls, path):
        """Load a model saved by `save`. The result predicts. `classes_` is
        rebuilt from the label columns exactly as `fit` builds it and must
        equal the file's `classes` member, or the file is refused."""
        obj, arrays, (o2d, n_out) = _load_knn(cls, path)
        y_cols = _load_y_cols(arrays, path, "<i4", obj.n_samples_fit_, n_out)
        classes = _serialize.exact(arrays, "classes", "<i8").tolist()
        counts = _serialize.exact(arrays, "class_counts", "<i8").tolist()
        if len(counts) != n_out or sum(counts) != len(classes):
            raise ValueError(f"mojolearn: {path!r} classes and class_counts disagree with n_outputs")
        rebuilt = [sorted_classes(y_cols[i].tolist())[0] for i in range(n_out)]
        off = 0
        for i, count in enumerate(counts):
            saved = [int(c) for c in classes[off:off + count]]
            off += count
            if saved != rebuilt[i]:
                raise ValueError(
                    f"mojolearn: {path!r} classes for output {i} are not the sorted "
                    "unique labels of its y_cols column"
                )
        obj._y_cols = y_cols
        obj._classes_list = rebuilt
        obj.outputs_2d_ = o2d
        return obj

    def fit(self, X, y):
        """Store the index and the labels. `y` is int, 1-D or 2-D."""
        super().fit(X)
        shape = _shape_of(y)
        if len(shape) not in (1, 2):
            raise ValueError(
                f"mojolearn: y must be 1-D or 2-D, got {len(shape)}-D"
            )
        if shape[0] != self.n_samples_fit_:
            raise ValueError(
                f"mojolearn: y has {shape[0]} rows, X has "
                f"{self.n_samples_fit_}"
            )
        if not _is_integer_labels(y):
            raise ValueError(
                f"mojolearn: y must be integer class labels, got "
                f"{_dtype_name(y)}; cuML converts to int32 "
                "(check_dtype=np.int32) and so does this surface -- cast "
                "your labels"
            )
        ya, _ = as_i64_c(y, ndim=len(shape), name="y")
        if ya.min() < -(1 << 31) or ya.max() > (1 << 31) - 1:
            raise ValueError("mojolearn: y does not fit int32")
        n = shape[0]
        n_out = 1 if len(shape) == 1 else shape[1]
        # POLICY 6 (neighbors/estimator.mojo): the binding takes `n_outputs`
        # CONTIGUOUS columns -- cuML's order='F' `y`. One transpose at fit,
        # through the O(rows * n_outputs) label loop the contract permits
        # (DEVIATION 2374), which also narrows int64 -> int32 exactly after
        # the range check above.
        rows = ya.tolist() if n_out > 1 else None
        if n_out == 1:
            cols = [ya.reshape((n,)).tolist()]
        else:
            cols = [[row[j] for row in rows] for j in range(n_out)]
        self._y_cols = Array.from_list(cols, "<i4")
        self.outputs_2d_ = len(shape) == 2 and shape[1] != 1
        # `np.unique` per column, under the package-wide classes_ ORDER
        # RULE (`_labels.sorted_classes`, DEVIATION 2340): a Python list
        # per output; int labels, so a sort by value.
        self._classes_list = [sorted_classes(col)[0] for col in cols]
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
        q, _ = as_f32_c(X, ndim=2, name="X")
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
        n_classes = [len(c) for c in self._classes_list]
        # The native contract writes exactly one of these two outputs.  Keep
        # the unselected address valid without materializing its full public
        # shape: at million-row scale the unused probability matrix can be
        # tens or hundreds of MiB, and predict_proba likewise has no use for
        # a labels matrix.  Both GPU and host bindings promise not to read or
        # write the unselected pointer (their one-element sentinel contract).
        labels = (empty((1,), "<i4") if want_proba else
                  empty((nq, n_out), "<i4"))
        proba = (empty((nq * sum(n_classes),), "<f4") if want_proba else
                 empty((1,), "<f4"))
        uniq = empty((sum(n_classes),), "<i4")
        idx = self._index
        y_cols = self._y_cols
        binding = self._bind("_mojolearn")
        # THE INDEX STAYS ON THE DEVICE (DEVIATION 3002, the classifier's
        # door of DEVIATION 2921): the first call uploads it, every later
        # call classifies against the device copy through
        # `knn_classify_resident`; the search and the vote are the same
        # statements over the same bytes. A binding without that entry
        # takes the per-call path.
        resident = self._resident_door(binding, "knn_classify_resident")
        handle = self._resident_index_handle(binding, idx) if resident is not None else None
        # ORDER MATCHES bindings/_mojolearn.mojo::knn_classify_binding and
        # ::knn_classify_resident_binding.
        # n_index, n_queries, n_features, k, query_tile, n_outputs,
        # want_proba, then n_classes per output
        params = [idx.shape[0], nq, idx.shape[1], k, self.query_tile, n_out,
                  1 if want_proba else 0] + n_classes
        if handle is not None:
            # The handle rides at the front of `params`: a binding takes at
            # most eight arguments and the classifier uses all eight.
            self.used_query_tile_ = resident(
                addr_ro(idx, name="idx"),
                addr_ro(q, name="q"),
                addr_ro(y_cols, name="y_cols"),
                addr(labels, name="labels"),
                addr(proba, name="proba"),
                addr(uniq, name="uniq"),
                [handle] + params,
                # metric, metric_arg, weights -- see _dist_triple there.
                self._dist_params(),
            )
        else:
            self.used_query_tile_ = binding.knn_classify(
                addr_ro(idx, name="idx"),
                addr_ro(q, name="q"),
                addr_ro(y_cols, name="y_cols"),
                addr(labels, name="labels"),
                addr(proba, name="proba"),
                addr(uniq, name="uniq"),
                params,
                self._dist_params(),
            )
        # POLICY 7: the implementation's class set against ours, made visible. `uniq`
        # is split at the running class counts (the old np.cumsum /
        # np.split), in Python over O(classes) ints.
        flat = uniq.tolist()
        off = 0
        for i, (count, b) in enumerate(zip(n_classes, self._classes_list)):
            a = flat[off:off + count]
            off += count
            if a != b:
                raise RuntimeError(
                    f"mojolearn: output {i}: the implemented getUniquelabels found "
                    f"classes {a[:8]}..., the host sort found "
                    f"{b[:8]}...; the two class sets disagree"
                )
        return labels, proba, n_classes

    def predict(self, X):
        """The voted class per query: shape `(n_queries,)`, or
        `(n_queries, n_outputs)` for a 2-D `y`. Original label values."""
        labels, _, _ = self._predict(X, want_proba=False)
        if self.outputs_2d_:
            return labels.astype("<i8")
        # (nq, 1) -> (nq,) is a C-order reshape; int32 -> int64 is exact.
        return labels.reshape((labels.shape[0],)).astype("<i8")

    def predict_proba(self, X):
        """Vote fractions per class, columns in `classes_` order; a list of
        arrays for a 2-D `y`."""
        labels, proba, n_classes = self._predict(X, want_proba=True)
        # `labels` is the one-element sentinel in this arm.  The selected
        # flat probability buffer records the query count without retaining
        # an otherwise-unused labels matrix.
        nq = proba.shape[0] // sum(n_classes)
        out = []
        off = 0
        for n in n_classes:
            # A slice of an Array COPIES (the _array contract).
            out.append(proba[off:off + nq * n].reshape((nq, n)))
            off += nq * n
        if self.outputs_2d_:
            return out
        return out[0]


class KNeighborsRegressor(NearestNeighbors):
    """k-NN regression by the UNWEIGHTED mean of the `n_neighbors` nearest
    targets. Reference: cuML's `KNeighborsRegressor` (`ML::knn_regress`,
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

    #: scikit-learn's estimator kind: `cross_val_score` stratifies a
    #: classifier's default folds, as scikit-learn's does.
    _estimator_type = "regressor"

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

    _HAS_WEIGHTS = True

    @property
    def _weights_value(self):
        """`weights` as the value `bindings/_mojolearn.mojo::_dist_triple`
        reads. `_check_refusals` is what raises on a bad one; this
        property is only reached after it has run."""
        return _resolve_weights(type(self).__name__, self.weights)

    def _check_refusals(self):
        super()._check_refusals()
        _resolve_weights(type(self).__name__, self.weights)

    def save(self, path):
        """Write the fitted model to `path` as an npz: the index and
        parameters `NearestNeighbors.save` writes, `weights` and the `<f4`
        target columns the binding takes (`_y_cols`, policy 6) (the knn host
        inference lane, 2026-09-14). `mojolearn.host_model(path)` predicts
        from it on a CPU with no GPU."""
        if self._y_cols is None:
            raise RuntimeError("this estimator is not fitted yet")
        arrays = _knn_arrays(self)
        arrays["y_cols"] = self._y_cols
        return _serialize.write_npz(path, arrays)

    @classmethod
    def load(cls, path):
        """Load a model saved by `save`. The result predicts."""
        obj, arrays, (o2d, n_out) = _load_knn(cls, path)
        obj._y_cols = _load_y_cols(arrays, path, "<f4", obj.n_samples_fit_, n_out)
        obj.outputs_2d_ = o2d
        return obj

    def fit(self, X, y):
        """Store the index and the targets. `y` is float, 1-D or 2-D."""
        super().fit(X)
        shape = _shape_of(y)
        if len(shape) not in (1, 2):
            raise ValueError(
                f"mojolearn: y must be 1-D or 2-D, got {len(shape)}-D"
            )
        if shape[0] != self.n_samples_fit_:
            raise ValueError(
                f"mojolearn: y has {shape[0]} rows, X has "
                f"{self.n_samples_fit_}"
            )
        ya, _ = as_f32_c(y, ndim=len(shape), name="y")
        n = shape[0]
        n_out = 1 if len(shape) == 1 else shape[1]
        if n_out == 1:
            # One column IS one contiguous row of the transposed layout: a
            # reshape, no copy (as `ascontiguousarray(y2.T)` was for 1-D y).
            self._y_cols = ya.reshape((1, n))
        else:
            # The transpose is a Python loop over O(rows * n_outputs)
            # targets (DEVIATION 2374); the values are already float32, so
            # `from_list` reproduces them exactly.
            rows = ya.tolist()
            self._y_cols = Array.from_list(
                [[row[j] for row in rows] for j in range(n_out)], "<f4"
            )
        self.outputs_2d_ = len(shape) == 2 and shape[1] != 1
        return self

    def predict(self, X):
        """The mean target per query: `(n_queries,)`, or
        `(n_queries, n_outputs)` for a 2-D `y`. float32."""
        if self._index is None or self._y_cols is None:
            raise ValueError("mojolearn: call fit before predict")
        q, _ = as_f32_c(X, ndim=2, name="X")
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
        out = empty((nq, n_out), "<f4")
        idx = self._index
        y_cols = self._y_cols
        binding = self._bind("_mojolearn")
        # DEVIATION 3002: the regressor's resident door; see the classifier.
        resident = self._resident_door(binding, "knn_regress_resident")
        handle = self._resident_index_handle(binding, idx) if resident is not None else None
        # ORDER MATCHES bindings/_mojolearn.mojo::knn_regress_binding and
        # ::knn_regress_resident_binding.
        # n_index, n_queries, n_features, k, query_tile, n_outputs
        params = [idx.shape[0], nq, idx.shape[1], k, self.query_tile, n_out]
        if handle is not None:
            self.used_query_tile_ = resident(
                handle,
                addr_ro(idx, name="idx"),
                addr_ro(q, name="q"),
                addr_ro(y_cols, name="y_cols"),
                addr(out, name="out"),
                params,
                # metric, metric_arg, weights -- see _dist_triple there.
                self._dist_params(),
            )
        else:
            self.used_query_tile_ = binding.knn_regress(
                addr_ro(idx, name="idx"),
                addr_ro(q, name="q"),
                addr_ro(y_cols, name="y_cols"),
                addr(out, name="out"),
                params,
                self._dist_params(),
            )
        if self.outputs_2d_:
            return out
        return out.reshape((nq,))


class RadiusNeighbors(NumericModeMixin):
    """Every neighbour inside a radius, over the random ball cover.

    scikit-learn's `RadiusNeighborsMixin.radius_neighbors` shape: a ragged
    `(distances, indices)` pair, one variable-length array per query row.

    **This is not brute force and it is not one of scikit-learn's trees.**
    The index is cuVS's random ball cover
    (`neighbors/impl/ball_cover/`), which DBSCAN has used for its
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
        if not math.isfinite(self.radius) or self.radius <= 0:
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
        idx, _ = as_f32_c(X, ndim=2, name="X")
        self._index = idx
        self.n_samples_fit_ = idx.shape[0]
        self.n_features_in_ = idx.shape[1]
        return self

    def save(self, path):
        """Write the fitted index to `path` as an npz (`_RADIUS_FORMAT`): the
        `<f4` index as fitted, `radius`, `metric`, `p` and `algorithm`.
        `mojolearn.host_model(path)` answers `radius_neighbors` from it on a
        CPU with no GPU."""
        if self._index is None:
            raise RuntimeError("this estimator is not fitted yet")
        return _serialize.write_npz(path, {
            "format": _RADIUS_FORMAT,
            "estimator": type(self).__name__,
            "numeric_mode": _saved_mode(self),
            "metric": str(self.metric),
            "algorithm": str(self.algorithm),
            "radius": Array.from_list([float(self.radius)], "<f8"),
            "p": Array.from_list([float(self.p)], "<f8"),
            "index": self._index,
            "meta": Array.from_list([int(self.n_features_in_), int(self.n_samples_fit_)], "<i8"),
        })

    @classmethod
    def load(cls, path):
        """Load a model saved by `save`. The refusals `fit` raises are raised
        here too; every array is read at its saved dtype and never cast."""
        arrays = _serialize.read_npz(path, _RADIUS_FORMAT)
        _check_saved_by(arrays, path, cls)
        meta = _serialize.exact(arrays, "meta", "<i8")
        if meta.size != 2:
            raise ValueError(f"mojolearn: {path!r} meta holds {meta.size} fields, 2 are needed")
        radius = _serialize.exact(arrays, "radius", "<f8")
        p = _serialize.exact(arrays, "p", "<f8")
        if radius.size != 1 or p.size != 1:
            raise ValueError(f"mojolearn: {path!r} radius and p must hold one value each")
        obj = cls(
            radius=float(radius[0]),
            metric=_serialize.scalar_str(arrays, "metric"),
            algorithm=_serialize.scalar_str(arrays, "algorithm"),
            p=float(p[0]),
        )
        _restore_mode(obj, arrays)
        obj._check_refusals()
        nf, ns = int(meta[0]), int(meta[1])
        index = _serialize.exact(arrays, "index", "<f4")
        if index.ndim != 2 or tuple(index.shape) != (ns, nf) or ns < 1 or nf < 1:
            raise ValueError(
                f"mojolearn: {path!r} index has shape {tuple(index.shape)}, meta says ({ns}, {nf})"
            )
        obj._index = index
        obj.n_samples_fit_ = ns
        obj.n_features_in_ = nf
        return obj

    def radius_neighbors(
        self, X=None, radius=None, return_distance=True, sort_results=False
    ):
        """Neighbours within `radius`, as ragged arrays, one row per query.

        Returns `(distances, indices)` when `return_distance` is True and
        `indices` alone otherwise, matching scikit-learn. Each element is a
        1-D Array; the rows have different lengths, so the containers are
        Python lists rather than a rectangular block (DEVIATION 2375).

        `X=None` queries the fitted data against itself, as scikit-learn
        does. Note that scikit-learn EXCLUDES each point from its own
        neighbour list in that case and this does NOT: the ball cover returns
        the self-edge, DBSCAN counts on it, and dropping it here would make
        the Python surface disagree with the CSR every other consumer sees.
        That difference is named rather than papered over.

        DEVIATION 2375: the two containers are Python LISTS of `Array`s
        (int64 indices, float32 distances), one per query row, where they
        were object-dtype ndarrays of ndarrays. Indexing with `[i]` and
        `len()` read the same; `np.asarray(container)` does not (it was
        never a rectangular block either).
        """
        if self._index is None:
            raise ValueError(
                "mojolearn RadiusNeighbors: call fit() before "
                "radius_neighbors()"
            )
        r = float(self.radius if radius is None else radius)
        if not math.isfinite(r) or r <= 0:
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
            q, _ = as_f32_c(X, ndim=2, name="X")
            if q.shape[1] != idx.shape[1]:
                raise ValueError(
                    f"mojolearn RadiusNeighbors: X has {q.shape[1]} features "
                    f"but the index was fit on {idx.shape[1]}"
                )
        nq = q.shape[0]

        indptr = empty((nq + 1,), "<i4")
        nnz = self._bind("_mojolearn").radius_neighbors_count(
            addr_ro(idx, name="idx"), addr_ro(q, name="q"), addr(indptr, name="indptr"),
            # ORDER MATCHES bindings/_mojolearn.mojo::
            # radius_neighbors_count_binding. n_index, n_queries, n_features,
            # radius, metric, metric_arg
            [idx.shape[0], nq, idx.shape[1], r,
             self._metric_value, self._metric_arg],
        )
        cols = empty((nnz,), "<i4")
        dists = empty((nnz,), "<f4")
        # NO NEIGHBOUR IN THE WHOLE CALL (2026-09-14 night, found by the
        # batch part of tools/identity_break.py): a zero-length `cols` has no
        # address, and the fill binding refused it with "null int32 buffer
        # address", so a query row with no neighbour raised when asked ALONE
        # and answered an empty row inside a larger batch (base fixture,
        # held-out rows 1 and 7 of the radius lane). The counting pass has
        # already written every indptr entry (all zero), so there is nothing
        # to fill.
        got = 0 if nnz == 0 else self._bind("_mojolearn").radius_neighbors_fill(
            addr_ro(idx, name="idx"), addr_ro(q, name="q"), addr(indptr, name="indptr"), addr(cols, name="cols"),
            addr(dists, name="dists"),
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

        # DEVIATION 2375 -- API CHANGE: the ragged containers are Python
        # LISTS of Arrays (int64 indices, float32 distances), one per query
        # row, where they were object-dtype ndarrays. Same per-row contents,
        # indexed with `[i]` as before; `len()` is the query count.
        ptr = indptr.tolist()
        ind = []
        dst = []
        for i in range(nq):
            a, b = ptr[i], ptr[i + 1]
            row_i = cols[a:b].astype("<i8")
            row_d = dists[a:b]
            if sort_results:
                # STABLE, and the stability is the point: the row arrives in
                # ascending index order under `identical`, so ties in
                # distance keep that order and the result is
                # (distance, index) lexicographic without a second key.
                # Python's `sorted` is stable by definition, over O(row)
                # items (it was `np.argsort(kind="stable")`).
                d = row_d.tolist()
                order = sorted(range(len(d)), key=d.__getitem__)
                ii = row_i.tolist()
                row_i = Array.from_list([ii[j] for j in order], "<i8")
                row_d = Array.from_list([d[j] for j in order], "<f4")
            ind.append(row_i)
            dst.append(row_d)
        if return_distance:
            return dst, ind
        return ind
