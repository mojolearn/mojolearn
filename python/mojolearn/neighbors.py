# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# References: `NearestNeighbors` follows cuVS's brute force;
# `KNeighborsClassifier` and `KNeighborsRegressor` follow cuML's
# `kneighbors_classifier.pyx` / `kneighbors_regressor.pyx` over
# `ML::knn_classify` / `ML::knn_regress` (`neighbors/impl/knn/knn.mojo`,
# `neighbors/impl/selection/knn.mojo`).
"""Exact nearest-neighbor search on the GPU, and the k-NN classifier,
k-NN regressor and radius search built on it.

`NearestNeighbors` follows cuVS's brute force; `KNeighborsClassifier` and
`KNeighborsRegressor` follow cuML's estimators of the same names;
`RadiusNeighbors` returns every neighbor within a radius over a random ball
cover. Every search is exact, never approximate.
"""

from . import _portable_math as math

from . import _mojolearn, _serialize
from ._array import Array
from ._buffer import addr, addr_ro, as_f32_c, as_f32_dense_c, as_i64_c, empty
from ._labels import encode_labels, sorted_classes
from ._mode import NumericModeMixin
from ._mode import ClassifierScoreMixin, RegressorScoreMixin
from .linear_model import (
    _check_saved_by, _dtype_name, _is_integer_labels, _restore_mode,
    _saved_mode, _shape_of,
)

#: the x_neighbors binding, which holds the p2m_* ops the data loops below
#: run through (lane apple-fast-py2mojo-neighbors)
_XN = "_mojolearn_x_neighbors"

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
    nf, ns, k, qt, o2d, n_out = (int(meta[i]) for i in range(_KNN_META_FIELDS))  # glue: six saved metadata fields
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
_DIST_INNER_PRODUCT = 6
_DIST_LINF = 7
_DIST_CANBERRA = 8
_DIST_LP_UNEXPANDED = 9
_DIST_CORRELATION_EXPANDED = 10
_DIST_BRAY_CURTIS = 14
_DIST_JENSEN_SHANNON = 15

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
    # 2026-09-27 (lane x-neighbors-metrics): distance_ops.mojo::
    # extra_metric_cell, one thread per cell on the device and the same
    # function on the host.
    "canberra": _DIST_CANBERRA,
    "braycurtis": _DIST_BRAY_CURTIS,
    "correlation": _DIST_CORRELATION_EXPANDED,
    "jensenshannon": _DIST_JENSEN_SHANNON,
    "inner_product": _DIST_INNER_PRODUCT,
}

#: Names in cuML's `VALID_METRICS["brute"]` (`neighbors/__init__.py:27-48`)
#: that this tree does not compute. Refused BY NAME so a caller learns the
#: metric is UNIMPLEMENTED rather than unknown. haversine needs arcsin, and
#: `checks/numerics.mojo` carries no pinned arcsin: a vendor asin is a
#: last-bit vendor choice (IDENTITY_PATHS row 12), so it cannot be one
#: arithmetic on every column.
_UNSUPPORTED_METRICS = (
    "haversine",
)

#: The metrics that need X >= 0 (their logarithms), refused on negative
#: input as scipy's jensenshannon is undefined there.
_NONNEGATIVE_METRICS = frozenset({_DIST_JENSEN_SHANNON})


def _check_metric_input(cls_name, value, x):
    """Refuse input a metric is undefined on, before any device work."""
    if value in _NONNEGATIVE_METRICS and x.size and float(x.min()) < 0.0:
        raise ValueError(
            f"mojolearn {cls_name}: metric='jensenshannon' needs X >= 0 "
            "(it takes logarithms of the coordinates)"
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
            + ", ".join(sorted(_METRIC_TABLE))  # glue: metric names for the error message
        )
    else:
        raise ValueError(
            f"mojolearn {cls_name}: unknown metric {metric!r}. Implemented: "
            + ", ".join(sorted(_METRIC_TABLE))  # glue: metric names for the error message
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
    if key in _METRIC_TABLE and key not in _RBC_METRIC_TABLE:
        raise ValueError(
            f"mojolearn {cls_name}: metric={metric!r} is computed by brute "
            "force only (NearestNeighbors(algorithm='brute')). The random "
            "ball cover's pruning is the triangle inequality on its "
            "landmark radii and its kernels carry the Minkowski family; "
            "cuML's own VALID_METRICS['rbc'] is {euclidean, haversine, l2}"
        )
    if key not in _RBC_METRIC_TABLE:
        raise ValueError(
            f"mojolearn {cls_name}: unknown metric {metric!r}. The random "
            "ball cover admits: " + ", ".join(sorted(_RBC_METRIC_TABLE))  # glue: metric names for the error message
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


# Maintainer notes (moved from the docstring):
# - Reference: cuVS's fused L2 kernel. Exact by deliberate scope: an
#   approximate index is a different algorithm with a different contract.
# - Honored/refused rows were measured row by row by `tools/e2u_matrix_fit.py`.
# - n_neighbors above n_samples_fit is refused: the reference's short-index
#   fill is not implemented (knn_brute_force.mojo). Under
#   IDENTICAL a k above 1024 selects in rounds of 1024 and
#   merges index tiles by binary search (DEVIATION 6300,
#   neighbors/checks/select_radix_identical.mojo); the answer is the same
#   ascending (distance, index) list. FAST runs k > 256 through the radix
#   select (reference: RAFT).
# - query_tile is a memory number only (`check_knn_tiled_is_query_tile_
#   invariant`). The planner: NVIDIA IDENTICAL starts at 512, other modes and
#   columns at 256.
# - metric: the rows of cuML's `_build_metric_type`
#   (nearest_neighbors.pyx:520-553) this tree computes, plus the extra
#   metrics (distance_ops.mojo::extra_metric_cell); haversine is refused by
#   name (no pinned arcsin).
# - p: refused at p <= 0, inf, NaN and subnormal (DEVIATION 552 -- 1/p and
#   the vendor flush policy). cuML passes self.p for every metric
#   (nearest_neighbors.pyx:852-854) and every non-Lp op discards it; here a
#   non-default p under another metric is refused rather than ignored.
# - algorithm 'rbc': exact k-NN over a random ball cover (DEVIATIONS 558 to
#   567). 'kd_tree' / 'ball_tree' are an engineering refusal (see the
#   message in `_check_refusals`).
# - The arm (fused vs tiled, `KNN_METHOD_AUTO`) is not a parameter by design
#   (POLICY CHOICE 4); under NUMERIC_IDENTICAL AUTO is pinned to the tiled
#   arm on every column (DEVIATION 509).
class NearestNeighbors(NumericModeMixin):
    """Exact k-nearest-neighbor search.

    Every query is compared against every fitted point (``algorithm='brute'``,
    the default), or searched over an exact random ball cover index
    (``algorithm='rbc'``). Both return the exact neighbors; there is no
    approximate index and no recall to trade away. Input is converted to
    float32. Neighbors come back nearest first, with ties in distance broken
    by the lower index.

    Parameters
    ----------
    n_neighbors : int, default 5
        Number of neighbors returned by `kneighbors`, scikit-learn's default.
        Must be between 1 and the number of fitted samples; a larger value
        raises ``ValueError``.
    query_tile : int, default 0
        Queries processed per pass. It changes memory use only, never the
        answer. ``0`` lets the library choose, within its workspace limit and
        clamped to the query count; a positive value requests a fixed
        starting size. ``used_query_tile_`` reports the size that ran.
    metric : str, default 'euclidean'
        Distance, by name. With ``algorithm='brute'`` or ``'auto'``:
        'euclidean'/'l2', 'sqeuclidean', 'manhattan'/'l1'/'cityblock'/
        'taxicab', 'chebyshev'/'linf', 'cosine', 'minkowski'/'lp',
        'canberra', 'braycurtis', 'correlation', 'jensenshannon' (needs
        ``X >= 0``) and 'inner_product' (a similarity: larger is nearer, and
        `kneighbors` returns the products themselves). With
        ``algorithm='rbc'``, only true metrics: 'euclidean'/'l2',
        'manhattan'/'l1'/'cityblock'/'taxicab', 'chebyshev'/'linf' and
        'minkowski'/'lp' with ``p >= 1``. 'haversine', a callable and
        unknown names raise ``ValueError``.
    algorithm : {'brute', 'auto', 'rbc'}, default 'brute'
        'brute' and 'auto' (which means 'brute') compare every pair. 'rbc'
        searches an exact random ball cover, whose pruning rests on the
        triangle inequality, so it accepts only true metrics. 'kd_tree' and
        'ball_tree' raise ``ValueError``.
    p : float, default 2
        Minkowski exponent, read only by ``metric='minkowski'`` (or 'lp').
        It must be finite and positive (and at least 1 with
        ``algorithm='rbc'``). Under any other metric, a value other than 2
        raises ``ValueError`` instead of being ignored. Unlike scikit-learn,
        ``metric='minkowski', p=2`` runs the Minkowski computation rather
        than switching to Euclidean; the two agree to a few units in the last
        place.
    numeric_mode : {'identical', 'fast'}, optional
        Numeric mode for this estimator; ``None`` uses the process default
        (`mojolearn.set_numeric_mode`). 'fast' is available on Apple GPUs
        only.

    Attributes
    ----------
    n_samples_fit_ : int
        Number of fitted samples.
    n_features_in_ : int
        Number of features seen at `fit`.
    used_query_tile_ : int or None
        Query batch size the last `kneighbors` call ran with; ``None`` after
        an ``algorithm='rbc'`` search.
    n_candidate_distances_ : int
        Set by an ``algorithm='rbc'`` search: the number of distances the
        ball cover computed.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> nn = mojolearn.NearestNeighbors(n_neighbors=5).fit(X)
    >>> distances, indices = nn.kneighbors(X[:10])
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
        # `used_query_tile_` is set by the first query, not here: a public
        # attribute ending in an underscore marks a fitted estimator.

    # The knn host inference lane, 2026-09-14: the file holds what
    # `kneighbors` reads and nothing else.
    def save(self, path):
        """Save the fitted model to an ``.npz`` file.

        The file holds the float32 index as fitted and the parameters
        (`n_neighbors`, `query_tile`, `metric`, `p`, `algorithm`, numeric
        mode). Equal models write identical files.
        `mojolearn.host_model(path)` can search the saved model on a CPU with
        no GPU (brute force, 'euclidean' and 'sqeuclidean').

        Parameters
        ----------
        path : str or path-like
            Destination file.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        """
        return _serialize.write_npz(path, _knn_arrays(self))

    @classmethod
    def load(cls, path):
        """Load a model written by `save`.

        Arrays are read at their saved dtypes and never cast. A file naming
        a metric or algorithm this class refuses raises ``ValueError``, as
        `fit` would.

        Parameters
        ----------
        path : str or path-like
            File written by `save`.

        Returns
        -------
        NearestNeighbors
            A fitted model, ready for `kneighbors`.
        """
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
        if (getattr(self, "weights", None) == "distance"
                and isinstance(self.metric, str) and self.metric.lower() == "inner_product"):
            raise ValueError(
                f"mojolearn {type(self).__name__}: weights='distance' is refused "
                "with metric='inner_product': it weighs a neighbour by 1 / d, "
                "and an inner product is a similarity (larger is nearer, and "
                "it can be zero or negative), not a distance"
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
        """Store the points to search.

        Parameter checks run here, so an unsupported metric, algorithm or
        `p` raises ``ValueError`` before any search.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Points to search; converted to float32.
        y : ignored
            Accepted for scikit-learn compatibility.

        Returns
        -------
        self
        """
        self._check_refusals()
        # A refit drops the device copy of the previous index HERE, not at
        # the next call: the copy is keyed on the array's address and shape,
        # and a new array of the same shape can land at a freed address, so
        # a key match after a refit would serve the old bytes.
        self._release_resident_index()
        idx, _ = as_f32_dense_c(X, ndim=2, name="X")
        if self.algorithm != "rbc":
            _check_metric_input(type(self).__name__, _resolve_metric(type(self).__name__, self.metric, self.p)[0], idx)
        # Held on the instance so the memory outlives this call: the Mojo side
        # borrows the address at `kneighbors` time and owns nothing.
        self._index = idx
        self.n_samples_fit_ = idx.shape[0]
        self.n_features_in_ = idx.shape[1]
        return self

    # Maintainer notes (moved from the docstring): the kernel computes squared
    # Euclidean distances and the square root is taken on the way out, over
    # n_queries * k values. The index stays on the device (DEVIATION 2921,
    # `neighbors/resident_index.mojo`): the resident search is
    # `knn_search`'s body after its upload, so the bits are `knn_search`'s.
    # The copy is keyed on the index array's address and shape, released at
    # refit and when the instance is collected. The CPU host binding has no
    # resident entry and reads the caller's memory on every call.
    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        """Find the nearest fitted points to each query, nearest first.

        Distances are the chosen metric's ('euclidean' returns true, not
        squared, distances). Ties in distance are broken by the lower index,
        matching scikit-learn's layout and ordering.

        On a GPU the fitted points are copied to the device at the first call
        after `fit` and reused by later calls. Changing the array passed to
        `fit` in place after that is not seen; call `fit` again instead (as
        with cuML, whose `fit` copies the data to the device).

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Query points; converted to float32.
        n_neighbors : int, optional
            Neighbors per query; defaults to the constructor's
            `n_neighbors`. Must be between 1 and `n_samples_fit_`.
        return_distance : bool, default True
            Return the distances as well as the indices.

        Returns
        -------
        distances : Array of shape (n_queries, n_neighbors), float32
            Returned only when `return_distance` is True.
        indices : Array of shape (n_queries, n_neighbors), int64
            Row indices into the fitted data.

        Raises
        ------
        ValueError
            If the model is not fitted, `X` has the wrong number of features,
            or `n_neighbors` is out of range.
        """
        if self._index is None:
            raise ValueError("mojolearn: call fit before kneighbors")
        k = self.n_neighbors if n_neighbors is None else n_neighbors
        q, _ = as_f32_dense_c(X, ndim=2, name="X")
        if self.algorithm != "rbc":
            _check_metric_input(type(self).__name__, self._dist_params()[0], q)
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
            # an inner-product search hands back the products themselves:
            # the binding negates its selection keys on the device before
            # the readback (the host binding in its copy loop; lane
            # pyglue-numeric, the negation was a second call here)
            return dist, ind.astype("<i8")
        return ind.astype("<i8")


# Maintainer notes (moved from the docstring):
# - Reference: cuML's `KNeighborsClassifier`; `predict` is `knn_classify`,
#   the tie rule cuML's `class_vote_kernel`
#   (`src_prims/selection/knn.cuh:74-109`), scikit-learn's `mode`. The vote
#   is a serial fold per query in neighbor order, so it is bit-identical in
#   both numeric modes by construction (`neighbors/impl/selection/knn.mojo`,
#   DEVIATION 542). Rows measured by `tools/e2u_matrix_fit.py`.
# - weights='distance' is DEVIATION 556 (scikit-learn `_base.py:108-113`).
#   The weighted arm asks the search for the rooted distance where the
#   uniform arm asks for the squared one, because a weight reads the value
#   and a vote only the order (estimator.mojo policy 8).
# - algorithm='rbc' is refused: knn_classify searches inside its own entry
#   point, so 'rbc' would be accepted and inert.
# - y: any int values, multi-output as cuML's `vector<int*> y`.
# - classes_: the pyx's `cp.unique`; `_labels.sorted_classes`, DEVIATION
#   2340 (a list rather than an ndarray). The Mojo side recomputes the set
#   with `getUniquelabels` and `predict` asserts the two agree.
class KNeighborsClassifier(ClassifierScoreMixin, NearestNeighbors):
    """Classifier that votes among the k nearest training points.

    For each query, the classes of its `n_neighbors` nearest training points
    are counted and the most frequent wins; a tie goes to the lowest class in
    sorted order, as in cuML and scikit-learn. With ``weights='distance'``
    each neighbor's vote is weighted by ``1 / distance``. Neighbor search is
    exact brute force, as in `NearestNeighbors`.

    Parameters
    ----------
    n_neighbors : int, default 5
        Number of neighbors that vote. Must be between 1 and the number of
        training samples.
    query_tile : int, default 0
        As in `NearestNeighbors`: memory use only, never the answer.
    weights : {'uniform', 'distance'} or None, default 'uniform'
        'uniform' (or ``None``) gives every neighbor one vote. 'distance'
        follows scikit-learn: weight ``1 / d``, and when a query has
        neighbors at distance exactly zero, those neighbors get weight 1 and
        the others in that row get 0. A callable raises ``ValueError``.
        'distance' with ``metric='inner_product'`` raises ``ValueError``
        (an inner product is a similarity, not a distance).
    metric : str, default 'euclidean'
        As in `NearestNeighbors` with ``algorithm='brute'``.
    algorithm : {'brute', 'auto'}, default 'brute'
        Exact brute force. 'rbc' raises ``ValueError`` here; use
        ``NearestNeighbors(algorithm='rbc').kneighbors`` and vote on the
        result instead.
    p : float, default 2
        As in `NearestNeighbors`.
    numeric_mode : {'identical', 'fast'}, optional
        As in `NearestNeighbors`. 'fast' is available on Apple GPUs only.

    Attributes
    ----------
    classes_ : list of int, or list of lists for a 2-D `y`
        Sorted unique labels per output, as Python ints (a list, not an
        ndarray).
    outputs_2d_ : bool
        Whether `y` had more than one output column.
    n_samples_fit_ : int
        Number of training samples.
    n_features_in_ : int
        Number of features seen at `fit`.
    used_query_tile_ : int
        Query batch size the last prediction ran with.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = (X[:, 0] > 0.5).astype(np.int32)
    >>> clf = mojolearn.KNeighborsClassifier(n_neighbors=5).fit(X, y)
    >>> labels = clf.predict(X[:10])
    >>> proba = clf.predict_proba(X[:10])
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

    # File members: the index and parameters `NearestNeighbors.save` writes,
    # `weights`, the `<i4` label columns the binding takes (`_y_cols`,
    # policy 6), `classes` (`<i8`, every output's sorted classes
    # concatenated) and `class_counts` (`<i8`, one per output). The knn host
    # inference lane, 2026-09-14.
    def save(self, path):
        """Save the fitted model to an ``.npz`` file.

        The file holds the training points, labels, classes and parameters.
        `mojolearn.host_model(path)` can predict from it on a CPU with no
        GPU.

        Parameters
        ----------
        path : str or path-like
            Destination file.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        """
        if self._y_cols is None:
            raise RuntimeError("this estimator is not fitted yet")
        arrays = _knn_arrays(self)
        arrays["y_cols"] = self._y_cols
        arrays["classes"] = Array.from_list(
            [int(c) for cl in self._classes_list for c in cl], "<i8"  # glue: saved classes per output column
        )
        arrays["class_counts"] = Array.from_list(
            [len(cl) for cl in self._classes_list], "<i8"  # glue: saved classes per output column
        )
        return _serialize.write_npz(path, arrays)

    @classmethod
    def load(cls, path):
        """Load a model written by `save`.

        `classes_` is rebuilt from the saved labels as `fit` builds it; a
        file whose stored classes disagree with its labels raises
        ``ValueError``.

        Parameters
        ----------
        path : str or path-like
            File written by `save`.

        Returns
        -------
        KNeighborsClassifier
            A fitted model, ready for `predict`.
        """
        obj, arrays, (o2d, n_out) = _load_knn(cls, path)
        y_cols = _load_y_cols(arrays, path, "<i4", obj.n_samples_fit_, n_out)
        classes = _serialize.exact(arrays, "classes", "<i8").tolist()
        counts = _serialize.exact(arrays, "class_counts", "<i8").tolist()
        if len(counts) != n_out or sum(counts) != len(classes):  # glue: class count check over the outputs
            raise ValueError(f"mojolearn: {path!r} classes and class_counts disagree with n_outputs")
        rebuilt = [encode_labels(y_cols[i])[0] for i in range(n_out)]  # glue: one native encoding per output column
        off = 0
        for i, count in enumerate(counts):  # glue: saved class check per output
            saved = [int(c) for c in classes[off:off + count]]  # glue: saved class check per output
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
        """Store the training points and their labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training points; converted to float32.
        y : array-like of int, shape (n_samples,) or (n_samples, n_outputs)
            Integer class labels of any values (negative values and gaps are
            fine) within the int32 range. A 2-D `y` votes each column
            separately. Float labels raise ``ValueError``; cast them to int
            first (cuML requires int32 labels too).

        Returns
        -------
        self
        """
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
        self.outputs_2d_ = len(shape) == 2 and shape[1] != 1
        if n_out == 1:
            # lane/gap-linear-nv: one output is already its own contiguous
            # column. The int64 -> int32 narrowing is a C-level copy (exact
            # after the range check above) and the classes come from the
            # native encoder on that buffer: the same ints, in the same
            # sorted order, the list path below produced through a 2 x n
            # Python-object round trip (tolist, from_list) that cost more
            # than the GPU search on the board's 200,000-row index.
            y32 = ya.reshape((n,)).astype("<i4")
            self._y_cols = y32.reshape((1, n))
            self._classes_list = [encode_labels(y32)[0]]
            return self
        # the narrowing is a C-level copy (exact after the range check) and
        # the transpose a device op (`xn_p2m_transpose_i`; the host binding's
        # same op on a CPU-only install); the classes per output column from
        # the native encoder, as the one-output path above
        y32 = ya.astype("<i4")
        yc = empty((n_out, n), "<i4")
        if n:
            self._bind(_XN).xn_p2m_transpose_i(
                [addr_ro(y32, name="y"), addr(yc, name="y_cols")], [n, n_out], [])
        self._y_cols = yc
        self._classes_list = [encode_labels(yc[j])[0] for j in range(n_out)]  # glue: one native encoding per output column
        return self

    @property
    def classes_(self):
        """Sorted unique labels seen at `fit`, as a list of ints; a list of
        such lists for a 2-D `y`."""
        if self._y_cols is None:
            raise AttributeError("classes_ is set by fit")
        if self.outputs_2d_:
            return self._classes_list
        return self._classes_list[0]

    def _predict(self, X, want_proba):
        if self._index is None or self._y_cols is None:
            raise ValueError("mojolearn: call fit before predict")
        q, _ = as_f32_dense_c(X, ndim=2, name="X")
        _check_metric_input(type(self).__name__, self._dist_params()[0], q)
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
        n_classes = [len(c) for c in self._classes_list]  # glue: class counts per output column
        # The native contract writes exactly one of these two outputs.  Keep
        # the unselected address valid without materializing its full public
        # shape: at million-row scale the unused probability matrix can be
        # tens or hundreds of MiB, and predict_proba likewise has no use for
        # a labels matrix.  Both GPU and host bindings promise not to read or
        # write the unselected pointer (their one-element sentinel contract).
        labels = (empty((1,), "<i4") if want_proba else
                  empty((nq, n_out), "<i4"))
        proba = (empty((nq * sum(n_classes),), "<f4") if want_proba else  # glue: class counts per output column
                 empty((1,), "<f4"))
        uniq = empty((sum(n_classes),), "<i4")  # glue: class counts per output column
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
        for i, (count, b) in enumerate(zip(n_classes, self._classes_list)):  # glue: class set check per output column
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
        """Predict the class of each query by neighbor vote.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Query points; converted to float32.

        Returns
        -------
        Array of int64, shape (n_queries,) or (n_queries, n_outputs)
            The voted labels, in the original label values; 2-D when `y`
            was 2-D.

        Raises
        ------
        ValueError
            If the model is not fitted, `X` has the wrong number of features,
            or `n_neighbors` exceeds the number of training samples.
        """
        labels, _, _ = self._predict(X, want_proba=False)
        if self.outputs_2d_:
            return labels.astype("<i8")
        # (nq, 1) -> (nq,) is a C-order reshape; int32 -> int64 is exact.
        return labels.reshape((labels.shape[0],)).astype("<i8")

    def predict_proba(self, X):
        """Return each query's vote fractions per class.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Query points; converted to float32.

        Returns
        -------
        Array of float32, shape (n_queries, n_classes), or a list of them
            Columns follow `classes_` order. For a 2-D `y`, one array per
            output, as cuML and scikit-learn return it.
        """
        labels, proba, n_classes = self._predict(X, want_proba=True)
        # `labels` is the one-element sentinel in this arm.  The selected
        # flat probability buffer records the query count without retaining
        # an otherwise-unused labels matrix.
        nq = proba.shape[0] // sum(n_classes)  # glue: class counts per output column
        out = []
        off = 0
        for n in n_classes:  # glue: probability block slices per output
            # A slice of an Array COPIES (the _array contract).
            out.append(proba[off:off + nq * n].reshape((nq, n)))
            off += nq * n
        if self.outputs_2d_:
            return out
        return out[0]


# Maintainer notes (moved from the docstring): reference cuML's
# `KNeighborsRegressor` (`ML::knn_regress`, `regress_avg_kernel`,
# `src_prims/selection/knn.cuh:112-131`). The mean is a serial float32 fold
# per query in neighbor order, then one division by k; the same arithmetic
# in both numeric modes, with the seams flushed under NUMERIC_IDENTICAL
# (DEVIATION 542). weights='distance' is DEVIATION 556. float64 targets are
# cast to float32 as cuML does (convert_to_dtype=np.float32).
class KNeighborsRegressor(RegressorScoreMixin, NearestNeighbors):
    """Regressor that averages the targets of the k nearest training points.

    Each prediction is the mean of the `n_neighbors` nearest targets, or with
    ``weights='distance'`` the weighted mean ``sum(w * y) / sum(w)`` with
    ``w = 1 / d``. Neighbor search is exact brute force, as in
    `NearestNeighbors`. Targets are converted to float32.

    Parameters
    ----------
    n_neighbors : int, default 5
        Number of neighbors averaged. Must be between 1 and the number of
        training samples.
    query_tile : int, default 0
        As in `NearestNeighbors`: memory use only, never the answer.
    weights : {'uniform', 'distance'} or None, default 'uniform'
        'uniform' (or ``None``) averages equally. 'distance' follows
        scikit-learn, including its rule for neighbors at distance exactly
        zero (they get weight 1, the rest of that row 0). A callable raises
        ``ValueError``, as does 'distance' with ``metric='inner_product'``.
    metric : str, default 'euclidean'
        As in `NearestNeighbors` with ``algorithm='brute'``.
    algorithm : {'brute', 'auto'}, default 'brute'
        Exact brute force. 'rbc' raises ``ValueError`` here; use
        ``NearestNeighbors(algorithm='rbc').kneighbors`` instead.
    p : float, default 2
        As in `NearestNeighbors`.
    numeric_mode : {'identical', 'fast'}, optional
        As in `NearestNeighbors`. 'fast' is available on Apple GPUs only.

    Attributes
    ----------
    outputs_2d_ : bool
        Whether `y` had more than one output column.
    n_samples_fit_ : int
        Number of training samples.
    n_features_in_ : int
        Number of features seen at `fit`.
    used_query_tile_ : int
        Query batch size the last prediction ran with.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = X.sum(axis=1)
    >>> reg = mojolearn.KNeighborsRegressor(n_neighbors=5).fit(X, y)
    >>> pred = reg.predict(X[:10])
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

    # File members: the index and parameters `NearestNeighbors.save` writes,
    # `weights` and the `<f4` target columns the binding takes (`_y_cols`,
    # policy 6). The knn host inference lane, 2026-09-14.
    def save(self, path):
        """Save the fitted model to an ``.npz`` file.

        The file holds the training points, targets and parameters.
        `mojolearn.host_model(path)` can predict from it on a CPU with no
        GPU.

        Parameters
        ----------
        path : str or path-like
            Destination file.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        """
        if self._y_cols is None:
            raise RuntimeError("this estimator is not fitted yet")
        arrays = _knn_arrays(self)
        arrays["y_cols"] = self._y_cols
        return _serialize.write_npz(path, arrays)

    @classmethod
    def load(cls, path):
        """Load a model written by `save`.

        Parameters
        ----------
        path : str or path-like
            File written by `save`.

        Returns
        -------
        KNeighborsRegressor
            A fitted model, ready for `predict`.
        """
        obj, arrays, (o2d, n_out) = _load_knn(cls, path)
        obj._y_cols = _load_y_cols(arrays, path, "<f4", obj.n_samples_fit_, n_out)
        obj.outputs_2d_ = o2d
        return obj

    def fit(self, X, y):
        """Store the training points and their targets.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training points; converted to float32.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            Numeric targets; converted to float32. A 2-D `y` is averaged
            column by column.

        Returns
        -------
        self
        """
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
            # The transpose on the device (`xn_p2m_transpose`; the host
            # binding's same op on a CPU-only install)
            yc = empty((n_out, n), "<f4")
            if n:
                self._bind(_XN).xn_p2m_transpose(
                    [addr_ro(ya, name="y"), addr(yc, name="y_cols")], [n, n_out], [])
            self._y_cols = yc
        self.outputs_2d_ = len(shape) == 2 and shape[1] != 1
        return self

    def predict(self, X):
        """Predict each query's target from its neighbors.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Query points; converted to float32.

        Returns
        -------
        Array of float32, shape (n_queries,) or (n_queries, n_outputs)
            The (weighted) mean target; 2-D when `y` was 2-D.

        Raises
        ------
        ValueError
            If the model is not fitted, `X` has the wrong number of features,
            or `n_neighbors` exceeds the number of training samples.
        """
        if self._index is None or self._y_cols is None:
            raise ValueError("mojolearn: call fit before predict")
        q, _ = as_f32_dense_c(X, ndim=2, name="X")
        _check_metric_input(type(self).__name__, self._dist_params()[0], q)
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


# Maintainer notes (moved from the docstring):
# - The index is cuVS's random ball cover (`neighbors/impl/ball_cover/`),
#   which DBSCAN uses for its eps neighborhood. Exactness is asserted by
#   `neighbors/checks/radius_check.mojo` against a host brute-force oracle
#   per cell; the per-metric gate is `pixi run
#   check-ball-cover-knn-identical`. Metric refusals and their reasons live
#   in `_resolve_rbc_metric`.
# - The distances are recomputed, not stored by the search: a separate pass
#   walks the finished neighbor list (`neighbors/checks/
#   radius_distances.mojo` carries the argument that the value is the same;
#   under `identical` the check asserts it bit for bit).
# - sort_results: the device returns each row in ascending index order under
#   `identical` (DEVIATION 551), so a stable sort by distance gives
#   (distance, index) order with the tie-break free. The sort is the
#   segmented device sort `xn_p2m_row_sort`.
# - Two calls cross the boundary per query (count, then fill), because the
#   output size is not a function of the inputs; the ball cover is built in
#   each (`neighbors/estimator.mojo`).
class RadiusNeighbors(NumericModeMixin):
    """Find every fitted point within a radius of each query.

    Returns scikit-learn's `radius_neighbors` shape: a ragged
    ``(distances, indices)`` pair with one variable-length array per query.
    The search runs over a random ball cover index (neither brute force nor
    one of scikit-learn's trees) and returns the exact set: its pruning is a
    triangle-inequality bound, not an approximation. Input is converted to
    float32.

    Parameters
    ----------
    radius : float, default 1.0
        Default search radius; must be positive and finite.
    metric : str, default 'euclidean'
        Any distance that satisfies the triangle inequality:
        'euclidean'/'l2', 'manhattan'/'l1'/'cityblock'/'taxicab',
        'chebyshev'/'linf' and 'minkowski'/'lp' with ``p >= 1``. 'cosine',
        'sqeuclidean', Minkowski with ``p < 1`` and other names raise
        ``ValueError``, because the index would silently miss neighbors
        under them. For squared Euclidean, use 'euclidean' and square the
        distances.
    algorithm : {'auto', 'rbc'}, default 'auto'
        Both mean the random ball cover. 'brute', 'kd_tree' and 'ball_tree'
        raise ``ValueError``, since they would name a different algorithm.
    p : float, default 2
        Minkowski exponent, read only by ``metric='minkowski'`` (or 'lp');
        must be finite and at least 1. Under any other metric, a value other
        than 2 raises ``ValueError``.
    numeric_mode : {'identical', 'fast'}, optional
        Numeric mode for this estimator; ``None`` uses the process default
        (`mojolearn.set_numeric_mode`). 'fast' is available on Apple GPUs
        only.

    Attributes
    ----------
    n_samples_fit_ : int
        Number of fitted samples.
    n_features_in_ : int
        Number of features seen at `fit`.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> rn = mojolearn.RadiusNeighbors(radius=0.5).fit(X)
    >>> distances, indices = rn.radius_neighbors(X[:10], sort_results=True)
    >>> len(indices)
    10
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

    # The ball cover is not built at fit, a known cost rather than an
    # oversight: `neighbors/estimator.mojo`'s RADIUS NEIGHBOURS banner
    # records it.
    def fit(self, X, y=None):
        """Store the points to search.

        Parameter checks run here, so an unsupported metric, algorithm, `p`
        or radius raises ``ValueError`` before any search. The ball cover
        index is built at each `radius_neighbors` call, not here.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Points to search; converted to float32.
        y : ignored
            Accepted for scikit-learn compatibility.

        Returns
        -------
        self
        """
        self._check_refusals()
        idx, _ = as_f32_dense_c(X, ndim=2, name="X")
        self._index = idx
        self.n_samples_fit_ = idx.shape[0]
        self.n_features_in_ = idx.shape[1]
        return self

    def save(self, path):
        """Save the fitted model to an ``.npz`` file.

        The file holds the float32 points as fitted and the parameters
        (`radius`, `metric`, `p`, `algorithm`, numeric mode).
        `mojolearn.host_model(path)` can answer `radius_neighbors` from it on
        a CPU with no GPU.

        Parameters
        ----------
        path : str or path-like
            Destination file.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        """
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
        """Load a model written by `save`.

        Arrays are read at their saved dtypes and never cast. Parameters
        `fit` would refuse raise ``ValueError`` here too.

        Parameters
        ----------
        path : str or path-like
            File written by `save`.

        Returns
        -------
        RadiusNeighbors
            A fitted model, ready for `radius_neighbors`.
        """
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

    # DEVIATION 2375: the containers are Python lists of Arrays, where they
    # were object-dtype ndarrays of ndarrays. The self-edge under X=None is
    # kept because the ball cover returns it, DBSCAN counts on it, and
    # dropping it would make this surface disagree with the CSR every other
    # consumer sees.
    def radius_neighbors(
        self, X=None, radius=None, return_distance=True, sort_results=False
    ):
        """Find the fitted points within `radius` of each query.

        `X=None` queries the fitted points against themselves, as in
        scikit-learn. Unlike scikit-learn, each point is then included in
        its own neighbor list.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features), optional
            Query points; converted to float32. ``None`` uses the fitted
            points.
        radius : float, optional
            Search radius for this call; defaults to the constructor's
            `radius`. Must be positive and finite.
        return_distance : bool, default True
            Return the distances as well as the indices.
        sort_results : bool, default False
            Sort each row by increasing distance. Requires
            ``return_distance=True``. Under the 'identical' mode, rows come
            in ascending index order, so sorted rows break distance ties by
            the lower index.

        Returns
        -------
        distances : list of Array, float32
            One 1-D array per query. Returned only when `return_distance` is
            True.
        indices : list of Array, int64
            One 1-D array per query of row indices into the fitted data.
            Rows have different lengths, so these are Python lists rather
            than a rectangular array: indexing with ``[i]`` and ``len()``
            work, ``np.asarray`` on the list does not give a 2-D block.

        Raises
        ------
        ValueError
            If the model is not fitted, the radius is invalid, `X` has the
            wrong number of features, or ``sort_results=True`` with
            ``return_distance=False``.
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
            q, _ = as_f32_dense_c(X, ndim=2, name="X")
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
        if sort_results and nnz:
            # every row's stable sort by distance as one segmented sort on
            # the device (`xn_p2m_row_sort`, key (row, distance, position);
            # the rows arrive in ascending index order under `identical`, so
            # the result is (distance, index) lexicographic)
            from ._expansion_neighbors import _p2m_sort_rows
            cols, dists = _p2m_sort_rows(self, indptr, cols, dists, _XN)
        ptr = indptr.tolist()
        cols64 = cols.astype("<i8")
        # glue: one view per query row, the ragged API's list of Arrays
        ind = [cols64[ptr[i]:ptr[i + 1]] for i in range(nq)]  # glue: ragged API: one index view per query
        dst = [dists[ptr[i]:ptr[i + 1]] for i in range(nq)]  # glue: ragged API: one index view per query
        if return_distance:
            return dst, ind
        return ind


# scikit-learn keeps more estimators in this module than it defines; they resolve from
# the package at first use (`_sklearn_modules`), so `from mojolearn.neighbors import ...`
# takes the same names as scikit-learn's module.
from ._sklearn_modules import LAYOUT as _SKLEARN_LAYOUT, lazy_exports as _lazy_exports  # noqa: E402
__getattr__ = _lazy_exports("neighbors")
