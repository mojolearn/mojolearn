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

from . import _mojolearn, _native_archive_bridge, _native_family_bridge
from ._array import Array
from ._buffer import addr, addr_ro, as_f32_c, as_f32_dense_c, as_i64_c, empty
from ._labels import decode_labels, encode_labels, sorted_classes
from ._mode import NumericModeMixin
from ._native_neighbor_queries import native_method as _native_query_method
from ._mode import ClassifierScoreMixin, RegressorScoreMixin
from .linear_model import _dtype_name, _is_integer_labels, _shape_of

#: the x_neighbors binding, which holds the p2m_* ops the data loops below
#: run through (lane apple-fast-py2mojo-neighbors)
_XN = "_mojolearn_x_neighbors"

_DEFAULT_QUERY_TILE = 0  # Ask the compiled planner for its measured default.

#: The model file format (the knn host inference lane, 2026-09-14). `save`
#: writes exactly what `kneighbors` / `predict` read, raw bytes and exact
#: dtypes, natively (bindings/archive_schema_native.mojo), so equal models give
#: equal file hashes on every machine; `load` refuses a cast. The
#: index is the fitted `<f4` `(n_samples_fit_, n_features_in_)` block, the
#: classifier's labels the `<i4` `(n_outputs, n_samples_fit_)` column block
#: the binding takes (policy 6), the regressor's the same in `<f4`; `p`
#: travels as `<f8` and the three names (`metric`, `algorithm`, `weights`)
#: as text. `mojolearn.host_model(path)` predicts from the file on a CPU
#: with no GPU through `bindings/_mojolearn_core_host.mojo`.
_KNN_FORMAT = "mojolearn-knn-1"
#: RadiusNeighbors' model file (the neighbors and density inference lane,
#: 2026-09-15): the `<f4` index as fitted, `radius`, `p` as `<f8`, `metric`
#: and `algorithm` as text, `meta` `<i8` [n_features_in_, n_samples_fit_].
#: What `radius_neighbors` reads and nothing else; `mojolearn.host_model`
#: queries it on a CPU through the core host binding's
#: radius_neighbors_count and radius_neighbors_fill.
_RADIUS_FORMAT = "mojolearn-radius-1"

# The compiled saved-model owner, connected once at import: RadiusNeighbors'
# `save` and `load` are one native call each (bindings/archive_schema_native.mojo
# writes the schema registered below the class; host_archive_native.mojo
# builds the loaded model with `load`'s refusals). The k-NN classes are
# connected the same way: their device index is released by the native
# finalizer (`_native_archive_bridge.lifetime`), not a Python `__del__`.
_ARCHIVE, _ARCHIVE_CONTEXT = _native_archive_bridge.connect()
# The compiled family owner (bindings/neighbors_native.mojo): RadiusNeighbors
# `fit` is one native call; its refusals read the metric tables connected
# below them.
_NATIVE, _NATIVE_CONTEXT = _native_family_bridge.connect()


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

# The metric tables RadiusNeighbors' native refusals read (the names the
# saved-model loader reads them under).
_native_family_bridge.tables(host_archive_neighbor_metrics=dict(_METRIC_TABLE),
                             host_archive_rbc_metrics=dict(_RBC_METRIC_TABLE),
                             host_archive_unsupported_metrics=tuple(_UNSUPPORTED_METRICS))
# The shared native label, weight and score steps (bindings/classical_labels.mojo)
# the neighbors family's one-call methods read (encode/decode, accuracy, R^2).
from ._native_labels_bridge import label_context as _label_context  # noqa: E402
_native_family_bridge.tables(**_label_context())

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
        return _ARCHIVE(self, "archive_save", (path,), _ARCHIVE_CONTEXT)

    @classmethod
    @_native_query_method
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
        return _ARCHIVE(cls, "host_archive_load", (path,), _ARCHIVE_CONTEXT)

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

    def _release_resident_index(self):
        """Drop the device copy, if one is held. Quiet on a binding that
        cannot be reached any more (interpreter shutdown) and on a handle
        already released (a copied instance)."""
        _NATIVE(self, "family_xn_knn_release", (), _NATIVE_CONTEXT)

    # No Python `__del__`: dropping an instance releases its device index
    # through the native finalizer connected below the class
    # (`_native_archive_bridge.lifetime`, bindings/host_archive_lifetime.mojo).

    def __getstate__(self):
        """A pickle or a deepcopy carries no device handle: the integer is
        meaningful only in the process and registry that minted it, and an
        unpickled instance holding it could release ANOTHER model's live
        index. The copy uploads its own index at its first call."""
        return _NATIVE(self, "family_xn_knn_getstate", (), _NATIVE_CONTEXT)

    def __setstate__(self, state):
        return _NATIVE(self, "family_xn_knn_setstate", (state,), _NATIVE_CONTEXT)

    # The device-resident index handle and the per-query reports
    # (`used_query_tile_`, `n_candidate_distances_`) live in the holder dict
    # `_resident_holder` the fit creates, so a query never changes `__dict__`
    # (scikit-learn's check_dict_unchanged). The native owners write the
    # holder (bindings/neighbors_knn_native.mojo `_report`) and the connected
    # attribute slot reads it (bindings/neighbor_query_native.mojo: an
    # AttributeError before the first query that sets a report).

    def _mojolearn_after_fit(self):
        """After every fit: release the last fit's device index and give
        the instance an empty holder for the next one."""
        self._release_resident_index()
        self.__dict__["_resident_holder"] = {}

    @_native_query_method
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
        return _NATIVE(self, "family_xn_knn_nn_fit", (X, y), _NATIVE_CONTEXT)


    # Maintainer notes (moved from the docstring): the kernel computes squared
    # Euclidean distances and the square root is taken on the way out, over
    # n_queries * k values. The index stays on the device (DEVIATION 2921,
    # `neighbors/resident_index.mojo`): the resident search is
    # `knn_search`'s body after its upload, so the bits are `knn_search`'s.
    # The copy is keyed on the index array's address and shape, released at
    # refit and when the instance is collected. The CPU host binding has no
    # resident entry and reads the caller's memory on every call.
    @_native_query_method
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
        return _NATIVE_QUERY(self, "kneighbors", (X, n_neighbors, return_distance), _QUERY_CONTEXT)


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
        return _ARCHIVE(self, "archive_save", (path,), _ARCHIVE_CONTEXT)

    @classmethod
    @_native_query_method
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
        return _ARCHIVE(cls, "host_archive_load", (path,), _ARCHIVE_CONTEXT)

    @_native_query_method
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
        return _NATIVE(self, "family_xn_knn_clf_fit", (None, X, y), _NATIVE_CONTEXT)


    @property
    def classes_(self):
        """Sorted unique labels seen at `fit`, as a list of ints; a list of
        such lists for a 2-D `y`."""
        return _NATIVE(self, "family_xn_knn_clf_classes", (), _NATIVE_CONTEXT)

    @_native_query_method
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
        return _NATIVE(self, "family_xn_knn_clf_predict", (None, X), _NATIVE_CONTEXT)


    @_native_query_method
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
        return _NATIVE(self, "family_xn_knn_clf_predict_proba", (None, X), _NATIVE_CONTEXT)

    @_native_query_method
    def score(self, X, y, sample_weight=None):
        """Mean accuracy of ``predict(X)`` against ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels.
        sample_weight : array-like of shape (n_samples,), default None
            Sample weights.

        Returns
        -------
        float
            Fraction of samples predicted correctly (weighted when
            ``sample_weight`` is given).
        """
        return _NATIVE(self, "family_xn_knn_clf_score", (None, X, y, sample_weight),
                       _NATIVE_CONTEXT)


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
        return _ARCHIVE(self, "archive_save", (path,), _ARCHIVE_CONTEXT)

    @classmethod
    @_native_query_method
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
        return _ARCHIVE(cls, "host_archive_load", (path,), _ARCHIVE_CONTEXT)

    @_native_query_method
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
        return _NATIVE(self, "family_xn_knn_reg_fit", (None, X, y), _NATIVE_CONTEXT)


    @_native_query_method
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
        return _NATIVE(self, "family_xn_knn_reg_predict", (None, X), _NATIVE_CONTEXT)

    @_native_query_method
    def score(self, X, y, sample_weight=None):
        """Coefficient of determination R^2 of ``predict(X)`` against ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True targets.
        sample_weight : array-like of shape (n_samples,), default None
            Sample weights.

        Returns
        -------
        float
            R^2, averaged uniformly over outputs; 1.0 is a perfect fit.
        """
        return _NATIVE(self, "family_xn_knn_reg_score",
                       (None, X, y, sample_weight, None), _NATIVE_CONTEXT)


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
        # Native (bindings/host_archive_native.mojo
        # `host_archive_radius_refusals`), the check `fit` and `load` run.
        _NATIVE(self, "family_xn_radius_refusals", (), _NATIVE_CONTEXT)

    # The ball cover is not built at fit, a known cost rather than an
    # oversight: `neighbors/estimator.mojo`'s RADIUS NEIGHBOURS banner
    # records it.
    @_native_query_method
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
        return _NATIVE(self, "family_xn_radius_fit", (X, y), _NATIVE_CONTEXT)

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
        return _ARCHIVE(self, "archive_save", (path,), _ARCHIVE_CONTEXT)

    @classmethod
    @_native_query_method
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
        return _ARCHIVE(cls, "host_archive_load", (path,), _ARCHIVE_CONTEXT)

    # DEVIATION 2375: the containers are Python lists of Arrays, where they
    # were object-dtype ndarrays of ndarrays. The self-edge under X=None is
    # kept because the ball cover returns it, DBSCAN counts on it, and
    # dropping it would make this surface disagree with the CSR every other
    # consumer sees.
    @_native_query_method
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
        return _NATIVE_QUERY(self, "radius_neighbors", (X, radius, return_distance, sort_results), _QUERY_CONTEXT)


_native_archive_bridge.register(
    RadiusNeighbors, _RADIUS_FORMAT, (
        ("need_set", "_index", "RuntimeError", "this estimator is not fitted yet"),
        *_native_archive_bridge.head(_RADIUS_FORMAT),
        ("str", "metric", "metric"),
        ("str", "algorithm", "algorithm"),
        ("f8", "radius", (("float", "radius"),)),
        ("f8", "p", (("float", "p"),)),
        ("array", "index", "_index"),
        ("i8", "meta", (("int", "n_features_in_"), ("int", "n_samples_fit_"))),
    ),
    host_archive_neighbor_metrics=dict(_METRIC_TABLE),
    host_archive_rbc_metrics=dict(_RBC_METRIC_TABLE),
    host_archive_unsupported_metrics=tuple(_UNSUPPORTED_METRICS),
)


# The k-NN classes the native refusals tell apart from NearestNeighbors
# (`isinstance(self, (KNeighborsClassifier, KNeighborsRegressor))`).
_native_family_bridge.tables(family_xn_knn_classes=(KNeighborsClassifier, KNeighborsRegressor))


def _knn_schema(*extra, require=()):
    """The k-NN save schema (the 0.8.55 `_knn_arrays` members, in order):
    the refusals of the class's `save`, the head, the names, `p`, the index
    and `meta` [n_features_in_, n_samples_fit_, n_neighbors, query_tile,
    outputs_2d, n_outputs], then the class's own members."""
    return (
        *require,
        ("need_set", "_index", "RuntimeError", "this estimator is not fitted yet"),
        *_native_archive_bridge.head(_KNN_FORMAT),
        ("str", "metric", "metric"),
        ("str", "algorithm", "algorithm"),
        ("when_none", "weights", (("format", "weights", "uniform"),)),
        ("when_set", "weights", (("str", "weights", "weights"),)),
        ("f8", "p", (("float", "p"),)),
        ("array", "index", "_index"),
        ("i8", "meta", (("int", "n_features_in_"), ("int", "n_samples_fit_"), ("int", "n_neighbors"),
                        ("int", "query_tile"), ("bool_or", "outputs_2d_", False),
                        ("none_or", "_y_cols", 1, ("dim", "_y_cols", 0)) if extra else ("const", 1))),
        *extra,
    )


_native_archive_bridge.register(NearestNeighbors, _KNN_FORMAT, _knn_schema())
_native_archive_bridge.register(KNeighborsClassifier, _KNN_FORMAT, _knn_schema(
    ("array", "y_cols", "_y_cols"),
    ("flatten", "classes", "class_counts", "<i8", "_classes_list"),
    require=(("need_set", "_y_cols", "RuntimeError", "this estimator is not fitted yet"),
             ("refuse_set", "_label_names", "ValueError",
              "mojolearn: save stores integer class labels; this model was fit on str labels")),
))
_native_archive_bridge.register(KNeighborsRegressor, _KNN_FORMAT, _knn_schema(
    ("array", "y_cols", "_y_cols"),
    require=(("need_set", "_y_cols", "RuntimeError", "this estimator is not fitted yet"),),
))
# The device index is released by the native finalizer, through the
# minting binding's `knn_index_release`; the handle lives in the holder
# dictionary `_resident_holder["index"]`.
_native_archive_bridge.lifetime(NearestNeighbors, "knn_index_release", "_resident_holder")


from ._native_neighbor_queries import connect as _connect_queries
_NATIVE_QUERY, _QUERY_CONTEXT = _connect_queries(
    {NearestNeighbors: "knn", KNeighborsClassifier: "knn", KNeighborsRegressor: "knn", RadiusNeighbors: "radius"},
    _NATIVE_CONTEXT, _ARCHIVE_CONTEXT, ("base", "sort"))
_QUERY_CONTEXT["neighbor_query_classifier_classes"] = {KNeighborsClassifier: True}
# Radius fit and query read settings through the same connected native owner.
_native_family_bridge.tables(family_xn_radius_attributes=_QUERY_CONTEXT)

# scikit-learn keeps more estimators in this module than it defines; they resolve from
# the package at first use (`_sklearn_modules`), so `from mojolearn.neighbors import ...`
# takes the same names as scikit-learn's module.
from ._sklearn_modules import LAYOUT as _SKLEARN_LAYOUT, lazy_exports as _lazy_exports  # noqa: E402
__getattr__ = _lazy_exports("neighbors")

# Runtime binding selection belongs to the native family owner.
_native_family_bridge.connect_bindings("_mojolearn", "_mojolearn_x_neighbors", "_mojolearn_metrics")
