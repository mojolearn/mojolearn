# SPDX-License-Identifier: Apache-2.0
"""k-NN door RUN: the checks on typed values the NearestNeighbors and k-NN
classifier/regressor public methods make, their refusal wording, and the
typed host scans and copies the door used to run inline.

`bindings/neighbors_knn_native.mojo` is the UNPACK/PACK side: it reads the
estimator's fields, admits the arrays, resolves the metric tables and calls
these checks at the point the public method made them, so the refusal order
and wording are unchanged. Nothing here imports Python.
"""
from std.math import isfinite
from std.memory import unsafe_memcpy
from std.sys.compile import is_defined
from std.time import perf_counter_ns
from core.shared_slot import SharedSlot
from core.family_cast_impl import family_min_f32
from core.host_finite_scan import host_all_finite_f32, host_all_finite_f64, host_finite_status

#: cuVS `DistanceType` values the refusals and the input checks read
#: (python/mojolearn/neighbors.py `_DIST_LP_UNEXPANDED`, `_DIST_JENSEN_SHANNON`).
comptime KNN_DIST_LP = 9
comptime KNN_DIST_JS = 15
#: cuVS `DistanceType.CosineExpanded`: its zero-row refusal reads host query rows.
comptime KNN_DIST_COSINE = 2
comptime KNN_FLT_MIN = 1.1754943508222875e-38
#: `_array._NATIVE_CODE` and `_REDUCE_*` of the core helper `reduce_stat`.
comptime KNN_CODE_F32 = 0
comptime KNN_CODE_F64 = 1
comptime KNN_CODE_I32 = 2
comptime KNN_CODE_I64 = 3
comptime KNN_REDUCE_MIN = 0
comptime KNN_REDUCE_MAX = 1
comptime KNN_REDUCE_INTEGRAL = 4
#: `knn_minkowski_status` words.
comptime KNN_MINKOWSKI_OK = 0
comptime KNN_MINKOWSKI_NOT_POSITIVE = 1
comptime KNN_MINKOWSKI_SUBNORMAL = 2
comptime KNN_INT32_LOW = -(1 << 31)
comptime KNN_INT32_HIGH = (1 << 31) - 1
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]


# ------------------------------------------------------------------ refusals at fit


def knn_refuse_rbc_vote(cls_name: String) raises:
    """The k-NN classes refuse algorithm='rbc' (their vote has no index seam)."""
    raise Error("mojolearn " + cls_name + ": algorithm='rbc' is "
                + "refused. The ball cover's k-NN query is wired into "
                + "NearestNeighbors.kneighbors only; this class runs cuML's "
                + "knn_classify / knn_regress, which perform their own "
                + "brute-force search inside one call and have no seam to "
                + "hand an index to. Accepting 'rbc' here would be accepting "
                + "a parameter that changes nothing. Use "
                + "NearestNeighbors(algorithm='rbc').kneighbors and vote on "
                + "the result, or leave this at 'brute'.")


def knn_refuse_distance_inner_product(cls_name: String) raises:
    raise Error("mojolearn " + cls_name + ": weights='distance' is refused "
                + "with metric='inner_product': it weighs a neighbour by 1 / d, "
                + "and an inner product is a similarity (larger is nearer, and "
                + "it can be zero or negative), not a distance")


def knn_refuse_algorithm(algorithm_repr: String) raises:
    """Any algorithm but 'brute', 'auto' and 'rbc'."""
    raise Error("mojolearn NearestNeighbors: algorithm=" + algorithm_repr + " is "
                + "refused. Three are accepted: 'brute' (exact brute force), "
                + "'auto' (which means 'brute' here), and 'rbc' (exact k-NN "
                + "over a RANDOM BALL COVER, an index). "
                + "'kd_tree' and 'ball_tree' are refused, and it is an "
                + "ENGINEERING refusal rather than an attribution one: a "
                + "kd-tree query is a per-query stack walk with "
                + "data-dependent branching and divergent memory access, "
                + "which is the shape a GPU is worst at, and above roughly "
                + "15 dimensions the pruning bound stops firing and it "
                + "degenerates to a full scan, so it would lose to 'brute' "
                + "on exactly the workloads it claims to help. If what you "
                + "want is an index instead of a full scan, that is 'rbc': "
                + "it is EXACT, not approximate, it has no per-query stack, "
                + "and it is the same structure RadiusNeighbors uses "
                + "(neighbors/impl/ball_cover/knn.mojo).")


# ------------------------------------------------------------------ metric


#: cuVS `DistanceType` values (`cuvs/distance/distance.h:22-69`), mirrored
#: from `neighbors/impl/distance/detail/distance_ops.mojo`. The value gaps
#: are theirs. (Moved from python/mojolearn/neighbors.py `_DIST_*`, lane
#: py-neighbors 2026-11-14: the tables below are the only copy.)
comptime KNN_DIST_L2_EXPANDED = 0
comptime KNN_DIST_L2_SQRT_EXPANDED = 1
comptime KNN_DIST_L1 = 3
comptime KNN_DIST_L2_SQRT_UNEXPANDED = 5
comptime KNN_DIST_INNER_PRODUCT = 6
comptime KNN_DIST_LINF = 7
comptime KNN_DIST_CANBERRA = 8
comptime KNN_DIST_CORRELATION_EXPANDED = 10
comptime KNN_DIST_BRAY_CURTIS = 14
#: `sorted(_METRIC_TABLE)` joined by ", ": the refusals' "Implemented:" list.
comptime KNN_METRIC_NAMES: StaticString = (
    "braycurtis, canberra, chebyshev, cityblock, correlation, cosine, euclidean, inner_product, "
    "jensenshannon, l1, l2, linf, lp, manhattan, minkowski, sqeuclidean, taxicab"
)
#: `sorted(_RBC_METRIC_TABLE)` joined by ", ": the ball cover's admitted list.
comptime KNN_RBC_METRIC_NAMES: StaticString = (
    "chebyshev, cityblock, euclidean, l1, l2, linf, lp, manhattan, minkowski, taxicab"
)


def knn_metric_code(key: String) -> Int:
    """The brute-force table (cuML's `NearestNeighbors._build_metric_type`,
    `nearest_neighbors.pyx:520-553`, the rows this tree computes, plus the
    extra metrics of `distance_ops.mojo::extra_metric_cell`): the
    `DistanceType` of a lower-cased metric name, -1 when it is not a row.
    NOT the `pairwise_distances` table: cuML has two and they disagree about
    'euclidean' (that one sends it to L2SqrtUnexpanded, this one to
    L2SqrtExpanded)."""
    if key == "euclidean" or key == "l2":
        return KNN_DIST_L2_SQRT_EXPANDED
    if key == "sqeuclidean":
        return KNN_DIST_L2_EXPANDED
    if key == "cityblock" or key == "l1" or key == "manhattan" or key == "taxicab":
        return KNN_DIST_L1
    if key == "chebyshev" or key == "linf":
        return KNN_DIST_LINF
    if key == "cosine":
        return KNN_DIST_COSINE
    if key == "minkowski" or key == "lp":
        return KNN_DIST_LP
    if key == "canberra":
        return KNN_DIST_CANBERRA
    if key == "braycurtis":
        return KNN_DIST_BRAY_CURTIS
    if key == "correlation":
        return KNN_DIST_CORRELATION_EXPANDED
    if key == "jensenshannon":
        return KNN_DIST_JS
    if key == "inner_product":
        return KNN_DIST_INNER_PRODUCT
    return -1


def knn_rbc_metric_code(key: String) -> Int:
    """THE BALL COVER'S OWN TABLE, and it is NOT `knn_metric_code`. Two rows
    differ, both for reasons about the INDEX and not the metric's name
    (`neighbors/impl/ball_cover/common.mojo`, DEVIATIONS 564 and 565):
    'euclidean'/'l2' goes to L2SqrtUNexpanded (the index stores radii
    computed by summing differences directly, and the query compares
    against them; DEVIATION 2 in `ball_cover.mojo`), and 'sqeuclidean' and
    'cosine' are absent (not metrics: the pruning IS the triangle
    inequality). -1 when the name is not a row."""
    if key == "euclidean" or key == "l2":
        return KNN_DIST_L2_SQRT_UNEXPANDED
    if key == "cityblock" or key == "l1" or key == "manhattan" or key == "taxicab":
        return KNN_DIST_L1
    if key == "chebyshev" or key == "linf":
        return KNN_DIST_LINF
    if key == "minkowski" or key == "lp":
        return KNN_DIST_LP
    return -1


def knn_metric_unsupported(key: String) -> Bool:
    """Names in cuML's `VALID_METRICS["brute"]` (`neighbors/__init__.py:27-48`)
    this tree does not compute, refused BY NAME so a caller learns the metric
    is UNIMPLEMENTED rather than unknown. haversine needs arcsin, and
    `checks/numerics.mojo` carries no pinned arcsin: a vendor asin is a
    last-bit vendor choice (IDENTITY_PATHS row 12)."""
    return key == "haversine"


def knn_metric_resolve(cls_name: String, key: String, metric_repr: String) raises -> Int:
    """The brute-force `DistanceType` of the lower-cased name `key`, or the
    refusal by name (unimplemented row of cuML's table, else unknown)."""
    var value = knn_metric_code(key)
    if value >= 0:
        return value
    if knn_metric_unsupported(key):
        knn_refuse_metric_unimplemented(cls_name, metric_repr, String(KNN_METRIC_NAMES))
    knn_refuse_metric_unknown(cls_name, metric_repr, String(KNN_METRIC_NAMES))
    return -1


def knn_rbc_metric_resolve(cls_name: String, key: String, metric_repr: String) raises -> Int:
    """The ball cover's `DistanceType` of the lower-cased name `key`, or the
    refusal by name, in `neighbors._resolve_rbc_metric`'s order and wording."""
    var head = "mojolearn " + cls_name + ": "
    if key == "sqeuclidean":
        raise Error(head + "metric='sqeuclidean' is REFUSED on the "
            + "random ball cover. The pruning here IS THE TRIANGLE "
            + "INEQUALITY on the landmark radii "
            + "(neighbors/impl/ball_cover/common.mojo, DEVIATION "
            + "565), and SQUARED Euclidean distance does not satisfy it even "
            + "though Euclidean distance does: on three collinear points at "
            + "unit spacing it gives 4 against 1 + 1. A cover built on it "
            + "would prune away true neighbours silently. Ask for "
            + "metric='euclidean' and square the returned distances, which "
            + "is the same answer and is exact.")
    if key == "cosine":
        raise Error(head + "metric='cosine' is REFUSED on the "
            + "random ball cover. The pruning here IS THE TRIANGLE "
            + "INEQUALITY on the landmark radii, and `1 - cos` does not "
            + "satisfy it: three unit vectors at 0, 60 and 120 degrees give "
            + "1.5 against 0.5 + 0.5. It also fails identity of "
            + "indiscernibles, d(x, 2x) = 0, so a landmark radius stops "
            + "bounding its ball. A cover built on it would prune away true "
            + "neighbours silently rather than return them slowly. Use "
            + "NearestNeighbors with algorithm='brute', which needs no "
            + "inequality and honors every implemented metric.")
    if knn_metric_unsupported(key):
        raise Error(head + "metric=" + metric_repr + " is in cuML's "
            + "VALID_METRICS['brute'] but is NOT IMPLEMENTED "
            + "(neighbors/NOT_IMPLEMENTED.tsv)")
    var value = knn_rbc_metric_code(key)
    if value < 0 and knn_metric_code(key) >= 0:
        raise Error(head + "metric=" + metric_repr + " is computed by brute "
            + "force only (NearestNeighbors(algorithm='brute')). The random "
            + "ball cover's pruning is the triangle inequality on its "
            + "landmark radii and its kernels carry the Minkowski family; "
            + "cuML's own VALID_METRICS['rbc'] is {euclidean, haversine, l2}")
    if value < 0:
        raise Error(head + "unknown metric " + metric_repr + ". The random "
            + "ball cover admits: " + String(KNN_RBC_METRIC_NAMES))
    return value


def knn_refuse_metric_name(cls_name: String, metric_repr: String) raises:
    raise Error("mojolearn " + cls_name + ": metric must be a name, got " + metric_repr + ". "
                + "A callable metric is not implemented: it would have to run inside a GPU kernel.")


def knn_refuse_metric_unimplemented(cls_name: String, metric_repr: String, implemented: String) raises:
    raise Error("mojolearn " + cls_name + ": metric=" + metric_repr + " is in cuML's "
                + "VALID_METRICS['brute'] but is NOT IMPLEMENTED "
                + "(neighbors/NOT_IMPLEMENTED.tsv). Implemented: " + implemented)


def knn_refuse_metric_unknown(cls_name: String, metric_repr: String, implemented: String) raises:
    raise Error("mojolearn " + cls_name + ": unknown metric " + metric_repr + ". Implemented: " + implemented)


def knn_refuse_inert_p(cls_name: String, metric_repr: String, p_repr: String) raises:
    """`neighbors._refuse_inert_p`: only metric='minkowski'/'lp' reads p."""
    raise Error("mojolearn " + cls_name + ": p is read only by metric='minkowski' "
                + "(or 'lp'); metric=" + metric_repr + " never reads it, so p=" + p_repr
                + " would change nothing. Leave p at its default (2) or ask for metric='minkowski'.")


def knn_minkowski_status(arg: Float64) -> Int:
    """The Minkowski p's verdict: `not (arg > 0.0) or arg != arg or arg ==
    float("inf")` is NOT_POSITIVE, a float32-subnormal p is SUBNORMAL."""
    if not (arg > 0.0) or arg != arg or (arg > 0.0 and not isfinite(arg)):
        return KNN_MINKOWSKI_NOT_POSITIVE
    if arg < KNN_FLT_MIN:
        return KNN_MINKOWSKI_SUBNORMAL
    return KNN_MINKOWSKI_OK


def knn_refuse_minkowski(cls_name: String, status: Int, p_repr: String) raises:
    """The refusal of a non-OK `knn_minkowski_status` word; OK returns."""
    if status == KNN_MINKOWSKI_NOT_POSITIVE:
        raise Error("mojolearn " + cls_name + ": metric='minkowski' needs a finite "
                    + "p > 0, got " + p_repr + ". p = 0 makes 1/p infinite, p < 0 is not "
                    + "a metric, and p = infinity is metric='chebyshev'.")
    if status == KNN_MINKOWSKI_SUBNORMAL:
        raise Error("mojolearn " + cls_name + ": metric='minkowski' p=" + p_repr + " is "
                    + "subnormal in float32; the flush policy differs by vendor "
                    + "so this cannot be one arithmetic (DEVIATION 552)")


def knn_check_metric_input(cls_name: String, metric: Int, x_address: Int, count: Int) raises:
    """Jensen-Shannon takes logarithms: X >= 0 (the native minimum of the
    admitted float32 block)."""
    if (metric == KNN_DIST_JS and count > 0
            and Float64(family_min_f32(_F32P(unsafe_from_address=x_address), count)) < 0.0):
        raise Error("mojolearn " + cls_name + ": metric='jensenshannon' needs X >= 0 "
                    + "(it takes logarithms of the coordinates)")


def knn_metric_reads_host_queries(metric: Int) -> Bool:
    """Cosine's zero-row refusal and Jensen-Shannon's sign check read host
    query words, so a device-gathered fold cannot serve them."""
    return metric == KNN_DIST_JS or metric == KNN_DIST_COSINE


# ------------------------------------------------------------------ weights


def knn_refuse_callable_weights(cls_name: String) raises:
    raise Error("mojolearn " + cls_name + ": weights=<callable> is NOT IMPLEMENTED. "
                + "scikit-learn calls it on the distance matrix in Python; the "
                + "vote here runs in a GPU kernel and there is no portable way "
                + "to lift a Python function into one. Use 'uniform' or "
                + "'distance'.")


def knn_refuse_weights(cls_name: String, weights_repr: String) raises:
    raise Error("mojolearn " + cls_name + ": weights=" + weights_repr + " is not a "
                + "weighting; use 'uniform' or 'distance'")


def knn_weights_code(name: String) -> Int:
    """'uniform' 0, 'distance' 1, any other name -1."""
    if name == "uniform":
        return 0
    if name == "distance":
        return 1
    return -1


# ------------------------------------------------------------------ fit


def knn_check_y_rank(rank: Int) raises:
    if rank != 1 and rank != 2:
        raise Error("mojolearn: y must be 1-D or 2-D, got " + String(rank) + "-D")


def knn_check_y_rows(rows: Int, n_fit: Int) raises:
    if rows != n_fit:
        raise Error("mojolearn: y has " + String(rows) + " rows, X has " + String(n_fit))


def knn_check_y_ndim(ndim: Int, rank: Int) raises:
    if ndim != rank:
        raise Error("mojolearn: y must be " + String(rank) + "-D, got " + String(ndim) + "-D")


def knn_int32_low_fits(low: Int) -> Bool:
    """True when the labels' minimum fits int32 (the maximum is then read)."""
    return low >= KNN_INT32_LOW


def knn_check_int32_labels(low: Int, high: Int) raises:
    if low < KNN_INT32_LOW or high > KNN_INT32_HIGH:
        raise Error("mojolearn: y does not fit int32")


#: wave2-knn-0844-gap (2026-11-13): B arm of the owed A/B. Defined, the
#: classifier fit widens int32 labels to int64, reads their minimum and maximum
#: (two device reductions: two uploads of the label words, two launches, two
#: synchronisations) and narrows them back, as every release since 0.8.44 did.
comptime KNN_I32_LABEL_RANGE_OLD = is_defined["MOJOLEARN_IDN_KNN_I32_LABEL_RANGE_OLD"]()


def knn_labels_already_int32(typestr: String) -> Bool:
    """True when labels of dtype `typestr` are int32 words already: every value
    fits int32, so the classifier fit's range check cannot refuse them and the
    int64 widening it reads is skipped. The label columns are a plain copy of
    the same words (the widen-then-narrow round trip is the identity on them),
    so the bits do not change. Every other dtype keeps the widening and the
    range check."""
    comptime if KNN_I32_LABEL_RANGE_OLD:
        return False
    return typestr == "<i4"


def _order_byte(c: Int) -> Bool:
    return (c == ord("@") or c == ord("=") or c == ord("<") or c == ord(">")
            or c == ord("!") or c == ord("|"))


def knn_float_format(fmt: String) -> Bool:
    """`fmt.lstrip("@=<>!|")` is a float format: f, d, e, g, f2, f4 or f8."""
    var bytes = fmt.as_bytes()
    var n = len(bytes)
    var start = 0
    while start < n and _order_byte(Int(bytes[start])):  # driver-loop: over a buffer format's byte-order prefix
        start += 1
    var rest = n - start
    if rest == 1:
        var c = Int(bytes[start])
        return c == ord("f") or c == ord("d") or c == ord("e") or c == ord("g")
    if rest == 2:
        var d = Int(bytes[start + 1])
        return Int(bytes[start]) == ord("f") and (d == ord("2") or d == ord("4") or d == ord("8"))
    return False


def knn_labels_finite(address: Int, count: Int, wide: Bool) -> Bool:
    """Every float label is finite (`wide`: float64 words, else float32);
    an empty block is finite."""
    if count <= 0:
        return True
    if wide:
        return host_all_finite_f64(_F64P(unsafe_from_address=address), count)
    return host_all_finite_f32(_F32P(unsafe_from_address=address), count)


def knn_labels_skip_integral(typestr: String, count: Int, longest: Int) -> Bool:
    """`whole_float_labels` answers True without the integral test: not a
    float32/float64 block, empty, or not one flat axis."""
    return (typestr != "<f4" and typestr != "<f8") or count == 0 or count != longest


# ------------------------------------------------------------------ predict


def knn_finite_status(address: Int, count: Int) -> Int:
    """The CPU column's (and RBC's) finite test of `count` borrowed float32
    words: 0 when every word is finite."""
    # parallel: host_finite_status splits the words across the host's cores
    return Int(host_finite_status(address, count))


def knn_check_width(columns: Int, features: Int) raises:
    if columns != features:
        raise Error("mojolearn: X has " + String(columns) + " features, index has " + String(features))


def knn_proba_rows(words: Int, total_classes: Int) -> Int:
    """Query rows of the flat probability buffer."""
    return words // total_classes


def knn_copy_proba_block(dst_address: Int, src_address: Int, offset: Int, rows: Int, classes: Int):
    """One output's (rows, classes) float32 block, copied from word `offset`
    of the flat probability buffer."""
    unsafe_memcpy(dest=_F32P(unsafe_from_address=dst_address),
                  src=_F32P(unsafe_from_address=src_address + 4 * offset), count=rows * classes)


# ------------------------------------------------------------------ door switches and timers

#: fix-knn-door-fixed-cost (2026-11-14): B arm of the owed A/B (row id
#: fix-knn-door-fixed-cost). Defined, every KNeighborsClassifier /
#: KNeighborsRegressor fit resolves and admits the x_neighbors binding up front
#: (estimator_arguments + xn_admit_arguments, the form b412ea02f4 introduced
#: with the native door), whether or not the fit uses it. Default (0.8.55's
#: form, neighbors.py `self._bind(_XN)` only inside the multi-output branch):
#: the binding is resolved only where a fit runs its one consumer, the
#: multi-output label transpose (`n_outputs > 1`). Same kernels, same words:
#: no bit change.
comptime KNN_DOOR_FIT_XN_EAGER = is_defined["MOJOLEARN_IDN_KNN_DOOR_FIT_XN_EAGER"]()


def knn_fit_xn_eager() -> Bool:
    """True when the k-NN fit door resolves the x_neighbors binding before
    its work (the B arm); False when only the multi-output transpose
    resolves it (the default, 0.8.55's form)."""
    return KNN_DOOR_FIT_XN_EAGER


#: fix-knn-door-fixed-cost (2026-11-14): the door phases of the k-NN public
#: operations under the existing `-D MOJOLEARN_KNN_PHASE_TIMERS=1`
#: (neighbors/impl/detail/knn_brute_force.mojo KNN_PHASE_TIMERS, which times
#: the device request's own phases). Defined, every public k-NN operation
#: prints one line `KNN_PHASE_TIMERS door <operation> <phase>_ms <ms> ...
#: total_ms <ms>`: host wall time between consecutive marks, in program order,
#: so the segments sum to the total. Undefined, every call below is an empty
#: body after `comptime if`: no clock read, no slot, no string.
comptime KNN_DOOR_TIMERS = is_defined["MOJOLEARN_KNN_PHASE_TIMERS"]()


struct KnnDoorClock(Defaultable, Movable):
    var start: Int
    var last: Int
    var line: String

    def __init__(out self):
        self.start = 0
        self.last = 0
        self.line = String()


comptime _KNN_DOOR_CLOCK = SharedSlot[KnnDoorClock, "MojolearnKnnDoorClockV1"]


def knn_door_clock_open(operation: String) raises:
    """Start one public operation's door line (timers on only)."""
    comptime if KNN_DOOR_TIMERS:
        var clock = _KNN_DOOR_CLOCK.get_or_create_ptr()
        var now = Int(perf_counter_ns())
        clock[].start = now
        clock[].last = now
        clock[].line = String("KNN_PHASE_TIMERS door ") + operation


def knn_door_clock_mark(phase: StaticString) raises:
    """Close the segment since the last mark under `phase` (timers on only)."""
    comptime if KNN_DOOR_TIMERS:
        var clock = _KNN_DOOR_CLOCK.get_or_create_ptr()
        var now = Int(perf_counter_ns())
        var ms = Float64(now - clock[].last) / 1000000.0
        clock[].line += String(" ") + String(phase) + String("_ms ") + String(ms)
        clock[].last = now


def knn_door_clock_close() raises:
    """Print the operation's door line (timers on only)."""
    comptime if KNN_DOOR_TIMERS:
        var clock = _KNN_DOOR_CLOCK.get_or_create_ptr()
        var now = Int(perf_counter_ns())
        var ms = Float64(now - clock[].start) / 1000000.0
        clock[].line += String(" total_ms ") + String(ms)
        print(clock[].line)
        clock[].line = String()
