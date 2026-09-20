# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Brute-force k-NN inference on the host, for a box with no GPU (the knn
host inference lane, 2026-09-14; brief
docs/lanes/BRIEF_forest_host_inference_2026-09-13.md, "Classical lanes",
rank 8).

HOST ONLY. Nothing here imports `max.gpu`, `std.gpu` or a `DeviceContext`,
and the GPU bindings do not import this file. It exists because the three
entries of `bindings/_mojolearn.mojo` that NearestNeighbors.kneighbors,
KNeighborsClassifier.predict / predict_proba and KNeighborsRegressor.predict
call (`knn_search`, `knn_classify`, `knn_regress`) reach
`neighbors/estimator.mojo` (imports `max.gpu.host` at `:122`) and, under
it, kernels in files that import `std.gpu` at module level, while the
arithmetic each kernel performs is `checks/numerics.mojo` calls plus
integer bookkeeping, which is GPU-free.

WHAT IS RESTATED, AND WHERE THE ORIGINAL IS. Every function below names the
kernel it mirrors and keeps its statements in its order. The GPU path under
IDENTICAL is `knn_search_traced` -> `brute_force_knn_impl` with AUTO pinned
to the TILED arm on every column (DEVIATION 509, `neighbors/impl/detail/
knn_brute_force.mojo:1391-1440`), and the tiled arm's three distance
spellings (row-major scalar, transposed scalar, transposed register tile)
and three selectors (small-k composite key, radix rank pass, the fused
distance-and-select launch) are all written to the SAME contract, which is
what this file states once:

  `host_row_norm`          `row_norm_kernel`, `core/row_norms.mojo:66-113`,
                           `take_sqrt = 0` (the L2 expanded pair wants the
                           SQUARED norm, `compute_norms_for_metric`,
                           `knn_brute_force.mojo:346-380`): NORM_TPB strided
                           partials `acc = ftz(identical_mul_add(v, v, acc))`
                           over `v = ftz(a[col])`, then `pinned_block_sum`'s
                           halving tree `red[t] = red[t] + red[t + step]`,
                           `step = NORM_TPB/2 .. 1` (`core/pinned_reduce.mojo:
                           95-125`; the two-phase form there is the same
                           additions in the same order), then `ftz(total)`.
                           NORM_TPB is `lib_block_size_for[K_LIB_ROW_NORM]`,
                           a NUMERIC row that resolves to the bit-identical
                           column under IDENTICAL, so it is 128 on Apple,
                           NVIDIA, AMD and here (asserted below).
  `host_l2_expanded_cell`  `pinned_distance_tile_kernel`, `neighbors/checks/
                           pinned_distance_tile.mojo:66-107`: the feature
                           chain `acc = ftz(identical_mul_add(ftz(q[f]),
                           ftz(y[f]), acc))` ascending, then
                           `dist = ftz(identical_mul_add(-2.0, acc,
                           ftz(ftz(qn) + ftz(yn))))`, clamp `dist <= 0 ->
                           0`, `ftz(identical_sqrt(dist))` when the metric
                           roots. The register-tile kernel (`:109-230`) and
                           the fused launch (`neighbors/checks/
                           fused_distance_select_identical.mojo:100-110`)
                           spell the same chain with a per-column repair
                           whose documented purpose is to return THIS
                           value (a correctly rounded fma, then the flush).
  `host_composite_key`     `composite_key`, `neighbors/checks/
                           select_radix_identical.mojo:102-119`, over
                           `twiddle_in` (`neighbors/impl/matrix/detail/
                           select_radix.mojo:155-170`, `select_min`):
                           distance bits, order-twiddled, in the high half
                           and the column index in the low half.
  `host_select_k`          the k smallest keys of a row, which is what the
                           small-k selector (`select_smallk_identical_
                           candidate.mojo:85-135`, carry insertion into a
                           k-list), the radix rank pass and the partial
                           merge over index tiles all return, then the
                           estimator's own host insertion sort by
                           `(distance, index)` (`neighbors/estimator.mojo:
                           664-682`), whose compare is `db < dv or (db == dv
                           and ib <= iv)`.
  `host_unique_labels`     `getUniquelabels`, `neighbors/impl/label/
                           classlabels.mojo:62-92` (DEVIATION 541, a host
                           sort and unique already).
  `host_monotonic`         `make_monotonic` over `[y; uniq]` then the
                           `_subtract_one_kernel` (`neighbors/impl/
                           selection/knn.mojo:333-347`): each label's
                           position in the sorted unique set.
  `host_class_probs`       `class_probs_kernel`, `knn.mojo:119-148`:
                           `n_neigh_inv = ftz(1.0 / k)`, then per slot in
                           slot order `out[row, label] = ftz(out + n_neigh_inv)`.
  `host_class_vote`        `class_vote_kernel`, `knn.mojo:151-180`: strict
                           `>` against `cur_max = -1.0`, lowest class wins
                           a tie, the ORIGINAL label written.
  `host_regress_avg`       `regress_avg_kernel`, `knn.mojo:183-211`:
                           `pred = ftz(pred + ftz(y[nbr]))` then
                           `ftz(pred / k)`.
  `host_distance_weights`  `neighbors/impl/selection/distance_weights.mojo:
                           152-244`, already host code (DEVIATION 554),
                           relocated because its file imports `std.gpu`.
  `host_weighted_class_probs`, `host_weighted_regress_avg`
                           `weighted_class_probs_kernel` (`:247-306`) and
                           `weighted_regress_avg_kernel` (`:309-348`) of
                           the same file.

THE FOUR UNEXPANDED AND COSINE METRICS (lane/cpu-training-batch3,
2026-09-14, lanes knn-manhattan, knn-chebyshev, knn-minkowski-p3 and
knn-cosine). Under IDENTICAL every metric but the L2 expanded pair reaches
`metric_distance_kernel` (`neighbors/impl/distance/detail/distance_ops.mojo`)
from `tiled_brute_force_knn` (`knn_brute_force.mojo`, "THEIR `else` AT
`:224`" and the cosine arm beside it): one cell per thread, the feature
axis ascending, `ftz` on each loaded operand, then the op's core and
epilogue. `host_metric_cell` calls THE SAME CORES (`l1_core`, `linf_core`,
`lp_unexp_core` and `lp_unexp_epilog`, `inner_product_core` and
`cosine_epilog`), which are host-callable `@always_inline` functions, the
way the ball cover calls them (`neighbors/impl/ball_cover/common.mojo`,
DEVIATION 564). Cosine's norm is `cosine_row_norm_kernel`'s: the row norm
tree above with the clamp and `identical_sqrt`. The refusals are
`knn_search_traced`'s, before any distance: `validate_metric_arg`
(DEVIATION 552) and the all-zero cosine row (DEVIATION 553), index first.
The selection and the sort are the L2 pair's, unchanged; the answer is a
top-k under a total order, so the tiling that differs by column moves no
bit.

THE BALL COVER'S TWO QUERIES, AS AN EXHAUSTIVE SCAN (lane/cpu-training-batch3,
2026-09-14, lanes radius, radius-manhattan, radius-chebyshev,
radius-minkowski-p3 and knn-rbc). `radius_neighbors_count` /
`radius_neighbors_fill` and `rbc_knn_search` (`neighbors/estimator.mojo`)
build a random ball cover and PRUNE with the triangle inequality; the
pruning is exact (`neighbors/impl/ball_cover/knn.mojo`, "THE BOUNDS, AND THE
PROOF THAT EACH ONE IS EXACT"; DEVIATION 567's slack on the k-NN arm), so
the answer is a function of the comparison-space distances alone and not of
the landmark draw. The host therefore computes EVERY pair and keeps what the
cover keeps:
  `host_rbc_cmp_dist`      `rbc_cmp_dist` (`common.mojo`, DEVIATION 564):
                           the Euclidean arm is `eps_dist_sq` (`diff =
                           ftz(ftz(a) - ftz(b))`, `acc = ftz(fma(diff, diff,
                           acc))`), the other three call the same op cores
                           as `metric_distance_kernel`, query first.
  `host_rbc_radius_row`    a query's eps row: every index column `c`
                           ascending with `host_rbc_cmp_dist(q, x_c) <=
                           rbc_cmp_bound(metric, eps)`, which is the fill
                           kernel's membership test and DEVIATION 551's
                           canonical column order.
  `host_rbc_edge_distance` `rbc_edge_distance_kernel` (`neighbors/checks/
                           radius_distances.mojo`): the same distance
                           recomputed against the ORIGINAL rows,
                           `rbc_true_dist` when `return_sqrt`.
  `host_rbc_knn_row`       the k smallest `(comparison-space distance,
                           index)` pairs ascending, the total order at the
                           top of `knn.mojo`, reported as `rbc_true_dist`.
The distance count `rbc_knn_search` returns is the scan's, `n_queries *
n_index`; no lane hashes it.

WHAT IS NOT HERE. The L2 unexpanded metrics on the brute arm, which no
public metric name reaches (`neighbors.py::_METRIC_TABLE`). The host binding
refuses those BY NAME; nothing here computes something else under their
name.

The restatement is a prediction until measured. tools/classical_host_gate.py
(lanes knn, knn-clf, knn-reg) is the measurement, and the brief records
what it has shown.
"""
from std.memory import bitcast
from std.sys.compile import is_defined

from max.algorithm import sync_parallelize

from core.host_predict_threads import (
    HostF32Ptr,
    host_list_ptr,
    host_list_ptr_u32,
    host_predict_chunk,
    host_predict_task_count,
)
from checks.kernel_matrix import (
    COLUMN_AMD,
    COLUMN_APPLE,
    COLUMN_CPU,
    COLUMN_NVIDIA,
    K_LIB_ROW_NORM,
    lib_block_size_for,
)
from checks.numerics import ftz, identical_div, identical_mul_add, identical_sqrt
from neighbors.impl.distance.detail.distance_ops import (
    DIST_COSINE_EXPANDED,
    DIST_L1,
    DIST_LINF,
    DIST_LP_UNEXPANDED,
    cosine_epilog,
    cosine_zero_norm_row,
    inner_product_core,
    l1_core,
    linf_core,
    lp_unexp_core,
    lp_unexp_epilog,
    validate_metric_arg,
)
from neighbors.impl.ball_cover.common import (
    rbc_cmp_bound,
    rbc_true_dist,
)


#: The gate's negative control, the same define the classical and phase 1
#: host bindings read (`core/classical_host_predict.mojo:63`). A build with
#: it is wrong on purpose in four places, one per thing the brief names as
#: the work: every distance dot product's feature chain walks DESCENDING
#: (a serial float32 fold in the other order is a different bit pattern
#: on almost every cell), the selection key admits the HIGHER index at a
#: tied boundary, the vote's tie goes to the LAST maximal class, and the
#: regressor's mean folds its slots descending. The gate must catch each
#: of them. Read back by `core_host_sabotage`.
#:
#: THE VALUE ARMS (lane/ties-sabotage, 2026-09-15). A fold walked in the
#: other order is EXACT on the integer-grid `ties` fixture, so on that
#: fixture the order arms above moved no bit of knn-cosine, knn-rbc, radius
#: or radius-manhattan (the neighbors and density inference lane's sabotage
#: column and saved-model check). The same build therefore also moves a
#: VALUE the caller reads: `host_sabotage_value_flip` on every L1, Lp and
#: cosine cell distance of the brute arm, on every ball cover edge distance
#: and on every ball cover k-NN distance it reports. Production builds
#: compile none of it.
comptime KNN_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()


@always_inline
def host_sabotage_value_flip(v: Float32) -> Float32:
    """The value arm of the sabotage build: a float32 whose bits always
    differ from `v`'s. A magnitude below the smallest normal (either zero,
    or a subnormal) becomes the smallest positive normal, which no flush
    folds back; every other value steps its mantissa by one unit. Compiled
    only under KNN_HOST_SABOTAGE; the caller guards it."""
    var bits = bitcast[DType.uint32](v)
    if (bits & UInt32(0x7FFFFFFF)) < UInt32(0x00800000):
        return bitcast[DType.float32](UInt32(0x00800000))
    return bitcast[DType.float32](bits + UInt32(1))

#: `NORM_TPB` of `core/row_norms.mojo`, read through the same accessor.
#: `K_LIB_ROW_NORM` is a NUMERIC row (`lib_block_bounds_a_float_fold`), so
#: under IDENTICAL `lib_block_size_for` resolves it on the bit-identical
#: column whatever `column` says; the asserts in `host_row_norm` make that
#: a compile-time fact rather than a sentence.
comptime KNN_HOST_NORM_TPB = lib_block_size_for[K_LIB_ROW_NORM, COLUMN_CPU]()

#: cuVS `DistanceType` values (`neighbors/impl/distance/detail/distance_ops.
#: mojo:212-219`), the two this file computes, plus the sentinel every
#: pre-metric caller passes (`knn_brute_force.mojo:297`).
comptime KNN_HOST_DIST_L2_EXPANDED = 0
comptime KNN_HOST_DIST_L2_SQRT_EXPANDED = 1
comptime KNN_HOST_METRIC_FROM_IS_SQRT = -1

#: `distance_weights.mojo:129-130`.
comptime KNN_HOST_WEIGHTS_UNIFORM = 0
comptime KNN_HOST_WEIGHTS_DISTANCE = 1


def host_resolve_metric(metric: Int, is_sqrt: Bool) raises -> Int:
    """`resolve_metric` (`knn_brute_force.mojo:300-312`) narrowed to what
    this file computes: the sentinel picks the L2 expanded member `is_sqrt`
    names, the two members pass through, and every other cuVS value is
    refused BY NAME as having no CPU implementation (the GPU binding
    computes six more; see the module docstring)."""
    if metric == KNN_HOST_METRIC_FROM_IS_SQRT:
        return KNN_HOST_DIST_L2_SQRT_EXPANDED if is_sqrt else KNN_HOST_DIST_L2_EXPANDED
    if metric == KNN_HOST_DIST_L2_EXPANDED or metric == KNN_HOST_DIST_L2_SQRT_EXPANDED:
        return metric
    if (
        metric == DIST_L1
        or metric == DIST_LINF
        or metric == DIST_LP_UNEXPANDED
        or metric == DIST_COSINE_EXPANDED
    ):
        return metric
    raise Error(
        "knn host: no CPU implementation of cuVS DistanceType value "
        + String(metric)
        + " yet; the host computes sqeuclidean (0), euclidean/l2 (1),"
        " cosine (2), manhattan (3), chebyshev (7) and minkowski (9)"
        " only (core/knn_host_predict.mojo)"
    )


def host_row_norm(x: List[Float32], row: Int, d: Int) -> Float32:
    """`row_norm_kernel` at `take_sqrt = 0` for one row: NORM_TPB strided
    partials, the halving tree, `ftz` on the total. The asserts hold the
    fold width to the three GPU columns' (see KNN_HOST_NORM_TPB)."""
    comptime assert KNN_HOST_NORM_TPB == lib_block_size_for[K_LIB_ROW_NORM, COLUMN_APPLE](), (
        "knn host: the row norm fold width differs from the Apple column's"
    )
    comptime assert KNN_HOST_NORM_TPB == lib_block_size_for[K_LIB_ROW_NORM, COLUMN_NVIDIA](), (
        "knn host: the row norm fold width differs from the NVIDIA column's"
    )
    comptime assert KNN_HOST_NORM_TPB == lib_block_size_for[K_LIB_ROW_NORM, COLUMN_AMD](), (
        "knn host: the row norm fold width differs from the AMD column's"
    )
    comptime assert (KNN_HOST_NORM_TPB & (KNN_HOST_NORM_TPB - 1)) == 0, (
        "knn host: the halving tree needs a power-of-two block"
    )
    var red = List[Float32](length=KNN_HOST_NORM_TPB, fill=Float32(0.0))
    for tid in range(KNN_HOST_NORM_TPB):
        var acc = Float32(0.0)
        var col = tid
        while col < d:
            var v = ftz(x[row * d + col])
            acc = ftz(identical_mul_add(v, v, acc))
            col += KNN_HOST_NORM_TPB
        red[tid] = acc
    var step = KNN_HOST_NORM_TPB // 2
    while step > 0:
        for t in range(step):
            red[t] = red[t] + red[t + step]
        step //= 2
    return ftz(red[0])


def host_cosine_row_norms(x: List[Float32], n_rows: Int, d: Int) -> List[Float32]:
    """`cosine_row_norm_kernel` (`distance_ops.mojo`), every row: the same
    strided partials and halving tree as `host_row_norm` (its fold width is
    the same kernel-matrix row), then the clamp `total <= 0 -> 0` and
    `ftz(identical_sqrt(total))`, the TRUE L2 norm cosine's epilogue
    divides by."""
    var out = List[Float32](length=n_rows, fill=Float32(0.0))
    for row in range(n_rows):
        var total = host_row_norm(x, row, d)
        if total <= Float32(0.0):
            total = Float32(0.0)
        out[row] = ftz(identical_sqrt(total))
    return out^


def host_metric_cell(
    q: List[Float32], row: Int, y: List[Float32], col: Int, d: Int,
    qn: Float32, yn: Float32, metric: Int, metric_arg: Float32,
) -> Float32:
    """`host_metric_cell_ptr` over two Lists, the checks' door."""
    return host_metric_cell_ptr(
        host_list_ptr(q), row, host_list_ptr(y), col, d, qn, yn, metric, metric_arg
    )


@always_inline
def host_metric_cell_ptr(
    q: HostF32Ptr, row: Int, y: HostF32Ptr, col: Int, d: Int,
    qn: Float32, yn: Float32, metric: Int, metric_arg: Float32,
) -> Float32:
    """One cell of `metric_distance_kernel` for the four metrics above,
    `x = q` (the query tile) and `y` (the index tile), the feature axis
    ascending, each operand `ftz`'d as it is loaded. `qn` and `yn` are the
    TRUE L2 norms and are read by cosine only.

    THE SABOTAGE ARM walks the feature chain DESCENDING for L1, Lp and
    cosine (a float fold in the other order), and for Chebyshev, whose
    running max is order-free, returns the next float32 above the maximum."""
    var acc = Float32(0.0)
    if metric == DIST_LINF:
        for f in range(d):
            acc = linf_core(
                acc, ftz(q.unsafe_load(row * d + f)), ftz(y.unsafe_load(col * d + f))
            )
        comptime if KNN_HOST_SABOTAGE:
            acc = bitcast[DType.float32](bitcast[DType.uint32](acc) + UInt32(1))
        return acc
    for g in range(d):
        var f = g
        comptime if KNN_HOST_SABOTAGE:
            f = d - 1 - g
        var qv = ftz(q.unsafe_load(row * d + f))
        var yv = ftz(y.unsafe_load(col * d + f))
        if metric == DIST_L1:
            acc = l1_core(acc, qv, yv)
        elif metric == DIST_LP_UNEXPANDED:
            acc = lp_unexp_core(acc, qv, yv, metric_arg)
        else:
            acc = inner_product_core(acc, qv, yv)
    var out: Float32
    if metric == DIST_L1:
        out = acc
    elif metric == DIST_LP_UNEXPANDED:
        # `:67`: `one_over_p` formed once per cell, a pure function of p.
        var one_over_p = ftz(identical_div(Float32(1.0), metric_arg))
        out = lp_unexp_epilog(acc, one_over_p)
    else:
        out = cosine_epilog(acc, ftz(qn), ftz(yn))
    comptime if KNN_HOST_SABOTAGE:
        # THE VALUE ARM (see KNN_HOST_SABOTAGE): the descending fold above is
        # exact on integer data, this is not.
        out = host_sabotage_value_flip(out)
    return out


def host_row_norms(x: List[Float32], n_rows: Int, d: Int) -> List[Float32]:
    """`compute_norms(ctx, a, a_norm, n_rows, d, False)`, every row."""
    var out = List[Float32](length=n_rows, fill=Float32(0.0))
    for row in range(n_rows):
        out[row] = host_row_norm(x, row, d)
    return out^


def host_l2_expanded_cell(
    q: List[Float32], row: Int, y: List[Float32], col: Int, d: Int,
    qn: Float32, yn: Float32, is_sqrt: Bool,
) -> Float32:
    """`host_l2_expanded_cell_ptr` over two Lists, the checks' door."""
    return host_l2_expanded_cell_ptr(
        host_list_ptr(q), row, host_list_ptr(y), col, d, qn, yn, is_sqrt
    )


@always_inline
def host_l2_expanded_cell_ptr(
    q: HostF32Ptr, row: Int, y: HostF32Ptr, col: Int, d: Int,
    qn: Float32, yn: Float32, is_sqrt: Bool,
) -> Float32:
    """One cell of `pinned_distance_tile_kernel` (`pinned_distance_tile.mojo:
    91-107`):

        var acc = Float32(0.0)
        for f in range(d):
            acc = ftz(identical_mul_add(ftz(q[row*d+f]), ftz(y[col*d+f]), acc))
        var dist = ftz(identical_mul_add(-2.0, acc, ftz(ftz(qn) + ftz(yn))))
        if dist <= 0.0: dist = 0.0
        if is_sqrt: dist = ftz(identical_sqrt(dist))
    """
    var acc = Float32(0.0)
    comptime if KNN_HOST_SABOTAGE:
        # THE SABOTAGE ARM: the same chain, walked DESCENDING. Wrong on
        # purpose; see KNN_HOST_SABOTAGE.
        for g in range(d):
            var f = d - 1 - g
            var qv = ftz(q.unsafe_load(row * d + f))
            var yv = ftz(y.unsafe_load(col * d + f))
            acc = ftz(identical_mul_add(qv, yv, acc))
    else:
        for f in range(d):
            var qv = ftz(q.unsafe_load(row * d + f))
            var yv = ftz(y.unsafe_load(col * d + f))
            acc = ftz(identical_mul_add(qv, yv, acc))
    var dist = ftz(
        identical_mul_add(Float32(-2.0), acc, ftz(ftz(qn) + ftz(yn)))
    )
    if dist <= Float32(0.0):
        dist = Float32(0.0)
    if is_sqrt:
        dist = ftz(identical_sqrt(dist))
    return dist


@always_inline
def host_twiddle_in(value: Float32) -> UInt32:
    """`twiddle_in(value, select_min=True)` (`select_radix.mojo:155-170`):
    flip the sign bit for positives and every bit for negatives, so the
    unsigned order of the result is the float order of the input."""
    var bits = bitcast[DType.uint32](value)
    if (bits & UInt32(0x80000000)) != 0:
        bits = bits ^ UInt32(0xFFFFFFFF)
    else:
        bits = bits ^ UInt32(0x80000000)
    return bits


@always_inline
def host_composite_key(value: Float32, index: UInt32) -> UInt64:
    """`composite_key(value, index, select_min=True)`. The sabotage arm
    reverses the index half, so a tie in distance at the k-th boundary
    admits the HIGHER index: the wrong tie order the brief warns about,
    which the gate must catch on the fixtures with exact ties."""
    comptime if KNN_HOST_SABOTAGE:
        return (UInt64(host_twiddle_in(value)) << UInt64(32)) | UInt64(
            UInt32(0xFFFFFFFF) - index
        )
    return (UInt64(host_twiddle_in(value)) << UInt64(32)) | UInt64(index)


#: The small-k selector's empty slot (`select_smallk_identical_candidate.
#: mojo:95`). No real key reaches it: its high half is the twiddle of a
#: positive NaN and its low half an index no buffer can hold.
comptime KNN_HOST_KEY_SENTINEL = UInt64(18446744073709551615)


def host_select_k(
    dist_row: List[Float32], n_index: Int, k: Int,
    mut out_dist: List[Float32], mut out_idx: List[UInt32], base: Int,
):
    """The k smallest composite keys of one query row, written at
    `out_*[base .. base + k)` ascending, then the estimator's insertion
    sort by `(distance, index)` over that slice (`neighbors/estimator.mojo:
    664-682`), restated so the caller-visible order is the estimator's and
    not merely equal to it."""
    var best = List[UInt64](length=k, fill=KNN_HOST_KEY_SENTINEL)
    for col in range(n_index):
        var pending = host_composite_key(dist_row[col], UInt32(col))
        if pending < best[k - 1]:
            # Carry insertion, `smallk_identical_kernel:100-106`.
            for slot in range(k):
                if pending < best[slot]:
                    var previous = best[slot]
                    best[slot] = pending
                    pending = previous
    for rank in range(k):
        var selected = UInt32(best[rank] & UInt64(4294967295))
        comptime if KNN_HOST_SABOTAGE:
            # The sabotage key carries the REVERSED index; read it back.
            selected = UInt32(0xFFFFFFFF) - selected
        out_idx[base + rank] = selected
        out_dist[base + rank] = dist_row[Int(selected)]
    # THE ESTIMATOR'S SORT, `knn_search_traced`'s host pass, verbatim.
    for a in range(1, k):
        var dv = out_dist[base + a]
        var iv = out_idx[base + a]
        var b = a - 1
        while b >= 0:
            var db = out_dist[base + b]
            var ib = out_idx[base + b]
            if db < dv or (db == dv and ib <= iv):
                break
            out_dist[base + b + 1] = db
            out_idx[base + b + 1] = ib
            b -= 1
        out_dist[base + b + 1] = dv
        out_idx[base + b + 1] = iv


def host_knn_search(
    index: List[Float32], n_index: Int,
    queries: List[Float32], n_queries: Int, d: Int, k: Int,
    metric: Int, return_sqrt: Bool,
    mut out_dist: List[Float32], mut out_idx: List[UInt32],
    metric_arg: Float32 = Float32(2.0),
) raises:
    """`knn_search_traced` (`neighbors/estimator.mojo:389-700`) on the
    host: the shape refusals in its order, the metric resolved, both norm
    vectors, every distance of every query row through
    `host_l2_expanded_cell`, the selection and the sort. `out_dist` and
    `out_idx` hold `n_queries * k` each."""
    if n_index <= 0:
        raise Error("knn_search: n_index must be positive, got " + String(n_index))
    if n_queries <= 0:
        raise Error(
            "knn_search: n_queries must be positive, got " + String(n_queries)
        )
    if d <= 0:
        raise Error(
            "knn_search: n_features must be positive, got " + String(d)
        )
    if k <= 0:
        raise Error("knn_search: k must be positive, got " + String(k))
    if k > n_index:
        raise Error(
            "knn_search: k ("
            + String(k)
            + ") exceeds n_index ("
            + String(n_index)
            + "); the upstream's short-index fill is not implemented"
        )
    var mtr = host_resolve_metric(metric, return_sqrt)
    validate_metric_arg(mtr, metric_arg)  # DEVIATION 552
    if mtr == DIST_COSINE_EXPANDED:
        # DEVIATION 553, `knn_search_traced`'s words, index first.
        var zi = cosine_zero_norm_row(index, n_index, d)
        if zi >= 0:
            raise Error(
                "knn_search: metric='cosine' but index row "
                + String(zi)
                + " is all zeros; cosine distance divides by ||x|| and is"
                " undefined at the origin (DEVIATION 553)"
            )
        var zq = cosine_zero_norm_row(queries, n_queries, d)
        if zq >= 0:
            raise Error(
                "knn_search: metric='cosine' but query row "
                + String(zq)
                + " is all zeros; cosine distance divides by ||x|| and is"
                " undefined at the origin (DEVIATION 553)"
            )
    # DEVIATION 2920 (lane/infer-speed-classical, 2026-09-17): the query
    # rows are split into contiguous tasks (`core/host_predict_threads.
    # mojo`). A query row's distances, its selection and its sort read the
    # index, the norms and its own row only and write its own `k` slots,
    # so the split moves no bit; each task keeps one `dist_row` scratch
    # and the row's cells are still written ascending by one thread.
    var l2_pair = mtr == KNN_HOST_DIST_L2_EXPANDED or mtr == KNN_HOST_DIST_L2_SQRT_EXPANDED
    var is_sqrt = mtr == KNN_HOST_DIST_L2_SQRT_EXPANDED
    var index_norm = List[Float32](length=n_index, fill=Float32(0.0))
    var query_norm = List[Float32](length=n_queries, fill=Float32(0.0))
    if l2_pair:
        index_norm = host_row_norms(index, n_index, d)
        query_norm = host_row_norms(queries, n_queries, d)
    elif mtr == DIST_COSINE_EXPANDED:
        # `compute_norms_for_metric`: cosine's TRUE norm, none for the rest.
        index_norm = host_cosine_row_norms(index, n_index, d)
        query_norm = host_cosine_row_norms(queries, n_queries, d)
    var tasks = host_predict_task_count(n_queries)
    var chunk = host_predict_chunk(n_queries, tasks)
    var ip = host_list_ptr(index)
    var qp = host_list_ptr(queries)
    var inp = host_list_ptr(index_norm)
    var qnp = host_list_ptr(query_norm)
    var odp = host_list_ptr(out_dist)
    var oip = host_list_ptr_u32(out_idx)

    def _rows(c: Int) {imm ip, imm qp, imm inp, imm qnp, imm odp, imm oip, imm chunk, imm n_queries, imm n_index, imm d, imm k, imm mtr, imm metric_arg, imm l2_pair, imm is_sqrt}:
        var dist_row = List[Float32](length=n_index, fill=Float32(0.0))
        var sel_dist = List[Float32](length=k, fill=Float32(0.0))
        var sel_idx = List[UInt32](length=k, fill=UInt32(0))
        var lo = c * chunk
        var hi = min(lo + chunk, n_queries)
        for row in range(lo, hi):
            var qn = qnp.unsafe_load(row)
            if l2_pair:
                for col in range(n_index):
                    dist_row[col] = host_l2_expanded_cell_ptr(
                        qp, row, ip, col, d, qn, inp.unsafe_load(col), is_sqrt
                    )
            else:
                for col in range(n_index):
                    dist_row[col] = host_metric_cell_ptr(
                        qp, row, ip, col, d, qn, inp.unsafe_load(col), mtr, metric_arg
                    )
            host_select_k(dist_row, n_index, k, sel_dist, sel_idx, 0)
            for rank in range(k):
                odp.unsafe_store(row * k + rank, sel_dist[rank])
                oip.unsafe_store(row * k + rank, sel_idx[rank])
        _ = dist_row^
        _ = sel_dist^
        _ = sel_idx^

    if tasks == 1:
        _rows(0)
    else:
        sync_parallelize(_rows, tasks)
    _ = index_norm^
    _ = query_norm^


def host_unique_labels(y: List[Int32], n: Int) -> List[Int32]:
    """`getUniquelabels` (`classlabels.mojo:62-92`): the sorted distinct
    values of `y[0:n]`, by binary-search insertion."""
    var uniq = List[Int32]()
    for i in range(n):
        var v = y[i]
        var lo = 0
        var hi = len(uniq)
        while lo < hi:
            var mid = (lo + hi) // 2
            if uniq[mid] < v:
                lo = mid + 1
            else:
                hi = mid
        if lo < len(uniq) and uniq[lo] == v:
            continue
        uniq.insert(lo, v)
    return uniq^


def host_monotonic(y: List[Int32], n: Int, uniq: List[Int32]) -> List[Int32]:
    """`make_monotonic(out, [y; uniq], n + n_uniq, zero_based=False)` then
    `_subtract_one_kernel` over the first `n` (`knn.mojo:333-347`):
    `map_label_kernel`'s linear scan, the FIRST `i` with `y == uniq[i]`,
    `+1` for `zero_based = False`, then `-1`. `uniq` is `y`'s own unique
    set, so the appended tail changes nothing and every label is found."""
    var out = List[Int32](length=n, fill=Int32(0))
    for tid in range(n):
        var v = y[tid]
        # `uniq` is sorted by `host_unique_labels`.  The device reference's
        # linear scan selects the sole equal entry. Keep that cheaper scan for
        # tiny class sets; lower_bound changes no observable value while
        # avoiding O(n * n_unique) host inference for larger label sets.
        if len(uniq) <= 16:
            for i in range(len(uniq)):
                if v == uniq[i]:
                    out[tid] = Int32(i)
                    break
            continue
        var lo = 0
        var hi = len(uniq)
        while lo < hi:
            var mid = (lo + hi) // 2
            if uniq[mid] < v:
                lo = mid + 1
            else:
                hi = mid
        # Every value in y produced uniq, hence lower_bound is an exact hit.
        out[tid] = Int32(lo)
    return out^


def host_class_probs(
    idx: List[UInt32], labels: List[Int32], n_uniq: Int,
    n_queries: Int, k: Int,
) -> List[Float32]:
    """`class_probs_kernel` (`knn.mojo:119-148`) over a zeroed tally:
    `n_neigh_inv = ftz(1.0 / k)`, then per slot in slot order
    `out[row * n_uniq + labels[idx]] = ftz(out + n_neigh_inv)`."""
    var out = List[Float32](length=n_queries * n_uniq, fill=Float32(0.0))
    var n_neigh_inv = ftz(Float32(1.0) / Float32(k))
    for row in range(n_queries):
        var i = row * k
        for j in range(k):
            var out_label = Int(labels[Int(idx[i + j])])
            var out_idx = row * n_uniq + out_label
            out[out_idx] = ftz(out[out_idx] + n_neigh_inv)
    return out^


def host_class_vote(
    proba: List[Float32], uniq: List[Int32], n_uniq: Int, n_queries: Int,
    mut out: List[Int32], n_outputs: Int, output_offset: Int,
):
    """`class_vote_kernel` (`knn.mojo:151-180`): `cur_max = -1.0`, strict
    `>`, the first maximal class wins, `out[row * n_outputs +
    output_offset] = uniq[cur_label]`. The sabotage arm compares with
    `>=`, so a tied vote goes to the LAST maximal class: the vote's tie
    rule, wrong on purpose."""
    for row in range(n_queries):
        var i = row * n_uniq
        var cur_max = Float32(-1.0)
        var cur_label = -1
        for j in range(n_uniq):
            var cur_proba = proba[i + j]
            comptime if KNN_HOST_SABOTAGE:
                if cur_proba >= cur_max:
                    cur_max = cur_proba
                    cur_label = j
            else:
                if cur_proba > cur_max:
                    cur_max = cur_proba
                    cur_label = j
        out[row * n_outputs + output_offset] = uniq[cur_label]


def host_regress_avg(
    idx: List[UInt32], y: List[Float32], n_queries: Int, k: Int,
    mut out: List[Float32], n_outputs: Int, output_offset: Int,
):
    """`regress_avg_kernel` (`knn.mojo:183-211`): `pred = ftz(pred +
    ftz(y[idx]))` over the k slots in slot order, then
    `ftz(pred / Float32(k))`. The sabotage arm walks the slots DESCENDING,
    so the negative control reaches this fold on its own and not only
    through the neighbour set it consumes (the uniform tally has no such
    arm: j copies of 1/k sum to the same float in any order, as
    `knn.mojo`'s header says)."""
    for row in range(n_queries):
        var i = row * k
        var pred = Float32(0.0)
        comptime if KNN_HOST_SABOTAGE:
            for g in range(k):
                var j = k - 1 - g
                pred = ftz(pred + ftz(y[Int(idx[i + j])]))
        else:
            for j in range(k):
                pred = ftz(pred + ftz(y[Int(idx[i + j])]))
        out[row * n_outputs + output_offset] = ftz(pred / Float32(k))


def host_distance_weights(
    dist: List[Float32], n_queries: Int, k: Int
) raises -> List[Float32]:
    """`host_distance_weights` (`distance_weights.mojo:152-244`), relocated
    statement for statement: `w = 1/d` by `identical_div`, a row holding
    any infinity replaced WHOLESALE by its infinity mask, a negative
    distance refused, a row whose normalizer is not positive refused
    (DEVIATION 555)."""
    if n_queries <= 0 or k <= 0:
        raise Error(
            "host_distance_weights: n_queries and k must be positive, got "
            + String(n_queries) + ", " + String(k)
        )
    if len(dist) != n_queries * k:
        raise Error(
            "host_distance_weights: dist holds " + String(len(dist))
            + " values, expected " + String(n_queries * k)
        )
    var w = List[Float32](capacity=n_queries * k)
    for _ in range(n_queries * k):
        w.append(Float32(0.0))
    var pos_inf = Float32(1.0) / Float32(0.0)
    for i in range(n_queries):
        var base = i * k
        var any_inf = False
        for j in range(k):
            var dv = ftz(dist[base + j])
            var v = ftz(identical_div(Float32(1.0), dv))
            if v == pos_inf or v == -pos_inf:
                any_inf = True
            w[base + j] = v
        if any_inf:
            for j in range(k):
                var v = w[base + j]
                if v == pos_inf:
                    w[base + j] = Float32(1.0)
                elif v == -pos_inf:
                    raise Error(
                        "host_distance_weights: query row "
                        + String(i)
                        + " slot "
                        + String(j)
                        + " has a NEGATIVE distance, which no implemented metric"
                        " can produce; refusing rather than weighting it"
                    )
                else:
                    w[base + j] = Float32(0.0)
            continue
        var s = Float32(0.0)
        for j in range(k):
            s = ftz(s + ftz(w[base + j]))
        if s <= Float32(0.0):
            raise Error(
                "knn weights='distance': every neighbour of query row "
                + String(i)
                + " has 1/d that underflows float32 (the nearest is at"
                " distance "
                + String(dist[base])
                + "), so the row's normalizer is zero on an FTZ column and"
                " not on a denormal-honoring one (DEVIATION 555)"
            )
    return w^


def host_weighted_class_probs(
    idx: List[UInt32], labels: List[Int32], w: List[Float32], n_uniq: Int,
    n_queries: Int, k: Int,
) -> List[Float32]:
    """`weighted_class_probs_kernel` (`distance_weights.mojo:247-306`):
    pass 1 scatters `ftz(out + ftz(w[slot]))` in slot order; pass 2 sums
    the row over the CLASS axis, `s = ftz(s + ftz(out[c]))`, and divides
    each class by it, `ftz(identical_div(ftz(out[c]), s))`."""
    var out = List[Float32](length=n_queries * n_uniq, fill=Float32(0.0))
    for row in range(n_queries):
        var i = row * k
        for j in range(k):
            var out_label = Int(labels[Int(idx[i + j])])
            var out_idx = row * n_uniq + out_label
            out[out_idx] = ftz(out[out_idx] + ftz(w[i + j]))
        var s = Float32(0.0)
        for c in range(n_uniq):
            s = ftz(s + ftz(out[row * n_uniq + c]))
        for c in range(n_uniq):
            var oi = row * n_uniq + c
            out[oi] = ftz(identical_div(ftz(out[oi]), s))
    return out^


def host_weighted_regress_avg(
    idx: List[UInt32], y: List[Float32], w: List[Float32],
    n_queries: Int, k: Int,
    mut out: List[Float32], n_outputs: Int, output_offset: Int,
):
    """`weighted_regress_avg_kernel` (`distance_weights.mojo:309-348`):
    `num = ftz(num + ftz(yv * wv))`, `den = ftz(den + wv)` in slot order
    over `wv = ftz(w[slot])`, `yv = ftz(y[idx])`, then
    `ftz(identical_div(num, den))`. The product is a plain `*`, as there."""
    for row in range(n_queries):
        var i = row * k
        var num = Float32(0.0)
        var den = Float32(0.0)
        for j in range(k):
            var wv = ftz(w[i + j])
            var yv = ftz(y[Int(idx[i + j])])
            num = ftz(num + ftz(yv * wv))
            den = ftz(den + wv)
        out[row * n_outputs + output_offset] = ftz(identical_div(num, den))


# ===========================================================================
# THE BALL COVER'S QUERIES AS AN EXHAUSTIVE SCAN (lane/cpu-training-batch3).
# See the module docstring.
# ===========================================================================

#: `DIST_L2_SQRT_UNEXPANDED`, the ball cover's Euclidean tag
#: (`distance_ops.mojo`, `RBC_METRIC_DEFAULT` in `common.mojo`).
comptime KNN_HOST_DIST_L2_SQRT_UNEXPANDED = 5


def host_rbc_cmp_dist(
    a: List[Float32], a_off: Int, b: List[Float32], b_off: Int, n_dims: Int,
    metric: Int, metric_arg: Float32,
) -> Float32:
    """`rbc_cmp_dist` over host lists: SQUARED Euclidean on the Euclidean
    arm, the plain metric on L1, Linf and Lp. The caller has validated the
    metric. THE SABOTAGE ARM walks the dimensions DESCENDING on the three
    folds and returns the next float32 above Chebyshev's maximum."""
    var acc = Float32(0.0)
    if metric == DIST_LINF:
        for i in range(n_dims):
            acc = linf_core(acc, ftz(a[a_off + i]), ftz(b[b_off + i]))
        comptime if KNN_HOST_SABOTAGE:
            acc = bitcast[DType.float32](bitcast[DType.uint32](acc) + UInt32(1))
        return acc
    for g in range(n_dims):
        var i = g
        comptime if KNN_HOST_SABOTAGE:
            i = n_dims - 1 - g
        if metric == KNN_HOST_DIST_L2_SQRT_UNEXPANDED:
            var diff = ftz(ftz(a[a_off + i]) - ftz(b[b_off + i]))
            acc = ftz(identical_mul_add(diff, diff, acc))
        elif metric == DIST_L1:
            acc = l1_core(acc, ftz(a[a_off + i]), ftz(b[b_off + i]))
        else:
            acc = lp_unexp_core(acc, ftz(a[a_off + i]), ftz(b[b_off + i]), metric_arg)
    if metric == DIST_LP_UNEXPANDED:
        return lp_unexp_epilog(acc, ftz(identical_div(Float32(1.0), metric_arg)))
    return acc


def host_rbc_radius_row(
    index: List[Float32], n_index: Int, queries: List[Float32], q: Int,
    d: Int, eps: Float32, metric: Int, metric_arg: Float32,
) -> List[Int32]:
    """One query's neighbors within `eps`, columns ascending."""
    var out = List[Int32]()
    var eps_cmp = rbc_cmp_bound(metric, eps)
    for c in range(n_index):
        var dist = host_rbc_cmp_dist(queries, q * d, index, c * d, d, metric, metric_arg)
        if dist <= eps_cmp:
            out.append(Int32(c))
    return out^


def host_rbc_radius_counts(
    index: List[Float32], n_index: Int, queries: List[Float32], n_queries: Int,
    d: Int, eps: Float32, metric: Int, metric_arg: Float32,
    mut counts: List[Int32], requested_tasks: Int = 0,
):
    """Count independent radius-query rows in parallel.

    Each task owns disjoint ``counts`` cells.  The feature fold and ascending
    index walk inside a row are unchanged, so one and many tasks produce the
    same bytes.
    """
    var tasks = requested_tasks
    if tasks <= 0:
        tasks = host_predict_task_count(n_queries)
    tasks = max(1, min(tasks, n_queries))
    var chunk = host_predict_chunk(n_queries, tasks)
    var eps_cmp = rbc_cmp_bound(metric, eps)
    def _rows(c: Int) {imm index, imm queries, mut counts, imm chunk, imm n_queries, imm n_index, imm d, imm eps_cmp, imm metric, imm metric_arg}:
        var lo = c * chunk
        var hi = min(lo + chunk, n_queries)
        for q in range(lo, hi):
            var count = Int32(0)
            for col in range(n_index):
                var dist = host_rbc_cmp_dist(
                    queries, q * d, index, col * d, d, metric, metric_arg
                )
                if dist <= eps_cmp:
                    count += Int32(1)
            counts[q] = count
    if tasks == 1:
        _rows(0)
    else:
        sync_parallelize(_rows, tasks)


def host_rbc_radius_fill_rows(
    index: List[Float32], n_index: Int, queries: List[Float32], n_queries: Int,
    d: Int, eps: Float32, return_sqrt: Bool, metric: Int, metric_arg: Float32,
    indptr: List[Int32], mut actual_counts: List[Int32],
    mut cols: List[Int32], mut dists: List[Float32],
    requested_tasks: Int = 0,
):
    """Fill independent CSR rows in parallel at their counted offsets."""
    var tasks = requested_tasks
    if tasks <= 0:
        tasks = host_predict_task_count(n_queries)
    tasks = max(1, min(tasks, n_queries))
    var chunk = host_predict_chunk(n_queries, tasks)
    var eps_cmp = rbc_cmp_bound(metric, eps)
    def _rows(c: Int) {imm index, imm queries, imm indptr, mut actual_counts, mut cols, mut dists, imm chunk, imm n_queries, imm n_index, imm d, imm eps_cmp, imm return_sqrt, imm metric, imm metric_arg}:
        var lo = c * chunk
        var hi = min(lo + chunk, n_queries)
        for q in range(lo, hi):
            var out = Int(indptr[q])
            var out_end = Int(indptr[q + 1])
            var actual = Int32(0)
            for col in range(n_index):
                var cmp_dist = host_rbc_cmp_dist(
                    queries, q * d, index, col * d, d, metric, metric_arg
                )
                if cmp_dist <= eps_cmp:
                    # The count call supplied this row's exact slice.  Clamp
                    # writes if the caller mutated an input between calls;
                    # the binding rejects the changed count after the join.
                    if out < out_end:
                        cols[out] = Int32(col)
                        var reported = cmp_dist
                        if return_sqrt:
                            reported = rbc_true_dist(metric, cmp_dist)
                        comptime if KNN_HOST_SABOTAGE:
                            reported = host_sabotage_value_flip(reported)
                        dists[out] = reported
                        out += 1
                    actual += Int32(1)
            actual_counts[q] = actual
    if tasks == 1:
        _rows(0)
    else:
        sync_parallelize(_rows, tasks)


def host_rbc_edge_distance(
    index: List[Float32], queries: List[Float32], q: Int, c: Int, d: Int,
    return_sqrt: Bool, metric: Int, metric_arg: Float32,
) -> Float32:
    """`rbc_edge_distance_kernel` for one edge."""
    var d2 = host_rbc_cmp_dist(queries, q * d, index, c * d, d, metric, metric_arg)
    var out = d2
    if return_sqrt:
        out = rbc_true_dist(metric, d2)
    comptime if KNN_HOST_SABOTAGE:
        # THE VALUE ARM (see KNN_HOST_SABOTAGE), after the root: a one-unit
        # step on the squared distance can round back through it.
        out = host_sabotage_value_flip(out)
    return out


def host_rbc_knn_row(
    index: List[Float32], n_index: Int, queries: List[Float32], q: Int,
    d: Int, k: Int, metric: Int, metric_arg: Float32,
    mut out_idx: List[Int32], mut out_dist: List[Float32],
):
    """The k smallest `(cmp distance, index)` of one query, ascending, the
    indices and the TRUE distances written at `q * k`."""
    var best_d = List[Float32](length=k, fill=Float32(0.0))
    var best_i = List[Int](length=k, fill=-1)
    var filled = 0
    for c in range(n_index):
        var dist = host_rbc_cmp_dist(queries, q * d, index, c * d, d, metric, metric_arg)
        # Insertion into the ascending list; the index ascends with `c`, so
        # a tie in distance keeps the earlier (smaller) index first.
        if filled == k and not (dist < best_d[k - 1]):
            continue
        var pos = filled if filled < k else k - 1
        while pos > 0 and dist < best_d[pos - 1]:
            if pos < k:
                best_d[pos] = best_d[pos - 1]
                best_i[pos] = best_i[pos - 1]
            pos -= 1
        best_d[pos] = dist
        best_i[pos] = c
        if filled < k:
            filled += 1
    for o in range(k):
        out_idx[q * k + o] = Int32(best_i[o])
        var reported = rbc_true_dist(metric, best_d[o])
        comptime if KNN_HOST_SABOTAGE:
            # THE VALUE ARM (see KNN_HOST_SABOTAGE), after the root.
            reported = host_sabotage_value_flip(reported)
        out_dist[q * k + o] = reported


def host_rbc_knn_search(
    index: List[Float32], n_index: Int, queries: List[Float32], n_queries: Int,
    d: Int, k: Int, metric: Int, metric_arg: Float32,
    mut out_idx: List[Int32], mut out_dist: List[Float32],
    requested_tasks: Int = 0,
):
    """All exact RBC k-NN rows, split over independent query rows.

    One requested task is the serial proof arm. Zero uses the common CPU
    inference policy. Tasks own disjoint output rows and preserve every
    distance fold and row selection order.
    """
    var tasks = requested_tasks
    if tasks <= 0:
        tasks = host_predict_task_count(n_queries)
    tasks = max(1, min(tasks, n_queries))
    var chunk = host_predict_chunk(n_queries, tasks)
    def _rows(c: Int) {imm index, imm queries, mut out_idx, mut out_dist, imm chunk, imm n_queries, imm n_index, imm d, imm k, imm metric, imm metric_arg}:
        var lo = c * chunk
        var hi = min(lo + chunk, n_queries)
        for q in range(lo, hi):
            host_rbc_knn_row(
                index, n_index, queries, q, d, k, metric, metric_arg,
                out_idx, out_dist,
            )
    if tasks == 1:
        _rows(0)
    else:
        sync_parallelize(_rows, tasks)
