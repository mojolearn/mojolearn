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

WHAT IS NOT HERE. The cosine, L1, Linf, L2 unexpanded and Lp metrics
(`metric_distance_kernel`, `neighbors/impl/distance/detail/distance_ops.
mojo`) and the random ball cover arm (`rbc_knn_search`). The host binding
refuses those BY NAME; nothing here computes something else under their
name.

The restatement is a prediction until measured. tools/classical_host_gate.py
(lanes knn, knn-clf, knn-reg) is the measurement, and the brief records
what it has shown.
"""
from std.memory import bitcast
from std.sys.compile import is_defined

from checks.kernel_matrix import (
    COLUMN_AMD,
    COLUMN_APPLE,
    COLUMN_CPU,
    COLUMN_NVIDIA,
    K_LIB_ROW_NORM,
    lib_block_size_for,
)
from checks.numerics import ftz, identical_div, identical_mul_add, identical_sqrt


#: The gate's negative control, the same define the classical and phase 1
#: host bindings read (`core/classical_host_predict.mojo:63`). A build with
#: it is wrong on purpose in four places, one per thing the brief names as
#: the work: every distance dot product's feature chain walks DESCENDING
#: (a serial float32 fold in the other order is a different bit pattern
#: on almost every cell), the selection key admits the HIGHER index at a
#: tied boundary, the vote's tie goes to the LAST maximal class, and the
#: regressor's mean folds its slots descending. The gate must catch each
#: of them. Read back by `core_host_sabotage`.
comptime KNN_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()

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
    raise Error(
        "knn host: no CPU implementation of cuVS DistanceType value "
        + String(metric)
        + " yet; the host computes euclidean/l2 (1) and sqeuclidean (0)"
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
            var qv = ftz(q[row * d + f])
            var yv = ftz(y[col * d + f])
            acc = ftz(identical_mul_add(qv, yv, acc))
    else:
        for f in range(d):
            var qv = ftz(q[row * d + f])
            var yv = ftz(y[col * d + f])
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
    var is_sqrt = mtr == KNN_HOST_DIST_L2_SQRT_EXPANDED
    var index_norm = host_row_norms(index, n_index, d)
    var query_norm = host_row_norms(queries, n_queries, d)
    var dist_row = List[Float32](length=n_index, fill=Float32(0.0))
    for row in range(n_queries):
        var qn = query_norm[row]
        for col in range(n_index):
            dist_row[col] = host_l2_expanded_cell(
                queries, row, index, col, d, qn, index_norm[col], is_sqrt
            )
        host_select_k(dist_row, n_index, k, out_dist, out_idx, row * k)


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
        for i in range(len(uniq)):
            if v == uniq[i]:
                out[tid] = Int32(i + 1)
                break
        out[tid] = out[tid] - Int32(1)
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
