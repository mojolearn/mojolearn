# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into a CPU host binding (python/mojolearn/host_surface.py names which); product, not only a check.
"""The label, regression and silhouette metrics on the host, for a box with
no GPU (workstream E batch 2, the metrics lane,
2026-09-14).

HOST ONLY. Nothing here imports `max.gpu`, `max.gpu` or a `DeviceContext`,
and no GPU binding imports this file. Every metric of the lane is spelled a
SECOND time from its device source: the integer kernels (a count, a
histogram, a contingency matrix, integer atomics whose sum no order can
move) become serial integer loops, and the float folds are the fixed slab
tree of `metrics/checks/pinned_sum.mojo` spelled again. The arithmetic
leaves are `checks/numerics.mojo`'s (`ftz`, `identical_mul_add`,
`identical_log`, `identical_sqrt`).

WHAT IS RESTATED, AND WHERE THE ORIGINAL IS.

  `host_tree_sum`          `virtual_block_sum` over every PINNED_SUM_W chunk
                           plus `host_fold_partials`, `metrics/checks/
                           pinned_sum.mojo:108-176`: slot `t` of chunk `c`
                           holds `ftz(values[c*W + t])` (`+0.0` past `n`),
                           the halving tree `slab[t] = ftz(slab[t] +
                           slab[t + step])` for `step = W/2 .. 1`, the chunk
                           totals folded ascending from `+0.0` through `ftz`.
  `host_canonicalize_nan`  `canonicalize_nan`, `pinned_sum.mojo:183`.
  `host_accuracy_score`    `accuracy_score`, `metrics/impl/stats/detail/
                           scores.mojo:106`: the integer count of agreeing
                           positions, `Float32(count) / Float32(n)`.
  `host_r2_score`          `r2_score_parts_traced`, `scores.mojo:262`: the
                           tree over `y`, `ratio = ftz(1 / n)`, `y_bar =
                           ftz(y_sum * ratio)` (two roundings, as
                           `mean.cuh` does), `d1 = ftz(y - y_hat)`, `d2 =
                           ftz(y - y_bar)`, the trees over `ftz(d1 * d1)`
                           and `ftz(d2 * d2)`, then `r2_epilogue`
                           (`scores.mojo:352`, DEVIATION 657).
  `host_count_unique`      `count_unique`, `metrics/impl/stats/detail/
                           adjusted_rand_index.mojo:66`: the min and max
                           label, the histogram, the nonzero bins.
  `host_contingency`       `contingency_matrix`, `metrics/impl/stats/detail/
                           contingency_matrix.mojo:207`: `C[(gt - min) *
                           width + (pd - min)] += 1`, integers.
  `host_adjusted_rand_score`
                           `compute_adjusted_rand_index_traced`, `adjusted_
                           rand_index.mojo:160`, in its order: the `size <
                           2` arm, the two unique counts, the union range,
                           the `nUniq == 1 or == size` arm, the matrix,
                           `nCTwo` in Int64, the five Float64 host ops.
  `host_entropy`           `entropy_traced` and `entropy_from_counts_
                           traced`, `entropy.mojo:117, 191`: the histogram
                           over `[lower, upper]`, then the IDENTICAL arm
                           (`p = ftz(count / size)`, `acc = ftz(fma(-p,
                           ftz(identical_log(p)), acc))`, widened).
  `host_mutual_info`       `mutual_info_score_traced` and `mutual_info_
                           from_contingency_traced`, `mutual_info_score.
                           mojo:107, 240`: the contingency matrix over
                           `[lower, upper]`, row and column sums in Int64,
                           the IDENTICAL arm ascending over `(i, j)`
                           (`ftz(identical_log(ftz(size * c)))`, `ftz(
                           identical_log(ftz(Float32(a * b))))`, `acc =
                           ftz(fma(c, diff, acc))`, `ftz(acc / size)`).
  `host_homogeneity_score` `homogeneity_score`, `homogeneity_score.mojo:26`:
                           `size == 0 -> 1.0`, `MI / H(truth)` or `1.0`.
  `host_v_measure`         `v_measure`, `v_measure.mojo:29`: `h`, `c` as
                           homogeneity with the arrays swapped, `c + h ==
                           0 -> 0`, else `(1 + beta) * h * c / (beta * h +
                           c)` in Float64 on the host.
  `host_l2sqrt_unexpanded` `metrics/checks/pinned_distance.mojo:59`,
                           spelled again: ascending features, `diff =
                           ftz(ftz(x) - ftz(y))`, `acc = ftz(fma(diff, diff,
                           acc))`, `ftz(identical_sqrt(acc))`.
  `host_sil_op`            `sil_op`, `metrics/impl/stats/detail/silhouette_
                           score.mojo:88` (DEVIATION 656).
  `host_silhouette`        `silhouette_rows_kernel` and `silhouette_score_
                           launch`, `metrics/impl/stats/detail/batched/
                           silhouette_score.mojo:97, 187` (DEVIATION 654):
                           the refusals in their words, the label counts,
                           per row `a` and `b[c]` seeded as `fill_b_kernel`
                           seeds them, the per-cluster tree over `ftz(d /
                           denom)` with the chunk totals added ascending,
                           the positional min over `c` under a strict `<`,
                           `sil_op`, then the tree over the scores and
                           `ftz(total / n_rows)`. The same host model
                           `metrics/checks/silhouette_check.mojo::_oracle_
                           silhouette` gates the device against bit for bit.

THE NEGATIVE CONTROL. `-D MOJOLEARN_HOST_SABOTAGE=1` shifts every slab
tree's chunk boundaries by one value (`values[(i + 1) % n]` fills slot
`i`), the partition `pinned_sum.mojo::sabotage_shifted_host_tree_sum`
measured to move the sums where a rotation inside a chunk cannot; it
reaches r2's three sums, the silhouette's `a`, `b` and mean. The integer
metrics have no fold and do not move under it, and on the `hashed`, `wide`,
`denormal` and `denormal_ftz` fixtures the shifted tree rounded to the same
r2 and silhouette bits (lane/metrics-sabotage-coverage, 2026-09-15), so the
same define also perturbs a VALUE each metric reads:

  `host_accuracy_score`    sample 0's prediction is read as a label that
                           flips its agreement (the count moves by one).
  `host_adjusted_rand_score`, `host_entropy`, `host_mutual_info`
                           label 0 of the first array is read as another
                           label of the same array's range
                           (`host_sabotage_label0`); homogeneity,
                           completeness and v-measure reach it through
                           entropy and mutual information. A constant label
                           array cannot move under an in-range label change:
                           homogeneity's zero-entropy branch also returns the
                           deliberately wrong 0.0 instead of 1.0 under the
                           define. This reaches the all-negative H/C/V fixture.
  `host_r2_score`          every prediction is read as `y_hat + 1 + |y|`,
                           so the residual outgrows the scale of `y`.
  `host_silhouette`        row 0 is read shifted by `1 + |x|` in every
                           feature of every distance it takes part in
                           (`host_sabotage_l2sqrt`), independent of the
                           chunk, so `silhouette_samples` moves at every
                           chunksize. `host_l2sqrt_unexpanded` itself is
                           unchanged (trustworthiness reads it).

`host_fowlkes_mallows` keeps its own arm. Read back by
`metrics_host_sabotage`.

The restatement is a prediction until measured. The four-column diff of
tools/identity_break.py on the metrics lane is the measurement.
"""
from std.math import sqrt
from std.memory import bitcast
from std.sys.compile import is_defined
from max.algorithm import sync_parallelize

from checks.numerics import ftz, identical_div, identical_log, identical_mul_add, identical_sqrt
from core.host_predict_threads import host_predict_chunk, host_predict_task_count


#: The gate's negative control (see THE NEGATIVE CONTROL above).
comptime METRICS_ORACLE_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()

#: `PINNED_SUM_W`, `metrics/checks/pinned_sum.mojo:70`. NUMERIC: a different
#: width is a different tree.
comptime PINNED_SUM_W = 256

#: `CANONICAL_NAN_BITS`, `pinned_sum.mojo:179`.
comptime CANONICAL_NAN_BITS: UInt32 = 0x7FC00000

#: `std::numeric_limits<float>::max()`, `batched/silhouette_score.mojo:88`.
comptime FLOAT32_MAX_BITS: UInt32 = 0x7F7FFFFF

#: `DistanceType::L2SqrtUnexpanded`, the one silhouette metric.
comptime DISTANCE_L2_SQRT_UNEXPANDED = 5


def host_chunk_count(n: Int) -> Int:
    return (n + PINNED_SUM_W - 1) // PINNED_SUM_W


def host_fold_partials(partials: List[Float32], chunks: Int) -> Float32:
    """`host_fold_partials`, `pinned_sum.mojo:197`."""
    var acc = Float32(0.0)
    for c in range(chunks):
        acc = ftz(acc + partials[c])
    return acc


def host_tree_sum(values: List[Float32], n: Int) -> Float32:
    """`host_tree_sum`, `pinned_sum.mojo:138` (module docstring)."""
    var partials = List[Float32]()
    var chunks = host_chunk_count(n)
    var slab = List[Float32](length=PINNED_SUM_W, fill=Float32(0.0))
    for c in range(chunks):
        for t in range(PINNED_SUM_W):
            var i = c * PINNED_SUM_W + t
            comptime if METRICS_ORACLE_HOST_SABOTAGE:
                # THE SABOTAGE ARM: the chunk boundaries shifted by one
                # value. Wrong on purpose; see METRICS_ORACLE_HOST_SABOTAGE.
                slab[t] = ftz(values[(i + 1) % n]) if i < n else Float32(0.0)
            else:
                slab[t] = ftz(values[i]) if i < n else Float32(0.0)
        var step = PINNED_SUM_W // 2
        while step > 0:
            for t in range(step):
                slab[t] = ftz(slab[t] + slab[t + step])
            step //= 2
        partials.append(slab[0])
    return host_fold_partials(partials, chunks)


def host_canonicalize_nan(x: Float32) -> Float32:
    if x != x:
        return bitcast[DType.float32](CANONICAL_NAN_BITS)
    return x


# ===========================================================================
# The value sabotage helpers. FOR THE SABOTAGE GATE ONLY: every call site is
# inside `comptime if METRICS_ORACLE_HOST_SABOTAGE`, so a production build
# never reaches them.
# ===========================================================================


def host_sabotage_label0(labels: List[Int32], n: Int) -> List[Int32]:
    """A copy of `labels` whose element 0 is another label of the same
    array: its minimum if element 0 is not the minimum, else its maximum.
    It stays inside every class range the caller validated, and moves
    whenever the array holds two distinct labels. Wrong on purpose."""
    var out = labels.copy()
    if n < 1:
        return out^
    var lo = labels[0]
    var hi = labels[0]
    for i in range(n):
        if labels[i] < lo:
            lo = labels[i]
        if labels[i] > hi:
            hi = labels[i]
    out[0] = lo if labels[0] != lo else hi
    return out^


def host_sabotage_shift(v: Float32) -> Float32:
    """`v + 1 + |v|`: never absorbed by rounding for a finite `v`, whatever
    its scale (a subnormal, a zero, a value of 1e4). Wrong on purpose."""
    var a = v if v >= Float32(0.0) else -v
    return ftz(ftz(v + Float32(1.0)) + a)


def host_sabotage_l2sqrt(
    x: List[Float32], i: Int, j: Int, n_cols: Int
) -> Float32:
    """`host_l2sqrt_unexpanded` with row 0 read through
    `host_sabotage_shift` in every feature. Wrong on purpose."""
    var acc = Float32(0.0)
    var bi = i * n_cols
    var bj = j * n_cols
    for f in range(n_cols):
        var xi = ftz(x[bi + f])
        var xj = ftz(x[bj + f])
        if i == 0:
            xi = host_sabotage_shift(xi)
        if j == 0:
            xj = host_sabotage_shift(xj)
        var diff = ftz(xi - xj)
        acc = ftz(identical_mul_add(diff, diff, acc))
    return ftz(identical_sqrt(acc))


# ===========================================================================
# accuracy_score
# ===========================================================================


def host_accuracy_score(
    y_true: List[Int32], y_pred: List[Int32], n: Int
) raises -> Float32:
    """`accuracy_score`, `scores.mojo:106`: the count, one division."""
    if n <= 0:
        raise Error(
            "accuracy_score: n must be positive, got "
            + String(n)
            + " (0 / 0 is refused by name)"
        )
    var count = 0
    for i in range(n):
        var p = y_pred[i]
        comptime if METRICS_ORACLE_HOST_SABOTAGE:
            # THE VALUE SABOTAGE ARM: sample 0's prediction read as a label
            # that flips its agreement. Wrong on purpose.
            if i == 0:
                p = y_true[0] + Int32(1) if p == y_true[0] else y_true[0]
        if y_true[i] == p:
            count += 1
    return Float32(count) / Float32(n)


def host_accuracy_score_ptr(
    y_true: MutPointer[Int32, MutUntrackedOrigin],
    y_pred: MutPointer[Int32, MutUntrackedOrigin],
    n: Int,
) raises -> Float32:
    """Pointer/parallel form of `host_accuracy_score`; integer counts merge exactly."""
    if n <= 0:
        raise Error(
            "accuracy_score: n must be positive, got " + String(n)
            + " (0 / 0 is refused by name)"
        )
    var tasks = host_predict_task_count(n)
    if n < 32768:
        tasks = 1
    var chunk = host_predict_chunk(n, tasks)
    var partials = List[Int64](length=tasks, fill=Int64(0))
    var pp = rebind[MutPointer[Int64, MutUntrackedOrigin]](partials.unsafe_ptr())

    def _rows(task: Int) {imm y_true, imm y_pred, imm n, imm chunk, imm pp}:
        var lo = task * chunk
        var hi = min(lo + chunk, n)
        var count = Int64(0)
        for i in range(lo, hi):
            var truth = y_true.unsafe_load(i)
            var p = y_pred.unsafe_load(i)
            comptime if METRICS_ORACLE_HOST_SABOTAGE:
                if i == 0:
                    p = truth + Int32(1) if p == truth else truth
            if truth == p:
                count += 1
        pp.unsafe_store(task, count)

    if tasks == 1:
        _rows(0)
    else:
        sync_parallelize(_rows, tasks)
    var count = Int64(0)
    for task in range(tasks):
        count += partials[task]
    return Float32(count) / Float32(n)


# ===========================================================================
# r2_score (DEVIATIONS 653, 657)
# ===========================================================================


def host_r2_epilogue(sse: Float32, ssto: Float32) -> Float32:
    """`r2_epilogue`, `scores.mojo:352`."""
    if ssto == Float32(0.0):
        return Float32(1.0) if sse == Float32(0.0) else Float32(0.0)
    var r2 = ftz(Float32(1.0) - ftz(sse / ssto))
    return host_canonicalize_nan(r2)


def host_r2_score(
    y: List[Float32], y_hat: List[Float32], n: Int
) raises -> Float32:
    """`r2_score_parts_traced`, `scores.mojo:262`, the ratio."""
    if n <= 0:
        raise Error("r2_score: n must be positive, got " + String(n))
    var y_sum = host_tree_sum(y, n)
    var ratio = ftz(Float32(1.0) / Float32(n))
    var y_bar = ftz(y_sum * ratio)
    var se = List[Float32](length=n, fill=Float32(0.0))
    var st = List[Float32](length=n, fill=Float32(0.0))
    for i in range(n):
        var yi = y[i]
        var yhi = y_hat[i]
        comptime if METRICS_ORACLE_HOST_SABOTAGE:
            # THE VALUE SABOTAGE ARM: every prediction read as
            # `y_hat + 1 + |y|`. Wrong on purpose.
            var ya = yi if yi >= Float32(0.0) else -yi
            yhi = ftz(ftz(yhi + Float32(1.0)) + ya)
        var d1 = ftz(yi - yhi)
        var d2 = ftz(yi - y_bar)
        se[i] = ftz(d1 * d1)
        st[i] = ftz(d2 * d2)
    var sse = host_tree_sum(se, n)
    var ssto = host_tree_sum(st, n)
    return host_r2_epilogue(sse, ssto)


# ===========================================================================
# The integer kernels of the label metrics
# ===========================================================================


def host_histogram(
    data: List[Int32], n: Int, min_label: Int32, nbins: Int
) -> List[Int32]:
    """`histogram`, `metrics/impl/stats/detail/histogram.mojo:50`: `bins[val
    - min] += 1`, integers."""
    var bins = List[Int32](length=nbins, fill=Int32(0))
    for i in range(n):
        var b = Int(data[i] - min_label)
        bins[b] = bins[b] + Int32(1)
    return bins^


def host_count_unique(arr: List[Int32], n: Int) -> Tuple[Int, Int32, Int32]:
    """`count_unique`, `adjusted_rand_index.mojo:66`: `(numUniques,
    minLabel, maxLabel)`."""
    var min_label = Int32(2147483647)
    var max_label = Int32(-2147483648)
    for i in range(n):
        var v = arr[i]
        if v < min_label:
            min_label = v
        if v > max_label:
            max_label = v
    var total_labels = Int(max_label - min_label + 1)
    var bins = host_histogram(arr, n, min_label, total_labels)
    var n_uniq = 0
    for i in range(total_labels):
        if bins[i] != Int32(0):
            n_uniq += 1
    return (n_uniq, min_label, max_label)


def host_contingency(
    ground_truth: List[Int32],
    predicted: List[Int32],
    n: Int,
    min_label: Int32,
    max_label: Int32,
) raises -> List[Int32]:
    """`contingency_matrix`, `contingency_matrix.mojo:207`, read back as
    `k * k` row-major ints."""
    var k = Int(max_label - min_label + 1)
    if k <= 0:
        raise Error(
            "contingency_matrix: maxLabel < minLabel ("
            + String(max_label)
            + " < "
            + String(min_label)
            + ")"
        )
    var c = List[Int32](length=k * k, fill=Int32(0))
    for i in range(n):
        var gt = ground_truth[i]
        var pd = predicted[i]
        var idx = Int((gt - min_label) * Int32(k) + pd - min_label)
        c[idx] = c[idx] + Int32(1)
    return c^


def host_row_sums(c: List[Int32], k: Int) -> List[Int64]:
    var a = List[Int64]()
    for i in range(k):
        var s = Int64(0)
        for j in range(k):
            s += Int64(c[i * k + j])
        a.append(s)
    return a^


def host_col_sums(c: List[Int32], k: Int) -> List[Int64]:
    var b = List[Int64]()
    for j in range(k):
        var s = Int64(0)
        for i in range(k):
            s += Int64(c[i * k + j])
        b.append(s)
    return b^


def host_n_c_two(v: Int64) -> Int64:
    """`nCTwo`, `adjusted_rand_index.mojo:59`."""
    if v % 2 != 0:
        return ((v - 1) >> 1) * v
    return (v >> 1) * (v - 1)


# ===========================================================================
# adjusted_rand_score
# ===========================================================================


def host_adjusted_rand_score(
    first: List[Int32], second: List[Int32], size: Int
) raises -> Float64:
    """`compute_adjusted_rand_index_traced`, `adjusted_rand_index.mojo:160`
    (module docstring)."""
    comptime if METRICS_ORACLE_HOST_SABOTAGE:
        # THE VALUE SABOTAGE ARM: label 0 of `first` read as another label.
        return _host_adjusted_rand_score(host_sabotage_label0(first, size), second, size)
    else:
        return _host_adjusted_rand_score(first, second, size)


def _host_adjusted_rand_score(
    first: List[Int32], second: List[Int32], size: Int
) raises -> Float64:
    if size < 2:
        return 1.0
    var u1 = host_count_unique(first, size)
    var u2 = host_count_unique(second, size)
    var n_uniq_first = u1[0]
    var n_uniq_second = u2[0]
    var lower = u1[1] if u1[1] < u2[1] else u2[1]
    var upper = u1[2] if u1[2] > u2[2] else u2[2]
    var k = Int(upper - lower + 1)
    if n_uniq_first == n_uniq_second:
        if n_uniq_first == 1 or n_uniq_first == size:
            return 1.0
    var c = host_contingency(first, second, size, lower, upper)
    var n_choose_two_sum = Int64(0)
    for idx in range(k * k):
        n_choose_two_sum += host_n_c_two(Int64(c[idx]))
    var a = host_row_sums(c, k)
    var b = host_col_sums(c, k)
    var a_c_two_sum = Int64(0)
    var b_c_two_sum = Int64(0)
    for i in range(k):
        a_c_two_sum += host_n_c_two(a[i])
        b_c_two_sum += host_n_c_two(b[i])
    var n_choose_two = Float64(size) * Float64(size - 1) / 2.0
    var expected_index = (
        Float64(a_c_two_sum) * Float64(b_c_two_sum) / n_choose_two
    )
    var max_index = (Float64(b_c_two_sum) + Float64(a_c_two_sum)) / 2.0
    var index = Float64(n_choose_two_sum)
    if max_index - expected_index != 0.0:
        return (index - expected_index) / (max_index - expected_index)
    return 0.0


# ===========================================================================
# weighted accuracy_score and weighted r2_score
# (metrics/impl/weighted_scores.mojo; the sabotage arm is host_tree_sum's)
# ===========================================================================


def host_weighted_accuracy(
    y_true: List[Int32], y_pred: List[Int32], w: List[Float32], n: Int
) raises -> Float32:
    """`weighted_accuracy_score`: `tree(w where equal) / tree(w)` through
    `identical_div`."""
    if n <= 0:
        raise Error("weighted accuracy_score: n must be positive, got " + String(n))
    var num = List[Float32](length=n, fill=Float32(0.0))
    var den = List[Float32](length=n, fill=Float32(0.0))
    for i in range(n):
        var wi = ftz(w[i])
        comptime if METRICS_ORACLE_HOST_SABOTAGE:
            # THE WEIGHTED SABOTAGE ARM: sample 0's weight dropped. Wrong on
            # purpose; the tree shift alone cannot move a tied fixture.
            if i == 0 and n > 1:
                wi = Float32(0.0)
        den[i] = wi
        if y_true[i] == y_pred[i]:
            num[i] = wi
    var sn = host_tree_sum(num, n)
    var sd = host_tree_sum(den, n)
    if sd <= Float32(0.0):
        raise Error("weighted accuracy_score: the weights must have positive total")
    return ftz(identical_div(sn, sd))


def host_weighted_accuracy_ptr(
    y_true: MutPointer[Int32, MutUntrackedOrigin],
    y_pred: MutPointer[Int32, MutUntrackedOrigin],
    w: MutPointer[Float32, MutUntrackedOrigin],
    n: Int,
) raises -> Float32:
    """Pointer/slab spelling with the pinned tree unchanged."""
    if n <= 0:
        raise Error("weighted accuracy_score: n must be positive, got " + String(n))
    var chunks = host_chunk_count(n)
    var num_partials = List[Float32](length=chunks, fill=Float32(0.0))
    var den_partials = List[Float32](length=chunks, fill=Float32(0.0))
    var num_slab = List[Float32](length=PINNED_SUM_W, fill=Float32(0.0))
    var den_slab = List[Float32](length=PINNED_SUM_W, fill=Float32(0.0))
    for c in range(chunks):
        for t in range(PINNED_SUM_W):
            var i = c * PINNED_SUM_W + t
            if i < n:
                comptime if METRICS_ORACLE_HOST_SABOTAGE:
                    i = (i + 1) % n
                var wi = ftz(w.unsafe_load(i))
                comptime if METRICS_ORACLE_HOST_SABOTAGE:
                    if i == 0 and n > 1:
                        wi = Float32(0.0)
                den_slab[t] = wi
                num_slab[t] = wi if y_true.unsafe_load(i) == y_pred.unsafe_load(i) else Float32(0.0)
            else:
                num_slab[t] = Float32(0.0)
                den_slab[t] = Float32(0.0)
        var step = PINNED_SUM_W // 2
        while step > 0:
            for t in range(step):
                num_slab[t] = ftz(num_slab[t] + num_slab[t + step])
                den_slab[t] = ftz(den_slab[t] + den_slab[t + step])
            step //= 2
        num_partials[c] = num_slab[0]
        den_partials[c] = den_slab[0]
    var sn = host_fold_partials(num_partials, chunks)
    var sd = host_fold_partials(den_partials, chunks)
    if sd <= Float32(0.0):
        raise Error("weighted accuracy_score: the weights must have positive total")
    return ftz(identical_div(sn, sd))


def host_weighted_r2(
    y: List[Float32], y_hat: List[Float32], w: List[Float32], n: Int
) raises -> Float32:
    """`weighted_r2_score`: the weighted mean, then `tree(w * (y - y_hat)^2)`
    and `tree(w * (y - y_avg)^2)`, then the unweighted metric's epilogue."""
    if n <= 0:
        raise Error("weighted r2_score: n must be positive, got " + String(n))
    var wy = List[Float32](length=n, fill=Float32(0.0))
    var ww = List[Float32](length=n, fill=Float32(0.0))
    for i in range(n):
        var wi = ftz(w[i])
        comptime if METRICS_ORACLE_HOST_SABOTAGE:
            # THE WEIGHTED SABOTAGE ARM, as in host_weighted_accuracy.
            if i == 0 and n > 1:
                wi = Float32(0.0)
        wy[i] = ftz(wi * ftz(y[i]))
        ww[i] = wi
    var swy = host_tree_sum(wy, n)
    var sw = host_tree_sum(ww, n)
    if sw <= Float32(0.0):
        raise Error("weighted r2_score: the weights must have positive total")
    var y_avg = ftz(identical_div(swy, sw))
    var se = List[Float32](length=n, fill=Float32(0.0))
    var st = List[Float32](length=n, fill=Float32(0.0))
    for i in range(n):
        var wi = ftz(w[i])
        comptime if METRICS_ORACLE_HOST_SABOTAGE:
            if i == 0 and n > 1:
                wi = Float32(0.0)
        var yi = ftz(y[i])
        var d1 = ftz(yi - ftz(y_hat[i]))
        var d2 = ftz(yi - y_avg)
        se[i] = ftz(wi * ftz(d1 * d1))
        st[i] = ftz(wi * ftz(d2 * d2))
    var sse = host_tree_sum(se, n)
    var ssto = host_tree_sum(st, n)
    return host_r2_epilogue(sse, ssto)


# ===========================================================================
# fowlkes_mallows_score
# ===========================================================================


def host_fowlkes_mallows(
    first: List[Int32], second: List[Int32], size: Int, lower: Int32, upper: Int32
) raises -> Float64:
    """`fowlkes_mallows_from_contingency`, `metrics/impl/fowlkes_mallows.mojo`
    (scikit-learn `fowlkes_mallows_score`): exact Int64 `tk`, `pk`, `qk`
    from the contingency matrix, then two Float64 divisions, two square
    roots and one multiply, `tk == 0` returning 0.0."""
    var k = Int(upper - lower + 1)
    var c = host_contingency(first, second, size, lower, upper)
    var n = Int64(size)
    var sum_c2 = Int64(0)
    for idx in range(k * k):
        var v = Int64(c[idx])
        sum_c2 += v * v
    var a = host_row_sums(c, k)
    var b = host_col_sums(c, k)
    var sum_b2 = Int64(0)
    var sum_a2 = Int64(0)
    for j in range(k):
        sum_b2 += b[j] * b[j]
    for i in range(k):
        sum_a2 += a[i] * a[i]
    var tk = sum_c2 - n
    comptime if METRICS_ORACLE_HOST_SABOTAGE:
        # THE SABOTAGE ARM: one pair dropped from the numerator's count.
        # Wrong on purpose; see METRICS_ORACLE_HOST_SABOTAGE.
        if tk > 2:
            tk -= 2
    var pk = sum_b2 - n
    var qk = sum_a2 - n
    if tk == 0:
        return 0.0
    var ft = Float64(tk)
    return sqrt(ft / Float64(pk)) * sqrt(ft / Float64(qk))


# ===========================================================================
# entropy, mutual information, homogeneity, completeness, v-measure
# (DEVIATIONS 650, 651: the float epilogues are host code on the GPU path
# too, in the IDENTICAL Float32 arm)
# ===========================================================================


def host_entropy(
    labels: List[Int32], size: Int, lower: Int32, upper: Int32
) raises -> Float64:
    """`entropy_traced`, `entropy.mojo:191`, the IDENTICAL arm."""
    comptime if METRICS_ORACLE_HOST_SABOTAGE:
        # THE VALUE SABOTAGE ARM: label 0 read as another label.
        return _host_entropy(host_sabotage_label0(labels, size), size, lower, upper)
    else:
        return _host_entropy(labels, size, lower, upper)


def _host_entropy(
    labels: List[Int32], size: Int, lower: Int32, upper: Int32
) raises -> Float64:
    if size == 0:
        return 1.0
    var n_unique = Int(upper - lower + 1)
    var counts = host_histogram(labels, size, lower, n_unique)
    var acc = Float32(0.0)
    var fsize = Float32(size)
    for i in range(n_unique):
        var p = ftz(Float32(counts[i]) / fsize)
        if p != Float32(0.0):
            var lp = ftz(identical_log(p))
            acc = ftz(identical_mul_add(-p, lp, acc))
    return Float64(acc)


def host_entropy_ptr(
    labels: MutPointer[Int32, MutUntrackedOrigin],
    size: Int,
    lower: Int32,
    upper: Int32,
) -> Float64:
    """Pointer-input entropy with the original histogram and class fold."""
    if size == 0:
        return 1.0
    var n_unique = Int(upper - lower + 1)
    var counts = List[Int32](length=n_unique, fill=Int32(0))
    for row in range(size):
        var value = labels.unsafe_load(row)
        comptime if METRICS_ORACLE_HOST_SABOTAGE:
            if row == 0 and value == Int32(0):
                value = Int32(1)
        var bin = Int(value - lower)
        counts[bin] = counts[bin] + Int32(1)
    var acc = Float32(0.0)
    var fsize = Float32(size)
    for i in range(n_unique):
        var p = ftz(Float32(counts[i]) / fsize)
        if p != Float32(0.0):
            var lp = ftz(identical_log(p))
            acc = ftz(identical_mul_add(-p, lp, acc))
    return Float64(acc)


def host_mutual_info(
    first: List[Int32], second: List[Int32], size: Int, lower: Int32, upper: Int32
) raises -> Float64:
    """`mutual_info_score_traced`, `mutual_info_score.mojo:240`, the
    IDENTICAL arm."""
    comptime if METRICS_ORACLE_HOST_SABOTAGE:
        # THE VALUE SABOTAGE ARM: label 0 of `first` read as another label.
        return _host_mutual_info(host_sabotage_label0(first, size), second, size, lower, upper)
    else:
        return _host_mutual_info(first, second, size, lower, upper)


def _host_mutual_info(
    first: List[Int32], second: List[Int32], size: Int, lower: Int32, upper: Int32
) raises -> Float64:
    if size <= 0:
        raise Error(
            "mutual_info_score: size must be positive, got "
            + String(size)
            + " (0 / 0 is refused by name)"
        )
    var k = Int(upper - lower + 1)
    var c = host_contingency(first, second, size, lower, upper)
    var a = host_row_sums(c, k)
    var b = host_col_sums(c, k)
    var acc = Float32(0.0)
    var fsize = Float32(size)
    for i in range(k):
        for j in range(k):
            var cij = c[i * k + j]
            var ab = a[i] * b[j]
            if ab != Int64(0) and cij != Int32(0):
                var fc = Float32(cij)
                var l1 = ftz(identical_log(ftz(fsize * fc)))
                var l2 = ftz(identical_log(ftz(Float32(ab))))
                var diff = ftz(l1 - l2)
                acc = ftz(identical_mul_add(fc, diff, acc))
    return Float64(ftz(acc / fsize))


def host_mutual_info_ptr(
    first: MutPointer[Int32, MutUntrackedOrigin],
    second: MutPointer[Int32, MutUntrackedOrigin],
    size: Int,
    lower: Int32,
    upper: Int32,
) raises -> Float64:
    """Pointer-input spelling of ``host_mutual_info``.

    The contingency counts and float epilogue retain exactly the original
    iteration order; this only avoids copying both million-row label arrays
    at the Python binding boundary.
    """
    if size <= 0:
        raise Error(
            "mutual_info_score: size must be positive, got "
            + String(size)
            + " (0 / 0 is refused by name)"
        )
    var k = Int(upper - lower + 1)
    if k <= 0:
        raise Error(
            "contingency_matrix: maxLabel < minLabel ("
            + String(upper)
            + " < "
            + String(lower)
            + ")"
        )
    var c = List[Int32](length=k * k, fill=Int32(0))
    for row in range(size):
        var gt = first.unsafe_load(row)
        comptime if METRICS_ORACLE_HOST_SABOTAGE:
            if row == 0 and gt == Int32(0):
                gt = Int32(1)
        var pd = second.unsafe_load(row)
        var idx = Int((gt - lower) * Int32(k) + pd - lower)
        c[idx] = c[idx] + Int32(1)
    var a = host_row_sums(c, k)
    var b = host_col_sums(c, k)
    var acc = Float32(0.0)
    var fsize = Float32(size)
    for i in range(k):
        for j in range(k):
            var cij = c[i * k + j]
            var ab = a[i] * b[j]
            if ab != Int64(0) and cij != Int32(0):
                var fc = Float32(cij)
                var l1 = ftz(identical_log(ftz(fsize * fc)))
                var l2 = ftz(identical_log(ftz(Float32(ab))))
                var diff = ftz(l1 - l2)
                acc = ftz(identical_mul_add(fc, diff, acc))
    return Float64(ftz(acc / fsize))


def host_homogeneity_score(
    truth: List[Int32], pred: List[Int32], size: Int, lower: Int32, upper: Int32
) raises -> Float64:
    """`homogeneity_score`, `homogeneity_score.mojo:26`."""
    if size == 0:
        return 1.0
    var computed_mi = host_mutual_info(truth, pred, size, lower, upper)
    var computed_entropy = host_entropy(truth, size, lower, upper)
    if computed_entropy != 0.0:
        return computed_mi / computed_entropy
    comptime if METRICS_ORACLE_HOST_SABOTAGE:
        # A constant partition has no alternate in-range label. Changing
        # label 0 cannot test this branch; intentionally break its perfect
        # homogeneity convention instead (CPU H/C/V, negative fixture).
        return 0.0
    return 1.0


def host_v_measure(
    truth: List[Int32],
    pred: List[Int32],
    size: Int,
    lower: Int32,
    upper: Int32,
    beta: Float64,
) raises -> Float64:
    """`v_measure`, `v_measure.mojo:29`."""
    var h = host_homogeneity_score(truth, pred, size, lower, upper)
    var c = host_homogeneity_score(pred, truth, size, lower, upper)
    if c + h == 0.0:
        return 0.0
    var num = (1.0 + beta) * h * c
    var den = beta * h + c
    return num / den


# ===========================================================================
# silhouette (DEVIATIONS 654, 656)
# ===========================================================================


def host_l2sqrt_unexpanded(
    x: List[Float32], i: Int, j: Int, n_cols: Int
) -> Float32:
    """`host_l2sqrt_unexpanded`, `pinned_distance.mojo:59`."""
    var acc = Float32(0.0)
    var bi = i * n_cols
    var bj = j * n_cols
    for f in range(n_cols):
        var diff = ftz(ftz(x[bi + f]) - ftz(x[bj + f]))
        acc = ftz(identical_mul_add(diff, diff, acc))
    return ftz(identical_sqrt(acc))


def host_sil_op(a: Float32, b: Float32) -> Float32:
    """`sil_op`, `silhouette_score.mojo:88`."""
    if (a == Float32(0.0) and b == Float32(0.0)) or a == b:
        return Float32(0.0)
    elif a == Float32(-1.0):
        return Float32(0.0)
    var s: Float32
    if a > b:
        s = ftz(ftz(b - a) / a)
    else:
        s = ftz(ftz(b - a) / b)
    if s != s:
        return Float32(0.0)
    return s


def _float32_max() -> Float32:
    return bitcast[DType.float32](FLOAT32_MAX_BITS)


def host_silhouette(
    x: List[Float32],
    y: List[Int32],
    n_rows: Int,
    n_cols: Int,
    n_labels: Int,
    chunk: Int,
    metric: Int,
    mut scores: List[Float32],
) raises -> Float32:
    """`silhouette_score_launch` and `silhouette_rows_kernel` (module
    docstring). `scores` is cleared and refilled with the `n_rows`
    per-sample coefficients; the return is their mean."""
    if not (n_labels >= 2 and n_labels <= n_rows - 1):
        raise Error(
            "silhouette_score: silhouette Score not defined for the given"
            " number of labels (n_labels="
            + String(n_labels)
            + ", n_rows="
            + String(n_rows)
            + ")"
        )
    if n_cols <= 0:
        raise Error("silhouette_score: n_cols must be positive")
    if metric != DISTANCE_L2_SQRT_UNEXPANDED:
        raise Error(
            "silhouette_score: metric "
            + String(metric)
            + " is refused; only DistanceType::L2SqrtUnexpanded (5, cuML"
            " 'euclidean'/'l2') is implemented (NOT_IMPLEMENTED.tsv)"
        )
    if chunk < 1:
        raise Error(
            "silhouette_score: chunk (cuML chunksize) must be >= 1, got "
            + String(chunk)
        )
    # get_cluster_counts -> countLabels over [0, n_labels)
    var counts = host_histogram(y, n_rows, Int32(0), n_labels)
    scores.clear()
    for _ in range(n_rows):
        scores.append(Float32(0.0))
    var sp = scores.unsafe_ptr()
    var tasks = host_predict_task_count(n_rows)
    var row_chunk = host_predict_chunk(n_rows, tasks)

    def _rows(task: Int) {imm x, imm y, imm counts, imm sp, imm row_chunk, imm n_rows, imm n_cols, imm n_labels}:
        # Scratch belongs to the task.  A score owns one row, and its distance
        # and pinned cluster folds keep exactly the serial statement order.
        var dist = List[Float32](length=n_rows, fill=Float32(0.0))
        var terms = List[Float32](length=n_rows, fill=Float32(0.0))
        var lo = task * row_chunk
        var hi = min(lo + row_chunk, n_rows)
        for i in range(lo, hi):
            var rc = Int(y[i])
            var singleton = Int(counts[rc]) == 1
            var a = Float32(0.0)
            var b = List[Float32](length=n_labels, fill=Float32(0.0))
            for c in range(n_labels):
                if c == rc or Int(counts[c]) == 0:
                    b[c] = Float32(0.0) if singleton else _float32_max()
            if not singleton:
                for j in range(n_rows):
                    if j == i:
                        dist[j] = Float32(0.0)
                    else:
                        comptime if METRICS_ORACLE_HOST_SABOTAGE:
                            # THE VALUE SABOTAGE ARM: row 0 read shifted.
                            dist[j] = host_sabotage_l2sqrt(x, i, j, n_cols)
                        else:
                            dist[j] = host_l2sqrt_unexpanded(x, i, j, n_cols)
                for c in range(n_labels):
                    var cc = Int(counts[c])
                    var denom = Float32(cc - 1) if c == rc else Float32(cc)
                    for j in range(n_rows):
                        if j != i and Int(y[j]) == c:
                            terms[j] = ftz(dist[j] / denom)
                        else:
                            terms[j] = Float32(0.0)
                    var s = host_tree_sum(terms, n_rows)
                    if c == rc:
                        a = ftz(a + ftz(s))
                    else:
                        b[c] = ftz(b[c] + ftz(s))
            var bmin = _float32_max()
            for c in range(n_labels):
                if b[c] < bmin:
                    bmin = b[c]
            sp.unsafe_store(i, host_sil_op(a, bmin))

    if tasks == 1:
        _rows(0)
    else:
        sync_parallelize(_rows, tasks)
    return ftz(host_tree_sum(scores, n_rows) / Float32(n_rows))
