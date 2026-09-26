# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into a CPU host binding (python/mojolearn/host_surface.py names which); product, not only a check.
"""The metrics the metrics-classification lane reaches that the metrics
lane does not, on the host, for a box with no GPU (2026-09-14).

HOST ONLY. Nothing here imports `max.gpu`, `std.gpu` or a `DeviceContext`,
and no GPU binding imports this file. Each device kernel is spelled a
SECOND time: integer atomics become serial integer loops (an integer sum no
order can move), the float folds are DEVIATION 653's slab tree through
`metrics/host/metrics_oracle.mojo::host_tree_sum`, and the arithmetic
leaves are `checks/numerics.mojo`'s.

WHAT IS RESTATED, AND WHERE THE ORIGINAL IS.

  `host_rand_score`          `compute_rand_index`, `metrics/impl/stats/
                             detail/rand_index.mojo:121`: `size < 2 -> 1.0`,
                             the pairs `j < i` counted into `a` (same in
                             both) and `b` (different in both) in Int64,
                             `(a + b) / (n (n - 1) / 2)` in Float64.
  `host_kl_divergence`       `kld_op` and `kl_divergence_launch_traced`,
                             `kl_divergence.mojo:73, 180`: the flushed
                             operands, `p == 0 -> 0`, `ftz(p * ftz(log p -
                             log q))`, the slab tree, `canonicalize_nan`.
  `host_regression_error`    `error_chunks_kernel` and
                             `error_finalize_kernel`, `metrics/impl/
                             regression_errors.mojo:19, 45`: `ftz(ftz(y) -
                             ftz(p))`, `abs` or `ftz(d * d)`, the slab tree,
                             `ftz(identical_div(total, n))`, the root through
                             `portable_sqrtf`, flushed.
  `host_log_loss`            `log_loss_chunks_kernel` and
                             `log_loss_finalize_kernel`, `metrics/impl/
                             log_loss.mojo:15, 38`: the probability of the
                             true class clipped to `[eps, 1 - eps]`,
                             `ftz(-identical_log(p))`, the slab tree, the
                             optional `ftz(identical_div(total, n))`.
  `host_confusion_matrix_i64`, `host_confusion_matrix_f32`
                             `count_labels_kernel[matrix]` and
                             `confusion_finish_kernel`, `metrics/impl/
                             classification.mojo:23, 51`: counts skipping a
                             `-1` label, the total when normalizing by all,
                             `count_ratio` per cell over the row, column or
                             total.
  `host_precision_recall_fscore`
                             `count_labels_kernel[not matrix]` and
                             `prf_finish_kernel`, `classification.mojo:23,
                             73`: the three O(k) count vectors, the Int64
                             selected sums, and per metric the micro ratio,
                             the per-class ratios, the binary score, the
                             macro and weighted ascending accumulation and
                             its division, the undefined flags.
  `host_binary_ranking`      `binary_ranking`, `metrics/impl/
                             binary_ranking.mojo:139`: the canonical-zero
                             score key, the stable ascending 32-bit radix
                             order (a stable LSD sort by the same key here;
                             every output reads only group boundaries and
                             label counts before and within a group, which
                             any stable ascending order gives), the
                             exclusive prefix of positives, the group
                             starts, then the curve's precision, recall and
                             threshold per group with the terminal (1, 0),
                             or the AUC's Int64 contributions folded
                             ascending and divided once.
  `host_trustworthiness`     `trustworthiness_rank_sum` and
                             `trustworthiness_score_traced`, `metrics/impl/
                             stats/detail/trustworthiness_score.mojo:137,
                             271`: the refusals in their words, the embedded
                             k-NN through `core/knn_host_predict.mojo::
                             host_knn_search` at `k + 1` with the default
                             metric (the host restatement of `knn_search`),
                             the rank of each embedded neighbor counted as
                             `trust_rank_kernel` counts it, the Int64 sum
                             and the Float64 closed form.

THE NEGATIVE CONTROL. `-D MOJOLEARN_HOST_SABOTAGE=1` already shifts every
slab tree's chunk boundaries in `host_tree_sum` (the KL divergence, the three
regression errors and the log loss move), and the k-NN host restatement's own
arm moves the embedded neighbors trustworthiness reads. The integer metrics
(the Rand index, the confusion counts, the precision, recall and F1 ratios,
the ranking) have no fold and are not expected to move.

The restatement is a prediction until measured. The four-column diff of
tools/identity_break.py on the metrics-classification lane is the
measurement.
"""
from std.math import fma
from std.memory import bitcast
from max.algorithm import sync_parallelize

from checks.numerics import ftz, identical_div, identical_log, portable_sqrtf
from core.knn_host_predict import host_knn_search
from core.host_predict_threads import host_predict_chunk, host_predict_task_count
from metrics.host.metrics_oracle import (
    METRICS_ORACLE_HOST_SABOTAGE,
    PINNED_SUM_W,
    host_canonicalize_nan,
    host_l2sqrt_unexpanded,
    host_tree_sum,
)


#: `MAX_CONFUSION_CLASSES` and `MAX_PRF_CLASSES`, `classification.mojo:18-19`.
comptime MAX_CONFUSION_CLASSES = 4096
comptime MAX_PRF_CLASSES = 715827882

#: `TRUST_MAX_K`, `trustworthiness_score.mojo:118`.
comptime TRUST_MAX_K = 256

#: `METRIC_FROM_IS_SQRT`, the default metric `knn_search_traced` receives
#: from the trustworthiness call (`L2SqrtExpanded` under `return_sqrt`).
comptime KNN_METRIC_FROM_IS_SQRT = -1


# ===========================================================================
# rand_score (DEVIATION 652)
# ===========================================================================


def host_rand_score(first: List[Int32], second: List[Int32], size: Int) -> Float64:
    """`compute_rand_index`: the integer pair counts, one Float64 division."""
    if size < 2:
        return 1.0
    # A pair agrees when it is in the same joint-label bucket, or when it
    # differs in both partitions.  Counting those buckets gives exactly the
    # same integer numerator as the quadratic pair walk:
    #
    #   same_both + different_both
    # = total_pairs - same_first - same_second + 2 * same_both.
    #
    # The packed key is injective over the two Int32 bit patterns, including
    # negative raw labels accepted by the binding.
    var first_counts = Dict[Int32, Int64]()
    var second_counts = Dict[Int32, Int64]()
    var joint_counts = Dict[UInt64, Int64]()
    for i in range(size):
        var fi = first[i]
        var si = second[i]
        first_counts[fi] = first_counts.get(fi, Int64(0)) + 1
        second_counts[si] = second_counts.get(si, Int64(0)) + 1
        var key = (UInt64(bitcast[DType.uint32](fi)) << 32) | UInt64(bitcast[DType.uint32](si))
        joint_counts[key] = joint_counts.get(key, Int64(0)) + 1
    var same_first = Int64(0)
    var same_second = Int64(0)
    var same_both = Int64(0)
    for count in first_counts.values():
        same_first += count * (count - 1) // 2
    for count in second_counts.values():
        same_second += count * (count - 1) // 2
    for count in joint_counts.values():
        same_both += count * (count - 1) // 2
    var n = Int64(size)
    var n_choose_two = n * (n - 1) // 2
    var agreeing = n_choose_two - same_first - same_second + 2 * same_both
    return Float64(agreeing) / Float64(n_choose_two)


# ===========================================================================
# kl_divergence (DEVIATIONS 653, 658)
# ===========================================================================


def host_kld_op(model_pdf_in: Float32, candidate_pdf_in: Float32) -> Float32:
    """`kld_op`, `kl_divergence.mojo:73`."""
    var model_pdf = ftz(model_pdf_in)
    var candidate_pdf = ftz(candidate_pdf_in)
    if model_pdf == Float32(0.0):
        return Float32(0.0)
    var lp = ftz(identical_log(model_pdf))
    var lq = ftz(identical_log(candidate_pdf))
    return ftz(model_pdf * ftz(lp - lq))


def host_kl_divergence(p: List[Float32], q: List[Float32], size: Int) raises -> Float32:
    """`kl_divergence_launch_traced`: the per-term map, the tree, the NaN
    washer."""
    if size <= 0:
        raise Error("kl_divergence: size must be positive, got " + String(size))
    var terms = List[Float32](length=size, fill=Float32(0.0))
    for i in range(size):
        terms[i] = host_kld_op(p[i], q[i])
    return host_canonicalize_nan(host_tree_sum(terms, size))


# ===========================================================================
# mean_squared_error, mean_absolute_error, root_mean_squared_error
# ===========================================================================


def host_regression_error(
    y: List[Float32], prediction: List[Float32], n: Int, absolute: Bool, root: Bool,
) raises -> Float32:
    """`regression_error[absolute, root]`: the residual map, the tree, the
    division and the optional root."""
    if n <= 0 or n > 2147483647 or n > len(y) or n > len(prediction):
        raise Error("regression_error: invalid input length")
    # Form each residual directly in its pinned-tree slab.  Materializing an
    # n-value terms list only for host_tree_sum to copy it into the same slab
    # added two full memory passes and a large transient allocation.
    var chunks = (n + PINNED_SUM_W - 1) // PINNED_SUM_W
    var partials = List[Float32](length=chunks, fill=Float32(0.0))
    var slab = List[Float32](length=PINNED_SUM_W, fill=Float32(0.0))
    for c in range(chunks):
        for t in range(PINNED_SUM_W):
            var i = c * PINNED_SUM_W + t
            if i < n:
                comptime if METRICS_ORACLE_HOST_SABOTAGE:
                    i = (i + 1) % n
                var difference = ftz(ftz(y[i]) - ftz(prediction[i]))
                slab[t] = abs(difference) if absolute else ftz(difference * difference)
            else:
                slab[t] = Float32(0.0)
        var step = PINNED_SUM_W // 2
        while step > 0:
            for t in range(step):
                slab[t] = ftz(slab[t] + slab[t + step])
            step //= 2
        partials[c] = slab[0]
    var total = Float32(0.0)
    for c in range(chunks):
        total = ftz(total + partials[c])
    var value = ftz(identical_div(total, Float32(n)))
    if root:
        value = portable_sqrtf(value)
    return ftz(value)


def host_regression_error_ptr(
    y: MutPointer[Float32, MutUntrackedOrigin],
    prediction: MutPointer[Float32, MutUntrackedOrigin],
    n: Int, absolute: Bool, root: Bool,
) raises -> Float32:
    """Pointer/parallel partials with the original ascending final fold."""
    if n <= 0 or n > 2147483647:
        raise Error("regression_error: invalid input length")
    var chunks = (n + PINNED_SUM_W - 1) // PINNED_SUM_W
    var partials = List[Float32](length=chunks, fill=Float32(0.0))
    var pp = rebind[MutPointer[Float32, MutUntrackedOrigin]](partials.unsafe_ptr())
    var tasks = host_predict_task_count(chunks)
    if n < 32768:
        tasks = 1
    var per = host_predict_chunk(chunks, tasks)
    def _chunks(task: Int) {imm y, imm prediction, imm n, imm chunks, imm per, imm pp, imm absolute}:
        var lo = task * per
        var hi = min(lo + per, chunks)
        var slab = List[Float32](length=PINNED_SUM_W, fill=Float32(0.0))
        for c in range(lo, hi):
            for t in range(PINNED_SUM_W):
                var i = c * PINNED_SUM_W + t
                if i < n:
                    comptime if METRICS_ORACLE_HOST_SABOTAGE:
                        i = (i + 1) % n
                    var difference = ftz(ftz(y.unsafe_load(i)) - ftz(prediction.unsafe_load(i)))
                    slab[t] = abs(difference) if absolute else ftz(difference * difference)
                else:
                    slab[t] = Float32(0.0)
            var step = PINNED_SUM_W // 2
            while step > 0:
                for t in range(step):
                    slab[t] = ftz(slab[t] + slab[t + step])
                step //= 2
            pp.unsafe_store(c, slab[0])
    if tasks == 1:
        _chunks(0)
    else:
        sync_parallelize(_chunks, tasks)
    var total = Float32(0.0)
    for c in range(chunks):
        total = ftz(total + partials[c])
    var value = ftz(identical_div(total, Float32(n)))
    if root:
        value = portable_sqrtf(value)
    return ftz(value)


# ===========================================================================
# log_loss
# ===========================================================================


def host_log_loss(
    truth: List[Int32], probability: List[Float32], n: Int, k: Int, normalize: Int,
) raises -> Float32:
    """`log_loss`: the clipped probability of the true class, the negative
    log, the tree, the optional mean."""
    if n <= 0 or n > 2147483647 or k < 2 or k > 2147483647 // n:
        raise Error("log_loss: invalid input dimensions")
    if n > len(truth) or n * k > len(probability) or normalize < 0 or normalize > 1:
        raise Error("log_loss: invalid input length or normalization")
    var terms = List[Float32](length=n, fill=Float32(0.0))
    for i in range(n):
        var p = probability[i * k + Int(truth[i])]
        p = min(max(p, Float32(0.00000011920928955078125)), Float32(0.99999988079071044921875))
        terms[i] = ftz(-identical_log(p))
    var total = host_tree_sum(terms, n)
    if normalize != 0:
        total = ftz(identical_div(total, Float32(n)))
    return total


# ===========================================================================
# confusion_matrix, precision_recall_fscore
# ===========================================================================


def host_count_ratio(numerator: Int64, denominator: Int64, zero: Int32) -> Float32:
    """`count_ratio`, `classification.mojo:44`."""
    if denominator == 0:
        return Float32(zero)
    return ftz(identical_div(Float32(numerator), Float32(denominator)))


def host_confusion_counts(
    y: List[Int32], p: List[Int32], n: Int, k: Int, count_total: Bool,
) -> List[Int32]:
    """`count_labels_kernel[matrix=True, count_total]`: `k * k + 1` cells."""
    var counts = List[Int32](length=k * k + 1, fill=Int32(0))
    for i in range(n):
        var yi = Int(y[i])
        var pi = Int(p[i])
        if yi >= 0 and pi >= 0:
            counts[yi * k + pi] = counts[yi * k + pi] + Int32(1)
            if count_total:
                counts[k * k] = counts[k * k] + Int32(1)
    return counts^


def host_confusion_counts_ptr(
    y: MutPointer[Int32, MutUntrackedOrigin],
    p: MutPointer[Int32, MutUntrackedOrigin],
    n: Int, k: Int, count_total: Bool,
) -> List[Int32]:
    """Pointer/parallel count table; task-local integer tables merge exactly."""
    var tasks = host_predict_task_count(n)
    if n < 32768 or k > 1024:
        tasks = 1
    var width = k * k + 1
    var local = List[Int32](length=tasks * width, fill=Int32(0))
    var lp = rebind[MutPointer[Int32, MutUntrackedOrigin]](local.unsafe_ptr())
    var chunk = host_predict_chunk(n, tasks)

    def _rows(task: Int) {imm y, imm p, imm n, imm k, imm count_total, imm width, imm chunk, imm lp}:
        var lo = task * chunk
        var hi = min(lo + chunk, n)
        var base = task * width
        for i in range(lo, hi):
            var yi = Int(y.unsafe_load(i))
            var pi = Int(p.unsafe_load(i))
            if yi >= 0 and pi >= 0:
                var at = base + yi * k + pi
                lp.unsafe_store(at, lp.unsafe_load(at) + Int32(1))
                if count_total:
                    lp.unsafe_store(base + k * k, lp.unsafe_load(base + k * k) + Int32(1))

    if tasks == 1:
        _rows(0)
    else:
        sync_parallelize(_rows, tasks)
    var counts = List[Int32](length=width, fill=Int32(0))
    for task in range(tasks):
        for i in range(width):
            counts[i] += local[task * width + i]
    return counts^


def host_confusion_matrix_ptr(
    y: MutPointer[Int32, MutUntrackedOrigin],
    p: MutPointer[Int32, MutUntrackedOrigin],
    n: Int, k: Int, normalization: Int,
) raises -> Tuple[List[Int64], List[Float32]]:
    """Pointer/parallel form returning exactly one populated output list."""
    if normalization < 0 or normalization > 3:
        raise Error("confusion_matrix: invalid normalization")
    var counts = host_confusion_counts_ptr(y, p, n, k, normalization == 3)
    if normalization == 0:
        var raw = List[Int64](length=k * k, fill=Int64(0))
        for i in range(k * k):
            raw[i] = Int64(counts[i])
        return (raw^, List[Float32]())
    var out = List[Float32](length=k * k, fill=Float32(0.0))
    if normalization == 3:
        for i in range(k * k):
            out[i] = host_count_ratio(Int64(counts[i]), Int64(counts[k * k]), 0)
        return (List[Int64](), out^)
    for i in range(k):
        var denominator = Int64(0)
        for j in range(k):
            var index = i * k + j if normalization == 1 else j * k + i
            denominator += Int64(counts[index])
        for j in range(k):
            var index = i * k + j if normalization == 1 else j * k + i
            out[index] = host_count_ratio(Int64(counts[index]), denominator, 0)
    return (List[Int64](), out^)


def host_confusion_matrix_i64(
    y: List[Int32], p: List[Int32], n: Int, k: Int, normalization: Int,
) raises -> List[Int64]:
    """`confusion_matrix[DType.int64]`: the raw counts."""
    if normalization < 0 or normalization > 3:
        raise Error("confusion_matrix: invalid normalization")
    if normalization != 0:
        raise Error("confusion_matrix: raw counts require normalization0")
    if n <= 0 or n > 2147483647 or n > len(y) or n > len(p):
        raise Error("classification metrics: invalid n (requires1..Int32.max)")
    if k <= 0 or k > MAX_CONFUSION_CLASSES:
        raise Error("classification metrics: class allocation limit exceeded")
    var counts = host_confusion_counts(y, p, n, k, False)
    var out = List[Int64](length=k * k, fill=Int64(0))
    for i in range(k * k):
        out[i] = Int64(counts[i])
    return out^


def host_confusion_matrix_f32(
    y: List[Int32], p: List[Int32], n: Int, k: Int, normalization: Int,
) raises -> List[Float32]:
    """`confusion_matrix[DType.float32]`: normalized by row (1), column (2)
    or the total (3)."""
    if normalization < 0 or normalization > 3:
        raise Error("confusion_matrix: invalid normalization")
    if normalization == 0:
        raise Error("confusion_matrix: normalized output requires mode1..3")
    if n <= 0 or n > 2147483647 or n > len(y) or n > len(p):
        raise Error("classification metrics: invalid n (requires1..Int32.max)")
    if k <= 0 or k > MAX_CONFUSION_CLASSES:
        raise Error("classification metrics: class allocation limit exceeded")
    var counts = host_confusion_counts(y, p, n, k, normalization == 3)
    var out = List[Float32](length=k * k, fill=Float32(0.0))
    if normalization == 3:
        for i in range(k * k):
            out[i] = host_count_ratio(Int64(counts[i]), Int64(counts[k * k]), 0)
        return out^
    for i in range(k):
        var denominator = Int64(0)
        for j in range(k):
            var index = i * k + j if normalization == 1 else j * k + i
            denominator += Int64(counts[index])
        for j in range(k):
            var index = i * k + j if normalization == 1 else j * k + i
            out[index] = host_count_ratio(Int64(counts[index]), denominator, 0)
    return out^


def host_precision_recall_fscore(
    y: List[Int32], p: List[Int32], n: Int, k: Int, average: Int,
    positive: Int, zero: Int, selected: Int,
) raises -> List[Float32]:
    """`precision_recall_fscore`: `3 * width + 3` values, the three metric
    rows (or scalars) then the three undefined flags."""
    if average < 0 or average > 4 or zero < 0 or zero > 1:
        raise Error("precision_recall_fscore: invalid average/zero_division")
    if selected <= 0 or selected > k or (average == 1 and (positive < 0 or positive >= k)):
        raise Error("precision_recall_fscore: invalid selected/positive class")
    if n <= 0 or n > 2147483647 or n > len(y) or n > len(p):
        raise Error("classification metrics: invalid n (requires1..Int32.max)")
    if k <= 0 or k > MAX_PRF_CLASSES:
        raise Error("classification metrics: class allocation limit exceeded")
    var counts = List[Int32](length=3 * k, fill=Int32(0))
    for i in range(n):
        var yi = Int(y[i])
        var pi = Int(p[i])
        counts[k + yi] = counts[k + yi] + Int32(1)
        counts[2 * k + pi] = counts[2 * k + pi] + Int32(1)
        if yi == pi:
            counts[yi] = counts[yi] + Int32(1)
    var width = selected if average == 0 else 1
    var result = List[Float32](length=3 * width + 3, fill=Float32(0.0))
    var zero32 = Int32(zero)
    var tp_sum = Int64(0)
    var true_sum = Int64(0)
    var pred_sum = Int64(0)
    for c in range(selected):
        tp_sum += Int64(counts[c])
        true_sum += Int64(counts[k + c])
        pred_sum += Int64(counts[2 * k + c])
    for metric in range(3):
        var total = Float32(0)
        var undefined = False
        if average == 2:
            var numerator = tp_sum if metric < 2 else 2 * tp_sum
            var denominator = pred_sum if metric == 0 else (true_sum if metric == 1 else true_sum + pred_sum)
            total = host_count_ratio(numerator, denominator, zero32)
            undefined = denominator == 0
        else:
            var count = 1 if average == 1 else selected
            for j in range(count):
                var c = positive if average == 1 else j
                var tp = Int64(counts[c])
                var support = Int64(counts[k + c])
                var predicted = Int64(counts[2 * k + c])
                var denominator = predicted if metric == 0 else (support if metric == 1 else support + predicted)
                var numerator = tp if metric < 2 else 2 * tp
                var score = host_count_ratio(numerator, denominator, zero32)
                undefined = undefined or denominator == 0
                if average == 0:
                    result[metric * width + j] = score
                elif average == 1:
                    total = score
                else:
                    if average == 4 and true_sum > 0:
                        total = ftz(total + ftz(score * Float32(support)))
                    else:
                        total = ftz(total + score)
            if average == 3 or average == 4:
                var denominator = true_sum if average == 4 and true_sum > 0 else Int64(selected)
                total = ftz(identical_div(total, Float32(denominator)))
        if average != 0:
            result[metric] = total
        result[3 * width + metric] = Float32(1 if undefined else 0)
    return result^


# ===========================================================================
# roc_auc_score, precision_recall_curve
# ===========================================================================


def _ranking_key(score: Float32) -> UInt32:
    """`ranking_keys_kernel`: both zeros to `+0.0`, then the order-preserving
    unsigned key."""
    var bits = bitcast[DType.uint32](score)
    if (bits & UInt32(0x7fffffff)) == 0:
        bits = UInt32(0)
    if (bits & UInt32(0x80000000)) != 0:
        return ~bits
    return bits | UInt32(0x80000000)


def host_binary_ranking(
    y: List[Int32], scores: List[Float32], n: Int, curve: Bool,
) raises -> Tuple[List[Float32], Int]:
    """`binary_ranking[curve]`: the output buffer (`3 n + 2` for the curve,
    one value for the AUC) and the group count."""
    if n <= 0 or n > 2147483647 or n > len(y) or n > len(scores):
        raise Error("binary ranking: invalid input length")
    var keys = List[UInt32](length=n, fill=UInt32(0))
    var labels = List[UInt32](length=n, fill=UInt32(0))
    for i in range(n):
        keys[i] = _ranking_key(scores[i])
        labels[i] = UInt32(y[i])
    # The stable ascending order, four 8-bit LSD passes.
    var tk = List[UInt32](length=n, fill=UInt32(0))
    var tl = List[UInt32](length=n, fill=UInt32(0))
    for pass_ in range(4):
        var shift = UInt32(8 * pass_)
        var bucket = List[Int](length=257, fill=0)
        for i in range(n):
            var bkt = Int((keys[i] >> shift) & UInt32(255))
            bucket[bkt + 1] = bucket[bkt + 1] + 1
        for b in range(256):
            bucket[b + 1] = bucket[b + 1] + bucket[b]
        for i in range(n):
            var bkt = Int((keys[i] >> shift) & UInt32(255))
            var dst = bucket[bkt]
            tk[dst] = keys[i]
            tl[dst] = labels[i]
            bucket[bkt] = dst + 1
        # The next byte pass can consume the destination buffers directly.
        # Four passes is even, so the final sorted data still ends in `keys`
        # and `labels`; swapping avoids a fifth full-array walk per pass.
        swap(keys, tk)
        swap(labels, tl)
    # The exclusive prefix of positives (bit 0 of the label).
    var prefix = List[Int64](length=n, fill=Int64(0))
    var run = Int64(0)
    for i in range(n):
        prefix[i] = run
        run += Int64(labels[i] & UInt32(1))
    var starts = List[Int]()
    for i in range(n):
        if i == 0 or keys[i] != keys[i - 1]:
            starts.append(i)
    var m = len(starts)
    var positives = prefix[n - 1] + Int64(labels[n - 1])
    var output_size = 3 * n + 2 if curve else 1
    var output = List[Float32](length=output_size, fill=Float32(0.0))
    if curve:
        for group in range(m):
            var i = starts[group]
            var before = prefix[i]
            var tp = positives - before
            output[group] = ftz(identical_div(Float32(tp), Float32(n - i)))
            var recall = Float32(1) if positives == 0 else ftz(identical_div(Float32(tp), Float32(positives)))
            output[n + 1 + group] = recall
            var key = keys[i]
            var bits = key & UInt32(0x7fffffff) if (key & UInt32(0x80000000)) != 0 else ~key
            output[2 * (n + 1) + group] = bitcast[DType.float32](bits)
        output[m] = Float32(1)
        output[n + 1 + m] = Float32(0)
        return (output^, m)
    var negatives = Int64(n) - positives
    var total = Int64(0)
    for group in range(m):
        var i = starts[group]
        var j = n if group + 1 == m else starts[group + 1]
        var before = prefix[i]
        var through = positives if j == n else prefix[j]
        var group_positive = through - before
        var group_negative = Int64(j - i) - group_positive
        total += group_positive * (2 * (Int64(i) - before) + group_negative)
    output[0] = ftz(identical_div(Float32(total), Float32(2 * positives * negatives)))
    return (output^, m)


# ===========================================================================
# trustworthiness (DEVIATION 655)
# ===========================================================================


def host_trustworthiness(
    x: List[Float32], x_embedded: List[Float32], n: Int, m: Int, d: Int,
    n_neighbors: Int, batch_size: Int,
) raises -> Float64:
    """`trustworthiness_score_traced` and `trustworthiness_rank_sum`."""
    if batch_size < 1:
        raise Error("trustworthiness: batchSize must be >= 1")
    if n_neighbors < 1:
        raise Error("trustworthiness: n_neighbors must be >= 1")
    if 2 * n_neighbors >= n:
        raise Error(
            "trustworthiness: n_neighbors ("
            + String(n_neighbors)
            + ") must be >= 1 and < n_samples / 2; n_samples is "
            + String(n)
            + " (cuML trustworthiness.pyx:114)"
        )
    if n_neighbors + 1 > TRUST_MAX_K:
        raise Error(
            "trustworthiness: n_neighbors + 1 = "
            + String(n_neighbors + 1)
            + " exceeds TRUST_MAX_K = "
            + String(TRUST_MAX_K)
            + " (refused by name)"
        )
    var k1 = n_neighbors + 1
    var emb_dist = List[Float32](length=n * k1, fill=Float32(0.0))
    var emb_ind = List[UInt32](length=n * k1, fill=UInt32(0))
    host_knn_search(
        x_embedded, n, x_embedded, n, d, k1, KNN_METRIC_FROM_IS_SQRT, True,
        emb_dist, emb_ind,
    )
    var total = Int64(0)
    var d_e = List[Float32](length=k1, fill=Float32(0.0))
    var e_idx = List[Int](length=k1, fill=0)
    var counts = List[Int32](length=k1, fill=Int32(0))
    for row in range(n):
        for e in range(k1):
            var idx = Int(emb_ind[row * k1 + e])
            e_idx[e] = idx
            d_e[e] = host_l2sqrt_unexpanded(x, row, idx, m)
            counts[e] = Int32(0)
        for j in range(n):
            var dj = host_l2sqrt_unexpanded(x, row, j, m)
            for q in range(k1):
                var de = d_e[q]
                if dj < de or (dj == de and j < e_idx[q]):
                    counts[q] = counts[q] + Int32(1)
        var row_total = Int32(0)
        for q in range(k1):
            var tmp = counts[q] - Int32(k1) + Int32(1)
            if tmp > Int32(0):
                row_total += tmp
        total += Int64(row_total)
    var t = Float64(total)
    var nn = Float64(n)
    var kk = Float64(n_neighbors)
    # `1 - q*t` in ONE rounding, as the default build fused it (lane/pinned-mul-contract-free)
    return fma(-(2.0 / ((nn * kk) * ((2.0 * nn) - (3.0 * kk) - 1.0))), t, 1.0)
