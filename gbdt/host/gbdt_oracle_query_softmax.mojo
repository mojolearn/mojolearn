# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The QuerySoftMax target on the host (lane/gbdt-rest): the device sequence of
`gbdt/targets/kernel/query_softmax.mojo` restated statement for statement in
plain loops, for `gbdt/host/gbdt_oracle_losses.mojo`'s symmetric fit.

HOST ONLY. Nothing here imports `max.gpu`, `std.gpu`, a `DeviceContext` or a
module that defines a kernel. Unit row weights only: the host binding refuses
`sample_weight`, so every `weight` below is 1.0 and every `t * w` is
`t * 1.0`, the same float32 multiply the device performs. The host binding
restates the Cosine search only (it refuses the other score functions), so
the search weight plane is the row weight.

WHAT IS RESTATED, IN THE DEVICE'S ORDER

  1. The gather (`query_softmax_gather_kernel`): the point in row order, read
     through the inverse bin order in the estimator.
  2. The group maxima and weighted-target sums (`compute_group_maximals_kernel`):
     per query, lane `x` of 32 walks rows `x, x + 32, ...`, keeping the
     `fmaxf` of the approx over rows with `w > 0` from `-FLT_MAX` and the
     float32 sum of `t * w`; the 32 lanes meet in the halving tree
     `line[x] = op(line[x], line[x + s])`, `s = 16 .. 1`; the sum is flushed.
  3. The exponents (`query_softmax_exponents_kernel`):
     `ftz(identical_exp(beta * (a - max[qid])) * w)`, row order.
  4. The group sums of the exponents (`compute_group_sums_kernel`), the same
     tree with add, flushed.
  5. The derivatives and the value (`query_softmax_kernel`), per 256-row block
     through `_halving_fold`.

THE NEGATIVE CONTROL is the fit's: `GBDT_ORACLE_HOST_SABOTAGE` adds 1.0 to the
walker's lambda in `gbdt_oracle_losses.mojo`, so every QuerySoftMax leaf moves.

The restatement is a prediction until measured; the four-column diff of
tools/identity_break.py on the gbdt-query-softmax lane is the measurement.
"""

from checks.numerics import ftz, identical_exp, identical_log
from gbdt.host.gbdt_oracle import GBDT_MSE_BLOCK, _halving_fold, _partition_stat
from gbdt.host.gbdt_oracle_query import query_ids, query_offsets

#: `QUERY_LANES` (`query_helper.mojo`)
comptime _LANES = 32


@fieldwise_init
struct HostQuerySoftMax(ImplicitlyCopyable, Movable):
    """The loss parameters as Float32 (the device kernel's arguments) and the
    Float64 loss weight."""

    var lambda_reg: Float32
    var beta: Float32
    var total_weighted_target: Float64


def _fmaxf(a: Float32, b: Float32) -> Float32:
    """`query_softmax_fmax`."""
    if a != a:
        return b
    if b != b:
        return a
    return b if a < b else a


def query_softmax_host_total(targets: List[Float32]) raises -> Float64:
    """`query_softmax_total_weighted_target` at unit weights."""
    var total = Float64(0.0)
    for i in range(len(targets)):
        total += Float64(targets[i]) * Float64(1.0)
    if not (total > Float64(0.0)):
        raise Error(
            "Observation targets and weights should be greater or equal zero."
            " Total weighted target should be greater, than zero"
        )
    return total


def _softmax_rows(
    targets: List[Float32],
    point: List[Float32],
    group_sizes: List[Int],
    n_rows: Int,
    beta: Float32,
    mut approx_exp: List[Float32],
    mut approx_sum: List[Float32],
    mut sum_wt: List[Float32],
    mut qids: List[Int],
):
    """Steps 2 to 4 over the row-order point."""
    var offsets = query_offsets(group_sizes)
    qids = query_ids(group_sizes, n_rows)
    var q_count = len(group_sizes)
    var maximals = List[Float32](length=q_count, fill=Float32(0.0))
    sum_wt = List[Float32](length=q_count, fill=Float32(0.0))
    for q in range(q_count):
        var line_max = List[Float32](length=_LANES, fill=Float32(0.0))
        var line_sum = List[Float32](length=_LANES, fill=Float32(0.0))
        var base = offsets[q]
        var size = group_sizes[q]
        for lane in range(_LANES):
            var m = -Float32.MAX_FINITE
            var s = Float32(0.0)
            var i = lane
            while i < size:
                var t = targets[base + i]
                var w = Float32(1.0)
                var a = point[base + i]
                if w > Float32(0.0):
                    m = _fmaxf(m, a)
                s = s + t * w
                i += _LANES
            line_max[lane] = m
            line_sum[lane] = s
        var step = _LANES // 2
        while step > 0:
            for lane in range(step):
                line_max[lane] = _fmaxf(line_max[lane], line_max[lane + step])
                line_sum[lane] = line_sum[lane] + line_sum[lane + step]
            step //= 2
        maximals[q] = line_max[0]
        sum_wt[q] = ftz(line_sum[0])
    approx_exp = List[Float32](length=n_rows, fill=Float32(0.0))
    for i in range(n_rows):
        var weight = Float32(1.0)
        approx_exp[i] = ftz(identical_exp(beta * (point[i] - maximals[qids[i]])) * weight)
    approx_sum = List[Float32](length=q_count, fill=Float32(0.0))
    for q in range(q_count):
        var line = List[Float32](length=_LANES, fill=Float32(0.0))
        var base = offsets[q]
        var size = group_sizes[q]
        for lane in range(_LANES):
            var s = Float32(0.0)
            var i = lane
            while i < size:
                s = s + approx_exp[base + i]
                i += _LANES
            line[lane] = s
        var step = _LANES // 2
        while step > 0:
            for lane in range(step):
                line[lane] = line[lane] + line[lane + step]
            step //= 2
        approx_sum[q] = ftz(line[0])


def _row_terms(
    target_val: Float32,
    approx: Float32,
    a_sum: Float32,
    sum_targets: Float32,
    lambda_reg: Float32,
    beta: Float32,
    mut der: Float32,
    mut der2: Float32,
    mut score: Float32,
):
    """`query_softmax_kernel`'s in-range body at weight 1.0."""
    var weight = Float32(1.0)
    var softmax = approx / a_sum
    var wt = weight * target_val
    var guard = weight > Float32(0.0) and sum_targets > Float32(0.0)
    var first = Float32(0.0)
    if guard:
        first = (-sum_targets) * softmax
    der = ftz(beta * (first + wt))
    der2 = Float32(0.0)
    if guard:
        der2 = ftz(beta * sum_targets * (beta * softmax * (Float32(1.0) - softmax) + lambda_reg))
    score = Float32(0.0)
    if weight > Float32(0.0) and target_val > Float32(0.0):
        score = wt * identical_log(softmax)


def query_softmax_search_pass(
    targets: List[Float32],
    cursor: List[Float32],
    group_sizes: List[Int],
    n_rows: Int,
    params: HostQuerySoftMax,
    mut stats: List[Float32],
    mut fv_partials: List[Float32],
    mut mag_partials: List[Float32],
):
    """`launch_query_softmax_with[False, False]` with the value and the
    magnitudes: planes `[1, der]` in row order."""
    var approx_exp = List[Float32]()
    var approx_sum = List[Float32]()
    var sum_wt = List[Float32]()
    var qids = List[Int]()
    _softmax_rows(targets, cursor, group_sizes, n_rows, params.beta, approx_exp, approx_sum, sum_wt, qids)
    var blocks = (n_rows + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    for b in range(blocks):
        var s_score = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        var s_w = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        var s_g = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        for t in range(GBDT_MSE_BLOCK):
            var i = b * GBDT_MSE_BLOCK + t
            if i < n_rows:
                var der = Float32(0.0)
                var der2 = Float32(0.0)
                var score = Float32(0.0)
                var q = qids[i]
                _row_terms(
                    targets[i], approx_exp[i], approx_sum[q], sum_wt[q],
                    params.lambda_reg, params.beta, der, der2, score,
                )
                var weight = Float32(1.0)
                stats[i] = weight
                stats[n_rows + i] = der
                s_score[t] = score
                s_w[t] = abs(weight)
                s_g[t] = abs(der)
        fv_partials[b] = _halving_fold(s_score)
        mag_partials[2 * b] = _halving_fold(s_w)
        mag_partials[2 * b + 1] = _halving_fold(s_g)


def query_softmax_value(
    targets: List[Float32],
    cursor: List[Float32],
    group_sizes: List[Int],
    n_rows: Int,
    params: HostQuerySoftMax,
) -> List[Float32]:
    """The final learn-loss pass's per-block score partials."""
    var stats = List[Float32](length=2 * n_rows, fill=Float32(0.0))
    var blocks = (n_rows + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    var fv = List[Float32](length=blocks, fill=Float32(0.0))
    var mags = List[Float32](length=2 * blocks, fill=Float32(0.0))
    query_softmax_search_pass(targets, cursor, group_sizes, n_rows, params, stats, fv, mags)
    return fv^


def query_softmax_eval(
    targets: List[Float32],
    g_cursor: List[Float32],
    row_index: List[Int],
    group_sizes: List[Int],
    offsets_leaf: List[Int],
    sizes_leaf: List[Int],
    n_rows: Int,
    params: HostQuerySoftMax,
    lambda_reg: Float64,
    mut value: Float64,
    mut gradient: List[Float64],
    mut cached_der2: List[Float64],
):
    """`BinOptimizedOracle.write_value_and_first_derivatives`' QuerySoftMax
    arm: the bin-ordered point read back to row order through the inverse of
    `row_index`, the planes `[der, der2]` at each row's bin position, then the
    per-leaf partition stats, the Hessian plus lambda and the host Float32
    fold of the value partials, as `query_rmse_eval`."""
    var inverse = List[Int](length=n_rows, fill=0)
    for pos in range(n_rows):
        inverse[row_index[pos]] = pos
    var point = List[Float32](length=n_rows, fill=Float32(0.0))
    for i in range(n_rows):
        point[i] = g_cursor[inverse[i]]
    var approx_exp = List[Float32]()
    var approx_sum = List[Float32]()
    var sum_wt = List[Float32]()
    var qids = List[Int]()
    _softmax_rows(targets, point, group_sizes, n_rows, params.beta, approx_exp, approx_sum, sum_wt, qids)
    var blocks = (n_rows + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    var stats = List[Float32](length=2 * n_rows, fill=Float32(0.0))
    var fv = List[Float32](length=blocks, fill=Float32(0.0))
    for b in range(blocks):
        var s_score = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        for t in range(GBDT_MSE_BLOCK):
            var i = b * GBDT_MSE_BLOCK + t
            if i < n_rows:
                var der = Float32(0.0)
                var der2 = Float32(0.0)
                var score = Float32(0.0)
                var q = qids[i]
                _row_terms(
                    targets[i], approx_exp[i], approx_sum[q], sum_wt[q],
                    params.lambda_reg, params.beta, der, der2, score,
                )
                var dst = inverse[i]
                stats[dst] = der
                stats[n_rows + dst] = der2
                s_score[t] = score
        fv[b] = _halving_fold(s_score)
    gradient.clear()
    cached_der2.clear()
    for leaf in range(len(sizes_leaf)):
        gradient.append(
            Float64(_partition_stat(stats, n_rows, 0, offsets_leaf[leaf], sizes_leaf[leaf]))
        )
        cached_der2.append(
            Float64(_partition_stat(stats, n_rows, 1, offsets_leaf[leaf], sizes_leaf[leaf]))
            + lambda_reg
        )
    var fv32 = Float32(0.0)
    for b in range(blocks):
        fv32 += fv[b]
    value = Float64(fv32)
