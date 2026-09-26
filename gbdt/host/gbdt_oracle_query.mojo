# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The QueryRMSE target on the host (lane/gbdt-learning-to-rank, stage 2):
the device sequence of `gbdt/targets/kernel/query_rmse.mojo` over the group
kernels of `gbdt/gpu_data/kernel/query_helper.mojo`, restated statement for
statement in plain loops, for `gbdt/host/gbdt_oracle_losses.mojo`'s
symmetric fit.

HOST ONLY. Nothing here imports `max.gpu`, `max.gpu`, a `DeviceContext` or a
module that defines a kernel. Unit row weights only: the host binding refuses
`sample_weight`, so every `weight` below is 1.0 and every `t * w` is `t * 1.0`,
the same float32 multiply the device performs.

WHAT IS RESTATED, IN THE DEVICE'S ORDER

  1. The residual (`query_rmse_residual_kernel`): `v = p * -1`, then
     `v = v + t`, row order, with `p` read through the inverse bin order in
     the estimator.
  2. The query means (`compute_group_means_kernel`): for each query, lane `x`
     of 32 sums `t * w` and `w` over rows `x, x + 32, ...` in row order; the
     32 lane sums meet in the halving tree `line[x] += line[x + s]`,
     `s = 16 .. 1`; the mean is `sum / weight` when the weight is not zero,
     else 0, flushed.
  3. The derivatives (`query_rmse_kernel`): `direction = v - mean[qid]`,
     `der = ftz(w * direction)`; SEARCH planes `[w, der]` at the row,
     ESTIMATION planes `[der, w]` at the row's bin position; the score
     partial `(-w * d) * d` and the magnitudes `|w|`, `|der|`, one of each
     per 256-row block through `_halving_fold`.

THE NEGATIVE CONTROL is the fit's: `GBDT_ORACLE_HOST_SABOTAGE` adds 1.0 to the
walker's lambda in `gbdt_oracle_losses.mojo`, so every QueryRMSE leaf moves.

The restatement is a prediction until measured; the four-column diff of
tools/identity_break.py on the gbdt-query-rmse lane is the measurement.
"""

from checks.numerics import ftz
from gbdt.host.gbdt_oracle import GBDT_MSE_BLOCK, _halving_fold, _partition_stat

#: `QUERY_HELPER_BLOCK_SIZE` and `QUERY_LANES` (`query_helper.mojo`)
comptime GBDT_QUERY_LANES = 32


def query_offsets(group_sizes: List[Int]) -> List[Int]:
    """Each query's first row, the pool's biased offsets at bias 0."""
    var offsets = List[Int](capacity=len(group_sizes))
    var at = 0
    for q in range(len(group_sizes)):
        offsets.append(at)
        at += group_sizes[q]
    return offsets^


def query_ids(group_sizes: List[Int], n_rows: Int) -> List[Int]:
    """`compute_group_ids_kernel`: every row of query `q` holds `q`."""
    var qids = List[Int](length=n_rows, fill=0)
    var at = 0
    for q in range(len(group_sizes)):
        for k in range(group_sizes[q]):
            qids[at + k] = q
        at += group_sizes[q]
    return qids^


def query_means(
    mse_der: List[Float32], offsets: List[Int], group_sizes: List[Int]
) -> List[Float32]:
    """`compute_group_means_kernel` at unit weights."""
    var means = List[Float32](length=len(group_sizes), fill=Float32(0.0))
    for q in range(len(group_sizes)):
        var line_t = List[Float32](length=GBDT_QUERY_LANES, fill=Float32(0.0))
        var line_w = List[Float32](length=GBDT_QUERY_LANES, fill=Float32(0.0))
        var base = offsets[q]
        var size = group_sizes[q]
        for lane in range(GBDT_QUERY_LANES):
            var sum_t = Float32(0.0)
            var sum_w = Float32(0.0)
            var i = lane
            while i < size:
                sum_t = sum_t + mse_der[base + i] * Float32(1.0)
                sum_w = sum_w + Float32(1.0)
                i += GBDT_QUERY_LANES
            line_t[lane] = sum_t
            line_w[lane] = sum_w
        var step = GBDT_QUERY_LANES // 2
        while step > 0:
            for lane in range(step):
                line_t[lane] = line_t[lane] + line_t[lane + step]
                line_w[lane] = line_w[lane] + line_w[lane + step]
            step //= 2
        var mean = Float32(0.0)
        if line_w[0] != Float32(0.0):
            mean = line_t[0] / line_w[0]
        means[q] = ftz(mean)
    return means^


def query_residuals(
    targets: List[Float32],
    cursor: List[Float32],
    inverse: List[Int],
    has_inverse: Bool,
    n_rows: Int,
) -> List[Float32]:
    """`query_rmse_residual_kernel`: `(p * -1) + t` in row order, `p` read at
    `inverse[i]` when the cursor is bin-ordered."""
    var mse_der = List[Float32](length=n_rows, fill=Float32(0.0))
    for i in range(n_rows):
        var p = cursor[inverse[i]] if has_inverse else cursor[i]
        var v = p * Float32(-1.0)
        v = v + targets[i]
        mse_der[i] = v
    return mse_der^


def query_rmse_search_pass(
    targets: List[Float32],
    cursor: List[Float32],
    group_sizes: List[Int],
    n_rows: Int,
    mut stats: List[Float32],
    mut fv_partials: List[Float32],
    mut mag_partials: List[Float32],
):
    """`launch_approximate_query_rmse[False]` with the value and the
    magnitudes: planes `[1, der]` in row order."""
    var offsets = query_offsets(group_sizes)
    var qids = query_ids(group_sizes, n_rows)
    var mse_der = query_residuals(targets, cursor, List[Int](), False, n_rows)
    var means = query_means(mse_der, offsets, group_sizes)
    var blocks = (n_rows + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    for b in range(blocks):
        var s_score = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        var s_w = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        var s_g = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        for t in range(GBDT_MSE_BLOCK):
            var i = b * GBDT_MSE_BLOCK + t
            if i < n_rows:
                var weight = Float32(1.0)
                var val = mse_der[i]
                var mean = means[qids[i]]
                var direction = val - mean
                var der = ftz(weight * direction)
                stats[i] = weight
                stats[n_rows + i] = der
                s_score[t] = -weight * (val - mean) * (val - mean)
                s_w[t] = abs(weight)
                s_g[t] = abs(der)
        fv_partials[b] = _halving_fold(s_score)
        mag_partials[2 * b] = _halving_fold(s_w)
        mag_partials[2 * b + 1] = _halving_fold(s_g)


def query_rmse_value(
    targets: List[Float32], cursor: List[Float32], group_sizes: List[Int], n_rows: Int
) -> List[Float32]:
    """The final learn-loss pass's per-block score partials (the caller folds
    them with `_deterministic_sum_lanes`, as the device's
    `deterministic_sum_lanes_kernel[1]`)."""
    var stats = List[Float32](length=2 * n_rows, fill=Float32(0.0))
    var blocks = (n_rows + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    var fv = List[Float32](length=blocks, fill=Float32(0.0))
    var mags = List[Float32](length=2 * blocks, fill=Float32(0.0))
    query_rmse_search_pass(targets, cursor, group_sizes, n_rows, stats, fv, mags)
    return fv^


def query_rmse_eval(
    targets: List[Float32],
    g_cursor: List[Float32],
    row_index: List[Int],
    group_sizes: List[Int],
    offsets_leaf: List[Int],
    sizes_leaf: List[Int],
    n_rows: Int,
    lambda_reg: Float64,
    mut value: Float64,
    mut gradient: List[Float64],
    mut cached_der2: List[Float64],
):
    """`BinOptimizedOracle.write_value_and_first_derivatives`' querywise arm:
    the point is bin-ordered (`g_cursor[pos]` is row `row_index[pos]`), the
    targets stay in row order, the inverse of `row_index` reads the point
    back, and the planes `[der, 1]` land at each row's bin position. Then
    the per-leaf partition stats, the Hessian plus lambda and the host
    Float32 fold of the value partials, as `_loss_eval`."""
    var inverse = List[Int](length=n_rows, fill=0)
    for pos in range(n_rows):
        inverse[row_index[pos]] = pos
    var offsets = query_offsets(group_sizes)
    var qids = query_ids(group_sizes, n_rows)
    var mse_der = query_residuals(targets, g_cursor, inverse, True, n_rows)
    var means = query_means(mse_der, offsets, group_sizes)
    var blocks = (n_rows + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    var stats = List[Float32](length=2 * n_rows, fill=Float32(0.0))
    var fv = List[Float32](length=blocks, fill=Float32(0.0))
    for b in range(blocks):
        var s_score = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        for t in range(GBDT_MSE_BLOCK):
            var i = b * GBDT_MSE_BLOCK + t
            if i < n_rows:
                var weight = Float32(1.0)
                var val = mse_der[i]
                var mean = means[qids[i]]
                var direction = val - mean
                var dst = inverse[i]
                stats[dst] = ftz(weight * direction)
                stats[n_rows + dst] = weight
                s_score[t] = -weight * (val - mean) * (val - mean)
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
