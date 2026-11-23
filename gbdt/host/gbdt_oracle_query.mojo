# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The QueryRMSE target on the host (lane/gbdt-learning-to-rank, stage 2):
the device sequence of `gbdt/targets/kernel/query_rmse.mojo` over the group
kernels of `gbdt/gpu_data/kernel/query_helper.mojo`, restated statement for
statement in plain loops, for `gbdt/host/gbdt_oracle_losses.mojo`'s
symmetric fit.

HOST ONLY. Nothing here imports `max.gpu`, `std.gpu`, a `DeviceContext` or a
module that defines a kernel. Row weights (lane gbdt-cpu-rest, 2026-11-23):
the fit's `sample_weight` in row order, the device's `has_weights` arm; an
empty list is its unit arm (every `weight` 1.0). The per-row math is the
device's own bodies (`gbdt/targets/query_bodies.mojo`).

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
from gbdt.targets.query_bodies import query_mean_step, query_rmse_der, query_rmse_score
from core.host_parallel import host_parallelize
from core.host_predict_threads import HostF32Ptr, host_predict_chunk
from gbdt.host.gbdt_oracle import (
    GBDT_MSE_BLOCK,
    GbdtIntPtr,
    _gbdt_units,
    _invert_rows,
    _deterministic_sum_lanes,
    _halving_fold,
    _partition_stats_level,
    GBDT_PARTITION_CHUNKS,
)

#: `QUERY_HELPER_BLOCK_SIZE` and `QUERY_LANES` (`query_helper.mojo`)
comptime GBDT_QUERY_LANES = 32


def query_offsets(group_sizes: List[Int]) -> List[Int]:
    """Each query's first row, the pool's biased offsets at bias 0.

    Serial by nature: a running sum over the queries (one add each)."""
    var offsets = List[Int](capacity=len(group_sizes))
    var at = 0
    for q in range(len(group_sizes)):  # driver-loop: over the queries
        offsets.append(at)
        at += group_sizes[q]
    return offsets^


def query_ids(group_sizes: List[Int], n_rows: Int) -> List[Int]:
    """`compute_group_ids_kernel`: every row of query `q` holds `q`."""
    var qids = List[Int](length=n_rows, fill=0)
    # THE SPLIT (lane gbdt-host-par): a query writes its own rows; queries
    # go round robin to `_gbdt_units` tasks from their first rows (the
    # running sum `query_offsets`)
    var n_queries = len(group_sizes)
    var offsets = query_offsets(group_sizes)
    var units = _gbdt_units(n_queries, n_rows)
    var off_p = rebind[GbdtIntPtr](offsets.unsafe_ptr())
    var sz_p = rebind[GbdtIntPtr](group_sizes.unsafe_ptr())
    var qids_p = rebind[GbdtIntPtr](qids.unsafe_ptr())

    def fill(unit: Int) {imm units, imm n_queries, imm off_p, imm sz_p, imm qids_p}:
        var q = unit
        while q < n_queries:
            var at = off_p[q]
            for k in range(sz_p[q]):
                qids_p[at + k] = q
            q += units

    host_parallelize(fill, units)
    _ = len(offsets)
    _ = len(group_sizes)
    return qids^


def query_means(
    mse_der: List[Float32],
    offsets: List[Int],
    group_sizes: List[Int],
    weights: List[Float32] = List[Float32](),
) -> List[Float32]:
    """`compute_group_means_kernel`: `weights` (row order) is its
    `has_weights` arm, empty its unit weight."""
    var means = List[Float32](length=len(group_sizes), fill=Float32(0.0))
    # THE SPLIT (lane gbdt-host-par): the unit is one QUERY, the kernel's
    # block: its 32 lanes' strided sums ascending and their halving tree
    # read the query's own rows. Queries go round robin to `_gbdt_units`
    # tasks; a task holds the two lane lines, every lane rewritten per query.
    var n_queries = len(group_sizes)
    var units = _gbdt_units(n_queries, 4 * len(mse_der) + 64 * n_queries)
    var der_p = rebind[HostF32Ptr](mse_der.unsafe_ptr())
    var off_p = rebind[GbdtIntPtr](offsets.unsafe_ptr())
    var sz_p = rebind[GbdtIntPtr](group_sizes.unsafe_ptr())
    var means_p = rebind[HostF32Ptr](means.unsafe_ptr())
    var has_weights = len(weights) > 0
    # an empty `weights` is never read (`has_weights` guards it)
    var w_p = rebind[HostF32Ptr](weights.unsafe_ptr())

    def query_folds(unit: Int) {imm units, imm n_queries, imm der_p, imm off_p, imm sz_p, imm means_p, imm has_weights, imm w_p}:
        var line_t = List[Float32](length=GBDT_QUERY_LANES, fill=Float32(0.0))
        var line_w = List[Float32](length=GBDT_QUERY_LANES, fill=Float32(0.0))
        var q = unit
        while q < n_queries:
            var base = off_p[q]
            var size = sz_p[q]
            for lane in range(GBDT_QUERY_LANES):
                var sum_t = Float32(0.0)
                var sum_w = Float32(0.0)
                var i = lane
                while i < size:
                    var w = Float32(1.0)
                    if has_weights:
                        w = w_p[base + i]
                    sum_t = query_mean_step(der_p[base + i], w, sum_t)
                    sum_w = sum_w + w
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
            means_p[q] = ftz(mean)
            q += units

    host_parallelize(query_folds, units)
    _ = len(mse_der)
    _ = len(offsets)
    _ = len(group_sizes)
    _ = len(weights)
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
    # one statement per row, the rows in `_gbdt_units` contiguous ranges
    var units = _gbdt_units(n_rows, 4 * n_rows)
    var chunk = host_predict_chunk(n_rows, units)
    var targets_p = rebind[HostF32Ptr](targets.unsafe_ptr())
    var cursor_p = rebind[HostF32Ptr](cursor.unsafe_ptr())
    # an empty `inverse` is never read (`has_inverse` guards it)
    var inv_p = rebind[GbdtIntPtr](inverse.unsafe_ptr())
    var der_p = rebind[HostF32Ptr](mse_der.unsafe_ptr())

    def rows(unit: Int) {imm chunk, imm n_rows, imm has_inverse, imm targets_p, imm cursor_p, imm inv_p, imm der_p}:
        for i in range(unit * chunk, min(n_rows, (unit + 1) * chunk)):
            var p = cursor_p[inv_p[i]] if has_inverse else cursor_p[i]
            var v = p * Float32(-1.0)
            v = v + targets_p[i]
            der_p[i] = v

    host_parallelize(rows, units)
    _ = len(targets)
    _ = len(cursor)
    _ = len(inverse)
    return mse_der^


def query_rmse_search_pass(
    targets: List[Float32],
    cursor: List[Float32],
    group_sizes: List[Int],
    n_rows: Int,
    mut stats: List[Float32],
    mut fv_partials: List[Float32],
    mut mag_partials: List[Float32],
    weights: List[Float32] = List[Float32](),
):
    """`launch_approximate_query_rmse[False]` with the value and the
    magnitudes: planes `[w, der]` in row order (`weights` empty: unit)."""
    var offsets = query_offsets(group_sizes)
    var qids = query_ids(group_sizes, n_rows)
    var mse_der = query_residuals(targets, cursor, List[Int](), False, n_rows)
    var means = query_means(mse_der, offsets, group_sizes, weights)
    var blocks = (n_rows + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    # THE SPLIT (lane gbdt-host-par), the unit of `_logloss_search_pass`:
    # one 256-thread device block owns its rows' two planes and its three
    # partials' halving trees; blocks go round robin to `_gbdt_units` tasks.
    var units = _gbdt_units(blocks, 12 * n_rows)
    var der_p = rebind[HostF32Ptr](mse_der.unsafe_ptr())
    var means_p = rebind[HostF32Ptr](means.unsafe_ptr())
    var qids_p = rebind[GbdtIntPtr](qids.unsafe_ptr())
    var stats_p = rebind[HostF32Ptr](stats.unsafe_ptr())
    var fv_p = rebind[HostF32Ptr](fv_partials.unsafe_ptr())
    var mag_p = rebind[HostF32Ptr](mag_partials.unsafe_ptr())
    var has_weights = len(weights) > 0
    # an empty `weights` is never read (`has_weights` guards it)
    var w_p = rebind[HostF32Ptr](weights.unsafe_ptr())

    def score_blocks(unit: Int) {imm units, imm blocks, imm n_rows, imm der_p, imm means_p, imm qids_p, imm stats_p, imm fv_p, imm mag_p, imm has_weights, imm w_p}:
        var s_score = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        var s_w = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        var s_g = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        var b = unit
        while b < blocks:
            for t in range(GBDT_MSE_BLOCK):
                var i = b * GBDT_MSE_BLOCK + t
                # an out-of-range thread adds 0.0 (the slabs are reused)
                s_score[t] = Float32(0.0)
                s_w[t] = Float32(0.0)
                s_g[t] = Float32(0.0)
                if i < n_rows:
                    var weight = Float32(1.0)
                    if has_weights:
                        weight = w_p[i]
                    var val = der_p[i]
                    var mean = means_p[qids_p[i]]
                    var der = query_rmse_der(val, mean, weight)
                    stats_p[i] = weight
                    stats_p[n_rows + i] = der
                    s_score[t] = query_rmse_score(val, mean, weight)
                    s_w[t] = abs(weight)
                    s_g[t] = abs(der)
            fv_p[b] = _halving_fold(s_score)
            mag_p[2 * b] = _halving_fold(s_w)
            mag_p[2 * b + 1] = _halving_fold(s_g)
            b += units

    host_parallelize(score_blocks, units)
    _ = len(mse_der)
    _ = len(means)
    _ = len(qids)
    _ = len(stats)
    _ = len(fv_partials)
    _ = len(mag_partials)
    _ = len(weights)


def query_rmse_value(
    targets: List[Float32],
    cursor: List[Float32],
    group_sizes: List[Int],
    n_rows: Int,
    weights: List[Float32] = List[Float32](),
) -> List[Float32]:
    """The final learn-loss pass's per-block score partials (the caller folds
    them with `_deterministic_sum_lanes`, as the device's
    `deterministic_sum_lanes_kernel[1]`)."""
    var stats = List[Float32](length=2 * n_rows, fill=Float32(0.0))
    var blocks = (n_rows + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    var fv = List[Float32](length=blocks, fill=Float32(0.0))
    var mags = List[Float32](length=2 * blocks, fill=Float32(0.0))
    query_rmse_search_pass(targets, cursor, group_sizes, n_rows, stats, fv, mags, weights)
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
    weights: List[Float32] = List[Float32](),
):
    """`BinOptimizedOracle.write_value_and_first_derivatives`' querywise arm:
    the point is bin-ordered (`g_cursor[pos]` is row `row_index[pos]`), the
    targets stay in row order, the inverse of `row_index` reads the point
    back, and the planes `[der, w]` land at each row's bin position (`weights`
    in row order, empty: unit). Then
    the per-leaf partition stats, the Hessian plus lambda and the host
    Float32 fold of the value partials, as `_loss_eval`."""
    var inverse = List[Int](length=n_rows, fill=0)
    _invert_rows(row_index, n_rows, inverse)
    var offsets = query_offsets(group_sizes)
    var qids = query_ids(group_sizes, n_rows)
    var mse_der = query_residuals(targets, g_cursor, inverse, True, n_rows)
    var means = query_means(mse_der, offsets, group_sizes, weights)
    var blocks = (n_rows + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    var stats = List[Float32](length=2 * n_rows, fill=Float32(0.0))
    var fv = List[Float32](length=blocks, fill=Float32(0.0))
    # THE SPLIT (lane gbdt-host-par): one 256-thread device block per unit
    # (its rows' planes land at distinct bin positions through the inverse;
    # its score partial's halving tree), round robin to `_gbdt_units` tasks
    var units = _gbdt_units(blocks, 12 * n_rows)
    var der_p = rebind[HostF32Ptr](mse_der.unsafe_ptr())
    var means_p = rebind[HostF32Ptr](means.unsafe_ptr())
    var qids_p = rebind[GbdtIntPtr](qids.unsafe_ptr())
    var inv_p = rebind[GbdtIntPtr](inverse.unsafe_ptr())
    var stats_p = rebind[HostF32Ptr](stats.unsafe_ptr())
    var fv_p = rebind[HostF32Ptr](fv.unsafe_ptr())
    var has_weights = len(weights) > 0
    # an empty `weights` is never read (`has_weights` guards it)
    var w_p = rebind[HostF32Ptr](weights.unsafe_ptr())

    def score_blocks(unit: Int) {imm units, imm blocks, imm n_rows, imm der_p, imm means_p, imm qids_p, imm inv_p, imm stats_p, imm fv_p, imm has_weights, imm w_p}:
        var s_score = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        var b = unit
        while b < blocks:
            for t in range(GBDT_MSE_BLOCK):
                var i = b * GBDT_MSE_BLOCK + t
                # an out-of-range thread adds 0.0 (the slab is reused)
                s_score[t] = Float32(0.0)
                if i < n_rows:
                    var weight = Float32(1.0)
                    if has_weights:
                        weight = w_p[i]
                    var val = der_p[i]
                    var mean = means_p[qids_p[i]]
                    var dst = inv_p[i]
                    stats_p[dst] = query_rmse_der(val, mean, weight)
                    stats_p[n_rows + dst] = weight
                    s_score[t] = query_rmse_score(val, mean, weight)
            fv_p[b] = _halving_fold(s_score)
            b += units

    host_parallelize(score_blocks, units)
    _ = len(mse_der)
    _ = len(means)
    _ = len(qids)
    _ = len(inverse)
    _ = len(weights)
    gradient.clear()
    cached_der2.clear()
    # the leaves' two sums (`compute_partition_stats`): the (leaf, plane)
    # launches across the cores (lane gbdt-host-par)
    var leaf_sums = _partition_stats_level(
        stats, n_rows, 2, GBDT_PARTITION_CHUNKS, offsets_leaf, sizes_leaf, len(sizes_leaf)
    )
    for leaf in range(len(sizes_leaf)):
        gradient.append(Float64(leaf_sums[2 * leaf]))
        cached_der2.append(Float64(leaf_sums[2 * leaf + 1]) + lambda_reg)
    # lane cpu4-gbdt: the oracle's value fold is the device's
    # `deterministic_sum_lanes_kernel[1]` order (was an ascending chain)
    value = Float64(_deterministic_sum_lanes(fv, 1, blocks)[0])
