# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""GPU-free. The host restatement of `gbdt/targets/kernel/yeti_rank.mojo` for the CPU
column: the same task table (`gbdt/data/yeti_rank_tasks.mojo`), the same
per-task order (draws per round, thread and lane; the stable order by query
ascending and key descending; for each lane phase 1 then phase 2), the same
flushes, and `identical_exp` / `identical_pow` where the device calls
`routed_exp` / `identical_pow` (the pair that stage 3's gbdt-pair-logit lane
shows bit-equal). The unit-weight query means come from `query_means`, the
host restatement of the device's 32-lane stride and halving tree.
"""

from std.memory import bitcast

from checks.numerics import ftz, identical_exp, identical_pow
from gbdt.data.yeti_rank_tasks import (
    YETI_TASK_POSITIONS,
    YetiRankTasks,
    yeti_rank_advance_seed32,
    yeti_rank_cuda_seed,
    yeti_rank_task_seed,
)
from gbdt.host.gbdt_oracle import GBDT_MSE_BLOCK, _halving_fold, _partition_stat
from gbdt.host.gbdt_oracle_query import query_ids, query_means, query_offsets

comptime _THREADS = 256
comptime _LANES = 4


def _b_first(key_a: UInt32, idx_a: UInt32, key_b: UInt32, idx_b: UInt32) -> Bool:
    """The kernel's merge predicate: (key_b, idx_b) strictly first."""
    var qa = (idx_a >> UInt32(10)) & UInt32(1023)
    var qb = (idx_b >> UInt32(10)) & UInt32(1023)
    if qb != qa:
        return qb < qa
    return key_b > key_a


def yeti_rank_row_derivatives(
    targets: List[Float32],
    weights: List[Float32],
    has_weights: Bool,
    point: List[Float32],
    group_sizes: List[Int],
    tasks: YetiRankTasks,
    seed: UInt64,
    permutations: Int,
    decay_speed: Float32,
    n_rows: Int,
    mut der: List[Float32],
    mut pair_weight_sum: List[Float32],
):
    """One call's derivative and pair weight per row, `point` in row order
    (`yeti_rank_center_kernel` then `yeti_rank_task_kernel`)."""
    var offsets = query_offsets(group_sizes)
    var qids = query_ids(group_sizes, n_rows)
    var means = query_means(point, offsets, group_sizes)
    var centered = List[Float32](length=n_rows, fill=Float32(0.0))
    for i in range(n_rows):
        centered[i] = ftz(point[i] - means[qids[i]])
    der = List[Float32](length=n_rows, fill=Float32(0.0))
    pair_weight_sum = List[Float32](length=n_rows, fill=Float32(0.0))
    var cuda_seed = yeti_rank_cuda_seed(seed)
    var s_exp = List[Float32](length=YETI_TASK_POSITIONS, fill=Float32(0.0))
    var s_relev = List[Float32](length=YETI_TASK_POSITIONS, fill=Float32(0.0))
    var s_src = List[UInt32](length=YETI_TASK_POSITIONS, fill=UInt32(0))
    var s_begin = List[Int](length=YETI_TASK_POSITIONS, fill=0)
    var s_key = List[UInt32](length=YETI_TASK_POSITIONS, fill=UInt32(0))
    var s_idx = List[UInt32](length=YETI_TASK_POSITIONS, fill=UInt32(0))
    var t_key = List[UInt32](length=YETI_TASK_POSITIONS, fill=UInt32(0))
    var t_idx = List[UInt32](length=YETI_TASK_POSITIONS, fill=UInt32(0))
    var seeds = List[UInt32](length=_THREADS, fill=UInt32(0))
    for task in range(tasks.count()):
        var offset = Int(tasks.offsets[task])
        var size = Int(tasks.sizes[task])
        var task_qid = tasks.qids[task]
        var first_qid = UInt32(qids[offset])
        var pad_qid = UInt32(qids[offset + size - 1]) + UInt32(1) - first_qid
        for tid in range(_THREADS):
            seeds[tid] = yeti_rank_task_seed(task_qid, tid, cuda_seed)
        var prev_qid = UInt32(0xFFFFFFFF)
        var begin = 0
        for p in range(YETI_TASK_POSITIONS):
            var qid = pad_qid
            if p < size:
                qid = UInt32(qids[offset + p]) - first_qid
            if p == 0 or qid != prev_qid:
                begin = p
            prev_qid = qid
            s_begin[p] = begin
            s_src[p] = UInt32(p) | (qid << UInt32(10))
            if p < size:
                var w = Float32(1.0)
                if has_weights:
                    w = weights[offset + p]
                s_relev[p] = ftz(targets[offset + p] * w)
                s_exp[p] = identical_exp(min(centered[offset + p], Float32(70.0)))
            else:
                s_relev[p] = Float32(1000.0)
                s_exp[p] = Float32(1000.0)
        for _ in range(permutations):
            for tid in range(_THREADS):
                var s = seeds[tid]
                for k in range(_LANES):
                    var p = tid + _THREADS * k
                    var val = Float32(-1000.0)
                    if p < size:
                        val = s_exp[p]
                    s = yeti_rank_advance_seed32(s)
                    var uni = Float32(s) * Float32(2.328306435996595e-10)
                    val = val * (uni / (Float32(1.000001) - uni))
                    var bits = bitcast[DType.uint32](val)
                    if (bits & UInt32(0x80000000)) != UInt32(0):
                        bits = bits ^ UInt32(0xFFFFFFFF)
                    else:
                        bits = bits ^ UInt32(0x80000000)
                    s_key[p] = bits
                    s_idx[p] = s_src[p]
                seeds[tid] = s
            var width = 1
            var in_tmp = False
            while width < YETI_TASK_POSITIONS:
                var lo = 0
                while lo < YETI_TASK_POSITIONS:
                    var mid = lo + width
                    var hi = lo + 2 * width
                    var a = lo
                    var b = mid
                    var o = lo
                    while o < hi:
                        var take_b = a >= mid
                        if (not take_b) and b < hi:
                            if in_tmp:
                                take_b = _b_first(t_key[a], t_idx[a], t_key[b], t_idx[b])
                            else:
                                take_b = _b_first(s_key[a], s_idx[a], s_key[b], s_idx[b])
                        var from_pos = a
                        if take_b:
                            from_pos = b
                            b += 1
                        else:
                            a += 1
                        if in_tmp:
                            s_key[o] = t_key[from_pos]
                            s_idx[o] = t_idx[from_pos]
                        else:
                            t_key[o] = s_key[from_pos]
                            t_idx[o] = s_idx[from_pos]
                        o += 1
                    lo = hi
                in_tmp = not in_tmp
                width = width * 2
            for k in range(_LANES):
                for phase in range(2):
                    for tid in range(_THREADS):
                        var j = tid + _THREADS * k
                        var qb = s_begin[j]
                        if j != qb:
                            var idx1 = Int(s_idx[j - 1] & UInt32(1023))
                            var idx2 = Int(s_idx[j] & UInt32(1023))
                            var relev1 = s_relev[idx1]
                            var relev2 = s_relev[idx2]
                            var approx1 = s_exp[idx1]
                            var approx2 = s_exp[idx2]
                            var decay = Float32(0.15) * identical_pow(
                                decay_speed, Float32(j - qb - 1)
                            )
                            var pw = ftz(decay * abs(relev1 - relev2) / Float32(permutations))
                            var sel = -approx1
                            if relev1 > relev2:
                                sel = approx2
                            var ll = ftz(pw * sel / (approx2 + approx1))
                            if phase == 0:
                                if idx1 < size:
                                    var r1 = offset + idx1
                                    pair_weight_sum[r1] = ftz(pair_weight_sum[r1] + pw)
                                    der[r1] = ftz(der[r1] + ll)
                            elif idx2 < size:
                                var r2 = offset + idx2
                                pair_weight_sum[r2] = ftz(pair_weight_sum[r2] + pw)
                                der[r2] = ftz(der[r2] + (-ll))


def yeti_rank_search_pass(
    targets: List[Float32],
    weights: List[Float32],
    has_weights: Bool,
    cursor: List[Float32],
    group_sizes: List[Int],
    tasks: YetiRankTasks,
    seed: UInt64,
    permutations: Int,
    decay_speed: Float32,
    n_rows: Int,
    mut stats: List[Float32],
    mut fv_partials: List[Float32],
    mut mag_partials: List[Float32],
):
    """`launch_yeti_rank_with[False]`: planes `[pair weight, der]` in row
    order, every value partial 0.0, the magnitudes per 256 rows as
    `pair_logit_search_pass` folds them."""
    var der = List[Float32]()
    var pw = List[Float32]()
    yeti_rank_row_derivatives(
        targets, weights, has_weights, cursor, group_sizes, tasks, seed,
        permutations, decay_speed, n_rows, der, pw,
    )
    for b in range(len(fv_partials)):
        fv_partials[b] = Float32(0.0)
    var row_blocks = (n_rows + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    for b in range(row_blocks):
        var s_w = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        var s_g = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        for t in range(GBDT_MSE_BLOCK):
            var r = b * GBDT_MSE_BLOCK + t
            if r >= n_rows:
                continue
            stats[r] = pw[r]
            stats[n_rows + r] = der[r]
            s_w[t] = abs(pw[r])
            s_g[t] = abs(der[r])
        mag_partials[2 * b] = _halving_fold(s_w)
        mag_partials[2 * b + 1] = _halving_fold(s_g)


def yeti_rank_eval(
    targets: List[Float32],
    weights: List[Float32],
    has_weights: Bool,
    g_cursor: List[Float32],
    row_index: List[Int],
    group_sizes: List[Int],
    tasks: YetiRankTasks,
    seed: UInt64,
    permutations: Int,
    decay_speed: Float32,
    offsets_leaf: List[Int],
    sizes_leaf: List[Int],
    n_rows: Int,
    lambda_reg: Float64,
    mut value: Float64,
    mut gradient: List[Float64],
    mut cached_der2: List[Float64],
):
    """`BinOptimizedOracle.write_value_and_first_derivatives`' YetiRank arm:
    the bin-ordered point read back to row order through the inverse of
    `row_index`, the derivatives scattered to each row's bin position as
    `[der, pair weight]`, then the per-leaf partition stats, the Hessian plus
    lambda, and a value of 0.0 (`FillBuffer(FunctionValue, 0)`)."""
    var inverse = List[Int](length=n_rows, fill=0)
    for pos in range(n_rows):
        inverse[row_index[pos]] = pos
    var point = List[Float32](length=n_rows, fill=Float32(0.0))
    for i in range(n_rows):
        point[i] = g_cursor[inverse[i]]
    var der = List[Float32]()
    var pw = List[Float32]()
    yeti_rank_row_derivatives(
        targets, weights, has_weights, point, group_sizes, tasks, seed,
        permutations, decay_speed, n_rows, der, pw,
    )
    var stats = List[Float32](length=2 * n_rows, fill=Float32(0.0))
    for i in range(n_rows):
        var dst = inverse[i]
        stats[dst] = der[i]
        stats[n_rows + dst] = pw[i]
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
    value = Float64(0.0)
