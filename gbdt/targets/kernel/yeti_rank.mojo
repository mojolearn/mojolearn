# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Reference: `catboost/cuda/targets/kernel/yeti_rank_pointwise.cu`
(`RemoveQueryMeansImpl`, `:15-33`; `YetiRankGradientSingleGroup`, `:36-180`;
`YetiRankGradientImpl`, `:182-247`; `YetiRankGradient`, `:249-283`), the two
sort passes of `radix_sort_block.cuh` (`RadixSortSingleBlock4`, `:187-306`),
`TYetiRankKernel::Run` (`targets/kernel.h:420-476`) and the YetiRank arms of
`TQuerywiseTargetsImpl` (`querywise_targets_impl.h:131-158`, `:213-229`,
`:313-321`).

WHAT ONE CALL COMPUTES. The point is read in row order (gathered through the
inverse bin order during leaf estimation, whose derivatives are scattered back
to bin positions, `kernel.h:470-473`), centered by its unit-weight query means
(`ComputeGroupMeans(approx, nullptr, ...)` then `RemoveQueryMeans`), and then,
task by task (`gbdt/data/yeti_rank_tasks.mojo`), for `permutations` rounds:

    each position p of the task's 1024 (the rows, then padding) draws
        val = exp(min(approx, 70)) * uni / (1.000001 - uni)
    from its thread's stream (thread p mod 256, lanes drawn in order);
    the positions are ordered by query ascending, then by val descending,
    then by position (the two stable radix passes);
    each ordered position j after its query's first adds, for the documents
    at j - 1 and j with relevances r1, r2 (`relev * weight`) and exps a1, a2:
        pairWeight = 0.15 * decay^(j - queryBegin - 1) * |r1 - r2| / permutations
        ll = pairWeight * (r1 > r2 ? a2 : -a1) / (a2 + a1)
    der[doc1] += ll, weight[doc1] += pairWeight   (every j of lane k, phase 1)
    der[doc2] -= ll, weight[doc2] += pairWeight   (every j of lane k, phase 2)

`der2` is the accumulated pair weight and `functionValue` is ONE 0.0
(`kernel.h:421-423`): YetiRank has no value here; its score metric is PFound.
`GradientAt` calls `NewtonAt` for YetiRank (`querywise_targets_impl.h:
131-134`), so both search arms put the pair weights in plane 0.

================= DEVIATION BLOCK: one thread per task, sequential =================
The reference runs a task on a 256-thread block with 4 lanes per thread, shared
memory and `__syncthreads` between steps. Within one step every thread writes a
distinct document (a document is `doc1` at one ordered position and `doc2` at
the next), so each document's float sums have ONE order: (round t, lane k,
phase). This kernel runs the task on ONE thread in exactly that order: the
draws per (round, thread, lane), the sort, then for each lane phase 1 over the
256 threads and phase 2 over the 256 threads. The two radix passes are one
stable merge sort on (query ascending, key descending) from position order,
which is the order the passes produce (each pass is a stable partition per
bit, `radix_sort_block.cuh:198-240`). Scratch lives in fit-long device buffers
at `[task * 1024 + position]` and `[task * 256 + thread]`, so no threadgroup
memory is used.
====================================================================================

DEVIATION (flush at derivation, IDENTITY_PATHS row 10): the centered point, each
`relev * weight`, `pairWeight`, `ll` and every accumulator store pass through
`ftz`. DEVIATION 258: `exp` and `pow` go through `routed_exp` and
`identical_pow`. DEVIATION (the seed stream): see `doc_parallel_boosting.mojo`,
where each call's `NextUniformL` is drawn from a YetiRank stream of its own.

================= DEVIATION 3040: one BLOCK per task on the NVIDIA column =================
MEASURED FIRST (RTX 4090, Istella-S LETOR 2,043,304 x 220, 19,245 queries, 2,046
tasks, 2026-09-17): the
one-thread-per-task kernel above ran 29.07 ms a call, two calls a tree (the
search gradient and the leaf estimation), 58.1 of the fit's 58.2 ms per tree
and 88.6 percent of all GPU kernel time. One GPU thread walking 1024 positions
through ten rounds of draws, a ten-pass merge sort and 2048 pair steps in
device memory is latency bound; the device sat idle beside it.

`yeti_rank_task_block_kernel` runs a task on a 256-thread block with 4 lanes a
thread, the reference's own shape (`YetiRankGradientSingleGroup`), with the
sort keys, relevances, exps and the two accumulators in block shared memory
and a `barrier()` where the reference has `__syncthreads()`.

WHY NO BIT MOVES. (1) The draws: thread `tid` owns positions `tid + 256 * k`
and advances ONE stream through its lanes in order every round, the stream the
sequential kernel kept in `s_seed[tid]`; the float expressions are the same
text. (2) The sort computes NO float. Its result is the unique ascending order
of the composite key `(query << 42) | (~key << 10) | position`, which is
(query ascending, key descending, position ascending), the order `_b_first`
gives the stable merge; every composite is distinct (the position is in it),
so any correct sort yields this one permutation. Here it is a ten-pass merge
in which every element finds its output slot by a binary search of the
sibling run. (3) The pairs: `pairWeight` and `ll` are the same expressions on
the same operands (`decay` is hoisted out of the round loop; it is a pure
function of the position and its query begin). Within one (round, lane, phase)
step every thread writes a DISTINCT document, and the steps are separated by
barriers, so each document's float sums keep the one order the docstring
above names: (round t, lane k, phase). (4) The accumulators start at 0.0 in
shared memory and are stored once to `der_acc` / `weight_acc`, the same values
the sequential kernel accumulated in place.

THE ROW. `yeti_block_parallel_for[column]` is True on the NVIDIA column only
(the column this was measured and identity-checked on); Apple (whose 32 KiB
threadgroup limit this kernel fills exactly) and AMD keep the sequential
kernel until their columns run it. `-D MOJOLEARN_3040_YETI_SEQUENTIAL=1` is the
kill switch and the BEFORE arm; `-D MOJOLEARN_3040_YETI_BLOCK=1` opts another
column in. `-D MOJOLEARN_GBDT_YETI_SABOTAGE=1` runs phase 2 before phase 1 on
the block kernel, the order this DEVIATION must not change.
============================================================================================
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import bitcast, stack_allocation
from std.sys import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.kernel_matrix import COLUMN_NVIDIA, TARGET_COLUMN
from checks.numerics import ftz, identical_mul, identical_pow
from gbdt.data.yeti_rank_tasks import (
    YETI_TASK_POSITIONS,
    yeti_rank_cuda_seed,
    yeti_rank_tasks,
)
from gbdt.gpu_data.kernel.query_helper import (
    launch_compute_group_ids,
    launch_compute_group_means,
)
from gbdt.gpu_util.kernel.transform import launch_gather_with_mask_f32
from gbdt.targets.kernel.pointwise_targets import (
    MSE_BLOCK_SIZE,
    pinned_block_sum,
    routed_exp,
)
from gbdt.targets.kernel.query_rmse import (
    QuerywiseTargetBuffers,
    make_querywise_target_buffers,
)

comptime YETI_THREADS = 256
comptime YETI_LANES = 4


def yeti_block_parallel_for[column: Int]() -> Bool:
    """SCHEDULING row (DEVIATION 3040): whether a YetiRank task runs on a
    256-thread block (`yeti_rank_task_block_kernel`) or on one thread
    (`yeti_rank_task_kernel`). Same bits either way (the DEVIATION block in
    the module docstring); NVIDIA is the column it was measured and
    identity-checked on. The kill switch wins over the opt-in."""
    comptime if is_defined["MOJOLEARN_3040_YETI_SEQUENTIAL"]():
        return False
    comptime if is_defined["MOJOLEARN_3040_YETI_BLOCK"]():
        return True
    return column == COLUMN_NVIDIA


comptime YETI_BLOCK_PARALLEL = yeti_block_parallel_for[TARGET_COLUMN]()
#: negative control for DEVIATION 3040: phase 2 before phase 1 (default off)
comptime YETI_SABOTAGE = is_defined["MOJOLEARN_GBDT_YETI_SABOTAGE"]()


def _advance_seed32(seed: UInt32) -> UInt32:
    """`AdvanceSeed32` (`random_gen.cuh:40-43`), restated for the kernel."""
    return UInt32(1664525) * seed + UInt32(1013904223)


def _task_seed(task_qid: UInt32, tid: Int, cuda_seed: UInt32) -> UInt32:
    """`yeti_rank_pointwise.cu:224-237`, restated for the kernel."""
    var s = UInt32(127) * task_qid + UInt32(16807) * UInt32(tid) + UInt32(1)
    s = _advance_seed32(_advance_seed32(_advance_seed32(s)))
    s = s + cuda_seed
    s = _advance_seed32(_advance_seed32(_advance_seed32(s)))
    return s


def _b_first(key_a: UInt32, idx_a: UInt32, key_b: UInt32, idx_b: UInt32) -> Bool:
    """True when (key_b, idx_b) comes strictly before (key_a, idx_a): a lower
    query, or the same query and a larger key. Ties keep position order."""
    var qa = (idx_a >> UInt32(10)) & UInt32(1023)
    var qb = (idx_b >> UInt32(10)) & UInt32(1023)
    if qb != qa:
        return qb < qa
    return key_b > key_a


def yeti_rank_center_kernel(
    src: MutPointer[Float32, MutAnyOrigin],
    row_qids: MutPointer[UInt32, MutAnyOrigin],
    query_means: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    dst: MutPointer[Float32, MutAnyOrigin],
):
    """`RemoveQueryMeansImpl` (`yeti_rank_pointwise.cu:15-22`), flushed.
    `src` may be `dst`."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_rows_in):
        var m = query_means.unsafe_load(Int(row_qids.unsafe_load(i)))
        dst.unsafe_store(i, ftz(src.unsafe_load(i) - m))


def yeti_rank_task_kernel(
    approx: MutPointer[Float32, MutAnyOrigin],
    relev: MutPointer[Float32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    has_weights: Int32,
    row_qids: MutPointer[UInt32, MutAnyOrigin],
    task_offsets: MutPointer[UInt32, MutAnyOrigin],
    task_sizes: MutPointer[UInt32, MutAnyOrigin],
    task_qids: MutPointer[UInt32, MutAnyOrigin],
    n_tasks_in: Int32,
    cuda_seed: UInt32,
    decay_speed: Float32,
    permutations_in: Int32,
    s_seed: MutPointer[UInt32, MutAnyOrigin],
    s_exp: MutPointer[Float32, MutAnyOrigin],
    s_relev: MutPointer[Float32, MutAnyOrigin],
    s_src: MutPointer[UInt32, MutAnyOrigin],
    s_begin: MutPointer[UInt32, MutAnyOrigin],
    s_key: MutPointer[UInt32, MutAnyOrigin],
    s_idx: MutPointer[UInt32, MutAnyOrigin],
    s_tmp_key: MutPointer[UInt32, MutAnyOrigin],
    s_tmp_idx: MutPointer[UInt32, MutAnyOrigin],
    der_acc: MutPointer[Float32, MutAnyOrigin],
    weight_acc: MutPointer[Float32, MutAnyOrigin],
):
    """One task on one thread (the DEVIATION block). Grid `(n_tasks, 1, 1)`,
    block `(1, 1, 1)`."""
    var task = Int(block_idx.x)
    if task < Int(n_tasks_in):
        var base = task * YETI_TASK_POSITIONS
        var seed_base = task * YETI_THREADS
        var offset = Int(task_offsets.unsafe_load(task))
        var size = Int(task_sizes.unsafe_load(task))
        var task_qid = task_qids.unsafe_load(task)
        var first_qid = row_qids.unsafe_load(offset)
        var pad_qid = row_qids.unsafe_load(offset + size - 1) + UInt32(1) - first_qid
        var perms = Int(permutations_in)

        # `FillBuffer(targetDst / weightDst, 0)` for this task's rows
        for p in range(size):
            der_acc.unsafe_store(offset + p, Float32(0.0))
            weight_acc.unsafe_store(offset + p, Float32(0.0))

        # the thread streams (`:224-237`), kept across the rounds
        for tid in range(YETI_THREADS):
            s_seed.unsafe_store(seed_base + tid, _task_seed(task_qid, tid, cuda_seed))

        # local query ids, query begins, relevances and exps (`:50-104`)
        var prev_qid = UInt32(0xFFFFFFFF)
        var begin = 0
        for p in range(YETI_TASK_POSITIONS):
            var qid = pad_qid
            if p < size:
                qid = row_qids.unsafe_load(offset + p) - first_qid
            if p == 0 or qid != prev_qid:
                begin = p
            prev_qid = qid
            s_begin.unsafe_store(base + p, UInt32(begin))
            s_src.unsafe_store(base + p, UInt32(p) | (qid << UInt32(10)))
            if p < size:
                var w = Float32(1.0)
                if has_weights != Int32(0):
                    w = weights.unsafe_load(offset + p)
                s_relev.unsafe_store(base + p, ftz(relev.unsafe_load(offset + p) * w))
                s_exp.unsafe_store(
                    base + p,
                    routed_exp(min(approx.unsafe_load(offset + p), Float32(70.0))),
                )
            else:
                s_relev.unsafe_store(base + p, Float32(1000.0))
                s_exp.unsafe_store(base + p, Float32(1000.0))

        for t in range(perms):
            # the draws (`:112-121`): thread by thread, lane by lane
            for tid in range(YETI_THREADS):
                var s = s_seed.unsafe_load(seed_base + tid)
                for k in range(YETI_LANES):
                    var p = tid + YETI_THREADS * k
                    var val = Float32(-1000.0)
                    if p < size:
                        val = s_exp.unsafe_load(base + p)
                    s = _advance_seed32(s)
                    # `NextUniformFloat32`: `v * 2.328306435996595e-10f`, whose
                    # float is exactly 2^-32
                    var uni = identical_mul(Float32(s), Float32(2.328306435996595e-10))  # exact; pinned (lane/pinned-mul-contract-free)
                    val = val * (uni / (Float32(1.000001) - uni))
                    var bits = bitcast[DType.uint32](val)
                    if (bits & UInt32(0x80000000)) != UInt32(0):
                        bits = bits ^ UInt32(0xFFFFFFFF)
                    else:
                        bits = bits ^ UInt32(0x80000000)
                    s_key.unsafe_store(base + p, bits)
                    s_idx.unsafe_store(base + p, s_src.unsafe_load(base + p))
                s_seed.unsafe_store(seed_base + tid, s)

            # the two stable radix passes as one bottom-up stable merge sort
            # over the 1024 positions; ten passes, so the order ends in
            # s_key / s_idx
            var src_key = s_key
            var src_idx = s_idx
            var dst_key = s_tmp_key
            var dst_idx = s_tmp_idx
            var width = 1
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
                            take_b = _b_first(
                                src_key.unsafe_load(base + a),
                                src_idx.unsafe_load(base + a),
                                src_key.unsafe_load(base + b),
                                src_idx.unsafe_load(base + b),
                            )
                        var from_pos = a
                        if take_b:
                            from_pos = b
                            b += 1
                        else:
                            a += 1
                        dst_key.unsafe_store(base + o, src_key.unsafe_load(base + from_pos))
                        dst_idx.unsafe_store(base + o, src_idx.unsafe_load(base + from_pos))
                        o += 1
                    lo = hi
                var swap_key = src_key
                var swap_idx = src_idx
                src_key = dst_key
                src_idx = dst_idx
                dst_key = swap_key
                dst_idx = swap_idx
                width = width * 2

            # the pairs (`:130-168`): for each lane, phase 1 then phase 2
            for k in range(YETI_LANES):
                for phase in range(2):
                    for tid in range(YETI_THREADS):
                        var j = tid + YETI_THREADS * k
                        var qb = Int(s_begin.unsafe_load(base + j))
                        if j != qb:
                            var idx1 = Int(s_idx.unsafe_load(base + j - 1) & UInt32(1023))
                            var idx2 = Int(s_idx.unsafe_load(base + j) & UInt32(1023))
                            var relev1 = s_relev.unsafe_load(base + idx1)
                            var relev2 = s_relev.unsafe_load(base + idx2)
                            var approx1 = s_exp.unsafe_load(base + idx1)
                            var approx2 = s_exp.unsafe_load(base + idx2)
                            var decay = Float32(0.15) * identical_pow(
                                decay_speed, Float32(j - qb - 1)
                            )
                            var pair_weight = ftz(
                                decay * abs(relev1 - relev2) / Float32(perms)
                            )
                            var sel = -approx1
                            if relev1 > relev2:
                                sel = approx2
                            var ll = ftz(pair_weight * sel / (approx2 + approx1))
                            if phase == 0:
                                if idx1 < size:
                                    var r1 = offset + idx1
                                    weight_acc.unsafe_store(
                                        r1, ftz(weight_acc.unsafe_load(r1) + pair_weight)
                                    )
                                    der_acc.unsafe_store(r1, ftz(der_acc.unsafe_load(r1) + ll))
                            elif idx2 < size:
                                var r2 = offset + idx2
                                weight_acc.unsafe_store(
                                    r2, ftz(weight_acc.unsafe_load(r2) + pair_weight)
                                )
                                der_acc.unsafe_store(r2, ftz(der_acc.unsafe_load(r2) + (-ll)))


def yeti_rank_task_block_kernel(
    approx: MutPointer[Float32, MutAnyOrigin],
    relev: MutPointer[Float32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    has_weights: Int32,
    row_qids: MutPointer[UInt32, MutAnyOrigin],
    q_offsets: MutPointer[UInt32, MutAnyOrigin],
    task_offsets: MutPointer[UInt32, MutAnyOrigin],
    task_sizes: MutPointer[UInt32, MutAnyOrigin],
    task_qids: MutPointer[UInt32, MutAnyOrigin],
    cuda_seed: UInt32,
    decay_speed: Float32,
    permutations_in: Int32,
    der_acc: MutPointer[Float32, MutAnyOrigin],
    weight_acc: MutPointer[Float32, MutAnyOrigin],
):
    """One task on one 256-thread block, 4 lanes a thread (DEVIATION 3040).
    Grid `(n_tasks, 1, 1)`, block `(YETI_THREADS, 1, 1)`: every block is a
    task, and every loop bound below is the same on every thread of a block,
    so every barrier is reached by every thread.

    Shared memory, 32 KiB: the composite sort keys and their ping-pong copy
    (2 x 1024 x 8 bytes), `relev * weight`, the exps and the two accumulators
    (4 x 1024 x 4 bytes)."""
    var sh_keys = stack_allocation[
        2 * YETI_TASK_POSITIONS,
        Scalar[DType.uint64],
        address_space = AddressSpace.SHARED,
    ]()
    var sh_relev = stack_allocation[
        YETI_TASK_POSITIONS,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var sh_exp = stack_allocation[
        YETI_TASK_POSITIONS,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var sh_der = stack_allocation[
        YETI_TASK_POSITIONS,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var sh_weight = stack_allocation[
        YETI_TASK_POSITIONS,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    var task = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var offset = Int(task_offsets.unsafe_load(task))
    var size = Int(task_sizes.unsafe_load(task))
    var first_qid = row_qids.unsafe_load(offset)
    var pad_qid = row_qids.unsafe_load(offset + size - 1) + UInt32(1) - first_qid
    var perms = Int(permutations_in)

    # this thread's stream (`:224-237`), advanced through its lanes in order
    # every round: what the sequential kernel kept in `s_seed[tid]`
    var s = _task_seed(task_qids.unsafe_load(task), tid, cuda_seed)

    # per lane: the local query id, the query begin and the pair decay
    var lane_qid = SIMD[DType.uint32, YETI_LANES](0)
    var lane_begin = SIMD[DType.int32, YETI_LANES](0)
    var lane_decay = SIMD[DType.float32, YETI_LANES](0.0)
    for k in range(YETI_LANES):
        var p = tid + YETI_THREADS * k
        var qid = pad_qid
        # the padding is one query that begins at the first padded position
        var begin = size
        if p < size:
            var row_qid = row_qids.unsafe_load(offset + p)
            qid = row_qid - first_qid
            begin = Int(q_offsets.unsafe_load(Int(row_qid))) - offset
            var w = Float32(1.0)
            if has_weights != Int32(0):
                w = weights.unsafe_load(offset + p)
            sh_relev.unsafe_store(p, ftz(relev.unsafe_load(offset + p) * w))
            sh_exp.unsafe_store(
                p, routed_exp(min(approx.unsafe_load(offset + p), Float32(70.0)))
            )
        else:
            sh_relev.unsafe_store(p, Float32(1000.0))
            sh_exp.unsafe_store(p, Float32(1000.0))
        sh_der.unsafe_store(p, Float32(0.0))
        sh_weight.unsafe_store(p, Float32(0.0))
        lane_qid[k] = qid
        lane_begin[k] = Int32(begin)
        if p != begin:
            # the round loop's `0.15 * pow(decaySpeed, j - queryBegin - 1)`,
            # hoisted: it depends on the position and its query begin only
            lane_decay[k] = Float32(0.15) * identical_pow(
                decay_speed, Float32(p - begin - 1)
            )
    barrier()

    for _ in range(perms):
        # the draws (`:112-121`)
        for k in range(YETI_LANES):
            var p = tid + YETI_THREADS * k
            var val = Float32(-1000.0)
            if p < size:
                val = sh_exp.unsafe_load(p)
            s = _advance_seed32(s)
            # `NextUniformFloat32`: `v * 2.328306435996595e-10f`, whose
            # float is exactly 2^-32
            var uni = identical_mul(Float32(s), Float32(2.328306435996595e-10))  # exact; pinned (lane/pinned-mul-contract-free)
            val = val * (uni / (Float32(1.000001) - uni))
            var bits = bitcast[DType.uint32](val)
            if (bits & UInt32(0x80000000)) != UInt32(0):
                bits = bits ^ UInt32(0xFFFFFFFF)
            else:
                bits = bits ^ UInt32(0x80000000)
            # (query ascending, key descending, position ascending) as ONE
            # ascending integer; the position makes every composite distinct
            var composite = (
                (UInt64(lane_qid[k]) << UInt64(42))
                | (UInt64(bits ^ UInt32(0xFFFFFFFF)) << UInt64(10))
                | UInt64(p)
            )
            sh_keys.unsafe_store(p, composite)
        barrier()

        # the two stable radix passes as a ten-pass merge: every element
        # finds its output slot by a binary search of the sibling run (the
        # composites are distinct, so "strictly below" needs no tie rule).
        # Ten passes, so the order ends where it began, at `sh_keys[0:1024]`.
        var src = 0
        var dst = YETI_TASK_POSITIONS
        var width = 1
        while width < YETI_TASK_POSITIONS:
            for k in range(YETI_LANES):
                var i = tid + YETI_THREADS * k
                var lo = (i // (2 * width)) * (2 * width)
                var mid = lo + width
                var sibling = mid
                var own_rank = i - lo
                if i >= mid:
                    sibling = lo
                    own_rank = i - mid
                var key = sh_keys.unsafe_load(src + i)
                var base = 0
                var length = width
                while length > 1:
                    var half = length // 2
                    if sh_keys.unsafe_load(src + sibling + base + half - 1) < key:
                        base += half
                    length -= half
                var below = base
                if sh_keys.unsafe_load(src + sibling + base) < key:
                    below += 1
                sh_keys.unsafe_store(dst + lo + own_rank + below, key)
            barrier()
            var swap = src
            src = dst
            dst = swap
            width = width * 2

        # the pairs (`:130-168`): for each lane, phase 1 then phase 2
        for k in range(YETI_LANES):
            var j = tid + YETI_THREADS * k
            var qb = Int(lane_begin[k])
            var has_pair = j != qb
            var idx1 = 0
            var idx2 = 0
            var pair_weight = Float32(0.0)
            var ll = Float32(0.0)
            if has_pair:
                idx1 = Int(sh_keys.unsafe_load(src + j - 1) & UInt64(1023))
                idx2 = Int(sh_keys.unsafe_load(src + j) & UInt64(1023))
                var relev1 = sh_relev.unsafe_load(idx1)
                var relev2 = sh_relev.unsafe_load(idx2)
                var approx1 = sh_exp.unsafe_load(idx1)
                var approx2 = sh_exp.unsafe_load(idx2)
                var decay = lane_decay[k]
                pair_weight = ftz(decay * abs(relev1 - relev2) / Float32(perms))
                var sel = -approx1
                if relev1 > relev2:
                    sel = approx2
                ll = ftz(pair_weight * sel / (approx2 + approx1))
            comptime for step in range(2):
                comptime phase = (1 - step) if YETI_SABOTAGE else step
                comptime if phase == 0:
                    if has_pair and idx1 < size:
                        sh_weight.unsafe_store(
                            idx1, ftz(sh_weight.unsafe_load(idx1) + pair_weight)
                        )
                        sh_der.unsafe_store(idx1, ftz(sh_der.unsafe_load(idx1) + ll))
                else:
                    if has_pair and idx2 < size:
                        sh_weight.unsafe_store(
                            idx2, ftz(sh_weight.unsafe_load(idx2) + pair_weight)
                        )
                        sh_der.unsafe_store(idx2, ftz(sh_der.unsafe_load(idx2) + (-ll)))
                barrier()

    for k in range(YETI_LANES):
        var p = tid + YETI_THREADS * k
        if p < size:
            der_acc.unsafe_store(offset + p, sh_der.unsafe_load(p))
            weight_acc.unsafe_store(offset + p, sh_weight.unsafe_load(p))


def yeti_rank_row_kernel[estimation: Bool](
    der_acc: MutPointer[Float32, MutAnyOrigin],
    weight_acc: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    write_map: MutPointer[UInt32, MutAnyOrigin],
    has_write_map: Int32,
    stats: MutPointer[Float32, MutAnyOrigin],
    function_value: MutPointer[Float32, MutAnyOrigin],
    compute_fv: Int32,
    plane_magnitudes: MutPointer[Float32, MutAnyOrigin],
    compute_magnitudes: Int32,
):
    """The planes: SEARCH `[pair weight, der]` at the row, ESTIMATION
    `[der, pair weight]` at `write_map[row]` (`Scatter`, `kernel.h:470-473`);
    every value partial is 0.0 (`FillBuffer(FunctionValue, 0)`)."""
    var n_rows = Int(n_rows_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var in_range = i < n_rows
    var der = Float32(0.0)
    var weight = Float32(0.0)
    if in_range:
        der = der_acc.unsafe_load(i)
        weight = weight_acc.unsafe_load(i)
        comptime if estimation:
            var dst = i
            if has_write_map != Int32(0):
                dst = Int(write_map.unsafe_load(i))
            stats.unsafe_store(dst, der)
            stats.unsafe_store(n_rows + dst, weight)
        else:
            stats.unsafe_store(i, weight)
            stats.unsafe_store(n_rows + i, der)
    if compute_fv != Int32(0) and thread_idx.x == 0:
        function_value.unsafe_store(Int(block_idx.x), Float32(0.0))
    if compute_magnitudes != Int32(0):
        var w_abs = Float32(0.0)
        var g_abs = Float32(0.0)
        if in_range:
            w_abs = abs(weight)
            g_abs = abs(der)
        var w_total = pinned_block_sum[block_size=MSE_BLOCK_SIZE](w_abs)
        var g_total = pinned_block_sum[block_size=MSE_BLOCK_SIZE](g_abs)
        if thread_idx.x == 0:
            plane_magnitudes.unsafe_store(2 * Int(block_idx.x), w_total)
            plane_magnitudes.unsafe_store(2 * Int(block_idx.x) + 1, g_total)


struct YetiRankTargetBuffers(Movable):
    """The querywise buffers (targets, weights, query offsets and sizes, query
    means, row query ids, inverse order, `no_indices`), the task table, the
    scratch and the accumulators. Built once per fit by
    `make_yeti_rank_target_buffers`; `handles()` gives views onto the same
    memory."""

    var query: QuerywiseTargetBuffers
    var n_tasks: Int
    var permutations: Int
    var decay: Float32
    var d_task_offsets: DeviceBuffer[DType.uint32]
    var d_task_sizes: DeviceBuffer[DType.uint32]
    var d_task_qids: DeviceBuffer[DType.uint32]
    var s_seed: DeviceBuffer[DType.uint32]
    var s_exp: DeviceBuffer[DType.float32]
    var s_relev: DeviceBuffer[DType.float32]
    var s_src: DeviceBuffer[DType.uint32]
    var s_begin: DeviceBuffer[DType.uint32]
    var s_key: DeviceBuffer[DType.uint32]
    var s_idx: DeviceBuffer[DType.uint32]
    var s_tmp_key: DeviceBuffer[DType.uint32]
    var s_tmp_idx: DeviceBuffer[DType.uint32]
    var d_der_acc: DeviceBuffer[DType.float32]
    var d_weight_acc: DeviceBuffer[DType.float32]
    var d_point: DeviceBuffer[DType.float32]

    def __init__(
        out self,
        var query: QuerywiseTargetBuffers,
        n_tasks: Int,
        permutations: Int,
        decay: Float32,
        var d_task_offsets: DeviceBuffer[DType.uint32],
        var d_task_sizes: DeviceBuffer[DType.uint32],
        var d_task_qids: DeviceBuffer[DType.uint32],
        var s_seed: DeviceBuffer[DType.uint32],
        var s_exp: DeviceBuffer[DType.float32],
        var s_relev: DeviceBuffer[DType.float32],
        var s_src: DeviceBuffer[DType.uint32],
        var s_begin: DeviceBuffer[DType.uint32],
        var s_key: DeviceBuffer[DType.uint32],
        var s_idx: DeviceBuffer[DType.uint32],
        var s_tmp_key: DeviceBuffer[DType.uint32],
        var s_tmp_idx: DeviceBuffer[DType.uint32],
        var d_der_acc: DeviceBuffer[DType.float32],
        var d_weight_acc: DeviceBuffer[DType.float32],
        var d_point: DeviceBuffer[DType.float32],
    ):
        self.query = query^
        self.n_tasks = n_tasks
        self.permutations = permutations
        self.decay = decay
        self.d_task_offsets = d_task_offsets^
        self.d_task_sizes = d_task_sizes^
        self.d_task_qids = d_task_qids^
        self.s_seed = s_seed^
        self.s_exp = s_exp^
        self.s_relev = s_relev^
        self.s_src = s_src^
        self.s_begin = s_begin^
        self.s_key = s_key^
        self.s_idx = s_idx^
        self.s_tmp_key = s_tmp_key^
        self.s_tmp_idx = s_tmp_idx^
        self.d_der_acc = d_der_acc^
        self.d_weight_acc = d_weight_acc^
        self.d_point = d_point^

    def handles(self) -> YetiRankTargetBuffers:
        """Handle copies onto the same device memory."""
        return YetiRankTargetBuffers(
            self.query.handles(), self.n_tasks, self.permutations, self.decay,
            self.d_task_offsets.copy(), self.d_task_sizes.copy(),
            self.d_task_qids.copy(), self.s_seed.copy(), self.s_exp.copy(),
            self.s_relev.copy(), self.s_src.copy(), self.s_begin.copy(),
            self.s_key.copy(), self.s_idx.copy(), self.s_tmp_key.copy(),
            self.s_tmp_idx.copy(), self.d_der_acc.copy(),
            self.d_weight_acc.copy(), self.d_point.copy(),
        )


def make_yeti_rank_target_buffers(
    ctx: DeviceContext,
    group_sizes: List[UInt32],
    n_rows: Int,
    mut targets: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
    has_weights: Bool,
    permutations: Int,
    decay: Float32,
) raises -> YetiRankTargetBuffers:
    """The querywise buffers, the task table (with `InitYetiRank`'s refusal)
    and the scratch."""
    var tasks = yeti_rank_tasks(group_sizes, n_rows)
    var n_tasks = tasks.count()
    var query = make_querywise_target_buffers(
        ctx, group_sizes, n_rows, targets, weights, has_weights
    )
    var h_off = ctx.enqueue_create_host_buffer[DType.uint32](n_tasks)
    var h_sz = ctx.enqueue_create_host_buffer[DType.uint32](n_tasks)
    var h_q = ctx.enqueue_create_host_buffer[DType.uint32](n_tasks)
    for t in range(n_tasks):
        h_off.unsafe_ptr().unsafe_store(t, tasks.offsets[t])
        h_sz.unsafe_ptr().unsafe_store(t, tasks.sizes[t])
        h_q.unsafe_ptr().unsafe_store(t, tasks.qids[t])
    var d_off = ctx.enqueue_create_buffer[DType.uint32](n_tasks)
    var d_sz = ctx.enqueue_create_buffer[DType.uint32](n_tasks)
    var d_q = ctx.enqueue_create_buffer[DType.uint32](n_tasks)
    ctx.enqueue_copy(dst_buf=d_off, src_ptr=h_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_sz, src_ptr=h_sz.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_q, src_ptr=h_q.unsafe_ptr())
    var positions = n_tasks * YETI_TASK_POSITIONS
    var s_seed = ctx.enqueue_create_buffer[DType.uint32](n_tasks * YETI_THREADS)
    var s_exp = ctx.enqueue_create_buffer[DType.float32](positions)
    var s_relev = ctx.enqueue_create_buffer[DType.float32](positions)
    var s_src = ctx.enqueue_create_buffer[DType.uint32](positions)
    var s_begin = ctx.enqueue_create_buffer[DType.uint32](positions)
    var s_key = ctx.enqueue_create_buffer[DType.uint32](positions)
    var s_idx = ctx.enqueue_create_buffer[DType.uint32](positions)
    var s_tmp_key = ctx.enqueue_create_buffer[DType.uint32](positions)
    var s_tmp_idx = ctx.enqueue_create_buffer[DType.uint32](positions)
    var d_der_acc = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var d_weight_acc = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var d_point = ctx.enqueue_create_buffer[DType.float32](n_rows)
    ctx.synchronize()
    _ = h_off^  # past the drain (step-33 race class)
    _ = h_sz^
    _ = h_q^
    return YetiRankTargetBuffers(
        query^, n_tasks, permutations, decay, d_off^, d_sz^, d_q^, s_seed^,
        s_exp^, s_relev^, s_src^, s_begin^, s_key^, s_idx^, s_tmp_key^,
        s_tmp_idx^, d_der_acc^, d_weight_acc^, d_point^,
    )


def _launch_yeti_rank_tasks_sequential(
    ctx: DeviceContext, mut y: YetiRankTargetBuffers, seed: UInt64
) raises:
    """The one-thread-per-task launch (every column but NVIDIA, and the
    DEVIATION 3040 kill switch)."""
    ctx.enqueue_function[yeti_rank_task_kernel](
        y.d_point.unsafe_ptr(), y.query.targets.unsafe_ptr(),
        y.query.weights.unsafe_ptr(),
        Int32(1) if y.query.has_weights else Int32(0),
        y.query.qids.unsafe_ptr(),
        y.d_task_offsets.unsafe_ptr(), y.d_task_sizes.unsafe_ptr(),
        y.d_task_qids.unsafe_ptr(), Int32(y.n_tasks),
        yeti_rank_cuda_seed(seed), y.decay, Int32(y.permutations),
        y.s_seed.unsafe_ptr(), y.s_exp.unsafe_ptr(), y.s_relev.unsafe_ptr(),
        y.s_src.unsafe_ptr(), y.s_begin.unsafe_ptr(), y.s_key.unsafe_ptr(),
        y.s_idx.unsafe_ptr(), y.s_tmp_key.unsafe_ptr(), y.s_tmp_idx.unsafe_ptr(),
        y.d_der_acc.unsafe_ptr(), y.d_weight_acc.unsafe_ptr(),
        grid_dim=(y.n_tasks, 1, 1),
        block_dim=(1, 1, 1),
    )


def launch_yeti_rank_with[estimation: Bool](
    ctx: DeviceContext,
    mut y: YetiRankTargetBuffers,
    mut predictions: DeviceBuffer[DType.float32],
    use_inverse: Bool,
    seed: UInt64,
    mut stats: DeviceBuffer[DType.float32],
    mut function_value: DeviceBuffer[DType.float32],
    compute_fv: Bool,
    mut plane_magnitudes: DeviceBuffer[DType.float32],
    compute_magnitudes: Bool,
) raises:
    """`TYetiRankKernel::Run` (`kernel.h:420-476`): the point in row order
    (through `y.query.inverse` when `use_inverse`), its unit-weight query means
    removed, the tasks, then the planes over 256-row blocks. `seed` is the
    call's `NextUniformL`."""
    var n_rows = y.query.n_rows
    var row_blocks = (n_rows + MSE_BLOCK_SIZE - 1) // MSE_BLOCK_SIZE
    # `ComputeGroupIds` (`kernel.h:444`), every call as the reference runs it:
    # `y.query.qids` is scratch that only this launch fills for YetiRank (the
    # QueryRMSE launcher fills its own), and the task kernel reads each row's
    # query from it. Without it every row of a task read as one query.
    launch_compute_group_ids(
        ctx, y.query.q_sizes, y.query.q_offsets, UInt32(0), y.query.q_count,
        y.query.qids,
    )
    if use_inverse:
        # the gathered row-order point is staged in `d_der_acc`, which the
        # task kernel zero-fills for every task's rows before it writes
        # (the tasks tile the rows), so it is free until then
        launch_gather_with_mask_f32(
            ctx, y.d_der_acc, predictions, y.query.inverse, n_rows, UInt32(0xFFFFFFFF)
        )
        launch_compute_group_means(
            ctx, y.d_der_acc, y.query.weights, False, y.query.q_offsets, UInt32(0),
            y.query.q_sizes, y.query.q_count, y.query.query_means,
        )
        ctx.enqueue_function[yeti_rank_center_kernel](
            y.d_der_acc.unsafe_ptr(), y.query.qids.unsafe_ptr(),
            y.query.query_means.unsafe_ptr(), Int32(n_rows), y.d_point.unsafe_ptr(),
            grid_dim=(row_blocks, 1, 1),
            block_dim=(MSE_BLOCK_SIZE, 1, 1),
        )
    else:
        launch_compute_group_means(
            ctx, predictions, y.query.weights, False, y.query.q_offsets, UInt32(0),
            y.query.q_sizes, y.query.q_count, y.query.query_means,
        )
        ctx.enqueue_function[yeti_rank_center_kernel](
            predictions.unsafe_ptr(), y.query.qids.unsafe_ptr(),
            y.query.query_means.unsafe_ptr(), Int32(n_rows), y.d_point.unsafe_ptr(),
            grid_dim=(row_blocks, 1, 1),
            block_dim=(MSE_BLOCK_SIZE, 1, 1),
        )
    comptime if YETI_BLOCK_PARALLEL:
        # DEVIATION 3040: one 256-thread block per task
        ctx.enqueue_function[yeti_rank_task_block_kernel](
            y.d_point.unsafe_ptr(), y.query.targets.unsafe_ptr(),
            y.query.weights.unsafe_ptr(),
            Int32(1) if y.query.has_weights else Int32(0),
            y.query.qids.unsafe_ptr(), y.query.q_offsets.unsafe_ptr(),
            y.d_task_offsets.unsafe_ptr(), y.d_task_sizes.unsafe_ptr(),
            y.d_task_qids.unsafe_ptr(),
            yeti_rank_cuda_seed(seed), y.decay, Int32(y.permutations),
            y.d_der_acc.unsafe_ptr(), y.d_weight_acc.unsafe_ptr(),
            grid_dim=(y.n_tasks, 1, 1),
            block_dim=(YETI_THREADS, 1, 1),
        )
    else:
        _launch_yeti_rank_tasks_sequential(ctx, y, seed)
    if use_inverse:
        ctx.enqueue_function[yeti_rank_row_kernel[estimation]](
            y.d_der_acc.unsafe_ptr(), y.d_weight_acc.unsafe_ptr(), Int32(n_rows),
            y.query.inverse.unsafe_ptr(), Int32(1),
            stats.unsafe_ptr(), function_value.unsafe_ptr(),
            Int32(1) if compute_fv else Int32(0),
            plane_magnitudes.unsafe_ptr(),
            Int32(1) if compute_magnitudes else Int32(0),
            grid_dim=(row_blocks, 1, 1),
            block_dim=(MSE_BLOCK_SIZE, 1, 1),
        )
    else:
        ctx.enqueue_function[yeti_rank_row_kernel[estimation]](
            y.d_der_acc.unsafe_ptr(), y.d_weight_acc.unsafe_ptr(), Int32(n_rows),
            y.query.no_indices.unsafe_ptr(), Int32(0),
            stats.unsafe_ptr(), function_value.unsafe_ptr(),
            Int32(1) if compute_fv else Int32(0),
            plane_magnitudes.unsafe_ptr(),
            Int32(1) if compute_magnitudes else Int32(0),
            grid_dim=(row_blocks, 1, 1),
            block_dim=(MSE_BLOCK_SIZE, 1, 1),
        )


def yeti_rank_zero_value_kernel(
    function_value: MutPointer[Float32, MutAnyOrigin],
    n_blocks_in: Int32,
):
    """One 0.0 value partial per block (`FillBuffer(FunctionValue, 0)`)."""
    var b = Int(block_idx.x)
    if b < Int(n_blocks_in) and thread_idx.x == 0:
        function_value.unsafe_store(b, Float32(0.0))


def launch_yeti_rank_zero_value(
    ctx: DeviceContext,
    mut function_value: DeviceBuffer[DType.float32],
    n_blocks: Int,
) raises:
    """The final learn-loss pass for YetiRank: every partial 0.0, no draw."""
    if n_blocks <= 0:
        return
    ctx.enqueue_function[yeti_rank_zero_value_kernel](
        function_value.unsafe_ptr(), Int32(n_blocks),
        grid_dim=(n_blocks, 1, 1),
        block_dim=(MSE_BLOCK_SIZE, 1, 1),
    )
