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
from gbdt.apple_fast_tree_experiments import AFT_N09, AFT_N10, AFT_N11
from std.gpu.primitives.warp import shuffle_xor
from std.memory import bitcast, stack_allocation
from std.sys import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.kernel_matrix import (
    COLUMN_AMD,
    COLUMN_APPLE,
    COLUMN_NVIDIA,
    TARGET_COLUMN,
)
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_FAST,
    NUMERIC_IDENTICAL,
    ftz,
    identical_mul,
    identical_pow,
)
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
    launch_aft_group_means,
    make_querywise_target_buffers,
)

comptime YETI_THREADS = 256
comptime YETI_LANES = 4
# N09 reuses a threadgroup's allocated task scratch across two complete
# tasks. Task IDs, per-task seed streams and query boundaries never change.
# N10 doubles merge outputs per active lane, halving co-rank searches while
# retaining the 256-thread RNG/pair ownership. N11 runs two rows per lane
# in each ORIGINAL 256-row partial tile; partial array sizes stay unchanged.
# All three are source-only, no performance/quality evidence, default OFF.
comptime AFT_YETI_TASKS_PER_BLOCK = 2 if AFT_N09 else 1
comptime AFT_YETI_MERGE_LANES = 8 if AFT_N10 else YETI_LANES
comptime AFT_YETI_ROW_THREADS = 128 if AFT_N11 else MSE_BLOCK_SIZE


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
    # FAST on Apple (lane apple-fast-trees2): the block kernel too. The
    # one-thread-per-task launch ran each task's permutations, its 1024-key
    # sort and its pairs on ONE GPU thread (block_dim 1); the block kernel is
    # the same bits (above), its 32 KiB of shared memory is Apple's
    # threadgroup limit, and its barriers order threadgroup memory only.
    # `-D MOJOLEARN_3040_YETI_SEQUENTIAL` is the A/B arm.
    comptime if column == COLUMN_APPLE and GLOBAL_NUMERIC_MODE == NUMERIC_FAST:
        return True
    # lane/fam-gbdt (2026-10-04), IDN_YETI_BLOCK_AMD: IDENTICAL on the AMD
    # column takes the block kernel too, default on. The AMD column ran each
    # task (its permutations, its 1024-key sort, its pairs) on ONE GPU
    # thread; the block kernel is the same bits (DEVIATION 3040: integer
    # sort of distinct composites, each document's float sums in the one
    # order the sequential kernel uses), needs 32 KiB of block shared memory
    # (the MI300/MI325 give 64 KiB) and no warp primitive outside the
    # Apple-only `YETI_FAST_SORT`. `-D MOJOLEARN_IDN_GBDT_YETI_BLOCK_AMD_OFF`
    # (or the master `-D MOJOLEARN_IDN_ALL_OFF`, or the kill switch above)
    # restores the sequential kernel on AMD.
    comptime if (
        column == COLUMN_AMD
        and GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
        and not (
            is_defined["MOJOLEARN_IDN_GBDT_YETI_BLOCK_AMD_OFF"]()
            or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
        )
    ):
        return True
    return column == COLUMN_NVIDIA


comptime YETI_BLOCK_PARALLEL = yeti_block_parallel_for[TARGET_COLUMN]()
#: negative control for DEVIATION 3040: phase 2 before phase 1 (default off)
comptime YETI_SABOTAGE = is_defined["MOJOLEARN_GBDT_YETI_SABOTAGE"]()


def yeti_est_reuse_search_for[column: Int]() -> Bool:
    """FAST Apple (lane apple-fast-yetirank): the leaf estimation's
    evaluation at the tree's starting point REUSES the search gradient's
    per-row derivatives and pair weights instead of relaunching the whole
    task kernel (draws, 1024-key sorts, pairs) on a fresh seed.

    WHY IT IS THE SAME POINT. The search call (`doc_parallel_boosting.mojo`,
    `launch_yeti_rank_with[False]` on `cursors[learn_p]`) and the
    estimation's first evaluation (`pointwise_oracle.mojo`, the same cursor
    gathered to bin order and read back through `query.inverse`) see the
    same row-order point; nothing moves the cursor between them, and no
    other launch writes `d_der_acc` / `d_weight_acc`. What changes is the
    SAMPLE: CatBoost draws an independent permutation set for estimation
    (its own `NextUniformL`); this arm estimates the leaves on the sample
    the tree was grown on. That moves FAST bits, never IDENTICAL ones
    (comptime FAST + Apple only), and the A/B gates quality (ndcg/map).
    One kernel launch (the scatter) replaces six, and the task kernel, the
    fit's dominant cost, runs once per tree instead of twice.

    `-D MOJOLEARN_YETI_EST_REUSE_SEARCH_OFF` is the A arm."""
    comptime if is_defined["MOJOLEARN_YETI_EST_REUSE_SEARCH_OFF"]():
        return False
    comptime if column == COLUMN_APPLE and GLOBAL_NUMERIC_MODE == NUMERIC_FAST:
        return True
    return False


comptime YETI_EST_REUSE_SEARCH = yeti_est_reuse_search_for[TARGET_COLUMN]()


def yeti_task_fused_for[column: Int]() -> Bool:
    """Lane af-sym-multi (2026-10-03), `-D MOJOLEARN_YR_TASK_FUSED` (or
    `-D MOJOLEARN_SYM_MULTI_ALL`), FAST + Apple, on top of the block kernel:
    ONE launch per YetiRank call. `launch_yeti_rank_with` runs five:
    `compute_group_ids` (the same row -> query map every call), the gather
    (estimation), `compute_group_means`, the centering, the task kernel and
    the row scatter (planes, zero value partials, magnitudes).
    `yeti_rank_task_fused_kernel` reads the point itself (through the index
    map in estimation), takes each query's unit-weight mean inside the task
    block (a segmented sum over the task's positions in the sort-key
    scratch; every task holds whole queries), centers, runs the block
    kernel's draws, sort and pairs unchanged, and stores the planes, the
    accumulators (for `YETI_EST_REUSE_SEARCH`), the zero value partials
    and per-TASK magnitude partials itself. The group ids are computed once
    per fit (`make_yeti_rank_target_buffers`). The query mean's fold order
    and the magnitude partition change (FAST bits move, fixed and
    repeatable); the pair arithmetic is the block kernel's. A fit with more
    tasks than 256-row blocks (never on the board data) keeps the five
    launches."""
    comptime if (
        is_defined["MOJOLEARN_YR_TASK_FUSED"]()
        or is_defined["MOJOLEARN_SYM_MULTI_ALL"]()
    ):
        comptime if column == COLUMN_APPLE and GLOBAL_NUMERIC_MODE == NUMERIC_FAST:
            return True
    return False


#: recovery 2026-10-04 (lane/apple-fast-rec-sym): source
#: lane/apple-fast-sym-multi@d2c832da0; the laptop built the .so (Metal side
#: unchecked), never timed. `-D MOJOLEARN_YR_TASK_FUSED` (or SYM_MULTI_ALL).
#: Port: the fused kernel now sorts with main's YETI_FAST_SORT arm (the
#: block kernel's sort, both arms).
comptime YETI_TASK_FUSED = yeti_task_fused_for[TARGET_COLUMN]()


def yeti_fast_sort_for[column: Int]() -> Bool:
    """FAST Apple (lane apple-fast-yetirank, default): the block kernel's
    per-round 1024-key sort as (1) a bitonic sort of 128 keys per simdgroup
    in registers (32 lanes x 4 contiguous keys, `shuffle_xor` across lanes,
    no shared memory and no barrier), then (2) three merge-path passes
    (widths 128, 256, 512) in which each thread finds the co-rank of its 4
    contiguous outputs with ONE binary search and merges them in order.
    The ten-pass rank merge it replaces did a binary search per element per
    pass (45 dependent shared loads per element) and ten barriers a round.

    SAME BITS. The composites are distinct (the position is in them), so
    every correct ascending sort yields the one permutation the merge gave;
    no float is computed. Apple simdgroups are 32 lanes and the block is
    256 threads = 8 simdgroups x 128 keys = YETI_TASK_POSITIONS.

    Default on FAST Apple since the M3 A/B aft-ab-ysort1 (gbdt-rank-yetirank
    istellarank, n=2, same hash): 5,303 -> 3,350 ms (-37%).
    `-D MOJOLEARN_YETI_FAST_SORT_OFF` is the A arm; the old
    `-D MOJOLEARN_YETI_FAST_SORT` is harmless."""
    comptime if is_defined["MOJOLEARN_YETI_FAST_SORT_OFF"]():
        return False
    comptime if column == COLUMN_APPLE and GLOBAL_NUMERIC_MODE == NUMERIC_FAST:
        return True
    return False


comptime YETI_FAST_SORT = yeti_fast_sort_for[TARGET_COLUMN]()
comptime YETI_SIMD_W = 32
comptime YETI_SIMD_RUN = YETI_SIMD_W * YETI_LANES


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


def yeti_task_16k_for[column: Int]() -> Bool:
    """SCHEDULING row (lane/apple-fast-trees-yeti, 2026-10-02): whether the
    block kernel is `yeti_rank_task_block16k_kernel` (16 KiB of threadgroup
    memory, accumulators in registers) instead of
    `yeti_rank_task_block_kernel` (32 KiB),
    FAST on Apple only. Same bits (the kernel's docstring).

    CAUSE. `yeti_rank_task_block_kernel` claims exactly Apple's 32 KiB
    threadgroup limit (`yeti_rank.mojo`, the DEVIATION 3040 block), so one
    task runs per GPU core: 8 simdgroups of the 24+ a core holds, with 19
    barriers a round and the pair phases' read-modify-writes scattered into
    threadgroup memory. The task kernel is the search gradient's whole cost
    (`tree_search` 6.0 s of the 9.56 s Istella fit). At 16 KiB two tasks
    share a core.

    Default on FAST Apple since the M3 A/B yeti-task16k (istellarank, n=2,
    same hash): 5,315 -> 5,102 ms (-4%); yeti-both 5,296 -> 5,110.
    `-D MOJOLEARN_YETI_SEARCH_TASK16K_OFF` is the A arm; the old
    `-D MOJOLEARN_YETI_SEARCH_TASK16K` is harmless.

    DEPENDENCY. This kernel keeps the rank-merge sort; `YETI_FAST_SORT`
    (default, M3 aft-ab-ysort1 -37%) lives only in the 32 KiB kernel. The
    16 KiB kernel therefore steps aside while `YETI_FAST_SORT` is on, and
    takes over when `-D MOJOLEARN_YETI_FAST_SORT_OFF` turns it off."""
    comptime if is_defined["MOJOLEARN_YETI_SEARCH_TASK16K_OFF"]():
        return False
    comptime if yeti_fast_sort_for[column]():
        return False
    comptime if column == COLUMN_APPLE and GLOBAL_NUMERIC_MODE == NUMERIC_FAST:
        return True
    return False


comptime YETI_TASK_16K = yeti_task_16k_for[TARGET_COLUMN]()


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
    n_tasks_in: Int32 = Int32(0),
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

    var task_count = Int(grid_dim.x)
    comptime if AFT_N09:
        if n_tasks_in > Int32(0):
            task_count = Int(n_tasks_in)
    comptime for task_lane in range(AFT_YETI_TASKS_PER_BLOCK):
        var task = Int(block_idx.x) + task_lane * Int(grid_dim.x)
        if task < task_count:
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
        
                var src = 0
                comptime if YETI_FAST_SORT:
                    comptime assert YETI_THREADS * YETI_LANES == YETI_TASK_POSITIONS
                    comptime assert YETI_THREADS % YETI_SIMD_W == 0
                    # (1) a bitonic sort of each simdgroup's 128 contiguous keys in
                    # registers: element e = lane * 4 + r of the simdgroup's run
                    var ln = tid % YETI_SIMD_W
                    var run_base = (tid // YETI_SIMD_W) * YETI_SIMD_RUN + ln * YETI_LANES
                    var v = SIMD[DType.uint64, YETI_LANES](0)
                    comptime for r in range(YETI_LANES):
                        v[r] = sh_keys.unsafe_load(run_base + r)
                    comptime for kk in range(1, 8):
                        comptime k = 1 << kk
                        comptime for t in range(kk):
                            comptime jj = kk - 1 - t
                            comptime j = 1 << jj
                            comptime if j < YETI_LANES:
                                comptime for r in range(YETI_LANES):
                                    comptime if (r & j) == 0:
                                        var a = v[r]
                                        var b = v[r | j]
                                        var asc = ((ln * YETI_LANES + r) & k) == 0
                                        if (a > b) == asc:
                                            v[r] = b
                                            v[r | j] = a
                            else:
                                comptime m = j // YETI_LANES
                                var lower = (ln & m) == 0
                                comptime for r in range(YETI_LANES):
                                    var x = v[r]
                                    var o_hi = shuffle_xor(UInt32(x >> UInt64(32)), UInt32(m))
                                    var o_lo = shuffle_xor(
                                        UInt32(x & UInt64(0xFFFFFFFF)), UInt32(m)
                                    )
                                    var o = (UInt64(o_hi) << UInt64(32)) | UInt64(o_lo)
                                    var asc = ((ln * YETI_LANES + r) & k) == 0
                                    if lower == asc:
                                        v[r] = min(x, o)
                                    else:
                                        v[r] = max(x, o)
                    comptime for r in range(YETI_LANES):
                        sh_keys.unsafe_store(run_base + r, v[r])
                    barrier()
                    # (2) merge-path passes: thread tid writes outputs 4 * tid .. +3
                    # of the merged pair of runs; one co-rank binary search, then a
                    # sequential merge with the run heads in registers. UInt64.MAX
                    # is a sentinel no composite reaches (they are below 2^52).
                    var dst = YETI_TASK_POSITIONS
                    var width = YETI_SIMD_RUN
                    var o0 = tid * AFT_YETI_MERGE_LANES
                    while width < YETI_TASK_POSITIONS:
                        if o0 < YETI_TASK_POSITIONS:
                            var lo = (o0 // (2 * width)) * (2 * width)
                            var d = o0 - lo
                            var a_base = src + lo
                            var b_base = a_base + width
                            var i_lo = max(0, d - width)
                            var i_hi = min(d, width)
                            while i_lo < i_hi:
                                var mid = (i_lo + i_hi) // 2
                                if sh_keys.unsafe_load(a_base + mid) < sh_keys.unsafe_load(
                                    b_base + d - mid - 1
                                ):
                                    i_lo = mid + 1
                                else:
                                    i_hi = mid
                            var ia = i_lo
                            var ib = d - i_lo
                            var va = UInt64.MAX
                            if ia < width:
                                va = sh_keys.unsafe_load(a_base + ia)
                            var vb = UInt64.MAX
                            if ib < width:
                                vb = sh_keys.unsafe_load(b_base + ib)
                            comptime for t in range(AFT_YETI_MERGE_LANES):
                                if va < vb:
                                    sh_keys.unsafe_store(dst + o0 + t, va)
                                    ia += 1
                                    va = UInt64.MAX
                                    if ia < width:
                                        va = sh_keys.unsafe_load(a_base + ia)
                                else:
                                    sh_keys.unsafe_store(dst + o0 + t, vb)
                                    ib += 1
                                    vb = UInt64.MAX
                                    if ib < width:
                                        vb = sh_keys.unsafe_load(b_base + ib)
                        barrier()
                        var swap = src
                        src = dst
                        dst = swap
                        width = width * 2
                else:
                    # the two stable radix passes as a ten-pass merge: every element
                    # finds its output slot by a binary search of the sibling run (the
                    # composites are distinct, so "strictly below" needs no tie rule).
                    # Ten passes, so the order ends where it began, at `sh_keys[0:1024]`.
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
            comptime if AFT_N09:
                barrier()


def yeti_rank_decay_table_kernel(
    decay_speed: Float32,
    table: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`0.15 * pow(decaySpeed, d)` for every pair distance `d` in
    `[0, YETI_TASK_POSITIONS)`: the block kernel's hoisted `lane_decay`,
    which is a pure function of `j - queryBegin - 1`, tabled once per call
    so `yeti_rank_task_block16k_kernel` can read it for a slot it does not
    own. The same expression on the same device, so the same bits."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        table.unsafe_store(
            i, Float32(0.15) * identical_pow(decay_speed, Float32(i))
        )


def _sort4_keys(mut v: SIMD[DType.uint64, 4]):
    """Ascending order of four distinct keys (a five-comparator network):
    the merge passes of width 1 and 2 over one aligned run of four."""
    var t = UInt64(0)
    if v[1] < v[0]:
        t = v[0]
        v[0] = v[1]
        v[1] = t
    if v[3] < v[2]:
        t = v[2]
        v[2] = v[3]
        v[3] = t
    if v[2] < v[0]:
        t = v[0]
        v[0] = v[2]
        v[2] = t
    if v[3] < v[1]:
        t = v[1]
        v[1] = v[3]
        v[3] = t
    if v[2] < v[1]:
        t = v[1]
        v[1] = v[2]
        v[2] = t


def yeti_rank_task_block16k_kernel(
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
    decay_table: MutPointer[Float32, MutAnyOrigin],
    der_acc: MutPointer[Float32, MutAnyOrigin],
    weight_acc: MutPointer[Float32, MutAnyOrigin],
    n_tasks_in: Int32 = Int32(0),
):
    """`yeti_rank_task_block_kernel` on 16 KiB of threadgroup memory
    (`YETI_TASK_16K`, lane/apple-fast-trees-yeti). Grid `(n_tasks, 1, 1)`,
    block `(YETI_THREADS, 1, 1)`; every loop bound is the same on every
    thread of a block, so every barrier is reached by every thread.

    Shared memory: the composite sort keys (1024 x 8 bytes), `relev *
    weight` and the exps (2 x 1024 x 4 bytes). No ping-pong copy and no
    accumulator arrays.

    WHY NO BIT MOVES against the 32 KiB kernel. (1) The draws are the same
    text in the same (round, lane) order on the same per-thread stream.
    (2) The sort computes no float: the merge of widths 1 and 2 is a
    sorting network over each aligned run of four, and the eight wider
    passes are the same rank merge, in place: every thread reads its four
    keys and finds their output slots, a barrier, then writes them, a
    barrier. Distinct composites, so the result is the one ascending order.
    (3) The pairs: `pairWeight` and `ll` are the same expressions on the
    same operands; the decay for a slot `j` is `decay_table[j - qb - 1]`,
    the hoisted `lane_decay` tabled by `yeti_rank_decay_table_kernel`.
    (4) The accumulation ORDER. In the 32 KiB kernel a document's sums take
    at most ONE contribution per (round, lane k, phase) step, each thread
    writing a distinct document, so the order of a document's float sums is
    (round, lane of the ordered slot, phase). Here the document's OWNER
    thread gathers those contributions instead of the slot's thread
    scattering them: a document at sorted rank `r` is `doc1` of slot
    `r + 1` (phase 1, lane `(r + 1) // 256`) when that slot is still its
    query, and `doc2` of slot `r` (phase 2, lane `r // 256`) when `r` is not
    its query's begin. The owner applies the two in that same (lane, phase)
    order: phase 2 first only when `r + 1` starts a new lane. The sums live
    in registers, start at 0.0, pass through `ftz` at every add as before,
    and are stored once. (5) The rank is a lower-bound search of the
    thread's own composite in the sorted keys."""
    var sh_keys = stack_allocation[
        YETI_TASK_POSITIONS,
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

    var task_count = Int(grid_dim.x)
    comptime if AFT_N09:
        if n_tasks_in > Int32(0):
            task_count = Int(n_tasks_in)
    comptime for task_lane in range(AFT_YETI_TASKS_PER_BLOCK):
        var task = Int(block_idx.x) + task_lane * Int(grid_dim.x)
        if task < task_count:
            var tid = Int(thread_idx.x)
            var offset = Int(task_offsets.unsafe_load(task))
            var size = Int(task_sizes.unsafe_load(task))
            var first_qid = row_qids.unsafe_load(offset)
            var pad_qid = row_qids.unsafe_load(offset + size - 1) + UInt32(1) - first_qid
            var perms = Int(permutations_in)
        
            # this thread's stream (`:224-237`), advanced through its lanes in order
            # every round
            var s = _task_seed(task_qids.unsafe_load(task), tid, cuda_seed)
        
            # per lane: the local query id, the query begin, this document's
            # `relev * weight` and exp, and its two accumulators
            var lane_qid = SIMD[DType.uint32, YETI_LANES](0)
            var lane_begin = SIMD[DType.int32, YETI_LANES](0)
            var own_relev = SIMD[DType.float32, YETI_LANES](0.0)
            var own_exp = SIMD[DType.float32, YETI_LANES](0.0)
            var acc_der = SIMD[DType.float32, YETI_LANES](0.0)
            var acc_weight = SIMD[DType.float32, YETI_LANES](0.0)
            var my_key = SIMD[DType.uint64, YETI_LANES](0)
            for k in range(YETI_LANES):
                var p = tid + YETI_THREADS * k
                var qid = pad_qid
                # the padding is one query that begins at the first padded position
                var begin = size
                var rv = Float32(1000.0)
                var ev = Float32(1000.0)
                if p < size:
                    var row_qid = row_qids.unsafe_load(offset + p)
                    qid = row_qid - first_qid
                    begin = Int(q_offsets.unsafe_load(Int(row_qid))) - offset
                    var w = Float32(1.0)
                    if has_weights != Int32(0):
                        w = weights.unsafe_load(offset + p)
                    rv = ftz(relev.unsafe_load(offset + p) * w)
                    ev = routed_exp(min(approx.unsafe_load(offset + p), Float32(70.0)))
                sh_relev.unsafe_store(p, rv)
                sh_exp.unsafe_store(p, ev)
                own_relev[k] = rv
                own_exp[k] = ev
                lane_qid[k] = qid
                lane_begin[k] = Int32(begin)
            barrier()
        
            for _ in range(perms):
                # the draws (`:112-121`)
                for k in range(YETI_LANES):
                    var p = tid + YETI_THREADS * k
                    var val = Float32(-1000.0)
                    if p < size:
                        val = own_exp[k]
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
                    my_key[k] = composite
                barrier()
        
                # the merge passes of width 1 and 2: each thread sorts the aligned
                # run of four it reads back, in registers
                var run = SIMD[DType.uint64, YETI_LANES](0)
                for e in range(YETI_LANES):
                    run[e] = sh_keys.unsafe_load(YETI_LANES * tid + e)
                _sort4_keys(run)
                for e in range(YETI_LANES):
                    sh_keys.unsafe_store(YETI_LANES * tid + e, run[e])
                barrier()
        
                # the eight wider passes, the rank merge in place: every element
                # finds its output slot by a binary search of the sibling run (the
                # composites are distinct, so "strictly below" needs no tie rule);
                # all reads of a pass precede all its writes.
                var width = YETI_LANES
                while width < YETI_TASK_POSITIONS:
                    var dest = SIMD[DType.int32, YETI_LANES](0)
                    for e in range(YETI_LANES):
                        var i = YETI_LANES * tid + e
                        var lo = (i // (2 * width)) * (2 * width)
                        var mid = lo + width
                        var sibling = mid
                        var own_rank = i - lo
                        if i >= mid:
                            sibling = lo
                            own_rank = i - mid
                        var key = sh_keys.unsafe_load(i)
                        var base = 0
                        var length = width
                        while length > 1:
                            var half = length // 2
                            if sh_keys.unsafe_load(sibling + base + half - 1) < key:
                                base += half
                            length -= half
                        var below = base
                        if sh_keys.unsafe_load(sibling + base) < key:
                            below += 1
                        run[e] = key
                        dest[e] = Int32(lo + own_rank + below)
                    barrier()
                    for e in range(YETI_LANES):
                        sh_keys.unsafe_store(Int(dest[e]), run[e])
                    barrier()
                    width = width * 2
        
                # the pairs (`:130-168`), gathered by each document's owner in the
                # (lane, phase) order of the 32 KiB kernel's scatter
                for k in range(YETI_LANES):
                    var p = tid + YETI_THREADS * k
                    if p < size:
                        # this document's sorted rank: the lower bound of its own
                        # composite, which is present
                        var key = my_key[k]
                        var base = 0
                        var length = YETI_TASK_POSITIONS
                        while length > 1:
                            var half = length // 2
                            if sh_keys.unsafe_load(base + half - 1) < key:
                                base += half
                            length -= half
                        var r = base
                        if sh_keys.unsafe_load(base) < key:
                            r = base + 1
                        var qb = Int(lane_begin[k])
                        var relev_p = own_relev[k]
                        var exp_p = own_exp[k]
        
                        # phase 1 of slot r + 1: this document is `doc1`
                        var has1 = False
                        var pw1 = Float32(0.0)
                        var ll1 = Float32(0.0)
                        if r + 1 < YETI_TASK_POSITIONS:
                            var next_key = sh_keys.unsafe_load(r + 1)
                            if (next_key >> UInt64(42)) == UInt64(lane_qid[k]):
                                has1 = True
                                var idx2 = Int(next_key & UInt64(1023))
                                var relev2 = sh_relev.unsafe_load(idx2)
                                var approx2 = sh_exp.unsafe_load(idx2)
                                # slot j = r + 1: `0.15 * pow(decaySpeed, j - qb - 1)`
                                var decay = decay_table.unsafe_load(r - qb)
                                pw1 = ftz(decay * abs(relev_p - relev2) / Float32(perms))
                                var sel = -exp_p
                                if relev_p > relev2:
                                    sel = approx2
                                ll1 = ftz(pw1 * sel / (approx2 + exp_p))
        
                        # phase 2 of slot r: this document is `doc2`
                        var has2 = r != qb
                        var pw2 = Float32(0.0)
                        var ll2 = Float32(0.0)
                        if has2:
                            var prev_key = sh_keys.unsafe_load(r - 1)
                            var idx1 = Int(prev_key & UInt64(1023))
                            var relev1 = sh_relev.unsafe_load(idx1)
                            var approx1 = sh_exp.unsafe_load(idx1)
                            # slot j = r: `0.15 * pow(decaySpeed, j - qb - 1)`
                            var decay = decay_table.unsafe_load(r - qb - 1)
                            pw2 = ftz(decay * abs(relev1 - relev_p) / Float32(perms))
                            var sel = -approx1
                            if relev1 > relev_p:
                                sel = exp_p
                            ll2 = ftz(pw2 * sel / (exp_p + approx1))
        
                        # the scatter's step order: phase 2's slot r sits in an
                        # earlier lane than phase 1's slot r + 1 only when r + 1
                        # begins a lane
                        var two_first = has1 and has2 and ((r // YETI_THREADS) < ((r + 1) // YETI_THREADS))
                        if two_first:
                            acc_weight[k] = ftz(acc_weight[k] + pw2)
                            acc_der[k] = ftz(acc_der[k] + (-ll2))
                        if has1:
                            acc_weight[k] = ftz(acc_weight[k] + pw1)
                            acc_der[k] = ftz(acc_der[k] + ll1)
                        if has2 and not two_first:
                            acc_weight[k] = ftz(acc_weight[k] + pw2)
                            acc_der[k] = ftz(acc_der[k] + (-ll2))
                # the next round's draws overwrite the keys every thread just read
                barrier()
        
            for k in range(YETI_LANES):
                var p = tid + YETI_THREADS * k
                if p < size:
                    der_acc.unsafe_store(offset + p, acc_der[k])
                    weight_acc.unsafe_store(offset + p, acc_weight[k])
            comptime if AFT_N09:
                barrier()


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
    var w_abs = Float32(0.0)
    var g_abs = Float32(0.0)
    # Logical 256-row tiles preserve all caller allocation/fold contracts.
    comptime for row_lane in range(MSE_BLOCK_SIZE // AFT_YETI_ROW_THREADS):
        var i = Int(block_idx.x) * MSE_BLOCK_SIZE + Int(thread_idx.x) + row_lane * AFT_YETI_ROW_THREADS
        if i < n_rows:
            var der = der_acc.unsafe_load(i)
            var weight = weight_acc.unsafe_load(i)
            comptime if estimation:
                var dst = i
                if has_write_map != Int32(0):
                    dst = Int(write_map.unsafe_load(i))
                stats.unsafe_store(dst, der)
                stats.unsafe_store(n_rows + dst, weight)
            else:
                stats.unsafe_store(i, weight)
                stats.unsafe_store(n_rows + i, der)
            if compute_magnitudes != Int32(0):
                w_abs += abs(weight)
                g_abs += abs(der)
    if compute_fv != Int32(0) and thread_idx.x == 0:
        function_value.unsafe_store(Int(block_idx.x), Float32(0.0))
    if compute_magnitudes != Int32(0):
        var w_total = pinned_block_sum[block_size=AFT_YETI_ROW_THREADS](w_abs)
        var g_total = pinned_block_sum[block_size=AFT_YETI_ROW_THREADS](g_abs)
        if thread_idx.x == 0:
            plane_magnitudes.unsafe_store(2 * Int(block_idx.x), w_total)
            plane_magnitudes.unsafe_store(2 * Int(block_idx.x) + 1, g_total)


def yeti_rank_task_fused_kernel[estimation: Bool](
    predictions: MutPointer[Float32, MutAnyOrigin],
    index_map: MutPointer[UInt32, MutAnyOrigin],
    has_index_map: Int32,
    relev: MutPointer[Float32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    has_weights: Int32,
    row_qids: MutPointer[UInt32, MutAnyOrigin],
    q_offsets: MutPointer[UInt32, MutAnyOrigin],
    q_sizes: MutPointer[UInt32, MutAnyOrigin],
    task_offsets: MutPointer[UInt32, MutAnyOrigin],
    task_sizes: MutPointer[UInt32, MutAnyOrigin],
    task_qids: MutPointer[UInt32, MutAnyOrigin],
    cuda_seed: UInt32,
    decay_speed: Float32,
    permutations_in: Int32,
    n_rows_in: Int32,
    n_tasks_in: Int32,
    row_blocks_in: Int32,
    der_acc: MutPointer[Float32, MutAnyOrigin],
    weight_acc: MutPointer[Float32, MutAnyOrigin],
    stats: MutPointer[Float32, MutAnyOrigin],
    function_value: MutPointer[Float32, MutAnyOrigin],
    compute_fv: Int32,
    plane_magnitudes: MutPointer[Float32, MutAnyOrigin],
    compute_magnitudes: Int32,
):
    """`YETI_TASK_FUSED` (lane af-sym-multi): `yeti_rank_task_block_kernel`
    with the point load, the query means, the centering and the row
    scatter folded in. One block per task, `YETI_THREADS` threads, 4 lanes a
    thread; every loop bound is uniform over the block.

    The point: `predictions[index_map[row]]` when `has_index_map` (the
    estimation's bin-order cursor read through the inverse order), else
    `predictions[row]`. The mean: the task's positions hold whole queries,
    so a segmented inclusive sum over the 1024 positions (Hillis-Steele,
    ten steps, the query id as the segment key) leaves each query's total
    at its last position; `mean = ftz(total / size)` is the reference's
    `ComputeGroupMeans` with unit weights (`sum / sum of 1.0`) in another
    fold order; `centered = ftz(approx - mean)` is `RemoveQueryMeans`. The
    scan runs in the sort-key scratch (16 KiB: positions `[0, 1024)` and
    `[1024, 2048)` of its float view, the query ids at `[2048, 3072)` of
    its uint view) before the first draw writes a key, so the 32 KiB
    budget is unchanged. The stores: `der_acc` / `weight_acc` as the block
    kernel (the reuse arm reads them), the planes as `yeti_rank_row_kernel`
    (SEARCH `[pair weight, der]` at the row, ESTIMATION `[der, pair
    weight]` at `index_map[row]`), the value partials 0.0 over all
    `row_blocks` (grid-strided over the tasks), the magnitude partials one
    pair per TASK (the caller folds `n_tasks` of them)."""
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
    # the scan's views of the sort-key scratch
    var sh_sum = sh_keys.unsafe_bitcast[Float32]()
    var sh_qid = sh_keys.unsafe_bitcast[UInt32]()
    comptime QID_AT = 2 * YETI_TASK_POSITIONS

    var task = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var offset = Int(task_offsets.unsafe_load(task))
    var size = Int(task_sizes.unsafe_load(task))
    var n_rows = Int(n_rows_in)
    var first_qid = row_qids.unsafe_load(offset)
    var pad_qid = row_qids.unsafe_load(offset + size - 1) + UInt32(1) - first_qid
    var perms = Int(permutations_in)

    var s = _task_seed(task_qids.unsafe_load(task), tid, cuda_seed)

    var lane_qid = SIMD[DType.uint32, YETI_LANES](0)
    var lane_begin = SIMD[DType.int32, YETI_LANES](0)
    var lane_size = SIMD[DType.int32, YETI_LANES](0)
    var lane_approx = SIMD[DType.float32, YETI_LANES](0.0)
    var lane_decay = SIMD[DType.float32, YETI_LANES](0.0)
    # the point and the query ids into the scan scratch
    for k in range(YETI_LANES):
        var p = tid + YETI_THREADS * k
        var qid = pad_qid
        var begin = size
        var qsize = 0
        var a = Float32(0.0)
        if p < size:
            var row = offset + p
            var row_qid = row_qids.unsafe_load(row)
            qid = row_qid - first_qid
            begin = Int(q_offsets.unsafe_load(Int(row_qid))) - offset
            qsize = Int(q_sizes.unsafe_load(Int(row_qid)))
            var src = row
            if has_index_map != Int32(0):
                src = Int(index_map.unsafe_load(row))
            a = predictions.unsafe_load(src)
        sh_sum.unsafe_store(p, a)
        sh_qid.unsafe_store(QID_AT + p, qid)
        lane_qid[k] = qid
        lane_begin[k] = Int32(begin)
        lane_size[k] = Int32(qsize)
        lane_approx[k] = a
        if p != begin:
            lane_decay[k] = Float32(0.15) * identical_pow(
                decay_speed, Float32(p - begin - 1)
            )
    barrier()
    # the segmented inclusive sum: ten doubling steps, ping-pong halves
    var src_at = 0
    var dst_at = YETI_TASK_POSITIONS
    var seg_d = 1
    while seg_d < YETI_TASK_POSITIONS:
        for k in range(YETI_LANES):
            var p = tid + YETI_THREADS * k
            var v = sh_sum.unsafe_load(src_at + p)
            if p >= seg_d:
                if sh_qid.unsafe_load(QID_AT + p - seg_d) == sh_qid.unsafe_load(QID_AT + p):
                    v = v + sh_sum.unsafe_load(src_at + p - seg_d)
            sh_sum.unsafe_store(dst_at + p, v)
        barrier()
        var swap = src_at
        src_at = dst_at
        dst_at = swap
        seg_d = seg_d * 2
    # the centered exps and relevances; the accumulators zeroed
    for k in range(YETI_LANES):
        var p = tid + YETI_THREADS * k
        if p < size:
            var last = Int(lane_begin[k]) + Int(lane_size[k]) - 1
            var total = sh_sum.unsafe_load(src_at + last)
            var mean = ftz(total / Float32(Int(lane_size[k])))
            var centered = ftz(lane_approx[k] - mean)
            var w = Float32(1.0)
            if has_weights != Int32(0):
                w = weights.unsafe_load(offset + p)
            sh_relev.unsafe_store(p, ftz(relev.unsafe_load(offset + p) * w))
            sh_exp.unsafe_store(p, routed_exp(min(centered, Float32(70.0))))
        else:
            sh_relev.unsafe_store(p, Float32(1000.0))
            sh_exp.unsafe_store(p, Float32(1000.0))
        sh_der.unsafe_store(p, Float32(0.0))
        sh_weight.unsafe_store(p, Float32(0.0))
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

        # lane/apple-fast-rec-sym: the block kernel's sort, both arms
        # (main's `YETI_FAST_SORT` register bitonic + merge-path, or the
        # ten-pass merge under `-D MOJOLEARN_YETI_FAST_SORT_OFF`), so the
        # fused call sorts exactly as the five-launch arm it replaces; the
        # composites are distinct, so both give one order
        var src = 0
        comptime if YETI_FAST_SORT:
            comptime assert YETI_THREADS * YETI_LANES == YETI_TASK_POSITIONS
            comptime assert YETI_THREADS % YETI_SIMD_W == 0
            # (1) a bitonic sort of each simdgroup's 128 contiguous keys in
            # registers: element e = lane * 4 + r of the simdgroup's run
            var ln = tid % YETI_SIMD_W
            var run_base = (tid // YETI_SIMD_W) * YETI_SIMD_RUN + ln * YETI_LANES
            var v = SIMD[DType.uint64, YETI_LANES](0)
            comptime for r in range(YETI_LANES):
                v[r] = sh_keys.unsafe_load(run_base + r)
            comptime for kk in range(1, 8):
                comptime k = 1 << kk
                comptime for t in range(kk):
                    comptime jj = kk - 1 - t
                    comptime j = 1 << jj
                    comptime if j < YETI_LANES:
                        comptime for r in range(YETI_LANES):
                            comptime if (r & j) == 0:
                                var a = v[r]
                                var b = v[r | j]
                                var asc = ((ln * YETI_LANES + r) & k) == 0
                                if (a > b) == asc:
                                    v[r] = b
                                    v[r | j] = a
                    else:
                        comptime m = j // YETI_LANES
                        var lower = (ln & m) == 0
                        comptime for r in range(YETI_LANES):
                            var x = v[r]
                            var o_hi = shuffle_xor(UInt32(x >> UInt64(32)), UInt32(m))
                            var o_lo = shuffle_xor(
                                UInt32(x & UInt64(0xFFFFFFFF)), UInt32(m)
                            )
                            var o = (UInt64(o_hi) << UInt64(32)) | UInt64(o_lo)
                            var asc = ((ln * YETI_LANES + r) & k) == 0
                            if lower == asc:
                                v[r] = min(x, o)
                            else:
                                v[r] = max(x, o)
            comptime for r in range(YETI_LANES):
                sh_keys.unsafe_store(run_base + r, v[r])
            barrier()
            # (2) merge-path passes: thread tid writes outputs 4 * tid .. +3
            # of the merged pair of runs; one co-rank binary search, then a
            # sequential merge with the run heads in registers. UInt64.MAX
            # is a sentinel no composite reaches (they are below 2^52).
            var dst = YETI_TASK_POSITIONS
            var width = YETI_SIMD_RUN
            var o0 = tid * AFT_YETI_MERGE_LANES
            while width < YETI_TASK_POSITIONS:
                if o0 < YETI_TASK_POSITIONS:
                    var lo = (o0 // (2 * width)) * (2 * width)
                    var d = o0 - lo
                    var a_base = src + lo
                    var b_base = a_base + width
                    var i_lo = max(0, d - width)
                    var i_hi = min(d, width)
                    while i_lo < i_hi:
                        var mid = (i_lo + i_hi) // 2
                        if sh_keys.unsafe_load(a_base + mid) < sh_keys.unsafe_load(
                            b_base + d - mid - 1
                        ):
                            i_lo = mid + 1
                        else:
                            i_hi = mid
                    var ia = i_lo
                    var ib = d - i_lo
                    var va = UInt64.MAX
                    if ia < width:
                        va = sh_keys.unsafe_load(a_base + ia)
                    var vb = UInt64.MAX
                    if ib < width:
                        vb = sh_keys.unsafe_load(b_base + ib)
                    comptime for t in range(AFT_YETI_MERGE_LANES):
                        if va < vb:
                            sh_keys.unsafe_store(dst + o0 + t, va)
                            ia += 1
                            va = UInt64.MAX
                            if ia < width:
                                va = sh_keys.unsafe_load(a_base + ia)
                        else:
                            sh_keys.unsafe_store(dst + o0 + t, vb)
                            ib += 1
                            vb = UInt64.MAX
                            if ib < width:
                                vb = sh_keys.unsafe_load(b_base + ib)
                barrier()
                var swap = src
                src = dst
                dst = swap
                width = width * 2
        else:
            # the two stable radix passes as a ten-pass merge: every element
            # finds its output slot by a binary search of the sibling run (the
            # composites are distinct, so "strictly below" needs no tie rule).
            # Ten passes, so the order ends where it began, at `sh_keys[0:1024]`.
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


    # the stores: accumulators, planes, magnitudes
    var w_abs = Float32(0.0)
    var g_abs = Float32(0.0)
    for k in range(YETI_LANES):
        var p = tid + YETI_THREADS * k
        if p < size:
            var row = offset + p
            var der = sh_der.unsafe_load(p)
            var weight = sh_weight.unsafe_load(p)
            der_acc.unsafe_store(row, der)
            weight_acc.unsafe_store(row, weight)
            comptime if estimation:
                var dst = row
                if has_index_map != Int32(0):
                    dst = Int(index_map.unsafe_load(row))
                stats.unsafe_store(dst, der)
                stats.unsafe_store(n_rows + dst, weight)
            else:
                stats.unsafe_store(row, weight)
                stats.unsafe_store(n_rows + row, der)
            w_abs += abs(weight)
            g_abs += abs(der)
    if compute_fv != Int32(0) and tid == 0:
        # `FillBuffer(FunctionValue, 0)`: every row block's partial
        var b = task
        while b < Int(row_blocks_in):
            function_value.unsafe_store(b, Float32(0.0))
            b += Int(n_tasks_in)
    if compute_magnitudes != Int32(0):
        # lane/apple-fast-rec-sym: the two block sums fold in `sh_relev` /
        # `sh_exp` (free once the pairs are done) by a halving tree, NOT
        # `pinned_block_sum`: its FAST arm (`block.sum`) takes its own
        # shared slab, 64 B for the two calls, on top of this kernel's
        # 32 KiB, and Metal refused the launch at 32,832 B
        # (LEDGER 2026-10-03, SYM_MULTI_ALL yetirank at warm-up).
        barrier()
        sh_relev.unsafe_store(tid, w_abs)
        sh_exp.unsafe_store(tid, g_abs)
        barrier()
        var half = YETI_THREADS // 2
        while half > 0:
            if tid < half:
                sh_relev.unsafe_store(
                    tid, sh_relev.unsafe_load(tid) + sh_relev.unsafe_load(tid + half)
                )
                sh_exp.unsafe_store(
                    tid, sh_exp.unsafe_load(tid) + sh_exp.unsafe_load(tid + half)
                )
            barrier()
            half = half // 2
        if tid == 0:
            plane_magnitudes.unsafe_store(2 * task, sh_relev.unsafe_load(0))
            plane_magnitudes.unsafe_store(2 * task + 1, sh_exp.unsafe_load(0))


def launch_yeti_rank_fused[estimation: Bool](
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
    """`YETI_TASK_FUSED`: `launch_yeti_rank_with` as one launch. The caller
    checked `y.n_tasks <= row_blocks` (the magnitude partials, one pair per
    task, must fit the caller's per-row-block buffer). `y.query.qids` was
    filled once per fit by `make_yeti_rank_target_buffers`."""
    var n_rows = y.query.n_rows
    var row_blocks = (n_rows + MSE_BLOCK_SIZE - 1) // MSE_BLOCK_SIZE
    if use_inverse:
        ctx.enqueue_function[yeti_rank_task_fused_kernel[estimation]](
            predictions.unsafe_ptr(), y.query.inverse.unsafe_ptr(), Int32(1),
            y.query.targets.unsafe_ptr(), y.query.weights.unsafe_ptr(),
            Int32(1) if y.query.has_weights else Int32(0),
            y.query.qids.unsafe_ptr(), y.query.q_offsets.unsafe_ptr(),
            y.query.q_sizes.unsafe_ptr(),
            y.d_task_offsets.unsafe_ptr(), y.d_task_sizes.unsafe_ptr(),
            y.d_task_qids.unsafe_ptr(),
            yeti_rank_cuda_seed(seed), y.decay, Int32(y.permutations),
            Int32(n_rows), Int32(y.n_tasks), Int32(row_blocks),
            y.d_der_acc.unsafe_ptr(), y.d_weight_acc.unsafe_ptr(),
            stats.unsafe_ptr(), function_value.unsafe_ptr(),
            Int32(1) if compute_fv else Int32(0),
            plane_magnitudes.unsafe_ptr(),
            Int32(1) if compute_magnitudes else Int32(0),
            grid_dim=(y.n_tasks, 1, 1),
            block_dim=(YETI_THREADS, 1, 1),
        )
    else:
        ctx.enqueue_function[yeti_rank_task_fused_kernel[estimation]](
            predictions.unsafe_ptr(), y.query.no_indices.unsafe_ptr(), Int32(0),
            y.query.targets.unsafe_ptr(), y.query.weights.unsafe_ptr(),
            Int32(1) if y.query.has_weights else Int32(0),
            y.query.qids.unsafe_ptr(), y.query.q_offsets.unsafe_ptr(),
            y.query.q_sizes.unsafe_ptr(),
            y.d_task_offsets.unsafe_ptr(), y.d_task_sizes.unsafe_ptr(),
            y.d_task_qids.unsafe_ptr(),
            yeti_rank_cuda_seed(seed), y.decay, Int32(y.permutations),
            Int32(n_rows), Int32(y.n_tasks), Int32(row_blocks),
            y.d_der_acc.unsafe_ptr(), y.d_weight_acc.unsafe_ptr(),
            stats.unsafe_ptr(), function_value.unsafe_ptr(),
            Int32(1) if compute_fv else Int32(0),
            plane_magnitudes.unsafe_ptr(),
            Int32(1) if compute_magnitudes else Int32(0),
            grid_dim=(y.n_tasks, 1, 1),
            block_dim=(YETI_THREADS, 1, 1),
        )



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
    comptime if YETI_TASK_FUSED:
        # the row -> query map once per fit; the fused kernel reads it every
        # call (the five-launch arm recomputes it per call, as the reference)
        launch_compute_group_ids(
            ctx, query.q_sizes, query.q_offsets, UInt32(0), query.q_count,
            query.qids,
        )
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
    comptime if YETI_TASK_FUSED:
        # lane af-sym-multi: the whole call in one launch (its docstring)
        if y.n_tasks <= row_blocks:
            launch_yeti_rank_fused[estimation](
                ctx, y, predictions, use_inverse, seed, stats, function_value,
                compute_fv, plane_magnitudes, compute_magnitudes,
            )
            return
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
        launch_aft_group_means(
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
        launch_aft_group_means(
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
        comptime if YETI_TASK_16K:
            # lane/apple-fast-trees-yeti: the 16 KiB block kernel. The decay
            # table rides the first 1024 cells of `y.s_exp`, scratch of the
            # sequential kernel, which this arm never launches.
            ctx.enqueue_function[yeti_rank_decay_table_kernel](
                y.decay, y.s_exp.unsafe_ptr(), Int32(YETI_TASK_POSITIONS),
                grid_dim=((YETI_TASK_POSITIONS + YETI_THREADS - 1) // YETI_THREADS, 1, 1),
                block_dim=(YETI_THREADS, 1, 1),
            )
            ctx.enqueue_function[yeti_rank_task_block16k_kernel](
                y.d_point.unsafe_ptr(), y.query.targets.unsafe_ptr(),
                y.query.weights.unsafe_ptr(),
                Int32(1) if y.query.has_weights else Int32(0),
                y.query.qids.unsafe_ptr(), y.query.q_offsets.unsafe_ptr(),
                y.d_task_offsets.unsafe_ptr(), y.d_task_sizes.unsafe_ptr(),
                y.d_task_qids.unsafe_ptr(),
                yeti_rank_cuda_seed(seed), y.decay, Int32(y.permutations),
                y.s_exp.unsafe_ptr(),
                y.d_der_acc.unsafe_ptr(), y.d_weight_acc.unsafe_ptr(),
                Int32(y.n_tasks),
                grid_dim=((y.n_tasks + AFT_YETI_TASKS_PER_BLOCK - 1) // AFT_YETI_TASKS_PER_BLOCK, 1, 1),
                block_dim=(YETI_THREADS, 1, 1),
            )
        else:
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
                Int32(y.n_tasks),
                grid_dim=((y.n_tasks + AFT_YETI_TASKS_PER_BLOCK - 1) // AFT_YETI_TASKS_PER_BLOCK, 1, 1),
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
            block_dim=(AFT_YETI_ROW_THREADS, 1, 1),
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
            block_dim=(AFT_YETI_ROW_THREADS, 1, 1),
        )


def launch_yeti_rank_estimation_from_search(
    ctx: DeviceContext,
    mut y: YetiRankTargetBuffers,
    mut stats: DeviceBuffer[DType.float32],
    mut function_value: DeviceBuffer[DType.float32],
    compute_fv: Bool,
    mut plane_magnitudes: DeviceBuffer[DType.float32],
    compute_magnitudes: Bool,
) raises:
    """`YETI_EST_REUSE_SEARCH`: the estimation planes `[der, pair weight]`
    at each row's bin position (`query.inverse`), read from the
    accumulators the tree's search call left in `d_der_acc` /
    `d_weight_acc`. Only the scatter of `launch_yeti_rank_with[True]`; the
    caller guarantees the cursor has not moved since that search call and
    that no estimation launch has overwritten the accumulators."""
    var n_rows = y.query.n_rows
    var row_blocks = (n_rows + MSE_BLOCK_SIZE - 1) // MSE_BLOCK_SIZE
    ctx.enqueue_function[yeti_rank_row_kernel[True]](
        y.d_der_acc.unsafe_ptr(), y.d_weight_acc.unsafe_ptr(), Int32(n_rows),
        y.query.inverse.unsafe_ptr(), Int32(1),
        stats.unsafe_ptr(), function_value.unsafe_ptr(),
        Int32(1) if compute_fv else Int32(0),
        plane_magnitudes.unsafe_ptr(),
        Int32(1) if compute_magnitudes else Int32(0),
        grid_dim=(row_blocks, 1, 1),
        block_dim=(AFT_YETI_ROW_THREADS, 1, 1),
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
