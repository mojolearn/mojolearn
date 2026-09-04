# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The work-item structs the builder and its kernels share."""

from extratrees.impl.decisiontree.batched_levelalgo.split import Split


@fieldwise_init
struct InstanceRange(ImplicitlyCopyable, Movable):
    """The range of instances belonging to a node, as a range in `Dataset.row_ids`."""

    var begin: Int32
    var count: Int32


@fieldwise_init
struct NodeWorkItem(ImplicitlyCopyable, Movable):
    """One node awaiting a split. `builder_kernels.cuh:39-43`."""

    var idx: Int32
    """Index of the work item in the tree. Theirs is `size_t idx`."""

    var depth: Int32
    var instances: InstanceRange


@fieldwise_init
struct WorkloadInfo(ImplicitlyCopyable, Movable):
    """Which node a threadblock serves, and which slice of it."""

    var nodeid: Int32
    """Node in the batch this threadblock works on."""

    var large_nodeid: Int32
    """Counts only LARGE nodes -- those needing more than one block along the x dimension, and therefore a global-memory scratch slot."""

    var offset_blockid: Int32
    """This block's offset among all blocks working on this node."""

    var num_blocks: Int32
    """Total blocks working on this node."""


def split_not_valid(
    split: Split,
    min_impurity_decrease: Float32,
    min_samples_leaf: Int32,
    num_rows: Int32,
) -> Bool:
    """`builder_kernels.cuh:59-67` -- with DEVIATION 216 on the first clause."""
    return (
        split.best_metric_val < min_impurity_decrease
        or split.n_left < min_samples_leaf
        or (num_rows - split.n_left) < min_samples_leaf
    )



from std.ffi import external_call
from std.sys.compile import is_defined

from extratrees.checks.pcg_rng import (
    FNV1A32_BASIS,
    PCGenerator,
    fmix32,
    fnv1a32,
    uniform_int_u64,
)
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    identical_exp,
    identical_log,
    portable_logf,
)




def _log64(x: Float64) -> Float64:
    """`std::log(double)`, which is what `raft::log` becomes for a `double` argument on the host (`raft/core/math.hpp:324-331`)."""
    return external_call["log", Float64](x)


def _logf32(x: Float32) -> Float32:
    """`std::log(float)`, i.e."""
    return external_call["logf", Float32](x)


def _expf32(x: Float32) -> Float32:
    """`std::exp(float)`, i.e. `expf`. Same genericity as `_logf32`."""
    return external_call["expf", Float32](x)


def _ceil64(x: Float64) -> Float64:
    """`std::ceil(double)`, `builder.cuh:420`."""
    return external_call["ceil", Float64](x)




def excess_subsequence(
    thread_id: UInt32, tree_id: UInt32, node_id: UInt32
) -> UInt64:
    """The excess sampler's PCG subsequence."""
    var subsequence = FNV1A32_BASIS
    subsequence = fnv1a32(subsequence, thread_id)
    subsequence = fnv1a32(subsequence, tree_id)
    subsequence = fnv1a32(subsequence, node_id)
    return subsequence.cast[DType.uint64]()


comptime EXCESS_SELECTION_SALT = UInt32(0x5E1EC7ED)

comptime ET_SELECTION_HASH_FINALIZED = not is_defined[
    "MOJOLEARN_ET_RAW_SELECTION_HASH"
]()
"""A/B arm for DEVIATION 464: build with `-D MOJOLEARN_ET_RAW_SELECTION_HASH=1` to restore the pre-464 un-finalized fnv chain, host and device together (so `sampler_kernel_check`'s host-vs-device control stays bit-identical in BOTH builds)."""


def excess_selection_hash_raw(
    tree_id: UInt32, node_id: UInt32, col: UInt32
) -> UInt32:
    """The bare fnv1a32 chain -- the pre-DEVIATION-464 selection key, kept as the sabotage/A-B arm."""
    var h = fnv1a32(FNV1A32_BASIS, EXCESS_SELECTION_SALT)
    h = fnv1a32(h, tree_id)
    h = fnv1a32(h, node_id)
    h = fnv1a32(h, col)
    return h


def excess_selection_hash(
    tree_id: UInt32, node_id: UInt32, col: UInt32
) -> UInt32:
    """DEVIATION 215's selection key: which `k` of the uniques survive when the excess sampler drew MORE than `k`. The fix: rank the uniques by THIS keyed hash instead of by column id, and keep the `k` smallest BY HASH -- a uniform `k`-subset of the uniques, deterministic from `(tree, node, col)` alone, identical on host and device, and free of any cross-thread draw order."""
    var h = excess_selection_hash_raw(tree_id, node_id, col)
    comptime if ET_SELECTION_HASH_FINALIZED:
        return fmix32(h)
    return h


def algo_l_subsequence(tree_id: UInt32, node_id: UInt32) -> UInt64:
    """The reservoir sampler's PCG subsequence."""
    return (tree_id.cast[DType.uint64]() << 32) | node_id.cast[DType.uint64]()



comptime SAMPLER_BLOCK_THREADS: Int = 128
"""`builder.cuh:401`, `constexpr int block_threads = 128`."""

comptime SAMPLER_MAX_SAMPLES_PER_THREAD: Int = 72
"""`builder.cuh:402`, `constexpr int max_samples_per_thread = 72; // register spillage if more than this limit`."""

comptime SAMPLE_ALL_FEATURES: Int = 0
"""No sampling: every column is a candidate."""

comptime SAMPLE_EXCESS: Int = 1
"""`excess_sample_with_replacement_kernel`, `builder_kernels.cuh:152`."""

comptime SAMPLE_ALGO_L: Int = 2
"""`algo_L_sample_kernel`, `builder_kernels.cuh:268`."""


def NON_DRAWING_SENTINEL(n: Int) -> Int32:
    """A value above every valid column id, for a slot that did not draw."""
    return Int32(n)


def sampler_arm_name(arm: Int) -> String:
    """So a check and a benchmark can PRINT which kernel ran; rule 8."""
    if arm == SAMPLE_ALL_FEATURES:
        return String("all-features")
    if arm == SAMPLE_EXCESS:
        return String("excess-with-replacement")
    if arm == SAMPLE_ALGO_L:
        return String("algo-L-reservoir")
    return String("unknown")


@fieldwise_init
struct FeatureSamplerPlan(ImplicitlyCopyable, Movable):
    """Which sampler runs for an `(n, k)`, and with which instantiation."""

    var arm: Int
    var n_parallel_samples: Int
    """`builder.cuh:419-421`. Zero on the arms that do not use it."""

    var max_samples_per_thread: Int
    """1 or 72: the `MAX_SAMPLES_PER_THREAD` template argument of the excess kernel, chosen at `builder.cuh:434-455`."""

    var block_threads: Int

    def name(self) -> String:
        return sampler_arm_name(self.arm)


def n_parallel_samples_for(n: Int, k: Int) -> Int:
    """`builder.cuh:419-421`, transcribed with their comment intact."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return n_parallel_samples_portable(n, k)
    return n_parallel_samples_libm(n, k)


def n_parallel_samples_libm(n: Int, k: Int) -> Int:
    """DEVIATION 158's arm of `n_parallel_samples_for` -- both logs through HOST libm in double, cuML's types exactly."""
    var ratio = _log64(1.0 - Float64(k) / Float64(n))
    var per_draw = _log64(1.0 - 1.0 / Float64(n))
    return Int(_ceil64(ratio / per_draw))


def n_parallel_samples_portable(n: Int, k: Int) -> Int:
    """DEVIATION 457's arm -- the same `ceil(log(1 - k/n) / log(1 - 1/n))` with both logs through row 12's `portable_logf`, so every HOST computes the same count from the same `(n, k)`."""
    from std.math import ceil

    var ratio = portable_logf(Float32(1.0) - Float32(k) / Float32(n))
    var per_draw = portable_logf(Float32(1.0) - Float32(1.0) / Float32(n))
    return Int(ceil(ratio / per_draw))


def plan_feature_sampling(n: Int, k: Int) raises -> FeatureSamplerPlan:
    """THE DISPATCH. Their shape, with the guard that encloses it:: if (dataset.n_sampled_cols != dataset.N) { // :399 constexpr int block_threads = 128; // :401 constexpr int max_samples_per_thread = 72; // :402 IdxT n_parallel_samples = std::ceil(...); // :419-421 if (max_samples_per_thread * block_threads >= n_parallel_samples) { if (n_parallel_samples <= block_threads) excess_sample_with_replacement_kernel<IdxT, 1,..."""
    if k < 1 or k > n:
        raise Error(
            String(
                "feature sampling needs 1 <= k <= n; got k=",
                k,
                " n=",
                n,
                ". cuML's max_features clamp makes this unreachable there,"
                " so there is no upstream behaviour to mirror -- DEVIATION"
                " 159.",
            )
        )
    if k == n:
        return FeatureSamplerPlan(
            SAMPLE_ALL_FEATURES, 0, 0, SAMPLER_BLOCK_THREADS
        )

    var n_parallel_samples = n_parallel_samples_for(n, k)
    if SAMPLER_MAX_SAMPLES_PER_THREAD * SAMPLER_BLOCK_THREADS >= (
        n_parallel_samples
    ):
        var max_samples_per_thread = SAMPLER_MAX_SAMPLES_PER_THREAD
        if n_parallel_samples <= SAMPLER_BLOCK_THREADS:
            max_samples_per_thread = 1
        return FeatureSamplerPlan(
            SAMPLE_EXCESS,
            n_parallel_samples,
            max_samples_per_thread,
            SAMPLER_BLOCK_THREADS,
        )
    return FeatureSamplerPlan(
        SAMPLE_ALGO_L, n_parallel_samples, 0, SAMPLER_BLOCK_THREADS
    )



comptime EXCESS_MAX_ITERATIONS: Int = 1024
"""OURS, DEVIATION 159. Their `do { } while (n_uniques < k)` has no bound."""


def excess_sample_with_replacement(
    mut colids: List[Int32],
    work_items: List[NodeWorkItem],
    tree_ids: List[Int32],
    seed: UInt64,
    n: Int,
    k: Int,
    n_parallel_samples: Int,
    max_samples_per_thread: Int = SAMPLER_MAX_SAMPLES_PER_THREAD,
    block_threads: Int = SAMPLER_BLOCK_THREADS,
) raises:
    """`excess_sample_with_replacement_kernel`, `builder_kernels.cuh:152-248`. `colids` is `[len(work_items), k]` row-major, their `:259`."""
    var n_slots = block_threads * max_samples_per_thread
    var items = List[Int32](length=n_slots, fill=0)
    var mask = List[Int32](length=n_slots, fill=0)
    var col_indices = List[Int32](length=n_slots, fill=0)

    for block_id in range(len(work_items)):
        var node_id = UInt32(work_items[block_id].idx)  # `:165`

        var gens = List[PCGenerator]()
        for thread_id in range(block_threads):
            gens.append(
                PCGenerator(
                    seed,
                    excess_subsequence(
                        UInt32(thread_id),
                        UInt32(Int(tree_ids[block_id])),
                        node_id,
                    ),
                    UInt64(0),
                )
            )

        for slot in range(n_slots):
            mask[slot] = 0
            items[slot] = 0
            col_indices[slot] = 0

        var n_uniques: Int
        var iterations = 0
        while True:  # `:188` .. `:241`, a do-while
            iterations += 1
            if iterations > EXCESS_MAX_ITERATIONS:
                raise Error(
                    String(
                        "excess sampler did not reach k=",
                        k,
                        " uniques of n=",
                        n,
                        " in ",
                        EXCESS_MAX_ITERATIONS,
                        " iterations. Theirs has no such bound and would"
                        " hang; DEVIATION 159.",
                    )
                )

            for thread_id in range(block_threads):
                for local in range(max_samples_per_thread):
                    var slot = max_samples_per_thread * thread_id + local
                    if mask[slot] == 0 and slot < n_parallel_samples:
                        items[slot] = Int32(
                            uniform_int_u64(gens[thread_id], 0, UInt64(n))
                        )
                    elif mask[slot] == 0:
                        items[slot] = NON_DRAWING_SENTINEL(n)

            sort(items)

            for slot in range(n_slots - 1, 0, -1):
                mask[slot] = 0 if items[slot] == items[slot - 1] else 1
            mask[0] = 1

            for slot in range(n_slots):
                if items[slot] >= Int32(n):
                    mask[slot] = 0

            var running: Int32 = 0
            for slot in range(n_slots):
                col_indices[slot] = running
                running += mask[slot]
            n_uniques = Int(running)

            if n_uniques >= k:  # `:241`, `while (n_uniques < k)`
                break

        var col_offset = k * block_id
        if n_uniques <= k:
            for slot in range(n_slots):
                if mask[slot] != 0 and Int(col_indices[slot]) < k:
                    colids[col_offset + Int(col_indices[slot])] = items[slot]
        else:
            var tree_u = UInt32(Int(tree_ids[block_id]))
            for slot in range(n_slots):
                if mask[slot] == 0:
                    continue
                var col = items[slot]
                var h = excess_selection_hash(
                    tree_u, node_id, UInt32(Int(col))
                )
                var rank = 0
                for s in range(n_slots):
                    if mask[s] == 0:
                        continue
                    var c2 = items[s]
                    var h2 = excess_selection_hash(
                        tree_u, node_id, UInt32(Int(c2))
                    )
                    if h2 < h or (h2 == h and c2 < col):
                        rank += 1
                if rank < k:
                    colids[col_offset + rank] = col




def algo_l_sample(
    mut colids: List[Int32],
    work_items: List[NodeWorkItem],
    tree_ids: List[Int32],
    seed: UInt64,
    n: Int,
    k: Int,
):
    """`algo_L_sample_kernel`, `builder_kernels.cuh:268-316`. `colids` is `[work_items_size, k]` row-major, their `:259`."""
    for tid in range(len(work_items)):
        var node_id = UInt32(work_items[tid].idx)  # `:279`
        var gen = PCGenerator(
            seed,
            algo_l_subsequence(UInt32(Int(tree_ids[tid])), node_id),
            UInt64(0),
        )  # `:280-281`

        var fp_uniform_val = gen.next_float()
        var W = Float64(_expf32(_logf32(fp_uniform_val) / Float32(k)))

        var col = 0
        while True:
            colids[tid * k + col] = Int32(col)
            if col == k - 1:
                break
            else:
                col += 1

        while col < n:
            fp_uniform_val = gen.next_float()
            col += (
                Int(Float64(_logf32(fp_uniform_val)) / _log64(1.0 - W)) + 1
            )
            if col < n:
                var int_uniform_val = Int(
                    uniform_int_u64(gen, 0, UInt64(k))
                )
                colids[tid * k + int_uniform_val] = Int32(col)
                fp_uniform_val = gen.next_float()
                W *= Float64(_expf32(_logf32(fp_uniform_val) / Float32(k)))




def sample_features(
    mut colids: List[Int32],
    work_items: List[NodeWorkItem],
    tree_id: Int32,
    seed: UInt64,
    n: Int,
    k: Int,
) raises -> FeatureSamplerPlan:
    """One-tree form of `sample_features_pertree` below: every work item belongs to `tree_id`."""
    var tree_ids = List[Int32](length=len(work_items), fill=tree_id)
    return sample_features_pertree(colids, work_items, tree_ids, seed, n, k)


def sample_features_pertree(
    mut colids: List[Int32],
    work_items: List[NodeWorkItem],
    tree_ids: List[Int32],
    seed: UInt64,
    n: Int,
    k: Int,
) raises -> FeatureSamplerPlan:
    """Fill `colids` with `k` distinct columns per work item, and SAY WHICH KERNEL DID IT. `colids` must already be `len(work_items) * k` long; `tree_ids` holds one tree id PER WORK ITEM (DEVIATION 211: a batch may span trees, and each item's draws are keyed by ITS tree)."""
    if len(tree_ids) != len(work_items):
        raise Error(
            "sample_features_pertree: "
            + String(len(work_items))
            + " work items but "
            + String(len(tree_ids))
            + " tree ids"
        )
    var plan = plan_feature_sampling(n, k)
    if plan.arm == SAMPLE_ALL_FEATURES:
        for item in range(len(work_items)):
            for c in range(k):
                colids[item * k + c] = Int32(c)
    elif plan.arm == SAMPLE_EXCESS:
        excess_sample_with_replacement(
            colids,
            work_items,
            tree_ids,
            seed,
            n,
            k,
            plan.n_parallel_samples,
            plan.max_samples_per_thread,
            plan.block_threads,
        )
    else:
        algo_l_sample(colids, work_items, tree_ids, seed, n, k)
    return plan





from std.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv, exp, log
from std.memory import stack_allocation
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceContext
from max.gpu.primitives.block import prefix_sum as block_prefix_sum
from max.gpu.primitives.block import sum as block_sum
from max.gpu.sync import barrier



comptime SAMPLER_UNVISITED: Int32 = -3
"""The seed the CALLER must write into every cell of `report`."""

comptime SAMPLER_OVERRUN: Int32 = -2
"""`EXCESS_MAX_ITERATIONS` was reached."""



comptime SAMP_SAB_NONE: Int32 = 0
"""No sabotage. The shipping path."""

comptime SAMP_SAB_SORT_DESCENDING: Int32 = 1
"""Flip the bitonic comparator direction, so the network sorts DESCENDING."""

comptime SAMP_SAB_NO_DEDUPE: Int32 = 2
"""Skip the adjacent-difference test, so EVERY slot is a head."""

comptime SAMP_SAB_AGG_ONE_PER_THREAD: Int32 = 3
"""Scan ONE item per thread instead of `MAX_SAMPLES_PER_THREAD` of them -- the classic mis-port of a CUB collective that has an `ITEMS_PER_THREAD` template argument."""

comptime SAMP_SAB_KEY_NO_THREAD: Int32 = 4
"""Drop `threadIdx.x` from the fnv1a32 chain (`:168`), so every thread of the block draws the IDENTICAL stream."""

comptime SAMP_SAB_LOOP_ANY_UNIQUE: Int32 = 5
"""`while (n_uniques < k)` (`:241`) becomes `while (n_uniques < 1)`, so the do-while stops after one pass whatever it found."""

comptime SAMP_SAB_CUML_MASK0: Int32 = 6
"""RESTORE cuML's bug 164: compare the block minimum against `mask[0]`, the PREVIOUS iteration's FLAG, instead of giving it the unconditional head flag a minimum has by definition."""

comptime SAMP_SAB_CUML_FILLER: Int32 = 7
"""RESTORE cuML's bug 165: fill a slot past `n_parallel_samples` with `n - 1`, a REAL column id, instead of the out-of-range `NON_DRAWING_SENTINEL`."""

comptime SAMP_SAB_ALL_REVERSED: Int32 = 8
"""`sample_all_features_kernel` writes `k - 1 - c` instead of `c`."""

comptime SAMP_SAB_SMALLEST_K: Int32 = 9
"""cuML's ORIGINAL selection (`:243-246`): when the loop drew more than `k` uniques, keep the `k` smallest COLUMN IDS instead of DEVIATION 215's uniform keyed-hash subset. This is the bug arm: it under-samples high-indexed columns (0.38x at higgs's `(28, 5)`), and the uniformity gate in `sampler_kernel_check` must go red under it -- that is what proves the gate watches the selection distribution and not just the..."""

comptime SAMP_SAB_RAW_SELECTION_HASH: Int32 = 10
"""DEVIATION 464's pre-fix arm: rank the overshoot's uniques by the UN-FINALIZED fnv chain (`excess_selection_hash_raw`) instead of the avalanche-finalized hash."""




def next_pow2(x: Int) -> Int:
    """The smallest power of two `>= x`, and `1` for `x <= 1`."""
    var p = 1
    while p < x:
        p <<= 1
    return p


def excess_sort_span(plan: FeatureSamplerPlan) -> Int:
    """How many slots the bitonic network covers: `next_pow2` of `n_parallel_samples`, NOT of `block_threads * max_samples_per_thread`."""
    return next_pow2(plan.n_parallel_samples)


def excess_scratch_stride(plan: FeatureSamplerPlan) -> Int:
    """Int32s of scratch per BLOCK."""
    var n_slots = plan.block_threads * plan.max_samples_per_thread
    var span = excess_sort_span(plan)
    return n_slots if n_slots > span else span


def sampler_scratch_len(work_items_size: Int, n: Int, k: Int) raises -> Int:
    """Int32s the caller must allocate for `d_scratch`."""
    var plan = plan_feature_sampling(n, k)
    if plan.arm != SAMPLE_EXCESS:
        return 1
    return work_items_size * excess_scratch_stride(plan)


def sampler_report_len(work_items_size: Int) -> Int:
    """Int32s the caller must allocate for `d_report`, and seed with `SAMPLER_UNVISITED`."""
    return 3 * work_items_size


def device_has_float64() -> Bool:
    """Can a kernel on this box hold a `double`? Mojo 1.0 exposes no fp64 capability query, so this is written as the one fact that is true today: of the three GPU targets Mojo supports, Apple is the one whose backend refuses the TYPE."""
    return not has_apple_gpu_accelerator()




def excess_sample_kernel[
    BLOCK_THREADS: Int, MAX_SAMPLES_PER_THREAD: Int
](
    colids: MutPointer[Int32, MutAnyOrigin],
    scratch: MutPointer[Int32, MutAnyOrigin],
    report: MutPointer[Int32, MutAnyOrigin],
    work_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    tree_ids: MutPointer[Int32, MutAnyOrigin],
    work_items_size_in: Int32,
    seed: UInt64,
    n_in: Int32,
    k_in: Int32,
    n_parallel_samples_in: Int32,
    scratch_stride_in: Int32,
    sabotage_in: Int32,
):
    """`excess_sample_with_replacement_kernel`, `builder_kernels.cuh:152-248`, ON THE DEVICE. Given `BLOCK_THREADS` and `MAX_SAMPLES_PER_THREAD` the output is determined."""
    var b = Int(block_idx.x)
    if b >= Int(work_items_size_in):  # `:163`
        return
    var tid = Int(thread_idx.x)
    var sab = sabotage_in
    var n = Int(n_in)
    var k = Int(k_in)
    var nps = Int(n_parallel_samples_in)
    comptime M = MAX_SAMPLES_PER_THREAD
    comptime N_SLOTS = BLOCK_THREADS * MAX_SAMPLES_PER_THREAD

    var node_id = work_items[unsafe_offset=b].idx.cast[DType.uint32]()

    var items = scratch.unsafe_offset(b * Int(scratch_stride_in))

    var tree_u = tree_ids[unsafe_offset=b].cast[DType.uint32]()
    var subsequence: UInt64
    if sab == SAMP_SAB_KEY_NO_THREAD:
        var h = FNV1A32_BASIS
        h = fnv1a32(h, tree_u)
        h = fnv1a32(h, node_id)
        subsequence = h.cast[DType.uint64]()
    else:
        subsequence = excess_subsequence(UInt32(tid), tree_u, node_id)
    var gen = PCGenerator(seed, subsequence, UInt64(0))

    var mask = stack_allocation[M, Scalar[DType.int32]]()
    for l in range(M):
        mask[unsafe_offset=l] = Int32(0)

    var span = 1
    while span < nps:
        span <<= 1

    var pad = N_SLOTS + tid
    while pad < span:
        items[unsafe_offset=pad] = NON_DRAWING_SENTINEL(n)
        pad += BLOCK_THREADS

    var n_uniques = 0
    var thread_prefix = 0
    var iters = 0
    var overrun = False

    while True:  # `:188` .. `:241`, a do-while
        iters += 1
        if iters > EXCESS_MAX_ITERATIONS:
            overrun = True
            break

        for l in range(M):
            var slot = M * tid + l
            if mask[unsafe_offset=l] == Int32(0) and slot < nps:
                items[unsafe_offset=slot] = Int32(
                    uniform_int_u64(gen, 0, UInt64(n))
                )
            elif mask[unsafe_offset=l] == Int32(0):
                if sab == SAMP_SAB_CUML_FILLER:
                    items[unsafe_offset=slot] = Int32(n - 1)  # their `:201-203`
                else:
                    items[unsafe_offset=slot] = NON_DRAWING_SENTINEL(n)
        barrier()

        var kk = 2
        while kk <= span:
            var j = kk >> 1
            while j > 0:
                var i = tid
                while i < span:
                    var partner = i ^ j
                    if partner > i:
                        var a = items[unsafe_offset=i]
                        var c = items[unsafe_offset=partner]
                        var up = (i & kk) == 0
                        if sab == SAMP_SAB_SORT_DESCENDING:
                            up = not up
                        var swap = (a > c) if up else (a < c)
                        if swap:
                            items[unsafe_offset=i] = c
                            items[unsafe_offset=partner] = a
                    i += BLOCK_THREADS
                barrier()
                j >>= 1
            kk <<= 1

        var local_sum = Int32(0)
        for l in range(M):
            var slot = M * tid + l
            var here = items[unsafe_offset=slot]
            var m: Int32
            if slot == 0:
                if sab == SAMP_SAB_CUML_MASK0:
                    m = (
                        Int32(0) if here
                        == mask[unsafe_offset=0] else Int32(1)
                    )
                else:
                    m = Int32(1)
            else:
                var prev = items[unsafe_offset = slot - 1]
                m = Int32(0) if here == prev else Int32(1)
            if sab == SAMP_SAB_NO_DEDUPE:
                m = Int32(1)
            if here >= Int32(n):
                m = Int32(0)
            mask[unsafe_offset=l] = m
            local_sum += m
        barrier()

        var scan_in = local_sum
        if sab == SAMP_SAB_AGG_ONE_PER_THREAD:
            scan_in = mask[unsafe_offset=0]
        thread_prefix = Int(
            block_prefix_sum[block_size=BLOCK_THREADS, exclusive=True](scan_in)
        )
        barrier()
        n_uniques = Int(block_sum[block_size=BLOCK_THREADS](scan_in))
        barrier()

        var need = k  # `:241`, `while (n_uniques < k)`
        if sab == SAMP_SAB_LOOP_ANY_UNIQUE:
            need = 1
        if n_uniques >= need:
            break

    if not overrun:
        var col_offset = k * b
        if n_uniques <= k or sab == SAMP_SAB_SMALLEST_K:
            var running = Int32(thread_prefix)
            for l in range(M):
                var slot = M * tid + l
                var ci = running
                running += mask[unsafe_offset=l]
                if mask[unsafe_offset=l] != Int32(0) and Int(ci) < k:
                    colids[unsafe_offset = col_offset + Int(ci)] = items[
                        unsafe_offset=slot
                    ]
        else:
            var raw_hash = sab == SAMP_SAB_RAW_SELECTION_HASH
            for l in range(M):
                var slot = M * tid + l
                if mask[unsafe_offset=l] == Int32(0):
                    continue
                var col = items[unsafe_offset=slot]
                var h: UInt32
                if raw_hash:
                    h = excess_selection_hash_raw(
                        tree_u, node_id, col.cast[DType.uint32]()
                    )
                else:
                    h = excess_selection_hash(
                        tree_u, node_id, col.cast[DType.uint32]()
                    )
                var rank = 0
                for s in range(N_SLOTS):
                    var c2 = items[unsafe_offset=s]
                    if c2 >= Int32(n):
                        continue
                    var is_head: Bool
                    if s == 0:
                        is_head = True
                    else:
                        is_head = c2 != items[unsafe_offset = s - 1]
                    if not is_head:
                        continue
                    var h2: UInt32
                    if raw_hash:
                        h2 = excess_selection_hash_raw(
                            tree_u, node_id, c2.cast[DType.uint32]()
                        )
                    else:
                        h2 = excess_selection_hash(
                            tree_u, node_id, c2.cast[DType.uint32]()
                        )
                    if h2 < h or (h2 == h and c2 < col):
                        rank += 1
                if rank < k:
                    colids[unsafe_offset = col_offset + rank] = col

    if tid == 0:
        report[unsafe_offset = 3 * b + 0] = Int32(SAMPLE_EXCESS)
        report[unsafe_offset = 3 * b + 1] = Int32(M)
        report[unsafe_offset = 3 * b + 2] = (
            SAMPLER_OVERRUN if overrun else Int32(iters)
        )




def sample_all_features_kernel(
    colids: MutPointer[Int32, MutAnyOrigin],
    report: MutPointer[Int32, MutAnyOrigin],
    work_items_size_in: Int32,
    k_in: Int32,
    sabotage_in: Int32,
):
    """`k == n`: every column is a candidate."""
    var k = Int(k_in)
    var items_size = Int(work_items_size_in)
    var g = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if g < items_size * k:
        var c = g % k
        if sabotage_in == SAMP_SAB_ALL_REVERSED:
            c = k - 1 - c
        colids[unsafe_offset=g] = Int32(c)
    if g < items_size:
        report[unsafe_offset = 3 * g + 0] = Int32(SAMPLE_ALL_FEATURES)
        report[unsafe_offset = 3 * g + 1] = Int32(0)
        report[unsafe_offset = 3 * g + 2] = Int32(0)






def _dev_logf32(x: Float32) -> Float32:
    """`raft::log(float)`."""
    return identical_log(x)


def _dev_expf32(x: Float32) -> Float32:
    """`raft::exp(float)`."""
    return identical_exp(x)


def _dev_log64(x: Float64) -> Float64:
    """`raft::log(double)`. `_log64`'s device twin, and the one the Metal backend refuses."""
    return log(x)



def algo_l_sample_kernel[
    BLOCK_THREADS: Int
](
    colids: MutPointer[Int32, MutAnyOrigin],
    report: MutPointer[Int32, MutAnyOrigin],
    work_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    tree_ids: MutPointer[Int32, MutAnyOrigin],
    work_items_size_in: Int32,
    seed: UInt64,
    n_in: Int32,
    k_in: Int32,
    sabotage_in: Int32,
):
    """`algo_L_sample_kernel`, `builder_kernels.cuh:268-316`, ON THE DEVICE. DEVIATION 199.** It carries `W` as a `double` because cuML does, and the Metal backend refuses the type outright -- not the operation, the TYPE."""
    var tid = Int(block_idx.x) * BLOCK_THREADS + Int(thread_idx.x)
    if tid >= Int(work_items_size_in):  # `:278`
        return
    var n = Int(n_in)
    var k = Int(k_in)

    var node_id = work_items[unsafe_offset=tid].idx.cast[DType.uint32]()  # `:279`
    var gen = PCGenerator(
        seed,
        algo_l_subsequence(
            tree_ids[unsafe_offset=tid].cast[DType.uint32](), node_id
        ),
        UInt64(0),
    )  # `:280-281`

    var fp_uniform_val = gen.next_float()
    var W = Float64(_dev_expf32(_dev_logf32(fp_uniform_val) / Float32(k)))

    var col = 0
    while True:
        colids[unsafe_offset = tid * k + col] = Int32(col)
        if col == k - 1:
            break
        else:
            col += 1

    while col < n:
        fp_uniform_val = gen.next_float()
        col += (
            Int(Float64(_dev_logf32(fp_uniform_val)) / _dev_log64(1.0 - W)) + 1
        )
        if col < n:
            var int_uniform_val = Int(uniform_int_u64(gen, 0, UInt64(k)))
            colids[unsafe_offset = tid * k + int_uniform_val] = Int32(col)
            fp_uniform_val = gen.next_float()
            W *= Float64(_dev_expf32(_dev_logf32(fp_uniform_val) / Float32(k)))

    report[unsafe_offset = 3 * tid + 0] = Int32(SAMPLE_ALGO_L)
    report[unsafe_offset = 3 * tid + 1] = Int32(0)
    report[unsafe_offset = 3 * tid + 2] = Int32(0)




def sample_features_device[
    oc: MutOrigin, os: MutOrigin, orp: MutOrigin, ow: MutOrigin, ot: MutOrigin, //
](
    ctx: DeviceContext,
    d_colids: MutPointer[Int32, oc],
    d_scratch: MutPointer[Int32, os],
    d_report: MutPointer[Int32, orp],
    d_work_items: MutPointer[NodeWorkItem, ow],
    d_tree_ids: MutPointer[Int32, ot],
    work_items_size: Int,
    seed: UInt64,
    n: Int,
    k: Int,
    sabotage: Int32 = SAMP_SAB_NONE,
) raises -> FeatureSamplerPlan:
    """`sample_features`, but the sampling happens where cuML does it. Same contract as the host driver and the same returned plan, so a caller can name the arm that ran (rule 8) without reading `d_report` -- though `d_report` is the DEVICE's own statement of the same thing and is what a check must believe."""
    var plan = plan_feature_sampling(n, k)
    if plan.arm == SAMPLE_ALL_FEATURES:
        comptime TPB = 128
        var total = work_items_size * k
        ctx.enqueue_function[sample_all_features_kernel](
            d_colids,
            d_report,
            Int32(work_items_size),
            Int32(k),
            sabotage,
            grid_dim=ceildiv(total, TPB),
            block_dim=TPB,
        )
        return plan
    if plan.arm == SAMPLE_EXCESS:
        var stride = excess_scratch_stride(plan)
        if plan.max_samples_per_thread == 1:
            ctx.enqueue_function[
                excess_sample_kernel[SAMPLER_BLOCK_THREADS, 1]
            ](
                d_colids,
                d_scratch,
                d_report,
                d_work_items,
                d_tree_ids,
                Int32(work_items_size),
                seed,
                Int32(n),
                Int32(k),
                Int32(plan.n_parallel_samples),
                Int32(stride),
                sabotage,
                grid_dim=(work_items_size, 1, 1),
                block_dim=(SAMPLER_BLOCK_THREADS, 1, 1),
            )
        else:
            ctx.enqueue_function[
                excess_sample_kernel[
                    SAMPLER_BLOCK_THREADS, SAMPLER_MAX_SAMPLES_PER_THREAD
                ]
            ](
                d_colids,
                d_scratch,
                d_report,
                d_work_items,
                d_tree_ids,
                Int32(work_items_size),
                seed,
                Int32(n),
                Int32(k),
                Int32(plan.n_parallel_samples),
                Int32(stride),
                sabotage,
                grid_dim=(work_items_size, 1, 1),
                block_dim=(SAMPLER_BLOCK_THREADS, 1, 1),
            )
        return plan

    comptime if not device_has_float64():
        raise Error(
            String(
                "the algo-L reservoir arm has no device path on this target:"
                " cuML's algo_L_sample_kernel carries W, log(1 - W) and the"
                " truncated jump in DOUBLE (builder_kernels.cuh:291, :306),"
                " and this backend refuses the type. Refused by name rather"
                " than silently substituted in float32, which at k/n=",
                Float64(k) / Float64(n),
                " loses 1 - W to cancellation. DEVIATION 199. n=",
                n,
                " k=",
                k,
            )
        )
    else:
        ctx.enqueue_function[algo_l_sample_kernel[SAMPLER_BLOCK_THREADS]](
            d_colids,
            d_report,
            d_work_items,
            d_tree_ids,
            Int32(work_items_size),
            seed,
            Int32(n),
            Int32(k),
            sabotage,
            grid_dim=ceildiv(work_items_size, SAMPLER_BLOCK_THREADS),
            block_dim=SAMPLER_BLOCK_THREADS,
        )
    return plan
