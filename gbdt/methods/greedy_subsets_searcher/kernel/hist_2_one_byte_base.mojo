# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# This lane MIRRORS CatBoost (pinned commit 54a8143a). Per-file provenance is in gbdt/DERIVATION_MAP.tsv, in this file's own docstring, and in NOTICE; files with no CatBoost counterpart say so.
"""The fused TWO-STAT one-byte histogram: both stat columns in one pass.

PORT OF `hist_2_one_byte_base.cuh` (`TPointHist2OneByteBase`, the
`ComputeHist2OneByteBits` launch plumbing) and the two-stat kernel loop it
runs under, `compute_hist_loop_two_stats.cuh`
(`ComputeSplitPropertiesDirectLoadsTwoStastImpl`,
`ComputeSplitPropertiesTwoStatsGatherImpl`, `TComputeHistogramTwoStatsImpl`
and its `AlignMemoryAccess`), at CatBoost `54a8143a`. Transliterated. Do not
improve.

The per-bit accumulators live beside this file exactly as theirs do:
`hist_2_one_byte_5bit.mojo`, `_6bit.mojo`, `_7bit.mojo` mirror their
`hist_2_one_byte_{5,6,7}bit.cu`. Their CRTP (`TImpl* impl =
static_cast<TImpl*>(this)`) becomes a comptime `bits` dispatch in the three
`hist2_*` helpers below, which is the same static resolution spelled the way
Mojo can say it. The two-stat loop file is INLINED into the two kernels here
rather than kept as a separate module, exactly as the PASS family inlines
`compute_hist_loop_one_stat.cuh` into `hist_one_byte.mojo` (PORTING.md 10
and 13 record why the loop cannot be a freestanding function yet).

WHY THIS FAMILY EXISTS, AND WHEN CATBOOST TAKES IT
--------------------------------------------------
`ComputeHistOneByte` selects by `maxBins` (`hist_one_byte.cu:314-328`):

    maxBins <= 32   HIST2_PASS(5)
    maxBins <= 64   HIST2_PASS(6)
    maxBins <= 128  HIST2_PASS(7)    <- their own GPU default border count
    maxBins <= 255  PASS(8, numStats)

so for every one-byte shape up to 128 bins their dispatch runs THIS family,
which processes stat columns TWO AT A TIME (`numBlocks.z = numStats / 2`,
each block reading `stats` and `stats + statLineSize`), and the
`TPointHistOneByte` PASS family only above 128. Until 2026-08-19 this
repository routed everything through the PASS family, which is the
wrong-kernel-family misport PORTING_RULES 0b-i describes. When `numStats` is
odd, their `HIST2_PASS` macro first covers stat 0 with a one-stat
`PASS(Bits, 1)` launch and then runs this family over the remaining even
count with `SkipFirst = true` (`hist_one_byte.cu:306-312`); the `skip_first`
comptime parameter below is that template bool.

DEVIATION (PORTING.md 1): CatBoost runs this at `BlockSize = 384`
(`hist_2_one_byte_base.cuh:169`), so `384 * 32` floats is 49,152 bytes and
Apple gives 32,768. The matrix row `K_HIST_2_ONE_BYTE` resolves the block to
256, which asks for exactly 32,768 bytes and keeps their per-warp slice
arithmetic intact (8 warps times 1024 floats). The BLOCK shrinks, the
LAYOUT does not: every 1024-float warp slice, the 2048-float combined-hist
offset and the 256-thread reduce/writeback stages are theirs verbatim, and
all of them fit under a 256-thread block because their stages already cap
participation at `threadIdx.x < 256`.

DEVIATION (load batch): their `Unroll` is 1 below Volta and 2 at or above
(`hist_2_one_byte_base.cuh:28-35`), their `LoadSize` is `FourElements` on
everything after Maxwell (`:41-47`), and `TLoadSizeHist2<FourElements>` is 8
at or above Volta (`tuning_policy_enums.cuh:73-80`). This port takes the
MODERN side of each arch test, the same choice `hist_one_byte.mojo` records
for the PASS family: UNROLL 2, LOAD 4, batch 8. Scheduling, not numeric: the
same values are added in the same per-lane order.

DEVIATION (the `hist_smem_mode_for` NUMERIC row, 2026-08-19): on the APPLE
column (and under `NUMERIC_IDENTICAL` on every column) the shared-memory
accumulation is NOT theirs. Their warp-private float slices cap Apple at 256
resident threads/core inside Metal's 32 KB threadgroup ceiling -- 12.5%
occupancy by construction. The Apple arm shares each 1024-slot slice between
TWO warps and accumulates with LOCAL Int32 atomics in fixed point (Metal has
no local float atomics), halving shared memory per thread so the block
doubles to 512 at the same 32 KB. Measured basis (scratchpad
`histshare_probe.mojo`, interleaved): warp-private float 32 KB = 46.5 G
updates/s, 2-warp-shared Int32 = 90.2 G upd/s, 1.94x. Whole-tree, this file,
interleaved in one process at 800k x 100 x 128 folds, depth 6, identical
trees both arms: 43.5-48.9 ms/tree float, 32.3-35.9 ms/tree Int32,
1.33-1.51x. Pass structure, bin decoding and
slice-offset arithmetic stay theirs; only WHERE the add lands (shared pair
slice, atomic Int32) and the block size change. The mode branches are
factored into `hist2_smem_add` + `hist2_quantize` (histogram_utils.mojo,
whose docstrings carry the dither derivation and the two measured failure
modes of simpler quantizers), the pair-granularity line of
`hist2_slice_offset`, the guarded quantize-at-load lines of the two
kernels, and the Int32 arm of `hist2_add_to_global_memory`; everything
between them is mode-blind. The NVIDIA/AMD FAST columns compile their
design unchanged. Because integer addition is associative and the dither is
a pure function of the data position, the Apple arm's histogram is
DETERMINISTIC run to run, which CatBoost's own float path is not.
"""

from std.atomic import Atomic, Ordering
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.gpu.intrinsics import ldg
from std.memory import stack_allocation

from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from original.kernel_matrix import (
    HIST_SMEM_SHARED2_I32,
    K_HIST_2_ONE_BYTE,
    TARGET_COLUMN,
    deterministic_flush_for,
    hist2_block_size_for,
    hist_floats_per_thread_for,
    hist_smem_mode_for,
    lane_width_for,
    requires_uniform_iteration_for,
)
from original.numerics import PIN_DETERMINISM
from original.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL

from gbdt.methods.greedy_subsets_searcher.kernel.histogram_utils import (
    hist2_dither,
    hist2_quantize,
)
from gbdt.methods.greedy_subsets_searcher.kernel.hist_2_one_byte_5bit import (
    hist2_add_points_5,
    hist2_reduce_tail_5,
    hist2_slice_offset_5,
)
from gbdt.methods.greedy_subsets_searcher.kernel.hist_2_one_byte_6bit import (
    hist2_add_points_6,
    hist2_reduce_tail_6,
    hist2_slice_offset_6,
)
from gbdt.methods.greedy_subsets_searcher.kernel.hist_2_one_byte_7bit import (
    hist2_add_points_7,
    hist2_reduce_tail_7,
    hist2_slice_offset_7,
)


#: Same build mode as the other histogram kernels; the flush follows the
#: matrix.
comptime BUILD_MODE = GLOBAL_NUMERIC_MODE

#: Lanes moving in lockstep. READ FROM THE MATRIX, not pinned here.
comptime LANE_WIDTH = lane_width_for[
    TARGET_COLUMN, BUILD_MODE == NUMERIC_IDENTICAL
]()

#: THE ACCUMULATION-MODE ROW, read from the matrix once. Warp-private float
#: (theirs) on NVIDIA/AMD FAST; 2-warp-shared Int32 fixed point on Apple and
#: under `NUMERIC_IDENTICAL` everywhere. See the module docstring's last
#: DEVIATION and `kernel_matrix.hist_smem_mode_for` for the probe numbers.
comptime HIST2_SMEM_MODE = hist_smem_mode_for[
    TARGET_COLUMN, BUILD_MODE == NUMERIC_IDENTICAL
]()

#: Exported so `doc_parallel_boosting.fit` and the launch helper know the
#: build needs real fixed-point magnitudes and the Int32->float bridge.
comptime HIST2_SMEM_IS_I32 = HIST2_SMEM_MODE == HIST_SMEM_SHARED2_I32


def hist2_block_size[smem_mode: Int]() -> Int:
    """READ FROM THE MATRIX, per accumulation mode. Theirs is 384
    (`hist_2_one_byte_base.cuh:169`); Apple's 32 KB over 32 floats per
    thread yields 256 warp-private, 512 under the 2-warp-shared Int32 row.
    See the module docstring's DEVIATIONs."""
    return hist2_block_size_for[TARGET_COLUMN, smem_mode]()


def hist2_acc_dtype[smem_mode: Int]() -> DType:
    """The shared accumulator's element type: theirs is float, the shared
    slice variant is Int32 fixed point (`hist2_smem_add`)."""

    @parameter
    if smem_mode == HIST_SMEM_SHARED2_I32:
        return DType.int32
    else:
        return DType.float32


def hist2_smem_slots[smem_mode: Int]() -> Int:
    """`GetHistSize()` = `BlockSize * 32` (`hist_2_one_byte_base.cuh:20-22`)
    in their design, i.e. one 1024-slot slice per warp. The shared-Int32
    variant keeps 1024-slot slices but hands one to each warp PAIR, so it is
    half as many slices at the doubled block: the same 8192 slots at 512
    threads, or 4096 at 256."""
    comptime block = hist2_block_size[smem_mode]()

    @parameter
    if smem_mode == HIST_SMEM_SHARED2_I32:
        return (block // (2 * LANE_WIDTH)) * 1024
    else:
        return block * hist_floats_per_thread_for[K_HIST_2_ONE_BYTE]()


#: The default-mode values, kept under their old names for readers and for
#: `hist2_check`. Everything inside the kernels reads the mode-parameterized
#: functions instead, so both modes can coexist in one binary.
comptime HIST2_BLOCK_SIZE = hist2_block_size[HIST2_SMEM_MODE]()
comptime HIST2_HIST_SIZE = hist2_smem_slots[HIST2_SMEM_MODE]()

#: `Unroll(ECIndexLoadType)` (`hist_2_one_byte_base.cuh:28-35`): 1 below
#: Volta, 2 at or above. The modern side, per the module docstring.
comptime HIST2_UNROLL = 2

#: `ELoadSize::FourElements` (`hist_2_one_byte_base.cuh:41-47`).
comptime HIST2_LOAD_SIZE = 4

#: `AddPointsBatchSize()` = `TLoadSizeHist2<FourElements>::Size()`
#: (`hist_2_one_byte_base.cuh:24-26`, `tuning_policy_enums.cuh:73-80`): 4
#: below Volta, 8 at or above. Equal to `loadSize * Unroll` here, so their
#: `AddPoints<N>` loop over batches (`hist_2_one_byte_base.cuh:68-78`)
#: degenerates to ONE `AddPointsImpl<8>` call per iteration.
comptime HIST2_ADD_POINTS_BATCH = 8

#: `loadSize * N`, the points one thread takes per iteration.
comptime HIST2_POINTS_PER_ITER = HIST2_LOAD_SIZE * HIST2_UNROLL

def hist2_min_docs[smem_mode: Int]() -> Int:
    """`BlockLoadSize(indexLoadType)` = `TLoadSizeHist2<LoadSize()>::Size()
    * BlockSize * Unroll(...)` (`hist_2_one_byte_base.cuh:49-51`). It is what
    decides `activeBlockCount`, and it is NOT `loadSize * BlockSize * Unroll`
    as the PASS family's is: the hist_2 batch size doubles it. Their formula
    scales with the block, so the shared-Int32 mode's doubled block doubles
    it too."""
    return HIST2_ADD_POINTS_BATCH * hist2_block_size[smem_mode]() * HIST2_UNROLL


#: The default-mode value, kept under its old name for `hist2_check`.
comptime HIST2_MIN_DOCS_PER_BLOCK = hist2_min_docs[HIST2_SMEM_MODE]()


def hist2_slice_offset[bits: Int, smem_mode: Int](tid: Int) -> Int:
    """`TImpl::SliceOffset()`, resolved by `bits` the way their CRTP does.

    THE 1024 IS THEIRS AND IT IS 32-LANE: 32 lanes times the 32 floats per
    thread this accumulator takes. A 64-lane wavefront needs a 2048-float
    slice and CatBoost never wrote that layout, so the assert makes it a
    compile error rather than two warps quietly sharing one private copy --
    the same guard `hist_one_byte.mojo` carries. A 64-lane column never
    instantiates this family: `greedy_one_byte_fixed_for` (DEVIATION 1906)
    routes its one-byte work to the fused 8-bit kernel at the dispatch, and
    the assert stays for whatever reaches this file some other way.

    ================= DEVIATION BLOCK =================
    Under `HIST_SMEM_SHARED2_I32` the slice a thread lands in is keyed by its
    warp PAIR (`tid // 64`) instead of its warp (`tid // 32`): two warps
    deliberately share one 1024-slot slice through the Int32 atomics of
    `hist2_smem_add`, which is the measured 1.94x of the matrix row
    (46.5 -> 90.2 G upd/s, scratchpad `histshare_probe.mojo`). The
    within-slice arithmetic -- `innerHistStart`, feature/bin/parity slots --
    is theirs unchanged, so the substitution below rewrites ONLY the
    warp-offset term of their `SliceOffset`.
    ===================================================
    """
    comptime assert LANE_WIDTH == 32, (
        "the hist_2 accumulator's slice layout is 32-lane by construction"
        " (`hist_2_one_byte_{5,6,7}bit.cu` SliceOffset, and ReduceToOneWarp's"
        " warpHistSize at `hist_2_one_byte_base.cuh:92`); write the"
        " wide-wavefront layout before letting LANE_WIDTH be 64"
    )

    var off: Int

    @parameter
    if bits == 5:
        off = hist2_slice_offset_5(tid)
    elif bits == 6:
        off = hist2_slice_offset_6(tid)
    else:
        off = hist2_slice_offset_7(tid)

    @parameter
    if smem_mode == HIST_SMEM_SHARED2_I32:
        # `1024 * (tid / 32)` becomes `1024 * (tid / 64)`; nothing else moves.
        return off - 1024 * (tid // 32) + 1024 * (tid // 64)
    else:
        return off


def hist2_add_points[
    bits: Int, n: Int, dt: DType
](
    ci: InlineArray[UInt32, n],
    s1: InlineArray[Float32, n],
    s2: InlineArray[Float32, n],
    q1: InlineArray[Int32, n],
    q2: InlineArray[Int32, n],
    tid: Int,
    slice_base: Int,
    smem: UnsafePointer[
        Scalar[dt],
        address_space = AddressSpace.SHARED,
        origin=MutUntrackedOrigin,
    ],
):
    """`TImpl::AddPointsImpl<N>`, resolved by `bits`. At `n == 1` this is
    their `AddPoint` (`hist_2_one_byte_base.cuh:80-85`), which is what the
    head/tail peel calls. `dt` carries the accumulation mode and `q1`/`q2`
    the pre-quantized stats (`hist2_quantize` at the load sites); the
    per-bit bodies are mode-blind except the one `hist2_smem_add` call."""

    @parameter
    if bits == 5:
        hist2_add_points_5[n, dt](ci, s1, s2, q1, q2, tid, slice_base, smem)
    elif bits == 6:
        hist2_add_points_6[n, dt](ci, s1, s2, q1, q2, tid, slice_base, smem)
    else:
        hist2_add_points_7[n, dt](ci, s1, s2, q1, q2, tid, slice_base, smem)


def hist2_reduce_to_one_warp[
    dt: DType, smem_mode: Int
](
    tid: Int,
    smem: UnsafePointer[
        Scalar[dt],
        address_space = AddressSpace.SHARED,
        origin=MutUntrackedOrigin,
    ],
):
    """`TPointHist2OneByteBase::ReduceToOneWarp`
    (`hist_2_one_byte_base.cuh:87-104`), copied: fold the per-warp 1024-float
    copies down to one, parked at `smem[2048 + start]`.

        const int warpHistSize = 1024;
        for (int start = threadIdx.x; start < warpHistSize; start += BlockSize) {
            float sum = 0;
            for (int i = start; i < 32 * BlockSize; i += warpHistSize)
                sum += Histogram[i];
            Histogram[2048 + start] = sum;
        }

    Stride 1024 means one thread owns each residue class, so the write at
    `2048 + start` is by the same thread that already read that slot as part
    of its own sum. That is why their loop needs no barrier inside; the two
    `__syncthreads()` bracketing it are theirs.

    Their `Histogram -= impl->SliceOffset()` before the loop is the CRTP way
    of getting back to the raw buffer base; here the raw `smem` is passed
    directly, which is the same pointer.

    Mode-blind by construction: the loop folds however many 1024-slot slices
    exist (8 warp-private at 256 threads, or `block / 64` pair-shared under
    `HIST_SMEM_SHARED2_I32`), in `dt`'s own arithmetic -- Int32 sums are
    exact and associative, float sums are their order exactly. The
    residue-class ownership argument above holds at any block size at or
    above 256: `2048 + start` has residue `start` mod 1024, so only the
    thread that owns that residue ever touches it.
    """
    barrier()
    comptime WARP_HIST_SIZE = 1024
    comptime SLOTS = hist2_smem_slots[smem_mode]()
    comptime BLOCK = hist2_block_size[smem_mode]()
    var start = tid
    while start < WARP_HIST_SIZE:
        var acc = Scalar[dt](0)
        var i = start
        while i < SLOTS:
            acc += smem[i]
            i += WARP_HIST_SIZE
        smem[2048 + start] = acc
        start += BLOCK
    barrier()


def hist2_reduce[bits: Int, dt: DType, smem_mode: Int](
    tid: Int,
    smem: UnsafePointer[
        Scalar[dt],
        address_space = AddressSpace.SHARED,
        origin=MutUntrackedOrigin,
    ],
):
    """`TImpl::Reduce()`: `TParent::ReduceToOneWarp()` then the per-bit tail,
    then the tail's own trailing `__syncthreads()` (each `Reduce` ends with
    one; the tails leave it to this caller so it exists exactly once)."""
    hist2_reduce_to_one_warp[dt, smem_mode](tid, smem)

    @parameter
    if bits == 5:
        hist2_reduce_tail_5[dt](tid, smem)
    elif bits == 6:
        hist2_reduce_tail_6[dt](tid, smem)
    else:
        hist2_reduce_tail_7[dt](tid, smem)
    barrier()


def hist2_add_to_global_memory[
    bits: Int, dt: DType
](
    stat_id: Int,
    stat_count: Int,
    block_count: Int,
    feature_folds: MutPointer[UInt32, MutAnyOrigin],
    feature_fold_offset: MutPointer[UInt32, MutAnyOrigin],
    feature_group_offset: MutPointer[UInt32, MutAnyOrigin],
    feature_group_size: MutPointer[UInt32, MutAnyOrigin],
    feature_offset: Int,
    f_count: Int,
    leaf_id: Int,
    leaf_count: Int,
    bin_sums: MutPointer[Float32, MutAnyOrigin],
    acc_i32: MutPointer[Int32, MutAnyOrigin],
    fixed_scale: Float32,
    tid: Int,
    smem: UnsafePointer[
        Scalar[dt],
        address_space = AddressSpace.SHARED,
        origin=MutUntrackedOrigin,
    ],
):
    """`TPointHist2OneByteBase::AddToGlobalMemory`
    (`hist_2_one_byte_base.cuh:117-145`), copied. TWO stats leave in one
    call: threads 128..255 carry `isSecondStatFlag` and write `statId + 1`'s
    plane, threads 0..127 write `statId`'s, 32 fold-lanes per feature.

        const int isSecondStatFlag = threadIdx.x >= 128;
        const int fid = (threadIdx.x & 127) / 32;
        const int firstFoldIdx = threadIdx.x & 31;
        const int histSize = 1 << TImpl::MaxBits();
        ...
        for (int fold = firstFoldIdx; fold < features[fid].Folds; fold += 32)
            val = Histogram[isSecondStatFlag * 4 * histSize + fid * histSize + fold];

    `DstOffset` (`:107-115`) is the same expression the PASS family's
    writeback uses; `leaf_id` is the DENSE `blockIdx.y` and `leaf_count` is
    the caller's `max_leaves`, both exactly as `hist_one_byte.mojo`'s
    writeback takes them.

    THE FLUSH is their multi-block branch (`:135-141`): `atomicAdd` when
    `blockCount > 1`, a plain store otherwise. Wired through BOTH modes the
    way `hist_one_byte.mojo` wires it: `NUMERIC_FAST` takes CatBoost's float
    atomic verbatim; `NUMERIC_IDENTICAL` sends the replicated flush through
    the Int32 accumulator instead, because integer addition is associative
    and the histogram then does not depend on which block lands first.

    ================= DEVIATION BLOCK =================
    Under `HIST_SMEM_SHARED2_I32` (`dt == int32`) the shared histogram
    already holds fixed-point Int32 at `fixed_scale`, so the flush changes
    shape: the multi-block branch adds the Int32 cell DIRECTLY into the
    accumulator (exact, no dequantize/requantize round trip), and the
    single-block branch stores the dequantized `Float32(cell) / fixed_scale`.
    This holds in BOTH numeric modes -- routing the Apple FAST arm's
    replicated flush through a float atomic instead would re-lose the
    run-to-run determinism the Int32 accumulation just bought, for no
    measured gain. The launch helper runs `fixed_to_float_kernel` on hist_2
    blocks whenever this arm is compiled, so the accumulator drains under
    FAST exactly as it does under IDENTICAL. Their `abs(val) > 1e-20f` guard
    becomes `cell != 0`, the same never-write-a-zero-cell contract in the
    integer domain.
    ===================================================
    """
    comptime hist_size = 1 << bits

    if tid < 256:
        var is_second_stat = 0
        if tid >= 128:
            is_second_stat = 1
        var fid = (tid & 127) // 32
        var first_fold_idx = tid & 31

        if fid < f_count:
            var folds = Int(feature_folds.unsafe_load(feature_offset + fid))
            var group_offset = Int(
                feature_group_offset.unsafe_load(feature_offset + fid)
            )
            var group_size = Int(
                feature_group_size.unsafe_load(feature_offset + fid)
            )
            var fold_off = Int(
                feature_fold_offset.unsafe_load(feature_offset + fid)
            )

            # `DstOffset(statId + isSecondStatFlag, statCount, group, ...)`
            var device_offset = group_offset * stat_count * leaf_count
            var entries_per_leaf = stat_count * group_size
            var dst_base = (
                device_offset
                + leaf_id * entries_per_leaf
                + (stat_id + is_second_stat) * group_size
                + fold_off
            )
            var dst = bin_sums + dst_base

            var fold = first_fold_idx
            while fold < folds:
                var cell = smem[
                    is_second_stat * 4 * hist_size + fid * hist_size + fold
                ]

                @parameter
                if dt == DType.int32:
                    # The shared-Int32 arm: the cell is already fixed point
                    # at `fixed_scale`. See the DEVIATION BLOCK above.
                    var q = rebind[Scalar[DType.int32]](cell)
                    if q != Int32(0):
                        if block_count > 1:
                            # DEVIATION 1898: upstream's atomicAdd is relaxed;
                            # the non-Apple Mojo default is seq_cst.
                            _ = Atomic.fetch_add[ordering = Ordering.RELAXED](
                                acc_i32.unsafe_offset(dst_base + fold), q
                            )
                        else:
                            # SINGLE BLOCK GOES THROUGH THE ACCUMULATOR
                            # TOO (plain store: one grid block owns the
                            # cell). `q` is already fixed point, so the
                            # fused writeback dequantizes the IDENTICAL
                            # bits the store here used to; what it buys
                            # is a DEAD float scratch on this arm -- no
                            # store, no fallback read, no per-launch
                            # scratch memset. Binary and half-byte do
                            # NOT take this shape: their cells are exact
                            # floats and routing them through Int32
                            # would add a round trip that moves bits.
                            acc_i32.unsafe_store(dst_base + fold, q)
                else:
                    var val = rebind[Scalar[DType.float32]](cell)
                    if abs(val) > Float32(1e-20):
                        comptime det = deterministic_flush_for[
                            TARGET_COLUMN, PIN_DETERMINISM
                        ]()

                        @parameter
                        if det:
                            if block_count > 1:
                                # `NUMERIC_IDENTICAL`: partials sum as Int32.
                                var q = Int32(val * fixed_scale)
                                # DEVIATION 1898: upstream's atomicAdd is
                                # relaxed; the non-Apple Mojo default is
                                # seq_cst.
                                _ = Atomic.fetch_add[
                                    ordering = Ordering.RELAXED
                                ](
                                    acc_i32.unsafe_offset(dst_base + fold), q
                                )
                            else:
                                dst.unsafe_store(fold, val)
                        else:
                            # `atomicAdd(dst + fold, val)`, theirs verbatim
                            # (`hist_2_one_byte_base.cuh:137`).
                            if block_count > 1:
                                # DEVIATION 1898: upstream's atomicAdd is
                                # relaxed; the non-Apple Mojo default is
                                # seq_cst.
                                _ = Atomic.fetch_add[
                                    ordering = Ordering.RELAXED
                                ](
                                    dst.unsafe_offset(fold), val
                                )
                            else:
                                dst.unsafe_store(fold, val)
                fold += 32


def hist2_one_byte_kernel[bits: Int, skip_first: Bool, smem_mode: Int](
    # `TFeatureInBlock*`, flattened to four parallel arrays so the kernel
    # takes plain pointers, exactly as the PASS family's kernels do.
    feature_folds: MutPointer[UInt32, MutAnyOrigin],
    feature_fold_offset: MutPointer[UInt32, MutAnyOrigin],
    feature_group_offset: MutPointer[UInt32, MutAnyOrigin],
    feature_group_size: MutPointer[UInt32, MutAnyOrigin],
    f_count_in32: Int32,
    bins: MutPointer[UInt32, MutAnyOrigin],
    bins_line_size_in: Int32,
    cindex_base_in: Int32,
    stats: MutPointer[Float32, MutAnyOrigin],
    stat_line_size_in: Int32,
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    part_ids: MutPointer[UInt32, MutAnyOrigin],
    bin_sums: MutPointer[Float32, MutAnyOrigin],
    acc_i32: MutPointer[Int32, MutAnyOrigin],
    fixed_scale_ptr: MutPointer[Float32, MutAnyOrigin],
    leaf_count_in: Int32,
    stat_count_in: Int32,
):
    """`ComputeSplitPropertiesDirectLoadsTwoStastImpl` with `GroupSize = 4`
    (`compute_hist_loop_two_stats.cuh:495-557`), the multi-part overload.

    Grid, copied from `hist_2_one_byte_base.cuh:172-180`:
        z = (numStats - IsOdd) / 2 STAT PAIRS, y = partCount,
        x = ceil(fCount/4) * replication

    **`blockIdx.z` is a STAT PAIR, not a stat.** The block reads BOTH of its
    pair's columns, `stats` and `stats + statsLineSize`
    (`compute_hist_loop_two_stats.cuh:202-203`), and the writeback emits both
    planes in one call. `skip_first` is their `SkipFirst` template bool: true
    when an odd `numStats` had its stat 0 covered by a one-stat `PASS(Bits,
    1)` prelude, shifting every pair up by one column (`:530`, `:548-550`).

    `stat_count_in` is their `statCount = gridDim.z * 2 + (SkipFirst ? 1 :
    0)` (`:548`), passed as an argument because every kernel in this port
    takes it that way; the launch constructs the grid from the same number,
    so the two agree by construction.
    """
    var fixed_scale = fixed_scale_ptr.unsafe_load(0)
    var f_count_in = Int(f_count_in32)
    var bins_line_size = Int(bins_line_size_in)
    var stat_line_size = Int(stat_line_size_in)
    var leaf_count = Int(leaf_count_in)
    var stat_count = Int(stat_count_in)
    var tid = Int(thread_idx.x)

    var part_id = Int(part_ids.unsafe_load(Int(block_idx.y)))
    var p_offset = Int(part_offset.unsafe_load(part_id))
    var p_size = Int(part_size.unsafe_load(part_id))

    # `maxBlocksPerPart = gridDim.x / ceil(fCount / GroupSize)`
    var feature_blocks = (f_count_in + 3) // 4
    var max_blocks_per_part = Int(grid_dim.x) // feature_blocks
    var feature_offset = (Int(block_idx.x) // max_blocks_per_part) * 4
    var f_count = min(f_count_in - feature_offset, 4)

    # `bins += (binsLineSize * (blockIdx.x / maxBlocksPerPart))` plus the
    # policy's own column base, exactly as the PASS family's kernel notes.
    var bins_p = bins + Int(cindex_base_in) + bins_line_size * (
        Int(block_idx.x) // max_blocks_per_part
    )

    comptime MIN_DOCS = hist2_min_docs[smem_mode]()
    var local_block_idx = Int(block_idx.x) % max_blocks_per_part
    var active_block_count = min(
        (p_size + MIN_DOCS - 1) // MIN_DOCS,
        max_blocks_per_part,
    )
    if local_block_idx >= active_block_count:
        return

    # `stats += ((SkipFirst ? 1 : 0) + 2 * blockIdx.z) * statsLineSize`
    # (`compute_hist_loop_two_stats.cuh:530`). The pair's SECOND column is
    # read at `+ statsLineSize` throughout.
    comptime skip_int = 1 if skip_first else 0
    var stats_p = stats + (skip_int + 2 * Int(block_idx.z)) * stat_line_size

    # `TPointHist2OneByteBase`'s constructor, inlined (PORTING.md 10):
    #     for (i = threadIdx.x; i < histSize; i += BlockSize) hist[i] = 0;
    #     Histogram = hist + impl->SliceOffset();
    #     __syncthreads();
    comptime BLOCK = hist2_block_size[smem_mode]()
    comptime SLOTS = hist2_smem_slots[smem_mode]()
    comptime DT = hist2_acc_dtype[smem_mode]()
    var smem = stack_allocation[
        SLOTS,
        Scalar[DT],
        address_space = AddressSpace.SHARED,
    ]()
    var z = tid
    while z < SLOTS:
        smem[z] = Scalar[DT](0)
        z += BLOCK
    barrier()
    var slice_base = hist2_slice_offset[bits, smem_mode](tid)

    # --- AlignMemoryAccess, two-stat direct overload
    # (`compute_hist_loop_two_stats.cuh:57-108`): peel the unaligned HEAD and
    # TAIL of the partition on block 0 through `AddPoint`, so the striped
    # loop below sees a whole number of aligned warp iterations. Identical in
    # structure to the PASS family's peel; the difference is the second stat
    # column riding along (`:83`, `:102`).
    comptime ALIGN_SIZE = HIST2_LOAD_SIZE * LANE_WIDTH * HIST2_UNROLL

    var head_len = p_size
    var to_align = ALIGN_SIZE - (p_offset % ALIGN_SIZE)
    if to_align < head_len:
        head_len = to_align
    if head_len < 0:
        head_len = 0

    var body_size = p_size - head_len
    if body_size < 0:
        body_size = 0
    var tail_len = body_size % ALIGN_SIZE
    var tail_start = p_offset + head_len + (body_size - tail_len)

    var pe = tid
    while pe < ALIGN_SIZE:
        var hb = InlineArray[UInt32, 1](fill=UInt32(0))
        var hs1 = InlineArray[Float32, 1](fill=Float32(0.0))
        var hs2 = InlineArray[Float32, 1](fill=Float32(0.0))
        var hq1 = InlineArray[Int32, 1](fill=Int32(0))
        var hq2 = InlineArray[Int32, 1](fill=Int32(0))
        if local_block_idx == 0 and pe < head_len:
            # `Ldg(bins, idx)`, `Ldg(stats, idx)`,
            # `Ldg(stats, idx + statsLineSize)`
            # (`compute_hist_loop_two_stats.cuh:81-83`).
            hb[0] = ldg(bins_p + (p_offset + pe))
            hs1[0] = ldg(stats_p + (p_offset + pe))
            hs2[0] = ldg(stats_p + (p_offset + pe + stat_line_size))

            @parameter
            if DT == DType.int32:
                var u = hist2_dither(p_offset + pe)
                hq1[0] = hist2_quantize(hs1[0], fixed_scale, u)
                hq2[0] = hist2_quantize(hs2[0], fixed_scale, u)
        hist2_add_points[bits, 1, DT](
            hb, hs1, hs2, hq1, hq2, tid, slice_base, smem
        )

        var tb = InlineArray[UInt32, 1](fill=UInt32(0))
        var ts1 = InlineArray[Float32, 1](fill=Float32(0.0))
        var ts2 = InlineArray[Float32, 1](fill=Float32(0.0))
        var tq1 = InlineArray[Int32, 1](fill=Int32(0))
        var tq2 = InlineArray[Int32, 1](fill=Int32(0))
        if local_block_idx == 0 and pe < tail_len:
            # `Ldg(bins, tailOffset + idx)` and the two stat loads
            # (`compute_hist_loop_two_stats.cuh:100-102`).
            tb[0] = ldg(bins_p + (tail_start + pe))
            ts1[0] = ldg(stats_p + (tail_start + pe))
            ts2[0] = ldg(stats_p + (tail_start + pe + stat_line_size))

            @parameter
            if DT == DType.int32:
                var u = hist2_dither(tail_start + pe)
                tq1[0] = hist2_quantize(ts1[0], fixed_scale, u)
                tq2[0] = hist2_quantize(ts2[0], fixed_scale, u)
        hist2_add_points[bits, 1, DT](
            tb, ts1, ts2, tq1, tq2, tid, slice_base, smem
        )
        pe += BLOCK

    # the striped loop sees the ALIGNED MIDDLE only
    var aligned_offset = p_offset + head_len
    var aligned_size = body_size - tail_len

    var warps_per_block = BLOCK // LANE_WIDTH
    var global_warp_id = local_block_idx * warps_per_block + (
        tid // LANE_WIDTH
    )
    var entries_per_warp = LANE_WIDTH * HIST2_UNROLL * HIST2_LOAD_SIZE
    var stripe_size = entries_per_warp * warps_per_block * active_block_count
    var remaining = max(aligned_size - global_warp_id * entries_per_warp, 0)
    var local_idx = (tid & (LANE_WIDTH - 1)) * HIST2_LOAD_SIZE

    var base = aligned_offset + global_warp_id * entries_per_warp + local_idx
    var iter_count = (remaining - local_idx + stripe_size - 1) // stripe_size

    # THE BARRIER MUST NOT DIVERGE. Same matrix row and same uniform-count
    # workaround as the PASS family; see `hist_one_byte.mojo`'s block comment
    # at this point, which applies verbatim.
    comptime uniform = requires_uniform_iteration_for[TARGET_COLUMN]()

    @parameter
    if not uniform:
        return

    var max_iters = (aligned_size + stripe_size - 1) // stripe_size
    if max_iters < 1:
        max_iters = 1

    var b_ptr = bins_p + base
    var s_ptr = stats_p + base
    var pos_base = base

    for it in range(max_iters):
        var active = it < iter_count
        var local_bins = InlineArray[UInt32, HIST2_POINTS_PER_ITER](fill=0)
        var local_stats1 = InlineArray[Float32, HIST2_POINTS_PER_ITER](
            fill=Float32(0.0)
        )
        var local_stats2 = InlineArray[Float32, HIST2_POINTS_PER_ITER](
            fill=Float32(0.0)
        )
        var local_q1 = InlineArray[Int32, HIST2_POINTS_PER_ITER](
            fill=Int32(0)
        )
        var local_q2 = InlineArray[Int32, HIST2_POINTS_PER_ITER](
            fill=Int32(0)
        )

        # `Ldg((uint4*) bins, warpSize * k)`, `Ldg((float4*) stats, ...)`,
        # `Ldg((float4*) (stats + statsLineSize), ...)`
        # (`compute_hist_loop_two_stats.cuh:386-396`). Same element-space
        # stride and the same `alignment=4` note as `hist_one_byte.mojo`.
        @parameter
        for k in range(HIST2_UNROLL):
            if active:
                var vb = ldg[width=HIST2_LOAD_SIZE, alignment=4](
                    b_ptr + LANE_WIDTH * HIST2_LOAD_SIZE * k
                )
                var vs1 = ldg[width=HIST2_LOAD_SIZE, alignment=4](
                    s_ptr + LANE_WIDTH * HIST2_LOAD_SIZE * k
                )
                var vs2 = ldg[width=HIST2_LOAD_SIZE, alignment=4](
                    s_ptr + stat_line_size + LANE_WIDTH * HIST2_LOAD_SIZE * k
                )

                @parameter
                for e in range(HIST2_LOAD_SIZE):
                    local_bins[k * HIST2_LOAD_SIZE + e] = vb[e]
                    local_stats1[k * HIST2_LOAD_SIZE + e] = vs1[e]
                    local_stats2[k * HIST2_LOAD_SIZE + e] = vs2[e]

                    @parameter
                    if DT == DType.int32:
                        var u = hist2_dither(
                            pos_base + LANE_WIDTH * HIST2_LOAD_SIZE * k + e
                        )
                        local_q1[k * HIST2_LOAD_SIZE + e] = hist2_quantize(
                            vs1[e], fixed_scale, u
                        )
                        local_q2[k * HIST2_LOAD_SIZE + e] = hist2_quantize(
                            vs2[e], fixed_scale, u
                        )
            else:
                # No row: contribute zero, stay inside every sync.
                @parameter
                for e in range(HIST2_LOAD_SIZE):
                    local_bins[k * HIST2_LOAD_SIZE + e] = UInt32(0)
                    local_stats1[k * HIST2_LOAD_SIZE + e] = Float32(0.0)
                    local_stats2[k * HIST2_LOAD_SIZE + e] = Float32(0.0)
                    local_q1[k * HIST2_LOAD_SIZE + e] = Int32(0)
                    local_q2[k * HIST2_LOAD_SIZE + e] = Int32(0)

        # `hist.AddPoints<loadSize * N>(...)`: batch size equals the whole
        # iteration here, so this is ONE `AddPointsImpl<8>` call, their
        # `AddPoints` loop unrolled at trip count one.
        hist2_add_points[bits, HIST2_POINTS_PER_ITER, DT](
            local_bins, local_stats1, local_stats2, local_q1, local_q2,
            tid, slice_base, smem,
        )

        b_ptr += stripe_size
        s_ptr += stripe_size
        pos_base += stripe_size

    # `hist.Reduce()` then the kernel's own `__syncthreads()`
    # (`compute_hist_loop_two_stats.cuh:214-215`, `:546`).
    hist2_reduce[bits, DT, smem_mode](tid, smem)
    barrier()

    # `hist.AddToGlobalMemory((SkipFirst ? 1 : 0) + 2 * blockIdx.z,
    #                         statCount, activeBlockCount, ...)` (`:550-556`)
    hist2_add_to_global_memory[bits, DT](
        skip_int + 2 * Int(block_idx.z),
        stat_count,
        active_block_count,
        feature_folds,
        feature_fold_offset,
        feature_group_offset,
        feature_group_size,
        feature_offset,
        f_count,
        # `blockIdx.y`, DENSE, not `partIds[blockIdx.y]`, exactly as the
        # PASS family's writeback.
        Int(block_idx.y),
        leaf_count,
        bin_sums,
        acc_i32,
        fixed_scale,
        tid,
        smem,
    )


def hist2_one_byte_gather_kernel[
    bits: Int, skip_first: Bool, smem_mode: Int, ridx_stats: Bool = False
](
    feature_folds: MutPointer[UInt32, MutAnyOrigin],
    feature_fold_offset: MutPointer[UInt32, MutAnyOrigin],
    feature_group_offset: MutPointer[UInt32, MutAnyOrigin],
    feature_group_size: MutPointer[UInt32, MutAnyOrigin],
    f_count_in32: Int32,
    cindex: MutPointer[UInt32, MutAnyOrigin],
    bins_line_size_in: Int32,
    cindex_base_in: Int32,
    indices: MutPointer[UInt32, MutAnyOrigin],
    stats: MutPointer[Float32, MutAnyOrigin],
    stat_line_size_in: Int32,
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    part_ids: MutPointer[UInt32, MutAnyOrigin],
    bin_sums: MutPointer[Float32, MutAnyOrigin],
    acc_i32: MutPointer[Int32, MutAnyOrigin],
    fixed_scale_ptr: MutPointer[Float32, MutAnyOrigin],
    leaf_count_in: Int32,
    stat_count_in: Int32,
):
    """`ComputeSplitPropertiesTwoStatsGatherImpl` with `GroupSize = 4`
    (`compute_hist_loop_two_stats.cuh:560-625`), the multi-part overload.
    Identical to the direct kernel except the bin is read through `indices`
    (their `LoadByIndexBins`; see `launch_one_byte`'s naming note, which
    applies to this family unchanged)."""
    var fixed_scale = fixed_scale_ptr.unsafe_load(0)
    var f_count_in = Int(f_count_in32)
    var bins_line_size = Int(bins_line_size_in)
    var stat_line_size = Int(stat_line_size_in)
    var leaf_count = Int(leaf_count_in)
    var stat_count = Int(stat_count_in)
    var tid = Int(thread_idx.x)

    var part_id = Int(part_ids.unsafe_load(Int(block_idx.y)))
    var p_offset = Int(part_offset.unsafe_load(part_id))
    var p_size = Int(part_size.unsafe_load(part_id))

    var feature_blocks = (f_count_in + 3) // 4
    var max_blocks_per_part = Int(grid_dim.x) // feature_blocks
    var feature_offset = (Int(block_idx.x) // max_blocks_per_part) * 4
    var f_count = min(f_count_in - feature_offset, 4)

    var cindex_p = cindex + Int(cindex_base_in) + bins_line_size * (
        Int(block_idx.x) // max_blocks_per_part
    )
    var idx_p = indices

    comptime MIN_DOCS = hist2_min_docs[smem_mode]()
    var local_block_idx = Int(block_idx.x) % max_blocks_per_part
    var active_block_count = min(
        (p_size + MIN_DOCS - 1) // MIN_DOCS,
        max_blocks_per_part,
    )
    if local_block_idx >= active_block_count:
        return

    comptime skip_int = 1 if skip_first else 0
    var stats_p = stats + (skip_int + 2 * Int(block_idx.z)) * stat_line_size

    comptime BLOCK = hist2_block_size[smem_mode]()
    comptime SLOTS = hist2_smem_slots[smem_mode]()
    comptime DT = hist2_acc_dtype[smem_mode]()
    var smem = stack_allocation[
        SLOTS,
        Scalar[DT],
        address_space = AddressSpace.SHARED,
    ]()
    var z = tid
    while z < SLOTS:
        smem[z] = Scalar[DT](0)
        z += BLOCK
    barrier()
    var slice_base = hist2_slice_offset[bits, smem_mode](tid)

    # --- AlignMemoryAccess, two-stat gather overload
    # (`compute_hist_loop_two_stats.cuh:110-163`).
    comptime ALIGN_SIZE = HIST2_LOAD_SIZE * LANE_WIDTH * HIST2_UNROLL

    var head_len = p_size
    var to_align = ALIGN_SIZE - (p_offset % ALIGN_SIZE)
    if to_align < head_len:
        head_len = to_align
    if head_len < 0:
        head_len = 0

    var body_size = p_size - head_len
    if body_size < 0:
        body_size = 0
    var tail_len = body_size % ALIGN_SIZE
    var tail_start = p_offset + head_len + (body_size - tail_len)

    var pe = tid
    while pe < ALIGN_SIZE:
        var hb = InlineArray[UInt32, 1](fill=UInt32(0))
        var hs1 = InlineArray[Float32, 1](fill=Float32(0.0))
        var hs2 = InlineArray[Float32, 1](fill=Float32(0.0))
        var hq1 = InlineArray[Int32, 1](fill=Int32(0))
        var hq2 = InlineArray[Int32, 1](fill=Int32(0))
        if local_block_idx == 0 and pe < head_len:
            # `Ldg(indices, idx)`, `Ldg(cindex, loadIdx)`, both stat loads
            # (`compute_hist_loop_two_stats.cuh:134-137`).
            var hrow = Int(ldg(indices + (p_offset + pe)))
            hb[0] = ldg(cindex_p + hrow)

            @parameter
            if ridx_stats:
                # DEVIATION 1902: stationary planes, both stats ride the
                # same gathered row id as the bin -- same bits as the
                # permuted plane held here (`split_points_ridx.mojo`).
                hs1[0] = ldg(stats_p + hrow)
                hs2[0] = ldg(stats_p + (hrow + stat_line_size))
            else:
                hs1[0] = ldg(stats_p + (p_offset + pe))
                hs2[0] = ldg(stats_p + (p_offset + pe + stat_line_size))

            @parameter
            if DT == DType.int32:
                var u = hist2_dither(p_offset + pe)
                hq1[0] = hist2_quantize(hs1[0], fixed_scale, u)
                hq2[0] = hist2_quantize(hs2[0], fixed_scale, u)
        hist2_add_points[bits, 1, DT](
            hb, hs1, hs2, hq1, hq2, tid, slice_base, smem
        )

        var tb = InlineArray[UInt32, 1](fill=UInt32(0))
        var ts1 = InlineArray[Float32, 1](fill=Float32(0.0))
        var ts2 = InlineArray[Float32, 1](fill=Float32(0.0))
        var tq1 = InlineArray[Int32, 1](fill=Int32(0))
        var tq2 = InlineArray[Int32, 1](fill=Int32(0))
        if local_block_idx == 0 and pe < tail_len:
            # (`compute_hist_loop_two_stats.cuh:154-157`)
            var trow = Int(ldg(indices + (tail_start + pe)))
            tb[0] = ldg(cindex_p + trow)

            @parameter
            if ridx_stats:
                # DEVIATION 1902, as on the head peel above.
                ts1[0] = ldg(stats_p + trow)
                ts2[0] = ldg(stats_p + (trow + stat_line_size))
            else:
                ts1[0] = ldg(stats_p + (tail_start + pe))
                ts2[0] = ldg(stats_p + (tail_start + pe + stat_line_size))

            @parameter
            if DT == DType.int32:
                var u = hist2_dither(tail_start + pe)
                tq1[0] = hist2_quantize(ts1[0], fixed_scale, u)
                tq2[0] = hist2_quantize(ts2[0], fixed_scale, u)
        hist2_add_points[bits, 1, DT](
            tb, ts1, ts2, tq1, tq2, tid, slice_base, smem
        )
        pe += BLOCK

    var aligned_offset = p_offset + head_len
    var aligned_size = body_size - tail_len

    var warps_per_block = BLOCK // LANE_WIDTH
    var global_warp_id = local_block_idx * warps_per_block + (
        tid // LANE_WIDTH
    )
    var entries_per_warp = LANE_WIDTH * HIST2_UNROLL * HIST2_LOAD_SIZE
    var stripe_size = entries_per_warp * warps_per_block * active_block_count
    var remaining = max(aligned_size - global_warp_id * entries_per_warp, 0)
    var local_idx = (tid & (LANE_WIDTH - 1)) * HIST2_LOAD_SIZE

    var base = aligned_offset + global_warp_id * entries_per_warp + local_idx
    var iter_count = (remaining - local_idx + stripe_size - 1) // stripe_size

    comptime uniform = requires_uniform_iteration_for[TARGET_COLUMN]()

    @parameter
    if not uniform:
        return

    var max_iters = (aligned_size + stripe_size - 1) // stripe_size
    if max_iters < 1:
        max_iters = 1

    var i_ptr = idx_p + base
    var s_ptr = stats_p + base
    var pos_base = base

    for it in range(max_iters):
        var active = it < iter_count
        var local_bins = InlineArray[UInt32, HIST2_POINTS_PER_ITER](fill=0)
        var local_stats1 = InlineArray[Float32, HIST2_POINTS_PER_ITER](
            fill=Float32(0.0)
        )
        var local_stats2 = InlineArray[Float32, HIST2_POINTS_PER_ITER](
            fill=Float32(0.0)
        )
        var local_q1 = InlineArray[Int32, HIST2_POINTS_PER_ITER](
            fill=Int32(0)
        )
        var local_q2 = InlineArray[Int32, HIST2_POINTS_PER_ITER](
            fill=Int32(0)
        )

        # Their gather batch (`compute_hist_loop_two_stats.cuh:424-449`):
        # indices and BOTH stat columns load 4-wide; only the bins are
        # gathered one at a time, because a gather has no vector form.
        @parameter
        for k in range(HIST2_UNROLL):
            if active:
                var vi = ldg[width=HIST2_LOAD_SIZE, alignment=4](
                    i_ptr + LANE_WIDTH * HIST2_LOAD_SIZE * k
                )
                var vs1 = SIMD[DType.float32, HIST2_LOAD_SIZE](0.0)
                var vs2 = SIMD[DType.float32, HIST2_LOAD_SIZE](0.0)

                @parameter
                if not ridx_stats:
                    vs1 = ldg[width=HIST2_LOAD_SIZE, alignment=4](
                        s_ptr + LANE_WIDTH * HIST2_LOAD_SIZE * k
                    )
                    vs2 = ldg[width=HIST2_LOAD_SIZE, alignment=4](
                        s_ptr
                        + stat_line_size
                        + LANE_WIDTH * HIST2_LOAD_SIZE * k
                    )

                @parameter
                for e in range(HIST2_LOAD_SIZE):
                    local_bins[k * HIST2_LOAD_SIZE + e] = ldg(
                        cindex_p + Int(vi[e])
                    )

                    @parameter
                    if ridx_stats:
                        # DEVIATION 1902: both stats join the bins' scalar
                        # gather through the same loaded row id; the wide
                        # loads above are traded for it.
                        vs1[e] = ldg(stats_p + Int(vi[e]))
                        vs2[e] = ldg(
                            stats_p + (Int(vi[e]) + stat_line_size)
                        )
                    local_stats1[k * HIST2_LOAD_SIZE + e] = vs1[e]
                    local_stats2[k * HIST2_LOAD_SIZE + e] = vs2[e]

                    @parameter
                    if DT == DType.int32:
                        var u = hist2_dither(
                            pos_base + LANE_WIDTH * HIST2_LOAD_SIZE * k + e
                        )
                        local_q1[k * HIST2_LOAD_SIZE + e] = hist2_quantize(
                            vs1[e], fixed_scale, u
                        )
                        local_q2[k * HIST2_LOAD_SIZE + e] = hist2_quantize(
                            vs2[e], fixed_scale, u
                        )
            else:

                @parameter
                for e in range(HIST2_LOAD_SIZE):
                    local_bins[k * HIST2_LOAD_SIZE + e] = UInt32(0)
                    local_stats1[k * HIST2_LOAD_SIZE + e] = Float32(0.0)
                    local_stats2[k * HIST2_LOAD_SIZE + e] = Float32(0.0)
                    local_q1[k * HIST2_LOAD_SIZE + e] = Int32(0)
                    local_q2[k * HIST2_LOAD_SIZE + e] = Int32(0)

        hist2_add_points[bits, HIST2_POINTS_PER_ITER, DT](
            local_bins, local_stats1, local_stats2, local_q1, local_q2,
            tid, slice_base, smem,
        )

        i_ptr += stripe_size
        s_ptr += stripe_size
        pos_base += stripe_size

    hist2_reduce[bits, DT, smem_mode](tid, smem)
    barrier()

    hist2_add_to_global_memory[bits, DT](
        skip_int + 2 * Int(block_idx.z),
        stat_count,
        active_block_count,
        feature_folds,
        feature_fold_offset,
        feature_group_offset,
        feature_group_size,
        feature_offset,
        f_count,
        Int(block_idx.y),
        leaf_count,
        bin_sums,
        acc_i32,
        fixed_scale,
        tid,
        smem,
    )
