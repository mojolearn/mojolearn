"""The one-byte histogram kernel: 4 features per 4-byte load, 32 to 256 bins.

PORT OF `hist_one_byte.cu` and the base it derives from,
`hist_2_one_byte_base.cuh`, at CatBoost `54a8143a`. Transliterated. Do not
improve.

This is the odd one of the three and it does not derive from the other two.
Where binary and half-byte share `TPointHistHalfByteBase` at 16 floats per
thread, this one uses 32 and trades replication for serialization to make
wide features fit.

THE INNER-BITS TRICK, which is the whole file
---------------------------------------------
A one-byte feature has up to 256 bins, far more than a private per-lane slot
can hold. So the bin is split:

    InnerHistBitsCount = Bits - 5
    higherBin = (bin >> 5) & ((1 << InnerHistBitsCount) - 1)
    offset    = 4 * higherBin + f + ((bin & 31) << 5)

The low 5 bits of the bin index the slot directly. The HIGH bits are handled
by running `1 << InnerHistBitsCount` PASSES, where on pass `p` only the lanes
whose `higherBin == p` write:

    for (k = 0; k < (1 << InnerHistBitsCount); ++k) {
        const int pass = ((threadIdx.x >> 2) + k) & mask;
        syncTile.sync();
        if (pass == higherBin) Histogram[offset] += statToAdd;
    }

So a 5-bit feature costs one pass and a 8-bit feature costs eight. **Wide
features are paid for in TIME rather than in shared memory**, which is the
opposite trade from replicating the histogram, and it is why this kernel can
exist at all inside a 32 KB budget.

Their `SliceOffset` differs to match: `1024 * (threadIdx.x / 32)` gives each
warp 1024 floats (32 per lane), and the inner offset is masked by the number
of blocks the inner bits leave.

DEVIATION (PORTING.md 1): CatBoost runs this at `BlockSize = 384`, so
`384 * 32` floats is 49,152 bytes and Apple gives 32,768. `BLOCK_SIZE = 256`
asks for exactly 32,768 and keeps their per-warp slice arithmetic intact,
since 8 warps times 1024 floats is 8192 floats.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, grid_dim, thread_idx

from mojo_only.kernel_matrix import (
    TARGET_COLUMN,
    deterministic_flush_for,
    requires_uniform_iteration_for,
)
from mojo_only.numerics import NUMERIC_FAST, NUMERIC_IDENTICAL
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from mojo_only.kernel_matrix import (
    K_HIST_ONE_BYTE,
    TARGET_COLUMN,
    block_size_for,
    hist_floats_per_thread_for,
)


#: READ FROM THE MATRIX. Theirs is 384; Apple's 32 KB over 32 floats per
#: thread yields 256, which is exactly 32,768 bytes and keeps their per-warp
#: slice arithmetic intact at 8 warps of 1024 floats.
#: Same build mode as `hist_binary.mojo`; the flush follows the matrix.
comptime BUILD_MODE = NUMERIC_FAST

comptime ONE_BYTE_BLOCK_SIZE = block_size_for[K_HIST_ONE_BYTE, TARGET_COLUMN]()

#: `GetHistSize()` = `BlockSize * 32`, the 32 from the matrix.
comptime ONE_BYTE_HIST_SIZE = ONE_BYTE_BLOCK_SIZE * hist_floats_per_thread_for[
    K_HIST_ONE_BYTE
]()

#: `Unroll` for `__CUDA_ARCH__ >= 700`.
comptime ONE_BYTE_UNROLL = 2

#: Their unroll and load width, named as the loop expects them.
comptime UNROLL = ONE_BYTE_UNROLL

#: DEVIATION (PORTING.md 5): CatBoost picks a 4-element vector load here
#: (`hist_2_one_byte_base.cuh:44-48`); the port takes one element, which is
#: scheduling and not numeric.
#: `ELoadSize::FourElements` (`hist_one_byte.cu:47`). CatBoost picks it on
#: every GPU newer than Maxwell, and it is the reason a thread of theirs
#: consumes four points per load where ours used to consume one. The
#: histogram is bandwidth bound, so this is the load path, not a detail.
#:
#: Raising it REQUIRES the head/tail peel below: a partition offset is not
#: 4-aligned in general, and their whole reason for `AlignMemoryAccess` is
#: that the main loop may then issue aligned vector loads with no bounds
#: test inside it.
comptime LOAD_SIZE = 4

#: `loadSize * N`, the points one thread takes per iteration.
comptime POINTS_PER_ITER = UNROLL * LOAD_SIZE

comptime LANE_WIDTH = 32


def one_byte_slice_offset[bits: Int](tid: Int) -> Int:
    """`TPointHistOneByte::SliceOffset()`, copied.

        const int warpOffset = 1024 * (threadIdx.x / 32);
        const int blocks = 8 >> InnerHistBitsCount;
        const int innerHistStart =
            threadIdx.x & ((blocks - 1) << (InnerHistBitsCount + 2));
        return warpOffset + innerHistStart;

    `blocks` shrinks as the feature widens: a 5-bit feature gets 8 private
    sub-copies per warp, an 8-bit feature gets 1. That is the same trade as
    the pass loop, seen from the memory side.
    """
    comptime inner_bits = bits - 5
    var warp_offset = 1024 * (tid // LANE_WIDTH)
    comptime blocks = 8 >> inner_bits
    var inner_hist_start = tid & ((blocks - 1) << (inner_bits + 2))
    return warp_offset + inner_hist_start


def one_byte_bin_offset[bits: Int](ci: UInt32, tid: Int, i: Int) -> Int:
    """The slot for iteration `i`, as pure arithmetic (PORTING.md 10).

        int f = (threadIdx.x + i) & 3;
        int bin = (ci >> (24 - 8 * f)) & 255;
        const int higherBin = (bin >> 5) & mask;
        int offset = 4 * higherBin + f + ((bin & 31) << 5);

    Four features per word here, so the rotation is `& 3` rather than `& 7`.
    """
    comptime inner_bits = bits - 5
    comptime mask = (1 << inner_bits) - 1
    var f = (tid + i) & 3
    var bin = Int((ci >> UInt32(24 - 8 * f)) & UInt32(255))
    var higher_bin = (bin >> 5) & mask
    return 4 * higher_bin + f + ((bin & 31) << 5)


def add_one_byte_point[
    bits: Int
](
    ci: UInt32,
    stat: Float32,
    tid: Int,
    slice_base: Int,
    smem: UnsafePointer[
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
        origin=MutUntrackedOrigin,
    ],
):
    """`TPointHistOneByte::AddPoint`, one point, as a callable.

    Their `AddPoint` is a method, so `AlignMemoryAccess` calls the SAME code
    the main loop does (`compute_hist_loop_two_stats.cuh:81`). Ours had it
    inlined in the loop, which is fine until the head/tail peel needs it too,
    and duplicating twenty lines of barrier-carrying accumulation in four
    places is how the two copies drift.

    Four features per word, so the rotation is `& 3`. The PASS LOOP is what
    separates this from the half-byte accumulator: the low five bits of the
    bin index the slot directly and the HIGH bits are serialized, one pass
    each, with only the lanes whose `higherBin == pass` writing.

    **Every barrier here is threadgroup-wide and must be reached by every
    thread of the block.** Callers therefore run a UNIFORM number of trips;
    see the peel and the main loop.
    """

    @parameter
    for i in range(4):
        var slot = slice_base + one_byte_bin_offset[bits](ci, tid, i)
        comptime inner_bits = bits - 5

        @parameter
        if inner_bits == 0:
            barrier()
            smem[slot] = smem[slot] + stat
        else:
            var higher = one_byte_higher_bin[bits](ci, tid, i)
            comptime mask = (1 << inner_bits) - 1

            @parameter
            for kk in range(1 << inner_bits):
                var p = ((tid >> 2) + kk) & mask
                barrier()
                if p == higher:
                    smem[slot] = smem[slot] + stat


def one_byte_higher_bin[bits: Int](ci: UInt32, tid: Int, i: Int) -> Int:
    """`higherBin` alone: which PASS of the inner-bits loop may write."""
    comptime inner_bits = bits - 5
    comptime mask = (1 << inner_bits) - 1
    var f = (tid + i) & 3
    var bin = Int((ci >> UInt32(24 - 8 * f)) & UInt32(255))
    return (bin >> 5) & mask


def one_byte_hist_kernel[bits: Int](
    # `TFeatureInBlock*`, flattened to four parallel arrays so the kernel
    # takes plain pointers.
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
    fixed_scale: Float32,
    leaf_count_in: Int32,
    stat_count_in: Int32,
):
    """`ComputeSplitPropertiesDirectLoadsImpl` with `GroupSize = 4`.

    Grid, copied from `hist_one_byte.cu:284-290`:
        z = numStats, y = partCount, x = ceil(fCount/4) * replication

    `bits` is their template parameter, 5 to 8, and it decides how many
    PASSES the inner loop runs: a 5-bit feature costs one and an 8-bit
    feature costs eight. See `one_byte_bin_offset`.

    **`blockIdx.y` is the LEAF.** All leaves of a level are one launch, and
    the number of launches does not depend on the leaf count or on the
    dataset size. That is the design point (`compute_by_blocks_helper.h:87-92`
    states it outright) and it is the opposite of a per-leaf launch.
    """
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

    # `cindex += features->CompressedIndexOffset` in theirs: the base of
    # THIS POLICY's columns. Distinct from `bins_line_size`, which is the
    # stride between FEATURE BLOCKS inside the policy. Conflating them makes
    # every policy after the first read the first one's bits, which looks
    # like a tree that will not split.
    var bins_p = bins + Int(cindex_base_in) + bins_line_size * (
        Int(block_idx.x) // max_blocks_per_part
    )

    # THE RAGGED-LEAF TRICK, and it costs nothing. The grid is sized for the
    # worst-case leaf; every block reads its partition's size ON THE DEVICE
    # and the ones with no work return. No host round trip, no per-leaf
    # launch, no dependence on how unbalanced the level is.
    var local_block_idx = Int(block_idx.x) % max_blocks_per_part
    var min_docs_per_block = LANE_WIDTH * UNROLL * LOAD_SIZE * (
        ONE_BYTE_BLOCK_SIZE // LANE_WIDTH
    )
    var active_block_count = min(
        (p_size + min_docs_per_block - 1) // min_docs_per_block,
        max_blocks_per_part,
    )
    if local_block_idx >= active_block_count:
        return

    var stats_p = stats + Int(block_idx.z) * stat_line_size

    # `TPointHistHalfByteBase`'s constructor, inlined (PORTING.md 10):
    #     for (i = threadIdx.x; i < histSize; i += BlockSize) buff[i] = 0;
    #     __syncthreads();
    #     Histogram = buff + SliceOffset();
    var smem = stack_allocation[
        ONE_BYTE_HIST_SIZE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var z = tid
    while z < ONE_BYTE_HIST_SIZE:
        smem[z] = Float32(0.0)
        z += ONE_BYTE_BLOCK_SIZE
    barrier()
    var slice_base = one_byte_slice_offset[bits](tid)

    # --- AlignMemoryAccess, ported (`compute_hist_loop_two_stats.cuh:57`) --
    #
    # Peels the unaligned HEAD and TAIL of the partition on block 0 with
    # scalar adds, so what remains starts and ends on an `alignSize`
    # boundary. `alignSize = LoadSize * warpSize * N` is exactly one warp
    # iteration, which is what lets the striped loop below issue ALIGNED
    # 4-wide loads and carry no per-element bounds test.
    #
    #     int lastId = min(partSize, alignSize - (partOffset % alignSize));
    #     if (blockId == 0) for (idx = tid; idx < alignSize; idx += BlockSize)
    #     partSize = max(partSize - lastId, 0);
    #     const int unalignedTail = (partSize % alignSize);
    #     if (unalignedTail) { if (blockId == 0) { ...tail... } }
    #     partSize -= unalignedTail;
    #
    # Their head and tail loops run to a FIXED `alignSize` bound, so every
    # thread of block 0 makes the same number of trips and the threadgroup
    # barriers inside `AddPoint` stay uniform. Blocks other than 0 must still
    # make those trips, because the barriers are threadgroup-wide and Mojo
    # has no warp-local form; they contribute zeros.
    comptime ALIGN_SIZE = LOAD_SIZE * LANE_WIDTH * UNROLL

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
        var hb = UInt32(0)
        var hs = Float32(0.0)
        if local_block_idx == 0 and pe < head_len:
            hb = bins_p.unsafe_load(p_offset + pe)
            hs = stats_p.unsafe_load(p_offset + pe)
        add_one_byte_point[bits](hb, hs, tid, slice_base, smem)

        var tb = UInt32(0)
        var ts = Float32(0.0)
        if local_block_idx == 0 and pe < tail_len:
            tb = bins_p.unsafe_load(tail_start + pe)
            ts = stats_p.unsafe_load(tail_start + pe)
        add_one_byte_point[bits](tb, ts, tid, slice_base, smem)
        pe += ONE_BYTE_BLOCK_SIZE

    # the striped loop sees the ALIGNED MIDDLE only
    var aligned_offset = p_offset + head_len
    var aligned_size = body_size - tail_len

    var warps_per_block = ONE_BYTE_BLOCK_SIZE // LANE_WIDTH
    var global_warp_id = local_block_idx * warps_per_block + (tid // LANE_WIDTH)
    var entries_per_warp = LANE_WIDTH * UNROLL * LOAD_SIZE
    var stripe_size = entries_per_warp * warps_per_block * active_block_count
    var remaining = max(aligned_size - global_warp_id * entries_per_warp, 0)
    var local_idx = (tid & (LANE_WIDTH - 1)) * LOAD_SIZE

    var base = aligned_offset + global_warp_id * entries_per_warp + local_idx
    var iter_count = (remaining - local_idx + stripe_size - 1) // stripe_size

    # THE BARRIER MUST NOT DIVERGE. The requirement is a MATRIX ROW,
    # `requires_uniform_iteration_for`, not a local decision, so that when
    # Mojo exposes lane primitives the kernels follow the table instead of
    # each carrying its own workaround.
    comptime uniform = requires_uniform_iteration_for[TARGET_COLUMN]()

    @parameter
    if not uniform:
        # Deliberately unimplemented rather than silently wrong: the literal
        # CatBoost loop needs a lane-local sync, and reaching here means the
        # matrix claims one exists. Write that path before flipping the row.
        return

    # This is where the port stops being a transliteration. CatBoost syncs a `tiled_partition<8>`, which is
    # WARP-LOCAL, so warps with different iteration counts never wait on each
    # other. Mojo 1.0 has only the threadgroup-wide `barrier()`, and a
    # threadgroup barrier that some warps reach and others skip is undefined
    # behavior.
    #
    # It bites immediately rather than rarely: a 64-row partition over a
    # 512-thread block gives warp 0 one iteration and warps 1 to 15 zero, so
    # the first fifteen sixteenths of the block walk past barriers warp 0 is
    # waiting on. Measured result before this fix: every feature's histogram
    # came back 0.0.
    #
    # So every thread runs the SAME iteration count and the ones with no rows
    # contribute nothing. `block.max` is not available here, so the count is
    # derived from the partition size, which every thread already has.
    var max_iters = (aligned_size + stripe_size - 1) // stripe_size
    if max_iters < 1:
        max_iters = 1

    var b_ptr = bins_p + base
    var s_ptr = stats_p + base

    for it in range(max_iters):
        var active = it < iter_count
        # Their two unrolled loops: gather the batch, then add it. Kept in
        # that order because it is what keeps the loads in flight.
        var local_bins = InlineArray[UInt32, POINTS_PER_ITER](fill=0)
        var local_stats = InlineArray[Float32, POINTS_PER_ITER](fill=0)

        # `Ldg((uint4*) bins, warpSize * k)` and its `float4` twin
        # (`compute_hist_loop_two_stats.cuh:293`). Indexing a `uint4*` by
        # `warpSize * k` advances `warpSize * k * 4` ELEMENTS, which is the
        # element-space stride written here.
        #
        # NO per-element bounds test. The peel above leaves a whole number of
        # warp iterations, so an ACTIVE iteration is wholly in range. That is
        # the entire reason `AlignMemoryAccess` exists, and it is what makes
        # a 4-wide load legal as well as fast. Only the uniform-iteration
        # guard remains, and that one is ours.
        @parameter
        for k in range(UNROLL):
            if active:
                var vb = (b_ptr + LANE_WIDTH * LOAD_SIZE * k).load[
                    width=LOAD_SIZE
                ]()
                var vs = (s_ptr + LANE_WIDTH * LOAD_SIZE * k).load[
                    width=LOAD_SIZE
                ]()

                @parameter
                for e in range(LOAD_SIZE):
                    local_bins[k * LOAD_SIZE + e] = vb[e]
                    local_stats[k * LOAD_SIZE + e] = vs[e]
            else:
                # No row: contribute zero. Harmless, and it keeps this lane
                # inside every barrier below.
                @parameter
                for e in range(LOAD_SIZE):
                    local_bins[k * LOAD_SIZE + e] = UInt32(0)
                    local_stats[k * LOAD_SIZE + e] = Float32(0.0)

        # `hist.AddPoints<loadSize * N>(...)`: every point the batch loaded,
        # through the same `AddPoint` the peel calls.
        @parameter
        for k in range(POINTS_PER_ITER):
            add_one_byte_point[bits](
                local_bins[k], local_stats[k], tid, slice_base, smem
            )

        b_ptr += stripe_size
        s_ptr += stripe_size

    # `TPointHistOneByte::Reduce` (`hist_one_byte.cu:177-230`), copied. TWO
    # stages, and they are not interchangeable with one strided fold.
    #
    # A single fold at stride `4 * histSize` gives the RIGHT answer whenever
    # every cell holds the same value and the WRONG one otherwise, so a
    # histogram check built on `(r + f) % folds` passes it and real data does
    # not. That is how it survived: the uniform pattern gives consecutive
    # rows consecutive bins, every cell ends up equal, and summing the wrong
    # set of equal cells still lands on the right number.
    comptime inner_bits = bits - 5
    comptime hist_size_bins = 1 << (5 + inner_bits)
    comptime WARP_HIST_SIZE = 1024

    # Stage 1: fold the per-warp copies down to 1024 entries, parked at
    # `Histogram[1024 + start]`. Stride 1024 means one thread owns each
    # residue class, so the write at `1024 + start` is by the same thread
    # that already read it. That is why their loop needs no barrier inside.
    barrier()
    var start = tid
    while start < WARP_HIST_SIZE:
        var acc = Float32(0.0)
        var j = start
        while j < ONE_BYTE_HIST_SIZE:
            acc += smem[j]
            j += WARP_HIST_SIZE
        smem[WARP_HIST_SIZE + start] = acc
        start += ONE_BYTE_BLOCK_SIZE
    barrier()

    # Stage 2: UN-SCRAMBLE. The slot for feature `f`, bin `b` is
    # `((b & 31) << 5) + 4 * (b >> 5) + f + innerHistStart`, so the sub-copies
    # a warp keeps at `innerHistStart` in `{0, blockSize, 2*blockSize, ...}`
    # have to be gathered per feature. `i` here is the FEATURE.
    comptime warp_hist_block_count = 8 >> inner_bits
    comptime sub_block = 4 * (1 << inner_bits)
    var fold_r = tid
    var sums = InlineArray[Float32, 4](fill=Float32(0.0))
    if fold_r < hist_size_bins:
        var lower_bits_offset = (fold_r & 31) << 5
        var higher_bin = (fold_r >> 5) & ((1 << inner_bits) - 1)
        var src = WARP_HIST_SIZE + lower_bits_offset + 4 * higher_bin

        @parameter
        for blk in range(warp_hist_block_count):

            @parameter
            for i in range(4):
                sums[i] += smem[src + i + blk * sub_block]

    # Their `__syncthreads()` between the gather and the store: the read
    # window and the write window are disjoint here, but the barrier is
    # theirs and dropping it is a bet on that staying true.
    barrier()
    if fold_r < hist_size_bins:

        @parameter
        for i in range(4):
            smem[hist_size_bins * i + fold_r] = sums[i]
    barrier()

    # `AddToGlobalMemory`, one-byte: one thread per FOLD, looping features.
    var fold = tid
    for fid in range(f_count):
        var folds = Int(feature_folds.unsafe_load(feature_offset + fid))
        if fold < folds:
            var group_offset = Int(feature_group_offset.unsafe_load(feature_offset + fid))
            var group_size = Int(feature_group_size.unsafe_load(feature_offset + fid))
            var device_offset = group_offset * stat_count * leaf_count
            var entries_per_leaf = stat_count * group_size
            var dst = (
                bin_sums
                + device_offset
                # `blockIdx.y`, DENSE, not `partIds[blockIdx.y]`. See the
                # note above `entries_per_leaf`.
                + Int(block_idx.y) * entries_per_leaf
                + Int(block_idx.z) * group_size
                + Int(feature_fold_offset.unsafe_load(feature_offset + fid))
            )

            var val = smem[fid * hist_size_bins + fold]

            if abs(val) > Float32(1e-20):
                # `AddToGlobalMemory`, their multi-block branch:
                #
                #     if (blockCount > 1) { atomicAdd(dst + fold, val); }
                #     else               { dst[fold] = val; }
                #
                # DEVIATION (numerics.mojo, same as `hist_binary.mojo`):
                # Metal has no float atomic, so replicated blocks sum as
                # Int32 through an integer atomic, which is associative and
                # therefore reproducible run to run.
                #
                # Until this existed the kernel took the plain store on BOTH
                # branches, so replicating it left one block's partial
                # histogram standing and threw the other fifteen away. See
                # UNWIRED.md.
                comptime det = deterministic_flush_for[
                    TARGET_COLUMN, BUILD_MODE == NUMERIC_IDENTICAL
                ]()
                if det and active_block_count > 1:
                    var q = Int32(val * fixed_scale)
                    _ = Atomic.fetch_add(
                        acc_i32.unsafe_offset(
                            device_offset
                            + Int(block_idx.y) * entries_per_leaf
                            + Int(block_idx.z) * group_size
                            + Int(
                                feature_fold_offset.unsafe_load(
                                    feature_offset + fid
                                )
                            )
                            + fold
                        ),
                        q,
                    )
                    continue

                # DEVIATION (numerics.mojo): CatBoost uses atomicAdd when
                # blockCount > 1 and a plain store otherwise. The atomic is
                # order-nondeterministic and is what `NUMERIC_IDENTICAL`
                # replaces with a fixed-point accumulator. Direct store here;
                # the multi-block flush lands with the replication work.
                dst.unsafe_store(fold, val)


def one_byte_hist_gather_kernel[bits: Int](
    # `TFeatureInBlock*`, flattened to four parallel arrays so the kernel
    # takes plain pointers.
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
    fixed_scale: Float32,
    leaf_count_in: Int32,
    stat_count_in: Int32,
):
    """`ComputeSplitPropertiesGatherImpl` with `GroupSize = 4`.

    The gather instantiation, `GroupSize = 4`. Identical to the direct
    one except the bin is read through `indices`.

    Grid, copied from `hist_one_byte.cu:284-290`:
        z = numStats, y = partCount, x = ceil(fCount/4) * replication

    `bits` is their template parameter, 5 to 8, and it decides how many
    PASSES the inner loop runs: a 5-bit feature costs one and an 8-bit
    feature costs eight. See `one_byte_bin_offset`.

    **`blockIdx.y` is the LEAF.** All leaves of a level are one launch, and
    the number of launches does not depend on the leaf count or on the
    dataset size. That is the design point (`compute_by_blocks_helper.h:87-92`
    states it outright) and it is the opposite of a per-leaf launch.
    """
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

    # See the note in the direct variant: this is the policy's column base,
    # not the feature-block stride.
    var cindex_p = cindex + Int(cindex_base_in) + bins_line_size * (
        Int(block_idx.x) // max_blocks_per_part
    )
    var idx_p = indices

    # THE RAGGED-LEAF TRICK, and it costs nothing. The grid is sized for the
    # worst-case leaf; every block reads its partition's size ON THE DEVICE
    # and the ones with no work return. No host round trip, no per-leaf
    # launch, no dependence on how unbalanced the level is.
    var local_block_idx = Int(block_idx.x) % max_blocks_per_part
    var min_docs_per_block = LANE_WIDTH * UNROLL * LOAD_SIZE * (
        ONE_BYTE_BLOCK_SIZE // LANE_WIDTH
    )
    var active_block_count = min(
        (p_size + min_docs_per_block - 1) // min_docs_per_block,
        max_blocks_per_part,
    )
    if local_block_idx >= active_block_count:
        return

    var stats_p = stats + Int(block_idx.z) * stat_line_size

    # `TPointHistHalfByteBase`'s constructor, inlined (PORTING.md 10):
    #     for (i = threadIdx.x; i < histSize; i += BlockSize) buff[i] = 0;
    #     __syncthreads();
    #     Histogram = buff + SliceOffset();
    var smem = stack_allocation[
        ONE_BYTE_HIST_SIZE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var z = tid
    while z < ONE_BYTE_HIST_SIZE:
        smem[z] = Float32(0.0)
        z += ONE_BYTE_BLOCK_SIZE
    barrier()
    var slice_base = one_byte_slice_offset[bits](tid)

    # --- AlignMemoryAccess (gather), ported
    # (`compute_hist_loop_two_stats.cuh:110`). Same peel as the direct
    # variant; the difference is only that the bins come through `indices`.
    comptime ALIGN_SIZE = LOAD_SIZE * LANE_WIDTH * UNROLL

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
        var hb = UInt32(0)
        var hs = Float32(0.0)
        if local_block_idx == 0 and pe < head_len:
            var hrow = Int(indices.unsafe_load(p_offset + pe))
            hb = cindex_p.unsafe_load(hrow)
            hs = stats_p.unsafe_load(p_offset + pe)
        add_one_byte_point[bits](hb, hs, tid, slice_base, smem)

        var tb = UInt32(0)
        var ts = Float32(0.0)
        if local_block_idx == 0 and pe < tail_len:
            var trow = Int(indices.unsafe_load(tail_start + pe))
            tb = cindex_p.unsafe_load(trow)
            ts = stats_p.unsafe_load(tail_start + pe)
        add_one_byte_point[bits](tb, ts, tid, slice_base, smem)
        pe += ONE_BYTE_BLOCK_SIZE

    var aligned_offset = p_offset + head_len
    var aligned_size = body_size - tail_len

    var warps_per_block = ONE_BYTE_BLOCK_SIZE // LANE_WIDTH
    var global_warp_id = local_block_idx * warps_per_block + (tid // LANE_WIDTH)
    var entries_per_warp = LANE_WIDTH * UNROLL * LOAD_SIZE
    var stripe_size = entries_per_warp * warps_per_block * active_block_count
    var remaining = max(aligned_size - global_warp_id * entries_per_warp, 0)
    var local_idx = (tid & (LANE_WIDTH - 1)) * LOAD_SIZE

    var base = aligned_offset + global_warp_id * entries_per_warp + local_idx
    var iter_count = (remaining - local_idx + stripe_size - 1) // stripe_size

    # THE BARRIER MUST NOT DIVERGE. The requirement is a MATRIX ROW,
    # `requires_uniform_iteration_for`, not a local decision, so that when
    # Mojo exposes lane primitives the kernels follow the table instead of
    # each carrying its own workaround.
    comptime uniform = requires_uniform_iteration_for[TARGET_COLUMN]()

    @parameter
    if not uniform:
        # Deliberately unimplemented rather than silently wrong: the literal
        # CatBoost loop needs a lane-local sync, and reaching here means the
        # matrix claims one exists. Write that path before flipping the row.
        return

    # This is where the port stops being a transliteration. CatBoost syncs a `tiled_partition<8>`, which is
    # WARP-LOCAL, so warps with different iteration counts never wait on each
    # other. Mojo 1.0 has only the threadgroup-wide `barrier()`, and a
    # threadgroup barrier that some warps reach and others skip is undefined
    # behavior.
    #
    # It bites immediately rather than rarely: a 64-row partition over a
    # 512-thread block gives warp 0 one iteration and warps 1 to 15 zero, so
    # the first fifteen sixteenths of the block walk past barriers warp 0 is
    # waiting on. Measured result before this fix: every feature's histogram
    # came back 0.0.
    #
    # So every thread runs the SAME iteration count and the ones with no rows
    # contribute nothing. `block.max` is not available here, so the count is
    # derived from the partition size, which every thread already has.
    var max_iters = (aligned_size + stripe_size - 1) // stripe_size
    if max_iters < 1:
        max_iters = 1

    var i_ptr = idx_p + base
    var s_ptr = stats_p + base

    for it in range(max_iters):
        var active = it < iter_count
        # Their two unrolled loops: gather the batch, then add it. Kept in
        # that order because it is what keeps the loads in flight.
        var local_bins = InlineArray[UInt32, POINTS_PER_ITER](fill=0)
        var local_stats = InlineArray[Float32, POINTS_PER_ITER](fill=0)

        # Their gather batch (`compute_hist_loop_two_stats.cuh:410`):
        #
        #     localIndices[k] = Ldg((int4*) indices, warpSize * k);
        #     localBins[k].x  = Ldg(cindex, localIndices[k].x);   ...
        #     localStats1[k]  = Ldg((float4*) stats, warpSize * k);
        #
        # The INDICES and the STATS are contiguous, so both load 4-wide. Only
        # the BINS are gathered, one at a time, because a gather has no
        # vector form. That asymmetry is theirs and it is the point: two of
        # the three streams still get the wide load.
        @parameter
        for k in range(UNROLL):
            if active:
                var vi = (i_ptr + LANE_WIDTH * LOAD_SIZE * k).load[
                    width=LOAD_SIZE
                ]()
                var vs = (s_ptr + LANE_WIDTH * LOAD_SIZE * k).load[
                    width=LOAD_SIZE
                ]()

                @parameter
                for e in range(LOAD_SIZE):
                    # The compressed index is never permuted, so after a
                    # reorder a position no longer names a row and the index
                    # array is the only way back. See hist_binary.
                    local_bins[k * LOAD_SIZE + e] = cindex_p.unsafe_load(
                        Int(vi[e])
                    )
                    local_stats[k * LOAD_SIZE + e] = vs[e]
            else:
                @parameter
                for e in range(LOAD_SIZE):
                    local_bins[k * LOAD_SIZE + e] = UInt32(0)
                    local_stats[k * LOAD_SIZE + e] = Float32(0.0)

        # `hist.AddPoints<loadSize * N>(...)`
        @parameter
        for k in range(POINTS_PER_ITER):
            add_one_byte_point[bits](
                local_bins[k], local_stats[k], tid, slice_base, smem
            )

        i_ptr += stripe_size
        s_ptr += stripe_size

    # `TPointHistOneByte::Reduce` (`hist_one_byte.cu:177-230`), copied. TWO
    # stages, and they are not interchangeable with one strided fold.
    #
    # A single fold at stride `4 * histSize` gives the RIGHT answer whenever
    # every cell holds the same value and the WRONG one otherwise, so a
    # histogram check built on `(r + f) % folds` passes it and real data does
    # not. That is how it survived: the uniform pattern gives consecutive
    # rows consecutive bins, every cell ends up equal, and summing the wrong
    # set of equal cells still lands on the right number.
    comptime inner_bits = bits - 5
    comptime hist_size_bins = 1 << (5 + inner_bits)
    comptime WARP_HIST_SIZE = 1024

    # Stage 1: fold the per-warp copies down to 1024 entries, parked at
    # `Histogram[1024 + start]`. Stride 1024 means one thread owns each
    # residue class, so the write at `1024 + start` is by the same thread
    # that already read it. That is why their loop needs no barrier inside.
    barrier()
    var start = tid
    while start < WARP_HIST_SIZE:
        var acc = Float32(0.0)
        var j = start
        while j < ONE_BYTE_HIST_SIZE:
            acc += smem[j]
            j += WARP_HIST_SIZE
        smem[WARP_HIST_SIZE + start] = acc
        start += ONE_BYTE_BLOCK_SIZE
    barrier()

    # Stage 2: UN-SCRAMBLE. The slot for feature `f`, bin `b` is
    # `((b & 31) << 5) + 4 * (b >> 5) + f + innerHistStart`, so the sub-copies
    # a warp keeps at `innerHistStart` in `{0, blockSize, 2*blockSize, ...}`
    # have to be gathered per feature. `i` here is the FEATURE.
    comptime warp_hist_block_count = 8 >> inner_bits
    comptime sub_block = 4 * (1 << inner_bits)
    var fold_r = tid
    var sums = InlineArray[Float32, 4](fill=Float32(0.0))
    if fold_r < hist_size_bins:
        var lower_bits_offset = (fold_r & 31) << 5
        var higher_bin = (fold_r >> 5) & ((1 << inner_bits) - 1)
        var src = WARP_HIST_SIZE + lower_bits_offset + 4 * higher_bin

        @parameter
        for blk in range(warp_hist_block_count):

            @parameter
            for i in range(4):
                sums[i] += smem[src + i + blk * sub_block]

    # Their `__syncthreads()` between the gather and the store: the read
    # window and the write window are disjoint here, but the barrier is
    # theirs and dropping it is a bet on that staying true.
    barrier()
    if fold_r < hist_size_bins:

        @parameter
        for i in range(4):
            smem[hist_size_bins * i + fold_r] = sums[i]
    barrier()

    # `AddToGlobalMemory`, one-byte: one thread per FOLD, looping features.
    var fold = tid
    for fid in range(f_count):
        var folds = Int(feature_folds.unsafe_load(feature_offset + fid))
        if fold < folds:
            var group_offset = Int(feature_group_offset.unsafe_load(feature_offset + fid))
            var group_size = Int(feature_group_size.unsafe_load(feature_offset + fid))
            var device_offset = group_offset * stat_count * leaf_count
            var entries_per_leaf = stat_count * group_size
            var dst = (
                bin_sums
                + device_offset
                # `blockIdx.y`, DENSE, not `partIds[blockIdx.y]`. See the
                # note above `entries_per_leaf`.
                + Int(block_idx.y) * entries_per_leaf
                + Int(block_idx.z) * group_size
                + Int(feature_fold_offset.unsafe_load(feature_offset + fid))
            )

            var val = smem[fid * hist_size_bins + fold]

            if abs(val) > Float32(1e-20):
                # `AddToGlobalMemory`, their multi-block branch:
                #
                #     if (blockCount > 1) { atomicAdd(dst + fold, val); }
                #     else               { dst[fold] = val; }
                #
                # DEVIATION (numerics.mojo, same as `hist_binary.mojo`):
                # Metal has no float atomic, so replicated blocks sum as
                # Int32 through an integer atomic, which is associative and
                # therefore reproducible run to run.
                #
                # Until this existed the kernel took the plain store on BOTH
                # branches, so replicating it left one block's partial
                # histogram standing and threw the other fifteen away. See
                # UNWIRED.md.
                comptime det = deterministic_flush_for[
                    TARGET_COLUMN, BUILD_MODE == NUMERIC_IDENTICAL
                ]()
                if det and active_block_count > 1:
                    var q = Int32(val * fixed_scale)
                    _ = Atomic.fetch_add(
                        acc_i32.unsafe_offset(
                            device_offset
                            + Int(block_idx.y) * entries_per_leaf
                            + Int(block_idx.z) * group_size
                            + Int(
                                feature_fold_offset.unsafe_load(
                                    feature_offset + fid
                                )
                            )
                            + fold
                        ),
                        q,
                    )
                    continue

                # DEVIATION (numerics.mojo): CatBoost uses atomicAdd when
                # blockCount > 1 and a plain store otherwise. The atomic is
                # order-nondeterministic and is what `NUMERIC_IDENTICAL`
                # replaces with a fixed-point accumulator. Direct store here;
                # the multi-block flush lands with the replication work.
                dst.unsafe_store(fold, val)
