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

from std.atomic import Atomic, Ordering
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.gpu.intrinsics import ldg

from mojo_only.kernel_matrix import (
    lane_width_for,
    TARGET_COLUMN,
    deterministic_flush_for,
    requires_uniform_iteration_for,
)
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier, syncwarp

from mojo_only.kernel_matrix import (
    HIST_SMEM_SHARED2_I32,
    K_HIST_ONE_BYTE,
    TARGET_COLUMN,
    block_size_for,
    hist2_block_size_for,
    hist_floats_per_thread_for,
    hist_smem_mode_for,
)
from gbdt.methods.greedy_subsets_searcher.kernel.histogram_utils import (
    hist2_dither,
    hist2_quantize,
    hist2_smem_add,
)


#: READ FROM THE MATRIX. Theirs is 384; Apple's 32 KB over 32 floats per
#: thread yields 256, which is exactly 32,768 bytes and keeps their per-warp
#: slice arithmetic intact at 8 warps of 1024 floats.
#: Same build mode as `hist_binary.mojo`; the flush follows the matrix.
comptime BUILD_MODE = GLOBAL_NUMERIC_MODE

#: Lanes moving in lockstep. READ FROM THE MATRIX, not pinned here.
#:
#: This used to be a literal 32 in this file and in three others, with a
#: comment pointing at `kernel_matrix.column_lane_width` for why AMD's 64
#: must not reach it. That is a matrix row that EXISTS and was bypassed:
#: `column_lane_width` had fifteen call sites and every one of them was
#: inside the matrix itself or the table printer, so changing
#: `TARGET_COLUMN` to AMD would have compiled and silently kept 32 while the
#: replication geometry assumed a 32-wide slice on a 64-wide wavefront.
#: One number now flows through, which is the point of having the table.
comptime LANE_WIDTH = lane_width_for[
    TARGET_COLUMN, BUILD_MODE == NUMERIC_IDENTICAL
]()

comptime ONE_BYTE_BLOCK_SIZE = block_size_for[K_HIST_ONE_BYTE, TARGET_COLUMN]()

#: `GetHistSize()` = `BlockSize * 32`, the 32 from the matrix.
comptime ONE_BYTE_HIST_SIZE = ONE_BYTE_BLOCK_SIZE * hist_floats_per_thread_for[
    K_HIST_ONE_BYTE
]()

#: `TPointHistOneByte::Unroll(ECIndexLoadType)` (`hist_one_byte.cu:30-37`),
#: which returns 2 below Volta and 4 on everything after it.
#:
#: THIS FILE'S OWN TABLE. It was 2, cited as "`Unroll` for
#: `__CUDA_ARCH__ >= 700`", and 2 is the value on the OTHER side of that
#: test. Each of the three histogram kernels has a different table --
#: 4/1/2 for binary (`hist_binary.cu:18-26`), 4/1 for half byte
#: (`hist_half_byte.cu:19-25`), 2/4 here -- and they share the load path,
#: not the unroll.
#:
#: SCHEDULING row: it changes how many loads are in flight, not what is
#: summed into what. It does widen `ALIGN_SIZE` to
#: `LOAD_SIZE * LANE_WIDTH * UNROLL`, which is theirs as well.
comptime ONE_BYTE_UNROLL = 4

#: Their unroll and load width, named as the loop expects them.
comptime UNROLL = ONE_BYTE_UNROLL

#: `ELoadSize::FourElements` (`hist_one_byte.cu:43-47`). CatBoost picks it on
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

#: The `hist_smem_mode_for` matrix row, cached the way
#: `hist_2_one_byte_base.mojo` caches it as `HIST2_SMEM_MODE`. It is ONE
#: row for the whole one-byte family: both families take 32 floats of
#: shared memory per thread, so Apple's 32 KB wall and the measured 1.94x
#: of the 2-warp-shared Int32 slice (scratchpad `histshare_probe.mojo`)
#: apply to the PASS kernels exactly as they applied to hist_2.
comptime ONE_BYTE_SMEM_MODE = hist_smem_mode_for[
    TARGET_COLUMN, BUILD_MODE == NUMERIC_IDENTICAL
]()


def one_byte_block_size[smem_mode: Int]() -> Int:
    """The PASS family's block, per accumulation mode.

    Float mode is `ONE_BYTE_BLOCK_SIZE` unchanged (their 384, 256 under
    Apple's 32 KB). The shared-Int32 mode reads `hist2_block_size_for`,
    which is the same 32-floats-per-thread arithmetic -- 64 bytes/thread
    under 2-warp sharing, capped at the 512 that exactly fills 32 KB. The
    two kernel keys resolve to the same float block (`block_size_for`
    treats `K_HIST_ONE_BYTE` and `K_HIST_2_ONE_BYTE` as one geometry), so
    the row is the family's, not hist_2's.
    """
    if smem_mode == HIST_SMEM_SHARED2_I32:
        return hist2_block_size_for[TARGET_COLUMN, smem_mode]()
    return ONE_BYTE_BLOCK_SIZE


def one_byte_smem_slots[smem_mode: Int]() -> Int:
    """`GetHistSize()` per mode: 32 floats/thread warp-private, or one
    1024-slot slice per WARP PAIR under `HIST_SMEM_SHARED2_I32`."""
    comptime block = one_byte_block_size[smem_mode]()
    if smem_mode == HIST_SMEM_SHARED2_I32:
        return (block // 64) * 1024
    return block * hist_floats_per_thread_for[K_HIST_ONE_BYTE]()


def one_byte_acc_dtype[smem_mode: Int]() -> DType:
    """Float32 shared memory in their design; the 2-warp-shared slice
    variant is Int32 fixed point (`hist2_smem_add`)."""
    if smem_mode == HIST_SMEM_SHARED2_I32:
        return DType.int32
    return DType.float32



def one_byte_slice_offset[bits: Int, smem_mode: Int](tid: Int) -> Int:
    """`TPointHistOneByte::SliceOffset()`, copied.

        const int warpOffset = 1024 * (threadIdx.x / 32);
        const int blocks = 8 >> InnerHistBitsCount;
        const int innerHistStart =
            threadIdx.x & ((blocks - 1) << (InnerHistBitsCount + 2));
        return warpOffset + innerHistStart;

    `blocks` shrinks as the feature widens: a 5-bit feature gets 8 private
    sub-copies per warp, an 8-bit feature gets 1. That is the same trade as
    the pass loop, seen from the memory side.

    THE 1024 IS THEIRS AND IT IS 32-LANE. It is 32 lanes times the 32 floats
    per thread this accumulator takes, and `blocks = 8 >> InnerHistBitsCount`
    cuts that warp into up to eight 4-lane sub-copies, which is 32 lanes
    again. `Reduce`'s stage 1 folds at the same 1024 (`hist_one_byte.cu:182`).
    A 64-lane wavefront needs a 2048-float slice and sixteen sub-copies, and
    CatBoost never wrote that layout, so the assert makes it a compile error
    rather than two warps quietly sharing one private copy. A 64-lane
    column never instantiates this family: `greedy_one_byte_fixed_for`
    (DEVIATION 1901) routes its one-byte work to the fused 8-bit kernel at
    the dispatch, and the assert stays for whatever reaches this file some
    other way.
    """
    comptime assert LANE_WIDTH == 32, (
        "the one-byte accumulator's slice layout is 32-lane by construction"
        " (`hist_one_byte.cu:55-59`, and `Reduce`'s warpHistSize at `:182`);"
        " write the wide-wavefront layout before letting LANE_WIDTH be 64"
    )
    comptime inner_bits = bits - 5

    # ================= DEVIATION BLOCK =================
    # Under `HIST_SMEM_SHARED2_I32` the slice a thread lands in is keyed by
    # its warp PAIR (`tid // 64`) instead of its warp (`tid // 32`): two
    # warps deliberately share one 1024-slot slice through the Int32 atomics
    # of `hist2_smem_add`, exactly as `hist2_slice_offset` does it. The
    # within-slice arithmetic -- `innerHistStart`, the sub-copy blocks --
    # is theirs unchanged.
    # ===================================================
    var warp_offset: Int

    @parameter
    if smem_mode == HIST_SMEM_SHARED2_I32:
        warp_offset = 1024 * (tid // 64)
    else:
        warp_offset = 1024 * (tid // LANE_WIDTH)
    comptime blocks = 8 >> inner_bits
    var inner_hist_start = tid & ((blocks - 1) << (inner_bits + 2))
    return warp_offset + inner_hist_start


def one_byte_bin(ci: UInt32, tid: Int, i: Int) -> Int:
    """The BIN this lane reads on iteration `i`, before any slot arithmetic.

        int f = (threadIdx.x + i) & 3;
        int bin = (ci >> (24 - 8 * f)) & 255;
            (`hist_one_byte.cu:80-82`)

    Four features per word here, so the rotation is `& 3` rather than `& 7`.
    Factored out because the slot, the pass and the `statToAdd` guard are
    three readings of the SAME bin and had drifted into three copies of the
    shift.
    """
    var f = (tid + i) & 3
    return Int((ci >> UInt32(24 - 8 * f)) & UInt32(255))


def one_byte_bin_offset[bits: Int](ci: UInt32, tid: Int, i: Int) -> Int:
    """The slot for iteration `i`, as pure arithmetic (PORTING.md 10).

        const int higherBin = (bin >> 5) & mask;
        int offset = 4 * higherBin + f + ((bin & 31) << 5);
            (`hist_one_byte.cu:87-90`)
    """
    comptime inner_bits = bits - 5
    comptime mask = (1 << inner_bits) - 1
    var f = (tid + i) & 3
    var bin = one_byte_bin(ci, tid, i)
    var higher_bin = (bin >> 5) & mask
    return 4 * higher_bin + f + ((bin & 31) << 5)


def add_one_byte_point[
    bits: Int, dt: DType
](
    ci: UInt32,
    stat: Float32,
    qstat: Int32,
    tid: Int,
    slice_base: Int,
    smem: UnsafePointer[
        Scalar[dt],
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

    Their sync is WARP-LOCAL and so is ours:

        auto syncTile = tiled_partition<32>(this_thread_block());
        ...
        syncTile.sync();

    `SliceOffset()` gives every warp its own private copy of the histogram,
    which is why `AddPoint` needs no atomics: the only ordering required is
    among the 32 lanes sharing that copy. A threadgroup `barrier()` would
    also be correct and is strictly more expensive, and at 8 bits it costs
    4 features x 8 passes = 32 threadgroup barriers PER POINT.

    `syncwarp` was thought unavailable and is not: it imports from
    `max.gpu.sync`. Warp SHUFFLES really are missing in Mojo 1.0, and the two
    were conflated.

    Because the sync is warp-local, callers no longer need a uniform trip
    count across the BLOCK, only across each warp. The peel and the main loop
    keep their uniform counts anyway, which is stricter than required and
    costs nothing.
    """

    @parameter
    for i in range(4):
        var slot = slice_base + one_byte_bin_offset[bits](ci, tid, i)
        comptime inner_bits = bits - 5

        # `const float statToAdd = (bin >> Bits) == 0 ? t : 0;`
        # (`hist_one_byte.cu:85`). A BIN WIDER THAN THE FEATURE CONTRIBUTES
        # NOTHING, and this guard was missing outright.
        #
        # It is not cosmetic below 8 bits. The slot keeps only `bin & 31` and
        # masks `higherBin` to `InnerHistBitsCount` bits, so a bin at or above
        # `1 << Bits` does not fall off the end of the histogram -- it ALIASES
        # onto a real bin of the same feature and is counted there. At
        # `Bits == 8` the test is always true and this costs nothing, which is
        # why the omission could sit under the 8-bit path unseen.
        var stat_to_add = Float32(0.0)
        var q_to_add = Int32(0)
        if (one_byte_bin(ci, tid, i) >> bits) == 0:
            stat_to_add = stat
            q_to_add = qstat

        # The float arm keeps the pass structure and the warp-local sync
        # exactly as theirs: the passes exist to serialize lanes whose
        # PLAIN float adds would collide in the shared slice
        # (`hist_one_byte.cu:96-101`), and at 8 bits that is 8 gated
        # passes with a syncwarp each, per point, per feature.
        #
        # ================= DEVIATION BLOCK =================
        # The Int32 arm SKIPS the pass loop: its add is an ATOMIC
        # (`hist2_smem_add`'s `Atomic.fetch_add`), so lane collisions are
        # resolved by the hardware and the serialization protects
        # nothing. Integer addition is associative, so the histogram is
        # BIT-IDENTICAL with or without the passes -- `check-hist2`'s
        # bits-8 section compares this arm cell-for-cell against the
        # float arm and the host tally, and the 254-border oracle gates
        # the splits. The pass count is why 254 borders cost ~2x what
        # 128 did (8 passes vs hist_2's 4 at 7 bits) while CatBoost's
        # CPU pays nothing extra; measured standings live in RESUME.
        # NVIDIA/AMD float columns compile the pass loop unchanged.
        # ===================================================
        @parameter
        if dt == DType.int32:
            hist2_smem_add[dt](smem, slot, stat_to_add, q_to_add)
        elif inner_bits == 0:
            syncwarp()
            hist2_smem_add[dt](smem, slot, stat_to_add, q_to_add)
        else:
            var higher = one_byte_higher_bin[bits](ci, tid, i)
            comptime mask = (1 << inner_bits) - 1

            @parameter
            for kk in range(1 << inner_bits):
                var p = ((tid >> 2) + kk) & mask
                syncwarp()
                if p == higher:
                    hist2_smem_add[dt](smem, slot, stat_to_add, q_to_add)


def one_byte_higher_bin[bits: Int](ci: UInt32, tid: Int, i: Int) -> Int:
    """`higherBin` alone: which PASS of the inner-bits loop may write
    (`hist_one_byte.cu:88`)."""
    comptime inner_bits = bits - 5
    comptime mask = (1 << inner_bits) - 1
    return (one_byte_bin(ci, tid, i) >> 5) & mask


def one_byte_hist_kernel[bits: Int, smem_mode: Int](
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
    fixed_scale_ptr: MutPointer[Float32, MutAnyOrigin],
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
    var fixed_scale = fixed_scale_ptr.unsafe_load(0)
    var f_count_in = Int(f_count_in32)
    var bins_line_size = Int(bins_line_size_in)
    var stat_line_size = Int(stat_line_size_in)
    var leaf_count = Int(leaf_count_in)
    var stat_count = Int(stat_count_in)
    var tid = Int(thread_idx.x)

    comptime BLOCK = one_byte_block_size[smem_mode]()
    comptime SLOTS = one_byte_smem_slots[smem_mode]()
    comptime DT = one_byte_acc_dtype[smem_mode]()

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
        BLOCK // LANE_WIDTH
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
        SLOTS,
        Scalar[DT],
        address_space = AddressSpace.SHARED,
    ]()
    var z = tid
    while z < SLOTS:
        smem[z] = Scalar[DT](0)
        z += BLOCK
    barrier()
    var slice_base = one_byte_slice_offset[bits, smem_mode](tid)

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
        var hq = Int32(0)
        if local_block_idx == 0 and pe < head_len:
            # `Ldg(bins, idx)` and `Ldg(stats, idx)`
            # (`compute_hist_loop_one_stat.cuh:80-81`). `Ldg` is
            # `cub::ThreadLoad<cub::LOAD_LDG>` (`kernel_helpers.cuh:180`),
            # the read-only non-coherent load; `std.gpu.intrinsics.ldg` is
            # its Mojo spelling.
            hb = ldg(bins_p + (p_offset + pe))
            hs = ldg(stats_p + (p_offset + pe))

            @parameter
            if DT == DType.int32:
                hq = hist2_quantize(
                    hs, fixed_scale, hist2_dither(p_offset + pe)
                )
        add_one_byte_point[bits, DT](hb, hs, hq, tid, slice_base, smem)

        var tb = UInt32(0)
        var ts = Float32(0.0)
        var tq = Int32(0)
        if local_block_idx == 0 and pe < tail_len:
            # `Ldg(bins, tailOffset + idx)` and `Ldg(stats, tailOffset + idx)`
            # (`compute_hist_loop_one_stat.cuh:98-99`).
            tb = ldg(bins_p + (tail_start + pe))
            ts = ldg(stats_p + (tail_start + pe))

            @parameter
            if DT == DType.int32:
                tq = hist2_quantize(
                    ts, fixed_scale, hist2_dither(tail_start + pe)
                )
        add_one_byte_point[bits, DT](tb, ts, tq, tid, slice_base, smem)
        pe += BLOCK

    # the striped loop sees the ALIGNED MIDDLE only
    var aligned_offset = p_offset + head_len
    var aligned_size = body_size - tail_len

    var warps_per_block = BLOCK // LANE_WIDTH
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
    # The dither key for the Int32 arm: the storage POSITION each element
    # was loaded from, exactly as `hist_2_one_byte_base.mojo` keys it.
    var pos_base = base

    for it in range(max_iters):
        var active = it < iter_count
        # Their two unrolled loops: gather the batch, then add it. Kept in
        # that order because it is what keeps the loads in flight.
        var local_bins = InlineArray[UInt32, POINTS_PER_ITER](fill=0)
        var local_stats = InlineArray[Float32, POINTS_PER_ITER](fill=0)
        var local_q = InlineArray[Int32, POINTS_PER_ITER](fill=0)

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
                # `Ldg((uint4*) bins, warpSize * k)` and
                # `Ldg((float4*) stats, warpSize * k)`
                # (`compute_hist_loop_one_stat.cuh:368`, `:374`).
                # `alignment=4` is OURS and is stated rather than
                # inherited. Their `Ldg((uint4*) ...)` is a 16-byte load
                # because their column stride is `AlignedColumnSize()`; ours
                # is `n_rows` (`greedy_search_helper.mojo`), so a column base
                # is not 4-element aligned in general and the width-4 load
                # keeps the element alignment the plain load already assumed.
                var vb = ldg[width=LOAD_SIZE, alignment=4](
                    b_ptr + LANE_WIDTH * LOAD_SIZE * k
                )
                var vs = ldg[width=LOAD_SIZE, alignment=4](
                    s_ptr + LANE_WIDTH * LOAD_SIZE * k
                )

                @parameter
                for e in range(LOAD_SIZE):
                    local_bins[k * LOAD_SIZE + e] = vb[e]
                    local_stats[k * LOAD_SIZE + e] = vs[e]

                    @parameter
                    if DT == DType.int32:
                        var u = hist2_dither(
                            pos_base + LANE_WIDTH * LOAD_SIZE * k + e
                        )
                        local_q[k * LOAD_SIZE + e] = hist2_quantize(
                            vs[e], fixed_scale, u
                        )
            else:
                # No row: contribute zero. Harmless, and it keeps this lane
                # inside every barrier below.
                @parameter
                for e in range(LOAD_SIZE):
                    local_bins[k * LOAD_SIZE + e] = UInt32(0)
                    local_stats[k * LOAD_SIZE + e] = Float32(0.0)
                    local_q[k * LOAD_SIZE + e] = Int32(0)

        # `hist.AddPoints<loadSize * N>(...)`: every point the batch loaded,
        # through the same `AddPoint` the peel calls.
        @parameter
        for k in range(POINTS_PER_ITER):
            add_one_byte_point[bits, DT](
                local_bins[k], local_stats[k], local_q[k], tid, slice_base,
                smem,
            )

        b_ptr += stripe_size
        s_ptr += stripe_size
        pos_base += stripe_size

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
        var acc = Scalar[DT](0)
        var j = start
        while j < SLOTS:
            acc += smem[j]
            j += WARP_HIST_SIZE
        smem[WARP_HIST_SIZE + start] = acc
        start += BLOCK
    barrier()

    # Stage 2: UN-SCRAMBLE. The slot for feature `f`, bin `b` is
    # `((b & 31) << 5) + 4 * (b >> 5) + f + innerHistStart`, so the sub-copies
    # a warp keeps at `innerHistStart` in `{0, blockSize, 2*blockSize, ...}`
    # have to be gathered per feature. `i` here is the FEATURE.
    comptime warp_hist_block_count = 8 >> inner_bits
    comptime sub_block = 4 * (1 << inner_bits)
    var fold_r = tid
    var sums = InlineArray[Scalar[DT], 4](fill=Scalar[DT](0))
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
            # `blockIdx.y`, DENSE, not `partIds[blockIdx.y]`. See the
            # note above `entries_per_leaf`.
            var dst_base = (
                device_offset
                + Int(block_idx.y) * entries_per_leaf
                + Int(block_idx.z) * group_size
                + Int(feature_fold_offset.unsafe_load(feature_offset + fid))
            )
            var dst = bin_sums + dst_base

            var cell = smem[fid * hist_size_bins + fold]

            @parameter
            if DT == DType.int32:
                # The shared-Int32 arm: the cell is already fixed point at
                # `fixed_scale`, and the flush follows
                # `hist2_add_to_global_memory`'s DEVIATION BLOCK exactly --
                # the multi-block branch adds the Int32 cell DIRECTLY into
                # the accumulator (exact, no dequantize/requantize round
                # trip), the single-block branch stores the dequantized
                # value, and `cell != 0` is their never-write-a-zero-cell
                # guard in the integer domain. The launch helper runs
                # `fixed_to_float_kernel` on every one-byte block whenever
                # this arm is compiled, so the accumulator drains under
                # FAST exactly as it does under IDENTICAL.
                var q = rebind[Scalar[DType.int32]](cell)
                if q != Int32(0):
                    if active_block_count > 1:
                        # DEVIATION 1898: upstream's atomicAdd is relaxed; the
                        # non-Apple Mojo default is seq_cst.
                        _ = Atomic.fetch_add[ordering = Ordering.RELAXED](
                            acc_i32.unsafe_offset(dst_base + fold), q
                        )
                    else:
                        # through the accumulator, like hist_2's base
                        # family: q is already fixed point, bits cannot
                        # move, and the float scratch goes dead here.
                        acc_i32.unsafe_store(dst_base + fold, q)
            else:
                var val = rebind[Scalar[DType.float32]](cell)
                if abs(val) > Float32(1e-20):
                    # THE FLUSH. `AddToGlobalMemory`, their multi-block branch
                    # (`hist_one_byte.cu:253-259`):
                    #
                    #     if (blockCount > 1) { atomicAdd(dst + fold, val); }
                    #     else               { dst[fold] = val; }
                    #
                    # A plain store is only correct when ONE block owns the
                    # partition. Replicating blocks across a partition, which
                    # is what fills the machine, makes every block hold a
                    # PARTIAL histogram, and partials must be summed.
                    comptime det = deterministic_flush_for[
                        TARGET_COLUMN, BUILD_MODE == NUMERIC_IDENTICAL
                    ]()

                    @parameter
                    if det:
                        if active_block_count > 1:
                            # `NUMERIC_IDENTICAL`. Partials sum as Int32
                            # through an integer atomic, which is
                            # associative, so the histogram does not depend
                            # on which block lands first. That is the
                            # property CatBoost's float atomic gives up.
                            var q = Int32(val * fixed_scale)
                            # DEVIATION 1898: upstream's atomicAdd is relaxed;
                            # the non-Apple Mojo default is seq_cst.
                            _ = Atomic.fetch_add[ordering = Ordering.RELAXED](
                                acc_i32.unsafe_offset(dst_base + fold), q
                            )
                        else:
                            dst.unsafe_store(fold, val)
                    else:
                        # `atomicAdd(dst + fold, val)`, theirs verbatim.
                        #
                        # THIS BRANCH USED TO BE A PLAIN STORE. The condition
                        # was written `if det and active_block_count > 1`, so
                        # with `det` false and the block replicated it fell
                        # through to `dst[fold] = val` and left ONE block's
                        # partial histogram standing while every other
                        # block's was overwritten. It was a store where
                        # CatBoost has an atomic, which is not a deviation,
                        # it is a wrong answer.
                        #
                        # This is what held `deterministic_flush_for` at a
                        # hardcoded `True`: with the fixed-point path forced
                        # on, `det` was never false and the store was never
                        # taken, so flipping the row to the mode's answer was
                        # reported as "the float atomic breaks the mixed
                        # tree" -- 16 non-empty leaves at depth 4 dropping to
                        # 4. It was not the atomic. It was this branch not
                        # having one.
                        if active_block_count > 1:
                            # DEVIATION 1898: upstream's atomicAdd is relaxed;
                            # the non-Apple Mojo default is seq_cst.
                            _ = Atomic.fetch_add[ordering = Ordering.RELAXED](
                                dst.unsafe_offset(fold), val
                            )
                        else:
                            dst.unsafe_store(fold, val)


def one_byte_hist_gather_kernel[bits: Int, smem_mode: Int](
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
    fixed_scale_ptr: MutPointer[Float32, MutAnyOrigin],
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
    var fixed_scale = fixed_scale_ptr.unsafe_load(0)
    var f_count_in = Int(f_count_in32)
    var bins_line_size = Int(bins_line_size_in)
    var stat_line_size = Int(stat_line_size_in)
    var leaf_count = Int(leaf_count_in)
    var stat_count = Int(stat_count_in)
    var tid = Int(thread_idx.x)

    comptime BLOCK = one_byte_block_size[smem_mode]()
    comptime SLOTS = one_byte_smem_slots[smem_mode]()
    comptime DT = one_byte_acc_dtype[smem_mode]()

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
        BLOCK // LANE_WIDTH
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
        SLOTS,
        Scalar[DT],
        address_space = AddressSpace.SHARED,
    ]()
    var z = tid
    while z < SLOTS:
        smem[z] = Scalar[DT](0)
        z += BLOCK
    barrier()
    var slice_base = one_byte_slice_offset[bits, smem_mode](tid)

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
        var hq = Int32(0)
        if local_block_idx == 0 and pe < head_len:
            # `Ldg(indices, idx)`, `Ldg(cindex, loadIdx)`, `Ldg(stats, idx)`
            # (`compute_hist_loop_one_stat.cuh:130-132`). All three streams
            # are read-only for the kernel's lifetime, which is what `Ldg`
            # declares.
            var hrow = Int(ldg(indices + (p_offset + pe)))
            hb = ldg(cindex_p + hrow)
            hs = ldg(stats_p + (p_offset + pe))

            @parameter
            if DT == DType.int32:
                hq = hist2_quantize(
                    hs, fixed_scale, hist2_dither(p_offset + pe)
                )
        add_one_byte_point[bits, DT](hb, hs, hq, tid, slice_base, smem)

        var tb = UInt32(0)
        var ts = Float32(0.0)
        var tq = Int32(0)
        if local_block_idx == 0 and pe < tail_len:
            # `Ldg(indices, tailOffset + idx)`, `Ldg(cindex, loadIdx)`,
            # `Ldg(stats, tailOffset + idx)`
            # (`compute_hist_loop_one_stat.cuh:149-151`).
            var trow = Int(ldg(indices + (tail_start + pe)))
            tb = ldg(cindex_p + trow)
            ts = ldg(stats_p + (tail_start + pe))

            @parameter
            if DT == DType.int32:
                tq = hist2_quantize(
                    ts, fixed_scale, hist2_dither(tail_start + pe)
                )
        add_one_byte_point[bits, DT](tb, ts, tq, tid, slice_base, smem)
        pe += BLOCK

    var aligned_offset = p_offset + head_len
    var aligned_size = body_size - tail_len

    var warps_per_block = BLOCK // LANE_WIDTH
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
    # The dither key for the Int32 arm: the storage POSITION, not the
    # gathered document id, exactly as `hist_2_one_byte_base.mojo`'s gather
    # arm keys it.
    var pos_base = base

    for it in range(max_iters):
        var active = it < iter_count
        # Their two unrolled loops: gather the batch, then add it. Kept in
        # that order because it is what keeps the loads in flight.
        var local_bins = InlineArray[UInt32, POINTS_PER_ITER](fill=0)
        var local_stats = InlineArray[Float32, POINTS_PER_ITER](fill=0)
        var local_q = InlineArray[Int32, POINTS_PER_ITER](fill=0)

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
                # `Ldg((int4*) indices, warpSize * k)` and
                # `Ldg((float4*) stats, warpSize * k)`
                # (`compute_hist_loop_one_stat.cuh:407`, `:423`).
                # `alignment=4` is OURS and is stated rather than
                # inherited. Their `Ldg((uint4*) ...)` is a 16-byte load
                # because their column stride is `AlignedColumnSize()`; ours
                # is `n_rows` (`greedy_search_helper.mojo`), so a column base
                # is not 4-element aligned in general and the width-4 load
                # keeps the element alignment the plain load already assumed.
                var vi = ldg[width=LOAD_SIZE, alignment=4](
                    i_ptr + LANE_WIDTH * LOAD_SIZE * k
                )
                var vs = ldg[width=LOAD_SIZE, alignment=4](
                    s_ptr + LANE_WIDTH * LOAD_SIZE * k
                )

                @parameter
                for e in range(LOAD_SIZE):
                    # The compressed index is never permuted, so after a
                    # reorder a position no longer names a row and the index
                    # array is the only way back. See hist_binary.
                    # `localBins[k].x = Ldg(cindex, localIndices[k].x)` and
                    # its y, z, w twins
                    # (`compute_hist_loop_one_stat.cuh:415-418`). Scalar,
                    # because a gather has no vector form.
                    local_bins[k * LOAD_SIZE + e] = ldg(
                        cindex_p + Int(vi[e])
                    )
                    local_stats[k * LOAD_SIZE + e] = vs[e]

                    @parameter
                    if DT == DType.int32:
                        var u = hist2_dither(
                            pos_base + LANE_WIDTH * LOAD_SIZE * k + e
                        )
                        local_q[k * LOAD_SIZE + e] = hist2_quantize(
                            vs[e], fixed_scale, u
                        )
            else:
                @parameter
                for e in range(LOAD_SIZE):
                    local_bins[k * LOAD_SIZE + e] = UInt32(0)
                    local_stats[k * LOAD_SIZE + e] = Float32(0.0)
                    local_q[k * LOAD_SIZE + e] = Int32(0)

        # `hist.AddPoints<loadSize * N>(...)`
        @parameter
        for k in range(POINTS_PER_ITER):
            add_one_byte_point[bits, DT](
                local_bins[k], local_stats[k], local_q[k], tid, slice_base,
                smem,
            )

        i_ptr += stripe_size
        s_ptr += stripe_size
        pos_base += stripe_size

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
        var acc = Scalar[DT](0)
        var j = start
        while j < SLOTS:
            acc += smem[j]
            j += WARP_HIST_SIZE
        smem[WARP_HIST_SIZE + start] = acc
        start += BLOCK
    barrier()

    # Stage 2: UN-SCRAMBLE. The slot for feature `f`, bin `b` is
    # `((b & 31) << 5) + 4 * (b >> 5) + f + innerHistStart`, so the sub-copies
    # a warp keeps at `innerHistStart` in `{0, blockSize, 2*blockSize, ...}`
    # have to be gathered per feature. `i` here is the FEATURE.
    comptime warp_hist_block_count = 8 >> inner_bits
    comptime sub_block = 4 * (1 << inner_bits)
    var fold_r = tid
    var sums = InlineArray[Scalar[DT], 4](fill=Scalar[DT](0))
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
            # `blockIdx.y`, DENSE, not `partIds[blockIdx.y]`. See the
            # note above `entries_per_leaf`.
            var dst_base = (
                device_offset
                + Int(block_idx.y) * entries_per_leaf
                + Int(block_idx.z) * group_size
                + Int(feature_fold_offset.unsafe_load(feature_offset + fid))
            )
            var dst = bin_sums + dst_base

            var cell = smem[fid * hist_size_bins + fold]

            @parameter
            if DT == DType.int32:
                # The shared-Int32 arm: the cell is already fixed point at
                # `fixed_scale`, and the flush follows
                # `hist2_add_to_global_memory`'s DEVIATION BLOCK exactly --
                # the multi-block branch adds the Int32 cell DIRECTLY into
                # the accumulator (exact, no dequantize/requantize round
                # trip), the single-block branch stores the dequantized
                # value, and `cell != 0` is their never-write-a-zero-cell
                # guard in the integer domain. The launch helper runs
                # `fixed_to_float_kernel` on every one-byte block whenever
                # this arm is compiled, so the accumulator drains under
                # FAST exactly as it does under IDENTICAL.
                var q = rebind[Scalar[DType.int32]](cell)
                if q != Int32(0):
                    if active_block_count > 1:
                        # DEVIATION 1898: upstream's atomicAdd is relaxed; the
                        # non-Apple Mojo default is seq_cst.
                        _ = Atomic.fetch_add[ordering = Ordering.RELAXED](
                            acc_i32.unsafe_offset(dst_base + fold), q
                        )
                    else:
                        # through the accumulator, like hist_2's base
                        # family: q is already fixed point, bits cannot
                        # move, and the float scratch goes dead here.
                        acc_i32.unsafe_store(dst_base + fold, q)
            else:
                var val = rebind[Scalar[DType.float32]](cell)
                if abs(val) > Float32(1e-20):
                    # THE FLUSH. `AddToGlobalMemory`, their multi-block branch
                    # (`hist_one_byte.cu:253-259`):
                    #
                    #     if (blockCount > 1) { atomicAdd(dst + fold, val); }
                    #     else               { dst[fold] = val; }
                    #
                    # A plain store is only correct when ONE block owns the
                    # partition. Replicating blocks across a partition, which
                    # is what fills the machine, makes every block hold a
                    # PARTIAL histogram, and partials must be summed.
                    comptime det = deterministic_flush_for[
                        TARGET_COLUMN, BUILD_MODE == NUMERIC_IDENTICAL
                    ]()

                    @parameter
                    if det:
                        if active_block_count > 1:
                            # `NUMERIC_IDENTICAL`. Partials sum as Int32
                            # through an integer atomic, which is
                            # associative, so the histogram does not depend
                            # on which block lands first. That is the
                            # property CatBoost's float atomic gives up.
                            var q = Int32(val * fixed_scale)
                            # DEVIATION 1898: upstream's atomicAdd is relaxed;
                            # the non-Apple Mojo default is seq_cst.
                            _ = Atomic.fetch_add[ordering = Ordering.RELAXED](
                                acc_i32.unsafe_offset(dst_base + fold), q
                            )
                        else:
                            dst.unsafe_store(fold, val)
                    else:
                        # `atomicAdd(dst + fold, val)`, theirs verbatim.
                        #
                        # THIS BRANCH USED TO BE A PLAIN STORE. The condition
                        # was written `if det and active_block_count > 1`, so
                        # with `det` false and the block replicated it fell
                        # through to `dst[fold] = val` and left ONE block's
                        # partial histogram standing while every other
                        # block's was overwritten. It was a store where
                        # CatBoost has an atomic, which is not a deviation,
                        # it is a wrong answer.
                        #
                        # This is what held `deterministic_flush_for` at a
                        # hardcoded `True`: with the fixed-point path forced
                        # on, `det` was never false and the store was never
                        # taken, so flipping the row to the mode's answer was
                        # reported as "the float atomic breaks the mixed
                        # tree" -- 16 non-empty leaves at depth 4 dropping to
                        # 4. It was not the atomic. It was this branch not
                        # having one.
                        if active_block_count > 1:
                            # DEVIATION 1898: upstream's atomicAdd is relaxed;
                            # the non-Apple Mojo default is seq_cst.
                            _ = Atomic.fetch_add[ordering = Ordering.RELAXED](
                                dst.unsafe_offset(fold), val
                            )
                        else:
                            dst.unsafe_store(fold, val)
