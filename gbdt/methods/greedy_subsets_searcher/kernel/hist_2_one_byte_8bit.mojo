"""Fused two-stat 8-bit histograms: the >128-bin arm CatBoost never wrote.

================= DEVIATION BLOCK (whole file) =================
NO CATBOOST COUNTERPART. Their one-byte ladder fuses two stats only up
to 128 bins (`hist_one_byte.cu:315-323`): at 8 bits a warp-PRIVATE
two-stat slice is `256 bins * 2 stats * 4 features * 4B = 8 KB` per
warp, which no shared-memory budget carries at their block sizes, so
129-255 bins fall back to the one-stat PASS family -- `gridDim.z =
statCount`, TWO full walks over the compressed index per level.

The Apple shared-Int32 arm removes the constraint their fallback exists
for: slices are SHARED and the adds are ATOMIC, so FOUR warps share one
8 KB slice and a 512-thread block holds 4 slices in exactly 32 KB. One
walk, both stats, half the launches, half the cindex traffic. Measured
standings live in RESUME; the arm exists for the
`HIST_SMEM_SHARED2_I32` matrix row and, at EVERY one-byte width, for
64-lane columns (`greedy_one_byte_fixed_for`, DEVIATION 1906: nothing
here indexes by hardware lane -- Int32 atomics, block barriers, uniform
trip counts, `H8_LANE` a logical striping constant -- so it is the one
one-byte kernel a 64-wide wavefront can run). 32-lane NVIDIA/AMD float
columns keep CatBoost's PASS(8) design verbatim, and the dispatch keys
on the rows.

EXACTNESS IS STRUCTURAL, NOT STATISTICAL: the quantized addends are the
same `hist2_quantize(stat, fixed_scale, dither(position))` values the
PASS family adds (same positions, same draws), and integer addition is
associative, so the fused histogram is BIT-IDENTICAL to the PASS
family's -- `check-hist2`'s bits-8 section compares the two cell for
cell, and the 254-border oracle gates the splits.

At 8 bits there is NO skip mark and NO out-of-range bin: the mark would
be `1 << 8 = 256` and a u8 cannot hold it, so the `(bin >> bits) == 0`
guard of the narrower widths is vacuously true and not carried.
=================================================================

Slice layout: `cell = (bin << 3) + (feature << 2 >> 1) + stat` --
spelled `(bin << 3) + (f << 1) + stat` below -- 2048 Int32 cells per
slice, slice per warp QUAD (`tid // 128`). The reduce folds the four
slices into slice 0 by the same residue-class argument as the PASS
family's stage 1, then one thread per fold writes both stats through
the same dual-branch flush every histogram kernel in this package uses.
"""

from std.atomic import Atomic, Ordering
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.gpu.intrinsics import ldg
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from gbdt.methods.greedy_subsets_searcher.kernel.histogram_utils import (
    hist2_dither,
    hist2_quantize,
)
from mojo_only.kernel_matrix import TARGET_COLUMN, lane_width_for

comptime H8_BLOCK = 512
comptime H8_SLICE = 2048
comptime H8_SLICES = 4
comptime H8_SMEM = H8_SLICE * H8_SLICES
comptime H8_LANE = 32
comptime H8_UNROLL = 4
comptime H8_LOAD = 4
comptime H8_POINTS = H8_UNROLL * H8_LOAD

#: `BlockLoadSize`: one aligned warp iteration times the block's warps.
comptime H8_MIN_DOCS = H8_LANE * H8_UNROLL * H8_LOAD * (H8_BLOCK // H8_LANE)


def h8_slice_base(tid: Int) -> Int:
    """One 2048-cell slice per warp QUAD: four warps' atomics share it."""
    return H8_SLICE * (tid // 128)


def h8_add_point(
    ci: UInt32,
    q1: Int32,
    q2: Int32,
    tid: Int,
    slice_base: Int,
    smem: UnsafePointer[
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
        origin=MutUntrackedOrigin,
    ],
):
    """Both stats of one point, four features per word, atomic adds only
    -- no pass loop and no write turns, for the same reason the Int32
    arms of the other kernels dropped theirs: atomics make the
    serialization protect nothing, and associativity keeps the bits."""

    @parameter
    for i in range(4):
        var f = (tid + i) & 3
        var bin = Int((ci >> UInt32(24 - 8 * f)) & UInt32(255))
        var cell = slice_base + (bin << 3) + (f << 1)
        # DEVIATION 1898: upstream's atomicAdd is relaxed; the non-Apple Mojo
        # default is seq_cst.
        _ = Atomic.fetch_add[ordering = Ordering.RELAXED](
            smem.unsafe_offset(cell), q1
        )
        _ = Atomic.fetch_add[ordering = Ordering.RELAXED](
            smem.unsafe_offset(cell + 1), q2
        )


def h8_reduce_and_flush(
    tid: Int,
    active_block_count: Int,
    feature_folds: MutPointer[UInt32, MutAnyOrigin],
    feature_fold_offset: MutPointer[UInt32, MutAnyOrigin],
    feature_group_offset: MutPointer[UInt32, MutAnyOrigin],
    feature_group_size: MutPointer[UInt32, MutAnyOrigin],
    feature_offset: Int,
    f_count: Int,
    leaf_count: Int,
    stat_count: Int,
    bin_sums: MutPointer[Float32, MutAnyOrigin],
    acc_i32: MutPointer[Int32, MutAnyOrigin],
    fixed_scale: Float32,
    smem: UnsafePointer[
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
        origin=MutUntrackedOrigin,
    ],
):
    """Fold the four slices into slice 0 (stage 1's residue-class
    argument: each residue of 2048 is owned by one thread, which reads
    every copy of it before writing it), then one thread per FOLD writes
    both stats through the dual-branch flush."""
    barrier()
    var start = tid
    while start < H8_SLICE:
        var acc = smem[start]

        @parameter
        for s in range(1, H8_SLICES):
            acc += smem[start + s * H8_SLICE]
        smem[start] = acc
        start += H8_BLOCK
    barrier()

    var fold = tid
    for fid in range(f_count):
        var folds = Int(feature_folds.unsafe_load(feature_offset + fid))
        if fold < folds:
            var group_offset = Int(
                feature_group_offset.unsafe_load(feature_offset + fid)
            )
            var group_size = Int(
                feature_group_size.unsafe_load(feature_offset + fid)
            )
            var fold_off = Int(
                feature_fold_offset.unsafe_load(feature_offset + fid)
            )
            var device_offset = group_offset * stat_count * leaf_count
            var entries_per_leaf = stat_count * group_size

            @parameter
            for stat in range(2):
                var q = smem[(fold << 3) + (fid << 1) + stat]
                if q != Int32(0):
                    var dst_base = (
                        device_offset
                        + Int(block_idx.y) * entries_per_leaf
                        + stat * group_size
                        + fold_off
                    )
                    if active_block_count > 1:
                        # DEVIATION 1898: upstream's atomicAdd is relaxed; the
                        # non-Apple Mojo default is seq_cst.
                        _ = Atomic.fetch_add[ordering = Ordering.RELAXED](
                            acc_i32.unsafe_offset(dst_base + fold), q
                        )
                    else:
                        # through the accumulator, like the base family:
                        # q is already fixed point, bits cannot move,
                        # and the float scratch goes dead on this arm.
                        acc_i32.unsafe_store(dst_base + fold, q)


def hist2_8bit_kernel(
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
    """Direct loads (depth 0). Same partition walk as the PASS family's
    direct kernel; the differences are the second stat plane loaded per
    point (as the hist_2 kernels load it) and grid z = 1."""
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

    var bins_p = bins + Int(cindex_base_in) + bins_line_size * (
        Int(block_idx.x) // max_blocks_per_part
    )

    var local_block_idx = Int(block_idx.x) % max_blocks_per_part
    var active_block_count = min(
        (p_size + H8_MIN_DOCS - 1) // H8_MIN_DOCS, max_blocks_per_part
    )
    if local_block_idx >= active_block_count:
        return

    var smem = stack_allocation[
        H8_SMEM,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var z = tid
    while z < H8_SMEM:
        smem[z] = Int32(0)
        z += H8_BLOCK
    barrier()
    var slice_base = h8_slice_base(tid)

    comptime ALIGN_SIZE = H8_LOAD * H8_LANE * H8_UNROLL
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
        if local_block_idx == 0 and pe < head_len:
            var hb = ldg(bins_p + (p_offset + pe))
            var u = hist2_dither(p_offset + pe)
            var hq1 = hist2_quantize(
                ldg(stats + (p_offset + pe)), fixed_scale, u
            )
            var hq2 = hist2_quantize(
                ldg(stats + (stat_line_size + p_offset + pe)),
                fixed_scale, u,
            )
            h8_add_point(hb, hq1, hq2, tid, slice_base, smem)
        if local_block_idx == 0 and pe < tail_len:
            var tb = ldg(bins_p + (tail_start + pe))
            var u = hist2_dither(tail_start + pe)
            var tq1 = hist2_quantize(
                ldg(stats + (tail_start + pe)), fixed_scale, u
            )
            var tq2 = hist2_quantize(
                ldg(stats + (stat_line_size + tail_start + pe)),
                fixed_scale, u,
            )
            h8_add_point(tb, tq1, tq2, tid, slice_base, smem)
        pe += H8_BLOCK

    var aligned_offset = p_offset + head_len
    var aligned_size = body_size - tail_len
    var warps_per_block = H8_BLOCK // H8_LANE
    var global_warp_id = local_block_idx * warps_per_block + (
        tid // H8_LANE
    )
    var entries_per_warp = H8_LANE * H8_UNROLL * H8_LOAD
    var stripe_size = entries_per_warp * warps_per_block * active_block_count
    var remaining = max(aligned_size - global_warp_id * entries_per_warp, 0)
    var local_idx = (tid & (H8_LANE - 1)) * H8_LOAD
    var base = aligned_offset + global_warp_id * entries_per_warp + local_idx
    var iter_count = (remaining - local_idx + stripe_size - 1) // stripe_size
    var max_iters = (aligned_size + stripe_size - 1) // stripe_size
    if max_iters < 1:
        max_iters = 1

    var b_ptr = bins_p + base
    var s1_ptr = stats + base
    var s2_ptr = stats + (stat_line_size + base)
    var pos_base = base

    for it in range(max_iters):
        var active = it < iter_count
        var lb = InlineArray[UInt32, H8_POINTS](fill=0)
        var lq1 = InlineArray[Int32, H8_POINTS](fill=0)
        var lq2 = InlineArray[Int32, H8_POINTS](fill=0)

        @parameter
        for k in range(H8_UNROLL):
            if active:
                var vb = ldg[width=H8_LOAD, alignment=4](
                    b_ptr + H8_LANE * H8_LOAD * k
                )
                var v1 = ldg[width=H8_LOAD, alignment=4](
                    s1_ptr + H8_LANE * H8_LOAD * k
                )
                var v2 = ldg[width=H8_LOAD, alignment=4](
                    s2_ptr + H8_LANE * H8_LOAD * k
                )

                @parameter
                for e in range(H8_LOAD):
                    var u = hist2_dither(
                        pos_base + H8_LANE * H8_LOAD * k + e
                    )
                    lb[k * H8_LOAD + e] = vb[e]
                    lq1[k * H8_LOAD + e] = hist2_quantize(
                        v1[e], fixed_scale, u
                    )
                    lq2[k * H8_LOAD + e] = hist2_quantize(
                        v2[e], fixed_scale, u
                    )

        if active:

            @parameter
            for k in range(H8_POINTS):
                h8_add_point(lb[k], lq1[k], lq2[k], tid, slice_base, smem)
        b_ptr += stripe_size
        s1_ptr += stripe_size
        s2_ptr += stripe_size
        pos_base += stripe_size

    h8_reduce_and_flush(
        tid, active_block_count,
        feature_folds, feature_fold_offset, feature_group_offset,
        feature_group_size, feature_offset, f_count, leaf_count,
        stat_count, bin_sums, acc_i32, fixed_scale, smem,
    )


def hist2_8bit_gather_kernel[ridx_stats: Bool = False](
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
    """Indexed loads (below the root): bins through `indices`, stats
    contiguous (or gathered through the same index under `ridx_stats`,
    DEVIATION 1902), dither keyed on the storage position -- the same
    conventions as every gather kernel in this package."""
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

    var local_block_idx = Int(block_idx.x) % max_blocks_per_part
    var active_block_count = min(
        (p_size + H8_MIN_DOCS - 1) // H8_MIN_DOCS, max_blocks_per_part
    )
    if local_block_idx >= active_block_count:
        return

    var smem = stack_allocation[
        H8_SMEM,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var z = tid
    while z < H8_SMEM:
        smem[z] = Int32(0)
        z += H8_BLOCK
    barrier()
    var slice_base = h8_slice_base(tid)

    comptime ALIGN_SIZE = H8_LOAD * H8_LANE * H8_UNROLL
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
        if local_block_idx == 0 and pe < head_len:
            var hrow = Int(ldg(indices + (p_offset + pe)))
            var hb = ldg(cindex_p + hrow)
            var u = hist2_dither(p_offset + pe)
            # DEVIATION 1902 (`ridx_stats`): stationary planes, both stats
            # ride the same gathered row id as the bin. The dither key
            # stays the storage POSITION in both arms -- same bits, same
            # quantize (`split_points_ridx.mojo`'s invariant).
            var hs1: Float32
            var hs2: Float32

            @parameter
            if ridx_stats:
                hs1 = ldg(stats + hrow)
                hs2 = ldg(stats + (stat_line_size + hrow))
            else:
                hs1 = ldg(stats + (p_offset + pe))
                hs2 = ldg(stats + (stat_line_size + p_offset + pe))
            var hq1 = hist2_quantize(hs1, fixed_scale, u)
            var hq2 = hist2_quantize(hs2, fixed_scale, u)
            h8_add_point(hb, hq1, hq2, tid, slice_base, smem)
        if local_block_idx == 0 and pe < tail_len:
            var trow = Int(ldg(indices + (tail_start + pe)))
            var tb = ldg(cindex_p + trow)
            var u = hist2_dither(tail_start + pe)
            var ts1: Float32
            var ts2: Float32

            @parameter
            if ridx_stats:
                # DEVIATION 1902, as on the head peel above.
                ts1 = ldg(stats + trow)
                ts2 = ldg(stats + (stat_line_size + trow))
            else:
                ts1 = ldg(stats + (tail_start + pe))
                ts2 = ldg(stats + (stat_line_size + tail_start + pe))
            var tq1 = hist2_quantize(ts1, fixed_scale, u)
            var tq2 = hist2_quantize(ts2, fixed_scale, u)
            h8_add_point(tb, tq1, tq2, tid, slice_base, smem)
        pe += H8_BLOCK

    var aligned_offset = p_offset + head_len
    var aligned_size = body_size - tail_len
    var warps_per_block = H8_BLOCK // H8_LANE
    var global_warp_id = local_block_idx * warps_per_block + (
        tid // H8_LANE
    )
    var entries_per_warp = H8_LANE * H8_UNROLL * H8_LOAD
    var stripe_size = entries_per_warp * warps_per_block * active_block_count
    var remaining = max(aligned_size - global_warp_id * entries_per_warp, 0)
    var local_idx = (tid & (H8_LANE - 1)) * H8_LOAD
    var base = aligned_offset + global_warp_id * entries_per_warp + local_idx
    var iter_count = (remaining - local_idx + stripe_size - 1) // stripe_size
    var max_iters = (aligned_size + stripe_size - 1) // stripe_size
    if max_iters < 1:
        max_iters = 1

    var i_ptr = indices + base
    var s1_ptr = stats + base
    var s2_ptr = stats + (stat_line_size + base)
    var pos_base = base

    for it in range(max_iters):
        var active = it < iter_count
        var lb = InlineArray[UInt32, H8_POINTS](fill=0)
        var lq1 = InlineArray[Int32, H8_POINTS](fill=0)
        var lq2 = InlineArray[Int32, H8_POINTS](fill=0)

        @parameter
        for k in range(H8_UNROLL):
            if active:
                var vi = ldg[width=H8_LOAD, alignment=4](
                    i_ptr + H8_LANE * H8_LOAD * k
                )
                var v1 = SIMD[DType.float32, H8_LOAD](0.0)
                var v2 = SIMD[DType.float32, H8_LOAD](0.0)

                @parameter
                if not ridx_stats:
                    v1 = ldg[width=H8_LOAD, alignment=4](
                        s1_ptr + H8_LANE * H8_LOAD * k
                    )
                    v2 = ldg[width=H8_LOAD, alignment=4](
                        s2_ptr + H8_LANE * H8_LOAD * k
                    )

                @parameter
                for e in range(H8_LOAD):
                    var u = hist2_dither(
                        pos_base + H8_LANE * H8_LOAD * k + e
                    )
                    lb[k * H8_LOAD + e] = ldg(cindex_p + Int(vi[e]))

                    @parameter
                    if ridx_stats:
                        # DEVIATION 1902: both stats join the bins' scalar
                        # gather through the same loaded row id; the wide
                        # loads above are traded for it.
                        v1[e] = ldg(stats + Int(vi[e]))
                        v2[e] = ldg(
                            stats + (stat_line_size + Int(vi[e]))
                        )
                    lq1[k * H8_LOAD + e] = hist2_quantize(
                        v1[e], fixed_scale, u
                    )
                    lq2[k * H8_LOAD + e] = hist2_quantize(
                        v2[e], fixed_scale, u
                    )

        if active:

            @parameter
            for k in range(H8_POINTS):
                h8_add_point(lb[k], lq1[k], lq2[k], tid, slice_base, smem)
        i_ptr += stripe_size
        s1_ptr += stripe_size
        s2_ptr += stripe_size
        pos_base += stripe_size

    h8_reduce_and_flush(
        tid, active_block_count,
        feature_folds, feature_fold_offset, feature_group_offset,
        feature_group_size, feature_offset, f_count, leaf_count,
        stat_count, bin_sums, acc_i32, fixed_scale, smem,
    )
