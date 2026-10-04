# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Depthwise per-level experiments, FAST on Apple only (lane apple-fast-dwgap2).

Each arm has its own switch in `greedy_search_helper_depthwise.mojo`.
PART_VEC4 and SCAN_SMEM are FAST Apple defaults since the M3 A/B (off with
`-D <NAME>_OFF`); COPY_ZERO stays opt-in. IDENTICAL never reaches this file. The measured reason all three exist: on Metal a
one-float-per-thread copy ran at 11.0 GB/s and the 16-byte form at 65.2 GB/s
(`copy_histograms_vec4_kernel`'s deviation block), and the per-level chain
still moves the row index and the flag plane one element per thread.

  * `MOJOLEARN_GBDT_DW2_PART_VEC4`: the fused split chain
    (`kernel/split_chain_fused.mojo`) with four rows per thread. A chunk is
    2048 rows (512 threads x 4) laid out in ABSOLUTE 4-aligned coordinates
    (chunk 0 of a leaf starts at `offset & ~3`), so every interior group of
    four is one aligned 16-byte row-index load and one aligned 4-byte flag
    load/store; the two boundary groups of a leaf go element by element and
    never touch a neighbor leaf's rows. A quarter of the block scans and
    barriers per row. The permutation is the same stable partition (zeros
    first, both sides in row order: a thread's four rows are contiguous and
    precede the next thread's), so the row index, sizes and partition stats
    are the same integers.
  * `MOJOLEARN_GBDT_DW2_SCAN_SMEM`: the histogram prefix scan with the same
    serial per-(feature, leaf, stat) fold, but over threadgroup memory: a
    block takes 16 consecutive features of one (leaf, stat) plane, loads
    their contiguous cell span with aligned 16-byte loads, runs the
    unchanged serial `running = ftz(running + x)` in shared memory, and
    writes the span back. Same adds in the same order: the same bits. Falls
    back to the global serial loop when the features are not laid out
    back to back or the span exceeds the shared page.
"""

from gbdt.gpu_data.gpu_structures import CFeature
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.gpu.intrinsics import ldg
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.primitives.block import broadcast as block_broadcast
from max.gpu.primitives.block import prefix_sum as block_prefix_sum
from max.gpu.sync import barrier

from checks.numerics import ftz

# ---- MOJOLEARN_GBDT_DW2_PART_VEC4 ------------------------------------------
comptime DW2_PART_BLOCK = 512
comptime DW2_PART_VEC = 4
comptime DW2_PART_CHUNK = DW2_PART_BLOCK * DW2_PART_VEC
comptime DW2_COPY_BLOCK = 256


def dw2_part_max_chunks(n_rows: Int) -> Int:
    """The chunk-table stride: a leaf spans at most `n_rows + 3` absolute
    positions from its aligned base."""
    var c = (n_rows + 3 + DW2_PART_CHUNK - 1) // DW2_PART_CHUNK
    return c if c > 0 else 1


def _dw2_n_chunks(offset: Int, size: Int) -> Int:
    if size <= 0:
        return 0
    var a0 = offset - (offset % DW2_PART_VEC)
    return (offset + size - a0 + DW2_PART_CHUNK - 1) // DW2_PART_CHUNK


def _dw2_clamp(x: Int, lo: Int, hi: Int) -> Int:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def dw2_flags_count_kernel[GUARD: Bool = False](
    compressed_index: MutPointer[UInt32, MutAnyOrigin],
    row_index: MutPointer[UInt32, MutAnyOrigin],
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    leaf_ids: MutPointer[UInt32, MutAnyOrigin],
    split_features: MutPointer[CFeature, MutAnyOrigin],
    split_bins: MutPointer[UInt32, MutAnyOrigin],
    flags: MutPointer[UInt8, MutAnyOrigin],
    chunk_zeros: MutPointer[UInt32, MutAnyOrigin],
    max_chunks_in: Int32,
    n_split_dev: MutPointer[UInt32, MutAnyOrigin],
):
    """K1 of `fused_flags_count_kernel`, four rows per thread. Grid (chunk
    stride, split leaf)."""
    comptime if GUARD:
        if Int(block_idx.y) >= Int(n_split_dev.unsafe_load(0)):
            return
    var max_chunks = Int(max_chunks_in)
    var leaf_slot = Int(block_idx.y)
    var leaf_id = Int(leaf_ids.unsafe_load(leaf_slot))
    var offset = Int(part_offset.unsafe_load(leaf_id))
    var size = Int(part_size.unsafe_load(leaf_id))
    var end = offset + size
    var a0 = offset - (offset % DW2_PART_VEC)
    var n_chunks = _dw2_n_chunks(offset, size)
    # field by field: a whole-struct CFeature load kills the Metal compiler
    var f_offset = Int(split_features[unsafe_offset=leaf_slot].offset)
    var shift = split_features[unsafe_offset=leaf_slot].shift
    var one_hot = split_features[unsafe_offset=leaf_slot].one_hot_feature
    var value = UInt32(split_bins.unsafe_load(leaf_slot)) << shift
    var mask = split_features[unsafe_offset=leaf_slot].mask << shift
    var tid = Int(thread_idx.x)
    var chunk = Int(block_idx.x)
    while chunk < n_chunks:
        var g = a0 + chunk * DW2_PART_CHUNK + tid * DW2_PART_VEC
        var v = Int32(0)
        if g >= offset and g + DW2_PART_VEC <= end:
            var idx4 = row_index.unsafe_load[width=4, alignment=16](g)
            var f4 = SIMD[DType.uint8, 4](0)
            comptime for e in range(4):
                var feature_val = (
                    ldg(compressed_index + (f_offset + Int(idx4[e]))) & mask
                )
                var goes_right = (
                    feature_val == value
                ) if one_hot else (feature_val > value)
                if goes_right:
                    f4[e] = UInt8(1)
                else:
                    v += Int32(1)
            flags.unsafe_store[width=4, alignment=4](g, f4)
        else:
            comptime for e in range(4):
                var j = g + e
                if j >= offset and j < end:
                    var load_index = Int(row_index.unsafe_load(j))
                    var feature_val = (
                        ldg(compressed_index + (f_offset + load_index)) & mask
                    )
                    var goes_right = (
                        feature_val == value
                    ) if one_hot else (feature_val > value)
                    flags.unsafe_store(
                        j, UInt8(1) if goes_right else UInt8(0)
                    )
                    if not goes_right:
                        v += Int32(1)
        var inc = block_prefix_sum[
            block_size=DW2_PART_BLOCK, exclusive=False
        ](v)
        var total = block_broadcast[block_size=DW2_PART_BLOCK](
            inc, src_thread=DW2_PART_BLOCK - 1
        )
        if tid == 0:
            chunk_zeros.unsafe_store(
                leaf_slot * max_chunks + chunk, UInt32(Int(total))
            )
        barrier()
        chunk += Int(grid_dim.x)


def dw2_scan_update_kernel[GUARD: Bool = False](
    left_leaves: MutPointer[UInt32, MutAnyOrigin],
    right_leaves: MutPointer[UInt32, MutAnyOrigin],
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    chunk_zeros: MutPointer[UInt32, MutAnyOrigin],
    chunk_zero_offsets: MutPointer[UInt32, MutAnyOrigin],
    leaf_total_zeros: MutPointer[UInt32, MutAnyOrigin],
    slot_off: MutPointer[UInt32, MutAnyOrigin],
    slot_sz: MutPointer[UInt32, MutAnyOrigin],
    win_cells: MutPointer[UInt32, MutAnyOrigin],
    split_features: MutPointer[CFeature, MutAnyOrigin],
    bin_feature_count_in: Int32,
    stat_count_in: Int32,
    histograms: MutPointer[Float32, MutAnyOrigin],
    part_stats: MutPointer[Float32, MutAnyOrigin],
    max_chunks_in: Int32,
    n_split_dev: MutPointer[UInt32, MutAnyOrigin],
):
    """K2 of `fused_scan_update_kernel` over the 2048-row aligned chunks;
    the stats update and partition update are its body verbatim."""
    comptime if GUARD:
        if Int(block_idx.y) >= Int(n_split_dev.unsafe_load(0)):
            return
    var max_chunks = Int(max_chunks_in)
    var leaf_slot = Int(block_idx.y)
    var left_leaf = Int(left_leaves.unsafe_load(leaf_slot))
    var right_leaf = Int(right_leaves.unsafe_load(leaf_slot))
    var offset = Int(part_offset.unsafe_load(left_leaf))
    var size = Int(part_size.unsafe_load(left_leaf))
    var n_chunks = _dw2_n_chunks(offset, size)
    var tid = Int(thread_idx.x)

    var bin_feature_count = Int(bin_feature_count_in)
    var stat_count = Int(stat_count_in)
    var cell = Int(win_cells.unsafe_load(leaf_slot))
    var one_hot = split_features[unsafe_offset=leaf_slot].one_hot_feature
    var stat_id = tid
    while stat_id < stat_count:
        var parent = part_stats.unsafe_load(left_leaf * stat_count + stat_id)
        var cell_sum = histograms.unsafe_load(
            left_leaf * bin_feature_count * stat_count
            + stat_id * bin_feature_count
            + cell
        )
        var derived = parent - cell_sum
        if stat_id == 0:
            cell_sum = max(cell_sum, Float32(0.0))
            derived = max(derived, Float32(0.0))
        var left_sum = derived if one_hot else cell_sum
        var right_sum = cell_sum if one_hot else derived
        part_stats.unsafe_store(left_leaf * stat_count + stat_id, left_sum)
        part_stats.unsafe_store(right_leaf * stat_count + stat_id, right_sum)
        stat_id += Int(block_dim.x)

    var carry = 0
    var c = 0
    while c < n_chunks:
        var idx = c + tid
        var v = Int32(0)
        if idx < n_chunks:
            v = Int32(
                Int(chunk_zeros.unsafe_load(leaf_slot * max_chunks + idx))
            )
        var inc = block_prefix_sum[
            block_size=DW2_PART_BLOCK, exclusive=False
        ](v)
        var group_total = block_broadcast[block_size=DW2_PART_BLOCK](
            inc, src_thread=DW2_PART_BLOCK - 1
        )
        if idx < n_chunks:
            chunk_zero_offsets.unsafe_store(
                leaf_slot * max_chunks + idx,
                UInt32(carry + Int(inc) - Int(v)),
            )
        carry += Int(group_total)
        barrier()
        c += DW2_PART_BLOCK
    barrier()
    if tid == 0:
        leaf_total_zeros.unsafe_store(leaf_slot, UInt32(carry))
        slot_off.unsafe_store(leaf_slot, UInt32(offset))
        slot_sz.unsafe_store(leaf_slot, UInt32(size))
        part_size.unsafe_store(left_leaf, UInt32(carry))
        part_offset.unsafe_store(right_leaf, UInt32(offset + carry))
        part_size.unsafe_store(right_leaf, UInt32(size - carry))


def dw2_place_scatter_kernel[GUARD: Bool = False](
    slot_off: MutPointer[UInt32, MutAnyOrigin],
    slot_sz: MutPointer[UInt32, MutAnyOrigin],
    flags: MutPointer[UInt8, MutAnyOrigin],
    chunk_zero_offsets: MutPointer[UInt32, MutAnyOrigin],
    leaf_total_zeros: MutPointer[UInt32, MutAnyOrigin],
    row_index: MutPointer[UInt32, MutAnyOrigin],
    temp_index: MutPointer[UInt32, MutAnyOrigin],
    max_chunks_in: Int32,
    n_split_dev: MutPointer[UInt32, MutAnyOrigin],
):
    """K3 of `fused_place_scatter_kernel`, four contiguous rows per thread:
    one block scan per 2048 rows ranks the zeros, each thread then places
    its rows in order. Grid (chunk stride, split leaf)."""
    comptime if GUARD:
        if Int(block_idx.y) >= Int(n_split_dev.unsafe_load(0)):
            return
    var max_chunks = Int(max_chunks_in)
    var leaf_slot = Int(block_idx.y)
    var offset = Int(slot_off.unsafe_load(leaf_slot))
    var size = Int(slot_sz.unsafe_load(leaf_slot))
    var end = offset + size
    var a0 = offset - (offset % DW2_PART_VEC)
    var n_chunks = _dw2_n_chunks(offset, size)
    var tid = Int(thread_idx.x)
    var n_zeros = Int(leaf_total_zeros.unsafe_load(leaf_slot))
    var chunk = Int(block_idx.x)
    while chunk < n_chunks:
        var c0 = a0 + chunk * DW2_PART_CHUNK
        var zeros_before = Int(
            chunk_zero_offsets.unsafe_load(leaf_slot * max_chunks + chunk)
        )
        var elems_before = _dw2_clamp(c0, offset, end) - offset
        var ones_before = elems_before - zeros_before
        var g = c0 + tid * DW2_PART_VEC
        var full = g >= offset and g + DW2_PART_VEC <= end
        var zmask = 0
        var rmask = 0
        var z = 0
        if full:
            var f4 = flags.unsafe_load[width=4, alignment=4](g)
            rmask = 15
            comptime for e in range(4):
                if f4[e] == UInt8(0):
                    zmask |= 1 << e
                    z += 1
        else:
            comptime for e in range(4):
                var j = g + e
                if j >= offset and j < end:
                    rmask |= 1 << e
                    if flags.unsafe_load(j) == UInt8(0):
                        zmask |= 1 << e
                        z += 1
        var inc_zero = block_prefix_sum[
            block_size=DW2_PART_BLOCK, exclusive=False
        ](Int32(z))
        var rz = Int(inc_zero) - z
        # in-range rows of this chunk before this thread's first row
        var before_t = _dw2_clamp(g, offset, end) - _dw2_clamp(c0, offset, end)
        var ro = before_t - rz
        if full:
            var idx4 = row_index.unsafe_load[width=4, alignment=16](g)
            comptime for e in range(4):
                var dst = 0
                if (zmask >> e) & 1 == 1:
                    dst = zeros_before + rz
                    rz += 1
                else:
                    dst = n_zeros + ones_before + ro
                    ro += 1
                temp_index.unsafe_store(offset + dst, idx4[e])
        else:
            comptime for e in range(4):
                if (rmask >> e) & 1 == 1:
                    var dst = 0
                    if (zmask >> e) & 1 == 1:
                        dst = zeros_before + rz
                        rz += 1
                    else:
                        dst = n_zeros + ones_before + ro
                        ro += 1
                    temp_index.unsafe_store(
                        offset + dst, row_index.unsafe_load(g + e)
                    )
        barrier()
        chunk += Int(grid_dim.x)


def dw2_copy_back_kernel[GUARD: Bool = False](
    slot_off: MutPointer[UInt32, MutAnyOrigin],
    slot_sz: MutPointer[UInt32, MutAnyOrigin],
    temp_index: MutPointer[UInt32, MutAnyOrigin],
    row_index: MutPointer[UInt32, MutAnyOrigin],
    n_split_dev: MutPointer[UInt32, MutAnyOrigin],
):
    """K4 of `fused_copy_back_kernel` in aligned groups of four; the two
    boundary groups copy only the leaf's own rows."""
    comptime if GUARD:
        if Int(block_idx.y) >= Int(n_split_dev.unsafe_load(0)):
            return
    var leaf_slot = Int(block_idx.y)
    var offset = Int(slot_off.unsafe_load(leaf_slot))
    var size = Int(slot_sz.unsafe_load(leaf_slot))
    if size <= 0:
        return
    var end = offset + size
    var a0 = offset - (offset % DW2_PART_VEC)
    var n_groups = (end - a0 + DW2_PART_VEC - 1) // DW2_PART_VEC
    var gi = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while gi < n_groups:
        var g = a0 + gi * DW2_PART_VEC
        if g >= offset and g + DW2_PART_VEC <= end:
            row_index.unsafe_store[width=4, alignment=16](
                g, temp_index.unsafe_load[width=4, alignment=16](g)
            )
        else:
            comptime for e in range(4):
                var j = g + e
                if j >= offset and j < end:
                    row_index.unsafe_store(j, temp_index.unsafe_load(j))
        gi += stride


# ---- MOJOLEARN_GBDT_DW2_SCAN_SMEM ------------------------------------------
#: Features per block. 16 x 255 folds fits the page with room for the
#: alignment head and tail.
comptime DW2_SCAN_FT = 16
comptime DW2_SCAN_CAP = 4096 + 8
comptime DW2_SCAN_BLOCK = 256


def dw2_scan_histograms_smem_kernel(
    hist_ids: MutPointer[UInt32, MutAnyOrigin],
    feature_first_bin: MutPointer[UInt32, MutAnyOrigin],
    feature_folds: MutPointer[UInt32, MutAnyOrigin],
    feature_one_hot: MutPointer[UInt8, MutAnyOrigin],
    feature_count_in: Int32,
    bin_feature_count_in: Int32,
    histogram: MutPointer[Float32, MutAnyOrigin],
):
    """`scan_histograms_kernel`'s serial fold over a shared-memory copy of
    16 features' cells. Grid (ceil(features / 16), built leaf, stat), block
    256."""
    var feature_count = Int(feature_count_in)
    var bin_feature_count = Int(bin_feature_count_in)
    var f0 = Int(block_idx.x) * DW2_SCAN_FT
    if f0 >= feature_count:
        return
    var f_end = f0 + DW2_SCAN_FT
    if f_end > feature_count:
        f_end = feature_count
    var leaf_id = Int(hist_ids.unsafe_load(Int(block_idx.y)))
    var stat_id = Int(block_idx.z)
    var stat_count = Int(grid_dim.z)
    var base = (
        leaf_id * bin_feature_count * stat_count + stat_id * bin_feature_count
    )
    var tid = Int(thread_idx.x)

    # the tile's span, and whether its features sit back to back (every
    # thread computes the same answer, so the branch below is uniform)
    var lo = Int(feature_first_bin.unsafe_load(f0))
    var hi = lo
    var contig = True
    for f in range(f0, f_end):
        var fb = Int(feature_first_bin.unsafe_load(f))
        if fb != hi:
            contig = False
        hi = fb + Int(feature_folds.unsafe_load(f))
    var a_lo = base + lo
    var a_hi = base + hi
    var a4 = a_lo - (a_lo % 4)
    var plane_end = base + bin_feature_count

    if contig and hi <= bin_feature_count and a_hi - a4 <= DW2_SCAN_CAP:
        var s = stack_allocation[
            DW2_SCAN_CAP,
            Scalar[DType.float32],
            address_space = AddressSpace.SHARED,
        ]()
        var n_g = (a_hi - a4 + 3) // 4
        var gi = tid
        while gi < n_g:
            var g = a4 + gi * 4
            if g + 4 <= plane_end:
                var v = histogram.unsafe_load[width=4, alignment=16](g)
                comptime for e in range(4):
                    s[gi * 4 + e] = v[e]
            else:
                comptime for e in range(4):
                    if g + e < a_hi:
                        s[gi * 4 + e] = histogram.unsafe_load(g + e)
            gi += Int(block_dim.x)
        barrier()
        if tid < f_end - f0:
            var f = f0 + tid
            var folds = Int(feature_folds.unsafe_load(f))
            if feature_one_hot.unsafe_load(f) == UInt8(0) and folds > 1:
                var o = base + Int(feature_first_bin.unsafe_load(f)) - a4
                var running = Scalar[DType.float32](0.0)
                for i in range(folds):
                    running = ftz(running + s[o + i])
                    s[o + i] = running
        barrier()
        gi = tid
        while gi < n_g:
            var g = a4 + gi * 4
            if g >= a_lo and g + 4 <= a_hi:
                var v = SIMD[DType.float32, 4](
                    s[gi * 4], s[gi * 4 + 1], s[gi * 4 + 2], s[gi * 4 + 3]
                )
                histogram.unsafe_store[width=4, alignment=16](g, v)
            else:
                comptime for e in range(4):
                    var j = g + e
                    if j >= a_lo and j < a_hi:
                        histogram.unsafe_store(j, s[gi * 4 + e])
            gi += Int(block_dim.x)
    else:
        # the global serial loop of `scan_histograms_kernel`, verbatim
        if tid < f_end - f0:
            var f = f0 + tid
            var folds = Int(feature_folds.unsafe_load(f))
            if feature_one_hot.unsafe_load(f) == UInt8(0) and folds > 1:
                var b = base + Int(feature_first_bin.unsafe_load(f))
                var running = Scalar[DType.float32](0.0)
                for i in range(folds):
                    running = ftz(running + histogram.unsafe_load(b + i))
                    histogram.unsafe_store(b + i, running)


# ---- MOJOLEARN_GBDT_DW_BRIDGE_SCAN (opt-in FAST + Apple) -------------------
# Pending current-main M3 speed and fitted AUC/logloss gate. Each tile owns
# disjoint feature cells in the dense integer accumulator and the sparse
# leaf histogram. Keep the bridge's exact division and the scanner's serial
# Float32 fold; only their global-memory intermediate and launch disappear.
def dw_bridge_scan_kernel(
    hist_ids: MutPointer[UInt32, MutAnyOrigin],
    feature_first_bin: MutPointer[UInt32, MutAnyOrigin],
    feature_folds: MutPointer[UInt32, MutAnyOrigin],
    feature_one_hot: MutPointer[UInt8, MutAnyOrigin],
    feature_count_in: Int32,
    bin_feature_count_in: Int32,
    q_acc: MutPointer[Int32, MutAnyOrigin],
    fixed_scale_ptr: MutPointer[Float32, MutAnyOrigin],
    histogram: MutPointer[Float32, MutAnyOrigin],
):
    """Fuse qh_write_hist_kernel with the existing shared serial scan.

    One-byte quantized histograms only: at most 256 folds per feature.
    Grid (ceil(features / 16), dense built leaf, stat), block 256.
    The root keeps the separate bridge so mode selection sees raw bins.
    """
    var f0 = Int(block_idx.x) * DW2_SCAN_FT
    var nf = min(DW2_SCAN_FT, Int(feature_count_in) - f0)
    var tid = Int(thread_idx.x)
    var dense = Int(block_idx.y)
    var stat = Int(block_idx.z)
    var stat_count = Int(grid_dim.z)
    var cells = Int(bin_feature_count_in)
    var src_base = (dense * stat_count + stat) * cells
    var leaf = Int(hist_ids.unsafe_load(dense))
    var dst_base = (leaf * stat_count + stat) * cells
    var scale = fixed_scale_ptr.unsafe_load(0)
    var s = stack_allocation[
        DW2_SCAN_FT * 256, Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    # Adjacent lanes load adjacent bins; padded feature slots never touch
    # global memory and are not consumed by the feature's prefix fold.
    var i = tid
    while i < nf * 256:
        var f = f0 + (i >> 8)
        var bin = i & 255
        var folds = Int(feature_folds.unsafe_load(f))
        if bin < folds:
            var cell = Int(feature_first_bin.unsafe_load(f)) + bin
            var q = q_acc.unsafe_load(src_base + cell)
            var val = Float32(0.0)
            if q != Int32(0):
                val = ftz(Float32(Int(q)) / scale)
                q_acc.unsafe_store(src_base + cell, Int32(0))
            s[i] = val
        i += Int(block_dim.x)
    barrier()
    if tid < nf:
        var f = f0 + tid
        var folds = Int(feature_folds.unsafe_load(f))
        if feature_one_hot.unsafe_load(f) == UInt8(0) and folds > 1:
            var running = Float32(0.0)
            for bin in range(folds):
                running = ftz(running + s[tid * 256 + bin])
                s[tid * 256 + bin] = running
    barrier()
    i = tid
    while i < nf * 256:
        var f = f0 + (i >> 8)
        var bin = i & 255
        if bin < Int(feature_folds.unsafe_load(f)):
            var cell = Int(feature_first_bin.unsafe_load(f)) + bin
            histogram.unsafe_store(dst_base + cell, s[i])
        i += Int(block_dim.x)
