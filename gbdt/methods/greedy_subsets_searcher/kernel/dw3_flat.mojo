# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Depthwise split chain over a FLAT work list (FAST on Apple, opt-in
`-D MOJOLEARN_GBDT_DW_FLAT_GRID`, lane apple-fast-w3-dw). IDENTICAL never
reaches this file.

Why. `_launch_fused_split_chain`'s DW2_PART_VEC4 arm launches K1 (flags +
count), K3 (place + scatter) and K4 (copy back) on a 2D grid
`(min(max_chunks(n_rows), 2 * sm_count), n_split)`. The x extent is sized for
a leaf that holds EVERY row, and it is the same for every split leaf. A
depth-8 Depthwise tree splits up to 128 leaves at its last level. On taxi
(5.25M rows, M3 Ultra) that is 160 x 128 = 20,480 threadgroups of 512 threads
per kernel. Each leaf has about 2,560 / 128 = 20 chunks of real work, so about
88% of the threadgroups load their leaf header and return. Summed over a
tree's eight levels and three kernels, that is about 120k threadgroups per
tree, and the count depends on the device and the leaf count, not on the
rows. This matches the large fixed per-tree cost in the profile (14.3 ms per
tree at 1M rows against about 21.6 ms at 5.25M).

What. The same three bodies, but the grid is ONE-dimensional and walks a
list of (slot, chunk) work items. Each block first finds every split slot's
chunk count (one thread per slot, at most `DW3_MAX_SLOTS`) and takes one
block prefix sum into threadgroup memory. Then for work item `w = block_idx.x,
block_idx.x + grid_dim.x, ...` it binary-searches the slot and runs the
unchanged DW2 body on (slot, chunk). The host sizes the grid as
`min(total chunks upper bound, DW3_GRID_PER_SM * sm_count)`, so no block is
empty, and a large leaf no longer forces the grid width of a small one.

Bits. Every (slot, chunk) work item runs the same code on the same inputs as
in the DW2 kernels: the same flags, the same `chunk_zeros[slot * max_chunks +
chunk]`, the same scatter destinations, the same copy. Only which block runs
which chunk changes. Integer moves only, so the row index, partition sizes
and partition stats are the same integers. The define is bit-identical by
construction.

GUARD (DW_NO_LEVEL_SYNC): the host passes the scored-leaf count as the slot
bound. The device split count `n_split_dev[0]` (at most that bound) trims
the slot list inside the kernel, so the work list covers exactly the
device-selected splits.
"""
from gbdt.gpu_data.gpu_structures import CFeature
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.gpu.intrinsics import ldg
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.primitives.block import broadcast as block_broadcast
from max.gpu.primitives.block import prefix_sum as block_prefix_sum
from max.gpu.sync import barrier

from gbdt.methods.greedy_subsets_searcher.kernel.dw2_level import (
    DW2_PART_BLOCK,
    DW2_PART_CHUNK,
    DW2_PART_VEC,
)

#: Slots one block can index: one thread per slot in the prefix pass. The
#: launcher keeps the DW2 2D kernels when a level has more split candidates.
comptime DW3_MAX_SLOTS = DW2_PART_BLOCK
#: Flat grid cap in blocks per GPU core: the DW2 root level's
#: `split_points_grid_x` for one leaf (4 * sm_count), now for the whole level.
comptime DW3_GRID_PER_SM = 4


def dw3_flat_grid(n_split: Int, n_rows: Int, sm_count: Int) -> Int:
    """Blocks for a flat launch. Upper bound on total work items: the slots
    are disjoint row ranges, so their chunks sum to at most
    `ceil((n_rows + 3 * n_split) / CHUNK) + n_split` (one partial chunk and
    a 3-row alignment head per slot)."""
    var upper = (
        n_rows + 3 * n_split + DW2_PART_CHUNK - 1
    ) // DW2_PART_CHUNK + n_split
    var cap = DW3_GRID_PER_SM * sm_count
    if cap < 1:
        cap = 1
    var g = upper if upper < cap else cap
    return g if g > 0 else 1


@always_inline
def _dw3_n_chunks(offset: Int, size: Int) -> Int:
    """`dw2_level._dw2_n_chunks`, restated so this file needs no private
    import: chunks of a leaf laid out from its 4-aligned base."""
    if size <= 0:
        return 0
    var a0 = offset - (offset % DW2_PART_VEC)
    return (offset + size - a0 + DW2_PART_CHUNK - 1) // DW2_PART_CHUNK


@always_inline
def _dw3_clamp(x: Int, lo: Int, hi: Int) -> Int:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@always_inline
def _dw3_find_slot(
    pre: UnsafePointer[
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
        origin=MutUntrackedOrigin,
    ],
    n_slots: Int,
    w: Int,
) -> Int:
    """The largest slot `s < n_slots` with `pre[s] <= w`. `pre` is the
    exclusive prefix of chunk counts, so `pre[s] <= w < pre[s + 1]` and
    zero-chunk slots are skipped."""
    var lo = 0
    var hi = n_slots - 1
    while lo < hi:
        var mid = (lo + hi + 1) // 2
        if Int(pre[mid]) <= w:
            lo = mid
        else:
            hi = mid - 1
    return lo


def dw3_flat_flags_count_kernel[GUARD: Bool = False](
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
    n_split_in: Int32,
    n_split_dev: MutPointer[UInt32, MutAnyOrigin],
):
    """`dw2_flags_count_kernel` over the flat (slot, chunk) list. Grid
    (dw3_flat_grid, 1), block DW2_PART_BLOCK."""
    var n_split = Int(n_split_in)
    comptime if GUARD:
        var nd = Int(n_split_dev.unsafe_load(0))
        if nd < n_split:
            n_split = nd
    if n_split <= 0:
        return
    var max_chunks = Int(max_chunks_in)
    var tid = Int(thread_idx.x)
    var pre = stack_allocation[
        DW3_MAX_SLOTS + 1,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var nc = Int32(0)
    if tid < n_split:
        var lid = Int(leaf_ids.unsafe_load(tid))
        nc = Int32(
            _dw3_n_chunks(
                Int(part_offset.unsafe_load(lid)),
                Int(part_size.unsafe_load(lid)),
            )
        )
    var inc_slots = block_prefix_sum[
        block_size=DW2_PART_BLOCK, exclusive=False
    ](nc)
    pre[tid + 1] = inc_slots
    if tid == 0:
        pre[0] = Int32(0)
    barrier()
    var total = Int(pre[n_split])
    var w = Int(block_idx.x)
    while w < total:
        var leaf_slot = _dw3_find_slot(pre, n_split, w)
        var chunk = w - Int(pre[leaf_slot])
        var leaf_id = Int(leaf_ids.unsafe_load(leaf_slot))
        var offset = Int(part_offset.unsafe_load(leaf_id))
        var size = Int(part_size.unsafe_load(leaf_id))
        var end = offset + size
        var a0 = offset - (offset % DW2_PART_VEC)
        # field by field: a whole-struct CFeature load kills the Metal
        # compiler (the DW2 kernel's own note)
        var f_offset = Int(split_features[unsafe_offset=leaf_slot].offset)
        var shift = split_features[unsafe_offset=leaf_slot].shift
        var one_hot = split_features[unsafe_offset=leaf_slot].one_hot_feature
        var value = UInt32(split_bins.unsafe_load(leaf_slot)) << shift
        var mask = split_features[unsafe_offset=leaf_slot].mask << shift
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
        var chunk_total = block_broadcast[block_size=DW2_PART_BLOCK](
            inc, src_thread=DW2_PART_BLOCK - 1
        )
        if tid == 0:
            chunk_zeros.unsafe_store(
                leaf_slot * max_chunks + chunk, UInt32(Int(chunk_total))
            )
        barrier()
        w += Int(grid_dim.x)


def dw3_flat_place_scatter_kernel[GUARD: Bool = False](
    slot_off: MutPointer[UInt32, MutAnyOrigin],
    slot_sz: MutPointer[UInt32, MutAnyOrigin],
    flags: MutPointer[UInt8, MutAnyOrigin],
    chunk_zero_offsets: MutPointer[UInt32, MutAnyOrigin],
    leaf_total_zeros: MutPointer[UInt32, MutAnyOrigin],
    row_index: MutPointer[UInt32, MutAnyOrigin],
    temp_index: MutPointer[UInt32, MutAnyOrigin],
    max_chunks_in: Int32,
    n_split_in: Int32,
    n_split_dev: MutPointer[UInt32, MutAnyOrigin],
):
    """`dw2_place_scatter_kernel` over the flat (slot, chunk) list. The
    slot ranges are K2's parent snapshots (`slot_off`, `slot_sz`), the same
    ranges K1 counted, so the work list is the same one K1 walked."""
    var n_split = Int(n_split_in)
    comptime if GUARD:
        var nd = Int(n_split_dev.unsafe_load(0))
        if nd < n_split:
            n_split = nd
    if n_split <= 0:
        return
    var max_chunks = Int(max_chunks_in)
    var tid = Int(thread_idx.x)
    var pre = stack_allocation[
        DW3_MAX_SLOTS + 1,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var nc = Int32(0)
    if tid < n_split:
        nc = Int32(
            _dw3_n_chunks(
                Int(slot_off.unsafe_load(tid)), Int(slot_sz.unsafe_load(tid))
            )
        )
    var inc_slots = block_prefix_sum[
        block_size=DW2_PART_BLOCK, exclusive=False
    ](nc)
    pre[tid + 1] = inc_slots
    if tid == 0:
        pre[0] = Int32(0)
    barrier()
    var total = Int(pre[n_split])
    var w = Int(block_idx.x)
    while w < total:
        var leaf_slot = _dw3_find_slot(pre, n_split, w)
        var chunk = w - Int(pre[leaf_slot])
        var offset = Int(slot_off.unsafe_load(leaf_slot))
        var size = Int(slot_sz.unsafe_load(leaf_slot))
        var end = offset + size
        var a0 = offset - (offset % DW2_PART_VEC)
        var n_zeros = Int(leaf_total_zeros.unsafe_load(leaf_slot))
        var c0 = a0 + chunk * DW2_PART_CHUNK
        var zeros_before = Int(
            chunk_zero_offsets.unsafe_load(leaf_slot * max_chunks + chunk)
        )
        var elems_before = _dw3_clamp(c0, offset, end) - offset
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
        var before_t = _dw3_clamp(g, offset, end) - _dw3_clamp(c0, offset, end)
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
        w += Int(grid_dim.x)


def dw3_flat_copy_back_kernel[GUARD: Bool = False](
    slot_off: MutPointer[UInt32, MutAnyOrigin],
    slot_sz: MutPointer[UInt32, MutAnyOrigin],
    temp_index: MutPointer[UInt32, MutAnyOrigin],
    row_index: MutPointer[UInt32, MutAnyOrigin],
    n_split_in: Int32,
    n_split_dev: MutPointer[UInt32, MutAnyOrigin],
):
    """`dw2_copy_back_kernel` over the flat (slot, chunk) list: a work item
    is one 2048-row chunk of a slot (512 threads x 4 rows), interior groups
    one aligned 16-byte copy, the two boundary groups element by element
    inside the slot's own range."""
    var n_split = Int(n_split_in)
    comptime if GUARD:
        var nd = Int(n_split_dev.unsafe_load(0))
        if nd < n_split:
            n_split = nd
    if n_split <= 0:
        return
    var tid = Int(thread_idx.x)
    var pre = stack_allocation[
        DW3_MAX_SLOTS + 1,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var nc = Int32(0)
    if tid < n_split:
        nc = Int32(
            _dw3_n_chunks(
                Int(slot_off.unsafe_load(tid)), Int(slot_sz.unsafe_load(tid))
            )
        )
    var inc_slots = block_prefix_sum[
        block_size=DW2_PART_BLOCK, exclusive=False
    ](nc)
    pre[tid + 1] = inc_slots
    if tid == 0:
        pre[0] = Int32(0)
    barrier()
    var total = Int(pre[n_split])
    var w = Int(block_idx.x)
    while w < total:
        var leaf_slot = _dw3_find_slot(pre, n_split, w)
        var chunk = w - Int(pre[leaf_slot])
        var offset = Int(slot_off.unsafe_load(leaf_slot))
        var size = Int(slot_sz.unsafe_load(leaf_slot))
        var end = offset + size
        var a0 = offset - (offset % DW2_PART_VEC)
        var g = a0 + chunk * DW2_PART_CHUNK + tid * DW2_PART_VEC
        if g >= offset and g + DW2_PART_VEC <= end:
            row_index.unsafe_store[width=4, alignment=16](
                g, temp_index.unsafe_load[width=4, alignment=16](g)
            )
        else:
            comptime for e in range(4):
                var j = g + e
                if j >= offset and j < end:
                    row_index.unsafe_store(j, temp_index.unsafe_load(j))
        w += Int(grid_dim.x)
