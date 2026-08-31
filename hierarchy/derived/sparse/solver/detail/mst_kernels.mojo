# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's DERIVATION_MAP.tsv and in this file's own docstring. See NOTICE.
"""Boruvka's kernels, from RAFT.

PORT OF `raft/cpp/include/raft/sparse/solver/detail/mst_kernels.cuh` at
RAFT `661a3b8` (the `raft-v26.08.00` checkout carries the same file), plus
`get_1D_idx` from `detail/mst_utils.cuh`. Transliterated kernel for kernel,
in their order, with the ONE declared departure below. Do not improve.

DEVIATION 620 (see `hierarchy/original/edge_order.mojo` for the block):
`alteration_kernel` (`mst_kernels.cuh:289-307`) is NOT ported and nothing
reads `altered_weights`. Where theirs compares an altered `double`, ours
compares the triple `(weight_order_key(w), min(u,v), max(u,v))` through
`triple_less`, and the per-color minimum is taken in THREE integer
`atomicMin` phases instead of their one (`:94`): `kernel_min_edge_per_vertex`
publishes the weight key, then `min_edge_lo_per_color` publishes `min(u,v)`
among the vertices whose key equals the color's, then
`min_edge_hi_per_color` publishes `max(u,v)` among those. Each phase is an
integer min, so its result does not depend on which thread's atomic lands
first, and the three together are the lexicographic minimum of the triple.
Every other line is theirs.

THE SHAPE THAT IS THEIRS AND IS ALSO IDENTITY-SAFE. `kernel_min_edge_per_
vertex` is launched with ONE 32-THREAD BLOCK PER ROW (`mst_solver_inl.cuh:
287-295`: `n_threads = 32`, grid `v`), each lane scanning edges
`row_start + lane, +32, ...` and the 32 partials folded by a halving tree
in threadgroup memory (`:76-85`). Under a total order the minimum is the
same whatever the lane count or fold shape, so this kernel is launched at
their 32 on every vendor and no `kernel_matrix` row is needed for it.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from hierarchy.original.edge_order import (
    EDGE_SENTINEL,
    VERTEX_SENTINEL,
    WEIGHT_KEY_SENTINEL,
    edge_hi,
    edge_lo,
    sabotaged_lo_hi,
    triple_less,
    weight_order_key,
)


comptime MST_WARP = 32
"""Their `n_threads = 32` for the per-vertex kernel (`mst_solver_inl.cuh:
287`), which is ALSO the size of its three shared arrays (`mst_kernels.cuh:
34-36`). A fixed 32, not `WARP_SIZE`: on a 64-wide wavefront their kernel
still runs one 32-thread block per row, and so does ours."""


@always_inline
def get_1D_idx() -> Int:
    """`mst_utils.cuh:18-21`."""
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


def kernel_min_edge_per_vertex(
    offsets: MutPointer[Int32, MutAnyOrigin],
    indices: MutPointer[Int32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    color: MutPointer[Int32, MutAnyOrigin],
    color_index: MutPointer[Int32, MutAnyOrigin],
    new_mst_edge: MutPointer[Int32, MutAnyOrigin],
    mst_edge: MutPointer[UInt8, MutAnyOrigin],
    min_edge_color: MutPointer[Int32, MutAnyOrigin],
    v_in: Int32,
    sabotage: Int32,
):
    """`mst_kernels.cuh:18-97`. One 32-thread block per row; each lane keeps
    the minimum (under `triple_less`) of the edges it scanned, the block
    folds the 32 partials, lane 0 publishes the row's min edge and pushes
    its WEIGHT KEY into the color's slot (their `atomicMin(&min_edge_color
    [self_color], min_edge_weight[0])`, `:94`, phase one of three)."""
    var tid = get_1D_idx()
    var warp_id = tid // MST_WARP
    var lane_id = tid % MST_WARP

    var min_edge_index = stack_allocation[
        MST_WARP, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var min_edge_wk = stack_allocation[
        MST_WARP, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var min_edge_lo = stack_allocation[
        MST_WARP, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var min_edge_hi = stack_allocation[
        MST_WARP, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    # `min_color[32]` (`:36`, `:64`) is written and never read after the
    # fold in theirs; kept out rather than carried dead.

    min_edge_index[unsafe_offset=lane_id] = EDGE_SENTINEL
    min_edge_wk[unsafe_offset=lane_id] = WEIGHT_KEY_SENTINEL
    min_edge_lo[unsafe_offset=lane_id] = VERTEX_SENTINEL
    min_edge_hi[unsafe_offset=lane_id] = VERTEX_SENTINEL
    barrier()

    var v = Int(v_in)
    # `:44-45` read `color_index[warp_id]` BEFORE the `warp_id < v` guard;
    # with grid == v that index is always in range. Kept inside the guard
    # here so a padded grid cannot read past the buffer.
    var self_color = Int32(0)
    if warp_id < v:
        var self_color_idx = color_index.unsafe_load(warp_id)
        self_color = color.unsafe_load(Int(self_color_idx))

        var row_start = Int(offsets.unsafe_load(warp_id))
        var row_end = Int(offsets.unsafe_load(warp_id + 1))
        var e = row_start + lane_id
        while e < row_end:
            var successor = indices.unsafe_load(e)
            var successor_color_idx = color_index.unsafe_load(Int(successor))
            var successor_color = color.unsafe_load(Int(successor_color_idx))
            if mst_edge.unsafe_load(e) == 0 and self_color != successor_color:
                var wk = weight_order_key(weights.unsafe_load(e))
                var lh = sabotaged_lo_hi(
                    sabotage,
                    edge_lo(Int32(warp_id), successor),
                    edge_hi(Int32(warp_id), successor),
                )
                # `:63` `curr_edge_weight < min_edge_weight[lane_id]`
                if triple_less(
                    wk, lh[0], lh[1],
                    min_edge_wk[unsafe_offset=lane_id],
                    min_edge_lo[unsafe_offset=lane_id],
                    min_edge_hi[unsafe_offset=lane_id],
                ):
                    min_edge_wk[unsafe_offset=lane_id] = wk
                    min_edge_lo[unsafe_offset=lane_id] = lh[0]
                    min_edge_hi[unsafe_offset=lane_id] = lh[1]
                    min_edge_index[unsafe_offset=lane_id] = Int32(e)
            e += MST_WARP
    barrier()

    # `:76-85` reduce across the 32 lanes, halving.
    var offset = MST_WARP // 2
    while offset > 0:
        if lane_id < offset:
            # `:78` `min_edge_weight[lane_id] > min_edge_weight[lane_id + offset]`
            if triple_less(
                min_edge_wk[unsafe_offset=lane_id + offset],
                min_edge_lo[unsafe_offset=lane_id + offset],
                min_edge_hi[unsafe_offset=lane_id + offset],
                min_edge_wk[unsafe_offset=lane_id],
                min_edge_lo[unsafe_offset=lane_id],
                min_edge_hi[unsafe_offset=lane_id],
            ):
                min_edge_wk[unsafe_offset=lane_id] = min_edge_wk[unsafe_offset=lane_id + offset]
                min_edge_lo[unsafe_offset=lane_id] = min_edge_lo[unsafe_offset=lane_id + offset]
                min_edge_hi[unsafe_offset=lane_id] = min_edge_hi[unsafe_offset=lane_id + offset]
                min_edge_index[unsafe_offset=lane_id] = min_edge_index[unsafe_offset=lane_id + offset]
        barrier()
        offset //= 2

    # `:88-96` min edge may now be found in first thread
    if lane_id == 0 and warp_id < v:
        if min_edge_wk[unsafe_offset=0] != WEIGHT_KEY_SENTINEL:
            new_mst_edge.unsafe_store(warp_id, min_edge_index[unsafe_offset=0])
            _ = Atomic.min(
                min_edge_color.unsafe_offset(Int(self_color)), min_edge_wk[unsafe_offset=0]
            )


def min_edge_lo_per_color(
    offsets_row_of_edge: MutPointer[Int32, MutAnyOrigin],
    indices: MutPointer[Int32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    color: MutPointer[Int32, MutAnyOrigin],
    color_index: MutPointer[Int32, MutAnyOrigin],
    new_mst_edge: MutPointer[Int32, MutAnyOrigin],
    min_edge_color: MutPointer[Int32, MutAnyOrigin],
    min_edge_color_lo: MutPointer[Int32, MutAnyOrigin],
    v_in: Int32,
    sabotage: Int32,
):
    """DEVIATION 620, phase two. One thread per vertex: if its min edge's
    weight key equals its color's, push `min(u,v)` into the color's slot.
    `offsets_row_of_edge` is unused (the row is `tid` itself); it is kept
    in the signature so the three phases read alike."""
    var tid = get_1D_idx()
    if tid < Int(v_in):
        var edge_idx = new_mst_edge.unsafe_load(tid)
        if edge_idx != EDGE_SENTINEL:
            var c = color.unsafe_load(Int(color_index.unsafe_load(tid)))
            var wk = weight_order_key(weights.unsafe_load(Int(edge_idx)))
            if wk == min_edge_color.unsafe_load(Int(c)):
                var dst = indices.unsafe_load(Int(edge_idx))
                var lh = sabotaged_lo_hi(
                    sabotage, edge_lo(Int32(tid), dst), edge_hi(Int32(tid), dst)
                )
                _ = Atomic.min(min_edge_color_lo.unsafe_offset(Int(c)), lh[0])


def min_edge_hi_per_color(
    offsets_row_of_edge: MutPointer[Int32, MutAnyOrigin],
    indices: MutPointer[Int32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    color: MutPointer[Int32, MutAnyOrigin],
    color_index: MutPointer[Int32, MutAnyOrigin],
    new_mst_edge: MutPointer[Int32, MutAnyOrigin],
    min_edge_color: MutPointer[Int32, MutAnyOrigin],
    min_edge_color_lo: MutPointer[Int32, MutAnyOrigin],
    min_edge_color_hi: MutPointer[Int32, MutAnyOrigin],
    v_in: Int32,
    sabotage: Int32,
):
    """DEVIATION 620, phase three: `max(u,v)` among the vertices whose
    `(key, min)` equals the color's."""
    var tid = get_1D_idx()
    if tid < Int(v_in):
        var edge_idx = new_mst_edge.unsafe_load(tid)
        if edge_idx != EDGE_SENTINEL:
            var c = color.unsafe_load(Int(color_index.unsafe_load(tid)))
            var wk = weight_order_key(weights.unsafe_load(Int(edge_idx)))
            var dst = indices.unsafe_load(Int(edge_idx))
            var lh = sabotaged_lo_hi(
                sabotage, edge_lo(Int32(tid), dst), edge_hi(Int32(tid), dst)
            )
            if (
                wk == min_edge_color.unsafe_load(Int(c))
                and lh[0] == min_edge_color_lo.unsafe_load(Int(c))
            ):
                _ = Atomic.min(min_edge_color_hi.unsafe_offset(Int(c)), lh[1])


@always_inline
def _edge_is_color_min(
    src: Int32,
    edge_idx: Int32,
    c: Int32,
    indices: MutPointer[Int32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    min_edge_color: MutPointer[Int32, MutAnyOrigin],
    min_edge_color_lo: MutPointer[Int32, MutAnyOrigin],
    min_edge_color_hi: MutPointer[Int32, MutAnyOrigin],
    sabotage: Int32,
) -> Bool:
    """Their `min_edge_color[color] == altered_weights[edge_idx]`
    (`mst_kernels.cuh:127`, `:140`) on the triple."""
    var dst = indices.unsafe_load(Int(edge_idx))
    var wk = weight_order_key(weights.unsafe_load(Int(edge_idx)))
    var lh = sabotaged_lo_hi(sabotage, edge_lo(src, dst), edge_hi(src, dst))
    return (
        wk == min_edge_color.unsafe_load(Int(c))
        and lh[0] == min_edge_color_lo.unsafe_load(Int(c))
        and lh[1] == min_edge_color_hi.unsafe_load(Int(c))
    )


def min_edge_per_supervertex(
    color: MutPointer[Int32, MutAnyOrigin],
    color_index: MutPointer[Int32, MutAnyOrigin],
    new_mst_edge: MutPointer[Int32, MutAnyOrigin],
    mst_edge: MutPointer[UInt8, MutAnyOrigin],
    indices: MutPointer[Int32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    temp_src: MutPointer[Int32, MutAnyOrigin],
    temp_dst: MutPointer[Int32, MutAnyOrigin],
    temp_weights: MutPointer[Float32, MutAnyOrigin],
    min_edge_color: MutPointer[Int32, MutAnyOrigin],
    min_edge_color_lo: MutPointer[Int32, MutAnyOrigin],
    min_edge_color_hi: MutPointer[Int32, MutAnyOrigin],
    v_in: Int32,
    symmetrize_output: Int32,
    sabotage: Int32,
):
    """`mst_kernels.cuh:99-156`. `altered_weights` is the triple (DEVIATION
    620); everything else line for line, including the `!symmetrize_output`
    "vertices added each other" arm (`:131-143`), which is the arm
    `build_sorted_mst` takes (`cluster/detail/mst.cuh:297-298` passes
    `false, true`)."""
    var tid = get_1D_idx()
    if tid < Int(v_in):
        var vertex_color_idx = color_index.unsafe_load(tid)
        var vertex_color = color.unsafe_load(Int(vertex_color_idx))
        var edge_idx = new_mst_edge.unsafe_load(tid)

        if edge_idx != EDGE_SENTINEL:
            var add_edge = False
            if _edge_is_color_min(
                Int32(tid), edge_idx, vertex_color, indices, weights,
                min_edge_color, min_edge_color_lo, min_edge_color_hi,
                sabotage,
            ):
                add_edge = True
                var dst = indices.unsafe_load(Int(edge_idx))
                if symmetrize_output == 0:
                    var dst_edge_idx = new_mst_edge.unsafe_load(Int(dst))
                    var dst_color = color.unsafe_load(
                        Int(color_index.unsafe_load(Int(dst)))
                    )
                    # `:136-141` vertices added each other, only if the
                    # destination found an edge, it points back here, and it
                    # is the min edge of dst's color.
                    if (
                        dst_edge_idx != EDGE_SENTINEL
                        and indices.unsafe_load(Int(dst_edge_idx)) == Int32(tid)
                        and _edge_is_color_min(
                            dst, dst_edge_idx, dst_color, indices, weights,
                            min_edge_color, min_edge_color_lo,
                            min_edge_color_hi, sabotage,
                        )
                    ):
                        if vertex_color > dst_color:
                            add_edge = False

                if add_edge:
                    temp_src.unsafe_store(tid, Int32(tid))
                    temp_dst.unsafe_store(tid, dst)
                    temp_weights.unsafe_store(tid, weights.unsafe_load(Int(edge_idx)))
                    mst_edge.unsafe_store(Int(edge_idx), UInt8(1))

            if not add_edge:
                new_mst_edge.unsafe_store(tid, EDGE_SENTINEL)


def add_reverse_edge(
    new_mst_edge: MutPointer[Int32, MutAnyOrigin],
    indices: MutPointer[Int32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    temp_src: MutPointer[Int32, MutAnyOrigin],
    temp_dst: MutPointer[Int32, MutAnyOrigin],
    temp_weights: MutPointer[Float32, MutAnyOrigin],
    v_in: Int32,
    symmetrize_output: Int32,
):
    """`mst_kernels.cuh:158-204`. Only launched when `symmetrize_output`
    (`mst_solver_inl.cuh:342`); single linkage never does, so this kernel is
    ported and UNREACHED from `hierarchy/` (UNWIRED by design, see README)."""
    var tid = get_1D_idx()
    if tid < Int(v_in):
        var reverse_needed = False
        var edge_idx = new_mst_edge.unsafe_load(tid)
        if edge_idx != EDGE_SENTINEL:
            var neighbor_vertex = indices.unsafe_load(Int(edge_idx))
            var neighbor_edge_idx = new_mst_edge.unsafe_load(Int(neighbor_vertex))
            if neighbor_edge_idx == EDGE_SENTINEL:
                reverse_needed = True
            else:
                if symmetrize_output != 0:
                    var neighbor_vertex_neighbor = indices.unsafe_load(
                        Int(neighbor_edge_idx)
                    )
                    if Int32(tid) != neighbor_vertex_neighbor:
                        reverse_needed = True
            if reverse_needed:
                var v = Int(v_in)
                temp_src.unsafe_store(tid + v, neighbor_vertex)
                temp_dst.unsafe_store(tid + v, Int32(tid))
                temp_weights.unsafe_store(tid + v, weights.unsafe_load(Int(edge_idx)))


def min_pair_colors(
    v_in: Int32,
    indices: MutPointer[Int32, MutAnyOrigin],
    new_mst_edge: MutPointer[Int32, MutAnyOrigin],
    color: MutPointer[Int32, MutAnyOrigin],
    color_index: MutPointer[Int32, MutAnyOrigin],
    next_color: MutPointer[Int32, MutAnyOrigin],
):
    """`mst_kernels.cuh:206-237`. Integer `atomicMin`s, so the result of one
    launch is the same whatever order they land in."""
    var i = get_1D_idx()
    if i < Int(v_in):
        var edge_idx = new_mst_edge.unsafe_load(i)
        if edge_idx != EDGE_SENTINEL:
            var neighbor_vertex = indices.unsafe_load(Int(edge_idx))
            var self_color_idx = color_index.unsafe_load(i)
            var self_color = color.unsafe_load(Int(self_color_idx))
            var neighbor_color_idx = color_index.unsafe_load(Int(neighbor_vertex))
            var neighbor_super_color = color.unsafe_load(Int(neighbor_color_idx))
            _ = Atomic.min(
                next_color.unsafe_offset(Int(self_color_idx)), neighbor_super_color
            )
            _ = Atomic.min(
                next_color.unsafe_offset(Int(neighbor_color_idx)), self_color
            )


def update_colors(
    v_in: Int32,
    color: MutPointer[Int32, MutAnyOrigin],
    color_index: MutPointer[Int32, MutAnyOrigin],
    next_color: MutPointer[Int32, MutAnyOrigin],
    done: MutPointer[Int32, MutAnyOrigin],
):
    """`mst_kernels.cuh:239-260`. `done` is an Int32 flag (their `bool*`)."""
    var i = get_1D_idx()
    if i < Int(v_in):
        var self_color = color.unsafe_load(i)
        var self_color_idx = color_index.unsafe_load(i)
        var new_color = next_color.unsafe_load(Int(self_color_idx))
        if self_color > new_color:
            color.unsafe_store(i, new_color)
            done.unsafe_store(0, Int32(0))


def final_color_indices(
    v_in: Int32,
    color: MutPointer[Int32, MutAnyOrigin],
    color_index: MutPointer[Int32, MutAnyOrigin],
):
    """`mst_kernels.cuh:262-284`."""
    var i = get_1D_idx()
    if i < Int(v_in):
        var self_color_idx = color_index.unsafe_load(i)
        var self_color = color.unsafe_load(Int(self_color_idx))
        while self_color_idx != self_color:
            self_color_idx = color_index.unsafe_load(Int(self_color))
            self_color = color.unsafe_load(Int(self_color_idx))
        color_index.unsafe_store(i, self_color_idx)


def kernel_count_new_mst_edges(
    mst_src: MutPointer[Int32, MutAnyOrigin],
    mst_edge_count: MutPointer[Int32, MutAnyOrigin],
    v_in: Int32,
):
    """`mst_kernels.cuh:309-321`. Theirs counts per block with
    `__syncthreads_count` and adds once per block; ours adds once per
    qualifying thread. Both are INTEGER adds into one cell -- the total is
    the same whatever order they land in -- so the count is order-free
    either way and the block shape is not part of the result."""
    var tid = get_1D_idx()
    if tid < Int(v_in) and mst_src.unsafe_load(tid) != VERTEX_SENTINEL:
        _ = Atomic.fetch_add(mst_edge_count.unsafe_offset(0), Int32(1))


# ======================================================================
# The Thrust calls `MST_solver` makes, spelled as kernels. Thrust is OPEN
# (PORTING_RULES 0b-i) and these are what its calls do.
# ======================================================================

comptime MST_FILL_TPB = 256


def fill_i32_kernel(
    dst: MutPointer[Int32, MutAnyOrigin], value: Int32, n_in: Int32
):
    """`thrust::fill` on an Int32 range."""
    var i = get_1D_idx()
    if i < Int(n_in):
        dst.unsafe_store(i, value)


def fill_u8_kernel(
    dst: MutPointer[UInt8, MutAnyOrigin], value: UInt8, n_in: Int32
):
    """`cudaMemsetAsync` on the `bool` edge mask."""
    var i = get_1D_idx()
    if i < Int(n_in):
        dst.unsafe_store(i, value)


def sequence_i32_kernel(dst: MutPointer[Int32, MutAnyOrigin], n_in: Int32):
    """`thrust::sequence(..., 0)`."""
    var i = get_1D_idx()
    if i < Int(n_in):
        dst.unsafe_store(i, Int32(i))


def copy_i32_kernel(
    dst: MutPointer[Int32, MutAnyOrigin],
    src: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    """`raft::copy` between two device Int32 ranges."""
    var i = get_1D_idx()
    if i < Int(n_in):
        dst.unsafe_store(i, src.unsafe_load(i))


comptime COMPACT_TPB = 256


def compact_new_edges_kernel(
    temp_src: MutPointer[Int32, MutAnyOrigin],
    temp_dst: MutPointer[Int32, MutAnyOrigin],
    temp_weights: MutPointer[Float32, MutAnyOrigin],
    out_src: MutPointer[Int32, MutAnyOrigin],
    out_dst: MutPointer[Int32, MutAnyOrigin],
    out_weights: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    out_offset: Int32,
):
    """`thrust::copy_if(temp_src_dst_zip, ..., new_edges_functor)` of
    `MST_solver::append_src_dst_pair` (`mst_solver_inl.cuh:398-403`): a
    STABLE compaction of the slots whose `temp_src != max` onto the output
    from `out_offset`. ONE block, `COMPACT_TPB` threads, a Hillis-Steele
    exclusive scan of 0/1 flags per chunk in threadgroup memory and a
    running base in shared slot 0 -- integers throughout, so the output
    ORDER is the input order on every vendor (a `copy_if` is stable by
    contract, and this keeps that contract rather than an atomic counter's
    arbitrary one)."""
    var tid = Int(thread_idx.x)
    var flags = stack_allocation[
        COMPACT_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var base = stack_allocation[
        1, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    if tid == 0:
        base[unsafe_offset=0] = out_offset
    barrier()
    var n = Int(n_in)
    var start = 0
    while start < n:
        var i = start + tid
        var take = Int32(0)
        if i < n and temp_src.unsafe_load(i) != VERTEX_SENTINEL:
            take = Int32(1)
        flags[unsafe_offset=tid] = take
        barrier()
        # inclusive scan, Hillis-Steele
        var step = 1
        while step < COMPACT_TPB:
            var add = Int32(0)
            if tid >= step:
                add = flags[unsafe_offset=tid - step]
            barrier()
            flags[unsafe_offset=tid] = flags[unsafe_offset=tid] + add
            barrier()
            step *= 2
        if take != 0:
            var pos = Int(base[unsafe_offset=0]) + Int(flags[unsafe_offset=tid]) - 1
            out_src.unsafe_store(pos, temp_src.unsafe_load(i))
            out_dst.unsafe_store(pos, temp_dst.unsafe_load(i))
            out_weights.unsafe_store(pos, temp_weights.unsafe_load(i))
        barrier()
        if tid == 0:
            base[unsafe_offset=0] = base[unsafe_offset=0] + flags[unsafe_offset=COMPACT_TPB - 1]
        barrier()
        start += COMPACT_TPB
