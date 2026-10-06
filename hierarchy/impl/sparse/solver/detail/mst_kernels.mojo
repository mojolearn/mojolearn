from experiments.classical_identical_ideas.graph_controls import C34_PARALLEL_EDGES
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Boruvka's kernels, from RAFT.

Reference: `raft/cpp/include/raft/sparse/solver/detail/mst_kernels.cuh`
(RAFT `661a3b8`; the `raft-v26.08.00` checkout carries the same file), plus
`get_1D_idx` from `detail/mst_utils.cuh`. Kernels are in the reference's order,
with the ONE declared departure below.

DEVIATION 620 (see `hierarchy/checks/edge_order.mojo` for the block):
`alteration_kernel` (`mst_kernels.cuh:289-307`) is NOT implemented and nothing
reads `altered_weights`. Where the reference compares an altered `double`, this implementation
compares the triple `(weight_order_key(w), min(u,v), max(u,v))` through
`triple_less`, and the per-color minimum is taken in THREE integer
`atomicMin` phases instead of their one (`:94`): `kernel_min_edge_per_vertex`
publishes the weight key, then `min_edge_lo_per_color` publishes `min(u,v)`
among the vertices whose key equals the color's, then
`min_edge_hi_per_color` publishes `max(u,v)` among those. Each phase is an
integer min, so its result does not depend on which thread's atomic lands
first, and the three together are the lexicographic minimum of the triple.
Every other line matches the reference.

THE REFERENCE SHAPE THAT IS ALSO IDENTITY-SAFE. `kernel_min_edge_per_
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
from std.sys.compile import is_defined
from std.sys.defines import get_defined_int

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from hierarchy.checks.edge_order import (
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


#: fam2-cluster (2026-10-04), IDENTICAL. Four experiments on the Boruvka
#: solver; each has its own `_OFF` and all are off under
#: `-D MOJOLEARN_IDN_ALL_OFF=1`. None moves a bit: the MST under the total
#: edge order (DEVIATION 620) is unique, and every stage below is integer.
comptime _IDN_MST_ON = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)

#: `label_prop` without a readback. The round's kept edges are a forest
#: over the supervertices (one out-edge per color at most; of a mutual pair
#: the larger color's edge is dropped), so parent pointers + a BOUNDED run
#: of pointer jumps reach every tree's root, an integer `atomicMin` puts the
#: component's lowest color at the root, and every vertex takes it. That is
#: the fixed point the hop loop iterates to (every color becomes its
#: component's minimum), reached in a launch count the host knows up front,
#: where the hop loop drained the queue once per hop.
#: `-D MOJOLEARN_IDN_MST_LABEL_JUMP_OFF=1` restores the hop loop.
comptime IDN_MST_LABEL_JUMP = (
    _IDN_MST_ON and not is_defined["MOJOLEARN_IDN_MST_LABEL_JUMP_OFF"]()
)

#: The round loop decided on the device. A 4-cell state buffer holds
#: `[finished, prev_edge_count, rounds_run, edge_count]`; a one-thread kernel
#: closes each round (the host's `curr == prev` and `curr > max` tests), the
#: m^2 scan returns at once when `finished` is set, and every other kernel
#: of a finished round is a no-op on its own (no vertex finds an edge). The
#: host enqueues `ceil(log2 v) + 1` rounds (Boruvka's bound: each productive
#: round at least halves the supervertices that still have an out-edge) and
#: reads the state ONCE, where it read the count every round. Needs
#: IDN_MST_LABEL_JUMP. `-D MOJOLEARN_IDN_MST_ROUNDS_DEVICE_OFF=1` restores
#: the per-round read.
comptime IDN_MST_ROUNDS_DEVICE = (
    IDN_MST_LABEL_JUMP
    and not is_defined["MOJOLEARN_IDN_MST_ROUNDS_DEVICE_OFF"]()
)

#: `append_src_dst_pair`'s stable compaction as three parallel launches
#: (per-block counts, a scan of the block counts, per-block scan + scatter)
#: in place of ONE block walking all `2 v` slots chunk by chunk. Same
#: stable order. `-D MOJOLEARN_IDN_MST_PAR_COMPACT_OFF=1` restores it.
comptime IDN_MST_PAR_COMPACT = (
    _IDN_MST_ON and not is_defined["MOJOLEARN_IDN_MST_PAR_COMPACT_OFF"]()
)

#: CANDIDATE ARM (default 32 = the reference shape). Threads per row in
#: `kernel_min_edge_per_vertex`: a power of two, 32..1024. The row minimum
#: under a total order is the same at any lane count, so this is scheduling
#: only. Time `-D MOJOLEARN_IDN_MST_SCAN_LANES=64|128|256` against 32.
comptime IDN_MST_SCAN_LANES = 128 if C34_PARALLEL_EDGES else (
    get_defined_int["MOJOLEARN_IDN_MST_SCAN_LANES", 32]()
    if _IDN_MST_ON
    else 32
)


@always_inline
def _edge_dst[DENSE: Bool](
    indices: MutPointer[Int32, MutAnyOrigin], edge_idx: Int, v_in: Int32
) -> Int32:
    """The destination vertex of edge `edge_idx`. DENSE (lane/cluster-apple):
    the complete `v x v` graph `pairwise_distances` lays out row-major, whose
    `indices[e]` is `e % v` (`fill_indices2`), computed instead of loaded, so
    the dense caller never writes or reads the `v * v` index array. 32-bit:
    `v * v < 2^31` under PAIRWISE_MAX_ROWS."""
    comptime if DENSE:
        return Int32(UInt32(edge_idx) % UInt32(v_in))
    else:
        return indices.unsafe_load(edge_idx)


@always_inline
def get_1D_idx() -> Int:
    """`mst_utils.cuh:18-21`."""
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


def kernel_min_edge_per_vertex[
    DENSE: Bool = False, LANES: Int = MST_WARP, GATED: Bool = False
](
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
    rstate: MutPointer[Int32, MutAnyOrigin],
):
    """`mst_kernels.cuh:18-97`. One 32-thread block per row; each lane keeps
    the minimum (under `triple_less`) of the edges it scanned, the block
    folds the 32 partials, lane 0 publishes the row's min edge and pushes
    its WEIGHT KEY into the color's slot (their `atomicMin(&min_edge_color
    [self_color], min_edge_weight[0])`, `:94`, phase one of three).

    fam2-cluster: `LANES` threads per row (their 32 by default; a power of
    two, the block size the launch must use) and, when `GATED`, the whole
    block returns before any work if `rstate[0]` (the device's "finished"
    flag, IDN_MST_ROUNDS_DEVICE) is set. Every thread of a block reads the
    same cell, so the early return is uniform across the barriers."""
    comptime assert (
        LANES >= 32 and LANES <= 1024 and (LANES & (LANES - 1)) == 0
    ), "kernel_min_edge_per_vertex: LANES must be a power of two in 32..1024"
    comptime if GATED:
        if rstate.unsafe_load(0) != 0:
            return
    var tid = get_1D_idx()
    var warp_id = tid // LANES
    var lane_id = tid % LANES

    var min_edge_index = stack_allocation[
        LANES, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var min_edge_wk = stack_allocation[
        LANES, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var min_edge_lo = stack_allocation[
        LANES, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var min_edge_hi = stack_allocation[
        LANES, Scalar[DType.int32], address_space = AddressSpace.SHARED
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
            var successor: Int32
            comptime if DENSE:
                # row `warp_id` starts at warp_id * v: the column is e - start
                successor = Int32(e - row_start)
            else:
                successor = indices.unsafe_load(e)
            var successor_color_idx = color_index.unsafe_load(Int(successor))
            var successor_color = color.unsafe_load(Int(successor_color_idx))
            # the color test first: the same predicate, and an edge inside
            # one color never loads its `mst_edge` byte. DENSE reads no byte
            # at all: an edge is flagged only when a round adds it, and that
            # round's label_prop (run to its fixed point) gives both ends one
            # color, so every flagged edge already fails the color test.
            var fresh = True
            comptime if not DENSE:
                fresh = mst_edge.unsafe_load(e) == 0
            if self_color != successor_color and fresh:
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
            e += LANES
    barrier()

    # `:76-85` reduce across the 32 lanes, halving.
    var offset = LANES // 2
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


def min_edge_lo_per_color[DENSE: Bool = False](
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
                var dst = _edge_dst[DENSE](indices, Int(edge_idx), v_in)
                var lh = sabotaged_lo_hi(
                    sabotage, edge_lo(Int32(tid), dst), edge_hi(Int32(tid), dst)
                )
                _ = Atomic.min(min_edge_color_lo.unsafe_offset(Int(c)), lh[0])


def min_edge_hi_per_color[DENSE: Bool = False](
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
            var dst = _edge_dst[DENSE](indices, Int(edge_idx), v_in)
            var lh = sabotaged_lo_hi(
                sabotage, edge_lo(Int32(tid), dst), edge_hi(Int32(tid), dst)
            )
            if (
                wk == min_edge_color.unsafe_load(Int(c))
                and lh[0] == min_edge_color_lo.unsafe_load(Int(c))
            ):
                _ = Atomic.min(min_edge_color_hi.unsafe_offset(Int(c)), lh[1])


@always_inline
def _edge_is_color_min[DENSE: Bool = False](
    v_in: Int32,
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
    var dst = _edge_dst[DENSE](indices, Int(edge_idx), v_in)
    var wk = weight_order_key(weights.unsafe_load(Int(edge_idx)))
    var lh = sabotaged_lo_hi(sabotage, edge_lo(src, dst), edge_hi(src, dst))
    return (
        wk == min_edge_color.unsafe_load(Int(c))
        and lh[0] == min_edge_color_lo.unsafe_load(Int(c))
        and lh[1] == min_edge_color_hi.unsafe_load(Int(c))
    )


def min_edge_per_supervertex[DENSE: Bool = False](
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
            if _edge_is_color_min[DENSE](
                v_in, Int32(tid), edge_idx, vertex_color, indices, weights,
                min_edge_color, min_edge_color_lo, min_edge_color_hi,
                sabotage,
            ):
                add_edge = True
                var dst = _edge_dst[DENSE](indices, Int(edge_idx), v_in)
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
                        and _edge_dst[DENSE](indices, Int(dst_edge_idx), v_in) == Int32(tid)
                        and _edge_is_color_min[DENSE](
                            v_in, dst, dst_edge_idx, dst_color, indices, weights,
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
                    comptime if not DENSE:
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
    implemented and UNREACHED from `hierarchy/` (UNWIRED by design, see README)."""
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


def min_pair_colors[DENSE: Bool = False](
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
            var neighbor_vertex = _edge_dst[DENSE](indices, Int(edge_idx), v_in)
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
# (CONTRIBUTING.md (Algorithms and references)) and these are what its calls do.
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


# ======================================================================
# fam2-cluster (2026-10-04): label propagation by pointer jumping
# (IDN_MST_LABEL_JUMP), the device round state (IDN_MST_ROUNDS_DEVICE) and
# the parallel compaction (IDN_MST_PAR_COMPACT). Integers throughout.
# ======================================================================

comptime LP_NONE = Int32(0x7FFFFFFF)


def lp_parent_init(
    v_in: Int32,
    parent: MutPointer[Int32, MutAnyOrigin],
    cmin: MutPointer[Int32, MutAnyOrigin],
):
    """Every index its own parent; no component minimum yet."""
    var i = get_1D_idx()
    if i < Int(v_in):
        parent.unsafe_store(i, Int32(i))
        cmin.unsafe_store(i, LP_NONE)


def lp_parent_set[DENSE: Bool = False](
    v_in: Int32,
    indices: MutPointer[Int32, MutAnyOrigin],
    new_mst_edge: MutPointer[Int32, MutAnyOrigin],
    color_index: MutPointer[Int32, MutAnyOrigin],
    parent: MutPointer[Int32, MutAnyOrigin],
):
    """A supervertex whose kept edge leaves it points at the supervertex
    the edge reaches. At most one vertex per color keeps an edge after
    `min_edge_per_supervertex` (the color's min triple names one edge, and
    one of its ends is inside the color), so each cell has one writer."""
    var i = get_1D_idx()
    if i < Int(v_in):
        var edge_idx = new_mst_edge.unsafe_load(i)
        if edge_idx != EDGE_SENTINEL:
            var dst = _edge_dst[DENSE](indices, Int(edge_idx), v_in)
            parent.unsafe_store(
                Int(color_index.unsafe_load(i)),
                color_index.unsafe_load(Int(dst)),
            )


def lp_jump(v_in: Int32, parent: MutPointer[Int32, MutAnyOrigin]):
    """One pointer jump, in place. Every value a cell ever holds is one of
    its ancestors, so a racing read only jumps further; after k launches a
    cell is `min(2^k, depth)` steps up, and the root is the same whatever
    order the stores land in."""
    var i = get_1D_idx()
    if i < Int(v_in):
        var p = parent.unsafe_load(i)
        var gp = parent.unsafe_load(Int(p))
        if gp != p:
            parent.unsafe_store(i, gp)


def lp_min(
    v_in: Int32,
    color: MutPointer[Int32, MutAnyOrigin],
    color_index: MutPointer[Int32, MutAnyOrigin],
    parent: MutPointer[Int32, MutAnyOrigin],
    cmin: MutPointer[Int32, MutAnyOrigin],
):
    """Each supervertex (an index that is its own `color_index`) pushes its
    color into its root's cell: an integer `atomicMin`."""
    var i = get_1D_idx()
    if i < Int(v_in):
        if color_index.unsafe_load(i) == Int32(i):
            _ = Atomic.min(
                cmin.unsafe_offset(Int(parent.unsafe_load(i))),
                color.unsafe_load(i),
            )


def lp_color(
    v_in: Int32,
    color: MutPointer[Int32, MutAnyOrigin],
    color_index: MutPointer[Int32, MutAnyOrigin],
    parent: MutPointer[Int32, MutAnyOrigin],
    cmin: MutPointer[Int32, MutAnyOrigin],
):
    """Every vertex takes the lowest color of its supervertex's component:
    the hop loop's fixed point."""
    var i = get_1D_idx()
    if i < Int(v_in):
        var root = parent.unsafe_load(Int(color_index.unsafe_load(i)))
        var c = cmin.unsafe_load(Int(root))
        if c != LP_NONE:
            color.unsafe_store(i, c)


def round_close_kernel(
    mst_edge_count: MutPointer[Int32, MutAnyOrigin],
    rstate: MutPointer[Int32, MutAnyOrigin],
    max_edges: Int32,
):
    """The host's per-round tests, on the device (one thread).
    `rstate = [finished, prev_count, rounds_run, count]`; `finished` is 1 at
    the steady state (`curr == prev`) and 2 when the count passed
    `max_edges` (the host raises on it)."""
    if get_1D_idx() == 0:
        if rstate.unsafe_load(0) == 0:
            rstate.unsafe_store(2, rstate.unsafe_load(2) + 1)
            var c = mst_edge_count.unsafe_load(0)
            rstate.unsafe_store(3, c)
            if c > max_edges:
                rstate.unsafe_store(0, Int32(2))
            elif c == rstate.unsafe_load(1):
                rstate.unsafe_store(0, Int32(1))


def round_advance_kernel(
    mst_edge_count: MutPointer[Int32, MutAnyOrigin],
    rstate: MutPointer[Int32, MutAnyOrigin],
):
    """`prev_mst_edge_count = curr`, after the round's append (one thread)."""
    if get_1D_idx() == 0:
        if rstate.unsafe_load(0) == 0:
            rstate.unsafe_store(1, mst_edge_count.unsafe_load(0))


def compact_count_kernel(
    temp_src: MutPointer[Int32, MutAnyOrigin],
    bcount: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    """Pass 1 of the parallel compaction: `bcount[b]` = how many of block
    b's `COMPACT_TPB` slots are taken (a halving sum in threadgroup
    memory)."""
    var tid = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var flags = stack_allocation[
        COMPACT_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var i = blk * COMPACT_TPB + tid
    var take = Int32(0)
    if i < Int(n_in) and temp_src.unsafe_load(i) != VERTEX_SENTINEL:
        take = Int32(1)
    flags[unsafe_offset=tid] = take
    barrier()
    var offset = COMPACT_TPB // 2
    while offset > 0:
        if tid < offset:
            flags[unsafe_offset=tid] = (
                flags[unsafe_offset=tid] + flags[unsafe_offset=tid + offset]
            )
        barrier()
        offset //= 2
    if tid == 0:
        bcount.unsafe_store(blk, flags[unsafe_offset=0])


def compact_offsets_kernel(
    bcount: MutPointer[Int32, MutAnyOrigin], nb_in: Int32
):
    """Pass 2: `bcount` becomes its own exclusive prefix sum. ONE block over
    the `nb` block counts (`COMPACT_TPB` times fewer cells than the slots),
    the chunked Hillis-Steele scan of `compact_new_edges_kernel`."""
    var tid = Int(thread_idx.x)
    var flags = stack_allocation[
        COMPACT_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var base = stack_allocation[
        1, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    if tid == 0:
        base[unsafe_offset=0] = Int32(0)
    barrier()
    var nb = Int(nb_in)
    var start = 0
    while start < nb:
        var i = start + tid
        var own = Int32(0)
        if i < nb:
            own = bcount.unsafe_load(i)
        flags[unsafe_offset=tid] = own
        barrier()
        var step = 1
        while step < COMPACT_TPB:
            var add = Int32(0)
            if tid >= step:
                add = flags[unsafe_offset=tid - step]
            barrier()
            flags[unsafe_offset=tid] = flags[unsafe_offset=tid] + add
            barrier()
            step *= 2
        if i < nb:
            bcount.unsafe_store(
                i, base[unsafe_offset=0] + flags[unsafe_offset=tid] - own
            )
        barrier()
        if tid == 0:
            base[unsafe_offset=0] = (
                base[unsafe_offset=0] + flags[unsafe_offset=COMPACT_TPB - 1]
            )
        barrier()
        start += COMPACT_TPB


def compact_scatter_kernel[DEV: Bool = False](
    temp_src: MutPointer[Int32, MutAnyOrigin],
    temp_dst: MutPointer[Int32, MutAnyOrigin],
    temp_weights: MutPointer[Float32, MutAnyOrigin],
    out_src: MutPointer[Int32, MutAnyOrigin],
    out_dst: MutPointer[Int32, MutAnyOrigin],
    out_weights: MutPointer[Float32, MutAnyOrigin],
    bcount: MutPointer[Int32, MutAnyOrigin],
    rstate: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
    out_offset: Int32,
    out_cap: Int32,
):
    """Pass 3: each block scans its own flags and writes its taken slots at
    `offset + bcount[block] + local rank`: the stable order of the
    one-block kernel. `DEV`: the offset is the device's `prev_count`
    (`rstate[1]`) and a finished round writes nothing. A position at or
    past `out_cap` is not written (the host raises on that count)."""
    var tid = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var off = out_offset
    comptime if DEV:
        if rstate.unsafe_load(0) != 0:
            return
        off = rstate.unsafe_load(1)
    var flags = stack_allocation[
        COMPACT_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var i = blk * COMPACT_TPB + tid
    var take = Int32(0)
    if i < Int(n_in) and temp_src.unsafe_load(i) != VERTEX_SENTINEL:
        take = Int32(1)
    flags[unsafe_offset=tid] = take
    barrier()
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
        var pos = (
            Int(off) + Int(bcount.unsafe_load(blk))
            + Int(flags[unsafe_offset=tid]) - 1
        )
        if pos < Int(out_cap):
            out_src.unsafe_store(pos, temp_src.unsafe_load(i))
            out_dst.unsafe_store(pos, temp_dst.unsafe_load(i))
            out_weights.unsafe_store(pos, temp_weights.unsafe_load(i))
