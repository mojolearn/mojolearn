# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's DERIVATION_MAP.tsv and in this file's own docstring. See NOTICE.
"""RAFT's Boruvka MST solver: the control plane.

PORT OF `raft/cpp/include/raft/sparse/solver/mst_solver.cuh` (the
`Graph_COO` and `MST_solver` declarations) and
`raft/cpp/include/raft/sparse/solver/detail/mst_solver_inl.cuh` (their
bodies), RAFT `661a3b8`, plus the `mst()` entry of
`raft/cpp/include/raft/sparse/solver/mst.cuh`. One Mojo file for the
declaration header and its `_inl`, recorded in `hierarchy/DERIVATION_MAP.tsv`.
Transliterated, their order, their names. Do not improve.

WHAT IS NOT HERE, BY DECLARED DEVIATION. `alteration()`, `alteration_max()`,
`alteration_functor`, `curand_generate_uniformX`, the `altered_weights` and
`rand_values` buffers (`mst_solver_inl.cuh:40-52`, `:117-121`, `:171-238`):
DEVIATION 620, `hierarchy/original/edge_order.mojo`. In their place two
extra per-color buffers `min_edge_color_lo` / `min_edge_color_hi` and two
extra launches per round (`min_edge_per_vertex` below). The `sabotage`
argument exists so `linkage_check.mojo` can put a random tie-break back and
watch the gate fail; every production caller passes `LINK_SAB_NONE`.

HOST READBACKS ARE THEIRS. `mst_edge_count.value(stream)` (`:142`, `:147`,
`:163`, `:246`, `:384`) and `done.value(stream)` (`:262`) are synchronous
reads in their loop; ours are the same reads in the same places.
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from hierarchy.original.edge_order import (
    EDGE_SENTINEL,
    LINK_SAB_NONE,
    VERTEX_SENTINEL,
    WEIGHT_KEY_SENTINEL,
)
from hierarchy.derived.sparse.solver.detail.mst_kernels import (
    COMPACT_TPB,
    MST_FILL_TPB,
    MST_WARP,
    add_reverse_edge,
    compact_new_edges_kernel,
    copy_i32_kernel,
    fill_i32_kernel,
    fill_u8_kernel,
    final_color_indices,
    kernel_count_new_mst_edges,
    kernel_min_edge_per_vertex,
    min_edge_hi_per_color,
    min_edge_lo_per_color,
    min_edge_per_supervertex,
    min_pair_colors,
    sequence_i32_kernel,
    update_colors,
)


struct Graph_COO(Movable):
    """`mst_solver.cuh:18-29`. `src`, `dst`, `weights` are device vectors
    sized at construction (`max_mst_edges`), `n_edges` how many are live."""

    var src: DeviceBuffer[DType.int32]
    var dst: DeviceBuffer[DType.int32]
    var weights: DeviceBuffer[DType.float32]
    var n_edges: Int
    var n_rounds: Int
    """NOT THEIRS: how many Boruvka rounds `solve` ran, carried out for the
    identity card as an integer stage. Theirs keeps it in a local."""

    def __init__(out self, ctx: DeviceContext, size: Int) raises:
        var n = size if size > 0 else 1
        self.src = ctx.enqueue_create_buffer[DType.int32](n)
        self.dst = ctx.enqueue_create_buffer[DType.int32](n)
        self.weights = ctx.enqueue_create_buffer[DType.float32](n)
        self.n_edges = 0
        self.n_rounds = 0


def _blocks(n: Int, tpb: Int) -> Int:
    return (n + tpb - 1) // tpb if n > 0 else 1


def _read_scalar(
    ctx: DeviceContext,
    buf: DeviceBuffer[DType.int32],
    mut h_scalar: HostBuffer[DType.int32],
) raises -> Int:
    """`rmm::device_scalar::value(stream)`: a synchronous readback."""
    ctx.enqueue_copy(dst_ptr=h_scalar.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    return Int(h_scalar.unsafe_ptr().unsafe_load(0))


struct MST_solver(Movable):
    """`mst_solver.cuh:31-90` + `mst_solver_inl.cuh`.

    `max_blocks` / `max_threads` / `sm_count` (`:89-91`) come from the
    device properties in theirs and size the 1-D launches as
    `min(v, max_threads)` blocks of threads. Ours takes `tpb` as a
    constructor argument (default 256, their usual `maxThreadsPerBlock` is
    1024 on NVIDIA and 1024 on Apple) so `linkage_check.mojo` can launch the
    same problem at two block sizes and require the same bytes: the MST's
    kernels are all per-vertex or per-row with order-free atomics, so the
    block size is SCHEDULING and must not reach the output.
    """

    var v: Int
    var e: Int
    var symmetrize_output: Bool
    var initialize_colors: Bool
    var iterations: Int
    var tpb: Int
    var sabotage: Int32

    # CSR, borrowed (their `const edge_t* offsets` etc.)
    var offsets: DeviceBuffer[DType.int32]
    var indices: DeviceBuffer[DType.int32]
    var weights: DeviceBuffer[DType.float32]

    var color_index: DeviceBuffer[DType.int32]
    """The caller's `color` array: each vertex's supervertex representative."""
    var min_edge_color: DeviceBuffer[DType.int32]
    """Their `min_edge_color` (weight per color); the WEIGHT KEY here."""
    var min_edge_color_lo: DeviceBuffer[DType.int32]
    var min_edge_color_hi: DeviceBuffer[DType.int32]
    """DEVIATION 620: the second and third components of the per-color min."""
    var new_mst_edge: DeviceBuffer[DType.int32]
    var mst_edge_count: DeviceBuffer[DType.int32]
    var prev_mst_edge_count: Int
    """Theirs is a device scalar read back on the host each round; holding
    the readback IS their use of it (`:147`, `:159`, `:384`)."""
    var mst_edge: DeviceBuffer[DType.uint8]
    var next_color: DeviceBuffer[DType.int32]
    var color: DeviceBuffer[DType.int32]
    var temp_src: DeviceBuffer[DType.int32]
    var temp_dst: DeviceBuffer[DType.int32]
    var temp_weights: DeviceBuffer[DType.float32]
    var done: DeviceBuffer[DType.int32]
    var h_scalar: HostBuffer[DType.int32]
    var n_rounds: Int
    """How many Boruvka rounds ran; an integer stage for the identity card."""

    def __init__(
        out self,
        ctx: DeviceContext,
        offsets: DeviceBuffer[DType.int32],
        indices: DeviceBuffer[DType.int32],
        weights: DeviceBuffer[DType.float32],
        v: Int,
        e: Int,
        color: DeviceBuffer[DType.int32],
        symmetrize_output: Bool,
        initialize_colors: Bool,
        iterations: Int,
        tpb: Int = 256,
        sabotage: Int32 = LINK_SAB_NONE,
    ) raises:
        """`mst_solver_inl.cuh:54-106`."""
        self.v = v
        self.e = e
        self.symmetrize_output = symmetrize_output
        self.initialize_colors = initialize_colors
        self.iterations = iterations
        self.tpb = tpb
        self.sabotage = sabotage
        self.offsets = offsets
        self.indices = indices
        self.weights = weights
        self.color_index = color
        var vv = v if v > 0 else 1
        var ee = e if e > 0 else 1
        self.min_edge_color = ctx.enqueue_create_buffer[DType.int32](vv)
        self.min_edge_color_lo = ctx.enqueue_create_buffer[DType.int32](vv)
        self.min_edge_color_hi = ctx.enqueue_create_buffer[DType.int32](vv)
        self.new_mst_edge = ctx.enqueue_create_buffer[DType.int32](vv)
        self.mst_edge_count = ctx.enqueue_create_buffer[DType.int32](1)
        self.prev_mst_edge_count = 0
        self.mst_edge = ctx.enqueue_create_buffer[DType.uint8](ee)
        self.next_color = ctx.enqueue_create_buffer[DType.int32](vv)
        self.color = ctx.enqueue_create_buffer[DType.int32](vv)
        self.temp_src = ctx.enqueue_create_buffer[DType.int32](2 * vv)
        self.temp_dst = ctx.enqueue_create_buffer[DType.int32](2 * vv)
        self.temp_weights = ctx.enqueue_create_buffer[DType.float32](2 * vv)
        self.done = ctx.enqueue_create_buffer[DType.int32](1)
        self.h_scalar = ctx.enqueue_create_host_buffer[DType.int32](1)
        self.n_rounds = 0
        ctx.synchronize()

        # `:93-95` mst_edge_count = 0, prev = 0, mst_edge memset 0
        ctx.enqueue_memset(self.mst_edge_count, Int32(0))
        ctx.enqueue_function[fill_u8_kernel](
            self.mst_edge.unsafe_ptr(),
            UInt8(0),
            Int32(e),
            grid_dim=(_blocks(e, MST_FILL_TPB), 1, 1),
            block_dim=(MST_FILL_TPB, 1, 1),
        )
        # `:97-105` Initially, color holds the vertex id as color
        if initialize_colors:
            ctx.enqueue_function[sequence_i32_kernel](
                self.color.unsafe_ptr(),
                Int32(v),
                grid_dim=(_blocks(v, MST_FILL_TPB), 1, 1),
                block_dim=(MST_FILL_TPB, 1, 1),
            )
            ctx.enqueue_function[sequence_i32_kernel](
                self.color_index.unsafe_ptr(),
                Int32(v),
                grid_dim=(_blocks(v, MST_FILL_TPB), 1, 1),
                block_dim=(MST_FILL_TPB, 1, 1),
            )
        else:
            ctx.enqueue_function[copy_i32_kernel](
                self.color.unsafe_ptr(),
                self.color_index.unsafe_ptr(),
                Int32(v),
                grid_dim=(_blocks(v, MST_FILL_TPB), 1, 1),
                block_dim=(MST_FILL_TPB, 1, 1),
            )
        ctx.enqueue_function[sequence_i32_kernel](
            self.next_color.unsafe_ptr(),
            Int32(v),
            grid_dim=(_blocks(v, MST_FILL_TPB), 1, 1),
            block_dim=(MST_FILL_TPB, 1, 1),
        )
        ctx.synchronize()

    def solve(mut self, ctx: DeviceContext) raises -> Graph_COO:
        """`mst_solver_inl.cuh:108-169`."""
        if self.v <= 0:
            raise Error("mst: 0 vertices")
        if self.e <= 0:
            raise Error("mst: 0 edges")

        # `:117-121` alteration(): NOT PORTED, DEVIATION 620.

        var max_mst_edges = 2 * self.v - 2 if self.symmetrize_output else self.v - 1
        var mst_result = Graph_COO(ctx, max_mst_edges)

        # `:127-130` Boruvka original formulation says "while more than 1
        # supervertex remains"; adjusted to support spanning forests.
        var mst_iterations = self.iterations if self.iterations > 0 else self.v
        self.n_rounds = 0
        for _i in range(mst_iterations):
            self.min_edge_per_vertex(ctx)
            self.min_edge_per_supervertex(ctx)
            self.check_termination(ctx)
            self.n_rounds += 1

            var curr_mst_edge_count = _read_scalar(ctx, self.mst_edge_count, self.h_scalar)
            if curr_mst_edge_count > max_mst_edges:
                raise Error(
                    "mst: Number of edges found by MST is invalid. This may be"
                    " due to loss in precision. Try increasing precision of"
                    " weights. (found "
                    + String(curr_mst_edge_count)
                    + " > "
                    + String(max_mst_edges)
                    + ")"
                )
            if curr_mst_edge_count == self.prev_mst_edge_count:
                # `:147-150` exit here when reaching steady state
                break

            self.append_src_dst_pair(ctx, mst_result)
            self.label_prop(ctx)
            self.prev_mst_edge_count = curr_mst_edge_count

        # `:162-166` result packaging
        mst_result.n_edges = _read_scalar(ctx, self.mst_edge_count, self.h_scalar)
        mst_result.n_rounds = self.n_rounds
        return mst_result^

    def label_prop(mut self, ctx: DeviceContext) raises:
        """`mst_solver_inl.cuh:240-275`."""
        var blocks = _blocks(self.v, self.tpb)
        var done = False
        while not done:
            # done.set_value_async(true)
            ctx.enqueue_memset(self.done, Int32(1))
            ctx.enqueue_function[min_pair_colors](
                Int32(self.v),
                self.indices.unsafe_ptr(),
                self.new_mst_edge.unsafe_ptr(),
                self.color.unsafe_ptr(),
                self.color_index.unsafe_ptr(),
                self.next_color.unsafe_ptr(),
                grid_dim=(blocks, 1, 1),
                block_dim=(self.tpb, 1, 1),
            )
            ctx.enqueue_function[update_colors](
                Int32(self.v),
                self.color.unsafe_ptr(),
                self.color_index.unsafe_ptr(),
                self.next_color.unsafe_ptr(),
                self.done.unsafe_ptr(),
                grid_dim=(blocks, 1, 1),
                block_dim=(self.tpb, 1, 1),
            )
            done = _read_scalar(ctx, self.done, self.h_scalar) != 0
        ctx.enqueue_function[final_color_indices](
            Int32(self.v),
            self.color.unsafe_ptr(),
            self.color_index.unsafe_ptr(),
            grid_dim=(blocks, 1, 1),
            block_dim=(self.tpb, 1, 1),
        )
        ctx.synchronize()

    def min_edge_per_vertex(mut self, ctx: DeviceContext) raises:
        """`mst_solver_inl.cuh:277-304`, plus DEVIATION 620's two phases."""
        var vblocks = _blocks(self.v, MST_FILL_TPB)
        ctx.enqueue_function[fill_i32_kernel](
            self.min_edge_color.unsafe_ptr(),
            WEIGHT_KEY_SENTINEL,
            Int32(self.v),
            grid_dim=(vblocks, 1, 1),
            block_dim=(MST_FILL_TPB, 1, 1),
        )
        ctx.enqueue_function[fill_i32_kernel](
            self.min_edge_color_lo.unsafe_ptr(),
            VERTEX_SENTINEL,
            Int32(self.v),
            grid_dim=(vblocks, 1, 1),
            block_dim=(MST_FILL_TPB, 1, 1),
        )
        ctx.enqueue_function[fill_i32_kernel](
            self.min_edge_color_hi.unsafe_ptr(),
            VERTEX_SENTINEL,
            Int32(self.v),
            grid_dim=(vblocks, 1, 1),
            block_dim=(MST_FILL_TPB, 1, 1),
        )
        ctx.enqueue_function[fill_i32_kernel](
            self.new_mst_edge.unsafe_ptr(),
            EDGE_SENTINEL,
            Int32(self.v),
            grid_dim=(vblocks, 1, 1),
            block_dim=(MST_FILL_TPB, 1, 1),
        )
        # `:287`, `:295`: n_threads = 32, grid v -- one warp per row
        ctx.enqueue_function[kernel_min_edge_per_vertex](
            self.offsets.unsafe_ptr(),
            self.indices.unsafe_ptr(),
            self.weights.unsafe_ptr(),
            self.color.unsafe_ptr(),
            self.color_index.unsafe_ptr(),
            self.new_mst_edge.unsafe_ptr(),
            self.mst_edge.unsafe_ptr(),
            self.min_edge_color.unsafe_ptr(),
            Int32(self.v),
            self.sabotage,
            grid_dim=(self.v, 1, 1),
            block_dim=(MST_WARP, 1, 1),
        )
        var blocks = _blocks(self.v, self.tpb)
        ctx.enqueue_function[min_edge_lo_per_color](
            self.offsets.unsafe_ptr(),
            self.indices.unsafe_ptr(),
            self.weights.unsafe_ptr(),
            self.color.unsafe_ptr(),
            self.color_index.unsafe_ptr(),
            self.new_mst_edge.unsafe_ptr(),
            self.min_edge_color.unsafe_ptr(),
            self.min_edge_color_lo.unsafe_ptr(),
            Int32(self.v),
            self.sabotage,
            grid_dim=(blocks, 1, 1),
            block_dim=(self.tpb, 1, 1),
        )
        ctx.enqueue_function[min_edge_hi_per_color](
            self.offsets.unsafe_ptr(),
            self.indices.unsafe_ptr(),
            self.weights.unsafe_ptr(),
            self.color.unsafe_ptr(),
            self.color_index.unsafe_ptr(),
            self.new_mst_edge.unsafe_ptr(),
            self.min_edge_color.unsafe_ptr(),
            self.min_edge_color_lo.unsafe_ptr(),
            self.min_edge_color_hi.unsafe_ptr(),
            Int32(self.v),
            self.sabotage,
            grid_dim=(blocks, 1, 1),
            block_dim=(self.tpb, 1, 1),
        )

    def min_edge_per_supervertex(mut self, ctx: DeviceContext) raises:
        """`mst_solver_inl.cuh:306-352`."""
        var blocks = _blocks(self.v, self.tpb)
        ctx.enqueue_function[fill_i32_kernel](
            self.temp_src.unsafe_ptr(),
            VERTEX_SENTINEL,
            Int32(2 * self.v),
            grid_dim=(_blocks(2 * self.v, MST_FILL_TPB), 1, 1),
            block_dim=(MST_FILL_TPB, 1, 1),
        )
        ctx.enqueue_function[min_edge_per_supervertex](
            self.color.unsafe_ptr(),
            self.color_index.unsafe_ptr(),
            self.new_mst_edge.unsafe_ptr(),
            self.mst_edge.unsafe_ptr(),
            self.indices.unsafe_ptr(),
            self.weights.unsafe_ptr(),
            self.temp_src.unsafe_ptr(),
            self.temp_dst.unsafe_ptr(),
            self.temp_weights.unsafe_ptr(),
            self.min_edge_color.unsafe_ptr(),
            self.min_edge_color_lo.unsafe_ptr(),
            self.min_edge_color_hi.unsafe_ptr(),
            Int32(self.v),
            Int32(1 if self.symmetrize_output else 0),
            self.sabotage,
            grid_dim=(blocks, 1, 1),
            block_dim=(self.tpb, 1, 1),
        )
        if self.symmetrize_output:
            ctx.enqueue_function[add_reverse_edge](
                self.new_mst_edge.unsafe_ptr(),
                self.indices.unsafe_ptr(),
                self.weights.unsafe_ptr(),
                self.temp_src.unsafe_ptr(),
                self.temp_dst.unsafe_ptr(),
                self.temp_weights.unsafe_ptr(),
                Int32(self.v),
                Int32(1),
                grid_dim=(blocks, 1, 1),
                block_dim=(self.tpb, 1, 1),
            )

    def check_termination(mut self, ctx: DeviceContext) raises:
        """`mst_solver_inl.cuh:354-366`."""
        var n = 2 * self.v
        ctx.enqueue_function[kernel_count_new_mst_edges](
            self.temp_src.unsafe_ptr(),
            self.mst_edge_count.unsafe_ptr(),
            Int32(n),
            grid_dim=(_blocks(n, self.tpb), 1, 1),
            block_dim=(self.tpb, 1, 1),
        )

    def append_src_dst_pair(mut self, ctx: DeviceContext, mut mst_result: Graph_COO) raises:
        """`mst_solver_inl.cuh:378-404`: copy the new edges to the final
        output after the ones from previous rounds."""
        ctx.enqueue_function[compact_new_edges_kernel](
            self.temp_src.unsafe_ptr(),
            self.temp_dst.unsafe_ptr(),
            self.temp_weights.unsafe_ptr(),
            mst_result.src.unsafe_ptr(),
            mst_result.dst.unsafe_ptr(),
            mst_result.weights.unsafe_ptr(),
            Int32(2 * self.v),
            Int32(self.prev_mst_edge_count),
            grid_dim=(1, 1, 1),
            block_dim=(COMPACT_TPB, 1, 1),
        )


def mst(
    ctx: DeviceContext,
    offsets: DeviceBuffer[DType.int32],
    indices: DeviceBuffer[DType.int32],
    weights: DeviceBuffer[DType.float32],
    v: Int,
    e: Int,
    color: DeviceBuffer[DType.int32],
    symmetrize_output: Bool = True,
    initialize_colors: Bool = True,
    iterations: Int = 0,
    tpb: Int = 256,
    sabotage: Int32 = LINK_SAB_NONE,
) raises -> Graph_COO:
    """`raft/sparse/solver/mst.cuh:37-62`. Returns the MST (or MSF) edges;
    `Graph_COO.n_rounds` carries the round count for the card."""
    var solver = MST_solver(
        ctx, offsets, indices, weights, v, e, color,
        symmetrize_output, initialize_colors, iterations, tpb, sabotage,
    )
    return solver.solve(ctx)
