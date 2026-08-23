"""The connectivities graph single linkage hands its MST.

PORT OF `cuvs/cpp/src/cluster/detail/connectivities.cuh`, cuVS `94c2819`,
the `Linkage::PAIRWISE` specialization (`:110-204`) and `get_distance_graph`
(`:222-239`). The `Linkage::KNN_GRAPH` specialization (`:60-108`) is NOT
ported in this rung and is REFUSED BY NAME below; `hierarchy/UNPORTED.tsv`.

THE DISTANCE STEP, AND WHICH ARM. `pairwise_distances` (`:133-176`) calls
`cuvs::cluster::kmeans::detail::pairwise_distance_kmeans` (`kmeans_common.
cuh:292-322`), which for `L2Expanded`/`L2SqrtExpanded` is
`cuvs::distance::distance<...>`: the expanded identity `||x||^2 + ||y||^2 -
2 x.y` over precomputed row norms. This tree's spelling of that identity is
the one `neighbors/ported/neighbors/detail/knn_brute_force.mojo:160-202`
already uses for the same upstream call:

    FAST       `core/row_norms.row_norm_kernel` -> `core/gemm.gemm_nt` (MAX
               matmul) -> `core/expand_distances.expand_distances_kernel`
    IDENTICAL  `core/row_norms.row_norm_kernel` (pinned fold) ->
               `neighbors/mojo_only/pinned_distance_tile.pinned_distance_
               tile_kernel` (DEVIATION 505: one thread per cell, feature
               axis ascending through `identical_mul_add`/`ftz`, sqrt
               through `identical_sqrt`)

so the weight bytes this lane feeds the MST are pinned by the SAME kernels
the k-NN lane's are, and their pins are gated there. `X` is both operands,
so `z` is exactly symmetric under IDENTICAL: `x_i.x_j` accumulates the same
products in the same order from either side, and `||x_i||^2 + ||x_j||^2`
commutes exactly.

THE SELF-LOOP. `:162-175` sets the diagonal to `numeric_limits<value_t>::
max()` after the distance call, with a `thrust::transform` over the zipped
counting iterator; `self_loop_max_kernel` below is that transform.

THE `nnz` TYPE. Theirs is `value_idx nnz = m * m` (`:145`), an `int` that
overflows silently at `m >= 46341`. Refused by name at `pairwise_distances`
rather than inherited.
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from core.expand_distances import expand_distances_kernel
from core.gemm import gemm_nt
from core.row_norms import NORM_TPB, row_norm_kernel
from hierarchy.mojo_only.edge_order import LINK_SAB_NONE
from hierarchy.mojo_only.sabotage_tile import sabotage_distance_tile_kernel
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from neighbors.mojo_only.pinned_distance_tile import (
    PINNED_TILE_TPB,
    pinned_distance_tile_kernel,
)


comptime LINKAGE_PAIRWISE = 0
comptime LINKAGE_KNN_GRAPH = 1
"""`cuvs/cluster/agglomerative.hpp:40-55` `enum Linkage`."""

comptime DISTANCE_L2_EXPANDED = 0
comptime DISTANCE_L2_SQRT_EXPANDED = 1
comptime DISTANCE_COSINE_EXPANDED = 2
comptime DISTANCE_L1 = 3
"""`cuml/common/distance_type.hpp:14-17`, the values cuML's Python layer
passes (`agglomerative.pyx:36-43`: euclidean/l2 -> L2SqrtExpanded, l1 ->
L1, cosine -> CosineExpanded)."""

comptime PAIRWISE_MAX_ROWS = 46340
"""`m * m` fits their `int nnz` (`connectivities.cuh:145`) only up to here."""

comptime FLOAT32_MAX = Float32(3.4028234663852886e38)

comptime CONN_TPB = 256


def fill_indices2(
    indices: MutPointer[Int32, MutAnyOrigin], m_in: Int32, nnz_in: Int32
):
    """`connectivities.cuh:110-117`. `indices[tid] = tid % m`."""
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if tid >= Int(nnz_in):
        return
    indices.unsafe_store(tid, Int32(tid % Int(m_in)))


def indptr_sequence_kernel(
    indptr: MutPointer[Int32, MutAnyOrigin], m_in: Int32
):
    """`thrust::sequence(indptr, indptr + m, 0, m)` (`:150`) and
    `indptr[m] = nnz` (`:152`), one launch."""
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var m = Int(m_in)
    if tid <= m:
        indptr.unsafe_store(tid, Int32(tid * m))


def self_loop_max_kernel(
    data: MutPointer[Float32, MutAnyOrigin], m_in: Int32, nnz_in: Int32
):
    """`connectivities.cuh:162-175`: self-loops get max distance.
    `idx % m == idx / m` is the diagonal."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(nnz_in):
        return
    var m = Int(m_in)
    if idx % m == idx // m:
        data.unsafe_store(idx, FLOAT32_MAX)


def pairwise_distances(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    metric: Int,
    mut indptr: DeviceBuffer[DType.int32],
    mut indices: DeviceBuffer[DType.int32],
    mut data: DeviceBuffer[DType.float32],
    mut norms: DeviceBuffer[DType.float32],
    tile_tpb: Int = PINNED_TILE_TPB,
    sabotage: Int32 = LINK_SAB_NONE,
) raises:
    """`connectivities.cuh:133-176`. `norms` is the caller's `m`-long
    scratch for the row norms (theirs lives inside `cuvs::distance`);
    `tile_tpb` is the IDENTICAL tile's block size, a scheduling knob the
    check varies to show the bytes do not move."""
    if m < 2:
        raise Error(
            "hierarchy.pairwise_distances: n_rows=" + String(m)
            + " < 2; single linkage needs at least two points"
        )
    if m > PAIRWISE_MAX_ROWS:
        raise Error(
            "hierarchy.pairwise_distances: n_rows=" + String(m)
            + " > " + String(PAIRWISE_MAX_ROWS)
            + "; their `int nnz = m * m` (connectivities.cuh:145) overflows"
            " and the dense connectivity matrix is refused by name"
        )
    if metric != DISTANCE_L2_SQRT_EXPANDED and metric != DISTANCE_L2_EXPANDED:
        raise Error(
            "hierarchy.pairwise_distances: metric=" + String(metric)
            + " refused by name; only L2SqrtExpanded (1, cuML's 'euclidean'/"
            "'l2') and L2Expanded (0) are ported (pairwise_distance_kmeans"
            " raises on every other metric too, kmeans_common.cuh:320)"
        )
    var nnz = m * m

    # `:147-148`
    ctx.enqueue_function[fill_indices2](
        indices.unsafe_ptr(),
        Int32(m),
        Int32(nnz),
        grid_dim=((nnz + CONN_TPB - 1) // CONN_TPB, 1, 1),
        block_dim=(CONN_TPB, 1, 1),
    )
    # `:150-152`
    ctx.enqueue_function[indptr_sequence_kernel](
        indptr.unsafe_ptr(),
        Int32(m),
        grid_dim=((m + 1 + CONN_TPB - 1) // CONN_TPB, 1, 1),
        block_dim=(CONN_TPB, 1, 1),
    )

    # `:157-160` pairwise_distance_kmeans(X, X, data, metric): the expanded
    # identity over precomputed row norms.
    ctx.enqueue_function[row_norm_kernel](
        norms.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(n),
        Int32(0),
        grid_dim=(m, 1, 1),
        block_dim=(NORM_TPB, 1, 1),
    )
    var is_sqrt = Int32(1 if metric == DISTANCE_L2_SQRT_EXPANDED else 0)
    var cells = nnz
    # `X` is both operands and `norms` both norm vectors; Mojo refuses one
    # buffer passed twice to a launch, so the second operand is a VIEW of
    # the same memory (what `make_device_matrix_view(X, m, n)` twice is).
    var x_view = x.create_sub_buffer[DType.float32](0, m * n)
    var norms_view = norms.create_sub_buffer[DType.float32](0, m)
    if sabotage != LINK_SAB_NONE:
        # THE CHECK'S ARMS ONLY: `hierarchy/mojo_only/sabotage_tile.mojo`.
        ctx.enqueue_function[sabotage_distance_tile_kernel](
            data.unsafe_ptr(),
            x.unsafe_ptr(),
            x_view.unsafe_ptr(),
            norms.unsafe_ptr(),
            norms_view.unsafe_ptr(),
            Int32(m),
            Int32(m),
            Int32(n),
            is_sqrt,
            sabotage,
            grid_dim=((cells + tile_tpb - 1) // tile_tpb, 1, 1),
            block_dim=(tile_tpb, 1, 1),
        )
    else:
        comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
            ctx.enqueue_function[pinned_distance_tile_kernel](
                data.unsafe_ptr(),
                x.unsafe_ptr(),
                x_view.unsafe_ptr(),
                norms.unsafe_ptr(),
                norms_view.unsafe_ptr(),
                Int32(m),
                Int32(m),
                Int32(n),
                is_sqrt,
                grid_dim=((cells + tile_tpb - 1) // tile_tpb, 1, 1),
                block_dim=(tile_tpb, 1, 1),
            )
        else:
            gemm_nt(ctx, data, x, x_view, m, m, n)
            ctx.enqueue_function[expand_distances_kernel](
                data.unsafe_ptr(),
                norms.unsafe_ptr(),
                norms_view.unsafe_ptr(),
                Int32(m),
                Int32(m),
                is_sqrt,
                grid_dim=((cells + CONN_TPB - 1) // CONN_TPB, 1, 1),
                block_dim=(CONN_TPB, 1, 1),
            )

    # `:162-175` self-loops get max distance
    ctx.enqueue_function[self_loop_max_kernel](
        data.unsafe_ptr(),
        Int32(m),
        Int32(nnz),
        grid_dim=((nnz + CONN_TPB - 1) // CONN_TPB, 1, 1),
        block_dim=(CONN_TPB, 1, 1),
    )
    ctx.synchronize()
    _ = x_view^
    _ = norms_view^


def get_distance_graph(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    metric: Int,
    dist_type: Int,
    c: Int,
    mut indptr: DeviceBuffer[DType.int32],
    mut indices: DeviceBuffer[DType.int32],
    mut data: DeviceBuffer[DType.float32],
    mut norms: DeviceBuffer[DType.float32],
    tile_tpb: Int = PINNED_TILE_TPB,
    sabotage: Int32 = LINK_SAB_NONE,
) raises:
    """`connectivities.cuh:222-239`. `indptr` must hold `m + 1`, `indices`
    and `data` `m * m` (the PAIRWISE `resize`s at `:199-200`, done by the
    caller here because Mojo buffers do not resize)."""
    if dist_type == LINKAGE_KNN_GRAPH:
        raise Error(
            "hierarchy.get_distance_graph: Linkage::KNN_GRAPH (connectivity="
            "'knn', c=" + String(c) + ") refused by name: the knn-graph"
            " connectivity (connectivities.cuh:60-108, knn_graph.cuh) and"
            " the cross-component connection it needs (mst.cuh:75-123,"
            " cross_component_nn.cuh) are rung 2 and not ported;"
            " use connectivity='pairwise'"
        )
    if dist_type != LINKAGE_PAIRWISE:
        raise Error(
            "hierarchy.get_distance_graph: unknown Linkage " + String(dist_type)
        )
    pairwise_distances(
        ctx, x, m, n, metric, indptr, indices, data, norms, tile_tpb, sabotage
    )
