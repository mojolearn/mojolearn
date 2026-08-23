"""`build_sorted_mst`: the MST, connected, sorted.

PORT OF `cuvs/cpp/src/cluster/detail/mst.cuh`, cuVS `94c2819`:
`build_sorted_mst` (`:276-343`) and the shape of its fix-up loop. The two
`connect_knn_graph` overloads (`:74-123`, `:139-249`) and `merge_msts`
(`:39-59`) are NOT ported in this rung: on the PAIRWISE connectivity the
graph is complete, Boruvka's first call returns one component, and the
loop body is never entered; a call into it raises by name so that a rung-2
caller finds the gap loudly. `hierarchy/UNPORTED.tsv` has the row.

`get_n_components` (`cuvs/sparse/neighbors/cross_component_nn.cuh:44-47`
-> `detail/cross_component_nn.cuh`) counts distinct colors; ours copies the
`m` colors back and counts distinct values on the host. A count is
order-free and the array is `m` ints.
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from hierarchy.mojo_only.edge_order import LINK_SAB_NONE
from hierarchy.ported.sparse.op.sort import coo_sort_by_weight, merge_sort_u64_with_index
from hierarchy.ported.sparse.solver.mst_solver import Graph_COO, mst
from hierarchy.ported.sparse.solver.detail.mst_kernels import (
    MST_FILL_TPB,
    copy_i32_kernel,
)


def get_n_components(
    ctx: DeviceContext, mut color: DeviceBuffer[DType.int32], m: Int
) raises -> Int:
    """`cross_component_nn.cuh:44-47`: the number of distinct colors."""
    var h = ctx.enqueue_create_host_buffer[DType.int32](m)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=color)
    ctx.synchronize()
    var keys = List[UInt64](capacity=m)
    var idx = List[Int](capacity=m)
    for i in range(m):
        keys.append(UInt64(Int(h.unsafe_ptr().unsafe_load(i)) & 0x7FFFFFFF))
        idx.append(i)
    merge_sort_u64_with_index(keys, idx)
    var n = 0
    for i in range(m):
        if i == 0 or keys[i] != keys[i - 1]:
            n += 1
    _ = h^
    return n


def connect_knn_graph(
    ctx: DeviceContext, n_components: Int
) raises:
    """`mst.cuh:74-123` / `:139-249`. NOT PORTED (rung 2); raises by name."""
    raise Error(
        "hierarchy.build_sorted_mst: the MST is a forest ("
        + String(n_components)
        + " components) and connect_knn_graph (cross_component_nn, mst.cuh"
        ":74-123) is not ported in rung 1; on the PAIRWISE connectivity this"
        " cannot happen, so a non-finite input row is the likely cause"
    )


def build_sorted_mst(
    ctx: DeviceContext,
    mut indptr: DeviceBuffer[DType.int32],
    mut indices: DeviceBuffer[DType.int32],
    mut pw_dists: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    mut mst_src: DeviceBuffer[DType.int32],
    mut mst_dst: DeviceBuffer[DType.int32],
    mut mst_weight: DeviceBuffer[DType.float32],
    mut color: DeviceBuffer[DType.int32],
    nnz: Int,
    max_iter: Int = 10,
    mst_tpb: Int = 256,
    sabotage: Int32 = LINK_SAB_NONE,
) raises -> Int:
    """`mst.cuh:276-343`. Returns the Boruvka round count (for the card).
    `reduction_op` and `metric` are arguments of the fix-up loop only and
    have no role until rung 2."""
    # `:296-298` We want to have MST initialize colors on first call.
    var mst_coo = mst(
        ctx, indptr, indices, pw_dists, m, nnz, color,
        symmetrize_output=False, initialize_colors=True, iterations=0,
        tpb=mst_tpb, sabotage=sabotage,
    )

    var iters = 1
    var n_components = get_n_components(ctx, color, m)

    while n_components > 1 and iters < max_iter:
        connect_knn_graph(ctx, n_components)
        iters += 1
        n_components = get_n_components(ctx, color, m)

    # `:330-335`
    if n_components != 1:
        raise Error(
            "hierarchy.build_sorted_mst: KNN graph could not be connected in "
            + String(max_iter)
            + " iterations. Please verify that the input knn graph is"
            " generated from X (and the same distance metric used), or"
            " increase 'max_iter'"
        )
    if mst_coo.n_edges != m - 1:
        raise Error(
            "hierarchy.build_sorted_mst: the MST has "
            + String(mst_coo.n_edges)
            + " edges for "
            + String(m)
            + " vertices; a spanning tree has m - 1"
        )

    # `:337-338`
    coo_sort_by_weight(
        ctx, mst_coo.src, mst_coo.dst, mst_coo.weights, mst_coo.n_edges, sabotage
    )

    # `:340-342`
    var n_edges = mst_coo.n_edges
    var blocks = (n_edges + MST_FILL_TPB - 1) // MST_FILL_TPB if n_edges > 0 else 1
    ctx.enqueue_function[copy_i32_kernel](
        mst_src.unsafe_ptr(),
        mst_coo.src.unsafe_ptr(),
        Int32(n_edges),
        grid_dim=(blocks, 1, 1),
        block_dim=(MST_FILL_TPB, 1, 1),
    )
    ctx.enqueue_function[copy_i32_kernel](
        mst_dst.unsafe_ptr(),
        mst_coo.dst.unsafe_ptr(),
        Int32(n_edges),
        grid_dim=(blocks, 1, 1),
        block_dim=(MST_FILL_TPB, 1, 1),
    )
    if n_edges > 0:
        var wsrc = mst_coo.weights.create_sub_buffer[DType.float32](0, n_edges)
        var wdst = mst_weight.create_sub_buffer[DType.float32](0, n_edges)
        ctx.enqueue_copy(dst_buf=wdst, src_buf=wsrc)
        _ = wsrc^
        _ = wdst^
    ctx.synchronize()
    var rounds = mst_coo.n_rounds
    _ = mst_coo^
    return rounds
