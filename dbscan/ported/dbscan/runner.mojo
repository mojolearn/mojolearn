"""The DBSCAN driver: neighborhood, core points, CSR, label propagation.

PORT OF `cuml/cpp/src/dbscan/runner.cuh` at cuML `7e29955c`. Partial.
Do not improve.

Their step order, copied:

    VertexDeg   -> adj (boolean) and vd (degrees)
    CorePoints  -> core[i] = vd[i] >= min_pts
    AdjGraph    -> exclusive scan of vd, then adj_to_csr
    WeakCC      -> label propagation with core points as the filter

The one thing worth understanding before changing anything here is WHERE the
core-point restriction is applied. It is not in the graph. The CSR contains
every edge, including edges out of border points, and the restriction lives
in the labeler's `filter_op`. Moving it earlier looks like an optimization
and quietly changes the answer.

NOT PORTED, and named in `dbscan/UNPORTED.tsv`: their batching over rows,
their multi-GPU label exchange, and the monotonic relabelling that renumbers
clusters `0..k-1` to match scikit-learn. Cluster labels are arbitrary up to a
permutation, so the check compares the PARTITION rather than the numbers,
which is the same reason the k-means check compares centroids as a
permutation.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.expand_distances import expand_distances_kernel
from core.gemm import gemm_nt
from core.row_norms import NORM_TPB, row_norm_kernel
from cluster.mojo_only.reduce_by_key import copy_f32_kernel
from dbscan.ported.dbscan.adjgraph.algo import (
    SCAN_TPB,
    compact_adjacency_kernel,
    exclusive_scan_kernel,
)
from dbscan.ported.dbscan.corepoints.compute import core_points_kernel
from dbscan.ported.dbscan.vertexdeg.algo import (
    VD_TPB,
    eps_neighborhood_kernel,
)
from dbscan.ported.sparse.detail.csr import (
    MAX_LABEL,
    WEAK_CC_TPB,
    weak_cc_init_kernel,
    weak_cc_label_kernel,
)


def dbscan_fit(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut x_norm: DeviceBuffer[DType.float32],
    mut dist: DeviceBuffer[DType.float32],
    mut adj: DeviceBuffer[DType.uint8],
    mut vd: DeviceBuffer[DType.int32],
    mut core: DeviceBuffer[DType.uint8],
    mut ex_scan: DeviceBuffer[DType.int32],
    mut col_ind: DeviceBuffer[DType.int32],
    mut labels: DeviceBuffer[DType.int32],
    mut x_alias: DeviceBuffer[DType.float32],
    mut xn_alias: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_features: Int,
    eps: Float64,
    min_pts: Int,
    batch_size: Int = 0,
    max_iterations: Int = 200,
) raises -> Int:
    """Returns the number of label-propagation passes it took to converge.

    `eps` is squared once here, not per pair.

    ROWS ARE BATCHED, WHICH IS THEIRS AND IS A MEMORY BOUND NOT A SPEEDUP
    ------------------------------------------------------------------
    `cuml/cpp/src/dbscan/runner.cuh:121` computes
    `n_batches = ceildiv(n_owned_rows, batch_size)` and every stage after it
    works on `batch_size x N` rather than `N x N`. Their own comment at
    `:150` gives the reason: "adjacency graph has a worst case size of
    N * batch_size elements".

    Without it this port allocates an `N x N` distance matrix AND an `N x N`
    adjacency, so it dies of memory long before it dies of time. The
    benchmark caps DBSCAN at 4,000 rows purely for that reason.

    `batch_size = 0` means one batch, which is the previous behaviour.

    DEVIATION, and it is the honest half of this. Theirs runs
    vertexdeg -> adjgraph -> weak_cc PER BATCH and then combines the
    per-batch labelings with `MergeLabels::run` (`runner.cuh:379`). That
    merge is NOT PORTED (`dbscan/UNPORTED.tsv`), and it is where the
    correctness trap lives. So this batches the DISTANCE and ADJACENCY work,
    which is what bounds the memory, and keeps ONE `weak_cc` over the whole
    graph, which needs no merge. The cost is that the distances are computed
    TWICE, once to count degrees and once to emit the CSR, because a batch's
    adjacency cannot be kept. Two passes of arithmetic to avoid an unported
    merge and an `N x N` allocation.
    """
    var eps_sq = Float32(eps * eps)
    var batch = batch_size if batch_size > 0 else n_rows
    if batch > n_rows:
        batch = n_rows
    var n_batches = (n_rows + batch - 1) // batch

    ctx.enqueue_function[row_norm_kernel](
        x_norm.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(n_features),
        Int32(0),
        grid_dim=(n_rows, 1, 1),
        block_dim=(NORM_TPB, 1, 1),
    )
    # DBSCAN's distance matrix is the dataset against ITSELF, so both GEMM
    # operands are `x` and both norm operands are `x_norm`. Mojo will not
    # accept the same buffer as two mutable kernel arguments (PORTING.md 24),
    # so the transposed operand is an aliased copy.
    ctx.enqueue_function[copy_f32_kernel](
        x_alias.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(n_rows * n_features),
        grid_dim=((n_rows * n_features + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )
    ctx.enqueue_function[copy_f32_kernel](
        xn_alias.unsafe_ptr(),
        x_norm.unsafe_ptr(),
        Int32(n_rows),
        grid_dim=((n_rows + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )
    ctx.synchronize()

    # PASS 1 over the batches: degrees only. `adj` holds one batch.
    var b = 0
    while b < n_batches:
        var r0 = b * batch
        var rows = min(batch, n_rows - r0)
        var xb = x.create_sub_buffer[DType.float32](
            r0 * n_features, rows * n_features
        )
        var xnb = x_norm.create_sub_buffer[DType.float32](r0, rows)
        var vdb = vd.create_sub_buffer[DType.int32](r0, rows)
        gemm_nt(ctx, dist, xb, x_alias, rows, n_rows, n_features)
        ctx.enqueue_function[expand_distances_kernel](
            dist.unsafe_ptr(),
            xnb.unsafe_ptr(),
            xn_alias.unsafe_ptr(),
            Int32(rows),
            Int32(n_rows),
            Int32(0),
            grid_dim=((rows * n_rows + 255) // 256, 1, 1),
            block_dim=(256, 1, 1),
        )
        ctx.enqueue_function[eps_neighborhood_kernel](
            adj.unsafe_ptr(),
            vdb.unsafe_ptr(),
            dist.unsafe_ptr(),
            Int32(n_rows),
            eps_sq,
            grid_dim=(rows, 1, 1),
            block_dim=(VD_TPB, 1, 1),
        )
        b += 1
    ctx.synchronize()

    ctx.enqueue_function[core_points_kernel](
        core.unsafe_ptr(),
        vd.unsafe_ptr(),
        Int32(n_rows),
        Int32(min_pts),
        grid_dim=((n_rows + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )
    ctx.enqueue_function[exclusive_scan_kernel](
        ex_scan.unsafe_ptr(),
        vd.unsafe_ptr(),
        Int32(n_rows),
        grid_dim=(1, 1, 1),
        block_dim=(SCAN_TPB, 1, 1),
    )
    # PASS 2: recompute each batch's adjacency and emit its CSR rows. The
    # distances are recomputed rather than kept, which is the deviation the
    # docstring prices: a batch's `adj` cannot survive to here without the
    # `N x N` allocation the batching exists to remove.
    b = 0
    while b < n_batches:
        var r0b = b * batch
        var rowsb = min(batch, n_rows - r0b)
        var xb2 = x.create_sub_buffer[DType.float32](
            r0b * n_features, rowsb * n_features
        )
        var xnb2 = x_norm.create_sub_buffer[DType.float32](r0b, rowsb)
        var vdb2 = vd.create_sub_buffer[DType.int32](r0b, rowsb)
        var exb = ex_scan.create_sub_buffer[DType.int32](r0b, rowsb)
        gemm_nt(ctx, dist, xb2, x_alias, rowsb, n_rows, n_features)
        ctx.enqueue_function[expand_distances_kernel](
            dist.unsafe_ptr(),
            xnb2.unsafe_ptr(),
            xn_alias.unsafe_ptr(),
            Int32(rowsb),
            Int32(n_rows),
            Int32(0),
            grid_dim=((rowsb * n_rows + 255) // 256, 1, 1),
            block_dim=(256, 1, 1),
        )
        ctx.enqueue_function[eps_neighborhood_kernel](
            adj.unsafe_ptr(),
            vdb2.unsafe_ptr(),
            dist.unsafe_ptr(),
            Int32(n_rows),
            eps_sq,
            grid_dim=(rowsb, 1, 1),
            block_dim=(VD_TPB, 1, 1),
        )
        ctx.enqueue_function[compact_adjacency_kernel](
            col_ind.unsafe_ptr(),
            adj.unsafe_ptr(),
            exb.unsafe_ptr(),
            Int32(rowsb),
            Int32(n_rows),
            grid_dim=(rowsb, 1, 1),
            block_dim=(SCAN_TPB, 1, 1),
        )
        b += 1
    ctx.synchronize()

    ctx.enqueue_function[weak_cc_init_kernel](
        labels.unsafe_ptr(),
        core.unsafe_ptr(),
        Int32(n_rows),
        grid_dim=((n_rows + WEAK_CC_TPB - 1) // WEAK_CC_TPB, 1, 1),
        block_dim=(WEAK_CC_TPB, 1, 1),
    )
    ctx.synchronize()

    # Their loop: repeat the propagation until a pass changes nothing. The
    # flag comes back to the host each pass, which is theirs too
    # (`weak_cc_batched` copies `state.m` and the caller loops).
    var d_changed = ctx.enqueue_create_buffer[DType.int32](1)
    var h_changed = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.synchronize()

    var passes = 0
    for _it in range(max_iterations):
        h_changed.unsafe_ptr().unsafe_store(0, Int32(0))
        ctx.enqueue_copy(dst_buf=d_changed, src_ptr=h_changed.unsafe_ptr())
        ctx.enqueue_function[weak_cc_label_kernel](
            labels.unsafe_ptr(),
            ex_scan.unsafe_ptr(),
            col_ind.unsafe_ptr(),
            core.unsafe_ptr(),
            d_changed.unsafe_ptr(),
            Int32(n_rows),
            grid_dim=((n_rows + WEAK_CC_TPB - 1) // WEAK_CC_TPB, 1, 1),
            block_dim=(WEAK_CC_TPB, 1, 1),
        )
        ctx.synchronize()
        ctx.enqueue_copy(dst_ptr=h_changed.unsafe_ptr(), src_buf=d_changed)
        ctx.synchronize()
        passes += 1
        if h_changed.unsafe_ptr().unsafe_load(0) == Int32(0):
            break

    return passes
