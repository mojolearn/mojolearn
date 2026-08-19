"""The batch-size policy and the workspace allocation.

PORT OF `cuml/cpp/src/dbscan/dbscan.cuh::compute_batch_size` and
`dbscanFitImpl` at cuML `00094f7`. Partial (single node, L2, no
`sample_weight`, no `core_sample_indices`, brute force). Do not improve.

THIS IS THE FILE THAT WAS MISSING, AND ITS ABSENCE CAPPED THE BENCHMARK
-----------------------------------------------------------------------
`batch_size` was a hand-passed argument with a default of "one batch", so
every caller had to guess, and the benchmark guessed by capping DBSCAN at a
few thousand rows. Upstream never guesses: it queries the device, subtracts
the dataset, takes 80% of what is left, and divides by a per-row estimate.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from dbscan.ported.dbscan.adjgraph.algo import scan_blocks_needed
from dbscan.ported.dbscan.runner import EPS_NN_BRUTE_FORCE, EPS_NN_RBC, dbscan_fit


def compute_batch_size(
    n_rows: Int,
    n_owned_rows: Int,
    max_mbytes_per_batch: Int = 0,
    neigh_per_row: Int = 0,
) raises -> Int:
    """`compute_batch_size` (`dbscan.cuh:34`), copied.

    Their comment on `neigh_per_row`, which is why the worst case is not
    used: "In real applications, it's unlikely that the sparse adjacency
    matrix comes even close to the worst-case memory usage, because if
    epsilon is so large that all points are connected to 10% or even more of
    other points, the clusters would probably not be interesting/relevant
    anymore". Their `///@todo: expose neigh_per_row to the user` still
    stands, and `<= 0` still means `n_rows`.

    Index type is Int32 throughout this port, so `sizeof(Index_) == 4` and
    `MAX_LABEL == 2147483647`.
    """
    var npr = neigh_per_row
    if npr <= 0:
        npr = n_rows

    # Memory needed per batch row:
    #  - Dense adj matrix: n_rows (bool)
    #  - Sparse adj matrix: neigh_per_row (Index_)
    #  - Vertex degrees: 1 (Index_)
    #  - Ex scan: 1 (Index_)
    var est_mem_per_row = n_rows * 1 + (npr + 2) * 4
    # Memory needed regardless of the batch size:
    #  - Temporary labels: n_rows (Index_)
    #  - Core point mask: n_rows (bool)
    var est_mem_fixed = n_rows * (4 + 1)

    if est_mem_per_row <= 0:
        raise Error("Estimated memory per row is 0 for DBSCAN")

    var budget = max_mbytes_per_batch * 1000000
    if budget <= est_mem_fixed:
        raise Error(
            "DBSCAN has no memory left for a single batch row: the fixed"
            " cost is " + String(est_mem_fixed) + " bytes against a budget of "
            + String(budget)
        )
    var batch_size = (budget - est_mem_fixed) // est_mem_per_row

    # Limit batch size to number of owned rows
    if batch_size > n_owned_rows:
        batch_size = n_owned_rows

    # To avoid overflow, we need: batch_size <= MAX_LABEL / n_rows (floor div)
    var max_label = 2147483647
    if batch_size > max_label // n_rows:
        batch_size = max_label // n_rows

    if batch_size < 1:
        raise Error(
            "DBSCAN batch size came out as " + String(batch_size)
            + " rows; the dataset does not fit this device"
        )
    return batch_size


def dbscan_fit_impl(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut labels: DeviceBuffer[DType.int32],
    n_rows: Int,
    n_features: Int,
    eps: Float64,
    min_pts: Int,
    max_mbytes_per_batch: Int = 0,
    max_iterations: Int = 200,
    eps_nn_method: Int = EPS_NN_RBC,
) raises -> Int:
    """`dbscanFitImpl` (`dbscan.cuh:101`): size the batch, allocate, run.

    Their memory estimate, copied from `dbscan.cuh:147-151`:

        // The estimate is: 80% * total - dataset
        max_mbytes_per_batch = (80 * total_memory / 100 - dataset_memory)/1e6;

    with `total_memory` from `cudaMemGetInfo`, whose counterpart is
    `DeviceContext.get_memory_info()`. Their note above it -- "we can't rely
    on the reported free memory" -- is why the TOTAL and not the free figure
    is used, and it is kept.

    Allocation happens inside the call, as it does in theirs (their
    `rmm::device_uvector<char> workspace(workspaceSize, stream)` at
    `dbscan.cuh:197`, sized by a first `Dbscan::run` with a null workspace).
    """
    if n_rows <= 0:
        raise Error("No rows in the input array. DBSCAN cannot be fitted!")

    var budget_mb = max_mbytes_per_batch
    if budget_mb == 0:
        var mem = ctx.get_memory_info()
        var total_memory = Int(mem[1])
        var dataset_memory = n_rows * n_features * 4
        budget_mb = (80 * total_memory // 100 - dataset_memory) // 1000000

    var batch = compute_batch_size(n_rows, n_rows, budget_mb)

    var adj = ctx.enqueue_create_buffer[DType.uint8](batch * n_rows)
    var core = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    var vd = ctx.enqueue_create_buffer[DType.int32](batch + 1)
    var ex_scan = ctx.enqueue_create_buffer[DType.int32](batch + 1)
    var labels_temp = ctx.enqueue_create_buffer[DType.int32](n_rows)
    var work_buffer = ctx.enqueue_create_buffer[DType.int32](n_rows)
    var block_sums = ctx.enqueue_create_buffer[DType.int32](
        scan_blocks_needed(n_rows) + 1
    )
    ctx.synchronize()

    return dbscan_fit(
        ctx,
        x,
        adj,
        vd,
        core,
        ex_scan,
        labels,
        labels_temp,
        work_buffer,
        block_sums,
        n_rows,
        n_features,
        eps,
        min_pts,
        batch,
        max_iterations,
        eps_nn_method,
    )
