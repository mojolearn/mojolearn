# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The batch-size policy and the workspace allocation.

PORT OF `cuml/cpp/src/dbscan/dbscan.cuh::compute_batch_size` and
`dbscanFitImpl` at cuML `00094f7`. Partial (single node, no
`core_sample_indices`). Do not improve.

`sample_weight` is plumbed since 2026-09-01 and `metric` carries an L1 arm
that has no upstream (DEVIATION 27, `dbscan/impl/neighbors/
epsilon_neighborhood.mojo`). Neither changes the batch-size estimate:
`compute_batch_size` is `dbscan.cuh:34` and their `est_mem_per_row` and
`est_mem_fixed` count neither the weight array nor `wght_sum`, which is
theirs -- `runner.cuh:176-177` sizes `wght_sum` INSIDE the workspace but
`dbscan.cuh:55-60` does not count it in the per-row estimate. Copying that
gap rather than closing it keeps the batch count a pure function of
`(n_rows, budget)`, which is what
`check_dbscan_batch_count_invariance` and `check_dbscan_max_mbytes_moves_
the_batch` rest on.

`eps_nn_method` defaults to RBC here and to BRUTE_FORCE in theirs; that
DEVIATION and its measurement live at the top of
`dbscan/impl/dbscan/runner.mojo`, not here.

THEIR FIXED ALGORITHM CODES, WHICH ARE NOT USER-VISIBLE (`dbscan.cuh:118-122`)

    algo_vd  = (metric == Precomputed) ? 2 : 1     -> 1 for every L2 fit
    algo_adj = 1
    algo_ccl = 2                                   -> `final_relabel` ALWAYS runs

`algo_ccl = 2` is worth reading twice: the monotonic relabel at
`runner.cuh:412` is guarded by `if (algo_ccl == 2)` and `dbscanFitImpl`
hardcodes 2, so it is not optional in their dispatch and is not optional
here either.

THIS IS THE FILE THAT WAS MISSING, AND ITS ABSENCE CAPPED THE BENCHMARK
-----------------------------------------------------------------------
`batch_size` was a hand-passed argument with a default of "one batch", so
every caller had to guess, and the benchmark guessed by capping DBSCAN at a
few thousand rows. Upstream never guesses: it queries the device, subtracts
the dataset, takes 80% of what is left, and divides by a per-row estimate.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from dbscan.impl.dbscan.adjgraph.algo import scan_blocks_needed
from dbscan.impl.dbscan.runner import EPS_NN_BRUTE_FORCE, EPS_NN_RBC, dbscan_fit
from dbscan.impl.neighbors.epsilon_neighborhood import DBSCAN_METRIC_L2


def compute_batch_size(
    n_rows: Int,
    n_owned_rows: Int,
    eps_nn_method: Int,
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

    `eps_nn_method` is theirs (`dbscan.cuh:37`) and it gates ONE thing,
    exactly as theirs does at `:71`: the `batch_size <= MAX_LABEL / n_rows`
    clamp is skipped when the method is RBC, because the RBC arm emits CSR
    directly and never materializes the `N * batch_size` dense adjacency the
    clamp exists to keep addressable. Nothing else in the estimate varies by
    method -- not `est_mem_per_row`, not `est_mem_fixed` -- and that too is
    theirs: their `:55` and `:60` are method-blind, their runner sizes
    `adj_size` unconditionally even when RBC leaves it unread
    (`runner.cuh:169`), and neither budget counts the RBC index, which their
    runner allocates OUTSIDE the workspace (`runner.cuh:233-240`) just as
    `dbscan_fit` allocates its `rbc_*` buffers beside the workspace.

    Their `estimated_memory` out-parameter (`:96`) feeds one debug log line
    (`dbscan.cuh:171-173`) and is not returned here.

    DEVIATION 37 (archive/reference/PORTING.md): their `:66` computes
    `max_mbytes_per_batch * 1000000 - est_mem_fixed` in `size_t`, so a
    nonzero budget smaller than the fixed cost WRAPS, and the `min` at `:69`
    turns the wrap into a full-size batch. Ours raises instead. Their
    missing `batch_size >= 1` floor (a zero batch would reach
    `raft::ceildiv` at `runner.cuh:131` and divide by zero) is a raise here
    for the same reason.
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

    # `dbscan.cuh:71`: `if (eps_nn_method != EpsNnMethod::RBC)`. The clamp
    # guards the dense `N * batch_size` adjacency and the worst-case CSR the
    # brute arm can emit; the RBC arm materializes neither, and its int32
    # bound is the ACTUAL edge count, refused at the query site
    # (runner.mojo, the `nnz1 > MAX_LABEL` raise). Their `:86-94` info about
    # a smaller sufficient index type is dead for Index_ == int32 and is not
    # ported.
    if eps_nn_method != EPS_NN_RBC:
        # To avoid overflow, we need: batch_size <= MAX_LABEL / n_rows
        # (floor div)
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
    phase_timing: Bool = False,
    metric: Int = DBSCAN_METRIC_L2,
) raises -> Int:
    """The UNWEIGHTED fit. Signature preserved; `metric` appended.

    This is `dbscan_fit_impl_weighted` with `sample_weight == nullptr`, which
    is their `Dbscan::run(..., sample_weight = nullptr, ...)`. It is kept as
    its own entry point rather than folded into the weighted one because
    every existing caller in `bench/` and in this lane's checks passes these
    arguments positionally, and because an unweighted fit should not have to
    build a weight buffer to say it has none.
    """
    var no_weight = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.synchronize()
    return dbscan_fit_impl_weighted(
        ctx,
        x,
        labels,
        no_weight,
        n_rows,
        n_features,
        eps,
        min_pts,
        max_mbytes_per_batch,
        max_iterations,
        eps_nn_method,
        phase_timing,
        metric,
        False,
    )


def dbscan_fit_impl_weighted(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut labels: DeviceBuffer[DType.int32],
    mut sample_weight: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_features: Int,
    eps: Float64,
    min_pts: Int,
    max_mbytes_per_batch: Int = 0,
    max_iterations: Int = 200,
    eps_nn_method: Int = EPS_NN_RBC,
    phase_timing: Bool = False,
    metric: Int = DBSCAN_METRIC_L2,
    has_weights: Bool = False,
) raises -> Int:
    """`dbscanFitImpl` (`dbscan.cuh:101`): size the batch, allocate, run.

    Their memory estimate, copied from `dbscan.cuh:157-158` (the
    `max_mbytes_per_batch == 0` guard around it is `:147`):

        // The estimate is: 80% * total - dataset
        max_mbytes_per_batch = (80 * total_memory / 100 - dataset_memory)/1e6;

    with `total_memory` from `cudaMemGetInfo`, whose counterpart is
    `DeviceContext.get_memory_info()`. Their note above it -- "we can't rely
    on the reported free memory" -- is why the TOTAL and not the free figure
    is used, and it is kept.

    Allocation happens inside the call, as it does in theirs (their
    `rmm::device_uvector<char> workspace(workspaceSize, stream)` at
    `dbscan.cuh:197`, sized by a first `Dbscan::run` with a null workspace).

    `max_mbytes_per_batch` is THE user-facing memory knob and both its name
    and its default are theirs: the C++ public `fit` spells it
    `max_bytes_per_batch` while documenting it as MEGABYTES
    (`cuml/cpp/include/cuml/cluster/dbscan.hpp:54-56`, default 0 at `:73`),
    and the Python layer spells it `max_mbytes_per_batch` with `None -> 0`
    (`dbscan.pyx:302`, `:316-317`). The internal spelling here matches their
    internal one (`dbscan.cuh:38`, `:111`). 0 means the 80%-of-total
    estimate below (`:147`); any other value is used as given, unvalidated,
    which is also theirs.

    `phase_timing` prints `PHASE <name> batch <i>/<n> <ms>` per phase; see
    `dbscan_fit` and archive/reference/PORTING.md 38. Off, nothing prints.
    """
    if n_rows <= 0:
        raise Error("No rows in the input array. DBSCAN cannot be fitted!")

    var budget_mb = max_mbytes_per_batch
    if budget_mb == 0:
        var mem = ctx.get_memory_info()
        var total_memory = Int(mem[1])
        var dataset_memory = n_rows * n_features * 4
        budget_mb = (80 * total_memory // 100 - dataset_memory) // 1000000

    var batch = compute_batch_size(n_rows, n_rows, eps_nn_method, budget_mb)
    if phase_timing:
        print(
            "PHASE budget mbytes " + String(budget_mb) + " batch "
            + String(batch)
        )

    var adj = ctx.enqueue_create_buffer[DType.uint8](batch * n_rows)
    var core = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    var vd = ctx.enqueue_create_buffer[DType.int32](batch + 1)
    var ex_scan = ctx.enqueue_create_buffer[DType.int32](batch + 1)
    var labels_temp = ctx.enqueue_create_buffer[DType.int32](n_rows)
    var work_buffer = ctx.enqueue_create_buffer[DType.int32](n_rows)
    var block_sums = ctx.enqueue_create_buffer[DType.int32](
        scan_blocks_needed(n_rows) + 1
    )
    # `runner.cuh:176-177`: `sample_weight != nullptr ? alignTo(sizeof(Type_f)
    # * batch_size) : 0`. Mojo has no null `DeviceBuffer`, so an unweighted
    # fit gets a one-element placeholder that nothing reads; `has_weights` is
    # their `sample_weight != nullptr`.
    var wght_sum = ctx.enqueue_create_buffer[DType.float32](
        batch if has_weights else 1
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
        sample_weight,
        wght_sum,
        n_rows,
        n_features,
        eps,
        min_pts,
        batch,
        max_iterations,
        eps_nn_method,
        phase_timing,
        metric,
        has_weights,
    )
