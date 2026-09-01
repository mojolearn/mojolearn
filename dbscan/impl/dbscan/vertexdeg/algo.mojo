# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Epsilon neighborhood: the boolean adjacency and the vertex degree.

PORT OF `cuml/cpp/src/dbscan/vertexdeg/algo.cuh::launcher` at cuML `00094f7`.
Partial. Do not improve.

Their `launcher` is a metric switch and then ONE call. For the L2 arm with no
ball-cover index -- which is `algo == 1`, `metric == L2SqrtExpanded` /
`L2SqrtUnexpanded`, `eps_nn_method == BRUTE_FORCE`, the default dispatch for
these parameters -- it is `algo.cuh:224-231`:

    eps2 = data.eps * data.eps;
    raft::neighbors::epsilon_neighborhood::epsUnexpL2SqNeighborhood<value_t,
      index_t>(data.adj, data.vd, data.x + start_vertex_id * k, data.x,
               n, m, k, eps2, stream);

so that is what `vertex_deg_run` calls on its L2 arm, and the kernel itself
lives beside its own upstream in
`dbscan/impl/neighbors/epsilon_neighborhood.mojo`.

**ON THE L2 ARM `eps` IS SQUARED ONCE ON THE HOST AND NEVER PER PAIR**
(`algo.cuh:225`). DBSCAN's radius is a distance and their accumulator is a
squared distance; squaring the threshold instead of rooting a million
distances is theirs.

**ON THE L1 ARM IT IS NOT SQUARED AT ALL**, because an L1 sum has no squared
form. That is DEVIATION 27, it lives in the kernel file it changes, and the
one function both sides call is `dbscan_metric_threshold`. `vertex_deg_run`
therefore no longer computes `eps * eps` itself; computing it here as well
would be the second opinion the deviation exists to prevent.

A point is its own neighbor here, exactly as in scikit-learn: the distance to
itself is zero, which is `<= eps`, so `vd` includes the point and `min_pts`
counts it. Getting that wrong shifts every core-point decision by one and
produces a plausible clustering that disagrees with sklearn everywhere.

THE SAMPLE-WEIGHT ARM IS PORTED (2026-09-01), AND IT ENTERS IN EXACTLY ONE
PLACE
--------------------------------------------------------------------------
`launcher`'s tail (`algo.cuh:234-257`) is the whole of it, and reading their
runner beside it is what makes the change small enough to trust. The weight
does NOT change the neighborhood, the adjacency, the CSR, the propagation or
the relabel. It changes ONE predicate: `runner.cuh:300-306`

    if (wght_sum != nullptr) {
      CorePoints::compute<Type_f, Index_>(handle, wght_sum, core_pts, min_pts, ...);
    } else {
      CorePoints::compute<Index_, Index_>(handle, vd, core_pts, min_pts, ...);
    }

so a point is core when the SUM OF ITS NEIGHBORS' WEIGHTS reaches `min_pts`
rather than when their COUNT does. scikit-learn's `_dbscan.py:451-455` is
the same sentence (`np.sum(sample_weight[neighbors]) >= min_samples`), and
its docstring's "a sample with a weight of at least min_samples is by itself
a core sample" falls out of the point being its own neighbor.

THEIR CAST IS NOT A BUG AND IS NOT PORTED AS ONE. `compute.cuh:50` is
`mask[idx + start] = (Index_)vd[idx] >= min_pts` with `vd` templated on the
VALUE type, so on the weighted arm a Float32 weight sum is truncated to
Int32 before the compare. For `min_pts >= 1` (enforced at the estimator) and
a non-negative sum that is exactly the same predicate: `trunc(w) >= p` and
`w >= p` agree for integer `p` when `w >= 0`, since `trunc` is monotone and
`trunc(p) == p`. This port compares the FLOAT directly, which is the same
answer without depending on that argument, and `check_dbscan_weighted_core_
matches_host_oracle` gates the predicate rather than the spelling.

Two producers of the weighted degree, because the two arms of `launcher`
produce two different data structures, and both are ported:
- `coalescedReduction` over the dense `adj` (`algo.cuh:243-254`), the
  brute-force arm. Ours is `weighted_vertex_deg_dense_kernel`.
- `accumulateWeights` over the CSR (`algo.cuh:62-91` and `:236-238`), the
  ball-cover arm. Ours is `weighted_vertex_deg_csr_kernel`.
See DEVIATION 28 below for the fold, which is the whole numeric risk.

NOT PORTED FROM THIS FILE, and named so it is not forgotten:
- the `CosineExpanded` arm (`:186-223`): row-normalize in place, use
  `eps2 = 2 * eps`, run the same neighborhood kernel, then un-normalize.
- `Precomputed::launcher` (`algo == 2`), for a precomputed distance matrix.
- the ball-cover arm, `eps_nn`, which is a different lane.

DEVIATION BLOCK 28: THE WEIGHTED DEGREE'S FOLD IS PINNED, AND THAT IS A
REPLACEMENT OF BOTH OF THEIR REDUCERS
-----------------------------------------------------------------------
THEIRS, dense arm: `raft::linalg::coalescedReduction<bool, value_t,
index_t>` (`algo.cuh:243`), a RAFT thread-strided reduction closed by CUB.
THEIRS, CSR arm: one WARP per row and `cub::WarpReduce<math_t>(...).Sum(...)`
(`algo.cuh:72-88`), 32 lanes wide because `warpsize` defaults to 32 at the
call site (`:236`).

OURS: one BLOCK per row in both, `WVD_TPB` threads striding the row
ascending, closed by `core/pinned_reduce.pinned_block_sum[WVD_TPB]`.

REASON, and it is the reason `pinned_reduce.mojo` exists at all. This is a
FLOAT sum, so the fold shape is the answer's last bits, and the last bits
decide `>= min_pts` for any point whose weight sum lands on the threshold.
Both of their reducers close on a lane-width-shaped stage: CUB folds at the
HARDWARE warp width, 32 on Apple and NVIDIA and 64 on a CDNA wavefront, so a
faithful port would make an AMD fit disagree with a CUDA fit about which
points are core, and from there about how many clusters there are. That is
not a last-bit difference in a reported number, it is a different model.
`pinned_block_sum` is a halving tree over threadgroup memory with no warp
primitive in it, so the sequence of additions is a pure function of
`WVD_TPB` and nothing consults the hardware.

`WVD_TPB` IS THEREFORE NUMERIC AND IS READ FROM THE MATRIX, not written
here: it is simultaneously the fold's width AND the stride the per-thread
partials are taken at, which is DEVIATION 524's shape exactly (two columns
carrying two values would build two different multisets of partials and then
fold each of them correctly). `checks/kernel_matrix.mojo` lists
`K_LIB_WEIGHTED_VERTEX_DEG` in `lib_block_bounds_a_float_fold`, so under
IDENTICAL it resolves to the identity floor on every vendor.
`check_dbscan_weighted_fold_is_pinned` gates that from this side.

WHAT THE TWO ARMS DO AND DO NOT PROMISE EACH OTHER. Each arm is a pure
function of its inputs and is the same on every vendor. They are NOT
promised to agree bit for bit with one another in general, because they sum
the same multiset in two different assignments: the dense arm's thread `t`
takes COLUMNS `t, t + B, ...` and the CSR arm's takes the row's `t`-th,
`(t + B)`-th, ... NEIGHBOR. They DO agree exactly whenever every weight and
every partial sum is exactly representable -- integer weights with a total
below 2^24, which is the regime of "uniform weights of 1 reproduce the
unweighted labels" and of "duplicating a point equals weight 2" -- because
float addition is exact there and order stops mattering. That is what
`check_dbscan_weighted_arms_agree` asserts, and it says so in its own
docstring rather than pretending to a stronger claim.

Under IDENTICAL the CSR arm's rows arrive in ASCENDING COLUMN ORDER, which
is not luck: DEVIATION 551 (`neighbors/checks/ball_cover_canonical_order.
mojo`) rewrites every ball-cover CSR row into ascending column index under
`PIN_CROSS_VENDOR`, precisely so `adj_ja` is comparable across vendors by a
byte compare. Without it a float sum over a ball-cover row would be ordered
by the emission ballot, which is lane-width shaped; with it the CSR arm's
order is the dense arm's order restricted to the neighbors.
"""

from checks.kernel_matrix import (
    K_LIB_EPS_NEIGHBORHOOD,
    K_LIB_WEIGHTED_VERTEX_DEG,
    TARGET_COLUMN,
    lib_block_size_for,
)


from std.atomic import Atomic
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.primitives.block import sum as block_sum
from max.gpu.sync import barrier
from std.memory import stack_allocation

from checks.numerics import ftz
from core.pinned_reduce import pinned_block_sum
from dbscan.impl.neighbors.epsilon_neighborhood import (
    DBSCAN_METRIC_L1,
    DBSCAN_METRIC_L2,
    dbscan_metric_threshold,
    eps_unexp_neighborhood,
)


# READ FROM THE MATRIX, not restated here. `checks/kernel_matrix.mojo`
# owns every tunable in this tree; changing TARGET_COLUMN there rebuilds
# this kernel for another vendor with no edit in this file.
comptime VD_TPB = lib_block_size_for[K_LIB_EPS_NEIGHBORHOOD, TARGET_COLUMN]()

#: NUMERIC, not scheduling. The weighted degree's fold width AND the stride
#: its per-thread partials are taken at -- see DEVIATION 28 above and the
#: `K_LIB_WEIGHTED_VERTEX_DEG` entry in `lib_block_bounds_a_float_fold`.
#: Under IDENTICAL this resolves to the identity floor on every vendor, so
#: two vendors fold the same multiset in the same order.
comptime WVD_TPB = lib_block_size_for[
    K_LIB_WEIGHTED_VERTEX_DEG, TARGET_COLUMN
]()


def vertex_deg_run[
    metric: Int
](
    ctx: DeviceContext,
    mut adj: DeviceBuffer[DType.uint8],
    mut vd: DeviceBuffer[DType.int32],
    mut x: DeviceBuffer[DType.float32],
    start_vertex_id: Int,
    n_points: Int,
    n_rows: Int,
    n_features: Int,
    eps: Float64,
) raises:
    """`VertexDeg::run` -> `Algo::launcher`, the brute-force arm.

    `n_points` is their `n` (the batch), `n_rows` their `m` (the dataset).

    `metric` is `DBSCAN_METRIC_L2` (theirs) or `DBSCAN_METRIC_L1`
    (DEVIATION 27). The threshold comes from `dbscan_metric_threshold`, which
    is the ONE place that knows the L2 arm squares it and the L1 arm does
    not.
    """
    eps_unexp_neighborhood[metric](
        ctx,
        adj,
        vd,
        x,
        start_vertex_id,
        n_points,
        n_rows,
        n_features,
        dbscan_metric_threshold(metric, eps),
    )


def vertex_deg_dispatch(
    ctx: DeviceContext,
    mut adj: DeviceBuffer[DType.uint8],
    mut vd: DeviceBuffer[DType.int32],
    mut x: DeviceBuffer[DType.float32],
    start_vertex_id: Int,
    n_points: Int,
    n_rows: Int,
    n_features: Int,
    eps: Float64,
    metric: Int,
) raises:
    """`launcher`'s metric switch (`algo.cuh:186`), as a HOST branch.

    Their switch is a runtime `if` on `ML::distance::DistanceType`; ours
    picks a comptime instantiation, so the branch is taken once per batch on
    the host and never inside the accumulate loop. It lives HERE, in one
    function, so the runner has no opinion about metrics and so a new arm is
    one `elif` rather than a search through the driver.
    """
    if metric == DBSCAN_METRIC_L1:
        vertex_deg_run[DBSCAN_METRIC_L1](
            ctx, adj, vd, x, start_vertex_id, n_points, n_rows, n_features,
            eps,
        )
    else:
        vertex_deg_run[DBSCAN_METRIC_L2](
            ctx, adj, vd, x, start_vertex_id, n_points, n_rows, n_features,
            eps,
        )


def weighted_vertex_deg_dense_kernel(
    wght_sum: MutPointer[Float32, MutAnyOrigin],
    adj: MutPointer[UInt8, MutAnyOrigin],
    sample_weight: MutPointer[Float32, MutAnyOrigin],
    n_cols_in: Int32,
    n_rows_in: Int32,
):
    """`coalescedReduction` over `adj` with `adj_ij ? sample_weight[j] : 0`.

    `algo.cuh:243-254`, the brute-force half of their sample-weight arm.
    `adj` is the batch's `n_points x n_cols` boolean adjacency, `n_cols` is
    their `data.N` (the whole dataset), and `wght_sum[i]` is the batch row's
    weighted degree.

    ONE BLOCK PER ROW. Thread `t` walks columns `t, t + WVD_TPB, ...`
    ASCENDING, adding `sample_weight[j]` where `adj[i][j]` is set, and the
    block closes on `pinned_block_sum`. Both the stride and the fold width
    are `WVD_TPB`; see DEVIATION 28 for why that number is NUMERIC and comes
    from the matrix.

    Every thread reaches `pinned_block_sum`, including threads whose strided
    subset was empty -- that is the primitive's stated contract and a lane
    that skips it deadlocks the block on the fold's barriers.
    """
    var n_cols = Int(n_cols_in)
    var row = Int(block_idx.x)
    if row >= Int(n_rows_in):
        return
    var tid = Int(thread_idx.x)

    var acc = Float32(0.0)
    var j = tid
    while j < n_cols:
        if adj.unsafe_load(row * n_cols + j) != UInt8(0):
            acc = ftz(acc + ftz(sample_weight.unsafe_load(j)))
        j += WVD_TPB

    var total = pinned_block_sum[WVD_TPB](acc)
    if tid == 0:
        wght_sum.unsafe_store(row, total)


def weighted_vertex_deg_csr_kernel(
    wght_sum: MutPointer[Float32, MutAnyOrigin],
    adj_ia: MutPointer[Int32, MutAnyOrigin],
    adj_ja: MutPointer[Int32, MutAnyOrigin],
    sample_weight: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
):
    """`accumulateWeights` (`algo.cuh:62-91`), the ball-cover half.

    Theirs is one WARP per row and `cub::WarpReduce::Sum`; ours is one BLOCK
    per row and `pinned_block_sum`, which is DEVIATION 28's replacement and
    is there so a 64-lane wavefront does not fold a different multiset than a
    32-lane warp.

    THEIR `if (weight_sum > 0)` GUARD AT `:90` IS NOT COPIED, and that is
    deliberate rather than an omission. Theirs leaves `weight_sums[idx]`
    UNWRITTEN when a row's weight sum is not positive, so the value the core
    test then reads is whatever the workspace held. It works for them because
    a non-positive sum cannot make a point core at `min_pts >= 1` either way;
    it does not work here, because this port's buffer is not zeroed by a
    preceding pass and reading uninitialised device memory is not a
    predicate. Ours writes every row unconditionally. scikit-learn documents
    negative weights as meaningful ("a sample with a negative weight may
    inhibit its eps-neighbor from being core", `_dbscan.py:414-415`), so an
    unwritten negative sum is a wrong answer and not only an untidy one.
    """
    var row = Int(block_idx.x)
    if row >= Int(n_rows_in):
        return
    var tid = Int(thread_idx.x)
    var start = Int(adj_ia.unsafe_load(row))
    var stop = Int(adj_ia.unsafe_load(row + 1))

    var acc = Float32(0.0)
    var p = start + tid
    while p < stop:
        var col = Int(adj_ja.unsafe_load(p))
        acc = ftz(acc + ftz(sample_weight.unsafe_load(col)))
        p += WVD_TPB

    var total = pinned_block_sum[WVD_TPB](acc)
    if tid == 0:
        wght_sum.unsafe_store(row, total)


def weighted_vertex_deg_dense(
    ctx: DeviceContext,
    mut wght_sum: DeviceBuffer[DType.float32],
    mut adj: DeviceBuffer[DType.uint8],
    mut sample_weight: DeviceBuffer[DType.float32],
    n_points: Int,
    n_rows: Int,
) raises:
    """`launcher`'s `coalescedReduction` call, `algo.cuh:243-254`."""
    ctx.enqueue_function[weighted_vertex_deg_dense_kernel](
        wght_sum.unsafe_ptr(),
        adj.unsafe_ptr(),
        sample_weight.unsafe_ptr(),
        Int32(n_rows),
        Int32(n_points),
        grid_dim=(n_points, 1, 1),
        block_dim=(WVD_TPB, 1, 1),
    )


def weighted_vertex_deg_csr(
    ctx: DeviceContext,
    mut wght_sum: DeviceBuffer[DType.float32],
    mut adj_ia: DeviceBuffer[DType.int32],
    mut adj_ja: DeviceBuffer[DType.int32],
    mut sample_weight: DeviceBuffer[DType.float32],
    n_points: Int,
) raises:
    """`launcher`'s `accumulateWeights` launch, `algo.cuh:236-238`.

    Theirs is `<<<ceildiv(n, 4), 128>>>` because four warps of a 128-thread
    block each own a row; ours is one block per row, which is the same
    coverage at this file's block shape.
    """
    ctx.enqueue_function[weighted_vertex_deg_csr_kernel](
        wght_sum.unsafe_ptr(),
        adj_ia.unsafe_ptr(),
        adj_ja.unsafe_ptr(),
        sample_weight.unsafe_ptr(),
        Int32(n_points),
        grid_dim=(n_points, 1, 1),
        block_dim=(WVD_TPB, 1, 1),
    )


def eps_neighborhood_kernel(
    adj: MutPointer[UInt8, MutAnyOrigin],
    vd: MutPointer[Int32, MutAnyOrigin],
    dist: MutPointer[Float32, MutAnyOrigin],
    n_cols_in: Int32,
    eps_sq_in: Float32,
):
    """THE REFERENCE ORACLE. Not the shipped path any more.

    This thresholds an ALREADY MATERIALIZED `m x n` distance tile, one block
    per row. It was the shipped path until the fused kernel landed, and it is
    kept for exactly one reason: `check_fused_eps_agrees_with_materialized`
    diffs the fused kernel against it CELL BY CELL, which is the only check
    that can catch a wrong `acc[i][j] -> (row, col)` mapping inside the tile.
    A materialized reference beside a fused kernel is worth keeping; a second
    unfused implementation of a step that is already unfused is not, which is
    why `core/gemm.mojo`'s standalone contraction is gone and this is not.

    It has no upstream counterpart: cuML never materializes this matrix.
    Nothing in `dbscan/gbdt/` calls it and nothing should.
    """
    var n_cols = Int(n_cols_in)
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    var count = Int32(0)
    var j = tid
    while j < n_cols:
        var inside = dist.unsafe_load(row * n_cols + j) <= eps_sq_in
        adj.unsafe_store(row * n_cols + j, UInt8(1) if inside else UInt8(0))
        if inside:
            count += 1
        j += VD_TPB

    var s0 = block_sum[block_size=VD_TPB](count)
    if tid == 0:
        vd.unsafe_store(row, s0)
