# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's PORTED_MAP.tsv and in this file's own docstring. See NOTICE.
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

so that is what `vertex_deg_run` calls, and the kernel itself lives beside
its own upstream in `dbscan/gbdt/neighbors/epsilon_neighborhood.mojo`.

**`eps` is squared once on the host and never per pair** (`algo.cuh:225`).
DBSCAN's radius is a distance and their accumulator is a squared distance;
squaring the threshold instead of rooting a million distances is theirs.

A point is its own neighbor here, exactly as in scikit-learn: the distance to
itself is zero, which is `<= eps`, so `vd` includes the point and `min_pts`
counts it. Getting that wrong shifts every core-point decision by one and
produces a plausible clustering that disagrees with sklearn everywhere.

NOT PORTED FROM THIS FILE, and named so it is not forgotten:
- the `CosineExpanded` arm (`:186-223`): row-normalize in place, use
  `eps2 = 2 * eps`, run the same neighborhood kernel, then un-normalize.
- the `sample_weight` arm (`:234-257`): `coalescedReduction` over `adj` with
  `adj_ij ? sample_weight[j] : 0`, which produces a WEIGHTED degree that
  `CorePoints::compute` then thresholds instead of the integer degree.
- `Precomputed::launcher` (`algo == 2`), for a precomputed distance matrix.
- the ball-cover arm, `eps_nn`, which is a different lane.
"""

from mojo_only.kernel_matrix import (
    K_LIB_EPS_NEIGHBORHOOD,
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

from dbscan.ported.neighbors.epsilon_neighborhood import (
    eps_unexp_l2_sq_neighborhood,
)


# READ FROM THE MATRIX, not restated here. `mojo_only/kernel_matrix.mojo`
# owns every tunable in this tree; changing TARGET_COLUMN there rebuilds
# this kernel for another vendor with no edit in this file.
comptime VD_TPB = lib_block_size_for[K_LIB_EPS_NEIGHBORHOOD, TARGET_COLUMN]()


def vertex_deg_run(
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
    """`VertexDeg::run` -> `Algo::launcher`, the L2 brute-force arm.

    `n_points` is their `n` (the batch), `n_rows` their `m` (the dataset).
    """
    var eps2 = Float32(eps * eps)
    eps_unexp_l2_sq_neighborhood(
        ctx, adj, vd, x, start_vertex_id, n_points, n_rows, n_features, eps2
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
