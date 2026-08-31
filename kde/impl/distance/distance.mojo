# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`pairwise_distance` for the metrics KDE reaches: the per-metric dispatch.

PORT OF cuVS `cpp/src/distance/detail/distance.cuh::distance_impl` at cuVS
`94c2819`, the `L2SqrtUnexpanded`, `L2Expanded`, `L1` and `Linf` tags only
(`distance-inl.cuh:286-302` is the `switch` that reaches them). Partial. Do
not improve.

THE EXPANDED ARM CALLS THE NEIGHBORS LANE'S TILE
------------------------------------------------
Their `L2Expanded` (`distance.cuh:461-520`) is `rowNorm` on both operands
followed by `l2_exp_distance_op` over a GEMM-shaped tile: `||x||^2 +
||y||^2 - 2 x.y`, clamped at zero (`l2_exp.cuh`). This tree already has
that arithmetic, pinned, in `neighbors/checks/pinned_distance_tile.mojo`
(DEVIATION 505, IDENTITY_PATHS row 24) and the norms in
`core/row_norms.mojo` (row 19): under IDENTICAL those two are CALLED, not
re-spelled, with `is_sqrt = 0`. Under FAST the arm is the vendor spelling
the k-NN tile uses -- `core/gemm.mojo::gemm_nt` (MAX matmul) plus
`core/expand_distances.mojo` -- which is their cuBLAS/CUTLASS design and
not reproducible across vendors, exactly as `knn_brute_force.mojo` records.

The unexpanded arms never touch a norm and go to
`kde/impl/distance/distance_ops.mojo::pairwise_unexpanded_kernel` in
both modes (its helpers compile away under FAST).

`workspace` / `worksize` / `fin_op` / `is_row_major` of their signature:
row-major only (cuML hands KDE C-contiguous arrays, `kernel_density.py:
258-263`), no `fin_op` (theirs is `identity_op` on this path), and the
norm workspace is allocated here.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.expand_distances import expand_distances_kernel
from core.gemm import gemm_nt
from core.row_norms import NORM_TPB, row_norm_kernel
from kde.impl.distance.distance_ops import (
    DIST_L1,
    DIST_L2_EXPANDED,
    DIST_L2_SQRT_UNEXPANDED,
    DIST_LINF,
    PAIRWISE_ELEM_TPB,
    pairwise_unexpanded_kernel,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from neighbors.checks.pinned_distance_tile import (
    pinned_distance_tile_kernel,
)


def pairwise_distance(
    ctx: DeviceContext,
    mut dist: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut y: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    metric: Int,
    elem_tpb: Int = PAIRWISE_ELEM_TPB,
) raises:
    """`dist[m x n] = metric(x[m x k], y[n x k])`, row-major.

    `elem_tpb` is the one-thread-per-cell block width: SCHEDULING, exposed
    so `kde/checks/kde_check.mojo` can prove the bytes do not move with
    it. `metric` is one of the four `DIST_*` values; anything else RAISES
    by value (the string-to-value table and its refusals by name are in
    `kernel_density.mojo::metric_from_name`, upstream of this).
    """
    if m <= 0 or n <= 0 or k <= 0:
        raise Error(
            "kde pairwise_distance: m, n, k must be positive, got "
            + String(m) + ", " + String(n) + ", " + String(k)
        )
    if elem_tpb <= 0:
        raise Error("kde pairwise_distance: elem_tpb must be positive")
    var cells = m * n
    var grid = (cells + elem_tpb - 1) // elem_tpb

    if (
        metric == DIST_L2_SQRT_UNEXPANDED
        or metric == DIST_L1
        or metric == DIST_LINF
    ):
        ctx.enqueue_function[pairwise_unexpanded_kernel](
            dist.unsafe_ptr(),
            x.unsafe_ptr(),
            y.unsafe_ptr(),
            Int32(m),
            Int32(n),
            Int32(k),
            Int32(metric),
            grid_dim=(grid, 1, 1),
            block_dim=(elem_tpb, 1, 1),
        )
        return

    if metric != DIST_L2_EXPANDED:
        raise Error(
            "kde pairwise_distance: metric value "
            + String(metric)
            + " is not one of the four ported DistanceType values"
        )

    # distance.cuh:497-511: rowNorm<L2Norm> on x (m rows) and y (n rows),
    # squared (identity_op), the `x != y` row-major branch.
    var x_norm = ctx.enqueue_create_buffer[DType.float32](m)
    var y_norm = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_function[row_norm_kernel](
        x_norm.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(k),
        Int32(0),
        grid_dim=(m, 1, 1),
        block_dim=(NORM_TPB, 1, 1),
    )
    ctx.enqueue_function[row_norm_kernel](
        y_norm.unsafe_ptr(),
        y.unsafe_ptr(),
        Int32(k),
        Int32(0),
        grid_dim=(n, 1, 1),
        block_dim=(NORM_TPB, 1, 1),
    )
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        # IDENTITY_PATHS row 24: the product and the epilogue as ONE kernel
        # with the feature axis in one thread; the k-NN lane's tile, called.
        ctx.enqueue_function[pinned_distance_tile_kernel](
            dist.unsafe_ptr(),
            x.unsafe_ptr(),
            y.unsafe_ptr(),
            x_norm.unsafe_ptr(),
            y_norm.unsafe_ptr(),
            Int32(m),
            Int32(n),
            Int32(k),
            Int32(0),
            grid_dim=(grid, 1, 1),
            block_dim=(elem_tpb, 1, 1),
        )
    else:
        # Their GEMM-shaped arm: the vendor product, then the epilogue.
        gemm_nt(ctx, dist, x, y, m, n, k)
        ctx.enqueue_function[expand_distances_kernel](
            dist.unsafe_ptr(),
            x_norm.unsafe_ptr(),
            y_norm.unsafe_ptr(),
            Int32(m),
            Int32(n),
            Int32(0),
            grid_dim=(grid, 1, 1),
            block_dim=(elem_tpb, 1, 1),
        )
    ctx.synchronize()
    _ = x_norm^
    _ = y_norm^
