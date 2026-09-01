# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`pairwise_distance` for the metrics KDE reaches: the per-metric dispatch.

PORT OF cuVS `cpp/src/distance/detail/distance.cuh::distance_impl` at cuVS
`94c2819`, the `L2SqrtUnexpanded`, `L2Expanded`, `L1`, `Linf`,
`CosineExpanded` (`:180-226`) and `LpUnexpanded` (`:643-667`) tags
(`distance-inl.cuh:261-311` is the `switch` that reaches them). Partial. Do
not improve.

COSINE AND MINKOWSKI, 2026-09-01. `metric='cosine'` and
`metric='minkowski'` were REFUSED BY NAME in this lane until now, and the
refusal string said they were "in cuML's pairwise_distances table but NOT
PORTED". They are ported. Both ops live with every other op in
`neighbors/impl/distance/detail/distance_ops.mojo` (cuVS's own layout: one
`distance_ops/` directory, every consumer reaching it), and this file's
dispatch is `distance_impl`'s per-metric prologue -- which norms to compute
and with or without the square root -- exactly as it already was for
`L2Expanded`. Minkowski's `p` arrives as `metric_arg`, their name for it,
which their `pairwise_distances` defaults to 2 (`pairwise_distances.pyx:
266`) and `KernelDensity.score_samples` fills from `metric_params`
(`kernel_density.py:302-313`).

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

The three older unexpanded arms (`L2SqrtUnexpanded`, `L1`, `Linf`) never
touch a norm and go to
`kde/impl/distance/distance_ops.mojo::pairwise_unexpanded_kernel` in
both modes (its helpers compile away under FAST). `LpUnexpanded` does not
touch one either (`lp_unexp.cuh:41`, `use_norms = false`) and goes to
`metric_distance_kernel` in both modes for the same reason.

`CosineExpanded` DOES touch a norm, and it is a DIFFERENT norm from the
expanded L2 arm's: the TRUE L2 norm, `rowNorm<L2Norm, true>(...,
raft::sqrt_op{})` at `distance.cuh:215-216`, where L2 wants the SQUARED
norm. Their comment at `knn_brute_force.cuh:122` is the whole rule. It is
computed by `cosine_row_norm_kernel` rather than by
`core/row_norms.mojo`'s `take_sqrt` arm, because that arm ends in the
STDLIB sqrt (`core/row_norms.mojo:102`) which is approximate on NVIDIA
(DEVIATION 258) and would make an IDENTICAL cosine agree on two columns
and differ on the third.

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
    DIST_COSINE_EXPANDED,
    DIST_L1,
    DIST_L2_EXPANDED,
    DIST_L2_SQRT_UNEXPANDED,
    DIST_LINF,
    DIST_LP_UNEXPANDED,
    PAIRWISE_ELEM_TPB,
    pairwise_unexpanded_kernel,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from neighbors.checks.pinned_distance_tile import (
    pinned_distance_tile_kernel,
)
from neighbors.impl.distance.detail.distance_ops import (
    COSINE_NORM_TPB,
    cosine_row_norm_kernel,
    metric_distance_kernel,
    validate_metric_arg,
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
    metric_arg: Float32 = Float32(2.0),
    elem_tpb: Int = PAIRWISE_ELEM_TPB,
) raises:
    """`dist[m x n] = metric(x[m x k], y[n x k])`, row-major.

    `elem_tpb` is the one-thread-per-cell block width: SCHEDULING, exposed
    so `kde/checks/kde_check.mojo` can prove the bytes do not move with
    it. `metric` is one of the six `DIST_*` values this lane reaches;
    anything else RAISES by value (the string-to-value table and its
    refusals by name are in `kernel_density.mojo::metric_from_name`,
    upstream of this).

    `metric_arg` is Minkowski's `p` (`pairwise_distances.pyx:266`'s
    `metric_arg=2` default, which `KernelDensity.score_samples:307-312`
    fills from `metric_params`); every other metric accepts and discards
    it, which is their `DataT)  // unused` at `distance.cuh:193`. It is
    validated by value here, before any launch (DEVIATION 552).

    COSINE AND MINKOWSKI, 2026-09-01. Both are rows of THEIR dense table
    (`pairwise_distances.pyx:70`, `:78`) that this lane refused by name
    until now. Neither goes through `pairwise_unexpanded_kernel`:
    Minkowski's op has no expanded form at any p, and cosine's IS
    expanded but with sqrt'd norms and a different epilogue, so both take
    `neighbors/impl/distance/detail/distance_ops.mojo::
    metric_distance_kernel`, which carries every op cuVS has in one place.
    """
    if m <= 0 or n <= 0 or k <= 0:
        raise Error(
            "kde pairwise_distance: m, n, k must be positive, got "
            + String(m) + ", " + String(n) + ", " + String(k)
        )
    if elem_tpb <= 0:
        raise Error("kde pairwise_distance: elem_tpb must be positive")
    validate_metric_arg(metric, metric_arg)
    var cells = m * n
    var grid = (cells + elem_tpb - 1) // elem_tpb

    if metric == DIST_LP_UNEXPANDED:
        # `lp_unexp.cuh:41`: `use_norms = false`. The two norm pointers are
        # never read, so `x` is passed twice rather than allocating a
        # buffer nothing reads (Mojo refuses a null pointer argument).
        ctx.enqueue_function[metric_distance_kernel](
            dist.unsafe_ptr(),
            x.unsafe_ptr(),
            y.unsafe_ptr(),
            x.unsafe_ptr(),
            x.unsafe_ptr(),
            Int32(m),
            Int32(n),
            Int32(k),
            Int32(metric),
            metric_arg,
            grid_dim=(grid, 1, 1),
            block_dim=(elem_tpb, 1, 1),
        )
        ctx.synchronize()
        return

    if metric == DIST_COSINE_EXPANDED:
        # `distance.cuh:204-225`: two `rowNorm<L2Norm, true>(..., sqrt_op{})`
        # into the workspace, then the op. Their `x == y && is_row_major`
        # fast path (`:209-211`, ONE norm over `max(m, n)`) is not taken
        # here because this lane's caller never passes the same pointer
        # twice; taking it would be a second code path with no second
        # answer.
        var xn = ctx.enqueue_create_buffer[DType.float32](m)
        var yn = ctx.enqueue_create_buffer[DType.float32](n)
        ctx.enqueue_function[cosine_row_norm_kernel](
            xn.unsafe_ptr(),
            x.unsafe_ptr(),
            Int32(k),
            grid_dim=(m, 1, 1),
            block_dim=(COSINE_NORM_TPB, 1, 1),
        )
        ctx.enqueue_function[cosine_row_norm_kernel](
            yn.unsafe_ptr(),
            y.unsafe_ptr(),
            Int32(k),
            grid_dim=(n, 1, 1),
            block_dim=(COSINE_NORM_TPB, 1, 1),
        )
        # ONE kernel for the dot product AND the epilogue in both modes.
        # KDE's caller is `pairwise_distances`, which is their DIRECT
        # distance entry (`kernel_density.py:315-317`) and not the k-NN
        # tiling path, so there is no vendor-matmul arm to inherit here:
        # `distance_impl` hands the whole cell to `pairwise_matrix` in
        # FAST and IDENTICAL alike. That is why this branch has no
        # `comptime if` where the `L2Expanded` branch below does.
        ctx.enqueue_function[metric_distance_kernel](
            dist.unsafe_ptr(),
            x.unsafe_ptr(),
            y.unsafe_ptr(),
            xn.unsafe_ptr(),
            yn.unsafe_ptr(),
            Int32(m),
            Int32(n),
            Int32(k),
            Int32(metric),
            metric_arg,
            grid_dim=(grid, 1, 1),
            block_dim=(elem_tpb, 1, 1),
        )
        ctx.synchronize()
        _ = xn^
        _ = yn^
        return

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
            + " is not one of the six ported DistanceType values"
            " (L2SqrtUnexpanded, L2Expanded, L1, Linf, CosineExpanded,"
            " LpUnexpanded)"
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
