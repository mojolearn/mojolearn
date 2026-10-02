# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MultiOutputRegressor(Ridge): every target in ONE ridge program (lane
apple-fast-meta, -D MOJOLEARN_MULTIOUT_RIDGE). FAST + Apple ONLY: the
estimators binding registers `ridge_fit_multi` and `ridge_predict_multi`
under MULTIOUT_RIDGE and nothing else references this file.

The reference route (python/mojolearn/_expansion_trees.py
MultiOutputRegressor) clones Ridge per target: X goes up per target, its
A^T A eigendecomposition is redone per target, and predict uploads X per
target. Here X goes up once, `svd_eig` runs once, `ridgeSolve`'s S and V
transforms run once, and each target is one U^T b and one V (S b) (the
reference solve's last two ops, glm/impl/ridge.mojo). Y arrives row-major
(n x m); its column means and centering (fit_intercept) are device
kernels, one thread per row. predict is one kernel, one thread per row,
every target's dot product and intercept, written row-major (n x m).
"""
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from core.xtdz_coalesced import xty_launch
from core.column_stats import STATS_TPB, column_mean_kernel
from core.gemm import gemv_n
from core.identity_trace import IdentityTrace
from glm.impl.linalg.detail.svd import svd_eig_traced
from glm.impl.ridge import RIDGE_SMALL_THRESH
from glm.impl.matrix.math import (
    MATRIX_ELEM_TPB,
    add_scalar_kernel,
    matrix_vector_binary_div_skip_zero_kernel,
    matrix_vector_binary_mult_kernel,
    power_kernel,
    set_small_values_zero_kernel,
)

#: the switch: FAST + Apple + the define
comptime MULTIOUT_RIDGE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_MULTIOUT_RIDGE"]()
)
comptime _ROW_TPB = 256


def _ridge_ycol_kernel(
    yc: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    mu: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    m_in: Int32,
    j_in: Int32,
    center: Int32,
):
    """yc[i] = y[i, j] - mu[j] (center) or y[i, j]: one thread per row."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    var v = y.unsafe_load(i * Int(m_in) + Int(j_in))
    if center != 0:
        v = v - mu.unsafe_load(Int(j_in))
    yc.unsafe_store(i, v)


def _ridge_predict_multi_kernel(
    out: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    coef: MutPointer[Float32, MutAnyOrigin],
    icpt: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    d_in: Int32,
    m_in: Int32,
):
    """out[i, j] = icpt[j] + x[i, :] . coef[j, :]: one thread per row, the
    row read once for every target."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    var d = Int(d_in)
    var m = Int(m_in)
    for j in range(m):
        var acc = icpt.unsafe_load(j)
        for c in range(d):
            acc = acc + x.unsafe_load(i * d + c) * coef.unsafe_load(j * d + c)
        out.unsafe_store(i * m + j, acc)


def _elem_grid(n: Int) -> Int:
    return (n + MATRIX_ELEM_TPB - 1) // MATRIX_ELEM_TPB


def ridge_fit_multi_host(
    ctx: DeviceContext,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    y_ptr: MutPointer[Float32, MutUntrackedOrigin],
    coef_ptr: MutPointer[Float32, MutUntrackedOrigin],
    ymean_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    n_cols: Int,
    n_targets: Int,
    alpha: Float32,
    center_y: Bool,
) raises:
    """coef (n_targets x n_cols, row-major) of ridge on the (centered, by
    the caller) design X for every column of Y (n_rows x n_targets,
    row-major); ymean (n_targets) the column means of Y, subtracted from
    each target before the solve when `center_y`."""
    if n_cols <= 1:
        raise Error("ridge_fit_multi: number of columns cannot be less than two")
    if n_rows <= 1:
        raise Error("ridge_fit_multi: number of rows cannot be less than two")
    if n_targets <= 0:
        raise Error("ridge_fit_multi: no targets")
    if alpha < Float32(0.0):
        raise Error("ridge_fit_multi: alpha must be non-negative")
    var x = ctx.enqueue_create_buffer[DType.float32](n_rows * n_cols)
    var y = ctx.enqueue_create_buffer[DType.float32](n_rows * n_targets)
    var mu = ctx.enqueue_create_buffer[DType.float32](n_targets)
    ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)
    ctx.enqueue_copy(dst_buf=y, src_ptr=y_ptr)
    ctx.synchronize()
    ctx.enqueue_function[column_mean_kernel](
        mu.unsafe_ptr(), y.unsafe_ptr(), Int32(n_rows), Int32(n_targets),
        grid_dim=(n_targets, 1, 1), block_dim=(STATS_TPB, 1, 1),
    )
    # svd_eig once, ridgeSolve's S and V transforms once
    var s = ctx.enqueue_create_buffer[DType.float32](n_cols)
    var v = ctx.enqueue_create_buffer[DType.float32](n_cols * n_cols)
    var u = ctx.enqueue_create_buffer[DType.float32](n_rows * n_cols)
    var s_nnz = ctx.enqueue_create_buffer[DType.float32](n_cols)
    ctx.synchronize()
    var trace = IdentityTrace.disabled()
    svd_eig_traced(ctx, x, n_rows, n_cols, s, u, v, True, trace, "ridge_multi.svd")
    ctx.enqueue_function[set_small_values_zero_kernel](
        s.unsafe_ptr(), Int32(n_cols), RIDGE_SMALL_THRESH,
        grid_dim=(_elem_grid(n_cols), 1, 1), block_dim=(MATRIX_ELEM_TPB, 1, 1),
    )
    ctx.enqueue_function[power_kernel](
        s_nnz.unsafe_ptr(), s.unsafe_ptr(), Int32(n_cols), Float32(1.0),
        grid_dim=(_elem_grid(n_cols), 1, 1), block_dim=(MATRIX_ELEM_TPB, 1, 1),
    )
    ctx.enqueue_function[add_scalar_kernel](
        s_nnz.unsafe_ptr(), Int32(n_cols), alpha,
        grid_dim=(_elem_grid(n_cols), 1, 1), block_dim=(MATRIX_ELEM_TPB, 1, 1),
    )
    ctx.enqueue_function[matrix_vector_binary_div_skip_zero_kernel](
        s.unsafe_ptr(), s_nnz.unsafe_ptr(), Int32(1), Int32(n_cols), Int32(1),
        grid_dim=(_elem_grid(n_cols), 1, 1), block_dim=(MATRIX_ELEM_TPB, 1, 1),
    )
    ctx.enqueue_function[matrix_vector_binary_mult_kernel](
        v.unsafe_ptr(), s.unsafe_ptr(), Int32(n_cols), Int32(n_cols),
        grid_dim=(_elem_grid(n_cols * n_cols), 1, 1), block_dim=(MATRIX_ELEM_TPB, 1, 1),
    )
    ctx.synchronize()
    # per target: its (centered) column, U^T b, w = V (S_over U^T b)
    var yc = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var utb = ctx.enqueue_create_buffer[DType.float32](n_cols)
    var w = ctx.enqueue_create_buffer[DType.float32](n_cols)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](n_cols)
    ctx.synchronize()
    for j in range(n_targets):
        ctx.enqueue_function[_ridge_ycol_kernel](
            yc.unsafe_ptr(), y.unsafe_ptr(), mu.unsafe_ptr(), Int32(n_rows), Int32(n_targets), Int32(j),
            Int32(1) if center_y else Int32(0),
            grid_dim=((n_rows + _ROW_TPB - 1) // _ROW_TPB, 1, 1), block_dim=(_ROW_TPB, 1, 1),
        )
        xty_launch(ctx, utb, u, yc, n_rows, n_cols)
        gemv_n(ctx, w, v, utb, n_cols, n_cols)
        ctx.enqueue_copy(dst_ptr=hw.unsafe_ptr(), src_buf=w)
        ctx.synchronize()
        for c in range(n_cols):
            coef_ptr.unsafe_store(j * n_cols + c, hw.unsafe_ptr().unsafe_load(c))
    var hmu = ctx.enqueue_create_host_buffer[DType.float32](n_targets)
    ctx.enqueue_copy(dst_ptr=hmu.unsafe_ptr(), src_buf=mu)
    ctx.synchronize()
    for j in range(n_targets):
        ymean_ptr.unsafe_store(j, hmu.unsafe_ptr().unsafe_load(j))
    _ = hw^
    _ = hmu^
    _ = yc^
    _ = utb^
    _ = w^
    _ = s^
    _ = v^
    _ = u^
    _ = s_nnz^
    _ = mu^
    _ = y^
    _ = x^


def ridge_predict_multi_host(
    ctx: DeviceContext,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    coef_ptr: MutPointer[Float32, MutUntrackedOrigin],
    icpt_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    n_cols: Int,
    n_targets: Int,
) raises:
    """out (n_rows x n_targets, row-major) = X coef^T + icpt, X up once."""
    var x = ctx.enqueue_create_buffer[DType.float32](n_rows * n_cols)
    var coef = ctx.enqueue_create_buffer[DType.float32](n_targets * n_cols)
    var icpt = ctx.enqueue_create_buffer[DType.float32](n_targets)
    var out = ctx.enqueue_create_buffer[DType.float32](n_rows * n_targets)
    ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)
    ctx.enqueue_copy(dst_buf=coef, src_ptr=coef_ptr)
    ctx.enqueue_copy(dst_buf=icpt, src_ptr=icpt_ptr)
    ctx.synchronize()
    ctx.enqueue_function[_ridge_predict_multi_kernel](
        out.unsafe_ptr(), x.unsafe_ptr(), coef.unsafe_ptr(), icpt.unsafe_ptr(),
        Int32(n_rows), Int32(n_cols), Int32(n_targets),
        grid_dim=((n_rows + _ROW_TPB - 1) // _ROW_TPB, 1, 1), block_dim=(_ROW_TPB, 1, 1),
    )
    var hout = ctx.enqueue_create_host_buffer[DType.float32](n_rows * n_targets)
    ctx.enqueue_copy(dst_ptr=hout.unsafe_ptr(), src_buf=out)
    ctx.synchronize()
    for i in range(n_rows * n_targets):
        out_ptr.unsafe_store(i, hout.unsafe_ptr().unsafe_load(i))
    _ = hout^
    _ = out^
    _ = icpt^
    _ = coef^
    _ = x^
