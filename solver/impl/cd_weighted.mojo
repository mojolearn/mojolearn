# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Weighted coordinate descent for Lasso / ElasticNet (`sample_weight`).

cuML's `cdFit` takes `sample_weight` through the weighted arm of
`preProcessData` and a square-root scaling of the input and the labels
(`cd.cuh:136-163`, undone at `:240-251`). This file is that construction,
with scikit-learn's weight convention (lane sk-kernels, 2026-10-15):

    1. w' = w * (n / sum(w))           scikit-learn's ElasticNet.fit rescales
                                       the weights to sum to n_samples
    2. fit_intercept: mu_x[j] = sum_i w'_i x_ij / n, mu_y = sum_i w'_i y_i / n
                      (the weighted means), x -= mu_x, y -= mu_y
    3. x_ij *= sqrt(w'_i), y_i *= sqrt(w'_i)
    4. the UNWEIGHTED `cd_fit_traced` with fit_intercept = False

Step 4 minimizes (1/(2n)) sum_i w'_i r_i^2 + alpha l1_ratio ||coef||_1 +
(alpha/2)(1 - l1_ratio) ||coef||^2, scikit-learn's weighted objective; the
intercept is mu_y - mu_x . coef. A weight of zero gives an all-zero row,
which contributes nothing. The unweighted entry never reaches this file, so
no unweighted bit moves.

Every row-length reduction is `profile_dot_into` (the contracted fold the
solver section already uses under IDENTICAL), in both numeric modes: the
weighted means are the profile dot against w' where the unweighted means
dot against ones. Every elementwise step is one `identical_*` seam with
`ftz`, so NVIDIA, AMD and Apple produce the same words, and the host column
(`solver/host/cd_oracle.mojo::cd_oracle_fit_weighted`) walks the same steps.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx

from checks.numerics import (
    ftz,
    identical_div,
    identical_mul,
    identical_mul_add,
    identical_sqrt,
)
from core.identity_trace import IdentityTrace
from glm.impl.preprocess import subtract_one_kernel
from solver.checks.profile_dot import profile_dot_into, profile_dot_workspace_floats
from solver.impl.cd import CdLaunch, cd_fit_traced
from solver.impl.stats.mean import mean_center, scale_n_kernel

comptime CD_SW_TPB = 256


def cd_sw_ratio_kernel(
    ratio: MutPointer[Float32, MutAnyOrigin],
    wsum: MutPointer[Float32, MutAnyOrigin],
    n_f: Float32,
):
    """`ratio[0] = n / sum(w)`, one thread (scikit-learn's rescale factor)."""
    ratio.unsafe_store(0, ftz(identical_div(n_f, ftz(wsum.unsafe_load(0)))))


def cd_sw_rescale_kernel(
    w: MutPointer[Float32, MutAnyOrigin],
    ratio: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`w[i] = w[i] * ratio[0]`: the weights rescaled to sum to n."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var r = ftz(ratio.unsafe_load(0))
        w.unsafe_store(i, ftz(identical_mul(ftz(w.unsafe_load(i)), r)))


def cd_sw_sqrt_scale_kernel(
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
):
    """Column-major `x` (`n_rows x n_cols`) and `y` scaled row-wise by
    `sqrt(w[i])`. Cell `idx < n_rows * n_cols` is `x[idx]`; the last
    `n_rows` cells are the labels."""
    var n = Int(n_rows_in)
    var d = Int(n_cols_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= n * (d + 1):
        return
    var i = idx % n
    var s = ftz(identical_sqrt(ftz(w.unsafe_load(i))))
    if idx < n * d:
        x.unsafe_store(idx, ftz(identical_mul(ftz(x.unsafe_load(idx)), s)))
    else:
        y.unsafe_store(i, ftz(identical_mul(ftz(y.unsafe_load(i)), s)))


def cd_fit_weighted(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    mut labels: DeviceBuffer[DType.float32],
    mut coef: DeviceBuffer[DType.float32],
    mut sw: DeviceBuffer[DType.float32],
    fit_intercept: Bool,
    epochs: Int,
    loss: Int,
    alpha: Float32,
    l1_ratio: Float32,
    tol: Float32,
    mut trace: IdentityTrace,
    prefix: String,
    launch: CdLaunch,
) raises -> Tuple[Int, Float32]:
    """Weighted `cdFit` (module docstring). `x` is column-major and, with
    `labels` and `sw` (`n_rows` non-negative weights, not all zero; the
    Python layer checks them), MUTATED IN PLACE: `x` and `labels` come back
    centered and scaled, `sw` rescaled. `coef` is the starting point (the
    caller zeroes it). Returns `(n_iter, intercept)`."""
    if n_cols <= 0:
        raise Error(
            "Parameter n_cols: number of columns cannot be less than one"
        )
    if n_rows <= 1:
        raise Error("Parameter n_rows: number of rows cannot be less than two")
    var ws = ctx.enqueue_create_buffer[DType.float32](
        profile_dot_workspace_floats(n_rows)
    )
    var ws_cols = ctx.enqueue_create_buffer[DType.float32](
        profile_dot_workspace_floats(n_cols)
    )
    var ones = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var wsum = ctx.enqueue_create_buffer[DType.float32](1)
    var ratio = ctx.enqueue_create_buffer[DType.float32](1)
    var mu_input = ctx.enqueue_create_buffer[DType.float32](n_cols)
    var mu_labels = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.enqueue_memset(ones, Float32(1.0))
    ctx.enqueue_memset(ws, Float32(0.0))
    ctx.enqueue_memset(ws_cols, Float32(0.0))
    # 1. w' = w * (n / sum(w))
    profile_dot_into(ctx, wsum, sw, ones, ws, n_rows, launch.dot_plan)
    ctx.enqueue_function[cd_sw_ratio_kernel](
        ratio.unsafe_ptr(), wsum.unsafe_ptr(), Float32(n_rows),
        grid_dim=(1, 1, 1), block_dim=(1, 1, 1),
    )
    ctx.enqueue_function[cd_sw_rescale_kernel](
        sw.unsafe_ptr(), ratio.unsafe_ptr(), Int32(n_rows),
        grid_dim=((n_rows + CD_SW_TPB - 1) // CD_SW_TPB, 1, 1),
        block_dim=(CD_SW_TPB, 1, 1),
    )
    # 2. the weighted means (the profile dot against w', then * (1/n), the
    # unweighted column_mean's IDENTICAL arm with w' for ones) and centering
    if fit_intercept:
        var inv_n = Float32(1.0) / Float32(n_rows)
        for j in range(n_cols):
            var a = x.create_sub_buffer[DType.float32](j * n_rows, n_rows)
            var c = mu_input.create_sub_buffer[DType.float32](j, 1)
            profile_dot_into(ctx, c, a, sw, ws, n_rows, launch.dot_plan)
            _ = a^
            _ = c^
        ctx.enqueue_function[scale_n_kernel](
            mu_input.unsafe_ptr(), Int32(n_cols), inv_n,
            grid_dim=((n_cols + CD_SW_TPB - 1) // CD_SW_TPB, 1, 1),
            block_dim=(CD_SW_TPB, 1, 1),
        )
        profile_dot_into(ctx, mu_labels, labels, sw, ws, n_rows, launch.dot_plan)
        ctx.enqueue_function[scale_n_kernel](
            mu_labels.unsafe_ptr(), Int32(1), inv_n,
            grid_dim=(1, 1, 1), block_dim=(CD_SW_TPB, 1, 1),
        )
        mean_center(ctx, x, mu_input, n_cols, n_rows)
        mean_center(ctx, labels, mu_labels, 1, n_rows)
    # 3. rows scaled by sqrt(w')
    var cells = n_rows * (n_cols + 1)
    ctx.enqueue_function[cd_sw_sqrt_scale_kernel](
        x.unsafe_ptr(), labels.unsafe_ptr(), sw.unsafe_ptr(),
        Int32(n_rows), Int32(n_cols),
        grid_dim=((cells + CD_SW_TPB - 1) // CD_SW_TPB, 1, 1),
        block_dim=(CD_SW_TPB, 1, 1),
    )
    ctx.synchronize()
    # 4. the unweighted solver on the transformed problem
    var residual_out = ctx.enqueue_create_buffer[DType.float32](1)
    var out = cd_fit_traced(
        ctx, x, n_rows, n_cols, labels, coef, False, epochs, loss, alpha,
        l1_ratio, False, tol, False, trace, prefix, launch, residual_out,
        False,
    )
    var intercept = Float32(0.0)
    if fit_intercept:
        # intercept = mu_y - mu_x . coef (postProcessData's arithmetic)
        var d_dot = ctx.enqueue_create_buffer[DType.float32](1)
        var d_icpt = ctx.enqueue_create_buffer[DType.float32](1)
        profile_dot_into(ctx, d_dot, mu_input, coef, ws_cols, n_cols, launch.dot_plan)
        ctx.enqueue_function[subtract_one_kernel](
            d_icpt.unsafe_ptr(), mu_labels.unsafe_ptr(), d_dot.unsafe_ptr(),
            grid_dim=(1, 1, 1), block_dim=(1, 1, 1),
        )
        var h = ctx.enqueue_create_host_buffer[DType.float32](1)
        ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=d_icpt)
        ctx.synchronize()
        intercept = h.unsafe_ptr().unsafe_load(0)
        _ = h^
        _ = d_dot^
        _ = d_icpt^
    ctx.synchronize()
    # [[mojo-buffer-freed-at-last-use]]: every scratch outlives the queue.
    _ = residual_out^
    _ = ws^
    _ = ws_cols^
    _ = ones^
    _ = wsum^
    _ = ratio^
    _ = mu_input^
    _ = mu_labels^
    return (out[0], intercept)
