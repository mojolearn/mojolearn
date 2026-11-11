# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Device work behind the linear-model bindings (lane p3-linear).

`ols_predict_wide_device`: `ols_predict_host` (glm/estimator.mojo) with the
public float64 prediction formed on the device. The same upload, `gemv_n`
and intercept epilogue produce the same float32 words; an exact widening
kernel (`checks/soft_f64.mojo` `sf64_from_f32`, binary64 bit patterns, since
the Apple GPU has no float64) then writes them, and one download lands the
binary64 words straight in the caller's output. No host loop over rows.
"""
from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from core.gemm import gemv_n
from checks.soft_f64 import sf64_from_f32
from glm.estimator import _add_scalar_kernel


def ols_widen_kernel(
    src: MutPointer[Float32, MutAnyOrigin], dst: MutPointer[UInt64, MutAnyOrigin], n_in: Int32
):
    """dst[i] = the exact binary64 widening of src[i] as bit patterns, one
    thread per element."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        dst.unsafe_store(i, sf64_from_f32(src.unsafe_load(i)))


def ols_predict_wide_device(
    ctx: DeviceContext,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    coef_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float64, MutUntrackedOrigin],
    n_rows: Int,
    n_features: Int,
    intercept: Float32,
) raises:
    """`out[n_rows]` (float64) = X . coef + intercept, the float32 words of
    `ols_predict_host` widened exactly on the device."""
    var x = ctx.enqueue_create_buffer[DType.float32](max(n_rows * n_features, 1))
    var coef = ctx.enqueue_create_buffer[DType.float32](max(n_features, 1))
    var out = ctx.enqueue_create_buffer[DType.float32](max(n_rows, 1))
    if n_rows * n_features > 0:
        ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)
    if n_features > 0:
        ctx.enqueue_copy(dst_buf=coef, src_ptr=coef_ptr)
    ctx.synchronize()
    gemv_n(ctx, out, x, coef, n_rows, n_features)
    # The same host compare of a caller-supplied constant as ols_predict_host
    # (DEVIATION 527): `x + 0.0` is skipped exactly as there.
    if intercept != Float32(0.0):
        ctx.enqueue_function[_add_scalar_kernel](
            out.unsafe_ptr(), Int32(n_rows), intercept,
            grid_dim=((n_rows + 255) // 256, 1, 1), block_dim=(256, 1, 1),
        )
    var wide = ctx.enqueue_create_buffer[DType.uint64](max(n_rows, 1))
    if n_rows > 0:
        ctx.enqueue_function[ols_widen_kernel](
            out.unsafe_ptr(), wide.unsafe_ptr(), Int32(n_rows),
            grid_dim=((n_rows + 255) // 256, 1, 1), block_dim=(256, 1, 1),
        )
        ctx.enqueue_copy(dst_ptr=out_ptr.bitcast[UInt64](), src_buf=wide)
    ctx.synchronize()
    _ = x^
    _ = coef^
    _ = out^
    _ = wide^
