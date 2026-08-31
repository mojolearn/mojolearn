# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`raft/linalg/detail/axpy.cuh::axpy<T, DevicePointerMode = true>`.

Theirs is ONE LINE: `cublasaxpy(cublas_h, n, alpha, x, incx, y, incy)` with
the handle in `CUBLAS_POINTER_MODE_DEVICE` (`axpy.cuh:23-32`), because
`cdFit` passes `alpha` as a DEVICE pointer both times it calls it -- the
current coefficient (`cd.cuh:201`, `coef_loc`) and the negated new one
(`cd.cuh:227`, `&convStateLoc->coef`) -- so that the per-coordinate loop
never reads a float back to the host. cuBLAS is CLOSED and MAX has no
`axpy`; the mirror is the elementwise kernel below, one thread per row,
which reads `alpha` from device memory exactly as their pointer mode does.
The host/device split is part of the algorithm (PORTING_RULES 2) and is
kept: nothing in `cdFit`'s inner loop syncs.

`y[i] = alpha * x[i] + y[i]`: under IDENTICAL one rounding through
`identical_mul_add` and the store through `ftz` (the residual is the seam
every later dot reads); under FAST the naive chain, which is what a cuBLAS
axpy computes up to the contraction the vendor picks. Order-free: each
thread owns one cell, no reduction, so the launch shape (the block size, a
1-D or 2-D grid) is SCHEDULING and the launch-invariance gate varies it.

`incx = incy = 1` only; a stride is refused by name at the caller.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, grid_dim, thread_idx

from original.numerics import ftz, identical_mul_add

#: SCHEDULING. One thread, one row. The gates run 256 and 64.
comptime AXPY_TPB = 256


def axpy_device_alpha_kernel(
    y: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    x_off_in: Int32,
    alpha_arr: MutPointer[Float32, MutAnyOrigin],
    alpha_idx_in: Int32,
    n_in: Int32,
):
    """`y += alpha_arr[alpha_idx] * x[x_off + i]`, `i` linearized over a
    grid that may be 1-D or 2-D (`block_idx.y * grid_dim.x + block_idx.x`),
    so the same cell is computed by the same expression whatever the
    geometry."""
    var n = Int(n_in)
    var blk = Int(block_idx.y) * Int(grid_dim.x) + Int(block_idx.x)
    var i = blk * Int(block_dim.x) + Int(thread_idx.x)
    if i < n:
        # Row 10: every operand flushed on load, the result flushed on
        # store. Bit-inert on an FTZ backend; it aligns a denormal-honoring
        # one to it (the labels are a raw input and may hold denormals).
        var alpha = ftz(alpha_arr.unsafe_load(Int(alpha_idx_in)))
        var xi = ftz(x.unsafe_load(Int(x_off_in) + i))
        var yi = ftz(y.unsafe_load(i))
        y.unsafe_store(i, ftz(identical_mul_add(alpha, xi, yi)))


def axpy_device_alpha(
    ctx: DeviceContext,
    mut y: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    x_off: Int,
    mut alpha_arr: DeviceBuffer[DType.float32],
    alpha_idx: Int,
    n: Int,
    tpb: Int = AXPY_TPB,
    two_d_grid: Bool = False,
) raises:
    """`raft::linalg::axpy<T, true>(handle, n, alpha_dev, x + x_off, 1, y,
    1, stream)`. `tpb` and `two_d_grid` are SCHEDULING knobs for the gates;
    production passes the defaults."""
    var blocks = (n + tpb - 1) // tpb
    if blocks < 1:
        blocks = 1
    var gx = blocks
    var gy = 1
    if two_d_grid:
        gx = 8
        if gx > blocks:
            gx = blocks
        gy = (blocks + gx - 1) // gx
    ctx.enqueue_function[axpy_device_alpha_kernel](
        y.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(x_off),
        alpha_arr.unsafe_ptr(),
        Int32(alpha_idx),
        Int32(n),
        grid_dim=(gx, gy, 1),
        block_dim=(tpb, 1, 1),
    )
