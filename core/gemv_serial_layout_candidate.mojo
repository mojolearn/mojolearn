# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Unqualified, opt-in GEMV layout experiment; no production dispatch uses it.

The existing pinned GEMV gives adjacent threads adjacent OUTPUT rows, so at
each serial K step their matrix loads are K floats apart. Here X is copied
bit for bit to [K, M], making those loads contiguous. Every output retains
the exact ascending FMA/FTZ chain and final +0 operation in
`core.gemm.pinned_gemv_n_kernel`; no partial sums cross threads.

This is a performance hypothesis, not a measured optimization. The full
operation adds a transpose and M*K floats of scratch. Measure that cost,
not just the product kernel. Reuse of an unchanged prepared matrix may be
priced separately. In particular, staging does not remove the serial FMA
dependency and may lose on small or single-use matrices.

Both launchers enqueue asynchronously. The caller owns all buffers and
must retain them until synchronization. X, XT, Y and Z must not overlap;
X and XT require M*K float32 elements, Y requires K, and Z requires M.
Prepared XT must satisfy XT[p*M+i] == X[i*K+p] bit for bit.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx

from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_mul_add,
)
from core.column_stats import CUDA_MAX_GRID_YZ, TRANSPOSE_TILE, transpose_kernel


comptime SERIAL_LAYOUT_TPB = 64


def serial_layout_gemv_kernel(
    z: MutPointer[Float32, MutAnyOrigin],
    xt: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    k_in: Int32,
):
    var m = Int(m_in)
    var k = Int(k_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= m:
        return
    var acc = Float32(0.0)
    for p in range(k):
        acc = ftz(
            identical_mul_add(
                ftz(xt.unsafe_load(p * m + i)), ftz(y.unsafe_load(p)), acc
            )
        )
    z.unsafe_store(i, ftz(Float32(0.0) + ftz(acc)))


def serial_layout_gemv_prepared(
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    mut xt: DeviceBuffer[DType.float32],
    mut y: DeviceBuffer[DType.float32],
    m: Int,
    k: Int,
) raises:
    """Product only, with caller-prepared XT; not an end-to-end GEMV timing."""
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("serial-layout GEMV candidate requires IDENTICAL mode")
    if m <= 0 or k <= 0 or m > 2147483647 or k > 2147483647:
        raise Error("serial-layout GEMV candidate requires positive Int32 dimensions")
    ctx.enqueue_function[serial_layout_gemv_kernel](
        z.unsafe_ptr(),
        xt.unsafe_ptr(),
        y.unsafe_ptr(),
        Int32(m),
        Int32(k),
        grid_dim=((m + SERIAL_LAYOUT_TPB - 1) // SERIAL_LAYOUT_TPB, 1, 1),
        block_dim=(SERIAL_LAYOUT_TPB, 1, 1),
    )


def serial_layout_gemv_into(
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut y: DeviceBuffer[DType.float32],
    mut xt: DeviceBuffer[DType.float32],
    m: Int,
    k: Int,
) raises:
    """Full transpose + product experiment; XT is caller-owned scratch."""
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("serial-layout GEMV candidate requires IDENTICAL mode")
    if m <= 0 or k <= 0 or m > 2147483647 or k > 2147483647:
        raise Error("serial-layout GEMV candidate requires positive Int32 dimensions")
    ctx.enqueue_function[transpose_kernel](
        xt.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(m),
        Int32(k),
        grid_dim=(
            (k + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE,
            min((m + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE, CUDA_MAX_GRID_YZ),
            1,
        ),
        block_dim=(TRANSPOSE_TILE, TRANSPOSE_TILE, 1),
    )
    serial_layout_gemv_prepared(ctx, z, xt, y, m, k)
