# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The matrix product."""

from layout import TileTensor
from layout.tile_layout import row_major
from linalg.matmul import matmul
from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx

from core.gram_splitk import (
    GRAM_MAX_CELLS_PER_THREAD,
    GRAM_MAX_COLS,
    GRAM_TPB,
    gemm_tn_splitk_into,
    gram_splitk_applies,
)
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_mul_add,
)
from std.sys.compile import is_defined

from gemm.checks.gemm_identical import identical_gemm
from gemm.checks.gemm_oracle import OP_NT




def pinned_gemm_nt_kernel(
    z: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
):
    """`z[m x n] = x[m x k] ."""
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= m * n:
        return
    var i = cell // n
    var j = cell % n
    var acc = Float32(0.0)
    for p in range(k):
        acc = ftz(
            identical_mul_add(
                ftz(x.unsafe_load(i * k + p)),
                ftz(y.unsafe_load(j * k + p)),
                acc,
            )
        )
    z.unsafe_store(cell, ftz(Float32(0.0) + ftz(acc)))


def pinned_gemm_nt_gram_kernel(
    z: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
):
    """`z[m x n] = x[m x k] ."""
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= m * n:
        return
    var i = cell // n
    var j = cell % n
    var acc = Float32(0.0)
    for p in range(k):
        acc = ftz(
            identical_mul_add(
                ftz(x.unsafe_load(i * k + p)),
                ftz(x.unsafe_load(j * k + p)),
                acc,
            )
        )
    z.unsafe_store(cell, ftz(Float32(0.0) + ftz(acc)))


def pinned_gemv_n_kernel(
    z: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    k_in: Int32,
):
    """`z[m] = x[m x k] ."""
    var m = Int(m_in)
    var k = Int(k_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= m:
        return
    var acc = Float32(0.0)
    for p in range(k):
        acc = ftz(
            identical_mul_add(
                ftz(x.unsafe_load(i * k + p)), ftz(y.unsafe_load(p)), acc
            )
        )
    z.unsafe_store(i, ftz(Float32(0.0) + ftz(acc)))


comptime GEMM_IDENT_SWAP_537 = is_defined["MOJOLEARN_537_GEMM_IDENT_SWAP"]()


comptime PINNED_GEMM_TPB = 256


comptime GEMM_VECLEN = 4
comptime GEMM_KBLK = 32
comptime GEMM_ACC_ROWS_PER_TH = 4
comptime GEMM_ACC_COLS_PER_TH = 4
comptime GEMM_ACC_TH_ROWS = 16
comptime GEMM_ACC_TH_COLS = 16

comptime GEMM_THREADS = GEMM_ACC_TH_ROWS * GEMM_ACC_TH_COLS
comptime GEMM_MBLK = GEMM_ACC_ROWS_PER_TH * GEMM_ACC_TH_ROWS
comptime GEMM_NBLK = GEMM_ACC_COLS_PER_TH * GEMM_ACC_TH_COLS
comptime GEMM_SMEM_STRIDE = GEMM_KBLK + GEMM_VECLEN
comptime GEMM_SMEM_PAGE_X = GEMM_SMEM_STRIDE * GEMM_MBLK
comptime GEMM_SMEM_PAGE_Y = GEMM_SMEM_STRIDE * GEMM_NBLK


def gemm_nt(
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut y: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
) raises:
    """`z[m x n] = x[m x k] ."""
    if n == 1:
        gemv_n(ctx, z, x, y, m, k)
        return
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        comptime if GEMM_IDENT_SWAP_537:
            identical_gemm(ctx, z, x, y, m, n, k, OP_NT)
            return
        ctx.enqueue_function[pinned_gemm_nt_kernel](
            z.unsafe_ptr(),
            x.unsafe_ptr(),
            y.unsafe_ptr(),
            Int32(m),
            Int32(n),
            Int32(k),
            grid_dim=((m * n + PINNED_GEMM_TPB - 1) // PINNED_GEMM_TPB, 1, 1),
            block_dim=(PINNED_GEMM_TPB, 1, 1),
        )
        return
    var tz = TileTensor(z, row_major(m, n))
    var tx = TileTensor(x, row_major(m, k))
    var ty = TileTensor(y, row_major(n, k))
    matmul[transpose_b=True, target="gpu"](tz, tx, ty, ctx)


def gemm_nt_gram(
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    xt: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
) raises:
    """`z[m x n] = xt[m x k] ."""
    if n == 1:
        raise Error(
            "gemm_nt_gram: n == 1 is not a Gram shape this entry serves."
            " gemm_nt's gemv route takes two mut buffers and cannot be"
            " handed one buffer twice; a 1 x 1 Gram is a dot product and"
            " belongs somewhere else."
        )
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        var xtm = xt
        ctx.enqueue_function[pinned_gemm_nt_gram_kernel](
            z.unsafe_ptr(),
            xtm.unsafe_ptr(),
            Int32(m),
            Int32(n),
            Int32(k),
            grid_dim=((m * n + PINNED_GEMM_TPB - 1) // PINNED_GEMM_TPB, 1, 1),
            block_dim=(PINNED_GEMM_TPB, 1, 1),
        )
        return
    var tz = TileTensor(z, row_major(m, n))
    var tx = TileTensor(xt, row_major(m, k))
    var ty = TileTensor(xt, row_major(n, k))
    matmul[transpose_b=True, target="gpu"](tz, tx, ty, ctx)


def gemm_tn(
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut xt: DeviceBuffer[DType.float32],
    mut xt2: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
) raises:
    """`z[m x n] = x[k x m]^T ."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        if not gram_splitk_applies(m, n, k):
            raise Error(
                "gemm_tn: NUMERIC_IDENTICAL refuses the Gram shape "
                + String(m)
                + " x "
                + String(n)
                + " x "
                + String(k)
                + ". The split-K kernel (core/gram_splitk.mojo) is the only"
                + " arm with a pinned summation order, and this shape is"
                + " outside its capacity (needs m == n <= "
                + String(GRAM_MAX_COLS)
                + " and m*n <= "
                + String(GRAM_TPB * GRAM_MAX_CELLS_PER_THREAD)
                + " register cells). The other arm is linalg.matmul, whose"
                + " k-split is a per-vendor summation order, so running it"
                + " would return a NON-identical model under a mode that"
                + " promises one. IDENTITY_PATHS row 27. To close this"
                + " refusal, widen the split-K kernel's staging tile; to"
                + " work around it today, reduce the feature count or run"
                + " NUMERIC_FAST and drop the cross-vendor claim."
            )
    if gram_splitk_applies(m, n, k):
        gemm_tn_splitk_into(ctx, z, x, xt, m, k)
        return
    gemm_tn_via_transpose(ctx, z, x, xt, xt2, m, n, k)


def gemm_tn_via_transpose(
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut xt: DeviceBuffer[DType.float32],
    mut xt2: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
) raises:
    """`z[m x n] = x[k x m]^T ."""
    from core.column_stats import (
        CUDA_MAX_GRID_YZ,
        TRANSPOSE_TILE,
        transpose_kernel,
    )

    if m != n:
        raise Error(
            "gemm_tn_via_transpose: m="
            + String(m)
            + " n="
            + String(n)
            + ". This entry is the GRAM case: one operand, one width. Its"
            " transpose is built at width m, and reading that block at width"
            " n runs off it. A genuine two-operand TN needs a second"
            " transpose at width n, which this does not do."
        )

    ctx.enqueue_function[transpose_kernel](
        xt.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(k),
        Int32(m),
        grid_dim=(
            (m + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE,
            min((k + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE, CUDA_MAX_GRID_YZ),
            1,
        ),
        block_dim=(TRANSPOSE_TILE, TRANSPOSE_TILE, 1),
    )
    gemm_nt_gram(ctx, z, xt, m, n, k)
    _ = xt2



from linalg.gemv import gemv_gpu


def gemv_n(
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut y: DeviceBuffer[DType.float32],
    m: Int,
    k: Int,
) raises:
    """`z[m] = x[m x k] ."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        ctx.enqueue_function[pinned_gemv_n_kernel](
            z.unsafe_ptr(),
            x.unsafe_ptr(),
            y.unsafe_ptr(),
            Int32(m),
            Int32(k),
            grid_dim=((m + PINNED_GEMM_TPB - 1) // PINNED_GEMM_TPB, 1, 1),
            block_dim=(PINNED_GEMM_TPB, 1, 1),
        )
        return
    var tz = TileTensor(z, row_major(m, Int(1)))
    var tx = TileTensor(x, row_major(m, k))
    var ty = TileTensor(y, row_major(k, Int(1)))
    gemv_gpu[transpose_b=False](tz, tx, ty, ctx)
