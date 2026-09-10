# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The LINEAR and RBF kernel matrices: `GramMatrixBase::linear`,
`RBFKernel::evaluate`, `rbf_kernel_expanded`, `matrixRowNormL2`.

FOLLOWS `cuvs/cpp/src/distance/detail/kernels/kernel_matrices.cu` at cuVS
`94c2819` (the `cuvs::distance::kernels` cuML 26.08 links; RAFT's
`distance/detail/kernels/kernel_matrices.cuh` is the same code one
repository earlier). Dense, row-major, FP32. POLYNOMIAL and TANH are NOT
implemented (refused by name in `svm_parameter.mojo`; `svm/NOT_IMPLEMENTED.tsv`).

THE ROUNDING SEQUENCE (svm/README.md, identity content section 1). Theirs:

    linear:  out = x1 . x2^T                          (cuBLAS gemm)
    rbf:     out = exp(-1.0 * gain * (norm_x[i] + norm_y[j] - out * 2))
             with norm = rowNorm<L2Norm> (SQUARED; a block fold), and for
             math_t = float the `-1.0 *` promotes to DOUBLE so the exp is
             the double one, rounded to float on store. No clamp at zero.

Ours, both modes the same association, the pins under IDENTICAL:

    dot  = gemm_nt (MAX matmul) under FAST / identical_gemm v1 under IDENTICAL
    norm = per-row SERIAL ascending chain, acc = ftz(fma(x, x, acc))
    s    = ftz( ftz(norm_x + norm_y) - ftz(2 * dot) )
    e    = ftz( (-gamma) * s )
    K    = ftz( identical_exp(e) )

# =========================================================================
# DEVIATION 630: the RBF exponential is FLOAT32 through `identical_exp`,
# not their promoted double `exp`. There is no float64 on the Apple GPU, so
# the double arm cannot exist on this column; the float arm is the one
# arithmetic on every vendor (IDENTITY_PATHS row 12). Measured against the
# host Float64 reference of THEIR spelling in `svc_check.mojo::
# check_rbf_float_vs_double_reference` (max ULP distance printed there).
# The norm is a serial chain rather than their block fold for the same
# reason `pinned_distance_tile.mojo` is one thread per cell: no fold shape
# to pin. Under FAST the only differences from theirs are the missing
# double promotion and the norm's fold shape; FAST makes no bit claim.
# =========================================================================
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.math import exp
from std.memory import stack_allocation
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from core.gemm import gemm_nt
from gemm.checks.gemm_identical import (
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
)
from gemm.checks.gemm_oracle import OP_NT
from checks.numerics import (
    NUMERIC_FAST,
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_exp,
    identical_mul_add,
)
from svm.impl.svm_parameter import KERNEL_LINEAR, KERNEL_RBF, KernelParams


#: SABOTAGE (svc_check "std exp under IDENTICAL"): route the RBF exponential
#: through the stdlib `exp` instead of `identical_exp`. Must FAIL the
#: device-vs-oracle gate under IDENTICAL on the RBF fixtures.
comptime SAB_STD_EXP = is_defined["MOJOLEARN_SVM_SABOTAGE_STD_EXP"]()

#: SABOTAGE ("drop ftz at the f seam"): see `smosolver.mojo`; listed here
#: so every sabotage define has one place where its name is spelled.
comptime SAB_NO_FTZ = is_defined["MOJOLEARN_SVM_SABOTAGE_NO_FTZ"]()

comptime KM_TPB = 256


def _grid(n: Int) -> Int:
    return (n + KM_TPB - 1) // KM_TPB


def row_norm_l2sq_kernel(
    out_norm: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
):
    """`matrixRowNormL2` -> `raft::linalg::rowNorm<L2Norm>` (the squared
    norm, no sqrt), one thread per row, ascending serial chain."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_rows_in):
        var k = Int(n_cols_in)
        var acc = Float32(0.0)
        for c in range(k):
            var v = ftz(x.unsafe_load(i * k + c))
            acc = ftz(identical_mul_add(v, v, acc))
        out_norm.unsafe_store(i, ftz(acc))


def rbf_kernel_expanded_kernel(
    inout: MutPointer[Float32, MutAnyOrigin],
    rows_in: Int32,
    cols_in: Int32,
    norm_x: MutPointer[Float32, MutAnyOrigin],
    norm_y: MutPointer[Float32, MutAnyOrigin],
    gain: Float32,
):
    """`rbf_kernel_expanded`: `inout[i, j] = exp(-gain * (norm_x[i] +
    norm_y[j] - inout[i, j] * 2))`, row-major `[rows x cols]` here where
    theirs is column-major with `ld`. One thread per cell."""
    var rows = Int(rows_in)
    var cols = Int(cols_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < rows * cols:
        var i = t // cols
        var j = t - i * cols
        var dot = inout.unsafe_load(t)
        comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
            var s = ftz(
                ftz(ftz(norm_x.unsafe_load(i)) + ftz(norm_y.unsafe_load(j)))
                - ftz(Float32(2.0) * ftz(dot))
            )
            var e = ftz((-gain) * s)
            comptime if SAB_STD_EXP:
                inout.unsafe_store(t, ftz(exp(e)))
            else:
                inout.unsafe_store(t, ftz(identical_exp(e)))
        else:
            var s = norm_x.unsafe_load(i) + norm_y.unsafe_load(j) - dot * Float32(2.0)
            inout.unsafe_store(t, exp(-gain * s))


def row_norms_l2sq(
    ctx: DeviceContext,
    mut out_norm: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
) raises:
    """`ML::SVM::matrixRowNorm(handle, matrix, out, L2Norm)`."""
    if n_rows <= 0:
        return
    ctx.enqueue_function[row_norm_l2sq_kernel](
        out_norm.unsafe_ptr(), x.unsafe_ptr(), Int32(n_rows), Int32(n_cols),
        grid_dim=_grid(n_rows), block_dim=KM_TPB,
    )


def kernel_workspace_floats(m: Int, n: Int, k: Int) -> Int:
    """What `kernel_op` needs in `ws` for an `m x n` product over `k`."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return identical_gemm_workspace_max_floats(m, n, k)
    return 1



# ---------------------------------------------------------------------------
# DEVIATION 2492 (2026-09-10): THE FUSED FAST RBF TILE
# ---------------------------------------------------------------------------
# `kernel_op` is a GEMM over k = n_features followed by the expansion
# epilogue, which is upstream's shape (`GramMatrixBase::evaluate`, then
# `rbf_kernel_expanded`). For the SMO's tiles k is small (tens of features)
# and the product is `nnz x n_rows` cells, so the GEMM is skinny in exactly
# the dimension a GEMM is tiled for: measured on the M4 at
# 530 x 50,000 x 28, `gemm_nt` took 16.5 ms (90 GFLOP/s) and the epilogue
# another 6 ms reading the tile back. FAST computes the RBF tile in ONE
# kernel instead: each thread owns one row of `b` (its features in
# registers, its squared norm), the rows of `a` stream through shared
# memory with their norms, and the cell is `exp(-gamma * (na + nb - 2 dot))`
# written once, coalesced along the thread axis. Register rows are
# specialized on k rounded up to a multiple of 4, up to 64 features;
# wider inputs keep the GEMM path. IDENTICAL keeps `identical_gemm_into`
# and the pinned epilogue (row 12 exp, row 10 ftz, row 24 one-thread
# feature axis); this arm is FAST only and the FAST arm is free to move
# (the gate for changing FAST is accuracy, and the SVC surface test and
# `svm/checks` run the FAST build against the host oracle).
comptime RBF_FUSED_TPB = 256
comptime RBF_FUSED_TILE_FLOATS = 3072
comptime RBF_FUSED_KREG = 64


def rbf_fused_tile_kernel[KPAD: Int](
    dst: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    norm_a: MutPointer[Float32, MutAnyOrigin],
    norm_b: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    gain: Float32,
):
    """`out[i, j] = exp(-gain * (norm_a[i] + norm_b[j] - 2 <a_i, b_j>))`,
    row-major `[m x n]`, one thread per column j, `a` tiled through shared
    memory. `KPAD` is k rounded up to a multiple of 4; pad lanes are zero
    on both sides and add nothing to the dot."""
    comptime ROWS = RBF_FUSED_TILE_FLOATS // KPAD
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var tid = Int(thread_idx.x)
    var j = Int(block_idx.x) * RBF_FUSED_TPB + tid
    var active = j < n

    var tile = stack_allocation[
        RBF_FUSED_TILE_FLOATS,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var tile_norm = stack_allocation[
        ROWS, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var breg = stack_allocation[KPAD, Scalar[DType.float32]]()
    var nb = Float32(0.0)
    comptime for f in range(KPAD):
        var v = Float32(0.0)
        if active and f < k:
            v = b.unsafe_load(j * k + f)
        breg.unsafe_store(f, v)
    if active:
        nb = norm_b.unsafe_load(j)

    var i0 = 0
    while i0 < m:
        var rows_here = m - i0
        if rows_here > ROWS:
            rows_here = ROWS
        barrier()
        var total = rows_here * KPAD
        var idx = tid
        while idx < total:
            var r = idx // KPAD
            var f = idx - r * KPAD
            var v = Float32(0.0)
            if f < k:
                v = a.unsafe_load((i0 + r) * k + f)
            tile.unsafe_store(idx, v)
            idx += RBF_FUSED_TPB
        idx = tid
        while idx < rows_here:
            tile_norm.unsafe_store(idx, norm_a.unsafe_load(i0 + idx))
            idx += RBF_FUSED_TPB
        barrier()
        if active:
            for r in range(rows_here):
                var tb = r * KPAD
                var dot = Float32(0.0)
                comptime for f in range(KPAD):
                    dot += breg.unsafe_load(f) * tile.unsafe_load(tb + f)
                var s = tile_norm.unsafe_load(r) + nb - dot * Float32(2.0)
                dst.unsafe_store((i0 + r) * n + j, exp(-gain * s))
        i0 += ROWS


def _rbf_fused_launch[KPAD: Int](
    ctx: DeviceContext,
    mut out: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut norm_a: DeviceBuffer[DType.float32],
    mut norm_b: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    gain: Float32,
) raises:
    ctx.enqueue_function[rbf_fused_tile_kernel[KPAD]](
        out.unsafe_ptr(), a.unsafe_ptr(), b.unsafe_ptr(),
        norm_a.unsafe_ptr(), norm_b.unsafe_ptr(),
        Int32(m), Int32(n), Int32(k), gain,
        grid_dim=(_grid_tpb(n, RBF_FUSED_TPB), 1, 1),
        block_dim=(RBF_FUSED_TPB, 1, 1),
    )


def _grid_tpb(n: Int, tpb: Int) -> Int:
    return (n + tpb - 1) // tpb


def rbf_fused_tile(
    ctx: DeviceContext,
    mut out: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut norm_a: DeviceBuffer[DType.float32],
    mut norm_b: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    gain: Float32,
) raises -> Bool:
    """DEVIATION 2492's dispatch: True when the fused kernel served the
    tile, False when k is too wide and the caller must take the GEMM path."""
    var kpad = ((k + 3) // 4) * 4
    if kpad == 4:
        _rbf_fused_launch[4](ctx, out, a, b, norm_a, norm_b, m, n, k, gain)
    elif kpad == 8:
        _rbf_fused_launch[8](ctx, out, a, b, norm_a, norm_b, m, n, k, gain)
    elif kpad == 12:
        _rbf_fused_launch[12](ctx, out, a, b, norm_a, norm_b, m, n, k, gain)
    elif kpad == 16:
        _rbf_fused_launch[16](ctx, out, a, b, norm_a, norm_b, m, n, k, gain)
    elif kpad == 20:
        _rbf_fused_launch[20](ctx, out, a, b, norm_a, norm_b, m, n, k, gain)
    elif kpad == 24:
        _rbf_fused_launch[24](ctx, out, a, b, norm_a, norm_b, m, n, k, gain)
    elif kpad == 28:
        _rbf_fused_launch[28](ctx, out, a, b, norm_a, norm_b, m, n, k, gain)
    elif kpad == 32:
        _rbf_fused_launch[32](ctx, out, a, b, norm_a, norm_b, m, n, k, gain)
    elif kpad == 36:
        _rbf_fused_launch[36](ctx, out, a, b, norm_a, norm_b, m, n, k, gain)
    elif kpad == 40:
        _rbf_fused_launch[40](ctx, out, a, b, norm_a, norm_b, m, n, k, gain)
    elif kpad == 44:
        _rbf_fused_launch[44](ctx, out, a, b, norm_a, norm_b, m, n, k, gain)
    elif kpad == 48:
        _rbf_fused_launch[48](ctx, out, a, b, norm_a, norm_b, m, n, k, gain)
    elif kpad == 52:
        _rbf_fused_launch[52](ctx, out, a, b, norm_a, norm_b, m, n, k, gain)
    elif kpad == 56:
        _rbf_fused_launch[56](ctx, out, a, b, norm_a, norm_b, m, n, k, gain)
    elif kpad == 60:
        _rbf_fused_launch[60](ctx, out, a, b, norm_a, norm_b, m, n, k, gain)
    elif kpad == 64:
        _rbf_fused_launch[64](ctx, out, a, b, norm_a, norm_b, m, n, k, gain)
    else:
        return False
    return True


def kernel_op(
    ctx: DeviceContext,
    kp: KernelParams,
    mut out: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    mut norm_a: DeviceBuffer[DType.float32],
    mut norm_b: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
) raises:
    """`KernelOp(handle, kernel, x1, n1, n_cols, x2, n2, out, norm_x1,
    norm_x2)`: `out[m x n] = K(a_i, b_j)`, row-major. `GramMatrixBase::
    evaluate` (linear) or `RBFKernel::evaluate` (linear + expansion).
    ASYNCHRONOUS; `ws` is the caller's identical-GEMM workspace, at least
    `kernel_workspace_floats(m, n, k)` floats."""
    if m <= 0 or n <= 0:
        return
    # DEVIATION 2492: FAST RBF in one fused kernel when k fits a register row.
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_FAST:
        if kp.kernel == KERNEL_RBF and k <= RBF_FUSED_KREG:
            if rbf_fused_tile(ctx, out, a, b, norm_a, norm_b, m, n, k, Float32(kp.gamma)):
                return
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        identical_gemm_into(ctx, out, a, b, ws, m, n, k, OP_NT)
    else:
        gemm_nt(ctx, out, a, b, m, n, k)
    if kp.kernel == KERNEL_RBF:
        ctx.enqueue_function[rbf_kernel_expanded_kernel](
            out.unsafe_ptr(), Int32(m), Int32(n),
            norm_a.unsafe_ptr(), norm_b.unsafe_ptr(), Float32(kp.gamma),
            grid_dim=_grid(m * n), block_dim=KM_TPB,
        )
    elif kp.kernel != KERNEL_LINEAR:
        raise Error("svm kernel_op: unimplemented kernel " + String(kp.kernel))
