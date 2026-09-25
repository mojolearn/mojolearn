# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The LINEAR and RBF kernel matrices: `GramMatrixBase::linear`,
`RBFKernel::evaluate`, `rbf_kernel_expanded`, `matrixRowNormL2`.

Reference: `cuvs/cpp/src/distance/detail/kernels/kernel_matrices.cu` (cuVS
`94c2819`; the `cuvs::distance::kernels` cuML 26.08 links; RAFT's
`distance/detail/kernels/kernel_matrices.cuh` is the same code one
repository earlier). Dense, row-major, FP32. POLYNOMIAL and TANH are NOT
implemented (refused by name in `svm_parameter.mojo`; `svm/NOT_IMPLEMENTED.tsv`).

THE ROUNDING SEQUENCE (svm/README.md, identity content section 1). The reference:

    linear:  out = x1 . x2^T                          (cuBLAS gemm)
    rbf:     out = exp(-1.0 * gain * (norm_x[i] + norm_y[j] - out * 2))
             with norm = rowNorm<L2Norm> (SQUARED; a block fold), and for
             math_t = float the `-1.0 *` promotes to DOUBLE so the exp is
             the double one, rounded to float on store. No clamp at zero.

This implementation, both modes the same association, the pins under IDENTICAL:

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
from std.sys.info import has_apple_gpu_accelerator
from std.math import exp
from std.memory import stack_allocation
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from core.gemm import gemm_nt
from core.multi_gpu import peer_clone
from core.step_phase import STEP_PHASE_TIMERS
from std.os import getenv
from max.algorithm import sync_parallelize
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
from svm.impl.svm_parameter import KERNEL_LINEAR, KERNEL_POLYNOMIAL, KERNEL_RBF, KernelParams
from kernel_methods.impl.distance.kernel_matrices import polynomial_epilogue_kernel


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
# epilogue, which is the reference's shape (`GramMatrixBase::evaluate`, then
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
    distribute: Bool = True,
) raises:
    """`KernelOp(handle, kernel, x1, n1, n_cols, x2, n2, out, norm_x1,
    norm_x2)`: `out[m x n] = K(a_i, b_j)`, row-major. `GramMatrixBase::
    evaluate` (linear) or `RBFKernel::evaluate` (linear + expansion).
    ASYNCHRONOUS; `ws` is the caller's identical-GEMM workspace, at least
    `kernel_workspace_floats(m, n, k)` floats."""
    if m <= 0 or n <= 0:
        return
    if distribute:
        var setting = String(getenv("MOJOLEARN_SVM_DEVICE_COUNT"))
        if setting != "" and setting != "1":
            var count = Int(setting)
            if count < 1 or count > 64 or GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
                raise Error("parallel SVM kernels require IDENTICAL and 1..64 devices")
            comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL: _kernel_rows(ctx, kp, out, a, b, m, n, k, norm_a, norm_b, count)
            return
    # DEVIATION 2492: FAST RBF in one fused kernel when k fits a register row.
    # Apple only. 0.8.19's NVPTX/AMDGPU FAST hang was NOT these kernels: it was the
    # kernel_op -> _kernel_rows -> kernel_op cycle reaching MAX matmul (gated above).
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator():
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
    elif kp.kernel == KERNEL_POLYNOMIAL:
        # cuVS `PolynomialKernel::evaluate`: the linear Gram above, then
        # `pow(gain * K + offset, exponent)` cell by cell, spelled as the
        # kernel_methods lane's DEVIATION 1663 epilogue (one fused
        # multiply-add, then an ascending repeated product, so a negative
        # base is legal where `identical_pow` would return NaN).
        ctx.enqueue_function[polynomial_epilogue_kernel](
            out.unsafe_ptr(), Int32(m * n), Int32(kp.degree),
            Float32(kp.gamma), Float32(kp.coef0),
            grid_dim=_grid(m * n), block_dim=KM_TPB,
        )
    elif kp.kernel != KERNEL_LINEAR:
        raise Error("svm kernel_op: unimplemented kernel " + String(kp.kernel))


@fieldwise_init
struct SVMKernelShard(Movable):
    var ctx: DeviceContext
    var a: DeviceBuffer[DType.float32]
    var b: DeviceBuffer[DType.float32]
    var norm_a: DeviceBuffer[DType.float32]
    var norm_b: DeviceBuffer[DType.float32]
    var out: DeviceBuffer[DType.float32]
    var ws: DeviceBuffer[DType.float32]
    var first: Int
    var rows: Int

    def __deinit__(deinit self):
        _ = self.a^
        _ = self.b^
        _ = self.norm_a^
        _ = self.norm_b^
        _ = self.out^
        _ = self.ws^
        try:
            self.ctx.synchronize()
        except:
            pass
        _ = self.ctx^


def _kernel_rows(ctx: DeviceContext, kp: KernelParams,
    mut out: DeviceBuffer[DType.float32], mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32], m: Int, n: Int, k: Int,
    mut norm_a: DeviceBuffer[DType.float32], mut norm_b: DeviceBuffer[DType.float32],
    count: Int,
) raises:
    """Partition output rows, retaining each original FP32-v1 contraction.

    The feature axis is never split or padded. Original norm bytes and the
    original RBF epilogue are reused. The SMO working set, updates and stopping
    rules remain on the root. Per-call staging is not a speed/capacity claim.

    THE NEGATIVE CONTROL. `-D MOJOLEARN_SVM_PARALLEL_SABOTAGE=1` makes every
    owner above rank 0 read its left operand rows, and its RBF row norms, one
    row early. `rows` is unchanged, the per-shard allocations keep their sizes
    and the write-back keeps the true `shard.first`, so nothing but the values
    contracted moves. It is a `comptime if`, so no production bit can move, and
    it is INERT AT ONE DEVICE: the shift is guarded by `rank > 0` and a
    one-device column has only rank 0. That is what makes a moved `par-svm`,
    `par-svm-svr`, `par-kernel-ridge` or `par-nystroem` cell attributable to the
    define rather than to the second device. (`MOJOLEARN_SVM_DEVICE_COUNT` is
    also what `kernel_methods/checks/kernel_matrix.mojo` reads, which is why
    the kernel-method lanes sit behind this one define.) Owed a two-device
    column (`MOJOLEARN_PAR_DEVICES=0,1`); no host binding restates this driver.
    """
    ctx.synchronize()
    var active = min(count, m)
    comptime if STEP_PHASE_TIMERS:
        if active > 1:
            raise Error("parallel SVM cannot use process-global GEMM phase counters")
    var shards = List[SVMKernelShard]()
    for rank in range(active):
        var first = m * rank // active
        var rows = m * (rank + 1) // active - first
        var source = first
        comptime if is_defined["MOJOLEARN_SVM_PARALLEL_SABOTAGE"]():
            # Check-only arm: later owners read their left rows one row early.
            if rank > 0:
                source = first - 1
        var device = DeviceContext(device_id=rank)
        var av = a.create_sub_buffer[DType.float32](source * k, rows * k)
        var bv = b.create_sub_buffer[DType.float32](0, n * k)
        var na = norm_a.create_sub_buffer[DType.float32](source if kp.kernel == KERNEL_RBF else 0,
                                                        rows if kp.kernel == KERNEL_RBF else 1)
        var nb = norm_b.create_sub_buffer[DType.float32](0, n if kp.kernel == KERNEL_RBF else 1)
        var local_a = peer_clone(ctx, device, av)
        var local_b = peer_clone(ctx, device, bv)
        var local_na = peer_clone(ctx, device, na)
        var local_nb = peer_clone(ctx, device, nb)
        var output = device.enqueue_create_buffer[DType.float32](rows * n)
        var workspace = device.enqueue_create_buffer[DType.float32](kernel_workspace_floats(rows, n, k))
        device.synchronize()
        shards.append(SVMKernelShard(device^, local_a^, local_b^, local_na^, local_nb^,
                                     output^, workspace^, first, rows))
    var failures = List[Int](length=active, fill=0)
    var sp = rebind[MutPointer[SVMKernelShard, MutUntrackedOrigin]](shards.unsafe_ptr())
    var fp = rebind[MutPointer[Int, MutUntrackedOrigin]](failures.unsafe_ptr())
    def task(rank: Int) {imm sp, imm fp, imm kp, imm n, imm k}:
        try:
            ref shard = sp[rank]
            kernel_op(shard.ctx, kp, shard.out, shard.a, shard.b, shard.rows, n, k,
                      shard.norm_a, shard.norm_b, shard.ws, False)
            shard.ctx.synchronize()
        except:
            fp[rank] = 1
    if active == 1:
        task(0)
    else:
        sync_parallelize(task, active)
    for rank in range(active):
        if failures[rank] != 0:
            raise Error("SVM kernel row shard failed: " + String(rank))
    for rank in range(active):
        ref shard = shards[rank]
        var destination = out.create_sub_buffer[DType.float32](shard.first * n, shard.rows * n)
        shard.out.enqueue_copy_to(destination)
        shard.ctx.synchronize()
    _ = shards^
    ctx.synchronize()
