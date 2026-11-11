# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Cholesky, the triangular solves and the log-determinant as executor step
ops (lane n7-device-linalg-ops, 2026-11-08; docs/plans/EXECUTOR_N7_OPS.md).

Every op only ENQUEUES: no host read, no drain. The factor's LAPACK `info`
stays in a device Int32 word, CHOL_DEFER_INFO on every vendor: each launch
of the factor, the solves and the log-determinant returns at once when the
word is nonzero, and the family either reads the word at its own sync point
(KRR: once per fit) or makes it join the loop's predicate
(`chol_op_info_joins_loop`: a failed factor stops the loop at that
iteration, before any later kernel of the body consumes the factor).

The info word, per iteration:
  0                  the factor ran and succeeded (L is in `a`)
  k > 0              the leading minor of order k is not positive definite
                     (LAPACK's info); `a` holds the factor frozen at the
                     failing panel, which nothing may consume
  CHOL_OP_GATED (-1) the loop was already done: nothing ran, nothing was
                     written (`chol_op_begin` sets it from the gate)

The factor (`chol_op_factor`) is `potrf_lower`'s arithmetic at the pinned
width CHOL_NB_PINNED on every column, in one read-free schedule per column:
  * NVIDIA, AMD, and Apple FAST: the strip schedule
    (`chol_strips_enqueue`, cholesky/checks/potrf_strip.mojo);
  * Apple IDENTICAL: the left-looking matrix-unit schedule
    (`chol_left_enqueue_deferred`, CHOL_APPLE_LEFT);
  * n < CHOL_NB_PINNED: the one guarded panel factor;
then the guarded upper-triangle zeros. Each schedule computes the profile's
cells in the profile's order (the strip and left-looking banners in
potrf.mojo), so every column gets the host twin's words
(`cholesky/host/chol_oracle.mojo::chol_host_factor_lower`). Apple FAST took
the vendor-GEMM panel route in `potrf_lower`; the op runs the strip
schedule there (one route; FAST needs no particular bits). RUN OWED: the
M3 A/B of the two on the FAST board's cholesky / gaussian-process rows.

The solves are the default kernels of `trsm_lower` / `trsm_upper`
(multi-RHS sweep from CHOL_MR_RB right-hand sides, the column sweep below
it, the back-substitution ring), guarded; every one is
`trsm_lower_kernel` / `trsm_upper_kernel`'s chain term for term, so the
host twins are `chol_host_trsm_lower` / `chol_host_trsm_upper`. Apple FAST's
blocked `fast_cho_solve` is not an op route (one route; RUN OWED with the
factor's A/B). The log-determinant is `enqueue_logdet`'s fold
(cholesky/logdet_fold.mojo) with a guarded store; host twin:
`CholHostFactor.logdet`.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu import block_idx, thread_idx

from cholesky.checks.potrf import (
    CHOL_APPLE_LEFT,
    CHOL_ELEM_TPB,
    CHOL_NB_PINNED,
    CHOL_PANEL_TPB,
    chol_left_enqueue_deferred,
    chol_strips_enqueue,
    enqueue_logdet_guarded,
    panel_factor_guarded_kernel,
    zero_upper_guarded_kernel,
)
from cholesky.checks.potrf_strip import CHOL_STRIP_FITS, CS_NB
from cholesky.checks.trsm import (
    CHOL_BACK_NT,
    CHOL_MR_NT,
    CHOL_MR_RB,
    CHOL_SWEEP_NT,
    trsm_lower_multi_rhs_guarded_kernel,
    trsm_lower_sweep_guarded_kernel,
    trsm_upper_ring_guarded_kernel,
)
from cholesky.impl.matrix.detail.matrix import copy_vector_from_matrix_diagonal_kernel
from cholesky.logdet_fold import logdet_blocks
from core.step_gate import StepI32, step_fail_joins_loop

#: The info word of an op whose loop was already done (nothing ran).
comptime CHOL_OP_GATED = Int32(-1)

comptime _F32 = MutPointer[Float32, MutAnyOrigin]


def chol_op_begin_kernel(gate: StepI32, info: StepI32):
    """One thread: the iteration's info word, 0 while the loop runs and
    CHOL_OP_GATED once it is done."""
    if Int(block_idx.x) != 0 or Int(thread_idx.x) != 0:
        return
    if gate.unsafe_load(0) != Int32(0):
        info.unsafe_store(0, CHOL_OP_GATED)
    else:
        info.unsafe_store(0, Int32(0))


def chol_op_begin(ctx: DeviceContext, mut gate: DeviceBuffer[DType.int32], mut info: DeviceBuffer[DType.int32]) raises:
    """Reset `info` for this iteration from the gate (word 0 of `gate`: the
    loop's done word, or `step_gate_open_word` outside a loop). Enqueue it
    before `chol_op_factor` every iteration."""
    ctx.enqueue_function[chol_op_begin_kernel](  # small-launch(one word): the factor's info word from the loop gate
        gate.unsafe_ptr(), info.unsafe_ptr(), grid_dim=(1, 1, 1), block_dim=(1, 1, 1),
    )


def chol_op_factor(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.float32],
    mut info: DeviceBuffer[DType.int32],
    n: Int,
) raises:
    """`A = L L^T` in place (lower, row major, n x n; the strict upper
    triangle becomes +0.0 on success), enqueued and read-free. `info` must
    have been set by `chol_op_begin` (or zeroed) on this stream. Any ridge
    is the caller's, added before (`add_jitter`); the matrix is not
    validated here (the family validates its input once, at admission)."""
    if n <= 0:
        raise Error("chol_op_factor: n must be positive, got " + String(n))
    if len(a) < n * n:
        raise Error("chol_op_factor: the matrix buffer holds " + String(len(a)) + " floats, n = " + String(n) + " needs " + String(n * n))
    if n < CS_NB:
        # one panel (potrf_lower at nb = n): the guarded unblocked factor
        ctx.enqueue_function[panel_factor_guarded_kernel](  # small-launch(n: below one panel): factors the whole n x n block, columns serial, rows across the block
            a.unsafe_ptr(), info.unsafe_ptr(), Int32(n), Int32(0), Int32(n),
            grid_dim=(1, 1, 1), block_dim=(CHOL_PANEL_TPB, 1, 1),
        )
    else:
        comptime if CHOL_APPLE_LEFT:
            _ = chol_left_enqueue_deferred(ctx, a, info, n, CHOL_NB_PINNED)
        else:
            comptime if not CHOL_STRIP_FITS:
                raise Error("chol_op_factor: the strip schedule's shared pages do not fit this column")
            chol_strips_enqueue(ctx, a, info, n)
    ctx.enqueue_function[zero_upper_guarded_kernel](
        a.unsafe_ptr(), info.unsafe_ptr(), Int32(n),
        grid_dim=((n * n + CHOL_ELEM_TPB - 1) // CHOL_ELEM_TPB, 1, 1), block_dim=(CHOL_ELEM_TPB, 1, 1),
    )


def _chol_op_check_solve(l: DeviceBuffer[DType.float32], b: DeviceBuffer[DType.float32], n: Int, nrhs: Int, what: String) raises:
    if n <= 0 or nrhs <= 0:
        raise Error(what + ": n and nrhs must both be positive, got n=" + String(n) + " nrhs=" + String(nrhs))
    if len(l) < n * n or len(b) < n * nrhs:
        raise Error(what + ": a buffer is shorter than its shape (n=" + String(n) + ", nrhs=" + String(nrhs) + ")")


def chol_op_solve_lower(
    ctx: DeviceContext,
    mut l: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut info: DeviceBuffer[DType.int32],
    n: Int,
    nrhs: Int,
) raises:
    """`L X = B` in place over `b` (n x nrhs row major), skipped when
    `info[0] != 0`."""
    _chol_op_check_solve(l, b, n, nrhs, "chol_op_solve_lower")
    if nrhs >= CHOL_MR_RB:
        ctx.enqueue_function[trsm_lower_multi_rhs_guarded_kernel](
            info.unsafe_ptr(), l.unsafe_ptr(), b.unsafe_ptr(), Int32(n), Int32(nrhs), Int32(n),
            grid_dim=((nrhs + CHOL_MR_RB - 1) // CHOL_MR_RB, 1, 1), block_dim=(CHOL_MR_NT, 1, 1),
        )
    else:
        ctx.enqueue_function[trsm_lower_sweep_guarded_kernel](
            info.unsafe_ptr(), l.unsafe_ptr(), b.unsafe_ptr(), Int32(n), Int32(nrhs), Int32(n),
            grid_dim=(nrhs, 1, 1), block_dim=(CHOL_SWEEP_NT, 1, 1),
        )


def chol_op_solve_upper(
    ctx: DeviceContext,
    mut l: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut info: DeviceBuffer[DType.int32],
    n: Int,
    nrhs: Int,
) raises:
    """`L^T X = B` in place over `b` (the same lower factor), skipped when
    `info[0] != 0`."""
    _chol_op_check_solve(l, b, n, nrhs, "chol_op_solve_upper")
    ctx.enqueue_function[trsm_upper_ring_guarded_kernel](
        info.unsafe_ptr(), l.unsafe_ptr(), b.unsafe_ptr(), Int32(n), Int32(nrhs), Int32(n),
        grid_dim=(nrhs, 1, 1), block_dim=(CHOL_BACK_NT, 1, 1),
    )


def chol_op_cho_solve(
    ctx: DeviceContext,
    mut l: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut info: DeviceBuffer[DType.int32],
    n: Int,
    nrhs: Int,
) raises:
    """`A X = B` given `A = L L^T` (potrs, lower): the forward then the back
    substitution, in place over `b`; skipped when `info[0] != 0`."""
    chol_op_solve_lower(ctx, l, b, info, n, nrhs)
    chol_op_solve_upper(ctx, l, b, info, n, nrhs)


def chol_op_logdet_scratch_floats(n: Int) -> Int:
    """Floats `chol_op_logdet` needs in `scratch`: the diagonal, then the
    fold's partials (two levels)."""
    return n + 2 * logdet_blocks(n)


def chol_op_logdet(
    ctx: DeviceContext,
    mut l: DeviceBuffer[DType.float32],
    mut info: DeviceBuffer[DType.int32],
    mut scratch: DeviceBuffer[DType.float32],
    mut out: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    """`out[0] = log det A = 2 sum log L_ii` (cholesky/logdet_fold.mojo's
    order), stored only when `info[0] == 0`. `scratch` holds
    `chol_op_logdet_scratch_floats(n)` floats."""
    if len(scratch) < chol_op_logdet_scratch_floats(n) or len(out) < 1:
        raise Error("chol_op_logdet: the scratch holds " + String(len(scratch)) + " floats, n = " + String(n) + " needs " + String(chol_op_logdet_scratch_floats(n)))
    var sp = _F32(unsafe_from_address=Int(scratch.unsafe_ptr()))
    ctx.enqueue_function[copy_vector_from_matrix_diagonal_kernel](
        sp, l.unsafe_ptr(), Int32(n), Int32(n),
        grid_dim=((n + CHOL_ELEM_TPB - 1) // CHOL_ELEM_TPB, 1, 1), block_dim=(CHOL_ELEM_TPB, 1, 1),
    )
    var ip = StepI32(unsafe_from_address=Int(info.unsafe_ptr()))
    var op = _F32(unsafe_from_address=Int(out.unsafe_ptr()))
    enqueue_logdet_guarded(ctx, ip, sp, sp + n, op, n)


def chol_op_info_joins_loop(
    ctx: DeviceContext, mut words: DeviceBuffer[DType.int32], mut info: DeviceBuffer[DType.int32], it: Int,
) raises:
    """A failed factor (info > 0) stops the loop at iteration `it`; the
    code lands in the register block's STEP_FAIL word
    (`LoopRegisters.failure()`). Enqueue it right after `chol_op_factor`."""
    step_fail_joins_loop(ctx, words, info, it)


def chol_op_zero_on_fail_kernel(info: StepI32, buf: _F32, count: Int32):
    """`buf[0, count) = +0.0` when the factor failed or was gated
    (`info[0] != 0`); nothing otherwise."""
    var t = Int(block_idx.x) * CHOL_ELEM_TPB + Int(thread_idx.x)
    if t >= Int(count):
        return
    if info.unsafe_load(0) != Int32(0):
        buf.unsafe_store(t, Float32(0.0))


def chol_op_zero_on_fail(
    ctx: DeviceContext, mut info: DeviceBuffer[DType.int32], mut buf: DeviceBuffer[DType.float32], count: Int,
) raises:
    """Zero `buf[0, count)` on the device when `info[0] != 0` (a failed
    factor's solution is handed back as zeros beside its nonzero info, the
    families' contract) without reading `info` on the host (lane
    n8-gp-krr-smo)."""
    if count <= 0:
        return
    if len(buf) < count:
        raise Error("chol_op_zero_on_fail: the buffer holds " + String(len(buf)) + " floats, count = " + String(count))
    var ip = StepI32(unsafe_from_address=Int(info.unsafe_ptr()))
    var bp = _F32(unsafe_from_address=Int(buf.unsafe_ptr()))
    ctx.enqueue_function[chol_op_zero_on_fail_kernel](
        ip, bp, Int32(count),
        grid_dim=((count + CHOL_ELEM_TPB - 1) // CHOL_ELEM_TPB, 1, 1), block_dim=(CHOL_ELEM_TPB, 1, 1),
    )
