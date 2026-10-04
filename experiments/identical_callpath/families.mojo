# SPDX-License-Identifier: Apache-2.0
"""Uncompiled internal session adapters; existing arithmetic and dispatch.

These are device-stage building blocks, not replacements for estimator host
validation, tracing, initialization, convergence checks or model ownership.
Use only when the baseline already calls the same primitive/profile.
"""
from core.row_norms import NORM_TPB, row_norm_kernel
from core.column_stats import STATS_TPB, column_mean_kernel, shift_columns_kernel
from core.gemm import gemm_nt
from gemm.checks.gemm_identical import (
    identical_gemm_into, identical_gemm_workspace_max_floats,
)
from gemm.checks.gemm_oracle import OP_NN, OP_NT, OP_TN
from experiments.identical_callpath.session import IdenticalCallSession


def _require_f32(session: IdenticalCallSession, slot: Int, count: Int) raises:
    if slot < 0 or slot >= len(session.f32.device):
        raise Error("invalid session Float32 slot")
    if count < 0 or session.f32.sizes[slot] < count:
        raise Error("session Float32 slot is too small")


def _matrix_cells(rows: Int, cols: Int) raises -> Int:
    if rows <= 0 or cols <= 0 or rows > 2147483647 or cols > 2147483647:
        raise Error("adapter requires positive Int32 matrix dimensions")
    var cells = rows * cols
    if cells > 2147483647:
        raise Error("adapter refuses more than Int32 matrix cells")
    return cells


def enqueue_row_norms(mut session: IdenticalCallSession, output: Int,
                      source: Int, rows: Int, cols: Int,
                      take_sqrt: Bool = False) raises:
    """Shared primitive used by neighbors, clustering and distance paths."""
    try:
        session.require_active()
        var cells = _matrix_cells(rows, cols)
        _require_f32(session, source, cells)
        _require_f32(session, output, rows)
        if source == output:
            raise Error("row norm source/output must not alias")
        session.ctx.enqueue_function[row_norm_kernel](
            session.f32.device[output].unsafe_ptr(), session.f32.device[source].unsafe_ptr(),
            Int32(cols), Int32(take_sqrt), grid_dim=(rows, 1, 1),
            block_dim=(NORM_TPB, 1, 1),
        )
    except e:
        session.abort()
        raise e


def enqueue_column_means(mut session: IdenticalCallSession, output: Int,
                        source: Int, rows: Int, cols: Int) raises:
    """Same PCA/OLS column mean reduction and pinned block width."""
    try:
        session.require_active()
        var cells = _matrix_cells(rows, cols)
        _require_f32(session, source, cells)
        _require_f32(session, output, cols)
        if source == output:
            raise Error("column mean source/output must not alias")
        session.ctx.enqueue_function[column_mean_kernel](
            session.f32.device[output].unsafe_ptr(), session.f32.device[source].unsafe_ptr(),
            Int32(rows), Int32(cols), grid_dim=(cols, 1, 1),
            block_dim=(STATS_TPB, 1, 1),
        )
    except e:
        session.abort()
        raise e


def enqueue_shift_columns(mut session: IdenticalCallSession, source: Int,
                          means: Int, rows: Int, cols: Int,
                          restore: Bool = False) raises:
    """Baseline PCA centering/restoration; keep its existing fused branch."""
    try:
        session.require_active()
        var cells = _matrix_cells(rows, cols)
        _require_f32(session, source, cells)
        _require_f32(session, means, cols)
        if source == means:
            raise Error("column means must not alias the shifted matrix")
        var sign = Float32(1.0) if restore else Float32(-1.0)
        session.ctx.enqueue_function[shift_columns_kernel](
            session.f32.device[source].unsafe_ptr(), session.f32.device[means].unsafe_ptr(),
            Int32(rows), Int32(cols), sign,
            grid_dim=((cells + 255) // 256, 1, 1), block_dim=(256, 1, 1),
        )
    except e:
        session.abort()
        raise e


def enqueue_core_gemm_nt(mut session: IdenticalCallSession, output: Int,
                        left: Int, right: Int, m: Int, n: Int, k: Int) raises:
    """Retain core.gemm.gemm_nt including its n==1 GEMV specialization."""
    try:
        session.require_active()
        _require_f32(session, output, _matrix_cells(m, n))
        _require_f32(session, left, _matrix_cells(m, k))
        _require_f32(session, right, _matrix_cells(n, k))
        if output == left or output == right:
            raise Error("GEMM output must not alias operands")
        gemm_nt(session.ctx, session.f32.device[output], session.f32.device[left],
                session.f32.device[right], m, n, k)
    except e:
        session.abort()
        raise e


def enqueue_identical_gemm(mut session: IdenticalCallSession, output: Int,
                          left: Int, right: Int, workspace: Int,
                          m: Int, n: Int, k: Int, op: Int) raises:
    """Existing identical_gemm_into dispatcher with pre-reserved workspace.

    Do not substitute this for core.gemm_nt or a different Gram profile.
    Internal plan-specific waits/allocations and trial hooks remain intact.
    """
    try:
        session.require_active()
        if op != OP_NN and op != OP_NT and op != OP_TN:
            raise Error("unsupported GEMM operation")
        _require_f32(session, output, _matrix_cells(m, n))
        _require_f32(session, left, _matrix_cells(m, k))
        _require_f32(session, right, _matrix_cells(n, k))
        _require_f32(session, workspace, identical_gemm_workspace_max_floats(m, n, k))
        if output == left or output == right or workspace == output or workspace == left or workspace == right:
            raise Error("GEMM output/workspace must not alias operands or each other")
        identical_gemm_into(session.ctx, session.f32.device[output],
            session.f32.device[left], session.f32.device[right],
            session.f32.device[workspace], m, n, k, op)
    except e:
        session.abort()
        raise e
