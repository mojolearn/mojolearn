# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device SVD and QR as executor step ops (lane n9b-device-svd-qr,
2026-11-09), next to the N7 ops (x_decomp/eigh_ops.mojo;
plan EXECUTOR_N7_OPS (git history) keeps the contract: enqueue only, the caller's
outputs written only while the gate is open, a status word for a solve that
can fail on the device).

  `so_desc_rank_kernel`, `so_desc_commit_kernel`  `svd_order`'s descending
                 order on the device (`DevExec.svd_desc_into`): s (n,
                 descending; ties keep the lower index, as the stable host
                 order of -s) and vt (n x n, row a = the right singular
                 vector of s[a]). (The one-block SVD op `svd_op_tall` was
                 deleted 2026-11-24 with MOJOLEARN_XD_FAST_SVD_ONE_BLOCK.)
  `qr_op_r`      R (n x n) of the bounded Householder QR / TSQR of a tall A
                 (m x n, CONSUMED), committed through the gate.
  `qr_op_q`      the first qc columns of Q of the sliced Householder QR
                 (x_decomp/qr_sliced_device.mojo `qs_geqrf_device`,
                 `qs_orgqr_device`) of A (m x n, CONSUMED): Q (m x qc),
                 committed through the gate.

HOST COLUMN. The words are the host column's as it stands, so the twins are
the existing host forms: the order kernels are `svd_order`
(`Kit[HostExec].svd`); `qr_op_r`
is `HostExec._qr_r`; `qr_op_q` is `HostExec.geqrf` + `HostExec.orgqr`.

BITS. None move: the order is the host's stable order.

APPLE. The bounded QR's passes keep their waits on Apple (they read nothing;
they pace a Metal queue, x_decomp/qr_bounded.mojo), as the sliced QR's
`_Pace` does; off Apple nothing here waits.
"""
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.kernel_matrix import COLUMN_APPLE, TARGET_COLUMN
from core.householder_qr import qr_slice_count
from core.step_gate import StepI32
from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.moves import order_key
from x_decomp.qr_bounded import QRB_CELLS, qr_factor_bounded
from x_decomp.qr_sliced_device import qs_geqrf_device, qs_orgqr_device

comptime SO_TPB = 256

#: the bounded QR waits only on Apple inside an op (see the module note)
comptime SO_QR_WAIT = TARGET_COLUMN == COLUMN_APPLE


@always_inline
def _so_blocks(count: Int) -> Int:
    return (count + SO_TPB - 1) // SO_TPB if count > 0 else 1


@always_inline
def _fp(buf: DeviceBuffer[DType.float32]) -> F32Ptr:
    return F32Ptr(unsafe_from_address=Int(buf.unsafe_ptr()))


@always_inline
def _ip(buf: DeviceBuffer[DType.int32]) -> I32Ptr:
    return I32Ptr(unsafe_from_address=Int(buf.unsafe_ptr()))


# ---------------------------------------------------------------- kernels


def so_desc_rank_kernel(gate: StepI32, s: F32Ptr, pos: I32Ptr, bad: I32Ptr, n_in: Int32):
    """Thread i: pos[i] = the place of s[i] in the stable ascending order of
    -s (`svd_order` / `order_f`: the key of -s, -0.0 as +0.0, ties in index
    order); a NaN sets bad[0] (`f32_key`'s refusal). Exact compares only."""
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    if gate.unsafe_load(0) != Int32(0):
        return
    var x = s.unsafe_load(i)
    # NaN on the bit pattern (no fast-math fold of x != x on any vendor)
    if (bitcast[DType.uint32](x) & UInt32(0x7FFFFFFF)) > UInt32(0x7F800000):
        bad.unsafe_store(0, Int32(1))
        pos.unsafe_store(i, Int32(i))
        return
    var ki = order_key(x, True)
    var rank = 0
    for j in range(n):
        var kj = order_key(s.unsafe_load(j), True)
        if kj < ki or (kj == ki and j < i):
            rank += 1
    pos.unsafe_store(i, Int32(rank))


def so_desc_commit_kernel(
    gate: StepI32, s: F32Ptr, vt: F32Ptr, pos: I32Ptr, bad: I32Ptr, s_out: F32Ptr, vt_out: F32Ptr, n_in: Int32
):
    """Cell (j, i) of V^T: row j goes to row pos[j] of vt_out, s[j] to
    s_out[pos[j]], while the gate is open (copies). Skipped on a NaN (the
    positions are not a permutation then; status reports it)."""
    var n = Int(n_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= n * n:
        return
    if gate.unsafe_load(0) != Int32(0) or bad.unsafe_load(0) != Int32(0):
        return
    var j = t // n
    var i = t - j * n
    var a = Int(pos.unsafe_load(j))
    vt_out.unsafe_store(a * n + i, vt.unsafe_load(t))
    if i == 0:
        s_out.unsafe_store(a, s.unsafe_load(j))


def so_commit_kernel(gate: StepI32, src: F32Ptr, dst: F32Ptr, count_in: Int32):
    """dst[t] = src[t] while the gate is open (a bit copy)."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(count_in):
        return
    if gate.unsafe_load(0) != Int32(0):
        return
    dst.unsafe_store(t, src.unsafe_load(t))


# ---------------------------------------------------------------- the ops


# TOMBSTONE: MOJOLEARN_XD_FAST_SVD_ONE_BLOCK (NOISE: NV L40S 8 cells, median 1.00, 0.97..1.05) deleted 2026-11-24 by lane/o3-decomp (ab-outcome-3); code recoverable at c23cb55fda.
# Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_XD_FAST_SVD_ONE_BLOCK.ab-outcome-3.patch
# (`svd_op_tall`, the thin SVD of a tall A with the Jacobi in ONE block, and its `so_status_kernel`: the
# solve of that arm only; `so_desc_rank_kernel` / `so_desc_commit_kernel` stay for `DevExec.svd_desc_into`.)


def qr_op_r(
    ctx: DeviceContext,
    mut gate: DeviceBuffer[DType.int32],
    mut a: DeviceBuffer[DType.float32],
    mut r: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
) raises:
    """R (n x n, upper, row major) of the bounded Householder QR / TSQR of a
    tall A (m x n, m >= n >= 1; CONSUMED), written only while the gate is
    open. Enqueued only. A launch macOS cut short leaves the NaN poison in
    R (x_decomp/qr_bounded.mojo); the caller's own read refuses it."""
    if n < 1 or m < n:
        raise Error("qr_op_r: needs m >= n >= 1 (a tall matrix), got " + String(m) + " x " + String(n))
    var nn = n * n
    if len(a) < m * n or len(r) < nn:
        raise Error("qr_op_r: a buffer is shorter than " + String(m) + " x " + String(n))
    var scratch = ctx.enqueue_create_buffer[DType.float32](qr_slice_count(m, n) * nn)
    var r_buf = ctx.enqueue_create_buffer[DType.float32](nn)
    _ = qr_factor_bounded(ctx, a, scratch, r_buf, m, n, QRB_CELLS, SO_QR_WAIT)
    ctx.enqueue_function[so_commit_kernel](
        _ip(gate), _fp(r_buf), _fp(r), Int32(nn), grid_dim=_so_blocks(nn), block_dim=SO_TPB
    )
    _ = scratch^
    _ = r_buf^


def qr_op_q(
    ctx: DeviceContext,
    mut gate: DeviceBuffer[DType.int32],
    mut a: DeviceBuffer[DType.float32],
    mut q: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    qc: Int,
) raises:
    """The first qc columns of Q (m x qc, row major) of the sliced Householder
    QR of A (m x n; CONSUMED: geqrf works in place), qc <= m, written only
    while the gate is open. Enqueued only (Apple: `_Pace`)."""
    if m < 1 or n < 1 or qc < 1 or qc > m:
        raise Error("qr_op_q: bad shape " + String(m) + " x " + String(n) + " with " + String(qc) + " columns")
    if len(a) < m * n or len(q) < m * qc:
        raise Error("qr_op_q: a buffer is shorter than its shape")
    var kk = m if m < n else n
    var tau = ctx.enqueue_create_buffer[DType.float32](kk)
    qs_geqrf_device(ctx, a, tau, m, n)
    var qb = ctx.enqueue_create_buffer[DType.float32](m * qc)
    qs_orgqr_device(ctx, a, tau, qb, m, n, kk, qc)
    ctx.enqueue_function[so_commit_kernel](
        _ip(gate), _fp(qb), _fp(q), Int32(m * qc), grid_dim=_so_blocks(m * qc), block_dim=SO_TPB
    )
    _ = tau^
    _ = qb^
