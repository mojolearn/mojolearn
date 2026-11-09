# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device eigh as executor step ops (lane n7-device-linalg-ops,
2026-11-08; docs/plans/EXECUTOR_N7_OPS.md). The host twins are in
x_decomp/eigh_ops_host.mojo.

Two ops, one algorithm each on every column (NVIDIA, AMD, Apple, the host):

  `eigh_op_dense`    a batch of dense symmetric k x k problems: the
                     round-robin Jacobi of x_decomp/rr.mojo, one block per
                     problem (`rr_batch_kernel`: the sweeps, the convergence
                     test, the sign flip and the ascending order all on the
                     device), after the power-of-two range scale
                     (x_decomp/eigh_scale.mojo). Host twin:
                     `host_eigh_rr_sorted`, problem by problem.
  `eigh_op_tridiag`  one symmetric tridiagonal T (diagonal d, off-diagonal
                     e; a Lanczos or Householder T): Sturm bisection in df64
                     and twisted-factorization vectors
                     (x_decomp/eigh_tridiag_cells.mojo `td_bisect_item`,
                     `td_gap_item`, `td_vec_item`), then the sign flip and
                     the ascending order. A refusal (a gap below
                     TD_GAP_MIN_REL max |lambda|, a nonfinite T or vector
                     norm) is decided on the device and falls back, on the
                     device, to the dense Jacobi of T: both are enqueued,
                     the Jacobi's blocks return at once unless the
                     tridiagonal solve refused, and one commit kernel takes
                     the answer that holds.

Neither op reads anything back or drains. Each writes the caller's w, v and
status only while the gate is open (word 0 of `gate`: the loop's done word,
or `step_gate_open_word` outside a loop); everything else it writes is its
own scratch, released in stream order. `status[0]` is the number of
problems that did not converge (0: every answer holds), so
`step_fail_joins_loop(ctx, registers.words, status, it)` stops a loop on an
unconverged solve, and outside a loop the family reads it at its own sync.

Outputs: w (batch x n) ascending; v (batch x n x n, row major) with
eigenvector c in COLUMN c, sign-flipped (the largest-magnitude entry of each
column positive; `host_sign_flip`).

BITS. The dense op is the Jacobi at every n: `DevExec.eigh` takes the
tridiagonal route from n = TD_MIN_N, so a family moving its dense solve onto
this op at n >= TD_MIN_N changes bits (every column with it; the host twin is
the same Jacobi). A family moving a host LAPACK/NumPy eigh of a Lanczos T
onto `eigh_op_tridiag` changes bits on every column together.
"""
from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import ftz
from core.device_zero import enqueue_fill
from core.step_gate import StepI32, step_fail_joins_loop
from decomposition.checks.jacobi_eigh_device import JACOBI_TOL
from decomposition.impl.linalg.detail.pca import SIGNFLIP_TPB, sign_flip_kernel
from decomposition.spectrum_order_device import enqueue_eigh_ascending
from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.eigh_scale import enqueue_es_scale, enqueue_es_unscale
from x_decomp.eigh_tridiag import TD_EIG_TPB, td_bisect_kernel, td_gap_kernel, td_vec_kernel
from x_decomp.rr import RR_EIGH_SWEEPS, RR_OFF_TPB
from x_decomp.rr_batch import rr_batch_gated_kernel, rrb_cs_len, rrb_part_len

comptime EO_TPB = 256


@always_inline
def _eo_blocks(count: Int) -> Int:
    return (count + EO_TPB - 1) // EO_TPB if count > 0 else 1


@always_inline
def _fp(buf: DeviceBuffer[DType.float32]) -> F32Ptr:
    return F32Ptr(unsafe_from_address=Int(buf.unsafe_ptr()))


@always_inline
def _ip(buf: DeviceBuffer[DType.int32]) -> I32Ptr:
    return I32Ptr(unsafe_from_address=Int(buf.unsafe_ptr()))


# ---------------------------------------------------------------- kernels


def eo_commit_kernel(gate: StepI32, src: F32Ptr, dst: F32Ptr, count_in: Int32):
    """dst[t] = src[t] while the gate is open (a bit copy)."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(count_in):
        return
    if gate.unsafe_load(0) != Int32(0):
        return
    dst.unsafe_store(t, src.unsafe_load(t))


def eo_status_zero_kernel(gate: StepI32, status: StepI32):
    """One thread: status[0] = 0 while the gate is open."""
    if Int(block_idx.x) != 0 or Int(thread_idx.x) != 0:
        return
    if gate.unsafe_load(0) != Int32(0):
        return
    status.unsafe_store(0, Int32(0))


def eo_status_count_kernel(gate: StepI32, marks: F32Ptr, status: StepI32, batch_in: Int32):
    """Thread b: problem b's mark (`rr_batch_kernel` info[2 b]: 1 converged,
    0 not, -1 its block never ran) adds one to status[0] unless it is 1.
    An integer count: the same word in any order."""
    var b = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if b >= Int(batch_in):
        return
    if gate.unsafe_load(0) != Int32(0):
        return
    if marks.unsafe_load(2 * b) != Float32(1.0):
        _ = Atomic.fetch_add(status, Int32(1))


def eo_ftz_copy_kernel(d: F32Ptr, e: F32Ptr, dd: F32Ptr, ee: F32Ptr, n_in: Int32):
    """The tridiagonal's words flushed (the denormal policy of
    x_decomp/eigh_tridiag_cells.mojo; a subnormal input must mean the same
    on a column that flushes on load): dd = ftz(d), ee = ftz(e) (n - 1
    off-diagonal words; ee[n - 1] = 0)."""
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    dd.unsafe_store(i, ftz(d.unsafe_load(i)))
    if i < n - 1:
        ee.unsafe_store(i, ftz(e.unsafe_load(i)))
    else:
        ee.unsafe_store(i, Float32(0.0))


def eo_tridiag_dense_kernel(dd: F32Ptr, ee: F32Ptr, t_out: F32Ptr, n_in: Int32):
    """Cell (i, j) of the dense T (the Jacobi fallback's input)."""
    var n = Int(n_in)
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if c >= n * n:
        return
    var i = c // n
    var j = c - i * n
    var v = Float32(0.0)
    if i == j:
        v = dd.unsafe_load(i)
    elif j == i + 1:
        v = ee.unsafe_load(i)
    elif i == j + 1:
        v = ee.unsafe_load(j)
    t_out.unsafe_store(c, v)


def eo_fallback_skip_kernel(gate: StepI32, td_info: F32Ptr, skip: StepI32):
    """One thread: the Jacobi fallback runs (skip = 0) only while the gate
    is open and the tridiagonal solve refused (td_info[2] != 0)."""
    if Int(block_idx.x) != 0 or Int(thread_idx.x) != 0:
        return
    var run = gate.unsafe_load(0) == Int32(0) and td_info.unsafe_load(2) != Float32(0.0)
    skip.unsafe_store(0, Int32(0) if run else Int32(1))


def eo_tridiag_commit_kernel(
    gate: StepI32, td_info: F32Ptr, jmarks: F32Ptr,
    w_td: F32Ptr, z_td: F32Ptr, w_j: F32Ptr, z_j: F32Ptr,
    w: F32Ptr, z: F32Ptr, status: StepI32, n_in: Int32,
):
    """Cell t of the answer while the gate is open: the tridiagonal solve's
    unless it refused, the Jacobi's otherwise; thread 0 writes status[0]
    (1 when the solve refused and the Jacobi did not converge). Loads go
    through branches, never through a pointer select (the Metal rule of
    autograd/ops.mojo `_ld_right`)."""
    var n = Int(n_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= n * n:
        return
    if gate.unsafe_load(0) != Int32(0):
        return
    var refused = td_info.unsafe_load(2) != Float32(0.0)
    if refused:
        z.unsafe_store(t, z_j.unsafe_load(t))
        if t < n:
            w.unsafe_store(t, w_j.unsafe_load(t))
    else:
        z.unsafe_store(t, z_td.unsafe_load(t))
        if t < n:
            w.unsafe_store(t, w_td.unsafe_load(t))
    if t == 0:
        var bad = refused and jmarks.unsafe_load(0) != Float32(1.0)
        status.unsafe_store(0, Int32(1) if bad else Int32(0))


# ---------------------------------------------------------------- the ops


def _eo_jacobi_enqueue(
    ctx: DeviceContext, mut skip: DeviceBuffer[DType.int32], mut a: DeviceBuffer[DType.float32], batch: Int, n: Int,
    mut bw: DeviceBuffer[DType.float32], mut bv: DeviceBuffer[DType.float32], mut marks: DeviceBuffer[DType.float32],
) raises:
    """The batched Jacobi into op scratch bw (batch x n), bv (batch x n x n)
    and marks (2 batch: converged, sweeps), its blocks skipped when
    skip[0] != 0. `a` is consumed (range-scaled in place)."""
    var dfac = enqueue_es_scale(ctx, _fp(a), batch, n)
    var dv = ctx.enqueue_create_buffer[DType.float32](batch * n * n)
    var dcs = ctx.enqueue_create_buffer[DType.float32](batch * rrb_cs_len(n))
    var dpart = ctx.enqueue_create_buffer[DType.float32](batch * rrb_part_len(n))
    var ddg = ctx.enqueue_create_buffer[DType.float32](batch * n)
    enqueue_fill(ctx, marks, Float32(-1.0))
    ctx.enqueue_function[rr_batch_gated_kernel](
        _ip(skip), _fp(a), _fp(dv), _fp(dcs), _fp(dpart), _fp(ddg), _fp(marks), _fp(bw), _fp(bv), Int32(n),
        Int32(RR_EIGH_SWEEPS), Float32(JACOBI_TOL), grid_dim=batch, block_dim=RR_OFF_TPB,
    )
    enqueue_es_unscale(ctx, _fp(bw), dfac, batch, n)
    _ = dv^
    _ = dcs^
    _ = dpart^
    _ = ddg^
    _ = dfac^


def eigh_op_dense(
    ctx: DeviceContext,
    mut gate: DeviceBuffer[DType.int32],
    mut a: DeviceBuffer[DType.float32],
    mut w: DeviceBuffer[DType.float32],
    mut v: DeviceBuffer[DType.float32],
    mut status: DeviceBuffer[DType.int32],
    batch: Int,
    n: Int,
) raises:
    """`batch` dense symmetric n x n problems stacked in `a` (row major,
    both triangles; CONSUMED: range-scaled and rotated in place) into w
    (batch x n, ascending) and v (batch x n x n, eigenvector c in column c),
    status[0] = problems not converged in RR_EIGH_SWEEPS sweeps. Enqueued
    only; the caller's buffers are written only while the gate is open."""
    if n < 1 or batch < 1:
        raise Error("eigh_op_dense: n and batch must be positive, got n=" + String(n) + " batch=" + String(batch))
    if len(a) < batch * n * n or len(w) < batch * n or len(v) < batch * n * n or len(status) < 1:
        raise Error("eigh_op_dense: a buffer is shorter than batch=" + String(batch) + " problems of n=" + String(n))
    var bw = ctx.enqueue_create_buffer[DType.float32](batch * n)
    var bv = ctx.enqueue_create_buffer[DType.float32](batch * n * n)
    var marks = ctx.enqueue_create_buffer[DType.float32](2 * batch)
    # the gate is the skip word: a done loop runs no sweep
    _eo_jacobi_enqueue(ctx, gate, a, batch, n, bw, bv, marks)
    var gp = _ip(gate)
    ctx.enqueue_function[eo_commit_kernel](
        gp, _fp(bw), _fp(w), Int32(batch * n), grid_dim=_eo_blocks(batch * n), block_dim=EO_TPB
    )
    ctx.enqueue_function[eo_commit_kernel](
        gp, _fp(bv), _fp(v), Int32(batch * n * n), grid_dim=_eo_blocks(batch * n * n), block_dim=EO_TPB
    )
    ctx.enqueue_function[eo_status_zero_kernel](  # small-launch(one word): the op's status word
        gp, _ip(status), grid_dim=(1, 1, 1), block_dim=(1, 1, 1)
    )
    ctx.enqueue_function[eo_status_count_kernel](
        gp, _fp(marks), _ip(status), Int32(batch), grid_dim=_eo_blocks(batch), block_dim=EO_TPB
    )
    _ = bw^
    _ = bv^
    _ = marks^


def eigh_op_tridiag(
    ctx: DeviceContext,
    mut gate: DeviceBuffer[DType.int32],
    mut d: DeviceBuffer[DType.float32],
    mut e: DeviceBuffer[DType.float32],
    mut w: DeviceBuffer[DType.float32],
    mut z: DeviceBuffer[DType.float32],
    mut status: DeviceBuffer[DType.int32],
    n: Int,
) raises:
    """The symmetric tridiagonal T with diagonal d (n) and off-diagonal e
    (e[i] = T[i, i + 1], n - 1 words read) into w (n, ascending) and z
    (n x n, eigenvector c in column c), status[0] = 1 when neither the
    tridiagonal solve nor its Jacobi fallback holds. d and e are only read.
    Enqueued only; w, z and status are written only while the gate is open."""
    if n < 1:
        raise Error("eigh_op_tridiag: n must be positive, got " + String(n))
    if len(d) < n or (n > 1 and len(e) < n - 1) or len(w) < n or len(z) < n * n or len(status) < 1:
        raise Error("eigh_op_tridiag: a buffer is shorter than n = " + String(n))
    var nn = n * n
    var dd = ctx.enqueue_create_buffer[DType.float32](n)
    var ee = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_function[eo_ftz_copy_kernel](
        _fp(d), _fp(e), _fp(dd), _fp(ee), Int32(n), grid_dim=_eo_blocks(n), block_dim=EO_TPB
    )
    # 1. the tridiagonal solve (td_info[0] = the scale, td_info[2] = refused)
    var td_info = ctx.enqueue_create_buffer[DType.float32](4)
    enqueue_fill(ctx, td_info, Float32(0.0))
    var wh = ctx.enqueue_create_buffer[DType.float32](n)
    var wl = ctx.enqueue_create_buffer[DType.float32](n)
    var wt = ctx.enqueue_create_buffer[DType.float32](n)
    var zt = ctx.enqueue_create_buffer[DType.float32](nn)
    var dph = ctx.enqueue_create_buffer[DType.float32](nn)
    var dpl = ctx.enqueue_create_buffer[DType.float32](nn)
    var dmh = ctx.enqueue_create_buffer[DType.float32](nn)
    var dml = ctx.enqueue_create_buffer[DType.float32](nn)
    var eg = (n + TD_EIG_TPB - 1) // TD_EIG_TPB
    ctx.enqueue_function[td_bisect_kernel](
        dd.unsafe_ptr(), ee.unsafe_ptr(), td_info.unsafe_ptr(), wh.unsafe_ptr(), wl.unsafe_ptr(), Int32(n),
        grid_dim=eg, block_dim=TD_EIG_TPB,
    )
    ctx.enqueue_function[td_gap_kernel](
        wh.unsafe_ptr(), wl.unsafe_ptr(), td_info.unsafe_ptr(), wt.unsafe_ptr(), Int32(n),
        grid_dim=eg, block_dim=TD_EIG_TPB,
    )
    ctx.enqueue_function[td_vec_kernel](
        dd.unsafe_ptr(), ee.unsafe_ptr(), td_info.unsafe_ptr(), wh.unsafe_ptr(), wl.unsafe_ptr(),
        dph.unsafe_ptr(), dpl.unsafe_ptr(), dmh.unsafe_ptr(), dml.unsafe_ptr(), zt.unsafe_ptr(), Int32(n),
        grid_dim=eg, block_dim=TD_EIG_TPB,
    )
    _ = dph^
    _ = dpl^
    _ = dmh^
    _ = dml^
    _ = wh^
    _ = wl^
    # the tail every eigh route shares: the sign flip, the ascending order
    ctx.enqueue_function[sign_flip_kernel](
        zt.unsafe_ptr(), Int32(n), grid_dim=(n, 1, 1), block_dim=(SIGNFLIP_TPB, 1, 1)
    )
    var w_td = ctx.enqueue_create_buffer[DType.float32](n)
    var z_td = ctx.enqueue_create_buffer[DType.float32](nn)
    var pos = ctx.enqueue_create_buffer[DType.int32](n)
    enqueue_eigh_ascending(ctx, _fp(wt), 1, _fp(zt), n, pos, _fp(w_td), _fp(z_td))
    _ = wt^
    _ = zt^
    _ = pos^
    # 2. the Jacobi of the dense T, every block skipped unless refused
    var skip = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_function[eo_fallback_skip_kernel](  # small-launch(one word): the fallback's gate
        _ip(gate), _fp(td_info), _ip(skip), grid_dim=(1, 1, 1), block_dim=(1, 1, 1)
    )
    var tm = ctx.enqueue_create_buffer[DType.float32](nn)
    ctx.enqueue_function[eo_tridiag_dense_kernel](
        _fp(dd), _fp(ee), _fp(tm), Int32(n), grid_dim=_eo_blocks(nn), block_dim=EO_TPB
    )
    var jw = ctx.enqueue_create_buffer[DType.float32](n)
    var jv = ctx.enqueue_create_buffer[DType.float32](nn)
    var jmarks = ctx.enqueue_create_buffer[DType.float32](2)
    _eo_jacobi_enqueue(ctx, skip, tm, 1, n, jw, jv, jmarks)
    # 3. the one commit
    ctx.enqueue_function[eo_tridiag_commit_kernel](
        _ip(gate), _fp(td_info), _fp(jmarks), _fp(w_td), _fp(z_td), _fp(jw), _fp(jv),
        _fp(w), _fp(z), _ip(status), Int32(n), grid_dim=_eo_blocks(nn), block_dim=EO_TPB,
    )
    _ = dd^
    _ = ee^
    _ = td_info^
    _ = w_td^
    _ = z_td^
    _ = skip^
    _ = tm^
    _ = jw^
    _ = jv^
    _ = jmarks^


def eigh_op_status_joins_loop(
    ctx: DeviceContext, mut words: DeviceBuffer[DType.int32], mut status: DeviceBuffer[DType.int32], it: Int,
) raises:
    """An unconverged eigh (status[0] > 0) stops the loop at iteration `it`
    (`LoopRegisters.failure()` holds the count). Enqueue it right after the
    op."""
    step_fail_joins_loop(ctx, words, status, it)
