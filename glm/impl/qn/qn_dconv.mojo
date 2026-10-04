# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-linsvr, QN_FAST_DCONV (FAST + Apple default, part of QN_LSVR_ALL, n_features <= QN_ALL_MAX_D only; -D MOJOLEARN_LSVR_ALL_OFF reverts):
L-BFGS iterations whose line-search decision and convergence test run on
the device, the host reading ONE state block every QN_DCONV_POLL iterations.

Compiled and called under FAST on Apple only (`qn_solvers.min_lbfgs` calls
`dconv_run` inside `comptime if QN_FAST_DCONV`); IDENTICAL never references
this module's functions, so it compiles main's code unchanged.

WHY A DEVICE LOOP AND NOT A DEVICE FLAG. The host loop already reads the
loss, the regularizer, the gradient norm and dg_init behind ONE synchronize
per objective evaluation; the convergence test is host arithmetic on those
words. A device flag alone removes nothing: the synchronize exists because
the HOST decides the line search. So the whole decision moves: one device
iteration is, all enqueued with no synchronize,

    dc_dir_kernel      `lbfgs_dir_kernel` (S/Y update, xp/gradp saves, the
                       two-loop direction, dg_init into scalar[0]) with
                       `end` / `n_vec` read from and advanced in the state
                       block instead of passed from the host
    dc_axpy_kernel     x = xp + 1 * drt                     (candidate 0)
    fused batch pass   objective + gradient at x, objectives of the next
                       QNF_LS_K - 1 backtracking steps (gated `qnf_*`)
    dc_ls_kernel       `ls_backtrack_batched`'s walk (Armijo, min/max step,
                       the max_linesearch budget) on those words
    dc_axpy_kernel     x = xp + step_c * drt   only if candidate c >= 1 won
    fused pass         objective + gradient at that x     (same condition)
    dc_check_kernel    `update_and_check` + `check_convergence` (gradient
                       norm vs epsilon * max(fx, epsilon), the `past`-deep
                       objective-change test), max_iterations, k += 1

THE FREEZE. State word 0 is the stop flag: 1 = stopped (retcode in word 1),
2 = handed off. Every kernel above returns at its first line when it is set
(`glm_base._dc_skip`), so the iterations enqueued after the stopping one are
no-ops: x, grad, fx and k are frozen at the iterate that stopped, and the
answer is the one the host loop would have returned at that iteration.

THE HANDOFF. Anything the device does not decide (a positive dg_init, all
QNF_LS_K candidates rejected, a min/max step stop, the budget) sets word 0
to 2 BEFORE the iteration writes anything the host loop's iteration-start
state depends on (xp, gradp, drt, S, Y, the device yhist, scalar[0..3] are
the direction kernel's and are left alone). The host then reruns that same
iteration with its own line search (`saved`, `dg_ready` true, the direction
resolved) and its exact outcome, then re-enters the device loop.

THE POLL, k = QN_DCONV_POLL = 8. One state readback per 8 iterations instead
of one per iteration; at most 7 gated no-op iterations follow the stop
(about 70 launches that return at their first line, no row work).

FAST on Apple promises no bits: the device Armijo product `step * dg_test +
fx_init` is `identical_mul_add` in device code (FAST: a plain multiply-add
the GPU compiler may contract), the host's may round differently, so a
candidate on the Armijo boundary can be decided the other way; the
convergence tolerance is the solver's own.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from checks.numerics import ftz, identical_mul_add
from core.column_stats import STATS_TPB
from glm.impl.qn.glm_base import GLMWithData, QN_FAST_DCONV, QN_IDN_DCONV, QNF_LS_K, _qnb_barrier
from std.sys.compile import is_defined
from glm.impl.qn.qn_util import (
    LBFGS_FUSED_MAX_M,
    LBFGS_LS_BT_ARMIJO,
    LBFGSParam,
    OPT_MAX_ITERS_REACHED,
    OPT_NUMERIC_ERROR,
    OPT_SUCCESS,
    _block_dot_bcast,
    _dev_barrier,
    _two_loop,
    host_le_eps_times,
)
from glm.impl.qn.simple_mat.dense import VEC_ELEM_TPB

#: iterations enqueued per state readback
comptime QN_DCONV_POLL = 8

# the state block, Float32 words (integers below 2^24 are exact)
comptime DC_STOP = 0  # 0 running, 1 stopped, 2 handed off to the host
comptime DC_RET = 1  # OPT_RETCODE when stopped
comptime DC_K = 2  # the iteration being run (host `k`)
comptime DC_FX = 3  # objective at the current iterate
comptime DC_FXP = 4  # objective at the iteration's start (for a restore)
comptime DC_END = 5  # L-BFGS ring position
comptime DC_NVEC = 6  # L-BFGS pairs stored
comptime DC_MAT = 7  # 1: the materialize pass must run
comptime DC_MSTEP = 8  # the step it materializes
comptime DC_LSIT = 9  # line-search candidates evaluated (last iteration)
comptime DC_GNORM = 10  # gradient norm at the current iterate
comptime DC_HIST = 16  # fx_hist[0 .. past)
comptime DC_MAX_PAST = 16
comptime DC_WORDS = DC_HIST + DC_MAX_PAST


@always_inline
def _f2i(v: Float32) -> Int:
    return Int(v.cast[DType.int32]())


@always_inline
def _finite(v: Float32) -> Bool:
    return (bitcast[DType.uint32](v) & UInt32(0x7FFFFFFF)) < UInt32(0x7F800000)


def dc_dir_kernel(
    st: MutPointer[Float32, MutAnyOrigin],
    drt: MutPointer[Float32, MutAnyOrigin],
    s_all: MutPointer[Float32, MutAnyOrigin],
    y_all: MutPointer[Float32, MutAnyOrigin],
    yhist: MutPointer[Float32, MutAnyOrigin],
    scalar: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    xp: MutPointer[Float32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    gradp: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    m_in: Int32,
):
    """`qn_util.lbfgs_dir_kernel` (use_pseudo 0, do_dot 1, neg_one -1),
    ONE block of STATS_TPB over the n_param words, with `end` and `n_vec`
    from the state block; thread 0 then does `lbfgs_search_dir_resolve`'s
    bookkeeping there (on a kept pair n_vec += 1, end advances). The verdict
    words still go to scalar[2..3] and dg_init to scalar[0], where the host
    line search reads them after a handoff. Every thread reads the state
    before the first barrier; thread 0 writes it after the last."""
    if st.unsafe_load(DC_STOP) != Float32(0.0):
        return
    var n = Int(n_in)
    var m = Int(m_in)
    var tid = Int(thread_idx.x)
    var end_prev = _f2i(st.unsafe_load(DC_END))
    var n_vec = _f2i(st.unsafe_load(DC_NVEC))
    var neg_one = Float32(-1.0)
    var sw = s_all + end_prev * n
    var yw = y_all + end_prev * n
    var i0 = tid
    while i0 < n:
        var xi = x.unsafe_load(i0)
        var gi = grad.unsafe_load(i0)
        sw.unsafe_store(i0, ftz(identical_mul_add(neg_one, xp.unsafe_load(i0), xi)))
        yw.unsafe_store(i0, ftz(identical_mul_add(neg_one, gradp.unsafe_load(i0), gi)))
        xp.unsafe_store(i0, xi)
        gradp.unsafe_store(i0, gi)
        i0 += STATS_TPB
    _dev_barrier()
    var ys = _block_dot_bcast(s_all + end_prev * n, y_all + end_prev * n, n, tid)
    var yv = y_all + end_prev * n
    var yy = _block_dot_bcast(yv, yv, n, tid)
    var verdict = scalar + 2
    var skipped = host_le_eps_times(ys, yy)
    if skipped:
        if tid == 0:
            verdict.unsafe_store(0, Float32(1.0))
            verdict.unsafe_store(1, ys)
    else:
        _two_loop(drt, grad, s_all, y_all, yhist, verdict, n, m, tid,
                  end_prev, n_vec, neg_one, ys, yy)
    var dg = _block_dot_bcast(grad, drt, n, tid)
    if tid == 0:
        scalar.unsafe_store(0, dg)
        if not skipped:
            st.unsafe_store(DC_NVEC, Float32(n_vec + 1))
            st.unsafe_store(DC_END, Float32((end_prev + 1) % m))


def dc_axpy_kernel(
    st: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    xp: MutPointer[Float32, MutAnyOrigin],
    drt: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    mat_in: Int32,
):
    """`axpy(x, step, drt, xp)` (`dense.axpy_kernel`'s expression), one
    thread per word: step 1 for candidate 0 (`mat_in` 0), or, `mat_in` 1,
    the step the device line search chose, only when it chose one past the
    first (`DC_MAT`)."""
    if st.unsafe_load(DC_STOP) != Float32(0.0):
        return
    var a = Float32(1.0)
    if Int(mat_in) != 0:
        if st.unsafe_load(DC_MAT) == Float32(0.0):
            return
        a = st.unsafe_load(DC_MSTEP)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        x.unsafe_store(i, ftz(identical_mul_add(a, drt.unsafe_load(i), xp.unsafe_load(i))))


@always_inline
def _dc_objective(
    slots: MutPointer[Float32, MutAnyOrigin], l2nz: Int32
) -> Float32:
    """`evaluate`'s host value: the loss, plus the regularizer when l2 != 0."""
    var loss = slots.unsafe_load(0)
    if Int(l2nz) == 0:
        return loss
    return ftz(loss + slots.unsafe_load(1))


@always_inline
def _dc_gnorm(slots: MutPointer[Float32, MutAnyOrigin], half: Int32) -> Float32:
    """`grad_norm`'s value from the speculative reduction (kind 1: * 0.5)."""
    var g = slots.unsafe_load(2)
    if Int(half) != 0:
        return g * Float32(0.5)
    return g


def dc_ls_kernel(
    st: MutPointer[Float32, MutAnyOrigin],
    slots: MutPointer[Float32, MutAnyOrigin],
    scalar: MutPointer[Float32, MutAnyOrigin],
    l2nz: Int32,
    gnorm_half: Int32,
    ftol: Float32,
    min_step: Float32,
    max_step: Float32,
    ls_dec: Float32,
    max_ls: Int32,
):
    """`qn_linesearch.ls_backtrack_batched`'s walk over the QNF_LS_K priced
    candidates, on the device (thread 0; a constant-size walk, no data
    loop). Candidate 0 accepted: its objective and gradient norm go to the
    state (x, grad are already its). Candidate c >= 1 accepted: DC_MAT and
    its step, for the materialize pass. Anything else (dg_init > 0, every
    candidate rejected, a min/max step stop, the budget): handoff, before
    any state word the host iteration-start depends on is written."""
    if Int(thread_idx.x) != 0:
        return
    if st.unsafe_load(DC_STOP) != Float32(0.0):
        return
    var fx_init = st.unsafe_load(DC_FX)
    var dg_init = scalar.unsafe_load(0)
    if dg_init > Float32(0.0):
        st.unsafe_store(DC_STOP, Float32(2.0))
        return
    var dg_test = ftol * dg_init
    var fx0 = _dc_objective(slots, l2nz)
    var step = Float32(1.0)
    var step_last = step
    var lsit = 0
    var last = 0
    var ok = False
    for c in range(QNF_LS_K):
        if lsit >= Int(max_ls):
            break
        var fxc = fx0 if c == 0 else slots.unsafe_load(4 + c - 1)
        lsit += 1
        last = c
        step_last = step
        if not (fxc > identical_mul_add(step, dg_test, fx_init)):
            ok = True
            break
        if step < min_step or step > max_step:
            break
        step *= ls_dec
    if not ok:
        st.unsafe_store(DC_STOP, Float32(2.0))
        return
    st.unsafe_store(DC_FXP, fx_init)
    st.unsafe_store(DC_LSIT, Float32(lsit))
    if last == 0:
        st.unsafe_store(DC_FX, fx0)
        st.unsafe_store(DC_GNORM, _dc_gnorm(slots, gnorm_half))
        st.unsafe_store(DC_MAT, Float32(0.0))
    else:
        st.unsafe_store(DC_MAT, Float32(1.0))
        st.unsafe_store(DC_MSTEP, step_last)


def dc_check_kernel(
    st: MutPointer[Float32, MutAnyOrigin],
    slots: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    xp: MutPointer[Float32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    gradp: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    l2nz: Int32,
    gnorm_half: Int32,
    epsilon: Float32,
    past_in: Int32,
    delta: Float32,
    max_iter: Int32,
):
    """`qn_solvers.update_and_check` for a successful line search plus
    `qn_util.check_convergence`, ONE block of STATS_TPB over the n_param
    words (the restore copy). Every thread reads the state and takes the
    same decision; a device-memory barrier (`_qnb_barrier`) orders every
    read before thread 0's writes. Not valid (NaN / inf objective):
    OPT_NUMERIC_ERROR and x, grad, fx restored to the iteration's start.
    Converged: OPT_SUCCESS, k unchanged. Else k += 1, and past
    max_iterations OPT_MAX_ITERS_REACHED (the host loop's exit leaves k at
    max_iterations + 1 too)."""
    if st.unsafe_load(DC_STOP) != Float32(0.0):
        return
    var tid = Int(thread_idx.x)
    var n = Int(n_in)
    var past = Int(past_in)
    var k = _f2i(st.unsafe_load(DC_K))
    var fx = st.unsafe_load(DC_FX)
    var gnorm = st.unsafe_load(DC_GNORM)
    if st.unsafe_load(DC_MAT) != Float32(0.0):
        fx = _dc_objective(slots, l2nz)
        gnorm = _dc_gnorm(slots, gnorm_half)
    var fxp = st.unsafe_load(DC_FXP)
    var valid = _finite(fx)
    var conv = False
    var write_hist = False
    var hidx = 0
    if valid:
        var fmag = max(fx, epsilon)
        if gnorm <= epsilon * fmag:
            conv = True
        elif past > 0:
            hidx = k % past
            if k >= past and abs(st.unsafe_load(DC_HIST + hidx) - fx) <= delta * fmag:
                conv = True
            else:
                write_hist = True
    var stop = Float32(0.0)
    var ret = OPT_MAX_ITERS_REACHED
    var k_out = k + 1
    if not valid:
        stop = Float32(1.0)
        ret = OPT_NUMERIC_ERROR
        k_out = k
        var i = tid
        while i < n:
            x.unsafe_store(i, xp.unsafe_load(i))
            grad.unsafe_store(i, gradp.unsafe_load(i))
            i += STATS_TPB
        fx = fxp
    elif conv:
        stop = Float32(1.0)
        ret = OPT_SUCCESS
        k_out = k
    elif k + 1 > Int(max_iter):
        stop = Float32(1.0)
        ret = OPT_MAX_ITERS_REACHED
    _qnb_barrier()
    if tid == 0:
        if write_hist:
            st.unsafe_store(DC_HIST + hidx, fx)
        st.unsafe_store(DC_FX, fx)
        st.unsafe_store(DC_GNORM, gnorm)
        st.unsafe_store(DC_K, Float32(k_out))
        st.unsafe_store(DC_RET, Float32(ret))
        st.unsafe_store(DC_MAT, Float32(0.0))
        st.unsafe_store(DC_STOP, stop)


def dconv_applies(param: LBFGSParam, mut f: GLMWithData, tracing: Bool) -> Bool:
    """The device loop serves this fit: FAST on Apple under the define, no
    identity trace (the card records every iteration), the fused objective
    (C == 1, d <= QNF_MAX_D, n_param <= STATS_TPB), the Armijo search, a
    `past` that fits the state block. Else the host loop runs as before."""
    comptime if not QN_FAST_DCONV:
        return False
    else:
        return (
            (not tracing)
            and f.dconv_applies()
            and param.linesearch == LBFGS_LS_BT_ARMIJO
            and param.past <= DC_MAX_PAST
            and param.m <= LBFGS_FUSED_MAX_M
        )


def dconv_run(
    ctx: DeviceContext,
    param: LBFGSParam,
    mut f: GLMWithData,
    mut x: DeviceBuffer[DType.float32],
    mut xp: DeviceBuffer[DType.float32],
    mut grad: DeviceBuffer[DType.float32],
    mut gradp: DeviceBuffer[DType.float32],
    mut drt: DeviceBuffer[DType.float32],
    mut s_all: DeviceBuffer[DType.float32],
    mut y_all: DeviceBuffer[DType.float32],
    mut hist: DeviceBuffer[DType.float32],
    mut scalar: DeviceBuffer[DType.float32],
    n: Int,
    mut k: Int,
    mut fx: Float32,
    mut end: Int,
    mut n_vec: Int,
    mut fx_hist: List[Float32],
    mut retcode: Int,
) raises -> Int:
    """Called by `min_lbfgs` where the host would enqueue the next search
    direction (iteration `k` passed its `update_and_check` without a stop).
    Uploads the state (iteration k + 1, fx, end, n_vec, fx_hist), enqueues
    QN_DCONV_POLL device iterations at a time and reads the state block
    back once per batch. Returns 1 when the device stopped (`k`, `fx`,
    `retcode` set: x holds the answer) or 2 on a handoff (`k`, `fx`, `end`,
    `n_vec`, `fx_hist` set to the start of the iteration the host must
    rerun with `saved` and `dg_ready` true and the direction resolved)."""
    comptime if not QN_FAST_DCONV:
        raise Error("qn: dconv_run is compiled under FAST + Apple (QN_FAST_DCONV) only")
    else:
        var dst = ctx.enqueue_create_buffer[DType.float32](DC_WORDS)
        var hs = ctx.enqueue_create_host_buffer[DType.float32](DC_WORDS)
        ctx.synchronize()
        var p = hs.unsafe_ptr()
        for i in range(DC_WORDS):
            p.unsafe_store(i, Float32(0.0))
        p.unsafe_store(DC_K, Float32(k + 1))
        p.unsafe_store(DC_FX, fx)
        p.unsafe_store(DC_END, Float32(end))
        p.unsafe_store(DC_NVEC, Float32(n_vec))
        for i in range(len(fx_hist)):
            p.unsafe_store(DC_HIST + i, fx_hist[i])
        ctx.enqueue_copy(dst_buf=dst, src_ptr=hs.unsafe_ptr())
        var gate = dst.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var l2nz = Int32(0) if f.l2 == Float32(0.0) else Int32(1)
        var half = Int32(1) if f._gnorm_kind() == 1 else Int32(0)
        var vgrid = (n + VEC_ELEM_TPB - 1) // VEC_ELEM_TPB
        var max_polls = param.max_iterations // QN_DCONV_POLL + 2
        var polls = 0
        var outcome = 0
        while outcome == 0:
            for _ in range(QN_DCONV_POLL):
                ctx.enqueue_function[dc_dir_kernel](  # small-launch(n: L-BFGS coefficients): n is the weight vector (d * C + bias, d <= QNF_MAX_D under dconv_applies), never rows; the two-loop over m history pairs, threads across n
                    dst.unsafe_ptr(), drt.unsafe_ptr(), s_all.unsafe_ptr(),
                    y_all.unsafe_ptr(), hist.unsafe_ptr(), scalar.unsafe_ptr(),
                    x.unsafe_ptr(), xp.unsafe_ptr(), grad.unsafe_ptr(),
                    gradp.unsafe_ptr(), Int32(n), Int32(param.m),
                    grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
                )
                ctx.enqueue_function[dc_axpy_kernel](
                    dst.unsafe_ptr(), x.unsafe_ptr(), xp.unsafe_ptr(),
                    drt.unsafe_ptr(), Int32(n), Int32(0),
                    grid_dim=(vgrid, 1, 1), block_dim=(VEC_ELEM_TPB, 1, 1),
                )
                f.enqueue_dconv_eval(ctx, x, grad, xp, drt, gate, -1, param.ls_dec, True)
                ctx.enqueue_function[dc_ls_kernel](
                    dst.unsafe_ptr(), f.slots.unsafe_ptr(), scalar.unsafe_ptr(),
                    l2nz, half, param.ftol, param.min_step, param.max_step,
                    param.ls_dec, Int32(param.max_linesearch),
                    grid_dim=(1, 1, 1), block_dim=(32, 1, 1),
                )
                ctx.enqueue_function[dc_axpy_kernel](
                    dst.unsafe_ptr(), x.unsafe_ptr(), xp.unsafe_ptr(),
                    drt.unsafe_ptr(), Int32(n), Int32(1),
                    grid_dim=(vgrid, 1, 1), block_dim=(VEC_ELEM_TPB, 1, 1),
                )
                f.enqueue_dconv_eval(ctx, x, grad, xp, drt, gate, DC_MAT, param.ls_dec, False)
                ctx.enqueue_function[dc_check_kernel](  # small-launch(n: L-BFGS coefficients): n is the weight vector (d * C + bias, d <= QNF_MAX_D under dconv_applies), never rows; norms over n by the block
                    dst.unsafe_ptr(), f.slots.unsafe_ptr(), x.unsafe_ptr(),
                    xp.unsafe_ptr(), grad.unsafe_ptr(), gradp.unsafe_ptr(),
                    Int32(n), l2nz, half, param.epsilon, Int32(param.past),
                    param.delta, Int32(param.max_iterations),
                    grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
                )
            ctx.enqueue_copy(dst_ptr=hs.unsafe_ptr(), src_buf=dst)
            ctx.synchronize()
            var stop = p.unsafe_load(DC_STOP)
            if stop == Float32(1.0):
                outcome = 1
            elif stop == Float32(2.0):
                outcome = 2
            else:
                polls += 1
                if polls > max_polls:
                    raise Error("qn: the device L-BFGS loop did not stop (QN_FAST_DCONV)")
        k = _f2i(p.unsafe_load(DC_K))
        fx = p.unsafe_load(DC_FX)
        if outcome == 1:
            retcode = _f2i(p.unsafe_load(DC_RET))
        else:
            end = _f2i(p.unsafe_load(DC_END))
            n_vec = _f2i(p.unsafe_load(DC_NVEC))
            for i in range(len(fx_hist)):  # small-loop(fx_hist: objective history window): the L-BFGS stopping window, a handful of past values
                fx_hist[i] = p.unsafe_load(DC_HIST + i)
        _ = len(dst)
        _ = hs.unsafe_ptr()
        return outcome


# ===========================================================================
# lane fam2-linear (2026-10-04): QN_IDN_DCONV, the device loop in IDENTICAL
# arithmetic. CANDIDATE ARM, default OFF (`-D MOJOLEARN_QN_IDN_DCONV`).
#
# One device iteration, all enqueued, no synchronize:
#
#     dc_dir_kernel       as above (`lbfgs_dir_kernel`'s words, the state's
#                         end / n_vec)
#     dc_axpy_kernel      x = xp + 1 * drt
#     gated evaluation    `GLMWithData.enqueue_idn_dconv_eval`: the fused row
#                         pass, the tile fold, the one-launch epilogue, the
#                         IDENTICAL chains the host-driven evaluate runs
#     dc_ls1_kernel       the step-1 Armijo test, the host's expression
#     dc_check_kernel     as above
#
# THE BIT ARGUMENT. Every word the device decides on is the word the host
# loop would hold: fx is `ftz(loss + reg)` of the same slots, gnorm the same
# reduction, dg_init the direction kernel's scalar[0], and the tests are the
# host's Float32 expressions (`ftol * dg_init`, `identical_mul_add(1,
# dg_test, fx_init)`, `gnorm <= epsilon * max(fx, epsilon)`, the `past`
# test). The device accepts ONLY a clean step-1 Armijo success on finite,
# non-tiny operands; anything else (a rejected step, dg_init > 0, a
# non-finite or tiny operand, where a target that flushes compare operands
# could decide differently) is a HANDOFF: the host reruns that iteration
# from xp / gradp with its own line search, exactly as it would have. So
# the iterate sequence, n_iter and the retcode are the host loop's.
# Launches enqueued after the stop are no-ops (`_dc_skip`).
#
# Poll interval candidates: default 4; `-D MOJOLEARN_QN_IDN_DCONV_POLL_2`,
# `-D MOJOLEARN_QN_IDN_DCONV_POLL_8`.
# ===========================================================================

comptime QN_IDN_DCONV_POLL = (
    2 if is_defined["MOJOLEARN_QN_IDN_DCONV_POLL_2"]()
    else (8 if is_defined["MOJOLEARN_QN_IDN_DCONV_POLL_8"]() else 4)
)
#: magnitudes below 2^-100 (bits) hand off: no product or compare on the
#: device then touches a subnormal
comptime DC_TINY_BITS = 0x0D800000


@always_inline
def _dc_tiny_or_bad(v: Float32) -> Bool:
    """Non-finite, or nonzero with magnitude below 2^-100 (by bits)."""
    var mag = bitcast[DType.uint32](v) & UInt32(0x7FFFFFFF)
    if mag >= UInt32(0x7F800000):
        return True
    return mag != UInt32(0) and mag < UInt32(DC_TINY_BITS)


def dc_ls1_kernel(
    st: MutPointer[Float32, MutAnyOrigin],
    slots: MutPointer[Float32, MutAnyOrigin],
    scalar: MutPointer[Float32, MutAnyOrigin],
    l2nz: Int32,
    gnorm_half: Int32,
    ftol: Float32,
    max_ls: Int32,
):
    """`qn_linesearch.ls_backtrack`'s first candidate (step 1, Armijo) on
    the device, thread 0. Accepted: its objective and gradient norm go to
    the state (x, grad are already its). Anything else hands off before any
    state word the host iteration-start depends on is written."""
    if Int(thread_idx.x) != 0:
        return
    if st.unsafe_load(DC_STOP) != Float32(0.0):
        return
    var fx_init = st.unsafe_load(DC_FX)
    var dg_init = scalar.unsafe_load(0)
    var bdg = bitcast[DType.uint32](dg_init)
    var dg_pos = (bdg >> 31) == UInt32(0) and (bdg & UInt32(0x7FFFFFFF)) != UInt32(0)
    if dg_pos or _dc_tiny_or_bad(dg_init) or _dc_tiny_or_bad(fx_init) or Int(max_ls) < 1:
        st.unsafe_store(DC_STOP, Float32(2.0))
        return
    var dg_test = ftol * dg_init
    var rhs = identical_mul_add(Float32(1.0), dg_test, fx_init)
    var fx0 = _dc_objective(slots, l2nz)
    if _dc_tiny_or_bad(dg_test) or _dc_tiny_or_bad(rhs) or _dc_tiny_or_bad(fx0):
        st.unsafe_store(DC_STOP, Float32(2.0))
        return
    if fx0 > rhs:
        st.unsafe_store(DC_STOP, Float32(2.0))
        return
    st.unsafe_store(DC_FXP, fx_init)
    st.unsafe_store(DC_LSIT, Float32(1.0))
    st.unsafe_store(DC_FX, fx0)
    st.unsafe_store(DC_GNORM, _dc_gnorm(slots, gnorm_half))
    st.unsafe_store(DC_MAT, Float32(0.0))


def dconv_idn_applies(param: LBFGSParam, mut f: GLMWithData, tracing: Bool) -> Bool:
    """The IDENTICAL device loop serves this fit: the define, no identity
    trace (the card records every iteration), the tiled `C == 1` objective,
    the Armijo search, a `past` that fits the state block."""
    comptime if not QN_IDN_DCONV:
        return False
    else:
        return (
            (not tracing)
            and f.idn_dconv_applies()
            and param.linesearch == LBFGS_LS_BT_ARMIJO
            and param.past <= DC_MAX_PAST
            and param.m <= LBFGS_FUSED_MAX_M
        )


def dconv_idn_run(
    ctx: DeviceContext,
    param: LBFGSParam,
    mut f: GLMWithData,
    mut x: DeviceBuffer[DType.float32],
    mut xp: DeviceBuffer[DType.float32],
    mut grad: DeviceBuffer[DType.float32],
    mut gradp: DeviceBuffer[DType.float32],
    mut drt: DeviceBuffer[DType.float32],
    mut s_all: DeviceBuffer[DType.float32],
    mut y_all: DeviceBuffer[DType.float32],
    mut hist: DeviceBuffer[DType.float32],
    mut scalar: DeviceBuffer[DType.float32],
    n: Int,
    mut k: Int,
    mut fx: Float32,
    mut end: Int,
    mut n_vec: Int,
    mut fx_hist: List[Float32],
    mut retcode: Int,
) raises -> Int:
    """`dconv_run`'s contract (1 stopped, 2 handed off) for the IDENTICAL
    device iteration above."""
    comptime if not QN_IDN_DCONV:
        raise Error("qn: dconv_idn_run is compiled under -D MOJOLEARN_QN_IDN_DCONV only")
    else:
        var dst = ctx.enqueue_create_buffer[DType.float32](DC_WORDS)
        var hs = ctx.enqueue_create_host_buffer[DType.float32](DC_WORDS)
        ctx.synchronize()
        var p = hs.unsafe_ptr()
        for i in range(DC_WORDS):
            p.unsafe_store(i, Float32(0.0))
        p.unsafe_store(DC_K, Float32(k + 1))
        p.unsafe_store(DC_FX, fx)
        p.unsafe_store(DC_END, Float32(end))
        p.unsafe_store(DC_NVEC, Float32(n_vec))
        for i in range(len(fx_hist)):
            p.unsafe_store(DC_HIST + i, fx_hist[i])
        ctx.enqueue_copy(dst_buf=dst, src_ptr=hs.unsafe_ptr())
        var gate = dst.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var l2nz = Int32(0) if f.l2 == Float32(0.0) else Int32(1)
        var half = Int32(1) if f._gnorm_kind() == 1 else Int32(0)
        var vgrid = (n + VEC_ELEM_TPB - 1) // VEC_ELEM_TPB
        var max_polls = param.max_iterations // QN_IDN_DCONV_POLL + 2
        var polls = 0
        var outcome = 0
        while outcome == 0:
            for _ in range(QN_IDN_DCONV_POLL):
                ctx.enqueue_function[dc_dir_kernel](  # small-launch(n: L-BFGS coefficients): n is the weight vector, never rows
                    dst.unsafe_ptr(), drt.unsafe_ptr(), s_all.unsafe_ptr(),
                    y_all.unsafe_ptr(), hist.unsafe_ptr(), scalar.unsafe_ptr(),
                    x.unsafe_ptr(), xp.unsafe_ptr(), grad.unsafe_ptr(),
                    gradp.unsafe_ptr(), Int32(n), Int32(param.m),
                    grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
                )
                ctx.enqueue_function[dc_axpy_kernel](
                    dst.unsafe_ptr(), x.unsafe_ptr(), xp.unsafe_ptr(),
                    drt.unsafe_ptr(), Int32(n), Int32(0),
                    grid_dim=(vgrid, 1, 1), block_dim=(VEC_ELEM_TPB, 1, 1),
                )
                f.enqueue_idn_dconv_eval(ctx, x, grad, gate)
                ctx.enqueue_function[dc_ls1_kernel](
                    dst.unsafe_ptr(), f.slots.unsafe_ptr(), scalar.unsafe_ptr(),
                    l2nz, half, param.ftol, Int32(param.max_linesearch),
                    grid_dim=(1, 1, 1), block_dim=(32, 1, 1),
                )
                ctx.enqueue_function[dc_check_kernel](  # small-launch(n: L-BFGS coefficients): n is the weight vector, never rows
                    dst.unsafe_ptr(), f.slots.unsafe_ptr(), x.unsafe_ptr(),
                    xp.unsafe_ptr(), grad.unsafe_ptr(), gradp.unsafe_ptr(),
                    Int32(n), l2nz, half, param.epsilon, Int32(param.past),
                    param.delta, Int32(param.max_iterations),
                    grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
                )
            ctx.enqueue_copy(dst_ptr=hs.unsafe_ptr(), src_buf=dst)
            ctx.synchronize()
            var stop = p.unsafe_load(DC_STOP)
            if stop == Float32(1.0):
                outcome = 1
            elif stop == Float32(2.0):
                outcome = 2
            else:
                polls += 1
                if polls > max_polls:
                    raise Error("qn: the device L-BFGS loop did not stop (QN_IDN_DCONV)")
        k = _f2i(p.unsafe_load(DC_K))
        fx = p.unsafe_load(DC_FX)
        if outcome == 1:
            retcode = _f2i(p.unsafe_load(DC_RET))
        else:
            end = _f2i(p.unsafe_load(DC_END))
            n_vec = _f2i(p.unsafe_load(DC_NVEC))
            for i in range(len(fx_hist)):  # small-loop(fx_hist: objective history window): the L-BFGS stopping window, a handful of past values
                fx_hist[i] = p.unsafe_load(DC_HIST + i)
        _ = len(dst)
        _ = hs.unsafe_ptr()
        return outcome
