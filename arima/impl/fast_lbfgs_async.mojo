# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-gap-arima (2026-10-03): two FAST-on-Apple switches for
`batched_fit.mojo::batched_min_lbfgs`, both default OFF.

`-D MOJOLEARN_ARIMA_FAST_LS_NOREAD=1` (ARIMA_FAST_LS_NOREAD). The lock-step
solver reads `any_searching` back right after `lbfgs_prelude_kernel`, a
host wait per L-BFGS iteration whose answer is "yes" whenever any series is
active (the prelude starts a search for every active series unless its step
or direction is invalid). The switch skips that first read: when no series
searches, the one extra evaluation proposes every series' current `x`, the
accept kernel changes nothing and the next read ends the search. Same bits,
one host wait fewer per iteration.

`-D MOJOLEARN_ARIMA_FAST_ASYNC=1` (ARIMA_FAST_ASYNC). The lock-step solver
moves every series through the same phase together: prelude, a shared line
search that runs until the LAST series accepts, verdict, with three host
waits per iteration. Here every series runs its own state machine in one
kernel (`async_step_kernel`): after each shared evaluation a series takes
its Armijo test and, when its line search is over, runs its verdict, its
history update, the next prelude and its next candidate in the same thread,
so the next evaluation already carries its next point. The per-series
statements are `lbfgs_device.mojo`'s kernels' in the same order on the same
operands (a series' evaluation reads only its own candidate: every Kalman
member, the infeasible mark and the gradient are per series), so each
series' iterates, `n_iter` and `retcode` are the lock-step solver's bit for
bit; what changes is that a series no longer waits for the slowest line
search of the batch, and the host reads one word every ASYNC_READ_EVERY
evaluations. The iteration number is per series (`kcur`).

Kernel arguments are packed (Metal's 31-buffer limit): the per-series ints
in `ist` (field-major, `ist[f * bs + b]`), the per-series floats in `fst`.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from std.math import isinf, isnan
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.memory import memcpy

from arima.impl.batched_kalman import KALMAN_FAST_EVAL_WS
from arima.impl.fast_eval_ws import FastEvalWS
from arima.impl.lbfgs_device import (
    LBFGS_TPB,
    _search_dir,
    d_armijo_ok,
    d_check_convergence,
    d_dot,
    d_nrm_max,
    lbfgs_init_kernel,
)
from arima.impl.tsa.arima_common import ARIMAOrder
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_FAST,
    ftz,
    identical_mul,
    identical_mul_add,
)
from glm.impl.qn.qn_util import (
    LS_INVALID_DIR,
    LS_INVALID_STEP,
    LS_INVALID_STEP_MAX,
    LS_INVALID_STEP_MIN,
    LS_MAX_ITERS_REACHED,
    LS_SUCCESS,
    OPT_LS_FAILED,
    OPT_NUMERIC_ERROR,
    OPT_SUCCESS,
    LBFGSParam,
)

comptime ARIMA_FAST_LS_NOREAD = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_ARIMA_FAST_LS_NOREAD"]()
    and not is_defined["MOJOLEARN_ARIMA_FAST_LS_NOREAD_OFF"]()
)

comptime ARIMA_FAST_ASYNC = (
    KALMAN_FAST_EVAL_WS
    and is_defined["MOJOLEARN_ARIMA_FAST_ASYNC"]()
    and not is_defined["MOJOLEARN_ARIMA_FAST_ASYNC_OFF"]()
)

#: evaluations between two reads of the "any series still running" word;
#: the rounds after the last series stops propose current points and change
#: nothing.
comptime ASYNC_READ_EVERY = 4

comptime FP = MutPointer[Float32, MutAnyOrigin]
comptime IP = MutPointer[Int32, MutAnyOrigin]

# `ist` fields
comptime I_ACTIVE = 0
comptime I_SEARCHING = 1
comptime I_ENDV = 2
comptime I_NVEC = 3
comptime I_LSRET = 4
comptime I_LSITERS = 5
comptime I_NITER = 6
comptime I_RETCODE = 7
comptime I_KCUR = 8
comptime I_N = 9
# `fst` fields
comptime F_FX = 0
comptime F_FXP = 1
comptime F_FXINIT = 2
comptime F_DGINIT = 3
comptime F_DGTEST = 4
comptime F_GNORM = 5
comptime F_STEP = 6
comptime F_FXC = 7
comptime F_N = 8


@always_inline
def _series() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


@always_inline
def _copy_x(cand: FP, x: FP, b: Int, n: Int):
    for i in range(n):
        cand[b * n + i] = x[b * n + i]


@always_inline
def _candidate(cand: FP, xp: FP, drt: FP, s: Float32, b: Int, n: Int):
    """`lbfgs_candidate_kernel`'s searching arm."""
    for i in range(n):
        cand[b * n + i] = ftz(identical_mul_add(s, drt[b * n + i], xp[b * n + i]))


@always_inline
def _prelude(
    ist: IP, fst: FP, x: FP, xp: FP, grad: FP, gradp: FP, drt: FP,
    b: Int, bs: Int, n: Int, ftol: Float32,
) -> Bool:
    """`lbfgs_prelude_kernel` for an active series; True = searching."""
    ist[I_SEARCHING * bs + b] = 0
    for i in range(n):
        xp[b * n + i] = x[b * n + i]
        gradp[b * n + i] = grad[b * n + i]
    fst[F_FXP * bs + b] = fst[F_FX * bs + b]
    if fst[F_STEP * bs + b] <= Float32(0.0):
        ist[I_LSRET * bs + b] = Int32(LS_INVALID_STEP)
        return False
    fst[F_FXINIT * bs + b] = fst[F_FX * bs + b]
    var dg = d_dot(grad, b * n, drt, b * n, n)
    fst[F_DGINIT * bs + b] = dg
    if dg > Float32(0.0):
        ist[I_LSRET * bs + b] = Int32(LS_INVALID_DIR)
        return False
    fst[F_DGTEST * bs + b] = ftz(identical_mul(ftol, dg))
    ist[I_LSITERS * bs + b] = 0
    ist[I_LSRET * bs + b] = Int32(LS_MAX_ITERS_REACHED)
    ist[I_SEARCHING * bs + b] = 1
    return True


@always_inline
def _verdict(
    ist: IP, fst: FP, x: FP, xp: FP, grad: FP, gradp: FP, drt: FP,
    S: FP, Y: FP, yhist: FP, alpha: FP, fx_hist: FP,
    b: Int, bs: Int, n: Int, m: Int, past: Int, k: Int,
    epsilon: Float32, delta: Float32, ftol: Float32,
) -> Bool:
    """`lbfgs_verdict_kernel` for an active series at iteration `k`;
    True = the series stops."""
    var g = d_nrm_max(grad, b * n, n)
    fst[F_GNORM * bs + b] = g
    var f = fst[F_FX * bs + b]
    var fp = fst[F_FXP * bs + b]
    var ls = Int(ist[I_LSRET * bs + b])
    var stop = False
    var converged = False
    var code = Int(ist[I_RETCODE * bs + b])
    var is_ls_valid = (not isnan(f)) and (not isinf(f))
    var is_ls_non_critical = ls == LS_INVALID_STEP_MIN or ls == LS_MAX_ITERS_REACHED
    var is_ls_in_doubt = is_ls_valid and f <= fp + ftol and is_ls_non_critical
    var is_ls_success = ls == LS_SUCCESS or is_ls_in_doubt
    if is_ls_valid:
        converged = d_check_convergence(k, f, g, fx_hist, b * past, past, epsilon, delta)
    if (not is_ls_success) and (not converged):
        code = OPT_LS_FAILED
        stop = True
    elif not is_ls_valid:
        code = OPT_NUMERIC_ERROR
        stop = True
    elif converged:
        code = OPT_SUCCESS
        stop = True
    elif is_ls_in_doubt and f + ftol >= fp:
        code = OPT_LS_FAILED
        stop = True
    var restore = (not is_ls_success) or (not is_ls_valid)
    ist[I_RETCODE * bs + b] = Int32(code)
    if restore:
        fst[F_FX * bs + b] = fp
        for i in range(n):
            x[b * n + i] = xp[b * n + i]
            grad[b * n + i] = gradp[b * n + i]
    ist[I_NITER * bs + b] = Int32(k)
    if stop:
        return True
    var e = Int(ist[I_ENDV * bs + b])
    for i in range(n):
        S[(b * m + e) * n + i] = ftz(
            identical_mul_add(Float32(-1.0), xp[b * n + i], x[b * n + i])
        )
        Y[(b * m + e) * n + i] = ftz(
            identical_mul_add(Float32(-1.0), gradp[b * n + i], grad[b * n + i])
        )
    var nv = Int(ist[I_NVEC * bs + b])
    ist[I_ENDV * bs + b] = Int32(_search_dir(m, nv, e, S, Y, grad, drt, yhist, alpha, b, n))
    ist[I_NVEC * bs + b] = Int32(nv)
    fst[F_STEP * bs + b] = Float32(1.0)
    return False


@always_inline
def _drive(
    ist: IP, fst: FP, x: FP, xp: FP, grad: FP, gradp: FP, drt: FP, cand: FP,
    S: FP, Y: FP, yhist: FP, alpha: FP, fx_hist: FP, any_active: IP,
    b: Int, bs: Int, n: Int, m: Int, past: Int, max_iter: Int,
    epsilon: Float32, delta: Float32, ftol: Float32, verdict_first: Bool,
):
    """From the end of a line search (`verdict_first`) or from the start of
    iteration `kcur`: verdict, next prelude, until a series proposes a
    candidate (raise `any_active`) or stops (proposes its `x`)."""
    var need_verdict = verdict_first
    while True:
        if need_verdict:
            var k = Int(ist[I_KCUR * bs + b])
            var stop = _verdict(
                ist, fst, x, xp, grad, gradp, drt, S, Y, yhist, alpha, fx_hist,
                b, bs, n, m, past, k, epsilon, delta, ftol,
            )
            if stop or k >= max_iter:
                ist[I_ACTIVE * bs + b] = 0
                _copy_x(cand, x, b, n)
                return
            ist[I_KCUR * bs + b] = Int32(k + 1)
        if _prelude(ist, fst, x, xp, grad, gradp, drt, b, bs, n, ftol):
            _candidate(cand, xp, drt, fst[F_STEP * bs + b], b, n)
            _ = Atomic[DType.int32].max(any_active, Int32(1))
            return
        need_verdict = True


def async_start_kernel(
    ist: IP, fst: FP, x: FP, xp: FP, grad: FP, gradp: FP, drt: FP, cand: FP,
    S: FP, Y: FP, yhist: FP, alpha: FP, fx_hist: FP, any_active: IP,
    bs_in: Int32, n_in: Int32, m_in: Int32, past_in: Int32, max_iter_in: Int32,
    epsilon: Float32, delta: Float32, ftol: Float32,
):
    """After `lbfgs_init_kernel`: iteration 1's prelude and first candidate."""
    var b = _series()
    var bs = Int(bs_in)
    if b >= bs:
        return
    var n = Int(n_in)
    ist[I_KCUR * bs + b] = 1
    if ist[I_ACTIVE * bs + b] == 0 or Int(max_iter_in) < 1:
        _copy_x(cand, x, b, n)
        return
    _drive(
        ist, fst, x, xp, grad, gradp, drt, cand, S, Y, yhist, alpha, fx_hist, any_active,
        b, bs, n, Int(m_in), Int(past_in), Int(max_iter_in), epsilon, delta, ftol, False,
    )


def async_step_kernel(
    ist: IP, fst: FP, x: FP, xp: FP, grad: FP, gradp: FP, drt: FP, cand: FP,
    gradc: FP, S: FP, Y: FP, yhist: FP, alpha: FP, fx_hist: FP, any_active: IP,
    bs_in: Int32, n_in: Int32, m_in: Int32, past_in: Int32, max_iter_in: Int32,
    max_ls_in: Int32, epsilon: Float32, delta: Float32, ftol: Float32,
    min_step: Float32, max_step: Float32, ls_dec: Float32,
):
    """After a shared evaluation of `cand`: `lbfgs_accept_kernel`'s
    statements, then (line search over) `_drive`."""
    var b = _series()
    var bs = Int(bs_in)
    if b >= bs:
        return
    var n = Int(n_in)
    if ist[I_ACTIVE * bs + b] == 0:
        _copy_x(cand, x, b, n)
        return
    for i in range(n):
        x[b * n + i] = cand[b * n + i]
        grad[b * n + i] = gradc[b * n + i]
    fst[F_FX * bs + b] = fst[F_FXC * bs + b]
    var lsi = ist[I_LSITERS * bs + b] + 1
    ist[I_LSITERS * bs + b] = lsi
    var s = fst[F_STEP * bs + b]
    var still = False
    if d_armijo_ok(fst[F_FX * bs + b], fst[F_FXINIT * bs + b], s, fst[F_DGTEST * bs + b]):
        ist[I_LSRET * bs + b] = Int32(LS_SUCCESS)
        ist[I_SEARCHING * bs + b] = 0
    elif s < min_step:
        ist[I_LSRET * bs + b] = Int32(LS_INVALID_STEP_MIN)
        ist[I_SEARCHING * bs + b] = 0
    elif s > max_step:
        ist[I_LSRET * bs + b] = Int32(LS_INVALID_STEP_MAX)
        ist[I_SEARCHING * bs + b] = 0
    else:
        fst[F_STEP * bs + b] = ftz(identical_mul(s, ls_dec))
        still = True
    if still and Int(lsi) < Int(max_ls_in):
        _candidate(cand, xp, drt, fst[F_STEP * bs + b], b, n)
        _ = Atomic[DType.int32].max(any_active, Int32(1))
        return
    _drive(
        ist, fst, x, xp, grad, gradp, drt, cand, S, Y, yhist, alpha, fx_hist, any_active,
        b, bs, n, Int(m_in), Int(past_in), Int(max_iter_in), epsilon, delta, ftol, True,
    )


@fieldwise_init
struct AsyncLBFGSOut(Movable):
    var x: List[Float32]
    var fx: List[Float32]
    var n_iter: List[Int32]
    var retcode: List[Int32]
    var n_eval: Int


def _down_f32(ctx: DeviceContext, buf: DeviceBuffer[DType.float32], off: Int, n: Int) raises -> List[Float32]:
    var h = ctx.enqueue_create_host_buffer[DType.float32](max(1, n))
    if n > 0:
        ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf.create_sub_buffer[DType.float32](off, n))
    ctx.synchronize()
    var out = List[Float32](length=n, fill=Float32(0.0))
    if n > 0:
        memcpy(dest=out.unsafe_ptr(), src=h.unsafe_ptr(), count=n)
    _ = h^
    return out^


def _down_i32(ctx: DeviceContext, buf: DeviceBuffer[DType.int32], off: Int, n: Int) raises -> List[Int32]:
    var h = ctx.enqueue_create_host_buffer[DType.int32](max(1, n))
    if n > 0:
        ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf.create_sub_buffer[DType.int32](off, n))
    ctx.synchronize()
    var out = List[Int32](length=n, fill=Int32(0))
    if n > 0:
        memcpy(dest=out.unsafe_ptr(), src=h.unsafe_ptr(), count=n)
    _ = h^
    return out^


def async_min_lbfgs(
    ctx: DeviceContext,
    mut ews: FastEvalWS,
    batch_size: Int,
    scale: Float32,
    order_kf: ARIMAOrder,
    x0: List[Float32],
    param: LBFGSParam,
    h: Float32,
) raises -> AsyncLBFGSOut:
    """`batched_min_lbfgs` with every series on its own schedule (module
    banner). The caller takes this arm only under ARIMA_FAST_ASYNC, with the
    held evaluation workspace and the identity trace off."""
    var n = order_kf.complexity()
    var bs = batch_size
    var b_n = bs * n
    var m = param.m
    var past = param.past if param.past > 0 else 0
    var grid = (bs + LBFGS_TPB - 1) // LBFGS_TPB
    var ist = ctx.enqueue_create_buffer[DType.int32](max(1, I_N * bs))
    var fst = ctx.enqueue_create_buffer[DType.float32](max(1, F_N * bs))
    var x = ctx.enqueue_create_buffer[DType.float32](max(1, b_n))
    var cand = ctx.enqueue_create_buffer[DType.float32](max(1, b_n))
    var xp = ctx.enqueue_create_buffer[DType.float32](max(1, b_n))
    var grad = ctx.enqueue_create_buffer[DType.float32](max(1, b_n))
    var gradp = ctx.enqueue_create_buffer[DType.float32](max(1, b_n))
    var gradc = ctx.enqueue_create_buffer[DType.float32](max(1, b_n))
    var drt = ctx.enqueue_create_buffer[DType.float32](max(1, b_n))
    var d_grad = ctx.enqueue_create_buffer[DType.float32](max(1, b_n))
    var d_x_pert = ctx.enqueue_create_buffer[DType.float32](max(1, b_n))
    var S = ctx.enqueue_create_buffer[DType.float32](max(1, b_n * m))
    var Y = ctx.enqueue_create_buffer[DType.float32](max(1, b_n * m))
    var yhist = ctx.enqueue_create_buffer[DType.float32](max(1, bs * m))
    var alpha = ctx.enqueue_create_buffer[DType.float32](max(1, bs * m))
    var fx_hist = ctx.enqueue_create_buffer[DType.float32](max(1, bs * (past if past > 0 else 1)))
    var bad = ctx.enqueue_create_buffer[DType.int32](max(1, bs))
    var any_active = ctx.enqueue_create_buffer[DType.int32](1)
    var flag_host = ctx.enqueue_create_host_buffer[DType.int32](1)
    var f_fx = fst.create_sub_buffer[DType.float32](F_FX * bs, max(1, bs))
    var f_fxc = fst.create_sub_buffer[DType.float32](F_FXC * bs, max(1, bs))
    ctx.enqueue_memset(ist, Int32(0))
    ctx.enqueue_memset(fst, Float32(0.0))
    ctx.enqueue_memset(xp, Float32(0.0))
    ctx.enqueue_memset(gradp, Float32(0.0))
    ctx.enqueue_memset(gradc, Float32(0.0))
    ctx.enqueue_memset(drt, Float32(0.0))
    ctx.enqueue_memset(S, Float32(0.0))
    ctx.enqueue_memset(Y, Float32(0.0))
    ctx.enqueue_memset(yhist, Float32(0.0))
    ctx.enqueue_memset(alpha, Float32(0.0))
    ctx.enqueue_memset(fx_hist, Float32(0.0))
    if b_n > 0:
        var hx = x0.copy()
        ctx.enqueue_copy(dst_buf=x.create_sub_buffer[DType.float32](0, b_n), src_ptr=hx.unsafe_ptr())
        ctx.synchronize()
        _ = hx^
    ctx.enqueue_copy(dst_buf=cand, src_buf=x)
    # the evaluation at x0, then `lbfgs_init_kernel` on the packed fields
    ews.eval(ctx, order_kf, h, scale, cand, d_grad, d_x_pert, f_fx, grad, bad)
    var n_eval = 1
    var ip = ist.unsafe_ptr()
    var fp = fst.unsafe_ptr()
    ctx.enqueue_memset(any_active, Int32(0))
    # one sub-buffer per field (one argument may not alias another)
    var s_fxp = fst.create_sub_buffer[DType.float32](F_FXP * bs, max(1, bs))
    var s_gn = fst.create_sub_buffer[DType.float32](F_GNORM * bs, max(1, bs))
    var s_st = fst.create_sub_buffer[DType.float32](F_STEP * bs, max(1, bs))
    var s_ac = ist.create_sub_buffer[DType.int32](I_ACTIVE * bs, max(1, bs))
    var s_se = ist.create_sub_buffer[DType.int32](I_SEARCHING * bs, max(1, bs))
    var s_en = ist.create_sub_buffer[DType.int32](I_ENDV * bs, max(1, bs))
    var s_nv = ist.create_sub_buffer[DType.int32](I_NVEC * bs, max(1, bs))
    var s_lr = ist.create_sub_buffer[DType.int32](I_LSRET * bs, max(1, bs))
    var s_li = ist.create_sub_buffer[DType.int32](I_LSITERS * bs, max(1, bs))
    var s_ni = ist.create_sub_buffer[DType.int32](I_NITER * bs, max(1, bs))
    var s_rc = ist.create_sub_buffer[DType.int32](I_RETCODE * bs, max(1, bs))
    ctx.enqueue_function[lbfgs_init_kernel](
        grad.unsafe_ptr(), drt.unsafe_ptr(), f_fx.unsafe_ptr(),
        s_fxp.unsafe_ptr(), fx_hist.unsafe_ptr(), s_gn.unsafe_ptr(), s_st.unsafe_ptr(),
        s_ac.unsafe_ptr(), s_se.unsafe_ptr(), s_en.unsafe_ptr(), s_nv.unsafe_ptr(),
        s_lr.unsafe_ptr(), s_li.unsafe_ptr(), s_ni.unsafe_ptr(), s_rc.unsafe_ptr(),
        any_active.unsafe_ptr(),
        Int32(bs), Int32(n), Int32(past), param.epsilon, param.delta,
        grid_dim=(grid, 1, 1), block_dim=(LBFGS_TPB, 1, 1),
    )
    ctx.enqueue_memset(any_active, Int32(0))
    ctx.enqueue_function[async_start_kernel](
        ip, fp, x.unsafe_ptr(), xp.unsafe_ptr(), grad.unsafe_ptr(), gradp.unsafe_ptr(),
        drt.unsafe_ptr(), cand.unsafe_ptr(), S.unsafe_ptr(), Y.unsafe_ptr(),
        yhist.unsafe_ptr(), alpha.unsafe_ptr(), fx_hist.unsafe_ptr(), any_active.unsafe_ptr(),
        Int32(bs), Int32(n), Int32(m), Int32(past), Int32(param.max_iterations),
        param.epsilon, param.delta, param.ftol,
        grid_dim=(grid, 1, 1), block_dim=(LBFGS_TPB, 1, 1),
    )
    # an upper bound on the rounds any series can need
    var max_rounds = param.max_iterations * max(1, param.max_linesearch) + 1
    var rounds = 0
    var running = True
    while running and rounds < max_rounds:
        if rounds % ASYNC_READ_EVERY == 0:
            ctx.enqueue_copy(dst_ptr=flag_host.unsafe_ptr(), src_buf=any_active)
            ctx.synchronize()
            if flag_host.unsafe_ptr()[0] == 0:
                running = False
                break
        ews.eval(ctx, order_kf, h, scale, cand, d_grad, d_x_pert, f_fxc, gradc, bad)
        n_eval += 1
        ctx.enqueue_memset(any_active, Int32(0))
        ctx.enqueue_function[async_step_kernel](
            ip, fp, x.unsafe_ptr(), xp.unsafe_ptr(), grad.unsafe_ptr(), gradp.unsafe_ptr(),
            drt.unsafe_ptr(), cand.unsafe_ptr(), gradc.unsafe_ptr(), S.unsafe_ptr(),
            Y.unsafe_ptr(), yhist.unsafe_ptr(), alpha.unsafe_ptr(), fx_hist.unsafe_ptr(),
            any_active.unsafe_ptr(),
            Int32(bs), Int32(n), Int32(m), Int32(past), Int32(param.max_iterations),
            Int32(param.max_linesearch), param.epsilon, param.delta, param.ftol,
            param.min_step, param.max_step, param.ls_dec,
            grid_dim=(grid, 1, 1), block_dim=(LBFGS_TPB, 1, 1),
        )
        rounds += 1
    _ = s_fxp^
    _ = s_gn^
    _ = s_st^
    _ = s_ac^
    _ = s_se^
    _ = s_en^
    _ = s_nv^
    _ = s_lr^
    _ = s_li^
    _ = s_ni^
    _ = s_rc^
    var x_out = _down_f32(ctx, x, 0, b_n)
    var fx_out = _down_f32(ctx, fst, F_FX * bs, bs)
    var n_iter_out = _down_i32(ctx, ist, I_NITER * bs, bs)
    var retcode_out = _down_i32(ctx, ist, I_RETCODE * bs, bs)
    _ = f_fx^
    _ = f_fxc^
    _ = ist^
    _ = fst^
    _ = x^
    _ = cand^
    _ = xp^
    _ = grad^
    _ = gradp^
    _ = gradc^
    _ = drt^
    _ = d_grad^
    _ = d_x_pert^
    _ = S^
    _ = Y^
    _ = yhist^
    _ = alpha^
    _ = fx_hist^
    _ = bad^
    _ = any_active^
    _ = flag_host^
    return AsyncLBFGSOut(
        x=x_out^, fx=fx_out^, n_iter=n_iter_out^, retcode=retcode_out^, n_eval=n_eval
    )
