# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The batched ARIMA L-BFGS state machine ON THE DEVICE (cpu-gpu-cleanup
n-seq, 2026-10-02): one GPU thread per series, every series' optimizer state
resident between launches. `arima/impl/batched_fit.mojo::batched_min_lbfgs`
drives it; the host only sequences the launches and reads one "any series
still active / still searching" word per step.

THE SAME ARITHMETIC AS THE HOST COLUMN, SERIES BY SERIES. Each function
below is the device spelling of one in `arima/impl/lbfgs_host.mojo` (which
the host oracle `arima/host/arima_oracle.mojo::_batched_min_lbfgs` runs):
the same serial ascending folds over a series' `n <= ~20` parameters
(`dot_at`, `nrm2_at`, `nrm_max_at`), the same decision rules (`armijo_ok`,
`check_convergence_at`, `lbfgs_verdict`) and the same two-loop recursion
(`lbfgs_search_dir_at`). Every product is `identical_mul` and every
quotient `identical_div` here, where the host writes `*` and `/`: under
IDENTICAL those are the correctly rounded operation the host's statement
computes (no operand of a quotient is subnormal: each is a flushed fold or a
flushed constant), and they keep a device compiler from contracting a
product into a neighbouring add. So a series' iterates, branch sequence and
iteration count are the host column's bit for bit.

A thread owns one series and walks its `n` parameters itself: the work per
series is a handful of `n`-length folds, and the series are the parallel
axis (thousands per batch). Nothing folds ACROSS series.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from std.math import inf, isinf, isnan

from checks.numerics import (
    ftz,
    identical_div,
    identical_mul,
    identical_mul_add,
    identical_sqrt,
)
from glm.impl.qn.qn_util import (
    FLOAT_EPSILON,
    LS_INVALID_DIR,
    LS_INVALID_STEP,
    LS_INVALID_STEP_MAX,
    LS_INVALID_STEP_MIN,
    LS_MAX_ITERS_REACHED,
    LS_SUCCESS,
    OPT_LS_FAILED,
    OPT_MAX_ITERS_REACHED,
    OPT_NUMERIC_ERROR,
    OPT_SUCCESS,
)

comptime FP = MutPointer[Float32, MutAnyOrigin]
comptime IP = MutPointer[Int32, MutAnyOrigin]

#: Threads per block of every kernel here (one thread per series).
comptime LBFGS_TPB = 128


@always_inline
def _series() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


@always_inline
def _raise_flag(flag: IP):
    _ = Atomic[DType.int32].max(flag, Int32(1))


# ---------------------------------------------------------------------------
# the folds (`lbfgs_host.mojo`'s, over a pointer and a base)
# ---------------------------------------------------------------------------


@always_inline
def d_nrm_max(v: FP, base: Int, n: Int) -> Float32:
    var acc = Float32(0.0)
    for i in range(n):
        var x = abs(v[base + i])
        if x > acc:
            acc = x
    return acc


@always_inline
def d_dot(u: FP, ub: Int, v: FP, vb: Int, n: Int) -> Float32:
    var acc = Float32(0.0)
    for i in range(n):
        acc = ftz(identical_mul_add(u[ub + i], v[vb + i], acc))
    return acc


@always_inline
def d_nrm2(v: FP, base: Int, n: Int) -> Float32:
    return ftz(identical_sqrt(d_dot(v, base, v, base, n)))


# ---------------------------------------------------------------------------
# the decision rules
# ---------------------------------------------------------------------------


@always_inline
def d_armijo_ok(fx: Float32, fx_init: Float32, step: Float32, dg_test: Float32) -> Bool:
    return not (fx > identical_mul_add(step, dg_test, fx_init))


@always_inline
def d_check_convergence(
    k: Int, fx: Float32, gnorm: Float32, fx_hist: FP, hist_base: Int,
    past: Int, epsilon: Float32, delta: Float32,
) -> Bool:
    var fmag = max(fx, epsilon)
    if gnorm <= identical_mul(epsilon, fmag):
        return True
    if past > 0:
        if k >= past and abs(fx_hist[hist_base + k % past] - fx) <= identical_mul(delta, fmag):
            return True
        fx_hist[hist_base + k % past] = fx
    return False


# ---------------------------------------------------------------------------
# the kernels, in the order `batched_min_lbfgs` launches them
# ---------------------------------------------------------------------------


def lbfgs_init_kernel(
    grad: FP, drt: FP, fx: FP, fxp: FP, fx_hist: FP, gnorm: FP, step: FP,
    active: IP, searching: IP, endv: IP, n_vec: IP, lsret: IP, ls_iters: IP,
    n_iter: IP, retcode: IP, any_active: IP,
    batch_in: Int32, n_in: Int32, past_in: Int32,
    epsilon: Float32, delta: Float32,
):
    """`min_lbfgs:161-173` for every series after the evaluation at x0:
    the state's start values, the early exit of a series that is already a
    minimizer (or infeasible), and the first direction `-grad` with step
    `1 / ||drt||`."""
    var b = _series()
    if b >= Int(batch_in):
        return
    var n = Int(n_in)
    var past = Int(past_in)
    searching[b] = 0
    endv[b] = 0
    n_vec[b] = 0
    lsret[b] = Int32(LS_SUCCESS)
    ls_iters[b] = 0
    n_iter[b] = 0
    retcode[b] = Int32(OPT_MAX_ITERS_REACHED)
    var g = d_nrm_max(grad, b * n, n)
    gnorm[b] = g
    if past > 0:
        fx_hist[b * past] = fx[b]
    if isinf(fx[b]):
        retcode[b] = Int32(OPT_NUMERIC_ERROR)
        active[b] = 0
        return
    if d_check_convergence(0, fx[b], g, fx_hist, b * past, past, epsilon, delta):
        retcode[b] = Int32(OPT_SUCCESS)
        active[b] = 0
        return
    for i in range(n):
        drt[b * n + i] = ftz(identical_mul(Float32(-1.0), grad[b * n + i]))
    step[b] = ftz(identical_div(Float32(1.0), d_nrm2(drt, b * n, n)))
    fxp[b] = fx[b]
    active[b] = 1
    _raise_flag(any_active)


def lbfgs_prelude_kernel(
    x: FP, xp: FP, grad: FP, gradp: FP, drt: FP,
    fx: FP, fxp: FP, fx_init: FP, dg_init: FP, dg_test: FP, step: FP,
    active: IP, searching: IP, lsret: IP, ls_iters: IP, any_searching: IP,
    batch_in: Int32, n_in: Int32, ftol: Float32,
):
    """`min_lbfgs:188-191` (save x, grad, fx) and `ls_backtrack:100-108`, the
    part of the line search before its loop."""
    var b = _series()
    if b >= Int(batch_in):
        return
    searching[b] = 0
    if active[b] == 0:
        return
    var n = Int(n_in)
    for i in range(n):
        xp[b * n + i] = x[b * n + i]
        gradp[b * n + i] = grad[b * n + i]
    fxp[b] = fx[b]
    if step[b] <= Float32(0.0):
        lsret[b] = Int32(LS_INVALID_STEP)
        return
    fx_init[b] = fx[b]
    var dg = d_dot(grad, b * n, drt, b * n, n)
    dg_init[b] = dg
    if dg > Float32(0.0):
        lsret[b] = Int32(LS_INVALID_DIR)
        return
    dg_test[b] = ftz(identical_mul(ftol, dg))
    ls_iters[b] = 0
    # the value `ls_backtrack` falls through to if the loop exhausts
    lsret[b] = Int32(LS_MAX_ITERS_REACHED)
    searching[b] = 1
    _raise_flag(any_searching)


def lbfgs_candidate_kernel(
    cand: FP, x: FP, xp: FP, drt: FP, step: FP, searching: IP,
    batch_in: Int32, n_in: Int32,
):
    """`axpy(x, step, drt, xp)`: `x = step * drt + xp`, one rounding, for a
    searching series; a series that is not searching proposes its current
    `x`, so the batch composition never depends on convergence."""
    var b = _series()
    if b >= Int(batch_in):
        return
    var n = Int(n_in)
    if searching[b] != 0:
        var s = step[b]
        for i in range(n):
            cand[b * n + i] = ftz(identical_mul_add(s, drt[b * n + i], xp[b * n + i]))
    else:
        for i in range(n):
            cand[b * n + i] = x[b * n + i]


def lbfgs_accept_kernel(
    x: FP, grad: FP, fx: FP, cand: FP, gradc: FP, fxc: FP,
    fx_init: FP, dg_test: FP, step: FP,
    searching: IP, lsret: IP, ls_iters: IP, any_searching: IP,
    batch_in: Int32, n_in: Int32,
    min_step: Float32, max_step: Float32, ls_dec: Float32,
):
    """`ls_backtrack:109-121` after the shared evaluation: take the
    candidate, then the Armijo test, or halve the step and keep searching."""
    var b = _series()
    if b >= Int(batch_in):
        return
    if searching[b] == 0:
        return
    var n = Int(n_in)
    for i in range(n):
        x[b * n + i] = cand[b * n + i]
        grad[b * n + i] = gradc[b * n + i]
    fx[b] = fxc[b]
    ls_iters[b] = ls_iters[b] + 1
    var s = step[b]
    if d_armijo_ok(fx[b], fx_init[b], s, dg_test[b]):
        lsret[b] = Int32(LS_SUCCESS)
        searching[b] = 0
    elif s < min_step:
        lsret[b] = Int32(LS_INVALID_STEP_MIN)
        searching[b] = 0
    elif s > max_step:
        lsret[b] = Int32(LS_INVALID_STEP_MAX)
        searching[b] = 0
    else:
        step[b] = ftz(identical_mul(s, ls_dec))
        _raise_flag(any_searching)


def _search_dir(
    m: Int, mut n_vec: Int, end_prev: Int,
    S: FP, Y: FP, g: FP, drt: FP, yhist: FP, alpha: FP, b: Int, n: Int,
) -> Int:
    """`lbfgs_search_dir_at`, for series `b`, over the same flat layouts."""
    var end = end_prev
    var sb = (b * m + end) * n
    var gb = b * n
    var ys = d_dot(S, sb, Y, sb, n)
    var yy = d_dot(Y, sb, Y, sb, n)
    if ys <= identical_mul(FLOAT_EPSILON, yy):
        return end
    n_vec += 1
    yhist[b * m + end] = ys
    for i in range(n):
        drt[gb + i] = ftz(identical_mul(Float32(-1.0), g[gb + i]))
    var bound = min(m, n_vec)
    end = (end + 1) % m
    var j = end
    for _ in range(bound):
        j = (j + m - 1) % m
        var a = ftz(identical_div(d_dot(S, (b * m + j) * n, drt, gb, n), yhist[b * m + j]))
        alpha[b * m + j] = a
        for i in range(n):
            drt[gb + i] = ftz(identical_mul_add(ftz(-a), Y[(b * m + j) * n + i], drt[gb + i]))
    var scale = ftz(identical_div(ys, yy))
    for i in range(n):
        drt[gb + i] = ftz(identical_mul(scale, drt[gb + i]))
    for _ in range(bound):
        var beta = ftz(identical_div(d_dot(Y, (b * m + j) * n, drt, gb, n), yhist[b * m + j]))
        var c = ftz(alpha[b * m + j] - beta)
        for i in range(n):
            drt[gb + i] = ftz(identical_mul_add(c, S[(b * m + j) * n + i], drt[gb + i]))
        j = (j + 1) % m
    return end


def lbfgs_verdict_kernel(
    x: FP, xp: FP, grad: FP, gradp: FP, drt: FP, S: FP, Y: FP, yhist: FP,
    alpha: FP, fx: FP, fxp: FP, fx_hist: FP, gnorm: FP, step: FP,
    active: IP, endv: IP, n_vec: IP, lsret: IP, n_iter: IP, retcode: IP,
    any_active: IP,
    batch_in: Int32, n_in: Int32, m_in: Int32, past_in: Int32, k_in: Int32,
    epsilon: Float32, delta: Float32, ftol: Float32,
):
    """`min_lbfgs:197-222`: the verdict (`update_and_check`, with the
    restore), the history update and the new direction."""
    var b = _series()
    if b >= Int(batch_in):
        return
    if active[b] == 0:
        return
    var n = Int(n_in)
    var m = Int(m_in)
    var past = Int(past_in)
    var k = Int(k_in)
    var g = d_nrm_max(grad, b * n, n)
    gnorm[b] = g
    # `lbfgs_verdict`
    var f = fx[b]
    var fp = fxp[b]
    var ls = Int(lsret[b])
    var stop = False
    var converged = False
    var code = Int(retcode[b])
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
    retcode[b] = Int32(code)
    if restore:
        fx[b] = fp
        for i in range(n):
            x[b * n + i] = xp[b * n + i]
            grad[b * n + i] = gradp[b * n + i]
    n_iter[b] = Int32(k)
    if stop:
        active[b] = 0
        return
    var e = Int(endv[b])
    for i in range(n):
        # `axpy(S[end], -1, xp, x)` and `axpy(Y[end], -1, gradp, grad)`
        S[(b * m + e) * n + i] = ftz(
            identical_mul_add(Float32(-1.0), xp[b * n + i], x[b * n + i])
        )
        Y[(b * m + e) * n + i] = ftz(
            identical_mul_add(Float32(-1.0), gradp[b * n + i], grad[b * n + i])
        )
    var nv = Int(n_vec[b])
    endv[b] = Int32(_search_dir(m, nv, e, S, Y, grad, drt, yhist, alpha, b, n))
    n_vec[b] = Int32(nv)
    step[b] = Float32(1.0)
    _raise_flag(any_active)


# ---------------------------------------------------------------------------
# the objective's device tail (`eval_batch`'s, without the host lists)
# ---------------------------------------------------------------------------


def arima_mark_infeasible_kernel(
    bad: IP, ll: FP, info0: IP, info1: IP, batch_in: Int32, members_in: Int32,
):
    """Series `b` is INFEASIBLE when any of its `members` evaluations (the
    base and, stacked, each forward-difference point at `m * batch + b`)
    has a Kalman refusal code set or a log-likelihood of -inf
    (`batched_loglike_x`'s `infeasible_inf` marking and `eval_batch`'s
    test, on the device)."""
    var b = _series()
    var bs = Int(batch_in)
    if b >= bs:
        return
    for mm in range(Int(members_in)):
        var j = mm * bs + b
        var v = ll[j]
        if info0[j] != 0 or info1[j] != 0 or (isinf(v) and v < Float32(0.0)):
            bad[b] = 1


def arima_eval_finish_kernel(
    f_out: FP, g_out: FP, ll: FP, g_raw: FP, bad: IP,
    batch_in: Int32, n_in: Int32, scale: Float32,
):
    """`f = -loglike / scale`, `g = -grad / scale` with the sequential path's
    flushes; an infeasible series is `f = +inf` with a zero gradient
    (`_infeasible_fg`)."""
    var b = _series()
    if b >= Int(batch_in):
        return
    var n = Int(n_in)
    if bad[b] != 0:
        f_out[b] = inf[DType.float32]()
        for i in range(n):
            g_out[b * n + i] = Float32(0.0)
        return
    f_out[b] = ftz(identical_div(ftz(-ll[b]), scale))
    for i in range(n):
        g_out[b * n + i] = ftz(identical_div(ftz(-g_raw[b * n + i]), scale))
