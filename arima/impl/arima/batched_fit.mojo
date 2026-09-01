# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`batched_fit`: the entry point that turns a log-likelihood into a fitted
model, and `batched_min_lbfgs`, the batched optimizer it drives.

STANDS IN FOR `cuml/cuml/tsa/arima.pyx::ARIMA.fit` (`:860-958`) and
`cuml/cuml/tsa/batched_lbfgs.py::batched_fmin_lbfgs_b`, at cuML 265b9da6
(v26.08.00).

=============================================================================
DEVIATION 679: cuML HAS NO L-BFGS OF ITS OWN HERE; THIS ONE IS WRITTEN OUT
=============================================================================
THEIRS. `batched_fmin_lbfgs_b` is a HOST PYTHON state machine that calls
`scipy.optimize._lbfgsb.setulb` once per series, in float64, against
Fortran working arrays (`wa`, `iwa`, `isave`, `dsave`), with the objective
and gradient evaluated ONCE FOR THE WHOLE BATCH on the GPU between rounds.
The call site passes `bounds = None`, so every `nbd[i]` is 0 and the "-B"
is inert: this is unconstrained L-BFGS with `m = 10`, `factr = 1000`,
`pgtol = 1e-5`, `maxls = 20`, `maxiter = 1000`.

OURS. The optimizer is written here. scipy is NOT taken as a runtime
dependency: the wheel depends on numpy alone, and a host float64 optimizer
from a third party in the middle of a certified path would put the fitted
coefficients outside anything this repository can reproduce or gate. The
SHAPE is theirs and is kept -- host state machine per series, ONE BATCHED
DEVICE EVALUATION per candidate point -- and the ALGORITHM is cuML's own
L-BFGS, already ported in `glm/impl/glm/qn/`, re-spelled per series in
`arima/impl/arima/lbfgs_host.mojo`. Read that file's banner for why calling
`glm::min_lbfgs` B times is not the answer; the short version is that it is
typed on a concrete `GLMWithData`, it evaluates the objective itself from
inside the line search, and its vector work is device reductions sized for
`n` in the millions where ours is `n <= 20`.

WHAT IS NOT scipy's, AND IS NOT PRETENDED TO BE. scipy's L-BFGS-B is not
cuML's L-BFGS: it is a different line search (More-Thuente / `dcsrch`
inside `mainlb`), a different history update and different stopping
constants. Substituting cuML's own already-ported solver is a REAL
DEVIATION and it means the iterate sequence differs from cuML's, not only
the last bits. What is claimed is a converged maximum-likelihood fit that
this repository can reproduce bit for bit on every vendor, gated against
planted coefficients and against a Float64 stationarity test; what is NOT
claimed is that any iterate, or the iteration count, matches cuML's.

=============================================================================
DEVIATION 687: THE FINITE-DIFFERENCE STEP IS 2^-10, NOT 1e-8
=============================================================================
THEIRS. `h = 1e-8` (`arima.pyx:863`), in float64. That is the textbook
forward-difference optimum there: `sqrt(eps_f64) = 1.49e-8`.

OURS. In Float32 `1e-8` IS BELOW eps (`1.19e-7`), so `x + h` is `x` for
every `|x| > 1e-1` and the gradient is exactly zero or pure noise. cuML's
value cannot be carried across DEVIATION 670 and there is no upstream
answer to inherit.

The gate had quietly been using `1e-3` since 2026-08-23
(`arima/checks/arima_check.mojo`), and nothing recorded that as a decision,
because the gate only ever asked whether the device equalled the oracle and
never whether the gradient was ACCURATE. A `fit` makes it load bearing.

    h = 2^-10 = 0.0009765625

THE DERIVATION. Forward differences carry truncation `~ (h/2)|f''|` and
roundoff `~ 2*delta_f/h`, where `delta_f` is the noise floor of the
objective. The objective is `-loglike / (n_obs - 1)`, which is O(1), and
`check_kalman_matches_float64` MEASURED the log-likelihood's Float32 gap at
4.7e-8 to 1.5e-7 relative with `n_diff = 0` and up to 1.8e-3 with
`n_diff > 0`. Taking `delta_f ~ 1e-7` and `|f''| ~ 1` gives an optimum near
`sqrt(2 * 1e-7) = 4.5e-4`, and the error is flat in `h` around it: at
`h = 9.8e-4` the two terms are 2e-4 and 4.9e-4.

WHY A POWER OF TWO, WHICH IS THE PART THAT IS NOT IN ANY TEXTBOOK. The
gradient is `(f(x+h) - f(x)) / h`, and a divide by a power of two is EXACT
in IEEE-754 for every finite non-overflowing operand. Choosing 1e-3, which
is not representable, puts a rounding on every gradient cell for nothing.
With `h = 2^-10` the gradient's last bit is a function of the two
log-likelihoods alone, and `arima/SEAMS.tsv`'s `gradient` row loses its one
rounding.

WHAT GATES IT. `check_grad_matches_float64` is new and it is the point:
the Float32 device gradient is compared against a FLOAT64 HOST central
difference on the same fixture, so the gate asks whether the gradient is
ACCURATE and not only whether two Float32 spellings agree. `h` is swept
over `2^-6 ... 2^-16` and the error curve is printed, so the choice above
is a measured minimum and not an argument.

=============================================================================
THE CONVERGENCE TOLERANCE IS NOT scipy's 1e-5, AND CANNOT BE
=============================================================================
This is the second thing DEVIATION 670 breaks and it is worth stating
separately because it is easy to carry `pgtol = 1e-5` across by reflex.

With `h = 2^-10` and an objective whose own Float32 noise floor is ~1e-7,
the gradient carries roughly 1e-3 of absolute error on an O(1) objective.
`pgtol = 1e-5` is TWO ORDERS OF MAGNITUDE BELOW THE GRADIENT'S OWN NOISE:
it can never be satisfied, so a solver asked for it runs to `maxiter` on
every series and reports failure on a converged fit. `epsilon = 1e-3` is
the noise floor, and stopping there is stopping when the information runs
out.

`epsilon` and `delta` below are DERIVED, not observed. `check_fit_is_a_
minimizer` prints the Float64 gradient infinity-norm actually achieved, and
a compile slot should replace both with what it sees, exactly as
`arima/README.md`'s bounds table asks for the other four.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from arima.impl.arima.batched_arima import _refuse_non_finite, batched_loglike_grad
from arima.impl.arima.estimate_x0 import StartParamsResult, estimate_x0
from arima.impl.arima.lbfgs_host import (
    armijo_ok,
    check_convergence_at,
    dot_at,
    lbfgs_search_dir_at,
    lbfgs_verdict,
    nrm2_at,
    nrm_max_at,
)
from arima.impl.timeSeries.arima_helpers import batched_jones_transform
from arima.impl.tsa.arima_common import (
    ARIMAOrder,
    ARIMAParams,
    pack,
    unpack,
    validate_order,
)
from checks.numerics import ftz, identical_mul_add
from core.identity_trace import IdentityTrace
from glm.impl.glm.qn.qn_util import (
    LBFGS_LS_BT_ARMIJO,
    LBFGSParam,
    LS_INVALID_DIR,
    LS_INVALID_STEP,
    LS_INVALID_STEP_MAX,
    LS_INVALID_STEP_MIN,
    LS_MAX_ITERS_REACHED,
    LS_SUCCESS,
    OPT_MAX_ITERS_REACHED,
    OPT_SUCCESS,
)
from std.math import isfinite
from tsa.impl.timeSeries.arima_helpers import prepare_data


#: DEVIATION 687. `2^-10`; see the banner. Written as the exact decimal of
#: the binary value, never as `1.0 / 1024.0`, so the literal in the source
#: is the number the machine uses.
comptime ARIMA_FIT_H = Float32(0.0009765625)


def arima_fit_params(max_iterations: Int = 1000) -> LBFGSParam:
    """The L-BFGS settings for an ARIMA fit.

    From cuML's call site (`arima.pyx:927-929`, `batched_lbfgs.py`):
    `m = 10`, `maxls = 20`, `maxiter = 1000`, no bounds. NOT from it:
    `epsilon` and `delta`, because scipy's `pgtol = 1e-5` and
    `factr * eps_mach = 2.2e-13` are both unreachable in Float32; see the
    module banner. The remaining fields are `LBFGSParam::defaults`
    (`qn_util.cuh:71-86`) unchanged, `linesearch = ARMIJO` included, which
    is cuML's own shipped line search."""
    return LBFGSParam(
        10,  # m: their scipy `m`
        Float32(1.0e-3),  # epsilon: the Float32 gradient noise floor. DERIVED
        10,  # past: their `factr` test, in a form Float32 can meet
        Float32(1.0e-6),  # delta: ~16x the Float32 resolution of an O(1) f
        max_iterations,
        LBFGS_LS_BT_ARMIJO,
        20,  # max_linesearch: their `maxls`
        Float32(1.0e-20),
        Float32(1.0e20),
        Float32(1.0e-4),  # ftol
        Float32(0.9),
        Float32(0.5),
        Float32(2.1),
    )


# ---------------------------------------------------------------------------
# host <-> device for the packed parameter vector
# ---------------------------------------------------------------------------


def _upload(ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], v: List[Float32]) raises:
    var n = len(v)
    if n == 0:
        return
    var h = v.copy()
    var view = buf.create_sub_buffer[DType.float32](0, n)
    ctx.enqueue_copy(dst_buf=view, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    _ = h^


def _download(ctx: DeviceContext, buf: DeviceBuffer[DType.float32], n: Int) raises -> List[Float32]:
    var h = ctx.enqueue_create_host_buffer[DType.float32](n if n > 0 else 1)
    if n > 0:
        var view = buf.create_sub_buffer[DType.float32](0, n)
        ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=view)
    ctx.synchronize()
    var out = List[Float32]()
    for i in range(n):
        out.append(h.unsafe_ptr().unsafe_load(i))
    _ = h^
    return out^


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


# ---------------------------------------------------------------------------
# the objective: `-loglike / (n_obs - 1)`, evaluated for the WHOLE BATCH
# ---------------------------------------------------------------------------


def eval_batch(
    ctx: DeviceContext,
    mut d_y_kf: DeviceBuffer[DType.float32],
    batch_size: Int,
    n_obs_kf: Int,
    order_kf: ARIMAOrder,
    mut d_x: DeviceBuffer[DType.float32],
    mut d_grad: DeviceBuffer[DType.float32],
    mut d_x_pert: DeviceBuffer[DType.float32],
    mut scratch: ARIMAParams,
    h: Float32,
    scale: Float32,
    xin: List[Float32],
    mut fout: List[Float32],
    mut gout: List[Float32],
) raises:
    """`fit_helper`'s `f` and `gf` (`arima.pyx:905-919`) in ONE call.

        f(x)  = -loglike(x, trans=True) / (n_obs - 1)
        gf(x) = -loglike_grad(x, h, trans=True) / (n_obs - 1)

    `scale` is `n_obs - 1` with the ORIGINAL `n_obs`, not the differenced
    one; theirs is `self.n_obs` and the differencing has already happened by
    the time the loglike is called.

    A SPELLING COLLAPSE, recorded because it removes a launch and moves no
    bit: theirs calls `func(xk)` and `fprime(xk)` separately, so the base
    log-likelihood is computed TWICE per candidate point.
    `batched_loglike_grad` already evaluates the base and returns it, so
    this takes both from one call. The base value is the same bits either
    way; what is saved is one full batched Kalman pass per evaluation.

    `check_finite = False`: `batched_fit` checks the input series ONCE
    before the loop. Leaving it on would copy the whole series to the host
    and synchronize `(N + 1)` times per candidate point, hundreds of times
    over, to re-answer a question about data nobody has touched."""
    _upload(ctx, d_x, xin)
    var ll = batched_loglike_grad(
        ctx, d_y_kf, batch_size, n_obs_kf, order_kf, d_x, d_grad, h, True,
        scratch, d_x_pert, False,
    )
    var g = _download(ctx, d_grad, len(xin))
    for b in range(batch_size):
        fout[b] = ftz(ftz(-ll[b]) / scale)
    for i in range(len(xin)):
        gout[i] = ftz(ftz(-g[i]) / scale)


# ---------------------------------------------------------------------------
# the batched L-BFGS
# ---------------------------------------------------------------------------


def _iter_tag(k: Int) -> String:
    """`fit.iterNNNN`, zero padded so the card's tags sort and align, as
    `glm/impl/glm/qn/qn_solvers.mojo::_iter_tag` does."""
    var s = String(k)
    while s.byte_length() < 4:
        s = "0" + s
    return "fit.iter" + s


@fieldwise_init
struct BatchedLBFGSResult(Movable):
    var x: List[Float32]
    var fx: List[Float32]
    var n_iter: List[Int32]
    var retcode: List[Int32]
    var n_eval: Int


def batched_min_lbfgs(
    ctx: DeviceContext,
    mut d_y_kf: DeviceBuffer[DType.float32],
    batch_size: Int,
    n_obs_kf: Int,
    scale: Float32,
    order_kf: ARIMAOrder,
    x0: List[Float32],
    param: LBFGSParam,
    h: Float32,
    mut trace: IdentityTrace,
) raises -> BatchedLBFGSResult:
    """B independent L-BFGS solvers driven from ONE host loop, sharing ONE
    batched device evaluation per candidate point.

    THE SHAPE, and the one thing to understand before editing it. The OUTER
    loop is `k`, the L-BFGS iteration. Inside it, the LINE SEARCH is also a
    shared loop: at step `t` every still-searching series proposes its own
    candidate `xp + step_b * drt_b` with its OWN step length, all B
    candidates go into one `d_x`, and ONE `batched_loglike_grad` evaluates
    them together. Each series then applies the Armijo test to its own
    result and either accepts or halves its own step. Series take different
    numbers of line-search steps and that is fine; the loop runs until none
    is still searching, at most `param.max_linesearch` times, and in
    practice once or twice because the first candidate is usually accepted.

    A SERIES THAT IS NOT SEARCHING STILL PROPOSES ITS CURRENT `x`, and its
    result is discarded. That is deliberate and it is not laziness: it keeps
    the BATCH COMPOSITION and the launch geometry a function of the fixture
    ALONE, never of how many series have converged. Without it the identity
    claim for a fit would rest on `check_kalman_launch_invariant`'s
    batch-composition arm rather than being true by construction. It costs
    nothing measurable, because the Kalman kernel is one thread per series.

    Every scalar below is HOST Float32 and every branch is on one. There is
    no device reduction anywhere in this function, so the branch sequence --
    and therefore the ITERATION COUNT, which the card records -- is a
    function of the log-likelihood bits alone."""
    if param.check_param() != 0:
        raise Error(
            "batched_min_lbfgs: invalid parameter (check_param code "
            + String(param.check_param()) + ")"
        )
    var n = order_kf.complexity()
    var b_n = batch_size * n
    var m = param.m
    var past = param.past if param.past > 0 else 0

    # device workspace, allocated ONCE for the whole solve
    var d_x = ctx.enqueue_create_buffer[DType.float32](b_n)
    var d_grad = ctx.enqueue_create_buffer[DType.float32](b_n)
    var d_x_pert = ctx.enqueue_create_buffer[DType.float32](b_n)
    var scratch = ARIMAParams(ctx, order_kf, batch_size)
    ctx.synchronize()

    # host state, flat and per series
    var x = x0.copy()
    var xp = _zeros(b_n)
    var cand = _zeros(b_n)
    var grad = _zeros(b_n)
    var gradp = _zeros(b_n)
    var gradc = _zeros(b_n)
    var drt = _zeros(b_n)
    var S = _zeros(b_n * m)
    var Y = _zeros(b_n * m)
    var yhist = _zeros(batch_size * m)
    var alpha = _zeros(batch_size * m)
    var fx_hist = _zeros(batch_size * (past if past > 0 else 1))
    var fx = _zeros(batch_size)
    var fxc = _zeros(batch_size)
    var fxp = _zeros(batch_size)
    var fx_init = _zeros(batch_size)
    var dg_init = _zeros(batch_size)
    var dg_test = _zeros(batch_size)
    var gnorm = _zeros(batch_size)
    var step = _zeros(batch_size)

    var active = List[Bool]()
    var searching = List[Bool]()
    var endv = List[Int]()
    var n_vec = List[Int]()
    var lsret = List[Int]()
    var ls_iters = List[Int]()
    var n_iter = List[Int32]()
    var retcode = List[Int32]()
    for _ in range(batch_size):
        active.append(True)
        searching.append(False)
        endv.append(0)
        n_vec.append(0)
        lsret.append(LS_SUCCESS)
        ls_iters.append(0)
        n_iter.append(Int32(0))
        retcode.append(Int32(OPT_MAX_ITERS_REACHED))

    var n_eval = 0

    # `min_lbfgs:161-173`: evaluate at x0, and exit early per series if it
    # is already a minimizer.
    eval_batch(
        ctx, d_y_kf, batch_size, n_obs_kf, order_kf, d_x, d_grad, d_x_pert,
        scratch, h, scale, x, fx, grad,
    )
    n_eval += 1
    trace.record_list_f32("fit.init.x", x)
    trace.record_list_f32("fit.init.loss", fx)
    trace.record_list_f32("fit.init.grad", grad)
    for b in range(batch_size):
        gnorm[b] = nrm_max_at(grad, b * n, n)
        if past > 0:
            fx_hist[b * past] = fx[b]
        if check_convergence_at(param, 0, fx[b], gnorm[b], fx_hist, b * past):
            retcode[b] = Int32(OPT_SUCCESS)
            active[b] = False
            n_iter[b] = Int32(0)
        else:
            for i in range(n):
                drt[b * n + i] = ftz(Float32(-1.0) * grad[b * n + i])
            step[b] = ftz(Float32(1.0) / nrm2_at(drt, b * n, n))
            fxp[b] = fx[b]

    var k = 1
    while k <= param.max_iterations:
        var any_active = False
        for b in range(batch_size):
            if active[b]:
                any_active = True
        if not any_active:
            break

        # `min_lbfgs:188-191`: save x, grad, fx
        for b in range(batch_size):
            if not active[b]:
                continue
            for i in range(n):
                xp[b * n + i] = x[b * n + i]
                gradp[b * n + i] = grad[b * n + i]
            fxp[b] = fx[b]

        # `ls_backtrack:100-108`, the part before the loop
        for b in range(batch_size):
            searching[b] = False
            if not active[b]:
                continue
            if step[b] <= Float32(0.0):
                lsret[b] = LS_INVALID_STEP
                continue
            fx_init[b] = fx[b]
            dg_init[b] = dot_at(grad, b * n, drt, b * n, n)
            if dg_init[b] > Float32(0.0):
                lsret[b] = LS_INVALID_DIR
                continue
            dg_test[b] = ftz(param.ftol * dg_init[b])
            ls_iters[b] = 0
            # the value `ls_backtrack` falls through to if the loop exhausts
            lsret[b] = LS_MAX_ITERS_REACHED
            searching[b] = True

        # THE SHARED LINE SEARCH (`ls_backtrack:109-121`, B at a time)
        for _t in range(param.max_linesearch):
            var any_s = False
            for b in range(batch_size):
                if searching[b]:
                    any_s = True
            if not any_s:
                break
            for b in range(batch_size):
                if searching[b]:
                    # `axpy(x, step, drt, xp)`: `x = step * drt + xp`, one
                    # rounding (`dense.mojo::axpy_kernel`, row 9)
                    for i in range(n):
                        cand[b * n + i] = ftz(
                            identical_mul_add(step[b], drt[b * n + i], xp[b * n + i])
                        )
                else:
                    for i in range(n):
                        cand[b * n + i] = x[b * n + i]
            eval_batch(
                ctx, d_y_kf, batch_size, n_obs_kf, order_kf, d_x, d_grad,
                d_x_pert, scratch, h, scale, cand, fxc, gradc,
            )
            n_eval += 1
            for b in range(batch_size):
                if not searching[b]:
                    continue
                for i in range(n):
                    x[b * n + i] = cand[b * n + i]
                    grad[b * n + i] = gradc[b * n + i]
                fx[b] = fxc[b]
                ls_iters[b] += 1
                if armijo_ok(fx[b], fx_init[b], step[b], dg_test[b]):
                    lsret[b] = LS_SUCCESS
                    searching[b] = False
                elif step[b] < param.min_step:
                    lsret[b] = LS_INVALID_STEP_MIN
                    searching[b] = False
                elif step[b] > param.max_step:
                    lsret[b] = LS_INVALID_STEP_MAX
                    searching[b] = False
                else:
                    step[b] = ftz(step[b] * param.ls_dec)

        # `min_lbfgs:197-222`: verdict, history update, new direction
        for b in range(batch_size):
            if not active[b]:
                continue
            gnorm[b] = nrm_max_at(grad, b * n, n)
            var code = Int(retcode[b])
            var restore = False
            var stop = lbfgs_verdict(
                param, k, lsret[b], fx[b], fxp[b], gnorm[b], fx_hist,
                b * past, code, restore,
            )
            retcode[b] = Int32(code)
            if restore:
                fx[b] = fxp[b]
                for i in range(n):
                    x[b * n + i] = xp[b * n + i]
                    grad[b * n + i] = gradp[b * n + i]
            n_iter[b] = Int32(k)
            if stop:
                active[b] = False
                continue
            var e = endv[b]
            for i in range(n):
                # `axpy(S[end], -1, xp, x)` and `axpy(Y[end], -1, gradp, grad)`
                S[(b * m + e) * n + i] = ftz(
                    identical_mul_add(Float32(-1.0), xp[b * n + i], x[b * n + i])
                )
                Y[(b * m + e) * n + i] = ftz(
                    identical_mul_add(
                        Float32(-1.0), gradp[b * n + i], grad[b * n + i]
                    )
                )
            var nv = n_vec[b]
            endv[b] = lbfgs_search_dir_at(
                param, nv, e, S, Y, grad, drt, yhist, alpha, b, n
            )
            n_vec[b] = nv
            step[b] = Float32(1.0)

        if trace.enabled:
            var tag = _iter_tag(k)
            trace.record_list_f32(tag + ".x", x)
            trace.record_list_f32(tag + ".loss", fx)
            trace.record_list_f32(tag + ".grad", grad)
            var ls = List[Int32]()
            for b in range(batch_size):
                ls.append(Int32(lsret[b]))
            for b in range(batch_size):
                ls.append(Int32(ls_iters[b]))
            trace.record_list_i32(tag + ".ls", ls)
        k += 1

    trace.record_list_f32("fit.x", x)
    trace.record_list_f32("fit.loss", fx)
    trace.record_list_i32("fit.n_iter", n_iter)
    trace.record_list_i32("fit.retcode", retcode)

    _ = d_x^
    _ = d_grad^
    _ = d_x_pert^
    _ = scratch^
    return BatchedLBFGSResult(
        x=x^, fx=fx^, n_iter=n_iter^, retcode=retcode^, n_eval=n_eval
    )


# ---------------------------------------------------------------------------
# fit (arima.pyx:860-958)
# ---------------------------------------------------------------------------


@fieldwise_init
struct FitResult(Movable):
    """What a fit produced, and the evidence for it.

    `x` is the UNCONSTRAINED optimum, the coordinates the optimizer works
    in; `t_x` is the packed FITTED model, forward-transformed, and it is
    what `predict` must be handed (`predict` is called with `trans = false`,
    `batched_arima.cu:175`). `params` is written in place with the same
    values unpacked.

    `x0` is kept because a fit that goes wrong is nearly always a fit that
    started wrong, and `estimate_x0` is the half with no upstream oracle."""

    var x: List[Float32]
    var t_x: List[Float32]
    var x0: List[Float32]
    var fx: List[Float32]
    var n_iter: List[Int32]
    var retcode: List[Int32]
    var n_eval: Int
    var start: StartParamsResult


def batched_fit(
    ctx: DeviceContext,
    mut d_y: DeviceBuffer[DType.float32],
    batch_size: Int,
    n_obs: Int,
    order: ARIMAOrder,
    mut params: ARIMAParams,
    mut trace: IdentityTrace,
    max_iterations: Int = 1000,
    h: Float32 = ARIMA_FIT_H,
) raises -> FitResult:
    """`ARIMA.fit` (`arima.pyx:860-958`) with `method = "ml"`,
    `start_params = None`, `simple_differencing = True` and no exog, which
    is every arm this lane can reach. `method = "css"` and `"css-ml"` are
    refused by name (the CSS log-likelihood is not ported); a caller-supplied
    `start_params` is not offered, because `set_fit_params` has no door here
    yet.

    Their five steps, in their order (`:938-957`):

        1. `_estimate_x0()`                    -> params
        2. `x0 = _batched_transform(pack(), True)`   INVERSE Jones
        3. `fit_helper(x0, "ml")`              -> x
        4. `_batched_transform(x)`             FORWARD Jones
        5. `unpack(...)`                       -> params

    `params` is BOTH the scratch step 1 writes and the output step 5 fills,
    which is theirs (`self`'s own parameter arrays play both roles).

    STEP 2 IS WHY DEVIATION 675's INVERSE HALF IS NO LONGER OFF THE PORTED
    PATH. Until this function existed, nothing in the lane called
    `two_atanh`; `arima/README.md` recorded that and said the decision to
    accept `identical_log` rather than land `identical_log1p` inverts the
    day the optimizer lands. It has landed and the decision is REAFFIRMED,
    for a different and better reason, which is written out in that file's
    DEVIATION 675 section and gated by
    `check_jones_inverse_is_below_the_fd_step`."""
    validate_order(order)
    if n_obs < 2:
        raise Error("batched_fit: n_obs must be at least 2 (got " + String(n_obs) + ")")
    # ONCE, here, rather than on every one of the hundreds of evaluations
    # the optimizer makes; see `eval_batch`'s `check_finite = False`.
    _refuse_non_finite(ctx, d_y, batch_size * n_obs, "y")

    # 1. the starting parameters
    var start = estimate_x0(ctx, params, d_y, batch_size, n_obs, order)
    trace.record_device[DType.float32](ctx, "fit.x0.sigma2", params.sigma2, batch_size)
    if order.p != 0:
        trace.record_device[DType.float32](ctx, "fit.x0.ar", params.ar, order.p * batch_size)
    if order.q != 0:
        trace.record_device[DType.float32](ctx, "fit.x0.ma", params.ma, order.q * batch_size)
    if order.P != 0:
        trace.record_device[DType.float32](ctx, "fit.x0.sar", params.sar, order.P * batch_size)
    if order.Q != 0:
        trace.record_device[DType.float32](ctx, "fit.x0.sma", params.sma, order.Q * batch_size)
    if order.k != 0:
        trace.record_device[DType.float32](ctx, "fit.x0.mu", params.mu, batch_size)
    # THE DECISION STAGES. `info_ls` is which series the least squares
    # refused and at which column (negative for the AR pre-fit); `invparams`
    # is `test_invparams`' verdict, one byte per series. Neither is
    # derivable from any float stage: a series whose AR block was zeroed by
    # `test_invparams` and a series whose AR block was genuinely estimated
    # at zero look the same in `fit.x0.ar`.
    if start.ns_run:
        trace.record_device[DType.int32](ctx, "fit.x0.ns.info_ls", start.ns.info, batch_size)
        trace.record_device[DType.uint8](ctx, "fit.x0.ns.invparams", start.ns.verdict, batch_size)
    if start.seasonal_run:
        trace.record_device[DType.int32](ctx, "fit.x0.sea.info_ls", start.seasonal.info, batch_size)
        trace.record_device[DType.uint8](ctx, "fit.x0.sea.invparams", start.seasonal.verdict, batch_size)

    # 2. into the unconstrained coordinates the optimizer works in
    var N = order.complexity()
    var inv = ARIMAParams(ctx, order, batch_size)
    batched_jones_transform(ctx, order, batch_size, True, params, inv)
    var d_x0 = ctx.enqueue_create_buffer[DType.float32](N * batch_size)
    pack(ctx, inv, order, batch_size, d_x0)
    ctx.synchronize()
    var x0 = _download(ctx, d_x0, N * batch_size)
    trace.record_list_f32("fit.x0", x0)
    # `:921-923`: "Initial parameter vector x has NaN or Inf."
    for i in range(N * batch_size):
        if not isfinite(x0[i]):
            raise Error(
                "batched_fit: the initial parameter vector has a non-finite"
                " value at index " + String(i)
                + "; estimate_x0 produced a parameter the inverse Jones"
                " transform could not map (arima/README.md, DEVIATION 678)"
            )

    # 3. difference ONCE, then optimize on the differenced series
    var diff = order.need_diff()
    var n_obs_kf = n_obs - order.n_diff() if diff else n_obs
    var order_kf = order.without_diff() if diff else order
    var y_kf = ctx.enqueue_create_buffer[DType.float32](n_obs_kf * batch_size)
    if diff:
        prepare_data(ctx, y_kf, d_y, batch_size, n_obs, order.d, order.D, order.s)
    else:
        ctx.enqueue_copy(
            dst_buf=y_kf,
            src_buf=d_y.create_sub_buffer[DType.float32](0, n_obs * batch_size),
        )
    ctx.synchronize()
    # `n_obs - 1` uses the ORIGINAL length, not the differenced one
    # (`arima.pyx:910`, `:918`: `self.n_obs`)
    var res = batched_min_lbfgs(
        ctx, y_kf, batch_size, n_obs_kf, Float32(n_obs - 1), order_kf, x0,
        arima_fit_params(max_iterations), h, trace,
    )

    # 4 and 5. forward-transform the answer and unpack it into `params`
    var d_x = ctx.enqueue_create_buffer[DType.float32](N * batch_size)
    _upload(ctx, d_x, res.x)
    var raw = ARIMAParams(ctx, order, batch_size)
    unpack(ctx, raw, order, batch_size, d_x)
    batched_jones_transform(ctx, order, batch_size, False, raw, params)
    var d_t_x = ctx.enqueue_create_buffer[DType.float32](N * batch_size)
    pack(ctx, params, order, batch_size, d_t_x)
    ctx.synchronize()
    var t_x = _download(ctx, d_t_x, N * batch_size)
    trace.record_list_f32("fit.t_x", t_x)

    _ = inv^
    _ = raw^
    _ = d_x0^
    _ = d_x^
    _ = d_t_x^
    _ = y_kf^
    return FitResult(
        x=res.x.copy(), t_x=t_x^, x0=x0^, fx=res.fx.copy(),
        n_iter=res.n_iter.copy(), retcode=res.retcode.copy(),
        n_eval=res.n_eval, start=start^,
    )
