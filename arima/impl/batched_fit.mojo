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
SHAPE is theirs and is kept -- one state machine per series, ONE BATCHED
DEVICE EVALUATION per candidate point -- and the ALGORITHM is cuML's own
L-BFGS, already implemented in `glm/impl/qn/`, re-spelled per series in
`arima/impl/lbfgs_host.mojo` (the host column) and
`arima/impl/lbfgs_device.mojo` (the same arithmetic, one GPU thread per
series, the state resident on the device; cpu-gpu-cleanup n-seq). Read that file's banner for why calling
`glm::min_lbfgs` B times is not the answer; the short version is that it is
typed on a concrete `GLMWithData`, it evaluates the objective itself from
inside the line search, and its vector work is device reductions sized for
`n` in the millions where ours is `n <= 20`.

WHAT IS NOT scipy's, AND IS NOT PRETENDED TO BE. scipy's L-BFGS-B is not
cuML's L-BFGS: it is a different line search (More-Thuente / `dcsrch`
inside `mainlb`), a different history update and different stopping
constants. Substituting cuML's own already-implemented solver is a REAL
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
value cannot be carried across DEVIATION 670 and there is no reference
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

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.memory import unsafe_memcpy

from arima.impl.batched_arima import _refuse_non_finite
from arima.impl.batched_kalman import ARIMA_EVAL_WS_EXOG, KALMAN_FAST_EVAL_WS, eval_ws_fits
from arima.impl.estimate_x0 import StartParamsResult, estimate_x0_x
from arima.impl.fast_eval_ws import FastEvalWS
from arima.impl.fast_order_state import ARIMA_ORDER_BATCH, order_min_lbfgs
from arima.impl.eval_batch import arima_eval
from arima.impl.fast_lbfgs_async import ARIMA_ASYNC_ALL, AsyncLBFGSOut, async_min_lbfgs
from arima.impl.lbfgs_device import (
    LBFGS_TPB,
    lbfgs_accept_kernel,
    lbfgs_candidate_kernel,
    lbfgs_init_kernel,
    lbfgs_prelude_kernel,
    lbfgs_verdict_kernel,
)
from arima.impl.timeSeries.arima_helpers import batched_jones_transform
from arima.impl.tsa.arima_common import (
    ARIMAOrder,
    ARIMAParams,
    pack,
    unpack,
    validate_order,
)
from core.identity_trace import IdentityTrace
from glm.impl.qn.qn_util import LBFGS_LS_BT_ARMIJO, LBFGSParam
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
# host <-> device for the packed parameter vector (the fit's x0 down and its
# answer up, once each)
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


# ---------------------------------------------------------------------------
# the objective (`eval_batch_device`, `arima_eval`) lives in
# `arima/impl/eval_batch.mojo` (lane n10-dictl-arima-loops, 2026-11-09)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# the batched L-BFGS
# ---------------------------------------------------------------------------


@fieldwise_init
struct BatchedLBFGSResult(Movable):
    var x: List[Float32]
    var fx: List[Float32]
    var n_iter: List[Int32]
    var retcode: List[Int32]
    var n_eval: Int


# TOMBSTONE: MOJOLEARN_IDN_ARIMA_ASYNC_OFF, MOJOLEARN_ARIMA_FAST_ASYNC_OFF (superseded: the lock-step device solver they kept) deleted 2026-11-09 by lane/n10-dictl-arima-loops; code recoverable at the lane's base f59d913b0b.
# Restore: git apply experiments/removed/ARIMA_LOCKSTEP_LBFGS.patch
# (the lock-step loop, `_read_flag`, `_record_iter`, `_iter_tag`,
# `_download_i32` and its per-iteration `fit.init.*` / `fit.iterNNNN.*` card,
# which nothing read)
# Restored 2026-11-09 by lane/r2-restore-unmeasured-defaults: the lock-step
# loop (`_lockstep_min_lbfgs`), `_read_flag` and `_download_i32` are back as the
# default wherever the fit holds no evaluation workspace (the routes that took
# it before the deletion; the per-series solver was never measured there).
# The per-iteration card (`_record_iter`, `_iter_tag`, `fit.init.*`,
# `fit.iterNNNN.*`; nothing read it) stays deleted, and the evaluation keeps
# the deletion's drain-free `eval_batch.mojo::arima_eval` on both solvers.


def _download_i32(ctx: DeviceContext, buf: DeviceBuffer[DType.int32], n: Int) raises -> List[Int32]:
    var h = ctx.enqueue_create_host_buffer[DType.int32](n if n > 0 else 1)
    if n > 0:
        var view = buf.create_sub_buffer[DType.int32](0, n)
        ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=view)
    ctx.synchronize()
    var out = List[Int32](length=n, fill=Int32(0))
    if n > 0:
        unsafe_memcpy(dest=out.unsafe_ptr(), src=h.unsafe_ptr(), count=n)
    _ = h^
    return out^


def _read_flag(ctx: DeviceContext, mut flag: DeviceBuffer[DType.int32], mut host: HostBuffer[DType.int32]) raises -> Bool:
    """The one control word a step reads back: whether any series is still
    active (or still searching). One 4-byte copy and its wait."""
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=flag)
    ctx.synchronize()
    return host.unsafe_ptr()[0] != 0


def batched_min_lbfgs(
    ctx: DeviceContext,
    mut d_y_kf: DeviceBuffer[DType.float32],
    mut d_exog_kf: DeviceBuffer[DType.float32],
    batch_size: Int,
    n_obs_kf: Int,
    scale: Float32,
    order_kf: ARIMAOrder,
    x0: List[Float32],
    param: LBFGSParam,
    h: Float32,
    mut trace: IdentityTrace,
) raises -> BatchedLBFGSResult:
    """B independent L-BFGS solvers, every series' state ON THE DEVICE, one
    thread per series, sharing ONE batched device evaluation per round. Two
    device arms: the per-series solver
    (`fast_lbfgs_async.mojo::async_min_lbfgs`, described below) when the fit
    holds the stacked evaluation workspace (FAST on Apple without exog,
    IDENTICAL within `IDN_EVAL_WS_MAX_CELLS`), and the lock-step solver
    (`_lockstep_min_lbfgs`) otherwise (FAST on NVIDIA / AMD, FAST on Apple
    with exog, IDENTICAL above the bound); `-D MOJOLEARN_ARIMA_ASYNC_ALL`
    takes the per-series solver everywhere. Both give each series the same
    iterates, `n_iter` and `retcode`.

    THE SHAPE. Every still-running series proposes its own candidate, all B
    candidates go into one evaluation (`eval_batch.mojo::arima_eval`: the
    held stacked workspace when the fit has one, else the sequential Kalman
    passes; the Kalman filter is the body's kernel either way), and each
    series then takes its own Armijo test, verdict, history update and next
    direction in its own thread. A series that has stopped still proposes its
    current `x` and its result is discarded, so the batch composition and the
    launch geometry are a function of the fixture alone. Each series'
    iterates, `n_iter` and `retcode` are the lock-step solver's bit for bit and equal the host column's
    (`arima_oracle._batched_min_lbfgs`); the host reads one word every
    `ASYNC_READ_EVERY` rounds, never per line-search step. The identity
    trace records the answer (`fit.x`, `fit.loss`, `fit.n_iter`,
    `fit.retcode`)."""
    if param.check_param() != 0:
        raise Error(
            "batched_min_lbfgs: invalid parameter (check_param code "
            + String(param.check_param()) + ")"
        )
    var n = order_kf.complexity()
    var bs = batch_size
    # lane/apple-fast-tsa: the stacked evaluation's buffers, held for the
    # whole solve (`arima/impl/fast_eval_ws.mojo`); `None` when the build or
    # the fit's size keeps the sequential evaluation (same bits).
    var ews = Optional[FastEvalWS]()
    comptime if KALMAN_FAST_EVAL_WS:
        # IDENTICAL on every vendor since lane/fam-timeseries
        # (IDN_ARIMA_EVAL_WS, `batched_kalman.mojo`); `eval_ws_fits` is
        # always true under FAST and a size bound under IDENTICAL.
        # lane/fam2-timeseries (ARIMA_EVAL_WS_EXOG): exog fits take it too.
        if (order_kf.n_exog == 0 or ARIMA_EVAL_WS_EXOG) and eval_ws_fits((n + 1) * bs, n_obs_kf):
            var held = FastEvalWS(ctx, d_y_kf, bs, n_obs_kf, order_kf)
            if order_kf.n_exog != 0:
                held.attach_exog(d_exog_kf, order_kf.n_exog)
            ews = held^
    comptime if ARIMA_ORDER_BATCH:
        if ews:
            var ao = order_min_lbfgs(ctx, ews.value(), bs, scale, order_kf, x0, param, h)
            _ = ews^
            return _answer(trace, ao)
    # The per-series solver where the fit holds the evaluation workspace (its
    # measured default: M3 A/B autoarima taxi-hourly 39,639 -> 24,563 ms);
    # the lock-step solver everywhere else, as before lane
    # n10-dictl-arima-loops (never measured against the per-series one on
    # those routes). `-D MOJOLEARN_ARIMA_ASYNC_ALL` takes the per-series solver
    # on every route: the A/B's B arm.
    if Bool(ews) or ARIMA_ASYNC_ALL:
        var ar = async_min_lbfgs(ctx, ews, d_y_kf, d_exog_kf, n_obs_kf, bs, scale, order_kf, x0, param, h)
        _ = ews^
        return _answer(trace, ar)
    return _lockstep_min_lbfgs(
        ctx, ews, d_y_kf, d_exog_kf, bs, n_obs_kf, scale, order_kf, x0, param, h, trace
    )


def _lockstep_min_lbfgs(
    ctx: DeviceContext,
    mut ews: Optional[FastEvalWS],
    mut d_y_kf: DeviceBuffer[DType.float32],
    mut d_exog_kf: DeviceBuffer[DType.float32],
    batch_size: Int,
    n_obs_kf: Int,
    scale: Float32,
    order_kf: ARIMAOrder,
    x0: List[Float32],
    param: LBFGSParam,
    h: Float32,
    mut trace: IdentityTrace,
) raises -> BatchedLBFGSResult:
    """The lock-step solver (`arima/impl/lbfgs_device.mojo`'s kernels), one
    thread per series, sharing ONE batched device evaluation per candidate
    point (cpu-gpu-cleanup n-seq, 2026-10-02; restored 2026-11-09 by lane
    r2-restore-unmeasured-defaults).

    THE SHAPE. The OUTER loop is `k`, the L-BFGS iteration. Inside it, the
    LINE SEARCH is also a shared loop: at step `t` every still-searching
    series proposes its own candidate `xp + step_b * drt_b` with its OWN step
    length, all B candidates go into one `d_x`, and ONE evaluation
    (`eval_batch.mojo::arima_eval`) takes them together. Each series then
    applies the Armijo test to its own result and either accepts or halves
    its own step; the loop runs until none is still searching, at most
    `param.max_linesearch` times. A series that is not searching still
    proposes its current `x` and its result is discarded, so the batch
    composition and the launch geometry are a function of the fixture alone.

    The host sequences launches and reads one word per step: whether any
    series is still active (per iteration) or still searching (per
    line-search step), raised by the kernels with an atomic max. Each
    series' arithmetic is its own thread's, so its iterates, `n_iter` and
    `retcode` equal the per-series solver's and the host column's
    (`arima_oracle._batched_min_lbfgs`)."""
    var n = order_kf.complexity()
    var bs = batch_size
    var b_n = bs * n
    var m = param.m
    var past = param.past if param.past > 0 else 0
    var grid = (bs + LBFGS_TPB - 1) // LBFGS_TPB

    # device workspace, allocated ONCE for the whole solve
    var d_x = ctx.enqueue_create_buffer[DType.float32](max(1, b_n))
    var d_grad = ctx.enqueue_create_buffer[DType.float32](max(1, b_n))
    var d_x_pert = ctx.enqueue_create_buffer[DType.float32](max(1, b_n))
    var scratch = ARIMAParams(ctx, order_kf, bs)
    var x = ctx.enqueue_create_buffer[DType.float32](max(1, b_n))
    var xp = ctx.enqueue_create_buffer[DType.float32](max(1, b_n))
    var grad = ctx.enqueue_create_buffer[DType.float32](max(1, b_n))
    var gradp = ctx.enqueue_create_buffer[DType.float32](max(1, b_n))
    var gradc = ctx.enqueue_create_buffer[DType.float32](max(1, b_n))
    var drt = ctx.enqueue_create_buffer[DType.float32](max(1, b_n))
    var S = ctx.enqueue_create_buffer[DType.float32](max(1, b_n * m))
    var Y = ctx.enqueue_create_buffer[DType.float32](max(1, b_n * m))
    var yhist = ctx.enqueue_create_buffer[DType.float32](max(1, bs * m))
    var alpha = ctx.enqueue_create_buffer[DType.float32](max(1, bs * m))
    var fx_hist = ctx.enqueue_create_buffer[DType.float32](max(1, bs * (past if past > 0 else 1)))
    var fx = ctx.enqueue_create_buffer[DType.float32](max(1, bs))
    var fxc = ctx.enqueue_create_buffer[DType.float32](max(1, bs))
    var fxp = ctx.enqueue_create_buffer[DType.float32](max(1, bs))
    var fx_init = ctx.enqueue_create_buffer[DType.float32](max(1, bs))
    var dg_init = ctx.enqueue_create_buffer[DType.float32](max(1, bs))
    var dg_test = ctx.enqueue_create_buffer[DType.float32](max(1, bs))
    var gnorm = ctx.enqueue_create_buffer[DType.float32](max(1, bs))
    var step = ctx.enqueue_create_buffer[DType.float32](max(1, bs))
    var active = ctx.enqueue_create_buffer[DType.int32](max(1, bs))
    var searching = ctx.enqueue_create_buffer[DType.int32](max(1, bs))
    var endv = ctx.enqueue_create_buffer[DType.int32](max(1, bs))
    var n_vec = ctx.enqueue_create_buffer[DType.int32](max(1, bs))
    var lsret = ctx.enqueue_create_buffer[DType.int32](max(1, bs))
    var ls_iters = ctx.enqueue_create_buffer[DType.int32](max(1, bs))
    var n_iter = ctx.enqueue_create_buffer[DType.int32](max(1, bs))
    var retcode = ctx.enqueue_create_buffer[DType.int32](max(1, bs))
    var bad = ctx.enqueue_create_buffer[DType.int32](max(1, bs))
    var any_active = ctx.enqueue_create_buffer[DType.int32](1)
    var any_searching = ctx.enqueue_create_buffer[DType.int32](1)
    var flag_host = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_memset(xp, Float32(0.0))
    ctx.enqueue_memset(gradp, Float32(0.0))
    ctx.enqueue_memset(gradc, Float32(0.0))
    ctx.enqueue_memset(drt, Float32(0.0))
    ctx.enqueue_memset(S, Float32(0.0))
    ctx.enqueue_memset(Y, Float32(0.0))
    ctx.enqueue_memset(yhist, Float32(0.0))
    ctx.enqueue_memset(alpha, Float32(0.0))
    ctx.enqueue_memset(fx_hist, Float32(0.0))
    ctx.enqueue_memset(fxc, Float32(0.0))
    ctx.enqueue_memset(fxp, Float32(0.0))
    ctx.enqueue_memset(fx_init, Float32(0.0))
    ctx.enqueue_memset(dg_init, Float32(0.0))
    ctx.enqueue_memset(dg_test, Float32(0.0))
    ctx.enqueue_memset(gnorm, Float32(0.0))
    ctx.enqueue_memset(step, Float32(0.0))
    ctx.enqueue_memset(active, Int32(0))
    # x0 up ONCE; `x` is the state, `d_x` the evaluation's candidates
    _upload(ctx, x, x0)
    ctx.enqueue_copy(dst_buf=d_x, src_buf=x)

    # `min_lbfgs:161-173`: evaluate at x0, and exit early per series if it
    # is already a minimizer.
    arima_eval(
        ctx, ews, d_y_kf, d_exog_kf, bs, n_obs_kf, order_kf, d_x, d_grad, d_x_pert,
        scratch, h, scale, fx, grad, bad,
    )
    var n_eval = 1
    ctx.enqueue_memset(any_active, Int32(0))
    ctx.enqueue_function[lbfgs_init_kernel](
        grad.unsafe_ptr(), drt.unsafe_ptr(), fx.unsafe_ptr(), fxp.unsafe_ptr(),
        fx_hist.unsafe_ptr(), gnorm.unsafe_ptr(), step.unsafe_ptr(),
        active.unsafe_ptr(), searching.unsafe_ptr(), endv.unsafe_ptr(), n_vec.unsafe_ptr(),
        lsret.unsafe_ptr(), ls_iters.unsafe_ptr(), n_iter.unsafe_ptr(), retcode.unsafe_ptr(),
        any_active.unsafe_ptr(),
        Int32(bs), Int32(n), Int32(past), param.epsilon, param.delta,
        grid_dim=(grid, 1, 1), block_dim=(LBFGS_TPB, 1, 1),
    )

    var k = 1
    while k <= param.max_iterations:  # driver-loop: over the L-BFGS iterations (one flag word read each)
        if not _read_flag(ctx, any_active, flag_host):
            break

        # `min_lbfgs:188-191` and `ls_backtrack:100-108`
        ctx.enqueue_memset(any_searching, Int32(0))
        ctx.enqueue_function[lbfgs_prelude_kernel](
            x.unsafe_ptr(), xp.unsafe_ptr(), grad.unsafe_ptr(), gradp.unsafe_ptr(),
            drt.unsafe_ptr(), fx.unsafe_ptr(), fxp.unsafe_ptr(), fx_init.unsafe_ptr(),
            dg_init.unsafe_ptr(), dg_test.unsafe_ptr(), step.unsafe_ptr(),
            active.unsafe_ptr(), searching.unsafe_ptr(), lsret.unsafe_ptr(),
            ls_iters.unsafe_ptr(), any_searching.unsafe_ptr(),
            Int32(bs), Int32(n), param.ftol,
            grid_dim=(grid, 1, 1), block_dim=(LBFGS_TPB, 1, 1),
        )

        # THE SHARED LINE SEARCH (`ls_backtrack:109-121`, B at a time)
        for _t in range(param.max_linesearch):  # driver-loop: over the shared line-search steps (one flag word read each)
            if not _read_flag(ctx, any_searching, flag_host):
                break
            ctx.enqueue_memset(any_searching, Int32(0))
            ctx.enqueue_function[lbfgs_candidate_kernel](
                d_x.unsafe_ptr(), x.unsafe_ptr(), xp.unsafe_ptr(), drt.unsafe_ptr(),
                step.unsafe_ptr(), searching.unsafe_ptr(), Int32(bs), Int32(n),
                grid_dim=(grid, 1, 1), block_dim=(LBFGS_TPB, 1, 1),
            )
            arima_eval(
                ctx, ews, d_y_kf, d_exog_kf, bs, n_obs_kf, order_kf, d_x, d_grad, d_x_pert,
                scratch, h, scale, fxc, gradc, bad,
            )
            n_eval += 1
            ctx.enqueue_function[lbfgs_accept_kernel](
                x.unsafe_ptr(), grad.unsafe_ptr(), fx.unsafe_ptr(), d_x.unsafe_ptr(),
                gradc.unsafe_ptr(), fxc.unsafe_ptr(), fx_init.unsafe_ptr(),
                dg_test.unsafe_ptr(), step.unsafe_ptr(), searching.unsafe_ptr(),
                lsret.unsafe_ptr(), ls_iters.unsafe_ptr(), any_searching.unsafe_ptr(),
                Int32(bs), Int32(n), param.min_step, param.max_step, param.ls_dec,
                grid_dim=(grid, 1, 1), block_dim=(LBFGS_TPB, 1, 1),
            )

        # `min_lbfgs:197-222`: verdict, history update, new direction
        ctx.enqueue_memset(any_active, Int32(0))
        ctx.enqueue_function[lbfgs_verdict_kernel](
            x.unsafe_ptr(), xp.unsafe_ptr(), grad.unsafe_ptr(), gradp.unsafe_ptr(),
            drt.unsafe_ptr(), S.unsafe_ptr(), Y.unsafe_ptr(), yhist.unsafe_ptr(),
            alpha.unsafe_ptr(), fx.unsafe_ptr(), fxp.unsafe_ptr(), fx_hist.unsafe_ptr(),
            gnorm.unsafe_ptr(), step.unsafe_ptr(), active.unsafe_ptr(), endv.unsafe_ptr(),
            n_vec.unsafe_ptr(), lsret.unsafe_ptr(), n_iter.unsafe_ptr(), retcode.unsafe_ptr(),
            any_active.unsafe_ptr(),
            Int32(bs), Int32(n), Int32(m), Int32(past), Int32(k),
            param.epsilon, param.delta, param.ftol,
            grid_dim=(grid, 1, 1), block_dim=(LBFGS_TPB, 1, 1),
        )

        k += 1

    # the answer comes down ONCE
    var x_out = _download(ctx, x, b_n)
    var fx_out = _download(ctx, fx, bs)
    var n_iter_out = _download_i32(ctx, n_iter, bs)
    var retcode_out = _download_i32(ctx, retcode, bs)
    trace.record_list_f32("fit.x", x_out)
    trace.record_list_f32("fit.loss", fx_out)
    trace.record_list_i32("fit.n_iter", n_iter_out)
    trace.record_list_i32("fit.retcode", retcode_out)

    _ = d_x^
    _ = d_grad^
    _ = d_x_pert^
    _ = scratch^
    _ = x^
    _ = xp^
    _ = grad^
    _ = gradp^
    _ = gradc^
    _ = drt^
    _ = S^
    _ = Y^
    _ = yhist^
    _ = alpha^
    _ = fx_hist^
    _ = fx^
    _ = fxc^
    _ = fxp^
    _ = fx_init^
    _ = dg_init^
    _ = dg_test^
    _ = gnorm^
    _ = step^
    _ = active^
    _ = searching^
    _ = endv^
    _ = n_vec^
    _ = lsret^
    _ = ls_iters^
    _ = n_iter^
    _ = retcode^
    _ = bad^
    _ = any_active^
    _ = any_searching^
    _ = flag_host^
    return BatchedLBFGSResult(
        x=x_out^, fx=fx_out^, n_iter=n_iter_out^, retcode=retcode_out^, n_eval=n_eval
    )

def _answer(mut trace: IdentityTrace, ar: AsyncLBFGSOut) raises -> BatchedLBFGSResult:
    """The solve's answer, recorded on the identity trace."""
    trace.record_list_f32("fit.x", ar.x)
    trace.record_list_f32("fit.loss", ar.fx)
    trace.record_list_i32("fit.n_iter", ar.n_iter)
    trace.record_list_i32("fit.retcode", ar.retcode)
    return BatchedLBFGSResult(
        x=ar.x.copy(), fx=ar.fx.copy(), n_iter=ar.n_iter.copy(),
        retcode=ar.retcode.copy(), n_eval=ar.n_eval,
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
    started wrong, and `estimate_x0` is the half with no reference oracle."""

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
    """The `n_exog = 0` door the checks call: `batched_fit_x` with a
    placeholder it does not read."""
    if order.n_exog != 0:
        raise Error(
            "batched_fit: n_exog=" + String(order.n_exog)
            + " needs the exogenous series; call batched_fit_x"
        )
    var e0 = ctx.enqueue_create_buffer[DType.float32](1)
    var r = batched_fit_x(ctx, d_y, e0, batch_size, n_obs, order, params, trace, max_iterations, h)
    _ = e0^
    return r^


def batched_fit_x(
    ctx: DeviceContext,
    mut d_y: DeviceBuffer[DType.float32],
    mut d_exog: DeviceBuffer[DType.float32],
    batch_size: Int,
    n_obs: Int,
    order: ARIMAOrder,
    mut params: ARIMAParams,
    mut trace: IdentityTrace,
    max_iterations: Int = 1000,
    h: Float32 = ARIMA_FIT_H,
) raises -> FitResult:
    """`ARIMA.fit` (`arima.pyx:860-958`) with `method = "ml"`,
    `start_params = None` and `simple_differencing = True`, which is every
    arm this lane can reach. `d_exog` holds the regressors in the filter's
    layout over the `n_obs` observations (a placeholder when `n_exog = 0`);
    they are differenced once beside `y` (`arima.pyx:430-436`) and every
    likelihood evaluation reads the differenced copy. `method = "css"` and `"css-ml"` are
    refused by name (the CSS log-likelihood is not implemented); a caller-supplied
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

    STEP 2 IS WHY DEVIATION 675's INVERSE HALF IS NO LONGER OFF THE IMPLEMENTED
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
    var exog_info = ctx.enqueue_create_buffer[DType.int32](max(1, batch_size))
    var start = estimate_x0_x(ctx, params, d_y, d_exog, batch_size, n_obs, order, exog_info)
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
    if order.n_exog != 0:
        # The exog regression's coefficients and its DECISION stage: which
        # series' solve refused (beta zeroed) and at which column.
        trace.record_device[DType.float32](ctx, "fit.x0.beta", params.beta, order.n_exog * batch_size)
        trace.record_device[DType.int32](ctx, "fit.x0.exog.info_ls", exog_info, batch_size)
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
    var n_ser = order.n_exog * batch_size
    var exog_kf = ctx.enqueue_create_buffer[DType.float32](max(1, n_obs_kf * n_ser))
    if n_ser > 0:
        if diff:
            prepare_data(ctx, exog_kf, d_exog, n_ser, n_obs, order.d, order.D, order.s)
        else:
            ctx.enqueue_copy(
                dst_buf=exog_kf,
                src_buf=d_exog.create_sub_buffer[DType.float32](0, n_obs * n_ser),
            )
    ctx.synchronize()
    # `n_obs - 1` uses the ORIGINAL length, not the differenced one
    # (`arima.pyx:910`, `:918`: `self.n_obs`)
    var res = batched_min_lbfgs(
        ctx, y_kf, exog_kf, batch_size, n_obs_kf, Float32(n_obs - 1), order_kf, x0,
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
    _ = exog_kf^
    _ = exog_info^
    _ = y_kf^
    return FitResult(
        x=res.x.copy(), t_x=t_x^, x0=x0^, fx=res.fx.copy(),
        n_iter=res.n_iter.copy(), retcode=res.retcode.copy(),
        n_eval=res.n_eval, start=start^,
    )
