# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The batched ARIMA fit, predict and forecast on the host, a SECOND spelling
of the device lane (workstream E, the arima, arima-011 and arima-seasonal-c
lanes, 2026-09-14).

HOST ONLY. Nothing here imports `max.gpu`, `max.gpu`, a `DeviceContext` or
any module under `arima/`. The only library imports are the
`checks/numerics.mojo` seams (`ftz`, `identical_mul_add`, `identical_exp`,
`identical_log`, `identical_sqrt`). Every construct the device path reaches
is RESTATED below with the file and line of the routine it MIRRORS, so a
disagreement between the two is a finding and not a shared bug. The
existing oracles under `arima/checks/` are not imported either: they share
`param_to_poly` and the matrix host replays with the device files, and a
host binding must not move when a device helper is edited.

WHAT IS MIRRORED, IN THE ORDER `arima_fit_ptr_host`
(`arima/estimator.mojo:282-401`) REACHES IT

  1. `batched_fit` (`arima/impl/batched_fit.mojo:586-717`): the non-finite
     refusal (`batched_arima.mojo:149-169`), then
  2. `estimate_x0` (`arima/impl/estimate_x0.mojo:591-618`): `prepare_data`
     (`tsa/impl/timeSeries/arima_helpers.mojo:32-76`, the one-difference
     kernel `tsa/impl/linalg/batched/matrix.mojo:28-46` or the copy),
     `start_params` (`:540-588`, its three arms), `arma_least_squares`
     (`:445-513`, the degenerate arm `:214-238`, the LS_MAX_COLS refusal)
     and its per-series kernel (`:241-428`: the AR pre-fit, the lagged
     residual, the intercept and lag columns, the target, the ARMA solve,
     the sigma2 fold starting at q, `test_invparams` `:171-206` with its ONE
     rounding association), over `householder_qr_solve`
     (`arima/impl/linalg/batched/least_squares.mojo:111-210`, DEVIATION
     678's reflector sign, folds and rank test);
  3. the INVERSE Jones transform and the sigma2 floor
     (`arima/impl/timeSeries/arima_helpers.mojo:202-230`,
     `jones_transform.mojo:129-183` with DEVIATION 675's `tanh_half` and
     `two_atanh` `:97-120`), `pack` (`arima_common.mojo:162-201`) and the
     initial-vector finiteness refusal (`batched_fit.mojo:666-673`);
  4. the differencing again (`batched_fit.mojo:676-687`) and
     `batched_min_lbfgs` (`:290-555`) with `arima_fit_params` (`:151-175`):
     the shared line search at one batched evaluation per candidate, the
     non-searching series proposing its own `x`, and the per-series rules of
     `arima/impl/lbfgs_host.mojo` (`nrm_max_at` `:84-92`, `dot_at` `:95-100`,
     `nrm2_at` `:103-105`, `armijo_ok` `:113-126`, `check_convergence_at`
     `:129-151`, `lbfgs_verdict` `:154-201`, `lbfgs_search_dir_at`
     `:209-269`), restated here with the glm constants they read
     (`glm/impl/qn/qn_util.mojo:56-77`);
  5. `eval_batch` (`batched_fit.mojo:219-264`) over `batched_loglike_grad`
     (`batched_arima.mojo:413-469`: the base pass, the perturbation
     `ftz(ftz(x) + h)` of `:357-371`, the gradient `:396-410`, the reset by
     COPY of `:374-393`), each pass `batched_loglike` (`:104-146`) through
     `unpack` (`arima_common.mojo:204-241`), the forward Jones transform and
     `batched_kalman_filter` (`arima/impl/batched_kalman.mojo:722-789`): the
     matrices kernel (`:149-235`, `param_to_poly`, `reduced_polynomial` and
     `reduced_poly_indices` of `arima_helpers.mojo:49-73`), the initial
     state kernel (`:243-373`, `kron_minus_identity` and `lu_inverse` of
     `arima/impl/linalg/batched/matrix.mojo:79-180`, the matvec into a
     local, the r == 1 intercept nudge by sign bit), the two refusals by
     name, and the loop kernel (`:425-604`, `_mv`, `_mm`,
     `_numerical_stability` `:382-422`, the log-likelihood and the forecast
     loop);
  6. the forward transform, `pack` into `t_x`, and `_loglike_at`
     (`arima/estimator.mojo:232-274`), one more filter pass on the fitted
     parameters with `trans = false`.

EXOGENOUS REGRESSORS (lane/arima-exog, 2026-09-15), restated where the device
reaches them: `exog_regression_kernel` (`estimate_x0.mojo`, the exog block of
`_start_params`) before the ARMA least squares, the regressors differenced
beside `y` (`batched_fit_x`), `beta` packed after `mu` and copied through the
Jones transform, `obs_intercept_kernel` and the `has_exog` arms of the loop
kernel (`batched_kalman.mojo`, DEVIATION 995's fold), and
`prepare_future_data`'s one-difference arm (`arima_helpers.mojo`). The
regressors arrive in the filter's layout, `[bid*n_exog*n + i*n + t]`
(`bindings/arima_exog_layout.mojo`).

`arima_forecast_ptr_host` and `arima_predict_ptr_host`
(`arima/estimator.mojo:409-545`, `batched_arima.mojo:287-348`) are mirrored
for `start == n_obs`: the differencing, the non-finite refusal on the filter
input, the untransformed copy of the parameters, the filter with
`fc_steps`, `finalize_forecast`'s one-difference `_undiff_kernel`
(`arima_helpers.mojo:76-152`) and `copy_forecast_kernel`
(`batched_arima.mojo:261-277`).

EVERY FILTER PASS IS ON A DIFFERENCED ORDER (`n_diff = 0`). The fit
optimizes on `order.without_diff()` and `predict` runs with `pre_diff =
true`, so the device loop kernel's `n_diff > 0` arms (the Z products, the
diffuse kappa, DEVIATION 677's negative code) are unreachable from the
public door. They are not restated; `_kalman` raises by name if a caller
ever hands it `n_diff != 0`.

WHAT IS REFUSED BY NAME (`arima_host_refuse_unrestated`, the binding calls it
after the GPU binding's own validation): `p`, `q` or `P` above 1 (the Jones
recursion and `test_invparams` past one coefficient), any `Q` (the seasonal
MA arms of the matrices kernel and the seasonal least squares with an AR
pre-fit), `d + D == 2` (the second difference and `_undiff_kernel[True]`),
`p + q + k == 0` (the seasonal-only arm of `start_params`). Every refused
arm is spelled on the device and none is reached by the three lanes, so the
CPU identity gate could not hold a restatement of it to anything.

THE NEGATIVE CONTROL. `-D MOJOLEARN_HOST_SABOTAGE=1` doubles the
finite-difference step (2^-9 in place of DEVIATION 687's 2^-10), so every
gradient, every iterate after x0 and the fitted parameters of every fixture
move, and the forecasts computed from them move with them. The same define,
or `-D MOJOLEARN_ARIMA_EXOG_SABOTAGE=1` alone, flips the lowest bit of every
finite observation intercept (`ARIMA_ORACLE_EXOG_SABOTAGE`), so a model with
regressors moves through the exog arithmetic itself, in the fit and in the
forecast, and a model without them does not see it.

The restatement is a prediction until measured. The four-column diff of
tools/identity_break.py on the three ARIMA lanes is the measurement.
"""
from std.math import isfinite, isinf, isnan
from std.memory import bitcast
from std.sys.compile import is_defined

from checks.numerics import (
    ftz,
    identical_exp,
    identical_log,
    identical_mul_add,
    identical_sqrt,
)


#: The gate's negative control (see THE NEGATIVE CONTROL above).
comptime ARIMA_ORACLE_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()

#: The inference negative control (lane/inference-forecast-umap-pca,
#: 2026-09-15). The step sabotage above moves only a FIT, so a binding that
#: serves prediction alone (`bindings/_mojolearn_forecast_host.mojo`) would
#: build it and still answer the right bytes. Under either define every finite
#: value `arima_host_predict` and `arima_host_forecast` return has its lowest
#: bit flipped, one ulp, after all arithmetic.
comptime ARIMA_ORACLE_PREDICT_SABOTAGE = (
    is_defined["MOJOLEARN_HOST_SABOTAGE"]() or is_defined["MOJOLEARN_ARIMA_PREDICT_SABOTAGE"]()
)

#: The exogenous arithmetic's own negative control (see THE NEGATIVE CONTROL).
comptime ARIMA_ORACLE_EXOG_SABOTAGE = (
    is_defined["MOJOLEARN_HOST_SABOTAGE"]() or is_defined["MOJOLEARN_ARIMA_EXOG_SABOTAGE"]()
)

#: DEVIATION 687, `ARIMA_FIT_H` (`batched_fit.mojo:148`), and the sabotage
#: arm's doubled step.
comptime AH_FIT_H = Float32(0.0009765625)
comptime AH_FIT_H_SABOTAGE = Float32(0.001953125)

#: `jones_transform.mojo:91-93`.
comptime AH_JONES_MAX = 8
comptime AH_JONES_CLAMP = Float32(0.9999)

#: `MIN_SIGMA2` (`arima_helpers.mojo:45`).
comptime AH_MIN_SIGMA2 = Float32(1e-6)

#: `batched_kalman.mojo:132-135`.
comptime AH_RD_MAX = 8
comptime AH_RD2_MAX = 64
comptime AH_LOG_2PI = Float32(1.8378770664093453)

#: `LYAP_R_MAX`, `LYAP_R2_MAX` (`linalg/batched/matrix.mojo:74-75`) and the
#: Kronecker system's largest size.
comptime AH_LYAP_R2_MAX = 25
comptime AH_KRON_MAX = 625

#: `LS_MAX_COLS`, `LS_RANK_TOL` (`least_squares.mojo:104-107`).
comptime AH_LS_MAX_COLS = 17
comptime AH_LS_RANK_TOL = Float32(1.0e-5)

#: `arima_fit_params` (`batched_fit.mojo:151-175`): m, epsilon, past, delta,
#: max_linesearch, min_step, max_step, ftol, ls_dec. `wolfe` and `ls_inc`
#: are not read by the Armijo backtracking.
comptime AH_M = 10
comptime AH_EPSILON = Float32(1.0e-3)
comptime AH_PAST = 10
comptime AH_DELTA = Float32(1.0e-6)
comptime AH_MAX_LINESEARCH = 20
comptime AH_MIN_STEP = Float32(1.0e-20)
comptime AH_MAX_STEP = Float32(1.0e20)
comptime AH_FTOL = Float32(1.0e-4)
comptime AH_LS_DEC = Float32(0.5)

#: `glm/impl/qn/qn_util.mojo:62-77`.
comptime AH_LS_SUCCESS = 0
comptime AH_LS_INVALID_STEP_MIN = 1
comptime AH_LS_INVALID_STEP_MAX = 2
comptime AH_LS_MAX_ITERS_REACHED = 3
comptime AH_LS_INVALID_DIR = 4
comptime AH_LS_INVALID_STEP = 5
comptime AH_OPT_SUCCESS = 0
comptime AH_OPT_NUMERIC_ERROR = 1
comptime AH_OPT_LS_FAILED = 2
comptime AH_OPT_MAX_ITERS_REACHED = 3
comptime AH_FLOAT_EPSILON = Float32(1.1920928955078125e-7)


@always_inline
def arima_host_fit_h() -> Float32:
    """The finite-difference step this build uses."""
    comptime if ARIMA_ORACLE_HOST_SABOTAGE:
        return AH_FIT_H_SABOTAGE
    else:
        return AH_FIT_H


# ===========================================================================
# THE ORDER AND THE PARAMETERS (`arima/impl/tsa/arima_common.mojo`)
# ===========================================================================


@fieldwise_init
struct ArimaHostOrder(Copyable, Movable, ImplicitlyCopyable):
    """`ARIMAOrder` (`arima_common.mojo:65-105`), validated by the binding
    through the device's `validate_order`."""

    var p: Int
    var d: Int
    var q: Int
    var P: Int
    var D: Int
    var Q: Int
    var s: Int
    var k: Int
    var n_exog: Int

    def n_diff(self) -> Int:
        return self.d + self.s * self.D

    def n_phi(self) -> Int:
        return self.p + self.s * self.P

    def n_theta(self) -> Int:
        return self.q + self.s * self.Q

    def r(self) -> Int:
        var a = self.n_phi()
        var b = self.n_theta() + 1
        return a if a > b else b

    def rd(self) -> Int:
        return self.n_diff() + self.r()

    def complexity(self) -> Int:
        return self.p + self.P + self.q + self.Q + self.k + self.n_exog + 1

    def need_diff(self) -> Bool:
        return self.d + self.D != 0

    def without_diff(self) -> Self:
        return ArimaHostOrder(self.p, 0, self.q, self.P, 0, self.Q, self.s, self.k, self.n_exog)


def arima_host_refuse_unrestated(order: ArimaHostOrder, who: String) raises:
    """The parameter values whose arms this file does not restate, refused by
    name (see WHAT IS REFUSED BY NAME above). Called after the device's own
    `validate_order`, so every value that reaches here is one the GPU accepts."""
    var tail = String(
        "; the host restatement (arima/host/arima_oracle.mojo) covers p, q and"
        " P of at most 1, Q = 0, d + D of at most 1 and p + q + k of at least"
        " 1, the arms the arima, arima-011 and arima-seasonal-c lanes reach"
        " and the CPU identity gate holds to the GPU columns"
    )
    if order.p > 1:
        raise Error(
            who + ": no CPU implementation of ARIMA with p=" + String(order.p)
            + " (the Jones recursion and test_invparams past one AR coefficient)"
            + tail
        )
    if order.q > 1:
        raise Error(
            who + ": no CPU implementation of ARIMA with q=" + String(order.q)
            + " (the Jones recursion and test_invparams past one MA coefficient)"
            + tail
        )
    if order.P > 1:
        raise Error(
            who + ": no CPU implementation of ARIMA with P=" + String(order.P)
            + " (the Jones recursion past one seasonal AR coefficient)" + tail
        )
    if order.Q != 0:
        raise Error(
            who + ": no CPU implementation of ARIMA with Q=" + String(order.Q)
            + " (the seasonal MA arms of the state space matrices and the"
            " seasonal least squares with an AR pre-fit)" + tail
        )
    if order.d + order.D > 1:
        raise Error(
            who + ": no CPU implementation of ARIMA with d + D = "
            + String(order.d + order.D)
            + " (the second difference and its two-term undifferencing)" + tail
        )
    if order.p + order.q + order.k == 0:
        raise Error(
            who + ": no CPU implementation of ARIMA with p + q + k = 0 (the"
            " seasonal-only arm of start_params, which estimates sigma2 from"
            " the seasonal least squares)" + tail
        )


def _zeros(n: Int) -> List[Float32]:
    return List[Float32](length=n, fill=Float32(0.0))


@fieldwise_init
struct ArimaHostParams(Copyable, Movable):
    """`ARIMAParams` (`arima_common.mojo:140-159`) on the host: one list per
    kind, laid out `[bid * n_kind + i]`, at least one value long."""

    var mu: List[Float32]
    var beta: List[Float32]
    var ar: List[Float32]
    var ma: List[Float32]
    var sar: List[Float32]
    var sma: List[Float32]
    var sigma2: List[Float32]


def _new_params(order: ArimaHostOrder, batch_size: Int) -> ArimaHostParams:
    return ArimaHostParams(
        mu=_zeros(max(1, order.k * batch_size)),
        beta=_zeros(max(1, order.n_exog * batch_size)),
        ar=_zeros(max(1, order.p * batch_size)),
        ma=_zeros(max(1, order.q * batch_size)),
        sar=_zeros(max(1, order.P * batch_size)),
        sma=_zeros(max(1, order.Q * batch_size)),
        sigma2=_zeros(max(1, batch_size)),
    )


def _pack(params: ArimaHostParams, order: ArimaHostOrder, batch_size: Int) -> List[Float32]:
    """`pack_kernel` (`arima_common.mojo:162-201`): `[mu, beta, ar, ma, sar,
    sma, sigma2]` per series, a copy."""
    var N = order.complexity()
    var out = _zeros(N * batch_size)
    for bid in range(batch_size):
        var o = bid * N
        if order.k != 0:
            out[o] = params.mu[bid]
            o += 1
        for i in range(order.n_exog):
            out[o + i] = params.beta[order.n_exog * bid + i]
        o += order.n_exog
        for i in range(order.p):
            out[o + i] = params.ar[order.p * bid + i]
        o += order.p
        for i in range(order.q):
            out[o + i] = params.ma[order.q * bid + i]
        o += order.q
        for i in range(order.P):
            out[o + i] = params.sar[order.P * bid + i]
        o += order.P
        for i in range(order.Q):
            out[o + i] = params.sma[order.Q * bid + i]
        o += order.Q
        out[o] = params.sigma2[bid]
    return out^


def _unpack(x: List[Float32], order: ArimaHostOrder, batch_size: Int) -> ArimaHostParams:
    """`unpack_kernel` (`arima_common.mojo:204-241`), the inverse copy."""
    var params = _new_params(order, batch_size)
    var N = order.complexity()
    for bid in range(batch_size):
        var o = bid * N
        if order.k != 0:
            params.mu[bid] = x[o]
            o += 1
        for i in range(order.n_exog):
            params.beta[order.n_exog * bid + i] = x[o + i]
        o += order.n_exog
        for i in range(order.p):
            params.ar[order.p * bid + i] = x[o + i]
        o += order.p
        for i in range(order.q):
            params.ma[order.q * bid + i] = x[o + i]
        o += order.q
        for i in range(order.P):
            params.sar[order.P * bid + i] = x[o + i]
        o += order.P
        for i in range(order.Q):
            params.sma[order.Q * bid + i] = x[o + i]
        o += order.Q
        params.sigma2[bid] = x[o]
    return params^


# ===========================================================================
# DIFFERENCING AND UNDIFFERENCING
# ===========================================================================


def _prepare_data(
    y: List[Float32], batch_size: Int, n_obs: Int, order: ArimaHostOrder
) raises -> List[Float32]:
    """`prepare_data` (`tsa/impl/timeSeries/arima_helpers.mojo:32-76`) for
    `d + D <= 1`: `batched_diff_kernel` (`tsa/impl/linalg/batched/
    matrix.mojo:28-46`, `out[i] = ftz(ftz(in[i + period]) - ftz(in[i]))`, one
    cell per thread, no fold), or the raw COPY when `d + D == 0` (no flush)."""
    var dD = order.d + order.D
    if dD == 0:
        return y.copy()
    if dD != 1:
        raise Error(
            "arima host: prepare_data reached with d + D = " + String(dD)
            + "; arima_host_refuse_unrestated refuses it first"
        )
    var period = 1 if order.d != 0 else order.s
    var n_out = n_obs - period
    var out = _zeros(max(1, n_out * batch_size))
    for b in range(batch_size):
        var batch_in = b * n_obs
        var batch_out = b * n_out
        for i in range(n_out):
            var hi = ftz(y[batch_in + i + period])
            var lo = ftz(y[batch_in + i])
            out[batch_out + i] = ftz(hi - lo)
    return out^


def _finalize_forecast(
    mut fc: List[Float32], y: List[Float32], num_steps: Int, batch_size: Int,
    in_ld: Int, n_in: Int, order: ArimaHostOrder,
) raises:
    """`finalize_forecast` (`arima_helpers.mojo:124-152`) with `d + D == 1`:
    `undiff_kernel[False]` (`:76-111`), serial over the steps, the forecast
    read back through `_select_read` (`:115-121`) after its own update."""
    var dD = order.d + order.D
    if dD == 0:
        return
    if dD != 1:
        raise Error(
            "arima host: finalize_forecast reached with d + D = " + String(dD)
            + "; arima_host_refuse_unrestated refuses it first"
        )
    var s0 = 1 if order.d != 0 else order.s
    for bid in range(batch_size):
        var fc_base = bid * num_steps
        var in_base = bid * in_ld
        for i in range(num_steps):
            var cur = ftz(fc[fc_base + i])
            var idx = i - s0
            var x: Float32
            if idx < 0:
                x = ftz(y[in_base + n_in + idx])
            else:
                x = ftz(fc[fc_base + idx])
            fc[fc_base + i] = ftz(cur + x)


def _prepare_future(
    past: List[Float32], fut: List[Float32], n_series: Int, n_past: Int, n_fut: Int,
    order: ArimaHostOrder,
) raises -> List[Float32]:
    """`prepare_future_data` (`arima_helpers.mojo`) for `d + D <= 1`:
    `future_diff_kernel`'s one-difference arm, `ftz(ftz(fut[i]) -
    sel(i - period))` with `_select_read` flushing its load, or the COPY."""
    var dD = order.d + order.D
    if dD == 0:
        return fut.copy()
    if dD != 1:
        raise Error(
            "arima host: prepare_future_data reached with d + D = " + String(dD)
            + "; arima_host_refuse_unrestated refuses it first"
        )
    var period = 1 if order.d != 0 else order.s
    var out = _zeros(max(1, n_fut * n_series))
    for sid in range(n_series):
        var pb = sid * n_past
        var fb = sid * n_fut
        for i in range(n_fut):
            var a = ftz(fut[fb + i])
            var idx = i - period
            var b: Float32
            if idx < 0:
                b = ftz(past[pb + n_past + idx])
            else:
                b = ftz(fut[fb + idx])
            out[fb + i] = ftz(a - b)
    return out^


def _obs_intercept(
    exog: List[Float32], beta: List[Float32], batch_size: Int, n: Int, n_exog: Int
) -> List[Float32]:
    """`obs_intercept_kernel` (`batched_kalman.mojo`): `sum_i exog * beta_i`,
    DEVIATION 995's serial ascending fma from 0, series major. Under
    `ARIMA_ORACLE_EXOG_SABOTAGE` every finite value has its lowest bit
    flipped."""
    var out = _zeros(max(1, n * batch_size))
    for bid in range(batch_size):
        var xb = bid * n_exog * n
        var bb = bid * n_exog
        for t in range(n):
            var acc = Float32(0.0)
            for i in range(n_exog):
                var xv = ftz(exog[xb + i * n + t])
                var bv = ftz(beta[bb + i])
                acc = ftz(identical_mul_add(xv, bv, acc))
            comptime if ARIMA_ORACLE_EXOG_SABOTAGE:
                if isfinite(acc):
                    acc = bitcast[DType.float32](bitcast[DType.uint32](acc) ^ UInt32(1))
            out[bid * n + t] = acc
    return out^


def _refuse_non_finite(y: List[Float32], n: Int, name: String) raises:
    """`_refuse_non_finite` (`batched_arima.mojo:149-169`), its words."""
    for i in range(n):
        var v = y[i]
        if not isfinite(v):
            raise Error(
                "batched_loglike: " + name + " contains a non-finite value at index "
                + String(i)
                + "; missing observations are not implemented and are refused by name (arima/NOT_IMPLEMENTED.tsv)"
            )


# ===========================================================================
# THE JONES TRANSFORM (`arima/impl/timeSeries/jones_transform.mojo`)
# ===========================================================================


@always_inline
def _tanh_half(x: Float32) -> Float32:
    """`tanh_half`'s IDENTICAL arm (`jones_transform.mojo:97-109`,
    DEVIATION 675). The host binding builds IDENTICAL only."""
    if x > Float32(80.0):
        return Float32(1.0)
    if x < Float32(-80.0):
        return Float32(-1.0)
    var e = ftz(identical_exp(x))
    var num = ftz(e - Float32(1.0))
    var den = ftz(e + Float32(1.0))
    return ftz(num / den)


@always_inline
def _two_atanh(v: Float32) -> Float32:
    """`two_atanh`'s IDENTICAL arm (`jones_transform.mojo:113-120`)."""
    var num = ftz(Float32(1.0) + v)
    var den = ftz(Float32(1.0) - v)
    return ftz(identical_log(ftz(num / den)))


def _jones_transform(
    params: List[Float32], batch_size: Int, parameter: Int, is_ar: Bool, is_inv: Bool
) -> List[Float32]:
    """`jones_transform_kernel` (`jones_transform.mojo:129-183`) with `clamp`
    on, per series. The recursion's inner line is `sign * (a * x)`: `a * x`
    rounds on its own and only the outer product fuses into the add (TWO
    roundings, the 2026-08-23 correction)."""
    var out = _zeros(max(1, parameter * batch_size))
    for model in range(batch_size):
        var tmp = Array[Float32, AH_JONES_MAX](fill=Float32(0.0))
        var mine = Array[Float32, AH_JONES_MAX](fill=Float32(0.0))
        for i in range(parameter):
            var v = ftz(params[model * parameter + i])
            tmp[i] = v
            mine[i] = v
        if is_inv:
            var sign = Float32(1.0) if is_ar else Float32(-1.0)
            var j = parameter - 1
            while j > 0:
                var a = mine[j]
                var den = ftz(identical_mul_add(-a, a, Float32(1.0)))
                for k in range(j):
                    var prod = ftz(a * mine[j - k - 1])
                    var num = ftz(identical_mul_add(sign, prod, mine[k]))
                    tmp[k] = ftz(num / den)
                for it in range(j):
                    mine[it] = tmp[it]
                j -= 1
            for i in range(parameter):
                mine[i] = _two_atanh(mine[i])
        else:
            for i in range(parameter):
                tmp[i] = _tanh_half(tmp[i])
                mine[i] = tmp[i]
            var sign = Float32(-1.0) if is_ar else Float32(1.0)
            for j in range(1, parameter):
                var a = mine[j]
                for k in range(j):
                    var prod = ftz(a * mine[j - k - 1])
                    tmp[k] = ftz(identical_mul_add(sign, prod, tmp[k]))
                for it in range(j):
                    mine[it] = tmp[it]
            for i in range(parameter):
                # `jones_clamp` (`:124-126`), value-first.
                mine[i] = max(min(mine[i], AH_JONES_CLAMP), -AH_JONES_CLAMP)
        for i in range(parameter):
            out[model * parameter + i] = mine[i]
    return out^


def _batched_jones(
    order: ArimaHostOrder, batch_size: Int, is_inv: Bool, params: ArimaHostParams
) -> ArimaHostParams:
    """`batched_jones_transform` (`arima_helpers.mojo:202-230`): AR, MA, SAR,
    SMA when present, `sigma2_floor_kernel` (`:189-199`, `max(ftz(v),
    1e-6)`) on BOTH directions, `mu` and `beta` copied."""
    var t = _new_params(order, batch_size)
    if order.p != 0:
        t.ar = _jones_transform(params.ar, batch_size, order.p, True, is_inv)
    if order.q != 0:
        t.ma = _jones_transform(params.ma, batch_size, order.q, False, is_inv)
    if order.P != 0:
        t.sar = _jones_transform(params.sar, batch_size, order.P, True, is_inv)
    if order.Q != 0:
        t.sma = _jones_transform(params.sma, batch_size, order.Q, False, is_inv)
    for i in range(batch_size):
        t.sigma2[i] = max(ftz(params.sigma2[i]), AH_MIN_SIGMA2)
    if order.k != 0:
        for i in range(batch_size):
            t.mu[i] = params.mu[i]
    for i in range(order.n_exog * batch_size):
        t.beta[i] = params.beta[i]
    return t^


# ===========================================================================
# THE KALMAN FILTER (`arima/impl/batched_kalman.mojo`), n_diff = 0
# ===========================================================================


@always_inline
def _param_to_poly(is_ar: Bool, param0: Float32, idx: Int, lags: Int) -> Float32:
    """`param_to_poly` (`arima_helpers.mojo:49-57`)."""
    if idx > lags:
        return Float32(0.0)
    elif idx != 0:
        return -param0 if is_ar else param0
    return Float32(1.0)


@always_inline
def _reduced_polynomial(is_ar: Bool, coef0: Float32, coef1: Float32) -> Float32:
    """`reduced_polynomial` (`arima_helpers.mojo:61-65`): one rounding, the
    sign after (so an AR zero product is -0.0)."""
    var prod = ftz(coef0 * coef1)
    return -prod if is_ar else prod


def _lu_inverse(
    mut a: Array[Float32, AH_KRON_MAX],
    mut inv: Array[Float32, AH_KRON_MAX],
    n: Int,
) -> Int32:
    """`lu_inverse` (`linalg/batched/matrix.mojo:79-148`): getrf with the
    first-largest pivot over flushed magnitudes, the raw row swap, `l =
    ftz(ftz(a) / pivot)`, the fused trailing update; then getri column by
    column through the swaps, unit-L forward and U backward substitution,
    ascending folds. 0, or `j + 1` at the first zero pivot."""
    var piv = Array[Int, 25](fill=0)
    for j in range(n):
        var best = j
        var best_mag = abs(ftz(a[j + j * n]))
        for i in range(j + 1, n):
            var m = abs(ftz(a[i + j * n]))
            if m > best_mag:
                best_mag = m
                best = i
        piv[j] = best
        if best != j:
            for c in range(n):
                var t0 = a[j + c * n]
                a[j + c * n] = a[best + c * n]
                a[best + c * n] = t0
        var pivot = ftz(a[j + j * n])
        if pivot == Float32(0.0):
            return Int32(j + 1)
        for i in range(j + 1, n):
            var l = ftz(ftz(a[i + j * n]) / pivot)
            a[i + j * n] = l
            for c in range(j + 1, n):
                var u = ftz(a[j + c * n])
                var cur = ftz(a[i + c * n])
                a[i + c * n] = ftz(identical_mul_add(-l, u, cur))
    for col in range(n):
        for i in range(n):
            inv[i + col * n] = Float32(1.0) if i == col else Float32(0.0)
        for j in range(n):
            var pj = piv[j]
            if pj != j:
                var t0 = inv[j + col * n]
                inv[j + col * n] = inv[pj + col * n]
                inv[pj + col * n] = t0
        for i in range(n):
            var acc = ftz(inv[i + col * n])
            for k in range(i):
                var l = ftz(a[i + k * n])
                var yk = ftz(inv[k + col * n])
                acc = ftz(identical_mul_add(-l, yk, acc))
            inv[i + col * n] = acc
        var i = n - 1
        while i >= 0:
            var acc = ftz(inv[i + col * n])
            for k in range(i + 1, n):
                var u = ftz(a[i + k * n])
                var xk = ftz(inv[k + col * n])
                acc = ftz(identical_mul_add(-u, xk, acc))
            var d = ftz(a[i + i * n])
            inv[i + col * n] = ftz(acc / d)
            i -= 1
    return Int32(0)


@always_inline
def _mv(
    n: Int, alpha: Float32, a: Array[Float32, AH_RD2_MAX],
    v: Array[Float32, AH_RD_MAX], mut out_v: Array[Float32, AH_RD_MAX],
):
    """`_mv` (`batched_kalman.mojo:382-394`)."""
    for i in range(n):
        var acc = Float32(0.0)
        for j in range(n):
            acc = ftz(identical_mul_add(a[i + j * n], v[j], acc))
        out_v[i] = ftz(alpha * acc)


@always_inline
def _mm(
    n: Int, a: Array[Float32, AH_RD2_MAX], b: Array[Float32, AH_RD2_MAX],
    bT: Bool, mut out_v: Array[Float32, AH_RD2_MAX],
):
    """`_mm` (`batched_kalman.mojo:398-409`)."""
    for i in range(n):
        for j in range(n):
            var acc = Float32(0.0)
            for k in range(n):
                var bkj = b[j + k * n] if bT else b[k + j * n]
                acc = ftz(identical_mul_add(a[i + k * n], bkj, acc))
            out_v[i + j * n] = acc


@always_inline
def _numerical_stability(n: Int, mut a: Array[Float32, AH_RD2_MAX]):
    """`_numerical_stability` (`batched_kalman.mojo:413-422`)."""
    for i in range(n - 1):
        for j in range(i + 1, n):
            var s = ftz(a[j * n + i] + a[i * n + j])
            var new_val = ftz(Float32(0.5) * s)
            a[j * n + i] = new_val
            a[i * n + j] = new_val
    for i in range(n):
        a[i * n + i] = abs(a[i * n + i])


@fieldwise_init
struct KalmanHostOut(Movable):
    var loglike: List[Float32]
    var fc: List[Float32]
    #: `d_pred` (`batched_kalman.mojo:496-502`), the one-step prediction at
    #: every observation, series major at `bid * nobs + it`.
    var pred: List[Float32]


def _kalman(
    ys: List[Float32],
    exog: List[Float32],
    exog_fut: List[Float32],
    nobs: Int,
    t: ArimaHostParams,
    order: ArimaHostOrder,
    batch_size: Int,
    fc_steps: Int,
) raises -> KalmanHostOut:
    """`batched_kalman_filter` (`batched_kalman.mojo:722-789`) on TRANSFORMED
    parameters and a differenced order: every series' matrices and initial
    state, the initial-state refusal by name, every series' loop, the
    innovation-variance refusal by name, in that order."""
    var n_diff = order.n_diff()
    if n_diff != 0:
        raise Error(
            "arima host: the Kalman filter was handed n_diff = " + String(n_diff)
            + "; the public door filters differenced series only and the"
            " n_diff > 0 arms are not restated"
        )
    var p = order.p
    var q = order.q
    var P = order.P
    var Q = order.Q
    var s = order.s
    var n_phi = order.n_phi()
    var n_theta = order.n_theta()
    var r = order.r()
    var rd = order.rd()
    var rd2 = rd * rd
    var r2 = r * r
    var T_all = _zeros(rd2 * batch_size)
    var RQR_all = _zeros(rd2 * batch_size)
    var P_all = _zeros(rd2 * batch_size)
    var alpha_all = _zeros(rd * batch_size)
    var info0 = List[Int32](length=batch_size, fill=Int32(0))
    var has_exog = order.n_exog != 0
    var obs = _zeros(1)
    var obs_fut = _zeros(1)
    if has_exog:
        obs = _obs_intercept(exog, t.beta, batch_size, nobs, order.n_exog)
        if fc_steps > 0:
            obs_fut = _obs_intercept(exog_fut, t.beta, batch_size, fc_steps, order.n_exog)

    for bid in range(batch_size):
        # -- init_batched_kalman_matrices_kernel (:149-235), n_diff = 0
        var R = Array[Float32, AH_RD_MAX](fill=Float32(0.0))
        var T = Array[Float32, AH_RD2_MAX](fill=Float32(0.0))
        R[n_diff] = Float32(1.0)
        for i in range(n_theta):
            var idx = i + 1
            var idx1 = idx // s if s != 0 else 0
            var idx0 = idx - s * idx1
            var c0 = _param_to_poly(False, t.ma[bid * q + idx0 - 1] if (idx0 != 0 and idx0 <= q) else Float32(0.0), idx0, q)
            var c1 = _param_to_poly(False, t.sma[bid * Q + idx1 - 1] if (idx1 != 0 and idx1 <= Q) else Float32(0.0), idx1, Q)
            R[n_diff + i + 1] = _reduced_polynomial(False, c0, c1)
        for i in range(n_phi):
            var idx = i + 1
            var idx1 = idx // s if s != 0 else 0
            var idx0 = idx - s * idx1
            var c0 = _param_to_poly(True, t.ar[bid * p + idx0 - 1] if (idx0 != 0 and idx0 <= p) else Float32(0.0), idx0, p)
            var c1 = _param_to_poly(True, t.sar[bid * P + idx1 - 1] if (idx1 != 0 and idx1 <= P) else Float32(0.0), idx1, P)
            T[n_diff * (rd + 1) + i] = _reduced_polynomial(True, c0, c1)
        for i in range(r - 1):
            T[(n_diff + i + 1) * rd + n_diff + i] = Float32(1.0)
        if rd == 2 and p == 2:
            var t1 = ftz(T[1])
            if abs(ftz(t1 + Float32(1.0))) < Float32(0.01):
                T[1] = Float32(-0.99)

        # -- kalman_init_state_kernel (:243-373)
        var info = Int32(0)
        var RQ = Array[Float32, AH_RD_MAX](fill=Float32(0.0))
        var RQR = Array[Float32, AH_RD2_MAX](fill=Float32(0.0))
        var Pm = Array[Float32, AH_RD2_MAX](fill=Float32(0.0))
        var alpha = Array[Float32, AH_RD_MAX](fill=Float32(0.0))
        var sigma2 = ftz(t.sigma2[bid])
        for i in range(rd):
            RQ[i] = ftz(ftz(R[i]) * sigma2)
        for j in range(rd):
            var rj = ftz(R[j])
            for i in range(rd):
                RQR[i + j * rd] = ftz(ftz(RQ[i]) * rj)
        # `kron_minus_identity` (matrix.mojo:152-180): I - A (x) A, two
        # roundings per cell with the exact -1.
        var imaa = Array[Float32, AH_KRON_MAX](fill=Float32(0.0))
        var imaa_inv = Array[Float32, AH_KRON_MAX](fill=Float32(0.0))
        for ia in range(r):
            for ja in range(r):
                var a_ia_ja = -ftz(T[(ia + n_diff) + (ja + n_diff) * rd])
                for ib in range(r):
                    for jb in range(r):
                        var i_ab = ia * r + ib
                        var j_ab = ja * r + jb
                        var b_val = ftz(T[(ib + n_diff) + (jb + n_diff) * rd])
                        var v = ftz(a_ia_ja * b_val)
                        if i_ab == j_ab:
                            v = ftz(v + Float32(1.0))
                        imaa[i_ab + j_ab * r2] = v
        var inf1 = _lu_inverse(imaa, imaa_inv, r2)
        if inf1 != Int32(0):
            info = inf1
        else:
            var vecq = Array[Float32, AH_LYAP_R2_MAX](fill=Float32(0.0))
            for j in range(r):
                for i in range(r):
                    vecq[i + j * r] = ftz(RQR[(i + n_diff) + (j + n_diff) * rd])
            var xloc = Array[Float32, AH_LYAP_R2_MAX](fill=Float32(0.0))
            for i in range(r2):
                var acc = Float32(0.0)
                for k in range(r2):
                    var av = ftz(imaa_inv[i + k * r2])
                    var xv = ftz(vecq[k])
                    acc = ftz(identical_mul_add(av, xv, acc))
                xloc[i] = acc
            for j in range(r):
                for i in range(r):
                    Pm[(i + n_diff) + (j + n_diff) * rd] = xloc[i + j * r]
        if order.k != 0:
            var imt = Array[Float32, AH_KRON_MAX](fill=Float32(0.0))
            var imt_inv = Array[Float32, AH_KRON_MAX](fill=Float32(0.0))
            for j in range(r):
                for i in range(r):
                    var delta = Float32(1.0) if i == j else Float32(0.0)
                    var tij = ftz(T[(i + n_diff) + (j + n_diff) * rd])
                    imt[i + j * r] = ftz(delta - tij)
            if r == 1:
                var v = imt[0]
                if abs(v) < Float32(1e-3):
                    # raft::signPrim: signbit(x) ? -1 : +1
                    var neg = (bitcast[DType.uint32](v) >> 31) != 0
                    imt[0] = Float32(-1e-3) if neg else Float32(1e-3)
            var inf2 = _lu_inverse(imt, imt_inv, r)
            if inf2 != Int32(0) and info == Int32(0):
                info = inf2
            var mu = ftz(t.mu[bid])
            if inf2 == Int32(0):
                for i in range(r):
                    alpha[i + n_diff] = ftz(ftz(imt_inv[i]) * mu)
        info0[bid] = info
        for i in range(rd2):
            T_all[bid * rd2 + i] = T[i]
            RQR_all[bid * rd2 + i] = RQR[i]
            P_all[bid * rd2 + i] = Pm[i]
        for i in range(rd):
            alpha_all[bid * rd + i] = alpha[i]

    for b in range(batch_size):
        if info0[b] != Int32(0):
            raise Error(
                "batched_kalman_filter: series " + String(b) + ": the initial-state system (I - T (x) T, or I - T* for the intercept) is singular at column "
                + String(info0[b]) + "; a unit-root parameter set is refused by name rather than filtered with a non-finite P0"
            )

    var loglike = _zeros(batch_size)
    var fc = _zeros(max(1, fc_steps * batch_size))
    var pred_all = _zeros(max(1, nobs * batch_size))
    var info1 = List[Int32](length=batch_size, fill=Int32(0))
    for bid in range(batch_size):
        # -- batched_kalman_loop_kernel (:425-604), n_diff = 0
        var l_RQR = Array[Float32, AH_RD2_MAX](fill=Float32(0.0))
        var l_T = Array[Float32, AH_RD2_MAX](fill=Float32(0.0))
        var l_P = Array[Float32, AH_RD2_MAX](fill=Float32(0.0))
        var l_alpha = Array[Float32, AH_RD_MAX](fill=Float32(0.0))
        var l_K = Array[Float32, AH_RD_MAX](fill=Float32(0.0))
        var l_tmp = Array[Float32, AH_RD2_MAX](fill=Float32(0.0))
        var l_TP = Array[Float32, AH_RD2_MAX](fill=Float32(0.0))
        var l_v = Array[Float32, AH_RD_MAX](fill=Float32(0.0))
        var b_rd = bid * rd
        var b_rd2 = bid * rd2
        for i in range(rd2):
            l_RQR[i] = ftz(RQR_all[b_rd2 + i])
            l_T[i] = ftz(T_all[b_rd2 + i])
            l_P[i] = ftz(P_all[b_rd2 + i])
        for i in range(rd):
            l_alpha[i] = ftz(alpha_all[b_rd + i])
        var b_sum_logFs = Float32(0.0)
        var b_ll_s2 = Float32(0.0)
        var n_obs_ll = 0
        var info = Int32(0)
        var b_ys = bid * nobs
        var mu = ftz(t.mu[bid]) if order.k != 0 else Float32(0.0)
        for it in range(nobs):
            # 1. v = y - Z*alpha
            var pred = Float32(0.0)
            if has_exog:
                pred = ftz(pred + ftz(obs[b_ys + it]))
            pred = ftz(pred + l_alpha[0])
            pred_all[b_ys + it] = pred
            var yt = ftz(ys[b_ys + it])
            var vs_it = ftz(yt - pred)
            # 2. F = Z*P*Z'
            var _Fs = l_P[0]
            if _Fs <= Float32(0.0) and info == Int32(0):
                info = Int32(it + 1) if it >= n_diff else Int32(-(it + 1))
            if it >= n_diff:
                if _Fs > Float32(0.0):
                    b_sum_logFs = ftz(b_sum_logFs + ftz(identical_log(_Fs)))
                    var v2 = ftz(vs_it * vs_it)
                    b_ll_s2 = ftz(b_ll_s2 + ftz(v2 / _Fs))
                n_obs_ll += 1
            # 3. K = 1/Fs * T*P*Z'
            _mm(rd, l_T, l_P, False, l_TP)
            var _1_Fs = ftz(Float32(1.0) / _Fs)
            for i in range(rd):
                l_K[i] = ftz(_1_Fs * l_TP[i])
            # 4. alpha = T*alpha + K*vs + c
            _mv(rd, Float32(1.0), l_T, l_alpha, l_v)
            for i in range(rd):
                l_alpha[i] = ftz(identical_mul_add(l_K[i], vs_it, l_v[i]))
            l_alpha[n_diff] = ftz(l_alpha[n_diff] + mu)
            # 5. L = T - K*Z
            for i in range(rd2):
                l_tmp[i] = l_T[i]
            for i in range(rd):
                l_tmp[i] = ftz(l_tmp[i] - l_K[i])
            # 6. P = T*P*L' + R*Q*R'
            _mm(rd, l_TP, l_tmp, True, l_P)
            for i in range(rd2):
                l_P[i] = ftz(l_P[i] + l_RQR[i])
            _numerical_stability(rd, l_P)
        var n_obs_ll_f = Float32(n_obs_ll)
        b_ll_s2 = ftz(b_ll_s2 / n_obs_ll_f)
        var inner = ftz(b_ll_s2 + AH_LOG_2PI)
        var tot = ftz(identical_mul_add(n_obs_ll_f, inner, b_sum_logFs))
        loglike[bid] = ftz(Float32(-0.5) * tot)
        info1[bid] = info
        var b_fc = bid * fc_steps
        for it in range(fc_steps):
            var pred = Float32(0.0)
            if has_exog:
                pred = ftz(pred + ftz(obs_fut[b_fc + it]))
            pred = ftz(pred + l_alpha[0])
            fc[b_fc + it] = pred
            _mv(rd, Float32(1.0), l_T, l_alpha, l_v)
            for i in range(rd):
                l_alpha[i] = l_v[i]
            l_alpha[n_diff] = ftz(l_alpha[n_diff] + mu)

    for b in range(batch_size):
        if info1[b] > Int32(0):
            raise Error(
                "batched_kalman_filter: series " + String(b) + ": innovation variance F <= 0 at step "
                + String(info1[b] - 1) + "; refused by name rather than carrying log(F) into the likelihood"
            )
        if info1[b] < Int32(0):
            raise Error(
                "batched_kalman_filter: series " + String(b)
                + ": innovation variance F <= 0 at DIFFUSE step " + String(-info1[b] - 1)
                + " (before it >= n_diff = " + String(n_diff) + ", so it contributes no term to the "
                + "log-likelihood); refused by name rather than carrying 1/F = inf into the gain "
                + "(DEVIATION 677)"
            )
    return KalmanHostOut(loglike=loglike^, fc=fc^, pred=pred_all^)


def _loglike_packed(
    y_kf: List[Float32], exog_kf: List[Float32], batch_size: Int, n_obs_kf: Int,
    order_kf: ArimaHostOrder, x: List[Float32],
) raises -> List[Float32]:
    """`batched_loglike_packed` (`batched_arima.mojo:186-203`) with `trans =
    true` and `check_finite = false`: unpack, the forward transform, the
    filter, the host copy of the log-likelihood."""
    var raw = _unpack(x, order_kf, batch_size)
    var t = _batched_jones(order_kf, batch_size, False, raw)
    var out = _kalman(y_kf, exog_kf, _zeros(1), n_obs_kf, t, order_kf, batch_size, 0)
    return out.loglike.copy()


# ===========================================================================
# THE OBJECTIVE AND ITS FORWARD DIFFERENCE
# ===========================================================================


def _eval_batch(
    y_kf: List[Float32], exog_kf: List[Float32], batch_size: Int, n_obs_kf: Int,
    order_kf: ArimaHostOrder,
    h: Float32, scale: Float32, xin: List[Float32],
    mut fout: List[Float32], mut gout: List[Float32],
) raises:
    """`eval_batch` (`batched_fit.mojo:219-264`) over `batched_loglike_grad`
    (`batched_arima.mojo:413-469`): the base pass on `x`, then for each
    parameter `i` the pass on `x` with cell `N*bid + i` replaced by
    `ftz(ftz(x) + h)` (`perturb_kernel` `:357-371`), the gradient
    `ftz(ftz(ftz(pert) - ftz(base)) / h)` (`grad_kernel` `:396-410`), the
    reset by COPY (`reset_param_kernel` `:374-393`); then `f = ftz(ftz(-ll) /
    scale)` and `g = ftz(ftz(-grad) / scale)`."""
    var N = order_kf.complexity()
    var base = _loglike_packed(y_kf, exog_kf, batch_size, n_obs_kf, order_kf, xin)
    var x_pert = xin.copy()
    var grad = _zeros(N * batch_size)
    for i in range(N):
        for bid in range(batch_size):
            var idx = N * bid + i
            x_pert[idx] = ftz(ftz(xin[idx]) + h)
        var pert = _loglike_packed(y_kf, exog_kf, batch_size, n_obs_kf, order_kf, x_pert)
        for bid in range(batch_size):
            var diff = ftz(ftz(pert[bid]) - ftz(base[bid]))
            grad[N * bid + i] = ftz(diff / h)
        for bid in range(batch_size):
            var idx = N * bid + i
            x_pert[idx] = xin[idx]
    for b in range(batch_size):
        fout[b] = ftz(ftz(-base[b]) / scale)
    for i in range(len(xin)):
        gout[i] = ftz(ftz(-grad[i]) / scale)


# ===========================================================================
# THE L-BFGS RULES (`arima/impl/lbfgs_host.mojo`)
# ===========================================================================


def _nrm_max_at(v: List[Float32], base: Int, n: Int) -> Float32:
    """`nrm_max_at` (`lbfgs_host.mojo:84-92`)."""
    var acc = Float32(0.0)
    for i in range(n):
        var x = abs(v[base + i])
        if x > acc:
            acc = x
    return acc


def _dot_at(u: List[Float32], ub: Int, v: List[Float32], vb: Int, n: Int) -> Float32:
    """`dot_at` (`lbfgs_host.mojo:95-100`), serial ascending."""
    var acc = Float32(0.0)
    for i in range(n):
        acc = ftz(identical_mul_add(u[ub + i], v[vb + i], acc))
    return acc


def _nrm2_at(v: List[Float32], base: Int, n: Int) -> Float32:
    """`nrm2_at` (`lbfgs_host.mojo:103-105`)."""
    return ftz(identical_sqrt(_dot_at(v, base, v, base, n)))


def _armijo_ok(fx: Float32, fx_init: Float32, step: Float32, dg_test: Float32) -> Bool:
    """`armijo_ok` (`lbfgs_host.mojo:113-126`), the fused test."""
    return not (fx > identical_mul_add(step, dg_test, fx_init))


def _check_convergence(
    k: Int, fx: Float32, gnorm: Float32, mut fx_hist: List[Float32], hist_base: Int
) -> Bool:
    """`check_convergence_at` (`lbfgs_host.mojo:129-151`)."""
    var fmag = max(fx, AH_EPSILON)
    if gnorm <= AH_EPSILON * fmag:
        return True
    if AH_PAST > 0:
        if (
            k >= AH_PAST
            and abs(fx_hist[hist_base + k % AH_PAST] - fx) <= AH_DELTA * fmag
        ):
            return True
        fx_hist[hist_base + k % AH_PAST] = fx
    return False


def _lbfgs_verdict(
    iter: Int, lsret: Int, fx: Float32, fxp: Float32, gnorm: Float32,
    mut fx_hist: List[Float32], hist_base: Int, mut outcode: Int, mut restore: Bool,
) -> Bool:
    """`lbfgs_verdict` (`lbfgs_host.mojo:154-201`), the in-doubt arm included."""
    var stop = False
    var converged = False
    var is_ls_valid = (not isnan(fx)) and (not isinf(fx))
    var is_ls_non_critical = (
        lsret == AH_LS_INVALID_STEP_MIN or lsret == AH_LS_MAX_ITERS_REACHED
    )
    var is_ls_in_doubt = is_ls_valid and fx <= fxp + AH_FTOL and is_ls_non_critical
    var is_ls_success = lsret == AH_LS_SUCCESS or is_ls_in_doubt

    if is_ls_valid:
        converged = _check_convergence(iter, fx, gnorm, fx_hist, hist_base)

    if (not is_ls_success) and (not converged):
        outcode = AH_OPT_LS_FAILED
        stop = True
    elif not is_ls_valid:
        outcode = AH_OPT_NUMERIC_ERROR
        stop = True
    elif converged:
        outcode = AH_OPT_SUCCESS
        stop = True
    elif is_ls_in_doubt and fx + AH_FTOL >= fxp:
        outcode = AH_OPT_LS_FAILED
        stop = True

    restore = (not is_ls_success) or (not is_ls_valid)
    return stop


def _lbfgs_search_dir(
    mut n_vec: Int,
    end_prev: Int,
    S: List[Float32],
    Y: List[Float32],
    g: List[Float32],
    mut drt: List[Float32],
    mut yhist: List[Float32],
    mut alpha: List[Float32],
    bid: Int,
    n: Int,
) -> Int:
    """`lbfgs_search_dir_at` (`lbfgs_host.mojo:209-269`): the skipping test
    `ys <= eps * yy`, then the two-loop recursion for series `bid`."""
    var end = end_prev
    var m = AH_M
    var sb = (bid * m + end) * n
    var yb = (bid * m + end) * n
    var gb = bid * n
    var ys = _dot_at(S, sb, Y, yb, n)
    var yy = _dot_at(Y, yb, Y, yb, n)
    if ys <= AH_FLOAT_EPSILON * yy:
        return end
    n_vec += 1
    yhist[bid * m + end] = ys

    for i in range(n):
        drt[gb + i] = ftz(Float32(-1.0) * g[gb + i])
    var bound = min(m, n_vec)
    end = (end + 1) % m
    var j = end
    for _ in range(bound):
        j = (j + m - 1) % m
        var a = ftz(_dot_at(S, (bid * m + j) * n, drt, gb, n) / yhist[bid * m + j])
        alpha[bid * m + j] = a
        for i in range(n):
            drt[gb + i] = ftz(
                identical_mul_add(ftz(-a), Y[(bid * m + j) * n + i], drt[gb + i])
            )

    var scale = ftz(ys / yy)
    for i in range(n):
        drt[gb + i] = ftz(scale * drt[gb + i])

    for _ in range(bound):
        var beta = ftz(_dot_at(Y, (bid * m + j) * n, drt, gb, n) / yhist[bid * m + j])
        var c = ftz(alpha[bid * m + j] - beta)
        for i in range(n):
            drt[gb + i] = ftz(
                identical_mul_add(c, S[(bid * m + j) * n + i], drt[gb + i])
            )
        j = (j + 1) % m

    return end


@fieldwise_init
struct LbfgsHostOut(Movable):
    var x: List[Float32]
    var fx: List[Float32]
    var n_iter: List[Int32]
    var retcode: List[Int32]


def _batched_min_lbfgs(
    y_kf: List[Float32], exog_kf: List[Float32], batch_size: Int, n_obs_kf: Int, scale: Float32,
    order_kf: ArimaHostOrder, x0: List[Float32], max_iterations: Int, h: Float32,
) raises -> LbfgsHostOut:
    """`batched_min_lbfgs` (`batched_fit.mojo:290-555`), statement for
    statement without the trace records (no arithmetic)."""
    var n = order_kf.complexity()
    var b_n = batch_size * n
    var m = AH_M
    var past = AH_PAST

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
        lsret.append(AH_LS_SUCCESS)
        ls_iters.append(0)
        n_iter.append(Int32(0))
        retcode.append(Int32(AH_OPT_MAX_ITERS_REACHED))

    # `:389-409`: evaluate at x0, exit early per series at a minimizer.
    _eval_batch(y_kf, exog_kf, batch_size, n_obs_kf, order_kf, h, scale, x, fx, grad)
    for b in range(batch_size):
        gnorm[b] = _nrm_max_at(grad, b * n, n)
        if past > 0:
            fx_hist[b * past] = fx[b]
        if _check_convergence(0, fx[b], gnorm[b], fx_hist, b * past):
            retcode[b] = Int32(AH_OPT_SUCCESS)
            active[b] = False
            n_iter[b] = Int32(0)
        else:
            for i in range(n):
                drt[b * n + i] = ftz(Float32(-1.0) * grad[b * n + i])
            step[b] = ftz(Float32(1.0) / _nrm2_at(drt, b * n, n))
            fxp[b] = fx[b]

    var k = 1
    while k <= max_iterations:
        var any_active = False
        for b in range(batch_size):
            if active[b]:
                any_active = True
        if not any_active:
            break

        # `:421-427`: save x, grad, fx
        for b in range(batch_size):
            if not active[b]:
                continue
            for i in range(n):
                xp[b * n + i] = x[b * n + i]
                gradp[b * n + i] = grad[b * n + i]
            fxp[b] = fx[b]

        # `:430-446`: the line search's preamble
        for b in range(batch_size):
            searching[b] = False
            if not active[b]:
                continue
            if step[b] <= Float32(0.0):
                lsret[b] = AH_LS_INVALID_STEP
                continue
            fx_init[b] = fx[b]
            dg_init[b] = _dot_at(grad, b * n, drt, b * n, n)
            if dg_init[b] > Float32(0.0):
                lsret[b] = AH_LS_INVALID_DIR
                continue
            dg_test[b] = ftz(AH_FTOL * dg_init[b])
            ls_iters[b] = 0
            lsret[b] = AH_LS_MAX_ITERS_REACHED
            searching[b] = True

        # `:449-490`: the shared line search
        for _t in range(AH_MAX_LINESEARCH):
            var any_s = False
            for b in range(batch_size):
                if searching[b]:
                    any_s = True
            if not any_s:
                break
            for b in range(batch_size):
                if searching[b]:
                    for i in range(n):
                        cand[b * n + i] = ftz(
                            identical_mul_add(step[b], drt[b * n + i], xp[b * n + i])
                        )
                else:
                    for i in range(n):
                        cand[b * n + i] = x[b * n + i]
            _eval_batch(y_kf, exog_kf, batch_size, n_obs_kf, order_kf, h, scale, cand, fxc, gradc)
            for b in range(batch_size):
                if not searching[b]:
                    continue
                for i in range(n):
                    x[b * n + i] = cand[b * n + i]
                    grad[b * n + i] = gradc[b * n + i]
                fx[b] = fxc[b]
                ls_iters[b] += 1
                if _armijo_ok(fx[b], fx_init[b], step[b], dg_test[b]):
                    lsret[b] = AH_LS_SUCCESS
                    searching[b] = False
                elif step[b] < AH_MIN_STEP:
                    lsret[b] = AH_LS_INVALID_STEP_MIN
                    searching[b] = False
                elif step[b] > AH_MAX_STEP:
                    lsret[b] = AH_LS_INVALID_STEP_MAX
                    searching[b] = False
                else:
                    step[b] = ftz(step[b] * AH_LS_DEC)

        # `:493-529`: verdict, history update, new direction
        for b in range(batch_size):
            if not active[b]:
                continue
            gnorm[b] = _nrm_max_at(grad, b * n, n)
            var code = Int(retcode[b])
            var restore = False
            var stop = _lbfgs_verdict(
                k, lsret[b], fx[b], fxp[b], gnorm[b], fx_hist, b * past, code, restore,
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
                S[(b * m + e) * n + i] = ftz(
                    identical_mul_add(Float32(-1.0), xp[b * n + i], x[b * n + i])
                )
                Y[(b * m + e) * n + i] = ftz(
                    identical_mul_add(Float32(-1.0), gradp[b * n + i], grad[b * n + i])
                )
            var nv = n_vec[b]
            endv[b] = _lbfgs_search_dir(nv, e, S, Y, grad, drt, yhist, alpha, b, n)
            n_vec[b] = nv
            step[b] = Float32(1.0)
        k += 1

    return LbfgsHostOut(x=x^, fx=fx^, n_iter=n_iter^, retcode=retcode^)


# ===========================================================================
# THE STARTING POINT (`arima/impl/estimate_x0.mojo`)
# ===========================================================================


@always_inline
def _ls_p_ar(p: Int, q: Int, s: Int) -> Int:
    """`ls_p_ar` (`estimate_x0.mojo:113-117`)."""
    var ps = p * s
    var qs2 = 2 * q * s
    return ps if ps > qs2 else qs2


@always_inline
def _ls_r(p: Int, q: Int, s: Int) -> Int:
    """`ls_r` (`estimate_x0.mojo:121-125`)."""
    var ps = p * s
    var a = _ls_p_ar(p, q, s) + q * s
    return a if a > ps else ps


def _qr_solve(mut a: List[Float32], m: Int, n: Int, mut b: List[Float32]) -> Int32:
    """`householder_qr_solve` (`least_squares.mojo:111-210`) over one system
    at offset 0, `a` destroyed, the solution in `b[0..n)`. The two in-place
    subtractions read the raw cell, as the device's do (`:169`, `:184`)."""
    var rdiag = Array[Float32, AH_LS_MAX_COLS](fill=Float32(0.0))
    for j in range(n):
        var sigma = Float32(0.0)
        for i in range(j, m):
            var v = ftz(a[i + j * m])
            sigma = ftz(identical_mul_add(v, v, sigma))
        if sigma == Float32(0.0):
            return Int32(j + 1)
        var normx = ftz(identical_sqrt(sigma))
        var ajj = ftz(a[j + j * m])
        var s = Float32(-1.0) if ajj >= Float32(0.0) else Float32(1.0)
        var r_jj = ftz(s * normx)
        var u1 = ftz(ajj - r_jj)
        if u1 == Float32(0.0):
            return Int32(j + 1)
        for i in range(j + 1, m):
            a[i + j * m] = ftz(ftz(a[i + j * m]) / u1)
        var tau = ftz(ftz(ftz(-s) * u1) / normx)
        rdiag[j] = r_jj
        for c in range(j + 1, n):
            var acc = ftz(a[j + c * m])
            for i in range(j + 1, m):
                var w = ftz(a[i + j * m])
                var x = ftz(a[i + c * m])
                acc = ftz(identical_mul_add(w, x, acc))
            var td = ftz(tau * acc)
            a[j + c * m] = ftz(a[j + c * m] - td)
            for i in range(j + 1, m):
                var w = ftz(a[i + j * m])
                var cur = ftz(a[i + c * m])
                a[i + c * m] = ftz(identical_mul_add(-td, w, cur))
        var accb = ftz(b[j])
        for i in range(j + 1, m):
            var w = ftz(a[i + j * m])
            var x = ftz(b[i])
            accb = ftz(identical_mul_add(w, x, accb))
        var tdb = ftz(tau * accb)
        b[j] = ftz(b[j] - tdb)
        for i in range(j + 1, m):
            var w = ftz(a[i + j * m])
            var cur = ftz(b[i])
            b[i] = ftz(identical_mul_add(-tdb, w, cur))
    var rmax = Float32(0.0)
    for j in range(n):
        var v = abs(rdiag[j])
        if v > rmax:
            rmax = v
    if rmax == Float32(0.0):
        return Int32(1)
    for j in range(n):
        if abs(rdiag[j]) <= ftz(AH_LS_RANK_TOL * rmax):
            return Int32(j + 1)
    var i = n - 1
    while i >= 0:
        var acc = ftz(b[i])
        for c in range(i + 1, n):
            var u = ftz(a[i + c * m])
            var xc = ftz(b[c])
            acc = ftz(identical_mul_add(-u, xc, acc))
        b[i] = ftz(acc / rdiag[i])
        i -= 1
    return Int32(0)


def _test_invparams(params: List[Float32], base: Int, pq: Int, is_ar: Bool) -> Bool:
    """`test_invparams` (`estimate_x0.mojo:171-206`): the inverse recursion
    stopped before atanh, with `coef * a * x` as `(coef*a) * x`, ONE
    rounding, then strictly inside (-1, 1)."""
    var new_params = Array[Float32, AH_JONES_MAX](fill=Float32(0.0))
    var tmp = Array[Float32, AH_JONES_MAX](fill=Float32(0.0))
    for i in range(pq):
        var v = ftz(params[base + i])
        tmp[i] = v
        new_params[i] = v
    var j = pq - 1
    while j > 0:
        var a = new_params[j]
        var coef_a = a if is_ar else ftz(-a)
        var den = ftz(identical_mul_add(-a, a, Float32(1.0)))
        for k in range(j):
            var num = ftz(
                identical_mul_add(coef_a, new_params[j - k - 1], new_params[k])
            )
            tmp[k] = ftz(num / den)
        for it in range(j):
            new_params[it] = tmp[it]
        j -= 1
    var result = True
    for i in range(pq):
        var v = new_params[i]
        result = result and not (v <= Float32(-1.0) or v >= Float32(1.0))
    return result


def _ls_refusal_fill(
    bid: Int, p: Int, q: Int, k: Int, est_sigma2: Bool,
    mut ar: List[Float32], mut ma: List[Float32],
    mut sigma2: List[Float32], mut mu: List[Float32],
):
    """The fill of a refused solve (`estimate_x0.mojo:320-328`, `:377-385`)
    and of the degenerate arm (`:225-238`): zeros, and 1 for sigma2."""
    if k != 0:
        mu[bid] = Float32(0.0)
    for i in range(p):
        ar[p * bid + i] = Float32(0.0)
    for i in range(q):
        ma[q * bid + i] = Float32(0.0)
    if est_sigma2:
        sigma2[bid] = Float32(1.0)


def _arma_least_squares(
    mut ar: List[Float32], mut ma: List[Float32],
    mut sigma2: List[Float32], mut mu: List[Float32],
    yd: List[Float32], batch_size: Int, n_obs_d: Int,
    p: Int, q: Int, s: Int, est_sigma2: Bool, k: Int,
) raises:
    """`arma_least_squares` (`estimate_x0.mojo:445-513`) and its kernel
    (`:241-428`) one series at a time."""
    var p_ar = _ls_p_ar(p, q, s)
    var r_ls = _ls_r(p, q, s)
    if (q != 0 and p_ar >= n_obs_d - p_ar) or (p + q + k >= n_obs_d - r_ls):
        for bid in range(batch_size):
            _ls_refusal_fill(bid, p, q, k, est_sigma2, ar, ma, sigma2, mu)
        return
    if p + q + k > AH_LS_MAX_COLS or p_ar > AH_LS_MAX_COLS:
        raise Error(
            "estimate_x0: the least-squares system has "
            + String(max(p + q + k, p_ar))
            + " columns, above LS_MAX_COLS = " + String(AH_LS_MAX_COLS)
            + "; refused by name (arima/NOT_IMPLEMENTED.tsv)"
        )
    var ps = p * s
    var qs = q * s
    var m1 = n_obs_d - r_ls
    var ncols = p + q + k
    for bid in range(batch_size):
        var yb = bid * n_obs_d
        var lsar = _zeros(max(1, m1 * ncols))
        var afit = _zeros(max(1, m1))
        var fres = _zeros(max(1, m1))

        # -- 1. residuals of an AR(p_ar) fit (:290-349)
        if q != 0:
            var m2 = n_obs_d - p_ar
            var pre = _zeros(max(1, m2 * p_ar))
            for lag in range(p_ar):
                var src = yb + (p_ar - lag - 1)
                for i in range(m2):
                    pre[lag * m2 + i] = ftz(yd[src + i])
            var arfit = _zeros(max(1, m2))
            var resid = _zeros(max(1, m2))
            for i in range(m2):
                var v = ftz(yd[yb + p_ar + i])
                arfit[i] = v
                resid[i] = v
            var preq = pre.copy()
            var inf_pre = _qr_solve(preq, m2, p_ar, arfit)
            if inf_pre != Int32(0):
                _ls_refusal_fill(bid, p, q, k, est_sigma2, ar, ma, sigma2, mu)
                continue
            for i in range(m2):
                var acc = ftz(resid[i])
                for c in range(p_ar):
                    var av = ftz(pre[c * m2 + i])
                    var xv = ftz(arfit[c])
                    acc = ftz(identical_mul_add(-av, xv, acc))
                resid[i] = acc
            var res_offset = r_ls - p_ar - qs
            for lag in range(q):
                var src = res_offset + s * (q - lag - 1)
                var dst = m1 * (k + p) + lag * m1
                for i in range(m1):
                    lsar[dst + i] = resid[src + i]

        # -- 2. the intercept column (:352-354)
        if k != 0:
            for i in range(m1):
                lsar[i] = Float32(1.0)

        # -- 3. lags of y (:357-362)
        var ar_offset = r_ls - ps
        for lag in range(p):
            var src = yb + ar_offset + s * (p - lag - 1)
            var dst = m1 * k + lag * m1
            for i in range(m1):
                lsar[dst + i] = ftz(yd[src + i])

        # -- 4. the target and the residual it seeds (:365-369)
        for i in range(m1):
            var v = ftz(yd[yb + r_ls + i])
            afit[i] = v
            if est_sigma2:
                fres[i] = v

        # -- 5. the ARMA fit (:372-385)
        var lsarq = lsar.copy()
        var inf = _qr_solve(lsarq, m1, ncols, afit)
        if inf != Int32(0):
            _ls_refusal_fill(bid, p, q, k, est_sigma2, ar, ma, sigma2, mu)
            continue

        # -- 6. the solution into the parameter vectors (:388-393)
        if k != 0:
            mu[bid] = afit[0]
        for i in range(p):
            ar[p * bid + i] = afit[i + k]
        for i in range(q):
            ma[q * bid + i] = afit[i + p + k]

        # -- 7. sigma2 from the final residual (:396-410)
        if est_sigma2:
            for i in range(m1):
                var acc = ftz(fres[i])
                for c in range(ncols):
                    var av = ftz(lsar[c * m1 + i])
                    var xv = ftz(afit[c])
                    acc = ftz(identical_mul_add(-av, xv, acc))
                fres[i] = acc
            var acc2 = Float32(0.0)
            for i in range(q, m1):
                var res = ftz(fres[i])
                acc2 = ftz(identical_mul_add(res, res, acc2))
            sigma2[bid] = ftz(acc2 / Float32(m1 - q))

        # -- 8. zero what the inverse transform would reject (:413-427)
        if p != 0:
            if not _test_invparams(ar, p * bid, p, True):
                for ip in range(p):
                    ar[p * bid + ip] = Float32(0.0)
        if q != 0:
            if not _test_invparams(ma, q * bid, q, False):
                for iq in range(q):
                    ma[q * bid + iq] = Float32(0.0)


def _exog_regression(
    xd: List[Float32], mut yd: List[Float32], mut beta: List[Float32],
    batch_size: Int, m: Int, n_exog: Int,
):
    """`exog_regression_kernel` (`estimate_x0.mojo`) one series at a time:
    the QR solve of the differenced `y` on the differenced regressors, `beta
    = 0` when it refuses, then `y - exog * beta` with DEVIATION 995's fold
    over the original regressors."""
    for bid in range(batch_size):
        var xb = bid * n_exog * m
        var yb = bid * m
        var a = _zeros(m * n_exog)
        for i in range(n_exog):
            for t in range(m):
                a[i * m + t] = ftz(xd[xb + i * m + t])
        var b = _zeros(m)
        for t in range(m):
            b[t] = ftz(yd[yb + t])
        var inf = _qr_solve(a, m, n_exog, b)
        for i in range(n_exog):
            beta[bid * n_exog + i] = b[i] if inf == Int32(0) else Float32(0.0)
        for t in range(m):
            var acc = Float32(0.0)
            for i in range(n_exog):
                var xv = ftz(xd[xb + i * m + t])
                var bv = ftz(beta[bid * n_exog + i])
                acc = ftz(identical_mul_add(xv, bv, acc))
            var yv = ftz(yd[yb + t])
            yd[yb + t] = ftz(yv - acc)


def _estimate_x0(
    y: List[Float32], exog: List[Float32], batch_size: Int, n_obs: Int, order: ArimaHostOrder
) raises -> ArimaHostParams:
    """`estimate_x0` (`estimate_x0.mojo:591-618`) then `start_params`
    (`:540-588`): the non-seasonal call estimates sigma2; the seasonal call
    (P, Q at period s, no intercept) does not when the first ran."""
    var d_sD = order.n_diff()
    if n_obs <= d_sD:
        raise Error(
            "estimate_x0: n_obs (" + String(n_obs)
            + ") must be greater than d + s*D (" + String(d_sD)
            + ") for differencing"
        )
    var n_obs_d = n_obs - d_sD
    var yd = _prepare_data(y, batch_size, n_obs, order)
    var params = _new_params(order, batch_size)
    if order.n_exog > 0:
        # `estimate_x0_x`: the regressors differenced over `n_exog *
        # batch_size` series, the regression when there are more rows than
        # regressors, else beta = 0 and y untouched.
        var xd = _prepare_data(exog, order.n_exog * batch_size, n_obs, order)
        if n_obs_d > order.n_exog:
            _exog_regression(xd, yd, params.beta, batch_size, n_obs_d, order.n_exog)
    var ns_run = order.p + order.q + order.k != 0
    var seasonal_run = order.P + order.Q != 0
    if not ns_run:
        raise Error(
            "arima host: start_params reached with p + q + k = 0;"
            " arima_host_refuse_unrestated refuses it first"
        )
    _arma_least_squares(
        params.ar, params.ma, params.sigma2, params.mu, yd, batch_size, n_obs_d,
        order.p, order.q, 1, True, order.k,
    )
    if seasonal_run:
        _arma_least_squares(
            params.sar, params.sma, params.sigma2, params.mu, yd, batch_size,
            n_obs_d, order.P, order.Q, order.s, False, 0,
        )
    return params^


# ===========================================================================
# THE PUBLIC ENTRIES
# ===========================================================================


@fieldwise_init
struct ArimaHostFit(Movable):
    """`FitResult` (`batched_fit.mojo:563-583`) plus `_loglike_at`'s value."""

    var t_x: List[Float32]
    var x: List[Float32]
    var x0: List[Float32]
    var loglike: List[Float32]
    var fx: List[Float32]
    var n_iter: List[Int32]
    var retcode: List[Int32]


def arima_host_fit(
    y: List[Float32], exog: List[Float32], batch_size: Int, n_obs: Int, order: ArimaHostOrder,
    max_iterations: Int,
) raises -> ArimaHostFit:
    """`batched_fit` (`batched_fit.mojo:586-717`) and `_loglike_at`
    (`arima/estimator.mojo:232-274`). The caller has run the entry's
    validation (`arima_fit_ptr_host:361-368`) and
    `arima_host_refuse_unrestated`."""
    if n_obs < 2:
        raise Error("batched_fit: n_obs must be at least 2 (got " + String(n_obs) + ")")
    _refuse_non_finite(y, batch_size * n_obs, "y")

    # 1. the starting parameters
    var start = _estimate_x0(y, exog, batch_size, n_obs, order)

    # 2. into the unconstrained coordinates
    var N = order.complexity()
    var inv = _batched_jones(order, batch_size, True, start)
    var x0 = _pack(inv, order, batch_size)
    for i in range(N * batch_size):
        if not isfinite(x0[i]):
            raise Error(
                "batched_fit: the initial parameter vector has a non-finite"
                " value at index " + String(i)
                + "; estimate_x0 produced a parameter the inverse Jones"
                " transform could not map (arima/README.md, DEVIATION 678)"
            )

    # 3. difference once, then optimize on the differenced series
    var diff = order.need_diff()
    var n_obs_kf = n_obs - order.n_diff() if diff else n_obs
    var order_kf = order.without_diff() if diff else order
    var y_kf = _prepare_data(y, batch_size, n_obs, order)
    var exog_kf = _zeros(1)
    if order.n_exog > 0:
        exog_kf = _prepare_data(exog, order.n_exog * batch_size, n_obs, order)
    var res = _batched_min_lbfgs(
        y_kf, exog_kf, batch_size, n_obs_kf, Float32(n_obs - 1), order_kf, x0,
        max_iterations, arima_host_fit_h(),
    )

    # 4 and 5. forward-transform the answer
    var raw = _unpack(res.x, order, batch_size)
    var fitted = _batched_jones(order, batch_size, False, raw)
    var t_x = _pack(fitted, order, batch_size)

    # `_loglike_at`: one more pass at the fitted point, trans = false (the
    # `_copy_params` arm, no transform and no floor)
    var at = _kalman(y_kf, exog_kf, _zeros(1), n_obs_kf, fitted, order_kf, batch_size, 0)
    return ArimaHostFit(
        t_x=t_x^, x=res.x.copy(), x0=x0^, loglike=at.loglike.copy(),
        fx=res.fx.copy(), n_iter=res.n_iter.copy(), retcode=res.retcode.copy(),
    )


def _filter_exog(
    exog: List[Float32], exog_fut: List[Float32], batch_size: Int, n_obs: Int,
    num_steps: Int, order: ArimaHostOrder,
) raises -> Tuple[List[Float32], List[Float32]]:
    """`predict_x`'s regressors for the filter (`batched_arima.mojo`): the
    past differenced like `y` and the future through `prepare_future_data`,
    or both as they are when nothing is differenced; placeholders when
    `n_exog = 0`."""
    if order.n_exog == 0:
        return (_zeros(1), _zeros(1))
    var n_ser = order.n_exog * batch_size
    var past = _prepare_data(exog, n_ser, n_obs, order)
    var fut = _zeros(1)
    if num_steps > 0:
        fut = _prepare_future(exog, exog_fut, n_ser, n_obs, num_steps, order)
    return (past^, fut^)


def arima_host_forecast(
    y: List[Float32], exog: List[Float32], exog_fut: List[Float32],
    params_packed: List[Float32], batch_size: Int, n_obs: Int,
    n_steps: Int, order: ArimaHostOrder,
) raises -> List[Float32]:
    """`predict` (`batched_arima.mojo:287-348`) with `start == n_obs`, `end =
    n_obs + n_steps` and `pre_diff = true`: the differencing or the copy, the
    filter's non-finite refusal on its input (`batched_loglike:129-130`),
    `unpack` and the `trans = false` copy, the filter with `fc_steps`,
    `finalize_forecast`, and `copy_forecast_kernel` (`:261-277`) at offset
    `n_obs - start = 0`. Series major, `n_steps * batch_size` values. The
    caller has run `_predict_into`'s checks (`arima/estimator.mojo:422-433`)."""
    var diff = order.need_diff()
    var n_obs_kf = n_obs - order.n_diff() if diff else n_obs
    var order_kf = order.without_diff() if diff else order
    var y_kf = _prepare_data(y, batch_size, n_obs, order)
    _refuse_non_finite(y_kf, n_obs_kf * batch_size, "y")
    var params = _unpack(params_packed, order, batch_size)
    var xs = _filter_exog(exog, exog_fut, batch_size, n_obs, n_steps, order)
    var kf = _kalman(y_kf, xs[0], xs[1], n_obs_kf, params, order_kf, batch_size, n_steps)
    var fc = kf.fc.copy()
    if diff:
        _finalize_forecast(fc, y, n_steps, batch_size, n_obs, n_obs, order)
    var out = _zeros(n_steps * batch_size)
    for bid in range(batch_size):
        for i in range(n_steps):
            out[bid * n_steps + i] = fc[n_steps * bid + i]
    _predict_sabotage(out)
    return out^


def _predict_sabotage(mut out: List[Float32]):
    """ARIMA_ORACLE_PREDICT_SABOTAGE: the lowest bit of every finite value."""
    comptime if ARIMA_ORACLE_PREDICT_SABOTAGE:
        for i in range(len(out)):
            if isfinite(out[i]):
                out[i] = bitcast[DType.float32](bitcast[DType.uint32](out[i]) ^ UInt32(1))


def arima_host_predict(
    y: List[Float32], exog: List[Float32], exog_fut: List[Float32],
    params_packed: List[Float32], batch_size: Int, n_obs: Int,
    start: Int, end: Int, order: ArimaHostOrder,
) raises -> List[Float32]:
    """`predict` (`batched_arima.mojo:287-348`) with `pre_diff = true` for
    any `0 <= start < end` with `start <= n_obs` (lane/inference-forecast-umap-pca,
    2026-09-15): the differencing or the copy, the filter's non-finite refusal
    on its input, `unpack`, the filter with `num_steps = max(end - n_obs, 0)`
    forecast steps, then `in_sample_prediction_kernel` (`:211-250`) over
    `[start, min(n_obs, end))`: the canonical quiet NaN (DEVIATION 676) before
    `res_offset = d + s * D`, the filter's one-step prediction when nothing
    was differenced, and `ftz(y[i - period1] + pred[i - res_offset])` after
    one difference; then `finalize_forecast` and `copy_forecast_kernel`
    (`:261-277`) at offset `n_obs - start`. Series major, `(end - start) *
    batch_size` values. `arima_host_refuse_unrestated` has refused `d + D >
    1` before this is reached, so the two-difference arm is not restated."""
    var diff = order.need_diff()
    var n_obs_kf = n_obs - order.n_diff() if diff else n_obs
    var order_kf = order.without_diff() if diff else order
    var num_steps = end - n_obs if end > n_obs else 0
    var y_kf = _prepare_data(y, batch_size, n_obs, order)
    _refuse_non_finite(y_kf, n_obs_kf * batch_size, "y")
    var params = _unpack(params_packed, order, batch_size)
    var xs = _filter_exog(exog, exog_fut, batch_size, n_obs, num_steps, order)
    var kf = _kalman(y_kf, xs[0], xs[1], n_obs_kf, params, order_kf, batch_size, num_steps)
    var ld = end - start
    var out = _zeros(ld * batch_size)
    if start < n_obs:
        var res_offset = order.n_diff() if diff else 0
        var p_start = start if start > res_offset else res_offset
        var p_end = n_obs if n_obs < end else end
        var dD = order.d + order.D if diff else 0
        var period1 = 1 if order.d != 0 else order.s
        if dD > 1:
            raise Error(
                "arima host: in-sample prediction reached with d + D = " + String(dD)
                + "; arima_host_refuse_unrestated refuses it first"
            )
        for bid in range(batch_size):
            for i in range(res_offset - start):
                out[bid * ld + i] = bitcast[DType.float32](UInt32(0x7FC00000))
            for i in range(p_start, p_end):
                var v: Float32
                if dD == 0:
                    v = ftz(kf.pred[bid * n_obs + i])
                else:
                    var a = ftz(y[bid * n_obs + i - period1])
                    var b = ftz(kf.pred[bid * n_obs_kf + i - res_offset])
                    v = ftz(a + b)
                out[bid * ld + i - start] = v
    if num_steps > 0:
        var fc = kf.fc.copy()
        if diff:
            _finalize_forecast(fc, y, num_steps, batch_size, n_obs, n_obs, order)
        var off = n_obs - start
        for bid in range(batch_size):
            for i in range(num_steps):
                out[bid * ld + off + i] = fc[num_steps * bid + i]
    _predict_sabotage(out)
    return out^
