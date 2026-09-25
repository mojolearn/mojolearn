# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`batched_loglike`, `predict`, `batched_loglike_grad`, `batched_diff`: the
C++ entry points the Python `ARIMA` class calls.

Reference: `cuml/cpp/src/arima/batched_arima.cu` (cuML 265b9da6, v26.08.00):
`batched_diff` (:60-70), `predict` (:86-267), `batched_loglike` (the
`ARIMAParams` overload :393-469 and the packed-vector overload :471-513),
`batched_loglike_grad` (:515-591).

NOT IMPLEMENTED, refused by name: `method == CSS` (`conditional_sum_of_squares`,
`sum_of_squares_kernel` :270-391; only `MLE` is offered, `truncate` must
be 0), `information_criterion` (:592-625), `detect_missing` (NaN is
refused, not detected), `level > 0` (confidence intervals). See
`arima/NOT_IMPLEMENTED.tsv`.

EXOGENOUS REGRESSORS (lane/arima-exog, 2026-09-15): the `_x` entries take the
filter's exogenous inputs, `d_exog` over the observations and `d_exog_fut`
over the forecast steps, both `[bid*n_exog*n + i*n + t]`. `predict_x`
differences them as theirs does (`:117-157`: `prepare_data` over `n_exog *
batch_size` series, `prepare_future_data` for the future). The entries
without `_x` are the `n_exog = 0` doors the checks and the card call; each
hands the `_x` entry one-float placeholders nothing reads.

CORRECTED 2026-09-01. This header used to list `estimate_x0` /
`_start_params` / `_arma_least_squares` (`:627-1010`) as unimplemented because
they reach cuBLAS `b_gels`, and to say "there is no `fit`". Both sentences
are now false. That chain is implemented in `arima/impl/estimate_x0.mojo`,
the closed `b_gels` is written out as a Householder QR (DEVIATION 678,
`arima/impl/linalg/batched/least_squares.mojo`), the optimizer is
DEVIATION 679 in `arima/impl/batched_fit.mojo`, and `batched_fit` is
the public entry point. A closed vendor library is a reason to write the
routine, not a reason to refuse the capability.

PREDICT (`:86-267`) with `simple_differencing` (`pre_diff`), their default:
the series is differenced (`prepare_data`), the filter runs on the
differenced series with `d = D = 0`, the in-sample prediction at `i` is
`y[i - period1] + pred[i - res_offset]` (one difference) or
`((y[i-p1] + y[i-p2]) - y[i-p1-p2]) + pred[...]` (two; C++ left to right),
the forecasts are undifferenced by `finalize_forecast`, and the first
`res_offset - start` predictions are UNDEFINED.

=============================================================================
DEVIATION 676: THE UNDEFINED PREDICTIONS ARE THE CANONICAL NaN, BY CONSTANT
=============================================================================
THEIRS. `d_y_p[..] = nan("")` (`:209`), the vendor's quiet NaN (Apple
0x7fc00000, NVIDIA 0x7fffffff) -- a payload that differs per vendor in a
buffer the card records.
OURS. The sentinel is the bit pattern `0x7FC00000` written as a constant
(`bitcast`), never computed, so the recorded `arima.pred` bytes are the
same on every vendor; the caller sees a NaN exactly where theirs does.

FOUND IN THEIRS, NOT REPRODUCED (`batched_arima.cu:207`): `d_y_p[0] = 0.0`
is the FIRST statement of the per-series lambda, and it writes element 0 of
the WHOLE output -- not of `bid`'s row. Every one of the `batch_size`
threads writes it. An earlier revision of this file called that "always
overwritten by the loop below it, a benign race with no effect"; the audit
of 2026-08-23 read the lambda again and that is WRONG. Thread 0 writes the
real value of `d_y_p[0]` (either the NaN sentinel at `i = 0`, or `i =
p_start`'s prediction) inside its own loop, but threads `bid > 0` are
unordered against it and any of them may land its `0.0` AFTER. `d_y_p[0]`
can therefore come back 0.0 instead of the prediction or the sentinel, on
their hardware, for any batch of more than one series. It is a genuine
data race with an observable result, it is in `arima/NOT_IMPLEMENTED.tsv` as an
reference defect, and the statement is not implemented: `assume-our-code-is-
broken`'s rule is to fix their bug rather than copy it, and the fix here
is to not write the cell at all.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.math import isfinite
from std.memory import bitcast

from arima.impl.batched_kalman import KalmanWorkspace, batched_kalman_filter_x
from arima.impl.timeSeries.arima_helpers import (
    batched_jones_transform,
    finalize_forecast,
    prepare_future_data,
)
from arima.impl.tsa.arima_common import ARIMAOrder, ARIMAParams, unpack, validate_order
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, ftz
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from tsa.impl.timeSeries.arima_helpers import prepare_data


comptime CANONICAL_NAN_BITS = UInt32(0x7FC00000)


def canonical_nan() -> Float32:
    return bitcast[DType.float32](CANONICAL_NAN_BITS)


def batched_diff(
    ctx: DeviceContext,
    mut d_y_diff: DeviceBuffer[DType.float32],
    mut d_y: DeviceBuffer[DType.float32],
    batch_size: Int,
    n_obs: Int,
    order: ARIMAOrder,
) raises:
    """`:60-70`."""
    prepare_data(ctx, d_y_diff, d_y, batch_size, n_obs, order.d, order.D, order.s)


@fieldwise_init
struct LoglikeResult(Movable):
    """`batched_loglike`'s outputs: the workspace (its `loglike`, `pred`, `vs`,
    `fc`, `P0`, `alpha0` are the card's stages) and the transformed
    parameters (`arima.jones`)."""

    var ws: KalmanWorkspace
    var t_params: ARIMAParams
    var loglike: List[Float32]


def _placeholder(ctx: DeviceContext) raises -> DeviceBuffer[DType.float32]:
    """One float, for an exogenous input an `n_exog = 0` order never reads."""
    var b = ctx.enqueue_create_buffer[DType.float32](1)
    return b^


def _refuse_exog_order(order: ARIMAOrder, who: String) raises:
    if order.n_exog != 0:
        raise Error(
            who + ": n_exog=" + String(order.n_exog)
            + " needs the exogenous series; call " + who + "_x"
        )


def batched_loglike(
    ctx: DeviceContext,
    mut d_y: DeviceBuffer[DType.float32],
    batch_size: Int,
    n_obs: Int,
    order: ARIMAOrder,
    mut params: ARIMAParams,
    trans: Bool,
    fc_steps: Int = 0,
    kalman_tpb: Int = 32,
    check_finite: Bool = True,
) raises -> LoglikeResult:
    """The `n_exog = 0` door (see the module docstring)."""
    _refuse_exog_order(order, "batched_loglike")
    var e0 = _placeholder(ctx)
    var e1 = _placeholder(ctx)
    var r = batched_loglike_x(
        ctx, d_y, e0, e1, batch_size, n_obs, order, params, trans, fc_steps,
        kalman_tpb, check_finite,
    )
    _ = e0^
    _ = e1^
    return r^


def batched_loglike_x(
    ctx: DeviceContext,
    mut d_y: DeviceBuffer[DType.float32],
    mut d_exog: DeviceBuffer[DType.float32],
    mut d_exog_fut: DeviceBuffer[DType.float32],
    batch_size: Int,
    n_obs: Int,
    order: ARIMAOrder,
    mut params: ARIMAParams,
    trans: Bool,
    fc_steps: Int = 0,
    kalman_tpb: Int = 32,
    check_finite: Bool = True,
) raises -> LoglikeResult:
    """`:393-469`, the `MLE` arm with `host_loglike = true`: the Jones
    transform when `trans`, the Kalman filter, the host copy of the
    log-likelihood. `method = CSS` is refused by name by the caller.

    `check_finite` is OURS and defaults to theirs' behaviour (see
    `_refuse_non_finite`). It exists for ONE caller,
    `batched_fit`/`eval_batch`, which checks the series once before the
    optimizer starts and then evaluates this function `(N + 1)` times per
    candidate point, hundreds of times over, on a buffer nothing has
    written in between. The check is a full device-to-host copy and a
    synchronize; leaving it on inside the optimizer's inner loop is the
    dominant cost of a fit and answers a question whose answer cannot have
    changed. No other caller passes it and no bit moves either way."""
    if check_finite:
        _refuse_non_finite(ctx, d_y, batch_size * n_obs, "y")
    validate_order(order)
    var t_params = ARIMAParams(ctx, order, batch_size)
    if trans:
        batched_jones_transform(ctx, order, batch_size, False, params, t_params)
    else:
        # non-transformed case: just use original parameters (:447-452)
        _copy_params(ctx, params, t_params, order, batch_size)
    var ws = batched_kalman_filter_x(
        ctx, d_y, d_exog, d_exog_fut, n_obs, t_params, order, batch_size, fc_steps, kalman_tpb
    )
    var h = ctx.enqueue_create_host_buffer[DType.float32](batch_size)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=ws.loglike)
    ctx.synchronize()
    var ll = List[Float32]()
    for i in range(batch_size):
        ll.append(h.unsafe_ptr().unsafe_load(i))
    _ = h^
    return LoglikeResult(ws=ws^, t_params=t_params^, loglike=ll^)


def _refuse_non_finite(
    ctx: DeviceContext, buf: DeviceBuffer[DType.float32], n: Int, name: String
) raises:
    """Ours (ADDENDUM 11), as tsa's `kpss_test` does: theirs lets NaN in
    (`detect_missing`, the `isnan(yt)` arms) because NaN MEANS missing
    there; missing observations are not implemented, so a non-finite input is
    refused by name instead of silently taking the missing-data arms."""
    var h = ctx.enqueue_create_host_buffer[DType.float32](n if n > 0 else 1)
    if n > 0:
        var view = buf.create_sub_buffer[DType.float32](0, n)
        ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=view)
    ctx.synchronize()
    for i in range(n):
        var v = h.unsafe_ptr().unsafe_load(i)
        if not isfinite(v):
            raise Error(
                "batched_loglike: " + name + " contains a non-finite value at index "
                + String(i)
                + "; missing observations are not implemented and are refused by name (arima/NOT_IMPLEMENTED.tsv)"
            )
    _ = h^


def _copy_params(ctx: DeviceContext, mut src: ARIMAParams, mut dst: ARIMAParams, order: ARIMAOrder, batch_size: Int) raises:
    if order.k != 0:
        ctx.enqueue_copy(dst_buf=dst.mu.create_sub_buffer[DType.float32](0, batch_size), src_buf=src.mu.create_sub_buffer[DType.float32](0, batch_size))
    if order.n_exog != 0:
        ctx.enqueue_copy(dst_buf=dst.beta.create_sub_buffer[DType.float32](0, order.n_exog * batch_size), src_buf=src.beta.create_sub_buffer[DType.float32](0, order.n_exog * batch_size))
    if order.p != 0:
        ctx.enqueue_copy(dst_buf=dst.ar.create_sub_buffer[DType.float32](0, order.p * batch_size), src_buf=src.ar.create_sub_buffer[DType.float32](0, order.p * batch_size))
    if order.q != 0:
        ctx.enqueue_copy(dst_buf=dst.ma.create_sub_buffer[DType.float32](0, order.q * batch_size), src_buf=src.ma.create_sub_buffer[DType.float32](0, order.q * batch_size))
    if order.P != 0:
        ctx.enqueue_copy(dst_buf=dst.sar.create_sub_buffer[DType.float32](0, order.P * batch_size), src_buf=src.sar.create_sub_buffer[DType.float32](0, order.P * batch_size))
    if order.Q != 0:
        ctx.enqueue_copy(dst_buf=dst.sma.create_sub_buffer[DType.float32](0, order.Q * batch_size), src_buf=src.sma.create_sub_buffer[DType.float32](0, order.Q * batch_size))
    ctx.enqueue_copy(dst_buf=dst.sigma2.create_sub_buffer[DType.float32](0, batch_size), src_buf=src.sigma2.create_sub_buffer[DType.float32](0, batch_size))


def batched_loglike_packed(
    ctx: DeviceContext,
    mut d_y: DeviceBuffer[DType.float32],
    batch_size: Int,
    n_obs: Int,
    order: ARIMAOrder,
    mut d_params: DeviceBuffer[DType.float32],
    trans: Bool,
    mut params: ARIMAParams,
    check_finite: Bool = True,
) raises -> LoglikeResult:
    """The `n_exog = 0` door."""
    _refuse_exog_order(order, "batched_loglike_packed")
    var e0 = _placeholder(ctx)
    var r = batched_loglike_packed_x(
        ctx, d_y, e0, batch_size, n_obs, order, d_params, trans, params, check_finite
    )
    _ = e0^
    return r^


def batched_loglike_packed_x(
    ctx: DeviceContext,
    mut d_y: DeviceBuffer[DType.float32],
    mut d_exog: DeviceBuffer[DType.float32],
    batch_size: Int,
    n_obs: Int,
    order: ARIMAOrder,
    mut d_params: DeviceBuffer[DType.float32],
    trans: Bool,
    mut params: ARIMAParams,
    check_finite: Bool = True,
) raises -> LoglikeResult:
    """`:471-513`: unpack the packed vector into `params`, then the overload
    above (`fc_steps = 0`, so the future exogenous input is a placeholder).
    `params` is the caller's scratch (their `arima_mem.params_*`)."""
    unpack(ctx, params, order, batch_size, d_params)
    var fut = _placeholder(ctx)
    var r = batched_loglike_x(
        ctx, d_y, d_exog, fut, batch_size, n_obs, order, params, trans, 0, 32, check_finite
    )
    _ = fut^
    return r^


# ---------------------------------------------------------------------------
# predict (:86-267)
# ---------------------------------------------------------------------------


def in_sample_prediction_kernel(
    d_y_p: MutPointer[Float32, MutAnyOrigin],
    d_y: MutPointer[Float32, MutAnyOrigin],
    d_pred: MutPointer[Float32, MutAnyOrigin],
    batch_size_in: Int32,
    n_obs_in: Int32,
    n_obs_kf_in: Int32,
    start_in: Int32,
    predict_ld_in: Int32,
    res_offset_in: Int32,
    p_start_in: Int32,
    p_end_in: Int32,
    dD_in: Int32,
    period1_in: Int32,
    period2_in: Int32,
):
    """`:206-228`, one thread per series (DEVIATION 676 for the sentinel);
    their `d_y_p[0] = 0.0` (`:207`) is the reference race above, not implemented."""
    var bid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if bid >= Int(batch_size_in):
        return
    var n_obs = Int(n_obs_in)
    var n_obs_kf = Int(n_obs_kf_in)
    var start = Int(start_in)
    var ld = Int(predict_ld_in)
    var res_offset = Int(res_offset_in)
    var dD = Int(dD_in)
    var p1 = Int(period1_in)
    var p2 = Int(period2_in)
    for i in range(res_offset - start):
        d_y_p.unsafe_store(bid * ld + i, bitcast[DType.float32](CANONICAL_NAN_BITS))
    for i in range(Int(p_start_in), Int(p_end_in)):
        var v: Float32
        if dD == 0:
            v = ftz(d_pred.unsafe_load(bid * n_obs + i))
        elif dD == 1:
            var a = ftz(d_y.unsafe_load(bid * n_obs + i - p1))
            var b = ftz(d_pred.unsafe_load(bid * n_obs_kf + i - res_offset))
            v = ftz(a + b)
        else:
            var a = ftz(d_y.unsafe_load(bid * n_obs + i - p1))
            var b = ftz(d_y.unsafe_load(bid * n_obs + i - p2))
            var c = ftz(d_y.unsafe_load(bid * n_obs + i - p1 - p2))
            var pr = ftz(d_pred.unsafe_load(bid * n_obs_kf + i - res_offset))
            var t0 = ftz(a + b)
            var t1 = ftz(t0 - c)
            v = ftz(t1 + pr)
        d_y_p.unsafe_store(bid * ld + i - start, v)


def copy_forecast_kernel(
    d_y_p: MutPointer[Float32, MutAnyOrigin],
    d_y_fc: MutPointer[Float32, MutAnyOrigin],
    batch_size_in: Int32,
    num_steps_in: Int32,
    predict_ld_in: Int32,
    n_obs_minus_start_in: Int32,
):
    """`:244-250`: copy the forecast into `d_y_p` after the in-sample part."""
    var bid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if bid >= Int(batch_size_in):
        return
    var num_steps = Int(num_steps_in)
    var ld = Int(predict_ld_in)
    var off = Int(n_obs_minus_start_in)
    for i in range(num_steps):
        d_y_p.unsafe_store(bid * ld + off + i, d_y_fc.unsafe_load(num_steps * bid + i))


@fieldwise_init
struct PredictResult(Movable):
    var y_p: DeviceBuffer[DType.float32]
    var predict_ld: Int
    var ll: LoglikeResult


def predict(
    ctx: DeviceContext,
    mut d_y: DeviceBuffer[DType.float32],
    batch_size: Int,
    n_obs: Int,
    start: Int,
    end: Int,
    order: ARIMAOrder,
    mut params: ARIMAParams,
    pre_diff: Bool,
    kalman_tpb: Int = 32,
) raises -> PredictResult:
    """The `n_exog = 0` door."""
    _refuse_exog_order(order, "predict")
    var e0 = _placeholder(ctx)
    var e1 = _placeholder(ctx)
    var r = predict_x(
        ctx, d_y, e0, e1, batch_size, n_obs, start, end, order, params, pre_diff, kalman_tpb
    )
    _ = e0^
    _ = e1^
    return r^


def predict_x(
    ctx: DeviceContext,
    mut d_y: DeviceBuffer[DType.float32],
    mut d_exog: DeviceBuffer[DType.float32],
    mut d_exog_fut: DeviceBuffer[DType.float32],
    batch_size: Int,
    n_obs: Int,
    start: Int,
    end: Int,
    order: ARIMAOrder,
    mut params: ARIMAParams,
    pre_diff: Bool,
    kalman_tpb: Int = 32,
) raises -> PredictResult:
    """`:86-267` with `level = 0`. `params` are the FITTED (already
    transformed) parameters, so `batched_loglike` is called with `trans =
    false` as theirs (`:175`). Returns `(end - start) * batch_size`
    predictions, series-major.

    `d_exog` is the regressors over the `n_obs` observations and `d_exog_fut`
    over the `max(end - n_obs, 0)` forecast steps, both in the filter's
    layout and UNDIFFERENCED; with `diff` they are differenced here exactly
    as `y` is (`:120-146`), otherwise handed to the filter as they are
    (`:154-156`)."""
    if start < 0 or end <= start:
        raise Error("predict: need 0 <= start < end (start=" + String(start) + ", end=" + String(end) + ")")
    validate_order(order)
    var diff = order.need_diff() and pre_diff
    var num_steps = end - n_obs if end > n_obs else 0
    var n_obs_kf: Int
    var order_after_prep = order
    var y_kf: DeviceBuffer[DType.float32]
    var n_ser = batch_size * order.n_exog
    var exog_kf: DeviceBuffer[DType.float32]
    var exog_fut_kf: DeviceBuffer[DType.float32]
    if diff:
        n_obs_kf = n_obs - order.n_diff()
        y_kf = ctx.enqueue_create_buffer[DType.float32](n_obs_kf * batch_size)
        prepare_data(ctx, y_kf, d_y, batch_size, n_obs, order.d, order.D, order.s)
        order_after_prep = order.without_diff()
    else:
        n_obs_kf = n_obs
        y_kf = ctx.enqueue_create_buffer[DType.float32](n_obs * batch_size)
        ctx.enqueue_copy(dst_buf=y_kf, src_buf=d_y.create_sub_buffer[DType.float32](0, n_obs * batch_size))
    if n_ser == 0:
        exog_kf = _placeholder(ctx)
        exog_fut_kf = _placeholder(ctx)
    elif diff:
        # `:121-145`: the regressors' past through prepare_data, their
        # future through prepare_future_data against that past.
        exog_kf = ctx.enqueue_create_buffer[DType.float32](n_obs_kf * n_ser)
        prepare_data(ctx, exog_kf, d_exog, n_ser, n_obs, order.d, order.D, order.s)
        exog_fut_kf = ctx.enqueue_create_buffer[DType.float32](max(1, num_steps * n_ser))
        if num_steps > 0:
            prepare_future_data(
                ctx, exog_fut_kf, d_exog, d_exog_fut, n_ser, n_obs, num_steps,
                order.d, order.D, order.s,
            )
    else:
        exog_kf = ctx.enqueue_create_buffer[DType.float32](n_obs * n_ser)
        ctx.enqueue_copy(dst_buf=exog_kf, src_buf=d_exog.create_sub_buffer[DType.float32](0, n_obs * n_ser))
        exog_fut_kf = ctx.enqueue_create_buffer[DType.float32](max(1, num_steps * n_ser))
        if num_steps > 0:
            ctx.enqueue_copy(
                dst_buf=exog_fut_kf,
                src_buf=d_exog_fut.create_sub_buffer[DType.float32](0, num_steps * n_ser),
            )
    var ll = batched_loglike_x(
        ctx, y_kf, exog_kf, exog_fut_kf, batch_size, n_obs_kf, order_after_prep, params,
        False, num_steps, kalman_tpb,
    )
    var predict_ld = end - start
    var y_p = ctx.enqueue_create_buffer[DType.float32](predict_ld * batch_size)
    comptime TPB = 128
    var grid = (batch_size + TPB - 1) // TPB
    if start < n_obs:
        var res_offset = order.d + order.s * order.D if diff else 0
        var p_start = start if start > res_offset else res_offset
        var p_end = n_obs if n_obs < end else end
        var dD = order.d + order.D if diff else 0
        var period1 = 1 if order.d != 0 else order.s
        var period2 = 1 if order.d == 2 else order.s
        ctx.enqueue_function[in_sample_prediction_kernel](
            y_p.unsafe_ptr(), d_y.unsafe_ptr(), ll.ws.pred.unsafe_ptr(),
            Int32(batch_size), Int32(n_obs), Int32(n_obs_kf), Int32(start), Int32(predict_ld),
            Int32(res_offset), Int32(p_start), Int32(p_end), Int32(dD), Int32(period1), Int32(period2),
            grid_dim=(grid, 1, 1), block_dim=(TPB, 1, 1),
        )
    if num_steps > 0:
        if diff:
            finalize_forecast(ctx, ll.ws.fc, d_y, num_steps, batch_size, n_obs, n_obs, order.d, order.D, order.s)
        ctx.enqueue_function[copy_forecast_kernel](
            y_p.unsafe_ptr(), ll.ws.fc.unsafe_ptr(), Int32(batch_size), Int32(num_steps),
            Int32(predict_ld), Int32(n_obs - start),
            grid_dim=(grid, 1, 1), block_dim=(TPB, 1, 1),
        )
    ctx.synchronize()
    _ = exog_kf^
    _ = exog_fut_kf^
    _ = y_kf^
    return PredictResult(y_p=y_p^, predict_ld=predict_ld, ll=ll^)


# ---------------------------------------------------------------------------
# batched_loglike_grad (:515-590): forward finite differences, one
# parameter at a time, every series at once
# ---------------------------------------------------------------------------


def perturb_kernel(
    d_x_pert: MutPointer[Float32, MutAnyOrigin],
    d_x: MutPointer[Float32, MutAnyOrigin],
    batch_size_in: Int32,
    N_in: Int32,
    i_in: Int32,
    h: Float32,
):
    """`:558-561`: `x_pert[N*bid + i] = x[N*bid + i] + h` (the statement
    itself is `:560`)."""
    var bid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if bid >= Int(batch_size_in):
        return
    var idx = Int(N_in) * bid + Int(i_in)
    d_x_pert.unsafe_store(idx, ftz(ftz(d_x.unsafe_load(idx)) + h))


def reset_param_kernel(
    d_x_pert: MutPointer[Float32, MutAnyOrigin],
    d_x: MutPointer[Float32, MutAnyOrigin],
    batch_size_in: Int32,
    N_in: Int32,
    i_in: Int32,
):
    """`:585-588`: `x_pert[N*bid + i] = x[N*bid + i]` (the statement
    itself is `:587`), a COPY.

    An earlier revision reused `perturb_kernel` with `h = 0` for this. That
    is NOT their statement: `x + 0.0` maps `-0.0` to `+0.0`, so a parameter
    that is negative zero came back positive zero after its own reset and
    every LATER parameter's log-likelihood was then evaluated on a vector
    one bit away from `d_x`. Their reset is an assignment; so is this."""
    var bid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if bid >= Int(batch_size_in):
        return
    var idx = Int(N_in) * bid + Int(i_in)
    d_x_pert.unsafe_store(idx, d_x.unsafe_load(idx))


def grad_kernel(
    d_grad: MutPointer[Float32, MutAnyOrigin],
    d_ll_pert: MutPointer[Float32, MutAnyOrigin],
    d_ll_base: MutPointer[Float32, MutAnyOrigin],
    batch_size_in: Int32,
    N_in: Int32,
    i_in: Int32,
    h: Float32,
):
    """`:576-579`: `grad[N*bid + i] = (ll_pert - ll_base) / h`."""
    var bid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if bid >= Int(batch_size_in):
        return
    var diff = ftz(ftz(d_ll_pert.unsafe_load(bid)) - ftz(d_ll_base.unsafe_load(bid)))
    d_grad.unsafe_store(Int(N_in) * bid + Int(i_in), ftz(diff / h))


def batched_loglike_grad(
    ctx: DeviceContext,
    mut d_y: DeviceBuffer[DType.float32],
    batch_size: Int,
    n_obs: Int,
    order: ARIMAOrder,
    mut d_x: DeviceBuffer[DType.float32],
    mut d_grad: DeviceBuffer[DType.float32],
    h: Float32,
    trans: Bool,
    mut params: ARIMAParams,
    mut d_x_pert: DeviceBuffer[DType.float32],
    check_finite: Bool = True,
) raises -> List[Float32]:
    """The `n_exog = 0` door."""
    _refuse_exog_order(order, "batched_loglike_grad")
    var e0 = _placeholder(ctx)
    var r = batched_loglike_grad_x(
        ctx, d_y, e0, batch_size, n_obs, order, d_x, d_grad, h, trans, params,
        d_x_pert, check_finite,
    )
    _ = e0^
    return r^


#: FAST on Apple: the forward-difference gradient's N + 1 log-likelihoods
#: (base, then one per perturbed parameter) are ONE batched evaluation over
#: (N + 1) x batch_size members -- the series replicated, member m's
#: parameters perturbed in parameter m - 1 -- instead of N + 1 sequential
#: filter passes. Members are independent threads of the same kernel, so
#: every log-likelihood, and so the gradient, is the one the sequential form
#: computes. `-D MOJOLEARN_ARIMA_FAST_BATCH_GRAD_OFF` keeps the sequence.
comptime ARIMA_FAST_BATCH_GRAD = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_ARIMA_FAST_BATCH_GRAD_OFF"]()
)


def _batched_loglike_grad_stacked(
    ctx: DeviceContext,
    mut d_y: DeviceBuffer[DType.float32],
    mut d_exog: DeviceBuffer[DType.float32],
    batch_size: Int,
    n_obs: Int,
    order: ARIMAOrder,
    mut d_x: DeviceBuffer[DType.float32],
    mut d_grad: DeviceBuffer[DType.float32],
    h: Float32,
    trans: Bool,
    mut d_x_pert: DeviceBuffer[DType.float32],
    check_finite: Bool,
) raises -> List[Float32]:
    var N = order.complexity()
    var M1 = N + 1
    var eb = M1 * batch_size
    var nb_y = batch_size * n_obs
    var nb_x = batch_size * N
    var y_ext = ctx.enqueue_create_buffer[DType.float32](eb * n_obs)
    var x_ext = ctx.enqueue_create_buffer[DType.float32](eb * N)
    for m in range(M1):
        ctx.enqueue_copy(
            dst_buf=y_ext.create_sub_buffer[DType.float32](m * nb_y, nb_y),
            src_buf=d_y.create_sub_buffer[DType.float32](0, nb_y),
        )
        ctx.enqueue_copy(
            dst_buf=x_ext.create_sub_buffer[DType.float32](m * nb_x, nb_x),
            src_buf=d_x.create_sub_buffer[DType.float32](0, nb_x),
        )
    comptime TPB = 128
    var grid = (batch_size + TPB - 1) // TPB
    for i in range(N):
        var blk = x_ext.unsafe_ptr().unsafe_offset((i + 1) * nb_x)
        var blk_src = MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(blk))
        ctx.enqueue_function[perturb_kernel](
            blk, blk_src, Int32(batch_size), Int32(N), Int32(i), h,
            grid_dim=(grid, 1, 1), block_dim=(TPB, 1, 1),
        )
    var p_ext = ARIMAParams(ctx, order, eb)
    var r = batched_loglike_packed_x(
        ctx, y_ext, d_exog, eb, n_obs, order, x_ext, trans, p_ext, check_finite
    )
    for i in range(N):
        ctx.enqueue_function[grad_kernel](
            d_grad.unsafe_ptr(),
            r.ws.loglike.unsafe_ptr().unsafe_offset((i + 1) * batch_size),
            MutPointer[Float32, MutAnyOrigin](
                unsafe_from_address=Int(r.ws.loglike.unsafe_ptr())
            ),
            Int32(batch_size), Int32(N), Int32(i), h,
            grid_dim=(grid, 1, 1), block_dim=(TPB, 1, 1),
        )
    # the caller's scratch ends equal to d_x, as the sequential form leaves it
    ctx.enqueue_copy(
        dst_buf=d_x_pert.create_sub_buffer[DType.float32](0, nb_x),
        src_buf=d_x.create_sub_buffer[DType.float32](0, nb_x),
    )
    ctx.synchronize()
    var ll = List[Float32](capacity=batch_size)
    for b in range(batch_size):
        ll.append(r.loglike[b])
    _ = y_ext^
    _ = x_ext^
    _ = p_ext^
    _ = r^
    return ll^


def batched_loglike_grad_x(
    ctx: DeviceContext,
    mut d_y: DeviceBuffer[DType.float32],
    mut d_exog: DeviceBuffer[DType.float32],
    batch_size: Int,
    n_obs: Int,
    order: ARIMAOrder,
    mut d_x: DeviceBuffer[DType.float32],
    mut d_grad: DeviceBuffer[DType.float32],
    h: Float32,
    trans: Bool,
    mut params: ARIMAParams,
    mut d_x_pert: DeviceBuffer[DType.float32],
    check_finite: Bool = True,
) raises -> List[Float32]:
    """`:515-591`. Returns the base log-likelihood (host) beside the device
    gradient, because the L-BFGS caller needs both and theirs evaluates the
    base inside this call.

    `d_x_pert` is the CALLER'S scratch, which is what theirs is too
    (`arima_mem.x_pert`, `:534`). It was a local here until 2026-08-23.
    Making it the caller's is what lets a gate READ IT BACK after the call
    and assert it returned to `d_x` bitwise, and that assertion is the ONLY
    thing that catches the reset regressing: the `-0.0` a broken reset
    destroys does not survive far enough through the filter to move the
    log-likelihood, so the indirect gate written first was INERT and
    sabotage (g) moved nothing against it. See
    `check_grad_reset_preserves_negative_zero`."""
    var N = order.complexity()
    comptime if ARIMA_FAST_BATCH_GRAD:
        if order.n_exog == 0:
            return _batched_loglike_grad_stacked(
                ctx, d_y, d_exog, batch_size, n_obs, order, d_x, d_grad, h,
                trans, d_x_pert, check_finite,
            )
    ctx.enqueue_copy(dst_buf=d_x_pert.create_sub_buffer[DType.float32](0, N * batch_size), src_buf=d_x.create_sub_buffer[DType.float32](0, N * batch_size))
    var base = batched_loglike_packed_x(
        ctx, d_y, d_exog, batch_size, n_obs, order, d_x, trans, params, check_finite
    )
    comptime TPB = 128
    var grid = (batch_size + TPB - 1) // TPB
    for i in range(N):
        ctx.enqueue_function[perturb_kernel](
            d_x_pert.unsafe_ptr(), d_x.unsafe_ptr(), Int32(batch_size), Int32(N), Int32(i), h,
            grid_dim=(grid, 1, 1), block_dim=(TPB, 1, 1),
        )
        var pert = batched_loglike_packed_x(
            ctx, d_y, d_exog, batch_size, n_obs, order, d_x_pert, trans, params, check_finite
        )
        ctx.enqueue_function[grad_kernel](
            d_grad.unsafe_ptr(), pert.ws.loglike.unsafe_ptr(), base.ws.loglike.unsafe_ptr(),
            Int32(batch_size), Int32(N), Int32(i), h,
            grid_dim=(grid, 1, 1), block_dim=(TPB, 1, 1),
        )
        ctx.enqueue_function[reset_param_kernel](
            d_x_pert.unsafe_ptr(), d_x.unsafe_ptr(), Int32(batch_size), Int32(N), Int32(i),
            grid_dim=(grid, 1, 1), block_dim=(TPB, 1, 1),
        )
        ctx.synchronize()
        _ = pert^
    ctx.synchronize()
    var ll = base.loglike.copy()
    _ = base^
    return ll^
