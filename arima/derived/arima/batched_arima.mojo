# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's DERIVATION_MAP.tsv and in this file's own docstring. See NOTICE.
"""`batched_loglike`, `predict`, `batched_loglike_grad`, `batched_diff`: the
C++ entry points the Python `ARIMA` class calls.

PORT OF `cuml/cpp/src/arima/batched_arima.cu` at cuML 265b9da6 (v26.08.00):
`batched_diff` (:60-70), `predict` (:86-267), `batched_loglike` (the
`ARIMAParams` overload :393-469 and the packed-vector overload :471-513),
`batched_loglike_grad` (:515-591). COPY, DO NOT IMPROVE.

NOT PORTED, refused by name: `method == CSS` (`conditional_sum_of_squares`,
`sum_of_squares_kernel` :270-391; only `MLE` is offered, `truncate` must
be 0), `information_criterion` (:592-625), `estimate_x0` / `_start_params`
/ `_arma_least_squares` (:627-1010: cuBLAS `b_gels`, `b_lagged_mat`),
`detect_missing` (NaN is refused, not detected), exogenous regressors,
`level > 0` (confidence intervals). See `arima/NOT_IMPLEMENTED.tsv`.

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
upstream defect, and the statement is not ported: `assume-our-code-is-
broken`'s rule is to fix their bug rather than mirror it, and the fix here
is to not write the cell at all.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.math import isfinite
from std.memory import bitcast

from arima.derived.arima.batched_kalman import KalmanWorkspace, batched_kalman_filter
from arima.derived.timeSeries.arima_helpers import batched_jones_transform, finalize_forecast
from arima.derived.tsa.arima_common import ARIMAOrder, ARIMAParams, unpack, validate_order
from original.numerics import ftz
from tsa.derived.timeSeries.arima_helpers import prepare_data


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
) raises -> LoglikeResult:
    """`:393-469`, the `MLE` arm with `host_loglike = true`: the Jones
    transform when `trans`, the Kalman filter, the host copy of the
    log-likelihood. `method = CSS` is refused by name by the caller."""
    _refuse_non_finite(ctx, d_y, batch_size * n_obs, "y")
    validate_order(order)
    var t_params = ARIMAParams(ctx, order, batch_size)
    if trans:
        batched_jones_transform(ctx, order, batch_size, False, params, t_params)
    else:
        # non-transformed case: just use original parameters (:447-452)
        _copy_params(ctx, params, t_params, order, batch_size)
    var ws = batched_kalman_filter(ctx, d_y, n_obs, t_params, order, batch_size, fc_steps, kalman_tpb)
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
    there; missing observations are not ported, so a non-finite input is
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
                + "; missing observations are not ported and are refused by name (arima/NOT_IMPLEMENTED.tsv)"
            )
    _ = h^


def _copy_params(ctx: DeviceContext, mut src: ARIMAParams, mut dst: ARIMAParams, order: ARIMAOrder, batch_size: Int) raises:
    if order.k != 0:
        ctx.enqueue_copy(dst_buf=dst.mu.create_sub_buffer[DType.float32](0, batch_size), src_buf=src.mu.create_sub_buffer[DType.float32](0, batch_size))
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
) raises -> LoglikeResult:
    """`:471-513`: unpack the packed vector into `params`, then the overload
    above (`fc_steps = 0`). `params` is the caller's scratch (their
    `arima_mem.params_*`)."""
    unpack(ctx, params, order, batch_size, d_params)
    return batched_loglike(ctx, d_y, batch_size, n_obs, order, params, trans, 0)


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
    their `d_y_p[0] = 0.0` (`:207`) is the upstream race above, not ported."""
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
    """`:86-267` with `level = 0` and no exog. `params` are the FITTED
    (already transformed) parameters, so `batched_loglike` is called with
    `trans = false` as theirs (`:175`). Returns `(end - start) * batch_size`
    predictions, series-major."""
    if start < 0 or end <= start:
        raise Error("predict: need 0 <= start < end (start=" + String(start) + ", end=" + String(end) + ")")
    validate_order(order)
    var diff = order.need_diff() and pre_diff
    var num_steps = end - n_obs if end > n_obs else 0
    var n_obs_kf: Int
    var order_after_prep = order
    var y_kf: DeviceBuffer[DType.float32]
    if diff:
        n_obs_kf = n_obs - order.n_diff()
        y_kf = ctx.enqueue_create_buffer[DType.float32](n_obs_kf * batch_size)
        prepare_data(ctx, y_kf, d_y, batch_size, n_obs, order.d, order.D, order.s)
        order_after_prep = order.without_diff()
    else:
        n_obs_kf = n_obs
        y_kf = ctx.enqueue_create_buffer[DType.float32](n_obs * batch_size)
        ctx.enqueue_copy(dst_buf=y_kf, src_buf=d_y.create_sub_buffer[DType.float32](0, n_obs * batch_size))
    var ll = batched_loglike(ctx, y_kf, batch_size, n_obs_kf, order_after_prep, params, False, num_steps, kalman_tpb)
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
    ctx.enqueue_copy(dst_buf=d_x_pert.create_sub_buffer[DType.float32](0, N * batch_size), src_buf=d_x.create_sub_buffer[DType.float32](0, N * batch_size))
    var base = batched_loglike_packed(ctx, d_y, batch_size, n_obs, order, d_x, trans, params)
    comptime TPB = 128
    var grid = (batch_size + TPB - 1) // TPB
    for i in range(N):
        ctx.enqueue_function[perturb_kernel](
            d_x_pert.unsafe_ptr(), d_x.unsafe_ptr(), Int32(batch_size), Int32(N), Int32(i), h,
            grid_dim=(grid, 1, 1), block_dim=(TPB, 1, 1),
        )
        var pert = batched_loglike_packed(ctx, d_y, batch_size, n_obs, order, d_x_pert, trans, params)
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
