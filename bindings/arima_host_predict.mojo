# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The ARIMA prediction entries on the host, shared by two bindings
(lane/inference-forecast-umap-pca, 2026-09-15).

`bindings/_mojolearn_arima_host.mojo` (the internal reference binding, fit
included) and `bindings/_mojolearn_forecast_host.mojo` (the inference binding
the wheels ship, no fit) both register `arima_predict` and `arima_forecast`
from here, so the two binaries answer a saved model through the same source.
Not a binding itself: it registers nothing, and the host surface tests glob
only `_mojolearn_*_host.mojo`.

The contract is `bindings/_mojolearn_arima.mojo`'s: the addresses are (y, exog, exog_fut,
params, out), `arima_predict`'s `params` is (batch_size, n_obs, start, end,
p, d, q, P, D, Q, s, k, n_exog) and `arima_forecast`'s (batch_size, n_obs,
n_steps, p, d, q, P, D, Q, s, k, n_exog, reserved). The validation is `arima/estimator.mojo`'s, in its order:
`_order` through the device's own `validate_order`, `_refuse_shape`,
`_predict_into`'s start and end checks, then
`arima_host_refuse_unrestated`. Since 2026-09-15 an in-sample prediction
(`start < n_obs`) runs through `arima_host_predict` instead of refusing.
"""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from bindings.arima_exog_layout import exog_filter_layout
from bindings.hostptr import f32_ptr, read_f32
from arima.host.arima_oracle import (
    ArimaHostOrder,
    arima_host_forecast,
    arima_host_predict,
    arima_host_refuse_unrestated,
)
from arima.impl.tsa.arima_common import ARIMAOrder, validate_order


def _order(
    p: Int, d: Int, q: Int, P: Int, D: Int, Q: Int, s: Int, k: Int, n_exog: Int
) raises -> ArimaHostOrder:
    """`arima/estimator.mojo::_order` (`:150-162`): the device's
    `validate_order` on the nine integers, then the host order."""
    var order = ARIMAOrder(p, d, q, P, D, Q, s, k, n_exog)
    validate_order(order)
    return ArimaHostOrder(p, d, q, P, D, Q, s, k, n_exog)


def _refuse_shape(batch_size: Int, n_obs: Int, who: String) raises:
    """`arima/estimator.mojo::_refuse_shape` (`:165-175`), its words."""
    if batch_size < 1:
        raise Error(
            who + ": batch_size must be >= 1 (batch_size=" + String(batch_size) + ")"
        )
    if n_obs < 2:
        raise Error(
            who + ": n_obs must be at least 2 (n_obs=" + String(n_obs) + ")"
        )


def _forecast_into(
    y_address: Int,
    exog_address: Int,
    exog_fut_address: Int,
    params_address: Int,
    op: MutPointer[Float32, MutUntrackedOrigin],
    batch_size: Int,
    n_obs: Int,
    n_steps: Int,
    order: ArimaHostOrder,
) raises -> Int:
    """The body both prediction entries share once `start == n_obs`, as
    `_predict_into` is shared on the device."""
    var N = order.complexity()
    var y = read_f32(y_address, batch_size * n_obs)
    var pr = read_f32(params_address, N * batch_size)
    var exog = exog_filter_layout(exog_address, batch_size, n_obs, order.n_exog, "exog")
    var fut = exog_filter_layout(
        exog_fut_address, batch_size, n_steps, order.n_exog, "exog (future values)"
    )
    var fc = arima_host_forecast(y, exog, fut, pr, batch_size, n_obs, n_steps, order)
    if len(fc) != n_steps * batch_size:
        raise Error(
            "arima_forecast: the host oracle returned a forecast of an"
            " unexpected length; nothing written"
        )
    for i in range(n_steps * batch_size):
        op.unsafe_store(i, fc[i])
    return n_steps * batch_size


def _predict_in_sample_into(
    y_address: Int,
    exog_address: Int,
    exog_fut_address: Int,
    params_address: Int,
    op: MutPointer[Float32, MutUntrackedOrigin],
    batch_size: Int,
    n_obs: Int,
    start: Int,
    end: Int,
    order: ArimaHostOrder,
) raises -> Int:
    """`start < n_obs`: `arima_host_predict`, the in-sample block and any
    forecast after it, written series major."""
    var N = order.complexity()
    var y = read_f32(y_address, batch_size * n_obs)
    var pr = read_f32(params_address, N * batch_size)
    var ld = end - start
    var num_steps = end - n_obs if end > n_obs else 0
    var exog = exog_filter_layout(exog_address, batch_size, n_obs, order.n_exog, "exog")
    var fut = exog_filter_layout(
        exog_fut_address, batch_size, num_steps, order.n_exog, "exog (future values)"
    )
    var out = arima_host_predict(y, exog, fut, pr, batch_size, n_obs, start, end, order)
    if len(out) != ld * batch_size:
        raise Error(
            "arima_predict: the host oracle returned a prediction of an"
            " unexpected length; nothing written"
        )
    for i in range(ld * batch_size):
        op.unsafe_store(i, out[i])
    return ld * batch_size


def arima_predict_binding(
    y_addr: PythonObject,
    exog_addr: PythonObject,
    exog_fut_addr: PythonObject,
    params_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`bindings/_mojolearn_arima.mojo::arima_predict_binding`'s contract on
    the host: `params` is (batch_size, n_obs, start, end, p, d, q, P, D, Q,
    s, k, n_exog); returns `(end - start) * batch_size`. `start == n_obs` is
    the forecast body; `start < n_obs` is `arima_host_predict`."""
    if len(params) != 13:
        raise Error(
            "arima_predict: params must contain 13 values (batch_size,"
            " n_obs, start, end, p, d, q, P, D, Q, s, k, n_exog), got "
            + String(len(params))
        )
    var y_address = Int(py=y_addr)
    _ = f32_ptr(y_address)
    var exog_address = Int(py=exog_addr)
    _ = f32_ptr(exog_address)
    var exog_fut_address = Int(py=exog_fut_addr)
    _ = f32_ptr(exog_fut_address)
    var params_address = Int(py=params_addr)
    _ = f32_ptr(params_address)
    var op = f32_ptr(Int(py=out_addr))
    var batch_size = Int(py=params[0])
    var n_obs = Int(py=params[1])
    var start = Int(py=params[2])
    var end = Int(py=params[3])
    var p = Int(py=params[4])
    var d = Int(py=params[5])
    var q = Int(py=params[6])
    var P = Int(py=params[7])
    var D = Int(py=params[8])
    var Q = Int(py=params[9])
    var s = Int(py=params[10])
    var k = Int(py=params[11])
    var n_exog = Int(py=params[12])
    var written = 0
    with GILReleased(Python()):
        # `arima_predict_ptr_host` then `_predict_into`
        # (`arima/estimator.mojo:502-506`, `:422-433`), in their order.
        var order = _order(p, d, q, P, D, Q, s, k, n_exog)
        _refuse_shape(batch_size, n_obs, "arima_predict")
        if start < 0 or end <= start:
            raise Error(
                "arima_predict: need 0 <= start < end (start=" + String(start)
                + ", end=" + String(end) + ")"
            )
        if start > n_obs:
            raise Error(
                "arima_predict: there can't be a gap between the data and the"
                " prediction (start=" + String(start) + ", n_obs="
                + String(n_obs) + ")"
            )
        arima_host_refuse_unrestated(order, "arima_predict")
        if start < n_obs:
            written = _predict_in_sample_into(
                y_address, exog_address, exog_fut_address, params_address, op,
                batch_size, n_obs, start, end, order,
            )
        else:
            written = _forecast_into(
                y_address, exog_address, exog_fut_address, params_address, op,
                batch_size, n_obs, end - n_obs, order,
            )
    return PythonObject(written)


def arima_forecast_binding(
    y_addr: PythonObject,
    exog_addr: PythonObject,
    exog_fut_addr: PythonObject,
    params_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`bindings/_mojolearn_arima.mojo::arima_forecast_binding`'s contract on
    the host: `params` is (batch_size, n_obs, n_steps, p, d, q, P, D, Q, s,
    k, n_exog, reserved); returns `n_steps * batch_size`."""
    if len(params) != 13:
        raise Error(
            "arima_forecast: params must contain 13 values (batch_size,"
            " n_obs, n_steps, p, d, q, P, D, Q, s, k, n_exog, reserved),"
            " got " + String(len(params))
        )
    var y_address = Int(py=y_addr)
    _ = f32_ptr(y_address)
    var exog_address = Int(py=exog_addr)
    _ = f32_ptr(exog_address)
    var exog_fut_address = Int(py=exog_fut_addr)
    _ = f32_ptr(exog_fut_address)
    var params_address = Int(py=params_addr)
    _ = f32_ptr(params_address)
    var op = f32_ptr(Int(py=out_addr))
    var batch_size = Int(py=params[0])
    var n_obs = Int(py=params[1])
    var n_steps = Int(py=params[2])
    var p = Int(py=params[3])
    var d = Int(py=params[4])
    var q = Int(py=params[5])
    var P = Int(py=params[6])
    var D = Int(py=params[7])
    var Q = Int(py=params[8])
    var s = Int(py=params[9])
    var k = Int(py=params[10])
    var n_exog = Int(py=params[11])
    var reserved = Int(py=params[12])
    if reserved != 0:
        raise Error(
            "arima_forecast: params[12] is reserved and must be 0, got "
            + String(reserved)
            + ". cuML's `level` (confidence intervals) is the parameter this"
            " slot is held for and it is NOT IMPLEMENTED"
            " (arima/NOT_IMPLEMENTED.tsv): the confidence_intervals kernel"
            " at batched_kalman.cu:824-838 and the P = T P T' + RR'"
            " propagation beside it have no implementation"
        )
    var written = 0
    with GILReleased(Python()):
        # `arima_forecast_ptr_host` then `_predict_into`
        # (`arima/estimator.mojo:536-545`, `:422-433`), in their order.
        if n_steps < 1:
            raise Error(
                "arima_forecast: n_steps must be >= 1 (n_steps="
                + String(n_steps) + ")"
            )
        var order = _order(p, d, q, P, D, Q, s, k, n_exog)
        _refuse_shape(batch_size, n_obs, "arima_forecast")
        arima_host_refuse_unrestated(order, "arima_forecast")
        written = _forecast_into(
            y_address, exog_address, exog_fut_address, params_address, op,
            batch_size, n_obs, n_steps, order,
        )
    return PythonObject(written)
