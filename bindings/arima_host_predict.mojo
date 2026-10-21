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

from bindings.hostptr import f32_ptr
from arima.host.arima_oracle import arima_host_refuse_unrestated
# The typed bodies (lane thin-trees-metrics, 2026-10-21): moved word for
# word to arima/host/arima_host_entries.mojo; this file is the Python door.
from arima.host.arima_host_entries import (
    _forecast_into,
    _order,
    _predict_in_sample_into,
    _refuse_shape,
)


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
