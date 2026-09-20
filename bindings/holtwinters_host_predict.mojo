# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Holt-Winters prediction entries on the host, shared by three bindings
(lane/inference-holtwinters, 2026-09-15).

`bindings/_mojolearn_tsa_host.mojo` (the internal reference binding, fit
included) and `bindings/_mojolearn_forecast_host.mojo` (the inference binding
the wheels ship, no fit) both register `holtwinters_forecast` and
`holtwinters_predict` from here, so the two binaries answer a saved model
through the same source. The GPU binding `bindings/_mojolearn_tsa.mojo`
registers `holtwinters_predict` from here too: the in-sample prediction is
host arithmetic over the fitted components on every install (its forecast
stays the device path). Not a binding itself: it registers nothing, and the
host surface tests glob only `_mojolearn_*_host.mojo`.

The arithmetic is `holtwinters/host/hw_predict.mojo`, which imports only
`checks/numerics.mojo`: no decomposition, no BFGS, no line search.

THE SABOTAGE ARM, `HW_PREDICT_SABOTAGE`: under `-D MOJOLEARN_HOST_SABOTAGE=1`
(the host builds' negative control) or `-D MOJOLEARN_HW_PREDICT_SABOTAGE=1`,
every finite value either entry writes has its lowest bit flipped, a VALUE
perturbation that leaves every order and shape alone, so a saved-model gate
must read a mismatch on every forecast and in-sample prediction. The GPU
build scripts pass neither define.
"""
from std.math import isfinite
from std.memory import bitcast
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.sys.compile import is_defined

from bindings.hostptr import f32_ptr
from holtwinters.host.hw_predict import (
    hw_forecast_from_state_ptr,
    hw_predict_in_sample_ptr,
)
from holtwinters.impl.tsa.holtwinters_params import (
    SEASONAL_ADDITIVE,
    seasonal_from_name,
)

comptime HW_PREDICT_SABOTAGE = (
    is_defined["MOJOLEARN_HOST_SABOTAGE"]()
    or is_defined["MOJOLEARN_HW_PREDICT_SABOTAGE"]()
)


def _hw_index(value: PythonObject) raises -> Int:
    var type_name = String(py=value.__class__.__name__)
    if type_name == "bool" or type_name == "bool_":
        raise Error("holtwinters host: integers expected, not booleans")
    var operator_module = Python.import_module("operator")
    return Int(py=operator_module.index(value))


@always_inline
def _store(op: MutPointer[Float32, MutUntrackedOrigin], i: Int, v: Float32):
    comptime if HW_PREDICT_SABOTAGE:
        if isfinite(v):
            op.unsafe_store(
                i, bitcast[DType.float32](bitcast[DType.uint32](v) ^ UInt32(1))
            )
            return
    op.unsafe_store(i, v)


def _refuse_state(n: Int, batch_size: Int, frequency: Int, who: String) raises:
    """`holtwinters_forecast_ptr`'s guards (`holtwinters/estimator.mojo`),
    in its words and order, after `seasonal_from_name`."""
    if n <= frequency:
        raise Error(
            "holtwinters "
            + who
            + ": n ("
            + String(n)
            + ") must exceed frequency ("
            + String(frequency)
            + "); there would be no fitted components"
        )
    if batch_size < 1:
        raise Error(
            "holtwinters "
            + who
            + ": batch_size must be >= 1 (batch_size="
            + String(batch_size)
            + ")"
        )


def holtwinters_forecast_binding(
    comps_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
    seasonal: PythonObject,
) raises -> PythonObject:
    """`.forecast(h)` on the host by `hw_forecast_from_state`. Returns
    `h * batch_size`.

    `params` is, in this exact order (the same order in
    `python/mojolearn/_tsa_impl.py` and in the GPU binding):

        0  n              the FIT's observations per series
        1  batch_size
        2  frequency      the FIT's seasonal_periods
        3  h              steps to forecast

    `comps_addr` reads `3 * components_len` float32 in the packed order
    `holtwinters_fit` wrote (level, trend, season, each `components_len =
    (n - frequency) * batch_size` and TIME-MAJOR); `out_addr` is written with
    `h * batch_size` float32, TIME-MAJOR."""
    if len(params) != 4:
        raise Error(
            "holtwinters_forecast: params must contain 4 values (n,"
            " batch_size, frequency, h), got "
            + String(len(params))
        )
    var cp = f32_ptr(_hw_index(comps_addr))
    var op = f32_ptr(_hw_index(out_addr))
    var n = _hw_index(params[0])
    var batch_size = _hw_index(params[1])
    var frequency = _hw_index(params[2])
    var h = _hw_index(params[3])
    var sname = String(py=seasonal)
    var written = 0
    with GILReleased(Python()):
        var st = seasonal_from_name(sname)
        _refuse_state(n, batch_size, frequency, "forecast")
        if h <= 0:
            raise Error("h must be > 0. Currently: " + String(h))
        var components_len = (n - frequency) * batch_size
        var fc = hw_forecast_from_state_ptr(
            cp,
            components_len,
            n,
            batch_size,
            frequency,
            st == SEASONAL_ADDITIVE,
            h,
        )
        for i in range(h * batch_size):
            _store(op, i, fc[i])
        written = h * batch_size
    return PythonObject(written)


def holtwinters_predict_binding(
    comps_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
    seasonal: PythonObject,
) raises -> PythonObject:
    """The in-sample one-step predictions at times `[start, end)` by
    `hw_predict_in_sample`. Returns `(end - start) * batch_size`.

    `params` is, in this exact order (the same order in
    `python/mojolearn/_tsa_impl.py`):

        0  n              the FIT's observations per series
        1  batch_size
        2  frequency      the FIT's seasonal_periods
        3  start          0 <= start
        4  end            start < end <= n

    `comps_addr` reads the packed components as `holtwinters_forecast`
    does; `out_addr` is written with `(end - start) * batch_size` float32,
    TIME-MAJOR, the canonical quiet NaN where `t < 2 * frequency`. A time
    at or beyond `n` is the forecast's, and `ExponentialSmoothing.predict`
    asks `holtwinters_forecast` for it."""
    if len(params) != 5:
        raise Error(
            "holtwinters_predict: params must contain 5 values (n,"
            " batch_size, frequency, start, end), got "
            + String(len(params))
        )
    var cp = f32_ptr(_hw_index(comps_addr))
    var op = f32_ptr(_hw_index(out_addr))
    var n = _hw_index(params[0])
    var batch_size = _hw_index(params[1])
    var frequency = _hw_index(params[2])
    var start = _hw_index(params[3])
    var end = _hw_index(params[4])
    var sname = String(py=seasonal)
    var written = 0
    with GILReleased(Python()):
        var st = seasonal_from_name(sname)
        _refuse_state(n, batch_size, frequency, "predict")
        if start < 0 or end <= start:
            raise Error(
                "holtwinters predict: need 0 <= start < end (start="
                + String(start)
                + ", end="
                + String(end)
                + ")"
            )
        if end > n:
            raise Error(
                "holtwinters predict: the in-sample prediction ends at n (end="
                + String(end)
                + ", n="
                + String(n)
                + "); later times are the forecast's"
            )
        var components_len = (n - frequency) * batch_size
        var out = hw_predict_in_sample_ptr(
            cp,
            components_len,
            n,
            batch_size,
            frequency,
            st == SEASONAL_ADDITIVE,
            start,
            end,
        )
        written = (end - start) * batch_size
        for i in range(written):
            _store(op, i, out[i])
    return PythonObject(written)
