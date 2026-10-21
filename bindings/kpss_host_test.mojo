# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The KPSS stationarity test on the host, shared by two bindings
(lane/expose-inference-surface, 2026-09-16).

`bindings/_mojolearn_tsa_host.mojo` (the internal reference binding, the
Holt-Winters fit included) and `bindings/_mojolearn_forecast_host.mojo` (the
inference binding the wheels ship, no fit) both register `kpss_test` from
here, so the two binaries answer through the same source. Not a binding
itself: it registers nothing, and the host surface tests glob only
`_mojolearn_*_host.mojo`.

WHY IT SHIPS. `kpss_test` does not train a model. It computes a statistic
from the caller's own series and returns a flag and a value, so the
saved-model inference boundary (public CPU inference, CPU training internal)
never had a side for it to fall on, and it sat unreachable on a CPU-only
install because the family that carried it, `tsa`, holds `holtwinters_fit`
and so stays a source reference build. Registering it here puts the test on
the shipped side without shipping any fit: the `_mojolearn_tsa` route already
falls back to `_mojolearn_forecast_host` when the reference binding is not
built (`host_surface.inference_routes()`), so `python/mojolearn/_tsa_impl.py`
reaches this entry unchanged.

The arithmetic is `tsa/checks/kpss_oracle.mojo::kpss_host_f32`, the serial
Float32 replay, unchanged by this move.

THE SABOTAGE ARM is that oracle's own `KPSS_ORACLE_HOST_SABOTAGE`, raised by
`-D MOJOLEARN_HOST_SABOTAGE=1` (the host builds' negative control). It is
re-exported here so both bindings' `*_host_sabotage` read-backs report it and
a sabotage build is refused outside the gate.
"""
from std.python import Python, PythonObject
from bindings.py_balanced import py_call as _balanced_call
from std.python._cpython import GILReleased

from core.hostptr import f32_ptr, i32_ptr
from tsa.checks.kpss_oracle import KPSS_ORACLE_HOST_SABOTAGE, kpss_host_f32
from core.kpss_host_test_impl import KPSS_DECISION_SABOTAGE, _kpss_flags_host, kpss_select_d_host


def _kpss_index(value: PythonObject) raises -> Int:
    var type_name = String(py=value.__class__.__name__)
    if type_name == "bool" or type_name == "bool_":
        raise Error("kpss host: integers expected, not booleans")
    var operator_module = Python.import_module("operator")
    # operator.index through the balanced call: PythonObject.__call__ keeps one
    # reference on each argument (bindings/py_balanced.mojo), which leaked every
    # buffer address passed to a host entry.
    return Int(py=_balanced_call(operator_module.index, value))


def kpss_test_binding(
    y_addr: PythonObject,
    flags_addr: PythonObject,
    stat_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`kpss_test` on the host by `kpss_host_f32`. Returns `batch_size`.

    `params` is, in this exact order (the GPU binding's, mirrored in
    `python/mojolearn/_tsa_impl.py`):

        0  batch_size
        1  n_obs
        2  d               order of simple differencing
        3  D               order of seasonal differencing
        4  s               seasonal period
        5  pval_threshold  (float)

    `y_addr` reads `batch_size * n_obs` float32, each series contiguous.
    `flags_addr` is written with `batch_size` int32 (1 stationary, 0 not),
    `stat_addr` with `batch_size` float32 statistics."""
    if len(params) != 6:
        raise Error(
            "kpss_test: params must contain 6 values (batch_size, n_obs, d,"
            " D, s, pval_threshold), got " + String(len(params))
        )
    var y_address = _kpss_index(y_addr)
    var fp = i32_ptr(_kpss_index(flags_addr))
    var sp = f32_ptr(_kpss_index(stat_addr))
    var batch_size = _kpss_index(params[0])
    var n_obs = _kpss_index(params[1])
    var d = _kpss_index(params[2])
    var D = _kpss_index(params[3])
    var s = _kpss_index(params[4])
    var pval = Float32(Float64(py=params[5]))
    with GILReleased(Python()):
        _kpss_flags_host(y_address, batch_size, n_obs, d, D, s, pval, fp, sp)
    return PythonObject(batch_size)


def select_d_host_binding(
    y_addr: PythonObject,
    d_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """auto_arima's "Choose the hyper-parameter d" on the host (lane
    py-runtime-b; it was a Python loop over `kpss_test`'s flags in
    `_tsa_impl.select_d`): d = the first order in 0 .. d_max - 1 whose KPSS
    test (`_kpss_flags_host`, the words `kpss_test` writes) judges the series
    stationary, else d_max. Orders stop once every series is decided, as the
    Python loop did, so a later order's refusal is raised only when it is
    reached. The same params and layout as the GPU binding's `select_d`:

        0  batch_size
        1  n_obs
        2  D               order of seasonal differencing, GIVEN not chosen
        3  s               seasonal period
        4  d_max           0 <= d_max <= 2 - D
        5  pval_threshold  (float)

    `d_addr` is written with `batch_size` int32. Returns `batch_size`."""
    if len(params) != 6:
        raise Error(
            "select_d: params must contain 6 values (batch_size, n_obs, D, s,"
            " d_max, pval_threshold), got " + String(len(params))
        )
    var y_address = _kpss_index(y_addr)
    var dp = i32_ptr(_kpss_index(d_addr))
    var batch_size = _kpss_index(params[0])
    var n_obs = _kpss_index(params[1])
    var D = _kpss_index(params[2])
    var s = _kpss_index(params[3])
    var d_max = _kpss_index(params[4])
    var pval = Float32(Float64(py=params[5]))
    if d_max < 0 or d_max + D > 2:
        raise Error(
            "select_d: d_max must satisfy 0 <= d_max <= 2 - D (d_max=" + String(d_max)
            + ", D=" + String(D) + "), refused by name"
        )
    with GILReleased(Python()):
        kpss_select_d_host(y_address, batch_size, n_obs, D, s, d_max, pval, dp)
    return PythonObject(batch_size)
