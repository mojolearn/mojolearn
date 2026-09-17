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
from std.sys.compile import is_defined
from std.math import isfinite
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from bindings.hostptr import f32_ptr, i32_ptr, read_f32
from tsa.checks.kpss_oracle import KPSS_ORACLE_HOST_SABOTAGE, kpss_host_f32


def _kpss_index(value: PythonObject) raises -> Int:
    var type_name = String(py=value.__class__.__name__)
    if type_name == "bool" or type_name == "bool_":
        raise Error("kpss host: integers expected, not booleans")
    var operator_module = Python.import_module("operator")
    return Int(py=operator_module.index(value))


# A reduction-order perturbation can change the statistic without crossing the
# stationarity threshold. This separate negative control targets the decision
# consumed by select_d. Production builds contain neither arm.
comptime KPSS_DECISION_SABOTAGE = is_defined["MOJOLEARN_KPSS_DECISION_SABOTAGE"]()


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
        # `kpss_test_host`'s `_refuse_empty_shape`.
        if batch_size < 1:
            raise Error(
                "kpss_test: batch_size must be >= 1 (batch_size=" + String(batch_size) + ")"
            )
        if n_obs < 1:
            raise Error("kpss_test: n_obs must be >= 1 (n_obs=" + String(n_obs) + ")")
        # `kpss_test`'s, in its order.
        var d_sD = d + s * D
        if n_obs <= d_sD:
            raise Error(
                "stationarity: n_obs (" + String(n_obs)
                + ") must be greater than d + s*D (" + String(d_sD) + ")"
            )
        var y = read_f32(y_address, batch_size * n_obs)
        for i in range(batch_size * n_obs):
            if not isfinite(y[i]):
                raise Error(
                    "kpss_test: y contains a non-finite value at index "
                    + String(i) + "; missing or infinite observations are refused by name"
                )
        # `prepare_data`'s, reached only when there is differencing to do.
        if d != 0 or D != 0:
            if d + D > 2:
                raise Error(
                    "prepare_data: d + D must be <= 2 (d=" + String(d) + ", D="
                    + String(D) + "), refused by name (arima.pyx:313)"
                )
            if D > 0 and s < 2:
                raise Error(
                    "prepare_data: seasonal differencing needs s >= 2 (s=" + String(s)
                    + "), refused by name (arima.pyx:310)"
                )
        var st = kpss_host_f32(y, batch_size, n_obs, d, D, s, pval)
        for b in range(batch_size):
            fp[b] = Int32(1) if st.stationary[b] else Int32(0)
            comptime if KPSS_DECISION_SABOTAGE:
                fp[b] = Int32(1) - fp[b]
            sp[b] = st.stat[b]
    return PythonObject(batch_size)
