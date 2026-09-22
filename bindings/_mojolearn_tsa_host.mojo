# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the `_mojolearn_tsa` family, Holt-Winters today (the CPU
training lane, phase 1, 2026-09-13).

HOST ONLY. No DeviceContext, no kernel launch, no GPU. The arithmetic is
`holtwinters/host/hw_oracle.mojo::oracle_fit[DType.float32]` and
`oracle_forecast`, "HoltWintersFitHelper on the host: transpose,
decompose, BFGS, final eval", the float32 arm the device is held to BIT
FOR BIT under IDENTICAL by `hw_check::check_hw_device_equals_oracle`
(sse, alpha, beta, gamma, niter, criterion, level, trend, season and the
forecast). The validation is the GPU entry's, in the GPU entry's order
(`holtwinters/estimator.mojo::holtwinters_fit_ptr` then
`holtwinters_fit_host_traced`: the extent guards, `seasonal_from_name`,
`holtwinters_validate_params`, `holtwinters_validate_data`), through the
same host-only functions, so a bad call raises the same error.

THE EXPORTED NAMES ARE THE GPU BINDING'S NAMES for the fits this covers, so
`python/mojolearn/_tsa_impl.py::ExponentialSmoothing` runs unchanged on a
CPU-only install through `_backend._HOST_MODULES` (`"_mojolearn_tsa":
"_mojolearn_tsa_host"`): `holtwinters_fit` and `holtwinters_forecast` with
the SAME address contract and packed layouts (level, trend, season each
`components_len = (n - frequency) * batch_size` and TIME-MAJOR; sse, alpha,
beta, gamma; niter, criterion; mirrored word for word in `_tsa_impl.py`
and `bindings/_mojolearn_tsa.mojo`), and `tsa_vendor` answering "cpu".
`select_d` (ARIMA's) is deliberately absent and refuses BY NAME through
`_HostBinding`.

`kpss_test` (lane/cpu-training-batch3, 2026-09-14, the kpss lane) is
`tsa/checks/kpss_oracle.mojo::kpss_host_f32`, "the serial Float32 REPLAY of
every device stage in `tsa/impl/timeSeries/stationarity.mojo`, statement
for statement", which `stationarity_check` holds the device to bit for bit
under IDENTICAL. The guards are `kpss_test_host`'s, `kpss_test`'s and
`prepare_data`'s, in their order and words (`tsa/estimator.mojo`,
`tsa/impl/timeSeries/stationarity.mojo`, `tsa/impl/timeSeries/
arima_helpers.mojo`); the flag is `kpss_pvalue(stat) > pval_threshold`.
"""
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from bindings.hostptr import f32_ptr, i32_ptr, read_f32
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from std.math import isfinite
from std.memory import bitcast

from checks.numerics import GLOBAL_NUMERIC_MODE
from bindings.holtwinters_host_predict import (
    HW_PREDICT_SABOTAGE,
    holtwinters_forecast_binding,
    holtwinters_predict_binding,
)
from holtwinters.host.hw_oracle import (
    HW_ORACLE_HOST_SABOTAGE,
    oracle_fit,
)
from holtwinters.impl.runner import (
    holtwinters_validate_data,
    holtwinters_validate_params,
)
from holtwinters.impl.tsa.holtwinters_params import seasonal_from_name
from bindings.kpss_host_test import KPSS_DECISION_SABOTAGE, kpss_test_binding
from tsa.checks.kpss_oracle import KPSS_ORACLE_HOST_SABOTAGE


def _index(value: PythonObject) raises -> Int:
    var type_name = String(py=value.__class__.__name__)
    if type_name == "bool" or type_name == "bool_":
        raise Error("tsa host: integers expected, not booleans")
    var operator_module = Python.import_module("operator")
    return Int(py=operator_module.index(value))


def tsa_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def tsa_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def tsa_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu".
    THE COLUMN IS THE CPU COLUMN, OR THIS DOES NOT BUILD; the assert lives
    in a function body PyInit registers, so it is compiled in every build.
    `bindings/build_tsa_host.sh` passes -D MOJOLEARN_COLUMN_CPU."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "tsa host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_tsa_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


# There is no `tsa_host_detected_column` read-back, for the reason 8d16ce2f
# removed it from the forest and byte LM host bindings: the detected column
# folds to the GPU of the machine that ran the build, so its name would land
# in the vendor-neutral binary. The comptime assert above is the check.


def tsa_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary splits the SSE fused multiply-add and walks the
    KPSS series sums descending on purpose (-D MOJOLEARN_HOST_SABOTAGE=1,
    the gate's negative control; one define, every arm, the lowest-bit flip
    of `bindings/holtwinters_host_predict.mojo` included)."""
    return PythonObject(HW_ORACLE_HOST_SABOTAGE or KPSS_ORACLE_HOST_SABOTAGE or KPSS_DECISION_SABOTAGE or HW_PREDICT_SABOTAGE)


# The GPU binding's names, same contract.


def tsa_vendor_binding() raises -> PythonObject:
    """"cpu". On a CPU-only install `_backend.vendor()` is "cpu" and the
    read-back cross-check expects that string from every host binding."""
    return PythonObject(String("cpu"))


def holtwinters_fit_binding(
    data_addr: PythonObject,
    comps_addr: PythonObject,
    stats_addr: PythonObject,
    flags_addr: PythonObject,
    params: PythonObject,
    seasonal: PythonObject,
) raises -> PythonObject:
    """`ExponentialSmoothing(...).fit()` on the host by
    `oracle_fit[DType.float32]`. Returns `components_len = (n - frequency)
    * batch_size`.

    `params` is, in this exact order (mirrored in
    `python/mojolearn/_tsa_impl.py` and in the GPU binding):

        0  n              observations per series
        1  batch_size     cuML's ts_num
        2  frequency      cuML's seasonal_periods
        3  start_periods
        4  eps            (float)
        5  init_method    OPTIONAL: 0 heuristic (absent), 1 estimated

    `data_addr` reads `batch_size * n` float32, SERIES-MAJOR. `comps_addr`
    is written with `3 * components_len` float32 (level, trend, season,
    each TIME-MAJOR: series `s` at step `i` is `[s + i * batch_size]`),
    `stats_addr` with `4 * batch_size` float32 (sse, alpha, beta, gamma),
    `flags_addr` with `2 * batch_size` int32 (niter, criterion)."""
    if len(params) != 5 and len(params) != 6:
        raise Error(
            "holtwinters_fit: params must contain 5 or 6 values (n, batch_size,"
            " frequency, start_periods, eps[, init_method]), got " + String(len(params))
        )
    var data_address = _index(data_addr)
    var cp = f32_ptr(_index(comps_addr))
    var sp = f32_ptr(_index(stats_addr))
    var fp = i32_ptr(_index(flags_addr))
    var n = _index(params[0])
    var batch_size = _index(params[1])
    var frequency = _index(params[2])
    var start_periods = _index(params[3])
    var eps = Float32(Float64(py=params[4]))
    var init_method = _index(params[5]) if len(params) == 6 else 0
    if init_method != 0 and init_method != 1:
        raise Error("holtwinters_fit: init_method must be 0 (heuristic) or 1 (estimated), got " + String(init_method))
    var sname = String(py=seasonal)
    var components_len = 0
    with GILReleased(Python()):
        # `holtwinters_fit_ptr`'s guards on the extents the read dereferences
        # (DEVIATION 931), in its words, before the read.
        if batch_size < 1:
            raise Error(
                "holtwinters fit: batch_size must be >= 1 (batch_size="
                + String(batch_size) + "); the input read is batch_size * n cells"
            )
        if n < 1:
            raise Error(
                "holtwinters fit: n must be >= 1 (n=" + String(n)
                + "); the input read is batch_size * n cells"
            )
        var cells = batch_size * n
        if cells < 0 or cells // batch_size != n:
            raise Error(
                "holtwinters fit: batch_size * n overflowed (batch_size="
                + String(batch_size) + ", n=" + String(n) + ")"
            )
        var data = read_f32(data_address, cells)
        # `holtwinters_fit_host_traced`'s validation, in its order.
        var st = seasonal_from_name(sname)
        holtwinters_validate_params(n, batch_size, frequency, start_periods, eps)
        holtwinters_validate_data(data, n, batch_size, st)
        # THE ONE CALL THAT COMPUTES ANYTHING. trace_iters is a record, not
        # an arithmetic input; 0 keeps no per-iteration trace.
        var fitted = oracle_fit[DType.float32](
            data, n, batch_size, frequency, start_periods, st, eps, 0,
            init_method=init_method,
        )
        components_len = len(fitted.level)
        if (
            components_len != (n - frequency) * batch_size
            or len(fitted.trend) != components_len
            or len(fitted.season) != components_len
            or len(fitted.sse) != batch_size
            or len(fitted.alpha) != batch_size
            or len(fitted.beta) != batch_size
            or len(fitted.gamma) != batch_size
            or len(fitted.niter) != batch_size
            or len(fitted.criterion) != batch_size
        ):
            raise Error(
                "holtwinters_fit: the host oracle returned components of an"
                " unexpected length; nothing written"
            )
        for i in range(components_len):
            cp[i] = fitted.level[i]
            cp[components_len + i] = fitted.trend[i]
            cp[2 * components_len + i] = fitted.season[i]
        for b in range(batch_size):
            sp[b] = fitted.sse[b]
            sp[batch_size + b] = fitted.alpha[b]
            sp[2 * batch_size + b] = fitted.beta[b]
            sp[3 * batch_size + b] = fitted.gamma[b]
            fp[b] = Int32(fitted.niter[b])
            fp[batch_size + b] = Int32(fitted.criterion[b])
        comptime if HW_ORACLE_HOST_SABOTAGE:
            # THE FITTED-STATE ARM (lane/inference-holtwinters, 2026-09-15):
            # the split SSE multiply-add leaves the fitted bytes unchanged on
            # some fixtures (denormal, denormal_ftz, wide), so a saved model's
            # file could not move under the negative control. Every finite
            # per-series float (sse, alpha, beta, gamma) also has its lowest
            # bit flipped, a value perturbation, so every saved file moves.
            # The components are left alone: a flip there, followed by the
            # forecast binding's own output flip, restored the forecast's
            # bytes on two fixtures (the third x86 pod, DIVERGENT=16).
            for i in range(4 * batch_size):
                if isfinite(sp[i]):
                    sp[i] = bitcast[DType.float32](bitcast[DType.uint32](sp[i]) ^ UInt32(1))
    return PythonObject(components_len)


# `holtwinters_forecast` and `holtwinters_predict` are registered from
# `bindings/holtwinters_host_predict.mojo` (lane/inference-holtwinters,
# 2026-09-15), the source the shipped forecast inference binding registers
# them from; the forecast body is `oracle_forecast`'s, one spelling.


@export
def PyInit__mojolearn_tsa_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_tsa_host")
        module.def_function[tsa_host_numeric_mode_binding]("tsa_host_numeric_mode")
        module.def_function[tsa_host_vendor_binding]("tsa_host_vendor")
        module.def_function[tsa_host_column_binding]("tsa_host_column")
        module.def_function[tsa_host_sabotage_binding]("tsa_host_sabotage")
        module.def_function[tsa_vendor_binding]("tsa_vendor")
        module.def_function[holtwinters_fit_binding]("holtwinters_fit")
        module.def_function[holtwinters_forecast_binding]("holtwinters_forecast")
        module.def_function[holtwinters_predict_binding]("holtwinters_predict")
        module.def_function[kpss_test_binding]("kpss_test")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_tsa_host: ", error))
