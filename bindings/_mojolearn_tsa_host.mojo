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
and `bindings/_mojolearn_tsa.mojo`).
`select_d` (ARIMA's; lane py-runtime-b) is the first stationary order of
the same KPSS host test (`bindings/kpss_host_test.mojo`), so the CPU-only
route no longer loops over the flags in Python.

`kpss_test` (lane/cpu-training-batch3, 2026-09-14, the kpss lane) is
`tsa/checks/kpss_oracle.mojo::kpss_host_f32`, "the serial Float32 REPLAY of
every device stage in `tsa/impl/timeSeries/stationarity.mojo`,
statement for statement", which `stationarity_check` holds the device to bit for bit
under IDENTICAL. The guards are `kpss_test_host`'s, `kpss_test`'s and
`prepare_data`'s, in their order and words (`tsa/estimator.mojo`,
`tsa/impl/timeSeries/stationarity.mojo`, `tsa/impl/timeSeries/
arima_helpers.mojo`); the flag is `kpss_pvalue(stat) > pval_threshold`.
"""
from core.experiment_guards_series import SERIES_GUARDS_OK
from std.os import abort
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call as _balanced_call
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from core.hostptr import f32_ptr, i32_ptr, read_f32
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)

from checks.numerics import GLOBAL_NUMERIC_MODE
from core.holtwinters_host_predict_impl import HW_PREDICT_SABOTAGE
from holtwinters.host.hw_oracle import (
    HW_ORACLE_HOST_SABOTAGE,
    oracle_fit,
)
from holtwinters.impl.runner import (
    holtwinters_validate_data,
    holtwinters_validate_params,
)
from holtwinters.impl.tsa.holtwinters_params import seasonal_from_name
from core.kpss_host_test_impl import KPSS_DECISION_SABOTAGE
from tsa.checks.kpss_oracle import KPSS_ORACLE_HOST_SABOTAGE
from tsa.tsa_ops_host import publish_tsa_hw_ops_host


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


# TOMBSTONE: hdbscan_host_predict_sabotage and the host tsa_vendor aliases (dead native entries: the host loader reads <prefix>_sabotage / <prefix>_vendor) deleted 2026-10-26 by 2013bdd1e; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_host_aliases.patch


def build__mojolearn_tsa_host() -> PythonObject:
    comptime assert SERIES_GUARDS_OK, "invalid integrated experiment configuration (series guards)"
    try:
        var module = PythonModuleBuilder("_mojolearn_tsa_host")
        module.def_function[tsa_host_numeric_mode_binding]("tsa_host_numeric_mode")
        module.def_function[tsa_host_vendor_binding]("tsa_host_vendor")
        module.def_function[tsa_host_column_binding]("tsa_host_column")
        module.def_function[tsa_host_sabotage_binding]("tsa_host_sabotage")
        # TOMBSTONE: hdbscan_host_predict_sabotage and the host tsa_vendor aliases (dead native entries: the host loader reads <prefix>_sabotage / <prefix>_vendor) deleted 2026-10-26 by 2013bdd1e; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_host_aliases.patch
        # TOMBSTONE: dead Python exports holtwinters_fit, holtwinters_forecast, holtwinters_predict (no product caller: the ExponentialSmoothing owner reaches these kernels through the published Holt-Winters operations table, core/tsa_entry_table.mojo; lane dw-forecast-tail) deleted 2026-11-21; code recoverable at 8e1fedd052.
        # Restore: git apply experiments/removed/dead_entries_tsa_hw_calls.patch
        # TOMBSTONE: dead Python exports kpss_test, select_d (no product caller: the stationarity functions and AutoARIMA's search reach these kernels through the published tsa operations table, core/tsa_entry_table.mojo TSA_OP_KPSS_TEST / TSA_OP_SELECT_D; lane dw-forecast-tail) deleted 2026-11-21; code recoverable at 038b12431e.
        # Restore: git apply experiments/removed/dead_entries_tsa_kpss_calls.patch
        # The Holt-Winters operations table (core/tsa_entry_table.mojo): the
        # ExponentialSmoothing owner resolves it once per estimator and calls
        # through it; published once the module is ready.
        var published_module = module.finalize()
        publish_tsa_hw_ops_host()
        return published_module
    except error:
        abort(String("failed to create _mojolearn_tsa_host: ", error))
