# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the `_mojolearn_arima` family, batched ARIMA (workstream E,
the arima, arima-011 and arima-seasonal-c lanes, 2026-09-14).

HOST ONLY. No DeviceContext, no kernel launch, no GPU. The arithmetic is
`arima/host/arima_oracle.mojo`: `arima_host_fit` (estimate_x0, the inverse
Jones transform, the batched L-BFGS over the finite-difference Kalman
likelihood, the forward transform and the log-likelihood at the fitted
point) and `arima_host_forecast` (the filter's forecast, undifferenced),
the device lane restated on the host so the fitted parameters and the
forecasts are meant to be the GPU columns' bytes.

THE EXPORTED NAMES ARE THE GPU BINDING'S NAMES, so
`python/mojolearn/_arima_impl.py::ARIMA` runs unchanged on a CPU-only
install through `_backend._HOST_MODULES` (`"_mojolearn_arima":
"_mojolearn_arima_host"`): `arima_fit`, `arima_predict` and
`arima_forecast` with the SAME address contract and the SAME 13-slot
params lists (the docstrings of `bindings/_mojolearn_arima.mojo` are the
contract and are not repeated here), `arima_vendor` answering "cpu" and
`arima_numeric_mode`.

THE VALIDATION IS THE GPU ENTRY'S, IN ITS ORDER (`arima/estimator.mojo`):
`_refuse_method` (restated below in its words, `:114-142`), the order
through the device's own host function `validate_order`
(`arima/impl/tsa/arima_common.mojo:108-132`, imported, so an order the GPU
refuses is refused here in the same sentence), `_refuse_shape`
(`:165-175`), the `max_iterations` bound, `_predict_into`'s start and end
checks (`:422-433`) and the forecast's `n_steps` bound (`:536-540`). After
it, `arima_host_refuse_unrestated` refuses by name the parameter values
whose arms the host file does not restate (p, q, P or Q above 1, d + D
of 2, p + q + k of 0). Since 2026-09-15 `arima_predict` answers an in-sample
prediction (`start < n_obs`) through `arima_host_predict`, from
`bindings/arima_host_predict.mojo`, which the forecast inference binding
registers too.

The sabotage arm (`arima_host_sabotage`) is
`arima/host/arima_oracle.mojo::ARIMA_ORACLE_HOST_SABOTAGE`: the
finite-difference step doubles, so every fitted model this binary returns
differs.
"""
from core.experiment_guards_series import SERIES_GUARDS_OK
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from arima.host.arima_oracle import ARIMA_ORACLE_HOST_SABOTAGE
# The fit, the prediction and the forecast are operations of this image's
# published table (arima/arima_ops_host.mojo); this root registers only the
# read-backs the host loader asks for.
from arima.arima_ops_host import publish_arima_ops_host


def arima_host_numeric_mode_binding() raises -> PythonObject:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, (
        "arima host: this binding restates the IDENTICAL arms only;"
        " bindings/build_host_family.sh passes -D MOJOLEARN_NUMERIC_IDENTICAL=1"
    )
    return PythonObject(GLOBAL_NUMERIC_MODE)


def arima_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def arima_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu".
    THE COLUMN IS THE CPU COLUMN, OR THIS DOES NOT BUILD; the assert lives
    in a function body PyInit registers, so it is compiled in every build.
    `bindings/build_host_family.sh` passes -D MOJOLEARN_COLUMN_CPU."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "arima host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_arima_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


# There is no `arima_host_detected_column` read-back, for the reason 8d16ce2f
# removed it from the forest and byte LM host bindings: the detected column
# folds to the GPU of the machine that ran the build. The comptime assert
# above is the check.


def arima_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary doubles the finite-difference step on purpose
    (-D MOJOLEARN_HOST_SABOTAGE=1, the gate's negative control;
    `arima/host/arima_oracle.mojo::ARIMA_ORACLE_HOST_SABOTAGE`)."""
    return PythonObject(ARIMA_ORACLE_HOST_SABOTAGE)


# The GPU binding's names, same contract.


def arima_vendor_binding() raises -> PythonObject:
    """"cpu". On a CPU-only install `_backend.vendor()` is "cpu" and the
    read-back cross-check expects that string from every host binding."""
    return PythonObject(String("cpu"))


#: lane/fam2-timeseries: the host column of ARIMA_ORDER_IC_DEVICE
#: (`arima/impl/fast_order_search.mojo`), spelled from the same defines
#: because this binding does not import the device modules: AutoARIMA's
#: criterion is the float32 one (`ic_running_min_f32`) exactly when the
#: device binding built from the same defines chooses orders on the device.
# TOMBSTONE: MOJOLEARN_IDN_ALL_OFF, MOJOLEARN_IDN_ARIMA_EVAL_WS_OFF, MOJOLEARN_IDN_ARIMA_LLONLY_OFF, MOJOLEARN_IDN_ARIMA_ORDER_BATCH_OFF, MOJOLEARN_ARIMA_ORDER_BATCH_OFF (owned here) and MOJOLEARN_IDN_ARIMA_ORDER_DEVICE_OFF, MOJOLEARN_IDN_ARIMA_IC_DEVICE_OFF (non-owner sites) old-path terms deleted 2026-11-15 by lane/expdel2-37-series; code recoverable at 6d8c0423cf.
# Restore: git apply experiments/removed/<DEFINE>.patch (owned) or experiments/removed/<DEFINE>.expdel2-37-series.patch (non-owner); record in docs/TOMBSTONES.md.
def arima_numeric_mode_binding() raises -> PythonObject:
    """The build's tier as the `NUMERIC_*` code, which `_arima_impl.py`
    cross-checks against the mode the package asked for."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def build__mojolearn_arima_host() -> PythonObject:
    comptime assert SERIES_GUARDS_OK, "invalid integrated experiment configuration (series guards)"
    try:
        var module = PythonModuleBuilder("_mojolearn_arima_host")
        module.def_function[arima_host_numeric_mode_binding]("arima_host_numeric_mode")
        module.def_function[arima_host_vendor_binding]("arima_host_vendor")
        module.def_function[arima_host_column_binding]("arima_host_column")
        module.def_function[arima_host_sabotage_binding]("arima_host_sabotage")
        module.def_function[arima_vendor_binding]("arima_vendor")
        module.def_function[arima_numeric_mode_binding]("arima_numeric_mode")
        # TOMBSTONE: dead Python exports arima_order_caps, arima_fit, arima_predict, arima_forecast (no product caller: the ARIMA and AutoARIMA owners reach these kernels through the published ARIMA operations table, core/arima_entry_table.mojo; lane dw-forecast-tail) deleted 2026-11-21; code recoverable at d8f4bf534c.
        # Restore: git apply experiments/removed/dead_entries_arima_calls.patch
        # The ARIMA operations table of this image (core/arima_entry_table.mojo).
        var published_module = module.finalize()
        publish_arima_ops_host()
        return published_module
    except error:
        abort(String("failed to create _mojolearn_arima_host: ", error))
