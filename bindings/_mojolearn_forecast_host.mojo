# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CPU INFERENCE binding for the forecasters (lane/inference-forecast-umap-pca,
2026-09-15): prediction from a saved ARIMA model, with no fit in the binary.

HOST ONLY, IDENTICAL ONLY. It registers the GPU binding's `arima_predict`,
`arima_forecast`, `arima_vendor` and `arima_numeric_mode` names from
`bindings/arima_host_predict.mojo`, the same source the internal reference
binding `bindings/_mojolearn_arima_host.mojo` registers them from, and
nothing that fits. `arima_fit` is absent, so a CPU-only install that holds
only this binding refuses a fit by name twice: `ARIMA.fit` in Python outside
the internal reference context, and the `_HostBinding` proxy for the absent
name.

Why a binding of its own: the reference binding carries the whole fit
(estimate_x0, the least squares, the L-BFGS and the finite-difference
likelihood), and training-only code does not ship in the inference wheels.
The manifest declares
this family with `routes=None` and `serves=("_mojolearn_arima",)`:
`_backend` routes `_mojolearn_arima` here on a CPU-only install only when
the reference binding is not built, and `mojolearn.host_model` binds it for
a saved ARIMA model on any machine.

Since lane/inference-holtwinters (2026-09-15) it also serves saved
Holt-Winters models: `holtwinters_forecast` and `holtwinters_predict` (the
in-sample one-step predictions), from
`bindings/holtwinters_host_predict.mojo`, the source the reference
`bindings/_mojolearn_tsa_host.mojo` registers them from, over
`holtwinters/host/hw_predict.mojo`. `holtwinters_fit` is absent, and so is
every name of the decomposition, the BFGS and its line search. The manifest's
`serves` routes `_mojolearn_tsa` here when the reference tsa binding is not
built; `kpss_test` then refuses by name.

The sabotage arm (`forecast_host_sabotage`) is
`arima/host/arima_oracle.mojo::ARIMA_ORACLE_PREDICT_SABOTAGE` and
`bindings/holtwinters_host_predict.mojo::HW_PREDICT_SABOTAGE`: under
`-D MOJOLEARN_HOST_SABOTAGE=1` every finite predicted or forecast value has
its lowest bit flipped, so the saved-model gate must read a mismatch.
"""
from core.experiment_guards_series import SERIES_GUARDS_OK
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from arima.host.arima_oracle import ARIMA_ORACLE_PREDICT_SABOTAGE
from arima.arima_ops_forecast_host import publish_arima_ops_forecast_host
from core.holtwinters_host_predict_impl import HW_PREDICT_SABOTAGE
from bindings.kpss_host_test_unpack import KPSS_DECISION_SABOTAGE, KPSS_ORACLE_HOST_SABOTAGE, kpss_test_binding, select_d_host_binding
from tsa.tsa_ops_forecast_host import publish_tsa_hw_ops_forecast_host


def forecast_host_numeric_mode_binding() raises -> PythonObject:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, (
        "forecast host: this binding restates the IDENTICAL arms only;"
        " bindings/build_host_family.sh passes -D MOJOLEARN_NUMERIC_IDENTICAL=1"
    )
    return PythonObject(GLOBAL_NUMERIC_MODE)


def forecast_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def forecast_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu"."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "forecast host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_forecast_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def forecast_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary flips the lowest bit of every prediction on
    purpose (MOJOLEARN_HOST_SABOTAGE, MOJOLEARN_ARIMA_PREDICT_SABOTAGE or
    MOJOLEARN_HW_PREDICT_SABOTAGE), or walks the KPSS series sums descending
    (KPSS_ORACLE_HOST_SABOTAGE, since lane/expose-inference-surface): one
    read-back over every arm this binary carries, so a sabotage build of it
    is refused outside the gate whichever arm was raised."""
    return PythonObject(
        ARIMA_ORACLE_PREDICT_SABOTAGE or HW_PREDICT_SABOTAGE or KPSS_ORACLE_HOST_SABOTAGE or KPSS_DECISION_SABOTAGE
    )


def arima_vendor_binding() raises -> PythonObject:
    """"cpu", as every host binding answers."""
    return PythonObject(String("cpu"))


# TOMBSTONE: hdbscan_host_predict_sabotage and the host tsa_vendor aliases (dead native entries: the host loader reads <prefix>_sabotage / <prefix>_vendor) deleted 2026-10-26 by 2013bdd1e; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_host_aliases.patch


def arima_numeric_mode_binding() raises -> PythonObject:
    """The build's tier as the `NUMERIC_*` code, which `_arima_impl.py`
    cross-checks against the mode the saved model records."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_forecast_host.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_forecast_host() -> PythonObject:
    comptime assert SERIES_GUARDS_OK, "invalid integrated experiment configuration (series guards)"
    try:
        var module = PythonModuleBuilder("_mojolearn_forecast_host")
        module.def_function[forecast_host_numeric_mode_binding]("forecast_host_numeric_mode")
        module.def_function[forecast_host_vendor_binding]("forecast_host_vendor")
        module.def_function[forecast_host_column_binding]("forecast_host_column")
        module.def_function[forecast_host_sabotage_binding]("forecast_host_sabotage")
        module.def_function[arima_vendor_binding]("arima_vendor")
        module.def_function[arima_numeric_mode_binding]("arima_numeric_mode")
        # TOMBSTONE: dead Python exports arima_predict, arima_forecast (no product caller: the ARIMA and AutoARIMA owners reach these kernels through the published ARIMA operations table, core/arima_entry_table.mojo; lane dw-forecast-tail) deleted 2026-11-21; code recoverable at SHA_PLACEHOLDER.
        # Restore: git apply experiments/removed/dead_entries_arima_calls.patch
        # TOMBSTONE: hdbscan_host_predict_sabotage and the host tsa_vendor aliases (dead native entries: the host loader reads <prefix>_sabotage / <prefix>_vendor) deleted 2026-10-26 by 2013bdd1e; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_host_aliases.patch
        # TOMBSTONE: dead Python exports holtwinters_forecast, holtwinters_predict (no product caller: the ExponentialSmoothing owner reaches these kernels through the published Holt-Winters operations table, core/tsa_entry_table.mojo; lane dw-forecast-tail) deleted 2026-11-21; code recoverable at 8e1fedd052.
        # Restore: git apply experiments/removed/dead_entries_tsa_hw_calls.patch
        # The KPSS stationarity test (lane/expose-inference-surface,
        # 2026-09-16). It trains no model, so it belongs on the shipped side;
        # the `_mojolearn_tsa` route reaches it here when the reference
        # binding, which holds holtwinters_fit, is not built.
        module.def_function[kpss_test_binding]("kpss_test")
        module.def_function[select_d_host_binding]("select_d")
        # The Holt-Winters operations table of this inference image
        # (core/tsa_entry_table.mojo, family TSA_TABLE_FORECAST): the explicit
        # host class's route resolves it once per estimator.
        var published_module = module.finalize()
        publish_tsa_hw_ops_forecast_host()
        # The ARIMA operations table of this image (core/arima_entry_table.mojo,
        # family ARIMA_TABLE_FORECAST): prediction and forecast, no fit.
        publish_arima_ops_forecast_host()
        return published_module
    except error:
        abort(String("failed to create _mojolearn_forecast_host: ", error))
