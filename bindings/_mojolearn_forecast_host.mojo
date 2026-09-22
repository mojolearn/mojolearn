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
in-sample one-step predictions) and `tsa_vendor`, from
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
from bindings.arima_host_predict import (
    arima_forecast_binding,
    arima_predict_binding,
)
from bindings.holtwinters_host_predict import (
    HW_PREDICT_SABOTAGE,
    holtwinters_forecast_binding,
    holtwinters_predict_binding,
)
from bindings.kpss_host_test import KPSS_DECISION_SABOTAGE, KPSS_ORACLE_HOST_SABOTAGE, kpss_test_binding


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


def tsa_vendor_binding() raises -> PythonObject:
    """"cpu": the `_mojolearn_tsa` route's vendor read-back when this binding
    serves a saved Holt-Winters model."""
    return PythonObject(String("cpu"))


def arima_numeric_mode_binding() raises -> PythonObject:
    """The build's tier as the `NUMERIC_*` code, which `_arima_impl.py`
    cross-checks against the mode the saved model records."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


@export
def PyInit__mojolearn_forecast_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_forecast_host")
        module.def_function[forecast_host_numeric_mode_binding]("forecast_host_numeric_mode")
        module.def_function[forecast_host_vendor_binding]("forecast_host_vendor")
        module.def_function[forecast_host_column_binding]("forecast_host_column")
        module.def_function[forecast_host_sabotage_binding]("forecast_host_sabotage")
        module.def_function[arima_vendor_binding]("arima_vendor")
        module.def_function[arima_numeric_mode_binding]("arima_numeric_mode")
        module.def_function[arima_predict_binding]("arima_predict")
        module.def_function[arima_forecast_binding]("arima_forecast")
        module.def_function[tsa_vendor_binding]("tsa_vendor")
        module.def_function[holtwinters_forecast_binding]("holtwinters_forecast")
        module.def_function[holtwinters_predict_binding]("holtwinters_predict")
        # The KPSS stationarity test (lane/expose-inference-surface,
        # 2026-09-16). It trains no model, so it belongs on the shipped side;
        # the `_mojolearn_tsa` route reaches it here when the reference
        # binding, which holds holtwinters_fit, is not built.
        module.def_function[kpss_test_binding]("kpss_test")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_forecast_host: ", error))
