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
whose arms the host file does not restate (p, q or P above 1, any Q, d + D
of 2, p + q + k of 0). Since 2026-09-15 `arima_predict` answers an in-sample
prediction (`start < n_obs`) through `arima_host_predict`, from
`bindings/arima_host_predict.mojo`, which the forecast inference binding
registers too.

The sabotage arm (`arima_host_sabotage`) is
`arima/host/arima_oracle.mojo::ARIMA_ORACLE_HOST_SABOTAGE`: the
finite-difference step doubles, so every fitted model this binary returns
differs.
"""
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from bindings.arima_exog_layout import exog_filter_layout
from bindings.hostptr import f32_ptr, i32_ptr, read_f32
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from arima.host.arima_oracle import (
    ARIMA_ORACLE_HOST_SABOTAGE,
    arima_host_fit,
    arima_host_refuse_unrestated,
)
# `arima_predict` and `arima_forecast`, with `_order` and `_refuse_shape`,
# live in bindings/arima_host_predict.mojo since 2026-09-15, shared with the
# inference binding bindings/_mojolearn_forecast_host.mojo. `_order` calls
# the device's own `validate_order(order)` there.
from bindings.arima_host_predict import (
    _order,
    _refuse_shape,
    arima_forecast_binding,
    arima_predict_binding,
)


comptime ARIMA_METHOD_MLE = 0
comptime ARIMA_METHOD_CSS = 1
comptime ARIMA_METHOD_CSS_ML = 2


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


def arima_numeric_mode_binding() raises -> PythonObject:
    """The build's tier as the `NUMERIC_*` code, which `_arima_impl.py`
    cross-checks against the mode the package asked for."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def _refuse_method(method: Int) raises:
    """`arima/estimator.mojo::_refuse_method` (`:114-142`), its words."""
    if method == ARIMA_METHOD_MLE:
        return
    if method == ARIMA_METHOD_CSS:
        raise Error(
            "ARIMA: method='css' is not implemented; the conditional sum of"
            " squares log-likelihood (batched_arima.cu:271-391,"
            " conditional_sum_of_squares and sum_of_squares_kernel) and its"
            " `truncate` parameter have no implementation. Only MLE is offered;"
            " refused by name (arima/NOT_IMPLEMENTED.tsv)"
        )
    if method == ARIMA_METHOD_CSS_ML:
        raise Error(
            "ARIMA: method='css-ml' is not implemented; it starts the maximum"
            " likelihood fit from the CSS optimum, and the CSS"
            " log-likelihood (batched_arima.cu:271-391) has no implementation. Only"
            " MLE is offered; refused by name (arima/NOT_IMPLEMENTED.tsv)"
        )
    raise Error(
        "ARIMA: method code " + String(method) + " is not one of MLE ("
        + String(ARIMA_METHOD_MLE) + "), CSS (" + String(ARIMA_METHOD_CSS)
        + ") or CSS-ML (" + String(ARIMA_METHOD_CSS_ML) + ")"
    )


def arima_fit_binding(
    y_addr: PythonObject,
    exog_addr: PythonObject,
    params_addr: PythonObject,
    x_addr: PythonObject,
    x0_addr: PythonObject,
    stats_addr: PythonObject,
    flags_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`bindings/_mojolearn_arima.mojo::arima_fit_binding`'s contract on the
    host: the addresses are (y, exog, params, x, x0, stats, flags), `params`
    is (batch_size, n_obs, p, d, q, P, D, Q, s, k, n_exog, method,
    max_iterations); returns `N * batch_size`."""
    if len(params) != 13:
        raise Error(
            "arima_fit: params must contain 13 values (batch_size, n_obs, p,"
            " d, q, P, D, Q, s, k, n_exog, method, max_iterations), got "
            + String(len(params))
        )
    var y_address = Int(py=y_addr)
    _ = f32_ptr(y_address)
    var exog_address = Int(py=exog_addr)
    _ = f32_ptr(exog_address)
    var pp = f32_ptr(Int(py=params_addr))
    var xp = f32_ptr(Int(py=x_addr))
    var x0p = f32_ptr(Int(py=x0_addr))
    var sp = f32_ptr(Int(py=stats_addr))
    var fp = i32_ptr(Int(py=flags_addr))
    var batch_size = Int(py=params[0])
    var n_obs = Int(py=params[1])
    var p = Int(py=params[2])
    var d = Int(py=params[3])
    var q = Int(py=params[4])
    var P = Int(py=params[5])
    var D = Int(py=params[6])
    var Q = Int(py=params[7])
    var s = Int(py=params[8])
    var k = Int(py=params[9])
    var n_exog = Int(py=params[10])
    var method = Int(py=params[11])
    var max_iterations = Int(py=params[12])
    var written = 0
    with GILReleased(Python()):
        # `arima_fit_ptr_host` (`arima/estimator.mojo:361-369`), in its order.
        _refuse_method(method)
        var order = _order(p, d, q, P, D, Q, s, k, n_exog)
        _refuse_shape(batch_size, n_obs, "arima_fit")
        if max_iterations < 1:
            raise Error(
                "arima_fit: max_iterations must be >= 1 (max_iterations="
                + String(max_iterations) + ")"
            )
        arima_host_refuse_unrestated(order, "arima_fit")
        var N = order.complexity()
        var y = read_f32(y_address, batch_size * n_obs)
        var exog = exog_filter_layout(exog_address, batch_size, n_obs, order.n_exog, "exog")
        var r = arima_host_fit(y, exog, batch_size, n_obs, order, max_iterations)
        if (
            len(r.t_x) != N * batch_size
            or len(r.x) != N * batch_size
            or len(r.x0) != N * batch_size
            or len(r.loglike) != batch_size
            or len(r.fx) != batch_size
            or len(r.n_iter) != batch_size
            or len(r.retcode) != batch_size
        ):
            raise Error(
                "arima_fit: the host oracle returned arrays of an unexpected"
                " length; nothing written"
            )
        for i in range(N * batch_size):
            pp.unsafe_store(i, r.t_x[i])
            xp.unsafe_store(i, r.x[i])
            x0p.unsafe_store(i, r.x0[i])
        for b in range(batch_size):
            sp.unsafe_store(b, r.loglike[b])
            sp.unsafe_store(batch_size + b, r.fx[b])
            fp.unsafe_store(b, r.n_iter[b])
            fp.unsafe_store(batch_size + b, r.retcode[b])
        written = N * batch_size
    return PythonObject(written)


@export
def PyInit__mojolearn_arima_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_arima_host")
        module.def_function[arima_host_numeric_mode_binding]("arima_host_numeric_mode")
        module.def_function[arima_host_vendor_binding]("arima_host_vendor")
        module.def_function[arima_host_column_binding]("arima_host_column")
        module.def_function[arima_host_sabotage_binding]("arima_host_sabotage")
        module.def_function[arima_vendor_binding]("arima_vendor")
        module.def_function[arima_numeric_mode_binding]("arima_numeric_mode")
        module.def_function[arima_fit_binding]("arima_fit")
        module.def_function[arima_predict_binding]("arima_predict")
        module.def_function[arima_forecast_binding]("arima_forecast")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_arima_host: ", error))
