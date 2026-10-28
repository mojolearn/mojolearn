# SPDX-License-Identifier: Apache-2.0
# Prophet constructor and custom-seasonality operations, including admission
# and owner release. The public shells perform one compiled binding call.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.classical_pandas import classical_admit_value
from providers.python.autoarima_lifecycle_pyapi import _fields
from providers.python.forecast_initialize_pyapi import _scalar
from providers.python.forecast_owner_pyapi import forecast_commit


def prophet_fields(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = context["xseq_prophet_class"]
    if py_type(cls) is not b.type or py_type(model) is not cls:
        raise Error("TypeError: ProphetForecaster requires its connected native class")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    var layouts = context["xseq_prophet_layouts"]
    if len(mro) != len(layouts):
        raise Error("TypeError: ProphetForecaster inheritance changed")
    for i in range(len(layouts)):  # driver-loop: fixed connected storage/protocol metadata before state access
        if mro[i] is not layouts[i][0]:
            raise Error("TypeError: ProphetForecaster inheritance changed")
        var namespace = py_call(b.type.__dict__["__dict__"].__get__, mro[i])
        var hooks = layouts[i][1]
        for j in range(len(hooks)):  # driver-loop: native descriptor identities, never user hooks
            if py_get(namespace, hooks[j][0]) is not hooks[j][1]:
                raise Error("TypeError: ProphetForecaster methods or storage hooks changed")
    return _fields(model, cls)


def prophet_scalar(value: PythonObject, context: PythonObject) raises:
    _scalar(value, context["xseq_lifecycle_context"])


def prophet_initialize(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = prophet_fields(model, context)
    var b = native_loaded_module("builtins")
    for i in range(len(args)):  # driver-loop: passive constructor settings before comparisons or conversions
        if i == 1:
            classical_admit_value(args[i], context["xseq_lifecycle_context"])
        else:
            prophet_scalar(args[i], context)
    if py_type(args[0]) is not b.str or String(args[0]) != "linear":
        raise Error("NotImplementedError: ProphetForecaster: requested growth is not implemented ('linear')")
    if py_type(args[7]) is not b.str or (String(args[7]) != "additive" and String(args[7]) != "multiplicative"):
        raise Error("ValueError: seasonality_mode must be 'additive' or 'multiplicative'")
    var names = Python.tuple("growth", "changepoints", "n_changepoints", "changepoint_range",
        "yearly_seasonality", "weekly_seasonality", "daily_seasonality", "seasonality_mode",
        "seasonality_prior_scale", "holidays_prior_scale", "changepoint_prior_scale", "max_iter", "numeric_mode")
    var updates = Python.dict()
    for i in range(len(names)):  # driver-loop: native scalar normalization into staged constructor state
        var value = args[i]
        if i == 2 or i == 11:
            value = py_call(b.int, value)
        elif i == 3 or (i >= 8 and i <= 10):
            value = py_call(b.float, value)
        py_setitem(updates, names[i], value)
    py_setitem(updates, "seasonalities", Python.dict())
    forecast_commit(fields, updates, context["xseq_forecast_ownership"])
    return PythonObject(None)


def prophet_add_seasonality(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = prophet_fields(model, context)
    var b = native_loaded_module("builtins")
    for i in range(len(args)):  # driver-loop: custom seasonality settings before native conversion
        prophet_scalar(args[i], context)
    if py_type(args[0]) is not b.str:
        raise Error("TypeError: seasonality name requires a native string")
    var order = py_call(b.int, args[2])
    var period = py_call(b.float, args[1])
    if Bool(py=order < PythonObject(1)) or Float64(py=period) <= 0:
        raise Error("ValueError: add_seasonality: period > 0 and fourier_order >= 1")
    var prior = args[3]
    if prior is None:
        prior = fields["seasonality_prior_scale"]
        prophet_scalar(prior, context)
    else:
        prior = py_call(b.float, prior)
    var seasons = fields["seasonalities"]
    if py_type(seasons) is not b.dict:
        raise Error("TypeError: seasonalities require a native dictionary")
    var keys = py_call(b.list, seasons)
    for i in range(len(keys)):  # driver-loop: exact keys before seasonality lookup or replacement
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: seasonality names require native strings")
    var spec = Python.dict()
    py_setitem(spec, "period", period)
    py_setitem(spec, "fourier_order", order)
    py_setitem(spec, "prior_scale", prior)
    var updates = Python.dict()
    py_setitem(updates, args[0], spec)
    # Preserve the existing dictionary identity while admitting any old spec
    # before its replacement can release application-owned objects.
    forecast_commit(seasons, updates, context["xseq_forecast_ownership"])
    return model
