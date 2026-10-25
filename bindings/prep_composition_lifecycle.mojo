# SPDX-License-Identifier: Apache-2.0
# Public RFE/IterativeImputer lifecycle, before entering their existing owners.
# Inherited Python fit decorators must not materialize callback-bearing inputs
# before native admission or drive fitted-state/output handling after the call.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.classical_pandas import classical_admit_value
from bindings.classical_validate import classical_check_x
from bindings.classical_exceptions import classical_not_fitted_class
from bindings.classifier_public_lifecycle import _feature_names
from bindings.metrics_native_support import metric_warn
from bindings.optimizer_public_words import stash_python_error
from bindings.prep_native_estimator import prep_estimator_admit
from bindings.x_prep_native_inputs import _dense_values, _target


def composition_value(value: PythonObject, context: PythonObject, depth: Int = 0) raises:
    if depth > 64:
        raise Error("TypeError: native composition configuration exceeds the supported nesting depth")
    classical_admit_value(value, context)
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if cls is b.list or cls is b.tuple:
        for i in range(len(value)):  # driver-loop: admitted configuration containers
            composition_value(value[i], context, depth + 1)
    elif cls is py_get(context, "ndarray") and String(value.dtype.kind) == "O":
        # Exact ndarray storage may contain arbitrary Python objects. Inspect
        # the native flat iterator's references before later bool/float casts.
        var flat = value.flat
        for i in range(Int(py=value.size)):  # driver-loop: object-array configuration references
            composition_value(flat[i], context, depth + 1)


def composition_fields(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var cls = py_type(model)
    var b = native_loaded_module("builtins")
    if py_type(cls) is not b.type:
        raise Error("TypeError: native preprocessing composition refuses Python metaclass hooks")
    var plan = py_get(context["prep_composition_plans"], cls)
    if plan is None:
        raise Error("TypeError: native preprocessing composition requires its exact connected class; Python subclasses are not supported")
    var fields = py_call(plan["fields_get"], model, cls)
    # __dict__ assignment accepts dict subclasses. Refuse before iteration,
    # hashing or lookup can invoke a subclass's Python protocol methods.
    if py_type(fields) is not b.dict:
        raise Error("TypeError: native preprocessing state requires an exact dict")
    var keys = py_call(b.list, fields)
    for i in range(len(keys)):  # driver-loop: instance state keys before native dictionary lookup
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: native preprocessing state requires exact string keys")
    var names = plan["parameters"]
    for i in range(len(names)):  # driver-loop: declared composition configuration, before coercion or comparison
        var value = fields[names[i]]
        if String(names[i]) == "estimator":
            if value is not None:
                prep_estimator_admit(value, String(cls.__name__), context)
        else:
            composition_value(value, context)
    return fields


def composition_require_fitted(model: PythonObject, fields: PythonObject, operation: String,
                               context: PythonObject) raises:
    if py_contains(fields, "n_features_in_"):
        return
    var cls = classical_not_fitted_class(context)
    var message = "This " + String(py_type(model).__name__) + " instance is not fitted yet. Call 'fit' with appropriate arguments before using " + operation + "."
    ref cpy = Python().cpython()
    cpy.PyErr_SetString(cls._obj_ptr, message.as_c_string_slice().unsafe_ptr().as_unsafe_any_origin())
    var failure = stash_python_error()
    raise failure^


def composition_before(model: PythonObject, kind: String, operation: String, args: PythonObject,
                       context: PythonObject) raises -> PythonObject:
    var fields = composition_fields(model, context)
    var fitting = operation == "fit" or operation == "fit_transform"
    if not fitting:
        composition_require_fitted(model, fields, operation, context)
        if kind == "rfe":
            composition_value(fields["_mask"], context)
    if operation == "get_support":
        composition_value(args[0], context)
        return args
    # Admit each argument before conversion. A dictionary of fit keywords is
    # metadata; inspect its exact names, then admit each native buffer value.
    var b = native_loaded_module("builtins")
    if fitting and len(args) > 2:
        if py_type(args[2]) is not b.dict:
            raise Error("TypeError: native composition fit parameters require an exact dict")
        var keys = py_call(b.list, args[2])
        for i in range(len(keys)):  # driver-loop: user-supplied fit keywords, no Python protocols
            if py_type(keys[i]) is not b.str:
                raise Error("TypeError: native composition keyword names must be exact strings")
            composition_value(args[2][keys[i]], context)
    var x = classical_check_x(args[0], String(py_type(model).__name__), context) if kind == "rfe" else _dense_values(args[0], context)
    var rows = Int(py=x.shape[0])
    if len(args) > 1:
        classical_admit_value(args[1], context)
        _target(args[1], rows, context)
    if not fitting:
        var names = _feature_names(args[0], context)
        var fitted = py_get(fields, "feature_names_in_")
        if names is not None and fitted is not None:
            classical_admit_value(fitted, context)
            var previous = py_call(b.list, fitted)
            for i in range(len(previous)):  # driver-loop: fitted feature-name references before equality
                if py_type(previous[i]) is not b.str:
                    raise Error("TypeError: fitted feature names require exact strings")
            if py_call(b.list, names) != previous:
                raise Error("The feature names should match those that were passed during fit.")
        elif names is not None:
            metric_warn("X has feature names, but " + String(py_type(model).__name__) + " was fitted without feature names", b.UserWarning, context)
        elif fitted is not None:
            metric_warn("X does not have valid feature names, but " + String(py_type(model).__name__) + " was fitted with feature names", b.UserWarning, context)
    if len(args) == 1:
        return Python.tuple(x)
    if len(args) == 2:
        return Python.tuple(x, args[1])
    return Python.tuple(x, args[1], args[2])


def composition_after_fit(model: PythonObject, source: PythonObject, context: PythonObject) raises:
    var fields = composition_fields(model, context)
    var names = _feature_names(source, context)
    if names is None:
        _ = py_call(fields.pop, "feature_names_in_", PythonObject(None))
    else:
        py_setitem(fields, "feature_names_in_", names)
    py_setitem(fields, "_mojolearn_fitted", PythonObject(True))
