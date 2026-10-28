# SPDX-License-Identifier: Apache-2.0
# Native constructor/parameter metadata; no invocation of Python constructors,
# signature inspection, attribute descriptors or nested estimator callbacks.
from bindings.native_loaded_module import native_loaded_module
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.sequential_description import sequential_fields
from bindings.native_owner_lifetime import native_release_admit


def classical_parameter_names(ridge: Bool) raises -> PythonObject:
    if ridge:
        return Python.tuple("alpha", "fit_intercept", "normalize", "numeric_mode", "solver")
    return Python.tuple("fit_intercept", "numeric_mode")


def classical_initialize(model: PythonObject, args: PythonObject, ridge: Bool,
                         context: PythonObject) raises -> PythonObject:
    var fields = sequential_fields(model)
    # Explicit ABI order follows each public constructor signature.
    var names = Python.tuple("alpha", "solver", "fit_intercept", "normalize", "numeric_mode") if ridge else Python.tuple("fit_intercept", "numeric_mode")
    if len(args) != len(names):
        raise Error("TypeError: native linear constructor requires its complete configuration")
    for k in range(len(names)):  # driver-loop: admit every old owner before publishing any constructor field
        var name = names[k]
        var replacing = String(name) != "numeric_mode" or args[k] is not None or not py_contains(fields, name)
        if replacing and py_contains(fields, name) and fields[name] is not args[k]:
            native_release_admit(fields[name], context)
    for k in range(len(names)):  # driver-loop: preserve numeric_mode on repeated construction with None
        var name = names[k]
        if String(name) != "numeric_mode" or args[k] is not None or not py_contains(fields, name):
            py_setitem(fields, name, args[k])
    return PythonObject(None)


def classical_fit_replacements_admit(fields: PythonObject, ridge: Bool, context: PythonObject) raises:
    var names = Python.tuple("coef_", "intercept_", "_x_mean", "_y_mean", "n_features_in_", "input_copied_", "_mojolearn_fitted")
    for k in range(len(names)):  # driver-loop: refit replaces these retained scalar/buffer owners
        if py_contains(fields, names[k]):
            native_release_admit(fields[names[k]], context)
    if ridge and py_contains(fields, "solver_"):
        native_release_admit(fields["solver_"], context)


def classical_get_params(model: PythonObject, ridge: Bool) raises -> PythonObject:
    var fields = sequential_fields(model)
    var result = Python.dict()
    var names = classical_parameter_names(ridge)
    for k in range(len(names)):  # driver-loop: over fixed constructor parameter metadata
        var name = names[k]
        if not py_contains(fields, name):
            raise Error("TypeError: constructor parameter is not stored under its own name: " + String(name))
        py_setitem(result, name, fields[name])
    return result


def classical_set_params(model: PythonObject, params: PythonObject, ridge: Bool,
                         context: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    if py_type(params) is not bi.dict:
        raise Error("TypeError: parameter keywords must be a native dict")
    if len(params) == 0:
        return model
    var values = classical_get_params(model, ridge)
    var names = py_call(bi.list, params.keys())
    for k in range(len(names)):  # driver-loop: over caller parameter keyword names
        var key = names[k]
        if py_type(key) is not bi.str:
            raise Error("TypeError: parameter names must be native strings")
        var parts = py_call(key.partition, "__")
        var name = parts[0]
        if not py_contains(values, name):
            raise Error("Invalid parameter " + String(py_call(bi.repr, name)) + " for estimator " + String(py_type(model).__name__)
                        + ". Valid parameters are: " + String(py_call(bi.repr, py_call(bi.sorted, classical_parameter_names(ridge)))) + ".")
        if String(parts[1]) != "":
            # This family has scalar parameters, so no supported nested native
            # estimator exists to dispatch. Never invoke a Python set_params.
            raise Error("AttributeError: scalar parameter has no set_params")
        py_setitem(values, name, params[key])
    # Match ParamsMixin: only a successful parameter update resets fit state.
    var fields = sequential_fields(model)
    var old = py_call(bi.list, fields.items())
    for k in range(len(old)):  # driver-loop: reset removes all state, including user-added owners outside the fitted schema
        var key = old[k][0]
        if py_type(key) is not bi.str:
            raise Error("TypeError: native linear state requires exact string keys")
    for k in range(len(old)):  # driver-loop: unchanged parameter owners stay retained by values throughout clear/update
        var key = old[k][0]
        if not py_contains(values, key) or values[key] is not old[k][1]:
            native_release_admit(old[k][1], context)
    fields.clear()
    py_call(fields.update, values)
    return model


def classical_is_fitted(model: PythonObject) raises -> PythonObject:
    var fields = sequential_fields(model)
    if Bool(py=py_get(fields, "_mojolearn_fitted", False)) or Bool(py=py_get(fields, "_mojolearn_fitting", False)):
        return PythonObject(True)
    var pairs = py_call(native_loaded_module("builtins").list, fields.items())
    for k in range(len(pairs)):  # driver-loop: over stored fitted attribute metadata
        var key = pairs[k][0]
        if py_type(key) is not native_loaded_module("builtins").str:
            continue
        if Bool(py=py_call(key.endswith, "_")) and not Bool(py=py_call(key.startswith, "_")) and pairs[k][1] is not None:
            return PythonObject(True)
    return PythonObject(False)
