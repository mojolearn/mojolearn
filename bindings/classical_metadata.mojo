# SPDX-License-Identifier: Apache-2.0
# Native constructor/parameter metadata; no invocation of Python constructors,
# signature inspection, attribute descriptors or nested estimator callbacks.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.sequential_description import sequential_fields


def classical_parameter_names(ridge: Bool) raises -> PythonObject:
    if ridge:
        return Python.tuple("alpha", "fit_intercept", "normalize", "numeric_mode", "solver")
    return Python.tuple("fit_intercept", "numeric_mode")


def classical_initialize(model: PythonObject, args: PythonObject, ridge: Bool) raises -> PythonObject:
    var fields = sequential_fields(model)
    if ridge:
        # Explicit ABI order follows Ridge's public signature.
        py_setitem(fields, "alpha", args[0])
        py_setitem(fields, "solver", args[1])
        py_setitem(fields, "fit_intercept", args[2])
        py_setitem(fields, "normalize", args[3])
        if args[4] is not None or not py_contains(fields, "numeric_mode"):
            py_setitem(fields, "numeric_mode", args[4])
    else:
        py_setitem(fields, "fit_intercept", args[0])
        if args[1] is not None or not py_contains(fields, "numeric_mode"):
            py_setitem(fields, "numeric_mode", args[1])
    return PythonObject(None)


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


def classical_set_params(model: PythonObject, params: PythonObject, ridge: Bool) raises -> PythonObject:
    var bi = Python.import_module("builtins")
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
            raise Error("Invalid parameter '" + String(name) + "' for estimator; valid parameters are " + String(py_call(bi.str, classical_parameter_names(ridge))))
        if String(parts[1]) != "":
            # This family has scalar parameters, so no supported nested native
            # estimator exists to dispatch. Never invoke a Python set_params.
            raise Error("AttributeError: scalar parameter has no set_params")
        py_setitem(values, name, params[key])
    # Match ParamsMixin: only a successful parameter update resets fit state.
    var fields = sequential_fields(model)
    fields.clear()
    py_call(fields.update, values)
    return model


def classical_is_fitted(model: PythonObject) raises -> PythonObject:
    var fields = sequential_fields(model)
    if Bool(py=py_get(fields, "_mojolearn_fitted", False)) or Bool(py=py_get(fields, "_mojolearn_fitting", False)):
        return PythonObject(True)
    var pairs = py_call(Python.import_module("builtins").list, fields.items())
    for k in range(len(pairs)):  # driver-loop: over stored fitted attribute metadata
        var key = pairs[k][0]
        if py_type(key) is not Python.import_module("builtins").str:
            continue
        if Bool(py=py_call(key.endswith, "_")) and not Bool(py=py_call(key.startswith, "_")) and pairs[k][1] is not None:
            return PythonObject(True)
    return PythonObject(False)
