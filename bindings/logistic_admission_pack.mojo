# SPDX-License-Identifier: Apache-2.0
# Fixed LogisticRegression metadata; never invoke ParamsMixin, a Python
# constructor, user scalar/protocol hooks, or mutable class descriptors.
# The PACK half of the former `bindings/logistic_admission.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `bindings/logistic_admission_unpack.mojo`.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.classical_attributes import classical_check_layout
from bindings.classical_pandas import classical_admit_value
from bindings.sequential_admission import _one_of
from bindings.classifier_public_lifecycle_door import classifier_label_list
from bindings.native_owner_lifetime import native_release_admit, native_release_weakrefs


from bindings.logistic_admission_unpack import (
    _parameter,
    logistic_fields,
    logistic_release_admit,
    logistic_storage,
)


def logistic_initialize(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = logistic_storage(model, context)
    var names = context["lr_parameters"]
    if len(args) != len(names):
        raise Error("TypeError: native LogisticRegression requires its full constructor configuration")
    for i in range(len(names)):  # driver-loop: refusal before publishing any constructor parameters
        _parameter(names[i], args[i], context, True)
    for i in range(len(names)):  # driver-loop: retain all prior constructor owners until complete release admission
        var name = names[i]
        var replacing = String(name) != "numeric_mode" or args[i] is not None or not py_contains(fields, name)
        if replacing and py_contains(fields, name) and fields[name] is not args[i]:
            logistic_release_admit(fields[name], context)
    for i in range(len(names)):  # driver-loop: unchanged native constructor assignments
        if String(names[i]) != "numeric_mode" or args[i] is not None or not py_contains(fields, names[i]):
            py_setitem(fields, names[i], args[i])
    return PythonObject(None)


def logistic_get_params(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = logistic_fields(model, context, True)
    var out = Python.dict()
    var names = context["lr_sorted_parameters"]
    for i in range(len(names)):  # driver-loop: ParamsMixin's sorted parameter names
        py_setitem(out, names[i], fields[names[i]])
    return out


def logistic_set_params(model: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(values) is not b.dict:
        raise Error("TypeError: native LogisticRegression updates require ordinary keyword storage")
    if len(values) == 0:
        return model
    var params = logistic_get_params(model, context)
    var names = py_call(b.list, values)
    for i in range(len(names)):  # driver-loop: names before key comparison or formatting
        if py_type(names[i]) is not b.str:
            raise Error("TypeError: native LogisticRegression parameter names must be strings")
    for i in range(len(names)):  # driver-loop: ordered scalar/native-container parameter updates
        var key = names[i]
        var parts = py_call(key.partition, "__")
        if not py_contains(params, parts[0]):
            raise Error("Invalid parameter " + String(py_call(b.repr, parts[0])) + " for estimator "
                        + String(py_type(model).__name__) + ". Valid parameters are: "
                        + String(py_call(b.repr, py_call(b.list, context["lr_sorted_parameters"]))) + ".")
        if String(parts[1]) != "":
            raise Error("AttributeError: scalar parameter has no set_params")
        _parameter(parts[0], values[key], context, True)
        py_setitem(params, key, values[key])
    # Preserve ParamsMixin's successful reconstruction: a new configuration
    # removes fitted state. A rejected update leaves the original model alone.
    var fields = logistic_storage(model, context)
    var old = py_call(b.list, fields.items())
    for i in range(len(old)):  # driver-loop: admit names before keyed checks against the replacement configuration
        if py_type(old[i][0]) is not b.str:
            raise Error("TypeError: native LogisticRegression state requires exact string keys")
    for i in range(len(old)):  # driver-loop: successful reset removes all non-parameter state, including unknown user owners
        var key = old[i][0]
        if not py_contains(params, key) or params[key] is not old[i][1]:
            logistic_release_admit(old[i][1], context)
    _ = py_call(fields.clear)
    _ = py_call(fields.update, params)
    return model
