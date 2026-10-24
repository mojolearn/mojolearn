# SPDX-License-Identifier: Apache-2.0
# Fixed import-time archive recipes; never call a public Python method.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module, _loaded_fields
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields
from bindings.family_common import family_call, family_repr
from bindings.classical_pandas import classical_admit_value
from bindings.classical_attributes import classical_connect_numpy
from bindings.host_archive_native import host_archive_build


def _archive_prediction_record(model: PythonObject, method: Int, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = py_type(model)
    if py_type(cls) is not b.type:
        raise Error("TypeError: saved host prediction requires an ordinary native class")
    var records = context["host_archive_predict_records"]
    for i in range(len(records)):  # driver-loop: over the fixed saved-model class inventory
        var record = records[i]
        if cls is not record[0]:
            continue
        var ancestry = py_call(b.type.__dict__["__mro__"].__get__, cls)
        if ancestry is not record[1]:
            raise Error("TypeError: saved host prediction class ancestry changed")
        var name = PythonObject("predict" if method == 0 else "predict_proba")
        var descriptor = PythonObject(None)
        for k in range(len(ancestry)):  # driver-loop: inspect method metadata without descriptors
            var fields = py_call(b.type.__dict__["__dict__"].__get__, ancestry[k])
            var keys = py_call(b.list, fields)
            for j in range(len(keys)):  # driver-loop: admit class keys before dictionary lookup
                if py_type(keys[j]) is not b.str:
                    raise Error("TypeError: saved host prediction requires native class field names")
            if py_contains(fields, name):
                descriptor = fields[name]
                break
        if descriptor is not record[3][method]:
            raise Error("TypeError: saved host prediction cannot invoke a replaced public method or descriptor")
        if record[4][method] is None:
            raise Error("AttributeError: '" + String(record[2]) + "' object has no attribute '" + String(name) + "'")
        var fields = py_fields(model)
        # available_if(DBSCAN.prediction_data) formerly ran a Python lambda
        # during method lookup, before argument preparation. Keep that order.
        if String(record[2]) == "HostDBSCAN" and not Bool(py=fields["prediction_data"]):
            raise Error("AttributeError: This 'HostDBSCAN' has no attribute 'predict'")
        return record
    raise Error("TypeError: saved host class has no connected native prediction schema")


def host_archive_predict(path: PythonObject, source: PythonObject, probability: Bool,
                         context: PythonObject) raises -> PythonObject:
    var model = host_archive_build(path, context, PythonObject(None), False)
    var record = _archive_prediction_record(model, 1 if probability else 0, context)
    var recipe = record[4][1 if probability else 0]
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    var mode = py_get(fields, "numeric_mode")
    if mode is not None and (py_type(mode) is not b.str or String(mode) != "identical"):
        raise Error("mojolearn: " + String(record[2]) + " runs IDENTICAL only on the host; this model was saved " + family_repr(mode))
    classical_connect_numpy(context)
    classical_admit_value(source, context)
    # Revalidate the mutable module surface; a native entry pointer alone
    # does not authorize module __getattr__ hooks or replaced helper exports.
    var modules = context["host_archive_predict_modules"]
    for i in range(len(modules)):  # driver-loop: fixed connected host ABI modules
        var namespace = _loaded_fields(modules[i][0])
        var exports = modules[i][1]
        for k in range(len(exports)):  # driver-loop: retained native export identities
            if py_get(namespace, exports[k][0]) is not exports[k][1]:
                raise Error("TypeError: saved host prediction binding export changed")
    if py_type(recipe[0]) is b.str:
        raise Error(String(recipe[0]))
    if py_type(recipe[0]) is not context["host_archive_predict_entry_type"]:
        raise Error("TypeError: saved host prediction requires a compiled native owner")
    var arguments = Python.list()
    var words = recipe[3]
    for i in range(len(words)):  # driver-loop: the fixed positional argument recipe
        var word = words[i]
        var value = source if String(word[0]) == "input" else word[1]
        if String(word[0]) != "input" and String(word[0]) != "constant":
            raise Error("TypeError: malformed saved host prediction recipe")
        if String(word[0]) == "constant" and py_type(value) is b.str and String(value).startswith("ImportError:"):
            raise Error(String(value))
        _ = py_call(arguments.append, value)
    var result = family_call(recipe[0], model, recipe[1], py_call(b.tuple, arguments), recipe[2])
    _ = model^
    return result
