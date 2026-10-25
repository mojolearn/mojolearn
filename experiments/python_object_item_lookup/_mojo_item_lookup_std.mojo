# SPDX-License-Identifier: Apache-2.0
# Testing-only independent reduction, using only public PythonObject operations.
# No mojolearn helpers, explicit CPython C APIs, FFI, or product replacement.
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.os import abort


def _resolved(cls: PythonObject, name: PythonObject, missing: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var ancestry = b.type.__dict__["__mro__"].__get__(cls)
    for i in range(len(ancestry)):
        var fields = b.type.__dict__["__dict__"].__get__(ancestry[i])
        if Bool(py=fields.__contains__(name)):
            return fields[name]
    return missing


def py_type(value: PythonObject) raises -> PythonObject:
    return Python.import_module("builtins").type(value)


def _value(model: PythonObject, fields: PythonObject, name: PythonObject,
           context: PythonObject, missing_error: String) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var cls = py_type(model)
    var value = _resolved(cls, name, context["missing"])
    if value is not context["missing"]:
        var kind = py_type(value)
        if kind is Python.import_module("types").MemberDescriptorType:
            var getter = value.__get__
            return getter(model, cls)
        if _resolved(kind, PythonObject("__get__"), context["missing"]) is not context["missing"]:
            raise Error("TypeError: native parameters cannot invoke Python properties; store constructor parameters as native fields")
    if Bool(py=fields.__contains__(name)):
        return fields[name]
    if value is not context["missing"]:
        return value
    raise Error(missing_error + ": constructor parameter is not stored under its own name: " + String(name))


def original_shaped(model: PythonObject, fields: PythonObject, name: PythonObject,
                    context: PythonObject, missing_error: PythonObject) raises -> PythonObject:
    return _value(model, fields, name, context, String(missing_error))


@export
def PyInit__mojo_item_lookup_std() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojo_item_lookup_std")
        module.def_function[original_shaped]("original_shaped")
        return module.finalize()
    except error:
        abort(String(error))
