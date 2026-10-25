# SPDX-License-Identifier: Apache-2.0
# Testing-only self-contained reduction candidates. No mojolearn imports.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import PythonModuleBuilder
from std.os import abort


def native_loaded_module(name: String) raises -> PythonObject:
    # The standard supported import operation is intentionally used in this
    # diagnostic only. No product connection or module-registry code remains.
    return Python.import_module(name)


def py_call(callable: PythonObject, *args: PythonObject) raises -> PythonObject:
    ref cpy = Python().cpython()
    var envelope = cpy.PyTuple_New(len(args))
    if not envelope:
        raise cpy.unsafe_get_error()
    for i in range(len(args)):
        _ = cpy.PyTuple_SetItem(envelope, i, cpy.Py_NewRef(args[i]._obj_ptr))
    var result = cpy.PyObject_Call(callable._obj_ptr, envelope, PyObjectPtr())
    cpy.Py_DecRef(envelope)
    if not result:
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=result)


def py_type(value: PythonObject) raises -> PythonObject:
    ref cpy = Python().cpython()
    return PythonObject(from_owned=cpy.PyObject_Type(value._obj_ptr))


def py_contains(container: PythonObject, key: PythonObject) raises -> Bool:
    ref cpy = Python().cpython()
    var found = cpy.PySequence_Contains(container._obj_ptr, key._obj_ptr)
    if found == -1:
        raise cpy.unsafe_get_error()
    return found == 1


def _class_fields(cls: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var fields = py_call(b.type.__dict__["__dict__"].__get__, cls)
    return fields


def _resolved(cls: PythonObject, name: PythonObject, missing: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var ancestry = py_call(b.type.__dict__["__mro__"].__get__, cls)
    for i in range(len(ancestry)):
        var fields = _class_fields(ancestry[i])
        if py_contains(fields, name):
            return fields[name]
    return missing


def _value(model: PythonObject, fields: PythonObject, name: PythonObject,
           context: PythonObject, missing_error: String) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = py_type(model)
    var value = _resolved(cls, name, context["missing"])
    if value is not context["missing"]:
        var kind = py_type(value)
        if kind is native_loaded_module("types").MemberDescriptorType:
            # Exact CPython member descriptors read native slot storage only.
            var getter = value.__get__
            var args = Python.tuple(model, cls)
            ref cpy = Python().cpython()
            var raw = cpy.PyObject_Call(getter._obj_ptr, args._obj_ptr, PyObjectPtr())
            if not raw:
                raise cpy.unsafe_get_error()
            return PythonObject(from_owned=raw)
        if _resolved(kind, PythonObject("__get__"), context["missing"]) is not context["missing"]:
            raise Error("TypeError: native parameters cannot invoke Python properties; store constructor parameters as native fields")
    if py_contains(fields, name):
        return fields[name]
    if value is not context["missing"]:
        return value
    raise Error(missing_error + ": constructor parameter is not stored under its own name: " + String(name))


def original_shaped(model: PythonObject, fields: PythonObject, name: PythonObject,
                    context: PythonObject, missing_error: PythonObject) raises -> PythonObject:
    return _value(model, fields, name, context, String(missing_error))


def conditional(fields: PythonObject, name: PythonObject) raises -> PythonObject:
    if py_contains(fields, name):
        return fields[name]
    raise Error("missing native dictionary key")


def candidate(fields: PythonObject, name: PythonObject, value: PythonObject,
              missing: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if value is not missing:
        if py_type(value) is b.property:
            raise Error("Python properties are refused")
    if py_contains(fields, name):
        return fields[name]
    if value is not missing:
        return value
    raise Error("missing native dictionary key")


def dictionary_candidate(fields: PythonObject, name: PythonObject, value: PythonObject,
                         context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if value is not context["missing"]:
        if py_type(value) is b.property:
            raise Error("Python properties are refused")
    if py_contains(fields, name):
        return fields[name]
    if value is not context["missing"]:
        return value
    raise Error("missing native dictionary key")


@export
def PyInit__mojo_item_lookup_min() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojo_item_lookup_min")
        module.def_function[original_shaped]("original_shaped")
        module.def_function[conditional]("conditional")
        module.def_function[candidate]("candidate")
        module.def_function[dictionary_candidate]("dictionary_candidate")
        return module.finalize()
    except error:
        abort(String(error))
