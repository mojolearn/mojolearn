# SPDX-License-Identifier: Apache-2.0
# Source-preserving reduction seed for the native parameter dictionary crash.
# The copied helper is the exact pre-repair product path. This small export
# does not execute any estimator or GPU owner.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import PythonModuleBuilder
from std.os import abort
from bindings.params_metadata_native import _resolved
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_contains
from bindings.optimizer_public_words import stash_python_error


# Exact _value body from cc6ee39cc12f27e8daf44f83870b9321021e70d4.
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
                var failure = stash_python_error()
                raise failure^
            return PythonObject(from_owned=raw)
        if _resolved(kind, PythonObject("__get__"), context["missing"]) is not context["missing"]:
            raise Error("TypeError: native parameters cannot invoke Python properties; store constructor parameters as native fields")
    if py_contains(fields, name):
        return fields[name]
    if value is not context["missing"]:
        return value
    raise Error(missing_error + ": constructor parameter is not stored under its own name: " + String(name))


def lookup(model: PythonObject, fields: PythonObject, name: PythonObject,
           context: PythonObject, missing_error: PythonObject) raises -> PythonObject:
    return _value(model, fields, name, context, String(missing_error))


def direct_lookup(fields: PythonObject, name: PythonObject) raises -> PythonObject:
    return fields[name]


def cpython_lookup(fields: PythonObject, name: PythonObject) raises -> PythonObject:
    # Testing control only. The probe supplies exact dict/string objects.
    # PyDict_GetItemWithError borrows the result; Py_NewRef establishes the
    # returned PythonObject's one owned reference, with no compensation for
    # another API's ownership and no CPython/toolchain modification.
    ref cpy = Python().cpython()
    var raw = cpy.lib.get_function[PyObjectPtr]("PyDict_GetItemWithError")(fields._obj_ptr, name._obj_ptr)
    if not raw:
        if cpy.PyErr_Occurred():
            raise cpy.unsafe_get_error()
        raise Error("KeyError: missing native dictionary key")
    return PythonObject(from_owned=cpy.Py_NewRef(raw))


@export
def PyInit__mojo_item_lookup() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojo_item_lookup")
        module.def_function[lookup]("lookup")
        module.def_function[direct_lookup]("direct_lookup")
        module.def_function[cpython_lookup]("cpython_lookup")
        return module.finalize()
    except error:
        abort(String(error))
