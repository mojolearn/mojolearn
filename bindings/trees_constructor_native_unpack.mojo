# SPDX-License-Identifier: Apache-2.0
# Native Signature.bind/apply_defaults shared by registered tree
# constructors. Preserve parameter object identities and defer value checks
# until fit, exactly as the public constructor contract requires.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.mode_metadata_native_door import _fields
from bindings.params_metadata_native_door import _resolved
from bindings.trees_public_state_unpack import _release_tree_values


def tree_parameter_storage(model: PythonObject, names: PythonObject) raises:
    for i in range(len(names)):  # driver-loop: refuse parameter setters before direct constructor storage writes
        var declared = _resolved(py_type(model), names[i], PythonObject(None))
        if declared is not None:
            var cls = py_type(declared)
            if _resolved(cls, "__set__", PythonObject(None)) is not None or _resolved(cls, "__delete__", PythonObject(None)) is not None:
                raise Error("TypeError: native tree lifecycle requires ordinary parameter storage, not descriptor setters")


def tree_store_parameters(fields: PythonObject, values: PythonObject,
                          context: PythonObject, reset: Bool) raises:
    var b = native_loaded_module("builtins")
    var previous = py_call(b.dict, fields)
    var replacement = Python.dict() if reset else py_call(b.dict, previous)
    _ = py_call(replacement.update, values)
    _release_tree_values(py_call(b.list, previous.values()), py_call(b.list, replacement.values()), context)
    if reset:
        _ = py_call(fields.clear)
    _ = py_call(fields.update, replacement)
    _ = previous
