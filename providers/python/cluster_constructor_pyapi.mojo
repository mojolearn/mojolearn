# SPDX-License-Identifier: Apache-2.0
# Constructors retain parameters unchanged; algorithm validation still belongs
# to fit. This replaces Python attribute assignment without executing an input
# protocol or eagerly converting a caller's buffers/settings.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_type, py_get, py_setitem, py_contains
from bindings.mode_metadata_native import _fields
from bindings.classical_attributes import _type_mro, classical_connect_numpy
from bindings.native_owner_lifetime import native_release_admit


def cluster_initialize_py(model: PythonObject, values: PythonObject,
                          context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(values) is not b.tuple:
        raise Error("TypeError: cluster constructor requires its native argument tuple")
    var schemas = context["cluster_public_parameters"]
    if py_type(schemas) is not b.dict:
        raise Error("TypeError: cluster constructor requires its connected parameter schema")
    var ancestry = _type_mro(py_type(model))
    var names = PythonObject(None)
    for i in range(len(ancestry)):  # driver-loop: locate the connected constructor through passive ancestry
        if names is None:
            names = py_get(schemas, ancestry[i])
    if names is None or py_type(names) is not b.tuple:
        raise Error("TypeError: cluster constructor requires its connected native class")
    if len(values) != len(names):
        raise Error("TypeError: cluster constructor argument count differs from its parameter schema")
    classical_connect_numpy(context)
    var fields = _fields(model)
    for i in range(len(names)):  # driver-loop: admit every released owner before changing constructor state
        if py_type(names[i]) is not b.str:
            raise Error("TypeError: cluster constructor parameter names require exact strings")
        if py_contains(fields, names[i]) and fields[names[i]] is not values[i]:
            native_release_admit(fields[names[i]], context)
    for i in range(len(names)):  # driver-loop: retain fully admitted constructor metadata, never numerical work
        py_setitem(fields, names[i], values[i])
    return PythonObject(None)
