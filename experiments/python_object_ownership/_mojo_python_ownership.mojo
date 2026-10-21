# SPDX-License-Identifier: Apache-2.0
"""Standalone standard-library ownership probe; no mojolearn imports."""
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.os import abort


def no_op(value: PythonObject) raises -> PythonObject:
    return PythonObject(None)


def assign_attribute(owner: PythonObject, value: PythonObject) raises -> PythonObject:
    owner.value = value
    return PythonObject(None)


def read_attribute(owner: PythonObject) raises -> PythonObject:
    return owner.value


def builtin_type(value: PythonObject) raises -> PythonObject:
    return Python.import_module("builtins").type(value)


def builtin_list(value: PythonObject) raises -> PythonObject:
    return Python.import_module("builtins").list(value)


def append_value(owner: PythonObject, value: PythonObject) raises -> PythonObject:
    owner.append(value)
    return PythonObject(None)


@export
def PyInit__mojo_python_ownership() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojo_python_ownership")
        module.def_function[no_op]("no_op")
        module.def_function[assign_attribute]("assign_attribute")
        module.def_function[read_attribute]("read_attribute")
        module.def_function[builtin_type]("builtin_type")
        module.def_function[builtin_list]("builtin_list")
        module.def_function[append_value]("append_value")
        return module.finalize()
    except error:
        abort(String(error))
