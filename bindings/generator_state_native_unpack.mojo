# SPDX-License-Identifier: Apache-2.0
"""Native Generator metadata and the inclusive exhausted next-stream sentinel."""
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_setitem


def generator_fields(value: PythonObject) raises -> PythonObject:
    return py_call(native_loaded_module("builtins").object.__getattribute__, value, "__dict__")
