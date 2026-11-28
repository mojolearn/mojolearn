# SPDX-License-Identifier: Apache-2.0
# Fixed native CV wire records. No Python class/function, pickle reducer or
# import name from the request is executed. Import-time maps select all owners.
from bindings.native_loaded_module_door import native_loaded_module
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains

def _record_size(record: PythonObject, size: Int) raises:
    if len(record) != size:
        raise Error("ValueError: malformed native CV value record length")


def _sequence(record: PythonObject) raises:
    var b = native_loaded_module("builtins")
    if py_type(record) is not b.list and py_type(record) is not b.tuple:
        raise Error("TypeError: native CV wire sequences require exact list/tuple metadata")


def msel_wire_needs_numpy(value: PythonObject) raises -> Bool:
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if cls is b.tuple or cls is b.list:
        if len(value) and py_type(value[0]) is b.str:
            if String(value[0]) == "numpy_scalar" or String(value[0]) == "ndarray":
                return True
        for i in range(len(value)):  # driver-loop: generated wire metadata, never application iterators
            if msel_wire_needs_numpy(value[i]):
                return True
    elif cls is b.dict:
        var values = py_call(b.list, value.values())
        for i in range(len(values)):  # driver-loop: generated configuration record fields
            if msel_wire_needs_numpy(values[i]):
                return True
    return False
