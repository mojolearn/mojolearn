# SPDX-License-Identifier: Apache-2.0
"""Shared C comparisons for already-admitted exact decomposition scalar owners."""
from std.python import Python, PythonObject
from std.python._cpython import ExternalFunction, PyObjectPtr
from core.shared_slot import SharedSlot
from bindings.py_balanced import py_type


comptime _DecompCompare = ExternalFunction[
    "PyObject_RichCompareBool", def(PyObjectPtr, PyObjectPtr, Int32) thin abi("C") -> Int32]


struct DecompScalarAPI(Defaultable, Movable):
    var compare: _DecompCompare.type

    def __init__(out self):
        ref cpy = Python().cpython()
        self.compare = _DecompCompare.load(cpy.lib.borrow())


comptime DECOMP_SCALAR_API = SharedSlot[DecompScalarAPI, "MojolearnDecompScalarAPIV1"]


def decomp_equal_admitted(value: PythonObject, other: PythonObject) raises -> Bool:
    # Both operands are exact admitted scalar storage, so their C comparison
    # slots cannot dispatch to user protocols. Never call operator.eq.
    var result = DECOMP_SCALAR_API.get_or_create_ptr()[].compare(value._obj_ptr, other._obj_ptr, Int32(2))
    _ = value
    _ = other
    if result < 0:
        ref cpy = Python().cpython()
        raise cpy.unsafe_get_error()
    return result != 0


def decomp_name_admitted(value: PythonObject, name: String) raises -> Bool:
    # Caller admits the scalar first. The common exact-string path compares
    # native strings; native NumPy scalar/0-D comparisons keep their C slots.
    if py_type(value) is py_type(PythonObject("")):
        return String(value) == name
    return decomp_equal_admitted(value, PythonObject(name))

