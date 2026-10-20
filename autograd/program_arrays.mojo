# SPDX-License-Identifier: Apache-2.0
# Native ABI allocation preserving the existing ndarray return contract.
# NumPy's ndarray C type wraps a native bytearray owner without numerical
# conversion, copying or a Python helper. Cache classes during Session setup,
# outside any graph run/fit. No ndarray.ctypes (a Python helper) is invoked.
from std.ffi import _Global
from std.python import Python, PythonObject
from autograd.rankn import ShapeN


struct _ProgramArrays(Defaultable, Movable):
    var ready: Bool
    var builtins: PythonObject
    var ndarray: PythonObject

    def __init__(out self):
        self.ready = False
        self.builtins = PythonObject(None)
        self.ndarray = PythonObject(None)


comptime _PROGRAM_ARRAYS = _Global[
    StorageType=_ProgramArrays, name="MojolearnProgramArraysV1", init_fn=_ProgramArrays.__init__,
]


def initialize_program_arrays() raises:
    var state = _PROGRAM_ARRAYS.get_or_create_ptr()
    if not state[].ready:
        state[].builtins = Python.import_module("builtins")
        state[].ndarray = Python.import_module("numpy").ndarray
        state[].ready = True


def output_array(shape: ShapeN) raises -> PythonObject:
    var state = _PROGRAM_ARRAYS.get_or_create_ptr()
    if not state[].ready:
        raise Error("program array boundary must be initialized during session setup")
    var dims = Python.list()
    for k in range(shape.ndim):  # small-loop(shape.ndim: tensor rank): shape metadata, never array elements
        dims.append(PythonObject(shape.d[k]))
    var storage = state[].builtins.bytearray(shape.count() * 4)
    # The C ndarray constructor retains storage as .base. Native executor
    # writes every value; no Python ndarray constructor/helper is involved.
    return state[].ndarray(dims, dtype="float32", buffer=storage)
