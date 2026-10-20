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
    var integer_type: PythonObject

    def __init__(out self):
        self.ready = False
        self.builtins = PythonObject(None)
        self.ndarray = PythonObject(None)
        self.integer_type = PythonObject(None)


comptime _PROGRAM_ARRAYS = _Global[
    StorageType=_ProgramArrays, name="MojolearnProgramArraysV1", init_fn=_ProgramArrays.__init__,
]


def initialize_program_arrays() raises:
    var state = _PROGRAM_ARRAYS.get_or_create_ptr()
    if not state[].ready:
        state[].builtins = Python.import_module("builtins")
        var numpy = Python.import_module("numpy")
        state[].ndarray = numpy.ndarray
        state[].integer_type = numpy.integer
        state[].ready = True


def output_array(shape: ShapeN, dtype: String = "float32") raises -> PythonObject:
    var state = _PROGRAM_ARRAYS.get_or_create_ptr()
    if not state[].ready:
        raise Error("program array boundary must be initialized during session setup")
    var dims = Python.list()
    for k in range(shape.ndim):  # small-loop(shape.ndim: tensor rank): shape metadata, never array elements
        dims.append(PythonObject(shape.d[k]))
    var storage = state[].builtins.bytearray(shape.count() * 4)
    # The C ndarray constructor retains storage as .base. Native executor
    # writes every value; no Python ndarray constructor/helper is involved.
    return state[].ndarray(dims, dtype=PythonObject(dtype), buffer=storage)


def is_integer_input(value: PythonObject) raises -> Bool:
    # ndarray exposes __index__ too, but an array is never a tensor handle
    # or an integer option. Match the public int/np.integer contract exactly.
    var state = _PROGRAM_ARRAYS.get_or_create_ptr()
    var b = state[].builtins
    return not Bool(py=b.isinstance(value, b.bool)) and (
        Bool(py=b.isinstance(value, b.int)) or Bool(py=b.isinstance(value, state[].integer_type)))
