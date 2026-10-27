# SPDX-License-Identifier: Apache-2.0
# Native ABI allocation preserving the existing ndarray return contract.
# NumPy's ndarray C type wraps a native bytearray owner without numerical
# conversion, copying or a Python helper. Cache classes during Session setup,
# outside any graph run/fit. No ndarray.ctypes (a Python helper) is invoked.
from std.ffi import _Global
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type
from bindings.classical_attributes import classical_connect_numpy
from bindings.sequential_admission import sequential_admit_value, _one_of
from bindings.native_array_view import native_array_ndarray_view
from std.memory import memcpy
from sequence.ops import FP
from autograd.rankn import ShapeN
from autograd.program_io import HostArray


struct _ProgramArrays(Defaultable, Movable):
    var ready: Bool
    var builtins: PythonObject
    var ndarray: PythonObject
    var integer_type: PythonObject
    var mmap_type: PythonObject
    var admission: PythonObject

    def __init__(out self):
        self.ready = False
        self.builtins = PythonObject(None)
        self.ndarray = PythonObject(None)
        self.integer_type = PythonObject(None)
        self.mmap_type = PythonObject(None)
        self.admission = PythonObject(None)


comptime _PROGRAM_ARRAYS = _Global[
    StorageType=_ProgramArrays, name="MojolearnProgramArraysV1", init_fn=_ProgramArrays.__init__,
]


def initialize_program_arrays(admission: PythonObject = None) raises:
    var state = _PROGRAM_ARRAYS.get_or_create_ptr()
    if admission is not None:
        classical_connect_numpy(admission)
        state[].admission = admission
    if not state[].ready:
        state[].builtins = Python.import_module("builtins")
        var numpy = Python.import_module("numpy")
        state[].ndarray = numpy.ndarray
        state[].integer_type = numpy.integer
        state[].mmap_type = Python.import_module("mmap").mmap
        state[].ready = True


def program_admit_value(value: PythonObject) raises:
    var state = _PROGRAM_ARRAYS.get_or_create_ptr()
    if state[].admission is None:
        raise Error("TypeError: program native input admission must be connected during package import")
    sequential_admit_value(value, state[].admission)


def program_numeric_scalar(value: PythonObject) raises -> Bool:
    var state = _PROGRAM_ARRAYS.get_or_create_ptr()
    var cls = py_type(value)
    if cls is py_type(PythonObject(0)) or cls is py_type(PythonObject(0.0)) or cls is py_type(PythonObject(True)):
        return True
    return state[].admission is not None and _one_of(cls, state[].admission["scalar_types"])


def program_native_input(value: PythonObject) raises -> PythonObject:
    # Reuse the common boundary's exact-type/protocol guard. It runs before
    # dtype, shape, scalar conversion or any foreign descriptor could execute.
    program_admit_value(value)
    var state = _PROGRAM_ARRAYS.get_or_create_ptr()
    if py_type(value) is state[].admission["array_class"]:
        # This is a zero-copy view over the Array's retained exporter. The
        # original public argument remains live through synchronous execution;
        # no copied temporary buffer is introduced or cached across calls.
        return native_array_ndarray_view(value, state[].admission)
    var cls = py_type(value)
    if (cls is not state[].ndarray and cls is not py_type(Python.list()) and cls is not py_type(Python.tuple())
            and not program_numeric_scalar(value)):
        raise Error("TypeError: program inputs require native numeric arrays, scalars or rectangular lists/tuples")
    return value


def output_array(shape: ShapeN, dtype: String = "float32") raises -> PythonObject:
    var state = _PROGRAM_ARRAYS.get_or_create_ptr()
    if not state[].ready:
        raise Error("program array boundary must be initialized during session setup")
    var dims = Python.list()
    for k in range(shape.ndim):  # small-loop(shape.ndim: tensor rank): shape metadata, never array elements
        _ = py_call(dims.append, PythonObject(shape.d[k]))
    var storage = py_call(state[].builtins.bytearray, shape.count() * 4)
    # The C ndarray constructor retains storage as .base. Native executor
    # writes every value; no Python ndarray constructor/helper is involved.
    return py_call(state[].ndarray, dims, dtype=PythonObject(dtype), buffer=storage)


def output_array_dims(dims: List[Int], dtype: String) raises -> PythonObject:
    """`output_array` in explicit dims (rank 0 to 8: an imported model's
    declared output shape), of 4-byte `dtype` items ("float32" or "int32")."""
    var state = _PROGRAM_ARRAYS.get_or_create_ptr()
    if not state[].ready:
        raise Error("program array boundary must be initialized during session setup")
    var shape = Python.list()
    var count = 1
    for k in range(len(dims)):  # small-loop(dims: tensor rank): shape metadata, never array elements
        _ = py_call(shape.append, PythonObject(dims[k]))
        count *= dims[k]
    var storage = py_call(state[].builtins.bytearray, count * 4)
    return py_call(state[].ndarray, shape, dtype=PythonObject(dtype), buffer=storage)


def is_ndarray(value: PythonObject) raises -> Bool:
    """A NumPy ndarray (or a subclass of it)."""
    var state = _PROGRAM_ARRAYS.get_or_create_ptr()
    if not state[].ready:
        raise Error("program array boundary must be initialized during session setup")
    return Bool(py=py_call(state[].builtins.isinstance, value, state[].ndarray))


def is_integer_input(value: PythonObject) raises -> Bool:
    # Exact native integers only: isinstance(int-subclass) can admit a custom
    # __int__/__index__ callback. The common registry also includes NumPy's
    # exact native integer scalar types and is connected at package import.
    var state = _PROGRAM_ARRAYS.get_or_create_ptr()
    if state[].admission is None:
        return py_type(value) is py_type(PythonObject(0))
    return _one_of(py_type(value), state[].admission["integer_types"])


def uint64_input(value: PythonObject, name: String) raises -> UInt64:
    if not is_integer_input(value):
        raise Error("TypeError: " + name + " must be an integer")
    var state = _PROGRAM_ARRAYS.get_or_create_ptr()
    var b = state[].builtins
    var number = py_call(b.int, value)
    var high = number.__rshift__(32)
    if Bool(py=number.__lt__(0)) or Bool(py=high.__gt__(4294967295)):
        raise Error("ValueError: " + name + " must be in [0, 2**64)")
    # Mojo's Python UInt64 conversion currently uses signed ssize_t.
    # Each limb fits signed Int; combine them only after native conversion.
    return (UInt64(Int(py=high)) << 32) | UInt64(Int(py=number.__and__(4294967295)))


def mapped_byte_array(path: PythonObject) raises -> PythonObject:
    # mmap is a native CPython buffer owner. The ndarray C type keeps the
    # mapping alive, so no Python safetensors reader or ndarray helper runs.
    var state = _PROGRAM_ARRAYS.get_or_create_ptr()
    var file = py_call(state[].builtins.open, path, "rb")
    var mapping = PythonObject(None)
    try:
        mapping = py_call(state[].mmap_type, file.fileno(), 0, access=1)
    finally:
        file.close()
    var dims = Python.list()
    _ = py_call(dims.append, len(mapping))
    return py_call(state[].ndarray, dims, dtype="uint8", buffer=mapping)


def host_array_py(h: HostArray) raises -> PythonObject:
    """A float32 ndarray of a typed host result: allocation and one memcpy."""
    var a = output_array(h.shape)
    if h.shape.count() > 0:
        memcpy(dest=FP(unsafe_from_address=Int(py=a.__array_interface__["data"][0])),
               src=h.data.unsafe_ptr(), count=h.shape.count())
    return a


def host_arrays_py(hs: List[HostArray]) raises -> PythonObject:
    """A Python list of host_array_py results."""
    var out = Python.list()
    for k in range(len(hs)):  # small-loop(hs: one array per program output): result packing, no data
        _ = py_call(out.append, host_array_py(hs[k]))
    return out
