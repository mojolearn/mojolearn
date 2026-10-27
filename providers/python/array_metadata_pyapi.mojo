# SPDX-License-Identifier: Apache-2.0
# Common operation-local buffer metadata. Python ownership stays at this edge;
# numerical owners receive the existing native CastIn and native dimensions.
# Callers first admit supported native types and capture shape once, preserving
# their API-specific rank/emptiness refusals before dtype admission.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call
from bindings.native_loaded_module import native_loaded_module
from bindings.classical_arrays import classical_is_array, classical_address
from core.cast_in import CastIn, cast_in_kind, NPY_HALF, NPY_FLOAT, NPY_DOUBLE, CAST_KIND_FLOAT


struct ArrayMetadata(Copyable, Movable):
    var keep: PythonObject
    var shape: List[Int]
    var source: CastIn
    var count: Int
    var dtype_number: Int
    var native_byte_order: Bool
    var c_order: Bool
    var is_array: Bool

    def __init__(out self, keep: PythonObject, var shape: List[Int], source: CastIn,
                 count: Int, dtype_number: Int, native_byte_order: Bool,
                 c_order: Bool, is_array: Bool):
        self.keep = keep
        self.shape = shape^
        self.source = source
        self.count = count
        self.dtype_number = dtype_number
        self.native_byte_order = native_byte_order
        self.c_order = c_order
        self.is_array = is_array

    def python_shape(self) raises -> PythonObject:
        var result = Python.list()
        for dimension in self.shape:  # driver-loop: over captured native shape words
            _ = py_call(result.append, dimension)
        return py_call(native_loaded_module("builtins").tuple, result)

    def f32_c(self) -> Bool:
        return self.dtype_number == NPY_FLOAT and self.source.itemsize == 4 and self.c_order

    def replace_f32(mut self, keep: PythonObject, address: Int):
        # A new packed buffer's metadata is known from the allocation. Never
        # fetch it back from Python to rediscover the shape/dtype we supplied.
        self.keep = keep
        self.source = CastIn(address, CAST_KIND_FLOAT, 4, self.source.rows, self.source.cols,
                             self.source.cols * 4 if len(self.shape) > 0 else 0,
                             4 if len(self.shape) > 1 else 0)
        self.dtype_number = NPY_FLOAT
        self.native_byte_order = True
        self.c_order = True
        self.is_array = True


def array_shape_once(value: PythonObject) raises -> List[Int]:
    var shape = value.shape
    var result = List[Int]()
    for i in range(len(shape)):  # driver-loop: over admitted native array rank metadata
        result.append(Int(py=shape[i]))
    return result^


def array_metadata_once(value: PythonObject, var shape: List[Int], name: String,
                        context: PythonObject) raises -> ArrayMetadata:
    var nd = len(shape)
    if nd > 2:
        raise Error(name + " metadata requires an admitted scalar/vector/matrix")
    var array = classical_is_array(value, context)
    var dtype = value.dtype
    var item = Int(py=value.itemsize)
    var native_order = True
    var code = 0
    var contiguous = False
    if array:
        var letter = String(dtype[1])
        if letter == "f":
            code = NPY_HALF if item == 2 else (NPY_FLOAT if item == 4 else NPY_DOUBLE)
        else:
            code = 1 if letter == "i" else 2
        contiguous = nd < 2 or String(value.order) == "C"
    else:
        native_order = Bool(py=dtype.isnative)
        if not native_order:
            raise Error("TypeError: " + name + " must be in native byte order")
        code = Int(py=dtype.num)
        contiguous = Bool(py=value.flags.c_contiguous)
    var kind = cast_in_kind(code, item, name)
    var strides = value.strides
    var rows = shape[0] if nd > 0 else 1
    var cols = shape[1] if nd > 1 else 1
    var s0 = Int(py=strides[0]) if nd > 0 else 0
    var s1 = Int(py=strides[1]) if nd > 1 else 0
    var source = CastIn(Int(py=classical_address(value)), kind, item, rows, cols, s0, s1)
    return ArrayMetadata(value, shape^, source, rows * cols, code, native_order, contiguous, array)
