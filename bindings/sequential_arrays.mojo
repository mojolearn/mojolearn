# SPDX-License-Identifier: Apache-2.0
"""Native Sequential buffer admission and ndarray ABI ownership."""
from bindings.native_loaded_module import native_loaded_module, NativeModules
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type
from bindings.sequential_inputs import sequential_input
from core.cast_in import CastIn, CAST_KIND_BOOL, CAST_KIND_FLOAT, CAST_KIND_SIGNED, CAST_KIND_UNSIGNED


def sequential_array(mut mods: NativeModules, shape: PythonObject, dtype: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = mods.module("builtins")
    var count = 1
    for k in range(len(shape)):  # driver-loop: over array rank metadata
        count *= Int(py=shape[k])
    var storage = py_call(context["native_bytearray_type"], count * Int(py=dtype.itemsize))
    return py_call(context["ndarray"], shape, dtype=dtype, buffer=storage)


def sequential_array(shape: PythonObject, dtype: PythonObject, context: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return sequential_array(mods, shape, dtype, context)


def sequential_address(value: PythonObject) raises -> PythonObject:
    return value.__array_interface__["data"][0]


struct SequentialInputSnapshot(Copyable, Movable):
    # Python/provider ownership stays at this edge. The existing native CastIn
    # is the internal matrix descriptor; shape is native sample metadata.
    # This object is local to one operation, never cached on the model.
    var keep: PythonObject
    var source: CastIn
    var dtype_num: Int
    var shape: List[Int]

    def __init__(out self, keep: PythonObject, source: CastIn, dtype_num: Int, var shape: List[Int]):
        self.keep = keep
        self.source = source
        self.dtype_num = dtype_num
        self.shape = shape^

    def abi(self) raises -> PythonObject:
        # Preserve the existing compiled model ABI at the provider boundary.
        return Python.list(self.source.addr, self.dtype_num, self.source.itemsize,
                           self.source.rows, self.source.cols, self.source.s0, self.source.s1)


def _sequential_snapshot(value: PythonObject, name: String, samples: Bool, nd: Int) raises -> SequentialInputSnapshot:
    if nd > 8 or (not samples and nd > 2):
        raise Error(name + " has too many dimensions")
    if Int(py=value.size) == 0:
        raise Error(name + " must not be empty")
    var dtype = value.dtype
    var kind = String(dtype.kind)
    var size = Int(py=dtype.itemsize)
    if kind != "b" and kind != "i" and kind != "u" and kind != "f":
        raise Error("TypeError: " + name + " must be a numeric array")
    if kind == "f" and size != 2 and size != 4 and size != 8:
        raise Error("TypeError: " + name + " must hold float16, float32 or float64")
    if not Bool(py=dtype.isnative):
        raise Error("TypeError: " + name + " must be in native byte order")
    var pyshape = value.shape
    var strides = value.strides
    var shape = List[Int]()
    for k in range(nd):  # driver-loop: over admitted input rank metadata
        shape.append(Int(py=pyshape[k]))
    var rows = 1
    var cols = 1
    var s0 = size
    var s1 = size
    if nd > 0:
        rows = shape[0]
        s0 = Int(py=strides[0])
    if nd == 2:
        cols = shape[1]
        s1 = Int(py=strides[1])
    elif nd > 2:
        if not Bool(py=value.flags.c_contiguous):
            raise Error(name + " with more than two dimensions must be C-contiguous")
        for k in range(1, nd):  # driver-loop: over native sample dimensions
            cols *= shape[k]
        s0 = cols * size
    var native_kind = CAST_KIND_UNSIGNED
    if kind == "b":
        native_kind = CAST_KIND_BOOL
    elif kind == "i":
        native_kind = CAST_KIND_SIGNED
    elif kind == "f":
        native_kind = CAST_KIND_FLOAT
    var source = CastIn(Int(py=sequential_address(value)), native_kind, size, rows, cols, s0, s1)
    return SequentialInputSnapshot(value, source, Int(py=dtype.num), shape^)


def sequential_snapshot(value: PythonObject, name: String, samples: Bool = False) raises -> SequentialInputSnapshot:
    return _sequential_snapshot(value, name, samples, Int(py=value.ndim))


def sequential_descriptor(value: PythonObject, name: String, samples: Bool = False) raises -> PythonObject:
    var snapshot = sequential_snapshot(value, name, samples)
    return snapshot.abi()


def sequential_x_snapshot(mut mods: NativeModules, value: PythonObject, name: String, context: PythonObject) raises -> SequentialInputSnapshot:
    var owned = sequential_input(mods, value, context, True)
    var nd = Int(py=owned.ndim)
    if nd < 2 or nd > 8:
        raise Error(name + " must have shape (n_samples, ...) with 2 to 8 dimensions")
    return _sequential_snapshot(owned, name, True, nd)


def sequential_x(mut mods: NativeModules, value: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    var snapshot = sequential_x_snapshot(mods, value, name, context)
    return snapshot.keep


def sequential_curve(value: PythonObject, count: Int) raises -> PythonObject:
    var result = Python.list()
    if count == 0:
        return result
    var pointer = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(value)))
    for k in range(count):  # driver-loop: over the native training engine's per-epoch scalar report
        py_call(result.append, PythonObject(pointer[k]))
    return result
