# SPDX-License-Identifier: Apache-2.0
"""Native Sequential buffer admission and ndarray ABI ownership."""
from std.python import Python, PythonObject
from memory import UnsafePointer, MutAnyOrigin


def sequential_array(shape: PythonObject, dtype: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var count = 1
    for k in range(len(shape)):  # driver-loop: over array rank metadata
        count *= Int(py=shape[k])
    var storage = b.bytearray(count * Int(py=dtype.itemsize))
    return context["ndarray"](shape, dtype=dtype, buffer=storage)


def sequential_address(value: PythonObject) raises -> PythonObject:
    return value.__array_interface__["data"][0]


def _sequence_values(value: PythonObject, depth: Int, mut dims: List[Int],
                     mut values: PythonObject, mut floating: Bool, mut integral: Bool,
                     mut signed_range: Bool, mut unsigned_range: Bool,
                     mut leaf_depth: Int, context: PythonObject) raises:
    var b = Python.import_module("builtins")
    if Bool(py=b.isinstance(value, b.list)) or Bool(py=b.isinstance(value, b.tuple)):
        var n = len(value)
        if depth >= 8 or n == 0:
            raise Error("input sequences must be nonempty and have rank at most eight")
        if leaf_depth >= 0 and depth >= leaf_depth:
            raise Error("input sequences must be rectangular")
        if depth == len(dims):
            dims.append(n)
        elif dims[depth] != n:
            raise Error("input sequences must be rectangular")
        for k in range(n):  # native input marshaling: Python scalar/container values become a native C buffer
            _sequence_values(value[k], depth + 1, dims, values, floating, integral,
                             signed_range, unsigned_range, leaf_depth, context)
    else:
        if leaf_depth < 0:
            leaf_depth = depth
        if depth != len(dims) or depth != leaf_depth:
            raise Error("input sequences must be rectangular")
        if Bool(py=b.isinstance(value, context["bool_types"])):
            values.append(value)
            return
        elif Bool(py=b.isinstance(value, context["float_types"])):
            floating = True
        elif Bool(py=b.isinstance(value, context["int_types"])):
            integral = True
            signed_range = signed_range and Bool(py=value >= context["int64_min"]) and Bool(py=value <= context["int64_max"])
            unsigned_range = unsigned_range and Bool(py=value >= PythonObject(0)) and Bool(py=value <= context["uint64_max"])
        else:
            raise Error("TypeError: inputs must contain numeric scalars")
        values.append(value)


def sequential_input(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    if Bool(py=b.isinstance(value, context["ndarray"])):
        return value
    if b.type(value) is context["array_class"]:
        # Array stores these as C slot descriptors. Never call its Python
        # __array_interface__, __buffer__, reshape or conversion helpers.
        var view = b.object.__getattribute__(value, "_mv")
        var shape = b.object.__getattribute__(value, "shape")
        var dtype = b.object.__getattribute__(value, "dtype")
        var order = b.object.__getattribute__(value, "order")
        return context["ndarray"](shape, dtype=dtype, buffer=view, order=order)
    if b.type(value) is context["array_type"]:
        var view = b.memoryview(value)
        return context["ndarray"](view.shape, dtype=view.format, buffer=view)
    var dims = List[Int]()
    var values = Python.list()
    var floating = False
    var integral = False
    var signed_range = True
    var unsigned_range = True
    var leaf_depth = -1
    _sequence_values(value, 0, dims, values, floating, integral, signed_range, unsigned_range, leaf_depth, context)
    var shape = Python.list()
    for k in range(len(dims)):  # driver-loop: over sequence input rank metadata
        shape.append(dims[k])
    var code = "d" if floating else ("q" if integral else "b")
    var dtype = context["float64"] if floating else (context["int64"] if integral else context["bool"])
    if not floating and integral and not signed_range:
        if not unsigned_range:
            raise Error("TypeError: input integers exceed the supported numeric array range")
        code = "Q"
        dtype = context["uint64"]
    var storage = context["array_type"](code, values)
    # The ndarray C type exposes the C array buffer without NumPy casting or
    # a Python ndarray helper; its base retains the storage through dispatch.
    return context["ndarray"](shape, dtype=dtype, buffer=storage)


def sequential_descriptor(value: PythonObject, name: String, samples: Bool = False) raises -> PythonObject:
    var nd = Int(py=value.ndim)
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
    var rows = 1
    var cols = 1
    var s0 = size
    var s1 = size
    if nd > 0:
        rows = Int(py=value.shape[0])
        s0 = Int(py=value.strides[0])
    if nd == 2:
        cols = Int(py=value.shape[1])
        s1 = Int(py=value.strides[1])
    elif nd > 2:
        if not Bool(py=value.flags.c_contiguous):
            raise Error(name + " with more than two dimensions must be C-contiguous")
        for k in range(1, nd):  # driver-loop: over sample dimensions in the native matrix descriptor
            cols *= Int(py=value.shape[k])
        s0 = cols * size
    return Python.list(sequential_address(value), dtype.num, size, rows, cols, s0, s1)


def sequential_x(value: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    var owned = sequential_input(value, context)
    if Int(py=owned.ndim) < 2 or Int(py=owned.ndim) > 8:
        raise Error(name + " must have shape (n_samples, ...) with 2 to 8 dimensions")
    _ = sequential_descriptor(owned, name, True)
    return owned


def sequential_curve(value: PythonObject, count: Int) raises -> PythonObject:
    var pointer = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(py=sequential_address(value)))
    var result = Python.list()
    for k in range(count):  # driver-loop: over the native training engine's per-epoch scalar report
        result.append(PythonObject(pointer[k]))
    return result
