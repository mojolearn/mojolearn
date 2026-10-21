# SPDX-License-Identifier: Apache-2.0
# Native classical ABI preparation. Only admitted native buffers are read;
# dtype conversion is the shared integer-defined CastIn conversion.
#
# Two array types cross this door. With NumPy installed, inputs are read as
# and outputs made as ndarrays (the context connects `ndarray`). Without
# NumPy, they are mojolearn Arrays: an input is admitted through the package's
# buffer conversion and every output is an Array of the same dtype and shape.
# The kernels see the same addresses, element kinds and strides either way,
# so both installs compute the same bits.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get
from core.cast_in import cast_in_kind, NPY_HALF, NPY_FLOAT, NPY_DOUBLE
from core.classical_arrays_impl import classical_cast_rows, classical_check_weights
from bindings.classical_warnings import classical_warn_column
from bindings.sequential_admission import sequential_admit_value, _ordinary_torch
from bindings.sequential_arrays import sequential_array, sequential_address
from bindings.sequential_inputs import sequential_input
from bindings.native_torch_admission import native_torch_numpy
from bindings.classical_pandas import native_pandas_exact, native_pandas_f32, classical_admit_value


def classical_numpy(context: PythonObject) raises -> Bool:
    """True when NumPy is connected: arrays are ndarrays. Otherwise they are
    mojolearn Arrays."""
    return py_get(context, "ndarray") is not None


def classical_is_array(value: PythonObject, context: PythonObject) raises -> Bool:
    """True for an exact mojolearn Array."""
    return py_type(value) is context["array_class"]


def classical_typed(shape: PythonObject, typestr: String, context: PythonObject) raises -> PythonObject:
    """A zero-filled array of `shape` holding "<f4", "<f8" or "<i8"."""
    if classical_numpy(context):
        var key = String("float32") if typestr == "<f4" else (String("float64") if typestr == "<f8" else String("int64"))
        return sequential_array(shape, context[PythonObject(key)], context)
    return py_call(context["array_class"], shape, PythonObject(typestr))


def classical_array(rows: Int, cols: Int, context: PythonObject, wide: Bool = False) raises -> PythonObject:
    return classical_typed(Python.tuple(rows) if cols == 0 else Python.tuple(rows, cols),
                           "<f8" if wide else "<f4", context)


def classical_input(value: PythonObject, context: PythonObject, name: String = "input") raises -> PythonObject:
    """An admitted input as an array, before any cast: an ndarray view with
    NumPy, an Array without it."""
    if classical_numpy(context):
        return sequential_input(value, context)
    if classical_is_array(value, context):
        return value
    # NumPy-free input step: the package's buffer conversion (`_buffer.
    # _materialize`) returns a zero-copy Array over a contiguous buffer, and
    # otherwise one C-order Array of the buffer's or nested sequence's numbers.
    return py_call(context["array_materialize"], value, PythonObject(name))[0]


def classical_kind(value: PythonObject, context: PythonObject) raises -> String:
    """The element kind letter: 'b', 'i', 'u', 'f' or 'c' for an ndarray;
    'i', 'u' or 'f' for an Array (its dtype is a little-endian typestr)."""
    if classical_is_array(value, context):
        return String(value.dtype[1])
    return String(value.dtype.kind)


def classical_cast_kind(value: PythonObject, item: Int, name: String, context: PythonObject) raises -> Int:
    """The CastIn element kind of an admitted array."""
    if classical_is_array(value, context):
        var letter = classical_kind(value, context)
        if letter == "f":
            return cast_in_kind(NPY_HALF if item == 2 else (NPY_FLOAT if item == 4 else NPY_DOUBLE), item, name)
        # Signed numbers are odd and unsigned even in the dtype numbering.
        return cast_in_kind(1 if letter == "i" else 2, item, name)
    return cast_in_kind(Int(py=value.dtype.num), item, name)


def classical_c_order(value: PythonObject, context: PythonObject) raises -> Bool:
    """True when the array is laid out in C order."""
    if classical_is_array(value, context):
        return String(value.order) == "C" or Int(py=value.ndim) < 2
    return Bool(py=value.flags.c_contiguous)


def classical_is_f32(value: PythonObject, context: PythonObject) raises -> Bool:
    if classical_is_array(value, context):
        return String(value.dtype) == "<f4"
    return value.dtype is context["float32"]


def classical_dtype_str(value: PythonObject, context: PythonObject) raises -> String:
    """The array's typestr, such as "<f4"."""
    if classical_is_array(value, context):
        return String(value.dtype)
    return String(value.dtype.str)


def classical_c_bytes(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """The array's bytes in C order: a borrowed byte view of a C-contiguous
    ndarray, otherwise one copy."""
    var b = Python.import_module("builtins")
    if classical_is_array(value, context):
        # ravel is a view of a C-order Array and a C-order copy of an F-order one.
        return py_call(py_call(value.ravel).tobytes)
    var view = py_call(b.memoryview, value)
    if Int(py=value.nbytes) == 0:
        return b.bytes()
    return py_call(view.cast, "B") if Bool(py=value.flags.c_contiguous) else py_call(view.tobytes, "C")


def classical_item(value: PythonObject, index: Int, context: PythonObject) raises -> PythonObject:
    """Element `index` of the array in C order, as a scalar."""
    if classical_is_array(value, context):
        return py_call(value.ravel)[index]
    return value.flat[index]


def classical_host_f32(source: PythonObject, ndim: Int, name: String, context: PythonObject, column_target: Bool = False) raises -> PythonObject:
    classical_admit_value(source, context)
    var b = Python.import_module("builtins")
    var prepared = source
    if native_pandas_exact(source):
        prepared = native_pandas_f32(source, context)
    elif _ordinary_torch(py_type(source), context):
        prepared = native_torch_numpy(source)
    var value = classical_input(prepared, context, name)
    var nd = Int(py=value.ndim)
    var column = column_target and ndim == 1 and nd == 2 and Int(py=value.shape[1]) == 1 and py_type(source) is not b.list and py_type(source) is not b.tuple and py_type(source) is not b.memoryview
    if column:
        classical_warn_column(context)
    if nd != ndim and not column:
        raise Error(name + " must be " + String(ndim) + "-D")
    var rows = 1 if nd == 0 else Int(py=value.shape[0])
    var cols = 1 if nd < 2 else Int(py=value.shape[1])
    if rows < 1 or cols < 1:
        raise Error(name + " is empty")
    # An Array is always in native (little-endian) byte order.
    if not classical_is_array(value, context) and not Bool(py=value.dtype.isnative):
        raise Error("TypeError: " + name + " must be in native byte order")
    var item = Int(py=value.itemsize)
    var kind = classical_cast_kind(value, item, name, context)
    var s0 = 0 if nd == 0 else Int(py=value.strides[0])
    var s1 = 0 if nd < 2 else Int(py=value.strides[1])
    var addr = Int(py=sequential_address(value))
    var borrowed = not column and classical_is_f32(value, context) and classical_c_order(value, context)
    var result = value if borrowed else classical_array(rows, cols if ndim == 2 else 0, context)
    var out = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(result)))
    classical_cast_rows(addr, s0, s1, rows, cols, item, kind, name, borrowed, out)
    _ = value^
    _ = prepared^
    return result


def classical_word(value: PythonObject, index: Int = 0) raises -> Float64:
    var address = Int(py=sequential_address(value))
    if Int(py=value.itemsize) == 8:
        return MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=address)[index]
    return Float64(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)[index])


def classical_host_weights(source: PythonObject, rows: Int, context: PythonObject) raises -> PythonObject:
    if source is None:
        return PythonObject(None)
    classical_admit_value(source, context)
    var prepared = native_pandas_f32(source, context) if native_pandas_exact(source) else source
    var raw = classical_input(prepared, context, "sample_weight")
    var scalar = Int(py=raw.ndim) == 0
    var value = classical_host_f32(source, 0 if scalar else 1, "sample_weight", context)
    if not scalar and Int(py=value.size) != rows:
        raise Error("sample_weight length differs from X")
    var out = classical_array(rows, 0, context) if scalar else value
    var input = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(value)))
    var output = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(out)))
    var any_positive = classical_check_weights(input, output, rows, scalar)
    if not any_positive:
        raise Error("Sample weights must contain at least one non-zero number.")
    _ = value^
    return out
