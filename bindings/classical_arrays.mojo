# SPDX-License-Identifier: Apache-2.0
# Native classical ABI preparation. Only admitted native buffers are read;
# dtype conversion is the shared integer-defined CastIn conversion.
#
# Two array types cross this door. With NumPy installed, inputs are read as
# and outputs made as ndarrays (the context connects `ndarray`). Without
# NumPy, they are mojolearn Arrays: a caller's Array is read in place, a buffer
# or a nested sequence is packed once into an Array, and every output and
# working buffer is an Array assembled from its slots (no Python code runs).
# The kernels see the same element values either way, so both installs
# compute the same bits.
from bindings.native_array_storage import native_array_from_bytes, native_array_zeros
from bindings.native_loaded_module import native_loaded_module
from std.python import Python, PythonObject
from bindings.native_buffer_owner import native_buffer_view, native_storage_allocate
from bindings.py_balanced import py_call, py_type, py_get, py_setattr
from core.cast_in import cast_in_kind, NPY_HALF, NPY_FLOAT, NPY_DOUBLE
from core.classical_arrays_impl import classical_cast_rows, classical_check_weights
from bindings.classical_warnings import classical_warn_column
from bindings.sequential_admission import sequential_admit_value, _ordinary_torch
from bindings.sequential_arrays import sequential_array, sequential_address
from bindings.sequential_inputs import sequential_input, sequential_sequence
from bindings.native_torch_admission import native_torch_numpy
from bindings.classical_pandas import native_pandas_exact, native_pandas_f32, classical_admit_value


def classical_numpy(context: PythonObject) raises -> Bool:
    """True when NumPy is connected: arrays are ndarrays. Otherwise they are
    mojolearn Arrays."""
    return py_get(context, "ndarray") is not None


def classical_is_array(value: PythonObject, context: PythonObject) raises -> Bool:
    """True for an exact mojolearn Array."""
    return py_type(value) is context["array_class"]


def classical_address(value: PythonObject) raises -> PythonObject:
    """The data address of an ndarray or a mojolearn Array. An Array's is its
    `_addr` slot, read without running any of its Python properties."""
    var module = py_get(native_loaded_module("sys").modules, "mojolearn._array")
    if module is not None and py_type(value) is module.Array:
        return value._addr
    return sequential_address(value)


def _store_code(typestr: String) raises -> String:
    """The `array.array` code that stores an Array dtype."""
    if typestr == "<f4":
        return "f"
    if typestr == "<f8":
        return "d"
    if typestr == "<i8":
        return "q"
    if typestr == "<i4":
        return "i"
    if typestr == "<u4":
        return "I"
    if typestr == "<u1":
        return "B"
    if typestr == "<i1":
        return "b"
    if typestr == "<f2" or typestr == "<u2":
        return "H"
    raise Error("TypeError: " + typestr + " is not an Array dtype")


def classical_wrap_array(store: PythonObject, shape: PythonObject, typestr: String, order: String,
                         context: PythonObject) raises -> PythonObject:
    """A mojolearn Array owning `store`, an `array.array` of the dtype's store
    code laid out in `order`. Assembled from its slots, so no Python runs."""
    var b = native_loaded_module("builtins")
    var item = Int(py=store.itemsize)
    var rank = len(shape)
    var size = 1
    for k in range(rank):  # driver-loop: over array rank metadata
        size *= Int(py=shape[k])
    if size != len(store):
        raise Error("RuntimeError: an array store does not match its shape")
    var steps = List[Int](length=rank, fill=0)
    var step = item
    for k in range(rank):  # driver-loop: over array rank metadata
        var axis = rank - 1 - k if order == "C" else k
        steps[axis] = step
        step *= Int(py=shape[axis])
    var strides = Python.list()
    for k in range(rank):  # driver-loop: over array rank metadata
        py_call(strides.append, steps[k])
    var array = native_storage_allocate(context["array_class"])
    py_setattr(array, "_store", store)
    py_setattr(array, "_base", PythonObject(None))
    py_setattr(array, "_pin", PythonObject(None))
    py_setattr(array, "_mv", native_buffer_view(store))
    py_setattr(array, "_addr", py_call(store.buffer_info)[0])
    py_setattr(array, "_readonly", PythonObject(False))
    py_setattr(array, "shape", py_call(py_type(Python.tuple()), shape))
    py_setattr(array, "dtype", PythonObject(typestr))
    py_setattr(array, "ndim", PythonObject(rank))
    py_setattr(array, "size", PythonObject(size))
    py_setattr(array, "nbytes", PythonObject(size * item))
    py_setattr(array, "itemsize", PythonObject(item))
    py_setattr(array, "strides", py_call(py_type(Python.tuple()), strides))
    py_setattr(array, "order", PythonObject(order))
    return array


def classical_typed(shape: PythonObject, typestr: String, context: PythonObject) raises -> PythonObject:
    """A zero-filled C-order array of `shape` holding "<f4", "<f8" or "<i8"."""
    if classical_numpy(context):
        var key = String("float32") if typestr == "<f4" else (String("float64") if typestr == "<f8" else String("int64"))
        return sequential_array(shape, context[PythonObject(key)], context)
    var b = native_loaded_module("builtins")
    var count = 1
    for k in range(len(shape)):  # driver-loop: over array rank metadata
        count *= Int(py=shape[k])
    var width = 4 if typestr == "<f4" else 8
    var store = native_array_zeros(String(_store_code(typestr)), count)
    return classical_wrap_array(store, shape, typestr, "C", context)


def classical_array(rows: Int, cols: Int, context: PythonObject, wide: Bool = False) raises -> PythonObject:
    return classical_typed(Python.tuple(rows) if cols == 0 else Python.tuple(rows, cols),
                           "<f8" if wide else "<f4", context)


def _buffer_typestr(view: PythonObject, name: String) raises -> String:
    """The Array dtype holding a native buffer format's elements."""
    var format = String(view.format)
    var item = Int(py=view.itemsize)
    var code = format
    if format.byte_length() == 2:
        var first = String(format[byte=0:1])
        if first == ">" or first == "!":
            raise Error("TypeError: " + name + " must be in native byte order")
        if first == "@" or first == "=" or first == "<":
            code = String(format[byte=1:2])
    if code == "f" and item == 4:
        return "<f4"
    if code == "d" and item == 8:
        return "<f8"
    if code == "e" and item == 2:
        return "<f2"
    if (code == "?" or code == "B") and item == 1:
        # A bool's bytes are 0 and 1, read as the same unsigned bytes.
        return "<u1"
    if code == "b" and item == 1:
        return "<i1"
    if code == "H" and item == 2:
        return "<u2"
    if code == "i" or code == "l" or code == "q" or code == "n":
        if item == 4:
            return "<i4"
        if item == 8:
            return "<i8"
    if (code == "I" or code == "L") and item == 4:
        return "<u4"
    raise Error("TypeError: " + name + " has buffer format '" + format + "'; without NumPy, mojolearn reads "
                + "bool, int8, uint8, uint16, int32, uint32, int64, float16, float32 and float64 buffers")


def classical_input(value: PythonObject, context: PythonObject, name: String = "input") raises -> PythonObject:
    """An admitted input as an array, before any cast: an ndarray view with
    NumPy, a mojolearn Array without it (the caller's own Array, or one
    C-order copy of a buffer's or a nested sequence's numbers)."""
    if classical_numpy(context):
        return sequential_input(value, context)
    if classical_is_array(value, context):
        return value
    var b = native_loaded_module("builtins")
    if py_type(value) is b.memoryview or py_type(value) is context["array_type"] or py_type(value) is b.bytearray:
        var view = native_buffer_view(value)
        var typestr = _buffer_typestr(view, name)
        # memoryview.tobytes packs any strides in C order, in C.
        var store = native_array_from_bytes(String(_store_code(typestr)), py_call(view.tobytes, "C"))
        return classical_wrap_array(store, view.shape, typestr, "C", context)
    var packed = sequential_sequence(value, context)
    var kind = String(packed[1])
    var size = Int(py=packed[2])
    var typestr = String("<f8")
    if kind == "b":
        typestr = "<i1"
    elif kind == "i":
        typestr = "<i8"
    elif kind != "f":
        raise Error("TypeError: " + name + " holds integers beyond the int64 range")
    if size != (1 if kind == "b" else 8):
        raise Error("TypeError: " + name + " holds unsupported numbers")
    return classical_wrap_array(packed[3], packed[0], typestr, "C", context)


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


def classical_c_bytes(value: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """The array's bytes in C order: a borrowed byte view of a C-ordered
    array, otherwise one copy."""
    var b = native_loaded_module("builtins")
    if classical_is_array(value, context):
        if not classical_c_order(value, context):
            raise Error("TypeError: " + name + " must be a C-ordered Array to be saved without NumPy")
        return b.bytes() if Int(py=value.nbytes) == 0 else py_call(value._mv.cast, "B")
    var view = native_buffer_view(value)
    if Int(py=value.nbytes) == 0:
        return b.bytes()
    return py_call(view.cast, "B") if Bool(py=value.flags.c_contiguous) else py_call(view.tobytes, "C")


def classical_i64(value: PythonObject, index: Int) raises -> Int:
    """Element `index` of a checked int64 array of at most two elements
    (C and Fortran order coincide)."""
    return Int(MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(value)))[index])


def classical_f64(value: PythonObject, index: Int) raises -> Float64:
    """Element `index` of a checked float64 array of at most two elements."""
    return MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(value)))[index]


def classical_host_f32(source: PythonObject, ndim: Int, name: String, context: PythonObject, column_target: Bool = False) raises -> PythonObject:
    classical_admit_value(source, context)
    var b = native_loaded_module("builtins")
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
    var addr = Int(py=classical_address(value))
    var borrowed = not column and classical_is_f32(value, context) and classical_c_order(value, context)
    var result = value if borrowed else classical_array(rows, cols if ndim == 2 else 0, context)
    var out = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(result)))
    classical_cast_rows(addr, s0, s1, rows, cols, item, kind, name, borrowed, out)
    _ = value^
    _ = prepared^
    return result


def classical_word(value: PythonObject, index: Int = 0) raises -> Float64:
    var address = Int(py=classical_address(value))
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
    var input = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(value)))
    var output = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(out)))
    var any_positive = classical_check_weights(input, output, rows, scalar)
    if not any_positive:
        raise Error("Sample weights must contain at least one non-zero number.")
    _ = value^
    return out
