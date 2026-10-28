# SPDX-License-Identifier: Apache-2.0
# Ordinary SciPy containers contribute storage and metadata only. No sparse
# Python method/property, scalar conversion hook or SciPy computation runs.
from std.memory import bitcast
from std.math import isfinite
from bindings.native_loaded_module import native_loaded_module
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.native_buffer_owner import native_buffer_view
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_setattr
from bindings.classical_pandas import _pd_type_fields
from bindings.classical_arrays import classical_address
from bindings.sequential_admission import _one_of
from bindings.host_array_native import host_archive_typed
from bindings.native_numpy_metadata import native_numpy_dtype
from core.cast_in import cast_in_kind, cast_f32_bits, cast_w64_bits
from core.classical_pandas_impl import _pd_double
from core.scaler_fit_impl import scaler_cast
from core.host_sparse_impl import host_sparse_place, host_sparse_compressed, host_sparse_coo, host_sparse_dia, host_sparse_uint64_bound


def _sparse_class(module_name: String, name: String) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var sys = native_loaded_module("sys")
    var module = py_get(sys.modules, module_name)
    if module is None or py_type(module) is not py_type(sys):
        return PythonObject(None)
    var cls = py_get(py_fields(module), name)
    if py_type(cls) is b.type:
        return cls
    # SciPy's ordinary sparse classes use ABCMeta, unlike pandas' classes.
    # Match its exact loaded identity; never call its __instancecheck__,
    # __subclasscheck__, attribute hooks or class construction methods.
    var abc_module = py_get(sys.modules, "abc")
    if abc_module is not None and py_type(abc_module) is py_type(sys):
        var abc_meta = py_get(py_fields(abc_module), "ABCMeta")
        if py_type(abc_meta) is b.type and py_type(cls) is abc_meta:
            return cls
    return PythonObject(None)


def host_sparse_format(value: PythonObject) raises -> String:
    var cls = py_type(value)
    var formats = Python.tuple("csr", "csc", "coo", "bsr", "dia", "dok", "lil")
    for i in range(len(formats)):  # driver-loop: seven supported native sparse storage schemas
        var name = String(formats[i])
        if cls is _sparse_class("scipy.sparse._" + name, name + "_matrix") or cls is _sparse_class("scipy.sparse._" + name, name + "_array"):
            return name
    return ""


def _sparse_fields(value: PythonObject) raises -> PythonObject:
    # The caller has matched an exact supported SciPy class. Explicit native
    # type descriptors read its ABCMeta-owned metadata without dispatching any
    # metaclass or instance attribute hook; only a native dictionary is used.
    var b = native_loaded_module("builtins")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, py_type(value))
    for i in range(len(mro)):  # driver-loop: admitted sparse class dictionary descriptors
        var descriptor = py_get(_pd_type_fields(mro[i]), "__dict__")
        if descriptor is not None and py_type(descriptor) is py_type(b.memoryview.obj):
            var fields = py_call(descriptor.__get__, value)
            if py_type(fields) is not b.dict:
                raise Error("TypeError: sparse storage requires its native instance dictionary")
            var keys = py_call(b.list, fields.keys())
            for j in range(len(keys)):  # driver-loop: stored sparse field-name admission before lookup
                if py_type(keys[j]) is not b.str:
                    raise Error("TypeError: sparse stored field names must be exact native strings")
            return fields
    raise Error("TypeError: sparse storage has no native dictionary descriptor")


def _sparse_aliases(context: PythonObject) raises -> PythonObject:
    var aliases = py_get(context, "host_sparse_integer_aliases")
    if aliases is None:
        var dtype = native_numpy_dtype(context)
        # NumPy exposes long long separately from its platform int64 dtype.
        # Both are existing SciPy storage types, with the same native words.
        aliases = Python.tuple(py_call(dtype, "q"), py_call(dtype, "Q"))
        py_setitem(context, "host_sparse_integer_aliases", aliases)
    return aliases


def _sparse_dtype_known(dtype: PythonObject, context: PythonObject) raises -> Bool:
    var types = py_call(native_loaded_module("builtins").list, context["numeric_dtypes"].values())
    var cls = py_type(dtype)
    var known = False
    for i in range(len(types)):  # driver-loop: native numeric dtype identities before descriptor reads
        known = known or cls is py_type(types[i])
    var aliases = _sparse_aliases(context)
    return known or cls is py_type(aliases[0]) or cls is py_type(aliases[1])


def _sparse_dtype(dtype: PythonObject, context: PythonObject) raises -> Tuple[Int, Int, Bool]:
    if not _sparse_dtype_known(dtype, context):
        raise Error("TypeError: sparse input requires a native numeric dtype")
    var width = Int(py=dtype.itemsize)
    var kind = cast_in_kind(Int(py=dtype.num), width, "sparse data")
    if kind == 2 and width == 2:
        raise Error("sparse float16 storage is not supported by the sparse input interface")
    return (kind, width, Bool(py=dtype.isnative))


struct _SparseArraySnapshot(Copyable, Movable):
    var keep: PythonObject
    var owner: PythonObject
    var dtype: PythonObject
    var shape: List[Int]
    var strides: List[Int]
    var count: Int
    var address: Int
    var width: Int
    var kind: Int
    var native: Bool

    def __init__(out self):
        self.keep = PythonObject(None)
        self.owner = PythonObject(None)
        self.dtype = PythonObject(None)
        self.shape = List[Int]()
        self.strides = List[Int]()
        self.count = 0
        self.address = 0
        self.width = 0
        self.kind = 0
        self.native = True


def _sparse_array(value: PythonObject, rank: Int, context: PythonObject,
                  role: Int = 0) raises -> _SparseArraySnapshot:
    # role: 0 numeric data, 1 signed indices, 2 LIL object-row references.
    if py_type(value) is not context["ndarray"]:
        raise Error("TypeError: sparse storage requires an ordinary ndarray of the declared rank")
    var captured = _SparseArraySnapshot()
    captured.keep = value
    # One exporter capture freezes layout and pins the original ndarray's
    # storage. The temporary public Py_buffer below is exported from that
    # native memoryview, never by rereading the original ndarray's metadata.
    var lease = native_buffer_view(value)
    var view = List[Int](length=10, fill=0)
    var pointer = view.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    ref cpy = Python().cpython()
    var status = cpy.lib.get_function[Int32]("PyObject_GetBuffer")(lease._obj_ptr, pointer, Int32(0x118))
    if status != 0:
        raise cpy.unsafe_get_error()
    try:
        # Public Py_buffer on supported 64-bit little-endian CPython ABIs:
        # buf,obj,len,itemsize, packed readonly/ndim,format,shape,strides,...
        var nd = Int(UInt64(view[4]) >> 32)
        if nd != rank:
            raise Error("TypeError: sparse storage requires an ordinary ndarray of the declared rank")
        # Rank refusal precedes dtype-role validation, as before, using the
        # same pinned export that supplies every later layout word.
        captured.dtype = value.dtype
        if role == 0:
            var info = _sparse_dtype(captured.dtype, context)
            captured.kind = info[0]
            captured.width = info[1]
            captured.native = info[2]
        elif role == 1:
            if not _sparse_dtype_known(captured.dtype, context):
                raise Error("TypeError: sparse indices require signed int32 or int64 buffers")
            captured.width = Int(py=captured.dtype.itemsize)
            if String(captured.dtype.kind) != "i" or (captured.width != 4 and captured.width != 8):
                raise Error("TypeError: sparse indices require signed int32 or int64 buffers")
            captured.kind = 1
            captured.native = Bool(py=captured.dtype.isnative)
        else:
            var object_dtype = py_call(native_numpy_dtype(context), "O")
            if py_type(captured.dtype) is not py_type(object_dtype):
                raise Error("sparse LIL storage requires one native object row per sample")
            captured.width = Int(py=captured.dtype.itemsize)
            captured.kind = -1
            captured.native = Bool(py=captured.dtype.isnative)
        if view[3] != captured.width or view[6] == 0 or view[7] == 0:
            raise Error("TypeError: sparse storage metadata changed during capture")
        var shape = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=view[6])
        var strides = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=view[7])
        captured.count = 1
        for axis in range(rank):  # small-loop(one frozen sparse storage shape and strides)
            captured.shape.append(shape[axis])
            captured.strides.append(strides[axis])
            captured.count *= shape[axis]
        captured.address = view[0]
        captured.owner = Python.tuple(value, lease, captured.dtype)
    finally:
        cpy.lib.get_function[NoneType]("PyBuffer_Release")(pointer)
    return captured^


def _sparse_indices(value: PythonObject, context: PythonObject) raises -> _SparseArraySnapshot:
    return _sparse_array(value, 1, context, 1)


def _sparse_object_at(captured: _SparseArraySnapshot, index: Int) raises -> PythonObject:
    # The exact object dtype and row bounds were admitted before this read.
    # No ndarray indexing can reread mutable shape/stride metadata here.
    var address = captured.address + index * captured.strides[0]
    var item = MutPointer[PyObjectPtr, MutUntrackedOrigin](unsafe_from_address=address)[0]
    return PythonObject(from_borrowed=item) if item else PythonObject(None)


def _sparse_scalar_word(value: PythonObject, kind: Int, width: Int, context: PythonObject) raises -> UInt64:
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    var source_kind = 0
    var source_width = 8
    var word = UInt64(0)
    if cls is b.int or cls is b.bool:
        if Bool(py=value < 0):
            source_kind = 1
            word = bitcast[DType.uint64](Int64(py=value))
        elif Bool(py=value > PythonObject(9223372036854775807)):
            # LIL stores uint64 data as exact Python integers. Its high bit
            # must survive preparation: converting through a signed index
            # raises before duplicate assignment and the final dtype guard.
            # int's native byte export reads the full supported word without
            # invoking a Python protocol or changing last-assignment semantics.
            if Int(py=py_call(value.bit_length)) > 64:
                raise Error("OverflowError: int too big to convert")
            var bytes = py_call(value.to_bytes, 8, "little")
            for i in range(8):  # driver-loop: bytes of one admitted native integer storage item
                word |= UInt64(py=bytes[i]) << UInt64(8 * i)
        else:
            word = UInt64(py=value)
    elif cls is b.float:
        source_kind = 2
        word = bitcast[DType.uint64](Float64(py=value))
    elif _one_of(cls, context["scalar_types"]) or cls is _sparse_aliases(context)[0].type or cls is _sparse_aliases(context)[1].type:
        var dtype = value.dtype
        source_width = Int(py=dtype.itemsize)
        source_kind = cast_in_kind(Int(py=dtype.num), source_width, "sparse scalar")
        var view = native_buffer_view(value)
        var bytes = py_call(view.tobytes)
        var native = Bool(py=dtype.isnative)
        for i in range(source_width):  # driver-loop: bytes of one native scalar storage item
            word |= UInt64(py=bytes[i if native else source_width - i - 1]) << UInt64(8 * i)
    else:
        raise Error("TypeError: sparse values require native numeric scalars without Python conversion hooks")
    if kind == source_kind and width == source_width:
        return word
    if kind == 2:
        return UInt64(cast_f32_bits(word, source_kind, source_width)) if width == 4 else bitcast[DType.uint64](_pd_double(word, source_kind, source_width))
    if kind == 3:
        return UInt64(_pd_double(word, source_kind, source_width) != 0)
    if source_kind == 2:
        var number = _pd_double(word, source_kind, source_width)
        if not isfinite(number):
            raise Error("sparse integer storage cannot contain a nonfinite scalar")
        if kind == 1:
            if number < -9223372036854775808.0 or number >= 9223372036854775808.0:
                raise Error("sparse integer value exceeds int64")
            return bitcast[DType.uint64](Int64(number))
        if number < 0 or number >= 18446744073709551616.0:
            raise Error("sparse integer value exceeds uint64")
        return UInt64(number)
    return cast_w64_bits(word, source_kind, source_width)


def _sparse_coordinate(value: PythonObject, limit: Int, context: PythonObject) raises -> Int:
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if cls is not b.int and not _one_of(cls, context["integer_types"]) and cls is not _sparse_aliases(context)[0].type and cls is not _sparse_aliases(context)[1].type:
        raise Error("TypeError: sparse coordinates must be native integers")
    var index = Int(bitcast[DType.int64](_sparse_scalar_word(value, 1, 8, context)))
    if index < 0 or index >= limit:
        raise Error("sparse coordinate exceeds its declared shape")
    return index


def host_sparse_dense(source: PythonObject, format: String, context: PythonObject) raises -> PythonObject:
    if format.byte_length() == 0 or host_sparse_format(source) != format:
        raise Error("TypeError: sparse storage requires an exact supported SciPy container")
    var b = native_loaded_module("builtins")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, py_type(source))
    for i in range(len(mro)):  # driver-loop: refuse sparse finalizers before acquiring storage owners
        if py_get(_pd_type_fields(mro[i]), "__del__") is not None:
            raise Error("TypeError: sparse storage cannot use Python finalizers")
    var fields = py_call(py_type(Python.dict()).__dict__["copy"], _sparse_fields(source))
    var shape = fields["_shape"]
    if py_type(shape) is not b.tuple or len(shape) != 2 or py_type(shape[0]) is not b.int or py_type(shape[1]) is not b.int:
        raise Error("sparse input must have a native two-dimensional shape")
    var rows = Int(py=shape[0])
    var columns = Int(py=shape[1])
    if rows < 1 or columns < 1:
        raise Error("X is empty")
    if rows > 2147483647 // columns:
        raise Error("X exceeds the native Int32 indexing bound")
    var owners = Python.list(source, shape)
    var objects = format == "dok" or format == "lil"
    var data = _SparseArraySnapshot()
    var kind = 0
    var width = 0
    var native = True
    if objects:
        var dtype = fields["dtype"]
        var info = _sparse_dtype(dtype, context)
        kind = info[0]
        width = info[1]
        native = info[2]
        _ = py_call(owners.append, dtype)
    else:
        data = _sparse_array(fields["data"], 3 if format == "bsr" else (2 if format == "dia" else 1), context)
        kind = data.kind
        width = data.width
        native = data.native
        _ = py_call(owners.append, data.owner)
    if (format == "bsr" or format == "dia") and not native:
        raise Error("sparse BSR and DIA data requires native byte order")
    var store = host_archive_typed(Python.tuple(rows * columns * width), "<u1", context["output_context"])
    var destination = Int(py=classical_address(store))
    if format == "csr" or format == "csc" or format == "bsr":
        var pointers = _sparse_indices(fields["indptr"], context)
        var indices = _sparse_indices(fields["indices"], context)
        _ = py_call(owners.append, pointers.owner)
        _ = py_call(owners.append, indices.owner)
        var br = data.shape[1] if format == "bsr" else 1
        var bc = data.shape[2] if format == "bsr" else 1
        if br < 1 or bc < 1 or rows % br != 0 or columns % bc != 0:
            raise Error("sparse blocks must evenly cover the declared dimensions")
        if pointers.count != (columns if format == "csc" else rows // br) + 1:
            raise Error("sparse pointer count differs from its shape")
        host_sparse_compressed(destination, rows, columns, format == "csc",
            pointers.address, pointers.strides[0], pointers.width, pointers.native,
            indices.address, indices.strides[0], indices.width, indices.native,
            data.address, data.strides[0], data.strides[1] if format == "bsr" else 0,
            data.strides[2] if format == "bsr" else 0, min(indices.count, data.shape[0]),
            width, kind, native, br, bc)
        _ = pointers^
        _ = indices^
    elif format == "coo":
        var coords = fields["coords"]
        if py_type(coords) is not b.tuple or len(coords) != 2:
            raise Error("sparse coordinates must contain two native index arrays")
        var row = _sparse_indices(coords[0], context)
        var col = _sparse_indices(coords[1], context)
        _ = py_call(owners.append, Python.tuple(coords, row.owner, col.owner))
        if row.count != data.count or col.count != data.count:
            raise Error("sparse coordinate counts differ from stored data")
        host_sparse_coo(destination, rows, columns,
            row.address, row.strides[0], row.width, row.native,
            col.address, col.strides[0], col.width, col.native,
            data.address, data.strides[0], data.count, width, kind, native)
        _ = row^
        _ = col^
        _ = coords^
    elif format == "dia":
        var offsets = _sparse_indices(fields["offsets"], context)
        _ = py_call(owners.append, offsets.owner)
        if offsets.count != data.shape[0]:
            raise Error("sparse diagonal count differs from stored data")
        host_sparse_dia(destination, rows, columns,
            offsets.address, offsets.strides[0], offsets.width, offsets.native,
            data.address, data.strides[0], data.strides[1], data.shape[0], data.shape[1], width, kind, native)
        _ = offsets^
    elif format == "dok":
        var dictionary = fields["_dict"]
        if py_type(dictionary) is not b.dict:
            raise Error("TypeError: sparse DOK storage requires a native dictionary")
        var pairs = py_call(b.list, dictionary.items())
        _ = py_call(owners.append, Python.tuple(dictionary, pairs))
        for i in range(len(pairs)):  # driver-loop: native dictionary's sparse coordinate records
            var key = pairs[i][0]
            if py_type(key) is not b.tuple or len(key) != 2:
                raise Error("TypeError: sparse DOK coordinates must be native pairs")
            var row = _sparse_coordinate(key[0], rows, context)
            var col = _sparse_coordinate(key[1], columns, context)
            var word = _sparse_scalar_word(pairs[i][1], kind, width, context)
            host_sparse_place(destination + (row * columns + col) * width, word, width, kind, True)
        _ = pairs^
        _ = dictionary^
    elif format == "lil":
        var indices = _sparse_array(fields["rows"], 1, context, 2)
        var values = _sparse_array(fields["data"], 1, context, 2)
        _ = py_call(owners.append, Python.tuple(indices.owner, values.owner))
        if indices.count != rows or values.count != rows:
            raise Error("sparse LIL storage requires one native object row per sample")
        for row in range(rows):  # driver-loop: captured sparse object-row storage
            var row_indices = _sparse_object_at(indices, row)
            var row_values = _sparse_object_at(values, row)
            if py_type(row_indices) is not b.list or py_type(row_values) is not b.list:
                raise Error("TypeError: sparse LIL rows require matching native lists")
            var saved_indices = py_call(b.list.__dict__["copy"], row_indices)
            var saved_values = py_call(b.list.__dict__["copy"], row_values)
            var entries = len(saved_indices)
            if entries != len(saved_values):
                raise Error("TypeError: sparse LIL rows require matching native lists")
            _ = py_call(owners.append, Python.tuple(row_indices, row_values, saved_indices, saved_values))
            for entry in range(entries):  # driver-loop: captured native sparse row coordinate records
                var col = _sparse_coordinate(saved_indices[entry], columns, context)
                var word = _sparse_scalar_word(saved_values[entry], kind, width, context)
                # LIL assigns each item; repeated columns keep the last word.
                host_sparse_place(destination + (row * columns + col) * width, word, width, kind, False)
        _ = indices^
        _ = values^
    else:
        raise Error("TypeError: unknown native sparse storage schema")
    if kind == 0 and width == 8:
        host_sparse_uint64_bound(destination, rows, columns, columns * width, width, True)
    var result = host_archive_typed(shape, "<f4", context["output_context"])
    scaler_cast(destination, columns * width, width, rows, columns, width, kind, True,
        MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(result))))
    # The dense Array's native owner slot carries original exporters through
    # downstream conversion/execution even if callers replace sparse fields.
    var previous_pin = result._pin
    if previous_pin is not None:
        _ = py_call(owners.append, previous_pin)
    py_setattr(result, "_pin", py_call(py_type(Python.tuple()), owners))
    _ = store^
    _ = data^
    _ = fields^
    return result
