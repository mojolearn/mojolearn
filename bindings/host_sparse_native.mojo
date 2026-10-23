# SPDX-License-Identifier: Apache-2.0
# Ordinary SciPy containers contribute storage and metadata only. No sparse
# Python method/property, scalar conversion hook or SciPy computation runs.
from std.memory import bitcast
from std.math import isfinite
from std.python import Python, PythonObject
from bindings.native_buffer_owner import native_buffer_view
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields
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
    var b = Python.import_module("builtins")
    var sys = Python.import_module("sys")
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
    var b = Python.import_module("builtins")
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


def _sparse_array(value: PythonObject, rank: Int, context: PythonObject) raises -> PythonObject:
    if py_type(value) is not context["ndarray"] or Int(py=value.ndim) != rank:
        raise Error("TypeError: sparse storage requires an ordinary ndarray of the declared rank")
    return value


def _sparse_indices(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var array = _sparse_array(value, 1, context)
    if String(array.dtype.kind) != "i" or (Int(py=array.itemsize) != 4 and Int(py=array.itemsize) != 8):
        raise Error("TypeError: sparse indices require signed int32 or int64 buffers")
    return array


def _sparse_aliases(context: PythonObject) raises -> PythonObject:
    var aliases = py_get(context, "host_sparse_integer_aliases")
    if aliases is None:
        var dtype = native_numpy_dtype(context)
        # NumPy exposes long long separately from its platform int64 dtype.
        # Both are existing SciPy storage types, with the same native words.
        aliases = Python.tuple(py_call(dtype, "q"), py_call(dtype, "Q"))
        py_setitem(context, "host_sparse_integer_aliases", aliases)
    return aliases


def _sparse_dtype(dtype: PythonObject, context: PythonObject) raises -> Int:
    var types = py_call(Python.import_module("builtins").list, context["numeric_dtypes"].values())
    var known = False
    for i in range(len(types)):  # driver-loop: native numeric dtype identities before descriptor reads
        known = known or py_type(dtype) is py_type(types[i])
    var aliases = _sparse_aliases(context)
    known = known or py_type(dtype) is py_type(aliases[0]) or py_type(dtype) is py_type(aliases[1])
    if not known:
        raise Error("TypeError: sparse input requires a native numeric dtype")
    var width = Int(py=dtype.itemsize)
    var kind = cast_in_kind(Int(py=dtype.num), width, "sparse data")
    if kind == 2 and width == 2:
        raise Error("sparse float16 storage is not supported by the sparse input interface")
    return kind


def _sparse_scalar_word(value: PythonObject, kind: Int, width: Int, context: PythonObject) raises -> UInt64:
    var b = Python.import_module("builtins")
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
        source_width = Int(py=value.dtype.itemsize)
        source_kind = cast_in_kind(Int(py=value.dtype.num), source_width, "sparse scalar")
        var view = native_buffer_view(value)
        var bytes = py_call(view.tobytes)
        var native = Bool(py=value.dtype.isnative)
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
    var b = Python.import_module("builtins")
    if py_type(value) is not b.int and not _one_of(py_type(value), context["integer_types"]) and py_type(value) is not _sparse_aliases(context)[0].type and py_type(value) is not _sparse_aliases(context)[1].type:
        raise Error("TypeError: sparse coordinates must be native integers")
    var index = Int(bitcast[DType.int64](_sparse_scalar_word(value, 1, 8, context)))
    if index < 0 or index >= limit:
        raise Error("sparse coordinate exceeds its declared shape")
    return index


def host_sparse_dense(source: PythonObject, format: String, context: PythonObject) raises -> PythonObject:
    if format.byte_length() == 0 or host_sparse_format(source) != format:
        raise Error("TypeError: sparse storage requires an exact supported SciPy container")
    var b = Python.import_module("builtins")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, py_type(source))
    for i in range(len(mro)):  # driver-loop: refuse sparse finalizers before acquiring storage owners
        if py_get(_pd_type_fields(mro[i]), "__del__") is not None:
            raise Error("TypeError: sparse storage cannot use Python finalizers")
    var fields = _sparse_fields(source)
    var shape = fields["_shape"]
    if py_type(shape) is not b.tuple or len(shape) != 2 or py_type(shape[0]) is not b.int or py_type(shape[1]) is not b.int:
        raise Error("sparse input must have a native two-dimensional shape")
    var rows = Int(py=shape[0])
    var columns = Int(py=shape[1])
    if rows < 1 or columns < 1:
        raise Error("X is empty")
    if rows > 2147483647 // columns:
        raise Error("X exceeds the native Int32 indexing bound")
    var objects = format == "dok" or format == "lil"
    var data = PythonObject(None)
    var dtype = PythonObject(None)
    if objects:
        dtype = fields["dtype"]
    else:
        data = _sparse_array(fields["data"], 3 if format == "bsr" else (2 if format == "dia" else 1), context)
        dtype = data.dtype
    var kind = _sparse_dtype(dtype, context)
    var width = Int(py=dtype.itemsize)
    if (format == "bsr" or format == "dia") and not Bool(py=dtype.isnative):
        raise Error("sparse BSR and DIA data requires native byte order")
    var store = host_archive_typed(Python.tuple(rows * columns * width), "<u1", context["output_context"])
    var destination = Int(py=classical_address(store))
    if format == "csr" or format == "csc" or format == "bsr":
        var pointers = _sparse_indices(fields["indptr"], context)
        var indices = _sparse_indices(fields["indices"], context)
        var br = Int(py=data.shape[1]) if format == "bsr" else 1
        var bc = Int(py=data.shape[2]) if format == "bsr" else 1
        if br < 1 or bc < 1 or rows % br != 0 or columns % bc != 0:
            raise Error("sparse blocks must evenly cover the declared dimensions")
        if Int(py=pointers.size) != (columns if format == "csc" else rows // br) + 1:
            raise Error("sparse pointer count differs from its shape")
        host_sparse_compressed(destination, rows, columns, format == "csc",
            Int(py=classical_address(pointers)), Int(py=pointers.strides[0]), Int(py=pointers.itemsize), Bool(py=pointers.dtype.isnative),
            Int(py=classical_address(indices)), Int(py=indices.strides[0]), Int(py=indices.itemsize), Bool(py=indices.dtype.isnative),
            Int(py=classical_address(data)), Int(py=data.strides[0]), Int(py=data.strides[1]) if format == "bsr" else 0,
            Int(py=data.strides[2]) if format == "bsr" else 0, min(Int(py=indices.size), Int(py=data.shape[0])),
            width, kind, Bool(py=dtype.isnative), br, bc)
        _ = pointers^
        _ = indices^
    elif format == "coo":
        var coords = fields["coords"]
        if py_type(coords) is not b.tuple or len(coords) != 2:
            raise Error("sparse coordinates must contain two native index arrays")
        var row = _sparse_indices(coords[0], context)
        var col = _sparse_indices(coords[1], context)
        if Int(py=row.size) != Int(py=data.size) or Int(py=col.size) != Int(py=data.size):
            raise Error("sparse coordinate counts differ from stored data")
        host_sparse_coo(destination, rows, columns,
            Int(py=classical_address(row)), Int(py=row.strides[0]), Int(py=row.itemsize), Bool(py=row.dtype.isnative),
            Int(py=classical_address(col)), Int(py=col.strides[0]), Int(py=col.itemsize), Bool(py=col.dtype.isnative),
            Int(py=classical_address(data)), Int(py=data.strides[0]), Int(py=data.size), width, kind, Bool(py=dtype.isnative))
        _ = row^
        _ = col^
        _ = coords^
    elif format == "dia":
        var offsets = _sparse_indices(fields["offsets"], context)
        if Int(py=offsets.size) != Int(py=data.shape[0]):
            raise Error("sparse diagonal count differs from stored data")
        host_sparse_dia(destination, rows, columns,
            Int(py=classical_address(offsets)), Int(py=offsets.strides[0]), Int(py=offsets.itemsize), Bool(py=offsets.dtype.isnative),
            Int(py=classical_address(data)), Int(py=data.strides[0]), Int(py=data.strides[1]), Int(py=data.shape[0]),
            Int(py=data.shape[1]), width, kind, Bool(py=dtype.isnative))
        _ = offsets^
    elif format == "dok":
        var dictionary = fields["_dict"]
        if py_type(dictionary) is not b.dict:
            raise Error("TypeError: sparse DOK storage requires a native dictionary")
        var pairs = py_call(b.list, dictionary.items())
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
        var indices = _sparse_array(fields["rows"], 1, context)
        var values = _sparse_array(fields["data"], 1, context)
        if Int(py=indices.size) != rows or Int(py=values.size) != rows or String(indices.dtype.kind) != "O" or String(values.dtype.kind) != "O":
            raise Error("sparse LIL storage requires one native object row per sample")
        for row in range(rows):  # driver-loop: native sparse row-list storage
            var row_indices = indices[row]
            var row_values = values[row]
            if py_type(row_indices) is not b.list or py_type(row_values) is not b.list or len(row_indices) != len(row_values):
                raise Error("TypeError: sparse LIL rows require matching native lists")
            for entry in range(len(row_indices)):  # driver-loop: native sparse row coordinate records
                var col = _sparse_coordinate(row_indices[entry], columns, context)
                var word = _sparse_scalar_word(row_values[entry], kind, width, context)
                # LIL's dense representation assigns each item; a repeated
                # column therefore keeps its last stored value rather than sums.
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
    _ = store^
    _ = data^
    _ = fields^
    return result
