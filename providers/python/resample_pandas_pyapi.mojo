# SPDX-License-Identifier: Apache-2.0
# Pandas storage admission for the dtype-preserving resample row owner.
# This module never invokes pandas __array__, to_numpy, extension methods or
# object-cell protocols. Numeric conversion is typed Mojo; object cells move
# by owned CPython references. The caller supplies the shared classical context.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.memory import bitcast, memcpy
from bindings.native_buffer_owner import native_buffer_view
from bindings.arena_materialize_native import _address
from bindings.py_attr import py_attr
from bindings.py_balanced import py_call, py_type, py_get, py_fields
from bindings.native_loaded_module import native_loaded_module
from bindings.native_numpy_metadata import native_numpy_dtype, native_numpy_type
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_pandas import native_pandas_exact, _pd_class, _pd_fields, _pd_get, _pd_locations, _pd_type_fields
from bindings.msel_inputs import _pandas_empty_rows
from bindings.sequential_arrays import sequential_address
from bindings.sequential_inputs import _promote
from core.classical_pandas_impl import pd_raw_at, _pd_double
from core.cast_in import cast_in_kind, cast_w64_bits
from core.sequence_array_cast import sequence_store_word


def _rp_item(module: String, name: String) raises -> PythonObject:
    var sys = native_loaded_module("sys")
    var loaded = py_get(sys.modules, module)
    if loaded is None or py_type(loaded) is not py_type(sys):
        raise Error("TypeError: pandas native metadata requires its already-loaded module " + module)
    var fields = py_fields(loaded)
    _rp_string_keys(fields)
    return py_get(fields, name)


def _rp_string_keys(fields: PythonObject) raises:
    if py_type(fields) is not py_type(Python.dict()):
        raise Error("TypeError: pandas metadata requires an exact native dictionary")
    var keys = py_call(py_type(Python.list()), fields)
    for i in range(len(keys)):  # driver-loop: prevent custom-key equality hooks during native metadata lookup
        if py_type(keys[i]) is not py_type(PythonObject("")):
            raise Error("TypeError: pandas metadata requires exact native string keys")


def _rp_dict(value: PythonObject) raises -> PythonObject:
    # Used only after exact stored-class admission. GenericGetDict reads the
    # actual instance dictionary, never a replacement __dict__ property.
    ref cpy = Python().cpython()
    var raw = cpy.lib.get_function[PyObjectPtr]("PyObject_GenericGetDict")(value._obj_ptr, PyObjectPtr())
    if not raw:
        raise cpy.unsafe_get_error()
    var result = PythonObject(from_owned=raw)
    var keys = py_call(py_type(Python.list()), result)
    for i in range(len(keys)):  # driver-loop: admit dictionary keys before native metadata lookup
        if py_type(keys[i]) is not py_type(PythonObject("")):
            raise Error("TypeError: pandas storage field names must be exact strings")
    return result


struct _RPArray(Movable):
    var owner: PythonObject
    var dtype: PythonObject
    var code: String
    var kind: String
    var item: Int
    var rank: Int
    var rows: Int
    var cols: Int
    var row_stride: Int
    var col_stride: Int
    var address: Int
    var native: Bool

    def __init__(out self, value: PythonObject, context: PythonObject) raises:
        if py_type(value) is not context["ndarray"]:
            raise Error("TypeError: pandas storage requires an exact native ndarray")
        self.owner = value
        self.dtype = value.dtype
        var admitted_dtype = False
        var classes = context["resample_pandas_dtype_classes"]
        for i in range(len(classes)):  # driver-loop: exact builtin dtype descriptors before any dtype metadata read
            admitted_dtype = admitted_dtype or py_type(self.dtype) is classes[i]
        if not admitted_dtype:
            raise Error("TypeError: pandas storage cannot invoke custom dtype metadata or scalar hooks")
        self.kind = String(self.dtype.kind)
        self.item = Int(py=self.dtype.itemsize)
        # No custom dtype getitem/cast, structured-descr Python helpers, or
        # reference-bearing user dtype can enter raw storage copies.
        if self.dtype.fields is not None or self.dtype.subdtype is not None or Int(py=self.dtype.num) > 23:
            raise Error("TypeError: pandas resample storage requires a built-in scalar dtype")
        if self.kind != "O" and Bool(py=self.dtype.hasobject):
            raise Error("TypeError: pandas resample storage has unsupported object-bearing dtype")
        self.code = String(self.dtype.str)
        self.native = Bool(py=self.dtype.isnative)
        self.rank = Int(py=value.ndim)
        if self.rank != 1 and self.rank != 2:
            raise Error("ValueError: pandas block rank must be one or two")
        self.rows = Int(py=value.shape[self.rank - 1])
        self.cols = 1 if self.rank == 1 else Int(py=value.shape[0])
        self.row_stride = Int(py=value.strides[self.rank - 1])
        self.col_stride = 0 if self.rank == 1 else Int(py=value.strides[0])
        self.address = Int(py=sequential_address(value))

    def at(self, row: Int, col: Int = 0) -> Int:
        return self.address + row * self.row_stride + col * self.col_stride


def _rp_new(shape: PythonObject, dtype: PythonObject, context: PythonObject) raises -> PythonObject:
    return py_call(context["ndarray"], shape, dtype=dtype)


def _rp_object(destination: Int, value: PythonObject) raises:
    ref cpy = Python().cpython()
    var slot = MutPointer[PyObjectPtr, MutUntrackedOrigin](unsafe_from_address=destination)
    var previous = slot[0]
    slot[0] = cpy.Py_NewRef(value._obj_ptr)
    # Every destination is a fresh ndarray slot, initially None/null. Never
    # overwrite a caller-owned cell or release its last reference here.
    if previous:
        cpy.Py_DecRef(previous)
    _ = value


def _rp_number(source: _RPArray, address: Int, imaginary: Bool = False) raises -> Float64:
    var kind = source.kind
    var size = source.item
    var offset = 0
    if kind == "c":
        kind = "f"
        size //= 2
        if imaginary:
            offset = size
    elif imaginary:
        return Float64(0)
    if size > 8:
        raise Error("TypeError: mixed pandas numeric conversion requires native widths up to float64")
    var tag = 2 if kind == "f" else (1 if kind == "i" else 0)
    return _pd_double(pd_raw_at(address, offset, size, source.native), tag, size)


def _rp_box(source: _RPArray, address: Int) raises -> PythonObject:
    if source.kind == "O":
        var raw = MutPointer[PyObjectPtr, MutUntrackedOrigin](unsafe_from_address=address)[0]
        return PythonObject(from_borrowed=raw) if raw else PythonObject(None)
    if source.kind == "b":
        return PythonObject(pd_raw_at(address, 0, source.item, source.native) != 0)
    if source.kind == "i" or source.kind == "u":
        var tag = 1 if source.kind == "i" else 0
        var raw = cast_w64_bits(pd_raw_at(address, 0, source.item, source.native), tag, source.item)
        if tag == 1:
            return PythonObject(bitcast[DType.int64](raw))
        ref cpy = Python().cpython()
        var result = cpy.lib.get_function[PyObjectPtr]("PyLong_FromUnsignedLongLong")(raw)
        if not result:
            raise cpy.unsafe_get_error()
        return PythonObject(from_owned=result)
    if source.kind == "f":
        return PythonObject(_rp_number(source, address))
    if source.kind == "c":
        ref cpy = Python().cpython()
        var result = cpy.lib.get_function[PyObjectPtr]("PyComplex_FromDoubles")(_rp_number(source, address), _rp_number(source, address, True))
        if not result:
            raise cpy.unsafe_get_error()
        return PythonObject(from_owned=result)
    if source.kind == "S":
        var count = source.item
        var bytes = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
        while count > 0 and bytes[count - 1] == 0:  # driver-loop: trim fixed-width byte-string padding only
            count -= 1
        ref cpy = Python().cpython()
        var raw = cpy.lib.get_function[PyObjectPtr]("PyBytes_FromStringAndSize")(bytes, count)
        if not raw:
            raise cpy.unsafe_get_error()
        return PythonObject(from_owned=raw)
    if source.kind == "U":
        var count = source.item // 4
        var words = List[UInt32](length=count, fill=0)
        for i in range(count):  # driver-loop: native endian normalization of fixed-width Unicode storage
            words[i] = UInt32(pd_raw_at(address, i * 4, 4, source.native))
        while count > 0 and words[count - 1] == 0:  # driver-loop: trim fixed-width Unicode padding only
            count -= 1
        ref cpy = Python().cpython()
        var raw = cpy.lib.get_function[PyObjectPtr]("PyUnicode_FromKindAndData")(Int32(4), words.unsafe_ptr(), count)
        if not raw:
            raise cpy.unsafe_get_error()
        _ = words^
        return PythonObject(from_owned=raw)
    if source.kind == "M" or source.kind == "m":
        return _rp_time_box(source, address, PythonObject(None))
    raise Error("TypeError: pandas object promotion requires native scalar boxing for this dtype")


def _rp_copy(source: _RPArray, address: Int, destination: Int, target: _RPArray) raises:
    if target.kind == "O":
        _rp_object(destination, _rp_box(source, address))
        return
    if source.code == target.code:
        memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=destination),
               src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address), count=source.item)
        return
    if (target.kind == "M" or target.kind == "m") and source.kind == target.kind:
        var delta = _rp_time_unit(target.code) - _rp_time_unit(source.code)
        if delta < 0:
            raise Error("ValueError: pandas common temporal dtype must retain the finest resolution")
        var word = pd_raw_at(address, 0, 8, source.native)
        if word != UInt64(1) << 63:
            for i in range(delta):  # driver-loop: storage unit scaling to the common finest temporal dtype
                word *= UInt64(1000)
        sequence_store_word(destination, word, 8, target.native)
        return
    if target.kind == "f" or target.kind == "c":
        var components = 2 if target.kind == "c" else 1
        var size = target.item // components
        for component in range(components):  # driver-loop: native ABI scalar conversion, no NumPy conversion calls
            var value = _rp_number(source, address, component == 1)
            var word = UInt64(0)
            if size == 8:
                word = bitcast[DType.uint64](value)
            elif size == 4:
                word = UInt64(bitcast[DType.uint32](Float32(value)))
            elif size == 2:
                word = UInt64(bitcast[DType.uint16](Float16(value)))
            else:
                raise Error("TypeError: mixed pandas floating widths above float64 need native conversion support")
            sequence_store_word(destination + component * size, word, size, target.native)
        return
    if (target.kind == "i" or target.kind == "u" or target.kind == "b") and (source.kind == "i" or source.kind == "u" or source.kind == "b"):
        var word = cast_w64_bits(pd_raw_at(address, 0, source.item, source.native), 1 if source.kind == "i" else 0, source.item)
        sequence_store_word(destination, word, target.item, target.native)
        return
    raise Error("TypeError: pandas blocks require unsupported native dtype promotion")


def _rp_missing_numeric(destination: Int, target: _RPArray) raises:
    var width = target.item // 2 if target.kind == "c" else target.item
    if (target.kind != "f" and target.kind != "c") or (width != 2 and width != 4 and width != 8):
        raise Error("TypeError: pandas missing-value storage requires a native floating component width")
    var nan = UInt64(0x7ff8000000000000) if width == 8 else (UInt64(0x7fc00000) if width == 4 else UInt64(0x7e00))
    sequence_store_word(destination, nan, width, target.native)
    if target.kind == "c":
        sequence_store_word(destination + width, UInt64(0), width, target.native)


def _rp_time_unit(code: String) raises -> Int:
    if code.endswith("[s]"):
        return 7
    if code.endswith("[ms]"):
        return 8
    if code.endswith("[us]"):
        return 9
    if code.endswith("[ns]"):
        return 10
    raise Error("TypeError: pandas temporal storage requires seconds, milliseconds, microseconds or nanoseconds")


def _rp_na() raises -> PythonObject:
    return _rp_item("pandas._libs.missing", "NA")


def _rp_nan() -> PythonObject:
    return PythonObject(bitcast[DType.float64](UInt64(0x7ff8000000000000)))


def _rp_distinguish_na() raises -> Bool:
    var module = py_get(native_loaded_module("sys").modules, "pandas._config.config")
    if module is None:
        return False
    if py_type(module) is not py_type(native_loaded_module("sys")):
        raise Error("TypeError: pandas options require their loaded native module")
    var module_fields = py_fields(module)
    _rp_string_keys(module_fields)
    var config = py_get(module_fields, "_global_config")
    if py_type(config) is not py_type(Python.dict()):
        raise Error("TypeError: pandas options require ordinary native dictionaries")
    _rp_string_keys(config)
    var future = py_get(config, "future")
    if future is None:
        return False
    if py_type(future) is not py_type(Python.dict()):
        raise Error("TypeError: pandas future options require an ordinary native dictionary")
    _rp_string_keys(future)
    var distinguish = py_get(future, "distinguish_nan_and_na")
    if distinguish is None:
        return False
    if py_type(distinguish) is not py_type(PythonObject(True)):
        raise Error("TypeError: pandas missing-value option must be a native bool")
    return Bool(py=distinguish)


def _rp_masked(values: PythonObject, force_object: Bool, distinguish: Bool, context: PythonObject) raises -> PythonObject:
    var fields = _pd_fields(values)
    var source = _RPArray(py_get(fields, "_data"), context)
    var mask = _RPArray(py_get(fields, "_mask"), context)
    var cls = py_type(values)
    if (cls is _pd_class("pandas.core.arrays.integer", "IntegerArray") and source.kind != "i" and source.kind != "u") or (cls is _pd_class("pandas.core.arrays.floating", "FloatingArray") and source.kind != "f") or (cls is _pd_class("pandas.core.arrays.boolean", "BooleanArray") and source.kind != "b"):
        raise Error("TypeError: pandas nullable dtype and native storage disagree")
    if source.rank != 1 or mask.rank != 1 or mask.rows != source.rows or mask.kind != "b":
        raise Error("ValueError: pandas nullable data and mask metadata disagree")
    var missing = False
    for row in range(mask.rows):  # driver-loop: capture missing-value presence from the admitted native mask
        missing = missing or pd_raw_at(mask.at(row), 0, 1, True) != 0
    var objects = force_object or (missing and (source.kind == "b" or distinguish))
    var dtype = PythonObject("O") if objects else (PythonObject("f8") if missing and (source.kind == "i" or source.kind == "u") else source.dtype)
    var result = _rp_new(Python.tuple(source.rows), dtype, context)
    var target = _RPArray(result, context)
    var na = _rp_na()
    for row in range(source.rows):  # driver-loop: native masked storage copy and finished scalar boxing
        var dst = target.at(row)
        if pd_raw_at(mask.at(row), 0, 1, True) != 0:
            if objects:
                _rp_object(dst, na)
            else:
                sequence_store_word(dst, UInt64(0x7ff8000000000000) if target.item == 8 else UInt64(0x7fc00000), target.item, target.native)
        else:
            _rp_copy(source, source.at(row), dst, target)
    _ = values
    return result



def _rp_native_method(cls: PythonObject, name: String) raises -> PythonObject:
    # Admit the raw C classmethod descriptor before binding: looking up a
    # replaceable class attribute first could execute a Python descriptor.
    var b = native_loaded_module("builtins")
    var native_classmethod = py_type(b.dict.__dict__["fromkeys"])
    var ancestry = py_call(b.type.__dict__["__mro__"].__get__, cls)
    for i in range(len(ancestry)):  # driver-loop: native boxing descriptors live on _Timestamp/_Timedelta/_Period bases
        var owner = ancestry[i]
        var descriptor = py_get(_pd_type_fields(owner), name)
        if descriptor is None:
            continue
        if py_type(descriptor) is not native_classmethod or descriptor.__objclass__ is not owner:
            raise Error("TypeError: pandas scalar boxing requires its native classmethod descriptor")
        return py_call(descriptor.__get__, PythonObject(None), cls)
    raise Error("TypeError: pandas scalar boxing native classmethod is unavailable")


def _rp_timezone(tz: PythonObject) raises:
    if tz is None:
        return
    if py_type(tz) is _pd_class("datetime", "timezone") or py_type(tz) is _pd_class("zoneinfo", "ZoneInfo"):
        return
    # Custom/dateutil/pytz tzinfo methods are Python callbacks. Admission must
    # precede Timestamp's native timezone conversion, which calls tzinfo.
    raise Error("TypeError: pandas timezone storage requires datetime.timezone or zoneinfo.ZoneInfo without Python callbacks")


def _rp_time_box(source: _RPArray, address: Int, tz: PythonObject) raises -> PythonObject:
    var value = bitcast[DType.int64](pd_raw_at(address, 0, 8, source.native))
    if value == Int64.MIN:
        return _rp_item("pandas._libs.tslibs.nattype", "NaT")
    var unit = _rp_time_unit(source.code)
    if source.kind == "M":
        return py_call(_rp_native_method(_pd_class("pandas._libs.tslibs.timestamps", "Timestamp"), "_from_value_and_reso"), PythonObject(value), unit, tz)
    return py_call(_rp_native_method(_pd_class("pandas._libs.tslibs.timedeltas", "Timedelta"), "_from_value_and_reso"), PythonObject(value), unit)


def _rp_arrow(values: PythonObject, force_object: Bool, distinguish: Bool, context: PythonObject) raises -> PythonObject:
    var fields = _pd_fields(values)
    var chunked = py_get(fields, "_pa_array")
    if py_type(chunked) is not _pd_class("pyarrow.lib", "ChunkedArray"):
        raise Error("TypeError: pandas Arrow storage requires an exact native ChunkedArray")
    var arrow_type = chunked.type
    var type_class = py_type(arrow_type)
    if type_class is not _pd_class("pyarrow.lib", "DataType"):
        raise Error("TypeError: pandas Arrow storage requires a primitive native type")
    # Arrow's public Type enum; dtype comes from the type even for zero chunks.
    var type_id = Int(py=arrow_type.id)
    var dtype = String()
    var text = type_id == 13 or type_id == 34
    var binary = type_id == 14 or type_id == 35
    var offset_size = 8 if type_id == 34 or type_id == 35 else 4
    if type_id == 1:
        dtype = "b1"
    elif type_id >= 2 and type_id <= 9:
        dtype = ("u" if type_id % 2 == 0 else "i") + String(1 << ((type_id - 2) // 2))
    elif type_id >= 10 and type_id <= 12:
        dtype = "f" + String(1 << (type_id - 9))
    elif text or binary or type_id == 0:
        dtype = "O"
    else:
        raise Error("TypeError: pandas Arrow storage requires native numeric, UTF-8, binary or null buffers")
    var rows = len(chunked)
    var count = Int(py=chunked.num_chunks)
    var nulls = Int(py=chunked.null_count)
    var objects = force_object or text or binary or type_id == 0 or (nulls > 0 and (type_id == 1 or distinguish))
    var output_dtype = "O" if objects else ("f8" if nulls > 0 and type_id >= 2 and type_id <= 9 else dtype)
    var result = _rp_new(Python.tuple(rows), PythonObject(output_dtype), context)
    var target = _RPArray(result, context)
    var na = _rp_na()
    if py_type(values) is _pd_class("pandas.core.arrays.string_arrow", "ArrowStringArray"):
        var string_dtype = py_get(fields, "_dtype")
        if py_type(string_dtype) is not _pd_class("pandas.core.arrays.string_", "StringDtype"):
            raise Error("TypeError: pandas Arrow strings require their exact stored dtype")
        na = py_get(_pd_fields(string_dtype), "_na_value")
        if na is None:
            na = _rp_na()
    var position = 0
    var buffer_type = _pd_class("pyarrow.lib", "Buffer")
    var chunk_names = Python.tuple("NullArray", "BooleanArray", "UInt8Array", "Int8Array", "UInt16Array", "Int16Array", "UInt32Array", "Int32Array", "UInt64Array", "Int64Array", "HalfFloatArray", "FloatArray", "DoubleArray")
    var expected = _pd_class("pyarrow.lib", "LargeStringArray" if offset_size == 8 else "StringArray") if text else (_pd_class("pyarrow.lib", "LargeBinaryArray" if offset_size == 8 else "BinaryArray") if binary else _pd_class("pyarrow.lib", String(chunk_names[type_id])))
    for c in range(count):  # driver-loop: retain each exact Arrow chunk and its native buffers through copying
        var chunk = py_call(chunked.chunk, c)
        if py_type(chunk) is not expected:
            raise Error("TypeError: pandas Arrow chunks disagree with their primitive native type")
        var length = len(chunk)
        var offset = Int(py=chunk.offset)
        var buffers = py_call(chunk.buffers)
        var expected_buffers = 1 if type_id == 0 else (3 if text or binary else 2)
        if py_type(buffers) is not py_type(Python.list()) or len(buffers) != expected_buffers:
            raise Error("TypeError: pandas Arrow buffers disagree with their primitive layout")
        var views = Python.list()
        var addresses = List[Int](length=expected_buffers, fill=0)
        var sizes = List[Int](length=expected_buffers, fill=0)
        for i in range(expected_buffers):  # driver-loop: native buffer admission and owner retention
            if buffers[i] is not None:
                if py_type(buffers[i]) is not buffer_type:
                    raise Error("TypeError: pandas Arrow buffer requires its exact native class")
                var view = native_buffer_view(buffers[i])
                addresses[i] = _address(view)
                sizes[i] = Int(py=view.nbytes)
                py_call(views.append, view)
        if offset < 0 or length < 0 or position > rows - length:
            raise Error("ValueError: pandas Arrow chunk offsets exceed output bounds")
        if addresses[0] != 0 and (offset + length + 7) // 8 > sizes[0]:
            raise Error("ValueError: pandas Arrow validity buffer is too small")
        var numeric = PythonObject(None)
        var numeric_sources = List[_RPArray]()
        if not text and not binary and type_id != 0 and type_id != 1:
            var item = Int(py=py_call(context["resample_pandas_dtype_factory"], dtype).itemsize)
            if (offset + length) > sizes[1] // item:
                raise Error("ValueError: pandas Arrow numeric data buffer is too small")
            # A native ndarray view supplies admitted scalar metadata; data
            # never passes through Arrow/NumPy conversion or compute methods.
            numeric = py_call(context["ndarray"], Python.tuple(length), dtype=PythonObject(dtype), buffer=buffers[1], offset=offset * Int(py=py_call(context["resample_pandas_dtype_factory"], dtype).itemsize))
        if numeric is not None:
            numeric_sources.append(_RPArray(numeric, context))
        for row in range(length):  # driver-loop: unpack Arrow validity and copy/box one stored cell
            var index = offset + row
            var valid = type_id != 0
            if addresses[0] != 0:
                valid = valid and ((MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=addresses[0])[index >> 3] >> UInt8(index & 7)) & UInt8(1)) != 0
            var dst = target.at(position + row)
            if not valid:
                if objects:
                    _rp_object(dst, na)
                else:
                    sequence_store_word(dst, UInt64(0x7ff8000000000000) if target.item == 8 else (UInt64(0x7fc00000) if target.item == 4 else UInt64(0x7e00)), target.item, target.native)
            elif text or binary:
                if (index + 2) * offset_size > sizes[1]:
                    raise Error("ValueError: pandas Arrow string offsets exceed their buffer")
                var first = Int(pd_raw_at(addresses[1], index * offset_size, offset_size, True))
                var last = Int(pd_raw_at(addresses[1], (index + 1) * offset_size, offset_size, True))
                if first < 0 or last < first or last > sizes[2]:
                    raise Error("ValueError: pandas Arrow string span exceeds its native data buffer")
                ref cpy = Python().cpython()
                var data = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=addresses[2] + first)
                var raw = cpy.lib.get_function[PyObjectPtr]("PyUnicode_DecodeUTF8")(data, last - first, Optional[MutPointer[UInt8, MutUntrackedOrigin]](None)) if text else cpy.lib.get_function[PyObjectPtr]("PyBytes_FromStringAndSize")(data, last - first)
                if not raw:
                    raise cpy.unsafe_get_error()
                _rp_object(dst, PythonObject(from_owned=raw))
            elif type_id == 1:
                if (index + 8) // 8 > sizes[1]:
                    raise Error("ValueError: pandas Arrow boolean buffer is too small")
                var bit = (MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=addresses[1])[index >> 3] >> UInt8(index & 7)) & UInt8(1)
                if objects:
                    _rp_object(dst, PythonObject(bit != 0))
                else:
                    sequence_store_word(dst, UInt64(bit), 1, True)
            else:
                ref source = numeric_sources[0]
                _rp_copy(source, source.at(row), dst, target)
        position += length
        _ = numeric^
        _ = views^
        _ = buffers^
        _ = chunk^
    if position != rows:
        raise Error("ValueError: pandas Arrow chunks do not fill the declared row count")
    _ = values
    _ = chunked^
    return result

def _rp_storage(values: PythonObject, force_object: Bool, distinguish: Bool, context: PythonObject, depth: Int = 0) raises -> PythonObject:
    if depth > 32:
        raise Error("TypeError: pandas native extension nesting exceeds the supported depth")
    var cls = py_type(values)
    if cls is context["ndarray"]:
        return values
    if cls is _pd_class("pandas.core.arrays.integer", "IntegerArray") or cls is _pd_class("pandas.core.arrays.floating", "FloatingArray") or cls is _pd_class("pandas.core.arrays.boolean", "BooleanArray"):
        return _rp_masked(values, force_object, distinguish, context)
    if cls is _pd_class("pandas.core.arrays.string_", "StringArray") or cls is _pd_class("pandas.core.arrays.numpy_", "NumpyExtensionArray"):
        return _pd_get("pandas._libs.arrays", "NDArrayBacked", "_ndarray", values)
    if cls is _pd_class("pandas.core.arrays.categorical", "Categorical"):
        var codes = _RPArray(_pd_get("pandas._libs.arrays", "NDArrayBacked", "_ndarray", values), context)
        var dtype = _pd_get("pandas._libs.arrays", "NDArrayBacked", "_dtype", values)
        # CategoricalDtype has its own metaclass. Read the known exact class
        # identity without invoking its equality/instancecheck machinery.
        if py_type(dtype) is not _rp_item("pandas.core.dtypes.dtypes", "CategoricalDtype"):
            raise Error("TypeError: pandas categorical requires its exact stored dtype")
        var categories = py_get(_rp_dict(dtype), "_categories")
        var category_class = py_type(categories)
        if category_class is not _pd_class("pandas.core.indexes.range", "RangeIndex") and category_class is not _pd_class("pandas.core.indexes.base", "Index") and category_class is not _pd_class("pandas.core.indexes.datetimes", "DatetimeIndex") and category_class is not _pd_class("pandas.core.indexes.timedeltas", "TimedeltaIndex") and category_class is not _pd_class("pandas.core.indexes.period", "PeriodIndex"):
            raise Error("TypeError: categorical categories require ordinary native index storage")
        var category_values = PythonObject(None)
        if category_class is _pd_class("pandas.core.indexes.range", "RangeIndex"):
            var stored_range = py_get(_pd_fields(categories), "_range")
            if py_type(stored_range) is not py_type(py_call(native_loaded_module("builtins").range, 0)):
                raise Error("TypeError: categorical RangeIndex requires native range storage")
            category_values = _rp_new(Python.tuple(len(stored_range)), PythonObject("i8"), context)
            var range_array = _RPArray(category_values, context)
            for i in range(range_array.rows):  # driver-loop: materialize the native categorical range metadata
                sequence_store_word(range_array.at(i), UInt64(Int(py=stored_range[i])), 8, True)
        else:
            category_values = py_get(_pd_fields(categories), "_data")
        var category_na = _rp_nan()
        var category_type = py_type(category_values)
        if category_type is _pd_class("pandas.core.arrays.datetimes", "DatetimeArray") or category_type is _pd_class("pandas.core.arrays.timedeltas", "TimedeltaArray") or category_type is _pd_class("pandas.core.arrays.period", "PeriodArray"):
            category_na = _rp_item("pandas._libs.tslibs.nattype", "NaT")
        elif category_type is _pd_class("pandas.core.arrays.integer", "IntegerArray") or category_type is _pd_class("pandas.core.arrays.floating", "FloatingArray") or category_type is _pd_class("pandas.core.arrays.boolean", "BooleanArray") or category_type is _pd_class("pandas.core.arrays.arrow.array", "ArrowExtensionArray"):
            category_na = _rp_na()
        elif category_type is _pd_class("pandas.core.arrays.string_", "StringArray") or category_type is _pd_class("pandas.core.arrays.string_arrow", "ArrowStringArray"):
            var string_dtype = _pd_get("pandas._libs.arrays", "NDArrayBacked", "_dtype", category_values) if category_type is _pd_class("pandas.core.arrays.string_", "StringArray") else py_get(_pd_fields(category_values), "_dtype")
            if py_type(string_dtype) is not _pd_class("pandas.core.arrays.string_", "StringDtype"):
                raise Error("TypeError: pandas string categories require their exact native dtype")
            category_na = py_get(_pd_fields(string_dtype), "_na_value")
            if category_na is None:
                category_na = _rp_na()
        var source = _RPArray(_rp_storage(category_values, False, distinguish, context, depth + 1), context)
        if source.kind == "M" or source.kind == "m":
            category_na = _rp_item("pandas._libs.tslibs.nattype", "NaT")
        if source.rank != 1 or codes.rank != 1 or codes.kind != "i":
            raise Error("ValueError: pandas categorical requires one-dimensional values and signed codes")
        var missing = False
        for row in range(codes.rows):  # driver-loop: categorical code validation precedes allocation
            var code = Int(bitcast[DType.int64](cast_w64_bits(pd_raw_at(codes.at(row), 0, codes.item, codes.native), 1, codes.item)))
            if code < -1 or code >= source.rows:
                raise Error("ValueError: pandas categorical code exceeds its category storage")
            missing = missing or code == -1
        var output_dtype = source.dtype
        if force_object or (missing and source.kind == "b"):
            output_dtype = PythonObject("O")
        elif missing and (source.kind == "i" or source.kind == "u"):
            output_dtype = PythonObject("f8")
        var result = _rp_new(Python.tuple(codes.rows), output_dtype, context)
        var target = _RPArray(result, context)
        for row in range(codes.rows):  # driver-loop: categorical gather preserves category object identity
            var code = Int(bitcast[DType.int64](cast_w64_bits(pd_raw_at(codes.at(row), 0, codes.item, codes.native), 1, codes.item)))
            if code >= 0:
                _rp_copy(source, source.at(code), target.at(row), target)
            elif target.kind == "O":
                _rp_object(target.at(row), category_na)
            elif target.kind == "M" or target.kind == "m":
                sequence_store_word(target.at(row), UInt64(1) << 63, 8, target.native)
            else:
                _rp_missing_numeric(target.at(row), target)
        _ = source.owner
        _ = values
        return result
    if cls is _pd_class("pandas.core.arrays.datetimes", "DatetimeArray") or cls is _pd_class("pandas.core.arrays.timedeltas", "TimedeltaArray"):
        var dtype = _pd_get("pandas._libs.arrays", "NDArrayBacked", "_dtype", values)
        var data = _pd_get("pandas._libs.arrays", "NDArrayBacked", "_ndarray", values)
        if py_type(dtype) is _pd_class("pandas.core.dtypes.dtypes", "DatetimeTZDtype"):
            var tz = py_get(_pd_fields(dtype), "_tz")
            _rp_timezone(tz)
            var source = _RPArray(data, context)
            if source.kind != "M" or source.item != 8:
                raise Error("TypeError: pandas timezone storage requires native datetime64 words")
            var result = _rp_new(data.shape, PythonObject("O"), context)
            var target = _RPArray(result, context)
            for col in range(source.cols):  # driver-loop: native timezone-preserving Timestamp storage boxing
                for row in range(source.rows):  # driver-loop: one stored temporal value per object slot
                    _rp_object(target.at(row, col), _rp_time_box(source, source.at(row, col), tz))
            _ = values
            return result
        return data
    if cls is _pd_class("pandas.core.arrays.period", "PeriodArray"):
        var source = _RPArray(_pd_get("pandas._libs.arrays", "NDArrayBacked", "_ndarray", values), context)
        if source.kind != "i" or source.item != 8:
            raise Error("TypeError: pandas period storage requires native signed int64 ordinals")
        var dtype = _pd_get("pandas._libs.arrays", "NDArrayBacked", "_dtype", values)
        if py_type(dtype) is not _pd_class("pandas.core.dtypes.dtypes", "PeriodDtype"):
            raise Error("TypeError: pandas period storage requires its exact dtype")
        var freq = py_get(_pd_fields(dtype), "_freq")
        var offsets = Python.tuple("Day", "MonthEnd", "QuarterEnd", "YearEnd", "Week", "Hour", "Minute", "Second", "Milli", "Micro", "Nano", "BusinessDay")
        var admitted = False
        for k in range(len(offsets)):  # driver-loop: exact native period frequencies, never user offset subclasses
            admitted = admitted or py_type(freq) is _pd_class("pandas._libs.tslibs.offsets", String(offsets[k]))
        if not admitted:
            raise Error("TypeError: pandas period requires an ordinary native frequency")
        var cls_period = _pd_class("pandas._libs.tslibs.period", "Period")
        var entry = _rp_native_method(cls_period, "_from_ordinal")
        var result = _rp_new(source.owner.shape, PythonObject("O"), context)
        var target = _RPArray(result, context)
        for col in range(source.cols):  # driver-loop: native period metadata boxing
            for row in range(source.rows):  # driver-loop: preserve stored ordinals and NaT
                var ordinal = bitcast[DType.int64](pd_raw_at(source.at(row, col), 0, 8, source.native))
                var boxed = _rp_item("pandas._libs.tslibs.nattype", "NaT") if ordinal == Int64.MIN else py_call(entry, PythonObject(ordinal), freq)
                _rp_object(target.at(row, col), boxed)
        _ = values
        return result
    if cls is _pd_class("pandas.core.arrays.arrow.array", "ArrowExtensionArray") or cls is _pd_class("pandas.core.arrays.string_arrow", "ArrowStringArray"):
        return _rp_arrow(values, force_object, distinguish, context)
    raise Error("TypeError: pandas resample extension requires a native storage reader; Python conversion hooks cannot execute")


def _rp_blocks(value: PythonObject, series: Bool) raises -> PythonObject:
    # Same manager/descriptor admission as msel_inputs._pandas_blocks, extended
    # to the ordinary datetime blocks that the numeric-only reader excludes.
    var manager = py_get(_pd_fields(value), "_mgr")
    var blocks = PythonObject(None)
    if series:
        if py_type(manager) is not _pd_class("pandas.core.internals.managers", "SingleBlockManager"):
            raise Error("TypeError: resample requires ordinary pandas Series storage")
        blocks = py_get(_pd_fields(manager), "blocks")
    else:
        if py_type(manager) is not _pd_class("pandas.core.internals.managers", "BlockManager"):
            raise Error("TypeError: resample requires ordinary pandas DataFrame storage")
        blocks = _pd_get("pandas._libs.internals", "BlockManager", "blocks", manager)
    if py_type(blocks) is not py_type(Python.tuple()) or (series and len(blocks) != 1):
        raise Error("ValueError: pandas block metadata is invalid")
    for i in range(len(blocks)):  # driver-loop: admit exact native block classes before descriptor reads
        var cls = py_type(blocks[i])
        if cls is not _pd_class("pandas.core.internals.blocks", "NumpyBlock") and cls is not _pd_class("pandas.core.internals.blocks", "NumericBlock") and cls is not _pd_class("pandas.core.internals.blocks", "ObjectBlock") and cls is not _pd_class("pandas.core.internals.blocks", "ExtensionBlock") and cls is not _pd_class("pandas.core.internals.blocks", "DatetimeLikeBlock") and cls is not _pd_class("pandas.core.internals.blocks", "DatetimeTZBlock"):
            raise Error("TypeError: pandas resample cannot invoke block subclass hooks")
    return blocks


def _rp_materialize(value: PythonObject, context: PythonObject) raises -> PythonObject:
    if not native_pandas_exact(value):
        raise Error("TypeError: pandas resample requires an exact DataFrame or Series")
    var series = py_type(value) is _pd_class("pandas.core.series", "Series")
    var blocks = _rp_blocks(value, series)
    if len(blocks) == 0:
        return _rp_new(Python.tuple(_pandas_empty_rows(value, context), 0), PythonObject("f8"), context)
    var distinguish = _rp_distinguish_na()
    var owned = List[_RPArray]()
    var placements = Python.list()
    var raw_owners = Python.list()
    var force_object = False
    var all_categorical = len(blocks) > 1
    var categorical_dtype = PythonObject(None)
    var same_categorical_dtype = True
    # A mixed manager's extension common dtype exports as object. Retain the
    # original extension owners as well as the admitted storage snapshots.
    for k in range(len(blocks)):  # driver-loop: capture original block values once per operation
        var raw = _pd_get("pandas._libs.internals", "Block", "values", blocks[k])
        py_call(raw_owners.append, raw)
        var cls = py_type(raw)
        var categorical = cls is _pd_class("pandas.core.arrays.categorical", "Categorical")
        all_categorical = all_categorical and categorical
        if categorical:
            var dtype = _pd_get("pandas._libs.arrays", "NDArrayBacked", "_dtype", raw)
            if categorical_dtype is None:
                categorical_dtype = dtype
            else:
                same_categorical_dtype = same_categorical_dtype and dtype is categorical_dtype
        if len(blocks) > 1 and not categorical and cls is not context["ndarray"] and cls is not _pd_class("pandas.core.arrays.datetimes", "DatetimeArray") and cls is not _pd_class("pandas.core.arrays.timedeltas", "TimedeltaArray"):
            force_object = True
    if all_categorical:
        if not same_categorical_dtype:
            # Distinct categorical dictionaries need common-category dtype
            # comparison. Do not silently change the dtype or invoke pandas'
            # Python equality/hash machinery for arbitrary category objects.
            raise Error("TypeError: pandas resample with distinct categorical dictionaries requires native common-category admission")
        force_object = True
    var rows = -1
    var cols = 0
    var common_kind = String()
    var common_size = 0
    var common_code = String()
    var same = True
    var booleans = False
    var other = False
    var complex_values = False
    for k in range(len(blocks)):  # driver-loop: capture each native array's dtype, shape and signed strides once
        var data = _RPArray(_rp_storage(raw_owners[k], force_object, distinguish, context), context)
        if (rows >= 0 and data.rows != rows) or (series and data.rank != 1):
            raise Error("ValueError: pandas blocks disagree on row dimensions")
        rows = data.rows
        cols += data.cols
        var positions = Python.tuple(0) if series else _pd_locations(blocks[k], context)
        if len(positions) != data.cols:
            raise Error("ValueError: pandas block placement and storage widths disagree")
        py_call(placements.append, positions)
        if k == 0:
            common_code = data.code
        else:
            same = same and data.code == common_code
        booleans = booleans or data.kind == "b"
        other = other or data.kind != "b"
        if data.kind == "O":
            force_object = True
        elif data.kind == "b" or data.kind == "i" or data.kind == "u" or data.kind == "f" or data.kind == "c":
            var numeric_kind = "f" if data.kind == "c" else data.kind
            var numeric_size = data.item // 2 if data.kind == "c" else data.item
            complex_values = complex_values or data.kind == "c"
            if numeric_size <= 8:
                _promote(common_kind, common_size, numeric_kind, numeric_size)
        owned.append(data^)
    force_object = force_object or (booleans and other)
    var dtype = owned[0].dtype
    if force_object:
        dtype = PythonObject("O")
    elif not same:
        var temporal = owned[0].kind == "M" or owned[0].kind == "m"
        var finest = 7
        var numeric = True
        for k in range(len(owned)):  # driver-loop: classify common native dtype without Python promotion hooks
            var kind = owned[k].kind
            temporal = temporal and kind == owned[0].kind
            numeric = numeric and (kind == "b" or kind == "i" or kind == "u" or kind == "f" or kind == "c")
            if kind == "M" or kind == "m":
                finest = max(finest, _rp_time_unit(owned[k].code))
        if temporal:
            var units = Python.tuple("s", "ms", "us", "ns")
            dtype = PythonObject(owned[0].kind + "8[" + String(units[finest - 7]) + "]")
        elif numeric:
            if common_size == 0:
                raise Error("TypeError: pandas numeric promotion requires an implemented native scalar width")
            dtype = PythonObject(("c" if complex_values else common_kind) + String(common_size * (2 if complex_values else 1)))
            for k in range(len(owned)):  # driver-loop: never silently narrow an unimplemented extended precision source
                if (owned[k].kind == "f" and owned[k].item > 8) or (owned[k].kind == "c" and owned[k].item > 16):
                    raise Error("TypeError: mixed pandas extended precision requires native conversion support")
        else:
            dtype = PythonObject("O")
    var result = _rp_new(Python.tuple(rows) if series else Python.tuple(rows, cols), dtype, context)
    # Output is row-major, while source block snapshots are column-major.
    var target = _RPArray(result, context)
    var seen = List[Bool](length=cols, fill=False)
    for k in range(len(owned)):  # driver-loop: scatter complete admitted block storage to its stable output columns
        ref source = owned[k]
        var positions = placements[k]
        for col in range(source.cols):  # driver-loop: validate column placement before native cell copies
            var dest = Int(py=positions[col])
            if dest < 0 or dest >= cols or seen[dest]:
                raise Error("ValueError: pandas block columns overlap or exceed output shape")
            seen[dest] = True
            for row in range(rows):  # driver-loop: native dtype-preserving storage copy, not model computation
                _rp_copy(source, source.at(row, col), target.address + (row * cols + dest) * target.item, target)
    _ = owned^
    _ = raw_owners^
    _ = blocks^
    _ = value
    return result


def resample_pandas_array(value: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    # Snapshot this operation's native factories/classes; the shared bridge
    # context is not mutated with temporary resample metadata.
    var dtype = native_numpy_dtype(context)
    var ndarray = native_numpy_type("ndarray", context)
    var classes = Python.list()
    var codes = Python.tuple("?", "i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "q", "Q", "l", "L", "f2", "f4", "f8", "g", "c8", "c16", "G", "O", "S", "U", "V", "M8[ns]", "m8[ns]")
    for i in range(len(codes)):  # driver-loop: operation-stable admission identities for builtin native dtype classes
        py_call(classes.append, py_type(py_call(dtype, codes[i])))
    var local = Python.dict(ndarray=ndarray, modules=context["modules"],
        resample_pandas_dtype_classes=classes, resample_pandas_dtype_factory=dtype)
    return _rp_materialize(value, local)
