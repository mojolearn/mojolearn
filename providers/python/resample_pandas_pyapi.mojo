# SPDX-License-Identifier: Apache-2.0
# Pandas storage admission for the dtype-preserving resample row owner.
# This module never invokes pandas __array__, to_numpy, extension methods or
# object-cell protocols. Numeric conversion is typed Mojo; object cells move
# by owned CPython references. The caller supplies the shared classical context.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.memory import bitcast, memcpy
from std.sys import size_of
from std.sys.info import CompilationTarget
from bindings.sequence_string_storage import sequence_storage_image, sequence_string_nan_raw
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
from core.cast_in import cast_in_kind, cast_w64_bits, _top_bit
from core.sequence_array_cast import sequence_store_word


comptime _RPScalarBox = def(MutPointer[UInt8, MutUntrackedOrigin], PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr
comptime _RPScalarBytes = def(PyObjectPtr, MutPointer[UInt8, MutUntrackedOrigin]) thin abi("C") -> NoneType


def _rp_api(dtype: PythonObject) raises -> PythonObject:
    var sys = native_loaded_module("sys")
    var module = py_get(sys.modules, "numpy._core._multiarray_umath")
    if module is None:
        module = py_get(sys.modules, "numpy.core._multiarray_umath")
    if module is None or py_type(module) is not py_type(sys):
        raise Error("TypeError: pandas scalar storage requires NumPy's native module")
    var fields = py_fields(module)
    _rp_string_keys(fields)
    var capsule = py_get(fields, "_ARRAY_API")
    ref cpy = Python().cpython()
    var null_name = Optional[MutPointer[UInt8, MutUntrackedOrigin]](None)
    if cpy.lib.get_function[Int32]("PyCapsule_IsValid")(capsule._obj_ptr, null_name) == 0:
        raise Error("TypeError: pandas scalar storage requires NumPy's public API capsule")
    var pointer = cpy.lib.get_function[Optional[MutPointer[UInt8, MutUntrackedOrigin]]]("PyCapsule_GetPointer")(capsule._obj_ptr, null_name)
    if not pointer:
        raise cpy.unsafe_get_error()
    var address = Int(pointer.value())
    var owner = sequence_storage_image(Int(dtype._obj_ptr))
    if sequence_storage_image(address) != owner:
        raise Error("TypeError: pandas scalar storage API has a foreign native owner")
    var entries = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=address)
    # Public stable NumPy C API: PyArray_Scalar=60, ScalarAsCtype=62.
    # These copy/box admitted scalar storage; they do not cast an array or
    # execute dtype arithmetic. Exact builtin dtype admission occurs first.
    for slot in [60, 62]:  # driver-loop: admit the two scalar storage function pointers before calling them
        if sequence_storage_image(entries[slot]) != owner:
            raise Error("TypeError: pandas scalar storage function has a foreign native owner")
    return Python.tuple(module, capsule, address)


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
    var number: Int
    var api_owner: PythonObject
    var scalar_box: _RPScalarBox

    def __init__(out self, value: PythonObject, context: PythonObject) raises:
        if py_type(value) is not context["ndarray"]:
            raise Error("TypeError: pandas storage requires an exact native ndarray")
        var snapshots = context["resample_pandas_array_snapshots"]
        for i in range(len(snapshots)):  # driver-loop: reuse once-captured metadata and the retained independent native ndarray view
            var snapshot = snapshots[i]
            if snapshot[0] is value:
                self.owner = snapshot[1]
                self.dtype = snapshot[2]
                self.code = String(snapshot[3])
                self.kind = String(snapshot[4])
                self.item = Int(py=snapshot[5])
                self.rank = Int(py=snapshot[6])
                self.rows = Int(py=snapshot[7])
                self.cols = Int(py=snapshot[8])
                self.row_stride = Int(py=snapshot[9])
                self.col_stride = Int(py=snapshot[10])
                self.address = Int(py=snapshot[11])
                self.native = Bool(py=snapshot[12])
                self.number = Int(py=snapshot[13])
                self.api_owner = snapshot[14]
                var table = Int(py=self.api_owner[2])
                self.scalar_box = MutPointer[_RPScalarBox, MutUntrackedOrigin](unsafe_from_address=table + 60 * size_of[Int]())[0]
                return
        # Exact ndarray C descriptor; no subtype/finalize callback. The view
        # pins the original storage and owns independent shape/stride metadata.
        self.owner = py_call(context["resample_pandas_view"], value)
        self.dtype = self.owner.dtype
        var admitted_dtype = False
        var classes = context["resample_pandas_dtype_classes"]
        for i in range(len(classes)):  # driver-loop: exact builtin dtype descriptors before any dtype metadata read
            admitted_dtype = admitted_dtype or py_type(self.dtype) is classes[i]
        if not admitted_dtype:
            raise Error("TypeError: pandas storage cannot invoke custom dtype metadata or scalar hooks")
        self.kind = String(self.dtype.kind)
        self.number = Int(py=self.dtype.num)
        self.api_owner = context["resample_pandas_scalar_api"]
        var table = Int(py=self.api_owner[2])
        self.scalar_box = MutPointer[_RPScalarBox, MutUntrackedOrigin](unsafe_from_address=table + 60 * size_of[Int]())[0]
        self.item = Int(py=self.dtype.itemsize)
        # No custom dtype getitem/cast, structured-descr Python helpers, or
        # reference-bearing user dtype can enter raw storage copies.
        if self.dtype.fields is not None or self.dtype.subdtype is not None or (self.number < 0 or self.number > 23):
            raise Error("TypeError: pandas resample storage requires a built-in scalar dtype")
        if self.kind != "O" and Bool(py=self.dtype.hasobject):
            raise Error("TypeError: pandas resample storage has unsupported object-bearing dtype")
        self.code = String(self.dtype.str)
        self.native = Bool(py=self.dtype.isnative)
        self.rank = Int(py=self.owner.ndim)
        var shape = self.owner.shape
        var strides = self.owner.strides
        if self.rank != 1 and self.rank != 2:
            raise Error("ValueError: pandas block rank must be one or two")
        self.rows = Int(py=shape[self.rank - 1])
        self.cols = 1 if self.rank == 1 else Int(py=shape[0])
        self.row_stride = Int(py=strides[self.rank - 1])
        self.col_stride = 0 if self.rank == 1 else Int(py=strides[0])
        self.address = Int(py=sequential_address(self.owner))
        py_call(snapshots.append, Python.tuple(value, self.owner, self.dtype, self.code, self.kind, self.item, self.rank, self.rows, self.cols, self.row_stride, self.col_stride, self.address, self.native, self.number, self.api_owner))

    def at(self, row: Int, col: Int = 0) -> Int:
        return self.address + row * self.row_stride + col * self.col_stride


def _rp_new(shape: PythonObject, dtype: PythonObject, context: PythonObject) raises -> PythonObject:
    return py_call(context["ndarray"], shape, dtype=dtype)


def _rp_object(destination: Int, value: PythonObject) raises:
    ref cpy = Python().cpython()
    var slot = MutPointer[PyObjectPtr, MutUntrackedOrigin](unsafe_from_address=destination)
    var previous = slot[0]
    slot[0] = cpy.Py_NewRef(value._obj_ptr)
    # Destinations belong to fresh output arrays. A sparse fill may already
    # occupy a slot; its original owner stays retained through replacement.
    # Never mutate a caller-owned slot or release its last reference here.
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
    if source.number == 13 or source.number == 16:
        var raw = source.scalar_box(MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address), source.dtype._obj_ptr, source.owner._obj_ptr)
        if not raw:
            raise Python().cpython().unsafe_get_error()
        _ = source.api_owner
        _ = source.owner
        return PythonObject(from_owned=raw)
    if source.kind == "f":
        return PythonObject(_rp_number(source, address))
    if source.kind == "c":
        ref cpy = Python().cpython()
        var result = cpy.lib.get_function[PyObjectPtr]("PyComplex_FromDoubles")(_rp_number(source, address), _rp_number(source, address, True))
        if not result:
            raise cpy.unsafe_get_error()
        return PythonObject(from_owned=result)
    if source.kind == "S" or source.kind == "V":
        var count = source.item
        var bytes = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
        while source.kind == "S" and count > 0 and bytes[count - 1] == 0:  # driver-loop: trim fixed-width byte-string padding only
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



def _rp_scalar_array(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var classes = context["resample_pandas_scalar_classes"]
    var admitted = False
    for i in range(len(classes)):  # driver-loop: scalar storage API cannot receive subclasses or user conversion hooks
        admitted = admitted or py_type(value) is classes[i]
    if not admitted:
        raise Error("TypeError: pandas fill storage requires an exact native NumPy scalar")
    var result = _rp_new(Python.tuple(1), value.dtype, context)
    var target = _RPArray(result, context)
    if target.kind != "b" and target.kind != "i" and target.kind != "u" and target.kind != "f" and target.kind != "c" and target.kind != "M" and target.kind != "m" and target.kind != "S" and target.kind != "U" and target.kind != "V":
        raise Error("TypeError: pandas scalar byte storage requires a native builtin scalar dtype")
    var api = context["resample_pandas_scalar_api"]
    var extract = MutPointer[_RPScalarBytes, MutUntrackedOrigin](unsafe_from_address=Int(py=api[2]) + 62 * size_of[Int]())[0]
    if target.kind == "S" or target.kind == "U" or target.kind == "V":
        var address = Int(0)
        extract(value._obj_ptr, MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(MutPointer(to=address))))
        if target.item > 0:
            if address == 0:
                raise Error("TypeError: native flexible scalar has no stored bytes")
            memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=target.address), src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address), count=target.item)
    else:
        extract(value._obj_ptr, MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=target.address))
    _ = value
    _ = api^
    return result


def _rp_promote(mut kind: String, mut size: Int, incoming: String, width: Int) raises:
    if incoming == "f" and width == 16:
        kind = "f"
        size = 16
    elif kind == "f" and size == 16:
        if incoming != "b" and incoming != "i" and incoming != "u" and incoming != "f":
            raise Error("TypeError: pandas dtype promotion requires native numeric storage")
    else:
        _promote(kind, size, incoming, width)


def _rp_wide_word(destination: Int, low: UInt64, high: UInt64, native: Bool):
    var out = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=destination)
    for byte in range(16):  # driver-loop: store the native extended floating representation without a foreign cast
        var word = low if byte < 8 else high
        out[byte if native else 15 - byte] = UInt8(word >> UInt64((byte % 8) * 8))


def _rp_widen(source: _RPArray, address: Int, destination: Int, imaginary: Bool, native: Bool) raises:
    if imaginary and source.kind != "c":
        _rp_wide_word(destination, 0, 0, native)
        return
    var component = source.item // 2 if source.kind == "c" else source.item
    var src = address + (component if imaginary else 0)
    if component == 16 and (source.kind == "f" or source.kind == "c"):
        var input = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=src)
        var output = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=destination)
        for byte in range(16):  # driver-loop: same-format extended precision is a raw native storage copy
            output[byte] = input[byte if source.native == native else 15 - byte]
        return
    var negative = False
    var exponent = 0
    var significant = UInt64(0)
    var special = 0
    if source.kind == "i" or source.kind == "u" or source.kind == "b":
        var word = cast_w64_bits(pd_raw_at(src, 0, component, source.native), 1 if source.kind == "i" else 0, component)
        negative = source.kind == "i" and (word >> 63) != 0
        var magnitude = (~word + UInt64(1)) if negative else word
        if magnitude != 0:
            exponent = _top_bit(magnitude)
            significant = magnitude << UInt64(63 - exponent)
    else:
        var word = bitcast[DType.uint64](_rp_number(source, address, imaginary))
        negative = (word >> 63) != 0
        var encoded = Int((word >> 52) & UInt64(0x7ff))
        var fraction = word & UInt64(0xfffffffffffff)
        if encoded == 2047:
            special = 1 if fraction == 0 else 2
            significant = UInt64(0x8000000000000000) | (fraction << 11)
            if special == 2:
                significant |= UInt64(0x4000000000000000)
        elif encoded == 0:
            if fraction != 0:
                var top = _top_bit(fraction)
                exponent = top - 1074
                significant = fraction << UInt64(63 - top)
        else:
            exponent = encoded - 1023
            significant = (fraction | UInt64(0x10000000000000)) << 11
    var biased = 0 if significant == 0 else (32767 if special != 0 else exponent + 16383)
    var sign = UInt64(1) if negative else UInt64(0)
    comptime if CompilationTarget.is_x86():
        # SysV x86 long double: explicit integer bit, 15-bit exponent, padding.
        _rp_wide_word(destination, significant, (sign << 15) | UInt64(biased), native)
    else:
        # AArch64 Linux long double: IEEE binary128. Apple's long double is
        # binary64 and never takes this width-16 branch.
        var low = significant << 49
        var high = (sign << 63) | (UInt64(biased) << 48) | ((significant >> 15) & UInt64(0xffffffffffff))
        _rp_wide_word(destination, low, high, native)

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
            if size == 16:
                _rp_widen(source, address, destination + component * size, component == 1, target.native)
                continue
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
        if target.kind == "b":
            word = UInt64(1) if word != 0 else UInt64(0)
        sequence_store_word(destination, word, target.item, target.native)
        return
    raise Error("TypeError: pandas blocks require unsupported native dtype promotion")


def _rp_missing_numeric(destination: Int, target: _RPArray) raises:
    var width = target.item // 2 if target.kind == "c" else target.item
    if width == 16 and (target.kind == "f" or target.kind == "c"):
        comptime if CompilationTarget.is_x86():
            _rp_wide_word(destination, UInt64(0xc000000000000000), UInt64(0x7fff), target.native)
        else:
            _rp_wide_word(destination, UInt64(0), UInt64(0x7fff800000000000), target.native)
        if target.kind == "c":
            _rp_wide_word(destination + 16, UInt64(0), UInt64(0), target.native)
        return
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
    var cls = py_type(tz)
    if cls is _pd_class("datetime", "timezone") or cls is _pd_class("zoneinfo", "ZoneInfo"):
        var method = py_get(_pd_type_fields(cls), "utcoffset")
        var b = native_loaded_module("builtins")
        if py_type(method) is not py_type(b.list.__dict__["copy"]) or method.__objclass__ is not cls:
            raise Error("TypeError: native timezone requires its native UTC-offset descriptor")
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


def _rp_index_values(categories: PythonObject, context: PythonObject) raises -> PythonObject:
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
    return category_values


def _rp_category_record(values: PythonObject, distinguish: Bool, context: PythonObject, depth: Int) raises -> PythonObject:
    var records = context["resample_pandas_category_records"]
    for i in range(len(records)):  # driver-loop: categorical metadata and category buffers are captured once per operation
        if records[i][0] is values:
            return records[i]
    var dtype = _pd_get("pandas._libs.arrays", "NDArrayBacked", "_dtype", values)
    if py_type(dtype) is not _rp_item("pandas.core.dtypes.dtypes", "CategoricalDtype"):
        raise Error("TypeError: pandas categorical requires its exact stored dtype")
    var fields = _rp_dict(dtype)
    var categories = py_get(fields, "_categories")
    var ordered = py_get(fields, "_ordered")
    if ordered is not None and py_type(ordered) is not py_type(PythonObject(True)):
        raise Error("TypeError: pandas categorical ordering metadata must be a native bool")
    var raw = _rp_index_values(categories, context)
    var data = _rp_storage(raw, False, distinguish, context, depth + 1)
    var record = Python.tuple(values, dtype, categories, raw, data, ordered)
    py_call(records.append, record)
    return record




def _rp_category_logical(record: PythonObject, context: PythonObject) raises -> PythonObject:
    if py_type(record[3]) is _pd_class("pandas.core.arrays.sparse.array", "SparseArray"):
        return _rp_sparse_record(record[3], context)[2]
    return record[4]

def _rp_comparison_scalar(source: _RPArray, address: Int) raises -> PythonObject:
    if source.kind == "f":
        return PythonObject(_rp_number(source, address))
    if source.kind == "c":
        ref cpy = Python().cpython()
        var raw = cpy.lib.get_function[PyObjectPtr]("PyComplex_FromDoubles")(_rp_number(source, address), _rp_number(source, address, True))
        if not raw:
            raise cpy.unsafe_get_error()
        return PythonObject(from_owned=raw)
    return _rp_box(source, address)

def _rp_category_cell_equal(left: _RPArray, li: Int, right: _RPArray, ri: Int) raises -> Bool:
    var la = left.at(li)
    var ra = right.at(ri)
    if left.kind == "O" or right.kind == "O":
        # The common exported dtype is object for either object category
        # table regardless of dictionary equality. Never compare its objects.
        return True
    if (left.kind == "M" or left.kind == "m") or (right.kind == "M" or right.kind == "m"):
        if left.kind != right.kind:
            return False
        var lv = bitcast[DType.int64](pd_raw_at(la, 0, 8, left.native))
        var rv = bitcast[DType.int64](pd_raw_at(ra, 0, 8, right.native))
        var delta = _rp_time_unit(left.code) - _rp_time_unit(right.code)
        var value = rv if delta > 0 else lv
        for i in range(abs(delta)):  # driver-loop: ordered temporal category equality compares exact common-resolution instants
            if value > Int64.MAX // 1000 or value < -(Int64.MAX // 1000):
                return False
            value *= 1000
        return lv == value if delta > 0 else value == rv
    if left.kind == "V" or right.kind == "V":
        if left.code != right.code:
            return False
        var lp = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=la)
        var rp = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=ra)
        for byte in range(left.item):  # driver-loop: opaque native category storage equality
            if lp[byte] != rp[byte]:
                return False
        return True
    if (left.kind == "f" and left.item > 8) or (right.kind == "f" and right.item > 8) or (left.kind == "c" and left.item > 16) or (right.kind == "c" and right.item > 16):
        var components = 2 if left.kind == "c" or right.kind == "c" else 1
        for component in range(components):  # driver-loop: ordered numerical categories compare after exact native widening
            var lbytes = Array[UInt8, 16](uninitialized=True)
            var rbytes = Array[UInt8, 16](uninitialized=True)
            _rp_widen(left, la, Int(lbytes.unsafe_ptr()), component == 1, True)
            _rp_widen(right, ra, Int(rbytes.unsafe_ptr()), component == 1, True)
            var equal = True
            var lzero = True
            var rzero = True
            var meaningful = 16
            comptime if CompilationTarget.is_x86():
                meaningful = 10
            for byte in range(meaningful):  # driver-loop: compare meaningful floating bits, ignoring ABI padding and zero sign
                var lv = lbytes[byte]
                var rv = rbytes[byte]
                equal = equal and lv == rv
                if byte == meaningful - 1:
                    lv &= UInt8(0x7f)
                    rv &= UInt8(0x7f)
                lzero = lzero and lv == 0
                rzero = rzero and rv == 0
            if not equal and not (lzero and rzero):
                return False
        return True
    # Both scalars are freshly boxed exact CPython primitives (or strings),
    # so native comparison cannot enter a user-defined equality protocol.
    var a = _rp_comparison_scalar(left, la)
    var b = _rp_comparison_scalar(right, ra)
    ref cpy = Python().cpython()
    var equal = cpy.lib.get_function[Int32]("PyObject_RichCompareBool")(a._obj_ptr, b._obj_ptr, Int32(2))
    if equal < 0:
        raise cpy.unsafe_get_error()
    return equal != 0


def _rp_categories_equal(left: PythonObject, right: PythonObject, distinguish: Bool, context: PythonObject) raises -> Bool:
    var lrecord = _rp_category_record(left, distinguish, context, 0)
    var rrecord = _rp_category_record(right, distinguish, context, 0)
    if lrecord[1] is rrecord[1]:
        return True
    var ordered_left = False if lrecord[5] is None else Bool(py=lrecord[5])
    var ordered_right = False if rrecord[5] is None else Bool(py=rrecord[5])
    if ordered_left != ordered_right:
        return False
    var ldata = _RPArray(lrecord[4], context)
    var rdata = _RPArray(rrecord[4], context)
    if ldata.kind == "O" or rdata.kind == "O":
        return True
    if ldata.rank != 1 or rdata.rank != 1 or ldata.rows != rdata.rows:
        return False
    if not ordered_left:
        var ll = _RPArray(_rp_category_logical(lrecord, context), context)
        var rl = _RPArray(_rp_category_logical(rrecord, context), context)
        if ll.code != rl.code:
            return False
        var ls = py_type(lrecord[3]) is _pd_class("pandas.core.arrays.sparse.array", "SparseArray")
        var rs = py_type(rrecord[3]) is _pd_class("pandas.core.arrays.sparse.array", "SparseArray")
        if ls != rs:
            return False
        if ls and not _rp_sparse_fills_equal(_rp_sparse_record(lrecord[3], context)[5], _rp_sparse_record(rrecord[3], context)[5], context):
            return False
    for i in range(ldata.rows):  # driver-loop: common categorical dtype depends on dictionary values, not object equality hooks
        var found = False
        var first = i if ordered_left else 0
        var last = i + 1 if ordered_left else rdata.rows
        for j in range(first, last):  # driver-loop: unordered native category dictionaries require a bijection
            if _rp_category_cell_equal(ldata, i, rdata, j):
                found = True
                break
        if not found:
            return False
    _ = ldata.owner
    _ = rdata.owner
    return True


def _rp_sparse_record(values: PythonObject, context: PythonObject) raises -> PythonObject:
    var records = context["resample_pandas_sparse_records"]
    for i in range(len(records)):  # driver-loop: one immutable sparse metadata snapshot per public operation
        if records[i][0] is values:
            return records[i]
    var fields = _rp_dict(values)
    var dtype = py_get(fields, "_dtype")
    if py_type(dtype) is not _pd_class("pandas.core.dtypes.dtypes", "SparseDtype"):
        raise Error("TypeError: pandas sparse storage requires its exact stored dtype")
    var dtype_fields = _rp_dict(dtype)
    var fill = py_get(dtype_fields, "_fill_value")
    var index = py_get(fields, "_sparse_index")
    if py_type(index) is not _pd_class("pandas._libs.sparse", "IntIndex") and py_type(index) is not _pd_class("pandas._libs.sparse", "BlockIndex"):
        raise Error("TypeError: pandas sparse placement requires an exact native index")
    var rows = _rp_sparse_field(index, "length")
    var points = _rp_sparse_field(index, "npoints")
    var source = _RPArray(py_get(fields, "_sparse_values"), context)
    var subtype = py_get(dtype_fields, "_dtype")
    if py_type(subtype) is not py_type(source.dtype) or String(subtype.str) != source.code:
        raise Error("TypeError: pandas sparse logical subtype and stored values disagree")
    var record = Python.tuple(values, fields, py_get(fields, "_sparse_values"), dtype, dtype_fields, fill, index, rows, points)
    py_call(records.append, record)
    return record


def _rp_sparse_field(value: PythonObject, name: String) raises -> PythonObject:
    # Caller already admitted exact IntIndex/BlockIndex. Public Cython fields
    # may be inherited from SparseIndex; inspect descriptors without attribute
    # resolution or a sparse index conversion method.
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    var ancestry = py_call(b.type.__dict__["__mro__"].__get__, cls)
    for i in range(len(ancestry)):  # driver-loop: resolve only native field descriptors on the admitted index hierarchy
        var owner = ancestry[i]
        var descriptor = py_get(_pd_type_fields(owner), name)
        if descriptor is None:
            continue
        if (py_type(descriptor) is not py_type(b.memoryview.obj) and py_type(descriptor) is not py_type(b.slice.start)) or descriptor.__objclass__ is not owner:
            raise Error("TypeError: pandas sparse metadata requires its native field descriptor")
        return py_call(descriptor.__get__, value, cls)
    raise Error("TypeError: pandas sparse metadata field is unavailable")


def _rp_sparse_index_word(source: _RPArray, index: Int) raises -> Int:
    if source.rank != 1 or source.kind != "i" or source.item != 4:
        raise Error("TypeError: pandas sparse index requires one-dimensional native int32 storage")
    return Int(bitcast[DType.int64](cast_w64_bits(pd_raw_at(source.at(index), 0, 4, source.native), 1, 4)))



def _rp_temporal_scalar(value: PythonObject) raises -> Bool:
    var cls = py_type(value)
    return cls is _pd_class("pandas._libs.tslibs.timestamps", "Timestamp") or cls is _pd_class("pandas._libs.tslibs.timedeltas", "Timedelta") or cls is _pd_class("datetime", "datetime") or cls is _pd_class("datetime", "date") or cls is _pd_class("datetime", "timedelta")


def _rp_temporal_scalar_array(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var cls = py_type(value)
    var kind = String("M")
    var unit = String("us")
    var word = Int64(0)
    if cls is _pd_class("pandas._libs.tslibs.timestamps", "Timestamp") or cls is _pd_class("pandas._libs.tslibs.timedeltas", "Timedelta"):
        kind = "M" if cls is _pd_class("pandas._libs.tslibs.timestamps", "Timestamp") else "m"
        if kind == "M":
            _rp_timezone(_rp_sparse_field(value, "tzinfo"))
        var reso = Int(py=_rp_sparse_field(value, "_creso"))
        if reso < 7 or reso > 10:
            raise Error("TypeError: native pandas temporal fill has an unsupported resolution")
        var units = Python.tuple("s", "ms", "us", "ns")
        unit = String(units[reso - 7])
        word = Int64(py=_rp_sparse_field(value, "_value"))
    elif cls is _pd_class("datetime", "timedelta"):
        kind = "m"
        var days = Int64(py=_rp_sparse_field(value, "days"))
        var seconds = Int64(py=_rp_sparse_field(value, "seconds"))
        var us = Int64(py=_rp_sparse_field(value, "microseconds"))
        if us == 0:
            unit = "s"
            word = days * 86400 + seconds
        else:
            if days > Int64.MAX // 86400000000 or days < -(Int64.MAX // 86400000000):
                raise Error("OverflowError: native timedelta fill cannot fit microsecond storage")
            word = days * 86400000000 + seconds * 1000000 + us
    else:
        var is_datetime = cls is _pd_class("datetime", "datetime")
        if not is_datetime and cls is not _pd_class("datetime", "date"):
            raise Error("TypeError: native temporal fill requires an exact supported scalar")
        var year = Int64(py=_rp_sparse_field(value, "year"))
        var month = Int64(py=_rp_sparse_field(value, "month"))
        var day = Int64(py=_rp_sparse_field(value, "day"))
        year -= Int64(1 if month <= 2 else 0)
        var era = year // 400
        var yoe = year - era * 400
        var mp = month + Int64(-3 if month > 2 else 9)
        var doy = (153 * mp + 2) // 5 + day - 1
        var doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
        word = (era * 146097 + doe - 719468) * 86400000000
        if is_datetime:
            word += Int64(py=_rp_sparse_field(value, "hour")) * 3600000000
            word += Int64(py=_rp_sparse_field(value, "minute")) * 60000000
            word += Int64(py=_rp_sparse_field(value, "second")) * 1000000
            word += Int64(py=_rp_sparse_field(value, "microsecond"))
            var tz = _rp_sparse_field(value, "tzinfo")
            _rp_timezone(tz)
            if tz is not None:
                var method = py_get(_pd_type_fields(cls), "utcoffset")
                var b = native_loaded_module("builtins")
                if py_type(method) is not py_type(b.list.__dict__["copy"]) or method.__objclass__ is not cls:
                    raise Error("TypeError: datetime fill requires its native UTC-offset method")
                var offset = py_call(method, value)
                if offset is not None:
                    if py_type(offset) is not _pd_class("datetime", "timedelta"):
                        raise Error("TypeError: native timezone offset must be an exact timedelta")
                    word -= Int64(py=_rp_sparse_field(offset, "days")) * 86400000000 + Int64(py=_rp_sparse_field(offset, "seconds")) * 1000000 + Int64(py=_rp_sparse_field(offset, "microseconds"))
    var result = _rp_new(Python.tuple(1), PythonObject(kind + "8[" + unit + "]"), context)
    var target = _RPArray(result, context)
    sequence_store_word(target.address, bitcast[DType.uint64](word), 8, target.native)
    _ = value
    return result

def _rp_sparse_fill_array(value: PythonObject, context: PythonObject) raises -> PythonObject:
    # Produce one native stored scalar without invoking scalar protocols.
    var cls = py_type(value)
    if _rp_temporal_scalar(value):
        return _rp_temporal_scalar_array(value, context)
    var b = native_loaded_module("builtins")
    var dtype = String()
    var real = UInt64(0)
    var imaginary = UInt64(0)
    if cls is b.bool:
        dtype = "b1"
        real = UInt64(Bool(py=value))
    elif cls is b.int:
        ref cpy = Python().cpython()
        var overflow = Int32(0)
        var number = cpy.lib.get_function[Int64]("PyLong_AsLongLongAndOverflow")(value._obj_ptr, MutPointer(to=overflow))
        if overflow == 0:
            dtype = "i8"
            real = bitcast[DType.uint64](number)
        elif overflow > 0:
            var unsigned = cpy.lib.get_function[UInt64]("PyLong_AsUnsignedLongLong")(value._obj_ptr)
            if cpy.lib.get_function[PyObjectPtr]("PyErr_Occurred")():
                cpy.lib.get_function[NoneType]("PyErr_Clear")()
                dtype = "f8"
                real = bitcast[DType.uint64](Float64(py=value))
            else:
                dtype = "u8"
                real = unsigned
        else:
            dtype = "f8"
            real = bitcast[DType.uint64](Float64(py=value))
    elif cls is b.float:
        dtype = "f8"
        real = bitcast[DType.uint64](Float64(py=value))
    elif cls is b.complex:
        dtype = "c16"
        # These are exact builtin complex C getsets, not application protocols.
        real = bitcast[DType.uint64](Float64(py=value.real))
        imaginary = bitcast[DType.uint64](Float64(py=value.imag))
    else:
        return _rp_scalar_array(value, context)
    var result = _rp_new(Python.tuple(1), PythonObject(dtype), context)
    var target = _RPArray(result, context)
    sequence_store_word(target.address, real, 1 if dtype == "b1" else 8, target.native)
    if dtype == "c16":
        sequence_store_word(target.address + 8, imaginary, 8, target.native)
    _ = value
    return result


def _rp_sparse_fill_type(value: PythonObject, context: PythonObject) raises -> PythonObject:
    # np.result_type(dtype, TYPE(fill)) treats builtin int/float/complex as
    # fixed i8/f8/c16, independent of value or narrower stored sparse subtype.
    var cls = py_type(value)
    var b = native_loaded_module("builtins")
    if cls is b.bool:
        return Python.tuple("b", 1)
    if cls is b.int:
        return Python.tuple("i", 8)
    if cls is b.float:
        return Python.tuple("f", 8)
    if cls is b.complex:
        return Python.tuple("c", 16)
    var codes = Python.tuple("?", "i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "q", "Q", "l", "L", "f2", "f4", "f8", "g", "c8", "c16", "G", "M8[ns]", "m8[ns]")
    for i in range(len(codes)):  # driver-loop: exact builtin NumPy scalar identity before reading its dtype
        var dtype = py_call(context["resample_pandas_dtype_factory"], codes[i])
        if cls is dtype.type:
            var actual = value.dtype
            return Python.tuple(String(actual.kind), Int(py=actual.itemsize))
    # Object, None, NA, Timestamp, Decimal and other stored object fill values
    # require no protocols: NumPy cannot promote their Python type numerically.
    return Python.tuple("O", 0)



def _rp_datetime_type(name: String) raises -> PythonObject:
    var cls = _pd_class("datetime", name)
    ref cpy = Python().cpython()
    if cls is None or (cpy.lib.get_function[UInt64]("PyType_GetFlags")(cls._obj_ptr) & UInt64(512)) != 0:
        raise Error("TypeError: temporal storage boxing requires the static native datetime type")
    return cls


def _rp_numpy_temporal_box(source: _RPArray, address: Int) raises -> PythonObject:
    # NumPy's native object-storage convention differs from pandas' Timestamp
    # boxing: ns and finer stay integers, NaT becomes None, coarse dates and
    # times use native CPython datetime values. See NumPy C API datetime.c,
    # convert_datetime_to_pyobject / convert_timedelta_to_pyobject.
    var raw = bitcast[DType.int64](pd_raw_at(address, 0, 8, source.native))
    if raw == Int64.MIN:
        return PythonObject(None)
    var code = source.code
    if code.endswith("[ns]") or code.endswith("[ps]") or code.endswith("[fs]") or code.endswith("[as]"):
        return PythonObject(raw)
    var years = code.endswith("[Y]")
    var months = code.endswith("[M]")
    if years or months:
        if source.kind == "m":
            return PythonObject(raw)
        var year = Int64(0)
        var month = Int64(1)
        if years:
            if raw < -1969 or raw > 8029:
                return PythonObject(raw)
            year = raw + 1970
        else:
            if raw < -23628 or raw > 96359:
                return PythonObject(raw)
            var total = raw + 23640
            year = total // 12
            month = total % 12 + 1
        return py_call(_rp_datetime_type("date"), PythonObject(year), PythonObject(month), 1)
    var per_day = Int64(1)
    var microseconds = Int64(0)
    var date_only = code.endswith("[D]") or code.endswith("[W]")
    var value = raw
    if code.endswith("[W]"):
        if raw > Int64.MAX // 7 or raw < -(Int64.MAX // 7):
            return PythonObject(raw)
        value *= 7
    elif code.endswith("[h]"):
        per_day = 24
        microseconds = 3600000000
    elif code.endswith("[m]"):
        per_day = 1440
        microseconds = 60000000
    elif code.endswith("[s]"):
        per_day = 86400
        microseconds = 1000000
    elif code.endswith("[ms]"):
        per_day = 86400000
        microseconds = 1000
    elif code.endswith("[us]"):
        per_day = 86400000000
        microseconds = 1
    elif not code.endswith("[D]"):
        raise Error("TypeError: temporal object storage requires an ordinary native unit without a multiplier")
    var days = value // per_day
    var rest = value % per_day
    if rest < 0:
        days -= 1
        rest += per_day
    var usec = rest * microseconds
    if source.kind == "m":
        if days < -999999999 or days > 999999999:
            return PythonObject(raw)
        return py_call(_rp_datetime_type("timedelta"), PythonObject(days), PythonObject(usec // 1000000), PythonObject(usec % 1000000))
    if days < -719162 or days > 2932896:
        return PythonObject(raw)
    # Gregorian civil date from Unix days; only years 1..9999 reach here.
    var shifted = days + 719468
    var era = shifted // 146097
    var doe = shifted - era * 146097
    var yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    var year = yoe + era * 400
    var doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    var mp = (5 * doy + 2) // 153
    var day = doy - (153 * mp + 2) // 5 + 1
    var month = mp + Int64(3 if mp < 10 else -9)
    year += Int64(1 if month <= 2 else 0)
    if date_only:
        return py_call(_rp_datetime_type("date"), PythonObject(year), PythonObject(month), PythonObject(day))
    return py_call(_rp_datetime_type("datetime"), PythonObject(year), PythonObject(month), PythonObject(day), PythonObject(usec // 3600000000), PythonObject((usec // 60000000) % 60), PythonObject((usec // 1000000) % 60), PythonObject(usec % 1000000))


def _rp_sparse_object_fill(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var scalars = context["resample_pandas_scalar_classes"]
    for i in range(len(scalars)):  # driver-loop: exact NumPy scalar fills use native storage-to-object boxing
        if py_type(value) is scalars[i]:
            var source = _RPArray(_rp_scalar_array(value, context), context)
            if source.kind == "M" or source.kind == "m":
                return _rp_numpy_temporal_box(source, source.address)
            return _rp_box(source, source.address)
    return value


def _rp_sparse_fill_missing(value: PythonObject, context: PythonObject) raises -> Bool:
    if value is None or value is _rp_na() or value is _rp_item("pandas._libs.tslibs.nattype", "NaT"):
        return True
    var kind = String(_rp_sparse_fill_type(value, context)[0])
    if kind != "f" and kind != "c" and kind != "M" and kind != "m":
        return False
    var source = _RPArray(_rp_sparse_fill_array(value, context), context)
    if source.kind == "c":
        var width = source.item // 2
        return (sequence_string_nan_raw(source.address, width) & 1) != 0 or (sequence_string_nan_raw(source.address + width, width) & 1) != 0
    return _rp_interval_missing(source, 0)


def _rp_sparse_fills_equal(left: PythonObject, right: PythonObject, context: PythonObject) raises -> Bool:
    if left is right:
        return True
    var a = py_type(left)
    var b = py_type(right)
    if _rp_sparse_fill_missing(left, context) or _rp_sparse_fill_missing(right, context):
        ref cpy = Python().cpython()
        return cpy.lib.get_function[Int32]("PyType_IsSubtype")(a._obj_ptr, b._obj_ptr) != 0 or cpy.lib.get_function[Int32]("PyType_IsSubtype")(b._obj_ptr, a._obj_ptr) != 0
    var builtins = native_loaded_module("builtins")
    if (a is builtins.int or a is builtins.float or a is builtins.bool or a is builtins.complex) and (b is builtins.int or b is builtins.float or b is builtins.bool or b is builtins.complex):
        ref cpy = Python().cpython()
        var result = cpy.lib.get_function[Int32]("PyObject_RichCompareBool")(left._obj_ptr, right._obj_ptr, Int32(2))
        if result < 0:
            raise cpy.unsafe_get_error()
        return result != 0
    if _rp_temporal_scalar(left) and _rp_temporal_scalar(right):
        var date_type = _pd_class("datetime", "date")
        if (a is date_type) != (b is date_type):
            return False
        var dt_type = _pd_class("datetime", "datetime")
        var pd_type = _pd_class("pandas._libs.tslibs.timestamps", "Timestamp")
        var ldatetime = a is dt_type or a is pd_type
        var rdatetime = b is dt_type or b is pd_type
        if ldatetime != rdatetime:
            return False
        if ldatetime:
            var ltz = _rp_sparse_field(left, "tzinfo")
            var rtz = _rp_sparse_field(right, "tzinfo")
            _rp_timezone(ltz)
            _rp_timezone(rtz)
            if (ltz is None) != (rtz is None):
                return False
        if (a is dt_type or a is date_type or a is _pd_class("datetime", "timedelta")) and (b is dt_type or b is date_type or b is _pd_class("datetime", "timedelta")):
            ref cpy = Python().cpython()
            var equal = cpy.lib.get_function[Int32]("PyObject_RichCompareBool")(left._obj_ptr, right._obj_ptr, Int32(2))
            if equal < 0:
                raise cpy.unsafe_get_error()
            return equal != 0
        var ldata = _RPArray(_rp_temporal_scalar_array(left, context), context)
        var rdata = _RPArray(_rp_temporal_scalar_array(right, context), context)
        return _rp_category_cell_equal(ldata, 0, rdata, 0)
    if String(_rp_sparse_fill_type(left, context)[0]) == "O" or String(_rp_sparse_fill_type(right, context)[0]) == "O":
        # Object sparse subtypes already export object; no user equality is
        # needed to choose that dtype for distinct arbitrary object fills.
        return False
    var ldata = _RPArray(_rp_sparse_fill_array(left, context), context)
    var rdata = _RPArray(_rp_sparse_fill_array(right, context), context)
    return _rp_category_cell_equal(ldata, 0, rdata, 0)


def _rp_float_integer_fill(source: _RPArray, destination: Int, target: _RPArray) raises:
    var negative = False
    var exponent = 0
    var high = UInt64(0)
    var low = UInt64(0)
    var precision = 53
    var nonfinite = False
    if source.item == 16:
        low = pd_raw_at(source.address, 0 if source.native else 8, 8, source.native)
        high = pd_raw_at(source.address, 8 if source.native else 0, 8, source.native)
        comptime if CompilationTarget.is_x86():
            negative = (high & UInt64(0x8000)) != 0
            var encoded = Int(high & UInt64(0x7fff))
            exponent = encoded - 16383
            nonfinite = encoded == 32767
            precision = 64
            high = low
            low = 0
        else:
            negative = (high >> 63) != 0
            var encoded = Int((high >> 48) & UInt64(0x7fff))
            exponent = encoded - 16383
            nonfinite = encoded == 32767
            precision = 113
            high = (high & UInt64(0xffffffffffff)) | UInt64(0x1000000000000)
    else:
        var word = bitcast[DType.uint64](_rp_number(source, source.address))
        negative = (word >> 63) != 0
        var encoded = Int((word >> 52) & UInt64(0x7ff))
        exponent = encoded - 1023
        nonfinite = encoded == 2047
        high = (word & UInt64(0xfffffffffffff)) | UInt64(0x10000000000000)
    if target.kind == "b":
        # Floating zero alone is false; NaN and infinity are nonzero.
        var zero = True
        var bytes = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=source.address)
        var meaningful = source.item
        comptime if CompilationTarget.is_x86():
            if meaningful == 16:
                meaningful = 10
        for byte in range(meaningful):  # driver-loop: native floating-to-bool scalar storage conversion
            var value = bytes[byte if source.native else source.item - 1 - byte]
            if byte == meaningful - 1:
                value &= UInt8(0x7f)
            zero = zero and value == 0
        sequence_store_word(destination, UInt64(0) if zero else UInt64(1), 1, target.native)
        return
    if nonfinite:
        raise Error("ValueError: a nonfinite sparse fill cannot be represented in the selected integer dtype")
    if exponent > 63:
        raise Error("OverflowError: sparse floating fill exceeds the selected integer dtype")
    var magnitude = UInt64(0)
    if exponent >= 0:
        if precision == 113:
            magnitude = (high >> UInt64(48 - exponent)) if exponent <= 48 else ((high << UInt64(exponent - 48)) | (low >> UInt64(112 - exponent)))
        elif exponent >= precision - 1:
            magnitude = high << UInt64(exponent - precision + 1)
        else:
            magnitude = high >> UInt64(precision - 1 - exponent)
    var limit = UInt64.MAX if target.item == 8 else (UInt64(1) << UInt64(target.item * 8)) - 1
    if target.kind == "i":
        limit = UInt64(1) << UInt64(target.item * 8 - 1)
        if not negative:
            limit -= 1
    if (negative and magnitude != 0 and target.kind == "u") or magnitude > limit:
        raise Error("OverflowError: sparse floating fill exceeds the selected integer dtype")
    var word = (~magnitude + UInt64(1)) if negative else magnitude
    sequence_store_word(destination, word, target.item, target.native)

def _rp_sparse_fill_copy(source: _RPArray, destination: Int, target: _RPArray) raises:
    if source.kind == "f" and (target.kind == "i" or target.kind == "u" or target.kind == "b"):
        _rp_float_integer_fill(source, destination, target)
        return
    if (source.kind == "i" or source.kind == "u" or source.kind == "b") and (target.kind == "i" or target.kind == "u"):
        var word = cast_w64_bits(pd_raw_at(source.address, 0, source.item, source.native), 1 if source.kind == "i" else 0, source.item)
        var negative = source.kind == "i" and (word >> 63) != 0
        var magnitude = (~word + UInt64(1)) if negative else word
        var limit = UInt64.MAX if target.item == 8 else (UInt64(1) << UInt64(target.item * 8)) - 1
        if target.kind == "i":
            limit = UInt64(1) << UInt64(target.item * 8 - 1)
            if not negative:
                limit -= 1
        if (negative and target.kind == "u") or magnitude > limit:
            raise Error("OverflowError: pandas sparse fill is outside the selected integer dtype")
    # A scalar's actual unit can be finer than the sparse dtype even though
    # its TYPE in result_type is a generic datetime64/timedelta64 descriptor.
    if (target.kind == "M" or target.kind == "m") and source.kind == target.kind:
        var delta = _rp_time_unit(target.code) - _rp_time_unit(source.code)
        if delta < 0:
            var value = bitcast[DType.int64](pd_raw_at(source.address, 0, 8, source.native))
            if value != Int64.MIN:
                for i in range(-delta):  # driver-loop: native scalar assignment uses floor when reducing temporal resolution
                    value = value // 1000
            sequence_store_word(destination, bitcast[DType.uint64](value), 8, target.native)
            return
    _rp_copy(source, source.address, destination, target)


def _rp_sparse(values: PythonObject, force_object: Bool, context: PythonObject,
               requested_dtype: PythonObject) raises -> PythonObject:
    var record = _rp_sparse_record(values, context)
    var fields = record[1]
    var source = _RPArray(record[2], context)
    if source.rank != 1:
        raise Error("ValueError: pandas sparse values require one-dimensional storage")
    var dtype = record[3]
    var dtype_fields = record[4]
    var fill = record[5]
    var index = record[6]
    var integer_index = py_type(index) is _pd_class("pandas._libs.sparse", "IntIndex")
    var rows = Int(py=record[7])
    var points = Int(py=record[8])
    if rows < 0 or points < 0 or points > rows or points != source.rows:
        raise Error("ValueError: pandas sparse length and stored point count disagree")
    var positions = List[Int](capacity=points)
    var owners = Python.list()
    if integer_index:
        var placed = _RPArray(_rp_sparse_field(index, "indices"), context)
        if placed.rows != points:
            raise Error("ValueError: pandas sparse IntIndex count differs from stored values")
        var previous = -1
        for i in range(points):  # driver-loop: admit every sorted sparse position before allocating output
            var position = _rp_sparse_index_word(placed, i)
            if position <= previous or position >= rows:
                raise Error("ValueError: pandas sparse positions overlap or exceed row bounds")
            positions.append(position)
            previous = position
        py_call(owners.append, placed.owner)
    else:
        var starts = _RPArray(_rp_sparse_field(index, "blocs"), context)
        var lengths = _RPArray(_rp_sparse_field(index, "blengths"), context)
        var blocks = Int(py=_rp_sparse_field(index, "nblocks"))
        if blocks != starts.rows or blocks != lengths.rows:
            raise Error("ValueError: pandas sparse BlockIndex metadata lengths disagree")
        var end = 0
        for i in range(blocks):  # driver-loop: expand native block placement without SparseIndex methods
            var start = _rp_sparse_index_word(starts, i)
            var length = _rp_sparse_index_word(lengths, i)
            if start < end or start < 0 or length <= 0 or length > rows - start or length > points - len(positions):
                raise Error("ValueError: pandas sparse blocks overlap or exceed row bounds")
            for j in range(length):  # driver-loop: one native output position per stored sparse point
                positions.append(start + j)
            end = start + length
        if len(positions) != points:
            raise Error("ValueError: pandas sparse blocks do not cover their declared points")
        py_call(owners.append, starts.owner)
        py_call(owners.append, lengths.owner)
    # A gap-free sparse array ignores fill type entirely; no fill protocols or
    # even scalar metadata need to be examined in that case.
    var target_dtype = source.dtype
    var nat_datetime_fill = False
    if points < rows:
        var fill_type = _rp_sparse_fill_type(fill, context)
        var fill_kind = String(fill_type[0])
        var fill_size = Int(py=fill_type[1])
        nat_datetime_fill = source.kind == "M" and fill is _rp_item("pandas._libs.tslibs.nattype", "NaT")
        if nat_datetime_fill or ((source.kind == "M" or source.kind == "m") and fill_kind == source.kind):
            target_dtype = source.dtype
        elif source.kind == "O" or fill_kind == "O":
            target_dtype = PythonObject("O")
        elif (source.kind == "b" or source.kind == "i" or source.kind == "u" or source.kind == "f" or source.kind == "c") and (fill_kind == "b" or fill_kind == "i" or fill_kind == "u" or fill_kind == "f" or fill_kind == "c"):
            var complex_values = source.kind == "c" or fill_kind == "c"
            var kind = "f" if source.kind == "c" else source.kind
            var size = source.item // 2 if source.kind == "c" else source.item
            _rp_promote(kind, size, "f" if fill_kind == "c" else fill_kind, fill_size // 2 if fill_kind == "c" else fill_size)
            var extended = source.number == 13 or source.number == 16 or py_type(fill) is py_call(context["resample_pandas_dtype_factory"], "g").type or py_type(fill) is py_call(context["resample_pandas_dtype_factory"], "G").type
            target_dtype = PythonObject(("G" if complex_values else "g") if extended else (("c" if complex_values else kind) + String(size * (2 if complex_values else 1))))
        else:
            target_dtype = PythonObject("O")
    if requested_dtype is not None:
        target_dtype = requested_dtype
    if force_object:
        target_dtype = PythonObject("O")
    var result = _rp_new(Python.tuple(rows), target_dtype, context)
    var target = _RPArray(result, context)
    if points < rows:
        if target.kind == "O":
            var boxed_fill = _rp_sparse_object_fill(fill, context)
            for row in range(rows):  # driver-loop: retain arbitrary objects and box exact native scalar fills as pandas does
                _rp_object(target.at(row), boxed_fill)
            _ = boxed_fill
        elif nat_datetime_fill:
            for row in range(rows):  # driver-loop: native missing datetime fill has the iNaT representation
                sequence_store_word(target.at(row), UInt64(1) << 63, 8, target.native)
        else:
            var component_width = target.item // 2 if target.kind == "c" else target.item
            if component_width > 8 and (target.kind == "f" or target.kind == "c") and py_type(fill) is py_type(PythonObject(0)):
                # The ordinary scalar helper can route a larger Python int
                # through float64. Never silently narrow a wide floating fill;
                # arbitrary-precision integer encoding remains an explicit gap.
                var b = native_loaded_module("builtins")
                var bits = Int(py=py_call(b.int.__dict__["bit_length"], fill))
                ref cpy = Python().cpython()
                var zero = PythonObject(0)
                var negative = cpy.lib.get_function[Int32]("PyObject_RichCompareBool")(fill._obj_ptr, zero._obj_ptr, Int32(0)) != 0
                var minimum = PythonObject(Int64.MIN)
                var is_minimum = cpy.lib.get_function[Int32]("PyObject_RichCompareBool")(fill._obj_ptr, minimum._obj_ptr, Int32(2)) != 0
                if bits > 64 or (negative and bits == 64 and not is_minimum):
                    raise Error("TypeError: extended sparse fills beyond native 64-bit integer storage require native arbitrary-precision encoding")
            var fill_source = _RPArray(_rp_sparse_fill_array(fill, context), context)
            for row in range(rows):  # driver-loop: native scalar storage assignment; no NumPy full/astype
                _rp_sparse_fill_copy(fill_source, target.at(row), target)
    for i in range(points):  # driver-loop: scatter stored sparse values; fill owner remains retained on overwrite
        if target.kind == "O" and (source.kind == "M" or source.kind == "m"):
            _rp_object(target.at(positions[i]), _rp_numpy_temporal_box(source, source.at(i)))
        else:
            _rp_copy(source, source.at(i), target.at(positions[i]), target)
    _ = owners^
    _ = index
    _ = fill
    _ = dtype_fields
    _ = dtype
    _ = fields
    _ = values
    return result


def _rp_interval_constructor() raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = _pd_class("pandas._libs.interval", "Interval")
    if cls is None or py_type(cls) is not b.type:
        raise Error("TypeError: pandas Interval requires its exact native constructor type")
    var fields = _pd_type_fields(cls)
    var new = py_get(fields, "__new__")
    var init = py_get(fields, "__init__")
    # Read raw class fields before descriptor binding. Borrowed native slots on
    # a substituted Python class fail their bound owner checks before a call.
    if new is None or py_type(new) is not py_type(b.object.__dict__["__new__"]) or new.__self__ is not cls:
        raise Error("TypeError: pandas Interval requires its native allocation entry")
    if init is None or py_type(init) is not py_type(b.object.__dict__["__init__"]) or init.__objclass__ is not cls:
        raise Error("TypeError: pandas Interval requires its native initialization entry")
    return cls


def _rp_interval_missing(source: _RPArray, row: Int) raises -> Bool:
    var address = source.at(row)
    if source.kind == "M" or source.kind == "m":
        return pd_raw_at(address, 0, 8, source.native) == UInt64(1) << 63
    if source.kind == "f":
        if source.native:
            return (sequence_string_nan_raw(address, source.item) & 1) != 0
        var native = List[UInt8](length=source.item, fill=0)
        var input = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
        for i in range(source.item):  # driver-loop: normalize admitted floating storage before bit classification
            native[i] = input[source.item - 1 - i]
        var missing = (sequence_string_nan_raw(Int(native.unsafe_ptr()), source.item) & 1) != 0
        _ = native^
        return missing
    if source.kind == "O":
        var value = _rp_box(source, address)
        if value is _rp_na() or value is _rp_item("pandas._libs.tslibs.nattype", "NaT") or value is None:
            return True
        if py_type(value) is py_type(PythonObject(0.0)):
            var number = Float64(py=value)
            return number != number
    return False


def _rp_interval_scalar(value: PythonObject, context: PythonObject) raises:
    var cls = py_type(value)
    var b = native_loaded_module("builtins")
    if cls is b.int or cls is b.float or cls is b.bool:
        return
    if cls is _pd_class("pandas._libs.tslibs.timestamps", "Timestamp"):
        _rp_timezone(_rp_sparse_field(value, "tzinfo"))
        return
    if cls is _pd_class("pandas._libs.tslibs.timedeltas", "Timedelta"):
        return
    var codes = Python.tuple("i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "q", "Q", "l", "L", "f2", "f4", "f8", "g")
    for i in range(len(codes)):  # driver-loop: exact native numerical endpoints only, before Interval comparisons
        var dtype = py_call(context["resample_pandas_dtype_factory"], codes[i])
        if cls is dtype.type:
            return
    raise Error("TypeError: pandas Interval endpoints cannot invoke Python scalar comparison hooks")


def _rp_interval(values: PythonObject, distinguish: Bool, context: PythonObject, depth: Int) raises -> PythonObject:
    var fields = _rp_dict(values)
    var dtype = py_get(fields, "_dtype")
    if py_type(dtype) is not _pd_class("pandas.core.dtypes.dtypes", "IntervalDtype"):
        raise Error("TypeError: pandas interval storage requires its exact stored dtype")
    var closed = py_get(_rp_dict(dtype), "_closed")
    if py_type(closed) is not py_type(PythonObject("")):
        raise Error("TypeError: pandas interval closure must be an exact native string")
    var closure = String(closed)
    if closure != "left" and closure != "right" and closure != "both" and closure != "neither":
        raise Error("ValueError: pandas interval closure is invalid")
    var left_owner = py_get(fields, "_left")
    var right_owner = py_get(fields, "_right")
    var left = _RPArray(_rp_storage(left_owner, False, distinguish, context, depth + 1), context)
    var right = _RPArray(_rp_storage(right_owner, False, distinguish, context, depth + 1), context)
    if left.rank != 1 or right.rank != 1 or left.rows != right.rows:
        raise Error("ValueError: pandas interval endpoints require matching one-dimensional storage")
    var constructor = _rp_interval_constructor()
    var result = _rp_new(Python.tuple(left.rows), PythonObject("O"), context)
    var target = _RPArray(result, context)
    var nan = _rp_nan()
    for row in range(left.rows):  # driver-loop: native Interval scalar boxing and missing-value placement
        if _rp_interval_missing(left, row):
            _rp_object(target.at(row), nan)
        else:
            var lo = _rp_box(left, left.at(row))
            var hi = _rp_box(right, right.at(row))
            _rp_interval_scalar(lo, context)
            _rp_interval_scalar(hi, context)
            _rp_object(target.at(row), py_call(constructor, lo, hi, closed))
    _ = constructor
    _ = left_owner
    _ = right_owner
    _ = dtype
    _ = fields
    _ = values
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
        var record = _rp_category_record(values, distinguish, context, depth)
        var category_values = record[3]
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
        var source = _RPArray(record[4], context)
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
        if force_object or (missing and source.kind == "b") or (missing and distinguish and category_na is _rp_na()):
            output_dtype = PythonObject("O")
        elif missing and (source.kind == "i" or source.kind == "u"):
            output_dtype = PythonObject("f8")
        var result = _rp_new(Python.tuple(codes.rows), output_dtype, context)
        var target = _RPArray(result, context)
        for row in range(codes.rows):  # driver-loop: categorical gather preserves category object identity
            var code = Int(bitcast[DType.int64](cast_w64_bits(pd_raw_at(codes.at(row), 0, codes.item, codes.native), 1, codes.item)))
            if code >= 0:
                if target.kind == "O" and (source.kind == "M" or source.kind == "m") and category_type is _pd_class("pandas.core.arrays.sparse.array", "SparseArray"):
                    _rp_object(target.at(row), _rp_numpy_temporal_box(source, source.at(code)))
                else:
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
    if cls is _pd_class("pandas.core.arrays.sparse.array", "SparseArray"):
        return _rp_sparse(values, force_object, context, PythonObject(None))
    if cls is _pd_class("pandas.core.arrays.interval", "IntervalArray"):
        return _rp_interval(values, distinguish, context, depth)
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
    var logical = List[_RPArray]()
    var logical_sparse = Python.list()
    var placements = Python.list()
    var raw_owners = Python.list()
    var force_object = False
    var has_sparse = False
    var has_categorical = False
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
        has_categorical = has_categorical or categorical
        has_sparse = has_sparse or cls is _pd_class("pandas.core.arrays.sparse.array", "SparseArray")
        all_categorical = all_categorical and categorical
        if categorical:
            var record = _rp_category_record(raw, distinguish, context, 0)
            var category_kind = py_type(record[3])
            if len(blocks) > 1 and category_kind is not _pd_class("pandas.core.arrays.sparse.array", "SparseArray") and category_kind is not context["ndarray"] and category_kind is not _pd_class("pandas.core.arrays.datetimes", "DatetimeArray") and category_kind is not _pd_class("pandas.core.arrays.timedeltas", "TimedeltaArray"):
                force_object = True
            var dtype = record[1]
            if categorical_dtype is None:
                categorical_dtype = dtype
            else:
                same_categorical_dtype = same_categorical_dtype and dtype is categorical_dtype
        if len(blocks) > 1 and not categorical and cls is not _pd_class("pandas.core.arrays.sparse.array", "SparseArray") and cls is not context["ndarray"] and cls is not _pd_class("pandas.core.arrays.datetimes", "DatetimeArray") and cls is not _pd_class("pandas.core.arrays.timedeltas", "TimedeltaArray"):
            force_object = True
    var use_sparse_promotion = has_sparse and not has_categorical
    if has_categorical:
        for k in range(len(raw_owners)):  # driver-loop: categorical unwrapping preserves SparseDtype of the category index itself
            if py_type(raw_owners[k]) is _pd_class("pandas.core.arrays.categorical", "Categorical"):
                var record = _rp_category_record(raw_owners[k], distinguish, context, 0)
                use_sparse_promotion = use_sparse_promotion or py_type(record[3]) is _pd_class("pandas.core.arrays.sparse.array", "SparseArray")
    if all_categorical:
        var equal_categories = True
        for i in range(1, len(raw_owners)):  # driver-loop: resolve common categorical dtype without pandas equality/hash methods
            equal_categories = equal_categories and _rp_categories_equal(raw_owners[0], raw_owners[i], distinguish, context)
        force_object = force_object or equal_categories
    var rows = -1
    var cols = 0
    var common_kind = String()
    var common_size = 0
    var common_code = String()
    var same = True
    var booleans = False
    var other = False
    var complex_values = False
    var extended_values = False
    for k in range(len(blocks)):  # driver-loop: capture each native array's dtype, shape and signed strides once
        var mixed_sparse = len(blocks) > 1 and py_type(raw_owners[k]) is _pd_class("pandas.core.arrays.sparse.array", "SparseArray")
        var sparse = _rp_sparse_record(raw_owners[k], context) if mixed_sparse else PythonObject(None)
        var data = _RPArray(sparse[2] if mixed_sparse else _rp_storage(raw_owners[k], force_object, distinguish, context), context)
        var actual_rows = Int(py=sparse[7]) if mixed_sparse else data.rows
        if (rows >= 0 and actual_rows != rows) or (series and data.rank != 1):
            raise Error("ValueError: pandas blocks disagree on row dimensions")
        rows = actual_rows
        cols += data.cols
        var logical_owner = data.owner
        var sparse_logical = sparse if mixed_sparse and not has_categorical else PythonObject(None)
        if len(blocks) > 1 and py_type(raw_owners[k]) is _pd_class("pandas.core.arrays.categorical", "Categorical"):
            var record = _rp_category_record(raw_owners[k], distinguish, context, 0)
            if py_type(record[3]) is _pd_class("pandas.core.arrays.sparse.array", "SparseArray"):
                logical_owner = _rp_category_logical(record, context)
                sparse_logical = _rp_sparse_record(record[3], context)
        py_call(logical_sparse.append, sparse_logical)
        logical.append(_RPArray(logical_owner, context))
        ref promotion = logical[k]
        var positions = Python.tuple(0) if series else _pd_locations(blocks[k], context)
        if len(positions) != data.cols:
            raise Error("ValueError: pandas block placement and storage widths disagree")
        py_call(placements.append, positions)
        if k == 0:
            common_code = promotion.code
        else:
            same = same and promotion.code == common_code
        extended_values = extended_values or promotion.number == 13 or promotion.number == 16
        booleans = booleans or promotion.kind == "b"
        other = other or promotion.kind != "b"
        if promotion.kind == "O":
            force_object = True
        elif promotion.kind == "b" or promotion.kind == "i" or promotion.kind == "u" or promotion.kind == "f" or promotion.kind == "c":
            var numeric_kind = "f" if promotion.kind == "c" else promotion.kind
            var numeric_size = promotion.item // 2 if promotion.kind == "c" else promotion.item
            complex_values = complex_values or promotion.kind == "c"
            _rp_promote(common_kind, common_size, numeric_kind, numeric_size)
        owned.append(data^)
    if len(blocks) > 1 and use_sparse_promotion:
        var logical_same = same
        var first_sparse = PythonObject(None)
        for k in range(len(blocks)):  # driver-loop: original sparse dtype equality includes native fill semantics
            if logical_sparse[k] is None:
                logical_same = False
            else:
                var record = logical_sparse[k]
                if first_sparse is None:
                    first_sparse = record
                elif record[3] is not first_sparse[3]:
                    logical_same = logical_same and _rp_sparse_fills_equal(first_sparse[5], record[5], context)
        same = same and logical_same
    force_object = force_object or (booleans and other and not use_sparse_promotion)
    var dtype = logical[0].dtype
    if force_object:
        dtype = PythonObject("O")
    elif not same:
        var temporal = logical[0].kind == "M" or logical[0].kind == "m"
        var finest = 7
        var numeric = True
        for k in range(len(logical)):  # driver-loop: classify common native dtype without Python promotion hooks
            var kind = logical[k].kind
            temporal = temporal and kind == logical[0].kind
            numeric = numeric and (kind == "b" or kind == "i" or kind == "u" or kind == "f" or kind == "c")
            if kind == "M" or kind == "m":
                finest = max(finest, _rp_time_unit(logical[k].code))
        if use_sparse_promotion and (temporal or not numeric):
            force_object = True
            dtype = PythonObject("O")
        elif temporal:
            var units = Python.tuple("s", "ms", "us", "ns")
            dtype = PythonObject(logical[0].kind + "8[" + String(units[finest - 7]) + "]")
        elif numeric:
            if common_size == 0:
                raise Error("TypeError: pandas numeric promotion requires an implemented native scalar width")
            dtype = PythonObject(("c" if complex_values else common_kind) + String(common_size * (2 if complex_values else 1)))
        else:
            force_object = True
            dtype = PythonObject("O")
    if not force_object and extended_values:
        dtype = PythonObject("G" if complex_values else "g")
    if len(blocks) > 1 and has_sparse:
        for k in range(len(owned)):  # driver-loop: expand sparse fills only after selecting the manager's original-subtype common dtype
            if py_type(raw_owners[k]) is _pd_class("pandas.core.arrays.sparse.array", "SparseArray"):
                owned[k] = _RPArray(_rp_sparse(raw_owners[k], force_object, context, dtype), context)
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
    _ = logical^
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
    var view = py_get(_pd_type_fields(ndarray), "view")
    if py_type(view) is not py_type(native_loaded_module("builtins").list.__dict__["copy"]) or view.__objclass__ is not ndarray:
        raise Error("TypeError: pandas snapshots require the native ndarray.view descriptor")
    var classes = Python.list()
    var scalars = Python.list()
    var codes = Python.tuple("?", "i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "q", "Q", "l", "L", "f2", "f4", "f8", "g", "c8", "c16", "G", "O", "S", "U", "V", "M8[ns]", "m8[ns]")
    for i in range(len(codes)):  # driver-loop: operation-stable admission identities for builtin native dtype classes
        var descriptor = py_call(dtype, codes[i])
        py_call(classes.append, py_type(descriptor))
        py_call(scalars.append, descriptor.type)
    var local = Python.dict(ndarray=ndarray, modules=context["modules"],
        resample_pandas_dtype_classes=classes, resample_pandas_dtype_factory=dtype,
        resample_pandas_scalar_classes=scalars, resample_pandas_scalar_api=_rp_api(dtype), resample_pandas_view=view,
        resample_pandas_array_snapshots=Python.list(), resample_pandas_category_records=Python.list(), resample_pandas_sparse_records=Python.list())
    return _rp_materialize(value, local)
