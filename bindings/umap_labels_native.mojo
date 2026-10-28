# SPDX-License-Identifier: Apache-2.0
# Native categorical equality encoding. The UMAP categorical graph consumes
# only equality and the -1 unknown marker (ug_categorical_weight), so encounter
# order codes preserve the graph without Python set/sorted or comparison hooks.
from std.collections import Dict
from std.memory import bitcast
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get
from bindings.native_loaded_module import native_loaded_module, _loaded_fields
from bindings.sequence_dtype_native import sequence_dtype_classes
from bindings.classical_pandas import classical_admit_value, native_pandas_exact
from bindings.classical_labels import _pandas_values
from bindings.classical_validate import classical_view
from bindings.classical_arrays import classical_address
from bindings.family_common import FAMILY_F32P, family_array
from bindings.sequential_admission import _one_of
from bindings.native_scalar_decode import native_scalar_buffer
from bindings.umap_parameters_native import umap_is_string
from bindings.manifold_lifecycle_native import _target_list_kind, _target_finite
from bindings.classical_warnings import COLUMN_WARNING, _warning_category
from bindings.metrics_native_support import metric_warn
from providers.python.array_metadata_pyapi import array_shape_once, array_metadata_once
from core.native_scalar_decode import NativeRealBits, native_scalar_real, _scalar_ieee


def _real_key(value: NativeRealBits, identity: Int) -> String:
    if value.equals_integer(-1):
        return "unknown"
    if value.special == 2:
        # Python containers identify repeated references to one NaN, while
        # distinct NaN objects remain unequal. Numeric ndarray NaNs are refused
        # by the public float32/64 target check before this branch.
        return "nan:" + String(identity)
    if value.special == 1:
        return "n-inf" if value.negative else "n+inf"
    if not value.truth():
        return "n0"
    var low = 0
    var high = value.highest()
    while low <= high and value.bit(low) == 0:  # driver-loop: normalize one native scalar significand
        low += 1
    var result = String("n-") if value.negative else String("n+")
    for i in range(high - low + 1):  # driver-loop: exact numeric label significand, at most 128 bits
        result += "1" if value.bit(high - i) != 0 else "0"
    return result + "@" + String(value.scale + low)


def _integer_key(value: PythonObject) raises -> String:
    # Public CPython integer transport for an already-admitted exact int/bool.
    # PyNumber_ToBase cannot reach an application __index__ for these types.
    ref cpy = Python().cpython()
    var pointer = cpy.lib.get_function[PyObjectPtr]("PyNumber_ToBase")(value._obj_ptr, Int32(2))
    if not pointer:
        raise cpy.unsafe_get_error()
    var binary = PythonObject(from_owned=pointer)
    var text = String(binary)
    var negative = text.startswith("-")
    var begin = 3 if negative else 2
    if text == "-0b1":
        return "unknown"
    if text == "0b0":
        return "n0"
    var end = text.byte_length()
    var shift = 0
    while end > begin and text[byte=end - 1] == "0":  # driver-loop: exact arbitrary-precision integer label normalization
        end -= 1
        shift += 1
    return (String("n-") if negative else String("n+")) + String(text[byte=begin:end]) + "@" + String(shift)


def _unicode_key(value: PythonObject) raises -> String:
    # Code-point transport supports embedded NUL and unpaired surrogate labels;
    # no UTF-8 codec, __str__, hashing or Python equality is involved.
    ref cpy = Python().cpython()
    var count = cpy.lib.get_function[Int]("PyUnicode_GetLength")(value._obj_ptr)
    var address = cpy.lib.get_function[Int]("PyUnicode_AsUCS4Copy")(value._obj_ptr)
    if count < 0 or address == 0:
        raise cpy.unsafe_get_error()
    var pointer = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=address)
    var result = String("u:")
    for i in range(count):  # driver-loop: native categorical string code points
        result += String(pointer[i]) + ":"
    cpy.lib.get_function[NoneType]("PyMem_Free")(address)
    return result


def _bytes_key(value: PythonObject) raises -> String:
    ref cpy = Python().cpython()
    var words = List[Int](length=2, fill=0)
    var pointer = words.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    if cpy.lib.get_function[Int32]("PyBytes_AsStringAndSize")(value._obj_ptr, pointer, pointer + 1) != 0:
        raise cpy.unsafe_get_error()
    var data = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=words[0])
    var result = String("b:")
    for i in range(words[1]):  # driver-loop: native categorical byte-string bytes
        result += String(data[i]) + ":"
    return result


def _ordinal(year: Int, month: Int, day: Int) -> Int:
    var previous = year - 1
    var result = 365 * previous + previous // 4 - previous // 100 + previous // 400 + day
    var lengths: List[Int] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for i in range(month - 1):  # small-loop(calendar: at most eleven preceding months)
        result += lengths[i]
    if month > 2 and year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
        result += 1
    return result


def _floor_div(value: Int, divisor: Int) -> Int:
    # Unsigned magnitude also handles the minimum signed value, independent
    # of the language's signed division/remainder rounding convention.
    if value >= 0:
        return Int(UInt64(value) // UInt64(divisor))
    var magnitude = UInt64(0) - UInt64(value)
    var quotient = magnitude // UInt64(divisor)
    if magnitude % UInt64(divisor) != 0:
        quotient += 1
    return Int(bitcast[DType.int64](UInt64(0) - quotient))


def _machine_key(value: Int) -> String:
    var negative = value < 0
    var magnitude = UInt64(-(value + 1)) + UInt64(1) if negative else UInt64(value)
    return _real_key(NativeRealBits(low=magnitude, negative=negative), 0)


def _datetime_key(raw: Int, code: Int, spelling: String) raises -> String:
    # Preserve ndarray.tolist's documented scalar categories without calling
    # tolist/item: dates through D, datetimes through us, raw integers finer
    # than us or outside Python's calendar range, and None for NaT.
    # Semantics: numpy/_core/src/multiarray/datetime.c,
    # convert_datetime_to_pyobject / convert_timedelta_to_pyobject.
    if raw == -9223372036854775807 - 1:
        return "none"
    var text = spelling.as_bytes()
    var begin = 0
    while begin < spelling.byte_length() and text[begin] != UInt8(91):  # small-loop(dtype: fixed datetime unit spelling up to "[", byte 91)
        begin += 1
    if begin == spelling.byte_length():
        return "none" if code == 21 else _machine_key(raw)
    begin += 1
    var multiple = 0
    while begin < spelling.byte_length():  # small-loop(dtype: optional native unit multiplier digits)
        var digit = Int(text[begin]) - 48  # 48 is "0"
        if digit < 0 or digit > 9:
            break
        multiple = multiple * 10 + digit
        begin += 1
    multiple = max(multiple, 1)
    var unit = String(spelling[byte=begin:spelling.byte_length() - 1])
    if unit == "ns" or unit == "ps" or unit == "fs" or unit == "as" or (code == 22 and (unit == "Y" or unit == "M")):
        return _machine_key(raw)
    # NumPy applies the metadata multiplier in its signed 64-bit storage.
    var value = Int(bitcast[DType.int64](UInt64(raw) * UInt64(multiple)))
    if code == 21 and (unit == "Y" or unit == "M"):
        var years = value if unit == "Y" else _floor_div(value, 12)
        var year = Int(bitcast[DType.int64](UInt64(years) + UInt64(1970)))
        if year < 1 or year > 9999:
            return _machine_key(raw)
        var month = 1 if unit == "Y" else value - _floor_div(value, 12) * 12 + 1
        return "date:" + String(_ordinal(year, month, 1))
    var days = 0
    var remainder = 0
    if unit == "W":
        days = Int(bitcast[DType.int64](UInt64(value) * UInt64(7)))
    elif unit == "D":
        days = value
    else:
        var per_day = 24 if unit == "h" else 1440 if unit == "m" else 86400 if unit == "s" else 86400000 if unit == "ms" else 86400000000 if unit == "us" else 0
        if per_day == 0:
            raise Error("TypeError: UMAP datetime label metadata has an unsupported unit")
        days = _floor_div(value, per_day)
        var units = Int(UInt64(value) - UInt64(days) * UInt64(per_day))
        remainder = units * (86400000000 // per_day)
    if code == 22:
        if days < -999999999 or days > 999999999:
            return _machine_key(raw)
        return "delta:" + String(days) + ":" + String(remainder // 1000000) + ":" + String(remainder % 1000000)
    if days < -719162 or days > 2932896:
        return _machine_key(raw)
    if unit == "W" or unit == "D":
        return "date:" + String(days + 719163)
    return "datetime:naive:" + String((days + 719163) * 86400000000 + remainder)


def _object_key(value: PythonObject, context: PythonObject) raises -> String:
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if cls is b.list or cls is b.tuple:
        raise Error("UMAP categorical y must be one label per row")
    if value is None:
        return "none"
    if cls is b.int or cls is b.bool:
        return _integer_key(value)
    if cls is b.float:
        var bits = bitcast[DType.uint64](Float64(py=value))
        return _real_key(_scalar_ieee(bits, 0, 52, 11), Int(value._obj_ptr))
    if cls is b.complex:
        ref cpy_complex = Python().cpython()
        var real = cpy_complex.lib.get_function[Float64]("PyComplex_RealAsDouble")(value._obj_ptr)
        var imag = cpy_complex.lib.get_function[Float64]("PyComplex_ImagAsDouble")(value._obj_ptr)
        var real_key = _real_key(_scalar_ieee(bitcast[DType.uint64](real), 0, 52, 11), Int(value._obj_ptr))
        if imag == 0:
            return real_key^
        return "complex:" + real_key + ":" + _real_key(_scalar_ieee(bitcast[DType.uint64](imag), 0, 52, 11), Int(value._obj_ptr))
    if umap_is_string(value):
        return _unicode_key(value)
    ref cpy = Python().cpython()
    var flags = cpy.lib.get_function[UInt64]("PyType_GetFlags")(cls._obj_ptr)
    if (flags & UInt64(1 << 27)) != 0 and (flags & UInt64(1 << 9)) == 0:
        return _bytes_key(value)
    var datetime = py_get(native_loaded_module("sys").modules, "_datetime")
    if datetime is not None:
        var types = _loaded_fields(native_loaded_module("_datetime"))
        if cls is py_get(types, "date"):
            return "date:" + String(_ordinal(Int(py=value.year), Int(py=value.month), Int(py=value.day)))
        if cls is py_get(types, "timedelta"):
            return "delta:" + String(Int(py=value.days)) + ":" + String(Int(py=value.seconds)) + ":" + String(Int(py=value.microseconds))
        if cls is py_get(types, "datetime") or cls is py_get(types, "time"):
            # A Python tzinfo can execute callbacks from equality/hash. The
            # native fixed-offset timezone is the supported aware-data route.
            var timezone = value.tzinfo
            if timezone is not None and py_type(timezone) is not py_get(types, "timezone"):
                raise Error("TypeError: UMAP datetime labels require naive or native fixed-offset timezone data")
            var seconds = Int(py=value.hour) * 3600 + Int(py=value.minute) * 60 + Int(py=value.second)
            var micros = Int(py=value.microsecond)
            var days = _ordinal(Int(py=value.year), Int(py=value.month), Int(py=value.day)) if cls is py_get(types, "datetime") else 0
            var prefix = String("datetime:") if cls is py_get(types, "datetime") else String("time:")
            if timezone is not None:
                var offset = py_call(value.utcoffset)
                seconds -= Int(py=offset.days) * 86400 + Int(py=offset.seconds)
                micros -= Int(py=offset.microseconds)
                prefix += "aware:"
            else:
                prefix += "naive:"
            var total = (days * 86400 + seconds) * 1000000 + micros
            return prefix + String(total)
    var scalar = _one_of(cls, context["scalar_types"])
    if not scalar and py_get(context, "scalar_type") is not None:
        var classes = sequence_dtype_classes(context)
        for i in range(len(classes)):  # small-loop(dtype: exact registered NumPy scalar identities)
            scalar = scalar or cls is classes[i][1]
    if scalar:
        var kind = String(value.dtype.kind)
        if kind == "S":
            return _bytes_key(value)
        if kind == "b" or kind == "i" or kind == "u" or kind == "f":
            return _real_key(native_scalar_buffer(value).real, Int(value._obj_ptr))
        if kind == "c":
            var number = native_scalar_buffer(value)
            var real_key = _real_key(number.real, Int(value._obj_ptr))
            if not number.imag.truth():
                return real_key^
            return "complex:" + real_key + ":" + _real_key(number.imag, Int(value._obj_ptr))
    raise Error("TypeError: UMAP categorical labels require native strings, bytes, real numbers or None; Python hashing/comparison protocols are not supported")


def _sort_domain(value: PythonObject, key: String) raises -> String:
    # Only admitted values reach this metadata step. A native sort is
    # unnecessary for equality labels, but its historical error domains are
    # observable and must remain: distinct builtin complex values, and mixed
    # aware/naive native datetime or time values of one exact type.
    if py_type(value) is native_loaded_module("builtins").complex:
        return "complex"
    if key.startswith("datetime:"):
        return "datetime"
    if key.startswith("time:"):
        return "time"
    return ""


def _label_code(code: Int) raises -> Float32:
    # Every target consumer is a storage copy or ug_categorical_weight:
    # exact equality and exact -1 only, on both device and host. Finite normal
    # bit patterns avoid float32 rounding collisions and vendor FTZ behavior.
    # Positive plus negative normals cover the device graph's Int32 row
    # bound; skip the single reserved -1 word on the negative side.
    # No label-count refusal: past the 2 * 0x7F000000 - 1 distinct finite
    # codes the index wraps and codes collide silently, as the former
    # Float32(sorted index) coding collided past 2**24 labels.
    var span = 2 * 0x7F000000 - 1
    var index = code % span
    var bits = UInt32(0)
    if index < 0x7F000000:
        bits = UInt32(0x00800000) + UInt32(index)
    else:
        var offset = index - 0x7F000000
        if offset >= 0x3F000000:
            offset += 1
        bits = UInt32(0x80800000) + UInt32(offset)
    return bitcast[DType.float32](bits)


def _word(pointer: MutPointer[UInt8, MutUntrackedOrigin], offset: Int, width: Int,
          native_order: Bool) -> UInt64:
    var result = UInt64(0)
    for i in range(width):  # small-loop(itemsize: one native Unicode code point or byte)
        var at = i if native_order else width - 1 - i
        result |= UInt64(pointer[offset + at]) << UInt64(8 * i)
    return result


def _buffer_key(pointer: MutPointer[UInt8, MutUntrackedOrigin], offset: Int, code: Int,
                width: Int, native_order: Bool, identity: Int, dtype_name: String) raises -> String:
    if code == 21 or code == 22:
        if width != 8:
            raise Error("TypeError: UMAP datetime labels require 64-bit storage")
        return _datetime_key(Int(bitcast[DType.int64](_word(pointer, offset, width, native_order))), code, dtype_name)
    if code == 18 or code == 19:
        var unit = 1 if code == 18 else 4
        if width % unit != 0:
            raise Error("TypeError: UMAP categorical string itemsize is invalid")
        var count = width // unit
        while count > 0 and _word(pointer, offset + (count - 1) * unit, unit, native_order) == 0:  # driver-loop: NumPy fixed-string trailing padding
            count -= 1
        var key = String("b:") if code == 18 else String("u:")
        for i in range(count):  # driver-loop: native fixed-string label payload
            key += String(_word(pointer, offset + i * unit, unit, native_order)) + ":"
        return key
    var format = String("?") if code == 0 else String("q") if code >= 1 and code <= 10 and code % 2 == 1 else String("Q") if code >= 1 and code <= 10 else String("f") if code == 11 else String("d") if code == 12 else String("g") if code == 13 else String("e") if code == 23 else String("")
    if format.byte_length() == 0:
        raise Error("TypeError: UMAP categorical y requires primitive native label storage")
    return _real_key(native_scalar_real(pointer, offset, format, width, native_order), identity)


def umap_categorical(y: PythonObject, n: Int, context: PythonObject) raises -> Tuple[PythonObject, Int]:
    var b = native_loaded_module("builtins")
    var keys = List[String](capacity=n)
    var domains = List[String](capacity=n)
    var source = y
    if native_pandas_exact(source):
        source = _pandas_values(source)
    var cls = py_type(source)
    if cls is b.list or cls is b.tuple:
        if len(source) != n:
            raise Error("UMAP categorical y must be one label per row")
        var kind = _target_list_kind(source, context)
        if kind == 3:
            raise Error("Complex data not supported: UMAP got a complex y.")
        for i in range(n):  # cpu-route: native categorical label admission and exact equality keys
            var key = _object_key(source[i], context)
            if kind == 2 and (key.startswith("nan:") or key == "n-inf" or key == "n+inf"):
                raise Error("Input y contains NaN or infinity (UMAP).")
            domains.append(_sort_domain(source[i], key))
            keys.append(key^)
    else:
        classical_admit_value(source, context)
        var view = classical_view(source, context, "y")
        var shape = array_shape_once(view, context)
        var snapshot = array_metadata_once(view, shape^, "y", context,
                                           allow_non_native=True, allow_unsupported=True)
        var code = snapshot.dtype_number
        if code == 14 or code == 15 or code == 16:
            raise Error("Complex data not supported: UMAP got a complex y.")
        var column = len(snapshot.shape) == 2 and snapshot.shape[1] == 1
        if column:
            metric_warn(COLUMN_WARNING, _warning_category(context), context)
        if (len(snapshot.shape) != 1 and not column) or snapshot.shape[0] != n:
            raise Error("UMAP categorical y must be one label per row")
        if not _target_finite(snapshot):
            raise Error("Input y contains NaN or infinity (UMAP).")
        var width = snapshot.source.itemsize
        if code == 17 and width != 8:
            raise Error("TypeError: UMAP object label pointers require the supported 64-bit ABI")
        if code != 17 and code != 18 and code != 19 and code != 21 and code != 22 and not ((code >= 0 and code <= 13) or code == 23):
            raise Error("TypeError: UMAP categorical y requires primitive native label storage")
        var pointer = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=snapshot.source.addr)
        for i in range(n):  # cpu-route: native categorical buffer words, no tolist or Python scalar callbacks
            var offset = i * snapshot.strides[0]
            if code == 17:
                var cells = MutPointer[PyObjectPtr, MutUntrackedOrigin](unsafe_from_address=snapshot.source.addr + offset)
                var item = PythonObject(from_borrowed=cells[0])
                var key = _object_key(item, context)
                domains.append(_sort_domain(item, key))
                keys.append(key^)
            else:
                keys.append(_buffer_key(pointer, offset, code, width, snapshot.native_byte_order, i, snapshot.dtype_name))
                domains.append("")
        _ = snapshot
        _ = view
    var output = family_array(Python.tuple(n), "<f4", context)
    var address = Int(py=classical_address(output))
    var pointer = FAMILY_F32P(unsafe_from_address=address)
    var codes = Dict[String, Int]()
    var complex_count = 0
    var awareness = Dict[String, Bool]()
    for i in range(n):  # cpu-route: categorical equality classes with the established -1 unknown marker
        var key = keys[i]
        if key == "unknown":
            pointer[i] = Float32(-1.0)
        else:
            if key not in codes:
                var domain = domains[i]
                if domain == "complex":
                    complex_count += 1
                    if complex_count > 1:
                        raise Error("TypeError: '<' not supported between instances of 'complex' and 'complex'")
                elif domain == "datetime" or domain == "time":
                    var aware = key.startswith(domain + ":aware:")
                    if domain in awareness and awareness[domain] != aware:
                        var noun = String("datetimes") if domain == "datetime" else String("times")
                        raise Error("TypeError: can't compare offset-naive and offset-aware " + noun)
                    awareness[domain] = aware
                codes[key] = len(codes)
            pointer[i] = _label_code(codes[key])
    _ = source
    _ = y
    return (output, address)
