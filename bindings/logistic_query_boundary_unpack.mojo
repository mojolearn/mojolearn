# SPDX-License-Identifier: Apache-2.0
"""Exact public feature-field UNPACK for native Logistic query decisions."""
from std.memory import bitcast
from std.python import Python, PythonObject
from std.python._cpython import ExternalFunction, PyObjectPtr
from core.shared_slot import SharedSlot
from core.native_scalar_decode import NativeRealBits, native_scalar_real, _scalar_ieee
from core.native_array_layout import native_array_itemsize
from core.logistic_query_state import LogisticQueryWidth, logistic_width_integer
from core.logistic_query_boundary_run import logistic_torch_width_run
from core.sequence_array_cast import SEQ_CAST_OVERFLOW
from bindings.cpython_table import cpython_table
from bindings.py_balanced import py_call, py_type, py_get
from bindings.native_scalar_decode_unpack import native_scalar_buffer_dtype, native_scalar_buffer_bytes
from bindings.native_scalar_promotion_unpack import native_scalar_weak_promotion
from bindings.native_numpy_metadata_unpack import native_numpy_dtype
from bindings.native_torch_admission import native_torch_exact, native_torch_scalar
from bindings.optimizer_public_words_unpack import stash_python_error
from bindings.sequence_array_native_unpack import sequence_array_release
from bindings.sequence_array_warnings_unpack import sequence_array_warn, sequence_array_report
from bindings.sequence_array_warnings_pack import sequence_array_errstate

from bindings.logistic_query_boundary_pack import (
    _logistic_width_item,
    _logistic_width_numpy_dtype,
)


comptime _LogisticComplexReal = ExternalFunction[
    "PyComplex_RealAsDouble", def(PyObjectPtr) thin abi("C") -> Float64]
comptime _LogisticComplexImag = ExternalFunction[
    "PyComplex_ImagAsDouble", def(PyObjectPtr) thin abi("C") -> Float64]


struct LogisticWidthApi(Defaultable, Movable):
    var real: _LogisticComplexReal.type
    var imag: _LogisticComplexImag.type

    def __init__(out self):
        ref cpy = Python().cpython()
        self.real = _LogisticComplexReal.load(cpy.lib.borrow())
        self.imag = _LogisticComplexImag.load(cpy.lib.borrow())


comptime LOGISTIC_WIDTH_API = SharedSlot[LogisticWidthApi, "MojolearnLogisticWidthApiV1"]


def _logistic_width_builtin(value: PythonObject, context: PythonObject) raises -> LogisticQueryWidth:
    # Only exact builtin scalar classes reach their C conversion slots.
    var cls = py_type(value)
    if cls is py_type(PythonObject(0)) or cls is py_type(PythonObject(True)):
        var overflow = Array[Int32, 1](fill=0)
        var number = cpython_table()[].PyLong_AsLongLongAndOverflow(value._obj_ptr, Int(overflow.unsafe_ptr()))
        if cpython_table()[].PyErr_Occurred():
            var failure = stash_python_error()
            raise failure^
        if overflow[0] != 0:
            return LogisticQueryWidth(kind=1)
        return LogisticQueryWidth(logistic_width_integer(Int(number)), kind=3, integer_value=Int(number))
    if cls is py_type(PythonObject(0.0)):
        return LogisticQueryWidth(_scalar_ieee(bitcast[DType.uint64](Float64(py=value)), 0, 52, 11))
    if cls is context["lr_complex_type"]:
        var api = LOGISTIC_WIDTH_API.get_or_create_ptr()
        var real = api[].real(value._obj_ptr)
        var imag = api[].imag(value._obj_ptr)
        if cpython_table()[].PyErr_Occurred():
            var failure = stash_python_error()
            raise failure^
        return LogisticQueryWidth(_scalar_ieee(bitcast[DType.uint64](real), 0, 52, 11),
                                  _scalar_ieee(bitcast[DType.uint64](imag), 0, 52, 11))
    return LogisticQueryWidth(kind=1)


def _logistic_width_array(value: PythonObject, context: PythonObject) raises -> LogisticQueryWidth:
    # Native Array scalar comparison requires exact representability, unlike
    # NumPy's weak scalar promotion. Read its admitted fixed slots directly.
    var slots = context["array_fields"]
    if Int(py=py_call(slots["size"], value)) != 1:
        raise Error("ValueError: mojolearn: the truth value of an Array with more than one element is ambiguous")
    var dtype = String(py_call(slots["dtype"], value))
    var width = native_array_itemsize(dtype)
    var code = String("q") if dtype[byte=1] == "i" else String("Q")
    if dtype[byte=1] == "f":
        code = String("e") if width == 2 else (String("f") if width == 4 else String("d"))
    var address = Int(py=py_call(slots["_addr"], value))
    if address <= 0 or address > 9223372036854775807 - width:
        raise Error("ValueError: LogisticRegression feature metadata has invalid scalar storage")
    var data = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
    var result = LogisticQueryWidth(native_scalar_real(data, 0, code, width, True))
    _ = value
    return result


def logistic_query_width(value: PythonObject, context: PythonObject) raises -> LogisticQueryWidth:
    var current = value
    for depth in range(65):  # driver-loop: bounded object-array metadata unboxing, no row traversal
        var cls = py_type(current)
        if (cls is py_type(PythonObject(0)) or cls is py_type(PythonObject(True)) or
                cls is py_type(PythonObject(0.0)) or cls is context["lr_complex_type"]):
            return _logistic_width_builtin(current, context)
        if cls is context["array_class"]:
            return _logistic_width_array(current, context)
        if native_torch_exact(cls):
            var unpacked = native_torch_scalar(current)
            var result = _logistic_width_builtin(unpacked[0], context)
            var dtype = unpacked[1].copy()
            result = logistic_torch_width_run(result, dtype)
            _ = unpacked^
            return result
        var array = cls is py_get(context, "ndarray")
        if py_get(context, "ndarray") is None:
            return LogisticQueryWidth(kind=1)
        var dtype = _logistic_width_numpy_dtype(current, context, array)
        if dtype is None:
            # Exact builtins/ordinary containers do not compare equal to a
            # machine feature count. Unrecognized object-array elements are
            # refused before invoking their equality/conversion hooks.
            if depth != 0:
                var allowed = cls is py_type(Python.list()) or cls is py_type(Python.tuple())
                var types = context["native_value_types"]
                for i in range(len(types)):  # driver-loop: exact connected nonnumeric owners, no element comparison
                    allowed = allowed or cls is types[i]
                types = context["mapping_types"]
                for i in range(len(types)):  # driver-loop: exact native mapping owners
                    allowed = allowed or cls is types[i]
                if not allowed:
                    raise Error("TypeError: LogisticRegression feature metadata cannot invoke Python equality hooks")
            return LogisticQueryWidth(kind=1)
        var kind = String(dtype.kind)
        if kind == "V":
            raise Error("TypeError: Cannot compare structured or void to non-void arrays.")
        if array:
            var count = Int(py=current.size)
            if count != 1:
                if count == 0 and sequence_array_release(context) < 202:
                    sequence_array_warn("The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.", context["sequence_array_warnings"], "DeprecationWarning")
                    return LogisticQueryWidth(kind=2)
                if count == 0:
                    raise Error("ValueError: The truth value of an empty array is ambiguous. Use `array.size > 0` to check that an array is not empty.")
                raise Error("ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()")
            if kind == "O" or kind == "T":
                var item = _logistic_width_item(current)
                current = item^
                continue
        if kind == "S" or kind == "U" or kind == "M":
            return LogisticQueryWidth(kind=1)
        if kind == "m":
            if sequence_array_release(context) >= 205:
                sequence_array_warn("The 'generic' unit for NumPy timedelta is deprecated, and will raise an error in the future. This includes implicit conversion of bare integers (e.g. `+ 1`). Please use a specific unit instead.", context["sequence_array_warnings"], "DeprecationWarning")
            var bytes = native_scalar_buffer_bytes(current, 8)
            var pointer = bytes.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
            var result = LogisticQueryWidth(native_scalar_real(pointer, 0, "q", 8, String(dtype.byteorder) != ">"))
            _ = bytes^
            return result
        if kind != "b" and kind != "i" and kind != "u" and kind != "f" and kind != "c":
            raise Error("TypeError: LogisticRegression feature metadata requires native scalar values")
        var scalar = native_scalar_buffer_dtype(current, dtype)
        var result = LogisticQueryWidth(scalar.real, scalar.imag)
        if (kind == "f" or kind == "c") and native_scalar_weak_promotion(context, "LogisticRegression feature metadata"):
            var width = Int(py=dtype.itemsize) // (2 if kind == "c" else 1)
            if width <= 8:
                result.precision = width * 8
                result.numpy_cast = True
        _ = current
        return result
    raise Error("TypeError: LogisticRegression feature metadata exceeds native nesting bounds")


def logistic_query_width_check(features: Int, value: PythonObject, context: PythonObject) raises:
    var width = logistic_query_width(value, context)
    if width.cast_overflow(features):
        var settings = sequence_array_errstate(context)
        sequence_array_report(SEQ_CAST_OVERFLOW, settings, context)
    width.validate(features)
    _ = value
