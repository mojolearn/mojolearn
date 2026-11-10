# SPDX-License-Identifier: Apache-2.0
"""Exact public feature-field UNPACK for native Logistic query decisions."""
from std.memory import bitcast
from std.python import Python, PythonObject
from std.python._cpython import ExternalFunction, PyObjectPtr
from core.shared_slot import SharedSlot
from core.native_scalar_decode import NativeRealBits, native_scalar_real, _scalar_ieee
from core.native_array_layout import native_array_itemsize
from core.logistic_query_state import LogisticQueryWidth, logistic_width_integer
from core.sequence_array_cast import SEQ_CAST_OVERFLOW
from bindings.cpython_table import cpython_table
from bindings.py_balanced import py_call, py_type, py_get
from bindings.native_scalar_decode import native_scalar_buffer_dtype, native_scalar_buffer_bytes
from bindings.native_scalar_promotion import native_scalar_weak_promotion
from bindings.native_numpy_metadata import native_numpy_dtype
from bindings.native_torch_admission import native_torch_exact, native_torch_scalar
from bindings.optimizer_public_words import stash_python_error
from bindings.sequence_array_native import sequence_array_release
from bindings.sequence_array_warnings import sequence_array_warn, sequence_array_errstate, sequence_array_report

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
        var overflow = InlineArray[Int32, 1](fill=0)
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


def _logistic_width_numpy_dtype(value: PythonObject, context: PythonObject,
                                array: Bool) raises -> PythonObject:
    # Descriptor identities, not user dtype attributes, admit the storage.
    # Ordinary fitted integer fields never enter this exceptional metadata
    # path. The existing native dtype constructor supplies immutable types.
    var cls = py_type(value)
    var scalar = False
    var classes = context["scalar_types"]
    for i in range(len(classes)):  # driver-loop: fixed connected primitive scalar classes
        scalar = scalar or cls is classes[i]
    if scalar:
        # An exact connected primitive scalar has its immutable native dtype;
        # do not reconstruct a dtype or rescan its type before each query.
        return value.dtype
    var constructor = native_numpy_dtype(context)
    if not array and not scalar:
        for code in ["q", "Q", "g", "F", "D", "G", "M8", "m8", "S", "U", "V"]:  # driver-loop: native scalar aliases exposed by object-array UNPACK
            scalar = scalar or cls is py_call(constructor, code).type
    if not array and not scalar:
        return PythonObject(None)
    var dtype = value.dtype
    if scalar:
        return dtype
    var numeric = context["numeric_dtypes"]
    if py_type(numeric) is not py_type(Python.dict()):
        raise Error("TypeError: LogisticRegression requires connected native dtype metadata")
    var position = Int(0)
    var key = PyObjectPtr()
    var item = PyObjectPtr()
    while cpython_table()[].PyDict_Next(numeric._obj_ptr, Int(MutPointer(to=position)), Int(MutPointer(to=key)), Int(MutPointer(to=item))) != 0:  # driver-loop: retained primitive dtype owners, no list or dtype allocation
        var known = PythonObject(from_borrowed=item)
        if py_type(dtype) is py_type(known):
            return dtype
    for code in ["q", "Q", "g", "F", "D", "G", "O", "S", "U", "V", "M8", "m8"]:  # driver-loop: remaining native dtype identities, never element data
        if py_type(dtype) is py_type(py_call(constructor, code)):
            return dtype
    if sequence_array_release(context) >= 200:
        if py_type(dtype) is py_type(py_call(constructor, "T")):
            return dtype
    raise Error("TypeError: LogisticRegression feature metadata requires a built-in native dtype")


def _logistic_width_item(value: PythonObject) raises -> PythonObject:
    # Exact ndarray native indexing unpacks the one element. No array method
    # or user scalar comparison executes; the returned owner is admitted on
    # the next bounded walk iteration before any conversion can use it.
    var rank = Int(py=value.ndim)
    ref cpy = Python().cpython()
    var index = cpy.PyTuple_New(rank)
    if not index:
        raise cpy.unsafe_get_error()
    var owner = PythonObject(from_owned=index)
    for i in range(rank):  # driver-loop: one all-zero index, bounded by ndarray rank
        var zero = PythonObject(0)
        if cpy.PyTuple_SetItem(index, i, zero.steal_data()) != 0:
            raise cpy.unsafe_get_error()
    var pointer = cpython_table()[].PyObject_GetItem(value._obj_ptr, owner._obj_ptr)
    _ = owner^
    if not pointer:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=pointer)


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
            if result.kind == 3:
                result.kind = 0  # the tensor dtype controls weak scalar conversion
            if dtype == "torch.float16" or dtype == "torch.complex32":
                result.precision = 16
            elif dtype == "torch.bfloat16":
                result.precision = 8
            elif dtype == "torch.float32" or dtype == "torch.complex64":
                result.precision = 32
            elif dtype == "torch.float64" or dtype == "torch.complex128":
                result.precision = 64
            elif dtype == "torch.uint8" or dtype == "torch.uint16" or dtype == "torch.uint32" or dtype == "torch.uint64":
                result.integer_bits = 8 if dtype == "torch.uint8" else (16 if dtype == "torch.uint16" else (32 if dtype == "torch.uint32" else 64))
            elif dtype == "torch.int8" or dtype == "torch.int16" or dtype == "torch.int32" or dtype == "torch.int64":
                result.integer_bits = 8 if dtype == "torch.int8" else (16 if dtype == "torch.int16" else (32 if dtype == "torch.int32" else 64))
                result.integer_signed = True
            elif dtype != "torch.bool":
                raise Error("TypeError: LogisticRegression feature metadata requires a primitive numeric Tensor")
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
