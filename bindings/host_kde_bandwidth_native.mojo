# SPDX-License-Identifier: Apache-2.0
# KDE bandwidth conversion policy; bytes are decoded by the shared native
# scalar implementation, while warning/error timing stays at the consumer.
from std.python import Python, PythonObject
from std.memory import bitcast
from bindings.py_balanced import py_call, py_type, py_get
from bindings.native_numpy_metadata import native_numpy_dtype, native_numpy_module_fields
from bindings.native_scalar_decode import native_scalar_buffer
from bindings.classical_warnings import classical_warning
from core.native_scalar_decode import native_scalar_real
from core.native_temporal_scalar import temporal_result_kind


def kde_bandwidth_code(value: PythonObject, context: PythonObject) raises -> String:
    var cls = py_type(value)
    if py_get(context, "ndarray") is None:
        return String("")
    var constructor = native_numpy_dtype(context)
    var array = cls is context["ndarray"]
    var codes = Python.tuple("?", "b", "B", "h", "H", "i", "I", "l", "L", "q", "Q", "e", "f", "d", "g", "F", "D", "G", "M", "m")
    for i in range(len(codes)):  # driver-loop: exact built-in numeric and temporal scalar/dtype identities
        var dtype = py_call(constructor, codes[i])
        if cls is dtype.type or (array and py_type(value.dtype) is py_type(dtype)):
            return String(codes[i])
    return String("")


def kde_bandwidth_warning(message: String, category: PythonObject, context: PythonObject) raises:
    var frame = py_call(context["classical_getframe"], 1)
    var module = py_get(frame.f_globals, "__name__", "")
    if py_type(module) is not py_type(PythonObject("")):
        module = PythonObject("")
    classical_warning(message, category, context,
        Python.tuple(frame.f_code.co_filename, frame.f_lineno, module), "HostInferenceWarning")


def kde_numpy_release(context: PythonObject) raises -> Int:
    var numpy = py_get(context["modules"], "numpy")
    if numpy is None or py_type(numpy) is not context["native_module_type"]:
        raise Error("TypeError: KDE scalar conversion requires native NumPy metadata")
    var version = py_get(native_numpy_module_fields(numpy), "__version__")
    if py_type(version) is not py_type(PythonObject("")):
        raise Error("TypeError: KDE scalar conversion requires native NumPy version text")
    var parts = String(version).split(".")
    if len(parts) < 2:
        raise Error("TypeError: KDE scalar conversion requires NumPy major.minor metadata")
    return Int(String(parts[0])) * 100 + Int(String(parts[1]))


def kde_bandwidth_array_policy(value: PythonObject, context: PythonObject) raises:
    if py_type(value) is context["ndarray"]:
        var ndim = Int(py=value.ndim)
        if ndim > 0:
            if kde_numpy_release(context) >= 204:
                raise Error("TypeError: only 0-dimensional arrays can be converted to Python scalars")
        if Int(py=value.size) != 1:
            raise Error("TypeError: only size-1 arrays can be converted to Python scalars")
        if ndim > 0:
            kde_bandwidth_warning("Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)",
                context["warning_deprecation_type"], context)


def kde_temporal_bandwidth(value: PythonObject, code: String, context: PythonObject) raises -> PythonObject:
    var dtype = value.dtype
    var text = String(dtype.str)
    var parts = text.split("[")
    var unit = String("generic")
    var multiplier = 1
    if len(parts) == 2:
        var tail = String(parts[1])
        var length = tail.byte_length() - 1
        var start = 0
        while start < length:  # driver-loop: parse exact native dtype unit metadata
            var digit = String(tail[byte=start:start + 1])
            if digit < "0" or digit > "9":
                break
            start += 1
        if start > 0:
            multiplier = Int(String(tail[byte=0:start]))
        unit = String(tail[byte=start:length])
    # Acquire only an admitted exact native temporal owner. Public no-format
    # buffers support byte-swapped temporals even where memoryview formatting
    # does not. Both storage allocations precede acquisition; every export
    # is released before decoding or raising a conversion error.
    var view = List[Int](length=10, fill=0)
    var bytes = List[UInt8](length=8, fill=UInt8(0))
    var pointer = view.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var data = bytes.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    ref cpy = Python().cpython()
    var status = cpy.lib.get_function[Int32]("PyObject_GetBuffer")(value._obj_ptr, pointer, Int32(0x118))
    if status != 0:
        raise cpy.unsafe_get_error()
    try:
        if view[2] != 8 or view[3] != 8:
            raise Error("TypeError: temporal scalar decoding requires one native element")
        status = cpy.lib.get_function[Int32]("PyBuffer_ToContiguous")(data, pointer, 8, Int8(67))
        if status != 0:
            raise cpy.unsafe_get_error()
    finally:
        cpy.lib.get_function[NoneType]("PyBuffer_Release")(pointer)
    var little = String(dtype.byteorder) != ">"
    var word = UInt64(0)
    for i in range(8):  # driver-loop: one signed native temporal word, either byte order
        word |= UInt64(bytes[i]) << UInt64(8 * (i if little else 7 - i))
    var kind = temporal_result_kind(Int(bitcast[DType.int64](word)), code == "M", unit,
                                   multiplier, kde_numpy_release(context) < 202)
    var number = native_scalar_real(data, 0, String("q"), 8, little)
    _ = bytes^
    _ = view^
    if kind != 0:
        var name = String("NoneType") if kind == 1 else String("datetime.date") if kind == 2 else String("datetime.datetime") if kind == 3 else String("datetime.timedelta")
        raise Error("TypeError: float() argument must be a string or a real number, not '" + name + "'")
    return PythonObject(bitcast[DType.float64](number.float64_bits()))


def kde_native_bandwidth(value: PythonObject, code: String, context: PythonObject) raises -> PythonObject:
    kde_bandwidth_array_policy(value, context)
    if code == "M" or code == "m":
        return kde_temporal_bandwidth(value, code, context)
    if code != "F" and code != "D" and code != "G":
        return native_scalar_buffer(value).to_python_number(False)
    if py_type(value) is context["ndarray"]:
        # NumPy boxes F/D ndarray elements as builtin complex, whose float
        # conversion refuses them. G retains an extended complex scalar even
        # on ABIs where its component width happens to equal double.
        if code != "G":
            raise Error("TypeError: float() argument must be a string or a real number, not 'complex'")
    var exceptions = py_get(context["modules"], "numpy.exceptions")
    if exceptions is None or py_type(exceptions) is not context["native_module_type"]:
        raise Error("TypeError: complex scalar warnings require NumPy's native exception metadata")
    var category = py_get(native_numpy_module_fields(exceptions), "ComplexWarning")
    kde_bandwidth_warning("Casting complex values to real discards the imaginary part", category, context)
    return native_scalar_buffer(value).to_python_number(False)
