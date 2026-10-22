# SPDX-License-Identifier: Apache-2.0
# KDE bandwidth conversion policy; bytes are decoded by the shared native
# scalar implementation, while warning/error timing stays at the consumer.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get
from bindings.native_numpy_metadata import native_numpy_dtype, native_numpy_module_fields
from bindings.native_scalar_decode import native_scalar_buffer
from bindings.classical_warnings import classical_warning


def kde_complex_code(value: PythonObject, context: PythonObject) raises -> String:
    var cls = py_type(value)
    if py_get(context, "ndarray") is None:
        return String("")
    var constructor = native_numpy_dtype(context)
    var array = cls is context["ndarray"]
    var codes = Python.tuple("F", "D", "G")
    for i in range(len(codes)):  # driver-loop: exact built-in complex scalar/dtype identities
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


def kde_complex_bandwidth(value: PythonObject, code: String, context: PythonObject) raises -> PythonObject:
    if py_type(value) is context["ndarray"]:
        var ndim = Int(py=value.ndim)
        if ndim > 0:
            var numpy = py_get(context["modules"], "numpy")
            if numpy is None or py_type(numpy) is not context["native_module_type"]:
                raise Error("TypeError: KDE scalar conversion requires native NumPy metadata")
            var version = py_get(native_numpy_module_fields(numpy), "__version__")
            if py_type(version) is not py_type(PythonObject("")):
                raise Error("TypeError: KDE scalar conversion requires native NumPy version text")
            var parts = String(version).split(".")
            if len(parts) < 2:
                raise Error("TypeError: KDE scalar conversion requires NumPy major.minor metadata")
            if Int(String(parts[0])) * 100 + Int(String(parts[1])) >= 204:
                raise Error("TypeError: only 0-dimensional arrays can be converted to Python scalars")
        if Int(py=value.size) != 1:
            raise Error("TypeError: only size-1 arrays can be converted to Python scalars")
        if ndim > 0:
            kde_bandwidth_warning("Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)",
                context["warning_deprecation_type"], context)
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
