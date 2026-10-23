# SPDX-License-Identifier: Apache-2.0
# Shared native metric metadata. Numeric ABCs and warnings.warn can call
# Python even when reached through a C builtin, so neither is a runtime door.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get
from bindings.sequential_admission import _one_of
from bindings.native_numpy_metadata import native_numpy_module_fields
from bindings.classical_warnings import classical_warning, _warning_class_fields
from bindings.optimizer_public_words import stash_python_error
from bindings.native_scalar_decode import native_scalar_buffer


def metric_is_bool(value: PythonObject, context: PythonObject) raises -> Bool:
    var cls = py_type(value)
    if cls is py_type(PythonObject(True)):
        return True
    var types = py_get(context, "bool_types")
    return types is not None and _one_of(cls, types)


def metric_is_real(value: PythonObject, context: PythonObject) raises -> Bool:
    var cls = py_type(value)
    return (cls is py_type(PythonObject(0)) or cls is py_type(PythonObject(0.0))
            or metric_is_bool(value, context) or _one_of(cls, context["scalar_types"]))


def metric_is_integral_type(cls: PythonObject, context: PythonObject) raises -> Bool:
    if cls is py_type(PythonObject(True)) or _one_of(cls, context["integer_types"]):
        return True
    var types = py_get(context, "bool_types")
    return types is not None and _one_of(cls, types)


def metric_is_integral(value: PythonObject, context: PythonObject) raises -> Bool:
    return metric_is_integral_type(py_type(value), context)


def metric_float(value: PythonObject, context: PythonObject) raises -> Float64:
    var cls = py_type(value)
    if cls is py_type(PythonObject(0)) or cls is py_type(PythonObject(0.0)) or cls is py_type(PythonObject(True)):
        return Float64(py=value)
    if not _one_of(cls, context["scalar_types"]):
        raise Error("TypeError: metrics require a native real scalar without Python conversion hooks")
    return Float64(py=native_scalar_buffer(value).to_python_number(False))


def metric_integer(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var cls = py_type(value)
    if cls is py_type(PythonObject(0)) or cls is py_type(PythonObject(True)):
        return py_call(py_type(PythonObject(0)), value)
    if not metric_is_integral(value, context):
        raise Error("TypeError: metric labels require native integer storage")
    return native_scalar_buffer(value).to_python_number(True)


def metric_arguments(args: PythonObject, first: Int, context: PythonObject) raises -> PythonObject:
    # Scalar options become ordinary native primitives before comparisons.
    # Containers/buffers keep their dtype and storage; changing their leaf
    # types here would alter the existing input-dtype inference contract.
    var out = Python.list()
    for index in range(len(args)):  # driver-loop: fixed public metric arguments
        var value = args[index]
        if index >= first and _one_of(py_type(value), context["scalar_types"]):
            var scalar = native_scalar_buffer(value)
            if metric_is_bool(value, context):
                value = PythonObject(scalar.truth())
            else:
                value = scalar.to_python_number(metric_is_integral(value, context))
        py_call(out.append, value)
    return py_call(py_type(Python.tuple()), out)


def metric_warning_location(stacklevel: Int) raises -> PythonObject:
    # CPython's public frame accessors read the suspended Python caller;
    # no inspect/traceback/linecache helper or frame hook executes here.
    ref cpy = Python().cpython()
    var pointer = cpy.lib.get_function[PyObjectPtr]("PyEval_GetFrame")()
    if not pointer:
        return PythonObject(None)
    var frame = PythonObject(from_borrowed=pointer)
    for _ in range(max(0, stacklevel - 1)):  # driver-loop: requested warning attribution depth
        pointer = cpy.lib.get_function[PyObjectPtr]("PyFrame_GetBack")(frame._obj_ptr)
        if not pointer:
            break
        frame = PythonObject(from_owned=pointer)
    pointer = cpy.lib.get_function[PyObjectPtr]("PyFrame_GetCode")(frame._obj_ptr)
    if not pointer:
        raise cpy.unsafe_get_error()
    var code = PythonObject(from_owned=pointer)
    var line = cpy.lib.get_function[Int32]("PyFrame_GetLineNumber")(frame._obj_ptr)
    pointer = cpy.lib.get_function[PyObjectPtr]("PyFrame_GetGlobals")(frame._obj_ptr)
    if not pointer:
        raise cpy.unsafe_get_error()
    var fields = PythonObject(from_owned=pointer)
    var module = py_get(fields, "__name__", "<unknown>")
    if py_type(module) is not py_type(PythonObject("")):
        raise Error("TypeError: warning module names must be native strings")
    return Python.tuple(code.co_filename, Int(line), module)


def metric_warn(var message: String, category: PythonObject, context: PythonObject,
                stacklevel: Int = 2) raises:
    _ = _warning_class_fields(category)
    try:
        classical_warning(message, category, context, metric_warning_location(stacklevel))
    except error:
        if not String(error).startswith("DataConversionWarning:"):
            raise error^
        # Preserve the actual admitted warning exception, including when a
        # metric is nested inside CV rather than called at the public door.
        ref cpy = Python().cpython()
        cpy.PyErr_SetString(category._obj_ptr, message.as_c_string_slice().unsafe_ptr().as_unsafe_any_origin())
        var failure = stash_python_error()
        raise failure^


def metric_undefined(message: String, context: PythonObject) raises:
    var category = context["family_metrics_undefined"]
    var module = py_get(context["modules"], "sklearn.exceptions")
    if module is not None:
        if py_type(module) is not context["native_module_type"]:
            raise Error("TypeError: metric warning categories require native module metadata")
        category = py_get(native_numpy_module_fields(module), "UndefinedMetricWarning", category)
    metric_warn(message, category, context)
