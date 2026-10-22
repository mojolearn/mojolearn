# SPDX-License-Identifier: Apache-2.0
# Explicit CPU KDE keeps its existing serial kernel and fixed sum order;
# query preparation, stored-buffer ownership and result publication are native.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.classical_attributes import classical_getattribute
from bindings.classical_pandas import _pd_type_fields
from bindings.native_numpy_metadata import native_numpy_dtype
from bindings.classical_arrays import classical_input, classical_address, classical_is_array, classical_is_f32, classical_c_order, classical_kind
from bindings.host_archive_native import host_archive_typed
from bindings.host_inference_scalars import host_inference_scalar, host_inference_primitive_number
from bindings.host_inference_features import host_inference_feature_names
from bindings.host_kernel_inference_native import _kernel_entry, _kernel_buffer
from bindings.host_svc_native import _svc_input


def _kde_option_value(value: PythonObject, context: PythonObject, numeric: Bool = False) raises -> PythonObject:
    var cls = py_type(value)
    var native_str = py_type(PythonObject(""))
    var native_bytes = context["host_kde_bytes_type"]
    var numpy_str = PythonObject(None)
    var numpy_bytes = PythonObject(None)
    if py_get(context, "ndarray") is not None:
        var constructor = native_numpy_dtype(context)
        numpy_str = py_call(constructor, "U").type
        numpy_bytes = py_call(constructor, "S").type
    var text = cls is native_str or cls is numpy_str
    if not text and py_type(cls) is py_type(native_str):
        var mro = py_call(py_type(native_str).__dict__["__mro__"].__get__, cls)
        for i in range(len(mro)):  # driver-loop: native string ancestry without value protocols
            if mro[i] is native_str:
                text = True
                break
        if text:
            var native_string_slot = native_str.__dict__["__str__"]
            for i in range(len(mro)):  # driver-loop: refuse overridden string/numeric hooks before conversion
                if mro[i] is native_str:
                    break
                var fields = _pd_type_fields(mro[i])
                var slot = py_get(fields, "__str__")
                if (slot is not None and slot is not native_string_slot) or py_contains(fields, "__del__") or (numeric and (py_contains(fields, "__float__") or py_contains(fields, "__index__"))):
                    raise Error("TypeError: KDE option strings cannot invoke Python conversion hooks")
    if text:
        # CPython copies Unicode subclass contents directly. Neither __str__
        # nor another Python value protocol participates in this conversion.
        ref cpy = Python().cpython()
        var result = cpy.lib.get_function[PyObjectPtr]("PyUnicode_FromObject")(value._obj_ptr)
        if not result:
            raise cpy.unsafe_get_error()
        return PythonObject(from_owned=result)
    if cls is native_bytes or cls is numpy_bytes or cls is py_get(context, "host_kde_bytearray_type"):
        # The old numeric entry stringified byte options before reporting its
        # unknown kernel/metric error; bandwidth instead parsed their bytes.
        return value if numeric else py_call(native_str, value)
    if numeric:
        host_inference_scalar(value, context)
        return value
    raise Error("TypeError: KDE kernel and metric require native strings without Python conversion hooks")


def host_kde_admit_state(fields: PythonObject, context: PythonObject) raises:
    var names = Python.tuple("kernel", "metric")
    for i in range(len(names)):  # driver-loop: fixed KDE kernel/metric names
        if py_contains(fields, names[i]):
            _ = _kde_option_value(fields[names[i]], context)
    var bandwidth = py_get(fields, "bandwidth")
    if bandwidth is not None:
        _ = _kde_option_value(host_inference_primitive_number(bandwidth, context), context, True)
    var buffers = Python.tuple("_x", "_w")
    for i in range(len(buffers)):  # driver-loop: fixed saved KDE buffers
        var value = py_get(fields, buffers[i])
        if py_type(value) is py_get(context, "ndarray") and classical_kind(value, context) == "O":
            raise Error("TypeError: KDE saved buffers require native numeric storage")
    # An exact string was accepted by the old score's float(bandwidth).
    # Its native float conversion happens below; custom conversion hooks
    # never enter that constructor.


def host_kde_inference(model: PythonObject, fields: PythonObject, source: PythonObject, total: Bool,
                       context: PythonObject, public: Bool) raises -> PythonObject:
    if not py_contains(fields, "_x"):
        raise Error("KernelDensity requires fit before scoring")
    var query = _svc_input(source, context)
    var rows = Int(py=query.shape[0])
    var features = Int(py=fields["n_features_in_"])
    if Int(py=query.shape[1]) != features:
        raise Error("KernelDensity query width differs from fitted dimensions")
    if not Bool(py=py_get(fields, "_mojolearn_fitting", False)):
        host_inference_feature_names(model, fields, context)
    var train = classical_input(fields["_x"], context, "KDE fitted samples")
    if Int(py=train.ndim) != 2 or Int(py=train.shape[1]) != features:
        raise Error("KernelDensity fitted sample shape differs from fitted dimensions")
    if not classical_is_f32(train, context) or (not classical_is_array(train, context) and not Bool(py=train.dtype.isnative)):
        raise Error("KernelDensity fitted samples require native float32 storage")
    var f_order = String(train.order) == "F" if classical_is_array(train, context) else Bool(py=train.flags.f_contiguous)
    if not classical_c_order(train, context) and not f_order:
        raise Error("KernelDensity fitted samples require contiguous storage")
    var count = Int(py=train.shape[0])
    if features < 1 or count < 1 or count > 2147483647 // features:
        raise Error("KernelDensity fitted dimensions exceed native indexing")
    var weight = py_get(fields, "_w")
    var weights = PythonObject(None)
    if weight is not None:
        weights = _kernel_buffer(weight, count, "<f4", context)
    # Derive the fixed native constructor from a C-created scalar. A patched
    # builtins.float must never install a callback on this numerical boundary.
    var bandwidth_value = host_inference_primitive_number(
        classical_getattribute(model, PythonObject("bandwidth"), context), context)
    bandwidth_value = _kde_option_value(bandwidth_value, context, True)
    var bandwidth = py_call(py_type(PythonObject(Float64(0.0))), bandwidth_value)
    var kernel = _kde_option_value(classical_getattribute(model, PythonObject("kernel"), context), context)
    var metric = _kde_option_value(classical_getattribute(model, PythonObject("metric"), context), context)
    var out = host_archive_typed(Python.tuple(rows), "<f4", context["output_context"])
    # The explicit host entry has no resident-fit registry. Each call uses
    # the supplied saved buffers, as the original CPU path did.
    var entry = _kernel_entry("kde_score_samples", context)
    var result = PythonObject(None)
    try:
        result = py_call(entry, classical_address(train), classical_address(query),
            classical_address(weights) if weights is not None else PythonObject(0), classical_address(out),
            Python.list(count, rows, features, bandwidth, Int(weights is not None), Int(total)), kernel, metric)
    except error:
        # The public validation wrapper normalizes only plain native
        # nonfinite errors; other CPU kernel failures remain builtin Exception.
        var text = String(error)
        if ("NaN" in text or "Infinit" in text or "infinit" in text
                or "nonfinite" in text or "non-finite" in text
                or "not finite" in text or "requires finite" in text or "must be finite" in text):
            var name = py_call(Python.import_module("builtins").type.__dict__["__name__"].__get__, py_type(model))
            raise Error("Input X contains NaN or infinity (" + String(name) + ": " + text + ")")
        raise Error("HostKDEKernelError: " + text)
    _ = train^
    _ = query^
    _ = weights^
    if total:
        return result
    return classical_input(out, context, "log density") if public and py_get(context, "ndarray") is not None else out
