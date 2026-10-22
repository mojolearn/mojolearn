# SPDX-License-Identifier: Apache-2.0
# Explicit CPU KDE keeps its existing serial kernel and fixed sum order;
# query preparation, stored-buffer ownership and result publication are native.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.classical_arrays import classical_input, classical_address, classical_is_array, classical_is_f32, classical_c_order, classical_kind
from bindings.host_archive_native import host_archive_typed
from bindings.host_inference_scalars import host_inference_scalar
from bindings.host_inference_features import host_inference_feature_names
from bindings.host_kernel_inference_native import _kernel_entry, _kernel_buffer
from bindings.host_svc_native import _svc_input


def host_kde_admit_state(fields: PythonObject, context: PythonObject) raises:
    var b = Python.import_module("builtins")
    var names = Python.tuple("kernel", "metric")
    for i in range(len(names)):  # driver-loop: fixed KDE kernel/metric names
        if py_contains(fields, names[i]) and py_type(fields[names[i]]) is not b.str:
            raise Error("TypeError: KDE kernel and metric require exact native strings")
    var bandwidth = py_get(fields, "bandwidth")
    if bandwidth is not None and py_type(bandwidth) is not b.str:
        host_inference_scalar(bandwidth, context)
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
    var bandwidth = py_call(py_type(PythonObject(Float64(0.0))), fields["bandwidth"])
    var out = host_archive_typed(Python.tuple(rows), "<f4", context["output_context"])
    # The explicit host entry has no resident-fit registry. Each call uses
    # the supplied saved buffers, as the original CPU path did.
    var entry = _kernel_entry("kde_score_samples", context)
    var result = PythonObject(None)
    try:
        result = py_call(entry, classical_address(train), classical_address(query),
            classical_address(weights) if weights is not None else PythonObject(0), classical_address(out),
            Python.list(count, rows, features, bandwidth, Int(weights is not None), Int(total)), fields["kernel"], fields["metric"])
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
