# SPDX-License-Identifier: Apache-2.0
# Native public parallel output boundary; no Python format/conversion helper.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_arrays import classical_host_f32, classical_is_array, classical_typed, classical_address
from bindings.classical_labels import labels_cast
from bindings.scaler_lifecycle_native import _outside
from bindings.sequential_inputs import sequential_input


def _output(value: PythonObject, widen: Bool, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(value) is py_get(context, "ndarray"):
        if widen and String(value.dtype.kind) == "f" and Int(py=value.dtype.itemsize) < 8:
            var data = classical_host_f32(value, Int(py=value.ndim), "result", context)
            var wide = classical_typed(data.shape, "<f8", context)
            labels_cast(Int(py=classical_address(data)), "<f4", Int(py=classical_address(wide)), "<f8", Int(py=data.size), context)
            return wide
        return value
    if classical_is_array(value, context):
        var result = value
        if widen and String(result.dtype) == "<f4":
            var wide = classical_typed(result.shape, "<f8", context)
            labels_cast(Int(py=classical_address(result)), "<f4", Int(py=classical_address(wide)), "<f8", Int(py=result.size), context)
            _ = result
            result = wide
        return sequential_input(result, context)
    if py_type(value) is b.tuple or (py_type(value) is b.list and len(value) > 0 and classical_is_array(value[0], context)):
        var result = Python.list()
        for i in range(len(value)):  # driver-loop: completed multi-output buffers
            _ = py_call(result.append, _output(value[i], False, context))
        return py_call(b.tuple, result) if py_type(value) is b.tuple else result
    return value


def parallel_public_result(result: PythonObject, method: String, kind: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    if py_get(context, "ndarray") is None or not _outside(context, 1):
        return result
    var widen = (kind is not None and String(kind) == "regressor" and method == "predict")
    widen = widen or (kind is not None and String(kind) == "outlier_detector" and (method == "decision_function" or method == "score_samples"))
    return _output(result, widen, context)

