# SPDX-License-Identifier: Apache-2.0
# Native public parallel output boundary; no Python format/conversion helper.
# UNPACK and PACK only: the widening rule and the cast are typed, in
# bindings/parallel_public_result_run.mojo.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_arrays_door import classical_is_array, classical_typed, classical_address
from bindings.admitted_input_unpack import admit_x
from bindings.parallel_public_result_run import parallel_result_widens, parallel_result_widen
from bindings.scaler_lifecycle_native_door import _outside
from bindings.sequential_inputs_door import sequential_input


def _output(value: PythonObject, widen: Bool, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(value) is py_get(context, "ndarray"):
        if widen and String(value.dtype.kind) == "f" and Int(py=value.dtype.itemsize) < 8:
            var data = admit_x(value, Int(py=value.ndim), "result", context)
            var wide = classical_typed(data.shape, "<f8", context)
            parallel_result_widen(Int(py=classical_address(data)), Int(py=classical_address(wide)), Int(py=data.size))
            _ = data  # keep the admitted block alive through the cast
            return wide
        return value
    if classical_is_array(value, context):
        var result = value
        if widen and String(result.dtype) == "<f4":
            var wide = classical_typed(result.shape, "<f8", context)
            parallel_result_widen(Int(py=classical_address(result)), Int(py=classical_address(wide)), Int(py=result.size))
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
    # predict has one result for every caller (Andrew, 2026-11-22, lane
    # predict-dtype): the kernel's dtype as a NumPy view, with no caller
    # test and no float64 widening. The other results keep the caller test.
    var predicting = method == "predict"
    if py_get(context, "ndarray") is None or (not predicting and not _outside(context, 1)):
        return result
    return _output(result, parallel_result_widens(String("") if kind is None else String(kind), method), context)

