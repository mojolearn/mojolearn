# SPDX-License-Identifier: Apache-2.0
"""Exact native truth admission for decomposition public options."""
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_get, py_type
from bindings.host_inference_scalars import host_inference_scalar
from bindings.decomp_scalar_admission import decomp_scalar_admit
from bindings.native_numpy_metadata import native_numpy_dtype


def decomp_truth(value: PythonObject, context: PythonObject) raises -> Bool:
    # Truth uses length only for exact builtin containers; no element is
    # inspected. Keep those previously accepted values without a protocol.
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if (cls is b.list or cls is b.tuple or cls is b.dict or cls is b.set
            or cls is b.frozenset or cls is b.range or cls is b.complex):
        return Bool(py=value)
    try:
        host_inference_scalar(value, context, True)
    except failure:
        var ndarray = py_get(context, "ndarray")
        if ndarray is not None:
            var dtype = native_numpy_dtype(context)
            if cls is ndarray and Int(py=value.size) == 1:
                # Native text truth also accepts one item at any rank.
                for code in ["U", "S"]:  # driver-loop: native text dtype classes
                    if py_type(value.dtype) is py_type(py_call(dtype, code)):
                        return Bool(py=value)
            else:
                for code in ["F", "D", "G"]:  # driver-loop: exact native complex scalar types
                    if cls is py_call(dtype, code).type:
                        return Bool(py=value)
        decomp_scalar_admit(value, context)
    return Bool(py=value)

