# SPDX-License-Identifier: Apache-2.0
"""Native scalar admission for decomposition boundaries using int/float conversion.

This admits owners; the caller keeps its existing conversion and error order.
PCA's snapshot parser and integer-index query contract are separate contracts.
"""
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_get, py_type
from bindings.host_inference_scalars import host_inference_scalar
from bindings.native_loaded_module import native_loaded_module
from bindings.native_numpy_metadata import native_numpy_dtype


def decomp_scalar_admit(value: PythonObject, context: PythonObject) raises:
    try:
        host_inference_scalar(value, context)
    except failure:
        var ndarray = py_get(context, "ndarray")
        if ndarray is not None:
            var cls = py_type(value)
            var constructor = native_numpy_dtype(context)
            if cls is ndarray:
                # Exact native text storage has no object elements or user
                # conversion slots. Preserve int/float's existing zero-D
                # text conversion (and parse errors) just as for str/bytes.
                # Numeric zero-D arrays were admitted by the helper above.
                if Int(py=value.ndim) == 0 and Int(py=value.size) == 1:
                    var dtype = value.dtype
                    for code in ["U", "S"]:  # small-loop: native Unicode/bytes dtype classes
                        if py_type(dtype) is py_type(py_call(constructor, code)):
                            return
            else:
                var aliases = Python.tuple("U", "S", "q", "Q", "g")
                for i in range(len(aliases)):  # driver-loop: exact native scalar ABI aliases
                    if cls is py_call(constructor, aliases[i]).type:
                        return
        raise failure^


def decomp_scalar_repr(value: PythonObject, context: PythonObject) raises -> String:
    """Parameter-error text without entering NumPy's Python array printing.

    Array-valued algorithm/option names are unsupported. Their diagnostic
    identifies the admitted native container without invoking its repr or
    process-wide Python formatter hooks. Primitive scalar repr is unchanged.
    """
    decomp_scalar_admit(value, context)
    if py_type(value) is py_get(context, "ndarray"):
        return "<native scalar array>"
    return String(py_call(native_loaded_module("builtins").repr, value))
