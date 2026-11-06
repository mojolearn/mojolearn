# SPDX-License-Identifier: Apache-2.0
# Fitted-statistic admission for the preprocessing transforms (p3-prep).
# A statistic the fit published is an exact float32 C-order 1-D mojolearn
# Array: `_input` would admit it (module checks, classical_input, dtype /
# order reads) and hand back the same object. This returns it after three
# slot reads instead; anything else (a user-assigned ndarray, list, float64
# or strided array) still takes the full `_input` admission and conversion,
# with its errors. The live words are read by the program on every call, so
# an in-place write into the statistic is always seen (no copy is kept).
from std.python import PythonObject
from bindings.py_balanced import py_type
from bindings.scaler_native import _input


def prep_fitted_vector(value: PythonObject, context: PythonObject) raises -> PythonObject:
    if py_type(value) is context["array_class"]:
        if String(value.dtype) == "<f4" and Int(py=value.ndim) == 1 and Int(py=value.size) >= 1:
            return value
    return _input(value, 1, False, context)
