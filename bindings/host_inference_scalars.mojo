# SPDX-License-Identifier: Apache-2.0
# Container admission alone does not qualify scalar conversion: an exact
# object ndarray can call its element's Python __int__/__float__/__bool__.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get
from bindings.sequential_admission import _one_of
from bindings.native_numpy_metadata import native_numpy_dtype


def host_inference_scalar(value: PythonObject, context: PythonObject, truth: Bool = False) raises:
    var b = Python.import_module("builtins")
    var cls = py_type(value)
    if _one_of(cls, Python.tuple(b.str, b.bytes, b.bytearray, b.int, b.float, b.bool, py_type(PythonObject(None)))) or _one_of(cls, context["scalar_types"]):
        return
    if cls is py_get(context, "ndarray"):
        # Ordinary numeric 0-D arrays were accepted scalar state. NumPy's
        # native truth slot also accepts a single item at any rank. Admit the
        # exact primitive dtype before those C slots can read any element.
        if Int(py=value.size) != 1 or (not truth and Int(py=value.ndim) != 0):
            raise Error("TypeError: scalar model state requires a scalar or zero-dimensional numeric array")
        var dtype = value.dtype
        var types = py_call(b.list, context["numeric_dtypes"].values())
        for i in range(len(types)):  # driver-loop: primitive native scalar dtype identities
            if py_type(dtype) is py_type(types[i]):
                return
        # NumPy distinguishes long long aliases and long double from its
        # platform int64/float64 types. Their numeric C scalar slots likewise
        # cannot invoke Python element hooks; keep existing 0-D support.
        var constructor = native_numpy_dtype(context)
        var aliases = Python.tuple("q", "Q", "g")
        for i in range(len(aliases)):  # driver-loop: native numeric dtype aliases
            if py_type(dtype) is py_type(py_call(constructor, aliases[i])):
                return
        if truth:
            var complex_types = Python.tuple("F", "D", "G")
            for i in range(len(complex_types)):  # driver-loop: native complex truth slots, never numeric casts
                if py_type(dtype) is py_type(py_call(constructor, complex_types[i])):
                    return
    raise Error("TypeError: scalar model state requires native values without Python conversion hooks")
