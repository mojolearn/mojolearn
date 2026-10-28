# SPDX-License-Identifier: Apache-2.0
# Private operation-owned metric inputs. Admission happens at the public door;
# this snapshot preserves original container/array dtype inference and aliases.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.py_attr import py_attr
from bindings.native_loaded_module import native_loaded_module
from bindings.classical_arrays import classical_is_array
from bindings.linalg_ops_buffers import LinAdmission, lin_admitted_c
from bindings.native_owner_lifetime import native_release_admit
from providers.python.resample_rows_pyapi import _row_dtype_admit


def metric_input_snapshot(value: PythonObject, memo: PythonObject,
                          context: PythonObject, depth: Int = 0) raises -> PythonObject:
    if depth > 64:
        raise Error("TypeError: metric input nesting exceeds the supported depth")
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    var array = classical_is_array(value, context)
    var ndarray = py_get(context, "ndarray")
    var native_array = cls is context["array_type"]
    if not array and not native_array and cls is not ndarray and cls is not b.list and cls is not b.tuple and cls is not b.bytearray:
        # Exact scalar leaves retain their dtype inside containers. Pandas and
        # Torch retain their existing native materialization semantics.
        return value
    var key = py_call(b.id, value)
    var known = py_get(memo, key)
    if known is not None:
        return known
    var result = PythonObject(None)
    if array:
        var input = LinAdmission(value, context)
        var dtype = input.view.dtype
        # Copy the same dtype, with the original pin/slot owners retained until
        # copying finishes. Later readers see this private Array's metadata.
        var output = lin_admitted_c(input^, dtype, "metric input", context, True)
        result = output.keep
        _ = output^
    elif native_array:
        # Exact array.array construction preserves the native typecode and
        # copies its C storage without caller iteration/conversion hooks.
        result = py_call(native_loaded_module("array").array, value.typecode, value)
    elif cls is b.bytearray:
        result = py_call(b.bytearray, value)
    elif cls is ndarray:
        # Exact native ndarray.view captures the caller's metadata once. C
        # ndarray.copy only copies storage/references in the same dtype; no
        # NumPy computation, cast, __array__, or user __copy__ hook runs.
        var view = py_call(py_attr(ndarray, "view"), value)
        var dtype = view.dtype
        _row_dtype_admit(dtype, context)
        result = py_call(py_attr(ndarray, "copy"), view, order="K")
        if Bool(py=dtype.hasobject):
            # A caller may replace its object cells while a kernel runs. The
            # private snapshot must never become the final callback owner.
            native_release_admit(result, context)
        _ = view^
    else:
        var items = Python.list()
        for i in range(len(value)):  # driver-loop: stable nested native metric containers, preserving scalar leaf dtypes
            py_call(items.append, metric_input_snapshot(value[i], memo, context, depth + 1))
        result = py_call(b.tuple, items) if cls is b.tuple else items
    py_setitem(memo, key, result)
    return result
