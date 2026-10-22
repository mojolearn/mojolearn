# SPDX-License-Identifier: Apache-2.0
# Native validation of the two range endpoints and Boolean configuration.
from std.python import Python, PythonObject
from bindings.py_balanced import py_type, py_get
from bindings.classical_pandas import classical_admit_value
from bindings.classical_arrays import classical_input, classical_address, classical_is_array, classical_cast_kind
from bindings.sequential_admission import _one_of
from core.classical_pandas_impl import pd_raw_at, _pd_double

def _flag(value: PythonObject, name: String, context: PythonObject) raises -> Bool:
    var b = Python.import_module("builtins")
    var types = py_get(context,"bool_types")
    if py_type(value) is not b.bool and (types is None or py_type(value) is not types[1]):
        raise Error(name + " must be a bool")
    return Bool(py=value)


def _endpoint(value: PythonObject, context: PythonObject) raises -> Float64:
    var b = Python.import_module("builtins")
    var cls = py_type(value)
    var bools = py_get(context,"bool_types")
    if cls is b.bool or (bools is not None and cls is bools[1]):
        raise Error("feature_range must contain two numeric endpoints")
    if cls is b.int or cls is b.float:
        # Exact native scalar conversion preserves large Python integers that
        # round into the supported Float32 range, without int64 truncation.
        return Float64(py=value)
    if not _one_of(cls,context["scalar_types"]):
        raise Error("feature_range must contain two numeric endpoints")
    var input = classical_input(value,context)
    var item = Int(py=input.itemsize)
    var kind = classical_cast_kind(input,item,"feature_range",context)
    var native = classical_is_array(input,context) or Bool(py=input.dtype.isnative)
    var result = _pd_double(pd_raw_at(Int(py=classical_address(input)),0,item,native),kind,item)
    _ = input^
    return result


def minmax_range_pair(value: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_admit_value(value,context)
    var b = Python.import_module("builtins")
    var sequence = py_type(value) is b.tuple or py_type(value) is b.list
    var ndarray = py_get(context,"ndarray")
    if not sequence and not classical_is_array(value,context) and (ndarray is None or py_type(value) is not ndarray):
        raise Error("feature_range must be a reusable tuple, list or 1D array")
    if sequence:
        if len(value) != 2:
            raise Error("feature_range must contain two numeric endpoints")
        return Python.tuple(_endpoint(value[0],context),_endpoint(value[1],context))
    var input = classical_input(value,context)
    if Int(py=input.ndim) != 1:
        raise Error("feature_range must be one-dimensional")
    if Int(py=input.size) != 2:
        raise Error("feature_range must contain two numeric endpoints")
    if not classical_is_array(input,context) and String(input.dtype.kind) == "b":
        raise Error("feature_range must contain two numeric endpoints")
    if not classical_is_array(input,context) and String(input.dtype.kind) == "O":
        var result = Python.tuple(_endpoint(input[0],context),_endpoint(input[1],context))
        _ = input^
        return result
    var item = Int(py=input.itemsize)
    var kind = classical_cast_kind(input,item,"feature_range",context)
    var native = classical_is_array(input,context) or Bool(py=input.dtype.isnative)
    var addr = Int(py=classical_address(input))
    var stride = Int(py=input.strides[0])
    var low = _pd_double(pd_raw_at(addr,0,item,native),kind,item)
    var high = _pd_double(pd_raw_at(addr,stride,item,native),kind,item)
    _ = input^
    return Python.tuple(low,high)

