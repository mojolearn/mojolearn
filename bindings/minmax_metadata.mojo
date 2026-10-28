# SPDX-License-Identifier: Apache-2.0
# Native validation of the two range endpoints and Boolean configuration.
from std.python import Python, PythonObject
from bindings.py_balanced import py_type, py_get
from bindings.classical_pandas import classical_admit_value
from bindings.classical_arrays import classical_input, classical_is_array
from bindings.sequential_admission import _one_of
from core.classical_pandas_impl import pd_raw_at, _pd_double
from core.cast_in import cast_in_kind, NPY_BOOL
from providers.python.array_metadata_pyapi import array_shape_once, array_metadata_once

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
    var shape = array_shape_once(input,context)
    var metadata = array_metadata_once(input,shape^,"feature_range",context,allow_non_native=True)
    var item = metadata.source.itemsize
    var result = _pd_double(pd_raw_at(metadata.source.addr,0,item,metadata.native_byte_order),metadata.source.kind,item)
    _ = metadata^
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
    # An exact Array already owns native storage; converting it to an ndarray
    # here would recapture mutable slots before the operation snapshot.
    var input = value if classical_is_array(value,context) else classical_input(value,context)
    var shape = array_shape_once(input,context)
    if len(shape) != 1:
        raise Error("feature_range must be one-dimensional")
    if shape[0] != 2:
        raise Error("feature_range must contain two numeric endpoints")
    var metadata = array_metadata_once(input,shape^,"feature_range",context,
                                      allow_non_native=True,allow_unsupported=True)
    if not metadata.is_array and metadata.dtype_number == NPY_BOOL:
        raise Error("feature_range must contain two numeric endpoints")
    if not metadata.is_array and metadata.dtype_number == 17:  # NumPy NPY_OBJECT
        # Exact ndarray indexing only returns the retained object references;
        # _endpoint refuses callback/protocol values before converting them.
        var result = Python.tuple(_endpoint(input[0],context),_endpoint(input[1],context))
        _ = metadata^
        return result
    if not metadata.castable:
        # Reproduce the existing refusal from captured words, never reread
        # dtype or execute an admission-only CastIn placeholder.
        _ = cast_in_kind(metadata.dtype_number,metadata.source.itemsize,"feature_range")
        raise Error("feature_range must contain two numeric endpoints")
    var item = metadata.source.itemsize
    var kind = metadata.source.kind
    var low = _pd_double(pd_raw_at(metadata.source.addr,0,item,metadata.native_byte_order),kind,item)
    var high = _pd_double(pd_raw_at(metadata.source.addr,metadata.source.s0,item,metadata.native_byte_order),kind,item)
    _ = metadata^
    return Python.tuple(low,high)

