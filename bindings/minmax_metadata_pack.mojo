# SPDX-License-Identifier: Apache-2.0
# Native validation of the two range endpoints and Boolean configuration.
# The PACK half of the former `bindings/minmax_metadata.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `bindings/minmax_metadata_unpack.mojo`.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_type, py_get
from bindings.classical_pandas import classical_admit_value
from bindings.classical_arrays_door import classical_input, classical_is_array
from bindings.sequential_admission import _one_of
from core.classical_pandas_impl import pd_raw_at, _pd_double
from core.cast_in import cast_in_kind, NPY_BOOL
from providers.python.array_metadata_pyapi import array_shape_once, array_metadata_once

from bindings.minmax_metadata_unpack import (
    _endpoint,
)


def minmax_range_values(value: PythonObject, context: PythonObject) raises -> Tuple[Float64, Float64]:
    classical_admit_value(value,context)
    var b = native_loaded_module("builtins")
    var sequence = py_type(value) is b.tuple or py_type(value) is b.list
    var ndarray = py_get(context,"ndarray")
    if not sequence and not classical_is_array(value,context) and (ndarray is None or py_type(value) is not ndarray):
        raise Error("feature_range must be a reusable tuple, list or 1D array")
    if sequence:
        if len(value) != 2:
            raise Error("feature_range must contain two numeric endpoints")
        return (_endpoint(value[0],context),_endpoint(value[1],context))
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
        var result = (_endpoint(input[0],context),_endpoint(input[1],context))
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
    return (low,high)


def minmax_range_pair(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var endpoints = minmax_range_values(value,context)
    return Python.tuple(endpoints[0],endpoints[1])
