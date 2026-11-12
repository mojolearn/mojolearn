# SPDX-License-Identifier: Apache-2.0
# One public preprocessing input snapshot, before dtype conversion or execution.
# The PACK half of the former `providers/python/prep_transform_pyapi.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `providers/python/prep_transform_pyapi_unpack.mojo`.
from bindings.admitted_finite_wording import admitted_finite_wording
from std.os import getenv
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from bindings.native_buffer_owner_pack import native_buffer_view
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.classical_arrays_door import classical_input, classical_is_array, classical_typed, classical_address
from bindings.classical_pandas_door import classical_admit_value, native_pandas_exact, _pd_class, _pd_fields, _pd_get, _pd_locations
from bindings.classical_validate_unpack import RESHAPE_HINT
from bindings.host_sparse_native_door import host_sparse_format, _sparse_fields
from bindings.native_torch_admission_door import native_torch_numpy
from bindings.sequential_admission_door import _ordinary_torch, _one_of, sequential_admit_value
from bindings.sequential_inputs_door import _promote
from bindings.x_prep_native_inputs_door import _rows_fields
from providers.python.array_metadata_pyapi_door import ArrayMetadata, ArrayPin, array_slot_once, array_shape_once, array_metadata_once
from providers.python.prep_metadata_pyapi_unpack import _array_snapshot, _pandas_empty_shape, _description, prep_numeric_vector
from providers.python.prep_category_pyapi_unpack import prep_ndarray_backed, prep_category_array
from providers.python.preprocessing_input_pyapi_door import _frozen_values
from core.cast_in import CastIn, cast_in_kind, CAST_KIND_FLOAT
from core.scaler_fit_impl import scaler_cast
from core.prep_boundary_impl import prep_float_finite, prep_gather_rows, prep_mask_any, prep_mask_export
from core.classical_pandas_impl import pd_scatter_numeric

from providers.python.prep_transform_pyapi_unpack import (
    F32P,
)


def prep_transform_values(mut snapshot: ArrayMetadata, mut pin: ArrayPin, context: PythonObject) raises -> PythonObject:
    if snapshot.dtype_number == -102 or snapshot.dtype_number == -103:
        raise Error(snapshot.dtype_name)
    if snapshot.dtype_number == -101:
        raise Error("TypeError: Sparse data was passed for X, but dense data is required. Use '.toarray()' to convert to a dense numpy array. (MaxAbsScaler takes dense X only.)")
    if len(snapshot.shape) != 2:
        raise Error("MaxAbsScaler requires a nonempty two-dimensional input")
    var n = snapshot.source.rows
    var d = snapshot.source.cols
    if n < 1 or d < 1:
        raise Error("Scaler requires a nonempty two-dimensional input")
    if n > 2147483647 // d:
        raise Error("Scaler exceeds the native Int32 indexing bound")
    if not snapshot.castable:
        _ = cast_in_kind(snapshot.dtype_number, snapshot.source.itemsize, "input")
        raise Error("TypeError: preprocessing input cannot be cast")
    if snapshot.f32_c() and snapshot.native_byte_order:
        var view = pin.owner[3] if snapshot.is_array else native_buffer_view(pin.keep)
        # One module view for the type test and the admission (no release
        # window between them).
        var mods = NativeModules()
        if py_type(view) is not mods.module("builtins").memoryview:
            raise Error("TypeError: preprocessing input storage requires an exact native memoryview")
        sequential_admit_value(mods, view, context)
        view = py_call(py_call(view.cast, "B").cast, "f")
        return _frozen_values(snapshot, pin, view, context)
    # Allocate a private packed view, so the transport cannot reread any
    # original mutable Array slot after the operation snapshot was captured.
    var output = classical_typed(Python.tuple(n, d), "<f4", context["output_context"])
    var address = Int(py=classical_address(output))
    scaler_cast(snapshot.source.addr, snapshot.source.s0, snapshot.source.s1, n, d,
                snapshot.source.itemsize, snapshot.source.kind, snapshot.native_byte_order,
                F32P(unsafe_from_address=address))
    snapshot.replace_f32(address)
    pin.keep = output
    return output
