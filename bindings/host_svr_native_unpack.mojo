# SPDX-License-Identifier: Apache-2.0
# Explicit CPU SVR owns preparation, precomputed-column selection and output.
# Its existing SVM numerical entries and reduction order stay unchanged.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields
from bindings.native_error_class import error_without_class
from bindings.classical_attributes import classical_check_layout, classical_getattribute, classical_connect_numpy
from bindings.classical_arrays_door import classical_input, classical_address, classical_is_array, classical_is_f32, classical_c_order, classical_cast_kind, classical_kind
from bindings.classical_pandas import classical_admit_value, native_pandas_exact, _pd_type_fields
from bindings.sequential_admission import _ordinary_torch
from bindings.native_torch_admission import native_torch_numpy
from bindings.host_array_native_pack import host_archive_typed
from bindings.host_tree_native import host_tree_input, host_tree_constructor_fields, host_tree_wide
from bindings.host_sparse_native import host_sparse_format, host_sparse_dense
from bindings.host_inference_features_unpack import host_inference_feature_names
from bindings.host_inference_scalars_door import host_inference_scalar
from bindings.host_kernel_inference_native_unpack import _kernel_entry, _kernel_buffer
from bindings.scaler_lifecycle_native import _outside
from core.host_archive_impl import host_archive_support_bound
from core.host_sparse_impl import host_sparse_uint64_bound
from core.host_svr_impl import host_svr_bias, host_svr_indices


def host_svr_admit_state(fields: PythonObject, context: PythonObject) raises:
    var buffers = Python.tuple("dual_coef_", "support_vectors_")
    for i in range(len(buffers)):  # driver-loop: refuse callback-bearing storage before numeric shape validation
        var value = py_get(fields, buffers[i])
        if py_get(context, "ndarray") is not None and py_type(value) is context["ndarray"]:
            if classical_kind(value, context) == "O":
                raise Error("TypeError: saved SVR coefficients and support vectors require native numeric storage, not Python objects")
    var names = Python.tuple("n_features_in_", "n_support_", "_gamma", "_hp_cache_size", "_hp_degree", "_hp_coef0")
    for i in range(len(names)):  # driver-loop: fixed scalar fields consumed by the SVM ABI
        if py_contains(fields, names[i]):
            host_inference_scalar(fields[names[i]], context)
    if py_contains(fields, "_hp_kernel") and py_type(fields["_hp_kernel"]) is not native_loaded_module("builtins").str:
        raise Error("TypeError: saved SVR kernel must be an exact native string")



def _svr_matrix(value: PythonObject, rows: Int, columns: Int, context: PythonObject) raises -> PythonObject:
    var array = classical_input(value, context, "support_vectors_")
    if not classical_is_f32(array, context) or (not classical_is_array(array, context) and not Bool(py=array.dtype.isnative)):
        raise Error("SVR support vectors require native float32 storage")
    if Int(py=array.ndim) != 2 or Int(py=array.shape[0]) != rows or Int(py=array.shape[1]) != columns:
        raise Error("SVR support vector shape differs from fitted dimensions")
    var f_order = String(array.order) == "F" if classical_is_array(array, context) else Bool(py=array.flags.f_contiguous)
    # The old address-only ABI consumed C or F contiguous state in place.
    # Preserve those physical words rather than silently reorder model state.
    if not classical_c_order(array, context) and not f_order:
        raise Error("SVR support vectors require contiguous native storage")
    return array


def _svr_bias(value: PythonObject, context: PythonObject) raises -> Float64:
    var array = classical_input(value, context, "intercept_")
    if Int(py=array.ndim) != 1 or Int(py=array.size) < 1:
        raise Error("SVR intercept_ requires a nonempty native vector")
    var width = Int(py=array.itemsize)
    var result = host_svr_bias(Int(py=classical_address(array)), width,
        classical_cast_kind(array, width, "intercept_", context),
        classical_is_array(array, context) or Bool(py=array.dtype.isnative))
    _ = array^
    return result
