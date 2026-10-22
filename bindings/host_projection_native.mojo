# SPDX-License-Identifier: Apache-2.0
# Saved PCA/SVD projection owns preparation, validation, dispatch and outputs.
# Existing CPU numerical entries and their reduction order remain unchanged.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.classical_arrays import classical_input, classical_address, classical_is_array, classical_is_f32, classical_c_order, classical_kind
from bindings.classical_pandas import classical_admit_value, native_pandas_exact
from bindings.sequential_admission import _ordinary_torch
from bindings.native_torch_admission import native_torch_numpy
from bindings.host_archive_native import host_archive_typed
from bindings.host_tree_native import host_tree_input
from bindings.host_sparse_native import host_sparse_format, host_sparse_dense
from bindings.host_inference_features import host_inference_feature_names
from bindings.host_inference_preparation import host_inference_pandas_finite
from bindings.host_inference_scalars import host_inference_scalar
from bindings.host_kernel_inference_native import _kernel_entry
from core.host_inference_impl import host_inference_raw_finite
from core.host_sparse_impl import host_sparse_uint64_bound


def host_projection_admit_state(fields: PythonObject, pca: Bool, context: PythonObject) raises:
    var names = Python.tuple("n_components_", "n_samples_") if pca else Python.tuple("n_components_")
    for i in range(len(names)):  # driver-loop: declared native projection scalar fields
        if py_contains(fields, names[i]):
            host_inference_scalar(fields[names[i]], context)
    if pca and py_contains(fields, "whiten"):
        host_inference_scalar(fields["whiten"], context, True)


def _projection_buffer(value: PythonObject, rows: Int, columns: Int, matrix: Bool, c_required: Bool, context: PythonObject) raises -> PythonObject:
    var array = classical_input(value, context, "stored projection array")
    if not classical_is_f32(array, context) or (not classical_is_array(array, context) and not Bool(py=array.dtype.isnative)):
        raise Error("stored projection arrays require native float32 values")
    if Int(py=array.ndim) != (2 if matrix else 1) or Int(py=array.shape[0]) != rows:
        raise Error("stored projection array shape differs from fitted dimensions")
    if matrix and Int(py=array.shape[1]) != columns:
        raise Error("stored projection array shape differs from fitted dimensions")
    var c_order = classical_c_order(array, context)
    var f_order = String(array.order) == "F" if classical_is_array(array, context) else Bool(py=array.flags.f_contiguous)
    # Ordinary unwhitened calls passed contiguous buffers directly, including
    # Fortran storage. Retain that physical layout contract; whitening has
    # always explicitly required C order before its kernel sees any address.
    if not c_order and (c_required or not f_order):
        raise Error("stored projection arrays require contiguous native storage")
    return array


def host_projection_inference(model: PythonObject, fields: PythonObject, pca: Bool, inverse: Bool, source: PythonObject, context: PythonObject, public: Bool) raises -> PythonObject:
    var fitting = Bool(py=py_get(fields, "_mojolearn_fitting", False))
    var format = host_sparse_format(source)
    var x = PythonObject(None)
    if format.byte_length() > 0:
        # Sparse inputs bypassed the old dense validation scan, then expanded
        # in their original dtype before the shared f32 preparation step.
        x = host_sparse_dense(source, format, context)
    else:
        classical_admit_value(source, context)
        var prepared = native_torch_numpy(source) if _ordinary_torch(py_type(source), context) else source
        if native_pandas_exact(prepared):
            if not fitting:
                host_inference_pandas_finite(prepared, context)
        else:
            var raw = classical_input(prepared, context, "X")
            if classical_kind(raw, context) == "c":
                raise Error("Complex data not supported")
            var width = Int(py=raw.itemsize)
            if Int(py=raw.ndim) == 2 and classical_kind(raw, context) == "u" and width == 8:
                host_sparse_uint64_bound(Int(py=classical_address(raw)), Int(py=raw.shape[0]), Int(py=raw.shape[1]),
                    Int(py=raw.strides[0]), Int(py=raw.strides[1]), classical_is_array(raw, context) or Bool(py=raw.dtype.isnative))
            if not fitting and Int(py=raw.ndim) == 2 and classical_kind(raw, context) == "f" and (width == 4 or width == 8):
                host_inference_raw_finite(Int(py=classical_address(raw)), Int(py=raw.strides[0]), Int(py=raw.strides[1]),
                    Int(py=raw.shape[0]), Int(py=raw.shape[1]), width,
                    classical_is_array(raw, context) or Bool(py=raw.dtype.isnative))
            _ = raw^
        x = host_tree_input(prepared, context)
        _ = prepared^
    var rows = Int(py=x.shape[0])
    var features = Int(py=fields["n_features_in_"])
    var components = Int(py=fields["n_components_"])
    if features < 1 or components < 1 or components > 2147483647 // features or rows > 2147483647 // max(features, components):
        raise Error("projection dimensions must be positive and fit native indexing")
    if Int(py=x.shape[1]) != (components if inverse else features):
        raise Error("projection input width differs from fitted dimensions")
    if not fitting and not inverse:
        host_inference_feature_names(model, fields, context)
    var whiten = pca and Bool(py=fields["whiten"])
    var basis = _projection_buffer(fields["components_"], components, features, True, whiten, context)
    var mean = _projection_buffer(fields["mean_"], features, 1, False, whiten, context) if pca else basis
    var singular = PythonObject(None)
    var variance = PythonObject(None)
    var samples = 0
    if whiten:
        singular = _projection_buffer(fields["singular_values_"], components, 1, False, True, context)
        # The established public check inspects variance metadata only. The
        # whitening ABI consumes singular values, not explained variance;
        # retain that distinction for caller-assigned native model state.
        variance = _projection_buffer(fields["explained_variance_"], components, 1, False, True, context)
        samples = Int(py=fields["n_samples_"])
        if samples < 2:
            raise Error("PCA whitening requires fit rows >=2 and nonnegative singular values/variance")
    var out = host_archive_typed(Python.tuple(rows, features if inverse else components), "<f4", context["output_context"])
    if whiten and inverse:
        _ = py_call(_kernel_entry("pca_whiten_inverse_transform", context), classical_address(x), classical_address(basis),
            classical_address(singular), classical_address(mean), classical_address(out), Python.list(rows, features, components, samples))
    elif whiten:
        _ = py_call(_kernel_entry("pca_whiten_transform", context), classical_address(x), classical_address(mean),
            classical_address(basis), classical_address(singular), classical_address(out), Python.list(rows, features, components, samples))
    elif inverse:
        _ = py_call(_kernel_entry("inverse_transform", context), classical_address(x), classical_address(basis),
            classical_address(mean), classical_address(out), Python.list(rows, features, components, Int(pca)))
    elif pca:
        _ = py_call(_kernel_entry("pca_transform", context), classical_address(x), classical_address(mean),
            classical_address(basis), classical_address(out), Python.list(rows, features, components))
    else:
        _ = py_call(_kernel_entry("tsvd_transform", context), classical_address(x), classical_address(basis),
            classical_address(out), Python.list(rows, features, components))
    _ = x^
    _ = basis^
    _ = mean^
    _ = singular^
    _ = variance^
    return classical_input(out, context, "projection") if public and py_get(context, "ndarray") is not None else out
