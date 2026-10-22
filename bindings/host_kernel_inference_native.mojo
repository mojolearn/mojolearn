# SPDX-License-Identifier: Apache-2.0
# Native public inference for saved kernel models. Their established CPU
# kernels remain unchanged; this owner prepares all dense/sparse inputs,
# retains buffers through their consumers and publishes the completed result.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get
from bindings.classical_arrays import classical_input, classical_address, classical_is_array, classical_is_f32, classical_c_order, classical_kind
from bindings.classical_pandas import classical_admit_value, native_pandas_exact
from bindings.sequential_admission import _ordinary_torch
from bindings.native_torch_admission import native_torch_numpy
from bindings.host_archive_native import host_archive_typed
from bindings.host_tree_native import host_tree_input, host_tree_wide
from bindings.host_sparse_native import host_sparse_format, host_sparse_dense
from bindings.host_inference_features import host_inference_feature_names
from bindings.host_inference_preparation import host_inference_pandas_finite
from core.host_inference_impl import host_inference_raw_finite
from core.host_sparse_impl import host_sparse_uint64_bound


def _kernel_buffer(value: PythonObject, count: Int, dtype: String, context: PythonObject) raises -> PythonObject:
    var array = classical_input(value, context, "stored kernel array")
    var matches = String(array.dtype) == dtype if classical_is_array(array, context) else String(array.dtype.str) == dtype
    if not matches or Int(py=array.size) != count or not classical_c_order(array, context):
        raise Error("stored kernel arrays must preserve their native dtype, shape and C layout")
    return array


def _kernel_entry(name: String, context: PythonObject) raises -> PythonObject:
    var entry = py_get(context["host_inference_entries"], name)
    if entry is None:
        raise Error("ImportError: explicit CPU inference entry " + name + " is unavailable")
    if py_type(entry) is not context["host_tree_entry_type"]:
        raise Error("TypeError: explicit CPU inference requires a fixed native entry")
    return entry


def host_kernel_inference(model: PythonObject, fields: PythonObject, kind: String, source: PythonObject, context: PythonObject, public: Bool) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var fitting = Bool(py=py_get(fields, "_mojolearn_fitting", False))
    var format = host_sparse_format(source)
    var x = PythonObject(None)
    if format.byte_length() > 0:
        # Sparse validation historically skipped the wrapper's dense finite
        # scan. Preserve that boundary and the numerical kernel's behavior.
        x = host_sparse_dense(source, format, context)
    else:
        classical_admit_value(source, context)
        var prepared = native_torch_numpy(source) if _ordinary_torch(py_type(source), context) else source
        if native_pandas_exact(prepared):
            if kind == "rbf_sampler" and not fitting:
                host_inference_pandas_finite(prepared, context)
        else:
            var raw = classical_input(prepared, context, "X")
            if classical_kind(raw, context) == "c":
                raise Error("Complex data not supported")
            var width = Int(py=raw.itemsize)
            if Int(py=raw.ndim) == 2 and classical_kind(raw, context) == "u" and width == 8:
                host_sparse_uint64_bound(Int(py=classical_address(raw)), Int(py=raw.shape[0]), Int(py=raw.shape[1]),
                    Int(py=raw.strides[0]), Int(py=raw.strides[1]), classical_is_array(raw, context) or Bool(py=raw.dtype.isnative))
            if kind == "rbf_sampler" and not fitting and Int(py=raw.ndim) == 2 and classical_kind(raw, context) == "f" and (width == 4 or width == 8):
                host_inference_raw_finite(Int(py=classical_address(raw)), Int(py=raw.strides[0]), Int(py=raw.strides[1]),
                    Int(py=raw.shape[0]), Int(py=raw.shape[1]), width,
                    classical_is_array(raw, context) or Bool(py=raw.dtype.isnative))
            _ = raw^
        x = host_tree_input(prepared, context)
        _ = prepared^
    var rows = Int(py=x.shape[0])
    var features = Int(py=x.shape[1])
    if features != Int(py=fields["n_features_in_"]):
        raise Error("X feature count differs from the fitted kernel model")
    if not fitting:
        host_inference_feature_names(model, fields, context)
    var out = PythonObject(None)
    if kind == "kernel_ridge":
        var train = classical_input(fields["X_fit_"], context, "X_fit_")
        if Int(py=train.ndim) != 2 or Int(py=train.shape[1]) != features:
            raise Error("stored kernel training matrix has the wrong shape")
        var count = Int(py=train.shape[0])
        var targets = Int(py=fields["n_targets_"])
        if count < 1 or targets < 1 or rows > 2147483647 // targets:
            raise Error("kernel model dimensions must be positive and fit native indexing")
        train = _kernel_buffer(train, count * features, "<f4", context)
        var dual = _kernel_buffer(fields["dual_coef_"], count * targets, "<f4", context)
        var params = fields["_kernel_params"]
        if py_type(params) is not b.tuple or len(params) != 5:
            raise Error("stored KernelRidge parameters must contain five native values")
        out = host_archive_typed(Python.tuple(rows) if Bool(py=fields["_squeeze"]) else Python.tuple(rows, targets), "<f4", context["output_context"])
        if Int(py=out.size) != rows * targets:
            raise Error("squeezed kernel predictions require one target")
        _ = py_call(_kernel_entry("kernel_ridge_predict", context),
            Python.list(classical_address(train), classical_address(dual), classical_address(x), classical_address(out)),
            Python.list(count, features, targets, params[0], params[1], params[2], params[3], params[4], fields["info_"], rows))
        _ = train^
        _ = dual^
        _ = params^
    elif kind == "nystroem":
        var components = classical_input(fields["components_"], context, "components_")
        if Int(py=components.ndim) != 2 or Int(py=components.shape[1]) != features:
            raise Error("stored Nystroem components have the wrong shape")
        var count = Int(py=components.shape[0])
        if count < 1 or rows > 2147483647 // count or count > 2147483647 // count:
            raise Error("Nystroem dimensions must be positive and fit native indexing")
        components = _kernel_buffer(components, count * features, "<f4", context)
        var indices = _kernel_buffer(fields["component_indices_"], count, "<i4", context)
        var normalization = _kernel_buffer(fields["normalization_"], count * count, "<f4", context)
        var eigenvalues = _kernel_buffer(fields["eigenvalues_"], count, "<f4", context)
        var eigenvectors = _kernel_buffer(fields["eigenvectors_"], count * count, "<f4", context)
        var params = fields["_kernel_params"]
        if py_type(params) is not b.tuple or len(params) != 5:
            raise Error("stored Nystroem parameters must contain five native values")
        out = host_archive_typed(Python.tuple(rows, count), "<f4", context["output_context"])
        _ = py_call(_kernel_entry("nystroem_transform", context),
            Python.list(classical_address(components), classical_address(indices), classical_address(normalization), classical_address(eigenvalues),
                classical_address(eigenvectors), classical_address(x), classical_address(out)),
            Python.list(count, features, params[0], params[1], params[2], params[3], params[4], fields["sweeps_"], rows))
        _ = components^
        _ = indices^
        _ = normalization^
        _ = eigenvalues^
        _ = eigenvectors^
        _ = params^
    else:
        var weights = classical_input(fields["random_weights_"], context, "random_weights_")
        if Int(py=weights.ndim) != 2 or Int(py=weights.shape[0]) != features:
            raise Error("stored RBF weights have the wrong shape")
        var count = Int(py=weights.shape[1])
        if count < 1 or rows > 2147483647 // count:
            raise Error("RBF dimensions must be positive and fit native indexing")
        weights = _kernel_buffer(weights, features * count, "<f4", context)
        var offset = _kernel_buffer(fields["random_offset_"], count, "<f4", context)
        var params = fields["_params"]
        if py_type(params) is not b.tuple or len(params) != 2:
            raise Error("stored RBF parameters must contain two native values")
        out = host_archive_typed(Python.tuple(rows, count), "<f4", context["output_context"])
        _ = py_call(_kernel_entry("rbf_sampler_transform", context),
            Python.list(classical_address(weights), classical_address(offset), classical_address(x), classical_address(out)),
            Python.list(features, count, params[0], params[1], fields["sigma_"], fields["scale_"], rows))
        _ = weights^
        _ = offset^
        _ = params^
    _ = x^
    if public and py_get(context, "ndarray") is not None:
        return classical_input(host_tree_wide(out, context) if kind == "kernel_ridge" else out, context, "prediction")
    return out
