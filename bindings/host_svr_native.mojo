# SPDX-License-Identifier: Apache-2.0
# Explicit CPU SVR owns preparation, precomputed-column selection and output.
# Its existing SVM numerical entries and reduction order stay unchanged.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields
from bindings.native_error_class import error_without_class
from bindings.classical_attributes import classical_check_layout, classical_getattribute, classical_connect_numpy
from bindings.classical_arrays import classical_input, classical_address, classical_is_array, classical_is_f32, classical_c_order, classical_cast_kind, classical_kind
from bindings.classical_pandas import classical_admit_value, native_pandas_exact, _pd_type_fields
from bindings.sequential_admission import _ordinary_torch
from bindings.native_torch_admission import native_torch_numpy
from bindings.host_archive_native import host_archive_typed
from bindings.host_tree_native import host_tree_input, host_tree_constructor_fields, host_tree_wide
from bindings.host_sparse_native import host_sparse_format, host_sparse_dense
from bindings.host_inference_features import host_inference_feature_names
from bindings.host_inference_scalars import host_inference_scalar
from bindings.host_kernel_inference_native import _kernel_entry, _kernel_buffer
from bindings.scaler_lifecycle_native import _outside
from core.host_archive_impl import host_archive_support_bound
from core.host_sparse_impl import host_sparse_uint64_bound
from core.host_svr_impl import host_svr_bias, host_svr_indices


def host_svr_admit_state(fields: PythonObject, context: PythonObject) raises:
    var names = Python.tuple("n_features_in_", "n_support_", "_gamma", "_hp_cache_size", "_hp_degree", "_hp_coef0")
    for i in range(len(names)):  # driver-loop: fixed scalar fields consumed by the SVM ABI
        if py_contains(fields, names[i]):
            host_inference_scalar(fields[names[i]], context)
    if py_contains(fields, "_hp_kernel") and py_type(fields["_hp_kernel"]) is not Python.import_module("builtins").str:
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


def _svr_support(value: PythonObject, count: Int, columns: Int, context: PythonObject) raises -> PythonObject:
    var array = classical_input(value, context, "support_")
    if Int(py=array.ndim) != 1 or Int(py=array.size) != count:
        raise Error("SVR support index shape differs from n_support_")
    if classical_is_array(value, context):
        # Existing Array state is the saved int32 ABI. Ordinary ndarray/list
        # state went through _i32; preserve its native integer conversion.
        if String(value.dtype) != "<i4":
            raise Error("saved SVR support indices must retain int32 storage")
        _ = host_archive_support_bound(Int(py=classical_address(array)), count, columns)
        return array
    var width = Int(py=array.itemsize)
    var kind = classical_cast_kind(array, width, "support_", context)
    var out = host_archive_typed(Python.tuple(count), "<i4", context["output_context"])
    host_svr_indices(Int(py=classical_address(array)), Int(py=array.strides[0]), count,
        width, kind, classical_is_array(array, context) or Bool(py=array.dtype.isnative), columns, Int(py=classical_address(out)))
    _ = array^
    return out


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


def host_svr_predict(model: PythonObject, fields: PythonObject, source: PythonObject, context: PythonObject, public: Bool) raises -> PythonObject:
    var fitting = Bool(py=py_get(fields, "_mojolearn_fitting", False))
    var format = host_sparse_format(source)
    var x = PythonObject(None)
    if format.byte_length() > 0:
        x = host_sparse_dense(source, format, context)
    else:
        classical_admit_value(source, context)
        var prepared = native_torch_numpy(source) if _ordinary_torch(py_type(source), context) else source
        if not native_pandas_exact(prepared):
            var raw = classical_input(prepared, context, "X")
            if classical_kind(raw, context) == "c":
                raise Error("Complex data not supported")
            var width = Int(py=raw.itemsize)
            if Int(py=raw.ndim) == 2 and classical_kind(raw, context) == "u" and width == 8:
                host_sparse_uint64_bound(Int(py=classical_address(raw)), Int(py=raw.shape[0]), Int(py=raw.shape[1]),
                    Int(py=raw.strides[0]), Int(py=raw.strides[1]), classical_is_array(raw, context) or Bool(py=raw.dtype.isnative))
            # SVR's existing query path casts directly; nonfinite inputs
            # reach the unchanged SVM kernel rather than a dense finite scan.
            _ = raw^
        x = host_tree_input(prepared, context)
        _ = prepared^
    var rows = Int(py=x.shape[0])
    var features = Int(py=fields["n_features_in_"])
    var count = Int(py=fields["n_support_"])
    if Int(py=x.shape[1]) != features:
        raise Error("SVR input width differs from fitted dimensions")
    if count < 0 or count > 2147483647 // features or rows > 2147483647 // max(count, 1):
        raise Error("SVR support dimensions exceed native indexing")
    if not fitting:
        host_inference_feature_names(model, fields, context)
    var kernel = String(fields["_hp_kernel"])
    var code = 0 if kernel == "linear" else (1 if kernel == "poly" else (2 if kernel == "rbf" else (3 if kernel == "sigmoid" else (4 if kernel == "precomputed" else -1))))
    if code < 0:
        raise Error("SVR kernel name is not recognized")
    var dual = PythonObject(None)
    var support = PythonObject(None)
    var columns = PythonObject(None)
    if count > 0:
        dual = _kernel_buffer(fields["dual_coef_"], count, "<f4", context)
        if code == 4:
            columns = _svr_support(fields["support_"], count, features, context)
            var gathered = host_archive_typed(Python.tuple(rows, count), "<f4", context["output_context"])
            _ = py_call(_kernel_entry("svc_pair_epilogue", context), classical_address(x), 0,
                classical_address(columns), classical_address(gathered), Python.list(6, rows, features, rows, count))
            _ = x^
            x = gathered
            features = count
            support = dual
        else:
            support = _svr_matrix(fields["support_vectors_"], count, features, context)
    var params = Python.list(rows, features, count, _svr_bias(fields["intercept_"], context), code,
        fields["_gamma"], fields["_hp_cache_size"])
    if code == 1 or code == 3:
        _ = py_call(params.append, py_call(Python.import_module("builtins").int, fields["_hp_degree"]))
        _ = py_call(params.append, Float64(py=fields["_hp_coef0"]))
    var out = host_archive_typed(Python.tuple(rows), "<f4", context["output_context"])
    _ = py_call(_kernel_entry("svr_predict", context), classical_address(x),
        classical_address(dual) if count > 0 else PythonObject(0),
        classical_address(support) if count > 0 else PythonObject(0), classical_address(out), params)
    _ = x^
    _ = dual^
    _ = support^
    _ = columns^
    return classical_input(host_tree_wide(out, context), context, "prediction") if public and py_get(context, "ndarray") is not None else out


def _svr_coefficients(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = host_tree_constructor_fields(model, context)
    if not py_contains(fields, "dual_coef_"):
        raise Error("AttributeError: SVR coef_ requires a fitted model")
    var keys = py_call(Python.import_module("builtins").list, fields.keys())
    for i in range(len(keys)):  # driver-loop: native stored state admission before computed attribute work
        classical_admit_value(fields[keys[i]], context)
    host_svr_admit_state(fields, context)
    if String(fields["_hp_kernel"]) != "linear":
        raise Error("AttributeError: SVR coef_ is available only for kernel='linear'")
    var features = Int(py=fields["n_features_in_"])
    var count = Int(py=fields["n_support_"])
    if features < 1 or count < 0 or count > 2147483647 // features:
        raise Error("SVR support dimensions exceed native indexing")
    var out = host_archive_typed(Python.tuple(1, features), "<f4", context["output_context"])
    if count > 0:
        var mode = py_get(fields, "numeric_mode")
        if mode is not None and (py_type(mode) is not Python.import_module("builtins").str or String(mode) != "identical"):
            raise Error("explicit host models run identical mode only")
        var dual = _kernel_buffer(fields["dual_coef_"], count, "<f4", context)
        var support = _svr_matrix(fields["support_vectors_"], count, features, context)
        _ = py_call(_kernel_entry("svc_dual_gemv", context), classical_address(dual), classical_address(support),
            Python.list(count, features, 0), classical_address(out))
        _ = dual^
        _ = support^
    return classical_input(out, context, "coef_") if _outside(context, 0) and py_get(context, "ndarray") is not None else out


def host_svr_attribute_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        var context = args[0]
        var model = args[1]
        var name = args[2]
        var b = Python.import_module("builtins")
        classical_connect_numpy(context)
        classical_check_layout(model, context)
        if py_type(name) is not b.str:
            raise Error("TypeError: native SVR attribute names must be exact strings")
        if String(name) == "coef_":
            var mro = py_call(b.type.__dict__["__mro__"].__get__, py_type(model))
            for i in range(len(mro)):  # driver-loop: fixed computed-property identity and normal MRO precedence
                var fields = _pd_type_fields(mro[i])
                if py_contains(fields, "coef_"):
                    if fields["coef_"] is context["host_svr_coefficient_property"]:
                        var coefficients = _svr_coefficients(model, context)
                        return coefficients.steal_data()
                    break
        var value = classical_getattribute(model, name, context)
        if not _outside(context, 0):
            var original = py_get(py_fields(model), name)
            if classical_is_array(original, context):
                value = original
        return value.steal_data()
    except error:
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        return raise_python_exception(error, ExceptionType.ValueError)
