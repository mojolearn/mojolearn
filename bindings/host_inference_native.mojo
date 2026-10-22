# SPDX-License-Identifier: Apache-2.0
# Explicit CPU model operations. Each connected family enters one native
# owner; other families remain unchanged until their full contract is ported.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_pandas import classical_admit_value, native_pandas_exact
from bindings.sequential_admission import _ordinary_torch
from bindings.classical_exceptions import classical_require_fitted
from bindings.host_inference_features import host_inference_feature_names
from bindings.host_inference_preparation import host_inference_pandas_finite
from bindings.host_kernel_inference_native import host_kernel_inference, host_kernel_admit_state
from bindings.host_inference_scalars import host_inference_scalar
from bindings.host_projection_native import host_projection_inference, host_projection_admit_state
from bindings.host_svr_native import host_svr_admit_state, host_svr_predict
from bindings.host_svc_native import host_svc_admit_state, host_svc_inference
from bindings.host_kde_native import host_kde_admit_state, host_kde_inference
from bindings.host_iforest_native import host_iforest_admit_option, host_iforest_admit_state, host_iforest_inference
from bindings.host_logistic_native import host_logistic_classes, host_logistic_inference
from bindings.native_torch_admission import native_torch_numpy
from bindings.classical_arrays import classical_input, classical_address, classical_is_f32, classical_is_array, classical_c_order, classical_kind
from bindings.host_archive_native import host_archive_typed
from bindings.host_tree_native import host_tree_input, host_tree_constructor_fields, host_tree_scalar, host_tree_wide
from core.host_tree_impl import host_tree_column_copy
from core.host_inference_impl import host_inference_finite, host_inference_raw_finite


def host_inference_entry(name: String, context: PythonObject) raises -> PythonObject:
    var entry = py_get(context["host_inference_entries"], name)
    if entry is None:
        raise Error("ImportError: explicit CPU inference entry " + name + " is unavailable")
    if py_type(entry) is not context["host_tree_entry_type"]:
        raise Error("TypeError: explicit CPU inference requires a fixed native entry")
    return entry


def host_inference_operation(model: PythonObject, operation: String, args: PythonObject, context: PythonObject, error_context: PythonObject, public: Bool) raises -> PythonObject:
    classical_connect_numpy(context)
    var fields = host_tree_constructor_fields(model, context)
    var b = Python.import_module("builtins")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, py_type(model))
    var kind = PythonObject(None)
    for i in range(len(mro)):  # driver-loop: fixed explicit CPU class ancestry
        kind = py_get(context["host_inference_kinds"], mro[i])
        if kind is not None:
            break
    if kind is None:
        raise Error("TypeError: no native explicit CPU prediction owner for this class")
    var logistic = String(kind) == "logistic"
    var svc = String(kind) == "svc"
    var kde = String(kind) == "kde"
    var iforest = String(kind) == "iforest"
    var iforest_options = Python.tuple("n_estimators", "max_samples", "max_depth", "max_features", "bootstrap", "contamination", "random_state", "warm_start", "n_jobs", "verbose", "output_type")
    var labels = PythonObject(None)
    if (logistic or svc) and py_contains(fields, "classes_"):
        labels = host_logistic_classes(fields["classes_"], context)
    var keys = py_call(b.list, fields.keys())
    for i in range(len(keys)):  # driver-loop: admitted stored model values
        var name = String(keys[i])
        # KDE's option owner admits native Unicode/byte scalar subclasses
        # before conversion; ordinary container admission cannot classify them.
        var kde_option = kde and (name == "bandwidth" or name == "kernel" or name == "metric")
        if iforest and py_contains(iforest_options, keys[i]):
            host_iforest_admit_option(fields[keys[i]], context)
        elif not kde_option and (not (logistic or svc) or name != "classes_"):
            classical_admit_value(fields[keys[i]], context)
    host_inference_scalar(py_get(fields, "_mojolearn_fitted", False), context, True)
    host_inference_scalar(py_get(fields, "_mojolearn_fitting", False), context, True)
    if py_contains(fields, "n_features_in_"):
        host_inference_scalar(fields["n_features_in_"], context)
    var kernel = String(kind) == "kernel_ridge" or String(kind) == "nystroem" or String(kind) == "rbf_sampler"
    if kernel:
        host_kernel_admit_state(fields, String(kind), context)
    var projection = String(kind) == "pca" or String(kind) == "tsvd"
    if projection:
        host_projection_admit_state(fields, String(kind) == "pca", context)
    if logistic and py_contains(fields, "fit_intercept"):
        host_inference_scalar(fields["fit_intercept"], context, True)
    if svc:
        host_svc_admit_state(fields, context)
    if iforest:
        host_iforest_admit_state(fields, context)
    if kde:
        host_kde_admit_state(fields, context)
    var svr = String(kind) == "svr"
    if svr:
        host_svr_admit_state(fields, context)
    var method = "inverse_transform" if operation == "host_infer_inverse_transform" else ("transform" if operation == "host_infer_transform" else "predict")
    if operation == "host_infer_decision_function":
        method = "decision_function"
    elif operation == "host_infer_predict_proba":
        method = "predict_proba"
    elif operation == "host_infer_predict_log_proba":
        method = "predict_log_proba"
    elif operation == "host_infer_score_samples":
        method = "score_samples"
    elif operation == "host_infer_score":
        method = "score"
    classical_require_fitted(model, False, method, error_context)
    var mode = py_get(fields, "numeric_mode")
    if mode is not None and (py_type(mode) is not b.str or String(mode) != "identical"):
        raise Error("explicit host models run identical mode only")
    if kernel:
        return host_kernel_inference(model, fields, String(kind), args[0], context, public)
    if projection:
        return host_projection_inference(model, fields, String(kind) == "pca", operation == "host_infer_inverse_transform", args[0], context, public)
    if iforest:
        return host_iforest_inference(model, fields, args[0], 2 if operation == "host_infer_predict" else (1 if operation == "host_infer_decision_function" else 0), context, public)
    if kde:
        return host_kde_inference(model, fields, args[0], operation == "host_infer_score", context, public)
    if svc:
        return host_svc_inference(model, fields, labels, operation, args[0], context, public)
    if svr:
        return host_svr_predict(model, fields, args[0], context, public)
    if logistic:
        return host_logistic_inference(model, fields, labels, operation, args[0], context, public)
    classical_admit_value(args, context)
    var cd = String(kind) == "cd"
    if cd and not py_contains(fields, "coef_"):
        raise Error("call fit before predict")
    if not cd and py_get(fields, "cluster_centers_") is None:
        raise Error("RuntimeError: this estimator is not fitted yet")
    var fitting = Bool(py=py_get(fields, "_mojolearn_fitting", False))
    var prepared = native_torch_numpy(args[0]) if _ordinary_torch(py_type(args[0]), context) else args[0]
    if cd and not fitting and native_pandas_exact(prepared):
        host_inference_pandas_finite(prepared, context)
    elif cd and not fitting:
        var source = classical_input(prepared, context, "X")
        if classical_kind(source, context) == "c":
            raise Error("Complex data not supported")
        var width = Int(py=source.itemsize)
        # Preserve the existing pre-cast f4/f8 finiteness check: finite f8
        # values which overflow when rounded to f4 still reach the CPU kernel.
        if Int(py=source.ndim) == 2 and classical_kind(source, context) == "f" and (width == 4 or width == 8):
            host_inference_raw_finite(Int(py=classical_address(source)), Int(py=source.strides[0]), Int(py=source.strides[1]),
                Int(py=source.shape[0]), Int(py=source.shape[1]), width,
                classical_is_array(source, context) or Bool(py=source.dtype.isnative))
        _ = source^
    var x = host_tree_input(prepared, context)
    _ = prepared^
    var rows = Int(py=x.shape[0])
    var features = Int(py=x.shape[1])
    if cd:
        if features != Int(py=fields["n_features_in_"]):
            raise Error("feature count differs from fit")
        if not fitting:
            host_inference_feature_names(model, fields, context)
        var coef = classical_input(fields["coef_"], context, "coef_")
        if Int(py=coef.ndim) != 1 or Int(py=coef.size) != features or not classical_is_f32(coef, context) or not classical_c_order(coef, context) or (not classical_is_array(coef, context) and not Bool(py=coef.dtype.isnative)):
            raise Error("stored coefficients must be a native float32 feature vector")
        host_tree_scalar(fields["intercept_"], context)
        var column = host_archive_typed(Python.tuple(rows, features), "<f4", context["output_context"])
        host_tree_column_copy(Int(py=classical_address(x)), rows, features, Int(py=classical_address(column)))
        var out = host_archive_typed(Python.tuple(rows), "<f4", context["output_context"])
        _ = py_call(host_inference_entry("cd_predict", context), classical_address(column), classical_address(coef), classical_address(out),
            Python.list(rows, features, py_call(b.float, fields["intercept_"])))
        _ = column^
        _ = coef^
        _ = x^
        return classical_input(host_tree_wide(out, context), context, "prediction") if public and py_get(context, "ndarray") is not None else out
    var centers = host_tree_input(fields["cluster_centers_"], context)
    if features != Int(py=centers.shape[1]):
        raise Error("X feature count differs from the fitted centers")
    if not fitting:
        host_inference_feature_names(model, fields, context)
    host_inference_finite(Int(py=classical_address(x)), rows * features)
    var metric = fields["metric"]
    var code = PythonObject(None)
    if py_type(metric) is b.str:
        code = py_get(context["host_archive_kmeans_metrics"], metric)
        if code is None:
            raise Error("unknown KMeans metric")
    else:
        host_tree_scalar(metric, context)
        code = py_call(b.int, metric)
    var clusters = Int(py=centers.shape[0])
    var transform = operation == "host_infer_transform"
    var out = host_archive_typed(Python.tuple(rows, clusters) if transform else Python.tuple(rows), "<f4" if transform else "<i4", context["output_context"])
    _ = py_call(host_inference_entry("kmeans_transform" if transform else "kmeans_predict", context), classical_address(x), classical_address(centers), classical_address(out),
        Python.list(rows, features, clusters, code))
    _ = centers^
    _ = x^
    return classical_input(out, context, "prediction") if public and py_get(context, "ndarray") is not None else out
