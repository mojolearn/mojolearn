# SPDX-License-Identifier: Apache-2.0
# Incremental native classical owner. Public activation waits for full API
# migration; this private entry is exercised directly by strict audit tests.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.classical_error_boundary import classical_python_error
from bindings.sequential_admission import sequential_admit_value
from bindings.sequential_description import sequential_fields
from bindings.classical_ridge import classical_ridge_fit, classical_ridge_parameters, classical_linear_predict, classical_linear_predict_public
from bindings.classical_ols import classical_ols_fit
from bindings.classical_score import classical_linear_score
from bindings.classical_metadata import classical_initialize, classical_get_params, classical_set_params, classical_is_fitted, classical_parameter_names
from bindings.classical_attributes import classical_check_layout, classical_getattribute, classical_connect_numpy
from bindings.classical_archive import classical_archive_save
from bindings.classical_archive_read import classical_archive_load, classical_host_model
from bindings.host_archive_native import host_archive_build
from bindings.host_archive_predict_native import host_archive_predict
from bindings.host_knn_predict_native import host_knn_predict
from bindings.archive_schema_native import archive_schema_save
from bindings.linear_score_native import linear_regressor_score
from bindings.family_native import family_operation
from bindings.host_tree_native import host_tree_operation
from bindings.trees_public_native import trees_public_operation
from bindings.host_inference_native import host_inference_operation
from bindings.host_kde_fit_native import host_kde_fit
from bindings.host_iforest_fit_native import host_iforest_fit
from bindings.host_density_constructor_native import host_density_initialize
from bindings.host_density_archive_native import host_density_save
from bindings.host_density_hash_native import host_density_hash
from bindings.classical_exceptions import classical_require_fitted, classical_error_restore, classical_error_reduce
from bindings.classical_pandas import classical_admit_value
from bindings.classical_tags import classical_tags
from bindings.classical_validate import classical_refuse_sparse, classical_estimator_name
from bindings.classical_host_owner import classical_host_entries, classical_host_hash
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.native_caller_frame import native_caller_globals


def classical_caller_outside(context: PythonObject) raises -> Bool:
    """True when the Python frame that called the public method belongs to
    code outside the package (the rule of _mode._caller_outside). Frame and
    globals metadata are read through CPython's native frame APIs only."""
    var globs = PythonObject(None)
    try:
        # Depth 0 is the public method's own one-call frame.
        globs = native_caller_globals(1)
    except:
        return True
    if globs is None:
        return True
    if py_get(globs, "__mojolearn_user_boundary__") is PythonObject(True):
        return True
    if py_get(globs, "__mojolearn_internal__") is PythonObject(True):
        return False
    var name = py_get(globs, "__name__")
    if py_type(name) is not py_type(PythonObject("")):
        return True
    var module = String(name)
    return module != "mojolearn" and not module.startswith("mojolearn.") and not module.startswith("mojolearn_verify_all_")


def classical_entries(model: PythonObject, context: PythonObject) raises -> PythonObject:
    if Bool(py=py_get(context, "classical_explicit_cpu", False)):
        return classical_host_entries(model, context)
    var fields = sequential_fields(model)
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not Python.import_module("builtins").str:
        raise Error("numeric_mode must be fast, identical or None")
    mode = mode.strip().lower()
    if String(mode) != "identical" and String(mode) != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    var entries = context["classical_bindings"][mode]
    if py_type(entries) is Python.import_module("builtins").str:
        raise Error("ImportError: " + String(entries))
    if String(entries["estimators_vendor"]()) != String(context["vendor"]):
        raise Error("RuntimeError: classical native vendor differs from selected execution")
    if Int(py=entries["estimators_numeric_mode"]()) != (1 if String(mode) == "identical" else 0):
        raise Error("RuntimeError: classical native mode differs from selected execution")
    return entries


def classical_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var model = call[0]
        var operation = String(call[1])
        var args = call[2]
        var context = call[3]
        var b = Python.import_module("builtins")
        if operation == "restore_not_fitted":
            var error = classical_error_restore(args, context)
            return error.steal_data()
        if operation == "reduce_not_fitted":
            var reduced = classical_error_reduce(model, context)
            return reduced.steal_data()
        if operation == "host_kde_fit":
            var result = host_kde_fit(model, args, context["host_inference_context"])
            return result.steal_data()
        if operation == "host_iforest_fit":
            var result = host_iforest_fit(model, args, context["host_inference_context"])
            return result.steal_data()
        if operation == "host_density_save":
            var result = host_density_save(model, args, context["host_inference_context"])
            return result.steal_data()
        if operation == "host_density_hash":
            var result = host_density_hash(model, args, context["host_inference_context"])
            return result.steal_data()
        if operation == "host_density_initialize":
            var result = host_density_initialize(model, args, context["host_inference_context"])
            return result.steal_data()
        if operation.startswith("host_infer_"):
            var result = host_inference_operation(model, operation, args, context["host_inference_context"], context, classical_caller_outside(context))
            return result.steal_data()
        if operation == "archive_infer_predict" or operation == "archive_infer_predict_proba":
            var route = "host_infer_predict" if operation == "archive_infer_predict" else "host_infer_predict_proba"
            var result = host_inference_operation(model, route, args, context["host_inference_context"], context, False)
            return result.steal_data()
        if operation == "archive_knn_classifier" or operation == "archive_knn_regressor":
            var result = host_knn_predict(model, args[0], operation == "archive_knn_classifier", Bool(py=args[1]),
                                          context["host_archive_predict_context"])
            return result.steal_data()
        if operation.startswith("trees_"):
            # The public tree ensembles' one-call methods
            # (bindings/trees_public_native.mojo).
            var result = trees_public_operation(model, operation, args, context)
            return result.steal_data()
        if operation.startswith("host_tree_"):
            var result = host_tree_operation(model, operation, args, context["host_tree_context"])
            return result.steal_data()
        if operation == "host_archive_predict" or operation == "host_archive_predict_proba":
            var predict_context = py_get(context, "host_archive_predict_context")
            if predict_context is None:
                raise Error("RuntimeError: native host archive schemas are not connected")
            var predicted = host_archive_predict(args[0], args[1], operation == "host_archive_predict_proba", predict_context)
            return predicted.steal_data()
        if operation == "host_archive_model" or operation == "host_archive_load" or operation == "host_archive_classical_model":
            var archive_context = py_get(context, "host_archive_context")
            if archive_context is None:
                raise Error("RuntimeError: native host archive schemas are not connected")
            # Checkpoint input admission belongs to the archive reader: its
            # exact native file descriptors and standard path layouts include
            # BytesIO/FileIO/buffered streams, which numeric-array admission
            # correctly refuses for unrelated estimator operations.
            var restored = host_archive_build(args[0], archive_context, model if operation == "host_archive_load" else PythonObject(None), operation == "host_archive_classical_model")
            return restored.steal_data()
        if operation.startswith("family_"):
            var family = family_operation(model, operation, args, context)
            return family.steal_data()
        if operation == "linear_regressor_score":
            # The linear expansion regressors' one-call score: predict and
            # R^2 in one native operation.
            var scored = linear_regressor_score(model, args, context)
            return scored.steal_data()
        if operation == "archive_save":
            # The public estimators' one-call save: the class's connected
            # schema, written by the deterministic NPZ writer.
            var saved = archive_schema_save(model, args[0], context)
            return saved.steal_data()
        if operation == "host_model":
            var host_context = py_get(context, "classical_host_context")
            if host_context is None:
                raise Error("RuntimeError: the explicit host linear models are not connected")
            classical_connect_numpy(host_context)
            classical_admit_value(args, host_context)
            var hosted = classical_host_model(args[0], host_context)
            return hosted.steal_data()
        var loading = operation == "load"
        var class_operation = loading or operation == "param_names"
        var cls = model if class_operation else py_type(model)
        var host = py_get(context, "classical_host_context")
        if host is not None and py_contains(host["classical_kinds"], cls):
            context = host
        classical_connect_numpy(context)
        # classical_kinds maps each connected class (Ridge, LinearRegression,
        # their explicit host classes and subclasses) to True for the Ridge
        # family and False for LinearRegression.
        var kind = py_get(context["classical_kinds"], cls)
        if kind is None:
            raise Error("TypeError: the native classical owner requires a connected Ridge or LinearRegression class")
        var ridge = Bool(py=kind)
        var linear = not ridge
        if operation == "param_names":
            var names = py_call(b.list, classical_parameter_names(ridge))
            return names.steal_data()
        if not class_operation:
            classical_check_layout(model, context)
        if operation == "getattribute":
            var attribute = classical_getattribute(model, args[0], context)
            return attribute.steal_data()
        if loading:
            var loaded = classical_archive_load(model, args[0], ridge, context)
            return loaded.steal_data()
        if operation == "ridge_fit" or operation == "ols_fit" or operation == "predict_public" or operation == "predict_score" or operation == "score_public":
            # scikit-learn's sparse refusal comes before input admission.
            classical_refuse_sparse(args[0], classical_estimator_name(model))
        if operation != "save":
            classical_admit_value(args, context)
        var fields = sequential_fields(model)
        var names = context["classical_fields"]
        for k in range(len(names)):  # driver-loop: over connected estimator metadata fields
            if py_contains(fields, names[k]):
                sequential_admit_value(fields[names[k]], context)
        var result = PythonObject(None)
        if operation == "initialize":
            result = classical_initialize(model, args, ridge)
            return result.steal_data()
        if operation == "validate_params":
            if not ridge:
                raise Error("TypeError: validate_params requires Ridge")
            classical_ridge_parameters(sequential_fields(model))
            return PythonObject(None).steal_data()
        if operation == "get_params":
            result = classical_get_params(model, ridge)
            return result.steal_data()
        if operation == "set_params":
            result = classical_set_params(model, args[0], ridge)
            return result.steal_data()
        if operation == "is_fitted":
            result = classical_is_fitted(model)
            return result.steal_data()
        if operation == "sklearn_tags":
            result = classical_tags(context)
            return result.steal_data()
        if operation == "model_sha256":
            if not Bool(py=py_get(context, "classical_explicit_cpu", False)):
                raise Error("AttributeError: model_sha256 belongs to the explicit host model interface")
            result = classical_host_hash(model, context)
            return result.steal_data()
        if operation == "save":
            result = classical_archive_save(model, args[0], ridge, context)
            return result.steal_data()
        if operation == "predict_public" or operation == "predict_score" or operation == "score_public":
            classical_require_fitted(model, ridge, "score" if operation == "score_public" else "predict", context)
        var entries = classical_entries(model, context)
        if operation == "ridge_fit":
            if not ridge:
                raise Error("TypeError: ridge_fit requires Ridge")
            result = classical_ridge_fit(model, args, entries, context)
        elif operation == "ols_fit":
            if not linear:
                raise Error("TypeError: ols_fit requires LinearRegression")
            result = classical_ols_fit(model, args, entries, context)
        elif operation == "predict" or operation == "predict_score":
            # Native scorers consume the same float32 kernel result that
            # historical package-internal callers received, without relying
            # on a Python frame to distinguish them from public predict.
            result = classical_linear_predict(model, args[0], entries, context)
        elif operation == "predict_public":
            # A caller outside the package gets scikit-learn's float64
            # regression output; mojolearn's own code (and its identity
            # harness) keeps the float32 prediction the kernels produce.
            if classical_caller_outside(context):
                result = classical_linear_predict_public(model, args[0], entries, context)
            else:
                result = classical_linear_predict(model, args[0], entries, context)
        elif operation == "score" or operation == "score_public":
            result = classical_linear_score(model, args, entries, context)
        elif operation == "numeric_mode_used":
            result = PythonObject("identical" if Int(py=entries["estimators_numeric_mode"]()) == 1 else "fast")
        elif operation == "vendor_used":
            result = entries["estimators_vendor"]()
        else:
            raise Error("unknown private classical operation")
        return result.steal_data()
    except error:
        return classical_python_error(error^, args_ptr)
