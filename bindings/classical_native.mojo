# SPDX-License-Identifier: Apache-2.0
# Incremental native classical owner. Public activation waits for full API
# migration; this private entry is exercised directly by strict audit tests.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.classical_error_boundary import classical_python_error
from bindings.sequential_description import sequential_fields
from bindings.classical_ridge import classical_ridge_fit, classical_ridge_parameters, classical_linear_predict, classical_linear_predict_public
from bindings.classical_ols import classical_ols_fit
from bindings.classical_score import classical_linear_score
from bindings.classical_metadata import classical_initialize, classical_get_params, classical_set_params, classical_is_fitted, classical_parameter_names
from bindings.classical_attributes import classical_check_layout_call, classical_getattribute, classical_connect_numpy
from bindings.classical_archive import classical_archive_save
from bindings.classical_archive_read import classical_archive_load, classical_host_model
from bindings.host_archive_native import host_archive_build
from bindings.host_archive_predict_native import host_archive_predict
from bindings.host_knn_predict_native import host_knn_predict
from bindings.archive_schema_native import archive_schema_save
from bindings.linear_score_native import linear_regressor_score
from bindings.family_native import family_operation, family_operation_routed
from bindings.native_operation_codes import operation_table, OP_ROUTE_TREES, OP_ROUTE_HOST_TREE
from bindings.host_tree_native import host_tree_operation
from bindings.trees_public_native import trees_public_operation
from bindings.host_inference_native import host_inference_operation
from bindings.host_kde_fit_native import host_kde_fit
from bindings.host_iforest_fit_native import host_iforest_fit
from bindings.host_density_constructor_native import host_density_initialize
from bindings.host_density_archive_native import host_density_save
from bindings.host_density_hash_native import host_density_hash
from bindings.classical_exceptions import classical_require_fitted, classical_error_restore, classical_error_reduce
from bindings.classical_pandas import classical_admit_value, classical_admit_fields
from bindings.classical_tags import classical_tags
from bindings.classical_validate import classical_refuse_sparse, classical_estimator_name, classical_weight_refusal
from bindings.classical_host_owner import classical_host_entries, classical_host_hash
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.native_caller_frame import native_caller_globals
from bindings.native_loaded_module import native_loaded_module, NativeModules
from bindings.admitted_call import AdmittedCall
from providers.python.classical_family_admission_pyapi import classical_family_admit


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


def classical_entries(mut call: AdmittedCall, model: PythonObject, context: PythonObject) raises -> PythonObject:
    # The door's admission record supplies the model fields and builtins it
    # already admitted (ne1-e-admitted-call); the resolved entry family is
    # kept on the record as this call's binding.
    if Bool(py=py_get(context, "classical_explicit_cpu", False)):
        return classical_host_entries(model, context)
    if not call.fields_admitted:
        call.admit_fields(sequential_fields(model))
    var mode = py_get(call.fields, "numeric_mode")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    var b = call.builtins()
    if py_type(mode) is not b.str:
        raise Error("numeric_mode must be fast, identical or None")
    # `mode.strip().lower()` in native string operations, once: no Python
    # method lookups or calls. ASCII whitespace and case are handled here; a
    # word that is not then exactly "identical" or "fast" takes Python's
    # Unicode-aware strip/lower (same refusal, same accepted spellings).
    var mode_name = String(String(String(mode).strip()).lower())
    if mode_name != "identical" and mode_name != "fast":
        mode_name = String(mode.strip().lower())
    if mode_name != "identical" and mode_name != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    mode = PythonObject(mode_name)
    var entries = context["classical_bindings"][mode]
    if py_type(entries) is b.str:
        raise Error("ImportError: " + String(entries))
    # The vendor and mode of a connected entry family are fixed when its
    # binding module loads. Check them once per (entry family, selected
    # vendor) identity and keep the verdict in the long-lived connection
    # context; a reconnection installs new entry dictionaries (or a new
    # vendor object), so the identities differ and the check runs again.
    var vendor = context["vendor"]
    var verdicts = py_get(context, "classical_entries_admitted")
    if verdicts is not None:
        var seen = py_get(verdicts, mode)
        if seen is not None and seen[0] is entries and seen[1] is vendor:
            call.remember_binding("classical", mode_name, True, entries)
            return entries
    if String(entries["estimators_vendor"]()) != String(vendor):
        raise Error("RuntimeError: classical native vendor differs from selected execution")
    if Int(py=entries["estimators_numeric_mode"]()) != (1 if mode_name == "identical" else 0):
        raise Error("RuntimeError: classical native mode differs from selected execution")
    if verdicts is None:
        verdicts = Python.dict()
        py_setitem(context, "classical_entries_admitted", verdicts)
    py_setitem(verdicts, mode, Python.tuple(entries, vendor))
    call.remember_binding("classical", mode_name, True, entries)
    return entries


def operation_code_names() raises -> PythonObject:
    """The operation names of the code table, in code order, as a tuple of
    str (the Python shells map each name to its index once, at import)."""
    var table = operation_table()
    var names = Python.list()
    for i in range(len(table[].names)):  # driver-loop: over the fixed operation names, once per process
        _ = py_call(names.append, PythonObject(table[].names[i]))
    return py_call(native_loaded_module("builtins").tuple, names)


def classical_coded_operation(model: PythonObject, code: Int, args: PythonObject,
                              context: PythonObject) raises -> PythonObject:
    """One `family_*`, `trees_*` or `host_tree_*` operation by its integer
    code: the route and the name come from the table, and the owner is
    called as the str path calls it."""
    var table = operation_table()
    if code < 0 or code >= len(table[].names):
        raise Error("TypeError: unknown native operation code")
    var route = table[].routes[code]
    var name = table[].names[code].copy()
    if route == OP_ROUTE_TREES:
        return trees_public_operation(model, name, args, context)
    if route == OP_ROUTE_HOST_TREE:
        return host_tree_operation(model, name, args, context["host_tree_context"])
    _ = classical_family_admit(model)
    return family_operation_routed(model, table[].family_routes[code], name, args, context)


def classical_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var request = PythonObject(from_borrowed=args_ptr)
        if len(request) != 4:
            raise Error("TypeError: classical_native_call requires model, operation, arguments and context")
        # The one admission record of this public call: modules, the entry
        # family, the model fields and the layout verdict are admitted once
        # here and passed by reference to the helpers below.
        var call = AdmittedCall()
        var b = call.builtins()
        if py_type(request[1]) is b.int and py_type(request[3]) is b.dict:
            # Lane ne1-h-labels: an integer operation code from the one table
            # (bindings/native_operation_codes.mojo): no String copy of the
            # operation and no match chain before the owner.
            var coded = classical_coded_operation(request[0], Int(py=request[1]), request[2], request[3])
            return coded.steal_data()
        if py_type(request[1]) is not b.str or py_type(request[3]) is not b.dict:
            raise Error("TypeError: native classical operations require an exact operation string and context dictionary")
        var model = request[0]
        var operation = String(request[1])
        var args = request[2]
        var context = request[3]
        if operation == "operation_codes":
            # Read once at import by python/mojolearn/_native_ops.py.
            var names = operation_code_names()
            return names.steal_data()
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
            _ = classical_family_admit(model)
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
            classical_admit_value(call.modules, args, host_context)
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
            _ = classical_check_layout_call(call, model, context)
        if operation == "getattribute":
            var attribute = classical_getattribute(call, model, args[0], context)
            return attribute.steal_data()
        if loading:
            var loaded = classical_archive_load(model, args[0], ridge, context)
            return loaded.steal_data()
        if operation == "ridge_fit" or operation == "ols_fit" or operation == "predict_public" or operation == "predict_score" or operation == "score_public":
            # scikit-learn's sparse refusal comes before input admission.
            classical_refuse_sparse(args[0], classical_estimator_name(model))
        # The record's module handle serves this call's input and field admissions.
        if (operation == "ridge_fit" or operation == "ols_fit") and len(args) == 3:
            # Same order as the whole-tuple admission (X, y, then the
            # weight); a refused sample_weight names the accepted types
            # (9bdccfa76) instead of the shared admission's generic text.
            classical_admit_value(call.modules, args[0], context, 1)
            classical_admit_value(call.modules, args[1], context, 1)
            try:
                classical_admit_value(call.modules, args[2], context, 1)
            except error:
                var refusal = classical_weight_refusal(args[2], error)
                raise refusal
        elif operation != "save":
            classical_admit_value(call.modules, args, context)
        call.admit_fields(sequential_fields(model))
        # One admission pass with one set of type vectors for every field.
        classical_admit_fields(call.modules, call.fields, context["classical_fields"], context)
        var result = PythonObject(None)
        if operation == "initialize":
            result = classical_initialize(model, args, ridge, context)
            return result.steal_data()
        if operation == "validate_params":
            if not ridge:
                raise Error("TypeError: validate_params requires Ridge")
            classical_ridge_parameters(call.fields)
            return PythonObject(None).steal_data()
        if operation == "get_params":
            result = classical_get_params(model, ridge)
            return result.steal_data()
        if operation == "set_params":
            result = classical_set_params(model, args[0], ridge, context)
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
        var entries = classical_entries(call, model, context)
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
