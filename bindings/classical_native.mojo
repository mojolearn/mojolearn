# SPDX-License-Identifier: Apache-2.0
# Incremental native classical owner. Public activation waits for full API
# migration; this private entry is exercised directly by strict audit tests.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.classical_error_boundary import classical_python_error
from bindings.sequential_description import sequential_fields
from providers.python.linear_parameter_boundary import classical_ridge_parameters
from providers.python.linear_fit_boundary import classical_ridge_fit
from providers.python.linear_predict_boundary import classical_linear_predict, classical_linear_predict_public
from providers.python.linear_ols_boundary import classical_ols_fit
from providers.python.linear_score_boundary import classical_linear_score
from providers.python.linear_metadata_boundary import classical_initialize, classical_get_params, classical_set_params, classical_is_fitted, classical_parameter_list
from bindings.classical_attributes import classical_check_layout_call, classical_getattribute, classical_connect_numpy
from bindings.classical_archive import classical_archive_save, classical_archive_hash as classical_host_hash
from bindings.classical_archive_read import classical_archive_load, classical_host_model
from bindings.archive_schema_native import archive_schema_save
from bindings.linear_score_native import linear_regressor_score
from bindings.native_operation_codes import operation_table, OP_ROUTE_TREES, OP_ROUTE_HOST_TREE
from bindings.classical_door_routes import (
    CLASSICAL_TREES_DOOR, CLASSICAL_FAMILY_DOOR, classical_trees_operation, classical_family_operation,
)
from bindings.classical_exceptions import classical_require_fitted, classical_error_restore, classical_error_reduce
from bindings.classical_pandas import classical_admit_value, classical_admit_fields
# TOMBSTONE: P1_NATIVE_SKLEARN_TAGS (removed: native code importing sklearn.utils._tags and building its Tag dataclasses; the tags are built by the Python compatibility layer, _mode.sklearn_tags) deleted 2026-11-09 by lane p1-python-callbacks-worst; code recoverable at e68654a757.
# Restore: experiments/removed/P1_NATIVE_SKLEARN_TAGS.patch.
from bindings.classical_validate import classical_refuse_sparse, classical_estimator_name, classical_weight_refusal
from bindings.classical_host_owner import classical_host_entries
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.classical_caller import classical_caller_outside
from bindings.native_loaded_module import native_loaded_module, NativeModules
from bindings.admitted_call import AdmittedCall
from core.linear_fit_types import LF_MODE, LF_VENDOR
from core.linear_metadata import linear_mode_name, linear_vendor_code, linear_vendor_name
from providers.python.linear_execution_boundary import linear_execution_word
from providers.python.linear_entry_boundary import linear_entry_identity


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
    var mode_name = linear_mode_name(String(mode))
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
    var connection_identity = linear_entry_identity(entries)
    var verdicts = py_get(context, "classical_entries_admitted")
    if verdicts is not None:
        var seen = py_get(verdicts, mode)
        if seen is not None and len(seen) == 3 and seen[0] is entries and seen[1] is vendor and seen[2] is connection_identity:
            call.remember_binding("classical", mode_name, True, entries)
            return entries
    if linear_execution_word(entries, LF_VENDOR) != linear_vendor_code(String(vendor)):
        raise Error("RuntimeError: classical native vendor differs from selected execution")
    if linear_execution_word(entries, LF_MODE) != (1 if mode_name == "identical" else 0):
        raise Error("RuntimeError: classical native mode differs from selected execution")
    if verdicts is None:
        verdicts = Python.dict()
        py_setitem(context, "classical_entries_admitted", verdicts)
    py_setitem(verdicts, mode, Python.tuple(entries, vendor, connection_identity))
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


# The 2026-11-07 host split (docs/build/HOT_GROUPS.md section 6): the tree,
# density, saved-model inference and host archive operations, and the
# `family_*` operations, are compiled in their own host bindings
# (bindings/classical_trees_door.mojo, bindings/classical_family_door.mojo).
# Import setup retains their compiled entries in the connection context
# (python/mojolearn/_native_classical_bridge.py `_classical_doors`); this door
# hands them the caller's own argument tuple, so the owner, its steps and its
# error mapping are the ones that ran here before. The door names and the
# operation split are in the leaf bindings/classical_door_routes.mojo.
def classical_door_call(context: PythonObject, key: String, entry: String, args_ptr: PyObjectPtr) raises -> PyObjectPtr:
    """The compiled door `context[key]` (the export `entry` of its host
    binding) called with this call's argument tuple: its result, or NULL with
    the exception it set. Callers spell `entry` as a literal: it is the name
    the door runs (tools/lane_select.py follows a native body's literal export
    names into the binding that registers them)."""
    var door = py_get(context, PythonObject(key))
    if door is None:
        raise Error("ImportError: the compiled " + entry + " is not connected; build the complete host layout and start a fresh process")
    ref cpy = Python().cpython()
    return cpy.PyObject_Call(door._obj_ptr, args_ptr, PyObjectPtr())


def classical_coded_trees(code: Int) raises -> Bool:
    """Whether the operation with this integer code is the tree door's (else
    the family door's)."""
    var table = operation_table()
    if code < 0 or code >= len(table[].names):
        raise Error("TypeError: unknown native operation code")
    var route = table[].routes[code]
    return route == OP_ROUTE_TREES or route == OP_ROUTE_HOST_TREE


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
            # operation and no match chain before the owner, which is a tree
            # or family door (host split, above).
            if classical_coded_trees(Int(py=request[1])):
                return classical_door_call(request[3], CLASSICAL_TREES_DOOR, "trees_native_call", args_ptr)
            return classical_door_call(request[3], CLASSICAL_FAMILY_DOOR, "family_native_call", args_ptr)
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
        if classical_trees_operation(operation):
            return classical_door_call(context, CLASSICAL_TREES_DOOR, "trees_native_call", args_ptr)
        if classical_family_operation(operation):
            return classical_door_call(context, CLASSICAL_FAMILY_DOOR, "family_native_call", args_ptr)
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
            var names = classical_parameter_list(ridge)
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
            result = PythonObject("identical" if linear_execution_word(entries, LF_MODE) == 1 else "fast")
        elif operation == "vendor_used":
            result = PythonObject(linear_vendor_name(linear_execution_word(entries, LF_VENDOR)))
        else:
            raise Error("unknown private classical operation")
        return result.steal_data()
    except error:
        return classical_python_error(error^, args_ptr)
