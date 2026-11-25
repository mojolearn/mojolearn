# SPDX-License-Identifier: Apache-2.0
# Incremental native classical owner. Public activation waits for full API
# migration; this private entry is exercised directly by strict audit tests.
from providers.python.device_route_boundary_door import archive_load_device
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.classical_error_boundary_pack import classical_python_error
from bindings.sequential_description_unpack import sequential_fields
from providers.python.linear_parameter_boundary_unpack import classical_ridge_parameters
from providers.python.linear_fit_boundary_unpack import classical_ridge_fit
from providers.python.linear_predict_boundary_unpack import classical_linear_predict
from providers.python.linear_ols_boundary_unpack import classical_ols_fit
from providers.python.linear_score_boundary_unpack import classical_linear_score
from providers.python.linear_metadata_boundary_door import classical_initialize, classical_get_params, classical_set_params, classical_is_fitted, classical_parameter_list
from bindings.classical_attributes_door import classical_check_layout_call, classical_getattribute, classical_connect_numpy
from bindings.classical_archive_door import classical_archive_save, classical_archive_hash as classical_host_hash
from bindings.classical_archive_read_door import classical_archive_load, classical_host_model
from bindings.archive_schema_native_door import archive_schema_save
from bindings.archive_npz_door import archive_npz_call
from bindings.fitted_digest_door import fitted_digest_call
from bindings.native_operation_codes import operation_table, OP_LEAF_NONE, OP_ROUTE_CLASSICAL, OP_ROUTE_FAMILY
from bindings.native_operation_names_unpack import operation_lookup, operation_code
from bindings.classical_native_run import classical_coded_trees
from bindings.classical_door_routes import classical_trees_operation, classical_family_operation
from bindings.classical_exceptions_door import classical_require_fitted, classical_error_restore, classical_error_reduce
from bindings.classical_pandas_door import classical_admit_value, classical_admit_fields
# TOMBSTONE: P1_NATIVE_SKLEARN_TAGS (removed: native code importing sklearn.utils._tags and building its Tag dataclasses; the tags are built by the Python compatibility layer, _mode.sklearn_tags) deleted 2026-11-09 by lane p1-python-callbacks-worst; code recoverable at e68654a757.
# Restore: experiments/removed/P1_NATIVE_SKLEARN_TAGS.patch.
from bindings.classical_validate_unpack import classical_refuse_sparse, classical_estimator_name, classical_weight_refusal
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from bindings.admitted_call_door import AdmittedCall
from core.linear_metadata import linear_vendor_name, linear_constructor_check, linear_constructor_device
from providers.python.linear_route_boundary_door import linear_route_call, linear_device_admit
from providers.python.device_route_boundary_door import device_route_clear
from bindings.classical_keys_unpack import classical_keys
from bindings.operation_names_classical import (
    CLASSICAL_TREES_FIRST, CLASSICAL_LEAF_TREES_DOOR, CLASSICAL_LEAF_FAMILY_DOOR,
    CLASSICAL_OPN_ARCHIVE_NPZ, CLASSICAL_OPN_ARCHIVE_SAVE, CLASSICAL_OPN_FITTED_DIGEST,
    CLASSICAL_OPN_GETATTRIBUTE, CLASSICAL_OPN_GET_PARAMS,
    CLASSICAL_OPN_HOST_MODEL, CLASSICAL_OPN_INITIALIZE, CLASSICAL_OPN_IS_FITTED,
    CLASSICAL_OPN_LOAD, CLASSICAL_OPN_MODEL_SHA256,
    CLASSICAL_OPN_NUMERIC_MODE_USED, CLASSICAL_OPN_OLS_FIT, CLASSICAL_OPN_OPERATION_CODES,
    CLASSICAL_OPN_PARAM_NAMES, CLASSICAL_OPN_PREDICT, CLASSICAL_OPN_PREDICT_PUBLIC, CLASSICAL_OPN_PREDICT_SCORE,
    CLASSICAL_OPN_REDUCE_NOT_FITTED, CLASSICAL_OPN_RESTORE_NOT_FITTED, CLASSICAL_OPN_RIDGE_FIT, CLASSICAL_OPN_SAVE,
    CLASSICAL_OPN_SCORE, CLASSICAL_OPN_SCORE_PUBLIC, CLASSICAL_OPN_SET_PARAMS, CLASSICAL_OPN_VALIDATE_PARAMS,
    CLASSICAL_OPN_VENDOR_USED,
)
from bindings.py_balanced import py_item


# TOMBSTONE: `classical_entries` (the per-call entry-family selection: the mode String, the bindings and
# verdict dictionary reads, the vendor and tier questions; replaced by the estimator's route owner, resolved once,
# providers/python/linear_route_boundary_door.mojo `linear_route_call`) deleted 2026-11-20 by lane
# pilot-ridge-device; code recoverable at bc0418b70a.


def operation_code_names() raises -> PythonObject:
    """The operation names of the code table, in code order, as a tuple of
    str (the Python shells map each name to its index once, at import)."""
    var table = operation_table()
    var names = Python.list()
    for i in range(len(table[].names)):  # driver-loop: over the fixed operation names, once per process
        if table[].routes[i] == OP_ROUTE_CLASSICAL:
            # The classical door's own string operations are registered last
            # and never passed by code: the codes Python maps end here.
            break
        _ = py_call(names.append, PythonObject(table[].names[i]))
    return py_call(native_loaded_module("builtins").tuple, names)


@always_inline
def classical_operation_leaf(name: PythonObject) raises -> Int:
    """The leaf of a caller's operation str: ONE dictionary lookup of the str
    object itself in the operation table (no String is built). A name of this
    door or its tree door gives its `CLASSICAL_OPN_*` code; a `trees_*` /
    `host_tree_*` table name gives CLASSICAL_LEAF_TREES_DOOR, a `family_*`
    table name CLASSICAL_LEAF_FAMILY_DOOR; a name the table does not hold
    gives OP_LEAF_NONE."""
    var names = operation_lookup()
    var code = operation_code(names, name)
    if code < 0:
        return OP_LEAF_NONE
    var route = names[].table.routes[code]
    if route == OP_ROUTE_CLASSICAL:
        return names[].table.leaves[code]
    if route == OP_ROUTE_FAMILY:
        return CLASSICAL_LEAF_FAMILY_DOOR
    return CLASSICAL_LEAF_TREES_DOOR


# The 2026-11-07 host split (docs/build/HOT_GROUPS.md section 6): the tree,
# density, saved-model inference and host archive operations, and the
# `family_*` operations, are compiled in their own host bindings
# (bindings/classical_trees_door.mojo, bindings/classical_family_door.mojo).
# Import setup retains their compiled entries in the connection context
# (python/mojolearn/_native_classical_bridge.py `_classical_doors`); this door
# hands them the caller's own argument tuple, so the owner, its steps and its
# error mapping are the ones that ran here before. The door names and the
# operation split are in the leaf bindings/classical_door_routes.mojo.
def classical_door_call(context: PythonObject, key: PythonObject, entry: String, args_ptr: PyObjectPtr) raises -> PyObjectPtr:
    """The compiled door `context[key]` (the export `entry` of its host
    binding) called with this call's argument tuple: its result, or NULL with
    the exception it set. `key` is the held key of the door's context name
    (bw2-classical: it was a str built from a String per call). Callers spell
    `entry` as a literal: it is the name the door runs (tools/lane_select.py
    follows a native body's literal export names into the binding that
    registers them)."""
    var door = py_get(context, key)
    if door is None:
        raise Error("ImportError: the compiled " + entry + " is not connected; build the complete host layout and start a fresh process")
    ref cpy = Python().cpython()
    return cpy.PyObject_Call(door._obj_ptr, args_ptr, PyObjectPtr())


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
        var nk = classical_keys()
        if py_type(request[1]) is b.int and py_type(request[3]) is b.dict:
            # Lane ne1-h-labels: an integer operation code from the one table
            # (bindings/native_operation_codes.mojo): no String copy of the
            # operation and no match chain before the owner, which is a tree
            # or family door (host split, above).
            if classical_coded_trees(Int(py=request[1])):
                return classical_door_call(request[3], nk[].k_classical_trees_door, "trees_native_call", args_ptr)
            return classical_door_call(request[3], nk[].k_classical_family_door, "family_native_call", args_ptr)
        if py_type(request[1]) is not b.str or py_type(request[3]) is not b.dict:
            raise Error("TypeError: native classical operations require an exact operation string and context dictionary")
        var model = request[0]
        # bw2-classical: the caller's own str is resolved by ONE dictionary
        # lookup in the operation table to this door's leaf code (or to the
        # door that owns a table name). It was copied into a String and walked
        # through up to 26 exact compares and the tree door's 23 name and
        # prefix tests on every call. Only a name the table does not hold
        # builds its String, for the old prefix routes; any other unknown
        # name keeps OP_LEAF_NONE, matches no branch below and gets the same
        # error at the same step as before.
        var operation = classical_operation_leaf(request[1])
        var args = request[2]
        var context = request[3]
        if operation == CLASSICAL_LEAF_TREES_DOOR or operation >= CLASSICAL_TREES_FIRST:
            return classical_door_call(context, nk[].k_classical_trees_door, "trees_native_call", args_ptr)
        if operation == CLASSICAL_LEAF_FAMILY_DOOR:
            return classical_door_call(context, nk[].k_classical_family_door, "family_native_call", args_ptr)
        if operation == OP_LEAF_NONE:
            var name = String(request[1])
            if classical_trees_operation(name):
                return classical_door_call(context, nk[].k_classical_trees_door, "trees_native_call", args_ptr)
            if classical_family_operation(name):
                return classical_door_call(context, nk[].k_classical_family_door, "family_native_call", args_ptr)
        if operation == CLASSICAL_OPN_OPERATION_CODES:
            # Read once at import by python/mojolearn/_native_ops.py.
            var names = operation_code_names()
            return names.steal_data()
        if operation == CLASSICAL_OPN_RESTORE_NOT_FITTED:
            var error = classical_error_restore(args, context)
            return error.steal_data()
        if operation == CLASSICAL_OPN_REDUCE_NOT_FITTED:
            var reduced = classical_error_reduce(model, context)
            return reduced.steal_data()
        # TOMBSTONE (lane dw-xlinear, 2026-11-20): operation
        # "linear_regressor_score" (bindings/linear_score_native_unpack.mojo:
        # the linear expansion regressors' score, which selected the process
        # images by name on every call) is the x_linear door's
        # "regressor_score", on the estimator's route. Code recoverable at
        # 0121caad3d.
        if operation == CLASSICAL_OPN_ARCHIVE_SAVE:
            # The public estimators' one-call save: the class's connected
            # schema, written by the deterministic NPZ writer.
            var saved = archive_schema_save(model, args[0], context)
            return saved.steal_data()
        if operation == CLASSICAL_OPN_FITTED_DIGEST:
            # `model.digest_` (args ()) and `model.training_digest_` (args
            # ("training",)): hashed at each read, nothing kept
            # (bindings/fitted_digest_door.mojo). Never reached by a fit.
            var digest = fitted_digest_call(model, args, context)
            return digest.steal_data()
        if operation == CLASSICAL_OPN_ARCHIVE_NPZ:
            # `mojolearn._serialize`: the npz/npy codec helpers, one native
            # call each (bindings/archive_npz_door.mojo). An uncoded string
            # operation beside archive_save; coded operations never reach it.
            var npz = archive_npz_call(args, context)
            return npz.steal_data()
        if operation == CLASSICAL_OPN_HOST_MODEL:
            var host_context = py_get(context, nk[].k_classical_host_context)
            if host_context is None:
                raise Error("RuntimeError: the explicit host linear models are not connected")
            classical_connect_numpy(host_context)
            classical_admit_value(call.modules, args, host_context)
            var hosted = classical_host_model(args[0], host_context)
            return hosted.steal_data()
        var loading = operation == CLASSICAL_OPN_LOAD
        var class_operation = loading or operation == CLASSICAL_OPN_PARAM_NAMES
        var cls = model if class_operation else py_type(model)
        var host = py_get(context, nk[].k_classical_host_context)
        if host is not None and py_contains(py_item(host, nk[].k_classical_kinds), cls):
            context = host
        classical_connect_numpy(context)
        # classical_kinds maps each connected class (Ridge, LinearRegression,
        # their explicit host classes and subclasses) to True for the Ridge
        # family and False for LinearRegression.
        var kind = py_get(py_item(context, nk[].k_classical_kinds), cls)
        if kind is None:
            raise Error("TypeError: the native classical owner requires a connected Ridge or LinearRegression class")
        var ridge = Bool(py=kind)
        var linear = not ridge
        if operation == CLASSICAL_OPN_PARAM_NAMES:
            var names = classical_parameter_list(ridge)
            return names.steal_data()
        if not class_operation:
            _ = classical_check_layout_call(call, model, context)
        if operation == CLASSICAL_OPN_GETATTRIBUTE:
            var attribute = classical_getattribute(call, model, args[0], context)
            return attribute.steal_data()
        if loading:
            var loaded = classical_archive_load(model, args[0], ridge, context)
            if len(args) > 1:
                # `load(path, device=...)` (gd-verify-view): the one shared device step
                archive_load_device(loaded, args[1], context)
            return loaded.steal_data()
        if operation == CLASSICAL_OPN_RIDGE_FIT or operation == CLASSICAL_OPN_OLS_FIT or operation == CLASSICAL_OPN_PREDICT_PUBLIC or operation == CLASSICAL_OPN_PREDICT_SCORE or operation == CLASSICAL_OPN_SCORE_PUBLIC:
            # scikit-learn's sparse refusal comes before input admission.
            classical_refuse_sparse(args[0], classical_estimator_name(model))
        # The record's module handle serves this call's input and field admissions.
        if (operation == CLASSICAL_OPN_RIDGE_FIT or operation == CLASSICAL_OPN_OLS_FIT) and len(args) == 3:
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
        elif operation != CLASSICAL_OPN_SAVE:
            classical_admit_value(call.modules, args, context)
        call.admit_fields(sequential_fields(model))
        # One admission pass with one set of type vectors for every field.
        classical_admit_fields(call.modules, call.fields, py_item(context, nk[].k_classical_fields), context)
        var result = PythonObject(None)
        if operation == CLASSICAL_OPN_INITIALIZE:
            # `device` is checked where it is given: an unknown value, and an
            # explicit 'gpu' in a CPU-only process, are refused in words
            # before any field is published. The default None (this
            # installation's device) is one identity compare. It is stored
            # verbatim like every constructor parameter and read once, by the
            # estimator's first operation, into its route owner
            # (providers/python/linear_route_boundary_door.mojo). A default
            # construction costs, more than before: that compare and one
            # lookup of the estimator's address in the model record table
            # (the route clear below). No owner is allocated here.
            linear_constructor_check(len(args), ridge)
            linear_device_admit(args[linear_constructor_device(ridge)], context, b)
            result = classical_initialize(model, args, ridge, context)
            # A constructor run again on a live estimator: its next operation
            # reads the device and the mode again.
            device_route_clear(model)
            return result.steal_data()
        if operation == CLASSICAL_OPN_VALIDATE_PARAMS:
            if not ridge:
                raise Error("TypeError: validate_params requires Ridge")
            classical_ridge_parameters(call.fields)
            return PythonObject(None).steal_data()
        if operation == CLASSICAL_OPN_GET_PARAMS:
            result = classical_get_params(model, ridge)
            return result.steal_data()
        if operation == CLASSICAL_OPN_SET_PARAMS:
            if py_type(args[0]) is b.dict:
                # A new device is refused in words before the estimator changes.
                var device_key = PythonObject("device")
                if py_contains(args[0], device_key):
                    linear_device_admit(py_item(args[0], device_key), context, b)
            result = classical_set_params(model, args[0], ridge, context)
            # The device or the numeric mode may have changed: the next
            # operation resolves the route again.
            device_route_clear(model)
            return result.steal_data()
        if operation == CLASSICAL_OPN_IS_FITTED:
            result = classical_is_fitted(model)
            return result.steal_data()
        if operation == CLASSICAL_OPN_MODEL_SHA256:
            if not Bool(py=py_get(context, nk[].k_classical_explicit_cpu, False)):
                raise Error("AttributeError: model_sha256 belongs to the explicit host model interface")
            result = classical_host_hash(model, context)
            return result.steal_data()
        if operation == CLASSICAL_OPN_SAVE:
            result = classical_archive_save(model, args[0], ridge, context)
            return result.steal_data()
        if operation == CLASSICAL_OPN_PREDICT_PUBLIC or operation == CLASSICAL_OPN_PREDICT_SCORE or operation == CLASSICAL_OPN_SCORE_PUBLIC:
            classical_require_fitted(model, ridge, "score" if operation == CLASSICAL_OPN_SCORE_PUBLIC else "predict", context)
        # The estimator's device route: resolved once per estimator (its
        # `device` and numeric mode, the device table or the CPU table), read
        # here as one record and three pointer compares. The operations below
        # call through its entries; none of them tests the device.
        var routed = linear_route_call(model, call.fields, ridge, context, b)
        var route = routed.route
        if operation == CLASSICAL_OPN_RIDGE_FIT:
            if not ridge:
                raise Error("TypeError: ridge_fit requires Ridge")
            result = classical_ridge_fit(model, args, route, routed.owner, context)
        elif operation == CLASSICAL_OPN_OLS_FIT:
            if not linear:
                raise Error("TypeError: ols_fit requires LinearRegression")
            result = classical_ols_fit(model, args, route, routed.owner, context)
        elif (operation == CLASSICAL_OPN_PREDICT or operation == CLASSICAL_OPN_PREDICT_SCORE
              or operation == CLASSICAL_OPN_PREDICT_PUBLIC):
            # One result for every caller (Andrew, 2026-11-22, lane
            # predict-dtype): the kernel's float32 predictions, with no
            # caller test and no float64 widening.
            result = classical_linear_predict(model, args[0], route, routed.owner, context)
        elif operation == CLASSICAL_OPN_SCORE or operation == CLASSICAL_OPN_SCORE_PUBLIC:
            result = classical_linear_score(model, args, route, routed.owner, context)
        elif operation == CLASSICAL_OPN_NUMERIC_MODE_USED:
            # The words the route's compiled fit entry reported when it was resolved.
            result = PythonObject("identical" if route.mode == 1 else "fast")
        elif operation == CLASSICAL_OPN_VENDOR_USED:
            result = PythonObject(linear_vendor_name(route.vendor))
        else:
            raise Error("unknown private classical operation")
        return result.steal_data()
    except error:
        return classical_python_error(error^, args_ptr)
