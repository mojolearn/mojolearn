# SPDX-License-Identifier: Apache-2.0
# UNPACK part of the public cluster lifecycle (MiniBatchKMeans,
# BisectingKMeans, MeanShift, OPTICS, AffinityPropagation, the x_cluster
# queries and BayesianGaussianMixture's): the model admission, its per-class
# and per-model records, the replaced-state admission of a fit, X and the
# public arguments, read once into `ClusterLifecycleCall`. The operator and
# PACK are providers/python/cluster_public_lifecycle_pyapi_pack.mojo; the
# method classification is RUN's (cluster/cluster_public_lifecycle_run.mojo).
from bindings.py_cpython_table import cpython_table
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import NativeModules, native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_fields, py_item
from bindings.mode_metadata_native_door import _fields, _class_fields
from bindings.native_class_admission_door import class_record, class_admit
from bindings.cluster_side_records_unpack import cluster_record_drop
from bindings.cluster_side_records_pack import cluster_record_key, cluster_record_get
from bindings.public_attributes_native_door import _plan, _lookup, _string_keys
from bindings.classical_attributes_door import _type_mro
from bindings.classical_pandas_door import classical_admit_value
from bindings.classical_validate_unpack import classical_check_x, classical_estimator_name
from bindings.classical_validate_run import classical_check_width
from bindings.classical_exceptions_door import classical_require_fitted
from bindings.native_owner_lifetime_door import native_release_admit
from bindings.host_inference_scalars_door import host_inference_scalar
from providers.python.estimator_binding_pyapi_door import estimator_binding
from providers.python.estimator_lifecycle_pyapi_unpack import estimator_feature_check
from providers.python.estimator_tabular_boundary_pyapi_unpack import _target
from bindings.py_optional import py_present
from cluster.cluster_public_lifecycle_run import (
    _method, _method_text, cluster_method_fitting, cluster_depth_nested, cluster_width_applies,
    cluster_argument_stop)
from bindings.cluster_keys_unpack import cluster_keys


def _scalar_state(value: PythonObject, context: PythonObject, depth: Int = 0) raises:
    if depth > 8:
        raise Error("TypeError: cluster scalar state exceeds its native nesting depth")
    var b = native_loaded_module("builtins")
    var kind = py_type(value)
    if kind is b.tuple or kind is b.list:
        for i in range(len(value)):  # driver-loop: native partial-fit seed and scalar state words
            _scalar_state(value[i], context, depth + 1)
    else:
        host_inference_scalar(value, context)


# Per-model and per-class records (lane perf3-cluster-gp-gate; the helpers
# of bindings/cluster_side_records.mojo and bindings/native_class_admission.mojo).
# A query (predict, transform, score, ...) of KMeans, BayesianGaussianMixture
# and the other x_cluster / mixture / cfa classes re-ran the whole admission
# below on every call: the ancestry walks, the attribute plan, every
# constructor parameter and every fitted field through classical_admit_value.
# The class facts (the connected parameter schema, no Python field setter, no
# fit callback) are kept per class state: any change of the class or an
# ancestor drops them. The model facts are kept per model, outside its
# `__dict__`, under an identity key of every parameter and learned field the
# admission read (cluster_record_key: exact containers item by item), so
# set_params, a refit, an assignment or an in-place edit of a stored list,
# tuple or dict misses the record and the full admission runs again with its
# refusals in their original order. Not noticed: an in-place write into a
# stored array's values (its type, dtype and identity are unchanged, which is
# all the admission checks of it). A model is recorded only when every
# parameter and learned field was read from its instance dictionary.
comptime _XC_CLASS_SLOT = "cluster_public_class_v1"
comptime _XC_ADMIT_SLOT = "cluster_public_admitted_v1"


def _learned_names() raises -> PythonObject:
    return Python.tuple("cluster_centers_", "labels_", "n_features_in_", "counts_", "_partial_init", "_partial_batch", "_partial_state", "tree_nodes_", "_tree_centers", "weights_", "means_", "precisions_cholesky_", "log_det_chol_", "_log_consts", "_full_pchol", "_score_words", "_warm", "n_steps_", "lower_bound_")


def _query_key(model: PythonObject) raises -> PythonObject:
    """The identity key of a query's admission, or None (no class record yet,
    or a parameter not stored in the instance dictionary)."""
    var modules = NativeModules()
    var parameters = class_record(modules, _XC_CLASS_SLOT, py_type(model))
    if parameters is None:
        return PythonObject(None)
    var fields = _fields(modules, model)
    return cluster_record_key(model, fields, parameters, _learned_names())


def _admit_model(model: PythonObject, context: PythonObject, mut direct: Bool) raises -> PythonObject:
    """The full admission. `direct` is set when every parameter and learned
    field it read came from the instance dictionary (a per-model record may
    then stand for it)."""
    direct = True
    var b = native_loaded_module("builtins")
    var module_fields = py_fields(native_loaded_module("mojolearn._native_public_attributes"))
    _string_keys(module_fields)
    var nk = cluster_keys()
    var attributes = py_get(module_fields, nk[].k__CONTEXT)
    if py_type(attributes) is not b.dict:
        raise Error("TypeError: native cluster attribute metadata requires its connected dictionary")
    _string_keys(attributes)
    var modules = NativeModules()
    var plan = _plan(modules, model, attributes)
    var fields = _fields(model)
    var ancestry = _type_mro(py_type(model))
    # Optional, not a PythonObject read and reassigned in a loop (modular/modular#7546 pattern; runtime-fix-save-forest sweep).
    var found_parameters = Optional[PythonObject]()
    for i in range(len(ancestry)):  # driver-loop: passive subclasses share their connected native owner
        if not found_parameters:
            var candidate = py_get(py_item(context, nk[].k_cluster_public_parameters), ancestry[i])
            if candidate is not None:
                found_parameters = candidate
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var parameters = Optional[PythonObject]()
    if found_parameters:
        parameters = py_present(found_parameters.value())
    if not parameters:
        raise Error("TypeError: cluster execution requires a connected native estimator class")
    var setter_seen = False
    for i in range(len(ancestry)):  # driver-loop: admit native field writes before entering algorithm owners
        var names = _class_fields(ancestry[i])
        if not setter_seen and py_contains(names, nk[].k___setattr__):
            if py_item(names, nk[].k___setattr__) is not py_item(b.object.__dict__, nk[].k___setattr__):
                raise Error("TypeError: native cluster execution cannot invoke Python field setters")
            setter_seen = True
        if py_contains(names, nk[].k__mojolearn_after_fit):
            raise Error("TypeError: native cluster execution cannot invoke Python fit callbacks")
    if py_contains(fields, nk[].k__mojolearn_after_fit):
        raise Error("TypeError: native cluster execution cannot invoke Python fit callbacks")
    for i in range(len(parameters.value())):  # driver-loop: fixed model parameter metadata, never a Python property
        var name = parameters.value()[i]
        if String(name) == "device" and not py_contains(fields, name):
            # An instance without `device` (built without its constructor,
            # saved before the parameter existed) runs this installation's
            # device: the route reads it with that default
            # (device_route_boundary_door `device_request`). Reading it here
            # through the public attribute lookup refused every query of such
            # an instance ('native model has no attribute device').
            direct = False
            continue
        var value = _lookup(model, name, fields, plan, attributes)
        direct = direct and py_contains(fields, name) and value is fields[name]
        if String(name) == "init":
            if cpython_table()[].PyCallable_Check(value._obj_ptr) != 0:  # resolved once per process
                raise Error("TypeError: Python init callbacks are not accepted; pass explicit initial centers or use 'k-means++' or 'random'")
        classical_admit_value(value, context)
        if String(name) == "metric_params" and value is not None:
            # OPTICS consumes only p from this native mapping. An admitted
            # object ndarray still must not invoke element conversion hooks.
            var mappings = py_item(context, nk[].k_mapping_types)
            var mapping = False
            for j in range(len(mappings)):  # driver-loop: exact native mapping owners, before method lookup
                mapping = mapping or py_type(value) is mappings[j]
            if not mapping:
                raise Error("TypeError: OPTICS metric_params requires a native mapping or None")
            var p = py_call(value.get, "p", PythonObject(None))
            if p is not None:
                host_inference_scalar(p, context)
        if (not py_contains(py_item(context, nk[].k_cluster_public_array_parameters), name)
                or (py_type(value) is py_get(context, nk[].k_ndarray) and Int(py=value.ndim) == 0)):
            # Exact ndarray admission alone would still permit an object-dtype
            # scalar to execute Python __float__/__int__ during old-owner casts.
            host_inference_scalar(value, context, True)
    var learned = _learned_names()
    for i in range(len(learned)):  # driver-loop: admit only fitted storage the cluster owners actually consume
        if py_contains(fields, learned[i]):
            var value = _lookup(model, learned[i], fields, plan, attributes)
            direct = direct and value is fields[learned[i]]
            classical_admit_value(value, context)
            var name = String(learned[i])
            if name == "n_features_in_" or name == "_partial_batch" or name == "n_steps_" or name == "lower_bound_":
                host_inference_scalar(value, context)
            elif name == "_partial_init" or name == "_partial_state":
                _scalar_state(value, context)
            elif name == "_warm" and value is not None:
                if (py_type(value) is not b.tuple and py_type(value) is not b.list) or len(value) != 2:
                    raise Error("TypeError: cluster warm state requires its native key and array pair")
                _scalar_state(value[0], context)
    # The class-level checks above passed for this class state: keep its
    # parameter schema so a later query can build its identity key.
    class_admit(modules, _XC_CLASS_SLOT, py_type(model), parameters.value())
    return fields


def _admit_replaced_state(model: PythonObject, fields: PythonObject, context: PythonObject) raises:
    var b = native_loaded_module("builtins")
    var ancestry = _type_mro(py_type(model))
    # Optional, not a PythonObject read and reassigned in a loop (modular/modular#7546 pattern; runtime-fix-save-forest sweep).
    var found_names = Optional[PythonObject]()
    var nk = cluster_keys()
    for i in range(len(ancestry)):  # driver-loop: use the connected owner's fit-state schema
        if not found_names:
            var candidate = py_get(py_item(context, nk[].k_cluster_public_replaced_fields), ancestry[i])
            if candidate is not None:
                found_names = candidate
    var names = PythonObject(None)
    if found_names:
        names = found_names.value()
    if py_type(names) is not b.tuple:
        raise Error("TypeError: cluster refit requires its native replaced-field schema")
    for i in range(len(names)):  # driver-loop: prior fitted owners before any numerical/state mutation
        if py_type(names[i]) is not b.str:
            raise Error("TypeError: cluster fitted-field names require exact strings")
        if py_contains(fields, names[i]):
            native_release_admit(fields[names[i]], context)
    for name in ["_mojolearn_fitted", "_mojolearn_fitting", "feature_names_in_"]:  # driver-loop: shared lifecycle writes/removals
        if py_contains(fields, name):
            native_release_admit(fields[name], context)


@fieldwise_init
struct ClusterLifecycleCall(Movable):
    """One admitted public call: the instance dictionary, the record key of a
    query (None for a fit), X as given and as admitted, the owners' argument
    tuple (the x_cluster image of the estimator's route first), and the typed
    facts the operator and the refusal rewriting need."""
    var fields: PythonObject
    var key: PythonObject
    var source: PythonObject
    var prepared: PythonObject
    var name: String
    var rows: Int
    var columns: Int
    var fitting: Bool
    var nested: Bool
    var recorded: Bool
    var direct: Bool


def cluster_lifecycle_unpack(model: PythonObject, owner: Int, operation: Int, args: PythonObject,
                             context: PythonObject) raises -> ClusterLifecycleCall:
    """Admission, fitted check, X, the width and feature checks and the
    owners' arguments, in the order the public methods refused them."""
    var method = _method(owner, operation)
    var fitting = cluster_method_fitting(method)
    var key = PythonObject(None)
    var recorded = False
    var direct = False
    var fields = PythonObject(None)
    var nk = cluster_keys()
    if fitting:
        # A fit replaces the fitted state: release the old record now.
        cluster_record_drop(_XC_ADMIT_SLOT, model)
        fields = _admit_model(model, context, direct)
    else:
        key = _query_key(model)
        recorded = cluster_record_get(_XC_ADMIT_SLOT, model, key) is not None
        if recorded:
            fields = _fields(model)
            if py_contains(fields, nk[].k__mojolearn_after_fit):
                raise Error("TypeError: native cluster execution cannot invoke Python fit callbacks")
        else:
            fields = _admit_model(model, context, direct)
    if fitting:
        _admit_replaced_state(model, fields, context)
    var fitted_flag = py_get(fields, nk[].k__mojolearn_fitted, PythonObject(False))
    if py_type(fitted_flag) is not py_type(PythonObject(True)) and py_type(fitted_flag) is not py_type(PythonObject(0)):
        raise Error("TypeError: native fitted flags cannot invoke Python boolean hooks")
    var depth_value = py_get(fields, nk[].k__mojolearn_fitting, PythonObject(0))
    if py_type(depth_value) is not py_type(PythonObject(0)):
        raise Error("TypeError: native fit depth must be an exact integer")
    var nested = cluster_depth_nested(Int(py=depth_value))
    if not fitting:
        classical_require_fitted(model, False, _method_text(method), context)
    var name = classical_estimator_name(model)
    var source = args[1]
    var x = classical_check_x(source, name, context)
    var shape = x.shape
    var rows = Int(py=shape[0])
    var columns = Int(py=shape[1])
    if not nested:
        if fitting:
            # All six classes retain their y argument, including fit_predict.
            # It follows the old owner arguments in the public call envelope.
            _target(args[len(args) - 1], rows, name, context)
        if cluster_width_applies(fitting, method):
            var precomputed = False
            # The loop name differs from the record `key` above so the two never share a binding.
            for setting in ["metric", "affinity"]:  # driver-loop: two supported precomputed-mode settings
                var value = py_get(fields, setting)
                precomputed = precomputed or (py_type(value) is py_type(PythonObject("")) and String(value) == "precomputed")
            if not precomputed:
                var width = py_get(fields, nk[].k_n_features_in_)
                if py_type(width) is py_type(PythonObject(0)):
                    classical_check_width(columns, Int(py=width), name)
                estimator_feature_check(model, fields, source, context)
    var values = Python.list(PythonObject(None), x)
    # The trailing y belongs to lifecycle validation, not the algorithm ABI.
    var stop = cluster_argument_stop(len(args), fitting)
    for i in range(2, stop):  # driver-loop: fixed public argument positions
        classical_admit_value(args[i], context)
        _ = py_call(values.append, args[i])
    # The x_cluster image of the estimator's binding route (the shared
    # estimator door): one record read and pointer compares on a hit. It is the
    # owners' first argument, resolved where `estimator_arguments` resolved the
    # per-call `estimator_binding(model, "_mojolearn_x_cluster", ...)` (a second
    # list, its appends, and the registry walk of every public call).
    py_setitem(values, PythonObject(0), estimator_binding(model, "_mojolearn_x_cluster", context))
    var prepared = py_call(py_type(Python.tuple()), values)
    return ClusterLifecycleCall(fields=fields, key=key, source=source, prepared=prepared, name=name, rows=rows,
                                columns=columns, fitting=fitting, nested=nested, recorded=recorded, direct=direct)
