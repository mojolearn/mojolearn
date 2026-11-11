# SPDX-License-Identifier: Apache-2.0
# DBSCAN and KernelDensity public lifecycle, including retained state ownership.
# Python submits the original arguments once; this owner selects native entries,
# admits metadata, tracks fitting and publishes the completed public result.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from bindings.native_loaded_module import native_loaded_module
from bindings.family_raw_attr import family_raw_get, family_raw_get_or
from bindings.classical_attributes import classical_connect_numpy
from bindings.cluster_side_records import cluster_is_subclass, cluster_refuse_fit_callbacks
from bindings.classical_pandas import classical_admit_value
from bindings.classical_validate import classical_check_x, classical_check_width, classical_estimator_name
from bindings.classical_exceptions import classical_require_fitted
from bindings.host_inference_scalars import host_inference_scalar
from bindings.native_owner_lifetime import native_release_admit
from bindings.native_array_view import native_array_ndarray_view
from bindings.cfb_dbscan import cfb_dbscan_operation
from bindings.dbscan_native_boundary import dbscan_holder_admit, dbscan_holder_empty, dbscan_release
from bindings.neighbors_kde_native import neighbors_kde_operation, _kde_release
from bindings.kde_native_boundary import kde_owner_admit
from bindings.kde_admission_boundary import kde_admission_hit, kde_admission_remember, kde_admission_drop
from bindings.public_lifecycle_owner import lifecycle_admit_fields
from providers.python.host_estimator_binding_pyapi import host_estimator_binding_current as estimator_binding
from providers.python.estimator_lifecycle_pyapi import estimator_outside, estimator_feature_check, estimator_fit_begin, estimator_fit_end, estimator_fit_complete
from providers.python.cluster_public_lifecycle_pyapi import _target, _refusal


def _keys(value: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(value) is not b.dict:
        raise Error("TypeError: density state requires an exact dictionary")
    var keys = py_call(b.list, value)
    for i in range(len(keys)):  # driver-loop: reject hashing/equality hooks before field reads
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: density state requires exact string keys")
    return keys


def _resident(value: PythonObject, dbscan: Bool, context: PythonObject) raises:
    if dbscan:
        dbscan_holder_admit(value)
    else:
        kde_owner_admit(value)


def _initialize(model: PythonObject, args: PythonObject, names: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    if len(args) != len(names):
        raise Error("TypeError: density constructor argument count differs from its native schema")
    var values = List[PythonObject]()
    for i in range(len(names)):  # driver-loop: fixed constructor parameter schema
        var value = args[i]
        if String(names[i]) == "numeric_mode" and value is None:
            value = family_raw_get_or(model, "numeric_mode", PythonObject(None))
        if py_contains(fields, names[i]) and fields[names[i]] is not value:
            native_release_admit(fields[names[i]], context)
        values.append(value)
    for i in range(len(names)):  # driver-loop: publish only after every replaced owner passed admission
        py_setitem(fields, names[i], values[i])
    return PythonObject(None)


def _state(model: PythonObject, args: PythonObject, dbscan: Bool, restore: Bool, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var resident_name = String("_resident_holder") if dbscan else String("_resident")
    if not restore:
        var state = py_call(native_loaded_module("builtins").dict, fields)
        _ = py_call(state.pop, resident_name, PythonObject(None))
        return state
    var state = args[0]
    var keys = _keys(state)
    # All old owners are retained until admission completes; incoming values
    # remain owned by the caller's exact state dictionary throughout publication.
    for i in range(len(keys)):  # driver-loop: restoring ordinary stored values never invokes input protocols
        if String(keys[i]) != resident_name and py_contains(fields, keys[i]) and fields[keys[i]] is not state[keys[i]]:
            native_release_admit(fields[keys[i]], context)
    _resident(py_get(fields, resident_name), dbscan, context)
    if dbscan:
        dbscan_release(model)
    else:
        _kde_release(fields)
    lifecycle_admit_fields(fields)
    lifecycle_admit_fields(state)
    for i in range(len(keys)):  # driver-loop: native pickle restoration after complete replacement admission
        if String(keys[i]) != resident_name:
            # Quiet native handle teardown may release the GIL. Admit any
            # intervening replacement before publication drops its owner.
            if py_contains(fields, keys[i]) and fields[keys[i]] is not state[keys[i]]:
                native_release_admit(fields[keys[i]], context)
            py_setitem(fields, keys[i], state[keys[i]])
    _resident(py_get(fields, resident_name), dbscan, context)
    py_setitem(fields, resident_name, dbscan_holder_empty() if dbscan else PythonObject(None))
    return PythonObject(None)


# Both families use the same weak native metadata/content registry. The
# historical KDE helper/type name is retained to keep one canonical owner.
comptime _DBSCAN_ADMIT_SLOT = "dbscan_density_admitted_v2"
comptime _KDE_ADMIT_SLOT = "kde_density_admitted_v2"


def _admit(model: PythonObject, names: PythonObject, dbscan: Bool, fitting: Bool, context: PythonObject,
           slot: String) raises -> Bool:
    """The public admission. Perf (lane perf2-cluster): a query whose `key`
    matches this model's admission record skips the per-value admissions of
    every setting, fitted field and the residency holder; the flag checks and
    the callback refusal (its ancestry walk once per class state) run on
    every call. Returns whether the record matched."""
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    var recorded = False
    if not fitting:
        recorded = kde_admission_hit(model, names, True, slot, context)
    if not recorded:
        for i in range(len(names)):  # driver-loop: consumed constructor metadata
            var name = String(names[i])
            var value = family_raw_get_or(model, name, PythonObject(None)) if name == "numeric_mode" else family_raw_get(model, name)
            if name == "metric_params":
                if value is not None:
                    if py_type(value) is not b.dict:
                        raise Error("TypeError: KernelDensity metric_params requires an exact dictionary")
                    var pairs = py_call(b.list, value.items())
                    for k in range(len(pairs)):  # driver-loop: KDE reads values only; native keys remain safe in diagnostic repr
                        host_inference_scalar(pairs[k][0], context)
                        host_inference_scalar(pairs[k][1], context)
            else:
                host_inference_scalar(value, context, True)
        var scalars = Python.tuple("input_copied_", "n_features_in_", "n_samples_fit_", "n_iter_", "_mojolearn_fitted", "_mojolearn_fitting")
        var arrays = Python.tuple("labels_", "components_", "core_sample_indices_", "_core_labels", "_x", "_w")
        for i in range(len(scalars)):  # driver-loop: learned numeric metadata consumed by the owner
            if py_contains(fields, scalars[i]):
                host_inference_scalar(fields[scalars[i]], context)
                if fitting:
                    native_release_admit(fields[scalars[i]], context)
        for i in range(len(arrays)):  # driver-loop: retained fitted buffer owners
            if py_contains(fields, arrays[i]):
                classical_admit_value(fields[arrays[i]], context)
                if fitting:
                    native_release_admit(fields[arrays[i]], context)
    if fitting and py_contains(fields, "feature_names_in_"):
        native_release_admit(fields["feature_names_in_"], context)
    var flag = py_get(fields, "_mojolearn_fitted", PythonObject(False))
    if py_type(flag) is not b.bool and py_type(flag) is not b.int:
        raise Error("TypeError: native fitted flags require exact booleans or integers")
    if py_type(py_get(fields, "_mojolearn_fitting", PythonObject(0))) is not b.int:
        raise Error("TypeError: native fitting depth requires an exact integer")
    cluster_refuse_fit_callbacks(py_type(model), "TypeError: density execution cannot invoke Python fit callbacks")
    if py_contains(fields, "_mojolearn_after_fit"):
        raise Error("TypeError: density execution cannot invoke Python fit callbacks")
    if not recorded:
        _resident(py_get(fields, "_resident_holder" if dbscan else "_resident"), dbscan, context)
    return recorded


def density_public_operation(model: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    var b = native_loaded_module("builtins")
    if py_type(args) is not b.tuple:
        raise Error("TypeError: density public arguments require an exact tuple")
    var dbscan = operation.startswith("family_density_dbscan_")
    var owner = context["density_dbscan_class"] if dbscan else context["density_kde_class"]
    # Passive subclasses are accepted: CPython's MRO test
    # (bindings/cluster_side_records.mojo), not a Python-level walk.
    if not cluster_is_subclass(py_type(model), owner):
        raise Error("TypeError: density execution requires its connected estimator class")
    lifecycle_admit_fields(py_fields(model))
    var prefix = String("family_density_dbscan_") if dbscan else String("family_density_kde_")
    var method = String(operation[byte=prefix.byte_length():])
    var names = context["density_dbscan_parameters"] if dbscan else context["density_kde_parameters"]
    if method == "initialize":
        return _initialize(model, args, names, context)
    if method == "getstate" or method == "setstate":
        return _state(model, args, dbscan, method == "setstate", context)
    if method == "sample":
        raise Error("NotImplementedError: mojolearn KernelDensity: sample() is not implemented (cuML has none)")
    if method == "release":
        var fields = py_fields(model)
        _resident(py_get(fields, "_resident"), False, context)
        _kde_release(fields)
        return PythonObject(None)
    var fitting = method == "fit" or method == "fit_predict"
    var query = not fitting and (method == "predict" or method == "score_samples" or method == "score")
    var slot = String(_DBSCAN_ADMIT_SLOT) if dbscan else String(_KDE_ADMIT_SLOT)
    var recorded = _admit(model, names, dbscan, fitting, context, slot)
    var fields = py_fields(model)
    if fitting:
        # A fit replaces the fitted state: release the old record now.
        kde_admission_drop(model, slot)
    var binding = estimator_binding(model, "_mojolearn_estimators", context)
    if not dbscan:
        # KDE already owns fitted tracking, y admission and public outputs in
        # the shared neighbor lifecycle. Preserve that compiled owner once.
        var prepared = Python.tuple(binding, args[0], args[2], args[1]) if fitting else Python.tuple(binding, args[0])
        if query and not recorded:
            # Stamp admitted state before RUN. Concurrent replacement while
            # RUN releases the GIL must never become a freshly admitted record.
            kde_admission_remember(model, names, True, _KDE_ADMIT_SLOT, context)
        return neighbors_kde_operation(model, "family_xn_kde_" + method, prepared, context)
    var outside = estimator_outside()
    var depth_value = Int(py=py_get(fields, "_mojolearn_fitting", PythonObject(0)))
    if not fitting:
        classical_require_fitted(model, False, method, context)
    var source = args[0]
    var name = classical_estimator_name(model)
    var x = classical_check_x(source, name, context)
    var rows = Int(py=x.shape[0])
    var columns = Int(py=x.shape[1])
    if depth_value == 0:
        if fitting:
            _target(args[1], rows, name, context)
        else:
            var width = py_get(fields, "n_features_in_")
            if py_type(width) is b.int:
                classical_check_width(columns, Int(py=width), name)
            estimator_feature_check(model, fields, source, context)
    # The fourth element tells bindings/cfb_dbscan.mojo that `x` is this
    # door's admitted X (classical_check_x above), so the fit does not
    # snapshot and finite-scan it a second time.
    var prepared = Python.tuple(binding, x, args[2], PythonObject(True)) if fitting else Python.tuple(binding, x)
    if fitting:
        classical_admit_value(args[2], context)
    var depth = estimator_fit_begin(fields) if fitting else 0
    var result = PythonObject(None)
    if query and not recorded:
        kde_admission_remember(model, names, True, slot, context)
    try:
        result = cfb_dbscan_operation(model, "family_cfb_dbscan_" + method, prepared, context)
    except error:
        var refused = _refusal(error, fitting, depth_value > 0, rows, columns, name)
        raise refused^
    finally:
        if fitting:
            estimator_fit_end(model, depth)
    if fitting:
        estimator_fit_complete(model, source, False, context)
    if outside and py_type(result) is context["array_class"] and py_get(context, "ndarray") is not None:
        return native_array_ndarray_view(result, context)
    return result
