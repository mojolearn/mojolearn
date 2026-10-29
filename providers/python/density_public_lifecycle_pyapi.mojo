# SPDX-License-Identifier: Apache-2.0
# DBSCAN and KernelDensity public lifecycle, including retained state ownership.
# Python submits the original arguments once; this owner selects native entries,
# admits metadata, tracks fitting and publishes the completed public result.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from bindings.native_loaded_module import native_loaded_module
from bindings.family_raw_attr import family_raw_get, family_raw_get_or
from bindings.classical_attributes import classical_connect_numpy, _type_mro
from bindings.mode_metadata_native import _class_fields
from bindings.classical_pandas import classical_admit_value
from bindings.classical_validate import classical_check_x, classical_check_width, classical_estimator_name
from bindings.classical_exceptions import classical_require_fitted
from bindings.host_inference_scalars import host_inference_scalar
from bindings.native_owner_lifetime import native_release_admit
from bindings.native_array_view import native_array_ndarray_view
from bindings.cfb_dbscan import cfb_dbscan_operation
from bindings.cfb_resident import cfb_release
from bindings.neighbors_kde_native import neighbors_kde_operation, _kde_release
from providers.python.host_estimator_binding_pyapi import host_estimator_binding_current as estimator_binding
from providers.python.estimator_lifecycle_pyapi import estimator_outside, estimator_feature_check, estimator_fit_begin, estimator_fit_end, estimator_fit_complete
from providers.python.cluster_public_lifecycle_pyapi import _target, _refusal
from providers.python.neural_descriptor_pyapi import neural_descriptor_entry


def _keys(value: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(value) is not b.dict:
        raise Error("TypeError: density state requires an exact dictionary")
    var keys = py_call(b.list, value)
    for i in range(len(keys)):  # driver-loop: reject hashing/equality hooks before field reads
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: density state requires exact string keys")
    return keys


def _keep_binding(binding: PythonObject, context: PythonObject, required: Bool = False) raises:
    var b = native_loaded_module("builtins")
    if py_type(binding) is not py_type(b):
        raise Error("TypeError: density residency requires its retained native provider")
    var retained = context["density_retained_bindings"]
    if py_type(retained) is not b.list:
        raise Error("TypeError: density provider lifetime requires its native owner list")
    for i in range(len(retained)):  # driver-loop: identity-only provider owners retained across cache replacement
        if retained[i] is binding:
            return
    if required:
        var connected = False
        var records = context["family_binding_connections"]["_mojolearn_estimators"]
        for mode in ["identical", "fast"]:  # driver-loop: import-retained native modules can own preexisting caches
            connected = connected or binding is records[mode][0]
        if not connected:
            raise Error("TypeError: density resident state has an unconnected provider owner")
    _ = py_call(retained.append, binding)


def _resident(value: PythonObject, dbscan: Bool, context: PythonObject) raises:
    if value is None:
        return
    var b = native_loaded_module("builtins")
    var cached = value
    if dbscan:
        var keys = _keys(value)
        for i in range(len(keys)):  # driver-loop: arbitrary extra holder owners cannot run finalizers on replacement
            if String(keys[i]) != "index":
                native_release_admit(value[keys[i]], context)
        cached = py_get(value, "index")
        if cached is None:
            return
    if py_type(cached) is not b.tuple or len(cached) != 3:
        raise Error("TypeError: density residency requires its native key, handle and provider tuple")
    if py_type(cached[1]) is not b.int:
        raise Error("TypeError: density resident handles require exact integers")
    _keep_binding(cached[2], context, True)
    var key = cached[0]
    if py_type(key) is not b.tuple or len(key) != 6:
        raise Error("TypeError: density resident keys require their native metadata tuple")
    native_release_admit(key, context)
    if dbscan:
        for i in range(3):  # driver-loop: retained reference buffers
            classical_admit_value(key[i], context)
        for i in range(3, 6):  # driver-loop: row, feature and metric scalars
            host_inference_scalar(key[i], context)
    else:
        for i in [0, 2, 3, 4, 5]:  # driver-loop: pointer, weight pointer, kernel, metric and bandwidth
            host_inference_scalar(key[i], context)
        if py_type(key[1]) is not b.tuple:
            raise Error("TypeError: density resident shapes require an exact tuple")
        for i in range(len(key[1])):  # driver-loop: shape dimensions only
            if py_type(key[1][i]) is not b.int:
                raise Error("TypeError: density resident shape dimensions require exact integers")
    _ = neural_descriptor_entry(cached[2], "labeled_reference_release" if dbscan else "kde_fit_release", False)


def _initialize(model: PythonObject, args: PythonObject, names: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    if len(args) != len(names):
        raise Error("TypeError: density constructor argument count differs from its native schema")
    var values = Python.list()
    for i in range(len(names)):  # driver-loop: fixed constructor parameter schema
        var value = args[i]
        if String(names[i]) == "numeric_mode" and value is None:
            value = family_raw_get_or(model, "numeric_mode", PythonObject(None))
        if py_contains(fields, names[i]) and fields[names[i]] is not value:
            native_release_admit(fields[names[i]], context)
        _ = py_call(values.append, value)
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
        cfb_release(model)
    else:
        _kde_release(fields)
    for i in range(len(keys)):  # driver-loop: native pickle restoration after complete replacement admission
        if String(keys[i]) != resident_name:
            py_setitem(fields, keys[i], state[keys[i]])
    py_setitem(fields, resident_name, Python.dict() if dbscan else PythonObject(None))
    return PythonObject(None)


def _admit(model: PythonObject, names: PythonObject, dbscan: Bool, fitting: Bool, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
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
    var ancestry = _type_mro(py_type(model))
    for i in range(len(ancestry)):  # driver-loop: no fit-completion callback may enter Python
        if py_contains(_class_fields(ancestry[i]), "_mojolearn_after_fit"):
            raise Error("TypeError: density execution cannot invoke Python fit callbacks")
    if py_contains(fields, "_mojolearn_after_fit"):
        raise Error("TypeError: density execution cannot invoke Python fit callbacks")
    _resident(py_get(fields, "_resident_holder" if dbscan else "_resident"), dbscan, context)
    return fields


def density_public_operation(model: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    var b = native_loaded_module("builtins")
    if py_type(args) is not b.tuple:
        raise Error("TypeError: density public arguments require an exact tuple")
    var dbscan = operation.startswith("family_density_dbscan_")
    var owner = context["density_dbscan_class"] if dbscan else context["density_kde_class"]
    var ancestry = _type_mro(py_type(model))
    var known = False
    for i in range(len(ancestry)):  # driver-loop: connected owner identity, preserving passive subclasses
        known = known or ancestry[i] is owner
    if not known:
        raise Error("TypeError: density execution requires its connected estimator class")
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
    var fields = _admit(model, names, dbscan, fitting, context)
    var binding = estimator_binding(model, "_mojolearn_estimators", context)
    _keep_binding(binding, context)
    if not dbscan:
        # KDE already owns fitted tracking, y admission and public outputs in
        # the shared neighbor lifecycle. Preserve that compiled owner once.
        var prepared = Python.tuple(binding, args[0], args[2], args[1]) if fitting else Python.tuple(binding, args[0])
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
    var prepared = Python.tuple(binding, x, args[2]) if fitting else Python.tuple(binding, x)
    if fitting:
        classical_admit_value(args[2], context)
    var depth = estimator_fit_begin(fields) if fitting else 0
    var result = PythonObject(None)
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
