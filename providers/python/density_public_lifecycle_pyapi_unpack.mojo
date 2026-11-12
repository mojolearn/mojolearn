# SPDX-License-Identifier: Apache-2.0
# DBSCAN and KernelDensity public lifecycle, including retained state ownership.
# Python submits the original arguments once; this owner selects native entries,
# admits metadata, tracks fitting and publishes the completed public result.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from bindings.native_loaded_module_door import native_loaded_module
from bindings.family_raw_attr_pack import family_raw_get, family_raw_get_or
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.cluster_side_records_unpack import cluster_is_subclass, cluster_refuse_fit_callbacks
from bindings.classical_pandas_door import classical_admit_value
from bindings.classical_validate_unpack import classical_check_x, classical_estimator_name
from bindings.classical_validate_run import classical_check_width
from bindings.classical_exceptions_door import classical_require_fitted
from bindings.host_inference_scalars_door import host_inference_scalar
from bindings.native_owner_lifetime_door import native_release_admit
from bindings.native_array_view_pack import native_array_ndarray_view
from bindings.cfb_dbscan_door import cfb_dbscan_operation
from bindings.dbscan_native_boundary_door import dbscan_holder_admit, dbscan_holder_empty, dbscan_release
from bindings.neighbors_kde_native_door import neighbors_kde_operation, _kde_release
from bindings.kde_native_boundary_unpack import kde_owner_admit
from bindings.kde_admission_boundary_unpack import kde_admission_hit, kde_admission_remember, kde_admission_drop
from bindings.public_lifecycle_owner_door import lifecycle_admit_fields
from providers.python.host_estimator_binding_pyapi_door import host_estimator_binding_current as estimator_binding
from providers.python.estimator_lifecycle_pyapi_unpack import estimator_outside, estimator_feature_check, estimator_fit_begin, estimator_fit_end, estimator_fit_complete
from providers.python.cluster_public_lifecycle_pyapi_door import _target, _refusal


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
