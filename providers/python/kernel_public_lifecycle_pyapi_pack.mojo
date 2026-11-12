# SPDX-License-Identifier: Apache-2.0
# Kernel public lifecycle formerly owned by _mode's Python decorators. The
# numerical owner retains operation-local input descriptors through completion.
# The PACK half of the former `providers/python/kernel_public_lifecycle_pyapi.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `providers/python/kernel_public_lifecycle_pyapi_unpack.mojo`.
from std.python import Python, PythonObject
from std.math import isfinite
from bindings.admitted_family_door import admit_family_meta
from bindings.native_loaded_module_door import NativeModules, native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_fields
from bindings.mode_metadata_native_door import _fields, _fitted
from bindings.public_attributes_native_door import _plan, _lookup, _string_keys
from bindings.classical_attributes_door import _type_mro, classical_connect_numpy
from bindings.classifier_public_lifecycle_door import _feature_names
from bindings.classical_pandas_door import classical_admit_value
from bindings.classical_exceptions_door import classical_not_fitted_class
from bindings.classical_warnings_unpack import classical_warn_column
from bindings.sequential_admission_door import _ordinary_torch
from bindings.cluster_fit_a_native_door import _pending, _pending_text, _drop_pending
from bindings.optimizer_public_words_unpack import PYTHON_ERROR_PENDING
from bindings.classical_arrays_door import classical_address
from bindings.classical_labels import labels_shape_of, _flatten_labels
from bindings.label_integer_storage_unpack import labels_integer_storage, labels_float_storage
from bindings.label_input_sources_unpack import labels_storage_source, labels_list_integer_storage, labels_text_storage, labels_list_scalar_kind, LABELS_LIST_FLOAT
from core.label_storage_impl import LABELS_FLOAT_OK, LABELS_FLOAT_NONFINITE
from bindings.host_inference_scalars_door import host_inference_scalar
from bindings.metrics_native_support_door import metric_float, metric_is_real, metric_is_integral
from bindings.host_sparse_native_door import host_sparse_format, _sparse_fields
from bindings.native_array_view_pack import native_array_ndarray_view
from bindings.family_common_pack import family_array
from core.admitted_input import admitted_cast_f64
from core.host_finite_scan import host_all_finite_f32
from core.cast_in import CAST_KIND_FLOAT
from providers.python.array_metadata_pyapi_door import ArrayMetadata, ArrayPin, array_python_shape
from providers.python.estimator_lifecycle_pyapi_unpack import estimator_feature_check
from bindings.native_class_admission_door import class_record, class_admit
from bindings.trees_side_records_unpack import trees_key_same
from bindings.gp_svm_records_door import gpsvm_side_get, gpsvm_side_put, gpsvm_side_drop
from bindings.native_model_records import (
    records_key_append, records_stamped_get, records_stamped_put, records_stampable, records_drop,
    ContentRecord, content_value, RECORDS_GP_SVM,
)
from bindings.gp_svm_records_door import gpsvm_key_value
from bindings.py_optional import py_present


from providers.python.kernel_public_lifecycle_pyapi_unpack import (
    _KM_ADMIT_SLOT,
    _KM_STAMP,
    _km_admitted,
    _km_remember,
)


def kernel_public_admit(model: PythonObject, method: String, fitting: Bool,
                        source: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    var fields = _fields(model)
    var attributes = py_get(py_fields(native_loaded_module("mojolearn._native_public_attributes")), "_CONTEXT")
    if py_type(attributes) is not py_type(Python.dict()):
        raise Error("TypeError: kernel methods require their native attribute metadata")
    _string_keys(attributes)
    var modules = NativeModules()
    var plan = _plan(modules, model, attributes)
    var ancestry = _type_mro(py_type(model))
    # Optional, not a PythonObject read and reassigned in a loop (modular/modular#7546 pattern; runtime-fix-save-forest sweep).
    var found_parameters = Optional[PythonObject]()
    for i in range(len(ancestry)):  # driver-loop: passive subclasses inherit the connected schema
        if not found_parameters:
            var candidate = py_get(context["family_km_public_parameters"], ancestry[i])
            if candidate is not None:
                found_parameters = candidate
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var parameters = Optional[PythonObject]()
    if found_parameters:
        parameters = py_present(found_parameters.value())
    if not parameters:
        raise Error("TypeError: kernel execution requires a connected estimator class")
    var learned = Python.tuple("n_features_in_", "n_targets_", "_kernel_params", "_params", "_squeeze",
                               "X_fit_", "dual_coef_", "info_", "components_", "component_indices_",
                               "normalization_", "eigenvalues_", "eigenvectors_", "sweeps_",
                               "random_weights_", "random_offset_", "sigma_", "scale_", "feature_names_in_")
    # Every non-fit call used to look up and admit ~26 parameters and learned
    # fields; a per-model identity record skips them while nothing they read
    # changed (bindings/gp_svm_records.mojo). Fit always admits in full.
    if fitting:
        # A refit replaces the learned fields: release the record now.
        gpsvm_side_drop(_KM_ADMIT_SLOT, model)
        records_drop(RECORDS_GP_SVM, model, _KM_STAMP)
    if fitting or not _km_admitted(modules, model, fields, parameters.value(), learned, plan.plan, attributes, context):
        var direct = not fitting
        for i in range(len(parameters.value())):  # driver-loop: fixed constructor parameter names
            var value = _lookup(model, parameters.value()[i], fields, plan, attributes)
            classical_admit_value(value, context)
            direct = direct and py_contains(fields, parameters.value()[i]) and value is fields[parameters.value()[i]]
        for i in range(len(learned)):  # driver-loop: only fitted storage consumed by kernel owners
            if py_contains(fields, learned[i]):
                var value = _lookup(model, learned[i], fields, plan, attributes)
                classical_admit_value(value, context)
                direct = direct and value is fields[learned[i]]
        if direct:
            # Every value came from plain instance storage: record it.
            _km_remember(modules, model, fields, parameters.value(), learned, plan.plan, attributes, context)
    var flag = py_get(fields, "_mojolearn_fitted", PythonObject(False))
    if py_type(flag) is not py_type(PythonObject(True)) and py_type(flag) is not py_type(PythonObject(0)):
        raise Error("TypeError: kernel fitted flags must be native booleans or integers")
    var depth = py_get(fields, "_mojolearn_fitting", PythonObject(0))
    if py_type(depth) is not py_type(PythonObject(0)):
        raise Error("TypeError: kernel fit depth must be a native integer")
    if not fitting:
        if not Bool(py=_fitted(model)):
            _ = classical_not_fitted_class(context)
            raise Error("NotFittedError: This " + String(py_type(model).__name__) +
                        " instance is not fitted yet. Call 'fit' with appropriate arguments before using " + method + ".")
        var kernel = py_get(fields, "kernel")
        var precomputed = py_type(kernel) is py_type(PythonObject("")) and String(kernel) == "precomputed"
        if Int(py=depth) == 0 and not precomputed:
            estimator_feature_check(model, fields, source, context)
    return fields


def kernel_public_result(result: PythonObject, prediction: Bool, outside: Bool,
                         context: PythonObject) raises -> PythonObject:
    if not outside or py_get(context, "ndarray") is None or py_type(result) is not context["array_class"]:
        return result
    if not prediction:
        return native_array_ndarray_view(result, context)
    # The established regressor public output is float64; internal predictions
    # retain float32. Widen the finished result in native code before publishing.
    # The prediction is the binding's own fresh C-order float32 Array (lane
    # p3-svm-gp-kernel, rows 245/806): widened straight from its store, with
    # no second input admission of a result we just made.
    var shape = result.shape
    var rank = len(shape)
    if String(result.dtype) == "<f4" and String(result.order) == "C" and (rank == 1 or rank == 2):
        var rows = Int(py=shape[0])
        var cols = Int(py=shape[1]) if rank == 2 else 1
        var widened = family_array(shape, "<f8", context)
        admitted_cast_f64(Int(py=classical_address(result)), cols * 4, 4, rows, cols, 4, CAST_KIND_FLOAT,
                             MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(widened))))
        _ = result
        return native_array_ndarray_view(widened, context)
    var pin = ArrayPin()
    var value = admit_family_meta(result, -1, "prediction", context, pin)
    var wide = family_array(array_python_shape(value), "<f8", context)
    var source = value.source
    admitted_cast_f64(source.addr, source.s0, source.s1, source.rows, source.cols,
                         source.itemsize, source.kind,
                         MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(wide))))
    _ = value^
    _ = pin^  # the door's pin held the prediction through the cast
    return native_array_ndarray_view(wide, context)
