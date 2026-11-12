# SPDX-License-Identifier: Apache-2.0
# AgglomerativeClustering's public lifecycle, setting admission and captured X.
# Public values remain pins until their established conversion points. Typed
# native options and requests carry the fit/query execution between providers.
# The PACK half of the former `providers/python/hierarchy_public_lifecycle_pyapi.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `providers/python/hierarchy_public_lifecycle_pyapi_unpack.mojo`.
from bindings.admitted_finite_wording import admitted_finite_wording
from bindings.admitted_family_door import admit_family_meta
from bindings.native_class_admission_door import class_record, class_admit
from bindings.py_cpython_table import cpython_table
from core.p3nc_scan import P3NC_REAL_OK, P3NC_REAL_NONFINITE, P3NC_REAL_CONTINUOUS
from core.host_finite_scan import host_all_finite_f32
from bindings.p3nc_label_lists_unpack import p3nc_float_list_verdict
from std.math import isfinite, floor
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import NativeModules, native_loaded_module, _loaded_fields
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.mode_metadata_native_door import _fields, _class_fields, _fitted
from bindings.public_attributes_native_door import _plan, _lookup
from bindings.classical_attributes_door import classical_connect_numpy, _type_mro
from bindings.classical_pandas_door import classical_admit_value
from bindings.classical_validate_unpack import classical_estimator_name
from bindings.classical_validate_run import classical_check_width
from bindings.classical_exceptions_door import classical_not_fitted_class
from bindings.classical_labels_door import labels_shape_of, _flatten_labels
from bindings.label_integer_storage_unpack import labels_integer_storage, labels_float_storage
from bindings.label_input_sources_unpack import labels_storage_source, labels_list_integer_storage, labels_text_storage
from core.label_storage_impl import LABELS_FLOAT_OK, LABELS_FLOAT_NONFINITE, LABELS_FLOAT_CONTINUOUS
from bindings.classical_warnings_unpack import classical_warn_column
from bindings.host_inference_scalars_door import host_inference_scalar
from bindings.native_array_view_pack import native_array_ndarray_view
from bindings.family_common_unpack import FAMILY_F32P
from bindings.metrics_native_support_door import metric_float, metric_is_real, metric_is_integral
from bindings.host_sparse_native_door import host_sparse_format, _sparse_fields
from bindings.cluster_fit_a_native_door import _pending, _pending_text, _drop_pending
from bindings.optimizer_public_words_unpack import PYTHON_ERROR_PENDING
from providers.python.array_metadata_pyapi_door import ArrayMetadata, ArrayPin
from providers.python.hierarchy_settings_pyapi_unpack import AggloSettings
from bindings.public_lifecycle_owner_door import lifecycle_admit_fields
from bindings.py_optional import py_present
from providers.python.estimator_lifecycle_pyapi_unpack import (
    estimator_outside, estimator_feature_check, estimator_fit_begin, estimator_fit_end, estimator_fit_complete,
)


from providers.python.hierarchy_public_lifecycle_pyapi_unpack import (
    HierarchyPublicCall,
)


def hierarchy_public_finish(model: PythonObject, result: PythonObject,
                            call: HierarchyPublicCall, context: PythonObject) raises -> PythonObject:
    if call.fitting:
        estimator_fit_complete(model, call.source, False, context)
    if call.outside and py_type(result) is context["array_class"] and py_get(context, "ndarray") is not None:
        return native_array_ndarray_view(result, context)
    return result
