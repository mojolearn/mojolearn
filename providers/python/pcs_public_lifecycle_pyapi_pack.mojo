# SPDX-License-Identifier: Apache-2.0
# PolynomialCountSketch public fit/transform envelope. One X snapshot feeds
# finite/width checks and both fit_transform stages; target owners survive it.
# The PACK half of the former `providers/python/pcs_public_lifecycle_pyapi.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `providers/python/pcs_public_lifecycle_pyapi_unpack.mojo`.
from bindings.admitted_finite_wording import admitted_finite_wording
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.native_loaded_module_door import NativeModules, native_loaded_module, _loaded_fields
from bindings.mode_metadata_native_door import _fields, _class_fields, _fitted
from bindings.classical_attributes_door import classical_connect_numpy, _type_mro
from bindings.classical_exceptions_door import classical_not_fitted_class
from bindings.native_class_admission_door import class_record, class_admit
from bindings.classical_validate_unpack import classical_estimator_name
from bindings.classical_validate_run import classical_check_width
from bindings.public_attributes_native_door import _plan, _lookup
from bindings.host_inference_scalars_door import host_inference_scalar, host_inference_primitive_number
from bindings.admitted_family_door import admit_family_input, admit_family_cast
from core.host_finite_scan import host_all_finite_f32_strided
from bindings.native_numpy_metadata_unpack import native_numpy_dtype
from core.native_scalar_decode import native_scalar_real
from core.cast_in import CAST_KIND_FLOAT
from providers.python.array_metadata_pyapi_door import ArrayMetadata, ArrayPin
from providers.python.radius_public_lifecycle_pyapi_door import radius_target
from providers.python.neighbors_public_lifecycle_pyapi_door import NeighborPublicCall
from providers.python.estimator_lifecycle_pyapi_unpack import estimator_outside, estimator_fit_begin, estimator_feature_check


def pcs_feature_names(model: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    var b = native_loaded_module("builtins")
    var fields = _fields(model)
    var attributes = py_get(_loaded_fields(native_loaded_module("mojolearn._native_public_attributes")), "_CONTEXT")
    var modules = NativeModules()
    var plan = _plan(modules, model, attributes)
    var count = _lookup(model, PythonObject("n_components"), fields, plan, attributes)
    host_inference_scalar(count, context)
    var length = Int(py=py_call(b.int, host_inference_primitive_number(count, context, True)))
    var cls_name = String(py_call(b.type.__dict__["__name__"].__get__, py_type(model))).lower()
    var result = Python.list()
    for i in range(max(length, 0)):  # driver-loop: finished public feature-name metadata
        _ = py_call(result.append, PythonObject(cls_name + String(i)))
    return result
