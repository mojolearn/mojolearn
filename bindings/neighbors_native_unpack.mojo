# SPDX-License-Identifier: Apache-2.0
# Public neighbor-family dispatch. PolynomialCountSketch UNPACK/PACK passes
# typed options, dimensions and requests to core.pcs_native_api; native PCS
# draw/sketch/FFT kernels and the existing route defaults remain unchanged.
from std.os import getenv
from std.python import PythonObject
from bindings.py_balanced import py_type, py_setitem
from bindings.classical_arrays_door import classical_address
from bindings.classical_validate_unpack import classical_view
from providers.python.array_metadata_pyapi_door import ArrayMetadata, ArrayPin, array_shape_once, array_metadata_once
from core.cast_in import CAST_KIND_SIGNED
from providers.python.pcs_rng_pyapi_pack import pcs_seed
from core.pcs_native_api import PCSFitPlan, pcs_options, pcs_feature_width
from bindings.pcs_native_boundary_unpack import PCSConnection, pcs_seed_unpack, pcs_run_draw, pcs_run_query
from bindings.pcs_native_boundary_pack import pcs_binding, pcs_array
from bindings.public_lifecycle_owner_door import lifecycle_admit_fields
from providers.python.pcs_public_lifecycle_pyapi_unpack import pcs_public_begin
from providers.python.pcs_public_lifecycle_pyapi_pack import pcs_feature_names
from providers.python.neighbors_public_lifecycle_pyapi_door import neighbors_public_end, neighbors_public_finish, neighbors_public_refusal
from bindings.family_common_pack import family_field
from providers.python.radius_public_lifecycle_pyapi_door import radius_fit, radius_refusals
from bindings.neighbors_knn_native_door import neighbors_knn_operation, knn_release
from bindings.neighbors_local_native_door import neighbors_local_operation
from bindings.neighbors_semi_native_door import neighbors_semi_operation, schi2_public_weights, schi2_public_set_weights
from providers.python.neighbors_feature_names_pyapi_door import neighbors_feature_names
from bindings.neighbors_graph_native_door import neighbors_graph_operation
from bindings.neighbors_ann_native_door import neighbors_ann_operation
from bindings.neighbors_kde_native_door import neighbors_kde_operation


def _pcs_int(value: PythonObject, name: String) raises -> Int:
    if py_type(value) is not py_type(PythonObject(0)):
        raise Error("TypeError: PolynomialCountSketch " + name + " must be a native int")
    return Int(py=value)


def _pcs_real(value: PythonObject, name: String) raises -> Float64:
    var cls = py_type(value)
    if cls is not py_type(PythonObject(0)) and cls is not py_type(PythonObject(0.0)):
        raise Error("TypeError: PolynomialCountSketch " + name + " must be a native real number")
    return Float64(py=value)


def _pcs_hash(source: PythonObject, degree: Int, name: String, context: PythonObject,
              mut pin: ArrayPin) raises -> ArrayMetadata:
    var view = classical_view(source, context, name)
    var shape = array_shape_once(view, context)
    if len(shape) != 2 or shape[0] != degree:
        raise Error("ValueError: PolynomialCountSketch hash tables must match degree")
    var result = array_metadata_once(view, shape^, name, context, pin)
    if result.source.kind != CAST_KIND_SIGNED or result.source.itemsize != 4 or not result.c_order:
        raise Error("TypeError: PolynomialCountSketch hash tables must be contiguous native int32 arrays")
    return result^


def _pcs_fit(fields: PythonObject, settings: PythonObject, binding: PythonObject,
             plan: PCSFitPlan, context: PythonObject) raises:
    """Draw directly into public tables; no additional table copy or owner."""
    var seed = pcs_seed(family_field(settings, "random_state"), context)
    if plan.augmented > 0:
        var connection = PCSConnection(binding)
        connection.driver.require_draw()
        var indices = pcs_array(plan.hash_shape(), "<i4", context)
        var signs = pcs_array(plan.hash_shape(), "<i4", context)
        var word = pcs_seed_unpack(seed)
        var request = plan.draw(word, Int(py=classical_address(indices)), Int(py=classical_address(signs)))
        try:
            pcs_run_draw(connection, request)
            lifecycle_admit_fields(fields)
            py_setitem(fields, "indexHash_", indices)
            py_setitem(fields, "bitHash_", signs)
        finally:
            _ = indices
            _ = signs
    else:
        lifecycle_admit_fields(fields)
        py_setitem(fields, "indexHash_", pcs_array(plan.hash_shape(), "<i4", context))
        py_setitem(fields, "bitHash_", pcs_array(plan.hash_shape(), "<i4", context))
    py_setitem(fields, "n_features_in_", PythonObject(plan.features))
