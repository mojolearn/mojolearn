# SPDX-License-Identifier: Apache-2.0
# Public neighbor-family dispatch. PolynomialCountSketch UNPACK/PACK passes
# typed options, dimensions and requests to core.pcs_native_api; native PCS
# draw/sketch/FFT kernels and the existing route defaults remain unchanged.
# The PACK half of the former `bindings/neighbors_native.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/neighbors_native_unpack.mojo`.
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


from bindings.neighbors_native_unpack import (
    _pcs_fit,
    _pcs_hash,
    _pcs_int,
    _pcs_real,
)


def _pcs_transform(settings: PythonObject, binding: PythonObject, x: ArrayMetadata,
                   plan: PCSFitPlan, gamma: Float64, coef0: Float64,
                   context: PythonObject) raises -> PythonObject:
    """Capture mutable public hashes anew; RUN sees only pinned addresses."""
    pcs_feature_width(x.source.cols, _pcs_int(family_field(settings, "n_features_in_"), "n_features_in_"))
    var index_pin = ArrayPin()
    var bit_pin = ArrayPin()
    var index_hash = _pcs_hash(family_field(settings, "indexHash_"), plan.options.degree, "indexHash_", context, index_pin)
    var bit_hash = _pcs_hash(family_field(settings, "bitHash_"), plan.options.degree, "bitHash_", context, bit_pin)
    plan.admit_hash_widths(index_hash.source.cols, bit_hash.source.cols)
    var dst = pcs_array(plan.output_shape(x.source.rows), "<f4", context)
    var old_items = getenv("MOJOLEARN_XN_OLD_ITEMS") == "1"
    var request = plan.query(x.source.addr, index_hash.source.addr, bit_hash.source.addr,
        Int(py=classical_address(dst)), x.source.rows, gamma, coef0, old_items)
    try:
        pcs_run_query(binding, request)
    finally:
        _ = dst
        _ = index_hash
        _ = bit_hash
        _ = index_pin^  # the door's pins hold both hash tables through RUN
        _ = bit_pin^
        _ = x
    return dst


def _pcs_operation(model: PythonObject, operation: String, args: PythonObject,
                   context: PythonObject) raises -> PythonObject:
    var public = pcs_public_begin(model, operation, args, context)
    var result = PythonObject(None)
    try:
        var binding = pcs_binding(model, args, context)
        var settings = public.settings
        var fields = public.call.fields
        var degree = _pcs_int(settings["degree"], "degree")
        var components = _pcs_int(settings["n_components"], "n_components")
        var options = pcs_options(degree, components)
        var coef0 = _pcs_real(settings["coef0"], "coef0")
        var plan = PCSFitPlan(options, public.x.source.cols, coef0)
        if public.call.fitting:
            _pcs_fit(fields, settings, binding, plan, context)
            lifecycle_admit_fields(fields)
            py_setitem(settings, "n_features_in_", fields["n_features_in_"])
            py_setitem(settings, "indexHash_", fields["indexHash_"])
            py_setitem(settings, "bitHash_", fields["bitHash_"])
        if operation == "family_xn_pcs_fit":
            result = model
        else:
            var gamma = _pcs_real(settings["gamma"], "gamma")
            result = _pcs_transform(settings, binding, public.x, plan, gamma, coef0, context)
    except error:
        var refused = neighbors_public_refusal(error, public.call, model)
        raise refused^
    finally:
        neighbors_public_end(model, public.call)
    return neighbors_public_finish(model, operation, result, public.call, context)


def neighbors_operation(model: PythonObject, operation: String, args: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    if operation == "family_xn_knn_release":
        knn_release(model)
        return PythonObject(None)
    if operation == "family_xn_radius_refusals":
        radius_refusals(model, context)
        return PythonObject(None)
    if operation == "family_xn_radius_fit":
        return radius_fit(model, args, context)
    if operation == "family_xn_pcs_feature_names":
        return pcs_feature_names(model, context)
    if operation.startswith("family_xn_names_"):
        return neighbors_feature_names(model, operation, args, context)
    if operation == "family_xn_schi2_weights":
        return schi2_public_weights(model, context)
    if operation == "family_xn_schi2_set_weights":
        schi2_public_set_weights(model, args[0])
        return PythonObject(None)
    if (operation == "family_xn_pcs_fit" or operation == "family_xn_pcs_transform"
            or operation == "family_xn_pcs_fit_transform"):
        return _pcs_operation(model, operation, args, context)
    if operation.startswith("family_xn_knn_"):
        return neighbors_knn_operation(model, operation, args, context)
    if operation.startswith("family_xn_loc_"):
        return neighbors_local_operation(model, operation, args, context)
    if operation.startswith("family_xn_semi_"):
        return neighbors_semi_operation(model, operation, args, context)
    if operation.startswith("family_xn_graph_"):
        return neighbors_graph_operation(model, operation, args, context)
    if operation.startswith("family_xn_ann_"):
        return neighbors_ann_operation(model, operation, args, context)
    if operation.startswith("family_xn_kde_"):
        return neighbors_kde_operation(model, operation, args, context)
    raise Error("RuntimeError: unknown neighbors operation " + operation)
