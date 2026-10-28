# SPDX-License-Identifier: Apache-2.0
# PolynomialCountSketch `fit`, `transform` and `fit_transform`, one native
# call each (python/mojolearn/_expansion_neighbors.py), and RadiusNeighbors
# `fit` (python/mojolearn/neighbors.py: its refusals, shared with `load`, then
# the stored index). The PCS owner selects the import-connected binding after
# native admission; this owner reads the parameters with the
# Python methods' refusals, draws the hash tables with the binding's
# counter-based generator (`x_neighbors_kfeat_pcs_draw_idn`) and runs the
# sketch (`xn_pcs_resident`, or `xn_pcs` under the MOJOLEARN_XN_OLD_ITEMS A/B
# arm) with the argument lists x_neighbors documents. The same kernels on the
# same inputs: the bits do not move.
from std.os import getenv
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_fields, py_setitem
from bindings.classical_arrays import classical_address
from bindings.classical_validate import classical_view
from providers.python.array_metadata_pyapi import ArrayMetadata, array_shape_once, array_metadata_once
from providers.python.estimator_binding_pyapi import estimator_arguments
from providers.python.x_neighbors_descriptor_pyapi import xn_admit_arguments, xn_execute, xn_pcs_draw_entry, xn_pcs_draw
from core.cast_in import CAST_KIND_SIGNED
from providers.python.pcs_rng_pyapi import pcs_seed
from providers.python.pcs_public_lifecycle_pyapi import pcs_public_begin, pcs_feature_names
from providers.python.neighbors_public_lifecycle_pyapi import neighbors_public_end, neighbors_public_finish, neighbors_public_refusal
from bindings.classical_attributes import classical_connect_numpy
from bindings.family_common import (
    family_repr, family_call, family_call_kw, family_array,
    family_field, family_f32_snapshot, family_owned_snapshot,
)
from providers.python.radius_public_lifecycle_pyapi import radius_fit, radius_refusals
from bindings.neighbors_knn_native import neighbors_knn_operation, knn_release
from bindings.neighbors_local_native import neighbors_local_operation
from bindings.neighbors_semi_native import neighbors_semi_operation, schi2_public_weights, schi2_public_set_weights
from providers.python.neighbors_feature_names_pyapi import neighbors_feature_names
from bindings.neighbors_graph_native import neighbors_graph_operation
from bindings.neighbors_ann_native import neighbors_ann_operation
from bindings.neighbors_kde_native import neighbors_kde_operation


def _pcs_int(value: PythonObject, name: String) raises -> Int:
    if py_type(value) is not py_type(PythonObject(0)):
        raise Error("TypeError: PolynomialCountSketch " + name + " must be a native int")
    return Int(py=value)


def _pcs_real(value: PythonObject, name: String) raises -> Float64:
    var cls = py_type(value)
    if cls is not py_type(PythonObject(0)) and cls is not py_type(PythonObject(0.0)):
        raise Error("TypeError: PolynomialCountSketch " + name + " must be a native real number")
    return Float64(py=value)


def _pcs_hash(source: PythonObject, degree: Int, name: String, context: PythonObject) raises -> ArrayMetadata:
    var view = classical_view(source, context, name)
    var shape = array_shape_once(view, context)
    if len(shape) != 2 or shape[0] != degree:
        raise Error("ValueError: PolynomialCountSketch hash tables must match degree")
    var result = array_metadata_once(view, shape^, name, context)
    if result.source.kind != CAST_KIND_SIGNED or result.source.itemsize != 4 or not result.c_order:
        raise Error("TypeError: PolynomialCountSketch hash tables must be contiguous native int32 arrays")
    return result^


def _pcs_fit(fields: PythonObject, settings: PythonObject, binding: PythonObject, x: ArrayMetadata,
             degree: Int, components: Int, coef0: Float64, context: PythonObject) raises:
    """Fit hash tables from the operation's single admitted input snapshot."""
    var d = x.source.cols
    var nf = d + (1 if coef0 != 0.0 else 0)
    var seed = pcs_seed(family_field(settings, "random_state"), context)
    if nf > 0:
        var draw = xn_pcs_draw_entry(binding)
        var ih = family_array(Python.tuple(degree, nf), "<i4", context)
        var bh = family_array(Python.tuple(degree, nf), "<i4", context)
        if py_type(seed) is not py_type(PythonObject(0)):
            raise Error("TypeError: PolynomialCountSketch random_state must resolve to a native int")
        ref cpy = Python().cpython()
        # Preserve arbitrary-sized integer seeds without Python operator.and_.
        var raw_seed = cpy.lib.get_function[UInt64]("PyLong_AsUnsignedLongLongMask")(seed._obj_ptr)
        var word = Int(raw_seed & 0xFFFFFFFF)
        xn_pcs_draw(binding, draw, word, components, degree * nf,
                    Int(py=classical_address(ih)), Int(py=classical_address(bh)))
        py_setitem(fields, "indexHash_", ih)
        py_setitem(fields, "bitHash_", bh)
    else:
        py_setitem(fields, "indexHash_", family_array(Python.tuple(degree, 0), "<i4", context))
        py_setitem(fields, "bitHash_", family_array(Python.tuple(degree, 0), "<i4", context))
    py_setitem(fields, "n_features_in_", PythonObject(d))


def _pcs_transform(settings: PythonObject, binding: PythonObject, x: ArrayMetadata,
                   degree: Int, components: Int, gamma: Float64, coef0: Float64,
                   context: PythonObject) raises -> PythonObject:
    """Transform with captured input/hash metadata and the admitted provider."""
    var n = x.source.rows
    var d = x.source.cols
    if d != _pcs_int(family_field(settings, "n_features_in_"), "n_features_in_"):
        raise Error("Number of features of test samples does not match that of training samples.")
    var index_hash = _pcs_hash(family_field(settings, "indexHash_"), degree, "indexHash_", context)
    var bit_hash = _pcs_hash(family_field(settings, "bitHash_"), degree, "bitHash_", context)
    var nf = index_hash.source.cols
    if bit_hash.source.cols != nf or nf != d + (1 if coef0 != 0.0 else 0):
        raise Error("ValueError: PolynomialCountSketch hash tables do not match the fitted features")
    var out = family_array(Python.tuple(n, components), "<f4", context)
    var name = String("pcs") if getenv("MOJOLEARN_XN_OLD_ITEMS") == "1" else String("pcs_resident")
    xn_execute(binding, name,
               Python.list(x.source.addr, index_hash.source.addr, bit_hash.source.addr, classical_address(out)),
               Python.list(n, d, nf, components, degree),
               Python.list(Float64(Float32(gamma)), Float64(Float32(coef0))))
    _ = out
    _ = index_hash^
    _ = bit_hash^
    _ = x
    return out


def _pcs_operation(model: PythonObject, operation: String, args: PythonObject,
                   context: PythonObject) raises -> PythonObject:
    var public = pcs_public_begin(model, operation, args, context)
    var result = PythonObject(None)
    try:
        var admitted = xn_admit_arguments(estimator_arguments(model, "_mojolearn_x_neighbors", args, context), context)
        var settings = public.settings
        var fields = public.call.fields
        var degree = _pcs_int(settings["degree"], "degree")
        var components = _pcs_int(settings["n_components"], "n_components")
        if degree < 1 or components < 1:
            raise Error("degree and n_components must be >= 1")
        var coef0 = _pcs_real(settings["coef0"], "coef0")
        if public.call.fitting:
            _pcs_fit(fields, settings, admitted[0], public.x, degree, components, coef0, context)
            py_setitem(settings, "n_features_in_", fields["n_features_in_"])
            py_setitem(settings, "indexHash_", fields["indexHash_"])
            py_setitem(settings, "bitHash_", fields["bitHash_"])
        if operation == "family_xn_pcs_fit":
            result = model
        else:
            var gamma = _pcs_real(settings["gamma"], "gamma")
            result = _pcs_transform(settings, admitted[0], public.x, degree, components, gamma, coef0, context)
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
