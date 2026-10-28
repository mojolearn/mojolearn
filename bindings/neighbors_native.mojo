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
from providers.python.x_neighbors_descriptor_pyapi import xn_admit_arguments, xn_execute, xn_module
from providers.python.neural_descriptor_pyapi import neural_descriptor_entry
from core.cast_in import CAST_KIND_SIGNED
from bindings.classical_attributes import classical_connect_numpy
from bindings.family_common import (
    family_repr, family_call, family_call_kw, family_array,
    family_field, family_f32_snapshot, family_owned_snapshot,
)
from bindings.host_archive_native import host_archive_radius_refusals
from bindings.neighbors_knn_native import neighbors_knn_operation, knn_release
from bindings.neighbors_local_native import neighbors_local_operation
from bindings.neighbors_semi_native import neighbors_semi_operation
from bindings.neighbors_graph_native import neighbors_graph_operation
from bindings.neighbors_ann_native import neighbors_ann_operation
from bindings.neighbors_kde_native import neighbors_kde_operation


def _pcs_seed(seed: PythonObject) raises -> PythonObject:
    """`_expansion_neighbors._seed_word`: an int as given; None (numpy's
    global RandomState) or a caller's RandomState gives one draw in
    [0, 2^31); anything else is refused."""
    var b = Python.import_module("builtins")
    if py_type(seed) is py_type(PythonObject(0)):
        return seed
    # Remaining provider work: NumPy RandomState/None must be admitted through
    # a native RNG state API. This retained path still executes NumPy randint
    # through CPython and is not a completed native migration.
    var state = seed
    if seed is None:
        var numpy = PythonObject(None)
        try:
            numpy = Python.import_module("numpy")
        except:
            raise Error("ImportError: mojolearn: _expansion_neighbors requires optional NumPy support. "
                        + "Install it with: python -m pip install \"mojolearn[numpy]\" "
                        + "(\"mojolearn[verify]\" also includes NumPy).")
        state = numpy.random.mtrand._rand
    elif not (Bool(py=py_call(b.hasattr, seed, PythonObject("randint")))
              and Bool(py=py_call(b.hasattr, seed, PythonObject("random_sample")))
              and Bool(py=py_call(b.hasattr, seed, PythonObject("uniform")))):
        raise Error(family_repr(seed) + " cannot be used to seed a numpy.random.RandomState instance")
    var keywords = Python.dict()
    py_setitem(keywords, "size", PythonObject(1))
    var draw = family_call_kw(py_call(b.getattr, state, PythonObject("randint")), keywords,
                              PythonObject(0), PythonObject(1 << 31))
    return family_call(b.int, family_call(py_call(b.getattr, draw, PythonObject("__getitem__")), PythonObject(0)))


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


def _pcs_fit(fields: PythonObject, binding: PythonObject, x: ArrayMetadata,
             degree: Int, components: Int, coef0: Float64, context: PythonObject) raises:
    """Fit hash tables from the operation's single admitted input snapshot."""
    var d = x.source.cols
    var nf = d + (1 if coef0 != 0.0 else 0)
    var seed = _pcs_seed(family_field(fields, "random_state"))
    if nf > 0:
        var draw = neural_descriptor_entry(xn_module(binding), "x_neighbors_kfeat_pcs_draw_idn", False)
        if draw is None:
            raise Error("NotImplementedError: mojolearn PolynomialCountSketch: this binary lacks "
                        + "x_neighbors_kfeat_pcs_draw_idn (rebuild without -D MOJOLEARN_IDN_XN_SKETCH_CTR_OFF)")
        var ih = family_array(Python.tuple(degree, nf), "<i4", context)
        var bh = family_array(Python.tuple(degree, nf), "<i4", context)
        if py_type(seed) is not py_type(PythonObject(0)):
            raise Error("TypeError: PolynomialCountSketch random_state must resolve to a native int")
        ref cpy = Python().cpython()
        # Preserve arbitrary-sized integer seeds without Python operator.and_.
        var raw_seed = cpy.lib.get_function[UInt64]("PyLong_AsUnsignedLongLongMask")(seed._obj_ptr)
        var word = Int(raw_seed & 0xFFFFFFFF)
        # Remaining provider work: PCS drawing is not yet in XN's table.
        # The numerical extension entry below is still CPython-dispatched.
        _ = family_call(draw, Python.list(word, components, degree * nf),
                        classical_address(ih), classical_address(bh))
        _ = ih
        _ = bh
        py_setitem(fields, "indexHash_", ih)
        py_setitem(fields, "bitHash_", bh)
    else:
        py_setitem(fields, "indexHash_", family_array(Python.tuple(degree, 0), "<i4", context))
        py_setitem(fields, "bitHash_", family_array(Python.tuple(degree, 0), "<i4", context))
    py_setitem(fields, "n_features_in_", PythonObject(d))


def _pcs_transform(fields: PythonObject, binding: PythonObject, x: ArrayMetadata,
                   degree: Int, components: Int, gamma: Float64, coef0: Float64,
                   context: PythonObject) raises -> PythonObject:
    """Transform with captured input/hash metadata and the admitted provider."""
    var n = x.source.rows
    var d = x.source.cols
    if d != _pcs_int(family_field(fields, "n_features_in_"), "n_features_in_"):
        raise Error("Number of features of test samples does not match that of training samples.")
    var index_hash = _pcs_hash(family_field(fields, "indexHash_"), degree, "indexHash_", context)
    var bit_hash = _pcs_hash(family_field(fields, "bitHash_"), degree, "bitHash_", context)
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
    classical_connect_numpy(context)
    var admitted = xn_admit_arguments(estimator_arguments(model, "_mojolearn_x_neighbors", args, context), context)
    var fields = py_fields(model)
    var x = family_f32_snapshot(admitted[1], 2, "X", context, True)
    var degree = _pcs_int(family_field(fields, "degree"), "degree")
    var components = _pcs_int(family_field(fields, "n_components"), "n_components")
    if degree < 1 or components < 1:
        raise Error("degree and n_components must be >= 1")
    var coef0 = _pcs_real(family_field(fields, "coef0"), "coef0")
    if operation != "family_xn_pcs_transform":
        _pcs_fit(fields, admitted[0], x, degree, components, coef0, context)
    if operation == "family_xn_pcs_fit":
        _ = x^
        return model
    var gamma = _pcs_real(family_field(fields, "gamma"), "gamma")
    var result = _pcs_transform(fields, admitted[0], x, degree, components, gamma, coef0, context)
    _ = x^
    return result


def _radius_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`RadiusNeighbors.fit(X)`: the refusals (`_check_refusals`, which also
    resolves the metric), then the points stored as float32."""
    classical_connect_numpy(context)
    var fields = py_fields(model)
    host_archive_radius_refusals(fields, String(py_type(model).__name__), context)
    var snapshot = family_f32_snapshot(args[0], 2, "X", context, True)
    var index = family_owned_snapshot(snapshot, context)
    py_setitem(fields, "_index", index)
    py_setitem(fields, "n_samples_fit_", PythonObject(snapshot.source.rows))
    py_setitem(fields, "n_features_in_", PythonObject(snapshot.source.cols))
    _ = snapshot^


def neighbors_operation(model: PythonObject, operation: String, args: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    if operation == "family_xn_knn_release":
        knn_release(model)
        return PythonObject(None)
    if operation == "family_xn_radius_refusals":
        host_archive_radius_refusals(py_fields(model), String(py_type(model).__name__), context)
        return PythonObject(None)
    if operation == "family_xn_radius_fit":
        _radius_fit(model, args, context)
        return model
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
