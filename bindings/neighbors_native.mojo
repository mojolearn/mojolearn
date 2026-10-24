# SPDX-License-Identifier: Apache-2.0
# PolynomialCountSketch `fit`, `transform` and `fit_transform`, one native
# call each (python/mojolearn/_expansion_neighbors.py), and RadiusNeighbors
# `fit` (python/mojolearn/neighbors.py: its refusals, shared with `load`, then
# the stored index). The shell hands over
# the binding `_bind` resolved; this owner reads the parameters with the
# Python methods' refusals, draws the hash tables with the binding's
# counter-based generator (`x_neighbors_kfeat_pcs_draw_idn`) and runs the
# sketch (`xn_pcs_resident`, or `xn_pcs` under the MOJOLEARN_XN_OLD_ITEMS A/B
# arm) with the argument lists x_neighbors documents. The same kernels on the
# same inputs: the bits do not move.
from std.os import getenv
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from bindings.classical_arrays import classical_address
from bindings.classical_attributes import classical_connect_numpy
from bindings.family_common import (
    family_repr, family_is_int, family_call, family_call_kw, family_entry, family_array,
    family_as_f32_dense, family_field, family_owned,
)
from bindings.host_archive_native import host_archive_radius_refusals
from bindings.neighbors_knn_native import neighbors_knn_operation
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
    if family_is_int(seed):
        return seed
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


def _f32_scalar(value: PythonObject) raises -> PythonObject:
    """`_f32_scalar`: a Python double rounded once to float32."""
    return PythonObject(Float64(Float32(Float64(py=family_call(Python.import_module("builtins").float, value)))))


def _pcs_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`PolynomialCountSketch.fit(X)`: the hash tables."""
    classical_connect_numpy(context)
    var b = Python.import_module("builtins")
    var operator = Python.import_module("operator")
    var fields = py_fields(model)
    var x = family_as_f32_dense(args[1], 2, "X", context)
    var d = Int(py=x.shape[1])
    _ = x^
    var nf = d + (1 if Bool(py=family_call(operator.ne, family_field(fields, "coef0"), PythonObject(0))) else 0)
    var deg = Int(py=family_call(b.int, family_field(fields, "degree")))
    var nc = Int(py=family_call(b.int, family_field(fields, "n_components")))
    if deg < 1 or nc < 1:
        raise Error("degree and n_components must be >= 1")
    var seed = _pcs_seed(family_field(fields, "random_state"))
    if nf > 0:
        var draw = py_call(b.getattr, args[0], PythonObject("x_neighbors_kfeat_pcs_draw_idn"), PythonObject(None))
        if draw is None:
            # The MT19937 Python draw behind the _OFF define is deleted; a
            # binary without the device draw refuses.
            raise Error("NotImplementedError: mojolearn PolynomialCountSketch: this binary lacks "
                        + "x_neighbors_kfeat_pcs_draw_idn (rebuild without -D MOJOLEARN_IDN_XN_SKETCH_CTR_OFF)")
        if py_type(draw) is not context["family_builtin_type"]:
            raise Error("TypeError: mojolearn: x_neighbors_kfeat_pcs_draw_idn must be a compiled entry")
        var ih = family_array(Python.tuple(deg, nf), "<i4", context)
        var bh = family_array(Python.tuple(deg, nf), "<i4", context)
        var word = family_call(operator.and_, seed, PythonObject(0xFFFFFFFF))
        _ = family_call(draw, Python.list(word, PythonObject(nc), PythonObject(deg * nf)),
                        classical_address(ih), classical_address(bh))
        _ = bh  # keep the Python object alive through the native call above
        _ = ih  # keep the Python object alive through the native call above
        py_setitem(fields, "indexHash_", ih)
        py_setitem(fields, "bitHash_", bh)
    else:
        py_setitem(fields, "indexHash_", family_array(Python.tuple(deg, 0), "<i4", context))
        py_setitem(fields, "bitHash_", family_array(Python.tuple(deg, 0), "<i4", context))
    py_setitem(fields, "n_features_in_", PythonObject(d))


def _pcs_transform(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`PolynomialCountSketch.transform(X)`: the count sketches' circular
    convolution."""
    classical_connect_numpy(context)
    var b = Python.import_module("builtins")
    var fields = py_fields(model)
    var x = family_as_f32_dense(args[1], 2, "X", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    if d != Int(py=family_field(fields, "n_features_in_")):
        raise Error("Number of features of test samples does not match that of training samples.")
    var index_hash = family_field(fields, "indexHash_")
    var bit_hash = family_field(fields, "bitHash_")
    var nf = Int(py=index_hash.shape[1])
    var nc = Int(py=family_call(b.int, family_field(fields, "n_components")))
    var deg = Int(py=family_call(b.int, family_field(fields, "degree")))
    var out = family_array(Python.tuple(n, nc), "<f4", context)
    var old_items = getenv("MOJOLEARN_XN_OLD_ITEMS") == "1"
    var name = String("xn_pcs_resident")
    if old_items:
        name = String("xn_pcs")
    var op = family_entry(args[0], name, context)
    # X, indexHash_, bitHash_, out; n, d, nf, nc, degree; gamma, coef0 (float32).
    _ = family_call(op, Python.list(classical_address(x), classical_address(index_hash), classical_address(bit_hash),
                                    classical_address(out)),
                    Python.list(PythonObject(n), PythonObject(d), PythonObject(nf), PythonObject(nc), PythonObject(deg)),
                    Python.list(_f32_scalar(family_field(fields, "gamma")), _f32_scalar(family_field(fields, "coef0"))))
    _ = bit_hash  # keep the Python object alive through the native call above
    _ = index_hash  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = x^
    _ = index_hash^
    _ = bit_hash^
    return out


def _radius_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`RadiusNeighbors.fit(X)`: the refusals (`_check_refusals`, which also
    resolves the metric), then the points stored as float32."""
    classical_connect_numpy(context)
    var fields = py_fields(model)
    host_archive_radius_refusals(fields, String(py_type(model).__name__), context)
    var index = family_owned(family_as_f32_dense(args[0], 2, "X", context), context)
    py_setitem(fields, "_index", index)
    py_setitem(fields, "n_samples_fit_", PythonObject(Int(py=index.shape[0])))
    py_setitem(fields, "n_features_in_", PythonObject(Int(py=index.shape[1])))


def _knn_release(model: PythonObject) raises:
    """`NearestNeighbors._release_resident_index`: drop the device copy of
    the index, if one is held, through the minting binding's
    `knn_index_release`. Quiet on a binding that cannot be reached any more
    and on a handle already released."""
    var b = Python.import_module("builtins")
    var fields = py_fields(model)
    var holder = py_get(fields, PythonObject("_resident_holder"), PythonObject(None))
    if holder is None or py_type(holder) is not b.dict:
        return
    var held = py_get(holder, PythonObject("index"), PythonObject(None))
    if held is None:
        return
    py_setitem(holder, "index", PythonObject(None))
    try:
        _ = py_call(py_call(b.getattr, held[2], PythonObject("knn_index_release")), held[1])
    except:
        pass


def neighbors_operation(model: PythonObject, operation: String, args: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    if operation == "family_xn_knn_release":
        _knn_release(model)
        return PythonObject(None)
    if operation == "family_xn_radius_refusals":
        host_archive_radius_refusals(py_fields(model), String(py_type(model).__name__), context)
        return PythonObject(None)
    if operation == "family_xn_radius_fit":
        _radius_fit(model, args, context)
        return model
    if operation == "family_xn_pcs_fit":
        _pcs_fit(model, args, context)
        return model
    if operation == "family_xn_pcs_transform":
        return _pcs_transform(model, args, context)
    if operation == "family_xn_pcs_fit_transform":
        _pcs_fit(model, args, context)
        return _pcs_transform(model, args, context)
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
