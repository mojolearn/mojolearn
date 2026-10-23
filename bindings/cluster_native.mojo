# SPDX-License-Identifier: Apache-2.0
# The x_cluster estimators' inference methods, one native call each
# (python/mojolearn/_expansion_cluster.py): `_CentersMixin.predict`,
# `transform` and `score` (MiniBatchKMeans, BisectingKMeans), MeanShift
# `predict`, BisectingKMeans `predict` (the bisection tree descent) and
# AffinityPropagation `predict`. The shell hands over the binding `_bind`
# resolved; this owner makes the Python methods' refusals, calls the
# binding's `x_cluster_call(which, x, x_size, aux, aux_size, ints, floats)`
# with the same words, and builds the Arrays from the lists it returns, as
# `Array._from_flat` did. Same entry, same inputs: the bits do not move.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields
from bindings.classical_arrays import classical_address, classical_wrap_array, classical_c_bytes
from core.family_cast_impl import family_fill_i32
from bindings.classical_attributes import classical_connect_numpy
from bindings.family_common import (
    FAMILY_I32P, family_call, family_entry, family_array, family_f32, family_as_f32, family_nonfinite,
    family_shape_repr,
)

comptime _E_NEAREST = 0
comptime _E_DISTANCES = 1
comptime _E_BISECT_PREDICT = 4


def _xc_fitted(model: PythonObject, attr: String) raises -> PythonObject:
    """`_XCluster._check_fitted`: the attribute, or the refusal."""
    var b = Python.import_module("builtins")
    var value = py_call(b.getattr, model, PythonObject(attr), PythonObject(None))
    if value is None:
        raise Error("RuntimeError: this " + String(py_type(model).__name__) + " is not fitted yet")
    return value


def _xc_input(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_XCluster._input_like_fit`: `_f32(X)` (as_f32_c, nonempty, finite)
    and the fitted feature count."""
    var x = PythonObject(None)
    try:
        x = family_f32(source, 2, "X", context)
    except error:
        if family_nonfinite(error, "X"):
            raise Error("mojolearn: X contains NaN or infinity")
        raise Error(String(error))
    var d = Int(py=x.shape[1])
    var b = Python.import_module("builtins")
    var fitted = py_call(b.getattr, model, PythonObject("n_features_in_"))
    if d != Int(py=fitted):
        raise Error("mojolearn: X has " + String(d) + " features, " + String(py_type(model).__name__)
                    + " was fitted with " + String(Int(py=fitted)))
    return x


def _xc_call(binding: PythonObject, which: Int, x: PythonObject, aux: PythonObject, ints: PythonObject,
             context: PythonObject) raises -> PythonObject:
    """`_XCluster._call` (no float parameters): (f, i, s)."""
    var entry = family_entry(binding, "x_cluster_call", context)
    var aux_address = PythonObject(0)
    var aux_size = 0
    if aux is not None:
        aux_size = Int(py=aux.size)
        if aux_size:
            aux_address = classical_address(aux)
    return family_call(entry, PythonObject(which), classical_address(x), PythonObject(Int(py=x.size)), aux_address,
                       PythonObject(aux_size), ints, Python.list())


def _xc_array(flat: PythonObject, shape: PythonObject, typestr: String, code: String,
              context: PythonObject) raises -> PythonObject:
    """`Array._from_flat(flat, shape, typestr)`."""
    var store = py_call(context["array_type"], PythonObject(code), flat)
    return classical_wrap_array(store, shape, typestr, "C", context)


def _xc_nearest(model: PythonObject, binding: PythonObject, x: PythonObject, centers_field: PythonObject,
                context: PythonObject) raises -> PythonObject:
    var centers = family_as_f32(centers_field, 2, "cluster_centers_", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var result = _xc_call(binding, _E_NEAREST, x, centers,
                          Python.list(PythonObject(n), PythonObject(Int(py=centers.shape[0])), PythonObject(d)), context)
    _ = centers^
    return _xc_array(result[1][0], Python.tuple(n), "<i4", "i", context)


def family_as_i32_words(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """The int32 words of a C-ordered int32 array as a list of ints
    (`[int(v) for v in flat_bytes(value).cast("i")]`)."""
    var store = py_call(context["array_type"], PythonObject("i"))
    _ = py_call(store.frombytes, classical_c_bytes(value, "_tree_nodes", context))
    return py_call(Python.import_module("builtins").list, store)


def cluster_operation(model: PythonObject, operation: String, args: PythonObject,
                      context: PythonObject) raises -> PythonObject:
    """args[0] is the resolved binding, args[1] X."""
    classical_connect_numpy(context)
    var b = Python.import_module("builtins")
    var binding = args[0]
    if operation == "family_xc_predict":
        var centers = _xc_fitted(model, "cluster_centers_")
        var x = _xc_input(model, args[1], context)
        return _xc_nearest(model, binding, x, centers, context)
    if operation == "family_xc_transform":
        var centers_field = _xc_fitted(model, "cluster_centers_")
        var x = _xc_input(model, args[1], context)
        var centers = family_as_f32(centers_field, 2, "cluster_centers_", context)
        var n = Int(py=x.shape[0])
        var d = Int(py=x.shape[1])
        var k = Int(py=centers.shape[0])
        var result = _xc_call(binding, _E_DISTANCES, x, centers,
                              Python.list(PythonObject(n), PythonObject(k), PythonObject(d)), context)
        _ = centers^
        return _xc_array(result[0][0], Python.tuple(n, k), "<f4", "f", context)
    if operation == "family_xc_score":
        var centers_field = _xc_fitted(model, "cluster_centers_")
        var x = _xc_input(model, args[1], context)
        var centers = family_as_f32(centers_field, 2, "cluster_centers_", context)
        var n = Int(py=x.shape[0])
        var d = Int(py=x.shape[1])
        # the float-float fold of the distances, in the binding
        var result = _xc_call(binding, _E_NEAREST, x, centers,
                              Python.list(PythonObject(n), PythonObject(Int(py=centers.shape[0])), PythonObject(d),
                                          PythonObject(1)), context)
        _ = centers^
        return family_call(Python.import_module("operator").neg, result[2][0])
    if operation == "family_xc_bisect_predict":
        _ = _xc_fitted(model, "cluster_centers_")
        var x = _xc_input(model, args[1], context)
        var n = Int(py=x.shape[0])
        var d = Int(py=x.shape[1])
        var fields = py_fields(model)
        # [n, d] + the fitted bisection tree's node words.
        var nodes = family_as_i32_words(fields["_tree_nodes"], context)
        var ints = Python.list(PythonObject(n), PythonObject(d))
        _ = py_call(ints.extend, nodes)
        var result = _xc_call(binding, _E_BISECT_PREDICT, x, fields["_tree_centers"], ints, context)
        return _xc_array(result[1][0], Python.tuple(n), "<i4", "i", context)
    if operation == "family_xc_ap_predict":
        _ = _xc_fitted(model, "labels_")
        var affinity = py_call(b.getattr, model, PythonObject("affinity"))
        if py_type(affinity) is b.str and String(affinity) == "precomputed":
            raise Error("Predict method is not supported when affinity='precomputed'.")
        var x = _xc_input(model, args[1], context)
        var centers = py_call(b.getattr, model, PythonObject("cluster_centers_"))
        if Int(py=centers.shape[0]) == 0:
            var labels = family_array(Python.tuple(Int(py=x.shape[0])), "<i4", context)
            var words = FAMILY_I32P(unsafe_from_address=Int(py=classical_address(labels)))
            family_fill_i32(words, Int(py=x.shape[0]), -1)
            return labels
        return _xc_nearest(model, binding, x, centers, context)
    raise Error("RuntimeError: unknown x_cluster operation " + operation)
