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
from core.family_cast_impl import family_fill_i32, family_all_finite_f32
from bindings.classical_attributes import classical_connect_numpy
from bindings.family_common import (
    FAMILY_F32P, FAMILY_I32P, family_repr, family_call, family_entry, family_array, family_f32, family_as_f32, family_nonfinite,
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


def _reference_queries(source: PythonObject, n_features: Int, where: String, context: PythonObject) raises -> PythonObject:
    """`density._check_queries`: float32 C order, the fitted feature count,
    at least one row, every value finite."""
    var q = family_as_f32(source, 2, "X", context)
    var d = Int(py=q.shape[1])
    if d != n_features:
        raise Error("mojolearn " + where + ": X has " + String(d) + " features, the fit saw " + String(n_features))
    if Int(py=q.shape[0]) < 1:
        raise Error("mojolearn " + where + ": X has no rows; refused by name")
    if not family_all_finite_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(q))), Int(py=q.size)):
        raise Error("mojolearn " + where + ": X contains a NaN or an infinity; a non-finite "
                    + "query has no distance order to take a nearest row from, so it is "
                    + "refused by name")
    return q


def _labeled_reference(binding: PythonObject, refs: PythonObject, keys: PythonObject, labels: PythonObject,
                       q: PythonObject, n_refs: Int, n_features: Int, metric: Int, eps: PythonObject,
                       has_thresh: Int, context: PythonObject) raises -> PythonObject:
    """The binding's `labeled_reference_predict`: the label of the nearest
    reference row (ORDER MATCHES bindings/_mojolearn_estimators.mojo::
    labeled_reference_predict_binding)."""
    var nq = Int(py=q.shape[0])
    var out = family_array(Python.tuple(nq), "<i4", context)
    var chosen = family_array(Python.tuple(nq), "<i4", context)
    var entry = family_entry(binding, "labeled_reference_predict", context)
    _ = family_call(entry, Python.list(classical_address(refs), classical_address(keys), classical_address(labels),
                                       classical_address(q), classical_address(out), classical_address(chosen)),
                    Python.list(PythonObject(n_refs), PythonObject(nq), PythonObject(n_features), PythonObject(metric),
                                eps, PythonObject(has_thresh)))
    _ = chosen^
    return out


def _dbscan_predict(model: PythonObject, binding: PythonObject, source: PythonObject,
                    context: PythonObject) raises -> PythonObject:
    """`DBSCAN.predict`: the nearest core sample's label within eps."""
    var b = Python.import_module("builtins")
    if not Bool(py=family_call(b.hasattr, model, PythonObject("labels_"))):
        raise Error("mojolearn DBSCAN.predict: this DBSCAN instance is not fitted yet; call fit first")
    var components = py_call(b.getattr, model, PythonObject("components_"), PythonObject(None))
    if not Bool(py=py_call(b.bool, py_call(b.getattr, model, PythonObject("prediction_data")))) or components is None:
        raise Error("mojolearn DBSCAN.predict: prediction data was not stored. Fit with "
                    + "DBSCAN(prediction_data=True), which keeps the core samples this "
                    + "rule needs (DEVIATION 2740)")
    var n_features = Int(py=py_call(b.getattr, model, PythonObject("n_features_in_")))
    var q = _reference_queries(source, n_features, "DBSCAN.predict", context)
    var nq = Int(py=q.shape[0])
    var n_core = Int(py=components.shape[0])
    if n_core == 0:
        var labels = family_array(Python.tuple(nq), "<i4", context)
        family_fill_i32(FAMILY_I32P(unsafe_from_address=Int(py=classical_address(labels))), nq, -1)
        return labels
    var metric_name = py_call(py_call(b.str, py_call(b.getattr, model, PythonObject("metric"))).lower)
    var metric = Int(py=context["family_dbscan_metrics"][metric_name])
    var eps = family_call(b.float, py_call(b.getattr, model, PythonObject("eps")))
    var result = _labeled_reference(binding, components, py_call(b.getattr, model, PythonObject("core_sample_indices_")),
                                    py_call(b.getattr, model, PythonObject("_core_labels")), q, n_core, n_features,
                                    metric, eps, 1, context)
    _ = q^
    return result


def _agglo_predict(model: PythonObject, binding: PythonObject, source: PythonObject,
                   context: PythonObject) raises -> PythonObject:
    """`AgglomerativeClustering.predict`: the nearest training row's cluster
    (single linkage, euclidean)."""
    var b = Python.import_module("builtins")
    if not Bool(py=family_call(b.hasattr, model, PythonObject("labels_"))):
        raise Error("mojolearn AgglomerativeClustering.predict: this instance is not "
                    + "fitted yet; call fit first")
    var linkage = py_call(b.getattr, model, PythonObject("linkage"))
    var metric = py_call(b.getattr, model, PythonObject("metric"))
    # `_predicts`: single linkage under a euclidean metric name.
    var predicts = py_type(linkage) is b.str and String(linkage) == "single"
    if predicts:
        predicts = Bool(py=family_call(Python.import_module("operator").contains, context["family_agglo_predict_metrics"], metric))
    if not predicts:
        raise Error("NotImplementedError: mojolearn AgglomerativeClustering.predict: linkage=" + family_repr(linkage)
                    + ", metric=" + family_repr(metric) + " has no predict rule; DEVIATION 2740 is single "
                    + "linkage's own criterion (the nearest training row's cluster) under "
                    + "the euclidean metric only")
    var refs = py_call(b.getattr, model, PythonObject("_fit_X"), PythonObject(None))
    if refs is None:
        raise Error("mojolearn AgglomerativeClustering.predict: prediction data was not "
                    + "stored. Fit with AgglomerativeClustering(prediction_data=True), "
                    + "which keeps the training rows this rule needs (DEVIATION 2740)")
    var n_features = Int(py=py_call(b.getattr, model, PythonObject("n_features_in_")))
    var q = _reference_queries(source, n_features, "AgglomerativeClustering.predict", context)
    var labels = py_call(b.getattr, model, PythonObject("labels_"))
    # The key of every training row is its label: ties go to the lowest
    # cluster label, then the lowest training index.
    var result = _labeled_reference(binding, refs, labels, labels, q, Int(py=refs.shape[0]), n_features, 0,
                                    PythonObject(0.0), 0, context)
    _ = q^
    return result


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
    if operation == "family_xc_dbscan_predict":
        return _dbscan_predict(model, binding, args[1], context)
    if operation == "family_xc_agglo_predict":
        return _agglo_predict(model, binding, args[1], context)
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
