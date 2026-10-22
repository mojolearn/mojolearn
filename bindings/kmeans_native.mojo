# SPDX-License-Identifier: Apache-2.0
# KMeans' public methods, one native call each (python/mojolearn/cluster.py:
# `fit`, `predict`, `transform`, `fit_predict`, `fit_transform`). The shell
# hands over the binding `_bind` resolved for the model's numeric mode; this
# owner reads the parameters, makes the refusals the Python method made (same
# wording, same order), allocates the outputs, runs the binding's
# `kmeans_fit` / `kmeans_predict` / `kmeans_transform` with the argument lists
# `bindings/_mojolearn.mojo` documents, and stores the fitted attributes. The
# kernels and their inputs are unchanged, so the bits are too.
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from bindings.classical_arrays import classical_address
from bindings.classical_attributes import classical_connect_numpy
from bindings.family_common import (
    FAMILY_F32P, family_repr, family_str, family_is_str, family_is_int, family_isinstance,
    family_type_name, family_call, family_call_kw, family_entry, family_array, family_view,
    family_cast_f32, family_f32, family_nonfinite, family_sample_weight, family_field,
)

comptime _INIT_ARRAY = 2


def _kmeans_nonfinite(name: String, where: String) -> Error:
    return Error("mojolearn KMeans." + where + ": " + name + " contains a NaN or an infinity; "
                 + "a non-finite row has no distance to any center, so it is refused by name")


def _kmeans_f32(source: PythonObject, name: String, where: String, context: PythonObject) raises -> PythonObject:
    """`as_f32_c(source, ndim=2)` then the by-name non-finite refusal."""
    try:
        return family_f32(source, 2, name, context)
    except error:
        if family_nonfinite(error, name):
            raise _kmeans_nonfinite(name, where)
        raise Error(String(error))


def _kmeans_metric(fields: PythonObject, context: PythonObject) raises -> Int:
    """`KMeans._metric_code`: a name from the table, or `int(metric)`."""
    var metric = family_field(fields, "metric")
    if family_is_str(metric):
        var table = context["family_kmeans_metrics"]
        if not py_contains(table, metric):
            raise Error("mojolearn: metric must be one of "
                        + family_repr(py_call(Python.import_module("builtins").sorted, table))
                        + ", got " + family_repr(metric))
        return Int(py=table[metric])
    return Int(py=family_call(Python.import_module("builtins").int, metric))


def _kmeans_callable_init(init: PythonObject, source: PythonObject, k: PythonObject,
                          random_state: PythonObject) raises -> PythonObject:
    """scikit-learn's callable `init` (`init(X, n_clusters,
    random_state=random_state)`), called once with the fit's rows: a NumPy
    array and a RandomState seeded by `random_state` when NumPy is
    installed, the input and the int otherwise."""
    var data = source
    var state = random_state
    var numpy = PythonObject(None)
    try:
        numpy = Python.import_module("numpy")
    except:
        numpy = PythonObject(None)
    if numpy is not None:
        data = family_call(numpy.asarray, source)
        state = family_call(numpy.random.RandomState, random_state)
    var keywords = Python.dict()
    py_setitem(keywords, "random_state", state)
    return family_call_kw(init, keywords, data, k)


def _kmeans_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`KMeans.fit(X, sample_weight)`; returns the fitted labels."""
    classical_connect_numpy(context)
    var b = Python.import_module("builtins")
    var binding = args[0]
    var source = args[1]
    var sample_weight = args[2]
    var fields = py_fields(model)
    var k_obj = family_field(fields, "n_clusters")
    var init = family_field(fields, "init")
    var init_centroids = family_field(fields, "init_centroids")
    var init_code: Int
    if family_is_str(init):
        var inits = context["family_kmeans_inits"]
        if not py_contains(inits, init):
            raise Error("mojolearn: init must be one of " + family_repr(py_call(b.sorted, inits))
                        + ", got " + family_repr(init))
        init_code = Int(py=inits[init])
    elif Bool(py=py_call(b.callable, init)):
        init_code = _INIT_ARRAY
        init_centroids = _kmeans_callable_init(init, source, k_obj, family_field(fields, "random_state"))
    elif family_is_int(init):
        init_code = Int(py=family_call(b.int, init))
    else:
        init_code = _INIT_ARRAY
        init_centroids = init
    var metric_code = _kmeans_metric(fields, context)
    var factor = family_field(fields, "oversampling_factor")
    if family_isinstance(factor, b.bool) or not family_isinstance(factor, Python.tuple(b.int, b.float)):
        raise Error("TypeError: mojolearn: oversampling_factor must be a float, got " + family_type_name(factor))
    var oversampling = family_call(b.float, factor)
    var x = _kmeans_f32(source, "X", "fit", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    if Bool(py=family_call(Python.import_module("operator").gt, k_obj, PythonObject(n))):
        raise Error("mojolearn: n_clusters=" + family_str(k_obj) + " exceeds n_samples=" + String(n))
    var k = Int(py=family_call(b.int, k_obj))
    var centers = family_array(Python.tuple(k, d), "<f4", context)
    if init_code == _INIT_ARRAY:
        if init_centroids is None:
            raise Error("mojolearn: init='array' needs init_centroids")
        var view = family_view(init_centroids, 2, "init_centroids", context)
        if Int(py=view.shape[0]) != k or Int(py=view.shape[1]) != d:
            raise Error("mojolearn: init_centroids must be (" + family_str(k_obj) + ", " + String(d)
                        + "), got " + family_repr(py_call(b.tuple, view.shape)))
        var c0 = PythonObject(None)
        try:
            c0 = family_cast_f32(view, "init_centroids", context)
        except error:
            if family_nonfinite(error, "init_centroids"):
                raise _kmeans_nonfinite("init_centroids", "fit")
            raise Error(String(error))
        # A COPY, on purpose: the kernel writes the centroids in place and
        # `c0` may borrow the caller's array.
        memcpy(dest=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(centers))),
               src=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(c0))), count=k * d)
        _ = c0^
        _ = view^
    # int32 cluster ids: the kernel writes uint32 ids below 2**31, the same
    # bytes, into the array the model keeps (DEVIATION 2672).
    var labels = family_array(Python.tuple(n), "<i4", context)
    var n_weights = 0
    var weights = x
    if sample_weight is not None:
        weights = family_sample_weight(sample_weight, n, True, context)
        n_weights = n
    # ORDER MATCHES bindings/_mojolearn.mojo::kmeans_fit_binding: n_samples,
    # n_features, n_clusters, n_weights, max_iter, tol, seed, n_init, init,
    # metric, oversampling_factor.
    var params = Python.list()
    _ = py_call(params.append, PythonObject(n))
    _ = py_call(params.append, PythonObject(d))
    _ = py_call(params.append, k_obj)
    _ = py_call(params.append, PythonObject(n_weights))
    _ = py_call(params.append, family_field(fields, "max_iter"))
    _ = py_call(params.append, family_call(b.float, family_field(fields, "tol")))
    _ = py_call(params.append, family_call(b.int, family_field(fields, "random_state")))
    _ = py_call(params.append, family_field(fields, "n_init"))
    _ = py_call(params.append, PythonObject(init_code))
    _ = py_call(params.append, PythonObject(metric_code))
    _ = py_call(params.append, oversampling)
    var fit = family_entry(binding, "kmeans_fit", context)
    var result = family_call(fit, classical_address(x), classical_address(centers), classical_address(labels),
                             classical_address(weights), params)
    _ = x^
    _ = weights^
    py_setitem(fields, "cluster_centers_", centers)
    py_setitem(fields, "labels_", labels)
    py_setitem(fields, "inertia_", result[0])
    py_setitem(fields, "n_iter_", result[1])
    py_setitem(fields, "sum_scale_", result[2])
    py_setitem(fields, "weight_scale_", result[3])
    py_setitem(fields, "n_features_in_", PythonObject(d))
    return labels


def _kmeans_inference(model: PythonObject, binding: PythonObject, source: PythonObject, transform: Bool,
                      context: PythonObject) raises -> PythonObject:
    """`KMeans.predict` (labels) or `KMeans.transform` (distances)."""
    classical_connect_numpy(context)
    var fields = py_fields(model)
    var centers = py_get(fields, PythonObject("cluster_centers_"), PythonObject(None))
    if centers is None:
        raise Error("RuntimeError: this estimator is not fitted yet")
    var metric_code = _kmeans_metric(fields, context)
    var x = _kmeans_f32(source, "X", "predict/transform", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var dc = Int(py=centers.shape[1])
    if d != dc:
        raise Error("mojolearn: X has " + String(d) + " features, KMeans was fitted with " + String(dc))
    var c = family_f32(centers, 2, "cluster_centers_", context)
    var k = Int(py=c.shape[0])
    # ORDER MATCHES bindings/_mojolearn.mojo::kmeans_predict_binding and
    # kmeans_transform_binding: n_samples, n_features, n_clusters, metric.
    var params = Python.list(PythonObject(n), PythonObject(d), PythonObject(k), PythonObject(metric_code))
    var out: PythonObject
    if transform:
        out = family_array(Python.tuple(n, k), "<f4", context)
        _ = family_call(family_entry(binding, "kmeans_transform", context), classical_address(x),
                        classical_address(c), classical_address(out), params)
    else:
        out = family_array(Python.tuple(n), "<i4", context)
        _ = family_call(family_entry(binding, "kmeans_predict", context), classical_address(x),
                        classical_address(c), classical_address(out), params)
    _ = x^
    _ = c^
    return out


def kmeans_operation(model: PythonObject, operation: String, args: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    """The KMeans public operations: args[0] is the resolved binding."""
    if operation == "family_kmeans_fit":
        _ = _kmeans_fit(model, args, context)
        return model
    if operation == "family_kmeans_fit_predict":
        return _kmeans_fit(model, args, context)
    if operation == "family_kmeans_fit_transform":
        _ = _kmeans_fit(model, args, context)
        return _kmeans_inference(model, args[0], args[1], True, context)
    if operation == "family_kmeans_predict":
        return _kmeans_inference(model, args[0], args[1], False, context)
    if operation == "family_kmeans_transform":
        return _kmeans_inference(model, args[0], args[1], True, context)
    raise Error("unknown KMeans operation " + operation)
