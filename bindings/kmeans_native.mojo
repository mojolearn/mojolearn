# SPDX-License-Identifier: Apache-2.0
# KMeans' public methods, one native call each (python/mojolearn/cluster.py:
# `fit`, `predict`, `transform`, `fit_predict`, `fit_transform`). The shell
# hands over the binding `_bind` resolved for the model's numeric mode; this
# owner reads the parameters, makes the refusals the Python method made (same
# wording, same order), allocates the outputs, runs the binding's
# `kmeans_fit` / `kmeans_predict` / `kmeans_transform` with the argument lists
# `bindings/_mojolearn.mojo` documents, and stores the fitted attributes. The
# kernels and their inputs are unchanged, so the bits are too.
# Lane migrate-classical-b2: `fit_transform` uploads X once
# (`kmeans_fit_transform`), and `predict` / `transform` read the fitted
# centroids from a device copy minted at the first query after a fit
# (`kmeans_centers_prepare`, cluster/resident_centers.mojo) instead of
# uploading them per call; a binding without those doors (the CPU host
# binding) takes the per-call entries as before.
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from bindings.classical_arrays import classical_address
from bindings.classical_attributes import classical_connect_numpy
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.family_common import (
    FAMILY_F32P, family_repr, family_str, family_is_str, family_is_int, family_isinstance,
    family_type_name, family_call, family_call_kw, family_entry, family_array, family_view,
    family_cast_f32, family_f32, family_nonfinite, family_sample_weight, family_field,
)
from providers.python.family_public_lifecycle_pyapi import family_lifecycle_spec, family_lifecycle_begin, family_lifecycle_finish
from providers.python.neighbors_public_lifecycle_pyapi import neighbors_public_end, neighbors_public_refusal

comptime _INIT_ARRAY = 2
comptime _HOLDER = "_resident_holder"


def _kmeans_optional_entry(binding: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """The binding's compiled entry `name`, or None where the loaded binding
    has no such door (the CPU host binding refuses an absent name with
    ImportError, a module with AttributeError: both mean "no residency
    here", and the pending exception is discarded)."""
    try:
        return family_entry(binding, name, context)
    except error:
        if String(error) != PYTHON_ERROR_PENDING:
            raise error
        _ = optimizer_python_error(error)
        ref cpy = Python().cpython()
        cpy.lib.get_function[NoneType]("PyErr_Clear")()
        return PythonObject(None)


def _kmeans_release(model: PythonObject) raises:
    """Drop the resident centroids, if held, through the minting binding's
    `kmeans_centers_release`. Quiet on a binding that cannot be reached any
    more and on a handle already released (a copied instance)."""
    var b = Python.import_module("builtins")
    var holder = py_get(py_fields(model), PythonObject(_HOLDER), PythonObject(None))
    if holder is None or py_type(holder) is not b.dict:
        return
    var held = py_get(holder, PythonObject("index"), PythonObject(None))
    if held is None:
        return
    py_setitem(holder, "index", PythonObject(None))
    try:
        _ = py_call(py_call(b.getattr, held[2], PythonObject("kmeans_centers_release")), held[1])
    except:
        pass


def _kmeans_resident(model: PythonObject, binding: PythonObject, centers: PythonObject, c: PythonObject,
                     k: Int, d: Int, context: PythonObject) raises -> PythonObject:
    """The handle of the device copy of the fitted centroids, uploaded at
    the first query after a fit and reused while `cluster_centers_` is the
    same object (the held key keeps a reference to it, so it cannot be
    replaced by a new array at the same address) and the binding is the one
    that minted it; None where the binding has no resident door. The holder
    dictionary changes, never the model's `__dict__` keys after a fit."""
    var prepare = _kmeans_optional_entry(binding, "kmeans_centers_prepare", context)
    if prepare is None:
        return PythonObject(None)
    var b = Python.import_module("builtins")
    var fields = py_fields(model)
    var holder = py_get(fields, PythonObject(_HOLDER), PythonObject(None))
    if holder is None or py_type(holder) is not b.dict:
        holder = Python.dict()
        py_setitem(fields, _HOLDER, holder)
    var held = py_get(holder, PythonObject("index"), PythonObject(None))
    if held is not None:
        if held[0][0] is centers and held[2] is binding and Int(py=held[0][1]) == k and Int(py=held[0][2]) == d:
            return held[1]
        _kmeans_release(model)
    var handle = family_call(prepare, classical_address(c), Python.list(PythonObject(k), PythonObject(d)))
    # The minting binding travels with the handle: the release (here, and
    # the native finalizer when the instance is dropped) is its own
    # `kmeans_centers_release`.
    py_setitem(holder, "index", Python.tuple(Python.tuple(centers, PythonObject(k), PythonObject(d)), handle, binding))
    return handle


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


def _kmeans_fit(model: PythonObject, args: PythonObject, context: PythonObject,
                transform: Bool = False) raises -> PythonObject:
    """`KMeans.fit(X, sample_weight)`; returns the fitted labels, or with
    `transform` the distances of the fitted rows to the final centroids
    (`KMeans.fit_transform`): one `kmeans_fit_transform` call, X uploaded
    once, where the binding has that door; otherwise the fit and then the
    transform, as before."""
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
        _ = centers  # keep the Python object alive through the native call above
        _ = c0  # keep the Python object alive through the native call above
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
    var result: PythonObject
    var distances = PythonObject(None)
    var joint = PythonObject(None)
    if transform:
        joint = _kmeans_optional_entry(binding, "kmeans_fit_transform", context)
    if joint is not None:
        distances = family_array(Python.tuple(n, k), "<f4", context)
        result = family_call(joint, classical_address(x), classical_address(centers), classical_address(labels),
                             classical_address(weights), classical_address(distances), params)
        _ = labels  # keep the Python object alive through the native call above
        _ = centers  # keep the Python object alive through the native call above
        _ = x  # keep the Python object alive through the native call above
        _ = distances  # keep the Python object alive through the native call above
        _ = weights  # keep the Python object alive through the native call above
    else:
        var fit = family_entry(binding, "kmeans_fit", context)
        result = family_call(fit, classical_address(x), classical_address(centers), classical_address(labels),
                             classical_address(weights), params)
        _ = labels  # keep the Python object alive through the native call above
        _ = centers  # keep the Python object alive through the native call above
        _ = x  # keep the Python object alive through the native call above
        _ = weights  # keep the Python object alive through the native call above
    _ = x^
    _ = weights^
    py_setitem(fields, "cluster_centers_", centers)
    py_setitem(fields, "labels_", labels)
    py_setitem(fields, "inertia_", result[0])
    py_setitem(fields, "n_iter_", result[1])
    py_setitem(fields, "sum_scale_", result[2])
    py_setitem(fields, "weight_scale_", result[3])
    py_setitem(fields, "n_features_in_", PythonObject(d))
    # The last fit's resident centroids are released; the next query
    # uploads the new ones once (`_kmeans_resident`).
    _kmeans_release(model)
    py_setitem(fields, _HOLDER, Python.dict())
    if transform:
        if distances is None:
            return _kmeans_inference(model, binding, source, True, context)
        return distances
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
    var handle = _kmeans_resident(model, binding, centers, c, k, d, context)
    if transform:
        out = family_array(Python.tuple(n, k), "<f4", context)
        if handle is not None:
            _ = family_call(family_entry(binding, "kmeans_transform_resident", context), handle,
                            classical_address(x), classical_address(out), params)
            _ = x  # keep the Python object alive through the native call above
        else:
            _ = family_call(family_entry(binding, "kmeans_transform", context), classical_address(x),
                            classical_address(c), classical_address(out), params)
            _ = x  # keep the Python object alive through the native call above
            _ = c  # keep the Python object alive through the native call above
    else:
        out = family_array(Python.tuple(n), "<i4", context)
        if handle is not None:
            _ = family_call(family_entry(binding, "kmeans_predict_resident", context), handle,
                            classical_address(x), classical_address(out), params)
            _ = x  # keep the Python object alive through the native call above
        else:
            _ = family_call(family_entry(binding, "kmeans_predict", context), classical_address(x),
                            classical_address(c), classical_address(out), params)
            _ = x  # keep the Python object alive through the native call above
            _ = c  # keep the Python object alive through the native call above
    _ = x^
    _ = c^
    return out


def _kmeans_operation_route(model: PythonObject, operation: String, args: PythonObject,
                            context: PythonObject) raises -> PythonObject:
    """The KMeans public operations: args[0] is the resolved binding."""
    if operation == "family_kmeans_fit":
        _ = _kmeans_fit(model, args, context)
        return model
    if operation == "family_kmeans_fit_predict":
        return _kmeans_fit(model, args, context)
    if operation == "family_kmeans_fit_transform":
        return _kmeans_fit(model, args, context, True)
    if operation == "family_kmeans_getstate":
        # A pickle or a deepcopy carries no device handle: it is meaningful
        # only in the process and registry that minted it. The copy uploads
        # its own centroids at its first query.
        var state = py_call(Python.import_module("builtins").dict, py_fields(model))
        _ = py_call(state.pop, PythonObject(_HOLDER), PythonObject(None))
        return state
    if operation == "family_kmeans_setstate":
        var fields = py_fields(model)
        _ = py_call(fields.update, args[0])
        py_setitem(fields, _HOLDER, Python.dict())
        return PythonObject(None)
    if operation == "family_kmeans_predict":
        return _kmeans_inference(model, args[0], args[1], False, context)
    if operation == "family_kmeans_transform":
        return _kmeans_inference(model, args[0], args[1], True, context)
    raise Error("unknown KMeans operation " + operation)


def kmeans_operation(model: PythonObject, operation: String, args: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    """The public operation: the estimators that formerly took the `_mode`
    class-creation wrappers run their whole lifecycle around the owner
    (providers/python/family_public_lifecycle_pyapi.mojo); every other
    operation goes straight to the owner."""
    var lifecycle = family_lifecycle_spec(operation)
    if lifecycle.method.byte_length() == 0:
        return _kmeans_operation_route(model, operation, args, context)
    var call = family_lifecycle_begin(model, operation, lifecycle, args, 1, context)
    var result = PythonObject(None)
    try:
        result = _kmeans_operation_route(model, operation, call.args, context)
    except error:
        var failure = neighbors_public_refusal(error, call, model)
        raise failure^
    finally:
        neighbors_public_end(model, call)
    return family_lifecycle_finish(model, lifecycle, result, call, context)
