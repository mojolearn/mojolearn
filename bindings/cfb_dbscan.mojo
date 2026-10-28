# SPDX-License-Identifier: Apache-2.0
# DBSCAN `fit`, `fit_predict` and `predict`, one native call each
# (python/mojolearn/density.py). The shell hands over the binding `_bind`
# resolved; this owner makes the Python method's refusals in its order and
# wording, calls the binding's `dbscan_fit_core` and `dbscan_core_arrays`
# with the argument lists the Python made, and stores the attributes the
# Python stored. With `prediction_data=True` the core rows become resident
# on the device at fit (bindings/cfb_resident.mojo), so `predict` uploads
# only its queries. Same entries, same inputs: the bits do not move.
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_contains, py_setitem, py_fields
from bindings.classical_arrays import classical_address, classical_wrap_array, classical_is_array
from core.family_cast_impl import family_fill_i32, family_all_finite_f32
from bindings.family_common import (
    FAMILY_F32P, FAMILY_I32P, family_repr, family_call, family_array, family_as_f32,
    family_sample_weight, family_min, family_field,
)
from bindings.cluster_native import _reference_queries
from bindings.cfb_resident import cfb_fresh_holder, cfb_mint, cfb_reference_labels
from bindings.native_loaded_module import native_loaded_module
from providers.python.neural_descriptor_pyapi import neural_descriptor_entry
from bindings.family_raw_attr import family_raw_get, family_raw_get_or, family_raw_has

comptime _METRIC_PRECOMPUTED = 3
comptime _EPS_NN_BRUTE_FORCE = 0


def cfb_borrow_copy(x: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`x if input_copied_ else x.copy()` as a mojolearn Array: an Array the
    cast made is kept, anything else (the caller's own buffer) is copied so
    a later write by the caller cannot move what predict reads."""
    if classical_is_array(x, context) and not (x is source):
        return x
    var b = native_loaded_module("builtins")
    var out = family_array(py_call(b.tuple, x.shape), "<f4", context)
    memcpy(dest=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(out))),
           src=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(x))), count=Int(py=x.size))
    _ = out  # keep the Python object alive through the native call above
    return out


def _wrap(store: PythonObject, shape: PythonObject, typestr: String, context: PythonObject) raises -> PythonObject:
    """`frombytes(store.tobytes(), typestr, shape)`: the binding's
    `array.array` as the Array's store (the same bytes, no second copy)."""
    return classical_wrap_array(store, shape, typestr, "C", context)


def _dbscan_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`DBSCAN.fit(X, y=None, sample_weight=None)`."""
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    var binding = args[0]
    var source = args[1]
    var weights_source = args[2]
    var prediction_data = family_field(fields, "prediction_data")
    if py_type(prediction_data) is not b.bool:
        raise Error("TypeError: mojolearn DBSCAN: prediction_data must be a bool, got "
                    + String(py_type(prediction_data).__name__))
    var metrics = context["family_dbscan_metrics"]
    var metric_attr = family_field(fields, "metric")
    var metric_key = family_call(family_call(b.str, metric_attr).lower)
    if not py_contains(metrics, metric_key):
        raise Error("mojolearn DBSCAN: metric=" + family_repr(metric_attr) + " is refused; it "
                    + "must be one of " + family_repr(py_call(b.sorted, metrics)) + ". 'euclidean'/'l2' is the "
                    + "implemented cuML arm; 'manhattan'/'l1'/'cityblock' is this "
                    + "library's own L1 arm (DEVIATION 27, dbscan/impl/neighbors/"
                    + "epsilon_neighborhood.mojo); 'cosine' and 'precomputed' "
                    + "are cuML's other two (dbscan.pyx:110-115, DEVIATIONS 5113 "
                    + "and 5114)")
    var metric = Int(py=metrics[metric_key])
    var algorithms = context["family_cfb_dbscan_algorithms"]
    var algorithm = family_field(fields, "algorithm")
    if not py_contains(algorithms, algorithm):
        raise Error("mojolearn DBSCAN: algorithm=" + family_repr(algorithm) + " is refused; "
                    + "it must be one of " + family_repr(py_call(b.sorted, algorithms)) + ". Both name an "
                    + "eps-neighborhood SEARCH: 'rbc' is the random ball cover "
                    + "(the default, DEVIATION 35) and 'brute' is the fused "
                    + "all-pairs scan. scikit-learn's 'kd_tree' and 'ball_tree' "
                    + "are not offered because a recursive, data-dependent tree "
                    + "descent is the wrong shape for a GPU -- divergent branches "
                    + "and pointer chasing instead of coalesced reads -- and its "
                    + "pruning decays past about ten features anyway. The ball "
                    + "cover prunes arithmetically over two flat arrays and "
                    + "measured 2.7x-27x over brute force at 16k-200k rows on "
                    + "this hardware; see the class docstring")
    if not py_contains(context["family_cfb_dbscan_arms"][algorithm], PythonObject(metric)):
        raise Error("mojolearn DBSCAN: metric=" + family_repr(metric_attr) + " with "
                    + "algorithm=" + family_repr(algorithm) + " is refused. The ball cover "
                    + "computes Euclidean distances for its landmark radii and "
                    + "its three pruning bounds (neighbors/impl/"
                    + "ball_cover/) against the radius itself, so it serves "
                    + "'euclidean' only ('cosine' compares unit rows against "
                    + "2 * eps, a threshold the index does not take). Pass "
                    + "algorithm='brute', which serves this metric. This is a "
                    + "scope boundary and not a property of the algorithm: the "
                    + "ball cover's pruning rests on the triangle inequality, "
                    + "which L1 satisfies, so an L1 index is reachable work in "
                    + "that lane")
    var predicting = Bool(py=prediction_data)
    if metric == _METRIC_PRECOMPUTED and predicting:
        raise Error("mojolearn DBSCAN: prediction_data=True with "
                    + "metric='precomputed' is refused by name: predict needs each "
                    + "query's distances to the core samples, which a distance "
                    + "matrix over the training rows does not give")
    var eps = family_call(b.float, family_field(fields, "eps"))
    if Float64(py=eps) <= 0:
        raise Error("mojolearn DBSCAN eps must be positive")
    var min_samples = Int(py=family_call(b.int, family_field(fields, "min_samples")))
    if min_samples < 1:
        raise Error("mojolearn DBSCAN min_samples must be at least 1")
    var cap_attr = family_field(fields, "max_iterations")
    var cap = 0 if cap_attr is None else Int(py=family_call(b.int, cap_attr))
    if cap < 0:
        raise Error("mojolearn DBSCAN max_iterations must be None (the fixed "
                    + "point) or a positive cap")
    var x = family_as_f32(source, 2, "X", context)
    py_setitem(fields, "input_copied_", PythonObject(not (x is source)))
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    if not family_all_finite_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(x))), Int(py=x.size)):
        # Before this check a NaN row reached the kernel and came back
        # labelled noise (-1), silently; scikit-learn refuses it.
        raise Error("mojolearn DBSCAN: X contains a NaN or an infinity; a "
                    + "non-finite row has no distance to any other, so it is "
                    + "refused by name")
    _ = x  # keep the Python object alive through the native calls above
    if metric == _METRIC_PRECOMPUTED:
        if n != d:
            raise Error("mojolearn DBSCAN: metric='precomputed' needs a square "
                        + "distance matrix, got shape " + family_repr(py_call(b.tuple, x.shape)))
        if Float64(family_min(x)) < 0:
            raise Error("mojolearn DBSCAN: metric='precomputed' refuses a "
                        + "negative distance, as scikit-learn does")
    var labels = family_array(Python.tuple(n), "<i4", context)
    var budget_attr = family_field(fields, "max_mbytes_per_batch")
    var budget = 0 if budget_attr is None else Int(py=family_call(b.int, budget_attr))
    if budget < 0:
        raise Error("mojolearn DBSCAN max_mbytes_per_batch cannot be negative")
    # sample_weight: `0` is the binding's `sample_weight == nullptr`. The
    # weights are held until after the call (their address is in flight).
    var w = PythonObject(None)
    var weight_addr = PythonObject(0)
    if weights_source is not None:
        w = family_sample_weight(weights_source, n, True, context)
        weight_addr = classical_address(w)
        _ = w  # keep the Python object alive through the native call above
    # ORDER MATCHES bindings/_mojolearn_estimators.mojo::dbscan_fit_binding:
    # n_rows, n_features, eps, min_samples, budget_mb, max_iter,
    # eps_nn_method, metric. precomputed takes the brute arm's adjacency path
    # whatever `algorithm` names (DEVIATION 5114).
    var method = _EPS_NN_BRUTE_FORCE if metric == _METRIC_PRECOMPUTED else Int(py=algorithms[algorithm])
    var params = Python.list(PythonObject(n), PythonObject(d), eps, PythonObject(min_samples),
                             PythonObject(budget), PythonObject(cap), PythonObject(method), PythonObject(metric))
    var core = family_array(Python.tuple(n), "<u1", context)
    var fit = neural_descriptor_entry(binding, "dbscan_fit_core")
    var n_iter = family_call(fit, classical_address(x), classical_address(labels), weight_addr,
                             classical_address(core), params)
    _ = labels  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _ = core  # keep the Python object alive through the native call above
    _ = w^
    py_setitem(fields, "n_iter_", n_iter)
    _ = core  # keep the Python object alive through the native call above
    _ = labels  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, "labels_", labels)
    py_setitem(fields, "n_features_in_", PythonObject(d))
    py_setitem(fields, "n_samples_fit_", PythonObject(n))
    # `_store_core`: the core rows, their training indices and their labels,
    # ascending training index, from the fit's own core mask.
    var gather = neural_descriptor_entry(binding, "dbscan_core_arrays")
    var parts = family_call(gather, classical_address(x), classical_address(labels), classical_address(core),
                            Python.list(PythonObject(n), PythonObject(d)))
    _ = core  # keep the Python object alive through the native call above
    _ = labels  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    var m = len(parts[0])
    _ = core  # keep the Python object alive through the native call above
    _ = labels  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    var indices = _wrap(parts[0], Python.tuple(m), "<i4", context)
    _ = core  # keep the Python object alive through the native call above
    _ = labels  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    var components = _wrap(parts[1], Python.tuple(m, d), "<f4", context)
    _ = core  # keep the Python object alive through the native call above
    _ = labels  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    var core_labels = _wrap(parts[2], Python.tuple(m), "<i4", context)
    _ = core  # keep the Python object alive through the native call above
    _ = labels  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, "core_sample_indices_", indices)
    py_setitem(fields, "components_", components)
    py_setitem(fields, "_core_labels", core_labels)
    _ = core^
    _ = x^
    cfb_fresh_holder(model)
    if predicting:
        cfb_mint(model, binding, components, indices, core_labels, m, d, metric, context)


def _dbscan_predict(model: PythonObject, binding: PythonObject, source: PythonObject,
                    context: PythonObject) raises -> PythonObject:
    """`DBSCAN.predict`: the nearest core sample's label within eps, over
    the resident core rows."""
    var b = native_loaded_module("builtins")
    if not family_raw_has(model, "labels_"):
        raise Error("mojolearn DBSCAN.predict: this DBSCAN instance is not fitted yet; call fit first")
    var components = family_raw_get_or(model, "components_", PythonObject(None))
    if not Bool(py=py_call(b.bool, family_raw_get(model, "prediction_data"))) or components is None:
        raise Error("mojolearn DBSCAN.predict: prediction data was not stored. Fit with "
                    + "DBSCAN(prediction_data=True), which keeps the core samples this "
                    + "rule needs (DEVIATION 2740)")
    var n_features = Int(py=family_raw_get(model, "n_features_in_"))
    var q = _reference_queries(source, n_features, "DBSCAN.predict", context)
    var nq = Int(py=q.shape[0])
    var n_core = Int(py=components.shape[0])
    if n_core == 0:
        var labels = family_array(Python.tuple(nq), "<i4", context)
        family_fill_i32(FAMILY_I32P(unsafe_from_address=Int(py=classical_address(labels))), nq, -1)
        _ = labels  # keep the Python object alive through the native call above
        return labels
    var metric_name = py_call(py_call(b.str, family_raw_get(model, "metric")).lower)
    var metric = Int(py=context["family_dbscan_metrics"][metric_name])
    var eps = family_call(b.float, family_raw_get(model, "eps"))
    var result = cfb_reference_labels(model, binding, components,
                                      family_raw_get(model, "core_sample_indices_"),
                                      family_raw_get(model, "_core_labels"), q, n_core,
                                      n_features, metric, eps, 1, context)
    _ = q^
    return result


def cfb_dbscan_operation(model: PythonObject, operation: String, args: PythonObject,
                         context: PythonObject) raises -> PythonObject:
    """args: (binding, X[, sample_weight])."""
    if operation == "family_cfb_dbscan_fit":
        _dbscan_fit(model, args, context)
        return model
    if operation == "family_cfb_dbscan_fit_predict":
        _dbscan_fit(model, args, context)
        return py_fields(model)["labels_"]
    if operation == "family_cfb_dbscan_predict":
        return _dbscan_predict(model, args[0], args[1], context)
    raise Error("RuntimeError: unknown DBSCAN operation " + operation)
