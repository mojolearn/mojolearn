# SPDX-License-Identifier: Apache-2.0
# The compiled owner of the fits of MiniBatchKMeans (fit, partial_fit),
# BisectingKMeans, MeanShift, OPTICS and AffinityPropagation
# (python/mojolearn/_expansion_cluster.py, with the centers' fit_predict and
# fit_transform) and of HDBSCAN (python/mojolearn/hdbscan.py): one native call
# per public method, reached from bindings/family_native.mojo for every
# `family_cfa_*` operation.
#
# Each fit makes the Python method's refusals (same wording, same order),
# builds the same integer and float words, calls the same binding entry once
# (`x_cluster_call`, or HDBSCAN's `hdbscan_fit` and
# `hdbscan_generate_prediction_data`) with the same inputs, and stores the
# attributes the Python set, with the same names, dtypes and shapes. Every
# iteration, batch, bisection and seed loop already runs inside the entry,
# which uploads X once per fit. Same entries, same operands, same order: the
# bits do not move.
from std.math import isinf
from std.python import Python, PythonObject
from bindings.admitted_family_door import admit_family, admit_family_weights
from providers.python.array_metadata_pyapi_door import ArrayPin
from bindings.native_loaded_module_door import native_loaded_module
from bindings.family_raw_attr_pack import family_raw_get, family_raw_get_or
from std.python._cpython import PyObjectPtr, GILReleased
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setattr
from bindings.classical_arrays_door import classical_address
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.family_common_unpack import (
    family_repr,
    family_str,
    family_is_str,
    family_isinstance,
    family_nonfinite,
    family_shape_repr,
)
from bindings.family_common_pack import (
    family_call,
    family_array,
)
from bindings.cluster_pending_error_unpack import _pending, _pending_text, _drop_pending
from bindings.cluster_fit_public_native_unpack import (
    _attr, _attr_or_none, _truthy, _int, _float, _eq, _one_of, _set, _f32, _seed, _weights, _init_array,
)
from bindings.cluster_fit_public_native_pack import meanshift_fit
from cluster.cluster_fit_door_run import cluster_rows_cover
from bindings.cluster_public_native_pack import cluster_operation, _xc_array
from bindings.operation_names_xc import XC_OPN_FAMILY_XC_VARIANTS_TRANSFORM
from bindings.operation_names_cfa import (
    CFA_OPN_FAMILY_CFA_MBK_FIT, CFA_OPN_FAMILY_CFA_MBK_PARTIAL_FIT, CFA_OPN_FAMILY_CFA_BISECT_FIT,
    CFA_OPN_FAMILY_CFA_FIT_PREDICT, CFA_OPN_FAMILY_CFA_FIT_TRANSFORM, CFA_OPN_FAMILY_CFA_MEANSHIFT_FIT,
    CFA_OPN_FAMILY_CFA_MEANSHIFT_FIT_PREDICT, CFA_OPN_FAMILY_CFA_OPTICS_FIT, CFA_OPN_FAMILY_CFA_OPTICS_FIT_PREDICT,
    CFA_OPN_FAMILY_CFA_AP_FIT, CFA_OPN_FAMILY_CFA_AP_FIT_PREDICT, CFA_OPN_FAMILY_CFA_HDBSCAN_FIT)
from bindings.py_cluster_resident import CRES_X_CLUSTER, cres_forget
from core.kmeans_variants_api import (
    MiniBatchFitRequest, MiniBatchPartialRequest, BisectFitRequest, VariantsResultSlot, VariantsSpan, VARIANTS_CAP_DEVSCAN,
)
from core.kmeans_variants_plan import MiniBatchStreamState
from core.cluster_centroid_fit_api import MeanShiftFitRequest, AffinityFitRequest, CentroidFitResultSlot
from core.affinity_preference_plan import AffinityPreferencePlan
from core.optics_fit_api import OpticsMetric, OpticsFitRequest, OpticsFitResultSlot
from bindings.hdbscan_fit_boundary_pack import hdbscan_fit
from bindings.optics_fit_boundary_door import optics_fit_connection, OpticsPackedResult
from bindings.cluster_centroid_fit_boundary_door import centroid_fit_connection, CentroidPackedResult
from bindings.affinity_preference_boundary_door import affinity_preference_prepare
from bindings.kmeans_variants_boundary_door import (
    variants_connection, variants_span, VariantsAuxiliaryPins, VariantsPackedResult,
)
from bindings.py_optional import py_or_none, py_present
from bindings.cluster_keys_unpack import cluster_keys
from bindings.py_balanced import py_item

# x_cluster/entries.mojo: the entry numbers of `x_cluster_call`
# OPTICS metric codes (x_cluster/optics.mojo)
comptime _OPTICS_PRECOMPUTED = 5


# ---------------------------------------------------------------- MiniBatchKMeans

def _mbk_fit(model: PythonObject, binding: PythonObject, source: PythonObject, sample_weight: PythonObject,
             context: PythonObject) raises:
    """`MiniBatchKMeans.fit(X, sample_weight)`."""
    var connection = variants_connection(binding)
    var devscan = (connection.capabilities & VARIANTS_CAP_DEVSCAN) != 0
    var x: PythonObject
    if devscan:
        x = admit_family(source, 2, "X", context).array
    else:
        x = _f32(source, "X", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var k = Int(py=_int(_attr(model, "n_clusters")))
    cluster_rows_cover(k, n)
    var weights = _weights(sample_weight, n, context)
    var init = _attr(model, "init")
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var init_arr = Optional[PythonObject]()
    var init_random = False
    if family_is_str(init):
        if not _one_of(init, "k-means++", "random"):
            raise Error("init must be 'k-means++', 'random' or an array, got " + family_repr(init))
        init_random = _eq(init, PythonObject("random"))
    else:
        init_arr = py_present(_init_array(init, k, d, context))
    var n_init_field = _attr(model, "n_init")
    var n_init: Int
    if _eq(n_init_field, PythonObject("auto")):
        n_init = 3 if (not init_arr and init_random) else 1
    else:
        n_init = Int(py=_int(n_init_field))
    if n_init < 1:
        raise Error("n_init must be >= 1")
    var mni_field = _attr(model, "max_no_improvement")
    var mni = PythonObject(-1) if mni_field is None else _int(mni_field)
    var max_iter = _int(_attr(model, "max_iter"))
    var batch_size = _int(_attr(model, "batch_size"))
    var init_size_field = _attr(model, "init_size")
    var init_size = PythonObject(0) if init_size_field is None else _int(init_size_field)
    var seed = _seed(_attr(model, "random_state"))
    var parts: List[PythonObject] = [py_or_none(init_arr), weights]
    var auxiliary = VariantsAuxiliaryPins(parts, context)
    var tolerance = _float(_attr(model, "tol"))
    var ratio = _float(_attr(model, "reassignment_ratio"))
    var source_span = variants_span(x)
    # Preserve native-word narrowing after auxiliary and float admission.
    var request = MiniBatchFitRequest(source_span, auxiliary.plan.span(), n, d, k,
        Int(py=max_iter), Int(py=batch_size), Int(py=mni), Int(py=init_size), n_init,
        Bool(init_arr), seed, not init_arr and init_random, weights is not None,
        devscan, Float64(py=tolerance), Float64(py=ratio))
    var result = VariantsResultSlot()
    try:
        with GILReleased(Python()):
            connection.driver.minibatch_fit(request, result)
    except error:
        if devscan and _pending_text(error).find("contains NaN or infinity") >= 0:
            _drop_pending()
            raise Error("mojolearn: X contains NaN or infinity")
        raise Error(String(error))
    finally:
        _ = connection.provider
        _ = auxiliary
        _ = x
        _ = init_arr
        _ = weights
    var packed = VariantsPackedResult(result, False)
    _set(model, "cluster_centers_", _xc_array(packed.centers, Python.tuple(k, d), "<f4", "f", context))
    _set(model, "counts_", _xc_array(packed.counts, Python.tuple(k), "<f4", "f", context))
    _set(model, "labels_", _xc_array(packed.labels, Python.tuple(n), "<i4", "i", context))
    _set(model, "inertia_", PythonObject(result.storage[0].inertia))
    _set(model, "n_steps_", PythonObject(result.storage[0].steps))
    _set(model, "n_iter_", PythonObject(result.storage[0].iterations))
    _set(model, "n_features_in_", PythonObject(d))


def _mbk_partial_fit(model: PythonObject, binding: PythonObject, source: PythonObject,
                     sample_weight: PythonObject, context: PythonObject) raises:
    """`MiniBatchKMeans.partial_fit(X, sample_weight)`: one step on all of X;
    the random stream and the reassignment counter carry over."""
    var b = native_loaded_module("builtins")
    var x = _f32(source, "X", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var k = Int(py=_int(_attr(model, "n_clusters")))
    var first = _attr_or_none(model, "cluster_centers_") is None
    if not first and not Bool(py=family_call(b.hasattr, model, PythonObject("_partial_state"))):
        # a model from fit(): continue from its centers and counts, as sklearn
        var n0 = min(Int(py=_int(_attr(model, "batch_size"))), n)
        _set(model, "_partial_batch", PythonObject(n0))
        _set(model, "_partial_init", Python.tuple(PythonObject(0), PythonObject(0)))
        var seed0 = _seed(_attr(model, "random_state"))
        _set(model, "_partial_state",
             Python.tuple(Python.tuple(PythonObject(seed0 >> 32), PythonObject(seed0 & 0xFFFFFFFF)), PythonObject(0)))
    if not first:
        var fitted = _attr(model, "n_features_in_")
        if _truthy(family_call(native_loaded_module("operator").ne, PythonObject(d), fitted)):
            raise Error("X has " + String(d) + " features, MiniBatchKMeans was fitted with " + family_str(fitted))
    var parts = List[PythonObject]()
    var mode = PythonObject(0)
    var init_size = PythonObject(0)
    var batch = PythonObject(0)
    var state0 = PythonObject(0)
    var state1 = PythonObject(0)
    var since = PythonObject(0)
    if first:
        cluster_rows_cover(k, n)
        var batch_eff = min(Int(py=_int(_attr(model, "batch_size"))), n)
        var size_field = _attr(model, "init_size")
        var size = 3 * batch_eff if size_field is None else Int(py=_int(size_field))
        if size < k:
            size = 3 * k
        size = min(size, n)
        var init = _attr(model, "init")
        var code: Int
        if family_is_str(init):
            if not _one_of(init, "k-means++", "random"):
                raise Error("init must be 'k-means++', 'random' or an array, got " + family_repr(init))
            code = 0 if _eq(init, PythonObject("k-means++")) else 1
        else:
            parts.append(_init_array(init, k, d, context))
            code = 2
        var seed = _seed(_attr(model, "random_state"))
        state0 = PythonObject(seed >> 32)
        state1 = PythonObject(seed & 0xFFFFFFFF)
        since = PythonObject(0)
        mode = PythonObject(code)
        init_size = PythonObject(size)
        batch = PythonObject(batch_eff)
        _set(model, "_partial_batch", batch)
        _set(model, "_partial_init", Python.tuple(mode, init_size))
        _set(model, "n_steps_", PythonObject(0))
    else:
        var pinit = _attr(model, "_partial_init")
        mode = pinit[0]
        init_size = pinit[1]
        batch = _attr(model, "_partial_batch")
        var pstate = _attr(model, "_partial_state")
        var words = pstate[0]
        since = pstate[1]
        state0 = words[0]
        state1 = words[1]
        parts.append(_attr(model, "cluster_centers_"))
        parts.append(_attr(model, "counts_"))
    parts.append(_weights(sample_weight, n, context))
    # Builtin int admission keeps the old order; native narrowing remains
    # after auxiliary/float preparation, as in the former compiled binding.
    var init_mode_word = _int(mode)
    var init_size_word = _int(init_size)
    var batch_word = _int(batch)
    var since_word = _int(since)
    var hi_word = _int(state0)
    var lo_word = _int(state1)
    var auxiliary = VariantsAuxiliaryPins(parts, context)
    var ratio = _float(_attr(model, "reassignment_ratio"))
    var connection = variants_connection(binding)
    var source_span = variants_span(x)
    var state = MiniBatchStreamState(Int(py=init_mode_word), Int(py=init_size_word), Int(py=batch_word),
        Int(py=since_word), Int(py=hi_word), Int(py=lo_word))
    var request = MiniBatchPartialRequest(source_span, auxiliary.plan.span(), n, d, k, first,
        state.init_mode, state.init_size, state.batch_size, state.since_reassignment,
        state.state_hi, state.state_lo, sample_weight is not None, Float64(py=ratio))
    var result = VariantsResultSlot()
    try:
        with GILReleased(Python()):
            connection.driver.minibatch_partial(request, result)
    finally:
        _ = connection.provider
        _ = auxiliary
        _ = parts
        _ = x
    var packed = VariantsPackedResult(result, False)
    _set(model, "cluster_centers_", _xc_array(packed.centers, Python.tuple(k, d), "<f4", "f", context))
    _set(model, "counts_", _xc_array(packed.counts, Python.tuple(k), "<f4", "f", context))
    _set(model, "labels_", _xc_array(packed.labels, Python.tuple(n), "<i4", "i", context))
    _set(model, "inertia_", PythonObject(result.storage[0].inertia))
    _set(model, "_partial_state", Python.tuple(Python.tuple(PythonObject(result.storage[0].state_hi),
        PythonObject(result.storage[0].state_lo)), PythonObject(result.storage[0].since_reassignment)))
    var steps = family_call(b.getattr, model, PythonObject("n_steps_"), PythonObject(0))
    _set(model, "n_steps_", family_call(native_loaded_module("operator").add, steps, PythonObject(1)))
    _set(model, "n_features_in_", PythonObject(d))


# ---------------------------------------------------------------- BisectingKMeans

def _bisect_fit(model: PythonObject, binding: PythonObject, source: PythonObject, sample_weight: PythonObject,
                context: PythonObject) raises:
    """`BisectingKMeans.fit(X, sample_weight)`."""
    var init = _attr(model, "init")
    if not _one_of(init, "random", "k-means++"):
        raise Error("NotImplementedError: mojolearn BisectingKMeans: init=" + family_repr(init)
                    + " is not implemented; 'random' or 'k-means++'")
    var strategy = _attr(model, "bisecting_strategy")
    if not _one_of(strategy, "biggest_inertia", "largest_cluster"):
        raise Error("bisecting_strategy must be 'biggest_inertia' or 'largest_cluster', got " + family_repr(strategy))
    var algorithm = _attr(model, "algorithm")
    if not _one_of(algorithm, "lloyd", "elkan"):
        raise Error("algorithm must be 'lloyd' or 'elkan', got " + family_repr(algorithm))
    var x = _f32(source, "X", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var k = Int(py=_int(_attr(model, "n_clusters")))
    cluster_rows_cover(k, n)
    var n_init = _int(_attr(model, "n_init"))
    if Int(py=n_init) < 1:
        raise Error("n_init must be >= 1")
    var random_init = _eq(init, PythonObject("random"))
    var max_iter = _int(_attr(model, "max_iter"))
    var seed = _seed(_attr(model, "random_state"))
    var largest_cluster = _eq(strategy, PythonObject("largest_cluster"))
    var w = PythonObject(None)
    if sample_weight is not None:
        w = _weights(sample_weight, n, context)
    var tolerance = _float(_attr(model, "tol"))
    var connection = variants_connection(binding)
    var weight_span = variants_span(w)
    var source_span = variants_span(x)
    var request = BisectFitRequest(source_span, weight_span, n, d, k, Int(py=n_init), random_init,
        Int(py=max_iter), seed, largest_cluster, Float64(py=tolerance))
    var result = VariantsResultSlot()
    try:
        with GILReleased(Python()):
            connection.driver.bisect_fit(request, result)
    finally:
        _ = connection.provider
        _ = x
        _ = w
    var packed = VariantsPackedResult(result, True)
    _set(model, "cluster_centers_", _xc_array(packed.centers, Python.tuple(k, d), "<f4", "f", context))
    var m = len(result.storage[0].tree_nodes) // 3
    _set(model, "_tree_centers", _xc_array(packed.tree_centers, Python.tuple(m, d), "<f4", "f", context))
    _set(model, "tree_nodes_", _xc_array(packed.tree_nodes, Python.tuple(m, 3), "<i4", "i", context))
    _set(model, "labels_", _xc_array(packed.labels, Python.tuple(n), "<i4", "i", context))
    _set(model, "inertia_", PythonObject(result.storage[0].inertia))
    _set(model, "n_features_in_", PythonObject(d))


def _centers_fit(model: PythonObject, binding: PythonObject, source: PythonObject,
                 context: PythonObject) raises:
    """`self.fit(X)` of a centers estimator (fit_predict and fit_transform
    pass no sample_weight)."""
    var nk = cluster_keys()
    if family_isinstance(model, py_item(context, nk[].k_family_cfa_minibatch)):
        _mbk_fit(model, binding, source, PythonObject(None), context)
    else:
        _bisect_fit(model, binding, source, PythonObject(None), context)


# ---------------------------------------------------------------- OPTICS

def _optics_metric(model: PythonObject, context: PythonObject) raises -> OpticsMetric:
    """`OPTICS._metric_code`: (code, p); euclidean at p=2 keeps the
    squared-distance route (-1)."""
    var b = native_loaded_module("builtins")
    var metric = _attr(model, "metric")
    var params = _attr(model, "metric_params")
    var p_field = _attr(model, "p")
    var p: PythonObject
    if _truthy(params):
        p = _float(family_call(params.get, PythonObject("p"), p_field))
    else:
        p = _float(p_field) if p_field is not None else PythonObject(2.0)
    var pv = Float64(py=p)
    if _eq(metric, PythonObject("minkowski")):
        if not pv >= 1:
            raise Error("p must be >= 1 for minkowski, got " + family_str(p))
        if pv == 2:
            return OpticsMetric(-1, 2.0)
        if pv == 1:
            return OpticsMetric(1, 1.0)
        if isinf(pv) and pv > 0:
            return OpticsMetric(2, pv)
        return OpticsMetric(3, pv)
    if _one_of(metric, "euclidean", "l2"):
        return OpticsMetric(-1, 2.0)
    var nk = cluster_keys()
    var table = py_item(context, nk[].k_family_cfa_optics_metrics)
    if not py_contains(table, metric):
        var names = py_call(native_loaded_module("operator").add, py_call(b.sorted, table),
                            Python.list(PythonObject("minkowski")))
        raise Error("NotImplementedError: mojolearn OPTICS: metric=" + family_repr(metric)
                    + " is not implemented; one of " + family_repr(names) + " (x_cluster/NOT_IMPLEMENTED.tsv)")
    return OpticsMetric(Int(py=table[metric]), pv)


def _optics_size(value: PythonObject, n: Int, name: String) raises -> Int:
    """`OPTICS._size`: an int count in [2, n], or a fraction in (0, 1] of n
    (at least 2)."""
    var b = native_loaded_module("builtins")
    var op = native_loaded_module("operator")
    if family_isinstance(value, b.bool) or value is None:
        raise Error(name + " must be an int >= 2 or a float in (0, 1]")
    if family_isinstance(value, b.int) or (family_isinstance(value, b.float)
                                           and _truthy(family_call(op.gt, value, PythonObject(1)))):
        if (_truthy(family_call(op.ne, _int(value), value)) or _truthy(family_call(op.lt, value, PythonObject(2)))
                or _truthy(family_call(op.gt, value, PythonObject(n)))):
            raise Error(name + " must be no greater than the number of samples (" + String(n) + ") and >= 2, got "
                        + family_str(value))
        return Int(py=_int(value))
    if not (_truthy(family_call(op.lt, PythonObject(0), value)) and _truthy(family_call(op.le, value, PythonObject(1)))):
        raise Error(name + " must be in (0, 1], got " + family_str(value))
    return max(2, Int(py=_int(family_call(op.mul, value, PythonObject(n)))))


def _optics_fit(model: PythonObject, binding: PythonObject, source: PythonObject,
                context: PythonObject) raises:
    """`OPTICS.fit(X)`."""
    var code = _optics_metric(model, context)
    var metric = code.code
    var method = _attr(model, "cluster_method")
    if not _one_of(method, "xi", "dbscan"):
        raise Error("cluster_method must be 'xi' or 'dbscan', got " + family_repr(method))
    var x = _f32(source, "X", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var ms = _optics_size(_attr(model, "min_samples"), n, "min_samples")
    var mcs_field = _attr(model, "min_cluster_size")
    var mcs = ms if mcs_field is None else _optics_size(mcs_field, n, "min_cluster_size")
    var max_eps = _float(_attr(model, "max_eps"))
    var eps_field = _attr(model, "eps")
    var eps = max_eps if eps_field is None else _float(eps_field)
    var is_xi = _eq(method, PythonObject("xi"))
    if _eq(method, PythonObject("dbscan")) and Float64(py=eps) > Float64(py=max_eps):
        raise Error("Specify an epsilon smaller than " + family_str(max_eps) + ". Got " + family_str(eps) + ".")
    var xi = _float(_attr(model, "xi"))
    if not (Float64(py=xi) >= 0 and Float64(py=xi) <= 1):
        raise Error("xi must be in [0, 1]")
    if metric == _OPTICS_PRECOMPUTED and n != d:
        raise Error("metric='precomputed' needs a square distance matrix, got (" + String(n) + ", " + String(d) + ")")
    # ORDER MATCHES x_cluster/entries.mojo::optics_entry: n, d, min_samples,
    # min_cluster_size, method (0 xi), predecessor_correction, metric;
    # max_eps, xi, eps, p.
    var predecessor_correction = _truthy(_attr(model, "predecessor_correction"))
    # Preserve the second xi read at its original stage after the flag.
    var repeated_xi = Float64(py=_float(_attr(model, "xi")))
    var connection = optics_fit_connection(binding)
    var request = OpticsFitRequest(variants_span(x), n, d, ms, mcs,
        0 if is_xi else 1, predecessor_correction, code.copy(),
        Float64(py=max_eps), repeated_xi, Float64(py=eps))
    var result = OpticsFitResultSlot()
    try:
        with GILReleased(Python()):
            connection.driver.fit(request, result)
    finally:
        _ = connection.provider
        _ = x
    var packed = OpticsPackedResult(result)
    _set(model, "core_distances_", _xc_array(packed.core_distances, Python.tuple(n), "<f4", "f", context))
    _set(model, "reachability_", _xc_array(packed.reachability, Python.tuple(n), "<f4", "f", context))
    _set(model, "ordering_", _xc_array(packed.ordering, Python.tuple(n), "<i4", "i", context))
    _set(model, "predecessor_", _xc_array(packed.predecessor, Python.tuple(n), "<i4", "i", context))
    _set(model, "labels_", _xc_array(packed.labels, Python.tuple(n), "<i4", "i", context))
    if is_xi:
        var rows = len(result.storage[0].hierarchy) // 2
        _set(model, "cluster_hierarchy_", _xc_array(packed.hierarchy, Python.tuple(rows, 2), "<i4", "i", context))
    _set(model, "n_features_in_", PythonObject(d))


# ---------------------------------------------------------------- AffinityPropagation

def _affinity_fit(model: PythonObject, binding: PythonObject, source: PythonObject,
                  context: PythonObject) raises:
    """`AffinityPropagation.fit(X)`."""
    var b = native_loaded_module("builtins")
    var affinity = _attr(model, "affinity")
    if not _one_of(affinity, "euclidean", "precomputed"):
        raise Error("affinity must be 'euclidean' or 'precomputed', got " + family_repr(affinity))
    var damping_field = _attr(model, "damping")
    var damping = Float64(py=_float(damping_field))
    if not (0.5 <= damping and damping < 1):
        raise Error("damping must be in [0.5, 1), got " + family_str(damping_field))
    if Int(py=_int(_attr(model, "max_iter"))) < 1 or Int(py=_int(_attr(model, "convergence_iter"))) < 1:
        raise Error("max_iter and convergence_iter must be >= 1")
    var x = _f32(source, "X", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var pre = _eq(affinity, PythonObject("precomputed"))
    if pre and n != d:
        raise Error("The matrix of similarities must be a square array. Got (" + String(n) + ", " + String(d)
                    + ") instead.")
    var preference_plan = AffinityPreferencePlan()
    var mode = 0
    var scalar = Float64(0.0)
    var preference = _attr(model, "preference")
    if preference is not None:
        if family_isinstance(preference, Python.tuple(b.int, b.float)):
            mode = 1
            scalar = Float64(py=_float(preference))
        else:
            preference_plan = affinity_preference_prepare(preference, context)
            if preference_plan.count != n:
                raise Error("preference must be a scalar or have one value per sample")
            mode = 2
    # ORDER MATCHES x_cluster/entries.mojo::affinity_entry: n, d,
    # precomputed, preference mode, max_iter, convergence_iter, seed;
    # damping, the scalar preference.
    var max_iter = _int(_attr(model, "max_iter"))
    var conv = _int(_attr(model, "convergence_iter"))
    var seed = _seed(_attr(model, "random_state"))
    var repeated_damping = _float(damping_field)
    var connection = centroid_fit_connection(binding)
    var request = AffinityFitRequest(variants_span(x), preference_plan.span(), n, d, pre, mode,
        Int(py=max_iter), Int(py=conv), seed, Float64(py=repeated_damping), scalar)
    var result = CentroidFitResultSlot()
    try:
        with GILReleased(Python()):
            connection.driver.affinity(request, result)
    finally:
        _ = connection.provider
        _ = x
        _ = preference_plan
    var packed = CentroidPackedResult(result, True)
    var k = len(result.storage[0].center_indices)
    _set(model, "cluster_centers_indices_", _xc_array(packed.center_indices, Python.tuple(k), "<i4", "i", context))
    _set(model, "labels_", _xc_array(packed.labels, Python.tuple(n), "<i4", "i", context))
    _set(model, "affinity_matrix_", _xc_array(packed.affinity, Python.tuple(n, n), "<f4", "f", context))
    # Observable final availability/responsibility diagonals stay public.
    _set(model, "message_diag_", _xc_array(packed.message_diagonal, Python.tuple(len(result.storage[0].message_diagonal)), "<f4", "f", context))
    _set(model, "n_iter_", PythonObject(result.storage[0].iterations))
    _set(model, "n_features_in_", PythonObject(d))
    if not pre:
        _set(model, "cluster_centers_", _xc_array(packed.centers, Python.tuple(k, d), "<f4", "f", context))


# ---------------------------------------------------------------- HDBSCAN
# HDBSCAN.fit is `hdbscan_fit` (bindings/hdbscan_fit_boundary_pack.mojo: UNPACK,
# typed RUN, PACK), called directly by HDBSCAN's lifecycle. The one code
# `family_cfa_hdbscan_fit` below stays for the parallel fit plan
# (bindings/parallel_plan_specs.mojo), which enters here.


# ---------------------------------------------------------------- door

def cluster_fit_a_unknown_operation(operation: String) raises:
    """A `family_cfa_*` name the operation table does not hold (the shells
    pass table names only): the text the name chain raised."""
    raise Error("RuntimeError: unknown family operation " + operation)


def cluster_fit_a_operation(model: PythonObject, operation: Int, args: PythonObject,
         context: PythonObject) raises -> PythonObject:
    """args[0] is the resolved binding, args[1] X, args[2] sample_weight
    where the method takes one. `operation` is the leaf code of
    bindings/operation_names_cfa.mojo (lane bw2-cluster; it was the name,
    matched by up to thirteen String compares per call)."""
    classical_connect_numpy(context)
    var binding = args[0]
    if operation == CFA_OPN_FAMILY_CFA_MBK_FIT:
        _mbk_fit(model, binding, args[1], args[2], context)
        return model
    if operation == CFA_OPN_FAMILY_CFA_MBK_PARTIAL_FIT:
        _mbk_partial_fit(model, binding, args[1], args[2], context)
        return model
    if operation == CFA_OPN_FAMILY_CFA_BISECT_FIT:
        _bisect_fit(model, binding, args[1], args[2], context)
        return model
    if operation == CFA_OPN_FAMILY_CFA_FIT_PREDICT:
        _centers_fit(model, binding, args[1], context)
        return _attr(model, "labels_")
    if operation == CFA_OPN_FAMILY_CFA_FIT_TRANSFORM:
        _centers_fit(model, binding, args[1], context)
        return cluster_operation(model, XC_OPN_FAMILY_XC_VARIANTS_TRANSFORM, Python.tuple(binding, args[1]), context)
    if operation == CFA_OPN_FAMILY_CFA_MEANSHIFT_FIT:
        meanshift_fit(model, binding, args[1], context)
        return model
    if operation == CFA_OPN_FAMILY_CFA_MEANSHIFT_FIT_PREDICT:
        meanshift_fit(model, binding, args[1], context)
        return _attr(model, "labels_")
    if operation == CFA_OPN_FAMILY_CFA_OPTICS_FIT:
        _optics_fit(model, binding, args[1], context)
        return model
    if operation == CFA_OPN_FAMILY_CFA_OPTICS_FIT_PREDICT:
        _optics_fit(model, binding, args[1], context)
        return _attr(model, "labels_")
    if operation == CFA_OPN_FAMILY_CFA_AP_FIT:
        _affinity_fit(model, binding, args[1], context)
        return model
    if operation == CFA_OPN_FAMILY_CFA_AP_FIT_PREDICT:
        _affinity_fit(model, binding, args[1], context)
        return _attr(model, "labels_")
    if operation == CFA_OPN_FAMILY_CFA_HDBSCAN_FIT:
        _ = hdbscan_fit(model, binding, args[1], context)
        return model
    raise Error("RuntimeError: unknown family operation code " + String(operation))
