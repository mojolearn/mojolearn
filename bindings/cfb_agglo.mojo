# SPDX-License-Identifier: Apache-2.0
# AgglomerativeClustering public operation owner: native lifecycle, admission,
# binding selection, preparation and finished results. Solver linkage and the
# general x_cluster route use native descriptors. Resident-reference numerical
# calls remain explicit debt before this family can be called complete.
from std.memory import memcpy
from std.os import getenv
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.classical_arrays import classical_address
from bindings.classical_validate import classical_view
from core.family_cast_impl import family_all_finite_f32
from core.cfb_impl import cfb_any_positive_i32
from bindings.family_common import (
    FAMILY_F32P, FAMILY_I32P, family_repr, family_str, family_call, family_entry, family_array, family_as_f32,
    family_field, family_is_str, family_f32_snapshot, family_view,
)
from bindings.mode_metadata_native import _fields
from bindings.metrics_native_support import metric_warn
from providers.python.array_metadata_pyapi import ArrayMetadata, array_metadata_once, array_shape_once
from providers.python.estimator_binding_pyapi import estimator_binding, estimator_binding_mode
from providers.python.host_estimator_binding_pyapi import host_estimator_binding_mode
from providers.python.neural_descriptor_pyapi import neural_descriptor_entry
from providers.python.x_neighbors_descriptor_pyapi import xn_admit, xn_execute
from providers.python.hierarchy_public_lifecycle_pyapi import hierarchy_public_begin, hierarchy_public_end, hierarchy_public_finish, hierarchy_public_refusal
from providers.python.hierarchy_connectivity_pyapi import HierarchyConnectivity, hierarchy_connectivity
from providers.python.hierarchy_cluster_descriptor_pyapi import hierarchy_cluster_admit, hierarchy_cluster_execute
from providers.python.hierarchy_resident_pyapi import hierarchy_fresh_holder, hierarchy_mint, hierarchy_reference_labels
from providers.python.hierarchy_solver_descriptor_pyapi import hierarchy_solver_admit, hierarchy_solver_linkage
from bindings.native_loaded_module import native_loaded_module
from bindings.family_raw_attr import family_raw_get, family_raw_get_or, family_raw_has

#: The dense connectivity matrix is `m * m` of cuVS's `int`.
comptime _PAIRWISE_MAX_ROWS = 46340


def _truth(value: PythonObject) raises -> Bool:
    """`bool(value)`, the user's own truth test."""
    return Bool(py=family_call(native_loaded_module("builtins").bool, value))


def _eq(a: PythonObject, b: PythonObject) raises -> Bool:
    """`a == b` as a truth value."""
    return _truth(family_call(native_loaded_module("operator").eq, a, b))


def _agglo_validate(settings: PythonObject, context: PythonObject) raises:
    """`AgglomerativeClustering._validate_params`: the normalized
    connectivity and compute_distances go to `_connectivity` and
    `_compute_distances`."""
    var b = native_loaded_module("builtins")
    var n_clusters = family_field(settings, "n_clusters")
    var metric = family_field(settings, "metric")
    var connectivity = family_field(settings, "connectivity")
    var linkage = family_field(settings, "linkage")
    var c = family_field(settings, "c")
    var memory = family_field(settings, "memory")
    var compute_full_tree = family_field(settings, "compute_full_tree")
    var distance_threshold = family_field(settings, "distance_threshold")
    var linkages = context["family_cfb_agglo_linkages"]
    if not py_contains(linkages, linkage):
        raise Error("linkage must be one of " + family_repr(py_call(b.sorted, linkages)) + ", got " + family_repr(linkage))
    if connectivity is None:
        # scikit-learn's spelling of "no connectivity constraint": the full
        # dense graph, cuML's 'pairwise'.
        connectivity = PythonObject("pairwise")
    if family_is_str(connectivity):
        if not py_contains(context["family_cfb_agglo_connectivities"], connectivity):
            raise Error("'connectivity' can only be one of {'knn', 'pairwise'}, "
                        + "a connectivity matrix or a callable")
        if String(connectivity) == "knn":
            raise Error("NotImplementedError: mojolearn AgglomerativeClustering: connectivity='knn' is "
                        + "REFUSED BY NAME. The Linkage::KNN_GRAPH specialization "
                        + "(connectivities.cuh:49), the cross-component fix-up its "
                        + "forest MST needs (connect_knn_graph, mst.cuh:67 and :131) "
                        + "and merge_msts are rung 2 and NOT IMPLEMENTED -- and their host "
                        + "overload picks a RANDOM vertex per component from "
                        + "std::mt19937(std::random_device()), which would have to be "
                        + "pinned first. Use connectivity='pairwise' (cuML's C++ "
                        + "default and scikit-learn's dense tree), or pass the k-NN "
                        + "graph itself as a connectivity matrix (scikit-learn's "
                        + "kneighbors_graph). See hierarchy/NOT_IMPLEMENTED.tsv")
    var all_metrics = context["family_cfb_agglo_all_metrics"]
    if not py_contains(all_metrics, metric):
        raise Error("NotImplementedError: mojolearn AgglomerativeClustering: metric=" + family_repr(metric) + " is "
                    + "refused by name; one of " + family_repr(py_call(b.sorted, all_metrics)) + ". A callable "
                    + "metric would compute the dissimilarities outside every "
                    + "identity column. See x_cluster/NOT_IMPLEMENTED.tsv")
    if _eq(linkage, PythonObject("ward")) and not py_contains(Python.tuple("euclidean", "l2"), metric):
        raise Error(family_str(metric) + " was provided as metric. Ward can only work with "
                    + "euclidean distances (i.e. 'euclidean' and 'l2').")
    if _truth(family_call(native_loaded_module("operator").ne, c, PythonObject(15))):
        raise Error("NotImplementedError: mojolearn AgglomerativeClustering: c=" + family_repr(c) + " is refused. It "
                    + "tunes k = log(n) + c for the k-NN graph, cuML forwards it "
                    + "only when connectivity='knn' (linkage.cu:40, "
                    + "'use_knn ? c : 0'), and that arm is refused by name here, "
                    + "so any value you pass would reach nothing")
    if memory is not None:
        raise Error("NotImplementedError: mojolearn AgglomerativeClustering: memory is refused; it is "
                    + "scikit-learn's joblib cache for a host tree build and there "
                    + "is no host tree here")
    if not py_contains(Python.tuple(PythonObject("auto"), PythonObject(True), PythonObject(False)), compute_full_tree):
        raise Error("compute_full_tree must be 'auto', True or False, got " + family_repr(compute_full_tree))
    if not ((n_clusters is None) != (distance_threshold is None)):
        raise Error("Exactly one of n_clusters and distance_threshold has to be "
                    + "set, and the other needs to be None.")
    if distance_threshold is not None and not _truth(compute_full_tree):
        raise Error("compute_full_tree must be True if distance_threshold is set.")
    py_setitem(settings, "_connectivity", connectivity)
    py_setitem(settings, "_compute_distances", PythonObject(_truth(family_field(settings, "compute_distances"))))


def _fast_mst(n_cols: Int) raises -> Bool:
    """`_backend.requested_mode() == "fast" and sys.platform == "darwin" and
    n_cols <= 64`: FAST on Apple builds the Euclidean MST from Boruvka rounds
    without the dense graph."""
    var mode = String(getenv("MOJOLEARN_NUMERIC_MODE", "identical").strip()).lower()
    if mode != "fast" and mode != "identical":
        raise Error("ImportError: mojolearn: MOJOLEARN_NUMERIC_MODE=" + family_repr(PythonObject(mode))
                    + "; it must be 'fast', or 'identical' (the default)")
    return mode == "fast" and String(py=native_loaded_module("sys").platform) == "darwin" and n_cols <= 64


def _agglo_legacy(fields: PythonObject, settings: PythonObject, args: PythonObject, x_snapshot: ArrayMetadata, context: PythonObject) raises:
    """cuML's single-linkage Boruvka route (`_mojolearn_solver.linkage_fit`)."""
    var b = native_loaded_module("builtins")
    var x = x_snapshot.keep
    var n_rows = x_snapshot.shape[0]
    var n_cols = x_snapshot.shape[1]
    if n_rows < 2:
        # scikit-learn's check_array wording (ensure_min_samples=2)
        raise Error("Found array with " + String(n_rows) + " sample(s) (shape=(" + String(n_rows) + ", "
                    + String(n_cols) + ")) while a minimum of 2 is required.")
    if n_rows > _PAIRWISE_MAX_ROWS and not _fast_mst(n_cols):
        raise Error("mojolearn AgglomerativeClustering: n_rows=" + String(n_rows) + " > "
                    + String(_PAIRWISE_MAX_ROWS) + "; the dense connectivity matrix is "
                    + "m * m of cuVS's int and overflows past that "
                    + "(hierarchy/impl/cluster/detail/connectivities.mojo "
                    + "refuses it by name too)")
    var k = Int(py=family_call(b.int, family_field(settings, "n_clusters")))
    if k < 1 or k > n_rows:
        raise Error("Expected 1 <= n_clusters <= n_rows (" + String(n_rows) + "), got "
                    + "n_clusters=" + String(k))
    var children = family_array(Python.tuple(n_rows - 1, 2), "<i4", context)
    var labels = family_array(Python.tuple(n_rows), "<i4", context)
    var info = family_array(Python.tuple(2), "<i4", context)
    var metric = Int(py=context["family_agglo_predict_metrics"][family_field(settings, "metric")])
    var use_knn = Int(py=context["family_cfb_agglo_connectivities"][family_field(settings, "_connectivity")])
    var solver = hierarchy_solver_admit(args[1])
    var children_address = Int(py=classical_address(children))
    var labels_address = Int(py=classical_address(labels))
    var info_address = Int(py=classical_address(info))
    # Existing linkage ABI, now direct native descriptor dispatch. Input metadata
    # is the retained operation snapshot; these outputs keep their owners below.
    _ = hierarchy_solver_linkage(solver,
        [x_snapshot.source.addr, children_address, labels_address, info_address],
        [n_rows, n_cols, k, metric, use_knn])
    _ = solver^
    _ = labels  # keep the Python object alive through the native call above
    _ = children  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _ = info  # keep the Python object alive through the native call above
    var words = FAMILY_I32P(unsafe_from_address=info_address)
    _ = info  # keep the Python object alive through the native call above
    py_setitem(fields, "labels_", labels)
    py_setitem(fields, "children_", children)
    py_setitem(fields, "n_clusters_", PythonObject(k))
    py_setitem(fields, "n_leaves_", PythonObject(n_rows))
    # -1 on a CPU-only install: the host binding runs Kruskal.
    py_setitem(fields, "n_boruvka_rounds_", PythonObject(Int(words[0])))
    _ = info  # keep the Python object alive through the native call above
    py_setitem(fields, "n_connected_components_", PythonObject(Int(words[1])))
    _ = info  # keep the Python object alive through the native call above
    py_setitem(fields, "n_features_in_", PythonObject(n_cols))
    var predicting = Bool(py=family_field(settings, "prediction_data"))
    var fit_x = _fit_copy(x_snapshot, context) if predicting else PythonObject(None)
    py_setitem(fields, "_fit_X", fit_x)
    _ = x^
    hierarchy_fresh_holder(fields)
    if predicting:
        hierarchy_mint(fields, args[0], fit_x, labels, Int(py=classical_address(fit_x)), labels_address, n_rows, n_cols)


def _fit_copy(snapshot: ArrayMetadata, context: PythonObject) raises -> PythonObject:
    # A cast-created buffer is owned already. Caller-owned float32 Arrays and
    # ndarrays still copy, so writes after fit cannot move prediction references.
    if snapshot.copied:
        return snapshot.keep
    var result = family_array(snapshot.python_shape(), "<f4", context)
    memcpy(dest=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(result))),
           src=FAMILY_F32P(unsafe_from_address=snapshot.source.addr), count=snapshot.count)
    return result


def _agglo_x(fields: PythonObject, settings: PythonObject, args: PythonObject, x_snapshot: ArrayMetadata, context: PythonObject) raises:
    """The x_cluster route (`x_cluster/agglo.mojo`, ENTRY_AGGLO): every
    linkage, metric, connectivity matrix and the per-merge distances, with
    scikit-learn's `_fit` bookkeeping around it."""
    var b = native_loaded_module("builtins")
    var x = x_snapshot.keep
    var n = x_snapshot.shape[0]
    var d = x_snapshot.shape[1]
    if n < 2:
        raise Error("mojolearn AgglomerativeClustering: n_rows=" + String(n) + " < 2; a tree "
                    + "needs at least two points")
    var metric = Int(py=context["family_cfb_agglo_all_metrics"][family_field(settings, "metric")])
    if metric == 5 and n != d:
        raise Error("Distance matrix should be square, got matrix of shape (" + String(n) + ", " + String(d) + ")")
    if metric == 4:
        # the zero-row test on the device (the x_neighbors binding's
        # `xn_row_all_zero`, one flag per row from the bit patterns).
        if d == 0:
            raise Error("Cosine affinity cannot be used when X contains zero vectors")
        var zrow = family_array(Python.tuple(n), "<i4", context)
        var zrow_address = Int(py=classical_address(zrow))
        xn_execute(args[3], "row_all_zero", Python.list(PythonObject(x_snapshot.source.addr), PythonObject(zrow_address)),
                   Python.list(PythonObject(n), PythonObject(d)), Python.list())
        _ = zrow  # keep the Python object alive through the native call above
        _ = x  # keep the Python object alive through the native call above
        if cfb_any_positive_i32(FAMILY_I32P(unsafe_from_address=zrow_address), n):
            raise Error("Cosine affinity cannot be used when X contains zero vectors")
        _ = zrow  # keep the Python object alive through the native calls above
    var n_clusters = family_field(settings, "n_clusters")
    var requested_clusters = 0
    if n_clusters is not None:
        requested_clusters = Int(py=family_call(b.int, n_clusters))
        if requested_clusters < 1 or requested_clusters > n:
            raise Error("Cannot extract more clusters than samples: " + String(requested_clusters) + " clusters "
                        + "were given for a tree with " + String(n) + " leaves.")
    var connectivity = family_field(settings, "_connectivity")
    var thr = family_field(settings, "distance_threshold")
    var constrained = not family_is_str(connectivity)
    var full = True
    if constrained:
        var tree = family_field(settings, "compute_full_tree")
        if family_is_str(tree) and String(tree) == "auto":
            # `int(n_clusters) < max(100, 0.02 * n)`
            var bound = 0.02 * Float64(n)
            if bound < 100.0:
                bound = 100.0
            full = thr is not None or Float64(requested_clusters) < bound
        else:
            full = _truth(tree)
    var n_merges = n - 1 if full else n - requested_clusters
    var aux = HierarchyConnectivity(PythonObject(None), PythonObject(None), 0, 0, -1, 0)
    if constrained:
        aux = hierarchy_connectivity(connectivity, n, context)
    var cut = 0
    if full:
        cut = 1 if thr is not None else requested_clusters
    var ip: List[Int] = [x_snapshot.count, aux.count, n, d,
                         Int(py=context["family_cfb_agglo_linkages"][family_field(settings, "linkage")]),
                         metric, aux.edges, n_merges, aux.mode, cut,
                         1 if (full and thr is not None) else 0]
    var fp: List[Float64] = [2.0, Float64(py=family_call(b.float, thr)) if thr is not None else 0.0]
    var provider = hierarchy_cluster_admit(args[2])
    # Capture each fresh output address once. The descriptor copies the typed
    # ClusterOut slots directly; no array.array/result-list extension ABI.
    var children = family_array(Python.tuple(n_merges, 2), "<i4", context)
    var labels = family_array(Python.tuple(n), "<i4", context)
    var distances = family_array(Python.tuple(n_merges), "<f4", context)
    var children_address = Int(py=classical_address(children))
    var labels_address = Int(py=classical_address(labels))
    var distances_address = Int(py=classical_address(distances))
    var scalars = List[Float64](length=2, fill=Float64(0))
    hierarchy_cluster_execute(provider,
        [x_snapshot.source.addr, aux.address, children_address, labels_address,
         distances_address, Int(scalars.unsafe_ptr())], ip, fp)
    _ = provider^
    _ = x_snapshot
    _ = aux^
    py_setitem(fields, "children_", children)
    py_setitem(fields, "n_leaves_", PythonObject(n))
    var components = Int(scalars[0])
    py_setitem(fields, "n_connected_components_", PythonObject(components))
    if constrained and components > 1:
        metric_warn("the number of connected components of the connectivity matrix is " + String(components)
                    + " > 1. Completing it to avoid stopping the tree early.", b.UserWarning, context)
    if thr is not None or Bool(py=family_field(settings, "_compute_distances")):
        py_setitem(fields, "distances_", distances)
    else:
        _ = py_call(fields.pop, PythonObject("distances_"), PythonObject(None))
    var clusters = PythonObject(None)
    if thr is not None and full:
        clusters = PythonObject(Int(scalars[1]))
    else:
        clusters = PythonObject(requested_clusters)
    _ = scalars^
    _ = children^
    _ = distances^
    py_setitem(fields, "n_clusters_", clusters)
    py_setitem(fields, "labels_", labels)
    py_setitem(fields, "n_boruvka_rounds_", PythonObject(-1))
    py_setitem(fields, "n_features_in_", PythonObject(d))
    var fit_x = PythonObject(None)
    var predicting = Bool(py=family_field(settings, "prediction_data")) and _predicts(settings, context)
    if predicting:
        fit_x = _fit_copy(x_snapshot, context)
    py_setitem(fields, "_fit_X", fit_x)
    _ = x^
    hierarchy_fresh_holder(fields)
    if predicting:
        hierarchy_mint(fields, args[0], fit_x, labels, Int(py=classical_address(fit_x)), labels_address, n, d)
    _ = labels^


def _predicts(settings: PythonObject, context: PythonObject) raises -> Bool:
    """`_predicts`: single linkage under a euclidean metric name."""
    return (_eq(family_field(settings, "linkage"), PythonObject("single"))
            and py_contains(context["family_agglo_predict_metrics"], family_field(settings, "metric")))


def _agglo_fit(fields: PythonObject, settings: PythonObject, args: PythonObject, x: ArrayMetadata, context: PythonObject) raises:
    """`AgglomerativeClustering.fit(X)`."""
    var b = native_loaded_module("builtins")
    _agglo_validate(settings, context)
    py_setitem(fields, "_connectivity", settings["_connectivity"])
    py_setitem(fields, "_compute_distances", settings["_compute_distances"])
    var prediction_data = family_field(settings, "prediction_data")
    if py_type(prediction_data) is not b.bool:
        raise Error("TypeError: mojolearn AgglomerativeClustering: prediction_data must be a "
                    + "bool, got " + String(py_type(prediction_data).__name__))
    # `_legacy_route`: cuML's single-linkage Boruvka route keeps the recorded
    # `agglomerative` lanes' bits; everything else is the x_cluster route.
    var legacy = (_predicts(settings, context) and family_is_str(family_field(settings, "_connectivity"))
                  and family_field(settings, "distance_threshold") is None
                  and not Bool(py=family_field(settings, "_compute_distances")))
    if legacy:
        _agglo_legacy(fields, settings, args, x, context)
    else:
        _agglo_x(fields, settings, args, x, context)


def _agglo_predict(fields: PythonObject, settings: PythonObject, binding: PythonObject,
                   query: ArrayMetadata, context: PythonObject) raises -> PythonObject:
    if not _predicts(settings, context):
        raise Error("NotImplementedError: mojolearn AgglomerativeClustering.predict: only single linkage under the euclidean metric has a predict rule (DEVIATION 2740)")
    var refs_source = py_get(fields, "_fit_X")
    if refs_source is None:
        raise Error("mojolearn AgglomerativeClustering.predict: prediction data was not stored. Fit with AgglomerativeClustering(prediction_data=True)")
    var refs = family_f32_snapshot(refs_source, 2, "fitted reference rows", context)
    if refs.shape[1] != query.shape[1]:
        raise Error("mojolearn AgglomerativeClustering.predict: fitted reference width differs from X")
    var labels = classical_view(fields["labels_"], context, "labels_")
    var labels_shape = array_shape_once(labels, context)
    if len(labels_shape) != 1:
        raise Error("TypeError: hierarchy fitted labels must be one-dimensional")
    var labels_snapshot = array_metadata_once(labels, labels_shape^, "labels_", context)
    if labels_snapshot.source.kind != 1 or labels_snapshot.source.itemsize != 4 or not labels_snapshot.c_order or labels_snapshot.count != refs.shape[0]:
        raise Error("TypeError: hierarchy fitted labels require one native contiguous int32 value per reference row")
    var result = hierarchy_reference_labels(fields, binding, refs.keep, labels, refs.source.addr, labels_snapshot.source.addr,
                                           query.source.addr, refs.shape[0], query.shape[0], query.shape[1], context)
    _ = refs^
    _ = labels_snapshot^
    return result


def cfb_agglo_operation(model: PythonObject, operation: String, args: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    # Public ABI: (None, X, y) for fit/fit_predict, (None, X) for predict.
    # Pickle operations transport plain admitted instance dictionaries only.
    if operation == "family_cfb_agglo_getstate":
        var fields = _fields(model)
        var state = py_call(native_loaded_module("builtins").dict, fields)
        _ = py_call(state.pop, "_resident_holder", PythonObject(None))
        return state
    if operation == "family_cfb_agglo_setstate":
        var b = native_loaded_module("builtins")
        var state = args[0]
        if py_type(state) is not b.dict:
            raise Error("TypeError: hierarchy pickle state requires a native dictionary")
        var names = py_call(b.list, state.keys())
        for i in range(len(names)):  # driver-loop: exact restored field names before dictionary update
            if py_type(names[i]) is not b.str:
                raise Error("TypeError: hierarchy pickle field names must be native strings")
        var fields = _fields(model)
        _ = py_call(fields.update, state)
        py_setitem(fields, "_resident_holder", Python.dict())
        return PythonObject(None)
    if operation != "family_cfb_agglo_fit" and operation != "family_cfb_agglo_fit_predict" and operation != "family_cfb_agglo_predict":
        raise Error("RuntimeError: unknown AgglomerativeClustering operation " + operation)
    # Parallel graph plans retain their original explicit four-module ABI.
    # Admit the same captured operation without reselecting its pinned modules.
    var explicit = len(args) == 5 and args[0] is not None
    var public_args = Python.tuple(PythonObject(None), args[4], PythonObject(None)) if explicit else args
    var call = hierarchy_public_begin(model, operation, public_args, context)
    var result = PythonObject(None)
    try:
        var estimators = args[0] if args[0] is not None else host_estimator_binding_mode(model, call.settings["numeric_mode"], "_mojolearn_estimators", context)
        if call.fitting:
            var solver = args[1] if explicit else estimator_binding(PythonObject(None), "_mojolearn_solver", context)
            var cluster = args[2] if explicit else estimator_binding_mode(call.settings["numeric_mode"], "_mojolearn_x_cluster", context)
            var neighbors = xn_admit(args[3] if explicit else estimator_binding_mode(call.settings["numeric_mode"], "_mojolearn_x_neighbors", context))
            py_setitem(call.fields, "input_copied_", PythonObject(not (call.x.keep is call.source)))
            _agglo_fit(call.fields, call.settings, Python.tuple(estimators, solver, cluster, neighbors), call.x, context)
            result = call.fields["labels_"] if operation == "family_cfb_agglo_fit_predict" else model
        else:
            result = _agglo_predict(call.fields, call.settings, estimators, call.x, context)
    except error:
        var refused = hierarchy_public_refusal(error, call, model)
        raise refused^
    finally:
        hierarchy_public_end(model, call)
    return hierarchy_public_finish(model, result, call, context)
