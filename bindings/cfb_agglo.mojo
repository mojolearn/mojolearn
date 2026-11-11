# SPDX-License-Identifier: Apache-2.0
# AgglomerativeClustering public operation owner: native lifecycle, admission,
# binding selection, preparation and finished results. Solver linkage and the
# general x_cluster route and shared resident references use typed native
# descriptors. Public storage/image owners remain explicit boundary pins.
from std.memory import memcpy
from std.os import getenv
from std.python import Python, PythonObject
from bindings.admitted_family import admit_family_meta
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.classical_arrays import classical_address
from bindings.classical_validate import classical_view
from bindings.family_common import (
    FAMILY_F32P, family_repr, family_str, family_call,
    family_is_str,
)
from bindings.mode_metadata_native import _fields
from bindings.metrics_native_support import metric_warn
from providers.python.array_metadata_pyapi import ArrayMetadata, array_metadata_once, array_shape_once
from providers.python.estimator_binding_pyapi import estimator_binding, estimator_binding_mode
from providers.python.host_estimator_binding_pyapi import host_estimator_binding_mode
from providers.python.x_neighbors_descriptor_pyapi import xn_admit
from providers.python.hierarchy_public_lifecycle_pyapi import hierarchy_public_begin, hierarchy_public_end, hierarchy_public_finish, hierarchy_public_refusal
from providers.python.hierarchy_connectivity_pyapi import HierarchyConnectivity, hierarchy_connectivity
from providers.python.hierarchy_settings_pyapi import AggloSettings
from bindings.agglo_native_boundary import (
    AggloBindings, AggloSolverConnection, AggloClusterConnection, AggloZeroRowsConnection,
    agglo_solver_run, agglo_cluster_run, agglo_zero_rows_run, agglo_array, agglo_publish, agglo_scalar_repr,
    agglo_hash_admitted, agglo_integer_admitted, agglo_float_admitted,
    agglo_fresh_holder, agglo_mint, agglo_reference_labels, agglo_holder_empty, agglo_release, agglo_holder_admit,
)
from core.agglo_native_api import (
    AggloSolverRequest, AggloClusterRequest, AggloZeroRowsRequest,
    AggloSolverScratch, AggloClusterScratch, AggloZeroRowsScratch,
)
from core.agglo_native_state import (
    AggloOptions, AGGLO_PAIRWISE_MAX_ROWS, AGGLO_FULL_AUTO, AGGLO_FULL_TRUE, AGGLO_FULL_FALSE,
    agglo_metric_code, agglo_linkage_code, agglo_connectivity_code, agglo_predicts, agglo_tree_plan,
)
from bindings.decomp_scalar_compare import decomp_equal_admitted
from bindings.public_lifecycle_owner import lifecycle_admit_fields
from bindings.native_owner_lifetime import native_release_admit
from bindings.native_loaded_module import native_loaded_module

#: The dense connectivity matrix is `m * m` of cuVS's `int`.
comptime _PAIRWISE_MAX_ROWS = AGGLO_PAIRWISE_MAX_ROWS


def _truth(value: PythonObject) raises -> Bool:
    """Truth conversion of the already admitted exact public scalar."""
    return Bool(py=value)


def _eq(a: PythonObject, b: PythonObject) raises -> Bool:
    """`a == b` as a truth value."""
    return decomp_equal_admitted(a, b)


def _agglo_validate(settings: AggloSettings, context: PythonObject) raises -> AggloOptions:
    """`AgglomerativeClustering._validate_params`: the normalized
    connectivity and compute_distances go to `_connectivity` and
    `_compute_distances`."""
    var b = native_loaded_module("builtins")
    var n_clusters = settings.n_clusters
    var metric = settings.metric
    var connectivity = settings.connectivity
    var linkage = settings.linkage
    var c = settings.c
    var memory = settings.memory
    var compute_full_tree = settings.compute_full_tree
    var distance_threshold = settings.distance_threshold
    agglo_hash_admitted(linkage)
    var linkage_code = agglo_linkage_code(String(linkage)) if family_is_str(linkage) else -1
    if linkage_code < 0:
        raise Error("linkage must be one of ['average', 'complete', 'single', 'ward'], got " + agglo_scalar_repr(linkage, context))
    if connectivity is None:
        # scikit-learn's spelling of "no connectivity constraint": the full
        # dense graph, cuML's 'pairwise'.
        connectivity = PythonObject("pairwise")
    if family_is_str(connectivity):
        if agglo_connectivity_code(String(connectivity)) < 0:
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
    agglo_hash_admitted(metric)
    var metric_code = agglo_metric_code(String(metric)) if family_is_str(metric) else -2
    if metric_code == -2:
        raise Error("NotImplementedError: mojolearn AgglomerativeClustering: metric=" + agglo_scalar_repr(metric, context) + " is "
                    + "refused by name; one of ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan', 'precomputed']. A callable "
                    + "metric would compute the dissimilarities outside every "
                    + "identity column. See x_cluster/NOT_IMPLEMENTED.tsv")
    if linkage_code == 0 and metric_code != -1:
        raise Error(family_str(metric) + " was provided as metric. Ward can only work with "
                    + "euclidean distances (i.e. 'euclidean' and 'l2').")
    if not _eq(c, PythonObject(15)):
        raise Error("NotImplementedError: mojolearn AgglomerativeClustering: c=" + agglo_scalar_repr(c, context) + " is refused. It "
                    + "tunes k = log(n) + c for the k-NN graph, cuML forwards it "
                    + "only when connectivity='knn' (linkage.cu:40, "
                    + "'use_knn ? c : 0'), and that arm is refused by name here, "
                    + "so any value you pass would reach nothing")
    if memory is not None:
        raise Error("NotImplementedError: mojolearn AgglomerativeClustering: memory is refused; it is "
                    + "scikit-learn's joblib cache for a host tree build and there "
                    + "is no host tree here")
    if not (_eq(compute_full_tree, PythonObject("auto")) or _eq(compute_full_tree, PythonObject(True)) or _eq(compute_full_tree, PythonObject(False))):
        raise Error("compute_full_tree must be 'auto', True or False, got " + agglo_scalar_repr(compute_full_tree, context))
    if not ((n_clusters is None) != (distance_threshold is None)):
        raise Error("Exactly one of n_clusters and distance_threshold has to be "
                    + "set, and the other needs to be None.")
    if distance_threshold is not None and not _truth(compute_full_tree):
        raise Error("compute_full_tree must be True if distance_threshold is set.")
    var distances = _truth(settings.compute_distances)
    var tree_mode = AGGLO_FULL_AUTO if family_is_str(compute_full_tree) and String(compute_full_tree) == "auto" else (AGGLO_FULL_TRUE if _truth(compute_full_tree) else AGGLO_FULL_FALSE)
    return AggloOptions(linkage_code, metric_code, not family_is_str(connectivity),
        distance_threshold is not None, distances, False, tree_mode)


def _fast_mst(n_cols: Int) raises -> Bool:
    """`_backend.requested_mode() == "fast" and sys.platform == "darwin" and
    n_cols <= 64`: FAST on Apple builds the Euclidean MST from Boruvka rounds
    without the dense graph."""
    var mode = String(getenv("MOJOLEARN_NUMERIC_MODE", "identical").strip()).lower()
    if mode != "fast" and mode != "identical":
        raise Error("ImportError: mojolearn: MOJOLEARN_NUMERIC_MODE=" + family_repr(PythonObject(mode))
                    + "; it must be 'fast', or 'identical' (the default)")
    return mode == "fast" and String(py=native_loaded_module("sys").platform) == "darwin" and n_cols <= 64


def _agglo_legacy(fields: PythonObject, settings: AggloSettings, options: AggloOptions, bindings: AggloBindings, x_snapshot: ArrayMetadata, context: PythonObject) raises:
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
    var k = agglo_integer_admitted(settings.n_clusters)
    if k < 1 or k > n_rows:
        raise Error("Expected 1 <= n_clusters <= n_rows (" + String(n_rows) + "), got "
                    + "n_clusters=" + String(k))
    var children_shape: List[Int] = [n_rows - 1, 2]
    var children = agglo_array(children_shape, "<i4", context)
    var labels = agglo_array(List[Int](length=1, fill=n_rows), "<i4", context)
    var info = AggloSolverScratch()
    var solver = AggloSolverConnection(bindings.solver)
    var children_address = Int(py=classical_address(children))
    var labels_address = Int(py=classical_address(labels))
    # The solver ABI uses cuML's sqrt-Euclidean code 1; the reference ABI
    # uses code 0. Keep that existing distinction explicit in the request.
    var request = AggloSolverRequest(x_snapshot.source.addr, children_address, labels_address,
        info.address(), n_rows, n_cols, k, 1, False)
    agglo_solver_run(solver, request)
    _ = solver^
    _ = labels
    _ = children
    _ = x
    agglo_publish(fields, "labels_", labels, context)
    agglo_publish(fields, "children_", children, context)
    agglo_publish(fields, "n_clusters_", PythonObject(k), context)
    agglo_publish(fields, "n_leaves_", PythonObject(n_rows), context)
    # -1 on a CPU-only install: preserve the host numerical owner's marker.
    agglo_publish(fields, "n_boruvka_rounds_", PythonObject(info.rounds()), context)
    _ = info  # native scratch remains live through both result reads
    agglo_publish(fields, "n_connected_components_", PythonObject(info.components()), context)
    _ = info  # native scratch remains live through both result reads
    agglo_publish(fields, "n_features_in_", PythonObject(n_cols), context)
    var predicting = options.prediction_data
    var fit_x = _fit_copy(x_snapshot, context) if predicting else PythonObject(None)
    agglo_publish(fields, "_fit_X", fit_x, context)
    _ = x^
    agglo_fresh_holder(fields)
    if predicting:
        agglo_mint(fields, bindings.estimators, fit_x, labels, Int(py=classical_address(fit_x)), labels_address, n_rows, n_cols, context)


def _fit_copy(snapshot: ArrayMetadata, context: PythonObject) raises -> PythonObject:
    # A cast-created buffer is owned already. Caller-owned float32 Arrays and
    # ndarrays still copy, so writes after fit cannot move prediction references.
    if snapshot.copied:
        return snapshot.keep
    var result = agglo_array(snapshot.shape, "<f4", context)
    memcpy(dest=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(result))),
           src=FAMILY_F32P(unsafe_from_address=snapshot.source.addr), count=snapshot.count)
    return result


def _agglo_x(fields: PythonObject, settings: AggloSettings, options: AggloOptions, bindings: AggloBindings, x_snapshot: ArrayMetadata, context: PythonObject) raises:
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
    var metric = options.metric
    if metric == 5 and n != d:
        raise Error("Distance matrix should be square, got matrix of shape (" + String(n) + ", " + String(d) + ")")
    if metric == 4:
        # the zero-row test on the device (the x_neighbors binding's
        # `xn_row_all_zero`, one flag per row from the bit patterns).
        if d == 0:
            raise Error("Cosine affinity cannot be used when X contains zero vectors")
        var zrow = AggloZeroRowsScratch(n)
        var zero_connection = AggloZeroRowsConnection(bindings.neighbors)
        var zero_request = AggloZeroRowsRequest(x_snapshot.source.addr, zrow.address(), n, d)
        agglo_zero_rows_run(zero_connection, zero_request)
        _ = zero_connection^
        _ = x
        if zrow.any_zero():
            raise Error("Cosine affinity cannot be used when X contains zero vectors")
        _ = zrow^
    var n_clusters = settings.n_clusters
    var requested_clusters = 0
    if n_clusters is not None:
        requested_clusters = agglo_integer_admitted(n_clusters)
        if requested_clusters < 1 or requested_clusters > n:
            raise Error("Cannot extract more clusters than samples: " + String(requested_clusters) + " clusters "
                        + "were given for a tree with " + String(n) + " leaves.")
    var connectivity = settings.connectivity
    var thr = settings.distance_threshold
    var constrained = options.constrained
    var plan = agglo_tree_plan(options, n, requested_clusters)
    var n_merges = plan.merges
    var aux = HierarchyConnectivity(PythonObject(None), PythonObject(None), 0, 0, -1, 0)
    if constrained:
        aux = hierarchy_connectivity(connectivity, n, context)
    var threshold = agglo_float_admitted(thr) if options.has_threshold else Float64(0)
    var provider = AggloClusterConnection(bindings.cluster)
    # Capture each fresh output address once. The descriptor copies the typed
    # ClusterOut slots directly; no array.array/result-list extension ABI.
    var children_shape: List[Int] = [n_merges, 2]
    var children = agglo_array(children_shape, "<i4", context)
    var labels = agglo_array(List[Int](length=1, fill=n), "<i4", context)
    var distances = agglo_array(List[Int](length=1, fill=n_merges), "<f4", context)
    var children_address = Int(py=classical_address(children))
    var labels_address = Int(py=classical_address(labels))
    var distances_address = Int(py=classical_address(distances))
    var scalars = AggloClusterScratch()
    var request = AggloClusterRequest(x_snapshot.source.addr, aux.address, children_address,
        labels_address, distances_address, scalars.address(), x_snapshot.count, aux.count,
        n, d, options.linkage, metric, aux.edges, plan.merges, aux.mode, plan.cut,
        plan.use_threshold, 2.0, threshold)
    agglo_cluster_run(provider, request)
    _ = provider^
    _ = x_snapshot
    _ = aux^
    agglo_publish(fields, "children_", children, context)
    agglo_publish(fields, "n_leaves_", PythonObject(n), context)
    var components = scalars.components()
    agglo_publish(fields, "n_connected_components_", PythonObject(components), context)
    if constrained and components > 1:
        metric_warn("the number of connected components of the connectivity matrix is " + String(components)
                    + " > 1. Completing it to avoid stopping the tree early.", b.UserWarning, context)
    if options.has_threshold or options.compute_distances:
        agglo_publish(fields, "distances_", distances, context)
    else:
        lifecycle_admit_fields(fields)
        native_release_admit(py_get(fields, "distances_"), context)
        _ = py_call(fields.pop, PythonObject("distances_"), PythonObject(None))
    var clusters = PythonObject(None)
    if options.has_threshold and plan.full:
        clusters = PythonObject(scalars.clusters())
    else:
        clusters = PythonObject(requested_clusters)
    _ = scalars^
    _ = children^
    _ = distances^
    agglo_publish(fields, "n_clusters_", clusters, context)
    agglo_publish(fields, "labels_", labels, context)
    agglo_publish(fields, "n_boruvka_rounds_", PythonObject(-1), context)
    agglo_publish(fields, "n_features_in_", PythonObject(d), context)
    var fit_x = PythonObject(None)
    var predicting = options.prediction_data and agglo_predicts(options.linkage, options.metric)
    if predicting:
        fit_x = _fit_copy(x_snapshot, context)
    agglo_publish(fields, "_fit_X", fit_x, context)
    _ = x^
    agglo_fresh_holder(fields)
    if predicting:
        agglo_mint(fields, bindings.estimators, fit_x, labels, Int(py=classical_address(fit_x)), labels_address, n, d, context)
    _ = labels^


def _predicts(settings: AggloSettings, context: PythonObject) raises -> Bool:
    if not _eq(settings.linkage, PythonObject("single")):
        return False
    agglo_hash_admitted(settings.metric)
    var metric = agglo_metric_code(String(settings.metric)) if family_is_str(settings.metric) else -2
    return agglo_predicts(3, metric)


def _agglo_fit(fields: PythonObject, settings: AggloSettings, bindings: AggloBindings, x: ArrayMetadata, context: PythonObject) raises:
    """`AgglomerativeClustering.fit(X)`."""
    var b = native_loaded_module("builtins")
    var options = _agglo_validate(settings, context)
    var connectivity = PythonObject("pairwise") if settings.connectivity is None else settings.connectivity
    agglo_publish(fields, "_connectivity", connectivity, context)
    agglo_publish(fields, "_compute_distances", PythonObject(options.compute_distances), context)
    var prediction_data = settings.prediction_data
    if py_type(prediction_data) is not b.bool:
        raise Error("TypeError: mojolearn AgglomerativeClustering: prediction_data must be a "
                    + "bool, got " + String(py_call(b.type.__dict__["__name__"].__get__, py_type(prediction_data))))
    options.prediction_data = Bool(py=prediction_data)
    # Keep both existing numerical owners and their unchanged route predicate.
    if options.solver_route():
        _agglo_legacy(fields, settings, options, bindings, x, context)
    else:
        _agglo_x(fields, settings, options, bindings, x, context)


def _agglo_predict(fields: PythonObject, settings: AggloSettings, binding: PythonObject,
                   query: ArrayMetadata, context: PythonObject) raises -> PythonObject:
    if not _predicts(settings, context):
        raise Error("NotImplementedError: mojolearn AgglomerativeClustering.predict: only single linkage under the euclidean metric has a predict rule (DEVIATION 2740)")
    var refs_source = py_get(fields, "_fit_X")
    if refs_source is None:
        raise Error("mojolearn AgglomerativeClustering.predict: prediction data was not stored. Fit with AgglomerativeClustering(prediction_data=True)")
    var refs = admit_family_meta(refs_source, 2, "fitted reference rows", context)
    if refs.shape[1] != query.shape[1]:
        raise Error("mojolearn AgglomerativeClustering.predict: fitted reference width differs from X")
    var labels = classical_view(fields["labels_"], context, "labels_")
    var labels_shape = array_shape_once(labels, context)
    if len(labels_shape) != 1:
        raise Error("TypeError: hierarchy fitted labels must be one-dimensional")
    var labels_snapshot = array_metadata_once(labels, labels_shape^, "labels_", context)
    if labels_snapshot.source.kind != 1 or labels_snapshot.source.itemsize != 4 or not labels_snapshot.c_order or labels_snapshot.count != refs.shape[0]:
        raise Error("TypeError: hierarchy fitted labels require one native contiguous int32 value per reference row")
    # p3-neighbors-cluster: the resident record is keyed by the STORED field
    # objects (`_fit_X`, `labels_`), the ones fit minted it for; the per-call
    # snapshot and view objects were new every call, so it never hit.
    var result = agglo_reference_labels(fields, binding, refs_source, fields["labels_"], query.keep, refs.source.addr, labels_snapshot.source.addr,
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
        for i in range(len(names)):  # driver-loop: admit replaced public owners before native teardown
            if String(names[i]) != "_resident_holder" and py_contains(fields, names[i]) and fields[names[i]] is not state[names[i]]:
                native_release_admit(fields[names[i]], context)
        agglo_release(fields)
        lifecycle_admit_fields(fields)
        lifecycle_admit_fields(state)
        for i in range(len(names)):  # driver-loop: publish the admitted pickle schema after the release window
            if String(names[i]) != "_resident_holder":
                agglo_publish(fields, String(names[i]), state[names[i]], context)
        agglo_holder_admit(py_get(fields, "_resident_holder"))
        agglo_publish(fields, "_resident_holder", agglo_holder_empty(), context)
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
        var estimators = args[0] if args[0] is not None else host_estimator_binding_mode(model, call.settings.numeric_mode, "_mojolearn_estimators", context)
        if call.fitting:
            var solver = args[1] if explicit else estimator_binding(PythonObject(None), "_mojolearn_solver", context)
            var cluster = args[2] if explicit else estimator_binding_mode(call.settings.numeric_mode, "_mojolearn_x_cluster", context)
            var neighbors = xn_admit(args[3] if explicit else estimator_binding_mode(call.settings.numeric_mode, "_mojolearn_x_neighbors", context))
            agglo_publish(call.fields, "input_copied_", PythonObject(not (call.x.keep is call.source)), context)
            var bindings = AggloBindings(estimators, solver, cluster, neighbors)
            _agglo_fit(call.fields, call.settings, bindings, call.x, context)
            result = call.fields["labels_"] if operation == "family_cfb_agglo_fit_predict" else model
        else:
            result = _agglo_predict(call.fields, call.settings, estimators, call.x, context)
    except error:
        var refused = hierarchy_public_refusal(error, call, model)
        raise refused^
    finally:
        hierarchy_public_end(model, call)
    return hierarchy_public_finish(model, result, call, context)
