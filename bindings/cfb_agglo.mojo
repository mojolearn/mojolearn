# SPDX-License-Identifier: Apache-2.0
# AgglomerativeClustering `fit`, `fit_predict` and `predict`, one native call
# each (python/mojolearn/_hierarchy_impl.py). The shell hands over the
# estimators binding (predict, residency), the solver binding (cuML's
# Boruvka route, `linkage_fit`), the x_cluster binding (every other linkage,
# `x_cluster_call` ENTRY_AGGLO) and the x_neighbors binding (the cosine
# zero-row test). This owner makes `_validate_params`' and the fit's
# refusals in their order and wording, calls those entries with the
# argument lists the Python made, and stores the attributes the Python
# stored; with `prediction_data=True` the training rows become resident at
# fit (bindings/cfb_resident.mojo). Same entries, same inputs: the bits do
# not move.
from std.memory import memcpy
from std.os import getenv
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_contains, py_setitem, py_fields
from bindings.classical_arrays import classical_address
from core.family_cast_impl import family_all_finite_f32
from core.cfb_impl import cfb_any_positive_i32
from bindings.family_common import (
    FAMILY_F32P, FAMILY_I32P, family_repr, family_str, family_call, family_entry, family_array, family_as_f32,
    family_field, family_is_str,
)
from bindings.cluster_native import _reference_queries, _xc_array
from bindings.cfb_resident import cfb_fresh_holder, cfb_mint, cfb_reference_labels
from bindings.cfb_dbscan import cfb_borrow_copy

comptime _E_AGGLO = 11
#: The dense connectivity matrix is `m * m` of cuVS's `int`.
comptime _PAIRWISE_MAX_ROWS = 46340


def _truth(value: PythonObject) raises -> Bool:
    """`bool(value)`, the user's own truth test."""
    return Bool(py=family_call(Python.import_module("builtins").bool, value))


def _eq(a: PythonObject, b: PythonObject) raises -> Bool:
    """`a == b` as a truth value."""
    return _truth(family_call(Python.import_module("operator").eq, a, b))


def _agglo_validate(model: PythonObject, context: PythonObject) raises:
    """`AgglomerativeClustering._validate_params`: the normalized
    connectivity and compute_distances go to `_connectivity` and
    `_compute_distances`."""
    var b = Python.import_module("builtins")
    var fields = py_fields(model)
    var n_clusters = family_field(fields, "n_clusters")
    var metric = family_field(fields, "metric")
    var connectivity = family_field(fields, "connectivity")
    var linkage = family_field(fields, "linkage")
    var c = family_field(fields, "c")
    var memory = family_field(fields, "memory")
    var compute_full_tree = family_field(fields, "compute_full_tree")
    var distance_threshold = family_field(fields, "distance_threshold")
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
    if _truth(family_call(Python.import_module("operator").ne, c, PythonObject(15))):
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
    py_setitem(fields, "_connectivity", connectivity)
    py_setitem(fields, "_compute_distances", PythonObject(_truth(family_field(fields, "compute_distances"))))


def _sparse_like(source: PythonObject) raises -> Bool:
    """`hasattr(X, "toarray") or hasattr(X, "tocsr")`."""
    var b = Python.import_module("builtins")
    return (Bool(py=family_call(b.hasattr, source, PythonObject("toarray")))
            or Bool(py=family_call(b.hasattr, source, PythonObject("tocsr"))))


def _fast_mst(n_cols: Int) raises -> Bool:
    """`_backend.requested_mode() == "fast" and sys.platform == "darwin" and
    n_cols <= 64`: FAST on Apple builds the Euclidean MST from Boruvka rounds
    without the dense graph."""
    var mode = String(getenv("MOJOLEARN_NUMERIC_MODE", "identical").strip()).lower()
    if mode != "fast" and mode != "identical":
        raise Error("ImportError: mojolearn: MOJOLEARN_NUMERIC_MODE=" + family_repr(PythonObject(mode))
                    + "; it must be 'fast', or 'identical' (the default)")
    return mode == "fast" and String(py=Python.import_module("sys").platform) == "darwin" and n_cols <= 64


def _agglo_legacy(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """cuML's single-linkage Boruvka route (`_mojolearn_solver.linkage_fit`)."""
    var b = Python.import_module("builtins")
    var fields = py_fields(model)
    var source = args[4]
    if _sparse_like(source):
        raise Error("NotImplementedError: mojolearn AgglomerativeClustering: sparse X is refused; the "
                    + "pairwise connectivity arm builds its dense m x m graph from "
                    + "a dense float32 matrix")
    var x = family_as_f32(source, 2, "X", context)
    py_setitem(fields, "input_copied_", PythonObject(not (x is source)))
    var n_rows = Int(py=x.shape[0])
    var n_cols = Int(py=x.shape[1])
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
    var k = Int(py=family_call(b.int, family_field(fields, "n_clusters")))
    if k < 1 or k > n_rows:
        raise Error("Expected 1 <= n_clusters <= n_rows (" + String(n_rows) + "), got "
                    + "n_clusters=" + String(k))
    var children = family_array(Python.tuple(n_rows - 1, 2), "<i4", context)
    var labels = family_array(Python.tuple(n_rows), "<i4", context)
    var info = family_array(Python.tuple(2), "<i4", context)
    var metric = Int(py=context["family_agglo_predict_metrics"][family_field(fields, "metric")])
    var use_knn = Int(py=context["family_cfb_agglo_connectivities"][family_field(fields, "_connectivity")])
    var fit = family_entry(args[1], "linkage_fit", context)
    # ORDER MATCHES bindings/_mojolearn_solver.mojo::linkage_fit_binding:
    # n_rows, n_cols, n_clusters, metric, use_knn.
    _ = family_call(fit, classical_address(x), classical_address(children), classical_address(labels),
                    classical_address(info),
                    Python.list(PythonObject(n_rows), PythonObject(n_cols), PythonObject(k), PythonObject(metric),
                                PythonObject(use_knn)))
    _ = labels  # keep the Python object alive through the native call above
    _ = children  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _ = info  # keep the Python object alive through the native call above
    var words = FAMILY_I32P(unsafe_from_address=Int(py=classical_address(info)))
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
    var predicting = Bool(py=family_field(fields, "prediction_data"))
    var fit_x = cfb_borrow_copy(x, source, context) if predicting else PythonObject(None)
    py_setitem(fields, "_fit_X", fit_x)
    _ = x^
    cfb_fresh_holder(model)
    if predicting:
        cfb_mint(model, args[0], fit_x, labels, labels, n_rows, n_cols, 0, context)


def _concat_f32(parts: PythonObject, nnz: Int, context: PythonObject) raises -> PythonObject:
    """`frombytes(rows.tobytes() + cols.tobytes() + vals.tobytes(), "<f4",
    (3 * nnz,))`: three float32 vectors of `nnz` words, back to back."""
    var out = family_array(Python.tuple(3 * nnz), "<f4", context)
    var dst = FAMILY_F32P(unsafe_from_address=Int(py=classical_address(out)))
    _ = out  # keep the Python object alive through the native call above
    for k in range(3):  # small-loop(parts: the COO rows, columns and values): 3
        memcpy(dest=dst + k * nnz, src=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(parts[k]))), count=nnz)
        _ = out  # keep the Python object alive through the native call above
    return out


def _agglo_x(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """The x_cluster route (`x_cluster/agglo.mojo`, ENTRY_AGGLO): every
    linkage, metric, connectivity matrix and the per-merge distances, with
    scikit-learn's `_fit` bookkeeping around it."""
    var b = Python.import_module("builtins")
    var fields = py_fields(model)
    var source = args[4]
    if _sparse_like(source):
        raise Error("NotImplementedError: mojolearn AgglomerativeClustering: sparse X is refused; the "
                    + "dissimilarities are a dense float32 matrix")
    var x = family_as_f32(source, 2, "X", context)
    py_setitem(fields, "input_copied_", PythonObject(not (x is source)))
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    if n < 2:
        raise Error("mojolearn AgglomerativeClustering: n_rows=" + String(n) + " < 2; a tree "
                    + "needs at least two points")
    if not family_all_finite_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(x))), Int(py=x.size)):
        raise Error("mojolearn AgglomerativeClustering: X contains NaN or infinity")
    _ = x  # keep the Python object alive through the native calls above
    var metric = Int(py=context["family_cfb_agglo_all_metrics"][family_field(fields, "metric")])
    if metric == 5 and n != d:
        raise Error("Distance matrix should be square, got matrix of shape (" + String(n) + ", " + String(d) + ")")
    if metric == 4:
        # the zero-row test on the device (the x_neighbors binding's
        # `xn_row_all_zero`, one flag per row from the bit patterns).
        if d == 0:
            raise Error("Cosine affinity cannot be used when X contains zero vectors")
        var zrow = family_array(Python.tuple(n), "<i4", context)
        var zero = family_entry(args[3], "xn_row_all_zero", context)
        _ = family_call(zero, Python.list(classical_address(x), classical_address(zrow)),
                        Python.list(PythonObject(n), PythonObject(d)), Python.list())
        _ = zrow  # keep the Python object alive through the native call above
        _ = x  # keep the Python object alive through the native call above
        if cfb_any_positive_i32(FAMILY_I32P(unsafe_from_address=Int(py=classical_address(zrow))), n):
            raise Error("Cosine affinity cannot be used when X contains zero vectors")
        _ = zrow  # keep the Python object alive through the native calls above
    var n_clusters = family_field(fields, "n_clusters")
    if n_clusters is not None:
        var k = Int(py=family_call(b.int, n_clusters))
        if k < 1 or k > n:
            raise Error("Cannot extract more clusters than samples: " + String(k) + " clusters "
                        + "were given for a tree with " + String(n) + " leaves.")
    var connectivity = family_field(fields, "_connectivity")
    var thr = family_field(fields, "distance_threshold")
    var constrained = not family_is_str(connectivity)
    var full = True
    if constrained:
        var tree = family_field(fields, "compute_full_tree")
        if family_is_str(tree) and String(tree) == "auto":
            # `int(n_clusters) < max(100, 0.02 * n)`
            var bound = 0.02 * Float64(n)
            if bound < 100.0:
                bound = 100.0
            full = thr is not None or Float64(Int(py=family_call(b.int, n_clusters))) < bound
        else:
            full = _truth(tree)
    var n_merges = n - 1 if full else n - Int(py=family_call(b.int, n_clusters))
    # `_connectivity_aux`: the matrix as the binding takes it, mode 1 the
    # dense n x n float32 matrix, mode 2 the COO rows, columns and values as
    # float32, concatenated.
    var aux = PythonObject(None)
    var n_edges = -1
    var edge_mode = 0
    if constrained:
        var conn = connectivity
        if Bool(py=family_call(b.callable, conn)):
            conn = family_call(conn, x)
        var tocoo = py_call(b.getattr, conn, PythonObject("tocoo"), PythonObject(None))
        var shape = PythonObject(None)
        if Bool(py=family_call(b.callable, tocoo)):
            var coo = family_call(tocoo)
            shape = family_call(b.tuple, coo.shape)
            var rows = family_as_f32(coo.row, 1, "connectivity rows", context)
            var cols = family_as_f32(coo.col, 1, "connectivity cols", context)
            var vals = family_as_f32(coo.data, 1, "connectivity values", context)
            var nnz = Int(py=vals.shape[0])
            if nnz:
                aux = _concat_f32(Python.tuple(rows, cols, vals), nnz, context)
            edge_mode = 2
            n_edges = nnz
        else:
            var dense = family_as_f32(conn, 2, "connectivity", context)
            shape = py_call(b.tuple, dense.shape)
            aux = dense
            edge_mode = 1
            n_edges = Int(py=dense.size)
        if not _eq(shape, Python.tuple(n, n)):
            raise Error("Wrong shape for connectivity matrix: " + family_repr(shape) + " when X has "
                        + String(n) + " samples")
    var cut = 0
    if full:
        cut = 1 if thr is not None else Int(py=family_call(b.int, n_clusters))
    var ip = Python.list(PythonObject(n), PythonObject(d),
                         context["family_cfb_agglo_linkages"][family_field(fields, "linkage")],
                         PythonObject(metric), PythonObject(n_edges), PythonObject(n_merges), PythonObject(edge_mode),
                         PythonObject(cut), PythonObject(1 if (full and thr is not None) else 0))
    var fp = Python.list(PythonObject(2.0), family_call(b.float, thr) if thr is not None else PythonObject(0.0))
    var call = family_entry(args[2], "x_cluster_call", context)
    var aux_address = PythonObject(0)
    var aux_size = 0
    if aux is not None:
        aux_address = classical_address(aux)
        _ = aux  # keep the Python object alive through the native call above
        aux_size = Int(py=aux.size)
    var result = family_call(call, PythonObject(_E_AGGLO), classical_address(x), PythonObject(Int(py=x.size)),
                             aux_address, PythonObject(aux_size), ip, fp)
    _ = x  # keep the Python object alive through the native call above
    _ = aux^
    var f = result[0]
    _ = x  # keep the Python object alive through the native call above
    var i = result[1]
    _ = x  # keep the Python object alive through the native call above
    var sc = result[2]
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, "children_", _xc_array(i[0], Python.tuple(n_merges, 2), "<i4", "i", context))
    py_setitem(fields, "n_leaves_", PythonObject(n))
    var components = Int(py=family_call(b.int, sc[0]))
    py_setitem(fields, "n_connected_components_", PythonObject(components))
    if constrained and components > 1:
        _ = family_call(Python.import_module("warnings").warn,
                        PythonObject("the number of connected components of the connectivity matrix "
                                     + "is " + String(components) + " > 1. Completing it to avoid "
                                     + "stopping the tree early."))
    if thr is not None or Bool(py=family_field(fields, "_compute_distances")):
        py_setitem(fields, "distances_", _xc_array(f[0], Python.tuple(n_merges), "<f4", "f", context))
    else:
        _ = py_call(fields.pop, PythonObject("distances_"), PythonObject(None))
    var clusters = PythonObject(None)
    if thr is not None and full:
        clusters = family_call(b.int, sc[1])
    else:
        clusters = family_call(b.int, n_clusters)
    py_setitem(fields, "n_clusters_", clusters)
    py_setitem(fields, "labels_", _xc_array(i[1], Python.tuple(n), "<i4", "i", context))
    py_setitem(fields, "n_boruvka_rounds_", PythonObject(-1))
    py_setitem(fields, "n_features_in_", PythonObject(d))
    var fit_x = PythonObject(None)
    var predicting = Bool(py=family_field(fields, "prediction_data")) and _predicts(fields, context)
    if predicting:
        fit_x = cfb_borrow_copy(x, source, context)
    py_setitem(fields, "_fit_X", fit_x)
    _ = x^
    cfb_fresh_holder(model)
    if predicting:
        var labels = fields["labels_"]
        cfb_mint(model, args[0], fit_x, labels, labels, n, d, 0, context)


def _predicts(fields: PythonObject, context: PythonObject) raises -> Bool:
    """`_predicts`: single linkage under a euclidean metric name."""
    return (_eq(family_field(fields, "linkage"), PythonObject("single"))
            and py_contains(context["family_agglo_predict_metrics"], family_field(fields, "metric")))


def _agglo_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`AgglomerativeClustering.fit(X)`."""
    var b = Python.import_module("builtins")
    var fields = py_fields(model)
    _agglo_validate(model, context)
    var prediction_data = family_field(fields, "prediction_data")
    if py_type(prediction_data) is not b.bool:
        raise Error("TypeError: mojolearn AgglomerativeClustering: prediction_data must be a "
                    + "bool, got " + String(py_type(prediction_data).__name__))
    # `_legacy_route`: cuML's single-linkage Boruvka route keeps the recorded
    # `agglomerative` lanes' bits; everything else is the x_cluster route.
    var legacy = (_predicts(fields, context) and family_is_str(family_field(fields, "_connectivity"))
                  and family_field(fields, "distance_threshold") is None
                  and not Bool(py=family_field(fields, "_compute_distances")))
    if legacy:
        _agglo_legacy(model, args, context)
    else:
        _agglo_x(model, args, context)


def _agglo_predict(model: PythonObject, binding: PythonObject, source: PythonObject,
                   context: PythonObject) raises -> PythonObject:
    """`AgglomerativeClustering.predict`: the nearest training row's cluster
    (single linkage, euclidean), over the resident training rows."""
    var b = Python.import_module("builtins")
    if not Bool(py=family_call(b.hasattr, model, PythonObject("labels_"))):
        raise Error("mojolearn AgglomerativeClustering.predict: this instance is not "
                    + "fitted yet; call fit first")
    var linkage = py_call(b.getattr, model, PythonObject("linkage"))
    var metric = py_call(b.getattr, model, PythonObject("metric"))
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
    var result = cfb_reference_labels(model, binding, refs, labels, labels, q, Int(py=refs.shape[0]), n_features, 0,
                                      PythonObject(0.0), 0, context)
    _ = q^
    return result


def cfb_agglo_operation(model: PythonObject, operation: String, args: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    """fit args: (estimators binding, solver binding, x_cluster binding,
    x_neighbors binding, X); predict args: (estimators binding, X)."""
    if operation == "family_cfb_agglo_fit":
        _agglo_fit(model, args, context)
        return model
    if operation == "family_cfb_agglo_fit_predict":
        _agglo_fit(model, args, context)
        return py_fields(model)["labels_"]
    if operation == "family_cfb_agglo_predict":
        return _agglo_predict(model, args[0], args[1], context)
    raise Error("RuntimeError: unknown AgglomerativeClustering operation " + operation)
