# SPDX-License-Identifier: Apache-2.0
# Native handlers for the neighbor/IVF shard protocol. The same compiled
# search/vote entries as the plain API; no estimator Python method is invoked.
from std.ffi import _Global
from std.math import isfinite
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.classical_arrays import classical_address
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_labels import labels_cast, labels_reshape, labels_view, labels_check_rank, classical_decode_labels
from bindings.family_common import family_call, family_entry, family_array, family_as_f32_dense
from bindings.parallel_shard_common import shard_model, shard_rows
from bindings.host_archive_native import host_archive_rbc_metric_object
from bindings.neighbors_knn_native import (
    _nn_fit, _dist_params, _check_metric_input, _resident_handle, _report, knn_release,
    _clf_predict, _clf_predict_proba, _reg_predict, knn_door, _reduce,
)
from bindings.neighbors_kde_native import _kde_scores, _kde_release
from bindings.neighbors_ann_native import _ivf_metric, _ann_key_entry
from bindings.host_inference_scalars import host_inference_scalar


struct NeighborSnapshots(Defaultable, Movable):
    var models: PythonObject

    def __init__(out self):
        self.models = PythonObject(None)


comptime NEIGHBOR_SNAPSHOTS = _Global[StorageType=NeighborSnapshots,
    name="MojoParallelNeighborSnapshots", init_fn=NeighborSnapshots.__init__]


def _snapshots() raises -> PythonObject:
    var state = NEIGHBOR_SNAPSHOTS.get_or_create_ptr()
    if state[].models is None:
        state[].models = Python.dict()
    return state[].models


def neighbor_binding(tables: PythonObject, name: String) raises -> PythonObject:
    var result = py_get(tables["worker_bindings"], name)
    if result is None:
        raise Error("ImportError: native shard binding is not connected: " + name)
    if py_type(result) is py_type(PythonObject("")):
        raise Error("ImportError: " + String(result))
    return result


def _flag(value: PythonObject, name: String, context: PythonObject) raises -> Bool:
    # Retain ordinary native scalar truth semantics (including NumPy bool),
    # while refusing Python-defined __bool__ before it can execute.
    host_inference_scalar(value, context, True)
    return Bool(py=value)


def _wide(source: PythonObject, shape: PythonObject, context: PythonObject) raises -> PythonObject:
    var result = family_array(shape, "<i8", context)
    labels_cast(Int(py=classical_address(source)), String(source.dtype), Int(py=classical_address(result)),
                "<i8", Int(py=source.size), context)
    _ = source
    return result


def neighbor_kneighbors(model: PythonObject, source: PythonObject, k_arg: PythonObject,
                        distance_arg: PythonObject, binding: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    var index = py_get(fields, "_index")
    if index is None:
        raise Error("ValueError: mojolearn: call fit before kneighbors")
    var k_obj = fields["n_neighbors"] if k_arg is None else k_arg
    var integral = py_type(k_obj) is b.bool
    var integers = context["integer_types"]
    for i in range(len(integers)):  # driver-loop: connected native integer scalar identities
        integral = integral or py_type(k_obj) is integers[i]
    if not integral:
        raise Error("TypeError: n_neighbors must be an int")
    var k = Int(py=k_obj)
    var q = family_as_f32_dense(source, 2, "X", context)
    var rbc = String(fields["algorithm"]) == "rbc"
    var params = PythonObject(None)
    if not rbc:
        params = _dist_params(model, context)
        _check_metric_input(String(py_type(model).__name__), Int(py=params[0]), q, context)
    if Int(py=q.shape[1]) != Int(py=fields["n_features_in_"]):
        raise Error("ValueError: input feature count differs from fit")
    if k < 1 or k > Int(py=fields["n_samples_fit_"]):
        raise Error("ValueError: n_neighbors must be in [1, reference rows]")
    var return_distance = _flag(distance_arg, "return_distance", context)
    var nq = Int(py=q.shape[0])
    var dist = family_array(Python.tuple(nq, k), "<f4", context)
    var ids = family_array(Python.tuple(nq, k), "<i4" if rbc else "<u4", context)
    if rbc:
        var metric = host_archive_rbc_metric_object(String(py_type(model).__name__), fields["metric"], fields["p"], context)
        var count = family_call(family_entry(binding, "rbc_knn_search", context),
            classical_address(index), classical_address(q), classical_address(ids), classical_address(dist),
            Python.list(index.shape[0], nq, index.shape[1], k, metric[0], metric[1]))
        _report(fields, "n_candidate_distances_", count)
        _report(fields, "used_query_tile_", PythonObject(None))
        # The old Array.min() uses this same native int32 reduction helper.
        if Int(py=_reduce(Int(py=classical_address(ids)), 2, nq * k, 0, context)) < 0:
            raise Error("RuntimeError: RBC returned an unfilled neighbor slot")
    else:
        var handle = _resident_handle(model, binding, index)
        var sizes = Python.list(index.shape[0], nq, index.shape[1], k, 1, fields["query_tile"])
        var used: PythonObject
        if handle is None:
            used = family_call(family_entry(binding, "knn_search", context), classical_address(index), classical_address(q),
                classical_address(dist), classical_address(ids), sizes, params)
        else:
            used = family_call(family_entry(binding, "knn_search_resident", context), handle,
                classical_address(index), classical_address(q), classical_address(dist), classical_address(ids), sizes, params)
        _report(fields, "used_query_tile_", used)
    _ = index
    _ = q
    _ = dist
    var wide = _wide(ids, Python.tuple(nq, k), context)
    return Python.tuple(dist, wide) if return_distance else wide


def _radius(model: PythonObject, source: PythonObject, kwargs: PythonObject,
            binding: PythonObject, tables: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    var index = py_get(fields, "_index")
    if index is None:
        raise Error("ValueError: fit the estimator before radius_neighbors")
    var radius = py_get(kwargs, "radius")
    radius = fields["radius"] if radius is None else radius
    host_inference_scalar(radius, context)
    var r = Float64(py=radius)
    if not isfinite(r) or r <= 0:
        raise Error("ValueError: radius must be positive and finite")
    var distances = _flag(py_get(kwargs, "return_distance", True), "return_distance", context)
    var sort = _flag(py_get(kwargs, "sort_results", False), "sort_results", context)
    if sort and not distances:
        raise Error("ValueError: sort_results=True requires return_distance=True")
    var q = index if source is None else family_as_f32_dense(source, 2, "X", context)
    if Int(py=q.shape[1]) != Int(py=index.shape[1]):
        raise Error("ValueError: input feature count differs from fit")
    var nq = Int(py=q.shape[0])
    var ptr = family_array(Python.tuple(nq + 1), "<i4", context)
    var nnz = Int(py=family_call(family_entry(binding, "radius_neighbors_count", context),
        classical_address(index), classical_address(q), classical_address(ptr),
        Python.list(index.shape[0], nq, index.shape[1], r, fields["_metric_value"], fields["_metric_arg"])))
    var ids = family_array(Python.tuple(nnz), "<i4", context)
    var dist = family_array(Python.tuple(nnz), "<f4", context)
    if nnz > 0:
        var got = Int(py=family_call(family_entry(binding, "radius_neighbors_fill", context),
            classical_address(index), classical_address(q), classical_address(ptr), classical_address(ids), classical_address(dist),
            Python.list(index.shape[0], nq, index.shape[1], r, nnz, 1, fields["_metric_value"], fields["_metric_arg"])))
        if got != nnz:
            raise Error("RuntimeError: radius counting and filling passes disagree")
    _ = index
    _ = q
    if sort and nnz > 0:
        var sort_binding = neighbor_binding(tables, "_mojolearn_x_neighbors")
        var power = 1
        var lg = 0
        while power < nnz:  # driver-loop: segmented sorting network shape
            power *= 2
            lg += 1
        var sorted_ids = family_array(Python.tuple(nnz), "<i4", context)
        var sorted_dist = family_array(Python.tuple(nnz), "<f4", context)
        _ = family_call(family_entry(sort_binding, "xn_p2m_row_sort", context),
            Python.list(classical_address(ptr), classical_address(ids), classical_address(dist),
                        classical_address(sorted_ids), classical_address(sorted_dist)),
            Python.list(nq, nnz, power, lg * (lg + 1) // 2), Python.list())
        _ = ptr
        _ = ids
        _ = dist
        ids = sorted_ids
        dist = sorted_dist
    var wide = _wide(ids, Python.tuple(nnz), context)
    var ip = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(ptr)))
    var rows = Python.list()
    var drows = Python.list()
    for i in range(nq):  # driver-loop: native ragged output assembly
        var lo = Int(ip[i])
        var hi = Int(ip[i + 1])
        _ = py_call(rows.append, shard_rows(wide, lo, hi, context))
        _ = py_call(drows.append, shard_rows(dist, lo, hi, context))
    _ = ptr
    return Python.tuple(drows, rows) if distances else rows


def _vote(model: PythonObject, distances: PythonObject, indices: PythonObject, method: String,
          tables: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var nq = Int(py=distances.shape[0])
    var k = Int(py=distances.shape[1])
    var targets = fields["_y_cols"]
    var outputs = Int(py=targets.shape[0])
    var binding = neighbor_binding(tables, "neighbors_base")
    var sizes = Python.list(fields["n_samples_fit_"], nq, 0, k, 0, outputs)
    var dist_params = _dist_params(model, context)
    if py_type(model) is tables["knr_class"] and method == "predict":
        var out = family_array(Python.tuple(nq, outputs), "<f4", context)
        _ = family_call(family_entry(binding, "knn_regress_neighbors", context), classical_address(distances),
            classical_address(indices), classical_address(targets), classical_address(out), sizes, dist_params)
        _ = targets
        _ = distances
        _ = indices
        return out if Bool(py=fields["outputs_2d_"]) else labels_reshape(out, Python.tuple(nq), context)
    if py_type(model) is not tables["knc_class"] or (method != "predict" and method != "predict_proba"):
        raise Error("ValueError: unsupported reference-sharded vote")
    var classes = fields["_classes_list"]
    var total = 0
    _ = py_call(sizes.append, 1 if method == "predict_proba" else 0)
    for i in range(len(classes)):  # driver-loop: class counts per output
        total += len(classes[i])
        _ = py_call(sizes.append, len(classes[i]))
    var labels = family_array(Python.tuple(nq, outputs), "<i4", context)
    var proba = family_array(Python.tuple(nq * total), "<f4", context)
    var unique = family_array(Python.tuple(total), "<i4", context)
    _ = family_call(family_entry(binding, "knn_classify_neighbors", context), classical_address(distances),
        classical_address(indices), classical_address(targets), classical_address(labels), classical_address(proba),
        classical_address(unique), sizes, dist_params)
    var up = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(unique)))
    var offset = 0
    for i in range(len(classes)):  # driver-loop: original native/fitted class-set check
        for j in range(len(classes[i])):  # driver-loop: class metadata
            if Int(up[offset + j]) != Int(py=classes[i][j]):
                raise Error("RuntimeError: native and fitted class sets differ")
        offset += len(classes[i])
    _ = unique
    _ = targets
    _ = distances
    _ = indices
    if method == "predict":
        var wide = _wide(labels, Python.tuple(nq, outputs) if Bool(py=fields["outputs_2d_"]) else Python.tuple(nq), context)
        var names = py_get(fields, "_label_names")
        if names is not None and not Bool(py=fields["outputs_2d_"]):
            return classical_decode_labels(names, wide, context)
        return wide
    var result = Python.list()
    offset = 0
    for i in range(len(classes)):  # driver-loop: probability blocks per output
        var count = len(classes[i])
        var part = shard_rows(proba, offset, offset + nq * count, context)
        _ = py_call(result.append, labels_reshape(part, Python.tuple(nq, count), context))
        offset += nq * count
    return result if Bool(py=fields["outputs_2d_"]) else result[0]


def _release(token: PythonObject, tables: PythonObject, context: PythonObject) raises:
    var model = py_call(_snapshots().pop, token, PythonObject(None))
    if model is None:
        return
    var cls = py_type(model)
    if cls is tables["ivf_class"]:
        var held = py_get(py_fields(model), "_resident")
        if held is not None:
            _ = family_call(family_entry(held[2], "ivf_flat_index_release", context), held[1])
            py_setitem(py_fields(model), "_resident", PythonObject(None))
    elif cls is tables["kde_class"]:
        _kde_release(py_fields(model))
    else:
        knn_release(model)


def _ivf_partial(model: PythonObject, source: PythonObject, tables: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var q = family_as_f32_dense(source, 2, "queries", context)
    if Int(py=q.shape[1]) != Int(py=fields["n_features_in_"]):
        raise Error("ValueError: query features differ from index")
    var m = Int(py=q.shape[0])
    var k = Int(py=fields["n_neighbors"])
    var dist = family_array(Python.tuple(m, k), "<f4", context)
    var ids = family_array(Python.tuple(m, k), "<i4", context)
    var counts = family_array(Python.tuple(m), "<i4", context)
    var binding = neighbor_binding(tables, "_mojolearn_ivf")
    var stored = Python.list(classical_address(fields["centers_"]), classical_address(fields["center_norms_"]),
        classical_address(fields["list_offsets_"]), classical_address(fields["list_indices_"]), classical_address(fields["list_data_"]))
    var sizes = Python.list(fields["n_rows_"], fields["n_features_in_"], fields["n_lists_"], fields["metric_code_"])
    var held = py_get(fields, "_resident")
    var prepare = knn_door(binding, "ivf_flat_index_prepare")
    # Worker snapshots are private and immutable until released. Unlike a public
    # mutable index, their five input buffers never change during a query round.
    if held is None and prepare is not None:
        var prepare_sizes = py_call(native_loaded_module("builtins").list, sizes)
        _ = py_call(prepare_sizes.append, 1)
        var handle = family_call(prepare, stored, prepare_sizes)
        held = Python.tuple(PythonObject(None), handle, binding)
        py_setitem(fields, "_resident", held)
    _ = py_call(sizes.extend, Python.list(m, k, fields["n_probes"]))
    var output = Python.list(classical_address(q), classical_address(dist), classical_address(ids), classical_address(counts))
    if held is not None:
        _ = py_call(sizes.append, 1)
        _ = family_call(family_entry(binding, "ivf_flat_index_search", context), held[1], output, sizes)
    else:
        _ = py_call(stored.extend, output)
        _ = family_call(family_entry(binding, "ivf_flat_partial_search", context), stored, sizes)
    _ = q
    _ = model
    return Python.tuple(dist, ids, counts)


def parallel_neighbor_worker(operation: String, args: PythonObject, tables: PythonObject,
                             context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if operation == "reference_shard":
        var model = shard_model(tables["nearest_class"], args[0])
        _nn_fit(model, args[1], context)
        try:
            return neighbor_kneighbors(model, args[2], args[3], PythonObject(True), neighbor_binding(tables, "neighbors_base"), context)
        finally:
            knn_release(model)
    if operation == "reference_vote":
        var model = shard_model(tables["query_classes"][args[0]], args[1])
        return _vote(model, args[2], args[3], String(args[4]), tables, context)
    if operation == "query_state":
        if args[2] is None:
            _release(args[0], tables, context)
        else:
            _release(args[0], tables, context)
            var model = shard_model(tables["query_classes"][args[1]], args[2])
            py_setitem(py_fields(model), "_resident_holder", Python.dict())
            py_setitem(_snapshots(), args[0], model)
        return PythonObject(None)
    if operation == "release":
        _release(args[0], tables, context)
        return PythonObject(None)
    if operation == "ivf_store":
        var model = shard_model(tables["ivf_class"], args[1])
        _ = family_entry(neighbor_binding(tables, "_mojolearn_ivf"), "ivf_flat_partial_search", context)
        py_setitem(_snapshots(), args[0], model)
        return py_fields(model)["n_rows_"]
    var model = py_get(_snapshots(), args[0])
    if model is None:
        raise Error("RuntimeError: shard worker has no stored snapshot")
    if operation == "ivf_search":
        return _ivf_partial(model, args[1], tables, context)
    if operation != "query_rows":
        raise Error("ValueError: unsupported native neighbor shard operation")
    var method = String(args[2])
    var kwargs = args[3]
    if py_type(kwargs) is not b.dict:
        raise Error("TypeError: query keywords require an ordinary dict")
    var allowed = Python.tuple()
    if method == "kneighbors":
        allowed = Python.tuple("n_neighbors", "return_distance")
    elif method == "radius_neighbors":
        allowed = Python.tuple("radius", "return_distance", "sort_results")
    var keys = py_call(b.list, kwargs)
    for i in range(len(keys)):  # driver-loop: explicit query keyword admission
        if py_type(keys[i]) is not b.str or not py_contains(allowed, keys[i]):
            raise Error("TypeError: unsupported keyword for " + method)
    var binding = PythonObject(None)
    if method != "score_samples":
        binding = neighbor_binding(tables, "neighbors_base")
    var result: PythonObject
    if method == "kneighbors":
        result = neighbor_kneighbors(model, args[1], py_get(kwargs, "n_neighbors"),
            py_get(kwargs, "return_distance", True), binding, context)
    elif method == "radius_neighbors":
        result = _radius(model, args[1], kwargs, binding, tables, context)
    elif method == "predict_proba":
        result = _clf_predict_proba(model, binding, args[1], context)
    elif method == "predict" and py_type(model) is tables["knr_class"]:
        result = _reg_predict(model, binding, args[1], context)
    elif method == "predict":
        result = _clf_predict(model, binding, args[1], context)
    elif method == "score_samples" and py_type(model) is tables["kde_class"]:
        result = _kde_scores(model, Python.tuple(neighbor_binding(tables, "neighbor_density"), args[1]), False, context)
    else:
        raise Error("ValueError: unsupported native neighbor query method")
    var diagnostics = Python.dict()
    var holder = py_get(py_fields(model), "_resident_holder", Python.dict())
    var names = Python.tuple("used_query_tile_", "n_candidate_distances_")
    for i in range(len(names)):  # small-loop(two query diagnostic fields)
        var name = names[i]
        if py_contains(holder, name):
            py_setitem(diagnostics, name, holder[name])
    return Python.tuple(result, diagnostics)


def _i32_vector(source: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    var view = labels_view(source, name, context)
    labels_check_rank(view, 1, name)
    var n = view.size()
    var out = family_array(Python.tuple(n), "<i4", context)
    var src = view.addr
    var contiguous = PythonObject(None)
    if view.strides[0] != view.item:
        contiguous = family_array(Python.tuple(n), view.typestr, context)
        var dst = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(contiguous)))
        for i in range(n):  # driver-loop: gather a strided metadata vector before one native cast
            memcpy(dest=dst + i * view.item,
                   src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=src + i * view.strides[0]), count=view.item)
        src = Int(py=classical_address(contiguous))
    labels_cast(src, view.typestr, Int(py=classical_address(out)), "<i4", n, context)
    _ = view.keep
    _ = contiguous
    return out


def parallel_ivf_admit(index: PythonObject, tables: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(index) is not tables["ivf_class"]:
        raise Error("TypeError: requires mojolearn.IVFIndex")
    var fields = py_fields(index)
    if not py_contains(fields, "list_data_") and not py_contains(fields, "_device_list_data"):
        raise Error("ValueError: fit or load the index first")
    var mode = py_get(fields, "numeric_mode")
    if mode is None or (py_type(mode) is b.str and String(mode) == ""):
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not b.str or String(mode) != "identical":
        raise Error("ValueError: distributed IVF requires IDENTICAL numeric mode")
    var metric = fields["metric"]
    if py_type(metric) is not b.str and py_type(metric) is not b.int and py_type(metric) is not b.float:
        raise Error("TypeError: IVF metric requires an ordinary name or numeric code")
    if Int(py=_ivf_metric(fields, context)) != Int(py=fields["metric_code_"]):
        raise Error("ValueError: metric differs from the fitted index")
    var names = Python.tuple("n_lists", "n_probes", "n_neighbors")
    for i in range(len(names)):  # small-loop(three integer parameters)
        if py_type(fields[names[i]]) is not b.int:
            raise Error("TypeError: " + String(names[i]) + " must be an int")
    if Int(py=fields["n_probes"]) < 1 or Int(py=fields["n_probes"]) > Int(py=fields["n_lists_"]) or Int(py=fields["n_neighbors"]) < 1:
        raise Error("ValueError: invalid IVF probe or neighbor count")
    var binding = neighbor_binding(tables, "_mojolearn_ivf")
    var getter = knn_door(binding, "ivf_numeric_mode")
    if getter is not None and Int(py=family_call(getter)) != 1:
        raise Error("RuntimeError: IVF binding reports another compile-time mode")
    var n = Int(py=fields["n_rows_"])
    var dim = Int(py=fields["n_features_in_"])
    if py_contains(fields, "_device_list_data"):
        var cached = py_get(fields, "_resident")
        if cached is None:
            raise Error("RuntimeError: resident IVF index was released before its list data was read")
        var data = family_array(Python.tuple(n, dim), "<f4", context)
        _ = family_call(family_entry(binding, "ivf_flat_index_export", context), cached[1], classical_address(data), PythonObject(n * dim))
        py_setitem(fields, "list_data_", data)
        _ = py_call(fields.pop, "_device_list_data")
        var arrays = Python.tuple(_ann_key_entry(fields, "centers_"), _ann_key_entry(fields, "center_norms_"),
            _ann_key_entry(fields, "list_offsets_"), _ann_key_entry(fields, "list_indices_"), _ann_key_entry(fields, "list_data_"))
        var key = Python.tuple(arrays, n, dim, fields["n_lists_"], fields["metric_code_"], False, py_call(b.id, cached[2]))
        py_setitem(fields, "_resident", Python.tuple(key, cached[1], cached[2]))
    var data = fields["list_data_"]
    if Int(py=data.ndim) != 2 or Int(py=data.shape[0]) != n or Int(py=data.shape[1]) != dim:
        raise Error("ValueError: IVF stored data shape differs from index metadata")
    var ids = _i32_vector(fields["list_indices_"], "list_indices_", context)
    var offsets = _i32_vector(fields["list_offsets_"], "list_offsets_", context)
    if Int(py=ids.shape[0]) != n:
        raise Error("ValueError: IVF original IDs must be a permutation of stored row IDs")
    if Int(py=offsets.shape[0]) != Int(py=fields["n_lists_"]) + 1:
        raise Error("ValueError: invalid global IVF list offsets")
    return Python.tuple(binding, ids, offsets)
