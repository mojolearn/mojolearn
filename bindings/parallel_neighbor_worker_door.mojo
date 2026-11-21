# SPDX-License-Identifier: Apache-2.0
# Native handlers for the neighbor/IVF shard protocol. The same compiled
# search/vote entries as the plain API; no estimator Python method is invoked.
# UNPACK and PACK only: every refusal that compares a count, a shape, a
# radius or a mode word, the method codes and keywords, the sort network's
# shape and the class-set check are typed in
# bindings/parallel_neighbor_worker_run.mojo. The neighbor family's compiled
# entries are reached through `family_call` (boundary calls) until that
# family publishes an entry table.
from std.python import Python, PythonObject
from bindings.admitted_family_door import admit_family, admit_family_meta
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.classical_arrays_door import classical_address
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_labels_door import labels_cast, labels_reshape, classical_decode_labels
from bindings.admitted_input_unpack import admit_ids
from bindings.family_common_pack import family_call, family_entry, family_array
from bindings.parallel_shard_common_door import shard_model, shard_rows, shard_dtype
from bindings.parallel_shard_common_run import shard_item
from bindings.arena_materialize_native_door import arena_ragged_rows
from bindings.host_archive_native_door import host_archive_rbc_metric_object
from providers.python.array_metadata_pyapi_door import ArrayPin
from bindings.neighbors_knn_native_door import (
    _nn_fit, knn_index_finite_once, _dist_params, _check_metric_input, _resident_handle, _report, knn_release,
    _clf_predict, _clf_predict_proba, _reg_predict, knn_door, _reduce,
    knn_admit_x, knn_x_tested, knn_device_tests, knn_host_finite, knn_with_status,
)
from bindings.admitted_input_unpack import admitted_finite_refuse
from bindings.kde_public_native_pack import _kde_scores, _kde_release
from bindings.neighbors_ann_native_door import _ivf_metric, _ann_key_entry
from bindings.host_inference_scalars_door import host_inference_scalar
from core.shared_slot import SharedSlot
from bindings.parallel_keys_unpack import (
    parallel_keys,
    PARALLEL_SHARD_WORKER_OPN_IVF_SEARCH,
    PARALLEL_SHARD_WORKER_OPN_IVF_STORE,
    PARALLEL_SHARD_WORKER_OPN_QUERY_ROWS,
    PARALLEL_SHARD_WORKER_OPN_QUERY_STATE,
    PARALLEL_SHARD_WORKER_OPN_REFERENCE_QUERY,
    PARALLEL_SHARD_WORKER_OPN_REFERENCE_SHARD,
    PARALLEL_SHARD_WORKER_OPN_REFERENCE_STORE,
    PARALLEL_SHARD_WORKER_OPN_REFERENCE_VOTE,
    PARALLEL_SHARD_WORKER_OPN_RELEASE,
)
from bindings.py_balanced import py_item
from bindings.parallel_neighbor_worker_run import (
    NEIGHBOR_METHOD_KNEIGHBORS, NEIGHBOR_METHOD_RADIUS, NEIGHBOR_METHOD_PREDICT, NEIGHBOR_METHOD_PREDICT_PROBA,
    NEIGHBOR_METHOD_SCORE_SAMPLES, neighbor_method_code, neighbor_keyword_admit, neighbor_features_admit,
    neighbor_k_admit, neighbor_filled_admit, neighbor_radius_admit, neighbor_sort_admit, neighbor_fill_admit,
    neighbor_sort_network, neighbor_radius_offsets_admit, neighbor_class_sets_admit, neighbor_ivf_features_admit,
    neighbor_ivf_mode_admit, neighbor_ivf_metric_admit, neighbor_ivf_counts_admit, neighbor_ivf_binding_admit,
    neighbor_ivf_data_admit, neighbor_ivf_lists_admit,
)


struct NeighborSnapshots(Defaultable, Movable):
    var models: PythonObject

    def __init__(out self):
        self.models = PythonObject(None)


comptime NEIGHBOR_SNAPSHOTS = SharedSlot[NeighborSnapshots, "MojolearnParallelNeighborSnapshotsV1"]


def _snapshots() raises -> PythonObject:
    var state = NEIGHBOR_SNAPSHOTS.get_or_create_ptr()
    if state[].models is None:
        state[].models = Python.dict()
    return state[].models


def neighbor_binding(tables: PythonObject, name: String) raises -> PythonObject:
    var nk = parallel_keys()
    var result = py_get(py_item(tables, nk[].k_worker_bindings), name)
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
    labels_cast(Int(py=classical_address(source)), shard_dtype(source), Int(py=classical_address(result)),
                "<i8", Int(py=source.size), context)
    _ = source
    return result


def neighbor_kneighbors(model: PythonObject, source: PythonObject, k_arg: PythonObject,
                        distance_arg: PythonObject, binding: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    var nk = parallel_keys()
    var index = py_get(fields, nk[].k__index)
    if index is None:
        raise Error("ValueError: mojolearn: call fit before kneighbors")
    var k_obj = py_item(fields, nk[].k_n_neighbors) if k_arg is None else k_arg
    var integral = py_type(k_obj) is b.bool
    var integers = py_item(context, nk[].k_integer_types)
    for i in range(len(integers)):  # driver-loop: connected native integer scalar identities
        integral = integral or py_type(k_obj) is integers[i]
    if not integral:
        raise Error("TypeError: n_neighbors must be an int")
    var k = Int(py=k_obj)
    # One operation-owned admission supplies the metric check, dimensions and
    # dispatch address. In particular Jensen-Shannon must not recapture the
    # same converted Array merely to read its minimum. No metadata is cached
    # across public calls or shard requests; mutable/released inputs re-admit.
    var query_pin = ArrayPin()
    var query = knn_admit_x(source, model, context, query_pin)
    var rbc = String(py_item(fields, nk[].k_algorithm)) == "rbc"
    var params = PythonObject(None)
    if not rbc:
        params = _dist_params(model, context)
        _check_metric_input(String(py_type(model).__name__), Int(py=params[0]), query)
    neighbor_features_admit(query.shape[1], Int(py=py_item(fields, nk[].k_n_features_in_)))
    neighbor_k_admit(k, Int(py=py_item(fields, nk[].k_n_samples_fit_)))
    var return_distance = _flag(distance_arg, "return_distance", context)
    var nq = query.shape[0]
    var dist = family_array(Python.tuple(nq, k), "<f4", context)
    var ids = family_array(Python.tuple(nq, k), "<i4" if rbc else "<u4", context)
    # Lane a7b: a borrowed query is tested behind the resident entry's upload
    # (status word); RBC and the CPU column scan it on the host.
    var tested = knn_x_tested(query, source, context)
    var status = List[Int32](length=1, fill=Int32(0))
    try:
        if not tested and (rbc or not knn_device_tests(binding)):
            knn_host_finite(model, query)
        if rbc:
            knn_index_finite_once(model, index)  # the index test, first query only (`_nn_fit`)
            var metric = host_archive_rbc_metric_object(String(py_type(model).__name__), py_item(fields, nk[].k_metric), py_item(fields, nk[].k_p), context)
            var count = family_call(family_entry(binding, "rbc_knn_search", context),
                classical_address(index), PythonObject(query.source.addr), classical_address(ids), classical_address(dist),
                Python.list(index.shape[0], nq, index.shape[1], k, metric[0], metric[1]))
            _report(fields, "n_candidate_distances_", count)
            _report(fields, "used_query_tile_", PythonObject(None))
            # The old Array.min() uses this same native int32 reduction helper.
            neighbor_filled_admit(Int(_reduce(Int(py=classical_address(ids)), 2, nq * k, 0, context).i))
        else:
            var handle = _resident_handle(model, binding, index)
            var sizes = Python.list(index.shape[0], nq, index.shape[1], k, 1, py_item(fields, nk[].k_query_tile))
            var used: PythonObject
            if handle is None:
                knn_index_finite_once(model, index)  # the CPU column's index test, first query only
                used = family_call(family_entry(binding, "knn_search", context), classical_address(index), PythonObject(query.source.addr),
                    classical_address(dist), classical_address(ids), sizes, params)
            else:
                used = family_call(family_entry(binding, "knn_search_resident", context), handle,
                    classical_address(index), PythonObject(query.source.addr), classical_address(dist), classical_address(ids), sizes,
                    params if tested else knn_with_status(params, status))
            admitted_finite_refuse(Int(status[0]), String(py_type(model).__name__))
            _report(fields, "used_query_tile_", used)
    finally:
        # family_call may release the GIL or raise; keep the admitted source
        # owner alive until every consumer of its captured address has left.
        _ = query
        _ = query_pin
    _ = index
    _ = dist
    var wide = _wide(ids, Python.tuple(nq, k), context)
    return Python.tuple(dist, wide) if return_distance else wide


def _radius(model: PythonObject, source: PythonObject, kwargs: PythonObject,
            binding: PythonObject, tables: PythonObject, context: PythonObject) raises -> PythonObject:
    """The worker's radius query as its CSR triple (lane ne2-i-j, plan row
    N451): `(row offsets int32 [nq + 1], ids int64 [nnz], distances float32
    [nnz], return_distance)`. The ragged per-row Arrays are built once in the
    parent (`neighbor_radius_rows`), so the reply carries three buffers, not
    one or two Python Arrays per query row."""
    classical_connect_numpy(context)
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    var nk = parallel_keys()
    var index = py_get(fields, nk[].k__index)
    if index is None:
        raise Error("ValueError: fit the estimator before radius_neighbors")
    var radius = py_get(kwargs, nk[].k_radius)
    radius = py_item(fields, nk[].k_radius) if radius is None else radius
    host_inference_scalar(radius, context)
    var r = Float64(py=radius)
    neighbor_radius_admit(r)
    var distances = _flag(py_get(kwargs, nk[].k_return_distance, True), "return_distance", context)
    var sort = _flag(py_get(kwargs, nk[].k_sort_results, False), "sort_results", context)
    neighbor_sort_admit(sort, distances)
    # Lane a8b-device-x-neighbors-rest: the index was tested ONCE, at fit
    # (radius_public_lifecycle_pyapi.mojo radius_fit) or at archive load
    # (host_archive_native.mojo host_archive_radius); only a borrowed query
    # is tested here, behind radius_neighbors_count's query upload (one
    # status word) on the GPU binding, by one parallel host scan on the CPU
    # column. A query of the index itself (source None) is not tested again.
    var q = index
    var status = List[Int32](length=1, fill=Int32(0))
    var count_params = Python.list(index.shape[0], PythonObject(0), index.shape[1], r, py_item(fields, nk[].k__metric_value),
                                   py_item(fields, nk[].k__metric_arg))
    if source is not None:
        var query_pin = ArrayPin()
        var query = knn_admit_x(source, model, context, query_pin)
        q = query_pin.keep
        if not knn_x_tested(query, source, context):
            if knn_device_tests(binding):
                _ = py_call(count_params.append, PythonObject(Int(status.unsafe_ptr())))
            else:
                knn_host_finite(model, query)
        _ = query^
        _ = query_pin
    neighbor_features_admit(Int(py=q.shape[1]), Int(py=index.shape[1]))
    var nq = Int(py=q.shape[0])
    py_setitem(count_params, PythonObject(1), PythonObject(nq))
    var ptr = family_array(Python.tuple(nq + 1), "<i4", context)
    var nnz = Int(py=family_call(family_entry(binding, "radius_neighbors_count", context),
        classical_address(index), classical_address(q), classical_address(ptr), count_params))
    admitted_finite_refuse(Int(status[0]), String(py_type(model).__name__))
    _ = status^
    var ids = family_array(Python.tuple(nnz), "<i4", context)
    var dist = family_array(Python.tuple(nnz), "<f4", context)
    if nnz > 0:
        var got = Int(py=family_call(family_entry(binding, "radius_neighbors_fill", context),
            classical_address(index), classical_address(q), classical_address(ptr), classical_address(ids), classical_address(dist),
            Python.list(index.shape[0], nq, index.shape[1], r, nnz, 1, py_item(fields, nk[].k__metric_value), py_item(fields, nk[].k__metric_arg))))
        neighbor_fill_admit(got, nnz)
    _ = index
    _ = q
    if sort and nnz > 0:
        var sort_binding = neighbor_binding(tables, "_mojolearn_x_neighbors")
        var network = neighbor_sort_network(nnz)
        var sorted_ids = family_array(Python.tuple(nnz), "<i4", context)
        var sorted_dist = family_array(Python.tuple(nnz), "<f4", context)
        _ = family_call(family_entry(sort_binding, "xn_p2m_row_sort", context),
            Python.list(classical_address(ptr), classical_address(ids), classical_address(dist),
                        classical_address(sorted_ids), classical_address(sorted_dist)),
            Python.list(nq, nnz, network[0], network[1]), Python.list())
        _ = ptr
        _ = ids
        _ = dist
        ids = sorted_ids
        dist = sorted_dist
    var wide = _wide(ids, Python.tuple(nnz), context)
    return Python.tuple(ptr, wide, dist, distances)


def neighbor_radius_rows(triple: PythonObject, context: PythonObject) raises -> PythonObject:
    """One query shard's radius result from its CSR triple (`_radius`): the
    ragged rows the worker used to return, `(distance rows, id rows)` with
    `return_distance`, else the id rows. The same copies of the same words:
    one output Array per query row (the ragged radius contract)."""
    var b = native_loaded_module("builtins")
    if py_type(triple) is not b.tuple or len(triple) != 4:
        raise Error("RuntimeError: radius worker returned an invalid result")
    var ptr = triple[0]
    var wide = triple[1]
    var dist = triple[2]
    var distances = Bool(py=triple[3])
    var nq = neighbor_radius_offsets_admit(Int(py=ptr.shape[0]))
    var rows = _ragged_rows(wide, ptr, nq, context)
    var drows = _ragged_rows(dist, ptr, nq, context) if distances else Python.list()
    _ = ptr
    return Python.tuple(drows, rows) if distances else rows


def _ragged_rows(source: PythonObject, ptr: PythonObject, nq: Int, context: PythonObject) raises -> PythonObject:
    # One output Array per query row (the ragged radius contract), built by
    # `arena_ragged_rows`: the source dtype, item size, address, the row
    # offsets, the Array class admission, the zero seed and the slot setters
    # are resolved once per call, not once per row (lane fb2-neighbors: the
    # per-row shard_array -> tsa_array -> _owned path re-resolved them for
    # every row, ~15 us x 2 x n_queries per radius_neighbors call). Each row
    # is still one native copy into its own store. The distance rows are
    # built only when they are returned.
    var dtype = shard_dtype(source)
    var item = shard_item(dtype)
    var rows = arena_ragged_rows(item[0], dtype, item[1], Int(py=classical_address(source)),
                                 Int(py=classical_address(ptr)), nq, context)
    _ = source  # keep the Python object alive through the native copies above
    _ = ptr
    return rows


def _vote(model: PythonObject, distances: PythonObject, indices: PythonObject, method: String,
          tables: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var nq = Int(py=distances.shape[0])
    var k = Int(py=distances.shape[1])
    var nk = parallel_keys()
    var targets = py_item(fields, nk[].k__y_cols)
    var outputs = Int(py=targets.shape[0])
    var binding = neighbor_binding(tables, "neighbors_base")
    var sizes = Python.list(py_item(fields, nk[].k_n_samples_fit_), nq, 0, k, 0, outputs)
    var dist_params = _dist_params(model, context)
    if py_type(model) is py_item(tables, nk[].k_knr_class) and method == "predict":
        var out = family_array(Python.tuple(nq, outputs), "<f4", context)
        _ = family_call(family_entry(binding, "knn_regress_neighbors", context), classical_address(distances),
            classical_address(indices), classical_address(targets), classical_address(out), sizes, dist_params)
        _ = targets
        _ = distances
        _ = indices
        return out if Bool(py=py_item(fields, nk[].k_outputs_2d_)) else labels_reshape(out, Python.tuple(nq), context)
    if py_type(model) is not py_item(tables, nk[].k_knc_class) or (method != "predict" and method != "predict_proba"):
        raise Error("ValueError: unsupported reference-sharded vote")
    var classes = py_item(fields, nk[].k__classes_list)
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
    var fitted = List[Int](capacity=total)
    for i in range(len(classes)):  # driver-loop: fitted class ids per output
        for j in range(len(classes[i])):  # driver-loop: class metadata
            fitted.append(Int(py=classes[i][j]))
    neighbor_class_sets_admit(Int(py=classical_address(unique)), fitted)
    var offset = 0
    _ = unique
    _ = targets
    _ = distances
    _ = indices
    if method == "predict":
        var wide = _wide(labels, Python.tuple(nq, outputs) if Bool(py=py_item(fields, nk[].k_outputs_2d_)) else Python.tuple(nq), context)
        var names = py_get(fields, nk[].k__label_names)
        if names is not None and not Bool(py=py_item(fields, nk[].k_outputs_2d_)):
            return classical_decode_labels(names, wide, context)
        return wide
    var result = Python.list()
    offset = 0
    for i in range(len(classes)):  # driver-loop: probability blocks per output
        var count = len(classes[i])
        var part = shard_rows(proba, offset, offset + nq * count, context)
        _ = py_call(result.append, labels_reshape(part, Python.tuple(nq, count), context))
        offset += nq * count
    return result if Bool(py=py_item(fields, nk[].k_outputs_2d_)) else result[0]


def _release(token: PythonObject, tables: PythonObject, context: PythonObject) raises:
    var model = py_call(_snapshots().pop, token, PythonObject(None))
    if model is None:
        return
    var cls = py_type(model)
    var nk = parallel_keys()
    if cls is py_item(tables, nk[].k_ivf_class):
        var held = py_get(py_fields(model), nk[].k__resident)
        if held is not None:
            _ = family_call(family_entry(held[2], "ivf_flat_index_release", context), held[1])
            py_setitem(py_fields(model), nk[].k__resident, PythonObject(None))
    elif cls is py_item(tables, nk[].k_kde_class):
        _kde_release(py_fields(model))
    else:
        knn_release(model)


def _ivf_partial(model: PythonObject, source: PythonObject, tables: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var q = admit_family(source, 2, "queries", context, dense=True).array
    var nk = parallel_keys()
    neighbor_ivf_features_admit(Int(py=q.shape[1]), Int(py=py_item(fields, nk[].k_n_features_in_)))
    var m = Int(py=q.shape[0])
    var k = Int(py=py_item(fields, nk[].k_n_neighbors))
    var dist = family_array(Python.tuple(m, k), "<f4", context)
    var ids = family_array(Python.tuple(m, k), "<i4", context)
    var counts = family_array(Python.tuple(m), "<i4", context)
    var binding = neighbor_binding(tables, "_mojolearn_ivf")
    var stored = Python.list(classical_address(py_item(fields, nk[].k_centers_)), classical_address(py_item(fields, nk[].k_center_norms_)),
        classical_address(py_item(fields, nk[].k_list_offsets_)), classical_address(py_item(fields, nk[].k_list_indices_)), classical_address(py_item(fields, nk[].k_list_data_)))
    var sizes = Python.list(py_item(fields, nk[].k_n_rows_), py_item(fields, nk[].k_n_features_in_), py_item(fields, nk[].k_n_lists_), py_item(fields, nk[].k_metric_code_))
    var held = py_get(fields, nk[].k__resident)
    var prepare = knn_door(binding, "ivf_flat_index_prepare")
    # Worker snapshots are private and immutable until released. Unlike a public
    # mutable index, their five input buffers never change during a query round.
    if held is None and prepare is not None:
        var prepare_sizes = py_call(native_loaded_module("builtins").list, sizes)
        _ = py_call(prepare_sizes.append, 1)
        var handle = family_call(prepare, stored, prepare_sizes)
        held = Python.tuple(PythonObject(None), handle, binding)
        py_setitem(fields, nk[].k__resident, held)
    _ = py_call(sizes.extend, Python.list(m, k, py_item(fields, nk[].k_n_probes)))
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


def parallel_neighbor_worker(operation: Int, args: PythonObject, tables: PythonObject,
                             context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var nk = parallel_keys()
    if operation == PARALLEL_SHARD_WORKER_OPN_REFERENCE_SHARD:
        var model = shard_model(py_item(tables, nk[].k_nearest_class), args[0])
        _nn_fit(model, args[1], context)
        try:
            return neighbor_kneighbors(model, args[2], args[3], PythonObject(True), neighbor_binding(tables, "neighbors_base"), context)
        finally:
            knn_release(model)
    if operation == PARALLEL_SHARD_WORKER_OPN_REFERENCE_STORE:
        # (token, params, block): one reference block fitted once and kept
        # under the driver's token until its "release" (pass 3, rank 108).
        _release(args[0], tables, context)
        var stored = shard_model(py_item(tables, nk[].k_nearest_class), args[1])
        _nn_fit(stored, args[2], context)
        py_setitem(_snapshots(), args[0], stored)
        return PythonObject(None)
    if operation == PARALLEL_SHARD_WORKER_OPN_REFERENCE_VOTE:
        var model = shard_model(py_item(tables, nk[].k_query_classes)[args[0]], args[1])
        return _vote(model, args[2], args[3], String(args[4]), tables, context)
    if operation == PARALLEL_SHARD_WORKER_OPN_QUERY_STATE:
        if args[2] is None:
            _release(args[0], tables, context)
        else:
            _release(args[0], tables, context)
            var model = shard_model(py_item(tables, nk[].k_query_classes)[args[1]], args[2])
            py_setitem(py_fields(model), nk[].k__resident_holder, Python.dict())
            py_setitem(_snapshots(), args[0], model)
        return PythonObject(None)
    if operation == PARALLEL_SHARD_WORKER_OPN_RELEASE:
        _release(args[0], tables, context)
        return PythonObject(None)
    if operation == PARALLEL_SHARD_WORKER_OPN_IVF_STORE:
        var model = shard_model(py_item(tables, nk[].k_ivf_class), args[1])
        _ = family_entry(neighbor_binding(tables, "_mojolearn_ivf"), "ivf_flat_partial_search", context)
        py_setitem(_snapshots(), args[0], model)
        return py_item(py_fields(model), nk[].k_n_rows_)
    var model = py_get(_snapshots(), args[0])
    if model is None:
        raise Error("RuntimeError: shard worker has no stored snapshot")
    if operation == PARALLEL_SHARD_WORKER_OPN_IVF_SEARCH:
        return _ivf_partial(model, args[1], tables, context)
    if operation == PARALLEL_SHARD_WORKER_OPN_REFERENCE_QUERY:
        # (token, query rows, k): the stored block's k nearest, with distances.
        return neighbor_kneighbors(model, args[1], args[2], PythonObject(True), neighbor_binding(tables, "neighbors_base"), context)
    if operation != PARALLEL_SHARD_WORKER_OPN_QUERY_ROWS:
        raise Error("ValueError: unsupported native neighbor shard operation")
    var method = String(args[2])
    var kwargs = args[3]
    if py_type(kwargs) is not b.dict:
        raise Error("TypeError: query keywords require an ordinary dict")
    var code = neighbor_method_code(method)
    var keys = py_call(b.list, kwargs)
    for i in range(len(keys)):  # driver-loop: explicit query keyword admission
        var exact = py_type(keys[i]) is b.str
        neighbor_keyword_admit(code, method, exact, String(keys[i]) if exact else String(""))
    var binding = PythonObject(None)
    if code != NEIGHBOR_METHOD_SCORE_SAMPLES:
        binding = neighbor_binding(tables, "neighbors_base")
    var result: PythonObject
    if code == NEIGHBOR_METHOD_KNEIGHBORS:
        result = neighbor_kneighbors(model, args[1], py_get(kwargs, nk[].k_n_neighbors),
            py_get(kwargs, nk[].k_return_distance, True), binding, context)
    elif code == NEIGHBOR_METHOD_RADIUS:
        result = _radius(model, args[1], kwargs, binding, tables, context)
    elif code == NEIGHBOR_METHOD_PREDICT_PROBA:
        result = _clf_predict_proba(model, binding, args[1], context)
    elif code == NEIGHBOR_METHOD_PREDICT and py_type(model) is py_item(tables, nk[].k_knr_class):
        result = _reg_predict(model, binding, args[1], context)
    elif code == NEIGHBOR_METHOD_PREDICT:
        result = _clf_predict(model, binding, args[1], context)
    elif code == NEIGHBOR_METHOD_SCORE_SAMPLES and py_type(model) is py_item(tables, nk[].k_kde_class):
        result = _kde_scores(model, Python.tuple(neighbor_binding(tables, "neighbor_density"), args[1]), False, context)
    else:
        raise Error("ValueError: unsupported native neighbor query method")
    var diagnostics = Python.dict()
    var holder = py_get(py_fields(model), nk[].k__resident_holder, Python.dict())
    var names = Python.tuple("used_query_tile_", "n_candidate_distances_")
    for i in range(len(names)):  # small-loop(two query diagnostic fields)
        var name = names[i]
        if py_contains(holder, name):
            py_setitem(diagnostics, name, holder[name])
    return Python.tuple(result, diagnostics)


def _i32_vector(source: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    # The one integer admission (lane a1b: was labels_view, a serial strided
    # gather and labels_cast): one parallel read of the exporter's strides.
    return admit_ids(source, name, context, wide=False)


def parallel_ivf_admit(index: PythonObject, tables: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var nk = parallel_keys()
    if py_type(index) is not py_item(tables, nk[].k_ivf_class):
        raise Error("TypeError: requires mojolearn.IVFIndex")
    var fields = py_fields(index)
    if not py_contains(fields, nk[].k_list_data_) and not py_contains(fields, nk[].k__device_list_data):
        raise Error("ValueError: fit or load the index first")
    var mode = py_get(fields, nk[].k_numeric_mode)
    if mode is None or (py_type(mode) is b.str and String(mode) == ""):
        mode = py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
    var exact_mode = py_type(mode) is b.str
    neighbor_ivf_mode_admit(exact_mode, String(mode) if exact_mode else String(""))
    var metric = py_item(fields, nk[].k_metric)
    if py_type(metric) is not b.str and py_type(metric) is not b.int and py_type(metric) is not b.float:
        raise Error("TypeError: IVF metric requires an ordinary name or numeric code")
    neighbor_ivf_metric_admit(Int(py=_ivf_metric(fields, context)), Int(py=py_item(fields, nk[].k_metric_code_)))
    var names = Python.tuple("n_lists", "n_probes", "n_neighbors")
    for i in range(len(names)):  # small-loop(three integer parameters)
        if py_type(fields[names[i]]) is not b.int:
            raise Error("TypeError: " + String(names[i]) + " must be an int")
    neighbor_ivf_counts_admit(Int(py=py_item(fields, nk[].k_n_probes)), Int(py=py_item(fields, nk[].k_n_lists_)),
                              Int(py=py_item(fields, nk[].k_n_neighbors)))
    var binding = neighbor_binding(tables, "_mojolearn_ivf")
    var getter = knn_door(binding, "ivf_numeric_mode")
    if getter is not None:
        neighbor_ivf_binding_admit(Int(py=family_call(getter)))
    var n = Int(py=py_item(fields, nk[].k_n_rows_))
    var dim = Int(py=py_item(fields, nk[].k_n_features_in_))
    if py_contains(fields, nk[].k__device_list_data):
        var cached = py_get(fields, nk[].k__resident)
        if cached is None:
            raise Error("RuntimeError: resident IVF index was released before its list data was read")
        var data = family_array(Python.tuple(n, dim), "<f4", context)
        _ = family_call(family_entry(binding, "ivf_flat_index_export", context), cached[1], classical_address(data), PythonObject(n * dim))
        py_setitem(fields, nk[].k_list_data_, data)
        _ = py_call(fields.pop, "_device_list_data")
        var arrays = Python.tuple(_ann_key_entry(fields, "centers_"), _ann_key_entry(fields, "center_norms_"),
            _ann_key_entry(fields, "list_offsets_"), _ann_key_entry(fields, "list_indices_"), _ann_key_entry(fields, "list_data_"))
        var key = Python.tuple(arrays, n, dim, py_item(fields, nk[].k_n_lists_), py_item(fields, nk[].k_metric_code_), False, py_call(b.id, cached[2]))
        py_setitem(fields, nk[].k__resident, Python.tuple(key, cached[1], cached[2]))
    var data = py_item(fields, nk[].k_list_data_)
    var matrix = Int(py=data.ndim) == 2
    neighbor_ivf_data_admit(Int(py=data.ndim), Int(py=data.shape[0]) if matrix else -1, Int(py=data.shape[1]) if matrix else -1, n, dim)
    var ids = _i32_vector(py_item(fields, nk[].k_list_indices_), "list_indices_", context)
    var offsets = _i32_vector(py_item(fields, nk[].k_list_offsets_), "list_offsets_", context)
    neighbor_ivf_lists_admit(Int(py=ids.shape[0]), n, Int(py=offsets.shape[0]), Int(py=py_item(fields, nk[].k_n_lists_)))
    return Python.tuple(binding, ids, offsets)
