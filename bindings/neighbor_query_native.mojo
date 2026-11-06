# SPDX-License-Identifier: Apache-2.0
# Complete standalone queries, sharing the worker's native search ownership.
from std.memory import memcpy
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_loaded_module import native_loaded_module, NativeModules
from bindings.native_prepared_owner import prepared_get, prepared_put, prepared_drop, prepared_key_add, prepared_key_same
from bindings.native_error_class import error_without_class
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_buffer_owner import native_buffer_view
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.classical_attributes import classical_connect_numpy, classical_check_layout, classical_getattribute
from bindings.classical_arrays import classical_address, classical_is_array, classical_c_order
from bindings.classical_labels import labels_view, labels_check_rank, labels_reshape
from bindings.classical_pandas import classical_admit_value, native_pandas_exact, _pd_class, _pd_fields, _pd_get, _pd_values
from bindings.sequential_admission import _ordinary_torch
from bindings.native_torch_admission import native_torch_numpy
from bindings.sequential_inputs import sequential_input, sequential_sequence
from bindings.family_common import family_array, family_as_f32, family_entry, family_call, family_call_kw
from bindings.family_common import family_f32_snapshot
from providers.python.array_metadata_pyapi import ArrayMetadata, array_shape_once, array_metadata_once
from core.cast_in import CAST_KIND_FLOAT, CAST_KIND_SIGNED
from bindings.neighbors_knn_native import knn_door, knn_state_values, knn_state_unchanged
from bindings.neighbors_ann_native import _ivf_metric, _ann_release, IVF_QUERY_STATE_SLOT
from bindings.neighbors_ann_query_native import ann_pq_materialize
from bindings.parallel_neighbor_worker import neighbor_kneighbors, _radius, _i32_vector, _flag
from bindings.host_inference_scalars import host_inference_scalar
from bindings.parallel_public_result import parallel_public_result
from bindings.scaler_lifecycle_native import _outside
from bindings.p3nc_fitted_freeze import fitted_is_freezable, fitted_freeze


def _query_kind(model: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_check_layout(model, context)
    var kind = py_get(context["neighbor_query_classes"], py_type(model))
    if kind is None:
        raise Error("TypeError: neighbor query requires an exact connected native model")
    return kind


def _query_route(model: PythonObject, kind: PythonObject, group: String, context: PythonObject) raises -> PythonObject:
    # `kind` comes from the caller's `_query_kind` (one layout check per call).
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    var mode = py_get(fields, "numeric_mode")
    var host = Bool(py=kind[1])
    if host and mode is None:
        mode = PythonObject("identical")
    elif mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not b.str:
        raise Error("TypeError: numeric_mode must be a string or None")
    if host and String(mode) != "identical":
        raise Error("ValueError: saved CPU models require identical numeric mode")
    # Perf (perf2-neighbors): only a non-canonical spelling pays the two str
    # method calls (as estimator_binding_mode does); an exact canonical str is
    # already its own normal form.
    var spelled = String(mode)
    if spelled != "identical" and spelled != "fast":
        mode = py_call(py_call(mode.strip).lower)
    if String(mode) != "identical" and String(mode) != "fast":
        raise Error("ValueError: numeric_mode must be identical or fast")
    var routes = context["neighbor_query_host_routes"] if host else context["neighbor_query_routes"]
    var result = routes[group][mode]
    if py_type(result) is b.str:
        raise Error("ImportError: " + String(result))
    # The routes are fixed import-time modules: check each one's compile-time
    # mode once per process, not on every query.
    var checked = py_get(context, "neighbor_query_checked_routes")
    if checked is None or py_type(checked) is not b.dict:
        checked = Python.dict()
        py_setitem(context, "neighbor_query_checked_routes", checked)
    var key = Python.tuple(py_call(b.id, result), mode)
    var seen = py_get(checked, key)
    if seen is not None and seen is result:
        return result
    var mode_name = String("mojolearn_numeric_mode") if group == "base" else String("x_neighbors_numeric_mode") if group == "sort" else String("ivf_numeric_mode")
    var getter = knn_door(result, mode_name)
    if getter is not None and Int(py=family_call(getter)) != (1 if String(mode) == "identical" else 0):
        raise Error("RuntimeError: neighbor binding reports a different compile-time numeric mode")
    py_setitem(checked, key, result)
    return result


def _resident_owner(cached: PythonObject, context: PythonObject) raises -> PythonObject:
    if cached is None:
        return cached
    var b = native_loaded_module("builtins")
    if py_type(cached) is not b.tuple or len(cached) != 3 or py_type(cached[1]) is not b.int:
        raise Error("TypeError: native resident state was replaced")
    var connected = py_get(context["neighbor_query_binding_aliases"], py_call(b.id, cached[2]))
    if connected is None or connected[0] is not cached[2]:
        raise Error("TypeError: resident binding must be an import-time connected native owner")
    classical_admit_value(cached[0], context)
    if cached[2] is connected[1]:
        return cached  # unchanged identity keeps the admission stamp valid
    return Python.tuple(cached[0], cached[1], connected[1])


comptime _RADIUS_STATE_SLOT = "neighbor_radius_state_v1"


def _ivf_state_names() raises -> PythonObject:
    """The fitted array owners then the scalar configuration slots of the
    IVF/k-NN admission stamp, in `_admit_state`'s order."""
    var names = Python.tuple("_index", "centers_", "center_norms_", "list_offsets_", "list_indices_", "list_data_")
    var scalars = Python.tuple("n_samples_fit_", "n_features_in_", "n_rows_", "n_lists_", "metric_code_", "metric", "p", "algorithm", "query_tile", "_metric_value", "_metric_arg")
    return names + scalars


def _ivf_state_stamp(fields: PythonObject) raises -> PythonObject:
    """The IVF prepared-registry stamp of `fields` as it stands (the slot
    objects, then the stored `_resident`), as `_admit_state` writes it when
    the resident tuple is unchanged."""
    return knn_state_values(fields, _ivf_state_names()) + Python.tuple(py_get(fields, "_resident"))


def _admit_state(model: PythonObject, kind: PythonObject, context: PythonObject) raises -> Bool:
    """Admit the fitted state, once while it is unchanged: True when every
    fitted array, scalar and resident tuple is the object admitted before
    (or has just been admitted), so the query reads only its own input.
    k-NN stamps live in the resident holder (created by fit, never pickled);
    IVF and radius stamps in the native prepared registry, weakly keyed by the
    model and outside `__dict__` (IVF's is dropped by fit and extend)."""
    var fields = py_fields(model)
    var b = native_loaded_module("builtins")
    var names = Python.tuple("_index", "centers_", "center_norms_", "list_offsets_", "list_indices_", "list_data_")
    var scalars = Python.tuple("n_samples_fit_", "n_features_in_", "n_rows_", "n_lists_", "metric_code_", "metric", "p", "algorithm", "query_tile", "_metric_value", "_metric_arg")
    var text = String(kind[0])
    var holder = PythonObject(None)
    if text == "knn":
        holder = py_get(fields, "_resident_holder")
        if holder is not None:
            if py_type(holder) is not b.dict:
                raise Error("TypeError: neighbor resident holder requires a native dictionary")
            var keys = py_call(b.tuple, holder.keys())
            for i in range(len(keys)):  # driver-loop: resident metadata keys, before dictionary comparisons
                if py_type(keys[i]) is not b.str:
                    raise Error("TypeError: resident metadata requires exact string keys")
    var mods = NativeModules()
    if text == "ivf" or holder is not None:
        var resident = py_get(holder, "index") if holder is not None else py_get(fields, "_resident")
        var stamp = knn_state_values(fields, names + scalars) + Python.tuple(resident)
        var stored = py_get(holder, "admitted_query") if holder is not None else prepared_get(mods, IVF_QUERY_STATE_SLOT, model)
        if knn_state_unchanged(stored, stamp):
            return True
    # Radius models have no resident holder: their stamp (every slot object
    # and exact container items) lives in the native prepared registry,
    # weakly keyed by the model and outside `__dict__`.
    var radius_stamp = PythonObject(None)
    if text == "radius":
        radius_stamp = Python.list()
        for i in range(len(names)):  # small-loop(fixed fitted array owners)
            prepared_key_add(mods, radius_stamp, py_get(fields, names[i]))
        for i in range(len(scalars)):  # small-loop(fixed native query configuration)
            prepared_key_add(mods, radius_stamp, py_get(fields, scalars[i]))
        var record = prepared_get(mods, _RADIUS_STATE_SLOT, model)
        if record is not None and prepared_key_same(record, radius_stamp):
            return True
    for i in range(len(names)):  # small-loop(fixed fitted array owners)
        var value = py_get(fields, names[i])
        if value is not None:
            classical_admit_value(value, context)
            if not classical_is_array(value, context) and py_type(value) is not py_get(context, "ndarray"):
                raise Error("TypeError: fitted neighbor arrays require native array storage")
            if not classical_c_order(value, context):
                raise Error("ValueError: fitted neighbor arrays require C-contiguous storage")
    for i in range(len(scalars)):  # small-loop(fixed native query configuration)
        var value = py_get(fields, scalars[i])
        if value is not None:
            host_inference_scalar(value, context)
    if text == "ivf":
        # The query leaves `__dict__` unchanged: `_resident` is rewritten only
        # when the connected binding alias replaced the stored one, and the
        # admission stamp goes to the native prepared registry (never pickled).
        var cached = py_get(fields, "_resident")
        var owner = _resident_owner(cached, context)
        if owner is not cached:
            py_setitem(fields, "_resident", owner)
        var ivf_stamp = knn_state_values(fields, names + scalars) + Python.tuple(owner)
        var ivf_self_reference = False
        for i in range(len(ivf_stamp)):  # driver-loop: over the stamp objects, refusing a self-reference
            if ivf_stamp[i] is model:
                ivf_self_reference = True
                break
        if not ivf_self_reference:
            prepared_put(mods, IVF_QUERY_STATE_SLOT, model, ivf_stamp)
    elif text == "knn" and holder is not None:
        py_setitem(holder, "index", _resident_owner(py_get(holder, "index"), context))
        py_setitem(holder, "admitted_query", knn_state_values(fields, names + scalars) + Python.tuple(py_get(holder, "index")))
    elif text == "radius":
        var self_reference = False
        for i in range(len(radius_stamp)):  # driver-loop: over the stamp objects, refusing a self-reference
            if radius_stamp[i] is model:
                self_reference = True
                break
        if not self_reference:
            prepared_put(mods, _RADIUS_STATE_SLOT, model, radius_stamp)
    return True


def _ivf_key_entry(fields: PythonObject, name: String) raises -> PythonObject:
    var value = py_get(fields, name)
    if value is None:
        return Python.tuple("device")
    var b = native_loaded_module("builtins")
    return Python.tuple(py_call(b.id, value), classical_address(value), py_call(b.tuple, value.shape))


def _ivf_materialize(fields: PythonObject, context: PythonObject) raises:
    if not py_contains(fields, "_device_list_data"):
        return
    var cached = py_get(fields, "_resident")
    if cached is None:
        raise Error("RuntimeError: resident IVF index was released before its list data was read")
    var key = cached[0]
    var tuple_type = native_loaded_module("builtins").tuple
    if py_type(key) is not tuple_type or len(key) != 7 or py_type(key[0]) is not tuple_type or len(key[0]) != 5:
        raise Error("TypeError: resident IVF index requires its native storage key")
    var n = Int(py=fields["n_rows_"])
    var dim = Int(py=fields["n_features_in_"])
    var data = family_array(Python.tuple(n, dim), "<f4", context)
    _ = family_call(family_entry(cached[2], "ivf_flat_index_export", context), cached[1], classical_address(data), PythonObject(n * dim))
    py_setitem(fields, "list_data_", data)
    _ = py_call(fields.pop, "_device_list_data")
    # Export changes only the device-data marker. Preserve every other saved
    # key so a caller's replaced centers/offsets still invalidate this handle.
    var arrays = Python.tuple(key[0][0], key[0][1], key[0][2], key[0][3], _ivf_key_entry(fields, "list_data_"))
    var replacement = Python.tuple(arrays, key[1], key[2], key[3], key[4], key[5], key[6])
    py_setitem(fields, "_resident", Python.tuple(replacement, cached[1], cached[2]))


def _ivf_filter(source: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    if source is None:
        return PythonObject(None)
    classical_admit_value(source, context)
    var b = native_loaded_module("builtins")
    var value = source
    var bool_source = False
    var cls = py_type(value)
    if cls is b.list or cls is b.tuple:
        var packed = sequential_sequence(value, context)
        bool_source = String(packed[1]) == "b"
    elif native_pandas_exact(value):
        if cls is not _pd_class("pandas.core.series", "Series"):
            raise Error("ValueError: IVF filter must be a one-dimensional boolean array")
        var manager = py_get(_pd_fields(value), "_mgr")
        if py_type(manager) is not _pd_class("pandas.core.internals.managers", "SingleBlockManager"):
            raise Error("TypeError: IVF filter requires native pandas storage")
        var blocks = py_get(_pd_fields(manager), "blocks")
        if py_type(blocks) is not b.tuple or len(blocks) != 1:
            raise Error("ValueError: IVF filter requires one native boolean column")
        var block = blocks[0]
        var block_cls = py_type(block)
        if block_cls is not _pd_class("pandas.core.internals.blocks", "NumpyBlock") and block_cls is not _pd_class("pandas.core.internals.blocks", "NumericBlock") and block_cls is not _pd_class("pandas.core.internals.blocks", "ExtensionBlock"):
            raise Error("TypeError: IVF filter requires ordinary native pandas blocks")
        value = _pd_values(_pd_get("pandas._libs.internals", "Block", "values", block), True, context)[0]
        bool_source = String(native_buffer_view(value).format) == "?"
    else:
        if _ordinary_torch(cls, context):
            value = native_torch_numpy(value)
        if py_type(value) is not context["array_class"]:
            bool_source = String(native_buffer_view(value).format) == "?"
    if not bool_source:
        raise Error("ValueError: IVF filter must have boolean dtype")
    var view = labels_view(value, "filter", context)
    labels_check_rank(view, 1, "filter")
    if view.size() != n:
        raise Error("ValueError: IVF filter must have one flag per indexed row")
    return _i32_vector(value, "filter", context)


def _ivf_search(model: PythonObject, kind: PythonObject, args: PythonObject, admitted: Bool,
                context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    if not py_contains(fields, "list_data_") and not py_contains(fields, "_device_list_data"):
        raise Error("ValueError: call fit or load before search")
    var q = family_f32_snapshot(args[0], 2, "queries", context)
    var fitted = List[Int]([Int(py=fields["n_rows_"]), Int(py=fields["n_features_in_"]),
                           Int(py=fields["n_lists_"]), Int(py=fields["metric_code_"])])
    var dim = fitted[1]
    if q.shape[1] != dim:
        raise Error("ValueError: query feature count differs from the fitted index")
    var names = Python.tuple("n_lists", "n_probes", "n_neighbors", "kmeans_n_iters", "random_state")
    for i in range(len(names)):  # small-loop(five public integer parameters)
        if py_type(fields[names[i]]) is not b.int:
            raise Error("TypeError: " + String(names[i]) + " must be an int")
    var metric = fields["metric"]
    if py_type(metric) is not b.str and py_type(metric) is not b.int and py_type(metric) is not b.float:
        raise Error("TypeError: metric requires a native name or code")
    if Int(py=_ivf_metric(fields, context)) != fitted[3]:
        raise Error("ValueError: metric differs from the fitted index")
    var binding = _query_route(model, kind, "ivf", context)
    # Perf (perf2-neighbors): `_query_route` already checked this binding's
    # `ivf_numeric_mode` getter against the requested canonical mode (once per
    # process; codes identical=1, fast=0 as `_MODE_CODE`), so a canonical
    # request needs no second getter call per search. A non-canonical spelling
    # has no `_MODE_CODE` entry and keeps its refusal when the getter exists.
    var requested = py_get(fields, "numeric_mode")
    if requested is None:
        requested = PythonObject("identical") if Bool(py=kind[1]) else context["backend_fields"]["_DEFAULT_MODE"]
    var requested_text = String(requested)
    if requested_text != "identical" and requested_text != "fast":
        var mode = knn_door(binding, "ivf_numeric_mode")
        if mode is not None:
            var expected = py_get(context["family_xn_ann_ivf_mode_codes"], requested)
            if expected is None or Int(py=family_call(mode)) != Int(py=expected):
                raise Error("RuntimeError: IVF binding numeric mode differs from the requested mode")
    var n = fitted[0]
    var m = q.shape[0]
    var k = Int(py=fields["n_neighbors"])
    if k < 1:
        raise Error("ValueError: n_neighbors must be positive")
    var dist = family_array(Python.tuple(m, k), "<f4", context)
    var ids = family_array(Python.tuple(m, k), "<i4", context)
    var count = family_array(Python.tuple(m), "<i4", context)
    var keep = _ivf_filter(args[1], n, context)
    var stored_snapshots = _ivf_search_snapshots(fields, fitted, admitted, context)
    var handle = _ivf_snapshot_resident(fields, binding, stored_snapshots, fitted, context)
    var addresses = Python.list(q.source.addr, classical_address(dist), classical_address(ids), classical_address(count))
    var sizes = Python.list(n, dim, fitted[2], fitted[3], m, k, fields["n_probes"])
    if handle is None:
        var stored = _ivf_snapshot_addresses(stored_snapshots)
        _ = py_call(stored.extend, addresses)
        addresses = stored
    else:
        _ = py_call(sizes.append, 0)
    if keep is not None:
        _ = py_call(addresses.append, classical_address(keep))
    if handle is None:
        _ = family_call(family_entry(binding, "ivf_flat_search", context), addresses, sizes)
    else:
        _ = family_call(family_entry(binding, "ivf_flat_index_search", context), handle, addresses, sizes)
    _ = q
    _ = stored_snapshots
    _ = keep
    py_setitem(fields, "n_candidates_", count)
    return Python.tuple(dist, ids)


def _ivf_stored_snapshot(fields: PythonObject, name: String, expected: List[Int],
                         kind: Int, context: PythonObject, admitted: Bool = False) raises -> ArrayMetadata:
    # Fitted arrays are mutable public state. Admit exact storage before any
    # metadata read, then carry its owner and descriptor through the operation.
    # `admitted`: this exact object passed `_admit_state` (unchanged since).
    var value = py_get(fields, name)
    if not admitted:
        classical_admit_value(value, context)
    if not classical_is_array(value, context) and py_type(value) is not py_get(context, "ndarray"):
        raise Error("TypeError: " + name + " requires native array storage")
    var shape = array_shape_once(value, context)
    if len(shape) != len(expected):
        raise Error("ValueError: " + name + " has the wrong fitted shape")
    for i in range(len(shape)):  # small-loop(fitted array rank)
        if shape[i] != expected[i]:
            raise Error("ValueError: " + name + " has the wrong fitted shape")
    var snapshot = array_metadata_once(value, shape^, name, context)
    if not snapshot.c_order:
        raise Error("ValueError: fitted neighbor arrays require C-contiguous storage")
    if not snapshot.native_byte_order or snapshot.source.kind != kind or snapshot.source.itemsize != 4:
        raise Error("TypeError: " + name + " requires its C-contiguous 32-bit fitted dtype")
    return snapshot^


def _ivf_search_snapshots(fields: PythonObject, fitted: List[Int], admitted: Bool,
                          context: PythonObject) raises -> List[ArrayMetadata]:
    # Each fitted exporter is admitted and captured once for this search.
    # These retained owners cover cache-key construction, optional upload and
    # execution even if mutable Array slots are replaced during native work.
    var n = fitted[0]
    var dim = fitted[1]
    var lists = fitted[2]
    if n < 1 or dim < 1 or lists < 1 or lists > n:
        raise Error("ValueError: invalid fitted IVF index dimensions")
    var snapshots = List[ArrayMetadata]()
    snapshots.append(_ivf_stored_snapshot(fields, "centers_", [lists, dim], CAST_KIND_FLOAT, context, admitted))
    snapshots.append(_ivf_stored_snapshot(fields, "center_norms_", [lists], CAST_KIND_FLOAT, context, admitted))
    snapshots.append(_ivf_stored_snapshot(fields, "list_offsets_", [lists + 1], CAST_KIND_SIGNED, context, admitted))
    snapshots.append(_ivf_stored_snapshot(fields, "list_indices_", [n], CAST_KIND_SIGNED, context, admitted))
    if py_get(fields, "list_data_") is not None:
        snapshots.append(_ivf_stored_snapshot(fields, "list_data_", [n, dim], CAST_KIND_FLOAT, context, admitted))
    return snapshots^


def _ivf_snapshot_key(fields: PythonObject, binding: PythonObject,
                      snapshots: List[ArrayMetadata], fitted: List[Int]) raises -> PythonObject:
    var keys = Python.list()
    var b = native_loaded_module("builtins")
    for i in range(len(snapshots)):  # small-loop(five captured fitted buffer keys)
        _ = py_call(keys.append, Python.tuple(py_call(b.id, snapshots[i].keep),
            snapshots[i].source.addr, snapshots[i].python_shape()))
    if len(snapshots) == 4:
        _ = py_call(keys.append, Python.tuple("device"))
    return Python.tuple(py_call(b.tuple, keys), fitted[0], fitted[1], fitted[2], fitted[3], False, py_call(b.id, binding))


def _ivf_key_equal(left: PythonObject, right: PythonObject) raises -> Bool:
    # Cache metadata is native tuples, scalar words and the device marker.
    # Exact admission prevents tuple equality from invoking scalar protocols.
    var kind = py_type(left)
    if kind is not py_type(right):
        return False
    var b = native_loaded_module("builtins")
    if kind is b.int:
        return Int(py=left) == Int(py=right)
    if kind is b.bool:
        return Bool(py=left) == Bool(py=right)
    if kind is b.str:
        return String(left) == String(right)
    if kind is not b.tuple:
        raise Error("TypeError: resident IVF key requires exact native tuple/int/bool/string metadata")
    if len(left) != len(right):
        return False
    for i in range(len(left)):  # small-loop(fixed resident IVF key metadata)
        if not _ivf_key_equal(left[i], right[i]):
            return False
    return True


def _ivf_snapshot_materialize(fields: PythonObject, mut snapshots: List[ArrayMetadata],
                              fitted: List[Int], context: PythonObject) raises:
    if not py_contains(fields, "_device_list_data"):
        return
    var cached = py_get(fields, "_resident")
    if cached is None:
        raise Error("RuntimeError: resident IVF index was released before its list data was read")
    var key = cached[0]
    var b = native_loaded_module("builtins")
    if py_type(key) is not b.tuple or len(key) != 7 or py_type(key[0]) is not b.tuple or len(key[0]) != 5:
        raise Error("TypeError: resident IVF index requires its native storage key")
    # Capture only the fresh export output; reuse every fitted input snapshot.
    var n = fitted[0]
    var dim = fitted[1]
    var data = family_array(Python.tuple(n, dim), "<f4", context)
    var output = array_metadata_once(data, [n, dim], "list_data_", context)
    _ = family_call(family_entry(cached[2], "ivf_flat_index_export", context), cached[1],
                    PythonObject(output.source.addr), PythonObject(n * dim))
    py_setitem(fields, "list_data_", data)
    _ = py_call(fields.pop, "_device_list_data")
    # Only replace the device marker, preserving stale keys for other arrays
    # so replacing public fitted storage still invalidates the resident handle.
    var data_key = Python.tuple(py_call(b.id, data), output.source.addr, output.python_shape())
    var arrays = Python.tuple(key[0][0], key[0][1], key[0][2], key[0][3], data_key)
    var replacement = Python.tuple(arrays, key[1], key[2], key[3], key[4], key[5], key[6])
    py_setitem(fields, "_resident", Python.tuple(replacement, cached[1], cached[2]))
    if len(snapshots) == 4:
        snapshots.append(output^)
    else:
        snapshots[4] = output^


def _ivf_snapshot_addresses(snapshots: List[ArrayMetadata]) raises -> PythonObject:
    if len(snapshots) != 5:
        raise Error("RuntimeError: fitted IVF vector storage is unavailable")
    var result = Python.list()
    for i in range(len(snapshots)):  # small-loop(five captured IVF storage addresses)
        _ = py_call(result.append, snapshots[i].source.addr)
    return result


def _ivf_snapshot_resident(fields: PythonObject, binding: PythonObject,
                           mut snapshots: List[ArrayMetadata], fitted: List[Int], context: PythonObject) raises -> PythonObject:
    var prepare = knn_door(binding, "ivf_flat_index_prepare")
    if prepare is None:
        _ivf_snapshot_materialize(fields, snapshots, fitted, context)
        return PythonObject(None)
    var key = _ivf_snapshot_key(fields, binding, snapshots, fitted)
    var cached = py_get(fields, "_resident")
    if cached is not None and _ivf_key_equal(cached[0], key):
        return cached[1]
    _ivf_snapshot_materialize(fields, snapshots, fitted, context)
    _ann_release(fields, "ivf_flat_index_release")
    # p3-neighbors-cluster (hazard class A): the resident key holds identity,
    # address and shape, never values, so the fitted arrays are made read-only
    # when the device copy is taken (bindings/p3nc_fitted_freeze.mojo). A
    # caller-assigned array that cannot be frozen keeps the per-call search.
    for i in range(len(snapshots)):  # small-loop(five captured fitted arrays)
        if not fitted_is_freezable(snapshots[i].keep, context):
            return PythonObject(None)
    for i in range(len(snapshots)):  # small-loop(five captured fitted arrays)
        _ = fitted_freeze(snapshots[i].keep, context)
    var handle = family_call(prepare, _ivf_snapshot_addresses(snapshots), Python.list(fitted[0], fitted[1], fitted[2], fitted[3], 0))
    py_setitem(fields, "_resident", Python.tuple(_ivf_snapshot_key(fields, binding, snapshots, fitted), handle, binding))
    return handle


def _ivf_extend(model: PythonObject, kind: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    # Perf (perf2-neighbors): a stamp hit (the same fitted objects a search,
    # a clone or `_admit_state` admitted) skips the per-array admission of
    # the four stored arrays and `list_data_`; their shape/dtype/C-order
    # checks below still run. Read before the drop.
    var stamp_mods = NativeModules()
    var admitted = knn_state_unchanged(prepared_get(stamp_mods, IVF_QUERY_STATE_SLOT, model), _ivf_state_stamp(fields))
    # Extend replaces the fitted arrays: drop the admission stamp (and the
    # old arrays it holds); the next search admits the extended index once.
    prepared_drop(stamp_mods, IVF_QUERY_STATE_SLOT, model)
    var b = native_loaded_module("builtins")
    if not py_contains(fields, "list_data_") and not py_contains(fields, "_device_list_data"):
        raise Error("ValueError: mojolearn IVFIndex: call fit (or load) before extend")
    var x = family_f32_snapshot(args[0], 2, "X", context)
    var names = Python.tuple("n_rows_", "n_features_in_", "n_lists_", "metric_code_")
    for i in range(len(names)):  # small-loop(four fitted integer parameters)
        if py_type(fields[names[i]]) is not b.int:
            raise Error("TypeError: " + String(names[i]) + " must be an exact native integer")
    var n = Int(py=fields["n_rows_"])
    var dim = Int(py=fields["n_features_in_"])
    var n_lists = Int(py=fields["n_lists_"])
    var metric = Int(py=fields["metric_code_"])
    if n < 1 or dim < 1 or n_lists < 1 or n_lists > n or (metric != 0 and metric != 1):
        raise Error("ValueError: invalid fitted IVF index dimensions or metric")
    var m = x.shape[0]
    if x.shape[1] != dim:
        raise Error("ValueError: mojolearn IVFIndex: X has " + String(x.shape[1])
                    + " features, the index has " + String(dim))
    if Int(py=_ivf_metric(fields, context)) != metric:
        raise Error("ValueError: mojolearn IVFIndex: metric does not name the metric this index was built under")
    var centers = _ivf_stored_snapshot(fields, "centers_", [n_lists, dim], CAST_KIND_FLOAT, context, admitted)
    var norms = _ivf_stored_snapshot(fields, "center_norms_", [n_lists], CAST_KIND_FLOAT, context, admitted)
    var old_offsets = _ivf_stored_snapshot(fields, "list_offsets_", [n_lists + 1], CAST_KIND_SIGNED, context, admitted)
    var old_indices = _ivf_stored_snapshot(fields, "list_indices_", [n], CAST_KIND_SIGNED, context, admitted)
    var cached = _resident_owner(py_get(fields, "_resident"), context)
    py_setitem(fields, "_resident", cached)
    if py_contains(fields, "_device_list_data"):
        if cached is None:
            raise Error("RuntimeError: resident IVF index was released before its list data was read")
        var exported = family_array(Python.tuple(n, dim), "<f4", context)
        _ = family_call(family_entry(cached[2], "ivf_flat_index_export", context), cached[1],
                        classical_address(exported), PythonObject(n * dim))
        py_setitem(fields, "list_data_", exported)
        _ = py_call(fields.pop, "_device_list_data")
    # The device export above is a fresh native Array; any other list_data_
    # is the stamped object when `admitted`.
    var old_data = _ivf_stored_snapshot(fields, "list_data_", [n, dim], CAST_KIND_FLOAT, context, admitted)
    var binding = _query_route(model, kind, "ivf", context)
    var entry = family_entry(binding, "ivf_flat_extend", context)
    var total = n + m
    var offsets = family_array(Python.tuple(n_lists + 1), "<i4", context)
    var indices = family_array(Python.tuple(total), "<i4", context)
    var data = family_array(Python.tuple(total, dim), "<f4", context)
    var labels = family_array(Python.tuple(m), "<i4", context)
    _ann_release(fields, "ivf_flat_index_release")
    # Existing compiled IVF algorithm and address order; no arithmetic changes.
    _ = family_call(entry, Python.list(centers.source.addr, norms.source.addr,
        old_offsets.source.addr, old_indices.source.addr, old_data.source.addr, x.source.addr,
        classical_address(offsets), classical_address(indices), classical_address(data), classical_address(labels)),
        Python.list(n, dim, n_lists, metric, m))
    _ = x
    _ = centers
    _ = norms
    _ = old_offsets
    _ = old_indices
    _ = old_data
    py_setitem(fields, "list_offsets_", offsets)
    py_setitem(fields, "list_indices_", indices)
    py_setitem(fields, "list_data_", data)
    py_setitem(fields, "n_rows_", PythonObject(total))
    py_setitem(fields, "extend_labels_", labels)
    return model


def _ivf_clone(model: PythonObject, kind: PythonObject, context: PythonObject) raises -> PythonObject:
    """`IVFIndex._clone` (cuVS `ivf_flat::clone`'s role): a new instance of
    the model's class holding copies of the five index arrays, so an extend
    on it leaves this one as it was.

    Perf (perf2-neighbors): formerly Python glue making about 18 hooked
    attribute reads, five `tobytes`/`frombytes` round trips and an
    admission per array on every call. Now the fitted state is admitted
    through the IVF prepared-registry stamp (one compare while unchanged),
    each array is captured once and copied with one `memcpy` into a new
    Array, and the clone's `__dict__` gets the same keys, in the same order,
    as the Python version produced (constructor parameters, `numeric_mode`,
    the five arrays, then `n_features_in_`, `n_rows_`, `n_lists_`,
    `metric_code_`). A resident build's device-only list vectors are
    exported into this index first, as the former `getattr(self,
    "list_data_")` did."""
    var fields = py_fields(model)
    if not py_contains(fields, "list_data_") and not py_contains(fields, "_device_list_data"):
        raise Error("ValueError: mojolearn IVFIndex: call fit (or load) before _clone")
    var admitted = _admit_state(model, kind, context)
    var device = py_contains(fields, "_device_list_data")
    if device:
        _ivf_materialize(fields, context)
    var b = native_loaded_module("builtins")
    var counts = Python.tuple("n_features_in_", "n_rows_", "n_lists_", "metric_code_")
    for i in range(len(counts)):  # small-loop(four fitted integer parameters)
        if py_type(py_get(fields, counts[i])) is not b.int:
            raise Error("TypeError: " + String(counts[i]) + " must be an exact native integer")
    var fitted = List[Int]([Int(py=fields["n_rows_"]), Int(py=fields["n_features_in_"]),
                           Int(py=fields["n_lists_"]), Int(py=fields["metric_code_"])])
    # The freshly exported `list_data_` is not covered by the stamp: admit
    # every array again in that (first-clone-only) case.
    var snapshots = _ivf_search_snapshots(fields, fitted, admitted and not device, context)
    if len(snapshots) != 5:
        raise Error("RuntimeError: fitted IVF vector storage is unavailable")
    var keywords = Python.dict()
    var params = Python.tuple("n_lists", "n_probes", "n_neighbors", "kmeans_n_iters", "metric", "random_state")
    for i in range(len(params)):  # small-loop(six constructor parameters)
        py_setitem(keywords, params[i], py_get(fields, params[i]))
    # The class's own constructor (constructor-time parameter handling).
    var clone = family_call_kw(py_type(model), keywords)
    var target = py_fields(clone)
    py_setitem(target, "numeric_mode", py_get(fields, "numeric_mode"))
    var names = Python.tuple("centers_", "center_norms_", "list_offsets_", "list_indices_", "list_data_")
    var dtypes: List[String] = ["<f4", "<f4", "<i4", "<i4", "<f4"]
    for i in range(len(snapshots)):  # small-loop(five fitted index arrays, one memcpy each)
        var copy = family_array(snapshots[i].python_shape(), dtypes[i], context)
        if snapshots[i].count > 0:
            memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(copy))),
                   src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=snapshots[i].source.addr),
                   count=snapshots[i].count * 4)
        py_setitem(target, names[i], copy)
    _ = snapshots
    for i in range(len(counts)):  # small-loop(four fitted integer parameters)
        py_setitem(target, counts[i], fields[counts[i]])
    # The clone holds fresh native Arrays and the source's admitted scalar
    # objects: stamp it, so its first extend or search admits nothing again.
    var mods = NativeModules()
    prepared_put(mods, IVF_QUERY_STATE_SLOT, clone, _ivf_state_stamp(target))
    return clone


def neighbor_query(model: PythonObject, op: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    var kind = _query_kind(model, context)
    if op == "extend" and String(kind[0]) == "ivf":
        return _ivf_extend(model, kind, args, context)
    if op == "clone" and String(kind[0]) == "ivf":
        return _ivf_clone(model, kind, context)
    var admitted = _admit_state(model, kind, context)
    if op == "getstate" and String(kind[0]) == "ivf":
        var fields = py_fields(model)
        _ivf_materialize(fields, context)
        var state = py_call(native_loaded_module("builtins").dict, fields)
        _ = py_call(state.pop, "_resident", PythonObject(None))
        return state
    if op == "search" and String(kind[0]) == "ivf":
        return _ivf_search(model, kind, args, admitted, context)
    var binding = _query_route(model, kind, "base", context)
    if op == "kneighbors" and String(kind[0]) == "knn":
        return parallel_public_result(neighbor_kneighbors(model, args[0], args[1], args[2], binding, context), op, PythonObject(None), context)
    if op == "radius_neighbors" and String(kind[0]) == "radius":
        var kwargs = Python.dict()
        py_setitem(kwargs, "radius", args[1])
        py_setitem(kwargs, "return_distance", args[2])
        py_setitem(kwargs, "sort_results", args[3])
        var tables = Python.dict()
        var entries = Python.dict()
        # Resolve the fixed sort route only when requested; an unsorted CPU
        # query never needs the optional expansion binary.
        if _flag(args[3], "sort_results", context):
            py_setitem(entries, "_mojolearn_x_neighbors", _query_route(model, kind, "sort", context))
        py_setitem(tables, "worker_bindings", entries)
        return _radius(model, args[0], kwargs, binding, tables, context)
    raise Error("TypeError: unsupported native neighbor query for this model")


def neighbor_query_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        return neighbor_query(args[0], String(args[1]), args[2], args[3]).steal_data()
    except error:
        # family_call stashes the native binding's Python exception. Restore
        # its original type/message instead of exposing the sentinel (the
        # malformed IVF-index public test exercises this boundary).
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error^)
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if String(error).startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)


def neighbor_query_attribute_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        var context = args[0]
        var model = args[1]
        var name = args[2]
        classical_connect_numpy(context)
        # Perf: every attribute read on a connected neighbor model lands here
        # (about 20 per _clone()). The public read below goes through
        # classical_getattribute, which runs classical_check_layout itself,
        # so the ordinary read checks the layout ONCE instead of twice. The
        # branches that answer or materialize before that read keep their own
        # check first, and an unconnected class still refuses through
        # _query_kind (layout refusal first, then the kind refusal).
        var kind = py_get(context["neighbor_query_classes"], py_type(model))
        if kind is None:
            kind = _query_kind(model, context)
        if py_type(name) is not native_loaded_module("builtins").str:
            raise Error("TypeError: attribute name must be a string")
        var text = String(name)
        var family = String(kind[0])
        var early = (family == "ivf" and text == "list_data_") or (family == "ann_pq" and text == "codes_") or (
            family == "knn" and (text == "_resident" or text == "used_query_tile_"
                                 or text == "n_candidate_distances_" or text == "classes_"))
        if early:
            classical_check_layout(model, context)
        var fields = py_fields(model) if early else PythonObject(None)
        if family == "ivf" and text == "list_data_":
            _ = _admit_state(model, kind, context)
            _ivf_materialize(fields, context)
        if family == "ann_pq" and text == "codes_":
            _ = ann_pq_materialize(model, context)
        if family == "knn":
            if text == "_resident":
                return py_get(py_get(fields, "_resident_holder", Python.dict()), "index").steal_data()
            if text == "used_query_tile_" or text == "n_candidate_distances_":
                var holder = py_get(fields, "_resident_holder", Python.dict())
                if not py_contains(holder, name):
                    raise Error("AttributeError: query diagnostic is not available")
                return holder[name].steal_data()
            if text == "classes_" and Bool(py=py_get(context["neighbor_query_classifier_classes"], py_type(model), False)):
                if py_get(fields, "_y_cols") is None:
                    raise Error("AttributeError: classes_ is set by fit")
                var labels = py_get(fields, "_label_names")
                if labels is not None:
                    return labels.steal_data()
                var classes = fields["_classes_list"]
                if Bool(py=fields["outputs_2d_"]):
                    return classes.steal_data()
                return classes[0].steal_data()
        var result = classical_getattribute(model, name, context)
        if not _outside(context, 0):
            if not early:
                fields = py_fields(model)
            var original = py_get(fields, name)
            if classical_is_array(original, context):
                result = original
        return result.steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error^)
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
