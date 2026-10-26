# SPDX-License-Identifier: Apache-2.0
# Complete standalone queries, sharing the worker's native search ownership.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_loaded_module import native_loaded_module
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
from bindings.family_common import family_array, family_as_f32, family_entry, family_call
from bindings.neighbors_knn_native import knn_door
from bindings.neighbors_ann_native import _ivf_metric, _ann_release
from bindings.parallel_neighbor_worker import neighbor_kneighbors, _radius, _i32_vector, _flag
from bindings.host_inference_scalars import host_inference_scalar
from bindings.parallel_public_result import parallel_public_result
from bindings.scaler_lifecycle_native import _outside


def _query_kind(model: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_check_layout(model, context)
    var kind = py_get(context["neighbor_query_classes"], py_type(model))
    if kind is None:
        raise Error("TypeError: neighbor query requires an exact connected native model")
    return kind


def _query_route(model: PythonObject, group: String, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var kind = _query_kind(model, context)
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
    mode = py_call(py_call(mode.strip).lower)
    if String(mode) != "identical" and String(mode) != "fast":
        raise Error("ValueError: numeric_mode must be identical or fast")
    var routes = context["neighbor_query_host_routes"] if host else context["neighbor_query_routes"]
    var result = routes[group][mode]
    if py_type(result) is b.str:
        raise Error("ImportError: " + String(result))
    var mode_name = String("mojolearn_numeric_mode") if group == "base" else String("x_neighbors_numeric_mode") if group == "sort" else String("ivf_numeric_mode")
    var getter = knn_door(result, mode_name)
    if getter is not None and Int(py=family_call(getter)) != (1 if String(mode) == "identical" else 0):
        raise Error("RuntimeError: neighbor binding reports a different compile-time numeric mode")
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
    return Python.tuple(cached[0], cached[1], connected[1])


def _admit_state(model: PythonObject, kind: PythonObject, context: PythonObject) raises:
    var fields = py_fields(model)
    var names = Python.tuple("_index", "centers_", "center_norms_", "list_offsets_", "list_indices_", "list_data_")
    for i in range(len(names)):  # small-loop(fixed fitted array owners)
        var value = py_get(fields, names[i])
        if value is not None:
            classical_admit_value(value, context)
            if not classical_is_array(value, context) and py_type(value) is not py_get(context, "ndarray"):
                raise Error("TypeError: fitted neighbor arrays require native array storage")
            if not classical_c_order(value, context):
                raise Error("ValueError: fitted neighbor arrays require C-contiguous storage")
    var scalars = Python.tuple("n_samples_fit_", "n_features_in_", "n_rows_", "n_lists_", "metric_code_", "metric", "p", "algorithm", "query_tile", "_metric_value", "_metric_arg")
    for i in range(len(scalars)):  # small-loop(fixed native query configuration)
        var value = py_get(fields, scalars[i])
        if value is not None:
            host_inference_scalar(value, context)
    if String(kind[0]) == "ivf":
        py_setitem(fields, "_resident", _resident_owner(py_get(fields, "_resident"), context))
    elif String(kind[0]) == "knn":
        var holder = py_get(fields, "_resident_holder")
        if holder is not None:
            if py_type(holder) is not native_loaded_module("builtins").dict:
                raise Error("TypeError: neighbor resident holder requires a native dictionary")
            var keys = py_call(native_loaded_module("builtins").tuple, holder.keys())
            for i in range(len(keys)):  # driver-loop: resident metadata keys, before dictionary comparisons
                if py_type(keys[i]) is not native_loaded_module("builtins").str:
                    raise Error("TypeError: resident metadata requires exact string keys")
            py_setitem(holder, "index", _resident_owner(py_get(holder, "index"), context))


def _ivf_key_entry(fields: PythonObject, name: String) raises -> PythonObject:
    var value = py_get(fields, name)
    if value is None:
        return Python.tuple("device")
    var b = native_loaded_module("builtins")
    return Python.tuple(py_call(b.id, value), classical_address(value), py_call(b.tuple, value.shape))


def _ivf_key(fields: PythonObject, binding: PythonObject) raises -> PythonObject:
    var arrays = Python.tuple(_ivf_key_entry(fields, "centers_"), _ivf_key_entry(fields, "center_norms_"),
        _ivf_key_entry(fields, "list_offsets_"), _ivf_key_entry(fields, "list_indices_"), _ivf_key_entry(fields, "list_data_"))
    return Python.tuple(arrays, fields["n_rows_"], fields["n_features_in_"], fields["n_lists_"],
        fields["metric_code_"], False, py_call(native_loaded_module("builtins").id, binding))


def _ivf_materialize(fields: PythonObject, context: PythonObject) raises:
    if not py_contains(fields, "_device_list_data"):
        return
    var cached = py_get(fields, "_resident")
    if cached is None:
        raise Error("RuntimeError: resident IVF index was released before its list data was read")
    var n = Int(py=fields["n_rows_"])
    var dim = Int(py=fields["n_features_in_"])
    var data = family_array(Python.tuple(n, dim), "<f4", context)
    _ = family_call(family_entry(cached[2], "ivf_flat_index_export", context), cached[1], classical_address(data), PythonObject(n * dim))
    py_setitem(fields, "list_data_", data)
    _ = py_call(fields.pop, "_device_list_data")
    py_setitem(fields, "_resident", Python.tuple(_ivf_key(fields, cached[2]), cached[1], cached[2]))


def _ivf_addresses(fields: PythonObject) raises -> PythonObject:
    return Python.list(classical_address(fields["centers_"]), classical_address(fields["center_norms_"]),
        classical_address(fields["list_offsets_"]), classical_address(fields["list_indices_"]), classical_address(fields["list_data_"]))


def _ivf_resident(fields: PythonObject, binding: PythonObject, context: PythonObject) raises -> PythonObject:
    var prepare = knn_door(binding, "ivf_flat_index_prepare")
    if prepare is None:
        _ivf_materialize(fields, context)
        return PythonObject(None)
    var key = _ivf_key(fields, binding)
    var cached = py_get(fields, "_resident")
    if cached is not None and Bool(py=cached[0] == key):
        return cached[1]
    _ivf_materialize(fields, context)
    _ann_release(fields, "ivf_flat_index_release")
    var handle = family_call(prepare, _ivf_addresses(fields), Python.list(fields["n_rows_"],
        fields["n_features_in_"], fields["n_lists_"], fields["metric_code_"], 0))
    py_setitem(fields, "_resident", Python.tuple(_ivf_key(fields, binding), handle, binding))
    return handle


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


def _ivf_search(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    if not py_contains(fields, "list_data_") and not py_contains(fields, "_device_list_data"):
        raise Error("ValueError: call fit or load before search")
    var q = family_as_f32(args[0], 2, "queries", context)
    var dim = Int(py=fields["n_features_in_"])
    if Int(py=q.shape[1]) != dim:
        raise Error("ValueError: query feature count differs from the fitted index")
    var names = Python.tuple("n_lists", "n_probes", "n_neighbors", "kmeans_n_iters", "random_state")
    for i in range(len(names)):  # small-loop(five public integer parameters)
        if py_type(fields[names[i]]) is not b.int:
            raise Error("TypeError: " + String(names[i]) + " must be an int")
    var metric = fields["metric"]
    if py_type(metric) is not b.str and py_type(metric) is not b.int and py_type(metric) is not b.float:
        raise Error("TypeError: metric requires a native name or code")
    if Int(py=_ivf_metric(fields, context)) != Int(py=fields["metric_code_"]):
        raise Error("ValueError: metric differs from the fitted index")
    var binding = _query_route(model, "ivf", context)
    var mode = knn_door(binding, "ivf_numeric_mode")
    var requested = py_get(fields, "numeric_mode")
    if requested is None:
        requested = PythonObject("identical") if Bool(py=_query_kind(model, context)[1]) else context["backend_fields"]["_DEFAULT_MODE"]
    if mode is not None:
        var expected = py_get(context["family_xn_ann_ivf_mode_codes"], requested)
        if expected is None or Int(py=family_call(mode)) != Int(py=expected):
            raise Error("RuntimeError: IVF binding numeric mode differs from the requested mode")
    var n = Int(py=fields["n_rows_"])
    var m = Int(py=q.shape[0])
    var k = Int(py=fields["n_neighbors"])
    if k < 1:
        raise Error("ValueError: n_neighbors must be positive")
    var dist = family_array(Python.tuple(m, k), "<f4", context)
    var ids = family_array(Python.tuple(m, k), "<i4", context)
    var count = family_array(Python.tuple(m), "<i4", context)
    var keep = _ivf_filter(args[1], n, context)
    var handle = _ivf_resident(fields, binding, context)
    var addresses = Python.list(classical_address(q), classical_address(dist), classical_address(ids), classical_address(count))
    var sizes = Python.list(n, dim, fields["n_lists_"], fields["metric_code_"], m, k, fields["n_probes"])
    if handle is None:
        var stored = _ivf_addresses(fields)
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
    _ = keep
    py_setitem(fields, "n_candidates_", count)
    return Python.tuple(dist, ids)


def neighbor_query(model: PythonObject, op: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    var kind = _query_kind(model, context)
    _admit_state(model, kind, context)
    if op == "search" and String(kind[0]) == "ivf":
        return _ivf_search(model, args, context)
    var binding = _query_route(model, "base", context)
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
            py_setitem(entries, "_mojolearn_x_neighbors", _query_route(model, "sort", context))
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
        var kind = _query_kind(model, context)
        if py_type(name) is not native_loaded_module("builtins").str:
            raise Error("TypeError: attribute name must be a string")
        var fields = py_fields(model)
        var text = String(name)
        if String(kind[0]) == "ivf" and text == "list_data_":
            _admit_state(model, kind, context)
            _ivf_materialize(fields, context)
        if String(kind[0]) == "knn":
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
