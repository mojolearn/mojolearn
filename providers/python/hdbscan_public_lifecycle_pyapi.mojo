# SPDX-License-Identifier: Apache-2.0
# HDBSCAN's complete public boundary: metadata admission, current-provider
# selection, fit tracking, output publication and soft-membership batching.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from bindings.native_loaded_module import native_loaded_module
from bindings.classical_attributes import classical_connect_numpy, _type_mro
from bindings.mode_metadata_native import _class_fields
from bindings.classical_pandas import classical_admit_value
from bindings.classical_arrays import classical_address, classical_dtype_str, classical_c_order
from bindings.classical_validate import classical_check_x, classical_view, classical_estimator_name
from bindings.host_inference_scalars import host_inference_scalar
from bindings.native_owner_lifetime import native_release_admit
from bindings.native_array_view import native_array_ndarray_view
from bindings.family_raw_attr import family_raw_get, family_raw_get_or
from bindings.family_common import family_f32, family_array, family_call
from bindings.cluster_fit_a_native import cluster_fit_a_operation, _hdbscan_mode
from bindings.metrics_native_support import metric_warn
from providers.python.estimator_binding_pyapi import estimator_binding
from providers.python.neural_descriptor_pyapi import neural_descriptor_entry
from providers.python.estimator_lifecycle_pyapi import estimator_outside, estimator_fit_begin, estimator_fit_end, estimator_fit_complete
from providers.python.estimator_tabular_boundary_pyapi import _target, _refusal


def _schema(context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var names = context["hdbscan_public_parameters"]
    if py_type(names) is not b.tuple:
        raise Error("TypeError: HDBSCAN constructor schema requires its native tuple")
    for i in range(len(names)):  # driver-loop: exact fixed parameter names before hashing or lookup
        if py_type(names[i]) is not b.str:
            raise Error("TypeError: HDBSCAN parameter names require exact strings")
    return names


def _initialize(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var names = _schema(context)
    if len(names) != len(args):
        raise Error("TypeError: HDBSCAN constructor metadata differs")
    var replacements = Python.list()
    for i in range(len(names)):  # driver-loop: fixed constructor fields; admit every replaced owner first
        var value = args[i]
        if String(names[i]) == "numeric_mode" and value is None:
            value = family_raw_get_or(model, "numeric_mode", PythonObject(None))
        if py_contains(fields, names[i]) and fields[names[i]] is not value:
            native_release_admit(fields[names[i]], context)
        _ = py_call(replacements.append, value)
    for i in range(len(names)):  # driver-loop: publish after complete lifetime admission
        py_setitem(fields, names[i], replacements[i])
    return PythonObject(None)


def _admit(model: PythonObject, context: PythonObject, fitting: Bool) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    var ancestry = _type_mro(py_type(model))
    for i in range(len(ancestry)):  # driver-loop: no fit callbacks, including inherited hooks
        if py_contains(_class_fields(ancestry[i]), "_mojolearn_after_fit"):
            raise Error("TypeError: HDBSCAN cannot invoke Python fit callbacks")
    if py_contains(fields, "_mojolearn_after_fit"):
        raise Error("TypeError: HDBSCAN cannot invoke Python fit callbacks")
    var names = _schema(context)
    for i in range(len(names)):  # driver-loop: scalar settings before legacy numerical-owner casts
        var name = String(names[i])
        var value = family_raw_get_or(model, name, PythonObject(None)) if name == "numeric_mode" else family_raw_get(model, name)
        host_inference_scalar(value, context, True)
    var scalars = Python.tuple("n_features_in_", "n_clusters_", "n_outliers_", "n_boruvka_rounds_", "n_condensed_clusters_", "_mojolearn_fitted", "_mojolearn_fitting")
    var arrays = Python.tuple("labels_", "probabilities_", "core_distances_", "_prediction_data", "_raw_data", "feature_names_in_")
    for i in range(len(scalars)):  # driver-loop: fitted shape and count metadata
        if py_contains(fields, scalars[i]):
            host_inference_scalar(fields[scalars[i]], context)
            if fitting:
                native_release_admit(fields[scalars[i]], context)
    for i in range(len(arrays)):  # driver-loop: fitted storage retained through native execution
        if py_contains(fields, arrays[i]):
            classical_admit_value(fields[arrays[i]], context)
            if fitting:
                native_release_admit(fields[arrays[i]], context)
    var depth = py_get(fields, "_mojolearn_fitting", PythonObject(0))
    if py_type(depth) is not b.int:
        raise Error("TypeError: native fitting depth requires an exact integer")
    var fitted = py_get(fields, "_mojolearn_fitted", PythonObject(False))
    if py_type(fitted) is not b.int and py_type(fitted) is not b.bool:
        raise Error("TypeError: native fitted flags require exact booleans or integers")
    return fields


def _buffer(source: PythonObject, name: String, dtype: String, count: Int, context: PythonObject) raises -> PythonObject:
    var view = classical_view(source, context, name)
    if classical_dtype_str(view, context) != dtype or not classical_c_order(view, context):
        raise Error("TypeError: HDBSCAN " + name + " requires C-order " + dtype + " storage")
    if Int(py=view.size) < count:
        raise Error("ValueError: HDBSCAN " + name + " is smaller than its fitted metadata")
    return view


def _inference(model: PythonObject, method: String, args: PythonObject, fields: PythonObject,
               binding: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if not py_contains(fields, "labels_"):
        raise Error("mojolearn " + method + ": this HDBSCAN instance is not fitted yet; call fit first")
    var pd = py_get(fields, "_prediction_data")
    if pd is None:
        raise Error("mojolearn " + method + ": Prediction data not yet generated. Fit with HDBSCAN(prediction_data=True); cuML refuses the same call the same way (hdbscan.pyx:1104-1108)")
    if py_type(pd) is not b.dict:
        raise Error("TypeError: HDBSCAN prediction metadata requires an exact dictionary")
    var keys = py_call(b.list, pd)
    for i in range(len(keys)):  # driver-loop: no dictionary key protocols during metadata access
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: HDBSCAN prediction metadata requires exact string keys")
    for name in ["n_edges", "n_condensed", "n_exemplars"]:  # driver-loop: fixed tree counts
        if py_type(pd[name]) is not b.int or Int(py=pd[name]) < 0:
            raise Error("TypeError: HDBSCAN tree counts require non-negative native integers")
    var edges = Int(py=pd["n_edges"])
    var condensed = Int(py=pd["n_condensed"])
    var exemplars = Int(py=pd["n_exemplars"])
    var selected = Int(py=fields["n_clusters_"])
    var columns = Int(py=fields["n_features_in_"])
    if selected < 0 or columns < 1:
        raise Error("ValueError: HDBSCAN fitted dimensions are invalid")
    var approximate = method == "approximate_predict"
    var all_points = method == "all_points_membership_vectors"
    var batch = 1
    if not approximate:
        var supplied = args[0] if all_points else args[1]
        if py_type(supplied) is not b.int:
            raise Error("TypeError: mojolearn " + method + ": batch_size must be an int")
        batch = Int(py=supplied)
        if batch <= 0:
            raise Error("mojolearn " + method + ": batch_size must be > 0, got " + String(batch))
    if approximate and selected == 0:
        metric_warn("Clusterer does not have any defined clusters, new data will be automatically predicted as outliers.", b.UserWarning, context)
    var raw = _buffer(fields["_raw_data"], "training data", "<f4", 0, context)
    if Int(py=raw.ndim) != 2 or Int(py=raw.shape[1]) != columns:
        raise Error("ValueError: HDBSCAN training storage differs from its fitted shape")
    var rows = Int(py=raw.shape[0])
    if rows < 2:
        raise Error("ValueError: HDBSCAN training storage requires at least two rows")
    var queries = raw if all_points else family_f32(args[0], 2, "points_to_predict", context)
    var nq = Int(py=queries.shape[0])
    var d = Int(py=queries.shape[1])
    if d != columns:
        raise Error("mojolearn " + method + ": points_to_predict has " + String(d) + " features, the clusterer was fit on " + String(columns))
    var output = family_array(Python.tuple(nq), "<i4", context) if approximate else family_array(Python.tuple(nq, selected), "<f4", context)
    if not approximate and (selected == 0 or nq == 0):
        return output
    var min_samples = family_raw_get(model, "min_samples")
    if min_samples is None or not Bool(py=min_samples):
        min_samples = family_raw_get(model, "min_cluster_size")
    var neighbors = Int(py=min_samples)
    if neighbors < 1 or neighbors > rows:
        raise Error("ValueError: HDBSCAN min_samples must fit its training row count")
    var core = _buffer(fields["core_distances_"], "core_distances_", "<f4", rows, context)
    var labels = _buffer(fields["labels_"], "labels_", "<i4", rows, context)
    var parents = _buffer(pd["parents"], "tree parents", "<i4", edges, context)
    var lambdas = _buffer(pd["lambdas"], "tree lambdas", "<f4", edges, context)
    var deaths = _buffer(pd["deaths"], "deaths", "<f4", condensed, context)
    var clusters = _buffer(pd["selected_clusters"], "selected_clusters", "<i4", selected, context)
    var children = _buffer(pd["index_into_children"], "index_into_children", "<i4", edges + 1, context)
    _hdbscan_mode(model, binding, context)
    if approximate:
        var probabilities = family_array(Python.tuple(nq), "<f4", context)
        _ = family_call(neural_descriptor_entry(binding, "hdbscan_approximate_predict"),
            Python.list(classical_address(raw), classical_address(core), classical_address(labels), classical_address(lambdas),
                        classical_address(deaths), classical_address(clusters), classical_address(children), classical_address(queries),
                        classical_address(output), classical_address(probabilities)),
            Python.list(rows, columns, edges, condensed, selected, nq, neighbors))
        _ = raw^
        _ = queries^
        _ = core^
        _ = labels^
        _ = parents^
        _ = lambdas^
        _ = deaths^
        _ = clusters^
        _ = children^
        return Python.tuple(output, probabilities)
    var indices = _buffer(pd["exemplar_idx"], "exemplar_idx", "<i4", exemplars, context)
    var offsets = _buffer(pd["exemplar_label_offsets"], "exemplar_label_offsets", "<i4", selected + 1, context)
    var tree = Python.list(classical_address(parents), classical_address(lambdas), classical_address(deaths), classical_address(clusters),
                           classical_address(children), classical_address(indices), classical_address(offsets))
    var entry = neural_descriptor_entry(binding, "hdbscan_all_points_membership_vectors" if all_points else "hdbscan_membership_vector")
    # The admitted views and output stay owned until the last synchronous
    # native batch returns. Capture their addresses and invariant argument
    # words once per public operation. The connected bindings only read these
    # private lists; each iteration replaces the varying slice/count words.
    var query_address = 0 if all_points else Int(py=classical_address(queries))
    var output_address = Int(py=classical_address(output))
    var addrs = Python.list(classical_address(raw))
    if not all_points:
        _ = py_call(addrs.append, classical_address(core))
        _ = py_call(addrs.append, classical_address(labels))
    _ = py_call(addrs.extend, tree)
    if not all_points:
        _ = py_call(addrs.append, query_address)
    _ = py_call(addrs.append, output_address)
    var output_slot = 8 if all_points else 11
    var params = Python.list(rows, columns, edges, condensed, selected, exemplars, 0, neighbors)
    var first = 0
    while first < nq:  # driver-loop: public batching stays inside the single native operation
        var count = min(batch, nq - first)
        if not all_points:
            py_setitem(addrs, 10, query_address + first * columns * 4)
        py_setitem(addrs, output_slot, output_address + first * selected * 4)
        py_setitem(params, 6, first if all_points else count)
        if all_points:
            py_setitem(params, 7, count)
        _ = family_call(entry, addrs, params)
        first += count
    # Retain every input and output buffer owner through the final launch.
    _ = raw^
    _ = queries^
    _ = core^
    _ = labels^
    _ = parents^
    _ = lambdas^
    _ = deaths^
    _ = clusters^
    _ = children^
    _ = indices^
    _ = offsets^
    return output


def hdbscan_public_operation(model: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    var b = native_loaded_module("builtins")
    if py_type(args) is not b.tuple:
        raise Error("TypeError: HDBSCAN arguments require their native tuple")
    var ancestry = _type_mro(py_type(model))
    var known = False
    for i in range(len(ancestry)):  # driver-loop: passive subclasses share the connected native owner
        known = known or ancestry[i] is context["hdbscan_public_class"]
    if not known:
        raise Error("TypeError: HDBSCAN execution requires its connected class")
    var method = String(operation[byte=String("family_hdbscan_public_").byte_length():])
    if method == "initialize":
        return _initialize(model, args, context)
    var fitting = method == "fit" or method == "fit_predict"
    var fields = _admit(model, context, fitting)
    var binding = estimator_binding(model, "_mojolearn_hdbscan", context, False)
    if method == "binding":
        _hdbscan_mode(model, binding, context)
        return binding
    if not fitting:
        return _inference(model, method, args, fields, binding, context)
    var name = classical_estimator_name(model)
    var x = classical_check_x(args[0], name, context)
    var rows = Int(py=x.shape[0])
    var columns = Int(py=x.shape[1])
    _target(args[1], rows, name, context)
    var depth = estimator_fit_begin(fields)
    var result = PythonObject(None)
    try:
        result = cluster_fit_a_operation(model, "family_cfa_hdbscan_" + method, Python.tuple(binding, x), context)
    except problem:
        var refused = _refusal(problem, True, depth > 0, rows, columns, name)
        raise refused^
    finally:
        estimator_fit_end(model, depth)
    estimator_fit_complete(model, args[0], False, context)
    if estimator_outside() and py_type(result) is context["array_class"] and py_get(context, "ndarray") is not None:
        return native_array_ndarray_view(result, context)
    return result
