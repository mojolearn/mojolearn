# SPDX-License-Identifier: Apache-2.0
# Complete saved-host KNN prediction. The host ABI has no resident-index
# entry: retain the established per-call search/vote and exact output layout.
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module, _loaded_fields
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields
from bindings.family_common import family_as_f32_dense, family_call, family_min
from bindings.classical_arrays import classical_address, classical_wrap_array
from bindings.classical_attributes import classical_connect_numpy
from bindings.host_archive_native import host_archive_neighbor_metric
from bindings.host_array_native import host_archive_typed
from core.hotpath_impl import cast_elements, HP_I32, HP_I64


def _knn_entry(context: PythonObject, name: String) raises -> PythonObject:
    var binding = context["host_archive_predict_knn_binding"]
    if py_type(binding) is py_type(PythonObject("")):
        raise Error(String(binding))
    var entry = py_get(_loaded_fields(binding), name)
    if entry is None:
        raise Error("ImportError: CPU neighbor binding does not export " + name)
    if py_type(entry) is not context["host_archive_predict_entry_type"]:
        raise Error("TypeError: CPU neighbor inference requires its compiled native entry")
    return entry


def host_knn_predict(model: PythonObject, source: PythonObject, classifier: Bool,
                     probability: Bool, context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    var fields = py_fields(model)
    var index = py_get(fields, "_index")
    var targets = py_get(fields, "_y_cols")
    if index is None or targets is None:
        raise Error("mojolearn: call fit before predict")
    var query = family_as_f32_dense(source, 2, "X", context)
    var metric = host_archive_neighbor_metric(fields["metric"], Float64(py=fields["p"]), False, context)
    if String(fields["metric"]).lower() == "jensenshannon" and Int(py=query.size) > 0 and family_min(query) < 0:
        var name = "HostKNeighborsClassifier" if classifier else "HostKNeighborsRegressor"
        raise Error("mojolearn " + name + ": metric='jensenshannon' needs X >= 0 (it takes logarithms of the coordinates)")
    var rows = Int(py=query.shape[0])
    var features = Int(py=query.shape[1])
    var fitted_features = Int(py=fields["n_features_in_"])
    if features != fitted_features:
        raise Error("mojolearn: X has " + String(features) + " features, index has " + String(fitted_features))
    var count = Int(py=fields["n_samples_fit_"])
    var k = Int(py=fields["n_neighbors"])
    if k < 1 or k > count:
        raise Error("mojolearn: n_neighbors must be in [1, " + String(count) + "], got " + String(k))
    var outputs = Int(py=targets.shape[0])
    var multi = Bool(py=fields["outputs_2d_"])
    var options = Python.list(metric[0], metric[1], PythonObject(1 if String(fields["weights"]) == "distance" else 0))
    var params = Python.list(index.shape[0], PythonObject(rows), index.shape[1], PythonObject(k), fields["query_tile"], PythonObject(outputs))
    if not classifier:
        var output = host_archive_typed(Python.tuple(rows, outputs), "<f4", context)
        var tile = family_call(_knn_entry(context, "knn_regress"), classical_address(index), classical_address(query),
            classical_address(targets), classical_address(output), params, options)
        _ = index^
        _ = query^
        _ = targets^
        py_setitem(fields, "used_query_tile_", tile)
        if multi:
            return output
        return classical_wrap_array(output._store, Python.tuple(rows), "<f4", "C", context)

    var classes = fields["_classes_list"]
    var total = 0
    var counts = List[Int]()
    _ = py_call(params.append, PythonObject(1 if probability else 0))
    for i in range(outputs):  # driver-loop: distinct class counts for each output column
        var width = len(classes[i])
        counts.append(width)
        total += width
        _ = py_call(params.append, PythonObject(width))
    var labels = host_archive_typed(Python.tuple(1) if probability else Python.tuple(rows, outputs), "<i4", context)
    var proba = host_archive_typed(Python.tuple(rows * total if probability else 1), "<f4", context)
    var unique = host_archive_typed(Python.tuple(total), "<i4", context)
    var tile = family_call(_knn_entry(context, "knn_classify"), classical_address(index), classical_address(query),
        classical_address(targets), classical_address(labels), classical_address(proba), classical_address(unique), params, options)
    _ = index^
    _ = query^
    _ = targets^
    py_setitem(fields, "used_query_tile_", tile)
    var actual = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(unique)))
    var offset = 0
    for output in range(outputs):  # driver-loop: cross-check the completed distinct class metadata
        var expected = classes[output]
        for c in range(counts[output]):  # driver-loop: sorted unique class names for one output
            if Int(actual[offset + c]) != Int(py=expected[c]):
                raise Error("RuntimeError: mojolearn: output " + String(output) + ": the implemented getUniquelabels and host sort class sets disagree")
        offset += counts[output]
    _ = unique^
    if not probability:
        var shape = Python.tuple(rows, outputs) if multi else Python.tuple(rows)
        var widened = host_archive_typed(shape, "<i8", context)
        _ = cast_elements(Int(py=classical_address(labels)), HP_I32, Int(py=classical_address(widened)), HP_I64, rows * outputs)
        _ = labels^
        # The established saved-neighbor schema refuses _label_names at
        # save, and restores integer classes. Live string-label KNN is a
        # different public schema; no unsupported conversion is introduced.
        return widened
    var result = Python.list()
    offset = 0
    for output in range(outputs):  # driver-loop: publish the native per-output probability blocks
        var width = counts[output]
        var block = host_archive_typed(Python.tuple(rows, width), "<f4", context)
        memcpy(dest=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(block))),
               src=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(proba)) + 4 * offset),
               count=rows * width)
        _ = py_call(result.append, block)
        offset += rows * width
    _ = proba^
    return result if multi else result[0]
