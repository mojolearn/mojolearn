# SPDX-License-Identifier: Apache-2.0
# Whole-operation CPU tree inference: native input admission, saved-state
# validation, buffer ownership, kernel calls and completed output publication.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.sequential_description import sequential_fields
from bindings.sequential_admission import _ordinary_torch, _one_of
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_arrays import classical_input, classical_address, classical_is_array, classical_is_f32, classical_cast_kind, classical_c_order, classical_wrap_array
from bindings.classical_pandas import classical_admit_value, native_pandas_exact, native_pandas_f32
from bindings.native_torch_admission import native_torch_numpy
from bindings.host_archive_native import host_archive_check_layout, host_archive_typed
from core.scaler_fit_impl import scaler_cast
from core.host_tree_impl import host_tree_widen, host_tree_argmax, host_tree_column_copy
from core.sha256 import Sha256


def host_tree_entry(context: PythonObject, name: String) raises -> PythonObject:
    var entry = py_get(context["host_tree_entries"], name)
    if entry is None:
        raise Error("ImportError: the CPU tree binding does not export " + name)
    if py_type(entry) is not context["host_tree_entry_type"]:
        raise Error("TypeError: CPU tree inference requires a fixed native entry")
    return entry


def host_tree_input(source: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_admit_value(source, context)
    var prepared = source
    if native_pandas_exact(source):
        prepared = native_pandas_f32(source, context, False, True)
    elif _ordinary_torch(py_type(source), context):
        prepared = native_torch_numpy(source)
    var value = classical_input(prepared, context, "X")
    if Int(py=value.ndim) != 2:
        raise Error("X must be 2-D")
    var rows = Int(py=value.shape[0])
    var cols = Int(py=value.shape[1])
    if rows < 1 or cols < 1:
        raise Error("X is empty")
    if rows > 2147483647 // cols:
        raise Error("X exceeds the native Int32 indexing bound")
    var native = classical_is_array(value, context) or Bool(py=value.dtype.isnative)
    if classical_is_f32(value, context) and classical_c_order(value, context) and native:
        return value
    var output = host_archive_typed(Python.tuple(rows, cols), "<f4", context["output_context"])
    var item = Int(py=value.itemsize)
    var kind = classical_cast_kind(value, item, "X", context)
    scaler_cast(Int(py=classical_address(value)), Int(py=value.strides[0]), Int(py=value.strides[1]),
        rows, cols, item, kind, native,
        MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(output))))
    _ = value^
    _ = prepared^
    return output


def host_tree_array(fields: PythonObject, name: String, dtype: String, context: PythonObject) raises -> PythonObject:
    var value = fields[name]
    if py_type(value) is not context["array_class"]:
        raise Error("TypeError: stored tree arrays must be native mojolearn Arrays")
    if String(value.dtype) != dtype or Int(py=value.ndim) != 1 or String(value.order) != "C":
        raise Error("stored tree field " + name + " must be a C-order " + dtype + " vector")
    return value


def host_tree_fields(model: PythonObject, context: PythonObject) raises -> PythonObject:
    host_archive_check_layout(py_type(model), context)
    var b = Python.import_module("builtins")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, py_type(model))
    var known = False
    for i in range(len(mro)):  # driver-loop: connected CPU tree class ancestry
        known = known or mro[i] is context["host_tree_classes"][0] or mro[i] is context["host_tree_classes"][1]
    if not known:
        raise Error("TypeError: this operation requires a connected CPU tree model")
    var fields = sequential_fields(model)
    var keys = py_call(b.list, fields.keys())
    for i in range(len(keys)):  # driver-loop: stored native tree state names
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: tree state names must be native strings")
        var key = String(keys[i])
        if key != "_binding" and key != "_groves":
            classical_admit_value(fields[key], context)
    if fields["_binding"] is not context["host_archive_forest_binding"]:
        raise Error("TypeError: tree state must retain its connected CPU binding")
    return fields


def host_tree_groves(fields: PythonObject, addresses: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var cached = fields["_groves"]
    if cached is not None:
        if py_type(cached) is not b.tuple or len(cached) != 2 or py_type(cached[0]) is not b.int or cached[1] is not None:
            raise Error("TypeError: CPU groves require a native lifetime handle")
        return cached[0]
    var family = 0 if String(fields["family"]) == "rf" else 1
    var token = py_call(host_tree_entry(context, "forest_host_groves_prepare"), addresses,
        Python.list(fields["_n_trees"], fields["n_features_in_"], fields["_num_outputs"], fields["_colid"].size), family)
    py_setitem(fields, "_groves", Python.tuple(token, PythonObject(None)))
    return token


def host_tree_vote(fields: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    var x = host_tree_input(source, context)
    var rows = Int(py=x.shape[0])
    var features = Int(py=x.shape[1])
    if features != Int(py=fields["n_features_in_"]):
        raise Error("X feature count differs from the fitted model")
    var outputs = Int(py=fields["_num_outputs"])
    if outputs < 1:
        raise Error("stored forest output count must be positive")
    var addresses = Python.list()
    var names = Python.tuple("_offsets", "_colid", "_quesval", "_left_child", "_leaves")
    for i in range(5):  # driver-loop: the five fixed forest model buffers
        var array = host_tree_array(fields, String(names[i]), "<f4" if i == 2 or i == 4 else "<i4", context)
        _ = py_call(addresses.append, classical_address(array))
    var out = host_archive_typed(Python.tuple(rows, outputs), "<f4", context["output_context"])
    var wrote = PythonObject(0)
    if String(fields["_inference_engine"]) == "parallel_groves":
        var token = host_tree_groves(fields, addresses, context)
        wrote = py_call(host_tree_entry(context, "forest_host_groves_predict"), token, classical_address(x), classical_address(out),
            Python.list(rows, features, outputs), 0 if String(fields["family"]) == "rf" else 1)
    else:
        var entry = String("forest_host_et_predict")
        if String(fields["family"]) == "rf":
            entry = "forest_host_rf_predict_proba" if Bool(py=fields["is_classifier"]) else "forest_host_rf_predict_reg"
        _ = py_call(addresses.append, classical_address(x))
        _ = py_call(addresses.append, classical_address(out))
        wrote = py_call(host_tree_entry(context, entry), addresses,
            Python.list(rows, features, fields["_n_trees"], outputs, fields["_colid"].size))
    if Int(py=wrote) != rows:
        raise Error("RuntimeError: forest prediction did not write every input row")
    return out


def host_tree_wide(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var out = host_archive_typed(value.shape, "<f8", context["output_context"])
    host_tree_widen(Int(py=classical_address(value)), Int(py=value.size), Int(py=classical_address(out)))
    return out


def host_tree_labels(classes: PythonObject, vote: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    if py_type(classes) is not b.list or len(classes) != Int(py=vote.shape[1]):
        raise Error("stored forest classes do not match its output count")
    classical_admit_value(classes, context)
    var integers = True
    var numbers = True
    var fits = True
    for i in range(len(classes)):  # driver-loop: distinct saved class label metadata
        var cls = py_type(classes[i])
        var integer = _one_of(cls, context["integer_types"]) and cls is not b.bool
        var number = integer or cls is b.float
        if not number and _one_of(cls, context["scalar_types"]):
            var kind = String(classes[i].dtype.kind)
            integer = kind == "i" or kind == "u"
            number = integer or kind == "f"
        integers = integers and integer
        numbers = numbers and number
        if integer:
            var value = py_call(b.int, classes[i])
            fits = fits and Bool(py=value >= PythonObject(-9223372036854775807 - 1)) and Bool(py=value <= PythonObject(9223372036854775807))
    var rows = Int(py=vote.shape[0])
    var indices = host_tree_argmax(Int(py=classical_address(vote)), rows, Int(py=vote.shape[1]))
    if (integers and fits) or (numbers and not integers):
        var out = host_archive_typed(Python.tuple(rows), "<i8" if integers else "<f8", context["output_context"])
        for i in range(rows):  # driver-loop: publish completed native label selections
            if integers:
                MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(out)))[i] = Int64(Int(py=classes[indices[i]]))
            else:
                MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(out)))[i] = Float64(py=classes[indices[i]])
        return out
    var out = Python.list()
    for i in range(rows):  # driver-loop: publish completed native string/bool/large-integer labels
        _ = py_call(out.append, classes[indices[i]])
    return out


def host_gbdt_predict(fields: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    var x = host_tree_input(source, context)
    var rows = Int(py=x.shape[0])
    var features = Int(py=x.shape[1])
    if features != Int(py=fields["n_features_in_"]):
        raise Error("X feature count differs from the fitted GBDT model")
    var a = fields["_arrays"]
    var dim = Int(py=fields["approx_dim_"])
    var columns = Int(py=fields["n_model_columns_"])
    var row_major = Int(py=a["n_ctr_tables"]) == 0 and Int(py=a["n_tensor_tables"]) == 0
    if not row_major:
        var column = host_archive_typed(Python.tuple(rows * features), "<f4", context["output_context"])
        host_tree_column_copy(Int(py=classical_address(x)), rows, features, Int(py=classical_address(column)))
        var expanded = host_archive_typed(Python.tuple(rows * columns), "<f4", context["output_context"])
        var addresses = Python.list(classical_address(column))
        var names = Python.tuple("ctr_ints", "ctr_floats", "tensor_ints", "tensor_floats", "ctr_counts", "border_offsets", "borders", "one_hot")
        for i in range(len(names)):  # driver-loop: fixed native CTR expansion buffers
            var array = host_tree_array(a, String(names[i]), "<f4" if i == 1 or i == 3 or i == 6 else "<i4", context)
            _ = py_call(addresses.append, classical_address(array))
        _ = py_call(addresses.append, classical_address(expanded))
        var wrote = py_call(host_tree_entry(context, "forest_host_gbdt_expand_ctr"), addresses,
            Python.list(rows, features, columns, a["n_ctr_tables"], a["n_ctr_ints"], a["n_ctr_floats"],
                a["n_tensor_tables"], a["n_tensor_ints"], a["n_tensor_floats"], a["n_ctr_counts"], a["n_borders"]))
        if Int(py=wrote) != columns:
            raise Error("RuntimeError: GBDT expansion did not write every model column")
        x = expanded
        features = columns
    var out = host_archive_typed(Python.tuple(rows) if dim == 1 else Python.tuple(rows, dim), "<f4", context["output_context"])
    var names = Python.tuple("border_offsets", "borders", "fold_counts", "one_hot", "nan_treatment", "tree_offsets",
        "split_feature", "split_bin", "split_take_bin", "node_left", "node_right", "leaf_offsets", "leaves")
    var addresses = Python.list()
    for i in range(len(names)):  # driver-loop: fixed native GBDT prediction buffers
        var array = host_tree_array(a, String(names[i]), "<f4" if i == 1 or i == 12 else "<i4", context)
        _ = py_call(addresses.append, classical_address(array))
    _ = py_call(addresses.append, classical_address(x))
    _ = py_call(addresses.append, classical_address(out))
    var wrote = py_call(host_tree_entry(context, "forest_host_gbdt_predict"), addresses,
        Python.list(rows, features, a["n_trees"], dim, 1 if Bool(py=a["non_symmetric"]) else 0,
            a["n_splits"], a["n_leaf_values"], a["n_borders"], 1 if row_major else 0), fields["bias_"])
    if Int(py=wrote) != rows:
        raise Error("RuntimeError: GBDT prediction did not write every input row")
    return out


def host_tree_operation(model: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    classical_admit_value(args, context)
    if operation == "host_tree_setstate":
        host_archive_check_layout(py_type(model), context)
        if py_type(args[0]) is not Python.import_module("builtins").dict:
            raise Error("TypeError: saved tree state must be a native dictionary")
        var restored = sequential_fields(model)
        _ = py_call(restored.update, args[0])
        py_setitem(restored, "_binding", context["host_archive_forest_binding"])
        _ = host_tree_fields(model, context)
        return PythonObject(None)
    var fields = host_tree_fields(model, context)
    var gbdt = py_contains(fields, "_arrays")
    if operation == "host_tree_getstate":
        var state = py_call(Python.import_module("builtins").dict, fields)
        py_setitem(state, "_groves", PythonObject(None))
        _ = py_call(state.pop, "_binding", PythonObject(None))
        return state
    if operation == "host_tree_groves":
        var addresses = Python.list()
        var names = Python.tuple("_offsets", "_colid", "_quesval", "_left_child", "_leaves")
        for i in range(5):  # driver-loop: the five fixed forest model buffers
            var array = host_tree_array(fields, String(names[i]), "<f4" if i == 2 or i == 4 else "<i4", context)
            _ = py_call(addresses.append, classical_address(array))
        return host_tree_groves(fields, addresses, context)
    if operation == "host_tree_count":
        return fields["_arrays"]["n_trees"] if gbdt else fields["_n_trees"]
    if operation == "host_tree_engine":
        return PythonObject("sequential") if gbdt else fields["_inference_engine"]
    if operation == "host_tree_hash":
        var hash = Sha256()
        if gbdt:
            hash.update_text(String(fields["model_"]))
        else:
            var names = Python.tuple("_offsets", "_colid", "_quesval", "_left_child", "_leaves")
            for i in range(5):  # driver-loop: the fixed forest model hash fields
                var value = host_tree_array(fields, String(names[i]), "<f4" if i == 2 or i == 4 else "<i4", context)
                hash.update(Int(py=classical_address(value)), Int(py=value.nbytes))
        return PythonObject(hash.hexdigest())
    var probability = operation == "host_tree_proba"
    var pair = operation == "host_tree_predict_pair"
    if gbdt:
        if probability:
            if String(fields["loss"]) == "MultiClass" or String(fields["loss"]) == "MultiClassOneVsAll":
                raise Error("NotImplementedError: HostGBDT.predict_proba has no qualified multiclass transform")
            if not Bool(py=fields["has_proba"]):
                raise Error("predict_proba is defined for Logloss and CrossEntropy")
        var raw = host_gbdt_predict(fields, args[0], context)
        if not probability:
            return raw
        var wide = host_tree_wide(raw, context)
        var rows = Int(py=raw.shape[0])
        var out = host_archive_typed(Python.tuple(rows, 2), "<f8", context["output_context"])
        var wrote = py_call(host_tree_entry(context, "forest_host_gbdt_sigmoid_pair"), classical_address(wide), classical_address(out), rows)
        if Int(py=wrote) != rows:
            raise Error("RuntimeError: GBDT probability conversion did not write every row")
        return out
    if (probability or pair) and not Bool(py=fields["is_classifier"]):
        raise Error("AttributeError: this forest regressor has no predict_proba")
    var vote = host_tree_vote(fields, args[0], context)
    if operation == "host_tree_vote":
        return vote
    if probability or pair:
        var proba = vote if String(fields["family"]) == "rf" else host_tree_wide(vote, context)
        return Python.tuple(host_tree_labels(fields["classes_"], vote, context), proba) if pair else proba
    if Bool(py=fields["is_classifier"]):
        return host_tree_labels(fields["classes_"], vote, context)
    var flat = classical_wrap_array(vote._store, Python.tuple(vote.shape[0]), "<f4", "C", context["output_context"])
    return flat if String(fields["family"]) == "rf" else host_tree_wide(flat, context)
