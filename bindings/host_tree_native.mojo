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
from bindings.host_archive_native import host_archive_check_layout, host_archive_typed, host_archive_allocate, host_archive_class_values, host_archive_gbdt_state
from bindings.host_archive_lifetime import host_archive_finalize
from bindings.native_numpy_metadata import native_numpy_dtype
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
    var model_owner = fields
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
    # Pointer lists do not retain the buffers they name. In particular, x
    # can be the sole owner of a native cast/copy, rather than a caller's
    # ndarray. Its lifetime must include the consuming native call.
    _ = x^
    _ = model_owner^
    if Int(py=wrote) != rows:
        raise Error("RuntimeError: forest prediction did not write every input row")
    return out


def host_tree_wide(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var owner = value
    var out = host_archive_typed(value.shape, "<f8", context["output_context"])
    host_tree_widen(Int(py=classical_address(value)), Int(py=value.size), Int(py=classical_address(out)))
    _ = owner^
    return out


def host_tree_labels(classes: PythonObject, vote: PythonObject, context: PythonObject) raises -> PythonObject:
    var vote_owner = vote
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
    _ = vote_owner^
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
    var model_owner = fields
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
        var input_owner = x
        var column = host_archive_typed(Python.tuple(rows * features), "<f4", context["output_context"])
        host_tree_column_copy(Int(py=classical_address(x)), rows, features, Int(py=classical_address(column)))
        _ = input_owner^
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
        # The address envelope owns only integers, not this transposed
        # allocation. Retain the column storage until expansion completes.
        _ = column^
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
    # Retain both the prepared/expanded input and the parsed model arrays
    # through the last native pointer consumer, including helper-created
    # models that have no reference held by the Python caller.
    _ = x^
    _ = a^
    _ = model_owner^
    if Int(py=wrote) != rows:
        raise Error("RuntimeError: GBDT prediction did not write every input row")
    return out


def host_tree_constructor_fields(model: PythonObject, context: PythonObject) raises -> PythonObject:
    host_archive_check_layout(py_type(model), context)
    var fields = sequential_fields(model)
    var b = Python.import_module("builtins")
    var keys = py_call(py_type(Python.list()), fields.keys())
    for i in range(len(keys)):  # driver-loop: native constructor dictionary admission
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: tree state names must be native strings")
    return fields


def host_tree_scalar(value: PythonObject, context: PythonObject) raises:
    # Conversions below call only native scalar slots. Container formatting
    # can transitively invoke element repr hooks and is not a scalar contract.
    var b = Python.import_module("builtins")
    var cls = py_type(value)
    if not _one_of(cls, Python.tuple(b.str, b.bytes, b.bytearray, b.int, b.float, b.bool, py_type(PythonObject(None)))) and not _one_of(cls, context["scalar_types"]):
        raise Error("TypeError: tree scalar options require native scalar values without Python conversion hooks")


def host_tree_initialize_forest(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    var fields = host_tree_constructor_fields(model, context)
    var b = Python.import_module("builtins")
    var estimator = args[0]
    var engine = args[11]
    if py_type(estimator) is not b.str or not py_contains(context["host_archive_forest_estimators"], estimator):
        raise Error("not a forest estimator this loader reads")
    if py_type(engine) is not b.str or (String(engine) != "sequential" and String(engine) != "parallel_groves"):
        raise Error("inference_engine must be sequential or parallel_groves")
    var metadata = context["host_archive_forest_estimators"][estimator]
    var classifier = Bool(py=metadata[1])
    var names = Python.tuple("_offsets", "_colid", "_quesval", "_left_child", "_leaves")
    var arrays = Python.dict()
    for i in range(5):  # driver-loop: fixed constructor buffer fields
        py_setitem(arrays, names[i], args[i + 1])
        _ = host_tree_array(arrays, String(names[i]), "<f4" if i == 2 or i == 4 else "<i4", context)
    for i in range(6, 9):  # driver-loop: positive constructor dimensions
        if py_type(args[i]) is not b.int or Bool(py=args[i] < PythonObject(1)):
            raise Error("forest features, trees and outputs must be positive ints")
    var features = Int(py=args[6])
    var trees = Int(py=args[7])
    var outputs = Int(py=args[8])
    var nodes = Int(py=args[2].size)
    if Int(py=args[1].size) != trees + 1 or nodes < trees or Int(py=args[3].size) != nodes or Int(py=args[4].size) != nodes or Int(py=args[5].size) // outputs != nodes or Int(py=args[5].size) % outputs != 0:
        raise Error("forest model array shapes do not match metadata")
    if Int(py=args[1]._mv[0]) != 0 or Int(py=args[1]._mv[trees]) != nodes:
        raise Error("forest model array shapes do not match metadata")
    var classes = Python.list()
    if classifier:
        var labels = args[9]
        if labels is None:
            raise Error("a classifier archive must carry num_outputs classes")
        if py_type(labels) is context["array_class"]:
            if Int(py=labels.ndim) != 1:
                raise Error("forest class labels must be a native vector")
            classes = host_archive_class_values(labels, context)
        elif py_type(labels) is b.list or py_type(labels) is b.tuple or py_type(labels) is b.str or py_type(labels) is b.bytes or py_type(labels) is b.bytearray or py_type(labels) is py_get(context, "ndarray"):
            # Exact ndarray indexing and native sequence indexing hold element
            # references; they do not run Array.__iter__ or user protocols.
            for i in range(len(labels)):  # driver-loop: distinct constructor class labels
                var label = labels[i]
                classical_admit_value(label, context)
                _ = py_call(classes.append, label)
        else:
            raise Error("TypeError: class labels require a native sequence or array")
        if len(classes) != outputs:
            raise Error("a classifier archive must carry num_outputs classes")
    elif outputs != 1:
        raise Error("a regressor archive must carry one output")
    var module = context["host_archive_forest_binding"]
    if module is None:
        raise Error("ImportError: " + String(context["host_archive_forest_binding_error"]))
    if String(engine) == "parallel_groves" and not Bool(py=context["host_archive_forest_has_groves"]):
        raise Error("RuntimeError: forest host binding predates the parallel_groves engine")
    # Reinitialization drops a previous native grove snapshot exactly once.
    host_archive_finalize(model, context)
    py_setitem(fields, "estimator", estimator)
    py_setitem(fields, "family", metadata[0])
    py_setitem(fields, "is_classifier", classifier)
    py_setitem(fields, "device", args[10])
    py_setitem(fields, "_inference_engine", engine)
    py_setitem(fields, "_groves", PythonObject(None))
    _ = py_call(fields.update, arrays)
    py_setitem(fields, "n_features_in_", features)
    py_setitem(fields, "_n_trees", trees)
    py_setitem(fields, "_num_outputs", outputs)
    py_setitem(fields, "_binding", module)
    if classifier:
        py_setitem(fields, "classes_", classes)
        py_setitem(fields, "n_classes_", len(classes))


def host_tree_initialize_gbdt(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    var fields = host_tree_constructor_fields(model, context)
    var b = Python.import_module("builtins")
    var estimator = args[7]
    if py_type(estimator) is not b.str or not py_contains(context["host_archive_gbdt_estimators"], estimator):
        raise Error("not a gradient boosting class this loader reads")
    for i in range(5):  # driver-loop: native constructor scalar conversions
        host_tree_scalar(args[i], context)
    host_tree_scalar(args[6], context)
    var loss = py_call(b.str, args[0])
    var text = py_call(b.str, args[1])
    var losses = context["host_archive_gbdt_estimators"][estimator]
    if losses is not None and not py_contains(losses, loss):
        raise Error("a " + String(estimator) + " archive with loss " + String(loss) + "; that class saves " + String(py=py_call(PythonObject(", ").join, losses)) + " only")
    var state = host_archive_gbdt_state(text, context["output_context"])
    if Int(py=py_call(b.int, args[2])) != Int(py=state["n_input_features"]):
        raise Error("archive feature count disagrees with model text")
    if Int(py=py_call(b.int, args[3])) != Int(py=state["dim"]):
        raise Error("archive approx_dim disagrees with model text")
    if args[6] is not None and Float64(py=py_call(b.float, args[6])) != Float64(py=state["bias"]):
        raise Error("archive bias disagrees with model text")
    var classes = PythonObject(None) if args[4] is None else py_call(b.int, args[4])
    var module = context["host_archive_forest_binding"]
    if module is None:
        raise Error("ImportError: " + String(context["host_archive_forest_binding_error"]))
    py_setitem(fields, "estimator", estimator)
    py_setitem(fields, "loss", loss)
    py_setitem(fields, "numeric_mode", args[5])
    py_setitem(fields, "model_", text)
    py_setitem(fields, "_arrays", state)
    py_setitem(fields, "n_features_in_", state["n_input_features"])
    py_setitem(fields, "n_model_columns_", state["n_features"])
    py_setitem(fields, "approx_dim_", state["dim"])
    py_setitem(fields, "bias_", state["bias"])
    py_setitem(fields, "n_classes_", classes)
    py_setitem(fields, "is_classifier", py_contains(context["host_archive_gbdt_classification_losses"], loss))
    py_setitem(fields, "has_proba", py_contains(context["host_archive_gbdt_proba_losses"], loss))
    py_setitem(fields, "_binding", module)


def host_tree_operation(model: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    # NumPy string/byte labels also expose native scalar slots. Connect their
    # exact loaded types without a Python setup callback or ndarray iteration.
    var numpy = py_get(context["modules"], "numpy")
    if numpy is not None and py_type(numpy) is context["native_module_type"]:
        var scalars = py_call(Python.import_module("builtins").list, context["scalar_types"])
        var dtype_class = native_numpy_dtype(context)
        var codes = Python.tuple("U", "S")
        for i in range(2):  # driver-loop: fixed native label scalar ABI types
            var cls = py_call(dtype_class, codes[i]).type
            if not _one_of(cls, context["scalar_types"]):
                _ = py_call(scalars.append, cls)
        py_setitem(context, "scalar_types", py_call(Python.import_module("builtins").tuple, scalars))
    classical_admit_value(args, context)
    if operation == "host_tree_init_forest":
        host_tree_initialize_forest(model, args, context)
        return PythonObject(None)
    if operation == "host_tree_init_gbdt":
        host_tree_initialize_gbdt(model, args, context)
        return PythonObject(None)
    if operation == "host_tree_parse" or operation == "host_tree_ctr_dim":
        host_tree_scalar(args[0], context)
        var text = py_call(Python.import_module("builtins").str, args[0])
        var state = host_archive_gbdt_state(text, context["output_context"])
        return state["dim"] if operation == "host_tree_ctr_dim" else state
    if operation == "host_tree_ctr_predict":
        var restored = host_archive_allocate(context["host_tree_classes"][1], context)
        host_tree_initialize_gbdt(restored, Python.tuple(args[1], args[2], args[3], args[4], args[5], PythonObject(None), PythonObject(None), args[6]), context)
        return host_gbdt_predict(sequential_fields(restored), args[0], context)
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
        _ = wide^
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
