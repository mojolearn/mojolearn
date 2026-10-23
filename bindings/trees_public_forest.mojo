# SPDX-License-Identifier: Apache-2.0
# The forests' public operations in one native call each: RandomForest*,
# ExtraTrees* and DecisionTree* `fit`, `predict` and `predict_proba`
# (python/mojolearn/randomforest.py, extratrees.py, _expansion_trees.py,
# _forest_protocol.py). This is the Python orchestration those methods ran,
# step for step: the scikit-learn style parameter refresh and the
# constructor checks with their wording, the captured numeric mode, the
# label encoding, the sample and class weights, the fit entry and layout
# choice (row-major, column-major, weighted, auxiliary outputs, a tree-ID
# shard), the native model export, the fitted state, and at predict time
# the inference engine (sequential walk, the resident strict snapshot or
# the 32-grove resident snapshot) and the label decode. The kernels and
# their arguments are the ones the Python path launched, in the same order,
# so the fitted model and every prediction keep their bits.
#
# Round trips removed against the Python path: one native call replaces the
# Python sequence (labels, weights, finiteness scan, params, fit, export,
# unpack) of every fit; X, y and the weights are read once and handed to the
# fit entry without a Python step between the staging and the launch; the
# resident device snapshot is prepared and queried in the same call, and a
# classifier `predict` decodes the resident labels (or the argmax of the
# vote) without returning to Python between the vote and the decode.
from std.math import sqrt
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_fields
from bindings.py_typed_call import py_call_typed
from bindings.classical_arrays import classical_wrap_array, _store_code
from bindings.trees_public_inputs import (
    TREES_NONFINITE, TreesMatrix, TreesLabels, trees_builtins, trees_table, trees_entry, trees_need,
    trees_helper, trees_env, trees_empty, trees_vector, trees_reshape, trees_freeze, trees_address,
    trees_f32, trees_forest_layout, trees_all_finite, trees_check_sample_weight, trees_encode_labels,
    trees_decode_labels, trees_argmax_rows, trees_class_rows_only, trees_weighted_rows, trees_cast,
    trees_view, trees_call,
)
from bindings.trees_lifetime import trees_resident_new
from checks.numerics import portable_log2_64

comptime _INT32_MAX = 2147483647
comptime _MODEL_FIELDS = 5


# ------------------------------------------------------------- utilities


def _repr(value: PythonObject) raises -> String:
    return String(py_call(trees_builtins().repr, value))


def _str(value: PythonObject) raises -> String:
    return String(py_call(trees_builtins().str, value))


def _truthy(value: PythonObject) raises -> Bool:
    return Bool(py=py_call(trees_builtins().bool, value))


def _is_bool(value: PythonObject) raises -> Bool:
    """`_labels.is_bool`: Python's bool or NumPy's bool_."""
    var t = py_type(value)
    return t is trees_builtins().bool or String(t.__name__) == "bool_"


def _is_integral(value: PythonObject, context: PythonObject) raises -> Bool:
    return Bool(py=py_call(trees_builtins().isinstance, value, context["trees_integral_type"]))


def _int(value: PythonObject) raises -> PythonObject:
    """Python's `int(value)`, with its refusal."""
    return py_call_typed(trees_builtins().int, value)


def _float(value: PythonObject) raises -> PythonObject:
    """Python's `float(value)`, with its refusal."""
    return py_call_typed(trees_builtins().float, value)


def _refuse(name: String, why: String) raises:
    """`randomforest._refuse`."""
    raise Error("NotImplementedError: " + name + " is not implemented: " + why
                + " Refused by name rather than accepted and ignored.")


def _class_fields(cls: PythonObject) raises -> PythonObject:
    var b = trees_builtins()
    return py_call(b.type.__dict__["__dict__"].__get__, cls)


def _mro(cls: PythonObject) raises -> PythonObject:
    var b = trees_builtins()
    return py_call(b.type.__dict__["__mro__"].__get__, cls)


def forest_kind(model: PythonObject, context: PythonObject) raises -> String:
    """The connected schema of the model's class: the nearest registered
    class in its ancestry ("rf_classifier", "dt_regressor", "et_classifier",
    ...)."""
    var cls = py_type(model)
    var mro = _mro(cls)
    var table = context["trees_classes"]
    for i in range(len(mro)):  # driver-loop: over the class ancestry metadata
        var kind = py_get(table, mro[i])
        if kind is not None:
            return String(kind)
    raise Error("TypeError: " + String(cls.__name__) + " is not a connected tree ensemble")


def _family(kind: String) -> String:
    return "et" if kind.startswith("et_") else "rf"


def _classifier(kind: String) -> Bool:
    return kind.endswith("_classifier")


# ------------------------------------------------------- parameters


def forest_get_params(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """`ForestProtocol.get_params()`: the stored constructor parameters, by
    name, with its two refusals."""
    var b = trees_builtins()
    var cls = py_type(model)
    var mro = _mro(cls)
    for i in range(len(mro)):  # driver-loop: over the class ancestry metadata
        var own = _class_fields(mro[i])
        if py_contains(own, "__init__"):
            if not py_contains(own, "_parameter_names"):
                raise Error("TypeError: Forest subclasses with a custom constructor must register "
                            + "their parameters before cloning or refitting")
            break
    var fields = py_fields(model)
    var has = py_get(fields, "_has_constructor_parameters")
    if has is None:
        has = py_get(_class_fields(cls), "_has_constructor_parameters", PythonObject(False))
    if not _truthy(has):
        raise Error("This inference-only forest archive has no constructor "
                    + "parameters; construct a new estimator for cloning/refitting")
    var names = cls._parameter_names
    var params = Python.dict()
    for i in range(len(names)):  # driver-loop: over the constructor parameter names
        var value = py_get(fields, names[i], context["trees_missing"])
        if value is context["trees_missing"]:
            raise Error("AttributeError: '" + String(cls.__name__) + "' object has no attribute '" + String(names[i]) + "'")
        py_setitem(params, names[i], value)
    return params


def _param(params: PythonObject, name: String, default: PythonObject) raises -> PythonObject:
    return py_get(params, PythonObject(name), default)


def _check_mode(mode: PythonObject) raises -> PythonObject:
    """`ForestProtocol._validated_mode`: None, or 'fast' / 'identical'."""
    var b = trees_builtins()
    if mode is None:
        return mode
    if py_type(mode) is not b.str and not Bool(py=py_call(b.isinstance, mode, b.str)):
        raise Error("numeric_mode must be fast, identical or None")
    var clean = py_call(py_call(mode.strip).lower)
    if String(clean) != "fast" and String(clean) != "identical":
        raise Error("numeric_mode must be fast, identical or None")
    return clean


def _check_engine(engine: PythonObject) raises -> String:
    """`ForestProtocol._validate_inference_engine`."""
    var b = trees_builtins()
    if not Bool(py=py_call(b.isinstance, engine, b.str)):
        raise Error("inference_engine must be 'auto', 'sequential' or 'parallel_groves'")
    var name = String(engine)
    if name != "auto" and name != "sequential" and name != "parallel_groves":
        raise Error("inference_engine must be 'auto', 'sequential' or 'parallel_groves'")
    return name


def _criterion_code(criterion: PythonObject, table: PythonObject, unsupported: PythonObject, who: String) raises -> PythonObject:
    """`randomforest._criterion_code`."""
    var b = trees_builtins()
    if py_contains(table, criterion):
        return table[criterion]
    if py_contains(unsupported, criterion):
        _refuse("criterion=" + _repr(criterion), String(unsupported[criterion]))
    raise Error("criterion=" + _repr(criterion) + " is not a " + who + " criterion here; accepted are "
                + _repr(py_call(b.sorted, table)))


def _min_weight_fraction(value: PythonObject, who: String) raises -> PythonObject:
    """`randomforest._min_weight_fraction`."""
    var b = trees_builtins()
    var f = PythonObject(None)
    try:
        f = py_call_typed(b.float, value)
    except error:
        var message = String(error)
        if not message.startswith("TypeError: ") and not message.startswith("ValueError: "):
            raise error^
        f = PythonObject(None)
    var ok = f is not None and not _is_bool(value)
    if ok:
        var x = Float64(py=f)
        ok = x >= 0.0 and x <= 0.5
    if not ok:
        raise Error("The 'min_weight_fraction_leaf' parameter of " + who + " must be a float"
                    + " in the range [0.0, 0.5]. Got " + _repr(value) + " instead.")
    return f


def _max_leaves_slot(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`randomforest._max_leaves_slot`."""
    if value is None:
        return PythonObject(-1)
    if _is_bool(value) or not _is_integral(value, context):
        raise Error("max_leaves=" + _repr(value) + " must be None, -1 (cuML's sentinel"
                    + " for unlimited) or a positive int count")
    var v = _int(value)
    var n = Int(py=v)
    if n == -1 or n > 0:
        return v
    raise Error("max_leaves=" + _str(v) + " is not a leaf cap: cuML accepts -1 (unlimited) or"
                + " a positive count, and ensemble/decisiontree/decisiontree.mojo"
                + ":257-258 raises on anything else")


def _rf_base(params: PythonObject, criterion_code: PythonObject, class_weight: PythonObject,
             tables: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_RandomForestBase.__init__`: its refusals in order, then the native
    configuration `_cfg`. `params` hold the base's arguments by name."""
    var b = trees_builtins()
    var device = params["device"]
    if not Bool(py=device == PythonObject("gpu")):
        raise Error("device must be 'gpu', got " + _repr(device) + ": the fit runs on the"
                    + " GPU binding, or, on a CPU-only install with the rf host"
                    + " binding built, on its host restatement"
                    + " (ensemble/host/rf_oracle.mojo) under the same device='gpu'")
    if params["max_leaf_nodes"] is not None:
        _refuse("max_leaf_nodes", String(tables["max_leaf_nodes_why"]))
    var oob = params["oob_score"]
    if not _is_bool(oob):
        raise Error("oob_score must be a boolean; callable scoring is not supported")
    if _truthy(oob) and not _truthy(params["bootstrap"]):
        raise Error("oob_score=True requires bootstrap=True")
    if _truthy(params["warm_start"]):
        _refuse("warm_start=True", "there is no incremental fit here,"
                + " and accepting it would silently refit from scratch.")
    if _truthy(params["ccp_alpha"]):
        _refuse("ccp_alpha", "cost-complexity pruning is a"
                + " post-processing pass neither cuML nor this implementation"
                + " implements. Only 0.0 is accepted.")
    if class_weight is not None and not Bool(py=py_call(b.isinstance, class_weight, b.dict)):
        if not Bool(py=py_call(b.isinstance, class_weight, b.str)) or not Bool(py=class_weight == PythonObject("balanced")):
            raise Error("class_weight must be None, a dict, or 'balanced'; balanced_subsample is unsupported")
    if params["monotonic_cst"] is not None:
        _refuse("monotonic_cst", "no cuML counterpart.")
    if params["n_jobs"] is not None:
        _refuse("n_jobs", "there is no CPU thread pool here -- the fit"
                + " is one host thread driving the GPU.")
    if _truthy(params["verbose"]):
        _refuse("verbose", "nothing in the Mojo layer logs per-tree progress.")
    var max_samples = params["max_samples"]
    if max_samples is not None:
        var f = Float64(py=_float(max_samples))
        if not (f > 0.0 and f <= 1.0):
            raise Error("max_samples=" + _str(max_samples) + " is outside (0, 1]")
    var cfg = Python.dict()
    py_setitem(cfg, "criterion", _int(criterion_code))
    py_setitem(cfg, "n_estimators", _int(params["n_estimators"]))
    var depth = params["max_depth"]
    py_setitem(cfg, "max_depth", PythonObject(_INT32_MAX) if depth is None else _int(depth))
    py_setitem(cfg, "max_leaves", _max_leaves_slot(params["max_leaves"], context))
    py_setitem(cfg, "max_features", params["max_features"])
    py_setitem(cfg, "n_bins", _int(params["n_bins"]))
    py_setitem(cfg, "min_samples_leaf", _int(params["min_samples_leaf"]))
    py_setitem(cfg, "min_samples_split", _int(params["min_samples_split"]))
    py_setitem(cfg, "min_impurity_decrease", _float(params["min_impurity_decrease"]))
    py_setitem(cfg, "bootstrap", PythonObject(_truthy(params["bootstrap"])))
    py_setitem(cfg, "max_samples", PythonObject(1.0) if max_samples is None else _float(max_samples))
    var seed = params["random_state"]
    py_setitem(cfg, "random_state", PythonObject(0) if seed is None else _int(seed))
    py_setitem(cfg, "n_streams", _int(params["n_streams"]))
    py_setitem(cfg, "max_batch_size", _int(params["max_batch_size"]))
    py_setitem(cfg, "oob_score", PythonObject(_truthy(oob)))
    return cfg


def _rf_arguments(params: PythonObject, kind: String, context: PythonObject) raises -> PythonObject:
    """The base constructor's arguments: the estimator's own for a forest;
    for a DecisionTree*, `_dt_common` first, then its fixed single-tree
    arguments (one tree, no bootstrap, one stream) over the RF defaults."""
    if not kind.startswith("dt_"):
        return params
    var b = trees_builtins()
    var splitter = params["splitter"]
    if not Bool(py=splitter == PythonObject("best")) and not Bool(py=splitter == PythonObject("random")):
        raise Error("splitter must be 'best' or 'random', got " + _repr(splitter))
    var best = Bool(py=splitter == PythonObject("best"))
    if best and params["max_leaf_nodes"] is not None:
        _refuse("max_leaf_nodes with splitter='best'", "best-first growth exists on the random"
                + " splitter's builder (extratrees/, DEVIATIONS 466-469); pass splitter='random'.")
    var args = py_call(b.dict, params)
    py_setitem(args, "n_estimators", PythonObject(1))
    py_setitem(args, "max_leaf_nodes", PythonObject(None) if not best else params["max_leaf_nodes"])
    py_setitem(args, "bootstrap", PythonObject(False))
    py_setitem(args, "n_streams", PythonObject(1))
    py_setitem(args, "oob_score", PythonObject(False))
    py_setitem(args, "n_jobs", PythonObject(None))
    py_setitem(args, "verbose", PythonObject(0))
    py_setitem(args, "warm_start", PythonObject(False))
    py_setitem(args, "max_samples", PythonObject(None))
    py_setitem(args, "max_leaves", PythonObject(-1))
    py_setitem(args, "max_batch_size", PythonObject(4096))
    if kind == "dt_regressor":
        py_setitem(args, "class_weight", PythonObject(None))
        var bins = params["n_bins"]
        py_setitem(args, "n_bins", PythonObject(128) if bins is None else bins)
    return args


def _rf_derive(model: PythonObject, params: PythonObject, kind: String, context: PythonObject) raises -> PythonObject:
    """The state the RandomForest*/DecisionTree* constructor derived (in
    its order): `device` and `_cfg`."""
    var tables = context["trees_rf_tables"]
    var args = _rf_arguments(params, kind, context)
    var who = String(py_type(model).__name__)
    var classifier = _classifier(kind)
    var code = _criterion_code(args["criterion"], tables["cls_criteria" if classifier else "reg_criteria"],
                               tables["unsupported"], "classifier" if classifier else "regressor")
    var mwf = _min_weight_fraction(args["min_weight_fraction_leaf"], who)
    var cfg = _rf_base(args, code, _param(args, "class_weight", PythonObject(None)) if classifier else PythonObject(None),
                       tables, context)
    py_setitem(cfg, "min_weight_fraction_leaf", mwf)
    var derived = Python.dict()
    py_setitem(derived, "device", args["device"])
    py_setitem(derived, "_cfg", cfg)
    return derived


def _et_derive(model: PythonObject, params: PythonObject, kind: String, context: PythonObject) raises -> PythonObject:
    """`ExtraTrees*.__init__`: `_validate_device`, `_refuse_forest_knobs`,
    the criterion, then `device`, `_criterion_code` and `_cfg` (the raw
    parameters; `_fit_params` converts them at fit)."""
    var b = trees_builtins()
    var tables = context["trees_et_tables"]
    var device = params["device"]
    if not Bool(py=py_call(b.isinstance, device, b.str)) or not Bool(py=device == PythonObject("gpu")):
        raise Error("Extra Trees training is GPU-only; device must be 'gpu', got " + _repr(device))
    if params["n_jobs"] is not None:
        raise Error("NotImplementedError: n_jobs is not implemented: there is no CPU thread pool here -- the"
                    + " fit runs one host thread driving the GPU (device='gpu')."
                    + " Refused by name rather than"
                    + " accepted and ignored.")
    if _truthy(params["verbose"]):
        raise Error("NotImplementedError: verbose is not implemented: nothing in the Mojo layer logs per-tree"
                    + " progress. Refused by name rather than accepted and ignored.")
    var classifier = _classifier(kind)
    var criterion = params["criterion"]
    var table = tables["cls_criteria" if classifier else "reg_criteria"]
    if not py_contains(table, criterion):
        var reason = py_get(tables["unsupported"], criterion)
        var tail = String("; 'gini', 'entropy' and its alias 'log_loss' are.") if classifier else \
            String("; 'squared_error', 'poisson', 'gamma' and 'inverse_gaussian' are.")
        raise Error("NotImplementedError: criterion=" + _repr(criterion) + " is not implemented"
                    + (" (" + String(reason[1]) + ")" if reason is not None else String("")) + tail)
    var cfg = Python.dict()
    var names = Python.tuple("n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
                             "min_weight_fraction_leaf", "max_features", "max_leaf_nodes",
                             "min_impurity_decrease", "bootstrap", "oob_score", "random_state",
                             "warm_start", "class_weight", "ccp_alpha", "max_samples", "monotonic_cst")
    for i in range(len(names)):  # small-loop(names): the sixteen fixed configuration names
        var name = names[i]
        if String(name) == "class_weight" and not classifier:
            py_setitem(cfg, name, PythonObject(None))
        else:
            py_setitem(cfg, name, params[name])
    if not classifier:
        var mf = params["max_features"]
        if Bool(py=mf == PythonObject(1.0)) and not Bool(py=py_call(b.isinstance, mf, b.int)):
            py_setitem(cfg, "max_features", PythonObject(None))
    var derived = Python.dict()
    py_setitem(derived, "device", device)
    py_setitem(derived, "_criterion_code", table[criterion])
    py_setitem(derived, "_cfg", cfg)
    return derived


def forest_refresh(model: PythonObject, kind: String, context: PythonObject) raises -> PythonObject:
    """`ForestProtocol._refresh_config()`: the constructor checks run on the
    stored parameters (a refused parameter leaves the instance as it was);
    then the instance holds exactly what a new constructor plus
    `_validate_params` would hold, and the two bookkeeping fields a running
    fit owns. Returns the instance dictionary."""
    var params = forest_get_params(model, context)
    _ = _check_mode(_param(params, "numeric_mode", PythonObject(None)))
    _ = _check_engine(_param(params, "inference_engine", PythonObject("sequential")))
    var derived = _et_derive(model, params, kind, context) if _family(kind) == "et" else _rf_derive(model, params, kind, context)
    var fields = py_fields(model)
    var keep = Python.dict()
    var kept = Python.tuple("_parent_callback_ctx", "_mojolearn_fitting")
    for i in range(2):  # small-loop(kept): the two bookkeeping field names
        if py_contains(fields, kept[i]):
            py_setitem(keep, kept[i], fields[kept[i]])
    _ = py_call(fields.clear)
    _ = py_call(fields.update, params)
    py_setitem(fields, "_has_constructor_parameters", PythonObject(True))
    _ = py_call(fields.update, derived)
    _ = py_call(fields.update, keep)
    return fields


def forest_capture_mode(fields: PythonObject, context: PythonObject) raises -> String:
    """`ForestProtocol._capture_fit_mode()`: the requested mode, or the
    process default, stored as `_fit_numeric_mode`."""
    var requested = _check_mode(py_get(fields, "numeric_mode"))
    var mode = requested if requested is not None else context["backend_fields"]["_DEFAULT_MODE"]
    var clean = _check_mode(mode)
    py_setitem(fields, "_fit_numeric_mode", clean)
    return String(clean)


def forest_effective_mode(fields: PythonObject, context: PythonObject) raises -> String:
    """`ForestProtocol._effective_mode()`."""
    var requested = _check_mode(py_get(fields, "numeric_mode"))
    var captured = py_get(fields, "_fit_numeric_mode")
    if captured is not None:
        if requested is not None and not Bool(py=requested == captured):
            raise Error("This forest was fitted with numeric_mode=" + _repr(captured) + "; "
                        + "refit or use set_params before changing numeric_mode")
        return String(captured)
    if requested is not None:
        return String(requested)
    return String(_check_mode(context["backend_fields"]["_DEFAULT_MODE"]))


# ------------------------------------------------------------ fit export


def _list_array(values: PythonObject, typestr: String, context: PythonObject) raises -> PythonObject:
    """`Array.from_list(values, typestr)` of a legacy fit's flat list."""
    var store = py_call(context["array_type"], PythonObject(_store_code(typestr)), values)
    return classical_wrap_array(store, Python.tuple(len(values)), typestr, "C", context)


def _check_count(value: PythonObject, message: String) raises -> Int:
    var b = trees_builtins()
    if _is_bool(value) or not Bool(py=py_call(b.isinstance, value, b.int)):
        raise Error(message)
    return Int(py=value)


def _dtypes() raises -> PythonObject:
    return Python.tuple("<i4", "<i4", "<f4", "<i4", "<f4")


def forest_export(native: PythonObject, descriptor: PythonObject, selection: String,
                  context: PythonObject) raises -> PythonObject:
    """`_forest_protocol._export_fit_result`: one owned native fit copied
    into the five model Arrays, its handle always released. Returns
    `(offsets, colid, quesval, left_child, leaves, meta)`."""
    var b = trees_builtins()
    if not Bool(py=py_call(b.isinstance, descriptor, Python.tuple(b.list, b.tuple))) or len(descriptor) == 0:
        raise Error("native forest export returned an invalid descriptor")
    var handle = descriptor[0]
    if _is_bool(handle) or not Bool(py=py_call(b.isinstance, handle, b.int)) or Int(py=handle) < 1:
        raise Error("native forest export returned an invalid handle")
    var release = trees_need(native, "forest_export_release", "rebuild the forest binding for caller-buffer model export")
    try:
        if len(descriptor) != 5:
            raise Error("native forest export descriptor requires five fields")
        var trees = _check_count(descriptor[1], "native forest export counts must be integers")
        var nodes = _check_count(descriptor[2], "native forest export counts must be integers")
        var outputs = _check_count(descriptor[3], "native forest export counts must be integers")
        var meta = descriptor[4]
        if not (1 <= trees and trees < 2147483647 and nodes >= trees and outputs >= 1
                and nodes <= 2147483647 // outputs):
            raise Error("native forest export exceeds supported counts")
        if not Bool(py=py_call(b.isinstance, meta, Python.tuple(b.list, b.tuple))) or len(meta) < 2 \
                or Int(py=meta[0]) != trees or Int(py=meta[1]) != outputs:
            raise Error("native forest export metadata disagrees with counts")
        for i in range(len(meta)):  # driver-loop: over the export metadata words
            _ = _check_count(meta[i], "native forest export metadata must contain integers")
        var dtypes = _dtypes()
        var sizes = Python.tuple(trees + 1, nodes, nodes, nodes, nodes * outputs)
        var arrays = Python.list()
        for i in range(_MODEL_FIELDS):  # driver-loop: over the five model arrays
            _ = py_call(arrays.append, trees_vector(Int(py=sizes[i]), String(dtypes[i]), context))
        _ = py_call_typed(trees_need(native, "forest_export", "rebuild the forest binding for caller-buffer model export"),
                          handle, PythonObject(trees_address(arrays[0])), PythonObject(trees_address(arrays[1])),
                          PythonObject(trees_address(arrays[2])), PythonObject(trees_address(arrays[3])),
                          PythonObject(trees_address(arrays[4])), Python.list(trees, nodes, outputs))
        if selection == "verify":
            var old = py_call_typed(trees_need(native, "forest_export_legacy", "forest export legacy metadata mismatch"), handle)
            if len(old) != 6 or not Bool(py=py_call(b.list, old[5]) == py_call(b.list, meta)):
                raise Error("RuntimeError: forest export legacy metadata mismatch")
            var names = Python.tuple("_offsets", "_colid", "_quesval", "_left_child", "_leaves")
            for i in range(_MODEL_FIELDS):  # driver-loop: over the five model arrays
                var expected = _list_array(old[i], String(dtypes[i]), context)
                if not Bool(py=py_call(py_call(b.memoryview, arrays[i]._store).tobytes) == py_call(py_call(b.memoryview, expected._store).tobytes)):
                    raise Error("RuntimeError: forest export byte mismatch: " + String(names[i]))
        _ = py_call(arrays.append, py_call(b.list, meta))
        _ = py_call_typed(release, handle)
        return py_call(b.tuple, arrays)
    except error:
        try:
            _ = py_call_typed(release, handle)
        except:
            pass
        raise error^


def forest_fit_arrays(result: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_forest_protocol._forest_fit_arrays`: an exported fit's Arrays as
    they are, a legacy fit's lists as Arrays."""
    var b = trees_builtins()
    if len(result) != 6:
        raise Error("forest fit must return five model fields and metadata")
    var dtypes = _dtypes()
    var out = Python.list()
    for i in range(_MODEL_FIELDS):  # driver-loop: over the five model fields
        var field = result[i]
        if py_type(field) is context["array_class"]:
            if String(field.dtype) != String(dtypes[i]) or Int(py=field.ndim) != 1:
                raise Error("forest fit exported an unexpected model dtype or shape")
            _ = py_call(out.append, field)
        else:
            _ = py_call(out.append, _list_array(field, String(dtypes[i]), context))
    _ = py_call(out.append, result[5])
    return py_call(b.tuple, out)


def _export_selection() raises -> String:
    """`MOJOLEARN_FOREST_EXPORT`: 'into' (default), 'legacy' or 'verify'."""
    var selection = trees_env("MOJOLEARN_FOREST_EXPORT", "into")
    if selection != "legacy" and selection != "into" and selection != "verify":
        raise Error("MOJOLEARN_FOREST_EXPORT must be legacy, into or verify")
    return selection


struct ForestFit(Copyable, Movable):
    """The fit entry an operation launches: its compiled name, whether it
    takes the RF criterion argument, the trailing arguments (weights,
    auxiliary buffers, a shard start) and whether it reads X row-major."""
    var name: String
    var criterion: Bool
    var tail: PythonObject
    var exported: Bool

    def __init__(out self, name: String, criterion: Bool, tail: PythonObject, exported: Bool):
        self.name = name
        self.criterion = criterion
        self.tail = tail
        self.exported = exported


def _launch(native: PythonObject, fit: ForestFit, x: TreesMatrix, y_addr: Int, params: PythonObject,
            criterion: PythonObject, device_scan: Bool, context: PythonObject) raises -> PythonObject:
    """One fit launch and its export (or a legacy fit's lists)."""
    var b = trees_builtins()
    var selection = _export_selection() if fit.exported else String("legacy")
    var name = fit.name + ("_export" if fit.exported and selection != "legacy" else "")
    var entry = trees_need(native, name, "rebuild the forest binding for caller-buffer model export")
    var args = Python.list(PythonObject(x.addr), PythonObject(y_addr), params)
    if fit.criterion:
        _ = py_call(args.append, criterion)
    for i in range(len(fit.tail)):  # small-loop(tail): the entry's trailing argument words
        _ = py_call(args.append, fit.tail[i])
    var out = PythonObject(None)
    try:
        out = trees_call(entry, args)
    except error:
        if device_scan and TREES_NONFINITE in String(error):
            raise Error(TREES_NONFINITE)
        raise error^
    _ = x.keep
    if fit.exported and selection != "legacy":
        return forest_export(native, out, selection, context)
    return out




# ------------------------------------------------------------- fit params


def _max_features_fraction(max_features: PythonObject, n_features: Int, context: PythonObject) raises -> Float64:
    """`randomforest._max_features_fraction`: the fraction the RF binding
    takes. 'sqrt' uses the correctly rounded square root and 'log2' the
    pinned `portable_log2_64` the builder uses, as the Python helper did."""
    var b = trees_builtins()
    if max_features is None:
        return 1.0
    if Bool(py=py_call(b.isinstance, max_features, b.str)):
        if String(max_features) == "sqrt":
            return sqrt(Float64(n_features)) / Float64(n_features)
        if String(max_features) == "log2":
            return portable_log2_64(Float64(max(2, n_features))) / Float64(n_features)
        raise Error("max_features=" + _repr(max_features) + " is not a recognised form;"
                    + " accepted are 'sqrt', 'log2', None, a float fraction or an"
                    + " int count")
    if _is_integral(max_features, context) and not _is_bool(max_features):
        var count = Int(py=_int(max_features))
        if not (1 <= count and count <= n_features):
            raise Error("max_features=" + _str(max_features) + " is outside [1, " + String(n_features) + "]")
        return Float64(count) / Float64(n_features)
    var f = Float64(py=_float(max_features))
    if not (f > 0.0 and f <= 1.0):
        raise Error("max_features=" + _str(PythonObject(f)) + " is outside (0, 1]")
    return f


def _dt_tier_bins(fields: PythonObject, rf: PythonObject) raises -> PythonObject:
    """`_expansion_trees._dt_tier_bins`: the tree's own n_bins, or for
    n_bins=None its binding's `rf_dt_default_bins` (128 without one)."""
    var bins = py_get(fields, "n_bins")
    if bins is not None:
        return _int(bins)
    var query = trees_entry(rf, "rf_dt_default_bins")
    if query is None:
        return PythonObject(128)
    return _int(py_call_typed(query))


def _rf_params(fields: PythonObject, kind: String, rows: Int, cols: Int, classes: Int, rf: PythonObject,
               context: PythonObject) raises -> PythonObject:
    """`_RandomForestBase._fit_params`: the 16-slot list in
    _mojolearn_rf.mojo's order, then min_weight_fraction_leaf when it is
    not 0; DecisionTreeRegressor's slot 7 is its tier's bin count."""
    var cfg = fields["_cfg"]
    var params = Python.list(PythonObject(rows), PythonObject(cols), PythonObject(classes),
                             cfg["n_estimators"], cfg["max_depth"], cfg["max_leaves"],
                             PythonObject(_max_features_fraction(cfg["max_features"], cols, context)),
                             cfg["n_bins"], cfg["min_samples_leaf"], cfg["min_samples_split"],
                             cfg["min_impurity_decrease"], PythonObject(1 if _truthy(cfg["bootstrap"]) else 0),
                             cfg["max_samples"], cfg["random_state"], cfg["n_streams"], cfg["max_batch_size"])
    var mwf = py_get(cfg, "min_weight_fraction_leaf", PythonObject(0.0))
    if _truthy(mwf):
        _ = py_call(params.append, mwf)
    if kind == "dt_regressor":
        py_setitem(params, PythonObject(7), _dt_tier_bins(fields, rf))
    return params


def _et_max_features(max_features: PythonObject, context: PythonObject) raises -> PythonObject:
    """`extratrees._max_features_slots`: (spec, fraction)."""
    var b = trees_builtins()
    var tables = context["trees_et_tables"]
    if max_features is None:
        return Python.tuple(tables["mf_all"], PythonObject(0.0))
    if Bool(py=py_call(b.isinstance, max_features, b.str)):
        if String(max_features) == "sqrt":
            return Python.tuple(tables["mf_sqrt"], PythonObject(0.0))
        if String(max_features) == "log2":
            return Python.tuple(tables["mf_log2"], PythonObject(0.0))
        raise Error("max_features=" + _repr(max_features) + " is not a recognised form; sklearn"
                    + " accepts 'sqrt', 'log2', None, a float fraction or an int count")
    if _is_integral(max_features, context) and not _is_bool(max_features):
        return Python.tuple(_int(max_features), PythonObject(0.0))
    return Python.tuple(PythonObject(0), _float(max_features))


def _et_bootstrap_rows(rows: Int, max_samples: PythonObject, context: PythonObject) raises -> PythonObject:
    """`extratrees._n_samples_bootstrap`."""
    if max_samples is None:
        return PythonObject(0)
    if _is_bool(max_samples):
        raise Error("max_samples must be None, an int >= 1 or a float")
    if _is_integral(max_samples, context):
        if Bool(py=max_samples < PythonObject(1)):
            raise Error("max_samples=" + _str(max_samples) + " must be >= 1 when an int")
        return _int(max_samples)
    var f = Float64(py=_float(max_samples))
    if not f > 0.0:
        raise Error("max_samples=" + _str(max_samples) + " must be > 0 when a float")
    var count = Int(py=_int(PythonObject(f * Float64(rows))))
    return PythonObject(max(count, 1))


def _et_params(fields: PythonObject, cfg: PythonObject, rows: Int, cols: Int, classes: Int,
               context: PythonObject) raises -> PythonObject:
    """`extratrees._fit_params`: the 22-slot list in _mojolearn_trees.mojo's
    order."""
    var device = fields["device"]
    var b = trees_builtins()
    if not Bool(py=py_call(b.isinstance, device, b.str)) or not Bool(py=device == PythonObject("gpu")):
        raise Error("Extra Trees training is GPU-only; device must be 'gpu', got " + _repr(device))
    var slots = _et_max_features(cfg["max_features"], context)
    var depth = cfg["max_depth"]
    var seed = cfg["random_state"]
    var leaves = cfg["max_leaf_nodes"]
    return Python.list(
        PythonObject(rows), PythonObject(cols), PythonObject(classes), _int(cfg["n_estimators"]),
        PythonObject(-1) if depth is None else _int(depth),
        _int(cfg["min_samples_split"]), _int(cfg["min_samples_leaf"]), _float(cfg["min_weight_fraction_leaf"]),
        slots[0], slots[1], _float(cfg["min_impurity_decrease"]),
        PythonObject(1 if _truthy(cfg["bootstrap"]) else 0), PythonObject(1 if _truthy(cfg["oob_score"]) else 0),
        PythonObject(0) if seed is None else _int(seed), PythonObject(1 if _truthy(cfg["warm_start"]) else 0),
        _float(cfg["ccp_alpha"]), PythonObject(0 if cfg["class_weight"] is None else 1),
        PythonObject(0 if cfg["monotonic_cst"] is None else 1), _et_bootstrap_rows(rows, cfg["max_samples"], context),
        PythonObject(-1) if leaves is None else _int(leaves), PythonObject(1), _int(fields["_criterion_code"]),
    )


# -------------------------------------------------------------------- fit


def _device_scan(native: PythonObject, name: String) raises -> Bool:
    var scan = trees_entry(native, name)
    return scan is not None and Int(py=py_call_typed(scan)) == 1


def _store_model(fields: PythonObject, arrays: PythonObject, cols: Int) raises:
    """The fitted model arrays, frozen (the copy `_mojolearn_after_fit`
    took), an empty resident holder, and the dimensions."""
    var names = Python.tuple("_offsets", "_colid", "_quesval", "_left_child", "_leaves")
    for i in range(_MODEL_FIELDS):  # driver-loop: over the five model arrays
        trees_freeze(arrays[i])
        py_setitem(fields, names[i], arrays[i])
    var meta = arrays[5]
    py_setitem(fields, "n_features_in_", PythonObject(cols))
    py_setitem(fields, "_n_trees", _int(meta[0]))
    py_setitem(fields, "_num_outputs", _int(meta[1]))
    py_setitem(fields, "_resident_holder", Python.dict())


def _y_rows_check(n: Int, rows: Int) raises:
    if n != rows:
        raise Error("y has " + String(n) + " rows, X has " + String(rows))


def _x_for(source: PythonObject, rowmajor: Bool, context: PythonObject) raises -> TreesMatrix:
    """`_fit_arrays`' X: the forest layout when the fit has a row-major
    entry, else the column-major float32 copy."""
    if rowmajor:
        return trees_forest_layout(source, "X", context)
    return trees_f32(source, 2, "F", "X", context)


def _finite_or_refuse(x: TreesMatrix, device_scan: Bool, context: PythonObject) raises:
    if not device_scan and not trees_all_finite(x.addr, x.rows * x.cols, False, context):
        raise Error(TREES_NONFINITE)


def _rf_fit_arrays(fields: PythonObject, kind: String, x: TreesMatrix, y_addr: Int, y_rows: Int,
                   classes: Int, var fit: ForestFit, rowmajor: Optional[ForestFit], rf: PythonObject,
                   context: PythonObject) raises -> PythonObject:
    """`_RandomForestBase._fit_arrays` on X already in the entry's layout
    (`_x_for`): the length and finiteness checks, the params, the launch
    and the model export. Returns the fit result tuple."""
    if rowmajor and x.row_major:
        fit = rowmajor.value().copy()
    _y_rows_check(y_rows, x.rows)
    var device_scan = _device_scan(rf, "rf_device_finite_scan")
    _finite_or_refuse(x, device_scan, context)
    var params = _rf_params(fields, kind, x.rows, x.cols, classes, rf, context)
    var result = forest_fit_arrays(_launch(rf, fit, x, y_addr, params, fields["_cfg"]["criterion"], device_scan, context), context)
    _store_model(fields, result, x.cols)
    return result


def _has(native: PythonObject, name: String) raises -> Bool:
    return trees_entry(native, name) is not None


def _rf_entries(rf: PythonObject, base: String) raises -> Optional[ForestFit]:
    """`_rowmajor_fit_function(binding, base)`: the row-major fit, or None
    when the binary predates it."""
    if not _has(rf, base + "_rowmajor_export"):
        return None
    return ForestFit(base + "_rowmajor", True, Python.list(), True)


def rf_fit(model: PythonObject, args: PythonObject, kind: String, context: PythonObject) raises -> PythonObject:
    """RandomForest*/DecisionTree*(splitter='best') `fit` and
    `_fit_with_tree_start`."""
    var b = trees_builtins()
    var X = args[0]
    var y = args[1]
    var sample_weight = args[2]
    var importances_request = args[3]
    var tree_start = args[4]
    if not _is_bool(importances_request):
        raise Error("feature_importances must be a boolean fit-output request")
    var importances = _truthy(importances_request)
    var fields = forest_refresh(model, kind, context)
    var mode = forest_capture_mode(fields, context)
    var rf = trees_table(context, mode, "_mojolearn_rf")
    var cfg = fields["_cfg"]
    if _classifier(kind):
        return _rf_fit_classifier(model, fields, kind, X, y, sample_weight, importances, tree_start, mode, rf, context)
    return _rf_fit_regressor(model, fields, kind, X, y, sample_weight, importances, tree_start, mode, rf, context)


def _rf_fit_classifier(model: PythonObject, fields: PythonObject, kind: String, X: PythonObject, y: PythonObject,
                       sample_weight: PythonObject, importances: Bool, tree_start: PythonObject, mode: String,
                       rf: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = trees_builtins()
    var labels = trees_encode_labels(y, context)
    py_setitem(fields, "classes_", labels.classes)
    var k = len(labels.classes)
    py_setitem(fields, "n_classes_", PythonObject(k))
    if k < 2:
        raise Error("y has fewer than 2 classes")
    var xtrees = trees_table(context, mode, "_mojolearn_x_trees")
    var class_weight = py_get(fields, "class_weight")
    var weights: Optional[TreesMatrix] = None
    if sample_weight is not None:
        weights = trees_weighted_rows(sample_weight, class_weight, labels, xtrees, context)
    elif class_weight is not None:
        weights = trees_class_rows_only(class_weight, labels, xtrees, context)
    var names = Python.tuple("oob_decision_function_", "oob_score_", "feature_importances_")
    for i in range(3):  # small-loop(names): the three requested-output attribute names
        _ = py_call(fields.pop, names[i], PythonObject(None))
    var y_addr = trees_address(labels.codes)
    var cfg = fields["_cfg"]
    if _truthy(cfg["oob_score"]) or importances:
        if tree_start is not None:
            raise Error("auxiliary outputs require a complete forest, not a tree-ID shard")
        if not _has(rf, "rf_classifier_fit_aux_export"):
            raise Error("NotImplementedError: this RF binding does not expose native requested fit outputs")
        var oob = PythonObject(None)
        var score = PythonObject(None)
        if _truthy(cfg["oob_score"]):
            oob = trees_empty(Python.tuple(labels.n, k), "<f8", context)
            score = trees_vector(1, "<f8", context)
        var x = trees_f32(X, 2, "F", "X", context)
        var importance = trees_vector(x.cols, "<f4", context) if importances else PythonObject(None)
        var tail = Python.list(PythonObject(0 if not weights else weights.value().addr),
                               PythonObject(0 if oob is None else trees_address(oob)),
                               PythonObject(0 if score is None else trees_address(score)),
                               PythonObject(0 if importance is None else trees_address(importance)))
        _ = _rf_fit_arrays(fields, kind, x, y_addr, labels.n, k,
                           ForestFit("rf_classifier_fit_aux", True, tail, True), None, rf, context)
        if oob is not None:
            py_setitem(fields, "oob_decision_function_", oob)
            py_setitem(fields, "oob_score_", PythonObject(_F64(score, 0)))
        if importance is not None:
            py_setitem(fields, "feature_importances_", importance)
        _ = weights^
        return model
    if tree_start is not None:
        if not _has(rf, "rf_classifier_fit_shard"):
            raise Error("ImportError: rebuild RF binding for global tree-ID shards")
        var tail = Python.list(tree_start, PythonObject(0 if not weights else weights.value().addr))
        _ = _rf_fit_arrays(fields, kind, trees_f32(X, 2, "F", "X", context), y_addr, labels.n, k,
                           ForestFit("rf_classifier_fit_shard", True, tail, False), None, rf, context)
        _ = weights^
        return model
    if weights:
        if not _has(rf, "rf_classifier_fit_weighted"):
            raise Error("RuntimeError: rebuild the RF binding for class_weight support")
        var tail = Python.list(PythonObject(weights.value().addr))
        _ = _rf_fit_arrays(fields, kind, trees_f32(X, 2, "F", "X", context), y_addr, labels.n, k,
                           ForestFit("rf_classifier_fit_weighted", True, tail, True), None, rf, context)
        _ = weights^
        return model
    _ = _rf_fit_arrays(fields, kind, _x_for(X, Bool(_rf_entries(rf, "rf_classifier_fit")), context),
                       y_addr, labels.n, k, ForestFit("rf_classifier_fit", True, Python.list(), True),
                       _rf_entries(rf, "rf_classifier_fit"), rf, context)
    return model


def _F64(array: PythonObject, index: Int) raises -> Float64:
    return MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=trees_address(array))[index]


def _reduce(addr: Int, count: Int, what: Int, context: PythonObject) raises -> PythonObject:
    """`Array.min()` / `max()` of a float32 vector: the core helper
    `reduce_stat` (0 min, 1 max), sequential in storage order."""
    return py_call_typed(trees_helper(context, "reduce_stat"), PythonObject(addr), PythonObject(0),
                         PythonObject(count), PythonObject(what))


def _check_regression_target(y: TreesMatrix, code: Int, criterion: PythonObject, family: String,
                             context: PythonObject) raises:
    """The deviance criteria's target refusals (RF and ET wording)."""
    var tables = context["trees_rf_tables"] if family == "rf" else context["trees_et_tables"]
    var n = y.rows
    if code == Int(py=tables["poisson"]):
        if not trees_all_finite(y.addr, n, False, context) or Bool(py=_reduce(y.addr, n, 0, context) < PythonObject(0)) \
                or not Bool(py=_reduce(y.addr, n, 1, context) > PythonObject(0)):
            if family == "rf":
                raise Error("criterion='poisson' requires y >= 0 with a positive"
                            + " sum: PoissonGain returns -max() for a non-positive"
                            + " label sum (objectives.cuh:251-253), which would fit"
                            + " a stump silently")
            raise Error("criterion='poisson' requires y >= 0 with a positive sum: the"
                        + " Poisson gain is -max() for a non-positive label sum"
                        + " (objectives.cuh:251-253), which would fit a stump silently")
    elif code == Int(py=tables["gamma"]) or code == Int(py=tables["inverse_gaussian"]):
        if not trees_all_finite(y.addr, n, False, context) or Bool(py=_reduce(y.addr, n, 0, context) <= PythonObject(0)):
            if family == "rf":
                raise Error("criterion=" + _repr(criterion) + " requires y > 0: its"
                            + " gain returns -max() for a non-positive label sum"
                            + " (objectives.cuh:279-281, :306-308), which would fit"
                            + " a stump silently")
            raise Error("criterion=" + _repr(criterion) + " requires y > 0: its gain is"
                        + " -max() for a non-positive label sum, which would fit a stump"
                        + " silently")


def _rf_fit_regressor(model: PythonObject, fields: PythonObject, kind: String, X: PythonObject, y: PythonObject,
                      sample_weight: PythonObject, importances: Bool, tree_start: PythonObject, mode: String,
                      rf: PythonObject, context: PythonObject) raises -> PythonObject:
    var cfg = fields["_cfg"]
    var y32 = trees_f32(y, 1, "C", "y", context)
    var code = Int(py=cfg["criterion"])
    _check_regression_target(y32, code, fields["criterion"], "rf", context)
    var names = Python.tuple("oob_prediction_", "oob_score_", "feature_importances_")
    for i in range(3):  # small-loop(names): the three requested-output attribute names
        _ = py_call(fields.pop, names[i], PythonObject(None))
    if sample_weight is not None:
        if tree_start is not None or _truthy(cfg["oob_score"]) or importances:
            raise Error("NotImplementedError: sample_weight is not implemented together with oob_score, feature "
                        + "importances or a tree-ID shard")
        if not _has(rf, "rf_regressor_fit_weighted_export"):
            raise Error("NotImplementedError: sample_weight for a regression forest needs a forest binding that exports "
                        + "rf_regressor_fit_weighted; this install's binding predates it")
        var w = trees_check_sample_weight(sample_weight, y32.rows, context)
        var w_max = _float(_reduce(w.addr, y32.rows, 1, context))
        var tail = Python.list(PythonObject(w.addr), w_max)
        _ = _rf_fit_arrays(fields, kind, trees_f32(X, 2, "F", "X", context), y32.addr, y32.rows, 0,
                           ForestFit("rf_regressor_fit_weighted", True, tail, True), None, rf, context)
        _ = w^
        _ = y32^
        return model
    if _truthy(cfg["oob_score"]) or importances:
        if tree_start is not None:
            raise Error("auxiliary outputs require a complete forest, not a tree-ID shard")
        if not _has(rf, "rf_regressor_fit_aux_export"):
            raise Error("NotImplementedError: this RF binding does not expose native requested fit outputs")
        var oob = PythonObject(None)
        var score = PythonObject(None)
        if _truthy(cfg["oob_score"]):
            oob = trees_vector(y32.rows, "<f8", context)
            score = trees_vector(1, "<f8", context)
        var x = trees_f32(X, 2, "F", "X", context)
        var importance = trees_vector(x.cols, "<f4", context) if importances else PythonObject(None)
        var tail = Python.list(PythonObject(0 if oob is None else trees_address(oob)),
                               PythonObject(0 if score is None else trees_address(score)),
                               PythonObject(0 if importance is None else trees_address(importance)))
        _ = _rf_fit_arrays(fields, kind, x, y32.addr, y32.rows, 0,
                           ForestFit("rf_regressor_fit_aux", True, tail, True), None, rf, context)
        if oob is not None:
            py_setitem(fields, "oob_prediction_", oob)
            py_setitem(fields, "oob_score_", PythonObject(_F64(score, 0)))
        if importance is not None:
            py_setitem(fields, "feature_importances_", importance)
        _ = y32^
        return model
    if tree_start is not None:
        if not _has(rf, "rf_regressor_fit_shard"):
            raise Error("ImportError: rebuild RF binding for global tree-ID shards")
        _ = _rf_fit_arrays(fields, kind, trees_f32(X, 2, "F", "X", context), y32.addr, y32.rows, 0,
                           ForestFit("rf_regressor_fit_shard", True, Python.list(tree_start), False), None, rf, context)
        _ = y32^
        return model
    _ = _rf_fit_arrays(fields, kind, _x_for(X, Bool(_rf_entries(rf, "rf_regressor_fit")), context),
                       y32.addr, y32.rows, 0, ForestFit("rf_regressor_fit", True, Python.list(), True),
                       _rf_entries(rf, "rf_regressor_fit"), rf, context)
    _ = y32^
    return model


def _et_fit_arrays(fields: PythonObject, cfg: PythonObject, X: PythonObject, y_addr: Int, y_rows: Int,
                   classes: Int, fit: ForestFit, rowmajor: Optional[ForestFit], et: PythonObject,
                   context: PythonObject) raises:
    """`_ExtraTreesBase._fit_arrays`: X in the entry's layout, the length
    and finiteness checks, the 22 params, the launch, the export and the
    fitted state with its four extra-trees fields."""
    var x = _x_for(X, Bool(rowmajor), context)
    var chosen = fit.copy()
    if rowmajor and x.row_major:
        chosen = rowmajor.value().copy()
    _y_rows_check(y_rows, x.rows)
    var device_scan = _device_scan(et, "trees_device_finite_scan")
    _finite_or_refuse(x, device_scan, context)
    var params = _et_params(fields, cfg, x.rows, x.cols, classes, context)
    var result = forest_fit_arrays(_launch(et, chosen, x, y_addr, params, PythonObject(None), device_scan, context), context)
    _store_model(fields, result, x.cols)
    var meta = result[5]
    py_setitem(fields, "depth_cap_bound_", PythonObject(_truthy(meta[2])))
    py_setitem(fields, "max_depth_resolved_", _int(meta[3]))
    py_setitem(fields, "max_features_", _int(meta[4]))
    var bootstrap_rows = _int(meta[5])
    py_setitem(fields, "_n_samples_bootstrap", bootstrap_rows if Int(py=bootstrap_rows) != 0 else PythonObject(None))


def _et_rowmajor(et: PythonObject, base: String, tail: PythonObject) raises -> Optional[ForestFit]:
    if not _has(et, base + "_rowmajor_export"):
        return None
    return ForestFit(base + "_rowmajor", False, tail, True)


def et_fit(model: PythonObject, args: PythonObject, kind: String, context: PythonObject) raises -> PythonObject:
    """ExtraTrees* `fit` and `_fit_with_tree_start`."""
    var b = trees_builtins()
    var X = args[0]
    var y = args[1]
    var sample_weight = args[2]
    var tree_start = args[4]
    var fields = forest_refresh(model, kind, context)
    var mode = forest_capture_mode(fields, context)
    var et = trees_table(context, mode, "_mojolearn_trees")
    var cfg = fields["_cfg"]
    if _classifier(kind):
        var labels = trees_encode_labels(y, context)
        py_setitem(fields, "classes_", labels.classes)
        var k = len(labels.classes)
        py_setitem(fields, "n_classes_", PythonObject(k))
        # `codes.astype('<f4')`: the ET classifier reads float32 class codes.
        var codes = trees_vector(labels.n, "<f4", context)
        trees_cast(trees_address(labels.codes), "<i4", trees_address(codes), "<f4", labels.n, context)
        var class_weight = cfg["class_weight"]
        if sample_weight is not None or class_weight is not None:
            var xtrees = trees_table(context, mode, "_mojolearn_x_trees")
            var weights = trees_weighted_rows(sample_weight, class_weight, labels, xtrees, context)
            if not _has(et, "et_classifier_fit_weighted_export"):
                raise Error("RuntimeError: rebuild the trees binding for weighted ExtraTrees fits")
            var unweighted = py_call(b.dict, cfg)
            py_setitem(unweighted, "class_weight", PythonObject(None))
            if tree_start is not None:
                if not _has(et, "et_classifier_fit_weighted_shard"):
                    raise Error("RuntimeError: rebuild the trees binding for weighted ExtraTrees shard fits")
                _et_fit_arrays(fields, unweighted, X, trees_address(codes), labels.n, k,
                               ForestFit("et_classifier_fit_weighted_shard", False,
                                         Python.list(PythonObject(weights.addr), tree_start), False),
                               None, et, context)
                _ = weights^
                return model
            var tail = Python.list(PythonObject(weights.addr))
            _et_fit_arrays(fields, unweighted, X, trees_address(codes), labels.n, k,
                           ForestFit("et_classifier_fit_weighted", False, tail, True),
                           _et_rowmajor(et, "et_classifier_fit_weighted", tail), et, context)
            _ = weights^
            return model
        if tree_start is not None:
            if not _has(et, "et_classifier_fit_shard"):
                raise Error("ImportError: rebuild ExtraTrees binding for global tree-ID shards")
            _et_fit_arrays(fields, cfg, X, trees_address(codes), labels.n, k,
                           ForestFit("et_classifier_fit_shard", False, Python.list(tree_start), False), None, et, context)
            return model
        _et_fit_arrays(fields, cfg, X, trees_address(codes), labels.n, k,
                       ForestFit("et_classifier_fit", False, Python.list(), True),
                       _et_rowmajor(et, "et_classifier_fit", Python.list()), et, context)
        _ = codes^
        return model
    var y32 = trees_f32(y, 1, "C", "y", context)
    _check_regression_target(y32, Int(py=fields["_criterion_code"]), fields["criterion"], "et", context)
    if sample_weight is not None:
        var weights = trees_check_sample_weight(sample_weight, y32.rows, context)
        if not _has(et, "et_regressor_fit_weighted_export"):
            raise Error("RuntimeError: rebuild the trees binding for weighted ExtraTrees fits")
        if tree_start is not None:
            if not _has(et, "et_regressor_fit_weighted_shard"):
                raise Error("RuntimeError: rebuild the trees binding for weighted ExtraTrees shard fits")
            _et_fit_arrays(fields, cfg, X, y32.addr, y32.rows, 0,
                           ForestFit("et_regressor_fit_weighted_shard", False,
                                     Python.list(PythonObject(weights.addr), tree_start), False), None, et, context)
            _ = weights^
            return model
        var tail = Python.list(PythonObject(weights.addr))
        _et_fit_arrays(fields, cfg, X, y32.addr, y32.rows, 0, ForestFit("et_regressor_fit_weighted", False, tail, True),
                       _et_rowmajor(et, "et_regressor_fit_weighted", tail), et, context)
        _ = weights^
        return model
    if tree_start is not None:
        if not _has(et, "et_regressor_fit_shard"):
            raise Error("ImportError: rebuild ExtraTrees binding for global tree-ID shards")
        _et_fit_arrays(fields, cfg, X, y32.addr, y32.rows, 0,
                       ForestFit("et_regressor_fit_shard", False, Python.list(tree_start), False), None, et, context)
        return model
    _et_fit_arrays(fields, cfg, X, y32.addr, y32.rows, 0, ForestFit("et_regressor_fit", False, Python.list(), True),
                   _et_rowmajor(et, "et_regressor_fit", Python.list()), et, context)
    _ = y32^
    return model


def dt_random_fit(model: PythonObject, args: PythonObject, kind: String, context: PythonObject) raises -> PythonObject:
    """`_expansion_trees._dt_random_fit`: splitter='random' grows ONE
    extra-trees tree. The inner ExtraTrees estimator is assembled with the
    state its constructor stores (its defaults, then the tree's settings),
    fitted by the extra-trees operation above, and its flat model becomes
    this tree's; predict delegates to it."""
    var b = trees_builtins()
    var fields = forest_refresh(model, kind, context)
    var inner_cls = context["trees_et_classifier"] if _classifier(kind) else context["trees_et_regressor"]
    var inner = py_call(b.object.__new__, inner_cls)
    var inner_fields = py_fields(inner)
    _ = py_call(inner_fields.update, context["trees_defaults"][inner_cls])
    var max_features = fields["max_features"]
    var seed = fields["random_state"]
    var settings = Python.dict()
    py_setitem(settings, "n_estimators", PythonObject(1))
    py_setitem(settings, "bootstrap", PythonObject(False))
    py_setitem(settings, "max_features", PythonObject(1.0) if max_features is None else max_features)
    var copied = Python.tuple("max_depth", "min_samples_split", "min_samples_leaf", "max_leaf_nodes",
                              "min_impurity_decrease", "criterion", "min_weight_fraction_leaf")
    for i in range(len(copied)):  # small-loop(copied): the seven settings the inner tree takes as given
        py_setitem(settings, copied[i], fields[copied[i]])
    py_setitem(settings, "random_state", PythonObject(0) if seed is None else seed)
    py_setitem(settings, "numeric_mode", fields["numeric_mode"])
    if _classifier(kind):
        py_setitem(settings, "class_weight", fields["class_weight"])
    _ = py_call(inner_fields.update, settings)
    py_setitem(inner_fields, "_has_constructor_parameters", PythonObject(True))
    var inner_kind = String("et_classifier") if _classifier(kind) else String("et_regressor")
    var inner_args = Python.tuple(args[0], args[1], args[2], PythonObject(False), PythonObject(None))
    _ = et_fit(inner, inner_args, inner_kind, context)
    py_setitem(inner_fields, "_mojolearn_fitted", PythonObject(True))
    py_setitem(fields, "_random_inner", inner)
    var names = Python.tuple("_offsets", "_colid", "_quesval", "_left_child", "_leaves", "_n_trees", "_num_outputs",
                             "n_features_in_")
    for i in range(len(names)):  # small-loop(names): the fitted attributes the tree shares with its inner forest
        py_setitem(fields, names[i], inner_fields[names[i]])
    if py_contains(inner_fields, "classes_"):
        py_setitem(fields, "classes_", inner_fields["classes_"])
        py_setitem(fields, "n_classes_", PythonObject(len(inner_fields["classes_"])))
    return model


def forest_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`fit` / `_fit_with_tree_start` of any connected forest class."""
    var kind = forest_kind(model, context)
    if _family(kind) == "et":
        return et_fit(model, args, kind, context)
    if kind.startswith("dt_"):
        var splitter = py_get(py_fields(model), "splitter")
        if splitter is not None and Bool(py=splitter == PythonObject("random")):
            return dt_random_fit(model, args, kind, context)
    return rf_fit(model, args, kind, context)


# ---------------------------------------------------------------- predict


def _engine(fields: PythonObject, mode: String) raises -> String:
    """`ForestProtocol._prediction_engine()`: IDENTICAL `auto` predicts as
    'sequential', FAST `auto` as 'parallel_groves'."""
    var engine = _check_engine(py_get(fields, "inference_engine", PythonObject("sequential")))
    if engine == "auto":
        return "parallel_groves" if mode == "fast" else "sequential"
    return engine


def _ordered_export(native: PythonObject) raises -> Bool:
    """The binding's `forest_ordered_resident` export (a GPU binding)."""
    return trees_entry(native, "forest_ordered_resident") is not None


def _model_arrays(fields: PythonObject) raises -> PythonObject:
    var names = Python.tuple("_offsets", "_colid", "_quesval", "_left_child", "_leaves")
    var out = Python.list()
    for i in range(_MODEL_FIELDS):  # driver-loop: over the five model arrays
        _ = py_call(out.append, fields[names[i]])
    return py_call(trees_builtins().tuple, out)


def _frozen(array: PythonObject, typestr: String, context: PythonObject) raises -> PythonObject:
    """`_forest_protocol._frozen_copy`: a read-only copy of a flat model
    array, its bytes copied in C."""
    var b = trees_builtins()
    var raw = py_call(py_call(b.memoryview, array._store).tobytes)
    var store = py_call(context["array_type"], PythonObject(_store_code(typestr)))
    _ = py_call(store.frombytes, raw)
    var copy = classical_wrap_array(store, Python.tuple(len(store)), typestr, "C", context)
    trees_freeze(copy)
    return copy


def _resident(model: PythonObject, fields: PythonObject, native: PythonObject, mode: String,
              context: PythonObject) raises -> Int:
    """`ForestProtocol._prepare_resident_forest()` through the owner: the
    device snapshot of the five model arrays, prepared once and reused while
    the arrays, dimensions, mode and aggregation are the same. The snapshot
    is released when the holder drops it (a refit, a new snapshot, the
    model's end). Returns its handle."""
    var b = trees_builtins()
    var engine = _engine(fields, mode)
    var ordered_export = _ordered_export(native)
    if engine != "parallel_groves" and not (engine == "sequential" and ordered_export):
        raise Error("resident forest preparation requires parallel_groves")
    if not py_contains(fields, "_offsets"):
        raise Error("RuntimeError: this estimator is not fitted yet")
    var prepare = trees_entry(native, "forest_prepare_gpu")
    var query = trees_entry(native, "forest_predict_resident_reuse_gpu")
    var release = trees_entry(native, "forest_release_gpu")
    if prepare is None or query is None or release is None:
        raise Error("RuntimeError: rebuild the forest binding for resident parallel_groves inference")
    var features = Int(py=_int(fields["n_features_in_"]))
    var trees = Int(py=_int(fields["_n_trees"]))
    var outputs = Int(py=_int(fields["_num_outputs"]))
    # The aggregation (`_resident_ordered_flag`): strict for the sequential
    # engine, the 32-grove graph named explicitly under IDENTICAL, the
    # binary's default otherwise.
    var ordered = PythonObject(None)
    if ordered_export:
        if engine == "sequential":
            ordered = PythonObject(True)
        elif mode == "identical":
            ordered = PythonObject(False)
    var arrays = _model_arrays(fields)
    var holder = py_get(fields, "_resident_holder")
    if holder is None:
        holder = Python.dict()
        py_setitem(fields, "_resident_holder", holder)
    var cached = py_get(holder, "native")
    if cached is not None:
        var same = cached[2] is native and Int(py=cached[3]) == features and Int(py=cached[4]) == trees \
            and Int(py=cached[5]) == outputs and String(cached[6]) == mode and Bool(py=cached[7] == ordered)
        for i in range(_MODEL_FIELDS):  # driver-loop: over the five model arrays
            same = same and cached[8 + i] is arrays[i]
        if same:
            return Int(py=cached[1])
    var dtypes = _dtypes()
    var frozen = Python.list()
    for i in range(_MODEL_FIELDS):  # driver-loop: over the five model arrays
        var a = arrays[i]
        if py_type(a) is not context["array_class"] or String(a.dtype) != String(dtypes[i]) or Int(py=a.ndim) != 1:
            raise Error("forest model arrays must have their original flat dtypes")
        _ = py_call(frozen.append, a)
    var nodes = Int(py=frozen[1].size)
    var offsets = frozen[0]
    var last = Int(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=trees_address(offsets))[Int(py=offsets.size) - 1]) \
        if Int(py=offsets.size) > 0 else -1
    if Int(py=offsets.size) != trees + 1 or nodes < 1 or Int(py=frozen[2].size) != nodes \
            or Int(py=frozen[3].size) != nodes or Int(py=frozen[4].size) != nodes * outputs or last != nodes:
        raise Error("forest model array shapes do not match metadata")
    var names = Python.tuple("_offsets", "_colid", "_quesval", "_left_child", "_leaves")
    for i in range(_MODEL_FIELDS):  # driver-loop: over the five model arrays
        var a = frozen[i]
        if not Bool(py=a._readonly):
            # A caller retaining a mutable alias cannot alter the snapshot.
            var copy = _frozen(a, String(dtypes[i]), context)
            py_setitem(frozen, PythonObject(i), copy)
            py_setitem(fields, names[i], copy)
    var params = Python.list(PythonObject(features), PythonObject(trees), PythonObject(outputs))
    if ordered is not None:
        _ = py_call(params.append, PythonObject(1 if Bool(py=ordered) else 0))
    var handle = py_call_typed(prepare, PythonObject(trees_address(frozen[0])), PythonObject(trees_address(frozen[1])),
                               PythonObject(trees_address(frozen[2])), PythonObject(trees_address(frozen[3])),
                               PythonObject(trees_address(frozen[4])), params)
    var lifetime = trees_resident_new(release, Int(py=handle))
    var entry = Python.list(lifetime, handle, native, PythonObject(features), PythonObject(trees),
                            PythonObject(outputs), PythonObject(mode), ordered)
    for i in range(_MODEL_FIELDS):  # driver-loop: over the five model arrays
        _ = py_call(entry.append, frozen[i])
    # Replacing the entry drops the previous snapshot's lifetime, which
    # releases it.
    py_setitem(holder, "native", py_call(b.tuple, entry))
    return Int(py=handle)


struct ForestInput(Copyable, Movable):
    var x: TreesMatrix
    var mode: String
    var native: PythonObject

    def __init__(out self, var x: TreesMatrix, mode: String, native: PythonObject):
        self.x = x^
        self.mode = mode
        self.native = native


def _predict_input(model: PythonObject, fields: PythonObject, kind: String, X: PythonObject,
                   context: PythonObject) raises -> ForestInput:
    """`_check_predict_input` / `_vote`'s checks: fitted, X as C-order
    float32, the fitted width."""
    if not py_contains(fields, "_offsets"):
        raise Error("RuntimeError: this estimator is not fitted yet")
    var x = trees_f32(X, 2, "C", "X", context)
    var fitted = Int(py=_int(fields["n_features_in_"]))
    if x.cols != fitted:
        raise Error("X has " + String(x.cols) + " features, fit saw " + String(fitted))
    var mode = forest_effective_mode(fields, context)
    var native = trees_table(context, mode, "_mojolearn_trees" if _family(kind) == "et" else "_mojolearn_rf")
    return ForestInput(x^, mode, native)


def _vote(model: PythonObject, fields: PythonObject, kind: String, staged: ForestInput, sequential_name: String,
          context: PythonObject) raises -> PythonObject:
    """`ForestProtocol._predict_forest`: the rows' votes or values, flat
    float32, through the model's engine."""
    var rows = staged.x.rows
    var outputs = Int(py=_int(fields["_num_outputs"]))
    var out = trees_vector(rows * outputs, "<f4", context)
    var engine = _engine(fields, staged.mode)
    var wrote = PythonObject(None)
    if engine == "sequential" and not _ordered_export(staged.native):
        var function = trees_need(staged.native, sequential_name,
                                  "rebuild the forest binding for inference_engine=" + _repr(PythonObject(engine)))
        var arrays = _model_arrays(fields)
        var dims = Python.list(PythonObject(rows), PythonObject(staged.x.cols), _int(fields["_n_trees"]), PythonObject(outputs))
        var call = Python.list()
        for i in range(_MODEL_FIELDS):  # driver-loop: over the five model arrays
            _ = py_call(call.append, PythonObject(trees_view(arrays[i], "forest model", context).addr))
        _ = py_call(call.append, PythonObject(staged.x.addr))
        _ = py_call(call.append, PythonObject(trees_address(out)))
        _ = py_call(call.append, dims)
        wrote = trees_call(function, call)
        _ = arrays^
    else:
        var handle = _resident(model, fields, staged.native, staged.mode, context)
        wrote = py_call_typed(trees_entry(staged.native, "forest_predict_resident_reuse_gpu"), PythonObject(handle),
                              PythonObject(staged.x.addr), PythonObject(trees_address(out)),
                              Python.list(PythonObject(rows), PythonObject(staged.x.cols), PythonObject(outputs)))
    if Int(py=wrote) != rows:
        raise Error("RuntimeError: " + sequential_name + " wrote " + _str(wrote) + " of " + String(rows) + " rows")
    return out


def _resident_labels(model: PythonObject, fields: PythonObject, staged: ForestInput, context: PythonObject) raises -> PythonObject:
    """`ForestProtocol._predict_forest_labels`: the resident classifier's
    int32 codes, or None where the fused label route does not apply."""
    var engine = _engine(fields, staged.mode)
    var capability = trees_entry(staged.native, "forest_identical_fused_labels")
    if staged.mode == "identical":
        if capability is None or not _truthy(py_call_typed(capability)):
            return PythonObject(None)
        if not (engine == "sequential" and _ordered_export(staged.native)) and engine != "parallel_groves":
            return PythonObject(None)
    elif staged.mode != "fast" or engine != "parallel_groves":
        return PythonObject(None)
    var function = trees_entry(staged.native, "forest_predict_resident_labels_gpu")
    if function is None:
        return PythonObject(None)
    var rows = staged.x.rows
    var out = trees_vector(rows, "<i4", context)
    var handle = _resident(model, fields, staged.native, staged.mode, context)
    var wrote = py_call_typed(function, PythonObject(handle), PythonObject(staged.x.addr), PythonObject(trees_address(out)),
                              Python.list(PythonObject(rows), PythonObject(staged.x.cols), _int(fields["_num_outputs"])))
    if Int(py=wrote) != rows:
        raise Error("RuntimeError: forest label prediction wrote " + _str(wrote) + " of " + String(rows) + " rows")
    return out


def _widen(vote: PythonObject, shape: PythonObject, context: PythonObject) raises -> PythonObject:
    """`vote.astype('<f8')` under `shape`."""
    var n = Int(py=vote.size)
    var out = trees_empty(shape, "<f8", context)
    trees_cast(trees_address(vote), "<f4", trees_address(out), "<f8", n, context)
    return out


def _inner(fields: PythonObject) raises -> PythonObject:
    """A splitter='random' tree's inner extra-trees forest, or None."""
    return py_get(fields, "_random_inner")


def forest_predict_proba(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`predict_proba`: RF and DT the float32 vote fractions; ET the vote
    widened to float64; a random-splitter tree its inner forest's."""
    var fields = py_fields(model)
    var kind = forest_kind(model, context)
    var inner = _inner(fields) if kind.startswith("dt_") else PythonObject(None)
    if inner is not None:
        return forest_predict_proba(inner, args, context)
    var staged = _predict_input(model, fields, kind, args[0], context)
    var outputs = Int(py=_int(fields["_num_outputs"]))
    var shape = Python.tuple(staged.x.rows, outputs)
    if _family(kind) == "et":
        return _widen(_vote(model, fields, kind, staged, "et_predict", context), shape, context)
    return trees_reshape(_vote(model, fields, kind, staged, "rf_predict_proba", context), shape, context)


def forest_predict(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`predict`: a classifier's labels (the resident fused labels, else the
    first-max argmax of the vote) decoded through `classes_`; a regressor's
    values (RF and DT float32, ET float64)."""
    var fields = py_fields(model)
    var kind = forest_kind(model, context)
    var inner = _inner(fields) if kind.startswith("dt_") else PythonObject(None)
    if inner is not None:
        return forest_predict(inner, args, context)
    var staged = _predict_input(model, fields, kind, args[0], context)
    var rows = staged.x.rows
    if _classifier(kind):
        var classes = fields["classes_"]
        var codes = _resident_labels(model, fields, staged, context)
        if codes is not None:
            return trees_decode_labels(classes, codes, context)
        var outputs = Int(py=_int(fields["_num_outputs"]))
        var vote = _vote(model, fields, kind, staged, "et_predict" if _family(kind) == "et" else "rf_predict_proba", context)
        return trees_decode_labels(classes, trees_argmax_rows(vote, rows, outputs, False, context), context)
    if _family(kind) == "et":
        return _widen(_vote(model, fields, kind, staged, "et_predict", context), Python.tuple(rows), context)
    return _vote(model, fields, kind, staged, "rf_predict_reg", context)
