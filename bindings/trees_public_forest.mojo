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
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_fields, py_item
from bindings.py_typed_call import py_call_typed
from bindings.trees_public_inputs import (
    TREES_NONFINITE, TreesLabels, trees_builtins, trees_table,
    trees_helper, trees_reshape, trees_freeze, trees_address,
    trees_forest_x, trees_finite_f32, trees_encode_labels,
    trees_decode_labels, trees_argmax_rows, trees_class_rows_only, trees_weighted_rows, trees_cast,
)
from bindings.admitted_input import (
    AdmittedX, admit_f32, admit_weight_words, admitted_words, admitted_array, admitted_vector, classical_view,
    admitted_finite_refuse, admitted_finite_entry_refuse,
)
from bindings.classical_validate import classical_estimator_name
from bindings.classical_arrays import classical_address
# TOMBSTONE: trees_f32 / trees_forest_layout / trees_all_finite / trees_check_sample_weight / trees_view / trees_vector / trees_empty uses (the trees admission aliases) deleted 2026-11-09 by lane a1b-doors-opt-in-trees-linear; code recoverable at f59d913b0b.
# Restore: experiments/removed/A1B_LABELS_ADMISSION_COPY.patch.
from bindings.trees_side_records import trees_finite_known, trees_finite_mark, trees_side_get, trees_side_put
from bindings.trees_resident_table import (
    TREES_RESIDENT_FOREST, TREES_RESIDENT_FOREST_STRICT, trees_resident_find, trees_resident_put_native, trees_resident_pin, trees_resident_unpin,
    trees_resident_watch, trees_resident_forget,
)
from bindings.native_model_records import model_identity_key, model_generation
from bindings.p3nc_fitted_freeze import fitted_freeze
from bindings.metrics_native_support import metric_is_integral, metric_warning_location
from bindings.classical_warnings import classical_warning
from ensemble.rf_notices import rf_notice_text
from xtrees.trees_forest_run import (
    ForestFit, ForestHandle, FOREST_MODEL_FIELDS, forest_export_selection, forest_export_counts, forest_export_dtype,
    forest_export_verdict_check, forest_fit_notice_codes, trees_refuse, trees_check_features, forest_fit_core,
)
from checks.numerics import portable_log2_64
from bindings.forest_fit_boundary import forest_fit_request_py, forest_request_words_py, forest_request_from_words_py
from core.forest_entry_table import ForestEntryKey, FOREST_TABLE_RF, FOREST_TABLE_ET
from core.trees_entry_table import TreesEntryKey
from core.forest_fit_request import (
    ForestFitArgs, XF_CAP_DEVICE_SCAN, XF_CAP_ORDERED_RESIDENT, XF_CAP_FUSED_LABELS, XF_CAP_RESIDENT_LABELS, XF_CAP_STRICT,
    XF_RESIDENT_RELEASE, XF_STRICT_RELEASE,
)
from core.entry_table import EntryRef
from core.forest_calls import (
    ForestExport, xf_ref, xf_caps, xf_dt_default_bins, xf_fit_dart, xf_export_sizes, xf_export_into,
    xf_export_verify, xf_export_release, xf_predict, xf_resident_prepare, xf_resident_predict, xf_resident_labels,
    xf_strict_prepare, xf_strict_predict, xf_snapshot_release,
)

comptime _INT32_MAX = 2147483647
comptime _MODEL_FIELDS = FOREST_MODEL_FIELDS


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
    return metric_is_integral(value, context)


def _int(value: PythonObject) raises -> PythonObject:
    """Python's `int(value)`, with its refusal."""
    return py_call_typed(trees_builtins().int, value)


def _float(value: PythonObject) raises -> PythonObject:
    """Python's `float(value)`, with its refusal."""
    return py_call_typed(trees_builtins().float, value)


def _refuse(name: String, why: String) raises:
    """`randomforest._refuse` (xtrees/trees_forest_run.mojo)."""
    trees_refuse(name, why)


def _class_fields(cls: PythonObject) raises -> PythonObject:
    var b = trees_builtins()
    return py_call(b.type.__dict__["__dict__"].__get__, cls)


def _mro(cls: PythonObject) raises -> PythonObject:
    var b = trees_builtins()
    return py_call(b.type.__dict__["__mro__"].__get__, cls)


comptime _KIND_SLOT = "trees_forest_kind_v1"


def forest_kind(model: PythonObject, context: PythonObject) raises -> String:
    """The connected schema of the model's class: the nearest registered
    class in its ancestry ("rf_classifier", "dt_regressor", "et_classifier",
    ...)."""
    var cls = py_type(model)
    var table = context["trees_classes"]
    # Once per class (p3-trees): the forest door and the forest predict each
    # walked the class ancestry on every call. The record lives in the native
    # side table keyed weakly on the class and holds the connected class
    # table, the class's `__mro__` tuple and the answer; a reconnection (a
    # new table) or a `__bases__` assignment (a new `__mro__` tuple) walks
    # again.
    var mro = _mro(cls)
    var held = trees_side_get(_KIND_SLOT, cls)
    if held is not None and held[0] is table and held[1] is mro:
        return String(held[2])
    for i in range(len(mro)):  # driver-loop: over the class ancestry metadata
        var kind = py_get(table, mro[i])
        if kind is not None:
            _ = trees_side_put(_KIND_SLOT, cls, Python.tuple(table, mro, kind))
            return String(kind)
    raise Error("TypeError: " + String(cls.__name__) + " is not a connected tree ensemble")


def _family(kind: String) -> String:
    return "et" if kind.startswith("et_") else "rf"


def forest_check_fit_device(model: PythonObject, context: PythonObject) raises:
    """Extra Trees: refuse a non-GPU `device` at fit before any input is
    admitted or converted (scikit-learn's convention: fit validates the
    stored parameters first). The caller has already admitted the stored
    constructor fields as native values, so the comparison runs no Python
    hook. Random forests have no device parameter."""
    if _family(forest_kind(model, context)) != "et":
        return
    var device = py_get(py_fields(model), "device", PythonObject(None))
    var b = trees_builtins()
    if not Bool(py=py_call(b.isinstance, device, b.str)) or not Bool(py=device == PythonObject("gpu")):
        raise Error("Extra Trees training is GPU-only; device must be 'gpu', got " + _repr(device))


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
    if py_type(mode) is b.str:
        # The stored value is usually already clean: no strip()/lower() calls
        # (p3-trees). The value returned compares equal to the cleaned one.
        var exact = String(py=mode)
        if exact == "fast" or exact == "identical":
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
        return py_item(table, criterion)  # tail-call fix (2026-11-12): a std subscript returned directly is emitted as an LLVM tail call; LLVM may drop the borrowed key's stack store (Mojo 1.2.0). Read through the C API.
    if py_contains(unsupported, criterion):
        _refuse("criterion=" + _repr(criterion), String(py_item(unsupported, criterion)))
    raise Error("criterion=" + _repr(criterion) + " is not a " + who + " criterion here; accepted are "
                + _repr(py_call(b.sorted, table)))


def _float_or_absent(value: PythonObject) raises -> Optional[PythonObject]:
    """`float(value)`, or absent when it raises TypeError or ValueError."""
    # Returned from inside try/except, not a PythonObject reassigned in both
    # arms (suspected wrong reference drop under Mojo 1.2.0, the
    # modular/modular#7546 family; runtime-fix-save-forest).
    var b = trees_builtins()
    try:
        return Optional[PythonObject](py_call_typed(b.float, value))
    except error:
        var message = String(error)
        if not message.startswith("TypeError: ") and not message.startswith("ValueError: "):
            raise error^
        return Optional[PythonObject]()


def _min_weight_fraction(value: PythonObject, who: String) raises -> PythonObject:
    """`randomforest._min_weight_fraction`."""
    var f = _float_or_absent(value)
    var ok = Bool(f) and not _is_bool(value)
    if ok:
        var x = Float64(py=f.value())
        ok = x >= 0.0 and x <= 0.5
    if not ok:
        raise Error("The 'min_weight_fraction_leaf' parameter of " + who + " must be a float"
                    + " in the range [0.0, 0.5]. Got " + _repr(value) + " instead.")
    return f.value()


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
        var tail = (String("; 'gini', 'entropy' and its alias 'log_loss' are.") if classifier else
            String("; 'squared_error', 'poisson', 'gamma' and 'inverse_gaussian' are."))
        raise Error("NotImplementedError: criterion=" + _repr(criterion) + " is not implemented"
                    + (" (" + String(reason[1]) + ")" if reason is not None else String("")) + tail)
    var cfg = Python.dict()
    var names = Python.tuple("n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
                             "min_weight_fraction_leaf", "max_features", "max_leaf_nodes",
                             "min_impurity_decrease", "bootstrap", "oob_score", "random_state",
                             "warm_start", "class_weight", "ccp_alpha", "max_samples", "monotonic_cst")
    for i in range(len(names)):  # driver-loop: over fixed call structure, small-loop(names): the sixteen fixed configuration names
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
    for i in range(2):  # driver-loop: over fixed call structure, small-loop(kept): the two bookkeeping field names
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


def forest_key(native: PythonObject) raises -> EntryRef:
    """The forest entry, resolved once per operation (lane
    entry-table-zero-alloc), of the image a tier connected (the bridge
    records it at import as `rf_native_key` / `trees_native_key`): every
    fit, export and predict of the door runs through that image's published
    native entry (core/forest_calls.mojo), not through its Python entries."""
    var family = FOREST_TABLE_RF
    var key = py_get(native, PythonObject("rf_native_key"))
    if key is None:
        family = FOREST_TABLE_ET
        key = py_get(native, PythonObject("trees_native_key"))
    if key is None:
        raise Error("ImportError: mojolearn: the forest binding publishes no native entry table")
    return xf_ref(ForestEntryKey(family, TreesEntryKey(Int(py=key[0]), Int(py=key[1]))))


def _check_count(value: PythonObject, message: String) raises -> Int:
    var b = trees_builtins()
    if _is_bool(value) or not Bool(py=py_call(b.isinstance, value, b.int)):
        raise Error(message)
    return Int(py=value)


def forest_export(native: PythonObject, descriptor: PythonObject, selection: String,
                  context: PythonObject) raises -> PythonObject:
    """`_forest_protocol._export_fit_result` of a fit a Python entry returned
    (the boost door's session and AdaBoost member fits): the descriptor's
    handle (UNPACK), then `forest_export_handle`."""
    var b = trees_builtins()
    if not Bool(py=py_call(b.isinstance, descriptor, Python.tuple(b.list, b.tuple))) or len(descriptor) == 0:
        raise Error("native forest export returned an invalid descriptor")
    var handle = descriptor[0]
    if _is_bool(handle) or not Bool(py=py_call(b.isinstance, handle, b.int)) or Int(py=handle) < 1:
        raise Error("native forest export returned an invalid handle")
    var key = forest_key(native)
    if len(descriptor) != 5:
        xf_export_release(key, Int(py=handle))
        raise Error("native forest export descriptor requires five fields")
    return forest_export_handle(key, Int(py=handle), selection, context)


def forest_export_handle(key: EntryRef, handle: Int, selection: String,
                         context: PythonObject) raises -> PythonObject:
    """One owned native fit copied into the five model Arrays, its handle
    always released. Returns `(offsets, colid, quesval, left_child, leaves,
    meta)`. The sizes, the copy and the `verify` comparison are native
    calls of the image (XF_EXPORT_*)."""
    try:
        var sizes = xf_export_sizes(key, handle)
        var counts = forest_export_counts(sizes)
        var arrays = Python.list()
        var dst = List[Int]()
        for i in range(_MODEL_FIELDS):  # driver-loop: over the five model arrays
            var array = admitted_vector(counts[i], forest_export_dtype(i), context)
            dst.append(trees_address(array))
            _ = py_call(arrays.append, array)
        xf_export_into(key, sizes, dst)
        if selection == "verify":
            forest_export_verdict_check(xf_export_verify(key, handle, dst))
        _ = arrays  # the five Arrays own the addresses the calls above wrote
        var meta = Python.list()
        for k in range(len(sizes.meta)):  # driver-loop: over fixed call structure, small-loop(meta: at most eight export metadata words): PACK
            _ = py_call(meta.append, PythonObject(sizes.meta[k]))
        _ = py_call(arrays.append, meta)
        xf_export_release(key, handle)
        return py_call(trees_builtins().tuple, arrays)
    except error:
        try:
            xf_export_release(key, handle)
        except:
            pass
        raise error^


def forest_fit_request(fields: PythonObject, cfg: PythonObject, kind: String, rows: Int, cols: Int, classes: Int,
                       x_addr: Int, y_addr: Int, fit: ForestFit, native: PythonObject,
                       context: PythonObject) raises -> ForestFitArgs:
    """UNPACK, once per fit: the typed request (`ForestFitArgs`) from the
    fitted configuration (the RF 16/17-slot or the ET 22-slot params list,
    read into int and float views once), the X and y (label codes or
    targets) addresses and the trailing words of `fit` (row weights, the
    tree-ID start, the auxiliary outputs, the largest weight). `cfg` is the
    ET params' configuration (the unweighted copy for a class-weighted ET
    fit); the RF params read `fields["_cfg"]`. The public fit and the
    tree-range shard driver (`forest_shard_prepare`) both build their
    request here; `forest_fit_core` runs it."""
    var params: PythonObject
    var criterion: PythonObject
    if _family(kind) == "et":
        params = _et_params(fields, cfg, rows, cols, classes, context)
        criterion = PythonObject(None)
    else:
        params = _rf_params(fields, kind, rows, cols, classes, native, context)
        criterion = py_item(py_item(fields, PythonObject("_cfg")), PythonObject("criterion"))
    var args = forest_fit_request_py(x_addr, y_addr, params, criterion)
    args.weights = fit.weights
    args.tree_start = fit.tree_start
    args.oob = fit.oob
    args.score = fit.score
    args.importance = fit.importance
    args.weight_max = fit.weight_max
    return args^


def _launch(native: PythonObject, fit: ForestFit, x: AdmittedX, args: ForestFitArgs, device_scan: Bool,
            context: PythonObject) raises -> PythonObject:
    """One fit launch and its export, through the connected image's native
    entry: the typed fit core (`forest_fit_core`, XF_FIT) on the request
    `forest_fit_request` built, then the owned export copied into the model
    Arrays (PACK)."""
    var key = forest_key(native)
    var selection = forest_export_selection()
    var handle = forest_fit_core(key, fit.classifier, fit.rowmajor_ok and x.c_order, args, device_scan)
    _ = x.array
    return forest_export_handle(key, handle, selection, context)


def forest_fit_dart_member(native: PythonObject, dart_handle: Int, cls: Int, rows_addr: Int, cols_addr: Int,
                           params: PythonObject, criterion: PythonObject, context: PythonObject) raises -> PythonObject:
    """A DART member's one-tree fit on the DART session (XF_FIT_DART, the
    former `rf_regressor_fit_dart_export`), exported into the model Arrays."""
    var key = forest_key(native)
    var args = forest_fit_request_py(rows_addr, cols_addr, params, criterion)
    args.session = dart_handle
    args.cls = cls
    return forest_export_handle(key, xf_fit_dart(key, args), "into", context)


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
    # XF_DT_DEFAULT_BINS: the image's bin count (128 in one without its own)
    return PythonObject(xf_dt_default_bins(forest_key(rf)))


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


def _device_scan(native: PythonObject) raises -> Bool:
    """The image's fits refuse a non-finite X cell on the device
    (XF_CAP_DEVICE_SCAN, the former `rf_device_finite_scan` /
    `trees_device_finite_scan`)."""
    return (xf_caps(forest_key(native)) & XF_CAP_DEVICE_SCAN) != 0


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


def _x_for(source: PythonObject, rowmajor: Bool, context: PythonObject) raises -> AdmittedX:
    """`_fit_arrays`' X: the forest layout when the fit has a row-major
    entry, else the column-major float32 copy."""
    var x = trees_forest_x(source, "X", context) if rowmajor else admit_f32(source, 2, "X", context, order="F", finite=False)
    # A float32 copy of an input the public door admitted finite is finite.
    if x.array is not source and trees_finite_known(source):
        trees_finite_mark(x.array)
    return x^


def _finite_or_refuse(x: AdmittedX, device_scan: Bool, context: PythonObject) raises:
    # The public door's classical_check_x already refused NaN and infinity
    # in this X (trees_finite_known; lane perf-trees-owed, 2026-11-06): the
    # second O(n d) scan is skipped for it, and kept for every other caller.
    if device_scan or trees_finite_known(x.array):
        return
    if not trees_finite_f32(x.address, x.rows * x.cols):
        raise Error(TREES_NONFINITE)


def _rf_fit_arrays(fields: PythonObject, kind: String, x: AdmittedX, y_addr: Int, y_rows: Int,
                   classes: Int, fit: ForestFit, rf: PythonObject,
                   context: PythonObject) raises -> PythonObject:
    """`_RandomForestBase._fit_arrays` on X already in the entry's layout
    (`_x_for`): the length and finiteness checks, the params, the launch
    and the model export. Returns the fit result tuple."""
    _y_rows_check(y_rows, x.rows)
    var device_scan = _device_scan(rf)
    _finite_or_refuse(x, device_scan, context)
    var request = forest_fit_request(fields, fields["_cfg"], kind, x.rows, x.cols, classes, x.address, y_addr, fit,
                                     rf, context)
    var result = _launch(rf, fit, x, request, device_scan, context)
    _rf_fit_notices(result[5], context)
    _store_model(fields, result, x.cols)
    return result


def _rf_fit_notices(meta: PythonObject, context: PythonObject) raises:
    """The fit's notices (the third metadata word, an
    `ensemble/rf_notices.mojo` bit mask) as Python UserWarnings, in the
    order the fit reached them, attributed to the caller of `fit`. A
    binding without the word has none to report."""
    if len(meta) < 3:
        return
    var codes = forest_fit_notice_codes(Int(py=meta[2]))
    for i in range(len(codes)):  # driver-loop: over fixed call structure, small-loop(codes): at most the three fit notice bits (PACK)
        classical_warning(rf_notice_text(codes[i]), trees_builtins().UserWarning, context,
                          metric_warning_location(2), "UserWarning")


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
    var weights: Optional[AdmittedX] = None
    if sample_weight is not None:
        weights = trees_weighted_rows(sample_weight, class_weight, labels, xtrees, context)
    elif class_weight is not None:
        weights = trees_class_rows_only(class_weight, labels, xtrees, context)
    var names = Python.tuple("oob_decision_function_", "oob_score_", "feature_importances_")
    for i in range(3):  # driver-loop: over fixed call structure, small-loop(names): the three requested-output attribute names
        _ = py_call(fields.pop, names[i], PythonObject(None))
    var y_addr = trees_address(labels.codes)
    var cfg = fields["_cfg"]
    if _truthy(cfg["oob_score"]) or importances:
        if tree_start is not None:
            raise Error("auxiliary outputs require a complete forest, not a tree-ID shard")
        # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
        var oob = Optional[PythonObject]()
        # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
        var score = Optional[PythonObject]()
        if _truthy(cfg["oob_score"]):
            oob = admitted_array(Python.tuple(labels.n, k), "<f8", context)
            score = admitted_vector(1, "<f8", context)
        var x = _x_for(X, False, context)
        var importance = admitted_vector(x.cols, "<f4", context) if importances else PythonObject(None)
        var fit = ForestFit(True, weights=0 if not weights else weights.value().address,
                            oob=0 if not oob else trees_address(oob.value()),
                            score=0 if not score else trees_address(score.value()),
                            importance=0 if importance is None else trees_address(importance))
        _ = _rf_fit_arrays(fields, kind, x, y_addr, labels.n, k, fit, rf, context)
        _ = labels  # y_addr borrows the encoded codes, not the stored classes_
        _ = importance  # keep the Python object alive through the native call above
        _ = score  # keep the Python object alive through the native call above
        _ = oob  # keep the Python object alive through the native call above
        if Bool(oob):
            py_setitem(fields, "oob_decision_function_", oob.value())
            py_setitem(fields, "oob_score_", PythonObject(_F64(score.value(), 0)))
        if importance is not None:
            py_setitem(fields, "feature_importances_", importance)
        _ = weights^
        return model
    if tree_start is not None:
        var fit = ForestFit(True, weights=0 if not weights else weights.value().address,
                            tree_start=Int(py=tree_start))
        _ = _rf_fit_arrays(fields, kind, _x_for(X, False, context), y_addr, labels.n, k, fit, rf, context)
        _ = labels  # retain shard labels through the address-only fit
        _ = weights^
        return model
    if weights:
        _ = _rf_fit_arrays(fields, kind, _x_for(X, False, context), y_addr, labels.n, k,
                           ForestFit(True, weights=weights.value().address), rf, context)
        _ = labels
        _ = weights^
        return model
    _ = _rf_fit_arrays(fields, kind, _x_for(X, True, context), y_addr, labels.n, k,
                       ForestFit(True, rowmajor_ok=True), rf, context)
    _ = labels
    return model


def _F64(array: PythonObject, index: Int) raises -> Float64:
    return MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=trees_address(array))[index]


def _reduce(addr: Int, count: Int, what: Int, context: PythonObject) raises -> PythonObject:
    """`Array.min()` / `max()` of a float32 vector: the core helper
    `reduce_stat` (0 min, 1 max), sequential in storage order."""
    # call-table: base helper (`_buffer._native`, sibling lane); left as a py_call
    return py_call_typed(trees_helper(context, "reduce_stat"), PythonObject(addr), PythonObject(0),
                         PythonObject(count), PythonObject(what))


def _check_regression_target(y: AdmittedX, code: Int, criterion: PythonObject, family: String,
                             context: PythonObject) raises:
    """The deviance criteria's target refusals (RF and ET wording)."""
    var tables = context["trees_rf_tables"] if family == "rf" else context["trees_et_tables"]
    var n = y.rows
    if code == Int(py=tables["poisson"]):
        if (not trees_finite_f32(y.address, n) or Bool(py=_reduce(y.address, n, 0, context) < PythonObject(0))
                or not Bool(py=_reduce(y.address, n, 1, context) > PythonObject(0))):
            if family == "rf":
                raise Error("criterion='poisson' requires y >= 0 with a positive"
                            + " sum: PoissonGain returns -max() for a non-positive"
                            + " label sum (objectives.cuh:251-253), which would fit"
                            + " a stump silently")
            raise Error("criterion='poisson' requires y >= 0 with a positive sum: the"
                        + " Poisson gain is -max() for a non-positive label sum"
                        + " (objectives.cuh:251-253), which would fit a stump silently")
    elif code == Int(py=tables["gamma"]) or code == Int(py=tables["inverse_gaussian"]):
        if not trees_finite_f32(y.address, n) or Bool(py=_reduce(y.address, n, 0, context) <= PythonObject(0)):
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
    var y32 = admit_f32(y, 1, "y", context, finite=False)
    var code = Int(py=cfg["criterion"])
    _check_regression_target(y32, code, fields["criterion"], "rf", context)
    var names = Python.tuple("oob_prediction_", "oob_score_", "feature_importances_")
    for i in range(3):  # driver-loop: over fixed call structure, small-loop(names): the three requested-output attribute names
        _ = py_call(fields.pop, names[i], PythonObject(None))
    if sample_weight is not None:
        if tree_start is not None or _truthy(cfg["oob_score"]) or importances:
            raise Error("NotImplementedError: sample_weight is not implemented together with oob_score, feature "
                        + "importances or a tree-ID shard")
        var w = admit_weight_words(sample_weight, y32.rows, context)
        var w_max = Float64(py=_float(_reduce(w.address, y32.rows, 1, context)))
        _ = _rf_fit_arrays(fields, kind, _x_for(X, False, context), y32.address, y32.rows, 0,
                           ForestFit(False, weights=w.address, weight_max=w_max), rf, context)
        _ = y32  # keep the Python object alive through the native call above
        _ = w  # keep the Python object alive through the native call above
        _ = w^
        _ = y32^
        return model
    if _truthy(cfg["oob_score"]) or importances:
        if tree_start is not None:
            raise Error("auxiliary outputs require a complete forest, not a tree-ID shard")
        # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
        var oob = Optional[PythonObject]()
        # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
        var score = Optional[PythonObject]()
        if _truthy(cfg["oob_score"]):
            oob = admitted_vector(y32.rows, "<f8", context)
            score = admitted_vector(1, "<f8", context)
        var x = _x_for(X, False, context)
        var importance = admitted_vector(x.cols, "<f4", context) if importances else PythonObject(None)
        var fit = ForestFit(False, oob=0 if not oob else trees_address(oob.value()),
                            score=0 if not score else trees_address(score.value()),
                            importance=0 if importance is None else trees_address(importance))
        _ = _rf_fit_arrays(fields, kind, x, y32.address, y32.rows, 0, fit, rf, context)
        _ = importance  # keep the Python object alive through the native call above
        _ = score  # keep the Python object alive through the native call above
        _ = oob  # keep the Python object alive through the native call above
        if Bool(oob):
            py_setitem(fields, "oob_prediction_", oob.value())
            py_setitem(fields, "oob_score_", PythonObject(_F64(score.value(), 0)))
        if importance is not None:
            py_setitem(fields, "feature_importances_", importance)
        _ = y32^
        return model
    if tree_start is not None:
        _ = _rf_fit_arrays(fields, kind, _x_for(X, False, context), y32.address, y32.rows, 0,
                           ForestFit(False, tree_start=Int(py=tree_start)), rf, context)
        _ = y32^
        return model
    _ = _rf_fit_arrays(fields, kind, _x_for(X, True, context), y32.address, y32.rows, 0,
                       ForestFit(False, rowmajor_ok=True), rf, context)
    _ = y32^
    return model


def _et_fit_arrays(fields: PythonObject, cfg: PythonObject, X: PythonObject, y_addr: Int, y_rows: Int,
                   classes: Int, fit: ForestFit, et: PythonObject,
                   context: PythonObject) raises:
    """`_ExtraTreesBase._fit_arrays`: X in the entry's layout, the length
    and finiteness checks, the 22 params, the launch, the export and the
    fitted state with its four extra-trees fields."""
    var x = _x_for(X, fit.rowmajor_ok, context)
    _y_rows_check(y_rows, x.rows)
    var device_scan = _device_scan(et)
    _finite_or_refuse(x, device_scan, context)
    # "et_forest": an ET kind for `_family` (the ET params do not read the kind)
    var request = forest_fit_request(fields, cfg, "et_forest", x.rows, x.cols, classes, x.address, y_addr, fit, et, context)
    var result = _launch(et, fit, x, request, device_scan, context)
    _store_model(fields, result, x.cols)
    var meta = result[5]
    py_setitem(fields, "depth_cap_bound_", PythonObject(_truthy(meta[2])))
    py_setitem(fields, "max_depth_resolved_", _int(meta[3]))
    py_setitem(fields, "max_features_", _int(meta[4]))
    var bootstrap_rows = _int(meta[5])
    py_setitem(fields, "_n_samples_bootstrap", bootstrap_rows if Int(py=bootstrap_rows) != 0 else PythonObject(None))


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
        var codes = admitted_vector(labels.n, "<f4", context)
        trees_cast(trees_address(labels.codes), "<i4", trees_address(codes), "<f4", labels.n, context)
        _ = codes  # keep the Python object alive through the native call above
        var class_weight = cfg["class_weight"]
        if sample_weight is not None or class_weight is not None:
            var xtrees = trees_table(context, mode, "_mojolearn_x_trees")
            var weights = trees_weighted_rows(sample_weight, class_weight, labels, xtrees, context)
            var unweighted = py_call(b.dict, cfg)
            py_setitem(unweighted, "class_weight", PythonObject(None))
            if tree_start is not None:
                _et_fit_arrays(fields, unweighted, X, trees_address(codes), labels.n, k,
                               ForestFit(True, weights=weights.address, tree_start=Int(py=tree_start)), et, context)
                _ = codes  # keep the Python object alive through the native call above
                _ = weights  # keep the Python object alive through the native call above
                _ = weights^
                return model
            _et_fit_arrays(fields, unweighted, X, trees_address(codes), labels.n, k,
                           ForestFit(True, rowmajor_ok=True, weights=weights.address), et, context)
            _ = codes  # keep the Python object alive through the native call above
            _ = weights^
            return model
        if tree_start is not None:
            _et_fit_arrays(fields, cfg, X, trees_address(codes), labels.n, k,
                           ForestFit(True, tree_start=Int(py=tree_start)), et, context)
            _ = codes  # keep the Python object alive through the native call above
            return model
        _et_fit_arrays(fields, cfg, X, trees_address(codes), labels.n, k,
                       ForestFit(True, rowmajor_ok=True), et, context)
        _ = codes  # keep the Python object alive through the native call above
        _ = codes^
        return model
    var y32 = admit_f32(y, 1, "y", context, finite=False)
    _check_regression_target(y32, Int(py=fields["_criterion_code"]), fields["criterion"], "et", context)
    if sample_weight is not None:
        var weights = admit_weight_words(sample_weight, y32.rows, context)
        if tree_start is not None:
            _et_fit_arrays(fields, cfg, X, y32.address, y32.rows, 0,
                           ForestFit(False, weights=weights.address, tree_start=Int(py=tree_start)), et, context)
            _ = y32  # keep the Python object alive through the native call above
            _ = weights  # keep the Python object alive through the native call above
            _ = weights^
            return model
        _et_fit_arrays(fields, cfg, X, y32.address, y32.rows, 0,
                       ForestFit(False, rowmajor_ok=True, weights=weights.address), et, context)
        _ = y32  # keep the Python object alive through the native call above
        _ = weights^
        return model
    if tree_start is not None:
        _et_fit_arrays(fields, cfg, X, y32.address, y32.rows, 0,
                       ForestFit(False, tree_start=Int(py=tree_start)), et, context)
        _ = y32  # keep the Python object alive through the native call above
        return model
    _et_fit_arrays(fields, cfg, X, y32.address, y32.rows, 0, ForestFit(False, rowmajor_ok=True), et, context)
    _ = y32  # keep the Python object alive through the native call above
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
    for i in range(len(copied)):  # driver-loop: over fixed call structure, small-loop(copied): the seven settings the inner tree takes as given
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
    for i in range(len(names)):  # driver-loop: over fixed call structure, small-loop(names): the fitted attributes the tree shares with its inner forest
        py_setitem(fields, names[i], inner_fields[names[i]])
    if py_contains(inner_fields, "classes_"):
        py_setitem(fields, "classes_", inner_fields["classes_"])
        py_setitem(fields, "n_classes_", PythonObject(len(inner_fields["classes_"])))
    return model


def forest_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`fit` / `_fit_with_tree_start` of any connected forest class."""
    var kind = forest_kind(model, context)
    # A refit releases the previous fit's kept snapshots now (the holder dict
    # the fit replaced did the same), not at the next predict.
    trees_resident_forget(model_identity_key(model))
    var result: PythonObject
    var splitter = py_get(py_fields(model), "splitter") if kind.startswith("dt_") else PythonObject(None)
    if _family(kind) == "et":
        result = et_fit(model, args, kind, context)
    elif splitter is not None and Bool(py=splitter == PythonObject("random")):
        result = dt_random_fit(model, args, kind, context)
    else:
        result = rf_fit(model, args, kind, context)
    _report_engine(model, context)
    return result


def _report_engine(model: PythonObject, context: PythonObject) raises:
    """`inference_engine_`: the engine predict runs for the fitted model
    ('sequential' or 'parallel_groves'; `auto` resolved for the fitted
    numeric mode), set once at fit (predict leaves `__dict__` unchanged)."""
    var fields = py_fields(model)
    if not py_contains(fields, "_offsets"):
        return
    py_setitem(fields, "inference_engine_", PythonObject(_engine(fields, forest_effective_mode(fields, context))))


# ---------------------------------------------------------------- predict


def _engine(fields: PythonObject, mode: String) raises -> String:
    """`ForestProtocol._prediction_engine()`: IDENTICAL `auto` predicts as
    'sequential', FAST `auto` as 'parallel_groves'."""
    var engine = _check_engine(py_get(fields, "inference_engine", PythonObject("sequential")))
    if engine == "auto":
        return "parallel_groves" if mode == "fast" else "sequential"
    return engine


def _ordered_export(native: PythonObject) raises -> Bool:
    """The image takes the resident snapshot's ordered word
    (XF_CAP_ORDERED_RESIDENT, the former `forest_ordered_resident` export
    of a GPU binding)."""
    return (xf_caps(forest_key(native)) & XF_CAP_ORDERED_RESIDENT) != 0


def _model_arrays(fields: PythonObject) raises -> PythonObject:
    var names = Python.tuple("_offsets", "_colid", "_quesval", "_left_child", "_leaves")
    var out = Python.list()
    for i in range(_MODEL_FIELDS):  # driver-loop: over the five model arrays
        _ = py_call(out.append, fields[names[i]])
    return py_call(trees_builtins().tuple, out)


def forest_handle_done(use: ForestHandle):
    """End a predict's use of its snapshot (non-raising: error paths)."""
    if use.kept:
        trees_resident_unpin(use.handle)
        return
    try:
        xf_snapshot_release(use.key, use.release_op, use.handle)
    except:
        pass


def _model_addresses(arrays: PythonObject) raises -> List[Int]:
    """The five exact model Arrays' data addresses (their `_addr` slots)."""
    var model = List[Int]()
    for i in range(_MODEL_FIELDS):  # driver-loop: over the five model arrays
        model.append(trees_address(arrays[i]))
    return model^


def _resident(model: PythonObject, fields: PythonObject, native: PythonObject, mode: String,
              context: PythonObject) raises -> Int:
    """`ForestProtocol._prepare_resident_forest()` through the owner: the
    device snapshot of the five model arrays, prepared once and kept in the
    tree resident table (bindings/trees_resident_table.mojo) while the
    model, its generation, the arrays, dimensions, mode and aggregation are
    the same. Returns its handle, or 0 when nothing could be kept (the
    snapshot was released again)."""
    var engine = _engine(fields, mode)
    var key = forest_key(native)
    var ordered_export = (xf_caps(key) & XF_CAP_ORDERED_RESIDENT) != 0
    var use = _resident_with(model, fields, native, key, mode, engine, ordered_export, context)
    var handle = use.handle if use.kept else 0
    forest_handle_done(use)
    return handle


def _resident_with(model: PythonObject, fields: PythonObject, native: PythonObject, key: EntryRef, mode: String,
                   engine: String, ordered_export: Bool, context: PythonObject) raises -> ForestHandle:
    """`_resident` with the engine and the ordered export its caller already
    resolved (ForestInput: read once per predict). The returned handle is
    pinned (or owned by the call); the caller ends it with
    `forest_handle_done` after its launch.

    Workstream A (ne1-a-trees, 2026-11-06): the snapshot lives in the native
    resident table keyed by (model_identity_key, model_generation), never in
    the model's `__dict__` (it was `fields["_resident_holder"]`). The mirrored
    arrays are frozen in place (`fitted_freeze`) instead of copied, so an
    in-place NumPy write raises and the snapshot stays the same bytes as the
    public arrays; replacing an array changes its identity and the next call
    uploads again. No bit change: `forest_prepare_gpu` reads the same five
    arrays it read before."""
    if engine != "parallel_groves" and not (engine == "sequential" and ordered_export):
        raise Error("resident forest preparation requires parallel_groves")
    if not py_contains(fields, "_offsets"):
        raise Error("RuntimeError: this estimator is not fitted yet")
    # perf-audit-3: the caller's entry (ForestInput.key, resolved once per
    # predict) is passed in, not resolved again here.
    var features = Int(py=_int(fields["n_features_in_"]))
    var trees = Int(py=_int(fields["_n_trees"]))
    var outputs = Int(py=_int(fields["_num_outputs"]))
    # The aggregation (`_resident_ordered_flag`): strict for the sequential
    # engine, the 32-grove graph named explicitly under IDENTICAL, the
    # binary's default otherwise. Word: -1 the default, 1 strict, 0 groves.
    var ordered = -1
    if ordered_export:
        if engine == "sequential":
            ordered = 1
        elif mode == "identical":
            ordered = 0
    var arrays = _model_arrays(fields)
    var mirrors = List[PythonObject]()
    for i in range(_MODEL_FIELDS):  # driver-loop: over the five model arrays
        mirrors.append(arrays[i])
    var words = List[Int]()
    words.append(features)
    words.append(trees)
    words.append(outputs)
    words.append(1 if mode == "fast" else 0)
    words.append(ordered)
    var model_key = model_identity_key(model)
    var generation = model_generation(model)
    var held = trees_resident_find(model_key, generation, TREES_RESIDENT_FOREST, native, words, mirrors)
    if held > 0:
        trees_resident_pin(held)
        return ForestHandle(held, True, key, XF_RESIDENT_RELEASE)
    for i in range(_MODEL_FIELDS):  # driver-loop: over the five model arrays
        var a = arrays[i]
        if py_type(a) is not context["array_class"] or String(a.dtype) != forest_export_dtype(i) or Int(py=a.ndim) != 1:
            raise Error("forest model arrays must have their original flat dtypes")
    var nodes = Int(py=arrays[1].size)
    var offsets = arrays[0]
    var last = (Int(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=trees_address(offsets))[Int(py=offsets.size) - 1])
        if Int(py=offsets.size) > 0 else -1)
    _ = offsets  # keep the Python object alive through the native call above
    if (Int(py=offsets.size) != trees + 1 or nodes < 1 or Int(py=arrays[2].size) != nodes
            or Int(py=arrays[3].size) != nodes or Int(py=arrays[4].size) != nodes * outputs or last != nodes):
        raise Error("forest model array shapes do not match metadata")
    _ = offsets  # keep the Python object alive through the native calls above
    for i in range(_MODEL_FIELDS):  # driver-loop: over the five model arrays
        # The read-only rule (p3nc_fitted_freeze): every mirrored exact Array
        # is read-only from here on (the fit stored them frozen already; an
        # Array assigned later is frozen in place, not copied).
        _ = fitted_freeze(arrays[i], context)
    var handle = xf_resident_prepare(key, _model_addresses(arrays), features, trees, outputs, ordered)
    _ = arrays  # keep the Python objects alive through the native call above
    # The snapshot's device bytes: the five flat arrays, 4 bytes an item.
    var bytes = 4 * ((trees + 1) + 3 * nodes + nodes * outputs)
    var kept = False
    try:
        if trees_resident_watch(model, model_key):
            kept = trees_resident_put_native(model_key, generation, TREES_RESIDENT_FOREST, native, key.call,
                                             XF_RESIDENT_RELEASE, handle, words^, mirrors^, bytes)
        if kept:
            trees_resident_pin(handle)
    except error:
        xf_snapshot_release(key, XF_RESIDENT_RELEASE, handle)
        raise error^
    return ForestHandle(handle, kept, key, XF_RESIDENT_RELEASE)


def _strict_vote(model: PythonObject, fields: PythonObject, staged: ForestInput, dest: PythonObject,
                 context: PythonObject) raises -> Int:
    """The sequential vote of a host binding over its strict snapshot
    (bindings/forest_host_strict_binding.mojo), kept in the tree resident
    table like the device snapshots: keyed by the model's identity and
    generation, the dimensions and the identities of the five frozen model
    arrays. Returns the rows written."""
    var native = staged.native
    var key = staged.key.copy()
    var features = Int(py=_int(fields["n_features_in_"]))
    var trees = Int(py=_int(fields["_n_trees"]))
    var outputs = staged.outputs
    var arrays = _model_arrays(fields)
    var mirrors = List[PythonObject]()
    for i in range(_MODEL_FIELDS):  # driver-loop: over the five model arrays
        mirrors.append(arrays[i])
    var words = List[Int]()
    words.append(features)
    words.append(trees)
    words.append(outputs)
    var model_key = model_identity_key(model)
    var generation = model_generation(model)
    var handle = trees_resident_find(model_key, generation, TREES_RESIDENT_FOREST_STRICT, native, words, mirrors)
    var kept = handle > 0
    if not kept:
        for i in range(_MODEL_FIELDS):  # driver-loop: over the five model arrays
            var a = arrays[i]
            if py_type(a) is not context["array_class"] or String(a.dtype) != forest_export_dtype(i) or Int(py=a.ndim) != 1:
                raise Error("forest model arrays must have their original flat dtypes")
            _ = fitted_freeze(a, context)
        handle = xf_strict_prepare(key, _model_addresses(arrays), features, trees, outputs)
        _ = arrays  # keep the Python objects alive through the native call above
        var nodes = Int(py=arrays[1].size)
        var bytes = 4 * ((trees + 1) + 3 * nodes + nodes * outputs)
        try:
            if trees_resident_watch(model, model_key):
                kept = trees_resident_put_native(model_key, generation, TREES_RESIDENT_FOREST_STRICT, native,
                                                 key.call, XF_STRICT_RELEASE, handle, words^, mirrors^, bytes)
        except error:
            xf_snapshot_release(key, XF_STRICT_RELEASE, handle)
            raise error^
    if kept:
        trees_resident_pin(handle)
    var use = ForestHandle(handle, kept, key, XF_STRICT_RELEASE)
    var wrote = 0
    try:
        wrote = xf_strict_predict(key, handle, staged.x.address, trees_address(dest), staged.x.rows, staged.x.cols,
                                  outputs)
    finally:
        forest_handle_done(use)
    _ = dest  # keep the Python object alive through the native call above
    return wrote


struct ForestInput(Copyable, Movable):
    """One predict's resolved state, read once (lane perf-trees-owed,
    2026-11-06; `_int`, `_engine` and `_ordered_export` were re-read two or
    three times per predict): X, the effective mode, the binding, the
    engine, whether the binding has the ordered resident export, and the
    fitted output width."""
    var x: AdmittedX
    var mode: String
    var native: PythonObject
    var key: EntryRef
    var caps: Int
    var engine: String
    var ordered_export: Bool
    var outputs: Int

    def __init__(out self, var x: AdmittedX, mode: String, native: PythonObject, key: EntryRef, caps: Int,
                 engine: String, outputs: Int):
        self.x = x^
        self.mode = mode
        self.native = native
        self.key = key.copy()
        self.caps = caps
        self.engine = engine
        self.ordered_export = (caps & XF_CAP_ORDERED_RESIDENT) != 0
        self.outputs = outputs


def _stage(fields: PythonObject, kind: String, var x: AdmittedX, context: PythonObject) raises -> ForestInput:
    """The resolved predict state of a fitted model over an admitted X."""
    if not py_contains(fields, "_offsets"):
        raise Error("RuntimeError: this estimator is not fitted yet")
    var mode = forest_effective_mode(fields, context)
    var native = trees_table(context, mode, "_mojolearn_trees" if _family(kind) == "et" else "_mojolearn_rf")
    var engine = _engine(fields, mode)
    var key = forest_key(native)
    var caps = xf_caps(key)
    var outputs = Int(py=_int(fields["_num_outputs"]))
    return ForestInput(x^, mode, native, key, caps, engine, outputs)


def _predict_input(model: PythonObject, fields: PythonObject, kind: String, X: PythonObject,
                   context: PythonObject) raises -> ForestInput:
    """`_check_predict_input` / `_vote`'s checks: fitted, X as C-order
    float32, the fitted width."""
    if not py_contains(fields, "_offsets"):
        raise Error("RuntimeError: this estimator is not fitted yet")
    var x = admit_f32(X, 2, "X", context, finite=False)
    var fitted = Int(py=_int(fields["n_features_in_"]))
    trees_check_features(x.cols, fitted)
    return _stage(fields, kind, x^, context)


def _vote(model: PythonObject, fields: PythonObject, kind: String, staged: ForestInput, sequential_name: String,
          context: PythonObject) raises -> PythonObject:
    """`ForestProtocol._predict_forest`: the rows' votes or values, flat
    float32, through the model's engine."""
    var rows = staged.x.rows
    var outputs = staged.outputs
    var out = admitted_vector(rows * outputs, "<f4", context)
    var engine = staged.engine
    var wrote = 0
    if engine == "sequential" and not staged.ordered_export and (staged.caps & XF_CAP_STRICT) != 0:
        # The host column's kept trees (N044): the same sequential vote over
        # trees rebuilt once per fitted model, not once per call.
        wrote = _strict_vote(model, fields, staged, out, context)
    elif engine == "sequential" and not staged.ordered_export:
        var arrays = _model_arrays(fields)
        var model_addrs = List[Int]()
        var array_class = context["array_class"]
        var views = List[PythonObject]()
        for i in range(_MODEL_FIELDS):  # driver-loop: over the five model arrays
            # An exact mojolearn Array's data address is its `_addr` slot, the
            # one `trees_view` reads; any other storage is viewed as before
            # (UNPACK: the view is kept alive through the vote below).
            var a = arrays[i]
            if py_type(a) is array_class:
                model_addrs.append(trees_address(a))
            else:
                var view = classical_view(a, context, "forest model")
                model_addrs.append(Int(py=classical_address(view)))
                views.append(view)
        wrote = xf_predict(staged.key, model_addrs, staged.x.address, trees_address(out), rows, staged.x.cols,
                           Int(py=_int(fields["_n_trees"])), outputs, sequential_name == "rf_predict_reg")
        _ = out  # keep the Python object alive through the native call above
        _ = views^
        _ = arrays^
    else:
        var use = _resident_with(model, fields, staged.native, staged.key, staged.mode, engine, staged.ordered_export, context)
        try:
            wrote = xf_resident_predict(staged.key, use.handle, staged.x.address, trees_address(out), rows,
                                        staged.x.cols, outputs)
        finally:
            forest_handle_done(use)
        _ = out  # keep the Python object alive through the native call above
    if wrote != rows:
        raise Error("RuntimeError: " + sequential_name + " wrote " + String(wrote) + " of " + String(rows) + " rows")
    return out


def _fused_labels_capable(staged: ForestInput) -> Bool:
    """The image's compiled T38 fused labels (XF_CAP_FUSED_LABELS, read
    once per predict with the other capability bits; the former
    `forest_identical_fused_labels` entry and its per-binary answer cache)."""
    return (staged.caps & XF_CAP_FUSED_LABELS) != 0


def _resident_labels(model: PythonObject, fields: PythonObject, staged: ForestInput, context: PythonObject) raises -> PythonObject:
    """`ForestProtocol._predict_forest_labels`: the resident classifier's
    int32 codes, or None where the fused label route does not apply."""
    var engine = staged.engine
    if staged.mode == "identical":
        if not _fused_labels_capable(staged):
            return PythonObject(None)
        if not (engine == "sequential" and staged.ordered_export) and engine != "parallel_groves":
            return PythonObject(None)
    elif staged.mode != "fast" or engine != "parallel_groves":
        return PythonObject(None)
    if (staged.caps & XF_CAP_RESIDENT_LABELS) == 0:
        return PythonObject(None)
    var rows = staged.x.rows
    var out = admitted_vector(rows, "<i4", context)
    var use = _resident_with(model, fields, staged.native, staged.key, staged.mode, engine, staged.ordered_export, context)
    var wrote = 0
    try:
        wrote = xf_resident_labels(staged.key, use.handle, staged.x.address, trees_address(out), rows, staged.x.cols,
                                   staged.outputs)
    finally:
        forest_handle_done(use)
    _ = out  # keep the Python object alive through the native call above
    if wrote != rows:
        raise Error("RuntimeError: forest label prediction wrote " + String(wrote) + " of " + String(rows) + " rows")
        _ = out  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native calls above
    return out


def _widen(vote: PythonObject, shape: PythonObject, context: PythonObject) raises -> PythonObject:
    """`vote.astype('<f8')` under `shape`."""
    var n = Int(py=vote.size)
    var out = admitted_array(shape, "<f8", context)
    trees_cast(trees_address(vote), "<f4", trees_address(out), "<f8", n, context)
    _ = out  # keep the Python object alive through the native call above
    return out


def _inner(fields: PythonObject) raises -> PythonObject:
    """A splitter='random' tree's inner extra-trees forest, or None."""
    return py_get(fields, "_random_inner")


def _proba_staged(model: PythonObject, fields: PythonObject, kind: String, staged: ForestInput,
                  context: PythonObject) raises -> PythonObject:
    var shape = Python.tuple(staged.x.rows, staged.outputs)
    if _family(kind) == "et":
        return _widen(_vote(model, fields, kind, staged, "et_predict", context), shape, context)
    return trees_reshape(_vote(model, fields, kind, staged, "rf_predict_proba", context), shape, context)


def _codes_staged(model: PythonObject, fields: PythonObject, kind: String, staged: ForestInput,
                  context: PythonObject) raises -> PythonObject:
    """A classifier's undecoded class codes: the resident fused labels
    (int32), else the first-max argmax of the vote (int64)."""
    var codes = _resident_labels(model, fields, staged, context)
    if codes is not None:
        return codes
    var vote = _vote(model, fields, kind, staged, "et_predict" if _family(kind) == "et" else "rf_predict_proba", context)
    return trees_argmax_rows(vote, staged.x.rows, staged.outputs, False, context)


def _predict_staged(model: PythonObject, fields: PythonObject, kind: String, staged: ForestInput,
                    context: PythonObject) raises -> PythonObject:
    if _classifier(kind):
        return trees_decode_labels(fields["classes_"], _codes_staged(model, fields, kind, staged, context), context)
    if _family(kind) == "et":
        return _widen(_vote(model, fields, kind, staged, "et_predict", context), Python.tuple(staged.x.rows), context)
    return _vote(model, fields, kind, staged, "rf_predict_reg", context)


def forest_predict_proba(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`predict_proba`: RF and DT the float32 vote fractions; ET the vote
    widened to float64; a random-splitter tree its inner forest's."""
    # A random-splitter tree answers through its inner forest: followed in a
    # loop, not by self-recursion (ne-fix-hang2; kept on its own merits,
    # the compile hang was elsewhere); same model reached, same calls on it.
    # Optional, not a PythonObject read and reassigned in a loop (modular/modular#7546 pattern; runtime-fix-save-forest sweep).
    var walked = Optional[PythonObject](model)
    var walked_fields = Optional[PythonObject](py_fields(model))
    var kind = forest_kind(model, context)
    var inner = Optional[PythonObject](_inner(walked_fields.value()) if kind.startswith("dt_") else PythonObject(None))
    while inner.value() is not None:  # driver-loop: over a tree's inner-forest chain
        walked = Optional[PythonObject](inner.value())
        walked_fields = Optional[PythonObject](py_fields(walked.value()))
        kind = forest_kind(walked.value(), context)
        inner = Optional[PythonObject](_inner(walked_fields.value()) if kind.startswith("dt_") else PythonObject(None))
    var target = walked.value()
    var fields = walked_fields.value()
    var staged = _predict_input(target, fields, kind, args[0], context)
    return _proba_staged(target, fields, kind, staged, context)


def forest_predict(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`predict`: a classifier's labels (the resident fused labels, else the
    first-max argmax of the vote) decoded through `classes_`; a regressor's
    values (RF and DT float32, ET float64)."""
    # A random-splitter tree answers through its inner forest: followed in a
    # loop, not by self-recursion (ne-fix-hang2; kept on its own merits,
    # the compile hang was elsewhere); same model reached, same calls on it.
    # Optional, not a PythonObject read and reassigned in a loop (modular/modular#7546 pattern; runtime-fix-save-forest sweep).
    var walked = Optional[PythonObject](model)
    var walked_fields = Optional[PythonObject](py_fields(model))
    var kind = forest_kind(model, context)
    var inner = Optional[PythonObject](_inner(walked_fields.value()) if kind.startswith("dt_") else PythonObject(None))
    while inner.value() is not None:  # driver-loop: over a tree's inner-forest chain
        walked = Optional[PythonObject](inner.value())
        walked_fields = Optional[PythonObject](py_fields(walked.value()))
        kind = forest_kind(walked.value(), context)
        inner = Optional[PythonObject](_inner(walked_fields.value()) if kind.startswith("dt_") else PythonObject(None))
    var target = walked.value()
    var fields = walked_fields.value()
    var staged = _predict_input(target, fields, kind, args[0], context)
    return _predict_staged(target, fields, kind, staged, context)


def forest_predict_codes(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """A forest classifier's `predict` before its decode:
    `(classes_, codes)`, the codes an int32 or int64 Array indexing
    `classes_`. Internal callers that work in codes (AdaBoost's members,
    the string-class accuracy of `score`) skip the decode and re-encode."""
    # A random-splitter tree answers through its inner forest: followed in a
    # loop, not by self-recursion (ne-fix-hang2; kept on its own merits,
    # the compile hang was elsewhere); same model reached, same calls on it.
    # Optional, not a PythonObject read and reassigned in a loop (modular/modular#7546 pattern; runtime-fix-save-forest sweep).
    var walked = Optional[PythonObject](model)
    var walked_fields = Optional[PythonObject](py_fields(model))
    var kind = forest_kind(model, context)
    var inner = Optional[PythonObject](_inner(walked_fields.value()) if kind.startswith("dt_") else PythonObject(None))
    while inner.value() is not None:  # driver-loop: over a tree's inner-forest chain
        walked = Optional[PythonObject](inner.value())
        walked_fields = Optional[PythonObject](py_fields(walked.value()))
        kind = forest_kind(walked.value(), context)
        inner = Optional[PythonObject](_inner(walked_fields.value()) if kind.startswith("dt_") else PythonObject(None))
    var target = walked.value()
    var fields = walked_fields.value()
    if not _classifier(kind):
        raise Error("TypeError: forest class codes require a classifier")
    var staged = _predict_input(target, fields, kind, args[0], context)
    return Python.tuple(fields["classes_"], _codes_staged(target, fields, kind, staged, context))


def forest_public_predict(model: PythonObject, kind: String, x: AdmittedX, proba: Bool, codes: Bool,
                          context: PythonObject) raises -> PythonObject:
    """The public door's predict over the X it already admitted and checked
    (classical_check_x: C-order float32; its width against the fitted width;
    the fitted check): no second conversion, view or width check (lane
    perf-trees-owed, 2026-11-06; X was viewed and checked twice per call).
    `codes` returns `(classes_, codes)` as forest_predict_codes does. A
    random-splitter tree predicts through its inner forest, whose own width
    and fitted state are checked as before."""
    var fields = py_fields(model)
    var inner = _inner(fields) if kind.startswith("dt_") else PythonObject(None)
    if inner is not None:
        var inner_fields = py_fields(inner)
        var inner_kind = forest_kind(inner, context)
        if not py_contains(inner_fields, "_offsets"):
            raise Error("RuntimeError: this estimator is not fitted yet")
        var fitted = Int(py=_int(inner_fields["n_features_in_"]))
        trees_check_features(x.cols, fitted)
        return _public_deferred(model, inner, inner_fields, inner_kind,
                                _stage(inner_fields, inner_kind, x.copy(), context), proba, codes, context)
    return _public_deferred(model, model, fields, kind, _stage(fields, kind, x.copy(), context), proba, codes, context)


def _public_deferred(public: PythonObject, model: PythonObject, fields: PythonObject, kind: String,
                     staged: ForestInput, proba: Bool, codes: Bool, context: PythonObject) raises -> PythonObject:
    """`_public_staged` with the deferred finite test of the door's X (lane
    a7c): the public door admits X with `defer_finite`, so a borrowed X
    reaches here untested (`finite` False). Every resident engine tests the
    words it uploads (the device scan after the upload on the GPU, the
    row tasks' scan of the host groves on the CPU column) and refuses in
    its own words, which the one scikit-learn wording replaces; the
    sequential engines (`forest_strict_predict`, `rf_predict_*`,
    `et_predict`) test nothing, so a deferred X takes the door's parallel
    host scan before them (the scan admit_x made before, now only there).
    Refusals name the public estimator (`public`)."""
    var est = classical_estimator_name(public)
    if not staged.x.finite and staged.engine == "sequential" and not staged.ordered_export:
        if not trees_finite_f32(staged.x.address, staged.x.rows * staged.x.cols):
            admitted_finite_refuse(1, est)
    var result = PythonObject(None)
    try:
        result = _public_staged(model, fields, kind, staged, proba, codes, context)
    except error:
        if not staged.x.finite:
            admitted_finite_entry_refuse(error, est)
        raise error^
    return result


def _public_staged(model: PythonObject, fields: PythonObject, kind: String, staged: ForestInput, proba: Bool,
                   codes: Bool, context: PythonObject) raises -> PythonObject:
    if proba:
        return _proba_staged(model, fields, kind, staged, context)
    if codes:
        return Python.tuple(fields["classes_"], _codes_staged(model, fields, kind, staged, context))
    return _predict_staged(model, fields, kind, staged, context)
