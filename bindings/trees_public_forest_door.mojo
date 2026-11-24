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
from bindings.mode_answer_door import answer_mode_object
from bindings.labels_descriptor_boundary_door import labels_scope
from core.mode_answer import MODE_FAMILY_FOREST
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_fields, py_item
from providers.python.device_route_boundary_door import device_unpack
from bindings.py_typed_call import py_call_typed
from bindings.trees_public_inputs_door import (
    TreesLabels,
    trees_builtins,
    trees_table,
    trees_helper,
    trees_reshape,
    trees_freeze,
    trees_address,
    trees_forest_x,
    trees_encode_labels,
    trees_decode_labels,
    trees_argmax_rows,
    trees_class_rows_only,
    trees_weighted_rows,
    trees_cast,
)
from bindings.trees_public_inputs_run import (
    TREES_NONFINITE,
    trees_finite_f32,
)
from bindings.admitted_input_unpack import (
    AdmittedX,
    admit_f32,
    admit_weight_words,
    admitted_words,
    admitted_array,
    admitted_vector,
    classical_view,
    admitted_finite_refuse,
)
from bindings.admitted_input_run import admitted_finite_entry_refuse
from bindings.admitted_input_run import AdmittedXView
from bindings.classical_validate_unpack import classical_estimator_name, classical_check_y
from bindings.classical_arrays_door import classical_address
# TOMBSTONE: trees_f32 / trees_forest_layout / trees_all_finite / trees_check_sample_weight / trees_view / trees_vector / trees_empty uses (the trees admission aliases) deleted 2026-11-09 by lane a1b-doors-opt-in-trees-linear; code recoverable at f59d913b0b.
# Restore: experiments/removed/A1B_LABELS_ADMISSION_COPY.patch.
from bindings.trees_side_records_unpack import trees_finite_known, trees_finite_mark, trees_side_get, trees_side_put
from bindings.py_trees_resident import (
    TREES_RESIDENT_FOREST, TREES_RESIDENT_FOREST_STRICT, trees_resident_find, trees_resident_keep, trees_resident_ids,
    trees_resident_pin, trees_resident_unpin, trees_resident_watch, trees_resident_forget,
)
from bindings.py_model_records import model_identity_key, model_generation
from bindings.p3nc_fitted_freeze_unpack import fitted_freeze
from bindings.metrics_native_support_door import metric_is_integral, metric_warning_location
from bindings.classical_warnings_unpack import classical_warning
from bindings.rf_fit_notices_pack import rf_fit_notices_word
from bindings.forest_fit_loop_pack import forest_fit_loop_store
from xtrees.trees_forest_run import (
    _refuse, _family, _classifier, _y_rows_check,
    ForestFit, ForestHandle, FOREST_MODEL_FIELDS, forest_export_selection, forest_export_counts, forest_export_dtype,
    forest_export_verdict_check, forest_fit_notice_codes, trees_refuse, trees_check_features, forest_fit_core,
)
from bindings.forest_config_run import (
    ForestFeatures, ForestConfig, forest_features_fraction, forest_request, forest_target_need, forest_target_check,
    FOREST_FEATURES_SQRT, FOREST_FEATURES_LOG2, FOREST_FEATURES_COUNT, FOREST_FEATURES_FRACTION,
    FOREST_FEATURES_REFUSED, FOREST_TARGET_ANY, FOREST_TARGET_POISSON,
)
from bindings.forest_predict_run import (
    ForestState, FOREST_ROUTE_SEQUENTIAL, FOREST_ROUTE_STRICT, FOREST_ROUTE_RESIDENT, forest_labels_route,
    forest_shape_check, forest_snapshot_prepare, forest_vote, forest_labels, forest_wrote_check,
)
from bindings.forest_fit_boundary_unpack import forest_request_from_words_py
from bindings.gbdt_fit_boundary_unpack import gbdt_fit_param_py
from core.forest_fit_request import ForestFitArgs
from bindings.forest_fit_boundary_pack import forest_request_words_py
from core.forest_entry_table import ForestEntryKey, FOREST_TABLE_RF, FOREST_TABLE_ET
from core.trees_entry_table import TreesEntryKey
from core.forest_fit_request import (
    ForestFitArgs, XF_CAP_DEVICE_SCAN, XF_CAP_ORDERED_RESIDENT, XF_CAP_STRICT,
    XF_RESIDENT_RELEASE, XF_STRICT_RELEASE,
)
from core.entry_table import EntryRef
from core.forest_calls import (
    ForestExport, xf_ref, xf_caps, xf_dt_default_bins, xf_export_sizes, xf_export_into,
    xf_export_verify, xf_export_release, xf_snapshot_release,
)
from bindings.trees_keys_unpack import trees_keys

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


def _class_fields(cls: PythonObject) raises -> PythonObject:
    var b = trees_builtins()
    var nk = trees_keys()
    return py_call(py_item(b.type.__dict__, nk[].k___dict__).__get__, cls)


def _mro(cls: PythonObject) raises -> PythonObject:
    var b = trees_builtins()
    var nk = trees_keys()
    return py_call(py_item(b.type.__dict__, nk[].k___mro__).__get__, cls)


comptime _KIND_SLOT = "trees_forest_kind_v1"


def forest_kind(model: PythonObject, context: PythonObject) raises -> String:
    """The connected schema of the model's class: the nearest registered
    class in its ancestry ("rf_classifier", "dt_regressor", "et_classifier",
    ...)."""
    var cls = py_type(model)
    var nk = trees_keys()
    var table = py_item(context, nk[].k_trees_classes)
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


# TOMBSTONE: forest_check_fit_device (Extra Trees' fit-time `device must be
# 'gpu'` test) deleted 2026-11-20 by lane dw-trees: the estimator's device is
# admitted where it is given: the constructor and `set_params`
# (bindings/trees_route_boundary_door.mojo `trees_device_admit`).


def forest_get_params(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """`ForestProtocol.get_params()`: the stored constructor parameters, by
    name, with its two refusals."""
    var b = trees_builtins()
    var cls = py_type(model)
    var mro = _mro(cls)
    var nk = trees_keys()
    for i in range(len(mro)):  # driver-loop: over the class ancestry metadata
        var own = _class_fields(mro[i])
        if py_contains(own, nk[].k___init__):
            if not py_contains(own, nk[].k__parameter_names):
                raise Error("TypeError: Forest subclasses with a custom constructor must register "
                            + "their parameters before cloning or refitting")
            break
    var fields = py_fields(model)
    var has = py_get(fields, nk[].k__has_constructor_parameters)
    if has is None:
        has = py_get(_class_fields(cls), nk[].k__has_constructor_parameters, PythonObject(False))
    if not _truthy(has):
        raise Error("This inference-only forest archive has no constructor "
                    + "parameters; construct a new estimator for cloning/refitting")
    var names = cls._parameter_names
    var params = Python.dict()
    for i in range(len(names)):  # driver-loop: over the constructor parameter names
        var value = py_get(fields, names[i], py_item(context, nk[].k_trees_missing))
        if value is py_item(context, nk[].k_trees_missing):
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
        if exact == "identical":
            return mode
        if exact == "fast":
            var answered = mode
            answer_mode_object(answered, exact, MODE_FAMILY_FOREST)
            return answered
    if py_type(mode) is not b.str and not Bool(py=py_call(b.isinstance, mode, b.str)):
        raise Error("numeric_mode must be fast, identical or None")
    var clean = py_call(py_call(mode.strip).lower)
    var clean_word = String(clean)
    if clean_word != "fast" and clean_word != "identical":
        raise Error("numeric_mode must be fast, identical or None")
    answer_mode_object(clean, clean_word, MODE_FAMILY_FOREST)
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
    var nk = trees_keys()
    if py_item(params, nk[].k_max_leaf_nodes) is not None:
        _refuse("max_leaf_nodes", String(py_item(tables, nk[].k_max_leaf_nodes_why)))
    var oob = py_item(params, nk[].k_oob_score)
    if not _is_bool(oob):
        raise Error("oob_score must be a boolean; callable scoring is not supported")
    if _truthy(oob) and not _truthy(py_item(params, nk[].k_bootstrap)):
        raise Error("oob_score=True requires bootstrap=True")
    if _truthy(py_item(params, nk[].k_warm_start)):
        _refuse("warm_start=True", "there is no incremental fit here,"
                + " and accepting it would silently refit from scratch.")
    if _truthy(py_item(params, nk[].k_ccp_alpha)):
        _refuse("ccp_alpha", "cost-complexity pruning is a"
                + " post-processing pass neither cuML nor this implementation"
                + " implements. Only 0.0 is accepted.")
    if class_weight is not None and not Bool(py=py_call(b.isinstance, class_weight, b.dict)):
        if not Bool(py=py_call(b.isinstance, class_weight, b.str)) or not Bool(py=class_weight == PythonObject("balanced")):
            raise Error("class_weight must be None, a dict, or 'balanced'; balanced_subsample is unsupported")
    if py_item(params, nk[].k_monotonic_cst) is not None:
        _refuse("monotonic_cst", "no cuML counterpart.")
    if py_item(params, nk[].k_n_jobs) is not None:
        _refuse("n_jobs", "there is no CPU thread pool here -- the fit"
                + " is one host thread driving the GPU.")
    if _truthy(py_item(params, nk[].k_verbose)):
        _refuse("verbose", "nothing in the Mojo layer logs per-tree progress.")
    var max_samples = py_item(params, nk[].k_max_samples)
    if max_samples is not None:
        var f = Float64(py=_float(max_samples))
        if not (f > 0.0 and f <= 1.0):
            raise Error("max_samples=" + _str(max_samples) + " is outside (0, 1]")
    var cfg = Python.dict()
    py_setitem(cfg, nk[].k_criterion, _int(criterion_code))
    py_setitem(cfg, nk[].k_n_estimators, _int(py_item(params, nk[].k_n_estimators)))
    var depth = py_item(params, nk[].k_max_depth)
    py_setitem(cfg, nk[].k_max_depth, PythonObject(_INT32_MAX) if depth is None else _int(depth))
    py_setitem(cfg, nk[].k_max_leaves, _max_leaves_slot(py_item(params, nk[].k_max_leaves), context))
    py_setitem(cfg, nk[].k_max_features, py_item(params, nk[].k_max_features))
    py_setitem(cfg, nk[].k_n_bins, _int(py_item(params, nk[].k_n_bins)))
    py_setitem(cfg, nk[].k_min_samples_leaf, _int(py_item(params, nk[].k_min_samples_leaf)))
    py_setitem(cfg, nk[].k_min_samples_split, _int(py_item(params, nk[].k_min_samples_split)))
    py_setitem(cfg, nk[].k_min_impurity_decrease, _float(py_item(params, nk[].k_min_impurity_decrease)))
    py_setitem(cfg, nk[].k_bootstrap, PythonObject(_truthy(py_item(params, nk[].k_bootstrap))))
    py_setitem(cfg, nk[].k_max_samples, PythonObject(1.0) if max_samples is None else _float(max_samples))
    var seed = py_item(params, nk[].k_random_state)
    py_setitem(cfg, nk[].k_random_state, PythonObject(0) if seed is None else _int(seed))
    py_setitem(cfg, nk[].k_n_streams, _int(py_item(params, nk[].k_n_streams)))
    py_setitem(cfg, nk[].k_max_batch_size, _int(py_item(params, nk[].k_max_batch_size)))
    py_setitem(cfg, nk[].k_oob_score, PythonObject(_truthy(oob)))
    return cfg


def _rf_arguments(params: PythonObject, kind: String, context: PythonObject) raises -> PythonObject:
    """The base constructor's arguments: the estimator's own for a forest;
    for a DecisionTree*, `_dt_common` first, then its fixed single-tree
    arguments (one tree, no bootstrap, one stream) over the RF defaults."""
    if not kind.startswith("dt_"):
        return params
    var b = trees_builtins()
    var nk = trees_keys()
    var splitter = py_item(params, nk[].k_splitter)
    if not Bool(py=splitter == PythonObject("best")) and not Bool(py=splitter == PythonObject("random")):
        raise Error("splitter must be 'best' or 'random', got " + _repr(splitter))
    var best = Bool(py=splitter == PythonObject("best"))
    if best and py_item(params, nk[].k_max_leaf_nodes) is not None:
        _refuse("max_leaf_nodes with splitter='best'", "best-first growth exists on the random"
                + " splitter's builder (extratrees/, DEVIATIONS 466-469); pass splitter='random'.")
    var args = py_call(b.dict, params)
    py_setitem(args, nk[].k_n_estimators, PythonObject(1))
    py_setitem(args, nk[].k_max_leaf_nodes, PythonObject(None) if not best else py_item(params, nk[].k_max_leaf_nodes))
    py_setitem(args, nk[].k_bootstrap, PythonObject(False))
    py_setitem(args, nk[].k_n_streams, PythonObject(1))
    py_setitem(args, nk[].k_oob_score, PythonObject(False))
    py_setitem(args, nk[].k_n_jobs, PythonObject(None))
    py_setitem(args, nk[].k_verbose, PythonObject(0))
    py_setitem(args, nk[].k_warm_start, PythonObject(False))
    py_setitem(args, nk[].k_max_samples, PythonObject(None))
    py_setitem(args, nk[].k_max_leaves, PythonObject(-1))
    py_setitem(args, nk[].k_max_batch_size, PythonObject(4096))
    if kind == "dt_regressor":
        py_setitem(args, nk[].k_class_weight, PythonObject(None))
        var bins = py_item(params, nk[].k_n_bins)
        py_setitem(args, nk[].k_n_bins, PythonObject(128) if bins is None else bins)
    return args


def _rf_derive(model: PythonObject, params: PythonObject, kind: String, context: PythonObject) raises -> PythonObject:
    """The state the RandomForest*/DecisionTree* constructor derived (in
    its order): `device` and `_cfg`."""
    var nk = trees_keys()
    var tables = py_item(context, nk[].k_trees_rf_tables)
    var args = _rf_arguments(params, kind, context)
    var who = String(py_type(model).__name__)
    var classifier = _classifier(kind)
    var code = _criterion_code(py_item(args, nk[].k_criterion), tables["cls_criteria" if classifier else "reg_criteria"],
                               py_item(tables, nk[].k_unsupported), "classifier" if classifier else "regressor")
    var mwf = _min_weight_fraction(py_item(args, nk[].k_min_weight_fraction_leaf), who)
    var cfg = _rf_base(args, code, _param(args, "class_weight", PythonObject(None)) if classifier else PythonObject(None),
                       tables, context)
    py_setitem(cfg, nk[].k_min_weight_fraction_leaf, mwf)
    var derived = Python.dict()
    py_setitem(derived, nk[].k_device, py_item(args, nk[].k_device))
    py_setitem(derived, nk[].k__cfg, cfg)
    return derived


def _et_derive(model: PythonObject, params: PythonObject, kind: String, context: PythonObject) raises -> PythonObject:
    """`ExtraTrees*.__init__`: `_validate_device`, `_refuse_forest_knobs`,
    the criterion, then `device`, `_criterion_code` and `_cfg` (the raw
    parameters; `_fit_params` converts them at fit)."""
    var b = trees_builtins()
    var nk = trees_keys()
    var tables = py_item(context, nk[].k_trees_et_tables)
    var device = py_item(params, nk[].k_device)
    if py_item(params, nk[].k_n_jobs) is not None:
        raise Error("NotImplementedError: n_jobs is not implemented: there is no CPU thread pool here -- the"
                    + " fit runs one host thread driving the GPU (device='gpu')."
                    + " Refused by name rather than"
                    + " accepted and ignored.")
    if _truthy(py_item(params, nk[].k_verbose)):
        raise Error("NotImplementedError: verbose is not implemented: nothing in the Mojo layer logs per-tree"
                    + " progress. Refused by name rather than accepted and ignored.")
    var classifier = _classifier(kind)
    var criterion = py_item(params, nk[].k_criterion)
    var table = tables["cls_criteria" if classifier else "reg_criteria"]
    if not py_contains(table, criterion):
        var reason = py_get(py_item(tables, nk[].k_unsupported), criterion)
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
        var mf = py_item(params, nk[].k_max_features)
        if Bool(py=mf == PythonObject(1.0)) and not Bool(py=py_call(b.isinstance, mf, b.int)):
            py_setitem(cfg, nk[].k_max_features, PythonObject(None))
    var derived = Python.dict()
    py_setitem(derived, nk[].k_device, device)
    py_setitem(derived, nk[].k__criterion_code, table[criterion])
    py_setitem(derived, nk[].k__cfg, cfg)
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
    var nk = trees_keys()
    py_setitem(fields, nk[].k__has_constructor_parameters, PythonObject(True))
    _ = py_call(fields.update, derived)
    _ = py_call(fields.update, keep)
    return fields


def forest_validate_stored(model: PythonObject, context: PythonObject) raises:
    """`ForestProtocol._validate_params()` (lane py-trees, 2026-11-14: the
    Python method is one call here): the constructor checks on the stored
    parameters, in the constructor's order (numeric_mode, inference_engine,
    then `_et_derive` / `_rf_derive`), and the derived state (`device`,
    `_cfg`, ET's `_criterion_code`) stored; the parameters stay as given.
    A parameter missing from the instance raises getattr's AttributeError."""
    var kind = forest_kind(model, context)
    var cls = py_type(model)
    var names = cls._parameter_names
    var fields = py_fields(model)
    var nk = trees_keys()
    var missing = py_item(context, nk[].k_trees_missing)
    var params = Python.dict()
    for i in range(len(names)):  # driver-loop: over the constructor parameter names
        var value = py_get(fields, names[i], missing)
        if value is missing:
            raise Error("AttributeError: '" + String(cls.__name__) + "' object has no attribute '" + String(names[i]) + "'")
        py_setitem(params, names[i], value)
    _ = _check_mode(_param(params, "numeric_mode", PythonObject(None)))
    _ = _check_engine(_param(params, "inference_engine", PythonObject("sequential")))
    # `device`: None, 'gpu' or 'cpu' (the shared rule and words). Checked by
    # this explicit validation, the constructor and `set_params`; a fit reads
    # no device (lane dw-trees, 2026-11-21: the per-fit check in `_rf_base`,
    # `_et_derive` and the ET fit params is gone).
    _ = device_unpack(py_item(params, nk[].k_device), trees_builtins())
    var derived = _et_derive(model, params, kind, context) if _family(kind) == "et" else _rf_derive(model, params, kind, context)
    _ = py_call(fields.update, derived)


def forest_prediction_engine(fields: PythonObject, context: PythonObject) raises -> String:
    """`ForestProtocol._prediction_engine()`: the engine checked first, then
    (for `auto` only) the effective mode, in the Python method's order."""
    var nk = trees_keys()
    var engine = _check_engine(py_get(fields, nk[].k_inference_engine, PythonObject("sequential")))
    if engine == "auto":
        return "parallel_groves" if forest_effective_mode(fields, context) == "fast" else "sequential"
    return engine


def forest_ordered_resident_auto(model: PythonObject, fields: PythonObject, context: PythonObject) raises -> Bool:
    """`ForestProtocol._ordered_resident_auto()`: the sequential engine on an
    image whose resident snapshot takes the ordered word
    (XF_CAP_ORDERED_RESIDENT, the former `forest_ordered_resident` export).
    A binding that cannot be connected, or publishes no native entry, has no
    such route (the Python method's ImportError/AttributeError absence)."""
    if forest_prediction_engine(fields, context) != "sequential":
        return False
    var kind = forest_kind(model, context)
    var mode = forest_effective_mode(fields, context)
    try:
        return _ordered_export(trees_table(context, mode, "_mojolearn_trees" if _family(kind) == "et" else "_mojolearn_rf"))
    except error:
        var message = String(error)
        if message.startswith("ImportError") or message.startswith("AttributeError"):
            return False
        raise error^


def forest_capture_mode(fields: PythonObject, context: PythonObject) raises -> String:
    """`ForestProtocol._capture_fit_mode()`: the requested mode, or the
    process default, stored as `_fit_numeric_mode`."""
    var nk = trees_keys()
    var requested = _check_mode(py_get(fields, nk[].k_numeric_mode))
    var mode = requested if requested is not None else py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
    var clean = _check_mode(mode)
    py_setitem(fields, nk[].k__fit_numeric_mode, clean)
    return String(clean)


def forest_effective_mode(fields: PythonObject, context: PythonObject) raises -> String:
    """`ForestProtocol._effective_mode()`."""
    var nk = trees_keys()
    var requested = _check_mode(py_get(fields, nk[].k_numeric_mode))
    var captured = py_get(fields, nk[].k__fit_numeric_mode)
    if captured is not None:
        if requested is not None and not Bool(py=requested == captured):
            raise Error("This forest was fitted with numeric_mode=" + _repr(captured) + "; "
                        + "refit or use set_params before changing numeric_mode")
        return String(captured)
    if requested is not None:
        return String(requested)
    return String(_check_mode(py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)))


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


# TOMBSTONE: forest_fit_request / _launch (the typed request built from a member's Python fields per fit, and
# the launch through a binding object, for the boost door's AdaBoost member rounds, their last callers) deleted
# 2026-11-23 by lane mf-ada: xtrees/trees_ada_fit_run.mojo builds every round's request from the ForestConfig
# the door reads once per fit. Not an experiment arm: git revert of that commit restores them.


def _launch_on(key: EntryRef, fit: ForestFit, x: AdmittedXView, args: ForestFitArgs, device_scan: Bool,
               context: PythonObject) raises -> PythonObject:
    """One fit launch and its export, through the image's native entry `key`
    (resolved once per fit): the typed fit core (`forest_fit_core`, XF_FIT)
    on the request `forest_request` built (RUN), then the owned export
    copied into the model Arrays (PACK)."""
    var selection = forest_export_selection()
    var handle = forest_fit_core(key, fit.classifier, fit.rowmajor_ok and x.c_order, args, device_scan)
    return forest_export_handle(key, handle, selection, context)


# `forest_fit_dart_member` (the DART member's XF_FIT_DART fit and export) moved
# to bindings/trees_dart_rounds_run.mojo `_member_fit` (lane py-trees, 2026-11-14).


# ------------------------------------------------------------- fit params


# TOMBSTONE: forest_export (the export of a fit a Python entry returned as a descriptor) / _rf_params / _max_features_fraction (the 16-slot Python list of an RF fit and its fraction read; the boost door's session and AdaBoost member fits were their last callers and now build the typed request, `forest_fit_request`, and call XF_FIT_SESSION / XF_FIT_ADA_ROUND) deleted 2026-11-21 by lane dw-trees-boost; code recoverable at cc2c81eb61.
# Restore: git apply experiments/removed/dead_entries_rf_session_ada_calls.patch


def _dt_tier_bins(fields: PythonObject, rf: PythonObject) raises -> PythonObject:
    """`_expansion_trees._dt_tier_bins`: the tree's own n_bins, or for
    n_bins=None its binding's `rf_dt_default_bins` (128 without one)."""
    var nk = trees_keys()
    var bins = py_get(fields, nk[].k_n_bins)
    if bins is not None:
        return _int(bins)
    # XF_DT_DEFAULT_BINS: the image's bin count (128 in one without its own)
    return PythonObject(xf_dt_default_bins(forest_key(rf)))


def _note(first: String, text: String) -> String:
    """The first refusal of a fit's parameter conversions."""
    if first != "":
        return first
    return text


def _word(value: Int, mut ints: List[Int], mut floats: List[Float64], mut bad: List[UInt8]):
    """A slot whose value the door holds as an integer (both readings)."""
    ints.append(value)
    floats.append(Float64(value))
    bad.append(UInt8(0))


def _slot_int(value: PythonObject, mut ints: List[Int], mut floats: List[Float64],
              mut bad: List[UInt8]) raises -> String:
    """`int(value)` as a slot. A refusal is returned as its text (the slot
    stays empty) for RUN to raise where the fit reached it."""
    try:
        gbdt_fit_param_py(_int(value), ints, floats, bad)
    except error:
        _word(0, ints, floats, bad)
        return String(error)
    return String("")


def _slot_float(value: PythonObject, mut ints: List[Int], mut floats: List[Float64],
                mut bad: List[UInt8]) raises -> String:
    """`float(value)` as a slot; a refusal as `_slot_int` returns it."""
    try:
        gbdt_fit_param_py(_float(value), ints, floats, bad)
    except error:
        _word(0, ints, floats, bad)
        return String(error)
    return String("")


def _rf_features(max_features: PythonObject, context: PythonObject) raises -> ForestFeatures:
    """UNPACK of RandomForest's `max_features`: its form as plain values. A
    form `randomforest._max_features_fraction` refuses is kept with its
    words; `forest_features_fraction` raises them (and the count's range
    against the fitted width) where the fit reached them."""
    var b = trees_builtins()
    var given = ForestFeatures()
    if max_features is None:
        return given^
    if Bool(py=py_call(b.isinstance, max_features, b.str)):
        var word = String(max_features)
        if word == "sqrt":
            given.form = FOREST_FEATURES_SQRT
        elif word == "log2":
            given.form = FOREST_FEATURES_LOG2
        else:
            given.form = FOREST_FEATURES_REFUSED
            given.refusal = ("max_features=" + _repr(max_features) + " is not a recognised form;"
                             + " accepted are 'sqrt', 'log2', None, a float fraction or an"
                             + " int count")
        return given^
    try:
        if _is_integral(max_features, context) and not _is_bool(max_features):
            given.count = Int(py=_int(max_features))
            given.form = FOREST_FEATURES_COUNT
        else:
            var f = Float64(py=_float(max_features))
            if f > 0.0 and f <= 1.0:
                given.fraction = f
                given.form = FOREST_FEATURES_FRACTION
            else:
                given.refusal = "max_features=" + _str(PythonObject(f)) + " is outside (0, 1]"
                given.form = FOREST_FEATURES_REFUSED
    except error:
        given.refusal = String(error)
        given.form = FOREST_FEATURES_REFUSED
    return given^


def _target_codes(mut config: ForestConfig, tables: PythonObject) raises:
    """The deviance criteria's codes the regression target check compares
    (the gamma and inverse Gaussian codes only when the criterion is not
    Poisson's, as the check read them)."""
    var nk = trees_keys()
    config.poisson = Int(py=py_item(tables, nk[].k_poisson))
    if config.target_code != config.poisson:
        config.gamma = Int(py=py_item(tables, nk[].k_gamma))
        config.inverse_gaussian = Int(py=py_item(tables, nk[].k_inverse_gaussian))


def _rf_config(fields: PythonObject, cfg: PythonObject, kind: String, targets: Bool,
               context: PythonObject) raises -> ForestConfig:
    """`_RandomForestBase._fit_params` as plain values: the 16 slots in
    _mojolearn_rf.mojo's order (then min_weight_fraction_leaf when it is not
    0), read from the configuration once. The rows, columns, classes and the
    feature fraction are the fit's (`forest_request`); DecisionTreeRegressor's
    slot 7 is its own n_bins or its image's bin count."""
    var nk = trees_keys()
    var ints = List[Int](capacity=24)
    var floats = List[Float64](capacity=24)
    var bad = List[UInt8](capacity=24)
    var refusal = String("")
    var bins_image = False
    _word(0, ints, floats, bad)
    _word(0, ints, floats, bad)
    _word(0, ints, floats, bad)
    gbdt_fit_param_py(py_item(cfg, nk[].k_n_estimators), ints, floats, bad)
    gbdt_fit_param_py(py_item(cfg, nk[].k_max_depth), ints, floats, bad)
    gbdt_fit_param_py(py_item(cfg, nk[].k_max_leaves), ints, floats, bad)
    _word(0, ints, floats, bad)
    if kind == "dt_regressor":
        # `_expansion_trees._dt_tier_bins`: the tree's own n_bins, or for
        # n_bins=None its image's default.
        var bins = py_get(fields, nk[].k_n_bins)
        if bins is not None:
            refusal = _slot_int(bins, ints, floats, bad)
        else:
            bins_image = True
            _word(0, ints, floats, bad)
    else:
        gbdt_fit_param_py(py_item(cfg, nk[].k_n_bins), ints, floats, bad)
    gbdt_fit_param_py(py_item(cfg, nk[].k_min_samples_leaf), ints, floats, bad)
    gbdt_fit_param_py(py_item(cfg, nk[].k_min_samples_split), ints, floats, bad)
    gbdt_fit_param_py(py_item(cfg, nk[].k_min_impurity_decrease), ints, floats, bad)
    _word(1 if _truthy(py_item(cfg, nk[].k_bootstrap)) else 0, ints, floats, bad)
    gbdt_fit_param_py(py_item(cfg, nk[].k_max_samples), ints, floats, bad)
    gbdt_fit_param_py(py_item(cfg, nk[].k_random_state), ints, floats, bad)
    gbdt_fit_param_py(py_item(cfg, nk[].k_n_streams), ints, floats, bad)
    gbdt_fit_param_py(py_item(cfg, nk[].k_max_batch_size), ints, floats, bad)
    var mwf = py_get(cfg, nk[].k_min_weight_fraction_leaf, PythonObject(0.0))
    if _truthy(mwf):
        gbdt_fit_param_py(mwf, ints, floats, bad)
    var config = ForestConfig(False, ints^, floats^, bad^)
    config.refusal = refusal
    config.bins_image = bins_image
    config.features = _rf_features(py_item(cfg, nk[].k_max_features), context)
    var criterion = py_item(cfg, nk[].k_criterion)
    if criterion is not None:
        config.criterion = Int(py=criterion)
    var oob = py_get(cfg, nk[].k_oob_score)
    if oob is not None:
        config.oob_score = _truthy(oob)
    if targets:
        config.target_code = config.criterion
        _target_codes(config, py_item(context, nk[].k_trees_rf_tables))
    return config^


def _et_config(fields: PythonObject, cfg: PythonObject, targets: Bool,
               context: PythonObject) raises -> ForestConfig:
    """`extratrees._fit_params` as plain values: the 22 slots in
    _mojolearn_trees.mojo's order, read from the configuration once, in the
    order the Python helper converted them (max_features first). A
    conversion Python refuses is kept as its text: `forest_request` raises
    the first one where the fit reached it. The bootstrap row count of a
    float `max_samples` is the fit's (`forest_request`)."""
    var nk = trees_keys()
    var refusal = String("")
    var late = String("")
    var f_ints = List[Int](capacity=2)
    var f_floats = List[Float64](capacity=2)
    var f_bad = List[UInt8](capacity=2)
    try:
        var slots = _et_max_features(py_item(cfg, nk[].k_max_features), context)
        gbdt_fit_param_py(slots[0], f_ints, f_floats, f_bad)
        gbdt_fit_param_py(slots[1], f_ints, f_floats, f_bad)
    except error:
        refusal = String(error)
    while len(f_ints) < 2:  # small-loop(the two max_features slots of a refused form)
        _word(0, f_ints, f_floats, f_bad)
    var ints = List[Int](capacity=24)
    var floats = List[Float64](capacity=24)
    var bad = List[UInt8](capacity=24)
    _word(0, ints, floats, bad)
    _word(0, ints, floats, bad)
    _word(0, ints, floats, bad)
    refusal = _note(refusal, _slot_int(py_item(cfg, nk[].k_n_estimators), ints, floats, bad))
    var depth = py_item(cfg, nk[].k_max_depth)
    if depth is None:
        _word(-1, ints, floats, bad)
    else:
        refusal = _note(refusal, _slot_int(depth, ints, floats, bad))
    refusal = _note(refusal, _slot_int(py_item(cfg, nk[].k_min_samples_split), ints, floats, bad))
    refusal = _note(refusal, _slot_int(py_item(cfg, nk[].k_min_samples_leaf), ints, floats, bad))
    refusal = _note(refusal, _slot_float(py_item(cfg, nk[].k_min_weight_fraction_leaf), ints, floats, bad))
    for k in range(2):  # small-loop(the two max_features slots)
        ints.append(f_ints[k])
        floats.append(f_floats[k])
        bad.append(f_bad[k])
    refusal = _note(refusal, _slot_float(py_item(cfg, nk[].k_min_impurity_decrease), ints, floats, bad))
    _word(1 if _truthy(py_item(cfg, nk[].k_bootstrap)) else 0, ints, floats, bad)
    _word(1 if _truthy(py_item(cfg, nk[].k_oob_score)) else 0, ints, floats, bad)
    var seed = py_item(cfg, nk[].k_random_state)
    if seed is None:
        _word(0, ints, floats, bad)
    else:
        refusal = _note(refusal, _slot_int(seed, ints, floats, bad))
    _word(1 if _truthy(py_item(cfg, nk[].k_warm_start)) else 0, ints, floats, bad)
    refusal = _note(refusal, _slot_float(py_item(cfg, nk[].k_ccp_alpha), ints, floats, bad))
    var class_weighted = py_item(cfg, nk[].k_class_weight) is not None
    _word(1 if class_weighted else 0, ints, floats, bad)
    _word(0 if py_item(cfg, nk[].k_monotonic_cst) is None else 1, ints, floats, bad)
    # `extratrees._n_samples_bootstrap`: None, an int >= 1, or a float
    # fraction of the fit's rows.
    var rows_fraction = 0.0
    var max_samples = py_item(cfg, nk[].k_max_samples)
    if max_samples is None:
        _word(0, ints, floats, bad)
    elif _is_bool(max_samples):
        _word(0, ints, floats, bad)
        refusal = _note(refusal, String("max_samples must be None, an int >= 1 or a float"))
    elif _is_integral(max_samples, context):
        if Bool(py=max_samples < PythonObject(1)):
            _word(0, ints, floats, bad)
            refusal = _note(refusal, "max_samples=" + _str(max_samples) + " must be >= 1 when an int")
        else:
            refusal = _note(refusal, _slot_int(max_samples, ints, floats, bad))
    else:
        _word(0, ints, floats, bad)
        try:
            var f = Float64(py=_float(max_samples))
            if f > 0.0:
                rows_fraction = f
            else:
                refusal = _note(refusal, "max_samples=" + _str(max_samples) + " must be > 0 when a float")
        except error:
            refusal = _note(refusal, String(error))
    var leaves = py_item(cfg, nk[].k_max_leaf_nodes)
    if leaves is None:
        _word(-1, ints, floats, bad)
    else:
        late = _slot_int(leaves, ints, floats, bad)
    _word(1, ints, floats, bad)
    late = _note(late, _slot_int(py_item(fields, nk[].k__criterion_code), ints, floats, bad))
    var config = ForestConfig(True, ints^, floats^, bad^)
    config.refusal = refusal
    config.refusal_late = late
    config.rows_fraction = rows_fraction
    config.class_weighted = class_weighted
    if targets:
        config.target_code = Int(py=py_item(fields, nk[].k__criterion_code))
        _target_codes(config, py_item(context, nk[].k_trees_et_tables))
    return config^


def forest_config(fields: PythonObject, cfg: PythonObject, kind: String, targets: Bool,
                  context: PythonObject) raises -> ForestConfig:
    """UNPACK, once per fit: the derived configuration (`cfg`: the instance's
    `_cfg`; for Extra Trees also its `_criterion_code`) as plain values
    (`ForestConfig`, bindings/forest_config_run.mojo). Everything a fit
    decides after this reads the struct; `forest_request` builds the request
    from it. `targets` also reads the criterion codes a regressor's target
    check compares.

    `_cfg` itself stays in the instance dictionary: the boost door reads it
    by name (its member and session fits)."""
    if _family(kind) == "et":
        return _et_config(fields, cfg, targets, context)
    return _rf_config(fields, cfg, kind, targets, context)


def _target_check(y: AdmittedXView, config: ForestConfig, context: PythonObject) raises:
    """A regressor's target against its criterion: UNPACK reads what the
    deviance criteria test of y (finite; its smallest and, for Poisson, its
    largest value, each only when the one before passed), and
    `forest_target_check` refuses in the RF or ET words."""
    var need = forest_target_need(config)
    if need == FOREST_TARGET_ANY:
        return
    var n = y.rows
    var finite = trees_finite_f32(y.address, n)
    var lowest = 0.0
    var highest = 1.0
    if finite:
        lowest = Float64(py=_reduce(y.address, n, 0, context))
        if need == FOREST_TARGET_POISSON and not lowest < 0.0:
            highest = Float64(py=_reduce(y.address, n, 1, context))
    forest_target_check(config, finite, lowest, highest)


def _et_max_features(max_features: PythonObject, context: PythonObject) raises -> PythonObject:
    """`extratrees._max_features_slots`: (spec, fraction)."""
    var b = trees_builtins()
    var nk = trees_keys()
    var tables = py_item(context, nk[].k_trees_et_tables)
    if max_features is None:
        return Python.tuple(py_item(tables, nk[].k_mf_all), PythonObject(0.0))
    if Bool(py=py_call(b.isinstance, max_features, b.str)):
        if String(max_features) == "sqrt":
            return Python.tuple(py_item(tables, nk[].k_mf_sqrt), PythonObject(0.0))
        if String(max_features) == "log2":
            return Python.tuple(py_item(tables, nk[].k_mf_log2), PythonObject(0.0))
        raise Error("max_features=" + _repr(max_features) + " is not a recognised form; sklearn"
                    + " accepts 'sqrt', 'log2', None, a float fraction or an int count")
    if _is_integral(max_features, context) and not _is_bool(max_features):
        return Python.tuple(_int(max_features), PythonObject(0.0))
    return Python.tuple(PythonObject(0), _float(max_features))


# TOMBSTONE: _et_params (the 22-slot Python list of an Extra Trees fit, built
# and read back per fit) deleted 2026-11-21 by lane dw-trees. (`_rf_params`
# followed it the same day, lane dw-trees-boost: tombstone above `_dt_tier_bins`.)
# TOMBSTONE: _rf_slots / _et_slots / _et_bootstrap_rows / _check_regression_target /
# _device_scan (the per-fit reads of the `_cfg` dictionary, each scalar boxed
# as a Python object, and the Python comparisons of the target check) deleted
# 2026-11-21 by lane dw-trees: `forest_config` reads the configuration once
# into `ForestConfig` and bindings/forest_config_run.mojo builds the request
# and checks the target on plain values.


def _store_model(fields: PythonObject, arrays: PythonObject, cols: Int) raises:
    """The fitted model arrays, frozen (the copy `_mojolearn_after_fit`
    took), an empty resident holder, and the dimensions."""
    var names = Python.tuple("_offsets", "_colid", "_quesval", "_left_child", "_leaves")
    for i in range(_MODEL_FIELDS):  # driver-loop: over the five model arrays
        trees_freeze(arrays[i])
        py_setitem(fields, names[i], arrays[i])
    var meta = arrays[5]
    var nk = trees_keys()
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(cols))
    py_setitem(fields, nk[].k__n_trees, _int(meta[0]))
    py_setitem(fields, nk[].k__num_outputs, _int(meta[1]))
    py_setitem(fields, nk[].k__resident_holder, Python.dict())


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


def _rf_fit_arrays(fields: PythonObject, config: ForestConfig, x: AdmittedX, y_addr: Int, y_rows: Int,
                   classes: Int, fit: ForestFit, rf: PythonObject,
                   context: PythonObject) raises -> PythonObject:
    """`_RandomForestBase._fit_arrays` on X already in the entry's layout
    (`_x_for`): the length and finiteness checks, the request, the launch
    and the model export. Returns the fit result tuple."""
    _y_rows_check(y_rows, x.rows)
    var key = forest_key(rf)
    var device_scan = (xf_caps(key) & XF_CAP_DEVICE_SCAN) != 0
    _finite_or_refuse(x, device_scan, context)
    # `fit_loop_` (lane alm-forest): the image writes the tree loop's two
    # words here (XF_W_LOOP); -1 until it does. On this frame, no heap.
    var loop_words: Array[Int, 2] = [-1, -1]
    var timed = fit.copy()
    timed.loop = Int(loop_words.unsafe_ptr())
    var request = forest_request(config, key, x.rows, x.cols, classes, x.address, y_addr, timed, False)
    var result = _launch_on(key, timed, x.view(), request, device_scan, context)
    _rf_fit_notices(result[5], context)
    _store_model(fields, result, x.cols)
    forest_fit_loop_store(fields, loop_words[0], loop_words[1])
    _ = loop_words  # written through its address by the fit above
    return result


def _rf_fit_notices(meta: PythonObject, context: PythonObject) raises:
    """The fit's notices (the third metadata word, an
    `ensemble/rf_notices.mojo` bit mask) as Python UserWarnings, in the
    order the fit reached them, attributed to the caller of `fit`. A
    binding without the word has none to report."""
    if len(meta) < 3:
        return
    rf_fit_notices_word(Int(py=meta[2]), context)


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
    var nk = trees_keys()
    var config = forest_config(fields, py_item(fields, nk[].k__cfg), kind, not _classifier(kind), context)
    if _classifier(kind):
        return _rf_fit_classifier(model, fields, config, X, y, sample_weight, importances, tree_start, mode, rf, context)
    return _rf_fit_regressor(model, fields, config, X, y, sample_weight, importances, tree_start, mode, rf, context)


def _rf_fit_classifier(model: PythonObject, fields: PythonObject, config: ForestConfig, X: PythonObject, y: PythonObject,
                       sample_weight: PythonObject, importances: Bool, tree_start: PythonObject, mode: String,
                       rf: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = trees_builtins()
    var labels = trees_encode_labels(y, labels_scope(model, context))
    var nk = trees_keys()
    py_setitem(fields, nk[].k_classes_, labels.classes)
    var k = len(labels.classes)
    py_setitem(fields, nk[].k_n_classes_, PythonObject(k))
    if k < 2:
        raise Error("y has fewer than 2 classes")
    var xtrees = trees_table(context, mode, "_mojolearn_x_trees")
    var class_weight = py_get(fields, nk[].k_class_weight)
    var weights: Optional[AdmittedX] = None
    if sample_weight is not None:
        weights = trees_weighted_rows(sample_weight, class_weight, labels, xtrees, context)
    elif class_weight is not None:
        weights = trees_class_rows_only(class_weight, labels, xtrees, context)
    var names = Python.tuple("oob_decision_function_", "oob_score_", "feature_importances_")
    for i in range(3):  # driver-loop: over fixed call structure, small-loop(names): the three requested-output attribute names
        _ = py_call(fields.pop, names[i], PythonObject(None))
    var y_addr = trees_address(labels.codes)
    if config.oob_score or importances:
        if tree_start is not None:
            raise Error("auxiliary outputs require a complete forest, not a tree-ID shard")
        # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
        var oob = Optional[PythonObject]()
        # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
        var score = Optional[PythonObject]()
        if config.oob_score:
            oob = admitted_array(Python.tuple(labels.n, k), "<f8", context)
            score = admitted_vector(1, "<f8", context)
        var x = _x_for(X, False, context)
        var importance = admitted_vector(x.cols, "<f4", context) if importances else PythonObject(None)
        var fit = ForestFit(True, weights=0 if not weights else weights.value().address,
                            oob=0 if not oob else trees_address(oob.value()),
                            score=0 if not score else trees_address(score.value()),
                            importance=0 if importance is None else trees_address(importance))
        _ = _rf_fit_arrays(fields, config, x, y_addr, labels.n, k, fit, rf, context)
        _ = labels  # y_addr borrows the encoded codes, not the stored classes_
        _ = importance  # keep the Python object alive through the native call above
        _ = score  # keep the Python object alive through the native call above
        _ = oob  # keep the Python object alive through the native call above
        if Bool(oob):
            py_setitem(fields, nk[].k_oob_decision_function_, oob.value())
            py_setitem(fields, nk[].k_oob_score_, PythonObject(_F64(score.value(), 0)))
        if importance is not None:
            py_setitem(fields, nk[].k_feature_importances_, importance)
        _ = weights^
        return model
    if tree_start is not None:
        var fit = ForestFit(True, weights=0 if not weights else weights.value().address,
                            tree_start=Int(py=tree_start))
        _ = _rf_fit_arrays(fields, config, _x_for(X, False, context), y_addr, labels.n, k, fit, rf, context)
        _ = labels  # retain shard labels through the address-only fit
        _ = weights^
        return model
    if weights:
        _ = _rf_fit_arrays(fields, config, _x_for(X, False, context), y_addr, labels.n, k,
                           ForestFit(True, weights=weights.value().address), rf, context)
        _ = labels
        _ = weights^
        return model
    _ = _rf_fit_arrays(fields, config, _x_for(X, True, context), y_addr, labels.n, k,
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


def _rf_fit_regressor(model: PythonObject, fields: PythonObject, config: ForestConfig, X: PythonObject, y: PythonObject,
                      sample_weight: PythonObject, importances: Bool, tree_start: PythonObject, mode: String,
                      rf: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = trees_keys()
    var y32 = admit_f32(y, 1, "y", context, finite=False)
    _target_check(y32.view(), config, context)
    var names = Python.tuple("oob_prediction_", "oob_score_", "feature_importances_")
    for i in range(3):  # driver-loop: over fixed call structure, small-loop(names): the three requested-output attribute names
        _ = py_call(fields.pop, names[i], PythonObject(None))
    if sample_weight is not None:
        if tree_start is not None or config.oob_score:
            raise Error("NotImplementedError: sample_weight is not implemented together with oob_score "
                        + "or a tree-ID shard")
        var w = admit_weight_words(sample_weight, y32.rows, context)
        var w_max = Float64(py=_float(_reduce(w.address, y32.rows, 1, context)))
        var xw = _x_for(X, False, context)
        # Lane mf-rfe6: the weighted fit with the importance fit, as the classifier's (the same fit, then the
        # same importance body over the fitted trees).
        var weighted_importance = admitted_vector(xw.cols, "<f4", context) if importances else PythonObject(None)
        _ = _rf_fit_arrays(fields, config, xw, y32.address, y32.rows, 0,
                           ForestFit(False, weights=w.address, weight_max=w_max,
                                     importance=0 if weighted_importance is None else trees_address(weighted_importance)),
                           rf, context)
        _ = y32  # keep the Python object alive through the native call above
        _ = w  # keep the Python object alive through the native call above
        _ = weighted_importance  # written through the native call above
        if weighted_importance is not None:
            py_setitem(fields, nk[].k_feature_importances_, weighted_importance)
        _ = w^
        _ = y32^
        return model
    if config.oob_score or importances:
        if tree_start is not None:
            raise Error("auxiliary outputs require a complete forest, not a tree-ID shard")
        # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
        var oob = Optional[PythonObject]()
        # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
        var score = Optional[PythonObject]()
        if config.oob_score:
            oob = admitted_vector(y32.rows, "<f8", context)
            score = admitted_vector(1, "<f8", context)
        var x = _x_for(X, False, context)
        var importance = admitted_vector(x.cols, "<f4", context) if importances else PythonObject(None)
        var fit = ForestFit(False, oob=0 if not oob else trees_address(oob.value()),
                            score=0 if not score else trees_address(score.value()),
                            importance=0 if importance is None else trees_address(importance))
        _ = _rf_fit_arrays(fields, config, x, y32.address, y32.rows, 0, fit, rf, context)
        _ = importance  # keep the Python object alive through the native call above
        _ = score  # keep the Python object alive through the native call above
        _ = oob  # keep the Python object alive through the native call above
        if Bool(oob):
            py_setitem(fields, nk[].k_oob_prediction_, oob.value())
            py_setitem(fields, nk[].k_oob_score_, PythonObject(_F64(score.value(), 0)))
        if importance is not None:
            py_setitem(fields, nk[].k_feature_importances_, importance)
        _ = y32^
        return model
    if tree_start is not None:
        _ = _rf_fit_arrays(fields, config, _x_for(X, False, context), y32.address, y32.rows, 0,
                           ForestFit(False, tree_start=Int(py=tree_start)), rf, context)
        _ = y32^
        return model
    _ = _rf_fit_arrays(fields, config, _x_for(X, True, context), y32.address, y32.rows, 0,
                       ForestFit(False, rowmajor_ok=True), rf, context)
    _ = y32^
    return model


def _et_fit_arrays(fields: PythonObject, config: ForestConfig, unweighted: Bool, X: PythonObject, y_addr: Int,
                   y_rows: Int, classes: Int, fit: ForestFit, et: PythonObject,
                   context: PythonObject) raises:
    """`_ExtraTreesBase._fit_arrays`: X in the entry's layout, the length
    and finiteness checks, the request (`unweighted`: a fit whose class
    weights travel as row weights), the launch, the export and the fitted
    state with its four extra-trees fields."""
    var x = _x_for(X, fit.rowmajor_ok, context)
    _y_rows_check(y_rows, x.rows)
    var key = forest_key(et)
    var device_scan = (xf_caps(key) & XF_CAP_DEVICE_SCAN) != 0
    _finite_or_refuse(x, device_scan, context)
    # `fit_loop_` (lane alm-forest), as `_rf_fit_arrays`.
    var loop_words: Array[Int, 2] = [-1, -1]
    var timed = fit.copy()
    timed.loop = Int(loop_words.unsafe_ptr())
    var request = forest_request(config, key, x.rows, x.cols, classes, x.address, y_addr, timed, unweighted)
    var result = _launch_on(key, timed, x.view(), request, device_scan, context)
    forest_et_store_fit(fields, result, x.cols)
    forest_fit_loop_store(fields, loop_words[0], loop_words[1])
    _ = loop_words  # written through its address by the fit above


def forest_et_store_fit(fields: PythonObject, result: PythonObject, cols: Int) raises:
    """PACK of one extra-trees fit: the exported model (`result`, as
    `forest_export_handle` returns it) and the four extra-trees fields of
    its metadata. The door's ExtraTrees fits and RandomTreesEmbedding's
    forest (bindings/trees_public_boost_door.mojo `_rte_fit`) store through
    this."""
    _store_model(fields, result, cols)
    var meta = result[5]
    var nk = trees_keys()
    py_setitem(fields, nk[].k_depth_cap_bound_, PythonObject(_truthy(meta[2])))
    py_setitem(fields, nk[].k_max_depth_resolved_, _int(meta[3]))
    py_setitem(fields, nk[].k_max_features_, _int(meta[4]))
    var bootstrap_rows = _int(meta[5])
    py_setitem(fields, nk[].k__n_samples_bootstrap, bootstrap_rows if Int(py=bootstrap_rows) != 0 else PythonObject(None))


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
    var nk = trees_keys()
    var cfg = py_item(fields, nk[].k__cfg)
    var config = forest_config(fields, cfg, kind, not _classifier(kind), context)
    if _classifier(kind):
        var labels = trees_encode_labels(y, labels_scope(model, context))
        py_setitem(fields, nk[].k_classes_, labels.classes)
        var k = len(labels.classes)
        py_setitem(fields, nk[].k_n_classes_, PythonObject(k))
        # `codes.astype('<f4')`: the ET classifier reads float32 class codes.
        var codes = admitted_vector(labels.n, "<f4", context)
        trees_cast(trees_address(labels.codes), "<i4", trees_address(codes), "<f4", labels.n, context)
        _ = codes  # keep the Python object alive through the native call above
        var class_weight = py_item(cfg, nk[].k_class_weight)
        if sample_weight is not None or class_weight is not None:
            var xtrees = trees_table(context, mode, "_mojolearn_x_trees")
            var weights = trees_weighted_rows(sample_weight, class_weight, labels, xtrees, context)
            if tree_start is not None:
                _et_fit_arrays(fields, config, True, X, trees_address(codes), labels.n, k,
                               ForestFit(True, weights=weights.address, tree_start=Int(py=tree_start)), et, context)
                _ = codes  # keep the Python object alive through the native call above
                _ = weights  # keep the Python object alive through the native call above
                _ = weights^
                return model
            _et_fit_arrays(fields, config, True, X, trees_address(codes), labels.n, k,
                           ForestFit(True, rowmajor_ok=True, weights=weights.address), et, context)
            _ = codes  # keep the Python object alive through the native call above
            _ = weights^
            return model
        if tree_start is not None:
            _et_fit_arrays(fields, config, False, X, trees_address(codes), labels.n, k,
                           ForestFit(True, tree_start=Int(py=tree_start)), et, context)
            _ = codes  # keep the Python object alive through the native call above
            return model
        _et_fit_arrays(fields, config, False, X, trees_address(codes), labels.n, k,
                       ForestFit(True, rowmajor_ok=True), et, context)
        _ = codes  # keep the Python object alive through the native call above
        _ = codes^
        return model
    var y32 = admit_f32(y, 1, "y", context, finite=False)
    _target_check(y32.view(), config, context)
    if sample_weight is not None:
        var weights = admit_weight_words(sample_weight, y32.rows, context)
        if tree_start is not None:
            _et_fit_arrays(fields, config, False, X, y32.address, y32.rows, 0,
                           ForestFit(False, weights=weights.address, tree_start=Int(py=tree_start)), et, context)
            _ = y32  # keep the Python object alive through the native call above
            _ = weights  # keep the Python object alive through the native call above
            _ = weights^
            return model
        _et_fit_arrays(fields, config, False, X, y32.address, y32.rows, 0,
                       ForestFit(False, rowmajor_ok=True, weights=weights.address), et, context)
        _ = y32  # keep the Python object alive through the native call above
        _ = weights^
        return model
    if tree_start is not None:
        _et_fit_arrays(fields, config, False, X, y32.address, y32.rows, 0,
                       ForestFit(False, tree_start=Int(py=tree_start)), et, context)
        _ = y32  # keep the Python object alive through the native call above
        return model
    _et_fit_arrays(fields, config, False, X, y32.address, y32.rows, 0, ForestFit(False, rowmajor_ok=True), et, context)
    _ = y32  # keep the Python object alive through the native call above
    _ = y32^
    return model


def dt_random_fit(model: PythonObject, args: PythonObject, kind: String, context: PythonObject) raises -> PythonObject:
    """`_expansion_trees._dt_random_fit`: splitter='random' grows ONE
    extra-trees tree. The inner ExtraTrees estimator is assembled with the
    state its constructor stores (its defaults, then the tree's settings),
    fitted by the extra-trees operation above, and its flat model becomes
    this tree's; predict delegates to it."""
    var inner = forest_dt_random_inner(model, kind, context)
    var inner_kind = String("et_classifier") if _classifier(kind) else String("et_regressor")
    var inner_args = Python.tuple(args[0], args[1], args[2], PythonObject(False), PythonObject(None))
    _ = et_fit(inner, inner_args, inner_kind, context)
    forest_dt_random_adopt(model, inner)
    return model


def forest_dt_random_inner(model: PythonObject, kind: String, context: PythonObject) raises -> PythonObject:
    """UNPACK of a random-splitter tree's fit (`dt_random_fit`; lane mf-ada2
    shares it with AdaBoost's member rounds): the tree refreshed and its
    UNFITTED inner one-tree ExtraTrees estimator, assembled with the state
    its constructor stores (its defaults, then the tree's settings)."""
    var b = trees_builtins()
    var fields = forest_refresh(model, kind, context)
    var nk = trees_keys()
    var inner_cls = py_item(context, nk[].k_trees_et_classifier) if _classifier(kind) else py_item(context, nk[].k_trees_et_regressor)
    var inner = py_call(b.object.__new__, inner_cls)
    var inner_fields = py_fields(inner)
    _ = py_call(inner_fields.update, py_item(context, nk[].k_trees_defaults)[inner_cls])
    var max_features = py_item(fields, nk[].k_max_features)
    var seed = py_item(fields, nk[].k_random_state)
    var settings = Python.dict()
    py_setitem(settings, nk[].k_n_estimators, PythonObject(1))
    py_setitem(settings, nk[].k_bootstrap, PythonObject(False))
    py_setitem(settings, nk[].k_max_features, PythonObject(1.0) if max_features is None else max_features)
    var copied = Python.tuple("max_depth", "min_samples_split", "min_samples_leaf", "max_leaf_nodes",
                              "min_impurity_decrease", "criterion", "min_weight_fraction_leaf")
    for i in range(len(copied)):  # driver-loop: over fixed call structure, small-loop(copied): the seven settings the inner tree takes as given
        py_setitem(settings, copied[i], fields[copied[i]])
    py_setitem(settings, nk[].k_random_state, PythonObject(0) if seed is None else seed)
    py_setitem(settings, nk[].k_numeric_mode, py_item(fields, nk[].k_numeric_mode))
    if _classifier(kind):
        py_setitem(settings, nk[].k_class_weight, py_item(fields, nk[].k_class_weight))
    _ = py_call(inner_fields.update, settings)
    py_setitem(inner_fields, nk[].k__has_constructor_parameters, PythonObject(True))
    return inner


def forest_dt_random_adopt(model: PythonObject, inner: PythonObject) raises:
    """PACK of a random-splitter tree's fit: its FITTED inner forest kept
    (`_random_inner`) and the fitted attributes the tree shares with it."""
    var nk = trees_keys()
    var fields = py_fields(model)
    var inner_fields = py_fields(inner)
    py_setitem(inner_fields, nk[].k__mojolearn_fitted, PythonObject(True))
    py_setitem(fields, nk[].k__random_inner, inner)
    var names = Python.tuple("_offsets", "_colid", "_quesval", "_left_child", "_leaves", "_n_trees", "_num_outputs",
                             "n_features_in_")
    for i in range(len(names)):  # driver-loop: over fixed call structure, small-loop(names): the fitted attributes the tree shares with its inner forest
        py_setitem(fields, names[i], inner_fields[names[i]])
    if py_contains(inner_fields, nk[].k_classes_):
        py_setitem(fields, nk[].k_classes_, py_item(inner_fields, nk[].k_classes_))
        py_setitem(fields, nk[].k_n_classes_, PythonObject(len(py_item(inner_fields, nk[].k_classes_))))


def forest_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`fit` / `_fit_with_tree_start` of any connected forest class."""
    var kind = forest_kind(model, context)
    # A refit releases the previous fit's kept snapshots now (the holder dict
    # the fit replaced did the same), not at the next predict.
    trees_resident_forget(model_identity_key(model))
    var result: PythonObject
    var nk = trees_keys()
    var splitter = py_get(py_fields(model), nk[].k_splitter) if kind.startswith("dt_") else PythonObject(None)
    if _family(kind) == "et":
        result = et_fit(model, args, kind, context)
    elif splitter is not None and Bool(py=splitter == PythonObject("random")):
        result = dt_random_fit(model, args, kind, context)
    else:
        result = rf_fit(model, args, kind, context)
    _report_engine(model, context)
    return result


# ------------------------------------------------------ tree-range shards
# Lane forest-typed-fit (2026-11-12): `parallel_ensemble.fit_forest`
# (bindings/parallel_shard_native.mojo `_fit_forest`) fits whole global
# tree-ID ranges in worker processes. The driver runs this door's UNPACK once
# (`forest_shard_prepare`: the refresh, the labels or targets, the class
# weights and the typed request words); each worker receives the words and
# the arrays and calls the one typed fit core (`forest_shard_fit` ->
# `forest_fit_core`), never building an estimator; the driver's PACK
# (`forest_shard_publish`) stores the joined model as the plain fit does.


@fieldwise_init
struct ForestShardPlan(Movable):
    """The driver's prepared tree-range fit: the refreshed instance
    dictionary, the kind, the encoded class list (None for a regressor),
    the targets every shard fits (int32 codes for RF, float32 codes for ET,
    float32 targets), the class-weight row weights (None when unweighted),
    the worker state record `(rf, classifier, request words)` and X's
    width."""
    var fields: PythonObject
    var kind: String
    var classifier: Bool
    var classes: PythonObject
    var y: PythonObject
    var weights: PythonObject
    var state: PythonObject
    var cols: Int


def forest_shard_prepare(model: PythonObject, rows: Int, cols: Int, y_source: PythonObject,
                         context: PythonObject) raises -> ForestShardPlan:
    """UNPACK of a tree-range fit, once in the driver: the constructor
    checks and refresh the plain fit runs (IDENTICAL only), the labels or
    the checked targets, the class weights, and the one typed request
    (`forest_fit_request`) as words for the workers."""
    var kind = forest_kind(model, context)
    var params = forest_get_params(model, context)
    var mode = py_get(params, PythonObject("numeric_mode"), PythonObject(None))
    if mode is not None and (py_type(mode) is not trees_builtins().str or String(mode) != "identical"):
        raise Error("ValueError: parallel forests require IDENTICAL numeric mode")
    var nk = trees_keys()
    py_setitem(py_fields(model), nk[].k_numeric_mode, PythonObject("identical"))
    trees_resident_forget(model_identity_key(model))
    var fields = forest_refresh(model, kind, context)
    var captured = forest_capture_mode(fields, context)
    var et = _family(kind) == "et"
    var native = trees_table(context, captured, "_mojolearn_trees" if et else "_mojolearn_rf")
    var cfg = py_item(fields, nk[].k__cfg)
    var classifier = _classifier(kind)
    var config = forest_config(fields, cfg, kind, not classifier, context)
    var unweighted = False
    var classes = PythonObject(None)
    var y = PythonObject(None)
    var weights = PythonObject(None)
    var k = 0
    if classifier:
        var labels = trees_encode_labels(y_source, labels_scope(model, context))
        _y_rows_check(labels.n, rows)
        py_setitem(fields, nk[].k_classes_, labels.classes)
        k = len(labels.classes)
        py_setitem(fields, nk[].k_n_classes_, PythonObject(k))
        if k < 2:
            raise Error("y has fewer than 2 classes")
        classes = labels.classes
        var xtrees = trees_table(context, captured, "_mojolearn_x_trees")
        if et:
            # `codes.astype('<f4')`: the ET classifier reads float32 class codes.
            var codes = admitted_vector(labels.n, "<f4", context)
            trees_cast(trees_address(labels.codes), "<i4", trees_address(codes), "<f4", labels.n, context)
            y = codes
            var class_weight = py_item(cfg, nk[].k_class_weight)
            if class_weight is not None:
                weights = trees_weighted_rows(PythonObject(None), class_weight, labels, xtrees, context).array
                # the class weights travel as row weights
                unweighted = True
        else:
            y = labels.codes
            var class_weight = py_get(fields, nk[].k_class_weight)
            if class_weight is not None:
                var rows_only = trees_class_rows_only(class_weight, labels, xtrees, context)
                if rows_only:
                    weights = rows_only.value().array
    else:
        var checked = classical_check_y(y_source, rows, classical_estimator_name(model), context)
        var y32 = admit_f32(checked, 1, "y", context, finite=False)
        _y_rows_check(y32.rows, rows)
        _target_check(y32.view(), config, context)
        y = y32.array
    if not et and config.oob_score:
        raise Error("auxiliary outputs require a complete forest, not a tree-ID shard")
    # The addresses are the worker's own (forest_shard_fit); the words carry
    # the params views and the criterion.
    var request = forest_request(config, forest_key(native), rows, cols, k, 0, 0, ForestFit(classifier), unweighted)
    var state = Python.tuple(PythonObject(not et), PythonObject(classifier), forest_request_words_py(request))
    return ForestShardPlan(fields, kind^, classifier, classes, y, weights, state, cols)


def forest_shard_fit(state: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """A tree-range worker's fit: `state` is the plan's `(rf, classifier,
    request words)`, `args` `(X, y, weights or None, tree_start, trees)`.
    UNPACK (X as the column-major float32 copy the shard fit reads, the
    finiteness check, the request on the local addresses), the typed fit
    core, PACK the export as `{_offsets, _colid, _quesval, _left_child,
    _leaves, meta}`. No estimator object is built."""
    var rf = Bool(py=py_item(state, PythonObject(0)))
    var classifier = Bool(py=py_item(state, PythonObject(1)))
    var native = trees_table(context, "identical", "_mojolearn_rf" if rf else "_mojolearn_trees")
    var key = forest_key(native)
    var device_scan = (xf_caps(key) & XF_CAP_DEVICE_SCAN) != 0
    var x = admit_f32(py_item(args, PythonObject(0)), 2, "X", context, order="F", finite=False)
    var y = py_item(args, PythonObject(1))
    var weights = py_item(args, PythonObject(2))
    _y_rows_check(Int(py=y.shape[0]), x.rows)
    _finite_or_refuse(x, device_scan, context)
    var request = forest_request_from_words_py(py_item(state, PythonObject(2)), x.address, trees_address(y),
                                               Int(py=py_item(args, PythonObject(4))))
    request.weights = 0 if weights is None else trees_address(weights)
    request.tree_start = Int(py=py_item(args, PythonObject(3)))
    var selection = forest_export_selection()
    var handle = forest_fit_core(key, classifier, False, request, device_scan)
    _ = x.array  # keep X, y and the weights alive through the fit above
    _ = y
    _ = weights
    var exported = forest_export_handle(key, handle, selection, context)
    var out = Python.dict()
    var names = Python.tuple("_offsets", "_colid", "_quesval", "_left_child", "_leaves")
    for i in range(_MODEL_FIELDS):  # driver-loop: over the five model arrays
        py_setitem(out, names[i], exported[i])
    var nk = trees_keys()
    py_setitem(out, nk[].k_meta, exported[5])
    return out


def forest_shard_publish(plan: ForestShardPlan, merged: PythonObject, parts: PythonObject, count: Int,
                         context: PythonObject) raises:
    """PACK of a tree-range fit in the driver: the joined model arrays
    (`merged`, by field name) stored as the plain fit stores its export,
    `count` trees, the first range's metadata for the rest (the ET fitted
    fields; the RF notices), the ET depth-cap flag of any range."""
    var fields = plan.fields
    var nk = trees_keys()
    var meta = py_call(trees_builtins().list, py_item(parts[0], nk[].k_meta))
    py_setitem(meta, PythonObject(0), PythonObject(count))
    var names = Python.tuple("_offsets", "_colid", "_quesval", "_left_child", "_leaves")
    var arrays = Python.list()
    for i in range(_MODEL_FIELDS):  # driver-loop: over the five model arrays
        _ = py_call(arrays.append, merged[names[i]])
    _ = py_call(arrays.append, meta)
    _store_model(fields, arrays, plan.cols)
    if _family(plan.kind) == "et":
        var bound = False
        for r in range(len(parts)):  # driver-loop: one flag per tree range
            if _truthy(py_item(parts[r], nk[].k_meta)[2]):
                bound = True
        py_setitem(fields, nk[].k_depth_cap_bound_, PythonObject(bound))
        py_setitem(fields, nk[].k_max_depth_resolved_, _int(meta[3]))
        py_setitem(fields, nk[].k_max_features_, _int(meta[4]))
        var bootstrap_rows = _int(meta[5])
        py_setitem(fields, nk[].k__n_samples_bootstrap, bootstrap_rows if Int(py=bootstrap_rows) != 0 else PythonObject(None))
    else:
        _rf_fit_notices(meta, context)


def _report_engine(model: PythonObject, context: PythonObject) raises:
    """`inference_engine_`: the engine predict runs for the fitted model
    ('sequential' or 'parallel_groves'; `auto` resolved for the fitted
    numeric mode), set once at fit (predict leaves `__dict__` unchanged)."""
    var fields = py_fields(model)
    var nk = trees_keys()
    if not py_contains(fields, nk[].k__offsets):
        return
    py_setitem(fields, nk[].k_inference_engine_, PythonObject(_engine(fields, forest_effective_mode(fields, context))))


# ---------------------------------------------------------------- predict
# A predict is three steps (lane dw-trees, 2026-11-21):
#   UNPACK  `_stage`: the fitted state as plain values (`ForestState`,
#           bindings/forest_predict_run.mojo), read once; the model arrays
#           are admitted only where the route reads them (`_snapshot` on a
#           miss, `_walk_words` for the sequential walk).
#   RUN     `forest_snapshot_prepare`, `forest_vote`, `forest_labels`: typed
#           values and addresses only.
#   PACK    the result Array (allocated before RUN, which writes into it),
#           the widening, the label decode, and the Python pins of a kept
#           snapshot.
# The fitted bytes stay the five package arrays in the instance dictionary:
# the boost door, the explainers, save and pickling read them by name there.
# model.check, fitted-model form: the state transfer to the other device's twin would attach here (a class hook of `_check_both.on_device`; the shared-files lane owns it).


def _engine(fields: PythonObject, mode: String) raises -> String:
    """`ForestProtocol._prediction_engine()`: IDENTICAL `auto` predicts as
    'sequential', FAST `auto` as 'parallel_groves'."""
    var nk = trees_keys()
    var engine = _check_engine(py_get(fields, nk[].k_inference_engine, PythonObject("sequential")))
    if engine == "auto":
        return "parallel_groves" if mode == "fast" else "sequential"
    return engine


def _ordered_export(native: PythonObject) raises -> Bool:
    """The image takes the resident snapshot's ordered word
    (XF_CAP_ORDERED_RESIDENT, the former `forest_ordered_resident` export
    of a GPU binding)."""
    return (xf_caps(forest_key(native)) & XF_CAP_ORDERED_RESIDENT) != 0


def forest_handle_done(use: ForestHandle):
    """End a predict's use of its snapshot (non-raising: error paths)."""
    if use.kept:
        trees_resident_unpin(use.handle)
        return
    try:
        xf_snapshot_release(use.key, use.release_op, use.handle)
    except:
        pass


def _model_list(fields: PythonObject) raises -> List[PythonObject]:
    """The five model arrays of the instance dictionary, in export order."""
    var nk = trees_keys()
    var arrays = List[PythonObject](capacity=_MODEL_FIELDS)
    arrays.append(py_item(fields, nk[].k__offsets))
    arrays.append(py_item(fields, nk[].k__colid))
    arrays.append(py_item(fields, nk[].k__quesval))
    arrays.append(py_item(fields, nk[].k__left_child))
    arrays.append(py_item(fields, nk[].k__leaves))
    return arrays^


def _state(fields: PythonObject, native: PythonObject, key: EntryRef, caps: Int, mode: String,
           engine: String) raises -> ForestState:
    """UNPACK of a fitted forest for the image `key`: the mode and engine as
    words, the route, and the dimensions the route reads (the fitted width
    only where a snapshot is keyed on it)."""
    var nk = trees_keys()
    var groves = engine == "parallel_groves"
    var outputs = Int(py=_int(py_item(fields, nk[].k__num_outputs)))
    var trees = Int(py=_int(py_item(fields, nk[].k__n_trees)))
    var features = 0
    if groves or (caps & (XF_CAP_ORDERED_RESIDENT | XF_CAP_STRICT)) != 0:
        features = Int(py=_int(py_item(fields, nk[].k_n_features_in_)))
    return ForestState(key, Int(native._obj_ptr), caps, mode == "fast", groves, features, trees, outputs)


def _snapshot(model: PythonObject, fields: PythonObject, native: PythonObject, state: ForestState,
              context: PythonObject) raises -> ForestHandle:
    """`ForestProtocol._prepare_resident_forest()` through the owner: the
    snapshot of the five model arrays for the state's route (the device
    snapshot, or a host image's strict snapshot), prepared once and kept in
    the tree resident table (bindings/trees_resident_table_run.mojo) while
    the model, its generation, the arrays, dimensions, mode and aggregation
    are the same. The returned handle is pinned (or owned by the call); the
    caller ends it with `forest_handle_done` after its launch.

    The snapshot is keyed by (model_identity_key, model_generation), never
    held in the model's `__dict__`. The mirrored arrays are frozen in place
    (`fitted_freeze`), so an in-place write raises and the snapshot stays the
    same bytes as the public arrays; replacing an array changes its identity
    and the next call prepares again.

    A hit is the table read and the pin. On a miss the arrays are admitted
    (their flat dtypes, sizes and addresses: UNPACK), `forest_snapshot_prepare`
    prepares and lists the snapshot (RUN), and the Python pins of a kept
    snapshot are stored beside it before anything launches on it."""
    var strict = state.route == FOREST_ROUTE_STRICT
    var kind = TREES_RESIDENT_FOREST_STRICT if strict else TREES_RESIDENT_FOREST
    var release_op = XF_STRICT_RELEASE if strict else XF_RESIDENT_RELEASE
    var arrays = _model_list(fields)
    var model_key = model_identity_key(model)
    var generation = model_generation(model)
    var held = trees_resident_find(model_key, generation, kind, native, state.words, arrays)
    if held > 0:
        trees_resident_pin(held)
        return ForestHandle(held, True, state.key, release_op)
    var nk = trees_keys()
    var array_class = py_item(context, nk[].k_array_class)
    var addrs = List[Int](capacity=_MODEL_FIELDS)
    var sizes = List[Int](capacity=_MODEL_FIELDS)
    for i in range(_MODEL_FIELDS):  # driver-loop: over the five model arrays
        if (py_type(arrays[i]) is not array_class or String(arrays[i].dtype) != forest_export_dtype(i)
                or Int(py=arrays[i].ndim) != 1):
            raise Error("forest model arrays must have their original flat dtypes")
        sizes.append(Int(py=arrays[i].size))
        addrs.append(trees_address(arrays[i]))
    if not strict:
        forest_shape_check(state, sizes, addrs[0])
    for i in range(_MODEL_FIELDS):  # driver-loop: over the five model arrays
        # The read-only rule (p3nc_fitted_freeze): every mirrored exact Array
        # is read-only from here on (the fit stored them frozen already; an
        # Array assigned later is frozen in place, not copied).
        _ = fitted_freeze(arrays[i], context)
    var watchable = trees_resident_watch(model, model_key)
    var use = forest_snapshot_prepare(state, kind, release_op, model_key, generation, addrs, sizes[1],
                                      trees_resident_ids(arrays), watchable)
    if use.kept:
        try:
            trees_resident_keep(use.handle, native, arrays)
        except error:
            forest_handle_done(use)
            raise error^
    _ = arrays  # keep the Python objects alive through the native calls above
    return use^


def _resident(model: PythonObject, fields: PythonObject, native: PythonObject, mode: String,
              context: PythonObject) raises -> Int:
    """The kept device snapshot of a fitted forest on the binding `native`
    (the explainers' resident route): its handle, or 0 when nothing could be
    kept (the snapshot was released again)."""
    var engine = _engine(fields, mode)
    var key = forest_key(native)
    var caps = xf_caps(key)
    if engine != "parallel_groves" and (caps & XF_CAP_ORDERED_RESIDENT) == 0:
        raise Error("resident forest preparation requires parallel_groves")
    var nk = trees_keys()
    if not py_contains(fields, nk[].k__offsets):
        raise Error("RuntimeError: this estimator is not fitted yet")
    var state = _state(fields, native, key, caps, mode, engine)
    var use = _snapshot(model, fields, native, state, context)
    var handle = use.handle if use.kept else 0
    forest_handle_done(use)
    return handle


struct ForestStaged(Movable):
    """One predict's UNPACK: the admitted X, the fitted state as plain
    values, and the Python objects the boundary still reads or pins (the
    binding, the model whose snapshot is kept, its instance dictionary)."""
    var x: AdmittedX
    var state: ForestState
    var native: PythonObject
    var model: PythonObject
    var fields: PythonObject

    def __init__(out self, var x: AdmittedX, var state: ForestState, native: PythonObject, model: PythonObject,
                 fields: PythonObject):
        self.x = x^
        self.state = state^
        self.native = native
        self.model = model
        self.fields = fields


def _stage(model: PythonObject, fields: PythonObject, kind: String, var x: AdmittedX,
           context: PythonObject) raises -> ForestStaged:
    """The resolved predict state of a fitted model over an admitted X."""
    var nk = trees_keys()
    if not py_contains(fields, nk[].k__offsets):
        raise Error("RuntimeError: this estimator is not fitted yet")
    var mode = forest_effective_mode(fields, context)
    var native = trees_table(context, mode, "_mojolearn_trees" if _family(kind) == "et" else "_mojolearn_rf")
    var engine = _engine(fields, mode)
    var key = forest_key(native)
    var caps = xf_caps(key)
    var state = _state(fields, native, key, caps, mode, engine)
    return ForestStaged(x^, state^, native, model, fields)


def _predict_input(model: PythonObject, fields: PythonObject, kind: String, X: PythonObject,
                   context: PythonObject) raises -> ForestStaged:
    """`_check_predict_input` / `_vote`'s checks: fitted, X as C-order
    float32, the fitted width."""
    var nk = trees_keys()
    if not py_contains(fields, nk[].k__offsets):
        raise Error("RuntimeError: this estimator is not fitted yet")
    var x = admit_f32(X, 2, "X", context, finite=False)
    var fitted = Int(py=_int(py_item(fields, nk[].k_n_features_in_)))
    trees_check_features(x.cols, fitted)
    return _stage(model, fields, kind, x^, context)


def _vote(staged: ForestStaged, sequential_name: String, context: PythonObject) raises -> PythonObject:
    """`ForestProtocol._predict_forest`: the rows' votes or values, flat
    float32, through the state's route."""
    var rows = staged.x.rows
    var result = admitted_vector(rows * staged.state.outputs, "<f4", context)
    var dst = trees_address(result)
    var wrote = 0
    if staged.state.route == FOREST_ROUTE_SEQUENTIAL:
        var arrays = _model_list(staged.fields)
        var nk = trees_keys()
        var array_class = py_item(context, nk[].k_array_class)
        var addrs = List[Int](capacity=_MODEL_FIELDS)
        var views = List[PythonObject]()
        for i in range(_MODEL_FIELDS):  # driver-loop: over the five model arrays
            # An exact mojolearn Array's data address is its `_addr` slot; any
            # other storage is viewed (UNPACK: the view is kept alive through
            # the vote below).
            if py_type(arrays[i]) is array_class:
                addrs.append(trees_address(arrays[i]))
            else:
                var view = classical_view(arrays[i], context, "forest model")
                addrs.append(Int(py=classical_address(view)))
                views.append(view)
        wrote = forest_vote(staged.state, 0, addrs, staged.x.view(), dst, sequential_name == "rf_predict_reg")
        _ = views^
        _ = arrays^
    else:
        var none = List[Int]()
        var use = _snapshot(staged.model, staged.fields, staged.native, staged.state, context)
        try:
            wrote = forest_vote(staged.state, use.handle, none, staged.x.view(), dst, False)
        finally:
            forest_handle_done(use)
    _ = result  # keep the Python object alive through the native call above
    forest_wrote_check(wrote, rows, sequential_name)
    return result


def _resident_labels(staged: ForestStaged, context: PythonObject) raises -> PythonObject:
    """`ForestProtocol._predict_forest_labels`: the resident classifier's
    int32 codes, or None where the fused label route does not apply."""
    if not forest_labels_route(staged.state):
        return PythonObject(None)
    var rows = staged.x.rows
    var result = admitted_vector(rows, "<i4", context)
    var use = _snapshot(staged.model, staged.fields, staged.native, staged.state, context)
    var wrote = 0
    try:
        wrote = forest_labels(staged.state, use.handle, staged.x.view(), trees_address(result))
    finally:
        forest_handle_done(use)
    _ = result  # keep the Python object alive through the native call above
    forest_wrote_check(wrote, rows, "forest label prediction")
    return result


def _widen(vote: PythonObject, shape: PythonObject, context: PythonObject) raises -> PythonObject:
    """`vote.astype('<f8')` under `shape`."""
    var n = Int(py=vote.size)
    var wide = admitted_array(shape, "<f8", context)
    trees_cast(trees_address(vote), "<f4", trees_address(wide), "<f8", n, context)
    _ = wide  # keep the Python object alive through the native call above
    return wide


def _inner(fields: PythonObject) raises -> PythonObject:
    """A splitter='random' tree's inner extra-trees forest, or None."""
    var nk = trees_keys()
    return py_get(fields, nk[].k__random_inner)


def _proba_staged(kind: String, staged: ForestStaged, context: PythonObject) raises -> PythonObject:
    var shape = Python.tuple(staged.x.rows, staged.state.outputs)
    if _family(kind) == "et":
        return _widen(_vote(staged, "et_predict", context), shape, context)
    return trees_reshape(_vote(staged, "rf_predict_proba", context), shape, context)


def _codes_staged(kind: String, staged: ForestStaged, context: PythonObject) raises -> PythonObject:
    """A classifier's undecoded class codes: the resident fused labels
    (int32), else the first-max argmax of the vote (int64)."""
    var codes = _resident_labels(staged, context)
    if codes is not None:
        return codes
    var vote = _vote(staged, "et_predict" if _family(kind) == "et" else "rf_predict_proba", context)
    return trees_argmax_rows(vote, staged.x.rows, staged.state.outputs, False, context)


def _predict_staged(kind: String, staged: ForestStaged, context: PythonObject) raises -> PythonObject:
    if _classifier(kind):
        var nk = trees_keys()
        return trees_decode_labels(py_item(staged.fields, nk[].k_classes_), _codes_staged(kind, staged, context), context)
    # predict has one type (Andrew, 2026-11-22, lane predict-dtype): the
    # float32 mean vote for both families, with no float64 widening pass
    # (it was exact: the same values in a second buffer twice the size).
    if _family(kind) == "et":
        return _vote(staged, "et_predict", context)
    return _vote(staged, "rf_predict_reg", context)


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
    return _proba_staged(kind, staged, context)


def forest_predict(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`predict`: a classifier's labels (the resident fused labels, else the
    first-max argmax of the vote) decoded through `classes_`; a regressor's
    values (float32 for RF, ET and DT)."""
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
    return _predict_staged(kind, staged, context)


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
    var nk = trees_keys()
    return Python.tuple(py_item(fields, nk[].k_classes_), _codes_staged(kind, staged, context))


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
        var nk = trees_keys()
        if not py_contains(inner_fields, nk[].k__offsets):
            raise Error("RuntimeError: this estimator is not fitted yet")
        var fitted = Int(py=_int(py_item(inner_fields, nk[].k_n_features_in_)))
        trees_check_features(x.cols, fitted)
        var inner_staged = _stage(inner, inner_fields, inner_kind, x.copy(), context)
        return _public_deferred(model, inner_kind, inner_staged, proba, codes, context)
    var staged = _stage(model, fields, kind, x.copy(), context)
    return _public_deferred(model, kind, staged, proba, codes, context)


def _public_deferred(public: PythonObject, kind: String, staged: ForestStaged, proba: Bool, codes: Bool,
                     context: PythonObject) raises -> PythonObject:
    """`_public_staged` with the deferred finite test of the door's X (lane
    a7c): the public door admits X with `defer_finite`, so a borrowed X
    reaches here untested (`finite` False). The resident route tests the
    words it uploads (the device scan after the upload on the GPU, the
    row tasks' scan of the host groves on the CPU column) and refuses in
    its own words, which the one scikit-learn wording replaces; the
    sequential routes (`forest_strict_predict`, `rf_predict_*`,
    `et_predict`) test nothing, so a deferred X takes the door's parallel
    host scan before them (the scan admit_x made before, now only there).
    Refusals name the public estimator (`public`)."""
    var est = classical_estimator_name(public)
    if not staged.x.finite and staged.state.route != FOREST_ROUTE_RESIDENT:
        if not trees_finite_f32(staged.x.address, staged.x.rows * staged.x.cols):
            admitted_finite_refuse(1, est)
    var result = PythonObject(None)
    try:
        result = _public_staged(kind, staged, proba, codes, context)
    except error:
        if not staged.x.finite:
            admitted_finite_entry_refuse(error, est)
        raise error^
    return result


def _public_staged(kind: String, staged: ForestStaged, proba: Bool, codes: Bool,
                   context: PythonObject) raises -> PythonObject:
    if proba:
        return _proba_staged(kind, staged, context)
    if codes:
        var nk = trees_keys()
        return Python.tuple(py_item(staged.fields, nk[].k_classes_), _codes_staged(kind, staged, context))
    return _predict_staged(kind, staged, context)
