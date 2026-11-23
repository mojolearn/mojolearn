# SPDX-License-Identifier: Apache-2.0
"""The member boundary shared by the Tree (13), Forest (14) and ExtraTrees
(15) families (UNPACK / PACK of core/forest_member_run.mojo). Each family's
boundary file (providers/python/member_<family>_boundary_door.mojo) calls
these with the kind prefix its classes connect under ("dt_", "rf_", "et_").

  * `forest_member_admit_common` (UNPACK, once per member): the estimator's
    own constructor checks and derived configuration (`forest_get_params`,
    the RF / ET derive, `forest_config`) on a SCRATCH dictionary, so the
    template estimator a meta-estimator holds is never mutated; the fit
    mode and predict engine the estimator's own fit and predict would read;
    its route (the run context of its device, `trees_route_context`, and the
    forest entry of that context's image in `route.entry`); the parameter
    words of core/forest_member_run.mojo (word 10, the importance fit, left
    at 0: a caller that reads importances sets it; word 11, the out-of-bag
    fit, from the estimator's `oob_score`: lane mf-oob3). Returns False, so
    the caller keeps the estimator's own doors, for what the member does not
    publish: DecisionTree splitter='random' (its inner Extra Trees forest),
    `class_weight`, and the deviance criteria of a regressor (poisson, gamma, inverse_gaussian:
    their target check reads y).
    `caller_rows` (lane mf-ada2, AdaBoost's member rounds): the caller
    multiplies the class_weight row weights into the fit's row weights
    itself and runs the deviance criteria's target check on each fit's
    targets itself (xtrees/trees_ada_fit_run.mojo), so those two are
    admitted; splitter='random' still is not (the caller admits a
    random-splitter tree's inner forest itself).
  * `forest_member_pack_common` (PACK): the fitted attributes and model
    arrays the estimator's own fit writes, from the state's blocks, and
    `feature_importances_` when the state holds them (else removed, as the
    estimator's fit removes it), the out-of-bag outputs of a state fitted
    with word 11 (`oob_decision_function_` or `oob_prediction_`, float64,
    and `oob_score_`, as the estimator's oob fit writes them), and
    `classes_` / `n_classes_` of a
    classifier: the caller's labels (`member_pack_labeled`) are TAKEN before
    the refresh (which clears the instance dictionary, the held key with it)
    and written after it, as the estimator's own fit writes them.
  * `forest_member_state_common` (UNPACK): a member state from a fitted
    estimator's five model arrays and, when it has them, its
    `feature_importances_`.

SERIAL BY NATURE: one member's admission or publication (a few record reads
and five block copies); a caller runs its members in parallel.
"""
from std.memory import unsafe_memcpy, bitcast
from std.python import Python, PythonObject
from bindings.member_owner_unpack import MemberOwner, member_labels_take
from bindings.py_balanced import py_call, py_get, py_item, py_setitem, py_fields
from bindings.admitted_input_unpack import admitted_vector, admitted_array, classical_view, admitted_dtype
from bindings.classical_arrays_door import classical_address
from bindings.trees_public_inputs_door import trees_builtins, trees_table, trees_freeze, trees_address
from bindings.trees_route_boundary_door import trees_route_context
from bindings.trees_public_forest_door import (
    forest_kind, forest_get_params, forest_config, forest_capture_mode, forest_prediction_engine, forest_key,
    forest_refresh, _rf_derive, _et_derive,
)
from bindings.forest_config_run import forest_target_need, FOREST_TARGET_ANY, FOREST_FEATURES_REFUSED
from core.device_route import DeviceRoute, DEVICE_CPU, DEVICE_GPU
from core.forest_member_run import FM_WORD_OOB, FM_HEAD, FM_BLOCK_OOB, FM_BLOCK_OOB_SCORE
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_SCALARS, MEMBER_IMPORT_DONE,
    MEMBER_BLOCK_FEATURE_IMPORTANCES, MEMBER_F32, MEMBER_F64, MEMBER_I32,
)

comptime _NAMES = 5


def _model_names() raises -> PythonObject:
    return Python.tuple("_offsets", "_colid", "_quesval", "_left_child", "_leaves")


def _dtype(index: Int) -> Int:
    return MEMBER_F32 if index == 2 or index == 4 else MEMBER_I32


def _family_of(kind: String) -> String:
    return String("et") if kind.startswith("et_") else String("rf")


def forest_member_admit_common(model: PythonObject, kind_word: Int, prefix: String, context: PythonObject,
    mut owner: MemberOwner, caller_rows: Bool = False) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator (see the module docstring)."""
    var run = trees_route_context(model, context)
    var kind = forest_kind(model, run)
    var classifier = kind.endswith("_classifier")
    if not kind.startswith(prefix) or classifier != (kind_word == 0):
        return False
    var et = _family_of(kind) == "et"
    var params = forest_get_params(model, run)
    # The estimator's own constructor checks and derived state, on a scratch
    # dictionary (what `forest_refresh` would leave in the instance).
    var derived = _et_derive(model, params, kind, run) if et else _rf_derive(model, params, kind, run)
    var b = trees_builtins()
    var scratch = py_call(b.dict, params)
    _ = py_call(scratch.update, derived)
    if kind.startswith("dt_"):
        var splitter = py_get(scratch, PythonObject("splitter"))
        if splitter is not None and Bool(py=splitter == PythonObject("random")):
            return False
    if classifier and not caller_rows and py_get(scratch, PythonObject("class_weight")) is not None:
        return False
    var mode = forest_capture_mode(scratch, run)
    var engine = forest_prediction_engine(scratch, run)
    var config = forest_config(scratch, py_item(scratch, PythonObject("_cfg")), kind, not classifier, run)
    # `oob_score` (lane mf-oob3): word 11 below. Only the RF configuration
    # reads it (`_rf_config`); an ExtraTrees `oob_score=True` reaches its own
    # fit, which refuses it in its words, as before.
    if config.class_weighted and not caller_rows:
        return False
    if not classifier and not caller_rows and forest_target_need(config) != FOREST_TARGET_ANY:
        return False
    # The conversions the estimator's fit would refuse: refused now, in its words.
    if config.refusal != "":
        raise Error(config.refusal)
    if config.refusal_late != "":
        raise Error(config.refusal_late)
    if config.features.form == FOREST_FEATURES_REFUSED:
        raise Error(config.features.refusal)
    var native = trees_table(run, mode, "_mojolearn_trees" if et else "_mojolearn_rf")
    var key_words = py_get(native, PythonObject("trees_native_key" if et else "rf_native_key"))
    if key_words is None:
        raise Error("ImportError: mojolearn: the forest binding publishes no native entry table")
    var vendor = Int(py=key_words[1])
    var route = DeviceRoute(DEVICE_CPU if vendor == 0 else DEVICE_GPU)
    route.mode = Int(py=key_words[0])
    route.vendor = vendor
    route.entry = forest_key(native)
    owner.plan.route = route
    var slots = len(config.ints)
    var words = List[Float64](capacity=FM_HEAD + 3 * slots)
    words.append(0.0 if classifier else 1.0)
    words.append(1.0 if et else 0.0)
    words.append(Float64(config.criterion))
    words.append(Float64(config.features.form))
    words.append(Float64(config.features.count))
    words.append(config.features.fraction)
    words.append(1.0 if config.bins_image else 0.0)
    words.append(config.rows_fraction)
    words.append(1.0 if mode == "fast" else 0.0)
    words.append(1.0 if engine == "parallel_groves" else 0.0)
    words.append(0.0)  # word 10 (FM_WORD_IMPORTANCES): no importance fit unless the caller asks
    words.append(1.0 if config.oob_score else 0.0)  # word 11 (FM_WORD_OOB): the estimator's oob_score
    words.append(Float64(slots))  # word 12 (FM_WORD_SLOTS)
    for k in range(slots):  # driver-loop: over the configuration's parameter slots (about twenty)
        words.append(bitcast[DType.float64](Int64(config.ints[k])))
        words.append(config.floats[k])
        words.append(1.0 if config.bad[k] != 0 else 0.0)
    owner.plan.params = words^
    owner.pins = Python.tuple(native, run)
    return True


def forest_member_pack_common(model: PythonObject, context: PythonObject, b: PythonObject, plan: MemberPlan,
    state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`, as
    the estimator's own fit does (the refreshed configuration, the fit mode,
    the five frozen model arrays, the dimensions and a classifier's
    `classes_`). A RandomForest / DecisionTree fit's notices are the
    state's fit notices, raised by the caller (core/member_notices.mojo),
    never here."""
    var run = trees_route_context(model, context)
    var kind = forest_kind(model, run)
    # Taken before `forest_refresh` clears the instance dictionary.
    var labels = member_labels_take(py_fields(model), b)
    var fields = forest_refresh(model, kind, run)
    _ = forest_capture_mode(fields, run)
    var names = _model_names()
    var call = MemberCall(plan.route, state)
    for i in range(_NAMES):  # driver-loop: over the five model arrays
        call.block = MEMBER_BLOCK_FAMILY + i
        _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
        var count = call.dst.rows * call.dst.cols
        var array = admitted_vector(count, "<f4" if _dtype(i) == MEMBER_F32 else "<i4", run)
        unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=trees_address(array)),
            src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=call.dst.data), count=4 * count)
        trees_freeze(array)
        py_setitem(fields, names[i], array)
    call.block = MEMBER_BLOCK_SCALARS
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var scalars = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
    var features = Int(scalars[0])
    var classes = Int(scalars[4])
    var meta = List[Int]()
    for k in range(Int(scalars[5])):  # small-loop(meta): at most XF_META_MAX export metadata words
        meta.append(Int(scalars[6 + k]))
    py_setitem(fields, "n_features_in_", PythonObject(features))
    py_setitem(fields, "_n_trees", PythonObject(Int(scalars[1])))
    py_setitem(fields, "_num_outputs", PythonObject(Int(scalars[3])))
    py_setitem(fields, "_resident_holder", Python.dict())
    if classes > 0:
        py_setitem(fields, "n_classes_", PythonObject(classes))
        if labels is not None:
            py_setitem(fields, "classes_", labels)
    # ET's third metadata word is its depth cap (the RF / DT layout's is the
    # fit notice mask, carried by the state: core/member_notices.mojo).
    if _family_of(kind) == "et" and len(meta) >= 6:
        py_setitem(fields, "depth_cap_bound_", PythonObject(meta[2] != 0))
        py_setitem(fields, "max_depth_resolved_", PythonObject(meta[3]))
        py_setitem(fields, "max_features_", PythonObject(meta[4]))
        py_setitem(fields, "_n_samples_bootstrap", PythonObject(meta[5]) if meta[5] != 0 else PythonObject(None))
    # The block count (6, or 7 with importances) of the last export.
    if call.out_cols > 6:
        call.block = MEMBER_BLOCK_FEATURE_IMPORTANCES
        _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
        var width = call.dst.rows * call.dst.cols
        var importance = admitted_vector(width, "<f4", run)
        unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=trees_address(importance)),
            src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=call.dst.data), count=4 * width)
        py_setitem(fields, "feature_importances_", importance)
    else:
        _ = py_call(fields.pop, PythonObject("feature_importances_"), PythonObject(None))
    # The out-of-bag outputs (word 11), as `_rf_fit_classifier` /
    # `_rf_fit_regressor` write them: a new float64 Array and the score as a
    # Python float.
    if len(plan.params) > FM_WORD_OOB and plan.params[FM_WORD_OOB] != 0.0:
        call.block = FM_BLOCK_OOB
        _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
        var rows = call.dst.rows
        var cols = call.dst.cols
        var oob = admitted_array(Python.tuple(rows, cols), "<f8", run) if classes > 0 else admitted_vector(
            rows, "<f8", run)
        unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=trees_address(oob)),
            src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=call.dst.data), count=8 * rows * cols)
        call.block = FM_BLOCK_OOB_SCORE
        _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
        var score = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)[0]
        py_setitem(fields, "oob_decision_function_" if classes > 0 else "oob_prediction_", oob)
        py_setitem(fields, "oob_score_", PythonObject(score))
    py_setitem(fields, "_mojolearn_fitted", True)


def forest_member_state_common(model: PythonObject, context: PythonObject, plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the five model arrays of the
    fitted `model`. The caller frees it (`member_free`)."""
    var fields = py_fields(model)
    var names = _model_names()
    var views = List[PythonObject]()
    var addresses = List[Int]()
    var counts = List[Int]()
    for i in range(_NAMES):  # driver-loop: over the five model arrays
        var source = py_get(fields, names[i])
        if source is None:
            raise Error("NotFittedError: this member is not fitted yet")
        var view = classical_view(source, context, "forest model")
        addresses.append(Int(py=classical_address(view)))
        counts.append(Int(py=view.size))
        views.append(view)
    var trees = Int(py=py_item(fields, PythonObject("_n_trees")))
    var outputs = Int(py=py_item(fields, PythonObject("_num_outputs")))
    var classes_source = py_get(fields, PythonObject("n_classes_"))
    var classes = 0 if classes_source is None else Int(py=classes_source)
    var scalars: List[Float64] = [Float64(py=py_item(fields, PythonObject("n_features_in_"))), Float64(trees),
        Float64(counts[1]), Float64(outputs), Float64(classes), 2.0, Float64(trees), Float64(outputs)]
    var importance_source = py_get(fields, PythonObject("feature_importances_"))
    var importance = PythonObject(None)
    if importance_source is not None:
        importance = classical_view(importance_source, context, "forest feature importances")
        if admitted_dtype(importance, context) != "<f4":
            raise Error("TypeError: mojolearn: a forest's feature_importances_ is float32")
    var call = member_new(plan)
    try:
        for i in range(_NAMES):  # driver-loop: over the five model arrays
            call.block = MEMBER_BLOCK_FAMILY + i
            call.x = MemberMatrix(addresses[i], 1, counts[i], _dtype(i))
            _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        if importance is not None:
            call.block = MEMBER_BLOCK_FEATURE_IMPORTANCES
            call.x = MemberMatrix(Int(py=classical_address(importance)), 1, Int(py=importance.size), MEMBER_F32)
            _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, len(scalars), MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = views^
    _ = importance
    _ = len(scalars)
    return call.state
