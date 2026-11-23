# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the Gbdt family (16): UNPACK / PACK of
core/gbdt_member.mojo, called by `bindings/member_boundary_door.mojo`.

Published classes: `GradientBoostingClassifier` (kind 0),
`GradientBoosting` (kind 1, the booster: its own predict returns raw scores)
and `GradientBoostingRegressor` (kind 2). The adapters (kinds 0 and 2) are
their inner learner: the admission builds the learner their fit builds
(`adapter_member_learner`, a new object; the adapter is only read) and
takes ITS config, table and route, so the member fits exactly the learner's
request; the pack writes one learner (`_learner_`, packed from the member
state) and the adapter's copies of its attributes, as the adapter's fit
door writes them. Not published: `OrderedRMSE` (no mark; its own door):
its fit is a second algorithm, not a config word (XG_ORDERED_FIT, five
settings, an ordered RMSE booster of its own), and it needs a per-fit
`permutation` argument that no member call carries.

  * `gbdt_member_admit` (UNPACK, once per member): the fit's own option
    checks (`gbdt_check_option_tables`) and the typed config
    (`gbdt_config`, with the Ordered checks' texts) read once from the
    unfitted estimator, as the fit door reads them; the route of the gbdt
    image its fit would run in (`gbdt_table` of its run context: mode,
    vendor, entry), and the cat_features / one_hot_features lists as NEW's
    feature-list tail (lane mf-img3: the member fits the category map, the
    one-hot flags, the multi-target losses and, with the caller's group ids
    in `call.groups`, the query-wise losses). Returns False only for a class
    but GradientBoosting itself. An eval set and pairs are fit arguments a
    member call does not pass, so a caller with them keeps the doors by
    construction.
  * `gbdt_member_pack` (PACK): the fitted attributes the fit door's PACK
    writes (`model_`, `bias_`, `best_iteration_`, `stopped_early_`,
    `loss_curve_`, `test_loss_curve_` None, `n_features_in_`,
    `category_map_` (the member's map block, None without cat_features),
    `approx_dim_`, `n_classes_`, `learning_rate_`,
    `boosting_type_`), the old resident copy dropped as the fit drops it.
  * `gbdt_member_state` (UNPACK): a member state from a fitted estimator's
    model text, loss curve, scalars and (a CTR model) its category map.

  * The adapters (kinds 0, 2): admission as above over the learner, plus
    the adapter's own refusals (`_new_learner`: numeric_mode, the
    regressor's class_weights; the image's compiled tier against the mode).
    In `fast` the classifier member's codes take the learner's class
    kernels, as its door does (core/gbdt_member.mojo `_fast_codes`). PACK: `_learner_`, `numeric_mode_`, `n_features_in_`,
    `model_`, the five learner attributes; the classifier's `classes_`
    (the caller's labels, normalized as the adapter's fit normalizes them),
    `n_classes_` 2 and `_label_kind_`; `_mojolearn_fitted`. The caller's
    PACK writes `feature_names_in_`. STATE: from the adapter's `_learner_`.

SERIAL BY NATURE: one member's admission or publication (a few record reads
and three block copies); a caller runs its members in parallel.
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.member_owner_unpack import MemberOwner
from bindings.py_balanced import py_call, py_get, py_item, py_setitem, py_fields, py_type
from bindings.classical_arrays_door import classical_address
from bindings.admitted_input_unpack import classical_view, admitted_vector
from bindings.trees_keys_unpack import trees_keys
from bindings.trees_route_boundary_door import trees_route_context
from bindings.trees_public_gbdt_door import gbdt_table, gbdt_entry_of, _field, _repr
from bindings.py_typed_call import py_call_typed
from bindings.native_loaded_module_door import native_loaded_module
from bindings.trees_public_gbdt_fit_door import (
    gbdt_config, gbdt_check_option_tables, _tables, _truthy, _in, _f64_curve,
    adapter_member_learner, adapter_label_kind,
)
from bindings.native_typed_context_pack import native_typed_context
from bindings.classifier_public_lifecycle_door import _public_labels
from bindings.member_owner_unpack import member_labels_take
from gbdt.fit_calls import xg_numeric_mode
from bindings.gbdt_config_run import gbdt_config_words
from gbdt.fit_request import gbdt_string_at
from core.entry_table import entry_ref_mode, entry_ref_vendor
from core.device_route import DeviceRoute, DEVICE_CPU, DEVICE_GPU
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_SCALARS, MEMBER_IMPORT_DONE,
    MEMBER_F64, MEMBER_I32, MEMBER_I64,
)


comptime _ADAPTER_COPIED = 5


def _adapter_name(kind: Int) -> String:
    return String("GradientBoostingClassifier") if kind == 0 else String("GradientBoostingRegressor")


def gbdt_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator (see the module docstring)."""
    var name = String(py=py_type(model).__name__)
    if kind == 1:
        if name != "GradientBoosting":
            return False
        return _learner_admit(model, 1.0, trees_route_context(model, context), owner)
    if (kind != 0 and kind != 2) or name != _adapter_name(kind):
        return False
    # The adapter's typed run context, as its lifecycle door hands its fit.
    var native = native_typed_context(trees_route_context(model, context), False, True)
    var learner = adapter_member_learner(model, native)
    var nk = trees_keys()
    var mode = String(py=py_item(py_fields(learner), nk[].k_numeric_mode))
    var code = xg_numeric_mode(gbdt_entry_of(gbdt_table(py_fields(learner), native)))
    if code != (0 if mode == "fast" else 1):
        raise Error("RuntimeError: GBDT adapter native numeric mode disagrees with requested mode")
    return _learner_admit(learner, Float64(kind), native, owner)


def _learner_admit(model: PythonObject, kind: Float64, run: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """The booster's admission over `model` (GradientBoosting itself, or an
    adapter's learner) in its run context `run`; `kind` is NEW's word 0."""
    var own = py_fields(model)
    var nk = trees_keys()
    var cat = _field(own, model, nk[].k_cat_features, "cat_features")
    var hot = _field(own, model, nk[].k_one_hot_features, "one_hot_features")
    var t = _tables(run)
    var loss = _field(own, model, nk[].k_loss, "loss")
    if _in(loss, py_item(t, nk[].k_multi_regression_losses)):
        # The fit door's refusals of a multi-target loss that read no rows
        # (`gbdt_fit_unpack`, in its order and words); the rows' own (y's
        # width, group ids) are the member fit's.
        if _field(own, model, nk[].k_class_weights, "class_weights") is not None:
            raise Error("mojolearn: class_weights do not apply to loss=" + _repr(loss))
        if _truthy(cat) or _truthy(hot):
            raise Error("NotImplementedError: mojolearn: loss=" + _repr(loss) + " with cat_features or "
                        + "one_hot_features is not carried: the CTR target "
                        + "binarization reads one target per row")
        if _truthy(_field(own, model, nk[].k_boost_from_average, "boost_from_average")):
            raise Error("NotImplementedError: mojolearn: boost_from_average=True with loss="
                        + _repr(loss) + " is not carried: its StartingPoint is "
                        + "per-dimension (optimal_const_for_loss.h:230-239) and "
                        + "this model's bias is one value (DEVIATION 5951)")
    # The fit's own checks and settings, read once (the fit door reads the
    # same at every fit): refused now, in its words.
    gbdt_check_option_tables(own, model, run)
    var config = gbdt_config(own, model, True, run)
    var table = gbdt_table(own, run)
    var entry = gbdt_entry_of(table)
    var vendor = entry_ref_vendor(entry)
    var route = DeviceRoute(DEVICE_CPU if vendor == 0 else DEVICE_GPU)
    route.mode = entry_ref_mode(entry)
    route.vendor = vendor
    route.entry = entry
    owner.plan.route = route
    var words: List[Float64] = [kind]
    words.extend(gbdt_config_words(config))
    # The feature-list tail (core/gbdt_member.mojo NEW): cat_features, then
    # one_hot_features, as the estimator holds them (`int` of each, the fit
    # door's `gbdt_category_columns` / `_flags` reads), then their counts.
    var counts = List[Int](length=2, fill=0)
    var b = native_loaded_module("builtins")
    for which in range(2):  # driver-loop: over fixed call structure, small-loop(lists): the two feature-list arguments
        var values = cat if which == 0 else hot
        if not _truthy(values):
            continue
        var listed = py_call(b.list, values)
        for k in range(len(listed)):  # driver-loop: over the feature-list argument indices
            words.append(Float64(Int(py=py_call_typed(b.int, listed[k]))))
        counts[which] = len(listed)
    words.append(Float64(counts[0]))
    words.append(Float64(counts[1]))
    owner.plan.params = words^
    owner.pins = Python.tuple(table, run)
    return True


def gbdt_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`, as
    the fit door's PACK writes them."""
    if plan.kind == 0 or plan.kind == 2:
        _adapter_pack(model, context, b, plan, state)
        return
    _learner_pack(model, context, plan, state)


def _adapter_pack(model: PythonObject, context: PythonObject, b: PythonObject, plan: MemberPlan, state: Int) raises:
    """PACK of an adapter: ONE learner packed from the member state, then the
    adapter's fields as `_fit_native` and `adapter_classifier_fit` write them."""
    var own = py_fields(model)
    var nk = trees_keys()
    var labels = PythonObject(None)
    if plan.kind == 0:
        labels = member_labels_take(own, b)
        if labels is None or len(labels) != 2:
            raise Error("member ABI: a GradientBoostingClassifier member's pack needs its two class labels")
    var native = native_typed_context(trees_route_context(model, context), False, True)
    var learner = adapter_member_learner(model, native)
    _learner_pack(learner, context, plan, state)
    var learner_fields = py_fields(learner)
    py_setitem(own, nk[].k__learner_, learner)
    py_setitem(own, nk[].k_numeric_mode_, py_item(learner_fields, nk[].k_numeric_mode))
    py_setitem(own, nk[].k_n_features_in_, py_item(learner_fields, nk[].k_n_features_in_))
    py_setitem(own, nk[].k_model_, py_item(learner_fields, nk[].k_model_))
    var copied = Python.tuple("loss_curve_", "test_loss_curve_", "best_iteration_", "stopped_early_", "learning_rate_")
    for i in range(_ADAPTER_COPIED):  # driver-loop: over fixed call structure, small-loop(copied): the five learner attributes
        py_setitem(own, copied[i], learner_fields[copied[i]])
    if plan.kind == 0:
        var kind = adapter_label_kind(labels, "y", native)
        py_setitem(own, nk[].k_classes_, _public_labels(kind[1], native))
        py_setitem(own, nk[].k_n_classes_, PythonObject(2))
        py_setitem(own, nk[].k__label_kind_, kind[0])
    py_setitem(own, PythonObject("_mojolearn_fitted"), PythonObject(True))


def _learner_pack(model: PythonObject, context: PythonObject, plan: MemberPlan, state: Int) raises:
    """The booster's fitted attributes of state `state` into `model`."""
    var run = trees_route_context(model, context)
    var nk = trees_keys()
    var own = py_fields(model)
    # A refit drops the previous model's device copy (its holder releases it).
    _ = py_call(own.pop, "_resident", PythonObject(None))
    var call = MemberCall(plan.route, state)
    call.block = MEMBER_BLOCK_SCALARS
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var s = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
    var features = Int(s[0])
    var dim = Int(s[1])
    var classes = Int(s[2])
    var bias = s[3]
    var best = Int(s[4])
    var stopped = s[5] != 0.0
    var rate = s[6]
    var nbytes = Int(s[7])
    var ordered = s[8] != 0.0
    call.block = MEMBER_BLOCK_FAMILY
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var text = gbdt_string_at(call.dst.data, nbytes)
    call.block = MEMBER_BLOCK_FAMILY + 1
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var count = call.dst.rows * call.dst.cols
    var learn = List[Float64](length=count, fill=0.0)
    if count > 0:
        unsafe_memcpy(dest=learn.unsafe_ptr(), src=MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data),
                      count=count)
    # The category map (a model fitted with cat_features): one int64 Array of
    # its stored words, as the fit door's `gbdt_category_fit` returns it.
    call.block = MEMBER_BLOCK_FAMILY + 2
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var map_words = call.dst.rows * call.dst.cols
    var category_map = PythonObject(None)
    if map_words > 0:
        category_map = admitted_vector(map_words, "<i8", run)
        unsafe_memcpy(dest=MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(category_map))),
                      src=MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=call.dst.data), count=map_words)
    py_setitem(own, nk[].k_model_, PythonObject(text))
    py_setitem(own, nk[].k_bias_, PythonObject(bias))
    py_setitem(own, nk[].k_best_iteration_, PythonObject(best))
    py_setitem(own, nk[].k_stopped_early_, PythonObject(stopped))
    py_setitem(own, nk[].k_loss_curve_, _f64_curve(learn, run))
    py_setitem(own, nk[].k_test_loss_curve_, PythonObject(None))
    py_setitem(own, nk[].k_n_features_in_, PythonObject(features))
    py_setitem(own, nk[].k_category_map_, category_map)
    py_setitem(own, nk[].k_approx_dim_, PythonObject(dim))
    py_setitem(own, nk[].k_n_classes_, PythonObject(classes) if classes > 0 else PythonObject(None))
    py_setitem(own, nk[].k_learning_rate_, PythonObject(rate))
    py_setitem(own, nk[].k_boosting_type_, PythonObject("Ordered" if ordered else "Plain"))


def gbdt_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted attributes of `model`
    (an adapter's: of its `_learner_`). The caller frees it (`member_free`)."""
    var nk = trees_keys()
    var own = py_fields(model)
    if plan.kind == 0 or plan.kind == 2:
        var learner = py_get(own, nk[].k__learner_)
        if learner is None:
            raise Error("NotFittedError: this member is not fitted yet")
        own = py_fields(learner)
    var source = py_get(own, nk[].k_model_)
    if source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var category_map = py_get(own, nk[].k_category_map_)
    var map_view = PythonObject(None)
    if category_map is not None:
        map_view = classical_view(category_map, context, "category_map_")
    var text = String(py=source)
    var nbytes = text.byte_length()
    var packed = List[Int32](length=max((nbytes + 3) // 4, 1), fill=0)
    unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(packed.unsafe_ptr())),
                  src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(text.unsafe_ptr())), count=nbytes)
    var curve = py_get(own, nk[].k_loss_curve_)
    var curve_view = classical_view(curve, context, "gbdt loss curve")
    var classes_source = py_get(own, nk[].k_n_classes_)
    var boosting = py_get(own, nk[].k_boosting_type_)
    var scalars: List[Float64] = [
        Float64(py=py_item(own, nk[].k_n_features_in_)), Float64(py=py_item(own, nk[].k_approx_dim_)),
        0.0 if classes_source is None else Float64(py=classes_source), Float64(py=py_item(own, nk[].k_bias_)),
        Float64(py=py_item(own, nk[].k_best_iteration_)), 1.0 if Bool(py=py_item(own, nk[].k_stopped_early_)) else 0.0,
        Float64(py=py_item(own, nk[].k_learning_rate_)), Float64(nbytes),
        1.0 if boosting is not None and Bool(py=boosting == PythonObject("Ordered")) else 0.0]
    var call = member_new(plan)
    try:
        call.block = MEMBER_BLOCK_FAMILY
        call.x = MemberMatrix(Int(packed.unsafe_ptr()), 1, len(packed), MEMBER_I32)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_FAMILY + 1
        call.x = MemberMatrix(Int(py=classical_address(curve_view)), 1, Int(py=curve_view.size), MEMBER_F64)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        if map_view is not None:
            call.block = MEMBER_BLOCK_FAMILY + 2
            call.x = MemberMatrix(Int(py=classical_address(map_view)), 1, Int(py=map_view.size), MEMBER_I64)
            _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, len(scalars), MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = curve_view
    _ = map_view
    _ = len(packed)
    _ = len(scalars)
    _ = text
    return call.state
