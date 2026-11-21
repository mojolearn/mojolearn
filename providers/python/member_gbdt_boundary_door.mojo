# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the Gbdt family (16): UNPACK / PACK of
core/gbdt_member.mojo, called by `bindings/member_boundary_door.mojo`.

Published class: `GradientBoosting` (kind 1, the booster: its own predict
returns raw scores). Not published (no mark, their own doors serve them):
`OrderedRMSE` (its fit is the text-only XG_ORDERED_FIT, not the typed fit
request) and the adapters `GradientBoostingClassifier` /
`GradientBoostingRegressor` (their fit encodes labels and builds an inner
`_learner_`; their PACK is not written yet).

  * `gbdt_member_admit` (UNPACK, once per member): the fit's own option
    checks (`gbdt_check_option_tables`) and the typed config
    (`gbdt_config`, with the Ordered checks' texts) read once from the
    unfitted estimator, as the fit door reads them; the route of the gbdt
    image its fit would run in (`gbdt_table` of its run context: mode,
    vendor, entry). Returns False (the caller keeps the estimator's doors)
    for what the typed member does not carry: cat_features /
    one_hot_features (the category map and CTR tables), the multi-target
    losses (y of n_targets columns), the query-wise losses (group_id /
    pairs) and any class but GradientBoosting itself. An eval set, groups
    and pairs are fit arguments a member call cannot pass, so a caller with
    them keeps the doors by construction.
  * `gbdt_member_pack` (PACK): the fitted attributes the fit door's PACK
    writes (`model_`, `bias_`, `best_iteration_`, `stopped_early_`,
    `loss_curve_`, `test_loss_curve_` None, `n_features_in_`,
    `category_map_` None, `approx_dim_`, `n_classes_`, `learning_rate_`,
    `boosting_type_`), the old resident copy dropped as the fit drops it.
  * `gbdt_member_state` (UNPACK): a member state from a fitted estimator's
    model text, loss curve and scalars. A CTR model (fitted with
    cat_features) is refused by the member's IMPORT; the caller then keeps
    the doors.

SERIAL BY NATURE: one member's admission or publication (a few record reads
and three block copies); a caller runs its members in parallel.
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.member_owner_unpack import MemberOwner
from bindings.py_balanced import py_call, py_get, py_item, py_setitem, py_fields, py_type
from bindings.classical_arrays_door import classical_address
from bindings.admitted_input_unpack import classical_view
from bindings.trees_keys_unpack import trees_keys
from bindings.trees_route_boundary_door import trees_route_context
from bindings.trees_public_gbdt_door import gbdt_table, gbdt_entry_of, _field
from bindings.trees_public_gbdt_fit_door import (
    gbdt_config, gbdt_check_option_tables, _tables, _truthy, _in, _f64_curve,
)
from bindings.gbdt_config_run import gbdt_config_words
from gbdt.fit_request import gbdt_string_at
from core.entry_table import entry_ref_mode, entry_ref_vendor
from core.device_route import DeviceRoute, DEVICE_CPU, DEVICE_GPU
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_SCALARS, MEMBER_IMPORT_DONE,
    MEMBER_F64, MEMBER_I32,
)


def gbdt_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator (see the module docstring)."""
    if kind != 1 or String(py=py_type(model).__name__) != "GradientBoosting":
        return False
    var run = trees_route_context(model, context)
    var own = py_fields(model)
    var nk = trees_keys()
    if (_truthy(_field(own, model, nk[].k_cat_features, "cat_features"))
            or _truthy(_field(own, model, nk[].k_one_hot_features, "one_hot_features"))):
        return False
    var t = _tables(run)
    var loss = _field(own, model, nk[].k_loss, "loss")
    if _in(loss, py_item(t, nk[].k_multi_regression_losses)) or _in(loss, py_item(t, nk[].k_querywise_losses)):
        return False
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
    var words: List[Float64] = [1.0]
    words.extend(gbdt_config_words(config))
    owner.plan.params = words^
    owner.pins = Python.tuple(table, run)
    return True


def gbdt_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`, as
    the fit door's PACK writes them."""
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
    py_setitem(own, nk[].k_model_, PythonObject(text))
    py_setitem(own, nk[].k_bias_, PythonObject(bias))
    py_setitem(own, nk[].k_best_iteration_, PythonObject(best))
    py_setitem(own, nk[].k_stopped_early_, PythonObject(stopped))
    py_setitem(own, nk[].k_loss_curve_, _f64_curve(learn, run))
    py_setitem(own, nk[].k_test_loss_curve_, PythonObject(None))
    py_setitem(own, nk[].k_n_features_in_, PythonObject(features))
    py_setitem(own, nk[].k_category_map_, PythonObject(None))
    py_setitem(own, nk[].k_approx_dim_, PythonObject(dim))
    py_setitem(own, nk[].k_n_classes_, PythonObject(classes) if classes > 0 else PythonObject(None))
    py_setitem(own, nk[].k_learning_rate_, PythonObject(rate))
    py_setitem(own, nk[].k_boosting_type_, PythonObject("Ordered" if ordered else "Plain"))


def gbdt_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted attributes of `model`.
    The caller frees it (`member_free`)."""
    var nk = trees_keys()
    var own = py_fields(model)
    var source = py_get(own, nk[].k_model_)
    if source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    if py_get(own, nk[].k_category_map_) is not None:
        raise Error("NotImplementedError: mojolearn: a GradientBoosting fitted with cat_features keeps its own doors")
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
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, len(scalars), MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = curve_view
    _ = len(packed)
    _ = len(scalars)
    _ = text
    return call.state
