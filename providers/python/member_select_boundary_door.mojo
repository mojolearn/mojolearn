# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the Select family: its place in
`bindings/member_boundary_door.mojo` (recipe: MEMBER-ABI-V2.md; RUN:
core/select_member.mojo). Published: VarianceThreshold (kind 0,
`_mojolearn_member = (36, 0)`), SelectKBest (kind 1, `(36, 1)`: f_classif,
chi2, f_regression, r_regression; admit answers False for the
mutual-information scores, which re-enter the encode owner's Python-called
entry, and for a negative k, whose words only the door's fit knows). RFE
carries no mark (see the lane's RESULT file): a caller keeps its door.

  * `select_member_admit` (UNPACK, once per member): the estimator's own
    stateless route (`stateless_entries`, `stateless_caps`: the x_prep
    transport entry and its capability bits, through the Prep family's
    `_route_admit`), the process's default mode (the door's fit runs on it),
    the threshold and the switches the door reads per fit (read once here).
  * `select_member_pack` (PACK): the fitted attributes of the door's fit
    (bindings/prep_stateless_native_door.mojo `_variance_fit`).
  * `select_member_state` (UNPACK): a member state from a fitted estimator.

SERIAL BY NATURE: one member's admission or publication (a few record reads,
the support list over the features); a caller runs its members in parallel.
"""
from std.python import Python, PythonObject
from bindings.member_owner_unpack import MemberOwner
from bindings.classical_arrays_door import classical_address
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_setitem, py_contains, py_get
from bindings.prep_stateless_route_unpack import stateless_entries, stateless_caps
from bindings.prep_stateless_native_door import _support_columns as select_support_columns
from bindings.prep_entry_cache_unpack import prep_entry_word
from bindings.x_prep_native_owner_run import xprep_r3
from bindings.p3_prep_fitted_unpack import prep_fitted_vector
from bindings.py_typed_call import py_call_typed
from bindings.scaler_lifecycle_native_door import scaler_entries
from bindings.prep_encode_native_door import enc_caps, _is_str as enc_is_str
from bindings.prep_stateless_native_door import _fam as stateless_fam
from providers.python.member_prep_boundary_door import _route_admit as prep_route_admit, _vector as prep_vector
from core.select_member import (
    SELECT_KIND_VARIANCE, SELECT_VARIANCE_PARAM_COUNT, SELECT_BLOCK_VARIANCES, SELECT_BLOCK_SUPPORT,
    SELECT_KIND_KBEST, SELECT_KBEST_PARAM_COUNT, SELECT_BLOCK_SCORES, SELECT_BLOCK_PVALUES,
    SELECT_SCORE_F_CLASSIF, SELECT_SCORE_CHI2, SELECT_SCORE_F_REGRESSION, SELECT_SCORE_R_REGRESSION,
)
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_SCALARS, MEMBER_IMPORT_DONE, MEMBER_F64,
)


def select_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator."""
    if kind == SELECT_KIND_KBEST:
        return _kbest_admit(model, fields, context, owner)
    if kind != SELECT_KIND_VARIANCE:
        raise Error("ValueError: mojolearn: Select member kind " + String(kind) + " is not published")
    var mode = context["backend_fields"]["_DEFAULT_MODE"]
    var entries = stateless_entries(model, mode, context)
    var words = List[Float64](length=SELECT_VARIANCE_PARAM_COUNT, fill=0.0)
    words[0] = Float64(SELECT_KIND_VARIANCE)
    # The door's float(threshold) (`_f64_input`), read once (the door per fit).
    words[1] = Float64(py=py_call(native_loaded_module("builtins").float, fields["threshold"]))
    # `_fam`: the identical binding's family switches, 0 on another tier.
    words[2] = Float64(prep_entry_word(entries, "x_prep_idn_fam")) if String(mode) == "identical" else 0.0
    words[3] = 1.0 if xprep_r3("work") else 0.0
    prep_route_admit(model, entries, context, owner, stateless_caps(model, entries), False)
    owner.plan.params = words^
    return True


def _kbest_admit(model: PythonObject, fields: PythonObject, context: PythonObject, mut owner: MemberOwner) raises -> Bool:
    # The encode door's `_select_fit` UNPACK, read once here (the door per fit).
    var b = native_loaded_module("builtins")
    var builtin = py_get(context["prep_encode_scores"], fields["score_func"])
    if builtin is None:
        raise Error("TypeError: mojolearn: SelectKBest score_func must be f_classif, chi2, f_regression, "
                    + "r_regression, mutual_info_classif or mutual_info_regression; a user callable is not supported")
    var name = String(builtin)
    var score = SELECT_SCORE_F_CLASSIF
    if name == "chi2":
        score = SELECT_SCORE_CHI2
    elif name == "f_regression":
        score = SELECT_SCORE_F_REGRESSION
    elif name == "r_regression":
        score = SELECT_SCORE_R_REGRESSION
    elif name != "f_classif":
        return False  # mutual_info_*: the encode owner's Python-called entry
    var k_arg = fields["k"]
    var k = -1
    if not (enc_is_str(k_arg) and String(k_arg) == "all"):
        k = Int(py=py_call_typed(b.int, k_arg))
        if k < 0:
            return False  # the door's fit raises with the feature count
    var mode = context["backend_fields"]["_DEFAULT_MODE"]
    var entries = scaler_entries(model, mode, context)
    var words = List[Float64](length=SELECT_KBEST_PARAM_COUNT, fill=0.0)
    words[0] = Float64(SELECT_KIND_KBEST)
    words[1] = Float64(score)
    words[2] = Float64(k)
    words[3] = Float64(stateless_fam(mode, entries))
    words[4] = 1.0 if xprep_r3("work") else 0.0
    prep_route_admit(model, entries, context, owner, enc_caps(model, entries), False)
    owner.plan.params = words^
    return True


def _export(plan: MemberPlan, mut call: MemberCall, block: Int) raises -> MemberMatrix:
    call.block = block
    call.flags = 0
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    return call.dst


def select_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`, as
    the door's fit does."""
    if plan.kind != SELECT_KIND_VARIANCE and plan.kind != SELECT_KIND_KBEST:
        raise Error("ValueError: mojolearn: Select member kind " + String(plan.kind) + " is not published")
    var call = MemberCall(plan.route, state)
    var scalars = _export(plan, call, MEMBER_BLOCK_SCALARS)
    var d = Int(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=scalars.data)[])
    if plan.kind == SELECT_KIND_KBEST:
        # The door's scores_ / pvalues_ (None for r_regression), then the mask.
        py_setitem(fields, "scores_", prep_vector(_export(plan, call, SELECT_BLOCK_SCORES), context))
        var pvalues = _export(plan, call, SELECT_BLOCK_PVALUES)
        py_setitem(fields, "pvalues_", prep_vector(pvalues, context) if pvalues.cols > 0 else PythonObject(None))
    else:
        py_setitem(fields, "variances_", prep_vector(_export(plan, call, SELECT_BLOCK_VARIANCES), context))
    var support = _export(plan, call, SELECT_BLOCK_SUPPORT)
    var words = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=support.data)
    var mask = Python.list()
    for j in range(d):  # driver-loop: over the fitted support mask (feature metadata)
        _ = py_call(mask.append, PythonObject(words[j] != 0))
    py_setitem(fields, "_mask", mask)
    py_setitem(fields, "numeric_mode_", context["backend_fields"]["_DEFAULT_MODE"])
    py_setitem(fields, "n_features_in_", d)


def select_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted attributes of `model`.
    The caller frees it (`member_free`)."""
    var kbest = plan.kind == SELECT_KIND_KBEST
    if plan.kind != SELECT_KIND_VARIANCE and not kbest:
        raise Error("ValueError: mojolearn: Select member kind " + String(plan.kind) + " is not published")
    var estimator = String("SelectKBest") if kbest else String("VarianceThreshold")
    if not py_contains(fields, "n_features_in_") or not py_contains(fields, "_mask"):
        raise Error("NotFittedError: this member is not fitted yet")
    if py_type(fields["n_features_in_"]) is not native_loaded_module("builtins").int:
        raise Error("TypeError: " + estimator + " n_features_in_ must be a native integer")
    var d = Int(py=fields["n_features_in_"])
    var keep = select_support_columns(fields["_mask"])
    var support = List[Float32](length=d, fill=Float32(0))
    for i in range(len(keep)):  # driver-loop: over the kept columns (feature metadata)
        if keep[i] < 0 or keep[i] >= d:
            raise Error("ValueError: " + estimator + " support mask does not match n_features_in_")
        support[keep[i]] = Float32(1)
    var scalars: List[Float64] = [Float64(d)]
    var variances = PythonObject(None)
    var first = String("scores_") if kbest else String("variances_")
    if py_contains(fields, first) and fields[first] is not None:
        variances = prep_fitted_vector(fields[first], context)
        if Int(py=variances.size) != d:
            raise Error("ValueError: " + estimator + " " + first + " must hold one value per feature")
    var pvalues = PythonObject(None)
    if kbest and py_contains(fields, "pvalues_") and fields["pvalues_"] is not None:
        pvalues = prep_fitted_vector(fields["pvalues_"], context)
        if Int(py=pvalues.size) != d:
            raise Error("ValueError: SelectKBest pvalues_ must hold one value per feature")
    var call = member_new(plan)
    try:
        call.flags = 0
        call.block = SELECT_BLOCK_SUPPORT
        call.x = MemberMatrix(Int(support.unsafe_ptr()), 1, d)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        if variances is not None:
            call.block = SELECT_BLOCK_VARIANCES
            call.x = MemberMatrix(Int(py=classical_address(variances)), 1, d)
            _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        if pvalues is not None:
            call.block = SELECT_BLOCK_PVALUES
            call.x = MemberMatrix(Int(py=classical_address(pvalues)), 1, d)
            _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, 1, MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = len(scalars)
    _ = len(support)
    _ = variances
    _ = pvalues
    return call.state
