# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the Logistic family: LogisticRegression is kind 0
(`_mojolearn_member = (MEMBER_FAMILY_LOGISTIC, 0)` on the class); RUN is
core/logistic_member.mojo.

  * `logistic_member_admit` (UNPACK, once per member): the estimator's own
    route (the owner its own doors resolve and hold: a hit is one record read)
    and the ten parameter words, read with the estimator's OWN checks (penalty,
    C, l1_ratio, tol and their promotion reports; max_iter,
    linesearch_max_iter; the class constants lbfgs_memory and
    penalty_normalized), then the class-weight tail (lane mf-adm3,
    core/member_class_weight.mojo): 'balanced', or a dict's PENDING table
    the caller fills over its labels (bindings/member_class_weight_unpack.mojo
    `member_cw_fill`). A value the estimator's fit refuses takes the INVALID
    tail (lane mf-left4): the FIT caller's UNPACK raises its words. A value
    the fit accepts in another form (a str subclass, a dict subclass) is
    not admitted (False: the estimator's own doors).
  * `logistic_member_pack` (PACK): `_w`, coef_, intercept_, n_iter_,
    objective_, retcode_ and n_features_in_ as the estimator's own fit writes
    them (bindings/logistic_state_boundary_unpack.mojo `logistic_fit_pack`),
    from the state's weight and scalars blocks. classes_ and
    feature_names_in_ are the caller's PACK (labels never cross the ABI).
  * `logistic_member_state` (UNPACK): a member state from a fitted estimator.

QNRegressor is kind 1 (`_mojolearn_member = (MEMBER_FAMILY_LOGISTIC, 1)`):
its admit / pack / state are providers/python/member_qn_regressor_boundary_door.mojo,
over the same held DeviceRoute its own doors read (`qn_route`).

SERIAL BY NATURE: one member's admission or publication (a few record reads
and one block copy); a caller runs its members in parallel.
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.admitted_input_unpack import admit_x, admitted_storage_native
from bindings.classical_arrays_door import classical_address
from bindings.classical_attributes_door import classical_connect_numpy, classical_getattribute
from bindings.classical_pandas_door import classical_admit_value
from bindings.classical_labels_door import labels_address
from bindings.logistic_admission_unpack import logistic_fit_replacements_admit
from bindings.logistic_config_boundary_unpack import (
    logistic_integer, logistic_validate_configuration, logistic_configuration_strengths,
    logistic_configuration_tolerance,
)
from bindings.logistic_keys_unpack import logistic_keys
from bindings.logistic_state_boundary_unpack import logistic_fit_pack
from bindings.member_owner_unpack import MemberOwner
from bindings.member_class_weight_unpack import member_cw_kind, member_cw_invalid_message, member_cw_kind_accepted
from core.member_families import MEMBER_FAMILY_LOGISTIC
from core.member_class_weight import (
    MEMBER_CW_BALANCED, MEMBER_CW_DICT, MEMBER_CW_INVALID, member_cw_balanced_words, member_cw_pending_words,
    member_cw_invalid_words,
)
from bindings.prep_lda_admission_door import lda_native_context
from bindings.py_balanced import py_get, py_item, py_type, py_setitem
from core.hostptr import f32_ptr
from core.logistic_config import LogisticPromotion
from core.logistic_model_state import LogisticFitLayout, logistic_fit_target_count
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_IMPORT_DONE, MEMBER_F64,
)
from providers.python.device_route_boundary_door import device_route_value
from providers.python.member_qn_regressor_boundary_door import (
    qn_regressor_member_admit, qn_regressor_member_pack, qn_regressor_member_state,
)
from providers.python.logistic_route_boundary_door import (
    logistic_route_held, logistic_route_context, logistic_route_current, logistic_route_resolve,
)


def _class_constant(model: PythonObject, fields: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`self.<name>` for LogisticRegression's class constants: the
    instance's own value, else the class attribute (the estimator's door's
    rule, bindings/logistic_native_door.mojo `_class_constant`)."""
    var own = py_get(fields, PythonObject(name))
    if own is not None:
        return own
    var value = classical_getattribute(model, PythonObject(name), context)
    classical_admit_value(value, context)
    return value


def logistic_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator. False for a class_weight the member does
    not read that the estimator's fit accepts (module note)."""
    if kind == 1:
        return qn_regressor_member_admit(model, fields, context, b, owner)
    if kind != 0:
        raise Error("RuntimeError: mojolearn: unknown Logistic member kind " + String(kind))
    var nk = logistic_keys()
    var weighting = member_cw_kind_accepted(MEMBER_FAMILY_LOGISTIC, model, py_item(fields, nk[].k_class_weight), b)
    # A value the estimator's fit door refuses (lane mf-left4): the INVALID
    # tail, whose words the FIT caller's UNPACK raises (`member_cw_fill_plan`);
    # a value the door accepts but the member does not read: not admitted.
    if weighting < 0:
        if member_cw_invalid_message(MEMBER_FAMILY_LOGISTIC, model, py_item(fields, nk[].k_class_weight), b).byte_length() == 0:
            return False
        weighting = MEMBER_CW_INVALID
    # The estimator's route, as its own door finds it: the held owner names
    # the context its class runs in; without one the class hierarchy is
    # walked against the explicit host classes, then the route is resolved.
    var base = context
    var scope = context
    var held = logistic_route_held(model, base)
    if held is not None:
        scope = logistic_route_context(held)
    else:
        var hosts = py_get(context, nk[].k_lr_host_contexts)
        if hosts is not None and len(hosts) > 0:
            var mro = py_type(model).__mro__
            for k in range(len(mro)):  # driver-loop: over the model's class hierarchy (explicit host classes)
                var host = py_get(hosts, mro[k])
                if host is not None:
                    scope = host
                    break
    var route_owner = held
    if not logistic_route_current(held, fields, scope):
        route_owner = logistic_route_resolve(model, fields, base, scope, b)
    classical_connect_numpy(scope)
    var checks = lda_native_context(scope)
    # The estimator's own parameter checks, in its fit's order.
    var penalty = logistic_validate_configuration(fields)
    var promotion = LogisticPromotion()
    var strengths = logistic_configuration_strengths(fields, penalty, promotion, checks)
    var intercept = Bool(py=py_item(fields, nk[].k_fit_intercept))
    var max_iter = logistic_integer(py_item(fields, nk[].k_max_iter))
    var linesearch = logistic_integer(py_item(fields, nk[].k_linesearch_max_iter))
    var memory = logistic_integer(_class_constant(model, fields, "lbfgs_memory", checks))
    var normalized = Bool(py=_class_constant(model, fields, "penalty_normalized", checks))
    var tolerances = logistic_configuration_tolerance(py_item(fields, nk[].k_tol), promotion, checks)
    owner.plan.route = device_route_value(route_owner)
    var words: List[Float64] = [0.0, 1.0 if intercept else 0.0, strengths[0], strengths[1], Float64(max_iter),
        Float64(linesearch), Float64(memory), 1.0 if normalized else 0.0, tolerances[0], tolerances[1]]
    if weighting == MEMBER_CW_BALANCED:
        member_cw_balanced_words(words)
    elif weighting == MEMBER_CW_DICT:
        member_cw_pending_words(words)
    elif weighting == MEMBER_CW_INVALID:
        member_cw_invalid_words(words)
    owner.plan.params = words^
    owner.pins = Python.tuple(route_owner, checks)
    return True


def logistic_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`, as
    the estimator's own fit does (the same names and buffers its own
    decision_function / predict / predict_proba / score read)."""
    if plan.params[0] == 1.0:
        qn_regressor_member_pack(model, fields, context, b, plan, state)
        return
    var call = MemberCall(plan.route, state)
    call.block = MEMBER_BLOCK_SCALARS
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var scalars = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
    var iterations = Int(scalars[0])
    var objective = Float32(scalars[1])
    var retcode = Float32(scalars[2])
    var features = Int(scalars[3])
    var classes = Int(scalars[4])
    call.block = MEMBER_BLOCK_FAMILY
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var count = call.dst.cols
    var layout = LogisticFitLayout(features, logistic_fit_target_count(classes), Bool(plan.params[1] != 0.0))
    if layout.weights != count:
        raise Error("RuntimeError: mojolearn: the logistic member's weight block does not match its layout")
    logistic_fit_replacements_admit(fields, context)
    var w = admitted_storage_native([count], "<f4", context, as_array=True)
    unsafe_memcpy(dest=f32_ptr(labels_address(w)), src=f32_ptr(call.dst.data), count=count)
    # The member was fitted on the caller's own rows, never on the user's array.
    py_setitem(fields, "input_copied_", True)
    logistic_fit_pack(fields, w, layout, iterations, objective, retcode, context)
    py_setitem(fields, "_mojolearn_fitted", True)
    _ = w


def logistic_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted attributes of `model`.
    The caller frees it (`member_free`)."""
    if plan.params[0] == 1.0:
        return qn_regressor_member_state(model, fields, context, b, plan)
    var weight_source = py_get(fields, "_w")
    var classes_source = py_get(fields, "classes_")
    if weight_source is None or classes_source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var weights = admit_x(weight_source, 1, "_w", context)
    var count = Int(py=weights.size)
    var iterations = Float64(0.0)
    var n_iter = py_get(fields, "n_iter_")
    if n_iter is not None:
        iterations = Float64(Int(py=n_iter[0]))
    var objective = Float64(0.0)
    var objective_source = py_get(fields, "objective_")
    if objective_source is not None:
        objective = Float64(py=objective_source)
    var retcode = Float64(0.0)
    var retcode_source = py_get(fields, "retcode_")
    if retcode_source is not None:
        retcode = Float64(Int(py=retcode_source))
    var scalars: List[Float64] = [iterations, objective, retcode, Float64(Int(py=fields["n_features_in_"])),
        Float64(len(classes_source))]
    var call = member_new(plan)
    try:
        call.block = MEMBER_BLOCK_FAMILY
        call.x = MemberMatrix(Int(py=classical_address(weights)), 1, count)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, 5, MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = weights
    _ = len(scalars)
    return call.state
