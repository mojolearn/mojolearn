# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the LinearSvm family; RUN is
core/linear_svm_member.mojo. Published kinds: 0 LinearSVC, 1 LinearSVR
(`_mojolearn_member = (MEMBER_FAMILY_LINEAR_SVM, kind)` on each class).

  * `linear_svm_member_admit` (UNPACK, once per member): the estimator's own
    refusals (bindings/svc_linear_boundary_unpack.mojo
    `linear_svc_check_fields` / `linear_svr_check_fields`), its own device
    route (bindings/linear_svc_boundary_unpack.mojo `linear_svc_route`: the
    driver of its device; a hit is one record read), for LinearSVR the
    metrics image's R^2 entry of the route's tier and device in the fifth
    slot when that image is loaded, and the 11 parameter words
    (core/linear_svm_member.mojo docstring). Not admitted (False; the
    caller keeps the estimator's own doors): any LinearSVC class_weight (its
    rows come from the x_trees class-weight programs), and a binary whose
    driver lacks the native class-code queries (its own predict then
    decodes through the label steps).
  * `linear_svm_member_pack` (PACK): the fitted attributes the own fit
    writes (bindings/svc_linear_boundary_pack.mojo), in its order; the
    member borrowed X, so `input_copied_` is False. `classes_` is the
    caller's.
  * `linear_svm_member_state` (UNPACK): a member state from a fitted
    estimator (`_w`, `n_iter_`, `objective_`, `retcode_`).

SERIAL BY NATURE: one member's admission or publication; a caller runs its
members in parallel.
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.admitted_input_unpack import admit_x
from bindings.classical_arrays_door import classical_address
from bindings.family_common_unpack import FAMILY_F32P, FAMILY_I64P
from bindings.family_common_pack import family_array
from bindings.member_owner_unpack import MemberOwner
from bindings.py_balanced import py_call, py_get, py_setitem
from bindings.linear_svc_boundary_unpack import linear_svc_route, LINEAR_SVC_ROUTE_ROWS, LINEAR_SVC_ROUTE_MULTICLASS
from bindings.svc_linear_boundary_unpack import linear_svc_check_fields, linear_svr_check_fields, linear_svm_strengths
from bindings.svc_common_unpack import svc_field, svc_attr, svc_float, svc_int, svc_truthy
from bindings.svm_keys_unpack import svm_keys
from core.entry_table import EntryRef, entry_ref_word
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_COEF, MEMBER_BLOCK_INTERCEPT, MEMBER_BLOCK_SCALARS,
    MEMBER_IMPORT_DONE, MEMBER_F64,
)
from core.linear_svm_member import (
    LINEAR_SVM_MEMBER_SVC, LINEAR_SVM_MEMBER_SVR, LINEAR_SVM_BLOCK_INFO, LINEAR_SVM_BLOCK_W,
)
from core.metrics_entry_table import MetricsEntryKey, metrics_r2_entry, MR_OP_COUNT
from providers.python.device_route_boundary_door import device_route_value


def linear_svm_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator. False: this configuration has no member
    entry (a LinearSVC class_weight, a driver without the class-code queries)."""
    var nk = svm_keys()
    if kind != LINEAR_SVM_MEMBER_SVC and kind != LINEAR_SVM_MEMBER_SVR:
        raise Error("RuntimeError: mojolearn: unknown LinearSvm member kind " + String(kind))
    var epsilon = 0.0
    if kind == LINEAR_SVM_MEMBER_SVR:
        epsilon = svc_float(linear_svr_check_fields(model, fields))
    else:
        linear_svc_check_fields(model, fields)
        if py_get(fields, nk[].k_class_weight, PythonObject(None)) is not None:
            return False
    var connection = linear_svc_route(model, fields, context)
    if kind == LINEAR_SVM_MEMBER_SVC and (connection.flags & (LINEAR_SVC_ROUTE_ROWS | LINEAR_SVC_ROUTE_MULTICLASS)) != (
            LINEAR_SVC_ROUTE_ROWS | LINEAR_SVC_ROUTE_MULTICLASS):
        return False
    var route = device_route_value(connection.provider)
    if kind == LINEAR_SVM_MEMBER_SVR and not route.has_fifth:
        # The metrics image's R^2 entry of the route's tier on its device,
        # when that image is loaded.
        try:
            var entry = metrics_r2_entry(MetricsEntryKey(route.mode, route.vendor))
            route.fifth = EntryRef(entry, entry_ref_word(route.mode, route.vendor, MR_OP_COUNT))
            route.has_fifth = True
        except problem:
            # Not loaded: score is not offered; the caller runs its own metric.
            _ = String(problem)
    var strengths = linear_svm_strengths(fields)
    var loss = Int(py=svc_attr(model, nk[].k__LOSSES)[svc_field(fields, nk[].k_loss, "loss")])
    var words: List[Float64] = [Float64(kind),
        1.0 if svc_truthy(svc_field(fields, nk[].k_fit_intercept, "fit_intercept")) else 0.0,
        Float64(svc_int(svc_field(fields, nk[].k_max_iter, "max_iter"))),
        Float64(svc_int(svc_field(fields, nk[].k_linesearch_max_iter, "linesearch_max_iter"))),
        Float64(svc_int(svc_field(fields, nk[].k_lbfgs_memory, "lbfgs_memory"))),
        1.0 if svc_truthy(svc_attr(model, nk[].k_penalty_normalized)) else 0.0,
        Float64(loss), strengths[0], strengths[1], svc_float(svc_field(fields, nk[].k_tol, "tol")), epsilon]
    owner.plan.route = route
    owner.plan.params = words^
    owner.pins = Python.tuple(connection.provider)
    return True


def linear_svm_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: the own fit's fields. LinearSVC: `input_copied_`, `_w`, `coef_`
    (fits, d), `intercept_` (fits,), `n_iter_`, `objective_`, `retcode_`
    (scalars for one machine, else lists), `n_features_in_`. LinearSVR:
    `input_copied_`, `objective_`, `retcode_`, `_w`, `coef_` (d,),
    `intercept_` (a float), `n_iter_`, `n_features_in_`."""
    var nk = svm_keys()
    var svr = Int(plan.params[0]) == LINEAR_SVM_MEMBER_SVR
    var intercept = plan.params[1] != 0.0
    var call = MemberCall(plan.route, state)
    call.block = LINEAR_SVM_BLOCK_W
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var width = call.dst.rows
    var fits = call.dst.cols
    var cols = width - Int(intercept)
    var w = family_array(Python.tuple(PythonObject(width * fits)), "<f4", context)
    unsafe_memcpy(dest=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(w))),
        src=FAMILY_F32P(unsafe_from_address=call.dst.data), count=width * fits)
    call.block = MEMBER_BLOCK_COEF
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var coef = family_array(Python.tuple(PythonObject(cols)), "<f4", context) if svr else family_array(
        Python.tuple(PythonObject(fits), PythonObject(cols)), "<f4", context)
    unsafe_memcpy(dest=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(coef))),
        src=FAMILY_F32P(unsafe_from_address=call.dst.data), count=fits * cols)
    call.block = MEMBER_BLOCK_INTERCEPT
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var bias = FAMILY_F32P(unsafe_from_address=call.dst.data)
    var intercepts = PythonObject(None)
    if svr:
        intercepts = PythonObject(Float64(bias[0]) if intercept else 0.0)
    else:
        intercepts = family_array(Python.tuple(PythonObject(fits)), "<f4", context)
        unsafe_memcpy(dest=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(intercepts))), src=bias,
            count=fits)
    call.block = MEMBER_BLOCK_SCALARS
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var n_iter = family_array(Python.tuple(PythonObject(1)), "<i8", context)
    FAMILY_I64P(unsafe_from_address=Int(py=classical_address(n_iter)))[0] = Int64(
        Int(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)[0]))
    call.block = LINEAR_SVM_BLOCK_INFO
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var info = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
    var objectives = Python.list()
    var retcodes = Python.list()
    for fit in range(fits):  # boundary PACK: one public scalar per machine
        _ = py_call(objectives.append, PythonObject(info[2 * fit]))
        _ = py_call(retcodes.append, PythonObject(Int(info[2 * fit + 1])))
    py_setitem(fields, nk[].k_input_copied_, PythonObject(False))
    if svr:
        py_setitem(fields, nk[].k_objective_, objectives[0])
        py_setitem(fields, nk[].k_retcode_, retcodes[0])
        py_setitem(fields, nk[].k__w, w)
        py_setitem(fields, nk[].k_coef_, coef)
        py_setitem(fields, nk[].k_intercept_, intercepts)
        py_setitem(fields, nk[].k_n_iter_, n_iter)
    else:
        py_setitem(fields, nk[].k__w, w)
        py_setitem(fields, nk[].k_coef_, coef)
        py_setitem(fields, nk[].k_intercept_, intercepts)
        py_setitem(fields, nk[].k_n_iter_, n_iter)
        py_setitem(fields, nk[].k_objective_, objectives[0] if fits == 1 else objectives)
        py_setitem(fields, nk[].k_retcode_, retcodes[0] if fits == 1 else retcodes)
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(cols))


def linear_svm_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding `_w`, `n_iter_`, `objective_` and
    `retcode_` (and the class count of `classes_` for LinearSVC). The caller
    frees it (`member_free`)."""
    var nk = svm_keys()
    var svr = Int(plan.params[0]) == LINEAR_SVM_MEMBER_SVR
    var name = String("LinearSVR") if svr else String("LinearSVC")
    var source = py_get(fields, nk[].k__w)
    if source is None:
        raise Error("mojolearn " + name + ": call fit first")
    var classes = 0
    var fits = 1
    if not svr:
        classes = len(svc_field(fields, nk[].k_classes_, "classes_"))
        fits = 1 if classes == 2 else classes
    var w = admit_x(source, 1, "_w", context)
    var count = Int(py=w.size)
    if fits < 1 or count % fits != 0:
        raise Error("mojolearn " + name + ": the fitted _w does not hold one column per machine")
    var objective = svc_field(fields, nk[].k_objective_, "objective_")
    var retcode = svc_field(fields, nk[].k_retcode_, "retcode_")
    var info = List[Float64](length=2 * fits, fill=0.0)
    if fits == 1:
        info[0] = svc_float(objective)
        info[1] = Float64(svc_int(retcode))
    else:
        for fit in range(fits):  # boundary UNPACK: one public scalar per machine
            info[2 * fit] = svc_float(objective[fit])
            info[2 * fit + 1] = Float64(svc_int(retcode[fit]))
    var scalars: List[Float64] = [Float64(svc_int(svc_field(fields, nk[].k_n_iter_, "n_iter_")[0]))]
    var call = member_new(plan)
    try:
        call.block = LINEAR_SVM_BLOCK_W
        call.x = MemberMatrix(Int(py=classical_address(w)), count // fits, fits)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, 1, MEMBER_F64)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = LINEAR_SVM_BLOCK_INFO
        call.x = MemberMatrix(Int(info.unsafe_ptr()), fits, 2, MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        call.classes = classes
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = w
    _ = len(scalars)
    _ = len(info)
    return call.state
