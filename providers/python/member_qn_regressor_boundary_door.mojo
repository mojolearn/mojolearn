# SPDX-License-Identifier: Apache-2.0
"""The member boundary of QNRegressor, kind 1 of the Logistic family
(providers/python/member_logistic_boundary_door.mojo hands kind 1 here);
RUN is core/qn_regressor_member.mojo.

  * `qn_regressor_member_admit` (UNPACK, once per member): the estimator's
    own device route (bindings/qn_boundary_unpack.mojo `qn_route`: the SAME
    held DeviceRouteOwner its own fit and predict read, no second resolver;
    a hit is one record read), the x_metrics program entry of the route's
    tier and device in the fifth slot when that image is loaded, and the 11
    parameter words with the own fit's checks (the loss code, the change
    tolerance).
  * `qn_regressor_member_pack` (PACK): the fields `qn_fit_pack` writes, in
    its order; the member borrowed X, so `input_copied_` is False.
  * `qn_regressor_member_state` (UNPACK): a member state from a fitted
    QNRegressor (`_w`, `n_iter_`, `objective_`, `retcode_`).

SERIAL BY NATURE: one member's admission or publication.
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.admitted_input_unpack import admit_x
from bindings.classical_arrays_door import classical_address, classical_typed
from bindings.member_owner_unpack import MemberOwner
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_get, py_item, py_setitem
from bindings.qn_boundary_unpack import qn_route
from bindings.qn_keys_unpack import qn_keys
from core.entry_table import EntryRef, entry_ref_word
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_COEF, MEMBER_BLOCK_SCALARS, MEMBER_IMPORT_DONE, MEMBER_F64,
)
from providers.python.metrics_route_boundary_door import member_metrics_program_fifth
from core.qn_regressor_member import QN_REGRESSOR_MEMBER_KIND, QN_REGRESSOR_BLOCK_W
from glm.qn_door_run import qn_change_tolerance

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]


def _int(value: PythonObject) raises -> Int:
    return Int(py=py_call(native_loaded_module("builtins").int, value))


def _float(value: PythonObject) raises -> Float64:
    return Float64(py=py_call(native_loaded_module("builtins").float, value))


def qn_regressor_member_admit(model: PythonObject, fields: PythonObject, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted QNRegressor."""
    var nk = qn_keys()
    var call = qn_route(model, fields, context)
    var tol = _float(py_item(fields, nk[].k_tol))
    var delta = py_item(fields, nk[].k_delta)
    var delta_value = Optional[Float64](None)
    if delta is not None:
        delta_value = _float(delta)
    var change_tol = qn_change_tolerance(delta_value, tol)
    var loss = py_get(py_item(context, nk[].k_qn_losses), py_item(fields, nk[].k_loss))
    if loss is None:
        raise Error("mojolearn QNRegressor: unknown loss " + String(py_call(b.repr, py_item(fields, nk[].k_loss))))
    var route = call.route
    if not route.has_fifth:
        # The x_metrics program entry of the route's tier on its device (the
        # program QNRegressor's own score runs), when that image is loaded;
        # without it score is not offered.
        member_metrics_program_fifth(route, route.mode, route.vendor)
    var words: List[Float64] = [Float64(QN_REGRESSOR_MEMBER_KIND),
        1.0 if Bool(py=py_item(fields, nk[].k_fit_intercept)) else 0.0,
        Float64(_int(py_item(fields, nk[].k_max_iter))), Float64(_int(py_item(fields, nk[].k_linesearch_max_iter))),
        Float64(_int(py_item(fields, nk[].k_lbfgs_memory))),
        1.0 if Bool(py=py_item(fields, nk[].k_penalty_normalized)) else 0.0, Float64(_int(loss)),
        _float(py_item(fields, nk[].k_l1_strength)), _float(py_item(fields, nk[].k_l2_strength)), tol, change_tol]
    owner.plan.route = route
    owner.plan.params = words^
    owner.pins = Python.tuple(call.owner)
    return True


def qn_regressor_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: `input_copied_`, `_w`, `coef_`, `intercept_`, `n_iter_`,
    `objective_`, `retcode_`, `n_features_in_`, as `qn_fit_pack`."""
    var nk = qn_keys()
    var output = py_item(context, nk[].k_output_context)
    var intercept = plan.params[1] != 0.0
    var call = MemberCall(plan.route, state)
    call.block = QN_REGRESSOR_BLOCK_W
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var width = call.dst.cols
    var cols = width - (1 if intercept else 0)
    var w = classical_typed(Python.tuple(width), "<f4", output)
    if width > 0:
        unsafe_memcpy(dest=_F32P(unsafe_from_address=Int(py=classical_address(w))),
            src=_F32P(unsafe_from_address=call.dst.data), count=width)
    var bias = Float64(_F32P(unsafe_from_address=call.dst.data)[cols]) if intercept else 0.0
    var coef = classical_typed(Python.tuple(cols), "<f4", output)
    if cols > 0:
        unsafe_memcpy(dest=_F32P(unsafe_from_address=Int(py=classical_address(coef))),
            src=_F32P(unsafe_from_address=call.dst.data), count=cols)
    call.block = MEMBER_BLOCK_SCALARS
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var scalars = _F64P(unsafe_from_address=call.dst.data)
    var iterations = classical_typed(Python.tuple(1), "<i8", output)
    _I64P(unsafe_from_address=Int(py=classical_address(iterations)))[0] = Int64(Int(scalars[0]))
    py_setitem(fields, nk[].k_input_copied_, PythonObject(False))
    py_setitem(fields, nk[].k__w, w)
    py_setitem(fields, nk[].k_coef_, coef)
    py_setitem(fields, nk[].k_intercept_, PythonObject(bias))
    py_setitem(fields, nk[].k_n_iter_, iterations)
    py_setitem(fields, nk[].k_objective_, PythonObject(scalars[1]))
    py_setitem(fields, nk[].k_retcode_, PythonObject(Int(scalars[2])))
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(cols))


def qn_regressor_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding `_w`, `n_iter_`, `objective_` and
    `retcode_`. The caller frees it (`member_free`)."""
    var nk = qn_keys()
    var source = py_get(fields, nk[].k__w)
    if source is None:
        raise Error("mojolearn QNRegressor: call fit before predict")
    var w = admit_x(source, 1, "_w", context)
    var count = Int(py=w.size)
    var scalars: List[Float64] = [Float64(_int(py_item(py_item(fields, nk[].k_n_iter_), PythonObject(0)))),
        _float(py_item(fields, nk[].k_objective_)), Float64(_int(py_item(fields, nk[].k_retcode_)))]
    var call = member_new(plan)
    try:
        call.block = QN_REGRESSOR_BLOCK_W
        call.x = MemberMatrix(Int(py=classical_address(w)), 1, count)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, 3, MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = w
    _ = len(scalars)
    return call.state
