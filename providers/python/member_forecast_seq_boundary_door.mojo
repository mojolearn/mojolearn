# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the ForecastSeq family (family 43): the theta
forecasters (AutoTheta, Theta, OptimizedTheta, DynamicTheta,
DynamicOptimizedTheta), kind 0.

UNPACK / PACK only (GIL held, once per member estimator, never per fold):
  * `forecast_seq_member_admit`: the estimator's own setting checks
    (`_forecast_fields`: the exact connected class, every consumed setting
    admitted), its route through the sequence resolver (`xseq_route`: the
    glue entry of its device and mode), and the NEW words of
    core/forecast_seq_member.mojo (the model name mapped by the connected
    code table, as the estimator's own predict maps it).
  * `forecast_seq_member_pack`: the fitted attributes the estimator's own
    fit writes (`_one`: True for one series, `_y` the (series, observations)
    float32 copy, `_fitted_h` None). Its own predict then estimates the
    model as after its own fit.
  * `forecast_seq_member_state`: a state holding a fitted model's `_y`.
"""
from std.python import Python, PythonObject
from bindings.member_owner_unpack import MemberOwner
from bindings.py_balanced import py_get, py_item, py_setitem, py_type
from bindings.xseq_keys_unpack import xseq_keys
from bindings.xseq_native_common_door import xseq_route, xseq_zeros, xseq_addr, xseq_int
from bindings.xseq_forecast_native_door import _forecast_fields, _set, _float_or_zero, _stored_buffer, _KIND_THETA
from providers.python.device_route_boundary_door import DeviceRouteOwner
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_IMPORT_DONE, MEMBER_F64,
)
from sequence.xseq_forecast_run import forecast_run_theta_mask
from sequence.xseq_neural_run import xseq_run_copy_f32

comptime _MEMBER_THETA = 0


def _connected(model: PythonObject, records: PythonObject) raises -> Bool:
    var cls = py_type(model)
    var found = False
    for i in range(len(records)):  # driver-loop: over connected forecast classes
        found = found or cls is records[i][0]
    return found


def forecast_seq_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator. False: a kind that is not the class's."""
    var nk = xseq_keys()
    if kind != _MEMBER_THETA or not _connected(model, py_item(context, nk[].k_xseq_theta_classes)):
        return False
    var checked = _forecast_fields(model, _KIND_THETA, context)
    var held = xseq_route(model, checked, context)
    owner.plan.route = held.owner.downcast_value_ptr[DeviceRouteOwner]()[].route
    var chosen = py_get(checked, nk[].k_model)
    var code = -1 if chosen is None else Int(py=py_item(context, nk[].k_xseq_theta_codes)[chosen])
    var additive = not Bool(py=py_item(checked, nk[].k_decomposition_type) == PythonObject("multiplicative"))
    var initial = py_item(checked, nk[].k_initial_smoothed)
    var alpha = py_item(checked, nk[].k_alpha)
    var theta = py_item(checked, nk[].k_theta)
    var mask = forecast_run_theta_mask(_set(initial), _set(alpha), _set(theta))
    var words: List[Float64] = [
        Float64(_MEMBER_THETA), Float64(Int(py=py_item(checked, nk[].k_season_length))), Float64(code),
        1.0 if additive else 0.0, Float64(mask), _float_or_zero(initial), _float_or_zero(alpha),
        _float_or_zero(theta),
    ]
    owner.plan.params = words^
    owner.pins = Python.tuple(held.owner)
    return True


def forecast_seq_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`, as
    the estimator's own fit does."""
    var nk = xseq_keys()
    var call = MemberCall(plan.route, state)
    call.block = MEMBER_BLOCK_FAMILY
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var batch = call.dst.rows
    var n = call.dst.cols
    var y = xseq_zeros(Python.tuple(batch, n), context)
    xseq_run_copy_f32(Int(py=xseq_addr(y)), call.dst.data, batch * n)
    # The member's y is observations x series: one series predicts 1-D.
    py_setitem(fields, nk[].k__one, PythonObject(batch == 1))
    py_setitem(fields, nk[].k__y, y)
    py_setitem(fields, nk[].k__fitted_h, PythonObject(None))


def forecast_seq_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted `_y` of `model`. The
    caller frees it (`member_free`)."""
    var nk = xseq_keys()
    var source = py_get(fields, nk[].k__y)
    if source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var captured = _stored_buffer(source, "Theta observations", 2, context)
    var batch = captured.shape[0]
    var n = captured.shape[1]
    var scalars: List[Float64] = [Float64(plan.kind), Float64(batch), Float64(n), 0.0]
    var call = member_new(plan)
    try:
        call.block = MEMBER_BLOCK_FAMILY
        call.x = MemberMatrix(captured.address, batch, n)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, len(scalars), MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = captured^
    _ = len(scalars)
    return call.state
