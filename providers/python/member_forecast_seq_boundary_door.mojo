# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the ForecastSeq family (family 43): the theta
forecasters (AutoTheta, Theta, OptimizedTheta, DynamicTheta,
DynamicOptimizedTheta), kind 0, and ETS (ETS, DampedETS), kind 1.

UNPACK / PACK only (GIL held, once per member estimator, never per fold):
  * `forecast_seq_member_admit`: the estimator's own setting checks
    (`_forecast_fields`: the exact connected class, every consumed setting
    admitted), its route through the sequence resolver (`xseq_route`: the
    glue entry of its device and mode), and the NEW words of
    core/forecast_seq_member.mojo (the model name mapped by the connected
    code table, as the estimator's own predict maps it; the ETS spec as
    letter codes). NOT admitted (False: the caller keeps the estimator's own
    path): an ETS with the FAST stop (`_fast_stall`), and an ETS whose spec
    may need positive data (error or season M) on any route but a CPU route
    beside a GPU set: there the estimator's fit scans through the base
    binding's Python entry `all_finite_f32`
    (bindings/forecast_device_door.mojo `forecast_finite`), a Python door.
  * `forecast_seq_member_pack`: the fitted attributes the estimator's own
    fit writes (`_one`: True for one series, `_y` the (series, observations)
    float32 copy; theta `_fitted_h` None; ETS `_season` the kept season
    letter and `_last` None). Its own predict then estimates the model as
    after its own fit.
  * `forecast_seq_member_state`: a state holding a fitted model's `_y`.
"""
from std.python import Python, PythonObject
from bindings.member_owner_unpack import MemberOwner
from bindings.py_balanced import py_get, py_item, py_setitem, py_type
from bindings.xseq_keys_unpack import xseq_keys
from bindings.xseq_native_common_door import xseq_route, xseq_zeros, xseq_addr, xseq_int, xseq_truth
from bindings.xseq_forecast_native_door import (
    _forecast_fields, _set, _float_or_zero, _stored_buffer, _KIND_THETA, _KIND_ETS,
)
from providers.python.device_route_boundary_door import DeviceRouteOwner
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_IMPORT_DONE, MEMBER_F64,
)
from sequence.xseq_forecast_run import forecast_run_theta_mask, forecast_run_ets_mask, forecast_run_letter
from sequence.xseq_neural_run import xseq_run_copy_f32

comptime _MEMBER_THETA = 0
comptime _MEMBER_ETS = 1


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
    if kind == _MEMBER_ETS:
        return _ets_admit(model, context, owner)
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


def _ets_admit(model: PythonObject, context: PythonObject, mut owner: MemberOwner) raises -> Bool:
    var nk = xseq_keys()
    if not _connected(model, py_item(context, nk[].k_xseq_ets_classes)):
        return False
    var checked = _forecast_fields(model, _KIND_ETS, context)
    if py_get(checked, nk[].k__fast_stall) is not None:
        return False
    var spec = String(py=py_item(checked, nk[].k_model))
    var error = String("AM").find(forecast_run_letter(spec, 0))
    var trend = String("NA").find(forecast_run_letter(spec, 1))
    var season = String("NAM").find(forecast_run_letter(spec, 2))
    if error < 0 or trend < 0 or season < 0:
        return False
    var held = xseq_route(model, checked, context)
    if (error == 1 or season == 2) and not held.cpu_table:
        return False
    owner.plan.route = held.owner.downcast_value_ptr[DeviceRouteOwner]()[].route
    var alpha = py_item(checked, nk[].k_alpha)
    var beta = py_item(checked, nk[].k_beta)
    var phi = py_item(checked, nk[].k_phi)
    var gamma = py_item(checked, nk[].k_gamma)
    var mask = forecast_run_ets_mask(_set(alpha), _set(beta), _set(phi), _set(gamma))
    var words: List[Float64] = [
        Float64(_MEMBER_ETS), Float64(xseq_int(py_item(checked, nk[].k_season_length))), Float64(error),
        Float64(trend), Float64(season), 1.0 if xseq_truth(py_item(checked, nk[].k_damped)) else 0.0, Float64(mask),
        _float_or_zero(alpha), _float_or_zero(beta), _float_or_zero(phi), _float_or_zero(gamma),
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
    call.block = MEMBER_BLOCK_SCALARS
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var letter = Int(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)[3])
    call.block = MEMBER_BLOCK_FAMILY
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var batch = call.dst.rows
    var n = call.dst.cols
    var y = xseq_zeros(Python.tuple(batch, n), context)
    xseq_run_copy_f32(Int(py=xseq_addr(y)), call.dst.data, batch * n)
    # The member's y is observations x series: one series predicts 1-D.
    py_setitem(fields, nk[].k__one, PythonObject(batch == 1))
    py_setitem(fields, nk[].k__y, y)
    if plan.kind == _MEMBER_ETS:
        py_setitem(fields, nk[].k__season, PythonObject(forecast_run_letter("NAM", letter)))
        py_setitem(fields, nk[].k__last, PythonObject(None))
    else:
        py_setitem(fields, nk[].k__fitted_h, PythonObject(None))


def forecast_seq_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted `_y` of `model`. The
    caller frees it (`member_free`)."""
    var nk = xseq_keys()
    var source = py_get(fields, nk[].k__y)
    if source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var ets = plan.kind == _MEMBER_ETS
    var captured = _stored_buffer(source, String("ETS observations") if ets else String("Theta observations"), 2, context)
    var batch = captured.shape[0]
    var n = captured.shape[1]
    var letter = 0
    if ets:
        letter = String("NAM").find(String(py=py_item(fields, nk[].k__season)))
        if letter < 0:
            raise Error("ValueError: ETS: the stored season letter is not N, A or M")
    var scalars: List[Float64] = [Float64(plan.kind), Float64(batch), Float64(n), Float64(letter)]
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
