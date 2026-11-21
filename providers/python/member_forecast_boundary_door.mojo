# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the Forecast family (family 37): ARIMA, kind 0.
AutoARIMA (kind 1) is not published (see `forecast_member_admit`).

UNPACK / PACK only (GIL held, once per member estimator, never per fold):
  * `forecast_member_admit`: the estimator's own checks (`_order`: order
    and seasonal_order; `k_` 0 / 1; `method` an exact string mapped by the
    connected `arima_methods` codes; `maxiter`), its route through the
    ARIMA resolver (`arima_route`: the operations entry of its device and
    mode), and the NEW words of core/forecast_member.mojo.
  * `forecast_member_pack`: the fitted attributes the estimator's own fit
    writes (`ARIMA._adopt_fit`'s updates, committed through the forecast
    owner as its door does).
  * `forecast_member_state`: a state holding a fitted model's `_y`,
    `_exog` and `params_`.
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.member_owner_unpack import MemberOwner
from bindings.mode_metadata_native_door import _fields
from bindings.py_balanced import py_get, py_item, py_setitem, py_type
from bindings.tsa_keys_unpack import tsa_keys
from bindings.tsa_hw_native_door import tsa_array
from bindings.classical_arrays_door import classical_address
from bindings.tsa_arima_native_door import _order
from bindings.tsa_arima_route_unpack import arima_route
from providers.python.autoarima_lifecycle_pyapi_door import _integer
from providers.python.arima_buffer_pyapi_door import arima_buffer
from providers.python.forecast_owner_pyapi_unpack import forecast_commit
from core.xseq_marshal import xseq_widen_f32_f64
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_IMPORT_DONE, MEMBER_F64,
)

comptime _MEMBER_ARIMA = 0


def forecast_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator. False for AutoARIMA (kind 1: its fit is
    the order search of bindings/tsa_autoarima_native_door.mojo, which
    builds per-group ARIMA estimators in Python and fits them through
    their doors; nothing is published for it)."""
    if kind != _MEMBER_ARIMA:
        return False
    var nk = tsa_keys()
    var own = _fields(model)
    var order = _order(own, context)
    var k = _integer(py_item(own, nk[].k_k_), context)
    if k < 0 or k > 1:
        raise Error("ValueError: ARIMA k must be 0 or 1")
    var method = py_item(own, nk[].k_method)
    if py_type(method) is not b.str:
        raise Error("TypeError: ARIMA method requires an exact string")
    var maxiter = _integer(py_item(own, nk[].k_maxiter), context)
    var call = arima_route(model, own, context, b)
    owner.plan.route = call.route
    var method_code = Int(py=py_item(context, nk[].k_arima_methods)[method])
    var words: List[Float64] = [
        Float64(_MEMBER_ARIMA), Float64(order[0]), Float64(order[1]), Float64(order[2]), Float64(order[3]),
        Float64(order[4]), Float64(order[5]), Float64(order[6]), Float64(k), Float64(method_code), Float64(maxiter),
    ]
    owner.plan.params = words^
    owner.pins = Python.tuple(call.owner)
    return True


def _block(plan: MemberPlan, mut call: MemberCall, block: Int) raises -> MemberMatrix:
    call.block = block
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    return call.dst


def _copy_out(view: MemberMatrix, first: Int, count: Int, code: String, typestr: String, width: Int,
    shape: PythonObject, context: PythonObject) raises -> PythonObject:
    """A new Array of `shape` holding `count` items of `view` from item `first`."""
    var out = tsa_array(code, typestr, shape, count, width, context)
    if count > 0:
        unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(out))),
            src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=view.data + width * first),
            count=count * width)
    return out


def forecast_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`, as
    the estimator's own fit does (its own predict and forecast work on the
    result)."""
    var nk = tsa_keys()
    var call = MemberCall(plan.route, state)
    var scalars_view = _block(plan, call, MEMBER_BLOCK_SCALARS)
    var scalars = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=scalars_view.data)
    var batch = Int(scalars[1])
    var n_obs = Int(scalars[2])
    var n_exog = Int(scalars[3])
    var big_n = Int(scalars[4])
    var updates = Python.dict()
    py_setitem(updates, nk[].k_input_copied_, True)
    py_setitem(updates, nk[].k_batch_size_, batch)
    py_setitem(updates, nk[].k_n_obs_, n_obs)
    py_setitem(updates, nk[].k__y, _copy_out(_block(plan, call, MEMBER_BLOCK_FAMILY), 0, batch * n_obs, "f", "<f4", 4,
        Python.tuple(batch, n_obs), context))
    # The member's y is observations x series: one series is a 1-D y.
    py_setitem(updates, nk[].k__y_ndim, 1 if batch == 1 else 2)
    py_setitem(updates, nk[].k_n_exog_, n_exog)
    py_setitem(updates, nk[].k_complexity_, big_n)
    if n_exog > 0:
        py_setitem(updates, nk[].k__exog, _copy_out(_block(plan, call, MEMBER_BLOCK_FAMILY + 1), 0,
            batch * n_obs * n_exog, "f", "<f4", 4, Python.tuple(batch, n_obs, n_exog), context))
    else:
        py_setitem(updates, nk[].k__exog, PythonObject(None))
    var square = Python.tuple(batch, big_n)
    py_setitem(updates, nk[].k_params_, _copy_out(_block(plan, call, MEMBER_BLOCK_FAMILY + 2), 0, batch * big_n,
        "f", "<f4", 4, square, context))
    py_setitem(updates, nk[].k_x_, _copy_out(_block(plan, call, MEMBER_BLOCK_FAMILY + 3), 0, batch * big_n,
        "f", "<f4", 4, square, context))
    py_setitem(updates, nk[].k_x0_, _copy_out(_block(plan, call, MEMBER_BLOCK_FAMILY + 4), 0, batch * big_n,
        "f", "<f4", 4, square, context))
    var one = Python.tuple(batch)
    var flags = _block(plan, call, MEMBER_BLOCK_FAMILY + 6)
    py_setitem(updates, nk[].k_n_iter_, _copy_out(flags, 0, batch, "i", "<i4", 4, one, context))
    py_setitem(updates, nk[].k_retcode_, _copy_out(flags, batch, batch, "i", "<i4", 4, one, context))
    var stats = _block(plan, call, MEMBER_BLOCK_FAMILY + 5)
    # llf_ = stats[:batch].astype("<f8"): each float32 widened exactly
    var llf = tsa_array("d", "<f8", one, batch, 8, context)
    xseq_widen_f32_f64(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=stats.data),
        MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(llf))), batch)
    py_setitem(updates, nk[].k_llf_, llf)
    py_setitem(updates, nk[].k_fx_, _copy_out(stats, batch, batch, "f", "<f4", 4, one, context))
    var ics = _block(plan, call, MEMBER_BLOCK_FAMILY + 7)
    py_setitem(updates, nk[].k_aic_, _copy_out(ics, 0, batch, "d", "<f8", 8, one, context))
    py_setitem(updates, nk[].k_bic_, _copy_out(ics, batch, batch, "d", "<f8", 8, one, context))
    forecast_commit(_fields(model), updates, py_item(context, nk[].k_tsa_owner_context))


def forecast_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted `_y`, `_exog` and
    `params_` of `model`. The caller frees it (`member_free`)."""
    var nk = tsa_keys()
    var own = _fields(model)
    if py_get(own, nk[].k_params_) is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var batch = _integer(py_item(own, nk[].k_batch_size_), context)
    var n_obs = _integer(py_item(own, nk[].k_n_obs_), context)
    var n_exog = _integer(py_get(own, nk[].k_n_exog_, PythonObject(0)), context)
    var big_n = _integer(py_item(own, nk[].k_complexity_), context)
    var y = arima_buffer(py_item(own, nk[].k__y), "y", context, True)
    var params = arima_buffer(py_item(own, nk[].k_params_), "params", context, True)
    var scalars: List[Float64] = [Float64(_MEMBER_ARIMA), Float64(batch), Float64(n_obs), Float64(n_exog),
        Float64(big_n)]
    var call = member_new(plan)
    try:
        call.block = MEMBER_BLOCK_FAMILY
        call.x = MemberMatrix(y.address, batch, y.count // max(batch, 1))
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_FAMILY + 2
        call.x = MemberMatrix(params.address, batch, params.count // max(batch, 1))
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        if n_exog > 0:
            var ex = arima_buffer(py_item(own, nk[].k__exog), "exog", context, True)
            call.block = MEMBER_BLOCK_FAMILY + 1
            call.x = MemberMatrix(ex.address, ex.count // n_exog, n_exog)
            _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
            _ = ex^
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, len(scalars), MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = y^
    _ = params^
    _ = len(scalars)
    return call.state
