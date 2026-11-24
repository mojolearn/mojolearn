# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the Cd family; RUN is core/cd_member.mojo.
Published kinds: 0 ElasticNet, 1 Lasso (`_mojolearn_member =
(MEMBER_FAMILY_CD, kind)` on each class).

  * `cd_member_admit` (UNPACK, once per member): the estimator's own
    refusals (bindings/x_linear_cd_unpack.mojo `_xl_cd_validate`), its own
    route (bindings/x_linear_route_unpack.mojo `xl_route(..., cd=True)`: the
    solver entry; a hit is one record read), the x_metrics program entry of the
    process's tier on the route's device in the route's fifth slot when that
    image is loaded, and the 7 parameter words (core/cd_member.mojo
    docstring). An explicit host inference class (no fit) is not admitted.
  * `cd_member_pack` (PACK): the fitted attributes `xl_cd_fit_pack` writes
    (bindings/x_linear_cd_pack.mojo), in its order; the member borrowed X,
    so `input_copied_` and `fortran_copied_` are False.
  * `cd_member_state` (UNPACK): a member state from a fitted estimator.

SERIAL BY NATURE: one member's admission or publication; a caller runs its
members in parallel.
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.admitted_input_unpack import admit_x
from bindings.classical_arrays_door import classical_array, classical_address
from bindings.member_owner_unpack import MemberOwner
from bindings.py_balanced import py_get, py_item, py_setitem
from bindings.x_linear_common_unpack import _truth, _float, _int
from bindings.x_linear_keys_unpack import x_linear_keys
from bindings.x_linear_cd_unpack import _xl_cd_validate, xl_cd_host_fit_words
from bindings.x_linear_route_unpack import xl_route, xl_ops_require, xl_host_only
from core.entry_table import EntryRef, entry_ref_word
from core.hostptr import f32_ptr
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_COEF, MEMBER_BLOCK_SCALARS, MEMBER_IMPORT_DONE, MEMBER_F64,
)
from core.cd_member import CD_MEMBER_ELASTIC_NET, CD_MEMBER_LASSO
from providers.python.metrics_route_boundary_door import member_metrics_program_fifth
from providers.python.device_route_boundary_door import device_route_value


def cd_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator."""
    var nk = x_linear_keys()
    if kind != CD_MEMBER_ELASTIC_NET and kind != CD_MEMBER_LASSO:
        raise Error("RuntimeError: mojolearn: unknown Cd member kind " + String(kind))
    if not owner.serving and xl_host_only(model):
        # An explicit host class predicts a saved model only: its own fit door's refusal, in its words and
        # first, as `xl_cd_fit_unpack` (lane mf-lin6; a serving caller that only predicts keeps the class's
        # own predict, `member_admit_serving` never raises). Lane w3f2-main (2026-11-24): read from the
        # class word the fit door reads (`xl_host_only`), before any route resolution; the former
        # `ops.host_only` of `xl_route` was never set (only `xl_route_host` sets it), so HostLasso and
        # HostElasticNet were fitted as members where their own fit refuses. A serving admission
        # (`owner.serving`: the caller only predicts on fitted state) is not a fit and is admitted.
        raise Error(xl_cd_host_fit_words(model))
    var ops = xl_route(model, fields, context, True)
    _xl_cd_validate(fields, context)
    xl_ops_require(ops)
    var shuffle = _int(py_item(context, nk[].k_cd_selection)[py_item(fields, nk[].k_selection)]) != 0
    var words: List[Float64] = [Float64(kind), 1.0 if _truth(py_item(fields, nk[].k_fit_intercept)) else 0.0,
        Float64(_int(py_item(fields, nk[].k_max_iter))), _float(py_item(fields, nk[].k_alpha)),
        _float(py_item(fields, nk[].k_l1_ratio)), _float(py_item(fields, nk[].k_tol)), 1.0 if shuffle else 0.0]
    var route = device_route_value(ops.owner)
    if not route.has_fifth:
        # The x_metrics program entry of the process tier on the route's
        # device (the program the estimator's own score runs), when that
        # image is loaded.
        var metric_mode = 1 if String(py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)) == "identical" else 0
        member_metrics_program_fifth(route, metric_mode, route.vendor)
    owner.plan.route = route
    owner.plan.params = words^
    owner.pins = Python.tuple(ops.owner, ops.scope)
    return True


def cd_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: `input_copied_`, `fortran_copied_`, `coef_` (d,), `intercept_`
    (a float), `n_iter_` and `n_features_in_`, as `xl_cd_fit_pack`."""
    var nk = x_linear_keys()
    var call = MemberCall(plan.route, state)
    call.block = MEMBER_BLOCK_COEF
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var cols = call.dst.cols
    var coef = classical_array(cols, 0, context)
    unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(coef))), src=f32_ptr(call.dst.data), count=cols)
    call.block = MEMBER_BLOCK_SCALARS
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var scalars = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
    var intercept = scalars[0]
    var n_iter = Int(scalars[1])
    py_setitem(fields, nk[].k_input_copied_, PythonObject(False))
    py_setitem(fields, nk[].k_fortran_copied_, PythonObject(False))
    py_setitem(fields, nk[].k_coef_, coef)
    py_setitem(fields, nk[].k_intercept_, PythonObject(intercept))
    py_setitem(fields, nk[].k_n_iter_, PythonObject(n_iter))
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(cols))


def cd_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding `coef_`, `intercept_` and
    `n_iter_`. The caller frees it (`member_free`)."""
    var nk = x_linear_keys()
    var coefficient_source = py_get(fields, nk[].k_coef_)
    if coefficient_source is None:
        raise Error("mojolearn ElasticNet: call fit before predict")
    var coef = admit_x(coefficient_source, 1, "coef_", context)
    var cols = Int(py=coef.size)
    var n_iter = py_get(fields, nk[].k_n_iter_)
    var scalars: List[Float64] = [_float(py_item(fields, nk[].k_intercept_)),
        0.0 if n_iter is None else Float64(_int(n_iter))]
    var call = member_new(plan)
    try:
        call.block = MEMBER_BLOCK_COEF
        call.x = MemberMatrix(Int(py=classical_address(coef)), 1, cols)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, 2, MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = coef
    _ = len(scalars)
    return call.state
