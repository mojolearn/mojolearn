# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the Linear family (LinearRegression, Ridge): the
worked form of a family's place in `bindings/member_boundary_door.mojo`
(recipe: MEMBER-ABI-V2.md). Kind 0 is LinearRegression, kind 1 Ridge
(`_mojolearn_member = (MEMBER_FAMILY_LINEAR, kind)` on each class).

  * `linear_member_admit` (UNPACK, once per member): the estimator's own
    route through the family resolver (the record its own doors read: a hit
    is one record read), the metrics entry its score will call in the
    route's fifth slot, and the three parameter words of
    core/linear_member.mojo. Parameter refusals are the estimator's own.
  * `linear_member_pack` (PACK): the fitted attributes and the native state
    record of the estimator's own fit, from the member state's blocks.
  * `linear_member_state` (UNPACK): a member state from a fitted estimator.

SERIAL BY NATURE: one member's admission or publication (a few record reads
and two block copies of one row); a caller runs its members in parallel.
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.admitted_input_unpack import admit_x
from bindings.classical_arrays_door import classical_array, classical_address
from bindings.member_owner_unpack import MemberOwner
from bindings.py_balanced import py_get, py_type, py_setitem
from core.entry_table import EntryRef, entry_ref_word
from core.hostptr import f32_ptr
from core.linear_entry_table import LINEAR_TABLE_METRICS, linear_table_entry
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_COEF, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY,
    MEMBER_IMPORT_DONE, MEMBER_F64,
)
from providers.python.linear_entry_boundary_door import linear_entry_connection
from providers.python.linear_parameter_boundary_unpack import linear_ridge_parameters_values
from providers.python.linear_route_boundary_door import linear_route_call
from providers.python.linear_state_boundary_door import linear_state_publish
from x_decomp.descriptor_schema import xd_vendor_code
from x_metrics.descriptor_schema import METRICS_PROGRAM_ABI, MP_OPERATION_COUNT


def linear_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator."""
    var ridge = kind == 1
    var alpha = Float64(0.0)
    if ridge:
        var alpha_source = fields["alpha"]
        var solver_source = fields["solver"]
        var normalize_source = fields["normalize"]
        alpha = linear_ridge_parameters_values(alpha_source, solver_source, normalize_source)
    var intercept_source = fields["fit_intercept"]
    var intercept = Bool(py=intercept_source)
    var held = linear_route_call(model, fields, ridge, context, b)
    var route = held.route
    var metric_pins = PythonObject(None)
    var metric_owner = PythonObject(None)
    if not route.has_fifth:
        # The process's metrics entry, as the estimator's own score door
        # selects it (providers/python/linear_score_boundary_unpack.mojo); a
        # CPU route beside a GPU set already holds its CPU metrics entry.
        var metric_mode = context["backend_fields"]["_DEFAULT_MODE"]
        metric_pins = context["classical_metrics"][metric_mode]
        if py_type(metric_pins) is b.str:
            raise Error("ImportError: " + String(metric_pins))
        var metric_connection = linear_entry_connection(metric_pins, LINEAR_TABLE_METRICS)
        var metric_key = metric_connection.key.copy()
        metric_owner = metric_connection.images
        var vendor = xd_vendor_code(String(py_get(context, "classical_metric_vendor", context["vendor"])))
        var mode = 1 if String(metric_mode) == "identical" else 0
        if metric_key.vendor != vendor:
            raise Error("RuntimeError: native linear score metrics vendor differs from selected execution")
        if metric_key.mode != mode:
            raise Error("RuntimeError: native linear score metrics mode differs from selected execution")
        var entry = linear_table_entry[LINEAR_TABLE_METRICS](metric_key, 0, String(METRICS_PROGRAM_ABI),
            MP_OPERATION_COUNT)
        route.fifth = EntryRef(entry, entry_ref_word(metric_key.mode, metric_key.vendor, MP_OPERATION_COUNT))
        route.has_fifth = True
    owner.plan.route = route
    var words: List[Float64] = [1.0 if ridge else 0.0, 1.0 if intercept else 0.0, alpha]
    owner.plan.params = words^
    owner.pins = Python.tuple(held.owner, metric_pins, metric_owner)
    return True


def linear_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`, as
    the estimator's own fit does (the same names, and the same native state
    record its predict and score read)."""
    var call = MemberCall(plan.route, state)
    call.block = MEMBER_BLOCK_COEF
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var cols = call.dst.cols
    var coef = classical_array(cols, 0, context)
    unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(coef))), src=f32_ptr(call.dst.data), count=cols)
    call.block = MEMBER_BLOCK_FAMILY
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var mean = classical_array(cols, 0, context)
    unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(mean))), src=f32_ptr(call.dst.data), count=cols)
    call.block = MEMBER_BLOCK_SCALARS
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var scalars = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
    var intercept = scalars[0]
    var ymean = scalars[1]
    py_setitem(fields, "coef_", coef)
    py_setitem(fields, "intercept_", intercept)
    py_setitem(fields, "_x_mean", mean)
    py_setitem(fields, "_y_mean", ymean)
    py_setitem(fields, "n_features_in_", cols)
    if plan.kind == 1:
        py_setitem(fields, "solver_", "eig")
    # The member was fitted on the caller's own rows, never on the user's array.
    py_setitem(fields, "input_copied_", True)
    py_setitem(fields, "_mojolearn_fitted", True)
    linear_state_publish(model, coef, mean, cols, intercept, ymean, True)


def linear_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted attributes of `model`.
    The caller frees it (`member_free`)."""
    var coefficient_source = py_get(fields, "coef_")
    if coefficient_source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var intercept_source = fields["intercept_"]
    var mean_source = py_get(fields, "_x_mean")
    var ymean_source = py_get(fields, "_y_mean")
    var coef = admit_x(coefficient_source, 1, "coef_", context)
    var cols = Int(py=coef.size)
    var scalars: List[Float64] = [Float64(py=intercept_source), 0.0]
    if ymean_source is not None:
        scalars[1] = Float64(py=ymean_source)
    var call = member_new(plan)
    try:
        call.block = MEMBER_BLOCK_COEF
        call.x = MemberMatrix(Int(py=classical_address(coef)), 1, cols)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        if mean_source is not None:
            var mean = admit_x(mean_source, 1, "_x_mean", context)
            if Int(py=mean.size) != cols:
                raise Error("ValueError: the fitted column means and coef_ differ in length")
            call.block = MEMBER_BLOCK_FAMILY
            call.x = MemberMatrix(Int(py=classical_address(mean)), 1, cols)
            _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
            _ = mean
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
