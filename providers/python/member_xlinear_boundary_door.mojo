# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the XLinear family; RUN is core/xlinear_member.mojo.
Published kinds: 0 PoissonRegressor, 1 GammaRegressor, 2 TweedieRegressor
(`_mojolearn_member = (MEMBER_FAMILY_XLINEAR, kind)` on each class).

  * `xlinear_member_admit` (UNPACK, once per member): the estimator's own
    route (bindings/x_linear_route_unpack.mojo `xl_route`: a hit is one
    record read), the metrics R^2 entry of the process's tier on the route's
    device in the route's fifth slot when that image is loaded, and the seven
    parameter words, after the estimator's own checks
    (bindings/x_linear_regressors_unpack.mojo `xl_fit_glm_unpack`'s solver,
    warm_start and alpha refusals; Tweedie's power / link rule).
  * `xlinear_member_pack` (PACK): coef_, intercept_, n_iter_,
    n_features_in_ and `_link`, as `xl_fit_glm_pack` writes them.
  * `xlinear_member_state` (UNPACK): a member state from a fitted estimator.

The family's other classes (HuberRegressor, BayesianRidge, ARDRegression,
Lars, LassoLars, QuantileRegressor, RidgeClassifier, RidgeCV, LassoCV,
ElasticNetCV, LogisticRegressionCV, IsotonicRegression) are not published
yet: each takes its own kind here, the same pattern over its own fit
request and result layout (bindings/x_linear_regressors_unpack.mojo,
x_linear_cv_unpack.mojo, x_linear_isotonic_unpack.mojo).

SERIAL BY NATURE: one member's admission or publication; a caller runs its
members in parallel.
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.admitted_input_unpack import admit_x
from bindings.classical_arrays_door import classical_array, classical_address
from bindings.member_owner_unpack import MemberOwner
from bindings.py_balanced import py_get, py_item, py_contains, py_setitem
from bindings.x_linear_common_unpack import _truth, _ge, _float, _int, _name, _class_attr
from bindings.x_linear_keys_unpack import x_linear_keys
from bindings.x_linear_regressors_unpack import _xl_tweedie_link
from bindings.x_linear_route_unpack import xl_route, xl_ops_require
from core.entry_table import EntryRef, entry_ref_word
from core.hostptr import f32_ptr
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_COEF, MEMBER_BLOCK_SCALARS, MEMBER_IMPORT_DONE, MEMBER_F64,
)
from core.metrics_entry_table import MetricsEntryKey, metrics_r2_entry, MR_OP_COUNT
from providers.python.device_route_boundary_door import device_route_value
from x_linear.door_run import LINK_IDENTITY, LINK_EXP


def xlinear_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator."""
    if kind < 0 or kind > 2:
        raise Error("RuntimeError: mojolearn: unknown XLinear member kind " + String(kind))
    var nk = x_linear_keys()
    var name = _name(model)
    if not py_contains(Python.tuple("lbfgs", "newton-cholesky"), py_item(fields, nk[].k_solver)):
        raise Error("mojolearn " + name + ": solver must be 'lbfgs' or 'newton-cholesky'")
    if _truth(py_item(fields, nk[].k_warm_start)):
        raise Error("mojolearn " + name + ": warm_start is not implemented")
    if not _ge(py_item(fields, nk[].k_alpha), PythonObject(0)):
        raise Error("mojolearn " + name + ": alpha must be >= 0")
    var ops = xl_route(model, fields, context, False)
    xl_ops_require(ops)
    var tweedie = kind == 2
    var link = _xl_tweedie_link(fields) if tweedie else 1
    var power = _float(py_item(fields, nk[].k_power)) if tweedie else _float(_class_attr(model, "_power"))
    var route = device_route_value(ops.owner)
    if not route.has_fifth:
        # The metrics image's R^2 entry the estimator's own score reaches
        # (its process tier, the route's device), when that image is loaded.
        var metric_mode = 1 if String(py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)) == "identical" else 0
        try:
            var entry = metrics_r2_entry(MetricsEntryKey(metric_mode, route.vendor))
            route.fifth = EntryRef(entry, entry_ref_word(metric_mode, route.vendor, MR_OP_COUNT))
            route.has_fifth = True
        except problem:
            # Not loaded: score is refused in words; the caller runs its own metric.
            _ = String(problem)
    owner.plan.route = route
    var words: List[Float64] = [Float64(kind), 1.0 if _truth(py_item(fields, nk[].k_fit_intercept)) else 0.0,
        Float64(_int(py_item(fields, nk[].k_max_iter))), Float64(link), power, _float(py_item(fields, nk[].k_alpha)),
        _float(py_item(fields, nk[].k_tol))]
    owner.plan.params = words^
    owner.pins = Python.tuple(ops.owner, ops.scope)
    return True


def xlinear_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`, as
    the estimator's own fit does."""
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
    var intercept = Float64(Float32(scalars[0]))
    var iterations = Int(scalars[1])
    py_setitem(fields, nk[].k_coef_, coef)
    py_setitem(fields, nk[].k_intercept_, PythonObject(intercept))
    py_setitem(fields, nk[].k_n_iter_, PythonObject(iterations))
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(cols))
    py_setitem(fields, nk[].k__link, PythonObject(LINK_EXP if plan.params[3] == 1.0 else LINK_IDENTITY))


def xlinear_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted attributes of `model`.
    The caller frees it (`member_free`)."""
    var nk = x_linear_keys()
    var coefficient_source = py_get(fields, nk[].k_coef_)
    if coefficient_source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var coef = admit_x(coefficient_source, 1, "coef_", context)
    var cols = Int(py=coef.size)
    var iterations = Float64(0.0)
    var n_iter = py_get(fields, nk[].k_n_iter_)
    if n_iter is not None:
        iterations = Float64(Int(py=n_iter))
    var scalars: List[Float64] = [Float64(py=py_item(fields, nk[].k_intercept_)), iterations]
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
