# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the XLinear family; RUN is core/xlinear_member.mojo.
Published kinds: 0 PoissonRegressor, 1 GammaRegressor, 2 TweedieRegressor,
3 HuberRegressor, 4 BayesianRidge (`_mojolearn_member = (MEMBER_FAMILY_XLINEAR, kind)` on
each class).

  * `xlinear_member_admit` (UNPACK, once per member): the estimator's own
    route (bindings/x_linear_route_unpack.mojo `xl_route`: a hit is one
    record read), the metrics R^2 entry of the process's tier on the route's
    device in the route's fifth slot when that image is loaded, and the
    kind's parameter words (core/xlinear_member.mojo docstring) after the
    estimator's own checks (bindings/x_linear_regressors_unpack.mojo: the
    kind's `xl_fit_<kind>_unpack` refusals).
  * `xlinear_member_pack` (PACK): the fitted attributes the kind's
    `xl_fit_<kind>_pack` writes (bindings/x_linear_regressors_pack.mojo).
  * `xlinear_member_state` (UNPACK): a member state from a fitted estimator.

The family's classes not published yet (each takes its own kind here, the
same pattern over its own fit request and result layout) are listed in the
lane's result file.

SERIAL BY NATURE: one member's admission or publication; a caller runs its
members in parallel.
"""
from std.ffi import c_int, c_long
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.admitted_input_unpack import admit_x
from bindings.classical_arrays_door import classical_array, classical_address
from bindings.member_owner_unpack import MemberOwner
from bindings.py_balanced import py_get, py_item, py_contains, py_setitem
from bindings.x_linear_common_unpack import _truth, _ge, _float, _int, _name, _class_attr, _xl_bayes_refuse, _xl_or
from bindings.x_linear_keys_unpack import x_linear_keys
from bindings.x_linear_regressors_unpack import _xl_tweedie_link
from bindings.x_linear_route_unpack import xl_route, xl_ops_require
from core.entry_table import EntryRef, entry_ref_word
from core.hostptr import f32_ptr
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_COEF, MEMBER_BLOCK_SCALARS, MEMBER_IMPORT_DONE, MEMBER_F64,
)
from core.xlinear_member import XL_MEMBER_HUBER, XL_MEMBER_BAYES, XL_MEMBER_BLOCK_OUTLIERS
from core.metrics_entry_table import MetricsEntryKey, metrics_r2_entry, MR_OP_COUNT
from providers.python.device_route_boundary_door import device_route_value
from x_linear.door_run import LINK_IDENTITY, LINK_EXP


def _xl_glm_words(model: PythonObject, fields: PythonObject, kind: Int) raises -> List[Float64]:
    """Kinds 0..2: `xl_fit_glm_unpack`'s refusals, then the seven words."""
    var nk = x_linear_keys()
    var name = _name(model)
    if not py_contains(Python.tuple("lbfgs", "newton-cholesky"), py_item(fields, nk[].k_solver)):
        raise Error("mojolearn " + name + ": solver must be 'lbfgs' or 'newton-cholesky'")
    if _truth(py_item(fields, nk[].k_warm_start)):
        raise Error("mojolearn " + name + ": warm_start is not implemented")
    if not _ge(py_item(fields, nk[].k_alpha), PythonObject(0)):
        raise Error("mojolearn " + name + ": alpha must be >= 0")
    var tweedie = kind == 2
    var link = _xl_tweedie_link(fields) if tweedie else 1
    var power = _float(py_item(fields, nk[].k_power)) if tweedie else _float(_class_attr(model, "_power"))
    return [Float64(kind), _intercept_word(fields), Float64(_int(py_item(fields, nk[].k_max_iter))), Float64(link),
        power, _float(py_item(fields, nk[].k_alpha)), _float(py_item(fields, nk[].k_tol))]


def _xl_huber_words(fields: PythonObject) raises -> List[Float64]:
    """Kind 3: `xl_fit_huber_unpack`'s refusals, then the six words."""
    var nk = x_linear_keys()
    var epsilon = py_item(fields, nk[].k_epsilon)
    if not _ge(epsilon, PythonObject(1.0)):
        raise Error("mojolearn HuberRegressor: epsilon must be >= 1.0")
    if _truth(py_item(fields, nk[].k_warm_start)):
        raise Error("mojolearn HuberRegressor: warm_start is not implemented")
    return [Float64(XL_MEMBER_HUBER), _intercept_word(fields), Float64(_int(py_item(fields, nk[].k_max_iter))),
        _float(epsilon), _float(py_item(fields, nk[].k_alpha)), _float(py_item(fields, nk[].k_tol))]


def _xl_bayes_words(model: PythonObject, fields: PythonObject) raises -> List[Float64]:
    """Kind 4: `xl_fit_bayes_unpack`'s refusals, then the ten words."""
    var nk = x_linear_keys()
    _xl_bayes_refuse(model, fields, PythonObject(False))
    return [Float64(XL_MEMBER_BAYES), _intercept_word(fields), Float64(_int(py_item(fields, nk[].k_max_iter))),
        _float(py_item(fields, nk[].k_tol)), _float(py_item(fields, nk[].k_alpha_1)),
        _float(py_item(fields, nk[].k_alpha_2)), _float(py_item(fields, nk[].k_lambda_1)),
        _float(py_item(fields, nk[].k_lambda_2)), _xl_or(py_item(fields, nk[].k_alpha_init), -1.0),
        _xl_or(py_item(fields, nk[].k_lambda_init), -1.0)]


def _intercept_word(fields: PythonObject) raises -> Float64:
    var nk = x_linear_keys()
    return 1.0 if _truth(py_item(fields, nk[].k_fit_intercept)) else 0.0


def xlinear_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator."""
    var nk = x_linear_keys()
    var words: List[Float64]
    if kind >= 0 and kind <= 2:
        words = _xl_glm_words(model, fields, kind)
    elif kind == XL_MEMBER_HUBER:
        words = _xl_huber_words(fields)
    elif kind == XL_MEMBER_BAYES:
        words = _xl_bayes_words(model, fields)
    else:
        raise Error("RuntimeError: mojolearn: unknown XLinear member kind " + String(kind))
    var ops = xl_route(model, fields, context, False)
    xl_ops_require(ops)
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
    owner.plan.params = words^
    owner.pins = Python.tuple(ops.owner, ops.scope)
    return True


def _bool_list(address: Int, count: Int) raises -> PythonObject:
    """One Python list of bools from `count` int32 words (0 / 1), published
    straight into its slots (bindings/x_linear_regressors_pack.mojo
    `_xl_huber_mask`'s form)."""
    ref cpy = Python().cpython()
    var raw = cpy.lib.get_function[PyObjectPtr]("PyList_New")(count)
    if not raw:
        raise cpy.unsafe_get_error()
    var out = PythonObject(from_owned=raw)
    var box = cpy.lib.get_function[PyObjectPtr]("PyBool_FromLong")
    var set_item = cpy.lib.get_function[c_int]("PyList_SetItem")
    var data = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=address)
    for i in range(count):  # driver-loop: publish completed native result words as Python bool slots
        var item = box(c_long(data[i] != 0))
        if not item:
            raise cpy.unsafe_get_error()
        if set_item(out._obj_ptr, i, item) != 0:
            raise cpy.unsafe_get_error()
    return out


def xlinear_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`, as
    the estimator's own fit does (bindings/x_linear_regressors_pack.mojo)."""
    var nk = x_linear_keys()
    var kind = Int(plan.params[0])
    var call = MemberCall(plan.route, state)
    call.block = MEMBER_BLOCK_COEF
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var cols = call.dst.cols
    var coef = classical_array(cols, 0, context)
    unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(coef))), src=f32_ptr(call.dst.data), count=cols)
    var blocks = call.out_cols
    call.block = MEMBER_BLOCK_SCALARS
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var scalars = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
    # float32 words, published as Python floats (the result words' rounding)
    var intercept = Float64(Float32(scalars[0]))
    py_setitem(fields, nk[].k_coef_, coef)
    py_setitem(fields, nk[].k_intercept_, PythonObject(intercept))
    if kind == XL_MEMBER_HUBER:
        py_setitem(fields, nk[].k_scale_, PythonObject(Float64(Float32(scalars[1]))))
        py_setitem(fields, nk[].k_n_iter_, PythonObject(Int(scalars[2])))
        py_setitem(fields, nk[].k_n_features_in_, PythonObject(cols))
        if blocks > 2:
            call.block = XL_MEMBER_BLOCK_OUTLIERS
            _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
            py_setitem(fields, nk[].k_outliers_, _bool_list(call.dst.data, call.dst.cols))
        return
    if kind == XL_MEMBER_BAYES:
        py_setitem(fields, nk[].k_alpha_, PythonObject(Float64(Float32(scalars[1]))))
        py_setitem(fields, nk[].k_lambda_, PythonObject(Float64(Float32(scalars[2]))))
        py_setitem(fields, nk[].k_n_iter_, PythonObject(Int(scalars[3])))
        py_setitem(fields, nk[].k_n_features_in_, PythonObject(cols))
        return
    py_setitem(fields, nk[].k_n_iter_, PythonObject(Int(scalars[1])))
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(cols))
    py_setitem(fields, nk[].k__link, PythonObject(LINK_EXP if plan.params[3] == 1.0 else LINK_IDENTITY))


def _count_word(fields: PythonObject, key: PythonObject) raises -> Float64:
    var value = py_get(fields, key)
    if value is None:
        return 0.0
    return Float64(Int(py=value))


def xlinear_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted attributes of `model`
    (coef_ and the kind's scalars; training records are not imported).
    The caller frees it (`member_free`)."""
    var nk = x_linear_keys()
    var kind = Int(plan.params[0])
    var coefficient_source = py_get(fields, nk[].k_coef_)
    if coefficient_source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var coef = admit_x(coefficient_source, 1, "coef_", context)
    var cols = Int(py=coef.size)
    var intercept = Float64(py=py_item(fields, nk[].k_intercept_))
    var scalars: List[Float64]
    if kind == XL_MEMBER_HUBER:
        scalars = [intercept, Float64(py=py_item(fields, nk[].k_scale_)), _count_word(fields, nk[].k_n_iter_)]
    elif kind == XL_MEMBER_BAYES:
        scalars = [intercept, Float64(py=py_item(fields, nk[].k_alpha_)), Float64(py=py_item(fields, nk[].k_lambda_)),
            _count_word(fields, nk[].k_n_iter_)]
    else:
        scalars = [intercept, _count_word(fields, nk[].k_n_iter_)]
    var call = member_new(plan)
    try:
        call.block = MEMBER_BLOCK_COEF
        call.x = MemberMatrix(Int(py=classical_address(coef)), 1, cols)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, len(scalars), MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = coef
    _ = len(scalars)
    return call.state
