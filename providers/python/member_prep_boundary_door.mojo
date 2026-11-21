# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the Prep family: its place in
`bindings/member_boundary_door.mojo` (recipe: MEMBER-ABI-V2.md; RUN:
core/prep_member.mojo). Published: RobustScaler (kind 0,
`_mojolearn_member = (35, 0)`). The family's other classes (MaxAbsScaler
OrdinalEncoder OneHotEncoder TargetEncoder SimpleImputer IterativeImputer
KBinsDiscretizer QuantileTransformer PowerTransformer Normalizer
PolynomialFeatures SplineTransformer Binarizer) carry no mark yet: a caller
keeps their own doors, and admit refuses their kinds in words.

  * `prep_member_admit` (UNPACK, once per member): the estimator's own route
    (`scaler_entries`, `scaler_route_held`: one record read; the x_prep
    transport entry and its capability bits), the switches the door reads
    per call (read once here), and the parameter words with the door's own
    checks in its order.
  * `prep_member_pack` (PACK): the fitted attributes of the door's fit
    (bindings/robust_native_pack.mojo `robust_fit`), from the state's blocks.
  * `prep_member_state` (UNPACK): a member state from a fitted estimator.

SERIAL BY NATURE: one member's admission or publication (a few record reads
and two block copies); a caller runs its members in parallel.
"""
from std.os import getenv
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.member_owner_unpack import MemberOwner
from bindings.classical_arrays_door import classical_address
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_get, py_type, py_setitem, py_contains
from bindings.scaler_lifecycle_native_door import scaler_entries, scaler_requested_mode
from bindings.scaler_native_door import scaler_door_calls
from bindings.scaler_route_unpack import scaler_route_held, scaler_held_caps
from bindings.x_prep_native_owner_door import xprep_caps, xprep_storage, xprep_take
from bindings.x_prep_native_owner_run import xprep_r3
from bindings.p3_prep_fitted_unpack import prep_fitted_vector
from bindings.robust_native_pack import _quantiles as robust_quantiles
from bindings.robust_native_unpack import _truth as robust_truth
from core.device_route import DeviceRoute, DEVICE_CPU, DEVICE_GPU
from core.entry_table import entry_ref_mode, entry_ref_vendor
from core.hostptr import f32_ptr
from core.prep_member import (
    PREP_KIND_ROBUST, PREP_ROBUST_PARAM_COUNT, PREP_BLOCK_CENTER, PREP_BLOCK_SCALE,
    PREP_ROUTE_OUT, PREP_ROUTE_SCRATCH, PREP_ROUTE_RANGES, PREP_ROUTE_DEV_PUT,
)
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_SCALARS, MEMBER_IMPORT_DONE, MEMBER_F64,
)


def _robust_admit(model: PythonObject, fields: PythonObject, context: PythonObject, mut owner: MemberOwner) raises:
    var mode = scaler_requested_mode(fields, context)
    var entries = scaler_entries(model, mode, context)
    var quantiles = robust_quantiles(fields["quantile_range"], context)
    var lo = Float64(py=quantiles[0])
    var hi = Float64(py=quantiles[1])
    if not (0.0 <= lo and lo <= hi and hi <= 100.0):
        var b = native_loaded_module("builtins")
        var given = fields["quantile_range"]
        if py_type(given) is b.tuple or py_type(given) is b.list:
            var safe = True
            for k in range(2):  # small-loop(params: the two admitted quantile metadata values)
                var cls = py_type(given[k])
                safe = safe and (cls is b.float or cls is b.int or cls is b.bool or cls is b.str or cls is b.bytes)
            if safe:
                raise Error("mojolearn: invalid quantile range " + String(py_call(b.repr, given)))
        raise Error("mojolearn: invalid quantile range")
    var unit = robust_truth(fields["unit_variance"], context)
    if unit and not (0.0 < lo and lo < hi and hi < 100.0):
        raise Error("mojolearn: RobustScaler(unit_variance=True) needs 0 < q_min < q_max < 100 (norm.ppf of 0 or 1 is infinite)")
    var words = List[Float64](length=PREP_ROBUST_PARAM_COUNT, fill=0.0)
    words[0] = Float64(PREP_KIND_ROBUST)
    words[1] = 1.0 if robust_truth(fields["with_centering"], context) else 0.0
    words[2] = 1.0 if robust_truth(fields["with_scaling"], context) else 0.0
    words[3] = lo
    words[4] = hi
    words[5] = 1.0 if unit else 0.0
    var select = False
    if String(mode) == "fast" and Bool(py=context["robust_qselect"]) and not py_contains(entries, "x_prep_host_column"):
        var vendor = py_get(entries, "x_prep_vendor")
        select = vendor is not None and String(py_call(vendor)) == "metal"
    words[6] = 1.0 if select else 0.0
    # The door reads these per call; a member reads them once, here.
    words[7] = 1.0 if String(getenv("MOJOLEARN_XPREP_BLOCKED", "1").strip()) != "0" else 0.0
    words[8] = 1.0 if xprep_r3("work") else 0.0
    var caps = scaler_held_caps(scaler_route_held(model, entries))
    if not caps.held:
        caps = xprep_caps(entries)
    var flags = 0
    if caps.has_out:
        flags |= PREP_ROUTE_OUT
    if caps.has_scratch:
        flags |= PREP_ROUTE_SCRATCH
    if caps.has_ranges and String(getenv("MOJOLEARN_ARENA_RANGES", "1").strip()) != "0":
        flags |= PREP_ROUTE_RANGES
    if caps.dev_put:
        flags |= PREP_ROUTE_DEV_PUT
    # The image words of the preprocessing route (the scaler member's): the
    # member table key this family is published under.
    var calls = scaler_door_calls(model, entries, context)
    if not calls.has_table:
        raise Error("ImportError: mojolearn: this preprocessing image publishes no scaler kernel table entry")
    var vendor = entry_ref_vendor(calls.table.scaler)
    var route = DeviceRoute(DEVICE_CPU if vendor == 0 else DEVICE_GPU)
    route.mode = entry_ref_mode(calls.table.scaler)
    route.vendor = vendor
    route.second = caps.key
    route.has_second = True
    route.flags = flags
    owner.plan.route = route
    owner.plan.params = words^
    owner.pins = entries


def prep_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator."""
    if kind == PREP_KIND_ROBUST:
        _robust_admit(model, fields, context, owner)
        return True
    raise Error("ValueError: mojolearn: Prep member kind " + String(kind) + " is not published")


def _export(plan: MemberPlan, mut call: MemberCall, block: Int) raises -> MemberMatrix:
    call.block = block
    call.flags = 0
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    return call.dst


def _vector(view: MemberMatrix, context: PythonObject) raises -> PythonObject:
    # The door's fitted vector type (`xprep_take` of a storage), so the
    # door's transform admits it as an exact fitted Array.
    var count = view.rows * view.cols
    var storage = xprep_storage(count, context)
    if count > 0:
        unsafe_memcpy(dest=f32_ptr(Int(py=storage[1])), src=f32_ptr(view.data), count=count)
    return xprep_take(storage[0], 0, count, 0, context["output_context"])


def prep_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`, as
    the door's fit does."""
    if plan.kind != PREP_KIND_ROBUST:
        raise Error("ValueError: mojolearn: Prep member kind " + String(plan.kind) + " is not published")
    var call = MemberCall(plan.route, state)
    var scalars = _export(plan, call, MEMBER_BLOCK_SCALARS)
    var d = Int(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=scalars.data)[])
    var center = _export(plan, call, PREP_BLOCK_CENTER)
    py_setitem(fields, "center_", _vector(center, context) if center.cols > 0 else PythonObject(None))
    var scale = _export(plan, call, PREP_BLOCK_SCALE)
    py_setitem(fields, "scale_", _vector(scale, context) if scale.cols > 0 else PythonObject(None))
    py_setitem(fields, "numeric_mode_", scaler_requested_mode(fields, context))
    py_setitem(fields, "n_features_in_", d)
    py_setitem(fields, "_mojolearn_fitted", True)
    if py_contains(fields, "feature_names_in_"):
        _ = py_call(fields.pop, "feature_names_in_")


def _import(plan: MemberPlan, mut call: MemberCall, fields: PythonObject, name: String, block: Int, d: Int,
    context: PythonObject) raises:
    var source = fields[name]
    call.block = block
    call.flags = 0
    if source is None:
        call.x = MemberMatrix(0, 1, 0)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        return
    var vector = prep_fitted_vector(source, context)
    if Int(py=vector.size) != d:
        raise Error("RobustScaler " + name + " must hold one value per feature")
    call.x = MemberMatrix(Int(py=classical_address(vector)), 1, d)
    _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    _ = vector


def prep_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted attributes of `model`.
    The caller frees it (`member_free`)."""
    if plan.kind != PREP_KIND_ROBUST:
        raise Error("ValueError: mojolearn: Prep member kind " + String(plan.kind) + " is not published")
    if not py_contains(fields, "n_features_in_") or not py_contains(fields, "center_"):
        raise Error("NotFittedError: this member is not fitted yet")
    var bi = native_loaded_module("builtins")
    if py_type(fields["n_features_in_"]) is not bi.int:
        raise Error("TypeError: RobustScaler n_features_in_ must be a native integer")
    var d = Int(py=fields["n_features_in_"])
    var scalars: List[Float64] = [Float64(d)]
    var call = member_new(plan)
    try:
        _import(plan, call, fields, "center_", PREP_BLOCK_CENTER, d, context)
        _import(plan, call, fields, "scale_", PREP_BLOCK_SCALE, d, context)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, 1, MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = len(scalars)
    return call.state
