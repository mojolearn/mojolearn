# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the Pls family (PLSRegression, PLSCanonical, CCA):
the family's place in `bindings/member_boundary_door.mojo` (recipe:
MEMBER-ABI-V2.md; RUN: core/pls_member.mojo). Kind 0 PLSRegression,
1 PLSCanonical, 2 CCA (`_mojolearn_member = (MEMBER_FAMILY_PLS, kind)` on
each class). `context` is the PLS owner's context (the one its own door
receives).

  * `pls_member_admit` (UNPACK, once per member): the estimator's own
    settings snapshot and refusals (`_pls_settings`, `pls_algorithm_check`),
    its x_decomp entries through its device route (`decomp_entries_routed`),
    the kit words of that image in the route (`entry`, `flags`), on a GPU
    image its device pool as the scope transport (`route.fifth` with
    SEQ_ROUTE_STORE, the XDecomp member's `_store_route`: the member then
    answers a row index and a scope, core/pls_member.mojo THE RESIDENT
    ROUTE), and the eight NEW words of core/pls_member.mojo.
  * `pls_member_pack` (PACK): the fitted fields the estimator's own fit
    writes (`pls_fit_pack`'s names, into the instance and its family state),
    from the member state's blocks, and the fitted mark.
  * `pls_member_state` (UNPACK): a member state from a fitted estimator.

SERIAL BY NATURE: one member's admission or publication (record reads and
one copy per fitted block); a caller runs its members in parallel.
"""
from std.python import Python, PythonObject
from bindings.admitted_input_unpack import admit_f32
from bindings.decomp_kit_native_door import DecompDoor, decomp_entries_routed, decomp_kit_unpack, decomp_kit_publish, decomp_kit_out_shape
from bindings.member_owner_unpack import MemberOwner
from bindings.pls_boundary_pack import pls_fit_mark
from bindings.pls_boundary_unpack import _pls_settings, pls_algorithm_check, pls_store
from bindings.pls_keys_unpack import pls_keys, PLS_OPN_FIT
from bindings.py_balanced import py_get, py_item, py_call
from core.device_route import DeviceRoute, DEVICE_CPU, DEVICE_GPU
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_COEF, MEMBER_BLOCK_INTERCEPT, MEMBER_BLOCK_SCALARS,
    MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE, MEMBER_F64,
)
from providers.python.classical_family_admission_pyapi_unpack import classical_family_admit
from providers.python.member_xdecomp_boundary_door import _store_route as xdecomp_store_route

# The family blocks `MEMBER_BLOCK_FAMILY + j` (core/pls_member.mojo) that
# are public fields, by j; j 11 (the iteration counts) is `n_iter_`.
comptime _FAMILY_FIELD_COUNT = 11


def _family_field(j: Int) -> String:
    if j == 0:
        return "_x_mean"
    if j == 1:
        return "_x_std"
    if j == 2:
        return "_y_std"
    if j == 3:
        return "x_weights_"
    if j == 4:
        return "y_weights_"
    if j == 5:
        return "x_loadings_"
    if j == 6:
        return "y_loadings_"
    if j == 7:
        return "x_rotations_"
    if j == 8:
        return "y_rotations_"
    if j == 9:
        return "_x_scores"
    return "_y_scores"


def pls_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator."""
    var nk = pls_keys()
    var state = classical_family_admit(model)
    var settings = _pls_settings(model, state, PLS_OPN_FIT, PythonObject("fit"), context)
    pls_algorithm_check(settings)
    var entries = decomp_entries_routed(model, settings, py_item(settings, nk[].k_numeric_mode), context)
    var kit = decomp_kit_unpack(entries)
    var route = DeviceRoute(DEVICE_CPU)
    var key = py_get(entries, "x_decomp_native_key")
    if key is not None:
        route.mode = Int(py=key[0])
        route.vendor = Int(py=key[1])
        route.device = DEVICE_CPU if route.vendor == 0 else DEVICE_GPU
    route.entry = kit.entry
    route.flags = (1 if kit.keyed else 0) | (2 if kit.resident else 0) | (kit.present << 2)
    var resident = kit.resident and kit.keyed
    kit.close()
    if resident:
        # Once per member (UNPACK): the same pool the kit's operations use.
        xdecomp_store_route(entries, route)
    owner.plan.route = route
    var deflation = String(py=py_item(settings, nk[].k__deflation))
    var pmode = String(py=py_item(settings, nk[].k__pmode))
    var svd = String(py=py_get(settings, nk[].k_algorithm, PythonObject("nipals"))) == "svd"
    var words: List[Float64] = [
        Float64(kind), Float64(Int(py=py_item(settings, nk[].k_n_components))),
        Float64(Int(py=py_item(settings, nk[].k_max_iter))), 1.0 if pmode == "B" else 0.0,
        1.0 if deflation == "canonical" else 0.0, 1.0 if svd else 0.0,
        1.0 if Bool(py=py_item(settings, nk[].k_scale)) else 0.0, Float64(py=py_item(settings, nk[].k_tol)),
    ]
    owner.plan.params = words^
    owner.pins = Python.tuple(entries, state)
    return True


def pls_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`, as
    the estimator's own fit does (the same names, so its own predict,
    transform and score work on the result)."""
    var nk = pls_keys()
    var family_state = classical_family_admit(model)
    var settings = _pls_settings(model, family_state, PLS_OPN_FIT, PythonObject("fit"), context)
    var entries = decomp_entries_routed(model, settings, py_item(settings, nk[].k_numeric_mode), context)
    var door = DecompDoor(entries, py_item(context, nk[].k_output_context))
    var kit = decomp_kit_unpack(entries)
    try:
        var call = MemberCall(plan.route, state)
        call.block = MEMBER_BLOCK_SCALARS
        _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
        var scalars = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
        var p = Int(scalars[1])
        var q = Int(scalars[2])
        call.block = MEMBER_BLOCK_COEF
        _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
        pls_store(fields, family_state, "coef_",
            decomp_kit_publish(door, kit, kit.borrow(call.dst.data, call.dst.rows, call.dst.cols)))
        call.block = MEMBER_BLOCK_INTERCEPT
        _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
        var y_mean = decomp_kit_out_shape(door, kit, kit.borrow(call.dst.data, 1, q), Python.tuple(q))
        pls_store(fields, family_state, "intercept_", y_mean)
        pls_store(fields, family_state, "_y_mean",
            decomp_kit_out_shape(door, kit, kit.borrow(call.dst.data, 1, q), Python.tuple(q)))
        for j in range(_FAMILY_FIELD_COUNT):  # driver-loop: over the fitted blocks
            call.block = MEMBER_BLOCK_FAMILY + j
            _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
            var matrix = kit.borrow(call.dst.data, call.dst.rows, call.dst.cols)
            var value: PythonObject
            if j <= 2:
                value = decomp_kit_out_shape(door, kit, matrix, Python.tuple(call.dst.rows * call.dst.cols))
            else:
                value = decomp_kit_publish(door, kit, matrix)
            pls_store(fields, family_state, _family_field(j), value)
        call.block = MEMBER_BLOCK_FAMILY + _FAMILY_FIELD_COUNT
        _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
        var counts = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
        var all_iterations = Python.list()
        for k in range(call.dst.cols):  # driver-loop: publish completed component iteration counts
            _ = py_call(all_iterations.append, PythonObject(Int(counts[k])))
        pls_store(fields, family_state, "n_iter_", all_iterations)
        pls_store(fields, family_state, "n_features_in_", PythonObject(p))
        pls_store(fields, family_state, "numeric_mode_", py_item(settings, nk[].k_numeric_mode))
        # The member's y is a (rows, targets) view: one target predicts 1-D.
        pls_store(fields, family_state, "_predict_1d", PythonObject(q == 1))
        if Bool(py=py_item(settings, nk[].k__pls_regression)):
            pls_store(fields, family_state, "x_scores_", py_item(fields, PythonObject("_x_scores")))
            pls_store(fields, family_state, "y_scores_", py_item(fields, PythonObject("_y_scores")))
        pls_fit_mark(family_state, PythonObject(None))
    finally:
        kit.close()
        door.keep_alive()


def _import_field(mut call: MemberCall, plan: MemberPlan, fields: PythonObject, name: String, block: Int,
    context: PythonObject, required: Bool) raises:
    var source = py_get(fields, name)
    if source is None:
        if required:
            raise Error("NotFittedError: this member is not fitted yet (" + name + ")")
        return
    var admitted = admit_f32(source, -1, name, context, finite=False)
    call.block = block
    var two_d = Int(py=admitted.array.ndim) == 2
    call.x = MemberMatrix(admitted.address, admitted.rows if two_d else 1,
        admitted.cols if two_d else admitted.rows * admitted.cols)
    call.flags = 0
    _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    _ = admitted^


def pls_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted attributes of `model`.
    The caller frees it (`member_free`)."""
    var coef = py_get(fields, "coef_")
    if coef is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var call = member_new(plan)
    try:
        _import_field(call, plan, fields, "coef_", MEMBER_BLOCK_COEF, context, True)
        _import_field(call, plan, fields, "_y_mean", MEMBER_BLOCK_INTERCEPT, context, True)
        for j in range(_FAMILY_FIELD_COUNT):  # driver-loop: over the fitted blocks
            _import_field(call, plan, fields, _family_field(j), MEMBER_BLOCK_FAMILY + j, context, j < 2)
        var p = Int(py=fields["n_features_in_"])
        var rotations = py_get(fields, "x_rotations_")
        var components = 0 if rotations is None else Int(py=rotations.shape[1])
        var q = Int(py=py_get(fields, "_y_mean").size)
        var scalars: List[Float64] = [Float64(components), Float64(p), Float64(q), 0.0]
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, 4, MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        _ = len(scalars)
    except error:
        member_free(plan, call)
        raise error
    return call.state
