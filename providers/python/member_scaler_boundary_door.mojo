# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the Scaler family (StandardScaler, MinMaxScaler):
its place in `bindings/member_boundary_door.mojo` (recipe: MEMBER-ABI-V2.md;
RUN: core/scaler_member.mojo). Kind 0 is StandardScaler, kind 1
MinMaxScaler (`_mojolearn_member = (34, kind)` on each class).

  * `scaler_member_admit` (UNPACK, once per member): the estimator's own
    parameter checks and mode (the fit doors' words), the image entries the
    fit door selects (`scaler_fit_entries`, `_minmax_entries`) and the typed
    calls its route holds (`scaler_door_calls`: the scaler kernel entry, the
    x_prep transport entry, the direct answer), turned into the member's
    `DeviceRoute`; the six parameter words of the member file.
  * `scaler_member_pack` (PACK): the fitted attributes through the fit doors'
    own publications (`scaler_standard_publish`, `minmax_fit_publish`).
  * `scaler_member_state` (UNPACK): a member state from a fitted estimator.
    A StandardScaler whose n_samples_seen_ is a per-feature row (a fit over
    NaN columns) is refused in words: refit it as a member.

SERIAL BY NATURE: one member's admission or publication (a few record reads
and two or three block copies); a caller runs its members in parallel.
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.admitted_input_unpack import admit_x
from bindings.classical_arrays_door import classical_address
from bindings.member_owner_unpack import MemberOwner, member_host_only
from bindings.mode_answer_door import answer_mode_object
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_get, py_item, py_type
from bindings.scaler_keys_unpack import scaler_keys
from bindings.scaler_native_door import (
    _bool as scaler_bool, scaler_fit_entries, scaler_door_calls, scaler_standard_publish,
)
from bindings.minmax_native_door import (
    _configuration as minmax_configuration, _minmax_entries as minmax_entries, minmax_fit_publish,
)
from core.device_route import DeviceRoute, DEVICE_CPU, DEVICE_GPU
from core.entry_table import entry_ref_mode, entry_ref_vendor
from core.hostptr import f32_ptr
from core.mode_answer import MODE_FAMILY_SCALER
from core.scaler_core import ScalerStandardFit, SCALER_SEEN_INT, SCALER_SEEN_FLOAT
from core.scaler_member import SCALER_KIND_STANDARD, SCALER_BLOCK_STATS, SCALER_BLOCK_SEEN, SCALER_ROUTE_DIRECT
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_SCALARS, MEMBER_IMPORT_DONE, MEMBER_F64,
)


def _standard_mode(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    """StandardScaler.fit's mode word (the door's admission, in its words)."""
    var nk = scaler_keys()
    var b = native_loaded_module("builtins")
    var mode = py_item(fields, nk[].k_numeric_mode)
    if mode is None:
        mode = py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
    if py_type(mode) is not b.str:
        raise Error("numeric_mode must be fast, identical or None")
    mode = py_call(py_call(mode.strip).lower)
    var mode_word = String(mode)
    if mode_word != "fast" and mode_word != "identical":
        raise Error("numeric_mode must be fast, identical or None")
    answer_mode_object(mode, mode_word, MODE_FAMILY_SCALER)
    return mode


def scaler_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator."""
    var nk = scaler_keys()
    # A host scaler class, by its context or (lane mf-iterimp3b) by its own host-only word: its plan is the
    # public class's (python/mojolearn/_native_msel_more.py `_connect_host_members`), whose context binds the
    # device route, so the word is read first; the refusal is the one a host scaler always had.
    if member_host_only(model) or Bool(py=py_get(context, nk[].k_scaler_explicit_host, False)):
        raise Error("TypeError: mojolearn: a host scaler class does not run as a member")
    var words = List[Float64](length=6, fill=0.0)
    words[0] = Float64(kind)
    var mode = PythonObject(None)
    var entries = PythonObject(None)
    if kind == SCALER_KIND_STANDARD:
        _ = scaler_bool(py_item(fields, nk[].k_copy), context)
        words[1] = 1.0 if scaler_bool(py_item(fields, nk[].k_with_mean), context) else 0.0
        words[2] = 1.0 if scaler_bool(py_item(fields, nk[].k_with_std), context) else 0.0
        mode = _standard_mode(fields, context)
        entries = scaler_fit_entries(model, mode, context)
    elif kind == 1:
        var configuration = minmax_configuration(fields, context)
        mode = configuration[0]
        words[1] = 1.0 if Bool(py=configuration[4]) else 0.0
        words[3] = Float64(py=configuration[1])
        words[4] = Float64(py=configuration[2])
        entries = minmax_entries(model, mode, context)
        if py_get(entries, "minmax_fit") is None:
            raise Error("ImportError: mojolearn: MinMaxScaler training requires the preprocessing binding's minmax_fit")
    else:
        raise Error("ValueError: mojolearn: unknown scaler member kind " + String(kind))
    words[5] = 1.0 if String(mode) == "identical" else 0.0
    var calls = scaler_door_calls(model, entries, context)
    if not calls.has_table:
        raise Error("ImportError: mojolearn: this scaler image publishes no scaler kernel table entry")
    var vendor = entry_ref_vendor(calls.table.scaler)
    var route = DeviceRoute(DEVICE_CPU if vendor == 0 else DEVICE_GPU)
    route.mode = entry_ref_mode(calls.table.scaler)
    route.vendor = vendor
    route.entry = calls.table.scaler
    route.second = calls.table.xprep
    route.has_second = True
    route.flags = SCALER_ROUTE_DIRECT if calls.table.use_direct else 0
    owner.plan.route = route
    owner.plan.params = words^
    # The image entries the fit door selected pin the images for the member's life.
    owner.pins = Python.tuple(entries, mode)
    return True


def _export(plan: MemberPlan, mut call: MemberCall, block: Int) raises -> MemberMatrix:
    call.block = block
    call.flags = 0
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    return call.dst


def scaler_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model` with
    the fit doors' own publications (the same names, Arrays and seen word)."""
    var call = MemberCall(plan.route, state)
    var scalars_view = _export(plan, call, MEMBER_BLOCK_SCALARS)
    var scalars = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=scalars_view.data)
    var d = Int(scalars[0])
    var mode = PythonObject("identical") if plan.params[5] != 0.0 else PythonObject("fast")
    if plan.kind == SCALER_KIND_STANDARD:
        var fit = ScalerStandardFit(d)
        fit.seen_kind = Int(scalars[1])
        fit.seen_scalar = scalars[2]
        var stats = _export(plan, call, SCALER_BLOCK_STATS)
        unsafe_memcpy(dest=fit.stats.unsafe_ptr(), src=f32_ptr(stats.data), count=3 * d)
        var seen = _export(plan, call, SCALER_BLOCK_SEEN)
        if seen.cols > 0:
            fit.seen = List[Float32](length=seen.cols, fill=0.0)
            unsafe_memcpy(dest=fit.seen.unsafe_ptr(), src=f32_ptr(seen.data), count=seen.cols)
        scaler_standard_publish(fields, fit, d, plan.params[1] != 0.0, plan.params[2] != 0.0, mode, context)
        return
    var n = Int(scalars[1])
    var view = _export(plan, call, SCALER_BLOCK_STATS)
    var rows = List[Float32](length=5 * d, fill=0.0)
    unsafe_memcpy(dest=rows.unsafe_ptr(), src=f32_ptr(view.data), count=5 * d)
    var configuration = Python.tuple(mode, plan.params[3], plan.params[4], True, plan.params[1] != 0.0)
    minmax_fit_publish(fields, rows, n, d, configuration, context)


def _row(fields: PythonObject, name: String, d: Int, mut stats: List[Float32], offset: Int,
    context: PythonObject) raises:
    """Copy the fitted row `name` (None: left zero) into `stats` at `offset`."""
    var source = py_get(fields, name)
    if source is None:
        return
    var words = admit_x(source, 1, name, context)
    if Int(py=words.size) != d:
        raise Error("ValueError: the fitted " + name + " and n_features_in_ differ in length")
    unsafe_memcpy(dest=stats.unsafe_ptr() + offset, src=f32_ptr(Int(py=classical_address(words))), count=d)
    _ = words


def scaler_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted attributes of `model`.
    The caller frees it (`member_free`)."""
    var d_source = py_get(fields, "n_features_in_")
    var seen_source = py_get(fields, "n_samples_seen_")
    if d_source is None or seen_source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var d = Int(py=d_source)
    var standard = plan.kind == SCALER_KIND_STANDARD
    var stats = List[Float32](length=(3 if standard else 5) * d, fill=0.0)
    var scalars = List[Float64]()
    if standard:
        var kind = SCALER_SEEN_INT
        if py_type(seen_source) is b.float:
            kind = SCALER_SEEN_FLOAT
        elif py_type(seen_source) is not b.int:
            raise Error("NotImplementedError: mojolearn: a StandardScaler with a per-feature n_samples_seen_ "
                        + "does not become a member state; fit it as a member")
        scalars = [Float64(d), Float64(kind), Float64(py=seen_source)]
        _row(fields, "mean_", d, stats, 0, context)
        _row(fields, "var_", d, stats, d, context)
        _row(fields, "scale_", d, stats, 2 * d, context)
    else:
        scalars = [Float64(d), Float64(Int(py=seen_source))]
        var names: List[String] = ["data_min_", "data_max_", "data_range_", "scale_", "min_"]
        for index in range(5):  # driver-loop: the five fitted MinMax rows
            if py_get(fields, names[index]) is None:
                raise Error("NotFittedError: this member is not fitted yet")
            _row(fields, names[index], d, stats, index * d, context)
    var call = member_new(plan)
    try:
        call.block = SCALER_BLOCK_STATS
        call.x = MemberMatrix(Int(stats.unsafe_ptr()), 1, len(stats))
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, len(scalars), MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = len(stats)
    _ = len(scalars)
    return call.state
