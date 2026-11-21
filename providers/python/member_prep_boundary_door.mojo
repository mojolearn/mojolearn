# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the Prep family: its place in
`bindings/member_boundary_door.mojo` (recipe: MEMBER-ABI-V2.md; RUN:
core/prep_member.mojo). Published: RobustScaler (kind 0,
`_mojolearn_member = (35, 0)`), MaxAbsScaler (kind 1, `(35, 1)`; not on the
Apple fast direct fit entry, where admit answers False and the caller keeps
the door), Binarizer (kind 2, `(35, 2)`), Normalizer (kind 3, `(35, 3)`),
PolynomialFeatures (kind 4, `(35, 4)`), QuantileTransformer (kind 5,
`(35, 5)`; admit answers False for a subsample below 1, which only the door
handles), PowerTransformer (kind 6, `(35, 6)`).
The family's other classes (OrdinalEncoder OneHotEncoder TargetEncoder
SimpleImputer IterativeImputer KNNImputer KBinsDiscretizer SplineTransformer)
carry no mark: their RUN is the encode door's (bindings/prep_owner_native_door.mojo),
a caller keeps their own doors, and admit refuses their kinds in words.

  * `prep_member_admit` (UNPACK, once per member): the estimator's own route
    (`scaler_entries`, `scaler_route_held`: one record read; the x_prep
    transport entry and its capability bits), the switches the door reads
    per call (read once here), and the parameter words with the door's own
    checks in its order.
  * `prep_member_pack` (PACK): the fitted attributes of the door's fit
    (bindings/robust_native_pack.mojo `robust_fit`, bindings/maxabs_native_door.mojo
    `maxabs_fit`), from the state's blocks.
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
from bindings.prep_stateless_route_unpack import stateless_entries, stateless_caps
from bindings.prep_normalizer_lifecycle_unpack import _normalizer_norm as normalizer_norm
from bindings.prep_stateless_native_door import (
    _degrees as poly_degrees, _fam as stateless_fam, _pt_flags as stateless_pt_flags, _repr as stateless_repr,
)
from bindings.scaler_native_door import _input as scaler_input
from bindings.classical_arrays_door import classical_typed
from x_prep.prep_stateless_native_run import I64P, _poly_powers
from bindings.x_prep_native_owner_run import XprepCaps
from bindings.scaler_route_unpack import (
    scaler_route_held, scaler_held_caps, scaler_held_flags, SCALER_CAP_MAXABS_DIRECT,
)
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
    PREP_KIND_MAXABS, PREP_MAXABS_PARAM_COUNT, PREP_BLOCK_MAX_ABS,
    PREP_KIND_BINARIZER, PREP_KIND_NORMALIZER, PREP_STATELESS_PARAM_COUNT,
    PREP_KIND_POLYNOMIAL, PREP_KIND_QUANTILE, PREP_KIND_POWER,
    PREP_POLY_PARAM_COUNT, PREP_QUANTILE_PARAM_COUNT, PREP_POWER_PARAM_COUNT,
    PREP_BLOCK_EXTRA, PREP_BLOCK_ANCHOR, PREP_BLOCK_ANCHOR_KIND,
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
    _route_admit(model, entries, context, owner, scaler_held_caps(scaler_route_held(model, entries)), True)
    owner.plan.params = words^


def _route_admit(model: PythonObject, entries: PythonObject, context: PythonObject, mut owner: MemberOwner,
    held_caps: XprepCaps, scaler_route: Bool) raises:
    # The x_prep transport entry and its capability bits (the route's held
    # selection, else read by name), the switch MOJOLEARN_ARENA_RANGES (read
    # once here; the doors per run) and the preprocessing image's words.
    # `scaler_route`: the estimator holds a scaler device route (the scalers);
    # the stateless classes read the table from their entries.
    var caps = held_caps
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
    var calls = scaler_door_calls(model, entries, context) if scaler_route else scaler_door_calls(entries)
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
    owner.pins = entries


def _maxabs_admit(model: PythonObject, fields: PythonObject, context: PythonObject, mut owner: MemberOwner) raises -> Bool:
    var mode = scaler_requested_mode(fields, context)
    var entries = scaler_entries(model, mode, context)
    var held = scaler_route_held(model, entries)
    if (scaler_held_flags(held, entries) & SCALER_CAP_MAXABS_DIRECT) != 0:
        # The Apple fast direct fit entry is a Python-called export with its
        # own upload: not a member route. The caller keeps the door.
        return False
    var words = List[Float64](length=PREP_MAXABS_PARAM_COUNT, fill=0.0)
    words[0] = Float64(PREP_KIND_MAXABS)
    # The door reads these per fit; a member reads them once, here.
    var blocked = String(getenv("MOJOLEARN_XPREP_BLOCKED", "1").strip()) != "0"
    words[1] = 1.0 if blocked else 0.0
    words[2] = 1.0 if blocked and xprep_r3("work") else 0.0
    _route_admit(model, entries, context, owner, scaler_held_caps(held), True)
    owner.plan.params = words^
    return True


def _stateless_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    mut owner: MemberOwner) raises:
    # The doors' fit runs on the process's default mode and the estimator's
    # stateless route (bindings/prep_stateless_native_door.mojo, prep_normalizer_fit_pack.mojo).
    var mode = context["backend_fields"]["_DEFAULT_MODE"]
    var entries = stateless_entries(model, mode, context)
    var words = List[Float64](length=PREP_STATELESS_PARAM_COUNT, fill=0.0)
    words[0] = Float64(kind)
    if kind == PREP_KIND_BINARIZER:
        # The door's float(threshold), read once (the door per transform).
        words[1] = Float64(py=py_call(native_loaded_module("builtins").float, fields["threshold"]))
    else:
        words[1] = Float64(normalizer_norm(fields["norm"]))
    _route_admit(model, entries, context, owner, stateless_caps(model, entries), False)
    owner.plan.params = words^


def _poly_admit(model: PythonObject, fields: PythonObject, context: PythonObject, mut owner: MemberOwner) raises:
    # The door's `_fit` checks, in its order (order, degree), read once here.
    var b = native_loaded_module("builtins")
    var mode = context["backend_fields"]["_DEFAULT_MODE"]
    var entries = stateless_entries(model, mode, context)
    var order = fields["order"]
    if py_type(order) is not b.str or (String(order) != "C" and String(order) != "F"):
        raise Error("mojolearn: PolynomialFeatures order must be 'C' or 'F'")
    var degrees = poly_degrees(fields, context)
    var words = List[Float64](length=PREP_POLY_PARAM_COUNT, fill=0.0)
    words[0] = Float64(PREP_KIND_POLYNOMIAL)
    words[1] = Float64(degrees[0])
    words[2] = Float64(degrees[1])
    words[3] = 1.0 if Bool(py=fields["interaction_only"]) else 0.0
    words[4] = 1.0 if Bool(py=fields["include_bias"]) else 0.0
    _route_admit(model, entries, context, owner, stateless_caps(model, entries), False)
    owner.plan.params = words^


def _quantile_admit(model: PythonObject, fields: PythonObject, context: PythonObject,
    mut owner: MemberOwner) raises -> Bool:
    # The door's `_quantile_fit` reads, once here (the door per fit).
    var b = native_loaded_module("builtins")
    var distribution = fields["output_distribution"]
    if py_type(distribution) is not b.str or (String(distribution) != "uniform" and String(distribution) != "normal"):
        raise Error("mojolearn: invalid output_distribution " + stateless_repr(distribution))
    var mode = context["backend_fields"]["_DEFAULT_MODE"]
    var entries = stateless_entries(model, mode, context)
    var words = List[Float64](length=PREP_QUANTILE_PARAM_COUNT, fill=0.0)
    words[0] = Float64(PREP_KIND_QUANTILE)
    words[1] = 1.0 if String(distribution) == "normal" else 0.0
    # max(1, n_quantiles): the door's max(1, min(n_quantiles, n)) is unchanged by it.
    words[2] = Float64(max(1, Int(py=py_call(b.int, fields["n_quantiles"]))))
    var subsample = fields["subsample"]
    words[3] = -1.0
    if subsample is not None:
        var rows = Int(py=py_call(b.int, subsample))
        if rows < 1:
            return False  # only the door's own program handles it
        words[3] = Float64(rows)
    var state = fields["random_state"]
    var seed = PythonObject(0) if state is None else py_call(b.int, state)
    var mask = PythonObject(0xFFFFFFFF)
    words[4] = Float64(Int(py=py_call(b.int.__dict__["__and__"], seed, mask)))
    words[5] = Float64(Int(py=py_call(b.int.__dict__["__and__"], py_call(b.int.__dict__["__rshift__"], seed,
        PythonObject(32)), mask)))
    words[6] = Float64(stateless_fam(mode, entries))
    words[7] = 1.0 if xprep_r3("work") else 0.0
    _route_admit(model, entries, context, owner, stateless_caps(model, entries), False)
    owner.plan.params = words^
    return True


def _power_admit(model: PythonObject, fields: PythonObject, context: PythonObject, mut owner: MemberOwner) raises:
    # The door's `_power_fit` reads, once here (the door per fit).
    var b = native_loaded_module("builtins")
    var method_name = fields["method"]
    if py_type(method_name) is not b.str or (String(method_name) != "yeo-johnson" and String(method_name) != "box-cox"):
        raise Error("mojolearn: invalid method " + stateless_repr(method_name))
    var mode = context["backend_fields"]["_DEFAULT_MODE"]
    var entries = stateless_entries(model, mode, context)
    var words = List[Float64](length=PREP_POWER_PARAM_COUNT, fill=0.0)
    words[0] = Float64(PREP_KIND_POWER)
    words[1] = 1.0 if String(method_name) == "box-cox" else 0.0
    words[2] = 1.0 if Bool(py=fields["standardize"]) else 0.0
    words[3] = Float64(stateless_fam(mode, entries))
    words[4] = Float64(stateless_pt_flags(mode, entries))
    words[5] = 1.0 if py_get(entries, "x_prep_host_column") is not None else 0.0
    words[6] = 1.0 if String(mode) == "identical" else 0.0
    words[7] = 1.0 if xprep_r3("work") else 0.0
    _route_admit(model, entries, context, owner, stateless_caps(model, entries), False)
    owner.plan.params = words^


def prep_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator."""
    if kind == PREP_KIND_ROBUST:
        _robust_admit(model, fields, context, owner)
        return True
    if kind == PREP_KIND_MAXABS:
        return _maxabs_admit(model, fields, context, owner)
    if kind == PREP_KIND_BINARIZER or kind == PREP_KIND_NORMALIZER:
        _stateless_admit(model, fields, kind, context, owner)
        return True
    if kind == PREP_KIND_POLYNOMIAL:
        _poly_admit(model, fields, context, owner)
        return True
    if kind == PREP_KIND_QUANTILE:
        return _quantile_admit(model, fields, context, owner)
    if kind == PREP_KIND_POWER:
        _power_admit(model, fields, context, owner)
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
    if plan.kind < PREP_KIND_ROBUST or plan.kind > PREP_KIND_POWER:
        raise Error("ValueError: mojolearn: Prep member kind " + String(plan.kind) + " is not published")
    var call = MemberCall(plan.route, state)
    var scalars = _export(plan, call, MEMBER_BLOCK_SCALARS)
    var words = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=scalars.data)
    var d = Int(words[0])
    if plan.kind == PREP_KIND_POLYNOMIAL or plan.kind == PREP_KIND_QUANTILE or plan.kind == PREP_KIND_POWER:
        _pack_stateless_fitted(fields, context, plan, call, d, Int(words[1]) if plan.kind != PREP_KIND_POWER else 0)
        return
    if plan.kind == PREP_KIND_BINARIZER or plan.kind == PREP_KIND_NORMALIZER:
        # The doors' fit fields (Binarizer `_fit`; Normalizer normalizer_fit
        # also sets the fitted flag).
        py_setitem(fields, "n_features_in_", d)
        py_setitem(fields, "numeric_mode_", context["backend_fields"]["_DEFAULT_MODE"])
        if plan.kind == PREP_KIND_NORMALIZER:
            py_setitem(fields, "_mojolearn_fitted", True)
        return
    if plan.kind == PREP_KIND_MAXABS:
        # The door's maxabs_fit attributes, in its order.
        var seen = Int(words[1])
        py_setitem(fields, "max_abs_", _vector(_export(plan, call, PREP_BLOCK_MAX_ABS), context))
        py_setitem(fields, "scale_", _vector(_export(plan, call, PREP_BLOCK_SCALE), context))
        py_setitem(fields, "n_features_in_", d)
        py_setitem(fields, "n_samples_seen_", seen)
        py_setitem(fields, "numeric_mode_", scaler_requested_mode(fields, context))
        py_setitem(fields, "_mojolearn_fitted", True)
        if py_contains(fields, "feature_names_in_"):
            _ = py_call(fields.pop, "feature_names_in_")
        return
    var center = _export(plan, call, PREP_BLOCK_CENTER)
    py_setitem(fields, "center_", _vector(center, context) if center.cols > 0 else PythonObject(None))
    var scale = _export(plan, call, PREP_BLOCK_SCALE)
    py_setitem(fields, "scale_", _vector(scale, context) if scale.cols > 0 else PythonObject(None))
    py_setitem(fields, "numeric_mode_", scaler_requested_mode(fields, context))
    py_setitem(fields, "n_features_in_", d)
    py_setitem(fields, "_mojolearn_fitted", True)
    if py_contains(fields, "feature_names_in_"):
        _ = py_call(fields.pop, "feature_names_in_")


def _matrix(view: MemberMatrix, rows: Int, cols: Int, context: PythonObject) raises -> PythonObject:
    # A (rows, cols) C-order fitted array of the door's storage type.
    var count = view.rows * view.cols
    if count != rows * cols:
        raise Error("RuntimeError: mojolearn: a fitted Prep member block has the wrong size")
    var storage = xprep_storage(count, context)
    if count > 0:
        unsafe_memcpy(dest=f32_ptr(Int(py=storage[1])), src=f32_ptr(view.data), count=count)
    return xprep_take(storage[0], 0, rows, cols, context["output_context"])


def _optional(view: MemberMatrix, context: PythonObject) raises -> PythonObject:
    return _vector(view, context) if view.rows * view.cols > 0 else PythonObject(None)


def _typed_words(view: MemberMatrix, context: PythonObject) raises -> PythonObject:
    # The door's `_words`: index words as a "<f4" classical array (count >= 1).
    var count = view.rows * view.cols
    var out = classical_typed(Python.tuple(count), "<f4", context["output_context"])
    unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(out))), src=f32_ptr(view.data), count=count)
    return out


def _pack_stateless_fitted(fields: PythonObject, context: PythonObject, plan: MemberPlan, mut call: MemberCall,
    d: Int, second: Int) raises:
    """The stateless door's fitted attributes (`_fit` polynomial branch,
    `_quantile_fit`, `_power_fit`), in its order. `second` is the scalars
    block's second word (n_output_features_, n_quantiles_)."""
    var mode = context["backend_fields"]["_DEFAULT_MODE"]
    if plan.kind == PREP_KIND_POLYNOMIAL:
        var nout = second
        var index_view = _export(plan, call, PREP_BLOCK_CENTER)
        var index_words = _typed_words(index_view, context)
        var start_view = _export(plan, call, PREP_BLOCK_SCALE)
        var start_words = _typed_words(start_view, context)
        # powers_ from the term table, as the door's `_poly_powers`.
        var index = List[Int](capacity=index_view.cols)
        var start = List[Int](capacity=start_view.cols)
        var ip = f32_ptr(index_view.data)
        var sp = f32_ptr(start_view.data)
        for i in range(index_view.rows * index_view.cols):  # driver-loop: over polynomial term table words (feature metadata)
            index.append(Int(ip[i]))
        for i in range(start_view.rows * start_view.cols):  # driver-loop: over polynomial term starts (feature metadata)
            start.append(Int(sp[i]))
        var powers = classical_typed(Python.tuple(max(nout, 1), d), "<i8", context["output_context"])
        _poly_powers(I64P(unsafe_from_address=Int(py=classical_address(powers))), d, nout, index, start)
        _ = powers  # keep the Python object alive through the native call above
        py_setitem(fields, "_poly_index", index_words)
        py_setitem(fields, "_poly_start", start_words)
        py_setitem(fields, "n_output_features_", nout)
        py_setitem(fields, "powers_", powers)
        py_setitem(fields, "n_features_in_", d)
        py_setitem(fields, "numeric_mode_", mode)
        return
    if plan.kind == PREP_KIND_QUANTILE:
        var nq = second
        py_setitem(fields, "_q", _vector(_export(plan, call, PREP_BLOCK_CENTER), context))
        py_setitem(fields, "quantiles_", _matrix(_export(plan, call, PREP_BLOCK_EXTRA), nq, d, context))
        py_setitem(fields, "references_", _vector(_export(plan, call, PREP_BLOCK_SCALE), context))
        py_setitem(fields, "n_quantiles_", nq)
        py_setitem(fields, "numeric_mode_", mode)
        py_setitem(fields, "n_features_in_", d)
        return
    py_setitem(fields, "lambdas_", _vector(_export(plan, call, PREP_BLOCK_EXTRA), context))
    py_setitem(fields, "_pt_anchor", _optional(_export(plan, call, PREP_BLOCK_ANCHOR), context))
    py_setitem(fields, "_pt_anchor_kind", _optional(_export(plan, call, PREP_BLOCK_ANCHOR_KIND), context))
    py_setitem(fields, "_mean", _optional(_export(plan, call, PREP_BLOCK_CENTER), context))
    py_setitem(fields, "_scale", _optional(_export(plan, call, PREP_BLOCK_SCALE), context))
    py_setitem(fields, "_method", Int(plan.params[1]))
    py_setitem(fields, "numeric_mode_", mode)
    py_setitem(fields, "n_features_in_", d)


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
        raise Error(("MaxAbsScaler " if plan.kind == PREP_KIND_MAXABS else "RobustScaler ") + name
                    + " must hold one value per feature")
    call.x = MemberMatrix(Int(py=classical_address(vector)), 1, d)
    _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    _ = vector


def prep_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted attributes of `model`.
    The caller frees it (`member_free`)."""
    var maxabs = plan.kind == PREP_KIND_MAXABS
    if plan.kind == PREP_KIND_BINARIZER or plan.kind == PREP_KIND_NORMALIZER:
        return _stateless_state(fields, plan)
    if plan.kind == PREP_KIND_POLYNOMIAL or plan.kind == PREP_KIND_QUANTILE or plan.kind == PREP_KIND_POWER:
        return _fitted_state(fields, context, plan)
    if plan.kind != PREP_KIND_ROBUST and not maxabs:
        raise Error("ValueError: mojolearn: Prep member kind " + String(plan.kind) + " is not published")
    if (not py_contains(fields, "n_features_in_")
            or not py_contains(fields, "max_abs_" if maxabs else "center_")):
        raise Error("NotFittedError: this member is not fitted yet")
    var bi = native_loaded_module("builtins")
    var estimator = String("MaxAbsScaler") if maxabs else String("RobustScaler")
    if py_type(fields["n_features_in_"]) is not bi.int:
        raise Error("TypeError: " + estimator + " n_features_in_ must be a native integer")
    var d = Int(py=fields["n_features_in_"])
    var scalars: List[Float64] = [Float64(d)]
    if maxabs:
        if not py_contains(fields, "n_samples_seen_") or py_type(fields["n_samples_seen_"]) is not bi.int:
            raise Error("TypeError: MaxAbsScaler n_samples_seen_ must be a native integer")
        scalars.append(Float64(Int(py=fields["n_samples_seen_"])))
        if fields["max_abs_"] is None or fields["scale_"] is None:
            raise Error("NotFittedError: this MaxAbsScaler member needs max_abs_ and scale_")
    var call = member_new(plan)
    try:
        _import(plan, call, fields, "max_abs_" if maxabs else "center_", PREP_BLOCK_CENTER, d, context)
        _import(plan, call, fields, "scale_", PREP_BLOCK_SCALE, d, context)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, len(scalars), MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = len(scalars)
    return call.state


def _stateless_state(fields: PythonObject, plan: MemberPlan) raises -> Int:
    # A fitted Binarizer / Normalizer is its width.
    if not py_contains(fields, "n_features_in_"):
        raise Error("NotFittedError: this member is not fitted yet")
    if py_type(fields["n_features_in_"]) is not native_loaded_module("builtins").int:
        raise Error("TypeError: n_features_in_ must be a native integer")
    var scalars: List[Float64] = [Float64(Int(py=fields["n_features_in_"]))]
    var call = member_new(plan)
    try:
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, 1, MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = len(scalars)
    return call.state


def _int_field(fields: PythonObject, name: String) raises -> Int:
    if not py_contains(fields, name):
        raise Error("NotFittedError: this member is not fitted yet")
    if py_type(fields[name]) is not native_loaded_module("builtins").int:
        raise Error("TypeError: " + name + " must be a native integer")
    return Int(py=fields[name])


def _import_vector(plan: MemberPlan, mut call: MemberCall, source: PythonObject, block: Int, count: Int,
    context: PythonObject) raises:
    # One fitted vector (None: an empty block), its size checked.
    call.block = block
    call.flags = 0
    if source is None:
        call.x = MemberMatrix(0, 1, 0)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        return
    var vector = prep_fitted_vector(source, context)
    if Int(py=vector.size) != count:
        raise Error("RuntimeError: mojolearn: a fitted attribute does not match the fitted sizes")
    call.x = MemberMatrix(Int(py=classical_address(vector)), 1, count)
    _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    _ = vector


def _fitted_state(fields: PythonObject, context: PythonObject, plan: MemberPlan) raises -> Int:
    # PolynomialFeatures / QuantileTransformer / PowerTransformer: the
    # attributes the stateless door's transform reads.
    var d = _int_field(fields, "n_features_in_")
    var scalars: List[Float64] = [Float64(d)]
    if plan.kind == PREP_KIND_POLYNOMIAL:
        scalars.append(Float64(_int_field(fields, "n_output_features_")))
    elif plan.kind == PREP_KIND_QUANTILE:
        scalars.append(Float64(_int_field(fields, "n_quantiles_")))
    var call = member_new(plan)
    try:
        if plan.kind == PREP_KIND_POLYNOMIAL:
            var index = prep_fitted_vector(fields["_poly_index"], context)
            _import_vector(plan, call, index, PREP_BLOCK_CENTER, Int(py=index.size), context)
            _import_vector(plan, call, fields["_poly_start"], PREP_BLOCK_SCALE, Int(scalars[1]) + 1, context)
        elif plan.kind == PREP_KIND_QUANTILE:
            var nq = Int(scalars[1])
            _import_vector(plan, call, fields["_q"], PREP_BLOCK_CENTER, nq * d, context)
            _import_vector(plan, call, fields["references_"], PREP_BLOCK_SCALE, nq, context)
            # quantiles_ is (n_quantiles_, features): read as a C-order float32 matrix.
            var quantiles = scaler_input(fields["quantiles_"], 2, True, context)
            if Int(py=quantiles.size) != nq * d:
                raise Error("RuntimeError: QuantileTransformer fitted quantiles do not match n_quantiles_")
            call.block = PREP_BLOCK_EXTRA
            call.flags = 0
            call.x = MemberMatrix(Int(py=classical_address(quantiles)), 1, nq * d)
            _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
            _ = quantiles
        else:
            _import_vector(plan, call, fields["lambdas_"], PREP_BLOCK_EXTRA, d, context)
            _import_vector(plan, call, fields["_mean"], PREP_BLOCK_CENTER, d, context)
            _import_vector(plan, call, fields["_scale"], PREP_BLOCK_SCALE, d, context)
            _import_vector(plan, call, py_get(fields, "_pt_anchor"), PREP_BLOCK_ANCHOR, d, context)
            _import_vector(plan, call, py_get(fields, "_pt_anchor_kind"), PREP_BLOCK_ANCHOR_KIND, d, context)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, len(scalars), MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = len(scalars)
    return call.state
