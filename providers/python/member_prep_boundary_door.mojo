# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the Prep family: its place in
`bindings/member_boundary_door.mojo` (recipe: MEMBER-ABI-V2.md; RUN:
core/prep_member.mojo). Published: RobustScaler (kind 0,
`_mojolearn_member = (35, 0)`), MaxAbsScaler (kind 1, `(35, 1)`; not on the
Apple fast direct fit entry, where admit answers False and the caller keeps
the door), Binarizer (kind 2, `(35, 2)`), Normalizer (kind 3, `(35, 3)`),
PolynomialFeatures (kind 4, `(35, 4)`), QuantileTransformer (kind 5,
`(35, 5)`; admit answers False for a subsample below 1, which only the door
handles), PowerTransformer (kind 6, `(35, 6)`), SimpleImputer (kind 7, `(35, 7)`),
KBinsDiscretizer (kind 8, `(35, 8)`; admit answers False for a subsample
or random_state that is not an integer), SplineTransformer (kind 9,
`(35, 9)`), OrdinalEncoder (kind 10, `(35, 10)`; admit answers False for
categories other than 'auto', for infrequent grouping (min_frequency or
max_categories set) and on the FAST image whose packed category route the
door tries first: the caller keeps the door), OneHotEncoder (kind 11,
`(35, 11)`; False for the same settings, for a `drop` given per column and
for every setting whose transform warns: handle_unknown='warn', or
'ignore' / 'infrequent_if_exist' with a drop; the door's UserWarning is
raised only on its own route, since a member transform has no PACK here
and the notice word of lane fx-audit is not on integration). The family's other classes
(TargetEncoder IterativeImputer) carry no mark (each blocker in the lane's
RESULT file): a caller keeps their own doors, and admit refuses their kinds
in words.

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
from bindings.prep_encode_native_door import (
    enc_caps, _missing_mark as enc_missing_mark, _qselect as enc_qselect, _truth as enc_truth,
    _float as enc_float, _is_str as enc_is_str, _real_like as enc_real_like, _int_list as enc_int_list,
    IDN_STATS_BLOCKED, _kbins_strategy as enc_kbins_strategy, _seed64 as enc_seed64,
    _spline_checks as enc_spline_checks, _extrapolation as enc_extrapolation, _spline_fused as enc_spline_fused,
    _knot_rows as enc_knot_rows, _f32 as enc_f32,
    _check_infrequent_params as enc_check_infrequent, _fast_unique as enc_fast_unique, _cls2_cat as enc_cls2_cat,
    _c08_route as enc_c08_route, _category_record as enc_category_record, _index_list as enc_index_list,
    encode_auto_categories, encode_drop_word, encode_auto_drops,
)
from bindings.prep_encode_cats_run import CatAuto
from bindings.prep_encode_onehot_run import onehot_auto_drops, OH_DROP_NONE
from bindings.prep_stateless_native_door import _int_like as stateless_int_like
from bindings.py_typed_call import py_call_typed
from bindings.prep_encode_imputer_run import IMP_MEAN, IMP_MEDIAN, IMP_MOST_FREQUENT, IMP_CONSTANT
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
    PREP_KIND_IMPUTER, PREP_IMPUTER_PARAM_COUNT,
    PREP_KIND_KBINS, PREP_KBINS_HEADER, PREP_KBINS_ONEHOT, PREP_KBINS_ONEHOT_DENSE, PREP_KBINS_ORDINAL,
    PREP_KIND_SPLINE, PREP_SPLINE_HEADER, PREP_KIND_ORDINAL, PREP_ORDINAL_PARAM_COUNT,
    PREP_KIND_ONEHOT, PREP_ONEHOT_PARAM_COUNT, PREP_ONEHOT_ERROR, PREP_ONEHOT_IGNORE, PREP_ONEHOT_INFREQUENT,
    PREP_ROUTE_OUT, PREP_ROUTE_SCRATCH, PREP_ROUTE_RANGES, PREP_ROUTE_DEV_PUT, PREP_ROUTE_TAKE_ROWS,
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
        # The device row gather of a fold (the door's DeviceRows test, read
        # once here; the door per run).
        if py_get(entries, "x_prep_dev_take_rows") is not None:
            flags |= PREP_ROUTE_TAKE_ROWS
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


def _imputer_admit(model: PythonObject, fields: PythonObject, context: PythonObject, mut owner: MemberOwner) raises:
    # The encode door's `_imputer_fit` UNPACK, in its order, read once here.
    var strategy_field = fields["strategy"]
    var name = String(strategy_field) if enc_is_str(strategy_field) else String("")
    var strategy = IMP_MEAN
    if name == "median":
        strategy = IMP_MEDIAN
    elif name == "most_frequent":
        strategy = IMP_MOST_FREQUENT
    elif name == "constant":
        strategy = IMP_CONSTANT
    elif name != "mean":
        raise Error("mojolearn: SimpleImputer strategy " + stateless_repr(strategy_field) + " is not valid")
    var konst = strategy == IMP_CONSTANT
    var fill_value = fields["fill_value"]
    if konst and fill_value is not None and not enc_real_like(fill_value, context):
        raise Error("TypeError: mojolearn: SimpleImputer fill_value must be numeric")
    var mode = context["backend_fields"]["_DEFAULT_MODE"]
    var entries = scaler_entries(model, mode, context)
    var mark = enc_missing_mark(fields["missing_values"])
    var words = List[Float64](length=PREP_IMPUTER_PARAM_COUNT, fill=0.0)
    words[0] = Float64(PREP_KIND_IMPUTER)
    words[1] = Float64(strategy)
    words[2] = 1.0 if mark[0] else 0.0
    words[3] = mark[1]
    words[4] = 1.0 if strategy == IMP_MEDIAN and enc_qselect(mode, entries, context) else 0.0
    words[5] = 1.0 if (stateless_fam(mode, entries) & IDN_STATS_BLOCKED) != 0 else 0.0
    words[6] = enc_float(fill_value) if konst and fill_value is not None else 0.0
    words[7] = 1.0 if enc_truth(fields["keep_empty_features"]) else 0.0
    words[8] = 1.0 if enc_truth(fields["add_indicator"]) else 0.0
    words[9] = 1.0 if xprep_r3("work") else 0.0
    _route_admit(model, entries, context, owner, enc_caps(model, entries), False)
    owner.plan.params = words^


def _kbins_admit(model: PythonObject, fields: PythonObject, context: PythonObject, mut owner: MemberOwner) raises -> Bool:
    # The encode door's `_kbins_fit` UNPACK, in its order, read once here
    # (the door per fit). False (the caller keeps the door): a random_state
    # or subsample that is not an integer (the door reads them only when it
    # draws, which depends on the rows).
    var b = native_loaded_module("builtins")
    var encode = fields["encode"]
    var encode_text = String(encode) if enc_is_str(encode) else String("")
    if encode_text != "onehot" and encode_text != "onehot-dense" and encode_text != "ordinal":
        raise Error("mojolearn: invalid encode " + stateless_repr(encode))
    var strategy = enc_kbins_strategy(fields)
    var mode = context["backend_fields"]["_DEFAULT_MODE"]
    var entries = scaler_entries(model, mode, context)
    var words = List[Float64](length=PREP_KBINS_HEADER, fill=0.0)
    words[0] = Float64(PREP_KIND_KBINS)
    words[1] = Float64(PREP_KBINS_ORDINAL if encode_text == "ordinal" else (
        PREP_KBINS_ONEHOT_DENSE if encode_text == "onehot-dense" else PREP_KBINS_ONEHOT))
    words[2] = Float64(strategy)
    words[3] = -1.0
    var subsample = fields["subsample"]
    if subsample is not None:
        if not stateless_int_like(subsample, context):
            return False
        words[3] = Float64(Int(py=py_call_typed(b.int, subsample)))
        var state = fields["random_state"]
        if state is not None and not stateless_int_like(state, context):
            return False
        var seed = enc_seed64(state)
        words[4] = Float64(seed[0])
        words[5] = Float64(seed[1])
    words[6] = 1.0 if (stateless_fam(mode, entries) & IDN_STATS_BLOCKED) != 0 else 0.0
    words[7] = 1.0 if xprep_r3("work") else 0.0
    var n_bins = fields["n_bins"]
    if stateless_int_like(n_bins, context):
        words[9] = Float64(Int(py=py_call_typed(b.int, n_bins)))
    else:
        var items = py_call_typed(b.list, n_bins)
        words[8] = Float64(len(items))
        for i in range(len(items)):  # driver-loop: over the per-column n_bins argument
            words.append(Float64(Int(py=py_call_typed(b.int, items[i]))))
    _route_admit(model, entries, context, owner, enc_caps(model, entries), False)
    owner.plan.params = words^
    return True


def _spline_admit(model: PythonObject, fields: PythonObject, context: PythonObject, mut owner: MemberOwner) raises:
    # The encode door's `_spline_fit` UNPACK, in its order, read once here
    # (the door per fit): `_spline_checks`, the extrapolation, the degree,
    # the knots argument (the user's rows as float32 words; their shape and
    # order are checked at the fit, where the feature count is known).
    var b = native_loaded_module("builtins")
    enc_spline_checks(fields)
    var extrap_field = fields["extrapolation"]
    var extrap = enc_extrapolation(extrap_field)
    var words = List[Float64](length=PREP_SPLINE_HEADER, fill=0.0)
    words[0] = Float64(PREP_KIND_SPLINE)
    words[1] = Float64(Int(py=py_call_typed(b.int, fields["degree"])))
    words[4] = Float64(extrap)
    words[5] = 1.0 if String(extrap_field) == "error" else 0.0
    words[6] = 1.0 if String(fields["handle_missing"]) == "error" else 0.0
    words[7] = 1.0 if enc_truth(fields["include_bias"]) else 0.0
    words[11] = 1.0 if String(fields["order"]) == "F" else 0.0
    var knots_arg = fields["knots"]
    if enc_is_str(knots_arg):
        var knot_kind = String(knots_arg)
        if knot_kind != "uniform" and knot_kind != "quantile":
            raise Error("mojolearn: invalid knots " + stateless_repr(knots_arg))
        words[2] = 1.0 if knot_kind == "quantile" else 2.0
        words[3] = Float64(Int(py=py_call_typed(b.int, fields["n_knots"])))
    else:
        var rows = enc_knot_rows(knots_arg, context)
        var r = len(rows)
        var c = len(rows[0]) if r > 0 else 0
        for i in range(r):  # driver-loop: over the user knots rows
            if len(rows[i]) != c:
                c = -1
        words[2] = 0.0
        words[12] = Float64(r)
        words[13] = Float64(c)
        if c > 0:
            for i in range(r):  # driver-loop: over the user knots rows
                for j in range(c):  # driver-loop: over one user knots row
                    words.append(Float64(enc_f32(rows[i][j])))
    var mode = context["backend_fields"]["_DEFAULT_MODE"]
    var entries = scaler_entries(model, mode, context)
    words[8] = 1.0 if enc_spline_fused(mode, entries) else 0.0
    words[9] = 1.0 if (stateless_fam(mode, entries) & IDN_STATS_BLOCKED) != 0 else 0.0
    words[10] = 1.0 if xprep_r3("work") else 0.0
    _route_admit(model, entries, context, owner, enc_caps(model, entries), False)
    owner.plan.params = words^


def _ordinal_admit(model: PythonObject, fields: PythonObject, context: PythonObject, mut owner: MemberOwner) raises -> Bool:
    # The encode door's `_precheck` and `_ordinal_fit` UNPACK, in its order
    # and words, read once here (the door per fit). False (the caller keeps
    # the door): categories not 'auto', infrequent grouping, the FAST image's
    # packed category route (`_auto_fit` tries it first).
    var grouping = enc_check_infrequent(fields, context)
    var handle = fields["handle_unknown"]
    var handle_text = String(handle) if enc_is_str(handle) else String("")
    if handle_text != "error" and handle_text != "use_encoded_value":
        raise Error("mojolearn: invalid handle_unknown " + stateless_repr(handle))
    var use = handle_text == "use_encoded_value"
    var unknown_value = fields["unknown_value"]
    if use and not enc_real_like(unknown_value, context):
        raise Error("TypeError: mojolearn: unknown_value must be a number when handle_unknown='use_encoded_value'")
    var emv = fields["encoded_missing_value"]
    if not enc_real_like(emv, context):
        raise Error("TypeError: mojolearn: encoded_missing_value must be a number (or NaN)")
    var categories = fields["categories"]
    if grouping or not (enc_is_str(categories) and String(categories) == "auto"):
        return False
    var mode = context["backend_fields"]["_DEFAULT_MODE"]
    var entries = scaler_entries(model, mode, context)
    var fast = enc_fast_unique(mode, entries)
    if (enc_cls2_cat(mode, entries) & 1) != 0 and fast:
        return False
    var words = List[Float64](length=PREP_ORDINAL_PARAM_COUNT, fill=0.0)
    words[0] = Float64(PREP_KIND_ORDINAL)
    words[1] = 1.0 if fast else 0.0  # CAT_FIT_SCAN, else CAT_FIT_UNIQUE
    words[2] = 1.0 if enc_c08_route(mode, entries, 4) else 0.0
    words[3] = 1.0 if xprep_r3("work") else 0.0
    words[4] = 1.0 if use else 0.0
    if use:
        words[5] = enc_float(unknown_value)
        words[6] = 1.0 if stateless_int_like(unknown_value, context) else 0.0
    words[7] = enc_float(emv)
    words[8] = 1.0 if stateless_int_like(emv, context) else 0.0
    _route_admit(model, entries, context, owner, enc_caps(model, entries), False)
    owner.plan.params = words^
    return True


def _onehot_admit(model: PythonObject, fields: PythonObject, context: PythonObject, mut owner: MemberOwner) raises -> Bool:
    # The encode door's `_precheck` and `_onehot_fit` UNPACK, in its order
    # and words, read once here. False (the caller keeps the door): the
    # ordinal admit's settings, a drop given per column, and every setting
    # whose transform warns.
    var grouping = enc_check_infrequent(fields, context)
    var handle = fields["handle_unknown"]
    var text = String(handle) if enc_is_str(handle) else String("")
    if text != "error" and text != "ignore" and text != "infrequent_if_exist" and text != "warn":
        raise Error("mojolearn: OneHotEncoder handle_unknown=" + stateless_repr(handle) + " is not valid")
    var drop = fields["drop"]
    if enc_is_str(drop) and String(drop) != "first" and String(drop) != "if_binary":
        raise Error("mojolearn: OneHotEncoder drop must be None, 'first', 'if_binary' or one category "
                    + "per feature")
    var categories = fields["categories"]
    if grouping or not (enc_is_str(categories) and String(categories) == "auto"):
        return False
    var word = encode_drop_word(drop)
    if word < 0 or text == "warn" or (word != OH_DROP_NONE and text != "error"):
        return False
    var mode = context["backend_fields"]["_DEFAULT_MODE"]
    var entries = scaler_entries(model, mode, context)
    var fast = enc_fast_unique(mode, entries)
    if (enc_cls2_cat(mode, entries) & 1) != 0 and fast:
        return False
    var words = List[Float64](length=PREP_ONEHOT_PARAM_COUNT, fill=0.0)
    words[0] = Float64(PREP_KIND_ONEHOT)
    words[1] = 1.0 if fast else 0.0  # CAT_FIT_SCAN, else CAT_FIT_UNIQUE
    words[2] = 1.0 if enc_c08_route(mode, entries, 2) else 0.0
    words[3] = 1.0 if xprep_r3("work") else 0.0
    words[4] = Float64(PREP_ONEHOT_ERROR if text == "error" else (
        PREP_ONEHOT_IGNORE if text == "ignore" else PREP_ONEHOT_INFREQUENT))
    words[5] = Float64(word)
    _route_admit(model, entries, context, owner, enc_caps(model, entries), False)
    owner.plan.params = words^
    return True


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
    if kind == PREP_KIND_IMPUTER:
        _imputer_admit(model, fields, context, owner)
        return True
    if kind == PREP_KIND_KBINS:
        return _kbins_admit(model, fields, context, owner)
    if kind == PREP_KIND_SPLINE:
        _spline_admit(model, fields, context, owner)
        return True
    if kind == PREP_KIND_ORDINAL:
        return _ordinal_admit(model, fields, context, owner)
    if kind == PREP_KIND_ONEHOT:
        return _onehot_admit(model, fields, context, owner)
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
    if plan.kind < PREP_KIND_ROBUST or plan.kind > PREP_KIND_ONEHOT:
        raise Error("ValueError: mojolearn: Prep member kind " + String(plan.kind) + " is not published")
    var call = MemberCall(plan.route, state)
    var scalars = _export(plan, call, MEMBER_BLOCK_SCALARS)
    var words = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=scalars.data)
    var d = Int(words[0])
    if plan.kind == PREP_KIND_IMPUTER:
        _pack_imputer(fields, context, plan, call, d, words[1] != 0.0)
        return
    if plan.kind == PREP_KIND_KBINS:
        _pack_kbins(fields, context, plan, call, d, Int(words[1]))
        return
    if plan.kind == PREP_KIND_SPLINE:
        _pack_spline(fields, context, plan, call, d, Int(words[1]), Int(words[2]), Int(words[3]))
        return
    if plan.kind == PREP_KIND_ORDINAL or plan.kind == PREP_KIND_ONEHOT:
        _pack_encoder(model, fields, context, plan, call, d, Int(words[1]))
        return
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


def _pack_kbins(fields: PythonObject, context: PythonObject, plan: MemberPlan, mut call: MemberCall, d: Int,
    stride: Int) raises:
    """The encode door's `_kbins_fit` PACK, in its order: bin_edges_ (one
    vector of n_bins_ + 1 edges per column), n_bins_ (int64), the flat
    `_edges` / `_ne` the door's transform reads, `_stride`."""
    var edges = _export(plan, call, PREP_BLOCK_CENTER)
    var counts = _export(plan, call, PREP_BLOCK_SCALE)
    if edges.rows * edges.cols != d * stride or counts.rows * counts.cols != d:
        raise Error("RuntimeError: mojolearn: a fitted KBinsDiscretizer block has the wrong size")
    var output = context["output_context"]
    var bin_edges = Python.list()
    var n_bins_ = classical_typed(Python.tuple(d), "<i8", output)
    var dest = I64P(unsafe_from_address=Int(py=classical_address(n_bins_)))
    var ne = f32_ptr(counts.data)
    for j in range(d):  # driver-loop: over the per-column edge counts (feature metadata)
        var count = Int(ne[j])
        _ = py_call(bin_edges.append, _vector(MemberMatrix(edges.data + 4 * j * stride, 1, count), context))
        dest[j] = Int64(count - 1)
        _ = n_bins_  # keep the Python object alive through the native write above
    py_setitem(fields, "bin_edges_", bin_edges)
    py_setitem(fields, "n_bins_", n_bins_)
    py_setitem(fields, "_edges", _vector(edges, context))
    py_setitem(fields, "_ne", _vector(counts, context))
    py_setitem(fields, "_stride", stride)
    py_setitem(fields, "numeric_mode_", context["backend_fields"]["_DEFAULT_MODE"])
    py_setitem(fields, "n_features_in_", d)


def _encoder_state_of(plan: MemberPlan, mut call: MemberCall, d: Int, kmax: Int) raises -> CatAuto:
    # The fitted category state from the member's blocks (the docstring's table).
    var block = _export(plan, call, PREP_BLOCK_CENTER)
    var sizes = _export(plan, call, PREP_BLOCK_SCALE)
    var missing = _export(plan, call, PREP_BLOCK_EXTRA)
    if (block.rows * block.cols != max(d * kmax, 1) or sizes.rows * sizes.cols != d
            or missing.rows * missing.cols != d):
        raise Error("RuntimeError: mojolearn: a fitted encoder block has the wrong size")
    var state = CatAuto()
    state.d = d
    state.kmax = kmax
    state.block = List[Float32](length=block.rows * block.cols, fill=Float32(0))
    unsafe_memcpy(dest=f32_ptr(Int(state.block.unsafe_ptr())), src=f32_ptr(block.data), count=len(state.block))
    var sp = f32_ptr(sizes.data)
    var mp = f32_ptr(missing.data)
    for j in range(d):  # driver-loop: over the per-column category counts (feature metadata)
        state.sizes.append(Int(sp[j]))
        state.missing.append(Int(mp[j]))
    return state^


def _pack_encoder(model: PythonObject, fields: PythonObject, context: PythonObject, plan: MemberPlan,
    mut call: MemberCall, d: Int, kmax: Int) raises:
    """The encode door's restricted fit PACK (bindings/prep_encode_native_door.mojo
    `encode_auto_categories`: categories_, the category record, `_missing`,
    `_infrequent` / `_grouping` None; OneHotEncoder's `_drop_after` /
    `drop_idx_`), then its fields, in its order."""
    var state = _encoder_state_of(plan, call, d, kmax)
    encode_auto_categories(model, fields, state, context)
    if plan.kind == PREP_KIND_ONEHOT:
        # `_onehot_fit`'s drops: the drop word's indices (the RUN the member's
        # fit ran on the same sizes), as `_drop_after` / `drop_idx_`.
        var word = Int(plan.params[5])
        encode_auto_drops(fields, onehot_auto_drops(state.sizes, word), word == OH_DROP_NONE)
    py_setitem(fields, "numeric_mode_", context["backend_fields"]["_DEFAULT_MODE"])
    py_setitem(fields, "n_features_in_", d)


def _pack_spline(fields: PythonObject, context: PythonObject, plan: MemberPlan, mut call: MemberCall, d: Int,
    nk: Int, k: Int, features: Int) raises:
    """The encode door's `_spline_fit` PACK, in its order: `_knots`, then
    per column bsplines_ (its knot row), `_lo`, `_hi`; `_nk`, `_k`,
    n_features_out_, numeric_mode_, n_features_in_."""
    var wd = nk + 2 * k
    var knots = _export(plan, call, PREP_BLOCK_CENTER)
    if knots.rows * knots.cols != d * wd:
        raise Error("RuntimeError: mojolearn: a fitted SplineTransformer block has the wrong size")
    py_setitem(fields, "_knots", _vector(knots, context))
    var bsplines = Python.list()
    var lows = Python.list()
    var highs = Python.list()
    var words = f32_ptr(knots.data)
    for c in range(d):  # driver-loop: per-column fitted spline knots (feature count)
        _ = py_call(bsplines.append, _vector(MemberMatrix(knots.data + 4 * c * wd, 1, wd), context))
        _ = py_call(lows.append, PythonObject(Float64(words[c * wd + k])))
        _ = py_call(highs.append, PythonObject(Float64(words[c * wd + k + nk - 1])))
    py_setitem(fields, "bsplines_", bsplines)
    py_setitem(fields, "_lo", lows)
    py_setitem(fields, "_hi", highs)
    py_setitem(fields, "_nk", nk)
    py_setitem(fields, "_k", k)
    py_setitem(fields, "n_features_out_", features)
    py_setitem(fields, "numeric_mode_", context["backend_fields"]["_DEFAULT_MODE"])
    py_setitem(fields, "n_features_in_", d)


def _spline_state(fields: PythonObject, context: PythonObject, plan: MemberPlan) raises -> Int:
    # The attributes the encode door's `_spline_transform` reads.
    var d = _int_field(fields, "n_features_in_")
    var nk = _int_field(fields, "_nk")
    var k = _int_field(fields, "_k")
    var scalars: List[Float64] = [Float64(d), Float64(nk), Float64(k), Float64(_int_field(fields, "n_features_out_"))]
    var call = member_new(plan)
    try:
        _import_vector(plan, call, fields["_knots"], PREP_BLOCK_CENTER, d * (nk + 2 * k), context)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, len(scalars), MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = len(scalars)
    return call.state


def _kbins_state(fields: PythonObject, context: PythonObject, plan: MemberPlan) raises -> Int:
    # The attributes the encode door's `_kbins_transform` reads.
    var d = _int_field(fields, "n_features_in_")
    var stride = _int_field(fields, "_stride")
    var scalars: List[Float64] = [Float64(d), Float64(stride)]
    var call = member_new(plan)
    try:
        _import_vector(plan, call, fields["_edges"], PREP_BLOCK_CENTER, d * stride, context)
        _import_vector(plan, call, fields["_ne"], PREP_BLOCK_SCALE, d, context)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, len(scalars), MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = len(scalars)
    return call.state


def _encoder_state(model: PythonObject, fields: PythonObject, context: PythonObject, plan: MemberPlan) raises -> Int:
    """The attributes the encode door's transform reads: the category record
    of `categories_` (`_category_record`: kept in the prepared-owner registry,
    never in the instance dictionary) and `_missing`. A fitted infrequent
    grouping is not the restricted form: refused in the member ABI's words
    (the caller keeps the door)."""
    var d = _int_field(fields, "n_features_in_")
    if not py_contains(fields, "categories_") or not py_contains(fields, "_missing"):
        raise Error("NotFittedError: this member is not fitted yet")
    if not py_contains(fields, "_grouping") or fields["_grouping"] is not None:
        raise Error("NotImplementedError: member ABI: an encoder member serves no infrequent grouping")
    var record = enc_category_record(model, fields["categories_"], context)
    if len(record[4]) != d:
        raise Error("RuntimeError: mojolearn: a fitted attribute does not match the fitted sizes")
    var kmax = Int(py=record[3])
    var missing_list = enc_index_list(fields["_missing"])
    if len(missing_list) != d:
        raise Error("RuntimeError: mojolearn: a fitted attribute does not match the fitted sizes")
    var missing = List[Float32](capacity=d)
    for j in range(d):  # driver-loop: over the per-column NaN slots (feature metadata)
        missing.append(Float32(missing_list[j]))
    # OneHotEncoder: `_drop_after` (None: no drop), as the door's `_onehot_drops`.
    var drops = List[Float32](length=d, fill=Float32(-1))
    if plan.kind == PREP_KIND_ONEHOT:
        if not py_contains(fields, "_drop_after"):
            raise Error("NotFittedError: this member is not fitted yet")
        var after = fields["_drop_after"]
        if after is not None:
            if len(after) != d:
                raise Error("RuntimeError: mojolearn: a fitted attribute does not match the fitted sizes")
            for j in range(d):  # driver-loop: over the per-column drop indices (feature metadata)
                if after[j] is not None:
                    drops[j] = Float32(Int(py=after[j]))
    var scalars: List[Float64] = [Float64(d), Float64(kmax)]
    var call = member_new(plan)
    try:
        _import_vector(plan, call, record[2], PREP_BLOCK_CENTER, Int(py=record[6]), context)
        _import_vector(plan, call, record[5], PREP_BLOCK_SCALE, d, context)
        call.block = PREP_BLOCK_EXTRA
        call.flags = 0
        call.x = MemberMatrix(Int(missing.unsafe_ptr()), 1, d)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        if plan.kind == PREP_KIND_ONEHOT:
            call.block = PREP_BLOCK_ANCHOR
            call.x = MemberMatrix(Int(drops.unsafe_ptr()), 1, d)
            _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, len(scalars), MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = len(scalars)
    _ = len(missing)
    _ = len(drops)
    _ = record
    return call.state


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


def _index_list(view: MemberMatrix) raises -> List[Int]:
    var out = List[Int](capacity=view.rows * view.cols)
    var words = f32_ptr(view.data)
    for i in range(view.rows * view.cols):  # driver-loop: over fitted column indices (feature metadata)
        out.append(Int(words[i]))
    return out^


def _pack_imputer(fields: PythonObject, context: PythonObject, plan: MemberPlan, mut call: MemberCall, d: Int,
    shared: Bool) raises:
    """The encode door's `_imputer_fit` PACK, in its order (statistics_ and
    `_fill` are one object unless the strategy is constant or a column was
    empty)."""
    var statistics = _vector(_export(plan, call, PREP_BLOCK_CENTER), context)
    py_setitem(fields, "statistics_", statistics)
    py_setitem(fields, "_fill", statistics if shared else _vector(_export(plan, call, PREP_BLOCK_SCALE), context))
    py_setitem(fields, "_keep", enc_int_list(_index_list(_export(plan, call, PREP_BLOCK_EXTRA))))
    py_setitem(fields, "_indicator", enc_int_list(_index_list(_export(plan, call, PREP_BLOCK_ANCHOR))))
    py_setitem(fields, "numeric_mode_", context["backend_fields"]["_DEFAULT_MODE"])
    py_setitem(fields, "n_features_in_", d)


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
    if plan.kind == PREP_KIND_IMPUTER:
        return _imputer_state(fields, context, plan)
    if plan.kind == PREP_KIND_KBINS:
        return _kbins_state(fields, context, plan)
    if plan.kind == PREP_KIND_SPLINE:
        return _spline_state(fields, context, plan)
    if plan.kind == PREP_KIND_ORDINAL or plan.kind == PREP_KIND_ONEHOT:
        return _encoder_state(model, fields, context, plan)
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


def _import_indices(plan: MemberPlan, mut call: MemberCall, value: PythonObject, block: Int, d: Int) raises:
    # A fitted list of column indices (`_keep`, `_indicator`) as float32 words.
    var b = native_loaded_module("builtins")
    var items = py_call(b.list, value)
    var words = List[Float32](capacity=len(items))
    for i in range(len(items)):  # driver-loop: over fitted column indices (feature metadata)
        var j = Int(py=py_call(b.int, items[i]))
        if j < 0 or j >= d:
            raise Error("RuntimeError: mojolearn: a SimpleImputer fitted column index is out of range")
        words.append(Float32(j))
    call.block = block
    call.flags = 0
    call.x = MemberMatrix(Int(words.unsafe_ptr()), 1, len(words))
    _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    _ = len(words)


def _imputer_state(fields: PythonObject, context: PythonObject, plan: MemberPlan) raises -> Int:
    # The attributes the encode door's `_imputer_transform` reads.
    var d = _int_field(fields, "n_features_in_")
    var shared = fields["statistics_"] is fields["_fill"]
    var scalars: List[Float64] = [Float64(d), 1.0 if shared else 0.0]
    var call = member_new(plan)
    try:
        _import_vector(plan, call, fields["statistics_"], PREP_BLOCK_CENTER, d, context)
        _import_vector(plan, call, fields["_fill"], PREP_BLOCK_SCALE, d, context)
        _import_indices(plan, call, fields["_keep"], PREP_BLOCK_EXTRA, d)
        _import_indices(plan, call, fields["_indicator"], PREP_BLOCK_ANCHOR, d)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, len(scalars), MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = len(scalars)
    return call.state
