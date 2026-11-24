# SPDX-License-Identifier: Apache-2.0
# The native owner of IterativeImputer (dispatched by bindings/prep_owner_native.mojo).
# fit, fit_transform and transform are each one call here: UNPACK (the
# parameter refusals, the selection, X, the initial SimpleImputer's words and,
# for transform, the fitted fields and the addresses of the fitted vectors),
# the typed RUN (x_prep/iterimp_run.mojo `iterimp_fit` / `iterimp_transform`:
# the initial fill, the statistics and orders, the n_nearest_features draws,
# every round of the built-in Bayesian ridge in one program, the indicator),
# then PACK (the fitted attributes, read from the arenas the runner holds).
# A user estimator whose class is a native member (core/member_abi.mojo; lane
# mf-iterimp, 2026-11-23) is admitted ONCE per fit (`_member`: its family's
# member entry, the kind's mask) and its rounds are the typed RUN
# x_prep/iterimp_run.mojo `iterimp_member_rounds` (per step: the gather
# program, the member's FIT on the gathered rows and PREDICT, or PREDICT_STD
# and the mean for sample_posterior, then the scatter program), the member
# calls with the GIL released for a kind that allows it; PACK writes each
# step's state into a clone of the estimator (`member_pack`: the attributes
# its own fit writes) for `imputation_sequence_`, HANDS the state to that
# clone (`member_held_adopt`, lane mf-iterimp2) and raises the fits'
# notices. transform with such an estimator (lane mf-iterimp2) reads each
# fitted step's held state once (`_held_steps`: a hit is a record read, no
# import) and runs `iterimp_member_transform` (per step the gather program,
# the member's PREDICT / PREDICT_STD on the held state, the scatter
# program). sample_posterior goes through the member route for the
# families whose PREDICT_STD also answers the same query's mean
# (`_posterior_family`: GaussianProcessRegressor, BayesianRidge,
# ARDRegression; these are the only plans that serve predict_std, so
# `_refuse` refuses sample_posterior for every other estimator first).
# Lane mf-iterimp3: a CLASSIFIER member is served too (each step's classes
# and codes made natively, the labels published as the classifier's own fit
# publishes them, kept as `_step_classes` for transform); fewer than two
# kept features is scikit-learn's edge case (n_iter_ = 0, the initial
# imputation, no estimator fitted: `iterimp_initial_only`); a kept column
# with no observed value is marked not missing as scikit-learn marks it
# (`_empty_kept`; every route). Lane mf-iterimp3b: every explicit Host*
# class whose public class has a member mark carries its own
# (python/mojolearn/_classical_host.py), so HostLinearRegression / HostRidge
# run as linear members on their CPU context, and a MEMBER-ONLY plan whose
# member does not serve the fit is refused in its pre-mark words
# (`_refuse_member_only`), never sent to the door rounds. The door rounds
# below stay ONLY for: a class with no member entry (the meta-estimators of
# the Ensemble, Search and Pipeline families, whose member entries are
# unpublished until the META member kind lands), an admission or a
# held-state import the family refuses for a class with a full plan (since
# lane mf-split4 / mf-cal5 every class_weight the fit door accepts is read by
# the member: a str or dict subclass, a value equal to 'balanced'; since lane mf-left4 a class_weight the estimator
# refuses and a dict key that is not a number are native: their words are
# raised by the RUN's first classifier fit; an image without the
# member's entry, an unaddressable GP kernel, the gbdt adapter's fast mode
# and categorical or multi-target losses, a CV splitter object: the
# fallback rule, a mark never turns a call that worked into a refusal; lane
# mf-adm3 serves class_weight 'balanced' and numeric-keyed dicts, read per
# step over the step's classes, and kernel='precomputed'). (The member-set B
# arm, MOJOLEARN_MEMBER_SET_OFF, was deleted by lane/ab-prep-classical: these
# rounds are no longer its off path; lane mf-del5.) There the estimator is cloned, fitted and
# asked to predict once per (round, feature) step with the float32 blocks
# the gather programs produce, and its predictions are scattered back by one
# program; the clone, fit and predict are the compiled operations of the
# estimator's connected native plan (bindings/prep_native_estimator.mojo);
# its Python fit, predict, get_params or constructor is never called.
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.member_boundary_door import member_admit_serving_valued, member_pack, member_pack_labeled
from bindings.member_fit_notices_pack import member_fit_notices_raise
from bindings.member_owner_unpack import MemberOwner
from bindings.classifier_public_lifecycle_door import _public_labels
from core.member_abi import (
    MemberPlan, MemberCall, member_call, member_kind_capabilities, member_call_of, member_free,
    member_output_dtype,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_PREDICT, MEMBER_I32,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_PREDICT_STD, MEMBER_CAN_CLASSIFIER, MEMBER_BORROWS_FIT,
    MEMBER_GIL_FREE, MEMBER_GIL_DOOR,
)
from core.member_families import MEMBER_FAMILY_GP, MEMBER_FAMILY_XLINEAR
from bindings.member_held_door import MemberHeld, member_held, member_held_adopt, member_held_release
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.classical_arrays_door import classical_typed, classical_wrap_array
from bindings.admitted_input_unpack import classical_view
from bindings.classical_arrays_door import classical_dtype_str, classical_address
from bindings.scaler_native_door import _ptr
from core.preprocessing_program import prep_program_stage as _stage
from bindings.scaler_lifecycle_native_door import _entries, scaler_entries
from bindings.x_prep_native_owner_door import xprep_values, xprep_storage, xprep_run_records
from bindings.x_prep_native_owner_run import XprepInput, XprepCaps, XPREP_OWNER_NATIVE, xprep_r3
from bindings.user_objects_native_door import user_objects_call
from bindings.native_array_storage_door import native_array_unfilled
from bindings.normal_dist_helpers_pack import truncnorm_draws_binding
from bindings.prep_stateless_native_door import _int_like, _repr, _name, _fam
from bindings.prep_native_estimator_door import (
    prep_estimator_admit, prep_estimator_clone, prep_estimator_call, prep_estimator_has, prep_estimator_plan)
from bindings.prep_encode_prog_run import EncProgram, enc_ii_rows
from bindings.prep_encode_imputer_run import (
    ImputerFit, ImputerApply, IMP_MEAN, IMP_MEDIAN, IMP_MOST_FREQUENT, IMP_CONSTANT)
from bindings.prep_encode_native_door import (
    EncProg, enc_caps, IDN_STATS_BLOCKED, _precheck, _mark_missing, _missing_mark, _qselect, _real_like, _fitted_flat,
    _imputer_transform, _default_mode, _word, _take1, _take2, _seed64, _truth, _float, _is_str,
    _check_width, _int_list, _index_list)
from bindings.prep_owner_route_unpack import prep_route_context
from bindings.prep_keys_unpack import (
    prep_keys,
    PREP_OPN_FIT, PREP_OPN_FIT_TRANSFORM, PREP_OPN_TRANSFORM,
)
from bindings.py_balanced import py_item
from x_prep.iterimp_run import (
    IterImpRunner, IterImpFit, IterImpFitted, IterImpApply, IterImpStep, iterimp_fit, iterimp_prepare,
    iterimp_transform, iterimp_indicator, II_NONE, II_ASCENDING, II_DESCENDING, II_ROMAN, II_ARABIC, II_RANDOM,
    IterImpMember, IterImpMemberFit, iterimp_member_rounds, iterimp_member_free, IterImpHeldStep,
    IterImpApplied, iterimp_member_transform, iterimp_initial_only, iterimp_unmark_empty)

comptime F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime F64P = MutPointer[Float64, MutUntrackedOrigin]

# x_prep stage codes (_expansion_prep._OPS) of the user-estimator route.
comptime OP_II_CONV = 59
comptime OP_NAN_MASK = 60
comptime OP_II_ROWABS = 107
comptime OP_II_GATHER = 138
comptime OP_II_SCATTER = 139


# ---------------------------------------------------------------- small helpers
def _b() raises -> PythonObject:
    return native_loaded_module("builtins")


def _u64_of(value: PythonObject) raises -> UInt64:
    # A Python int (already reduced mod 2**64) as its 64-bit word.
    var words = _seed64(value)
    return (UInt64(words[1]) << 32) | UInt64(words[0])


def _u64_obj(value: UInt64) raises -> PythonObject:
    # The 64-bit word as a nonnegative Python int.
    return py_call(_b().int, PythonObject(String(value)))


def _floats_of(values: List[Int]) -> List[Float64]:
    var out = List[Float64]()
    for i in range(len(values)):  # driver-loop: over parameter words
        out.append(Float64(values[i]))
    return out^


def _one(value: Float64) -> List[Float64]:
    var out = List[Float64]()
    out.append(value)
    return out^









def _empty1(context: PythonObject) raises -> PythonObject:
    var nk = prep_keys()
    # `Array.from_list([], "<f4")`
    return classical_wrap_array(py_call(py_item(context, nk[].k_array_type), "f"), Python.tuple(0), "<f4", "C", py_item(context, nk[].k_output_context))


def _f32_vector(values: List[Float64], context: PythonObject) raises -> PythonObject:
    var nk = prep_keys()
    # The values as a float32 vector (`Array(...).astype("<f4")`).
    var out = classical_typed(Python.tuple(len(values)), "<f4", py_item(context, nk[].k_output_context))
    var dest = _ptr(out)
    _ = out  # keep the Python object alive through the native call above
    for i in range(len(values)):  # cpu-route: one feature's estimator predictions, rounded once to float32
        dest[i] = Float32(values[i])
        _ = out  # keep the Python object alive through the native call above
    return out


def _floats_from(value: PythonObject) raises -> List[Float64]:
    """`_user_objects.call("floats", v)`: float(x) of every entry."""
    var words = user_objects_call(Python.tuple("floats", value))
    var n = len(words)
    var src = F64P(unsafe_from_address=Int(py=py_call(words.buffer_info)[0]))
    # One native copy of the float64 words (no per-entry append).
    var out = List[Float64](length=n, fill=0.0)
    if n > 0:
        unsafe_memcpy(dest=F64P(unsafe_from_address=Int(out.unsafe_ptr())), src=src, count=n)
    _ = words  # keep the Python object alive through the native copy above
    return out^


def _pred_f64(v: PythonObject, context: PythonObject) raises -> List[Float64]:
    """`_pred_f64`: a float32 or float64 vector's values widened (exact);
    anything else float(x) per entry of its `tolist()` (`_as_list`)."""
    var b = _b()
    var cls = py_type(v)
    if not (cls is b.list or cls is b.tuple):
        var found = True
        var view = PythonObject(None)
        try:
            view = classical_view(v, context, "predict")
        except:
            found = False
        if found:
            var t = classical_dtype_str(view, context)
            var size = Int(py=view.shape[0]) if Int(py=view.ndim) == 1 else 0
            if Int(py=view.ndim) == 1 and (t == "<f4" or t == "<f8") and size > 0:
                var out = List[Float64](length=size, fill=0.0)
                var step = Int(py=view.strides[0])
                var addr = Int(py=classical_address(view))
                var wide = t == "<f8"
                for i in range(size):  # unpack: typed widening of the native prediction buffer, one per row (data; no CPython call)
                    if wide:
                        out[i] = F64P(unsafe_from_address=addr + i * step)[0]
                    else:
                        out[i] = Float64(F32P(unsafe_from_address=addr + i * step)[0])
                _ = view  # keep the Python object alive through the reads above
                return out^
    var listed = py_call(v.tolist) if Bool(py=py_call(b.hasattr, v, "tolist")) else v
    return _floats_from(listed)


# ---------------------------------------------------------------- parameters
# `_refuse`'s sample_posterior words; the member admission raises the same (lane mf-ens6b).
comptime _POSTERIOR_WORDS = "mojolearn: If 'sample_posterior' is True, the estimator must support 'return_std' in its 'predict' method."


def _refuse(model: PythonObject, context: PythonObject) raises:
    var nk = prep_keys()
    # `_refuse`: the parameter refusals, before X is read.
    var b = _b()
    var fields = py_fields(model)
    var estimator = py_item(fields, nk[].k_estimator)
    if estimator is not None:
        # A user estimator is a mojolearn estimator with a connected native
        # plan (its clone, fit and predict are compiled operations); any other
        # class is refused by name first.
        prep_estimator_admit(estimator, "IterativeImputer", context)
    var mi = py_item(fields, nk[].k_max_iter)
    if Bool(py=py_call(b.isinstance, mi, b.bool)) or not _int_like(mi, context) or Int(py=py_call(b.int, mi)) < 0:
        raise Error("The 'max_iter' parameter of IterativeImputer must be an int in the range [0, inf). Got "
                    + _repr(mi) + " instead.")
    var order = py_item(fields, nk[].k_imputation_order)
    var text = String(order) if _is_str(order) else String("")
    if text != "ascending" and text != "descending" and text != "roman" and text != "arabic" and text != "random":
        raise Error("mojolearn: invalid imputation_order " + _repr(order))
    var nnf = py_item(fields, nk[].k_n_nearest_features)
    if nnf is not None and (not _int_like(nnf, context) or Int(py=py_call(b.int, nnf)) < 1):
        raise Error("mojolearn: n_nearest_features must be an int >= 1; got " + _repr(nnf))
    if (_truth(py_item(fields, nk[].k_sample_posterior)) and estimator is not None
            and not prep_estimator_has(estimator, "predict_std", context)):
        # The native plan declares the supported return_std operation;
        # inspecting/calling the estimator's Python predict is unnecessary.
        raise Error(_POSTERIOR_WORDS)


def _per_bound(v: PythonObject, d: Int) raises -> List[Float64]:
    # `_bounds.per`: a list, tuple or array-like as given, else v for every column.
    var b = _b()
    var vals = PythonObject(None)
    if Bool(py=py_call(b.isinstance, v, Python.tuple(b.list, b.tuple))) or Bool(py=py_call(b.hasattr, v, "tolist")):
        vals = py_call(b.list, v)
    else:
        vals = Python.list()
        for _ in range(d):  # driver-loop: over the columns (one bound argument each)
            _ = py_call(vals.append, v)
    var out = List[Float64]()
    for i in range(len(vals)):  # driver-loop: over the columns' min_value / max_value parameters (one per feature)
        out.append(_float(vals[i]))
    return out^


def _float_list(values: List[Float64]) raises -> PythonObject:
    var out = Python.list()
    for i in range(len(values)):  # driver-loop: over fitted per-column bounds
        _ = py_call(out.append, PythonObject(values[i]))
    return out


def _bounds_of(model: PythonObject) raises -> List[Float64]:
    var nk = prep_keys()
    # The fitted `_bounds_k` list.
    var items = py_call(_b().list, py_item(py_fields(model), nk[].k__bounds_k))
    var out = List[Float64]()
    for i in range(len(items)):  # driver-loop: over fitted per-column bounds
        out.append(_float(items[i]))
    return out^


def _mask_arr(values: PythonObject, keep: List[Int], empty: List[Int], missing: PythonObject, entries: PythonObject,
              context: PythonObject) raises -> PythonObject:
    """`_mask_arr`: the (n, dk) float32 missing mask (1.0 missing), the
    kept positions `empty` (no observed value at fit) marked not missing as
    scikit-learn marks them, run in the process mode as the Python ran it:
    `entries` is the caller's selection for that mode (selected once per
    operation)."""
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var dk = len(keep)
    var p = EncProg()
    var xo = _mark_missing(p, p.put_x(values), n * d, missing, context)
    var mo = p.alloc(n * dk)
    var ko = p.put_floats(_floats_of(keep), context)
    _stage(p.prog.program, OP_NAN_MASK, n * dk, xo, n, d, ko, dk, mo)
    iterimp_unmark_empty(p.prog, mo, n, dk, empty)
    var ran = p.run(entries, context)
    return _take2(ran, mo, n, dk, context)


# ---------------------------------------------------------------- the user-estimator route
struct _Take(Movable):
    # `_ii_take`'s (Xo, yo, Xm, missing rows as int32 words, m).
    var xo: PythonObject
    var yo: PythonObject
    var xm: PythonObject
    var rows: PythonObject
    var m: Int

    def __init__(out self, xo: PythonObject, yo: PythonObject, xm: PythonObject, rows: PythonObject, m: Int):
        self.xo = xo
        self.yo = yo
        self.xm = xm
        self.rows = rows
        self.m = m


def _ii_block(ran: PythonObject, xo: Int, m: Int, nc: Int, context: PythonObject) raises -> PythonObject:
    # `_ii_block`'s X: a 0-row block `from_list([])`, a 0-column one its m empty rows.
    if m == 0:
        return _empty1(context)
    return _take2(ran, xo, m, nc, context)


def _ii_take(xt: PythonObject, mask: PythonObject, n: Int, dk: Int, j: Int, nbl: List[Int], entries: PythonObject,
             fit: Bool, context: PythonObject) raises -> _Take:
    """`_ii_take`: one program: feature j's missing rows and the estimator's
    predict block Xt[mis][:, nbl] and, with fit, its training blocks
    Xt[obs][:, nbl] and Xt[obs, j]."""
    var nc = len(nbl)
    var p = EncProg()
    var xo = p.put(xt, n * dk)
    var mo = p.put(mask, n * dk)
    var cols = List[Float64]()
    for i in range(nc):  # driver-loop: over one step's predictor features
        cols.append(Float64(nbl[i]))
    if nc == 0:
        cols.append(0.0)
    var co = p.put_floats(cols, context)
    var jo = p.put_floats(_one(Float64(j)), context)
    var mis = enc_ii_rows(p.prog, mo, n, dk, j, True)
    var gm = p.alloc(n * nc)
    if nc > 0:
        _stage(p.prog.program, OP_II_GATHER, n * nc, xo, dk, mis[0], mis[1], co, nc, gm)
    var obs = (0, 0)
    var go = 0
    var gy = 0
    if fit:
        obs = enc_ii_rows(p.prog, mo, n, dk, j, False)
        go = p.alloc(n * nc)
        gy = p.alloc(n)
        if nc > 0:
            _stage(p.prog.program, OP_II_GATHER, n * nc, xo, dk, obs[0], obs[1], co, nc, go)
        _stage(p.prog.program, OP_II_GATHER, n, xo, dk, obs[0], obs[1], jo, 1, gy)
    var ran = p.run(entries, context)
    var m = Int(_word(ran, mis[1]))
    var x_fit = PythonObject(None)
    var y_fit = PythonObject(None)
    if fit:
        var mob = Int(_word(ran, obs[1]))
        x_fit = _ii_block(ran, go, mob, nc, context)
        y_fit = _empty1(context) if mob == 0 else _take1(ran, gy, mob, context)
    var xm = _ii_block(ran, gm, m, nc, context)
    # the rows' int32 words as they are (a 4-byte store the next program copies)
    var rows = _take1(ran, mis[0], m, context) if m > 0 else PythonObject(None)
    return _Take(x_fit, y_fit, xm, rows, m)


def _truncnorm(mus: PythonObject, sig: PythonObject, lo: Float64, hi: Float64, mut rng: UInt64,
               context: PythonObject) raises -> List[Float64]:
    # Preserve the supplied estimator route's original binary64 draws and
    # splitmix stream. Inputs come from the native predict_std operation.
    var mu = _pred_f64(mus, context)
    var sd = _pred_f64(sig, context)
    var k = min(len(mu), len(sd))
    # lane fix-output-zero-fill (2026-11-14): unfilled; the loop below writes all k of mb and sb,
    # truncnorm_draws_binding all k draws of out.
    var mb = native_array_unfilled("d", k)
    var sb = native_array_unfilled("d", k)
    var out = native_array_unfilled("d", k)
    var mp = F64P(unsafe_from_address=Int(py=py_call(mb.buffer_info)[0]))
    var sp = F64P(unsafe_from_address=Int(py=py_call(sb.buffer_info)[0]))
    for i in range(k):  # cpu-route: one feature's native posterior buffers
        mp[i] = mu[i]
        sp[i] = sd[i]
    var state = Python.tuple(Int(rng >> 32), Int(rng & 0xFFFFFFFF))
    var hi_lo = truncnorm_draws_binding(py_call(mb.buffer_info)[0], py_call(sb.buffer_info)[0], PythonObject(k),
                                       Python.tuple(lo, hi), state, py_call(out.buffer_info)[0])
    rng = (UInt64(Int(py=hi_lo[0])) << 32) | UInt64(Int(py=hi_lo[1]))
    var src = F64P(unsafe_from_address=Int(py=py_call(out.buffer_info)[0]))
    # One native copy of the completed posterior draws.
    var vals = List[Float64](length=k, fill=0.0)
    if k > 0:
        unsafe_memcpy(dest=F64P(unsafe_from_address=Int(vals.unsafe_ptr())), src=src, count=k)
    _ = out  # keep the draw store alive through the native copy above
    _ = mb
    _ = sb
    return vals^


def _ii_put(xt: PythonObject, xm: PythonObject, rows: PythonObject, m: Int, n: Int, dk: Int, j: Int,
            est: PythonObject, posterior: Bool, bounds: List[Float64], entries: PythonObject, mut rng: UInt64,
            context: PythonObject) raises -> PythonObject:
    """`_ii_put`: predict on the missing rows' block, then (one program) the
    clipped float32 store into Xt[rows, j]; sample_posterior's truncated
    normal draw is stored unclipped."""
    if m == 0:
        return xt
    var lo = bounds[2 * j]
    var hi = bounds[2 * j + 1]
    var v = List[Float64]()
    var clip = 1
    if posterior:
        var pair = prep_estimator_call(est, "predict_std", Python.tuple(xm), context)
        if py_type(pair) is not _b().tuple or len(pair) != 2:
            raise Error("TypeError: native posterior prediction must return a mean/std pair")
        v = _truncnorm(pair[0], pair[1], lo, hi, rng, context)
        clip = 0
    else:
        v = _pred_f64(prep_estimator_call(est, "predict", Python.tuple(xm), context), context)
    var k = min(len(v), m)  # the reference's zip stops at the shorter
    if k == 0:
        return xt
    while len(v) > k:  # driver-loop: drops predictions past the missing rows
        _ = v.pop()
    var p = EncProg()
    var xo = p.put_inout(xt, n * dk)
    var ro = p.put(rows, m)
    var vo = p.put(_f32_vector(v, context), k)
    var lohi = List[Float64]()
    lohi.append(lo)
    lohi.append(hi)
    var bo = p.put_floats(lohi, context)
    _stage(p.prog.program, OP_II_SCATTER, k, xo, dk, ro, j, vo, bo, clip)
    var ran = p.run(entries, context)
    return _take2(ran, xo, n, dk, context)


def _ii_stop(xt: PythonObject, prev: PythonObject, n: Int, dk: Int, tol: Float64, entries: PythonObject,
             context: PythonObject) raises -> Bool:
    # `_ii_stop`: the round's stop, the inf-norm of Xt - prev below tol.
    var p = EncProg()
    var xo = p.put(xt, n * dk)
    var po = p.put(prev, n * dk)
    var to = p.put_floats(_one(tol), context)
    var flag = p.alloc(1)
    var niter = p.alloc(1)
    var rowabs = p.alloc(n)
    _stage(p.prog.program, OP_II_ROWABS, n, xo, po, dk, rowabs, flag)
    _stage(p.prog.program, OP_II_CONV, 1, xo, po, n * dk, to, flag, niter, dk, rowabs + 1)
    var ran = p.run(entries, context)
    return _word(ran, flag) != 0


def _others(dk: Int, j: Int) -> List[Int]:
    # `[a for a in range(dk) if a != j]`
    var out = List[Int]()
    for a in range(dk):  # driver-loop: over the kept features
        if a != j:
            out.append(a)
    return out^


def _nbl_of(value: PythonObject) raises -> List[Int]:
    return _index_list(value)


# ---------------------------------------------------------------- the runner
struct _DoorRunner(IterImpRunner):
    """The RUN's programs over the x_prep transport of the operation's image
    (`xprep_run_records`: UNPACK of X's record, the typed run with the GIL
    released, cleanup). Owner 0 is X as `xprep_values` converted it; a
    RUN-held or door-held native buffer (owner k >= 1) is an
    XPREP_OWNER_NATIVE record at `seconds[k - 1]`. Each arena (and output
    store) is held until `release` or the runner's end; PACK reads them
    with `_take1` / `_take2` (`ran`). `release_members`: a supplied
    estimator's member operations run with the GIL released (its kind has
    MEMBER_GIL_FREE or MEMBER_GIL_DOOR; set by the door after `_member`)."""
    var owners: List[PythonObject]
    var entries: PythonObject
    var context: PythonObject
    var caps: XprepCaps
    var held: List[PythonObject]
    var addresses: List[Int]
    var release_members: Bool

    def __init__(out self, values: PythonObject, entries: PythonObject, context: PythonObject, caps: XprepCaps):
        self.owners = List[PythonObject]()
        self.owners.append(values)
        self.entries = entries
        self.context = context
        self.caps = caps
        self.held = List[PythonObject]()
        self.addresses = List[Int]()
        self.release_members = False

    def member(mut self, plan: MemberPlan, operation: Int, mut call: MemberCall) raises -> Int:
        # THE GIL RULE (core/member_abi.mojo): released around the one call
        # for a kind that allows it (every argument is a typed word, read
        # before the region), else the call runs with the GIL held.
        if not self.release_members:
            return member_call(plan.member, operation, call)
        var result = 0
        var py = Python()
        with GILReleased(py):
            result = member_call(plan.member, operation, call)
        return result

    def run(mut self, mut prog: EncProgram, seconds: List[Int]) raises -> Int:
        prog.check_bound()
        var host = prog.host
        var scratch = prog.scratch_size
        var out_words = prog.out_words
        var device_scratch = scratch > 0 and (self.caps.has_out or self.caps.has_scratch)
        var arena = xprep_storage(host + (0 if device_scratch else scratch), self.context)
        var store = xprep_storage(out_words, self.context) if out_words > 0 else PythonObject(None)
        var records = List[XprepInput](capacity=len(prog.inputs))
        for i in range(len(prog.inputs)):  # driver-loop: over the program's input records
            var record = prog.inputs[i]
            if record.owner >= 1:
                records.append(XprepInput(record.offset, record.words, seconds[record.owner - 1], XPREP_OWNER_NATIVE))
            else:
                records.append(record)
        xprep_run_records(arena, host, scratch, store, out_words, prog.stage_words(), records, self.owners,
                          self.entries, self.context, out_words, prog.inout_ranges, self.caps)
        self.held.append(Python.tuple(arena, store))
        self.addresses.append(Int(py=arena[1]))
        return len(self.held) - 1

    def words(self, handle: Int) -> Int:
        return self.addresses[handle]

    def release(mut self, handle: Int):
        self.held[handle] = PythonObject(None)
        self.addresses[handle] = 0


def _f32_array(values: List[Float32], context: PythonObject) raises -> PythonObject:
    # A fitted float32 vector (`_Prog.get(off, count)`) from the RUN's copy.
    var nk = prep_keys()
    var out = classical_typed(Python.tuple(len(values)), "<f4", py_item(context, nk[].k_output_context))
    if len(values) > 0:
        unsafe_memcpy(dest=F32P(unsafe_from_address=Int(py=classical_address(out))),
                      src=F32P(unsafe_from_address=Int(values.unsafe_ptr())), count=len(values))
    _ = values  # the RUN's copy stays alive through the native copy above
    return out


# ---------------------------------------------------------------- fit: UNPACK
def _initial(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """The initial `SimpleImputer(missing_values, strategy=initial_strategy,
    fill_value, keep_empty_features)`: a fresh instance holding
    SimpleImputer.__init__'s six parameters and this estimator's `device`
    (None for an instance saved before the parameter existed), its route
    resolved for this operation's context, its parameter refusals raised.
    The RUN fits it; PACK stores its fitted fields."""
    var nk = prep_keys()
    var fields = py_fields(model)
    var cls = py_item(py_item(context, nk[].k_prep_owner_classes), nk[].k_SimpleImputer)
    var imputer = py_call(_b().object.__new__, cls)
    var own = py_fields(imputer)
    py_setitem(own, nk[].k_missing_values, py_item(fields, nk[].k_missing_values))
    py_setitem(own, nk[].k_strategy, py_item(fields, nk[].k_initial_strategy))
    py_setitem(own, nk[].k_fill_value, py_item(fields, nk[].k_fill_value))
    py_setitem(own, nk[].k_copy, PythonObject(True))
    py_setitem(own, nk[].k_add_indicator, PythonObject(False))
    py_setitem(own, nk[].k_keep_empty_features, py_item(fields, nk[].k_keep_empty_features))
    py_setitem(own, nk[].k_device, py_get(fields, nk[].k_device))
    _ = prep_route_context(imputer, context)
    _precheck(imputer, "simple_imputer", context)
    return imputer


def _initial_request(imputer: PythonObject, n: Int, d: Int, mode: PythonObject, entries: PythonObject,
                     context: PythonObject) raises -> ImputerFit:
    # `_imputer_fit`'s UNPACK of the initial imputer's parameters (X is owner 0).
    var fields = py_fields(imputer)
    var strategy_field = fields["strategy"]
    var name = String(strategy_field) if _is_str(strategy_field) else String("")
    var strategy = IMP_MEAN
    if name == "median":
        strategy = IMP_MEDIAN
    elif name == "most_frequent":
        strategy = IMP_MOST_FREQUENT
    elif name == "constant":
        strategy = IMP_CONSTANT
    elif name != "mean":
        raise Error("mojolearn: SimpleImputer strategy " + _repr(strategy_field) + " is not valid")
    var konst = strategy == IMP_CONSTANT
    var fill_value = fields["fill_value"]
    if konst and fill_value is not None and not _real_like(fill_value, context):
        raise Error("TypeError: mojolearn: SimpleImputer fill_value must be numeric")
    var mark = _missing_mark(fields["missing_values"])
    var fill = 0.0
    if konst and fill_value is not None:
        fill = _float(fill_value)
    return ImputerFit(n, d, 0, mark[0], mark[1], strategy,
                      strategy == IMP_MEDIAN and _qselect(mode, entries, context),
                      (_fam(mode, entries) & IDN_STATS_BLOCKED) != 0, fill, _truth(fields["keep_empty_features"]))


def _order_code(fields: PythonObject) raises -> Int:
    var nk = prep_keys()
    var name = String(py_item(fields, nk[].k_imputation_order))  # admitted by `_refuse`
    if name == "descending":
        return II_DESCENDING
    if name == "roman":
        return II_ROMAN
    if name == "arabic":
        return II_ARABIC
    if name == "random":
        return II_RANDOM
    return II_ASCENDING


def _fit_request(model: PythonObject, values: PythonObject, init: ImputerFit, mode: PythonObject,
                 entries: PythonObject, caps: XprepCaps) raises -> IterImpFit:
    # UNPACK: this estimator's parameters, read once into the typed request.
    var nk = prep_keys()
    var b = _b()
    var fields = py_fields(model)
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var mark = _missing_mark(py_item(fields, nk[].k_missing_values))
    var nnf_field = py_item(fields, nk[].k_n_nearest_features)
    var nnf = II_NONE if nnf_field is None else Int(py=py_call(b.int, nnf_field))
    return IterImpFit(init.copy(), n, d, mark[0], mark[1], (_fam(mode, entries) & IDN_STATS_BLOCKED) != 0,
                      xprep_r3("work"), caps.has_out, _order_code(fields),
                      _truth(py_item(fields, nk[].k_skip_complete)), _truth(py_item(fields, nk[].k_sample_posterior)),
                      _truth(py_item(fields, nk[].k_add_indicator)), Int(py=py_call(b.int, py_item(fields, nk[].k_max_iter))),
                      nnf, _float(py_item(fields, nk[].k_tol)), _u64_of(py_item(fields, nk[].k_random_state)),
                      _per_bound(py_item(fields, nk[].k_min_value), d), _per_bound(py_item(fields, nk[].k_max_value), d))


# ---------------------------------------------------------------- fit: PACK
def _pack_initial(model: PythonObject, imputer: PythonObject, fitted: IterImpFitted, mode: PythonObject, d: Int,
                  context: PythonObject) raises:
    """The initial imputer's fitted fields (`_imputer_fit`'s PACK: one object
    for statistics_ and _fill when they are the same values), then this
    estimator's fields that the prologue decided."""
    var nk = prep_keys()
    var own = py_fields(imputer)
    var statistics = _f32_array(fitted.statistics, context)
    py_setitem(own, "statistics_", statistics)
    py_setitem(own, "_fill", statistics if fitted.shared else _f32_array(fitted.fill, context))
    py_setitem(own, "_keep", _int_list(fitted.keep))
    py_setitem(own, "_indicator", _int_list(List[Int]()))
    py_setitem(own, "numeric_mode_", mode)
    py_setitem(own, "n_features_in_", d)
    var fields = py_fields(model)
    py_setitem(fields, nk[].k_initial_imputer_, imputer)
    py_setitem(fields, nk[].k__keep, _int_list(fitted.keep))
    # the kept positions with no observed value at fit, which transform marks
    # not missing as scikit-learn's `_is_empty_feature` does
    py_setitem(fields, "_empty_kept", _int_list(fitted.empty))
    py_setitem(fields, nk[].k__bounds_k, _float_list(fitted.bounds))
    py_setitem(fields, nk[].k__indicator, _int_list(fitted.indicator))
    py_setitem(fields, nk[].k_n_features_with_missing_, PythonObject(fitted.n_missing))
    py_setitem(fields, nk[].k_numeric_mode_, mode)
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))


def _nbl_obj(has: Bool, nbl: List[Int]) raises -> PythonObject:
    if not has:
        return PythonObject(None)
    return _int_list(nbl)


def _xt_address(xt: PythonObject, words: Int) raises -> Int:
    # The (n, dk) float32 block's first word (an empty block is read by no record).
    return Int(py=classical_address(xt)) if words > 0 else 0


# ---------------------------------------------------------------- a supplied estimator as a member
struct _Member(Movable):
    """The supplied estimator's hold as a native member, built once per fit
    (`_member`). `native` False: the door rounds serve it. True: `owner`
    (its plan and the pins that keep its images loaded, alive through PACK),
    `template` the first step's clone (the admission read its fields) and
    `fields` its instance dictionary, `member_context` the family's context,
    `fields_get` the plan's field reader (for the other steps' clones),
    `plan_key` the class's import-time plan (the held-state record's key),
    `borrows` / `release` the kind's MEMBER_BORROWS_FIT and GIL bits,
    `classifier` a classifier kind whose PREDICT writes codes of
    `code_dtype` (lane mf-iterimp3); `cw_refusal` the estimator's own words
    for a class_weight it refuses (lane mf-left4: raised by the RUN at its
    first classifier fit, where the door rounds' first fit raised them)."""
    var native: Bool
    var owner: MemberOwner
    var plan_key: PythonObject
    var template: PythonObject
    var fields: PythonObject
    var member_context: PythonObject
    var fields_get: PythonObject
    var borrows: Bool
    var release: Bool
    var classifier: Bool
    var code_dtype: Int
    var cw_refusal: String
    # Lane mf-ens6b: a sample_posterior fit whose admitted member family
    # answers no posterior pair (`_posterior_family`): refused natively in
    # `_refuse`'s words, never sent to the door rounds.
    var posterior_refused: Bool

    def __init__(out self):
        self.native = False
        self.cw_refusal = String("")
        self.posterior_refused = False
        self.owner = MemberOwner()
        self.plan_key = PythonObject(None)
        self.template = PythonObject(None)
        self.fields = PythonObject(None)
        self.member_context = PythonObject(None)
        self.fields_get = PythonObject(None)
        self.borrows = False
        self.release = False
        self.classifier = False
        self.code_dtype = MEMBER_I32


def _posterior_family(family: Int) -> Bool:
    """The families whose PREDICT_STD writes the same query's mean to
    `call.y` when given (core/gp_member.mojo, core/xlinear_member.mojo): the
    pair a sample_posterior step draws from, in one member call."""
    return family == MEMBER_FAMILY_GP or family == MEMBER_FAMILY_XLINEAR


def _member(estimator: PythonObject, posterior: Bool, dk: Int, context: PythonObject) -> _Member:
    """UNPACK, once per fit: the estimator as a native member when its class
    is a published member whose KIND fits and predicts one target (a
    regressor, or a classifier: lane mf-iterimp3, fitted on each step's
    codes as scikit-learn fits it on the float target), and for
    sample_posterior one whose kind publishes PREDICT_STD answering the
    same query's mean too (`_posterior_family`: GaussianProcessRegressor,
    BayesianRidge, ARDRegression). Never raises: anything else, and an
    admission the family refuses, keeps the door rounds (the fallback
    rule), which then answer or refuse in the estimator's own words."""
    var out = _Member()
    # TOMBSTONE: MOJOLEARN_MEMBER_SET_OFF (plumbing) deleted 2026-11-23 by lane/ab-prep-classical; code recoverable at 4d76a9e16e.
    try:
        var b = _b()
        var plan = prep_estimator_plan(estimator, context)
        if plan is None or py_get(plan, "member") is None:
            return out^
        var classifier = Bool(py=plan["classifier"])
        if classifier and posterior:
            return out^  # no classifier plan serves predict_std (`_refuse` already refused it)
        var needs = MEMBER_CAN_FIT | MEMBER_CAN_PREDICT
        if posterior:
            needs |= MEMBER_CAN_PREDICT_STD
        var template = prep_estimator_clone(estimator, context)
        var member_context = plan["context"]
        var fields_get = plan["fields_get"]
        var fields = py_call(fields_get, template, py_type(template))
        # A classifier's class_weight dict is keyed by the step's values,
        # read into the KEYED tail here (lane mf-adm3); each step makes its
        # table over its classes (x_prep/iterimp_run.mojo `_member_step`). A
        # value or a key the estimator refuses leaves its words in
        # `refusal` for the RUN's first classifier fit (lane mf-left4).
        var refusal = String("")
        var owner = member_admit_serving_valued(template, fields, plan, member_context, b, needs, refusal)
        if not owner.published:
            return out^
        if posterior and not _posterior_family(owner.plan.family):
            out.posterior_refused = True  # `_refuse`'s words, raised by the fit (lane mf-ens6b)
            return out^
        # The mask of THIS kind (the family's union holds its other kinds'
        # bits): one state minted, asked and released, once per fit.
        var mask = member_kind_capabilities(owner.plan)
        if ((mask & MEMBER_CAN_CLASSIFIER) != 0) != classifier or (mask & needs) != needs:
            return out^
        if classifier:
            # The element type of its predicted codes, asked once (two classes
            # stand in for the steps' counts: the type is the kind's).
            var two = owner.plan.copy()
            two.classes = 2
            out.code_dtype = member_output_dtype(two, MEMBER_OP_PREDICT, max(dk - 1, 1), 1, MEMBER_I32)
            out.classifier = True
        out.cw_refusal = refusal
        out.borrows = (mask & MEMBER_BORROWS_FIT) != 0
        out.release = (mask & (MEMBER_GIL_FREE | MEMBER_GIL_DOOR)) != 0
        out.owner = owner^
        out.plan_key = plan
        out.template = template
        out.fields = fields
        out.member_context = member_context
        out.fields_get = fields_get
        out.native = True
    except:
        return _Member()
    return out^


def _refuse_member_only(estimator: PythonObject, context: PythonObject) raises:
    """A class whose plan is MEMBER-ONLY (registered from its member mark
    alone: python/mojolearn/_native_msel_more.py, the Host* classes of lane
    mf-cal3b / mf-iterimp3b and the derived plans) has no compiled fit or
    predict for the door rounds, so when its member does not serve this fit
    it is refused here in the words it had before its mark
    (`prep_estimator_admit`'s), never by a door-round call (lane
    mf-iterimp3b, 2026-11-23). One plan read, on this fallback only."""
    var plan = prep_estimator_plan(estimator, context)
    if plan is not None and py_get(plan, "member_only") is not None:
        raise Error("TypeError: mojolearn: IterativeImputer requires a mojolearn estimator; got "
                    + String(py_type(estimator).__name__))


def _member_pack(estimator: PythonObject, member: _Member, mut fit: IterImpMemberFit, labels: PythonObject,
                 context: PythonObject) raises -> PythonObject:
    """PACK of the member rounds: `imputation_sequence_`, one (feat_idx,
    neighbor_feat_idx, estimator) triple per finished step, each estimator a
    clone of the supplied one (the first is the admission's) with the step's
    member state written into it (`member_pack`: the attributes its own fit
    writes). One clone per step, as the door rounds made. Each step's state
    is then HANDED to its estimator (`member_held_adopt`, lane mf-iterimp2):
    transform predicts through it with no import; its word in `fit.states`
    becomes 0 so `iterimp_member_free` leaves it to the estimator. A
    classifier step's estimator gets its labels as its own fit publishes
    them for the float target (`_public_labels` over the classes as Python
    floats, `member_pack_labeled`), its plan the step's class count; the
    labels are also kept in `labels` (one entry per step, None for a
    regressor), which the door stores as `_step_classes` for transform."""
    var b = _b()
    var seq = Python.list()
    for s in range(len(fit.states)):  # driver-loop: over the fitted imputation steps (one clone and one pack each)
        var est = member.template
        var fields = member.fields
        if s > 0:
            est = prep_estimator_clone(estimator, context)
            fields = py_call(member.fields_get, est, py_type(est))
        if len(fit.classes[s]) == 0:
            member_pack(est, fields, member.member_context, b, member.owner.plan, fit.states[s])
            if member_held_adopt(est, member.plan_key, member.member_context, member.owner.pins, member.owner.plan,
                                 fit.states[s]):
                fit.states[s] = 0  # the estimator's now
            _ = py_call(labels.append, PythonObject(None))
        else:
            var plan = member.owner.plan.copy()
            plan.classes = len(fit.classes[s])
            var values = Python.list()
            for c in range(len(fit.classes[s])):  # driver-loop: over one step's classes
                _ = py_call(values.append, PythonObject(Float64(fit.classes[s][c])))
            var published = _public_labels(values, member.member_context)
            member_pack_labeled(est, fields, member.member_context, b, plan, fit.states[s], published)
            if member_held_adopt(est, member.plan_key, member.member_context, member.owner.pins, plan,
                                 fit.states[s]):
                fit.states[s] = 0  # the estimator's now
            _ = py_call(labels.append, published)
        _ = py_call(seq.append, Python.tuple(fit.seq_j[s], _int_list(fit.seq_nbl[s]), est))
    return seq


# ---------------------------------------------------------------- fit
def _fit_transform(model: PythonObject, x: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = prep_keys()
    # UNPACK: the refusals, the selection (ONCE per operation: every program
    # of the rounds runs on it), X, the initial imputer and the request.
    _refuse(model, context)
    var fields = py_fields(model)
    var mode = _default_mode(context)
    var entries = scaler_entries(model, mode, context)
    var caps = enc_caps(model, entries)
    var values = xprep_values(x, entries, context)
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var imputer = _initial(model, context)
    var request = _fit_request(model, values, _initial_request(imputer, n, d, mode, entries, context), mode,
                               entries, caps)
    var runner = _DoorRunner(values, entries, context, caps)
    var estimator = py_item(fields, nk[].k_estimator)
    if estimator is None:
        # RUN: the built-in Bayesian ridge, every program typed (x_prep/iterimp_run.mojo)
        var fitted = iterimp_fit(runner, request)
        # PACK
        _pack_initial(model, imputer, fitted, mode, d, context)
        var dk = len(fitted.keep)
        var got = runner.held[fitted.rounds]
        py_setitem(fields, nk[].k_n_iter_, PythonObject(fitted.n_iter))
        var sequence = Python.list()
        for s in range(len(fitted.seq_j)):  # driver-loop: fitted imputation sequence views (finished imputation steps)
            _ = py_call(sequence.append, Python.tuple(fitted.seq_j[s], _take1(got, fitted.seq_coef[s], dk, context),
                                                      _take1(got, fitted.seq_inter[s], 1, context)))
        py_setitem(fields, nk[].k_imputation_sequence_, sequence)
        var post = Python.list()
        for s in range(len(fitted.post_sig)):  # driver-loop: fitted posterior views (finished imputation steps)
            var side = fitted.post_side[s]
            _ = py_call(post.append, Python.tuple(_nbl_obj(fitted.post_has_nbl[s], fitted.post_nbl[s]),
                                                  _take1(got, fitted.post_sig[s], side * side, context),
                                                  _take1(got, fitted.post_al[s], 2, context),
                                                  _take1(got, fitted.post_means[s], dk, context)))
        py_setitem(fields, nk[].k__posterior, post)
        py_setitem(fields, nk[].k__rng, _u64_obj(fitted.rng))
        py_setitem(fields, nk[].k__post_step, PythonObject(fitted.post_step))
        return _take2(runner.held[fitted.out_handle], fitted.out_offset, n, fitted.cols, context)
    # A supplied estimator: the typed prologue (programs 1 to 4), then the
    # reference's rounds over copies of the user estimator (its native plan)
    var fitted = IterImpFitted()
    var plan = iterimp_prepare(runner, request, fitted)
    _pack_initial(model, imputer, fitted, mode, d, context)
    var keep = fitted.keep.copy()
    var dk = len(keep)
    if dk < 2:
        # scikit-learn's edge case (fewer than two kept features): n_iter_ = 0,
        # no step, the initial imputation; the estimator is never fitted
        iterimp_initial_only(runner, request, plan, fitted)
        py_setitem(fields, nk[].k__rng, _u64_obj(fitted.rng))
        py_setitem(fields, nk[].k__post_step, PythonObject(0))
        py_setitem(fields, nk[].k_n_iter_, PythonObject(0))
        py_setitem(fields, nk[].k_imputation_sequence_, Python.list())
        py_setitem(fields, "_step_classes", Python.list())
        return _take2(runner.held[fitted.out_handle], fitted.out_offset, n, fitted.cols, context)
    var member = _member(estimator, request.posterior, dk, context)
    if member.posterior_refused:
        raise Error(_POSTERIOR_WORDS)
    if not member.native:
        _refuse_member_only(estimator, context)
    if member.native:
        # RUN: the rounds over the native member (x_prep/iterimp_run.mojo)
        runner.release_members = member.release
        var run = IterImpMember(member.owner.plan.copy(), member.borrows, request.posterior)
        run.classifier = member.classifier
        run.code_dtype = member.code_dtype
        run.cw_refusal = member.cw_refusal
        var fit = IterImpMemberFit()
        iterimp_member_rounds(runner, request, plan, fitted, run, fit)
        # PACK (every state is released after it, on every path)
        try:
            var labels = Python.list()
            var seq = _member_pack(estimator, member, fit, labels, context)
            py_setitem(fields, nk[].k__rng, _u64_obj(fit.rng))
            py_setitem(fields, nk[].k__post_step, PythonObject(0))
            py_setitem(fields, nk[].k_n_iter_, PythonObject(fit.n_iter))
            py_setitem(fields, nk[].k_imputation_sequence_, seq)
            py_setitem(fields, "_step_classes", labels)
        finally:
            iterimp_member_free(run, fit)
        # The steps' fit notices, in step order, as the estimator's own fits raised them.
        member_fit_notices_raise(fit.notices, member.member_context)
        _ = member.owner.published  # the member's images stay loaded past its last call
        return _take2(runner.held[fit.out_handle], fit.out_offset, n, fit.cols, context)
    var bounds = fitted.bounds.copy()
    var posterior = request.posterior
    var rng = plan.rng
    var mask = _mask_arr(values, keep, fitted.empty, py_item(fields, nk[].k_missing_values), entries, context)
    var xt = _take2(runner.held[plan.xf_handle], plan.xf_offset, n, dk, context)
    runner.release(plan.xf_handle)
    var seq = Python.list()
    var done = 0
    for r in range(len(plan.orders)):  # api-bound-loop: over the imputation rounds of the user-supplied estimator object (public type: a Mojolearn estimator); one clone and one fit of its native plan per feature each, through the Python object
        var check = not posterior and len(plan.orders[r]) > 0
        var prev = xt
        for i in range(len(plan.orders[r])):  # api-bound-loop: over the round's features; one clone and one fit of the user-supplied estimator object's native plan each
            var j = plan.orders[r][i]
            var nbl = plan.draws.next(rng) if plan.draws.on else _others(dk, j)
            var est = prep_estimator_clone(estimator, context)
            var took = _ii_take(xt, mask, n, dk, j, nbl, entries, True, context)
            _ = prep_estimator_call(est, "fit", Python.tuple(took.xo, took.yo), context)
            _ = py_call(seq.append, Python.tuple(j, _int_list(nbl), est))
            xt = _ii_put(xt, took.xm, took.rows, took.m, n, dk, j, est, posterior, bounds, entries, rng, context)
        done = r + 1
        if check and _ii_stop(xt, prev, n, dk, plan.tol, entries, context):
            break
    py_setitem(fields, nk[].k__rng, _u64_obj(rng))
    py_setitem(fields, nk[].k__post_step, PythonObject(0))
    py_setitem(fields, nk[].k_n_iter_, PythonObject(done if plan.any_order else min(1, len(plan.orders))))
    py_setitem(fields, nk[].k_imputation_sequence_, seq)
    py_setitem(fields, "_step_classes", PythonObject(None))  # the door rounds' steps are served by these rounds
    if len(fitted.indicator) == 0:
        return xt
    var made = iterimp_indicator(runner, n, d, dk, request.mark, request.missing, fitted.indicator,
                                 _xt_address(xt, n * dk), request.work_on, request.region)
    _ = xt  # the block the indicator program read stays alive past it
    return _take2(runner.held[made[0]], made[1], n, dk + len(fitted.indicator), context)


# ---------------------------------------------------------------- transform
def _steps(sequence: PythonObject, posts: PythonObject, posterior: Bool) raises -> List[IterImpStep]:
    """UNPACK of the fitted steps: each step's feature and the addresses of its
    coef and intercept vectors (and, with sample_posterior, its neighbour list
    and the addresses and sizes of its sigma, alpha and means). The door
    keeps `sequence` and `posts` alive through the RUN."""
    var out_steps = List[IterImpStep]()
    for s in range(len(sequence)):  # driver-loop: over the fitted imputation steps (one record read each)
        var step = sequence[s]
        var has = False
        var nbl = List[Int]()
        var sig = 0
        var sig_words = 0
        var al = 0
        var al_words = 0
        var means = 0
        var means_words = 0
        if posterior:
            var post = posts[s]
            if post[0] is not None:
                has = True
                nbl = _nbl_of(post[0])
            sig_words = Int(py=post[1].size)
            al_words = Int(py=post[2].size)
            means_words = Int(py=post[3].size)
            sig = Int(py=classical_address(post[1])) if sig_words > 0 else 0
            al = Int(py=classical_address(post[2])) if al_words > 0 else 0
            means = Int(py=classical_address(post[3])) if means_words > 0 else 0
        out_steps.append(IterImpStep(Int(py=step[0]), Int(py=classical_address(step[1])),
                                     Int(py=classical_address(step[2])), has, nbl^, sig, sig_words, al, al_words,
                                     means, means_words))
    return out_steps^


struct _HeldSteps(Movable):
    """transform's hold on the fitted steps' member states (`_held_steps`).
    `native` False: the door rounds serve the transform. True: `steps` (in
    order, each step's held plan and state), `keepers` the estimators' owner
    objects (kept named until the RUN has joined), `loose` the steps whose
    state no owner keeps (freed by `_held_free`), `release` every step's
    kind allows its member calls with the GIL released."""
    var native: Bool
    var steps: List[IterImpHeldStep]
    var keepers: PythonObject
    var loose: List[Bool]
    var release: Bool

    def __init__(out self):
        self.native = False
        self.steps = List[IterImpHeldStep]()
        self.keepers = PythonObject(None)
        self.loose = List[Bool]()
        self.release = False


def _held_free(mut held: _HeldSteps):
    """Free the states no estimator keeps (never raises; after the RUN, or
    on a refusal)."""
    for s in range(len(held.steps)):  # driver-loop: over the fitted imputation steps (one state each)
        if held.loose[s] and held.steps[s].state != 0:
            try:
                var call = member_call_of(held.steps[s].plan, held.steps[s].state)
                member_free(held.steps[s].plan, call)
            except:
                pass  # member_call_of builds words only; a state that cannot be freed is the store's
            held.steps[s].state = 0
    held.native = False


def _held_steps(sequence: PythonObject, step_classes: PythonObject, posterior: Bool,
                context: PythonObject) -> _HeldSteps:
    """UNPACK, once per transform: every fitted step's estimator as its HELD
    member state (bindings/member_held_door.mojo `member_held`: the state its
    fit handed it, a hit; else one import, kept), when each is a kind
    serving PREDICT (and, with sample_posterior, PREDICT_STD with the mean:
    `_posterior_family`). A classifier step (lane mf-iterimp3) also needs
    the labels its fit stored (`step_classes`, the fitted `_step_classes`)
    and the element type of its codes (asked once per transform: the kind's).
    An empty sequence (fewer than two kept features: n_iter_ = 0) is served
    with no step. Never raises: a step the member route cannot serve sends
    the whole transform to the door rounds (the fallback rule), and every
    state taken so far is released."""
    var out = _HeldSteps()
    # TOMBSTONE: MOJOLEARN_MEMBER_SET_OFF (plumbing) deleted 2026-11-23 by lane/ab-prep-classical; code recoverable at 4d76a9e16e.
    try:
        var b = _b()
        out.keepers = Python.list()
        var needs = MEMBER_CAN_PREDICT
        if posterior:
            needs |= MEMBER_CAN_PREDICT_STD
        var release = True
        var code_dtype = -1
        for s in range(len(sequence)):  # driver-loop: over the fitted imputation steps (one held-state read each)
            var step = sequence[s]
            var est = step[2]
            var plan = prep_estimator_plan(est, context)
            if plan is None or py_get(plan, "member") is None:
                raise Error("door rounds")
            var classifier = Bool(py=plan["classifier"])
            var labels = PythonObject(None)
            if classifier:
                if posterior or step_classes is None or s >= len(step_classes):
                    raise Error("door rounds")
                labels = step_classes[s]
                if labels is None:
                    raise Error("door rounds")
            var member_context = plan["context"]
            var fields = py_call(plan["fields_get"], est, py_type(est))
            var held = member_held(est, fields, plan, member_context, b, needs)
            if not held.published:
                raise Error("door rounds")
            # held first, so a refusal below releases it with the others
            out.steps.append(IterImpHeldStep(Int(py=step[0]), _nbl_of(step[1]), held.plan.copy(), held.state))
            out.loose.append(held.loose)
            _ = py_call(out.keepers.append, held.keeper)
            if posterior and not _posterior_family(held.plan.family):
                raise Error("door rounds")
            var ask = MemberCall(held.plan.route, held.state)
            var mask = member_call(held.plan.member, MEMBER_OP_CAPABILITIES, ask)
            if ((mask & MEMBER_CAN_CLASSIFIER) != 0) != classifier or (mask & needs) != needs:
                raise Error("door rounds")
            if classifier:
                var last = len(out.steps) - 1
                var values = _floats_from(labels)
                if len(values) == 0:
                    raise Error("door rounds")
                for c in range(len(values)):  # driver-loop: over one step's classes
                    out.steps[last].classes.append(Float32(values[c]))
                if code_dtype < 0:
                    code_dtype = member_output_dtype(held.plan, MEMBER_OP_PREDICT, max(len(out.steps[last].nbl), 1),
                                                     1, MEMBER_I32)
                out.steps[last].code_dtype = code_dtype
            release = release and (mask & (MEMBER_GIL_FREE | MEMBER_GIL_DOOR)) != 0
        out.release = release
        out.native = True
    except:
        _held_free(out)
        return _HeldSteps()
    return out^


def _apply_request(model: PythonObject, fields: PythonObject, imputer: PythonObject, n: Int, d: Int,
                   mark: Tuple[Bool, Float64], var keep: List[Int], var bounds: List[Float64],
                   var indicator: List[Int], posterior: Bool, rng: UInt64, caps: XprepCaps,
                   var steps: List[IterImpStep], mut fill: PythonObject, context: PythonObject) raises -> IterImpApply:
    """UNPACK of transform's typed request: the initial imputer's transform
    words (`_imputer_transform`'s UNPACK; its fitted fill vector is returned
    in `fill`, which the caller keeps alive through the RUN) and the fitted
    fields."""
    var nk = prep_keys()
    var own = py_fields(imputer)
    _check_width(imputer, d)
    var init_mark = _missing_mark(own["missing_values"])
    fill = _fitted_flat(imputer, "_fill", own["_fill"], context)
    var fill_words = Int(py=fill[2])
    var init = ImputerApply(n, d, 0, init_mark[0], init_mark[1], 1, fill_words, _index_list(own["_keep"]),
                            _index_list(own["_indicator"]))
    return IterImpApply(init^, Int(py=classical_address(fill[1])), n, d, mark[0], mark[1], xprep_r3("work"),
                        caps.has_out, keep^, bounds^, indicator^, posterior, Int(rng & 0x7FFFFFFF),
                        Int(py=py_call(_b().int, py_item(fields, nk[].k__post_step))), steps^)


def _transform(model: PythonObject, x: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = prep_keys()
    # UNPACK
    var fields = py_fields(model)
    if not py_contains(fields, nk[].k_imputation_sequence_):
        raise Error("RuntimeError: mojolearn: this IterativeImputer instance is not fitted yet")
    var mode = py_item(fields, nk[].k_numeric_mode_)
    var entries = scaler_entries(model, mode, context)  # once per operation; passed to every round's programs
    var caps = enc_caps(model, entries)
    var values = xprep_values(x, entries, context)
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    _check_width(model, d)
    var keep = _index_list(py_item(fields, nk[].k__keep))
    var dk = len(keep)
    var bounds = _bounds_of(model)
    var imputer = py_item(fields, nk[].k_initial_imputer_)
    if not py_contains(py_fields(imputer), nk[].k_n_features_in_):
        raise Error("RuntimeError: mojolearn: this " + _name(imputer) + " instance is not fitted yet")
    var posterior = _truth(py_item(fields, nk[].k_sample_posterior))
    var rng = _u64_of(py_item(fields, nk[].k__rng))
    var sequence = py_item(fields, nk[].k_imputation_sequence_)
    var indicator = _index_list(py_item(fields, nk[].k__indicator))
    var mark = _missing_mark(py_item(fields, nk[].k_missing_values))
    # the kept positions marked not missing (scikit-learn's `_is_empty_feature`)
    var empty_field = py_get(fields, "_empty_kept")
    var empty = _index_list(empty_field) if empty_field is not None else List[Int]()
    var runner = _DoorRunner(values, entries, context, caps)
    if py_item(fields, nk[].k_estimator) is not None:
        # The fitted steps' held member states (lane mf-iterimp2): the same
        # step programs as the door rounds below, the members' PREDICT /
        # PREDICT_STD on their fitted states (no import on a hit, no Python
        # estimator call).
        var held = _held_steps(sequence, py_get(fields, "_step_classes"), posterior, context)
        if held.native:
            var fill = PythonObject(None)
            var request = _apply_request(model, fields, imputer, n, d, mark, keep.copy(), bounds.copy(),
                                         indicator.copy(), posterior, rng, caps, List[IterImpStep](), fill, context)
            request.empty = empty.copy()
            runner.release_members = held.release
            var applied = IterImpApplied(II_NONE, 0, 0, 0)
            try:
                # RUN (x_prep/iterimp_run.mojo)
                applied = iterimp_member_transform(runner, request, held.steps, rng)
            except error:
                _held_free(held)
                raise error^
            _held_free(held)
            _ = held.keepers  # the estimators' owner objects stay alive past the member calls
            _ = fill  # the fitted fill vector the RUN read stays alive past it
            # PACK
            py_setitem(fields, nk[].k__rng, _u64_obj(rng))
            return _take2(runner.held[applied.out_handle], applied.out_offset, n, applied.cols, context)
        # `initial_imputer_.transform(arr)`: X as converted above
        var xf = _imputer_transform(imputer, values, context)
        # The mask runs in the process mode (as the Python ran it): the
        # operation's selection when that is the fitted mode's object, else
        # its own selection, once.
        var mask_mode = _default_mode(context)
        var mask = _mask_arr(values, keep, empty, py_item(fields, nk[].k_missing_values),
                             entries if mask_mode is mode else _entries(mask_mode, context), context)
        var xt = xf
        for s in range(len(sequence)):  # api-bound-loop: over the fitted imputation steps; one predict of the user-supplied fitted estimator object's native plan each
            var step = sequence[s]
            var j = Int(py=step[0])
            var took = _ii_take(xt, mask, n, dk, j, _nbl_of(step[1]), entries, False, context)
            xt = _ii_put(xt, took.xm, took.rows, took.m, n, dk, j, step[2], posterior, bounds, entries, rng, context)
        py_setitem(fields, nk[].k__rng, _u64_obj(rng))
        if len(indicator) == 0:
            return xt
        var made = iterimp_indicator(runner, n, d, dk, mark[0], mark[1], indicator, _xt_address(xt, n * dk),
                                     xprep_r3("work"), caps.has_out)
        _ = xt  # the block the indicator program read stays alive past it
        return _take2(runner.held[made[0]], made[1], n, dk + len(indicator), context)
    # The initial imputer's transform words (`_imputer_transform`'s UNPACK)
    var posts = py_item(fields, nk[].k__posterior) if posterior else PythonObject(None)
    var fill = PythonObject(None)
    var request = _apply_request(model, fields, imputer, n, d, mark, keep^, bounds^, indicator^, posterior, rng, caps,
                                 _steps(sequence, posts, posterior), fill, context)
    request.empty = empty^
    # RUN (x_prep/iterimp_run.mojo)
    var applied = iterimp_transform(runner, request)
    _ = fill  # the fitted fill vector the RUN read stays alive past it
    _ = sequence  # and the fitted steps' vectors
    _ = posts
    # PACK
    py_setitem(fields, nk[].k__post_step, PythonObject(applied.post_step))
    return _take2(runner.held[applied.out_handle], applied.out_offset, n, applied.cols, context)


# ---------------------------------------------------------------- the door
def prep_iterimp_call(model: PythonObject, op: Int, operation: String, args: PythonObject,
                      context: PythonObject) raises -> PythonObject:
    """IterativeImputer.fit(X, y=None) (returns the model), fit_transform(X,
    y=None) and transform(X)."""
    if op == PREP_OPN_FIT:
        _ = _fit_transform(model, args[0], context)
        return model
    if op == PREP_OPN_FIT_TRANSFORM:
        return _fit_transform(model, args[0], context)
    if op == PREP_OPN_TRANSFORM:
        return _transform(model, args[0], context)
    raise Error("TypeError: the native IterativeImputer owner does not serve " + operation)
