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
# its own fit writes) for `imputation_sequence_`, and raises the fits'
# notices. The door rounds below stay for an estimator the member route does
# not serve, and nothing else: a class with no member entry, an admission the
# family refuses (the fallback rule), sample_posterior with any family but
# GaussianProcessRegressor (the only member whose PREDICT_STD also answers
# the same query's mean), fewer than two kept features, a kept column with
# no observed value (its fit would see no rows: the estimator's own door
# refuses that in its words), and the member-set B arm
# (MOJOLEARN_MEMBER_SET_OFF). There the estimator is cloned, fitted and
# asked to predict once per (round, feature) step with the float32 blocks the
# gather programs produce, and its predictions are scattered back by one
# program; the clone, fit and predict are the compiled operations of the
# estimator's connected native plan (bindings/prep_native_estimator.mojo);
# its Python fit, predict, get_params or constructor is never called.
# transform with a user estimator predicts with the fitted estimators of
# `imputation_sequence_` (public objects) through their plans, as before.
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.member_boundary_door import member_admit_serving, member_pack
from bindings.member_fit_notices_pack import member_fit_notices_raise
from bindings.member_owner_unpack import MemberOwner
from bindings.member_set_door import MEMBER_SET_NATIVE
from core.member_abi import (
    MemberPlan, MemberCall, member_call, member_kind_capabilities,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_PREDICT_STD, MEMBER_CAN_CLASSIFIER, MEMBER_BORROWS_FIT,
    MEMBER_GIL_FREE, MEMBER_GIL_DOOR,
)
from core.member_families import MEMBER_FAMILY_GP
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
    IterImpMember, IterImpMemberFit, iterimp_member_rounds, iterimp_member_free)

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
        raise Error("mojolearn: If 'sample_posterior' is True, the estimator must support "
                    + "'return_std' in its 'predict' method.")


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


def _mask_arr(values: PythonObject, keep: List[Int], missing: PythonObject, entries: PythonObject,
              context: PythonObject) raises -> PythonObject:
    """`_mask_arr`: the (n, dk) float32 missing mask (1.0 missing), run in
    the process mode as the Python ran it: `entries` is the caller's
    selection for that mode (selected once per operation)."""
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var dk = len(keep)
    var p = EncProg()
    var xo = _mark_missing(p, p.put_x(values), n * d, missing, context)
    var mo = p.alloc(n * dk)
    var ko = p.put_floats(_floats_of(keep), context)
    _stage(p.prog.program, OP_NAN_MASK, n * dk, xo, n, d, ko, dk, mo)
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
    `borrows` / `release` the kind's MEMBER_BORROWS_FIT and GIL bits."""
    var native: Bool
    var owner: MemberOwner
    var template: PythonObject
    var fields: PythonObject
    var member_context: PythonObject
    var fields_get: PythonObject
    var borrows: Bool
    var release: Bool

    def __init__(out self):
        self.native = False
        self.owner = MemberOwner()
        self.template = PythonObject(None)
        self.fields = PythonObject(None)
        self.member_context = PythonObject(None)
        self.fields_get = PythonObject(None)
        self.borrows = False
        self.release = False


def _member(estimator: PythonObject, posterior: Bool, context: PythonObject) -> _Member:
    """UNPACK, once per fit: the estimator as a native member when its class
    is a published member whose KIND fits and predicts one target (a
    regressor), and for sample_posterior is the GaussianProcessRegressor
    member (its PREDICT_STD answers the same query's mean too). Never
    raises: anything else, and an admission the family refuses, keeps the
    door rounds (the fallback rule), which then answer or refuse in the
    estimator's own words."""
    var out = _Member()
    comptime if not MEMBER_SET_NATIVE:
        return out^
    try:
        var b = _b()
        var plan = prep_estimator_plan(estimator, context)
        if plan is None or py_get(plan, "member") is None:
            return out^
        if Bool(py=plan["classifier"]):
            return out^
        var needs = MEMBER_CAN_FIT | MEMBER_CAN_PREDICT
        if posterior:
            needs |= MEMBER_CAN_PREDICT_STD
        var template = prep_estimator_clone(estimator, context)
        var member_context = plan["context"]
        var fields_get = plan["fields_get"]
        var fields = py_call(fields_get, template, py_type(template))
        var owner = member_admit_serving(template, fields, plan, member_context, b, needs)
        if not owner.published:
            return out^
        if posterior and owner.plan.family != MEMBER_FAMILY_GP:
            return out^
        # The mask of THIS kind (the family's union holds its other kinds'
        # bits): one state minted, asked and released, once per fit.
        var mask = member_kind_capabilities(owner.plan)
        if (mask & MEMBER_CAN_CLASSIFIER) != 0 or (mask & needs) != needs:
            return out^
        out.borrows = (mask & MEMBER_BORROWS_FIT) != 0
        out.release = (mask & (MEMBER_GIL_FREE | MEMBER_GIL_DOOR)) != 0
        out.owner = owner^
        out.template = template
        out.fields = fields
        out.member_context = member_context
        out.fields_get = fields_get
        out.native = True
    except:
        return _Member()
    return out^


def _member_pack(estimator: PythonObject, member: _Member, fit: IterImpMemberFit,
                 context: PythonObject) raises -> PythonObject:
    """PACK of the member rounds: `imputation_sequence_`, one (feat_idx,
    neighbor_feat_idx, estimator) triple per finished step, each estimator a
    clone of the supplied one (the first is the admission's) with the step's
    member state written into it (`member_pack`: the attributes its own fit
    writes). One clone per step, as the door rounds made."""
    var b = _b()
    var seq = Python.list()
    for s in range(len(fit.states)):  # driver-loop: over the fitted imputation steps (one clone and one pack each)
        var est = member.template
        var fields = member.fields
        if s > 0:
            est = prep_estimator_clone(estimator, context)
            fields = py_call(member.fields_get, est, py_type(est))
        member_pack(est, fields, member.member_context, b, member.owner.plan, fit.states[s])
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
    var member = _Member()
    if dk >= 2 and not fitted.empty_kept:
        member = _member(estimator, request.posterior, context)
    if member.native:
        # RUN: the rounds over the native member (x_prep/iterimp_run.mojo)
        runner.release_members = member.release
        var run = IterImpMember(member.owner.plan.copy(), member.borrows, request.posterior)
        var fit = IterImpMemberFit()
        iterimp_member_rounds(runner, request, plan, fitted, run, fit)
        # PACK (every state is released after it, on every path)
        try:
            var seq = _member_pack(estimator, member, fit, context)
            py_setitem(fields, nk[].k__rng, _u64_obj(fit.rng))
            py_setitem(fields, nk[].k__post_step, PythonObject(0))
            py_setitem(fields, nk[].k_n_iter_, PythonObject(fit.n_iter))
            py_setitem(fields, nk[].k_imputation_sequence_, seq)
        finally:
            iterimp_member_free(run, fit)
        # The steps' fit notices, in step order, as the estimator's own fits raised them.
        member_fit_notices_raise(fit.notices, member.member_context)
        _ = member.owner.published  # the member's images stay loaded past its last call
        return _take2(runner.held[fit.out_handle], fit.out_offset, n, fit.cols, context)
    var bounds = fitted.bounds.copy()
    var posterior = request.posterior
    var rng = plan.rng
    var mask = _mask_arr(values, keep, py_item(fields, nk[].k_missing_values), entries, context)
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
    var runner = _DoorRunner(values, entries, context, caps)
    if py_item(fields, nk[].k_estimator) is not None:
        # `initial_imputer_.transform(arr)`: X as converted above
        var xf = _imputer_transform(imputer, values, context)
        # The mask runs in the process mode (as the Python ran it): the
        # operation's selection when that is the fitted mode's object, else
        # its own selection, once.
        var mask_mode = _default_mode(context)
        var mask = _mask_arr(values, keep, py_item(fields, nk[].k_missing_values),
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
    var own = py_fields(imputer)
    _check_width(imputer, d)
    var init_mark = _missing_mark(own["missing_values"])
    var fill = _fitted_flat(imputer, "_fill", own["_fill"], context)
    var fill_words = Int(py=fill[2])
    var init = ImputerApply(n, d, 0, init_mark[0], init_mark[1], 1, fill_words, _index_list(own["_keep"]),
                            _index_list(own["_indicator"]))
    var posts = py_item(fields, nk[].k__posterior) if posterior else PythonObject(None)
    var request = IterImpApply(init^, Int(py=classical_address(fill[1])), n, d, mark[0], mark[1], xprep_r3("work"),
                               caps.has_out, keep^, bounds^, indicator^, posterior, Int(rng & 0x7FFFFFFF),
                               Int(py=py_call(_b().int, py_item(fields, nk[].k__post_step))),
                               _steps(sequence, posts, posterior))
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
