# SPDX-License-Identifier: Apache-2.0
# The native owner of IterativeImputer (dispatched by bindings/prep_owner_native.mojo).
# fit, fit_transform and transform are each one call here. The owner checks the
# parameters, fits the initial SimpleImputer through the encode owner's native
# SimpleImputer (bindings/prep_encode_native.mojo), and builds the x_prep programs
# the Python `_Prog` built (the same stages, operands and arena order, so the same
# bits): the statistics and imputation-order program, the n_nearest_features draw
# program, the one program of rounds for the built-in Bayesian ridge, and
# `_with_indicator`. A user estimator keeps the reference's rounds: it is cloned,
# fitted and asked to predict once per (round, feature) step with the float32
# blocks the gather programs produce, and its predictions are scattered back by
# one program. The clone, fit and predict are the compiled operations of the
# estimator's connected native plan (bindings/prep_native_estimator.mojo); its
# Python fit, predict, get_params or constructor is never called.
from std.math import isfinite
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.classical_arrays import classical_typed, classical_wrap_array
from bindings.classical_labels import LabelsView, labels_view
from bindings.scaler_native import _stage, _ptr
from bindings.scaler_lifecycle_native import _entries
from bindings.x_prep_native_owner import xprep_values
from bindings.user_objects_native import user_objects_call
from bindings.prep_stateless_native import _int_like, _repr, _name
from bindings.prep_native_estimator import prep_estimator_admit, prep_estimator_clone, prep_estimator_call
from bindings.prep_encode_native import (
    EncProg, _cs, _mark_missing, _imputer_fit, _imputer_transform, _precheck, _default_mode, _word, _iword,
    _take1, _take2, _seed64, _seed_words, _truth, _float, _is_str, _check_width, _int_list, _index_list)

alias F32P = MutPointer[Float32, MutUntrackedOrigin]
alias F64P = MutPointer[Float64, MutUntrackedOrigin]

# x_prep stage codes (_expansion_prep._OPS).
comptime OP_COL_STATS = 1
comptime OP_EIGH = 18
comptime OP_II_MEAN = 53
comptime OP_II_GRAM = 54
comptime OP_II_SUB = 55
comptime OP_II_BR = 56
comptime OP_II_PREDICT = 57
comptime OP_II_SNAPSHOT = 58
comptime OP_II_CONV = 59
comptime OP_NAN_MASK = 60
comptime OP_II_SIGMA = 99
comptime OP_II_POST = 100
comptime OP_II_ROWABS = 107
comptime OP_UNIQ_SCAN = 121
comptime OP_II_RCOUNT = 136
comptime OP_II_RWRITE = 137
comptime OP_II_GATHER = 138
comptime OP_II_SCATTER = 139
comptime OP_HCAT = 140
comptime OP_ABSCORR_CELL = 211
comptime OP_ABSCORR_NORM = 212
comptime OP_F2_WPICK = 234
comptime OP_C2_II_MISS = 258
comptime OP_C2_II_POS = 259
comptime OP_C2_II_ORD = 260
comptime OP_C2_II_RAND = 261
comptime OP_C2_COLMAX = 262

comptime NONE = -1
comptime II_CH = 1024
comptime GOLDEN: UInt64 = 0x9E3779B97F4A7C15


# ---------------------------------------------------------------- small helpers
def _b() raises -> PythonObject:
    return Python.import_module("builtins")


def _u64_of(value: PythonObject) raises -> UInt64:
    # A Python int (already reduced mod 2**64) as its 64-bit word.
    var words = _seed64(value)
    return (UInt64(words[1]) << 32) | UInt64(words[0])


def _u64_obj(value: UInt64) raises -> PythonObject:
    # The 64-bit word as a nonnegative Python int.
    return py_call(_b().int, PythonObject(String(value)))


def _seed_put(mut p: EncProg, value: UInt64, context: PythonObject) raises -> Int:
    # `_seed_words(pr, seed)`: the low and high words as signed int32.
    return p.put_ints(_seed_words(Int(value & 0xFFFFFFFF), Int(value >> 32)), context)


def _ones(nbl: List[Int], dk: Int) -> List[Float64]:
    # `[1 if a in nbl else 0 for a in range(dk)]`
    var out = List[Float64]()
    for a in range(dk):  # driver-loop: over the kept features (one step's neighbour mask)
        var hit = 0.0
        for i in range(len(nbl)):  # driver-loop: over one step's drawn features
            if nbl[i] == a:
                hit = 1.0
        out.append(hit)
    return out^


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
    # `Array.from_list([], "<f4")`
    return classical_wrap_array(py_call(context["array_type"], "f"), Python.tuple(0), "<f4", "C", context["output_context"])


def _f32_vector(values: List[Float64], context: PythonObject) raises -> PythonObject:
    # The values as a float32 vector (`Array(...).astype("<f4")`).
    var out = classical_typed(Python.tuple(len(values)), "<f4", context["output_context"])
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
    var out = List[Float64]()
    _ = words  # keep the Python object alive through the native call above
    for i in range(n):  # cpu-route: the user estimator's answer, read once
        out.append(src[i])
        _ = words  # keep the Python object alive through the native call above
    return out^


def _pred_f64(v: PythonObject, context: PythonObject) raises -> List[Float64]:
    """`_pred_f64`: a float32 or float64 vector's values widened (exact);
    anything else float(x) per entry of its `tolist()` (`_as_list`)."""
    var b = _b()
    var cls = py_type(v)
    if not (cls is b.list or cls is b.tuple):
        var found = Optional[LabelsView](None)
        try:
            found = labels_view(v, "predict", context)
        except:
            found = None
        if found:
            var view = found.value().copy()
            var t = view.typestr
            if view.ndim() == 1 and (t == "<f4" or t == "<f8") and view.size() > 0:
                var out = List[Float64]()
                var step = view.strides[0]
                for i in range(view.size()):  # cpu-route: the user estimator's predictions, widened once
                    if t == "<f4":
                        out.append(Float64(F32P(unsafe_from_address=view.addr + i * step)[0]))
                    else:
                        out.append(F64P(unsafe_from_address=view.addr + i * step)[0])
                _ = view  # keep the Python object alive through the reads above
                return out^
    var listed = py_call(v.tolist) if Bool(py=py_call(b.hasattr, v, "tolist")) else v
    return _floats_from(listed)


# ---------------------------------------------------------------- parameters
def _refuse(model: PythonObject, context: PythonObject) raises:
    # `_refuse`: the parameter refusals, before X is read.
    var b = _b()
    var fields = py_fields(model)
    var estimator = fields["estimator"]
    if estimator is not None:
        # A user estimator is a mojolearn estimator with a connected native
        # plan (its clone, fit and predict are compiled operations); any other
        # class is refused by name first.
        prep_estimator_admit(estimator, "IterativeImputer", context)
    var mi = fields["max_iter"]
    if Bool(py=py_call(b.isinstance, mi, b.bool)) or not _int_like(mi, context) or Int(py=py_call(b.int, mi)) < 0:
        raise Error("The 'max_iter' parameter of IterativeImputer must be an int in the range [0, inf). Got "
                    + _repr(mi) + " instead.")
    var order = fields["imputation_order"]
    var text = String(order) if _is_str(order) else String("")
    if text != "ascending" and text != "descending" and text != "roman" and text != "arabic" and text != "random":
        raise Error("mojolearn: invalid imputation_order " + _repr(order))
    var nnf = fields["n_nearest_features"]
    if nnf is not None and (not _int_like(nnf, context) or Int(py=py_call(b.int, nnf)) < 1):
        raise Error("mojolearn: n_nearest_features must be an int >= 1; got " + _repr(nnf))
    if _truth(fields["sample_posterior"]) and estimator is not None:
        # No connected native plan's predict takes return_std (the plans'
        # predict operations take X only), so this is the reference's refusal
        # without inspecting the Python method's signature.
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
    for i in range(len(vals)):  # cpu-route: converts the min_value / max_value arguments once
        out.append(_float(vals[i]))
    return out^


def _bounds_k(model: PythonObject, d: Int, keep: List[Int]) raises -> List[Float64]:
    # `_bounds(d)` interleaved (zip stops at the shorter), then the kept columns'.
    var fields = py_fields(model)
    var lo = _per_bound(fields["min_value"], d)
    var hi = _per_bound(fields["max_value"], d)
    var bounds = List[Float64]()
    for i in range(min(len(lo), len(hi))):  # driver-loop: over the per-column bound pairs
        bounds.append(lo[i])
        bounds.append(hi[i])
    var out = List[Float64]()
    for c in range(len(keep)):  # driver-loop: over the kept columns
        for h in range(2):  # small-loop(h: the lower and upper bound)
            var at = 2 * keep[c] + h
            if at >= len(bounds):
                raise Error("IndexError: list index out of range")
            out.append(bounds[at])
    return out^


def _float_list(values: List[Float64]) raises -> PythonObject:
    var out = Python.list()
    for i in range(len(values)):  # driver-loop: over fitted per-column bounds
        _ = py_call(out.append, PythonObject(values[i]))
    return out


def _bounds_of(model: PythonObject) raises -> List[Float64]:
    # The fitted `_bounds_k` list.
    var items = py_call(_b().list, py_fields(model)["_bounds_k"])
    var out = List[Float64]()
    for i in range(len(items)):  # driver-loop: over fitted per-column bounds
        out.append(_float(items[i]))
    return out^


# ---------------------------------------------------------------- the draws
struct _Draws(Movable):
    """`_NeighbourDraws`: every (round, feature) step's n_nearest_features
    draw, handed out in fit order; each hand-out moves the stream past the
    words the steps taken so far own (k per step)."""
    var lists: List[List[Int]]
    var at: Int
    var k: Int
    var start: UInt64
    var on: Bool

    def __init__(out self):
        self.lists = List[List[Int]]()
        self.at = 0
        self.k = 0
        self.start = 0
        self.on = False

    def next(mut self, mut rng: UInt64) -> List[Int]:
        var out = self.lists[self.at].copy()
        self.at += 1
        rng = self.start + UInt64(self.at) * UInt64(self.k) * GOLDEN
        return out^


def _neighbours(xf: PythonObject, n: Int, dk: Int, mode: PythonObject, js: List[Int], k: Int, rng: UInt64,
                context: PythonObject) raises -> List[List[Int]]:
    """`_neighbours_device`: the |corr| matrix on the device and one thread per
    step drawing its k features (f2_wpick, step c takes splitmix64 words
    c*k .. c*k + k - 1 of the stream). Returns each step's ascending picks."""
    var entries = _entries(mode, context)
    var p = EncProg()
    var fo = p.put(xf, n * dk)
    var mz = p.alloc(n * dk)
    var means = p.alloc(dk)
    var cnt = p.alloc(1)
    var g = p.alloc(dk * dk)
    var flag = p.alloc(1)
    _stage(p.program, OP_II_MEAN, dk, fo, n, dk, mz, 0, means, cnt, flag)
    _stage(p.program, OP_II_GRAM, dk * dk, fo, n, dk, mz, 0, means, g, flag)
    var m = p.alloc(dk * dk)
    _stage(p.program, OP_ABSCORR_CELL, dk * dk, g, dk, m)
    _stage(p.program, OP_ABSCORR_NORM, dk, m, dk)
    var calls = len(js)
    var fl = p.work(calls * dk)
    var out = p.alloc(calls * (k + 1))
    var jo = p.put_ints(js, context)
    var so = _seed_put(p, rng, context)
    _stage(p.program, OP_F2_WPICK, calls, m, dk, jo, k, so, fl, out)
    var ran = p.run(entries, context)
    var lists = List[List[Int]]()
    for c in range(calls):  # driver-loop: one feature list per step
        var got = _iword(ran, out + c * (k + 1) + k)
        if got < k:
            raise Error("mojolearn: IterativeImputer: no feature left to draw")
        var picks = List[Int]()
        for i in range(got):  # driver-loop: over one step's drawn features
            picks.append(_iword(ran, out + c * (k + 1) + i))
        lists.append(picks^)
    return lists^


# ---------------------------------------------------------------- shared programs
def _prepare(mut p: EncProg, values: PythonObject, xf: PythonObject, keep: List[Int], bounds: List[Float64],
             missing: PythonObject, context: PythonObject) raises -> Tuple[Int, Int, Int]:
    """`_prepare(pr, arr, Xf, inout=True)`: the filled block (rewritten in
    place and read back), its missing mask, the per-column bounds."""
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var dk = len(keep)
    var xo = _mark_missing(p, p.put_x(values), n * d, missing, context)
    var fo = p.put_inout(xf, n * dk)
    var ko = p.put_floats(_floats_of(keep), context)
    var mo = p.alloc(n * dk)
    _stage(p.program, OP_NAN_MASK, n * dk, xo, n, d, ko, dk, mo)
    var bo = p.put_floats(bounds, context)
    return (fo, mo, bo)


def _with_indicator(model: PythonObject, values: PythonObject, xt: PythonObject, dk: Int,
                    context: PythonObject) raises -> PythonObject:
    """`_with_indicator`: Xt, then MissingIndicator's columns (add_indicator)."""
    var fields = py_fields(model)
    var indicator = _index_list(fields["_indicator"])
    var m = len(indicator)
    if m == 0:
        return xt
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var entries = _entries(fields["numeric_mode_"], context)
    var p = EncProg()
    var xo = _mark_missing(p, p.put_x(values), n * d, fields["missing_values"], context)
    var mo = p.alloc(n * m)
    var io = p.put_floats(_floats_of(indicator), context)
    _stage(p.program, OP_NAN_MASK, n * m, xo, n, d, io, m, mo)
    var to = p.put(xt, n * dk)
    var hc = p.output(n * (dk + m), entries)
    _stage(p.program, OP_HCAT, n * (dk + m), to, dk, mo, m, hc)
    var ran = p.run(entries, context)
    return _take2(ran, hc, n, dk + m, context)


def _mask_arr(values: PythonObject, keep: List[Int], missing: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_mask_arr`: the (n, dk) float32 missing mask (1.0 missing), run in
    the process mode as the Python ran it."""
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var dk = len(keep)
    var mode = _default_mode(context)
    var p = EncProg()
    var xo = _mark_missing(p, p.put_x(values), n * d, missing, context)
    var mo = p.alloc(n * dk)
    var ko = p.put_floats(_floats_of(keep), context)
    _stage(p.program, OP_NAN_MASK, n * dk, xo, n, d, ko, dk, mo)
    var ran = p.run(_entries(mode, context), context)
    return _take2(ran, mo, n, dk, context)


# ---------------------------------------------------------------- the user-estimator route
def _ii_rows(mut p: EncProg, mo: Int, n: Int, dk: Int, j: Int, missing: Bool) -> Tuple[Int, Int]:
    """`_ii_rows`: the ascending rows whose mask word for feature j is (or is
    not) zero, as int32 words, and their count."""
    var ch = max(II_CH, (n + II_CH - 1) // II_CH)
    var nch = (n + ch - 1) // ch
    var cnt = p.alloc(nch)
    var off = p.alloc(nch)
    var tot = p.alloc(1)
    var rows = p.alloc(n)
    var flag = 1 if missing else 0
    _stage(p.program, OP_II_RCOUNT, nch, mo, n, dk, j, flag, ch, cnt)
    _stage(p.program, OP_UNIQ_SCAN, 1, cnt, nch, off, tot)
    _stage(p.program, OP_II_RWRITE, nch, mo, n, dk, j, flag, ch, off, rows)
    return (rows, tot)


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


def _ii_take(xt: PythonObject, mask: PythonObject, n: Int, dk: Int, j: Int, nbl: List[Int], mode: PythonObject,
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
    var mis = _ii_rows(p, mo, n, dk, j, True)
    var gm = p.alloc(n * nc)
    if nc > 0:
        _stage(p.program, OP_II_GATHER, n * nc, xo, dk, mis[0], mis[1], co, nc, gm)
    var obs = (0, 0)
    var go = 0
    var gy = 0
    if fit:
        obs = _ii_rows(p, mo, n, dk, j, False)
        go = p.alloc(n * nc)
        gy = p.alloc(n)
        if nc > 0:
            _stage(p.program, OP_II_GATHER, n * nc, xo, dk, obs[0], obs[1], co, nc, go)
        _stage(p.program, OP_II_GATHER, n, xo, dk, obs[0], obs[1], jo, 1, gy)
    var ran = p.run(_entries(mode, context), context)
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


def _ii_put(xt: PythonObject, xm: PythonObject, rows: PythonObject, m: Int, n: Int, dk: Int, j: Int,
            est: PythonObject, posterior: Bool, bounds: List[Float64], mode: PythonObject, mut rng: UInt64,
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
        # `_refuse` refused sample_posterior with a user estimator.
        raise Error("mojolearn: If 'sample_posterior' is True, the estimator must support "
                    + "'return_std' in its 'predict' method.")
    v = _pred_f64(prep_estimator_call(est, "predict", Python.tuple(xm), context), context)
    var k = min(len(v), m)  # the reference's zip stops at the shorter
    if k == 0:
        return xt
    while len(v) > k:  # driver-loop: drops predictions past the missing rows
        _ = v.pop()
    var entries = _entries(mode, context)
    var p = EncProg()
    var xo = p.put_inout(xt, n * dk)
    var ro = p.put(rows, m)
    var vo = p.put(_f32_vector(v, context), k)
    var lohi = List[Float64]()
    lohi.append(lo)
    lohi.append(hi)
    var bo = p.put_floats(lohi, context)
    _stage(p.program, OP_II_SCATTER, k, xo, dk, ro, j, vo, bo, clip)
    var ran = p.run(entries, context)
    return _take2(ran, xo, n, dk, context)


def _ii_stop(xt: PythonObject, prev: PythonObject, n: Int, dk: Int, tol: Float64, mode: PythonObject,
             context: PythonObject) raises -> Bool:
    # `_ii_stop`: the round's stop, the inf-norm of Xt - prev below tol.
    var p = EncProg()
    var xo = p.put(xt, n * dk)
    var po = p.put(prev, n * dk)
    var to = p.put_floats(_one(tol), context)
    var flag = p.alloc(1)
    var niter = p.alloc(1)
    var rowabs = p.alloc(n)
    _stage(p.program, OP_II_ROWABS, n, xo, po, dk, rowabs, flag)
    _stage(p.program, OP_II_CONV, 1, xo, po, n * dk, to, flag, niter, dk, rowabs + 1)
    var ran = p.run(_entries(mode, context), context)
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


# ---------------------------------------------------------------- fit
def _initial(model: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    """`SimpleImputer(missing_values, strategy=initial_strategy, fill_value,
    keep_empty_features).fit(arr)`, run by the encode owner's native fit."""
    var fields = py_fields(model)
    # a fresh instance holding SimpleImputer.__init__'s six parameters
    var cls = context["prep_owner_classes"]["SimpleImputer"]
    var imputer = py_call(_b().object.__new__, cls)
    var own = py_fields(imputer)
    py_setitem(own, "missing_values", fields["missing_values"])
    py_setitem(own, "strategy", fields["initial_strategy"])
    py_setitem(own, "fill_value", fields["fill_value"])
    py_setitem(own, "copy", PythonObject(True))
    py_setitem(own, "add_indicator", PythonObject(False))
    py_setitem(own, "keep_empty_features", fields["keep_empty_features"])
    _precheck(imputer, "simple_imputer", context)
    _imputer_fit(imputer, values, context)
    return imputer


def _fit_transform(model: PythonObject, x: PythonObject, context: PythonObject) raises -> PythonObject:
    _refuse(model, context)
    var b = _b()
    var fields = py_fields(model)
    var mode = _default_mode(context)
    var entries = _entries(mode, context)
    var values = xprep_values(x, entries, context)
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var rng = _u64_of(fields["random_state"])
    var post_step = 0
    py_setitem(fields, "_rng", _u64_obj(rng))
    py_setitem(fields, "_post_step", PythonObject(0))
    var imputer = _initial(model, values, context)
    py_setitem(fields, "initial_imputer_", imputer)
    var xf = _imputer_transform(imputer, values, context)
    var keep = _index_list(py_fields(imputer)["_keep"])
    py_setitem(fields, "_keep", py_call(b.list, py_fields(imputer)["_keep"]))
    var dk = len(keep)
    var bounds = _bounds_k(model, d, keep)
    py_setitem(fields, "_bounds_k", _float_list(bounds))
    var missing = fields["missing_values"]
    var order_name = String(fields["imputation_order"])
    var skip = _truth(fields["skip_complete"])
    var posterior = _truth(fields["sample_posterior"])
    # the statistics, the scale and the imputation orders: one program
    var p = EncProg()
    var xo = _mark_missing(p, p.put_x(values), n * d, missing, context)
    var st = p.alloc(6 * d)
    _cs(p, mode, entries, xo, n, d, st)
    var rounds = Int(py=py_call(b.int, fields["max_iter"]))
    var R = max(rounds, 1)
    var ordw = p.alloc(max(R * dk, 1))
    var lens = p.alloc(R)
    var sc = p.alloc(1)
    _stage(p.program, OP_C2_COLMAX, 1, st + 5 * d, d, sc)
    if dk > 0:
        var missd = p.work(dk)
        var ko = p.put_ints(keep, context)
        _stage(p.program, OP_C2_II_MISS, dk, st, ko, n, missd)
        if order_name == "random":
            var so = _seed_put(p, rng, context)
            _stage(p.program, OP_C2_II_RAND, R, missd, dk, 1 if skip else 0, so, ordw, lens)
        else:
            var pos = p.work(dk)
            var mcode = 0
            if order_name == "descending":
                mcode = 1
            elif order_name == "roman":
                mcode = 2
            elif order_name == "arabic":
                mcode = 3
            _stage(p.program, OP_C2_II_POS, dk, missd, dk, mcode, pos)
            _stage(p.program, OP_C2_II_ORD, dk, missd, dk, pos, R, ordw, lens)
    var ran = p.run(entries, context)
    var scale = Float64(_word(ran, sc))
    if not isfinite(scale) and scale == scale and scale > 0.0:
        raise Error("Input X contains infinity or a value too large for dtype('float64').")
    var indicator = List[Int]()
    if _truth(fields["add_indicator"]):
        for j in range(d):  # driver-loop: over the per-column non-missing counts (feature metadata)
            if Int(_word(ran, st + j)) < n:
                indicator.append(j)
    py_setitem(fields, "_indicator", _int_list(indicator))
    var ln = List[Int]()
    for r in range(R):  # driver-loop: over the imputation rounds
        ln.append(_iword(ran, lens + r) if dk > 0 else 0)
    var orders = List[List[Int]]()
    var any_order = False
    var total_steps = 0
    for r in range(rounds):  # driver-loop: the device's orders as control lists (imputation rounds)
        var order = List[Int]()
        for i in range(ln[r]):  # driver-loop: over one round's feature order
            order.append(_iword(ran, ordw + r * dk + i))
        any_order = any_order or len(order) > 0
        total_steps += len(order)
        orders.append(order^)
    py_setitem(fields, "n_features_with_missing_", PythonObject(ln[0]))
    if order_name == "random":
        var m = ln[0] if skip else dk
        rng = rng + UInt64(rounds) * UInt64(max(m - 1, 0)) * GOLDEN
        py_setitem(fields, "_rng", _u64_obj(rng))
    py_setitem(fields, "numeric_mode_", mode)
    py_setitem(fields, "n_features_in_", PythonObject(d))
    _ = ran^
    # the n_nearest_features draws, every step up front in one program
    var draws = _Draws()
    var nnf = fields["n_nearest_features"]
    if nnf is not None and Int(py=py_call(b.int, nnf)) < dk:
        draws.on = True
        draws.k = Int(py=py_call(b.int, nnf))
        draws.start = rng
        var js = List[Int]()
        for r in range(len(orders)):  # driver-loop: the (round, feature) step list
            for i in range(len(orders[r])):  # driver-loop: over one round's feature order
                js.append(orders[r][i])
        if len(js) > 0:
            draws.lists = _neighbours(xf, n, dk, mode, js, draws.k, rng, context)
    var tol = _float(fields["tol"]) * scale
    var estimator = fields["estimator"]
    if estimator is not None:
        # the reference's rounds over copies of the user estimator
        var mask = _mask_arr(values, keep, missing, context)
        var xt = xf
        var seq = Python.list()
        var done = 0
        for r in range(len(orders)):  # cpu-route: the user estimator, fitted once per imputation round (user code)
            var check = not posterior and len(orders[r]) > 0
            var prev = xt
            for i in range(len(orders[r])):  # cpu-route: the user estimator, fitted once per feature (user code)
                var j = orders[r][i]
                var nbl = draws.next(rng) if draws.on else _others(dk, j)
                var est = prep_estimator_clone(estimator, context)
                var took = _ii_take(xt, mask, n, dk, j, nbl, mode, True, context)
                _ = prep_estimator_call(est, "fit", Python.tuple(took.xo, took.yo), context)
                _ = py_call(seq.append, Python.tuple(j, _int_list(nbl), est))
                xt = _ii_put(xt, took.xm, took.rows, took.m, n, dk, j, est, posterior, bounds, mode, rng, context)
            done = r + 1
            if check and _ii_stop(xt, prev, n, dk, tol, mode, context):
                break
        py_setitem(fields, "_rng", _u64_obj(rng))
        py_setitem(fields, "n_iter_", PythonObject(done if any_order else min(1, len(orders))))
        py_setitem(fields, "imputation_sequence_", seq)
        return _with_indicator(model, values, xt, dk, context)
    # the built-in Bayesian ridge: every round in one program
    var q = EncProg()
    var prep = _prepare(q, values, xf, keep, bounds, missing, context)
    var fo = prep[0]
    var mo = prep[1]
    var bo = prep[2]
    var to = q.put_floats(_one(tol), context)
    var flag = q.alloc(1)
    var niter = q.alloc(1)
    var prev = q.alloc(n * dk)
    var cnt = q.alloc(1)
    var g = q.alloc(dk * dk)
    var p1 = max(dk - 1, 1)
    var gs = q.alloc(p1 * p1)
    var eig = q.alloc(p1)
    var vec = q.alloc(p1 * p1)
    var w = q.alloc(p1)
    var seq_j = List[Int]()
    var seq_coef = List[Int]()
    var seq_inter = List[Int]()
    var post_nbl = Python.list()
    var post_sig = List[Int]()
    var post_al = List[Int]()
    var post_means = List[Int]()
    var post_pp = List[Int]()
    var seed = Int(rng & 0x7FFFFFFF)
    var conv = not posterior
    var rowabs = -1
    for r in range(rounds):  # driver-loop: drives the device imputation rounds (imputation rounds)
        if len(orders[r]) > 0 and conv:
            _stage(q.program, OP_II_SNAPSHOT, n * dk, fo, prev, flag)
        for i in range(len(orders[r])):  # driver-loop: drives one device regression per feature (per-round feature order)
            var j = orders[r][i]
            var coef = q.alloc(dk)
            var inter = q.alloc(1)
            var means = q.alloc(dk)
            var nb1 = 0
            var pp = dk - 1
            var nbl_obj = PythonObject(None)
            if draws.on:
                var nbl = draws.next(rng)
                nb1 = q.put_floats(_ones(nbl, dk), context) + 1
                pp = len(nbl)
                nbl_obj = _int_list(nbl)
            var al = q.alloc(2) if posterior else -1
            seq_j.append(j)
            seq_coef.append(coef)
            seq_inter.append(inter)
            _stage(q.program, OP_II_MEAN, dk, fo, n, dk, mo, j, means, cnt, flag)
            _stage(q.program, OP_II_GRAM, dk * dk, fo, n, dk, mo, j, means, g, flag)
            if pp > 0:
                _stage(q.program, OP_II_SUB, 1, g, dk, j, gs, flag, nb1)
                _stage(q.program, OP_EIGH, 1, gs, pp, 0, eig, vec, 1)  # q[5]=1: cyclic eigh_unit, not round-robin
            _stage(q.program, OP_II_BR, 1, g, dk, j, eig, vec, means, cnt, coef, inter, flag, w, nb1, al + 1)
            if posterior:
                var sig = q.alloc(max(pp, 1) * max(pp, 1))
                _stage(q.program, OP_II_SIGMA, pp * pp, eig, vec, pp, al, sig, flag)
                var key_words = List[Int]()
                key_words.append(seed)
                key_words.append(post_step)
                var key = q.put_ints(key_words, context)
                post_step += 1
                _stage(q.program, OP_II_POST, n, fo, n, dk, mo, j, coef, inter, bo, flag, means, sig, al, nb1, key)
                _ = py_call(post_nbl.append, nbl_obj)
                post_sig.append(sig)
                post_al.append(al)
                post_means.append(means)
                post_pp.append(pp)
            else:
                _stage(q.program, OP_II_PREDICT, n, fo, n, dk, mo, j, coef, inter, bo, flag)
        if len(orders[r]) > 0 and conv:
            if rowabs < 0:
                rowabs = q.alloc(n)
            _stage(q.program, OP_II_ROWABS, n, fo, prev, dk, rowabs, flag)
            _stage(q.program, OP_II_CONV, 1, fo, prev, n * dk, to, flag, niter, dk, rowabs + 1)
    var got = q.run(entries, context)
    var done = 0
    if any_order:
        done = Int(_word(got, niter)) if conv else rounds
    py_setitem(fields, "n_iter_", PythonObject(done if any_order else min(1, rounds)))
    var steps = 0
    for r in range(min(done, len(orders))):  # driver-loop: count of finished imputation steps (imputation rounds)
        steps += len(orders[r])
    var sequence = Python.list()
    for s in range(min(steps, len(seq_j))):  # driver-loop: fitted imputation sequence views (finished imputation steps)
        _ = py_call(sequence.append, Python.tuple(seq_j[s], _take1(got, seq_coef[s], dk, context),
                                                  _take1(got, seq_inter[s], 1, context)))
    py_setitem(fields, "imputation_sequence_", sequence)
    var post = Python.list()
    for s in range(min(steps, len(post_sig))):  # driver-loop: fitted posterior views (finished imputation steps)
        var side = max(post_pp[s], 1)
        _ = py_call(post.append, Python.tuple(post_nbl[s], _take1(got, post_sig[s], side * side, context),
                                              _take1(got, post_al[s], 2, context), _take1(got, post_means[s], dk, context)))
    py_setitem(fields, "_posterior", post)
    py_setitem(fields, "_rng", _u64_obj(rng))
    py_setitem(fields, "_post_step", PythonObject(post_step))
    var xt = _take2(got, fo, n, dk, context)
    return _with_indicator(model, values, xt, dk, context)


# ---------------------------------------------------------------- transform
def _transform(model: PythonObject, x: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    if not py_contains(fields, "imputation_sequence_"):
        raise Error("RuntimeError: mojolearn: this IterativeImputer instance is not fitted yet")
    var mode = fields["numeric_mode_"]
    var entries = _entries(mode, context)
    var values = xprep_values(x, entries, context)
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    _check_width(model, d)
    var keep = _index_list(fields["_keep"])
    var dk = len(keep)
    var bounds = _bounds_of(model)
    var imputer = fields["initial_imputer_"]
    if not py_contains(py_fields(imputer), "n_features_in_"):
        raise Error("RuntimeError: mojolearn: this " + _name(imputer) + " instance is not fitted yet")
    # `initial_imputer_.transform(arr)`: X as converted above
    var xf = _imputer_transform(imputer, values, context)
    var posterior = _truth(fields["sample_posterior"])
    var rng = _u64_of(fields["_rng"])
    var sequence = fields["imputation_sequence_"]
    if fields["estimator"] is not None:
        var mask = _mask_arr(values, keep, fields["missing_values"], context)
        var xt = xf
        for s in range(len(sequence)):  # cpu-route: the fitted user estimators, one predict per step (user code)
            var step = sequence[s]
            var j = Int(py=step[0])
            var took = _ii_take(xt, mask, n, dk, j, _nbl_of(step[1]), mode, False, context)
            xt = _ii_put(xt, took.xm, took.rows, took.m, n, dk, j, step[2], posterior, bounds, mode, rng, context)
        py_setitem(fields, "_rng", _u64_obj(rng))
        return _with_indicator(model, values, xt, dk, context)
    var p = EncProg()
    var prep = _prepare(p, values, xf, keep, bounds, fields["missing_values"], context)
    var fo = prep[0]
    var mo = prep[1]
    var bo = prep[2]
    var seed = Int(rng & 0x7FFFFFFF)
    var post_step = Int(py=py_call(_b().int, fields["_post_step"]))
    var posts = fields["_posterior"] if posterior else PythonObject(None)
    for s in range(len(sequence)):  # driver-loop: drives one device regression per step (fitted imputation steps)
        var step = sequence[s]
        var j = Int(py=step[0])
        var coef = step[1]
        var inter = step[2]
        var co = p.put(coef, dk)
        var io = p.put(inter, 1)
        if posterior:
            var post = posts[s]
            var nb1 = 0
            if post[0] is not None:
                nb1 = p.put_floats(_ones(_nbl_of(post[0]), dk), context) + 1
            var key_words = List[Int]()
            key_words.append(seed)
            key_words.append(post_step)
            var key = p.put_ints(key_words, context)
            post_step += 1
            var sig = post[1]
            var al = post[2]
            var means = post[3]
            var mo2 = p.put(means, Int(py=means.size))
            var so2 = p.put(sig, Int(py=sig.size))
            var ao2 = p.put(al, Int(py=al.size))
            _stage(p.program, OP_II_POST, n, fo, n, dk, mo, j, co, io, bo, NONE, mo2, so2, ao2, nb1, key)
        else:
            _stage(p.program, OP_II_PREDICT, n, fo, n, dk, mo, j, co, io, bo, NONE)
    var ran = p.run(entries, context)
    py_setitem(fields, "_post_step", PythonObject(post_step))
    var xt = _take2(ran, fo, n, dk, context)
    return _with_indicator(model, values, xt, dk, context)


# ---------------------------------------------------------------- the door
def prep_iterimp_call(model: PythonObject, operation: String, args: PythonObject,
                      context: PythonObject) raises -> PythonObject:
    """IterativeImputer.fit(X, y=None) (returns the model), fit_transform(X,
    y=None) and transform(X)."""
    if operation == "fit":
        _ = _fit_transform(model, args[0], context)
        return model
    if operation == "fit_transform":
        return _fit_transform(model, args[0], context)
    if operation == "transform":
        return _transform(model, args[0], context)
    raise Error("TypeError: the native IterativeImputer owner does not serve " + operation)
