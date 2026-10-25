# SPDX-License-Identifier: Apache-2.0
# The native owner of RFE and of mutual_info_classif / mutual_info_regression
# (dispatched by bindings/prep_owner_native.mojo). Each public operation is
# one call: the parameter refusals, the input conversion, the x_prep
# programs the Python built (the same stages, operands and arena order, so
# the same bits) and the fitted attributes or the returned scores.
#
# RFE runs its elimination rounds here (one driver loop over the rounds); its
# estimator must be a mojolearn estimator with a connected native plan. Its
# clone, fit, predict, predict_proba, decision_function and score run as the
# plan's compiled operations (bindings/prep_native_estimator.mojo), so a round
# never enters a Python method: neither the estimator's fit / predict nor its
# get_params or constructor.
# Its programs run on the shared arena runner (bindings/prep_classify_prog.mojo
# `ClsProg`, the classifier owner's context `select_context`) because the
# support mask and the ranking are in-out inputs, as `_Prog.put(inout=True)`
# made them. The mutual-information programs and the selector transform run
# on the encode owner's transport (bindings/prep_encode_native.mojo `EncProg`).
from std.memory import memcpy
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.native_loaded_module import native_loaded_module
from bindings.optimizer_public_words import stash_python_error
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.classical_arrays import classical_address, classical_typed, classical_input, classical_host_f32
from bindings.classical_labels import labels_f32, labels_vector, classical_encode_labels
from bindings.scaler_native import _stage, _ptr
from bindings.scaler_lifecycle_native import _entries
from bindings.x_prep_native_owner import xprep_values
from bindings.prep_stateless_native import _select_transform
from bindings.prep_encode_native import (
    EncProg, _cs, _codes, _fit_categories, _take1, _take2, _word, _iword, _put_codes, _default_mode,
    _seed64, _score_words)
from bindings.prep_classify_prog import ClsProg, f32_array
from bindings.prep_classify_common import cls_x2d, cls_mode
from bindings.prep_native_estimator import (
    prep_estimator_admit, prep_estimator_has, prep_estimator_clone, prep_estimator_call, prep_estimator_attribute)

comptime F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime I64P = MutPointer[Int64, MutUntrackedOrigin]
comptime INT32_MAX = 2147483647

# x_prep stage codes (_expansion_prep._OPS).
comptime OP_GATHER_COLS = 61
comptime OP_MI_COLSCALE = 66
comptime OP_MI_NOISE = 67
comptime OP_MI_CC = 68
comptime OP_MI_CD = 69
comptime OP_MI_REDUCE = 70
comptime OP_SQSUM_COLS = 71
comptime OP_CODE_COUNTS = 88
comptime OP_I2F = 10
comptime OP_MI_DC = 94
comptime OP_MI_DD = 95
comptime OP_P2M_CCOUNT = 200
comptime OP_P2M_CSCAN = 201
comptime OP_C2_KEY64 = 253
comptime OP_C2_RFE_STEP = 256
comptime OP_C2_RFE_RANK = 257
# `_II_CH`: the p2m units' chunk rows.
comptime II_CH = 1024


# ---------------------------------------------------------------- calls into estimators
def _call(callable: PythonObject, args: PythonObject, kwargs: PythonObject) raises -> PythonObject:
    """`callable(*args, **kwargs)` (args a tuple, kwargs a dict or None); an
    exception it raises is restored at the door unchanged."""
    ref cpy = Python().cpython()
    var keywords = PyObjectPtr()
    if kwargs is not None:
        keywords = kwargs._obj_ptr
    var result = cpy.PyObject_Call(callable._obj_ptr, args._obj_ptr, keywords)
    if not result:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=result)


def _attr(owner: PythonObject, var name: String) raises -> PythonObject:
    """`owner.name`; its AttributeError (or any other) reaches the caller unchanged."""
    ref cpy = Python().cpython()
    var result = cpy.PyObject_GetAttrString(owner._obj_ptr, name^)
    if not result:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=result)


def _class_name(value: PythonObject) raises -> String:
    return String(py_type(value).__name__)


def _fit_clone(estimator: PythonObject, x: PythonObject, y: PythonObject, fit_params: PythonObject,
               context: PythonObject) raises -> PythonObject:
    """RFE's `_clone(estimator).fit(x, y)`: the clone built from the
    constructor schema and fitted by the plan's compiled fit (no Python
    get_params, constructor or fit). Returns the fitted clone."""
    var est = prep_estimator_clone(estimator, context)
    _ = prep_estimator_call(est, "fit", Python.tuple(x, y), context, fit_params)
    return est


# ---------------------------------------------------------------- RFE programs
def _gather_cls(arr_value: PythonObject, n: Int, d: Int, cols: List[Int], mode: PythonObject,
                cc: PythonObject) raises -> PythonObject:
    """`_gather(arr, cols, mode)`: the columns `cols` of X as an (n, k) float32 Array."""
    var k = len(cols)
    var pr = ClsProg()
    var xo = pr.put(arr_value)
    var fl = List[Float64]()
    for i in range(k):  # driver-loop: over the selected column ids (feature metadata)
        fl.append(Float64(cols[i]))
    var ko = pr.put_list(fl, cc)
    var out = pr.alloc(n * k)
    pr.stage(OP_GATHER_COLS, n * k, xo, n, d, ko, k, out)
    pr.run(mode, cc)
    return pr.get(out, n, k, cc)


def _importance_keys(mut pr: ClsProg, est: PythonObject, m: Int, getter: PythonObject, context: PythonObject,
                     cc: PythonObject) raises -> Int:
    """`_stage_importance_keys(pr, est, m, getter)`: the ordered keys
    (c2_key64) of the fitted estimator's m column importances: coef_ squared
    (summed over rows when 2-D) on the device, else feature_importances_ as
    given; a dotted attribute path picks the importances instead (squared).
    Returns the keys' offset."""
    var b = native_loaded_module("builtins")
    var auto = py_type(getter) is b.str and String(getter) == "auto"
    var coef = PythonObject(None)
    if not auto:
        coef = prep_estimator_attribute(est, getter, context)
    else:
        coef = prep_estimator_attribute(est, PythonObject("coef_"), context)
    var key = pr.work(2 * m)
    if coef is None and auto:
        var imp = prep_estimator_attribute(est, PythonObject("feature_importances_"), context)
        if imp is None:
            raise Error("mojolearn: RFE needs an estimator with coef_ or feature_importances_")
        var got = _score_words(imp, m, context)
        var wo = pr.put_words(got[0])
        pr.stage(OP_C2_KEY64, m, wo, got[1], key)
        return key
    # `as_f32_c(coef, ndim=None)` then `rows, d = (1, len) if 1-D else shape`
    var raw = classical_input(coef, cc, "coef_")
    var nd = Int(py=raw.ndim)
    var rows = 1
    var d = 0
    if nd == 1:
        d = Int(py=raw.shape[0])
    elif nd == 2:
        rows = Int(py=raw.shape[0])
        d = Int(py=raw.shape[1])
    elif nd == 0:
        raise Error("not enough values to unpack (expected 2, got 0)")
    else:
        raise Error("too many values to unpack (expected 2)")
    if d != m:
        raise Error("mojolearn: RFE importances have " + String(d) + " columns, expected " + String(m))
    var c = classical_host_f32(coef, nd, "coef_", cc)
    var out = pr.work(d)
    var co = pr.put(c)
    pr.stage(OP_SQSUM_COLS, d, co, rows, d, out)
    pr.stage(OP_C2_KEY64, d, out, 0, key)
    return key


def _fit_params(args: PythonObject) raises -> PythonObject:
    var params = args[2] if len(args) > 2 else PythonObject(None)
    if params is None:
        return Python.dict()
    return params


def _rfe_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """RFE.fit(X, y, **fit_params): the elimination rounds, then the final
    estimator on the selected columns."""
    var b = native_loaded_module("builtins")
    var cc = context["select_context"]
    var fields = py_fields(model)
    var estimator = fields["estimator"]
    prep_estimator_admit(estimator, "RFE", context)
    var g = fields["importance_getter"]
    if not Bool(py=py_call(b.isinstance, g, b.str)):
        if Bool(py=py_call(b.callable, g)):
            raise Error("TypeError: mojolearn: RFE importance_getter must be 'auto' or an attribute path; a callable is not supported")
        raise Error("mojolearn: RFE importance_getter must be 'auto', a str or a callable")
    var y = args[1]
    var fit_params = _fit_params(args)
    if py_type(fit_params) is not b.dict:
        raise Error("TypeError: mojolearn: RFE.fit parameters require an exact dict")
    var arr = cls_x2d(args[0], cc)
    var n = arr.n
    var d = arr.d
    var mode = cls_mode(cc)
    # n_features_to_select: None -> d // 2; a float in (0, 1) -> int(nsel * d); then max(1, int(nsel))
    var nsel_arg = fields["n_features_to_select"]
    var nsel = 0
    if nsel_arg is None:
        nsel = d // 2
    elif Bool(py=py_call(b.isinstance, nsel_arg, b.float)) and Float64(py=nsel_arg) > 0.0 and Float64(py=nsel_arg) < 1.0:
        nsel = Int(Float64(py=nsel_arg) * Float64(d))
    else:
        nsel = Int(py=py_call_typed(b.int, nsel_arg))
    nsel = max(1, nsel)
    # step: a float below 1 -> int(max(1, step * d)); else int(step)
    var step_arg = fields["step"]
    var step = 0
    if Bool(py=py_call(b.isinstance, step_arg, b.float)) and Float64(py=step_arg) < 1.0:
        var v = Float64(py=step_arg) * Float64(d)
        step = 1 if v <= 1.0 else Int(v)
    else:
        step = Int(py=py_call_typed(b.int, step_arg))
    if step <= 0:
        raise Error("mojolearn: step must be > 0")
    var support = List[Bool](length=d, fill=True)
    var ranking = f32_array(d, cc)
    var rp = _ptr(ranking)
    for j in range(d):  # driver-loop: over the initial ranking (feature count)
        rp[j] = 1.0
        _ = ranking  # keep the Python object alive through the native call above
    var nsup = d
    while nsup > nsel:  # driver-loop: over elimination rounds
        var features = List[Int]()
        for j in range(d):  # driver-loop: over the support mask (feature count)
            if support[j]:
                features.append(j)
        var m = len(features)
        var est = _fit_clone(estimator, _gather_cls(arr.value, n, d, features, mode, cc), y, fit_params, context)
        var pr = ClsProg()
        var key = _importance_keys(pr, est, m, g, context, cc)
        var mask = List[Float64]()
        for j in range(d):  # driver-loop: over the support mask (feature count)
            mask.append(1.0 if support[j] else 0.0)
        var sup = pr.put_list(mask, cc, True)
        var rk = pr.put(ranking, True)
        var fo = pr.put_ints(features, cc)
        pr.stage(OP_C2_RFE_STEP, m, key, fo, m, min(step, nsup - nsel), sup)
        pr.stage(OP_C2_RFE_RANK, d, sup, rk)
        pr.run(mode, cc)
        nsup = 0
        for j in range(d):  # driver-loop: over the device's support mask (feature count)
            support[j] = pr.value(sup + j) != 0.0
            if support[j]:
                nsup += 1
        ranking = pr.get(rk, d, 0, cc)
    var features = List[Int]()
    var mask_list = Python.list()
    for j in range(d):  # driver-loop: over the fitted support (feature count)
        if support[j]:
            features.append(j)
        _ = py_call(mask_list.append, PythonObject(support[j]))
    var final = _fit_clone(estimator, _gather_cls(arr.value, n, d, features, mode, cc), y, fit_params, context)
    py_setitem(fields, "estimator_", final)
    py_setitem(fields, "_mask", mask_list)
    py_setitem(fields, "support_", py_call(b.list, mask_list))
    # ranking.astype("<i8")
    var ranks = classical_typed(Python.tuple(d), "<i8", cc["output_context"])
    var dst = I64P(unsafe_from_address=Int(py=classical_address(ranks)))
    var src = _ptr(ranking)
    for j in range(d):  # driver-loop: over the fitted ranking (feature count)
        dst[j] = Int64(Int(src[j]))
        _ = ranks  # keep the Python object alive through the native call above
        _ = ranking  # keep the Python object alive through the native call above
    py_setitem(fields, "ranking_", ranks)
    py_setitem(fields, "n_features_", len(features))
    py_setitem(fields, "numeric_mode_", mode)
    py_setitem(fields, "n_features_in_", d)


def _rfe_transform(model: PythonObject, x: PythonObject, context: PythonObject) raises -> PythonObject:
    # `_SelectorMixin.transform`, as the stateless owner runs it.
    var fields = py_fields(model)
    if not py_contains(fields, "n_features_in_"):
        raise Error("RuntimeError: mojolearn: this " + _class_name(model) + " instance is not fitted yet")
    var values = xprep_values(x, _entries(fields["numeric_mode_"], context), context)
    var out = _select_transform(model, values, context)
    _ = values^
    return out


def prep_rfe_call(model: PythonObject, operation: String, args: PythonObject,
                  context: PythonObject) raises -> PythonObject:
    if operation == "fit":
        _rfe_fit(model, args, context)
        return model
    if operation == "fit_transform":
        _rfe_fit(model, args, context)
        return _rfe_transform(model, args[0], context)
    if operation == "predict" or operation == "predict_proba" or operation == "decision_function" or operation == "score":
        # `self.estimator_.<op>(self.transform(X)[, y])`: the method is looked
        # up first, in the fitted estimator's native plan
        var fitted = py_fields(model)
        if not py_contains(fitted, "estimator_"):
            raise Error("AttributeError: '" + _class_name(model) + "' object has no attribute 'estimator_'")
        var est = fitted["estimator_"]
        prep_estimator_admit(est, "RFE", context)
        if not prep_estimator_has(est, operation, context):
            raise Error("AttributeError: '" + _class_name(est) + "' object has no attribute '" + operation + "'")
        var xs = _rfe_transform(model, args[0], context)
        if operation == "score":
            return prep_estimator_call(est, operation, Python.tuple(xs, args[1]), context)
        return prep_estimator_call(est, operation, Python.tuple(xs), context)
    raise Error("TypeError: the native RFE owner does not serve " + operation)


# ---------------------------------------------------------------- mutual information
def _discrete_mask(df: PythonObject, d: Int, context: PythonObject) raises -> List[Bool]:
    """`_mi_discrete_mask(discrete_features, d)`: 'auto' (none), a bool for
    every column, a bool mask of length d, or column indices."""
    var b = native_loaded_module("builtins")
    if Bool(py=py_call(b.isinstance, df, b.str)):
        if String(df) != "auto":
            raise Error("mojolearn: Invalid string value for discrete_features.")
        return List[Bool](length=d, fill=False)
    if Bool(py=py_call(b.isinstance, df, b.bool)) or String(py_type(df).__name__) == "bool_":
        return List[Bool](length=d, fill=Bool(py=py_call(b.bool, df)))
    var vals = PythonObject(None)
    if Bool(py=py_call(b.hasattr, df, "tolist")):
        vals = _call(_attr(df, "tolist"), Python.tuple(), PythonObject(None))
    else:
        vals = py_call_typed(b.list, df)
    if Bool(py=py_call(b.isinstance, vals, b.bool)):
        return List[Bool](length=d, fill=Bool(py=vals))
    if py_type(vals) is not b.list:
        vals = py_call_typed(b.list, vals)
    var count = len(vals)
    var all_bool = count > 0
    for i in range(count):  # driver-loop: over the discrete_features argument
        if not Bool(py=py_call(b.isinstance, vals[i], b.bool)):
            all_bool = False
            break
    if all_bool:
        if count != d:
            raise Error("mojolearn: discrete_features mask has " + String(count) + " entries; X has " + String(d) + " features")
        var mask = List[Bool]()
        for i in range(count):  # driver-loop: over the discrete_features mask
            mask.append(Bool(py=vals[i]))
        return mask^
    var mask = List[Bool](length=d, fill=False)
    var integral = context["integral_type"] if py_contains(context, "integral_type") else Python.import_module("numbers").Integral
    for i in range(count):  # driver-loop: over the discrete_features indices
        var v = vals[i]
        if Bool(py=py_call(b.isinstance, v, b.bool)) or not Bool(py=py_call(b.isinstance, v, integral)):
            raise Error("mojolearn: discrete_features must be 'auto', a bool, a bool mask or indices")
        var j = Int(py=py_call_typed(b.int, v))
        if not (-d <= j and j < d):
            raise Error("IndexError: mojolearn: discrete_features index " + String(j) + " is out of bounds for " + String(d) + " features")
        mask[j + d if j < 0 else j] = True
    return mask^


def _gather_enc(values: PythonObject, cols: List[Int], entries: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_gather(arr, cols, mode)` on the encode transport; returns the
    gathered block as an `xprep_values` record (0, Array, None, n, k)."""
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var k = len(cols)
    var p = EncProg()
    var xo = p.put_x(values)
    var fl = List[Float64]()
    for i in range(k):  # driver-loop: over the selected column ids (feature metadata)
        fl.append(Float64(cols[i]))
    var ko = p.put_floats(fl, context)
    var out = p.alloc(n * k)
    _stage(p.program, OP_GATHER_COLS, n * k, xo, n, d, ko, k, out)
    var ran = p.run(entries, context)
    return Python.tuple(0, _take2(ran, out, n, k, context), PythonObject(None), n, k)


def _class_counts(codes: PythonObject, n: Int, K: Int, entries: PythonObject, context: PythonObject) raises -> List[Int]:
    """`_class_counts(codes, K)`: one program of p2m_ccount / p2m_cscan."""
    if n == 0 or K <= 0:
        return List[Int](length=max(K, 0), fill=0)
    var p = EncProg()
    var co = _put_codes(p, codes)
    # `_p2m_chunks(n, K)`
    var ch = max(II_CH, (n + II_CH - 1) // II_CH, (n * K + 16777215) // 16777216, 1)
    var nch = max(1, (n + ch - 1) // ch)
    var cnt = p.alloc(nch * K)
    var tot = p.alloc(K)
    _stage(p.program, OP_P2M_CCOUNT, nch, co, n, K, ch, cnt)
    _stage(p.program, OP_P2M_CSCAN, K, cnt, nch, K, tot)
    var ran = p.run(entries, context)
    var out = List[Int]()
    for i in range(K):  # driver-loop: over the class counts (class count)
        out.append(_iword(ran, tot + i))
    return out^


def _work(mut p: EncProg, words: Int, scratch: Bool) -> Int:
    # `work`: pr.work under MOJOLEARN_MI_WORK on the FAST tier, else pr.alloc.
    return p.work(words) if scratch else p.alloc(words)


def prep_mutual_info_call(operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_mutual_info(X, y, discrete_target, discrete_features, n_neighbors,
    random_state)`: continuous columns scaled and noised (the 1e-10-scaled
    noise kept as a second word), then per column Kraskov, Ross or the
    contingency table, in one program."""
    var b = native_loaded_module("builtins")
    var discrete_target = operation == "mutual_info_classif"
    var x = args[0]
    var y = args[1]
    var k = Int(py=py_call_typed(b.int, args[3]))
    if not (1 <= k and k <= 32):
        raise Error("NotImplementedError: mojolearn: mutual_info supports 1 <= n_neighbors <= 32")
    var mode = _default_mode(context)
    var entries = _entries(mode, context)
    var values = xprep_values(x, entries, context)
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    if n == 0 or d == 0:
        raise Error("mojolearn: X must be a nonempty two-dimensional array")
    if n * d > INT32_MAX:
        raise Error("mojolearn: X exceeds the native Int32 indexing bound")
    var mask = _discrete_mask(args[2], d, context)
    var cont = List[Int]()
    var disc = List[Int]()
    for j in range(d):  # driver-loop: over the column kinds (feature count)
        if mask[j]:
            disc.append(j)
        else:
            cont.append(j)
    var seed = _seed64(args[4])[0] & 0x3FFFFFFF
    var scratch = Bool(py=context["select_mi_work"]) and String(mode) == "fast"
    var p = EncProg()
    var yo = -1
    var lc = -1
    var K = 0
    var counts = List[Int]()
    var target = PythonObject(None)
    if discrete_target:
        var labels = classical_encode_labels(y, context)
        if labels.n != n:
            raise Error("mojolearn: X and y have different numbers of rows")
        K = len(labels.classes)
        counts = _class_counts(labels.codes, n, K, entries, context)
        var most = 0
        for i in range(len(counts)):  # driver-loop: over the class counts (class count)
            most = max(most, counts[i])
        if len(cont) > 0 and most < 2:
            raise Error("mojolearn: mutual_info: every class has one sample (the reference's "
                        + "neighbour search over the classes with more than one finds 0 samples)")
        yo = _put_codes(p, labels.codes)
        var fl = List[Float64]()
        for i in range(len(counts)):  # driver-loop: over the class counts (class count)
            fl.append(Float64(counts[i]))
        lc = p.put_floats(fl, context)
    else:
        var t = labels_f32(y, 1, "C", "y", context)
        if t.rows != n:
            raise Error("mojolearn: X and y have different numbers of rows")
        target = labels_vector(n, "<f4", context)
        memcpy(dest=_ptr(target), src=F32P(unsafe_from_address=t.addr), count=n)
        _ = target  # keep the Python object alive through the native call above
        _ = t^
    var dc = len(cont)
    var z = -1
    var zs = -1
    if dc > 0:
        var xo = 0
        if len(disc) > 0:
            xo = p.put_x(_gather_enc(values, cont, entries, context))
        else:
            xo = p.put_x(values)
        var st = p.alloc(6 * dc)
        var sc = p.alloc(dc)
        var ma = p.alloc(dc)
        z = _work(p, n * dc, scratch)
        zs = _work(p, n * dc, scratch)
        _cs(p, mode, entries, xo, n, dc, st)
        _stage(p.program, OP_MI_COLSCALE, dc, xo, n, dc, st, sc, ma)
        _stage(p.program, OP_MI_NOISE, n * dc, xo, n, dc, sc, ma, 2 * seed, z, zs + 1)
    var zy = -1
    var zys = -1
    if not discrete_target:
        yo = p.put(target, n)
        var sty = p.alloc(6)
        var scy = p.alloc(1)
        var may = p.alloc(1)
        zy = _work(p, n, scratch)
        zys = _work(p, n, scratch)
        _cs(p, mode, entries, yo, n, 1, sty)
        _stage(p.program, OP_MI_COLSCALE, 1, yo, n, 1, sty, scy, may)
        _stage(p.program, OP_MI_NOISE, n, yo, n, 1, scy, may, 2 * seed + 1, zy, zys + 1)
    var outc = -1
    if dc > 0:
        var term = _work(p, n * dc, scratch)
        outc = p.alloc(dc)
        if discrete_target:
            _stage(p.program, OP_MI_CD, n * dc, z, n, dc, yo, lc, k, term, zs + 1)
            _stage(p.program, OP_MI_REDUCE, dc, term, n, dc, 3, k, 0, outc, lc, len(counts))
        else:
            _stage(p.program, OP_MI_CC, n * dc, z, n, dc, zy, k, term, zs + 1, zys + 1)
            _stage(p.program, OP_MI_REDUCE, dc, term, n, dc, 0, k, n, outc)
    var dd = len(disc)
    var outd = -1
    if dd > 0:
        var xd = _gather_enc(values, disc, entries, context)
        var cats = _fit_categories(xd, mode, entries, context)
        var kx = List[Int]()
        var kmax = 0
        for c in range(dd):  # driver-loop: over the per-column category arrays (discrete feature count)
            kx.append(Int(py=cats[c].size))
            kmax = max(kmax, kx[c])
        if not discrete_target:
            for c in range(dd):  # driver-loop: over the per-column category counts (discrete feature count)
                if kx[c] == n:
                    raise Error("mojolearn: mutual_info: discrete feature " + String(disc[c]) + " has one sample per "
                                + "value (the reference's neighbour search finds 0 samples)")
        var coded = _codes(p, xd, cats, context)
        var xc = coded[0]
        outd = p.alloc(dd)
        if discrete_target:
            var ky = K
            var stride = kmax * ky + kmax + ky
            var tb = p.alloc(dd * stride)
            var fl = List[Float64]()
            for c in range(dd):  # driver-loop: over the per-column category counts (discrete feature count)
                fl.append(Float64(kx[c]))
            var kxo = p.put_floats(fl, context)
            _stage(p.program, OP_MI_DD, dd, xc, n, dd, yo, ky, kxo, tb, stride, outd)
        else:
            var cnti = p.alloc(dd * kmax)
            var cntf = p.alloc(dd * kmax)
            var term = _work(p, n * dd, scratch)
            _stage(p.program, OP_CODE_COUNTS, dd, xc, n, dd, kmax, cnti)
            _stage(p.program, OP_I2F, dd * kmax, cnti, cntf)
            _stage(p.program, OP_MI_DC, n * dd, zy, n, dd, xc, cntf, kmax, k, term, zys + 1)
            _stage(p.program, OP_MI_REDUCE, dd, term, n, dd, 2, k, 0, outd, cntf, kmax)
    var ran = p.run(entries, context)
    _ = target^
    if dd == 0:
        var res = _take1(ran, outc, d, context)
        _ = values^
        return res
    # the device's values scattered to their columns, as a float32 Array
    var out = classical_typed(Python.tuple(d), "<f4", context["output_context"])
    var dst = _ptr(out)
    for i in range(dc):  # driver-loop: scatters device MI values to their columns (continuous feature count)
        dst[cont[i]] = _word(ran, outc + i)
        _ = out  # keep the Python object alive through the native call above
    for i in range(dd):  # driver-loop: scatters device MI values to their columns (discrete feature count)
        dst[disc[i]] = _word(ran, outd + i)
        _ = out  # keep the Python object alive through the native call above
    _ = values^
    return out
