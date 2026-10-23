# SPDX-License-Identifier: Apache-2.0
# LinearSVC `fit`, `predict`, `score` and LinearSVR `fit`, `score`
# (python/mojolearn/svm.py), one native call each. The parameter checks of
# `_LinearSVMBase._check_common`, the label encoding, the per-row weights of
# `linear_model._qn_row_weights` (sample_weight times the class weights the
# x_trees binding spreads over the rows), the one `qn_fit_ovr` (or the
# per-target `qn_fit`) call with the same parameter lists, the prediction
# entries `qn_predict_binary` / `qn_predict_multiclass` /
# `qn_decision_function` and the score metrics are the Python methods' own,
# in the same order with the same arguments: the bits do not move.
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_fields
from bindings.classical_arrays import classical_address
from bindings.classical_attributes import classical_connect_numpy
from bindings.family_common import (
    FAMILY_F32P, FAMILY_F64P, FAMILY_I32P, FAMILY_I64P, family_repr, family_str, family_isinstance,
    family_call, family_entry, family_array, family_sample_weight,
)
from bindings.svm_mixture_native import _qn_scores
from core.family_weights_impl import family_fill_f32
from bindings.svc_common import (
    svc_builtins, svc_field, svc_attr, svc_has, svc_name, svc_float, svc_int, svc_truthy, svc_isfinite,
    svc_base, svc_mode, svc_binding, svc_zero_division, svc_f32, svc_shape_of, svc_labels_1d,
    svc_sorted_classes, svc_decode, svc_threshold_codes, svc_argmax_rows, svc_r2_sums, svc_accuracy,
)

comptime _PY2MOJO_ROWS = 2


# ---------------------------------------------------------------- parameters


def _linear_check_common(model: PythonObject, fields: PythonObject) raises:
    """`_LinearSVMBase._check_common`."""
    var b = svc_builtins()
    var operator = Python.import_module("operator")
    var name = svc_name(model)
    var penalty = svc_field(fields, "penalty")
    var loss = svc_field(fields, "loss")
    var c = svc_field(fields, "C")
    if not py_contains(Python.tuple(PythonObject("l1"), PythonObject("l2")), penalty):
        raise Error("Expected penalty to be one of ['l1', 'l2'], got " + family_repr(penalty))
    var losses = svc_attr(model, "_LOSSES")
    if not py_contains(losses, loss):
        raise Error("Expected loss to be one of " + family_repr(family_call(b.list, losses)) + ", got "
                    + family_repr(loss))
    if not Bool(py=family_call(operator.gt, c, PythonObject(0))):
        raise Error("mojolearn " + name + ": C must be positive, got " + family_str(c))
    if svc_truthy(svc_field(fields, "penalized_intercept")):
        raise Error("NotImplementedError: mojolearn " + name + ": penalized_intercept=True is not implemented "
                    + "(the reference appends a column of ones to X and fits without "
                    + "a bias, linear.cu:150-163); the intercept is never penalized here")
    # `_check_qn_solver_fields`
    var tol = svc_field(fields, "tol")
    if not Bool(py=family_call(operator.gt, tol, PythonObject(0))):
        raise Error("mojolearn " + name + ": tol must be positive, got " + family_str(tol))
    var names: List[String] = ["max_iter", "linesearch_max_iter", "lbfgs_memory"]
    for i in range(3):  # small-loop(solver fields: three integer checks)
        var value = svc_field(fields, names[i])
        if (Bool(py=family_call(operator.ne, family_call(b.int, value), value))
                or Bool(py=family_call(operator.lt, value, PythonObject(1)))):
            raise Error("mojolearn " + name + ": " + names[i] + " must be a positive integer, got " + family_str(value))


def _linear_strengths(fields: PythonObject) raises -> PythonObject:
    """`_strengths`: `1 / C` on the chosen penalty, (l1, l2)."""
    var inv_c = 1.0 / svc_float(svc_field(fields, "C"))
    if String(svc_field(fields, "penalty")) == "l1":
        return Python.tuple(PythonObject(inv_c), PythonObject(0.0))
    return Python.tuple(PythonObject(0.0), PythonObject(inv_c))


def _class_weight_message(model: PythonObject, class_weight: PythonObject) raises -> String:
    return ("The 'class_weight' parameter of " + svc_name(model) + " must be an "
            + "instance of 'dict', a str among {'balanced'} or None. "
            + "Got " + family_repr(class_weight) + " instead.")


def _class_weight_rows(class_weight: PythonObject, classes: PythonObject, codes: PythonObject,
                       xbind: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_class_weights._class_weight_rows` on the x_trees binding with int32
    codes: the per-row float32 class weight, or None when every class weight
    is 1."""
    var b = svc_builtins()
    var itertools = Python.import_module("itertools")
    var n = Int(py=codes.size)
    var k = len(classes)
    var values: PythonObject
    if family_isinstance(class_weight, b.str):
        var counts = family_array(Python.tuple(PythonObject(k)), "<i4", context)
        _ = family_call(family_entry(xbind, "x_trees_code_counts", context),
                        classical_address(codes) if n > 0 else PythonObject(0), classical_address(counts),
                        Python.list(PythonObject(n), PythonObject(k)))
        var wide = family_array(Python.tuple(PythonObject(k)), "<i8", context)
        for i in range(k):  # driver-loop: over the classes (the exact int32 -> int64 widening of astype)
            FAMILY_I64P(unsafe_from_address=Int(py=classical_address(wide)))[i] = Int64(
                FAMILY_I32P(unsafe_from_address=Int(py=classical_address(counts)))[i])
        var w64 = family_array(Python.tuple(PythonObject(k)), "<f8", context)
        if k > 0 and Int(py=family_call(svc_base("class_ratio_f64", context), classical_address(wide), PythonObject(k),
                                         PythonObject(n), PythonObject(0), classical_address(w64))) != 0:
            svc_zero_division(False)
        _ = w64  # keep the Python object alive through the native calls above
        _ = wide^
        _ = counts^
        values = Python.list()
        for i in range(k):  # driver-loop: over the classes
            _ = py_call(values.append, PythonObject(FAMILY_F64P(unsafe_from_address=Int(py=classical_address(w64)))[i]))
            _ = w64  # keep the Python object alive through the native call above
    else:
        var unknown = family_call(family_call(b.set, class_weight).difference, classes)
        if len(unknown) > 0:
            raise Error("class_weight contains labels absent from y")
        values = family_call(b.list, family_call(b.map, class_weight.get, classes, family_call(itertools.repeat, PythonObject(1.0))))
    var all_one = True
    for i in range(k):  # driver-loop: over the classes (the class weight argument checks)
        var v = values[i]
        var boolean = family_isinstance(v, b.bool) or String(py_type(v).__name__) == "bool_"
        if boolean or not family_isinstance(v, context["family_svc_real"]):
            raise Error("class weights must be finite nonnegative real numbers")
        var f = svc_float(v)
        if not svc_isfinite(f) or f < 0:
            raise Error("class weights must be finite nonnegative real numbers")
        if not Bool(py=family_call(Python.import_module("operator").eq, v, PythonObject(1))):
            all_one = False
    if all_one:
        return PythonObject(None)
    var narrowed = family_array(Python.tuple(PythonObject(k)), "<f4", context)
    var np = FAMILY_F32P(unsafe_from_address=Int(py=classical_address(narrowed)))
    var top = Float32(0)
    for i in range(k):  # driver-loop: over the classes (the float32 narrowing and its checks)
        var f = svc_float(values[i])
        np[i] = Float32(f)
        var g = Float64(np[i])
        if not svc_isfinite(g) or (f > 0 and g == 0):
            raise Error("class weights must remain finite and nonzero when positive in Float32")
        if i == 0 or np[i] > top:
            top = np[i]
    if not top > 0:
        raise Error("class weights must have positive total")
    var out = family_array(Python.tuple(PythonObject(n)), "<f4", context)
    if n > 0:
        _ = family_call(family_entry(xbind, "x_trees_class_rows", context), classical_address(codes),
                        classical_address(narrowed), classical_address(out), Python.list(PythonObject(n), PythonObject(k)))
    _ = narrowed^
    return out


def _qn_row_weights(model: PythonObject, fields: PythonObject, sample_weight: PythonObject,
                    class_weight: PythonObject, classes: PythonObject, codes: PythonObject, n: Int,
                    context: PythonObject) raises -> PythonObject:
    """`linear_model._qn_row_weights`."""
    var b = svc_builtins()
    if sample_weight is None and class_weight is None:
        return PythonObject(None)
    var sw: PythonObject
    if sample_weight is None:
        sw = family_array(Python.tuple(PythonObject(n)), "<f4", context)
        family_fill_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(sw))), n, Float32(1.0))
    else:
        sw = family_sample_weight(sample_weight, n, False, context)
    if class_weight is None:
        return sw
    if family_isinstance(class_weight, b.str):
        if String(class_weight) != "balanced":
            raise Error(_class_weight_message(model, class_weight))
    elif not family_isinstance(class_weight, b.dict):
        raise Error(_class_weight_message(model, class_weight))
    var xb = svc_binding("_mojolearn_x_trees", svc_mode(fields), context)
    var cw = _class_weight_rows(class_weight, family_call(b.list, classes), codes, xb, context)
    if cw is None:
        return sw
    var out = family_array(Python.tuple(PythonObject(n)), "<f4", context)
    _ = family_call(family_entry(xb, "x_trees_mul_f32", context), classical_address(sw), classical_address(cw),
                    classical_address(out), Python.list(PythonObject(n)))
    _ = sw^
    _ = cw^
    return out


def _qn_fit_one_target(fields: PythonObject, binding: PythonObject, x: PythonObject, y_enc: PythonObject,
                       n_classes: Int, loss: Int, l1: PythonObject, l2: PythonObject, grad_tol: Float64,
                       change_tol: Float64, svr_eps: Float64, sw: PythonObject, penalty_normalized: Bool,
                       context: PythonObject) raises -> PythonObject:
    """`linear_model._qn_fit_one_target`: (w, n_iter, objective, retcode)."""
    var rows = Int(py=x.shape[0])
    var cols = Int(py=x.shape[1])
    var fi = 1 if svc_truthy(svc_field(fields, "fit_intercept")) else 0
    var w = family_array(Python.tuple(PythonObject(cols + fi)), "<f4", context)
    var info = family_array(Python.tuple(PythonObject(2)), "<f4", context)
    var has_sw = 0 if sw is None else 1
    var params = Python.list(PythonObject(rows), PythonObject(cols), PythonObject(n_classes),
                             family_call(svc_builtins().float, l1), family_call(svc_builtins().float, l2),
                             PythonObject(grad_tol), PythonObject(change_tol),
                             PythonObject(svc_int(svc_field(fields, "max_iter"))),
                             PythonObject(svc_int(svc_field(fields, "linesearch_max_iter"))),
                             PythonObject(svc_int(svc_field(fields, "lbfgs_memory"))),
                             PythonObject(fi), PythonObject(1 if penalty_normalized else 0), PythonObject(has_sw),
                             PythonObject(loss), PythonObject(svr_eps))
    if has_sw == 1:
        _ = py_call(params.append, classical_address(sw))
    var n_iter = family_call(family_entry(binding, "qn_fit", context), classical_address(x), classical_address(y_enc),
                             classical_address(w), classical_address(info), params)
    _ = info  # keep the Python object alive through the native call above
    var fx = Float64(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(info)))[0])
    _ = info  # keep the Python object alive through the native call above
    var rc = Int(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(info)))[1])
    _ = info  # keep the Python object alive through the native call above
    var _keep_ret_info_r209 = Python.tuple(w, PythonObject(svc_int(n_iter)), PythonObject(fx), PythonObject(rc))
    _ = info  # keep the Python object alive through the native call above
    return _keep_ret_info_r209^


# ---------------------------------------------------------------- LinearSVR


def _linear_svr_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`LinearSVR.fit(X, y, sample_weight)`; args = (binding, X, y, sample_weight)."""
    var operator = Python.import_module("operator")
    var binding = args[0]
    var fields = py_fields(model)
    var epsilon = svc_field(fields, "epsilon")
    if not (Bool(py=family_call(operator.le, PythonObject(0.0), epsilon))
            and Bool(py=family_call(operator.lt, epsilon, family_call(svc_builtins().float, PythonObject("inf"))))):
        raise Error("mojolearn LinearSVR: epsilon must be finite and non-negative, got " + family_str(epsilon))
    _linear_check_common(model, fields)
    var xc = svc_f32(args[1], 2, "X", False, context)
    py_setitem(fields, PythonObject("input_copied_"), xc[1])
    var x = xc[0]
    var rows = Int(py=x.shape[0])
    var cols = Int(py=x.shape[1])
    # `_target_1d`
    if len(svc_shape_of(args[2])) != 1:
        raise Error("mojolearn LinearSVR requires a 1-D y")
    var t = svc_f32(args[2], 1, "y", False, context)[0]
    if Int(py=t.shape[0]) != rows:
        raise Error("mojolearn LinearSVR X and y lengths differ")
    var strengths = _linear_strengths(fields)
    var sw = PythonObject(None)
    if args[3] is not None:
        sw = family_sample_weight(args[3], rows, False, context)
    var tol = svc_float(svc_field(fields, "tol"))
    var loss = Int(py=svc_attr(model, "_LOSSES")[svc_field(fields, "loss")])
    var result = _qn_fit_one_target(fields, binding, x, t, 1, loss, strengths[0], strengths[1], tol, 0.1 * tol,
                                    svc_float(epsilon), sw, svc_truthy(svc_attr(model, "penalty_normalized")), context)
    var w = result[0]
    py_setitem(fields, PythonObject("objective_"), result[2])
    py_setitem(fields, PythonObject("retcode_"), result[3])
    py_setitem(fields, PythonObject("_w"), w)
    var coef = family_array(Python.tuple(PythonObject(cols)), "<f4", context)
    memcpy(dest=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(coef))),
           src=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(w))), count=cols)
    py_setitem(fields, PythonObject("coef_"), coef)
    if svc_truthy(svc_field(fields, "fit_intercept")):
        py_setitem(fields, PythonObject("intercept_"),
                   PythonObject(Float64(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(w)))[cols])))
    else:
        py_setitem(fields, PythonObject("intercept_"), PythonObject(0.0))
    var n_iter = family_array(Python.tuple(PythonObject(1)), "<i8", context)
    FAMILY_I64P(unsafe_from_address=Int(py=classical_address(n_iter)))[0] = Int64(Int(py=result[1]))
    py_setitem(fields, PythonObject("n_iter_"), n_iter)
    py_setitem(fields, PythonObject("n_features_in_"), PythonObject(cols))
    _ = x^
    _ = t^
    _ = sw^


def _linear_svr_predict(model: PythonObject, binding: PythonObject, source: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    if not svc_has(py_fields(model), "_w"):
        raise Error("mojolearn LinearSVR: call fit before predict")
    return _qn_scores(model, binding, source, 1, context)


def _r2_host(pred: PythonObject, y: PythonObject, context: PythonObject) raises -> PythonObject:
    """`linear_model._r2_host`: `1 - SS_res / SS_tot`, 0.0 for a constant y."""
    var sums = svc_r2_sums(pred, y, context)
    var ss_res = Float64(py=sums[0])
    var ss_tot = Float64(py=sums[1])
    if ss_tot != 0.0:
        return PythonObject(1.0 - ss_res / ss_tot)
    return PythonObject(0.0)


# ---------------------------------------------------------------- LinearSVC


def _linear_svc_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`LinearSVC.fit(X, y, sample_weight)`; args = (binding, X, y, sample_weight)."""
    var b = svc_builtins()
    var operator = Python.import_module("operator")
    var binding = args[0]
    var fields = py_fields(model)
    var multi_class = svc_field(fields, "multi_class")
    if Bool(py=family_call(operator.ne, multi_class, PythonObject("ovr"))):
        raise Error("mojolearn LinearSVC: multi_class must be 'ovr', got " + family_repr(multi_class))
    _linear_check_common(model, fields)
    var xc = svc_f32(args[1], 2, "X", False, context)
    py_setitem(fields, PythonObject("input_copied_"), xc[1])
    var x = xc[0]
    var rows = Int(py=x.shape[0])
    var cols = Int(py=x.shape[1])
    var labels = svc_labels_1d(args[2], context)[0]
    if labels is None:
        raise Error("mojolearn LinearSVC requires a 1-D y")
    if len(labels) != rows:
        raise Error("mojolearn LinearSVC X and y lengths differ")
    var encoded = svc_sorted_classes(labels, context)
    var classes = encoded[0]
    var codes = encoded[1]
    py_setitem(fields, PythonObject("classes_"), classes)
    var n_classes = len(classes)
    if n_classes < 2:
        raise Error("mojolearn LinearSVC: y has one class")
    var strengths = _linear_strengths(fields)
    var sw = _qn_row_weights(model, fields, args[3], svc_field(fields, "class_weight"), classes, codes, rows, context)
    # `OvrSelector` (linear.cu:50-54): two classes select class 1 and fit once.
    var first = 1 if n_classes == 2 else 0
    var n_fits = 1 if n_classes == 2 else n_classes
    var fi = 1 if svc_truthy(svc_field(fields, "fit_intercept")) else 0
    var n_coefs = cols + fi
    var pn = 1 if svc_truthy(svc_attr(model, "penalty_normalized")) else 0
    var tol = svc_float(svc_field(fields, "tol"))
    var loss = Int(py=svc_attr(model, "_LOSSES")[svc_field(fields, "loss")])
    var w_all = family_array(Python.tuple(PythonObject(n_fits), PythonObject(n_coefs)), "<f4", context)
    var n_iter = 0
    var objectives = Python.list()
    var retcodes = Python.list()
    var ovr = py_call(b.getattr, binding, PythonObject("qn_fit_ovr"), PythonObject(None))
    if ovr is not None:
        # X and the class codes cross the bus once; the binding builds each
        # 0/1 target on the device and runs the same per-class fits.
        var info = family_array(Python.tuple(PythonObject(n_fits), PythonObject(3)), "<f4", context)
        var params = Python.list(PythonObject(rows), PythonObject(cols), PythonObject(first), PythonObject(n_fits),
                                 family_call(b.float, strengths[0]), family_call(b.float, strengths[1]),
                                 PythonObject(tol), PythonObject(0.1 * tol),
                                 PythonObject(svc_int(svc_field(fields, "max_iter"))),
                                 PythonObject(svc_int(svc_field(fields, "linesearch_max_iter"))),
                                 PythonObject(svc_int(svc_field(fields, "lbfgs_memory"))),
                                 PythonObject(fi), PythonObject(pn), PythonObject(loss))
        if sw is not None:
            _ = py_call(params.append, classical_address(sw))  # field 15
        _ = family_call(family_entry(binding, "qn_fit_ovr", context), classical_address(x), classical_address(codes),
                        classical_address(w_all), classical_address(info), params)
        var ip = FAMILY_F32P(unsafe_from_address=Int(py=classical_address(info)))
        for k in range(n_fits):  # driver-loop: over the one-vs-rest fits (one result row each)
            n_iter = max(n_iter, Int(ip[3 * k + 2]))
            _ = py_call(objectives.append, PythonObject(Float64(ip[3 * k])))
            _ = py_call(retcodes.append, PythonObject(Int(ip[3 * k + 1])))
        _ = info^
    else:
        # a binding without the entry (the CPU host binding): one binary fit
        # per selected class on its 0/1 target
        var cp = FAMILY_I32P(unsafe_from_address=Int(py=classical_address(codes)))
        for k in range(n_fits):  # driver-loop: over the one-vs-rest fits
            var cls = first + k
            var y_enc = family_array(Python.tuple(PythonObject(rows)), "<f4", context)
            var yp = FAMILY_F32P(unsafe_from_address=Int(py=classical_address(y_enc)))
            for i in range(rows):  # cpu-route: the host binding's 0/1 target (equal_elements, cast_elements)
                yp[i] = Float32(1.0) if Int(cp[i]) == cls else Float32(0.0)
            var result = _qn_fit_one_target(fields, binding, x, y_enc, 2, loss, strengths[0], strengths[1], tol,
                                            0.1 * tol, 0.0, sw, pn == 1, context)
            memcpy(dest=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(w_all)) + 4 * k * n_coefs),
                   src=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(result[0]))), count=n_coefs)
            n_iter = max(n_iter, Int(py=result[1]))
            _ = py_call(objectives.append, result[2])
            _ = py_call(retcodes.append, result[3])
            _ = y_enc^
    # The (C, n_coefs) block's column-major copy is the `w[c + C*j]` block
    # `qn_decision_function` reads.
    var wsrc = FAMILY_F32P(unsafe_from_address=Int(py=classical_address(w_all)))
    var w = family_array(Python.tuple(PythonObject(n_fits * n_coefs)), "<f4", context)
    var wdst = FAMILY_F32P(unsafe_from_address=Int(py=classical_address(w)))
    var coef = family_array(Python.tuple(PythonObject(n_fits), PythonObject(cols)), "<f4", context)
    var cdst = FAMILY_F32P(unsafe_from_address=Int(py=classical_address(coef)))
    var intercept = family_array(Python.tuple(PythonObject(n_fits)), "<f4", context)
    var idst = FAMILY_F32P(unsafe_from_address=Int(py=classical_address(intercept)))
    for c in range(n_fits):  # driver-loop: over the one-vs-rest targets (exact copies of the fitted block)
        for j in range(n_coefs):  # driver-loop: over the coefficients of one target
            wdst[c + n_fits * j] = wsrc[c * n_coefs + j]
            if j < cols:
                cdst[c * cols + j] = wsrc[c * n_coefs + j]
        if fi == 1:
            idst[c] = wsrc[c * n_coefs + cols]
    py_setitem(fields, PythonObject("_w"), w)
    py_setitem(fields, PythonObject("coef_"), coef)
    py_setitem(fields, PythonObject("intercept_"), intercept)
    var it = family_array(Python.tuple(PythonObject(1)), "<i8", context)
    FAMILY_I64P(unsafe_from_address=Int(py=classical_address(it)))[0] = Int64(n_iter)
    py_setitem(fields, PythonObject("n_iter_"), it)
    py_setitem(fields, PythonObject("objective_"), objectives[0] if n_fits == 1 else objectives)
    py_setitem(fields, PythonObject("retcode_"), retcodes[0] if n_fits == 1 else retcodes)
    py_setitem(fields, PythonObject("n_features_in_"), PythonObject(cols))
    _ = x^
    _ = sw^
    _ = w_all^
    _ = codes^
    _ = encoded^


def _py2mojo_flags(binding: PythonObject) raises -> Int:
    var fn_obj = py_call(svc_builtins().getattr, binding, PythonObject("py2mojo_linear_flags"), PythonObject(None))
    if fn_obj is None:
        return 0
    return svc_int(family_call(fn_obj))


def _linear_svc_codes(model: PythonObject, fields: PythonObject, binding: PythonObject, source: PythonObject,
                      entry: String, n_classes: Int, context: PythonObject) raises -> PythonObject:
    """`qn_predict_binary` / `qn_predict_multiclass`: int64 class codes."""
    if not svc_has(fields, "_w"):
        raise Error("mojolearn LinearSVC: call fit first")
    var x = svc_f32(source, 2, "X", False, context)[0]
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    if d != svc_int(svc_field(fields, "n_features_in_")):
        raise Error("mojolearn " + svc_name(model) + " feature count differs from fit")
    var codes = family_array(Python.tuple(PythonObject(n)), "<i8", context)
    var fi = 1 if svc_truthy(svc_field(fields, "fit_intercept")) else 0
    var params = Python.list(PythonObject(n), PythonObject(d), PythonObject(fi))
    if n_classes > 2:
        _ = py_call(params.append, PythonObject(n_classes))
    var w = svc_field(fields, "_w")
    if n > 0:
        _ = family_call(family_entry(binding, entry, context), classical_address(x), classical_address(w),
                        classical_address(codes), params)
    _ = x^
    _ = w^
    return codes


def _linear_svc_predict(model: PythonObject, binding: PythonObject, source: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    """`LinearSVC.predict(X)`."""
    var b = svc_builtins()
    var fields = py_fields(model)
    var fitted = svc_has(fields, "classes_")
    if fitted:
        var classes = svc_field(fields, "classes_")
        var k = len(classes)
        if k == 2 and (_py2mojo_flags(binding) & _PY2MOJO_ROWS) != 0:
            # the `z > 0` codes in the binding: the same decision and threshold
            return svc_decode(classes, _linear_svc_codes(model, fields, binding, source, "qn_predict_binary", 2, context), context)
        if k > 2 and py_call(b.getattr, binding, PythonObject("qn_predict_multiclass"), PythonObject(None)) is not None:
            # the row argmax on the device beside the scores
            return svc_decode(classes, _linear_svc_codes(model, fields, binding, source, "qn_predict_multiclass", k, context), context)
    if not svc_has(fields, "_w"):
        raise Error("mojolearn LinearSVC: call fit first")
    var classes = svc_field(fields, "classes_")
    var k = len(classes)
    var scores = _qn_scores(model, binding, source, 1 if k == 2 else k, context)
    if k == 2:
        return svc_decode(classes, svc_threshold_codes(scores, 0.0, True, 0, 1, context), context)
    return svc_decode(classes, svc_argmax_rows(scores, context), context)


def _pred_count(pred: PythonObject) raises -> Int:
    """`len(pred)` of an Array (its first axis) or a list."""
    var b = svc_builtins()
    if py_type(pred) is b.list:
        return len(pred)
    return Int(py=pred.shape[0])


def svc_linear_operation(model: PythonObject, operation: String, args: PythonObject,
                         context: PythonObject) raises -> PythonObject:
    """The linear machines; args[0] is the resolved estimators binding."""
    classical_connect_numpy(context)
    if operation == "family_svc_linear_svr_fit":
        _linear_svr_fit(model, args, context)
        return model
    if operation == "family_svc_linear_svr_score":
        var pred = _linear_svr_predict(model, args[0], args[1], context)
        return _r2_host(pred, args[2], context)
    if operation == "family_svc_linear_svc_fit":
        _linear_svc_fit(model, args, context)
        return model
    if operation == "family_svc_linear_svc_predict":
        return _linear_svc_predict(model, args[0], args[1], context)
    if operation == "family_svc_linear_svc_score":
        var pred = _linear_svc_predict(model, args[0], args[1], context)
        return PythonObject(svc_accuracy(args[2], pred, _pred_count(pred), context))
    raise Error("RuntimeError: unknown linear SVM operation " + operation)
