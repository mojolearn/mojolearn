# SPDX-License-Identifier: Apache-2.0
# The existing Ridge route and arithmetic stay in their selected native
# kernels. This owner replaces only Python preparation and launch decisions.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.py_typed_call import py_call_typed
from std.memory import memcpy
from core.classical_ridge_impl import classical_widen_f32_f64
from bindings.classical_arrays import classical_array, classical_word, classical_address
from bindings.admitted_input import admit_x
from bindings.sequential_description import sequential_fields
from bindings.classical_metadata import classical_fit_replacements_admit
from bindings.classical_validate import classical_check_x, classical_check_y, classical_check_weights_input, classical_check_width, classical_estimator_name


def classical_means(x: PythonObject, rows: Int, cols: Int, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    var sums = classical_array(cols, 0, context, True)
    var m64 = classical_array(cols, 0, context, True)
    var m32 = classical_array(cols, 0, context)
    _ = py_call_typed(entries["lm_col_sums"], classical_address(x), classical_address(sums), Python.list(rows, cols))
    _ = py_call_typed(entries["lm_means_finish"], classical_address(sums), classical_address(m64), classical_address(m32), Python.list(cols, rows, 0), PythonObject(0.0))
    _ = sums^
    return Python.tuple(m64, m32)


def classical_center(x: PythonObject, mean: PythonObject, rows: Int, cols: Int,
                     entries: PythonObject, context: PythonObject) raises -> PythonObject:
    var out = classical_array(rows, cols if Int(py=x.ndim) == 2 else 0, context)
    _ = py_call_typed(entries["lm_center"], classical_address(x), classical_address(mean), classical_address(out), Python.list(rows, cols))
    return out


def classical_ols_prepare(x: PythonObject, y: PythonObject, weights: PythonObject, rows: Int, cols: Int,
                          intercept: Bool, mode: Int, prep: PythonObject, context: PythonObject) raises -> PythonObject:
    """`lm_ols_prep`: (work_x, work_y, xmean32, ymean64) of the general
    route in one native call. mode 0 separate means, 1 paired, 2 weighted."""
    var work_x = classical_array(rows, cols, context)
    var work_y = classical_array(rows, 0, context)
    var xm64 = classical_array(cols, 0, context, True)
    var xm32 = classical_array(cols, 0, context)
    var ym64 = classical_array(1, 0, context, True)
    var ym32 = classical_array(1, 0, context)
    var w_address = PythonObject(0) if weights is None else classical_address(weights)
    _ = py_call_typed(prep, Python.list(classical_address(x), classical_address(y), w_address,
                                        classical_address(work_x), classical_address(work_y),
                                        classical_address(xm64), classical_address(xm32),
                                        classical_address(ym64), classical_address(ym32)),
                      Python.list(rows, cols, 1 if intercept else 0, mode))
    _ = x  # keep the Python object alive through the native call above
    _ = y  # keep the Python object alive through the native call above
    _ = weights  # keep the Python object alive through the native call above
    _ = xm64^
    _ = ym32^
    return Python.tuple(work_x, work_y, xm32, ym64)


def classical_ridge_parameters(fields: PythonObject) raises:
    # Checked when fit runs, after the inputs, as the constructor stores them.
    var b = native_loaded_module("builtins")
    var alpha = fields["alpha"]
    var negative = False
    try:
        negative = Bool(py=alpha < PythonObject(0.0))
    except error:
        raise Error("TypeError: " + String(error))
    if negative:
        raise Error("alpha must be non-negative, got " + String(py_call(b.str, alpha)))
    var solver = fields["solver"]
    var name = String(solver) if py_type(solver) is b.str else String("")
    # repr(solver) is built only inside the refusals that show it.
    if name != "auto" and name != "eig" and name != "svd" and name != "cd":
        raise Error("TypeError: solver " + String(py_call(b.repr, solver)) + " is not supported")
    if name == "svd" or name == "cd":
        var shown = String(py_call(b.repr, solver))
        raise Error("NotImplementedError: mojolearn Ridge: solver=" + shown + " is not implemented "
                    + "(ridgeSVD is raft::linalg::svdQR -> cuSOLVER gesvd; 'cd' is cuml/solvers/cd.pyx); "
                    + "solver='eig' (cuML's 'auto') is the implemented arm. See glm/NOT_IMPLEMENTED.tsv")
    if Bool(py=fields["normalize"]):
        raise Error("NotImplementedError: mojolearn Ridge: normalize is not implemented "
                    + "(preprocess.cuh:76-108, the meanvar arm of preProcessData; glm/NOT_IMPLEMENTED.tsv)")


def classical_ridge_fit(model: PythonObject, args: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var fields = sequential_fields(model)
    classical_fit_replacements_admit(fields, True, context)
    var est = classical_estimator_name(model)
    var x = classical_check_x(args[0], est, context, admitted=True)
    var rows = Int(py=x.shape[0])
    var cols = Int(py=x.shape[1])
    var y = classical_check_y(args[1], rows, est, context, admitted=True)
    classical_ridge_parameters(fields)
    var alpha = Float64(py=fields["alpha"])
    var intercept = Bool(py=fields["fit_intercept"])
    var weights = classical_check_weights_input(args[2], rows, context, admitted=True)
    var coef = classical_array(cols, 0, context)
    var mean = classical_array(cols, 0, context)
    var ymean = Float64(0.0)
    var bias = Float64(0.0)
    if weights is not None or cols == 1:
        var targets = y
        var has_weights = weights is not None
        if has_weights:
            targets = classical_array(rows * 2, 0, context)
            var dst = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(targets)))
            var source_y = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(y)))
            var source_w = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(weights)))
            memcpy(dest=dst, src=source_y, count=rows * 4)
            memcpy(dest=dst + rows * 4, src=source_w, count=rows * 4)
        var n_out = cols + 5
        var values = classical_array(n_out, 0, context)
        _ = py_call_typed(entries["x_linear_fit"], PythonObject(8), classical_address(x), classical_address(targets),
            Python.list(rows, cols, rows * cols, rows * (2 if has_weights else 1), n_out,
                        max(3 * cols * cols + 4 * cols + 1 + 2 * rows, 1), 1, 4, 1),
            Python.list(1, 1 if intercept else 0, 1, 1 if has_weights else 0),
            Python.list(PythonObject(alpha)), classical_address(values))
        if classical_word(values, cols + 4) == 2:
            raise Error("mojolearn Ridge: X'X + alpha I is singular even in float-float; rescale X or use a larger alpha")
        var dst = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(coef)))
        var src = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(values)))
        memcpy(dest=dst, src=src, count=cols)
        bias = classical_word(values, cols) if intercept else Float64(0.0)
        _ = values^
        _ = targets^
    else:
        var done = False
        var output_mean = classical_array(1, 0, context, True)
        if py_get(entries, "linear_gram_fit") is not None and py_get(entries, "linear_gram_solve_default") is not None:
            if Bool(py=entries["linear_gram_solve_default"]()):
                var status = Int(py=py_call_typed(entries["linear_gram_fit"], 
                    classical_address(x), classical_address(y), classical_address(coef),
                    classical_address(mean), classical_address(output_mean),
                    Python.list(rows, cols, PythonObject(alpha), 1 if intercept else 0)))
                done = status == 0
        if not done and py_get(entries, "ridge_fit_resident") is not None and py_get(entries, "ridge_resident_default") is not None:
            if Bool(py=entries["ridge_resident_default"]()):
                _ = py_call_typed(entries["ridge_fit_resident"], 
                    classical_address(x), classical_address(y), classical_address(coef),
                    classical_address(mean), classical_address(output_mean),
                    Python.list(rows, cols, PythonObject(alpha), 1 if intercept else 0))
                done = True
        if done:
            if intercept:
                ymean = classical_word(output_mean)
            else:
                mean = classical_array(cols, 0, context)
        else:
            var work_x = x
            var work_y = y
            var prep = py_get(entries, "lm_ols_prep")
            if prep is not None and intercept:
                # ne2-g-linear (N403): the means and centering below in one
                # native call (the same typed steps, the same words).
                var mode = 0
                if py_get(entries, "lm_classical_stats") is not None and py_get(entries, "lm_col_sums_pair") is not None:
                    if Bool(py=entries["lm_classical_stats"]()):
                        mode = 1
                var prepared = classical_ols_prepare(x, y, PythonObject(None), rows, cols, True, mode, prep, context)
                work_x = prepared[0]
                work_y = prepared[1]
                mean = prepared[2]
                ymean = classical_word(prepared[3])
            elif intercept:
                var xm = PythonObject(None)
                var ym = PythonObject(None)
                if py_get(entries, "lm_classical_stats") is not None and py_get(entries, "lm_col_sums_pair") is not None:
                    if Bool(py=entries["lm_classical_stats"]()):
                        var sx = classical_array(cols, 0, context, True)
                        var sy = classical_array(1, 0, context, True)
                        var xm64 = classical_array(cols, 0, context, True)
                        var xm32 = classical_array(cols, 0, context)
                        var ym64 = classical_array(1, 0, context, True)
                        var ym32 = classical_array(1, 0, context)
                        _ = py_call_typed(entries["lm_col_sums_pair"], classical_address(x), classical_address(y),
                            classical_address(sx), classical_address(sy), Python.list(rows, cols))
                        _ = py_call_typed(entries["lm_means_finish"], classical_address(sx), classical_address(xm64), classical_address(xm32),
                                                       Python.list(cols, rows, 0), PythonObject(0.0))
                        _ = py_call_typed(entries["lm_means_finish"], classical_address(sy), classical_address(ym64), classical_address(ym32),
                                                       Python.list(1, rows, 0), PythonObject(0.0))
                        xm = Python.tuple(xm64, xm32)
                        ym = Python.tuple(ym64, ym32)
                        _ = sx^
                        _ = sy^
                if xm is None:
                    xm = classical_means(x, rows, cols, entries, context)
                    ym = classical_means(y, rows, 1, entries, context)
                mean = xm[1]
                ymean = classical_word(ym[0])
                work_x = classical_center(x, mean, rows, cols, entries, context)
                work_y = classical_center(y, ym[1], rows, 1, entries, context)
            _ = py_call_typed(entries["ridge_fit"], classical_address(work_x), classical_address(work_y),
                                      classical_address(coef), Python.list(rows, cols, PythonObject(alpha)))
            _ = work_x^
            _ = work_y^
        if intercept:
            bias = Float64(py=py_call_typed(entries["lm_intercept"], classical_address(mean), classical_address(coef),
                                                       Python.list(cols), PythonObject(ymean)))
    py_setitem(fields, "coef_", coef)
    py_setitem(fields, "intercept_", bias)
    py_setitem(fields, "_x_mean", mean)
    py_setitem(fields, "_y_mean", ymean)
    py_setitem(fields, "n_features_in_", cols)
    py_setitem(fields, "solver_", "eig")
    py_setitem(fields, "input_copied_", not (x is args[0]))
    py_setitem(fields, "_mojolearn_fitted", True)
    _ = x^
    _ = y^
    _ = weights^
    return model


def classical_linear_predict(model: PythonObject, value: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    return _classical_linear_predict(model, value, entries, context, False)


def _classical_linear_predict(model: PythonObject, value: PythonObject, entries: PythonObject, context: PythonObject,
                              wide: Bool) raises -> PythonObject:
    # wide: write the float64 public prediction in the same native call
    # (`ols_predict_wide`), one output buffer and no separate widening pass.
    var fields = sequential_fields(model)
    if py_get(fields, "coef_") is None:
        raise Error("ValueError: call fit before predict")
    var est = classical_estimator_name(model)
    var x = classical_check_x(value, est, context, admitted=True)
    var rows = Int(py=x.shape[0])
    var cols = Int(py=x.shape[1])
    classical_check_width(cols, Int(py=fields["n_features_in_"]), est)
    var coef = admit_x(fields["coef_"], 1, "coef_", context)
    if Int(py=coef.size) != cols:
        raise Error("coef_ length differs from fitted feature count")
    var out = classical_array(rows, 0, context, wide)
    _ = py_call_typed(entries["ols_predict_wide" if wide else "ols_predict"], classical_address(x),
                      classical_address(coef), classical_address(out), Python.list(rows, cols, fields["intercept_"]))
    _ = x^
    _ = coef^
    return out


def classical_linear_predict_public(model: PythonObject, value: PythonObject,
                                   entries: PythonObject, context: PythonObject) raises -> PythonObject:
    # PublicOutputsMixin historically widens the completed float32 prediction
    # to float64. Preserve this output conversion without changing the fitted
    # kernel or the float32 prediction consumed internally by R2 scoring.
    # A binding with `ols_predict_wide` writes the exact float64 widening of
    # the same float32 words in its own call (GIL released); an older binding
    # keeps the separate widening pass below.
    if py_get(entries, "ols_predict_wide") is not None:
        return _classical_linear_predict(model, value, entries, context, True)
    var narrow = classical_linear_predict(model, value, entries, context)
    var size = Int(py=narrow.size)
    var wide = classical_array(size, 0, context, True)
    classical_widen_f32_f64(Int(py=classical_address(narrow)), Int(py=classical_address(wide)), size)
    _ = narrow^
    return wide
