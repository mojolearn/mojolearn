# SPDX-License-Identifier: Apache-2.0
# Existing OLS preparation, route admission, TSQR and consumed coefficient
# publication, owned by Mojo. All fitting arithmetic stays in selected C ABI
# entries; this module never calls the Python _M/_Kit driver.
from std.python import Python, PythonObject
from bindings.classical_arrays import classical_array, classical_host_f32, classical_host_weights, classical_word
from bindings.classical_ridge import classical_means, classical_center
from bindings.classical_tsqr import classical_tsqr_solve
from bindings.sequential_arrays import sequential_address
from bindings.sequential_description import sequential_fields


def classical_scale_rows(x: PythonObject, weight: PythonObject, rows: Int, cols: Int,
                         entries: PythonObject, context: PythonObject) raises -> PythonObject:
    var out = classical_array(rows, cols if Int(py=x.ndim) == 2 else 0, context)
    _ = entries["lm_scale_rows"](sequential_address(x), sequential_address(weight), sequential_address(out), Python.list(rows, cols))
    return out


def classical_weighted_means(x: PythonObject, weight: PythonObject, rows: Int, cols: Int,
                             entries: PythonObject, context: PythonObject) raises -> PythonObject:
    var weight_sum = classical_array(1, 0, context, True)
    _ = entries["lm_col_sums"](sequential_address(weight), sequential_address(weight_sum), Python.list(rows, 1))
    var total = classical_word(weight_sum)
    if total <= 0:
        raise Error("sample_weight sums to zero")
    var scaled = classical_scale_rows(x, weight, rows, cols, entries, context)
    var sums = classical_array(cols, 0, context, True)
    var m64 = classical_array(cols, 0, context, True)
    var m32 = classical_array(cols, 0, context)
    _ = entries["lm_col_sums"](sequential_address(scaled), sequential_address(sums), Python.list(rows, cols))
    _ = entries["lm_means_finish"](sequential_address(sums), sequential_address(m64), sequential_address(m32),
                                   Python.list(cols, rows, 1), PythonObject(total))
    _ = scaled^
    _ = weight_sum^
    _ = sums^
    return Python.tuple(m64, m32)


def classical_paired_means(x: PythonObject, y: PythonObject, rows: Int, cols: Int,
                           entries: PythonObject, context: PythonObject) raises -> PythonObject:
    var sx = classical_array(cols, 0, context, True)
    var sy = classical_array(1, 0, context, True)
    var xm64 = classical_array(cols, 0, context, True)
    var xm32 = classical_array(cols, 0, context)
    var ym64 = classical_array(1, 0, context, True)
    var ym32 = classical_array(1, 0, context)
    _ = entries["lm_col_sums_pair"](sequential_address(x), sequential_address(y),
                                   sequential_address(sx), sequential_address(sy), Python.list(rows, cols))
    _ = entries["lm_means_finish"](sequential_address(sx), sequential_address(xm64), sequential_address(xm32),
                                   Python.list(cols, rows, 0), PythonObject(0.0))
    _ = entries["lm_means_finish"](sequential_address(sy), sequential_address(ym64), sequential_address(ym32),
                                   Python.list(1, rows, 0), PythonObject(0.0))
    _ = sx^
    _ = sy^
    return Python.tuple(Python.tuple(xm64, xm32), Python.tuple(ym64, ym32))


def classical_ols_fit(model: PythonObject, args: PythonObject, entries: PythonObject,
                      context: PythonObject) raises -> PythonObject:
    var fields = sequential_fields(model)
    var intercept = Bool(py=fields["fit_intercept"])
    var x = classical_host_f32(args[0], 2, "X", context)
    var y = classical_host_f32(args[1], 1, "y", context)
    var rows = Int(py=x.shape[0])
    var cols = Int(py=x.shape[1])
    if Int(py=y.size) != rows:
        raise Error("mojolearn LinearRegression X and y lengths differ")
    if rows < 2:
        raise Error("Found array with " + String(rows) + " sample(s) while a minimum of 2 is required.")
    var weights = classical_host_weights(args[2], rows, context)
    var fast_normal = False
    if entries.get("ols_normal_eq_default") is not None:
        fast_normal = Bool(py=entries["ols_normal_eq_default"]())
    # TSQR's actual factorization width limit, not a benchmark dimension rule.
    var tsqr = cols >= 1 and cols + 1 <= 512 and rows >= cols + 1 and not fast_normal
    var coef = classical_array(cols, 0, context)
    var mean = classical_array(cols, 0, context)
    var ymean = Float64(0.0)
    var done = False
    if fast_normal and weights is None and entries.get("ols_fit_resident") is not None:
        var ym = classical_array(1, 0, context, True)
        _ = entries["ols_fit_resident"](sequential_address(x), sequential_address(y), sequential_address(coef),
            sequential_address(mean), sequential_address(ym), Python.list(rows, cols, Int(intercept)))
        if intercept:
            ymean = classical_word(ym)
        else:
            mean = classical_array(cols, 0, context)
        done = True
        _ = ym^
    var decomp = PythonObject(None)
    if tsqr:
        decomp = entries.get("decomp")
        if Python.import_module("builtins").type(decomp) is not Python.import_module("builtins").dict:
            raise Error("ImportError: native LinearRegression TSQR entries unavailable: " + String(decomp))
        if String(decomp["x_decomp_vendor"]()) != String(context["vendor"]):
            raise Error("RuntimeError: TSQR vendor differs from selected execution")
        if Int(py=decomp["x_decomp_numeric_mode"]()) != Int(py=entries["estimators_numeric_mode"]()):
            raise Error("RuntimeError: TSQR numeric mode differs from selected execution")
    if not done and tsqr and intercept and weights is None and decomp.get("x_decomp_idn_flags") is not None:
        if Int(py=decomp["x_decomp_idn_flags"]()) & 1:
            var ra = classical_array(cols + 1, cols + 1, context)
            var ym = classical_array(1, 0, context, True)
            _ = decomp["x_decomp_ols_tsqr_r"](sequential_address(x), sequential_address(y), sequential_address(ra),
                sequential_address(mean), sequential_address(ym), Python.list(rows, cols))
            coef = classical_tsqr_solve(ra, rows, cols, decomp, context)
            ymean = classical_word(ym)
            done = True
            _ = ra^
            _ = ym^
    if not done:
        var work_x = x
        var work_y = y
        if intercept:
            var xm = PythonObject(None)
            var ym = PythonObject(None)
            if weights is not None:
                xm = classical_weighted_means(x, weights, rows, cols, entries, context)
                ym = classical_weighted_means(y, weights, rows, 1, entries, context)
            elif entries.get("lm_classical_stats") is not None and entries.get("lm_col_sums_pair") is not None:
                if Bool(py=entries["lm_classical_stats"]()):
                    var pair = classical_paired_means(x, y, rows, cols, entries, context)
                    xm = pair[0]
                    ym = pair[1]
            if xm is None:
                xm = classical_means(x, rows, cols, entries, context)
                ym = classical_means(y, rows, 1, entries, context)
            mean = xm[1]
            ymean = classical_word(ym[0])
            work_x = classical_center(x, mean, rows, cols, entries, context)
            work_y = classical_center(y, ym[1], rows, 1, entries, context)
        if weights is not None:
            var root = classical_array(rows, 0, context)
            _ = entries["py2mojo_rows"](PythonObject(5), sequential_address(weights), sequential_address(root), Python.list(rows, 1))
            work_x = classical_scale_rows(work_x, root, rows, cols, entries, context)
            work_y = classical_scale_rows(work_y, root, rows, 1, entries, context)
            _ = root^
        if tsqr:
            var ra = classical_array(cols + 1, cols + 1, context)
            _ = decomp["x_decomp_tsqr_r"](sequential_address(work_x), sequential_address(work_y), sequential_address(ra),
                                          Python.list(rows, cols, 1, 0))
            coef = classical_tsqr_solve(ra, rows, cols, decomp, context)
            _ = ra^
        else:
            _ = entries["ols_fit"](sequential_address(work_x), sequential_address(work_y), sequential_address(coef), Python.list(rows, cols))
        _ = work_x^
        _ = work_y^
    var bias = Float64(0.0)
    if intercept:
        bias = Float64(py=entries["lm_intercept"](sequential_address(mean), sequential_address(coef), Python.list(cols), PythonObject(ymean)))
    fields["coef_"] = coef
    fields["intercept_"] = bias
    fields["_x_mean"] = mean
    fields["_y_mean"] = ymean
    fields["n_features_in_"] = cols
    fields["input_copied_"] = not (x is args[0])
    fields["_mojolearn_fitted"] = True
    _ = x^
    _ = y^
    _ = weights^
    return model
