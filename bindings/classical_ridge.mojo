# SPDX-License-Identifier: Apache-2.0
# The existing Ridge route and arithmetic stay in their selected native
# kernels. This owner replaces only Python preparation and launch decisions.
from std.python import Python, PythonObject
from std.memory import memcpy
from bindings.classical_arrays import classical_array, classical_host_f32, classical_host_weights, classical_word
from bindings.sequential_arrays import sequential_address
from bindings.sequential_description import sequential_fields


def classical_means(x: PythonObject, rows: Int, cols: Int, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    var sums = classical_array(cols, 0, context, True)
    var m64 = classical_array(cols, 0, context, True)
    var m32 = classical_array(cols, 0, context)
    _ = entries["lm_col_sums"](sequential_address(x), sequential_address(sums), Python.list(rows, cols))
    _ = entries["lm_means_finish"](sequential_address(sums), sequential_address(m64), sequential_address(m32), Python.list(cols, rows, 0), PythonObject(0.0))
    _ = sums^
    return Python.tuple(m64, m32)


def classical_center(x: PythonObject, mean: PythonObject, rows: Int, cols: Int,
                     entries: PythonObject, context: PythonObject) raises -> PythonObject:
    var out = classical_array(rows, cols if Int(py=x.ndim) == 2 else 0, context)
    _ = entries["lm_center"](sequential_address(x), sequential_address(mean), sequential_address(out), Python.list(rows, cols))
    return out


def classical_ridge_fit(model: PythonObject, args: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var fields = sequential_fields(model)
    var alpha = Float64(py=fields["alpha"])
    if alpha < 0:
        raise Error("alpha must be non-negative")
    var solver = String(fields["solver"])
    if solver != "auto" and solver != "eig" and solver != "svd" and solver != "cd":
        raise Error("TypeError: solver is not supported")
    if solver == "svd" or solver == "cd":
        raise Error("NotImplementedError: Ridge implements solver='eig' (and auto); svd and cd are not implemented")
    if Bool(py=fields["normalize"]):
        raise Error("NotImplementedError: Ridge normalize is not implemented")
    var intercept = Bool(py=fields["fit_intercept"])
    var x = classical_host_f32(args[0], 2, "X", context)
    var y = classical_host_f32(args[1], 1, "y", context)
    var rows = Int(py=x.shape[0])
    var cols = Int(py=x.shape[1])
    if Int(py=y.size) != rows:
        raise Error("mojolearn Ridge X and y lengths differ")
    var weights = classical_host_weights(args[2], rows, context)
    var coef = classical_array(cols, 0, context)
    var mean = classical_array(cols, 0, context)
    var ymean = Float64(0.0)
    var bias = Float64(0.0)
    if weights is not None or cols == 1:
        var targets = y
        var has_weights = weights is not None
        if has_weights:
            targets = classical_array(rows * 2, 0, context)
            var dst = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(targets)))
            var source_y = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(y)))
            var source_w = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(weights)))
            memcpy(dest=dst, src=source_y, count=rows * 4)
            memcpy(dest=dst + rows * 4, src=source_w, count=rows * 4)
        var n_out = cols + 5
        var values = classical_array(n_out, 0, context)
        _ = entries["x_linear_fit"](PythonObject(8), sequential_address(x), sequential_address(targets),
            Python.list(rows, cols, rows * cols, rows * (2 if has_weights else 1), n_out,
                        max(3 * cols * cols + 4 * cols + 1 + 2 * rows, 1), 1, 4, 1),
            Python.list(1, 1 if intercept else 0, 1, 1 if has_weights else 0),
            Python.list(PythonObject(alpha)), sequential_address(values))
        if classical_word(values, cols + 4) == 2:
            raise Error("mojolearn Ridge: X'X + alpha I is singular even in float-float; rescale X or use a larger alpha")
        var dst = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(coef)))
        var src = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(values)))
        memcpy(dest=dst, src=src, count=cols)
        bias = classical_word(values, cols) if intercept else Float64(0.0)
        _ = values^
        _ = targets^
    else:
        var done = False
        var output_mean = classical_array(1, 0, context, True)
        if entries.get("linear_gram_fit") is not None and entries.get("linear_gram_solve_default") is not None:
            if Bool(py=entries["linear_gram_solve_default"]()):
                var status = Int(py=entries["linear_gram_fit"](
                    sequential_address(x), sequential_address(y), sequential_address(coef),
                    sequential_address(mean), sequential_address(output_mean),
                    Python.list(rows, cols, PythonObject(alpha), 1 if intercept else 0)))
                done = status == 0
        if not done and entries.get("ridge_fit_resident") is not None and entries.get("ridge_resident_default") is not None:
            if Bool(py=entries["ridge_resident_default"]()):
                _ = entries["ridge_fit_resident"](
                    sequential_address(x), sequential_address(y), sequential_address(coef),
                    sequential_address(mean), sequential_address(output_mean),
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
            if intercept:
                var xm = PythonObject(None)
                var ym = PythonObject(None)
                if entries.get("lm_classical_stats") is not None and entries.get("lm_col_sums_pair") is not None:
                    if Bool(py=entries["lm_classical_stats"]()):
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
            _ = entries["ridge_fit"](sequential_address(work_x), sequential_address(work_y),
                                      sequential_address(coef), Python.list(rows, cols, PythonObject(alpha)))
            _ = work_x^
            _ = work_y^
        if intercept:
            bias = Float64(py=entries["lm_intercept"](sequential_address(mean), sequential_address(coef),
                                                       Python.list(cols), PythonObject(ymean)))
    fields["coef_"] = coef
    fields["intercept_"] = bias
    fields["_x_mean"] = mean
    fields["_y_mean"] = ymean
    fields["n_features_in_"] = cols
    fields["solver_"] = "eig"
    fields["input_copied_"] = not (x is args[0])
    fields["_mojolearn_fitted"] = True
    _ = x^
    _ = y^
    _ = weights^
    return model


def classical_linear_predict(model: PythonObject, value: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = sequential_fields(model)
    if fields.get("coef_") is None:
        raise Error("ValueError: call fit before predict")
    var x = classical_host_f32(value, 2, "X", context)
    var rows = Int(py=x.shape[0])
    var cols = Int(py=x.shape[1])
    if cols != Int(py=fields["n_features_in_"]):
        raise Error("feature count differs from fit")
    var coef = classical_host_f32(fields["coef_"], 1, "coef_", context)
    if Int(py=coef.size) != cols:
        raise Error("coef_ length differs from fitted feature count")
    var out = classical_array(rows, 0, context)
    _ = entries["ols_predict"](sequential_address(x), sequential_address(coef), sequential_address(out),
                                Python.list(rows, cols, fields["intercept_"]))
    _ = x^
    _ = coef^
    return out
