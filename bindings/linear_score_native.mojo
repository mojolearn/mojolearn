# SPDX-License-Identifier: Apache-2.0
# One native `score` for the linear expansion regressors
# (python/mojolearn/_expansion_linear.py `_LinearRegressorMixin.score`):
# the prediction link(X @ coef^T + intercept) on the estimator's x_linear
# binding (`x_linear_decision`, the kernel `predict` runs) and R^2 on the
# process-default metrics binding (`r2_score`, the kernel
# `metrics.r2_score` runs), in one call. The same kernels in the same order
# as the Python path it replaces, so the score keeps its bits.
from std.math import isfinite
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields
from bindings.classical_arrays import classical_address, classical_array, classical_host_f32
from bindings.classical_attributes import classical_connect_numpy

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]


def _linear_mode(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    """The estimator's numeric mode, or the process default."""
    var b = Python.import_module("builtins")
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not b.str:
        raise Error("numeric_mode must be fast, identical or None")
    mode = py_call(py_call(mode.strip).lower)
    if String(mode) != "identical" and String(mode) != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    return mode


def _linear_entries(table: PythonObject, mode: PythonObject) raises -> PythonObject:
    var entries = table[mode]
    if py_type(entries) is py_type(PythonObject("")):
        raise Error("ImportError: " + String(entries))
    return entries


def _linear_f32(value: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """A fitted float32 field as one C-ordered float32 array, any rank."""
    var b = Python.import_module("builtins")
    if py_type(value) is b.float or py_type(value) is b.int:
        var one = classical_array(1, 0, context)
        var word = _F32P(unsafe_from_address=Int(py=classical_address(one)))
        word[0] = Float32(Float64(py=value))
        return one
    var rank = Int(py=value.ndim)
    return classical_host_f32(value, rank, name, context)


def _linear_finite(values: PythonObject, count: Int, name: String) raises:
    # The r2 kernel's NaN answer would be a washed sentinel: refused first,
    # as metrics.r2_score refuses it.
    var p = _F32P(unsafe_from_address=Int(py=classical_address(values)))
    for i in range(count):  # driver-loop: over the score's validated host input words
        if not isfinite(p[i]):
            raise Error("mojolearn metrics: " + name + " contains NaN or infinity; refused here "
                        + "(metrics/README.md HAND-OFF ask 3) rather than letting it reach a kernel "
                        + "whose answer for it is a washed sentinel")


def linear_regressor_score(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`score(X, y)`: R^2 of the linear prediction against y."""
    classical_connect_numpy(context)
    var fields = py_fields(model)
    var name = String(py_type(model).__name__)
    # BayesianRidge / ARDRegression refuse what their predict refuses.
    var compute_score = py_get(fields, "compute_score")
    if compute_score is not None and Bool(py=compute_score):
        raise Error("mojolearn " + name + ": compute_score is not implemented")
    if not py_contains(fields, "coef_"):
        raise Error("RuntimeError: mojolearn " + name + ": call fit first")
    var x = classical_host_f32(args[0], 2, "X", context)
    var rows = Int(py=x.shape[0])
    var cols = Int(py=x.shape[1])
    var fitted = Int(py=fields["n_features_in_"])
    if cols != fitted:
        raise Error("mojolearn " + name + ": X has " + String(cols) + " features, the model was fitted with " + String(fitted))
    var coef = _linear_f32(fields["coef_"], "coef_", context)
    var intercept = _linear_f32(fields["intercept_"], "intercept_", context)
    var k = Int(py=intercept.size)
    var width = Int(py=coef.size) // max(k, 1)
    if k != 1 or width != cols:
        raise Error("mojolearn " + name + ": a regressor's score needs one output column")
    # [coef row | intercept], the block _coef_block packed.
    var block = classical_array(cols + 1, 0, context)
    var wb = _F32P(unsafe_from_address=Int(py=classical_address(block)))
    memcpy(dest=wb, src=_F32P(unsafe_from_address=Int(py=classical_address(coef))), count=cols)
    wb[cols] = _F32P(unsafe_from_address=Int(py=classical_address(intercept)))[0]
    var link = 0
    var stored = py_get(fields, "_link")
    if stored is not None:
        link = Int(py=stored)
    var mode = _linear_mode(fields, context)
    var linear = _linear_entries(context["linear_score_bindings"], mode)
    var prediction = classical_array(rows, 1, context)
    _ = py_call(linear["x_linear_decision"], classical_address(x), classical_address(block),
                Python.list(rows, cols, 1, link), classical_address(prediction))
    var target = classical_host_f32(args[1], 1, "y", context)
    var size = Int(py=target.size)
    if size != rows:
        raise Error("mojolearn metrics: y_true has " + String(size) + " entries and y_pred has "
                    + String(rows) + "; they must be the same length")
    _linear_finite(target, size, "y_true")
    _linear_finite(prediction, rows, "y_pred")
    # metrics.r2_score(y, pred) with numeric_mode None: the default binding.
    var metrics = _linear_entries(context["linear_score_metrics"], context["backend_fields"]["_DEFAULT_MODE"])
    var r2 = py_call(metrics["r2_score"], classical_address(target), classical_address(prediction), Python.list(rows))
    var result = Float64(py=r2)
    _ = x^
    _ = coef^
    _ = intercept^
    _ = block^
    _ = prediction^
    _ = target^
    return PythonObject(result)
