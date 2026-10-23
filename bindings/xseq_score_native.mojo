# SPDX-License-Identifier: Apache-2.0
"""`score` of the sequence-expansion regressors (LSTM/GRU/RNN regressors and
MLPRegressor) in the one native call: the native prediction, then R^2 of it
against y on the metrics binding of the model's numeric mode, the kernel
`RegressorScoreMixin.score` reached through `metrics.r2_score`:

    truth = as_f32_c(y); pred = as_f32_c(predict(X))
    1-D (or (n, 1)) targets: `r2_score` / `r2_score_weighted` with the same
    refusals in the same words; other shapes: the native metrics owner's
    `family_metric_r2_options` (the x_metrics program `r2_score_options`
    ran), when it is connected.

The same kernels in the same order, so the score keeps its bits.
"""
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.xseq_native_common import xseq_addr, xseq_f32p, xseq_reshape
from bindings.xseq_neural_native import xseq_as_f32_c
from core.xseq_marshal import xseq_all_finite_f32, xseq_min_max_f32


def _shape_text(value: PythonObject) raises -> String:
    return String(py=py_call(Python.import_module("builtins").str, value.shape))


def _vector(value: PythonObject, name: String) raises -> PythonObject:
    """`_as_f32_1d` of an already float32 array: `_vector_shape`'s refusals,
    the column flattened, then the finiteness refusal."""
    var rank = Int(py=value.ndim)
    var column = rank == 2 and Int(py=value.shape[1]) == 1
    if rank != 1 and not column:
        raise Error("ValueError: mojolearn metrics: " + name + " must be 1-D, got shape " + _shape_text(value))
    var n = Int(py=value.shape[0])
    if n == 0:
        raise Error("ValueError: mojolearn metrics: " + name + " is empty")
    var out = xseq_reshape(value, Python.tuple(n)) if column else value
    if not xseq_all_finite_f32(xseq_f32p(out), n):
        raise Error("ValueError: mojolearn metrics: " + name + " contains NaN or infinity; refused "
                    + "here (metrics/README.md HAND-OFF ask 3) rather than letting it "
                    + "reach a kernel whose answer for it is a washed sentinel")
    _ = out  # keep the Python object alive through the native calls above
    return out


def _metrics_entries(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_metrics_impl._get_binding(model.numeric_mode)`: the metrics binding
    of that mode (or the process default), checked against its compiled mode."""
    var b = Python.import_module("builtins")
    var mode = py_get(py_fields(model), "numeric_mode")
    if mode is not None and (py_type(mode) is not b.str
                             or (String(py=py_call(py_call(mode.strip).lower)) != "fast"
                                 and String(py=py_call(py_call(mode.strip).lower)) != "identical")):
        raise Error("ValueError: numeric_mode must be 'fast' or 'identical'")
    var expected = context["backend_fields"]["_DEFAULT_MODE"] if mode is None else mode
    expected = py_call(py_call(expected.strip).lower)
    var table = context["xseq_metrics"]
    var entries = table[expected]
    if py_type(entries) is b.str:
        raise Error("ImportError: " + String(py=entries))
    if py_contains(entries, "metrics_numeric_mode"):
        var actual = Int(py=py_call_typed(entries["metrics_numeric_mode"]))
        var name = String("fast") if actual == 0 else String("identical")
        if name != String(py=expected):
            raise Error("RuntimeError: mojolearn metrics: requested " + String(py=expected) + ", binary reports "
                        + name + "; rebuild it")
    return entries


def _weights(source: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """`_sample_weight_f32(sample_weight, n, "r2_score")`."""
    var w = xseq_as_f32_c(source, 0, "sample_weight", context)
    if Int(py=w.ndim) != 1:
        raise Error("ValueError: mojolearn r2_score: sample_weight must be 1-D, got shape " + _shape_text(w))
    if Int(py=w.shape[0]) != n:
        raise Error("ValueError: mojolearn r2_score: sample_weight has " + String(Int(py=w.shape[0]))
                    + " entries for " + String(n) + " samples")
    var p = xseq_f32p(w)
    _ = w  # keep the Python object alive through the native call above
    if not xseq_all_finite_f32(p, n) or xseq_min_max_f32(p, n)[0] < Float32(0.0):
        raise Error("ValueError: mojolearn r2_score: sample_weight must have finite nonnegative entries")
    _ = w  # keep the Python object alive through the native calls above
    if not xseq_min_max_f32(p, n)[1] > Float32(0.0):
        raise Error("ValueError: mojolearn r2_score: sample_weight must have positive total weight")
    _ = w  # keep the Python object alive through the native calls above
    return w


def xseq_regressor_score(model: PythonObject, prediction: PythonObject, args: PythonObject,
                         context: PythonObject) raises -> PythonObject:
    """R^2 of `prediction` (the model's own predict of args[0]) against
    args[1], weighted by args[2] when given."""
    var truth = xseq_as_f32_c(args[1], 0, "y", context)
    var pred = xseq_as_f32_c(prediction, 0, "prediction", context)
    var sample_weight = args[2]
    if Int(py=truth.ndim) == 2 and Int(py=truth.shape[1]) != 1:
        # multi-output targets: the x_metrics program of `r2_score_options`
        var options = py_get(context, "xseq_r2_options")
        if options is None:
            raise Error("NotImplementedError: mojolearn: the multi-output score needs the native metrics owner "
                        + "(family_metric_r2_options); this install does not carry it")
        var mode = py_get(py_fields(model), "numeric_mode")
        return py_call(options[0], PythonObject(None), "family_metric_r2_options",
                       Python.tuple(truth, pred, sample_weight, "uniform_average", True, mode), options[1])
    var y = _vector(truth, "y_true")
    var yh = _vector(pred, "y_pred")
    var n = Int(py=y.shape[0])
    if Int(py=yh.shape[0]) != n:
        raise Error("ValueError: mojolearn metrics: y_true has " + String(n) + " entries and y_pred has "
                    + String(Int(py=yh.shape[0])) + "; they must be the same length")
    var entries = _metrics_entries(model, context)
    if sample_weight is not None:
        var w = _weights(sample_weight, n, context)
        var weighted = py_call_typed(entries["r2_score_weighted"], xseq_addr(y), xseq_addr(yh), xseq_addr(w), Python.list(n))
        _ = w  # keep the Python object alive through the native call above
        _ = yh  # keep the Python object alive through the native call above
        _ = y  # keep the Python object alive through the native call above
        _ = w^
        var _keep_ret_y = PythonObject(Float64(py=weighted))
        _ = yh  # keep the Python object alive through the native call above
        _ = y  # keep the Python object alive through the native call above
        return _keep_ret_y^
    var r2 = py_call_typed(entries["r2_score"], xseq_addr(y), xseq_addr(yh), Python.list(n))
    _ = yh  # keep the Python object alive through the native call above
    _ = y  # keep the Python object alive through the native call above
    _ = y^
    _ = yh^
    return PythonObject(Float64(py=r2))
