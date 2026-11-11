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
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_contains
from std.python._cpython import GILReleased
from core.metrics_entry_table import MetricsEntryKey, MR_R2, MR_R2_WEIGHTED
from bindings.xseq_native_common import xseq_addr, xseq_reshape
from bindings.xseq_neural_native import xseq_as_f32_c
from core.xseq_marshal import xseq_min_max_f32
from core.host_finite_scan import host_all_finite_f32
from bindings.metrics_reg_native import metrics_reg_operation
from bindings.metrics_cls_native import metrics_cls_operation
from sequence.xseq_score_run import (
    xseq_run_vector_rows, xseq_run_vector_finite, xseq_run_lengths_check, xseq_run_weights_check,
    xseq_run_metrics_mode, xseq_run_r2,
)


def _shape_words(value: PythonObject) raises -> List[Int]:
    """UNPACK: an ndarray's shape as words."""
    var shape = value.shape
    var words = List[Int]()
    for i in range(len(shape)):  # driver-loop: over shape words
        words.append(Int(py=shape[i]))
    return words^


def _vector(value: PythonObject, name: String) raises -> PythonObject:
    """`_as_f32_1d` of an already float32 array: `_vector_shape`'s refusals,
    the column flattened, then the finiteness refusal."""
    var shape = _shape_words(value)
    var n = xseq_run_vector_rows(shape, name)
    var out = xseq_reshape(value, Python.tuple(n)) if len(shape) == 2 else value
    xseq_run_vector_finite(Int(py=xseq_addr(out)), n, name)
    _ = out  # keep the Python object alive through the native scan above
    return out


def _metrics_key(model: PythonObject, context: PythonObject) raises -> MetricsEntryKey:
    """`_metrics_impl._get_binding(model.numeric_mode)`: the metrics image of
    that mode (or the process default), as its native R^2 table key (the
    bridge records it at import), checked against its compiled mode."""
    var b = native_loaded_module("builtins")
    var mode = py_get(py_fields(model), "numeric_mode")
    if mode is not None:
        if py_type(mode) is not b.str:
            raise Error("ValueError: numeric_mode must be 'fast' or 'identical'")
        var clean = String(py=py_call(py_call(mode.strip).lower))  # one strip/lower, not two
        if clean != "fast" and clean != "identical":
            raise Error("ValueError: numeric_mode must be 'fast' or 'identical'")
    var expected = context["backend_fields"]["_DEFAULT_MODE"] if mode is None else mode
    expected = py_call(py_call(expected.strip).lower)
    var table = context["xseq_metrics"]
    var record = table[expected]
    if py_type(record) is b.str:
        raise Error("ImportError: " + String(py=record))
    # The compiled-mode word the bridge read once at import replaced the
    # per-call `metrics_numeric_mode` Python call (lane calltable-xseq).
    var actual = Int(py=record[0])
    xseq_run_metrics_mode(String(py=expected), actual)
    return MetricsEntryKey(actual, Int(py=record[1]))


def _r2_native(key: MetricsEntryKey, operation: Int, var addresses: List[Int], n: Int) raises -> Float64:
    """The RUN R^2 call (sequence/xseq_score_run.mojo) with the GIL
    released. The caller keeps the addressed arrays alive."""
    var result = 0.0
    var py = Python()
    with GILReleased(py):
        result = xseq_run_r2(key, operation, addresses^, n)
    return result


def _weights(source: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """`_sample_weight_f32(sample_weight, n, "r2_score")`."""
    var w = xseq_as_f32_c(source, 0, "sample_weight", context)
    xseq_run_weights_check(_shape_words(w), n, Int(py=xseq_addr(w)))
    _ = w  # keep the Python object alive through the native scans above
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
        # pyout-metrics-cnn (2026-11-11): the metrics owner's body, called natively (no CPython call of
        # its family door); the operands are this call's own admitted arrays, which that owner admits in
        # its UNPACK, and options[1] is the metrics family context the door would have received.
        return metrics_reg_operation("family_metric_r2_options",
                                     Python.tuple(truth, pred, sample_weight, "uniform_average", True, mode),
                                     options[1])
    var y = _vector(truth, "y_true")
    var yh = _vector(pred, "y_pred")
    var n = Int(py=y.shape[0])
    xseq_run_lengths_check(n, Int(py=yh.shape[0]))
    var key = _metrics_key(model, context)
    if sample_weight is not None:
        var w = _weights(sample_weight, n, context)
        var weighted = _r2_native(key, MR_R2_WEIGHTED, [Int(py=xseq_addr(y)), Int(py=xseq_addr(yh)), Int(py=xseq_addr(w))], n)
        _ = w  # keep the Python object alive through the native call above
        _ = yh  # keep the Python object alive through the native call above
        _ = y  # keep the Python object alive through the native call above
        _ = w^
        var _keep_ret_y = PythonObject(weighted)
        _ = yh  # keep the Python object alive through the native call above
        _ = y  # keep the Python object alive through the native call above
        return _keep_ret_y^
    var r2 = _r2_native(key, MR_R2, [Int(py=xseq_addr(y)), Int(py=xseq_addr(yh))], n)
    _ = yh  # keep the Python object alive through the native call above
    _ = y  # keep the Python object alive through the native call above
    _ = y^
    _ = yh^
    return PythonObject(r2)


def xseq_classifier_score(model: PythonObject, prediction: PythonObject, args: PythonObject,
                          context: PythonObject) raises -> PythonObject:
    """`ClassifierScoreMixin.score` of the sequence classifiers (LSTM/GRU/RNN
    classifiers and MLPClassifier): the (weighted) accuracy of `prediction`
    (the model's own predict of args[0]) against args[1], weighted by
    args[2] when given, exactly `_expansion_metrics.accuracy_fraction(y,
    prediction, sample_weight, numeric_mode)`: the native metrics owner's
    `family_mcls_accuracy_fraction`, the same entry with the same arguments,
    so the score keeps its bits."""
    var owner = py_get(context, "xseq_r2_options")
    if owner is None:
        raise Error("NotImplementedError: mojolearn: the classifier score needs the native metrics owner "
                    + "(family_mcls_accuracy_fraction); this install does not carry it")
    var mode = py_get(py_fields(model), "numeric_mode")
    # pyout-metrics-cnn (2026-11-11): the metrics owner's body, called natively (no CPython call of its
    # family door); the labels are the caller's Python objects, admitted in that owner's UNPACK.
    return metrics_cls_operation(PythonObject(None), "family_mcls_accuracy_fraction",
                                 Python.tuple(args[1], prediction, args[2], mode), owner[1])
