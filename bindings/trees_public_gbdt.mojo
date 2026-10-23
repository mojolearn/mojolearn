# SPDX-License-Identifier: Apache-2.0
# The gradient-boosting public operations in one native call each:
# `GradientBoosting.predict` / `predict_proba` / `predict_classes`
# (python/mojolearn/ensemble.py, inherited by OrderedRMSE and
# ExperimentalTwoLevelFeatureFreq), the OrderedRMSE and FeatureFreq fits, and
# the scikit-learn adapters' `decision_function` / `predict` /
# `predict_proba` (python/mojolearn/_gbdt_adapters.py). This is the Python
# orchestration those methods ran, step for step: the fitted checks with
# their wording, the binding of the model's numeric mode, the device-resident
# model copy (prepared once, reused while the model text is the same bytes,
# released when its holder drops it), the input layout, the predict entry
# (resident, multi-output or single-output per-call parse, the CPU CTR
# reader) and the output shape. Same entries, same order, same bits.
#
# Round trips removed against the Python path: the resident handle lookup,
# the X staging and the launch run in one call (Python returned between the
# handle, the layout conversion and the predict entry); `predict_proba`'s
# binary fallback keeps the raw scores in the call and widens them for
# `gbdt_sigmoid_pair` without a Python `astype`; the adapters' margins go
# from the predict entry to the binary output kernel without a Python step.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_fields
from bindings.py_typed_call import py_call_typed
from bindings.classical_exceptions import classical_not_fitted_class
from bindings.host_tree_native import host_tree_operation
from bindings.trees_lifetime import trees_resident_new
from std.math import isfinite
from std.memory import memcpy
from core.trees_gbdt_metadata import gbdt_tree_metadata
from bindings.trees_public_inputs import (
    TreesMatrix, trees_builtins, trees_table, trees_entry, trees_need, trees_helper, trees_empty,
    trees_vector, trees_reshape, trees_address, trees_f32, trees_forest_layout, trees_all_finite,
    trees_cast, trees_decode_labels, trees_call, trees_view, trees_check_rank,
)

comptime _PREDICT_RAW = 0
comptime _PREDICT_SOFTMAX = 1
comptime _PREDICT_SIGMOID = 2
comptime _PREDICT_SIGMOID_PAIR = 3
comptime _PREDICT_CLASSES_PINNED = 5
comptime _PREDICT_CLASSES_OVA = 6


# --------------------------------------------------------------- binding


def _repr(value: PythonObject) raises -> String:
    return String(py_call(trees_builtins().repr, value))


def gbdt_mode(fields: PythonObject, context: PythonObject) raises -> String:
    """`NumericModeMixin._bind`'s mode: the estimator's `numeric_mode`, or
    the process default, checked as `_backend.binding` checks it."""
    var b = trees_builtins()
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if not Bool(py=py_call(b.isinstance, mode, b.str)):
        raise Error("numeric_mode must be fast, identical or None")
    var clean = String(py_call(py_call(mode.strip).lower))
    if clean != "fast" and clean != "identical":
        raise Error("numeric_mode must be fast, identical or None")
    return clean


def gbdt_table(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    return trees_table(context, gbdt_mode(fields, context), "_mojolearn_gbdt")


def gbdt_vendor(table: PythonObject) raises -> String:
    """`ensemble._binding_vendor`: the binding's vendor answer, or ''."""
    var entry = trees_entry(table, "gbdt_vendor")
    if entry is None:
        return ""
    return String(py_call(trees_builtins().str, py_call_typed(entry)))


def gbdt_has_ctr(text: PythonObject) raises -> Bool:
    """`ensemble._has_ctr_records`: a CTR table or tensor CTR registry."""
    var t = String(py_call(trees_builtins().str, text))
    var names: List[String] = ["ctr_columns", "ctr_table", "tensor_ctr_registry"]
    for i in range(3):  # small-loop(names): the three fixed record names
        if ("\n" + names[i] + " ") in t or t.startswith(names[i] + " "):
            return True
    return False


def gbdt_host_tree_context(context: PythonObject) raises -> PythonObject:
    """The CPU tree owner's context (`linear_model._CLASSICAL`'s
    `host_tree_context`, connected when the host surface loads), read from
    the loaded module's dictionary."""
    var module = py_get(context["modules"], "mojolearn.linear_model")
    if module is None:
        raise Error("ImportError: mojolearn: the CPU tree owner is not connected")
    var classical = py_get(py_fields(module), "_CLASSICAL")
    var host = py_get(classical, "host_tree_context") if classical is not None else PythonObject(None)
    if host is None:
        raise Error("ImportError: mojolearn: the CPU tree owner is not connected")
    return host


def _field(fields: PythonObject, model: PythonObject, name: String) raises -> PythonObject:
    """An attribute the model holds (instance first, then its class), or
    the AttributeError Python's attribute read raised."""
    var value = py_get(fields, PythonObject(name), PythonObject(None))
    if value is not None or py_contains(fields, PythonObject(name)):
        return value
    var cls = py_type(model)
    var b = trees_builtins()
    if Bool(py=py_call(b.hasattr, cls, PythonObject(name))):
        return py_call(b.getattr, cls, PythonObject(name))
    raise Error("AttributeError: '" + String(cls.__name__) + "' object has no attribute '" + name + "'")


def _int(value: PythonObject) raises -> Int:
    return Int(py=py_call_typed(trees_builtins().int, value))


# -------------------------------------------------------- resident model


def gbdt_resident(fields: PythonObject, table: PythonObject) raises -> Int:
    """`GradientBoosting._resident_handle`: the handle of the device copy of
    `model_`, prepared on the first call and reused while the binding and
    the text bytes are the same; -1 where the binding has no
    `gbdt_resident_prepare` (the per-call parse runs). The holder tuple
    (lifetime, binding, text, handle) releases the copy when it is
    dropped."""
    var prepare = trees_entry(table, "gbdt_resident_prepare")
    if prepare is None:
        return -1
    var text = fields["model_"]
    var cached = py_get(fields, "_resident")
    if cached is not None and py_type(cached) is trees_builtins().tuple and len(cached) == 4:
        if cached[1] is table and (cached[2] is text or Bool(py=cached[2] == text)):
            return Int(py=cached[3])
    # A different text or binding: dropping the holder releases its copy.
    _ = py_call(fields.pop, "_resident", PythonObject(None))
    var handle = _int(py_call_typed(prepare, text))
    var release = trees_entry(table, "gbdt_resident_release")
    var lifetime = trees_resident_new(release, handle)
    py_setitem(fields, "_resident", Python.tuple(lifetime, table, text, PythonObject(handle)))
    return handle


def _fitted_x(fields: PythonObject, X: PythonObject, layout: Bool, before: String, context: PythonObject) raises -> TreesMatrix:
    """`_check_fitted_layout` (layout) / `_check_fitted`: the fitted check,
    X in the entry's layout, the fitted width."""
    if fields["model_"] is None:
        raise Error("RuntimeError: mojolearn: " + before + " before fit()")
    var x = trees_forest_layout(X, "X", context) if layout else trees_f32(X, 2, "F", "X", context)
    var fitted = _int(fields["n_features_in_"])
    if x.cols != fitted:
        raise Error("mojolearn: model was fitted on " + String(fitted) + " features, got " + String(x.cols))
    return x^


def _resident_call(table: PythonObject, handle: Int, x: TreesMatrix, out: PythonObject, mode: Int) raises -> Int:
    var entry = trees_need(table, "gbdt_resident_predict", "mojolearn: this gbdt binding has no gbdt_resident_predict")
    return _int(py_call_typed(entry, PythonObject(handle), PythonObject(x.addr), PythonObject(trees_address(out)),
                              Python.list(PythonObject(x.rows), PythonObject(mode), PythonObject(1 if x.row_major else 0))))


# --------------------------------------------------------------- predict


def gbdt_predict(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """`GradientBoosting.predict`: raw float32 scores."""
    var fields = py_fields(model)
    var text = _field(fields, model, "model_")
    if text is None:
        raise Error("RuntimeError: mojolearn: predict() before fit()")
    var table = gbdt_table(fields, context)
    if gbdt_has_ctr(text) and gbdt_vendor(table) == "cpu":
        # The CPU restatement of a CTR model's apply (`HostGBDT`).
        var args = Python.tuple(X, fields["loss"], text, fields["n_features_in_"], fields["approx_dim_"],
                                fields["n_classes_"], py_type(model).__name__)
        return host_tree_operation(PythonObject(None), "host_tree_ctr_predict", args, gbdt_host_tree_context(context))
    var dim = _int(fields["approx_dim_"])
    var handle = gbdt_resident(fields, table)
    if handle >= 0:
        var x = _fitted_x(fields, X, True, "predict()", context)
        var out = trees_vector(x.rows * dim, "<f4", context)
        var width = _resident_call(table, handle, x, out, _PREDICT_RAW)
        if width != dim:
            raise Error("RuntimeError: mojolearn: predict wrote width " + String(width) + ", expected " + String(dim))
        if dim == 1:
            return out
        return trees_reshape(out, Python.tuple(x.rows, width), context)
    var x = _fitted_x(fields, X, False, "predict()", context)
    if dim > 1:
        var out = trees_vector(x.rows * dim, "<f4", context)
        var width = _int(py_call_typed(trees_need(table, "gbdt_predict_multi", "mojolearn: this gbdt binding has no gbdt_predict_multi"),
                                       text, PythonObject(x.addr), PythonObject(trees_address(out)),
                                       Python.list(PythonObject(x.rows), PythonObject(_PREDICT_RAW))))
        if width != dim:
            raise Error("RuntimeError: mojolearn: predict wrote width " + String(width) + ", expected " + String(dim))
        return trees_reshape(out, Python.tuple(x.rows, width), context)
    var out = trees_vector(x.rows, "<f4", context)
    var wrote = _int(py_call_typed(trees_need(table, "gbdt_predict", "mojolearn: this gbdt binding has no gbdt_predict"),
                                   text, PythonObject(x.addr), PythonObject(trees_address(out)), Python.list(PythonObject(x.rows))))
    if wrote != x.rows:
        raise Error("RuntimeError: mojolearn: predict wrote " + String(wrote) + " of " + String(x.rows) + " rows")
    return out


def _is_multi_output(loss: PythonObject) raises -> Bool:
    return Bool(py=loss == PythonObject("MultiClass")) or Bool(py=loss == PythonObject("MultiClassOneVsAll"))


def _is_binary(loss: PythonObject) raises -> Bool:
    return Bool(py=loss == PythonObject("Logloss")) or Bool(py=loss == PythonObject("CrossEntropy"))


def gbdt_predict_proba(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """`GradientBoosting.predict_proba`: softmax (MultiClass), elementwise
    sigmoid (MultiClassOneVsAll) or float64 `[1 - p, p]` (Logloss,
    CrossEntropy)."""
    var fields = py_fields(model)
    var loss = _field(fields, model, "loss")
    if _is_multi_output(loss):
        if _field(fields, model, "model_") is None:
            raise Error("RuntimeError: mojolearn: predict_proba() before fit()")
        var mode = _PREDICT_SIGMOID if Bool(py=loss == PythonObject("MultiClassOneVsAll")) else _PREDICT_SOFTMAX
        var table = gbdt_table(fields, context)
        var classes = _int(fields["n_classes_"])
        var handle = gbdt_resident(fields, table)
        var width = 0
        var rows = 0
        var out = PythonObject(None)
        if handle >= 0:
            var x = _fitted_x(fields, X, True, "predict()", context)
            rows = x.rows
            out = trees_vector(rows * classes, "<f4", context)
            width = _resident_call(table, handle, x, out, mode)
        else:
            var x = _fitted_x(fields, X, False, "predict()", context)
            rows = x.rows
            out = trees_vector(rows * classes, "<f4", context)
            width = _int(py_call_typed(trees_need(table, "gbdt_predict_multi", "mojolearn: this gbdt binding has no gbdt_predict_multi"),
                                       fields["model_"], PythonObject(x.addr), PythonObject(trees_address(out)),
                                       Python.list(PythonObject(rows), PythonObject(mode))))
        if width != classes:
            raise Error("RuntimeError: mojolearn: predict_proba wrote width " + String(width) + ", expected " + String(classes))
        return trees_reshape(out, Python.tuple(rows, width), context)
    if not _is_binary(loss):
        raise Error("mojolearn: predict_proba is defined for Logloss, "
                    + "CrossEntropy, MultiClass and MultiClassOneVsAll; this model was fitted with "
                    + _repr(loss) + ". Use predict() and apply the link "
                    + "yourself.")
    if _field(fields, model, "model_") is None:
        raise Error("RuntimeError: mojolearn: predict_proba() before fit()")
    var table = gbdt_table(fields, context)
    var handle = gbdt_resident(fields, table)
    if handle >= 0:
        var x = _fitted_x(fields, X, True, "predict()", context)
        var out = trees_empty(Python.tuple(x.rows, 2), "<f8", context)
        var width = _resident_call(table, handle, x, out, _PREDICT_SIGMOID_PAIR)
        if width != 2:
            raise Error("RuntimeError: mojolearn: predict_proba wrote width " + String(width) + ", expected 2")
        return out
    var raw32 = gbdt_predict(model, X, context)
    var n = Int(py=raw32.shape[0])
    var raw = trees_vector(Int(py=raw32.size), "<f8", context)
    trees_cast(trees_address(raw32), "<f4", trees_address(raw), "<f8", Int(py=raw32.size), context)
    var pair = trees_entry(table, "gbdt_sigmoid_pair")
    if pair is None:
        raise Error("RuntimeError: mojolearn: this gbdt binding predates gbdt_sigmoid_pair; rebuild it")
    var out = trees_empty(Python.tuple(n, 2), "<f8", context)
    var wrote = _int(py_call_typed(pair, PythonObject(trees_address(raw)), PythonObject(trees_address(out)), PythonObject(n)))
    if wrote != n:
        raise Error("RuntimeError: mojolearn: gbdt_sigmoid_pair wrote " + String(wrote) + " of " + String(n) + " rows")
    return out


def _numeric_mode_used(table: PythonObject) raises -> String:
    """`numeric_mode_used()`: the binary's compiled mode."""
    var getter = trees_entry(table, "gbdt_numeric_mode")
    if getter is None:
        raise Error("RuntimeError: mojolearn: _mojolearn_gbdt has no gbdt_numeric_mode")
    var code = _int(py_call_typed(getter))
    if code == 0:
        return "fast"
    if code == 1:
        return "identical"
    raise Error("RuntimeError: mojolearn: _mojolearn_gbdt returned unknown numeric mode " + String(code))


def _argmax_rows(scores: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_labels.argmax_rows` of a C-order float32 or float64 block."""
    var rows = Int(py=scores.shape[0])
    var cols = Int(py=scores.shape[1])
    var out = trees_vector(rows, "<i8", context)
    if rows == 0:
        return out
    var wide = String(scores.dtype) == "<f8"
    _ = py_call_typed(trees_helper(context, "argmax_rows_f64" if wide else "argmax_rows_f32"),
                      PythonObject(trees_address(scores)), PythonObject(rows), PythonObject(cols),
                      PythonObject(trees_address(out)))
    return out


def gbdt_predict_classes(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """`GradientBoosting.predict_classes`: int64 class codes; under FAST the
    resident direct-class kernels where they apply, else the argmax of
    `predict_proba`."""
    var fields = py_fields(model)
    var loss = _field(fields, model, "loss")
    if not _is_multi_output(loss) and not _is_binary(loss):
        raise Error("mojolearn: predict_classes needs a classification loss; "
                    + "this model was fitted with " + _repr(loss) + ".")
    var table = gbdt_table(fields, context)
    if _numeric_mode_used(table) == "fast":
        if _is_binary(loss):
            var direct = trees_entry(table, "gbdt_resident_binary_classes")
            if direct is not None:
                var handle = gbdt_resident(fields, table)
                if handle >= 0:
                    var x = _fitted_x(fields, X, True, "predict()", context)
                    var out = trees_vector(x.rows, "<i8", context)
                    var wrote = _int(py_call_typed(direct, PythonObject(handle), PythonObject(x.addr),
                                                   PythonObject(trees_address(out)),
                                                   Python.list(PythonObject(x.rows), PythonObject(1 if x.row_major else 0))))
                    if wrote != x.rows:
                        raise Error("RuntimeError: mojolearn: predict_classes wrote " + String(wrote) + " of "
                                    + String(x.rows) + " rows")
                    return out
        elif _int(fields["n_classes_"]) <= 3 and Bool(py=_field(fields, model, "grow_policy") == PythonObject("SymmetricTree")):
            if fields["model_"] is None:
                raise Error("RuntimeError: mojolearn: predict_proba() before fit()")
            var handle = gbdt_resident(fields, table)
            if handle >= 0:
                var x = _fitted_x(fields, X, True, "predict()", context)
                var out = trees_vector(x.rows, "<i8", context)
                var mode = _PREDICT_CLASSES_PINNED if Bool(py=loss == PythonObject("MultiClass")) else _PREDICT_CLASSES_OVA
                var width = _resident_call(table, handle, x, out, mode)
                if width != 1:
                    raise Error("RuntimeError: mojolearn: predict_classes wrote width " + String(width) + ", expected 1")
                return out
    return _argmax_rows(gbdt_predict_proba(model, X, context), context)


# ------------------------------------------------------------- adapters


def _adapter_fitted(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_GBDTAdapter._check_fitted`: the fitted learner, or NotFittedError."""
    var fields = py_fields(model)
    var learner = py_get(fields, "_learner_")
    var fitted = learner is not None and py_get(py_fields(learner), "model_") is not None
    if not fitted:
        _ = classical_not_fitted_class(context)
        raise Error("NotFittedError: " + String(py_type(model).__name__) + " is not fitted")
    return learner


def adapter_decision_function(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """`GradientBoostingClassifier.decision_function` and
    `GradientBoostingRegressor.predict`: the learner's raw scores."""
    return gbdt_predict(_adapter_fitted(model, context), X, context)


def _adapter_binary(model: PythonObject, X: PythonObject, probabilities: Bool, context: PythonObject) raises -> PythonObject:
    """`_GBDTAdapter._binary_output`: FAST labels straight from the
    learner's class kernels; else the margins through the binding's binary
    probability or class kernel."""
    var fields = py_fields(model)
    var mode = py_get(fields, "numeric_mode_")
    if not probabilities and mode is not None and Bool(py=mode == PythonObject("fast")):
        var learner = py_get(fields, "_learner_")
        if learner is None:
            raise Error("AttributeError: '" + String(py_type(model).__name__) + "' object has no attribute '_learner_'")
        return gbdt_predict_classes(learner, X, context)
    var margins = adapter_decision_function(model, X, context)
    if Int(py=margins.ndim) != 1 or String(margins.dtype) != "<f4" \
            or not trees_all_finite(trees_address(margins), Int(py=margins.size), False, context):
        raise Error("Classifier margins must be finite scalar Float32 values")
    var n = Int(py=margins.size)
    var table = trees_table(context, String(mode) if mode is not None else gbdt_mode(fields, context), "_mojolearn_gbdt")
    var output = PythonObject(None)
    var wrote = 0
    if probabilities:
        output = trees_empty(Python.tuple(n, 2), "<f4", context)
        wrote = _int(py_call_typed(trees_need(table, "gbdt_binary_probabilities", "mojolearn: this gbdt binding has no gbdt_binary_probabilities"),
                                   PythonObject(trees_address(margins)), PythonObject(trees_address(output)), Python.list(PythonObject(n))))
    else:
        output = trees_vector(n, "<i4", context)
        wrote = _int(py_call_typed(trees_need(table, "gbdt_binary_classes", "mojolearn: this gbdt binding has no gbdt_binary_classes"),
                                   PythonObject(trees_address(margins)), PythonObject(trees_address(output)), Python.list(PythonObject(n))))
    if wrote != (2 * n if probabilities else n):
        raise Error("RuntimeError: GBDT binary output returned an unexpected row count")
    return output


def adapter_predict_proba(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    return _adapter_binary(model, X, True, context)


def adapter_codes(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_binary_output(X, False)`: the class codes `score` compares."""
    return _adapter_binary(model, X, False, context)


def adapter_predict(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """`GradientBoostingClassifier.predict`: the class codes decoded through
    `classes_`."""
    _ = _adapter_fitted(model, context)
    var codes = _adapter_binary(model, X, False, context)
    return trees_decode_labels(py_fields(model)["classes_"], codes, context)


# -------------------------------------------------------- tree metadata


def gbdt_tree_metadata_op(model: PythonObject, which: String, context: PythonObject) raises -> PythonObject:
    """`get_tree_leaf_counts` (uint32 per tree) / `get_leaf_values`
    (float64, tree-major then leaf-major) from the model text."""
    var fields = py_fields(model)
    var text = _field(fields, model, "model_")
    if text is None:
        raise Error("RuntimeError: mojolearn: tree inspection before fit()")
    var parsed = gbdt_tree_metadata(String(py_call(trees_builtins().str, text)))
    if which == "counts":
        var n = len(parsed.counts)
        var out = trees_vector(n, "<u4", context)
        if n:
            memcpy(dest=MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=trees_address(out)),
                   src=parsed.counts.unsafe_ptr(), count=n)
        return out
    var n = len(parsed.values)
    var out = trees_vector(n, "<f8", context)
    if n:
        memcpy(dest=MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=trees_address(out)),
               src=parsed.values.unsafe_ptr(), count=n)
    return out


# ------------------------------------------------- OrderedRMSE / FeatureFreq


def _is_bool(value: PythonObject) raises -> Bool:
    var t = py_type(value)
    return t is trees_builtins().bool or String(t.__name__) == "bool_"


def _isinstance(value: PythonObject, cls: PythonObject) raises -> Bool:
    return Bool(py=py_call(trees_builtins().isinstance, value, cls))


def _ordered_options(fields: PythonObject, model: PythonObject, context: PythonObject) raises:
    """`OrderedRMSE._ordered_options` over the stored settings."""
    var b = trees_builtins()
    var names: List[String] = ["n_estimators", "max_depth", "border_count"]
    var upper: List[Int] = [2147483647, 8, 255]
    for i in range(3):  # small-loop(names): the three integer settings
        var value = _field(fields, model, names[i])
        var ok = not _is_bool(value) and _isinstance(value, context["trees_integral_type"])
        if ok:
            ok = Bool(py=value >= PythonObject(1)) and Bool(py=value <= PythonObject(upper[i]))
        if not ok:
            raise Error("mojolearn: " + names[i] + " must be an integer in 1.." + String(upper[i]))
    var reals: List[String] = ["learning_rate", "l2_leaf_reg"]
    for i in range(2):  # small-loop(reals): the two real settings
        var value = _field(fields, model, reals[i])
        var positive = i == 0
        var ok = not _is_bool(value) and _isinstance(value, context["trees_real_type"])
        if ok:
            var f = Float64(py=py_call_typed(b.float, value))
            ok = isfinite(f) and abs(f) <= 3.4028234663852886e+38 and f >= 0 and not (positive and Float32(f) <= 0)
        if not ok:
            raise Error("mojolearn: " + reals[i] + " must be finite float32 and "
                        + ("positive" if positive else "non-negative"))


def _type_or_value(error: Error) -> Bool:
    """A refusal Python's `except (TypeError, ValueError)` catches."""
    var message = String(error)
    return not (message.startswith("RuntimeError:") or message.startswith("ImportError:")
                or message.startswith("AttributeError:") or message.startswith("IndexError:")
                or message.startswith("NotImplementedError:") or message.startswith("OverflowError:"))


def _reduce(addr: Int, code: Int, n: Int, what: Int, context: PythonObject) raises -> PythonObject:
    """The core helper `reduce_stat` (0 min, 1 max, 4 integral)."""
    return py_call_typed(trees_helper(context, "reduce_stat"), PythonObject(addr), PythonObject(code),
                         PythonObject(n), PythonObject(what))


def _checked_weights(source: PythonObject, n: Int, wrap: Bool, context: PythonObject) raises -> TreesMatrix:
    """The OrderedRMSE / FeatureFreq weight checks: one value per row,
    finite, non-negative, a positive sum."""
    var w: TreesMatrix
    try:
        w = trees_f32(source, 1, "C", "sample_weight", context)
    except error:
        if wrap and _type_or_value(error):
            raise Error("mojolearn: sample_weight must have one value per row")
        raise error^
    if w.rows != n:
        raise Error("mojolearn: sample_weight must have one value per row")
    if not trees_all_finite(w.addr, n, False, context) or Bool(py=_reduce(w.addr, 0, n, 0, context) < PythonObject(0)):
        raise Error("mojolearn: sample_weight must be finite and non-negative")
    if not Bool(py=_reduce(w.addr, 0, n, 1, context) > PythonObject(0)):
        raise Error("mojolearn: sample_weight must have positive sum")
    return w^


def _fitted_text(fields: PythonObject, model: PythonObject, n_features: Int, iteration: PythonObject) raises:
    py_setitem(fields, "n_features_in_", PythonObject(n_features))
    py_setitem(fields, "approx_dim_", PythonObject(1))
    py_setitem(fields, "n_classes_", PythonObject(None))
    py_setitem(fields, "bias_", PythonObject(0.0))
    py_setitem(fields, "best_iteration_", iteration)
    py_setitem(fields, "stopped_early_", PythonObject(False))
    py_setitem(fields, "loss_curve_", PythonObject(None))
    py_setitem(fields, "test_loss_curve_", PythonObject(None))


def ordered_rmse_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`OrderedRMSE.fit(X, y, permutation=, sample_weight=)`."""
    var b = trees_builtins()
    var fields = py_fields(model)
    _ordered_options(fields, model, context)
    var x = trees_f32(args[0], 2, "F", "X", context)
    var n = x.rows
    if n < 4 or n > 4294967295 or x.cols < 1:
        raise Error("mojolearn: ordered RMSE requires >=4 rows and >=1 feature")
    if not trees_all_finite(x.addr, n * x.cols, False, context):
        raise Error("mojolearn: ordered RMSE X must be finite")
    var y: TreesMatrix
    try:
        y = trees_f32(args[1], 1, "C", "y", context)
    except error:
        if _type_or_value(error):
            raise Error("mojolearn: y must be one-dimensional with one value per row")
        raise error^
    if y.rows != n:
        raise Error("mojolearn: y must be one-dimensional with one value per row")
    if not trees_all_finite(y.addr, n, False, context):
        raise Error("mojolearn: y must be finite")
    var permutation = args[2]
    var bijection = String("mojolearn: permutation must be an integer bijection of row ids")
    if py_type(permutation) is b.list or py_type(permutation) is b.tuple:
        for i in range(len(permutation)):  # cpu-route: user permutation Python list type check [py-data-loop]
            var v = permutation[i]
            if _is_bool(v) or not _isinstance(v, context["trees_integral_type"]):
                raise Error(bijection)
    elif py_type(permutation) is context["array_class"]:
        var letter = String(String(permutation.dtype)[byte=1:2])
        if letter != "i" and letter != "u":
            raise Error(bijection)
    else:
        var view = PythonObject(None)
        try:
            view = py_call(b.memoryview, permutation)
        except:
            view = PythonObject(None)
        if view is not None:
            var fmt = String(view.format)
            var code = String(fmt[byte=fmt.byte_length() - 1:fmt.byte_length()])
            if code == "?" or code == "f" or code == "d" or code == "e" or code == "O":
                raise Error(bijection)
    var order64 = PythonObject(None)
    try:
        var view = trees_view_i64(permutation, context)
        order64 = view
    except error:
        if _type_or_value(error) or String(error).startswith("OverflowError:"):
            raise Error(bijection)
        raise error^
    var m = Int(py=order64.size)
    if m != n or n < 1 or _int(py_call_typed(trees_helper(context, "check_indices_i64"),
                                             PythonObject(trees_address(order64)), PythonObject(n), PythonObject(n))) != 0:
        raise Error(bijection)
    var order = trees_vector(n, "<u4", context)
    trees_cast(trees_address(order64), "<i8", trees_address(order), "<u4", n, context)
    var w_addr = y.addr
    var n_weights = 0
    var weights: Optional[TreesMatrix] = None
    if args[3] is not None:
        weights = _checked_weights(args[3], n, True, context)
        w_addr = weights.value().addr
        n_weights = n
    var table = gbdt_table(fields, context)
    var entry = trees_entry(table, "gbdt_fit_ordered_rmse")
    if entry is None:
        raise Error("RuntimeError: mojolearn: this GBDT binary predates OrderedRMSE; "
                    + "install or build a matching native extension")
    var params = Python.list(PythonObject(n), PythonObject(x.cols), PythonObject(n_weights), PythonObject(n),
                             py_call_typed(b.int, _field(fields, model, "n_estimators")),
                             py_call_typed(b.int, _field(fields, model, "max_depth")),
                             py_call_typed(b.int, _field(fields, model, "border_count")),
                             py_call_typed(b.float, _field(fields, model, "learning_rate")),
                             py_call_typed(b.float, _field(fields, model, "l2_leaf_reg")))
    var text = py_call_typed(entry, PythonObject(x.addr), PythonObject(y.addr), PythonObject(w_addr),
                             PythonObject(trees_address(order)), params)
    _ = weights^
    py_setitem(fields, "model_", text)
    _fitted_text(fields, model, x.cols, PythonObject(None))
    return model


def trees_view_i64(source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_buffer.as_i64_c(source, ndim=1, name="permutation")`: a C-order
    int64 vector (the input itself, or one converted copy)."""
    var b = trees_builtins()
    var prepared = source
    if py_type(source) is b.range:
        # `_materialize` lists a range's ints (a C iteration).
        prepared = py_call(b.list, source)
    var view = trees_view(prepared, "permutation", context)
    trees_check_rank(view, 1, "permutation")
    var n = view.size()
    var out = trees_vector(n, "<i8", context)
    trees_cast(view.addr, view.typestr, trees_address(out), "<i8", n, context)
    _ = view^
    return out


def feature_freq_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`ExperimentalTwoLevelFeatureFreq.fit(X, y, sample_weight=)`."""
    var b = trees_builtins()
    var fields = py_fields(model)
    var x = trees_f32(args[0], 2, "F", "X", context)
    var n = x.rows
    var d = x.cols
    if n == 0 or d < 2:
        raise Error("mojolearn: experimental FeatureFreq needs rows and columns")
    var sources = _field(fields, model, "sources")
    for i in range(len(sources)):  # driver-loop: over the named source columns
        var f = _int(sources[i])
        if f < 0 or f >= d:
            raise Error("mojolearn: sources " + _repr(sources) + " outside " + String(d) + " features")
    if not trees_all_finite(x.addr, n * d, False, context):
        raise Error("mojolearn: experimental FeatureFreq X must be finite")
    var is_source = List[Bool](length=d, fill=False)
    for i in range(len(sources)):  # driver-loop: over the named source columns, one native stat set each
        var f = _int(sources[i])
        is_source[f] = True
        var col = x.addr + 4 * f * n
        var hi = Float64(py=_reduce(col, 0, n, 1, context))
        if not _truthy_int(_reduce(col, 0, n, 4, context)) or Bool(py=_reduce(col, 0, n, 0, context) < PythonObject(0)):
            raise Error("mojolearn: experimental FeatureFreq source columns must "
                        + "contain non-negative integer category codes")
        var k = Int(hi) + 1
        var dense = 2 <= k and k <= n
        if dense:
            var codes = trees_vector(n, "<i8", context)
            trees_cast(col, "<f4", trees_address(codes), "<i8", n, context)
            var counts = trees_vector(k, "<i8", context)
            _ = py_call_typed(trees_helper(context, "bincount_i64"), PythonObject(trees_address(codes)), PythonObject(3),
                              PythonObject(n), PythonObject(k), PythonObject(trees_address(counts)), PythonObject(0))
            dense = Bool(py=_reduce(trees_address(counts), 3, k, 0, context) > PythonObject(0))
        if not dense:
            raise Error("mojolearn: every experimental FeatureFreq source must "
                        + "be densely coded 0..k-1; column " + String(f) + " has codes up to " + String(Int(hi))
                        + " with at least one missing (or fewer than two codes)")
    for f in range(d):  # driver-loop: over the feature columns, one native stat pair each
        if is_source[f]:
            continue
        var col = x.addr + 4 * f * n
        if Bool(py=py_call(b.float, _reduce(col, 0, n, 0, context)) == py_call(b.float, _reduce(col, 0, n, 1, context))):
            raise Error("mojolearn: experimental FeatureFreq numeric columns "
                        + "must vary; column " + String(f) + " is constant")
    var y = trees_f32(args[1], 1, "C", "y", context)
    if y.rows != n or not trees_all_finite(y.addr, n, False, context):
        raise Error("mojolearn: y must be finite with one value per row")
    var w_addr = y.addr
    var n_weights = 0
    var weights: Optional[TreesMatrix] = None
    if args[2] is not None:
        weights = _checked_weights(args[2], n, False, context)
        w_addr = weights.value().addr
        n_weights = n
    var k = len(sources)
    var source_array = trees_vector(k, "<u4", context)
    var sp = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=trees_address(source_array))
    for i in range(k):  # driver-loop: over the named source columns
        sp[i] = UInt32(_int(sources[i]))
    var table = gbdt_table(fields, context)
    var entry = trees_need(table, "gbdt_fit_two_level_feature_freq",
                           "mojolearn: this gbdt binding has no gbdt_fit_two_level_feature_freq", "AttributeError")
    var params = Python.list(PythonObject(n), PythonObject(d), PythonObject(n_weights), PythonObject(k),
                             py_call_typed(b.float, _field(fields, model, "learning_rate")),
                             py_call_typed(b.float, _field(fields, model, "l2_leaf_reg")),
                             py_call_typed(b.int, _field(fields, model, "random_state")))
    var text = py_call_typed(entry, PythonObject(x.addr), PythonObject(y.addr), PythonObject(w_addr),
                             PythonObject(trees_address(source_array)), params)
    _ = weights^
    py_setitem(fields, "model_", text)
    _fitted_text(fields, model, d, PythonObject(0))
    return model


def _truthy_int(value: PythonObject) raises -> Bool:
    return Int(py=py_call_typed(trees_builtins().int, value)) != 0
