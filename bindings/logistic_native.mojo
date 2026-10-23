# SPDX-License-Identifier: Apache-2.0
# The native owner of LogisticRegression's public fit, decision_function,
# predict, predict_proba, predict_log_proba and score. Each is one call into
# logistic_c, which checks the parameters (scikit-learn's and cuML's wording),
# converts X, encodes the labels, checks the sample and class weights, runs the
# existing estimators entries (qn_fit, qn_decision_function, qn_predict_binary,
# qn_predict_multiclass, qn_sigmoid, qn_softmax, py2mojo_rows) with the
# arguments the Python shell passed, in the same order, so the same bits, and
# stores the fitted attributes. score predicts and scores in the same call:
# the decoded prediction is handed straight to the native accuracy program
# (bindings/classical_labels.mojo `classical_accuracy`).
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_arrays import classical_address, classical_host_f32
from bindings.classical_labels import (
    ClassicalLabels, LabelsMatrix, classical_encode_labels, classical_decode_labels, classical_accuracy,
    labels_check_sample_weight, labels_class_weight_rows, labels_argmax_rows, labels_shape_of,
    labels_empty, labels_vector, labels_address,
)

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]

# linear_model._QN_LOSS_LOGISTIC / _QN_LOSS_SOFTMAX
comptime QN_LOSS_LOGISTIC = 0
comptime QN_LOSS_SOFTMAX = 2
# linear_model._ROWS_LOG and the py2mojo_linear_flags bit that routes it
comptime ROWS_LOG = 1
comptime PY2MOJO_ROWS = 2


def _builtins() raises -> PythonObject:
    return Python.import_module("builtins")


def _repr(value: PythonObject) raises -> String:
    return String(py_call(_builtins().repr, value))


def _name(model: PythonObject) raises -> String:
    return String(py_type(model).__name__)


def _mode(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_backend.binding(name, self.numeric_mode)`'s mode: the instance's,
    or the process default."""
    var b = _builtins()
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not b.str:
        raise Error("numeric_mode must be fast, identical or None")
    mode = py_call(py_call(mode.strip).lower)
    if String(mode) != "identical" and String(mode) != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    return mode


def _table(tables: PythonObject, mode: PythonObject) raises -> PythonObject:
    # A refusal recorded at import as "<Class>: <message>" is raised again.
    var table = tables[mode]
    if py_type(table) is _builtins().str:
        raise Error(String(table))
    return table


def _host_mode(model: PythonObject, fields: PythonObject) raises -> PythonObject:
    """`_HostBound._bind`'s mode: the explicit host classes run identical
    only."""
    var mode = py_get(fields, "numeric_mode")
    if mode is not None and Bool(py=mode != PythonObject("identical")):
        raise Error("mojolearn: " + _name(model) + " runs IDENTICAL only on the host; this model was saved "
                    + _repr(mode))
    return PythonObject("identical")


def _entries(model: PythonObject, fields: PythonObject, context: PythonObject) raises -> PythonObject:
    """The estimators entries of the estimator's mode, checked against the
    mode the binary was compiled for (`_backend.binding`'s refusal)."""
    var host = py_get(context, "lr_host_model")
    var mode = _host_mode(model, fields) if host is not None else _mode(fields, context)
    var entries = _table(context["lr_bindings"], mode)
    var compiled = Int(py=py_call_typed(entries["estimators_numeric_mode"]))
    var expected = 1 if String(mode) == "identical" else 0
    if compiled != expected:
        var name = String("identical") if compiled == 1 else (String("fast") if compiled == 0 else String("unknown"))
        raise Error("RuntimeError: mojolearn: _mojolearn_estimators was compiled for " + name
                    + ", but this call requested " + String(mode) + "; rebuild the " + String(mode) + " binding")
    return entries


def _class_constant(model: PythonObject, fields: PythonObject, name: String) raises -> PythonObject:
    """`self.<name>` for LogisticRegression's class constants
    (`lbfgs_memory`, `penalty_normalized`): the instance's own value, else
    the class attribute."""
    var own = py_get(fields, PythonObject(name))
    if own is not None:
        return own
    return py_call(_builtins().getattr, py_type(model), PythonObject(name))


def _need(entries: PythonObject, name: String) raises -> PythonObject:
    """A connected entry the operation requires, refused by name when the
    binary lacks it (an inference-only host binary has no qn_fit)."""
    var entry = py_get(entries, PythonObject(name))
    if entry is None:
        raise Error("AttributeError: mojolearn: the estimators binding has no " + name)
    return entry


def _int(value: PythonObject) raises -> Int:
    return Int(py=py_call(_builtins().int, value))


def _float(value: PythonObject) raises -> Float64:
    return Float64(py=py_call(_builtins().float, value))


# ------------------------------------------------------------ parameters


def _validate_params(fields: PythonObject) raises -> PythonObject:
    """`LogisticRegression._validate_params`: the constructor checks, run at
    the start of `fit`. Sets `_l1_ratio` and returns it."""
    var penalty = fields["penalty"]
    var solver = fields["solver"]
    var C = fields["C"]
    var l1_ratio = fields["l1_ratio"]
    var b = _builtins()
    if not py_contains(Python.tuple("l1", "l2", "elasticnet", PythonObject(None)), penalty):
        raise Error("`penalty` " + _repr(penalty) + " not supported.")
    if Bool(py=solver != PythonObject("qn")):
        raise Error("Only quasi-newton `qn` solver is supported, not " + String(py_call(b.str, solver)))
    if Bool(py=C <= PythonObject(0)):
        raise Error("C must be positive, got " + String(py_call(b.format, C)))
    py_setitem(fields, "_l1_ratio", PythonObject(None))
    if Bool(py=penalty == PythonObject("elasticnet")):
        if l1_ratio is None:
            raise Error("l1_ratio has to be specified for loss='elasticnet'")
        if Bool(py=l1_ratio < PythonObject(0.0)) or Bool(py=l1_ratio > PythonObject(1.0)):
            raise Error("l1_ratio value has to be between 0.0 and 1.0")
        py_setitem(fields, "_l1_ratio", l1_ratio)
    return fields["_l1_ratio"]


def _qn_params(fields: PythonObject, l1_ratio: PythonObject) raises -> Tuple[PythonObject, PythonObject]:
    """`_get_qn_params`: (l1_strength, l2_strength), in Python's own
    arithmetic on the parameter objects (the same doubles)."""
    var penalty = fields["penalty"]
    var C = fields["C"]
    var zero = PythonObject(0.0)
    if penalty is None:
        return (zero, zero)
    var strength = PythonObject(1.0) / C
    if Bool(py=penalty == PythonObject("l1")):
        return (strength, zero)
    if Bool(py=penalty == PythonObject("l2")):
        return (zero, strength)
    return (l1_ratio * strength, (PythonObject(1.0) - l1_ratio) * strength)


def _row_weights(model: PythonObject, sample_weight: PythonObject, class_weight: PythonObject,
                 labels: ClassicalLabels, n: Int, context: PythonObject) raises -> LabelsMatrix:
    """`linear_model._qn_row_weights`: no weights (address 0) for an
    unweighted fit, else the checked float32 sample weights times each row's
    class weight (their owner kept alive with the address)."""
    var b = _builtins()
    if sample_weight is None and class_weight is None:
        return LabelsMatrix(PythonObject(None), 0, 0, 0, True)
    var sw = labels_check_sample_weight(sample_weight, n, context)
    if class_weight is None:
        return sw^
    var refused = False
    if Bool(py=py_call(b.isinstance, class_weight, b.str)):
        refused = Bool(py=class_weight != PythonObject("balanced"))
    elif not Bool(py=py_call(b.isinstance, class_weight, b.dict)):
        refused = True
    if refused:
        raise Error("The 'class_weight' parameter of " + _name(model) + " must be an "
                    + "instance of 'dict', a str among {'balanced'} or None. "
                    + "Got " + _repr(class_weight) + " instead.")
    var host = py_get(context, "lr_host_model")
    var xtrees = _table(context["lr_x_trees"], _host_mode(model, py_fields(model)) if host is not None
                        else _mode(py_fields(model), context))
    var cw = labels_class_weight_rows(class_weight, labels, xtrees, context)
    if cw is None:  # every class weight is 1
        return sw^
    var out = labels_vector(n, "<f4", context)
    var mul = py_get(xtrees, "x_trees_mul_f32")
    if mul is None:
        raise Error("ImportError: the x_trees binding has no x_trees_mul_f32")
    _ = py_call_typed(mul, PythonObject(sw.addr), PythonObject(labels_address(cw)), PythonObject(labels_address(out)),
                      Python.list(n))
    _ = out  # keep the Python object alive through the native call above
    _ = cw  # keep the Python object alive through the native call above
    _ = sw^
    _ = cw^
    var _keep_ret_out = LabelsMatrix(out, labels_address(out), n, 1, True)
    _ = out  # keep the Python object alive through the native call above
    return _keep_ret_out^


# ------------------------------------------------------------ fit


def logistic_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = _builtins()
    var fields = py_fields(model)
    var l1_ratio = _validate_params(fields)
    var x = classical_host_f32(args[0], 2, "X", context)
    py_setitem(fields, "input_copied_", PythonObject(not (x is args[0])))
    var rows = Int(py=x.shape[0])
    var cols = Int(py=x.shape[1])
    var y = args[1]
    if len(labels_shape_of(y, context)) != 1:
        raise Error("mojolearn LogisticRegression requires a 1-D y")
    var labels = classical_encode_labels(y, context)
    py_setitem(fields, "classes_", labels.classes)
    if labels.n != rows:
        raise Error("mojolearn LogisticRegression X and y lengths differ")
    var n_classes = len(labels.classes)
    if n_classes < 2:
        raise Error("mojolearn LogisticRegression: y has one class")
    var n_targets = 1 if n_classes == 2 else n_classes
    var loss = QN_LOSS_LOGISTIC if n_targets == 1 else QN_LOSS_SOFTMAX
    # The dense codes as float32 (codes < 2**24: exact), `codes.astype('<f4')`.
    var y_enc = labels_vector(rows, "<f4", context)
    var codes = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=labels_address(labels.codes))
    var target = _F32P(unsafe_from_address=labels_address(y_enc))
    _ = y_enc  # keep the Python object alive through the native call above
    for i in range(rows):  # cpu-route: the label codes as solver targets, the explicit label input step
        target[i] = Float32(codes[i])
        _ = y_enc  # keep the Python object alive through the native call above
    var strengths = _qn_params(fields, l1_ratio)
    var l1 = strengths[0]
    var l2 = strengths[1]
    if n_targets > 1 and Bool(py=l1 != PythonObject(0.0)):
        raise Error("NotImplementedError: mojolearn LogisticRegression: penalty=" + _repr(fields["penalty"]) + " with "
                    + String(n_classes) + " classes selects OWL-QN on the softmax objective, "
                    + "which runs but has no identity gate (glm/checks/"
                    + "multinomial_check.mojo gates l2 and no penalty); refused by "
                    + "name until it is gated (CONTRIBUTING.md (Non-default paths))")
    var sw = _row_weights(model, args[2], fields["class_weight"], labels, rows, context)
    var entries = _entries(model, fields, context)
    var intercept = Bool(py=fields["fit_intercept"])
    var n_param = cols + (1 if intercept else 0)
    var w = labels_vector(n_param * n_targets, "<f4", context)
    var info = labels_vector(2, "<f4", context)
    var tol = fields["tol"]
    var has_sw = sw.addr != 0
    _ = sw  # keep the Python object alive through the native call above
    var params = Python.list(rows, cols, n_classes,
                             py_call(b.float, l1), py_call(b.float, l2), py_call(b.float, tol),
                             py_call(b.float, tol * PythonObject(0.01)),
                             _int(fields["max_iter"]), _int(fields["linesearch_max_iter"]),
                             _int(_class_constant(model, fields, "lbfgs_memory")), 1 if intercept else 0,
                             1 if Bool(py=_class_constant(model, fields, "penalty_normalized")) else 0,
                             1 if has_sw else 0, loss)
    if has_sw:
        # field 15 svr_eps (0), field 16 the weights
        _ = py_call(params.append, PythonObject(0.0))
        _ = py_call(params.append, PythonObject(sw.addr))
        _ = sw  # keep the Python object alive through the native call above
    var n_iter = py_call_typed(_need(entries, "qn_fit"), classical_address(x), PythonObject(labels_address(y_enc)),
                               PythonObject(labels_address(w)), PythonObject(labels_address(info)), params)
    _ = y_enc  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _ = info  # keep the Python object alive through the native call above
    _ = w  # keep the Python object alive through the native call above
    py_setitem(fields, "_w", w)
    # `_coef_from_w`: one target, the first `cols` words and the last; C
    # targets, cuML's column-major block w[c + C*j], the bias at j == cols.
    var source = _F32P(unsafe_from_address=labels_address(w))
    _ = w  # keep the Python object alive through the native call above
    var coef = labels_empty(Python.tuple(n_targets, cols), "<f4", context)
    var inter = labels_vector(n_targets, "<f4", context)
    var cp = _F32P(unsafe_from_address=labels_address(coef))
    _ = coef  # keep the Python object alive through the native call above
    var ip = _F32P(unsafe_from_address=labels_address(inter))
    _ = inter  # keep the Python object alive through the native call above
    for c in range(n_targets):  # driver-loop: over the fitted coefficient block (model structure)
        for j in range(cols):  # driver-loop: over the fitted coefficients of one target
            cp[c * cols + j] = source[c + n_targets * j]
            _ = coef  # keep the Python object alive through the native call above
            _ = w  # keep the Python object alive through the native call above
        if intercept:
            ip[c] = source[c + n_targets * cols]
            _ = inter  # keep the Python object alive through the native call above
            _ = w  # keep the Python object alive through the native call above
    py_setitem(fields, "coef_", coef)
    py_setitem(fields, "intercept_", inter)
    var iterations = labels_vector(1, "<i8", context)
    _I64P(unsafe_from_address=labels_address(iterations))[0] = Int64(_int(n_iter))
    _ = iterations  # keep the Python object alive through the native call above
    py_setitem(fields, "n_iter_", iterations)
    var words = _F32P(unsafe_from_address=labels_address(info))
    _ = info  # keep the Python object alive through the native call above
    py_setitem(fields, "objective_", PythonObject(Float64(words[0])))
    _ = info  # keep the Python object alive through the native call above
    py_setitem(fields, "retcode_", PythonObject(Int(words[1])))
    _ = info  # keep the Python object alive through the native call above
    py_setitem(fields, "n_features_in_", cols)
    _ = x^
    _ = y_enc^
    _ = sw^
    _ = info^
    _ = labels^
    return model


# ------------------------------------------------------------ inference


def _n_targets(model: PythonObject, fields: PythonObject) raises -> Int:
    """`_n_targets`: 1 for two classes, C for C > 2 (`len(self.classes_)`)."""
    var classes = py_get(fields, "classes_")
    if classes is None:
        raise Error("AttributeError: '" + _name(model) + "' object has no attribute 'classes_'")
    var k = len(classes)
    return 1 if k == 2 else k


def _fitted_w(model: PythonObject, fields: PythonObject) raises -> PythonObject:
    var w = py_get(fields, "_w")
    if w is None:
        raise Error("AttributeError: '" + _name(model) + "' object has no attribute '_w'")
    return w


def logistic_decision(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`decision_function`: (n,) float32 scores for two classes, (n, C)
    for more."""
    var fields = py_fields(model)
    if not py_contains(fields, "_w"):
        raise Error("mojolearn LogisticRegression: call fit first")
    var x = classical_host_f32(source, 2, "X", context)
    var rows = Int(py=x.shape[0])
    var cols = Int(py=x.shape[1])
    if Bool(py=PythonObject(cols) != fields["n_features_in_"]):
        raise Error("mojolearn LogisticRegression feature count differs from fit")
    var n_targets = _n_targets(model, fields)
    var entries = _entries(model, fields, context)
    var intercept = 1 if Bool(py=fields["fit_intercept"]) else 0
    var out = PythonObject(None)
    var params = Python.list(rows, cols, intercept)
    if n_targets == 1:
        out = labels_vector(rows, "<f4", context)
    else:
        out = labels_empty(Python.tuple(rows, n_targets), "<f4", context)
        _ = py_call(params.append, PythonObject(n_targets))
    _ = py_call_typed(_need(entries, "qn_decision_function"), classical_address(x),
                      classical_address(fields["_w"]), PythonObject(labels_address(out)), params)
    _ = x  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = x^
    return out


def logistic_predict(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`predict`: the device threshold (two classes) or row argmax (more)
    beside the scores, decoded to `classes_`."""
    var fields = py_fields(model)
    var n_targets = _n_targets(model, fields)
    if n_targets == 1:
        var x = classical_host_f32(source, 2, "X", context)
        var rows = Int(py=x.shape[0])
        var cols = Int(py=x.shape[1])
        var codes = labels_vector(rows, "<i8", context)
        var entries = _entries(model, fields, context)
        var native = py_get(entries, "qn_predict_binary")
        if native is None:
            raise Error("RuntimeError: mojolearn LogisticRegression: the estimators binding has no "
                        + "qn_predict_binary; rebuild it")
        var w = _fitted_w(model, fields)
        _ = py_call_typed(native, classical_address(x), classical_address(w),
                          PythonObject(labels_address(codes)),
                          Python.list(rows, cols, 1 if Bool(py=fields["fit_intercept"]) else 0))
        _ = w  # keep the Python object alive through the native call above
        _ = x  # keep the Python object alive through the native call above
        _ = codes  # keep the Python object alive through the native call above
        _ = x^
        return classical_decode_labels(fields["classes_"], codes, context)
    var entries = _entries(model, fields, context)
    var native = py_get(entries, "qn_predict_multiclass")
    if native is not None:
        if not py_contains(fields, "_w"):
            raise Error("mojolearn LogisticRegression: call fit first")
        var x = classical_host_f32(source, 2, "X", context)
        var rows = Int(py=x.shape[0])
        var cols = Int(py=x.shape[1])
        if Bool(py=PythonObject(cols) != fields["n_features_in_"]):
            raise Error("mojolearn LogisticRegression feature count differs from fit")
        var codes = labels_vector(rows, "<i8", context)
        if rows:
            _ = py_call_typed(native, classical_address(x), classical_address(fields["_w"]),
                              PythonObject(labels_address(codes)),
                              Python.list(rows, cols, 1 if Bool(py=fields["fit_intercept"]) else 0, n_targets))
            _ = x  # keep the Python object alive through the native call above
            _ = codes  # keep the Python object alive through the native call above
        _ = x^
        return classical_decode_labels(fields["classes_"], codes, context)
    # A binding without the device argmax keeps the host scan, the same
    # first-maximum rule (`_labels.argmax_rows`).
    var scores = logistic_decision(model, source, context)
    var rows = Int(py=scores.shape[0])
    var codes = labels_argmax_rows(scores, rows, n_targets, False, context)
    _ = scores^
    return classical_decode_labels(fields["classes_"], codes, context)


def logistic_proba(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`predict_proba`: the sigmoid (two classes) or softmax (more) of the
    decision scores, in float64."""
    var scores = logistic_decision(model, source, context)
    var fields = py_fields(model)
    var n_targets = _n_targets(model, fields)
    var entries = _entries(model, fields, context)
    var rows = Int(py=scores.shape[0])
    var out = PythonObject(None)
    if n_targets == 1:
        out = labels_empty(Python.tuple(rows, 2), "<f8", context)
        _ = py_call_typed(_need(entries, "qn_sigmoid"), PythonObject(labels_address(scores)), PythonObject(labels_address(out)),
                          Python.list(rows))
        _ = out  # keep the Python object alive through the native call above
        _ = scores  # keep the Python object alive through the native call above
    else:
        out = labels_empty(Python.tuple(rows, n_targets), "<f8", context)
        _ = py_call_typed(_need(entries, "qn_softmax"), PythonObject(labels_address(scores)), PythonObject(labels_address(out)),
                          Python.list(rows, n_targets))
        _ = out  # keep the Python object alive through the native call above
        _ = scores  # keep the Python object alive through the native call above
    _ = scores^
    return out


def logistic_log_proba(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`predict_log_proba`: `_log_or_inf` of every probability in the
    estimators binding (py2mojo_rows ROWS_LOG)."""
    var proba = logistic_proba(model, source, context)
    var entries = _entries(model, py_fields(model), context)
    var flags = py_get(entries, "py2mojo_linear_flags")
    var bits = Int(py=py_call_typed(flags)) if flags is not None else 0
    if (bits & PY2MOJO_ROWS) == 0:
        raise Error("RuntimeError: mojolearn: this estimators binding predates py2mojo_rows; "
                    + "rebuild it (pixi run build)")
    var shape = proba.shape
    var size = Int(py=proba.size)
    var out = labels_empty(shape, "<f8", context)
    if size:
        _ = py_call_typed(_need(entries, "py2mojo_rows"), PythonObject(ROWS_LOG), PythonObject(labels_address(proba)),
                          PythonObject(labels_address(out)), Python.list(size, 1))
        _ = proba  # keep the Python object alive through the native call above
        _ = out  # keep the Python object alive through the native call above
    _ = proba^
    return out


def logistic_score(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`score`: the mean accuracy of predict(X) against y, the prediction
    handed straight to the native accuracy program (`_accuracy_host`)."""
    var prediction = PythonObject(None)
    var host_predict = py_get(context, "lr_host_predict")
    if host_predict is not None:
        # The explicit host class predicts through its own native host
        # inference owner, as its `predict` does.
        prediction = py_call_typed(host_predict[0], model, PythonObject("host_infer_predict"), Python.tuple(args[0]),
                                   host_predict[1])
    else:
        prediction = logistic_predict(model, args[0], context)
    var y = args[1]
    var shape = labels_shape_of(y, context)
    # `len(pred)`: the Array's row count from its slots, or a list's length.
    var predicted = labels_shape_of(prediction, context)
    if len(shape) != 1 or shape[0] != predicted[0]:
        raise Error("mojolearn: y must be 1-D with one entry per row of X")
    return PythonObject(classical_accuracy(y, prediction, PythonObject(None), PythonObject(None), context))


def logistic_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var model = call[0]
        var operation = String(call[1])
        var args = call[2]
        var context = call[3]
        var hosts = py_get(context, "lr_host_contexts")
        if hosts is not None and len(hosts) > 0:
            var mro = py_type(model).__mro__
            for k in range(len(mro)):  # driver-loop: over the model's class hierarchy (explicit host classes)
                var host = py_get(hosts, mro[k])
                if host is not None:
                    context = host
                    break
        classical_connect_numpy(context)
        var result = PythonObject(None)
        if operation == "fit":
            result = logistic_fit(model, args, context)
        elif operation == "decision_function":
            result = logistic_decision(model, args[0], context)
        elif operation == "predict":
            result = logistic_predict(model, args[0], context)
        elif operation == "predict_proba":
            result = logistic_proba(model, args[0], context)
        elif operation == "predict_log_proba":
            result = logistic_log_proba(model, args[0], context)
        elif operation == "score":
            result = logistic_score(model, args, context)
        else:
            raise Error("unknown native LogisticRegression operation")
        return result.steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error^)
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if String(error).startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        if String(error).startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if String(error).startswith("ZeroDivisionError:"):
            return raise_python_exception(error_without_class(error, "ZeroDivisionError"), ExceptionType("PyExc_ZeroDivisionError"))
        if String(error).startswith("OverflowError:"):
            return raise_python_exception(error_without_class(error, "OverflowError"), ExceptionType("PyExc_OverflowError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
