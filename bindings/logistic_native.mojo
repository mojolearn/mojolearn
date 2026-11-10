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
from bindings.native_loaded_module import native_loaded_module, NativeModules
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.causal_exception import causal_set_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.py_typed_call import py_call_typed
from core.logistic_native_api import (
    QnFitRequest, QnQueryRequest, QnFoldStore, QN_NATIVE_DECISION, QN_NATIVE_BINARY,
    QN_NATIVE_MULTICLASS, QN_NATIVE_PROBA,
)
from core.logistic_config import LogisticPromotion
from core.logistic_model_state import LogisticFitLayout, logistic_target_count, logistic_fit_target_count
from bindings.logistic_state_boundary import logistic_parameters_pack
from core.linear_metadata import linear_mode_name
from bindings.logistic_config_boundary import (
    logistic_scalar_text, logistic_integer, logistic_validate_configuration,
    logistic_configuration_strengths, logistic_configuration_tolerance,
)
from bindings.logistic_descriptor_boundary import (
    logistic_compiled_mode, logistic_multiclass_available, logistic_run_fit,
    logistic_run_query, logistic_run_fold_fit, logistic_run_fold_query,
)
from bindings.classical_attributes import classical_connect_numpy, classical_getattribute
from bindings.classical_pandas import classical_admit_value
from bindings.classical_exceptions import classical_require_fitted, classical_not_fitted_class
from providers.python.linear_metadata_boundary import classical_is_fitted
from bindings.logistic_admission import logistic_storage, logistic_fields, logistic_initialize, logistic_get_params, logistic_set_params, logistic_fit_replacements_admit
from bindings.prep_lda_admission import lda_native_context, lda_native_labels
from bindings.classifier_public_lifecycle import classifier_after_fit, classifier_label_list
from bindings.scaler_lifecycle_native import _public_result
from bindings.classical_labels import _flatten_labels
from bindings.classical_arrays import classical_address
from bindings.admitted_input import admit_x, admit_x_unrecorded, admitted_finite_refuse
from bindings.device_rows_door import door_is_device_rows, door_rows_refuse
from bindings.classical_labels import (
    ClassicalLabels, classical_encode_labels, classical_decode_labels, classical_accuracy,
    labels_class_weight_rows, labels_argmax_rows, labels_shape_of,
    labels_address, labels_cast,
)
from bindings.admitted_input import AdmittedX, admit_weight_words, admitted_words, admitted_array, admitted_vector

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]

# linear_model._QN_LOSS_LOGISTIC / _QN_LOSS_SOFTMAX
comptime QN_LOSS_LOGISTIC = 0
comptime QN_LOSS_SOFTMAX = 2


def _builtins() raises -> PythonObject:
    return native_loaded_module("builtins")


def _repr(value: PythonObject) raises -> String:
    return logistic_scalar_text(value)


def _name(model: PythonObject) raises -> String:
    return String(py_type(model).__name__)


def _mode(fields: PythonObject, context: PythonObject) raises -> String:
    """`_backend.binding(name, self.numeric_mode)`'s mode: the instance's,
    or the process default."""
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not py_type(PythonObject("")):
        raise Error("numeric_mode must be fast, identical or None")
    return linear_mode_name(String(mode))


def _table(tables: PythonObject, mode: String) raises -> PythonObject:
    # A refusal recorded at import as "<Class>: <message>" is raised again.
    var table = tables[PythonObject(mode)]
    if py_type(table) is py_type(PythonObject("")):
        raise Error(String(table))
    return table


def _host_mode(model: PythonObject, fields: PythonObject) raises -> String:
    """`_HostBound._bind`'s mode: the explicit host classes run identical
    only."""
    var mode = py_get(fields, "numeric_mode")
    if mode is not None and (py_type(mode) is not py_type(PythonObject("")) or String(mode) != "identical"):
        raise Error("mojolearn: " + _name(model) + " runs IDENTICAL only on the host; this model was saved "
                    + _repr(mode))
    return String("identical")


def _entries(model: PythonObject, fields: PythonObject, context: PythonObject) raises -> PythonObject:
    """The estimators entries of the estimator's mode, checked against the
    mode the binary was compiled for (`_backend.binding`'s refusal)."""
    var host = py_get(context, "lr_host_model")
    var mode = _host_mode(model, fields) if host is not None else _mode(fields, context)
    var entries = _table(context["lr_bindings"], mode)
    # The boundary cache pins the extension and stores native mode metadata.
    var compiled = logistic_compiled_mode(entries)
    var expected = 1 if String(mode) == "identical" else 0
    if compiled != expected:
        var name = String("identical") if compiled == 1 else (String("fast") if compiled == 0 else String("unknown"))
        raise Error("RuntimeError: mojolearn: _mojolearn_estimators was compiled for " + name
                    + ", but this call requested " + String(mode) + "; rebuild the " + String(mode) + " binding")
    return entries


def _class_constant(model: PythonObject, fields: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`self.<name>` for LogisticRegression's class constants
    (`lbfgs_memory`, `penalty_normalized`): the instance's own value, else
    the class attribute."""
    var own = py_get(fields, PythonObject(name))
    if own is not None:
        return own
    var value = classical_getattribute(model, PythonObject(name), context)
    classical_admit_value(value, context)
    return value


def _row_weights(model: PythonObject, fields: PythonObject, sample_weight: PythonObject, class_weight: PythonObject,
                 labels: ClassicalLabels, n: Int, context: PythonObject) raises -> AdmittedX:
    """`linear_model._qn_row_weights`: no weights (address 0) for an
    unweighted fit, else the checked float32 sample weights times each row's
    class weight (their owner kept alive with the address)."""
    var b = _builtins()
    if sample_weight is None and class_weight is None:
        return admitted_words(PythonObject(None), 0, 0, 0)
    var sw = admit_weight_words(sample_weight, n, context)
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
    # The mode from the fit's admitted fields snapshot (numeric_mode is a
    # constructor parameter the fit already admitted), not a third admission.
    var xtrees = _table(context["lr_x_trees"], _host_mode(model, fields) if host is not None
                        else _mode(fields, context))
    var cw = labels_class_weight_rows(class_weight, labels, xtrees, context)
    if cw is None:  # every class weight is 1
        return sw^
    var out = admitted_vector(n, "<f4", context)
    var mul = py_get(xtrees, "x_trees_mul_f32")
    if mul is None:
        raise Error("ImportError: the x_trees binding has no x_trees_mul_f32")
    _ = py_call_typed(mul, PythonObject(sw.address), PythonObject(labels_address(cw)), PythonObject(labels_address(out)),
                      Python.list(n))
    _ = out  # keep the Python object alive through the native call above
    _ = cw  # keep the Python object alive through the native call above
    _ = sw^
    _ = cw^
    var _keep_ret_out = admitted_words(out, labels_address(out), n, 1)
    _ = out  # keep the Python object alive through the native call above
    return _keep_ret_out^


# ------------------------------------------------------------ fit


def _logistic_buffer_column(y: PythonObject, shape: List[Int], context: PythonObject) raises -> Bool:
    """True for an exact ndarray or native Array of shape (n, 1): the public
    fit door keeps such a column (after its column warning) instead of
    flattening it into a Python list, and the label encoder reads it in C
    order, which for one column is the flattened order (perf3-label-doors)."""
    if len(shape) != 2 or shape[1] != 1:
        return False
    var cls = py_type(y)
    var ndarray = py_get(context, "ndarray")
    if ndarray is not None and cls is ndarray:
        return True
    var array_class = py_get(context, "array_class")
    return array_class is not None and cls is array_class


def logistic_fit(model: PythonObject, source_x: PythonObject, source_y: PythonObject,
                 sample_weight: PythonObject, context: PythonObject, fields: PythonObject) raises -> PythonObject:
    # `fields`: the admitted live storage (`logistic_fields(model, context,
    # False)`), taken once by the public door for this call.
    logistic_fit_replacements_admit(fields, context)
    var penalty = logistic_validate_configuration(fields)
    var promotion = LogisticPromotion()
    # Lane a7c: the finite test of a borrowed X is the qn_fit entry's (its
    # 17th field, the status word); a cast X is tested in its cast pass.
    # Lane a7c: a cross-validation fold (DeviceRows) of an identical-mode
    # fit is gathered on the device out of model selection's resident base
    # by the base binding's `qn_fit_fold` (no host gather, no upload of X);
    # without it (the CPU column, FAST) the door gets the host rows.
    var fold_entry = py_get(context, "logistic_fold_native_api_v1")
    var store = QnFoldStore()
    var x = PythonObject(None)
    var rows = 0
    var cols = 0
    if (fold_entry is not None and _mode(fields, context) == "identical"
            and door_is_device_rows(source_x, context)):
        var fx = admit_x_unrecorded(source_x, _name(model), context, device_x=True)
        rows = fx.rows
        cols = fx.cols
        if fx.store.active():
            store = QnFoldStore(fx.store.base_id, fx.store.base_rows, fx.store.index_id)
        else:
            x = fx.array
        py_setitem(fields, "input_copied_", PythonObject(True))
    else:
        x = admit_x(source_x, 2, "X", context, defer_finite=True)
        py_setitem(fields, "input_copied_", PythonObject(not (x is source_x)))
        rows = Int(py=x.shape[0])
        cols = Int(py=x.shape[1])
    var y = source_y
    var y_shape = labels_shape_of(y, context)
    if len(y_shape) != 1 and not _logistic_buffer_column(y, y_shape, context):
        raise Error("mojolearn LogisticRegression requires a 1-D y")
    var labels = classical_encode_labels(y, context)
    py_setitem(fields, "classes_", labels.classes)
    if labels.n != rows:
        raise Error("mojolearn LogisticRegression X and y lengths differ")
    var n_classes = len(labels.classes)
    var n_targets = logistic_fit_target_count(n_classes)
    var loss = QN_LOSS_LOGISTIC if n_targets == 1 else QN_LOSS_SOFTMAX
    # The dense codes as float32 (codes < 2**24: exact), `codes.astype('<f4')`.
    var y_enc = admitted_vector(rows, "<f4", context)
    # One native cast of the int32 codes (exact below 2**24), as
    # LogisticRegressionCV does; no host loop over rows in this owner.
    labels_cast(labels_address(labels.codes), "<i4", labels_address(y_enc), "<f4", rows, context)
    _ = y_enc  # keep the Python object alive through the native call above
    var strengths = logistic_configuration_strengths(fields, penalty, promotion, context)
    var l1 = strengths[0]
    var l2 = strengths[1]
    if n_targets > 1 and l1 != 0.0:
        raise Error("NotImplementedError: mojolearn LogisticRegression: penalty=" + _repr(fields["penalty"]) + " with "
                    + String(n_classes) + " classes selects OWL-QN on the softmax objective, "
                    + "which runs but has no identity gate (glm/checks/"
                    + "multinomial_check.mojo gates l2 and no penalty); refused by "
                    + "name until it is gated (CONTRIBUTING.md (Non-default paths))")
    var sw = _row_weights(model, fields, sample_weight, fields["class_weight"], labels, rows, context)
    var entries = _entries(model, fields, context)
    var intercept = Bool(py=fields["fit_intercept"])
    var layout = LogisticFitLayout(cols, n_targets, intercept)
    var w = admitted_vector(layout.weights, "<f4", context)
    var info = admitted_vector(2, "<f4", context)
    var tol = fields["tol"]
    var has_sw = sw.address != 0
    _ = sw  # keep the Python object alive through the native call above
    var status = admitted_vector(1, "<i4", context)
    var tolerances = logistic_configuration_tolerance(tol, promotion, context)
    var request = QnFitRequest(0 if store.active() else Int(py=classical_address(x)),
        labels_address(y_enc), labels_address(w), labels_address(info), sw.address,
        labels_address(status), rows, cols, n_classes,
        logistic_integer(fields["max_iter"]), logistic_integer(fields["linesearch_max_iter"]),
        logistic_integer(_class_constant(model, fields, "lbfgs_memory", context)), intercept,
        Bool(py=_class_constant(model, fields, "penalty_normalized", context)), loss,
        l1, l2, tolerances[0], tolerances[1], 0.0)
    var n_iter = 0
    if store.active():
        var result = logistic_run_fold_fit(fold_entry, store, request)
        door_rows_refuse(result.finite, result.range_error, _name(model), store.base_rows)
        n_iter = result.iterations
    else:
        n_iter = logistic_run_fit(entries, request)
        admitted_finite_refuse(Int(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=labels_address(status))[0]),
                               _name(model))
    _ = status  # keep the Python object alive through the native call above
    _ = y_enc  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _ = info  # keep the Python object alive through the native call above
    _ = w  # keep the Python object alive through the native call above
    _ = sw  # keep the Python object alive through the native call above
    # Keep the new weight buffer private while native PACK releases the GIL.
    # Preserve the old failed-allocation state (_w was published before the
    # public coefficient allocations) without exposing an in-progress copy.
    var public_parameters: Tuple[PythonObject, PythonObject]
    try:
        public_parameters = logistic_parameters_pack(w, layout, context)
    except error:
        py_setitem(fields, "_w", w)
        raise error^
    py_setitem(fields, "_w", w)
    var coef = public_parameters[0]
    var inter = public_parameters[1]
    py_setitem(fields, "coef_", coef)
    py_setitem(fields, "intercept_", inter)
    var iterations = admitted_vector(1, "<i8", context)
    _I64P(unsafe_from_address=labels_address(iterations))[0] = Int64(n_iter)
    _ = iterations  # keep the Python object alive through the native call above
    _ = info  # keep the Python object alive through the native call above
    _ = w  # keep the Python object alive through the native call above
    _ = y_enc  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
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
    return logistic_target_count(k)


def _fitted_w(model: PythonObject, fields: PythonObject) raises -> PythonObject:
    var w = py_get(fields, "_w")
    if w is None:
        raise Error("AttributeError: '" + _name(model) + "' object has no attribute '_w'")
    return w


# `qn_infer_fold` modes (glm/qn_fold.mojo QN_FOLD_*).
comptime _FOLD_DECISION = 0
comptime _FOLD_CODES = 1
comptime _FOLD_PROBA = 2


struct _QueryX(Movable):
    """The X of one inference call (lane a8c-device-x-linear-prep-rfe):
    either host words for the estimators entries (`x`, tested by the entry
    over its own upload through the status word `status`, or already tested
    when admit_x cast it) or, for an identical-mode cross-validation fold,
    the fold's typed resident store ids (base id, base rows, index
    id) for the base binding's `qn_infer_fold`: no host gather, no upload."""
    var x: PythonObject
    var rows: Int
    var cols: Int
    var store: QnFoldStore
    var status: PythonObject

    def __init__(out self, x: PythonObject, rows: Int, cols: Int, store: QnFoldStore, status: PythonObject):
        self.x = x
        self.rows = rows
        self.cols = cols
        self.store = store.copy()
        self.status = status

    def fold(self) raises -> Bool:
        return self.store.active()

    def status_address(self) raises -> Int:
        return labels_address(self.status)

    def refuse(self, model: PythonObject) raises:
        """The deferred finite refusal of a host X (scikit-learn's wording)."""
        admitted_finite_refuse(Int(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=labels_address(self.status))[0]),
                               _name(model))


# TOMBSTONE: A8C_LOGISTIC_QUERY_HOST_SCAN (removed: superseded by the entries' device test of their upload) deleted 2026-11-09 by fb8823ee0a; code recoverable at fb8823ee0a^.
# Restore: git apply experiments/removed/A8C_LOGISTIC_QUERY_HOST_SCAN.patch
def _query_x(model: PythonObject, fields: PythonObject, source: PythonObject, context: PythonObject) raises -> _QueryX:
    """X for decision_function / predict / predict_proba. The door no longer
    scans a borrowed X on the host (lane a8c): `defer_finite`, and every
    estimators entry tests the words it uploads (glm/qn_infer.mojo
    qn_x_status; the host column's parallel scan) and returns the status
    word. A DeviceRows fold of an identical-mode GPU model is read on the
    device (`qn_infer_fold`); without it (the CPU column, FAST, a host
    class) the fold's rows are gathered on the host and tested there."""
    var fold_entry = py_get(context, "logistic_fold_native_api_v1")
    var status = admitted_vector(1, "<i4", context)
    if (fold_entry is not None and py_get(context, "lr_host_model") is None
            and _mode(fields, context) == "identical"
            and door_is_device_rows(source, context)):
        var fx = admit_x_unrecorded(source, _name(model), context, device_x=True)
        if fx.store.active():
            return _QueryX(PythonObject(None), fx.rows, fx.cols,
                           QnFoldStore(fx.store.base_id, fx.store.base_rows, fx.store.index_id), status)
        return _QueryX(fx.array, fx.rows, fx.cols, QnFoldStore(), status)
    var x = admit_x(source, 2, "X", context, defer_finite=True)
    return _QueryX(x, Int(py=x.shape[0]), Int(py=x.shape[1]), QnFoldStore(), status)


def _infer_fold(model: PythonObject, fields: PythonObject, q: _QueryX, dst: PythonObject, mode: Int,
                n_classes: Int, log_out: Bool, context: PythonObject) raises:
    """One `qn_infer_fold` call over the fold `q` names into `dst`, refused
    through door_rows_refuse on its words."""
    var operation = QN_NATIVE_DECISION if mode == _FOLD_DECISION else (QN_NATIVE_BINARY if mode == _FOLD_CODES else QN_NATIVE_PROBA)
    var request = QnQueryRequest(0, Int(py=classical_address(fields["_w"])),
        labels_address(dst), 0, q.rows, q.cols, n_classes, Bool(py=fields["fit_intercept"]), log_out)
    var result = logistic_run_fold_query(context["logistic_fold_native_api_v1"], operation, q.store, request)
    _ = dst
    door_rows_refuse(result.finite, result.range_error, _name(model), q.store.base_rows)


def logistic_decision(model: PythonObject, fields: PythonObject, mut entries: PythonObject, source: PythonObject,
                      context: PythonObject) raises -> PythonObject:
    """`decision_function`: (n,) float32 scores for two classes, (n, C)
    for more. `fields` is the admitted `logistic_fields` snapshot the public
    entry took once for this call; `entries` is the mode's estimators table
    when the caller already has it (None: resolved here, at the same point
    as before, and handed back for the caller's next step)."""
    if not py_contains(fields, "_w"):
        raise Error("mojolearn LogisticRegression: call fit first")
    var q = _query_x(model, fields, source, context)
    var rows = q.rows
    var cols = q.cols
    if Bool(py=PythonObject(cols) != fields["n_features_in_"]):
        raise Error("mojolearn LogisticRegression feature count differs from fit")
    var n_targets = _n_targets(model, fields)
    if entries is None:
        entries = _entries(model, fields, context)
    var intercept = 1 if Bool(py=fields["fit_intercept"]) else 0
    var out = PythonObject(None)
    if n_targets == 1:
        out = admitted_vector(rows, "<f4", context)
    else:
        out = admitted_array(Python.tuple(rows, n_targets), "<f4", context)
    if q.fold():
        _infer_fold(model, fields, q, out, _FOLD_DECISION, n_targets if n_targets > 1 else 1, False, context)
        return out
    # params: n_rows, n_features, fit_intercept, n_classes (1: the binary
    # shape), the status word's address (lane a8c)
    var request = QnQueryRequest(Int(py=classical_address(q.x)), Int(py=classical_address(fields["_w"])),
        labels_address(out), q.status_address(), rows, cols, n_targets, intercept != 0, False)
    logistic_run_query(entries, QN_NATIVE_DECISION, request)
    _ = q.x  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    q.refuse(model)
    return out


def logistic_predict(model: PythonObject, fields: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`predict`: the device threshold (two classes) or row argmax (more)
    beside the scores, decoded to `classes_`."""
    var n_targets = _n_targets(model, fields)
    if n_targets == 1:
        var q = _query_x(model, fields, source, context)
        var rows = q.rows
        var cols = q.cols
        if cols != Int(py=fields["n_features_in_"]):
            raise Error("mojolearn LogisticRegression feature count differs from fit")
        var codes = admitted_vector(rows, "<i8", context)
        var entries = _entries(model, fields, context)
        var w = _fitted_w(model, fields)
        if q.fold():
            _infer_fold(model, fields, q, codes, _FOLD_CODES, 1, False, context)
            return classical_decode_labels(fields["classes_"], codes, context)
        var request = QnQueryRequest(Int(py=classical_address(q.x)), Int(py=classical_address(w)),
            labels_address(codes), q.status_address(), rows, cols, 1, Bool(py=fields["fit_intercept"]), False)
        logistic_run_query(entries, QN_NATIVE_BINARY, request)
        _ = w  # keep the Python object alive through the native call above
        _ = q.x  # keep the Python object alive through the native call above
        _ = codes  # keep the Python object alive through the native call above
        q.refuse(model)
        return classical_decode_labels(fields["classes_"], codes, context)
    var entries = _entries(model, fields, context)
    if logistic_multiclass_available(entries):
        if not py_contains(fields, "_w"):
            raise Error("mojolearn LogisticRegression: call fit first")
        var q = _query_x(model, fields, source, context)
        var rows = q.rows
        var cols = q.cols
        if Bool(py=PythonObject(cols) != fields["n_features_in_"]):
            raise Error("mojolearn LogisticRegression feature count differs from fit")
        var codes = admitted_vector(rows, "<i8", context)
        if q.fold():
            _infer_fold(model, fields, q, codes, _FOLD_CODES, n_targets, False, context)
            return classical_decode_labels(fields["classes_"], codes, context)
        if rows:
            var request = QnQueryRequest(Int(py=classical_address(q.x)), Int(py=classical_address(fields["_w"])),
                labels_address(codes), q.status_address(), rows, cols, n_targets, Bool(py=fields["fit_intercept"]), False)
            logistic_run_query(entries, QN_NATIVE_MULTICLASS, request)
            _ = q.x  # keep the Python object alive through the native call above
            _ = codes  # keep the Python object alive through the native call above
            q.refuse(model)
        return classical_decode_labels(fields["classes_"], codes, context)
    # A binding without the device argmax keeps the host scan, the same
    # first-maximum rule (`_labels.argmax_rows`).
    var scores = logistic_decision(model, fields, entries, source, context)
    var rows = Int(py=scores.shape[0])
    var codes = labels_argmax_rows(scores, rows, n_targets, False, context)
    _ = scores^
    return classical_decode_labels(fields["classes_"], codes, context)


def _logistic_proba_fused(model: PythonObject, fields: PythonObject, mut entries: PythonObject,
                          source: PythonObject, context: PythonObject, log_out: Bool = False) raises -> PythonObject:
    """`predict_proba` through the binding's one-call `qn_proba` (decision
    and link in one crossing, no Python scores array between them), with
    `logistic_decision`'s checks in its order. None when the connected table
    has no `qn_proba` (the two-entry path then runs)."""
    if not py_contains(fields, "_w"):
        raise Error("mojolearn LogisticRegression: call fit first")
    var n_targets = _n_targets(model, fields)
    if entries is None:
        entries = _entries(model, fields, context)
    var q = _query_x(model, fields, source, context)
    var rows = q.rows
    var cols = q.cols
    if Bool(py=PythonObject(cols) != fields["n_features_in_"]):
        raise Error("mojolearn LogisticRegression feature count differs from fit")
    var intercept = 1 if Bool(py=fields["fit_intercept"]) else 0
    var out = admitted_array(Python.tuple(rows, 2 if n_targets == 1 else n_targets), "<f8", context)
    if q.fold():
        _infer_fold(model, fields, q, out, _FOLD_PROBA, n_targets if n_targets > 1 else 1, log_out, context)
        return out
    # ne2-g-linear (N361): the 5th word asks the entry for log(p) in the
    # same device session (py2mojo_rows ROWS_LOG's words); the 6th is the
    # status word's address (lane a8c).
    var request = QnQueryRequest(Int(py=classical_address(q.x)), Int(py=classical_address(fields["_w"])),
        labels_address(out), q.status_address(), rows, cols, n_targets, intercept != 0, log_out)
    logistic_run_query(entries, QN_NATIVE_PROBA, request)
    _ = q.x  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    q.refuse(model)
    return out


def logistic_proba(model: PythonObject, fields: PythonObject, mut entries: PythonObject, source: PythonObject,
                   context: PythonObject) raises -> PythonObject:
    """`predict_proba`: the sigmoid (two classes) or softmax (more) of the
    decision scores, in float64. One fields snapshot and one estimators
    table serve the decision and the link."""
    return _logistic_proba_fused(model, fields, entries, source, context)


def logistic_log_proba(model: PythonObject, fields: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`predict_log_proba`: `_log_or_inf` of every probability in the
    estimators binding (py2mojo_rows ROWS_LOG)."""
    if not py_contains(fields, "_w"):
        raise Error("mojolearn LogisticRegression: call fit first")
    var entries = _entries(model, fields, context)
    return _logistic_proba_fused(model, fields, entries, source, context, True)


def logistic_score(model: PythonObject, fields: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
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
        prediction = logistic_predict(model, fields, args[0], context)
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
        if py_type(py_type(model)) is not py_type(py_type(PythonObject(0))):
            raise Error("TypeError: LogisticRegression cannot invoke Python metaclass hooks")
        var hosts = py_get(context, "lr_host_contexts")
        if hosts is not None and len(hosts) > 0:
            var mro = py_type(model).__mro__
            for k in range(len(mro)):  # driver-loop: over the model's class hierarchy (explicit host classes)
                var host = py_get(hosts, mro[k])
                if host is not None:
                    context = host
                    break
        classical_connect_numpy(context)
        context = lda_native_context(context)
        var result = PythonObject(None)
        if operation == "initialize":
            result = logistic_initialize(model, args, context)
            return result.steal_data()
        # get_params/set_params/is_fitted validate nothing beyond admission
        # (scikit-learn's convention); consuming operations check values.
        var configuring = operation == "get_params" or operation == "set_params" or operation == "is_fitted"
        # fit reads the live storage (no dict or class-label copy): the same
        # admission, in the same order, as the snapshot the other operations
        # take; logistic_fit then uses it instead of admitting again.
        var fields = logistic_fields(model, context, operation != "fit", configuring)
        if operation == "get_params":
            # `logistic_get_params` from this call's snapshot (the same
            # `logistic_fields(model, context, True, True)` it would take).
            var params = Python.dict()
            var names = context["lr_sorted_parameters"]
            for i in range(len(names)):  # driver-loop: ParamsMixin's sorted parameter names
                py_setitem(params, names[i], fields[names[i]])
            result = params
            return result.steal_data()
        if operation == "set_params":
            result = logistic_set_params(model, args[0], context)
            return result.steal_data()
        if operation == "is_fitted":
            result = classical_is_fitted(model)
            return result.steal_data()
        if operation != "fit":
            classical_require_fitted(model, False, operation, context)
        # One module handle for this call's input admissions.
        var mods = NativeModules()
        classical_admit_value(mods, args[0], context)
        if operation == "fit":
            var target = classifier_label_list(args[1], context)
            lda_native_labels(target, context, True, "LogisticRegression")
            var shape = labels_shape_of(target, context)
            # perf3-label-doors: an exact ndarray / Array (n, 1) column stays
            # a column; the label encoder reads it natively in C order (the
            # flatten order), with the classes and codes of the flattened
            # list. Other columns (nested lists, torch) are still flattened.
            if len(shape) == 2 and shape[1] == 1 and not _logistic_buffer_column(target, shape, context):
                target = _flatten_labels(target, context)
            classical_admit_value(mods, args[2], context)
            result = logistic_fit(model, args[0], target, args[2], context, fields)
            classifier_after_fit(logistic_storage(model, context), args[0], context)
        # `fields` (above) is the one admitted snapshot of this call: the
        # operations below read it instead of re-admitting every parameter,
        # the fitted state and the class labels again (two to four times per
        # predict_proba / predict_log_proba before).
        elif operation == "decision_function":
            var entries = PythonObject(None)
            result = logistic_decision(model, fields, entries, args[0], context)
        elif operation == "predict":
            result = logistic_predict(model, fields, args[0], context)
        elif operation == "predict_proba":
            var entries = PythonObject(None)
            result = logistic_proba(model, fields, entries, args[0], context)
        elif operation == "predict_log_proba":
            result = logistic_log_proba(model, fields, args[0], context)
        elif operation == "score":
            var target = classifier_label_list(args[1], context)
            lda_native_labels(target, context, False, "LogisticRegression")
            result = logistic_score(model, fields, Python.tuple(args[0], target), context)
        else:
            raise Error("unknown native LogisticRegression operation")
        result = _public_result(result, context)
        return result.steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error^)
        if String(error).startswith("NotFittedError:"):
            try:
                var call = PythonObject(from_borrowed=args_ptr)
                var context = call[3]
                var cls = classical_not_fitted_class(context)
                return causal_set_exception(error_without_class(error, "NotFittedError"), cls._obj_ptr)
            except nested:
                return raise_python_exception(nested)
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
