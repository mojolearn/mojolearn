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
# (bindings/classical_labels_door.mojo `classical_accuracy`).
#
# UNPACK / RUN / PACK of this door. UNPACK is the admission in this file and
# its `*_unpack` leaves (X, the label codes, the weights, the configuration
# words) and ends in a typed QnFitRequest / QnQueryRequest. RUN is
# core/logistic_route.mojo `logistic_fit_run` / `logistic_query_run` over
# that request, through the estimator's route, with the GIL released
# (providers/python/logistic_route_boundary_door.mojo); it holds no Python
# object. PACK is `logistic_fit_pack` (bindings/logistic_state_boundary_unpack.mojo)
# and the label decode of predict.
#
# The device. `device=None | "gpu" | "cpu"` is read once into the estimator's
# route owner; every operation below calls through `routed.route`. No
# operation tests the device, builds a mode word or reads an entry table.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.optimizer_public_words_unpack import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.causal_exception_unpack import causal_set_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.host_logistic_query_boundary_unpack import host_logistic_score_predict
from core.logistic_native_api import (
    QnFitRequest, QnQueryRequest, QnFoldStore, QN_NATIVE_DECISION, QN_NATIVE_BINARY,
    QN_NATIVE_MULTICLASS, QN_NATIVE_PROBA,
)
from core.logistic_config import LogisticPromotion
from core.logistic_model_state import LogisticFitLayout, logistic_target_count, logistic_fit_target_count
from bindings.logistic_state_boundary_unpack import LogisticQueryOwners, logistic_fit_pack
from bindings.logistic_query_boundary_unpack import logistic_query_width_check
from core.logistic_query_state import logistic_query_integer_width
from core.label_weights_native_api import LabelWeightsRequest, LABEL_WEIGHTS_MULTIPLY
from bindings.label_weights_boundary_unpack import label_weights_run
from core.linear_metadata import linear_vendor_name
from bindings.logistic_config_boundary_unpack import (
    logistic_scalar_text, logistic_integer, logistic_validate_configuration,
    logistic_configuration_strengths, logistic_configuration_tolerance,
)
from bindings.logistic_descriptor_boundary_door import logistic_run_fold_fit, logistic_run_fold_query
from providers.python.logistic_route_boundary_door import (
    LogisticRouteCall, logistic_route_held, logistic_route_context, logistic_route_current, logistic_route_resolve,
    logistic_route_of, logistic_route_xtrees, logistic_route_fit, logistic_route_query,
    logistic_device_admit,
)
from providers.python.device_route_boundary_door import device_route_clear
from bindings.classical_attributes_door import classical_connect_numpy, classical_getattribute
from bindings.classical_pandas_door import classical_admit_value
from bindings.classical_exceptions_door import classical_require_fitted, classical_not_fitted_class
from providers.python.linear_metadata_boundary_door import classical_is_fitted, linear_fitted_unpack
from bindings.logistic_admission_unpack import logistic_storage, logistic_fields, logistic_fit_replacements_admit
from bindings.logistic_admission_pack import logistic_initialize, logistic_get_params, logistic_set_params
from bindings.prep_lda_admission_door import lda_native_context, lda_native_labels
from bindings.classifier_public_lifecycle_door import classifier_after_fit, classifier_label_list
from bindings.scaler_lifecycle_native_door import _public_result
from bindings.classical_labels_door import _flatten_labels
from bindings.labels_descriptor_boundary_door import labels_scope
from bindings.classical_arrays_door import classical_address
from bindings.admitted_input_unpack import admit_x, admit_x_unrecorded, admitted_finite_refuse
from bindings.device_rows_door import door_is_device_rows, door_rows_refuse
from bindings.classical_labels_door import (
    ClassicalLabels, classical_encode_labels, classical_decode_labels, classical_accuracy,
    labels_class_weight_rows, labels_argmax_rows, labels_shape_of,
    labels_address, labels_cast,
)
from bindings.admitted_input_unpack import AdmittedX, admit_weight_words, admitted_words, admitted_storage_native
from bindings.logistic_keys_unpack import (
    logistic_keys, LOGISTIC_OPN_INITIALIZE, LOGISTIC_OPN_GET_PARAMS, LOGISTIC_OPN_SET_PARAMS, LOGISTIC_OPN_IS_FITTED,
    LOGISTIC_OPN_FIT, LOGISTIC_OPN_DECISION_FUNCTION, LOGISTIC_OPN_PREDICT, LOGISTIC_OPN_PREDICT_PROBA,
    LOGISTIC_OPN_PREDICT_LOG_PROBA, LOGISTIC_OPN_SCORE, LOGISTIC_OPN_VENDOR_USED, LOGISTIC_OPN_NUMERIC_MODE_USED,
)
from bindings.py_keys import py_op_code
from bindings.py_balanced import py_item

# linear_model._QN_LOSS_LOGISTIC / _QN_LOSS_SOFTMAX
comptime QN_LOSS_LOGISTIC = 0
comptime QN_LOSS_SOFTMAX = 2


def _builtins() raises -> PythonObject:
    return native_loaded_module("builtins")


def _repr(value: PythonObject) raises -> String:
    return logistic_scalar_text(value)


def _name(model: PythonObject) raises -> String:
    return String(py_type(model).__name__)


# TOMBSTONE: the per-call entry selection of this door (`_mode`, `_mode_value`, `_table`, `_host_mode`,
# `_host_mode_value`, `_entries`, `_entries_value`: per public call the numeric-mode word build and its
# normalisation, the `lr_bindings` / `lr_x_trees` reads by mode word, the explicit-host-class question, the
# provider admission and connection scan, the compiled-tier compare; replaced by the estimator's route owner,
# resolved once, providers/python/logistic_route_boundary_door.mojo `logistic_route_resolve`) deleted 2026-11-20 by
# lane dev2-logistic-device; code recoverable at 5c252f1573.


def _route(model: PythonObject, fields: PythonObject, held: PythonObject, base: PythonObject,
           context: PythonObject) raises -> LogisticRouteCall:
    """The estimator's route for this operation. A hit (`held` is the owner
    resolved for this connection context, class and numeric mode) is two
    pointer compares and a copy of the route's words. A miss (first
    operation, a changed numeric mode, an instance built without its
    constructor) resolves and stores it; the builtins module is read on a
    miss only."""
    if logistic_route_current(held, fields, context):
        return logistic_route_of(held)
    var fresh = logistic_route_resolve(model, fields, base, context, _builtins())
    return logistic_route_of(fresh)


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


def _row_weights(model: PythonObject, sample_weight: PythonObject, class_weight: PythonObject,
                 labels: ClassicalLabels, n: Int, route_owner: PythonObject, context: PythonObject) raises -> AdmittedX:
    """`linear_model._qn_row_weights`: no weights (address 0) for an
    unweighted fit, else the checked float32 sample weights times each row's
    class weight (their owner kept alive with the address). The class-weight
    programs run on the table of the route's device (`route_owner`)."""
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
    # The class-weight table the route resolved (a table that is not
    # connected is refused here, in its recorded words).
    var xtrees = logistic_route_xtrees(route_owner)
    # The label scope of the estimator's device (labels_scope), once per fit.
    var cw = labels_class_weight_rows(class_weight, labels, xtrees, labels_scope(model, context))
    if cw is None:  # every class weight is 1
        return sw^
    var out = admitted_storage_native([n], "<f4", context, as_array=True)
    _ = label_weights_run(xtrees, LABEL_WEIGHTS_MULTIPLY,
        LabelWeightsRequest(sw.address, labels_address(cw), labels_address(out), n, 0))
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
    var nk = logistic_keys()
    var ndarray = py_get(context, nk[].k_ndarray)
    if ndarray is not None and cls is ndarray:
        return True
    var array_class = py_get(context, nk[].k_array_class)
    return array_class is not None and cls is array_class


def logistic_fit(model: PythonObject, source_x: PythonObject, source_y: PythonObject,
                 sample_weight: PythonObject, context: PythonObject, fields: PythonObject,
                 routed: LogisticRouteCall) raises -> PythonObject:
    # The public door admitted this live field storage once for fit.
    logistic_fit_replacements_admit(fields, context)
    var penalty = logistic_validate_configuration(fields)
    var promotion = LogisticPromotion()
    # Lane a7c: the finite test of a borrowed X is the qn_fit entry's (its
    # 17th field, the status word); a cast X is tested in its cast pass.
    # Lane a7c: a cross-validation fold (DeviceRows) of an identical-mode
    # fit is gathered on the device out of model selection's resident base
    # by the base binding's `qn_fit_fold` (no host gather, no upload of X);
    # without it (the CPU column, FAST, a `device="cpu"` estimator, a host
    # class) the door gets the host rows. `routed.route.fold` is that
    # decision, made once when the route was resolved.
    var nk = logistic_keys()
    var fold_entry = PythonObject(None)
    var store = QnFoldStore()
    var x = PythonObject(None)
    var rows = 0
    var cols = 0
    if routed.route.fold and door_is_device_rows(source_x, context):
        fold_entry = py_item(context, nk[].k_logistic_fold_native_api_v1)
        var fx = admit_x_unrecorded(source_x, _name(model), context, device_x=True)
        rows = fx.rows
        cols = fx.cols
        if fx.store.active():
            store = QnFoldStore(fx.store.base_id, fx.store.base_rows, fx.store.index_id)
        else:
            x = fx.array
        py_setitem(fields, nk[].k_input_copied_, PythonObject(True))
    else:
        x = admit_x(source_x, 2, "X", context, defer_finite=True)
        py_setitem(fields, nk[].k_input_copied_, PythonObject(not (x is source_x)))
        rows = Int(py=x.shape[0])
        cols = Int(py=x.shape[1])
    var y = source_y
    var y_shape = labels_shape_of(y, context)
    if len(y_shape) != 1 and not _logistic_buffer_column(y, y_shape, context):
        raise Error("mojolearn LogisticRegression requires a 1-D y")
    var labels = classical_encode_labels(y, labels_scope(model, context))
    py_setitem(fields, nk[].k_classes_, labels.classes)
    if labels.n != rows:
        raise Error("mojolearn LogisticRegression X and y lengths differ")
    var n_classes = len(labels.classes)
    var n_targets = logistic_fit_target_count(n_classes)
    var loss = QN_LOSS_LOGISTIC if n_targets == 1 else QN_LOSS_SOFTMAX
    # The dense codes as float32 (codes < 2**24: exact), `codes.astype('<f4')`.
    var y_enc = admitted_storage_native([rows], "<f4", context, as_array=True)
    # One native cast of the int32 codes (exact below 2**24), as
    # LogisticRegressionCV does; no host loop over rows in this owner.
    labels_cast(labels_address(labels.codes), "<i4", labels_address(y_enc), "<f4", rows, context)
    _ = y_enc  # keep the Python object alive through the native call above
    var strengths = logistic_configuration_strengths(fields, penalty, promotion, context)
    var l1 = strengths[0]
    var l2 = strengths[1]
    if n_targets > 1 and l1 != 0.0:
        raise Error("NotImplementedError: mojolearn LogisticRegression: penalty=" + _repr(py_item(fields, nk[].k_penalty)) + " with "
                    + String(n_classes) + " classes selects OWL-QN on the softmax objective, "
                    + "which runs but has no identity gate (glm/checks/"
                    + "multinomial_check.mojo gates l2 and no penalty); refused by "
                    + "name until it is gated (CONTRIBUTING.md (Non-default paths))")
    var sw = _row_weights(model, sample_weight, py_item(fields, nk[].k_class_weight), labels, rows, routed.owner, context)
    var intercept = Bool(py=py_item(fields, nk[].k_fit_intercept))
    var layout = LogisticFitLayout(cols, n_targets, intercept)
    var w = admitted_storage_native([layout.weights], "<f4", context, as_array=True)
    # Private solver outputs are native owners; only the final scalar values
    # are packed into the model's public objective_/retcode_ fields.
    var info = List[Float32](length=2, fill=Float32(0))
    var tol = py_item(fields, nk[].k_tol)
    _ = sw  # keep the Python object alive through the native call above
    var status = List[Int32](length=1, fill=Int32(0))
    var tolerances = logistic_configuration_tolerance(tol, promotion, context)
    var request = QnFitRequest(0 if store.active() else Int(py=classical_address(x)),
        labels_address(y_enc), labels_address(w), Int(info.unsafe_ptr()), sw.address,
        Int(status.unsafe_ptr()), rows, cols, n_classes,
        logistic_integer(py_item(fields, nk[].k_max_iter)), logistic_integer(py_item(fields, nk[].k_linesearch_max_iter)),
        logistic_integer(_class_constant(model, fields, "lbfgs_memory", context)), intercept,
        Bool(py=_class_constant(model, fields, "penalty_normalized", context)), loss,
        l1, l2, tolerances[0], tolerances[1], 0.0)
    var n_iter = 0
    if store.active():
        var result = logistic_run_fold_fit(fold_entry, store, request)
        door_rows_refuse(result.finite, result.range_error, _name(model), store.base_rows)
        n_iter = result.iterations
    else:
        # RUN: the solver of the route's image, GIL released.
        n_iter = logistic_route_fit(routed, request)
        admitted_finite_refuse(Int(status[0]), _name(model))
    _ = status^  # native owner survives the synchronous fit and status read
    _ = y_enc  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _ = w  # keep the Python object alive through the native call above
    _ = sw  # keep the Python object alive through the native call above
    # PACK: the fitted attributes on the admitted field storage.
    logistic_fit_pack(fields, w, layout, n_iter, info[0], info[1], context)
    _ = w  # keep the Python object alive through the native call above
    _ = y_enc  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _ = x^
    _ = y_enc^
    _ = sw^
    _ = info^
    _ = labels^
    return model


# ------------------------------------------------------------ inference


def _n_targets(model: PythonObject, state: LogisticQueryOwners) raises -> Int:
    """`_n_targets`: 1 for two classes, C for C > 2 (`len(self.classes_)`)."""
    var classes = state.classes
    if classes is None:
        raise Error("AttributeError: '" + _name(model) + "' object has no attribute 'classes_'")
    var k = len(classes)
    return logistic_target_count(k)


def _fitted_w(model: PythonObject, state: LogisticQueryOwners) raises -> PythonObject:
    var w = state.weights
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
    var status: List[Int32]

    def __init__(out self, x: PythonObject, rows: Int, cols: Int, store: QnFoldStore, var status: List[Int32]):
        self.x = x
        self.rows = rows
        self.cols = cols
        self.store = store.copy()
        self.status = status^

    def fold(self) raises -> Bool:
        return self.store.active()

    def status_address(self) -> Int:
        return Int(self.status.unsafe_ptr())

    def refuse(self, model: PythonObject) raises:
        """The deferred finite refusal of a host X (scikit-learn's wording)."""
        admitted_finite_refuse(Int(self.status[0]), _name(model))


# TOMBSTONE: A8C_LOGISTIC_QUERY_HOST_SCAN (removed: superseded by the entries' device test of their upload) deleted 2026-11-09 by fb8823ee0a; code recoverable at fb8823ee0a^.
# Restore: git apply experiments/removed/A8C_LOGISTIC_QUERY_HOST_SCAN.patch
def _query_x(model: PythonObject, fold: Bool, source: PythonObject, context: PythonObject) raises -> _QueryX:
    """X for decision_function / predict / predict_proba. The door no longer
    scans a borrowed X on the host (lane a8c): `defer_finite`, and every
    estimators entry tests the words it uploads (glm/qn_infer.mojo
    qn_x_status; the host column's parallel scan) and returns the status
    word. A DeviceRows fold of an identical-mode GPU model is read on the
    device (`qn_infer_fold`); without it (the CPU column, FAST, a
    `device="cpu"` estimator, a host class) the fold's rows are gathered on
    the host and tested there. `fold` is the route's answer (resolved once)."""
    var status = List[Int32](length=1, fill=Int32(0))
    if fold and door_is_device_rows(source, context):
        var fx = admit_x_unrecorded(source, _name(model), context, device_x=True)
        if fx.store.active():
            return _QueryX(PythonObject(None), fx.rows, fx.cols,
                           QnFoldStore(fx.store.base_id, fx.store.base_rows, fx.store.index_id), status^)
        return _QueryX(fx.array, fx.rows, fx.cols, QnFoldStore(), status^)
    var x = admit_x(source, 2, "X", context, defer_finite=True)
    return _QueryX(x, Int(py=x.shape[0]), Int(py=x.shape[1]), QnFoldStore(), status^)


def _infer_fold(model: PythonObject, state: LogisticQueryOwners, q: _QueryX, dst: PythonObject, mode: Int,
                n_classes: Int, log_out: Bool, context: PythonObject) raises:
    """One `qn_infer_fold` call over the fold `q` names into `dst`, refused
    through door_rows_refuse on its words."""
    var operation = QN_NATIVE_DECISION if mode == _FOLD_DECISION else (QN_NATIVE_BINARY if mode == _FOLD_CODES else QN_NATIVE_PROBA)
    var request = QnQueryRequest(0, Int(py=classical_address(state.weights)),
        labels_address(dst), 0, q.rows, q.cols, n_classes, Bool(py=state.intercept_owner), log_out)
    var nk = logistic_keys()
    var result = logistic_run_fold_query(py_item(context, nk[].k_logistic_fold_native_api_v1), operation, q.store, request)
    _ = dst
    door_rows_refuse(result.finite, result.range_error, _name(model), q.store.base_rows)


def logistic_decision(model: PythonObject, state: LogisticQueryOwners, routed: LogisticRouteCall, source: PythonObject,
                      context: PythonObject) raises -> PythonObject:
    """`decision_function`: (n,) float32 scores for two classes, (n, C)
    for more. `state` pins the admitted public fields for this call;
    `routed` is the estimator's resolved route."""
    if not state.has_weights:
        raise Error("mojolearn LogisticRegression: call fit first")
    var q = _query_x(model, routed.route.fold, source, context)
    var rows = q.rows
    var cols = q.cols
    logistic_query_width_check(cols, state.features(), context)
    var n_targets = _n_targets(model, state)
    var intercept = 1 if Bool(py=state.intercept_owner) else 0
    var out = PythonObject(None)
    if n_targets == 1:
        out = admitted_storage_native([rows], "<f4", context, as_array=True)
    else:
        out = admitted_storage_native([rows, n_targets], "<f4", context, as_array=True)
    if q.fold():
        _infer_fold(model, state, q, out, _FOLD_DECISION, n_targets if n_targets > 1 else 1, False, context)
        return out
    # params: n_rows, n_features, fit_intercept, n_classes (1: the binary
    # shape), the status word's address (lane a8c)
    var request = QnQueryRequest(Int(py=classical_address(q.x)), Int(py=classical_address(state.weights)),
        labels_address(out), q.status_address(), rows, cols, n_targets, intercept != 0, False)
    logistic_route_query(routed, QN_NATIVE_DECISION, request)
    _ = q.x  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    q.refuse(model)
    return out


def logistic_predict(model: PythonObject, state: LogisticQueryOwners, routed: LogisticRouteCall, source: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    """`predict`: the device threshold (two classes) or row argmax (more)
    beside the scores, decoded to `classes_`."""
    # The label scope of the estimator's device (labels_scope), once per call.
    return logistic_predict_scoped(model, state, routed, source, context, labels_scope(model, context))


def logistic_predict_scoped(model: PythonObject, state: LogisticQueryOwners, routed: LogisticRouteCall,
                            source: PythonObject, context: PythonObject,
                            label_context: PythonObject) raises -> PythonObject:
    """`logistic_predict` with the operation's label scope already read (score
    reads it once for the predict and the metric)."""
    var n_targets = _n_targets(model, state)
    if n_targets == 1:
        var q = _query_x(model, routed.route.fold, source, context)
        var rows = q.rows
        var cols = q.cols
        logistic_query_integer_width(cols, Int(py=state.features()))
        var codes = admitted_storage_native([rows], "<i8", context, as_array=True)
        var w = _fitted_w(model, state)
        if q.fold():
            _infer_fold(model, state, q, codes, _FOLD_CODES, 1, False, context)
            return classical_decode_labels(state.classes, codes, label_context)
        var request = QnQueryRequest(Int(py=classical_address(q.x)), Int(py=classical_address(w)),
            labels_address(codes), q.status_address(), rows, cols, 1, Bool(py=state.intercept_owner), False)
        logistic_route_query(routed, QN_NATIVE_BINARY, request)
        _ = w  # keep the Python object alive through the native call above
        _ = q.x  # keep the Python object alive through the native call above
        _ = codes  # keep the Python object alive through the native call above
        q.refuse(model)
        return classical_decode_labels(state.classes, codes, label_context)
    if routed.route.multiclass:
        if not state.has_weights:
            raise Error("mojolearn LogisticRegression: call fit first")
        var q = _query_x(model, routed.route.fold, source, context)
        var rows = q.rows
        var cols = q.cols
        logistic_query_width_check(cols, state.features(), context)
        var codes = admitted_storage_native([rows], "<i8", context, as_array=True)
        if q.fold():
            _infer_fold(model, state, q, codes, _FOLD_CODES, n_targets, False, context)
            return classical_decode_labels(state.classes, codes, label_context)
        if rows:
            var request = QnQueryRequest(Int(py=classical_address(q.x)), Int(py=classical_address(state.weights)),
                labels_address(codes), q.status_address(), rows, cols, n_targets, Bool(py=state.intercept_owner), False)
            logistic_route_query(routed, QN_NATIVE_MULTICLASS, request)
            _ = q.x  # keep the Python object alive through the native call above
            _ = codes  # keep the Python object alive through the native call above
            q.refuse(model)
        return classical_decode_labels(state.classes, codes, label_context)
    # A binding without the device argmax keeps the host scan, the same
    # first-maximum rule (`_labels.argmax_rows`).
    var scores = logistic_decision(model, state, routed, source, context)
    var rows = Int(py=scores.shape[0])
    var codes = labels_argmax_rows(scores, rows, n_targets, False, label_context)
    _ = scores^
    return classical_decode_labels(state.classes, codes, label_context)


def _logistic_proba_fused(model: PythonObject, state: LogisticQueryOwners, routed: LogisticRouteCall,
                          source: PythonObject, context: PythonObject, log_out: Bool = False) raises -> PythonObject:
    """The existing fused decision/link kernel through the typed QN driver.

    Decision checks retain their order and no intermediate score array is
    created between the two fused operations.
    """
    if not state.has_weights:
        raise Error("mojolearn LogisticRegression: call fit first")
    var n_targets = _n_targets(model, state)
    var q = _query_x(model, routed.route.fold, source, context)
    var rows = q.rows
    var cols = q.cols
    logistic_query_width_check(cols, state.features(), context)
    var intercept = 1 if Bool(py=state.intercept_owner) else 0
    var out = admitted_storage_native([rows, 2 if n_targets == 1 else n_targets], "<f8", context, as_array=True)
    if q.fold():
        _infer_fold(model, state, q, out, _FOLD_PROBA, n_targets if n_targets > 1 else 1, log_out, context)
        return out
    # ne2-g-linear (N361): the 5th word asks the entry for log(p) in the
    # same device session (py2mojo_rows ROWS_LOG's words); the 6th is the
    # status word's address (lane a8c).
    var request = QnQueryRequest(Int(py=classical_address(q.x)), Int(py=classical_address(state.weights)),
        labels_address(out), q.status_address(), rows, cols, n_targets, intercept != 0, log_out)
    logistic_route_query(routed, QN_NATIVE_PROBA, request)
    _ = q.x  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    q.refuse(model)
    return out


def logistic_proba(model: PythonObject, state: LogisticQueryOwners, routed: LogisticRouteCall, source: PythonObject,
                   context: PythonObject) raises -> PythonObject:
    """`predict_proba`: the sigmoid (two classes) or softmax (more) of the
    decision scores, in float64. One public owner snapshot and one
    resolved route serve the decision and the link."""
    return _logistic_proba_fused(model, state, routed, source, context)


def logistic_log_proba(model: PythonObject, state: LogisticQueryOwners, routed: LogisticRouteCall, source: PythonObject,
                       context: PythonObject) raises -> PythonObject:
    """`predict_log_proba`: `_log_or_inf` of every probability in the
    estimators binding (py2mojo_rows ROWS_LOG)."""
    if not state.has_weights:
        raise Error("mojolearn LogisticRegression: call fit first")
    return _logistic_proba_fused(model, state, routed, source, context, True)


def logistic_score(prediction: PythonObject, target: PythonObject, context: PythonObject) raises -> PythonObject:
    """`score`: the mean accuracy of the decoded prediction against y, handed
    straight to the native accuracy program (`_accuracy_host`). The caller
    predicts: through the estimator's route, or, for the explicit host
    class, through its own native host inference owner."""
    var y = target
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
        var nk = logistic_keys()
        var operation = py_op_code(nk[].ops, call[1])
        var args = call[2]
        var context = call[3]
        if py_type(py_type(model)) is not py_type(py_type(PythonObject(0))):
            raise Error("TypeError: LogisticRegression cannot invoke Python metaclass hooks")
        # The estimator's route owner, when it was resolved for this
        # connection context and this class, names the context the class
        # runs in (the explicit host class context or the public one): one
        # record read and two pointer compares. Without it (the constructor,
        # the first operation, after set_params) the class hierarchy is
        # walked against the explicit host classes, as every call did before.
        var base = context
        var held = PythonObject(None)
        if operation != LOGISTIC_OPN_INITIALIZE:
            held = logistic_route_held(model, base)
        if held is not None:
            context = logistic_route_context(held)
        else:
            var hosts = py_get(context, nk[].k_lr_host_contexts)
            if hosts is not None and len(hosts) > 0:
                var mro = py_type(model).__mro__
                for k in range(len(mro)):  # driver-loop: over the model's class hierarchy (explicit host classes)
                    var host = py_get(hosts, mro[k])
                    if host is not None:
                        context = host
                        break
        var routed_context = context
        classical_connect_numpy(context)
        context = lda_native_context(context)
        var result = PythonObject(None)
        if operation == LOGISTIC_OPN_INITIALIZE:
            # `device` (the last constructor argument) is checked where it is
            # given: an unknown value, and 'gpu' on a CPU-only installation,
            # are refused in words before any field changes. The default None
            # is one identity compare and reads nothing more. No owner is
            # allocated here; a reused instance (`__init__` called again)
            # forgets its route. A wrong argument count is refused inside.
            var constructor = py_item(context, nk[].k_lr_constructor_parameters)
            if len(args) == len(constructor):
                var device = args[len(constructor) - 1]
                if device is not None:
                    classical_admit_value(device, context)
                    logistic_device_admit(device, context, _builtins())
            result = logistic_initialize(model, args, context)
            device_route_clear(model)
            return result.steal_data()
        # get_params/set_params/is_fitted validate nothing beyond admission
        # (scikit-learn's convention); consuming operations check values.
        var reporting = operation == LOGISTIC_OPN_VENDOR_USED or operation == LOGISTIC_OPN_NUMERIC_MODE_USED
        var configuring = (operation == LOGISTIC_OPN_GET_PARAMS or operation == LOGISTIC_OPN_SET_PARAMS
                           or operation == LOGISTIC_OPN_IS_FITTED or reporting)
        # Admit live storage once; query owners below pin only consumed
        # public fields, without copying the complete Python dictionary.
        var fields = logistic_fields(model, context, configuring)
        if operation == LOGISTIC_OPN_GET_PARAMS:
            # PACK the admitted public parameters in their fixed order.
            var params = Python.dict()
            var names = py_item(context, nk[].k_lr_sorted_parameters)
            for i in range(len(names)):  # driver-loop: ParamsMixin's sorted parameter names
                # `device` alone may be absent (an instance built without
                # its constructor): None, this installation's device.
                py_setitem(params, names[i], py_get(fields, names[i]))
            result = params
            return result.steal_data()
        if operation == LOGISTIC_OPN_SET_PARAMS:
            # A new device is refused in words before the estimator changes
            # (inside). The device or the numeric mode may have changed: the
            # next operation resolves the route again.
            result = logistic_set_params(model, args[0], context)
            device_route_clear(model)
            return result.steal_data()
        if operation == LOGISTIC_OPN_IS_FITTED:
            result = classical_is_fitted(model)
            return result.steal_data()
        if reporting:
            # What the estimator runs on, in the words its resolved image
            # reported (fitted or not).
            var report = _route(model, fields, held, base, routed_context)
            if operation == LOGISTIC_OPN_VENDOR_USED:
                result = PythonObject(linear_vendor_name(report.route.vendor))
            else:
                result = PythonObject("identical" if report.route.mode == 1 else "fast")
            return result.steal_data()
        # The operation's name String is built only for the not-fitted message
        # (classical_require_fitted re-checks and raises it).
        if operation != LOGISTIC_OPN_FIT and not linear_fitted_unpack(model):
            classical_require_fitted(model, False, String(call[1]), context)
        # One module handle for this call's input admissions.
        var mods = NativeModules()
        classical_admit_value(mods, args[0], context)
        # The explicit host class scores through its own native host
        # inference owner, as its `predict` does; it takes no route.
        if operation == LOGISTIC_OPN_SCORE:
            var host_context = py_get(context, nk[].k_lr_host_score_context)
            if host_context is not None:
                var host_target = classifier_label_list(args[1], context)
                lda_native_labels(host_target, context, False, "LogisticRegression")
                var host_prediction = host_logistic_score_predict(model, args[0], host_context)
                result = logistic_score(host_prediction, host_target, context)
                result = _public_result(result, context)
                return result.steal_data()
        # The estimator's device route: resolved once per estimator (its
        # `device` and numeric mode, the device image or the CPU image), read
        # here by identity compares. fit and every query below call through
        # it; none of them tests the device.
        var routed = _route(model, fields, held, base, routed_context)
        if operation == LOGISTIC_OPN_FIT:
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
            result = logistic_fit(model, args[0], target, args[2], context, fields, routed)
            classifier_after_fit(logistic_storage(model, context), args[0], context)
        else:
            var query = LogisticQueryOwners(fields, context)
            try:
                if operation == LOGISTIC_OPN_DECISION_FUNCTION:
                    result = logistic_decision(model, query, routed, args[0], context)
                elif operation == LOGISTIC_OPN_PREDICT:
                    result = logistic_predict(model, query, routed, args[0], context)
                elif operation == LOGISTIC_OPN_PREDICT_PROBA:
                    result = logistic_proba(model, query, routed, args[0], context)
                elif operation == LOGISTIC_OPN_PREDICT_LOG_PROBA:
                    result = logistic_log_proba(model, query, routed, args[0], context)
                elif operation == LOGISTIC_OPN_SCORE:
                    var target = classifier_label_list(args[1], context)
                    lda_native_labels(target, context, False, "LogisticRegression")
                    # One label scope read for the predict and the metric.
                    var scope = labels_scope(model, context)
                    var prediction = logistic_predict_scoped(model, query, routed, args[0], context, scope)
                    result = logistic_score(prediction, target, scope)
                else:
                    raise Error("unknown native LogisticRegression operation")
            finally:
                _ = query^
        _ = routed.owner  # the route's image pins live until the operation's result is packed
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
        if String(error).startswith("KeyError:"):
            return raise_python_exception(error_without_class(error, "KeyError"), ExceptionType("PyExc_KeyError"))
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
