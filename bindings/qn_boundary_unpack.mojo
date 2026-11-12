# SPDX-License-Identifier: Apache-2.0
# UNPACK and PACK of QNRegressor's public fit and predict: the only part of
# the QN door that reads or writes Python objects. UNPACK reads the
# estimator's fields and admits X, y and the weights into one typed request
# (glm/qn_door_run.mojo) plus the Python owners that keep every addressed
# buffer alive; PACK publishes the fitted attributes. The registered door,
# bindings/qn_native.mojo, calls UNPACK, releases the GIL for RUN, then PACK
# (lane doorsplit-linear, 2026-11-11).
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.classical_arrays_door import classical_address, classical_typed, classical_dtype_str, classical_c_order
from bindings.admitted_input import admit_x, admit_x_unrecorded, admitted_vector
from bindings.device_rows_door import door_is_device_rows
from bindings.classical_validate import classical_view, classical_check_weights_input
from core.estimators_entry_table import EstimatorsEntryKey, estimators_vendor_word
from core.logistic_native_api import QnFitRequest, QnQueryRequest, QnFoldStore
from glm.qn_door_run import (
    QnDoorFitRequest, QnDoorFitResult, QnDoorPredictRequest,
    qn_check_lengths, qn_check_features, qn_check_coefficients, qn_change_tolerance,
)


def _qn_key(fields: PythonObject, context: PythonObject) raises -> EstimatorsEntryKey:
    # The estimator's numeric mode (None: the process mode) and the native
    # QN table key of its connected estimators image (the bridge records
    # it at import), checked against the selected vendor and mode.
    var b = native_loaded_module("builtins")
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not b.str:
        raise Error("numeric_mode must be fast, identical or None")
    # ASCII strip/lower natively; other spellings take Python's Unicode-aware
    # strip/lower (same accepted words, same refusal).
    var mode_name = String(String(String(mode).strip()).lower())
    if mode_name != "identical" and mode_name != "fast":
        mode_name = String(py_call(py_call(mode.strip).lower))
    if mode_name != "identical" and mode_name != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    var entries = context["qn_bindings"][PythonObject(mode_name)]
    if py_type(entries) is b.str:
        raise Error("ImportError: " + String(entries))
    # The image's compiled tier and vendor were read once at import; these
    # compares replace the per-call Python probes (`estimators_vendor`,
    # `estimators_numeric_mode`) and their per-table verdict cache.
    var native = entries["native_key"]
    var key = EstimatorsEntryKey(Int(py=native[0]), Int(py=native[1]))
    if key.vendor != estimators_vendor_word(String(context["vendor"])):
        raise Error("RuntimeError: QN native vendor differs from selected execution")
    if key.mode != (1 if mode_name == "identical" else 0):
        raise Error("RuntimeError: QN native mode differs from selected execution")
    return key^


def _int(value: PythonObject) raises -> Int:
    return Int(py=py_call(native_loaded_module("builtins").int, value))


def _float(value: PythonObject) raises -> Float64:
    return Float64(py=py_call(native_loaded_module("builtins").float, value))


def _addr(value: PythonObject) raises -> Int:
    return Int(py=classical_address(value))


struct QnFitUnpacked(Movable):
    """A fit's typed request and the Python owners of every buffer it
    addresses (held until PACK)."""
    var run: QnDoorFitRequest
    var x: PythonObject
    var target: PythonObject
    var weights: PythonObject
    var w: PythonObject
    var info: PythonObject
    var status: PythonObject
    var coef: PythonObject
    var iterations: PythonObject
    var copied: Bool

    def __init__(out self, var run: QnDoorFitRequest, var x: PythonObject, var target: PythonObject,
                 var weights: PythonObject, var w: PythonObject, var info: PythonObject, var status: PythonObject,
                 var coef: PythonObject, var iterations: PythonObject, copied: Bool):
        self.run = run^
        self.x = x^
        self.target = target^
        self.weights = weights^
        self.w = w^
        self.info = info^
        self.status = status^
        self.coef = coef^
        self.iterations = iterations^
        self.copied = copied


def qn_fit_unpack(model: PythonObject, args: PythonObject, context: PythonObject) raises -> QnFitUnpacked:
    var fields = py_fields(model)
    var key = _qn_key(fields, context)
    # Lane a7c: the finite test of a borrowed X is the QN fit entry's
    # (the request's status word: the device scan of the upload, the
    # CPU column's parallel scan); a cast X is tested in its cast pass.
    # An identical-mode cross-validation fold (DeviceRows) is fitted by the
    # base binding's resident-fold QN entry (core/estimators_entry_table.mojo
    # ESTIMATORS_TABLE_QN_FOLD) on rows gathered on the device out of model
    # selection's resident base (no host gather, no upload of X).
    var fold_key = py_get(context, "qn_fold_key")
    var use_fold = False
    var store = QnFoldStore()
    var x = PythonObject(None)
    var rows = 0
    var cols = 0
    if fold_key is not None and key.mode == 1 and door_is_device_rows(args[0], context):
        var fx = admit_x_unrecorded(args[0], "QNRegressor", context, device_x=True)
        rows = fx.rows
        cols = fx.cols
        if fx.store.active():
            use_fold = True
            store = QnFoldStore(fx.store.base_id, fx.store.base_rows, fx.store.index_id)
        else:
            x = fx.array
    else:
        x = admit_x(args[0], 2, "X", context, defer_finite=True)
        rows = Int(py=x.shape[0])
        cols = Int(py=x.shape[1])
    var view = classical_view(args[1], context, "y")
    if Int(py=view.ndim) != 1:
        raise Error("mojolearn QNRegressor requires a 1-D y")
    var target = admit_x(view, 1, "y", context)
    qn_check_lengths(rows, Int(py=target.shape[0]))
    var weights = classical_check_weights_input(args[2], rows, context)
    var tol = _float(fields["tol"])
    var delta = fields["delta"]
    var delta_value = Optional[Float64](None)
    if delta is not None:
        delta_value = _float(delta)
    var change_tol = qn_change_tolerance(delta_value, tol)
    var loss = py_get(context["qn_losses"], fields["loss"])
    if loss is None:
        raise Error("mojolearn QNRegressor: unknown loss " + String(py_call(native_loaded_module("builtins").repr, fields["loss"])))
    var intercept = Bool(py=fields["fit_intercept"])
    var output = context["output_context"]
    var w = classical_typed(Python.tuple(cols + (1 if intercept else 0)), "<f4", output)
    var info = classical_typed(Python.tuple(2), "<f4", output)
    var status = admitted_vector(1, "<i4", output)
    var coef = classical_typed(Python.tuple(cols), "<f4", output)
    var iterations = classical_typed(Python.tuple(1), "<i8", output)
    var status_addr = _addr(status)
    # qn_fit's fields as one typed request (the order and values the Python
    # params list carried: one target, gradient tolerance = tol, svr_eps 0,
    # the weights' address or 0, then the status word).
    var request = QnFitRequest(0 if use_fold else _addr(x), _addr(target), _addr(w), _addr(info),
        _addr(weights) if weights is not None else 0, 0 if use_fold else status_addr, rows, cols, 1,
        _int(fields["max_iter"]), _int(fields["linesearch_max_iter"]), _int(fields["lbfgs_memory"]), intercept,
        Bool(py=fields["penalty_normalized"]), _int(loss), _float(fields["l1_strength"]),
        _float(fields["l2_strength"]), tol, change_tol, 0.0)
    var fold = EstimatorsEntryKey(Int(py=fold_key[0]), Int(py=fold_key[1])) if use_fold else key.copy()
    var copied = not (x is args[0])
    var run = QnDoorFitRequest(key^, fold^, use_fold, store^, request^, status_addr, _addr(coef), _addr(iterations))
    return QnFitUnpacked(run^, x^, target^, weights^, w^, info^, status^, coef^, iterations^, copied)


struct QnPredictUnpacked(Movable):
    """A decision's typed request and the owners of X, the coefficients and
    the output."""
    var run: QnDoorPredictRequest
    var x: PythonObject
    var weights: PythonObject
    var result: PythonObject

    def __init__(out self, var run: QnDoorPredictRequest, var x: PythonObject, var weights: PythonObject,
                 var result: PythonObject):
        self.run = run^
        self.x = x^
        self.weights = weights^
        self.result = result^


def qn_predict_unpack(model: PythonObject, source: PythonObject, context: PythonObject) raises -> QnPredictUnpacked:
    var fields = py_fields(model)
    if not py_contains(fields, "_w"):
        raise Error("mojolearn QNRegressor: call fit before predict")
    var key = _qn_key(fields, context)
    var x = admit_x(source, 2, "X", context)
    var rows = Int(py=x.shape[0])
    var cols = Int(py=x.shape[1])
    qn_check_features(cols, Int(py=fields["n_features_in_"]))
    var weights = classical_view(fields["_w"], context, "fitted coefficients")
    var intercept = Bool(py=fields["fit_intercept"])
    qn_check_coefficients(classical_dtype_str(weights, context) == "<f4" and classical_c_order(weights, context),
                          Int(py=weights.size), cols, intercept)
    var result = classical_typed(Python.tuple(rows), "<f4", context["output_context"])
    # The qn_decision_function request: one target, no status word (X was
    # tested by admit_x), through the native estimators QN table.
    var request = QnQueryRequest(_addr(x), _addr(weights), _addr(result), 0, rows, cols, 1, intercept, False)
    return QnPredictUnpacked(QnDoorPredictRequest(key^, request^), x^, weights^, result^)


def qn_predict_pack(var predict: QnPredictUnpacked) raises -> PythonObject:
    var result = predict.result
    _ = predict^
    return result
