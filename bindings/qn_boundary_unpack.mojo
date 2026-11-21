# SPDX-License-Identifier: Apache-2.0
# UNPACK and PACK of QNRegressor's public fit and predict: the only part of
# the QN door that reads or writes Python objects. UNPACK reads the
# estimator's fields and admits X, y and the weights into one typed request
# (glm/qn_door_run.mojo) plus the Python owners that keep every addressed
# buffer alive; PACK publishes the fitted attributes. The registered door,
# bindings/qn_native.mojo, calls UNPACK, releases the GIL for RUN, then PACK
# (lane doorsplit-linear, 2026-11-11).
from core.mode_answer import MODE_FAMILY_LINEAR, answer_mode_word
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.classical_arrays_door import classical_address, classical_typed, classical_dtype_str, classical_c_order
from bindings.admitted_input_unpack import admit_x, admit_x_unrecorded, admitted_vector
from bindings.device_rows_door import door_is_device_rows
from bindings.classical_validate_unpack import classical_view, classical_check_weights_input
from core.estimators_entry_table import (
    EstimatorsEntryKey, estimators_vendor_word, estimators_table_entry, ESTIMATORS_TABLE_QN, ESTIMATORS_TABLE_QN_FOLD,
)
from core.entry_table import EntryRef, entry_ref_word
from core.device_route_dw_classical import FamilyRoute, family_mode_code, family_route_plan, family_route_admit
from providers.python.device_route_dw_classical_door import (
    FamilyRouteOwner, family_route_held, family_route_keep, family_route_value, family_device_request,
    family_route_second_refuse,
)
from core.logistic_native_api import QnFitRequest, QnQueryRequest, QnFoldStore, QN_NATIVE_COUNT, QN_FOLD_COUNT
from glm.qn_door_run import (
    QnDoorFitRequest, QnDoorFitResult, QnDoorPredictRequest,
    qn_check_lengths, qn_check_features, qn_check_coefficients, qn_change_tolerance,
)
from bindings.qn_keys_unpack import qn_keys
from bindings.py_balanced import py_item


# TOMBSTONE: `_qn_key` (the per-call entry selection: the mode String with its strip and lower, the
# `qn_bindings` read by a new str, the `native_key` tuple reads, the vendor String and its compares, and
# RUN's table slot read by that key at every fit and predict) deleted 2026-11-20 by lane dw-classical;
# the estimator's route owner holds the resolved entry (`qn_route`). Code recoverable at 08d3e8ce3b.


@fieldwise_init
struct QnRouteCall(Movable):
    """One operation's route: the resolved entries by value, and the owner
    whose pins keep their images loaded until the operation returns."""
    var route: FamilyRoute
    var owner: PythonObject


def _qn_route_build(device_request: Int, mode: PythonObject, default_mode: PythonObject,
    context: PythonObject, b: PythonObject) raises -> FamilyRouteOwner:
    """Resolve an estimator's QN entry once: its numeric mode (None: the
    process mode), its device (None: this installation's), the native QN
    table key of the image those select (the bridge recorded each image's
    key at import), checked against the selected vendor and tier, and the
    table read. Runs at an estimator's first operation, after a changed
    mode and after a new `__init__`; never per call."""
    var nk = qn_keys()
    var owner = FamilyRouteOwner(device_request)
    var selected = mode
    if mode is None:
        selected = default_mode
    if py_type(selected) is not b.str:
        raise Error("numeric_mode must be fast, identical or None")
    # ASCII strip/lower natively; other spellings take Python's Unicode-aware
    # strip/lower (same accepted words, same refusal).
    var mode_name = String(String(String(selected).strip()).lower())
    if mode_name != "identical" and mode_name != "fast":
        mode_name = String(py_call(py_call(selected.strip).lower))
    if mode_name != "identical" and mode_name != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    answer_mode_word(mode_name, MODE_FAMILY_LINEAR)
    var mode_code = family_mode_code(mode_name)
    var process_vendor = estimators_vendor_word(String(py_item(context, nk[].k_vendor)))
    var plan = family_route_plan(device_request, process_vendor, mode_code)
    var entries = PythonObject(None)
    if plan.cpu_table:
        # A GPU installation running `device="cpu"`: the CPU image loaded
        # beside the device image at import
        # (python/mojolearn/_native_qn_bridge.py connect_cpu_device).
        var cpu = py_get(context, nk[].k_qn_cpu_bindings)
        if cpu is None:
            raise Error("ImportError: device='cpu' is not connected in this installation")
        entries = py_item(cpu, PythonObject(mode_name))
    else:
        entries = py_item(py_item(context, nk[].k_qn_bindings), PythonObject(mode_name))
    if py_type(entries) is b.str:
        raise Error("ImportError: " + String(entries))
    var native = py_item(entries, nk[].k_native_key)
    var key = EstimatorsEntryKey(Int(py=native[0]), Int(py=native[1]))
    family_route_admit("QN", plan, key.mode, key.vendor, mode_code)
    var entry = estimators_table_entry[ESTIMATORS_TABLE_QN](key)
    owner.route.device = plan.device
    owner.route.mode = key.mode
    owner.route.vendor = key.vendor
    owner.route.entry = EntryRef(entry, entry_ref_word(key.mode, key.vendor, QN_NATIVE_COUNT))
    # The resident-fold entry of the GPU base image: an identical-mode
    # cross-validation fold on the device is fitted from the fold store
    # without a host gather. A CPU route never reads the device store.
    var fold_key = py_get(context, nk[].k_qn_fold_key)
    if fold_key is not None and key.mode == 1 and not plan.cpu_table and key.vendor != 0:
        owner.route.has_second = True
        try:
            var fold = EstimatorsEntryKey(Int(py=fold_key[0]), Int(py=fold_key[1]))
            var fold_entry = estimators_table_entry[ESTIMATORS_TABLE_QN_FOLD](fold)
            owner.route.second = EntryRef(fold_entry, entry_ref_word(fold.mode, fold.vendor, QN_FOLD_COUNT))
        except error:
            # Said by the fold fit that needs it, in the table's own words.
            owner.second_refusal = String(error)
    owner.mode = mode
    owner.default_mode = default_mode
    owner.context = context
    owner.images = Python.tuple(entries, py_item(context, nk[].k_qn_bindings))
    owner.resolved = True
    return owner^


def qn_route(model: PythonObject, fields: PythonObject, context: PythonObject) raises -> QnRouteCall:
    """The estimator's route for this operation. A hit is the mode read (and
    the default-mode read when it is None), one record read and three
    pointer compares; a miss resolves and stores it."""
    var nk = qn_keys()
    var mode = py_get(fields, nk[].k_numeric_mode)
    var default_mode = PythonObject(None)
    if mode is None:
        default_mode = py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
    var owner = family_route_held(model, mode, default_mode, context)
    if owner is None:
        var b = native_loaded_module("builtins")
        var device = family_device_request(fields, b)
        owner = family_route_keep(model, _qn_route_build(device, mode, default_mode, context, b))
    return QnRouteCall(family_route_value(owner), owner)


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
    # The route owner: its pins keep the entry's image loaded through RUN.
    var route_owner: PythonObject

    def __init__(out self, var run: QnDoorFitRequest, var x: PythonObject, var target: PythonObject,
                 var weights: PythonObject, var w: PythonObject, var info: PythonObject, var status: PythonObject,
                 var coef: PythonObject, var iterations: PythonObject, copied: Bool, var route_owner: PythonObject):
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
        self.route_owner = route_owner^


def qn_fit_unpack(model: PythonObject, args: PythonObject, context: PythonObject) raises -> QnFitUnpacked:
    var nk = qn_keys()
    var fields = py_fields(model)
    var call = qn_route(model, fields, context)
    var route = call.route
    # Lane a7c: the finite test of a borrowed X is the QN fit entry's
    # (the request's status word: the device scan of the upload, the
    # CPU column's parallel scan); a cast X is tested in its cast pass.
    # An identical-mode cross-validation fold (DeviceRows) is fitted by the
    # base binding's resident-fold QN entry (core/estimators_entry_table.mojo
    # ESTIMATORS_TABLE_QN_FOLD) on rows gathered on the device out of model
    # selection's resident base (no host gather, no upload of X).
    var use_fold = False
    var store = QnFoldStore()
    var x = PythonObject(None)
    var rows = 0
    var cols = 0
    if route.has_second and door_is_device_rows(args[0], context):
        var fx = admit_x_unrecorded(args[0], "QNRegressor", context, device_x=True)
        rows = fx.rows
        cols = fx.cols
        if fx.store.active():
            if route.second.word == 0:
                # The fold entry did not resolve (an unresolved ref's word is
                # 0): refuse in the table's own words.
                family_route_second_refuse(call.owner, "RuntimeError: the resident-fold QN entry is unavailable")
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
    var tol = _float(py_item(fields, nk[].k_tol))
    var delta = py_item(fields, nk[].k_delta)
    var delta_value = Optional[Float64](None)
    if delta is not None:
        delta_value = _float(delta)
    var change_tol = qn_change_tolerance(delta_value, tol)
    var loss = py_get(py_item(context, nk[].k_qn_losses), py_item(fields, nk[].k_loss))
    if loss is None:
        raise Error("mojolearn QNRegressor: unknown loss " + String(py_call(native_loaded_module("builtins").repr, py_item(fields, nk[].k_loss))))
    var intercept = Bool(py=py_item(fields, nk[].k_fit_intercept))
    var output = py_item(context, nk[].k_output_context)
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
        _int(py_item(fields, nk[].k_max_iter)), _int(py_item(fields, nk[].k_linesearch_max_iter)), _int(py_item(fields, nk[].k_lbfgs_memory)), intercept,
        Bool(py=py_item(fields, nk[].k_penalty_normalized)), _int(loss), _float(py_item(fields, nk[].k_l1_strength)),
        _float(py_item(fields, nk[].k_l2_strength)), tol, change_tol, 0.0)
    var copied = not (x is args[0])
    var run = QnDoorFitRequest(route.entry, route.second, use_fold, store^, request^, status_addr, _addr(coef), _addr(iterations))
    return QnFitUnpacked(run^, x^, target^, weights^, w^, info^, status^, coef^, iterations^, copied, call.owner)


struct QnPredictUnpacked(Movable):
    """A decision's typed request and the owners of X, the coefficients and
    the output."""
    var run: QnDoorPredictRequest
    var x: PythonObject
    var weights: PythonObject
    var result: PythonObject
    # The route owner: its pins keep the entry's image loaded through RUN.
    var route_owner: PythonObject

    def __init__(out self, var run: QnDoorPredictRequest, var x: PythonObject, var weights: PythonObject,
                 var result: PythonObject, var route_owner: PythonObject):
        self.run = run^
        self.x = x^
        self.weights = weights^
        self.result = result^
        self.route_owner = route_owner^


def qn_predict_unpack(model: PythonObject, source: PythonObject, context: PythonObject) raises -> QnPredictUnpacked:
    var nk = qn_keys()
    var fields = py_fields(model)
    if not py_contains(fields, nk[].k__w):
        raise Error("mojolearn QNRegressor: call fit before predict")
    var call = qn_route(model, fields, context)
    var x = admit_x(source, 2, "X", context)
    var rows = Int(py=x.shape[0])
    var cols = Int(py=x.shape[1])
    qn_check_features(cols, Int(py=py_item(fields, nk[].k_n_features_in_)))
    var weights = classical_view(py_item(fields, nk[].k__w), context, "fitted coefficients")
    var intercept = Bool(py=py_item(fields, nk[].k_fit_intercept))
    qn_check_coefficients(classical_dtype_str(weights, context) == "<f4" and classical_c_order(weights, context),
                          Int(py=weights.size), cols, intercept)
    var result = classical_typed(Python.tuple(rows), "<f4", py_item(context, nk[].k_output_context))
    # The qn_decision_function request: one target, no status word (X was
    # tested by admit_x), through the native estimators QN table.
    var request = QnQueryRequest(_addr(x), _addr(weights), _addr(result), 0, rows, cols, 1, intercept, False)
    return QnPredictUnpacked(QnDoorPredictRequest(call.route.entry, request^), x^, weights^, result^, call.owner)


def qn_predict_pack(var predict: QnPredictUnpacked) raises -> PythonObject:
    var result = predict.result
    _ = predict^
    return result
