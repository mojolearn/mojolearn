# SPDX-License-Identifier: Apache-2.0
"""RUN of QNRegressor's public fit and predict: typed Mojo values only, no
Python import (tools/python_free_dirs.sh). The registered door is
bindings/qn_native.mojo; its UNPACK and PACK are bindings/qn_boundary.mojo.
The door calls UNPACK with the GIL held, releases the GIL for the whole of a
RUN call (docs/plans/PYTHON_AT_THE_BOUNDARY.md section 2.1), then calls PACK.

The fit runs the qn_fit body through the native estimators QN table
(core/estimators_entry_table.mojo), or, for an identical-mode
cross-validation fold on the device, the base binding's resident-fold QN
entry, then copies the fitted coefficients and writes the iteration count
into the buffers UNPACK allocated. Same request, same entries, same bits as
before the split (lane doorsplit-linear, 2026-11-11)."""
from core.estimators_entry_table import (
    EstimatorsEntryKey, estimators_table_entry, ESTIMATORS_TABLE_QN, ESTIMATORS_TABLE_QN_FOLD,
)
from core.logistic_native_api import (
    QnNativeDriver, QnFoldDriver, QnFitRequest, QnQueryRequest, QnFoldStore, QN_NATIVE_DECISION,
)
from core.admitted_refusals import admitted_finite_refuse, door_rows_refuse
from std.memory import unsafe_memcpy

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]


@fieldwise_init
struct QnDoorFitRequest(Movable):
    """One QNRegressor fit, as UNPACK built it. `request` is qn_fit's typed
    request (its `status` is 0 on the fold route); `coef` (cols float32
    words) and `iterations` (one int64 word) are the public outputs UNPACK
    allocated; `status_word` is the deferred finite word of a host X."""
    var key: EstimatorsEntryKey
    var fold_key: EstimatorsEntryKey
    var use_fold: Bool
    var store: QnFoldStore
    var request: QnFitRequest
    var status_word: Int
    var coef: Int
    var iterations: Int


@fieldwise_init
struct QnDoorFitResult(Copyable, Movable):
    """What PACK publishes beside the buffers: the intercept (0.0 without
    one), the solver's objective and its return code."""
    var intercept: Float64
    var objective: Float64
    var retcode: Int


@fieldwise_init
struct QnDoorPredictRequest(Movable):
    """One QNRegressor decision: qn_decision_function's typed request (one
    target, no status word: X was tested by admission)."""
    var key: EstimatorsEntryKey
    var request: QnQueryRequest


def qn_check_lengths(rows: Int, target_rows: Int) raises:
    """X and y must have the same number of rows."""
    if target_rows != rows:
        raise Error("mojolearn QNRegressor X and y lengths differ")


def qn_check_features(cols: Int, fitted: Int) raises:
    """A predict's X has the fitted width."""
    if cols != fitted:
        raise Error("mojolearn QNRegressor feature count differs from fit")


def qn_check_coefficients(float32_c_order: Bool, size: Int, cols: Int, intercept: Bool) raises:
    """The fitted coefficients are contiguous float32 words of the fitted
    width (plus one for the intercept)."""
    if not float32_c_order or size != cols + (1 if intercept else 0):
        raise Error("TypeError: QN fitted coefficients require contiguous float32 storage matching the fitted width")


def qn_change_tolerance(delta: Optional[Float64], tol: Float64) -> Float64:
    """The function-change tolerance: `delta`, or tol / 100 when unset."""
    if delta:
        return delta.value()
    return tol * 0.01


def qn_door_fit_run(req: QnDoorFitRequest) raises -> QnDoorFitResult:
    """Fit, refuse a non-finite X or a bad fold index with admission's
    wording, copy the coefficients out of the solver's [coef | intercept]
    words and write the iteration count."""
    var n_iter = 0
    ref r = req.request
    if req.use_fold:
        var fold = estimators_table_entry[ESTIMATORS_TABLE_QN_FOLD](req.fold_key)
        var words = List[Int32](length=2, fill=0)
        var words_addr = Int(words.unsafe_ptr())
        n_iter = QnFoldDriver(fold).fit(req.store, r, words_addr)
        door_rows_refuse(Int(words[0]), Int(words[1]), "QNRegressor", req.store.base_rows)
        _ = words^
    else:
        var entry = estimators_table_entry[ESTIMATORS_TABLE_QN](req.key)
        n_iter = QnNativeDriver(entry).fit(r)
        admitted_finite_refuse(Int(_I32P(unsafe_from_address=req.status_word)[0]), "QNRegressor")
    return qn_door_fit_publish(r, n_iter, req.coef, req.iterations)


def qn_door_fit_publish(r: QnFitRequest, n_iter: Int, coef: Int, iterations: Int) -> QnDoorFitResult:
    """Copy the first `features` words of the solver's [coef | intercept]
    vector into `coef`, write `n_iter` into the int64 word at `iterations`,
    and read the intercept, objective and return code."""
    var source = _F32P(unsafe_from_address=r.coefficients)
    var dest = _F32P(unsafe_from_address=coef)
    # One byte copy of the `features` coefficient words (a loop over the
    # features is data-sized, not model structure).
    if r.features > 0:
        unsafe_memcpy(dest=dest, src=source, count=r.features)
    _I64P(unsafe_from_address=iterations)[0] = Int64(n_iter)
    var info = _F32P(unsafe_from_address=r.info)
    var intercept = Float64(source[r.features]) if r.intercept else 0.0
    return QnDoorFitResult(intercept, Float64(info[0]), Int(info[1]))


def qn_door_predict_run(req: QnDoorPredictRequest) raises:
    """X @ coef (+ intercept) into the request's output buffer."""
    var entry = estimators_table_entry[ESTIMATORS_TABLE_QN](req.key)
    QnNativeDriver(entry).query(QN_NATIVE_DECISION, req.request)
