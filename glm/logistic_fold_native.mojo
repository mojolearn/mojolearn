# SPDX-License-Identifier: Apache-2.0
"""Typed resident-fold QN calls using the existing model-selection store."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.logistic_native_api import (
    LOGISTIC_FOLD_ABI, QN_FOLD_COUNT, QN_NATIVE_FIT, QN_NATIVE_DECISION,
    QN_NATIVE_BINARY, QN_NATIVE_MULTICLASS, QN_NATIVE_PROBA,
    QnFoldStore, QnFitRequest, QnQueryRequest, qn_fold_f32_ptr,
)
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.msel_device_x import msel_fold_entry_context
from core.shared_slot import SharedSlot
from glm.qn_fold import qn_fit_fold, qn_infer_fold


def logistic_fold_fit(store: QnFoldStore, r: QnFitRequest, words: Int) raises -> Int:
    var yp = qn_fold_f32_ptr(r.y)
    var wp = qn_fold_f32_ptr(r.coefficients)
    var ip = qn_fold_f32_ptr(r.info)
    var sw = qn_fold_f32_ptr(r.weights) if r.weights != 0 else yp
    var status = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=words)
    var ctx = msel_fold_entry_context()
    return qn_fit_fold(ctx, store.base, store.base_rows, store.indices,
        yp, wp, ip, r.rows, r.features, r.classes, r.l1, r.l2,
        r.gradient_tolerance, r.change_tolerance, r.max_iter, r.linesearch_max_iter,
        r.memory, r.intercept, r.normalized, r.weights != 0, sw,
        r.loss, r.svr_epsilon, status)


def logistic_fold_query(operation: Int, store: QnFoldStore, r: QnQueryRequest, words: Int) raises:
    var cp = qn_fold_f32_ptr(r.coefficients)
    if r.dst <= 0 or r.dst % 4 != 0:
        raise Error("qn_infer_fold: the output address must be positive and aligned")
    var mode = 0
    if operation == QN_NATIVE_BINARY or operation == QN_NATIVE_MULTICLASS:
        mode = 1
    elif operation == QN_NATIVE_PROBA:
        mode = 2
    elif operation != QN_NATIVE_DECISION:
        raise Error("logistic fold ABI: invalid query operation")
    var ctx = msel_fold_entry_context()
    var result = qn_infer_fold(ctx, store.base, store.base_rows, store.indices,
        cp, r.dst, r.rows, r.features, r.intercept, r.targets, mode, r.logarithm)
    var status = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=words)
    status[0] = result[0]
    status[1] = result[1]


def logistic_fold_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        var fit = operation == QN_NATIVE_FIT
        if operation < 0 or operation >= QN_FOLD_COUNT or na != (5 if fit else 3) or np != (12 if fit else 8) or nf != (5 if fit else 0):
            raise Error("logistic fold ABI: invalid request spans")
        if aa <= 0 or aa % 8 or pp <= 0 or pp % 8 or result <= 0 or result % 8 or (fit and (ff <= 0 or ff % 8)):
            raise Error("logistic fold ABI: invalid metadata address")
        var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
        var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
        var f = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        var store = QnFoldStore(p[0], p[1], p[2])
        var words = a[4] if fit else a[2]
        if words <= 0 or words % 4:
            raise Error("logistic fold ABI: invalid status address")
        var iterations = 0
        if fit:
            var request = QnFitRequest(0, a[0], a[1], a[2], a[3], 0,
                p[3], p[4], p[5], p[6], p[7], p[8], p[9] != 0, p[10] != 0,
                p[11], f[0], f[1], f[2], f[3], f[4])
            iterations = logistic_fold_fit(store, request, words)
        else:
            var request = QnQueryRequest(0, a[0], a[1], 0, p[3], p[4], p[5], p[6] != 0, p[7] != 0)
            logistic_fold_query(operation, store, request, words)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = iterations
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct LogisticFoldAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(LOGISTIC_FOLD_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(logistic_fold_v1)


comptime _API_SLOT = "MojolearnLogisticFoldIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnLogisticFoldFastV1"
comptime LOGISTIC_FOLD_API = SharedSlot[LogisticFoldAPI, _API_SLOT]
