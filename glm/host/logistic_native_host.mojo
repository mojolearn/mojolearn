# SPDX-License-Identifier: Apache-2.0
"""Existing CPU QN operations behind the same typed requests as the GPU."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.hostptr import f32_ptr, f64_ptr, i32_ptr
from core.host_finite_scan import host_finite_status
from core.host_predict_threads import host_predict_task_count
from core.estimators_host_impl import _positive, qn_fit_host_run
from core.classical_host_predict import (
    host_qn_decision_into, host_qn_decision_multi_into, host_qn_predict_binary_into,
    host_qn_sigmoid_into, host_qn_softmax_into, host_rows_log_inplace,
)
from core.logistic_native_api import (
    LOGISTIC_NATIVE_ABI, QN_NATIVE_COUNT, QN_NATIVE_DECISION, QN_NATIVE_BINARY,
    QN_NATIVE_PROBA, QN_NATIVE_SIGMOID, QN_NATIVE_SOFTMAX,
    QnFitRequest, QnQueryRequest, QnLinkRequest, QnNativeKernels, qn_native_dispatch,
)
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.py2mojo_rows import ROWS_LOG, rows_call, rows_host
from core.shared_slot import SharedSlot


struct QnHostKernels(QnNativeKernels):
    @staticmethod
    def mode() -> Int:
        return Int(GLOBAL_NUMERIC_MODE)

    @staticmethod
    def host() -> Bool:
        return True

    @staticmethod
    def multiclass() -> Bool:
        # The host export keeps its existing decision + native argmax route.
        return False

    @staticmethod
    def fit(r: QnFitRequest) raises -> Int:
        var wp = f32_ptr(r.coefficients)
        var ip = f32_ptr(r.info)
        if r.status != 0:
            var word = host_finite_status(r.x, r.rows * r.features)
            i32_ptr(r.status)[0] = word
            if word != 0:
                return 0
        return qn_fit_host_run(r.x, r.y, wp, ip, r.rows, r.features, r.classes,
            r.l1, r.l2, r.gradient_tolerance, r.change_tolerance, r.max_iter,
            r.linesearch_max_iter, r.memory, r.intercept, r.normalized,
            r.weights != 0, r.loss, r.weights, r.svr_epsilon)

    @staticmethod
    def query(operation: Int, r: QnQueryRequest) raises:
        _positive(r.rows, "n_rows")
        _positive(r.features, "n_features")
        if r.status != 0:
            i32_ptr(r.status)[0] = host_finite_status(r.x, r.rows * r.features)
        var xp = f32_ptr(r.x)
        var wp = f32_ptr(r.coefficients)
        var tasks = host_predict_task_count(r.rows)
        if operation == QN_NATIVE_BINARY:
            var dst = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=r.dst)
            host_qn_predict_binary_into(xp, wp, dst, r.rows, r.features, r.intercept, tasks)
            return
        if operation == QN_NATIVE_DECISION:
            var dst = f32_ptr(r.dst)
            if r.targets > 2:
                host_qn_decision_multi_into(xp, wp, dst, r.rows, r.features, r.targets, r.intercept, tasks)
            else:
                host_qn_decision_into(xp, wp, dst, r.rows, r.features, r.intercept, tasks)
            return
        if operation != QN_NATIVE_PROBA:
            raise Error("logistic native ABI: invalid host query operation")
        var targets = r.targets if r.targets > 2 else 1
        var scores = List[Float32](length=max(r.rows * targets, 1), fill=0)
        var sp = f32_ptr(Int(scores.unsafe_ptr()))
        var dst = f64_ptr(r.dst)
        if targets > 2:
            host_qn_decision_multi_into(xp, wp, sp, r.rows, r.features, targets, r.intercept, tasks)
            host_qn_softmax_into(sp, dst, r.rows, targets, tasks)
        else:
            host_qn_decision_into(xp, wp, sp, r.rows, r.features, r.intercept, tasks)
            host_qn_sigmoid_into(sp, dst, r.rows, tasks)
        if r.logarithm:
            var count = r.rows * (targets if targets > 2 else 2)
            host_rows_log_inplace(dst, count, host_predict_task_count(count))
        _ = scores^

    @staticmethod
    def link(operation: Int, r: QnLinkRequest) raises:
        var dst = f64_ptr(r.dst)
        if operation == QN_NATIVE_SOFTMAX and r.classes < 3:
            raise Error("qn_softmax: n_classes must be at least 3; the binary link is qn_sigmoid")
        _positive(r.rows, "n_rows")
        var tasks = host_predict_task_count(r.rows)
        if operation == QN_NATIVE_SIGMOID:
            host_qn_sigmoid_into(f32_ptr(r.scores), dst, r.rows, tasks)
        elif operation == QN_NATIVE_SOFTMAX:
            host_qn_softmax_into(f32_ptr(r.scores), dst, r.rows, r.classes, tasks)
        else:
            raise Error("logistic native ABI: invalid host link operation")

    @staticmethod
    def logarithm(source: Int, dst: Int, count: Int) raises:
        var call = rows_call(ROWS_LOG, count, 1, source, dst, 0, 0.0)
        if call.n > 0:
            rows_host(call)


def qn_host_native_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return qn_native_dispatch[QnHostKernels](operation, aa, na, pp, np, ff, nf, result, context, sink)


struct QnHostAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(LOGISTIC_NATIVE_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(qn_host_native_v1)


comptime _API_SLOT = "MojolearnLogisticHostIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnLogisticHostFastV1"
comptime LOGISTIC_HOST_API = SharedSlot[QnHostAPI, _API_SLOT]
