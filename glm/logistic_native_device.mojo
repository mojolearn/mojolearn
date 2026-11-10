# SPDX-License-Identifier: Apache-2.0
"""Existing QN device operations behind typed requests; no Python execution."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.hostptr import f32_ptr, f64_ptr
from core.logistic_native_api import (
    LOGISTIC_NATIVE_ABI, QN_NATIVE_COUNT, QN_NATIVE_DECISION, QN_NATIVE_BINARY,
    QN_NATIVE_MULTICLASS, QN_NATIVE_PROBA, QN_NATIVE_SIGMOID, QN_NATIVE_SOFTMAX,
    QnFitRequest, QnQueryRequest, QnLinkRequest, QnNativeKernels, qn_native_dispatch,
)
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.neural_context import process_ctx
from core.shared_slot import SharedSlot
from core.estimators_impl import qn_link_from_device, qn_sigmoid_device, qn_softmax_device
from core.py2mojo_rows import ROWS_LOG, rows_call, rows_device
from glm.estimator import (
    qn_fit_host, qn_decision_function_device, qn_decision_function_host,
    qn_predict_binary_host, qn_predict_multiclass_host, QN_DEV_ARGMAX,
)

comptime _CONTEXT = "MojolearnEstimatorsContextIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnEstimatorsContextFastV3"


struct QnDeviceKernels(QnNativeKernels):
    @staticmethod
    def mode() -> Int:
        return Int(GLOBAL_NUMERIC_MODE)

    @staticmethod
    def host() -> Bool:
        return False

    @staticmethod
    def multiclass() -> Bool:
        return Bool(QN_DEV_ARGMAX)

    @staticmethod
    def fit(r: QnFitRequest) raises -> Int:
        var xp = f32_ptr(r.x)
        var yp = f32_ptr(r.y)
        var wp = f32_ptr(r.coefficients)
        var ip = f32_ptr(r.info)
        var sw = f32_ptr(r.weights) if r.weights != 0 else yp
        var ctx = process_ctx[_CONTEXT]()
        var iterations = qn_fit_host(ctx, xp, yp, wp, ip, r.rows, r.features,
            r.classes, r.l1, r.l2, r.gradient_tolerance, r.change_tolerance,
            r.max_iter, r.linesearch_max_iter, r.memory, r.intercept,
            r.normalized, r.weights != 0, sw, r.loss, r.svr_epsilon, r.status)
        ctx.synchronize()
        return iterations

    @staticmethod
    def query(operation: Int, r: QnQueryRequest) raises:
        var xp = f32_ptr(r.x)
        var wp = f32_ptr(r.coefficients)
        var ctx = process_ctx[_CONTEXT]()
        if operation == QN_NATIVE_PROBA:
            var dst = f64_ptr(r.dst)
            var targets = r.targets if r.targets > 2 else 1
            # Preserve the measured fused path: scores stay on the device.
            var scores = ctx.enqueue_create_buffer[DType.float32](max(r.rows * targets, 1))
            qn_decision_function_device(ctx, xp, wp, scores, r.rows,
                r.features, r.intercept, targets, r.status)
            qn_link_from_device(ctx, scores, dst, r.rows, targets, r.logarithm)
            _ = scores^
            _ = ctx^
            return
        if operation == QN_NATIVE_DECISION:
            qn_decision_function_host(ctx, xp, wp, f32_ptr(r.dst), r.rows,
                r.features, r.intercept, r.targets, r.status)
        elif operation == QN_NATIVE_BINARY:
            var dst = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=r.dst)
            qn_predict_binary_host(ctx, xp, wp, dst, r.rows, r.features, r.intercept, r.status)
        elif operation == QN_NATIVE_MULTICLASS:
            comptime if QN_DEV_ARGMAX:
                var dst = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=r.dst)
                qn_predict_multiclass_host(ctx, xp, wp, dst, r.rows,
                    r.features, r.intercept, r.targets, r.status)
            else:
                raise Error("logistic native ABI: multiclass prediction is not enabled")
        else:
            raise Error("logistic native ABI: invalid query operation")
        ctx.synchronize()

    @staticmethod
    def link(operation: Int, r: QnLinkRequest) raises:
        var src = f32_ptr(r.scores)
        var dst = f64_ptr(r.dst)
        if operation == QN_NATIVE_SOFTMAX and r.classes < 3:
            raise Error("qn_softmax: n_classes must be at least 3; the binary link is qn_sigmoid")
        if r.rows <= 0:
            return
        var ctx = process_ctx[_CONTEXT]()
        if operation == QN_NATIVE_SIGMOID:
            qn_sigmoid_device(ctx, src, dst, r.rows)
        elif operation == QN_NATIVE_SOFTMAX:
            qn_softmax_device(ctx, src, dst, r.rows, r.classes)
        else:
            raise Error("logistic native ABI: invalid link operation")
        _ = ctx^

    @staticmethod
    def logarithm(source: Int, dst: Int, count: Int) raises:
        var call = rows_call(ROWS_LOG, count, 1, source, dst, 0, 0.0)
        var ctx = process_ctx[_CONTEXT]()
        rows_device(ctx, call)


def qn_device_native_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return qn_native_dispatch[QnDeviceKernels](operation, aa, na, pp, np, ff, nf, result, context, sink)


struct QnDeviceAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(LOGISTIC_NATIVE_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(qn_device_native_v1)


comptime _API_SLOT = "MojolearnLogisticDeviceIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnLogisticDeviceFastV1"
comptime LOGISTIC_DEVICE_API = SharedSlot[QnDeviceAPI, _API_SLOT]
