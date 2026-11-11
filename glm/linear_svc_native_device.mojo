# SPDX-License-Identifier: Apache-2.0
"""Same one-upload GPU OVR solver and existing opt-out per-class route."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.hostptr import f32_ptr, i32_ptr
from core.neural_context import process_ctx
from core.shared_slot import SharedSlot
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.linear_svc_native_api import (
    LINEAR_SVC_NATIVE_ABI, LinearSvcFitRequest, LinearSvcFitState,
    LinearSvcOvrKernels, linear_svc_fit_separate, linear_svc_dispatch,
)
from glm.logistic_native_device import QnDeviceKernels
from glm.estimator import QN_OVR_ONE_UPLOAD, qn_fit_ovr_host

comptime _CONTEXT = "MojolearnEstimatorsContextIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnEstimatorsContextFastV3"


struct LinearSvcDeviceKernels(LinearSvcOvrKernels):
    @staticmethod
    def fit(r: LinearSvcFitRequest, mut state: LinearSvcFitState) raises:
        comptime if QN_OVR_ONE_UPLOAD:
            var xp = f32_ptr(r.source)
            var cp = i32_ptr(r.codes)
            var wp = f32_ptr(Int(state.coefficients.unsafe_ptr()))
            var ip = f32_ptr(Int(state.info.unsafe_ptr()))
            var sw = f32_ptr(r.weights) if r.weights != 0 else wp
            var o = r.options
            var ctx = process_ctx[_CONTEXT]()
            qn_fit_ovr_host(ctx, xp, cp, wp, ip, o.rows, o.features, o.first_class,
                o.fits, o.l1, o.l2, o.gradient_tolerance, o.change_tolerance,
                o.max_iter, o.linesearch_max_iter, o.memory, o.intercept,
                o.normalized, o.loss, r.weights != 0, sw)
            ctx.synchronize()
            for fit in range(o.fits):  # metadata: original Float32 OVR iteration result per fit
                state.max_iterations = max(state.max_iterations, Int(state.info[3 * fit + 2]))
        else:
            linear_svc_fit_separate[QnDeviceKernels](r, state)


def linear_svc_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return linear_svc_dispatch[QnDeviceKernels, LinearSvcDeviceKernels](
        operation, aa, na, pp, np, ff, nf, result, context, sink)


struct LinearSvcDeviceAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(LINEAR_SVC_NATIVE_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(linear_svc_device_v1)


comptime _API = "MojolearnLinearSvcDeviceIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnLinearSvcDeviceFastV1"
comptime LINEAR_SVC_DEVICE_API = SharedSlot[LinearSvcDeviceAPI, _API]
