# SPDX-License-Identifier: Apache-2.0
"""The existing narrow and wide prediction kernels behind one typed request."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.hostptr import f32_ptr, f64_ptr
from core.linear_boundary_types import LINEAR_PREDICT_ABI, LinearPredictRequest, linear_predict_request_from_spans
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.neural_context import process_ctx
from core.p3_linear_impl import ols_predict_wide_device
from core.shared_slot import SharedSlot
from glm.estimator import ols_predict_host

comptime _CONTEXT = "MojolearnEstimatorsContextIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnEstimatorsContextFastV3"


def linear_predict_device(request: LinearPredictRequest) raises:
    var ctx = process_ctx[_CONTEXT]()
    var intercept = Float32(request.intercept)
    if request.wide:
        ols_predict_wide_device(ctx, f32_ptr(request.x), f32_ptr(request.coef),
            f64_ptr(request.dst), request.rows, request.cols, intercept)
    else:
        ols_predict_host(ctx, f32_ptr(request.x), f32_ptr(request.coef),
            f32_ptr(request.dst), request.rows, request.cols, intercept)
        ctx.synchronize()


def linear_predict_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        var request = linear_predict_request_from_spans(operation, aa, na, pp, np, ff, nf, result)
        linear_predict_device(request)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = 0
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct LinearPredictDeviceAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(LINEAR_PREDICT_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(linear_predict_device_v1)


comptime _API_SLOT = "MojolearnLinearPredictDeviceIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnLinearPredictDeviceFastV1"
comptime LINEAR_PREDICT_DEVICE_API = SharedSlot[LinearPredictDeviceAPI, _API_SLOT]
