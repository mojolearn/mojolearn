# SPDX-License-Identifier: Apache-2.0
"""The host prediction column of the typed linear request, with its existing kernel."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.classical_ridge_impl import classical_widen_f32_f64
from core.hostptr import f32_ptr, f64_ptr
from core.host_predict_threads import host_predict_task_count
from core.estimators_host_impl import _positive
from core.host_finite_scan import host_all_finite_f32
from core.classical_host_predict import host_ols_predict_into
from core.linear_boundary_types import LINEAR_PREDICT_ABI, LinearPredictRequest, linear_predict_request_from_spans
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot


def linear_predict_host(request: LinearPredictRequest) raises -> Int:
    """The request's predictions; with `request.test_x` X is tested first
    (the parallel SIMD scan) and 1 is returned, nothing written, when any
    word is NaN or infinite. Else 0."""
    _positive(request.rows, "n_rows")
    _positive(request.cols, "n_features")
    if request.test_x and not host_all_finite_f32(f32_ptr(request.x), request.rows * request.cols):
        return 1
    var intercept = Float32(request.intercept)
    if request.wide:
        var narrow = List[Float32](length=max(request.rows, 1), fill=0)
        var np = f32_ptr(Int(narrow.unsafe_ptr()))
        host_ols_predict_into(f32_ptr(request.x), f32_ptr(request.coef), np,
            request.rows, request.cols, intercept, host_predict_task_count(request.rows))
        classical_widen_f32_f64(Int(np), request.dst, request.rows)
        _ = narrow^
    else:
        host_ols_predict_into(f32_ptr(request.x), f32_ptr(request.coef), f32_ptr(request.dst),
            request.rows, request.cols, intercept, host_predict_task_count(request.rows))
    return 0


def linear_predict_host_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        var request = linear_predict_request_from_spans(operation, aa, na, pp, np, ff, nf, result)
        var status = linear_predict_host(request)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = status
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct LinearPredictHostAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(LINEAR_PREDICT_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(linear_predict_host_v1)


comptime _API_SLOT = "MojolearnLinearPredictHostIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnLinearPredictHostFastV1"
comptime LINEAR_PREDICT_HOST_API = SharedSlot[LinearPredictHostAPI, _API_SLOT]
