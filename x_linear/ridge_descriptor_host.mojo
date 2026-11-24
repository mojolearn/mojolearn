# SPDX-License-Identifier: Apache-2.0
"""The host column of the native weighted/one-feature Ridge fit request."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.linear_fit_types import (
    RIDGE_FIT_ABI, ridge_fit_check_spans, ridge_fit_targets, ridge_fit_result_words, ridge_fit_work_words,
)
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot
from x_linear.dispatch import ALGO_RIDGE
from x_linear.x_linear_host_impl import x_linear_fit_host


def ridge_fit_host_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        ridge_fit_check_spans(operation, aa, na, pp, np, ff, nf, result)
        var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
        var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
        var f = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        var rows = p[0]
        var cols = p[1]
        # Lane w3f3-k: T targets (y rows x T, one Cholesky factor for every
        # target's right-hand side), the same words for T == 1.
        var targets = ridge_fit_targets(np, pp)
        var dims: List[Int] = [rows, cols, rows * cols, rows * (targets + (1 if p[3] else 0)),
            ridge_fit_result_words(targets, cols), ridge_fit_work_words(targets, cols, rows), 1, 4, 1]
        var ip: List[Int32] = [Int32(targets), Int32(p[2]), 1, Int32(p[3])]
        var fp: List[Float32] = [Float32(f[0])]
        var value = x_linear_fit_host(ALGO_RIDGE, a[0], a[1], dims, ip, fp, a[2])
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct RidgeFitHostAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(RIDGE_FIT_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(ridge_fit_host_v1)


comptime _API_SLOT = "MojolearnRidgeFitHostIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnRidgeFitHostFastV1"
comptime RIDGE_FIT_HOST_API = SharedSlot[RidgeFitHostAPI, _API_SLOT]
