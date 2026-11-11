# SPDX-License-Identifier: Apache-2.0
"""Original per-class host QN solver behind typed LinearSVC requests."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.shared_slot import SharedSlot
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.linear_svc_native_api import (
    LINEAR_SVC_NATIVE_ABI, LinearSvcFitRequest, LinearSvcFitState,
    LinearSvcOvrKernels, linear_svc_fit_separate, linear_svc_dispatch,
)
from glm.host.logistic_native_host import QnHostKernels


struct LinearSvcHostKernels(LinearSvcOvrKernels):
    @staticmethod
    def fit(r: LinearSvcFitRequest, mut state: LinearSvcFitState) raises:
        linear_svc_fit_separate[QnHostKernels](r, state)


def linear_svc_host_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return linear_svc_dispatch[QnHostKernels, LinearSvcHostKernels](
        operation, aa, na, pp, np, ff, nf, result, context, sink)


struct LinearSvcHostAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(LINEAR_SVC_NATIVE_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(linear_svc_host_v1)


comptime _API = "MojolearnLinearSvcHostIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnLinearSvcHostFastV1"
comptime LINEAR_SVC_HOST_API = SharedSlot[LinearSvcHostAPI, _API]
