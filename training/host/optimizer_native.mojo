# SPDX-License-Identifier: Apache-2.0
"""The host optimizer descriptor; typed dispatch preserves existing kernels."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.shared_slot import SharedSlot
from training.host.optimizer_kernels import HostOptKernels
from training.optimizer_native_api import OPTIMIZER_NATIVE_ABI, optimizer_native_dispatch


def optimizer_host_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return optimizer_native_dispatch[HostOptKernels](operation, aa, na, pp, np, ff, nf, result, context, sink)


struct OptimizerHostAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(OPTIMIZER_NATIVE_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(optimizer_host_v1)


comptime _API_SLOT = "MojolearnOptimizerHostIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnOptimizerHostFastV1"
comptime OPTIMIZER_NATIVE_HOST_API = SharedSlot[OptimizerHostAPI, _API_SLOT]
