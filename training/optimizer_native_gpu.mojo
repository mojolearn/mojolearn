# SPDX-License-Identifier: Apache-2.0
"""The gpu optimizer descriptor; typed dispatch preserves existing kernels."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.shared_slot import SharedSlot
from training.optimizer_kernels_gpu import GpuOptKernels
from training.optimizer_native_api import OPTIMIZER_NATIVE_ABI, optimizer_native_dispatch


def optimizer_gpu_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return optimizer_native_dispatch[GpuOptKernels](operation, aa, na, pp, np, ff, nf, result, context, sink)


struct OptimizerGpuAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(OPTIMIZER_NATIVE_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(optimizer_gpu_v1)


comptime _API_SLOT = "MojolearnOptimizerGpuIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnOptimizerGpuFastV1"
comptime OPTIMIZER_NATIVE_GPU_API = SharedSlot[OptimizerGpuAPI, _API_SLOT]
