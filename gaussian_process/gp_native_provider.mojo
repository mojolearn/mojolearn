# SPDX-License-Identifier: Apache-2.0
"""Image-owned GP descriptor table for the existing provider implementations."""
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.gp_native_api import GP_NATIVE_ABI, GPNativeKernels, gp_native_dispatch


def gp_native_v1[K: GPNativeKernels](operation: Int,
    aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int, result: Int,
    context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return gp_native_dispatch[K](operation, aa, na, pp, np, ff, nf, result, context, sink)


struct GPProviderAPI[K: GPNativeKernels](Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(GP_NATIVE_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(gp_native_v1[Self.K])
