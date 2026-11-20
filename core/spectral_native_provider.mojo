# SPDX-License-Identifier: Apache-2.0
"""Image-owned spectral descriptor entry over the established numerical providers."""
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.spectral_native_api import SPECTRAL_NATIVE_ABI, SpectralNativeKernels, spectral_native_dispatch


def spectral_native_v1[K: SpectralNativeKernels](operation: Int,
    aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int, result: Int,
    context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return spectral_native_dispatch[K](operation, aa, na, pp, np, ff, nf, result, context, sink)


struct SpectralProviderAPI[K: SpectralNativeKernels](Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(SPECTRAL_NATIVE_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(spectral_native_v1[Self.K])
