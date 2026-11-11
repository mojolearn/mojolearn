# SPDX-License-Identifier: Apache-2.0
"""Core host image's typed label descriptor; the existing host kernels."""
from core.labels_native_api import LABELS_NATIVE_ABI_NAME, labels_native_dispatch
from core.labels_native_host import LabelsHostKernels
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError


def labels_host_native_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return labels_native_dispatch[LabelsHostKernels](operation, aa, na, pp, np, ff, nf, result, context, sink)


struct LabelsHostNativeAPI[ABI: StaticString](Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(Self.ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(labels_host_native_v2)
