# SPDX-License-Identifier: Apache-2.0
# Compile-only root for the actual device descriptor. Parent owns compilation;
# do not execute this export as a product/runtime test.
from core.native_descriptor_abi import NativeDescriptorError
from hierarchy.cluster_descriptor_device import hierarchy_cluster_device_v1


@export
def peer1_hierarchy_cluster_device_compile(operation: Int, aa: Int, na: Int,
                                           pp: Int, np: Int, ff: Int, nf: Int,
                                           result: Int, context: Int,
                                           sink: NativeDescriptorError) abi("C") -> Int:
    return hierarchy_cluster_device_v1(operation, aa, na, pp, np, ff, nf, result, context, sink)
