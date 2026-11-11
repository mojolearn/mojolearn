# SPDX-License-Identifier: Apache-2.0
from core.shared_slot import SharedSlot
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.cluster_centroid_fit_api import CENTROID_FIT_ABI
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_cluster.variants_host import VariantsHostBackend
from x_cluster.centroid_fit_dispatch import centroid_fit_dispatch


def cluster_centroid_fit_host_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return centroid_fit_dispatch[VariantsHostBackend](operation, aa, na, pp, np, ff, nf, result, context, sink)


struct CentroidFitHostAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(CENTROID_FIT_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(cluster_centroid_fit_host_v1)


comptime CENTROID_FIT_HOST_API = SharedSlot[CentroidFitHostAPI,
    ("MojolearnCentroidFitHostIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnCentroidFitHostFastV1")]
