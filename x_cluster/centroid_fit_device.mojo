# SPDX-License-Identifier: Apache-2.0
from core.shared_slot import SharedSlot
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.cluster_centroid_fit_api import CENTROID_FIT_ABI
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_cluster.variants_device import VariantsDeviceBackend
from x_cluster.centroid_fit_dispatch import centroid_fit_dispatch


def cluster_centroid_fit_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return centroid_fit_dispatch[VariantsDeviceBackend](operation, aa, na, pp, np, ff, nf, result, context, sink)


struct CentroidFitDeviceAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(CENTROID_FIT_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(cluster_centroid_fit_device_v1)


comptime CENTROID_FIT_DEVICE_API = SharedSlot[CentroidFitDeviceAPI,
    ("MojolearnCentroidFitDeviceIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnCentroidFitDeviceFastV1")]
