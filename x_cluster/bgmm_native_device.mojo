# SPDX-License-Identifier: Apache-2.0
from core.shared_slot import SharedSlot
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.bgmm_native_api import BGMM_NATIVE_ABI
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_cluster.variants_device import VariantsDeviceBackend
from x_cluster.bgmm_native_dispatch import bgmm_native_dispatch


def bgmm_native_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return bgmm_native_dispatch[VariantsDeviceBackend](operation, aa, na, pp, np, ff, nf, result, context, sink)


struct BgmmNativeDeviceAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(BGMM_NATIVE_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(bgmm_native_device_v1)


comptime BGMM_NATIVE_DEVICE_API = SharedSlot[BgmmNativeDeviceAPI,
    ("MojolearnBgmmNativeDeviceIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnBgmmNativeDeviceFastV1")]
