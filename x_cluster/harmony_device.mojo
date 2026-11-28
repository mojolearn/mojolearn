# SPDX-License-Identifier: Apache-2.0
"""The GPU binding's Harmony table (`harmony_native_api_v1`, lane sc-stats-1): the fit on DeviceOps."""
from core.shared_slot import SharedSlot
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.harmony_api import HARMONY_ABI, HARMONY_MODE
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_cluster.device_ops import DeviceOps
from x_cluster.harmony_dispatch import harmony_check, harmony_answer, harmony_run


def harmony_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        harmony_check(operation, aa, na, pp, np, ff, nf, result)
        if operation == HARMONY_MODE:
            harmony_answer(result, Int(GLOBAL_NUMERIC_MODE))
            return 0
        var ops = DeviceOps()
        harmony_run(ops, aa, pp, ff)
        _ = ops^
        harmony_answer(result, 0)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct HarmonyDeviceAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(HARMONY_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(harmony_device_v1)


comptime HARMONY_DEVICE_API = SharedSlot[HarmonyDeviceAPI,
    ("MojolearnHarmonyDeviceIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnHarmonyDeviceFastV1")]
