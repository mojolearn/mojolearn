# SPDX-License-Identifier: Apache-2.0
"""Native metrics program entry, sharing the existing GPU execution routes."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.vendor import COMPILED_VENDOR
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot
from x_decomp.descriptor_schema import xd_vendor_code
from x_metrics.descriptor_schema import (
    METRICS_PROGRAM_ABI, MP_MODE, MP_VENDOR, MP_CAPABILITIES, MP_RUN,
    MP_RUN_OUT, MP_RUN_RANGES, MP_HAS_RANGES, metrics_check_spans,
)
from x_metrics.device import run_program_device, run_program_device_out, run_program_device_ranges


def metrics_program_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        metrics_check_spans(operation, aa, na, pp, np, nf, result)
        var value = 0
        if operation == MP_MODE:
            value = GLOBAL_NUMERIC_MODE
        elif operation == MP_VENDOR:
            value = xd_vendor_code(String(COMPILED_VENDOR))
        elif operation == MP_CAPABILITIES:
            value = MP_HAS_RANGES
        else:
            var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
            var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
            if operation == MP_RUN:
                run_program_device(a[0], p[0], a[1], p[1])
            elif operation == MP_RUN_OUT:
                run_program_device_out(a[0], p[0], a[1], p[1], a[2], p[2])
            elif operation == MP_RUN_RANGES:
                run_program_device_ranges(a[0], p[0], a[1], p[1], a[2], p[2], a[3], p[3])
            value = p[1]
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct MetricsProgramDeviceAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(METRICS_PROGRAM_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(metrics_program_device_v1)


comptime _API_SLOT = "MojolearnMetricsProgramDeviceIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnMetricsProgramDeviceFastV1"
comptime METRICS_PROGRAM_DEVICE_API = SharedSlot[MetricsProgramDeviceAPI, _API_SLOT]
