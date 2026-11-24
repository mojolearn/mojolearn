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
    MP_RUN_OUT, MP_RUN_RANGES, MP_HAS_RANGES, MP_HAS_STORE, MP_DEV_FIRST_NONFINITE, MP_DEV_FIRST_NEGATIVE,
    metrics_check_spans,
)
from core.metric_labels_device import metric_first_nonfinite_resident, metric_first_negative_resident
from x_metrics.device import run_program_device, run_program_device_out, run_program_device_ranges, metrics_ctx
from x_prep.store_slot import X_PREP_STORE


def metrics_program_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        metrics_check_spans(operation, aa, na, pp, np, nf, result)
        # TOMBSTONE: MP_DEV_PUT / _FREE / _TAKE_ROWS / _LIVE (a second resident store, x_metrics-only; lane
        # vdf-xm-transport) deleted 2026-11-24 by lane vdf-xm-wire: the one store is the x_prep image's, reached
        # through its transport entry (XP_DEV_*). Never on main: recoverable at
        # lane/vd-functions 55fed275ea (no patch, AGENTS.md "Deleting an experiment" part 3).
        var value = 0
        if operation == MP_MODE:
            value = GLOBAL_NUMERIC_MODE
        elif operation == MP_VENDOR:
            value = xd_vendor_code(String(COMPILED_VENDOR))
        elif operation == MP_CAPABILITIES:
            value = MP_HAS_RANGES | MP_HAS_STORE
        elif operation == MP_DEV_FIRST_NONFINITE or operation == MP_DEV_FIRST_NEGATIVE:
            # A door's refusal scan over a resident slot, in place (one
            # 4-byte readback): no word of the slot comes to the host.
            var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
            if p[1] == 0:
                value = -1
            elif operation == MP_DEV_FIRST_NONFINITE:
                value = metric_first_nonfinite_resident(
                    metrics_ctx(), Int(X_PREP_STORE.get_or_create_ptr()[].ptr(p[0], p[1])), p[1])
            else:
                value = metric_first_negative_resident(
                    metrics_ctx(), Int(X_PREP_STORE.get_or_create_ptr()[].ptr(p[0], p[1])), p[1])
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
