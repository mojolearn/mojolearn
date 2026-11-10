# SPDX-License-Identifier: Apache-2.0
"""Native metrics program entry for the existing host executor."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot
from x_metrics.descriptor_schema import (
    METRICS_PROGRAM_ABI, MP_MODE, MP_VENDOR, MP_CAPABILITIES, MP_RUN,
    MP_RUN_OUT, MP_RUN_RANGES, metrics_check_spans,
)
from x_metrics.host.program import run_program_host
from x_metrics.host.out_ranges_impl import check_out_ranges


def metrics_program_host_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        metrics_check_spans(operation, aa, na, pp, np, nf, result)
        var value = 0
        if operation == MP_MODE:
            value = GLOBAL_NUMERIC_MODE
        elif operation == MP_VENDOR or operation == MP_CAPABILITIES:
            value = 0
        else:
            var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
            var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
            if operation == MP_RUN_RANGES:
                raise Error("x_metrics: the host column has no range upload route")
            if operation == MP_RUN_OUT:
                check_out_ranges(a[2], p[2], p[0])
            run_program_host(a[0], p[0], a[1], p[1])
            value = p[1]
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct MetricsProgramHostAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(METRICS_PROGRAM_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(metrics_program_host_v1)


comptime _API_SLOT = "MojolearnMetricsProgramHostIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnMetricsProgramHostFastV1"
comptime METRICS_PROGRAM_HOST_API = SharedSlot[MetricsProgramHostAPI, _API_SLOT]
