# SPDX-License-Identifier: Apache-2.0
"""The CPU columns of the Holt-Winters operations table
(core/tsa_entry_table.mojo).

`tsa_hw_ops_host_v1` is the entry `_mojolearn_tsa_host` publishes (the fit,
the forecast and the in-sample prediction; no resident device copy).
The forecast and prediction body is `tsa_hw_ops_host_inference`
(tsa/tsa_ops_host_inference.mojo), which the forecast inference image's
entry (tsa/tsa_ops_forecast_host.mojo) calls too, so that image links no fit.

Each operation is the body of the former Python export of the same name with
its argument words read from the request spans. One request is one fit or
one forecast; it has no independent units to split here.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.hostptr import f32_ptr, i32_ptr
from core.mojolearn_tsa_host_impl import holtwinters_fit_host_into
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot
from core.tsa_entry_table import (
    TSA_OP_FLAGS, TSA_OP_VENDOR, TSA_OP_MODE, TSA_OP_HW_FIT, TSA_OP_HW_RESIDENT_COUNT, TSA_HAS_FIT, TSA_TABLE_TSA, tsa_ops_check_spans, tsa_ops_publish,
)
from holtwinters.impl.tsa.holtwinters_params import seasonal_name
from tsa.tsa_ops_host_inference import tsa_hw_ops_host_inference


def tsa_hw_ops_host_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        tsa_ops_check_spans(operation, aa, na, pp, np, ff, nf, result)
        var r = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)
        if operation == TSA_OP_FLAGS:
            r[] = TSA_HAS_FIT
        elif operation == TSA_OP_VENDOR:
            r[] = 0
        elif operation == TSA_OP_MODE:
            r[] = Int(GLOBAL_NUMERIC_MODE)
        elif operation == TSA_OP_HW_FIT:
            var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
            var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
            var f = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
            if p[4] != 0 and p[4] != 1:
                raise Error("holtwinters_fit: init_method must be 0 (heuristic) or 1 (estimated), got " + String(p[4]))
            r[] = holtwinters_fit_host_into(a[0], p[0], p[1], p[2], p[3], Float32(f[0]), p[4], seasonal_name(p[5]),
                f32_ptr(a[1]), f32_ptr(a[2]), i32_ptr(a[3]))
        elif operation == TSA_OP_HW_RESIDENT_COUNT:
            r[] = 0
        elif not tsa_hw_ops_host_inference(operation, aa, pp, result):
            raise Error("tsa ops: the CPU binding keeps no resident device copy (operation " + String(operation) + ")")
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct TsaHwOpsHostAPI(Defaultable, Movable):
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(tsa_hw_ops_host_v1)


comptime _OPS_SLOT = "MojolearnTsaHwOpsHostIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnTsaHwOpsHostFastV1"
# The publisher identity of this image's table (its entry list's address).
comptime TSA_HW_OPS_HOST_API = SharedSlot[TsaHwOpsHostAPI, _OPS_SLOT]


def publish_tsa_hw_ops_host() raises:
    """Publish this image's table; called once, after the module is ready."""
    var api = TSA_HW_OPS_HOST_API.get_or_create_ptr()
    tsa_ops_publish[TSA_TABLE_TSA](Int(GLOBAL_NUMERIC_MODE), 0, Int(api[].entries.unsafe_ptr()), tsa_hw_ops_host_v1)
