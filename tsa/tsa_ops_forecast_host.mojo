# SPDX-License-Identifier: Apache-2.0
"""The forecast inference image's column of the Holt-Winters operations
table (core/tsa_entry_table.mojo, family TSA_TABLE_FORECAST): the entry
`_mojolearn_forecast_host` publishes. Forecast and in-sample prediction from
fitted components; no fit is linked into this image and TSA_OP_HW_FIT is
refused by name.

One request is one forecast; it has no independent units to split here.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot
from core.tsa_entry_table import (
    TSA_OP_FLAGS, TSA_OP_VENDOR, TSA_OP_MODE, TSA_OP_HW_FIT, TSA_OP_HW_RESIDENT_COUNT,
    TSA_TABLE_FORECAST, tsa_ops_check_spans, tsa_ops_publish,
)
from tsa.tsa_ops_host_inference import tsa_hw_ops_host_inference


def tsa_hw_ops_forecast_host_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        tsa_ops_check_spans(operation, aa, na, pp, np, ff, nf, result)
        var r = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)
        if operation == TSA_OP_FLAGS:
            r[] = 0
        elif operation == TSA_OP_VENDOR:
            r[] = 0
        elif operation == TSA_OP_MODE:
            r[] = Int(GLOBAL_NUMERIC_MODE)
        elif operation == TSA_OP_HW_FIT:
            raise Error("holtwinters_fit: the forecast inference binding holds no fit")
        elif operation == TSA_OP_HW_RESIDENT_COUNT:
            r[] = 0
        elif not tsa_hw_ops_host_inference(operation, aa, pp, ff, result):
            raise Error("tsa ops: the CPU binding keeps no resident device copy (operation " + String(operation) + ")")
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct TsaHwOpsForecastHostAPI(Defaultable, Movable):
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(tsa_hw_ops_forecast_host_v1)


# The publisher identity of this image's table (its entry list's address).
comptime TSA_HW_OPS_FORECAST_HOST_API = SharedSlot[TsaHwOpsForecastHostAPI, "MojolearnTsaHwOpsForecastHostIdenticalV1"]


def publish_tsa_hw_ops_forecast_host() raises:
    """Publish this image's table; called once, after the module is ready."""
    var api = TSA_HW_OPS_FORECAST_HOST_API.get_or_create_ptr()
    tsa_ops_publish[TSA_TABLE_FORECAST](Int(GLOBAL_NUMERIC_MODE), 0, Int(api[].entries.unsafe_ptr()),
        tsa_hw_ops_forecast_host_v1)
