# SPDX-License-Identifier: Apache-2.0
"""The forecast inference image's column of the ARIMA operations table
(core/arima_entry_table.mojo, family ARIMA_TABLE_FORECAST): the entry
`_mojolearn_forecast_host` publishes. Prediction and forecast from fitted
parameters; no fit is linked into this image and ARIMA_OP_FIT is refused by
name.

One request is one prediction; it has no independent units to split here.
"""
from arima.arima_ops_host_inference import arima_ops_host_absent, arima_ops_host_inference
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.arima_entry_table import (
    ARIMA_OP_FLAGS, ARIMA_OP_VENDOR, ARIMA_OP_MODE, ARIMA_OP_FIT, ARIMA_OP_ORDER_CAPS,
    ARIMA_TABLE_FORECAST, arima_ops_check_spans, arima_ops_publish,
)
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot


def arima_ops_forecast_host_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        arima_ops_check_spans(operation, aa, na, pp, np, ff, nf, result)
        var r = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)
        if operation == ARIMA_OP_FLAGS:
            r[] = 0
        elif operation == ARIMA_OP_VENDOR:
            r[] = 0
        elif operation == ARIMA_OP_MODE:
            r[] = Int(GLOBAL_NUMERIC_MODE)
        elif operation == ARIMA_OP_ORDER_CAPS:
            r[] = 0
        elif operation == ARIMA_OP_FIT:
            raise Error("AttributeError: the ARIMA binding has no arima_fit")
        elif not arima_ops_host_inference(operation, aa, pp, result):
            arima_ops_host_absent(operation)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct ArimaOpsForecastHostAPI(Defaultable, Movable):
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(arima_ops_forecast_host_v1)


# The publisher identity of this image's table (its entry list's address).
comptime ARIMA_OPS_FORECAST_HOST_API = SharedSlot[ArimaOpsForecastHostAPI, "MojolearnArimaOpsForecastHostIdenticalV1"]


def publish_arima_ops_forecast_host() raises:
    """Publish this image's table; called once, after the module is ready."""
    var api = ARIMA_OPS_FORECAST_HOST_API.get_or_create_ptr()
    arima_ops_publish[ARIMA_TABLE_FORECAST](Int(GLOBAL_NUMERIC_MODE), 0, Int(api[].entries.unsafe_ptr()),
        arima_ops_forecast_host_v1)
