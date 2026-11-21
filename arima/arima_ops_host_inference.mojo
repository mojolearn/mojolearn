# SPDX-License-Identifier: Apache-2.0
"""The host prediction, forecast and the fixed answers of the ARIMA
operations table (core/arima_entry_table.mojo) that the two CPU images'
entries share (arima/arima_ops_host.mojo, arima/arima_ops_forecast_host.mojo).
No fit is imported here. One request is one prediction; it has no
independent units to split.
"""
from arima.host.arima_host_predict_run import arima_forecast_run, arima_predict_run
from core.arima_entry_table import (
    ARIMA_OP_PREDICT, ARIMA_OP_FORECAST, ARIMA_OP_RESIDENT_COUNT, ARIMA_OP_FAST_SEARCH_MODE,
    ARIMA_OP_SEARCH_SWITCHES,
)
from core.hostptr import f32_ptr


def arima_ops_host_inference(operation: Int, aa: Int, pp: Int, result: Int) raises -> Bool:
    """The prediction and the forecast on the host, and the answers every
    CPU image gives (no resident copy, no fast search arm, no grouped
    search); False when `operation` is none of them (the caller dispatches or
    refuses it). The spans were checked by the caller."""
    var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
    var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
    var r = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)
    if operation == ARIMA_OP_PREDICT:
        # Every address is live; the exog pair is read only when n_exog > 0.
        _ = f32_ptr(a[0])
        _ = f32_ptr(a[1])
        _ = f32_ptr(a[2])
        _ = f32_ptr(a[3])
        r[] = arima_predict_run(a[0], a[1], a[2], a[3], f32_ptr(a[4]), p[0], p[1], p[2], p[3],
            p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12])
        return True
    if operation == ARIMA_OP_FORECAST:
        _ = f32_ptr(a[0])
        _ = f32_ptr(a[1])
        _ = f32_ptr(a[2])
        _ = f32_ptr(a[3])
        r[] = arima_forecast_run(a[0], a[1], a[2], a[3], f32_ptr(a[4]), p[0], p[1], p[2],
            p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11])
        return True
    if (operation == ARIMA_OP_RESIDENT_COUNT or operation == ARIMA_OP_FAST_SEARCH_MODE
            or operation == ARIMA_OP_SEARCH_SWITCHES):
        r[] = 0
        return True
    return False


def arima_ops_host_absent(operation: Int) raises:
    """Refuse an operation no CPU image holds, by the name the former
    by-name lookup refused it with."""
    raise Error("AttributeError: the ARIMA binding has no native operation " + String(operation)
                + " on the CPU (no resident device copy, no grouped order search)")
