# SPDX-License-Identifier: Apache-2.0
"""The host forecast and in-sample prediction of the Holt-Winters operations
table (core/tsa_entry_table.mojo), shared by the two CPU images' entries
(tsa/tsa_ops_host.mojo, tsa/tsa_ops_forecast_host.mojo). No fit is imported
here. One request is one forecast; it has no independent units to split.
"""
from bindings.holtwinters_host_predict_run import holtwinters_forecast_run, holtwinters_predict_run
from core.hostptr import f32_ptr
from core.tsa_entry_table import TSA_OP_HW_FORECAST, TSA_OP_HW_PREDICT
from holtwinters.impl.tsa.holtwinters_params import seasonal_name


def tsa_hw_ops_host_inference(operation: Int, aa: Int, pp: Int, result: Int) raises -> Bool:
    """The forecast and in-sample prediction on the host; False when
    `operation` is neither (the caller dispatches or refuses it). The spans
    were checked by the caller."""
    var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
    var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
    var r = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)
    if operation == TSA_OP_HW_FORECAST:
        r[] = holtwinters_forecast_run(f32_ptr(a[0]), f32_ptr(a[1]), p[0], p[1], p[2], p[3], seasonal_name(p[4]))
        return True
    if operation == TSA_OP_HW_PREDICT:
        r[] = holtwinters_predict_run(f32_ptr(a[0]), f32_ptr(a[1]), p[0], p[1], p[2], p[3], p[4], seasonal_name(p[5]))
        return True
    return False
