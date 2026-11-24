# SPDX-License-Identifier: Apache-2.0
"""The host forecast and in-sample prediction of the Holt-Winters operations
table (core/tsa_entry_table.mojo), shared by the two CPU images' entries
(tsa/tsa_ops_host.mojo, tsa/tsa_ops_forecast_host.mojo). No fit is imported
here. One request is one forecast; it has no independent units to split.
"""
from bindings.holtwinters_host_predict_run import holtwinters_forecast_run, holtwinters_predict_run
from core.hostptr import f32_ptr, i32_ptr
from core.kpss_host_test_impl import _kpss_flags_host, kpss_select_d_host
from core.tsa_entry_table import TSA_OP_HW_FORECAST, TSA_OP_HW_PREDICT, TSA_OP_KPSS_TEST, TSA_OP_SELECT_D
from holtwinters.impl.tsa.holtwinters_params import seasonal_name


def tsa_hw_ops_host_inference(operation: Int, aa: Int, pp: Int, ff: Int, result: Int) raises -> Bool:
    """The forecast, the in-sample prediction and the two stationarity
    operations on the host; False when `operation` is none of them (the
    caller dispatches or refuses it). The spans were checked by the caller."""
    var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
    var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
    var r = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)
    if operation == TSA_OP_HW_FORECAST:
        r[] = holtwinters_forecast_run(f32_ptr(a[0]), f32_ptr(a[1]), p[0], p[1], p[2], p[3], seasonal_name(p[4]))
        return True
    if operation == TSA_OP_HW_PREDICT:
        r[] = holtwinters_predict_run(f32_ptr(a[0]), f32_ptr(a[1]), p[0], p[1], p[2], p[3], p[4], seasonal_name(p[5]))
        return True
    if operation == TSA_OP_KPSS_TEST:
        var f = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        _kpss_flags_host(a[0], p[0], p[1], p[2], p[3], p[4], Float32(f[0]),
                         MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=a[1]), f32_ptr(a[2]))
        r[] = p[0]
        return True
    if operation == TSA_OP_SELECT_D:
        var f = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        if p[4] < 0 or p[4] + p[2] > 2:
            raise Error(
                "select_d: d_max must satisfy 0 <= d_max <= 2 - D (d_max=" + String(p[4])
                + ", D=" + String(p[2]) + "), refused by name"
            )
        kpss_select_d_host(a[0], p[0], p[1], p[2], p[3], p[4], Float32(f[0]), i32_ptr(a[1]))
        r[] = p[0]
        return True
    return False
