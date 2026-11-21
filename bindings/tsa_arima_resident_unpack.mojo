# SPDX-License-Identifier: Apache-2.0
"""The prepared-owner record of a fitted ARIMA's resident device state (lane
ne1-a-decomp-tsa, plan NATIVE_ENTRY_POINTS_AFTER_0866 section A, N028/N029),
shared by `ARIMA.predict` / `forecast` (bindings/tsa_arima_native_door.mojo) and
AutoARIMA's per-group forecasts (providers/python/autoarima_lifecycle_pyapi_door.mojo).
The device copy itself is arima/resident_state.mojo's, reached through the
route's operations entry (core/arima_entry_table.mojo: RESIDENT_PREPARE,
PREDICT_RESIDENT, FORECAST_RESIDENT; RESIDENT_RELEASE by the record's
lifetime owner). Nothing here computes on data: the record names the fitted
objects by identity, never by content.
"""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.py_balanced import py_get
from bindings.tsa_hw_route_unpack import TsaHwCall, tsa_hw_resident_handle, tsa_resident_same_entry
from bindings.tsa_keys_unpack import tsa_keys
from core.arima_run import arima_resident_query_run, arima_route_resident


def arima_resident_hit(record: PythonObject, fields: PythonObject, call: TsaHwCall, words: List[Int]) raises -> Int:
    """The handle of this model's resident fitted state when its record was
    minted from these very `_y`, `params_` and `_exog` objects (frozen
    read-only when minted, held by the record so their addresses cannot be
    recycled) through this route's entry with these plan words; -1 on any
    miss."""
    var nk = tsa_keys()
    if not arima_route_resident(call.route):
        return -1
    if record[0] is not py_get(fields, nk[].k__y) or record[1] is not py_get(fields, nk[].k_params_):
        return -1
    if record[2] is not py_get(fields, nk[].k__exog, PythonObject(None)):
        return -1
    if not tsa_resident_same_entry(record[3], call.route.entry):
        return -1
    var held = record[4]
    if len(held) != len(words):
        return -1
    for i in range(len(words)):  # driver-loop: eleven fixed plan words
        if Int(py=held[i]) != words[i]:
            return -1
    return tsa_hw_resident_handle(record[3])


def arima_resident_call(call: TsaHwCall, handle: Int, future_addr: Int, result_address: Int, start: Int, end: Int,
                        width: Int, forecast: Bool) raises:
    """One resident predict (`[start, end)`) or forecast (`width` steps)
    writing float32 words at `result_address`. The caller keeps the result's
    owner, the future regressors' owner and the record alive through it; the
    route owner (the image pin) is named here."""
    var route = call.route
    var py = Python()
    try:
        with GILReleased(py):
            _ = arima_resident_query_run(route, handle, future_addr, result_address, start, end, width, forecast)
    finally:
        _ = call.owner  # the route owner pins the image through the call
