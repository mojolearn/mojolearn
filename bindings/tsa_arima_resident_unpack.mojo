# SPDX-License-Identifier: Apache-2.0
"""The prepared-owner record of a fitted ARIMA's resident device state (lane
ne1-a-decomp-tsa, plan NATIVE_ENTRY_POINTS_AFTER_0866 section A, N028/N029),
shared by `ARIMA.predict` / `forecast` (bindings/tsa_arima_native.mojo) and
AutoARIMA's per-group forecasts (providers/python/autoarima_lifecycle_pyapi.mojo).
The device copy itself is arima/resident_state.mojo's, reached through the
model's binding entries `arima_resident_prepare`, `arima_resident_release`,
`arima_predict_resident` and `arima_forecast_resident`. Nothing here computes
on data: the record names the fitted objects by identity, never by content.
"""
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from bindings.native_prepared_owner_unpack import prepared_get, prepared_put
from bindings.native_resident_capsule_pack import resident_capsule
from bindings.p3nc_fitted_freeze_unpack import fitted_is_freezable_any, fitted_freeze_any
from bindings.py_balanced import py_call, py_get, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.classical_arrays_door import classical_address

def arima_resident_hit(record: PythonObject, fields: PythonObject, entries: PythonObject, words: List[Int]) raises -> Int:
    """The handle of this model's resident fitted state when its record was
    minted from these very `_y`, `params_` and `_exog` objects (frozen
    read-only when minted, held by the record so their addresses cannot be
    recycled) by this binding with these plan words; -1 on any miss."""
    if record[0] is not py_get(fields, "_y") or record[1] is not py_get(fields, "params_"):
        return -1
    if record[2] is not py_get(fields, "_exog", PythonObject(None)):
        return -1
    if record[3] is not py_get(entries, "arima_resident_prepare"):
        return -1
    var held = record[4]
    if len(held) != len(words):
        return -1
    for i in range(len(words)):  # driver-loop: eleven fixed plan words
        if Int(py=held[i]) != words[i]:
            return -1
    return Int(py=record[5])


def arima_resident_call(entries: PythonObject, handle: Int, future_addr: Int, result: PythonObject, start: Int, end: Int,
                        width: Int, forecast: Bool) raises:
    """One resident predict (`[start, end]`) or forecast (`[width, 0]`)
    writing `result`'s float32 words."""
    if forecast:
        _ = py_call_typed(entries["arima_forecast_resident"], handle, future_addr, classical_address(result),
                          Python.list(width, 0))
    else:
        _ = py_call_typed(entries["arima_predict_resident"], handle, future_addr, classical_address(result),
                          Python.list(start, end))
