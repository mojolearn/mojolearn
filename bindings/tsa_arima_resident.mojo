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
from bindings.native_loaded_module import native_loaded_module, NativeModules
from bindings.native_prepared_owner import prepared_get, prepared_put
from bindings.native_resident_capsule import resident_capsule
from bindings.p3nc_fitted_freeze import fitted_is_freezable_any, fitted_freeze_any
from bindings.py_balanced import py_call, py_get, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.classical_arrays import classical_address

# The resident fitted state of a fitted model (lane ne1-a-decomp-tsa, plan
# section A, rows N028/N029): one prepared-owner record per model, outside
# its __dict__ and dropped with it, `(y_, params_, exog_ or None, prepare
# entry, the eleven plan words, handle, capsule)`. The capsule returns the
# device copy to the minting binding when the record is replaced (a refit
# stores new arrays: the next query misses and mints again) or dropped.
comptime ARIMA_RESIDENT_SLOT = "tsa_arima_resident_v1"


def arima_resident_words(batch: Int, n_obs: Int, order: List[Int], k: Int, n_exog: Int) -> List[Int]:
    """`arima_resident_prepare`'s eleven plan words, in its order."""
    return [batch, n_obs, order[0], order[1], order[2], order[3], order[4], order[5], order[6], k, n_exog]


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


def arima_resident_mint(model: PythonObject, fields: PythonObject, entries: PythonObject, words: List[Int],
                        y_address: Int, ex_address: Int, params_address: Int) raises -> PythonObject:
    """Upload the admitted fitted state once and record it; returns the
    record (its word 5 is the handle; the caller keeps the record alive
    through its query, since a model that cannot be weakly referenced keeps
    no record and the capsule would release the copy at once). None (the
    per-call path) on a binding without the resident entries (the host column) or
    fitted arrays that cannot be frozen (a caller-assigned ndarray: its
    words could change under a device copy, plan section A's read-only
    rule)."""
    var prepare = py_get(entries, "arima_resident_prepare")
    var release = py_get(entries, "arima_resident_release")
    if prepare is None or release is None or not py_contains(entries, "arima_predict_resident"):
        return PythonObject(None)
    var y_obj = py_get(fields, "_y")
    var params_obj = py_get(fields, "params_")
    var ex_obj = py_get(fields, "_exog", PythonObject(None))
    if not fitted_is_freezable_any(y_obj) or not fitted_is_freezable_any(params_obj):
        return PythonObject(None)
    if words[10] > 0 and not fitted_is_freezable_any(ex_obj):
        return PythonObject(None)
    _ = fitted_freeze_any(y_obj)
    _ = fitted_freeze_any(params_obj)
    if words[10] > 0:
        _ = fitted_freeze_any(ex_obj)
    var plan = Python.list()
    for i in range(len(words)):  # driver-loop: eleven fixed plan words
        _ = py_call(plan.append, words[i])
    var handle = Int(py=py_call_typed(prepare, y_address, ex_address, params_address, plan))
    # From here the capsule owns the device copy.
    var capsule = resident_capsule(release, handle)
    var record = Python.tuple(y_obj, params_obj, ex_obj, prepare,
                              py_call(native_loaded_module("builtins").tuple, plan), handle, capsule)
    var mods = NativeModules()
    prepared_put(mods, ARIMA_RESIDENT_SLOT, model, record)
    return record


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
