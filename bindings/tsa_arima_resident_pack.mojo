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
# The PACK half of the former `bindings/tsa_arima_resident.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/tsa_arima_resident_unpack.mojo`.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.native_loaded_module_door import NativeModules
from bindings.native_prepared_owner_unpack import prepared_put
from bindings.p3nc_fitted_freeze_unpack import fitted_is_freezable_any, fitted_freeze_any
from bindings.py_balanced import py_get
from bindings.tsa_arima_route_unpack import arima_resident_new
from bindings.tsa_hw_route_unpack import TsaHwCall
from bindings.tsa_keys_unpack import tsa_keys
from core.arima_run import arima_resident_prepare_run, arima_resident_release_run, arima_route_resident

# The resident fitted state of a fitted model (lane ne1-a-decomp-tsa, plan
# section A, rows N028/N029): one prepared-owner record per model, outside
# its __dict__ and dropped with it, `(y_, params_, exog_ or None, the
# lifetime owner of the handle, the eleven plan words)`. The lifetime owner
# returns the device copy through the entry that minted it when the record is
# replaced (a refit stores new arrays: the next query misses and mints again)
# or dropped.
comptime ARIMA_RESIDENT_SLOT = "tsa_arima_resident_v1"


def arima_resident_mint(model: PythonObject, fields: PythonObject, call: TsaHwCall, words: List[Int],
                        y_address: Int, ex_address: Int, params_address: Int) raises -> PythonObject:
    """Upload the admitted fitted state once and record it; returns the
    record (its item 3 is the lifetime owner of the handle; the caller keeps
    the record alive through its query, since a model that cannot be weakly
    referenced keeps no record and the owner would release the copy at
    once). None (the per-call path) on a route without the resident
    operations (a CPU binding) or fitted arrays that cannot be frozen (a
    caller-assigned ndarray: its words could change under a device copy,
    plan section A's read-only rule)."""
    var nk = tsa_keys()
    var route = call.route
    if not arima_route_resident(route):
        return PythonObject(None)
    var y_obj = py_get(fields, nk[].k__y)
    var params_obj = py_get(fields, nk[].k_params_)
    var ex_obj = py_get(fields, nk[].k__exog, PythonObject(None))
    if not fitted_is_freezable_any(y_obj) or not fitted_is_freezable_any(params_obj):
        return PythonObject(None)
    if words[10] > 0 and not fitted_is_freezable_any(ex_obj):
        return PythonObject(None)
    _ = fitted_freeze_any(y_obj)
    _ = fitted_freeze_any(params_obj)
    if words[10] > 0:
        _ = fitted_freeze_any(ex_obj)
    var handle = 0
    var py = Python()
    with GILReleased(py):
        handle = arima_resident_prepare_run(route, y_address, ex_address, params_address, words)
    # From here the lifetime owner owns the device copy.
    var resident = PythonObject(None)
    try:
        resident = arima_resident_new(route, handle, call.owner)
    except error:
        arima_resident_release_run(route, handle)
        raise error
    var plan = Python.tuple(words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7],
                            words[8], words[9], words[10])
    var record = Python.tuple(y_obj, params_obj, ex_obj, resident, plan)
    var mods = NativeModules()
    prepared_put(mods, ARIMA_RESIDENT_SLOT, model, record)
    return record
