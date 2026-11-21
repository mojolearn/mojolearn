# SPDX-License-Identifier: Apache-2.0
"""`ExponentialSmoothing.fit`, `predict` and `score`
(python/mojolearn/_tsa_impl.py) as one native call each, through
`tsa_native_call` (bindings/tsa_public_native.mojo):

    fit      `_check_dims` (the endog admitted 1-D or 2-D and cast to
             C-ordered float32 by the shared element cast), one
             `holtwinters_fit` on the model's tier, then `_set_fitted`'s
             state (the packed components, their time-major blocks and the
             per-series statistics) written natively
    predict  the argument refusals, `holtwinters_predict` and / or
             `holtwinters_forecast` into one time-major buffer, and
             `_shaped`'s three return shapes
    score    the fitted SSE, or one series' SSE

The same compiled kernels with the same argument words, so no bit moves.

UNPACK / RUN / PACK. This file is the boundary: it admits the model's fields
and buffers (UNPACK) and writes the fitted state and the returned arrays
(PACK). RUN is core/tsa_hw_run.mojo: one call through the estimator's
resolved operations entry (core/tsa_entry_table.mojo) over typed addresses
and words, with the GIL released. The entry is the estimator's device route
(`device=None | "gpu" | "cpu"`, providers/python/tsa_route_boundary_door.mojo),
resolved once per estimator; no operation here tests the device, looks an
entry up by name or builds a Python argument list. A host inference class
(`HostExponentialSmoothing`) resolves the forecast inference image's table.
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.native_loaded_module_door import native_loaded_module
from bindings.native_array_storage_door import native_array_zeros, native_array_unfilled
from bindings.arena_materialize_native_door import _owned
from bindings.classical_archive_door import classical_archive_write
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains
from bindings.classical_arrays_door import classical_input, classical_address, classical_cast_kind, classical_wrap_array
from bindings.classical_pandas_door import classical_admit_value
from providers.python.array_metadata_pyapi_door import ArrayMetadata
from bindings.tsa_admission_door import tsa_hw_fields, tsa_hw_class, tsa_scalar, tsa_component_index
from bindings.admitted_input_unpack import admit_view, AdmittedTypedPin
from bindings.admitted_input_run import admitted_view_kind
from bindings.tensor_view import TensorView
from core.cast_in import CAST_KIND_FLOAT, CAST_KIND_SIGNED
from bindings.native_owner_lifetime_door import native_release_admit
from bindings.classical_attributes_door import classical_connect_numpy
from providers.python.forecast_owner_pyapi_unpack import forecast_commit
from core.xseq_marshal import xseq_column_copy_f32
from tsa.tsa_hw_run import hw_component_gather
from core.admitted_input_nd import admitted_cast_nd_f32
from bindings.admitted_sequence_door import admitted_array_f32
from bindings.native_loaded_module_door import NativeModules
from bindings.native_prepared_owner_unpack import prepared_get, prepared_put
from core.tsa_hw_run import (
    tsa_hw_fit_run, tsa_hw_forecast_run, tsa_hw_predict_run, tsa_hw_resident_prepare_run,
    tsa_hw_resident_release_run, tsa_hw_query_resident_run,
)
from core.tsa_route import TsaHwRoute, tsa_hw_route_fits, tsa_hw_route_resident, tsa_vendor_word
from holtwinters.impl.tsa.holtwinters_params import seasonal_from_name
from providers.python.tsa_route_boundary_door import (
    TsaHwRouteCall, tsa_hw_route_call, tsa_hw_route_clear, tsa_hw_device_admit,
    tsa_hw_resident_new, tsa_hw_resident_matches, tsa_hw_resident_handle,
)
from bindings.p3nc_fitted_freeze_unpack import fitted_is_freezable_any, fitted_freeze_any
from bindings.tsa_keys_unpack import tsa_keys
from bindings.py_balanced import py_item

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _HW = "mojolearn ExponentialSmoothing: "
# The resident fitted components of a fitted model (lane ne1-a-decomp-tsa,
# plan section A, row N030): one prepared-owner record per model, outside
# its __dict__ and dropped with it, `(_comps, lifetime owner, n, ts_num,
# seasonal_periods)`. The lifetime owner (providers/python/tsa_route_boundary_door.mojo
# `TsaHwResidentOwner`) holds the handle and the entry that minted it. A
# refit stores a new `_comps`, so the next query misses, mints again and the
# replaced record's owner returns the old device copy through that entry.
comptime _HW_RESIDENT_SLOT = "tsa_hw_resident_v1"


def _b() raises -> PythonObject:
    # The import-time builtins connection: no __import__ hook can run.
    return native_loaded_module("builtins")


def _repr(value: PythonObject) raises -> String:
    return String(py=py_call(_b().repr, value))


def _type_text(value: PythonObject) raises -> String:
    """`str(type(value))`: "<class 'float'>"."""
    return String("unsupported native type")


def tsa_array(code: String, typestr: String, shape: PythonObject, count: Int, width: Int,
           context: PythonObject) raises -> PythonObject:
    """A zero-filled C-ordered mojolearn Array of `typestr` and `shape`."""
    if count < 0 or count > 9223372036854775807 // width:
        raise Error("ValueError: TSA output exceeds native indexing bounds")
    var store = native_array_zeros(code, count)
    return _owned(store, shape, typestr, context)


def tsa_array_unfilled(code: String, typestr: String, shape: PythonObject, count: Int, width: Int,
                       context: PythonObject) raises -> PythonObject:
    """`tsa_array` with NO fill pass (lane fix-output-zero-fill, 2026-11-14):
    the store is `native_array_unfilled`. Only for an output one native
    writer fills in full before anything reads it or it is returned."""
    if count < 0 or count > 9223372036854775807 // width:
        raise Error("ValueError: TSA output exceeds native indexing bounds")
    var store = native_array_unfilled(code, count)
    return _owned(store, shape, typestr, context)


def _is_int(value: PythonObject) raises -> Bool:
    """`isinstance(value, int) and not isinstance(value, bool)`."""
    var b = _b()
    return py_type(value) is b.int


def _hw_route(model: PythonObject, fields: PythonObject, context: PythonObject) raises -> TsaHwRouteCall:
    """The estimator's resolved device route (one record read and three
    pointer compares once resolved); the returned owner pins the image and
    is kept alive by the caller through its entry calls."""
    return tsa_hw_route_call(model, fields, tsa_hw_class(model, context), context, _b())


def _seasonal(fields: PythonObject) raises -> Int:
    """The SEASONAL_* word of the admitted `seasonal` name (`_settings`
    checked its type and spelling; an unknown name is refused here in the
    kernel's words)."""
    var nk = tsa_keys()
    return seasonal_from_name(String(py_item(fields, nk[].k_seasonal)))


# ---------------------------------------------------------------- fit

def _check_dims(fields: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_check_dims(ts_input)`: (C-ordered float32 (ts_num, n) data, n)."""
    var nk = tsa_keys()
    var value_pin = AdmittedTypedPin()
    var value = admit_view(source, "endog", context, value_pin)
    var kind = admitted_view_kind(value, "endog")
    var rank = len(value.shape)
    var ts_num = Int(py=py_item(fields, nk[].k_ts_num))
    if rank != 1 and rank != 2:
        raise Error("ValueError: " + _HW + "data input must have 1 or 2 dimensions")
    var rows = 1 if rank == 1 else value.shape[0]
    var n = value.shape[0] if rank == 1 else value.shape[1]
    if ts_num != rows:
        raise Error("ValueError: " + _HW + "initialized series count disagrees with data")
    if rows * n == 0:
        raise Error("ValueError: " + _HW + "endog is empty")
    var out = admitted_array_f32(Python.tuple(rows, n), rows * n, context)
    _ = admitted_cast_nd_f32(value.addr, value.shape, value.strides, kind[1],
                 kind[0], _F32P(unsafe_from_address=Int(py=classical_address(out))))
    _ = value_pin^
    _ = out
    return Python.tuple(out, n)


def _block(source: PythonObject, offset: Int, count: Int, code: String, typestr: String,
           shape: PythonObject, context: PythonObject) raises -> PythonObject:
    """A copy of `count` 4-byte words of `source` from word `offset`, as an
    Array of `shape` (the Python body's slice, and its reshape)."""
    var out = tsa_array(code, typestr, shape, count, 4, context)
    if count > 0:
        unsafe_memcpy(dest=_F32P(unsafe_from_address=Int(py=classical_address(out))),
               src=_F32P(unsafe_from_address=Int(py=classical_address(source)) + 4 * offset), count=count)
        _ = out  # keep the Python object alive through the native call above
    return out


def _set_fitted(fields: PythonObject, n: Int, comps: PythonObject, stats: PythonObject, flags: PythonObject,
                context: PythonObject) raises -> PythonObject:
    """`_set_fitted`: the packed components, the three time-major blocks and
    the per-series statistics; the lazily built series-major components are
    dropped."""
    var nk = tsa_keys()
    var period = Int(py=py_item(fields, nk[].k_seasonal_periods))
    var b = Int(py=py_item(fields, nk[].k_ts_num))
    var cl = (n - period) * b
    var rows = cl // b
    var updates = Python.dict()
    py_setitem(updates, nk[].k_n, n)
    py_setitem(updates, nk[].k__components_len, cl)
    py_setitem(updates, nk[].k__comps, comps)
    py_setitem(updates, nk[].k__comp_layout, Python.tuple(cl, b, rows))
    var time_major = _b().dict()
    var names = [String("level"), String("trend"), String("season")]
    for i in range(3):  # small-loop(the three packed components)
        py_setitem(time_major, names[i], _block(comps, i * cl, cl, "f", "<f4", Python.tuple(rows, b), context))
    py_setitem(updates, nk[].k__time_major, time_major)
    var stat_names = [String("sse_"), String("alpha_"), String("beta_"), String("gamma_")]
    for i in range(4):  # small-loop(the four per-series statistics)
        py_setitem(updates, stat_names[i], _block(stats, i * b, b, "f", "<f4", Python.tuple(b), context))
    py_setitem(updates, nk[].k_n_iter_, _block(flags, 0, b, "i", "<i4", Python.tuple(b), context))
    py_setitem(updates, nk[].k_criterion_, _block(flags, b, b, "i", "<i4", Python.tuple(b), context))
    py_setitem(updates, nk[].k_fit_executed_flag, True)

    return updates


def hw_fit(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = tsa_keys()
    # One admission of the class and instance keys serves the snapshot and
    # the commit below (nothing between them touches the model's storage).
    var live = tsa_hw_fields(model, context)
    var fields = _snapshot_of(live, context)
    var checked = _check_dims(fields, py_item(fields, nk[].k_endog), context)
    var data = checked[0]
    var n = Int(py=checked[1])
    var ts_num = Int(py=py_item(fields, nk[].k_ts_num))
    var period = Int(py=py_item(fields, nk[].k_seasonal_periods))
    if ts_num < 1 or n <= period or n - period > 9223372036854775807 // 12 // ts_num:
        raise Error("ValueError: Holt-Winters fitted extent is invalid")
    var cl = (n - period) * ts_num
    if cl <= 0:
        raise Error("ValueError: " + _HW + "n (" + String(n) + ") must exceed seasonal_periods (" + String(period) + ")")
    var comps = tsa_array("f", "<f4", Python.tuple(3 * cl), 3 * cl, 4, context)
    var stats = tsa_array("f", "<f4", Python.tuple(4 * ts_num), 4 * ts_num, 4, context)
    var flags = tsa_array("i", "<i4", Python.tuple(2 * ts_num), 2 * ts_num, 4, context)
    var call = _hw_route(model, fields, context)
    var route = call.route
    if not tsa_hw_route_fits(route):
        raise Error("AttributeError: the forecast binding has no holtwinters_fit")
    var init = Int(py=py_item(context, nk[].k_hw_init_codes)[py_item(fields, nk[].k_initialization_method)])
    var start_periods = Int(py=py_item(fields, nk[].k_start_periods))
    var eps = Float64(py=py_item(fields, nk[].k_eps))
    var seasonal = _seasonal(fields)
    var data_address = Int(py=classical_address(data))
    var comps_address = Int(py=classical_address(comps))
    var stats_address = Int(py=classical_address(stats))
    var flags_address = Int(py=classical_address(flags))
    var py = Python()
    try:
        with GILReleased(py):
            _ = tsa_hw_fit_run(route, data_address, comps_address, stats_address, flags_address,
                               n, ts_num, period, start_periods, init, seasonal, eps)
    finally:
        _ = comps  # keep the Python object alive through the native call above
        _ = data  # keep the Python object alive through the native call above
        _ = flags  # keep the Python object alive through the native call above
        _ = stats  # keep the Python object alive through the native call above
        _ = call.owner  # the route owner pins the image through the call
    var updates = _set_fitted(fields, n, comps, stats, flags, context)
    py_setitem(updates, nk[].k__data, data)
    _commit(live, updates, context, True)
    return model


# ---------------------------------------------------------------- predict

def _shaped(fields: PythonObject, out_buf: PythonObject, offset: Int, steps: Int, index: PythonObject,
            context: PythonObject) raises -> PythonObject:
    """`_shaped(out[offset:], steps, index)` over a time-major buffer."""
    var nk = tsa_keys()
    var b = Int(py=py_item(fields, nk[].k_ts_num))
    var base = Int(py=classical_address(out_buf)) + 4 * offset
    if index is not None:
        var s = Int(py=index)
        var one = tsa_array("f", "<f4", Python.tuple(steps), steps, 4, context)
        # one series' strided words (a copy, no arithmetic)
        xseq_column_copy_f32(_F32P(unsafe_from_address=base), _F32P(unsafe_from_address=Int(py=classical_address(one))),
                             steps, b, s)
        _ = one  # keep the Python object alive through the native call above
        return one
    if b == 1:
        return _block(out_buf, offset, steps, "f", "<f4", Python.tuple(steps), context)
    return _block(out_buf, offset, steps * b, "f", "<f4", Python.tuple(steps, b), context)


def _check_index(index: PythonObject, b: Int) raises:
    if index is not None and not _is_int(index):
        raise Error("TypeError: " + _HW + "index must be int or None, got " + _type_text(index))


def _hw_held(model: PythonObject, fields: PythonObject) raises -> PythonObject:
    """The held resident record when it was minted from this very `_comps`
    object (frozen read-only at the mint and held by the record, so its
    words, shape and address cannot have changed: the `_packed` admission it
    passed then passes now), else None. The caller still checks the binding
    (`_hw_matches`)."""
    var nk = tsa_keys()
    var mods = NativeModules()
    var held = prepared_get(mods, _HW_RESIDENT_SLOT, model)
    if held is None or held[0] is not py_get(fields, nk[].k__comps):
        return PythonObject(None)
    return held


def _hw_matches(held: PythonObject, route: TsaHwRoute, n: Int, b: Int, period: Int) raises -> Bool:
    """The held record was minted through this route's entry with these
    fitted words."""
    if held is None or not tsa_hw_route_resident(route):
        return False
    return (tsa_hw_resident_matches(held[1], route) and Int(py=held[2]) == n
            and Int(py=held[3]) == b and Int(py=held[4]) == period)


def _hw_mint(model: PythonObject, fields: PythonObject, call: TsaHwRouteCall, n: Int, b: Int, period: Int,
             comps_address: Int) raises -> PythonObject:
    """Upload the admitted components (`comps_address`, the `_packed`
    admission of `_comps`) once and record them; returns the record (word 1
    the lifetime owner of the handle; the caller keeps the record alive
    through its query, since a model that cannot be weakly referenced keeps
    no record and the owner would release the copy at once). None (the
    per-call path) on a route without the resident operations (a CPU
    binding) or components that cannot be frozen (a caller-assigned ndarray:
    plan section A's read-only rule)."""
    var nk = tsa_keys()
    var route = call.route
    if not tsa_hw_route_resident(route):
        return PythonObject(None)
    var comps_obj = py_get(fields, nk[].k__comps)
    if not fitted_is_freezable_any(comps_obj):
        return PythonObject(None)
    _ = fitted_freeze_any(comps_obj)
    var handle = 0
    var py = Python()
    with GILReleased(py):
        handle = tsa_hw_resident_prepare_run(route, comps_address, n, b, period)
    # From here the lifetime owner owns the device copy.
    var resident = PythonObject(None)
    try:
        resident = tsa_hw_resident_new(route, handle, call.owner)
    except error:
        tsa_hw_resident_release_run(route, handle)
        raise error
    var record = Python.tuple(comps_obj, resident, n, b, period)
    var mods = NativeModules()
    prepared_put(mods, _HW_RESIDENT_SLOT, model, record)
    return record


def _hw_query(call: TsaHwRouteCall, record: PythonObject, fields: PythonObject, start: Int, end: Int, b: Int,
              context: PythonObject) raises -> PythonObject:
    """One resident query: `(end - start) * b` float32, time-major, the
    in-sample steps then the forecast steps."""
    var steps = end - start
    var whole = tsa_array("f", "<f4", Python.tuple(steps * b), steps * b, 4, context)
    var route = call.route
    var handle = tsa_hw_resident_handle(record[1])
    var seasonal = _seasonal(fields)
    var whole_address = Int(py=classical_address(whole))
    var py = Python()
    try:
        with GILReleased(py):
            _ = tsa_hw_query_resident_run(route, handle, whole_address, start, end, seasonal)
    finally:
        _ = record  # the record (and its lifetime owner) outlive the query
        _ = whole  # keep the Python object alive through the native call above
        _ = call.owner  # the route owner pins the image through the call
    return whole


def _query_fields(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_snapshot`'s checks over the admitted instance storage itself, with
    no copy of the instance dict and no rewrite of `eps` (a query reads
    neither the copy nor `eps`; lane ne1-a-decomp-tsa, N030): the same
    refusals in the same order."""
    var fields = tsa_hw_fields(model, context)
    _settings(fields, context, False)
    _fitted_checks(fields)
    return fields


def hw_predict(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = tsa_keys()
    var fields = _query_fields(model, context)
    if not Bool(py=py_item(fields, nk[].k_fit_executed_flag)):
        raise Error("ValueError: " + _HW + "fit() the model before predict()")
    var n = Int(py=py_item(fields, nk[].k_n))
    var b = Int(py=py_item(fields, nk[].k_ts_num))
    var start_obj = args[0]
    var end_obj = PythonObject(n) if args[1] is None else args[1]
    var index = args[2]
    if not _is_int(start_obj):
        raise Error("TypeError: " + _HW + "start must be int, got " + _type_text(start_obj))
    if not _is_int(end_obj):
        raise Error("TypeError: " + _HW + "end must be int, got " + _type_text(end_obj))
    _check_index(index, b)
    var start = Int(py=start_obj)
    var end = Int(py=end_obj)
    if start < 0 or end <= start:
        raise Error("ValueError: " + _HW + "need 0 <= start < end (start=" + String(start) + ", end=" + String(end) + ")")
    if index is not None and (Int(py=index) < 0 or Int(py=index) >= b):
        raise Error("IndexError: " + _HW + "index input: " + String(Int(py=index)) + " outside of range [0, " + String(b) + ")")
    var call = _hw_route(model, fields, context)
    var route = call.route
    var period = Int(py=py_item(fields, nk[].k_seasonal_periods))
    var steps = end - start
    if steps > 9223372036854775807 // 4 // b or (end > n and end - n > 9223372036854775807 // 4 // b):
        raise Error("ValueError: Holt-Winters prediction extent exceeds native indexing bounds")
    # One call over the device copy (the in-sample steps, then the forecast
    # steps, time-major: the words of the two per-call operations) when the
    # components are resident or can be made so.
    var held = _hw_held(model, fields)
    if _hw_matches(held, route, n, b, period):
        return _shaped(fields, _hw_query(call, held, fields, start, end, b, context), 0, steps, index, context)
    var packed_pin = AdmittedTypedPin()
    var packed = _packed(py_item(fields, nk[].k__comps), 3 * (n - period) * b, context, packed_pin)
    var comps = packed_pin.keep
    var minted = _hw_mint(model, fields, call, n, b, period, packed.addr)
    if minted is not None:
        var whole = _hw_query(call, minted, fields, start, end, b, context)
        _ = comps
        _ = packed_pin^
        return _shaped(fields, whole, 0, steps, index, context)
    var seasonal = _seasonal(fields)
    var comps_address = Int(py=classical_address(comps))
    var py = Python()
    if start >= n:
        var fc = tsa_array("f", "<f4", Python.tuple((end - n) * b), (end - n) * b, 4, context)
        var fc_address = Int(py=classical_address(fc))
        try:
            with GILReleased(py):
                _ = tsa_hw_forecast_run(route, comps_address, fc_address, n, b, period, end - n, seasonal)
        finally:
            _ = fc  # keep the Python object alive through the native call above
            _ = comps  # keep the Python object alive through the native call above
            _ = call.owner  # the route owner pins the image through the call
        _ = packed_pin^
        return _shaped(fields, fc, (start - n) * b, steps, index, context)
    var out = tsa_array("f", "<f4", Python.tuple(steps * b), steps * b, 4, context)
    var k_in = min(end, n) - start
    var base = Int(py=classical_address(out))
    try:
        with GILReleased(py):
            _ = tsa_hw_predict_run(route, comps_address, base, n, b, period, start, start + k_in, seasonal)
            if end > n:
                _ = tsa_hw_forecast_run(route, comps_address, base + 4 * k_in * b, n, b, period, end - n, seasonal)
    finally:
        _ = out  # keep the Python object alive through the native calls above
        _ = comps  # keep the Python object alive through the native calls above
        _ = call.owner  # the route owner pins the image through the calls
    _ = packed_pin^
    return _shaped(fields, out, 0, steps, index, context)


def hw_score(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = tsa_keys()
    var fields = _query_fields(model, context)
    if not Bool(py=py_item(fields, nk[].k_fit_executed_flag)):
        raise Error("ValueError: " + _HW + "fit() the model before score()")
    var index = tsa_component_index(args[0], context)
    if index is None:
        return py_item(fields, nk[].k_sse_)
    var b = Int(py=py_item(fields, nk[].k_ts_num))
    _check_index(index, b)
    if Int(py=index) < 0 or Int(py=index) >= b:
        raise Error("IndexError: " + _HW + "index input: " + String(Int(py=index))
                    + " outside of range [0, " + String(b) + ")")
    # `self.sse_[index]`: the float32 word, as a Python float
    var packed_pin = AdmittedTypedPin()
    var packed = _packed(py_item(fields, nk[].k_sse_), b, context, packed_pin)
    var value = _F32P(unsafe_from_address=packed.addr)[Int(py=index)]
    _ = packed_pin^  # keep the pinned owners alive through the read above
    return PythonObject(Float64(value))


def _settings(fields: PythonObject, context: PythonObject, write_eps: Bool = True) raises:
    var nk = tsa_keys()
    var b = _b()
    for name in ["ts_num", "seasonal_periods", "start_periods"]:  # small-loop(three integer constructor settings)
        if not _is_int(fields[name]):
            raise Error("TypeError: " + _HW + name + " must be int")
    var seasonal = py_item(fields, nk[].k_seasonal)
    if py_type(seasonal) is not b.str:
        raise Error("TypeError: " + _HW + "seasonal must be a native string")
    if String(seasonal) != "additive" and String(seasonal) != "add" and String(seasonal) != "multiplicative" and String(seasonal) != "mul":
        raise Error("ValueError: " + _HW + "seasonal must be additive/add or multiplicative/mul")
    var method = py_item(fields, nk[].k_initialization_method)
    if py_type(method) is not b.str or (String(method) != "estimated" and String(method) != "heuristic" and String(method) != "cuml"):
        raise Error("ValueError: " + _HW + "initialization_method must be estimated, heuristic or cuml")
    var eps = py_call(b.float, tsa_scalar(py_item(fields, nk[].k_eps), context))
    if write_eps:
        py_setitem(fields, nk[].k_eps, eps)


def _snapshot(model: PythonObject, context: PythonObject) raises -> PythonObject:
    return _snapshot_of(tsa_hw_fields(model, context), context)


def _snapshot_of(live: PythonObject, context: PythonObject) raises -> PythonObject:
    """The checked copy of admitted instance storage `live` (the dict
    `tsa_hw_fields` returned): callers that commit to the model again pass
    the same `live` instead of admitting the class and keys a second time."""
    var fields = py_call(_b().dict, live)
    _settings(fields, context)
    _fitted_checks(fields)
    return fields


def _fitted_checks(fields: PythonObject) raises:
    """The fitted-flag and fitted-extent refusals of `_snapshot_of`."""
    var nk = tsa_keys()
    if py_type(py_item(fields, nk[].k_fit_executed_flag)) is not _b().bool:
        raise Error("TypeError: " + _HW + "fitted flag must be a native boolean")
    if Bool(py=py_item(fields, nk[].k_fit_executed_flag)):
        if not _is_int(py_item(fields, nk[].k_n)) or Int(py=py_item(fields, nk[].k_ts_num)) < 1 or Int(py=py_item(fields, nk[].k_seasonal_periods)) < 2 or Int(py=py_item(fields, nk[].k_n)) <= Int(py=py_item(fields, nk[].k_seasonal_periods)):
            raise Error("ValueError: " + _HW + "invalid fitted dimensions")
        var batch = Int(py=py_item(fields, nk[].k_ts_num))
        if Int(py=py_item(fields, nk[].k_n)) - Int(py=py_item(fields, nk[].k_seasonal_periods)) > 9223372036854775807 // 12 // batch:
            raise Error("ValueError: Holt-Winters fitted extent exceeds native indexing bounds")


def _commit(fields: PythonObject, updates: PythonObject, context: PythonObject, clear_components: Bool = False) raises:
    var nk = tsa_keys()
    var old = Python.list()
    var ownership = py_item(context, nk[].k_tsa_owner_context)
    classical_connect_numpy(ownership)
    if clear_components:
        for name in ["level_", "trend_", "season_"]:  # small-loop(lazy component owners are admitted before any mutation)
            var value = py_get(fields, name)
            _ = py_call(old.append, value)
            native_release_admit(value, ownership)
    forecast_commit(fields, updates, ownership)
    if clear_components:
        for name in ["level_", "trend_", "season_"]:  # small-loop(release only the preadmitted lazy component owners)
            _ = py_call(fields.pop, name, PythonObject(None))
    _ = old


def hw_vendor_used(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """`vendor_used()`: the vendor word the estimator's resolved entry
    itself reports (`cpu`, `metal`, `cuda`, `hip`), resolving the route when
    this is the estimator's first operation."""
    var fields = tsa_hw_fields(model, context)
    var call = _hw_route(model, fields, context)
    return PythonObject(tsa_vendor_word(call.route.vendor))


def hw_initialize(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = tsa_hw_fields(model, context)
    # `device` is checked before any field changes (an unknown value, or an
    # explicit "gpu" on a CPU-only installation, is refused in words) and
    # the route of an estimator constructed again is forgotten; it is read
    # into the route owner at the estimator's first operation.
    tsa_hw_device_admit(args[7], context, _b())
    var updates = Python.dict(endog=args[0], seasonal=args[1], seasonal_periods=args[2],
        start_periods=args[3], ts_num=args[4], eps=args[5], initialization_method=args[6], fit_executed_flag=False,
        device=args[7])
    _settings(updates, context)
    tsa_hw_route_clear(model)
    # Constructor retains endog without numerical preparation, matching its
    # deferred fit contract. Its replacement is admitted before publication.
    _commit(fields, updates, context)
    return PythonObject(None)


def _packed(value: PythonObject, count: Int, context: PythonObject, mut hold: AdmittedTypedPin) raises -> TensorView:
    # The typed view RUN reads; `hold` (the door's) pins the owners.
    var result = admit_view(value, "Holt-Winters fitted storage", context, hold)
    var kind = admitted_view_kind(result, "Holt-Winters fitted storage")
    if kind[0] != CAST_KIND_FLOAT or kind[1] != 4 or not result.c_contig or result.count() != count:
        raise Error("ValueError: " + _HW + "fitted storage must be contiguous float32 with the fitted extent")
    return result^


def hw_forecast(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = tsa_keys()
    var fields = _query_fields(model, context)
    if not Bool(py=py_item(fields, nk[].k_fit_executed_flag)):
        raise Error("ValueError: " + _HW + "fit() the model before forecast()")
    if not _is_int(args[0]):
        raise Error("TypeError: " + _HW + "h must be int")
    var h = Int(py=args[0])
    var b = Int(py=py_item(fields, nk[].k_ts_num))
    var index = args[1]
    _check_index(index, b)
    if h <= 0:
        raise Error("ValueError: " + _HW + "h must be > 0")
    if index is not None and (Int(py=index) < 0 or Int(py=index) >= b):
        raise Error("IndexError: " + _HW + "index outside series range")
    var n = Int(py=py_item(fields, nk[].k_n))
    var period = Int(py=py_item(fields, nk[].k_seasonal_periods))
    var held = _hw_held(model, fields)
    if held is not None:
        # A held record names this very frozen `_comps`, so the `_packed`
        # admission below would pass: resolving the binding first cannot
        # change which refusal a call meets.
        if h > 9223372036854775807 // 4 // b:
            raise Error("ValueError: Holt-Winters forecast extent exceeds native indexing bounds")
        var hit = _hw_route(model, fields, context)
        if _hw_matches(held, hit.route, n, b, period):
            return _shaped(fields, _hw_query(hit, held, fields, n, n + h, b, context), 0, h, index, context)
    var comps_pin = AdmittedTypedPin()
    var comps = _packed(py_item(fields, nk[].k__comps), 3 * (n - period) * b, context, comps_pin)
    if h > 9223372036854775807 // 4 // b:
        raise Error("ValueError: Holt-Winters forecast extent exceeds native indexing bounds")
    var out = tsa_array("f", "<f4", Python.tuple(h * b), h * b, 4, context)
    var call = _hw_route(model, fields, context)
    var minted = _hw_mint(model, fields, call, n, b, period, comps.addr)
    if minted is not None:
        var whole = _hw_query(call, minted, fields, n, n + h, b, context)
        _ = comps_pin^
        return _shaped(fields, whole, 0, h, index, context)
    var route = call.route
    var seasonal = _seasonal(fields)
    var comps_address = comps.addr
    var out_address = Int(py=classical_address(out))
    var py = Python()
    try:
        with GILReleased(py):
            _ = tsa_hw_forecast_run(route, comps_address, out_address, n, b, period, h, seasonal)
    finally:
        _ = out  # keep the Python object alive through the native call above
        _ = call.owner  # the route owner pins the image through the call
    _ = comps_pin^
    return _shaped(fields, out, 0, h, index, context)


def hw_component(model: PythonObject, args: PythonObject, context: PythonObject, attribute: Bool = False) raises -> PythonObject:
    var nk = tsa_keys()
    var name = args[0]
    if py_type(name) is not _b().str:
        raise Error("TypeError: " + _HW + "component name must be a native string")
    var which = -1
    if String(name) == "level_":
        which = 0
    elif String(name) == "trend_":
        which = 1
    elif String(name) == "season_":
        which = 2
    if which < 0:
        raise Error("AttributeError: ExponentialSmoothing has no such attribute")
    var live = tsa_hw_fields(model, context)
    var fields = _snapshot_of(live, context)
    if not Bool(py=py_item(fields, nk[].k_fit_executed_flag)):
        if attribute:
            raise Error("AttributeError: ExponentialSmoothing has no fitted component")
        raise Error("ValueError: " + _HW + "fit() the model to get components")
    var b = Int(py=py_item(fields, nk[].k_ts_num))
    var rows = Int(py=py_item(fields, nk[].k_n)) - Int(py=py_item(fields, nk[].k_seasonal_periods))
    var index = PythonObject(None) if attribute else tsa_component_index(args[1], context)
    _check_index(index, b)
    if index is not None and (Int(py=index) < 0 or Int(py=index) >= b):
        raise Error("IndexError: " + _HW + "index outside series range")
    var cached = py_get(fields, name)
    if not attribute and cached is not None and (index is not None or b == 1):
        var cached_pin = AdmittedTypedPin()
        _ = _packed(cached, b * rows, context, cached_pin)
        return _block(cached_pin.keep, 0 if index is None else Int(py=index) * rows,
                      rows, "f", "<f4", Python.tuple(rows), context)
    var comps_pin = AdmittedTypedPin()
    var comps = _packed(py_item(fields, nk[].k__comps), 3 * rows * b, context, comps_pin)
    if not attribute and index is not None:
        return _shaped(fields, comps_pin.keep, which * rows * b, rows, index, context)
    var out = tsa_array("f", "<f4", Python.tuple(b, rows), b * rows, 4, context)
    var base = Int(py=classical_address(out))
    hw_component_gather(comps.addr, base, which, rows, b)
    if cached is None:
        var updates = Python.dict()
        py_setitem(updates, name, out)
        _commit(live, updates, context)
    _ = comps_pin^
    if attribute:
        return out
    if b == 1:
        return _block(out, 0, rows, "f", "<f4", Python.tuple(rows), context)
    var time_major = py_item(fields, nk[].k__time_major)
    if py_type(time_major) is not _b().dict:
        raise Error("TypeError: Holt-Winters components require native dictionary storage")
    var keys = py_call(_b().list, time_major)
    for i in range(len(keys)):  # driver-loop: admit keys before retrieving the named native component
        if py_type(keys[i]) is not _b().str:
            raise Error("TypeError: Holt-Winters component keys must be native strings")
    var names = Python.tuple("level", "trend", "season")
    var stored_pin = AdmittedTypedPin()
    _ = _packed(time_major[names[which]], rows * b, context, stored_pin)
    return stored_pin.keep


def hw_save(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = tsa_keys()
    var fields = _snapshot(model, context)
    if not Bool(py=py_item(fields, nk[].k_fit_executed_flag)):
        raise Error("ValueError: " + _HW + "fit() the model before save()")
    var n = Int(py=py_item(fields, nk[].k_n))
    var b = Int(py=py_item(fields, nk[].k_ts_num))
    var period = Int(py=py_item(fields, nk[].k_seasonal_periods))
    var mode = py_get(fields, nk[].k_numeric_mode)
    if mode is None or (py_type(mode) is _b().str and len(mode) == 0):
        mode = py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
    if py_type(mode) is not _b().str:
        raise Error("ValueError: cannot save invalid numeric_mode")
    mode = py_call(py_call(mode.strip).lower)
    if String(mode) != "fast" and String(mode) != "identical":
        raise Error("ValueError: cannot save invalid numeric_mode")
    var values = Python.dict(format="mojolearn-holtwinters-1", estimator="ExponentialSmoothing",
        numeric_mode=mode, seasonal=py_item(fields, nk[].k_seasonal), initialization_method=py_item(fields, nk[].k_initialization_method))
    var components_pin = AdmittedTypedPin()
    _ = _packed(py_item(fields, nk[].k__comps), 3 * (n - period) * b, context, components_pin)
    py_setitem(values, nk[].k_components, components_pin.keep)
    var names = Python.tuple("sse", "alpha", "beta", "gamma", "n_iter", "criterion")
    for i in range(len(names)):  # driver-loop: snapshot all six saved native statistics before opening the archive
        var name = String(names[i])
        var stored_pin = AdmittedTypedPin()
        var stored = admit_view(fields[name + "_"], name, context, stored_pin)
        var skind = admitted_view_kind(stored, name)
        var want = CAST_KIND_FLOAT if i < 4 else CAST_KIND_SIGNED
        if stored.count() != b or not stored.c_contig or skind[1] != 4 or skind[0] != want:
            raise Error("ValueError: Holt-Winters archive statistics have invalid dtype or extent")
        py_setitem(values, names[i], stored_pin.keep)
    var meta = tsa_array("q", "<i8", Python.tuple(4), 4, 8, context)
    var words = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(meta)))
    words[0] = Int64(n)
    words[1] = Int64(b)
    words[2] = Int64(period)
    words[3] = Int64(Int(py=py_item(fields, nk[].k_start_periods)))
    var eps = tsa_array("d", "<f8", Python.tuple(1), 1, 8, context)
    MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(eps)))[0] = Float64(py=py_item(fields, nk[].k_eps))
    py_setitem(values, nk[].k_meta, meta)
    py_setitem(values, nk[].k_eps, eps)
    _ = components_pin^
    return classical_archive_write(args[0], values, context)
