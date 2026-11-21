# SPDX-License-Identifier: Apache-2.0
"""`ARIMA.fit` and `ARIMA.predict` (python/mojolearn/_arima_impl.py) as one
native call each, through `tsa_native_call` (bindings/tsa_public_native.mojo):

    fit      `_series_major` (y admitted 1-D or 2-D, C-ordered float32
             (batch_size, n_obs)), `_exog_array`'s shapes and refusals, the
             information-criterion penalties (the pinned `log` entry), one
             `arima_fit` on the model's tier, then `_adopt_fit`'s state
    predict  `_check_fitted`, the width refusal, `_future_exog`'s three
             checks in cuML's order and words, one `arima_predict`

UNPACK and PACK only: every kernel call goes through RUN
(core/arima_run.mojo) on the estimator's route
(bindings/tsa_arima_route_unpack.mojo: the device is read once into the
route owner; no call here reads or compares it), with the GIL released. The
argument layout is preserved. Compilation does not establish runtime
behavior or within-release bitwise identity. AutoARIMA's doors pass their
own route (`arima_fit_routed`), so a per-group model resolves none.
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from arima.tsa_arima_run import _penalties
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from bindings.mode_metadata_native_door import _fields
from providers.python.arima_lifecycle_pyapi_door import arima_order
from providers.python.autoarima_lifecycle_pyapi_door import _buffer, _packed, _integer, _integer_object
from providers.python.arima_buffer_pyapi_door import ArimaBuffer, arima_buffer
from providers.python.forecast_owner_pyapi_unpack import forecast_commit
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains
from bindings.classical_arrays_door import classical_input, classical_address, classical_cast_kind
from bindings.classical_pandas_door import classical_admit_value
from bindings.tsa_hw_native_door import tsa_array
from core.xseq_marshal import xseq_widen_f32_f64
from core.admitted_input_nd import admitted_cast_nd_f32
from bindings.admitted_sequence_door import admitted_array_f32
from bindings.native_prepared_owner_unpack import prepared_get
from bindings.tsa_arima_resident_unpack import (
    arima_resident_hit,
    arima_resident_call,
)
from bindings.tsa_hw_route_unpack import tsa_hw_resident_handle
from bindings.tsa_arima_resident_pack import (
    ARIMA_RESIDENT_SLOT,
    arima_resident_mint,
)
from arima.tsa_arima_resident_run import arima_resident_words
from bindings.tsa_arima_route_unpack import arima_route
from bindings.tsa_hw_route_unpack import TsaHwCall
from core.arima_run import (
    arima_fit_run, arima_forecast_run, arima_ic_run, arima_predict_run, arima_route_fits, arima_route_has_ic,
)
from core.device_route import DeviceRoute, DEVICE_INSTALL
from bindings.tsa_keys_unpack import tsa_keys
from bindings.py_balanced import py_item

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _A = "mojolearn ARIMA: "


def _b() raises -> PythonObject:
    # The import-time builtins connection: no __import__ hook can run.
    return native_loaded_module("builtins")


def _repr(value: PythonObject) raises -> String:
    return String(py=py_call(_b().repr, value))


def _shape_tuple(value: PythonObject) raises -> PythonObject:
    return py_call(_b().tuple, value.shape)


def _copy_series(source: ArimaBuffer, shape: PythonObject, context: PythonObject) raises -> PythonObject:
    var result = admitted_array_f32(shape, source.count, context)
    _ = admitted_cast_nd_f32(source.address, source.shape, source.strides, source.itemsize, source.kind,
                 _F32P(unsafe_from_address=Int(py=classical_address(result))))
    _ = source
    return result


def _series_major(source: PythonObject, context: PythonObject) raises -> PythonObject:
    var captured = arima_buffer(source, "y", context)
    var rank = len(captured.shape)
    if (rank != 1 and rank != 2) or captured.count == 0:
        raise Error("ValueError: " + _A + "y must be nonempty 1-D or 2-D (batch_size, n_obs)")
    var rows = 1 if rank == 1 else captured.shape[0]
    var cols = captured.shape[0] if rank == 1 else captured.shape[1]
    var result = _copy_series(captured, Python.tuple(rows, cols), context)
    # pandas storage is converted natively, so the caller's input was copied.
    return Python.tuple(result, rows, cols, captured.converted or not captured.f32_c(), rank)


def _no_exog(context: PythonObject) raises -> PythonObject:
    return tsa_array("f", "<f4", Python.tuple(1), 1, 4, context)


def _exog_array(source: PythonObject, batch: Int, n_rows: Int, y_ndim: Int, name: String,
                context: PythonObject) raises -> PythonObject:
    """`_exog_array`: (C-ordered float32 (batch_size, n_rows, n_exog), n_exog)."""
    var captured = arima_buffer(source, name, context)
    var rank = len(captured.shape)
    var shown = String("(")
    for axis in range(rank):  # driver-loop: captured shape words for a refusal message
        shown += String(captured.shape[axis]) + ","
    shown += ")"
    var d0 = captured.shape[0] if rank >= 1 else 0
    var d1 = captured.shape[1] if rank >= 2 else 0
    var d2 = captured.shape[2] if rank >= 3 else 0
    var want = PythonObject(None)
    var got = PythonObject(None)
    var s0: Int
    var s1: Int
    var s2: Int
    if rank == 3:
        want = Python.tuple(batch, n_rows)
        got = Python.tuple(d0, d1)
        s0 = d0
        s1 = d1
        s2 = d2
    elif rank == 2 and y_ndim == 1:
        want = Python.tuple(n_rows)
        got = Python.tuple(d0)
        s0 = 1
        s1 = d0
        s2 = d1
    elif rank == 2:
        want = Python.tuple(batch, n_rows)
        got = Python.tuple(d0, d1)
        s0 = d0
        s1 = d1
        s2 = 1
    elif rank == 1 and y_ndim == 1:
        want = Python.tuple(n_rows)
        got = Python.tuple(d0)
        s0 = 1
        s1 = d0
        s2 = 1
    else:
        raise Error("ValueError: " + _A + name + " must be (batch_size, n_rows, n_exog); a 1-D y also takes (n_rows,) "
                    + "or (n_rows, n_exog) and a 2-D y takes (batch_size, n_rows) for one regressor. Got "
                    + String(rank) + "-D shape " + shown + " beside a " + String(y_ndim) + "-D y")
    if not Bool(py=got == want):
        raise Error("ValueError: " + _A + "dimensions mismatch, " + name + " has shape " + shown + " and must lead with "
                    + _repr(want) + " (batch_size=" + String(batch) + ", rows=" + String(n_rows) + ")")
    if s2 < 1:
        raise Error("ValueError: " + _A + name + " has no regressor columns, shape " + shown)
    return Python.tuple(_copy_series(captured, Python.tuple(s0, s1, s2), context), s2)


def _order(fields: PythonObject, context: PythonObject) raises -> List[Int]:
    """[p, d, q, P, D, Q, s]."""
    var nk = tsa_keys()
    var o = arima_order(py_item(fields, nk[].k_order), 3, "order", context)
    var so = arima_order(py_item(fields, nk[].k_seasonal_order), 4, "seasonal_order", context)
    return [Int(py=o[0]), Int(py=o[1]), Int(py=o[2]), Int(py=so[0]), Int(py=so[1]), Int(py=so[2]), Int(py=so[3])]


def _copy_block(source: PythonObject, offset: Int, count: Int, code: String, typestr: String, width: Int,
                shape: PythonObject, context: PythonObject) raises -> PythonObject:
    """A copy of `count` items (of `width` bytes) of `source` from item
    `offset`, as an Array of `shape` (the Python body's slice or reshape)."""
    var out = tsa_array(code, typestr, shape, count, width, context)
    if count > 0:
        unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(out))),
               src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(source)) + width * offset),
               count=count * width)
        _ = out  # keep the Python object alive through the native call above
    return out


def arima_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`ARIMA.fit` on the model's own route."""
    return _arima_fit(model, args, context, False, DeviceRoute(DEVICE_INSTALL), PythonObject(None))


def arima_fit_routed(model: PythonObject, args: PythonObject, context: PythonObject, call: TsaHwCall) raises -> PythonObject:
    """`ARIMA.fit` of an AutoARIMA per-group model on its AutoARIMA's route
    (the caller keeps `call` named through this)."""
    return _arima_fit(model, args, context, True, call.route, call.owner)


def _arima_fit(model: PythonObject, args: PythonObject, context: PythonObject, routed: Bool,
               given_route: DeviceRoute, given_owner: PythonObject) raises -> PythonObject:
    var nk = tsa_keys()
    var fields = _fields(model)
    var series = _series_major(args[0], context)
    var arr = series[0]
    var batch = Int(py=series[1])
    var n_obs = Int(py=series[2])
    var y_ndim = Int(py=series[4])
    var order = _order(fields, context)
    var ex = PythonObject(None)
    var n_exog = 0
    if args[1] is None:
        ex = _no_exog(context)
    else:
        var e = _exog_array(args[1], batch, n_obs, y_ndim, "exog", context)
        ex = e[0]
        n_exog = Int(py=e[1])
    var k = _integer(py_item(fields, nk[].k_k_), context)
    if k < 0 or k > 1:
        raise Error("ValueError: ARIMA k must be 0 or 1")
    var method = py_item(fields, nk[].k_method)
    if py_type(method) is not _b().str:
        raise Error("TypeError: ARIMA method requires an exact string")
    var maxiter = _integer(py_item(fields, nk[].k_maxiter), context)
    var big_n = order[0] + order[2] + order[3] + order[5] + k + n_exog + 1
    var total = batch * big_n
    var params = tsa_array("f", "<f4", Python.tuple(batch, big_n), total, 4, context)
    var x = tsa_array("f", "<f4", Python.tuple(batch, big_n), total, 4, context)
    var x0 = tsa_array("f", "<f4", Python.tuple(batch, big_n), total, 4, context)
    var stats = tsa_array("f", "<f4", Python.tuple(2 * batch), 2 * batch, 4, context)
    var flags = tsa_array("i", "<i4", Python.tuple(2 * batch), 2 * batch, 4, context)
    var ics = tsa_array("d", "<f8", Python.tuple(2 * batch), 2 * batch, 8, context)
    var pens = _penalties(order, big_n, n_obs)
    var call = TsaHwCall(given_route, given_owner)
    if not routed:
        call = arima_route(model, fields, context, _b())
    var route = call.route
    if not arima_route_fits(route):
        raise Error("AttributeError: the ARIMA binding has no arima_fit")
    var method_code = Int(py=py_item(context, nk[].k_arima_methods)[method])
    var arr_address = Int(py=classical_address(arr))
    var ex_address = Int(py=classical_address(ex))
    var params_address = Int(py=classical_address(params))
    var x_address = Int(py=classical_address(x))
    var x0_address = Int(py=classical_address(x0))
    var stats_address = Int(py=classical_address(stats))
    var flags_address = Int(py=classical_address(flags))
    var ics_address = Int(py=classical_address(ics))
    var written = 0
    var py = Python()
    try:
        with GILReleased(py):
            written = arima_fit_run(route, arr_address, ex_address, params_address, x_address, x0_address,
                                    stats_address, flags_address, ics_address, batch, n_obs, order, k, n_exog,
                                    method_code, maxiter, pens[0], pens[1])
    finally:
        _ = ex  # keep the Python object alive through the native call above
        _ = arr  # keep the Python object alive through the native call above
        _ = x0  # keep the Python object alive through the native call above
        _ = x  # keep the Python object alive through the native call above
        _ = params  # keep the Python object alive through the native call above
        _ = ics  # keep the Python object alive through the native call above
        _ = flags  # keep the Python object alive through the native call above
        _ = stats  # keep the Python object alive through the native call above
        _ = call.owner  # the route owner pins the image through the call
    if written != total:
        raise Error("RuntimeError: " + _A + "the fit wrote " + String(written) + " parameters, batch_size * N is "
                    + String(total) + "; this side and arima/estimator.mojo disagree about ARIMAOrder.complexity()")
    _adopt(model, fields, arr, Bool(py=series[3]), y_ndim, batch, n_obs, ex, n_exog, big_n, params, x, x0, stats,
           flags, ics, context, route, call.owner)
    return model


def _adopt(model: PythonObject, fields: PythonObject, arr: PythonObject, copied: Bool, y_ndim: Int, batch: Int,
           n_obs: Int, ex: PythonObject, n_exog: Int, big_n: Int, params: PythonObject, x: PythonObject,
           x0: PythonObject, stats: PythonObject, flags: PythonObject, ics_in: PythonObject,
           context: PythonObject, route: DeviceRoute, route_owner: PythonObject) raises:
    """`ARIMA._adopt_fit`: the fitted state from the fit's output buffers;
    without the fit's criteria (`ics_in` None: an AutoARIMA search block)
    the plain fit's device kernel over the adopted log-likelihood row, on
    `route` (`route_owner` pins its image through that call)."""
    var nk = tsa_keys()
    var ics = ics_in
    if ics is None:
        ics = tsa_array("d", "<f8", Python.tuple(2 * batch), 2 * batch, 8, context)
        var pens = _penalties(_order(fields, context), big_n, n_obs)
        if not arima_route_has_ic(route):
            raise Error("AttributeError: the ARIMA binding has no arima_ic_from_loglike")
        var stats_address = Int(py=classical_address(stats))
        var ics_address = Int(py=classical_address(ics))
        var written = 0
        var py = Python()
        try:
            with GILReleased(py):
                written = arima_ic_run(route, stats_address, ics_address, batch, pens[0], pens[1])
        finally:
            _ = ics  # keep the Python object alive through the native call above
            _ = stats  # keep the Python object alive through the native call above
            _ = route_owner  # the route owner pins the image through the call
        if written != batch:
            raise Error("RuntimeError: " + _A + "the criterion kernel wrote " + String(written)
                        + " series, batch_size is " + String(batch))
    var updates = Python.dict()
    py_setitem(updates, nk[].k_input_copied_, copied)
    py_setitem(updates, nk[].k_batch_size_, batch)
    py_setitem(updates, nk[].k_n_obs_, n_obs)
    py_setitem(updates, nk[].k__y, arr)
    py_setitem(updates, nk[].k__y_ndim, y_ndim)
    py_setitem(updates, nk[].k_n_exog_, n_exog)
    py_setitem(updates, nk[].k_complexity_, big_n)
    py_setitem(updates, nk[].k__exog, ex if n_exog else PythonObject(None))
    py_setitem(updates, nk[].k_params_, params)
    py_setitem(updates, nk[].k_x_, x)
    py_setitem(updates, nk[].k_x0_, x0)
    py_setitem(updates, nk[].k_n_iter_, _copy_block(flags, 0, batch, "i", "<i4", 4, Python.tuple(batch), context))
    py_setitem(updates, nk[].k_retcode_, _copy_block(flags, batch, batch, "i", "<i4", 4, Python.tuple(batch), context))
    # llf_ = stats[:batch].astype("<f8"): each float32 widened exactly
    var llf = tsa_array("d", "<f8", Python.tuple(batch), batch, 8, context)
    xseq_widen_f32_f64(_F32P(unsafe_from_address=Int(py=classical_address(stats))),
                       MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(llf))), batch)
    _ = llf  # keep the Python object alive through the native call above
    py_setitem(updates, nk[].k_llf_, llf)
    py_setitem(updates, nk[].k_fx_, _copy_block(stats, batch, batch, "f", "<f4", 4, Python.tuple(batch), context))
    py_setitem(updates, nk[].k_aic_, _copy_block(ics, 0, batch, "d", "<f8", 8, Python.tuple(batch), context))
    py_setitem(updates, nk[].k_bic_, _copy_block(ics, batch, batch, "d", "<f8", 8, Python.tuple(batch), context))
    forecast_commit(fields, updates, py_item(context, nk[].k_tsa_owner_context))


def arima_predict(model: PythonObject, args: PythonObject, context: PythonObject, forecast: Bool = False) raises -> PythonObject:
    var nk = tsa_keys()
    # The route is resolved where the former entry selection ran: after the
    # argument and fitted-state refusals, once per call.
    var call = TsaHwCall(DeviceRoute(DEVICE_INSTALL), PythonObject(None))
    var routed = False
    var fields = _fields(model)
    if not py_contains(fields, nk[].k_params_):
        raise Error("ValueError: " + _A + "call fit() before " + ("forecast" if forecast else "predict"))
    var n_obs = _integer(py_item(fields, nk[].k_n_obs_), context)
    var start = n_obs if forecast else _integer(args[0], context)
    var end = Int(py=PythonObject(n_obs) + _integer_object(args[0], context)) if forecast else n_obs if args[1] is None else _integer(args[1], context)
    var exog = args[1] if forecast else args[2]
    var order = _order(fields, context)
    var width = Int(py=PythonObject(end) - start)
    if width <= 0:
        raise Error("ValueError: " + _A + ("steps must be >= 1" if forecast else "need start < end; end is excluded"))
    var n_exog = _integer(py_get(fields, nk[].k_n_exog_, PythonObject(0)), context)
    if n_exog > 0 and end > n_obs and exog is None:
        raise Error("ValueError: " + _A + "the model was fit with a regression component, so future values must be provided via `exog`")
    if n_exog == 0 and exog is not None:
        raise Error("ValueError: " + _A + "a value was given for `exog` but the model was fit without any regression component")
    if end <= n_obs and exog is not None:
        raise Error("ValueError: " + _A + "a value was given for `exog` but only in-sample predictions were requested")
    var batch = _integer(py_item(fields, nk[].k_batch_size_), context)
    var k = _integer(py_item(fields, nk[].k_k_), context)
    if batch <= 0 or n_obs <= 0 or n_exog < 0 or k < 0 or k > 1:
        raise Error("ValueError: ARIMA fitted dimension metadata is invalid")
    var complexity = Int(py=PythonObject(order[0]) + order[2] + order[3] + order[5] + k + n_exog + 1)
    var words = arima_resident_words(batch, n_obs, order, k, n_exog)
    var mods = NativeModules()
    var record = prepared_get(mods, ARIMA_RESIDENT_SLOT, model)
    if record is not None:
        # A resident record exists: the fitted arrays it names were admitted
        # and checked against these words when it was minted and are frozen
        # since, so a hit skips their admission and upload; only the future
        # regressors are admitted (the same checks, the same words).
        call = arima_route(model, fields, context, _b())
        routed = True
        var hit = arima_resident_hit(record, fields, call, words)
        if hit >= 0:
            var fut = PythonObject(None)
            var fut_addr = 0
            var fut_owner = PythonObject(None)
            if n_exog > 0 and end > n_obs:
                var fe = _exog_array(exog, batch, end - n_obs, _integer(py_get(fields, nk[].k__y_ndim, PythonObject(2)), context), "exog", context)
                if Int(py=fe[1]) != n_exog:
                    # The same wording as the cold path below, so a resident
                    # hit and a first query refuse one bad exog identically.
                    raise Error("ValueError: " + _A + "dimensions mismatch, `exog` has " + String(Int(py=fe[1]))
                                + " regressor column(s) and the model was fit with " + String(n_exog))
                fut = fe[0]
                var fb = arima_buffer(fut, "prepared future exog", context, True)
                fut_addr = fb.address
                fut_owner = Python.tuple(fb.keep, fb.owner)
                _ = fb
            var out = tsa_array("f", "<f4", Python.tuple(batch, width), Int(py=PythonObject(batch) * width), 4, context)
            if fut_addr == 0:
                # an unread future pointer must still be a live address
                fut_addr = Int(py=classical_address(out))
            arima_resident_call(call, hit, fut_addr, Int(py=classical_address(out)), start, end, width, forecast)
            _ = out  # keep the Python object alive through the native call above
            _ = record  # the record (and its lifetime owner) outlive the query
            _ = fut_owner
            _ = fut
            return out
    var y = arima_buffer(py_item(fields, nk[].k__y), "y", context, True)
    var params = arima_buffer(py_item(fields, nk[].k_params_), "params", context, True)
    if y.count != Int(py=PythonObject(batch) * n_obs) or params.count != Int(py=PythonObject(batch) * complexity):
        raise Error("ValueError: ARIMA fitted buffer dimensions disagree with model metadata")
    # Unused exog pointers remain live through the admitted y owner.
    var ex_addr = y.address
    var future_addr = y.address
    var ex_owner = y.copy()
    var future_owner = y.copy()
    var future = PythonObject(None)
    if n_exog > 0:
        ex_owner = arima_buffer(py_item(fields, nk[].k__exog), "exog", context, True)
        if ex_owner.count != Int(py=PythonObject(batch) * n_obs * n_exog):
            raise Error("ValueError: ARIMA fitted exog dimensions disagree")
        ex_addr = ex_owner.address
        if end > n_obs:
            var e = _exog_array(exog, batch, end - n_obs, _integer(py_get(fields, nk[].k__y_ndim, PythonObject(2)), context), "exog", context)
            if Int(py=e[1]) != n_exog:
                raise Error("ValueError: " + _A + "dimensions mismatch, `exog` has " + String(Int(py=e[1]))
                            + " regressor column(s) and the model was fit with " + String(n_exog))
            future = e[0]
            future_owner = arima_buffer(future, "prepared future exog", context, True)
            future_addr = future_owner.address
    var result = tsa_array("f", "<f4", Python.tuple(batch, width), Int(py=PythonObject(batch) * width), 4, context)
    if not routed:
        call = arima_route(model, fields, context, _b())
    var result_address = Int(py=classical_address(result))
    var minted = arima_resident_mint(model, fields, call, words, y.address, ex_addr, params.address)
    if minted is not None:
        arima_resident_call(call, tsa_hw_resident_handle(minted[3]), future_addr, result_address, start, end, width,
                            forecast)
        _ = result  # keep the Python object alive through the native call above
        _ = minted  # the record (and its lifetime owner) outlive the query
        _ = y
        _ = params
        _ = ex_owner
        _ = future_owner
        _ = future
        return result
    var route = call.route
    var y_address = y.address
    var params_address = params.address
    var py = Python()
    try:
        with GILReleased(py):
            if forecast:
                _ = arima_forecast_run(route, y_address, ex_addr, future_addr, params_address, result_address,
                                       batch, n_obs, width, order, k, n_exog)
            else:
                _ = arima_predict_run(route, y_address, ex_addr, future_addr, params_address, result_address,
                                      batch, n_obs, start, end, order, k, n_exog)
    finally:
        _ = result  # keep the Python object alive through the native call above
        _ = y
        _ = params
        _ = ex_owner
        _ = future_owner
        _ = future
        _ = call.owner  # the route owner pins the image through the call
    return result


def arima_forecast(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return arima_predict(model, args, context, True)
