# SPDX-License-Identifier: Apache-2.0
"""`ARIMA.fit` and `ARIMA.predict` (python/mojolearn/_arima_impl.py) as one
native call each, through `tsa_native_call` (bindings/tsa_public_native.mojo):

    fit      `_series_major` (y admitted 1-D or 2-D, C-ordered float32
             (batch_size, n_obs)), `_exog_array`'s shapes and refusals, the
             information-criterion penalties (the pinned `log` entry), one
             `arima_fit` on the model's tier, then `_adopt_fit`'s state
    predict  `_check_fitted`, the width refusal, `_future_exog`'s three
             checks in cuML's order and words, one `arima_predict`

The numerical entry selection and argument layout are preserved. Compilation
does not establish runtime behavior or within-release bitwise identity.
A host inference class (`HostARIMA`) reads its loaded forecast host binding
from the module table.
"""
from std.memory import memcpy
from std.python import Python, PythonObject
from x_metrics.epilogue import portable_log_c
from bindings.native_loaded_module import native_loaded_module, _loaded_fields
from bindings.mode_metadata_native import _fields, _class_fields, _truth
from providers.python.arima_lifecycle_pyapi import arima_order
from providers.python.autoarima_lifecycle_pyapi import _buffer, _packed, _integer, _integer_object
from providers.python.arima_buffer_pyapi import ArimaBuffer, arima_buffer
from providers.python.forecast_owner_pyapi import forecast_commit
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.classical_arrays import classical_input, classical_address, classical_cast_kind
from bindings.classical_pandas import classical_admit_value
from bindings.tsa_hw_native import tsa_array
from core.xseq_marshal import xseq_cast_nd, xseq_widen_f32_f64

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _A = "mojolearn ARIMA: "


def _b() raises -> PythonObject:
    # The import-time builtins connection: no __import__ hook can run.
    return native_loaded_module("builtins")


def _repr(value: PythonObject) raises -> String:
    return String(py=py_call(_b().repr, value))


def _shape_tuple(value: PythonObject) raises -> PythonObject:
    return py_call(_b().tuple, value.shape)


def _arima_entries(model: PythonObject, fields: PythonObject, context: PythonObject) raises -> PythonObject:
    """`ARIMA._extension()`: the `_mojolearn_arima` entries of the model's
    tier with the binary's own mode read back; a host inference class's
    loaded forecast host binding."""
    var b = _b()
    var cls = py_type(model)
    var ancestry = py_call(b.type.__dict__["__mro__"].__get__, cls)
    var host_only = False
    for i in range(len(ancestry)):  # driver-loop: native MRO metadata, never class lookup protocols
        var namespace = _class_fields(ancestry[i])
        if py_contains(namespace, "_HOST_INFERENCE_ONLY"):
            host_only = _truth(namespace["_HOST_INFERENCE_ONLY"])
            break
    if host_only:
        var saved = py_get(fields, "numeric_mode")
        if saved is not None and (py_type(saved) is not b.str or not Bool(py=saved == "identical")):
            raise Error("ValueError: mojolearn: " + String("ARIMA")
                        + " runs IDENTICAL only on the host; this model was saved " + _repr(saved))
        var module = py_get(context["modules"], context["hw_host_module"])
        if module is None:
            raise Error("RuntimeError: mojolearn: the host " + String("ARIMA")
                        + " has no loaded forecast binding; load the model with its own load()")
        var module_fields = _loaded_fields(module)
        var host = b.dict()
        var names = context["arima_entry_names"]
        for i in range(len(names)):  # driver-loop: over the fixed host entry names
            var entry = py_get(module_fields, names[i])
            if entry is not None:
                if py_type(entry) is not py_type(b.len):
                    raise Error("TypeError: host ARIMA entries must be compiled native functions")
                py_setitem(host, names[i], entry)
        return host
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not b.str:
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    var want = py_call(py_call(mode.strip).lower)
    if String(py=want) != "identical" and String(py=want) != "fast":
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    var entries = context["arima_bindings"][want]
    if py_type(entries) is b.str:
        raise Error(String(py=entries))
    if py_contains(entries, "arima_numeric_mode"):
        var got = Int(py=py_call_typed(entries["arima_numeric_mode"]))
        if got != (1 if String(py=want) == "identical" else 0):
            raise Error("RuntimeError: " + _A + "numeric_mode=" + _repr(want) + " was requested but "
                        + String(py=entries["module_name"]) + " reports compile-time mode code " + String(got)
                        + "; the binary and the directory it sits in disagree, rebuild it")
    return entries


def _entry(entries: PythonObject, name: String) raises -> PythonObject:
    if not py_contains(entries, name):
        raise Error("AttributeError: the ARIMA binding has no " + name)
    return entries[name]


def _copy_series(source: ArimaBuffer, shape: PythonObject, context: PythonObject) raises -> PythonObject:
    var result = tsa_array("f", "<f4", shape, source.count, 4, context)
    xseq_cast_nd(source.address, source.shape, source.strides, source.itemsize, source.kind, True,
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
    var o = arima_order(fields["order"], 3, "order", context)
    var so = arima_order(fields["seasonal_order"], 4, "seasonal_order", context)
    return [Int(py=o[0]), Int(py=o[1]), Int(py=o[2]), Int(py=so[0]), Int(py=so[1]), Int(py=so[2]), Int(py=so[3])]


def _penalties(order: List[Int], big_n: Int, n_obs: Int, context: PythonObject) raises -> Tuple[Float64, Float64]:
    """`_ic_penalties`: 2N, and log(T) N with the pinned log (T = n_obs after
    differencing)."""
    var t = n_obs - (order[1] + order[6] * order[4])
    var n_par = Float64(big_n)
    var log_t = Float64(0.0)
    if t > 0:
        # the pinned log: portable_log_c is packaging/portable_math.c `mojolearn_log`
        # operation for operation (what `_portable_math.log` returns)
        log_t = portable_log_c(Float64(t))
    return (2.0 * n_par, log_t * n_par)


def _copy_block(source: PythonObject, offset: Int, count: Int, code: String, typestr: String, width: Int,
                shape: PythonObject, context: PythonObject) raises -> PythonObject:
    """A copy of `count` items (of `width` bytes) of `source` from item
    `offset`, as an Array of `shape` (the Python body's slice or reshape)."""
    var out = tsa_array(code, typestr, shape, count, width, context)
    if count > 0:
        memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(out))),
               src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(source)) + width * offset),
               count=count * width)
        _ = out  # keep the Python object alive through the native call above
    return out


def arima_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
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
    var k = _integer(fields["k_"], context)
    if k < 0 or k > 1:
        raise Error("ValueError: ARIMA k must be 0 or 1")
    var method = fields["method"]
    if py_type(method) is not _b().str:
        raise Error("TypeError: ARIMA method requires an exact string")
    var maxiter = _integer_object(fields["maxiter"], context)
    var big_n = order[0] + order[2] + order[3] + order[5] + k + n_exog + 1
    var total = batch * big_n
    var params = tsa_array("f", "<f4", Python.tuple(batch, big_n), total, 4, context)
    var x = tsa_array("f", "<f4", Python.tuple(batch, big_n), total, 4, context)
    var x0 = tsa_array("f", "<f4", Python.tuple(batch, big_n), total, 4, context)
    var stats = tsa_array("f", "<f4", Python.tuple(2 * batch), 2 * batch, 4, context)
    var flags = tsa_array("i", "<i4", Python.tuple(2 * batch), 2 * batch, 4, context)
    var ics = tsa_array("d", "<f8", Python.tuple(2 * batch), 2 * batch, 8, context)
    var pens = _penalties(order, big_n, n_obs, context)
    var entries = _arima_entries(model, fields, context)
    var plan = Python.list(batch, n_obs, order[0], order[1], order[2], order[3], order[4], order[5], order[6], k,
                           n_exog, context["arima_methods"][method], maxiter,
                           classical_address(ics), pens[0], pens[1])
    _ = ics  # keep the Python object alive through the native call above
    var written = Int(py=py_call_typed(_entry(entries, "arima_fit"), classical_address(arr), classical_address(ex),
                                       classical_address(params), classical_address(x), classical_address(x0),
                                       classical_address(stats), classical_address(flags), plan))
    _ = ex  # keep the Python object alive through the native call above
    _ = arr  # keep the Python object alive through the native call above
    _ = x0  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _ = params  # keep the Python object alive through the native call above
    _ = ics  # keep the Python object alive through the native call above
    _ = flags  # keep the Python object alive through the native call above
    _ = stats  # keep the Python object alive through the native call above
    if written != total:
        raise Error("RuntimeError: " + _A + "the fit wrote " + String(written) + " parameters, batch_size * N is "
                    + String(total) + "; this side and arima/estimator.mojo disagree about ARIMAOrder.complexity()")
        _ = flags  # keep the Python object alive through the native call above
        _ = stats  # keep the Python object alive through the native call above
        _ = x0  # keep the Python object alive through the native call above
        _ = x  # keep the Python object alive through the native call above
        _ = params  # keep the Python object alive through the native call above
        _ = ex  # keep the Python object alive through the native call above
        _ = arr  # keep the Python object alive through the native call above
    _ = flags  # keep the Python object alive through the native calls above
    _ = stats  # keep the Python object alive through the native calls above
    _ = x0  # keep the Python object alive through the native calls above
    _ = x  # keep the Python object alive through the native calls above
    _ = params  # keep the Python object alive through the native calls above
    _ = ex  # keep the Python object alive through the native calls above
    _ = arr  # keep the Python object alive through the native calls above
    _adopt(model, fields, arr, Bool(py=series[3]), y_ndim, batch, n_obs, ex, n_exog, big_n, params, x, x0, stats,
           flags, ics, context)
    return model


def _adopt(model: PythonObject, fields: PythonObject, arr: PythonObject, copied: Bool, y_ndim: Int, batch: Int,
           n_obs: Int, ex: PythonObject, n_exog: Int, big_n: Int, params: PythonObject, x: PythonObject,
           x0: PythonObject, stats: PythonObject, flags: PythonObject, ics_in: PythonObject,
           context: PythonObject) raises:
    """`ARIMA._adopt_fit`: the fitted state from the fit's output buffers;
    without the fit's criteria (`ics_in` None: an AutoARIMA search block)
    the plain fit's device kernel over the adopted log-likelihood row."""
    var ics = ics_in
    if ics is None:
        ics = tsa_array("d", "<f8", Python.tuple(2 * batch), 2 * batch, 8, context)
        var pens = _penalties(_order(fields, context), big_n, n_obs, context)
        var entries = _arima_entries(model, fields, context)
        var written = Int(py=py_call_typed(_entry(entries, "arima_ic_from_loglike"), classical_address(stats),
                                           classical_address(ics), Python.list(batch, pens[0], pens[1])))
        _ = ics  # keep the Python object alive through the native call above
        if written != batch:
            raise Error("RuntimeError: " + _A + "the criterion kernel wrote " + String(written)
                        + " series, batch_size is " + String(batch))
            _ = ics  # keep the Python object alive through the native call above
        _ = ics  # keep the Python object alive through the native calls above
    var updates = Python.dict()
    py_setitem(updates, "input_copied_", copied)
    py_setitem(updates, "batch_size_", batch)
    py_setitem(updates, "n_obs_", n_obs)
    py_setitem(updates, "_y", arr)
    py_setitem(updates, "_y_ndim", y_ndim)
    py_setitem(updates, "n_exog_", n_exog)
    py_setitem(updates, "complexity_", big_n)
    py_setitem(updates, "_exog", ex if n_exog else PythonObject(None))
    py_setitem(updates, "params_", params)
    py_setitem(updates, "x_", x)
    py_setitem(updates, "x0_", x0)
    py_setitem(updates, "n_iter_", _copy_block(flags, 0, batch, "i", "<i4", 4, Python.tuple(batch), context))
    py_setitem(updates, "retcode_", _copy_block(flags, batch, batch, "i", "<i4", 4, Python.tuple(batch), context))
    # llf_ = stats[:batch].astype("<f8"): each float32 widened exactly
    var llf = tsa_array("d", "<f8", Python.tuple(batch), batch, 8, context)
    xseq_widen_f32_f64(_F32P(unsafe_from_address=Int(py=classical_address(stats))),
                       MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(llf))), batch)
    _ = llf  # keep the Python object alive through the native call above
    py_setitem(updates, "llf_", llf)
    py_setitem(updates, "fx_", _copy_block(stats, batch, batch, "f", "<f4", 4, Python.tuple(batch), context))
    py_setitem(updates, "aic_", _copy_block(ics, 0, batch, "d", "<f8", 8, Python.tuple(batch), context))
    py_setitem(updates, "bic_", _copy_block(ics, batch, batch, "d", "<f8", 8, Python.tuple(batch), context))
    forecast_commit(fields, updates, context["tsa_owner_context"])


def _gather_block(src: PythonObject, offset: Int, width: Int, nrows: Int, local: PythonObject, dst_addr: Int,
                  context: PythonObject) raises:
    """`_gather_block`: dst rows = src[offset:].reshape(nrows, width)[local],
    the base binding's byte gather of 4-byte rows."""
    if len(local):
        var item = Int(py=src.itemsize)
        _ = py_call(context["base_gather_rows_bytes"], Int(py=classical_address(src)) + item * offset, dst_addr,
                    classical_address(local), nrows, len(local), item * width)


def arima_adopt(model: PythonObject, y: PythonObject, reuse: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_adopt_search_fit(m, y, *reuse)`: `m` fitted from the AutoARIMA
    search's fit block for the group positions `local`."""
    var fields = _fields(model)
    var search_fit = reuse[0]
    var trial = Int(py=reuse[1])
    var local = reuse[2]
    var fit_f32 = search_fit[0]
    var fit_i32 = search_fit[1]
    var off = Int(py=search_fit[2][trial])
    var nb = Int(py=search_fit[3])
    var grouped_ll = search_fit[4]
    var series = _series_major(y, context)
    var arr = series[0]
    var b = Int(py=series[1])
    var n_obs = Int(py=series[2])
    var big_n = Int(py=fields["complexity_"])
    var params = tsa_array("f", "<f4", Python.tuple(b, big_n), b * big_n, 4, context)
    var x = tsa_array("f", "<f4", Python.tuple(b, big_n), b * big_n, 4, context)
    var x0 = tsa_array("f", "<f4", Python.tuple(b, big_n), b * big_n, 4, context)
    var stats = tsa_array("f", "<f4", Python.tuple(2 * b), 2 * b, 4, context)
    var flags = tsa_array("i", "<i4", Python.tuple(2 * b), 2 * b, 4, context)
    _gather_block(fit_f32, off, big_n, nb, local, Int(py=classical_address(params)), context)
    _ = params  # keep the Python object alive through the native call above
    _gather_block(fit_f32, off + nb * big_n, big_n, nb, local, Int(py=classical_address(x)), context)
    _ = x  # keep the Python object alive through the native call above
    _gather_block(fit_f32, off + 2 * nb * big_n, big_n, nb, local, Int(py=classical_address(x0)), context)
    _ = x0  # keep the Python object alive through the native call above
    var sa = Int(py=classical_address(stats))
    _ = stats  # keep the Python object alive through the native call above
    var fa = Int(py=classical_address(flags))
    _ = flags  # keep the Python object alive through the native call above
    _gather_block(grouped_ll, trial * nb, 1, nb, local, sa, context)
    _ = stats  # keep the Python object alive through the native call above
    _gather_block(fit_f32, off + 3 * nb * big_n, 1, nb, local, sa + 4 * b, context)
    _ = stats  # keep the Python object alive through the native call above
    _gather_block(fit_i32, 2 * nb * trial, 1, nb, local, fa, context)
    _ = flags  # keep the Python object alive through the native call above
    _gather_block(fit_i32, 2 * nb * trial + nb, 1, nb, local, fa + 4 * b, context)
    _ = flags  # keep the Python object alive through the native call above
    _adopt(model, fields, arr, Bool(py=series[3]), 2, b, n_obs, _no_exog(context), 0, big_n, params, x, x0, stats,
           flags, PythonObject(None), context)
    return model


def arima_predict(model: PythonObject, args: PythonObject, context: PythonObject, forecast: Bool = False) raises -> PythonObject:
    var fields = _fields(model)
    if not py_contains(fields, "params_"):
        raise Error("ValueError: " + _A + "call fit() before " + ("forecast" if forecast else "predict"))
    var n_obs = _integer(fields["n_obs_"], context)
    var start = n_obs if forecast else _integer(args[0], context)
    var end = Int(py=PythonObject(n_obs) + _integer_object(args[0], context)) if forecast else n_obs if args[1] is None else _integer(args[1], context)
    var exog = args[1] if forecast else args[2]
    var order = _order(fields, context)
    var width = Int(py=PythonObject(end) - start)
    if width <= 0:
        raise Error("ValueError: " + _A + ("steps must be >= 1" if forecast else "need start < end; end is excluded"))
    var n_exog = _integer(py_get(fields, "n_exog_", PythonObject(0)), context)
    if n_exog > 0 and end > n_obs and exog is None:
        raise Error("ValueError: " + _A + "the model was fit with a regression component, so future values must be provided via `exog`")
    if n_exog == 0 and exog is not None:
        raise Error("ValueError: " + _A + "a value was given for `exog` but the model was fit without any regression component")
    if end <= n_obs and exog is not None:
        raise Error("ValueError: " + _A + "a value was given for `exog` but only in-sample predictions were requested")
    var batch = _integer(fields["batch_size_"], context)
    var k = _integer(fields["k_"], context)
    if batch <= 0 or n_obs <= 0 or n_exog < 0 or k < 0 or k > 1:
        raise Error("ValueError: ARIMA fitted dimension metadata is invalid")
    var complexity = Int(py=PythonObject(order[0]) + order[2] + order[3] + order[5] + k + n_exog + 1)
    var y = arima_buffer(fields["_y"], "y", context, True)
    var params = arima_buffer(fields["params_"], "params", context, True)
    if y.count != Int(py=PythonObject(batch) * n_obs) or params.count != Int(py=PythonObject(batch) * complexity):
        raise Error("ValueError: ARIMA fitted buffer dimensions disagree with model metadata")
    # Unused exog pointers remain live through the admitted y owner.
    var ex_addr = y.address
    var future_addr = y.address
    var ex_owner = y.copy()
    var future_owner = y.copy()
    var future = PythonObject(None)
    if n_exog > 0:
        ex_owner = arima_buffer(fields["_exog"], "exog", context, True)
        if ex_owner.count != Int(py=PythonObject(batch) * n_obs * n_exog):
            raise Error("ValueError: ARIMA fitted exog dimensions disagree")
        ex_addr = ex_owner.address
        if end > n_obs:
            var e = _exog_array(exog, batch, end - n_obs, _integer(py_get(fields, "_y_ndim", PythonObject(2)), context), "exog", context)
            if Int(py=e[1]) != n_exog:
                raise Error("ValueError: " + _A + "dimensions mismatch, future exog has the wrong regressor count")
            future = e[0]
            future_owner = arima_buffer(future, "prepared future exog", context, True)
            future_addr = future_owner.address
    var result = tsa_array("f", "<f4", Python.tuple(batch, width), Int(py=PythonObject(batch) * width), 4, context)
    var entries = _arima_entries(model, fields, context)
    var plan = Python.list(batch, n_obs, width) if forecast else Python.list(batch, n_obs, start, end)
    for i in range(len(order)):  # driver-loop: seven admitted ARIMA order words
        _ = py_call(plan.append, order[i])
    _ = py_call(plan.extend, Python.tuple(k, n_exog, 0) if forecast else Python.tuple(k, n_exog))
    _ = py_call_typed(_entry(entries, "arima_forecast" if forecast else "arima_predict"), y.address, ex_addr,
                      future_addr, params.address, classical_address(result), plan)
    _ = y
    _ = params
    _ = ex_owner
    _ = future_owner
    _ = future
    return result


def arima_forecast(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return arima_predict(model, args, context, True)
