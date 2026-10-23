# SPDX-License-Identifier: Apache-2.0
"""`ARIMA.fit` and `ARIMA.predict` (python/mojolearn/_arima_impl.py) as one
native call each, through `tsa_native_call` (bindings/tsa_public_native.mojo):

    fit      `_series_major` (y admitted 1-D or 2-D, C-ordered float32
             (batch_size, n_obs)), `_exog_array`'s shapes and refusals, the
             information-criterion penalties (the pinned `log` entry), one
             `arima_fit` on the model's tier, then `_adopt_fit`'s state
    predict  `_check_fitted`, the width refusal, `_future_exog`'s three
             checks in cuML's order and words, one `arima_predict`

The same compiled entries with the same argument words, so no bit moves.
A host inference class (`HostARIMA`) reads its loaded forecast host binding
from the module table.
"""
from std.memory import memcpy
from std.python import Python, PythonObject
from x_metrics.epilogue import portable_log_c
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.classical_arrays import classical_input, classical_address, classical_cast_kind
from bindings.classical_pandas import classical_admit_value
from bindings.tsa_hw_native import tsa_array
from core.xseq_marshal import xseq_cast_nd, xseq_widen_f32_f64

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _A = "mojolearn ARIMA: "


def _b() raises -> PythonObject:
    return Python.import_module("builtins")


def _repr(value: PythonObject) raises -> String:
    return String(py=py_call(_b().repr, value))


def _shape_tuple(value: PythonObject) raises -> PythonObject:
    return py_call(_b().tuple, value.shape)


def _arima_entries(model: PythonObject, fields: PythonObject, context: PythonObject) raises -> PythonObject:
    """`ARIMA._extension()`: the `_mojolearn_arima` entries of the model's
    tier with the binary's own mode read back; a host inference class's
    loaded forecast host binding."""
    var b = _b()
    if Bool(py=py_call(b.getattr, py_type(model), "_HOST_INFERENCE_ONLY", False)):
        var saved = py_get(fields, "numeric_mode")
        if saved is not None and not Bool(py=saved == "identical"):
            raise Error("ValueError: mojolearn: " + String(py=py_type(model).__name__)
                        + " runs IDENTICAL only on the host; this model was saved " + _repr(saved))
        var module = py_get(context["modules"], context["hw_host_module"])
        if module is None:
            raise Error("RuntimeError: mojolearn: the host " + String(py=py_type(model).__name__)
                        + " has no loaded forecast binding; load the model with its own load()")
        var host = b.dict()
        var names = context["arima_entry_names"]
        for i in range(len(names)):  # driver-loop: over the fixed host entry names
            var entry = py_call(b.getattr, module, names[i], PythonObject(None))
            if entry is not None:
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


def _admit(source: PythonObject, name: String, message: String, context: PythonObject) raises -> PythonObject:
    """`probe(v)`, or a nested list materialized as numbers; `message` when
    neither works."""
    try:
        classical_admit_value(source, context)
        return classical_input(source, context, name)
    except:
        raise Error("ValueError: " + message)


def _f32_c(value: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`as_f32_c(value)`: one C-ordered float32 Array copy of an admitted
    array, any rank (the shared element cast)."""
    var rank = Int(py=value.ndim)
    var count = 1
    var dims = List[Int]()
    var steps = List[Int]()
    for k in range(rank):  # driver-loop: over array rank metadata
        dims.append(Int(py=value.shape[k]))
        steps.append(Int(py=value.strides[k]))
        count *= dims[k]
    var item = Int(py=value.itemsize)
    var kind = classical_cast_kind(value, item, name, context)
    var out = tsa_array("f", "<f4", _shape_tuple(value), count, 4, context)
    xseq_cast_nd(Int(py=classical_address(value)), dims, steps, item, kind, True,
                 _F32P(unsafe_from_address=Int(py=classical_address(out))))
    _ = out  # keep the Python object alive through the native call above
    return out


def _copied(value: PythonObject, context: PythonObject) raises -> Bool:
    """Whether `as_f32_c` would have copied: anything but C-ordered float32."""
    var typed = String(py=py_call(_b().str, value.dtype))
    var f32 = typed == "<f4" or typed == "float32"
    var c_order = True
    if Bool(py=py_call(_b().hasattr, value, "flags")):
        var flags = value.flags
        c_order = (Bool(py=flags["C_CONTIGUOUS"]) if py_type(flags) is _b().dict
                   else Bool(py=flags.c_contiguous))
    elif Bool(py=py_call(_b().hasattr, value, "order")):
        c_order = String(py=value.order) == "C" or Int(py=value.ndim) < 2
    return not (f32 and c_order)


def _series_major(source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_series_major(y, "y")`: (array (batch_size, n_obs), batch_size,
    n_obs, copied, the input rank)."""
    var value = _admit(source, "y", _A + "y must be 1-D (one series) or 2-D (batch_size, n_obs) -- an array or a "
                       + "nested list of numbers -- got " + String(py=py_type(source).__name__), context)
    var rank = Int(py=value.ndim)
    if rank != 1 and rank != 2:
        raise Error("ValueError: " + _A + "y must be 1-D (one series) or 2-D (batch_size, n_obs), got " + String(rank)
                    + "-D shape " + _repr(_shape_tuple(value)))
    if Int(py=value.size) == 0:
        raise Error("ValueError: " + _A + "y is empty, shape " + _repr(_shape_tuple(value)))
    var copied = _copied(value, context)
    var arr = _f32_c(value, "y", context)
    var rows = 1 if rank == 1 else Int(py=value.shape[0])
    var cols = Int(py=value.shape[0]) if rank == 1 else Int(py=value.shape[1])
    if rank == 1:
        var flat = arr
        arr = tsa_array("f", "<f4", Python.tuple(1, cols), cols, 4, context)
        memcpy(dest=_F32P(unsafe_from_address=Int(py=classical_address(arr))),
               src=_F32P(unsafe_from_address=Int(py=classical_address(flat))), count=cols)
        _ = arr  # keep the Python object alive through the native call above
        _ = flat  # keep the Python object alive through the native call above
    return Python.tuple(arr, rows, cols, copied, rank)


def _no_exog(context: PythonObject) raises -> PythonObject:
    return tsa_array("f", "<f4", Python.tuple(1), 1, 4, context)


def _exog_array(source: PythonObject, batch: Int, n_rows: Int, y_ndim: Int, name: String,
                context: PythonObject) raises -> PythonObject:
    """`_exog_array`: (C-ordered float32 (batch_size, n_rows, n_exog), n_exog)."""
    var value = _admit(source, name, _A + name + " must be an array or a nested list of numbers, got "
                       + String(py=py_type(source).__name__), context)
    var rank = Int(py=value.ndim)
    var shown = _repr(_shape_tuple(value))
    var d0 = Int(py=value.shape[0]) if rank >= 1 else 0
    var d1 = Int(py=value.shape[1]) if rank >= 2 else 0
    var d2 = Int(py=value.shape[2]) if rank >= 3 else 0
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
    var flat = _f32_c(value, name, context)
    var out = tsa_array("f", "<f4", Python.tuple(s0, s1, s2), s0 * s1 * s2, 4, context)
    memcpy(dest=_F32P(unsafe_from_address=Int(py=classical_address(out))),
           src=_F32P(unsafe_from_address=Int(py=classical_address(flat))), count=s0 * s1 * s2)
    _ = out  # keep the Python object alive through the native call above
    _ = flat  # keep the Python object alive through the native call above
    return Python.tuple(out, s2)


def _order(fields: PythonObject) raises -> List[Int]:
    """[p, d, q, P, D, Q, s]."""
    var o = fields["order"]
    var so = fields["seasonal_order"]
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
    var fields = py_fields(model)
    var series = _series_major(args[0], context)
    var arr = series[0]
    var batch = Int(py=series[1])
    var n_obs = Int(py=series[2])
    var y_ndim = Int(py=series[4])
    var order = _order(fields)
    var ex = PythonObject(None)
    var n_exog = 0
    if args[1] is None:
        ex = _no_exog(context)
    else:
        var e = _exog_array(args[1], batch, n_obs, y_ndim, "exog", context)
        ex = e[0]
        n_exog = Int(py=e[1])
    var k = Int(py=fields["k_"])
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
                           n_exog, context["arima_methods"][fields["method"]], fields["maxiter"],
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
        var pens = _penalties(_order(fields), big_n, n_obs, context)
        var entries = _arima_entries(model, fields, context)
        var written = Int(py=py_call_typed(_entry(entries, "arima_ic_from_loglike"), classical_address(stats),
                                           classical_address(ics), Python.list(batch, pens[0], pens[1])))
        _ = ics  # keep the Python object alive through the native call above
        if written != batch:
            raise Error("RuntimeError: " + _A + "the criterion kernel wrote " + String(written)
                        + " series, batch_size is " + String(batch))
            _ = ics  # keep the Python object alive through the native call above
        _ = ics  # keep the Python object alive through the native calls above
    py_setitem(fields, "input_copied_", copied)
    py_setitem(fields, "batch_size_", batch)
    py_setitem(fields, "n_obs_", n_obs)
    py_setitem(fields, "_y", arr)
    py_setitem(fields, "_y_ndim", y_ndim)
    py_setitem(fields, "n_exog_", n_exog)
    py_setitem(fields, "complexity_", big_n)
    py_setitem(fields, "_exog", ex if n_exog else PythonObject(None))
    py_setitem(fields, "params_", params)
    py_setitem(fields, "x_", x)
    py_setitem(fields, "x0_", x0)
    py_setitem(fields, "n_iter_", _copy_block(flags, 0, batch, "i", "<i4", 4, Python.tuple(batch), context))
    py_setitem(fields, "retcode_", _copy_block(flags, batch, batch, "i", "<i4", 4, Python.tuple(batch), context))
    # llf_ = stats[:batch].astype("<f8"): each float32 widened exactly
    var llf = tsa_array("d", "<f8", Python.tuple(batch), batch, 8, context)
    xseq_widen_f32_f64(_F32P(unsafe_from_address=Int(py=classical_address(stats))),
                       MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(llf))), batch)
    _ = llf  # keep the Python object alive through the native call above
    py_setitem(fields, "llf_", llf)
    py_setitem(fields, "fx_", _copy_block(stats, batch, batch, "f", "<f4", 4, Python.tuple(batch), context))
    py_setitem(fields, "aic_", _copy_block(ics, 0, batch, "d", "<f8", 8, Python.tuple(batch), context))
    py_setitem(fields, "bic_", _copy_block(ics, batch, batch, "d", "<f8", 8, Python.tuple(batch), context))


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
    var fields = py_fields(model)
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


def arima_predict(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    if not py_contains(fields, "params_"):
        raise Error("ValueError: " + _A + "call fit() before predict")
    var b = _b()
    var n_obs = Int(py=fields["n_obs_"])
    var start = Int(py=py_call(b.int, args[0]))
    var end = n_obs if args[1] is None else Int(py=py_call(b.int, args[1]))
    var exog = args[2]
    var order = _order(fields)
    var width = end - start
    if width <= 0:
        raise Error("ValueError: " + _A + "need start < end, got start=" + String(start) + ", end=" + String(end)
                    + ". `end` is EXCLUDED here, as it is in cuML; statsmodels' end is the last index returned")
    # `_future_exog`
    var n_exog = Int(py=py_get(fields, "n_exog_", PythonObject(0)))
    if n_exog > 0 and end > n_obs and exog is None:
        raise Error("ValueError: " + _A + "the model was fit with a regression component, so future values must be "
                    + "provided via `exog`")
    if n_exog == 0 and exog is not None:
        raise Error("ValueError: " + _A + "a value was given for `exog` but the model was fit without any regression "
                    + "component")
    if end <= n_obs and exog is not None:
        raise Error("ValueError: " + _A + "a value was given for `exog` but only in-sample predictions were requested")
    var batch = Int(py=fields["batch_size_"])
    var ex = PythonObject(None)
    var fut = PythonObject(None)
    if n_exog == 0:
        ex = _no_exog(context)
        fut = _no_exog(context)
    elif end <= n_obs:
        ex = fields["_exog"]
        fut = _no_exog(context)
    else:
        var e = _exog_array(exog, batch, end - n_obs, Int(py=py_get(fields, "_y_ndim", PythonObject(2))), "exog", context)
        if Int(py=e[1]) != n_exog:
            raise Error("ValueError: " + _A + "dimensions mismatch, `exog` has " + String(Int(py=e[1]))
                        + " regressor column(s) and the model was fit with " + String(n_exog))
        ex = fields["_exog"]
        fut = e[0]
    var out = tsa_array("f", "<f4", Python.tuple(batch, width), batch * width, 4, context)
    var entries = _arima_entries(model, fields, context)
    _ = py_call_typed(_entry(entries, "arima_predict"), classical_address(fields["_y"]), classical_address(ex),
                      classical_address(fut), classical_address(fields["params_"]), classical_address(out),
                      Python.list(batch, n_obs, start, end, order[0], order[1], order[2], order[3], order[4], order[5],
                                  order[6], Int(py=fields["k_"]), n_exog))
    _ = ex  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = fut  # keep the Python object alive through the native call above
    return out
