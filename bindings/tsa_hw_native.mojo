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

The same compiled entries with the same argument words, so no bit moves.
A host inference class (`HostExponentialSmoothing`) reads its loaded
forecast host binding from the module table.
"""
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.classical_arrays import classical_input, classical_address, classical_cast_kind, classical_wrap_array
from bindings.classical_pandas import classical_admit_value
from core.xseq_marshal import xseq_cast_nd, xseq_column_copy_f32

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _HW = "mojolearn ExponentialSmoothing: "


def _b() raises -> PythonObject:
    return Python.import_module("builtins")


def _repr(value: PythonObject) raises -> String:
    return String(py=py_call(_b().repr, value))


def _type_text(value: PythonObject) raises -> String:
    """`str(type(value))`: "<class 'float'>"."""
    return String(py=py_call(_b().str, py_type(value)))


def tsa_array(code: String, typestr: String, shape: PythonObject, count: Int, width: Int,
           context: PythonObject) raises -> PythonObject:
    """A zero-filled C-ordered mojolearn Array of `typestr` and `shape`."""
    var store = py_call(context["array_type"], PythonObject(code), py_call(context["native_bytes_type"], width * count))
    return classical_wrap_array(store, shape, typestr, "C", context)


def _is_int(value: PythonObject) raises -> Bool:
    """`isinstance(value, int) and not isinstance(value, bool)`."""
    var b = _b()
    return Bool(py=py_call(b.isinstance, value, b.int)) and not Bool(py=py_call(b.isinstance, value, b.bool))


def _hw_entries(model: PythonObject, fields: PythonObject, context: PythonObject) raises -> PythonObject:
    """`self._bind()`: the `_mojolearn_tsa` entries of the model's tier (the
    process default until a loaded model restores its own); a host inference
    class's loaded forecast host binding."""
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
        var names = context["hw_entry_names"]
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
    mode = py_call(py_call(mode.strip).lower)
    if String(py=mode) != "identical" and String(py=mode) != "fast":
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    var entries = context["hw_bindings"][mode]
    if py_type(entries) is b.str:
        raise Error(String(py=entries))
    return entries


def _entry(entries: PythonObject, name: String) raises -> PythonObject:
    if not py_contains(entries, name):
        raise Error("AttributeError: the forecast binding has no " + name)
    return entries[name]


# ---------------------------------------------------------------- fit

def _check_dims(fields: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_check_dims(ts_input)`: (C-ordered float32 (ts_num, n) data, n)."""
    var b = _b()
    var value = PythonObject(None)
    try:
        classical_admit_value(source, context)
        value = classical_input(source, context, "endog")
    except:
        raise Error("ValueError: " + _HW + "data input must be a 1-D or 2-D array or a nested list of numbers, got "
                    + String(py=py_type(source).__name__))
    var rank = Int(py=value.ndim)
    var ts_num = Int(py=fields["ts_num"])
    var err = String(_HW + "initialized with ") + String(ts_num) + " time series, but data has dimension "
    var rows: Int
    var n: Int
    if rank == 1:
        n = Int(py=value.shape[0])
        if ts_num != 1:
            raise Error("ValueError: " + err + "1.")
        rows = 1
    elif rank == 2:
        n = Int(py=value.shape[1])
        rows = Int(py=value.shape[0])
        if ts_num != rows:
            raise Error("ValueError: " + err + String(rows) + ".")
    else:
        raise Error("ValueError: " + _HW + "data input must have 1 or 2 dimensions, got " + String(rank))
    if rows * n == 0:
        raise Error("ValueError: mojolearn: endog is empty, shape " + String(py=py_call(b.str, py_call(b.tuple, value.shape))))
    var item = Int(py=value.itemsize)
    var kind = classical_cast_kind(value, item, "endog", context)
    var out = tsa_array("f", "<f4", Python.tuple(rows, n), rows * n, 4, context)
    var dims = List[Int]()
    var steps = List[Int]()
    if rank == 2:
        dims.append(rows)
        steps.append(Int(py=value.strides[0]))
        dims.append(n)
        steps.append(Int(py=value.strides[1]))
    else:
        dims.append(n)
        steps.append(Int(py=value.strides[0]))
    xseq_cast_nd(Int(py=classical_address(value)), dims, steps, item, kind, True,
                 _F32P(unsafe_from_address=Int(py=classical_address(out))))
    _ = value^
    return Python.tuple(out, n)


def _block(source: PythonObject, offset: Int, count: Int, code: String, typestr: String,
           shape: PythonObject, context: PythonObject) raises -> PythonObject:
    """A copy of `count` 4-byte words of `source` from word `offset`, as an
    Array of `shape` (the Python body's slice, and its reshape)."""
    var out = tsa_array(code, typestr, shape, count, 4, context)
    if count > 0:
        memcpy(dest=_F32P(unsafe_from_address=Int(py=classical_address(out))),
               src=_F32P(unsafe_from_address=Int(py=classical_address(source)) + 4 * offset), count=count)
    return out


def _set_fitted(fields: PythonObject, n: Int, comps: PythonObject, stats: PythonObject, flags: PythonObject,
                context: PythonObject) raises:
    """`_set_fitted`: the packed components, the three time-major blocks and
    the per-series statistics; the lazily built series-major components are
    dropped."""
    var period = Int(py=fields["seasonal_periods"])
    var b = Int(py=fields["ts_num"])
    var cl = (n - period) * b
    var rows = cl // b
    py_setitem(fields, "n", n)
    py_setitem(fields, "_components_len", cl)
    py_setitem(fields, "_comps", comps)
    for name in [String("level_"), String("trend_"), String("season_")]:  # small-loop(the three lazy component names)
        _ = py_call(fields.pop, name, PythonObject(None))
    py_setitem(fields, "_comp_layout", Python.tuple(cl, b, rows))
    var time_major = _b().dict()
    var names = [String("level"), String("trend"), String("season")]
    for i in range(3):  # small-loop(the three packed components)
        py_setitem(time_major, names[i], _block(comps, i * cl, cl, "f", "<f4", Python.tuple(rows, b), context))
    py_setitem(fields, "_time_major", time_major)
    var stat_names = [String("sse_"), String("alpha_"), String("beta_"), String("gamma_")]
    for i in range(4):  # small-loop(the four per-series statistics)
        py_setitem(fields, stat_names[i], _block(stats, i * b, b, "f", "<f4", Python.tuple(b), context))
    py_setitem(fields, "n_iter_", _block(flags, 0, b, "i", "<i4", Python.tuple(b), context))
    py_setitem(fields, "criterion_", _block(flags, b, b, "i", "<i4", Python.tuple(b), context))
    py_setitem(fields, "fit_executed_flag", True)


def hw_fit(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var checked = _check_dims(fields, fields["endog"], context)
    var data = checked[0]
    var n = Int(py=checked[1])
    py_setitem(fields, "_data", data)
    var ts_num = Int(py=fields["ts_num"])
    var period = Int(py=fields["seasonal_periods"])
    var cl = (n - period) * ts_num
    if cl <= 0:
        raise Error("ValueError: " + _HW + "n (" + String(n) + ") must exceed seasonal_periods (" + String(period) + ")")
    var comps = tsa_array("f", "<f4", Python.tuple(3 * cl), 3 * cl, 4, context)
    var stats = tsa_array("f", "<f4", Python.tuple(4 * ts_num), 4 * ts_num, 4, context)
    var flags = tsa_array("i", "<i4", Python.tuple(2 * ts_num), 2 * ts_num, 4, context)
    var entries = _hw_entries(model, fields, context)
    var init = context["hw_init_codes"][fields["initialization_method"]]
    _ = py_call_typed(_entry(entries, "holtwinters_fit"), classical_address(data), classical_address(comps),
                      classical_address(stats), classical_address(flags),
                      Python.list(n, ts_num, period, Int(py=fields["start_periods"]),
                                  PythonObject(Float64(py=py_call(_b().float, fields["eps"]))), init),
                      fields["seasonal"])
    _set_fitted(fields, n, comps, stats, flags, context)
    return model


# ---------------------------------------------------------------- predict

def _shaped(fields: PythonObject, out_buf: PythonObject, offset: Int, steps: Int, index: PythonObject,
            context: PythonObject) raises -> PythonObject:
    """`_shaped(out[offset:], steps, index)` over a time-major buffer."""
    var b = Int(py=fields["ts_num"])
    var base = Int(py=classical_address(out_buf)) + 4 * offset
    if index is not None:
        var s = Int(py=index)
        var one = tsa_array("f", "<f4", Python.tuple(steps), steps, 4, context)
        # one series' strided words (a copy, no arithmetic)
        xseq_column_copy_f32(_F32P(unsafe_from_address=base), _F32P(unsafe_from_address=Int(py=classical_address(one))),
                             steps, b, s)
        return one
    if b == 1:
        return _block(out_buf, offset, steps, "f", "<f4", Python.tuple(steps), context) if offset != 0 else out_buf
    return _block(out_buf, offset, steps * b, "f", "<f4", Python.tuple(steps, b), context)


def _check_index(index: PythonObject, b: Int) raises:
    if index is not None and not _is_int(index):
        raise Error("TypeError: " + _HW + "index must be int or None, got " + _type_text(index))


def hw_predict(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    if not Bool(py=fields["fit_executed_flag"]):
        raise Error("ValueError: " + _HW + "fit() the model before predict()")
    var n = Int(py=fields["n"])
    var b = Int(py=fields["ts_num"])
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
    var entries = _hw_entries(model, fields, context)
    var period = Int(py=fields["seasonal_periods"])
    var steps = end - start
    var comps = fields["_comps"]
    if start >= n:
        var fc = tsa_array("f", "<f4", Python.tuple((end - n) * b), (end - n) * b, 4, context)
        _ = py_call_typed(_entry(entries, "holtwinters_forecast"), classical_address(comps), classical_address(fc),
                          Python.list(n, b, period, end - n), fields["seasonal"])
        _ = comps  # keep the Python object alive through the native call above
        return _shaped(fields, fc, (start - n) * b, steps, index, context)
    var out = tsa_array("f", "<f4", Python.tuple(steps * b), steps * b, 4, context)
    var k_in = min(end, n) - start
    var base = Int(py=classical_address(out))
    _ = py_call_typed(_entry(entries, "holtwinters_predict"), classical_address(comps), PythonObject(base),
                      Python.list(n, b, period, start, start + k_in), fields["seasonal"])
    _ = comps  # keep the Python object alive through the native call above
    if end > n:
        _ = py_call_typed(_entry(entries, "holtwinters_forecast"), classical_address(comps),
                          PythonObject(base + 4 * k_in * b), Python.list(n, b, period, end - n), fields["seasonal"])
        _ = comps  # keep the Python object alive through the native call above
    return _shaped(fields, out, 0, steps, index, context)


def hw_score(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    if not Bool(py=fields["fit_executed_flag"]):
        raise Error("ValueError: " + _HW + "fit() the model before score()")
    var index = args[0]
    if index is None:
        return fields["sse_"]
    var b = Int(py=fields["ts_num"])
    if Bool(py=index < PythonObject(0)) or Bool(py=index >= PythonObject(b)):
        raise Error("IndexError: " + _HW + "index input: " + String(py=py_call(_b().str, index))
                    + " outside of range [0, " + String(b) + ")")
    # `self.sse_[index]`: the float32 word, as a Python float
    var sse = fields["sse_"]
    var value = _F32P(unsafe_from_address=Int(py=classical_address(sse)))[Int(py=index)]
    _ = sse^
    return PythonObject(Float64(value))
