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
from bindings.native_loaded_module import native_loaded_module
from bindings.native_array_storage import native_array_zeros
from bindings.arena_materialize_native import _owned
from bindings.native_numpy_metadata import native_numpy_module_fields
from bindings.classical_archive import classical_archive_write
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.classical_arrays import classical_input, classical_address, classical_cast_kind, classical_wrap_array
from bindings.classical_pandas import classical_admit_value
from providers.python.array_metadata_pyapi import ArrayMetadata
from bindings.tsa_admission import tsa_hw_fields, tsa_hw_class, tsa_buffer, tsa_scalar, tsa_component_index
from bindings.native_owner_lifetime import native_release_admit
from bindings.classical_attributes import classical_connect_numpy
from providers.python.forecast_owner_pyapi import forecast_commit
from core.xseq_marshal import xseq_column_copy_f32
from core.admitted_input_nd import admitted_cast_nd_f32
from bindings.native_loaded_module import NativeModules
from bindings.native_prepared_owner import prepared_get, prepared_put
from bindings.native_resident_capsule import resident_capsule
from bindings.p3nc_fitted_freeze import fitted_is_freezable_any, fitted_freeze_any

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _HW = "mojolearn ExponentialSmoothing: "
# The resident fitted components of a fitted model (lane ne1-a-decomp-tsa,
# plan section A, row N030): one prepared-owner record per model, outside
# its __dict__ and dropped with it, `(_comps, prepare entry, n, ts_num,
# seasonal_periods, handle, capsule)`. A refit stores a new `_comps`, so the
# next query misses, mints again and the replaced record's capsule returns
# the old device copy.
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


def _is_int(value: PythonObject) raises -> Bool:
    """`isinstance(value, int) and not isinstance(value, bool)`."""
    var b = _b()
    return py_type(value) is b.int


def _hw_entries(model: PythonObject, fields: PythonObject, context: PythonObject) raises -> PythonObject:
    """`self._bind()`: the `_mojolearn_tsa` entries of the model's tier (the
    process default until a loaded model restores its own); a host inference
    class's loaded forecast host binding."""
    var b = _b()
    if tsa_hw_class(model, context):
        var saved = py_get(fields, "numeric_mode")
        if saved is not None and (py_type(saved) is not b.str or String(saved) != "identical"):
            raise Error("ValueError: mojolearn: " + String("ExponentialSmoothing")
                        + " runs IDENTICAL only on the host")
        var module = py_get(context["modules"], context["hw_host_module"])
        if module is None:
            raise Error("RuntimeError: mojolearn: the host " + String("ExponentialSmoothing")
                        + " has no loaded forecast binding; load the model with its own load()")
        if py_type(module) is not py_type(native_loaded_module("sys")):
            raise Error("TypeError: host forecast requires its native module")
        var module_fields = native_numpy_module_fields(module)
        var host = b.dict()
        var names = context["hw_entry_names"]
        for i in range(len(names)):  # driver-loop: over the fixed host entry names
            var entry = py_get(module_fields, names[i])
            if entry is not None:
                if py_type(entry) is not py_type(b.len):
                    raise Error("TypeError: host forecast requires compiled entries")
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
    var value = tsa_buffer(source, "endog", context)
    var rank = len(value.shape)
    var ts_num = Int(py=fields["ts_num"])
    if rank != 1 and rank != 2:
        raise Error("ValueError: " + _HW + "data input must have 1 or 2 dimensions")
    var rows = 1 if rank == 1 else value.shape[0]
    var n = value.shape[0] if rank == 1 else value.shape[1]
    if ts_num != rows:
        raise Error("ValueError: " + _HW + "initialized series count disagrees with data")
    if rows * n == 0:
        raise Error("ValueError: " + _HW + "endog is empty")
    var out = tsa_array("f", "<f4", Python.tuple(rows, n), rows * n, 4, context)
    _ = admitted_cast_nd_f32(value.source.addr, value.shape, value.strides, value.source.itemsize,
                 value.source.kind, _F32P(unsafe_from_address=Int(py=classical_address(out))))
    _ = value
    _ = out
    return Python.tuple(out, n)


def _block(source: PythonObject, offset: Int, count: Int, code: String, typestr: String,
           shape: PythonObject, context: PythonObject) raises -> PythonObject:
    """A copy of `count` 4-byte words of `source` from word `offset`, as an
    Array of `shape` (the Python body's slice, and its reshape)."""
    var out = tsa_array(code, typestr, shape, count, 4, context)
    if count > 0:
        memcpy(dest=_F32P(unsafe_from_address=Int(py=classical_address(out))),
               src=_F32P(unsafe_from_address=Int(py=classical_address(source)) + 4 * offset), count=count)
        _ = out  # keep the Python object alive through the native call above
    return out


def _set_fitted(fields: PythonObject, n: Int, comps: PythonObject, stats: PythonObject, flags: PythonObject,
                context: PythonObject) raises -> PythonObject:
    """`_set_fitted`: the packed components, the three time-major blocks and
    the per-series statistics; the lazily built series-major components are
    dropped."""
    var period = Int(py=fields["seasonal_periods"])
    var b = Int(py=fields["ts_num"])
    var cl = (n - period) * b
    var rows = cl // b
    var updates = Python.dict()
    py_setitem(updates, "n", n)
    py_setitem(updates, "_components_len", cl)
    py_setitem(updates, "_comps", comps)
    py_setitem(updates, "_comp_layout", Python.tuple(cl, b, rows))
    var time_major = _b().dict()
    var names = [String("level"), String("trend"), String("season")]
    for i in range(3):  # small-loop(the three packed components)
        py_setitem(time_major, names[i], _block(comps, i * cl, cl, "f", "<f4", Python.tuple(rows, b), context))
    py_setitem(updates, "_time_major", time_major)
    var stat_names = [String("sse_"), String("alpha_"), String("beta_"), String("gamma_")]
    for i in range(4):  # small-loop(the four per-series statistics)
        py_setitem(updates, stat_names[i], _block(stats, i * b, b, "f", "<f4", Python.tuple(b), context))
    py_setitem(updates, "n_iter_", _block(flags, 0, b, "i", "<i4", Python.tuple(b), context))
    py_setitem(updates, "criterion_", _block(flags, b, b, "i", "<i4", Python.tuple(b), context))
    py_setitem(updates, "fit_executed_flag", True)

    return updates


def hw_fit(model: PythonObject, context: PythonObject) raises -> PythonObject:
    # One admission of the class and instance keys serves the snapshot and
    # the commit below (nothing between them touches the model's storage).
    var live = tsa_hw_fields(model, context)
    var fields = _snapshot_of(live, context)
    var checked = _check_dims(fields, fields["endog"], context)
    var data = checked[0]
    var n = Int(py=checked[1])
    var ts_num = Int(py=fields["ts_num"])
    var period = Int(py=fields["seasonal_periods"])
    if ts_num < 1 or n <= period or n - period > 9223372036854775807 // 12 // ts_num:
        raise Error("ValueError: Holt-Winters fitted extent is invalid")
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
                                  fields["eps"], init),
                      fields["seasonal"])
    _ = comps  # keep the Python object alive through the native call above
    _ = data  # keep the Python object alive through the native call above
    _ = flags  # keep the Python object alive through the native call above
    _ = stats  # keep the Python object alive through the native call above
    var updates = _set_fitted(fields, n, comps, stats, flags, context)
    py_setitem(updates, "_data", data)
    _commit(live, updates, context, True)
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
    var mods = NativeModules()
    var held = prepared_get(mods, _HW_RESIDENT_SLOT, model)
    if held is None or held[0] is not py_get(fields, "_comps"):
        return PythonObject(None)
    return held


def _hw_matches(held: PythonObject, entries: PythonObject, n: Int, b: Int, period: Int) raises -> Bool:
    """The held record was minted by this binding with these fitted words."""
    if held is None or py_get(entries, "holtwinters_query_resident") is None:
        return False
    return (held[1] is py_get(entries, "holtwinters_resident_prepare") and Int(py=held[2]) == n
            and Int(py=held[3]) == b and Int(py=held[4]) == period)


def _hw_mint(model: PythonObject, fields: PythonObject, entries: PythonObject, n: Int, b: Int, period: Int,
             comps_address: Int) raises -> PythonObject:
    """Upload the admitted components (`comps_address`, the `_packed`
    admission of `_comps`) once and record them; returns the record (word 5
    the handle; the caller keeps it alive through its query, since a model
    that cannot be weakly referenced keeps no record and the capsule would
    release the copy at once). None (the per-call path) on a binding without
    the resident entries (the host column) or components that cannot be
    frozen (a caller-assigned ndarray: plan section A's read-only rule)."""
    var prepare = py_get(entries, "holtwinters_resident_prepare")
    var release = py_get(entries, "holtwinters_resident_release")
    if prepare is None or release is None or py_get(entries, "holtwinters_query_resident") is None:
        return PythonObject(None)
    var comps_obj = py_get(fields, "_comps")
    if not fitted_is_freezable_any(comps_obj):
        return PythonObject(None)
    _ = fitted_freeze_any(comps_obj)
    var handle = Int(py=py_call_typed(prepare, PythonObject(comps_address), Python.list(n, b, period)))
    # From here the capsule owns the device copy.
    var capsule = resident_capsule(release, handle)
    var record = Python.tuple(comps_obj, prepare, n, b, period, handle, capsule)
    var mods = NativeModules()
    prepared_put(mods, _HW_RESIDENT_SLOT, model, record)
    return record


def _hw_query(entries: PythonObject, record: PythonObject, fields: PythonObject, start: Int, end: Int, b: Int,
              context: PythonObject) raises -> PythonObject:
    """One `holtwinters_query_resident` call: `(end - start) * b` float32,
    time-major, the in-sample steps then the forecast steps."""
    var steps = end - start
    var whole = tsa_array("f", "<f4", Python.tuple(steps * b), steps * b, 4, context)
    _ = py_call_typed(_entry(entries, "holtwinters_query_resident"), record[5], classical_address(whole),
                      Python.list(start, end), fields["seasonal"])
    _ = record  # the record (and its capsule) outlive the query
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
    var fields = _query_fields(model, context)
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
    if steps > 9223372036854775807 // 4 // b or (end > n and end - n > 9223372036854775807 // 4 // b):
        raise Error("ValueError: Holt-Winters prediction extent exceeds native indexing bounds")
    # One call over the device copy (the in-sample steps, then the forecast
    # steps, time-major: the words of the two per-call entries) when the
    # components are resident or can be made so.
    var held = _hw_held(model, fields)
    if _hw_matches(held, entries, n, b, period):
        return _shaped(fields, _hw_query(entries, held, fields, start, end, b, context), 0, steps, index, context)
    var packed = _packed(fields["_comps"], 3 * (n - period) * b, context)
    var comps = packed.keep
    var minted = _hw_mint(model, fields, entries, n, b, period, packed.source.addr)
    if minted is not None:
        var whole = _hw_query(entries, minted, fields, start, end, b, context)
        _ = comps
        return _shaped(fields, whole, 0, steps, index, context)
    if start >= n:
        var fc = tsa_array("f", "<f4", Python.tuple((end - n) * b), (end - n) * b, 4, context)
        _ = py_call_typed(_entry(entries, "holtwinters_forecast"), classical_address(comps), classical_address(fc),
                          Python.list(n, b, period, end - n), fields["seasonal"])
        _ = fc  # keep the Python object alive through the native call above
        _ = comps  # keep the Python object alive through the native call above
        return _shaped(fields, fc, (start - n) * b, steps, index, context)
    var out = tsa_array("f", "<f4", Python.tuple(steps * b), steps * b, 4, context)
    var k_in = min(end, n) - start
    var base = Int(py=classical_address(out))
    _ = out  # keep the Python object alive through the native call above
    _ = py_call_typed(_entry(entries, "holtwinters_predict"), classical_address(comps), PythonObject(base),
                      Python.list(n, b, period, start, start + k_in), fields["seasonal"])
    _ = out  # keep the Python object alive through the native call above
    _ = comps  # keep the Python object alive through the native call above
    if end > n:
        _ = py_call_typed(_entry(entries, "holtwinters_forecast"), classical_address(comps),
                          PythonObject(base + 4 * k_in * b), Python.list(n, b, period, end - n), fields["seasonal"])
        _ = comps  # keep the Python object alive through the native call above
        _ = out  # keep the Python object alive through the native call above
    return _shaped(fields, out, 0, steps, index, context)


def hw_score(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _query_fields(model, context)
    if not Bool(py=fields["fit_executed_flag"]):
        raise Error("ValueError: " + _HW + "fit() the model before score()")
    var index = tsa_component_index(args[0], context)
    if index is None:
        return fields["sse_"]
    var b = Int(py=fields["ts_num"])
    _check_index(index, b)
    if Int(py=index) < 0 or Int(py=index) >= b:
        raise Error("IndexError: " + _HW + "index input: " + String(Int(py=index))
                    + " outside of range [0, " + String(b) + ")")
    # `self.sse_[index]`: the float32 word, as a Python float
    var packed = _packed(fields["sse_"], b, context)
    var sse = packed.keep
    var value = _F32P(unsafe_from_address=packed.source.addr)[Int(py=index)]
    _ = sse  # keep the Python object alive through the native call above
    _ = sse^
    return PythonObject(Float64(value))


def _settings(fields: PythonObject, context: PythonObject, write_eps: Bool = True) raises:
    var b = _b()
    for name in ["ts_num", "seasonal_periods", "start_periods"]:  # small-loop(three integer constructor settings)
        if not _is_int(fields[name]):
            raise Error("TypeError: " + _HW + name + " must be int")
    var seasonal = fields["seasonal"]
    if py_type(seasonal) is not b.str:
        raise Error("TypeError: " + _HW + "seasonal must be a native string")
    if String(seasonal) != "additive" and String(seasonal) != "add" and String(seasonal) != "multiplicative" and String(seasonal) != "mul":
        raise Error("ValueError: " + _HW + "seasonal must be additive/add or multiplicative/mul")
    var method = fields["initialization_method"]
    if py_type(method) is not b.str or (String(method) != "estimated" and String(method) != "heuristic" and String(method) != "cuml"):
        raise Error("ValueError: " + _HW + "initialization_method must be estimated, heuristic or cuml")
    var eps = py_call(b.float, tsa_scalar(fields["eps"], context))
    if write_eps:
        py_setitem(fields, "eps", eps)


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
    if py_type(fields["fit_executed_flag"]) is not _b().bool:
        raise Error("TypeError: " + _HW + "fitted flag must be a native boolean")
    if Bool(py=fields["fit_executed_flag"]):
        if not _is_int(fields["n"]) or Int(py=fields["ts_num"]) < 1 or Int(py=fields["seasonal_periods"]) < 2 or Int(py=fields["n"]) <= Int(py=fields["seasonal_periods"]):
            raise Error("ValueError: " + _HW + "invalid fitted dimensions")
        var batch = Int(py=fields["ts_num"])
        if Int(py=fields["n"]) - Int(py=fields["seasonal_periods"]) > 9223372036854775807 // 12 // batch:
            raise Error("ValueError: Holt-Winters fitted extent exceeds native indexing bounds")


def _commit(fields: PythonObject, updates: PythonObject, context: PythonObject, clear_components: Bool = False) raises:
    var old = Python.list()
    var ownership = context["tsa_owner_context"]
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


def hw_initialize(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = tsa_hw_fields(model, context)
    var updates = Python.dict(endog=args[0], seasonal=args[1], seasonal_periods=args[2],
        start_periods=args[3], ts_num=args[4], eps=args[5], initialization_method=args[6], fit_executed_flag=False)
    _settings(updates, context)
    # Constructor retains endog without numerical preparation, matching its
    # deferred fit contract. Its replacement is admitted before publication.
    _commit(fields, updates, context)
    return PythonObject(None)


def _packed(value: PythonObject, count: Int, context: PythonObject) raises -> ArrayMetadata:
    var result = tsa_buffer(value, "Holt-Winters fitted storage", context)
    if not result.f32_c() or result.count != count:
        raise Error("ValueError: " + _HW + "fitted storage must be contiguous float32 with the fitted extent")
    return result^


def hw_forecast(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _query_fields(model, context)
    if not Bool(py=fields["fit_executed_flag"]):
        raise Error("ValueError: " + _HW + "fit() the model before forecast()")
    if not _is_int(args[0]):
        raise Error("TypeError: " + _HW + "h must be int")
    var h = Int(py=args[0])
    var b = Int(py=fields["ts_num"])
    var index = args[1]
    _check_index(index, b)
    if h <= 0:
        raise Error("ValueError: " + _HW + "h must be > 0")
    if index is not None and (Int(py=index) < 0 or Int(py=index) >= b):
        raise Error("IndexError: " + _HW + "index outside series range")
    var n = Int(py=fields["n"])
    var period = Int(py=fields["seasonal_periods"])
    var held = _hw_held(model, fields)
    if held is not None:
        # A held record names this very frozen `_comps`, so the `_packed`
        # admission below would pass: resolving the binding first cannot
        # change which refusal a call meets.
        if h > 9223372036854775807 // 4 // b:
            raise Error("ValueError: Holt-Winters forecast extent exceeds native indexing bounds")
        var hit_entries = _hw_entries(model, fields, context)
        if _hw_matches(held, hit_entries, n, b, period):
            return _shaped(fields, _hw_query(hit_entries, held, fields, n, n + h, b, context), 0, h, index, context)
    var comps = _packed(fields["_comps"], 3 * (n - period) * b, context)
    if h > 9223372036854775807 // 4 // b:
        raise Error("ValueError: Holt-Winters forecast extent exceeds native indexing bounds")
    var out = tsa_array("f", "<f4", Python.tuple(h * b), h * b, 4, context)
    var entries = _hw_entries(model, fields, context)
    var minted = _hw_mint(model, fields, entries, n, b, period, comps.source.addr)
    if minted is not None:
        var whole = _hw_query(entries, minted, fields, n, n + h, b, context)
        _ = comps
        return _shaped(fields, whole, 0, h, index, context)
    _ = py_call_typed(_entry(entries, "holtwinters_forecast"),
        PythonObject(comps.source.addr), classical_address(out), Python.list(n, b, period, h), fields["seasonal"])
    _ = comps
    _ = out
    return _shaped(fields, out, 0, h, index, context)


def hw_component(model: PythonObject, args: PythonObject, context: PythonObject, attribute: Bool = False) raises -> PythonObject:
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
    if not Bool(py=fields["fit_executed_flag"]):
        if attribute:
            raise Error("AttributeError: ExponentialSmoothing has no fitted component")
        raise Error("ValueError: " + _HW + "fit() the model to get components")
    var b = Int(py=fields["ts_num"])
    var rows = Int(py=fields["n"]) - Int(py=fields["seasonal_periods"])
    var index = PythonObject(None) if attribute else tsa_component_index(args[1], context)
    _check_index(index, b)
    if index is not None and (Int(py=index) < 0 or Int(py=index) >= b):
        raise Error("IndexError: " + _HW + "index outside series range")
    var cached = py_get(fields, name)
    if not attribute and cached is not None and (index is not None or b == 1):
        var stored = _packed(cached, b * rows, context)
        return _block(stored.keep, 0 if index is None else Int(py=index) * rows,
                      rows, "f", "<f4", Python.tuple(rows), context)
    var comps = _packed(fields["_comps"], 3 * rows * b, context)
    if not attribute and index is not None:
        return _shaped(fields, comps.keep, which * rows * b, rows, index, context)
    var out = tsa_array("f", "<f4", Python.tuple(b, rows), b * rows, 4, context)
    var base = Int(py=classical_address(out))
    for series in range(b):  # driver-loop: gather each native component column into its series-major output
        xseq_column_copy_f32(_F32P(unsafe_from_address=comps.source.addr + which * rows * b * 4),
            _F32P(unsafe_from_address=base + series * rows * 4), rows, b, series)
    if cached is None:
        var updates = Python.dict()
        py_setitem(updates, name, out)
        _commit(live, updates, context)
    _ = comps
    if attribute:
        return out
    if b == 1:
        return _block(out, 0, rows, "f", "<f4", Python.tuple(rows), context)
    var time_major = fields["_time_major"]
    if py_type(time_major) is not _b().dict:
        raise Error("TypeError: Holt-Winters components require native dictionary storage")
    var keys = py_call(_b().list, time_major)
    for i in range(len(keys)):  # driver-loop: admit keys before retrieving the named native component
        if py_type(keys[i]) is not _b().str:
            raise Error("TypeError: Holt-Winters component keys must be native strings")
    var names = Python.tuple("level", "trend", "season")
    var stored = _packed(time_major[names[which]], rows * b, context)
    return stored.keep


def hw_save(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _snapshot(model, context)
    if not Bool(py=fields["fit_executed_flag"]):
        raise Error("ValueError: " + _HW + "fit() the model before save()")
    var n = Int(py=fields["n"])
    var b = Int(py=fields["ts_num"])
    var period = Int(py=fields["seasonal_periods"])
    var mode = py_get(fields, "numeric_mode")
    if mode is None or (py_type(mode) is _b().str and len(mode) == 0):
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not _b().str:
        raise Error("ValueError: cannot save invalid numeric_mode")
    mode = py_call(py_call(mode.strip).lower)
    if String(mode) != "fast" and String(mode) != "identical":
        raise Error("ValueError: cannot save invalid numeric_mode")
    var values = Python.dict(format="mojolearn-holtwinters-1", estimator="ExponentialSmoothing",
        numeric_mode=mode, seasonal=fields["seasonal"], initialization_method=fields["initialization_method"])
    var components = _packed(fields["_comps"], 3 * (n - period) * b, context)
    py_setitem(values, "components", components.keep)
    var names = Python.tuple("sse", "alpha", "beta", "gamma", "n_iter", "criterion")
    for i in range(len(names)):  # driver-loop: snapshot all six saved native statistics before opening the archive
        var name = String(names[i])
        var stored = tsa_buffer(fields[name + "_"], name, context)
        if stored.count != b or not stored.c_order or stored.source.itemsize != 4 or ((i < 4 and stored.dtype_number != 11) or (i >= 4 and stored.source.kind != 1)):
            raise Error("ValueError: Holt-Winters archive statistics have invalid dtype or extent")
        py_setitem(values, names[i], stored.keep)
    var meta = tsa_array("q", "<i8", Python.tuple(4), 4, 8, context)
    var words = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(meta)))
    words[0] = Int64(n)
    words[1] = Int64(b)
    words[2] = Int64(period)
    words[3] = Int64(Int(py=fields["start_periods"]))
    var eps = tsa_array("d", "<f8", Python.tuple(1), 1, 8, context)
    MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(eps)))[0] = Float64(py=fields["eps"])
    py_setitem(values, "meta", meta)
    py_setitem(values, "eps", eps)
    _ = components
    return classical_archive_write(args[0], values, context)
