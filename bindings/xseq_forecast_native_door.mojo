# SPDX-License-Identifier: Apache-2.0
"""Native public operations of the sequence-expansion forecasters: the theta
family (`_x_sequence_theta.py`), Croston (`_x_sequence_croston.py`), ETS
(`_x_sequence_ets.py`), GARCH (`_x_sequence_garch.py`), VAR
(`_x_sequence_var.py`) and STL (`_x_sequence_stl.py`).

Each public `fit` and `predict` is one call: the input admission and float32
copy, the model checks in the Python bodies' words, the compiled
`_mojolearn_x_sequence` entry with the same argument words, and the fitted
attributes or the forecast dict. The stored-forecast reuse of theta and ETS
(a repeated `predict` on the same fitted series returns the stored arrays)
is kept as it was.

UNPACK/PACK only (lane doorsplit-cnn-xseq, 2026-11-11): the checks, setting
words, code words, word copies and glue calls run in
`sequence/xseq_forecast_run.mojo`; this file reads the fields, admits and
owns the buffers, releases the GIL around each glue call and publishes the
fitted state.
"""
from std.ffi import c_int
from std.python import Python, PythonObject
from bindings.base_helpers_key_unpack import base_helpers_key
from std.python._cpython import PyObjectPtr
from std.python._cpython import PyObjectPtr, GILReleased
from bindings.native_loaded_module_door import native_loaded_module
from bindings.classical_pandas_door import classical_admit_value, native_pandas_exact, native_pandas_f32
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_exceptions_door import classical_require_fitted
from bindings.sequential_admission_door import _one_of
from providers.python.autoarima_lifecycle_pyapi_door import _fields
from providers.python.array_metadata_pyapi_door import ArrayPin, array_shape_once, array_metadata_once
from providers.python.arima_buffer_pyapi_door import ArimaBuffer, arima_buffer
from core.cast_in import NPY_DOUBLE
from providers.python.forecast_initialize_pyapi_pack import forecast_initialize
from providers.python.forecast_matrix_pyapi_door import forecast_matrix
from providers.python.forecast_owner_pyapi_unpack import forecast_commit
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.xseq_native_common_door import (
    XF32P, XI64P, xseq_key, xseq_base, xseq_input, xseq_refuse_f64, xseq_zeros, xseq_addr, xseq_f32p,
    xseq_reshape, xseq_copy, xseq_item, xseq_first, xseq_column_f32, xseq_column_i64, xseq_mean_dict, xseq_int,
    xseq_float, xseq_truth, xseq_mode,
)
from core.xseq_marshal import xseq_var_blocks
from sequence.xseq_forecast_run import (
    forecast_run_letter, forecast_run_copy_f32_words, forecast_run_theta_check_n, forecast_run_theta_mask,
    forecast_run_theta, forecast_run_theta_forecast, forecast_run_theta_codes, forecast_run_croston_check_n,
    forecast_run_croston_check_h, forecast_run_croston, forecast_run_croston_forecast, forecast_run_ets_check_n,
    forecast_run_ets_season, forecast_run_ets_needs_positive, forecast_run_min_above_zero,
    forecast_run_ets_check_positive, forecast_run_ets_check_tiny, forecast_run_ets_check_counts,
    forecast_run_ets_mask, forecast_run_ets_words,
    forecast_run_ets, forecast_run_garch_check_n, forecast_run_garch_check_finite, forecast_run_garch_horizon,
    forecast_run_garch_words, forecast_run_garch_stall, forecast_run_garch,
)
from core.admitted_input_nd import admitted_cast_nd_f32
from bindings.admitted_sequence_door import admitted_ndarray_f32
from bindings.native_loaded_module_door import NativeModules
from bindings.native_prepared_owner_unpack import prepared_get, prepared_put
from bindings.xseq_keys_unpack import (
    xseq_keys, XSEQ_OPN_VAR_INITIALIZE, XSEQ_OPN_THETA_INITIALIZE, XSEQ_OPN_THETA_FIT, XSEQ_OPN_THETA_PREDICT,
    XSEQ_OPN_CROSTON_FIT, XSEQ_OPN_CROSTON_PREDICT, XSEQ_OPN_ETS_FIT, XSEQ_OPN_ETS_PREDICT, XSEQ_OPN_GARCH_FIT,
)
from bindings.py_balanced import py_item

# Theta's fitted info per model (lane ne1-a-decomp-tsa, plan section A, row
# N031): one prepared-owner record, outside the model's __dict__ and dropped
# with it, `(_y, settings key without h, info copy, model names)`. A predict
# whose settings match but whose horizon differs runs `theta_forecast` (the
# fit's forecast-only tail) instead of the whole fit; a refit stores a new
# `_y` object, so the record no longer matches.
comptime _THETA_FIT_SLOT = "xseq_theta_fit_v1"


def _getattr(owner: PythonObject, name: String) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    return py_call(b.getattr, owner, PythonObject(name))


def _series(source: PythonObject, label: String, copy: Bool, context: PythonObject,
            message: String = "float64 is refused; pass float32") raises -> PythonObject:
    """`(one, Y)`: `y = np.asarray(source)`, float64 refused, and `Y` the
    C-ordered float32 (batch, n_obs) view of it (one series reshaped to one
    row); `copy` keeps a private copy."""
    var nk = xseq_keys()
    var lifecycle = py_item(context, nk[].k_xseq_lifecycle_context)
    classical_admit_value(source, lifecycle)
    var prepared = source
    if native_pandas_exact(source):
        # Native pandas block storage as one float32 copy; storage pandas
        # promotes to float64 or object keeps this operation's refusal.
        prepared = native_pandas_f32(source, lifecycle, False, True, "TypeError: " + label + ": " + message)
    var y = xseq_input(prepared, context)
    var shape = array_shape_once(y, lifecycle)
    if len(shape) != 1 and len(shape) != 2:
        raise Error(label + ": input must be one series or a matrix of series")
    var pin = ArrayPin()
    var metadata = array_metadata_once(y, shape^, label, lifecycle, pin)
    if metadata.dtype_number == NPY_DOUBLE:
        raise Error("TypeError: " + label + ": " + message)
    var one = len(metadata.shape) == 1
    var rows = 1 if one else metadata.shape[0]
    var cols = metadata.shape[0] if one else metadata.shape[1]
    var arr = y
    var address = metadata.source.addr
    if not copy and metadata.f32_c() and py_type(y) is py_item(context, nk[].k_ndarray):
        if one:
            arr = xseq_reshape(y, Python.tuple(1, cols))
    else:
        # One parallel cast into uninitialised storage (REGRESSIONS F3/C3).
        arr = admitted_ndarray_f32(Python.tuple(rows, cols), rows * cols, context)
        var strides = List[Int]()
        strides.append(metadata.source.s0)
        if not one:
            strides.append(metadata.source.s1)
        _ = admitted_cast_nd_f32(metadata.source.addr, metadata.shape, strides, metadata.source.itemsize,
                     metadata.source.kind, xseq_f32p(arr))
        address = Int(py=xseq_addr(arr))
    _ = metadata
    _ = pin^  # the door's pin held the input storage through the cast
    return Python.tuple(one, arr, rows, cols, address)


def _no_exog(value: PythonObject, message: String) raises:
    if value is not None:
        raise Error("NotImplementedError: " + message)


def _store(fields: PythonObject, key: PythonObject, value: PythonObject) raises:
    # bw2-sequence (2026-11-20): takes the held key `nk[].k_<name>` (it built
    # the Python str from a String name on every store).
    py_setitem(fields, key, value)


def _set(value: PythonObject) -> Bool:
    return value is not None


def _float_or_zero(value: PythonObject) raises -> Float64:
    return 0.0 if value is None else xseq_float(value)


def _stored_buffer(value: PythonObject, name: String, rank: Int, context: PythonObject) raises -> ArimaBuffer:
    var nk = xseq_keys()
    if py_type(value) is not py_item(context, nk[].k_ndarray):
        raise Error("TypeError: stored forecast buffers require exact native ndarrays")
    var captured = arima_buffer(value, name, py_item(context, nk[].k_xseq_lifecycle_context), True)
    if len(captured.shape) != rank:
        raise Error("ValueError: stored forecast buffer has the wrong rank")
    return captured^


def _cache_key(value: PythonObject, context: PythonObject) raises:
    var pending = Python.list(value)
    var tuple_type = py_type(Python.tuple())
    var nk = xseq_keys()
    while len(pending):  # driver-loop: finite tuple/scalar cache configuration metadata
        var item = py_call(pending.pop)
        if py_type(item) is tuple_type:
            for i in range(len(item)):  # driver-loop: fixed cache key metadata words
                _ = py_call(pending.append, item[i])
        else:
            _setting(item, py_item(context, nk[].k_xseq_lifecycle_context))


def _cache_record(value: PythonObject, context: PythonObject) raises:
    if value is None:
        return
    if py_type(value) is not py_type(Python.tuple()) or len(value) != 4:
        raise Error("TypeError: forecast cache requires an exact four-item tuple")
    _cache_key(value[0], context)


def _cache_copy(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var captured = _stored_buffer(value, "forecast cache", 2, context)
    var result = xseq_zeros(Python.tuple(captured.shape[0], captured.shape[1]), context)
    forecast_run_copy_f32_words(Int(py=xseq_addr(result)), captured.address, captured.count)
    _ = result  # keep the Python object alive through the native copy above
    _ = captured
    return result


# ---------------------------------------------------------------- theta

def _theta_fit(model: PythonObject, fields: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    _no_exog(args[1], "Theta: exogenous regressors are not implemented")
    var series = _series(args[0], "Theta", True, context)
    var nk = xseq_keys()
    _store(fields, nk[].k__one, series[0])
    _store(fields, nk[].k__y, series[1])
    var y = series[1]
    forecast_run_theta_check_n(Int(py=series[3]))
    _store(fields, nk[].k__fitted_h, PythonObject(None))
    return model


def _theta_predict(model: PythonObject, fields: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    if args[2] is not None:
        raise Error("NotImplementedError: Theta: prediction intervals (level) are not implemented")
    var b = native_loaded_module("builtins")
    var h = xseq_int(args[0])
    var nk = xseq_keys()
    var models = py_item(context, nk[].k_xseq_theta_models)
    var chosen = py_get(fields, nk[].k_model)
    # the fit runs inside the forecast call; a repeat of the same call on
    # the same fitted series returns the stored answer. p3-cnn-sequence-tsa:
    # the key holds the RESOLVED mode (which also names the binding), so a
    # change of the process default between two predicts recomputes.
    var key = Python.tuple(h, py_item(fields, nk[].k_season_length), py_item(fields, nk[].k_decomposition_type), chosen,
                           py_item(fields, nk[].k_initial_smoothed), py_item(fields, nk[].k_alpha), py_item(fields, nk[].k_theta), xseq_mode(fields, context))
    var cached = py_get(fields, nk[].k__fitted_h)
    _cache_record(cached, context)
    var fit_key = Python.tuple(py_item(fields, nk[].k_season_length), py_item(fields, nk[].k_decomposition_type), chosen,
                               py_item(fields, nk[].k_initial_smoothed), py_item(fields, nk[].k_alpha), py_item(fields, nk[].k_theta), xseq_mode(fields, context))
    var mods = NativeModules()
    var fitted = prepared_get(mods, _THETA_FIT_SLOT, model)
    var forecast: PythonObject
    if cached is not None and Bool(py=cached[0] == key):
        if py_type(cached[3]) is not b.list and py_type(cached[3]) is not b.tuple:
            raise Error("TypeError: Theta cached model names require a native sequence")
        for i in range(len(cached[3])):  # driver-loop: cached per-series native model-name metadata
            if py_type(cached[3][i]) is not b.str:
                raise Error("TypeError: Theta cached model names must be exact strings")
        _store(fields, nk[].k_info_, _cache_copy(cached[2], context))
        _store(fields, nk[].k_model_, py_call(b.list, cached[3]))
        forecast = _cache_copy(cached[1], context)
    elif (fitted is not None and py_contains(fields, nk[].k__y) and fitted[0] is py_item(fields, nk[].k__y)
          and Bool(py=fitted[1] == fit_key)):
        # The same fitted series and settings at another horizon: the fit's
        # own info rows drive the forecast-only tail (no Nelder-Mead); the
        # words are the full call's (sequence/theta.mojo op_theta_forecast).
        # Every image publishes it in the sequence table, so the former
        # "entry present" test is gone (lane calltable-xseq).
        # The table key is `entry_key`: `key` stays the stored-forecast cache
        # key stored in `_fitted_h` below (the former inner `var key` of the
        # table key shadowed it there).
        var entry_key = xseq_key(fields, context)
        var captured = _stored_buffer(py_item(fields, nk[].k__y), "Theta observations", 2, context)
        var y = captured.keep
        var batch = captured.shape[0]
        var n = captured.shape[1]
        var info = _cache_copy(fitted[2], context)
        forecast = xseq_zeros(Python.tuple(batch, h), context)
        var y_addr = captured.address
        var info_addr = Int(py=xseq_addr(info))
        var forecast_addr = Int(py=xseq_addr(forecast))
        var season = Int(py=py_item(fields, nk[].k_season_length))
        var py = Python()
        with GILReleased(py):
            forecast_run_theta_forecast(entry_key, y_addr, info_addr, forecast_addr, batch, n, h, season)
        _ = forecast  # keep the Python object alive through the native call above
        _ = info  # keep the Python object alive through the native call above
        _ = y  # keep the Python object alive through the native call above
        _store(fields, nk[].k_info_, info)
        var names = py_call(b.list, fitted[3])
        _store(fields, nk[].k_model_, names)
        _store(fields, nk[].k__fitted_h, Python.tuple(key, xseq_copy(forecast), xseq_copy(info), py_call(b.list, names)))
    else:
        # `entry_key`, not `key`: see the forecast-only branch above.
        var entry_key = xseq_key(fields, context)
        if not py_contains(fields, nk[].k__y):
            raise Error("AttributeError: Theta: call fit before predict")
        var captured = _stored_buffer(py_item(fields, nk[].k__y), "Theta observations", 2, context)
        var y = captured.keep
        var batch = captured.shape[0]
        var n = captured.shape[1]
        forecast = xseq_zeros(Python.tuple(batch, h), context)
        var info = xseq_zeros(Python.tuple(batch, 8), context)
        var mask = forecast_run_theta_mask(_set(py_item(fields, nk[].k_initial_smoothed)), _set(py_item(fields, nk[].k_alpha)),
                                           _set(py_item(fields, nk[].k_theta)))
        var fp: List[Float64] = [_float_or_zero(py_item(fields, nk[].k_initial_smoothed)), _float_or_zero(py_item(fields, nk[].k_alpha)),
                                 _float_or_zero(py_item(fields, nk[].k_theta))]
        var code = -1 if chosen is None else Int(py=py_item(context, nk[].k_xseq_theta_codes)[chosen])
        var additive = not Bool(py=py_item(fields, nk[].k_decomposition_type) == PythonObject("multiplicative"))
        var y_addr = captured.address
        var forecast_addr = Int(py=xseq_addr(forecast))
        var info_addr = Int(py=xseq_addr(info))
        var season = Int(py=py_item(fields, nk[].k_season_length))
        var py = Python()
        with GILReleased(py):
            forecast_run_theta(entry_key, y_addr, forecast_addr, info_addr, batch, n, h, season, code, additive,
                               mask, fp)
        _ = forecast  # keep the Python object alive through the native call above
        _ = info  # keep the Python object alive through the native call above
        _ = y  # keep the Python object alive through the native call above
        _store(fields, nk[].k_info_, info)
        # the selected model of each series: info column 4 as an index into the names
        var names = _theta_model_names(models, info, batch)
        _store(fields, nk[].k_model_, names)
        _store(fields, nk[].k__fitted_h, Python.tuple(key, xseq_copy(forecast), xseq_copy(info), py_call(b.list, names)))
        prepared_put(mods, _THETA_FIT_SLOT, model, Python.tuple(py_item(fields, nk[].k__y), fit_key, xseq_copy(info),
                                                                 py_call(b.list, names)))
    return xseq_mean_dict(xseq_first(forecast, xseq_truth(py_item(fields, nk[].k__one))))


def _theta_model_names(models: PythonObject, info: PythonObject, batch: Int) raises -> PythonObject:
    """`list(map(models.__getitem__, info[:, 4].astype(np.int64).tolist()))`
    read natively: each row's float32 code word truncated toward zero (as
    astype), the connected name at that index (Python indexing, its own
    IndexError) stored straight into one new list. No int64 column copy,
    no `tolist` and no `map` over the series."""
    ref cpy = Python().cpython()
    var raw = cpy.lib.get_function[PyObjectPtr]("PyList_New")(batch)
    if not raw:
        raise cpy.unsafe_get_error()
    # Own the NULL-filled list at once: a failure releases every stored slot.
    var names = PythonObject(from_owned=raw)
    var set_item = cpy.lib.get_function[c_int]("PyList_SetItem")
    var codes = forecast_run_theta_codes(Int(py=xseq_addr(info)), batch)
    _ = info  # keep the code words alive through the native read above
    # perf-audit-3 (pattern 8): each connected name is read once, then each
    # series takes a new reference to its cached name. Before, every series
    # made a generic `models[code]` CPython item call. A code outside
    # [0, len(models)) still goes through Python indexing (its own result or
    # IndexError, as before).
    var model_count = len(models)
    var cached = List[PythonObject](capacity=model_count)
    for k in range(model_count):  # driver-loop: over the connected model names (a handful), read once
        cached.append(models[k])
    # PACK: one PyList_SetItem per fitted series (the public result is a list of str).
    for row in range(batch):
        var code = codes[row]
        var name = cached[code].copy() if code >= 0 and code < model_count else models[code]
        # SetItem steals the reference on both success and failure.
        if set_item(names._obj_ptr, row, name.steal_data()) != 0:
            raise cpy.unsafe_get_error()
    return names


# ---------------------------------------------------------------- Croston

def _croston_fit(model: PythonObject, fields: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    _no_exog(args[1], "Croston: exogenous regressors are not supported (nor in the reference)")
    var series = _series(args[0], "Croston", False, context)
    var nk = xseq_keys()
    _store(fields, nk[].k__one, series[0])
    _store(fields, nk[].k__y, series[1])
    var y = series[1]
    forecast_run_croston_check_n(Int(py=series[3]))
    var batch = Int(py=series[2])
    var n = Int(py=series[3])
    var mean = xseq_zeros(Python.tuple(batch), context)
    _store(fields, nk[].k_mean_, mean)
    var key = xseq_key(fields, context)
    var y_addr = Int(py=series[4])
    var mean_addr = Int(py=xseq_addr(mean))
    var variant = Int(py=_croston_variant(model, context))
    var py = Python()
    with GILReleased(py):
        forecast_run_croston(key, y_addr, mean_addr, batch, n, variant)
    _ = mean  # keep the Python object alive through the native call above
    _ = y  # keep the Python object alive through the native call above
    return model


def _croston_predict(model: PythonObject, fields: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    if args[2] is not None:
        raise Error("NotImplementedError: Croston: prediction intervals are not implemented")
    var h = xseq_int(args[0])
    forecast_run_croston_check_h(h)
    var nk = xseq_keys()
    if not py_contains(fields, nk[].k_mean_):
        raise Error("AttributeError: Croston: call fit before predict")
    var captured = _stored_buffer(py_item(fields, nk[].k_mean_), "Croston means", 1, context)
    var mean = captured.keep
    var batch = captured.shape[0]
    var forecast = xseq_zeros(Python.tuple(batch, h), context)
    # every row its series' mean (`croston_forecast`)
    if h:
        var key = xseq_key(fields, context)
        var mean_addr = captured.address
        var forecast_addr = Int(py=xseq_addr(forecast))
        var py = Python()
        with GILReleased(py):
            forecast_run_croston_forecast(key, mean_addr, forecast_addr, batch, h)
        _ = forecast  # keep the Python object alive through the native call above
        _ = mean  # keep the Python object alive through the native call above
    return xseq_mean_dict(xseq_first(forecast, xseq_truth(py_item(fields, nk[].k__one))))


# ---------------------------------------------------------------- ETS

def _ets_fit(model: PythonObject, fields: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    _no_exog(args[1], "ETS: exogenous regressors are not implemented")
    var series = _series(args[0], "ETS", True, context)
    var nk = xseq_keys()
    _store(fields, nk[].k__one, series[0])
    _store(fields, nk[].k__y, series[1])
    var y = series[1]
    forecast_run_ets_check_n(Int(py=series[3]))
    var spec = String(py=py_item(fields, nk[].k_model))
    var n = Int(py=series[3])
    var m = xseq_int(py_item(fields, nk[].k_season_length))
    var s = forecast_run_ets_season(spec, n, m)
    if forecast_run_ets_needs_positive(spec, s):
        # every value > 0: finite (`all_finite_f32`) and a first-wins minimum above zero (`reduce_stat` 0)
        var size = Int(py=series[2]) * Int(py=series[3])
        # call-table: base helper
        var positive = Bool(py=py_call_typed(xseq_base(context, "all_finite_f32"), series[4], PythonObject(size)))
        _ = y  # keep the Python object alive through the native call above
        if positive:
            # The data-sized reduce_stat runs with the GIL released, as the
            # released base helper (bindings/base_helpers_released.mojo) did
            # before the door split moved the call into RUN.
            var base_ref = base_helpers_key(py_item(context, nk[].k_xseq_base))
            var y_addr = Int(py=series[4])
            var ets_py = Python()
            with GILReleased(ets_py):
                positive = forecast_run_min_above_zero(base_ref, y_addr, size)
            _ = y  # keep the Python object alive through the native call above
        forecast_run_ets_check_positive(positive)
    var damped = xseq_truth(py_item(fields, nk[].k_damped))
    forecast_run_ets_check_tiny(spec, s, n, damped)
    # The free settings, read in the reference's order (a setting the model
    # does not use is not read).
    var alpha_free = py_item(fields, nk[].k_alpha) is None
    var beta_free = forecast_run_letter(spec, 1) != "N" and py_item(fields, nk[].k_beta) is None
    var gamma_free = s != "N" and py_item(fields, nk[].k_gamma) is None
    var phi_free = damped and py_item(fields, nk[].k_phi) is None
    forecast_run_ets_check_counts(spec, s, n, m, damped, alpha_free, beta_free, gamma_free, phi_free)
    _store(fields, nk[].k__season, PythonObject(s))
    _store(fields, nk[].k__last, PythonObject(None))
    return model


def _ets_predict(model: PythonObject, fields: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    if args[2] is not None:
        raise Error("NotImplementedError: ETS: prediction intervals are not implemented")
    var b = native_loaded_module("builtins")
    var nk = xseq_keys()
    if not py_contains(fields, nk[].k__y):
        raise Error("AttributeError: ETS: call fit before predict")
    var captured = _stored_buffer(py_item(fields, nk[].k__y), "ETS observations", 2, context)
    var y = captured.keep
    var batch = captured.shape[0]
    var n = captured.shape[1]
    var s = String(py=py_item(fields, nk[].k__season))
    var m = xseq_int(py_item(fields, nk[].k_season_length)) if s != "N" else 1
    var h = xseq_int(args[0])
    var spec = String(py=py_item(fields, nk[].k_model))
    var mask = forecast_run_ets_mask(_set(py_item(fields, nk[].k_alpha)), _set(py_item(fields, nk[].k_beta)), _set(py_item(fields, nk[].k_phi)),
                                     _set(py_item(fields, nk[].k_gamma)))
    var fp = Python.list(_float_or_zero(py_item(fields, nk[].k_alpha)), _float_or_zero(py_item(fields, nk[].k_beta)),
                         _float_or_zero(py_item(fields, nk[].k_phi)), _float_or_zero(py_item(fields, nk[].k_gamma)))
    var words = forecast_run_ets_words(batch, n, h, spec, xseq_truth(py_item(fields, nk[].k_damped)), mask, s, m)
    var floats: List[Float64] = [_float_or_zero(py_item(fields, nk[].k_alpha)), _float_or_zero(py_item(fields, nk[].k_beta)),
                                 _float_or_zero(py_item(fields, nk[].k_phi)), _float_or_zero(py_item(fields, nk[].k_gamma))]
    var ip = Python.list(batch, n, h, words[3], words[4], words[5], mask, words[7], m)
    var stall = py_get(fields, nk[].k__fast_stall)  # (iterations, relative drop): the FAST stop's quality sweep
    if stall is not None:
        var iterations = xseq_int(stall[0])
        var drop = xseq_float(stall[1])
        words.append(iterations)
        floats.append(drop)
        # `ip` and `fp` are the stored-forecast cache key's words (PACK state)
        _ = py_call(_getattr(ip, "append"), iterations)
        _ = py_call(_getattr(fp, "append"), drop)
    # the fit runs inside the forecast call; a repeat of the same call on
    # the same fitted series returns the stored answer (key: the resolved
    # mode, which also names the binding; p3-cnn-sequence-tsa)
    var key = Python.tuple(py_call(b.tuple, ip), py_call(b.tuple, fp), xseq_mode(fields, context))
    var last = py_get(fields, nk[].k__last)
    _cache_record(last, context)
    var forecast: PythonObject
    var info: PythonObject
    var states: PythonObject
    if last is not None and Bool(py=last[0] == key):
        forecast = _cache_copy(last[1], context)
        info = _cache_copy(last[2], context)
        states = _cache_copy(last[3], context)
    else:
        forecast = xseq_zeros(Python.tuple(batch, h), context)
        info = xseq_zeros(Python.tuple(batch, 10), context)
        states = xseq_zeros(Python.tuple(batch, m), context)
        var y_addr = captured.address
        var forecast_addr = Int(py=xseq_addr(forecast))
        var info_addr = Int(py=xseq_addr(info))
        var states_addr = Int(py=xseq_addr(states))
        var entry_key = xseq_key(fields, context)
        var py = Python()
        with GILReleased(py):
            forecast_run_ets(entry_key, y_addr, forecast_addr, info_addr, states_addr, words, floats)
        _ = forecast  # keep the Python objects alive through the native call above
        _ = info
        _ = states
        _ = y  # keep the Python object alive through the native call above
        _store(fields, nk[].k__last, Python.tuple(key, xseq_copy(forecast), xseq_copy(info), xseq_copy(states)))
    _store(fields, nk[].k_info_, info)
    var one = xseq_truth(py_item(fields, nk[].k__one))
    if s != "N":
        _store(fields, nk[].k_seasonal_states_, xseq_first(states, one))
    return xseq_mean_dict(xseq_first(forecast, one))


# ---------------------------------------------------------------- GARCH

def _garch_fit(model: PythonObject, fields: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var series = _series(args[0], "GARCH", False, context)
    var one = Bool(py=series[0])
    var nk = xseq_keys()
    _store(fields, nk[].k__one, series[0])
    var y = series[1]
    forecast_run_garch_check_n(Int(py=series[3]))
    # the finiteness scan (the base binding's `all_finite_f32`)
    # call-table: base helper
    forecast_run_garch_check_finite(Bool(py=py_call_typed(xseq_base(context, "all_finite_f32"), series[4],
                                                          PythonObject(Int(py=series[2]) * Int(py=series[3])))))
    _ = y  # keep the Python object alive through the native calls above
    var batch = Int(py=series[2])
    var n = Int(py=series[3])
    var h = forecast_run_garch_horizon(xseq_int(args[1]))
    var p = Int(py=py_item(fields, nk[].k_p))
    var o = Int(py=py_item(fields, nk[].k_o))
    var q = Int(py=py_item(fields, nk[].k_q))
    var k = 1 + p + o + q
    var params = xseq_zeros(Python.tuple(batch, 1 + k), context)
    var info = xseq_zeros(Python.tuple(batch, 4), context)
    var sigma = xseq_zeros(Python.tuple(batch, n), context)
    var forecast = xseq_zeros(Python.tuple(batch, h), context)
    var words = forecast_run_garch_words(batch, n, h, p, o, q, Bool(py=py_item(fields, nk[].k_mean) == PythonObject("Constant")))
    var stall = py_get(fields, nk[].k__fast_stall)  # (iterations, relative drop): the FAST stop's quality sweep
    if stall is not None:
        # Python's round() of the drop in parts per 1e9: its exact integer
        # (ties to even), as `forecast_run_round_half_even` (the former b.round call).
        forecast_run_garch_stall(words, xseq_int(stall[0]), xseq_float(stall[1]))
    var y_addr = Int(py=series[4])
    var params_addr = Int(py=xseq_addr(params))
    var info_addr = Int(py=xseq_addr(info))
    var sigma_addr = Int(py=xseq_addr(sigma))
    var forecast_addr = Int(py=xseq_addr(forecast))
    var entry_key = xseq_key(fields, context)
    var py = Python()
    with GILReleased(py):
        forecast_run_garch(entry_key, y_addr, params_addr, info_addr, sigma_addr, forecast_addr, words)
    _ = sigma  # keep the Python object alive through the native call above
    _ = info  # keep the Python object alive through the native call above
    _ = params  # keep the Python object alive through the native call above
    _ = y  # keep the Python object alive through the native call above
    _ = forecast  # keep the Python object alive through the native call above
    _store(fields, nk[].k__y, y)
    _store(fields, nk[].k_params_, xseq_first(params, one))
    _store(fields, nk[].k_loglikelihood_, xseq_first(xseq_column_f32(info, batch, 4, 0, context), one))
    _store(fields, nk[].k_n_iter_, xseq_first(xseq_column_i64(info, batch, 4, 1, context), one))  # Nelder-Mead iterations, both runs
    _store(fields, nk[].k_conditional_volatility_, xseq_first(sigma, one))
    _store(fields, nk[].k__fc, forecast)
    return model


def _croston_variant(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = xseq_keys()
    var records = py_item(context, nk[].k_xseq_croston_classes)
    for i in range(len(records)):  # driver-loop: over connected Croston class metadata
        if py_type(model) is records[i][0]:
            return records[i][1]
    raise Error("TypeError: Croston requires an exact connected forecaster class")


def _setting(value: PythonObject, context: PythonObject) raises:
    var cls = py_type(value)
    if value is None or cls is py_type(PythonObject("")) or cls is py_type(PythonObject(True)) or cls is py_type(PythonObject(0)) or cls is py_type(PythonObject(0.0)):
        return
    # The table only where the old test built its key: past the built-in types.
    var nk = xseq_keys()
    if _one_of(cls, py_item(context, nk[].k_scalar_types)):
        return
    raise Error("TypeError: forecast settings require native scalar metadata")


# bw2-sequence (2026-11-20): the forecaster kind is a code (its block of the
# xseq operation codes, bindings/xseq_keys_unpack.mojo); it was a String
# compared per call, and the class-record key was built from it per call.
comptime _KIND_THETA = 0
comptime _KIND_CROSTON = 1
comptime _KIND_ETS = 2
comptime _KIND_GARCH = 3


def _kind_name(kind: Int) -> String:
    """The forecaster's name, for a message or the initialize owner (which
    takes it by name)."""
    if kind == _KIND_THETA:
        return "theta"
    if kind == _KIND_CROSTON:
        return "croston"
    if kind == _KIND_ETS:
        return "ets"
    return "garch"


def _forecast_fields(model: PythonObject, kind: Int, context: PythonObject) raises -> PythonObject:
    var nk = xseq_keys()
    var records: PythonObject
    if kind == _KIND_THETA:
        records = py_item(context, nk[].k_xseq_theta_classes)
    elif kind == _KIND_CROSTON:
        records = py_item(context, nk[].k_xseq_croston_classes)
    elif kind == _KIND_ETS:
        records = py_item(context, nk[].k_xseq_ets_classes)
    else:
        records = py_item(context, nk[].k_xseq_garch_classes)
    var cls = py_type(model)
    var found = False
    for i in range(len(records)):  # driver-loop: over connected forecast classes
        found = found or cls is records[i][0]
    if not found:
        raise Error("TypeError: forecast requires an exact connected model class")
    var fields = _fields(model, cls)
    var names = Python.tuple(nk[].k_numeric_mode, nk[].k__one)
    if kind == _KIND_THETA:
        names = Python.tuple(nk[].k_numeric_mode, nk[].k__one, nk[].k_season_length, nk[].k_decomposition_type,
                             nk[].k_model, nk[].k_initial_smoothed, nk[].k_alpha, nk[].k_theta)
    elif kind == _KIND_ETS:
        names = Python.tuple(nk[].k_numeric_mode, nk[].k__one, nk[].k__season, nk[].k_season_length, nk[].k_model,
                             nk[].k_damped, nk[].k_alpha, nk[].k_beta, nk[].k_phi, nk[].k_gamma)
    elif kind == _KIND_GARCH:
        names = Python.tuple(nk[].k_numeric_mode, nk[].k_p, nk[].k_o, nk[].k_q, nk[].k_mean, nk[].k_dist,
                             nk[].k_power, nk[].k__one)
    var lifecycle = py_item(context, nk[].k_xseq_lifecycle_context)
    for i in range(len(names)):  # driver-loop: over consumed forecast scalar settings
        _setting(py_get(fields, names[i]), lifecycle)
    var stall = py_get(fields, nk[].k__fast_stall)
    if stall is not None:
        if py_type(stall) is not py_type(Python.tuple()) or len(stall) != 2:
            raise Error("TypeError: forecast stopping metadata requires a native pair")
        _setting(stall[0], lifecycle)
        _setting(stall[1], lifecycle)
    return fields


def _forecast_operation(model: PythonObject, kind: Int, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var target = _forecast_fields(model, kind, context)
    # Fit/predict owners stage state in an exact dictionary. Commit only after
    # every replaced original owner has passed release admission.
    var fields = py_call(native_loaded_module("builtins").dict, target)
    var nk = xseq_keys()
    var lifecycle = py_item(context, nk[].k_xseq_lifecycle_context)
    if kind == _KIND_GARCH:
        _setting(args[0], lifecycle)
        var h = xseq_int(args[0])
        var forecast = py_get(fields, nk[].k__fc)
        if forecast is None:
            raise Error("AttributeError: GARCH: call fit() before forecast()")
        classical_admit_value(forecast, lifecycle)
        if py_type(forecast) is not py_item(context, nk[].k_ndarray):
            raise Error("TypeError: GARCH forecast storage requires its native ndarray")
        var pin = ArrayPin()
        var metadata = array_metadata_once(forecast, array_shape_once(forecast, lifecycle), "forecast", lifecycle, pin)
        if len(metadata.shape) != 2 or not metadata.f32_c():
            raise Error("TypeError: GARCH forecast storage requires a contiguous float32 matrix")
        if h > metadata.shape[1]:
            var y = py_item(fields, nk[].k__y)
            classical_admit_value(y, lifecycle)
            if py_type(y) is not py_item(context, nk[].k_ndarray):
                raise Error("TypeError: GARCH stored observations require a native ndarray")
            _ = _garch_fit(model, fields, Python.tuple(xseq_first(y, xseq_truth(py_item(fields, nk[].k__one))), h), context)
            forecast = py_item(fields, nk[].k__fc)
        var b = native_loaded_module("builtins")
        # Exact ndarray native slicing preserves the original view semantics,
        # including zero/negative horizons and the one-series output rank.
        var result = forecast[Python.tuple(py_call(b.slice, PythonObject(None)), py_call(b.slice, PythonObject(None), h))]
        _ = metadata
        _ = pin^
        var finished = xseq_first(result, xseq_truth(py_item(fields, nk[].k__one)))
        forecast_commit(target, fields, py_item(context, nk[].k_xseq_forecast_ownership))
        return finished
    _setting(args[1], lifecycle)
    _setting(args[5], lifecycle)
    if xseq_truth(args[5]):
        raise Error("NotImplementedError: " + _kind_name(kind) + ": fitted values are not implemented")
    # Preserve the public conveniences' accepted exogenous arguments: Theta
    # forwards X to fit; Croston/ETS historically ignore X and X_future.
    var fit_args = Python.tuple(args[0], args[2] if kind == _KIND_THETA else PythonObject(None))
    var predict_args = Python.tuple(args[1], PythonObject(None), args[4])
    var result = PythonObject(None)
    if kind == _KIND_CROSTON:
        _ = _croston_fit(model, fields, fit_args, context)
        result = _croston_predict(model, fields, predict_args, context)
    elif kind == _KIND_ETS:
        _ = _ets_fit(model, fields, fit_args, context)
        result = _ets_predict(model, fields, predict_args, context)
    else:
        _ = _theta_fit(model, fields, fit_args, context)
        result = _theta_predict(model, fields, predict_args, context)
    forecast_commit(target, fields, py_item(context, nk[].k_xseq_forecast_ownership))
    return result



def _forecast_unknown(model: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """A name that is not an xseq operation (the shells pass literals only):
    the checks the former name chain made, in its order, before its
    unknown-operation error. The matrix owner raises its own."""
    if operation.startswith("var_") or operation.startswith("stl_"):
        return forecast_matrix(model, operation, args, context)
    var kind = (_KIND_THETA if operation.startswith("theta_") else _KIND_CROSTON if operation.startswith("croston_")
                else _KIND_ETS if operation.startswith("ets_") else _KIND_GARCH)
    _ = _forecast_fields(model, kind, context)
    var nk = xseq_keys()
    if operation.endswith("_predict"):
        classical_require_fitted(model, False, "predict", py_item(context, nk[].k_xseq_lifecycle_context))
    if operation.endswith("predict"):
        _setting(args[0], py_item(context, nk[].k_xseq_lifecycle_context))
    raise Error("RuntimeError: unknown sequence forecaster operation " + operation)


def xseq_forecast(model: PythonObject, operation: Int, name: PythonObject, args: PythonObject,
                  context: PythonObject) raises -> PythonObject:
    """`operation` is the xseq code (bindings/xseq_keys_unpack.mojo); `name`
    is the caller's str, read only by the owners that take a name (the VAR/STL
    matrix owner) and by a name that is not an operation (-1)."""
    var nk = xseq_keys()
    # Admission and release ownership use distinct contexts. Connect this one
    # on entry; another public operation need not have initialized it.
    classical_connect_numpy(py_item(context, nk[].k_xseq_lifecycle_context))
    if operation < XSEQ_OPN_THETA_INITIALIZE:
        if operation >= XSEQ_OPN_VAR_INITIALIZE:
            # providers/python/forecast_matrix_pyapi_door.mojo (another family) takes the name.
            return forecast_matrix(model, String(py=name), args, context)
        return _forecast_unknown(model, String(py=name), args, context)
    # One block of four codes per kind: initialize, forecast, fit, predict.
    var kind = (operation - XSEQ_OPN_THETA_INITIALIZE) // 4
    var step = (operation - XSEQ_OPN_THETA_INITIALIZE) % 4
    if step == 0:
        return forecast_initialize(model, _kind_name(kind), args, context)
    if step == 1:
        return _forecast_operation(model, kind, args, context)
    var target = _forecast_fields(model, kind, context)
    var fields = py_call(native_loaded_module("builtins").dict, target)
    if step == 3:
        # The fitted check the former `_mode` class-creation guard made:
        # NotFittedError (the xseq error boundary raises the active class)
        # unless a fit completed (`_mojolearn_fitted`) or a public fitted
        # attribute is set (scikit-learn's check_is_fitted rule). It comes
        # before the horizon is read, as that guard ran before the method
        # body: an unfitted predict raises NotFittedError whatever h is. It
        # reads only the model's own fields, never the arguments.
        classical_require_fitted(model, False, "predict", py_item(context, nk[].k_xseq_lifecycle_context))
        _setting(args[0], py_item(context, nk[].k_xseq_lifecycle_context))
    elif operation == XSEQ_OPN_GARCH_FIT:
        _setting(args[1], py_item(context, nk[].k_xseq_lifecycle_context))
    var result = PythonObject(None)
    if operation == XSEQ_OPN_THETA_FIT:
        result = _theta_fit(model, fields, args, context)
    elif operation == XSEQ_OPN_THETA_PREDICT:
        result = _theta_predict(model, fields, args, context)
    elif operation == XSEQ_OPN_CROSTON_FIT:
        result = _croston_fit(model, fields, args, context)
    elif operation == XSEQ_OPN_CROSTON_PREDICT:
        result = _croston_predict(model, fields, args, context)
    elif operation == XSEQ_OPN_ETS_FIT:
        result = _ets_fit(model, fields, args, context)
    elif operation == XSEQ_OPN_ETS_PREDICT:
        result = _ets_predict(model, fields, args, context)
    elif operation == XSEQ_OPN_GARCH_FIT:
        result = _garch_fit(model, fields, args, context)
    else:
        raise Error("RuntimeError: unknown sequence forecaster operation " + String(py=name))
    if step == 2:
        # Fit bookkeeping of the former `_mode` fit wrapper: a completed fit
        # marks the instance fitted (staged, committed with the fitted state).
        _store(fields, nk[].k__mojolearn_fitted, PythonObject(True))
    forecast_commit(target, fields, py_item(context, nk[].k_xseq_forecast_ownership))
    return result
