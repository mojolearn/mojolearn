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
"""
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.classical_pandas import classical_admit_value
from bindings.classical_attributes import classical_connect_numpy
from bindings.sequential_admission import _one_of
from providers.python.autoarima_lifecycle_pyapi import _fields
from providers.python.array_metadata_pyapi import array_shape_once, array_metadata_once
from providers.python.arima_buffer_pyapi import ArimaBuffer, arima_buffer
from core.cast_in import NPY_DOUBLE
from providers.python.forecast_initialize_pyapi import forecast_initialize
from providers.python.forecast_matrix_pyapi import forecast_matrix
from providers.python.forecast_owner_pyapi import forecast_commit
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.xseq_native_common import (
    XF32P, XI64P, xseq_entries, xseq_base, xseq_input, xseq_f32, xseq_refuse_f64, xseq_zeros, xseq_addr, xseq_f32p,
    xseq_reshape, xseq_copy, xseq_item, xseq_first, xseq_column_f32, xseq_column_i64, xseq_mean_dict, xseq_int,
    xseq_float, xseq_truth,
)
from core.xseq_marshal import xseq_var_blocks, xseq_cast_nd


def _getattr(owner: PythonObject, name: String) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    return py_call(b.getattr, owner, PythonObject(name))


def _series(source: PythonObject, label: String, copy: Bool, context: PythonObject,
            message: String = "float64 is refused; pass float32") raises -> PythonObject:
    """`(one, Y)`: `y = np.asarray(source)`, float64 refused, and `Y` the
    C-ordered float32 (batch, n_obs) view of it (one series reshaped to one
    row); `copy` keeps a private copy."""
    var lifecycle = context["xseq_lifecycle_context"]
    classical_admit_value(source, lifecycle)
    var y = xseq_input(source, context)
    var shape = array_shape_once(y, lifecycle)
    if len(shape) != 1 and len(shape) != 2:
        raise Error(label + ": input must be one series or a matrix of series")
    var metadata = array_metadata_once(y, shape^, label, lifecycle)
    if metadata.dtype_number == NPY_DOUBLE:
        raise Error("TypeError: " + label + ": " + message)
    var one = len(metadata.shape) == 1
    var rows = 1 if one else metadata.shape[0]
    var cols = metadata.shape[0] if one else metadata.shape[1]
    var arr = y
    var address = metadata.source.addr
    if not copy and metadata.f32_c() and py_type(y) is context["ndarray"]:
        if one:
            arr = xseq_reshape(y, Python.tuple(1, cols))
    else:
        arr = xseq_zeros(Python.tuple(rows, cols), context)
        var strides = List[Int]()
        strides.append(metadata.source.s0)
        if not one:
            strides.append(metadata.source.s1)
        xseq_cast_nd(metadata.source.addr, metadata.shape, strides, metadata.source.itemsize,
                     metadata.source.kind, False, xseq_f32p(arr))
        address = Int(py=xseq_addr(arr))
    _ = metadata
    return Python.tuple(one, arr, rows, cols, address)


def _no_exog(value: PythonObject, message: String) raises:
    if value is not None:
        raise Error("NotImplementedError: " + message)


def _store(fields: PythonObject, name: String, value: PythonObject) raises:
    py_setitem(fields, PythonObject(name), value)


def _letter(text: String, index: Int) raises -> String:
    return String(text[byte=index:index + 1])


def _flag(value: PythonObject) -> Int:
    return 0 if value is None else 1


def _float_or_zero(value: PythonObject) raises -> Float64:
    return 0.0 if value is None else xseq_float(value)


def _stored_buffer(value: PythonObject, name: String, rank: Int, context: PythonObject) raises -> ArimaBuffer:
    if py_type(value) is not context["ndarray"]:
        raise Error("TypeError: stored forecast buffers require exact native ndarrays")
    var captured = arima_buffer(value, name, context["xseq_lifecycle_context"], True)
    if len(captured.shape) != rank:
        raise Error("ValueError: stored forecast buffer has the wrong rank")
    return captured^


def _cache_key(value: PythonObject, context: PythonObject) raises:
    var pending = Python.list(value)
    var tuple_type = py_type(Python.tuple())
    while len(pending):  # driver-loop: finite tuple/scalar cache configuration metadata
        var item = py_call(pending.pop)
        if py_type(item) is tuple_type:
            for i in range(len(item)):  # driver-loop: fixed cache key metadata words
                _ = py_call(pending.append, item[i])
        else:
            _setting(item, context["xseq_lifecycle_context"])


def _cache_record(value: PythonObject, context: PythonObject) raises:
    if value is None:
        return
    if py_type(value) is not py_type(Python.tuple()) or len(value) != 4:
        raise Error("TypeError: forecast cache requires an exact four-item tuple")
    _cache_key(value[0], context)


def _cache_copy(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var captured = _stored_buffer(value, "forecast cache", 2, context)
    var result = xseq_zeros(Python.tuple(captured.shape[0], captured.shape[1]), context)
    if captured.count:
        memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=xseq_addr(result))),
               src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=captured.address), count=captured.count * 4)
    _ = captured
    return result


# ---------------------------------------------------------------- theta

def _theta_fit(model: PythonObject, fields: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    _no_exog(args[1], "Theta: exogenous regressors are not implemented")
    var series = _series(args[0], "Theta", True, context)
    _store(fields, "_one", series[0])
    _store(fields, "_y", series[1])
    var y = series[1]
    if Int(py=series[3]) <= 3:
        raise Error("Theta: each series needs more than 3 observations")
    _store(fields, "_fitted_h", PythonObject(None))
    return model


def _theta_predict(model: PythonObject, fields: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    if args[2] is not None:
        raise Error("NotImplementedError: Theta: prediction intervals (level) are not implemented")
    var b = native_loaded_module("builtins")
    var h = xseq_int(args[0])
    var models = context["xseq_theta_models"]
    var chosen = py_get(fields, "model")
    # the fit runs inside the forecast call; a repeat of the same call on
    # the same fitted series returns the stored answer
    var key = Python.tuple(h, fields["season_length"], fields["decomposition_type"], chosen,
                           fields["initial_smoothed"], fields["alpha"], fields["theta"], py_get(fields, "numeric_mode"))
    var cached = py_get(fields, "_fitted_h")
    _cache_record(cached, context)
    var forecast: PythonObject
    if cached is not None and Bool(py=cached[0] == key):
        if py_type(cached[3]) is not b.list and py_type(cached[3]) is not b.tuple:
            raise Error("TypeError: Theta cached model names require a native sequence")
        for i in range(len(cached[3])):  # driver-loop: cached per-series native model-name metadata
            if py_type(cached[3][i]) is not b.str:
                raise Error("TypeError: Theta cached model names must be exact strings")
        _store(fields, "info_", _cache_copy(cached[2], context))
        _store(fields, "model_", py_call(b.list, cached[3]))
        forecast = _cache_copy(cached[1], context)
    else:
        var entries = xseq_entries(fields, context)
        if not py_contains(fields, "_y"):
            raise Error("AttributeError: Theta: call fit before predict")
        var captured = _stored_buffer(fields["_y"], "Theta observations", 2, context)
        var y = captured.keep
        var batch = captured.shape[0]
        var n = captured.shape[1]
        forecast = xseq_zeros(Python.tuple(batch, h), context)
        var info = xseq_zeros(Python.tuple(batch, 8), context)
        var mask = (_flag(fields["initial_smoothed"]) | 2 * _flag(fields["alpha"]) | 4 * _flag(fields["theta"]))
        var fp = Python.list(_float_or_zero(fields["initial_smoothed"]), _float_or_zero(fields["alpha"]),
                             _float_or_zero(fields["theta"]))
        var code = -1 if chosen is None else Int(py=context["xseq_theta_codes"][chosen])
        var additive = 0 if Bool(py=fields["decomposition_type"] == PythonObject("multiplicative")) else 1
        _ = py_call_typed(entries["theta"], Python.list(captured.address, xseq_addr(forecast), xseq_addr(info)),
                          Python.list(batch, n, h, fields["season_length"], code, additive, mask), fp)
        _ = info  # keep the Python object alive through the native call above
        _ = y  # keep the Python object alive through the native call above
        _store(fields, "info_", info)
        # the selected model of each series: info column 4 as an index into the names
        var codes = xseq_column_i64(info, batch, 8, 4, context)
        var names = py_call(b.list, py_call(b.map, _getattr(models, "__getitem__"), py_call(_getattr(codes, "tolist"))))
        _store(fields, "model_", names)
        _store(fields, "_fitted_h", Python.tuple(key, xseq_copy(forecast), xseq_copy(info), py_call(b.list, names)))
    return xseq_mean_dict(xseq_first(forecast, xseq_truth(fields["_one"])))


# ---------------------------------------------------------------- Croston

def _croston_fit(model: PythonObject, fields: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    _no_exog(args[1], "Croston: exogenous regressors are not supported (nor in the reference)")
    var series = _series(args[0], "Croston", False, context)
    _store(fields, "_one", series[0])
    _store(fields, "_y", series[1])
    var y = series[1]
    if Int(py=series[3]) < 1:
        raise Error("Croston: y must hold at least one observation per series")
    var batch = Int(py=series[2])
    var n = Int(py=series[3])
    var mean = xseq_zeros(Python.tuple(batch), context)
    _store(fields, "mean_", mean)
    var entries = xseq_entries(fields, context)
    _ = py_call_typed(entries["croston"], Python.list(series[4], xseq_addr(mean)),
                      Python.list(batch, n, _croston_variant(model, context)))
    _ = mean  # keep the Python object alive through the native call above
    _ = y  # keep the Python object alive through the native call above
    return model


def _croston_predict(model: PythonObject, fields: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    if args[2] is not None:
        raise Error("NotImplementedError: Croston: prediction intervals are not implemented")
    var h = xseq_int(args[0])
    if h < 0:
        raise Error("Croston: h must be >= 0")
    if not py_contains(fields, "mean_"):
        raise Error("AttributeError: Croston: call fit before predict")
    var captured = _stored_buffer(fields["mean_"], "Croston means", 1, context)
    var mean = captured.keep
    var batch = captured.shape[0]
    var forecast = xseq_zeros(Python.tuple(batch, h), context)
    # every row its series' mean (`croston_forecast`)
    if h:
        var entries = xseq_entries(fields, context)
        _ = py_call_typed(entries["croston_forecast"], Python.list(captured.address, xseq_addr(forecast)), Python.list(batch, h))
        _ = forecast  # keep the Python object alive through the native call above
        _ = mean  # keep the Python object alive through the native call above
    return xseq_mean_dict(xseq_first(forecast, xseq_truth(fields["_one"])))


# ---------------------------------------------------------------- ETS

def _ets_fit(model: PythonObject, fields: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    _no_exog(args[1], "ETS: exogenous regressors are not implemented")
    var series = _series(args[0], "ETS", True, context)
    _store(fields, "_one", series[0])
    _store(fields, "_y", series[1])
    var y = series[1]
    if Int(py=series[3]) < 4:
        raise Error("ETS: each series needs at least 4 observations")
    var spec = String(py=fields["model"])
    var e = _letter(spec, 0)
    var tr = _letter(spec, 1)
    var s = _letter(spec, 2)
    var n = Int(py=series[3])
    var m = xseq_int(fields["season_length"])
    if s != "N" and n <= m:
        s = "N"  # the reference drops the season (ets_f)
    if e == "M" or s == "M":
        # every value > 0: finite (`all_finite_f32`) and a first-wins minimum above zero (`reduce_stat` 0)
        var size = Int(py=series[2]) * Int(py=series[3])
        var positive = Bool(py=py_call_typed(xseq_base(context, "all_finite_f32"), series[4], PythonObject(size)))
        _ = y  # keep the Python object alive through the native call above
        if positive:
            var low = py_call_typed(xseq_base(context, "reduce_stat"), series[4], PythonObject(0), PythonObject(size), PythonObject(0))
            _ = y  # keep the Python object alive through the native call above
            positive = Bool(py=low > PythonObject(0))
            _ = y  # keep the Python object alive through the native call above
        _ = y  # keep the Python object alive through the native calls above
        if not positive:
            raise Error("ETS: Inappropriate model for data with negative or zero values")
        _ = y  # keep the Python object alive through the native calls above
    var damped = xseq_truth(fields["damped"])
    var trend = 1 if tr != "N" else 0
    var season = 1 if s != "N" else 0
    var npars = 2 + 2 * trend + 2 * season + (1 if damped else 0)
    if n <= npars + 4:
        raise Error("NotImplementedError: ETS: tiny datasets (n <= parameter count + 4) are not implemented")
    var n_smooth = (1 if fields["alpha"] is None else 0) + (1 if trend == 1 and fields["beta"] is None else 0)
    n_smooth += (1 if season == 1 and fields["gamma"] is None else 0) + (1 if damped and fields["phi"] is None else 0)
    var n_par = n_smooth + 1 + trend + (m - 1 if season == 1 else 0)
    if n_par >= n - 1:
        raise Error("ETS: " + String(n_par) + " parameters for " + String(n)
                    + " observations (the reference needs n > parameters + 1)")
    _store(fields, "_season", PythonObject(s))
    _store(fields, "_last", PythonObject(None))
    return model


def _index(letters: String, letter: String) -> Int:
    return letters.find(letter)


def _ets_predict(model: PythonObject, fields: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    if args[2] is not None:
        raise Error("NotImplementedError: ETS: prediction intervals are not implemented")
    var b = native_loaded_module("builtins")
    if not py_contains(fields, "_y"):
        raise Error("AttributeError: ETS: call fit before predict")
    var captured = _stored_buffer(fields["_y"], "ETS observations", 2, context)
    var y = captured.keep
    var batch = captured.shape[0]
    var n = captured.shape[1]
    var s = String(py=fields["_season"])
    var m = xseq_int(fields["season_length"]) if s != "N" else 1
    var h = xseq_int(args[0])
    var spec = String(py=fields["model"])
    var mask = (_flag(fields["alpha"]) | 2 * _flag(fields["beta"]) | 4 * _flag(fields["phi"]) | 8 * _flag(fields["gamma"]))
    var fp = Python.list(_float_or_zero(fields["alpha"]), _float_or_zero(fields["beta"]),
                         _float_or_zero(fields["phi"]), _float_or_zero(fields["gamma"]))
    var ip = Python.list(batch, n, h, _index("AM", _letter(spec, 0)), _index("NA", _letter(spec, 1)),
                         1 if xseq_truth(fields["damped"]) else 0, mask, _index("NAM", s), m)
    var stall = py_get(fields, "_fast_stall")  # (iterations, relative drop): the FAST stop's quality sweep
    if stall is not None:
        _ = py_call(_getattr(ip, "append"), xseq_int(stall[0]))
        _ = py_call(_getattr(fp, "append"), xseq_float(stall[1]))
    # the fit runs inside the forecast call; a repeat of the same call on
    # the same fitted series returns the stored answer
    var key = Python.tuple(py_call(b.tuple, ip), py_call(b.tuple, fp), py_get(fields, "numeric_mode"))
    var last = py_get(fields, "_last")
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
        var entries = xseq_entries(fields, context)
        _ = py_call_typed(entries["ets"], Python.list(captured.address, xseq_addr(forecast), xseq_addr(info), xseq_addr(states)), ip, fp)
        _ = y  # keep the Python object alive through the native call above
        _store(fields, "_last", Python.tuple(key, xseq_copy(forecast), xseq_copy(info), xseq_copy(states)))
    _store(fields, "info_", info)
    var one = xseq_truth(fields["_one"])
    if s != "N":
        _store(fields, "seasonal_states_", xseq_first(states, one))
    return xseq_mean_dict(xseq_first(forecast, one))


# ---------------------------------------------------------------- GARCH

def _garch_fit(model: PythonObject, fields: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var series = _series(args[0], "GARCH", False, context)
    var one = Bool(py=series[0])
    _store(fields, "_one", series[0])
    var y = series[1]
    if Int(py=series[3]) < 10:
        raise Error("GARCH: each series needs at least 10 observations")
    # the finiteness scan (the base binding's `all_finite_f32`)
    if not Bool(py=py_call_typed(xseq_base(context, "all_finite_f32"), series[4], PythonObject(Int(py=series[2]) * Int(py=series[3])))):
        raise Error("GARCH: y holds a non-finite value")
    _ = y  # keep the Python object alive through the native calls above
    var batch = Int(py=series[2])
    var n = Int(py=series[3])
    var h = max(1, xseq_int(args[1]))
    var p = Int(py=fields["p"])
    var o = Int(py=fields["o"])
    var q = Int(py=fields["q"])
    var k = 1 + p + o + q
    var params = xseq_zeros(Python.tuple(batch, 1 + k), context)
    var info = xseq_zeros(Python.tuple(batch, 4), context)
    var sigma = xseq_zeros(Python.tuple(batch, n), context)
    var forecast = xseq_zeros(Python.tuple(batch, h), context)
    var ip = Python.list(batch, n, h, p, o, q, 1 if Bool(py=fields["mean"] == PythonObject("Constant")) else 0)
    var stall = py_get(fields, "_fast_stall")  # (iterations, relative drop): the FAST stop's quality sweep
    if stall is not None:
        _ = py_call(_getattr(ip, "append"), xseq_int(stall[0]))
        _ = py_call(_getattr(ip, "append"), py_call(b.round, PythonObject(xseq_float(stall[1]) * 1e9)))
    var entries = xseq_entries(fields, context)
    _ = py_call_typed(entries["garch"], Python.list(series[4], xseq_addr(params), xseq_addr(info), xseq_addr(sigma),
                      xseq_addr(forecast)), ip)
    _ = sigma  # keep the Python object alive through the native call above
    _ = info  # keep the Python object alive through the native call above
    _ = params  # keep the Python object alive through the native call above
    _ = y  # keep the Python object alive through the native call above
    _ = forecast  # keep the Python object alive through the native call above
    _store(fields, "_y", y)
    _store(fields, "params_", xseq_first(params, one))
    _store(fields, "loglikelihood_", xseq_first(xseq_column_f32(info, batch, 4, 0, context), one))
    _store(fields, "n_iter_", xseq_first(xseq_column_i64(info, batch, 4, 1, context), one))  # Nelder-Mead iterations, both runs
    _store(fields, "conditional_volatility_", xseq_first(sigma, one))
    _store(fields, "_fc", forecast)
    return model


def _croston_variant(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var records = context["xseq_croston_classes"]
    for i in range(len(records)):  # driver-loop: over connected Croston class metadata
        if py_type(model) is records[i][0]:
            return records[i][1]
    raise Error("TypeError: Croston requires an exact connected forecaster class")


def _setting(value: PythonObject, context: PythonObject) raises:
    var cls = py_type(value)
    if value is None or cls is py_type(PythonObject("")) or cls is py_type(PythonObject(True)) or cls is py_type(PythonObject(0)) or cls is py_type(PythonObject(0.0)) or _one_of(cls, context["scalar_types"]):
        return
    raise Error("TypeError: forecast settings require native scalar metadata")


def _forecast_fields(model: PythonObject, kind: String, context: PythonObject) raises -> PythonObject:
    var records = context["xseq_" + kind + "_classes"]
    var cls = py_type(model)
    var found = False
    for i in range(len(records)):  # driver-loop: over connected forecast classes
        found = found or cls is records[i][0]
    if not found:
        raise Error("TypeError: forecast requires an exact connected model class")
    var fields = _fields(model, cls)
    var names = Python.tuple("numeric_mode", "_one")
    if kind == "theta":
        names = Python.tuple("numeric_mode", "_one", "season_length", "decomposition_type", "model", "initial_smoothed", "alpha", "theta")
    elif kind == "ets":
        names = Python.tuple("numeric_mode", "_one", "_season", "season_length", "model", "damped", "alpha", "beta", "phi", "gamma")
    elif kind == "garch":
        names = Python.tuple("numeric_mode", "p", "o", "q", "mean", "dist", "power", "_one")
    var lifecycle = context["xseq_lifecycle_context"]
    for i in range(len(names)):  # driver-loop: over consumed forecast scalar settings
        _setting(py_get(fields, names[i]), lifecycle)
    var stall = py_get(fields, "_fast_stall")
    if stall is not None:
        if py_type(stall) is not py_type(Python.tuple()) or len(stall) != 2:
            raise Error("TypeError: forecast stopping metadata requires a native pair")
        _setting(stall[0], lifecycle)
        _setting(stall[1], lifecycle)
    return fields


def _forecast_operation(model: PythonObject, kind: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var target = _forecast_fields(model, kind, context)
    # Fit/predict owners stage state in an exact dictionary. Commit only after
    # every replaced original owner has passed release admission.
    var fields = py_call(native_loaded_module("builtins").dict, target)
    var lifecycle = context["xseq_lifecycle_context"]
    if kind == "garch":
        _setting(args[0], lifecycle)
        var h = xseq_int(args[0])
        var forecast = py_get(fields, "_fc")
        if forecast is None:
            raise Error("AttributeError: GARCH: call fit() before forecast()")
        classical_admit_value(forecast, lifecycle)
        if py_type(forecast) is not context["ndarray"]:
            raise Error("TypeError: GARCH forecast storage requires its native ndarray")
        var metadata = array_metadata_once(forecast, array_shape_once(forecast, lifecycle), "forecast", lifecycle)
        if len(metadata.shape) != 2 or not metadata.f32_c():
            raise Error("TypeError: GARCH forecast storage requires a contiguous float32 matrix")
        if h > metadata.shape[1]:
            var y = fields["_y"]
            classical_admit_value(y, lifecycle)
            if py_type(y) is not context["ndarray"]:
                raise Error("TypeError: GARCH stored observations require a native ndarray")
            _ = _garch_fit(model, fields, Python.tuple(xseq_first(y, xseq_truth(fields["_one"])), h), context)
            forecast = fields["_fc"]
        var b = native_loaded_module("builtins")
        # Exact ndarray native slicing preserves the original view semantics,
        # including zero/negative horizons and the one-series output rank.
        var result = forecast[Python.tuple(py_call(b.slice, PythonObject(None)), py_call(b.slice, PythonObject(None), h))]
        _ = metadata
        var finished = xseq_first(result, xseq_truth(fields["_one"]))
        forecast_commit(target, fields, context["xseq_forecast_ownership"])
        return finished
    _setting(args[1], lifecycle)
    _setting(args[5], lifecycle)
    if xseq_truth(args[5]):
        raise Error("NotImplementedError: " + kind + ": fitted values are not implemented")
    # Preserve the public conveniences' accepted exogenous arguments: Theta
    # forwards X to fit; Croston/ETS historically ignore X and X_future.
    var fit_args = Python.tuple(args[0], args[2] if kind == "theta" else PythonObject(None))
    var predict_args = Python.tuple(args[1], PythonObject(None), args[4])
    var result = PythonObject(None)
    if kind == "croston":
        _ = _croston_fit(model, fields, fit_args, context)
        result = _croston_predict(model, fields, predict_args, context)
    elif kind == "ets":
        _ = _ets_fit(model, fields, fit_args, context)
        result = _ets_predict(model, fields, predict_args, context)
    else:
        _ = _theta_fit(model, fields, fit_args, context)
        result = _theta_predict(model, fields, predict_args, context)
    forecast_commit(target, fields, context["xseq_forecast_ownership"])
    return result



def xseq_forecast(model: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    # Admission and release ownership use distinct contexts. Connect this one
    # on entry; another public operation need not have initialized it.
    classical_connect_numpy(context["xseq_lifecycle_context"])
    if operation.startswith("var_") or operation.startswith("stl_") or operation == "decompose_initialize":
        return forecast_matrix(model, operation, args, context)
    if operation == "croston_initialize" or operation == "ets_initialize" or operation == "theta_initialize" or operation == "garch_initialize":
        var kind = "croston" if operation == "croston_initialize" else ("ets" if operation == "ets_initialize" else ("theta" if operation == "theta_initialize" else "garch"))
        return forecast_initialize(model, kind, args, context)
    if operation == "croston_forecast" or operation == "ets_forecast" or operation == "theta_forecast" or operation == "garch_forecast":
        var kind = "croston" if operation == "croston_forecast" else ("ets" if operation == "ets_forecast" else ("theta" if operation == "theta_forecast" else "garch"))
        return _forecast_operation(model, kind, args, context)
    var kind = "theta" if operation.startswith("theta_") else "croston" if operation.startswith("croston_") else "ets" if operation.startswith("ets_") else "garch"
    var target = _forecast_fields(model, kind, context)
    var fields = py_call(native_loaded_module("builtins").dict, target)
    if operation.endswith("predict"):
        _setting(args[0], context["xseq_lifecycle_context"])
    elif operation == "garch_fit":
        _setting(args[1], context["xseq_lifecycle_context"])
    var result = PythonObject(None)
    if operation == "theta_fit":
        result = _theta_fit(model, fields, args, context)
    elif operation == "theta_predict":
        result = _theta_predict(model, fields, args, context)
    elif operation == "croston_fit":
        result = _croston_fit(model, fields, args, context)
    elif operation == "croston_predict":
        result = _croston_predict(model, fields, args, context)
    elif operation == "ets_fit":
        result = _ets_fit(model, fields, args, context)
    elif operation == "ets_predict":
        result = _ets_predict(model, fields, args, context)
    elif operation == "garch_fit":
        result = _garch_fit(model, fields, args, context)
    else:
        raise Error("RuntimeError: unknown sequence forecaster operation " + operation)
    forecast_commit(target, fields, context["xseq_forecast_ownership"])
    return result
