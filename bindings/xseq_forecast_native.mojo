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
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.classical_pandas import classical_admit_value
from bindings.sequential_admission import _one_of
from providers.python.autoarima_lifecycle_pyapi import _fields
from providers.python.array_metadata_pyapi import array_shape_once, array_metadata_once
from core.cast_in import NPY_DOUBLE
from providers.python.forecast_initialize_pyapi import forecast_initialize
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
    if not copy and metadata.f32_c():
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
    _ = metadata
    return Python.tuple(one, arr)


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


# ---------------------------------------------------------------- theta

def _theta_fit(model: PythonObject, fields: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    _no_exog(args[1], "Theta: exogenous regressors are not implemented")
    var series = _series(args[0], "Theta", True, context)
    _store(fields, "_one", series[0])
    _store(fields, "_y", series[1])
    var y = series[1]
    if Int(py=y.ndim) != 2 or Int(py=y.shape[1]) <= 3:
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
    var forecast: PythonObject
    if cached is not None and Bool(py=cached[0] == key):
        _store(fields, "info_", xseq_copy(cached[2]))
        _store(fields, "model_", py_call(b.list, cached[3]))
        forecast = xseq_copy(cached[1])
    else:
        var entries = xseq_entries(fields, context)
        var y = fields["_y"]
        var batch = Int(py=y.shape[0])
        var n = Int(py=y.shape[1])
        forecast = xseq_zeros(Python.tuple(batch, h), context)
        var info = xseq_zeros(Python.tuple(batch, 8), context)
        var mask = (_flag(fields["initial_smoothed"]) | 2 * _flag(fields["alpha"]) | 4 * _flag(fields["theta"]))
        var fp = Python.list(_float_or_zero(fields["initial_smoothed"]), _float_or_zero(fields["alpha"]),
                             _float_or_zero(fields["theta"]))
        var code = -1 if chosen is None else Int(py=context["xseq_theta_codes"][chosen])
        var additive = 0 if Bool(py=fields["decomposition_type"] == PythonObject("multiplicative")) else 1
        _ = py_call_typed(entries["theta"], Python.list(xseq_addr(y), xseq_addr(forecast), xseq_addr(info)),
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
    if Int(py=y.ndim) != 2 or Int(py=y.shape[1]) < 1:
        raise Error("Croston: y must hold at least one observation per series")
    var batch = Int(py=y.shape[0])
    var n = Int(py=y.shape[1])
    var mean = xseq_zeros(Python.tuple(batch), context)
    _store(fields, "mean_", mean)
    var entries = xseq_entries(fields, context)
    _ = py_call_typed(entries["croston"], Python.list(xseq_addr(y), xseq_addr(mean)),
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
    var mean = fields["mean_"]
    var batch = Int(py=mean.shape[0])
    var forecast = xseq_zeros(Python.tuple(batch, h), context)
    # every row its series' mean (`croston_forecast`)
    if h:
        var entries = xseq_entries(fields, context)
        _ = py_call_typed(entries["croston_forecast"], Python.list(xseq_addr(mean), xseq_addr(forecast)), Python.list(batch, h))
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
    if Int(py=y.ndim) != 2 or Int(py=y.shape[1]) < 4:
        raise Error("ETS: each series needs at least 4 observations")
    var spec = String(py=fields["model"])
    var e = _letter(spec, 0)
    var tr = _letter(spec, 1)
    var s = _letter(spec, 2)
    var n = Int(py=y.shape[1])
    var m = xseq_int(fields["season_length"])
    if s != "N" and n <= m:
        s = "N"  # the reference drops the season (ets_f)
    if e == "M" or s == "M":
        # every value > 0: finite (`all_finite_f32`) and a first-wins minimum above zero (`reduce_stat` 0)
        var size = Int(py=y.size)
        var positive = Bool(py=py_call_typed(xseq_base(context, "all_finite_f32"), xseq_addr(y), PythonObject(size)))
        _ = y  # keep the Python object alive through the native call above
        if positive:
            var low = py_call_typed(xseq_base(context, "reduce_stat"), xseq_addr(y), PythonObject(0), PythonObject(size), PythonObject(0))
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
    var y = fields["_y"]
    var batch = Int(py=y.shape[0])
    var n = Int(py=y.shape[1])
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
    var forecast: PythonObject
    var info: PythonObject
    var states: PythonObject
    if last is not None and Bool(py=last[0] == key):
        forecast = xseq_copy(last[1])
        info = xseq_copy(last[2])
        states = xseq_copy(last[3])
    else:
        forecast = xseq_zeros(Python.tuple(batch, h), context)
        info = xseq_zeros(Python.tuple(batch, 10), context)
        states = xseq_zeros(Python.tuple(batch, m), context)
        var entries = xseq_entries(fields, context)
        _ = py_call_typed(entries["ets"], Python.list(xseq_addr(y), xseq_addr(forecast), xseq_addr(info), xseq_addr(states)), ip, fp)
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
    if Int(py=y.ndim) != 2 or Int(py=y.shape[1]) < 10:
        raise Error("GARCH: each series needs at least 10 observations")
    # the finiteness scan (the base binding's `all_finite_f32`)
    if not Bool(py=py_call_typed(xseq_base(context, "all_finite_f32"), xseq_addr(y), PythonObject(Int(py=y.size)))):
        raise Error("GARCH: y holds a non-finite value")
    _ = y  # keep the Python object alive through the native calls above
    var batch = Int(py=y.shape[0])
    var n = Int(py=y.shape[1])
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
    _ = py_call_typed(entries["garch"], Python.list(xseq_addr(y), xseq_addr(params), xseq_addr(info), xseq_addr(sigma),
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


# ---------------------------------------------------------------- VAR

def _var_fit(model: PythonObject, fields: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if args[2] is not None:
        raise Error("NotImplementedError: VAR.fit: information-criterion order selection (ic=) is not implemented")
    if not Bool(py=args[1] == PythonObject("ols")):
        raise Error("VAR.fit: method must be 'ols'")
    var trend = args[3]
    var with_c = Bool(py=trend == PythonObject("c"))
    if not with_c and not Bool(py=trend == PythonObject("n")):
        raise Error("NotImplementedError: VAR.fit: trend=" + String(py=py_call(b.repr, trend))
                    + " is not implemented ('c' and 'n' are)")
    var p = xseq_int(args[0])
    var kt = 1 if with_c else 0
    var endog = fields["_endog"]
    var n = Int(py=endog.shape[0])
    var neqs = Int(py=endog.shape[1])
    var m = kt + neqs * p
    var params = xseq_zeros(Python.tuple(m, neqs), context)
    var sigma = xseq_zeros(Python.tuple(neqs, neqs), context)
    var nobs = max(n - p, 1)
    var resid = xseq_zeros(Python.tuple(nobs, neqs), context)
    var entries = xseq_entries(fields, context)
    var code = Int(py=py_call_typed(entries["var_fit"], Python.list(xseq_addr(endog), xseq_addr(params), xseq_addr(sigma),
                                    xseq_addr(resid)), Python.list(n, neqs, p, kt)))
    _ = sigma  # keep the Python object alive through the native call above
    _ = params  # keep the Python object alive through the native call above
    _ = endog  # keep the Python object alive through the native call above
    _ = resid  # keep the Python object alive through the native call above
    if code:
        raise Error("LinAlgError: VAR.fit: the lagged design is rank deficient (Cholesky pivot of column "
                    + String(code - 1) + ")")
        _ = resid  # keep the Python object alive through the native call above
        _ = sigma  # keep the Python object alive through the native call above
        _ = params  # keep the Python object alive through the native call above
        _ = endog  # keep the Python object alive through the native call above
    _ = resid  # keep the Python object alive through the native calls above
    _ = sigma  # keep the Python object alive through the native calls above
    _ = params  # keep the Python object alive through the native calls above
    _ = endog  # keep the Python object alive through the native calls above
    # VARResults(endog, params, sigma, resid, p, kt, numeric_mode), its
    # attributes set here (the same words its constructor derived)
    var cls = context["xseq_var_results"]
    var result = py_call(_getattr(b.object, "__new__"), cls)
    var state = py_fields(result)
    _store(state, "endog", endog)
    _store(state, "params", params)
    _store(state, "sigma_u", sigma)
    _store(state, "resid", resid)
    _store(state, "k_ar", PythonObject(p))
    _store(state, "k_trend", PythonObject(kt))
    _store(state, "neqs", PythonObject(neqs))
    var used = n - p
    _store(state, "nobs", PythonObject(used))
    _store(state, "df_resid", PythonObject(used - (neqs * p + kt)))
    var intercept = xseq_zeros(Python.tuple(neqs), context)
    var coefs = xseq_zeros(Python.tuple(p, neqs, neqs), context)
    # fittedvalues = (endog[k_ar:] - resid) in float32
    var rows = max(used, 0)
    if rows != nobs and rows != 0:
        raise Error("operands could not be broadcast together")
    var fitted = xseq_zeros(Python.tuple(rows, neqs), context)
    xseq_var_blocks(xseq_f32p(params), xseq_f32p(endog), xseq_f32p(resid), xseq_f32p(intercept), xseq_f32p(coefs),
                    xseq_f32p(fitted), p, kt, neqs, rows)
    _ = coefs  # keep the Python object alive through the native call above
    _ = intercept  # keep the Python object alive through the native call above
    _ = resid  # keep the Python object alive through the native call above
    _ = params  # keep the Python object alive through the native call above
    _ = endog  # keep the Python object alive through the native call above
    _ = fitted  # keep the Python object alive through the native call above
    _store(state, "intercept", intercept)
    _store(state, "coefs", coefs)
    _store(state, "fittedvalues", fitted)
    _store(state, "_numeric_mode", py_get(fields, "numeric_mode"))
    return result


# ---------------------------------------------------------------- STL

def _stl_fit(model: PythonObject, fields: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var robust = xseq_truth(fields["robust"])
    var inner = (2 if robust else 5) if args[0] is None else xseq_int(args[0])
    var outer = (15 if robust else 0) if args[1] is None else xseq_int(args[1])
    var endog = fields["_endog"]
    var batch = Int(py=endog.shape[0])
    var n = Int(py=endog.shape[1])
    var season = xseq_zeros(Python.tuple(batch, n), context)
    var trend = xseq_zeros(Python.tuple(batch, n), context)
    var weights = xseq_zeros(Python.tuple(batch, n), context)
    var resid = xseq_zeros(Python.tuple(batch, n), context)
    var entries = xseq_entries(fields, context)
    var degrees = Int(py=fields["seasonal_deg"]) + 2 * Int(py=fields["trend_deg"]) + 4 * Int(py=fields["low_pass_deg"])
    _ = py_call_typed(entries["stl"], Python.list(xseq_addr(endog), xseq_addr(season), xseq_addr(trend), xseq_addr(weights),
                      xseq_addr(resid)), Python.list(batch, n, fields["period"], fields["seasonal"], fields["trend"],
                      fields["low_pass"], degrees, fields["seasonal_jump"], fields["trend_jump"], fields["low_pass_jump"],
                      inner, outer))
    _ = weights  # keep the Python object alive through the native call above
    _ = trend  # keep the Python object alive through the native call above
    _ = season  # keep the Python object alive through the native call above
    _ = endog  # keep the Python object alive through the native call above
    _ = resid  # keep the Python object alive through the native call above
    var one = xseq_truth(fields["_one"])
    # DecomposeResult(observed, seasonal, trend, resid, weights)
    var result = py_call(_getattr(b.object, "__new__"), context["xseq_decompose_result"])
    var state = py_fields(result)
    _store(state, "observed", xseq_first(endog, one))
    _store(state, "seasonal", xseq_first(season, one))
    _store(state, "trend", xseq_first(trend, one))
    _store(state, "resid", xseq_first(resid, one))
    _store(state, "weights", xseq_first(weights, one))
    return result


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
    var names = Python.tuple("numeric_mode")
    if kind == "theta":
        names = Python.tuple("numeric_mode", "season_length", "decomposition_type", "model", "initial_smoothed", "alpha", "theta")
    elif kind == "ets":
        names = Python.tuple("numeric_mode", "season_length", "model", "damped", "alpha", "beta", "phi", "gamma")
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
    var fields = _forecast_fields(model, kind, context)
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
        return xseq_first(result, xseq_truth(fields["_one"]))
    _setting(args[1], lifecycle)
    _setting(args[5], lifecycle)
    if xseq_truth(args[5]):
        raise Error("NotImplementedError: " + kind + ": fitted values are not implemented")
    # Preserve the public conveniences' accepted exogenous arguments: Theta
    # forwards X to fit; Croston/ETS historically ignore X and X_future.
    var fit_args = Python.tuple(args[0], args[2] if kind == "theta" else PythonObject(None))
    var predict_args = Python.tuple(args[1], PythonObject(None), args[4])
    if kind == "croston":
        _ = _croston_fit(model, fields, fit_args, context)
        return _croston_predict(model, fields, predict_args, context)
    if kind == "ets":
        _ = _ets_fit(model, fields, fit_args, context)
        return _ets_predict(model, fields, predict_args, context)
    _ = _theta_fit(model, fields, fit_args, context)
    return _theta_predict(model, fields, predict_args, context)


def xseq_forecast(model: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    if operation == "croston_initialize" or operation == "ets_initialize" or operation == "theta_initialize" or operation == "garch_initialize":
        var kind = "croston" if operation == "croston_initialize" else ("ets" if operation == "ets_initialize" else ("theta" if operation == "theta_initialize" else "garch"))
        return forecast_initialize(model, kind, args, context)
    if operation == "croston_forecast" or operation == "ets_forecast" or operation == "theta_forecast" or operation == "garch_forecast":
        var kind = "croston" if operation == "croston_forecast" else ("ets" if operation == "ets_forecast" else ("theta" if operation == "theta_forecast" else "garch"))
        return _forecast_operation(model, kind, args, context)
    var fields = py_fields(model)
    if operation == "theta_fit":
        return _theta_fit(model, fields, args, context)
    if operation == "theta_predict":
        return _theta_predict(model, fields, args, context)
    if operation == "croston_fit":
        return _croston_fit(model, fields, args, context)
    if operation == "croston_predict":
        return _croston_predict(model, fields, args, context)
    if operation == "ets_fit":
        return _ets_fit(model, fields, args, context)
    if operation == "ets_predict":
        return _ets_predict(model, fields, args, context)
    if operation == "garch_fit":
        return _garch_fit(model, fields, args, context)
    if operation == "var_fit":
        return _var_fit(model, fields, args, context)
    if operation == "stl_fit":
        return _stl_fit(model, fields, args, context)
    raise Error("RuntimeError: unknown sequence forecaster operation " + operation)
