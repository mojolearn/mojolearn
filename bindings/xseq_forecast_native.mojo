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
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.xseq_native_common import (
    XF32P, XI64P, xseq_entries, xseq_base, xseq_input, xseq_f32, xseq_refuse_f64, xseq_zeros, xseq_addr, xseq_f32p,
    xseq_reshape, xseq_copy, xseq_item, xseq_first, xseq_column_f32, xseq_column_i64, xseq_mean_dict, xseq_int,
    xseq_float, xseq_truth,
)
from core.xseq_marshal import xseq_var_blocks


def _getattr(owner: PythonObject, name: String) raises -> PythonObject:
    var b = Python.import_module("builtins")
    return py_call(b.getattr, owner, PythonObject(name))


def _series(source: PythonObject, label: String, copy: Bool, context: PythonObject,
            message: String = "float64 is refused; pass float32") raises -> PythonObject:
    """`(one, Y)`: `y = np.asarray(source)`, float64 refused, and `Y` the
    C-ordered float32 (batch, n_obs) view of it (one series reshaped to one
    row); `copy` keeps a private copy."""
    var y = xseq_input(source, context)
    xseq_refuse_f64(y, label + ": " + message)
    var one = Int(py=y.ndim) == 1
    var arr = xseq_f32(y, label, copy, context)
    if one:
        arr = xseq_reshape(arr, Python.tuple(1, -1))
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
    var b = Python.import_module("builtins")
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
                      Python.list(batch, n, _getattr(py_type(model), "_VARIANT")))
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
        if positive:
            var low = py_call_typed(xseq_base(context, "reduce_stat"), xseq_addr(y), PythonObject(0), PythonObject(size), PythonObject(0))
            positive = Bool(py=low > PythonObject(0))
        if not positive:
            raise Error("ETS: Inappropriate model for data with negative or zero values")
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
    var b = Python.import_module("builtins")
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
        _store(fields, "_last", Python.tuple(key, xseq_copy(forecast), xseq_copy(info), xseq_copy(states)))
    _store(fields, "info_", info)
    var one = xseq_truth(fields["_one"])
    if s != "N":
        _store(fields, "seasonal_states_", xseq_first(states, one))
    return xseq_mean_dict(xseq_first(forecast, one))


# ---------------------------------------------------------------- GARCH

def _garch_fit(model: PythonObject, fields: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var series = _series(args[0], "GARCH", False, context)
    var one = Bool(py=series[0])
    _store(fields, "_one", series[0])
    var y = series[1]
    if Int(py=y.ndim) != 2 or Int(py=y.shape[1]) < 10:
        raise Error("GARCH: each series needs at least 10 observations")
    # the finiteness scan (the base binding's `all_finite_f32`)
    if not Bool(py=py_call_typed(xseq_base(context, "all_finite_f32"), xseq_addr(y), PythonObject(Int(py=y.size)))):
        raise Error("GARCH: y holds a non-finite value")
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
    _store(fields, "_y", y)
    _store(fields, "params_", xseq_first(params, one))
    _store(fields, "loglikelihood_", xseq_first(xseq_column_f32(info, batch, 4, 0, context), one))
    _store(fields, "n_iter_", xseq_first(xseq_column_i64(info, batch, 4, 1, context), one))  # Nelder-Mead iterations, both runs
    _store(fields, "conditional_volatility_", xseq_first(sigma, one))
    _store(fields, "_fc", forecast)
    return model


# ---------------------------------------------------------------- VAR

def _var_fit(model: PythonObject, fields: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
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
    if code:
        raise Error("LinAlgError: VAR.fit: the lagged design is rank deficient (Cholesky pivot of column "
                    + String(code - 1) + ")")
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
    _store(state, "intercept", intercept)
    _store(state, "coefs", coefs)
    _store(state, "fittedvalues", fitted)
    _store(state, "_numeric_mode", py_get(fields, "numeric_mode"))
    return result


# ---------------------------------------------------------------- STL

def _stl_fit(model: PythonObject, fields: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
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


def xseq_forecast(model: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
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
