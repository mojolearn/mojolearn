# SPDX-License-Identifier: Apache-2.0
# Complete Prophet fit/predict operation ownership: passive source snapshots,
# native preparation, compiled numerical entries and staged state publication.
from std.math import floor
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.xseq_native_common import xseq_entries, xseq_base, xseq_addr, xseq_column_i64, xseq_copy
from bindings.native_loaded_module import native_loaded_module
from bindings.sequential_arrays import sequential_array
from bindings.classical_pandas import classical_admit_value
from bindings.classical_exceptions import classical_require_fitted
from providers.python.prophet_lifecycle_pyapi import prophet_fields, prophet_scalar
from providers.python.prophet_buffers_pyapi import prophet_source, prophet_contiguous, prophet_nanoseconds
from providers.python.forecast_owner_pyapi import forecast_commit
from bindings.native_owner_lifetime import native_release_admit

comptime _P = "ProphetForecaster: "


def _b() raises -> PythonObject:
    return native_loaded_module("builtins")


def _state(original: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = _b()
    var fields = py_call(b.dict, original)
    var scalars = Python.tuple("growth", "n_changepoints", "changepoint_range", "yearly_seasonality",
        "weekly_seasonality", "daily_seasonality", "seasonality_mode", "seasonality_prior_scale",
        "holidays_prior_scale", "changepoint_prior_scale", "max_iter", "numeric_mode", "start_", "t_scale_",
        "_nh", "_K", "_one")
    for i in range(len(scalars)):  # driver-loop: model scalar settings before native comparisons/conversions
        if py_contains(fields, scalars[i]):
            prophet_scalar(fields[scalars[i]], context)
    var seasons = fields["seasonalities"]
    if py_type(seasons) is not b.dict:
        raise Error("TypeError: seasonalities require a native dictionary")
    var keys = py_call(b.list, seasons)
    var captured = Python.dict()
    for i in range(len(keys)):  # driver-loop: snapshot each native seasonality record once
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: seasonality names require native strings")
        var spec = seasons[keys[i]]
        if py_type(spec) is not b.dict:
            raise Error("TypeError: seasonalities require native dictionary records")
        var names = py_call(b.list, spec)
        for j in range(len(names)):  # driver-loop: exact keys before native lookup
            if py_type(names[j]) is not b.str:
                raise Error("TypeError: seasonality fields require native strings")
        var row = Python.dict()
        var required = Python.tuple("period", "fourier_order", "prior_scale")
        for j in range(len(required)):  # driver-loop: three passive scalar values retained for this operation
            var value = spec[required[j]]
            prophet_scalar(value, context)
            py_setitem(row, required[j], value)
        py_setitem(captured, keys[i], row)
    py_setitem(fields, "seasonalities", captured)
    return fields


def _commit(original: PythonObject, fields: PythonObject, context: PythonObject) raises:
    var b = _b()
    var seasons = original["seasonalities"]
    var captured = fields["seasonalities"]
    # Admit every replaced owner before the first mutation. Keep both old
    # seasonality dictionaries and their existing records' public identities.
    var names = py_call(b.list, fields)
    for i in range(len(names)):  # driver-loop: preflight public fitted-state replacement
        if names[i] != "seasonalities" and py_get(original, names[i]) is not fields[names[i]]:
            native_release_admit(py_get(original, names[i]), context["xseq_forecast_ownership"])
    var keys = py_call(b.list, captured)
    for i in range(len(keys)):  # driver-loop: publish only automatic seasonality additions
        if not py_contains(seasons, keys[i]):
            py_setitem(seasons, keys[i], captured[keys[i]])
    py_setitem(fields, "seasonalities", seasons)
    forecast_commit(original, fields, context["xseq_forecast_ownership"])


def _contig(value: PythonObject, dtype: PythonObject, context: PythonObject) raises -> PythonObject:
    return prophet_contiguous(prophet_source(value, context), dtype is context["float64"], context)


def _days(source: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_days(t, mode)`: (days float64 (N,), info [status, span, least
    spacing]), converted and checked by `prophet_days`."""
    var t = prophet_source(source, context)
    var src = PythonObject(None)
    var kind = 0
    if String(t.dtype.kind) == "M":
        src = prophet_nanoseconds(t, context)
        kind = 1
    else:
        src = py_call(prophet_contiguous(t, True, context).reshape, -1)
    var n = Int(py=src.size)
    var days = sequential_array(Python.tuple(n), context["float64"], context)
    var info = sequential_array(Python.tuple(3), context["float64"], context)
    if n:
        _ = py_call_typed(entries["prophet_days"], Python.list(xseq_addr(src), xseq_addr(days), xseq_addr(info)),
                          Python.list(n, kind))
        _ = info  # keep the Python object alive through the native call above
        _ = days  # keep the Python object alive through the native call above
        _ = src  # keep the Python object alive through the native call above
    return Python.tuple(days, info)


def _seasonal_rows(fields: PythonObject) raises -> PythonObject:
    return py_call(_b().list, py_call(fields["seasonalities"].values))


def _features(fields: PythonObject, days: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_features(days)`: (t scaled float32 (N,), frac float32 (N, max(ns, 1)))."""
    var rows = _seasonal_rows(fields)
    var periods = Python.list()
    for i in range(len(rows)):  # driver-loop: over the user seasonalities (one period each)
        _ = py_call(periods.append, rows[i]["period"])
    if len(periods) == 0:
        _ = py_call(periods.append, 1.0)
    var periods_a = _contig(periods, context["float64"], context)
    var ns = len(rows)
    var n = len(days)
    var tsc = sequential_array(Python.tuple(n), context["float32"], context)
    var frac = sequential_array(Python.tuple(n, max(ns, 1)), context["float32"], context)
    _ = py_call_typed(entries["prophet_features"],
                      Python.list(xseq_addr(days), xseq_addr(periods_a), xseq_addr(tsc), xseq_addr(frac)),
                      Python.list(n, ns),
                      Python.list(py_call(_b().float, fields["start_"]), py_call(_b().float, fields["t_scale_"])))
    _ = frac  # keep the Python object alive through the native call above
    _ = tsc  # keep the Python object alive through the native call above
    _ = periods_a  # keep the Python object alive through the native call above
    return Python.tuple(tsc, frac)


def _auto(fields: PythonObject, name: String, arg: PythonObject, default_order: Int, ok: Bool) raises:
    """`_auto(name, arg, default_order, ok)`."""
    var b = _b()
    var seasons = fields["seasonalities"]
    if py_contains(seasons, name):
        return
    var use: Bool
    var order: Int
    if Bool(py=py_call(b.isinstance, arg, b.str)) and String(py=arg) == "auto":
        use = ok
        order = default_order
    elif arg is PythonObject(True):
        use = True
        order = default_order
    elif arg is PythonObject(False):
        use = False
        order = 0
    else:
        use = True
        order = Int(py=py_call(b.int, arg))
    if use:
        var period = 365.25 if name == "yearly" else (7.0 if name == "weekly" else 1.0)
        var spec = b.dict()
        py_setitem(spec, "period", period)
        py_setitem(spec, "fourier_order", order)
        py_setitem(spec, "prior_scale", fields["seasonality_prior_scale"])
        py_setitem(seasons, name, spec)


def _orders(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    """The seasonal Fourier orders as float32 (one placeholder when there are
    none): (orders_, count)."""
    var rows = _seasonal_rows(fields)
    var orders = Python.list()
    for i in range(len(rows)):  # driver-loop: over the user seasonalities (one order each)
        _ = py_call(orders.append, rows[i]["fourier_order"])
    if len(orders) == 0:
        return Python.tuple(sequential_array(Python.tuple(1), context["float32"], context), 0)
    return Python.tuple(_contig(orders, context["float32"], context), len(orders))


def prophet_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = _b()
    var original = prophet_fields(model, context)
    var fields = _state(original, context)
    var entries = xseq_entries(fields, context)
    var dd = _days(args[0], entries, context)
    var days = dd[0]
    var info = dd[1]
    var status = Int(py=py_call(b.int, info[0]))
    var span = Float64(py=py_call(b.float, info[1]))
    var dmin = Float64(py=py_call(b.float, info[2]))
    if status == 1:
        raise Error("ValueError: " + _P + "t must not hold NaN")
    if status == 2:
        raise Error("ValueError: " + _P + "t must be sorted ascending")
    var y = prophet_source(args[1], context)
    if Bool(py=y.dtype == context["float64"]):
        raise Error("TypeError: " + _P + "float64 y is refused; pass float32")
    var one = Int(py=y.ndim) == 1
    if Int(py=y.ndim) != 1 and Int(py=y.ndim) != 2:
        raise Error("ValueError: ProphetForecaster: y must be a series or a batch of series")
    py_setitem(fields, "_one", one)
    var Y = prophet_contiguous(y, False, context)
    if one:
        Y = py_call(Y.reshape, 1, -1)
    var B = Int(py=Y.shape[0])
    var N = Int(py=Y.shape[1])
    if N != len(days) or N < 2:
        raise Error("ValueError: " + _P + "y and t lengths differ, or fewer than 2 points")
    if Int(py=py_call(xseq_base(context, "all_finite_f32"), xseq_addr(Y), Y.size)) != 1:
        raise Error("ValueError: " + _P + "y must be finite")
    _ = Y  # keep the Python object alive through the native calls above
    _auto(fields, "yearly", fields["yearly_seasonality"], 10, span >= 730)
    _auto(fields, "weekly", fields["weekly_seasonality"], 3, span >= 14 and dmin < 7)
    _auto(fields, "daily", fields["daily_seasonality"], 4, span >= 2 and dmin < 1)
    py_setitem(fields, "start_", py_call(b.float, days[0]))
    py_setitem(fields, "t_scale_", span if span > 0 else 1.0)
    var ft = _features(fields, days, entries, context)
    var user = PythonObject(None)
    var n_user: Int
    var hist: Int
    var n_cp: Int
    if fields["changepoints"] is not None:
        user = _days(fields["changepoints"], entries, context)[0]
        n_user = len(user)
        hist = 0
        n_cp = 0
    else:
        user = sequential_array(Python.tuple(1), context["float64"], context)
        n_user = -1
        # floor(N * changepoint_range): one IEEE product, floored exactly
        hist = Int(floor(Float64(N) * Float64(py=fields["changepoint_range"])))
        n_cp = max(min(Int(py=fields["n_changepoints"]), hist - 1), 0)
    var cpt = sequential_array(Python.tuple(max(n_user, max(n_cp, 1))), context["float32"], context)
    var S = Int(py=py_call_typed(entries["prophet_changepoints"],  # cpu-route: changepoint plan over a handful of changepoints
                                 Python.list(xseq_addr(days), xseq_addr(user), xseq_addr(cpt)),
                                 Python.list(N, n_user, hist, n_cp),
                                 Python.list(fields["start_"], fields["t_scale_"])))
    _ = cpt  # keep the Python object alive through the native call above
    _ = user  # keep the Python object alive through the native call above
    _ = days  # keep the Python object alive through the native call above
    if S < 0:
        raise Error("ValueError: Changepoints must fall within training data.")
    _ = cpt  # keep the Python object alive through the native calls above
    _ = user  # keep the Python object alive through the native calls above
    _ = days  # keep the Python object alive through the native calls above
    var cpts = xseq_copy(cpt.__getitem__(py_call(b.slice, PythonObject(None), S)))
    _ = user  # keep the Python object alive through the native call above
    _ = days  # keep the Python object alive through the native call above
    py_setitem(fields, "changepoints_t_", cpts)
    var nh = 0
    var H = sequential_array(Python.tuple(N, 1), context["float32"], context)
    if args[2] is not None:
        var holidays = prophet_source(args[2], context)
        H = py_call(prophet_contiguous(holidays, False, context).reshape, N, -1)
        nh = Int(py=H.shape[1])
    py_setitem(fields, "_nh", nh)
    var od = _orders(fields, context)
    # the prior scale of every Fourier parameter, then of every holiday column
    var sig = Python.list()
    var rows = _seasonal_rows(fields)
    for i in range(len(rows)):  # driver-loop: over the user seasonalities
        for _ in range(2 * Int(py=rows[i]["fourier_order"])):  # driver-loop: over one seasonality's Fourier parameters (model structure)
            _ = py_call(sig.append, rows[i]["prior_scale"])
    for _ in range(nh):  # driver-loop: over the holiday columns (model structure)
        _ = py_call(sig.append, fields["holidays_prior_scale"])
    var K = len(sig)
    py_setitem(fields, "_K", K)
    if K == 0:
        _ = py_call(sig.append, 1.0)
    var sig_a = _contig(sig, context["float32"], context)
    var S2 = len(cpts)
    var P = 3 + S2 + K
    var params = sequential_array(Python.tuple(B, P), context["float32"], context)
    var info_out = sequential_array(Python.tuple(B, 4), context["float32"], context)
    var multiplicative = 1 if String(py=fields["seasonality_mode"]) == "multiplicative" else 0
    _ = py_call_typed(entries["prophet_fit"],
                      Python.list(xseq_addr(Y), xseq_addr(ft[0]), xseq_addr(ft[1]), xseq_addr(od[0]), xseq_addr(H),
                                  xseq_addr(cpts), xseq_addr(sig_a), xseq_addr(params), xseq_addr(info_out)),
                      Python.list(B, N, od[1], nh, K, S2, multiplicative, fields["max_iter"]),
                      Python.list(fields["changepoint_prior_scale"]))
    # Raw addresses do not retain the arrays owned by these two tuples.
    # Both feature buffers and Fourier orders must survive the native call.
    _ = ft
    _ = od
    _ = H  # keep the Python object alive through the native call above
    _ = Y  # keep the Python object alive through the native call above
    _ = info_out  # keep the Python object alive through the native call above
    _ = params  # keep the Python object alive through the native call above
    _ = sig_a  # keep the Python object alive through the native call above
    _ = cpts  # keep the Python object alive through the native call above
    py_setitem(fields, "params_", params)
    py_setitem(fields, "info_", info_out)
    # diagnostics from info_ (y_scale, objective, iterations, 0): views only
    py_setitem(fields, "objective_", info_out.__getitem__(Python.tuple(py_call(b.slice, PythonObject(None)), 1)))
    py_setitem(fields, "n_iter_", xseq_column_i64(info_out, B, 4, 2, context))
    # fit bookkeeping of the former `_mode` fit wrapper, committed with the state
    py_setitem(fields, "_mojolearn_fitted", PythonObject(True))
    _commit(original, fields, context)
    return model


def prophet_predict(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = _b()
    # NotFittedError, as the former `_mode` predict guard raised it
    classical_require_fitted(model, False, "predict", context["xseq_lifecycle_context"])
    var original = prophet_fields(model, context)
    var fields = _state(original, context)
    if not py_contains(fields, "params_") or not py_contains(fields, "info_"):
        raise Error("NotFittedError: This " + String(py_type(model).__name__)
                    + " instance is not fitted yet. Call 'fit' with appropriate arguments before using predict.")
    var stored = Python.tuple("params_", "info_", "changepoints_t_")
    for i in range(len(stored)):  # driver-loop: native fitted buffers captured once before numerical execution
        var value = prophet_source(fields[stored[i]], context)
        if Int(py=value.dtype.num) != 11:
            raise Error("TypeError: ProphetForecaster fitted buffers require float32 storage")
        py_setitem(fields, stored[i], prophet_contiguous(value, False, context))
    if Int(py=fields["params_"].ndim) != 2 or Int(py=fields["info_"].ndim) != 2 or Int(py=fields["changepoints_t_"].ndim) != 1:
        raise Error("ValueError: ProphetForecaster fitted buffer ranks disagree")
    var entries = xseq_entries(fields, context)
    var days = _days(args[0], entries, context)[0]
    var M = len(days)
    var ft = _features(fields, days, entries, context)
    var nh = Int(py=fields["_nh"])
    var H = PythonObject(None)
    if nh:
        if args[1] is None:
            raise Error("ValueError: " + _P + "the model has holiday columns; pass them for t")
        H = py_call(_contig(args[1], context["float32"], context).reshape, M, -1)
    else:
        H = sequential_array(Python.tuple(M, 1), context["float32"], context)
    var od = _orders(fields, context)
    var feature_count = 0
    var rows = _seasonal_rows(fields)
    for i in range(len(rows)):  # driver-loop: fitted feature-schema bounds before the native feature writer
        var order = Int(py=py_call(b.int, rows[i]["fourier_order"]))
        if order < 0:
            raise Error("ValueError: ProphetForecaster Fourier orders must be nonnegative")
        feature_count += 2 * order
    if nh < 0 or Int(py=fields["_K"]) < 0 or feature_count + nh != Int(py=fields["_K"]):
        raise Error("ValueError: ProphetForecaster seasonalities changed after fit; fit the model again")
    var params = fields["params_"]
    var B = Int(py=params.shape[0])
    if Int(py=fields["info_"].shape[0]) != B or Int(py=fields["info_"].shape[1]) != 4:
        raise Error("ValueError: ProphetForecaster fitted diagnostics shape disagrees")
    if Int(py=params.shape[1]) != 3 + len(fields["changepoints_t_"]) + Int(py=fields["_K"]):
        raise Error("ValueError: ProphetForecaster fitted parameter shape disagrees")
    if nh and Int(py=H.shape[1]) != nh:
        raise Error("ValueError: ProphetForecaster holiday column count differs from fit")
    var yhat = sequential_array(Python.tuple(B, M), context["float32"], context)
    var trend = sequential_array(Python.tuple(B, M), context["float32"], context)
    var cpts = fields["changepoints_t_"]
    var multiplicative = 1 if String(py=fields["seasonality_mode"]) == "multiplicative" else 0
    _ = py_call_typed(entries["prophet_predict"],
                      Python.list(xseq_addr(params), xseq_addr(fields["info_"]), xseq_addr(ft[0]), xseq_addr(ft[1]),
                                  xseq_addr(od[0]), xseq_addr(H), xseq_addr(cpts), xseq_addr(yhat), xseq_addr(trend)),
                      Python.list(B, M, od[1], nh, Int(py=fields["_K"]), len(cpts), multiplicative))
    # Prediction has independently allocated feature and order owners too.
    _ = ft
    _ = od
    _ = params  # keep the Python object alive through the native call above
    _ = cpts  # keep the Python object alive through the native call above
    _ = trend  # keep the Python object alive through the native call above
    _ = yhat  # keep the Python object alive through the native call above
    _ = H  # keep the Python object alive through the native call above
    var one = Bool(py=fields["_one"])
    py_setitem(fields, "trend_", trend[0] if one else trend)
    # Snapshot copies are private to this prediction; preserve the fitted
    # parameter owners when publishing only the newly computed trend.
    var updates = Python.dict()
    py_setitem(updates, "trend_", fields["trend_"])
    forecast_commit(original, updates, context["xseq_forecast_ownership"])
    return yhat[0] if one else yhat
