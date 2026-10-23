# SPDX-License-Identifier: Apache-2.0
"""`ProphetForecaster.fit` and `ProphetForecaster.predict`
(python/mojolearn/_x_sequence_prophet.py) as one native call each, through
`xseq_native_call` (bindings/xseq_native.mojo):

    fit      the time axis converted and checked (`prophet_days`), y admitted
             (float64 refused, float32 C order, finite), the automatic
             seasonalities, the features (`prophet_features`), the
             changepoint plan (`prophet_changepoints`), the prior scales and
             one `prophet_fit`, then the fitted fields
    predict  the time axis and features, the holiday columns, one
             `prophet_predict`, `trend_` and the forecast

The same compiled entries with the same argument words, so no bit moves.
The numpy calls here are input conversion only (`asarray`, `astype` to
`datetime64[ns]` / float32 / float64, `reshape`); every computed number
comes from the compiled entries.
"""
from std.math import floor
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.xseq_native_common import xseq_entries, xseq_base, xseq_addr

comptime _P = "ProphetForecaster: "


def _b() raises -> PythonObject:
    return Python.import_module("builtins")


def _np() raises -> PythonObject:
    return Python.import_module("numpy")


def _contig(value: PythonObject, dtype: PythonObject) raises -> PythonObject:
    """`np.ascontiguousarray(value, dtype=dtype)`."""
    return py_call(_np().ascontiguousarray, value, dtype=dtype)


def _days(source: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_days(t, mode)`: (days float64 (N,), info [status, span, least
    spacing]), converted and checked by `prophet_days`."""
    var np = _np()
    var t = py_call(np.asarray, source)
    var src = PythonObject(None)
    var kind = 0
    if Bool(py=py_call(np.issubdtype, t.dtype, np.datetime64)):
        # numpy datetime64 crosses as int64 nanoseconds since 1970
        src = py_call(np.ascontiguousarray, py_call(py_call(t.astype, "datetime64[ns]").view, np.int64))
        kind = 1
    else:
        src = py_call(_contig(t, context["float64"]).reshape, -1)
    src = py_call(src.reshape, -1)
    var n = len(src)
    var days = py_call(np.empty, n, dtype=context["float64"])
    var info = py_call(np.zeros, 3, dtype=context["float64"])
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
    var np = _np()
    var rows = _seasonal_rows(fields)
    var periods = Python.list()
    for i in range(len(rows)):  # driver-loop: over the user seasonalities (one period each)
        _ = py_call(periods.append, rows[i]["period"])
    if len(periods) == 0:
        _ = py_call(periods.append, 1.0)
    var periods_a = _contig(periods, context["float64"])
    var ns = len(rows)
    var n = len(days)
    var tsc = py_call(np.empty, n, dtype=context["float32"])
    var frac = py_call(np.empty, Python.tuple(n, max(ns, 1)), dtype=context["float32"])
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
        return Python.tuple(py_call(_np().zeros, 1, dtype=context["float32"]), 0)
    return Python.tuple(_contig(orders, context["float32"]), len(orders))


def prophet_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var np = _np()
    var b = _b()
    var fields = py_fields(model)
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
    var y = py_call(np.asarray, args[1])
    if Bool(py=y.dtype == context["float64"]):
        raise Error("TypeError: " + _P + "float64 y is refused; pass float32")
    var one = Int(py=y.ndim) == 1
    py_setitem(fields, "_one", one)
    var Y = _contig(y, context["float32"])
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
        user = py_call(np.zeros, 1, dtype=context["float64"])
        n_user = -1
        # floor(N * changepoint_range): one IEEE product, floored exactly
        hist = Int(floor(Float64(N) * Float64(py=fields["changepoint_range"])))
        n_cp = max(min(Int(py=fields["n_changepoints"]), hist - 1), 0)
    var cpt = py_call(np.zeros, max(n_user, max(n_cp, 1)), dtype=context["float32"])
    var S = Int(py=py_call_typed(entries["prophet_changepoints"],  # cpu-route: changepoint plan over a handful of changepoints
                                 Python.list(xseq_addr(days), xseq_addr(user), xseq_addr(cpt)),
                                 Python.list(N, n_user, hist, n_cp),
                                 Python.list(fields["start_"], fields["t_scale_"])))
    _ = cpt  # keep the Python object alive through the native call above
    _ = user  # keep the Python object alive through the native call above
    _ = days  # keep the Python object alive through the native call above
    if S < 0:
        raise Error("ValueError: Changepoints must fall within training data.")
    var cpts = py_call(np.ascontiguousarray, cpt.__getitem__(py_call(b.slice, PythonObject(None), S)))
    py_setitem(fields, "changepoints_t_", cpts)
    var holidays = args[2]
    var nh = 0
    if holidays is not None:
        nh = Int(py=py_call(py_call(np.asarray, holidays).reshape, N, -1).shape[1])
    py_setitem(fields, "_nh", nh)
    var H = PythonObject(None)
    if nh:
        H = py_call(np.ascontiguousarray, py_call(py_call(np.asarray, holidays, dtype=context["float32"]).reshape, N, -1))
    else:
        H = py_call(np.zeros, Python.tuple(N, 1), context["float32"])
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
    var sig_a = _contig(sig, context["float32"])
    var S2 = len(cpts)
    var P = 3 + S2 + K
    var params = py_call(np.zeros, Python.tuple(B, P), dtype=context["float32"])
    var info_out = py_call(np.zeros, Python.tuple(B, 4), dtype=context["float32"])
    var multiplicative = 1 if String(py=fields["seasonality_mode"]) == "multiplicative" else 0
    _ = py_call_typed(entries["prophet_fit"],
                      Python.list(xseq_addr(Y), xseq_addr(ft[0]), xseq_addr(ft[1]), xseq_addr(od[0]), xseq_addr(H),
                                  xseq_addr(cpts), xseq_addr(sig_a), xseq_addr(params), xseq_addr(info_out)),
                      Python.list(B, N, od[1], nh, K, S2, multiplicative, fields["max_iter"]),
                      Python.list(fields["changepoint_prior_scale"]))
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
    py_setitem(fields, "n_iter_", py_call(info_out.__getitem__(Python.tuple(py_call(b.slice, PythonObject(None)), 2)).astype,
                                          context["int64"]))
    return model


def prophet_predict(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var np = _np()
    var b = _b()
    var fields = py_fields(model)
    var entries = xseq_entries(fields, context)
    var days = _days(args[0], entries, context)[0]
    var M = len(days)
    var ft = _features(fields, days, entries, context)
    var nh = Int(py=fields["_nh"])
    var H = PythonObject(None)
    if nh:
        if args[1] is None:
            raise Error("ValueError: " + _P + "the model has holiday columns; pass them for t")
        H = py_call(np.ascontiguousarray, py_call(py_call(np.asarray, args[1], dtype=context["float32"]).reshape, M, -1))
    else:
        H = py_call(np.zeros, Python.tuple(M, 1), context["float32"])
    var od = _orders(fields, context)
    var params = fields["params_"]
    var B = Int(py=params.shape[0])
    var yhat = py_call(np.zeros, Python.tuple(B, M), dtype=context["float32"])
    var trend = py_call(np.zeros, Python.tuple(B, M), dtype=context["float32"])
    var cpts = fields["changepoints_t_"]
    var multiplicative = 1 if String(py=fields["seasonality_mode"]) == "multiplicative" else 0
    _ = py_call_typed(entries["prophet_predict"],
                      Python.list(xseq_addr(params), xseq_addr(fields["info_"]), xseq_addr(ft[0]), xseq_addr(ft[1]),
                                  xseq_addr(od[0]), xseq_addr(H), xseq_addr(cpts), xseq_addr(yhat), xseq_addr(trend)),
                      Python.list(B, M, od[1], nh, Int(py=fields["_K"]), len(cpts), multiplicative))
    _ = params  # keep the Python object alive through the native call above
    _ = cpts  # keep the Python object alive through the native call above
    _ = trend  # keep the Python object alive through the native call above
    _ = yhat  # keep the Python object alive through the native call above
    _ = H  # keep the Python object alive through the native call above
    var one = Bool(py=fields["_one"])
    py_setitem(fields, "trend_", trend[0] if one else trend)
    return yhat[0] if one else yhat
