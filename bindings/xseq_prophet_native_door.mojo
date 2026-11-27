# SPDX-License-Identifier: Apache-2.0
# Complete Prophet fit/predict operation ownership: passive source snapshots,
# native preparation, compiled numerical entries and staged state publication.
# UNPACK/PACK only (lane doorsplit-cnn-xseq, 2026-11-11): the checks, plan
# words and glue calls run in sequence/xseq_prophet_run.mojo (the predict
# setup as one typed RUN over a typed fitted plan, lane bm-rest-graph); this file
# snapshots the fields, admits and owns the buffers, releases the GIL around
# each glue call and publishes the fitted state.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains
from bindings.xseq_native_common_door import xseq_route_of, xseq_addr, xseq_column_i64
from bindings.forecast_device_door import forecast_finite
from bindings.admitted_sequence_door import admit_nd_f32
from core.entry_table import EntryRef
from sequence.xseq_prophet_run import (
    ProphetPredictPlan, ProphetPredictInput, prophet_run_predict_setup, prophet_run_parameter_count,
    prophet_run_check_stored_dtype, prophet_run_check_ranks, prophet_run_check_holidays_given,
    ProphetSeason, ProphetAuto, ProphetTimes, ProphetFitInput, ProphetFitPlan, PROPHET_AUTO_AUTO, PROPHET_AUTO_ON,
    PROPHET_AUTO_OFF, PROPHET_AUTO_ORDER, prophet_run_prepare_fit, prophet_run_prior_count, prophet_run_fit_plan,
    prophet_run_copy_changepoints,
)
from bindings.native_loaded_module_door import native_loaded_module
from bindings.sequential_arrays_door import sequential_array
from bindings.classical_pandas_door import classical_admit_value
from bindings.classical_exceptions_door import classical_require_fitted
from providers.python.prophet_lifecycle_pyapi_unpack import prophet_scalar
from providers.python.prophet_lifecycle_pyapi_pack import prophet_fields
from providers.python.prophet_buffers_pyapi_door import prophet_source, prophet_contiguous, prophet_nanoseconds
from providers.python.forecast_owner_pyapi_unpack import forecast_commit
from bindings.native_owner_lifetime_door import native_release_admit
from bindings.xseq_keys_unpack import xseq_keys
from bindings.py_balanced import py_item

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
    var nk = xseq_keys()
    var seasons = py_item(fields, nk[].k_seasonalities)
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
    py_setitem(fields, nk[].k_seasonalities, captured)
    return fields


def _commit(original: PythonObject, fields: PythonObject, context: PythonObject) raises:
    var b = _b()
    var nk = xseq_keys()
    var seasons = py_item(original, nk[].k_seasonalities)
    var captured = py_item(fields, nk[].k_seasonalities)
    # Admit every replaced owner before the first mutation. Keep both old
    # seasonality dictionaries and their existing records' public identities.
    var names = py_call(b.list, fields)
    for i in range(len(names)):  # driver-loop: preflight public fitted-state replacement
        if names[i] != "seasonalities" and py_get(original, names[i]) is not fields[names[i]]:
            native_release_admit(py_get(original, names[i]), py_item(context, nk[].k_xseq_forecast_ownership))
    var keys = py_call(b.list, captured)
    for i in range(len(keys)):  # driver-loop: publish only automatic seasonality additions
        if not py_contains(seasons, keys[i]):
            py_setitem(seasons, keys[i], captured[keys[i]])
    py_setitem(fields, nk[].k_seasonalities, seasons)
    forecast_commit(original, fields, py_item(context, nk[].k_xseq_forecast_ownership))


def _contig(value: PythonObject, dtype: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = xseq_keys()
    return prophet_contiguous(prophet_source(value, context), dtype is py_item(context, nk[].k_float64), context)


def _times(source: PythonObject, context: PythonObject, mut owner: PythonObject) raises -> ProphetTimes:
    """UNPACK of a time column: float64 values, or int64 nanoseconds of a
    datetime64 column, contiguous (n,); `owner` holds the buffer."""
    var t = prophet_source(source, context)
    var datetime = False
    if String(t.dtype.kind) == "M":
        owner = prophet_nanoseconds(t, context)
        datetime = True
    else:
        owner = py_call(prophet_contiguous(t, True, context).reshape, -1)
    var n = Int(py=owner.size)
    return ProphetTimes(Int(py=xseq_addr(owner)) if n else 0, n, datetime)


def _seasonal_rows(fields: PythonObject) raises -> PythonObject:
    var nk = xseq_keys()
    return py_call(_b().list, py_call(py_item(fields, nk[].k_seasonalities).values))


def _auto_setting(fields: PythonObject, name: String, arg: PythonObject) raises -> ProphetAuto:
    """UNPACK of `yearly_seasonality`, ...: 'auto', True, False or an order.
    A season the model already has is never read (the former `_auto`
    returned before decoding its setting)."""
    var b = _b()
    var nk = xseq_keys()
    if py_contains(py_item(fields, nk[].k_seasonalities), name):
        return ProphetAuto(PROPHET_AUTO_OFF, 0)
    if Bool(py=py_call(b.isinstance, arg, b.str)) and String(py=arg) == "auto":
        return ProphetAuto(PROPHET_AUTO_AUTO, 0)
    if arg is PythonObject(True):
        return ProphetAuto(PROPHET_AUTO_ON, 0)
    if arg is PythonObject(False):
        return ProphetAuto(PROPHET_AUTO_OFF, 0)
    return ProphetAuto(PROPHET_AUTO_ORDER, Int(py=py_call(b.int, arg)))


def _seasons(fields: PythonObject) raises -> List[ProphetSeason]:
    """UNPACK of the snapshotted seasonality records, in dictionary order."""
    var b = _b()
    var nk = xseq_keys()
    var seasons = py_item(fields, nk[].k_seasonalities)
    var keys = py_call(b.list, seasons)
    var result = List[ProphetSeason]()
    for i in range(len(keys)):  # driver-loop: over the model's seasonalities (one record each)
        var spec = seasons[keys[i]]
        result.append(ProphetSeason(String(py=keys[i]), Float64(py=py_call(b.float, py_item(spec, nk[].k_period))),
                                    Int(py=py_call(b.int, py_item(spec, nk[].k_fourier_order))),
                                    Float64(py=py_call(b.float, py_item(spec, nk[].k_prior_scale)))))
    return result^


def _prepare(key: EntryRef, fit_input: ProphetFitInput) raises -> ProphetFitPlan:
    with GILReleased(Python()):
        return prophet_run_prepare_fit(key, fit_input)


def _fit_plan(key: EntryRef, plan: ProphetFitPlan, y: Int, batch: Int, holidays: Int, nh: Int,
              holidays_prior_scale: Float64, params: Int, info: Int, multiplicative: Bool, max_iter: Int,
              changepoint_prior_scale: Float64) raises:
    with GILReleased(Python()):
        prophet_run_fit_plan(key, plan, y, batch, holidays, nh, holidays_prior_scale, params, info, multiplicative,
                             max_iter, changepoint_prior_scale)


def prophet_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """UNPACK the settings and the t, y and holiday buffers into the typed
    fit entry (`prophet_run_prepare_fit`, `prophet_run_fit_plan`), then PACK
    the plan and the fitted buffers into the model (lane py-tail)."""
    var b = _b()
    var original = prophet_fields(model, context)
    var fields = _state(original, context)
    # The model's one route (resolved once); the device is read from the
    # model's own fields on a miss, the mode object is the one `_state` copied.
    var nk = xseq_keys()
    var route = xseq_route_of(model, py_get(fields, nk[].k_numeric_mode), original, context)
    var key = route.glue
    var t_owner = PythonObject(None)
    var t = _times(args[0], context, t_owner)
    var y = prophet_source(args[1], context)
    if Bool(py=y.dtype == py_item(context, nk[].k_float64)):
        # A float64 y is converted once to float32, the dtype the fit
        # computes in, by the shared cast (bindings/admitted_sequence_door.mojo
        # admit_nd_f32: round-to-nearest-even, a finite value outside
        # float32's range refused by its column); the times stay float64.
        # It replaces the fit's "float64 y is refused" (Andrew, 2026-11-27).
        y = admit_nd_f32(y, "y", False, context)
    var rank = Int(py=y.ndim)
    var one = rank == 1
    var Y = PythonObject(None)
    var B = 0
    var N = 0
    var finite = True
    if rank == 1 or rank == 2:
        Y = prophet_contiguous(y, False, context)
        if one:
            Y = py_call(Y.reshape, 1, -1)
        B = Int(py=Y.shape[0])
        N = Int(py=Y.shape[1])
        if N >= 2 and N == t.n:
            # a CPU route beside a GPU set scans on the host; every other
            # route through the base binding's `all_finite_f32`
            finite = forecast_finite(route.cpu_table, xseq_addr(Y), Int(py=Y.size), context)
    var cp_owner = PythonObject(None)
    var has_cp = py_item(fields, nk[].k_changepoints) is not None
    var cp_times = ProphetTimes(0, 0, False)
    var changepoint_range = 0.0
    var n_changepoints = 0
    if has_cp:
        cp_times = _times(py_item(fields, nk[].k_changepoints), context, cp_owner)
    else:
        changepoint_range = Float64(py=py_item(fields, nk[].k_changepoint_range))
        n_changepoints = Int(py=py_item(fields, nk[].k_n_changepoints))
    var fit_input = ProphetFitInput(
        t.copy(), rank, N, finite, _seasons(fields),
        _auto_setting(fields, "yearly", py_item(fields, nk[].k_yearly_seasonality)),
        _auto_setting(fields, "weekly", py_item(fields, nk[].k_weekly_seasonality)),
        _auto_setting(fields, "daily", py_item(fields, nk[].k_daily_seasonality)),
        Float64(py=py_call(b.float, py_item(fields, nk[].k_seasonality_prior_scale))), has_cp, cp_times.copy(), changepoint_range,
        n_changepoints)
    var plan = _prepare(key, fit_input)
    _ = t_owner  # keep the Python objects alive through the native call above
    _ = cp_owner
    _ = Y
    py_setitem(fields, nk[].k__one, one)
    # PACK: the automatic seasonalities this fit added (published by _commit)
    var seasons = py_item(fields, nk[].k_seasonalities)
    for i in range(len(plan.seasons) - plan.n_added, len(plan.seasons)):  # driver-loop: over the automatic seasonality additions (at most three)
        var spec = b.dict()
        py_setitem(spec, nk[].k_period, plan.seasons[i].period)
        py_setitem(spec, nk[].k_fourier_order, plan.seasons[i].fourier_order)
        py_setitem(spec, nk[].k_prior_scale, py_item(fields, nk[].k_seasonality_prior_scale))
        py_setitem(seasons, plan.seasons[i].name, spec)
    py_setitem(fields, nk[].k_start_, plan.start)
    py_setitem(fields, nk[].k_t_scale_, plan.t_scale)
    var cpts = sequential_array(Python.tuple(plan.s), py_item(context, nk[].k_float32), context)
    prophet_run_copy_changepoints(plan, Int(py=xseq_addr(cpts)) if plan.s else 0)
    py_setitem(fields, nk[].k_changepoints_t_, cpts)
    var nh = 0
    var H = sequential_array(Python.tuple(N, 1), py_item(context, nk[].k_float32), context)
    if args[2] is not None:
        var holidays = prophet_source(args[2], context)
        H = py_call(prophet_contiguous(holidays, False, context).reshape, N, -1)
        nh = Int(py=H.shape[1])
    py_setitem(fields, nk[].k__nh, nh)
    var K = prophet_run_prior_count(plan, nh)
    py_setitem(fields, nk[].k__K, K)
    var P = prophet_run_parameter_count(plan.s, K)
    var params = sequential_array(Python.tuple(B, P), py_item(context, nk[].k_float32), context)
    var info_out = sequential_array(Python.tuple(B, 4), py_item(context, nk[].k_float32), context)
    var multiplicative = String(py=py_item(fields, nk[].k_seasonality_mode)) == "multiplicative"
    var holidays_prior_scale = Float64(py=py_call(b.float, py_item(fields, nk[].k_holidays_prior_scale))) if nh > 0 else 0.0
    _fit_plan(key, plan, Int(py=xseq_addr(Y)), B, Int(py=xseq_addr(H)), nh, holidays_prior_scale,
              Int(py=xseq_addr(params)), Int(py=xseq_addr(info_out)), multiplicative, Int(py=py_item(fields, nk[].k_max_iter)),
              Float64(py=py_item(fields, nk[].k_changepoint_prior_scale)))
    _ = H  # keep the Python objects alive through the native call above
    _ = Y
    _ = info_out
    _ = params
    py_setitem(fields, nk[].k_params_, params)
    py_setitem(fields, nk[].k_info_, info_out)
    # diagnostics from info_ (y_scale, objective, iterations, 0): views only
    py_setitem(fields, nk[].k_objective_, info_out.__getitem__(Python.tuple(py_call(b.slice, PythonObject(None)), 1)))
    py_setitem(fields, nk[].k_n_iter_, xseq_column_i64(info_out, B, 4, 2, context))
    # fit bookkeeping of the former `_mode` fit wrapper, committed with the state
    py_setitem(fields, nk[].k__mojolearn_fitted, PythonObject(True))
    _commit(original, fields, context)
    return model


def prophet_predict(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = _b()
    # NotFittedError, as the former `_mode` predict guard raised it
    var nk = xseq_keys()
    classical_require_fitted(model, False, "predict", py_item(context, nk[].k_xseq_lifecycle_context))
    var original = prophet_fields(model, context)
    var fields = _state(original, context)
    if not py_contains(fields, nk[].k_params_) or not py_contains(fields, nk[].k_info_):
        raise Error("NotFittedError: This " + String(py_type(model).__name__)
                    + " instance is not fitted yet. Call 'fit' with appropriate arguments before using predict.")
    var stored = Python.tuple("params_", "info_", "changepoints_t_")
    for i in range(len(stored)):  # driver-loop: native fitted buffers captured once before numerical execution
        var value = prophet_source(fields[stored[i]], context)
        prophet_run_check_stored_dtype(Int(py=value.dtype.num))
        py_setitem(fields, stored[i], prophet_contiguous(value, False, context))
    prophet_run_check_ranks(Int(py=py_item(fields, nk[].k_params_).ndim), Int(py=py_item(fields, nk[].k_info_).ndim),
                            Int(py=py_item(fields, nk[].k_changepoints_t_).ndim))
    var key = xseq_route_of(model, py_get(fields, nk[].k_numeric_mode), original, context).glue
    # UNPACK (lane bm-rest-graph, 2026-11-23): the time column, the holiday
    # matrix and the fitted attributes predict reads, once, into the typed
    # `ProphetPredictPlan` / `ProphetPredictInput`; RUN the predict setup
    # (sequence/xseq_prophet_run.mojo `prophet_run_predict_setup`: the days,
    # periods, orders, the separate feature call of an empty `t`, the schema
    # refusals and the predict entry) with one GIL release (there were three:
    # days, features, predict); PACK the trend and the forecast. The holiday
    # admission now precedes the day conversion, and an empty `t`'s feature
    # refusal (a non-positive seasonal period) follows the holiday admission.
    var t_owner = PythonObject(None)
    var times = _times(args[0], context, t_owner)
    var M = times.n
    var nh = Int(py=py_item(fields, nk[].k__nh))
    var H = PythonObject(None)
    var holiday_cols = 0
    if nh:
        prophet_run_check_holidays_given(nh, args[1] is not None)
        H = py_call(_contig(args[1], py_item(context, nk[].k_float32), context).reshape, M, -1)
        holiday_cols = Int(py=H.shape[1])
    var rows = _seasonal_rows(fields)
    var periods = List[Float64](capacity=len(rows))
    var order_values = List[Float64](capacity=len(rows))
    var orders = List[Int](capacity=len(rows))
    for i in range(len(rows)):  # driver-loop: over the user seasonalities (period and Fourier order each)
        periods.append(Float64(py=py_call(b.float, py_item(rows[i], nk[].k_period))))
        var order = py_item(rows[i], nk[].k_fourier_order)
        order_values.append(Float64(py=py_call(b.float, order)))
        orders.append(Int(py=py_call(b.int, order)))
    var params = py_item(fields, nk[].k_params_)
    var info = py_item(fields, nk[].k_info_)
    var cpts = py_item(fields, nk[].k_changepoints_t_)
    var B = Int(py=params.shape[0])
    # `_K` is read only when `nh` is not negative (the schema check refuses
    # a negative `nh` before any read of it, as before the split).
    var k = Int(py=py_item(fields, nk[].k__K)) if nh >= 0 else 0
    var plan = ProphetPredictPlan(
        periods^, order_values^, orders^, Float64(py=py_call(b.float, py_item(fields, nk[].k_start_))),
        Float64(py=py_call(b.float, py_item(fields, nk[].k_t_scale_))), nh, k,
        String(py=py_item(fields, nk[].k_seasonality_mode)) == "multiplicative", Int(py=xseq_addr(params)), B,
        Int(py=params.shape[1]), Int(py=xseq_addr(info)), Int(py=info.shape[0]), Int(py=info.shape[1]),
        Int(py=xseq_addr(cpts)), len(cpts))
    var yhat = sequential_array(Python.tuple(B, M), py_item(context, nk[].k_float32), context)
    var trend = sequential_array(Python.tuple(B, M), py_item(context, nk[].k_float32), context)
    var request = ProphetPredictInput(times.copy(), Int(py=xseq_addr(H)) if nh else 0, holiday_cols,
                                      Int(py=xseq_addr(yhat)), Int(py=xseq_addr(trend)))
    var py = Python()
    with GILReleased(py):
        prophet_run_predict_setup(key, plan, request)
    _ = t_owner  # keep the Python objects alive through the native call above
    _ = H
    _ = params
    _ = info
    _ = cpts
    _ = trend
    _ = yhat
    _ = fields
    var one = Bool(py=py_item(fields, nk[].k__one))
    py_setitem(fields, nk[].k_trend_, trend[0] if one else trend)
    # Snapshot copies are private to this prediction; preserve the fitted
    # parameter owners when publishing only the newly computed trend.
    var updates = Python.dict()
    py_setitem(updates, nk[].k_trend_, py_item(fields, nk[].k_trend_))
    forecast_commit(original, updates, py_item(context, nk[].k_xseq_forecast_ownership))
    return yhat[0] if one else yhat
