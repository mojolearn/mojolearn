# SPDX-License-Identifier: Apache-2.0
"""The native owner of `mojolearn.parallel_forecasting`
(python/mojolearn/parallel_forecasting.py): `predict_arima`,
`forecast_arima`, `predict_exponential_smoothing` and
`forecast_exponential_smoothing` as one call each into
`parallel_forecast_call`, on the native local process pool
(bindings/native_process_pool_door.mojo).

The partition is the Python driver's: whole series in `series_per_shard`
ranges; each shard's series-major state (ARIMA's `_y` and `params_` rows,
the Holt-Winters packed components' columns) is copied into a plain state
record, the worker rebuilds the shard's model natively and runs the
native `predict` (bindings/tsa_arima_native.mojo, tsa_hw_native.mojo), and
each shard's block is placed at its own offset in range order. The same
compiled entries on the same words, so no bit moves.

UNPACK and PACK only: the span, step, index and mode refusals, the shard
bounds and the block copies are typed, in
bindings/parallel_forecast_native_run.mojo and
bindings/parallel_shard_common_run.mojo. The worker rebuilds the member and
calls its native fit / predict (`arima_predict`, `hw_predict`, `arima_fit`,
`hw_fit`: boundary calls of the TSA family) until that family publishes an
entry table; the frames are the codec's.
"""
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains
from bindings.msel_member_device_unpack import member_says_cpu, member_multi_gpu_refusal
from bindings.py_attr import py_attr
from bindings.classical_arrays_door import classical_address
from bindings.native_process_pool_door import (
    process_pool_create, process_pool_map, process_pool_close,
    process_worker_connect, process_worker_next, process_send,
)
from bindings.native_transport_door import transport_close
from bindings.tsa_hw_native_door import tsa_array, hw_predict, hw_fit, _check_dims, _set_fitted, _commit
from bindings.tsa_admission_door import tsa_hw_fields
from bindings.tsa_arima_native_door import arima_predict, arima_fit, _series_major
from bindings.parallel_shard_common_door import (
    shard_ranges,
    shard_devices,
    shard_run,
    shard_rows,
    shard_concat,
    shard_state,
    shard_model,
    shard_array,
    shard_identity,
)
from bindings.parallel_shard_common_run import (
    shard_read_shift, shard_width_admit, shard_bounds, shard_indices_admit, shard_block_copy, SHARD_DEVICE_NOT_AN_INDEX,
)
from bindings.parallel_forecast_native_run import (
    forecast_arima_mode_admit, forecast_process_mode_admit, forecast_exog_admit, forecast_span_admit,
    forecast_positive_admit, forecast_out_of_sample_admit, forecast_index_admit, forecast_batch_admit,
    forecast_length_admit, forecast_envelope_admit, forecast_rows_copy, forecast_rows_place, forecast_words_place,
)
from core.parallel_forecast_impl import forecast_gather_cols, forecast_scatter_cols
from core.shared_slot import SharedSlot
from bindings.parallel_keys_unpack import (
    parallel_keys,
    PARALLEL_FORECAST_OPN_FIT_ARIMA,
    PARALLEL_FORECAST_OPN_FIT_EXPONENTIAL_SMOOTHING,
    PARALLEL_FORECAST_OPN_FORECAST_ARIMA,
    PARALLEL_FORECAST_OPN_FORECAST_EXPONENTIAL_SMOOTHING,
    PARALLEL_FORECAST_OPN_INITIALIZE,
    PARALLEL_FORECAST_OPN_PREDICT_ARIMA,
    PARALLEL_FORECAST_OPN_PREDICT_EXPONENTIAL_SMOOTHING,
    PARALLEL_FORECAST_OPN_WORKER,
    PARALLEL_FORECAST_WORKER_OPN_ARIMA_FIT_SHARD,
    PARALLEL_FORECAST_WORKER_OPN_ARIMA_PREDICT_SHARD,
    PARALLEL_FORECAST_WORKER_OPN_HW_FIT_SHARD,
    PARALLEL_FORECAST_WORKER_OPN_HW_PREDICT_SHARD,
    PARALLEL_FORECAST_WORKER_OPN_IDENTITY,
)
from bindings.py_keys import py_op_code
from bindings.py_balanced import py_item

comptime _CHILD = "mojolearn._native_parallel_forecast_worker"


struct ParallelForecastBoundary(Defaultable, Movable):
    var arima_cls: PythonObject
    var hw_cls: PythonObject
    var arima_context: PythonObject
    var hw_context: PythonObject
    var backend: PythonObject

    def __init__(out self):
        self.arima_cls = PythonObject(None)
        self.hw_cls = PythonObject(None)
        self.arima_context = PythonObject(None)
        self.hw_context = PythonObject(None)
        self.backend = PythonObject(None)


comptime PARALLEL_FORECAST_BOUNDARY = SharedSlot[ParallelForecastBoundary, "MojolearnParallelForecastOwnerV1"]


def _b() raises -> PythonObject:
    return native_loaded_module("builtins")


def _arima_mode(mode: PythonObject) raises:
    """UNPACK of the model's own mode word for `forecast_arima_mode_admit`.
    Once per driver call, no loop: one type test and one String read."""
    var exact = mode is not None and py_type(mode) is _b().str
    forecast_arima_mode_admit(mode is not None, exact, String(py=mode) if exact else String(""))


def _width(width: PythonObject) raises -> Int:
    var exact = py_type(width) is _b().int
    shard_width_admit(exact, Int(py=width) if exact else 0, "series_per_shard")
    return Int(py=width)


def _ranges(count: Int, width: PythonObject) raises -> List[Int]:
    """`_ranges(count, width)`: the shard bounds [lo0, hi0, lo1, hi1, ...]."""
    return shard_bounds(count, _width(width), "at least one series is required")


def _member_device(estimator: PythonObject, driver: String) raises:
    """The forecasting drivers run the model on the GPUs in `devices=` only:
    a model that says device='cpu' in a process with a GPU set is refused in
    words before any shard is cut (bindings/msel_member_device_unpack.mojo).
    `_backend._CPU_ONLY` (None in a process with a GPU set) is read from the
    connected backend namespace only for a model that says 'cpu'."""
    if not member_says_cpu(py_fields(estimator), _b()):
        return
    var f = PARALLEL_FORECAST_BOUNDARY.get_or_create_ptr()
    if py_get(f[].backend, PythonObject("_CPU_ONLY")) is None:
        member_multi_gpu_refusal(driver)


def _devices(devices: PythonObject) raises:
    """`DevicePool`'s admission of the device list, in its words."""
    var b = _b()
    var selected = py_call(b.tuple, devices)
    var indices = List[Int](capacity=len(selected))
    for i in range(len(selected)):  # driver-loop: over the admitted device indices
        indices.append(Int(py=selected[i]) if py_type(selected[i]) is b.int else SHARD_DEVICE_NOT_AN_INDEX)
    shard_indices_admit(indices)


def _run(requests: PythonObject, devices: PythonObject) raises -> PythonObject:
    """`_run`: the pool's results in request order (an admitted, traced pool,
    bindings/parallel_shard_common_door.mojo `shard_run`); the pool always closes."""
    return shard_run(requests, devices, _CHILD)


def _rows(source: PythonObject, lo: Int, hi: Int, width: Int, context: PythonObject) raises -> PythonObject:
    """`source[lo:hi]` of a C-ordered (rows, width) float32 Array, a copy."""
    var out = tsa_array("f", "<f4", Python.tuple(hi - lo, width), (hi - lo) * width, 4, context)
    if (hi - lo) * width > 0:
        forecast_rows_copy(Int(py=classical_address(out)), Int(py=classical_address(source)), lo, hi, width)
        _ = out  # keep the Python object alive through the native call above
        _ = source
    return out


# ---------------------------------------------------------------- ARIMA

def _predict_arima(estimator: PythonObject, start_obj: PythonObject, end_obj: PythonObject, exog: PythonObject,
                   devices: PythonObject, sps: PythonObject) raises -> PythonObject:
    var f = PARALLEL_FORECAST_BOUNDARY.get_or_create_ptr()
    var b = _b()
    if py_type(estimator) is not f[].arima_cls:
        raise Error("TypeError: predict_arima requires mojolearn.ARIMA")
    _member_device(estimator, "predict_arima")
    var fields = py_fields(estimator)
    var nk = parallel_keys()
    if not py_contains(fields, nk[].k_params_):
        raise Error("ValueError: mojolearn ARIMA: call fit() before predict")
    _arima_mode(py_get(fields, nk[].k_numeric_mode))
    forecast_exog_admit(Int(py=py_get(fields, nk[].k_n_exog_, PythonObject(0))), exog is not None)
    var n_obs = Int(py=py_item(fields, nk[].k_n_obs_))
    var start = Int(py=py_call(b.int, start_obj))
    var end = n_obs if end_obj is None else Int(py=py_call(b.int, end_obj))
    forecast_span_admit(start, end, "need 0 <= start < end; end is exclusive")
    var batch = Int(py=py_item(fields, nk[].k_batch_size_))
    var ranges = _ranges(batch, sps)
    _devices(devices)
    var complexity = Int(py=py_item(fields, nk[].k_complexity_))
    var keep = Python.tuple("order", "seasonal_order", "trend", "k_", "method", "maxiter", "n_exog_", "complexity_",
                            "n_obs_", "_y_ndim", "numeric_mode")
    var requests = b.list()
    var context = f[].arima_context
    for r in range(len(ranges) // 2):  # driver-loop: one worker request per device shard
        var lo = ranges[2 * r]
        var hi = ranges[2 * r + 1]
        # only the two series-major arrays the prediction kernel reads travel
        var state = b.dict()
        for k in range(len(keep)):  # driver-loop: over the scalar fields a shard's predict reads
            if py_contains(fields, keep[k]):
                py_setitem(state, keep[k], fields[keep[k]])
        py_setitem(state, nk[].k_batch_size_, hi - lo)
        py_setitem(state, nk[].k__y, _rows(py_item(fields, nk[].k__y), lo, hi, n_obs, context))
        py_setitem(state, nk[].k_params_, _rows(py_item(fields, nk[].k_params_), lo, hi, complexity, context))
        _ = py_call(requests.append, Python.tuple(nk[].k_arima_predict_shard, state, Python.tuple(start, end)))
    var parts = _run(requests, devices)
    var width = end - start
    var result = tsa_array("f", "<f4", Python.tuple(batch, width), batch * width, 4, context)
    for r in range(len(ranges) // 2):  # driver-loop: places one result buffer per device shard
        var lo = ranges[2 * r]
        var hi = ranges[2 * r + 1]
        var value = parts[r]
        if not Bool(py=py_call(b.tuple, value.shape) == Python.tuple(hi - lo, width)) or not Bool(py=value.dtype == "<f4"):
            raise Error("ValueError: ARIMA worker returned an invalid prediction shape or dtype")
        forecast_rows_place(Int(py=classical_address(result)), Int(py=classical_address(value)), lo, hi, width)
        _ = result  # keep the Python object alive through the native call above
        _ = value  # keep the Python object alive through the native call above
    return result


def _forecast_arima(estimator: PythonObject, steps_obj: PythonObject, exog: PythonObject, devices: PythonObject,
                    sps: PythonObject) raises -> PythonObject:
    var fields = py_fields(estimator)
    var nk = parallel_keys()
    if not py_contains(fields, nk[].k_params_):
        raise Error("ValueError: mojolearn ARIMA: call fit() before forecast")
    var steps = Int(py=py_call(_b().int, steps_obj))
    forecast_positive_admit(steps, "steps")
    var n_obs = Int(py=py_item(fields, nk[].k_n_obs_))
    return _predict_arima(estimator, PythonObject(n_obs), PythonObject(n_obs + steps), exog, devices, sps)


# ---------------------------------------------------------------- Holt-Winters

def _predict_hw(estimator: PythonObject, start_obj: PythonObject, end_obj: PythonObject, index: PythonObject,
                devices: PythonObject, sps: PythonObject) raises -> PythonObject:
    var f = PARALLEL_FORECAST_BOUNDARY.get_or_create_ptr()
    var b = _b()
    if py_type(estimator) is not f[].hw_cls:
        raise Error("TypeError: requires mojolearn.ExponentialSmoothing")
    _member_device(estimator, "predict_exponential_smoothing")
    var fields = py_fields(estimator)
    var nk = parallel_keys()
    if not Bool(py=py_item(fields, nk[].k_fit_executed_flag)):
        raise Error("ValueError: fit the model before prediction")
    forecast_process_mode_admit(String(py=py_item(f[].backend, nk[].k__DEFAULT_MODE)),
                                "parallel forecasting requires IDENTICAL numeric mode")
    if py_type(start_obj) is not b.int or py_type(end_obj) is not b.int:
        raise Error("TypeError: start and end must be integers")
    var start = Int(py=start_obj)
    var end = Int(py=end_obj)
    forecast_span_admit(start, end, "need 0 <= start < end")
    var n = Int(py=py_item(fields, nk[].k_n))
    forecast_out_of_sample_admit(start, n)
    var batch = Int(py=py_item(fields, nk[].k_ts_num))
    if index is not None:
        if py_type(index) is not b.int:
            raise Error("TypeError: index must be int or None")
        forecast_index_admit(Int(py=index), batch)
    var ranges = _ranges(batch, sps)
    _devices(devices)
    var context = f[].hw_context
    var steps = n - Int(py=py_item(fields, nk[].k_seasonal_periods))
    var comps = py_item(fields, nk[].k__comps)
    var drop = py_call(_hw_fitted().union, Python.list("endog"))
    # prediction reads the component state and the constructor settings the
    # native snapshot admits (start_periods, eps, initialization_method,
    # numeric_mode, ...): every carried scalar field, without the batch's
    # fitted arrays, then each shard's slice. The carried fields are walked
    # once; each shard takes a shallow copy and sets its own words.
    var base = shard_state(fields, drop, context)
    var requests = b.list()
    for r in range(len(ranges) // 2):  # driver-loop: one worker request per device shard
        var lo = ranges[2 * r]
        var hi = ranges[2 * r + 1]
        var w = hi - lo
        var state = py_call(b.dict, base)
        py_setitem(state, nk[].k_n, n)
        py_setitem(state, nk[].k_fit_executed_flag, py_item(fields, nk[].k_fit_executed_flag))
        py_setitem(state, nk[].k_ts_num, w)
        py_setitem(state, nk[].k__components_len, steps * w)
        var part = tsa_array("f", "<f4", Python.tuple(3 * steps * w), 3 * steps * w, 4, context)
        # rows r = component * steps + step: dst[r * w + j] = src[r * batch + lo + j]
        forecast_gather_cols(Int(py=classical_address(part)), Int(py=classical_address(comps)), 3 * steps, batch, lo, w)
        _ = part  # keep the Python object alive through the native call above
        _ = comps  # keep the Python object alive through the native call above
        py_setitem(state, nk[].k__comps, part)
        _ = py_call(requests.append, Python.tuple(nk[].k_hw_predict_shard, state, Python.tuple(start, end)))
    var parts = _run(requests, devices)
    var width = end - start
    var result = tsa_array("f", "<f4", Python.tuple(width * batch), width * batch, 4, context)
    for r in range(len(ranges) // 2):  # driver-loop: places one result buffer per device shard
        var lo = ranges[2 * r]
        var hi = ranges[2 * r + 1]
        var value = parts[r]
        var expected = Python.tuple(width) if hi - lo == 1 else Python.tuple(width, hi - lo)
        if not Bool(py=py_call(b.tuple, value.shape) == expected) or not Bool(py=value.dtype == "<f4"):
            raise Error("ValueError: Holt-Winters worker returned an invalid prediction shape or dtype")
        forecast_scatter_cols(Int(py=classical_address(result)), Int(py=classical_address(value)), width, batch, lo, hi - lo)
        _ = value  # keep the Python object alive through the native call above
        _ = result  # keep the Python object alive through the native call above
    # `estimator._shaped(result, width, index)`: one series, the flat
    # block (one series in all), or the (width, ts_num) block
    if index is not None:
        var one = tsa_array("f", "<f4", Python.tuple(width), width, 4, context)
        forecast_gather_cols(Int(py=classical_address(one)), Int(py=classical_address(result)), width, batch, Int(py=index), 1)
        _ = one  # keep the Python object alive through the native call above
        _ = result  # keep the Python object alive through the native call above
        return one
    if batch == 1:
        return result
    var block = tsa_array("f", "<f4", Python.tuple(width, batch), width * batch, 4, context)
    shard_block_copy(Int(py=classical_address(block)), Int(py=classical_address(result)), width * batch * 4)
    _ = block  # keep the Python object alive through the native call above
    _ = result  # keep the Python object alive through the native call above
    return block


def _forecast_hw(estimator: PythonObject, h: PythonObject, index: PythonObject, devices: PythonObject,
                 sps: PythonObject) raises -> PythonObject:
    if py_type(h) is not _b().int:
        raise Error("TypeError: h must be an integer")
    forecast_positive_admit(Int(py=h), "h")
    var nk = parallel_keys()
    var n = Int(py=py_item(py_fields(estimator), nk[].k_n))
    return _predict_hw(estimator, PythonObject(n), PythonObject(n + Int(py=h)), index, devices, sps)


# ---------------------------------------------------------------- fits
# parallel_classical.fit_arima / fit_exponential_smoothing, one native call
# each (lane/native-parallel-driver-20261024). The partition, the driver
# negative control and the merge are the old Python driver's: whole series in
# `series_per_shard` ranges (each shard after the first on an owner above rank
# 0 reads one series early under the switch), each shard fitted by the plain
# native fit in its own worker, every fitted array placed at its shard's
# series offset. The plain fit's own words on the same series: no bit moves.

#: ARIMA fields the fit writes; a shard worker's state carries the estimator's
#: constructor fields only.
def _arima_fitted() raises -> PythonObject:
    return _b().frozenset(Python.list("params_", "x_", "x0_", "n_iter_", "retcode_", "llf_", "fx_", "aic_", "bic_",
                                      "_y", "_exog", "batch_size_", "n_obs_", "input_copied_", "_y_ndim"))


def _fit_arima(estimator: PythonObject, y: PythonObject, devices: PythonObject, sps: PythonObject,
               exog: PythonObject) raises -> PythonObject:
    var f = PARALLEL_FORECAST_BOUNDARY.get_or_create_ptr()
    var b = _b()
    if py_type(estimator) is not f[].arima_cls:
        raise Error("TypeError: fit_arima requires mojolearn.ARIMA")
    _member_device(estimator, "fit_arima")
    if exog is not None:
        raise Error("NotImplementedError: fit_arima does not shard exogenous regressors; ARIMA.fit(y, exog) fits "
                    + "them on one device")
    var fields = py_fields(estimator)
    var nk = parallel_keys()
    _arima_mode(py_get(fields, nk[].k_numeric_mode))
    _ = _width(sps)
    var context = f[].arima_context
    var series = _series_major(y, context)
    var data = series[0]
    var batch = Int(py=series[1])
    var n_obs = Int(py=series[2])
    var copied = series[3]
    var ranges = shard_ranges(batch, sps, "series_per_shard", "")
    var drop = _arima_fitted()
    var state = shard_state(fields, drop, context)
    py_setitem(state, nk[].k_numeric_mode, "identical")
    var n_devices = shard_devices(devices)
    var requests = b.list()
    for r in range(len(ranges) // 2):  # driver-loop: one worker request per series shard
        var lo = ranges[2 * r]
        var hi = ranges[2 * r + 1]
        var shift = shard_read_shift(r, lo, n_devices)
        _ = py_call(requests.append, Python.tuple(nk[].k_arima_fit_shard, state,
                                                  Python.tuple(shard_rows(data, lo - shift, hi - shift, context))))
    var parts = shard_run(requests, devices, _CHILD)
    var merged = b.dict()
    var names = [String("params_"), String("x_"), String("x0_"), String("n_iter_"), String("retcode_"),
                 String("llf_"), String("fx_"), String("aic_"), String("bic_")]
    for i in range(len(names)):  # small-loop(the nine series-major fitted arrays)
        py_setitem(merged, names[i], shard_concat(parts, names[i], batch, context))
    # `result = ARIMA(**params); result.__dict__.update(parts[0].__dict__)`,
    # the merged arrays, then the whole batch's series state
    _ = py_call(fields.clear)
    _ = py_call(fields.update, parts[0])
    _ = py_call(fields.update, merged)
    py_setitem(fields, nk[].k_batch_size_, batch)
    py_setitem(fields, nk[].k_n_obs_, n_obs)
    py_setitem(fields, nk[].k__y, data)
    py_setitem(fields, nk[].k__y_ndim, series[4])
    py_setitem(fields, nk[].k_input_copied_, copied)
    return estimator


#: Holt-Winters fields the fit writes.
def _hw_fitted() raises -> PythonObject:
    return _b().frozenset(Python.list("_data", "_comps", "_components_len", "_comp_layout", "_time_major", "sse_",
                                      "alpha_", "beta_", "gamma_", "n_iter_", "criterion_", "n", "level_", "trend_",
                                      "season_"))


def _fit_hw(estimator: PythonObject, devices: PythonObject, sps: PythonObject) raises -> PythonObject:
    var f = PARALLEL_FORECAST_BOUNDARY.get_or_create_ptr()
    var b = _b()
    if py_type(estimator) is not f[].hw_cls:
        raise Error("TypeError: requires mojolearn.ExponentialSmoothing")
    _member_device(estimator, "fit_exponential_smoothing")
    var nk = parallel_keys()
    forecast_process_mode_admit(String(py=py_item(f[].backend, nk[].k__DEFAULT_MODE)),
                                "ExponentialSmoothing requires IDENTICAL numeric mode")
    _ = _width(sps)
    var fields = py_fields(estimator)
    var batch = Int(py=py_item(fields, nk[].k_ts_num))
    forecast_batch_admit(batch)
    var context = f[].hw_context
    var checked = _check_dims(fields, py_item(fields, nk[].k_endog), context)
    var data = checked[0]
    var n = Int(py=checked[1])
    var ranges = shard_ranges(batch, sps, "series_per_shard", "")
    var drop = _hw_fitted()
    var state = shard_state(fields, drop, context)
    py_setitem(state, nk[].k_fit_executed_flag, False)
    var n_devices = shard_devices(devices)
    var requests = b.list()
    for r in range(len(ranges) // 2):  # driver-loop: one worker request per series shard
        var lo = ranges[2 * r]
        var hi = ranges[2 * r + 1]
        var shift = shard_read_shift(r, lo, n_devices)
        var shard_state_r = py_call(b.dict, state)
        # `ExponentialSmoothing(data[lo:hi], ts_num=hi - lo, **params)`
        py_setitem(shard_state_r, nk[].k_endog, shard_rows(data, lo - shift, hi - shift, context))
        py_setitem(shard_state_r, nk[].k_ts_num, hi - lo)
        _ = py_call(requests.append, Python.tuple(nk[].k_hw_fit_shard, shard_state_r, Python.tuple()))
    var parts = shard_run(requests, devices, _CHILD)
    # The plain fit's buffers for the whole batch, every shard's columns at
    # its series offset: the packed components (3 * steps rows of `batch`
    # series), the four per-series statistics and the two flag rows.
    var period = Int(py=py_item(fields, nk[].k_seasonal_periods))
    var steps = n - period
    var comps = shard_array("<f4", Python.tuple(3 * steps * batch), 3 * steps * batch, context)
    var stats = shard_array("<f4", Python.tuple(4 * batch), 4 * batch, context)
    var flags = shard_array("<i4", Python.tuple(2 * batch), 2 * batch, context)
    var stat_names = [String("sse_"), String("alpha_"), String("beta_"), String("gamma_")]
    for r in range(len(ranges) // 2):  # driver-loop: places one shard's blocks
        var lo = ranges[2 * r]
        var w = ranges[2 * r + 1] - lo
        var part = parts[r]
        forecast_length_admit(Int(py=py_item(part, nk[].k_n)), n)
        var shard_comps = py_item(part, nk[].k__comps)
        if steps > 0 and w > 0:
            forecast_scatter_cols(Int(py=classical_address(comps)), Int(py=classical_address(shard_comps)),
                                  3 * steps, batch, lo, w)
            _ = comps  # keep the Python object alive through the native call above
            _ = shard_comps  # keep the Python object alive through the native call above
        for i in range(4):  # small-loop(the four per-series statistics)
            var value = part[stat_names[i]]
            forecast_words_place(Int(py=classical_address(stats)), Int(py=classical_address(value)), i, batch, lo, w)
            _ = stats  # keep the Python object alive through the native call above
            _ = value  # keep the Python object alive through the native call above
        var iters = py_item(part, nk[].k_n_iter_)
        var criteria = py_item(part, nk[].k_criterion_)
        forecast_words_place(Int(py=classical_address(flags)), Int(py=classical_address(iters)), 0, batch, lo, w)
        forecast_words_place(Int(py=classical_address(flags)), Int(py=classical_address(criteria)), 1, batch, lo, w)
        _ = flags  # keep the Python object alive through the native call above
        _ = iters  # keep the Python object alive through the native call above
        _ = criteria  # keep the Python object alive through the native call above
    # `_set_fitted` returns the fitted updates; commit them as the plain
    # `hw_fit` does (discarding them left the estimator unfitted: "fit() the
    # model before forecast()" and no fitted components).
    var updates = _set_fitted(fields, n, comps, stats, flags, context)
    py_setitem(updates, nk[].k__data, data)
    _commit(tsa_hw_fields(estimator, context), updates, context, True)
    return estimator


def _fitted_record(model: PythonObject, names: PythonObject) raises -> PythonObject:
    """The fitted fields `names` of a shard worker's model."""
    var b = _b()
    var fields = py_fields(model)
    var out = b.dict()
    for k in range(len(names)):  # driver-loop: over the fitted field names a shard returns
        if py_contains(fields, names[k]):
            py_setitem(out, names[k], fields[names[k]])
    return out


# ---------------------------------------------------------------- the worker

def forecast_worker_dispatch(request: PythonObject) raises -> PythonObject:
    """One shard: the model rebuilt natively from its state record, then the
    native predict over (start, end)."""
    var f = PARALLEL_FORECAST_BOUNDARY.get_or_create_ptr()
    var b = _b()
    var nk = parallel_keys()
    if len(request) == 1 and py_op_code(nk[].ops_forecast_worker, request[0]) == PARALLEL_FORECAST_WORKER_OPN_IDENTITY:
        return shard_identity()
    forecast_envelope_admit(len(request))
    var operation = py_op_code(nk[].ops_forecast_worker, request[0])
    var state = request[1]
    var args = request[2]
    if operation == PARALLEL_FORECAST_WORKER_OPN_ARIMA_PREDICT_SHARD:
        var model = py_call(b.object.__new__, f[].arima_cls)
        var fields = py_fields(model)
        var keys = py_call(b.list, state)
        for k in range(len(keys)):  # driver-loop: over the shard's state record fields
            py_setitem(fields, keys[k], state[keys[k]])
        py_setitem(fields, nk[].k__exog, PythonObject(None))
        return arima_predict(model, Python.tuple(args[0], args[1], PythonObject(None)), f[].arima_context)
    if operation == PARALLEL_FORECAST_WORKER_OPN_HW_PREDICT_SHARD:
        var model = py_call(b.object.__new__, f[].hw_cls)
        var fields = py_fields(model)
        var keys = py_call(b.list, state)
        for k in range(len(keys)):  # driver-loop: over the shard's state record fields
            py_setitem(fields, keys[k], state[keys[k]])
        return hw_predict(model, Python.tuple(args[0], args[1], PythonObject(None)), f[].hw_context)
    if operation == PARALLEL_FORECAST_WORKER_OPN_ARIMA_FIT_SHARD:
        var model = shard_model(f[].arima_cls, state)
        _ = arima_fit(model, Python.tuple(args[0], PythonObject(None)), f[].arima_context)
        # everything the fit set except the shard's series copy
        var fields = py_call(b.dict, py_fields(model))
        _ = py_call(fields.pop, "_y", PythonObject(None))
        return shard_state(fields, b.frozenset(), f[].arima_context)
    if operation == PARALLEL_FORECAST_WORKER_OPN_HW_FIT_SHARD:
        var model = shard_model(f[].hw_cls, state)
        _ = hw_fit(model, f[].hw_context)
        return _fitted_record(model, Python.list("n", "_comps", "sse_", "alpha_", "beta_", "gamma_", "n_iter_",
                                                 "criterion_"))
    raise Error("parallel forecasting: unsupported worker operation")


def forecast_worker(argv: PythonObject) raises:
    var socket = process_worker_connect(argv)
    var nk = parallel_keys()
    try:
        while True:  # driver-loop: over native local forecast shard requests
            var request = process_worker_next(socket)
            if request is None:
                break
            var sequence = Int(py=py_get(request[0], nk[].k_sequence, -1))
            try:
                process_send(socket, "result", sequence, forecast_worker_dispatch(request[1]))
            except error:
                process_send(socket, "error", sequence, PythonObject(String(error)))
    except error:
        transport_close(socket)
        raise error^
    transport_close(socket)


def parallel_forecast_call(operation: Int, name: PythonObject, args: PythonObject) raises -> PythonObject:
    """`operation` is a `PARALLEL_FORECAST_OPN_*` code (-1 for a name the table
    does not hold); `name` is read only for the unknown operation's text."""
    var f = PARALLEL_FORECAST_BOUNDARY.get_or_create_ptr()
    if operation == PARALLEL_FORECAST_OPN_INITIALIZE:
        f[].arima_cls = args[0]
        f[].hw_cls = args[1]
        f[].arima_context = args[2]
        f[].hw_context = args[3]
        f[].backend = args[4]
        return PythonObject(None)
    if operation == PARALLEL_FORECAST_OPN_PREDICT_ARIMA:
        return _predict_arima(args[0], args[1], args[2], args[3], args[4], args[5])
    if operation == PARALLEL_FORECAST_OPN_FORECAST_ARIMA:
        return _forecast_arima(args[0], args[1], args[2], args[3], args[4])
    if operation == PARALLEL_FORECAST_OPN_PREDICT_EXPONENTIAL_SMOOTHING:
        return _predict_hw(args[0], args[1], args[2], args[3], args[4], args[5])
    if operation == PARALLEL_FORECAST_OPN_FORECAST_EXPONENTIAL_SMOOTHING:
        return _forecast_hw(args[0], args[1], args[2], args[3], args[4])
    if operation == PARALLEL_FORECAST_OPN_FIT_ARIMA:
        return _fit_arima(args[0], args[1], args[2], args[3], args[4])
    if operation == PARALLEL_FORECAST_OPN_FIT_EXPONENTIAL_SMOOTHING:
        return _fit_hw(args[0], args[1], args[2])
    if operation == PARALLEL_FORECAST_OPN_WORKER:
        forecast_worker(args)
        return PythonObject(None)
    raise Error("RuntimeError: parallel forecasting: unknown operation " + String(name))


def parallel_forecast_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var args = PythonObject(from_borrowed=args_ptr)
    try:
        var nk = parallel_keys()
        var result = parallel_forecast_call(py_op_code(nk[].ops_forecast, args[0]), args[0], args[1])
        return result.steal_data()
    except error:
        var message = String(error)
        if message.startswith("InterruptedError:"):
            return raise_python_exception(error_without_class(error, "InterruptedError"), ExceptionType("PyExc_InterruptedError"))
        if message.startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if message.startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if message.startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if message.startswith("IndexError:"):
            return raise_python_exception(error_without_class(error, "IndexError"), ExceptionType("PyExc_IndexError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
