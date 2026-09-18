# SPDX-License-Identifier: Apache-2.0
"""Explicit GPU series partitions for fitted forecasting models.

Each series retains its original arithmetic. Host state is sliced before it is
sent to a worker; the complete batch need not fit on any one GPU. Each individual
series must fit. Physical NVIDIA/AMD qualification is tracked separately.
"""
__all__ = ['predict_arima', 'forecast_arima', 'predict_exponential_smoothing',
           'forecast_exponential_smoothing']

import copy
from ._parallel_pool import DevicePool
from ._buffer import empty, addr, addr_ro
from ._bufcheck import memcopy


def _ranges(count, width):
    if type(width) is not int or width < 1:
        raise ValueError('series_per_shard must be a positive integer')
    if count < 1:
        raise ValueError('at least one series is required')
    return [(i, min(i + width, count)) for i in range(0, count, width)]


def _run(requests, devices):
    pool = DevicePool(devices)
    try:
        results = pool.map(requests)
        if len(results) != len(requests):
            raise ValueError("forecast workers returned an incomplete batch")
        return results
    finally:
        pool.close()


def predict_arima(estimator, start=0, end=None, *, exog=None, devices=(0,), series_per_shard=1):
    """Predict whole ARIMA series on GPU workers; ``end`` is exclusive.

    Exogenous models are deliberately refused until their partition contract is
    qualified. This does not alter public CPU inference on ordinary ARIMA.
    """
    from ._arima_impl import ARIMA
    if type(estimator) is not ARIMA:
        raise TypeError('predict_arima requires mojolearn.ARIMA')
    estimator._check_fitted('predict')
    if estimator.numeric_mode not in (None, 'identical'):
        raise ValueError('parallel ARIMA requires IDENTICAL numeric mode')
    if getattr(estimator, 'n_exog_', 0) or exog is not None:
        raise NotImplementedError('parallel ARIMA prediction does not shard exogenous regressors')
    start, end = int(start), int(estimator.n_obs_ if end is None else end)
    if start < 0 or end <= start:
        raise ValueError('need 0 <= start < end; end is exclusive')
    ranges = _ranges(estimator.batch_size_, series_per_shard)
    requests = []
    for lo, hi in ranges:
        part = copy.copy(estimator)
        part.batch_size_ = hi - lo
        # Only these two series-major arrays are read by the prediction kernel.
        # Remove unrelated full-batch fit statistics from the transported state.
        part.__dict__ = {k: v for k, v in estimator.__dict__.items()
                         if k not in ('params_', '_y', 'x_', 'x0_', 'n_iter_',
                                      'retcode_', 'llf_', 'fx_', 'aic_', 'bic_')}
        part.batch_size_ = hi - lo
        part._y, part.params_ = estimator._y[lo:hi], estimator.params_[lo:hi]
        requests.append(('forecast_predict', part, ('predict', (start, end))))
    parts = _run(requests, devices)
    result = empty((estimator.batch_size_, end - start), '<f4')
    for (lo, hi), value in zip(ranges, parts):
        expected = (hi - lo, end - start)
        if value.shape != expected or value.dtype != result.dtype:
            raise ValueError('ARIMA worker returned an invalid prediction shape or dtype')
        memcopy(addr(result, name='predictions') + lo * (end - start) * 4,
                addr_ro(value, name='prediction shard'), value.nbytes)
    return result


def forecast_arima(estimator, steps, *, exog=None, devices=(0,), series_per_shard=1):
    """Forecast independent ARIMA series with the prediction partition."""
    estimator._check_fitted('forecast')
    steps = int(steps)
    if steps < 1:
        raise ValueError('steps must be positive')
    return predict_arima(estimator, estimator.n_obs_, estimator.n_obs_ + steps,
                         exog=exog, devices=devices, series_per_shard=series_per_shard)


def predict_exponential_smoothing(estimator, start, end, *, index=None,
                                  devices=(0,), series_per_shard=1):
    """GPU out-of-sample Holt-Winters predictions, with exclusive ``end``.

    In-sample prediction uses host arithmetic in the underlying implementation
    and is explicitly outside this GPU API. Original single-series/index return
    shapes are preserved, including when shards have different widths.
    """
    from ._tsa_impl import ExponentialSmoothing
    from . import _backend
    if type(estimator) is not ExponentialSmoothing:
        raise TypeError('requires mojolearn.ExponentialSmoothing')
    if not estimator.fit_executed_flag:
        raise ValueError('fit the model before prediction')
    if _backend.default_mode() != 'identical':
        raise ValueError('parallel forecasting requires IDENTICAL numeric mode')
    if type(start) is not int or type(end) is not int:
        raise TypeError('start and end must be integers')
    if start < 0 or end <= start:
        raise ValueError('need 0 <= start < end')
    if start < estimator.n:
        raise NotImplementedError('in-sample Holt-Winters prediction uses host arithmetic')
    if index is not None:
        if type(index) is not int:
            raise TypeError('index must be int or None')
        if not 0 <= index < estimator.ts_num:
            raise IndexError('series index outside fitted batch')
    ranges = _ranges(estimator.ts_num, series_per_shard)
    requests = []
    steps, batch = estimator.n - estimator.seasonal_periods, estimator.ts_num
    for lo, hi in ranges:
        part = copy.copy(estimator)
        # Prediction reads only component state and these scalar parameters.
        part.__dict__ = {k: getattr(estimator, k) for k in
                         ('n', 'seasonal_periods', 'seasonal', 'fit_executed_flag')}
        part.ts_num = hi - lo
        part._components_len = steps * part.ts_num
        part._comps = empty((3 * part._components_len,), '<f4')
        for component in range(3):
            for row in range(steps):
                memcopy(addr(part._comps, name='components') +
                        (component * steps + row) * part.ts_num * 4,
                        addr_ro(estimator._comps, name='components') +
                        ((component * steps + row) * batch + lo) * 4,
                        part.ts_num * 4)
        requests.append(('forecast_predict', part, ('predict', (start, end))))
    parts = _run(requests, devices)
    width = end - start
    result = empty((width * batch,), '<f4')
    for (lo, hi), value in zip(ranges, parts):
        expected = (width,) if hi - lo == 1 else (width, hi - lo)
        if value.shape != expected or value.dtype != result.dtype:
            raise ValueError('Holt-Winters worker returned an invalid prediction shape or dtype')
        for row in range(width):
            memcopy(addr(result, name='predictions') + (row * batch + lo) * 4,
                    addr_ro(value, name='prediction shard') + row * (hi - lo) * 4,
                    (hi - lo) * 4)
    return estimator._shaped(result, width, index)


def forecast_exponential_smoothing(estimator, h=1, *, index=None, devices=(0,), series_per_shard=1):
    """Forecast Holt-Winters series on independent GPU workers."""
    if type(h) is not int:
        raise TypeError('h must be an integer')
    if h < 1:
        raise ValueError('h must be positive')
    return predict_exponential_smoothing(estimator, estimator.n, estimator.n + h,
                                         index=index, devices=devices,
                                         series_per_shard=series_per_shard)
