# SPDX-License-Identifier: Apache-2.0
"""Explicit GPU series partitions for fitted forecasting models.

Each series retains its original arithmetic. Host state is sliced before it is
sent to a worker; the complete batch need not fit on any one GPU. Each individual
series must fit. Physical NVIDIA/AMD qualification is tracked separately.
"""
__all__ = ['predict_arima', 'forecast_arima', 'predict_exponential_smoothing',
           'forecast_exponential_smoothing']

# The partition, the workers and the merge are the installed native owner
# (bindings/parallel_forecast_native.mojo, on the native local process pool):
# each public function is one call into it.
from . import _native_process  # noqa: F401  (connects the native process pool)
from . import _backend
from .tokenizer import _binding
from ._arima_impl import ARIMA as _ARIMA, _TSA_CONTEXT as _ARIMA_CONTEXT
from ._tsa_impl import ExponentialSmoothing as _HW, _TSA_CONTEXT as _HW_CONTEXT

_native_parallel_forecast = _binding()._module.parallel_forecast_call
_native_parallel_forecast('initialize', (_ARIMA, _HW, _ARIMA_CONTEXT, _HW_CONTEXT, vars(_backend)))


def predict_arima(estimator, start=0, end=None, *, exog=None, devices=(0,), series_per_shard=1):
    """Predict whole ARIMA series on GPU workers; ``end`` is exclusive.

    Exogenous models are deliberately refused until their partition contract is
    qualified. This does not alter public CPU inference on ordinary ARIMA.
    """
    return _native_parallel_forecast('predict_arima', (estimator, start, end, exog, devices, series_per_shard))


def forecast_arima(estimator, steps, *, exog=None, devices=(0,), series_per_shard=1):
    """Forecast independent ARIMA series with the prediction partition."""
    return _native_parallel_forecast('forecast_arima', (estimator, steps, exog, devices, series_per_shard))


def predict_exponential_smoothing(estimator, start, end, *, index=None,
                                  devices=(0,), series_per_shard=1):
    """GPU out-of-sample Holt-Winters predictions, with exclusive ``end``.

    In-sample prediction uses host arithmetic in the underlying implementation
    and is explicitly outside this GPU API. Original single-series/index return
    shapes are preserved, including when shards have different widths.
    """
    return _native_parallel_forecast('predict_exponential_smoothing',
                                     (estimator, start, end, index, devices, series_per_shard))


def forecast_exponential_smoothing(estimator, h=1, *, index=None, devices=(0,), series_per_shard=1):
    """Forecast Holt-Winters series on independent GPU workers."""
    return _native_parallel_forecast('forecast_exponential_smoothing',
                                     (estimator, h, index, devices, series_per_shard))
