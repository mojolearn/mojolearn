# Time series

The time-series models forecast future values of a series from its past, or split a series into trend
and seasonal parts. Most of them fit a whole batch of series at once: pass a 2-D array with one series
per row, and each series gets its own parameters.

| Use | Estimators |
|---|---|
| ARIMA and seasonal ARIMA | `ARIMA`, `AutoARIMA` |
| Exponential smoothing | `ETS`, `ExponentialSmoothing`, `DampedETS` |
| Theta methods | `Theta`, `AutoTheta`, `OptimizedTheta`, `DynamicTheta`, `DynamicOptimizedTheta` |
| Intermittent demand | `CrostonClassic`, `CrostonOptimized`, `CrostonSBA` |
| Several series that move together | `VAR` |
| Volatility | `GARCH` |
| Trend and season decomposition | `STL` |
| Trend, seasonality and holidays | `ProphetForecaster` |

## ARIMA

The data goes to `fit`, not to the constructor. `forecast(steps)` returns one row of forecasts per
series.

```python
import numpy as np
import mojolearn

y = np.random.default_rng(0).standard_normal((16, 200)).cumsum(axis=1).astype(np.float32)

model = mojolearn.ARIMA(order=(1, 1, 1)).fit(y)
print(model.forecast(10).shape)       # (16, 10)
```

`AutoARIMA` searches over orders for each series and picks the best by an information criterion:

```python
auto = mojolearn.AutoARIMA(y).search(d=range(2), p=range(3), q=range(3)).fit()
print(auto.forecast(10).shape)
```

## Exponential smoothing and theta

`ETS`, `Theta` and their relatives fit on `fit(y)` and forecast `h` steps with `predict(h)`, which
returns a dictionary with the forecasts under `"mean"`.

```python
t = np.arange(120, dtype=np.float32)
series = 50 + 0.2 * t + 5 * np.sin(2 * np.pi * t / 12).astype(np.float32)

ets = mojolearn.ETS(season_length=12, model="AAA", damped=False).fit(series)
print(ets.predict(12)["mean"].shape)  # (12,)

theta = mojolearn.Theta(season_length=12).fit(series)
print(theta.predict(12)["mean"])
```

## Decomposition

```python
parts = mojolearn.STL(series, period=12).fit()
trend, seasonal, resid = parts.trend, parts.seasonal, parts.resid
```

## Saving a model

`ARIMA` and `ExponentialSmoothing` save to a file and load back:

```python
model.save("arima.npz")
loaded = mojolearn.ARIMA.load("arima.npz")
```

## Devices and numeric modes

In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs. `identical` is the
default. On Apple GPUs you can choose `fast` instead, which is tuned for speed and does not promise
identical bits: call `mojolearn.set_numeric_mode()` with `"fast"`. See [Numeric modes](numeric_modes.md).

## See also

- [Choosing an estimator](choosing_an_estimator.md)
- API reference: [Time series](../api/time_series.rst)
