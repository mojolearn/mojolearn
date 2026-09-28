# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""STL, Season-Trend decomposition by Loess, statsmodels-shaped
(`statsmodels.tsa.seasonal.STL`): `STL(endog, period=12).fit()` returns the
seasonal, trend and residual components and the robustness weights. The
arithmetic is `sequence/stl.mojo`, statsmodels' `_stl.pyx` loops in float32,
one series per GPU thread (or host loop iteration).

`endog` may be one series (1-D) or a batch `(batch_size, n_obs)`, one series
per row, all sharing the configuration; the batch is the GPU's parallelism.
Refused by name: `period=None` (the reference infers it from a pandas index;
pass it), float64 input, degrees other than 0 and 1 (the reference's own
bound)."""
import math

from ._optional_numpy import require_numpy
np = require_numpy('_x_sequence_stl')

from . import _backend


class DecomposeResult:
    """observed, seasonal, trend, resid, weights: arrays of the input's
    shape."""

    def __init__(self, observed, seasonal, trend, resid, weights):
        self.observed = observed
        self.seasonal = seasonal
        self.trend = trend
        self.resid = resid
        self.weights = weights


class STL:
    def __init__(self, endog, period=None, seasonal=7, trend=None, low_pass=None, seasonal_deg=1,
                 trend_deg=1, low_pass_deg=1, robust=False, seasonal_jump=1, trend_jump=1,
                 low_pass_jump=1, numeric_mode=None):
        y = np.asarray(endog)
        if y.dtype == np.float64:
            raise TypeError("STL: float64 is refused; pass float32 (no float64 on the device)")
        self._one = y.ndim == 1
        y = np.ascontiguousarray(y, dtype=np.float32).reshape(1, -1) if self._one else \
            np.ascontiguousarray(y, dtype=np.float32)
        if y.ndim != 2 or y.shape[1] < 1:
            raise ValueError("STL: endog must be (n_obs,) or (batch_size, n_obs)")
        if period is None:
            raise ValueError("STL: period is required (the reference infers it from a pandas index only)")
        period = int(period)
        if period < 2:
            raise ValueError("period must be a positive integer >= 2")
        if trend is None:
            trend = int(math.ceil(1.5 * period / (1 - 1.5 / seasonal)))
            trend += (trend % 2) == 0
        if low_pass is None:
            low_pass = period + 1
            low_pass += (low_pass % 2) == 0
        for name, v in (("seasonal_deg", seasonal_deg), ("trend_deg", trend_deg), ("low_pass_deg", low_pass_deg)):
            if v not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1")
        self.endog = y
        self.period = period
        self.seasonal = int(seasonal)
        self.trend = int(trend)
        self.low_pass = int(low_pass)
        self.seasonal_deg, self.trend_deg, self.low_pass_deg = int(seasonal_deg), int(trend_deg), int(low_pass_deg)
        self.robust = bool(robust)
        self.seasonal_jump, self.trend_jump, self.low_pass_jump = int(seasonal_jump), int(trend_jump), int(low_pass_jump)
        self.numeric_mode = numeric_mode

    @property
    def config(self):
        return dict(period=self.period, seasonal=self.seasonal, seasonal_deg=self.seasonal_deg,
                    seasonal_jump=self.seasonal_jump, trend=self.trend, trend_deg=self.trend_deg,
                    trend_jump=self.trend_jump, low_pass=self.low_pass, low_pass_deg=self.low_pass_deg,
                    low_pass_jump=self.low_pass_jump, robust=self.robust)

    def fit(self, inner_iter=None, outer_iter=None):
        if inner_iter is None:
            inner_iter = 2 if self.robust else 5
        if outer_iter is None:
            outer_iter = 15 if self.robust else 0
        B, n = self.endog.shape
        outs = [np.zeros((B, n), dtype=np.float32) for _ in range(4)]
        b = _backend.binding("_mojolearn_x_sequence", self.numeric_mode)
        b.stl([self.endog.ctypes.data] + [o.ctypes.data for o in outs],
              [B, n, self.period, self.seasonal, self.trend, self.low_pass,
               self.seasonal_deg + 2 * self.trend_deg + 4 * self.low_pass_deg,
               self.seasonal_jump, self.trend_jump, self.low_pass_jump, int(inner_iter), int(outer_iter)])
        season, trend, weights, resid = (o[0] if self._one else o for o in outs)
        observed = self.endog[0] if self._one else self.endog
        return DecomposeResult(observed, season, trend, resid, weights)
