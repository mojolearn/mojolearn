# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""statsforecast's Theta family (`statsforecast.models`: Theta,
OptimizedTheta, DynamicTheta, DynamicOptimizedTheta, AutoTheta), fitted on
the GPU one series per thread (`sequence/theta.mojo`): auto_theta's seasonal
test and classical decomposition, the theta state-space models, their MSE
objective minimised by statsforecast's Nelder-Mead, and the point forecast.
`fit(y)` takes one series (1-D) or a batch `(batch_size, n_obs)`;
`predict(h)` returns {"mean": forecasts} shaped like the input's batch.

Refused by name (sequence/NOT_IMPLEMENTED.tsv): prediction intervals
(`level`), fitted values / in-sample residual output, non-normal error
distributions, exogenous regressors, float64 input."""
from ._optional_numpy import require_numpy
np = require_numpy('_x_sequence_theta')

from . import _backend

_MODELS = {"STM": 0, "OTM": 1, "DSTM": 2, "DOTM": 3}
_MODEL_NAMES = np.array(list(_MODELS), dtype=object)


class AutoTheta:
    """statsforecast AutoTheta: model None tries all four and keeps the least
    MSE."""
    _MODEL = None

    def __init__(self, season_length=1, decomposition_type="multiplicative", model=None,
                 initial_smoothed=None, alpha=None, theta=None, numeric_mode=None):
        if decomposition_type not in ("multiplicative", "additive"):
            raise ValueError("decomposition_type must be 'multiplicative' or 'additive'")
        if model is not None and model not in _MODELS:
            raise ValueError(f"model must be one of {sorted(_MODELS)} or None")
        self.season_length = int(season_length)
        self.decomposition_type = decomposition_type
        self.model = self._MODEL if self._MODEL is not None else model
        self.initial_smoothed, self.alpha, self.theta = initial_smoothed, alpha, theta
        self.numeric_mode = numeric_mode

    def fit(self, y, X=None):
        if X is not None:
            raise NotImplementedError("Theta: exogenous regressors are not implemented")
        y = np.asarray(y)
        if y.dtype == np.float64:
            raise TypeError("Theta: float64 is refused; pass float32")
        self._one = y.ndim == 1
        # a private copy: the fitted series cannot change under a stored forecast
        self._y = np.array(y, dtype=np.float32, order="C", copy=True).reshape(1, -1) if self._one else \
            np.array(y, dtype=np.float32, order="C", copy=True)
        if self._y.ndim != 2 or self._y.shape[1] <= 3:
            raise ValueError("Theta: each series needs more than 3 observations")
        self._fitted_h = None
        return self

    def _key(self, h):
        return (int(h), self.season_length, self.decomposition_type, self.model, self.initial_smoothed,
                self.alpha, self.theta, self.numeric_mode)

    def _run(self, h):
        # the fit runs inside the forecast call; a repeat of the same call on
        # the same fitted series returns the stored answer (lane py-sequence)
        key = self._key(h)
        cached = getattr(self, "_fitted_h", None)
        if cached is not None and cached[0] == key:
            self.info_, self.model_ = cached[2].copy(), list(cached[3])
            return cached[1].copy()
        B, n = self._y.shape
        f = np.zeros((B, h), dtype=np.float32)
        info = np.zeros((B, 8), dtype=np.float32)
        mask = (int(self.initial_smoothed is not None) | 2 * int(self.alpha is not None)
                | 4 * int(self.theta is not None))
        fp = [0.0 if v is None else float(v) for v in (self.initial_smoothed, self.alpha, self.theta)]
        _backend.binding("_mojolearn_x_sequence", self.numeric_mode).theta(
            [self._y.ctypes.data, f.ctypes.data, info.ctypes.data],
            [B, n, int(h), self.season_length, -1 if self.model is None else _MODELS[self.model],
             0 if self.decomposition_type == "multiplicative" else 1, mask], fp)
        self.info_ = info
        self.model_ = _MODEL_NAMES[info[:, 4].astype(np.intp)].tolist()
        self._fitted_h = (key, f.copy(), info.copy(), list(self.model_))
        return f

    def predict(self, h, X=None, level=None):
        if level is not None:
            raise NotImplementedError("Theta: prediction intervals (level) are not implemented")
        f = self._run(int(h))
        return {"mean": f[0] if self._one else f}

    def forecast(self, y, h, X=None, X_future=None, level=None, fitted=False):
        if fitted:
            raise NotImplementedError("Theta: fitted values are not implemented")
        return self.fit(y, X).predict(h, level=level)


class Theta(AutoTheta):
    """statsforecast Theta (the standard theta model, STM)."""
    _MODEL = "STM"


class OptimizedTheta(AutoTheta):
    """statsforecast OptimizedTheta (OTM)."""
    _MODEL = "OTM"


class DynamicTheta(AutoTheta):
    """statsforecast DynamicTheta (DSTM)."""
    _MODEL = "DSTM"


class DynamicOptimizedTheta(AutoTheta):
    """statsforecast DynamicOptimizedTheta (DOTM)."""
    _MODEL = "DOTM"
