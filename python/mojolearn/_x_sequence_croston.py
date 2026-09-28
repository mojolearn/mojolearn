# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""statsforecast's Croston forecasters for intermittent demand
(CrostonClassic, CrostonOptimized, CrostonSBA), one series per GPU thread
(`sequence/croston.mojo`). `fit(y)` takes one series or a batch
`(batch_size, n_obs)`; `predict(h)` returns {"mean": ...}, the flat forecast
repeated h times. Refused: prediction intervals, fitted values, float64."""
from ._optional_numpy import require_numpy
np = require_numpy('_x_sequence_croston')

from . import _backend


class _Croston:
    _VARIANT = 0

    def __init__(self, alias=None, numeric_mode=None):
        self.alias = alias
        self.numeric_mode = numeric_mode

    def fit(self, y, X=None):
        if X is not None:
            raise NotImplementedError("Croston: exogenous regressors are not supported (nor in the reference)")
        y = np.asarray(y)
        if y.dtype == np.float64:
            raise TypeError("Croston: float64 is refused; pass float32")
        self._one = y.ndim == 1
        self._y = np.ascontiguousarray(y, dtype=np.float32).reshape(1, -1) if self._one else \
            np.ascontiguousarray(y, dtype=np.float32)
        if self._y.ndim != 2 or self._y.shape[1] < 1:
            raise ValueError("Croston: y must hold at least one observation per series")
        B, n = self._y.shape
        self.mean_ = np.zeros(B, dtype=np.float32)
        _backend.binding("_mojolearn_x_sequence", self.numeric_mode).croston(
            [self._y.ctypes.data, self.mean_.ctypes.data], [B, n, self._VARIANT])
        return self

    def predict(self, h, X=None, level=None):
        if level is not None:
            raise NotImplementedError("Croston: prediction intervals are not implemented")
        f = np.repeat(self.mean_[:, None], int(h), axis=1)
        return {"mean": f[0] if self._one else f}

    def forecast(self, y, h, X=None, X_future=None, level=None, fitted=False):
        if fitted:
            raise NotImplementedError("Croston: fitted values are not implemented")
        return self.fit(y).predict(h, level=level)


class CrostonClassic(_Croston):
    """SES (alpha 0.1) of demands and intervals, their ratio."""
    _VARIANT = 0


class CrostonOptimized(_Croston):
    """As CrostonClassic with each alpha the golden-section optimum on [0.1, 0.3]."""
    _VARIANT = 1


class CrostonSBA(_Croston):
    """Syntetos-Boylan approximation: CrostonClassic times 0.95."""
    _VARIANT = 2
