# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Non-seasonal ETS with an optionally damped additive trend, statsforecast's
`AutoETS(model=..., damped=...)` for the models ANN, AAN, MNN, MAN (and their
damped forms), fitted on the GPU one series per thread (`sequence/ets.mojo`):
likelihood over smoothing parameters and initial states by statsforecast's
Nelder-Mead. `DampedETS` is AAN with damped=True (Holt's damped trend).
`fit(y)` takes one series or a batch `(batch_size, n_obs)`; `predict(h)`
returns {"mean": ...}.

Refused by name: seasonal components (season letter other than N),
multiplicative trend, 'Z' (automatic) letters, prediction intervals, fitted
values, float64."""
import numpy as np

from . import _backend


class ETS:
    def __init__(self, season_length=1, model="AAN", damped=True, alpha=None, beta=None, phi=None,
                 numeric_mode=None):
        model = str(model)
        if len(model) != 3:
            raise ValueError("ETS: model is three letters, error / trend / season")
        e, tr, s = model
        if "Z" in model:
            raise NotImplementedError("ETS: automatic ('Z') model selection is not implemented")
        if e not in "AM" or tr not in "NA" or s != "N":
            raise NotImplementedError(f"ETS: model {model!r} is not implemented (error A|M, trend N|A, season N)")
        if damped and tr == "N":
            raise ValueError("ETS: a damped model needs a trend")
        self.season_length, self.model, self.damped = int(season_length), model, bool(damped)
        self.alpha, self.beta, self.phi = alpha, beta, phi
        self.numeric_mode = numeric_mode

    def fit(self, y, X=None):
        if X is not None:
            raise NotImplementedError("ETS: exogenous regressors are not implemented")
        y = np.asarray(y)
        if y.dtype == np.float64:
            raise TypeError("ETS: float64 is refused; pass float32")
        self._one = y.ndim == 1
        self._y = np.ascontiguousarray(y, dtype=np.float32).reshape(1, -1) if self._one else \
            np.ascontiguousarray(y, dtype=np.float32)
        if self._y.ndim != 2 or self._y.shape[1] < 4:
            raise ValueError("ETS: each series needs at least 4 observations")
        return self

    def predict(self, h, X=None, level=None):
        if level is not None:
            raise NotImplementedError("ETS: prediction intervals are not implemented")
        B, n = self._y.shape
        f = np.zeros((B, int(h)), dtype=np.float32)
        info = np.zeros((B, 8), dtype=np.float32)
        mask = int(self.alpha is not None) | 2 * int(self.beta is not None) | 4 * int(self.phi is not None)
        fp = [0.0 if v is None else float(v) for v in (self.alpha, self.beta, self.phi)]
        _backend.binding("_mojolearn_x_sequence", self.numeric_mode).ets(
            [self._y.ctypes.data, f.ctypes.data, info.ctypes.data],
            [B, n, int(h), "AM".index(self.model[0]), "NA".index(self.model[1]), int(self.damped), mask], fp)
        self.info_ = info
        return {"mean": f[0] if self._one else f}

    def forecast(self, y, h, X=None, X_future=None, level=None, fitted=False):
        if fitted:
            raise NotImplementedError("ETS: fitted values are not implemented")
        return self.fit(y).predict(h, level=level)


class DampedETS(ETS):
    """Holt's damped additive trend, ETS(A,Ad,N) (or error='M')."""

    def __init__(self, season_length=1, error="A", alpha=None, beta=None, phi=None, numeric_mode=None):
        super().__init__(season_length, error + "AN", True, alpha, beta, phi, numeric_mode)
