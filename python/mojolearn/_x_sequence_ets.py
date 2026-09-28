# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""ETS with an optionally damped additive trend and an optional additive or
multiplicative season, statsforecast's `AutoETS(season_length=m,
model=..., damped=...)` for the models E T S with E in {A, M}, T in {N, A}
(damped or not) and S in {N, A, M}, fitted on the GPU one series per
thread (`sequence/ets.mojo`): likelihood over smoothing parameters and
initial states by statsforecast's Nelder-Mead. `DampedETS` is AAN with
damped=True (Holt's damped trend). `fit(y)` takes one series or a batch
`(batch_size, n_obs)`; `predict(h)` returns {"mean": ...}.

The reference's model checks are kept, in its words: a seasonal letter
with season_length 1 is "Nonseasonal data"; additive errors with a
multiplicative season are a "Forbidden model combination"; multiplicative
errors or season need positive data; n <= (parameter count) + 4 is the
reference's "tiny datasets" refusal; a series no longer than
season_length is fitted without a season, as there.

Refused by name: multiplicative trend, 'Z' (automatic) letters,
prediction intervals, fitted values, float64."""
import numpy as np

from . import _backend


class ETS:
    def __init__(self, season_length=1, model="AAN", damped=True, alpha=None, beta=None, phi=None,
                 gamma=None, numeric_mode=None):
        model = str(model)
        if len(model) != 3:
            raise ValueError("ETS: model is three letters, error / trend / season")
        e, tr, s = model
        if "Z" in model:
            raise NotImplementedError("ETS: automatic ('Z') model selection is not implemented")
        if tr == "M":
            raise NotImplementedError("ETS: multiplicative trend is not implemented")
        if e not in "AM" or tr not in "NA" or s not in "NAM":
            raise ValueError(f"ETS: invalid model {model!r} (error A|M, trend N|A, season N|A|M)")
        if damped and tr == "N":
            raise ValueError("ETS: a damped model needs a trend")
        season_length = int(season_length)
        if s != "N" and season_length == 1:
            raise ValueError("ETS: Nonseasonal data (a seasonal model needs season_length > 1)")
        if s != "N" and season_length > 58:
            raise NotImplementedError("ETS: season_length above 58 is not implemented (Nelder-Mead over at most 63 coordinates)")
        if e == "A" and s == "M":
            raise ValueError("ETS: Forbidden model combination (additive errors with a multiplicative season)")
        self.season_length, self.model, self.damped = season_length, model, bool(damped)
        self.alpha, self.beta, self.phi, self.gamma = alpha, beta, phi, gamma
        self.numeric_mode = numeric_mode

    def fit(self, y, X=None):
        if X is not None:
            raise NotImplementedError("ETS: exogenous regressors are not implemented")
        y = np.asarray(y)
        if y.dtype == np.float64:
            raise TypeError("ETS: float64 is refused; pass float32")
        self._one = y.ndim == 1
        # a private copy: the fitted series cannot change under a stored forecast
        self._y = np.array(y, dtype=np.float32, order="C", copy=True).reshape(1, -1) if self._one else \
            np.array(y, dtype=np.float32, order="C", copy=True)
        if self._y.ndim != 2 or self._y.shape[1] < 4:
            raise ValueError("ETS: each series needs at least 4 observations")
        e, tr, s = self.model
        n = self._y.shape[1]
        m = self.season_length
        if s != "N" and n <= m:
            s = "N"                                   # the reference drops the season (ets_f)
        if (e == "M" or s == "M") and not np.all(self._y > 0):
            raise ValueError("ETS: Inappropriate model for data with negative or zero values")
        npars = 2 + 2 * (tr != "N") + 2 * (s != "N") + int(self.damped)
        if n <= npars + 4:
            raise NotImplementedError("ETS: tiny datasets (n <= parameter count + 4) are not implemented")
        n_smooth = (self.alpha is None) + (tr != "N" and self.beta is None) + \
            (s != "N" and self.gamma is None) + (self.damped and self.phi is None)
        n_par = n_smooth + 1 + (tr != "N") + (m - 1 if s != "N" else 0)
        if n_par >= n - 1:
            raise ValueError(f"ETS: {n_par} parameters for {n} observations (the reference needs n > parameters + 1)")
        self._season = s
        self._last = None
        return self

    def predict(self, h, X=None, level=None):
        if level is not None:
            raise NotImplementedError("ETS: prediction intervals are not implemented")
        B, n = self._y.shape
        s = self._season
        m = self.season_length if s != "N" else 1
        f = np.zeros((B, int(h)), dtype=np.float32)
        info = np.zeros((B, 10), dtype=np.float32)
        ss = np.zeros((B, m), dtype=np.float32)
        mask = int(self.alpha is not None) | 2 * int(self.beta is not None) | 4 * int(self.phi is not None) \
            | 8 * int(self.gamma is not None)
        fp = [0.0 if v is None else float(v) for v in (self.alpha, self.beta, self.phi, self.gamma)]
        ip = [B, n, int(h), "AM".index(self.model[0]), "NA".index(self.model[1]), int(self.damped), mask,
              "NAM".index(s), m]
        stall = getattr(self, "_fast_stall", None)    # (iterations, relative drop): the FAST stop's
        if stall is not None:                          # quality sweep (tools/sequence_quality.py)
            ip, fp = ip + [int(stall[0])], fp + [float(stall[1])]
        # the fit runs inside the forecast call; a repeat of the same call on
        # the same fitted series returns the stored answer (lane py-sequence)
        key = (tuple(ip), tuple(fp), self.numeric_mode)
        last = getattr(self, "_last", None)
        if last is not None and last[0] == key:
            f, info, ss = (a.copy() for a in last[1:])
        else:
            _backend.binding("_mojolearn_x_sequence", self.numeric_mode).ets(
                [self._y.ctypes.data, f.ctypes.data, info.ctypes.data, ss.ctypes.data], ip, fp)
            self._last = (key, f.copy(), info.copy(), ss.copy())
        self.info_ = info
        if s != "N":
            self.seasonal_states_ = ss[0] if self._one else ss
        return {"mean": f[0] if self._one else f}

    def forecast(self, y, h, X=None, X_future=None, level=None, fitted=False):
        if fitted:
            raise NotImplementedError("ETS: fitted values are not implemented")
        return self.fit(y).predict(h, level=level)


class DampedETS(ETS):
    """Holt's damped additive trend, ETS(A,Ad,N) (or error='M')."""

    def __init__(self, season_length=1, error="A", alpha=None, beta=None, phi=None, numeric_mode=None):
        super().__init__(season_length, error + "AN", True, alpha, beta, phi, numeric_mode=numeric_mode)
