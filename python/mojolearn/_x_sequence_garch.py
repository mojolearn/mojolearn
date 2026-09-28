# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""GARCH(p, o, q) volatility models, the arch package's
(`arch_model(y, mean='Constant'|'Zero', vol='GARCH', p, o, q, dist='normal')`),
fitted on the GPU one series per thread (`sequence/garch.mojo`): arch's
variance recursion, backcast, variance bounds, starting-value grid and
Gaussian likelihood, maximised by statsforecast's Nelder-Mead inside arch's
bounds. `fit(y)` takes one series or a batch `(batch_size, n_obs)`.

Attributes after fit: `params_` (mu, omega, alpha[1..p], gamma[1..o],
beta[1..q]), `loglikelihood_`, `conditional_volatility_`. `forecast(h)`
returns the analytic variance forecasts. Refused by name: distributions
other than normal, other mean models (AR, HAR, LS), power != 2
(TARCH/APARCH), rescaling, float64 input."""
import numpy as np

from . import _backend


class GARCH:
    def __init__(self, p=1, o=0, q=1, mean="Constant", dist="normal", power=2.0, numeric_mode=None):
        if mean not in ("Constant", "Zero"):
            raise NotImplementedError(f"GARCH: mean={mean!r} is not implemented ('Constant', 'Zero')")
        if dist != "normal":
            raise NotImplementedError(f"GARCH: dist={dist!r} is not implemented ('normal')")
        if power != 2.0:
            raise NotImplementedError("GARCH: power != 2 is not implemented")
        self.p, self.o, self.q = int(p), int(o), int(q)
        if min(self.p, self.o, self.q) < 0 or self.p + self.o + self.q < 1:
            raise ValueError("GARCH: p, o, q >= 0 with at least one term")
        if self.o > 0 and self.p == 0:
            raise ValueError("GARCH: o > 0 needs p > 0 (arch refuses it too)")
        self.mean, self.dist, self.power = mean, dist, power
        self.numeric_mode = numeric_mode

    def fit(self, y, horizon=1):
        y = np.asarray(y)
        if y.dtype == np.float64:
            raise TypeError("GARCH: float64 is refused; pass float32")
        self._one = y.ndim == 1
        Y = np.ascontiguousarray(y, dtype=np.float32).reshape(1, -1) if self._one else \
            np.ascontiguousarray(y, dtype=np.float32)
        if Y.ndim != 2 or Y.shape[1] < 10:
            raise ValueError("GARCH: each series needs at least 10 observations")
        bad = np.flatnonzero(~np.isfinite(Y.ravel()))
        if bad.size:
            raise ValueError(f"GARCH: y holds a non-finite value at flat index {int(bad[0])}")
        B, n = Y.shape
        h = max(1, int(horizon))
        k = 1 + self.p + self.o + self.q
        params = np.zeros((B, 1 + k), dtype=np.float32)
        info = np.zeros((B, 4), dtype=np.float32)
        sigma = np.zeros((B, n), dtype=np.float32)
        fc = np.zeros((B, h), dtype=np.float32)
        ip = [B, n, h, self.p, self.o, self.q, int(self.mean == "Constant")]
        stall = getattr(self, "_fast_stall", None)    # (iterations, relative drop): the FAST stop's
        if stall is not None:                          # quality sweep (tools/sequence_quality.py)
            ip = ip + [int(stall[0]), int(round(float(stall[1]) * 1e9))]
        _backend.binding("_mojolearn_x_sequence", self.numeric_mode).garch(
            [Y.ctypes.data, params.ctypes.data, info.ctypes.data, sigma.ctypes.data, fc.ctypes.data], ip)
        self._y = Y
        pick = (lambda a: a[0]) if self._one else (lambda a: a)
        self.params_ = pick(params)
        self.loglikelihood_ = pick(info[:, 0])
        self.n_iter_ = pick(info[:, 1].astype(np.int64))  # Nelder-Mead iterations, both runs
        self.conditional_volatility_ = pick(sigma)
        self._fc = fc
        return self

    def forecast(self, horizon=1):
        """(horizon,) (or (batch, horizon)) conditional variance forecasts."""
        h = int(horizon)
        if h > self._fc.shape[1]:
            y = self._y[0] if self._one else self._y
            self.fit(y, horizon=h)
        f = self._fc[:, :h]
        return f[0] if self._one else f
