# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""AutoARIMA: the order search of cuML `python/cuml/cuml/tsa/auto_arima.pyx`
(`AutoARIMA.search`, `fit`, `predict`, `forecast`) over mojolearn's batched
`ARIMA` (`arima/`, the `_mojolearn_arima` binding) and the KPSS choice of `d`
(`tsa/`, `select_d`).

The search, as the reference's:
  1. `d` per series: the KPSS test at d = 0, 1, ... (`select_d`), unless one
     `d` is given; series are grouped by their `d`.
  2. For each group, every (p, q, P, Q, k) of the product of the options, in
     `itertools.product` order, skipping p + q + P + Q + k == 0, is fitted on
     the whole group as one batch; each series keeps the order of least
     information criterion, the FIRST such order on a tie (the reference's
     `_divide_by_min` is an argmin).
  3. `fit` refits every chosen order on its sub-batch.
The information criterion is batched_arima.cu `information_criterion`'s:
-2 loglike + 2N (aic), + 2N(N+1)/(n - N - 1) (aicc), log(n) N (bic), with
n = n_obs - d - s D and N the model's parameter count, computed in float64 on
the host from the fit's float32 log-likelihood.

Layout: mojolearn's ARIMA layout, `(batch_size, n_obs)`, one series per row
(cuML's AutoARIMA takes series in columns; transpose to call it the same way).
Refused by name: `seasonal_test="seas"` (statsmodels' STL, not a GPU path in
the reference either; pass `D` as one integer), `method` css / css-ml (this
ARIMA fits by `ml`), a `d` option list that is not 0..d_max, `truncate`,
`h` other than the default, prediction intervals (`level`)."""
import itertools

import numpy as np

from . import _portable_math

from ._arima_impl import ARIMA
from ._tsa_impl import select_d

_IC = ("aic", "aicc", "bic")


def _options(name, value, lo, hi):
    """The reference's `_parse_sequence`: an int or an iterable, clipped to
    [lo, hi]; empty is refused."""
    vals = [int(value)] if isinstance(value, (int, np.integer)) else [int(v) for v in value]
    vals = [v for v in vals if lo <= v <= hi]
    if not vals:
        raise ValueError(f"AutoARIMA: no valid option for {name} in [{lo}, {hi}]")
    return vals


class AutoARIMA:
    """Batched ARIMA order search. `endog` is `(batch_size, n_obs)` float32
    (or one 1-D series)."""

    #: The bindings the search runs, through `ARIMA` and `select_d` (named
    #: here so the lane map derives them and their CPU host bindings).
    _ARIMA_BINDING = "_mojolearn_arima"
    _TSA_BINDING = "_mojolearn_tsa"

    def __init__(self, endog):
        y = np.asarray(endog)
        if y.dtype == np.float64:
            raise TypeError("AutoARIMA: float64 is refused; pass float32 (no float64 on the device)")
        y = np.ascontiguousarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        if y.ndim != 2 or y.size == 0:
            raise ValueError("AutoARIMA: endog must be (batch_size, n_obs)")
        self.endog = y
        self.batch_size, self.n_obs = int(y.shape[0]), int(y.shape[1])
        self.models = []

    def search(self, s=None, d=range(3), D=range(2), p=range(1, 4), q=range(1, 4), P=range(3), Q=range(3),
               fit_intercept="auto", ic="aicc", test="kpss", seasonal_test="seas", h=1e-8, maxiter=20,
               method="auto", truncate=0):
        if ic not in _IC:
            raise ValueError(f"AutoARIMA: ic must be one of {_IC}")
        if test != "kpss":
            raise ValueError("AutoARIMA: test must be 'kpss', the reference's only stationarity test")
        if method not in ("auto", "ml"):
            raise NotImplementedError("AutoARIMA: this ARIMA fits by 'ml' only; css and css-ml are refused")
        if h != 1e-8 or truncate:
            raise NotImplementedError("AutoARIMA: h and truncate are not carried by mojolearn.ARIMA")
        s = int(s) if s else 0
        if s:
            if not isinstance(D, (int, np.integer)):
                raise NotImplementedError(
                    f"AutoARIMA: seasonal_test={seasonal_test!r} (statsmodels' STL in the reference) is "
                    "not implemented; pass D as one integer")
            D_ = int(D)
        else:
            D_ = 0
        d_opts = _options("d", d, 0, 2 - D_)
        p_opts = _options("p", p, 0, s - 1 if s else 4)
        q_opts = _options("q", q, 0, s - 1 if s else 4)
        P_opts = _options("P", P, 0, 4 if s else 0)
        Q_opts = _options("Q", Q, 0, 4 if s else 0)
        y = self.endog
        if len(d_opts) == 1:
            dser = np.full(self.batch_size, d_opts[0], dtype=np.int64)
        else:
            if d_opts != list(range(len(d_opts))):
                raise NotImplementedError("AutoARIMA: a d option list must be 0, 1, ..., d_max")
            # the whole batch in one call, (n_obs, n_series) (apple2: one call
            # per series was 2000 launches and waits, 2.4 s of the search on
            # an M4 Pro). select_d is a pure function of each series' bits
            # (tsa/checks/stationarity_check.mojo gates the batch composition
            # invariant), so every series' d is the one-series call's.
            dser = np.asarray(select_d(np.ascontiguousarray(y.T), D=D_, s=s, d_max=d_opts[-1]),
                              dtype=np.int64).reshape(-1)
        self.d_ = dser
        self.models, self._ids = [], []
        self.order_ = np.zeros((self.batch_size, 8), dtype=np.int64)
        self.ic_ = np.zeros(self.batch_size, dtype=np.float64)
        for d_ in sorted(set(dser.tolist())):
            ids = np.flatnonzero(dser == d_)
            sub = np.ascontiguousarray(y[ids])
            k_opts = ([1 if d_ + D_ <= 1 else 0] if fit_intercept == "auto"
                      else _options("k", fit_intercept, 0, 1))
            orders, ics = [], []
            for p_, q_, P_, Q_, k_ in itertools.product(p_opts, q_opts, P_opts, Q_opts, k_opts):
                if p_ + q_ + P_ + Q_ + k_ == 0:
                    continue
                s_ = s if (P_ + D_ + Q_) else 0
                m = ARIMA(order=(p_, d_, q_), seasonal_order=(P_, D_, Q_, s_),
                          trend="c" if k_ else "n", maxiter=maxiter).fit(sub)
                orders.append((p_, q_, P_, Q_, s_, k_))
                ics.append(self._ic(m, ic, d_, D_, s_))
            best = np.argmin(np.stack(ics, axis=1), axis=1)
            for i, (p_, q_, P_, Q_, s_, k_) in enumerate(orders):
                chosen = ids[best == i]
                if len(chosen) == 0:
                    continue
                self.models.append(((p_, d_, q_), (P_, D_, Q_, s_), k_))
                self._ids.append(chosen)
                self.order_[chosen] = [p_, d_, q_, P_, D_, Q_, s_, k_]
                self.ic_[chosen] = ics[i][best == i]
        self._fitted = [None] * len(self.models)
        return self

    def _ic(self, m, ic, d_, D_, s_):
        llf = np.asarray(m.llf_, dtype=np.float64)
        N = float(m.complexity_)
        n = float(self.n_obs - d_ - s_ * D_)
        base = -2.0 * llf
        if ic == "aic":
            return base + 2.0 * N
        if ic == "aicc":
            return base + 2.0 * N + 2.0 * N * (N + 1.0) / (n - N - 1.0)
        # the pinned log, not the host libm's (an argmin near a tie must not
        # flip between hosts)
        return base + _portable_math.log(n) * N

    def fit(self, h=1e-8, maxiter=1000, method="ml", truncate=0):
        if not self.models:
            raise RuntimeError("AutoARIMA: call search() before fit()")
        if h != 1e-8 or truncate or method != "ml":
            raise NotImplementedError("AutoARIMA.fit: method 'ml' with the default h only")
        for i, (order, sorder, k) in enumerate(self.models):
            self._fitted[i] = ARIMA(order=order, seasonal_order=sorder, trend="c" if k else "n",
                                    maxiter=maxiter).fit(np.ascontiguousarray(self.endog[self._ids[i]]))
        return self

    def _gather(self, fn, width):
        if any(m is None for m in self._fitted):
            raise RuntimeError("AutoARIMA: call fit() first")
        out = np.zeros((self.batch_size, width), dtype=np.float32)
        for m, ids in zip(self._fitted, self._ids):
            out[ids] = np.asarray(fn(m), dtype=np.float32).reshape(len(ids), width)
        return out

    def predict(self, start=0, end=None, level=None):
        if level is not None:
            raise NotImplementedError("AutoARIMA.predict: prediction intervals (level) are not implemented")
        end = self.n_obs if end is None else int(end)
        return self._gather(lambda m: m.predict(start, end), end - int(start))

    def forecast(self, nsteps, level=None):
        if level is not None:
            raise NotImplementedError("AutoARIMA.forecast: prediction intervals (level) are not implemented")
        return self._gather(lambda m: m.forecast(int(nsteps)), int(nsteps))
