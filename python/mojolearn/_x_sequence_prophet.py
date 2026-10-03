# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""ProphetForecaster: a Prophet-style forecaster (the model of
`prophet/forecaster.py` and `stan/prophet.stan`, linear growth): a
piecewise-linear trend with changepoints, Fourier seasonalities, holiday
regressors, additive or multiplicative, fitted by MAP on the GPU
(`sequence/prophet.mojo`, one series per thread for a batch sharing t).

Prophet's rules carried: y scaled by max|y| and t to [0, 1] over the
history; `n_changepoints` placed at `np.linspace(0, hist_size - 1, n + 1)`
of the first `changepoint_range` of the history, the first dropped; the
auto seasonalities (yearly 365.25 d order 10 when the history spans two
years, weekly 7 d order 3 when it spans two weeks at sub-weekly spacing,
daily 1 d order 4 when it spans two days at sub-daily spacing) and
`add_seasonality`; prior scales per feature; the linear initialisation.
Time is days (floats), or numpy datetime64 converted to days since 1970.

The fit is our float32 L-BFGS, not Stan's; parity with the prophet package
is at a tolerance (its fit is Stan). Refused by name
(sequence/NOT_IMPLEMENTED.tsv): logistic and flat growth, uncertainty
intervals and sampling (mcmc_samples, interval_width), extra regressors
beyond holiday indicator columns, conditional seasonalities, float64 y."""
from . import _portable_math as math

from ._optional_numpy import require_numpy
np = require_numpy('_x_sequence_prophet')

from . import _backend


def _bind(mode):
    return _backend.binding("_mojolearn_x_sequence", mode)


def _days(t, mode):
    """(days float64 (N,), info [status, span, least spacing]): the time axis
    in days, converted and checked in Mojo (`prophet_days`; numpy datetime64
    is handed over as int64 nanoseconds since 1970)."""
    t = np.asarray(t)
    if np.issubdtype(t.dtype, np.datetime64):
        src, kind = np.ascontiguousarray(t.astype("datetime64[ns]").view(np.int64)), 1
    else:
        src, kind = np.ascontiguousarray(t, dtype=np.float64).reshape(-1), 0
    src = src.reshape(-1)
    days = np.empty(len(src), dtype=np.float64)
    info = np.zeros(3, dtype=np.float64)
    if len(src):
        _bind(mode).prophet_days([src.ctypes.data, days.ctypes.data, info.ctypes.data], [len(src), kind])
    return days, info


class ProphetForecaster:
    def __init__(self, growth="linear", changepoints=None, n_changepoints=25, changepoint_range=0.8,
                 yearly_seasonality="auto", weekly_seasonality="auto", daily_seasonality="auto",
                 seasonality_mode="additive", seasonality_prior_scale=10.0, holidays_prior_scale=10.0,
                 changepoint_prior_scale=0.05, max_iter=2000, numeric_mode=None):
        if growth != "linear":
            raise NotImplementedError(f"ProphetForecaster: growth={growth!r} is not implemented ('linear')")
        if seasonality_mode not in ("additive", "multiplicative"):
            raise ValueError("seasonality_mode must be 'additive' or 'multiplicative'")
        self.growth = growth
        self.changepoints = changepoints
        self.n_changepoints = int(n_changepoints)
        self.changepoint_range = float(changepoint_range)
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.seasonality_prior_scale = float(seasonality_prior_scale)
        self.holidays_prior_scale = float(holidays_prior_scale)
        self.changepoint_prior_scale = float(changepoint_prior_scale)
        self.max_iter = int(max_iter)
        self.numeric_mode = numeric_mode
        self.seasonalities = {}

    def add_seasonality(self, name, period, fourier_order, prior_scale=None):
        if int(fourier_order) < 1 or float(period) <= 0:
            raise ValueError("add_seasonality: period > 0 and fourier_order >= 1")
        self.seasonalities[name] = dict(period=float(period), fourier_order=int(fourier_order),
                                        prior_scale=self.seasonality_prior_scale if prior_scale is None
                                        else float(prior_scale))
        return self

    def _auto(self, name, arg, default_order, ok):
        if name in self.seasonalities:
            return
        if arg == "auto":
            use, order = ok, default_order
        elif arg is True:
            use, order = True, default_order
        elif arg is False:
            use, order = False, 0
        else:
            use, order = True, int(arg)
        if use:
            self.seasonalities[name] = dict(period={"yearly": 365.25, "weekly": 7.0, "daily": 1.0}[name],
                                            fourier_order=order, prior_scale=self.seasonality_prior_scale)

    def _features(self, days):
        """(t scaled float32 (N,), frac float32 (N, max(ns, 1))): Prophet's
        time scaling and (t mod P) / P per seasonality, in Mojo
        (`prophet_features`: exact float64 fmod, one rounding to float32)."""
        periods = np.ascontiguousarray([s["period"] for s in self.seasonalities.values()] or [1.0],
                                       dtype=np.float64)
        ns = len(self.seasonalities)
        tsc = np.empty(len(days), dtype=np.float32)
        frac = np.empty((len(days), max(ns, 1)), dtype=np.float32)
        _bind(self.numeric_mode).prophet_features(
            [days.ctypes.data, periods.ctypes.data, tsc.ctypes.data, frac.ctypes.data], [len(days), ns],
            [float(self.start_), float(self.t_scale_)])
        return tsc, frac

    def fit(self, t, y, holidays=None):
        days, info = _days(t, self.numeric_mode)
        status, span, dmin = int(info[0]), float(info[1]), float(info[2])
        if status == 1:
            raise ValueError("ProphetForecaster: t must not hold NaN")
        if status == 2:
            raise ValueError("ProphetForecaster: t must be sorted ascending")
        y = np.asarray(y)
        if y.dtype == np.float64:
            raise TypeError("ProphetForecaster: float64 y is refused; pass float32")
        self._one = y.ndim == 1
        Y = np.ascontiguousarray(y, dtype=np.float32).reshape(1, -1) if self._one else \
            np.ascontiguousarray(y, dtype=np.float32)
        B, N = Y.shape
        if N != len(days) or N < 2:
            raise ValueError("ProphetForecaster: y and t lengths differ, or fewer than 2 points")
        from ._buffer import _native
        if int(_native("all_finite_f32")(Y.ctypes.data, Y.size)) != 1:
            raise ValueError("ProphetForecaster: y must be finite")
        self._auto("yearly", self.yearly_seasonality, 10, span >= 730)
        self._auto("weekly", self.weekly_seasonality, 3, span >= 14 and dmin < 7)
        self._auto("daily", self.daily_seasonality, 4, span >= 2 and dmin < 1)
        self.start_, self.t_scale_ = float(days[0]), (span if span > 0 else 1.0)
        tsc, frac_ = self._features(days)
        if self.changepoints is not None:
            user, _ = _days(self.changepoints, self.numeric_mode)
            n_user, hist, n_cp = len(user), 0, 0
        else:
            user, n_user = np.zeros(1, dtype=np.float64), -1
            hist = int(math.floor(N * self.changepoint_range))
            n_cp = max(min(self.n_changepoints, hist - 1), 0)
        cpt = np.zeros(max(n_user, n_cp, 1), dtype=np.float32)
        S = _bind(self.numeric_mode).prophet_changepoints(
            [days.ctypes.data, user.ctypes.data, cpt.ctypes.data], [N, n_user, hist, n_cp],
            [self.start_, self.t_scale_])
        if S < 0:
            raise ValueError("Changepoints must fall within training data.")
        self.changepoints_t_ = np.ascontiguousarray(cpt[:S])
        self._nh = 0 if holidays is None else np.asarray(holidays).reshape(N, -1).shape[1]
        H = np.ascontiguousarray(np.asarray(holidays, dtype=np.float32).reshape(N, -1)) if self._nh else \
            np.zeros((N, 1), np.float32)
        orders = np.asarray([s["fourier_order"] for s in self.seasonalities.values()], dtype=np.float32)
        sig = [s["prior_scale"] for s in self.seasonalities.values() for _ in range(2 * s["fourier_order"])]
        sig += [self.holidays_prior_scale] * self._nh
        self._K = len(sig)
        sig = np.ascontiguousarray(sig if sig else [1.0], dtype=np.float32)
        S = len(self.changepoints_t_)
        P = 3 + S + self._K
        self.params_ = np.zeros((B, P), dtype=np.float32)
        self.info_ = np.zeros((B, 4), dtype=np.float32)
        orders_ = orders if len(orders) else np.zeros(1, np.float32)
        _bind(self.numeric_mode).prophet_fit(
            [Y.ctypes.data, tsc.ctypes.data, frac_.ctypes.data, orders_.ctypes.data, H.ctypes.data,
             self.changepoints_t_.ctypes.data, sig.ctypes.data, self.params_.ctypes.data, self.info_.ctypes.data],
            [B, N, len(orders), self._nh, self._K, S, int(self.seasonality_mode == "multiplicative"),
             self.max_iter], [self.changepoint_prior_scale])
        # diagnostics from info_ (y_scale, objective, iterations, 0): views only
        self.objective_ = self.info_[:, 1]
        self.n_iter_ = self.info_[:, 2].astype(np.int64)
        return self

    def predict(self, t, holidays=None):
        """yhat (and `self.trend_`) at t: (M,) for one series, else (B, M)."""
        days, _ = _days(t, self.numeric_mode)
        M = len(days)
        tsc, frac_ = self._features(days)
        if self._nh:
            if holidays is None:
                raise ValueError("ProphetForecaster: the model has holiday columns; pass them for t")
            H = np.ascontiguousarray(np.asarray(holidays, dtype=np.float32).reshape(M, -1))
        else:
            H = np.zeros((M, 1), np.float32)
        orders = np.asarray([s["fourier_order"] for s in self.seasonalities.values()], dtype=np.float32)
        orders_ = orders if len(orders) else np.zeros(1, np.float32)
        B = self.params_.shape[0]
        yhat = np.zeros((B, M), dtype=np.float32)
        trend = np.zeros((B, M), dtype=np.float32)
        S = len(self.changepoints_t_)
        _bind(self.numeric_mode).prophet_predict(
            [self.params_.ctypes.data, self.info_.ctypes.data, tsc.ctypes.data, frac_.ctypes.data,
             orders_.ctypes.data, H.ctypes.data, self.changepoints_t_.ctypes.data, yhat.ctypes.data,
             trend.ctypes.data],
            [B, M, len(orders), self._nh, self._K, S, int(self.seasonality_mode == "multiplicative")])
        self.trend_ = trend[0] if self._one else trend
        return yhat[0] if self._one else yhat
