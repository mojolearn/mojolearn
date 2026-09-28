# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""VAR, the vector autoregression of statsmodels
(`statsmodels.tsa.vector_ar.var_model.VAR`): `VAR(endog).fit(maxlags=p)`
estimates one OLS regression per equation on the shared lagged design
[1, y_{t-1}, ..., y_{t-p}] and returns params, coefs, intercept, sigma_u,
resid and fittedvalues; `forecast(y, steps)` runs the recursion. The
arithmetic is `sequence/vecar.mojo` (a scaled Cholesky solve of the normal
equations in float32, where statsmodels calls lstsq in float64).

Refused by name: ic-based order selection (`ic=`), trend 'ct' and 'ctt',
exog, float64 input, a rank-deficient design (the solve reports it)."""
from ._optional_numpy import require_numpy
np = require_numpy('_x_sequence_var')

from . import _backend


class VARResults:
    def __init__(self, endog, params, sigma_u, resid, k_ar, k_trend, numeric_mode):
        self.endog = endog
        self.params = params
        self.sigma_u = sigma_u
        self.resid = resid
        self.k_ar = k_ar
        self.k_trend = k_trend
        self.neqs = int(endog.shape[1])
        self.nobs = int(endog.shape[0] - k_ar)
        self.df_resid = self.nobs - (self.neqs * k_ar + k_trend)
        K = self.neqs
        self.intercept = params[0].copy() if k_trend else np.zeros(K, dtype=np.float32)
        self.coefs = np.ascontiguousarray(params[k_trend:].reshape(k_ar, K, K).transpose(0, 2, 1))
        self.fittedvalues = (endog[k_ar:] - resid).astype(np.float32)
        self._numeric_mode = numeric_mode

    def forecast(self, y, steps):
        """(steps, K): the recursion from the last k_ar rows of `y`."""
        y = np.asarray(y)
        if y.dtype == np.float64:
            raise TypeError("VARResults.forecast: float64 is refused; pass float32")
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float32)[-self.k_ar:])
        if y.shape != (self.k_ar, self.neqs):
            raise ValueError(f"forecast: y needs at least {self.k_ar} rows of {self.neqs} columns")
        out = np.zeros((int(steps), self.neqs), dtype=np.float32)
        p = np.ascontiguousarray(self.params, dtype=np.float32)
        _backend.binding("_mojolearn_x_sequence", self._numeric_mode).var_forecast(
            [y.ctypes.data, p.ctypes.data, out.ctypes.data], [self.neqs, self.k_ar, self.k_trend, int(steps)])
        return out


class VAR:
    def __init__(self, endog, exog=None, numeric_mode=None):
        if exog is not None:
            raise NotImplementedError("VAR: exog is not implemented")
        y = np.asarray(endog)
        if y.dtype == np.float64:
            raise TypeError("VAR: float64 is refused; pass float32 (no float64 on the device)")
        y = np.ascontiguousarray(y, dtype=np.float32)
        if y.ndim != 2 or y.shape[1] < 1:
            raise ValueError("VAR: endog must be (n_obs, n_variables)")
        self.endog = y
        self.neqs = int(y.shape[1])
        self.numeric_mode = numeric_mode

    def fit(self, maxlags=1, method="ols", ic=None, trend="c"):
        if ic is not None:
            raise NotImplementedError("VAR.fit: information-criterion order selection (ic=) is not implemented")
        if method != "ols":
            raise ValueError("VAR.fit: method must be 'ols'")
        if trend not in ("c", "n"):
            raise NotImplementedError(f"VAR.fit: trend={trend!r} is not implemented ('c' and 'n' are)")
        p = int(maxlags)
        kt = 1 if trend == "c" else 0
        n, K = self.endog.shape
        m = kt + K * p
        params = np.zeros((m, K), dtype=np.float32)
        sigma = np.zeros((K, K), dtype=np.float32)
        resid = np.zeros((max(n - p, 1), K), dtype=np.float32)
        code = _backend.binding("_mojolearn_x_sequence", self.numeric_mode).var_fit(
            [self.endog.ctypes.data, params.ctypes.data, sigma.ctypes.data, resid.ctypes.data], [n, K, p, kt])
        if int(code):
            raise np.linalg.LinAlgError(
                f"VAR.fit: the lagged design is rank deficient (Cholesky pivot of column {int(code) - 1})")
        return VARResults(self.endog, params, sigma, resid, p, kt, self.numeric_mode)
