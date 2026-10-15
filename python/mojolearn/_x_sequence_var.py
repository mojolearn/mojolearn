# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Vector autoregression (VAR), following statsmodels'
``statsmodels.tsa.vector_ar.var_model.VAR``. Fitted on the GPU."""
from ._optional_numpy import require_numpy
from ._mode import ParamsMixin
np = require_numpy('_x_sequence_var')

from . import _backend


class VARResults:
    """Result of :meth:`VAR.fit`.

    Attributes
    ----------
    params : ndarray of shape (k_trend + neqs * k_ar, neqs), float32
        Stacked coefficients: the intercept row (if any), then the lag
        coefficients, one column per equation.
    coefs : ndarray of shape (k_ar, neqs, neqs), float32
        Lag coefficient matrices; ``coefs[i]`` multiplies ``y[t - i - 1]``.
    intercept : ndarray of shape (neqs,), float32
        Intercepts (zeros when ``trend="n"``).
    sigma_u : ndarray of shape (neqs, neqs), float32
        Residual covariance.
    resid : ndarray of shape (nobs, neqs), float32
        Residuals.
    fittedvalues : ndarray of shape (nobs, neqs), float32
        In-sample fitted values.
    k_ar : int
        Lag order.
    k_trend : int
        1 with an intercept, 0 without.
    neqs : int
        Number of variables.
    nobs : int
        Number of observations used, ``n_obs - k_ar``.
    df_resid : int
        Residual degrees of freedom.
    """

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
        """Forecast ``steps`` ahead from the last ``k_ar`` rows of ``y``.

        Parameters
        ----------
        y : array-like of shape (n, neqs), float32
            History to start from; at least ``k_ar`` rows. float64 raises
            TypeError.
        steps : int
            Number of steps to forecast.

        Returns
        -------
        ndarray of shape (steps, neqs), float32
        """
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


# The arithmetic is `sequence/vecar.mojo` (a scaled Cholesky solve of the
# normal equations in float32, where statsmodels calls lstsq in float64).
class VAR(ParamsMixin):
    """Vector autoregression estimated by OLS.

    Fits one least-squares regression per equation on the shared lagged
    design ``[1, y[t-1], ..., y[t-p]]``, following statsmodels' ``VAR``.
    The normal equations are solved in float32 by Cholesky, where statsmodels
    uses a float64 least-squares solve.

    Parameters
    ----------
    endog : array-like of shape (n_obs, n_variables)
        The multivariate series, float32. float64 raises TypeError.
    exog : None
        Exogenous variables are not supported; anything other than None
        raises NotImplementedError.
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'``
        or ``'fast'``); None uses the process default (see
        :func:`mojolearn.set_numeric_mode`). ``'fast'`` is available on Apple GPUs only.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> y = np.random.default_rng(0).standard_normal((200, 3)).cumsum(axis=0).astype(np.float32)
    >>> res = mojolearn.VAR(y).fit(maxlags=2)
    >>> res.forecast(y, steps=5).shape
    (5, 3)
    """

    def __init__(self, endog, exog=None, numeric_mode=None):
        if exog is not None:
            raise NotImplementedError("VAR: exog is not implemented")
        y = np.asarray(endog)
        if y.dtype == np.float64:
            raise TypeError("VAR: float64 is refused; pass float32 (no float64 on the device)")
        y = np.ascontiguousarray(y, dtype=np.float32)
        if y.ndim != 2 or y.shape[1] < 1:
            raise ValueError("VAR: endog must be (n_obs, n_variables)")
        self.endog = endog  # verbatim, as get_params / clone report it
        self.exog = exog
        self._endog = y
        self.neqs = int(y.shape[1])
        self.numeric_mode = numeric_mode

    def fit(self, maxlags=1, method="ols", ic=None, trend="c"):
        """Estimate the VAR with a fixed lag order.

        Parameters
        ----------
        maxlags : int, default 1
            The lag order ``p`` (used as is; there is no order selection).
        method : {"ols"}, default "ols"
            Estimation method.
        ic : None
            Information-criterion order selection is not supported; anything
            other than None raises NotImplementedError.
        trend : {"c", "n"}, default "c"
            ``"c"`` adds an intercept, ``"n"`` none. ``"ct"`` and ``"ctt"``
            raise NotImplementedError.

        Returns
        -------
        VARResults

        Raises
        ------
        numpy.linalg.LinAlgError
            If the lagged design is rank deficient.
        """
        if ic is not None:
            raise NotImplementedError("VAR.fit: information-criterion order selection (ic=) is not implemented")
        if method != "ols":
            raise ValueError("VAR.fit: method must be 'ols'")
        if trend not in ("c", "n"):
            raise NotImplementedError(f"VAR.fit: trend={trend!r} is not implemented ('c' and 'n' are)")
        p = int(maxlags)
        kt = 1 if trend == "c" else 0
        n, K = self._endog.shape
        m = kt + K * p
        params = np.zeros((m, K), dtype=np.float32)
        sigma = np.zeros((K, K), dtype=np.float32)
        resid = np.zeros((max(n - p, 1), K), dtype=np.float32)
        code = _backend.binding("_mojolearn_x_sequence", self.numeric_mode).var_fit(
            [self._endog.ctypes.data, params.ctypes.data, sigma.ctypes.data, resid.ctypes.data], [n, K, p, kt])
        if int(code):
            raise np.linalg.LinAlgError(  # glue: raises the numpy error class
                f"VAR.fit: the lagged design is rank deficient (Cholesky pivot of column {int(code) - 1})")
        return VARResults(self._endog, params, sigma, resid, p, kt, self.numeric_mode)
