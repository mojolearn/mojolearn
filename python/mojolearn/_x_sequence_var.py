# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Vector autoregression (VAR), following statsmodels'
``statsmodels.tsa.vector_ar.var_model.VAR``. Fitted on the GPU."""
from ._optional_numpy import require_numpy
from ._mode import ParamsMixin
np = require_numpy('_x_sequence_var')

from . import _native_xseq_bridge

# Package-import connection: fit is ONE call into the compiled owner
# (bindings/xseq_forecast_native.mojo), which builds the result object too.
_XSEQ, _XSEQ_CONTEXT = _native_xseq_bridge.connect()


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
        return _XSEQ(self, "var_results_initialize", (endog, params, sigma_u, resid, k_ar, k_trend, numeric_mode), _XSEQ_CONTEXT)

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
        return _XSEQ(self, "var_forecast", (y, steps), _XSEQ_CONTEXT)


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

    # Every public method is one xseq_native_call; none needs a fitted
    # guard (the native forecast refuses an unfitted model itself), so no
    # `_mode` class-creation wrappers are installed.
    _mojolearn_native_public = True

    def __init__(self, endog, exog=None, numeric_mode=None):
        return _XSEQ(self, "var_initialize", (endog, exog, numeric_mode), _XSEQ_CONTEXT)

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
        return _XSEQ(self, "var_fit", (maxlags, method, ic, trend), _XSEQ_CONTEXT)


_native_xseq_bridge.register(xseq_var_results=VARResults, xseq_var_class=VAR)
