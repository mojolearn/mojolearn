# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Theta forecasters: Theta, OptimizedTheta, DynamicTheta,
DynamicOptimizedTheta and AutoTheta, following the models of the same names in
``statsforecast.models``. Each series is fitted on the GPU."""
from ._optional_numpy import require_numpy
from ._mode import ParamsMixin
np = require_numpy('_x_sequence_theta')

from . import _native_xseq_bridge

# Package-import connection: fit and predict are ONE call each into the
# compiled owner (bindings/xseq_forecast_native.mojo).
_XSEQ, _XSEQ_CONTEXT = _native_xseq_bridge.connect()

_MODELS = {"STM": 0, "OTM": 1, "DSTM": 2, "DOTM": 3}


# Fitted on the GPU one series per thread (`sequence/theta.mojo`): auto_theta's
# seasonal test and classical decomposition, the theta state-space models, their
# MSE objective minimised by statsforecast's Nelder-Mead, and the point forecast.
# Refusals are listed by name in sequence/NOT_IMPLEMENTED.tsv.
class AutoTheta(ParamsMixin):
    """Automatic theta model selection.

    Fits the four theta models (standard, optimized, dynamic, dynamic
    optimized) and keeps, per series, the one with the least in-sample MSE,
    or fits the single model named by ``model``. Follows statsforecast's
    ``AutoTheta``: a seasonality test and classical decomposition, the theta
    state-space models, and Nelder-Mead minimization of the MSE.

    ``fit`` takes one series of shape ``(n_obs,)`` or a batch of shape
    ``(batch_size, n_obs)``. The model is estimated when :meth:`predict` is
    called; repeating the same ``predict`` on the same fitted series returns
    the stored result.

    Parameters
    ----------
    season_length : int, default 1
        Number of observations per season. 1 means no seasonality.
    decomposition_type : {"multiplicative", "additive"}, default "multiplicative"
        Seasonal decomposition used when the series tests as seasonal.
    model : {"STM", "OTM", "DSTM", "DOTM"} or None, default None
        The theta model to fit. None tries all four and keeps the one with the
        least MSE per series.
    initial_smoothed : float or None, default None
        Fixes the initial smoothed level; None estimates it.
    alpha : float or None, default None
        Fixes the smoothing parameter; None estimates it.
    theta : float or None, default None
        Fixes the theta coefficient; None estimates it.
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'``
        or ``'fast'``); None uses the process default (see
        :func:`mojolearn.set_numeric_mode`). ``'fast'`` is available on Apple GPUs only.

    Attributes
    ----------
    model_ : list of str
        The theta model used for each series (``"STM"``, ``"OTM"``,
        ``"DSTM"`` or ``"DOTM"``). Set by :meth:`predict`.
    info_ : ndarray of shape (batch_size, 8), float32
        Per-series fit summary; column 4 is the index of the chosen model in
        ``("STM", "OTM", "DSTM", "DOTM")``. Set by :meth:`predict`.

    Notes
    -----
    Input must be float32; float64 raises TypeError. Each series needs more
    than 3 observations. Prediction intervals (``level``), fitted values,
    non-normal error distributions and exogenous regressors are not supported
    and raise NotImplementedError.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> t = np.arange(96, dtype=np.float32)
    >>> y = 10 + 0.1 * t + np.sin(2 * np.pi * t / 12).astype(np.float32)
    >>> mojolearn.AutoTheta(season_length=12).fit(y).predict(12)["mean"].shape
    (12,)
    """
    _MODEL = None

    # Every public method is one native call whose owner does the whole
    # lifecycle (fit bookkeeping and the NotFittedError of predict in
    # bindings/xseq_forecast_native.mojo), so no `_mode` class-creation wrappers are installed.
    _mojolearn_native_public = True

    def __init__(self, season_length=1, decomposition_type="multiplicative", model=None,
                 initial_smoothed=None, alpha=None, theta=None, numeric_mode=None):
        _XSEQ(self, "theta_initialize", (season_length, decomposition_type, model, initial_smoothed, alpha, theta, numeric_mode), _XSEQ_CONTEXT)

    def fit(self, y, X=None):
        """Store one series or a batch of series for forecasting.

        The theta model itself is estimated by :meth:`predict`.

        Parameters
        ----------
        y : array-like of shape (n_obs,) or (batch_size, n_obs)
            The series, float32, more than 3 observations each. A private
            copy is kept.
        X : None
            Exogenous regressors are not supported; anything other than None
            raises NotImplementedError.

        Returns
        -------
        self

        Raises
        ------
        TypeError
            If ``y`` is float64.
        ValueError
            If a series has 3 or fewer observations.
        """
        return _XSEQ(self, "theta_fit", (y, X), _XSEQ_CONTEXT)

    def predict(self, h, X=None, level=None):
        """Fit the theta model and forecast h steps ahead.

        Parameters
        ----------
        h : int
            Forecast horizon.
        X : None
            Unused.
        level : None
            Prediction intervals are not supported; anything other than None
            raises NotImplementedError.

        Returns
        -------
        dict
            ``{"mean": forecasts}``, shaped ``(h,)`` for a single series or
            ``(batch_size, h)`` for a batch.
        """
        return _XSEQ(self, "theta_predict", (h, X, level), _XSEQ_CONTEXT)

    def forecast(self, y, h, X=None, X_future=None, level=None, fitted=False):
        """Fit ``y`` and forecast ``h`` steps in one call.

        Equivalent to ``fit(y).predict(h)``. ``fitted=True`` (in-sample
        fitted values) raises NotImplementedError.

        Returns
        -------
        dict
            ``{"mean": forecasts}``, as returned by :meth:`predict`.
        """
        return _XSEQ(self, "theta_forecast", (y, h, X, X_future, level, fitted), _XSEQ_CONTEXT)


class Theta(AutoTheta):
    """The standard theta model (STM). Follows statsforecast's ``Theta``.

    ``fit`` takes one series of shape ``(n_obs,)`` or a batch of shape
    ``(batch_size, n_obs)``; the model is estimated when :meth:`predict` is
    called.

    Parameters
    ----------
    season_length : int, default 1
        Number of observations per season. 1 means no seasonality.
    decomposition_type : {"multiplicative", "additive"}, default "multiplicative"
        Seasonal decomposition used when the series tests as seasonal.
    model : ignored
        This class always uses the STM model; the argument is accepted for
        signature compatibility with :class:`AutoTheta`.
    initial_smoothed : float or None, default None
        Fixes the initial smoothed level; None estimates it.
    alpha : float or None, default None
        Fixes the smoothing parameter; None estimates it.
    theta : float or None, default None
        Fixes the theta coefficient; None estimates it.
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'``
        or ``'fast'``); None uses the process default (see
        :func:`mojolearn.set_numeric_mode`). ``'fast'`` is available on Apple GPUs only.

    Attributes
    ----------
    model_ : list of str
        The theta model used for each series (``"STM"``, ``"OTM"``,
        ``"DSTM"`` or ``"DOTM"``). Set by :meth:`predict`.
    info_ : ndarray of shape (batch_size, 8), float32
        Per-series fit summary; column 4 is the index of the chosen model in
        ``("STM", "OTM", "DSTM", "DOTM")``. Set by :meth:`predict`.

    Notes
    -----
    Input must be float32; float64 raises TypeError. Each series needs more
    than 3 observations. Prediction intervals (``level``), fitted values,
    non-normal error distributions and exogenous regressors are not supported
    and raise NotImplementedError.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> t = np.arange(96, dtype=np.float32)
    >>> y = 10 + 0.1 * t + np.sin(2 * np.pi * t / 12).astype(np.float32)
    >>> mojolearn.Theta(season_length=12).fit(y).predict(12)["mean"].shape
    (12,)
    """
    _MODEL = "STM"


class OptimizedTheta(AutoTheta):
    """The optimized theta model (OTM). Follows statsforecast's
    ``OptimizedTheta``.

    ``fit`` takes one series of shape ``(n_obs,)`` or a batch of shape
    ``(batch_size, n_obs)``; the model is estimated when :meth:`predict` is
    called.

    Parameters
    ----------
    season_length : int, default 1
        Number of observations per season. 1 means no seasonality.
    decomposition_type : {"multiplicative", "additive"}, default "multiplicative"
        Seasonal decomposition used when the series tests as seasonal.
    model : ignored
        This class always uses the OTM model; the argument is accepted for
        signature compatibility with :class:`AutoTheta`.
    initial_smoothed : float or None, default None
        Fixes the initial smoothed level; None estimates it.
    alpha : float or None, default None
        Fixes the smoothing parameter; None estimates it.
    theta : float or None, default None
        Fixes the theta coefficient; None estimates it.
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'``
        or ``'fast'``); None uses the process default (see
        :func:`mojolearn.set_numeric_mode`). ``'fast'`` is available on Apple GPUs only.

    Attributes
    ----------
    model_ : list of str
        The theta model used for each series (``"STM"``, ``"OTM"``,
        ``"DSTM"`` or ``"DOTM"``). Set by :meth:`predict`.
    info_ : ndarray of shape (batch_size, 8), float32
        Per-series fit summary; column 4 is the index of the chosen model in
        ``("STM", "OTM", "DSTM", "DOTM")``. Set by :meth:`predict`.

    Notes
    -----
    Input must be float32; float64 raises TypeError. Each series needs more
    than 3 observations. Prediction intervals (``level``), fitted values,
    non-normal error distributions and exogenous regressors are not supported
    and raise NotImplementedError.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> t = np.arange(96, dtype=np.float32)
    >>> y = 10 + 0.1 * t + np.sin(2 * np.pi * t / 12).astype(np.float32)
    >>> mojolearn.OptimizedTheta(season_length=12).fit(y).predict(12)["mean"].shape
    (12,)
    """
    _MODEL = "OTM"


class DynamicTheta(AutoTheta):
    """The dynamic standard theta model (DSTM). Follows statsforecast's
    ``DynamicTheta``.

    ``fit`` takes one series of shape ``(n_obs,)`` or a batch of shape
    ``(batch_size, n_obs)``; the model is estimated when :meth:`predict` is
    called.

    Parameters
    ----------
    season_length : int, default 1
        Number of observations per season. 1 means no seasonality.
    decomposition_type : {"multiplicative", "additive"}, default "multiplicative"
        Seasonal decomposition used when the series tests as seasonal.
    model : ignored
        This class always uses the DSTM model; the argument is accepted for
        signature compatibility with :class:`AutoTheta`.
    initial_smoothed : float or None, default None
        Fixes the initial smoothed level; None estimates it.
    alpha : float or None, default None
        Fixes the smoothing parameter; None estimates it.
    theta : float or None, default None
        Fixes the theta coefficient; None estimates it.
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'``
        or ``'fast'``); None uses the process default (see
        :func:`mojolearn.set_numeric_mode`). ``'fast'`` is available on Apple GPUs only.

    Attributes
    ----------
    model_ : list of str
        The theta model used for each series (``"STM"``, ``"OTM"``,
        ``"DSTM"`` or ``"DOTM"``). Set by :meth:`predict`.
    info_ : ndarray of shape (batch_size, 8), float32
        Per-series fit summary; column 4 is the index of the chosen model in
        ``("STM", "OTM", "DSTM", "DOTM")``. Set by :meth:`predict`.

    Notes
    -----
    Input must be float32; float64 raises TypeError. Each series needs more
    than 3 observations. Prediction intervals (``level``), fitted values,
    non-normal error distributions and exogenous regressors are not supported
    and raise NotImplementedError.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> t = np.arange(96, dtype=np.float32)
    >>> y = 10 + 0.1 * t + np.sin(2 * np.pi * t / 12).astype(np.float32)
    >>> mojolearn.DynamicTheta(season_length=12).fit(y).predict(12)["mean"].shape
    (12,)
    """
    _MODEL = "DSTM"


class DynamicOptimizedTheta(AutoTheta):
    """The dynamic optimized theta model (DOTM). Follows statsforecast's
    ``DynamicOptimizedTheta``.

    ``fit`` takes one series of shape ``(n_obs,)`` or a batch of shape
    ``(batch_size, n_obs)``; the model is estimated when :meth:`predict` is
    called.

    Parameters
    ----------
    season_length : int, default 1
        Number of observations per season. 1 means no seasonality.
    decomposition_type : {"multiplicative", "additive"}, default "multiplicative"
        Seasonal decomposition used when the series tests as seasonal.
    model : ignored
        This class always uses the DOTM model; the argument is accepted for
        signature compatibility with :class:`AutoTheta`.
    initial_smoothed : float or None, default None
        Fixes the initial smoothed level; None estimates it.
    alpha : float or None, default None
        Fixes the smoothing parameter; None estimates it.
    theta : float or None, default None
        Fixes the theta coefficient; None estimates it.
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'``
        or ``'fast'``); None uses the process default (see
        :func:`mojolearn.set_numeric_mode`). ``'fast'`` is available on Apple GPUs only.

    Attributes
    ----------
    model_ : list of str
        The theta model used for each series (``"STM"``, ``"OTM"``,
        ``"DSTM"`` or ``"DOTM"``). Set by :meth:`predict`.
    info_ : ndarray of shape (batch_size, 8), float32
        Per-series fit summary; column 4 is the index of the chosen model in
        ``("STM", "OTM", "DSTM", "DOTM")``. Set by :meth:`predict`.

    Notes
    -----
    Input must be float32; float64 raises TypeError. Each series needs more
    than 3 observations. Prediction intervals (``level``), fitted values,
    non-normal error distributions and exogenous regressors are not supported
    and raise NotImplementedError.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> t = np.arange(96, dtype=np.float32)
    >>> y = 10 + 0.1 * t + np.sin(2 * np.pi * t / 12).astype(np.float32)
    >>> mojolearn.DynamicOptimizedTheta(season_length=12).fit(y).predict(12)["mean"].shape
    (12,)
    """
    _MODEL = "DOTM"


_native_xseq_bridge.register(xseq_theta_codes=_MODELS, xseq_theta_models=tuple(_MODELS))

_native_xseq_bridge.register(xseq_theta_classes=((AutoTheta, None), (Theta, "STM"), (OptimizedTheta, "OTM"), (DynamicTheta, "DSTM"), (DynamicOptimizedTheta, "DOTM")))
