# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Croston forecasters for intermittent demand: CrostonClassic,
CrostonOptimized and CrostonSBA, following statsforecast's models of the same
names. Each series is fitted on the GPU."""
from ._optional_numpy import require_numpy
np = require_numpy('_x_sequence_croston')

from . import _backend


class _Croston:
    """Shared fit/predict for the Croston variants (one series per GPU thread,
    `sequence/croston.mojo`)."""
    _VARIANT = 0

    def __init__(self, alias=None, numeric_mode=None):
        self.alias = alias
        self.numeric_mode = numeric_mode

    def fit(self, y, X=None):
        """Fit one series or a batch of series.

        Parameters
        ----------
        y : array-like of shape (n_obs,) or (batch_size, n_obs)
            The demand series, float32. One row per series for a batch.
        X : None
            Exogenous regressors are not supported; anything other than
            None raises NotImplementedError.

        Returns
        -------
        self
            The fitted model, with ``mean_`` set.

        Raises
        ------
        TypeError
            If ``y`` is float64.
        ValueError
            If a series has no observations.
        """
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
        """Forecast h steps ahead.

        The Croston forecast is flat: each series' fitted demand rate,
        repeated ``h`` times.

        Parameters
        ----------
        h : int
            Forecast horizon, ``h >= 0``.
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
        if level is not None:
            raise NotImplementedError("Croston: prediction intervals are not implemented")
        h = int(h)
        if h < 0:
            raise ValueError("Croston: h must be >= 0")
        B = self.mean_.shape[0]
        f = np.empty((B, h), dtype=np.float32)
        # every row its series' mean, in Mojo (`croston_forecast`)
        if h:
            _backend.binding("_mojolearn_x_sequence", self.numeric_mode).croston_forecast(
                [self.mean_.ctypes.data, f.ctypes.data], [B, h])
        return {"mean": f[0] if self._one else f}

    def forecast(self, y, h, X=None, X_future=None, level=None, fitted=False):
        """Fit ``y`` and forecast ``h`` steps in one call.

        Equivalent to ``fit(y).predict(h)``. ``fitted=True`` (in-sample
        fitted values) raises NotImplementedError.

        Returns
        -------
        dict
            ``{"mean": forecasts}``, as returned by :meth:`predict`.
        """
        if fitted:
            raise NotImplementedError("Croston: fitted values are not implemented")
        return self.fit(y).predict(h, level=level)


class CrostonClassic(_Croston):
    """Croston's method for intermittent demand.

    Simple exponential smoothing (alpha 0.1) of the nonzero demands and of
    the intervals between them; the forecast is their ratio. Follows
    statsforecast's ``CrostonClassic``. ``fit`` takes one series or a batch
    of shape ``(batch_size, n_obs)``.

    Parameters
    ----------
    alias : str or None, default None
        A display name for the model, stored as ``alias``.
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'``, ``'deterministic'``
        or ``'fast'``); None uses the process default (see
        :func:`mojolearn.set_numeric_mode`).

    Attributes
    ----------
    mean_ : ndarray of shape (batch_size,), float32
        The fitted demand rate of each series; the point forecast.

    Notes
    -----
    Input must be float32; float64 raises TypeError. Prediction intervals,
    fitted values and exogenous regressors are not supported and raise
    NotImplementedError.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> y = (rng.random(120) < 0.3).astype(np.float32) * rng.integers(1, 10, 120).astype(np.float32)
    >>> mojolearn.CrostonClassic().fit(y).predict(12)["mean"].shape
    (12,)
    """
    _VARIANT = 0


class CrostonOptimized(_Croston):
    """Croston's method with optimized smoothing parameters.

    As :class:`CrostonClassic`, but each smoothing parameter (for demands and
    for intervals) is chosen by golden-section search on [0.1, 0.3]. Follows
    statsforecast's ``CrostonOptimized``. ``fit`` takes one series or a batch
    of shape ``(batch_size, n_obs)``.

    Parameters
    ----------
    alias : str or None, default None
        A display name for the model, stored as ``alias``.
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'``, ``'deterministic'``
        or ``'fast'``); None uses the process default (see
        :func:`mojolearn.set_numeric_mode`).

    Attributes
    ----------
    mean_ : ndarray of shape (batch_size,), float32
        The fitted demand rate of each series; the point forecast.

    Notes
    -----
    Input must be float32; float64 raises TypeError. Prediction intervals,
    fitted values and exogenous regressors are not supported and raise
    NotImplementedError.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> y = (rng.random(120) < 0.3).astype(np.float32) * rng.integers(1, 10, 120).astype(np.float32)
    >>> mojolearn.CrostonOptimized().fit(y).predict(12)["mean"].shape
    (12,)
    """
    _VARIANT = 1


class CrostonSBA(_Croston):
    """Syntetos-Boylan approximation (SBA) of Croston's method.

    The :class:`CrostonClassic` forecast multiplied by 0.95, which corrects
    Croston's bias. Follows statsforecast's ``CrostonSBA``. ``fit`` takes one
    series or a batch of shape ``(batch_size, n_obs)``.

    Parameters
    ----------
    alias : str or None, default None
        A display name for the model, stored as ``alias``.
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'``, ``'deterministic'``
        or ``'fast'``); None uses the process default (see
        :func:`mojolearn.set_numeric_mode`).

    Attributes
    ----------
    mean_ : ndarray of shape (batch_size,), float32
        The fitted demand rate of each series; the point forecast.

    Notes
    -----
    Input must be float32; float64 raises TypeError. Prediction intervals,
    fitted values and exogenous regressors are not supported and raise
    NotImplementedError.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> y = (rng.random(120) < 0.3).astype(np.float32) * rng.integers(1, 10, 120).astype(np.float32)
    >>> mojolearn.CrostonSBA().fit(y).predict(12)["mean"].shape
    (12,)
    """
    _VARIANT = 2
