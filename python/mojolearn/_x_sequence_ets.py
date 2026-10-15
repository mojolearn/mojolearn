# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Exponential smoothing state-space (ETS) forecasters: ETS and DampedETS,
following statsforecast's ``AutoETS`` with a fixed model. Each series is
fitted on the GPU."""
from ._optional_numpy import require_numpy
from ._mode import ParamsMixin
np = require_numpy('_x_sequence_ets')

from . import _backend
from ._buffer import _native


# Fitted on the GPU one series per thread (`sequence/ets.mojo`): likelihood over
# smoothing parameters and initial states by statsforecast's Nelder-Mead.
# The reference's model checks are kept, in its words: a seasonal letter with
# season_length 1 is "Nonseasonal data"; additive errors with a multiplicative
# season are a "Forbidden model combination"; multiplicative errors or season
# need positive data; n <= (parameter count) + 4 is the reference's "tiny
# datasets" refusal; a series no longer than season_length is fitted without a
# season, as there (ets_f).
class ETS(ParamsMixin):
    """Exponential smoothing (ETS) with a fixed model.

    Fits ETS(E, T, S) by maximum likelihood over the smoothing parameters and
    initial states, following statsforecast's
    ``AutoETS(season_length=m, model=..., damped=...)``. Supported models:
    error ``A`` or ``M``; trend ``N`` or ``A`` (optionally damped); season
    ``N``, ``A`` or ``M``.

    Unlike statsforecast, the default is a damped additive-trend model
    (``model="AAN"``, ``damped=True``) and no model letter may be ``'Z'``
    (automatic selection). ``fit`` takes one series of shape ``(n_obs,)`` or a
    batch of shape ``(batch_size, n_obs)``; the model is estimated when
    :meth:`predict` is called, and repeating the same ``predict`` on the same
    fitted series returns the stored result. A series no longer than
    ``season_length`` is fitted without a season, as in statsforecast.

    Parameters
    ----------
    season_length : int, default 1
        Number of observations per season; must be greater than 1 for a
        seasonal model and at most 58.
    model : str, default "AAN"
        Three letters: error (``A``/``M``), trend (``N``/``A``) and season
        (``N``/``A``/``M``).
    damped : bool, default True
        Damp the trend. Requires a trend (``model[1] == "A"``).
    alpha, beta, phi, gamma : float or None, default None
        Fix the level, trend, damping and seasonal smoothing parameters;
        None estimates each.
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'``
        or ``'fast'``); None uses the process default (see
        :func:`mojolearn.set_numeric_mode`). ``'fast'`` is available on Apple GPUs only.

    Attributes
    ----------
    info_ : ndarray of shape (batch_size, 10), float32
        Per-series fit summary. Set by :meth:`predict`.
    seasonal_states_ : ndarray of shape (season_length,) or (batch_size, season_length), float32
        Final seasonal states, for seasonal models only. Set by
        :meth:`predict`.

    Raises
    ------
    ValueError
        For an invalid model string, ``damped=True`` without a trend, a
        seasonal model with ``season_length == 1`` ("Nonseasonal data"), or
        additive errors with a multiplicative season ("Forbidden model
        combination").
    NotImplementedError
        For a multiplicative trend, ``'Z'`` letters, or ``season_length``
        above 58 with a seasonal model.

    Notes
    -----
    Input must be float32; float64 raises TypeError. Each series needs at
    least 4 observations, and more than the number of parameters plus 4.
    Multiplicative errors or seasons need strictly positive data.
    Prediction intervals, fitted values and exogenous regressors are not
    supported and raise NotImplementedError.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> t = np.arange(120, dtype=np.float32)
    >>> y = 50 + 0.2 * t + 5 * np.sin(2 * np.pi * t / 12).astype(np.float32)
    >>> model = mojolearn.ETS(season_length=12, model="AAA", damped=False).fit(y)
    >>> model.predict(12)["mean"].shape
    (12,)
    """

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
        """Check and store one series or a batch of series.

        The model itself is estimated by :meth:`predict`.

        Parameters
        ----------
        y : array-like of shape (n_obs,) or (batch_size, n_obs)
            The series, float32. A private copy is kept.
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
            If a series has fewer than 4 observations, if a multiplicative
            model meets non-positive or non-finite values, or if there are
            too many parameters for the number of observations.
        NotImplementedError
            If ``n_obs`` is at most the parameter count plus 4.
        """
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
        # every value > 0, in Mojo: finite (`all_finite_f32`) and a minimum
        # above zero (`reduce_stat` 0, the first-wins min)
        if (e == "M" or s == "M") and not (int(_native("all_finite_f32")(self._y.ctypes.data, self._y.size))
                                           and _native("reduce_stat")(self._y.ctypes.data, 0, self._y.size, 0) > 0):
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
        """Fit the model and forecast h steps ahead.

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
        fp = [0.0 if v is None else float(v) for v in (self.alpha, self.beta, self.phi, self.gamma)]  # glue: the four smoothing parameter arguments
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
            f, info, ss = (a.copy() for a in last[1:])  # glue: the three stored result arrays
        else:
            _backend.binding("_mojolearn_x_sequence", self.numeric_mode).ets(
                [self._y.ctypes.data, f.ctypes.data, info.ctypes.data, ss.ctypes.data], ip, fp)
            self._last = (key, f.copy(), info.copy(), ss.copy())
        self.info_ = info
        if s != "N":
            self.seasonal_states_ = ss[0] if self._one else ss
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
            raise NotImplementedError("ETS: fitted values are not implemented")
        return self.fit(y).predict(h, level=level)


class DampedETS(ETS):
    """Holt's damped additive trend model, ETS(A,Ad,N).

    :class:`ETS` with ``model=error + "AN"`` and ``damped=True``: a damped
    additive trend and no season. Set ``error="M"`` for multiplicative
    errors (strictly positive data).

    Parameters
    ----------
    season_length : int, default 1
        Number of observations per season (no season is fitted).
    error : {"A", "M"}, default "A"
        Additive or multiplicative errors.
    alpha, beta, phi : float or None, default None
        Fix the level, trend and damping smoothing parameters; None
        estimates each.
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'``
        or ``'fast'``); None uses the process default (see
        :func:`mojolearn.set_numeric_mode`). ``'fast'`` is available on Apple GPUs only.

    Attributes
    ----------
    info_ : ndarray of shape (batch_size, 10), float32
        Per-series fit summary. Set by :meth:`predict`.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> y = (100 + np.cumsum(np.random.default_rng(0).normal(1, 2, 80))).astype(np.float32)
    >>> mojolearn.DampedETS().fit(y).predict(10)["mean"].shape
    (10,)
    """

    def __init__(self, season_length=1, error="A", alpha=None, beta=None, phi=None, numeric_mode=None):
        super().__init__(season_length, error + "AN", True, alpha, beta, phi, numeric_mode=numeric_mode)
        self.error = error  # stored so get_params / clone report it
