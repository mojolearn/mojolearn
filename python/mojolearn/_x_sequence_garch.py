# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""GARCH(p, o, q) volatility models, following the arch package's
``arch_model(..., vol='GARCH', dist='normal')``. Each series is fitted on the
GPU."""
from ._optional_numpy import require_numpy
np = require_numpy('_x_sequence_garch')

from . import _backend
from ._buffer import _native


# Fitted on the GPU one series per thread (`sequence/garch.mojo`): arch's
# variance recursion, backcast, variance bounds, starting-value grid and
# Gaussian likelihood, maximised by statsforecast's Nelder-Mead inside arch's
# bounds.
class GARCH:
    """GARCH(p, o, q) conditional volatility model with normal errors.

    Fits a constant- or zero-mean GARCH model by Gaussian maximum
    likelihood, following the arch package's
    ``arch_model(y, mean='Constant'|'Zero', vol='GARCH', p=p, o=o, q=q,
    dist='normal')``: arch's variance recursion, backcast, variance bounds
    and starting values. ``o > 0`` adds asymmetric (GJR) terms. ``fit``
    takes one series of shape ``(n_obs,)`` or a batch of shape
    ``(batch_size, n_obs)``.

    Parameters
    ----------
    p : int, default 1
        Order of the lagged squared innovations (ARCH terms).
    o : int, default 0
        Order of the asymmetric innovations; needs ``p > 0``.
    q : int, default 1
        Order of the lagged conditional variance (GARCH terms).
    mean : {"Constant", "Zero"}, default "Constant"
        Mean model. Other arch mean models (AR, HAR, LS) raise
        NotImplementedError.
    dist : {"normal"}, default "normal"
        Error distribution; only ``"normal"`` is supported.
    power : float, default 2.0
        Only 2.0 is supported (TARCH/APARCH powers raise
        NotImplementedError).
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'``, ``'deterministic'``
        or ``'fast'``); None uses the process default (see
        :func:`mojolearn.set_numeric_mode`). ``'fast'`` is available on Apple GPUs only.

    Attributes
    ----------
    params_ : ndarray of shape (2 + p + o + q,) or (batch_size, 2 + p + o + q), float32
        ``mu, omega, alpha[1..p], gamma[1..o], beta[1..q]``.
    loglikelihood_ : float32 or ndarray of shape (batch_size,)
        Maximized Gaussian log-likelihood.
    n_iter_ : int64 or ndarray of shape (batch_size,)
        Nelder-Mead iterations used.
    conditional_volatility_ : ndarray of shape (n_obs,) or (batch_size, n_obs), float32
        Fitted conditional volatility.

    Raises
    ------
    ValueError
        If p, o or q is negative, all are zero, or ``o > 0`` with ``p == 0``.

    Notes
    -----
    Input must be float32; float64 raises TypeError. Rescaling is not
    supported: scale the data yourself if needed.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> r = np.random.default_rng(0).standard_normal(500).astype(np.float32)
    >>> model = mojolearn.GARCH(p=1, q=1).fit(r)
    >>> model.forecast(5).shape
    (5,)
    """

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
        """Fit the model to one series or a batch of series.

        Parameters
        ----------
        y : array-like of shape (n_obs,) or (batch_size, n_obs)
            Returns (or other zero-centered data), float32, finite, at least
            10 observations per series.
        horizon : int, default 1
            Variance forecast horizon computed alongside the fit (at least 1).

        Returns
        -------
        self

        Raises
        ------
        TypeError
            If ``y`` is float64.
        ValueError
            If a series has fewer than 10 observations or ``y`` holds a
            non-finite value.
        """
        y = np.asarray(y)
        if y.dtype == np.float64:
            raise TypeError("GARCH: float64 is refused; pass float32")
        self._one = y.ndim == 1
        Y = np.ascontiguousarray(y, dtype=np.float32).reshape(1, -1) if self._one else \
            np.ascontiguousarray(y, dtype=np.float32)
        if Y.ndim != 2 or Y.shape[1] < 10:
            raise ValueError("GARCH: each series needs at least 10 observations")
        # the finiteness scan in Mojo (the base binding's `all_finite_f32`)
        if not int(_native("all_finite_f32")(Y.ctypes.data, Y.size)):
            raise ValueError("GARCH: y holds a non-finite value")
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
        """Analytic conditional variance forecasts.

        Parameters
        ----------
        horizon : int, default 1
            Number of steps ahead. A horizon longer than the one computed at
            fit refits the stored series.

        Returns
        -------
        ndarray of shape (horizon,) or (batch_size, horizon), float32
            Forecast conditional variances.
        """
        h = int(horizon)
        if h > self._fc.shape[1]:
            y = self._y[0] if self._one else self._y
            self.fit(y, horizon=h)
        f = self._fc[:, :h]
        return f[0] if self._one else f
