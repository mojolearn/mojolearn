# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""STL, Season-Trend decomposition by Loess, following
``statsmodels.tsa.seasonal.STL``. Series are decomposed on the GPU."""
from ._optional_numpy import require_numpy
from ._mode import ParamsMixin
np = require_numpy('_x_sequence_stl')

from . import _native_xseq_bridge

# Package-import connection: fit is ONE call into the compiled owner
# (bindings/xseq_forecast_native.mojo), which builds the result object too.
_XSEQ, _XSEQ_CONTEXT = _native_xseq_bridge.connect()


class DecomposeResult:
    """Result of :meth:`STL.fit`.

    Attributes
    ----------
    observed, seasonal, trend, resid, weights : ndarray, float32
        The input series, its seasonal, trend and residual components, and
        the robustness weights, each shaped like the input.
    """

    def __init__(self, observed, seasonal, trend, resid, weights):
        return _XSEQ(self, "decompose_initialize", (observed, seasonal, trend, resid, weights), _XSEQ_CONTEXT)


# The arithmetic is `sequence/stl.mojo`, statsmodels' `_stl.pyx` loops in
# float32, one series per GPU thread (or host loop iteration).
class STL(ParamsMixin):
    """Season-Trend decomposition using Loess.

    Splits each series into seasonal, trend and residual components with
    statsmodels' STL algorithm and parameters. ``endog`` may be one series
    of shape ``(n_obs,)`` or a batch of shape ``(batch_size, n_obs)``, one
    series per row, all sharing the configuration.

    Unlike statsmodels, ``period`` is required: statsmodels can infer it from
    a pandas index, which is not supported here.

    Parameters
    ----------
    endog : array-like of shape (n_obs,) or (batch_size, n_obs)
        The series, float32. float64 raises TypeError.
    period : int
        Periodicity of the season, at least 2. Required.
    seasonal : int, default 7
        Length of the seasonal smoother (odd).
    trend : int or None, default None
        Length of the trend smoother. None uses statsmodels' default, the
        smallest odd integer at least ``1.5 * period / (1 - 1.5 / seasonal)``.
    low_pass : int or None, default None
        Length of the low-pass filter. None uses the smallest odd integer
        greater than ``period``.
    seasonal_deg, trend_deg, low_pass_deg : {0, 1}, default 1
        Degree of each Loess fit (0 constant, 1 linear).
    robust : bool, default False
        Use robust fitting with outer iterations and robustness weights.
    seasonal_jump, trend_jump, low_pass_jump : int, default 1
        Evaluate each smoother every ``jump`` points and interpolate between.
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'``
        or ``'fast'``); None uses the process default (see
        :func:`mojolearn.set_numeric_mode`). ``'fast'`` is available on Apple GPUs only.

    Raises
    ------
    TypeError
        If ``endog`` is float64.
    ValueError
        If ``period`` is missing or below 2, a degree is not 0 or 1, or
        ``endog`` is not 1-D or 2-D.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> t = np.arange(144, dtype=np.float32)
    >>> y = 0.05 * t + np.sin(2 * np.pi * t / 12).astype(np.float32)
    >>> res = mojolearn.STL(y, period=12).fit()
    >>> res.seasonal.shape
    (144,)
    """

    def __init__(self, endog, period=None, seasonal=7, trend=None, low_pass=None, seasonal_deg=1,
                 trend_deg=1, low_pass_deg=1, robust=False, seasonal_jump=1, trend_jump=1,
                 low_pass_jump=1, numeric_mode=None):
        return _XSEQ(self, "stl_initialize", (endog, period, seasonal, trend, low_pass, seasonal_deg, trend_deg, low_pass_deg, robust, seasonal_jump, trend_jump, low_pass_jump, numeric_mode), _XSEQ_CONTEXT)

    @property
    def config(self):
        """The decomposition settings as a dict."""
        return _XSEQ(self, "stl_config", (), _XSEQ_CONTEXT)

    def fit(self, inner_iter=None, outer_iter=None):
        """Run the decomposition.

        Parameters
        ----------
        inner_iter : int or None, default None
            Inner loop iterations. None uses 2 when ``robust`` and 5
            otherwise, as statsmodels does.
        outer_iter : int or None, default None
            Outer (robustness) iterations. None uses 15 when ``robust`` and 0
            otherwise.

        Returns
        -------
        DecomposeResult
            ``observed``, ``seasonal``, ``trend``, ``resid`` and ``weights``,
            each shaped like ``endog``.
        """
        return _XSEQ(self, "stl_fit", (inner_iter, outer_iter), _XSEQ_CONTEXT)


_native_xseq_bridge.register(xseq_decompose_result=DecomposeResult, xseq_stl_class=STL)
