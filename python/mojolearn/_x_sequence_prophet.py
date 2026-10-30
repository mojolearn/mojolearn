# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""ProphetForecaster: a Prophet-style forecaster (linear growth, Fourier
seasonalities, holiday regressors) fitted on the GPU."""
from ._optional_numpy import require_numpy
from ._mode import ParamsMixin
np = require_numpy('_x_sequence_prophet')


# The model of `prophet/forecaster.py` and `stan/prophet.stan` (linear growth),
# fitted by MAP on the GPU (`sequence/prophet.mojo`, one series per thread for a
# batch sharing t). The fit is our float32 L-BFGS, not Stan's; parity with the
# prophet package is at a tolerance. Refusals are listed by name in
# sequence/NOT_IMPLEMENTED.tsv.
class ProphetForecaster(ParamsMixin):
    """Prophet-style additive or multiplicative forecaster with linear growth.

    Models a series as a piecewise-linear trend with changepoints, plus
    Fourier seasonalities and optional holiday indicator regressors, and
    fits the maximum a posteriori parameters with float32 L-BFGS (Prophet
    uses Stan, so results agree with the prophet package to a tolerance, not
    exactly). Prophet's conventions are kept: ``y`` is scaled by ``max|y|``
    and time to [0, 1] over the history; ``n_changepoints`` are placed
    evenly over the first ``changepoint_range`` of the history (the first
    dropped); per-feature prior scales; Prophet's linear initialization.

    Automatic seasonalities follow Prophet: yearly (period 365.25 days,
    order 10) when the history spans two years; weekly (7 days, order 3) when
    it spans two weeks at sub-weekly spacing; daily (1 day, order 4) when it
    spans two days at sub-daily spacing. Add others with
    :meth:`add_seasonality`.

    Time ``t`` is given in days (floats) or as numpy ``datetime64``
    (converted to days since 1970). ``y`` may be one series of shape
    ``(n_obs,)`` or a batch of shape ``(batch_size, n_obs)`` sharing ``t``.

    Not supported: logistic and flat growth, uncertainty intervals and
    sampling (``mcmc_samples``, ``interval_width``), extra regressors beyond
    holiday indicator columns, conditional seasonalities, float64 ``y``.

    Parameters
    ----------
    growth : {"linear"}, default "linear"
        Trend type; other values raise NotImplementedError.
    changepoints : array-like or None, default None
        Explicit changepoint times (same units as ``t``); they must fall
        within the training data. None places ``n_changepoints``
        automatically.
    n_changepoints : int, default 25
        Number of automatic changepoints.
    changepoint_range : float, default 0.8
        Fraction of the history in which automatic changepoints are placed.
    yearly_seasonality, weekly_seasonality, daily_seasonality : "auto", bool or int, default "auto"
        ``"auto"`` applies Prophet's rule above, True forces the seasonality
        on with its default order, False turns it off, and an int turns it on
        with that Fourier order.
    seasonality_mode : {"additive", "multiplicative"}, default "additive"
        How seasonalities and holidays combine with the trend.
    seasonality_prior_scale : float, default 10.0
        Prior scale of the seasonal coefficients.
    holidays_prior_scale : float, default 10.0
        Prior scale of the holiday coefficients.
    changepoint_prior_scale : float, default 0.05
        Prior scale of the trend changes (smaller is smoother).
    max_iter : int, default 2000
        Maximum L-BFGS iterations.
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'``
        or ``'fast'``); None uses the process default (see
        :func:`mojolearn.set_numeric_mode`). ``'fast'`` is available on Apple GPUs only.

    Attributes
    ----------
    seasonalities : dict
        Seasonalities in use, by name: ``period``, ``fourier_order`` and
        ``prior_scale``.
    changepoints_t_ : ndarray, float32
        Changepoint times on the scaled [0, 1] time axis.
    params_ : ndarray of shape (batch_size, 3 + n_changepoints + n_features), float32
        Fitted parameters per series.
    objective_ : ndarray of shape (batch_size,), float32
        Final objective value per series.
    n_iter_ : ndarray of shape (batch_size,), int64
        L-BFGS iterations per series.
    trend_ : ndarray
        Trend component at the times of the last :meth:`predict` call.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> t = np.arange(730, dtype=np.float64)
    >>> y = (10 + 0.01 * t + np.sin(2 * np.pi * t / 7)).astype(np.float32)
    >>> model = mojolearn.ProphetForecaster().fit(t, y)
    >>> model.predict(np.arange(730, 760, dtype=np.float64)).shape
    (30,)
    """

    # Every public method is one native call whose owner does the whole
    # lifecycle (fit bookkeeping and the NotFittedError of predict in
    # bindings/xseq_prophet_native.mojo), so no `_mode` class-creation wrappers are installed.
    _mojolearn_native_public = True

    def __init__(self, growth="linear", changepoints=None, n_changepoints=25, changepoint_range=0.8,
                 yearly_seasonality="auto", weekly_seasonality="auto", daily_seasonality="auto",
                 seasonality_mode="additive", seasonality_prior_scale=10.0, holidays_prior_scale=10.0,
                 changepoint_prior_scale=0.05, max_iter=2000, numeric_mode=None):
        return _XSEQ(self, "prophet_initialize", (growth, changepoints, n_changepoints, changepoint_range,
                     yearly_seasonality, weekly_seasonality, daily_seasonality, seasonality_mode,
                     seasonality_prior_scale, holidays_prior_scale, changepoint_prior_scale, max_iter,
                     numeric_mode), _XSEQ_CONTEXT)

    def add_seasonality(self, name, period, fourier_order, prior_scale=None):
        """Add a custom seasonality before :meth:`fit`.

        Parameters
        ----------
        name : str
            Seasonality name. Using ``"yearly"``, ``"weekly"`` or ``"daily"``
            replaces the automatic one.
        period : float
            Period in days, greater than 0.
        fourier_order : int
            Number of Fourier terms, at least 1.
        prior_scale : float or None, default None
            Prior scale; None uses ``seasonality_prior_scale``.

        Returns
        -------
        self
        """
        return _XSEQ(self, "prophet_add_seasonality", (name, period, fourier_order, prior_scale), _XSEQ_CONTEXT)

    def fit(self, t, y, holidays=None):
        """Fit the model.

        Parameters
        ----------
        t : array-like of shape (n_obs,)
            Times in days (floats) or numpy ``datetime64``, sorted ascending,
            no NaN.
        y : array-like of shape (n_obs,) or (batch_size, n_obs)
            Observations, float32 and finite; at least 2 points.
        holidays : array-like of shape (n_obs, n_holidays) or None, default None
            Holiday indicator columns.

        Returns
        -------
        self

        Raises
        ------
        TypeError
            If ``y`` is float64.
        ValueError
            If ``t`` holds NaN or is not sorted, ``y`` is not finite, the
            lengths differ, there are fewer than 2 points, or explicit
            changepoints fall outside the training data.
        """
        return _XSEQ(self, "prophet_fit", (t, y, holidays), _XSEQ_CONTEXT)

    def predict(self, t, holidays=None):
        """Predict at the given times.

        Parameters
        ----------
        t : array-like of shape (n_times,)
            Times in days or numpy ``datetime64``, as for :meth:`fit`.
        holidays : array-like of shape (n_times, n_holidays) or None, default None
            Holiday indicator columns for ``t``; required when the model was
            fitted with holidays.

        Returns
        -------
        ndarray of shape (n_times,) or (batch_size, n_times), float32
            The forecast ``yhat``. The trend component is stored in
            ``trend_``.
        """
        return _XSEQ(self, "prophet_predict", (t, holidays), _XSEQ_CONTEXT)


# The native owner of fit and predict (bindings/xseq_prophet_native.mojo),
# connected once at import.
from ._native_xseq_bridge import connect as _connect_xseq  # noqa: E402
_XSEQ, _XSEQ_CONTEXT = _connect_xseq()

# Install the same native attribute slot used by the public lazy-export door
# before retaining its identity. Otherwise the direct-module constructor works,
# but resolving mojolearn.ProphetForecaster later installs __getattribute__ and
# the original lifecycle snapshot correctly refuses that changed class layout.
# This is one-time import connection, not a refresh after arbitrary user edits.
import sys as _prophet_sys
from ._expansion_sequence import _connect_imported_module as _connect_prophet_module
_connect_prophet_module(_prophet_sys.modules[__name__])
del _connect_prophet_module, _prophet_sys

# Fixed class/protocol metadata is connected only during module import.
from ._native_xseq_bridge import register as _register_xseq
_register_xseq(xseq_prophet_class=ProphetForecaster,
               xseq_prophet_mro=ProphetForecaster.__mro__,
               xseq_prophet_layouts=tuple((owner, tuple((key, vars(owner).get(key)) for key in
                   ('__dict__', '__new__', '__init__', '__getattribute__', '__getattr__', '__setattr__',
                    '__del__', 'add_seasonality', 'fit', 'predict', 'seasonalities')))
                   for owner in ProphetForecaster.__mro__))  # glue: fixed native lifecycle identities

if "xseq_unavailable" not in _XSEQ_CONTEXT:
    if type(np.promote_types) is not type(len):
        raise TypeError("Prophet temporal metadata requires native NumPy dtype promotion")
    _register_xseq(xseq_prophet_complex_warning=getattr(np, "exceptions", np).ComplexWarning,
                   xseq_prophet_scalar_inputs=(bytes, np.str_, np.bytes_, np.longdouble),
                   xseq_prophet_string_type=np.str_,
                   xseq_prophet_datetime_type=np.datetime64,
                   xseq_prophet_timedelta_type=np.timedelta64,
                   xseq_prophet_datetime_ns=np.dtype('datetime64[ns]'),
                   xseq_prophet_object_dtype=np.dtype('O'),
                   xseq_prophet_big_i64=np.dtype('>i8'),
                   xseq_prophet_promote_types=np.promote_types,
                   xseq_prophet_extra_scalars=(np.str_, np.bytes_, np.longdouble, np.clongdouble, np.complex64, np.complex128),
                   xseq_prophet_complex_scalars=(np.complex64, np.complex128, np.clongdouble),
                   xseq_prophet_complex_dtype=np.dtype("c16"),
                   xseq_prophet_extended_scalars=(np.longdouble, np.clongdouble, np.complex64, np.complex128))
