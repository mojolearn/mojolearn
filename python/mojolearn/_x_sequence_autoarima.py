# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""AutoARIMA: batched automatic ARIMA order selection, following cuML's
``AutoARIMA``."""
from ._optional_numpy import require_numpy
from ._mode import ParamsMixin
from ._arima_impl import _TSA, _TSA_CONTEXT

# Import-time ndarray connection for the native finished-result provider.
require_numpy('_x_sequence_autoarima')


# The order search of cuML `python/cuml/cuml/tsa/auto_arima.pyx`
# (`AutoARIMA.search`, `fit`, `predict`, `forecast`) over mojolearn's batched
# `ARIMA` (`arima/`, the `_mojolearn_arima` binding) and the KPSS choice of `d`
# (`tsa/`, `select_d`). The search, as the reference's:
#   1. `d` per series: the KPSS test at d = 0, 1, ... (`select_d`), unless one
#      `d` is given; series are grouped by their `d`.
#   2. For each group, every (p, q, P, Q, k) of the product of the options, in
#      `itertools.product` order, skipping p + q + P + Q + k == 0, is fitted on
#      the whole group as one batch; each series keeps the order of least
#      information criterion, the FIRST such order on a tie (the reference's
#      `_divide_by_min` is an argmin).
#   3. `fit` refits every chosen order on its sub-batch.
# The information criterion is batched_arima.cu `information_criterion`'s:
# -2 loglike + 2N (aic), + 2N(N+1)/(n - N - 1) (aicc), log(n) N (bic), with
# n = n_obs - d - s D and N the model's parameter count, computed in float64 in
# Mojo (`ic_running_min_f64`) from the fit's float32 log-likelihood. Builds whose
# ARIMA binding reports `arima_order_caps() & 4` take the criterion in float32,
# one rounding of the same value, on the device with the order choice
# (`arima_order_search_device`) or, on the host column and for grids the grouped
# search declines, in Mojo (`ic_running_min_f32`).
# `s` of 0, None or 1 is non-seasonal (the reference's `if s is 1: s = None`).
# seasonal_test="seas" over a D list of two options is statsmodels' STL, not a
# GPU path in the reference either.
class AutoARIMA(ParamsMixin):
    """Automatic ARIMA order selection for a batch of series.

    Chooses, per series, the differencing order ``d`` by the KPSS test and
    then the (p, q, P, Q) orders and intercept with the least information
    criterion, following cuML's ``AutoARIMA.search``; :meth:`fit` then fits
    each chosen model with :class:`ARIMA`. Every candidate order is fitted on
    all the series that share a ``d`` as one batch; on a tie the first order
    tried wins.

    ``endog`` is laid out one series per row, ``(batch_size, n_obs)``;
    cuML's AutoARIMA takes series in columns, so transpose to call it the same
    way. Fitting is maximum likelihood only.

    Parameters
    ----------
    endog : array-like of shape (batch_size, n_obs) or (n_obs,)
        The series, float32. float64 raises TypeError.

    Attributes
    ----------
    d_ : ndarray of shape (batch_size,), int64
        Differencing order chosen for each series. Set by :meth:`search`.
    order_ : ndarray of shape (batch_size, 8), int64
        Chosen ``(p, d, q, P, D, Q, s, k)`` per series, ``k`` 1 with an
        intercept. Set by :meth:`search`.
    ic_ : ndarray of shape (batch_size,), float64
        Information criterion of the chosen order. Set by :meth:`search`.
    models : list
        Distinct chosen models as ``((p, d, q), (P, D, Q, s), k)``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> y = np.random.default_rng(0).standard_normal((8, 120)).cumsum(axis=1).astype(np.float32)
    >>> model = mojolearn.AutoARIMA(y).search(d=range(2), p=range(3), q=range(3)).fit()
    >>> model.forecast(10).shape
    (8, 10)
    """

    #: The bindings the search runs, through `ARIMA` and `select_d` (named
    #: here so the lane map derives them and their CPU host bindings).
    _ARIMA_BINDING = "_mojolearn_arima"
    _TSA_BINDING = "_mojolearn_tsa"

    def __init__(self, endog):
        _TSA(self, "autoarima_initialize", (endog,), _TSA_CONTEXT)

    def search(self, s=None, d=range(3), D=range(2), p=range(1, 4), q=range(1, 4), P=range(3), Q=range(3),
               fit_intercept="auto", ic="aicc", test="kpss", seasonal_test="seas", h=1e-8, maxiter=20,
               method="auto", truncate=0):
        """Select the orders for every series.

        Parameters
        ----------
        s : int or None, default None
            Seasonal period. None, 0 or 1 means non-seasonal (then D, P and Q
            are 0).
        d : int or iterable of int, default range(3)
            Differencing options; a list must be ``0, 1, ..., d_max``. With
            one value it is used for every series; otherwise the KPSS test
            picks ``d`` per series.
        D : int or iterable of int, default range(2)
            Seasonal differencing; with ``s`` set, give one value (the
            seasonal test over several values is not supported).
        p, q : int or iterable of int, default range(1, 4)
            Non-seasonal AR and MA order options, clipped to [0, 4] (or
            [0, s - 1] when seasonal).
        P, Q : int or iterable of int, default range(3)
            Seasonal AR and MA order options, clipped to [0, 4] (0 when
            non-seasonal).
        fit_intercept : bool or "auto", default "auto"
            Whether candidates include an intercept.
        ic : {"aic", "aicc", "bic"}, default "aicc"
            Information criterion.
        test : {"kpss"}, default "kpss"
            Stationarity test for ``d``.
        seasonal_test : str, default "seas"
            Only used with several ``D`` options, which are not supported.
        h : float, default 1e-8
            Only the default is supported.
        maxiter : int, default 20
            Optimizer iterations per candidate fit.
        method : {"auto", "ml"}, default "auto"
            Fitting method; maximum likelihood. ``"css"`` and ``"css-ml"``
            raise NotImplementedError.
        truncate : int, default 0
            Only 0 is supported.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            For an unknown ``ic`` or ``test``, or an order option with no
            valid value.
        NotImplementedError
            For css methods, a non-default ``h`` or ``truncate``, several
            ``D`` options with a season, or a ``d`` list that is not
            ``0..d_max``.
        """
        return _TSA(self, "autoarima_search", (s, d, D, p, q, P, Q, fit_intercept, ic, test,
                    seasonal_test, h, maxiter, method, truncate), _TSA_CONTEXT)




    def fit(self, h=1e-8, maxiter=1000, method="ml", truncate=0):
        """Fit the chosen model of every series.

        Call :meth:`search` first.

        Parameters
        ----------
        h : float, default 1e-8
            Only the default is supported.
        maxiter : int, default 1000
            Optimizer iterations.
        method : {"ml"}, default "ml"
            Maximum likelihood only.
        truncate : int, default 0
            Only 0 is supported.

        Returns
        -------
        self

        Raises
        ------
        RuntimeError
            If :meth:`search` has not been called.
        """
        # every chosen order group's ARIMA built and fitted (or adopted from
        # the search's fit block) in one native call
        # (bindings/tsa_autoarima_native.mojo)
        return _TSA(self, "autoarima_fit", (h, maxiter, method, truncate), _TSA_CONTEXT)


    def predict(self, start=0, end=None, level=None):
        """In-sample (and out-of-sample) predictions from ``start`` to ``end``.

        Parameters
        ----------
        start : int, default 0
            First index to predict.
        end : int or None, default None
            One past the last index; None means ``n_obs``.
        level : None
            Prediction intervals are not supported; anything other than None
            raises NotImplementedError.

        Returns
        -------
        ndarray of shape (batch_size, end - start), float32
        """
        return _TSA(self, "autoarima_predict", (start, end, level), _TSA_CONTEXT)

    def forecast(self, nsteps, level=None):
        """Forecast ``nsteps`` past the end of every series.

        Parameters
        ----------
        nsteps : int
            Forecast horizon.
        level : None
            Prediction intervals are not supported; anything other than None
            raises NotImplementedError.

        Returns
        -------
        ndarray of shape (batch_size, nsteps), float32
        """
        return _TSA(self, "autoarima_forecast", (nsteps, level), _TSA_CONTEXT)


_TSA_CONTEXT["autoarima_class"] = AutoARIMA
