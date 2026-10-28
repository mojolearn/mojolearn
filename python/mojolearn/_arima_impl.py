# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Batched ARIMA on the GPU, backed by a batched Kalman filter.

PRIVATE MODULE. `ARIMA` is re-exported from `mojolearn/__init__.py`.

WHAT LANDED HERE, AND WHAT USED TO STAND IN ITS PLACE. Until 2026-09-01
`python/mojolearn/__init__.py` carried an `_NOT_YET["ARIMA"]` entry saying
the lane had a likelihood, a gradient and a predict but NO `fit`, so an
`ARIMA` class would have to demand its own answer as an argument. That was
true and it is now false. `arima/impl/estimate_x0.mojo` (the starting
parameters, over an own-written Householder QR that beats the normal
equations 7.4e-07 against 1.5e-04 and is strictly better on 6 of 6 series)
and `arima/impl/batched_fit.mojo` (an own-written batched L-BFGS with
a shared line search, no scipy) closed that gap and are gated by
`arima/checks/fit_check.mojo` in BOTH numeric tiers. The `_NOT_YET` entry is
deleted rather than reworded, because the fix for a sentence explaining an
absence is to end the absence.

`ARIMA` IS BATCHED, AND THAT IS THE WHOLE POINT OF THE LANE
------------------------------------------------------------
`y` is 2-D, `(batch_size, n_obs)`. Every series in the batch is fitted at
once, with its OWN parameters, by one set of kernel launches; the batch is
not a convenience wrapper around a loop and it is not a multivariate model.
cuML's batched ARIMA is organized the same way (`ARIMAOrder`, `ARIMAParams` and every kernel in
`cpp/src/arima/` are indexed by a series id) and it is where the speed comes
from. It is ALSO the first thing that differs from statsmodels, whose
`ARIMA` takes ONE series, so the constructor and the methods below read like
statsmodels' and the shapes do not:

    statsmodels     ARIMA(y, order=(1,0,0)).fit()          y is (n_obs,)
    here            ARIMA(order=(1,0,0)).fit(y)            y is (batch, n_obs)

A 1-D `y` is accepted and treated as ONE series, because refusing it would
be pedantry; the returned arrays stay 2-D with a leading 1 either way, so no
shape here is ever a function of what the input's rank happened to be.

THE DATA GOES TO `fit`, NOT TO THE CONSTRUCTOR. Both references put it in the
constructor (`ARIMA(endog, order=...)` in cuML, `ARIMA(y, order=...)` in
statsmodels). This package's other twenty-six estimators take their data in
`fit`, and an ARIMA that did not would be the only class here you could not
clone, re-use on a second batch, or hand to anything expecting the house
shape.

EVERY KNOB IS ON THE CONSTRUCTOR AND `fit` TAKES ONLY DATA, which is the
other divergence from both references: `method` and `maxiter` are `fit`
arguments there and constructor arguments here, for the same reason.

WHAT IS NOT HERE
----------------
`AutoARIMA` is a separate public estimator. Its search, fitting and forecasts
are owned by native providers; the stationarity helpers remain available as
`mojolearn.select_d` and `mojolearn.kpss_test`.

CROSS-VENDOR STATUS, STATED PLAINLY BECAUSE IT IS THE LIBRARY'S HEADLINE
CLAIM AND THIS CLASS DOES NOT CARRY ALL OF IT. `arima/`'s identity card is
BIT-IDENTICAL ON THREE VENDORS at commit `221aa141`, 139 records, Apple M4
and NVIDIA and AMD MI325X (`bench/results/e1/CERT_2026-08-31.md`). That card
is `arima/arima_main.mojo`, which is the Kalman filter, the Jones transform
and their stages. THE FIT IS NOT IN IT. The fit landed after that leg and
the card was re-emitted BYTE IDENTICAL, which says the fit moved no stage
the card records. `fit`'s own gates are 16 of 16 on ONE Apple M4 in both
tiers, and THIS class's own gate,
`python/mojolearn/tests/test_arima_surface.py`, printed green on that same
one box on 2026-09-02, 88 checks and 0 failed in each of its two tier
processes. The three-vendor run THROUGH THIS SURFACE came later: the
identity harness fits and forecasts through this class, and its 136-lane
record at 4048e1b51
(bench/results/identity_break/2026-09-14_136-lanes/diff.three-columns.txt)
reads IDENTICAL x3 on all 27 training cells of the arima, arima-011 and
arima-seasonal-c lanes and on their infer cells, across the Apple M4, an
NVIDIA H100 and an AMD MI325X. That covers those three orders and the
harness fixtures, not every order this class accepts.

That property, where it exists at all, belongs to
`MOJOLEARN_NUMERIC_MODE=identical`, which is the default. The FAST build,
selected explicitly, makes no cross-vendor claim of any kind.
"""

from . import _native_archive_bridge
from ._mode import NumericModeMixin

# The log-likelihood method, as `arima/estimator.mojo` numbers it. Only MLE
# is offered; the other two are REFUSED BY NAME in that file, and the names
# are cuML's own (`arima.pyx:944-946`).
_METHODS = {"ml": 0, "css": 1, "css-ml": 2}

#: The saved-model format tag (`ARIMA.save`, lane/inference-forecast-umap-pca).
_ARIMA_FORMAT = "mojolearn-arima-1"

#: DEVIATION 998: the format of a model fitted WITH exogenous regressors
#: (lane/arima-exog, 2026-09-15). A model without them is still written as
#: `mojolearn-arima-1`, byte for byte what it was, so every saved-model hash
#: recorded before this lane stays valid; a model with them adds the fit's
#: regressors (`exog`, `(batch_size, n_obs, n_exog)` float32, which predict
#: reads) and a thirteenth `meta` field, `n_exog`, under this tag, which a
#: loader that predates it refuses by name instead of reading one block short.
_ARIMA_FORMAT_EXOG = "mojolearn-arima-2"

# The compiled saved-model owner, connected once at import: `save` and
# `load` are one native call each (bindings/archive_schema_native.mojo
# writes the schema registered after the class; host_archive_native.mojo
# `host_archive_arima` builds the loaded model).
_ARCHIVE, _ARCHIVE_CONTEXT = _native_archive_bridge.connect()

# The compiled owner of `fit` and `predict` (bindings/tsa_arima_native.mojo,
# through `tsa_native_call`), connected once at import.
from . import _native_tsa_bridge  # noqa: E402
_TSA, _TSA_CONTEXT = _native_tsa_bridge.connect()
_TSA_CONTEXT["arima_methods"] = _METHODS


#: The address every binding call passes for an exogenous input that is not
#: there. One float, never read (`n_exog == 0`, or no forecast steps).


# Batched ARIMA, backed by a batched Kalman filter and a
# batched L-BFGS (`arima/`, DEVIATIONS 670 to 687 and 990 to
# 993; `arima/README.md`), in statsmodels' constructor shape.
#
# `y` IS 2-D, `(batch_size, n_obs)`, AND THAT DIFFERS FROM statsmodels.
# Every series in the batch gets its own parameters and they are fitted
# together, in one set of launches. A 1-D `y` is taken as one series; the
# outputs are 2-D either way. The module docstring says what else follows
# from the batch being the point.
#
# WHAT IS HONORED, WHAT IS REFUSED, AND WHY, one line per parameter,
# because a parameter that is accepted and ignored is a wrong answer
# waiting for a caller (the house rule). The file named on a refusal is
# THE FILE THAT RAISES IT, and where that file is a `.mojo` one the
# exception arrives as a bare `Exception`, because a Mojo `Error` crossing
# `def_function` is not a named Python type.
#
#     order (p, d, q)   honored   `p > 8` and `q > 8` are refused by
#                                 `arima/impl/tsa/arima_common.mojo::
#                                 validate_order`, with the reference's message, as
#                                 is `d + D > 2` and an order with no
#                                 parameters at all
#     seasonal_order    honored   `(P, D, Q, s)`. A seasonal term with
#       (P, D, Q, s)              `s < 2`, and `s <= p` or `s <= q`, are
#                                 refused by `validate_order`, again with
#                                 the reference's message
#     rd > 8            refused   `validate_order`. `rd = d + s*D +
#                                 max(p + s*P, q + s*Q + 1)` selects
#                                 cuML's BLOCK-PER-SERIES Kalman kernel
#                                 (`batched_kalman.cu:335-745`), a
#                                 different fold shape and an unimplemented
#                                 one. This is a bound on the ORDER, so it
#                                 fires at fit, before any device work
#     r > 5             refused   `validate_order`. `r = max(p + s*P,
#                                 q + s*Q + 1)` selects their Schur /
#                                 Francis-QR Lyapunov solver
#                                 (`matrix.cuh:1899-1948`); the direct
#                                 Kronecker solve is the `r <= 5` arm and
#                                 is what this lane carries
#     trend             honored   for None, 'n' and 'c' ONLY, and it is
#                                 this class's spelling of cuML's
#                                 `fit_intercept` (DEVIATION 993 below).
#                                 't', 'ct' and a polynomial trend
#                                 specification are REFUSED BY NAME by
#                                 `_arima_impl.py`: cuML has no time trend
#                                 to hold one to; pass the time index as an
#                                 `exog` column instead
#     method            honored   for 'ml' ONLY. 'css' and 'css-ml' are
#                                 REFUSED BY NAME by
#                                 `arima/estimator.mojo::_refuse_method`,
#                                 not here: the conditional sum of squares
#                                 likelihood and its `truncate` parameter
#                                 (`batched_arima.cu:271-391`) have no
#                                 implementation. The string is turned into a code
#                                 and passed through UNCLAMPED so that
#                                 refusal stays reachable (DEVIATION 992)
#     maxiter           honored   the L-BFGS iteration cap, cuML's
#                                 `maxiter`, default 1000 as theirs.
#                                 Refused below 1 by `arima/estimator.mojo`
#     exog              honored   on `fit(y, exog)`, `predict(start, end,
#                                 exog)` and `forecast(steps, exog)`
#                                 (lane/arima-exog, 2026-09-15): regression
#                                 with ARIMA errors as cuML's (`beta` packed
#                                 after `mu`, the regressors differenced like
#                                 `y`, `beta` started by least squares before
#                                 the ARMA fit, `x_t beta` added to every
#                                 prediction). `(batch_size, n_obs, n_exog)`
#                                 (DEVIATION 996); at most 17 regressors
#                                 (DEVIATION 994, validate_order); a
#                                 non-finite one refused by name (DEVIATION
#                                 997). cuML's `exog` is a CONSTRUCTOR
#                                 argument; here it goes to `fit` with `y`
#     verbose           refused   `_arima_impl.py`, for anything truthy.
#                                 In the reference it selects LOG LINES; this implementation
#                                 prints none, so accepting it would be
#                                 accepting-and-ignoring
#     output_type       refused   `_arima_impl.py`. A cuML-internal
#                                 array-type selector; this package
#                                 returns `mojolearn.Array` (zero-copy
#                                 under `numpy.asarray`)
#     float64 y         CONVERTED to float32, and the conversion COPIES.
#                                 Named rather than hidden. cuML's ARIMA
#                                 is instantiated on `double` ONLY and
#                                 `arima.pyx:326` checks its input to
#                                 float64; Metal exposes no float64 on the
#                                 device, so there is no double arm here
#                                 to pick (DEVIATION 670). These are
#                                 float32 answers to a float32 problem
#     non-finite y      refused   `arima/impl/batched_arima.mojo::
#                                 _refuse_non_finite`, which names the
#                                 flat index. cuML instead has a MISSING
#                                 OBSERVATION path (`missing = isnan(yt)`
#                                 and the four branches it guards,
#                                 `batched_kalman.cu:191-246`) and that
#                                 path is NOT IMPLEMENTED, so a NaN here is
#                                 refused rather than treated as missing
#     level             absent    NOT A PARAMETER OF THIS CLASS, so
#                                 passing it is a TypeError naming it.
#                                 cuML's confidence intervals (the
#                                 `confidence_intervals` kernel at
#                                 `batched_kalman.cu:824-838` and the
#                                 `P = T P T' + RR'` propagation beside
#                                 it) are NOT IMPLEMENTED
#     simple_           absent    cuML's switch. This lane implements the
#       differencing              `True` arm only and does not carry the
#                                 other as a switch, so there is no value
#                                 to accept
#     start_params      absent    cuML's `set_fit_params` has no door
#                                 here. Every fit starts from
#                                 `estimate_x0`
#     truncate          absent    read only by the CSS likelihood, which
#                                 is refused
#     h                 absent    the finite-difference step, pinned at
#                                 2^-10 by DEVIATION 687. cuML's 1e-8 is
#                                 BELOW float32 epsilon and collapses the
#                                 gradient to zero, so this is not a knob
#                                 a caller may turn
#
# THE DEFAULT ORDER IS (1, 0, 0) AND IT IS NEITHER REFERENCE'S. cuML's is
# (1, 1, 1) and statsmodels' is (0, 0, 0), and the two cannot both be
# honored. (0, 0, 0) is not even reachable here: with `trend=None` it
# resolves to `k = 1` and fits a mean, and with `trend='n'` it is an order
# with no parameters at all, which `validate_order` refuses with the
# reference's message. (1, 1, 1) is a differencing model, and a default that silently
# differences a caller's data is a default that changes what the numbers
# mean. (1, 0, 0) is the smallest model that fits something, and the right
# thing to do with it is to pass your own order.
#
# DEVIATION 993: `trend` IS THIS CLASS'S SPELLING OF cuML's
# `fit_intercept`, AND THE DEFAULT IS statsmodels' RULE, NOT cuML's.
# cuML has a boolean `fit_intercept` defaulting to True, whatever `d`
# is. statsmodels has `trend`, and `trend=None` there resolves to 'c' when
# the series is not differenced and to 'n' when it is. This class takes
# statsmodels' spelling and statsmodels' default rule, so
#
#     trend=None  ->  k = 1 if d + D == 0 else 0
#     trend='n'   ->  k = 0
#     trend='c'   ->  k = 1
#
# and a caller who wants cuML's default on a differenced series writes
# `trend='c'` explicitly. `k` is what reaches the kernels; it is
# `ARIMAOrder::k` and it adds one `mu` per series to the parameter vector.
#
# DEVIATION 990: THE FITTED MODEL CROSSES BACK TO THE HOST AND IS UPLOADED
# AGAIN AT EVERY `predict`. The binding retains no device pointer (the
# rule `bindings/_mojolearn_estimators.mojo` states and every binding in
# this package inherits), so `params_` is a host array and `predict` and
# `forecast` hand it back down with the training series. The consequences,
# both real: the training series is RETAINED by this object, because
# `predict` needs it and there is nowhere else for it to live; and every
# prediction re-runs the Kalman filter over the whole series rather than
# resuming a stored state, which is what cuML does too.
#
# DEVIATION 991: `aic_` AND `bic_` ARE COMPUTED ON THE HOST IN FLOAT64 AND
# ARE NOT A DEVICE ANSWER. cuML's `information_criterion`
# (`batched_arima.cu:592-618`) is NOT IMPLEMENTED, and what it does beyond the
# log-likelihood is one `raft::stats::information_criterion_batched` unary
# op. This computes the same formula,
#
#     aic = 2 * N - 2 * llf
#     bic = log(T) * N - 2 * llf
#
# with `N` the parameter count `ARIMAOrder::complexity()` and `T` the
# number of observations AFTER differencing, `n_obs - (d + s * D)`, which
# is the `n_samples` their caller passes. `llf_` is the device's float32
# log-likelihood widened to float64; the two lines above are a summary of
# the answer, not the answer, and they are no part of any identity claim.
# AICc is not offered, because nothing in this lane needs it and an
# untested third arm is a liability.
#
# THE LOG-LIKELIHOOD IS EVALUATED AGAIN AT THE FITTED POINT, once, rather
# than recovered from the optimizer's objective. The optimizer minimizes
# `-loglike / (n_obs - 1)`, so inverting it would cost a float32 negate
# and a float32 divide whose rounding nobody has measured. That is
# `arima/estimator.mojo::_loglike_at`, and it is what cuML's own
# `information_criterion` does.
#
# Attributes (every array below is a `mojolearn.Array`; `numpy.asarray`
# on it is zero-copy -- DEVIATION 2416)
# ----------
# params_ : Array (batch_size, N) float32
#     The fitted model, forward transformed, packed per series in
#     `ARIMAParams::pack`'s order: `mu` (only when `k == 1`), `beta`
#     (n_exog), then `ar` (p), `ma` (q), `sar` (P), `sma` (Q), then
#     `sigma2`. `N = p + q + P + Q + k + n_exog + 1`. This is exactly the
#     array `predict` and `forecast` send back down.
# beta_ : Array (batch_size, n_exog) float32
#     The regression coefficients, one per regressor per series. Raises
#     AttributeError when the fit had no `exog`, as `mu_` does without an
#     intercept.
# n_exog_ : int
#     The regressor count the fit was handed, 0 without `exog`.
# mu_, ar_, ma_, sar_, sma_, sigma2_ : ndarray float32
#     Named blocks of `params_`. `mu_` and `sigma2_` are `(batch_size,)`;
#     `ar_`, `ma_`, `sar_` and `sma_` are `(batch_size, p)`,
#     `(batch_size, q)`, `(batch_size, P)` and `(batch_size, Q)`, and come
#     back `(batch_size, 0)` where the order has no such block, which is
#     an empty answer to a well-posed question. `mu_` is the exception and
#     raises AttributeError when `k == 0`, because a zero-width mean is
#     not an empty answer, it is a model that has no mean at all.
# llf_ : ndarray (batch_size,) float64
#     Log-likelihood at the fitted parameters, one per series.
# aic_, bic_ : ndarray (batch_size,) float64
#     DEVIATION 991 above.
# n_iter_ : ndarray (batch_size,) int32
#     L-BFGS iterations per series.
# retcode_ : ndarray (batch_size,) int32
#     0 is OPT_SUCCESS. A non-zero entry means that series ran to
#     `maxiter` without meeting the convergence bound, and it is NOT
#     raised on: a batch is fitted together and one bad series must not
#     deny the caller the other fifty. Read it.
# x_, x0_ : ndarray (batch_size, N) float32
#     The unconstrained optimum, and the starting point `estimate_x0`
#     produced, in the same packing. Neither is on cuML's Python surface.
#     They are here because a fit that goes wrong is nearly always a fit
#     that started wrong and `estimate_x0` is the half of this lane with
#     no reference oracle.
# n_obs_, batch_size_ : int
# k_ : int
#     0 or 1, what `trend` resolved to.
# complexity_ : int
#     `N`, the per-series parameter count.
# input_copied_ : bool
#     Whether `fit` had to copy `y` to reach float32 C order.
class ARIMA(NumericModeMixin):
    """Batched ARIMA and seasonal ARIMA fitted by maximum likelihood.

    Fits a separate (S)ARIMA model to every series in a batch at once, with a
    batched Kalman filter and a batched L-BFGS optimizer. The batch is not a
    multivariate model: each series gets its own parameters.

    Differences from statsmodels and cuML that change how you call it:

    - The data goes to ``fit``, not to the constructor: ``ARIMA(order=(1, 0,
      0)).fit(y)``. ``y`` is ``(batch_size, n_obs)``, one series per row; a
      1-D ``y`` is one series, and outputs are 2-D either way.
    - The default order is ``(1, 0, 0)`` (cuML uses ``(1, 1, 1)``, statsmodels
      ``(0, 0, 0)``).
    - ``trend`` follows statsmodels: None means an intercept (``'c'``) when the
      model is not differenced (``d + D == 0``) and none (``'n'``) otherwise.
      ``trend='c'`` is cuML's ``fit_intercept=True``.
    - In :meth:`predict`, ``end`` is excluded, as in cuML (statsmodels returns
      ``end`` itself).
    - Computation is float32; a float64 ``y`` is converted (and copied).
    - Only maximum likelihood (``method='ml'``) is available. Missing values
      (NaN) are not supported and are refused; prediction intervals are not
      available.

    Parameters
    ----------
    order : tuple of int (p, d, q), default (1, 0, 0)
        Non-seasonal AR order, differencing and MA order. ``p`` and ``q`` are at
        most 8 and ``d + D`` at most 2.
    seasonal_order : tuple of int (P, D, Q, s), default (0, 0, 0, 0)
        Seasonal AR order, differencing, MA order and period. A seasonal term
        needs ``s >= 2`` with ``s > p`` and ``s > q``. The state size is
        bounded: ``r = max(p + s P, q + s Q + 1)`` at most 5 and ``d + s D +
        r`` at most 8; larger orders, and an order with no parameters at all,
        are refused at fit.
    trend : {None, "n", "c"}, default None
        ``'c'`` fits an intercept, ``'n'`` none, None applies statsmodels' rule
        above. ``'t'``, ``'ct'`` and polynomial trends raise
        NotImplementedError; pass the time index as an ``exog`` column instead.
    method : {"ml"}, default "ml"
        Likelihood. ``'css'`` and ``'css-ml'`` are recognized but not
        implemented and raise an error at fit; other names raise ValueError.
    maxiter : int, default 1000
        L-BFGS iteration cap, at least 1.
    verbose : bool, default False
        Must be False; True raises NotImplementedError.
    output_type : None, default None
        Must be None (cuML's array-type selector is not supported).
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'`` or
        ``'fast'``); None uses the process default (see
        :func:`mojolearn.set_numeric_mode`). ``'fast'`` is available on Apple GPUs only.

    Attributes
    ----------
    params_ : Array of shape (batch_size, N), float32
        Fitted parameters per series, packed as ``mu`` (when there is an
        intercept), ``beta`` (one per regressor), ``ar`` (p), ``ma`` (q),
        ``sar`` (P), ``sma`` (Q), then ``sigma2``; ``N = p + q + P + Q + k +
        n_exog + 1``. ``numpy.asarray`` on a ``mojolearn.Array`` is zero-copy.
    mu_ : ndarray of shape (batch_size,), float32
        Intercepts. Raises AttributeError when the model has no intercept.
    beta_ : Array of shape (batch_size, n_exog), float32
        Regression coefficients. Raises AttributeError when the fit had no
        ``exog``.
    ar_, ma_, sar_, sma_ : ndarray of shape (batch_size, p), (batch_size, q), (batch_size, P), (batch_size, Q), float32
        Named blocks of ``params_``; zero-width when the order has no such
        block.
    sigma2_ : ndarray of shape (batch_size,), float32
        Innovation variance.
    llf_ : ndarray of shape (batch_size,), float64
        Log-likelihood at the fitted parameters.
    aic_, bic_ : ndarray of shape (batch_size,), float64
        ``2 N - 2 llf`` and ``log(T) N - 2 llf``, with ``T = n_obs - (d + s D)``.
    n_iter_ : ndarray of shape (batch_size,), int32
        L-BFGS iterations per series.
    retcode_ : ndarray of shape (batch_size,), int32
        0 when the series converged. A non-zero entry means that series reached
        ``maxiter`` without converging; this is reported, not raised, so check
        it.
    x_, x0_ : ndarray of shape (batch_size, N), float32
        The unconstrained optimum and the starting point, in the same packing.
    n_exog_ : int
        Number of exogenous regressors (0 without ``exog``).
    n_obs_, batch_size_ : int
        Shape of the fitted ``y``.
    k_ : int
        1 with an intercept, 0 without.
    complexity_ : int
        ``N``, the parameter count per series.
    input_copied_ : bool
        Whether ``fit`` had to copy ``y`` to reach float32 C order.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> y = np.random.default_rng(0).standard_normal((16, 200)).cumsum(axis=1).astype(np.float32)
    >>> model = mojolearn.ARIMA(order=(1, 1, 1)).fit(y)
    >>> model.forecast(10).shape
    (16, 10)
    """

    _BINDING = "_mojolearn_arima"

    def __init__(
        self,
        order=(1, 0, 0),
        *,
        seasonal_order=(0, 0, 0, 0),
        trend=None,
        method="ml",
        maxiter=1000,
        verbose=False,
        output_type=None,
    ):
        return _TSA(self, "arima_initialize", (order, seasonal_order, trend, method, maxiter, verbose, output_type), _TSA_CONTEXT)

    # -- the binding, and the tier it really is -----------------------------


    # -- fit ----------------------------------------------------------------

    # `ARIMA.fit` with `method='ml'` and `start_params=None`, which is
    # every arm this lane can reach.
    #
    # `y` is `(batch_size, n_obs)` float32, one series per ROW; a 1-D `y`
    # is one series. `exog` is `(batch_size, n_obs, n_exog)`, the
    # regressors beside every observation (see `_exog_array` for the
    # shorter spellings); without it the model has no regression.
    #
    # Returns `self`. Read `retcode_` before you believe a series: a batch
    # is fitted together and a series that ran to `maxiter` is reported,
    # not raised on.
    def fit(self, y, exog=None):
        """Fit one model per series.

        Parameters
        ----------
        y : array-like of shape (batch_size, n_obs) or (n_obs,)
            The series, one per row; a 1-D ``y`` is one series. Must be finite
            (NaN is refused, not treated as missing).
        exog : array-like of shape (batch_size, n_obs, n_exog) or None, default None
            Exogenous regressors beside every observation (regression with ARIMA
            errors, as in cuML; at most 17 regressors, finite). Shorter forms are
            accepted: ``(n_obs,)`` or ``(n_obs, n_exog)`` for a 1-D ``y``, and
            ``(batch_size, n_obs)`` for one regressor.

        Returns
        -------
        self
            Check ``retcode_``: a series that ran to ``maxiter`` is reported there,
            not raised on.
        """
        # `_series_major`, `_exog_array`, the fit and `_adopt_fit` in one
        # native call (bindings/tsa_arima_native.mojo)
        return _TSA(self, "arima_fit", (y, exog), _TSA_CONTEXT)


    # -- the named views into params_ ---------------------------------------


    @property
    def mu_(self):
        return _TSA(self, "arima_parameter", ("mu_",), _TSA_CONTEXT)

    @property
    def beta_(self):
        return _TSA(self, "arima_parameter", ("beta_",), _TSA_CONTEXT)

    @property
    def ar_(self):
        return _TSA(self, "arima_parameter", ("ar_",), _TSA_CONTEXT)

    @property
    def ma_(self):
        return _TSA(self, "arima_parameter", ("ma_",), _TSA_CONTEXT)

    @property
    def sar_(self):
        return _TSA(self, "arima_parameter", ("sar_",), _TSA_CONTEXT)

    @property
    def sma_(self):
        return _TSA(self, "arima_parameter", ("sma_",), _TSA_CONTEXT)

    @property
    def sigma2_(self):
        return _TSA(self, "arima_parameter", ("sigma2_",), _TSA_CONTEXT)

    # -- predict and forecast -----------------------------------------------


    # In-sample and out-of-sample prediction, `(batch_size, end - start)`.
    #
    # `exog` is the regressors' FUTURE values, `(batch_size, end - n_obs,
    # n_exog)`, required exactly when the model has a regression and `end >
    # n_obs`, and refused otherwise (cuML's rule and words).
    #
    # `end` IS EXCLUDED. That is cuML's convention, stated in the cuML
    # docstring ("Index where to end the predictions, excluded"), and it
    # is NOT statsmodels', where `end` is the last index RETURNED. This
    # sentence is on the class, on this method and in the two Mojo files
    # under it, because it is the one thing on this surface a caller can
    # get silently wrong by one. `predict(0, n_obs)` gives every in-sample
    # prediction; statsmodels' equivalent is `predict(0, n_obs - 1)`.
    #
    # `end=None` means `n_obs`, as in the reference. `end > n_obs` extends into a
    # forecast, which is what `forecast` is a name for.
    #
    # PREDICTIONS BEFORE `d + s * D` ARE NaN, and that is a value rather
    # than an omission (DEVIATION 676, the canonical quiet NaN
    # `0x7fc00000` on every vendor). Differencing consumes those
    # observations; cuML logs a warning and writes the same NaN.
    def predict(self, start=0, end=None, exog=None):
        """In-sample and out-of-sample predictions.

        ``end`` is excluded, as in cuML: ``predict(0, n_obs)`` returns every
        in-sample prediction (statsmodels' equivalent is ``predict(0, n_obs -
        1)``). ``end > n_obs`` continues into a forecast. Predictions before index
        ``d + s * D`` are NaN, because differencing consumes those observations.

        Parameters
        ----------
        start : int, default 0
            First index to predict.
        end : int or None, default None
            Index to stop before; None means ``n_obs``. Must be greater than
            ``start``.
        exog : array-like of shape (batch_size, end - n_obs, n_exog) or None, default None
            Future regressor values. Required when the model has regressors and
            ``end > n_obs``; refused otherwise.

        Returns
        -------
        Array of shape (batch_size, end - start), float32

        Raises
        ------
        ValueError
            If the model is not fitted, ``start >= end``, or ``exog`` is missing,
            unexpected or of the wrong width.
        """
        return _TSA(self, "arima_predict", (start, end, exog), _TSA_CONTEXT)

    # `(batch_size, steps)` out-of-sample forecasts, continuing each
    # series from its own last observation.
    #
    # In the reference this is literally `predict(n_obs, n_obs + steps)` and it is
    # that here too, so nothing in the answer depends on which of the two
    # you call. `exog` is the regressors' future values, `(batch_size,
    # steps, n_exog)`, required when the model has a regression. There is no NaN prefix in it: the in-sample kernel does
    # not launch at all when `start == n_obs`.
    def forecast(self, steps, exog=None):
        """Forecast ``steps`` past the end of every series.

        Identical to ``predict(n_obs, n_obs + steps)``.

        Parameters
        ----------
        steps : int
            Forecast horizon, at least 1.
        exog : array-like of shape (batch_size, steps, n_exog) or None, default None
            Future regressor values; required when the model has regressors.

        Returns
        -------
        Array of shape (batch_size, steps), float32
        """
        return _TSA(self, "arima_forecast", (steps, exog), _TSA_CONTEXT)

    # -- saved models (lane/inference-forecast-umap-pca, 2026-09-15) --------

    # Write the fitted model to `path` as an npz: the series `y` the
    # filter runs over (`(batch_size, n_obs)` float32), `params` in the
    # packed fitted order, `x`, `x0`, `fx` (float32), `n_iter`, `retcode`
    # (int32), `llf`, `aic`, `bic` (float64), `meta` `<i8` [p, d, q, P, D,
    # Q, s, k, batch_size, n_obs, maxiter, input_copied], `trend`, `method`
    # and `numeric_mode`, under format `mojolearn-arima-1`. A model fit with
    # `exog` (DEVIATION 998) is `mojolearn-arima-2`: the same members, the
    # fit's regressors as `exog` `(batch_size, n_obs, n_exog)` float32, and
    # `n_exog` appended to `meta`.
    #
    # A loaded model predicts and forecasts and answers every fitted
    # attribute (`params_`, `ar_`, `ma_`, `sar_`, `sma_`, `mu_`,
    # `sigma2_`, `llf_`, `aic_`, `bic_`, `n_iter_`, `retcode_`); it does
    # not refit. On a CPU-only install that is public inference through the
    # host binding (`mojolearn.host_model(path)`, or `ARIMA.load(path)`),
    # with no GPU and no CPU training.
    def save(self, path):
        """Save the fitted model to an ``.npz`` file.

        The file holds the training series (needed for prediction), the fitted
        parameters and diagnostics, and the settings. A loaded model predicts,
        forecasts and exposes every fitted attribute; it does not refit. On a
        CPU-only install a saved model can be used for inference with
        ``ARIMA.load(path)`` or ``mojolearn.host_model(path)``.

        Parameters
        ----------
        path : str or path-like
            Destination file.
        """
        return _ARCHIVE(self, "archive_save", (path,), _ARCHIVE_CONTEXT)

    @classmethod
    def load(cls, path):
        """Load a model written by :meth:`save`.

        Parameters
        ----------
        path : str or path-like
            File written by :meth:`save`.

        Returns
        -------
        ARIMA
            A fitted model that predicts and forecasts; it does not refit.

        Raises
        ------
        ValueError
            If the file is not a valid saved ARIMA model.
        """
        return _ARCHIVE(cls, "host_archive_load", (path,), _ARCHIVE_CONTEXT)

    def __repr__(self):
        return _TSA(self, "arima_repr", (), _TSA_CONTEXT)


def _archive_members(fmt, extra):
    """The members of one saved ARIMA format, in the 0.8.55 writer's order."""
    return (
        ("format", "format", fmt),
        ("estimator", "estimator"),
        ("mode", "numeric_mode", "mojolearn: "),
        ("lower", "trend", "trend"),
        ("str", "method", "method"),
        ("reshape", "y", "_y", (("int", "batch_size_"), ("int", "n_obs_"))),
        ("array", "params", "params_"),
        ("array", "x", "x_"),
        ("array", "x0", "x0_"),
        ("array", "fx", "fx_"),
        ("array", "n_iter", "n_iter_"),
        ("array", "retcode", "retcode_"),
        ("array", "llf", "llf_"),
        ("array", "aic", "aic_"),
        ("array", "bic", "bic_"),
        ("i8", "meta", (("int", ("order", 0)), ("int", ("order", 1)), ("int", ("order", 2)),
                        ("int", ("seasonal_order", 0)), ("int", ("seasonal_order", 1)),
                        ("int", ("seasonal_order", 2)), ("int", ("seasonal_order", 3)),
                        ("int", "k_"), ("int", "batch_size_"), ("int", "n_obs_"), ("int", "maxiter"),
                        ("bool_or", "input_copied_", False)) + extra[0]),
    ) + extra[1]


# A model fit without regressors (n_exog_ == 0) writes `mojolearn-arima-1`;
# with them, `mojolearn-arima-2` adds `n_exog` to `meta` and the `exog` member.
_native_archive_bridge.register(ARIMA, (_ARIMA_FORMAT, _ARIMA_FORMAT_EXOG), (
    ("need", "params_", "ValueError", "mojolearn ARIMA: call fit() before save"),
    ("when", "n_exog_", 0, _archive_members(_ARIMA_FORMAT, ((), ()))),
    ("unless", "n_exog_", 0, _archive_members(_ARIMA_FORMAT_EXOG, (
        (("int", "n_exog_"),),
        (("reshape", "exog", "_exog", (("int", "batch_size_"), ("int", "n_obs_"), ("int", "n_exog_"))),),
    ))),
))


# The class the AutoARIMA door builds its per-group models of.
_TSA_CONTEXT["arima_class"] = ARIMA


