# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Batched ARIMA on the GPU, backed by the ported cuML batched Kalman filter.

PRIVATE MODULE. `ARIMA` is re-exported from `mojolearn/__init__.py`.

WHAT LANDED HERE, AND WHAT USED TO STAND IN ITS PLACE. Until 2026-09-01
`python/mojolearn/__init__.py` carried an `_NOT_YET["ARIMA"]` entry saying
the lane had a likelihood, a gradient and a predict but NO `fit`, so an
`ARIMA` class would have to demand its own answer as an argument. That was
true and it is now false. `arima/impl/arima/estimate_x0.mojo` (the starting
parameters, over an own-written Householder QR that beats the normal
equations 7.4e-07 against 1.5e-04 and is strictly better on 6 of 6 series)
and `arima/impl/arima/batched_fit.mojo` (an own-written batched L-BFGS with
a shared line search, no scipy) closed that gap and are gated by
`arima/checks/fit_check.mojo` in BOTH numeric tiers. The `_NOT_YET` entry is
deleted rather than reworded, because the fix for a sentence explaining an
absence is to end the absence.

`ARIMA` IS BATCHED, AND THAT IS THE WHOLE POINT OF THE LANE
------------------------------------------------------------
`y` is 2-D, `(batch_size, n_obs)`. Every series in the batch is fitted at
once, with its OWN parameters, by one set of kernel launches; the batch is
not a convenience wrapper around a loop and it is not a multivariate model.
That is cuML's design (`ARIMAOrder`, `ARIMAParams` and every kernel in
`cpp/src/arima/` are indexed by a series id) and it is where the speed comes
from. It is ALSO the first thing that differs from statsmodels, whose
`ARIMA` takes ONE series, so the constructor and the methods below read like
statsmodels' and the shapes do not:

    statsmodels     ARIMA(y, order=(1,0,0)).fit()          y is (n_obs,)
    here            ARIMA(order=(1,0,0)).fit(y)            y is (batch, n_obs)

A 1-D `y` is accepted and treated as ONE series, because refusing it would
be pedantry; the returned arrays stay 2-D with a leading 1 either way, so no
shape here is ever a function of what the input's rank happened to be.

THE DATA GOES TO `fit`, NOT TO THE CONSTRUCTOR. Both upstreams put it in the
constructor (`ARIMA(endog, order=...)` in cuML, `ARIMA(y, order=...)` in
statsmodels). This package's other twenty-six estimators take their data in
`fit`, and an ARIMA that did not would be the only class here you could not
clone, re-use on a second batch, or hand to anything expecting the house
shape.

EVERY KNOB IS ON THE CONSTRUCTOR AND `fit` TAKES ONLY DATA, which is the
other divergence from both upstreams: `method` and `maxiter` are `fit`
arguments there and constructor arguments here, for the same reason.

WHAT IS NOT HERE
----------------
`AutoARIMA`. Its `p / q / P / Q / k` search and its information-criterion
arms are NOT PORTED (`arima/NOT_IMPLEMENTED.tsv`), and the differencing
half of that search IS ported and IS reachable, as
`mojolearn.select_d` and `mojolearn.kpss_test`.

CROSS-VENDOR STATUS, STATED PLAINLY BECAUSE IT IS THE LIBRARY'S HEADLINE
CLAIM AND THIS CLASS DOES NOT CARRY ALL OF IT. `arima/`'s identity card is
BIT-IDENTICAL ON THREE VENDORS at commit `221aa141`, 139 records, Apple M4
and NVIDIA and AMD MI325X (`bench/results/e1/CERT_2026-08-31.md`). That card
is `arima/arima_main.mojo`, which is the Kalman filter, the Jones transform
and their stages. THE FIT IS NOT IN IT. The fit landed after that leg and
the card was re-emitted BYTE IDENTICAL, which says the fit moved no stage
the card records; it does not say `fit` has ever run on a second vendor,
because it has not. `fit`'s own gates are 16 of 16 on ONE Apple M4 in both
tiers, and THIS class's own gate,
`python/mojolearn/tests/test_arima_surface.py`, printed green on that same
one box on 2026-09-02, 88 checks and 0 failed in each of its two tier
processes. That is still ONE VENDOR. A three-vendor run THROUGH THIS
SURFACE is OWED and nothing here may be described as certified across
vendors until it exists.

That property, where it exists at all, belongs to
`MOJOLEARN_NUMERIC_MODE=identical`. The FAST build, which is the default,
makes no cross-vendor claim of any kind.
"""

import math

import numpy as np

from . import _backend
from ._arrays import _addr, _addr_ro
from ._mode import NumericModeMixin

# The log-likelihood method, as `arima/estimator.mojo` numbers it. Only MLE
# is offered; the other two are REFUSED BY NAME in that file, and the names
# are cuML's own (`arima.pyx:944-946`).
_METHODS = {"ml": 0, "css": 1, "css-ml": 2}

#: What `arima_numeric_mode()` answers per tier, the `NUMERIC_*` constant in
#: `checks/numerics.mojo`. Duplicated from `_backend._MODE_CODE` on purpose:
#: the cross-check below is worth nothing if it reads its expectation from
#: the same object it is checking.
_MODE_CODE = {"fast": 0, "identical": 1, "deterministic": 2}


def _series_major(y, name):
    """A C-contiguous float32 `(batch_size, n_obs)` array of `y`.

    Returns `(array, batch_size, n_obs, copied)`. The caller MUST keep
    `array` alive across the Mojo call; the Mojo side borrows the address
    and holds nothing after it returns (`_arrays.py`).

    NO FINITENESS CHECK HAPPENS HERE, deliberately. A NaN or an infinity is
    refused BY NAME, with the flat index of the offender, in
    `arima/impl/arima/batched_arima.mojo::_refuse_non_finite`. Checking it
    here as well would make that refusal unreachable from Python and would
    silently take over a decision the ported code owns.
    """
    a = np.asarray(y)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.ndim != 2:
        raise ValueError(
            f"mojolearn ARIMA: {name} must be 1-D (one series) or 2-D "
            f"(batch_size, n_obs), got {a.ndim}-D shape {a.shape}"
        )
    if a.size == 0:
        raise ValueError(f"mojolearn ARIMA: {name} is empty, shape {a.shape}")
    copied = False
    if a.dtype != np.float32 or not a.flags["C_CONTIGUOUS"]:
        # Named rather than silent. On a long batch this is the dominant
        # cost of the call and a caller can avoid it by passing float32 in C
        # order. float64 in particular is CONVERTED, not run: Metal exposes
        # no float64 on the device and every kernel in this lane is float32
        # (DEVIATION 670). cuML's ARIMA is float64 ONLY, so these are
        # float32 answers to a float32 problem, never cuML's numbers.
        a = np.ascontiguousarray(a, dtype=np.float32)
        copied = True
    return a, int(a.shape[0]), int(a.shape[1]), copied


def _n_exog(exog):
    """The number of exogenous regressors the caller supplied, which is the
    number `arima/impl/tsa/arima_common.mojo::validate_order` refuses.

    THIS FUNCTION IS NOT A REFUSAL AND MUST NOT BECOME ONE. Exogenous
    regressors are genuinely unported: `ARIMAParams::beta`, `ARIMAOrder::
    n_exog`, `d_exog` / `d_exog_fut`, `obs_intercept`, the two
    `cublasgemmStridedBatched` calls at `batched_kalman.cu:930-972` and
    everything downstream of them have no port at all
    (`arima/NOT_IMPLEMENTED.tsv`). So the count is plumbed through as a
    NUMBER and the Mojo validator raises, which keeps that refusal reachable
    from every caller of the lane rather than from this file only. Any
    non-None `exog` yields at least 1 so that no shape can slip past as a
    zero.
    """
    if exog is None:
        return 0
    a = np.asarray(exog)
    if a.ndim >= 2 and a.shape[-1] >= 1:
        return int(a.shape[-1])
    return 1


class ARIMA(NumericModeMixin):
    """Batched ARIMA, backed by the ported cuML batched Kalman filter and an
    own-written batched L-BFGS (`arima/`, DEVIATIONS 670 to 687 and 990 to
    993; `arima/README.md`), in statsmodels' constructor shape.

    `y` IS 2-D, `(batch_size, n_obs)`, AND THAT DIFFERS FROM statsmodels.
    Every series in the batch gets its own parameters and they are fitted
    together, in one set of launches. A 1-D `y` is taken as one series; the
    outputs are 2-D either way. The module docstring says what else follows
    from the batch being the point.

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY, one line per parameter,
    because a parameter that is accepted and ignored is a wrong answer
    waiting for a caller (the house rule). The file named on a refusal is
    THE FILE THAT RAISES IT, and where that file is a `.mojo` one the
    exception arrives as a bare `Exception`, because a Mojo `Error` crossing
    `def_function` is not a named Python type.

        order (p, d, q)   honored   `p > 8` and `q > 8` are refused by
                                    `arima/impl/tsa/arima_common.mojo::
                                    validate_order`, in cuML's own words, as
                                    is `d + D > 2` and an order with no
                                    parameters at all
        seasonal_order    honored   `(P, D, Q, s)`. A seasonal term with
          (P, D, Q, s)              `s < 2`, and `s <= p` or `s <= q`, are
                                    refused by `validate_order`, again in
                                    cuML's words
        rd > 8            refused   `validate_order`. `rd = d + s*D +
                                    max(p + s*P, q + s*Q + 1)` selects
                                    cuML's BLOCK-PER-SERIES Kalman kernel
                                    (`batched_kalman.cu:335-745`), a
                                    different fold shape and an unported
                                    one. This is a bound on the ORDER, so it
                                    fires at fit, before any device work
        r > 5             refused   `validate_order`. `r = max(p + s*P,
                                    q + s*Q + 1)` selects their Schur /
                                    Francis-QR Lyapunov solver
                                    (`matrix.cuh:1899-1948`); the direct
                                    Kronecker solve is the `r <= 5` arm and
                                    is what this lane carries
        trend             honored   for None, 'n' and 'c' ONLY, and it is
                                    this class's spelling of cuML's
                                    `fit_intercept` (DEVIATION 993 below).
                                    't', 'ct' and a polynomial trend
                                    specification are REFUSED BY NAME by
                                    `_arima_impl.py`: a time trend is an
                                    exogenous column and exog has no port
        method            honored   for 'ml' ONLY. 'css' and 'css-ml' are
                                    REFUSED BY NAME by
                                    `arima/estimator.mojo::_refuse_method`,
                                    not here: the conditional sum of squares
                                    likelihood and its `truncate` parameter
                                    (`batched_arima.cu:271-391`) have no
                                    port. The string is turned into a code
                                    and passed through UNCLAMPED so that
                                    refusal stays reachable (DEVIATION 992)
        maxiter           honored   the L-BFGS iteration cap, cuML's
                                    `maxiter`, default 1000 as theirs.
                                    Refused below 1 by `arima/estimator.mojo`
        exog              refused   `arima/impl/tsa/arima_common.mojo::
                                    validate_order`, by name, as
                                    `n_exog != 0`. NOT refused here: this
                                    file COUNTS the columns and hands the
                                    count over, so the ported refusal is
                                    what fires. Exogenous regressors are
                                    unported end to end, `ARIMAParams` has
                                    no `beta` field anywhere in the lane
        verbose           refused   `_arima_impl.py`, for anything truthy.
                                    Upstream it selects LOG LINES; this port
                                    prints none, so accepting it would be
                                    accepting-and-ignoring
        output_type       refused   `_arima_impl.py`. A cuML-internal
                                    array-type selector; this package
                                    returns NumPy
        float64 y         CONVERTED to float32, and the conversion COPIES.
                                    Named rather than hidden. cuML's ARIMA
                                    is instantiated on `double` ONLY and
                                    `arima.pyx:326` checks its input to
                                    float64; Metal exposes no float64 on the
                                    device, so there is no double arm here
                                    to pick (DEVIATION 670). These are
                                    float32 answers to a float32 problem
        non-finite y      refused   `arima/impl/arima/batched_arima.mojo::
                                    _refuse_non_finite`, which names the
                                    flat index. cuML instead has a MISSING
                                    OBSERVATION path (`missing = isnan(yt)`
                                    and the four branches it guards,
                                    `batched_kalman.cu:191-246`) and that
                                    path is NOT PORTED, so a NaN here is
                                    refused rather than treated as missing
        level             absent    NOT A PARAMETER OF THIS CLASS, so
                                    passing it is a TypeError naming it.
                                    cuML's confidence intervals (the
                                    `confidence_intervals` kernel at
                                    `batched_kalman.cu:824-838` and the
                                    `P = T P T' + RR'` propagation beside
                                    it) are NOT PORTED
        simple_           absent    cuML's switch. This lane implements the
          differencing              `True` arm only and does not carry the
                                    other as a switch, so there is no value
                                    to accept
        start_params      absent    cuML's `set_fit_params` has no door
                                    here. Every fit starts from
                                    `estimate_x0`
        truncate          absent    read only by the CSS likelihood, which
                                    is refused
        h                 absent    the finite-difference step, pinned at
                                    2^-10 by DEVIATION 687. cuML's 1e-8 is
                                    BELOW float32 epsilon and collapses the
                                    gradient to zero, so this is not a knob
                                    a caller may turn

    THE DEFAULT ORDER IS (1, 0, 0) AND IT IS NEITHER UPSTREAM'S. cuML's is
    (1, 1, 1) and statsmodels' is (0, 0, 0), and the two cannot both be
    honored. (0, 0, 0) is not even reachable here: with `trend=None` it
    resolves to `k = 1` and fits a mean, and with `trend='n'` it is an order
    with no parameters at all, which `validate_order` refuses in cuML's own
    words. (1, 1, 1) is a differencing model, and a default that silently
    differences a caller's data is a default that changes what the numbers
    mean. (1, 0, 0) is the smallest model that fits something, and the right
    thing to do with it is to pass your own order.

    DEVIATION 993: `trend` IS THIS CLASS'S SPELLING OF cuML's
    `fit_intercept`, AND THE DEFAULT IS statsmodels' RULE, NOT cuML's.
    Upstream has a boolean `fit_intercept` defaulting to True, whatever `d`
    is. statsmodels has `trend`, and `trend=None` there resolves to 'c' when
    the series is not differenced and to 'n' when it is. This class takes
    statsmodels' spelling and statsmodels' default rule, so

        trend=None  ->  k = 1 if d + D == 0 else 0
        trend='n'   ->  k = 0
        trend='c'   ->  k = 1

    and a caller who wants cuML's default on a differenced series writes
    `trend='c'` explicitly. `k` is what reaches the kernels; it is
    `ARIMAOrder::k` and it adds one `mu` per series to the parameter vector.

    DEVIATION 990: THE FITTED MODEL CROSSES BACK TO THE HOST AND IS UPLOADED
    AGAIN AT EVERY `predict`. The binding retains no device pointer (the
    rule `bindings/_mojolearn_estimators.mojo` states and every binding in
    this package inherits), so `params_` is a host array and `predict` and
    `forecast` hand it back down with the training series. The consequences,
    both real: the training series is RETAINED by this object, because
    `predict` needs it and there is nowhere else for it to live; and every
    prediction re-runs the Kalman filter over the whole series rather than
    resuming a stored state, which is what cuML does too.

    DEVIATION 991: `aic_` AND `bic_` ARE COMPUTED ON THE HOST IN FLOAT64 AND
    ARE NOT A DEVICE ANSWER. cuML's `information_criterion`
    (`batched_arima.cu:592-618`) is NOT PORTED, and what it does beyond the
    log-likelihood is one `raft::stats::information_criterion_batched` unary
    op. That formula is transcribed here,

        aic = 2 * N - 2 * llf
        bic = log(T) * N - 2 * llf

    with `N` the parameter count `ARIMAOrder::complexity()` and `T` the
    number of observations AFTER differencing, `n_obs - (d + s * D)`, which
    is the `n_samples` their caller passes. `llf_` is the device's float32
    log-likelihood widened to float64; the two lines above are a summary of
    the answer, not the answer, and they are no part of any identity claim.
    AICc is not offered, because nothing in this lane needs it and an
    untested third arm is a liability.

    THE LOG-LIKELIHOOD IS EVALUATED AGAIN AT THE FITTED POINT, once, rather
    than recovered from the optimizer's objective. The optimizer minimizes
    `-loglike / (n_obs - 1)`, so inverting it would cost a float32 negate
    and a float32 divide whose rounding nobody has measured. That is
    `arima/estimator.mojo::_loglike_at`, and it is what cuML's own
    `information_criterion` does.

    Attributes
    ----------
    params_ : ndarray (batch_size, N) float32
        The fitted model, forward transformed, packed per series in
        `ARIMAParams::pack`'s order: `mu` (only when `k == 1`), then `ar`
        (p), `ma` (q), `sar` (P), `sma` (Q), then `sigma2`. `N = p + q + P +
        Q + k + 1`. This is exactly the array `predict` and `forecast` send
        back down.
    mu_, ar_, ma_, sar_, sma_, sigma2_ : ndarray float32
        Named blocks of `params_`. `mu_` and `sigma2_` are `(batch_size,)`;
        `ar_`, `ma_`, `sar_` and `sma_` are `(batch_size, p)`,
        `(batch_size, q)`, `(batch_size, P)` and `(batch_size, Q)`, and come
        back `(batch_size, 0)` where the order has no such block, which is
        an empty answer to a well-posed question. `mu_` is the exception and
        raises AttributeError when `k == 0`, because a zero-width mean is
        not an empty answer, it is a model that has no mean at all.
    llf_ : ndarray (batch_size,) float64
        Log-likelihood at the fitted parameters, one per series.
    aic_, bic_ : ndarray (batch_size,) float64
        DEVIATION 991 above.
    n_iter_ : ndarray (batch_size,) int32
        L-BFGS iterations per series.
    retcode_ : ndarray (batch_size,) int32
        0 is OPT_SUCCESS. A non-zero entry means that series ran to
        `maxiter` without meeting the convergence bound, and it is NOT
        raised on: a batch is fitted together and one bad series must not
        deny the caller the other fifty. Read it.
    x_, x0_ : ndarray (batch_size, N) float32
        The unconstrained optimum, and the starting point `estimate_x0`
        produced, in the same packing. Neither is on cuML's Python surface.
        They are here because a fit that goes wrong is nearly always a fit
        that started wrong and `estimate_x0` is the half of this lane with
        no upstream oracle.
    n_obs_, batch_size_ : int
    k_ : int
        0 or 1, what `trend` resolved to.
    complexity_ : int
        `N`, the per-series parameter count.
    input_copied_ : bool
        Whether `fit` had to copy `y` to reach float32 C order.
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
        p, d, q = _as_order(order, 3, "order")
        P, D, Q, s = _as_order(seasonal_order, 4, "seasonal_order")
        # NOTHING ABOUT THE ORDER IS VALIDATED HERE. `p > 8`, `d + D > 2`,
        # `s < 2` beside a seasonal term, `rd > 8`, `r > 5` and an order
        # with no parameters at all are every one of them refused by
        # `arima/impl/tsa/arima_common.mojo::validate_order`, in cuML's own
        # sentences, before any device context exists. A copy of those
        # bounds here would be a second place for them to be wrong.
        self.order = (p, d, q)
        self.seasonal_order = (P, D, Q, s)

        if trend is None:
            k = 0 if (d + D) else 1
        elif isinstance(trend, str) and trend.lower() in ("n", "c"):
            k = 1 if trend.lower() == "c" else 0
        elif isinstance(trend, str) and trend.lower() in ("t", "ct"):
            raise NotImplementedError(
                f"mojolearn ARIMA: trend={trend!r} is refused. A time trend "
                "is an exogenous regressor, and exogenous regressors are "
                "unported end to end in this lane: ARIMAParams has no `beta` "
                "field, ARIMAOrder.n_exog is refused at any non-zero value, "
                "and the two cublasgemmStridedBatched calls that apply them "
                "(batched_kalman.cu:930-972) have no port "
                "(arima/NOT_IMPLEMENTED.tsv). This class carries trend=None, "
                "'n' and 'c'; 'c' is cuML's fit_intercept=True"
            )
        else:
            raise NotImplementedError(
                f"mojolearn ARIMA: trend={trend!r} is refused. This class "
                "carries None (statsmodels' rule: 'c' when d + D == 0, "
                "otherwise 'n'), 'n' and 'c'. A polynomial trend "
                "specification is a list of exogenous columns and exog has "
                "no port (arima/NOT_IMPLEMENTED.tsv)"
            )
        self.trend = trend
        self.k_ = k

        # THE METHOD IS TRANSLATED, NOT JUDGED. Only the spelling is checked
        # here (cuML checks the same three strings at arima.pyx:944-946);
        # WHICH of them this port carries is `arima/estimator.mojo::
        # _refuse_method`'s decision, and the code goes down untouched so
        # that decision stays reachable (DEVIATION 992).
        if not isinstance(method, str):
            raise ValueError("mojolearn ARIMA: method is a name")
        m = method.lower()
        if m not in _METHODS:
            raise ValueError(
                f"mojolearn ARIMA: unknown method {method!r}; cuML's three "
                f"are {sorted(_METHODS)} and this port offers 'ml'"
            )
        self.method = m
        maxiter = int(maxiter)
        self.maxiter = maxiter
        if verbose:
            raise NotImplementedError(
                "mojolearn ARIMA: verbose is refused; upstream it selects "
                "log lines and this port prints none, so honoring it is "
                "impossible and accepting it would be accepting-and-ignoring"
            )
        self.verbose = False
        if output_type is not None:
            raise NotImplementedError(
                "mojolearn ARIMA: output_type is a cuML-internal array-type "
                "selector; this package returns NumPy"
            )
        self.output_type = None
        self.complexity_ = p + q + P + Q + k + 1

    # -- the binding, and the tier it really is -----------------------------

    def _extension(self):
        """The `_mojolearn_arima` binding for THIS estimator's tier, with
        the binary's own answer cross-checked against it.

        `_backend.load_set` already refuses a binary whose VENDOR disagrees
        with the directory it loaded from, and cross-checks the TIER through
        the gbdt binary only. This adds the tier read-back for this binary
        specifically, which is what `_svm_impl.py` does with
        `svm_numeric_mode()` and for the same reason: a wrong-arm
        measurement that is correctly labelled by accident is the failure
        the whole three-tier design exists to prevent.
        """
        mod = self._bind()
        want = getattr(self, "numeric_mode", None) or _backend.default_mode()
        fn = getattr(mod, "arima_numeric_mode", None)
        if fn is not None:
            got = int(fn())
            if got != _MODE_CODE.get(want):
                raise RuntimeError(
                    f"mojolearn ARIMA: numeric_mode={want!r} was requested "
                    f"but {mod.__name__} reports compile-time mode code "
                    f"{got}; the binary and the directory it sits in "
                    "disagree, rebuild it"
                )
        return mod

    # -- fit ----------------------------------------------------------------

    def fit(self, y, exog=None):
        """`ARIMA.fit` with `method='ml'` and `start_params=None`, which is
        every arm this lane can reach.

        `y` is `(batch_size, n_obs)` float32, one series per ROW; a 1-D `y`
        is one series. `exog` is refused by
        `arima/impl/tsa/arima_common.mojo::validate_order`, by name, and
        this method does not check it, it counts it.

        Returns `self`. Read `retcode_` before you believe a series: a batch
        is fitted together and a series that ran to `maxiter` is reported,
        not raised on.
        """
        arr, batch_size, n_obs, copied = _series_major(y, "y")
        p, d, q = self.order
        P, D, Q, s = self.seasonal_order
        N = self.complexity_

        params = np.empty(batch_size * N, dtype=np.float32)
        x = np.empty(batch_size * N, dtype=np.float32)
        x0 = np.empty(batch_size * N, dtype=np.float32)
        stats = np.empty(2 * batch_size, dtype=np.float32)
        flags = np.empty(2 * batch_size, dtype=np.int32)
        # EVERY ONE OF THOSE SIZES IS A FUNCTION OF (batch_size, n_obs,
        # order) ALONE, which is why this side can allocate before it calls.
        # There is no quantity in an ARIMA fit that is only known once the
        # solve finishes, so nothing here is a worst-case buffer the way
        # `SVR.fit`'s support-vector arrays are (DEVIATION 873).
        written = self._extension().arima_fit(
            _addr_ro(arr),
            _addr(params),
            _addr(x),
            _addr(x0),
            _addr(stats),
            _addr(flags),
            # ORDER MATCHES bindings/_mojolearn_arima.mojo::arima_fit_binding.
            # batch_size, n_obs, p, d, q, P, D, Q, s, k, n_exog, method,
            # max_iterations
            [batch_size, n_obs, p, d, q, P, D, Q, s, self.k_,
             _n_exog(exog), _METHODS[self.method], self.maxiter],
        )
        if int(written) != batch_size * N:
            raise RuntimeError(
                f"mojolearn ARIMA: the fit wrote {int(written)} parameters, "
                f"batch_size * N is {batch_size * N}; this side and "
                "arima/estimator.mojo disagree about ARIMAOrder.complexity()"
            )

        self.input_copied_ = copied
        self.batch_size_ = batch_size
        self.n_obs_ = n_obs
        self._y = arr
        self.params_ = np.ascontiguousarray(params.reshape(batch_size, N))
        self.x_ = np.ascontiguousarray(x.reshape(batch_size, N))
        self.x0_ = np.ascontiguousarray(x0.reshape(batch_size, N))
        self.n_iter_ = np.ascontiguousarray(flags[:batch_size])
        self.retcode_ = np.ascontiguousarray(flags[batch_size:])

        llf = np.asarray(stats[:batch_size], dtype=np.float64)
        self.llf_ = llf
        self.fx_ = np.ascontiguousarray(stats[batch_size:])
        # DEVIATION 991: the host half of cuML's information_criterion.
        # `T` is n_samples AFTER differencing, which is the number their
        # caller passes (`n_obs - order.n_diff()`).
        T = n_obs - (d + s * D)
        n_par = float(N)
        self.aic_ = 2.0 * n_par - 2.0 * llf
        self.bic_ = (math.log(T) if T > 0 else 0.0) * n_par - 2.0 * llf
        return self

    # -- the named views into params_ ---------------------------------------

    def _block(self, name, offset, width):
        if not hasattr(self, "params_"):
            raise AttributeError(f"mojolearn ARIMA: call fit() before {name}")
        return np.ascontiguousarray(self.params_[:, offset:offset + width])

    @property
    def mu_(self):
        if self.k_ == 0:
            raise AttributeError(
                "mojolearn ARIMA: there is no mu_ because this model has no "
                "intercept (trend resolved to 'n', k = 0). Construct it with "
                "trend='c' to fit one"
            )
        return self._block("mu_", 0, 1).reshape(-1)

    @property
    def ar_(self):
        return self._block("ar_", self.k_, self.order[0])

    @property
    def ma_(self):
        return self._block("ma_", self.k_ + self.order[0], self.order[2])

    @property
    def sar_(self):
        off = self.k_ + self.order[0] + self.order[2]
        return self._block("sar_", off, self.seasonal_order[0])

    @property
    def sma_(self):
        off = (self.k_ + self.order[0] + self.order[2]
               + self.seasonal_order[0])
        return self._block("sma_", off, self.seasonal_order[2])

    @property
    def sigma2_(self):
        return self._block("sigma2_", self.complexity_ - 1, 1).reshape(-1)

    # -- predict and forecast -----------------------------------------------

    def _check_fitted(self, who):
        if not hasattr(self, "params_"):
            raise ValueError(f"mojolearn ARIMA: call fit() before {who}")

    def predict(self, start=0, end=None, exog=None):
        """In-sample and out-of-sample prediction, `(batch_size, end - start)`.

        `end` IS EXCLUDED. That is cuML's convention, stated in their own
        docstring ("Index where to end the predictions, excluded"), and it
        is NOT statsmodels', where `end` is the last index RETURNED. This
        sentence is on the class, on this method and in the two Mojo files
        under it, because it is the one thing on this surface a caller can
        get silently wrong by one. `predict(0, n_obs)` gives every in-sample
        prediction; statsmodels' equivalent is `predict(0, n_obs - 1)`.

        `end=None` means `n_obs`, as upstream. `end > n_obs` extends into a
        forecast, which is what `forecast` is a name for.

        PREDICTIONS BEFORE `d + s * D` ARE NaN, and that is a value rather
        than an omission (DEVIATION 676, the canonical quiet NaN
        `0x7fc00000` on every vendor). Differencing consumes those
        observations; cuML logs a warning and writes the same NaN.
        """
        self._check_fitted("predict")
        if end is None:
            end = self.n_obs_
        start = int(start)
        end = int(end)
        p, d, q = self.order
        P, D, Q, s = self.seasonal_order
        width = end - start
        if width <= 0:
            # Raised here only because the OUTPUT BUFFER has to be allocated
            # before the call and a non-positive width has no allocation.
            # `arima/estimator.mojo` refuses the same thing again in cuML's
            # own words, and reaches it from any other caller.
            raise ValueError(
                f"mojolearn ARIMA: need start < end, got start={start}, "
                f"end={end}. `end` is EXCLUDED here, as it is in cuML; "
                "statsmodels' end is the last index returned"
            )
        out = np.empty(self.batch_size_ * width, dtype=np.float32)
        y = self._y
        pr = self.params_
        self._extension().arima_predict(
            _addr_ro(y),
            _addr_ro(pr),
            _addr(out),
            # ORDER MATCHES
            # bindings/_mojolearn_arima.mojo::arima_predict_binding.
            # batch_size, n_obs, start, end, p, d, q, P, D, Q, s, k, n_exog
            [self.batch_size_, self.n_obs_, start, end, p, d, q, P, D, Q, s,
             self.k_, _n_exog(exog)],
        )
        return out.reshape(self.batch_size_, width)

    def forecast(self, steps, exog=None):
        """`(batch_size, steps)` out-of-sample forecasts, continuing each
        series from its own last observation.

        Upstream this is literally `predict(n_obs, n_obs + steps)` and it is
        that here too, so nothing in the answer depends on which of the two
        you call. There is no NaN prefix in it: the in-sample kernel does
        not launch at all when `start == n_obs`.
        """
        self._check_fitted("forecast")
        steps = int(steps)
        if steps < 1:
            raise ValueError(
                f"mojolearn ARIMA: steps must be >= 1, got {steps}"
            )
        p, d, q = self.order
        P, D, Q, s = self.seasonal_order
        out = np.empty(self.batch_size_ * steps, dtype=np.float32)
        y = self._y
        pr = self.params_
        self._extension().arima_forecast(
            _addr_ro(y),
            _addr_ro(pr),
            _addr(out),
            # ORDER MATCHES
            # bindings/_mojolearn_arima.mojo::arima_forecast_binding.
            # batch_size, n_obs, n_steps, p, d, q, P, D, Q, s, k, n_exog,
            # reserved (MUST BE 0; the slot cuML's `level` would take, and
            # `level` is NOT PORTED)
            [self.batch_size_, self.n_obs_, steps, p, d, q, P, D, Q, s,
             self.k_, _n_exog(exog), 0],
        )
        return out.reshape(self.batch_size_, steps)

    def __repr__(self):
        return (
            "ARIMA(order={}, seasonal_order={}, trend={!r}, method={!r})"
            .format(self.order, self.seasonal_order, self.trend, self.method)
        )


def _as_order(value, width, name):
    """`(p, d, q)` or `(P, D, Q, s)` as a tuple of non-negative ints.

    A CONVERSION, NOT A POLICY. Everything about whether the numbers are a
    LEGAL order is `validate_order`'s, in Mojo, in cuML's words. What is
    refused here is a value that is not an order-shaped tuple of integers at
    all, which cannot reach the Mojo side as anything meaningful.
    """
    try:
        t = tuple(value)
    except TypeError:
        raise ValueError(
            f"mojolearn ARIMA: {name} must be a tuple of {width} "
            f"non-negative integers, got {value!r}"
        ) from None
    if len(t) != width:
        raise ValueError(
            f"mojolearn ARIMA: {name} must have {width} entries, got "
            f"{len(t)} ({value!r})"
        )
    out = []
    for v in t:
        i = int(v)
        if i != v or i < 0:
            raise ValueError(
                f"mojolearn ARIMA: {name} entries must be non-negative "
                f"integers, got {value!r}"
            )
        out.append(i)
    return tuple(out)
