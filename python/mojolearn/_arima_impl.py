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
`AutoARIMA`. Its `p / q / P / Q / k` search and its information-criterion
arms are NOT IMPLEMENTED (`arima/NOT_IMPLEMENTED.tsv`), and the differencing
half of that search IS implemented and IS reachable, as
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

from . import _portable_math as math

from . import _backend
from ._array import Array
from ._buffer import addr, addr_ro, as_f32_c, empty
from ._bufcheck import nelems, probe
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

#: What `arima_numeric_mode()` answers per tier, the `NUMERIC_*` constant in
#: `checks/numerics.mojo`. Duplicated from `_backend._MODE_CODE` on purpose:
#: the cross-check below is worth nothing if it reads its expectation from
#: the same object it is checking.
_MODE_CODE = {"fast": 0, "identical": 1, "deterministic": 2}


def _series_major(y, name):
    """A C-contiguous float32 `(batch_size, n_obs)` array of `y`.

    Returns `(array, batch_size, n_obs, copied)`. The caller MUST keep
    `array` alive across the Mojo call; the Mojo side borrows the address
    and holds nothing after it returns (`_buffer.py`).

    DEVIATION 2415: `y` is read through the buffer protocol -- a NumPy
    array, an `array.array`, a `mojolearn.Array`; a plain nested list is
    materialized once as float32 the way `np.asarray` used to materialize
    it -- and `_buffer.as_f32_c` does the conversion, reporting the copy.

    NO FINITENESS CHECK HAPPENS HERE, deliberately. A NaN or an infinity is
    refused BY NAME, with the flat index of the offender, in
    `arima/impl/batched_arima.mojo::_refuse_non_finite`. Checking it
    here as well would make that refusal unreachable from Python and would
    silently take over a decision the implemented code owns.
    """
    try:
        pb = probe(y)
    except TypeError:
        try:
            y = Array.from_list(y, "<f4")
        except Exception:
            raise ValueError(
                f"mojolearn ARIMA: {name} must be 1-D (one series) or 2-D "
                f"(batch_size, n_obs) -- an array or a nested list of "
                f"numbers -- got {type(y).__name__}"
            ) from None
        pb = probe(y)
    if pb.ndim not in (1, 2):
        raise ValueError(
            f"mojolearn ARIMA: {name} must be 1-D (one series) or 2-D "
            f"(batch_size, n_obs), got {pb.ndim}-D shape {pb.shape}"
        )
    if nelems(pb.shape) == 0:
        raise ValueError(f"mojolearn ARIMA: {name} is empty, shape {pb.shape}")
    # Named rather than silent. On a long batch the copy is the dominant
    # cost of the call and a caller can avoid it by passing float32 in C
    # order. float64 in particular is CONVERTED, not run: Metal exposes no
    # float64 on the device and every kernel in this lane is float32
    # (DEVIATION 670). cuML's ARIMA is float64 ONLY, so these are float32
    # answers to a float32 problem, never cuML's numbers.
    a, copied = as_f32_c(y, ndim=pb.ndim, name=name)
    if pb.ndim == 1:
        a = a.reshape((1, a.shape[0]))
    return a, int(a.shape[0]), int(a.shape[1]), copied


def _input_ndim(y):
    """The rank the caller handed `fit` for `y`, before `_series_major` makes
    it 2-D: the buffer's own `ndim`, or the nesting depth of a list."""
    try:
        return int(probe(y).ndim)
    except TypeError:
        depth, v = 0, y
        while isinstance(v, (list, tuple)):
            depth += 1
            v = v[0] if len(v) else None
        return depth


def _exog_array(exog, batch_size, n_rows, y_ndim, name):
    """A C-contiguous float32 `(batch_size, n_rows, n_exog)` array of `exog`.

    Returns `(array, n_exog)`. DEVIATION 996: `exog` is `(batch_size, n_obs,
    n_exog)`, `y`'s shape with a regressor axis last, which is statsmodels'
    `(n_obs, k_exog)` with the batch in front. cuML's is `(n_obs, n_exog *
    batch_size)` in Fortran order; the Mojo boundary
    (`bindings/arima_exog_layout.mojo`) permutes to that layout. Two shorter
    spellings are accepted because they are unambiguous:

        y 1-D (one series)   exog (n_rows,) one regressor, or (n_rows, n_exog)
        y 2-D                exog (batch_size, n_rows), one regressor

    A non-finite value is NOT checked here: it is refused by name, with the
    series, row and regressor, in `bindings/arima_exog_layout.mojo`
    (DEVIATION 997), which every binding reaches."""
    try:
        pb = probe(exog)
    except TypeError:
        try:
            exog = Array.from_list(exog, "<f4")
        except Exception:
            raise ValueError(
                f"mojolearn ARIMA: {name} must be an array or a nested list of "
                f"numbers, got {type(exog).__name__}"
            ) from None
        pb = probe(exog)
    shape = tuple(int(v) for v in pb.shape)
    if len(shape) == 3:
        want = (batch_size, n_rows)
        got = shape[:2]
    elif len(shape) == 2 and y_ndim == 1:
        want = (n_rows,)
        got = shape[:1]
        shape = (1, shape[0], shape[1])
    elif len(shape) == 2:
        want = (batch_size, n_rows)
        got = shape
        shape = (shape[0], shape[1], 1)
    elif len(shape) == 1 and y_ndim == 1:
        want = (n_rows,)
        got = shape
        shape = (1, shape[0], 1)
    else:
        raise ValueError(
            f"mojolearn ARIMA: {name} must be (batch_size, n_rows, n_exog); a "
            f"1-D y also takes (n_rows,) or (n_rows, n_exog) and a 2-D y takes "
            f"(batch_size, n_rows) for one regressor. Got {len(pb.shape)}-D "
            f"shape {tuple(pb.shape)} beside a {y_ndim}-D y"
        )
    if tuple(got) != tuple(want):
        raise ValueError(
            f"mojolearn ARIMA: dimensions mismatch, {name} has shape "
            f"{tuple(pb.shape)} and must lead with {want} "
            f"(batch_size={batch_size}, rows={n_rows})"
        )
    if shape[2] < 1:
        raise ValueError(f"mojolearn ARIMA: {name} has no regressor columns, shape {tuple(pb.shape)}")
    a, _copied = as_f32_c(exog, ndim=len(pb.shape), name=name)
    return a.reshape(shape), int(shape[2])


#: The address every binding call passes for an exogenous input that is not
#: there. One float, never read (`n_exog == 0`, or no forecast steps).
def _no_exog():
    return empty((1,), "<f4")


class ARIMA(NumericModeMixin):
    """Batched ARIMA, backed by a batched Kalman filter and a
    batched L-BFGS (`arima/`, DEVIATIONS 670 to 687 and 990 to
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
                                    validate_order`, with the reference's message, as
                                    is `d + D > 2` and an order with no
                                    parameters at all
        seasonal_order    honored   `(P, D, Q, s)`. A seasonal term with
          (P, D, Q, s)              `s < 2`, and `s <= p` or `s <= q`, are
                                    refused by `validate_order`, again with
                                    the reference's message
        rd > 8            refused   `validate_order`. `rd = d + s*D +
                                    max(p + s*P, q + s*Q + 1)` selects
                                    cuML's BLOCK-PER-SERIES Kalman kernel
                                    (`batched_kalman.cu:335-745`), a
                                    different fold shape and an unimplemented
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
                                    `_arima_impl.py`: cuML has no time trend
                                    to hold one to; pass the time index as an
                                    `exog` column instead
        method            honored   for 'ml' ONLY. 'css' and 'css-ml' are
                                    REFUSED BY NAME by
                                    `arima/estimator.mojo::_refuse_method`,
                                    not here: the conditional sum of squares
                                    likelihood and its `truncate` parameter
                                    (`batched_arima.cu:271-391`) have no
                                    implementation. The string is turned into a code
                                    and passed through UNCLAMPED so that
                                    refusal stays reachable (DEVIATION 992)
        maxiter           honored   the L-BFGS iteration cap, cuML's
                                    `maxiter`, default 1000 as theirs.
                                    Refused below 1 by `arima/estimator.mojo`
        exog              honored   on `fit(y, exog)`, `predict(start, end,
                                    exog)` and `forecast(steps, exog)`
                                    (lane/arima-exog, 2026-09-15): regression
                                    with ARIMA errors as cuML's (`beta` packed
                                    after `mu`, the regressors differenced like
                                    `y`, `beta` started by least squares before
                                    the ARMA fit, `x_t beta` added to every
                                    prediction). `(batch_size, n_obs, n_exog)`
                                    (DEVIATION 996); at most 17 regressors
                                    (DEVIATION 994, validate_order); a
                                    non-finite one refused by name (DEVIATION
                                    997). cuML's `exog` is a CONSTRUCTOR
                                    argument; here it goes to `fit` with `y`
        verbose           refused   `_arima_impl.py`, for anything truthy.
                                    In the reference it selects LOG LINES; this implementation
                                    prints none, so accepting it would be
                                    accepting-and-ignoring
        output_type       refused   `_arima_impl.py`. A cuML-internal
                                    array-type selector; this package
                                    returns `mojolearn.Array` (zero-copy
                                    under `numpy.asarray`)
        float64 y         CONVERTED to float32, and the conversion COPIES.
                                    Named rather than hidden. cuML's ARIMA
                                    is instantiated on `double` ONLY and
                                    `arima.pyx:326` checks its input to
                                    float64; Metal exposes no float64 on the
                                    device, so there is no double arm here
                                    to pick (DEVIATION 670). These are
                                    float32 answers to a float32 problem
        non-finite y      refused   `arima/impl/batched_arima.mojo::
                                    _refuse_non_finite`, which names the
                                    flat index. cuML instead has a MISSING
                                    OBSERVATION path (`missing = isnan(yt)`
                                    and the four branches it guards,
                                    `batched_kalman.cu:191-246`) and that
                                    path is NOT IMPLEMENTED, so a NaN here is
                                    refused rather than treated as missing
        level             absent    NOT A PARAMETER OF THIS CLASS, so
                                    passing it is a TypeError naming it.
                                    cuML's confidence intervals (the
                                    `confidence_intervals` kernel at
                                    `batched_kalman.cu:824-838` and the
                                    `P = T P T' + RR'` propagation beside
                                    it) are NOT IMPLEMENTED
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

    THE DEFAULT ORDER IS (1, 0, 0) AND IT IS NEITHER REFERENCE'S. cuML's is
    (1, 1, 1) and statsmodels' is (0, 0, 0), and the two cannot both be
    honored. (0, 0, 0) is not even reachable here: with `trend=None` it
    resolves to `k = 1` and fits a mean, and with `trend='n'` it is an order
    with no parameters at all, which `validate_order` refuses with the
    reference's message. (1, 1, 1) is a differencing model, and a default that silently
    differences a caller's data is a default that changes what the numbers
    mean. (1, 0, 0) is the smallest model that fits something, and the right
    thing to do with it is to pass your own order.

    DEVIATION 993: `trend` IS THIS CLASS'S SPELLING OF cuML's
    `fit_intercept`, AND THE DEFAULT IS statsmodels' RULE, NOT cuML's.
    cuML has a boolean `fit_intercept` defaulting to True, whatever `d`
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
    (`batched_arima.cu:592-618`) is NOT IMPLEMENTED, and what it does beyond the
    log-likelihood is one `raft::stats::information_criterion_batched` unary
    op. This computes the same formula,

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

    Attributes (every array below is a `mojolearn.Array`; `numpy.asarray`
    on it is zero-copy -- DEVIATION 2416)
    ----------
    params_ : Array (batch_size, N) float32
        The fitted model, forward transformed, packed per series in
        `ARIMAParams::pack`'s order: `mu` (only when `k == 1`), `beta`
        (n_exog), then `ar` (p), `ma` (q), `sar` (P), `sma` (Q), then
        `sigma2`. `N = p + q + P + Q + k + n_exog + 1`. This is exactly the
        array `predict` and `forecast` send back down.
    beta_ : Array (batch_size, n_exog) float32
        The regression coefficients, one per regressor per series. Raises
        AttributeError when the fit had no `exog`, as `mu_` does without an
        intercept.
    n_exog_ : int
        The regressor count the fit was handed, 0 without `exog`.
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
        no reference oracle.
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
        # `arima/impl/tsa/arima_common.mojo::validate_order`, with the reference's
        # messages, before any device context exists. A copy of those
        # bounds here would be a second place for them to be wrong.
        # The caller's own tuple when it already is the normalized one, so
        # `get_params` hands `clone` back the object it was given.
        self.order = order if _plain_order(order, (p, d, q)) else (p, d, q)
        self.seasonal_order = (seasonal_order if _plain_order(seasonal_order, (P, D, Q, s))
                               else (P, D, Q, s))

        if trend is None:
            k = 0 if (d + D) else 1
        elif isinstance(trend, str) and trend.lower() in ("n", "c"):
            k = 1 if trend.lower() == "c" else 0
        elif isinstance(trend, str) and trend.lower() in ("t", "ct"):
            raise NotImplementedError(
                f"mojolearn ARIMA: trend={trend!r} is refused. cuML has no "
                "time trend (its only switch is fit_intercept), so there is no "
                "reference to hold one to; a time trend is a regressor, so "
                "pass the time index as an exog column to fit, predict and "
                "forecast. This class carries trend=None, 'n' and 'c'; 'c' is "
                "cuML's fit_intercept=True"
            )
        else:
            raise NotImplementedError(
                f"mojolearn ARIMA: trend={trend!r} is refused. This class "
                "carries None (statsmodels' rule: 'c' when d + D == 0, "
                "otherwise 'n'), 'n' and 'c'. A polynomial trend "
                "specification is a list of regressors: pass them as exog"
            )
        self.trend = trend
        self.k_ = k

        # THE METHOD IS TRANSLATED, NOT JUDGED. Only the spelling is checked
        # here (cuML checks the same three strings at arima.pyx:944-946);
        # WHICH of them this implementation carries is `arima/estimator.mojo::
        # _refuse_method`'s decision, and the code goes down untouched so
        # that decision stays reachable (DEVIATION 992).
        if not isinstance(method, str):
            raise ValueError("mojolearn ARIMA: method is a name")
        m = method if method == method.lower() else method.lower()
        if m not in _METHODS:
            raise ValueError(
                f"mojolearn ARIMA: unknown method {method!r}; cuML's three "
                f"are {sorted(_METHODS)} and this implementation offers 'ml'"
            )
        self.method = m
        maxiter = int(maxiter)
        self.maxiter = maxiter
        if verbose:
            raise NotImplementedError(
                "mojolearn ARIMA: verbose is refused; upstream it selects "
                "log lines and this implementation prints none, so honoring it is "
                "impossible and accepting it would be accepting-and-ignoring"
            )
        self.verbose = False
        if output_type is not None:
            raise NotImplementedError(
                "mojolearn ARIMA: output_type is a cuML-internal array-type "
                "selector; this package returns mojolearn.Array (zero-copy "
                "under numpy.asarray)"
            )
        self.output_type = None
        self.n_exog_ = 0
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
        is one series. `exog` is `(batch_size, n_obs, n_exog)`, the
        regressors beside every observation (see `_exog_array` for the
        shorter spellings); without it the model has no regression.

        Returns `self`. Read `retcode_` before you believe a series: a batch
        is fitted together and a series that ran to `maxiter` is reported,
        not raised on.
        """
        y_ndim = _input_ndim(y)
        arr, batch_size, n_obs, copied = _series_major(y, "y")
        p, d, q = self.order
        P, D, Q, s = self.seasonal_order
        if exog is None:
            ex, n_exog = _no_exog(), 0
        else:
            ex, n_exog = _exog_array(exog, batch_size, n_obs, y_ndim, "exog")
        N = p + q + P + Q + self.k_ + n_exog + 1

        params = empty((batch_size * N,), "<f4")
        x = empty((batch_size * N,), "<f4")
        x0 = empty((batch_size * N,), "<f4")
        stats = empty((2 * batch_size,), "<f4")
        flags = empty((2 * batch_size,), "<i4")
        # EVERY ONE OF THOSE SIZES IS A FUNCTION OF (batch_size, n_obs,
        # order) ALONE, which is why this side can allocate before it calls.
        # There is no quantity in an ARIMA fit that is only known once the
        # solve finishes, so nothing here is a worst-case buffer the way
        # `SVR.fit`'s support-vector arrays are (DEVIATION 873).
        written = self._extension().arima_fit(
            addr_ro(arr, name="y"),
            addr_ro(ex, name="exog"),
            addr(params, name="params"),
            addr(x, name="x"),
            addr(x0, name="x0"),
            addr(stats, name="stats"),
            addr(flags, name="flags"),
            # ORDER MATCHES bindings/_mojolearn_arima.mojo::arima_fit_binding.
            # batch_size, n_obs, p, d, q, P, D, Q, s, k, n_exog, method,
            # max_iterations
            [batch_size, n_obs, p, d, q, P, D, Q, s, self.k_,
             n_exog, _METHODS[self.method], self.maxiter],
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
        self._y_ndim = y_ndim
        self.n_exog_ = n_exog
        self.complexity_ = N
        self._exog = ex if n_exog else None
        # DEVIATION 2416: every attribute is a `mojolearn.Array`. `reshape`
        # is a C-order view over the flat buffer the kernel wrote; a slice
        # copies (the Array contract), which is what `ascontiguousarray`
        # of a slice did.
        self.params_ = params.reshape((batch_size, N))
        self.x_ = x.reshape((batch_size, N))
        self.x0_ = x0.reshape((batch_size, N))
        self.n_iter_ = flags[:batch_size]
        self.retcode_ = flags[batch_size:]

        llf = stats[:batch_size].astype("<f8")
        self.llf_ = llf
        self.fx_ = stats[batch_size:]
        # DEVIATION 991: the host half of cuML's information_criterion.
        # `T` is n_samples AFTER differencing, which is the number their
        # caller passes (`n_obs - order.n_diff()`). Two float64 host
        # expressions per series, written out in Python over the
        # batch (O(batch_size), no part of any identity claim).
        T = n_obs - (d + s * D)
        n_par = float(N)
        log_t = math.log(T) if T > 0 else 0.0
        llf_list = llf.tolist()
        self.aic_ = Array.from_list(
            [2.0 * n_par - 2.0 * v for v in llf_list], "<f8")
        self.bic_ = Array.from_list(
            [log_t * n_par - 2.0 * v for v in llf_list], "<f8")
        return self

    # -- the named views into params_ ---------------------------------------

    def _block(self, name, offset, width):
        if not hasattr(self, "params_"):
            raise AttributeError(f"mojolearn ARIMA: call fit() before {name}")
        # A tuple-of-slices index COPIES into a fresh C-contiguous Array
        # (DEVIATION 2417), exactly what `ascontiguousarray` of the NumPy
        # column slice produced.
        return self.params_[:, offset:offset + width]

    @property
    def mu_(self):
        if self.k_ == 0:
            raise AttributeError(
                "mojolearn ARIMA: there is no mu_ because this model has no "
                "intercept (trend resolved to 'n', k = 0). Construct it with "
                "trend='c' to fit one"
            )
        return self._block("mu_", 0, 1).ravel()

    @property
    def beta_(self):
        if getattr(self, "n_exog_", 0) == 0:
            raise AttributeError(
                "mojolearn ARIMA: there is no beta_ because this model was fit "
                "without exogenous regressors; pass exog to fit"
            )
        return self._block("beta_", self.k_, self.n_exog_)

    @property
    def ar_(self):
        return self._block("ar_", self.k_ + self.n_exog_, self.order[0])

    @property
    def ma_(self):
        return self._block("ma_", self.k_ + self.n_exog_ + self.order[0], self.order[2])

    @property
    def sar_(self):
        off = self.k_ + self.n_exog_ + self.order[0] + self.order[2]
        return self._block("sar_", off, self.seasonal_order[0])

    @property
    def sma_(self):
        off = (self.k_ + self.n_exog_ + self.order[0] + self.order[2]
               + self.seasonal_order[0])
        return self._block("sma_", off, self.seasonal_order[2])

    @property
    def sigma2_(self):
        return self._block("sigma2_", self.complexity_ - 1, 1).ravel()

    # -- predict and forecast -----------------------------------------------

    def _check_fitted(self, who):
        if not hasattr(self, "params_"):
            raise ValueError(f"mojolearn ARIMA: call fit() before {who}")

    def _future_exog(self, exog, end):
        """The addresses' arrays for `predict(start, end, exog)`: the fit's
        regressors and their future values. cuML's three checks
        (`arima.pyx:688-696`), in its order and its words, then the shape
        (`:713-722`)."""
        n_exog = getattr(self, "n_exog_", 0)
        if n_exog > 0 and end > self.n_obs_ and exog is None:
            raise ValueError(
                "mojolearn ARIMA: the model was fit with a regression component, "
                "so future values must be provided via `exog`"
            )
        if n_exog == 0 and exog is not None:
            raise ValueError(
                "mojolearn ARIMA: a value was given for `exog` but the model was "
                "fit without any regression component"
            )
        if end <= self.n_obs_ and exog is not None:
            raise ValueError(
                "mojolearn ARIMA: a value was given for `exog` but only in-sample "
                "predictions were requested"
            )
        if n_exog == 0:
            return _no_exog(), _no_exog()
        if end <= self.n_obs_:
            return self._exog, _no_exog()
        fut, got = _exog_array(exog, self.batch_size_, end - self.n_obs_,
                               getattr(self, "_y_ndim", 2), "exog")
        if got != n_exog:
            raise ValueError(
                f"mojolearn ARIMA: dimensions mismatch, `exog` has {got} "
                f"regressor column(s) and the model was fit with {n_exog}"
            )
        return self._exog, fut

    def predict(self, start=0, end=None, exog=None):
        """In-sample and out-of-sample prediction, `(batch_size, end - start)`.

        `exog` is the regressors' FUTURE values, `(batch_size, end - n_obs,
        n_exog)`, required exactly when the model has a regression and `end >
        n_obs`, and refused otherwise (cuML's rule and words).

        `end` IS EXCLUDED. That is cuML's convention, stated in the cuML
        docstring ("Index where to end the predictions, excluded"), and it
        is NOT statsmodels', where `end` is the last index RETURNED. This
        sentence is on the class, on this method and in the two Mojo files
        under it, because it is the one thing on this surface a caller can
        get silently wrong by one. `predict(0, n_obs)` gives every in-sample
        prediction; statsmodels' equivalent is `predict(0, n_obs - 1)`.

        `end=None` means `n_obs`, as in the reference. `end > n_obs` extends into a
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
            # `arima/estimator.mojo` refuses the same thing again with the
            # reference's message, and reaches it from any other caller.
            raise ValueError(
                f"mojolearn ARIMA: need start < end, got start={start}, "
                f"end={end}. `end` is EXCLUDED here, as it is in cuML; "
                "statsmodels' end is the last index returned"
            )
        ex, fut = self._future_exog(exog, end)
        out = empty((self.batch_size_ * width,), "<f4")
        y = self._y
        pr = self.params_
        self._extension().arima_predict(
            addr_ro(y, name="y"),
            addr_ro(ex, name="exog"),
            addr_ro(fut, name="exog (future values)"),
            addr_ro(pr, name="params"),
            addr(out, name="out"),
            # ORDER MATCHES
            # bindings/_mojolearn_arima.mojo::arima_predict_binding.
            # batch_size, n_obs, start, end, p, d, q, P, D, Q, s, k, n_exog
            [self.batch_size_, self.n_obs_, start, end, p, d, q, P, D, Q, s,
             self.k_, self.n_exog_],
        )
        return out.reshape((self.batch_size_, width))

    def forecast(self, steps, exog=None):
        """`(batch_size, steps)` out-of-sample forecasts, continuing each
        series from its own last observation.

        In the reference this is literally `predict(n_obs, n_obs + steps)` and it is
        that here too, so nothing in the answer depends on which of the two
        you call. `exog` is the regressors' future values, `(batch_size,
        steps, n_exog)`, required when the model has a regression. There is no NaN prefix in it: the in-sample kernel does
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
        ex, fut = self._future_exog(exog, self.n_obs_ + steps)
        out = empty((self.batch_size_ * steps,), "<f4")
        y = self._y
        pr = self.params_
        self._extension().arima_forecast(
            addr_ro(y, name="y"),
            addr_ro(ex, name="exog"),
            addr_ro(fut, name="exog (future values)"),
            addr_ro(pr, name="params"),
            addr(out, name="out"),
            # ORDER MATCHES
            # bindings/_mojolearn_arima.mojo::arima_forecast_binding.
            # batch_size, n_obs, n_steps, p, d, q, P, D, Q, s, k, n_exog,
            # reserved (MUST BE 0; the slot cuML's `level` would take, and
            # `level` is NOT IMPLEMENTED)
            [self.batch_size_, self.n_obs_, steps, p, d, q, P, D, Q, s,
             self.k_, self.n_exog_, 0],
        )
        return out.reshape((self.batch_size_, steps))

    # -- saved models (lane/inference-forecast-umap-pca, 2026-09-15) --------

    def save(self, path):
        """Write the fitted model to `path` as an npz: the series `y` the
        filter runs over (`(batch_size, n_obs)` float32), `params` in the
        packed fitted order, `x`, `x0`, `fx` (float32), `n_iter`, `retcode`
        (int32), `llf`, `aic`, `bic` (float64), `meta` `<i8` [p, d, q, P, D,
        Q, s, k, batch_size, n_obs, maxiter, input_copied], `trend`, `method`
        and `numeric_mode`, under format `mojolearn-arima-1`. A model fit with
        `exog` (DEVIATION 998) is `mojolearn-arima-2`: the same members, the
        fit's regressors as `exog` `(batch_size, n_obs, n_exog)` float32, and
        `n_exog` appended to `meta`.

        A loaded model predicts and forecasts and answers every fitted
        attribute (`params_`, `ar_`, `ma_`, `sar_`, `sma_`, `mu_`,
        `sigma2_`, `llf_`, `aic_`, `bic_`, `n_iter_`, `retcode_`); it does
        not refit. On a CPU-only install that is public inference through the
        host binding (`mojolearn.host_model(path)`, or `ARIMA.load(path)`),
        with no GPU and no CPU training."""
        self._check_fitted("save")
        from . import _serialize
        from .decomposition import _saved_mode
        p, d, q = self.order
        P, D, Q, s = self.seasonal_order
        b, n = int(self.batch_size_), int(self.n_obs_)
        n_exog = int(getattr(self, "n_exog_", 0))
        meta = [p, d, q, P, D, Q, s, int(self.k_), b, n, int(self.maxiter),
                1 if getattr(self, "input_copied_", False) else 0]
        if n_exog:
            meta.append(n_exog)
        arrays = {
            "format": _ARIMA_FORMAT_EXOG if n_exog else _ARIMA_FORMAT,
            "estimator": type(self).__name__,
            "numeric_mode": _saved_mode(self),
            "trend": "none" if self.trend is None else str(self.trend).lower(),
            "method": str(self.method),
            "y": self._y.reshape((b, n)),
            "params": self.params_,
            "x": self.x_,
            "x0": self.x0_,
            "fx": self.fx_,
            "n_iter": self.n_iter_,
            "retcode": self.retcode_,
            "llf": self.llf_,
            "aic": self.aic_,
            "bic": self.bic_,
            "meta": Array.from_list(meta, "<i8"),
        }
        if n_exog:
            arrays["exog"] = self._exog.reshape((b, n, n_exog))
        return _serialize.write_npz(path, arrays)

    @classmethod
    def load(cls, path):
        """Load a model saved by `save`. The result predicts, forecasts and
        answers the fitted attributes; it does not refit."""
        from . import _serialize
        from .decomposition import _check_saved_by, _restore_mode
        arrays = _serialize.read_npz(path, (_ARIMA_FORMAT, _ARIMA_FORMAT_EXOG))
        _check_saved_by(arrays, path, cls)
        with_exog = _serialize.scalar_str(arrays, "format") == _ARIMA_FORMAT_EXOG
        meta = _serialize.exact(arrays, "meta", "<i8")
        want = 13 if with_exog else 12
        if meta.size != want:
            raise ValueError(f"mojolearn: {path!r} meta holds {meta.size} fields, {want} are needed")
        p, d, q, P, D, Q, s, k, b, n, maxiter, copied = (int(meta[i]) for i in range(12))
        n_exog = int(meta[12]) if with_exog else 0
        if with_exog and n_exog < 1:
            raise ValueError(f"mojolearn: {path!r} is {_ARIMA_FORMAT_EXOG!r} with n_exog={n_exog}")
        if not with_exog and "exog" in arrays:
            raise ValueError(f"mojolearn: {path!r} is {_ARIMA_FORMAT!r} and holds an exog member")
        trend = _serialize.scalar_str(arrays, "trend")
        obj = cls(order=(p, d, q), seasonal_order=(P, D, Q, s),
                  trend=None if trend == "none" else trend,
                  method=_serialize.scalar_str(arrays, "method"), maxiter=maxiter)
        if obj.k_ != k:
            raise ValueError(f"mojolearn: {path!r} records k={k}, its trend {trend!r} resolves to {obj.k_}")
        _restore_mode(obj, arrays)
        N = obj.complexity_ + n_exog
        shapes = {"y": ("<f4", (b, n)), "params": ("<f4", (b, N)), "x": ("<f4", (b, N)),
                  "x0": ("<f4", (b, N)), "fx": ("<f4", (b,)), "n_iter": ("<i4", (b,)),
                  "retcode": ("<i4", (b,)), "llf": ("<f8", (b,)), "aic": ("<f8", (b,)),
                  "bic": ("<f8", (b,))}
        if with_exog:
            shapes["exog"] = ("<f4", (b, n, n_exog))
        got = {}
        for name, (dtype, shape) in shapes.items():
            value = _serialize.exact(arrays, name, dtype)
            if tuple(value.shape) != shape:
                raise ValueError(f"mojolearn: {path!r} {name} has shape {tuple(value.shape)}, not {shape}")
            got[name] = value
        obj.batch_size_ = b
        obj.n_obs_ = n
        obj.input_copied_ = bool(copied)
        obj.n_exog_ = n_exog
        obj.complexity_ = N
        obj._exog = got["exog"] if with_exog else None
        obj._y = got["y"]
        obj.params_ = got["params"]
        obj.x_ = got["x"]
        obj.x0_ = got["x0"]
        obj.fx_ = got["fx"]
        obj.n_iter_ = got["n_iter"]
        obj.retcode_ = got["retcode"]
        obj.llf_ = got["llf"]
        obj.aic_ = got["aic"]
        obj.bic_ = got["bic"]
        return obj

    def __repr__(self):
        return (
            "ARIMA(order={}, seasonal_order={}, trend={!r}, method={!r})"
            .format(self.order, self.seasonal_order, self.trend, self.method)
        )


def _plain_order(value, normalized):
    """True when `value` already IS `normalized`: a tuple of Python ints."""
    return (type(value) is tuple and all(type(v) is int for v in value)
            and value == normalized)


def _as_order(value, width, name):
    """`(p, d, q)` or `(P, D, Q, s)` as a tuple of non-negative ints.

    A CONVERSION, NOT A POLICY. Everything about whether the numbers are a
    LEGAL order is `validate_order`'s, in Mojo, with the reference's messages. What is
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
