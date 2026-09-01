# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPython boundary for the batched ARIMA lane.

AN ELEVENTH EXTENSION MODULE, and a separate one on purpose. The header of
`bindings/_mojolearn_estimators.mojo` states the reason and it is the same
reason here, an independently changing binding must not become a merge
point. `arima/` is cuML's `cpp/src/arima/`, the batched Kalman filter, the
Jones transform, `estimate_x0` and an own-written batched L-BFGS, and it
shares no code with the Holt-Winters and KPSS lanes that
`bindings/_mojolearn_tsa.mojo` carries. All eleven land in one wheel.

WHAT THIS CLOSES. `bindings/_mojolearn_tsa.mojo` says in its own header that
`arima/` is deliberately absent because the lane has no `fit` and an `ARIMA`
class would have to ask the caller for the answer before computing it. THAT
STOPPED BEING TRUE when `arima/impl/arima/batched_fit.mojo` landed
(2026-09-01) with `estimate_x0`, the Householder QR least squares and the
batched L-BFGS, all gated by `arima/checks/fit_check.mojo` in both numeric
tiers. What was missing after that was only the Python door, and this file
is the middle third of it. THAT PARAGRAPH IN `_mojolearn_tsa.mojo` AND THE
MATCHING ONE IN `python/mojolearn/_tsa_impl.py` ARE NOW FALSE AND ARE OWED A
DELETION; they are not edited here only because those files belong to
another lane's owner.

Arrays cross as borrowed NumPy addresses; all device buffers and contexts
live for one call and no pointer is retained. The Python wrapper owns the
arrays and keeps them alive for the duration of the call
(`python/mojolearn/_arrays.py` is where that contract is written down).

SCALARS ARRIVE AS ONE LIST, NOT AS SEPARATE ARGUMENTS.
`PythonModuleBuilder.def_function` infers its signature from arity and stops
working above roughly nine arguments, so buffer addresses go positionally
and every scalar goes in one `params` list. THE ORDER OF THAT LIST IS
WRITTEN OUT IN A COMMENT ON BOTH SIDES IN THE SAME WORDS. A silent
reordering is a wrong answer, not a failure, and it is the most dangerous
thing at this boundary because nine of the thirteen entries are small
non-negative integers that would all look plausible in each other's slots.
The length check on every entry below is what stands between a swapped pair
and a believable number, and the three lists are deliberately the same
length and the same tail so the two comments diff by eye.

WHAT IS REFUSED, AND WHERE. Nothing is refused in this file except a null
address and a params list of the wrong length. Every model refusal lives one
or two layers down and is raised there by name, which is what keeps it
reachable from every caller and not only from Python:

    exog                 `arima/impl/tsa/arima_common.mojo::validate_order`
                         (`n_exog != 0`). `ARIMAParams` has no `beta`
                         anywhere in the lane
    method css, css-ml   `arima/estimator.mojo::_refuse_method`
    rd > 8               `validate_order`, cuML's block-per-series Kalman
    r > 5                `validate_order`, cuML's Schur Lyapunov solver
    d + D > 2, s < 2,
    p q P Q > 8,
    an empty order       `validate_order`, in cuML's own words
    a non-finite series  `batched_arima.mojo::_refuse_non_finite`, with the
                         flat index of the offender
    float64              by dtype, DEVIATION 670. Metal exposes no Float64
                         on the device and every kernel here is float32

DEVIATIONS 990-999 are this surface's. 990, 991, 992 and 993 are used and
each is named where it bites; 994-999 are unassigned.

`ARIMA` IS wired into `python/mojolearn/__init__.py` and
`_mojolearn_arima` IS registered in `python/mojolearn/_backend.py`'s
`_MODULES` and its `_build_script` dict. Both halves, because forgetting
either is silent: DEVIATION 869 records that an extension missing from
`_MODULES` is never re-pointed, so under `identical` a plain import resolves
to the FAST binary sitting beside it and returns fast arithmetic under the
identical label.

THE GIL is released around every device call, and nothing inside a
`GILReleased` block touches a `PythonObject`.
"""

from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR

from arima.estimator import (
    arima_fit_ptr_host,
    arima_forecast_ptr_host,
    arima_predict_ptr_host,
)


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null float32 buffer address")
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null int32 buffer address")
    return MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr)


def arima_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER, as the `NUMERIC_*` code itself, 0 FAST, 1
    IDENTICAL, 2 DETERMINISTIC. The same shape as `gbdt_numeric_mode` and
    for the same reason, the wrapper reads it once and refuses to run if the
    binary it loaded disagrees with the mode the package asked for. A
    wrong-arm measurement that is correctly labelled by accident is the
    failure this prevents, and a boolean could not do that job once a third
    tier existed, because DETERMINISTIC answered 0 and read back as
    "fast"."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def arima_vendor_binding() raises -> PythonObject:
    """THE ACCELERATOR API THIS BINARY WAS COMPILED FOR, 'metal', 'cuda',
    'hip' or 'none'. A compile-time constant folded in from
    `checks/vendor.mojo`, the same shape as the tier read-back. The answer
    comes from the binary that actually loaded, never from the directory it
    sat in and never from the environment.
    `python/mojolearn/_backend.py` refuses at import when this disagrees
    with the vendor directory the set was loaded from."""
    return PythonObject(String(COMPILED_VENDOR))


def arima_fit_binding(
    y_addr: PythonObject,
    params_addr: PythonObject,
    x_addr: PythonObject,
    x0_addr: PythonObject,
    stats_addr: PythonObject,
    flags_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`ARIMA(order, seasonal_order, trend).fit(y)` (arima/, DEVIATIONS 670
    to 687 and 990 to 993). Returns `N * batch_size`, the number of float32
    written to `params_addr`, with `N = p + q + P + Q + k + 1`.

    `params` is, in this exact order (mirrored in
    `python/mojolearn/_arima_impl.py`):

         0  batch_size
         1  n_obs
         2  p
         3  d
         4  q
         5  P
         6  D
         7  Q
         8  s
         9  k                fit_intercept, 0 or 1
        10  n_exog           REFUSED unless 0, by validate_order
        11  method           0 = MLE, 1 = CSS, 2 = CSS-ML; only MLE is
                             offered and the other two are REFUSED BY NAME
                             in arima/estimator.mojo
        12  max_iterations

    `y_addr` reads `batch_size * n_obs` float32 with each SERIES CONTIGUOUS,
    series `b` at `[b * n_obs, (b + 1) * n_obs)`.

    `params_addr` is written with `N * batch_size` float32, the FITTED and
    forward-transformed model, packed per series in this exact order (the
    same words in `arima/estimator.mojo` and
    `python/mojolearn/_arima_impl.py`):

        series b occupies [b * N, (b + 1) * N)
            mu       k values     (absent when k == 0)
            ar       p values
            ma       q values
            sar      P values
            sma      Q values
            sigma2   1 value

    `x_addr` and `x0_addr` are each written with `N * batch_size` float32 in
    that same packing, the unconstrained optimum and the starting point
    `estimate_x0` produced. Neither is on cuML's Python surface; both are
    here because `estimate_x0` is the half of this lane with no upstream
    oracle.

    `stats_addr` is written with `2 * batch_size` float32, in this exact
    order:

        [0 * batch_size, 1 * batch_size)   loglike
        [1 * batch_size, 2 * batch_size)   fx

    `flags_addr` is written with `2 * batch_size` int32, in this exact
    order:

        [0 * batch_size, 1 * batch_size)   n_iter
        [1 * batch_size, 2 * batch_size)   retcode, 0 is OPT_SUCCESS
    """
    if len(params) != 13:
        raise Error(
            "arima_fit: params must contain 13 values (batch_size, n_obs, p,"
            " d, q, P, D, Q, s, k, n_exog, method, max_iterations), got "
            + String(len(params))
        )
    var yp = _f32_ptr(Int(py=y_addr))
    var pp = _f32_ptr(Int(py=params_addr))
    var xp = _f32_ptr(Int(py=x_addr))
    var x0p = _f32_ptr(Int(py=x0_addr))
    var sp = _f32_ptr(Int(py=stats_addr))
    var fp = _i32_ptr(Int(py=flags_addr))
    var batch_size = Int(py=params[0])
    var n_obs = Int(py=params[1])
    var p = Int(py=params[2])
    var d = Int(py=params[3])
    var q = Int(py=params[4])
    var P = Int(py=params[5])
    var D = Int(py=params[6])
    var Q = Int(py=params[7])
    var s = Int(py=params[8])
    var k = Int(py=params[9])
    var n_exog = Int(py=params[10])
    var method = Int(py=params[11])
    var max_iterations = Int(py=params[12])
    var written = 0
    with GILReleased(Python()):
        written = arima_fit_ptr_host(
            yp, pp, xp, x0p, sp, fp, batch_size, n_obs,
            p, d, q, P, D, Q, s, k, n_exog, method, max_iterations,
        )
    return PythonObject(written)


def arima_predict_binding(
    y_addr: PythonObject,
    params_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`ARIMA.predict(start, end)` with no confidence level and no exog
    (arima/). Returns `(end - start) * batch_size`.

    `params` is, in this exact order (mirrored in
    `python/mojolearn/_arima_impl.py`):

         0  batch_size
         1  n_obs
         2  start
         3  end              EXCLUDED, cuML's convention, not statsmodels'
         4  p
         5  d
         6  q
         7  P
         8  D
         9  Q
        10  s
        11  k                fit_intercept, 0 or 1
        12  n_exog           REFUSED unless 0, by validate_order

    `y_addr` reads `batch_size * n_obs` float32, series contiguous, the SAME
    series the fit saw. `params_addr` reads `N * batch_size` float32 in
    `arima_fit`'s packed order, the FITTED parameters (DEVIATION 990, the
    model crosses back to the host and is uploaded again at every predict).

    `out_addr` is written with `(end - start) * batch_size` float32, SERIES
    MAJOR: series `b` at step `i` is `[b * (end - start) + i]`. That is a
    TRANSPOSE of cuML's `(end - start, batch_size)` output and it matches
    both this lane's kernel and the input layout, so the wrapper transposes
    nothing. Steps before `d + s * D` are the canonical NaN (DEVIATION 676),
    which is `arima.pyx:672-674`'s warning made into a value.
    """
    if len(params) != 13:
        raise Error(
            "arima_predict: params must contain 13 values (batch_size,"
            " n_obs, start, end, p, d, q, P, D, Q, s, k, n_exog), got "
            + String(len(params))
        )
    var yp = _f32_ptr(Int(py=y_addr))
    var pp = _f32_ptr(Int(py=params_addr))
    var op = _f32_ptr(Int(py=out_addr))
    var batch_size = Int(py=params[0])
    var n_obs = Int(py=params[1])
    var start = Int(py=params[2])
    var end = Int(py=params[3])
    var p = Int(py=params[4])
    var d = Int(py=params[5])
    var q = Int(py=params[6])
    var P = Int(py=params[7])
    var D = Int(py=params[8])
    var Q = Int(py=params[9])
    var s = Int(py=params[10])
    var k = Int(py=params[11])
    var n_exog = Int(py=params[12])
    var written = 0
    with GILReleased(Python()):
        written = arima_predict_ptr_host(
            yp, pp, op, batch_size, n_obs, start, end,
            p, d, q, P, D, Q, s, k, n_exog,
        )
    return PythonObject(written)


def arima_forecast_binding(
    y_addr: PythonObject,
    params_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`ARIMA.forecast(steps)` (arima/), which upstream is
    `predict(n_obs, n_obs + steps)` and is that here too. Returns
    `steps * batch_size`.

    `params` is, in this exact order (mirrored in
    `python/mojolearn/_arima_impl.py`). IT IS `arima_predict`'s LIST WITH
    `start` AND `end` REPLACED BY ONE `n_steps`, and the nine-entry order
    tail is the same in both:

         0  batch_size
         1  n_obs
         2  n_steps
         3  p
         4  d
         5  q
         6  P
         7  D
         8  Q
         9  s
        10  k                fit_intercept, 0 or 1
        11  n_exog           REFUSED unless 0, by validate_order
        12  reserved         MUST BE 0. It exists so that this list is the
                             same length as the other two and so that a
                             future step-scoped parameter (cuML's `level`
                             is the obvious candidate, and it is REFUSED BY
                             NAME today) does not renumber the tail

    `out_addr` is written with `steps * batch_size` float32, SERIES MAJOR,
    series `b` at step `i` at `[b * steps + i]`. There is no in-sample block
    in it and no NaN prefix: `start == n_obs`, so the in-sample kernel does
    not launch and every value comes from the Kalman forecast, undifferenced
    by `finalize_forecast` when `d + D > 0`.
    """
    if len(params) != 13:
        raise Error(
            "arima_forecast: params must contain 13 values (batch_size,"
            " n_obs, n_steps, p, d, q, P, D, Q, s, k, n_exog, reserved),"
            " got " + String(len(params))
        )
    var yp = _f32_ptr(Int(py=y_addr))
    var pp = _f32_ptr(Int(py=params_addr))
    var op = _f32_ptr(Int(py=out_addr))
    var batch_size = Int(py=params[0])
    var n_obs = Int(py=params[1])
    var n_steps = Int(py=params[2])
    var p = Int(py=params[3])
    var d = Int(py=params[4])
    var q = Int(py=params[5])
    var P = Int(py=params[6])
    var D = Int(py=params[7])
    var Q = Int(py=params[8])
    var s = Int(py=params[9])
    var k = Int(py=params[10])
    var n_exog = Int(py=params[11])
    var reserved = Int(py=params[12])
    if reserved != 0:
        raise Error(
            "arima_forecast: params[12] is reserved and must be 0, got "
            + String(reserved)
            + ". cuML's `level` (confidence intervals) is the parameter this"
            " slot is held for and it is NOT PORTED"
            " (arima/NOT_IMPLEMENTED.tsv): the confidence_intervals kernel"
            " at batched_kalman.cu:824-838 and the P = T P T' + RR'"
            " propagation beside it have no port"
        )
    var written = 0
    with GILReleased(Python()):
        written = arima_forecast_ptr_host(
            yp, pp, op, batch_size, n_obs, n_steps,
            p, d, q, P, D, Q, s, k, n_exog,
        )
    return PythonObject(written)


@export
def PyInit__mojolearn_arima() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_arima")
        m.def_function[arima_vendor_binding]("arima_vendor")
        m.def_function[arima_numeric_mode_binding]("arima_numeric_mode")
        m.def_function[arima_fit_binding]("arima_fit")
        m.def_function[arima_predict_binding]("arima_predict")
        m.def_function[arima_forecast_binding]("arima_forecast")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_arima: ", e))
