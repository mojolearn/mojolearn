# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPython boundary for the time-series lanes: Holt-Winters and the KPSS test.

Kept in a separate extension, deliberately, so an independently changing
binding does not become a merge point (the same reason
`_mojolearn_estimators.mojo` is separate from `_mojolearn.mojo`). Arrays
cross as borrowed NumPy addresses; all device buffers and contexts live for
one call and no pointer is retained.

WHAT IS BEHIND EACH ENTRY, AND WHAT IS NOT

    holtwinters_fit       holtwinters/, DEVIATIONS 660-665 and 697-699.
                          cuML `ML::HoltWinters::fit`, the seasonal BFGS
                          arm, float32.
    holtwinters_forecast  the same lane's `ML::HoltWinters::forecast`.
    kpss_test             tsa/, DEVIATIONS 671-672. cuML
                          `ML::Stationarity::kpss_test`, float32.
    select_d              the same lane's port of auto_arima's "Choose the
                          hyper-parameter d" block.

`arima/` IS DELIBERATELY ABSENT. That lane ports cuML's batched Kalman
filter, and it is finished as far as it goes: given ARIMA coefficients it
computes the log-likelihood, the finite-difference gradient, the in-sample
predictions and the forecast, all bit-identical against a host oracle on
one Apple M4. What it does NOT port is `estimate_x0` / `_start_params` /
`_arma_least_squares` (the initial guess) and `arima.pyx`'s batched L-BFGS
driver, both listed NOT PORTED in `arima/NOT_IMPLEMENTED.tsv`. There is therefore
no `fit`, and the coefficients its entry points require are exactly what
the unported optimizer would have produced. An `ARIMA` class here would
have to ask the caller for the answer before computing it, so there is no
`ARIMA` class here. See the report that landed this file.

THE ARGUMENT-COUNT RULE. `PythonModuleBuilder.def_function` infers its
signature from arity and stops working somewhere above nine arguments, so
buffer addresses go positionally and every scalar goes in ONE list. The
order of that list is written out in a comment on both sides of the
boundary IN THE SAME WORDS, here and in `python/mojolearn/_tsa_impl.py`.
A silently reordered list is a wrong answer rather than a failure, which
makes it the most dangerous thing at this boundary; the two comments are
meant to be diffable by eye.

THE GIL is released around every device call.
"""

from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from original.vendor import COMPILED_VENDOR

from holtwinters.estimator import holtwinters_fit_ptr, holtwinters_forecast_ptr
from tsa.estimator import kpss_test_host, select_d_host


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null float32 buffer address")
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null int32 buffer address")
    return MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr)


def holtwinters_fit_binding(
    data_addr: PythonObject,
    comps_addr: PythonObject,
    stats_addr: PythonObject,
    flags_addr: PythonObject,
    params: PythonObject,
    seasonal: PythonObject,
) raises -> PythonObject:
    """`ExponentialSmoothing(...).fit()` (holtwinters/, DEVIATIONS 660-665,
    697-699). Returns `components_len = (n - frequency) * batch_size`.

    `params` is, in this exact order (mirrored in
    `python/mojolearn/_tsa_impl.py`):

        0  n              observations per series
        1  batch_size     cuML's ts_num
        2  frequency      cuML's seasonal_periods
        3  start_periods
        4  eps            (float)

    `seasonal` is cuML's name string; `seasonal_from_name` refuses
    anything but 'additive'/'add'/'multiplicative'/'mul' BY NAME, in their
    words (`holtwinters.pyx:197`).

    `data_addr` reads `batch_size * n` float32, SERIES-MAJOR: series `s`
    occupies `[s * n, (s + 1) * n)`.

    `comps_addr` is written with `3 * components_len` float32, in this
    exact order (the same words in `holtwinters/estimator.mojo` and
    `python/mojolearn/_tsa_impl.py`):

        [0 * components_len, 1 * components_len)   level
        [1 * components_len, 2 * components_len)   trend
        [2 * components_len, 3 * components_len)   season

    Each block is TIME-MAJOR: series `s` at step `i` is `[s + i *
    batch_size]`. That is a TRANSPOSE of the input's layout, and it is
    cuML's (`holtwinters.pyx:341-344`).

    `stats_addr` is written with `4 * batch_size` float32, in this exact
    order:

        [0 * batch_size, 1 * batch_size)   sse
        [1 * batch_size, 2 * batch_size)   alpha
        [2 * batch_size, 3 * batch_size)   beta
        [3 * batch_size, 4 * batch_size)   gamma

    `flags_addr` is written with `2 * batch_size` int32, in this exact
    order:

        [0 * batch_size, 1 * batch_size)   niter
        [1 * batch_size, 2 * batch_size)   criterion
    """
    if len(params) != 5:
        raise Error(
            "holtwinters_fit: params must contain 5 values (n, batch_size,"
            " frequency, start_periods, eps), got " + String(len(params))
        )
    var dp = _f32_ptr(Int(py=data_addr))
    var cp = _f32_ptr(Int(py=comps_addr))
    var sp = _f32_ptr(Int(py=stats_addr))
    var fp = _i32_ptr(Int(py=flags_addr))
    var n = Int(py=params[0])
    var batch_size = Int(py=params[1])
    var frequency = Int(py=params[2])
    var start_periods = Int(py=params[3])
    var eps = Float32(Float64(py=params[4]))
    var sname = String(py=seasonal)
    var components_len = 0
    with GILReleased(Python()):
        components_len = holtwinters_fit_ptr(
            dp, cp, sp, fp, n, batch_size, frequency, start_periods, sname, eps,
        )
    return PythonObject(components_len)


def holtwinters_forecast_binding(
    comps_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
    seasonal: PythonObject,
) raises -> PythonObject:
    """`.forecast(h)` (holtwinters/). Returns `h * batch_size`.

    `params` is, in this exact order (mirrored in
    `python/mojolearn/_tsa_impl.py`):

        0  n              the FIT's observations per series
        1  batch_size
        2  frequency      the FIT's seasonal_periods
        3  h              steps to forecast

    `comps_addr` reads `3 * components_len` float32 in the SAME packed
    order `holtwinters_fit` wrote: level, then trend, then season, each
    `components_len = (n - frequency) * batch_size` and each time-major.

    `out_addr` is written with `h * batch_size` float32, TIME-MAJOR:
    series `s` at step `i` is `[s + i * batch_size]`.
    """
    if len(params) != 4:
        raise Error(
            "holtwinters_forecast: params must contain 4 values (n,"
            " batch_size, frequency, h), got " + String(len(params))
        )
    var cp = _f32_ptr(Int(py=comps_addr))
    var op = _f32_ptr(Int(py=out_addr))
    var n = Int(py=params[0])
    var batch_size = Int(py=params[1])
    var frequency = Int(py=params[2])
    var h = Int(py=params[3])
    var sname = String(py=seasonal)
    var written = 0
    with GILReleased(Python()):
        written = holtwinters_forecast_ptr(cp, op, n, batch_size, frequency, sname, h)
    return PythonObject(written)


def kpss_test_binding(
    y_addr: PythonObject,
    flags_addr: PythonObject,
    stat_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`cuml.tsa.stationarity.kpss_test(y, d, D, s, pval_threshold)` (tsa/,
    DEVIATIONS 671-672). Returns `batch_size`.

    `params` is, in this exact order (mirrored in
    `python/mojolearn/_tsa_impl.py`):

        0  batch_size
        1  n_obs
        2  d               order of simple differencing
        3  D               order of seasonal differencing
        4  s               seasonal period
        5  pval_threshold  (float)

    `y_addr` reads `batch_size * n_obs` float32 with each SERIES
    CONTIGUOUS: series `b` occupies `[b * n_obs, (b + 1) * n_obs)`. cuML's
    `y` is `(n_obs, batch_size)` in Fortran order, which is the same flat
    bytes; the Python wrapper does that transpose and names its cost.

    `flags_addr` is written with `batch_size` int32 (1 stationary, 0 not),
    which is cuML's return value. `stat_addr` is written with `batch_size`
    float32 KPSS statistics, which is OURS: cuML returns the flags only.
    """
    if len(params) != 6:
        raise Error(
            "kpss_test: params must contain 6 values (batch_size, n_obs, d,"
            " D, s, pval_threshold), got " + String(len(params))
        )
    var yp = _f32_ptr(Int(py=y_addr))
    var fp = _i32_ptr(Int(py=flags_addr))
    var sp = _f32_ptr(Int(py=stat_addr))
    var batch_size = Int(py=params[0])
    var n_obs = Int(py=params[1])
    var d = Int(py=params[2])
    var D = Int(py=params[3])
    var s = Int(py=params[4])
    var pval = Float32(Float64(py=params[5]))
    var count = 0
    with GILReleased(Python()):
        count = kpss_test_host(yp, fp, sp, batch_size, n_obs, d, D, s, pval)
    return PythonObject(count)


def select_d_binding(
    y_addr: PythonObject,
    d_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """auto_arima's "Choose the hyper-parameter d" block (tsa/). Returns
    `batch_size`.

    `params` is, in this exact order (mirrored in
    `python/mojolearn/_tsa_impl.py`):

        0  batch_size
        1  n_obs
        2  D               order of seasonal differencing, GIVEN not chosen
        3  s               seasonal period
        4  d_max           their d_options is range(0, 2 - D + 1)
        5  pval_threshold  (float)

    `y_addr` reads `batch_size * n_obs` float32 with each SERIES
    CONTIGUOUS, the same layout `kpss_test` takes. `d_addr` is written
    with `batch_size` int32.
    """
    if len(params) != 6:
        raise Error(
            "select_d: params must contain 6 values (batch_size, n_obs, D, s,"
            " d_max, pval_threshold), got " + String(len(params))
        )
    var yp = _f32_ptr(Int(py=y_addr))
    var dp = _i32_ptr(Int(py=d_addr))
    var batch_size = Int(py=params[0])
    var n_obs = Int(py=params[1])
    var D = Int(py=params[2])
    var s = Int(py=params[3])
    var d_max = Int(py=params[4])
    var pval = Float32(Float64(py=params[5]))
    var count = 0
    with GILReleased(Python()):
        count = select_d_host(yp, dp, batch_size, n_obs, D, s, d_max, pval)
    return PythonObject(count)


def tsa_vendor_binding() raises -> PythonObject:
    """THE ACCELERATOR API THIS BINARY WAS COMPILED FOR: 'metal', 'cuda',
    'hip' or 'none'. A compile-time constant folded in from
    `original/vendor.mojo`, the same shape as the tier read-back
    (`gbdt_numeric_mode`): the answer comes from the binary that actually
    loaded, never from the directory it sat in or from the environment.
    `python/mojolearn/_backend.py` refuses at import when this disagrees
    with the vendor directory the set was loaded from."""
    return PythonObject(String(COMPILED_VENDOR))


@export
def PyInit__mojolearn_tsa() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_tsa")
        m.def_function[tsa_vendor_binding]("tsa_vendor")
        m.def_function[holtwinters_fit_binding]("holtwinters_fit")
        m.def_function[holtwinters_forecast_binding]("holtwinters_forecast")
        m.def_function[kpss_test_binding]("kpss_test")
        m.def_function[select_d_binding]("select_d")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_tsa: ", e))
