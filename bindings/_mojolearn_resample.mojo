# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPython boundary for the resampling lane: `bootstrap`,
`permutation_test`, `monte_carlo_integrate` (workstream D, 2026-09-14).

A separate extension module, for `bindings/_mojolearn_gp.mojo`'s reason.
`resample/estimator.mojo`'s three host surfaces are reached and nothing is
re-decided: every refusal (`validate_positions`, `validate_pooled`, the
statistic and method codes, the BCa refusal DEVIATION 1699, the non-finite
cell, the sort-cell ceiling) is raised one layer down by name. This file
refuses a null address, a list of the wrong length and an integrand id
outside the three compiled arms, because `monte_carlo_integrate_host` takes
its integrand as a COMPTIME parameter and the dispatch below is the only
place a runtime id can become one.

THE ABI IS THE GP'S: two length-checked lists, orders written out below
and mirrored in `python/mojolearn/resample.py`. Every result struct crosses
as its arrays and scalars; the scalars widen float32 to float64 exactly.
"""

from std.os import abort
from bindings.hostptr import f32_ptr, f64_ptr, copy_f32, read_f32
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from resample.checks.statistics import MC_DIMS, MC_F_CONST, MC_F_PRODUCT, MC_F_SUM
from resample.estimator import (
    BootstrapResult,
    MonteCarloResult,
    PermutationResult,
    bootstrap_host,
    mc_closed_form_for,
    monte_carlo_integrate_host,
    permutation_test_host,
)


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    return f32_ptr(addr)


def _f64_ptr(addr: Int) raises -> MutPointer[Float64, MutUntrackedOrigin]:
    return f64_ptr(addr)


def resample_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER as the `NUMERIC_*` code: 0 FAST, 1 IDENTICAL, 2
    DETERMINISTIC."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def resample_vendor_binding() raises -> PythonObject:
    """'metal', 'cuda', 'hip' or 'none', from `checks/vendor.mojo`."""
    return PythonObject(String(COMPILED_VENDOR))


# ===========================================================================
# bootstrap
# ===========================================================================


def resample_ranges_parallel_available() raises -> PythonObject:
    """1: bootstrap_host, permutation_test_host and monte_carlo_integrate_host
    read MOJOLEARN_RESAMPLE_DEVICE_COUNT and move whole global replicate,
    permutation and PINNED_SUM_W sample-chunk ranges to owners."""
    return PythonObject(1)


def _bootstrap_run(
    x: List[Float32],
    n: Int,
    d: Int,
    statistic: Int,
    n_resamples: Int,
    seed: UInt64,
    method: Int,
    confidence_level: Float32,
    alternative: Int,
    q_or_prop: Float32,
    r_first: Int,
    dp: MutPointer[Float32, MutUntrackedOrigin],
    sdp: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
) raises:
    var r = bootstrap_host(
        x,
        n,
        d,
        statistic,
        n_resamples,
        seed,
        method,
        confidence_level,
        alternative,
        q_or_prop,
        r_first,
    )
    copy_f32(r.distribution.unsafe_ptr(), dp, n_resamples)
    copy_f32(r.sorted_distribution.unsafe_ptr(), sdp, n_resamples)
    sp.unsafe_store(0, Float64(r.point_estimate))
    sp.unsafe_store(1, Float64(r.standard_error))
    sp.unsafe_store(2, Float64(r.interval.low))
    sp.unsafe_store(3, Float64(r.interval.high))
    sp.unsafe_store(4, Float64(r.order_low))
    sp.unsafe_store(5, Float64(r.order_high))


def bootstrap_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`scipy.stats.bootstrap((x,), statistic, ...)` (`bootstrap_host`).
    Returns 0.

    `addrs`, in this exact order:

        0  x                 n * d float32, row-major, read (the resample
                              draws a ROW, so a two-column sample keeps
                              its pairing)
        1  distribution_out  n_resamples float32, WRITTEN
        2  sorted_out        n_resamples float32, WRITTEN
        3  scalars_out       6 float64, WRITTEN: point_estimate,
                              standard_error, low, high, order_low,
                              order_high

    `params`, in this exact order:

        0  n
        1  d                 n_features (1 or 2)
        2  statistic         STAT_* code
        3  n_resamples
        4  seed
        5  method            METHOD_* code (percentile 0, basic 1, bca 2
                              which is refused by name, DEVIATION 1699)
        6  confidence_level  (float)
        7  alternative       ALT_* code (two-sided 0, less 1, greater 2)
        8  q_or_prop         (float; q for quantile, proportiontocut for
                              trimmed_mean, unread otherwise)
        9  r_first           the batch-invariance handle
    """
    if len(addrs) != 4:
        raise Error(
            "bootstrap: addrs must contain 4 addresses (x, distribution_out,"
            " sorted_out, scalars_out), got "
            + String(len(addrs))
        )
    if len(params) != 10:
        raise Error(
            "bootstrap: params must contain 10 values (n, d, statistic,"
            " n_resamples, seed, method, confidence_level, alternative,"
            " q_or_prop, r_first), got "
            + String(len(params))
        )
    var xp = _f32_ptr(Int(py=addrs[0]))
    var dp = _f32_ptr(Int(py=addrs[1]))
    var sdp = _f32_ptr(Int(py=addrs[2]))
    var sp = _f64_ptr(Int(py=addrs[3]))
    var n = Int(py=params[0])
    var d = Int(py=params[1])
    var statistic = Int(py=params[2])
    var n_resamples = Int(py=params[3])
    var seed = UInt64(Int(py=params[4]))
    var method = Int(py=params[5])
    var confidence_level = Float32(Float64(py=params[6]))
    var alternative = Int(py=params[7])
    var q_or_prop = Float32(Float64(py=params[8]))
    var r_first = Int(py=params[9])
    var x = read_f32(Int(xp), max(0, n * d))
    with GILReleased(Python()):
        _bootstrap_run(
            x,
            n,
            d,
            statistic,
            n_resamples,
            seed,
            method,
            confidence_level,
            alternative,
            q_or_prop,
            r_first,
            dp,
            sdp,
            sp,
        )
    return PythonObject(0)


# ===========================================================================
# permutation_test
# ===========================================================================


def _permutation_run(
    x: List[Float32],
    y: List[Float32],
    statistic: Int,
    n_resamples: Int,
    seed: UInt64,
    alternative: Int,
    r_first: Int,
    np_: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
) raises:
    var r = permutation_test_host(
        x, y, statistic, n_resamples, seed, alternative, r_first
    )
    copy_f32(r.null_distribution.unsafe_ptr(), np_, n_resamples)
    sp.unsafe_store(0, Float64(r.observed))
    sp.unsafe_store(1, Float64(r.pvalue.p))
    sp.unsafe_store(2, Float64(r.pvalue.count_less))
    sp.unsafe_store(3, Float64(r.pvalue.count_greater))


def permutation_test_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`scipy.stats.permutation_test((x, y), statistic,
    permutation_type='independent', ...)` (`permutation_test_host`).
    Returns 0. The null is NEVER exhaustive (DEVIATION 1702).

    `addrs`, in this exact order:

        0  x           n_x float32, read
        1  y           n_y float32, read
        2  null_out    n_resamples float32, WRITTEN
        3  scalars_out 4 float64, WRITTEN: observed, p, count_less,
                        count_greater

    `params`, in this exact order:

        0  n_x
        1  n_y
        2  statistic     STAT_* code
        3  n_resamples
        4  seed
        5  alternative   ALT_* code
        6  r_first
    """
    if len(addrs) != 4:
        raise Error(
            "permutation_test: addrs must contain 4 addresses (x, y,"
            " null_out, scalars_out), got "
            + String(len(addrs))
        )
    if len(params) != 7:
        raise Error(
            "permutation_test: params must contain 7 values (n_x, n_y,"
            " statistic, n_resamples, seed, alternative, r_first), got "
            + String(len(params))
        )
    var n_x = Int(py=params[0])
    var n_y = Int(py=params[1])
    var statistic = Int(py=params[2])
    var n_resamples = Int(py=params[3])
    var seed = UInt64(Int(py=params[4]))
    var alternative = Int(py=params[5])
    var r_first = Int(py=params[6])
    var x = read_f32(Int(py=addrs[0]), max(0, n_x))
    var y = read_f32(Int(py=addrs[1]), max(0, n_y))
    var np_ = _f32_ptr(Int(py=addrs[2]))
    var sp = _f64_ptr(Int(py=addrs[3]))
    with GILReleased(Python()):
        _permutation_run(
            x, y, statistic, n_resamples, seed, alternative, r_first, np_, sp
        )
    return PythonObject(0)


# ===========================================================================
# monte_carlo_integrate
# ===========================================================================


def _mc_run(
    f_id: Int,
    lower: List[Float32],
    upper: List[Float32],
    n_samples: Int,
    seed: UInt64,
    i_first: Int,
    sp: MutPointer[Float64, MutUntrackedOrigin],
) raises:
    """The runtime id becomes the comptime parameter HERE and nowhere
    else; an id outside the three arms is refused by name."""
    var integral = Float32(0.0)
    var mean = Float32(0.0)
    var volume = Float32(0.0)
    var closed = Float32(0.0)
    if f_id == MC_F_CONST:
        var r = monte_carlo_integrate_host[MC_F_CONST](
            lower, upper, n_samples, seed, i_first
        )
        integral = r.integral
        mean = r.mean
        volume = r.volume
        closed = mc_closed_form_for[MC_F_CONST](lower, upper)
    elif f_id == MC_F_SUM:
        var r = monte_carlo_integrate_host[MC_F_SUM](
            lower, upper, n_samples, seed, i_first
        )
        integral = r.integral
        mean = r.mean
        volume = r.volume
        closed = mc_closed_form_for[MC_F_SUM](lower, upper)
    elif f_id == MC_F_PRODUCT:
        var r = monte_carlo_integrate_host[MC_F_PRODUCT](
            lower, upper, n_samples, seed, i_first
        )
        integral = r.integral
        mean = r.mean
        volume = r.volume
        closed = mc_closed_form_for[MC_F_PRODUCT](lower, upper)
    else:
        raise Error(
            "monte_carlo_integrate: integrand id "
            + String(f_id)
            + " is not one of the three compiled arms (0 const, 1 sum,"
            " 2 product); the integrand is a comptime parameter of"
            " monte_carlo_integrate_host and a new one is one arm plus one"
            " closed form (resample/checks/statistics.mojo)"
        )
    sp.unsafe_store(0, Float64(integral))
    sp.unsafe_store(1, Float64(mean))
    sp.unsafe_store(2, Float64(volume))
    sp.unsafe_store(3, Float64(closed))


def monte_carlo_integrate_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`volume * mean(f(x_i))` over uniform draws from `[lower, upper)`
    (`monte_carlo_integrate_host`). Returns 0.

    `addrs`, in this exact order:

        0  lower        MC_DIMS (2) float32, read
        1  upper        MC_DIMS (2) float32, read
        2  scalars_out  4 float64, WRITTEN: integral, mean, volume, and the
                         hand-derived closed form of the same integrand
                         (`mc_closed_form_for`), so a caller can see the
                         error without a second entry point

    `params`, in this exact order:

        0  f_id         0 const, 1 sum (x0 + x1), 2 product (x0 * x1)
        1  n_samples
        2  seed
        3  i_first      the batch-invariance handle
    """
    if len(addrs) != 3:
        raise Error(
            "monte_carlo_integrate: addrs must contain 3 addresses (lower,"
            " upper, scalars_out), got "
            + String(len(addrs))
        )
    if len(params) != 4:
        raise Error(
            "monte_carlo_integrate: params must contain 4 values (f_id,"
            " n_samples, seed, i_first), got "
            + String(len(params))
        )
    var f_id = Int(py=params[0])
    var n_samples = Int(py=params[1])
    var seed = UInt64(Int(py=params[2]))
    var i_first = Int(py=params[3])
    var lower = read_f32(Int(py=addrs[0]), MC_DIMS)
    var upper = read_f32(Int(py=addrs[1]), MC_DIMS)
    var sp = _f64_ptr(Int(py=addrs[2]))
    with GILReleased(Python()):
        _mc_run(f_id, lower, upper, n_samples, seed, i_first, sp)
    return PythonObject(0)


@export
def PyInit__mojolearn_resample() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_resample")
        m.def_function[resample_ranges_parallel_available]("resample_ranges_parallel_available")
        m.def_function[resample_vendor_binding]("resample_vendor")
        m.def_function[resample_numeric_mode_binding]("resample_numeric_mode")
        m.def_function[bootstrap_binding]("bootstrap")
        m.def_function[permutation_test_binding]("permutation_test")
        m.def_function[monte_carlo_integrate_binding]("monte_carlo_integrate")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_resample: ", e))
