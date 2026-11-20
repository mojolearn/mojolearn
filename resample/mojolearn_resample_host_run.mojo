# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN of `bindings/_mojolearn_resample_host.mojo` (wave 8 run split): the
host bootstrap, permutation-test and Monte Carlo bodies as typed Mojo.
Imports no Python. The binding keeps UNPACK (the two length-checked lists
into words, pointers and the read host lists), PACK (the returned status) and
the GIL: it releases the GIL around each call here exactly where it released
it before the split."""
from core.hostptr import f32_ptr, f64_ptr, copy_f32
from resample.checks.statistics import (
    MC_F_CONST,
    MC_F_PRODUCT,
    MC_F_SUM,
    mc_closed_form,
)
from resample.host.resample_host import (
    host_bootstrap,
    host_bootstrap_unpaired,
    host_monte_carlo_integrate,
    host_permutation_samples,
    host_permutation_test,
)


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    return f32_ptr(addr)


def _f64_ptr(addr: Int) raises -> MutPointer[Float64, MutUntrackedOrigin]:
    return f64_ptr(addr)


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
    var r = host_bootstrap(
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


def _bootstrap_unpaired_run(
    x: List[Float32],
    n_x: Int,
    y: List[Float32],
    n_y: Int,
    n_resamples: Int,
    seed: UInt64,
    method: Int,
    confidence_level: Float32,
    alternative: Int,
    r_first: Int,
    dp: MutPointer[Float32, MutUntrackedOrigin],
    sdp: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
) raises:
    var r = host_bootstrap_unpaired(
        x, n_x, y, n_y, n_resamples, seed, method, confidence_level,
        alternative, r_first,
    )
    copy_f32(r.distribution.unsafe_ptr(), dp, n_resamples)
    copy_f32(r.sorted_distribution.unsafe_ptr(), sdp, n_resamples)
    sp.unsafe_store(0, Float64(r.point_estimate))
    sp.unsafe_store(1, Float64(r.standard_error))
    sp.unsafe_store(2, Float64(r.interval.low))
    sp.unsafe_store(3, Float64(r.interval.high))
    sp.unsafe_store(4, Float64(r.order_low))
    sp.unsafe_store(5, Float64(r.order_high))


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
    var r = host_permutation_test(
        x, y, statistic, n_resamples, seed, alternative, r_first
    )
    copy_f32(r.null_distribution.unsafe_ptr(), np_, n_resamples)
    sp.unsafe_store(0, Float64(r.observed))
    sp.unsafe_store(1, Float64(r.pvalue.p))
    sp.unsafe_store(2, Float64(r.pvalue.count_less))
    sp.unsafe_store(3, Float64(r.pvalue.count_greater))


def _permutation_samples_run(
    x: List[Float32],
    y: List[Float32],
    n_y: Int,
    n_resamples: Int,
    seed: UInt64,
    alternative: Int,
    r_first: Int,
    np_: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
) raises:
    var r = host_permutation_samples(x, y, n_y > 0, n_resamples, seed, alternative, r_first)
    copy_f32(r.null_distribution.unsafe_ptr(), np_, n_resamples)
    sp.unsafe_store(0, Float64(r.observed))
    sp.unsafe_store(1, Float64(r.pvalue.p))
    sp.unsafe_store(2, Float64(r.pvalue.count_less))
    sp.unsafe_store(3, Float64(r.pvalue.count_greater))


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
        var r = host_monte_carlo_integrate[MC_F_CONST](
            lower, upper, n_samples, seed, i_first
        )
        integral = r.integral
        mean = r.mean
        volume = r.volume
        closed = mc_closed_form[MC_F_CONST](lower, upper)
    elif f_id == MC_F_SUM:
        var r = host_monte_carlo_integrate[MC_F_SUM](
            lower, upper, n_samples, seed, i_first
        )
        integral = r.integral
        mean = r.mean
        volume = r.volume
        closed = mc_closed_form[MC_F_SUM](lower, upper)
    elif f_id == MC_F_PRODUCT:
        var r = host_monte_carlo_integrate[MC_F_PRODUCT](
            lower, upper, n_samples, seed, i_first
        )
        integral = r.integral
        mean = r.mean
        volume = r.volume
        closed = mc_closed_form[MC_F_PRODUCT](lower, upper)
    else:
        raise Error(
            "monte_carlo_integrate: integrand id "
            + String(f_id)
            + " is not one of the three compiled arms (0 const, 1 sum,"
            " 2 product); the integrand is a comptime parameter of"
            " host_monte_carlo_integrate and a new one is one arm plus one"
            " closed form (resample/checks/statistics.mojo)"
        )
    sp.unsafe_store(0, Float64(integral))
    sp.unsafe_store(1, Float64(mean))
    sp.unsafe_store(2, Float64(volume))
    sp.unsafe_store(3, Float64(closed))
