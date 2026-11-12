# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN of `bindings/_mojolearn_resample.mojo` (wave 8 run split): the
bootstrap, permutation-test and Monte Carlo bodies as typed Mojo. Imports no
Python. The binding keeps UNPACK (the two length-checked lists into words and
pointers), PACK (the returned status) and the GIL: it releases the GIL around
each call here exactly where it released it before the split."""
from core.hostptr import f32_ptr, f64_ptr, copy_f32
from core.device_plan import DevicePlan, device_plan_current
from resample.checks.statistics import MC_F_CONST, MC_F_PRODUCT, MC_F_SUM
from resample.estimator import (
    bootstrap_host_into,
    bootstrap_unpaired_host_into,
    mc_closed_form_for,
    monte_carlo_integrate_host,
    permutation_samples_host_into,
    permutation_test_host_into,
)


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    return f32_ptr(addr)


def _f64_ptr(addr: Int) raises -> MutPointer[Float64, MutUntrackedOrigin]:
    return f64_ptr(addr)


def _resample_entry_plan() raises -> DevicePlan:
    """The device plan of one resample call, read ONCE here at the entry
    (docs/plans/MULTI_GPU_ENCAPSULATION_20261103.md); everything below takes
    the value. Its `setting` is the resample family's own, exactly what
    `resample/estimator.mojo` read before the move: the plan's size while a
    plan driving the RESAMPLE family is bound to this thread, else
    MOJOLEARN_RESAMPLE_DEVICE_COUNT."""
    return device_plan_current("MOJOLEARN_RESAMPLE_DEVICE_COUNT")


def _bootstrap_run(
    xp: MutPointer[Float32, MutUntrackedOrigin],
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
    # lane ne2-b-rounds (N078): the sample is uploaded from the caller's
    # buffer and the distribution downloaded into `dp` (bootstrap_host_into); the
    # sorted distribution is the List the interval read, copied once.
    var r = bootstrap_host_into(
        xp,
        max(0, n * d),
        n,
        d,
        statistic,
        n_resamples,
        seed,
        method,
        dp,
        confidence_level,
        alternative,
        q_or_prop,
        r_first,
        plan=Optional[DevicePlan](_resample_entry_plan()),
    )
    copy_f32(r.sorted_distribution.unsafe_ptr(), sdp, n_resamples)
    sp.unsafe_store(0, Float64(r.point_estimate))
    sp.unsafe_store(1, Float64(r.standard_error))
    sp.unsafe_store(2, Float64(r.interval.low))
    sp.unsafe_store(3, Float64(r.interval.high))
    sp.unsafe_store(4, Float64(r.order_low))
    sp.unsafe_store(5, Float64(r.order_high))


def _bootstrap_unpaired_run(
    xp: MutPointer[Float32, MutUntrackedOrigin],
    n_x: Int,
    yp: MutPointer[Float32, MutUntrackedOrigin],
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
    # lane ne2-b-rounds (N078): both samples read in place, the
    # distribution downloaded into `dp`
    var r = bootstrap_unpaired_host_into(
        xp, max(0, n_x), n_x, yp, max(0, n_y), n_y, n_resamples, seed,
        method, dp, confidence_level, alternative, r_first,
    )
    copy_f32(r.sorted_distribution.unsafe_ptr(), sdp, n_resamples)
    sp.unsafe_store(0, Float64(r.point_estimate))
    sp.unsafe_store(1, Float64(r.standard_error))
    sp.unsafe_store(2, Float64(r.interval.low))
    sp.unsafe_store(3, Float64(r.interval.high))
    sp.unsafe_store(4, Float64(r.order_low))
    sp.unsafe_store(5, Float64(r.order_high))


def _permutation_run(
    xp: MutPointer[Float32, MutUntrackedOrigin],
    n_x: Int,
    yp: MutPointer[Float32, MutUntrackedOrigin],
    n_y: Int,
    statistic: Int,
    n_resamples: Int,
    seed: UInt64,
    alternative: Int,
    r_first: Int,
    np_: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
) raises:
    # lane ne2-b-rounds (N078): x and y read in place, the null downloaded
    # into `np_`
    var r = permutation_test_host_into(
        xp, n_x, yp, n_y, statistic, n_resamples, seed, alternative, np_, r_first,
        plan=Optional[DevicePlan](_resample_entry_plan()),
    )
    sp.unsafe_store(0, Float64(r.observed))
    sp.unsafe_store(1, Float64(r.pvalue.p))
    sp.unsafe_store(2, Float64(r.pvalue.count_less))
    sp.unsafe_store(3, Float64(r.pvalue.count_greater))


def _permutation_samples_run(
    xp: MutPointer[Float32, MutUntrackedOrigin],
    n: Int,
    yp: MutPointer[Float32, MutUntrackedOrigin],
    n_y: Int,
    n_resamples: Int,
    seed: UInt64,
    alternative: Int,
    r_first: Int,
    np_: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
) raises:
    # lane ne2-b-rounds (N078): the samples read in place, the null
    # downloaded into `np_`
    var r = permutation_samples_host_into(
        xp, max(0, n), yp, max(0, n_y), n_y > 0, n_resamples, seed,
        alternative, np_, r_first,
    )
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
        var r = monte_carlo_integrate_host[MC_F_CONST](
            lower, upper, n_samples, seed, i_first,
            plan=Optional[DevicePlan](_resample_entry_plan()),
        )
        integral = r.integral
        mean = r.mean
        volume = r.volume
        closed = mc_closed_form_for[MC_F_CONST](lower, upper)
    elif f_id == MC_F_SUM:
        var r = monte_carlo_integrate_host[MC_F_SUM](
            lower, upper, n_samples, seed, i_first,
            plan=Optional[DevicePlan](_resample_entry_plan()),
        )
        integral = r.integral
        mean = r.mean
        volume = r.volume
        closed = mc_closed_form_for[MC_F_SUM](lower, upper)
    elif f_id == MC_F_PRODUCT:
        var r = monte_carlo_integrate_host[MC_F_PRODUCT](
            lower, upper, n_samples, seed, i_first,
            plan=Optional[DevicePlan](_resample_entry_plan()),
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
