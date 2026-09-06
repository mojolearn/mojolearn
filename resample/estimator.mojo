# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host-visible surface: `bootstrap`, `permutation_test`,
`monte_carlo_integrate`.

**NOT YET WIRED** into `bindings/_mojolearn_estimators.mojo` or
`python/mojolearn/` -- those directories are not this lane's. The README's
`WHAT THE ORCHESTRATOR MUST WIRE` names the exact Python surface; this file is
the entry it should reach, shaped like `kde/estimator.mojo::
kde_score_samples_host` and `glm/estimator.mojo::ols_fit_host`.

WHAT THESE THREE ARE, AND WHAT THEY ARE NOT. They take host lists, refuse
BY NAME everything they cannot do, upload, run the device path with the
environment's identity trace (`MOJOLEARN_IDENTITY_TRACE`), and return host
results. Every parameter means what SciPy's parameter of that name means, or
is named differently; `resample/README.md` carries the mapping table and it is
part of the contract rather than documentation of it.

WHAT RUNS WHERE, once, so no reader has to work it out from the code:

  * the DRAWS and the PER-REPLICATE FOLDS: device, one block per replicate;
  * the SORTS: device, `core/segmented_sort.mojo`;
  * the POINT ESTIMATE, the INTERVAL, the STANDARD ERROR and the P-VALUE:
    host, over the same pinned tree
    (`metrics/checks/pinned_sum.mojo::host_tree_sum`), because they are
    O(1) or O(R) scalar work on data that has to come back anyway, and
    because a host float32 add/multiply/divide/sqrt is correctly rounded on
    every host this runs on with NOT ONE LIBM CALL among them
    (`intervals.mojo`'s header).

BUILT AND GATED ON ONE APPLE M4 IN BOTH MODES, 2026-08-25. NO SECOND VENDOR
HAS RUN THIS UNDER IDENTICAL. See `resample/README.md` under Status.
"""

from std.math import ceildiv
from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from core.segmented_sort import SORT_BLOCK, segmented_sort_keys_f32
from metrics.checks.pinned_sum import (
    PINNED_SUM_W,
    chunk_count,
    host_fold_partials,
    host_tree_sum,
)
from checks.numerics import ftz, identical_div, identical_mul, identical_sqrt
from resample.checks.index_map import (
    RESAMPLE_KIND_BOOTSTRAP,
    RESAMPLE_KIND_MONTE_CARLO,
    RESAMPLE_KIND_PERMUTATION,
    bootstrap_index_kernel,
    key_hi,
    key_lo,
    monte_carlo_point_kernel,
    resample_key,
    validate_pooled,
    validate_positions,
)
from resample.checks.intervals import (
    ALT_TWO_SIDED,
    Interval,
    METHOD_BASIC,
    METHOD_BCA,
    PValue,
    alpha_for,
    basic_interval,
    bca_acceleration,
    bca_bias_percentile,
    bca_refuse,
    distribution_standard_error,
    jackknife_stat_kernel,
    narrow_for_alternative,
    percentile_interval,
    permutation_pvalue,
)
from resample.checks.statistics import (
    MC_DIMS,
    RESAMPLE_MAX_SORT_CELLS,
    STAT_DIFF_MEANS,
    STAT_MEAN,
    STAT_PEARSON,
    STAT_QUANTILE,
    STAT_STD,
    STAT_TRIMMED_MEAN,
    _mean_of_sum,
    bootstrap_stat_kernel,
    host_sort_stable,
    materialize_resample_kernel,
    mc_box_volume,
    mc_closed_form,
    monte_carlo_chunk_kernel,
    mc_finish_host,
    order_stat_kernel,
    perm_stat_kernel,
    quantile_of_sorted_host,
    stat_columns_needed,
    stat_name,
    stat_needs_sort,
    trim_count,
)


#: SCHEDULING. Threads per block for every kernel in this lane that folds.
#: It divides `PINNED_SUM_W`, and `virtual_block_sum` folds the same tree at
#: any such value -- `check_launch_invariance` runs 256 and 64 and requires
#: byte equality. `PINNED_SUM_W` itself is NUMERIC and is not a knob here.
comptime RESAMPLE_TPB = 256

#: SCHEDULING. Threads per block for the map-only kernels (one thread per
#: position, no fold at all), so this one is not even constrained to divide
#: `PINNED_SUM_W`.
comptime RESAMPLE_MAP_TPB = 256


# ===========================================================================
# Results
# ===========================================================================


@fieldwise_init
struct BootstrapResult(Movable):
    """`scipy.stats.bootstrap`'s `BootstrapResult`, plus the sorted
    distribution (which the caller has paid for and would otherwise have to
    recompute) and the two order-statistic positions the interval used."""

    var point_estimate: Float32
    var distribution: List[Float32]
    var sorted_distribution: List[Float32]
    var standard_error: Float32
    var interval: Interval
    var order_low: Int
    var order_high: Int


@fieldwise_init
struct PermutationResult(Movable):
    """`scipy.stats.permutation_test`'s `PermutationTestResult`."""

    var observed: Float32
    var null_distribution: List[Float32]
    var pvalue: PValue


@fieldwise_init
struct MonteCarloResult(Movable):
    """No SciPy counterpart; see `statistics.mojo`'s Monte Carlo header."""

    var integral: Float32
    var mean: Float32
    var volume: Float32


# ===========================================================================
# Upload / download, the same shape `kde/estimator.mojo` uses
# ===========================================================================


def _upload(
    ctx: DeviceContext, values: List[Float32]
) raises -> DeviceBuffer[DType.float32]:
    var n = len(values)
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    for i in range(n):
        host.unsafe_ptr().unsafe_store(i, values[i])
    ctx.enqueue_copy(dst_buf=buf, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return buf^


def _download_f32(
    ctx: DeviceContext, buf: DeviceBuffer[DType.float32], n: Int
) raises -> List[Float32]:
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    var out = List[Float32]()
    for i in range(n):
        out.append(host.unsafe_ptr().unsafe_load(i))
    _ = host^
    return out^


def _download_i32(
    ctx: DeviceContext, buf: DeviceBuffer[DType.int32], n: Int
) raises -> List[Int32]:
    var host = ctx.enqueue_create_host_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    var out = List[Int32]()
    for i in range(n):
        out.append(host.unsafe_ptr().unsafe_load(i))
    _ = host^
    return out^


# ===========================================================================
# THE LAUNCH DISPATCHES
#
# `stat` and `tpb` are both COMPTIME parameters of the kernels (the caller
# composes rather than passes a pointer; `statistics.mojo`'s header), so the
# runtime ids have to be resolved to comptime ones here. Two nested `if`
# ladders, written out rather than generated, because a reader auditing which
# arm ran should be able to see it.
#
# THIS IS PORTING_RULES RULE 8'S SITE. Every arm below is a switch that
# selects a kernel, so every arm needs a named check that runs it with the
# switch set explicitly. `resample_check.mojo` enumerates them.
# ===========================================================================


def _launch_bootstrap_stat_at[
    tpb: Int
](
    ctx: DeviceContext,
    mut theta: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    key: UInt64,
    r_first: Int,
    n_replicates: Int,
    n: Int,
    n_rows: Int,
    n_features: Int,
    stat: Int,
) raises:
    if stat == STAT_MEAN:
        comptime kern = bootstrap_stat_kernel[STAT_MEAN, tpb]
        ctx.enqueue_function[kern](
            theta.unsafe_ptr(),
            x.unsafe_ptr(),
            key_lo(key),
            key_hi(key),
            Int32(r_first),
            Int32(n_replicates),
            Int32(n),
            Int32(n_rows),
            Int32(n_features),
            grid_dim=(n_replicates, 1, 1),
            block_dim=(tpb, 1, 1),
        )
    elif stat == STAT_STD:
        comptime kern2 = bootstrap_stat_kernel[STAT_STD, tpb]
        ctx.enqueue_function[kern2](
            theta.unsafe_ptr(),
            x.unsafe_ptr(),
            key_lo(key),
            key_hi(key),
            Int32(r_first),
            Int32(n_replicates),
            Int32(n),
            Int32(n_rows),
            Int32(n_features),
            grid_dim=(n_replicates, 1, 1),
            block_dim=(tpb, 1, 1),
        )
    elif stat == STAT_PEARSON:
        comptime kern3 = bootstrap_stat_kernel[STAT_PEARSON, tpb]
        ctx.enqueue_function[kern3](
            theta.unsafe_ptr(),
            x.unsafe_ptr(),
            key_lo(key),
            key_hi(key),
            Int32(r_first),
            Int32(n_replicates),
            Int32(n),
            Int32(n_rows),
            Int32(n_features),
            grid_dim=(n_replicates, 1, 1),
            block_dim=(tpb, 1, 1),
        )
    elif stat == STAT_DIFF_MEANS:
        comptime kern4 = bootstrap_stat_kernel[STAT_DIFF_MEANS, tpb]
        ctx.enqueue_function[kern4](
            theta.unsafe_ptr(),
            x.unsafe_ptr(),
            key_lo(key),
            key_hi(key),
            Int32(r_first),
            Int32(n_replicates),
            Int32(n),
            Int32(n_rows),
            Int32(n_features),
            grid_dim=(n_replicates, 1, 1),
            block_dim=(tpb, 1, 1),
        )
    else:
        raise Error(
            "resample: statistic '"
            + stat_name(stat)
            + "' has no fold arm; the order statistics go through the sort"
            " path (bootstrap_order_statistic)"
        )
    # `[[mojo-buffer-freed-at-last-use]]`: a DeviceBuffer handed to a kernel
    # as a raw pointer is dead at `.unsafe_ptr()`. Keep a use past the
    # enqueue on both buffers.
    _ = theta.unsafe_ptr()
    _ = x.unsafe_ptr()


def _launch_bootstrap_stat(
    ctx: DeviceContext,
    mut theta: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    key: UInt64,
    r_first: Int,
    n_replicates: Int,
    n: Int,
    n_rows: Int,
    n_features: Int,
    stat: Int,
    tpb: Int,
) raises:
    """Resolve `tpb` to a comptime value. The three admitted widths all
    divide `PINNED_SUM_W = 256`, which `virtual_block_sum` requires and
    `comptime assert`s."""
    if tpb == 256:
        _launch_bootstrap_stat_at[256](
            ctx, theta, x, key, r_first, n_replicates, n, n_rows, n_features, stat
        )
    elif tpb == 128:
        _launch_bootstrap_stat_at[128](
            ctx, theta, x, key, r_first, n_replicates, n, n_rows, n_features, stat
        )
    elif tpb == 64:
        _launch_bootstrap_stat_at[64](
            ctx, theta, x, key, r_first, n_replicates, n, n_rows, n_features, stat
        )
    else:
        raise Error(
            "resample: threads-per-block must be 64, 128 or 256 (it must"
            " divide PINNED_SUM_W = "
            + String(PINNED_SUM_W)
            + ", metrics/checks/pinned_sum.mojo::virtual_block_sum); got "
            + String(tpb)
        )


def _launch_perm_stat_at[
    tpb: Int
](
    ctx: DeviceContext,
    mut null_dist: DeviceBuffer[DType.float32],
    mut pooled: DeviceBuffer[DType.float32],
    key: UInt64,
    r_first: Int,
    n_replicates: Int,
    n_pooled: Int,
    n_x: Int,
    stat: Int,
) raises:
    if stat == STAT_DIFF_MEANS:
        comptime kern = perm_stat_kernel[STAT_DIFF_MEANS, tpb]
        ctx.enqueue_function[kern](
            null_dist.unsafe_ptr(),
            pooled.unsafe_ptr(),
            key_lo(key),
            key_hi(key),
            Int32(r_first),
            Int32(n_replicates),
            Int32(n_pooled),
            Int32(n_x),
            grid_dim=(n_replicates, 1, 1),
            block_dim=(tpb, 1, 1),
        )
    elif stat == STAT_MEAN:
        comptime kern2 = perm_stat_kernel[STAT_MEAN, tpb]
        ctx.enqueue_function[kern2](
            null_dist.unsafe_ptr(),
            pooled.unsafe_ptr(),
            key_lo(key),
            key_hi(key),
            Int32(r_first),
            Int32(n_replicates),
            Int32(n_pooled),
            Int32(n_x),
            grid_dim=(n_replicates, 1, 1),
            block_dim=(tpb, 1, 1),
        )
    elif stat == STAT_STD:
        comptime kern3 = perm_stat_kernel[STAT_STD, tpb]
        ctx.enqueue_function[kern3](
            null_dist.unsafe_ptr(),
            pooled.unsafe_ptr(),
            key_lo(key),
            key_hi(key),
            Int32(r_first),
            Int32(n_replicates),
            Int32(n_pooled),
            Int32(n_x),
            grid_dim=(n_replicates, 1, 1),
            block_dim=(tpb, 1, 1),
        )
    else:
        raise Error(
            "permutation_test: statistic '"
            + stat_name(stat)
            + "' is NOT PORTED for the two-sample independent case. The"
            " ported arms are mean, std and diff_means. An order statistic"
            " would need a per-replicate sort of the permuted group (the"
            " bootstrap's sort path does not apply, because the group"
            " membership changes every replicate); pearson is SciPy's"
            " permutation_type='pairings', a different null, not ported."
        )
    _ = null_dist.unsafe_ptr()
    _ = pooled.unsafe_ptr()


def _launch_perm_stat(
    ctx: DeviceContext,
    mut null_dist: DeviceBuffer[DType.float32],
    mut pooled: DeviceBuffer[DType.float32],
    key: UInt64,
    r_first: Int,
    n_replicates: Int,
    n_pooled: Int,
    n_x: Int,
    stat: Int,
    tpb: Int,
) raises:
    if tpb == 256:
        _launch_perm_stat_at[256](
            ctx, null_dist, pooled, key, r_first, n_replicates, n_pooled, n_x, stat
        )
    elif tpb == 128:
        _launch_perm_stat_at[128](
            ctx, null_dist, pooled, key, r_first, n_replicates, n_pooled, n_x, stat
        )
    elif tpb == 64:
        _launch_perm_stat_at[64](
            ctx, null_dist, pooled, key, r_first, n_replicates, n_pooled, n_x, stat
        )
    else:
        raise Error(
            "permutation_test: threads-per-block must be 64, 128 or 256; got "
            + String(tpb)
        )


def _launch_order_stat_at[
    tpb: Int
](
    ctx: DeviceContext,
    mut theta: DeviceBuffer[DType.float32],
    mut sorted_vals: DeviceBuffer[DType.float32],
    n_replicates: Int,
    n: Int,
    q_or_prop: Float32,
    stat: Int,
) raises:
    if stat == STAT_QUANTILE:
        comptime kern = order_stat_kernel[STAT_QUANTILE, tpb]
        ctx.enqueue_function[kern](
            theta.unsafe_ptr(),
            sorted_vals.unsafe_ptr(),
            Int32(n_replicates),
            Int32(n),
            q_or_prop,
            grid_dim=(n_replicates, 1, 1),
            block_dim=(tpb, 1, 1),
        )
    else:
        comptime kern2 = order_stat_kernel[STAT_TRIMMED_MEAN, tpb]
        ctx.enqueue_function[kern2](
            theta.unsafe_ptr(),
            sorted_vals.unsafe_ptr(),
            Int32(n_replicates),
            Int32(n),
            q_or_prop,
            grid_dim=(n_replicates, 1, 1),
            block_dim=(tpb, 1, 1),
        )
    _ = theta.unsafe_ptr()
    _ = sorted_vals.unsafe_ptr()


def _launch_jackknife_at[
    tpb: Int
](
    ctx: DeviceContext,
    mut theta_i: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    n: Int,
    n_features: Int,
    stat: Int,
) raises:
    if stat == STAT_MEAN:
        comptime kern = jackknife_stat_kernel[STAT_MEAN, tpb]
        ctx.enqueue_function[kern](
            theta_i.unsafe_ptr(),
            x.unsafe_ptr(),
            Int32(n),
            Int32(n_features),
            grid_dim=(n, 1, 1),
            block_dim=(tpb, 1, 1),
        )
    elif stat == STAT_STD:
        comptime kern2 = jackknife_stat_kernel[STAT_STD, tpb]
        ctx.enqueue_function[kern2](
            theta_i.unsafe_ptr(),
            x.unsafe_ptr(),
            Int32(n),
            Int32(n_features),
            grid_dim=(n, 1, 1),
            block_dim=(tpb, 1, 1),
        )
    else:
        comptime kern3 = jackknife_stat_kernel[STAT_DIFF_MEANS, tpb]
        ctx.enqueue_function[kern3](
            theta_i.unsafe_ptr(),
            x.unsafe_ptr(),
            Int32(n),
            Int32(n_features),
            grid_dim=(n, 1, 1),
            block_dim=(tpb, 1, 1),
        )
    _ = theta_i.unsafe_ptr()
    _ = x.unsafe_ptr()


# ===========================================================================
# The host point estimate and the sorts
# ===========================================================================


def point_estimate_host(
    x: List[Float32], n: Int, n_features: Int, stat: Int, q_or_prop: Float32
) raises -> Float32:
    """`statistic(sample)` -- SciPy's `theta_hat`, which `method='basic'` and
    the BCa bias correction both read.

    HOST, over `host_tree_sum`, which is the SAME tree the device folds. Not
    a device launch because it is one statistic over one sample, and not an
    unpinned loop because `basic_interval` reflects the whole interval
    through it: a last bit here moves both endpoints.
    """
    var a = List[Float32]()
    var b = List[Float32]()
    for i in range(n):
        a.append(ftz(x[i * n_features]))
        if n_features > 1:
            b.append(ftz(x[i * n_features + 1]))
        else:
            b.append(Float32(0.0))

    if stat == STAT_MEAN:
        return _mean_of_sum(host_tree_sum(a, n), n)
    if stat == STAT_DIFF_MEANS:
        return ftz(
            _mean_of_sum(host_tree_sum(a, n), n)
            - _mean_of_sum(host_tree_sum(b, n), n)
        )
    if stat == STAT_STD:
        var m = _mean_of_sum(host_tree_sum(a, n), n)
        var sq = List[Float32]()
        for i in range(n):
            var d = ftz(a[i] - m)
            sq.append(ftz(identical_mul(d, d)))
        return ftz(
            identical_sqrt(
                ftz(identical_div(host_tree_sum(sq, n), Float32(n - 1)))
            )
        )
    if stat == STAT_PEARSON:
        var mx = _mean_of_sum(host_tree_sum(a, n), n)
        var my = _mean_of_sum(host_tree_sum(b, n), n)
        var cxy = List[Float32]()
        var cxx = List[Float32]()
        var cyy = List[Float32]()
        for i in range(n):
            var dx = ftz(a[i] - mx)
            var dy = ftz(b[i] - my)
            cxy.append(ftz(identical_mul(dx, dy)))
            cxx.append(ftz(identical_mul(dx, dx)))
            cyy.append(ftz(identical_mul(dy, dy)))
        var sxx = host_tree_sum(cxx, n)
        var syy = host_tree_sum(cyy, n)
        if sxx == Float32(0.0) or syy == Float32(0.0):
            raise Error(
                "bootstrap: the point estimate of 'pearson' is 0/0 -- a"
                " column of the sample is constant, so the correlation is"
                " undefined. SciPy returns NaN and warns; this lane refuses,"
                " because resample.point is a recorded card stage and a"
                " computed NaN carries the vendor's payload (IDENTITY_PATHS"
                " row 39 FACT 2)."
            )
        return ftz(
            identical_div(
                host_tree_sum(cxy, n),
                ftz(identical_sqrt(ftz(identical_mul(sxx, syy)))),
            )
        )

    # The two order arms, over the ONE host stable sort this lane has
    # (`statistics.mojo::host_sort_stable`, keyed by
    # `core/segmented_sort.mojo::float_to_sortable` -- the same twiddle the
    # device radix passes use).
    var s = host_sort_stable(a, 0, n)
    if stat == STAT_QUANTILE:
        return quantile_of_sorted_host(s, 0, n, q_or_prop)
    var k = trim_count(n, q_or_prop)
    var kept = n - 2 * k
    if kept < 1:
        raise Error(
            "bootstrap: trimmed_mean's proportiontocut leaves "
            + String(kept)
            + " observations of "
            + String(n)
            + "; scipy.stats.trim_mean cuts int(n * proportiontocut) from"
            " EACH end, so the proportion must be below 0.5"
        )
    var keptv = List[Float32]()
    for i in range(kept):
        keptv.append(s[k + i])
    return _mean_of_sum(host_tree_sum(keptv, kept), kept)


def _sort_segments(
    ctx: DeviceContext,
    mut src: DeviceBuffer[DType.float32],
    mut dst: DeviceBuffer[DType.float32],
    n_segments: Int,
    seg_size: Int,
) raises:
    """`core/segmented_sort.mojo::segmented_sort_keys_f32` with its four
    scratch buffers allocated here, which is CUB's contract too (the caller
    supplies every temporary).

    THE ORDER THIS SORT PRODUCES IS THE ORDER THIS LANE PINS, and it is worth
    naming because the sort is where a tie stops being invisible. The keys
    are `cub::NumericTraits<float>::TwiddleIn` of the float32 bits, so `-0.0`
    (key `0x7FFFFFFF`) sorts strictly BELOW `+0.0` (key `0x80000000`) even
    though they compare equal as floats, and the LSD radix is STABLE (its own
    `seg_reorder_one_bit_kernel` docstring states and relies on it), so
    bitwise-equal values come out in ascending replicate order. The full
    total order is therefore `(twiddle_in(theta_r), r)`.

    STATED HONESTLY: the replicate index half of that order is NOT OBSERVABLE
    in this lane's output. The interval reads VALUES at ranks, and two
    bitwise-equal values are the same bits whichever order they are in, so a
    stability defect could not move an endpoint. What IS observable, and is
    gated, is the zero half -- an endpoint can be `-0.0` or `+0.0` and those
    are different bits. `check_percentile_interval` asserts the zero
    ordering on a planted distribution and REPORTS the stability against a
    host stable sort rather than claiming a gate it cannot have.
    """
    var total = n_segments * seg_size
    var blocks_wide = ceildiv(seg_size, SORT_BLOCK)
    var work_a = ctx.enqueue_create_buffer[DType.uint32](total)
    var work_b = ctx.enqueue_create_buffer[DType.uint32](total)
    var offsets = ctx.enqueue_create_buffer[DType.int32](total)
    var block_sums = ctx.enqueue_create_buffer[DType.int32](
        n_segments * blocks_wide
    )
    ctx.synchronize()
    segmented_sort_keys_f32(
        ctx,
        n_segments,
        seg_size,
        src,
        dst,
        work_a,
        work_b,
        offsets,
        block_sums,
    )
    ctx.synchronize()
    _ = work_a^
    _ = work_b^
    _ = offsets^
    _ = block_sums^


# ===========================================================================
# ENTRY POINT 1: bootstrap
# ===========================================================================


def bootstrap_host(
    x: List[Float32],
    n: Int,
    n_features: Int,
    statistic: Int,
    n_resamples: Int,
    seed: UInt64,
    method: Int,
    confidence_level: Float32 = Float32(0.95),
    alternative: Int = ALT_TWO_SIDED,
    q_or_prop: Float32 = Float32(0.5),
    r_first: Int = 0,
    tpb: Int = RESAMPLE_TPB,
    with_bca_diagnostics: Bool = False,
    map_tpb: Int = RESAMPLE_MAP_TPB,
) raises -> BootstrapResult:
    """`scipy.stats.bootstrap((x,), statistic, n_resamples=..., rng=seed,
    method=..., confidence_level=..., alternative=...)`, one shot.

    `x` is row major, `n x n_features`. The resample draws a ROW, so a
    two-column sample keeps its pairing -- SciPy's `paired=True` -- which is
    what `pearson` and `diff_means` need and what "1-D or 2-D sample" means
    here.

    `r_first` IS THE BATCH-INVARIANCE HANDLE and it is part of the public
    surface, not a test hook: a caller who wants replicates 10000..99999 of a
    run whose first 10000 they already have passes `r_first=10000`, and the
    answers are bit-identical to the corresponding slice of the whole run.
    SciPy's equivalent -- passing `bootstrap_result` back in -- CONTINUES a
    stream and therefore cannot make that promise.
    """
    validate_positions(n_resamples, n)
    if r_first < 0:
        raise Error(
            "bootstrap: r_first must be non-negative; got " + String(r_first)
        )
    validate_positions(r_first + n_resamples, n)
    if n_features < stat_columns_needed(statistic):
        raise Error(
            "bootstrap: statistic '"
            + stat_name(statistic)
            + "' reads "
            + String(stat_columns_needed(statistic))
            + " column(s) of the sample and n_features is "
            + String(n_features)
        )
    if statistic == STAT_STD and n < 2:
        raise Error(
            "bootstrap: statistic 'std' is ddof=1 (DEVIATION 1697) and needs"
            " at least 2 observations; got n=" + String(n)
        )
    if len(x) < n * n_features:
        raise Error(
            "bootstrap: the sample holds "
            + String(len(x))
            + " values but n * n_features is "
            + String(n * n_features)
        )
    for i in range(n * n_features):
        var v = x[i]
        if v != v or v > Float32(3.4e38) or v < Float32(-3.4e38):
            raise Error(
                "bootstrap: the sample contains NaN or infinity at position "
                + String(i)
                + ". Every stage of this lane is a recorded card stage and a"
                " computed NaN carries the vendor's payload (IDENTITY_PATHS"
                " row 39 FACT 2), so non-finite input is refused before any"
                " launch -- the same rule kde/ applies in DEVIATION 604 and"
                " sklearn's validate_data applies as 'contains NaN'."
            )
    if confidence_level <= Float32(0.0) or confidence_level >= Float32(1.0):
        raise Error(
            "bootstrap: confidence_level must be in (0, 1); got "
            + String(confidence_level)
        )
    if method == METHOD_BCA:
        bca_refuse()
    if stat_needs_sort(statistic):
        if n_resamples * n > RESAMPLE_MAX_SORT_CELLS:
            raise Error(
                "bootstrap: statistic '"
                + stat_name(statistic)
                + "' needs each replicate SORTED, and n_resamples * n = "
                + String(n_resamples * n)
                + " exceeds RESAMPLE_MAX_SORT_CELLS = "
                + String(RESAMPLE_MAX_SORT_CELLS)
                + ". Batch the run with r_first (the answers are"
                " bit-identical to the unbatched ones, which is DEVIATION"
                " 1690(b)), or close this refusal by streaming the"
                " materialised segments through the sort in tiles."
            )
        if statistic == STAT_QUANTILE:
            if q_or_prop < Float32(0.0) or q_or_prop > Float32(1.0):
                raise Error(
                    "bootstrap: statistic 'quantile' needs q in [0, 1]; got "
                    + String(q_or_prop)
                )
        else:
            if q_or_prop < Float32(0.0) or q_or_prop >= Float32(0.5):
                raise Error(
                    "bootstrap: statistic 'trimmed_mean' needs"
                    " proportiontocut in [0, 0.5); got "
                    + String(q_or_prop)
                )

    var ctx = DeviceContext()
    var trace = IdentityTrace()
    trace.header(
        "resample bootstrap: n="
        + String(n)
        + " d="
        + String(n_features)
        + " statistic="
        + stat_name(statistic)
        + " n_resamples="
        + String(n_resamples)
        + " r_first="
        + String(r_first)
        + " method="
        + String(method)
        + " confidence_level="
        + String(confidence_level)
    )

    var key = resample_key(seed, RESAMPLE_KIND_BOOTSTRAP)
    var key_words: List[Int32] = [key_lo(key), key_hi(key)]
    trace.record_list_i32("resample.key", key_words)

    var dx = _upload(ctx, x)
    var theta = ctx.enqueue_create_buffer[DType.float32](n_resamples)
    ctx.synchronize()

    # The index map, recorded on a bounded window so the card stays a fixed
    # size whatever `n_resamples` is (rule 3 of core/identity_trace.mojo:
    # hash the logical buffer, and here the logical buffer is the WINDOW the
    # card is defined over).
    var win_r = 4 if n_resamples > 4 else n_resamples
    var win_i = 32 if n > 32 else n
    var idx_buf = ctx.enqueue_create_buffer[DType.int32](win_r * win_i)
    ctx.synchronize()
    ctx.enqueue_function[bootstrap_index_kernel](
        idx_buf.unsafe_ptr(),
        key_lo(key),
        key_hi(key),
        Int32(r_first),
        Int32(win_r),
        Int32(n),
        Int32(win_i),
        grid_dim=(ceildiv(win_r * win_i, map_tpb), 1, 1),
        block_dim=(map_tpb, 1, 1),
    )
    _ = idx_buf.unsafe_ptr()
    ctx.synchronize()
    trace.record_device(ctx, "resample.index_map", idx_buf, win_r * win_i)

    if stat_needs_sort(statistic):
        var cells = n_resamples * n
        var vals = ctx.enqueue_create_buffer[DType.float32](cells)
        var svals = ctx.enqueue_create_buffer[DType.float32](cells)
        ctx.synchronize()
        ctx.enqueue_function[materialize_resample_kernel](
            vals.unsafe_ptr(),
            dx.unsafe_ptr(),
            key_lo(key),
            key_hi(key),
            Int32(r_first),
            Int32(n_resamples),
            Int32(n),
            Int32(n),
            Int32(n_features),
            Int32(0),
            grid_dim=(ceildiv(cells, map_tpb), 1, 1),
            block_dim=(map_tpb, 1, 1),
        )
        _ = vals.unsafe_ptr()
        _ = dx.unsafe_ptr()
        ctx.synchronize()
        _sort_segments(ctx, vals, svals, n_resamples, n)
        if tpb == 256:
            _launch_order_stat_at[256](
                ctx, theta, svals, n_resamples, n, q_or_prop, statistic
            )
        elif tpb == 128:
            _launch_order_stat_at[128](
                ctx, theta, svals, n_resamples, n, q_or_prop, statistic
            )
        elif tpb == 64:
            _launch_order_stat_at[64](
                ctx, theta, svals, n_resamples, n, q_or_prop, statistic
            )
        else:
            raise Error(
                "bootstrap: threads-per-block must be 64, 128 or 256; got "
                + String(tpb)
            )
        ctx.synchronize()
        _ = vals^
        _ = svals^
    else:
        _launch_bootstrap_stat(
            ctx,
            theta,
            dx,
            key,
            r_first,
            n_resamples,
            n,
            n,
            n_features,
            statistic,
            tpb,
        )
        ctx.synchronize()

    trace.record_device(ctx, "resample.theta", theta, n_resamples)
    var dist = _download_f32(ctx, theta, n_resamples)

    var sorted_buf = ctx.enqueue_create_buffer[DType.float32](n_resamples)
    ctx.synchronize()
    _sort_segments(ctx, theta, sorted_buf, 1, n_resamples)
    trace.record_device(ctx, "resample.sorted", sorted_buf, n_resamples)
    var sorted_dist = _download_f32(ctx, sorted_buf, n_resamples)

    var theta_hat = point_estimate_host(x, n, n_features, statistic, q_or_prop)
    trace.record_scalar_f32("resample.point", theta_hat)

    var alpha = alpha_for(confidence_level, alternative)
    var interval: Interval
    if method == METHOD_BASIC:
        interval = basic_interval(sorted_dist, n_resamples, alpha, theta_hat)
    else:
        interval = percentile_interval(sorted_dist, n_resamples, alpha)
    interval = narrow_for_alternative(interval, alternative)

    # The two order-statistic POSITIONS the interval read, recorded because a
    # cross-vendor difference in an endpoint is either a different position
    # or a different value at the same position, and those have different
    # causes and different fixes.
    var h_lo = Float32(n_resamples - 1) * alpha
    var h_hi = Float32(n_resamples - 1) * ftz(Float32(1.0) - alpha)
    var pos_lo = Int(h_lo)
    var pos_hi = Int(h_hi)
    var pos_words: List[Int32] = [Int32(pos_lo), Int32(pos_hi)]
    trace.record_list_i32("resample.order_pos", pos_words)

    var se = distribution_standard_error(dist, n_resamples)
    trace.record_scalar_f32("resample.se", se)
    var ends: List[Float32] = [interval.low, interval.high]
    trace.record_list_f32("resample.interval", ends)

    if with_bca_diagnostics:
        # DEVIATION 1699's identical half: computed and RECORDED even though
        # the method is refused, so the construction is gated rather than
        # dead and the closure is one function away.
        var z0p = bca_bias_percentile(sorted_dist, n_resamples, theta_hat)
        trace.record_scalar_f32("resample.bca.z0p", z0p)
        var jack = ctx.enqueue_create_buffer[DType.float32](n)
        ctx.synchronize()
        if tpb == 256:
            _launch_jackknife_at[256](ctx, jack, dx, n, n_features, statistic)
        elif tpb == 128:
            _launch_jackknife_at[128](ctx, jack, dx, n, n_features, statistic)
        else:
            _launch_jackknife_at[64](ctx, jack, dx, n, n_features, statistic)
        ctx.synchronize()
        trace.record_device(ctx, "resample.jackknife", jack, n)
        var jack_h = _download_f32(ctx, jack, n)
        trace.record_scalar_f32(
            "resample.bca.ahat", bca_acceleration(jack_h, n)
        )
        _ = jack^

    _ = dx^
    _ = idx_buf^
    _ = theta^
    _ = sorted_buf^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return BootstrapResult(
        theta_hat, dist^, sorted_dist^, se, interval, pos_lo, pos_hi
    )


# ===========================================================================
# ENTRY POINT 2: permutation_test
# ===========================================================================


def permutation_test_host(
    x: List[Float32],
    y: List[Float32],
    statistic: Int,
    n_resamples: Int,
    seed: UInt64,
    alternative: Int,
    r_first: Int = 0,
    tpb: Int = RESAMPLE_TPB,
) raises -> PermutationResult:
    """`scipy.stats.permutation_test((x, y), statistic,
    permutation_type='independent', n_resamples=..., rng=seed,
    alternative=...)`.

    ONE-DIMENSIONAL `x` and `y`; the two-sample independent null pools them
    and re-splits, so a second column would have no meaning under it (a
    paired statistic is SciPy's `permutation_type='pairings'`, which is not
    ported -- see `resample/NOT_IMPLEMENTED.tsv`).

    THE NULL IS NEVER EXHAUSTIVE HERE. SciPy switches to enumerating all
    `C(n_x + n_y, n_x)` partitions when `n_resamples >= n_max` and then drops
    the `+1` adjustment. This lane always samples and always adjusts
    (DEVIATION 1702), which is CONSERVATIVE -- it can only make a p-value
    larger -- and is stated so nobody reads a floor of `1/(R+1)` as a claim
    of exactness.
    """
    var n_x = len(x)
    var n_y = len(y)
    var n_pooled = n_x + n_y
    if n_x <= 0 or n_y <= 0:
        raise Error(
            "permutation_test: both samples must be non-empty; got n_x="
            + String(n_x)
            + " n_y="
            + String(n_y)
        )
    validate_positions(n_resamples, n_pooled)
    validate_positions(r_first + n_resamples, n_pooled)
    validate_pooled(n_pooled)
    if statistic == STAT_STD and n_x < 2:
        raise Error(
            "permutation_test: statistic 'std' is ddof=1 and needs at least"
            " 2 observations in the first sample; got n_x=" + String(n_x)
        )
    var pooled = List[Float32]()
    for i in range(n_x):
        pooled.append(x[i])
    for i in range(n_y):
        pooled.append(y[i])
    for i in range(n_pooled):
        var v = pooled[i]
        if v != v or v > Float32(3.4e38) or v < Float32(-3.4e38):
            raise Error(
                "permutation_test: the pooled sample contains NaN or infinity"
                " at position "
                + String(i)
                + "; non-finite input is refused before any launch (see"
                " bootstrap_host for the row-39 reason)."
            )

    var ctx = DeviceContext()
    var trace = IdentityTrace()
    trace.header(
        "resample permutation_test: n_x="
        + String(n_x)
        + " n_y="
        + String(n_y)
        + " statistic="
        + stat_name(statistic)
        + " n_resamples="
        + String(n_resamples)
        + " r_first="
        + String(r_first)
    )
    var key = resample_key(seed, RESAMPLE_KIND_PERMUTATION)
    var key_words: List[Int32] = [key_lo(key), key_hi(key)]
    trace.record_list_i32("resample.key", key_words)

    var dpool = _upload(ctx, pooled)
    var null_buf = ctx.enqueue_create_buffer[DType.float32](n_resamples)
    ctx.synchronize()
    _launch_perm_stat(
        ctx,
        null_buf,
        dpool,
        key,
        r_first,
        n_resamples,
        n_pooled,
        n_x,
        statistic,
        tpb,
    )
    ctx.synchronize()
    trace.record_device(ctx, "resample.null", null_buf, n_resamples)
    var null_dist = _download_f32(ctx, null_buf, n_resamples)

    # The OBSERVED statistic is the pooled sample split where it already is,
    # i.e. the identity permutation. Host, over the same pinned tree, for
    # `point_estimate_host`'s reason: the p-value's tolerance `gamma` is a
    # multiple of it, so a last bit here moves a count.
    var vx = List[Float32]()
    var vy = List[Float32]()
    for j in range(n_pooled):
        var v = ftz(pooled[j])
        if j < n_x:
            vx.append(v)
            vy.append(Float32(0.0))
        else:
            vx.append(Float32(0.0))
            vy.append(v)
    var sx = host_tree_sum(vx, n_pooled)
    var sy = host_tree_sum(vy, n_pooled)
    var observed: Float32
    if statistic == STAT_DIFF_MEANS:
        observed = ftz(_mean_of_sum(sx, n_x) - _mean_of_sum(sy, n_y))
    elif statistic == STAT_MEAN:
        observed = _mean_of_sum(sx, n_x)
    else:
        var mx = _mean_of_sum(sx, n_x)
        var sq = List[Float32]()
        for j in range(n_pooled):
            if j < n_x:
                var d = ftz(ftz(pooled[j]) - mx)
                sq.append(ftz(identical_mul(d, d)))
            else:
                sq.append(Float32(0.0))
        observed = ftz(
            identical_sqrt(
                ftz(
                    identical_div(
                        host_tree_sum(sq, n_pooled), Float32(n_x - 1)
                    )
                )
            )
        )
    trace.record_scalar_f32("resample.observed", observed)

    var pv = permutation_pvalue(null_dist, n_resamples, observed, alternative)
    trace.record_scalar_f32("resample.pvalue", pv.p)

    _ = dpool^
    _ = null_buf^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return PermutationResult(observed, null_dist^, pv)


# ===========================================================================
# ENTRY POINT 3: monte_carlo_integrate
# ===========================================================================


def _launch_mc_at[
    f_id: Int, tpb: Int
](
    ctx: DeviceContext,
    mut partials: DeviceBuffer[DType.float32],
    mut lower: DeviceBuffer[DType.float32],
    mut span: DeviceBuffer[DType.float32],
    key: UInt64,
    i_first: Int,
    n_samples: Int,
    n_chunks: Int,
    grid_blocks: Int,
) raises:
    comptime kern = monte_carlo_chunk_kernel[f_id, tpb]
    ctx.enqueue_function[kern](
        partials.unsafe_ptr(),
        lower.unsafe_ptr(),
        span.unsafe_ptr(),
        key_lo(key),
        key_hi(key),
        Int32(i_first),
        Int32(n_samples),
        Int32(n_chunks),
        grid_dim=(grid_blocks, 1, 1),
        block_dim=(tpb, 1, 1),
    )
    _ = partials.unsafe_ptr()
    _ = lower.unsafe_ptr()
    _ = span.unsafe_ptr()


def monte_carlo_integrate_host[
    f_id: Int
](
    lower: List[Float32],
    upper: List[Float32],
    n_samples: Int,
    seed: UInt64,
    i_first: Int = 0,
    tpb: Int = RESAMPLE_TPB,
    grid_blocks: Int = 0,
    map_tpb: Int = RESAMPLE_MAP_TPB,
) raises -> MonteCarloResult:
    """`volume * mean(f(x_i))` over `n_samples` points drawn uniformly from
    the rectangle `[lower, upper)`.

    `f_id` IS A COMPTIME PARAMETER of this function, so the integrand is
    chosen at the call site and compiled in -- there is no runtime dispatch
    and no pointer. `statistics.mojo`'s Monte Carlo header says why.

    `grid_blocks = 0` means `ceil(n_chunks / 1)` blocks, one per chunk;
    anything else is a SCHEDULING choice and the answer does not move,
    because a physical block serves chunks `linear_block_id(),
    + physical_block_count(), ...` and the chunk INDEX -- not the block that
    computed it -- decides which values share a tree.
    """
    if n_samples <= 0:
        raise Error(
            "monte_carlo_integrate: n_samples must be positive; got "
            + String(n_samples)
        )
    if i_first < 0:
        raise Error(
            "monte_carlo_integrate: i_first must be non-negative; got "
            + String(i_first)
        )
    validate_positions(i_first + n_samples, MC_DIMS)
    if len(lower) != MC_DIMS or len(upper) != MC_DIMS:
        raise Error(
            "monte_carlo_integrate: the supplied integrands are"
            " "
            + String(MC_DIMS)
            + "-dimensional; got lower of length "
            + String(len(lower))
            + " and upper of length "
            + String(len(upper))
        )
    for d in range(MC_DIMS):
        if not (upper[d] > lower[d]):
            raise Error(
                "monte_carlo_integrate: upper must exceed lower in every"
                " dimension; dimension "
                + String(d)
                + " has lower="
                + String(lower[d])
                + " upper="
                + String(upper[d])
                + ". A degenerate or inverted box has no uniform measure, and"
                " a negative span would silently return a negative-volume"
                " answer rather than raising."
            )

    var span = List[Float32]()
    for d in range(MC_DIMS):
        span.append(ftz(upper[d] - lower[d]))
    var volume = mc_box_volume(lower, upper)

    var ctx = DeviceContext()
    var trace = IdentityTrace()
    trace.header(
        "resample monte_carlo_integrate: n_samples="
        + String(n_samples)
        + " i_first="
        + String(i_first)
        + " dims="
        + String(MC_DIMS)
    )
    var key = resample_key(seed, RESAMPLE_KIND_MONTE_CARLO)
    var key_words: List[Int32] = [key_lo(key), key_hi(key)]
    trace.record_list_i32("resample.key", key_words)

    var dlower = _upload(ctx, lower)
    var dspan = _upload(ctx, span)
    var n_chunks = chunk_count(n_samples)
    var partials = ctx.enqueue_create_buffer[DType.float32](n_chunks)
    ctx.synchronize()

    var blocks = grid_blocks if grid_blocks > 0 else n_chunks
    if tpb == 256:
        _launch_mc_at[f_id, 256](
            ctx, partials, dlower, dspan, key, i_first, n_samples, n_chunks, blocks
        )
    elif tpb == 128:
        _launch_mc_at[f_id, 128](
            ctx, partials, dlower, dspan, key, i_first, n_samples, n_chunks, blocks
        )
    elif tpb == 64:
        _launch_mc_at[f_id, 64](
            ctx, partials, dlower, dspan, key, i_first, n_samples, n_chunks, blocks
        )
    else:
        raise Error(
            "monte_carlo_integrate: threads-per-block must be 64, 128 or 256;"
            " got " + String(tpb)
        )
    ctx.synchronize()

    # A bounded window of coordinates, for the card and for the map gate.
    var win = 32 if n_samples > 32 else n_samples
    var pts = ctx.enqueue_create_buffer[DType.float32](win * MC_DIMS)
    ctx.synchronize()
    ctx.enqueue_function[monte_carlo_point_kernel](
        pts.unsafe_ptr(),
        dlower.unsafe_ptr(),
        dspan.unsafe_ptr(),
        key_lo(key),
        key_hi(key),
        Int32(i_first),
        Int32(win),
        Int32(MC_DIMS),
        grid_dim=(ceildiv(win * MC_DIMS, map_tpb), 1, 1),
        block_dim=(map_tpb, 1, 1),
    )
    _ = pts.unsafe_ptr()
    ctx.synchronize()
    trace.record_device(ctx, "resample.mc.points", pts, win * MC_DIMS)
    trace.record_device(ctx, "resample.mc.partials", partials, n_chunks)

    var host_partials = _download_f32(ctx, partials, n_chunks)
    var integral = mc_finish_host(host_partials, n_chunks, n_samples, volume)
    var mean = _mean_of_sum(
        host_fold_partials(host_partials, n_chunks), n_samples
    )
    trace.record_scalar_f32("resample.mc.mean", mean)
    trace.record_scalar_f32("resample.mc.integral", integral)

    _ = dlower^
    _ = dspan^
    _ = partials^
    _ = pts^

    # DEVIATION 1946: the context dies LAST, after every value built on it.
    # Mojo frees at LAST USE, so without this the buffer releases above run
    # against a context that is already gone. On sm_89 the next GPU call in
    # the process then never returns (GPU idle, host threads in futex wait);
    # Apple and AMD do not show it, which is how it stayed latent here.
    _ = ctx^
    return MonteCarloResult(integral, mean, volume)


def mc_closed_form_for[
    f_id: Int
](lower: List[Float32], upper: List[Float32]) -> Float32:
    """The hand-derived exact integral; re-exported so a caller (and
    `resample_main.mojo`) does not have to import `statistics.mojo`."""
    return mc_closed_form[f_id](lower, upper)
