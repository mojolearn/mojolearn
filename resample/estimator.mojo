# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host-visible surface: `bootstrap`, `permutation_test`,
`monte_carlo_integrate`.

**NOT YET WIRED** into `bindings/_mojolearn_estimators.mojo` or
`python/mojolearn/` -- those directories are not this lane's. The README's
`WHAT THE MAINTAINER MUST WIRE` names the exact Python surface; this file is
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

# DEVIATION 2486: bulk host staging; stream/lifetime boundaries unchanged.
from bindings.hostptr import copy_f32
from std.math import ceildiv
from std.os import getenv
from std.sys.compile import is_defined
from bindings.hostptr import f32_ptr, i32_ptr
from max.gpu.host import DeviceBuffer, DeviceContext
from resample.fast_apple import (
    RESAMPLE_FAST_APPLE,
    bootstrap_mean_fast,
    gather_rows_f32,
    perm_select_fast,
    rank_sort_f32,
)
from core.neural_context import process_ctx
from checks.numerics import GLOBAL_NUMERIC_MODE as _DEVCTX_MODE, NUMERIC_IDENTICAL as _DEVCTX_IDENTICAL

#: This binding's ONE process-lifetime DeviceContext (core/neural_context.mojo,
#: lane/devctx-lifetime): a context per call exhausts Metal command queues.
comptime _DEVCTX_SLOT = "MojoResampleContextIdentical" if _DEVCTX_MODE == _DEVCTX_IDENTICAL else "MojoResampleContextFast"


from core.identity_trace import IdentityTrace
from core.segmented_sort import SORT_BLOCK, segmented_sort_keys_f32
from metrics.checks.pinned_sum import (
    PINNED_SUM_W,
    canonicalize_nan,
    chunk_count,
    host_fold_partials,
    host_tree_sum,
)
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_div,
    identical_mul,
    identical_sqrt,
)
from resample.checks.index_map import (
    RESAMPLE_KIND_BOOTSTRAP,
    RESAMPLE_KIND_BOOTSTRAP_SECOND,
    RESAMPLE_KIND_MONTE_CARLO,
    RESAMPLE_KIND_PERM_SAMPLES,
    RESAMPLE_KIND_UTILS_PERMUTE,
    RESAMPLE_KIND_UTILS_REPLACE,
    utils_first_by_key,
    utils_validate,
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
    bca_acceleration_two,
    bca_bias_percentile,
    bca_interval,
    bca_validate,
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
    perm_samples_kernel,
    utils_draw_kernel,
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
# FAST + Apple switches (lane/apple-fast-resample, 2026-10-02). Read on the
# host at dispatch; every one defaults OFF and is compiled under
# `RESAMPLE_FAST_APPLE` only (resample/fast_apple.mojo has the kernels and
# the causes). IDENTICAL compiles the old path exactly.
# ===========================================================================


def _fast_rank_sort_on() -> Bool:
    """MOJOLEARN_RESAMPLE_FAST_RANK_SORT=1: the bootstrap distribution sorted
    by one rank launch instead of `_sort_segments` (130 launches, 32 of them
    one-thread scans, for one segment of n_resamples keys)."""
    comptime if RESAMPLE_FAST_APPLE:
        return String(getenv("MOJOLEARN_RESAMPLE_FAST_RANK_SORT")) == "1"
    else:
        return False


def _fast_one_fold_on() -> Bool:
    """MOJOLEARN_RESAMPLE_FAST_ONE_FOLD=1: mean / diff_means replicates folded
    once per block (registers, then block.sum) instead of once per 256-draw
    chunk (`_chunked_sum`'s virtual_block_sum per chunk)."""
    comptime if RESAMPLE_FAST_APPLE:
        return String(getenv("MOJOLEARN_RESAMPLE_FAST_ONE_FOLD")) == "1"
    else:
        return False


def _fast_perm_select_on() -> Bool:
    """MOJOLEARN_RESAMPLE_FAST_PERM_SELECT=1: the permutation null by radix
    select, no PERM_MAX_POOLED (perm_stat_kernel's O(N^2) counting rank and
    its 1024-position threadgroup bound refuse the board's 40,000)."""
    comptime if RESAMPLE_FAST_APPLE:
        return String(getenv("MOJOLEARN_RESAMPLE_FAST_PERM_SELECT")) == "1"
    else:
        return False


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
    copy_f32(values.unsafe_ptr(), host.unsafe_ptr(), n)
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
# THIS IS CONTRIBUTING.md RULE 8'S SITE. Every arm below is a switch that
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
            + "' is NOT IMPLEMENTED for the two-sample independent case. The"
            " implemented arms are mean, std and diff_means. An order statistic"
            " would need a per-replicate sort of the permuted group (the"
            " bootstrap's sort path does not apply, because the group"
            " membership changes every replicate); pearson is SciPy's"
            " permutation_type='pairings', a different null, not implemented."
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


def resample_device_count() raises -> Int:
    """MOJOLEARN_RESAMPLE_DEVICE_COUNT: replicate or sample ranges on owners."""
    var value = String(getenv("MOJOLEARN_RESAMPLE_DEVICE_COUNT"))
    if value == "":
        return 1
    var count = Int(value)
    if count < 1 or count > 64:
        raise Error("resample device count must be in [1, 64]")
    if count > 1 and GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("multi-GPU resampling requires IDENTICAL numeric mode")
    return count


def _owner_offset(rank: Int) -> Int:
    """Check-only arm: later owners compute their range one position late."""
    comptime if is_defined["MOJOLEARN_RESAMPLE_PARALLEL_SABOTAGE"]():
        if rank > 0:
            return 1
    return 0


def _bootstrap_theta(
    ctx: DeviceContext,
    mut theta: DeviceBuffer[DType.float32],
    mut dx: DeviceBuffer[DType.float32],
    key: UInt64,
    r_first: Int,
    n_resamples: Int,
    n: Int,
    n_features: Int,
    statistic: Int,
    q_or_prop: Float32,
    tpb: Int,
    map_tpb: Int,
) raises:
    """Replicates `[r_first, r_first + n_resamples)` into `theta`. Every
    replicate is a pure function of its global index and the sample
    (DEVIATION 1690(b), the r_first batch-invariance handle)."""
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
        # lane/apple-fast-resample (2026-10-02), MOJOLEARN_RESAMPLE_FAST_ONE_FOLD=1:
        # `bootstrap_stat_kernel` folds every 256-draw chunk through
        # `virtual_block_sum` (resample/checks/statistics.mojo `_chunked_sum`),
        # 79 block folds per replicate at n = 20,000; the FAST kernel folds a
        # replicate once. Same draws, FAST's own summation order.
        var folded = False
        comptime if RESAMPLE_FAST_APPLE:
            if (
                (statistic == STAT_MEAN or statistic == STAT_DIFF_MEANS)
                and tpb == 256
                and _fast_one_fold_on()
            ):
                bootstrap_mean_fast(
                    ctx, theta, dx, key, r_first, n_resamples, n, n_features,
                    statistic,
                )
                folded = True
        if not folded:
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


def _bootstrap_theta_owners(
    ctx: DeviceContext,
    mut theta: DeviceBuffer[DType.float32],
    x: List[Float32],
    key: UInt64,
    r_first: Int,
    n_resamples: Int,
    n: Int,
    n_features: Int,
    statistic: Int,
    q_or_prop: Float32,
    tpb: Int,
    map_tpb: Int,
    count: Int,
) raises:
    """Contiguous global replicate ranges on owners, bytes copied into place."""
    var active = min(count, n_resamples)
    ctx.synchronize()
    for rank in range(active):
        var begin = n_resamples * rank // active
        var rows = n_resamples * (rank + 1) // active - begin
        var owner = DeviceContext(device_id=rank)
        var ox = _upload(owner, x)
        var ot = owner.enqueue_create_buffer[DType.float32](rows)
        owner.synchronize()
        _bootstrap_theta(
            owner, ot, ox, key, r_first + begin + _owner_offset(rank), rows,
            n, n_features, statistic, q_or_prop, tpb, map_tpb,
        )
        owner.synchronize()
        var view = theta.create_sub_buffer[DType.float32](begin, rows)
        ot.enqueue_copy_to(view)
        owner.synchronize()
        ctx.synchronize()
        _ = view^
        _ = ox^
        _ = ot^
        _ = owner^
    ctx.synchronize()


def _perm_null_owners(
    ctx: DeviceContext,
    mut null_buf: DeviceBuffer[DType.float32],
    pooled: List[Float32],
    key: UInt64,
    r_first: Int,
    n_resamples: Int,
    n_pooled: Int,
    n_x: Int,
    statistic: Int,
    tpb: Int,
    count: Int,
) raises:
    """Contiguous global permutation ranges on owners, bytes copied into place."""
    var active = min(count, n_resamples)
    ctx.synchronize()
    for rank in range(active):
        var begin = n_resamples * rank // active
        var rows = n_resamples * (rank + 1) // active - begin
        var owner = DeviceContext(device_id=rank)
        var op = _upload(owner, pooled)
        var on = owner.enqueue_create_buffer[DType.float32](rows)
        owner.synchronize()
        _launch_perm_stat(
            owner, on, op, key, r_first + begin + _owner_offset(rank), rows,
            n_pooled, n_x, statistic, tpb,
        )
        owner.synchronize()
        var view = null_buf.create_sub_buffer[DType.float32](begin, rows)
        on.enqueue_copy_to(view)
        owner.synchronize()
        ctx.synchronize()
        _ = view^
        _ = op^
        _ = on^
        _ = owner^
    ctx.synchronize()


def _mc_partials[
    f_id: Int
](
    ctx: DeviceContext,
    mut partials: DeviceBuffer[DType.float32],
    mut dlower: DeviceBuffer[DType.float32],
    mut dspan: DeviceBuffer[DType.float32],
    key: UInt64,
    i_first: Int,
    n_samples: Int,
    n_chunks: Int,
    blocks: Int,
    tpb: Int,
) raises:
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

def _mc_partials_owners[
    f_id: Int
](
    ctx: DeviceContext,
    mut partials: DeviceBuffer[DType.float32],
    lower: List[Float32],
    span: List[Float32],
    key: UInt64,
    i_first: Int,
    n_samples: Int,
    n_chunks: Int,
    tpb: Int,
    count: Int,
) raises:
    """Whole PINNED_SUM_W chunks on owners. An owner starting at global chunk
    c0 draws from `i_first + c0 * PINNED_SUM_W`, so its chunk j is global
    chunk c0 + j with the same values and the same pinned fold; the root folds
    all partials in chunk order."""
    var active = min(count, n_chunks)
    ctx.synchronize()
    for rank in range(active):
        var c0 = n_chunks * rank // active
        var c1 = n_chunks * (rank + 1) // active
        var first_sample = c0 * PINNED_SUM_W
        var last_sample = c1 * PINNED_SUM_W
        if last_sample > n_samples:
            last_sample = n_samples
        var chunks = c1 - c0
        var owner = DeviceContext(device_id=rank)
        var ol = _upload(owner, lower)
        var ospan = _upload(owner, span)
        var opart = owner.enqueue_create_buffer[DType.float32](chunks)
        owner.synchronize()
        _mc_partials[f_id](
            owner, opart, ol, ospan, key,
            i_first + first_sample + _owner_offset(rank),
            last_sample - first_sample, chunks, chunks, tpb,
        )
        owner.synchronize()
        var view = partials.create_sub_buffer[DType.float32](c0, chunks)
        opart.enqueue_copy_to(view)
        owner.synchronize()
        ctx.synchronize()
        _ = view^
        _ = ol^
        _ = ospan^
        _ = opart^
        _ = owner^
    ctx.synchronize()


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
        bca_validate(statistic, n)
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

    var ctx = process_ctx[_DEVCTX_SLOT]()
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

    var owners = resample_device_count()
    if owners > 1 and n_resamples > 1:
        _bootstrap_theta_owners(
            ctx, theta, x, key, r_first, n_resamples, n, n_features,
            statistic, q_or_prop, tpb, map_tpb, owners,
        )
    else:
        _bootstrap_theta(
            ctx, theta, dx, key, r_first, n_resamples, n, n_features,
            statistic, q_or_prop, tpb, map_tpb,
        )
    trace.record_device(ctx, "resample.theta", theta, n_resamples)
    var dist = _download_f32(ctx, theta, n_resamples)

    var sorted_buf = ctx.enqueue_create_buffer[DType.float32](n_resamples)
    ctx.synchronize()
    # lane/apple-fast-resample (2026-10-02), MOJOLEARN_RESAMPLE_FAST_RANK_SORT=1:
    # `_sort_segments` runs core/segmented_sort.mojo's 32 one-bit passes (4
    # launches each, one of them `seg_scan_block_sums_kernel` on ONE thread)
    # over this single segment of n_resamples keys: 130 launches for 9,999
    # floats, the bootstrap's Apple cost (M3 16 ms vs 2 ms on the L40S).
    # The rank launch produces the same order and the same bits.
    var ranked = False
    comptime if RESAMPLE_FAST_APPLE:
        if _fast_rank_sort_on():
            rank_sort_f32(ctx, theta, sorted_buf, n_resamples)
            ranked = True
    if not ranked:
        _sort_segments(ctx, theta, sorted_buf, 1, n_resamples)
    trace.record_device(ctx, "resample.sorted", sorted_buf, n_resamples)
    var sorted_dist = _download_f32(ctx, sorted_buf, n_resamples)

    var theta_hat = point_estimate_host(x, n, n_features, statistic, q_or_prop)
    trace.record_scalar_f32("resample.point", theta_hat)

    # BCa (DEVIATION 1699, closed by 5410) needs the bias percentile and the
    # jackknife BEFORE the interval; the diagnostics record the same values
    # below, where they always were on the card.
    var need_jack = with_bca_diagnostics or method == METHOD_BCA
    var jack = ctx.enqueue_create_buffer[DType.float32](n if need_jack else 1)
    var jack_h = List[Float32]()
    var z0p = Float32(0.0)
    var ahat = Float32(0.0)
    if need_jack:
        z0p = bca_bias_percentile(sorted_dist, n_resamples, theta_hat)
        ctx.synchronize()
        if tpb == 256:
            _launch_jackknife_at[256](ctx, jack, dx, n, n_features, statistic)
        elif tpb == 128:
            _launch_jackknife_at[128](ctx, jack, dx, n, n_features, statistic)
        else:
            _launch_jackknife_at[64](ctx, jack, dx, n, n_features, statistic)
        ctx.synchronize()
        jack_h = _download_f32(ctx, jack, n)
        ahat = bca_acceleration(jack_h, n)

    var alpha = alpha_for(confidence_level, alternative)
    var interval: Interval
    var lvl_lo = alpha
    var lvl_hi = ftz(Float32(1.0) - alpha)
    if method == METHOD_BASIC:
        interval = basic_interval(sorted_dist, n_resamples, alpha, theta_hat)
    elif method == METHOD_BCA:
        var ends = bca_interval(sorted_dist, n_resamples, alpha, z0p, ahat)
        interval = ends.interval
        lvl_lo = ends.alpha_1
        lvl_hi = ends.alpha_2
    else:
        interval = percentile_interval(sorted_dist, n_resamples, alpha)
    interval = narrow_for_alternative(interval, alternative)

    # The two order-statistic POSITIONS the interval read, recorded because a
    # cross-vendor difference in an endpoint is either a different position
    # or a different value at the same position, and those have different
    # causes and different fixes.
    var h_lo = Float32(n_resamples - 1) * lvl_lo
    var h_hi = Float32(n_resamples - 1) * lvl_hi
    var pos_lo = Int(h_lo)
    var pos_hi = Int(h_hi)
    var pos_words: List[Int32] = [Int32(pos_lo), Int32(pos_hi)]
    trace.record_list_i32("resample.order_pos", pos_words)

    var se = distribution_standard_error(dist, n_resamples)
    trace.record_scalar_f32("resample.se", se)
    var ends: List[Float32] = [interval.low, interval.high]
    trace.record_list_f32("resample.interval", ends)
    if method == METHOD_BCA:
        var levels: List[Float32] = [lvl_lo, lvl_hi]
        trace.record_list_f32("resample.bca.levels", levels)

    if with_bca_diagnostics:
        # DEVIATION 1699's diagnostics: the bias percentile, the jackknife
        # and the acceleration, recorded for every statistic with an arm.
        trace.record_scalar_f32("resample.bca.z0p", z0p)
        trace.record_device(ctx, "resample.jackknife", jack, n)
        trace.record_scalar_f32("resample.bca.ahat", ahat)
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


def _unpaired_validate(
    x: List[Float32], n_x: Int, y: List[Float32], n_y: Int, n_resamples: Int,
    method: Int, confidence_level: Float32, r_first: Int,
) raises:
    """`bootstrap_unpaired_host`'s refusals, shared word for word with the
    host twin (`resample/host/resample_host.mojo`)."""
    validate_positions(n_resamples, n_x)
    validate_positions(n_resamples, n_y)
    if r_first < 0:
        raise Error(
            "bootstrap: r_first must be non-negative; got " + String(r_first)
        )
    validate_positions(r_first + n_resamples, n_x)
    validate_positions(r_first + n_resamples, n_y)
    if len(x) < n_x or len(y) < n_y:
        raise Error("bootstrap: paired=False: a sample holds fewer values than its n")
    for k in range(2):
        var m = n_x if k == 0 else n_y
        for i in range(m):
            var v = x[i] if k == 0 else y[i]
            if v != v or v > Float32(3.4e38) or v < Float32(-3.4e38):
                raise Error(
                    "bootstrap: sample " + String(k) + " contains NaN or"
                    " infinity at position " + String(i) + " (refused before"
                    " any launch, IDENTITY_PATHS row 39 FACT 2)"
                )
    if confidence_level <= Float32(0.0) or confidence_level >= Float32(1.0):
        raise Error(
            "bootstrap: confidence_level must be in (0, 1); got "
            + String(confidence_level)
        )
    if method == METHOD_BCA and (n_x < 2 or n_y < 2):
        raise Error(
            "bootstrap: method='BCa' needs at least 2 observations in each"
            " sample (each leave-one-out sample must itself have a mean); got"
            " n_x=" + String(n_x) + ", n_y=" + String(n_y)
        )


def bootstrap_unpaired_host(
    x: List[Float32],
    n_x: Int,
    y: List[Float32],
    n_y: Int,
    n_resamples: Int,
    seed: UInt64,
    method: Int,
    confidence_level: Float32 = Float32(0.95),
    alternative: Int = ALT_TWO_SIDED,
    r_first: Int = 0,
    tpb: Int = RESAMPLE_TPB,
) raises -> BootstrapResult:
    """`scipy.stats.bootstrap((x, y), diff_means, paired=False, ...)`: the
    two samples resampled INDEPENDENTLY (2026-09-28), each by its own map --
    sample 0 under kind 1 (so its replicate is exactly the one-sample
    bootstrap's), sample 1 under kind 4 -- and `theta[r] = mean(x*_r) -
    mean(y*_r)`, each mean the pinned tree of the one-sample `mean` arm, the
    difference one flushed subtraction (the paired `diff_means` spelling).
    BCa uses SciPy's multi-sample acceleration (`bca_acceleration_two`)
    over each sample's leave-one-out means with the other sample whole."""
    _unpaired_validate(x, n_x, y, n_y, n_resamples, method, confidence_level, r_first)
    var kx = resample_key(seed, RESAMPLE_KIND_BOOTSTRAP)
    var ky = resample_key(seed, RESAMPLE_KIND_BOOTSTRAP_SECOND)
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var dxb = _upload(ctx, x)
    var dyb = _upload(ctx, y)
    var tx = ctx.enqueue_create_buffer[DType.float32](n_resamples)
    var ty = ctx.enqueue_create_buffer[DType.float32](n_resamples)
    ctx.synchronize()
    _launch_bootstrap_stat(ctx, tx, dxb, kx, r_first, n_resamples, n_x, n_x, 1, STAT_MEAN, tpb)
    _launch_bootstrap_stat(ctx, ty, dyb, ky, r_first, n_resamples, n_y, n_y, 1, STAT_MEAN, tpb)
    ctx.synchronize()
    var hx = _download_f32(ctx, tx, n_resamples)
    var hy = _download_f32(ctx, ty, n_resamples)
    var dist = List[Float32](capacity=n_resamples)
    for i in range(n_resamples):
        dist.append(canonicalize_nan(ftz(hx[i] - hy[i])))
    var theta = _upload(ctx, dist)
    var sorted_buf = ctx.enqueue_create_buffer[DType.float32](n_resamples)
    ctx.synchronize()
    # MOJOLEARN_RESAMPLE_FAST_RANK_SORT=1: see bootstrap_host.
    var ranked = False
    comptime if RESAMPLE_FAST_APPLE:
        if _fast_rank_sort_on():
            rank_sort_f32(ctx, theta, sorted_buf, n_resamples)
            ranked = True
    if not ranked:
        _sort_segments(ctx, theta, sorted_buf, 1, n_resamples)
    var sorted_dist = _download_f32(ctx, sorted_buf, n_resamples)

    var mx = point_estimate_host(x, n_x, 1, STAT_MEAN, Float32(0.5))
    var my = point_estimate_host(y, n_y, 1, STAT_MEAN, Float32(0.5))
    var theta_hat = ftz(mx - my)
    var alpha = alpha_for(confidence_level, alternative)
    var interval: Interval
    var lvl_lo = alpha
    var lvl_hi = ftz(Float32(1.0) - alpha)
    if method == METHOD_BASIC:
        interval = basic_interval(sorted_dist, n_resamples, alpha, theta_hat)
    elif method == METHOD_BCA:
        var z0p = bca_bias_percentile(sorted_dist, n_resamples, theta_hat)
        var jx = ctx.enqueue_create_buffer[DType.float32](n_x)
        var jy = ctx.enqueue_create_buffer[DType.float32](n_y)
        ctx.synchronize()
        if tpb == 256:
            _launch_jackknife_at[256](ctx, jx, dxb, n_x, 1, STAT_MEAN)
            _launch_jackknife_at[256](ctx, jy, dyb, n_y, 1, STAT_MEAN)
        elif tpb == 128:
            _launch_jackknife_at[128](ctx, jx, dxb, n_x, 1, STAT_MEAN)
            _launch_jackknife_at[128](ctx, jy, dyb, n_y, 1, STAT_MEAN)
        else:
            _launch_jackknife_at[64](ctx, jx, dxb, n_x, 1, STAT_MEAN)
            _launch_jackknife_at[64](ctx, jy, dyb, n_y, 1, STAT_MEAN)
        ctx.synchronize()
        var jxh = _download_f32(ctx, jx, n_x)
        var jyh = _download_f32(ctx, jy, n_y)
        var j0 = List[Float32](capacity=n_x)
        for i in range(n_x):
            j0.append(ftz(jxh[i] - my))
        var j1 = List[Float32](capacity=n_y)
        for i in range(n_y):
            j1.append(ftz(mx - jyh[i]))
        var ends = bca_interval(
            sorted_dist, n_resamples, alpha, z0p, bca_acceleration_two(j0, n_x, j1, n_y)
        )
        interval = ends.interval
        lvl_lo = ends.alpha_1
        lvl_hi = ends.alpha_2
        _ = jx^
        _ = jy^
    else:
        interval = percentile_interval(sorted_dist, n_resamples, alpha)
    interval = narrow_for_alternative(interval, alternative)
    var pos_lo = Int(Float32(n_resamples - 1) * lvl_lo)
    var pos_hi = Int(Float32(n_resamples - 1) * lvl_hi)
    var se = distribution_standard_error(dist, n_resamples)
    _ = dxb^
    _ = dyb^
    _ = tx^
    _ = ty^
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
    implemented -- see `resample/NOT_IMPLEMENTED.tsv`).

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
    # lane/apple-fast-resample (2026-10-02), MOJOLEARN_RESAMPLE_FAST_PERM_SELECT=1:
    # `perm_stat_kernel` ranks the pooled positions by counting a total
    # order (O(N^2) per replicate, 8 N bytes of threadgroup memory), so
    # `validate_pooled` refuses N > PERM_MAX_POOLED = 1024 and the board's
    # 20,000 + 20,000 row is REFUSED. The radix select (fast_apple.mojo)
    # finds the same n_x smallest keys with no bound.
    var fast_perm = False
    comptime if RESAMPLE_FAST_APPLE:
        fast_perm = (
            (statistic == STAT_MEAN or statistic == STAT_DIFF_MEANS)
            and tpb == 256
            and _fast_perm_select_on()
        )
    if not fast_perm:
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

    var ctx = process_ctx[_DEVCTX_SLOT]()
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
    var owners = resample_device_count()
    var selected = False
    comptime if RESAMPLE_FAST_APPLE:
        if fast_perm:
            perm_select_fast(
                ctx, null_buf, dpool, key, r_first, n_resamples, n_pooled, n_x,
                statistic,
            )
            selected = True
    if selected:
        pass
    elif owners > 1 and n_resamples > 1:
        _perm_null_owners(
            ctx, null_buf, pooled, key, r_first, n_resamples, n_pooled, n_x,
            statistic, tpb, owners,
        )
    else:
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


def perm_samples_validate(
    x: List[Float32], y: List[Float32], two: Bool, n_resamples: Int, r_first: Int
) raises:
    """`permutation_samples_host`'s refusals (shared word for word by the
    host twin)."""
    var n = len(x)
    if n <= 0:
        raise Error("permutation_test(permutation_type='samples'): the sample is empty")
    if two and len(y) != n:
        raise Error(
            "permutation_test(permutation_type='samples'): the two samples"
            " must be PAIRED, the same length; got n_x=" + String(n)
            + ", n_y=" + String(len(y))
        )
    validate_positions(n_resamples, n)
    if r_first < 0:
        raise Error("permutation_test: r_first must be non-negative; got " + String(r_first))
    validate_positions(r_first + n_resamples, n)
    for k in range(2 if two else 1):
        for i in range(n):
            var v = x[i] if k == 0 else y[i]
            if v != v or v > Float32(3.4e38) or v < Float32(-3.4e38):
                raise Error(
                    "permutation_test: sample " + String(k) + " contains NaN"
                    " or infinity at position " + String(i) + " (refused"
                    " before any launch, IDENTITY_PATHS row 39 FACT 2)"
                )


def permutation_samples_host(
    x: List[Float32],
    y: List[Float32],
    two: Bool,
    n_resamples: Int,
    seed: UInt64,
    alternative: Int,
    r_first: Int = 0,
    tpb: Int = RESAMPLE_TPB,
) raises -> PermutationResult:
    """`scipy.stats.permutation_test(data, statistic,
    permutation_type='samples', ...)` (2026-09-28): `two` is `(x, y)` with
    `diff_means`, else `(x,)` with `mean` (SciPy's sign-flip convention).
    `perm_samples_kernel` draws each pair's coin at its own Philox position
    (kind 5); the observed statistic is the identity arrangement, the pinned
    tree on the host; the p-value is `permutation_pvalue` (DEVIATION 1702)."""
    perm_samples_validate(x, y, two, n_resamples, r_first)
    var n = len(x)
    var key = resample_key(seed, RESAMPLE_KIND_PERM_SAMPLES)
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var dx = _upload(ctx, x)
    var dy = _upload(ctx, x)
    if two:
        dy = _upload(ctx, y)
    var null_buf = ctx.enqueue_create_buffer[DType.float32](n_resamples)
    ctx.synchronize()
    if tpb != 256 and tpb != 128 and tpb != 64:
        raise Error("permutation_test: threads-per-block must be 64, 128 or 256; got " + String(tpb))
    comptime for t in range(3):
        comptime width = 256 if t == 0 else (128 if t == 1 else 64)
        if tpb == width:
            if two:
                comptime k2 = perm_samples_kernel[True, width]
                ctx.enqueue_function[k2](
                    null_buf.unsafe_ptr(), dx.unsafe_ptr(), dy.unsafe_ptr(), key_lo(key), key_hi(key),
                    Int32(r_first), Int32(n_resamples), Int32(n),
                    grid_dim=(n_resamples, 1, 1), block_dim=(width, 1, 1),
                )
            else:
                comptime k1 = perm_samples_kernel[False, width]
                ctx.enqueue_function[k1](
                    null_buf.unsafe_ptr(), dx.unsafe_ptr(), dy.unsafe_ptr(), key_lo(key), key_hi(key),
                    Int32(r_first), Int32(n_resamples), Int32(n),
                    grid_dim=(n_resamples, 1, 1), block_dim=(width, 1, 1),
                )
    ctx.synchronize()
    var null_dist = _download_f32(ctx, null_buf, n_resamples)
    var vx = List[Float32](capacity=n)
    for i in range(n):
        vx.append(ftz(x[i]))
    var observed = _mean_of_sum(host_tree_sum(vx, n), n)
    if two:
        var vy = List[Float32](capacity=n)
        for i in range(n):
            vy.append(ftz(y[i]))
        observed = ftz(observed - _mean_of_sum(host_tree_sum(vy, n), n))
    var pv = permutation_pvalue(null_dist, n_resamples, observed, alternative)
    _ = dx^
    _ = dy^
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

    var ctx = process_ctx[_DEVCTX_SLOT]()
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
    var owners = resample_device_count()
    if owners > 1 and n_chunks > 1:
        _mc_partials_owners[f_id](
            ctx, partials, lower, span, key, i_first, n_samples, n_chunks, tpb,
            owners,
        )
    else:
        _mc_partials[f_id](
            ctx, partials, dlower, dspan, key, i_first, n_samples, n_chunks,
            blocks, tpb,
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


# ===========================================================================
# ENTRY POINT: sklearn.utils.resample's row indices (2026-09-28)
# ===========================================================================


def resample_indices_host(
    n: Int, count: Int, replace: Bool, seed: UInt64, tpb: Int = 256
) raises -> List[Int32]:
    """The `count` row indices `sklearn.utils.resample(..., replace=...,
    n_samples=count, random_state=seed)` gathers: replace=True position `i`
    is `draw_row_index(key6, 0, i, n)` on the device; replace=False the
    device draws every position's 64-bit key (kind 7) and the host keeps the
    first `count` positions of the total order (`utils_first_by_key`)."""
    utils_validate(n, count, replace)
    var key = resample_key(
        seed, RESAMPLE_KIND_UTILS_REPLACE if replace else RESAMPLE_KIND_UTILS_PERMUTE
    )
    var m = count if replace else n
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var rows = ctx.enqueue_create_buffer[DType.int32](m if replace else 1)
    var keys = ctx.enqueue_create_buffer[DType.uint64](1 if replace else m)
    ctx.enqueue_function[utils_draw_kernel](
        rows.unsafe_ptr(), keys.unsafe_ptr(), key_lo(key), key_hi(key),
        Int32(n), Int32(m), Int32(1) if replace else Int32(0),
        grid_dim=(ceildiv(m, tpb), 1, 1), block_dim=(tpb, 1, 1),
    )
    ctx.synchronize()
    var out: List[Int32]
    if replace:
        out = _download_i32(ctx, rows, m)
    else:
        var host = ctx.enqueue_create_host_buffer[DType.uint64](m)
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=keys)
        ctx.synchronize()
        var kl = List[UInt64](capacity=m)
        for q in range(m):
            kl.append(host.unsafe_ptr().unsafe_load(q))
        _ = host^
        out = utils_first_by_key(kl, n, count)
    _ = rows^
    _ = keys^
    # DEVIATION 1946: the context dies LAST.
    _ = ctx^
    return out^


# ===========================================================================
# FAST + Apple: sklearn.utils.resample on the device end to end
# (lane/apple-fast-resample, 2026-10-02)
# ===========================================================================


def _draw_rows_device(
    ctx: DeviceContext, n: Int, count: Int, seed: UInt64, tpb: Int
) raises -> DeviceBuffer[DType.int32]:
    """`resample_indices_host`'s replace=True draw, left on the device."""
    var key = resample_key(seed, RESAMPLE_KIND_UTILS_REPLACE)
    var rows = ctx.enqueue_create_buffer[DType.int32](count)
    var keys = ctx.enqueue_create_buffer[DType.uint64](1)
    ctx.enqueue_function[utils_draw_kernel](
        rows.unsafe_ptr(), keys.unsafe_ptr(), key_lo(key), key_hi(key),
        Int32(n), Int32(count), Int32(1),
        grid_dim=(ceildiv(count, tpb), 1, 1), block_dim=(tpb, 1, 1),
    )
    _ = rows.unsafe_ptr()
    _ = keys.unsafe_ptr()
    ctx.synchronize()
    _ = keys^
    return rows^


def _copy_rows_out(
    ctx: DeviceContext,
    mut rows: DeviceBuffer[DType.int32],
    count: Int,
    out: MutPointer[Int32, MutUntrackedOrigin],
) raises:
    """The drawn indices to the caller's int32 buffer in ONE device-to-host
    copy and one tight store loop (the old path appends them one by one
    into a List and the binding stores them one by one again)."""
    var host = ctx.enqueue_create_host_buffer[DType.int32](count)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=rows)
    ctx.synchronize()
    var hp = host.unsafe_ptr()
    for i in range(count):
        out.unsafe_store(i, hp.unsafe_load(i))
    _ = host^


def resample_indices_fast_into(
    n: Int,
    count: Int,
    replace: Bool,
    seed: UInt64,
    out: MutPointer[Int32, MutUntrackedOrigin],
    tpb: Int = 256,
) raises -> Bool:
    """MOJOLEARN_RESAMPLE_FAST_IDX_BULK=1 (FAST + Apple, replace=True only):
    `resample_indices_host` with the indices copied out in bulk instead of
    `_download_i32`'s per-element List append plus the binding's
    per-element store (two host loops over n_samples = 1,000,000 on the
    board). Same draws, same bytes. False: the caller takes the old path."""
    comptime if RESAMPLE_FAST_APPLE:
        if not replace or String(getenv("MOJOLEARN_RESAMPLE_FAST_IDX_BULK")) != "1":
            return False
        utils_validate(n, count, replace)
        var ctx = process_ctx[_DEVCTX_SLOT]()
        var rows = _draw_rows_device(ctx, n, count, seed, tpb)
        _copy_rows_out(ctx, rows, count, out)
        _ = rows^
        # DEVIATION 1946: the context dies LAST.
        _ = ctx^
        return True
    else:
        return False


def resample_gather_fast_host(
    n: Int,
    count: Int,
    replace: Bool,
    seed: UInt64,
    idx_out: Int,
    srcs: List[Int],
    dsts: List[Int],
    widths: List[Int],
    tpb: Int = 256,
) raises -> Bool:
    """MOJOLEARN_RESAMPLE_FAST_GATHER=1 (FAST + Apple, replace=True only):
    `sklearn.utils.resample(*arrays)` for float32 row-major arrays with the
    draw AND the gathers on the device. `srcs[a]` / `dsts[a]` are the host
    addresses of array `a` (`n x widths[a]` in, `count x widths[a]` out);
    `idx_out` receives the `count` drawn indices (the same ones
    `resample_indices_host` returns). Each array crosses twice (one host
    copy into a pinned stage, one copy back) and the rows are gathered by
    `gather_rows_f32_kernel`, instead of the host's per-element index
    loops and numpy's fancy-index gather of 1,000,000 rows (the timed part
    of the board's resample row). False: the caller takes the old path."""
    comptime if RESAMPLE_FAST_APPLE:
        if not replace or String(getenv("MOJOLEARN_RESAMPLE_FAST_GATHER")) != "1":
            return False
        if len(srcs) != len(dsts) or len(srcs) != len(widths) or len(srcs) == 0:
            raise Error("resample: the gather takes one (src, dst, width) per array")
        utils_validate(n, count, replace)
        var ctx = process_ctx[_DEVCTX_SLOT]()
        var rows = _draw_rows_device(ctx, n, count, seed, tpb)
        for a in range(len(srcs)):
            var d = widths[a]
            if d <= 0:
                raise Error("resample: every gathered array needs a positive row width")
            var src_ptr = f32_ptr(srcs[a])
            var dst_ptr = f32_ptr(dsts[a])
            var hsrc = ctx.enqueue_create_host_buffer[DType.float32](n * d)
            var dsrc = ctx.enqueue_create_buffer[DType.float32](n * d)
            var dout = ctx.enqueue_create_buffer[DType.float32](count * d)
            var hout = ctx.enqueue_create_host_buffer[DType.float32](count * d)
            ctx.synchronize()
            copy_f32(src_ptr, hsrc.unsafe_ptr(), n * d)
            ctx.enqueue_copy(dst_buf=dsrc, src_ptr=hsrc.unsafe_ptr())
            gather_rows_f32(ctx, dout, dsrc, rows, count, d)
            ctx.enqueue_copy(dst_ptr=hout.unsafe_ptr(), src_buf=dout)
            ctx.synchronize()
            copy_f32(hout.unsafe_ptr(), dst_ptr, count * d)
            _ = hsrc^
            _ = dsrc^
            _ = dout^
            _ = hout^
        _copy_rows_out(ctx, rows, count, i32_ptr(idx_out))
        _ = rows^
        # DEVIATION 1946: the context dies LAST.
        _ = ctx^
        return True
    else:
        return False
