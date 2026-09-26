# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into a CPU host binding (python/mojolearn/host_surface.py names which); product, not only a check.
"""`bootstrap`, `permutation_test` and `monte_carlo_integrate` on the host,
for a box with no GPU (the bootstrap, permutation-test and monte-carlo lanes
of tools/identity_break.py; lane/cpu-training-misc, 2026-09-15).

HOST ONLY. No DeviceContext, no kernel launch, nothing here imports
`max.gpu` or `max.gpu`. `resample/estimator.mojo` is the device path this
restates: its three entry points validate on the host, launch one block per
replicate (or per sample chunk), sort on the device, download, and finish on
the host. The host half (the validation, the point estimate, the interval,
the standard error, the p-value, the Monte Carlo finish) is ALREADY host
code, and this file calls the same functions it calls, in its order, from
the same modules (`resample/checks/intervals.mojo`, `resample/checks/
statistics.mojo`, `metrics/checks/pinned_sum.mojo`). The device half is
spelled a SECOND time here, from its kernels:

  `host_bootstrap_fold_statistic`
                    `bootstrap_stat_kernel` (`statistics.mojo:496`) through
                    `_chunked_sum` (`:299`): replicate `r` draws row
                    `draw_row_index(key, r, i, n)` at position `i`, and each
                    pass folds `ftz(x[row, col])`, `ftz(identical_mul(d, d))`
                    or `ftz(identical_mul(da, db))` with `d = ftz(v - centre)`
                    over the fixed 256-wide slab tree, chunk totals folded
                    ascending from `+0.0`. The centre is thread 0's mean
                    (`_block_broadcast`), the same float. The draws are
                    computed once per replicate and reused by every pass;
                    the kernel recomputes them, and a pure function of
                    `(key, r, i)` gives the same row either way.
  `host_bootstrap_order_statistic`
                    `materialize_resample_kernel` (`:597`), the device
                    segmented sort (`core/segmented_sort.mojo`, the LSD radix
                    over `float_to_sortable` keys) and `order_stat_kernel`
                    (`:631`): the quantile's three lines, or the trimmed
                    mean's cut and the same tree over the kept values.
  `host_sorted_by_key`
                    `_sort_segments` (`estimator.mojo:629`): ascending by the
                    twiddled key. Two values with one key are the same bits,
                    so any sort by the key returns the radix sort's bits.
  `host_permutation_statistic`
                    `perm_stat_kernel` (`:707`): the 64-bit keys
                    `draw_permutation_key(key, r, j)`, every position's rank
                    under the total order `(key, position)`
                    (`permutation_key_lt`; the kernel COUNTS, this sorts the
                    keys and counts only inside a tie, the same rank), then
                    the membership-masked trees over the original positions.
  `host_mc_partials`
                    `monte_carlo_chunk_kernel` (`:960`): chunk `c` holds
                    `mc_integrand` at samples `[c*256, (c+1)*256)`, one tree
                    total per chunk.

THE NEGATIVE CONTROL. Under `-D MOJOLEARN_HOST_SABOTAGE=1`
(`RESAMPLE_HOST_SABOTAGE`) every per-replicate and per-chunk tree takes its
chunk boundaries shifted by one value (the first value wraps to the end),
what a partition that is not a pure function of `n` would fold. The point
estimate, the standard error and the p-value keep the pinned host tree, so
the moved bits are the distributions', which every lane hashes. The `const`
integrand folds exact integers and does not move.
"""
from std.memory import bitcast
from std.sys.compile import is_defined
from max.algorithm import sync_parallelize

from checks.numerics import ftz, identical_div, identical_mul, identical_sqrt
from core.segmented_sort import float_to_sortable
from core.host_predict_threads import host_predict_chunk, host_predict_task_count
from metrics.checks.pinned_sum import (
    PINNED_SUM_W,
    canonicalize_nan,
    chunk_count,
    host_fold_partials,
    host_tree_sum,
)
from resample.checks.index_map import (
    RESAMPLE_KIND_BOOTSTRAP,
    RESAMPLE_KIND_MONTE_CARLO,
    RESAMPLE_KIND_PERMUTATION,
    draw_permutation_key,
    draw_row_index,
    draw_uniform_in,
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
    bca_refuse,
    distribution_standard_error,
    narrow_for_alternative,
    percentile_interval,
    permutation_pvalue,
)
from resample.checks.statistics import (
    MC_DIMS,
    MC_F_CONST,
    MC_F_SUM,
    RESAMPLE_MAX_SORT_CELLS,
    STAT_DIFF_MEANS,
    STAT_MEAN,
    STAT_PEARSON,
    STAT_QUANTILE,
    STAT_STD,
    STAT_TRIMMED_MEAN,
    _mean_of_sum,
    mc_box_volume,
    mc_finish_host,
    quantile_of_sorted_host,
    stat_columns_needed,
    stat_name,
    stat_needs_sort,
    trim_count,
)


#: The gate's negative control (see THE NEGATIVE CONTROL above).
comptime RESAMPLE_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()


# ===========================================================================
# Results, the device entry points' structs without the device
# ===========================================================================


@fieldwise_init
struct HostBootstrapResult(Movable):
    """`estimator.mojo::BootstrapResult`."""

    var point_estimate: Float32
    var distribution: List[Float32]
    var sorted_distribution: List[Float32]
    var standard_error: Float32
    var interval: Interval
    var order_low: Int
    var order_high: Int


@fieldwise_init
struct HostPermutationResult(Movable):
    """`estimator.mojo::PermutationResult`."""

    var observed: Float32
    var null_distribution: List[Float32]
    var pvalue: PValue


@fieldwise_init
struct HostMonteCarloResult(ImplicitlyCopyable, Movable):
    """`estimator.mojo::MonteCarloResult`."""

    var integral: Float32
    var mean: Float32
    var volume: Float32


# ===========================================================================
# The kernel tree
# ===========================================================================


def host_chunk_partials(values: List[Float32], n: Int) -> List[Float32]:
    """One `virtual_block_sum` total per `PINNED_SUM_W` chunk of
    `values[0:n]`: slot `t` of chunk `c` holds `ftz(values[c*W + t])`
    (`+0.0` past `n`), the halving tree `slab[t] = ftz(slab[t] +
    slab[t + step])` (`pinned_sum.mojo:108-135`)."""
    var partials = List[Float32]()
    var slab = List[Float32](length=PINNED_SUM_W, fill=Float32(0.0))
    for c in range(chunk_count(n)):
        for t in range(PINNED_SUM_W):
            var i = c * PINNED_SUM_W + t
            comptime if RESAMPLE_HOST_SABOTAGE:
                # THE SABOTAGE ARM: chunk boundaries shifted by one value.
                slab[t] = ftz(values[(i + 1) % n]) if i < n else Float32(0.0)
            else:
                slab[t] = ftz(values[i]) if i < n else Float32(0.0)
        var step = PINNED_SUM_W // 2
        while step > 0:
            for t in range(step):
                slab[t] = ftz(slab[t] + slab[t + step])
            step //= 2
        partials.append(slab[0])
    return partials^


def host_kernel_fold(values: List[Float32], n: Int) -> Float32:
    """A kernel's whole fold: the chunk totals, then thread 0's ascending
    chain from `+0.0` through `ftz` (`_chunked_sum`'s `acc`)."""
    var partials = host_chunk_partials(values, n)
    return host_fold_partials(partials, chunk_count(n))


def host_sorted_by_key(values: List[Float32], base: Int, m: Int) -> List[Float32]:
    """`values[base : base + m]` ascending by `float_to_sortable`, the
    device radix sort's key (`core/segmented_sort.mojo:101`). The key is a
    bijection of the bits, so the sorted keys invert to the sorted values."""
    var keys = List[UInt32](capacity=m)
    for i in range(m):
        keys.append(float_to_sortable(bitcast[DType.uint32](values[base + i])))
    sort(keys)
    var out = List[Float32](capacity=m)
    for i in range(m):
        var k = keys[i]
        var bits: UInt32
        if (k & UInt32(0x80000000)) != UInt32(0):
            bits = k & UInt32(0x7FFFFFFF)
        else:
            bits = ~k
        out.append(bitcast[DType.float32](bits))
    return out^


# ===========================================================================
# The bootstrap replicate
# ===========================================================================


def _replicate_column(
    x: List[Float32], rows: List[Int], n: Int, n_features: Int, col: Int
) -> List[Float32]:
    var out = List[Float32](capacity=n)
    for i in range(n):
        out.append(ftz(x[rows[i] * n_features + col]))
    return out^


def _replicate_rows(key: UInt64, r: Int, n: Int) -> List[Int]:
    var rows = List[Int](capacity=n)
    for i in range(n):
        rows.append(Int(draw_row_index(key, r, i, Int32(n))))
    return rows^


def host_bootstrap_fold_statistic(
    x: List[Float32], n: Int, n_features: Int, key: UInt64, r: Int, stat: Int
) -> Float32:
    """`bootstrap_stat_kernel[stat]`'s `theta[r]` for mean, std, pearson and
    diff_means."""
    var rows = _replicate_rows(key, r, n)
    var a = _replicate_column(x, rows, n, n_features, 0)
    var value = Float32(0.0)
    if stat == STAT_MEAN:
        value = _mean_of_sum(host_kernel_fold(a, n), n)
    elif stat == STAT_DIFF_MEANS:
        var b = _replicate_column(x, rows, n, n_features, 1)
        value = ftz(
            _mean_of_sum(host_kernel_fold(a, n), n)
            - _mean_of_sum(host_kernel_fold(b, n), n)
        )
    elif stat == STAT_STD:
        var m = _mean_of_sum(host_kernel_fold(a, n), n)
        var sq = List[Float32](capacity=n)
        for i in range(n):
            var d = ftz(a[i] - m)
            sq.append(ftz(identical_mul(d, d)))
        value = ftz(identical_sqrt(ftz(identical_div(host_kernel_fold(sq, n), Float32(n - 1)))))
    else:
        var b2 = _replicate_column(x, rows, n, n_features, 1)
        var mx = _mean_of_sum(host_kernel_fold(a, n), n)
        var my = _mean_of_sum(host_kernel_fold(b2, n), n)
        var cxy = List[Float32](capacity=n)
        var cxx = List[Float32](capacity=n)
        var cyy = List[Float32](capacity=n)
        for i in range(n):
            var dx = ftz(a[i] - mx)
            var dy = ftz(b2[i] - my)
            cxy.append(ftz(identical_mul(dx, dy)))
            cxx.append(ftz(identical_mul(dx, dx)))
            cyy.append(ftz(identical_mul(dy, dy)))
        var sxy = host_kernel_fold(cxy, n)
        var sxx = host_kernel_fold(cxx, n)
        var syy = host_kernel_fold(cyy, n)
        # DEVIATION 1696: a degenerate resample is the canonical NaN.
        if sxx == Float32(0.0) or syy == Float32(0.0):
            value = canonicalize_nan(Float32(0.0) / Float32(0.0))
        else:
            # DEVIATION 1695: ONE sqrt of the product.
            value = ftz(identical_div(sxy, ftz(identical_sqrt(ftz(identical_mul(sxx, syy))))))
    return canonicalize_nan(value)


def host_bootstrap_order_statistic(
    x: List[Float32], n: Int, n_features: Int, key: UInt64, r: Int, stat: Int,
    q_or_prop: Float32,
) -> Float32:
    """`materialize_resample_kernel` + the segment sort + `order_stat_kernel`
    for replicate `r`."""
    var rows = _replicate_rows(key, r, n)
    var vals = _replicate_column(x, rows, n, n_features, 0)
    var s = host_sorted_by_key(vals, 0, n)
    if stat == STAT_QUANTILE:
        return canonicalize_nan(quantile_of_sorted_host(s, 0, n, q_or_prop))
    var k = trim_count(n, q_or_prop)
    var kept = n - 2 * k
    if kept < 1:
        return canonicalize_nan(Float32(0.0) / Float32(0.0))
    var keptv = List[Float32](capacity=kept)
    for i in range(kept):
        keptv.append(ftz(s[k + i]))
    return canonicalize_nan(_mean_of_sum(host_kernel_fold(keptv, kept), kept))


def host_point_estimate(
    x: List[Float32], n: Int, n_features: Int, stat: Int, q_or_prop: Float32
) raises -> Float32:
    """`estimator.mojo::point_estimate_host` (`:537`), line for line, with
    `host_sorted_by_key` for its stable insertion sort (the same bits)."""
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

    var s = host_sorted_by_key(a, 0, n)
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


def host_bootstrap(
    x: List[Float32],
    n: Int,
    n_features: Int,
    statistic: Int,
    n_resamples: Int,
    seed: UInt64,
    method: Int,
    confidence_level: Float32,
    alternative: Int,
    q_or_prop: Float32,
    r_first: Int,
) raises -> HostBootstrapResult:
    """`estimator.mojo::bootstrap_host` (`:951`) without the device: the
    same refusals in the same order and words, the replicates restated
    above, then the host finish it runs."""
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
    elif statistic != STAT_MEAN and statistic != STAT_STD and statistic != STAT_PEARSON and statistic != STAT_DIFF_MEANS:
        # `_launch_bootstrap_stat_at`'s last arm (`estimator.mojo:298`).
        raise Error(
            "resample: statistic '"
            + stat_name(statistic)
            + "' has no fold arm; the order statistics go through the sort"
            " path (bootstrap_order_statistic)"
        )

    var key = resample_key(seed, RESAMPLE_KIND_BOOTSTRAP)
    var dist = List[Float32](length=n_resamples, fill=Float32(0.0))
    var dp = dist.unsafe_ptr()
    var tasks = host_predict_task_count(n_resamples)
    var chunk = host_predict_chunk(n_resamples, tasks)
    var needs_sort = stat_needs_sort(statistic)
    # A replicate reads only the sample and its global replicate id, and owns
    # exactly one output slot.  Splitting contiguous replicate ranges changes
    # no statement inside a replicate; MOJOLEARN_CPU_THREADS=1 is the serial
    # form of this same walk.

    def _replicates(c: Int) {imm x, imm dp, imm chunk, imm n_resamples, imm n, imm n_features, imm key, imm r_first, imm statistic, imm q_or_prop, imm needs_sort}:
        var lo = c * chunk
        var hi = min(lo + chunk, n_resamples)
        for rr in range(lo, hi):
            if needs_sort:
                dp.unsafe_store(rr, host_bootstrap_order_statistic(
                    x, n, n_features, key, r_first + rr, statistic, q_or_prop
                ))
            else:
                dp.unsafe_store(rr, host_bootstrap_fold_statistic(
                    x, n, n_features, key, r_first + rr, statistic
                ))

    if tasks == 1:
        _replicates(0)
    else:
        sync_parallelize(_replicates, tasks)
    var sorted_dist = host_sorted_by_key(dist, 0, n_resamples)

    var theta_hat = host_point_estimate(x, n, n_features, statistic, q_or_prop)
    var alpha = alpha_for(confidence_level, alternative)
    var interval: Interval
    if method == METHOD_BASIC:
        interval = basic_interval(sorted_dist, n_resamples, alpha, theta_hat)
    else:
        interval = percentile_interval(sorted_dist, n_resamples, alpha)
    interval = narrow_for_alternative(interval, alternative)
    var h_lo = Float32(n_resamples - 1) * alpha
    var pos_lo = Int(h_lo)
    var h_hi = Float32(n_resamples - 1) * ftz(Float32(1.0) - alpha)
    var pos_hi = Int(h_hi)
    var se = distribution_standard_error(dist, n_resamples)
    return HostBootstrapResult(
        theta_hat, dist^, sorted_dist^, se, interval, pos_lo, pos_hi
    )


# ===========================================================================
# The permutation replicate
# ===========================================================================


def host_permutation_ranks(key: UInt64, r: Int, n_pooled: Int) -> List[Int]:
    """Every pooled position's rank under replicate `r`'s total order
    `(draw_permutation_key(key, r, j), j)`, the count `perm_stat_kernel`'s
    phase 2 takes: the number of positions strictly below in that order."""
    var keys = List[UInt64](capacity=n_pooled)
    for j in range(n_pooled):
        keys.append(draw_permutation_key(key, r, j))
    var sorted_keys = keys.copy()
    sort(sorted_keys)
    var ranks = List[Int](capacity=n_pooled)
    for j in range(n_pooled):
        var kj = keys[j]
        # lower bound of kj: how many keys are strictly below it
        var lo = 0
        var hi = n_pooled
        while lo < hi:
            var mid = (lo + hi) // 2
            if sorted_keys[mid] < kj:
                lo = mid + 1
            else:
                hi = mid
        var rank = lo
        # a tie is broken by position (`permutation_key_lt`)
        if lo + 1 < n_pooled and sorted_keys[lo + 1] == kj:
            for l in range(j):
                if keys[l] == kj:
                    rank += 1
        ranks.append(rank)
    return ranks^


def host_permutation_statistic(
    pooled: List[Float32], key: UInt64, r: Int, n_pooled: Int, n_x: Int, stat: Int
) -> Float32:
    """`perm_stat_kernel[stat]`'s `null_dist[r]`."""
    var ranks = host_permutation_ranks(key, r, n_pooled)
    var n_y = n_pooled - n_x
    var vx = List[Float32](capacity=n_pooled)
    var vy = List[Float32](capacity=n_pooled)
    for i in range(n_pooled):
        var v = ftz(pooled[i])
        if ranks[i] < n_x:
            vx.append(v)
            vy.append(Float32(0.0))
        else:
            vx.append(Float32(0.0))
            vy.append(v)
    var sum_x = host_kernel_fold(vx, n_pooled)
    var sum_y = host_kernel_fold(vy, n_pooled)
    var value = Float32(0.0)
    if stat == STAT_DIFF_MEANS:
        value = ftz(_mean_of_sum(sum_x, n_x) - _mean_of_sum(sum_y, n_y))
    elif stat == STAT_MEAN:
        value = _mean_of_sum(sum_x, n_x)
    else:
        var mx = _mean_of_sum(sum_x, n_x)
        var vs = List[Float32](capacity=n_pooled)
        for i in range(n_pooled):
            if ranks[i] < n_x:
                var dv = ftz(ftz(pooled[i]) - mx)
                vs.append(ftz(identical_mul(dv, dv)))
            else:
                vs.append(Float32(0.0))
        value = ftz(identical_sqrt(ftz(identical_div(host_kernel_fold(vs, n_pooled), Float32(n_x - 1)))))
    return canonicalize_nan(value)


def host_permutation_test(
    x: List[Float32],
    y: List[Float32],
    statistic: Int,
    n_resamples: Int,
    seed: UInt64,
    alternative: Int,
    r_first: Int,
) raises -> HostPermutationResult:
    """`estimator.mojo::permutation_test_host` (`:1191`) without the device."""
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
    if statistic != STAT_DIFF_MEANS and statistic != STAT_MEAN and statistic != STAT_STD:
        # `_launch_perm_stat_at`'s last arm (`estimator.mojo:405`).
        raise Error(
            "permutation_test: statistic '"
            + stat_name(statistic)
            + "' is NOT IMPLEMENTED for the two-sample independent case. The"
            " implemented arms are mean, std and diff_means. An order statistic"
            " would need a per-replicate sort of the permuted group (the"
            " bootstrap's sort path does not apply, because the group"
            " membership changes every replicate); pearson is SciPy's"
            " permutation_type='pairings', a different null, not implemented."
        )

    var key = resample_key(seed, RESAMPLE_KIND_PERMUTATION)
    var null_dist = List[Float32](length=n_resamples, fill=Float32(0.0))
    var np = null_dist.unsafe_ptr()
    var tasks = host_predict_task_count(n_resamples)
    var chunk = host_predict_chunk(n_resamples, tasks)
    # As above, the total order and both pinned folds remain wholly inside one
    # replicate.  Only independent output slots are assigned to workers.

    def _replicates(c: Int) {imm pooled, imm np, imm chunk, imm n_resamples, imm key, imm r_first, imm n_pooled, imm n_x, imm statistic}:
        var lo = c * chunk
        var hi = min(lo + chunk, n_resamples)
        for rr in range(lo, hi):
            np.unsafe_store(rr, host_permutation_statistic(
                pooled, key, r_first + rr, n_pooled, n_x, statistic
            ))

    if tasks == 1:
        _replicates(0)
    else:
        sync_parallelize(_replicates, tasks)

    # The observed statistic, `estimator.mojo:1299-1333`.
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
    var pv = permutation_pvalue(null_dist, n_resamples, observed, alternative)
    return HostPermutationResult(observed, null_dist^, pv)


# ===========================================================================
# Monte Carlo
# ===========================================================================


def host_mc_partials[
    f_id: Int
](
    lower: List[Float32], span: List[Float32], key: UInt64, i_first: Int, n_samples: Int
) -> List[Float32]:
    """`monte_carlo_chunk_kernel[f_id]`'s `partials`: `mc_integrand` at
    samples `[c*256, (c+1)*256)`, one tree per chunk."""
    var vals = List[Float32](capacity=n_samples)
    for i in range(n_samples):
        comptime if f_id == MC_F_CONST:
            vals.append(Float32(1.0))
        else:
            var x0 = draw_uniform_in(key, i_first + i, 0, lower[0], span[0])
            var x1 = draw_uniform_in(key, i_first + i, 1, lower[1], span[1])
            comptime if f_id == MC_F_SUM:
                vals.append(ftz(x0 + x1))
            else:
                vals.append(ftz(identical_mul(x0, x1)))
    var partials = host_chunk_partials(vals, n_samples)
    for c in range(len(partials)):
        partials[c] = ftz(partials[c])
    return partials^


def host_monte_carlo_integrate[
    f_id: Int
](
    lower: List[Float32],
    upper: List[Float32],
    n_samples: Int,
    seed: UInt64,
    i_first: Int,
) raises -> HostMonteCarloResult:
    """`estimator.mojo::monte_carlo_integrate_host` (`:1382`) without the
    device."""
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
    var key = resample_key(seed, RESAMPLE_KIND_MONTE_CARLO)
    var n_chunks = chunk_count(n_samples)
    var partials = host_mc_partials[f_id](lower, span, key, i_first, n_samples)
    var integral = mc_finish_host(partials, n_chunks, n_samples, volume)
    var mean = _mean_of_sum(host_fold_partials(partials, n_chunks), n_samples)
    return HostMonteCarloResult(integral, mean, volume)
