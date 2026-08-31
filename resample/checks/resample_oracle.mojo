# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Two host references, and they are two different things.

WHAT AN ORACLE IS HERE. PORTING_RULES 0b-ii: "a host reference used to CHECK a
device answer is not a CPU path". Neither of the two below is reachable from
`resample/estimator.mojo`; both exist to be compared against.

  * THE FLOAT32 REPLAY (`oracle_*_f32`). The SAME arithmetic in the SAME
    order as the device, on the host, so the comparison is BIT FOR BIT and
    not a tolerance. It is a SECOND SPELLING of the kernel, not an import of
    it: the kernel writes the tree with `virtual_block_sum` over threads, the
    replay writes it with `host_tree_sum` over a loop, and the two agree only
    if both are the tree they claim to be. `metrics/checks/pinned_sum.mojo`
    is the shared definition of what that tree is, and it is the file both
    read.

  * THE FLOAT64 REFERENCE (`reference_*_f64`). SciPy's semantics computed
    serially in double precision, ascending, with no pinning of any kind. It
    is what the answer OUGHT to be, to within float32; the gates report the
    gap and never assert a tight band on it. It is allowed to call `std.math`
    -- it is not a certified path and IDENTITY_PATHS row 18's host-libm
    concern does not reach a tolerance report.

NO FLOAT64 EVER TOUCHES THE DEVICE. `HOST_AND_DEVICE.md` and this
repository's hardware ledger: there is no float64 on the Apple column at all,
not as a precision preference but as an absent type. `reference_*_f64` runs on
the host and only on the host.

THE INDEX MAP HAS NO SEPARATE ORACLE, and that is not an omission. It is
integer arithmetic end to end -- multiply-high, xor, add, compare -- so
`index_map.mojo`'s own host mirrors (`bootstrap_index_host`,
`permutation_ranks_host`) ARE the reference, and the comparison against the
device is an exact integer equality rather than a numeric one. The generator
underneath them is already held to an oracle built by COMPILING cuRAND's own
code (`ensemble/bench/philox_oracle.txt`, `ensemble/tools/philox_oracle/`),
which is the strongest reference in this repository and is not duplicated
here.
"""

from std.math import sqrt as f64_sqrt
from std.memory import bitcast

from metrics.checks.pinned_sum import host_tree_sum
from checks.numerics import ftz, identical_div, identical_mul, identical_sqrt
from resample.checks.index_map import (
    draw_row_index,
    draw_uniform_in,
    permutation_ranks_host,
)
from resample.checks.statistics import (
    MC_F_CONST,
    MC_F_SUM,
    STAT_DIFF_MEANS,
    STAT_MEAN,
    STAT_PEARSON,
    STAT_QUANTILE,
    STAT_STD,
    STAT_TRIMMED_MEAN,
    _mean_of_sum,
    host_sort_stable,
    mc_box_volume,
    quantile_of_sorted_host,
    trim_count,
)


# ===========================================================================
# The float32 replay: the device's arithmetic, in the device's order
# ===========================================================================


def oracle_resampled_column(
    x: List[Float32],
    n_rows: Int,
    n_features: Int,
    key: UInt64,
    r: Int,
    n: Int,
    col: Int,
) -> List[Float32]:
    """Replicate `r`'s drawn values of one column, in position order.

    `draw_row_index` is IMPORTED, not re-spelled. The map is the one thing in
    this lane that must have exactly one implementation: a second copy of it
    on the host would let the gate pass with both copies wrong the same way,
    which is the failure a two-spelling oracle exists to prevent EVERYWHERE
    ELSE and cannot prevent here. What the replay is a second spelling of is
    the FOLD, and the fold is below.
    """
    var out = List[Float32]()
    for i in range(n):
        var row = Int(draw_row_index(key, r, i, Int32(n_rows)))
        out.append(ftz(x[row * n_features + col]))
    return out^


def oracle_bootstrap_statistic_f32(
    x: List[Float32],
    n_rows: Int,
    n_features: Int,
    key: UInt64,
    r: Int,
    n: Int,
    stat: Int,
    q_or_prop: Float32,
) raises -> Float32:
    """One replicate's statistic, bit for bit what `bootstrap_stat_kernel` or
    `order_stat_kernel` writes.

    `host_tree_sum` IS the kernel's fold: chunks of `PINNED_SUM_W`, a halving
    tree per chunk, the chunk totals folded ascending through `ftz`. The
    kernel spells that across threads and this spells it across a loop.
    """
    var a = oracle_resampled_column(x, n_rows, n_features, key, r, n, 0)

    if stat == STAT_MEAN:
        return _mean_of_sum(host_tree_sum(a, n), n)

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

    if stat == STAT_DIFF_MEANS:
        var b = oracle_resampled_column(x, n_rows, n_features, key, r, n, 1)
        return ftz(
            _mean_of_sum(host_tree_sum(a, n), n)
            - _mean_of_sum(host_tree_sum(b, n), n)
        )

    if stat == STAT_PEARSON:
        var b2 = oracle_resampled_column(x, n_rows, n_features, key, r, n, 1)
        var mx = _mean_of_sum(host_tree_sum(a, n), n)
        var my = _mean_of_sum(host_tree_sum(b2, n), n)
        var cxy = List[Float32]()
        var cxx = List[Float32]()
        var cyy = List[Float32]()
        for i in range(n):
            var dx = ftz(a[i] - mx)
            var dy = ftz(b2[i] - my)
            cxy.append(ftz(identical_mul(dx, dy)))
            cxx.append(ftz(identical_mul(dx, dx)))
            cyy.append(ftz(identical_mul(dy, dy)))
        var sxy = host_tree_sum(cxy, n)
        var sxx = host_tree_sum(cxx, n)
        var syy = host_tree_sum(cyy, n)
        # DEVIATION 1696, replayed: the canonical NaN, not the host's.
        if sxx == Float32(0.0) or syy == Float32(0.0):
            return bitcast[DType.float32](UInt32(0x7FC00000))
        # DEVIATION 1695, replayed: ONE sqrt of the product.
        return ftz(
            identical_div(sxy, ftz(identical_sqrt(ftz(identical_mul(sxx, syy)))))
        )

    if stat == STAT_QUANTILE:
        var s = host_sort_stable(a, 0, n)
        return quantile_of_sorted_host(s, 0, n, q_or_prop)

    if stat == STAT_TRIMMED_MEAN:
        var s2 = host_sort_stable(a, 0, n)
        var k = trim_count(n, q_or_prop)
        var kept = n - 2 * k
        if kept < 1:
            raise Error(
                "resample oracle: trimmed_mean cut everything (n="
                + String(n)
                + ", proportion leaves "
                + String(kept)
                + " observations)"
            )
        var keptv = List[Float32]()
        for i in range(kept):
            keptv.append(s2[k + i])
        return _mean_of_sum(host_tree_sum(keptv, kept), kept)

    raise Error("resample oracle: unknown statistic id " + String(stat))


def oracle_bootstrap_distribution_f32(
    x: List[Float32],
    n_rows: Int,
    n_features: Int,
    key: UInt64,
    r_first: Int,
    n_replicates: Int,
    n: Int,
    stat: Int,
    q_or_prop: Float32,
) raises -> List[Float32]:
    """The whole distribution, replicate by replicate, in replicate order."""
    var out = List[Float32]()
    for rr in range(n_replicates):
        out.append(
            oracle_bootstrap_statistic_f32(
                x, n_rows, n_features, key, r_first + rr, n, stat, q_or_prop
            )
        )
    return out^


def oracle_permutation_statistic_f32(
    pooled: List[Float32],
    key: UInt64,
    r: Int,
    n_pooled: Int,
    n_x: Int,
    stat: Int,
) raises -> Float32:
    """One null-distribution value, bit for bit what `perm_stat_kernel`
    writes.

    `permutation_ranks_host` is imported from `index_map.mojo` for the same
    reason `draw_row_index` is: the map has exactly one implementation. What
    is replayed here is the MASKED FOLD.
    """
    var ranks = permutation_ranks_host(key, r, n_pooled)
    var n_y = n_pooled - n_x
    var vx = List[Float32]()
    var vy = List[Float32]()
    for j in range(n_pooled):
        var v = ftz(pooled[j])
        if Int(ranks[j]) < n_x:
            vx.append(v)
            vy.append(Float32(0.0))
        else:
            vx.append(Float32(0.0))
            vy.append(v)
    var sx = host_tree_sum(vx, n_pooled)
    var sy = host_tree_sum(vy, n_pooled)

    if stat == STAT_DIFF_MEANS:
        return ftz(_mean_of_sum(sx, n_x) - _mean_of_sum(sy, n_y))
    if stat == STAT_MEAN:
        return _mean_of_sum(sx, n_x)
    if stat == STAT_STD:
        var mx = _mean_of_sum(sx, n_x)
        var sq = List[Float32]()
        for j in range(n_pooled):
            if Int(ranks[j]) < n_x:
                var d = ftz(ftz(pooled[j]) - mx)
                sq.append(ftz(identical_mul(d, d)))
            else:
                sq.append(Float32(0.0))
        return ftz(
            identical_sqrt(
                ftz(identical_div(host_tree_sum(sq, n_pooled), Float32(n_x - 1)))
            )
        )
    raise Error(
        "resample oracle: permutation_test supports mean, std and diff_means;"
        " got statistic id " + String(stat)
    )


def oracle_monte_carlo_f32[f_id: Int](
    key: UInt64,
    i_first: Int,
    n_samples: Int,
    lower: List[Float32],
    upper: List[Float32],
) -> Float32:
    """`volume * mean(f)` replayed on the host with the same tree and the
    same mean-then-scale order as `mc_finish_host`."""
    var span = List[Float32]()
    for d in range(len(lower)):
        span.append(ftz(upper[d] - lower[d]))
    var vals = List[Float32]()
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
    var mean = _mean_of_sum(host_tree_sum(vals, n_samples), n_samples)
    return ftz(identical_mul(mc_box_volume(lower, upper), mean))


# ===========================================================================
# The float64 reference: SciPy's semantics, serial, unpinned
# ===========================================================================


def reference_mean_f64(a: List[Float32], n: Int) -> Float64:
    """`np.mean` -- a plain ascending sum in double, then a divide."""
    var s = Float64(0.0)
    for i in range(n):
        s += Float64(a[i])
    return s / Float64(n)


def reference_std_f64(a: List[Float32], n: Int) -> Float64:
    """`np.std(a, ddof=1)`, two-pass in double."""
    var m = reference_mean_f64(a, n)
    var ssd = Float64(0.0)
    for i in range(n):
        var d = Float64(a[i]) - m
        ssd += d * d
    return f64_sqrt(ssd / Float64(n - 1))


def reference_bootstrap_statistic_f64(
    x: List[Float32],
    n_rows: Int,
    n_features: Int,
    key: UInt64,
    r: Int,
    n: Int,
    stat: Int,
    q_or_prop: Float32,
) raises -> Float64:
    """What the statistic OUGHT to be, to within float32. The drawn indices
    are the same integers on both sides -- the map is integer arithmetic --
    so this isolates the FLOATING-POINT gap and nothing else, which is what
    makes the reported number interpretable."""
    var a = oracle_resampled_column(x, n_rows, n_features, key, r, n, 0)

    if stat == STAT_MEAN:
        return reference_mean_f64(a, n)
    if stat == STAT_STD:
        return reference_std_f64(a, n)
    if stat == STAT_DIFF_MEANS:
        var b = oracle_resampled_column(x, n_rows, n_features, key, r, n, 1)
        return reference_mean_f64(a, n) - reference_mean_f64(b, n)
    if stat == STAT_PEARSON:
        var b2 = oracle_resampled_column(x, n_rows, n_features, key, r, n, 1)
        var mx = reference_mean_f64(a, n)
        var my = reference_mean_f64(b2, n)
        var sxy = Float64(0.0)
        var sxx = Float64(0.0)
        var syy = Float64(0.0)
        for i in range(n):
            var dx = Float64(a[i]) - mx
            var dy = Float64(b2[i]) - my
            sxy += dx * dy
            sxx += dx * dx
            syy += dy * dy
        if sxx == Float64(0.0) or syy == Float64(0.0):
            return Float64(0.0) / Float64(0.0)
        return sxy / f64_sqrt(sxx * syy)
    if stat == STAT_QUANTILE:
        var s = host_sort_stable(a, 0, n)
        if n == 1:
            return Float64(s[0])
        var h = Float64(n - 1) * Float64(q_or_prop)
        var lo = Int(h)
        if lo > n - 2:
            lo = n - 2
        var frac = h - Float64(lo)
        return Float64(s[lo]) + frac * (Float64(s[lo + 1]) - Float64(s[lo]))
    if stat == STAT_TRIMMED_MEAN:
        var s2 = host_sort_stable(a, 0, n)
        var k = trim_count(n, q_or_prop)
        var kept = n - 2 * k
        var acc = Float64(0.0)
        for i in range(kept):
            acc += Float64(s2[k + i])
        return acc / Float64(kept)
    raise Error("resample reference: unknown statistic id " + String(stat))
