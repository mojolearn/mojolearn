"""Host oracles for the KPSS test. NOT A PORT.

Two oracles, written before the kernels were gated and gated first:

- `kpss_host_f32`: the serial Float32 REPLAY of every device stage in
  `tsa/ported/timeSeries/stationarity.mojo`, statement for statement and
  through the same helpers (`ftz`, `identical_mul_add`), with the SAME fold
  shape as DEVIATION 671's pinned kernel: `STATS_TPB` strided partials,
  each serial ascending, then the halving tree `red[t] += red[t + step]`.
  Under IDENTICAL the device must match it BIT FOR BIT; under FAST the
  device folds with the library call and the comparison is a REPORT.

- `kpss_host_f64`: a Float64 reference that spells the paper's formulas
  (mean, centered series, s^2 with the Bartlett weights, the partial sums,
  eta, the statistic, the interpolated p-value) with NO fold shape and no
  helper -- the tolerance sanity check that says the Float32 pipeline is
  computing the right quantity and not merely the same wrong one twice.

`pinned_fold_host` is the host spelling of `core/pinned_reduce.
pinned_block_sum`'s IDENTICAL arm and is gated in the kmeans lane
(`check_pinned_fold_shape`); it is re-spelled here (eight lines) rather
than imported from a lane that owns a different directory.
"""

from core.column_stats import STATS_TPB
from mojo_only.numerics import ftz, identical_mul_add
from tsa.ported.timeSeries.arima_helpers import prepare_data_host
from tsa.ported.timeSeries.stationarity import (
    kpss_lags,
    kpss_pvalue,
    kpss_s2B_coefficients,
    kpss_stat_from_sums,
)


def pinned_fold_host(partials: List[Float32]) -> Float32:
    """The halving tree of `pinned_block_sum`'s IDENTICAL arm:
    `red[t] = red[t] + red[t + step]` for `step = TPB/2 .. 1`.

    EACH STEP IS FLUSHED HERE, AND THAT IS A FINDING (2026-08-23). The
    fixture's 2^-66 series has s2B partials of mixed sign near 2^-125;
    two of them cancel inside the tree to a SUBNORMAL partial, which the M4
    flushes and a gradual-underflow host keeps, and the device and the
    unflushed replay then disagreed at `s2B` cell 3 (0x82b2ab5d vs
    0x82b0b3d3 under IDENTICAL). `core/pinned_reduce.mojo`'s device tree
    does NOT call `ftz` between steps, so on a denormal-honoring backend
    (CUDA, HIP) ITS partial would be kept, not flushed: the Apple bits and
    the NVIDIA bits of `pinned_block_sum` differ whenever a subnormal
    partial arises inside the tree. The oracle models the FTZ column (the
    one this lane can gate); the one-line fix to the shared tree is handed
    off in arima/README.md ("HAND-OFF TO THE IDENTITY LANE") and is not
    made here because core/ is not this lane's."""
    var red = partials.copy()
    var step = len(red) // 2
    while step > 0:
        for t in range(step):
            red[t] = ftz(red[t] + red[t + step])
        step //= 2
    return red[0]


def series_sum_host[
    square: Bool
](data: List[Float32], base: Int, n: Int, scale: Float32) -> Float32:
    """`series_sum_kernel[square]` for one series, the same partials and
    the same tree."""
    var partials = List[Float32]()
    for tid in range(STATS_TPB):
        var acc = Float32(0.0)
        var t = tid
        while t < n:
            var x = ftz(data[base + t])
            comptime if square:
                acc = ftz(identical_mul_add(x, x, acc))
            else:
                acc = ftz(acc + x)
            t += STATS_TPB
        partials.append(acc)
    var s0 = ftz(pinned_fold_host(partials))
    return ftz(s0 * scale)


@fieldwise_init
struct KpssHostStages(Movable):
    """Every stage the card records, as the host computes it."""

    var y_diff: List[Float32]
    var n_obs_diff: Int
    var lags: Int
    var y_means: List[Float32]
    var y_cent: List[Float32]
    var s2A: List[Float32]
    var s2B_acc: List[Float32]
    var s2B: List[Float32]
    var cumsum: List[Float32]
    var eta: List[Float32]
    var stat: List[Float32]
    var stationary: List[Bool]


def kpss_host_f32(
    y: List[Float32],
    batch_size: Int,
    n_obs: Int,
    d: Int,
    D: Int,
    s: Int,
    pval_threshold: Float32,
) raises -> KpssHostStages:
    var y_diff = prepare_data_host(y, batch_size, n_obs, d, D, s)
    var n = n_obs - d - s * D
    var n_f = Float32(n)
    var ratio = Float32(1.0) / n_f
    var y_means = List[Float32]()
    var y_cent = List[Float32]()
    var s2A = List[Float32]()
    var s2B_acc = List[Float32]()
    var s2B = List[Float32]()
    var cumsum = List[Float32]()
    var eta = List[Float32]()
    var stat = List[Float32]()
    var stationary = List[Bool]()
    var lags = kpss_lags(n)
    var coeffs = kpss_s2B_coefficients(n, lags)
    var coeff_a = coeffs[0]
    var coeff_b = coeffs[1]
    for b in range(batch_size):
        y_means.append(series_sum_host[False](y_diff, b * n, n, ratio))
    for idx in range(batch_size * n):
        var b = idx // n
        var yv = ftz(y_diff[idx])
        var mv = ftz(y_means[b])
        y_cent.append(ftz(yv - mv))
    for b in range(batch_size):
        s2A.append(series_sum_host[True](y_cent, b * n, n, Float32(1.0)))
    for idx in range(batch_size * n):
        var sample = idx % n
        var acc = Float32(0.0)
        var x0 = ftz(y_cent[idx])
        var k = 1
        while k <= lags and sample < n - k:
            var xk = ftz(y_cent[idx + k])
            var dp = ftz(x0 * xk)
            var coeff = ftz(identical_mul_add(coeff_a, Float32(k), coeff_b))
            acc = ftz(identical_mul_add(coeff, dp, acc))
            k += 1
        s2B_acc.append(acc)
    for b in range(batch_size):
        s2B.append(series_sum_host[False](s2B_acc, b * n, n, Float32(1.0)))
    for b in range(batch_size):
        var acc = Float32(0.0)
        for t in range(n):
            acc = ftz(acc + ftz(y_cent[b * n + t]))
            cumsum.append(acc)
    for b in range(batch_size):
        eta.append(series_sum_host[True](cumsum, b * n, n, Float32(1.0)))
    for b in range(batch_size):
        var st = kpss_stat_from_sums(ftz(s2A[b]), ftz(s2B[b]), ftz(eta[b]), n_f)
        stat.append(st)
        stationary.append(kpss_pvalue(st) > pval_threshold)
    return KpssHostStages(
        y_diff=y_diff^, n_obs_diff=n, lags=lags, y_means=y_means^,
        y_cent=y_cent^, s2A=s2A^, s2B_acc=s2B_acc^, s2B=s2B^, cumsum=cumsum^,
        eta=eta^, stat=stat^, stationary=stationary^,
    )


def kpss_host_f64(
    y: List[Float32],
    batch_size: Int,
    n_obs: Int,
    d: Int,
    D: Int,
    s: Int,
) raises -> Tuple[List[Float64], List[Float64]]:
    """The paper's formulas in Float64: `(stat, pvalue)` per series. The
    differencing is exact in Float64 from Float32 inputs (a single
    subtraction of two floats is exact in double), so the reference sees the
    same differenced series as the Float32 arm up to that arm's rounding."""
    var n = n_obs - d - s * D
    var lags = kpss_lags(n)
    var stat_out = List[Float64]()
    var pval_out = List[Float64]()
    var p1 = 1 if d != 0 else s
    var p2 = 1 if d == 2 else s
    for b in range(batch_size):
        var series = List[Float64]()
        for i in range(n):
            var base = b * n_obs
            var v: Float64
            if d + D == 0:
                v = Float64(y[base + i])
            elif d + D == 1:
                v = Float64(y[base + i + p1]) - Float64(y[base + i])
            else:
                v = (
                    Float64(y[base + i + p1 + p2]) - Float64(y[base + i + p1])
                    - Float64(y[base + i + p2]) + Float64(y[base + i])
                )
            series.append(v)
        var mean = 0.0
        for i in range(n):
            mean += series[i]
        mean /= Float64(n)
        var e = List[Float64]()
        for i in range(n):
            e.append(series[i] - mean)
        var s2A = 0.0
        for i in range(n):
            s2A += e[i] * e[i]
        var s2B = 0.0
        for k in range(1, lags + 1):
            var w = 2.0 / Float64(n) * (1.0 - Float64(k) / (Float64(lags) + 1.0))
            var acc = 0.0
            for t in range(n - k):
                acc += e[t] * e[t + k]
            s2B += w * acc
        var eta = 0.0
        var run = 0.0
        for t in range(n):
            run += e[t]
            eta += run * run
        var den = s2A / Float64(n) + s2B
        var st = 0.0
        if den != 0.0:
            st = (eta / (Float64(n) * Float64(n))) / den
        stat_out.append(st)
        pval_out.append(_pvalue_f64(st))
    return (stat_out^, pval_out^)


def _pvalue_f64(st: Float64) -> Float64:
    var crit = [0.347, 0.463, 0.574, 0.739]
    var pv = [0.10, 0.05, 0.025, 0.01]
    var p = pv[0]
    for k in range(3):
        if st >= crit[k] and st < crit[k + 1]:
            p = pv[k] + (pv[k + 1] - pv[k]) * (st - crit[k]) / (crit[k + 1] - crit[k])
    if st >= crit[3]:
        p = pv[3]
    return p
