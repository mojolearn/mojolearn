# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The BCa interval of the host bootstrap against SciPy's `_bca_interval` construction in binary64 over OUR
bootstrap distribution (lane py2mojo-D, batch 5, 2026-11-22; ported from python/mojolearn/tests/test_resample_bca.py,
DEVIATION 1699 / 5410).

The two libraries draw different replicates, so what is compared is the interval CONSTRUCTION given a distribution,
as in the original. The bootstrap is the host column's body (resample/host/resample_host.mojo `host_bootstrap`, which
the host resample binding's `_bootstrap_run` calls) with method BCa, 4000 resamples, seed 3. The samples are the
original's draws (`np.random.default_rng(7)`: 600 gamma(2, 1.5) values, then a paired normal column), stored as
float32 words in `fixtures/resample_bca/` by `tools/py2mojo_native_fixtures_b5.py`.

    units 0-5  mean, std and diff_means x confidence 0.9, 0.95: the oracle written here, in binary64 from the
               returned distribution and point estimate: the bias percentile
               ((#dist < theta) + (#dist <= theta)) / 2B, z0 = ndtri of it, the jackknife of the float64 sample
               (leave-one-out mean, std with ddof 1, difference of the column means), the acceleration
               sum((mean - jack)^3) / (6 sum((mean - jack)^2)^1.5), the levels ndtr(z0 + n / (1 - a n)) for
               n = z0 +- ndtri(alpha), and the endpoints as numpy.percentile (linear) of the distribution; our
               interval's ends within 1e-3 of the interval's span, as the original
    unit 6     the oracle's own ndtr / ndtri against the hand-worked SciPy levels of
               resample/checks/resample_check.mojo `check_bca_interval` (z0p 0.6, a 0.1, alpha 0.025:
               0.11420295403892616, 0.9990185075970077) to 1e-12, so a wrong oracle cannot pass units 0-5
    unit 7     BCa on an order statistic (quantile) is refused

The method's spelling ('bca', 'BCA') and the refusal's words stay in the Python file. Every unit is independent and
the table runs across the host cores.
"""
from std.math import erfc, exp, log, sqrt
from resample.host.resample_host import host_bootstrap
from resample.checks.intervals import ALT_TWO_SIDED, METHOD_BCA
from resample.checks.statistics import STAT_DIFF_MEANS, STAT_MEAN, STAT_QUANTILE, STAT_STD
from checks.native_tests.tsa_metrics_fixture_words import fixture_f32
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases

comptime BCA_DIR = "checks/native_tests/fixtures/resample_bca/"
comptime BCA_N = 600
comptime BCA_B = 4000
comptime BCA_UNITS = 8


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


# ---------------------------------------------------------------- the oracle (binary64)


def ndtr(x: Float64) -> Float64:
    """scipy.special.ndtr: the standard normal CDF."""
    return 0.5 * erfc(-x / sqrt(2.0))


def ndtri(p: Float64) -> Float64:
    """scipy.special.ndtri: Acklam's rational start, then three Newton steps on ndtr (binary64 accuracy)."""
    var a: List[Float64] = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
                            1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    var b: List[Float64] = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
                            6.680131188771972e01, -1.328068155288572e01]
    var c: List[Float64] = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
                            -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    var d: List[Float64] = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
                            3.754408661907416e00]
    var x: Float64
    if p < 0.02425:
        var q = sqrt(-2.0 * log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p > 1.0 - 0.02425:
        var q = sqrt(-2.0 * log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    else:
        var q = p - 0.5
        var r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    for _ in range(3):  # small-loop(Newton refinement)
        var pdf = exp(-0.5 * x * x) / sqrt(2.0 * 3.141592653589793)
        x -= (ndtr(x) - p) / pdf
    return x


def bca_levels(z0p: Float64, a: Float64, alpha: Float64) -> Tuple[Float64, Float64]:
    """SciPy `_bca_interval`'s two levels."""
    var z0 = ndtri(z0p)
    var za = ndtri(alpha)
    var n1 = z0 + za
    var n2 = z0 - za
    return (ndtr(z0 + n1 / (1.0 - a * n1)), ndtr(z0 + n2 / (1.0 - a * n2)))


def percentile(sorted: List[Float64], q: Float64) -> Float64:
    """numpy.percentile(..., q * 100), the linear method, over sorted values."""
    var h = Float64(len(sorted) - 1) * q
    var lo = Int(h)
    if lo >= len(sorted) - 1:
        return sorted[len(sorted) - 1]
    return sorted[lo] + (h - Float64(lo)) * (sorted[lo + 1] - sorted[lo])


def jackknife(data: List[Float32], statistic: Int) -> List[Float64]:
    """The original's `_jack`: the statistic of the float64 sample without row i."""
    var n = BCA_N
    var out = List[Float64](length=n, fill=0.0)
    var cols = 2 if statistic == STAT_DIFF_MEANS else 1
    var total0 = Float64(0)
    var total1 = Float64(0)
    for i in range(n):  # small-loop(the float64 sample sums)
        total0 += Float64(data[cols * i])
        if cols == 2:
            total1 += Float64(data[cols * i + 1])
    for i in range(n):  # small-loop(one leave-one-out statistic per row)
        var m0 = (total0 - Float64(data[cols * i])) / Float64(n - 1)
        if statistic == STAT_MEAN:
            out[i] = m0
        elif statistic == STAT_STD:
            var ss = Float64(0)
            for j in range(n):  # small-loop(the deviations without row i)
                if j != i:
                    var e = Float64(data[j]) - m0
                    ss += e * e
            out[i] = sqrt(ss / Float64(n - 2))
        else:
            out[i] = m0 - (total1 - Float64(data[cols * i + 1])) / Float64(n - 1)
    return out^


def bca_unit(unit: Int) raises:
    if unit == 6:
        var lv = bca_levels(0.6, 0.1, 0.025)
        require(abs(lv[0] - 0.11420295403892616) <= 1e-12 and abs(lv[1] - 0.9990185075970077) <= 1e-12,
                "the oracle's levels " + String(lv[0]) + ", " + String(lv[1])
                + " are not SciPy's 0.11420295403892616, 0.9990185075970077")
        return
    var x = fixture_f32(BCA_DIR + "bca-x.f32", BCA_N)
    if unit == 7:
        var refused = False
        try:
            _ = host_bootstrap(x, BCA_N, 1, STAT_QUANTILE, 64, 0, METHOD_BCA, Float32(0.95), ALT_TWO_SIDED,
                               Float32(0.5), 0)
        except problem:
            refused = True
        require(refused, "BCa on an order statistic (quantile) was not refused")
        return
    var stats: List[Int] = [STAT_MEAN, STAT_STD, STAT_DIFF_MEANS]
    var names: List[String] = ["mean", "std", "diff_means"]
    var statistic = stats[unit // 2]
    var cl = 0.9 if unit % 2 == 0 else 0.95
    var data = x.copy()
    var d = 1
    if statistic == STAT_DIFF_MEANS:
        data = fixture_f32(BCA_DIR + "bca-two.f32", 2 * BCA_N)
        d = 2
    var res = host_bootstrap(data, BCA_N, d, statistic, BCA_B, 3, METHOD_BCA, Float32(cl), ALT_TWO_SIDED,
                             Float32(0.5), 0)
    var theta = Float64(res.point_estimate)
    var sorted = List[Float64](length=BCA_B, fill=0.0)
    var below = 0
    var at_or_below = 0
    for r in range(BCA_B):  # small-loop(the oracle over one distribution)
        var v = Float64(res.distribution[r])
        below += 1 if v < theta else 0
        at_or_below += 1 if v <= theta else 0
        sorted[r] = Float64(res.sorted_distribution[r])
    for r in range(1, BCA_B):  # small-loop(the returned sorted distribution is the distribution, ordered)
        require(sorted[r - 1] <= sorted[r], names[unit // 2] + ": sorted_distribution is not ascending")
    var pct = Float64(below + at_or_below) / Float64(2 * BCA_B)
    var jack = jackknife(data, statistic)
    var dot = Float64(0)
    for i in range(BCA_N):  # small-loop(oracle)
        dot += jack[i]
    dot /= Float64(BCA_N)
    var num = Float64(0)
    var den = Float64(0)
    for i in range(BCA_N):  # small-loop(oracle)
        var u = dot - jack[i]
        num += u * u * u
        den += u * u
    var accel = num / (6.0 * den * sqrt(den))
    var lv = bca_levels(pct, accel, (1.0 - cl) / 2.0)
    var lo = percentile(sorted, lv[0])
    var hi = percentile(sorted, lv[1])
    var span = hi - lo
    var got_lo = Float64(res.interval.low)
    var got_hi = Float64(res.interval.high)
    require(abs(got_lo - lo) <= 1e-3 * span and abs(got_hi - hi) <= 1e-3 * span,
            "BCa " + names[unit // 2] + " cl=" + String(cl) + ": ours [" + String(got_lo) + ", " + String(got_hi)
            + "], SciPy's construction [" + String(lo) + ", " + String(hi) + "] (levels " + String(lv[0]) + ", "
            + String(lv[1]) + ")")


def test_bca_interval() raises:
    check_cases("bca_interval", BCA_UNITS, bca_unit)


def main() raises:
    var suite = TestSuite()
    suite.test[test_bca_interval]()
    suite^.run()
