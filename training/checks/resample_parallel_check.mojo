# SPDX-License-Identifier: Apache-2.0
"""Cloud-only multi-GPU resampling gate: replicate, permutation and sample ranges.

Bootstrap (all six statistics, percentile and basic, three alternatives,
r_first 0 and 5), permutation tests (three statistics) and Monte Carlo
integration (three integrands, chunk boundaries, i_first 0 and 3) run with
MOJOLEARN_RESAMPLE_DEVICE_COUNT=1 and =2; identity traces and every result
bit must be equal. Build with -D MOJOLEARN_RESAMPLE_PARALLEL_SABOTAGE=1 to see
it fail.
"""
from std.os import getenv, setenv
from std.memory import bitcast

from core.identity_trace import first_divergence
from resample.checks.intervals import ALT_GREATER, ALT_LESS, ALT_TWO_SIDED, METHOD_BASIC, METHOD_PERCENTILE
from resample.checks.statistics import (
    MC_F_CONST,
    MC_F_PRODUCT,
    MC_F_SUM,
    STAT_DIFF_MEANS,
    STAT_MEAN,
    STAT_PEARSON,
    STAT_QUANTILE,
    STAT_STD,
    STAT_TRIMMED_MEAN,
)
from resample.estimator import bootstrap_host, monte_carlo_integrate_host, permutation_test_host


def _unit(seed: Int, a: Int) -> Float32:
    var u = UInt64(seed) * UInt64(0x9E3779B97F4A7C15) + UInt64(a) * UInt64(0xBF58476D1CE4E5B9)
    u ^= u >> 31
    u *= UInt64(0x94D049BB133111EB)
    u ^= u >> 29
    return Float32(Int((u >> 40) & 0xFFFF)) / Float32(8192.0) - Float32(4.0)


def _bits(v: Float32) -> UInt32:
    return bitcast[DType.uint32](v)


def _same(a: List[Float32], b: List[Float32], what: String) raises:
    if len(a) != len(b):
        raise Error(what + " length differs")
    for i in range(len(a)):
        if _bits(a[i]) != _bits(b[i]):
            raise Error(what + " bits differ at " + String(i))


def _arm(count: Int, path: String) raises:
    if not setenv("MOJOLEARN_RESAMPLE_DEVICE_COUNT", String(count), True):
        raise Error("setenv failed")
    with open(path, "w") as fh:
        fh.write("")
    if not setenv("MOJOLEARN_IDENTITY_TRACE", path, True):
        raise Error("setenv failed")


def _traces(root: String, name: String) raises:
    var d = first_divergence(root + "/" + name + "-one.trace", root + "/" + name + "-many.trace")
    if d != "":
        raise Error("trace differs " + name + ": " + d)


def boot_case(root: String, count: Int, n: Int, d: Int, stat: Int, method: Int,
    alt: Int, r: Int, r_first: Int) raises:
    var x = List[Float32]()
    for i in range(n * d):
        x.append(_unit(n * 3 + d, i))
    var qp = Float32(0.25) if stat == STAT_QUANTILE else Float32(0.125)
    var name = "boot-n" + String(n) + "-d" + String(d) + "-s" + String(stat) + "-m" + String(method) + "-a" + String(alt) + "-r" + String(r) + "-f" + String(r_first)
    _arm(1, root + "/" + name + "-one.trace")
    var one = bootstrap_host(x, n, d, stat, r, UInt64(77), method, Float32(0.9), alt, qp, r_first)
    _arm(count, root + "/" + name + "-many.trace")
    var many = bootstrap_host(x, n, d, stat, r, UInt64(77), method, Float32(0.9), alt, qp, r_first)
    _traces(root, name)
    _same(one.distribution, many.distribution, name + " distribution")
    _same(one.sorted_distribution, many.sorted_distribution, name + " sorted")
    if _bits(one.point_estimate) != _bits(many.point_estimate) or _bits(one.standard_error) != _bits(many.standard_error):
        raise Error(name + " point or se differs")
    if _bits(one.interval.low) != _bits(many.interval.low) or _bits(one.interval.high) != _bits(many.interval.high):
        raise Error(name + " interval differs")
    if one.order_low != many.order_low or one.order_high != many.order_high:
        raise Error(name + " order positions differ")
    print("PASS resample bootstrap", n, d, stat, method, alt, r, r_first, count)


def perm_case(root: String, count: Int, nx: Int, ny: Int, stat: Int, alt: Int, r: Int, r_first: Int) raises:
    var x = List[Float32]()
    var y = List[Float32]()
    for i in range(nx):
        x.append(_unit(11, i))
    for i in range(ny):
        y.append(_unit(13, i) + Float32(0.25))
    var name = "perm-" + String(nx) + "-" + String(ny) + "-s" + String(stat) + "-a" + String(alt) + "-r" + String(r) + "-f" + String(r_first)
    _arm(1, root + "/" + name + "-one.trace")
    var one = permutation_test_host(x, y, stat, r, UInt64(5), alt, r_first)
    _arm(count, root + "/" + name + "-many.trace")
    var many = permutation_test_host(x, y, stat, r, UInt64(5), alt, r_first)
    _traces(root, name)
    _same(one.null_distribution, many.null_distribution, name + " null")
    if _bits(one.observed) != _bits(many.observed) or _bits(one.pvalue.p) != _bits(many.pvalue.p):
        raise Error(name + " observed or p differs")
    if one.pvalue.count_less != many.pvalue.count_less or one.pvalue.count_greater != many.pvalue.count_greater:
        raise Error(name + " counts differ")
    print("PASS resample permutation", nx, ny, stat, alt, r, r_first, count)


def mc_case(root: String, count: Int, f: Int, n: Int, i_first: Int) raises:
    var lower: List[Float32] = [Float32(-1.5), Float32(0.25)]
    var upper: List[Float32] = [Float32(2.0), Float32(3.0)]
    var name = "mc-f" + String(f) + "-n" + String(n) + "-i" + String(i_first)
    var a = List[Float32]()
    var b = List[Float32]()
    for arm in range(2):
        _arm(1 if arm == 0 else count, root + "/" + name + ("-one.trace" if arm == 0 else "-many.trace"))
        var integral = Float32(0.0)
        var mean = Float32(0.0)
        if f == MC_F_CONST:
            var r = monte_carlo_integrate_host[MC_F_CONST](lower, upper, n, UInt64(9), i_first)
            integral = r.integral
            mean = r.mean
        elif f == MC_F_SUM:
            var r = monte_carlo_integrate_host[MC_F_SUM](lower, upper, n, UInt64(9), i_first)
            integral = r.integral
            mean = r.mean
        else:
            var r = monte_carlo_integrate_host[MC_F_PRODUCT](lower, upper, n, UInt64(9), i_first)
            integral = r.integral
            mean = r.mean
        if arm == 0:
            a.append(integral)
            a.append(mean)
        else:
            b.append(integral)
            b.append(mean)
    _traces(root, name)
    _same(a, b, name + " integral/mean")
    print("PASS resample monte_carlo", f, n, i_first, count)


def main() raises:
    if String(getenv("RUNPOD_POD_ID")) == "":
        raise Error("RunPod required; no local execution")
    var count = Int(String(getenv("MOJOLEARN_RESAMPLE_CHECK_DEVICES", "2")))
    var root = String(getenv("MOJOLEARN_RESAMPLE_CHECK_DIR", "/tmp"))
    var stats: List[Int] = [STAT_MEAN, STAT_STD, STAT_QUANTILE, STAT_PEARSON, STAT_DIFF_MEANS, STAT_TRIMMED_MEAN]
    var reps: List[Int] = [2, 7, 1000, 4099]
    for s in stats:
        for r in reps:
            boot_case(root, count, 53, 2, s, METHOD_PERCENTILE, ALT_TWO_SIDED, r, 0)
        boot_case(root, count, 53, 2, s, METHOD_BASIC, ALT_LESS, 513, 5)
        boot_case(root, count, 8, 2, s, METHOD_PERCENTILE, ALT_GREATER, 3, 5)
    var pstats: List[Int] = [STAT_DIFF_MEANS, STAT_MEAN, STAT_STD]
    for s in pstats:
        perm_case(root, count, 40, 33, s, ALT_TWO_SIDED, 2, 0)
        perm_case(root, count, 40, 33, s, ALT_LESS, 999, 0)
        perm_case(root, count, 9, 12, s, ALT_GREATER, 4097, 5)
    var fs: List[Int] = [MC_F_CONST, MC_F_SUM, MC_F_PRODUCT]
    var ns: List[Int] = [1, 255, 256, 257, 1000, 100003]
    for f in fs:
        for n in ns:
            mc_case(root, count, f, n, 0)
        mc_case(root, count, f, 70001, 3)
    if not setenv("MOJOLEARN_IDENTITY_TRACE", "", True):
        raise Error("setenv failed")
    print("PASS resample parallel gate")
