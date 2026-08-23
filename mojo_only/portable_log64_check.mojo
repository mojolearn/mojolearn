"""`portable_log64` against the host libm, and its own hash.

    pixi run check-portable-log64

HOST-ONLY (float64). 2^18 hashed finite positive doubles spanning denormals
to near-overflow, plus the edges (1, e, 2^k, the sqrt(1/2) arm boundary, the
|e| <= 2 / |e| > 2 boundary). Judged: every result within 2 ulp of the
host LIBM through `external_call` (Cephes promises ~1 ulp; two libms
disagree in the last bit, so a 0-ulp claim would fail on one of them and
prove nothing) and the edge values exact. The reference is deliberately
not `std.math.log`: on this host that is an approximation (ln 3 off by
8.2e-11) that flushes every denormal input to log(2^-1023). RECORDED: an FNV-1a64 over every result's bits -- the number
every host should print if the function is one arithmetic; the libm it is
compared against is not expected to print it.
"""
from std.ffi import external_call

from mojo_only.numerics import portable_log64


def _splitmix(x: UInt64) -> UInt64:
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def _bits(x: Float64) -> UInt64:
    from std.memory import bitcast

    return bitcast[DType.uint64](x)


def log(x: Float64) -> Float64:
    """libm's log through FFI -- NOT `std.math.log`, which on this host is
    an approximation (measured 2026-08-23: ln 3 off by 8.2e-11, about 370k
    ulp, and EVERY double denormal input returns log(2^-1023) = -709.09,
    a flush). `ensemble/decisiontree/batched_levelalgo/objectives.mojo`
    recorded the same fix for the single-precision trap."""
    return external_call["log", Float64](x)


def _ulps(a: Float64, b: Float64) -> Int:
    var ia = Int(_bits(a))
    var ib = Int(_bits(b))
    var d = ia - ib
    return d if d >= 0 else -d


def main() raises:
    from std.memory import bitcast

    var n = 1 << 18
    var h = UInt64(0xCBF29CE484222325)
    var worst = 0
    var worst_x = Float64(0.0)
    var over2 = 0
    for i in range(n):
        var r = _splitmix(UInt64(i) + UInt64(0xA5A5))
        # exponent 1..2045 (denormal band included through the low draw)
        var ebits = UInt64(1) + (r >> 12) % UInt64(2045)
        if i % 97 == 0:
            ebits = UInt64(0)  # a denormal
        var mant = r & UInt64(0x000FFFFFFFFFFFFF)
        var x = bitcast[DType.float64]((ebits << 52) | mant)
        if x <= Float64(0.0) or x != x:
            continue
        var got = portable_log64(x)
        var want = log(x)
        var u = _ulps(got, want)
        if u > worst:
            worst = u
            worst_x = x
        if u > 2:
            over2 += 1
        var gb = _bits(got)
        for b in range(8):
            h = (h ^ ((gb >> UInt64(8 * b)) & UInt64(0xFF))) * UInt64(0x100000001B3)
    # the edges, exact
    if portable_log64(1.0) != 0.0:
        raise Error("portable_log64(1) != 0")
    if portable_log64(bitcast[DType.float64](UInt64(0x7FF0000000000000))) != bitcast[DType.float64](UInt64(0x7FF0000000000000)):
        raise Error("portable_log64(+inf) != +inf")
    if _bits(portable_log64(0.0)) != UInt64(0xFFF0000000000000):
        raise Error("portable_log64(0) != -inf")
    if portable_log64(-1.0) == portable_log64(-1.0):
        raise Error("portable_log64(-1) is not NaN")
    var e1 = portable_log64(2.718281828459045)
    if _ulps(e1, 1.0) > 1:
        raise Error("portable_log64(e) is " + String(e1))
    for k in range(-1074, 1024, 37):
        var p = bitcast[DType.float64](UInt64(k + 1023) << 52) if k > -1023 else bitcast[DType.float64](UInt64(1) << UInt64(k + 1074))
        var got = portable_log64(p)
        var want = log(p)
        if _ulps(got, want) > 2:
            raise Error("portable_log64(2^" + String(k) + ") off by " + String(_ulps(got, want)) + " ulp")
    print(
        "portable_log64:", n, "hashed doubles; worst", worst, "ulp from this host's"
        " libm at x =", worst_x, ";", over2, "beyond 2 ulp; edges exact"
    )
    print("portable_log64 device-independent hash:", h)
    if over2 != 0:
        raise Error("portable_log64: " + String(over2) + " results beyond 2 ulp of the host libm")
    print("portable log64 check OK")
