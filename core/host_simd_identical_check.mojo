# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The measurement behind `core/host_simd_identical.mojo`: EVERY float32 bit
pattern (all 2^32) through `ftz_v`, `expf_v` and `logf_v`, compared bit for
bit with the scalar seam (`ftz`, `portable_expf`, `portable_logf`) on the
calling thread. PASS prints the count of patterns compared and zero
mismatches; the first mismatches are printed and the run raises.

    pixi run check-host-simd-identical
    # the sabotage arm (one expf coefficient moved one unit) must FAIL:
    mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_HOST_SIMD_SABOTAGE=1 \
        -I . core/host_simd_identical_check.mojo
"""
from std.memory import bitcast

from checks.numerics import ftz, portable_expf, portable_logf, portable_powf
from core.host_simd_identical import expf_v, ftz_v, logf_v, powf_v

comptime W = 8


def _same(a: Float32, b: Float32) -> Bool:
    return bitcast[DType.uint32](a) == bitcast[DType.uint32](b)


def main() raises:
    var bad_exp = 0
    var bad_log = 0
    var bad_ftz = 0
    var bad_pow = 0
    var shown = 0
    var base = UInt64(0)
    var total = UInt64(1) << UInt64(32)
    var lanes = SIMD[DType.uint32, W](0, 1, 2, 3, 4, 5, 6, 7)
    while base < total:
        var bits = SIMD[DType.uint32, W](UInt32(base)) + lanes
        var x = bitcast[DType.float32, W](bits)
        var ev = expf_v[W](x)
        var lv = logf_v[W](x)
        var fv = ftz_v[W](x)
        # powf at the Minkowski exponents the lanes use (p = 3 and 1/3),
        # every x.
        var pv3 = powf_v[W](x, Float32(3.0))
        var pvt = powf_v[W](x, Float32(1.0) / Float32(3.0))
        comptime for l in range(W):
            var xs = x[l]
            if not _same(ev[l], portable_expf(xs)):
                bad_exp += 1
                if shown < 8:
                    shown += 1
                    print("MISMATCH expf bits", bits[l], "vector", ev[l], "scalar", portable_expf(xs))
            if not _same(lv[l], portable_logf(xs)):
                bad_log += 1
                if shown < 8:
                    shown += 1
                    print("MISMATCH logf bits", bits[l], "vector", lv[l], "scalar", portable_logf(xs))
            if not _same(fv[l], ftz(xs)):
                bad_ftz += 1
            if not _same(pv3[l], portable_powf(xs, Float32(3.0))):
                bad_pow += 1
            if not _same(pvt[l], portable_powf(xs, Float32(1.0) / Float32(3.0))):
                bad_pow += 1
        base += UInt64(W)
    # powf's branches at the other exponents, on a sample of words
    var ps: List[Float32] = [0.0, -0.0, 0.5, 2.0, 1.5, -1.0, -2.5, 7.0]
    var sb = UInt32(12345)
    for _ in range(1 << 22):
        sb = sb * UInt32(1664525) + UInt32(1013904223)
        var xs = bitcast[DType.float32](sb)
        var xv = SIMD[DType.float32, W](xs)
        for pi in range(len(ps)):
            var pv = powf_v[W](xv, ps[pi])
            if not _same(pv[0], portable_powf(xs, ps[pi])):
                bad_pow += 1
    var nanp = bitcast[DType.float32](UInt32(0x7FC00000))
    if not _same(powf_v[W](SIMD[DType.float32, W](2.0), nanp)[0], portable_powf(Float32(2.0), nanp)):
        bad_pow += 1
    print("host_simd_identical: compared", total, "patterns; mismatches expf", bad_exp,
          "logf", bad_log, "ftz", bad_ftz, "powf", bad_pow)
    if bad_exp + bad_log + bad_ftz + bad_pow != 0:
        raise Error("host_simd_identical: FAIL")
    print("host_simd_identical: PASS")
