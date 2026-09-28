# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`core/host_lanes.mojo`'s span helpers against the scalar statements they
name, bit for bit (lane neural-cpu, 2026-09-28).

    pixi run check-host-lanes

Operands: 2^20 hashed words reinterpreted as Float32 (every class: normal,
subnormal, both zeros, infinities, NaNs), plus values in the ranges the
seams see in practice (softmax shifts in [-100, 0], activations in [-12, 12],
weights in [0, 1]), at span lengths that exercise the lanes and the scalar
tail (n not a multiple of the SIMD width, and n below it). The expf/silu
lanes themselves are held to the scalar seams over all 2^32 patterns by
training/checks/byte_lm_host_exp_check.mojo.
"""
from std.memory import bitcast

from checks.fixture_rng import splitmix_triple
from checks.numerics import (
    ftz,
    identical_div,
    identical_exp,
    identical_fmax,
    identical_mul,
    identical_silu,
)
from core.host_lanes import (
    span_add,
    span_div,
    span_exp_shift,
    span_fmax_fold,
    span_mul,
    span_scale,
    span_silu,
)


def _word(i: Int, salt: Int) -> Float32:
    return bitcast[DType.float32](UInt32(splitmix_triple(i, salt, 77) & 0xFFFFFFFF))


def _ranged(i: Int, salt: Int, lo: Float64, hi: Float64) -> Float32:
    var w = splitmix_triple(i, salt, 91)
    return Float32(lo + (hi - lo) * Float64(Int(w >> 11)) / Float64(1 << 53))


def _fill(n: Int, kind: Int, salt: Int) -> List[Float32]:
    var out = List[Float32](length=n, fill=Float32(0.0))
    for i in range(n):
        if kind == 0:
            out[i] = _word(i, salt)
        elif kind == 1:
            out[i] = _ranged(i, salt, -100.0, 0.0)
        elif kind == 2:
            out[i] = _ranged(i, salt, -12.0, 12.0)
        else:
            out[i] = _ranged(i, salt, 0.0, 1.0)
    return out^


def _bits(x: Float32) -> UInt32:
    return bitcast[DType.uint32](x)


def _cmp(name: String, got: List[Float32], want: List[Float32], mut bad: Int):
    """`got[3 + i]` against `want[i]` (the helpers write at offset 3)."""
    var d = 0
    for i in range(len(want)):
        if _bits(got[3 + i]) != _bits(want[i]):
            d += 1
    if d != 0:
        print("DIFFER", name, d, "of", len(want))
        bad += 1


def main() raises:
    var bad = 0
    var lens: List[Int] = [1, 7, 8, 9, 31, 1 << 20]
    for li in range(len(lens)):
        var n = lens[li]
        for kind in range(4):
            var a = _fill(n, kind, 1 + kind * 10 + li)
            var b = _fill(n, (kind + 1) % 4, 2 + kind * 10 + li)
            var got = List[Float32](length=n + 3, fill=Float32(0.0))
            var want = List[Float32](length=n, fill=Float32(0.0))
            var scalars: List[Float32] = [
                Float32(0.125), Float32(-3.5), a[0], b[n // 2],
                bitcast[DType.float32](UInt32(0x00000005)),
            ]
            for si in range(len(scalars)):
                var c = scalars[si]
                # span_scale
                span_scale(a, 0, n, c, got, 3)
                for i in range(n):
                    want[i] = ftz(identical_mul(ftz(a[i]), c))
                _cmp("scale", got, want, bad)
                # span_exp_shift
                span_exp_shift(a, 0, n, c, got, 3)
                for i in range(n):
                    want[i] = ftz(identical_exp(ftz(ftz(a[i]) - ftz(c))))
                _cmp("exp_shift", got, want, bad)
                # span_div
                span_div(a, 0, n, c, got, 3)
                for i in range(n):
                    want[i] = ftz(identical_div(ftz(a[i]), c))
                _cmp("div", got, want, bad)
            span_silu(a, 0, n, got, 3)
            for i in range(n):
                want[i] = ftz(identical_silu(ftz(a[i])))
            _cmp("silu", got, want, bad)
            span_mul(a, 0, b, 0, n, got, 3)
            for i in range(n):
                want[i] = ftz(identical_mul(ftz(a[i]), ftz(b[i])))
            _cmp("mul", got, want, bad)
            span_add(a, 0, b, 0, n, got, 3)
            for i in range(n):
                want[i] = ftz(ftz(a[i]) + ftz(b[i]))
            _cmp("add", got, want, bad)
            if n <= 4096:
                for start in range(0, n, 3):
                    var cnt = n - start
                    var mx = ftz(a[start])
                    for j in range(1, cnt):
                        mx = identical_fmax(mx, ftz(a[start + j]))
                    if _bits(span_fmax_fold(a, start, cnt)) != _bits(mx):
                        print("DIFFER fmax_fold n", n, "start", start)
                        bad += 1
                        break
    print("host_lanes_check:", bad, "differ")
    if bad != 0:
        raise Error("host_lanes_check: FAIL")
    print("host_lanes_check: PASS")
