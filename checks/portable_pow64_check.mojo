# SPDX-License-Identifier: Apache-2.0
"""HOST ONLY. Portable pow64 semantics, 2^18 hashed pairs, and libm admission.

Normal finite outputs: relative error <=2e-12; the absolute allowance is
at most two smallest binary64 subnormals. This is an approximation admission
bound, not a tight-ULP or correct-rounding claim. Observed ULP/relative maxima
are reported separately. Input and portable-output FNV hashes are independent
of host libm; matching hashes require separately executed host captures.
"""
from std.ffi import external_call
from std.memory import bitcast
from checks.numerics import portable_pow64, identical_pow64, identical_log2_64, portable_log2_64, GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


def _mix(x: UInt64) -> UInt64:
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def _hash(h: UInt64, bits: UInt64) -> UInt64:
    var out = h
    for byte in range(8):
        out = (out ^ ((bits >> UInt64(byte * 8)) & UInt64(255))) * UInt64(0x100000001B3)
    return out


struct Stats(Movable):
    var inputs: UInt64
    var outputs: UInt64
    var count: Int
    var worst_ulp: UInt64
    var worst_relative: Float64
    var failed: Int

    def __init__(out self):
        self.inputs = UInt64(0xCBF29CE484222325)
        self.outputs = self.inputs
        self.count = 0
        self.worst_ulp = 0
        self.worst_relative = 0
        self.failed = 0

    def sample(mut self, x: Float64, p: Float64) raises:
        var got = portable_pow64(x, p)
        var want = external_call["pow", Float64](x, p)
        var gb = bitcast[DType.uint64](got)
        var wb = bitcast[DType.uint64](want)
        var ga = gb & UInt64(0x7FFFFFFFFFFFFFFF)
        var wa = wb & UInt64(0x7FFFFFFFFFFFFFFF)
        self.inputs = _hash(_hash(self.inputs, bitcast[DType.uint64](x)), bitcast[DType.uint64](p))
        self.outputs = _hash(self.outputs, gb)
        self.count += 1
        if bitcast[DType.uint64](identical_pow64(x, p)) != gb:
            raise Error("identical_pow64 wrapper did not select portable result")
        var bad = False
        if wa > UInt64(0x7FF0000000000000):
            bad = ga != UInt64(0x7FF8000000000000)
        elif wa == UInt64(0x7FF0000000000000):
            bad = gb != wb
        elif ga >= UInt64(0x7FF0000000000000):
            bad = True
        else:
            var error = abs(got - want)
            var magnitude = abs(want)
            var allowance = magnitude * Float64(2e-12)
            var tiny = bitcast[DType.float64](UInt64(2))
            if allowance < tiny:
                allowance = tiny
            bad = error > allowance
            var ulp = ga - wa if ga >= wa else wa - ga
            if ulp > self.worst_ulp:
                self.worst_ulp = ulp
            if wa >= UInt64(0x0010000000000000) and magnitude > Float64(0):
                var relative = error / magnitude
                if relative > self.worst_relative:
                    self.worst_relative = relative
        if bad:
            if self.failed < 8:
                print("pow64 admission mismatch x_bits=", bitcast[DType.uint64](x), "p_bits=", bitcast[DType.uint64](p), "got_bits=", gb, "libm_bits=", wb)
            self.failed += 1


def exact(x: UInt64, p: UInt64, want: UInt64) raises:
    var got = bitcast[DType.uint64](portable_pow64(bitcast[DType.float64](x), bitcast[DType.float64](p)))
    if got != want:
        raise Error("pow64 special-case mismatch x=" + String(x) + " p=" + String(p) + " got=" + String(got) + " want=" + String(want))


def main() raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("pow64 gate requires IDENTICAL build")
    # NaNs, signed zero, negative parity, infinities and retained subnormals.
    exact(0x7FF8000000001234, 0, 0x3FF0000000000000)
    exact(0x3FF0000000000000, 0xFFF8000000001234, 0x3FF0000000000000)
    exact(0x4000000000000000, 0x7FF8000000001234, 0x7FF8000000000000)
    exact(0xBFF0000000000000, 0x7FF0000000000000, 0x3FF0000000000000)
    exact(0xC000000000000000, 0x3FE0000000000000, 0x7FF8000000000000)
    exact(0x8000000000000000, 0x4008000000000000, 0x8000000000000000)
    exact(0x8000000000000000, 0xC008000000000000, 0xFFF0000000000000)
    exact(0x8000000000000000, 0x3FE0000000000000, 0)
    exact(0xFFF0000000000000, 0x4008000000000000, 0xFFF0000000000000)
    exact(0xFFF0000000000000, 0xBFE0000000000000, 0)
    exact(0xBFF0000000000000, 0x4340000000000000, 0x3FF0000000000000)
    exact(0xBFF0000000000000, 0x433FFFFFFFFFFFFF, 0xBFF0000000000000)
    exact(1, 0x3FF0000000000000, 1)
    exact(0x8000000000000001, 0x3FF0000000000000, 0x8000000000000001)
    exact(0x4000000000000000, 0x4090000000000000, 0x7FF0000000000000)
    exact(0x4000000000000000, 0xC090C80000000000, 1)  # 2**-1074
    var stats = Stats()
    for i in range(1 << 18):
        var r = _mix(UInt64(i) + UInt64(0x504F573634))
        var q = _mix(r)
        var x: Float64
        var p: Float64
        if i % 4 == 0:
            # UMAP-relevant powers: roughly 2^-28..2^32, p in [0.125,4.125).
            x = bitcast[DType.float64](((UInt64(995) + (r >> 52) % UInt64(61)) << 52) | (r & UInt64(0x000FFFFFFFFFFFFF)))
            p = Float64(0.125) + Float64(q & UInt64(0xFFFFFF)) * Float64(4.0 / 16777216.0)
        elif i % 4 == 1:
            # Entire positive binary64 input range, signed exponent [-2,2).
            x = bitcast[DType.float64](((r >> 52) % UInt64(2047) << 52) | (r & UInt64(0x000FFFFFFFFFFFFF)))
            p = Float64(Int(q & UInt64(0xFFFFFF)) - 8388608) * Float64(1.0 / 4194304.0)
        elif i % 4 == 2:
            # Adjacent-to-one bases and large exponents (finite/overflow/zero).
            x = bitcast[DType.float64](UInt64(Int(0x3FF0000000000000) + Int(r % UInt64(65)) - 32))
            p = Float64(Int(q % UInt64(2000001)) - 1000000) * Float64(1e12)
        else:
            x = -bitcast[DType.float64](((UInt64(950) + (r >> 52) % UInt64(141)) << 52) | (r & UInt64(0x000FFFFFFFFFFFFF)))
            p = Float64(Int(q % UInt64(65)) - 32)
        stats.sample(x, p)
    var xb: List[UInt64] = [0, 1, 0x000FFFFFFFFFFFFF, 0x0010000000000000, 0x3FE0000000000000, 0x3FEFFFFFFFFFFFFF, 0x3FF0000000000000, 0x3FF0000000000001, 0x4000000000000000, 0x7FEFFFFFFFFFFFFF, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FF0000000000001]
    var exponents: List[Float64] = [-1075, -1074, -1022, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 1023, 1024, 1e16, -1e16, 1e308, -1e308]
    for xword in xb:
        for p in exponents:
            stats.sample(bitcast[DType.float64](xword), p)
    for k in range(2, 4097):
        if bitcast[DType.uint64](identical_log2_64(Float64(k))) != bitcast[DType.uint64](portable_log2_64(Float64(k))):
            raise Error("identical_log2_64 did not select portable arithmetic")
    print("portable_pow64 samples=", stats.count, "worst_ulp=", stats.worst_ulp, "worst_normal_relative=", stats.worst_relative, "failed=", stats.failed)
    print("portable_pow64 input_hash=", stats.inputs, "output_hash=", stats.outputs)
    if stats.failed != 0:
        raise Error("portable_pow64 accuracy admission failed")
    print("portable_pow64 special cases and approximation admission PASS")
