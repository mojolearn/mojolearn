# SPDX-License-Identifier: Apache-2.0
"""Experimental common FP32 sampling policy, not release-qualified.

Stable raw-IEEE ordering, portable FP32 exp/division, ordered compensated sums,
and 24 exact random bits replace host FP64 sums/53-bit draws. Seeded text and
rounding may change across versions. Every route must select the same policy.
One device worker per row is a correctness-first candidate, not a speed claim.
"""
from std.memory import bitcast
from checks.numerics import portable_expf, portable_divf, pinned_mul_f32, _ftz_always


def sampling_random(seed: UInt64, row: Int, position: Int) -> UInt64:
    var x = seed + UInt64(row) * UInt64(0xD2B74407B1CE6E93)
    x += UInt64(position + 1) * UInt64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> 27)) * UInt64(0x94D049BB133111EB)
    return x ^ (x >> 31)


def _sampling_key(value: Float32) -> UInt32:
    var bits = rebind[UInt32](value.to_bits())
    # Collapse signed zero before the token-id tie break. Raw bits preserve
    # subnormal ordering even on a device whose float comparisons flush them.
    if (bits & UInt32(0x7FFFFFFF)) == 0:
        bits = 0
    return ~bits if (bits & UInt32(0x80000000)) != 0 else bits ^ UInt32(0x80000000)


def _sampling_scale24(value: Float32) -> Float32:
    var bits = rebind[UInt32](value.to_bits())
    var exponent = (bits >> 23) & UInt32(255)
    if exponent == 0:
        # Every 23-bit mantissa converts exactly. Multiplication by 2^-125
        # yields the original subnormal times2^24 as a NORMAL float (or zero).
        var magnitude = pinned_mul_f32(Float32(bits & UInt32(0x7FFFFF)), Float32(2.350988701644575e-38))
        return -magnitude if (bits & UInt32(0x80000000)) != 0 else magnitude
    return bitcast[DType.float32](bits + UInt32(24 << 23))


def _sampling_weight(value: Float32, maximum: Float32, temperature: Float32) -> Float32:
    if _sampling_key(value) == _sampling_key(maximum):
        return 1  # Includes signed zero; never form0/0 for tiny temperatures.
    var te = (rebind[UInt32](temperature.to_bits()) >> 23) & UInt32(255)
    var delta: Float32
    var denominator = temperature
    if te <= 24:
        var ve = (rebind[UInt32](value.to_bits()) >> 23) & UInt32(255)
        var me = (rebind[UInt32](maximum.to_bits()) >> 23) & UInt32(255)
        if ve >= 231 or me >= 231:
            # Scaling such operands may overflow, but unequal finite values
            # here differ by at least2^80. With T<2^-102 their negative ratio
            # is far past portable exp's zero tail. Equality was handled above.
            return 0
        delta = _sampling_scale24(value) - _sampling_scale24(maximum)
        denominator = _sampling_scale24(temperature)
    else:
        # Common FTZ arithmetic for ordinary temperatures, including operands
        # before subtraction. Tiny-logit distinctions remain in ordering but
        # may round to equal weights; this is explicit policy, not a vendor fork.
        delta = _ftz_always(_ftz_always(value) - _ftz_always(maximum))
    return portable_expf(portable_divf(delta, denominator))


def _compensated_add(value: Float32, mut total: Float32, mut correction: Float32):
    # Flush every arithmetic boundary, including the correction. This avoids
    # host-only subnormal residuals; no software FP64 representation is used.
    var adjusted = _ftz_always(_ftz_always(value) - _ftz_always(correction))
    var updated = _ftz_always(_ftz_always(total) + adjusted)
    correction = _ftz_always(_ftz_always(updated - _ftz_always(total)) - adjusted)
    total = updated


def sampling_pick_f32(
    logits: MutPointer[Float32, MutAnyOrigin],
    order: MutPointer[Int32, MutAnyOrigin],
    merge: MutPointer[Int32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin], n: Int,
    temperature: Float32, top_k: Int, top_p: Float32,
    seed: UInt64, row: Int, position: Int,
) -> Tuple[Int, Int]:
    # Policy/capacity are host-admitted. A negative selected token reports the
    # first nonfinite token without device exceptions or partial row publication.
    for i in range(n):
        if (rebind[UInt32](logits[i].to_bits()) & UInt32(0x7F800000)) == UInt32(0x7F800000):
            return (-1, i)
        order[i] = Int32(i)
    var width = 1
    var src = order
    var dst = merge
    while width < n:
        var start = 0
        while start < n:
            var mid = min(start + width, n)
            var end = min(start + width * 2, n)
            var a = start
            var b = mid
            for dest in range(start, end):
                var take_a = a < mid
                if take_a and b < end:
                    var ai = Int(src[a])
                    var bi = Int(src[b])
                    var ak = _sampling_key(logits[ai])
                    var bk = _sampling_key(logits[bi])
                    take_a = ak > bk or (ak == bk and ai < bi)
                if take_a:
                    dst[dest] = src[a]
                    a += 1
                else:
                    dst[dest] = src[b]
                    b += 1
            start = end
        var held = src
        src = dst
        dst = held
        width *= 2
    var keep = n if top_k == 0 else top_k
    var maximum = logits[Int(src[0])]
    var total = Float32(0)
    var correction = Float32(0)
    for i in range(keep):
        var weight = _sampling_weight(logits[Int(src[i])], maximum, temperature)
        weights[i] = weight
        _compensated_add(weight, total, correction)
    # maxweight1 guarantees total>0. A positive top_p rounded/flushed to zero
    # still retains one token. All top_p arithmetic is now common FP32/FTZ.
    var limit = _ftz_always(pinned_mul_f32(_ftz_always(top_p), total))
    var retained = Float32(0)
    correction = 0
    var count = 0
    while count < keep:
        _compensated_add(weights[count], retained, correction)
        count += 1
        if retained >= limit:
            break
    var uniform = pinned_mul_f32(Float32(sampling_random(seed, row, position) >> 40), Float32(1.0 / 16777216.0))
    var target = pinned_mul_f32(uniform, retained)
    var prefix = Float32(0)
    correction = 0
    for i in range(count):
        _compensated_add(weights[i], prefix, correction)
        if target < prefix:
            return (Int(src[i]), -1)
    return (Int(src[count - 1]), -1)
