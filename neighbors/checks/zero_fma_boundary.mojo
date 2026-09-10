# SPDX-License-Identifier: Apache-2.0
"""Exact kNN-local repair of pre-round underflow flushing.

Called only when flushed hardware FMA returned zero and inputs were already
flushed. Compare exact product to the two round-to-minnormal intervals in
units of 2**-150. No floating-point operation enters this decision.
"""
from std.memory import bitcast

@always_inline
def _scaled_compare(p: UInt64, shift: Int, bound: UInt64) -> Int:
    var largest = UInt64(0xffffffffffffffff)
    if shift >= 0:
        if shift >= 64:
            return 1
        if p > (largest >> UInt64(shift)):
            return 1
        var v = p << UInt64(shift)
        return -1 if v < bound else (1 if v > bound else 0)
    if bound == 0:
        return 1
    var right = -shift
    if right >= 64:
        return -1
    if bound > (largest >> UInt64(right)):
        return -1
    var v = bound << UInt64(right)
    return -1 if p < v else (1 if p > v else 0)

@no_inline
def repair_zero_fma(a: Float32, b: Float32, c: Float32, zero: Float32) -> Float32:
    var aw = bitcast[DType.uint32](a)
    var bw = bitcast[DType.uint32](b)
    var cw = bitcast[DType.uint32](c)
    var ae = Int((aw >> 23) & 255)
    var be = Int((bw >> 23) & 255)
    var ce = Int((cw >> 23) & 255)
    if ae == 0 or be == 0 or ae == 255 or be == 255:
        return zero
    var product = UInt64((aw & 0x7fffff) | 0x800000) * UInt64((bw & 0x7fffff) | 0x800000)
    var shift = ae + be - 150
    var ps = (aw ^ bw) & 0x80000000
    var cs = cw & 0x80000000
    var lo = UInt64(0xffffff)
    var hi = UInt64(0x1000001)
    if ce == 0:
        if _scaled_compare(product, shift, lo) >= 0 and _scaled_compare(product, shift, hi) <= 0:
            return bitcast[DType.float32](ps | 0x00800000)
        return zero
    # For ce >= 40, product/cancellation lattice spacing exceeds 2**-149.
    # An exact result strictly below minnormal cannot round up to minnormal;
    # an exact normal result cannot be an underflow-flushed hardware zero.
    if ce >= 40 or ps == cs:
        return zero
    var center = UInt64((cw & 0x7fffff) | 0x800000) << UInt64(ce)
    var above_lower = center < hi
    if not above_lower:
        above_lower = _scaled_compare(product, shift, center - hi) >= 0
    if above_lower and _scaled_compare(product, shift, center - lo) <= 0:
        return bitcast[DType.float32](cs | 0x00800000)
    if _scaled_compare(product, shift, center + lo) >= 0 and _scaled_compare(product, shift, center + hi) <= 0:
        return bitcast[DType.float32](ps | 0x00800000)
    return zero
