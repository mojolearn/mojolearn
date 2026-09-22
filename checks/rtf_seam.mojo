# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The contract's round-then-flush product step, `ftz(fma_rn(a, b, acc))`,
computed EXACTLY on every column (lane `lane/apple-seam-repair`, 2026-09-18).

WHY. Apple's native FMA flushes BEFORE rounding (`fbr`): when the exact
`a*b + acc` lies in `[2^-126 - 2^-150, 2^-126)` it returns a signed zero where
round-then-flush (`rtf`, the contract, what NVIDIA and AMD compute) returns the
smallest normal `0x00800000`. Measured on the M4 over the 262,144-triple seam
probe, every Apple
lane hashed `fbr` `f269fc70e5625987`, the contract hashes `62a6b5621e27c707`.

THE REPAIR is the kNN column's exact integer repair of 2026-09-09
(`neighbors/checks/zero_fma_boundary.mojo`), lifted here so every rtf-spelled
seam can share it. The two semantics differ ONLY where the device result is a
signed zero, so the fast path is one zero test; the slow path decides the
window with integer arithmetic on the exact product and no floating-point
operation. With operands already flushed (normal or zero), a step can reach
the window only if both operands are nonzero and the product's lowest bit is
below `2^-149`, i.e. biased exponents sum below 151 (every normal float and
every flushed accumulator is an integer multiple of `2^-149`, and the window
holds no such multiple). That filter is exact, not a heuristic.

Selected by the kernel-matrix row `lib_zero_fma_repair_for`, which names the
Apple column only. NVIDIA, AMD and the CPU compile the unchanged spelling.
"""
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul_add
from checks.kernel_matrix import TARGET_COLUMN, lib_zero_fma_repair_for


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
def rtf_repair_zero(a: Float32, b: Float32, c: Float32, zero: Float32) -> Float32:
    """Exact rtf result for a step whose flush-before-round device result was
    the signed zero `zero`. Operands flushed. Same arithmetic as the kNN
    column's `repair_zero_fma`."""
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
    # For ce >= 40 the product/cancellation lattice spacing exceeds 2**-149:
    # an exact result strictly below minnormal cannot round up to minnormal.
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


#: The repair is compiled in (Apple, IDENTICAL) unless the price arm
#: `-D MOJOLEARN_NO_ZERO_FMA_REPAIR` removes it (never shipped).
comptime RTF_REPAIR = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and lib_zero_fma_repair_for[TARGET_COLUMN]()
)


@always_inline
def rtf_fix(a: Float32, b: Float32, acc: Float32, r: Float32) -> Float32:
    """Given the column's flushed step result `r` for flushed operands, return
    the contract's rtf result. The identity on every column but Apple."""
    comptime if RTF_REPAIR:
        if (bitcast[DType.uint32](r) & UInt32(0x7fffffff)) == UInt32(0):
            var ae = bitcast[DType.uint32](a) & UInt32(0x7f800000)
            var be = bitcast[DType.uint32](b) & UInt32(0x7f800000)
            if ae != UInt32(0) and be != UInt32(0) and ae + be < UInt32(151 << 23):
                return rtf_repair_zero(a, b, acc, r)
    return r


@always_inline
def rtf_mul_add(a: Float32, b: Float32, acc: Float32) -> Float32:
    """`ftz(fma_rn(a, b, acc))` exactly, operands and `acc` already flushed."""
    return rtf_fix(a, b, acc, ftz(identical_mul_add(a, b, acc)))
