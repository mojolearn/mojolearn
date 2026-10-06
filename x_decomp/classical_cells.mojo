# SPDX-License-Identifier: Apache-2.0
"""Classical-only C27 fused independent component cells.
NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
"""
from checks.numerics import ftz, identical_tanh, identical_mul_add
from x_decomp.cells import F32Ptr, mul, sub, exp_c

@always_inline
def contrast_pair(y: Float32, fun: Int, alpha: Float32) -> SIMD[DType.float32, 2]:
    """One immutable score supplies g and g'; every B rounding seam retained."""
    var x = ftz(y)
    var g = Float32(0)
    var gp = Float32(0)
    if fun == 0:
        g = ftz(identical_tanh(mul(x, alpha)))
        gp = mul(sub(Float32(1), mul(g, g)), alpha)
    elif fun == 1:
        var x2 = mul(x, x)
        var e = exp_c(mul(Float32(-0.5), x2))
        g = mul(x, e)
        gp = mul(sub(Float32(1), x2), e)
    else:
        var x2 = mul(x, x)
        g = mul(x2, x)
        gp = mul(Float32(3), x2)
    return SIMD[DType.float32, 2](g, gp)


def centered_gram_cell(x: F32Ptr, means: F32Ptr, n: Int, d: Int, i: Int, j: Int) -> Float32:
    """C23 profile v1: centered rows in panels256; binary left-before-right tree.
    32 levels cover the public int32 row domain; no raw moment subtraction.
    Empty tail leaves are absent, and singleton nodes promote unchanged.
    """
    var levels = InlineArray[Float32, 32](fill=Float32(0))
    var occupied = UInt32(0)
    var mi = means.unsafe_load(i)
    var mj = means.unsafe_load(j)
    for first in range(0, n, 256):
        var acc = Float32(0)
        for r in range(first, min(n, first + 256)):
            acc = ftz(identical_mul_add(sub(x.unsafe_load(r * d + i), mi), sub(x.unsafe_load(r * d + j), mj), acc))
        var level = 0
        while (occupied & (UInt32(1) << level)) != 0:
            acc = ftz(levels[level] + acc)
            occupied = occupied & ~(UInt32(1) << level)
            level += 1
        levels[level] = acc
        occupied = occupied | (UInt32(1) << level)
    var acc = Float32(0)
    var have = False
    for level in range(32):
        if (occupied & (UInt32(1) << level)) != 0:
            acc = ftz(levels[level] + acc) if have else levels[level]
            have = True
    return acc
