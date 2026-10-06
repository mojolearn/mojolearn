# SPDX-License-Identifier: Apache-2.0
"""T26 V1: no-sort RMSE/Newton one-step leaf statistics, host/device units.

NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.

Logical chunks contain 256 consecutive ORIGINAL rows. Each leaf folds its
members in ascending row order from +0; partials merge in an adjacent binary
tree, padding absent right children with no operation. Every sum/subtraction
is software IEEE binary64 followed by Float32 rounding then FTZ; no FMA.
Empty leaves return +0. Nonfinite inputs/intermediates are refused. This is a
new arithmetic version shared by the host and every GPU. Search/RNG/order,
missing routing and split ties are unchanged. Only unweighted, one-permutation,
non-symmetric RMSE with exactly one Newton step selects this implementation.
"""
from std.math import isfinite
from checks.numerics import ftz
from checks.soft_f64 import (
    SF64_ZERO, sf64_add, sf64_sub, sf64_div, sf64_from_f32, sf64_gt,
    sf64_to_f32,
)

comptime T26_CHUNK = 256
comptime T26_SCRATCH_BYTES = 64 * 1024 * 1024
comptime T26F = MutPointer[Float32, MutAnyOrigin]
comptime T26B = MutPointer[UInt32, MutAnyOrigin]

@always_inline
def t26_add(a: Float32, b: Float32) -> Float32:
    return ftz(sf64_to_f32(sf64_add(sf64_from_f32(a), sf64_from_f32(b))))

@always_inline
def t26_partial(
    leaf: Int, chunk: Int, n: Int, bins: T26B, y: T26F, cursor: T26F,
) -> SIMD[DType.float32, 2]:
    var grad = Float32(0.0)
    var mass = Float32(0.0)
    for row in range(chunk * T26_CHUNK, min((chunk + 1) * T26_CHUNK, n)):
        if Int(bins[row]) == leaf:
            var residual = ftz(sf64_to_f32(sf64_sub(
                sf64_from_f32(ftz(y[row])), sf64_from_f32(ftz(cursor[row])),
            )))
            grad = t26_add(grad, residual)
            mass = t26_add(mass, Float32(1.0))
    return SIMD[DType.float32, 2](grad, mass)

@always_inline
def t26_leaf(grad: Float32, mass: Float32, lam: Float32) -> Float32:
    if mass == Float32(0.0):
        return Float32(0.0)
    var h = sf64_add(sf64_from_f32(mass), sf64_from_f32(lam))
    if not sf64_gt(h, SF64_ZERO):
        return Float32(0.0)
    var v = sf64_to_f32(sf64_div(
        sf64_from_f32(grad), sf64_add(h, sf64_from_f32(Float32(1e-20))),
    ))
    return Float32(0.0) if v == Float32(0.0) else v


def t26_chunks(n: Int, leaves: Int, lam: Float32) raises -> Int:
    # Match the production device kernel's Int32 ABI on every column. The
    # scratch bound is a memory budget, independent of dataset/board shapes.
    if n < 1 or n > 2147483647 or leaves < 1 or leaves > 2147483647:
        raise Error("T26 requires positive row/leaf counts within Int32 ABI")
    if not isfinite(lam):
        raise Error("T26 refuses nonfinite regularization")
    var chunks = (n + T26_CHUNK - 1) // T26_CHUNK
    if chunks > T26_SCRATCH_BYTES // 8:
        raise Error("T26 canonical chunk scratch exceeds 64 MiB per leaf")
    return chunks


def t26_host_leaves(
    bins: List[UInt32], y: List[Float32], cursor: List[Float32],
    n: Int, leaves: Int, lam: Float32,
) raises -> List[Float32]:
    var chunks = t26_chunks(n, leaves, lam)
    var partial = List[Float32](length=2 * chunks, fill=0.0)
    var values = List[Float32](length=leaves, fill=0.0)
    var bp = bins.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()
    var yp = y.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()
    var cp = cursor.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()
    for leaf in range(leaves):
        for chunk in range(chunks):
            var p = t26_partial(leaf, chunk, n, bp, yp, cp)
            partial[2 * chunk] = p[0]
            partial[2 * chunk + 1] = p[1]
        var step = 1
        while step < chunks:
            var c = 0
            while c + step < chunks:
                partial[2 * c] = t26_add(partial[2 * c], partial[2 * (c + step)])
                partial[2 * c + 1] = t26_add(partial[2 * c + 1], partial[2 * (c + step) + 1])
                c += 2 * step
            step *= 2
        if not isfinite(partial[0]) or not isfinite(partial[1]):
            raise Error("T26 refuses nonfinite leaf statistics")
        values[leaf] = t26_leaf(partial[0], partial[1], lam)
        if not isfinite(values[leaf]):
            raise Error("T26 refuses a nonfinite leaf estimate")
    return values^
