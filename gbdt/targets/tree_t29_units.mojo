# SPDX-License-Identifier: Apache-2.0
"""T29 V1 generated-PairLogit arithmetic shared by host and every GPU.

NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.

Partners retain ascending original query-row order and every unequal-grade
pair. A row sums each consecutive 32-partner chunk serially, then merges
adjacent chunks with a binary carry tree; remaining subtrees are folded from
oldest to newest. Equal-grade partners contribute nothing. No query, pair,
weight, seed or permutation is sampled, capped or discarded. Logical group
summaries have exactly 256 lanes: row (i-begin) % 256 owns a lane, increasing
rows fold serially within it, then lanes fold at strides 128,64,...,1.

Every arithmetic operation below converts FTZ Float32 operands to software
binary64, rounds that result to Float32 and applies FTZ. Portable identical
exp/log are the only transcendentals. Products never fuse into sums. Empty
sums and signed zeros canonicalize to +0. Overflow and NaNs follow the shared
software operations; the incumbent finite-input checks and loss refusals
remain in their callers. This is a new same-version graph, not incumbent
bit equivalence or demonstrated cross-vendor qualification.
"""
from std.math import isfinite
from checks.numerics import ftz, identical_exp, identical_log
from checks.soft_f64 import (
    sf64_add, sf64_sub, sf64_mul, sf64_div, sf64_from_f32, sf64_to_f32,
)

comptime T29_PARTNER_CHUNK = 32
comptime T29_GROUP_LANES = 256
comptime T29_CARRY_LEVELS = 32
comptime T29F = MutPointer[Float32, MutAnyOrigin]


@always_inline
def t29_round(value: UInt64) -> Float32:
    var out = ftz(sf64_to_f32(value))
    return Float32(0.0) if out == Float32(0.0) else out


@always_inline
def t29_add(a: Float32, b: Float32) -> Float32:
    return t29_round(sf64_add(sf64_from_f32(ftz(a)), sf64_from_f32(ftz(b))))


@always_inline
def t29_sub(a: Float32, b: Float32) -> Float32:
    return t29_round(sf64_sub(sf64_from_f32(ftz(a)), sf64_from_f32(ftz(b))))


@always_inline
def t29_mul(a: Float32, b: Float32) -> Float32:
    return t29_round(sf64_mul(sf64_from_f32(ftz(a)), sf64_from_f32(ftz(b))))


@always_inline
def t29_div(a: Float32, b: Float32) -> Float32:
    return t29_round(sf64_div(sf64_from_f32(ftz(a)), sf64_from_f32(ftz(b))))


def t29_pair_row(
    point: T29F, grades: T29F, row: Int, begin: Int, end: Int,
    weight: Float32, compute_value: Bool,
) -> SIMD[DType.float32, 4]:
    """Return derivative, curvature and winner-only value, plus unused +0.

    Thirty-two carry levels cover the existing UInt32/Int32 row-index ABI.
    The bound is representation-derived, never a maximum accepted query size.
    Scratch is constant in the query length; the final short chunk is exact.
    """
    var ds = InlineArray[Float32, T29_CARRY_LEVELS](fill=0.0)
    var hs = InlineArray[Float32, T29_CARRY_LEVELS](fill=0.0)
    var vs = InlineArray[Float32, T29_CARRY_LEVELS](fill=0.0)
    var occupied = UInt32(0)
    var grade = grades[row]
    var p_i = point[row]
    var start = begin
    while start < end:
        var d = Float32(0.0)
        var h = Float32(0.0)
        var v = Float32(0.0)
        for partner in range(start, min(start + T29_PARTNER_CHUNK, end)):
            var other_grade = grades[partner]
            if other_grade == grade:
                continue
            var winner = grade > other_grade
            var diff = t29_sub(p_i, point[partner]) if winner else t29_sub(point[partner], p_i)
            var exp_diff = ftz(identical_exp(diff))
            var denom = t29_add(Float32(1.0), exp_diff)
            var probability = Float32(1.0)
            if isfinite(denom):
                probability = t29_div(exp_diff, denom)
            probability = max(min(probability, t29_sub(Float32(1.0), Float32(1e-40))), ftz(Float32(1e-40)))
            var direction = t29_sub(Float32(1.0), probability)
            var scale = t29_mul(probability, direction)
            var term = t29_mul(weight, direction)
            d = t29_add(d, term if winner else -term)
            h = t29_add(h, t29_mul(weight, scale))
            if winner and compute_value:
                var log_denom = diff
                if isfinite(denom):
                    log_denom = ftz(identical_log(denom))
                v = t29_add(v, t29_mul(weight, t29_sub(diff, log_denom)))
        var level = 0
        while (occupied & (UInt32(1) << UInt32(level))) != UInt32(0):
            d = t29_add(ds[level], d)
            h = t29_add(hs[level], h)
            v = t29_add(vs[level], v)
            occupied = occupied & ~(UInt32(1) << UInt32(level))
            level += 1
        ds[level] = d
        hs[level] = h
        vs[level] = v
        occupied = occupied | (UInt32(1) << UInt32(level))
        start += T29_PARTNER_CHUNK
    var d = Float32(0.0)
    var h = Float32(0.0)
    var v = Float32(0.0)
    var have_value = False
    for reverse in range(T29_CARRY_LEVELS):
        var level = T29_CARRY_LEVELS - 1 - reverse
        if (occupied & (UInt32(1) << UInt32(level))) != UInt32(0):
            if have_value:
                d = t29_add(d, ds[level])
                h = t29_add(h, hs[level])
                v = t29_add(v, vs[level])
            else:
                d = ds[level]
                h = hs[level]
                v = vs[level]
                have_value = True
    return SIMD[DType.float32, 4](d, h, v, Float32(0.0))


def t29_fold_lanes(mut lanes: InlineArray[Float32, T29_GROUP_LANES]) -> Float32:
    var stride = T29_GROUP_LANES // 2
    while stride > 0:
        for lane in range(stride):
            lanes[lane] = t29_add(lanes[lane], lanes[lane + stride])
        stride //= 2
    return lanes[0]
