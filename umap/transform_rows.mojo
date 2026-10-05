# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The UMAP transform's per-row statements, shared by the device kernels
(`umap/transform.mojo`) and the host column (`umap/host/umap_oracle.mojo`)
(lane cpu4-umap, 2026-10-04).

The transform used to run entirely on the host in Float64 after the device
k-NN. Every query row is independent (training coordinates never move), so
each stage is one thread per row on the device. The binary64 steps (the
sigma bisection, the memberships' exp, the edge schedule's ratio and
products, the attraction term's pow) are `checks/soft_f64.mojo` (integer
instructions, correctly rounded; `sf64_exp` and `sf64_pow` are
`portable_exp64` and `portable_pow64` statement for statement), since the
Apple GPU has no float64. Float32 steps go through the pinned seams
(`identical_mul_add`, `identical_mul`, `identical_div`) under row 10's flush
model (`ftz` on every float32 result), as `umap/optimizer_identical_device.mojo`
does, so no subnormal reaches an Apple add or compare. Both columns call THESE
functions, so they compute the same words by construction.

Bits moved against the previous host-only transform: the float32 flush model
(subnormal operands and results only) and, in FAST, the binary64 arithmetic
(it used the host stdlib exp/pow; it is now the portable soft arithmetic in
every mode, which FAST permits: same quality, one code path). This module
imports nothing from `max.gpu`, so the host oracle may import it.

Statuses: each helper returns `TR_OK` or the code of the refusal the old host
code raised; `tr_raise` turns a code into that message.
"""
from std.memory import bitcast

from checks.numerics import ftz, identical_div, identical_mul, identical_mul_add
from checks.soft_f64 import (
    SF64_ONE,
    SF64_SIGN,
    SF64_ZERO,
    sf64_add,
    sf64_div,
    sf64_exp,
    sf64_from_f32,
    sf64_from_int,
    sf64_gt,
    sf64_is_nan,
    sf64_lt,
    sf64_mul,
    sf64_neg,
    sf64_pow,
    sf64_sub,
    sf64_to_f32,
    sf64_to_int,
)


comptime TR_F32P = MutPointer[Float32, MutAnyOrigin]
comptime TR_U32P = MutPointer[UInt32, MutAnyOrigin]
comptime TR_U64P = MutPointer[UInt64, MutAnyOrigin]

# Status codes, in the old host code's refusal order (they double as the
# device flag slots; `tr_raise` reads the lowest set one).
comptime TR_OK = -1
comptime TR_TRAIN_NONFINITE = 0
comptime TR_QUERY_NONFINITE = 1
comptime TR_EMBED_NONFINITE = 2
comptime TR_BAD_DISTANCE = 3
comptime TR_UNSORTED = 4
comptime TR_NO_MEMBERSHIP = 5
comptime TR_BAD_EDGE = 6
comptime TR_INIT_NONFINITE = 7
comptime TR_REFINE_DISTANCE = 8
comptime TR_GRADIENT = 9
comptime TR_OUT_NONFINITE = 10
comptime TR_FLAGS = 11

# binary64 words: 1e-5 (the sigma tolerance), 0.001 (the sigma floor's
# factor), 0.5, 2 and -1.
comptime _TR_TOL = UInt64(0x3EE4F8B588E368F1)
comptime _TR_MILLI = UInt64(0x3F50624DD2F1A9FC)
comptime _TR_HALF = UInt64(0x3FE0000000000000)
comptime _TR_TWO = UInt64(0x4000000000000000)
comptime _TR_NEG_ONE = UInt64(0xBFF0000000000000)

comptime TR_GRAD_CLIP = Float32(4.0)
comptime TR_GOLDEN = UInt64(0x9E3779B97F4A7C15)


def tr_raise(code: Int) raises:
    """HOST: the old host transform's message for a status code."""
    if code == TR_OK:
        return
    if code == TR_TRAIN_NONFINITE:
        raise Error("UMAP transform training input must be finite")
    if code == TR_QUERY_NONFINITE:
        raise Error("UMAP transform queries must be finite")
    if code == TR_EMBED_NONFINITE:
        raise Error("UMAP transform training embedding must be finite")
    if code == TR_BAD_DISTANCE:
        raise Error("UMAP transform neighbor distances must be finite and nonnegative")
    if code == TR_UNSORTED:
        raise Error("UMAP transform neighbors must be distance-sorted")
    if code == TR_NO_MEMBERSHIP:
        raise Error("UMAP transform query has no positive memberships")
    if code == TR_BAD_EDGE:
        raise Error("UMAP transform membership or neighbor index is invalid")
    if code == TR_INIT_NONFINITE:
        raise Error("UMAP transform initialization is not finite")
    if code == TR_REFINE_DISTANCE:
        raise Error("UMAP transform refinement distance is not finite")
    if code == TR_GRADIENT:
        raise Error("UMAP transform gradient is not finite")
    raise Error("UMAP transform returned non-finite coordinates")


def tr_target(k: Int) -> UInt64:
    """HOST: the sigma search's target `log2(k)` as a binary64 word (one
    scalar of `k`, the host seam both columns call)."""
    from checks.numerics import identical_log2_64

    return bitcast[DType.uint64](identical_log2_64(Float64(k)))


def tr_alpha(epoch: Int, epochs: Int, learning_rate: Float32) -> Float32:
    """HOST: the epoch's learning rate, one scalar per launch, the same
    statement on both columns."""
    return ftz(identical_mul(
        identical_mul(Float32(0.25), learning_rate),
        Float32(Float64(epochs - epoch) / Float64(epochs)),
    ))


@always_inline
def tr_finite(v: Float32) -> Bool:
    return (bitcast[DType.uint32](v) & UInt32(0x7F800000)) != UInt32(0x7F800000)


@always_inline
def tr_splitmix64(value: UInt64) -> UInt64:
    """`umap/optimizer.mojo::_splitmix64`, restated (integer only)."""
    var z = value + TR_GOLDEN
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


@always_inline
def tr_clip(value: Float32) -> Float32:
    if value > TR_GRAD_CLIP:
        return TR_GRAD_CLIP
    if value < -TR_GRAD_CLIP:
        return -TR_GRAD_CLIP
    return value


@always_inline
def tr_row_memberships(dp: TR_F32P, wp: TR_F32P, row: Int, k: Int, target: UInt64) -> Int:
    """Row `row`'s local-connectivity-zero memberships into `wp`.

    BATCH INVARIANCE: the sigma floor's mean is this row's own k distances.
    64 fixed bisection steps in binary64 (slot zero skipped, a zero distance
    counts 1), the converged value kept; sigma floored at 0.001 x mean; each
    membership `Float32(exp(-d / sigma))` (1 at d = 0), flushed."""
    var base = row * k
    for j in range(k):
        var d = ftz(dp[base + j])
        if not tr_finite(d) or d < Float32(0.0):
            return TR_BAD_DISTANCE
        if j > 0 and d < ftz(dp[base + j - 1]):
            return TR_UNSORTED
    var mean = SF64_ZERO
    for j in range(k):
        mean = sf64_add(mean, sf64_from_f32(ftz(dp[base + j])))
    mean = sf64_div(mean, sf64_from_int(k))
    var lo = SF64_ZERO
    var hi = _TR_NEG_ONE
    var sigma = SF64_ONE
    for _ in range(64):
        var total = SF64_ZERO
        for j in range(1, k):
            var d = sf64_from_f32(ftz(dp[base + j]))
            if (d & ~SF64_SIGN) == 0:
                total = sf64_add(total, SF64_ONE)
            else:
                total = sf64_add(total, sf64_exp(sf64_div(sf64_neg(d), sigma)))
        # Keep the converged value while retaining the fixed step count.
        var err = sf64_sub(total, target) & ~SF64_SIGN
        if not sf64_is_nan(err) and not sf64_gt(err, _TR_TOL):
            continue
        if sf64_gt(total, target):
            hi = sigma
            sigma = sf64_mul(sf64_add(lo, hi), _TR_HALF)
        else:
            lo = sigma
            if sf64_lt(hi, SF64_ZERO):
                sigma = sf64_mul(sigma, _TR_TWO)
            else:
                sigma = sf64_mul(sf64_add(lo, hi), _TR_HALF)
    var floor_ = sf64_mul(_TR_MILLI, mean)
    if sf64_gt(floor_, sigma):
        sigma = floor_
    var row_sum = Float32(0.0)
    for j in range(k):
        var d = ftz(dp[base + j])
        var w = Float32(1.0)
        if d != Float32(0.0):
            w = ftz(sf64_to_f32(sf64_exp(sf64_div(sf64_neg(sf64_from_f32(d)), sigma))))
        wp[base + j] = w
        row_sum = ftz(row_sum + w)
    if not tr_finite(row_sum) or not (row_sum > Float32(0.0)):
        return TR_NO_MEMBERSHIP
    return TR_OK


@always_inline
def tr_row_initialize(
    ip: TR_U32P, wp: TR_F32P, ep: TR_F32P, op: TR_F32P,
    row: Int, k: Int, n_train: Int, comps: Int,
) -> Int:
    """Row `row`'s starting coordinates into `op`: the training point of the
    first membership equal to 1, else the membership-weighted mean of the
    neighbors' training coordinates (each term one fma, flushed)."""
    var base = row * k
    for j in range(k):
        var w = wp[base + j]
        if ip[base + j] >= UInt32(n_train) or not tr_finite(w) or w < Float32(0.0) or w > Float32(1.0):
            return TR_BAD_EDGE
    var total = Float32(0.0)
    var exact = -1
    for j in range(k):
        var w = wp[base + j]
        total = ftz(total + w)
        if exact < 0 and w == Float32(1.0):
            exact = Int(ip[base + j])
    if not tr_finite(total) or not (total > Float32(0.0)):
        return TR_NO_MEMBERSHIP
    for c in range(comps):
        var value = Float32(0.0)
        if exact >= 0:
            value = ep[exact * comps + c]
        else:
            for j in range(k):
                var tail = Int(ip[base + j])
                value = ftz(identical_mul_add(
                    ftz(identical_div(wp[base + j], total)), ep[tail * comps + c], value
                ))
        if not tr_finite(value):
            return TR_INIT_NONFINITE
        op[row * comps + c] = value
    return TR_OK


@always_inline
def tr_row_refine_prep(
    ip: TR_U32P, wp: TR_F32P, sp: TR_U64P, kp: TR_U64P, row: Int, k: Int
) -> Int:
    """BATCH INVARIANCE: each edge's schedule ratio `w / max_row(w)` (binary64
    word into `sp`) uses THIS row's largest membership, and the row's
    negative-sample key (into `kp`) hashes its own k neighbor INDICES."""
    var base = row * k
    var maximum = Float32(0.0)
    var key = TR_GOLDEN
    for j in range(k):
        maximum = max(maximum, wp[base + j])
        key = tr_splitmix64(key ^ UInt64(ip[base + j]))
    if not tr_finite(maximum) or not (maximum > Float32(0.0)):
        return TR_NO_MEMBERSHIP
    var mx = sf64_from_f32(maximum)
    for j in range(k):
        sp[base + j] = sf64_div(sf64_from_f32(wp[base + j]), mx)
    kp[row] = key
    return TR_OK


@always_inline
def tr_row_refine_epoch(
    rp: TR_F32P, ep: TR_F32P, ip: TR_U32P, sp: TR_U64P, key: UInt64,
    row: Int, k: Int, comps: Int, n_train: Int,
    epoch: Int, draw_epoch: Int, alpha: Float32,
    a: Float32, b_word: UInt64, neg2ab: Float32, rep2b: Float32,
    seed: UInt64, negative_rate: Int,
) -> Int:
    """One epoch of row `row`'s refinement, in place in `rp` (the row owns
    its coordinates; training coordinates never move). Edge `j` is due when
    `Int((epoch + 1) s) > Int(epoch s)` in binary64; its tail attracts, then
    `negative_rate` SplitMix64 draws (counter keyed by `draw_epoch`, the
    row key and the slot) repel. The attraction's `d^b` is binary64 pow."""
    var base = row * k
    var now = sf64_from_int(epoch)
    var nxt = sf64_from_int(epoch + 1)
    for j in range(k):
        var edge = base + j
        var s = sp[edge]
        if sf64_to_int(sf64_mul(nxt, s)) <= sf64_to_int(sf64_mul(now, s)):
            continue
        var edge_key = key ^ (UInt64(j) * TR_GOLDEN)
        var tail = Int(ip[edge])
        for slot in range(negative_rate + 1):
            var other = tail
            if slot > 0:
                var counter = seed ^ (UInt64(draw_epoch) * UInt64(0xD1B54A32D192ED03)) ^ (
                    edge_key * UInt64(0x94D049BB133111EB)
                ) ^ UInt64(slot - 1)
                other = Int(tr_splitmix64(counter) % UInt64(n_train))
            var distance = Float32(0.0)
            for c in range(comps):
                var delta = ftz(rp[row * comps + c] - ep[other * comps + c])
                distance = ftz(identical_mul_add(delta, delta, distance))
            if not tr_finite(distance):
                return TR_REFINE_DISTANCE
            if not (distance > Float32(0.0)):
                continue
            var powered = ftz(sf64_to_f32(sf64_pow(sf64_from_f32(distance), b_word)))
            var denom = ftz(identical_mul_add(a, powered, Float32(1.0)))
            var coeff: Float32
            if slot == 0:
                coeff = identical_div(identical_mul(neg2ab, identical_div(powered, distance)), denom)
            else:
                coeff = identical_div(rep2b, identical_mul(ftz(Float32(0.001) + distance), denom))
            if not tr_finite(coeff):
                return TR_GRADIENT
            for c in range(comps):
                var at = row * comps + c
                var delta = ftz(rp[at] - ep[other * comps + c])
                rp[at] = ftz(identical_mul_add(alpha, tr_clip(ftz(identical_mul(coeff, delta))), rp[at]))
    return TR_OK


def tr_neg2ab(a: Float32, b: Float32) -> Float32:
    """HOST scalar: `-2 a b`, both columns."""
    return ftz(identical_mul(identical_mul(Float32(-2.0), a), b))


def tr_rep2b(repulsion_strength: Float32, b: Float32) -> Float32:
    """HOST scalar: `2 gamma b`, both columns."""
    return ftz(identical_mul(identical_mul(Float32(2.0), repulsion_strength), b))
