# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Fixed-point accumulation, because Metal has no float atomic add.

NO CATBOOST COUNTERPART. This is in `original/` because CatBoost never has
to solve it: it flushes histograms with `atomicAdd` on `float` and accepts a
non-deterministic answer. Metal has no such instruction, so the port needs an
accumulator CatBoost's source does not contain, and inventing one is the
largest piece of non-port work this tree requires.

WHY INTEGERS
------------
Integer addition is associative and exact, so a fixed-point accumulation is
order-independent: however the partial sums combine, across lanes, across
blocks, across vendors, the total is the same. That is what makes the
bit-identical column achievable at all, and it is why the `IDENTICAL` mode
REPLACES the accumulator rather than configuring it (`original/numerics.mojo`).

THE SCALE, WHICH IS THE WHOLE PROBLEM
-------------------------------------
A gradient is a float and a slot is an `Int32`, so we multiply by a scale and
truncate. Choose the scale too small and every small gradient rounds to zero;
too large and a partial sum overflows and the answer is garbage rather than
merely imprecise. Overflow is the unacceptable failure, so the scale must be
chosen to make it IMPOSSIBLE, not unlikely.

The bound that makes it possible: **any leaf's rows are a subset of all
rows**, so the sum of magnitudes over any leaf is at most the sum of
magnitudes over the whole dataset. Compute that global sum once on the host,
and a scale that keeps IT inside `Int32` keeps every partial sum inside
`Int32`, at every node, at every depth, for the whole fit.

That is a genuinely stronger guarantee than a float accumulator gives, and it
costs one pass over the gradients per round.
"""


#: `Int32` holds up to 2^31 - 1. Reserve three bits of headroom so that the
#: sum, the sibling subtraction that follows it, and a rounding term cannot
#: reach the boundary even in the worst case.
comptime SCALE_HEADROOM_BITS = 3

#: The largest magnitude an accumulated plane may reach.
comptime SCALE_LIMIT = Float64((1 << (31 - SCALE_HEADROOM_BITS)) - 1)


def choose_scale(
    sum_of_magnitudes: Float64, row_count: Int = 0
) raises -> Float64:
    """The multiplier that makes overflow impossible for this round.

    `sum_of_magnitudes` is `sum over all rows of abs(value)` for the plane
    being accumulated, computed on the host before the round. Every partial
    sum the device forms is over a SUBSET of those rows, so its magnitude is
    at most this, and scaling it to the limit bounds every slot.

    `row_count` sharpens the bound. The blanket three headroom bits exist
    to absorb, among other things, the dithered quantizer's worst case of
    +1 unit per row (`histogram_utils.hist2_quantize`) at ANY row count.
    When the caller states its row count the allowance becomes exact:

        cell  <  sum_of_magnitudes * scale  +  row_count
              <=  (2^30 - 1 - row_count)    +  row_count   =  2^30 - 1

    which keeps one full bit of safety under Int32 and buys a scale 4x
    finer than the blanket limit. RESOLUTION IS NOT A LUXURY HERE: at 254
    borders on 800k x 100 (bench/interleaved, 2026-08-20) the blanket
    scale's dither noise flipped near-tied splits and cost 1.6% train mse
    against CatBoost (0.14375 vs 0.14145), while the row-count-aware scale
    reproduces CatBoost's mse to 7 decimals at the same speed. Callers
    that cannot state a row count get the old blanket limit, never a
    weaker scale than before.

    Returns a scale of 1.0 for an all-zero plane rather than dividing by
    zero: the accumulation is then exactly zero at any scale.
    """
    if sum_of_magnitudes < 0.0:
        raise Error(
            "sum_of_magnitudes is a sum of absolute values and cannot be"
            " negative; got "
            + String(sum_of_magnitudes)
        )
    if sum_of_magnitudes == 0.0:
        return 1.0
    var limit: Float64
    if row_count > 0:
        limit = Float64((1 << 30) - 1 - row_count)
        if limit < SCALE_LIMIT:
            limit = SCALE_LIMIT
    else:
        limit = SCALE_LIMIT

    # SNAPPED DOWN TO A POWER OF TWO (2026-08-21). A continuous scale is a
    # LEVER ARM for the last bits of `sum_of_magnitudes`: the dithered
    # quantizer's whole realization -- every cell's +-1 -- is a function of
    # the scale, so a 1e-6 wobble in the magnitude reduce re-rolled every
    # dither draw and could flip a knife-edge split. Measured the day this
    # landed: replacing the float-atomic magnitude reduce with an exact
    # fixed-order fold moved the magnitude by 1.3e-6 relative (the NEW
    # value was the correct one to seven digits against a float64 host
    # sum) and the 800k x 100 @ 128 model moved 2.7% train mse off its
    # CatBoost-matching realization. Snapping makes the scale a STEP
    # function of the magnitude -- identical bits for any magnitude within
    # a 2x band -- so the realization is pinned across runs, reduce
    # implementations, and platforms. Costs at most one bit of resolution;
    # the bound only gets safer (the snap is downward).
    # EXACT FORM (2026-08-21): the snap used to compare `snapped` against
    # the ROUNDED quotient `limit / mag`, whose last ulp could flip the
    # chosen power at an exact-ratio boundary depending on the platform's
    # division. The comparison is now `mag * snapped <= limit`, in which
    # every operation is exact (a power-of-two multiply reshuffles the
    # exponent, and `limit < 2^30` is exactly representable), so the
    # chosen power is a pure function of the magnitude's BITS -- which is
    # what lets `choose_scale_kernel` reproduce this host function
    # bit-for-bit on a device with no Float64. Identical result except
    # within one f64 ulp of an exact power-of-two ratio, where THIS form
    # is the correct one.
    var snapped = 1.0
    if sum_of_magnitudes <= limit:
        while sum_of_magnitudes * (snapped * 2.0) <= limit:
            snapped *= 2.0
    else:
        while sum_of_magnitudes * snapped > limit:
            snapped /= 2.0
    return snapped


def quantize(value: Float64, scale: Float64) -> Int32:
    """One value into its fixed-point slot.

    Truncation toward zero, not rounding: it is what makes the quantization
    a deterministic function of the input with no tie-breaking rule to get
    wrong on a different vendor.
    """
    return Int32(value * scale)


def dequantize(total: Int64, scale: Float64) -> Float64:
    """A finished accumulation back to a float.

    Takes `Int64` because a caller may widen before summing planes; the
    device slots are `Int32` and the scale guarantees they stay so.
    """
    return Float64(total) / scale


def max_representable(scale: Float64) raises -> Float64:
    """The largest magnitude this scale can hold. For a caller that wants to
    assert its own bound rather than trust the derivation above."""
    if scale <= 0.0:
        raise Error("scale must be positive; got " + String(scale))
    return SCALE_LIMIT / scale
