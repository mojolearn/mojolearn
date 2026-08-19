"""Fixed-point accumulation, because Metal has no float atomic add.

NO CATBOOST COUNTERPART. This is in `mojo_only/` because CatBoost never has
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
REPLACES the accumulator rather than configuring it (`mojo_only/numerics.mojo`).

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


def choose_scale(sum_of_magnitudes: Float64) raises -> Float64:
    """The multiplier that makes overflow impossible for this round.

    `sum_of_magnitudes` is `sum over all rows of abs(value)` for the plane
    being accumulated, computed on the host before the round. Every partial
    sum the device forms is over a SUBSET of those rows, so its magnitude is
    at most this, and scaling it to `SCALE_LIMIT` bounds every slot.

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
    return SCALE_LIMIT / sum_of_magnitudes


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
