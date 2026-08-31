# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Fixed-point label accumulation. DEVIATION 135, ruled and closed.

Andrew, 2026-08-21: **fixed point.** This file is the ruling.

WHY THE QUESTION EXISTED
------------------------
There is no `float64` on device. sklearn accumulates label sums in `float64`
throughout (`_criterion.pyx`, `sum_left`/`sum_right` are `float64_t[::1]`), and
a parallel float reduction has no fixed order, so a `Float32` accumulator makes
the device answer a function of how the reduction happened to be shaped. That
would weaken every downstream check from "exact, per cell" to "within eps", and
it would make the same fit on Metal, CUDA and HIP three different models.

Integer addition is associative and exact, so a fixed-point accumulation is
order-independent by construction: however the partial sums combine — across
lanes, across blocks, across vendors — the total is the same bits.

THIS IS A PORT OF A PRECEDENT IN THIS REPOSITORY, NOT A NEW IDEA
-----------------------------------------------------------------
`mojo_only/fixed_point.mojo` at the repository root already solves this for the
GBDT learner, where CatBoost flushes histograms with a float `atomicAdd` that
Metal does not have. Three of its four ideas are taken unchanged, with its
reasoning:

1. **The bound comes from the whole dataset, once.** Any node's rows are a
   SUBSET of all rows, so the sum of magnitudes over any node is at most the
   sum over the dataset. Compute that once on the host and a scale that keeps
   IT inside the slot keeps every partial sum inside the slot, at every node,
   at every depth, for the whole fit. Overflow becomes impossible rather than
   unlikely, which is the only acceptable standard: an overflowed accumulator
   is garbage, not merely imprecise.
2. **The scale is SNAPPED DOWN to a power of two.** A continuous scale is a
   lever arm for the last bits of the magnitude sum: root's measured case was a
   1.3e-6 relative wobble in the magnitude reduce moving a model 2.7% in train
   MSE. Snapping makes the scale a step function of the magnitude — identical
   bits for any magnitude inside a 2x band — so the realization is pinned
   across runs, reduce implementations and platforms. It costs at most one bit
   and the bound only gets safer, because the snap is downward.
3. **Truncation toward zero, not rounding.** A deterministic function of the
   input with no tie-breaking rule to get wrong on another vendor.

WHAT THIS FILE HAS TO DERIVE FOR ITSELF, AND IT IS NOT A DETAIL
----------------------------------------------------------------
Root's file targets a fixed `Int32` slot with three headroom bits. That bound
is about the ACCUMULATOR. This lane has a second, tighter constraint that root
does not: DEVIATION 144's exact comparison. Two candidate splits are ordered by
cross-multiplying the MSE proxy `sum_L^2/n_L + sum_R^2/n_R`, i.e.

    num = sum_L^2 * n_R + sum_R^2 * n_L        den = n_L * n_R
    a beats b  <=>  num_a * den_b > num_b * den_a

and that product, not the accumulator, is what has to fit in `Int128`. Writing
`M` for the largest scaled sum magnitude (`M = 2^b`) and `n` for the node's row
count:

    num <= 2 * M^2 * n          den <= n^2 / 4
    num * den <= M^2 * n^3 / 2

so the requirement `num * den <= 2^127` becomes

    2*b + 3*log2(n) <= 128

**A scale chosen only for the accumulator overflows the comparator.** At the
`2^30` slot root uses and this lane's classification row cap of `2^26`, the
product reaches `2^137` — ten bits past `Int128`. So `accumulator_bits_for`
below spends the accumulator's resolution against the row count, keeping the
full 30 bits up to about 4 million rows in one node and giving bits back
gradually above that. That is a real cost of the exactness and it is priced
here rather than discovered later.
"""

from std.testing import assert_true


comptime SLOT_BITS = 30
"""The hard cap on a scaled sum's magnitude, `2^SLOT_BITS`.

Root's file reserves three headroom bits under `Int32` for the sum, the sibling
subtraction and a rounding term. The same reservation applies here for the same
reasons, and it is also what lets a device accumulator be an `Int32` — which
matters because `Int32` atomics exist everywhere and 64-bit ones do not.
"""

comptime MIN_ACCUMULATOR_BITS = 16
"""Below this the quantization is too coarse to be worth calling exact, and a
caller is better served by an error than by a silently useless model."""


def ceil_log2(value: Int) raises -> Int:
    """Smallest `k` with `2^k >= value`. Exact integer arithmetic, no `log`.

    Deliberately not `Int(ceil(log2(x)))`: this repository has already been
    bitten once by `std.math.log` carrying ~5e-8 of absolute error and silently
    re-deciding a tie. A bound must not depend on a transcendental.
    """
    if value <= 0:
        raise Error("ceil_log2 needs a positive value; got " + String(value))
    var k = 0
    var acc = 1
    while acc < value:
        acc *= 2
        k += 1
    return k


def accumulator_bits_for(row_count: Int) raises -> Int:
    """How many bits a scaled sum may occupy, given the node's row count.

    From the derivation in the module docstring: the exact comparator needs
    `2*b + 3*log2(n) <= 128`, and two bits are held back so the bound is a
    strict inequality with room for the sign and for `num`'s factor of two.

        b = min(SLOT_BITS, (128 - 3*ceil_log2(n) - 2) / 2)

    Worked, so the cost is visible rather than implied:

        n <= 2^22  (4.2M)   ->  b = 30   (the slot cap; nothing is lost)
        n  = 2^24  (16.8M)  ->  b = 27
        n  = 2^26  (67M)    ->  b = 24

    Raises rather than returning a useless scale when the row count would push
    `b` below `MIN_ACCUMULATOR_BITS`.
    """
    if row_count <= 0:
        raise Error(
            "accumulator_bits_for needs a positive row count; got "
            + String(row_count)
        )
    var budget = (128 - 3 * ceil_log2(row_count) - 2) // 2
    var bits = budget if budget < SLOT_BITS else SLOT_BITS
    if bits < MIN_ACCUMULATOR_BITS:
        raise Error(
            "row count "
            + String(row_count)
            + " leaves only "
            + String(bits)
            + " bits of accumulator, below the "
            + String(MIN_ACCUMULATOR_BITS)
            + "-bit floor: the exact split comparison cannot be kept at this"
            " scale. See DEVIATION 135."
        )
    return bits


def choose_scale(sum_of_magnitudes: Float64, row_count: Int) raises -> Float64:
    """The power-of-two multiplier that makes overflow impossible for this fit.

    `sum_of_magnitudes` is `sum over ALL rows of abs(y)`, computed once on the
    host before the fit. Every partial sum any node forms is over a subset of
    those rows, so its magnitude is at most this.

    Returns `1.0` for an all-zero target rather than dividing by zero: the
    accumulation is exactly zero at any scale.
    """
    if sum_of_magnitudes < 0.0:
        raise Error(
            "sum_of_magnitudes is a sum of absolute values and cannot be"
            " negative; got "
            + String(sum_of_magnitudes)
        )
    if sum_of_magnitudes == 0.0:
        return 1.0

    var bits = accumulator_bits_for(row_count)
    # `- row_count` is root's sharpening, for the same reason: truncation can
    # lose up to one unit per row, so allowing one unit per row makes the
    # bound exact instead of merely probable.
    var limit = Float64((1 << bits) - 1 - row_count)
    if limit <= 0.0:
        raise Error(
            "row count "
            + String(row_count)
            + " exceeds the accumulator range at "
            + String(bits)
            + " bits"
        )
    # SNAP DOWN to a power of two. See the module docstring: a continuous
    # scale is a lever arm for the last bits of `sum_of_magnitudes`.
    #
    # THE COMPARISON IS EXACT, AND THAT IS NOT COSMETIC. This used to compute
    # `raw = limit / sum_of_magnitudes` and snap against that quotient, whose
    # last ulp can flip the chosen power at an exact-ratio boundary depending
    # on the platform's division. Written as `mag * snapped <= limit` every
    # operation is exact -- multiplying by a power of two only reshuffles the
    # exponent, and `limit < 2^30` is exactly representable -- so the chosen
    # power is a pure function of the magnitude's BITS. That is what a device
    # implementation with no `Float64` would have to reproduce, and it is what
    # makes the scale identical across platforms rather than merely close.
    #
    # Adopted from the root `mojo_only/fixed_point.mojo`, which took the same
    # refinement on 2026-08-21 for the same reason; this file's copy had the
    # rounded-quotient form until now.
    var snapped = 1.0
    if sum_of_magnitudes <= limit:
        while sum_of_magnitudes * (snapped * 2.0) <= limit:
            snapped *= 2.0
    else:
        while sum_of_magnitudes * snapped > limit:
            snapped /= 2.0
    return snapped


def quantize(value: Float64, scale: Float64) -> Int64:
    """One label into its fixed-point slot. Truncation toward zero.

    Returns `Int64` because the caller may accumulate into a wider type than a
    device slot; the SCALE is what guarantees the accumulated total still fits
    `2^SLOT_BITS`, not the return type.
    """
    return Int64(value * scale)


def dequantize(total: Int64, scale: Float64) -> Float64:
    """A finished accumulation back to a float."""
    return Float64(total) / scale


def max_representable(scale: Float64, row_count: Int) raises -> Float64:
    """The largest magnitude sum this scale can hold, for a caller that wants
    to assert its own bound rather than trust the derivation."""
    if scale <= 0.0:
        raise Error("scale must be positive; got " + String(scale))
    var bits = accumulator_bits_for(row_count)
    return Float64((1 << bits) - 1 - row_count) / scale


def comparator_product_fits(row_count: Int) raises -> Bool:
    """Does the exact MSE comparison fit `Int128` at this row count?

    Computes the worst case in `Int128` rather than asserting the algebra:
    `num <= 2 * M^2 * n` and `den <= n^2/4` with `M = 2^accumulator_bits_for`,
    then checks `num * den` against `2^127 - 1`. The derivation in the module
    docstring says it must; this function is what makes that a checked claim
    instead of a comment.
    """
    var bits = accumulator_bits_for(row_count)
    var m = Int128(1) << Int128(bits)
    var n = Int128(row_count)
    var num = Int128(2) * m * m * n
    var den = (n * n) // Int128(4)
    if den < Int128(1):
        den = Int128(1)
    var limit = (Int128(1) << Int128(126))
    # 2^126 rather than 2^127-1: one full bit of headroom, so that a caller
    # who adds a term to `num` later does not silently cross the boundary.
    return num * den <= limit


def mse_proxy_exact(
    sum_left: Int64, n_left: Int, sum_right: Int64, n_right: Int
) -> Tuple[Int128, Int128]:
    """sklearn's MSE proxy as an exact rational over FIXED-POINT sums.

    `_criterion.pyx:944-973` is `sum_left^2/n_left + sum_right^2/n_right`.
    Over integer sums that is exactly

        num = sum_left^2 * n_right + sum_right^2 * n_left
        den = n_left * n_right

    Returns `(num, den)` with `den >= 0`. An empty child gives `den == 0`,
    which `compare_mse_proxy_exact` treats as invalid — sklearn cannot reach
    that case because `min_samples_leaf >= 1`, and cuML's float form does not
    guard it (`objectives.cuh:57` divides by `nLeft` and yields `+inf`).
    """
    var sl = Int128(sum_left)
    var sr = Int128(sum_right)
    var nl = Int128(n_left)
    var nr = Int128(n_right)
    return (sl * sl * nr + sr * sr * nl, nl * nr)


def compare_mse_proxy_exact(
    a_num: Int128, a_den: Int128, b_num: Int128, b_den: Int128
) -> Int:
    """Order two candidates by their exact MSE proxy. `1` if a beats b, `-1` if
    b beats a, `0` if they tie EXACTLY.

    `a > b  <=>  a_num/a_den > b_num/b_den  <=>  a_num*b_den > b_num*a_den`,
    valid because both denominators are strictly positive. An invalid
    candidate (`den == 0`) loses to any valid one and ties with another
    invalid one.

    The cross-multiplication is the reason `accumulator_bits_for` exists; see
    the module docstring.
    """
    var a_valid = a_den > Int128(0)
    var b_valid = b_den > Int128(0)
    if not a_valid and not b_valid:
        return 0
    if not a_valid:
        return -1
    if not b_valid:
        return 1
    var left = a_num * b_den
    var right = b_num * a_den
    if left > right:
        return 1
    if left < right:
        return -1
    return 0
