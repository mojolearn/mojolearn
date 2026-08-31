# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The planted samples, ASSEMBLED FROM BITS, each with a job it alone does.

NOT A PORT. IDENTITY_PATHS row 32's lesson applies verbatim, and
`kde/original/kde_fixture.mojo` states it: a host `target += v * w` chain is a
contraction decision, so a fixture built by host arithmetic can hand two
machines different inputs before the first kernel runs. `mix64` and
`bits_value` below are that file's construction, unchanged and cited rather
than re-derived -- splitmix64 into a float32 by BITCAST, hashed mantissa,
exponent 126 or 127, sign from the hash. No floating-point operation is
performed on a hashed value.

THE SMALL INTEGER FIXTURES ARE DIFFERENT, AND DELIBERATELY SO. `FIX_ANALYTIC`
and `FIX_DUPES` are written-out small integers and halves, because their whole
value is that a reader can work the answer out on paper. Every value in them
is exactly representable in float32 and every intermediate the pinned tree
forms from them is too (the arithmetic bound is worked in `FIX_ANALYTIC`'s
docstring), so "assembled from bits" is satisfied by the values BEING exact
rather than by a bitcast.

WHY EACH FIXTURE EXISTS -- one line, and if a fixture ever stops doing its
job it should be deleted rather than kept for reassurance:

  FIX_ANALYTIC     every replicate's answer is derivable by hand, three ways
  FIX_SEPARABLE    the permutation p-value must sit on its floor
  FIX_SAME         the null must be roughly uniform (a REPORT, not a gate)
  FIX_DUPES        exact duplicates, so the resample tie path is reached
  FIX_SIGNED_ZERO  both zeros in the sample (row 39)
  FIX_HASHED       a general sample nothing about which is special
"""

from std.memory import bitcast


comptime FIX_ANALYTIC = 0
comptime FIX_SEPARABLE = 1
comptime FIX_SAME = 2
comptime FIX_DUPES = 3
comptime FIX_SIGNED_ZERO = 4
comptime FIX_HASHED = 5
comptime FIX_COUNT = 6


def fixture_name(fix: Int) -> String:
    if fix == FIX_ANALYTIC:
        return String("analytic")
    if fix == FIX_SEPARABLE:
        return String("separable")
    if fix == FIX_SAME:
        return String("same-distribution")
    if fix == FIX_DUPES:
        return String("duplicates")
    if fix == FIX_SIGNED_ZERO:
        return String("signed-zero")
    if fix == FIX_HASHED:
        return String("hashed")
    return String("?")


# ===========================================================================
# The hashed construction (kde/original/kde_fixture.mojo, unchanged)
# ===========================================================================


def mix64(a: Int, b: Int, salt: Int) -> UInt64:
    """splitmix64 over three integers. The same function
    `kde/original/kde_fixture.mojo::mix64` uses, spelled the same way."""
    var z = (
        UInt64(a + 1) * 0x9E3779B97F4A7C15
        + UInt64(b + 1) * 0xBF58476D1CE4E5B9
        + UInt64(salt + 1) * 0x94D049BB133111EB
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return z


def bits_value(z: UInt64, signed: Bool) -> Float32:
    """A float32 from bits: hashed mantissa, exponent 126 or 127 so `|v|` is
    in `[0.5, 2)`, sign from the hash when `signed`."""
    var mant = UInt32(z & 0x7FFFFF)
    var expo = UInt32(126 + Int((z >> 23) & 1))
    var sign = UInt32(0)
    if signed and ((z >> 24) & 1) == 1:
        sign = UInt32(1)
    return bitcast[DType.float32]((sign << 31) | (expo << 23) | mant)


# ===========================================================================
# Shapes
# ===========================================================================


def fixture_n(fix: Int) -> Int:
    """Rows in the ONE-SAMPLE fixtures (`bootstrap`)."""
    if fix == FIX_ANALYTIC:
        return 8
    if fix == FIX_DUPES:
        return 12
    if fix == FIX_SIGNED_ZERO:
        return 8
    if fix == FIX_HASHED:
        return 200
    return 32


def fixture_d(fix: Int) -> Int:
    """Columns. EVERY fixture is two-column, so every statistic arm --
    including `pearson` and `diff_means`, which read column 1 -- can run on
    every fixture without a second fixture family. A one-column fixture
    would make `stat_columns_needed`'s refusal the only thing those arms
    ever saw."""
    return 2


def fixture_two_sample_n(fix: Int) -> Int:
    """`n_x == n_y` for the permutation fixtures.

    16 and 16 is not a round number chosen for looks: `C(32, 16)` is
    601,080,390, so at the gates' 999 permutations the chance that a drawn
    permutation happens to reproduce the observed split -- which would put
    `check_permutation_separable`'s p-value one step ABOVE its floor, legally
    -- is about 1.7e-6. At `n_x = n_y = 8` it is `C(16,8) = 12870` and the
    same accident has probability 0.078, which would make the gate flap.
    """
    return 16


# ===========================================================================
# FIX_ANALYTIC -- the fixture a reader can check on paper
# ===========================================================================


def analytic_sample() -> List[Float32]:
    """Column 0 is `1, 2, ..., 8`; column 1 is `9 - column 0`.

    THREE HAND-DERIVED FACTS, and they are the reason this fixture exists.
    Every one of them is a statement about EVERY REPLICATE, not about the
    distribution's average, so none of them is a tolerance.

    (1) `mean`. A resample of 8 draws from `{1..8}` sums to an integer `k` in
        `[8, 64]`, so the replicate mean is EXACTLY `k/8` -- a multiple of
        0.125 in `[1, 8]`. Exact in float32 (`k <= 64`, and `/8` is an
        exponent shift). `check_bootstrap_known_answer` asserts the whole
        bootstrap distribution consists of multiples of 0.125 in that range.

    (2) `pearson`. Column 1 is an exact affine image of column 0 with a
        NEGATIVE slope, so every resample with at least two distinct rows has
        correlation EXACTLY `-1.0` (`0xbf800000`). The arithmetic is exact all
        the way: the resample mean `k/8` is exact; deviations are multiples of
        0.125 bounded by 7, so exact; their squares are multiples of 1/64
        bounded by 49, i.e. at most `49*64 = 3136 < 2^24`, so exact; the sum
        of eight of them is exact; `syy == sxx` and `sxy == -sxx` bit for bit
        because each term is the exact negation of its partner and the tree
        adds them in the same order; and `sqrt(fl(s*s)) == |s|` for every
        float32 `s` away from overflow and underflow (correctly-rounded
        multiply and sqrt), so the denominator is `sxx` exactly and the
        quotient is `-1.0` exactly. A resample that draws ONE row eight times
        has `sxx == 0` and is DEVIATION 1696's canonical NaN; it happens with
        probability `8 * 8^-8 = 4.8e-7` per replicate and the gate accepts
        exactly those two bit patterns and no third.

    (3) `diff_means`. `mean(col1) = 9 - mean(col0)`, so the paired difference
        is `2k/8 - 9`, an exact multiple of 0.25 in `[-7, 7]`.

    The ANALYTIC bootstrap moments, for the REPORT beside the assertions: the
    exact (all `n^n` resamples) bootstrap distribution of the mean has
    expectation `xbar = 4.5` and variance `sigma_hat^2 / n = 5.25/8 =
    0.65625`, so the standard error converges to `sqrt(0.65625) =
    0.8100925...`. A Monte Carlo bootstrap of `R` replicates estimates those
    with error of order `1/sqrt(R)`; the check reports the gap and asserts
    only a wide band, because asserting a tight one would be asserting a
    rounding accident.
    """
    var out = List[Float32]()
    for i in range(8):
        out.append(Float32(i + 1))
        out.append(Float32(9 - (i + 1)))
    return out^


# ===========================================================================
# FIX_DUPES -- exact duplicates, so the resample tie path is reached
# ===========================================================================


def dupes_sample() -> List[Float32]:
    """Twelve rows over three DISTINCT values, four rows each: `1.0`, `2.0`,
    `5.0` in column 0, and column 1 the same value again.

    WHAT IT REACHES that no hashed fixture does: two different draw positions
    returning DIFFERENT indices but the SAME value. That is the tie the
    percentile path has to order (the sorted replicate has runs of equal
    keys, so the radix sort's stability is exercised rather than assumed) and
    it is the tie the order statistic has to interpolate BETWEEN TWO EQUAL
    VALUES, where `identical_mul_add(frac, a - a, a)` must return `a`
    regardless of `frac`.

    The replicate mean is `k/12` for an integer `k` in `[12, 60]`, which is
    NOT exact in float32 -- deliberately, so the gate is not accidentally
    reading the same exactness `FIX_ANALYTIC` provides.
    """
    var vals = [Float32(1.0), Float32(2.0), Float32(5.0)]
    var out = List[Float32]()
    for i in range(12):
        var v = vals[i // 4]
        out.append(v)
        out.append(v)
    return out^


# ===========================================================================
# FIX_SIGNED_ZERO -- row 39
# ===========================================================================


def signed_zero_sample() -> List[Float32]:
    """`[-0.0, +0.0, -0.0, +0.0, 1.0, -1.0, 2.0, -2.0]` in both columns, the
    two zeros written BY BITS so no host arithmetic can normalise them away.

    WHAT IT IS FOR, and what it is honestly NOT for. Running it end to end
    shows that no `-0.0` reaches a recorded stage through the fold
    statistics, which is a PROOF this fixture makes checkable rather than a
    property it discovers:

      * every fold is seeded `+0.0` and padded with `+0.0`, and
        `(-0.0) + (+0.0) == +0.0`, so a replicate drawing only zeros sums to
        `+0.0`;
      * `x - x` is `+0.0` under round-to-nearest for every finite `x`, so
        `diff_means` of two equal means is `+0.0` and `basic_interval`'s
        reflection cannot produce `-0.0` either;
      * `quantile_interpolate` is `fma(frac, a_hi - a_lo, a_lo)`; with
        `a_lo == a_hi == -0.0` that is `fma(frac, +0.0, -0.0) == +0.0`.

    So the SAMPLE carries both zeros and the DISTRIBUTION cannot. The
    property that a `-0.0` really is ordered below a `+0.0` is therefore
    checked by PLANTING a distribution (`planted_signed_zero_distribution`)
    into the real sort and interval path, exactly as
    `kde/original/kde_check.mojo` plants rows into the real
    `logsumexp_kernel` because no legal input mixes the zeros there either.
    """
    var bits = [
        UInt32(0x80000000),
        UInt32(0x00000000),
        UInt32(0x80000000),
        UInt32(0x00000000),
        UInt32(0x3F800000),
        UInt32(0xBF800000),
        UInt32(0x40000000),
        UInt32(0xC0000000),
    ]
    var out = List[Float32]()
    for i in range(8):
        var v = bitcast[DType.float32](bits[i])
        out.append(v)
        out.append(v)
    return out^


def planted_signed_zero_distribution(n_resamples: Int) -> List[Float32]:
    """A bootstrap distribution PLANTED with both zeros and a run of
    duplicates, for the sort and the interval to order.

    Positions 0 and 1 hold `+0.0` then `-0.0` (the WRONG order, so a sort
    that did nothing would be visible), positions 2 and 3 hold `-0.0` then
    `+0.0` (the right order, so a sort that reversed would be visible),
    the next four hold `1.0` four times (a tie run), and the rest are hashed.
    CUB's `TwiddleIn` (`core/segmented_sort.mojo::float_to_sortable`) maps
    `-0.0` to `0x7FFFFFFF` and `+0.0` to `0x80000000`, so `-0.0` sorts FIRST
    and the two of them are distinct KEYS even though they compare equal as
    floats. That is the property this plant exists to make observable.
    """
    var out = List[Float32]()
    out.append(bitcast[DType.float32](UInt32(0x00000000)))
    out.append(bitcast[DType.float32](UInt32(0x80000000)))
    out.append(bitcast[DType.float32](UInt32(0x80000000)))
    out.append(bitcast[DType.float32](UInt32(0x00000000)))
    for _ in range(4):
        out.append(Float32(1.0))
    for i in range(8, n_resamples):
        out.append(bits_value(mix64(i, 3, 91), True))
    return out^


# ===========================================================================
# The hashed one-sample fixture
# ===========================================================================


def hashed_sample(n: Int, salt: Int) -> List[Float32]:
    """`n` rows, two columns, every cell an independent hash. Non-uniform and
    non-repeating per cell, so a permutation of rows or of columns changes
    the answer (`[[uniform-test-data-hides-permutation]]`)."""
    var out = List[Float32]()
    for i in range(n):
        out.append(bits_value(mix64(i, 0, salt), True))
        out.append(bits_value(mix64(i, 1, salt), True))
    return out^


def build_sample(fix: Int) -> List[Float32]:
    """The one-sample fixture, row major, `fixture_n(fix) x 2`."""
    if fix == FIX_ANALYTIC:
        return analytic_sample()
    if fix == FIX_DUPES:
        return dupes_sample()
    if fix == FIX_SIGNED_ZERO:
        return signed_zero_sample()
    return hashed_sample(fixture_n(fix), 17)


# ===========================================================================
# The two-sample (permutation) fixtures
# ===========================================================================


def separable_pair() -> List[Float32]:
    """The POOLED sample: 16 values at `+100 + i`, then 16 at `-100 - i`.

    SEPARABLE in the strong sense the gate needs: every value of the first
    group exceeds every value of the second by more than 200, so the observed
    difference of means is the LARGEST difference any split of these 32
    values admits, uniquely. Every permutation that moves even one value
    across the split strictly lowers it, so `count(null >= observed - gamma)`
    is 0 unless a drawn permutation reproduces the observed split exactly.

    `gamma` is `1.19e-5 * observed`, about 0.0024 here, and the second
    largest attainable difference is `observed - 2*(200 + ...)/16`, i.e.
    smaller by more than 25. So the tolerance cannot smear the two together,
    which is the failure mode that would make this gate pass for the wrong
    reason.
    """
    var out = List[Float32]()
    for i in range(16):
        out.append(Float32(100 + i))
    for i in range(16):
        out.append(Float32(-100 - i))
    return out^


def same_distribution_pair(salt: Int) -> List[Float32]:
    """32 pooled values from ONE hashed population, split 16/16.

    The two groups are drawn from the same construction with the same salt
    and differ only in position, so the null hypothesis is TRUE by
    construction and the p-value should be roughly uniform on `[0, 1]`
    across seeds. `check_permutation_null_is_uniform` prints that as a
    REPORT: a distributional claim needs many seeds and a real test, and
    asserting a band on one draw would be asserting a coin flip.
    """
    var out = List[Float32]()
    for i in range(32):
        out.append(bits_value(mix64(i, 5, salt), True))
    return out^


def build_pooled(fix: Int, salt: Int) -> List[Float32]:
    if fix == FIX_SEPARABLE:
        return separable_pair()
    return same_distribution_pair(salt)


# ===========================================================================
# The Monte Carlo boxes
# ===========================================================================


def unit_square_lower() -> List[Float32]:
    """`[0.0, 0.0]`."""
    return [Float32(0.0), Float32(0.0)]


def unit_square_upper() -> List[Float32]:
    """`[1.0, 1.0]`. Volume exactly 1, so a wrong volume factor cannot hide
    behind a rounding."""
    return [Float32(1.0), Float32(1.0)]


def shifted_box_lower() -> List[Float32]:
    """`[-2.0, 0.5]`, a box that is neither the unit square nor centred, so
    the affine map's `lo` term is load-bearing and a dropped `+ lower` shows
    up rather than cancelling."""
    return [Float32(-2.0), Float32(0.5)]


def shifted_box_upper() -> List[Float32]:
    """`[2.0, 2.5]`. Spans 4 and 2, volume exactly 8."""
    return [Float32(2.0), Float32(2.5)]
