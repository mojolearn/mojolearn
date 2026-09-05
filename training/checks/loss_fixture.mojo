# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The fixture of profile `mojolearn.identical.loss.ce.fp32.v1`.

**THIS BANNER WAS FALSE AND IS CORRECTED. COMPILED AND CONSUMED ON ONE
DEVICE.** Until 2026-08-31 this header read "NOTHING IN THIS FILE HAS EVER
BEEN COMPILED OR EXECUTED", added that no `mojo` process had read it and no
device had consumed a case from it, and called every "separates" and every
"is exact" below a PREDICTION. **Commit `ecd1a436` is the first execution of
this file, driven by `training/checks/loss_check.mojo`**, and it turned
those predictions into measurements: 24 cases and 61,925 cells device against
oracle BITWISE, plus 264 cells against a hand-written closed form with no
epsilon. Written 2026-08-25 by the training-gate lane, DEVIATIONS 1450-1469.

**Two limits survive the run and neither is a formality.** It ran on ONE
DEVICE and on no other vendor, and it ran at the DEFAULT vocabulary, not at
`V = 128256`, which is the shape the profile is actually for. See OWED item 5
at the bottom of this file.

WHAT IT IS FOR
---------------
The contract's section 10.1 lists twenty-five sabotage arms and, for
almost every one, the fixture property WITHOUT WHICH THE ARM IS BIT-INERT.
That column is the reason this file exists. An arm that fires on every
case is a smoke test; an arm that fires on ONE case and is provably inert
on ANOTHER is a reach proof, and only a case SET can express the second
one. So this file is a table of cases chosen so that each contested clause
has both halves.

The five the contract flags as most likely to be wrongly called inert, and
what in here answers each:

  * `L_SMOOTH_ALWAYS_SPELLED` needs a row whose `nll` is `-0.0`.
    `base_n1_v1` and `pair_n2_v1` are `V == 1` rows, where `shift[y]` is
    exactly `+0.0`, `denom` is exactly `1.0`, `logdenom` is exactly
    `+0.0`, `lp_y` is exactly `+0.0` and `neg_by_bits` of that is `-0.0`.
    `smooth_v1_eps` is the same shape at `EPS = 0.1` and is the INERT
    half, because the arm only spells a path that is already spelled.
  * `L_GRAD_DIVISOR_IS_N` needs an ignored row under MEAN.
    `ign_n6_v8_c3` has `N = 6` and `count = 3`; `base_n4_v8` has no
    ignored row at all and is the inert half.
  * `L_IGNORED_ROW_NEG_ZERO` must move `ce.row` and must NOT move
    `ce.total`. That is not a property of a case, it is a property of the
    GATE, and `loss_check.mojo` asserts both halves per stage.
  * `L_NEG_VIA_ZERO_SUB` needs `lp_y == +0.0`, which is the same `V == 1`
    row.
  * `L_MAX_PLAIN_COMPARE` needs a row carrying BOTH zero signs.
    `adv_signed_zeros` plants a `-0.0` beside a `+0.0`; every other case
    is asserted (not assumed -- `assert_no_zero_or_subnormal_logits`) to
    carry neither, which is what makes the inert half a measurement.

THE HASHED VALUES ARE BUILT SO THAT ZERO AND SUBNORMAL ARE UNREACHABLE
------------------------------------------------------------------------
DEVIATION 1450. `transformer/checks/transformer_fixture.mojo::
fixture_tensor` draws `lo + (hi - lo) * u` over a dyadic range, which CAN
land exactly on `+0.0` for one particular hash output. That is harmless
there and it is NOT harmless here, because this lane's inert halves are
statements of the form "no row of this case carries both zero signs" and a
stray zero from the hash would silently make an inert assertion false --
and it would do it on ONE case out of twenty-five, which is the hardest
kind of flake to read.

So `ce_hashed_logit` builds `+-(1 + m) * 2^e` with `m` a 23-bit fraction
and `e` in `[-3, 4]`. Three properties, and all three are structural
rather than statistical:

  1. **Exactly representable.** `1 + m` at 23 fraction bits is exact in
     `[1, 2)`, and multiplying by a power of two is exact. So the
     `Float32` cast rounds nothing and the value a reader computes by hand
     is the value in the buffer.
  2. **Never `+-0.0` and never subnormal.** The magnitude is in
     `[0.125, 32)`. `assert_no_zero_or_subnormal_logits` MEASURES this
     over every case rather than trusting the paragraph.
  3. **Both signs and eight binades.** A one-binade fixture makes the
     row maximum trivial and makes `exp(shift)` nearly constant, which is
     the shape that hides fold-order clauses.

WHAT THIS FILE DOES **NOT** DO
--------------------------------
* **It is not an independent reference.** Every expected value it can
  produce is either a hashed input or the closed form of contract 12.1 /
  12.2, and the second one covers only the two EXACT families. Everything
  else is gated against `loss_oracle.mojo`, which is ours. Contract OWED
  item 6 -- `training/corpus/` does not exist -- is not closed by anything
  here.
* **It plants no NaN and no infinity.** Contract section 8's audit plants
  those, and it plants them ON THE DEVICE, by bits, after the upload, and
  reads them back before the call. That is `loss_check.mojo`'s clause (f)
  and it cannot be expressed as a host list.
* **It carries no expected BITS for the ordinary cases.** Writing down
  what `ce.denom` should be at `V = 300` would be transcribing the oracle,
  which is a restatement and not a check. The two EXACT families are the
  only place a value is written by hand, and contract 12.1 is explicit
  that they gate CORRECTNESS and separate NO spelling from any other.

TRAPS THIS FILE IS WRITTEN AROUND, all measured in this repository
--------------------------------------------------------------------
* `[[mojo-amp-plus-is-bitwise-and]]`. Every `+` on a `UInt64` below is a
  real wrapping add. `x &+ k` computes `x & k` with NO COMPILE ERROR and
  has silently produced wrong hashes here twice, the second time on
  2026-08-25. Do not "fix" one of these into a `&+`.
* `[[mojo-list-float32-not-implicitly-copyable]]`. `var a = b` on a
  `List[Float32]` does not compile. Every copy below is an explicit
  `append` loop or `.copy()`.
* `[[mojo-string-float-roundtrip]]`. No decimal string is the source of
  any constant that matters. `CE_EPS_SMOOTH` carries its bit pattern in a
  comment and `ce_profile_constants_are_intact` checks it.
* `[[mojo-int-widening-sign-extends]]`. Every hash shift below is on
  `UInt64` and every narrowing is masked.

WHAT IS OWED, AND THIS FILE COVERS NONE OF IT
------------------------------------------------
* **A COMPILE.** Nothing here has been through the front end.
* **A RUN.** Zero bits observed, on any column.
* **`training/corpus/`.** The float64 reference of contract 12.3's A3 arm.
* **The shipped vocabulary.** `shipped_v128256` is in the table and is OFF
  by default (`MOJOLEARN_LOSS_CHECK_SHIPPED_V`), because the host oracle
  is `O(N*V)` scalar Mojo with a `List` copy per fold and Andrew's box may
  not be pushed (`[[no-heavy-local-compute]]`). Contract 3.2's carry is
  reachable at `V = 300` without it, which is why the default set is
  honest rather than merely cheap.
"""

from std.memory import bitcast

from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_exp,
    identical_fmax,
    identical_log,
    numeric_mode_name,
)
from training.checks.loss_oracle import (
    CE_NEG_INF_BITS,
    CeConfig,
    IGNORE_INDEX_DEFAULT,
    REDUCTION_MEAN,
    REDUCTION_NONE,
    REDUCTION_SUM,
    ce_count,
    ce_divisor,
    ce_fold,
    ce_ones,
    neg_by_bits,
)


# ===========================================================================
# BITS
# ===========================================================================
# Every one of these is a PATTERN and not a decimal, for
# `[[mojo-string-float-roundtrip]]`'s reason: this toolchain's Float32 text
# is lossy and a constant that was right when it was typed is not the same
# thing as a constant the toolchain agrees with. `ce_profile_constants_are_
# intact` prints them back, and `loss_check.mojo`'s preflight calls it
# BEFORE any device call so a bad constant fails in a second rather than
# after a case sweep.
# ===========================================================================


def f32_from_bits(u: UInt32) -> Float32:
    return bitcast[DType.float32](u)


def bits_of(v: Float32) -> UInt32:
    return bitcast[DType.uint32](v)


comptime BITS_POS_ZERO: UInt32 = 0x00000000
comptime BITS_NEG_ZERO: UInt32 = 0x80000000
comptime BITS_ONE: UInt32 = 0x3F800000
comptime BITS_POS_INF: UInt32 = 0x7F800000
comptime BITS_NEG_INF: UInt32 = 0xFF800000
comptime BITS_QNAN: UInt32 = 0x7FC00000
comptime BITS_MIN_SUBNORMAL: UInt32 = 0x00000001
comptime BITS_BIG_SUBNORMAL: UInt32 = 0x007FFFFF
comptime BITS_NEG_SUBNORMAL: UInt32 = 0x80000101

#: `0.1`, the label-smoothing coefficient of contract 10.1's `EPS = 0.1`
#: rows. `0x3DCCCCCD` is the nearest Float32 to one tenth and it is NOT one
#: tenth; the smoothing clauses are about ROUNDING and a reader who thinks
#: this is exact will misread every one of them.
comptime BITS_EPS_SMOOTH: UInt32 = 0x3DCCCCCD

#: The large common offset of contract 4.2(a), `+1e6`. `L_NLL_VIA_ADDBACK`
#: is INERT on a centered row and this is the whole separation: at `1e6` the
#: sum `m + logdenom` loses `logdenom`'s low bits entirely, and the pinned
#: `shift[y] - logdenom` does not move a bit, because `shift` is a
#: difference that the offset cancels out of EXACTLY (both operands carry
#: the same offset and the subtraction of two nearby large values is exact
#: by Sterbenz when they are within a factor of two, which they are here
#: because the hashed part is bounded by 32).
comptime BITS_BIG_OFFSET: UInt32 = 0x49742400

#: The saturating family's gap, `-200.0`. `portable_expf` returns EXACTLY
#: `+0.0` below `-87.33655`, so a cell at `a - 200` has `e == +0.0` exactly
#: and `w == +0.0` exactly, which is contract 12.2's underflow edge and is
#: also `L_NLL_VIA_LOG_W`'s separating condition (`-log(+0.0)` is `+inf`,
#: the pinned spelling gives an ordinary finite number near 200).
comptime BITS_SAT_GAP: UInt32 = 0xC3480000

#: The uniform family's value, `1.25`. Exact, normal, not a power of two
#: (so `ce.max` is not a value that flatters a wrong flush), and its own
#: `shift` is exactly `+0.0` at every cell.
comptime BITS_UNIFORM_VALUE: UInt32 = 0x3FA00000

#: **THE POISON.** DEVIATION 1456. `SAB_IGNORED_ROW_SKIPPED` is ALWAYS
#: INERT if the gate pre-fills the output buffers with zeros -- which is
#: what a fresh allocation may or may not contain -- and never inert if the
#: gate poisons. `loss.mojo`'s own switch docstring says exactly that. This
#: is a quiet NaN with a payload nothing computes, so a surviving cell is
#: unmistakably an UNWRITTEN cell and not a wrong value.
#:
#: **A SURVIVING POISON IN A CLEAN BUILD IS A FAILURE**, not a curiosity:
#: it means a stage the card records was never written and the card is
#: hashing whatever the allocator left there. `loss_check.mojo` asserts
#: zero survivors on every clean run.
comptime BITS_POISON: UInt32 = 0x7FC0DEAD


def ce_eps_smooth() -> Float32:
    return f32_from_bits(BITS_EPS_SMOOTH)


def ce_profile_constants_are_intact() -> Bool:
    """Every constant this fixture depends on, by BITS.

    A wrong constant here does not look like a wrong constant. It looks
    like a kernel bug on every stage downstream of it, which is the
    transformer lane's `profile_constants_are_intact` argument at a second
    site.
    """
    if bits_of(ce_eps_smooth()) != BITS_EPS_SMOOTH:
        return False
    if bits_of(f32_from_bits(BITS_UNIFORM_VALUE)) != BITS_UNIFORM_VALUE:
        return False
    if bits_of(f32_from_bits(BITS_SAT_GAP)) != BITS_SAT_GAP:
        return False
    if bits_of(f32_from_bits(BITS_BIG_OFFSET)) != BITS_BIG_OFFSET:
        return False
    if bits_of(Float32(1.0)) != BITS_ONE:
        return False
    return True


def bits32_hex(v: Float32) -> String:
    """Eight lowercase hex digits, `0x` prefixed. The report format every
    lane in this repository uses for a Float32, and the only one that can
    say `+0.0` and `-0.0` are different."""
    var u = bits_of(v)
    var digits = String("0123456789abcdef")
    var out = String("0x")
    for i in range(8):
        var shift = UInt32(28 - 4 * i)
        var nib = Int((u >> shift) & UInt32(0xF))
        out += String(digits[byte=nib])
    return out^

def is_nonfinite_bits(v: Float32) -> Bool:
    """NaN or infinity, BY BITS AND NEVER BY A COMPARE.

    Metal FLUSHES COMPARE OPERANDS (IDENTITY_PATHS row 49, DEVIATION
    746(i)), so `v != v` is a test with two meanings across columns while a
    mask-and-compare on the exponent field has one. Contract section 8 is
    written in these terms and this is the function that means it."""
    return (bits_of(v) & UInt32(0x7FFFFFFF)) >= UInt32(0x7F800000)


def is_zero_bits(v: Float32) -> Bool:
    """Either signed zero. Not `v == 0.0`, for `is_nonfinite_bits`'s
    reason, and because `v == 0.0` is TRUE for a subnormal on a column
    that flushes compare operands and FALSE on one that does not -- which
    would make the "no subnormal" half of
    `assert_no_zero_or_subnormal_logits` a different assertion per
    vendor."""
    return (bits_of(v) & UInt32(0x7FFFFFFF)) == UInt32(0)


def is_subnormal_bits(v: Float32) -> Bool:
    var a = bits_of(v) & UInt32(0x7FFFFFFF)
    return a != UInt32(0) and a < UInt32(0x00800000)


def is_exact_power_of_two(v: Float32) -> Bool:
    """A positive normal whose fraction field is zero. Used by contract
    12.4's guard 2 and by the two reciprocal-multiply arms, both of which
    are EXACT (and therefore inert) exactly here."""
    var b = bits_of(v)
    if (b & UInt32(0x80000000)) != UInt32(0):
        return False
    var e = (b >> 23) & UInt32(0xFF)
    if e == UInt32(0) or e == UInt32(0xFF):
        return False
    return (b & UInt32(0x007FFFFF)) == UInt32(0)


def mode_name() -> String:
    """The build's tier, from the ONE definition of it.

    Delegates to `numeric_mode_name()` since 2026-08-29. This used to
    be a local two-way `IDENTICAL`-or-`FAST`, written when there were
    two tiers, and it answered "FAST" for a DETERMINISTIC build -- so
    a driver run under the middle tier printed the wrong arm onto
    every line it produced. A correctly-labelled measurement of the
    wrong arm is the failure this tree has been bitten by repeatedly,
    and forty-four copies of a mode label is how it happens.
    """
    return numeric_mode_name()


def mode_is_identical() -> Bool:
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return True
    return False


# ===========================================================================
# THE HASH, spec `mojolearn.loss.fixture.hash.v1` (DEVIATION 1450)
# ===========================================================================


def ce_splitmix64(z_in: UInt64) -> UInt64:
    """splitmix64, Steele/Lea/Flood's finalizer, verbatim.

    **A THIRD COPY IN THIS REPOSITORY**, after `mamba/checks/
    mamba_fixture.mojo::corpus_splitmix64` and `transformer/checks/
    transformer_fixture.mojo::fixture_splitmix64` (which is DEVIATION 1000
    and states the cost in as many words: "two copies of a hash have two
    chances to be edited apart"). This one makes it three, and the cost is
    the same one multiplied.

    It is copied rather than imported for the reason DEVIATION 1000 gives
    -- exact integer arithmetic cannot drift the way a float seam can, and
    an import would point this lane's fixture at `mamba/` for no other
    reason -- and the mitigation is that `loss_check.mojo`'s preflight
    asserts all three agree on five seeds. If they ever disagree the
    likeliest cause is `[[mojo-amp-plus-is-bitwise-and]]`: Mojo's `&+`
    computes `x & k` with NO COMPILE ERROR, so a `+` "fixed" into a `&+` in
    ONE copy and not the others is exactly the edit that check catches.

    **EVERY `+` BELOW IS A REAL WRAPPING ADD ON `UInt64`. Do not touch
    them.**"""
    var z = z_in + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


#: ASCII-ish and DISTINCT from the transformer lane's `0x546672666D724C6C`
#: and the mamba corpus's `0x4D616D6261436F72`. Two lanes sharing a seed
#: base makes two different fixtures correlate, which is harmless right up
#: to the moment somebody compares a hash across lanes and reads meaning
#: into it.
comptime CE_SEED_BASE: UInt64 = 0x43654C6F73734631


def ce_case_seed(k: Int) -> UInt64:
    """Case `k`'s seed. The `0x1000` stride is the mamba corpus's and the
    transformer fixture's, kept so the three lanes' seed schedules look the
    same to a reader."""
    return CE_SEED_BASE + UInt64(0x1000) * UInt64(k)


#: Tensor ids, so a logit can never collide with a target draw or with a
#: plant even at the same index.
comptime TID_LOGITS = 1
comptime TID_TARGETS = 2
comptime TID_PLANT = 3


def ce_hashed_logit(seed: UInt64, idx: Int) -> Float32:
    """One hashed logit: `+-(1 + m) * 2^e`, `m` a 23-bit fraction, `e` in
    `[-3, 4]`.

    **THE THREE PROPERTIES, AND ALL THREE ARE STRUCTURAL.** They are argued
    in the module docstring and they are MEASURED by
    `assert_no_zero_or_subnormal_logits`, because a fixture property that
    is only argued is a fixture property that is wrong on one case.

      1. EXACTLY REPRESENTABLE. `1 + m` with 23 fraction bits is exact in
         `[1, 2)` -- that is precisely the Float32 significand -- and
         multiplying by a power of two is exact at every exponent in range.
         So the `Float32` cast below rounds NOTHING.
      2. NEVER `+-0.0` AND NEVER SUBNORMAL. The magnitude is in
         `[0.125, 32)`. Every inert half of `L_MAX_PLAIN_COMPARE` and every
         `ftz`-related clause rests on this.
      3. BOTH SIGNS, EIGHT BINADES. A one-binade fixture makes the row
         maximum trivial and `exp(shift)` nearly constant, which is exactly
         the shape that hides a fold-order clause.

    The scaling is a LOOP of exact multiplies and divides by `2.0` rather
    than a `pow`, because `portable_powf` is `exp(p * log(x))` through two
    Cephes polynomials and is not exact even at `p = 1` -- contract 5.1's
    own argument about `beta^t`, at a much smaller site."""
    var h = ce_splitmix64(seed + UInt64(idx))
    var frac = Int((h >> 41) & UInt64(0x7FFFFF))
    # 2^-23 exactly. Written as a decimal that IS the binary value, not as
    # a division, so nothing here depends on a host divide.
    var mant = Float64(frac) * 1.1920928955078125e-07
    var v = Float32(1.0 + mant)
    var e = Int((h >> 3) & UInt64(0x7)) - 3
    var k = e
    while k > 0:
        v = v * Float32(2.0)
        k -= 1
    while k < 0:
        v = v / Float32(2.0)
        k += 1
    if (h & UInt64(1)) != UInt64(0):
        v = -v
    return v


def ce_hashed_target(seed: UInt64, row: Int, vocab: Int) -> Int32:
    """A class in `[0, vocab)`, by INTEGER arithmetic only.

    Integers do not flush and do not round, so a target is the one input in
    this profile that has no seam at all. Contract section 4's "copies and
    integer work are NOT seams"."""
    if vocab <= 1:
        return Int32(0)
    var h = ce_splitmix64(seed + UInt64(0x5A5A0000) + UInt64(row))
    return Int32(Int(h % UInt64(vocab)))


# ===========================================================================
# THE PLANT KINDS
# ===========================================================================
# Each one exists for exactly one contract clause and the clause is named.
# A plant with no clause is decoration and the case set has none.
# ===========================================================================

#: No plant. The hashed values, unmodified.
comptime CE_PLANT_NONE = 0

#: Contract 5.1 / `L_MAX_PLAIN_COMPARE`. Cell 0 of every row is `+0.0` and
#: cell 1 is `-0.0`. IDENTITY_PATHS row 39 MEASURED `max(+0.0, -0.0)` as
#: `-0.0` on Apple (the second operand) and `+0.0` on NVIDIA and AMD, so
#: this is a LIVE three-vendor split and not a hypothetical. Under
#: `identical_fmax` the total-order key of `+0.0` is `0x80000000` and of
#: `-0.0` is `0x7FFFFFFF`, so `+0.0` WINS on every column; under a plain
#: `>` compare the two are equal and the ORDER decides, which is what the
#: arm models.
comptime CE_PLANT_SIGNED_ZEROS = 1

#: Contract 5.1 / `L_MAX_SEED_ZERO`. Every logit forced strictly negative,
#: which is what most rows of a trained language-model head look like after
#: the first few steps. A `+0.0` seed then clamps the maximum and the loss
#: is quietly wrong ON EVERY VENDOR IDENTICALLY -- bit-identity cannot see
#: it and only the oracle can, which is the sharpest possible illustration
#: of why clause (a) is against an oracle and not against a second device.
comptime CE_PLANT_ALL_NEGATIVE = 2

#: Contract 5.2 / `L_MAX_TOPK_PREFIX`. The row maximum is planted at index
#: `V - 1`, past the arm's 32-element prefix. Needs `V > 32` to mean
#: anything, which `ce_case` enforces.
comptime CE_PLANT_ARGMAX_TAIL = 3

#: The inert half of the same arm: the maximum is planted at index 0.
comptime CE_PLANT_ARGMAX_HEAD = 4

#: Contract 4.2(a) / `L_NLL_VIA_ADDBACK`. `+1e6` added to every logit. The
#: pinned `shift[y] - logdenom` does not move a bit; `(m + logdenom) - x_y`
#: loses `logdenom` entirely in the addition and then cancels
#: catastrophically.
comptime CE_PLANT_BIG_OFFSET = 5

#: Contract 4.2(b) / `L_NLL_VIA_LOG_W`. The TARGET cell is planted 200
#: below the row maximum, so `e[y]` is EXACTLY `+0.0` (`portable_expf`
#: returns `+0.0` below `-87.33655`) and `w[y]` is exactly `+0.0`. Then
#: `-log(w[y])` is `+inf` and the pinned spelling is an ordinary finite
#: number near 200. **This case also proves contract 8.1's `ce.exp` row is
#: reachable.**
comptime CE_PLANT_UNDERFLOW_TARGET = 6

#: IDENTITY_PATHS row 10's flush unit at the LOAD seam. Every fourth cell
#: is a subnormal. Under IDENTICAL `ftz` turns it into a zero of its own
#: sign before the subtraction; under FAST it does not, and that difference
#: is the whole of row 10.
comptime CE_PLANT_SUBNORMAL = 7

#: Contract 12.1's EXACT-ANALYTIC family. Every logit of every row is the
#: SAME value, so `shift` is exactly `+0.0`, `e` is exactly `1.0`, `denom`
#: is exactly `Float32(V)` IN EVERY FOLD ORDER, and `w` is exactly `1/V`
#: when `V` is a power of two. Every gradient cell is then a dyadic
#: rational a person can write down.
comptime CE_PLANT_UNIFORM = 8

#: Contract 12.2's second EXACT family. The first `high_count` cells hold
#: the uniform value and the rest hold it minus 200. Adds four things 12.1
#: lacks: a genuine argmax structure, an exercised underflow edge, exact
#: `+0.0` weights whose SIGN must be `+`, and a row whose TARGET is a low
#: cell (`dl[y]` is exactly `-1/divisor`).
comptime CE_PLANT_SATURATING = 9

#: Contract 4.2(a)'s INERT half. Every logit forced into `[-1, 1]` by a
#: single exact halving of the hashed magnitude band, which is where the
#: add-back has nothing to lose and agrees with the pin to the last bit.
comptime CE_PLANT_CENTERED = 10


def plant_name(p: Int) -> String:
    if p == CE_PLANT_NONE:
        return String("none")
    if p == CE_PLANT_SIGNED_ZEROS:
        return String("signed_zeros")
    if p == CE_PLANT_ALL_NEGATIVE:
        return String("all_negative")
    if p == CE_PLANT_ARGMAX_TAIL:
        return String("argmax_tail")
    if p == CE_PLANT_ARGMAX_HEAD:
        return String("argmax_head")
    if p == CE_PLANT_BIG_OFFSET:
        return String("big_offset")
    if p == CE_PLANT_UNDERFLOW_TARGET:
        return String("underflow_target")
    if p == CE_PLANT_SUBNORMAL:
        return String("subnormal")
    if p == CE_PLANT_UNIFORM:
        return String("uniform")
    if p == CE_PLANT_SATURATING:
        return String("saturating")
    if p == CE_PLANT_CENTERED:
        return String("centered")
    return String("plant?")


# ===========================================================================
# THE CASE TABLE
# ===========================================================================


@fieldwise_init
struct CeCase(Copyable, Movable):
    """One runnable loss call.

    `ignore_stride` is 0 for "no ignored rows" and `k > 0` for "rows
    `0, k, 2k, ...` carry `ignore_index`". It is a STRIDE rather than a
    count because the two things `L_GRAD_DIVISOR_IS_N` and
    `L_IGNORED_ROW_NEG_ZERO` need are (i) `count != N` and (ii) an ignored
    row that is NOT the last one, so that its `+0.0` sits in the MIDDLE of
    the batch fold where a `-0.0` would have to be laundered by the seed
    rather than by being at the end.

    `high_count` is only read by `CE_PLANT_SATURATING` and is contract
    12.2's `c`. It must be a power of two or `ce_case` raises, because the
    exactness argument depends on it (contract 12.4 guard 2) and an
    unchecked exactness argument is how a gate comes to assert what the
    code does rather than what it should do.
    """

    var name: StaticString
    var n_rows: Int
    var vocab: Int
    var reduction: Int
    var eps_on: Bool
    var ignore_stride: Int
    var num_items: Int
    var plant: Int
    var high_count: Int


#: Twenty-five cases. `ce_case` raises past this and `loss_check.mojo`'s
#: clause (a) runs a subset chosen by `clause_a_cases`.
comptime CE_CASE_COUNT = 25

#: The index of the shipped-vocabulary case. It is EXCLUDED from every
#: default set and joins on `MOJOLEARN_LOSS_CHECK_SHIPPED_V`. See the
#: module docstring; the short reason is `[[no-heavy-local-compute]]` and
#: the fact that contract 3.2's carry is already reachable at `V = 300`.
comptime CE_CASE_SHIPPED_V = 24


def ce_case(k: Int) raises -> CeCase:
    """Case `k`. Every row of the table below names the clause it exists
    for; a case with no clause would be a case nobody can delete safely.

    THE FOLD-SHAPE ARITHMETIC, since four arms turn on it and a reader
    should not have to recompute it. Under gemm v1, `L = contract_leaf_
    size(k)` and `P = ceil(k / L)`, and at `k <= 128` the leaf is the whole
    axis so `P == 1` and **the tree performs NO addition** (gemm contract
    7.3). Therefore:

        V or N     P     ragged last leaf?   carry (an odd level width)?
        1..128     1     n/a                 no  -- every fold arm INERT
        129        2     yes, 1 element      no  (widths 2, 1)
        256        2     no                  no  (widths 2, 1)
        300        3     yes, 44 elements    YES (widths 3 -> 2 -> 1)
        512        4     no                  no  (4, 2, 1)
        1024       8     no                  no  (8, 4, 2, 1)
        128256     1002  no                  YES three times (1002, 501,
                                                 251, 126, 63, 32, ...)

    So `V = 300` is the ONLY default case that exercises BOTH the ragged
    leaf and the carry, `V = 256` and `V = 1024` are the PAD_PLUS_ZERO
    inert halves, and `V = 129` is the most extreme raggedness the leaf
    rule can produce (a one-element last leaf).
    """
    if k == 0:
        # The centre of the box. MEAN over 4 rows with no ignored row, so
        # `count == N == 4` and the divisor is 4, an exact power of two --
        # which makes this case the INERT half of BOTH reciprocal-multiply
        # arms as well as of `L_GRAD_DIVISOR_IS_N`.
        return CeCase(
            "base_n4_v8", 4, 8, REDUCTION_MEAN, False, 0, 0, CE_PLANT_NONE, 0
        )
    if k == 1:
        # `V == 1`. The degenerate softmax: one class, `shift` exactly
        # `+0.0`, `denom` exactly `1.0`, `logdenom` exactly `+0.0`, `lp_y`
        # exactly `+0.0`, and `nll = neg_by_bits(+0.0)` EXACTLY `-0.0`.
        # **This is the witness for `L_NEG_VIA_ZERO_SUB` and for
        # `L_SMOOTH_ALWAYS_SPELLED` and there is no other shape in the set
        # that is.** It is also the case a hurried fixture omits, because
        # a one-class softmax looks pointless.
        return CeCase(
            "base_n1_v1", 1, 1, REDUCTION_MEAN, False, 0, 0, CE_PLANT_NONE, 0
        )
    if k == 2:
        # `count == 3`, a divisor that is NOT a power of two. The witness
        # for `L_MEAN_RECIPROCAL_MUL` and `L_GRAD_RECIPROCAL_MUL`, both of
        # which are EXACT (and therefore inert) at a power-of-two divisor.
        # `V == 3` is the smallest odd vocabulary.
        return CeCase(
            "base_n3_v3", 3, 3, REDUCTION_MEAN, False, 0, 0, CE_PLANT_NONE, 0
        )
    if k == 3:
        # `V = 300`: `P = 3`, a ragged 44-element last leaf, and an odd
        # level width so the carry fires. The GEMM contract's own clause-5
        # fixture lifted into this lane, and contract 3.2 says in as many
        # words that it is REQUIRED because the shipped vocabulary's 128256
        # is an exact multiple of 128 and never takes the ragged path.
        return CeCase(
            "wide_v300", 2, 300, REDUCTION_MEAN, False, 0, 0, CE_PLANT_NONE, 0
        )
    if k == 4:
        # `V = 256`: `P = 2`, no ragged leaf, and EVERY level width even.
        # The INERT half of `L_DENOM_PAD_PLUS_ZERO`.
        return CeCase(
            "wide_v256", 2, 256, REDUCTION_MEAN, False, 0, 0, CE_PLANT_NONE, 0
        )
    if k == 5:
        # `V = 1024`: `P = 8`, widths 8, 4, 2, 1, all even. A second and
        # deeper PAD_PLUS_ZERO inert half, and the largest default `V`.
        return CeCase(
            "wide_v1024", 2, 1024, REDUCTION_MEAN, False, 0, 0,
            CE_PLANT_NONE, 0,
        )
    if k == 6:
        # `V = 129`: `P = 2` with a ONE-ELEMENT ragged last leaf, the most
        # extreme raggedness the leaf rule can produce, and the smallest
        # `V` at which the fold arms are not inert.
        return CeCase(
            "wide_v129", 2, 129, REDUCTION_MEAN, False, 0, 0, CE_PLANT_NONE, 0
        )
    if k == 7:
        # `N = 300`. The BATCH fold's witness -- `L_REDUCE_SERIAL` is inert
        # at every `N <= 128` and `N = 300` gives it the same `P = 3`,
        # ragged, carrying tree the vocabulary fold gets at `V = 300`.
        # `count == 300`, not a power of two, so the divide arms bite here
        # too.
        return CeCase(
            "long_n300_v8", 300, 8, REDUCTION_MEAN, False, 0, 0,
            CE_PLANT_NONE, 0,
        )
    if k == 8:
        # `N = 512` under SUM with NO `num_items`, so the divisor is
        # exactly `+1.0` and the divide is BITWISE INERT -- contract 5.5
        # spells it anyway so there is one code path, and this case is what
        # proves the spelled divide costs nothing. `P = 4`, all level
        # widths even.
        return CeCase(
            "long_n512_sum", 512, 8, REDUCTION_SUM, False, 0, 0,
            CE_PLANT_NONE, 0,
        )
    if k == 9:
        # SUM WITH `num_items_in_batch`, `fixed_cross_entropy` :45. The
        # divisor is `Float32(7)`, not a power of two, and it comes from a
        # DIFFERENT branch of `ce_divisor` than the MEAN cases do -- which
        # is the only way to check that the ONE PRODUCER really is one.
        return CeCase(
            "sum_num_items7", 5, 8, REDUCTION_SUM, False, 0, 7,
            CE_PLANT_NONE, 0,
        )
    if k == 10:
        # An IGNORED ROW, and the ignored rows are 0, 2 and 4 of six, so
        # `count == 3`. Two properties at once and both are needed:
        #   * `count != N` (3 against 6) is `L_GRAD_DIVISOR_IS_N`'s
        #     separating condition and it has NO other case in the set.
        #   * an ignored row in the MIDDLE of the batch fold is what makes
        #     `L_IGNORED_ROW_NEG_ZERO`'s "must NOT move ce.total" a real
        #     claim: a `-0.0` at the END of a `+0.0`-seeded chain is
        #     laundered trivially, and one in the middle has to be
        #     laundered by the accumulator's own value.
        return CeCase(
            "ign_n6_v8_c3", 6, 8, REDUCTION_MEAN, False, 2, 0,
            CE_PLANT_NONE, 0,
        )
    if k == 11:
        # A second ignored-row shape whose `count` IS a power of two (2 of
        # 5), so `L_GRAD_DIVISOR_IS_N` bites here (count 2 vs N 5) while
        # both reciprocal arms stay inert. Two arms separated on one case.
        return CeCase(
            "ign_n5_v8_c2", 5, 8, REDUCTION_MEAN, False, 2, 0,
            CE_PLANT_NONE, 0,
        )
    if k == 12:
        # LABEL SMOOTHING at `EPS = 0.1`. The witness for
        # `L_SMOOTH_FOLDED_CONSTANT` and `L_SMOOTH_FUSED_COMBINE`, both of
        # which are inert at `EPS == 0` because the arm is not spelled
        # there at all (contract 6.2(c), DEVIATION 1155).
        return CeCase(
            "smooth_n4_v8", 4, 8, REDUCTION_MEAN, True, 0, 0, CE_PLANT_NONE, 0
        )
    if k == 13:
        # SMOOTHING AT `V == 1`. The INERT half of
        # `L_SMOOTH_ALWAYS_SPELLED`: the row's `nll` is `-0.0` exactly as
        # in case 1, but `EPS != 0` so the smoothing path is spelled on the
        # CLEAN build too and the arm changes nothing. Without this case
        # the arm has a witness and no inert half and is a smoke test.
        return CeCase(
            "smooth_v1_eps", 2, 1, REDUCTION_MEAN, True, 0, 0,
            CE_PLANT_NONE, 0,
        )
    if k == 14:
        # SMOOTHING at a vocabulary with a real fold, so `ce.logp_sum` is
        # a tree and not a single leaf. `L9` is the same call as `L4` at
        # the same `k`, so a divergence at `ce.logp_sum` that is NOT at
        # `ce.denom` is a routing defect and nothing else.
        return CeCase(
            "smooth_v300", 2, 300, REDUCTION_MEAN, True, 0, 0,
            CE_PLANT_NONE, 0,
        )
    if k == 15:
        # `REDUCTION_NONE`. There is no `ce.total`, no `ce.divisor`, no
        # `ce.loss` AND NO BACKWARD (contract section 11), so this case
        # exists to check that the ABSENT stages are absent rather than
        # zero-filled. An empty stage and a stage that says "nothing
        # happened" are different claims.
        return CeCase(
            "none_reduction", 4, 8, REDUCTION_NONE, False, 0, 0,
            CE_PLANT_NONE, 0,
        )
    if k == 16:
        # `L_MAX_PLAIN_COMPARE`'s WITNESS. `+0.0` at cell 0 and `-0.0` at
        # cell 1 of every row. Every other case in the set is asserted to
        # carry neither zero sign, which is what makes their inertness a
        # measurement instead of an assumption.
        return CeCase(
            "adv_signed_zeros", 2, 16, REDUCTION_MEAN, False, 0, 0,
            CE_PLANT_SIGNED_ZEROS, 0,
        )
    if k == 17:
        # `L_MAX_SEED_ZERO`'s WITNESS: every logit strictly negative.
        return CeCase(
            "adv_all_negative", 2, 16, REDUCTION_MEAN, False, 0, 0,
            CE_PLANT_ALL_NEGATIVE, 0,
        )
    if k == 18:
        # `L_MAX_TOPK_PREFIX`'s WITNESS: `V = 64 > 32` with the maximum at
        # index 63, past the arm's prefix.
        return CeCase(
            "adv_argmax_tail", 2, 64, REDUCTION_MEAN, False, 0, 0,
            CE_PLANT_ARGMAX_TAIL, 0,
        )
    if k == 19:
        # The INERT half of the same arm at the SAME shape: `V = 64` with
        # the maximum at index 0. Same `V`, same `N`, same seed schedule,
        # one plant apart -- which is what makes the pair a reach proof
        # rather than two unrelated observations.
        return CeCase(
            "adv_argmax_head", 2, 64, REDUCTION_MEAN, False, 0, 0,
            CE_PLANT_ARGMAX_HEAD, 0,
        )
    if k == 20:
        # `L_NLL_VIA_ADDBACK`'s WITNESS: `+1e6` on every logit.
        return CeCase(
            "adv_big_offset", 2, 16, REDUCTION_MEAN, False, 0, 0,
            CE_PLANT_BIG_OFFSET, 0,
        )
    if k == 21:
        # The INERT half: a CENTERED row, logits in `[-1, 1]`. Contract
        # 4.2's own "what would make L6 pass while gating nothing" names
        # exactly this shape as the one that hides the arm.
        return CeCase(
            "adv_centered", 2, 16, REDUCTION_MEAN, False, 0, 0,
            CE_PLANT_CENTERED, 0,
        )
    if k == 22:
        # `L_NLL_VIA_LOG_W`'s WITNESS: the TARGET cell 200 below the row
        # maximum, so `e[y]` is EXACTLY `+0.0` and the refused spelling
        # returns `+inf`. Also the only case that reaches contract 8.1's
        # `ce.exp` row (an exponential that underflows to exactly `+0.0`)
        # outside the saturating exact family.
        return CeCase(
            "adv_underflow_y", 2, 16, REDUCTION_MEAN, False, 0, 0,
            CE_PLANT_UNDERFLOW_TARGET, 0,
        )
    if k == 23:
        # IDENTITY_PATHS row 10's flush unit at the LOAD seam: every fourth
        # logit a subnormal. Under IDENTICAL these flush to a zero of their
        # own sign BEFORE the subtraction; under FAST they do not.
        return CeCase(
            "adv_subnormal", 2, 16, REDUCTION_MEAN, False, 0, 0,
            CE_PLANT_SUBNORMAL, 0,
        )
    if k == CE_CASE_SHIPPED_V:
        # The SHIPPED Llama-3 head. `V = 128256`, `P = 1002`, `L = 128`
        # exactly, no ragged leaf ever, and the carry three times (widths
        # 1002, 501, 251, 126, 63, ...). OFF by default; see the module
        # docstring. `N = 1` keeps it to about 2 MB of buffers.
        return CeCase(
            "shipped_v128256", 1, 128256, REDUCTION_MEAN, False, 0, 0,
            CE_PLANT_NONE, 0,
        )
    raise Error(
        String("loss_fixture: case ")
        + String(k)
        + " does not exist; there are "
        + String(CE_CASE_COUNT)
    )


def ce_case_by_name(name: String) raises -> Int:
    """A case by NAME, which is the spelling the contract's tables and this
    lane's ledger use.

    A name that does not exist RAISES rather than falling back to a
    default. A gate that quietly ran a different case from the one the
    operator asked for would report a green for a shape nobody chose, which
    is the same class of defect as a card written to an aliased path
    (DEVIATION 970)."""
    for k in range(CE_CASE_COUNT):
        var c = ce_case(k)
        if String(c.name) == name:
            return k
    raise Error(
        String("loss_fixture: no case named '")
        + name
        + "'. The names are in ce_case's table."
    )


def ce_case_config(c: CeCase) -> CeConfig:
    """The case's `CeConfig`. `eps` is a BIT PATTERN and not a decimal
    literal, for `[[mojo-string-float-roundtrip]]`'s reason."""
    var eps = Float32(0.0)
    if c.eps_on:
        eps = ce_eps_smooth()
    return CeConfig(c.vocab, IGNORE_INDEX_DEFAULT, c.reduction, eps,
                    c.num_items)


# ===========================================================================
# THE EXACT FAMILIES (contract section 12)
# ===========================================================================
# These two are not in the ordinary case table. They are built by
# `ce_exact_case`, they carry contract 12.4's vacuity guards as RAISES
# rather than as comments, and `loss_check.mojo`'s clause (e) asserts a
# hand-written closed form against them BY BITS with no epsilon anywhere.
#
# **AND THE HONEST OTHER HALF, WHICH CONTRACT 12.1 STATES AND WHICH IS
# EASY TO FORGET.** Every quantity in these families is EXACT, so they
# separate NO spelling from any other. `identical_div(1.0, 4.0)` and
# `1.0 * (1/4)` agree; a serial fold and a balanced tree of ones agree;
# `identical_exp` and `std.math.exp` agree at `+0.0`. **Run alone these
# families would pass every sabotage in contract 10.1 and gate nothing
# about the arithmetic.** They are the CORRECTNESS half. The SPELLING half
# is clause (a) on the ordinary cases, and `loss_check.mojo` carries
# contract 12.4's guard 4 -- an arm that DEMONSTRATES the vacuity rather
# than asserting it -- so no reader is entitled to believe otherwise.
# ===========================================================================


@fieldwise_init
struct CeExactCase(Copyable, Movable):
    """One EXACT-ANALYTIC case. `high_count == 0` means the UNIFORM family
    of contract 12.1; `high_count > 0` means the SATURATING family of 12.2
    with `high_count` cells at the top value."""

    var name: StaticString
    var n_rows: Int
    var vocab: Int
    var reduction: Int
    var ignore_stride: Int
    var num_items: Int
    var high_count: Int
    var target_low: Bool


comptime CE_EXACT_CASE_COUNT = 4


def ce_exact_case(k: Int) raises -> CeExactCase:
    """Contract 12.4's guards are enforced in `ce_exact_guard`, which every
    one of these goes through, and which RAISES rather than adjusting the
    fixture. `IDENTICAL_BACKWARD_PLAN.md`'s G2 is the pattern.

    Guard 3 -- "at least one A1 case with an ignored row and at least one
    without" -- is why there are four and not two: without the ignored-row
    pair `L_GRAD_DIVISOR_IS_N`'s predicted inert mask is untested on the
    exact family, and the exact family is the only place the gradient is
    asserted against a hand-written number."""
    if k == 0:
        # UNIFORM, no ignored row. `V = 8`, `N = 3`, SUM with no
        # `num_items` so `divisor` is exactly `+1.0`.
        #   V != N       (8 != 3)      guard 1
        #   V != divisor (8 != 1)      guard 1
        #   V, divisor powers of two   guard 2
        #   no ignored row             guard 3, the second half
        return CeExactCase(
            "exact_uniform_v8", 3, 8, REDUCTION_SUM, 0, 0, 0, False
        )
    if k == 1:
        # UNIFORM, WITH an ignored row. `V = 4`, `N = 3`, rows 0 and 2
        # ignored so `count = 1` and `divisor` is exactly `1.0`.
        #   V != N       (4 != 3)
        #   V != divisor (4 != 1)
        #   V, divisor powers of two
        #   an ignored row             guard 3, the first half
        return CeExactCase(
            "exact_uniform_ign", 3, 4, REDUCTION_MEAN, 2, 0, 0, False
        )
    if k == 2:
        # SATURATING, target on a HIGH cell. `V = 16`, `high_count = 4`,
        # `N = 3`, SUM with divisor `+1.0`.
        #   V != N       (16 != 3)
        #   V != divisor (16 != 1)
        #   V, high_count, divisor powers of two
        return CeExactCase(
            "exact_sat_high", 3, 16, REDUCTION_SUM, 0, 0, 4, False
        )
    if k == 3:
        # SATURATING, target on a LOW cell. `w[y]` is exactly `+0.0` and
        # `dl[y]` is exactly `(-1) / divisor` -- the case contract 12.2
        # says the uniform family cannot reach, and the one that would
        # catch a gradient that quietly clamps a zero weight.
        return CeExactCase(
            "exact_sat_low", 3, 16, REDUCTION_SUM, 0, 0, 4, True
        )
    raise Error(
        String("loss_fixture: exact case ")
        + String(k)
        + " does not exist; there are "
        + String(CE_EXACT_CASE_COUNT)
    )


def ce_exact_guard(e: CeExactCase) raises -> Float32:
    """Contract 12.4's guards 1 and 2, ENFORCED. Returns the divisor.

    Every one of these RAISES rather than adjusting the fixture, because an
    unchecked exactness argument is how a gate comes to assert what the
    code does rather than what it should do -- contract 12.4's own
    sentence, and `IDENTICAL_BACKWARD_PLAN.md`'s G2 discipline."""
    if e.vocab < 1 or (e.vocab & (e.vocab - 1)) != 0:
        raise Error(
            String("loss_fixture: exact case ")
            + String(e.name)
            + " has vocab "
            + String(e.vocab)
            + ", not a power of two; 1/vocab would not be exact and"
            + " contract 12.4 guard 2 forbids it"
        )
    if e.high_count > 0:
        if (e.high_count & (e.high_count - 1)) != 0:
            raise Error(
                String("loss_fixture: exact case ")
                + String(e.name)
                + " has high_count "
                + String(e.high_count)
                + ", not a power of two (contract 12.4 guard 2)"
            )
        if e.high_count > e.vocab:
            raise Error(
                String("loss_fixture: exact case ")
                + String(e.name)
                + " has high_count above vocab"
            )
    var targets = ce_exact_targets(e)
    var count = ce_count(targets, IGNORE_INDEX_DEFAULT)
    var divisor = ce_divisor(e.reduction, count, e.num_items)
    if e.vocab == e.n_rows:
        raise Error(
            String("loss_fixture: exact case ")
            + String(e.name)
            + " has V == N == "
            + String(e.vocab)
            + ". Contract 12.4 guard 1 forbids it: a square-ish fixture"
            + " lets a TRANSPOSED index produce a plausible matrix of the"
            + " right size and the gate would not see it."
        )
    if Float32(e.vocab) == divisor:
        raise Error(
            String("loss_fixture: exact case ")
            + String(e.name)
            + " has V == divisor == "
            + String(e.vocab)
            + " (contract 12.4 guard 1)"
        )
    if not is_exact_power_of_two(divisor):
        raise Error(
            String("loss_fixture: exact case ")
            + String(e.name)
            + " has divisor "
            + bits32_hex(divisor)
            + ", not a power of two; (1/V)/divisor would not be exact"
            + " (contract 12.4 guard 2)"
        )
    return divisor


def ce_exact_config(e: CeExactCase) -> CeConfig:
    """The exact families run at `EPS == 0` ALWAYS, and that is not a
    convenience.

    At `eps == 0` the target vector is exactly `(1.0, +0.0)`
    (`ce_smoothing_targets`'s own docstring), so the closed forms of
    contract 12.1 and 12.2 are the whole gradient with no smoothing term to
    write down. A smoothed exact family would need `T_OTHER = eps/V` to be
    exact too, which pins `eps` to a dyadic rational and buys nothing the
    ordinary smoothing cases do not already buy through clause (a)."""
    return CeConfig(e.vocab, IGNORE_INDEX_DEFAULT, e.reduction,
                    Float32(0.0), e.num_items)


def ce_exact_logits(e: CeExactCase) raises -> List[Float32]:
    """The exact family's logits.

    UNIFORM: every cell is `1.25`.
    SATURATING: cells `[0, high_count)` are `1.25` and the rest are
    `1.25 - 200.0 = -198.75`, both EXACT in Float32 and their difference
    exactly `-200.0` (Sterbenz does not even have to be invoked: both are
    exact multiples of `2^-2` well inside the significand)."""
    var hi = f32_from_bits(BITS_UNIFORM_VALUE)
    var lo = hi + f32_from_bits(BITS_SAT_GAP)
    var out = List[Float32]()
    for _r in range(e.n_rows):
        for v in range(e.vocab):
            if e.high_count > 0 and v >= e.high_count:
                out.append(lo)
            else:
                out.append(hi)
    return out^


def ce_exact_targets(e: CeExactCase) raises -> List[Int32]:
    """Targets for an exact case.

    On the SATURATING family with `target_low` the target is the LAST cell,
    which is a low cell whenever `high_count < vocab` -- and `ce_exact_
    guard` has already refused `high_count > vocab`, so the only way it is
    not low is `high_count == vocab`, which no case in the table uses and
    which `ce_exact_guard` would let through. That gap is CLOSED here
    rather than left to the reader."""
    var out = List[Int32]()
    if e.target_low and e.high_count >= e.vocab:
        raise Error(
            String("loss_fixture: exact case ")
            + String(e.name)
            + " asks for a LOW target and every cell is high, so"
            + " ce_exact_saturating_gradient's is_high would be True and"
            + " the case would silently be the high one"
            + " ([[reached-but-inert]])"
        )
    for r in range(e.n_rows):
        if e.ignore_stride > 0 and (r % e.ignore_stride) == 0:
            out.append(Int32(IGNORE_INDEX_DEFAULT))
            continue
        if e.target_low:
            out.append(Int32(e.vocab - 1))
        else:
            out.append(Int32(0))
    return out^


# ===========================================================================
# BUILDING A CASE'S INPUTS
# ===========================================================================


def ce_case_logits(c: CeCase) raises -> List[Float32]:
    """`[N, V]` row-major, hashed then planted.

    THE PLANT IS APPLIED AFTER THE HASH AND NEVER INSTEAD OF IT, so that
    two cases with the same shape and different plants differ ONLY in the
    planted cells -- which is what makes `adv_argmax_tail` against
    `adv_argmax_head` a controlled pair rather than two unrelated draws."""
    var seed = ce_case_seed(ce_case_index_of(c))
    var n = c.n_rows
    var v = c.vocab
    var out = List[Float32]()
    var base_seed = seed ^ (UInt64(TID_LOGITS) << 32)
    for i in range(n * v):
        out.append(ce_hashed_logit(base_seed, i))

    if c.plant == CE_PLANT_UNIFORM:
        var u = f32_from_bits(BITS_UNIFORM_VALUE)
        for i in range(n * v):
            out[i] = u
        return out^

    if c.plant == CE_PLANT_ALL_NEGATIVE:
        # Force strictly negative by CLEARING nothing and SETTING the sign
        # bit -- a bit operation, so the magnitude is untouched and the
        # case differs from `base` in the sign field alone. A `-abs(x)`
        # would be the same value and would go through a float op for no
        # reason.
        for i in range(n * v):
            out[i] = f32_from_bits(bits_of(out[i]) | UInt32(0x80000000))
        return out^

    if c.plant == CE_PLANT_CENTERED:
        # Halve the magnitude four times: exact, and it maps the
        # `[0.125, 32)` band into `[0.0078125, 2)`, which is inside
        # contract 4.2's "logits in [-1, 1]" spirit and stays clear of the
        # subnormal floor by twenty binades.
        for i in range(n * v):
            var x = out[i]
            for _s in range(4):
                x = x / Float32(2.0)
            out[i] = x
        return out^

    if c.plant == CE_PLANT_BIG_OFFSET:
        var off = f32_from_bits(BITS_BIG_OFFSET)
        for i in range(n * v):
            out[i] = out[i] + off
        return out^

    if c.plant == CE_PLANT_SIGNED_ZEROS:
        # Both zero signs in EVERY row, adjacent, at the head of the row so
        # that a block fold of any width sees them in the same slab.
        for r in range(n):
            out[r * v] = f32_from_bits(BITS_POS_ZERO)
            if v > 1:
                out[r * v + 1] = f32_from_bits(BITS_NEG_ZERO)
        return out^

    if c.plant == CE_PLANT_SUBNORMAL:
        # Every fourth cell, three different subnormal patterns including a
        # NEGATIVE one, because `ftz` flushes to a zero of ITS OWN SIGN and
        # a fixture with only positive subnormals cannot see a flush that
        # drops the sign.
        var pats: List[UInt32] = [
            BITS_MIN_SUBNORMAL,
            BITS_BIG_SUBNORMAL,
            BITS_NEG_SUBNORMAL,
        ]
        var t = 0
        var i = 0
        while i < n * v:
            out[i] = f32_from_bits(pats[t % 3])
            t += 1
            i += 4
        return out^

    if c.plant == CE_PLANT_ARGMAX_TAIL or c.plant == CE_PLANT_ARGMAX_HEAD:
        if v <= 32:
            raise Error(
                String("loss_fixture: case ")
                + String(c.name)
                + " plants an argmax position and has vocab "
                + String(v)
                + " <= 32. L_MAX_TOPK_PREFIX only truncates at V > 32, so"
                + " the case would be inert for a reason that has nothing"
                + " to do with the plant ([[reached-but-inert]])."
            )
        # A value one binade above the whole hashed band, so it is the
        # maximum of its row NO MATTER WHAT the hash drew. `32.0` is the
        # band's open upper bound and `64.0` is safely past it; both are
        # exact.
        var big = Float32(64.0)
        for r in range(n):
            var at = 0
            if c.plant == CE_PLANT_ARGMAX_TAIL:
                at = v - 1
            out[r * v + at] = big
        return out^

    if c.plant == CE_PLANT_UNDERFLOW_TARGET:
        # The TARGET cell, 200 below a value that is itself planted as the
        # row maximum, so `shift[y]` is exactly `-200.0` and
        # `identical_exp(-200)` is exactly `+0.0`. The maximum is planted
        # at index 0 and the target at index `v - 1`, and the two are
        # distinct because `ce_case` gives this case `v == 16`.
        var hi = f32_from_bits(BITS_UNIFORM_VALUE)
        var lo = hi + f32_from_bits(BITS_SAT_GAP)
        for r in range(n):
            out[r * v] = hi
            out[r * v + (v - 1)] = lo
        return out^

    return out^


def ce_case_targets(c: CeCase) raises -> List[Int32]:
    """`[N]` Int32, hashed, with the ignored rows stamped in.

    `CE_PLANT_UNDERFLOW_TARGET` overrides the hash and points every
    unignored row at the LAST class, which is the cell `ce_case_logits`
    planted 200 low. If the two ever disagree about which index that is,
    the case stops separating `L_NLL_VIA_LOG_W` and looks inert -- so the
    index is written once, here and there, as `vocab - 1`, and
    `loss_check.mojo`'s preflight MEASURES that the resulting `ce.exp[y]`
    is exactly `+0.0` rather than trusting this paragraph."""
    var seed = ce_case_seed(ce_case_index_of(c))
    var out = List[Int32]()
    for r in range(c.n_rows):
        if c.ignore_stride > 0 and (r % c.ignore_stride) == 0:
            out.append(Int32(IGNORE_INDEX_DEFAULT))
            continue
        if c.plant == CE_PLANT_UNDERFLOW_TARGET:
            out.append(Int32(c.vocab - 1))
            continue
        out.append(ce_hashed_target(seed ^ (UInt64(TID_TARGETS) << 32),
                                    r, c.vocab))
    return out^


def ce_case_index_of(c: CeCase) raises -> Int:
    """The case's index, from its name.

    A case's SEED must be a function of its index and not of its shape, or
    two cases with the same shape and different plants would draw the same
    hash and the pair would stop being a controlled comparison for the
    wrong reason. This walks the table rather than carrying the index in
    the struct, because a struct field can be set to the wrong number and a
    name lookup cannot."""
    return ce_case_by_name(String(c.name))


def ce_case_unignored_count(c: CeCase) raises -> Int:
    var t = ce_case_targets(c)
    return ce_count(t, IGNORE_INDEX_DEFAULT)


def ce_case_divisor(c: CeCase) raises -> Float32:
    """The case's divisor from `ce_divisor`, THE ONE PRODUCER. Raises for
    `REDUCTION_NONE`, which has no divisor and no backward."""
    return ce_divisor(c.reduction, ce_case_unignored_count(c), c.num_items)


def ce_case_has_backward(c: CeCase) -> Bool:
    return c.reduction != REDUCTION_NONE


def ce_poison(n: Int) -> List[Float32]:
    """`n` copies of the poison pattern. DEVIATION 1456.

    Every device output buffer is filled with this BEFORE the call, so that
    an UNWRITTEN cell is distinguishable from a cell written with a zero.
    `SAB_IGNORED_ROW_SKIPPED` is ALWAYS INERT without it and never inert
    with it, which `loss.mojo`'s own switch docstring says in as many
    words."""
    var out = List[Float32]()
    var p = f32_from_bits(BITS_POISON)
    for _i in range(n):
        out.append(p)
    return out^


def count_poison(values: List[Float32]) -> Int:
    """Surviving poison cells, BY BITS. A nonzero count on a CLEAN build
    means a stage the card records was never written, and the card is then
    hashing whatever the allocator left there."""
    var n = 0
    for i in range(len(values)):
        if bits_of(values[i]) == BITS_POISON:
            n += 1
    return n


# ===========================================================================
# THE FIXTURE'S OWN ASSERTIONS
#
# Everything below is a property the case table CLAIMS and that some inert
# half depends on. Every one is MEASURED here rather than argued in a
# comment, because a fixture property that is only argued is a fixture
# property that is wrong on one case out of twenty-five -- and that is the
# hardest kind of flake to read, since the arm simply looks inert.
# ===========================================================================


def assert_no_zero_or_subnormal_logits(c: CeCase) raises -> Int:
    """No `+-0.0` and no subnormal in this case's logits. Returns the cell
    count checked.

    **THIS IS WHAT MAKES THE INERT HALVES MEASUREMENTS.**
    `L_MAX_PLAIN_COMPARE` is inert exactly on a row that does not carry
    both zero signs, and the whole case set except `adv_signed_zeros`
    claims to be such a set. A stray `+0.0` from the hash would make one
    case's inert assertion false and the arm would look broken.

    The two planted cases that DO carry zeros or subnormals are exempt and
    named, and the exemption is by PLANT KIND rather than by case name so
    that a new case cannot inherit it by accident."""
    if c.plant == CE_PLANT_SIGNED_ZEROS or c.plant == CE_PLANT_SUBNORMAL:
        return 0
    var x = ce_case_logits(c)
    for i in range(len(x)):
        if is_zero_bits(x[i]):
            raise Error(
                String("loss_fixture: case ")
                + String(c.name)
                + " logit "
                + String(i)
                + " is a signed ZERO ("
                + bits32_hex(x[i])
                + "). The hash is built so that cannot happen"
                + " (ce_hashed_logit's property 2), and every INERT half"
                + " of L_MAX_PLAIN_COMPARE in this lane rests on it."
            )
        if is_subnormal_bits(x[i]):
            raise Error(
                String("loss_fixture: case ")
                + String(c.name)
                + " logit "
                + String(i)
                + " is SUBNORMAL ("
                + bits32_hex(x[i])
                + "). ce_hashed_logit's magnitude band is [0.125, 32)."
            )
        if is_nonfinite_bits(x[i]):
            raise Error(
                String("loss_fixture: case ")
                + String(c.name)
                + " logit "
                + String(i)
                + " is NON-FINITE. Contract section 8 refuses those as"
                + " INPUTS and the audit plants them on the DEVICE; a"
                + " non-finite in the fixture would make every case refuse."
            )
    return len(x)


def assert_row_has_a_positive_logit(c: CeCase) raises:
    """Every row carries at least one strictly positive logit.

    `L_MAX_SEED_ZERO`'s INERT condition, exactly. The arm seeds the fold
    with `+0.0` instead of `-inf`, so it is inert if and only if the true
    maximum of every row keys at or above `+0.0` -- and a strictly positive
    logit is the cheap sufficient condition. `adv_all_negative` is the
    witness and is exempt by plant kind.

    DEVIATION 1490, AND THE FIRST RUN OF THIS GATE IS WHAT FOUND IT. This
    assertion ran over EVERY case and raised on `base_n1_v1`, whose single
    hashed logit is not positive. The assertion was right to fire and the
    conclusion drawn from it would have been wrong.

    **AT `V == 1` THIS ARM IS A WITNESS, NOT AN INERT CASE.** The vocab fold
    has ONE term, so seeding `+0.0` instead of `-inf` gives `max(+0.0, x)`,
    which is `+0.0` and not `x` whenever that one logit is negative. The arm
    MOVES there. Asserting the inert condition over a case where the arm
    fires is asserting the opposite of the truth, so `V == 1` is exempt BY
    SHAPE and named here rather than silently skipped. `base_n4_v8` remains
    the declared inert case and is where the condition is load-bearing."""
    if c.plant == CE_PLANT_ALL_NEGATIVE:
        return
    if c.vocab == 1:
        return
    var x = ce_case_logits(c)
    for r in range(c.n_rows):
        var seen = False
        for v in range(c.vocab):
            var b = bits_of(x[r * c.vocab + v])
            if (b & UInt32(0x80000000)) == UInt32(0) and not is_zero_bits(
                x[r * c.vocab + v]
            ):
                seen = True
        if not seen:
            raise Error(
                String("loss_fixture: case ")
                + String(c.name)
                + " row "
                + String(r)
                + " has NO strictly positive logit, so L_MAX_SEED_ZERO is"
                + " inert there for a reason the case does not claim"
                + " ([[reached-but-inert]])."
            )


def row_max_oracle(x: List[Float32], base: Int, vocab: Int) -> Float32:
    """The contract's row maximum, ASCENDING, seeded `-inf`.

    A copy of `loss_oracle.mojo::_row_max`'s loop, and the copy is
    deliberate: the SABOTAGE PREDICTIONS below need a maximum computed
    without going through the function under test, so that a prediction and
    the thing it predicts are not the same code. `loss_check.mojo`'s
    preflight asserts this agrees with the oracle's on every case, which is
    what turns the copy from a risk into a cross-check."""
    var m = bitcast[DType.float32](CE_NEG_INF_BITS)
    for v in range(vocab):
        m = identical_fmax(m, x[base + v])
    return m


def row_max_seeded_zero(x: List[Float32], base: Int, vocab: Int) -> Float32:
    """`L_MAX_SEED_ZERO`'s answer, computed on the HOST. The fold shape is
    free (`identical_fmax` is exactly associative), so the ascending loop
    here and the device's block halving tree return the same bits, which is
    what makes this a legitimate prediction rather than a guess."""
    var m = Float32(0.0)
    for v in range(vocab):
        m = identical_fmax(m, x[base + v])
    return m


def row_max_topk_prefix(x: List[Float32], base: Int, vocab: Int) -> Float32:
    """`L_MAX_TOPK_PREFIX`'s answer: the fold over the first 32 classes
    only. Same associativity argument as `row_max_seeded_zero`."""
    var limit = vocab
    if limit > 32:
        limit = 32
    var m = bitcast[DType.float32](CE_NEG_INF_BITS)
    for v in range(limit):
        m = identical_fmax(m, x[base + v])
    return m


def case_has_both_zero_signs(c: CeCase) raises -> Bool:
    """Does some row carry BOTH a `+0.0` and a `-0.0`?

    `L_MAX_PLAIN_COMPARE`'s separating condition, and the only one of the
    twenty-five predictions in this file that is a CONDITION rather than a
    simulation. A plain `>` fold is NOT associative in the presence of
    signed zeros, so the device's block tree and a host ascending loop can
    disagree about it -- which means the arm's RESULT cannot honestly be
    predicted from the host, only its INERTNESS can. `loss_check.mojo`
    treats it as MUST-NOT-MOVE where this is False and as UNSPECIFIED where
    it is True, and says so."""
    var x = ce_case_logits(c)
    for r in range(c.n_rows):
        var pos = False
        var neg = False
        for v in range(c.vocab):
            var b = bits_of(x[r * c.vocab + v])
            if b == BITS_POS_ZERO:
                pos = True
            if b == BITS_NEG_ZERO:
                neg = True
        if pos and neg:
            return True
    return False


def case_row_nll_is_negative_zero(c: CeCase) raises -> Bool:
    """Does some row's `nll` come out EXACTLY `-0.0`?

    `L_NEG_VIA_ZERO_SUB`'s and `L_SMOOTH_ALWAYS_SPELLED`'s separating
    condition, computed here from the row's own arithmetic rather than
    assumed from `V == 1`. It is true whenever `lp_y` is exactly `+0.0`,
    which contract 8.1 says happens when the target IS the argmax and
    `denom` is exactly `1.0` -- reachable at `V == 1` and at any `V` where
    every non-target exponential underflows to `+0.0`.

    **THE SECOND CASE IS WHY THIS IS COMPUTED AND NOT HARDCODED TO
    `V == 1`.** A fixture that only ever asserted `V == 1` would miss a
    future case that reached the same row shape another way, and the arm
    would be reported inert on a case where it is not."""
    var x = ce_case_logits(c)
    var t = ce_case_targets(c)
    var cfg = ce_case_config(c)
    var v = c.vocab
    # `ce_fold`, THE CONTRACT'S FOLD, and not a hand-written chain. A chain
    # would be right only at `V <= 128` (where `P == 1` and the tree
    # performs no addition) and would be a DIFFERENT ANSWER above it, so a
    # prediction built on one would be wrong on exactly the cases the fold
    # arms exist for. Contract 5.3.
    var ones = ce_ones(v)
    for r in range(c.n_rows):
        var y = Int(t[r])
        if y == cfg.ignore_index:
            continue
        var base = r * v
        var m = row_max_oracle(x, base, v)
        var expo = List[Float32]()
        for vv in range(v):
            expo.append(identical_exp(ftz(ftz(x[base + vv]) - ftz(m))))
        var denom = ce_fold(expo, 0, v, ones)
        var logdenom = ftz(identical_log(ftz(denom)))
        var sy = ftz(ftz(x[base + y]) - ftz(m))
        var lp = ftz(ftz(sy) - ftz(logdenom))
        if bits_of(neg_by_bits(lp)) == BITS_NEG_ZERO:
            return True
    return False


def case_target_exp_underflows(c: CeCase) raises -> Bool:
    """Is some row's `e[y]` EXACTLY `+0.0`?

    `L_NLL_VIA_LOG_W`'s separating condition. The refused spelling
    `-log(w[y])` then returns `+inf` where the pinned one returns an
    ordinary finite number, so the arm is loud here and BITWISE INERT
    wherever every `e[y]` is normal."""
    var x = ce_case_logits(c)
    var t = ce_case_targets(c)
    var cfg = ce_case_config(c)
    var v = c.vocab
    for r in range(c.n_rows):
        var y = Int(t[r])
        if y == cfg.ignore_index:
            continue
        var base = r * v
        var m = row_max_oracle(x, base, v)
        var s = ftz(ftz(x[base + y]) - ftz(m))
        if is_zero_bits(identical_exp(s)):
            return True
    return False


def case_leaf_count(k: Int) -> Int:
    """`P` for a fold of length `k` under gemm v1, restated here so the
    fold-arm predictions do not have to import the GEMM oracle.

    **IT IS A RESTATEMENT AND A RESTATEMENT CAN DRIFT.**
    `loss_check.mojo`'s preflight asserts it against
    `gemm_oracle.contract_leaf_size` on every `k` the case set uses, which
    is the same mitigation the transformer lane applied to its copied
    splitmix64."""
    if k <= 0:
        return 0
    if k <= 128:
        return 1
    var p0 = (k + 127) // 128
    if p0 <= 1024:
        return p0
    return 1024


def case_fold_has_carry(k: Int) -> Bool:
    """Does the balanced tree over `P = case_leaf_count(k)` leaves ever
    carry an ODD level width?

    `L_DENOM_PAD_PLUS_ZERO`'s separating condition, and the reason
    `V = 256` and `V = 1024` are the arm's INERT halves while `V = 300` and
    the shipped 128256 are its witnesses. A level of even width pairs
    exactly and has nothing to pad; a level of odd width carries its last
    element up, and padding it with a `+0.0` instead is a DIFFERENT ANSWER
    only when there is something to pad."""
    var w = case_leaf_count(k)
    while w > 1:
        if (w % 2) != 0:
            return True
        w = w // 2
    return False


def describe_case(c: CeCase) raises -> String:
    """One line naming everything a reader needs to evaluate a verdict on
    this case, including the two fold shapes, because a fold arm's verdict
    is unreadable without `P`."""
    var count = ce_case_unignored_count(c)
    var red = String("NONE")
    if c.reduction == REDUCTION_SUM:
        red = String("SUM")
    elif c.reduction == REDUCTION_MEAN:
        red = String("MEAN")
    var eps = String("0")
    if c.eps_on:
        eps = String("0.1")
    var out = (
        String(c.name)
        + "  N="
        + String(c.n_rows)
        + " V="
        + String(c.vocab)
        + " "
        + red
        + " eps="
        + eps
        + " count="
        + String(count)
        + " plant="
        + plant_name(c.plant)
        + "  Pv="
        + String(case_leaf_count(c.vocab))
    )
    if case_fold_has_carry(c.vocab):
        out += "(carry)"
    out += " Pn=" + String(case_leaf_count(c.n_rows))
    if case_fold_has_carry(c.n_rows):
        out += "(carry)"
    if c.reduction != REDUCTION_NONE:
        out += "  divisor=" + bits32_hex(ce_case_divisor(c))
    return out^
