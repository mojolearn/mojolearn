# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The fixture of profile `mojolearn.identical.optimizer.fp32.v1`.

`optimizer_check.mojo` consumes these cases and records stage comparisons
at every step. Execution evidence belongs to the corresponding logs and
cards under `bench/results/`, scoped to their build, fixture and device.
The fixture alone does not establish numerical correctness or cross-vendor
identity.

WHAT IT IS FOR
---------------
Contract section 12's last column -- "must not pass on" -- is the fixture
property WITHOUT WHICH AN ARM IS BIT-INERT, and it is the reason this file
exists. Almost every arm in this lane has a configuration in which it
changes nothing, and for FIVE of them that configuration is THE REFERENCE'S
OWN DEFAULT:

  * `OPT_SAB_ADAMW_AS_ADAM` is inert at `weight_decay == 0`, and
    `torch.optim.Adam`'s default IS `weight_decay = 0`. Contract 7.4 calls
    this "the single most likely vacuous gate in the lane".
  * `OPT_SAB_MOMENTUM_FIRST_STEP` is inert at `dampening == 0`, which is
    the default, because `c_damp` is then exactly `1.0` and
    `identical_mul(1.0, g)` returns `g`.
  * `OPT_SAB_NESTEROV_ORDER` is inert at `momentum == 0` AND at `t = 1`,
    where the buffer is a COPY of `g` so `b == g` and the two operand
    orders are the same expression.
  * `OPT_SAB_POW_RUNNING` is inert at `t <= 6`. A gate whose fixture stops
    at `t = 4` -- which is what a hand-written fixture does -- is VACUOUS.
  * `OPT_SAB_MOMENT_LERP` is inert whenever `m_prev` is exactly `+0.0`,
    which is EVERY FIRST STEP FROM FRESH STATE. `m + c1*(g - m)` at
    `m = 0` is `c1*g`, and the pinned `fma(c1, g, ftz(beta1*0))` is
    `c1*g` too. **So an optimizer gate that only ever runs one step cannot
    see that arm at all**, and that is not in contract section 12's table.
    It is DEVIATION 1474 and it is why every case here carries a STEP
    COUNT rather than a step index.

THE CASES CARRY A STEP COUNT, NOT A STEP INDEX (DEVIATION 1474)
-----------------------------------------------------------------
`OptCase.steps` is how many steps to run FROM FRESH STATE, and the gate
compares the oracle and the device AT EVERY ONE of them. Three clauses need
that and none of them can be spelled with a single step:

  1. `m` and `v` and the momentum buffer are RUNNING STATE. An arm that
     changes how `m` is updated is invisible on the step that creates it.
  2. Contract 5.1's `beta^t` clause first separates at `t = 7`, so a
     fixture must reach 7 and should reach 1000.
  3. Contract 11(d)'s step-count invariance is a claim ABOUT TWO RUNS of
     different lengths and there is nothing to compare in one step.

THE HASHED VALUES, AND WHY THEY ARE BUILT THE SAME WAY AS THE LOSS LANE'S
---------------------------------------------------------------------------
`opt_hashed` builds `+-(1 + m) * 2^e` with `m` a 23-bit fraction, exactly as
`loss_fixture.ce_hashed_logit` does and for the same three reasons: the
value is EXACTLY REPRESENTABLE so the `Float32` cast rounds nothing, it is
NEVER `+-0.0` and never subnormal so every "inert on ordinary values"
assertion is a measurement, and it spans eight binades so a fold has
something to be wrong about.

**THE SUBNORMAL AND NEAR-SUBNORMAL VALUES ARE PLANTS AND ARE NEVER DRAWN.**
Contract section 6's `OPT_SAB_FTZ_LATE` needs a gradient near `1e-25`,
where the VALUE is a perfectly ordinary normal and the SQUARE is not
representable at all; contract 3.4c's `OPT_SAB_CLIP_SKIP_AT_ONE` needs a
SUBNORMAL gradient cell. Both are planted by name, so a case that does not
name them provably has neither.

WHAT THIS FILE CANNOT DO, and the two arms it could not build a separating
fixture for
-----------------------------------------------------------------------------
* **`OPT_SAB_RECIP_MUL` HAS NO CONSTRUCTIBLE INERT CASE.** Contract 4c says
  `x * (1/d)` is EXACT when `d` is a power of two, so an inert half needs
  the Adam denominator `dn = ftz(ftz(identical_div(ftz(identical_sqrt(v)),
  rt_bc2)) + eps)` to come out an exact power of two. Every term in that
  chain is data dependent -- `v` is running state, `rt_bc2` is
  `sqrt(1 - beta2^t)`, and `eps` is `1e-8` -- and there is no assignment of
  the fixture's free variables that makes the SUM a power of two other than
  by search. The arm therefore has a witness and no inert half in this
  file, it is a SMOKE TEST rather than a reach proof, and
  `optimizer_check.mojo` says so where the verdict is printed rather than
  letting a reader assume otherwise.
* **`OPT_SAB_MHAT_FORM` has no inert case either**, and contract section 12
  says so itself: "nothing known to make it inert, but it is a
  1-ulp-class arm -- report the cell count, do not assume". The check
  reports the count.

TRAPS THIS FILE IS WRITTEN AROUND
-----------------------------------
* `[[mojo-amp-plus-is-bitwise-and]]`. Every `+` on a `UInt64` here is a real
  wrapping add.
* `[[mojo-list-float32-not-implicitly-copyable]]`. Every copy is explicit.
* `[[mojo-string-float-roundtrip]]`. **Every hyperparameter is a BIT
  PATTERN**, which is contract section 1's rule and not this file's
  preference: `0.999` narrowed from a float64 parse and a direct `Float32`
  literal are the same number, but a run's card records the pattern and two
  runs are comparable only when the patterns are equal.
* `[[mojo-int-widening-sign-extends]]`. Every hash shift is on `UInt64`.

LIMITATIONS
-----------
* **A corpus.** Contract 16.10 lists five adversarial cases by name
  (`adv_subnormal_square`, `adv_dead_unit_v`, `adv_dampening_first_step`,
  `adv_param_order_five`, `adv_pow_step_1000`). All five exist HERE as
  fixture cases and NONE of them exists as a corpus record with an
  independently produced expected value, which is a different thing.
* **`OPT_SAB_RECIP_MUL`'s inert half**, above.
"""

from std.memory import bitcast

from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_mul,
    numeric_mode_name,
)
from training.checks.optimizer_oracle import (
    OPT_ADAM,
    OPT_ADAMW,
    OPT_SGD,
    OptimizerConfig,
    adam_element_oracle,
    clip_eps,
    step_scalars,
)


# ===========================================================================
# BITS
# ===========================================================================
# Contract section 1: **HYPERPARAMETERS ARE BIT PATTERNS, NOT DECIMAL
# STRINGS.** A run's card records each one as eight hex digits and two runs
# are comparable only when those patterns are equal. This is
# `[[mojo-string-float-roundtrip]]` applied to CONFIGURATION rather than to
# output, and it matters more than it looks: the DERIVED quantity
# `1 - beta2` is not the same number by a Float32 route and a float64 route,
# and contract 5.3 measures the gap at about 1.3e-5 RELATIVE, which is the
# third significant decimal of the coefficient that drives `v`.
# ===========================================================================


def f32_from_bits(u: UInt32) -> Float32:
    return bitcast[DType.float32](u)


def bits_of(v: Float32) -> UInt32:
    return bitcast[DType.uint32](v)


#: Contract section 1's table, verbatim.
comptime BITS_BETA1: UInt32 = 0x3F666666  # 0.9
comptime BITS_BETA2: UInt32 = 0x3F7FBE77  # 0.999
comptime BITS_ADAM_EPS: UInt32 = 0x322BCC77  # 1e-8
comptime BITS_CLIP_EPS: UInt32 = 0x358637BD  # 1e-6, and it is NOT a caller's
comptime BITS_ONE: UInt32 = 0x3F800000

#: `2^-10`, this fixture's learning rate. An EXACT power of two, chosen so
#: that `lr` itself contributes no rounding and every arm that turns on a
#: rounding is turning on its OWN rounding rather than the learning rate's.
comptime BITS_LR: UInt32 = 0x3A800000

#: `0.1`, the weight decay of the `wd != 0` cases. `lr * wd` is then
#: `2^-10 * 0.1`, which is NOT a power of two -- which is what
#: `OPT_SAB_DECAY_ADD_FORM` needs, because `p - lr*wd*p` and
#: `p * (1 - lr*wd)` agree when the product is exact.
comptime BITS_WD: UInt32 = 0x3DCCCCCD

#: `8.0`. `lr * wd` is then exactly `2^-7`, a power of two, which is
#: `OPT_SAB_DECAY_ADD_FORM`'s INERT half. It is a large decay and that is
#: fine: `decay_mul` is `1 - 2^-7`, an ordinary number.
comptime BITS_WD_POW2: UInt32 = 0x41000000

#: `0.9`, SGD's momentum.
comptime BITS_MOMENTUM: UInt32 = 0x3F666666

#: `0.25`, SGD's dampening. EXACT, and non-zero, which is the ONLY thing
#: that makes `OPT_SAB_MOMENTUM_FIRST_STEP` visible -- at `dampening == 0`,
#: `c_damp` is exactly `1.0` and `identical_mul(1.0, g)` returns `g` for
#: every finite `g`, so the reference's own default configuration cannot
#: see clause 7.3a at all.
comptime BITS_DAMPENING: UInt32 = 0x3E800000

#: `1.0`, the clip threshold of the clipping cases.
comptime BITS_MAX_NORM: UInt32 = 0x3F800000

#: `1e30`. A clip threshold so large that `coef` clamps to exactly `1.0` on
#: every fixture here, which is the configuration `OPT_SAB_CLIP_SKIP_AT_ONE`
#: models -- the arm SKIPS the rescale when `coef_c == 1.0`, so it is only
#: reachable when the clamp actually fires.
comptime BITS_MAX_NORM_HUGE: UInt32 = 0x7149F2CA

#: A gradient near `1e-25`. **THE VALUE IS A PERFECTLY ORDINARY NORMAL AND
#: THE SQUARE IS NOT REPRESENTABLE AT ALL** (`1e-50` is far below `2^-126`),
#: which is contract section 6's seam O7 and `OPT_SAB_FTZ_LATE`'s only
#: separating input. It is not reachable from a uniform random fixture.
comptime BITS_TINY_GRAD: UInt32 = 0x15F79688

#: A `v` in the `1e-20` to `1e-12` band contract 4d names. With
#: `eps = 1e-8`, `eps^2` is `1e-16`, so this is where `sqrt(v_hat) + eps`
#: and `sqrt(v_hat + eps)` stop agreeing. A real training run reaches it on
#: a dead unit and a synthetic fixture never reaches it by accident.
comptime BITS_DEAD_V: UInt32 = 0x26901D7D  # about 1e-15

#: A subnormal gradient cell, contract 3.4c. `identical_mul(1.0, x)` returns
#: `x` exactly for every finite NORMAL `x`, so the unconditional rescale and
#: a skip agree everywhere EXCEPT here, where the multiply's operand and
#: result flushes turn the value into a signed zero and the skip leaves the
#: original pattern in the buffer.
comptime BITS_SUBNORMAL_GRAD: UInt32 = 0x00300000
comptime BITS_NEG_SUBNORMAL_GRAD: UInt32 = 0x80000101

comptime BITS_POS_ZERO: UInt32 = 0x00000000
comptime BITS_NEG_ZERO: UInt32 = 0x80000000
comptime BITS_POS_INF: UInt32 = 0x7F800000
comptime BITS_QNAN: UInt32 = 0x7FC00000

#: **THE POISON.** Every device output buffer is pre-filled with this, so
#: that a cell NO KERNEL WROTE is distinguishable from a cell written with a
#: zero. There is no `SAB_*_SKIPPED` arm in this lane, but the same hazard
#: exists without one: `sgd_update_kernel` writes `dir_out` only under
#: `MOJOLEARN_OPT_RECORD`, and `adam_update_kernel` writes `denom_out` and
#: `q_out` the same way, so a gate that recorded those stages from a build
#: WITHOUT the define would be hashing the allocator's leftovers and calling
#: it a card.
comptime BITS_POISON: UInt32 = 0x7FC0DEAD


def bits32_hex(v: Float32) -> String:
    var u = bits_of(v)
    var digits = String("0123456789abcdef")
    var out = String("0x")
    for i in range(8):
        var shift = UInt32(28 - 4 * i)
        var nib = Int((u >> shift) & UInt32(0xF))
        out += String(digits[byte=nib])
    return out^

def is_nonfinite_bits(v: Float32) -> Bool:
    """NaN or infinity, BY BITS AND NEVER BY A COMPARE. Metal flushes
    compare operands (IDENTITY_PATHS row 49), so `v != v` is a test with two
    meanings across columns. Contract 8a is written in these terms."""
    return (bits_of(v) & UInt32(0x7FFFFFFF)) >= UInt32(0x7F800000)


def is_zero_bits(v: Float32) -> Bool:
    return (bits_of(v) & UInt32(0x7FFFFFFF)) == UInt32(0)


def is_subnormal_bits(v: Float32) -> Bool:
    var a = bits_of(v) & UInt32(0x7FFFFFFF)
    return a != UInt32(0) and a < UInt32(0x00800000)


def is_exact_power_of_two(v: Float32) -> Bool:
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


def opt_profile_constants_are_intact() -> Bool:
    """Contract section 1's table, checked BY BITS.

    A wrong `beta2` does not look like a wrong `beta2`. It looks like a
    kernel bug at `adam.v` and at every stage downstream of it, on every
    step, for ever. `optimizer_check.mojo`'s preflight calls this BEFORE
    any device call."""
    if bits_of(f32_from_bits(BITS_BETA2)) != BITS_BETA2:
        return False
    if bits_of(clip_eps()) != BITS_CLIP_EPS:
        return False
    if bits_of(Float32(1.0)) != BITS_ONE:
        return False
    if not is_exact_power_of_two(f32_from_bits(BITS_LR)):
        return False
    return True


# ===========================================================================
# THE HASH, spec `mojolearn.optimizer.fixture.hash.v1` (DEVIATION 1470)
# ===========================================================================


def opt_splitmix64(z_in: UInt64) -> UInt64:
    """splitmix64, Steele/Lea/Flood's finalizer, verbatim.

    **A FOURTH COPY IN THIS REPOSITORY**, after the mamba corpus's, the
    transformer fixture's (DEVIATION 1000) and the loss fixture's. The
    argument for copying is DEVIATION 1000's -- exact integer arithmetic
    cannot drift the way a float seam can -- and the cost is the one that
    docstring names, multiplied again: four copies of a hash have four
    chances to be edited apart. `optimizer_check.mojo`'s preflight asserts
    all four agree on five seeds.

    `[[mojo-amp-plus-is-bitwise-and]]`: **EVERY `+` BELOW IS A REAL WRAPPING
    ADD.** Mojo's `&+` computes `x & k` with no compile error and has
    silently produced wrong hashes here twice."""
    var z = z_in + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


#: Distinct from the loss lane's `0x43654C6F73734631`, the transformer
#: fixture's and the mamba corpus's. Two lanes sharing a seed base makes two
#: fixtures correlate, which is harmless right up to the moment somebody
#: compares a hash across lanes and reads meaning into it.
comptime OPT_SEED_BASE: UInt64 = 0x4F70744D6F6A6F31

comptime TID_PARAM = 1
comptime TID_GRAD = 2
comptime TID_M = 3
comptime TID_V = 4


def opt_case_seed(k: Int) -> UInt64:
    return OPT_SEED_BASE + UInt64(0x1000) * UInt64(k)


def opt_hashed(seed: UInt64, idx: Int, binade_shift: Int) -> Float32:
    """One hashed value: `+-(1 + m) * 2^(e + binade_shift)`, `m` a 23-bit
    fraction and `e` in `[-3, 4]`.

    THE THREE PROPERTIES, all structural and all MEASURED by
    `assert_no_zero_or_subnormal`:

      1. EXACTLY REPRESENTABLE. `1 + m` at 23 fraction bits is exact in
         `[1, 2)` -- that is the Float32 significand -- and scaling by a
         power of two is exact at every exponent in range. The cast rounds
         nothing.
      2. NEVER `+-0.0` AND NEVER SUBNORMAL, provided `binade_shift` stays
         inside `[-100, 100]`, which every caller here does. Every "inert on
         ordinary values" assertion in this lane rests on it.
      3. BOTH SIGNS AND EIGHT BINADES per tensor, and `binade_shift` moves
         whole TENSORS apart. **That is contract 3.1's requirement made
         concrete**: the two-level clip norm and a flat one agree at `J = 1`
         and diverge when the per-tensor norms differ by several binades,
         because each per-tensor `sqrt` rounds and is then squared again
         inside the outer norm.

    The scaling is a LOOP of exact multiplies and divides by `2.0` rather
    than a `pow`, for contract 5.1's reason at a much smaller site:
    `portable_powf` is `exp(p * log(x))` through two Cephes polynomials and
    is not exact even at `p = 1`."""
    var h = opt_splitmix64(seed + UInt64(idx))
    var frac = Int((h >> 41) & UInt64(0x7FFFFF))
    var mant = Float64(frac) * 1.1920928955078125e-07  # 2^-23, exactly
    var v = Float32(1.0 + mant)
    var e = Int((h >> 3) & UInt64(0x7)) - 3 + binade_shift
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


# ===========================================================================
# THE PARAMETER REGISTRY (contract 3.3)
# ===========================================================================
# **`param_id` IS PART OF THE PROFILE.** It is assigned once when the model
# is registered, it is written into the checkpoint, and it is NOT a
# dictionary iteration order, not the order the optimizer happened to
# receive the tensors in, not a device pointer order and not a name sort
# that changes when a layer is renamed. Two runs with the same parameters in
# a different `param_id` order are two different numerical experiments.
#
# It is the clause contract 3.3 says is most likely to be waved past, and
# the reason it is easy to wave past is that it LOOKS like bookkeeping. It
# is not. It is the cross-tensor summation order, and a summation order is
# the thing this repository exists to pin.
#
# `offsets` below IS that registry: `offsets[j] .. offsets[j+1]` is tensor
# `j`, `j` is the `param_id`, and ascending `j` is the fold order.
# ===========================================================================

comptime SHAPE_J5_MIXED = 0
comptime SHAPE_J2 = 1
comptime SHAPE_J1 = 2
comptime SHAPE_J3_SMALL = 3
comptime SHAPE_J3_RAGGED = 4
comptime SHAPE_J1_BIG = 5
comptime SHAPE_J3_SPREAD = 6


def opt_sizes(shape: Int) raises -> List[Int]:
    """The per-tensor element counts of a registry shape.

    THE FOLD ARITHMETIC, since four arms turn on it. Under gemm v1 the leaf
    is `contract_leaf_size(N)` and `P = ceil(N / L)`, and at `N <= 128` the
    leaf is the whole tensor so `P == 1` and **the tree performs NO
    addition** (gemm contract 7.3). So:

        N       P     ragged?          what it exercises
        8       1     n/a              nothing; the serial chain IS v1
        64      1     n/a              nothing
        128     1     n/a              the boundary, still nothing
        130     2     yes, 2 cells     the smallest real tree here
        200     2     yes, 72 cells    a second raggedness
        300     3     yes, 44 cells    ragged AND a CARRY (widths 3, 2, 1)
        1025    9     yes, 1 cell      P = 9, carries at 9 and at 5
        4096    32    no               thousands of lanes for OPT_SAB_RSQRT
    """
    var out = List[Int]()
    if shape == SHAPE_J5_MIXED:
        # J = 5. Contract 3.3: "J >= 3 is required and J = 5 is better --
        # gemm contract 7.2.2 notes P = 5 is the smallest P that carries an
        # odd tail twice." Five DIFFERENT lengths, one of them 300 (ragged
        # AND carrying) and one of them 1025 (P = 9).
        out.append(7)
        out.append(130)
        out.append(300)
        out.append(64)
        out.append(1025)
        return out^
    if shape == SHAPE_J2:
        # **`J == 2` CANNOT SEE A PARAMETER-ORDER SABOTAGE AT ALL**, because
        # reversing two elements swaps the two children of ONE tree node and
        # `a + b` equals `b + a` bitwise. That is exactly why this shape is
        # in the table: it is `OPT_SAB_CLIP_PARAM_ORDER`'s INERT half and
        # there is no other way to have one.
        out.append(130)
        out.append(300)
        return out^
    if shape == SHAPE_J1:
        # `J == 1`. `OPT_SAB_CLIP_FLAT_NORM`'s inert half: at one tensor the
        # two-level form is `sqrt(sqrt(s)^2)` and the flat one is `sqrt(s)`,
        # which agree for most `s` because squaring a rounded square root
        # usually lands back in the same binade. Contract 3.1 says so and
        # the arm's verdict here is a MEASUREMENT of "usually".
        out.append(300)
        return out^
    if shape == SHAPE_J3_SMALL:
        # Every `N <= 128`, so `P == 1` everywhere and the v1 answer IS the
        # serial ascending chain. `OPT_SAB_CLIP_SERIAL_FOLD`'s inert half,
        # and contract 3.2 names this shape as the one that makes the arm
        # pass while gating nothing.
        out.append(8)
        out.append(64)
        out.append(128)
        return out^
    if shape == SHAPE_J3_RAGGED:
        # `OPT_SAB_CLIP_SERIAL_FOLD`'s witness. One tensor at 300 -- `P = 3`
        # with a 44-element ragged last leaf, the gemm lane's own ragged
        # fixture -- beside two that are inert, so the arm's effect is
        # localized to one `clip.sumsq` cell.
        out.append(8)
        out.append(300)
        out.append(64)
        return out^
    if shape == SHAPE_J1_BIG:
        # 4096 elements. Contract 4b: DEVIATION 741 measured
        # `identical_rsqrt` off the correctly rounded rsqrt on 134,858 of
        # 520,133 positive normals, so about a quarter of inputs separate
        # the pin from the intrinsic and **a handful of round `v` values
        # will pass**. Thousands of hashed lanes is the fixture that does
        # not.
        out.append(4096)
        return out^
    if shape == SHAPE_J3_SPREAD:
        # Three tensors whose norms differ by several binades, which is
        # `OPT_SAB_CLIP_FLAT_NORM`'s stated requirement. The spread comes
        # from `opt_binade_shift`, not from the lengths.
        out.append(130)
        out.append(300)
        out.append(200)
        return out^
    raise Error(
        String("optimizer_fixture: shape ") + String(shape) + " does not exist"
    )


def opt_offsets(shape: Int) raises -> List[Int]:
    """`offsets`, length `J + 1`. `offsets[j] .. offsets[j+1]` is tensor
    `j`, and `j` IS the `param_id`."""
    var sizes = opt_sizes(shape)
    var out = List[Int]()
    out.append(0)
    var acc = 0
    for j in range(len(sizes)):
        acc += sizes[j]
        out.append(acc)
    return out^


def opt_binade_shift(shape: Int, j: Int) -> Int:
    """How many binades tensor `j`'s values are moved by.

    **THIS IS `OPT_SAB_CLIP_FLAT_NORM`'S SEPARATING PROPERTY AND NOTHING
    ELSE.** Contract 3.1: in exact arithmetic the two-level norm and the
    flat one are the same number, and in `Float32` they are not, because
    each per-tensor `sqrt` rounds and each result is then squared again
    inside the outer norm. Tensors whose norms are all within a binade give
    the rounding nothing to accumulate. `4 * j - 8` spreads five tensors
    over 32 binades, which is far enough that no reader has to wonder
    whether the spread is enough.

    The shift is EXACT (a power-of-two scaling) so it introduces no rounding
    of its own, and `assert_no_zero_or_subnormal` MEASURES that the result
    stays inside the normal range."""
    return 4 * j - 8


# ===========================================================================
# THE PLANTS
# ===========================================================================

comptime OPT_PLANT_NONE = 0

#: Contract 3.4c / `OPT_SAB_CLIP_SKIP_AT_ONE`. Subnormal gradient cells,
#: with a NEGATIVE one among them because `ftz` flushes to a zero of ITS OWN
#: SIGN and a fixture with only positive subnormals cannot see a flush that
#: drops the sign.
comptime OPT_PLANT_SUBNORMAL_GRAD = 1

#: Contract section 6 / `OPT_SAB_FTZ_LATE`. A gradient near `1e-25`, where
#: the VALUE is normal and the SQUARE is not.
comptime OPT_PLANT_TINY_GRAD = 2

#: Contract 4d / `OPT_SAB_EPS_INSIDE_SQRT`. The initial `v` planted in the
#: `1e-20` to `1e-12` band.
comptime OPT_PLANT_DEAD_V = 3

#: Contract 7.2d / `OPT_SAB_UNFUSED_UPDATE`. **BUILT TO SEPARATE.** See
#: `opt_case_param`.
comptime OPT_PLANT_FUSED_SEPARATOR = 4

#: The same construction pushed forty binades apart, so the product's low
#: bits vanish in the addition and the two spellings agree. The INERT half.
comptime OPT_PLANT_FUSED_INERT = 5

#: Contract 7.2b / `OPT_SAB_SQ_ASSOC`'s inert half. Every gradient EXACTLY
#: `1.0`, so `c2 * (g*g)` and `(c2*g) * g` are both exactly `c2`.
comptime OPT_PLANT_UNIT_GRAD = 6


def plant_name(p: Int) -> String:
    if p == OPT_PLANT_NONE:
        return String("none")
    if p == OPT_PLANT_SUBNORMAL_GRAD:
        return String("subnormal_grad")
    if p == OPT_PLANT_TINY_GRAD:
        return String("tiny_grad")
    if p == OPT_PLANT_DEAD_V:
        return String("dead_v")
    if p == OPT_PLANT_FUSED_SEPARATOR:
        return String("fused_separator")
    if p == OPT_PLANT_FUSED_INERT:
        return String("fused_inert")
    if p == OPT_PLANT_UNIT_GRAD:
        return String("unit_grad")
    return String("plant?")


# ===========================================================================
# THE CASE TABLE
# ===========================================================================


@fieldwise_init
struct OptCase(Copyable, Movable):
    """One runnable optimizer run.

    `steps` is how many steps to take FROM FRESH STATE, and the gate
    compares at EVERY one of them. DEVIATION 1474 and the module docstring
    are the argument; the short version is that `m`, `v` and the momentum
    buffer are RUNNING STATE, so three arms are invisible on the step that
    creates them and one clause (contract 11(d)) is a statement about two
    runs of different lengths.

    `hp` selects a hyperparameter preset (`opt_config`). It is an id rather
    than nine fields because contract section 1 makes every one of them a
    BIT PATTERN, and nine patterns per case restated twenty-seven times is
    twenty-seven chances for one of them to be typed wrong."""

    var name: StaticString
    var shape: Int
    var hp: Int
    var steps: Int
    var plant: Int


comptime HP_ADAM_PLAIN = 0
comptime HP_ADAM_CLIP = 1
comptime HP_ADAMW_WD = 2
comptime HP_ADAMW_WD0 = 3
comptime HP_ADAMW_WD_POW2 = 4
comptime HP_ADAM_WD = 5
comptime HP_SGD_DAMP = 6
comptime HP_SGD_NODAMP = 7
comptime HP_SGD_NESTEROV = 8
comptime HP_SGD_MOM0 = 9
comptime HP_ADAM_CLIP_HUGE = 10


def opt_config(hp: Int) raises -> OptimizerConfig:
    """One hyperparameter preset. Every scalar comes from a BIT PATTERN.

    `max_norm <= 0.0` means the clip is OFF, and contract 11(c) requires the
    parameter-count-invariance gate to run in exactly that configuration:
    **with clipping ON, one parameter's update depends on every other
    parameter in the model**, by the reference's own semantics and not by a
    defect (contract 3.5). That is not repairable and it is what a GLOBAL
    norm clip means."""
    var lr = f32_from_bits(BITS_LR)
    var b1 = f32_from_bits(BITS_BETA1)
    var b2 = f32_from_bits(BITS_BETA2)
    var eps = f32_from_bits(BITS_ADAM_EPS)
    var zero = Float32(0.0)
    if hp == HP_ADAM_PLAIN:
        return OptimizerConfig(
            OPT_ADAM, lr, b1, b2, eps, zero, zero, zero, False, zero
        )
    if hp == HP_ADAM_CLIP:
        return OptimizerConfig(
            OPT_ADAM, lr, b1, b2, eps, zero, zero, zero, False,
            f32_from_bits(BITS_MAX_NORM),
        )
    if hp == HP_ADAM_CLIP_HUGE:
        # `max_norm = 1e30` makes `coef` clamp to exactly `1.0`, which is
        # the ONLY configuration `OPT_SAB_CLIP_SKIP_AT_ONE` is reachable in
        # -- the arm skips the rescale when `coef_c == 1.0`.
        return OptimizerConfig(
            OPT_ADAM, lr, b1, b2, eps, zero, zero, zero, False,
            f32_from_bits(BITS_MAX_NORM_HUGE),
        )
    if hp == HP_ADAMW_WD:
        return OptimizerConfig(
            OPT_ADAMW, lr, b1, b2, eps, f32_from_bits(BITS_WD), zero, zero,
            False, zero,
        )
    if hp == HP_ADAMW_WD0:
        # **THE REFERENCE'S OWN DEFAULT.** At `weight_decay == 0` Adam and
        # AdamW are the SAME ARITHMETIC, so `OPT_SAB_ADAMW_AS_ADAM` is
        # bit-inert here. Contract 7.4 calls this the single most likely
        # vacuous gate in the lane, which is precisely why the preset
        # exists: it is the arm's INERT half and it has to be run.
        return OptimizerConfig(
            OPT_ADAMW, lr, b1, b2, eps, zero, zero, zero, False, zero
        )
    if hp == HP_ADAMW_WD_POW2:
        # `lr * wd` is exactly `2^-7`, so `OPT_SAB_DECAY_ADD_FORM`'s
        # `p - lr*wd*p` and the pinned `p * (1 - lr*wd)` have an EXACT
        # product to work with. Contract 12's "must not pass on: `lr*wd` a
        # power of two".
        return OptimizerConfig(
            OPT_ADAMW, lr, b1, b2, eps, f32_from_bits(BITS_WD_POW2), zero,
            zero, False, zero,
        )
    if hp == HP_ADAM_WD:
        # COUPLED decay, contract 7.4's other arm. The decay lands in the
        # GRADIENT and is therefore smoothed by `m` and normalized by `v`.
        return OptimizerConfig(
            OPT_ADAM, lr, b1, b2, eps, f32_from_bits(BITS_WD), zero, zero,
            False, zero,
        )
    if hp == HP_SGD_DAMP:
        return OptimizerConfig(
            OPT_SGD, lr, b1, b2, eps, zero, f32_from_bits(BITS_MOMENTUM),
            f32_from_bits(BITS_DAMPENING), False, zero,
        )
    if hp == HP_SGD_NODAMP:
        return OptimizerConfig(
            OPT_SGD, lr, b1, b2, eps, zero, f32_from_bits(BITS_MOMENTUM),
            zero, False, zero,
        )
    if hp == HP_SGD_NESTEROV:
        return OptimizerConfig(
            OPT_SGD, lr, b1, b2, eps, zero, f32_from_bits(BITS_MOMENTUM),
            zero, True, zero,
        )
    if hp == HP_SGD_MOM0:
        return OptimizerConfig(
            OPT_SGD, lr, b1, b2, eps, zero, zero, zero, False, zero
        )
    raise Error(
        String("optimizer_fixture: hyperparameter preset ")
        + String(hp)
        + " does not exist"
    )


comptime OPT_CASE_COUNT = 27

#: `adam_t1000` is the only expensive case in the table -- a thousand steps
#: of a 1526-element registry -- and it is OFF by default for
#: `[[no-heavy-local-compute]]`'s reason. Contract 5.1 says the gate "must
#: run to at least `t = 8` and should run to `t = 1000`", and `adam_t7` and
#: `adam_t8` already close the `t <= 6` vacuity; what `t = 1000` adds is the
#: DRIFT, which grows with `t`, and the approach to the flush point contract
#: 5.1 predicts at `t = 829` for `beta1`.
comptime OPT_CASE_T1000 = 6


def opt_case(k: Int) raises -> OptCase:
    """Case `k`. Every row names the clause it exists for."""
    if k == 0:
        # The centre of the box: Adam, five tensors, clipping ON, eight
        # steps. Eight because contract 11(d) says "`t = 8` at minimum,
        # because section 5.1's spellings agree through `t = 6`".
        return OptCase("adam_j5_t8", SHAPE_J5_MIXED, HP_ADAM_CLIP, 8,
                       OPT_PLANT_NONE)
    if k == 1:
        # The same, clipping OFF. **Contract 11(c)'s gate must run in this
        # configuration**: with clipping ON one parameter's update depends
        # on every other parameter in the model, by the reference's
        # semantics, so a parameter-count-invariance claim is FALSE there
        # and the gate would be asserting something contract 3.5 denies.
        return OptCase("adam_j5_t8_noclip", SHAPE_J5_MIXED, HP_ADAM_PLAIN,
                       8, OPT_PLANT_NONE)
    if k == 2:
        # ONE step. `OPT_SAB_MOMENT_LERP`'s INERT half (DEVIATION 1474):
        # `m + c1*(g - m)` at `m_prev == +0.0` is `c1*g`, and the pinned
        # `fma(c1, g, ftz(beta1 * 0))` is `c1*g` too. An optimizer gate
        # that only ever ran one step could not see that arm at all, and
        # contract section 12's table does not mention it.
        return OptCase("adam_t1", SHAPE_J3_RAGGED, HP_ADAM_PLAIN, 1,
                       OPT_PLANT_NONE)
    if k == 3:
        # SIX steps. `OPT_SAB_POW_RUNNING`'s INERT half: contract 5.1
        # predicts the binary exponentiation and the running product agree
        # EXACTLY through `t = 6` for both betas.
        return OptCase("adam_t6", SHAPE_J3_RAGGED, HP_ADAM_PLAIN, 6,
                       OPT_PLANT_NONE)
    if k == 4:
        # SEVEN steps. `OPT_SAB_POW_RUNNING`'s WITNESS: contract 5.1's
        # PREDICTED first difference is at `t = 7`. **If the arm is inert
        # here and loud at `adam_t8`, the contract's number is wrong by
        # one and THAT is the finding** -- the check prints the step it
        # first moved at, rather than only whether it moved.
        return OptCase("adam_t7", SHAPE_J3_RAGGED, HP_ADAM_PLAIN, 7,
                       OPT_PLANT_NONE)
    if k == 5:
        # EIGHT steps, contract 11(d)'s minimum for the step-count
        # invariance round trip.
        return OptCase("adam_t8", SHAPE_J3_RAGGED, HP_ADAM_PLAIN, 8,
                       OPT_PLANT_NONE)
    if k == OPT_CASE_T1000:
        # A THOUSAND steps. Contract 5.1's "should run to t = 1000", and
        # the approach to the predicted flush of `beta1^t` at `t = 829`.
        # OFF by default; see `OPT_CASE_T1000`.
        return OptCase("adam_t1000", SHAPE_J3_SMALL, HP_ADAM_PLAIN, 1000,
                       OPT_PLANT_NONE)
    if k == 7:
        # `OPT_SAB_ADAMW_AS_ADAM`'s WITNESS: AdamW with `wd != 0`.
        return OptCase("adamw_wd_t8", SHAPE_J3_RAGGED, HP_ADAMW_WD, 8,
                       OPT_PLANT_NONE)
    if k == 8:
        # Its INERT half, and it is THE REFERENCE'S OWN DEFAULT. Contract
        # 7.4: "at `weight_decay = 0.0` the two algorithms are the SAME
        # ARITHMETIC", so a fixture at torch's default cannot distinguish
        # Adam from AdamW at all.
        return OptCase("adamw_wd0_t8", SHAPE_J3_RAGGED, HP_ADAMW_WD0, 8,
                       OPT_PLANT_NONE)
    if k == 9:
        # `OPT_SAB_DECAY_ADD_FORM`'s WITNESS: `lr * wd` is `2^-10 * 0.1`,
        # NOT a power of two, so `p - lr*wd*p` and `p * (1 - lr*wd)` are
        # different numbers.
        return OptCase("adamw_decay_sep", SHAPE_J3_RAGGED, HP_ADAMW_WD, 8,
                       OPT_PLANT_NONE)
    if k == 10:
        # Its INERT half: `lr * wd` is exactly `2^-7`.
        return OptCase("adamw_decay_pow2", SHAPE_J3_RAGGED,
                       HP_ADAMW_WD_POW2, 8, OPT_PLANT_NONE)
    if k == 11:
        # COUPLED decay, so the `wd != 0` branch of `adam_element_oracle`'s
        # O4a is reached at all. Without it, `fma(wd, p, g)` is never
        # spelled and its negative-zero caveat is never exercised.
        return OptCase("adam_coupled_wd", SHAPE_J3_RAGGED, HP_ADAM_WD, 8,
                       OPT_PLANT_NONE)
    if k == 12:
        # `OPT_SAB_CLIP_PARAM_ORDER`'s WITNESS: `J = 5` with the norms
        # spread over binades. Contract 3.3 asks for `J >= 3` and prefers
        # 5 because `P = 5` is the smallest `P` that carries an odd tail
        # twice.
        return OptCase("clip_j5", SHAPE_J5_MIXED, HP_ADAM_CLIP, 2,
                       OPT_PLANT_NONE)
    if k == 13:
        # Its INERT half. **`J == 2` CANNOT SEE THE ARM AT ALL**: reversing
        # two elements swaps the two children of ONE tree node and
        # `a + b == b + a` bitwise. There is no other way to have an inert
        # half for this arm.
        return OptCase("clip_j2", SHAPE_J2, HP_ADAM_CLIP, 2, OPT_PLANT_NONE)
    if k == 14:
        # `OPT_SAB_CLIP_FLAT_NORM`'s WITNESS: three tensors whose norms
        # differ by several binades.
        return OptCase("clip_spread_j3", SHAPE_J3_SPREAD, HP_ADAM_CLIP, 2,
                       OPT_PLANT_NONE)
    if k == 15:
        # Its INERT half: `J == 1`, where `sqrt(sqrt(s)^2)` and `sqrt(s)`
        # agree for most `s`.
        return OptCase("clip_j1", SHAPE_J1, HP_ADAM_CLIP, 2, OPT_PLANT_NONE)
    if k == 16:
        # `OPT_SAB_CLIP_SERIAL_FOLD`'s WITNESS: one tensor at `N = 300`,
        # `P = 3`, ragged and carrying.
        return OptCase("clip_ragged_j3", SHAPE_J3_RAGGED, HP_ADAM_CLIP, 2,
                       OPT_PLANT_NONE)
    if k == 17:
        # Its INERT half: every `N <= 128`, where `P == 1`, the tree has no
        # arithmetic node, and the v1 answer IS the serial ascending chain.
        return OptCase("clip_small_j3", SHAPE_J3_SMALL, HP_ADAM_CLIP, 2,
                       OPT_PLANT_NONE)
    if k == 18:
        # `OPT_SAB_CLIP_SKIP_AT_ONE`'s WITNESS: a SUBNORMAL gradient cell
        # AND a clip threshold so large that `coef` clamps to exactly
        # `1.0`, which is the only configuration the arm is reachable in.
        # **The verdict must be read at `clip.grad`, not at `param.out`**:
        # contract 3.4c says the difference is CARD VISIBLE and DOWNSTREAM
        # INERT, because the optimizer's own first act is `ftz` on the
        # gradient load.
        return OptCase("clip_subnormal", SHAPE_J3_RAGGED,
                       HP_ADAM_CLIP_HUGE, 1, OPT_PLANT_SUBNORMAL_GRAD)
    if k == 19:
        # Its INERT half: the same configuration with ordinary NORMAL
        # gradients, where `identical_mul(1.0, x)` returns `x` exactly.
        return OptCase("clip_normal", SHAPE_J3_RAGGED, HP_ADAM_CLIP_HUGE,
                       1, OPT_PLANT_NONE)
    if k == 20:
        # `OPT_SAB_FTZ_LATE`'s WITNESS: a gradient near `1e-25`, where the
        # VALUE is a perfectly ordinary normal and `g*g` is `1e-50`, which
        # is not representable as a normal at all. Contract section 6's O7,
        # and the input class a uniform random fixture never reaches.
        return OptCase("adam_tiny_grad", SHAPE_J3_RAGGED, HP_ADAM_PLAIN, 3,
                       OPT_PLANT_TINY_GRAD)
    if k == 21:
        # `OPT_SAB_EPS_INSIDE_SQRT`'s WITNESS: `v` planted in the `1e-20`
        # to `1e-12` band. With `eps = 1e-8`, `eps^2` is `1e-16`, and above
        # that band `sqrt(v_hat) + eps` and `sqrt(v_hat + eps)` agree to
        # the last bit.
        return OptCase("adam_dead_v", SHAPE_J3_RAGGED, HP_ADAM_PLAIN, 2,
                       OPT_PLANT_DEAD_V)
    if k == 22:
        # Its INERT half: ordinary `v` around `1e-4`, which is where a
        # fixture of ordinary gradients puts it.
        return OptCase("adam_ordinary_v", SHAPE_J3_RAGGED, HP_ADAM_PLAIN,
                       2, OPT_PLANT_NONE)
    if k == 23:
        # `OPT_SAB_UNFUSED_UPDATE`'s WITNESS, and it is the most important
        # non-vacuity construction in this lane. See `opt_case_param`.
        return OptCase("adam_fused_sep", SHAPE_J3_SMALL, HP_ADAM_PLAIN, 1,
                       OPT_PLANT_FUSED_SEPARATOR)
    if k == 24:
        # Its INERT half: the same construction forty binades apart.
        return OptCase("adam_fused_inert", SHAPE_J3_SMALL, HP_ADAM_PLAIN,
                       1, OPT_PLANT_FUSED_INERT)
    if k == 25:
        # `OPT_SAB_SQ_ASSOC`'s INERT half: every gradient EXACTLY `1.0`, so
        # `c2 * (g*g)` and `(c2*g) * g` are both exactly `c2`. The witness
        # is any hashed case.
        return OptCase("adam_unit_grad", SHAPE_J3_SMALL, HP_ADAM_PLAIN, 2,
                       OPT_PLANT_UNIT_GRAD)
    if k == 26:
        # `OPT_SAB_RSQRT`'s witness: 4096 hashed lanes. DEVIATION 741
        # measured `identical_rsqrt` off the correctly rounded rsqrt on
        # 134,858 of 520,133 positive normals, so about three quarters of
        # inputs AGREE and **a handful of round `v` values will pass**.
        return OptCase("adam_hashed_4096", SHAPE_J1_BIG, HP_ADAM_PLAIN, 2,
                       OPT_PLANT_NONE)
    raise Error(
        String("optimizer_fixture: case ")
        + String(k)
        + " does not exist; there are "
        + String(OPT_CASE_COUNT)
    )


def opt_sgd_case(k: Int) raises -> OptCase:
    """The SGD cases, kept in their own table because SGD reads `m_state`
    as a MOMENTUM BUFFER and never touches `v_state`, and mixing the two
    families in one index space is how a gate comes to run an Adam
    expectation against an SGD verdict.

    `optimizer_step_oracle` carries the momentum buffer in `m_state`
    precisely so a checkpoint has ONE SHAPE for both algorithms, and that
    sharing is what makes the separate table necessary here rather than
    optional."""
    if k == 0:
        # `OPT_SAB_MOMENTUM_FIRST_STEP`'s WITNESS: `dampening != 0` AND
        # `t = 1`, the step that CREATES the buffer. Contract 7.3a: the
        # first momentum step is a COPY, `b_1 = g`, not `c_damp * g`.
        return OptCase("sgd_damp_t1", SHAPE_J3_RAGGED, HP_SGD_DAMP, 1,
                       OPT_PLANT_NONE)
    if k == 1:
        # The same at `t = 3`, to show the divergence PERSISTS rather than
        # washing out -- contract 7.3a asks for exactly that and a gate
        # that only compared `t = 1` could not tell a persistent divergence
        # from a transient one.
        return OptCase("sgd_damp_t3", SHAPE_J3_RAGGED, HP_SGD_DAMP, 3,
                       OPT_PLANT_NONE)
    if k == 2:
        # The INERT half, and it is **the reference's own default**: at
        # `dampening == 0`, `c_damp` is exactly `1.0` and
        # `identical_mul(1.0, g)` returns `g` for every finite `g`.
        return OptCase("sgd_nodamp_t1", SHAPE_J3_RAGGED, HP_SGD_NODAMP, 1,
                       OPT_PLANT_NONE)
    if k == 3:
        # `OPT_SAB_NESTEROV_ORDER`'s WITNESS: Nesterov at `t = 2`, where
        # the buffer is no longer a copy of `g` so `g + momentum*b` and
        # `b + momentum*g` are different expressions.
        return OptCase("sgd_nesterov_t2", SHAPE_J3_RAGGED, HP_SGD_NESTEROV,
                       2, OPT_PLANT_NONE)
    if k == 4:
        # Its INERT half, and it is a SUBTLE one: at `t = 1` the buffer is
        # a COPY of `g`, so `b == g` and the two operand orders are the
        # SAME EXPRESSION. Contract 7.3c names this in one clause -- "note
        # it is ALSO inert at `t = 1`" -- and it is exactly the case a
        # fixture built for the arm would choose by accident.
        return OptCase("sgd_nesterov_t1", SHAPE_J3_RAGGED, HP_SGD_NESTEROV,
                       1, OPT_PLANT_NONE)
    if k == 5:
        # `momentum == 0`. The INERT half of BOTH momentum arms at once,
        # and the shape in which S4 is not spelled at all.
        return OptCase("sgd_mom0_t3", SHAPE_J3_RAGGED, HP_SGD_MOM0, 3,
                       OPT_PLANT_NONE)
    if k == 6:
        # EIGHT steps of SGD with momentum AND dampening. **This case
        # exists for contract 11(d) and for nothing else**: the step-count
        # invariance round trip runs `2t` steps against a checkpoint at
        # `t`, and contract 11(d) fixes `t = 8` at minimum because contract
        # 5.1's two `beta^t` spellings agree through `t = 6`. Every other
        # SGD case here is 1 or 3 steps, chosen to sit exactly on the step
        # where a momentum arm is or is not inert, and NONE of them is long
        # enough for the round trip.
        #
        # It is also `GATE_RESUME_REINIT`'s witness, and it has to be the
        # SAME case: that arm IS clause (d)'s control 2 -- a resume that
        # drops the momentum-initialized flag -- so it can only be
        # evaluated on a case clause (d) can run at all.
        return OptCase("sgd_damp_t8", SHAPE_J3_RAGGED, HP_SGD_DAMP, 8,
                       OPT_PLANT_NONE)
    raise Error(
        String("optimizer_fixture: SGD case ")
        + String(k)
        + " does not exist; there are "
        + String(OPT_SGD_CASE_COUNT)
    )


comptime OPT_SGD_CASE_COUNT = 7


def opt_case_by_name(name: String) raises -> Int:
    for k in range(OPT_CASE_COUNT):
        if String(opt_case(k).name) == name:
            return k
    raise Error(
        String("optimizer_fixture: no Adam case named '")
        + name
        + "'. A name that does not exist RAISES rather than falling back to"
        + " a default: a gate that quietly ran a different case from the"
        + " one the operator asked for would report a green for a shape"
        + " nobody chose."
    )


def opt_sgd_case_by_name(name: String) raises -> Int:
    for k in range(OPT_SGD_CASE_COUNT):
        if String(opt_sgd_case(k).name) == name:
            return k
    raise Error(
        String("optimizer_fixture: no SGD case named '") + name + "'"
    )


# ===========================================================================
# BUILDING A CASE'S BUFFERS
# ===========================================================================


def opt_total(shape: Int) raises -> Int:
    var off = opt_offsets(shape)
    return off[len(off) - 1]


def opt_case_grad(c: OptCase, step: Int) raises -> List[Float32]:
    """The gradient at step `step` (one-based), `[sum N_j]` flat.

    **THE GRADIENT CHANGES EVERY STEP AND THAT IS NOT DECORATION.** A run
    that fed the SAME gradient at every step would make `m` and `v`
    converge to fixed points, and at a fixed point several of the running
    recurrences become bit-inert in ways that have nothing to do with the
    profile -- `m` stops moving, so `OPT_SAB_MOMENT_LERP` goes quiet again
    for the same reason it is quiet at `m == 0`. The step index is hashed
    in.

    Each tensor `j` is scaled by `opt_binade_shift(shape, j)` binades, which
    is EXACT and is `OPT_SAB_CLIP_FLAT_NORM`'s separating property."""
    var sizes = opt_sizes(c.shape)
    var seed = opt_case_seed(opt_case_index_of(c)) ^ (UInt64(TID_GRAD) << 32)
    seed = seed + UInt64(0x100000) * UInt64(step)
    var out = List[Float32]()
    for j in range(len(sizes)):
        var shift = opt_binade_shift(c.shape, j)
        for i in range(sizes[j]):
            out.append(opt_hashed(seed + UInt64(j * 1000003), i, shift))
    if c.plant == OPT_PLANT_UNIT_GRAD:
        for i in range(len(out)):
            out[i] = Float32(1.0)
        return out^
    if c.plant == OPT_PLANT_TINY_GRAD:
        # Every fourth cell near `1e-25`. NOT every cell: the arm must be
        # localizable, and a run where every gradient is tiny would make
        # `v` collapse everywhere and would hide which cells the arm
        # actually moved.
        var t = f32_from_bits(BITS_TINY_GRAD)
        var i = 0
        while i < len(out):
            out[i] = t
            i += 4
        return out^
    if c.plant == OPT_PLANT_SUBNORMAL_GRAD:
        var pats: List[UInt32] = [
            BITS_SUBNORMAL_GRAD, BITS_NEG_SUBNORMAL_GRAD
        ]
        var t = 0
        var i = 0
        while i < len(out):
            out[i] = f32_from_bits(pats[t % 2])
            t += 1
            i += 4
        return out^
    return out^


def opt_case_param(c: OptCase) raises -> List[Float32]:
    """The INITIAL parameters, `[sum N_j]` flat.

    **THE TWO FUSED PLANTS ARE THE MOST IMPORTANT CONSTRUCTION IN THIS
    FILE, AND DEVIATION 1472 IS WHY.** Contract 7.2d's non-vacuity note is
    the sharpest sentence in the whole contract:

    > `check-ieee-arith` scored Metal as UNFUSED over 2^20 HASHED patterns
    > and the verdict was WRONG -- ZERO of those 2^20 patterns separate a
    > fused `a*b + c` from an unfused one, because random exponents put the
    > product and the addend so far apart that both spellings round
    > identically. The fixture must be BUILT TO SEPARATE.

    So `OPT_PLANT_FUSED_SEPARATOR` does not draw `p`. It COMPUTES it.

      1. `q` at step 1 from fresh state depends only on `g`, `m_prev`,
         `v_prev` and the host scalars -- NOT on `p`, provided
         `weight_decay == 0`, which the preset `HP_ADAM_PLAIN` guarantees
         and `opt_case_param` REFUSES to proceed without.
      2. So `q` can be computed here by calling `adam_element_oracle` with
         `p_in = +0.0`, and then `p` is set to `ftz(identical_mul(
         step_size, q))` -- **the ROUNDED product**.
      3. At the real step, the UNFUSED spelling computes
         `p - round(step_size * q)`, which is `x - x` and therefore
         EXACTLY `+0.0`. The FUSED spelling computes
         `fma(-step_size, q, p)`, which is `-(step_size*q -
         round(step_size*q))`, the rounding error, and is NONZERO whenever
         the product is inexact.

    A difference between `+0.0` and a nonzero residual is not a last-bit
    difference; it is the whole value. **That is what "built to separate"
    means and it is why this arm is a real gate rather than a coin flip.**

    `OPT_PLANT_FUSED_INERT` takes the same `p` and scales it by `2^40`,
    which is EXACT. The product's rounding error is then about `2^-64`
    relative to `p`, far below `p`'s own ulp of `2^-24`, so both spellings
    round to the same number and the arm is provably inert. **The pair is
    what makes the arm a reach proof; the witness alone would be a smoke
    test.**

    A caveat stated rather than buried: this construction CALLS THE ORACLE
    to build a fixture the oracle is then compared against. That is
    tautological for clause (a) -- which is why the fused cases are not
    where clause (a) earns its keep -- and it is NOT tautological for the
    sabotage arm, because `optimizer_oracle.mojo` carries no sabotage
    switch of any kind (every `is_defined` in this lane is in
    `optimizer.mojo`), so the fixture is bit-identical in a clean build and
    in an armed one."""
    var sizes = opt_sizes(c.shape)
    var seed = opt_case_seed(opt_case_index_of(c)) ^ (UInt64(TID_PARAM) << 32)
    var out = List[Float32]()
    for j in range(len(sizes)):
        var shift = opt_binade_shift(c.shape, j)
        for i in range(sizes[j]):
            out.append(opt_hashed(seed + UInt64(j * 7919), i, shift))

    if c.plant != OPT_PLANT_FUSED_SEPARATOR and (
        c.plant != OPT_PLANT_FUSED_INERT
    ):
        return out^

    var cfg = opt_config(c.hp)
    if cfg.weight_decay != Float32(0.0):
        raise Error(
            String("optimizer_fixture: case ")
            + String(c.name)
            + " uses a FUSED plant with weight_decay != 0. The"
            + " construction depends on `q` being independent of `p`, and"
            + " with a coupled decay `g = fma(wd, p, g)` makes `q` a"
            + " function of `p`, so the fixed point below does not exist"
            + " and the plant would land on the wrong number"
            + " ([[reached-but-inert]])."
        )
    if c.steps != 1:
        raise Error(
            String("optimizer_fixture: case ")
            + String(c.name)
            + " uses a FUSED plant with steps="
            + String(c.steps)
            + ". The construction is exact only at the FIRST step from"
            + " fresh state, where `m_prev` and `v_prev` are both `+0.0`."
        )
    var sc = step_scalars(cfg, 1)
    var g = opt_case_grad(c, 1)
    var zero = Float32(0.0)
    for i in range(len(out)):
        var e = adam_element_oracle(zero, g[i], zero, zero, cfg, sc)
        var p_want = ftz(identical_mul(sc.step_size, e.q))
        if c.plant == OPT_PLANT_FUSED_INERT:
            # 2^40, EXACT.
            for _s in range(40):
                p_want = p_want * Float32(2.0)
        out[i] = p_want
    return out^


def opt_case_m(c: OptCase) raises -> List[Float32]:
    """The initial first moment. `+0.0` everywhere -- FRESH STATE, which is
    what `t = 1` means.

    It is `+0.0` and not `-0.0` deliberately: `ftz(identical_mul(beta1,
    -0.0))` is `-0.0` and `fma(c1, g, -0.0)` is not bitwise
    `fma(c1, g, +0.0)` when `c1*g` is `+0.0`, so a `-0.0` seed would make
    the first step of a zero-gradient element depend on a sign nobody
    chose."""
    var out = List[Float32]()
    for _i in range(opt_total(c.shape)):
        out.append(Float32(0.0))
    return out^


def opt_case_v(c: OptCase) raises -> List[Float32]:
    """The initial second moment.

    `OPT_PLANT_DEAD_V` plants the `1e-20` to `1e-12` band contract 4d
    names. **This is the one plant that has to go into STATE rather than
    into an input**, because `v` is a running average and a fixture cannot
    reach that band from ordinary gradients in two steps -- it takes a dead
    unit and thousands of them."""
    var out = List[Float32]()
    var n = opt_total(c.shape)
    if c.plant != OPT_PLANT_DEAD_V:
        for _i in range(n):
            out.append(Float32(0.0))
        return out^
    var dead = f32_from_bits(BITS_DEAD_V)
    var seed = opt_case_seed(opt_case_index_of(c)) ^ (UInt64(TID_V) << 32)
    for i in range(n):
        # A spread WITHIN the band rather than one repeated value, so the
        # arm's verdict is a cell count and not a single coincidence. The
        # spread is by exact halvings, so every value stays in the band and
        # stays exactly representable.
        var h = opt_splitmix64(seed + UInt64(i))
        var steps = Int(h % UInt64(8))
        var x = dead
        for _s in range(steps):
            x = x / Float32(2.0)
        out.append(x)
    return out^


def opt_case_index_of(c: OptCase) raises -> Int:
    """The case's index, from its name, so that a case's SEED is a function
    of its index and not of its shape -- two cases with the same shape and
    different plants must not draw the same hash, or the pair stops being a
    controlled comparison for the wrong reason.

    It walks the ADAM table and then the SGD table. A name in neither
    raises, which is what stops a synthetic case from silently borrowing a
    real one's seed."""
    var nm = String(c.name)
    for k in range(OPT_CASE_COUNT):
        if String(opt_case(k).name) == nm:
            return k
    for k in range(OPT_SGD_CASE_COUNT):
        if String(opt_sgd_case(k).name) == nm:
            return 1000 + k
    raise Error(
        String("optimizer_fixture: case '")
        + nm
        + "' is in neither table, so it has no seed"
    )


def opt_buf_initialized(c: OptCase) raises -> List[Bool]:
    """SGD's per-tensor "has the momentum buffer been created" flag, all
    False at the start of a run.

    **IT IS CHECKPOINT STATE AND IT IS THE HALF A CHECKPOINT FORMAT
    FORGETS**, because the flag is a `Bool` and everything around it is a
    tensor. Contract 7.3b: a resume that reinitializes it recomputes `b` as
    a COPY of the current gradient instead of continuing the recurrence,
    and the two runs diverge at that step and NEVER RECONVERGE. That is
    `OPT_SAB_RESUME_REINIT`, and contract section 12 lists it as a switch
    while `optimizer.mojo` implements no such switch -- so
    `optimizer_check.mojo` models it in the GATE by dropping the flag
    across its own checkpoint round trip (DEVIATION 1473)."""
    var out = List[Bool]()
    var off = opt_offsets(c.shape)
    for _j in range(len(off) - 1):
        out.append(False)
    return out^


def opt_poison(n: Int) -> List[Float32]:
    var out = List[Float32]()
    var p = f32_from_bits(BITS_POISON)
    for _i in range(n):
        out.append(p)
    return out^


def count_poison(values: List[Float32]) -> Int:
    var n = 0
    for i in range(len(values)):
        if bits_of(values[i]) == BITS_POISON:
            n += 1
    return n


# ===========================================================================
# THE FIXTURE'S OWN ASSERTIONS
# ===========================================================================


def assert_no_zero_or_subnormal(c: OptCase, step: Int) raises -> Int:
    """No `+-0.0` and no subnormal in this case's gradients or parameters.
    Returns the cell count checked.

    **THIS IS WHAT MAKES THE INERT HALVES MEASUREMENTS.**
    `OPT_SAB_CLIP_SKIP_AT_ONE` is inert exactly on NORMAL gradients, and
    `OPT_SAB_FTZ_LATE` is inert exactly when no intermediate lands
    subnormal. Both claims are about every case EXCEPT the two that plant
    the values, and a stray zero or subnormal from the hash would make one
    of those claims false on one case -- which is the hardest kind of flake
    to read, because the arm simply looks inert.

    The exemptions are by PLANT KIND rather than by case name, so a new case
    cannot inherit one by accident."""
    if c.plant == OPT_PLANT_SUBNORMAL_GRAD or c.plant == OPT_PLANT_TINY_GRAD:
        return 0
    if c.plant == OPT_PLANT_UNIT_GRAD:
        return 0
    var g = opt_case_grad(c, step)
    var p = opt_case_param(c)
    var checked = 0
    for side in range(2):
        var xs = g.copy()
        var who = String("grad")
        if side == 1:
            xs = p.copy()
            who = String("param")
        for i in range(len(xs)):
            if is_zero_bits(xs[i]):
                raise Error(
                    String("optimizer_fixture: case ")
                    + String(c.name)
                    + " "
                    + who
                    + "["
                    + String(i)
                    + "] is a signed ZERO. `opt_hashed`'s magnitude band"
                    + " is [0.125, 32) times a binade shift and cannot"
                    + " produce one."
                )
            if is_subnormal_bits(xs[i]):
                raise Error(
                    String("optimizer_fixture: case ")
                    + String(c.name)
                    + " "
                    + who
                    + "["
                    + String(i)
                    + "] is SUBNORMAL ("
                    + bits32_hex(xs[i])
                    + "). Every 'inert on normal values' assertion in this"
                    + " lane rests on this not happening."
                )
            if is_nonfinite_bits(xs[i]):
                raise Error(
                    String("optimizer_fixture: case ")
                    + String(c.name)
                    + " "
                    + who
                    + "["
                    + String(i)
                    + "] is NON-FINITE. Contract 8a refuses those as"
                    + " INPUTS, so a fixture that produced one would make"
                    + " every case refuse."
                )
            checked += 1
    return checked


def assert_binade_spread(c: OptCase) raises -> Int:
    """The per-tensor gradient norms of this case differ by at least four
    binades between the smallest and the largest. Returns the binade span.

    **`OPT_SAB_CLIP_FLAT_NORM`'S SEPARATING PROPERTY, MEASURED.** Contract
    3.1: "the fixture must carry at least three tensors of different lengths
    and of norms that differ by several binades", because at `J = 1` --
    and, in practice, at any `J` whose norms are all within a binade -- the
    two-level and flat forms agree.

    The norm is computed here with a PLAIN sum of squares and a plain
    `sqrt`, deliberately NOT through `clip_tensor_sumsq_oracle`: this is a
    statement about the fixture's MAGNITUDES, not about the fold, and
    routing it through the code under test would make the assertion depend
    on the thing it is characterizing."""
    var sizes = opt_sizes(c.shape)
    if len(sizes) < 2:
        return 0
    var g = opt_case_grad(c, 1)
    var off = opt_offsets(c.shape)
    var lo_exp = 1000
    var hi_exp = -1000
    for j in range(len(sizes)):
        var acc = Float64(0.0)
        for i in range(off[j], off[j + 1]):
            acc += Float64(g[i]) * Float64(g[i])
        if acc <= Float64(0.0):
            raise Error(
                String("optimizer_fixture: case ")
                + String(c.name)
                + " tensor "
                + String(j)
                + " has a ZERO sum of squares, so its norm is +0.0 and the"
                + " clip's outer fold has nothing to be wrong about there"
            )
        # The binade of the norm, by repeated halving on the SQUARE. No
        # `log2` and no `pow`, so this reads the exponent without a
        # transcendental.
        var e = 0
        var x = acc
        while x >= Float64(4.0):
            x = x / Float64(4.0)
            e += 1
        while x < Float64(1.0):
            x = x * Float64(4.0)
            e -= 1
        if e < lo_exp:
            lo_exp = e
        if e > hi_exp:
            hi_exp = e
    var span = hi_exp - lo_exp
    if span < 4:
        raise Error(
            String("optimizer_fixture: case ")
            + String(c.name)
            + " has per-tensor gradient norms spanning only "
            + String(span)
            + " binades. Contract 3.1 requires SEVERAL, because at a"
            + " narrow spread the two-level clip norm and a flat one agree"
            + " and OPT_SAB_CLIP_FLAT_NORM is inert for a reason the case"
            + " does not claim ([[reached-but-inert]])."
        )
    return span


def describe_case(c: OptCase) raises -> String:
    var off = opt_offsets(c.shape)
    var cfg = opt_config(c.hp)
    var kind = String("SGD")
    if cfg.kind == OPT_ADAM:
        kind = String("ADAM")
    elif cfg.kind == OPT_ADAMW:
        kind = String("ADAMW")
    var out = (
        String(c.name)
        + "  "
        + kind
        + " J="
        + String(len(off) - 1)
        + " N="
        + String(off[len(off) - 1])
        + " steps="
        + String(c.steps)
        + " plant="
        + plant_name(c.plant)
        + " lr="
        + bits32_hex(cfg.lr)
        + " b1="
        + bits32_hex(cfg.beta1)
        + " b2="
        + bits32_hex(cfg.beta2)
        + " eps="
        + bits32_hex(cfg.eps)
        + " wd="
        + bits32_hex(cfg.weight_decay)
    )
    if cfg.kind == OPT_SGD:
        out += (
            " mom="
            + bits32_hex(cfg.momentum)
            + " damp="
            + bits32_hex(cfg.dampening)
        )
        if cfg.nesterov:
            out += " nesterov"
    if cfg.max_norm > Float32(0.0):
        out += " clip=" + bits32_hex(cfg.max_norm)
    else:
        out += " clip=OFF"
    return out^
