# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`next_float` and `box_muller_transform`: the two RAFT functions this lane
needs from `rng_device.cuh` and that `core/philox.mojo` does not carry.

PORT OF `raft/cpp/include/raft/random/detail/rng_device.cuh` at RAFT
`ebf9268` (`upstream/raft-v26.08.00`), lines 481-487 and 133-146.

WHY THIS FILE EXISTS AND IS NOT IN `core/philox.mojo`. That file ports
RAFT's `PhiloxGenerator` and its `uniformInt` because those are what cuML's
Random Forest bootstrap reaches, and it carries `next_u64`, `next_double` and
`custom_next` for `UniformDistParams<double>`. It does NOT carry the FLOAT32
`next_float` (`:481-487`) or the normal transform (`:133-146`), because
nothing in this repository needed a float32 uniform or a Gaussian until now.
`core/` is not this lane's directory to edit, so the two missing functions are
ported HERE, beside their caller, and `kernel_methods/DERIVATION_MAP.tsv` records
this file as a THIRD PARTIAL MIRROR of one upstream header alongside
`core/philox.mojo` and `resample/original/index_map.mojo`.

**RECORDED FOR THE ORCHESTRATOR: THE RIGHT LONG-TERM HOME FOR BOTH IS
`core/philox.mojo`, and moving them there is a one-file change this lane must
not make.** `resample/original/index_map.mojo::draw_unit_float` already
inlines `next_float`'s two lines for its own use, so the move would delete a
duplication as well as this one. `kernel_methods/README.md`'s WHAT THE
ORCHESTRATOR MUST WIRE carries the item.

A LEAF MODULE ON PURPOSE. It imports `core/philox.mojo` and
`original/numerics.mojo` and NOTHING ELSE in this lane, so that both
`kernel_methods/original/random_features.mojo` (production) and
`kernel_methods/original/km_sabotage.mojo` (the arms) can import it without
a cycle. `random_features.mojo` imports the sabotage file, the sabotage file
imports this one, and this one imports neither.
"""

from std.memory import bitcast

from core.philox import PhiloxState
from original.numerics import (
    ftz,
    identical_cos,
    identical_log,
    identical_mul,
    identical_mul_add,
    identical_sin,
    identical_sqrt,
)


#: `2 pi` in float32. RAFT spells it `Type(2.0) * Type(3.141592653589793)`
#: (`rng_device.cuh:135`) and scikit-learn spells it `2 * np.pi` narrowed to
#: the input dtype (`kernel_approximation.py:385`); both land on this bit
#: pattern, `0x40C90FDB` = 6.2831854820251465. Written as BITS because
#: `String(Float32)` does not round trip (`[[mojo-string-float-roundtrip]]`).
comptime KM_TWO_PI_BITS: UInt32 = 0x40C90FDB

#: `2^-24`, the smallest positive value `next_float` can return: `val` is at
#: most `2^24 - 1` and the divisor is `2^24`. DEVIATION 1676 substitutes it
#: for a drawn `+0.0`.
comptime KM_MIN_UNIT_BITS: UInt32 = 0x33800000


def km_two_pi() -> Float32:
    return bitcast[DType.float32](KM_TWO_PI_BITS)


def km_min_unit() -> Float32:
    return bitcast[DType.float32](KM_MIN_UNIT_BITS)


def km_unit_float_from(mut gen: PhiloxState) -> Float32:
    """PORT OF `PhiloxGenerator::next_float`, `rng_device.cuh:481-487`:

        uint32_t val = next_u32() >> 8;
        ret = static_cast<float>(val) / float(uint32_t(1) << 24);

    EXACT ON EVERY VENDOR, and `resample/original/index_map.mojo`'s
    DEVIATION 1693 writes the proof out: `val < 2^24` so `Float32(val)` is
    exact, `2^24` is exact, and the quotient is a 24-bit significand times a
    power of two, so it rounds nowhere. No `identical_div` and no `ftz`, and
    the smallest nonzero result is `2^-24`, nowhere near subnormal.

    TAKES AN EXISTING GENERATOR. `index_map.mojo::draw_unit_float` is this
    function with a fresh per-position generator wrapped around it, which is
    what a single position-mapped uniform wants; the Box-Muller transform
    needs TWO words from ONE generator, which is what this signature is for.
    The two lines are the same two lines in both places and it is the reason
    the header above wants both in `core/philox.mojo`.
    """
    var val = gen.next_u32() >> UInt32(8)
    return Float32(Int(val)) / Float32(Int(UInt32(1) << 24))


def km_guard_unit(u: Float32) -> Float32:
    """DEVIATION 1676's guard, and it is OURS, not RAFT's.

    RAFT's transform opens with `raft::sqrt(minus2 * raft::log(val1))` and
    `next_float`'s range is `[0, 1)` -- CLOSED AT ZERO -- so at `val1 == +0.0`
    the log is `-inf`, `R` is `+inf`, and both returned normals are non-finite.
    RAFT never guards it. The probability is `2^-24` per pair, so a
    `128 x 1024` weight matrix draws 65,536 pairs and one fit in 256 contains
    a non-finite weight whose feature map is NaN for every row.

    A wrong answer with no error is the class `PORTING_RULES 0c` and the
    standing rule `assume-our-code-is-broken` say to FIX rather than port.
    The fix is the smallest one available: `+0.0` becomes `2^-24`, which is
    EXACTLY the smallest positive value the generator can produce, so the
    substituted draw is a value the generator itself produces and the
    distribution moves one point of a `2^24`-point grid onto its neighbour.
    Every other input is untouched on every vendor, bit for bit.

    The test is `u > 0.0` and not `u != 0.0`, so a `-0.0` -- which
    `next_float` cannot produce, but which a future caller or a sabotage
    could hand in -- is also replaced rather than sent into `log` to become
    `-inf` on one column and `NaN` on another.
    """
    if u > Float32(0.0):
        return u
    return km_min_unit()


def km_boxmuller_pair(
    u1: Float32, u2: Float32, sigma: Float32, mu: Float32
) -> Tuple[Float32, Float32]:
    """PORT OF `raft::random::detail::box_muller_transform`
    (`rng_device.cuh:133-142`), reached through the two-sigma overload at
    `:145-146` with `sigma2 = sigma1` and `mu2 = mu1`:

        constexpr Type twoPi  = Type(2.0) * Type(3.141592653589793);
        constexpr Type minus2 = -Type(2.0);
        Type R     = raft::sqrt(minus2 * raft::log(val1));
        Type theta = twoPi * val2;
        Type s, c;
        raft::sincos(theta, &s, &c);
        val1 = R * c * sigma1 + mu1;
        val2 = R * s * sigma2 + mu2;

    THEIR ORDER, unchanged, with three pins.

    **PIN 1, the transcendentals.** `identical_log`, `identical_sqrt`,
    `identical_cos` and `identical_sin` (IDENTITY_PATHS rows 10 and 12).
    `raft::sincos` computes both from ONE argument reduction; ours both reach
    `original/numerics.mojo::_cephes_sincosf_core` with the same Cody-Waite
    reduction and the same octant computation, so the pair is the pair a
    fused `sincos` would return rather than two independently reduced values.
    `theta` lands in `[0, 2 pi)`, which is the range `portable_cosf`'s
    docstring records as MEASURED at `<= 2 ulp`.

    **PIN 2, the `sigma`/`mu` seam.** `R * c * sigma + mu` is
    `((R*c)*sigma) + mu` in C, and a codegen may contract the last multiply
    into the add or not -- two spellings of one line, chosen per backend
    (IDENTITY_PATHS row 9). The rule here is the one
    `decomposition/original/jacobi_eigh_device.mojo::_rot_sub` states for
    its own seam: **name which product fuses.** `R*c` is rounded and flushed,
    then `sigma` is fused into the `mu` add through `identical_mul_add`. One
    rounding where theirs may be one or two, and the same one on Metal, PTX
    and AMDGPU.

    **PIN 3, `mu` IS AN ADD AND ITS SIGN-OF-ZERO CONSEQUENCE IS RECORDED.**
    Callers pass `mu = +0.0`, and `x + (+0.0)` is `x` for every finite `x`
    EXCEPT `x = -0.0`, which becomes `+0.0` (IDENTITY_PATHS row 39). So a
    weight the multiply produced as a negative zero is stored as a positive
    zero. That happens deterministically, on every vendor, on exactly the
    inputs where `R*c*sigma` is `-0.0`, and it is THEIR line: they add `mu1`
    unconditionally too. Recorded rather than optimized away -- dropping the
    add to "preserve" the sign would be a departure from their code that
    changes a bit.

    THE GUARD IS NOT APPLIED HERE. `km_guard_unit` is the caller's, so that
    `KMSAB_NO_BOXMULLER_GUARD` can drop it without touching this
    transcription and so that this function stays exactly their five lines.
    """
    var r = ftz(
        identical_sqrt(ftz(identical_mul(Float32(-2.0), identical_log(u1))))
    )
    var theta = ftz(identical_mul(km_two_pi(), u2))
    var c = identical_cos(theta)
    var s = identical_sin(theta)
    var v1 = ftz(identical_mul_add(ftz(identical_mul(r, c)), sigma, mu))
    var v2 = ftz(identical_mul_add(ftz(identical_mul(r, s)), sigma, mu))
    return (v1, v2)
