# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# This lane MIRRORS CatBoost (pinned commit 54a8143a). Per-file provenance is in gbdt/DERIVATION_MAP.tsv, in this file's own docstring, and in NOTICE; files with no CatBoost counterpart say so.
"""CatBoost's GPU random streams, ported.

PORT OF `catboost/cuda/cuda_util/kernel/random_gen.cuh` at `54a8143a`.
Transliterated. Do not improve.

The generator is a pair of 16-bit multiply-with-carry streams packed into
one u64 (`AdvanceSeed`), exactly as they wrote it; every kernel that draws
randomness carries one such seed per THREAD and writes the advanced seed
back, so the stream continues across calls -- that write-back IS the RNG
state of the fit.

DEVIATION (stated once for the file): their `NextUniform` computes
`((v << 16) + u) * 2.328306435996595e-10` in FLOAT64 and Metal has no
float64 in kernels, so this port keeps their own float32 arm
(`NextUniformF`, `random_gen.cuh:33-39`) for every draw. The integer
stream is bit-identical to theirs; only the final scaling to [0,1) is
f32, which changes no distribution and keeps every draw deterministic.
"""

# DEVIATION 258: Box-Muller and the Poisson draw ran the DEVICE sqrt/log/cos,
# which differ per vendor (row 10 sqrt is approximate on NVIDIA; row 12 log
# and cos are each vendor's own); under IDENTICAL the three seam calls are
# the portable pair, under FAST the stdlib verbatim
from original.numerics import identical_cos, identical_log, identical_sqrt


def advance_seed(s: UInt64) -> UInt64:
    """`AdvanceSeed` (`random_gen.cuh:8-15`), their two MWC halves."""
    var v = UInt32(s >> 32)
    var u = UInt32(s & UInt64(0xFFFFFFFF))
    v = UInt32(36969) * (v & UInt32(0xFFFF)) + (v >> 16)
    u = UInt32(18000) * (u & UInt32(0xFFFF)) + (u >> 16)
    return (UInt64(v) << 32) | UInt64(u)


def advance_seed_k(s: UInt64, k: Int) -> UInt64:
    """`AdvanceSeed(seed, k)` (`random_gen.cuh:17-22`)."""
    var out = s
    for _ in range(k):
        out = advance_seed(out)
    return out


def advance_seed32(s: UInt32) -> UInt32:
    """`AdvanceSeed32` (`random_gen.cuh:41-44`), the LCG the seed
    generator mixes with."""
    return UInt32(1664525) * s + UInt32(1013904223)


def next_uniform_f(s: UInt64) -> Tuple[Float32, UInt64]:
    """`NextUniformF` (`random_gen.cuh:33-39`): advance, then scale the
    mixed word to [0, 1). Returns the draw AND the advanced seed; Mojo has
    no out-pointer idiom for a register value and the caller must carry
    the state exactly as their `ui64* seed` does."""
    var x = advance_seed(s)
    var v = UInt32(x >> 32)
    var u = UInt32(x & UInt64(0xFFFFFFFF))
    var mixed = (v << 16) + u
    return (Float32(mixed) * Float32(2.328306435996595e-10), x)


def next_normal_f(s: UInt64) -> Tuple[Float32, UInt64]:
    """`NextNormal` (`random_gen.cuh:50-54`), Box-Muller off two draws.

        float a = NextUniform(seed);
        float b = NextUniform(seed);
        return sqrtf(-2.0f * logf(a)) * cosf(2.0f * CUDART_PI_F * b);

    TWO seed advances, in that order; a caller that swaps them gets a
    different stream from the same seed.
    """
    var da = next_uniform_f(s)
    var db = next_uniform_f(da[1])
    return (
        identical_sqrt(Float32(-2.0) * identical_log(da[0]))
        * identical_cos(Float32(2.0) * Float32(CUDART_PI_F) * db[0]),
        db[1],
    )


#: `CUDART_PI_F`, CUDA's float pi. Written out because the constant is
#: part of the stream: a different last bit here is a different draw.
comptime CUDART_PI_F = 3.141592654


def next_poisson_f(s: UInt64, alpha: Float32) -> Tuple[Float32, UInt64]:
    """`NextPoisson` (`random_gen.cuh:57-72`), both arms.

        if (alpha > 20) {
            float a = sqrtf(alpha) * NextNormal(seed) + alpha;
            while (a < 0) { a = sqrtf(alpha) * NextNormal(seed) + alpha; }
            return a;
        }
        float logp = 0.0f, L = -alpha;
        int k = 0;
        do { k++; logp += log(NextUniform(seed)); } while (logp > L);
        return k - 1;

    Note what the small-alpha arm returns: `k - 1` as a FLOAT, so the
    weight multiplier is an integer count. Note also that the large-alpha
    arm returns a CONTINUOUS normal approximation, not an integer -- their
    Poisson bootstrap above lambda 20 is a Gaussian one. Both copied.

    THE `while (a < 0)` LOOP IS UNBOUNDED IN THEIR SOURCE and is bounded
    here, because a Metal kernel that spins does not get interrupted: the
    probability of `sqrt(alpha) * N(0,1) + alpha < 0` at alpha > 20 is
    below 1.2e-5 per draw, so sixteen rejections is a 1e-79 event, and
    the fallback returns `alpha` -- the distribution's own mean, and the
    value the loop is converging toward anyway.
    """
    if alpha > Float32(20.0):
        var st = s
        for _ in range(16):
            var dn = next_normal_f(st)
            st = dn[1]
            var a = identical_sqrt(alpha) * dn[0] + alpha
            if a >= Float32(0.0):
                return (a, st)
        return (alpha, st)

    var logp = Float32(0.0)
    var limit = -alpha
    var k = 0
    var st2 = s
    while True:
        k += 1
        var d = next_uniform_f(st2)
        st2 = d[1]
        logp += identical_log(d[0])
        if logp <= limit:
            break
    return (Float32(k - 1), st2)
