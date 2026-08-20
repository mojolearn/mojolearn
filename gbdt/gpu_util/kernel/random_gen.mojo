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
