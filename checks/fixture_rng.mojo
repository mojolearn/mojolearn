# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The fixture RNG: one definition per DISTINCT behavior.

Every check fills its fixtures from integer hashes. Before this module the
repository held about 150 private copies of those hashes (`_mix`,
`splitmix`, `_splitmix`, `_u01`, `_hashed` and relatives). A copy that
drifts by one constant makes a cross-lane or cross-vendor comparison
compare two FIXTURES instead of two kernels, and nothing downstream sees
it, because both arms of a check read the same (wrong) values.

The behaviors below are DIFFERENT functions, kept apart on purpose: a
different offset, a different constant or a different output mapping is a
different stream, and every recorded hash depends on the stream its
fixture used. Never merge two of them, and never "fix" one to match
another. `checks/fixture_rng_gate.mojo` proves every existing copy equals
its counterpart here bit for bit (pixi run check-fixture-rng).

New check and binding code imports from here. Integer arithmetic only,
except where a behavior's own output mapping is a float; `+` and `*` on
`UInt64`/`UInt32` wrap, and are NEVER written `&+` (in Mojo `&+` is a
bitwise AND and compiles silently).
"""

comptime GOLDEN64 = UInt64(0x9E3779B97F4A7C15)
comptime SPLITMIX_M1 = UInt64(0xBF58476D1CE4E5B9)
comptime SPLITMIX_M2 = UInt64(0x94D049BB133111EB)


# --- 64-bit mixers of one word -------------------------------------------


def splitmix64(x: UInt64) -> UInt64:
    """splitmix64 (Steele, Lea, Flood): golden-ratio increment, then the
    30/27/31 finalizer."""
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def splitmix64_finalizer(x: UInt64) -> UInt64:
    """splitmix64's finalizer alone, with NO golden-ratio increment."""
    var z = x
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def splitmix64_fold(h: UInt64, v: UInt64) -> UInt64:
    """One digest step: the splitmix64 finalizer of `h ^ v`."""
    return splitmix64_finalizer(h ^ v)


def murmur3_fmix64(x: UInt64) -> UInt64:
    """Murmur3's 64-bit finalizer (33/33/33 shifts). NOT splitmix64."""
    var h = x
    h ^= h >> 33
    h *= 0xFF51AFD7ED558CCD
    h ^= h >> 33
    h *= 0xC4CEB9FE1A85EC53
    h ^= h >> 33
    return h


# --- 32-bit mixers of one word -------------------------------------------


def murmur3_fmix32(x: UInt32) -> UInt32:
    """Murmur3's 32-bit finalizer (16/13/16 shifts)."""
    var h = x
    h ^= h >> 16
    h *= 0x85EBCA6B
    h ^= h >> 13
    h *= 0xC2B2AE35
    h ^= h >> 16
    return h


def seeded_fmix32(x: UInt32) -> UInt32:
    """`x ^ 0x9E3779B9`, then Murmur3's fmix32 WITHOUT its first shift."""
    var h = x
    h = h ^ UInt32(0x9E3779B9)
    h = h * UInt32(0x85EBCA6B)
    h = h ^ (h >> 13)
    h = h * UInt32(0xC2B2AE35)
    h = h ^ (h >> 16)
    return h


def lowbias32(x: UInt32) -> UInt32:
    """The `lowbias32` mixer (0x7FEB352D, 0x846CA68B; 16/15/16 shifts)."""
    var h = x
    h ^= h >> 16
    h *= UInt32(0x7FEB352D)
    h ^= h >> 15
    h *= UInt32(0x846CA68B)
    h ^= h >> 16
    return h


def golden32_mix(x: UInt32) -> UInt32:
    """`x * 0x9E3779B1`, then 15/13/16 shifts with 0x85EBCA77, 0xC2B2AE3D."""
    var h = x * UInt32(0x9E3779B1)
    h = h ^ (h >> UInt32(15))
    h = h * UInt32(0x85EBCA77)
    h = h ^ (h >> UInt32(13))
    h = h * UInt32(0xC2B2AE3D)
    return h ^ (h >> UInt32(16))


# --- splitmix64 finalizer over a linear combination of indices -----------


def splitmix64_of_sum(i: Int, salt: Int) -> UInt64:
    """`splitmix64(UInt64(i) + UInt64(salt))`."""
    var z = UInt64(i) + UInt64(salt) + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def splitmix_pair(i: Int, salt: Int) -> UInt64:
    """Finalizer of `(i + 1) * GOLDEN + (salt + 1) * M1`."""
    var z = (
        UInt64(i + 1) * 0x9E3779B97F4A7C15
        + UInt64(salt + 1) * 0xBF58476D1CE4E5B9
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def splitmix_pair_unoffset(i: Int, salt: Int) -> UInt64:
    """Finalizer of `i * GOLDEN + salt * M1` (no `+ 1` on either index)."""
    var z = UInt64(i) * UInt64(0x9E3779B97F4A7C15) + UInt64(salt) * UInt64(
        0xBF58476D1CE4E5B9
    )
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def splitmix_triple(a: Int, b: Int, salt: Int) -> UInt64:
    """Finalizer of `(a + 1) * GOLDEN + (b + 1) * M1 + (salt + 1) * M2`."""
    var z = (
        UInt64(a + 1) * 0x9E3779B97F4A7C15
        + UInt64(b + 1) * 0xBF58476D1CE4E5B9
        + UInt64(salt + 1) * 0x94D049BB133111EB
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return z


def u01_triple(a: Int, b: Int, salt: Int) -> Float64:
    """`splitmix_triple` to `[0, 1)` at 53 bits."""
    return Float64(splitmix_triple(a, b, salt) >> 11) * (1.0 / 9007199254740992.0)


def _row_word(row: Int, k: Int, salt: Int) -> UInt64:
    """Finalizer of `row * GOLDEN + (k + 1) * M1 + (salt + 1) * M2`: the
    `row` index has NO `+ 1`, unlike `splitmix_triple`."""
    var z = (
        UInt64(row) * 0x9E3779B97F4A7C15
        + UInt64(k + 1) * 0xBF58476D1CE4E5B9
        + UInt64(salt + 1) * 0x94D049BB133111EB
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return z


def u01_row(row: Int, k: Int, salt: Int) -> Float64:
    """The bench `_u01` (twin of `bench/bench_sklearn.py::u01`): `_row_word`
    to `[0, 1)` at 53 bits."""
    return Float64(_row_word(row, k, salt) >> 11) * (1.0 / 9007199254740992.0)


def u16_row_f32(row: Int, k: Int, salt: Int) -> Float32:
    """`_row_word` bits 40..55 as `q / 65536`: exact in Float32."""
    var z = _row_word(row, k, salt)
    return Float32(Int((z >> 40) & UInt64(0xFFFF))) / Float32(65536.0)


def u01_row_feature(row: Int, feature: Int) -> Float64:
    """The neighbors `_hash01`: `u01_row(row, feature, 0)`, spelled with the
    salt term folded to `+ M2`."""
    var z = (
        UInt64(row) * UInt64(0x9E3779B97F4A7C15)
        + UInt64(feature + 1) * UInt64(0xBF58476D1CE4E5B9)
        + UInt64(0x94D049BB133111EB)
    )
    z = (z ^ (z >> UInt64(30))) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> UInt64(27))) * UInt64(0x94D049BB133111EB)
    z = z ^ (z >> UInt64(31))
    return Float64(z >> UInt64(11)) * (1.0 / 9007199254740992.0)


def symmetric_cell_f64(i: Int, j: Int, seed: Int) -> Float64:
    """A symmetric cell in `[-0.5, 0.5)`: the finalizer of
    `(min + 1) * GOLDEN + (max + 1) * M1 + seed * M2` (no `+ 1` on seed)."""
    var lo = i if i < j else j
    var hi = j if i < j else i
    var z = (
        UInt64(lo + 1) * 0x9E3779B97F4A7C15
        + UInt64(hi + 1) * 0xBF58476D1CE4E5B9
        + UInt64(seed) * 0x94D049BB133111EB
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return Float64(z >> 11) * (1.0 / 9007199254740992.0) - 0.5


def splitmix_low31(x: Int) -> Int:
    """Finalizer of `(x + 1) * GOLDEN`, low 31 bits, as a non-negative Int."""
    var h = UInt64(x + 1) * UInt64(0x9E3779B97F4A7C15)
    h ^= h >> 30
    h *= UInt64(0xBF58476D1CE4E5B9)
    h ^= h >> 27
    h *= UInt64(0x94D049BB133111EB)
    h ^= h >> 31
    return Int(h & UInt64(0x7FFFFFFF))


def golden_top24(tid: Int) -> UInt64:
    """`(tid * GOLDEN) >> 40`: a scattered 24-bit addend."""
    return (UInt64(tid) * 0x9E3779B97F4A7C15) >> 40


# --- the 29/32 two-round mixer (NOT the splitmix finalizer) --------------


def mix29_pair(i: Int, salt: Int) -> UInt64:
    """`i * GOLDEN + (salt + 1) * M1`, then `^>>29`, `*M2`, `^>>32`."""
    var h = UInt64(i) * UInt64(0x9E3779B97F4A7C15) + UInt64(salt + 1) * UInt64(
        0xBF58476D1CE4E5B9
    )
    h = h ^ (h >> UInt64(29))
    h = h * UInt64(0x94D049BB133111EB)
    return h ^ (h >> UInt64(32))


def mix29_triple(i: Int, f: Int, salt: Int) -> UInt64:
    """`(i + 1) * GOLDEN + (f + salt) * M1`, then `^>>29`, `*M2`, `^>>32`."""
    var h = UInt64(i + 1) * UInt64(0x9E3779B97F4A7C15) + UInt64(
        f + salt
    ) * UInt64(0xBF58476D1CE4E5B9)
    h = h ^ (h >> UInt64(29))
    h = h * UInt64(0x94D049BB133111EB)
    return h ^ (h >> UInt64(32))


# --- Knuth multiplicative hashes -----------------------------------------


def modhash_salted(x: Int, salt: Int) -> Int:
    """`((x + 1) * 2654435761 + salt * 40503) mod 1000003`, non-negative."""
    var v = ((x + 1) * 2654435761 + salt * 40503) % 1000003
    if v < 0:
        v += 1000003
    return v


def modhash(x: Int) -> Int:
    """`(x * 2654435761) mod 1000003`, non-negative (no offset, no salt)."""
    var h = (x * 2654435761) % 1000003
    if h < 0:
        h += 1000003
    return h


def xorshift32_of_index(i: Int) -> UInt32:
    """`UInt32(i * 2654435761 + 0x9E3779B9)`, then xorshift 13/17/5."""
    var x = UInt32(i * 2654435761 + 0x9E3779B9)
    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5
    return x


def xorshift32_of_pair(a: Int, b: Int) -> UInt32:
    """`UInt32(a * 2654435761 + b * 40503 + 0x2545F491)`, then xorshift
    13/17/5."""
    var x = UInt32(a * 2654435761 + b * 40503 + 0x2545F491)
    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5
    return x


# --- seeded output mappings over splitmix64 ------------------------------


def hashed_unit_f64(seed: UInt64, i: Int) -> Float64:
    """`splitmix64(seed ^ (i * 2654435761))` to `[0, 1)` at 53 bits."""
    var h = splitmix64(seed ^ UInt64(i * 2654435761))
    return Float64(h >> 11) * (1.0 / Float64(1 << 53))


def hashed_signed_f32(seed: UInt64, i: Int) -> Float32:
    """`splitmix64(seed ^ (i * 2654435761)) % 2000 / 1000 - 1`."""
    var h = splitmix64(seed ^ UInt64(i * 2654435761))
    return Float32(Int(h % UInt64(2000))) / Float32(1000.0) - Float32(1.0)


def hashed_unit_f32_24(tag: UInt64, i: Int, j: Int) -> Float32:
    """A 24-bit Float32 in `[0, 1)` from
    `splitmix64(tag * 0x100000001B3 + i * 7919 + j)`; exact."""
    var h = splitmix64(tag * UInt64(0x100000001B3) + UInt64(i) * UInt64(7919) + UInt64(j))
    var top = UInt32((h >> 40) & UInt64(0xFFFFFF))
    return Float32(top) * Float32(5.9604644775390625e-08)


def hashed_in_range(kind: Int, bid: Int, i: Int, salt: Int, lo: Float64, hi: Float64) -> Float32:
    """`lo + (hi - lo) * u01_triple(kind * 977 + bid, i, salt)` as Float32."""
    var u = u01_triple(kind * 977 + bid, i, salt)
    return Float32(lo + (hi - lo) * u)


def binade_hashed_f32(seed: UInt64, idx: Int, binade_shift: Int) -> Float32:
    """`+-(1 + m) * 2^(e + binade_shift)` from `splitmix64(seed + idx)`:
    `m` a 23-bit fraction, `e` in `[-3, 4]`. Exact; never zero and never
    subnormal while `binade_shift` is in `[-100, 100]`. Scaled by a loop of
    exact multiplies, never a `pow`."""
    var h = splitmix64(seed + UInt64(idx))
    var frac = Int((h >> 41) & UInt64(0x7FFFFF))
    var mant = Float64(frac) * 1.1920928955078125e-07
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
