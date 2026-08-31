# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CityHash64, CatBoost's OWN variant. PORT OF `util/digest/city.cpp` at
CatBoost `54a8143a`, the 64-bit unseeded entry point only -- the one
function `CalcCatFeatureHash` stands on (`libs/cat_feature/
cat_feature.cpp:6-8`, see `gbdt/cat_feature/cat_feature.mojo`).

THE TRAP THIS FILE EXISTS TO NOT FALL INTO: their `city.h` says it plainly
-- "These functions provide CityHash 1.0 implementation whose results are
*different* from the mainline version of CityHash." A port written from
Google's published CityHash (1.1+, which changed `HashLen0to16` and the
mixing constants) or checked against any public CityHash test vector would
be a DIFFERENT hash agreeing on nothing. Every function below is
transcribed from THEIR `city.cpp`, cited by line, and gated cell for cell
against their own file compiled by `tools/cityhash_oracle/`
(`pixi run check-cityhash`).

Why the hash matters at all: every category CatBoost ever stores or looks
up -- pool loading, the model file's `ctr_data.hash_map`, the apply-time
combination key chain (`libs/model/ctr_provider.h:94-122`) -- is keyed by
`CalcCatFeatureHash`, i.e. by THIS function's low 32 bits. This port's
dense sorted-unique codes are equivalent for training (a CTR value depends
only on counts), but the moment a model file must interop with theirs, or
raw strings arrive at `train()`/`predict()` without a prep script, the key
IS this hash.

Byte order: their `UNALIGNED_LOAD64/32` is `ReadUnaligned`
(`util/system/unaligned_mem.h:13`), a memcpy in NATIVE byte order. Every
target this repository ships on (x86-64, ARM64, and all three GPU vendors'
hosts) is little-endian, so `_load64`/`_load32` assemble little-endian and
say so here rather than pretending to a portability their original never
had.

Host-side only, like theirs: the GPU never hashes a string.
"""

# Some primes between 2^63 and 2^64 for various uses (city.cpp:51-54).
comptime K0 = UInt64(0xC3A5C85C97CB3127)
comptime K1 = UInt64(0xB492B66FBE98F273)
comptime K2 = UInt64(0x9AE16A3B2F90404F)
comptime K3 = UInt64(0xC949D7C7509E6557)

# Murmur-inspired mixing multiplier of Hash128to64 (city.h:38).
comptime K_MUL = UInt64(0x9DDFEA08EB382D69)


def _load64(s: Span[UInt8, _], i: Int) -> UInt64:
    var v = UInt64(0)
    for k in range(8):
        v |= UInt64(s[i + k]) << UInt64(8 * k)
    return v


def _load32(s: Span[UInt8, _], i: Int) -> UInt64:
    # Their call sites take UNALIGNED_LOAD32 straight into ui64 arithmetic
    # (city.cpp:85-86), so the zero extension happens here.
    var v = UInt64(0)
    for k in range(4):
        v |= UInt64(s[i + k]) << UInt64(8 * k)
    return v


def _rotate(val: UInt64, shift: Int) -> UInt64:
    # city.cpp:58-61, the shift==0 guard included: shifting by 64 is as
    # undefined in Mojo's LLVM as in theirs.
    if shift == 0:
        return val
    return (val >> UInt64(shift)) | (val << UInt64(64 - shift))


def _rotate_by_at_least_1(val: UInt64, shift: Int) -> UInt64:
    # city.cpp:66-68.
    return (val >> UInt64(shift)) | (val << UInt64(64 - shift))


def _shift_mix(val: UInt64) -> UInt64:
    # city.cpp:70-72.
    return val ^ (val >> 47)


def _hash_len_16(u: UInt64, v: UInt64) -> UInt64:
    # HashLen16 = Hash128to64 (city.cpp:74-76 over city.h:36-45).
    var a = (u ^ v) * K_MUL
    a ^= a >> 47
    var b = (v ^ a) * K_MUL
    b ^= b >> 47
    b *= K_MUL
    return b


def _hash_len_0_to_16(s: Span[UInt8, _], length: Int) -> UInt64:
    # city.cpp:78-97.
    if length > 8:
        var a = _load64(s, 0)
        var b = _load64(s, length - 8)
        return (
            _hash_len_16(a, _rotate_by_at_least_1(b + UInt64(length), length))
            ^ b
        )
    if length >= 4:
        var a = _load32(s, 0)
        return _hash_len_16(UInt64(length) + (a << 3), _load32(s, length - 4))
    if length > 0:
        var a = s[0]
        var b = s[length >> 1]
        var c = s[length - 1]
        var y = UInt32(a) + (UInt32(b) << 8)
        var z = UInt32(length) + (UInt32(c) << 2)
        return _shift_mix(UInt64(y) * K2 ^ UInt64(z) * K3) * K2
    return K2


def _hash_len_17_to_32(s: Span[UInt8, _], length: Int) -> UInt64:
    # city.cpp:101-108.
    var a = _load64(s, 0) * K1
    var b = _load64(s, 8)
    var c = _load64(s, length - 8) * K2
    var d = _load64(s, length - 16) * K0
    return _hash_len_16(
        _rotate(a - b, 43) + _rotate(c, 30) + d,
        a + _rotate(b ^ K3, 20) - c + UInt64(length),
    )


def _weak_hash_len_32_with_seeds(
    w: UInt64, x: UInt64, y: UInt64, z: UInt64, a_in: UInt64, b_in: UInt64
) -> Tuple[UInt64, UInt64]:
    # city.cpp:112-121.
    var a = a_in + w
    var b = _rotate(b_in + a + z, 21)
    var c = a
    a += x
    a += y
    b += _rotate(a, 44)
    return (a + z, b + c)


def _weak_hash_len_32_with_seeds_at(
    s: Span[UInt8, _], i: Int, a: UInt64, b: UInt64
) -> Tuple[UInt64, UInt64]:
    # city.cpp:124-132.
    return _weak_hash_len_32_with_seeds(
        _load64(s, i),
        _load64(s, i + 8),
        _load64(s, i + 16),
        _load64(s, i + 24),
        a,
        b,
    )


def _hash_len_33_to_64(s: Span[UInt8, _], length: Int) -> UInt64:
    # city.cpp:135-156.
    var z = _load64(s, 24)
    var a = _load64(s, 0) + (UInt64(length) + _load64(s, length - 16)) * K0
    var b = _rotate(a + z, 52)
    var c = _rotate(a, 37)
    a += _load64(s, 8)
    c += _rotate(a, 7)
    a += _load64(s, 16)
    var vf = a + z
    var vs = b + _rotate(a, 31) + c
    a = _load64(s, 16) + _load64(s, length - 32)
    z = _load64(s, length - 8)
    b = _rotate(a + z, 52)
    c = _rotate(a, 37)
    a += _load64(s, length - 24)
    c += _rotate(a, 7)
    a += _load64(s, length - 16)
    var wf = a + z
    var ws = b + _rotate(a, 31) + c
    var r = _shift_mix((vf + ws) * K2 + (wf + vs) * K0)
    return _shift_mix(r * K0 + vs) * K2


def city_hash_64(s: Span[UInt8, _]) -> UInt64:
    # city.cpp:158-196.
    var length = len(s)
    if length <= 32:
        if length <= 16:
            return _hash_len_0_to_16(s, length)
        else:
            return _hash_len_17_to_32(s, length)
    elif length <= 64:
        return _hash_len_33_to_64(s, length)

    # For strings over 64 bytes we hash the end first, and then as we
    # loop we keep 56 bytes of state: v, w, x, y, and z (city.cpp:169-195).
    var x = _load64(s, 0)
    var y = _load64(s, length - 16) ^ K1
    var z = _load64(s, length - 56) ^ K0
    var v = _weak_hash_len_32_with_seeds_at(s, length - 64, UInt64(length), y)
    var w = _weak_hash_len_32_with_seeds_at(
        s, length - 32, UInt64(length) * K1, K0
    )
    z += _shift_mix(v[1]) * K1
    x = _rotate(z + x, 39) * K1
    y = _rotate(y, 33) * K1

    var remaining = (length - 1) & ~63
    var pos = 0
    while True:
        x = _rotate(x + y + v[0] + _load64(s, pos + 16), 37) * K1
        y = _rotate(y + v[1] + _load64(s, pos + 48), 42) * K1
        x ^= w[1]
        y ^= v[0]
        z = _rotate(z ^ w[0], 33)
        v = _weak_hash_len_32_with_seeds_at(s, pos, v[1] * K1, x + w[0])
        w = _weak_hash_len_32_with_seeds_at(s, pos + 32, z + w[1], y)
        var t = z
        z = x
        x = t
        pos += 64
        remaining -= 64
        if remaining == 0:
            break
    return _hash_len_16(
        _hash_len_16(v[0], w[0]) + _shift_mix(y) * K1 + z,
        _hash_len_16(v[1], w[1]) + x,
    )
