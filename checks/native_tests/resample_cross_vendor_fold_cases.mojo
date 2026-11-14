# SPDX-License-Identifier: Apache-2.0
"""The cross-vendor ordered fold, moved from test_cross_vendor.py and
test_cross_vendor_native.py (lane tmig-msel-metrics-tsa-1, 2026-11-14).

`cross_vendor.ordered_fold(parts, prefix=None)` copies the prefix (or the
first part) and adds every later part in logical order with the base image's
`fold_pair_f32` (core/parallel_entry_table.mojo, PB_FOLD_PAIR_F32):
dst[i] = canon(ftz(ftz(dst[i]) + ftz(src[i]))), one float32 rounding per add.
`ordered_fold` below is that driver loop over the native entry; every
expected value is an integer word or an independent float32 oracle.

Fixtures (CROSS_VENDOR_FOLD_FIXTURES):
  0  left fold, one rounding per add: 1 + 2^-24 + 2^-24 is 1.0; 2^-24 +
     2^-24 + 1 is 1 + 2^-23
  1  subnormals flush keeping the sign: ftz(1e-39) + 0 = +0, ftz(-1e-39) +
     -0 = -0, 3 + -3 = +0
  2  the word contract: [1, -1e-39, 1e-39, nan] + [2^-24, -0, 0, 2] +
     [2^-24, -0, 0, -2] = 0x3f800000, 0x80000000, 0, 0x7fc00000 (left fold,
     signed-zero FTZ, canonical NaN), the inputs unchanged; the pair fold of
     parts 1 and 2 as a prefix of part 0 gives 0x3f800001 in word 0
  3  five parts of 4096 values x 10^e, e in [-40, 2], equal word for word an
     independent oracle ftz(ftz(acc) + ftz(g)) in float32
  4  six parts of 512 such values: the fold from the prefix fold of parts 0-1
     over parts 2-5 equals the flat fold; part 0 as the prefix of parts 1-5
     equals it; the reversed order differs (the comparison can fail)
Fixtures 3 and 4 regenerate their values from a SplitMix64 stream here (the
former NumPy PCG64 normal draws); the properties do not depend on the values.
The socket coordinator groups, the buffer-view admission and the state hash
field order stay in Python (boundary).
"""
from std.memory import bitcast
from core.entry_table import EntryRef
from core.parallel_entry_table import parallel_fold_pair_f32

comptime CROSS_VENDOR_FOLD_FIXTURES = 5


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def word(value: Float32) -> UInt32:
    return bitcast[DType.uint32](value)


def from_word(bits: UInt32) -> Float32:
    return bitcast[DType.float32](bits)


def ordered_fold(entry: EntryRef, parts: List[List[Float32]], first: Int, prefix: List[Float32]) raises -> List[Float32]:
    """`ordered_fold(parts[first:], prefix=prefix)`: a copy of `prefix`, then
    every part from `first` on folded into it in order."""
    var total = prefix.copy()
    for k in range(first, len(parts)):  # driver-loop: over logical shard buffers in immutable order
        require(len(parts[k]) == len(total), "ordered_fold: parts differ in length")
        _ = parallel_fold_pair_f32(entry, Int(total.unsafe_ptr()), Int(parts[k].unsafe_ptr()), len(total))
    return total^


def flat_fold(entry: EntryRef, parts: List[List[Float32]]) raises -> List[Float32]:
    return ordered_fold(entry, parts, 1, parts[0])


def same_words(a: List[Float32], b: List[Float32], what: String) raises:
    require(len(a) == len(b), what + ": extent " + String(len(a)) + " vs " + String(len(b)))
    for i in range(len(a)):
        require(word(a[i]) == word(b[i]), what + ": word " + String(i) + " " + String(word(a[i])) + " vs "
                + String(word(b[i])))


def oracle_ftz(value: Float32) -> Float32:
    """Independent FTZ: a subnormal word keeps only its sign bit."""
    var bits = word(value)
    if (bits & 0x7F800000) == 0 and (bits & 0x007FFFFF) != 0:
        return from_word(bits & 0x80000000)
    return value


def scaled_values(n: Int, seed: Int) -> List[Float32]:
    """u * 10^e in binary64, then float32: u uniform in [-4, 4), e uniform in
    [-40, 2] (a SplitMix64 stream), so zeros, subnormals and normals mix."""
    var state = UInt64(seed) * 0x9E3779B97F4A7C15 + 0x632BE59BD9B4E019
    var out = List[Float32](length=n, fill=0)
    for i in range(n):
        state += 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
        z = z ^ (z >> 31)
        var u = (z >> 40).cast[DType.float64]() * (1.0 / 16777216.0) * 8.0 - 4.0
        var e = Int((z & 0xFFFF) % 43) - 40
        var scale: Float64 = 1.0
        for _ in range(e if e > 0 else -e):
            scale = scale * 10.0
        out[i] = Float32(u * scale if e >= 0 else u / scale)
    return out^


def fixture_left_fold(entry: EntryRef) raises:
    var one: List[Float32] = [1.0]
    var tiny: List[Float32] = [from_word(0x33800000)]  # 2^-24
    var forward: List[List[Float32]] = [one.copy(), tiny.copy(), tiny.copy()]
    var backward: List[List[Float32]] = [tiny.copy(), tiny.copy(), one.copy()]
    require(word(flat_fold(entry, forward)[0]) == 0x3F800000, "1 + 2^-24 + 2^-24 is not 1.0")
    require(word(flat_fold(entry, backward)[0]) == 0x3F800001, "2^-24 + 2^-24 + 1 is not 1 + 2^-23")


def fixture_ftz_sign(entry: EntryRef) raises:
    var sub = from_word(0x000AE398)  # 1e-39 as float32, a subnormal
    var a: List[Float32] = [sub, -sub, 3.0]
    var b: List[Float32] = [0.0, from_word(0x80000000), -3.0]
    var parts: List[List[Float32]] = [a^, b^]
    var out = flat_fold(entry, parts)
    require(word(out[0]) == 0x00000000, "ftz(1e-39) + 0 is not +0")
    require(word(out[1]) == 0x80000000, "ftz(-1e-39) + -0 is not -0")
    require(word(out[2]) == 0x00000000, "3 + -3 is not +0")


def fixture_word_contract(entry: EntryRef) raises:
    var sub = from_word(0x000AE398)
    var nan = from_word(0x7FC00000)
    var a: List[Float32] = [1.0, -sub, sub, nan]
    var b: List[Float32] = [from_word(0x33800000), from_word(0x80000000), 0.0, 2.0]
    var c: List[Float32] = [from_word(0x33800000), from_word(0x80000000), 0.0, -2.0]
    var parts: List[List[Float32]] = [a.copy(), b.copy(), c.copy()]
    var out = flat_fold(entry, parts)
    var expected: List[UInt32] = [0x3F800000, 0x80000000, 0x00000000, 0x7FC00000]
    for i in range(4):
        require(word(out[i]) == expected[i], "contract word " + String(i) + ": " + String(word(out[i])))
    same_words(parts[0], a, "part 0 changed")
    same_words(parts[1], b, "part 1 changed")
    same_words(parts[2], c, "part 2 changed")
    var pair_parts: List[List[Float32]] = [b.copy(), c.copy()]
    var pair = flat_fold(entry, pair_parts)
    var only_a: List[List[Float32]] = [a.copy()]
    var prefixed = ordered_fold(entry, only_a, 0, pair)
    require(word(prefixed[0]) == 0x3F800001, "the pair prefix then part 0 is not 1 + 2^-23: " + String(word(prefixed[0])))


def fixture_oracle(entry: EntryRef) raises:
    var parts = List[List[Float32]]()
    for k in range(5):
        parts.append(scaled_values(4096, 7 + 1000 * k))
    var want = parts[0].copy()
    for k in range(1, 5):
        for i in range(4096):
            want[i] = oracle_ftz(oracle_ftz(want[i]) + oracle_ftz(parts[k][i]))
    var subnormal = 0
    for k in range(5):
        for i in range(4096):
            var bits = word(parts[k][i])
            if (bits & 0x7F800000) == 0 and (bits & 0x007FFFFF) != 0:
                subnormal += 1
    require(subnormal > 0, "the oracle fixture holds no subnormal input")
    same_words(flat_fold(entry, parts), want, "fold vs independent float32 oracle")


def fixture_prefix(entry: EntryRef) raises:
    var parts = List[List[Float32]]()
    for k in range(6):
        parts.append(scaled_values(512, 5 + 100 * k))
    var flat = flat_fold(entry, parts)
    var head: List[List[Float32]] = [parts[0].copy(), parts[1].copy()]
    var prefix = flat_fold(entry, head)
    same_words(ordered_fold(entry, parts, 2, prefix), flat, "prefix fold of parts 0-1 then 2-5")
    same_words(ordered_fold(entry, parts, 1, parts[0]), flat, "part 0 as the prefix of parts 1-5")
    var reversed = List[List[Float32]]()
    for k in range(6):
        reversed.append(parts[5 - k].copy())
    var back = flat_fold(entry, reversed)
    var differs = False
    for i in range(len(back)):
        if word(back[i]) != word(flat[i]):
            differs = True
    require(differs, "the reversed fold equals the flat fold: the comparison cannot fail")


def run_fold_fixture(entry: EntryRef, unit: Int) raises:
    if unit == 0:
        fixture_left_fold(entry)
    elif unit == 1:
        fixture_ftz_sign(entry)
    elif unit == 2:
        fixture_word_contract(entry)
    elif unit == 3:
        fixture_oracle(entry)
    else:
        fixture_prefix(entry)
