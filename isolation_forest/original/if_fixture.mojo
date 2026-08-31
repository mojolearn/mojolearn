# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Hashed fixtures for the Isolation Forest gates and the card, ASSEMBLED
FROM BITS. NOT A PORT.

Every value is a splitmix64 hash turned into a float32 by BITCAST (the
kde lane's `bits_value`, IDENTITY_PATHS row 32's lesson: a host
arithmetic chain is a contraction decision, so a fixture built by host
arithmetic can hand two machines different inputs). Non-uniform,
non-repeating per cell, so a permutation of rows or columns changes the
answer (`uniform-test-data-hides-permutation`).

The shapes the gates want:
  * a BLOB of `n` rows x `d` features with |v| in [0.5, 2), signed, and
    PLANTED OUTLIERS at fixed rows whose every feature has |v| in
    [16, 64): an isolation forest must score those above the blob median
    (the semantic gate) -- and the planted rows are also the rows whose
    path lengths are shortest, which separates a working forest from a
    plausible one;
  * DUPLICATE rows (a row copied bit for bit over later rows), which
    exercises the `n_node_samples <= 1` vs `candidate_min < candidate_max`
    stopping order (identical rows never split);
  * a CONSTANT column (every row the same hashed bits), which exercises
    the "try every feature from a random start" loop at `:174-190`;
  * a SIGNED-ZERO column (ADDENDUM 11): `-0.0` and `+0.0` planted among
    positive values so a node's min is a zero and WHICH zero is decided
    by the positional strict `<` fold, never by a hardware `min`;
  * `n_rows < max_samples` (the Floyd sampler at `start = 0`).
"""

from std.memory import bitcast


def mix64(a: Int, b: Int, salt: Int) -> UInt64:
    var z = (
        UInt64(a + 1) * 0x9E3779B97F4A7C15
        + UInt64(b + 1) * 0xBF58476D1CE4E5B9
        + UInt64(salt + 1) * 0x94D049BB133111EB
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return z


def bits_value(z: UInt64, signed: Bool, expo_base: Int = 126) -> Float32:
    """A float32 from bits: hashed mantissa, exponent `expo_base` or
    `expo_base + 1` (126 -> |v| in [0.5, 2)), sign from the hash when
    `signed`."""
    var mant = UInt32(z & 0x7FFFFF)
    var expo = UInt32(expo_base + Int((z >> 23) & 1))
    var sign = UInt32(0)
    if signed and ((z >> 24) & 1) == 1:
        sign = UInt32(1)
    return bitcast[DType.float32]((sign << 31) | (expo << 23) | mant)


def is_planted_outlier(i: Int, n: Int, n_outliers: Int) -> Bool:
    """Outlier rows are spread through the index range: row
    `(k * 7919 + 13) % n` for k < n_outliers (7919 is prime; distinct
    rows for n_outliers < n when n is not a multiple of 7919)."""
    for k in range(n_outliers):
        if (k * 7919 + 13) % n == i:
            return True
    return False


def blob_fixture(
    n: Int, d: Int, salt: Int, n_outliers: Int
) -> List[Float32]:
    """Row-major `n x d`: blob rows |v| in [0.5, 2) signed; the planted
    outliers' every feature |v| in [16, 64) (exponent 130/131), same
    sign rule."""
    var out = List[Float32]()
    for i in range(n):
        var outlier = is_planted_outlier(i, n, n_outliers)
        for k in range(d):
            var z = mix64(i, k, salt)
            if outlier:
                out.append(bits_value(z, True, 130))
            else:
                out.append(bits_value(z, True, 126))
    return out^


def plant_duplicates(mut x: List[Float32], n: Int, d: Int, src: Int, copies: Int, stride: Int):
    """Copy row `src` over rows `src + stride`, `src + 2*stride`, ...
    (`copies` of them), bit for bit."""
    for c in range(1, copies + 1):
        var dst = src + c * stride
        if dst >= n:
            return
        for k in range(d):
            x[dst * d + k] = x[src * d + k]


def plant_constant_column(mut x: List[Float32], n: Int, d: Int, col: Int, salt: Int):
    """Every row's `col` = one hashed value (|v| in [0.5, 2))."""
    var v = bits_value(mix64(col, 9001, salt), True)
    for i in range(n):
        x[i * d + col] = v


def plant_signed_zero_column(mut x: List[Float32], n: Int, d: Int, col: Int):
    """Column `col`: rows `i % 4 == 1` get `-0.0`, rows `i % 4 == 3` get
    `+0.0`, the rest keep their hashed value with the sign bit CLEARED
    (so every zero is the column's min and the hashed values are the
    max side). The first zero a node meets in partition order is its
    min; ADDENDUM 11's planted case."""
    for i in range(n):
        if i % 4 == 1:
            x[i * d + col] = bitcast[DType.float32](UInt32(0x80000000))
        elif i % 4 == 3:
            x[i * d + col] = Float32(0.0)
        else:
            var u = bitcast[DType.uint32](x[i * d + col]) & UInt32(0x7FFFFFFF)
            x[i * d + col] = bitcast[DType.float32](u)


def to_column_major(x: List[Float32], n: Int, d: Int) -> List[Float32]:
    """The `order="F"` conversion cuML's `fit` performs
    (`isolation_forest.pyx:599-605`): a copy, no arithmetic."""
    var out = List[Float32]()
    for k in range(d):
        for i in range(n):
            out.append(x[i * d + k])
    return out^
