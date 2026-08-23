"""Hashed fixtures for the KDE gates and the card, ASSEMBLED FROM BITS.

NOT A PORT. IDENTITY_PATHS row 32's lesson applies here verbatim: a host
`target += v * w` chain is a contraction decision, so a fixture built by
host arithmetic can hand two machines different inputs before the first
kernel runs. Every value below is a splitmix64 hash turned into a float32
by BITCAST -- hashed mantissa, exponent 126 or 127 (|v| in [0.5, 2)), sign
from the hash -- and the "near" query rows are a training row with its low
mantissa bits replaced. No floating-point operation is performed.

Non-uniform and non-repeating per cell, so a permutation of rows or
columns changes the answer (`uniform-test-data-hides-permutation`).
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


def bits_value(z: UInt64, signed: Bool) -> Float32:
    """A float32 from bits: hashed mantissa, exponent 126 or 127 so |v| is
    in [0.5, 2), sign from the hash when `signed`."""
    var mant = UInt32(z & 0x7FFFFF)
    var expo = UInt32(126 + Int((z >> 23) & 1))
    var sign = UInt32(0)
    if signed and ((z >> 24) & 1) == 1:
        sign = UInt32(1)
    return bitcast[DType.float32]((sign << 31) | (expo << 23) | mant)


def train_fixture(n: Int, d: Int, salt: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        for k in range(d):
            out.append(bits_value(mix64(i, k, salt), True))
    return out^


def query_fixture(
    train: List[Float32], n_train: Int, n_query: Int, d: Int, salt: Int
) -> List[Float32]:
    """Even rows: a training row (`(q * 7) % n_train`) with the low 12
    mantissa bits of every feature replaced -- a NEAR point, inside every
    compact kernel's support at the gates' bandwidth. Odd rows: fresh
    hashed rows, mostly outside it."""
    var out = List[Float32]()
    for q in range(n_query):
        for k in range(d):
            var z = mix64(q, k, salt + 7)
            if q % 2 == 0:
                var src = train[((q * 7) % n_train) * d + k]
                var u = bitcast[DType.uint32](src)
                u = (u & UInt32(0xFFFFF000)) | UInt32(z & 0xFFF)
                out.append(bitcast[DType.float32](u))
            else:
                out.append(bits_value(z, True))
    return out^


def weight_fixture(n: Int, salt: Int) -> List[Float32]:
    """Positive weights in [0.5, 2), hashed."""
    var out = List[Float32]()
    for i in range(n):
        out.append(bits_value(mix64(i, 99, salt + 13), False))
    return out^
