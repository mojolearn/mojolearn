# SPDX-License-Identifier: Apache-2.0
"""Shared helpers of the sequence checks ported from the x_sequence Python
tests (lane tmig-sequence-2, 2026-11-14): the refusal helper, NumPy's
`assert_allclose` relation, hashed fixture data (the original tests drew
from NumPy's PCG64; their references never depended on the stream, so the
data is regenerated from `checks/fixture_rng.mojo`), a stored float32
fixture reader, and the parallel fixture driver of the CPU suites.

No `main()`: this module is imported by the cases and suites.
"""
from std.math import cos, log, sqrt
from std.memory import bitcast
from checks.fixture_rng import u01_triple
from core.host_parallel import host_parallelize

comptime TWO_PI = 6.283185307179586


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def allclose(actual: Float64, desired: Float64, rtol: Float64, atol: Float64) -> Bool:
    """`numpy.testing.assert_allclose`'s relation for one value."""
    return abs(actual - desired) <= atol + rtol * abs(desired)


def u01(salt: Int, a: Int, b: Int = 0) -> Float64:
    """A hashed uniform in [0, 1)."""
    return u01_triple(a, b, salt)


def normal(salt: Int, i: Int) -> Float64:
    """A hashed standard normal (Box-Muller over two hashed uniforms)."""
    var u1 = 1.0 - u01(salt, i, 0)
    var u2 = u01(salt, i, 1)
    return sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2)


def read_f32_file(path: String) raises -> List[Float32]:
    """A stored little-endian float32 vector (checks/fixtures/)."""
    var f = open(path, "r")
    var bytes = f.read_bytes()
    f.close()
    require(len(bytes) % 4 == 0, path + ": not a float32 vector")
    var count = len(bytes) // 4
    var values = List[Float32](length=count, fill=0.0)
    var vp = values.unsafe_ptr()
    var bp = bytes.unsafe_ptr()
    def word(i: Int) {imm vp, imm bp}:
        var w = (UInt32(bp[4 * i]) | (UInt32(bp[4 * i + 1]) << 8) | (UInt32(bp[4 * i + 2]) << 16)
                 | (UInt32(bp[4 * i + 3]) << 24))
        vp.unsafe_store(i, bitcast[DType.float32](w))
    host_parallelize(word, count)
    _ = bytes^
    return values^


def same_bits(a: Float32, b: Float32) -> Bool:
    return bitcast[DType.uint32](a) == bitcast[DType.uint32](b)


def count_failures(failures: List[Int]) -> Int:
    var failed = 0
    for value in failures:
        failed += value
    return failed
