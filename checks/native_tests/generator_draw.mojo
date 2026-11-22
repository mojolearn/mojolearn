# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""The training Generator's draws against an independent Philox4x32-10
written here (lane tests-to-mojo-neural, 2026-11-15; moved from
python/mojolearn/tests/test_generator_draw_native.py, the `philox_words` /
`units` oracle comparisons).

The calls are the ones `Generator.uniform` / `.dropout` /
`.dropout_backward` reach: on the host column
`core/mojolearn_training_host_impl.mojo` `neural_rng_host_into`, on a device
`training/native_primitive_dispatch.mojo` `native_primitive_operation` 6
(`core/philox_neural.mojo` `neural_rng_host`), with the words the door
passes: kind 0 uniform (lo, span), 2 dropout forward and 3 backward
(p, scale), the seed split into its two 32-bit halves.

The oracle is the Python test's: counter (block lo, block hi, stream, 0),
key (seed lo, seed hi), ten rounds of the two Philox multipliers with the
Weyl key bumps; element i of a stream is word i % 4 of block i // 4 and its
unit is (word >> 8) * 2^-24, exact in float32.

  0  uniform, seed 0xfedcba9876543210, stream 0, low -2, high 6 (span 8):
     the 30 words of a fresh Generator (every shape of the Python test is a
     prefix of them) equal float32(unit * 8 - 2), bit for bit.
  1  dropout, seed 13, stream 0x80000003, offset 17, p 0.3, x = (-12..11) / 8:
     a word is x * scale where unit >= p and +0 elsewhere (scale is
     float32(1 / (1 - float32(0.3))) as the door rounds it); the backward of
     a constant 0.125 gradient is 0.125 * scale on the same mask.
  2  dropout chunks, seed 71, stream 0, p 0.25, x = (0..23) / 8: the four
     6-word pieces at offsets 0, 6, 12, 18 are the words of the whole draw,
     the whole draw obeys the oracle mask, and p 0 (stream 1) returns x.
  3  the last stream, seed 19, stream 0xffffffff: 11 uniform words of
     [0, 1) are the oracle units; an explicit stream 5 dropout of ones, p
     0.2, is the oracle mask times the scale.

The public shapes, the Generator's counter and state_dict, the input
layouts and the refusals stay in the Python test.
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases
from std.sys.compile import is_defined
from std.memory import bitcast
from checks.numerics import identical_mul
from core.mojolearn_training_host_impl import neural_rng_host_into
from training.native_primitive_dispatch import native_primitive_operation


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)

comptime CPU_COLUMN_BUILD = is_defined["MOJOLEARN_COLUMN_CPU"]()


comptime GEN_DRAW_FIXTURES = 4
comptime KIND_UNIFORM = 0
comptime KIND_DROPOUT_FORWARD = 2
comptime KIND_DROPOUT_BACKWARD = 3
comptime MASK32: UInt64 = 0xFFFFFFFF


# ---- The independent oracle ----
def philox_word(seed: UInt64, stream: UInt64, index: Int) -> UInt64:
    """Word `index % 4` of Philox4x32-10 block `index // 4`."""
    var block = UInt64(index // 4)
    var c0 = block & MASK32
    var c1 = block >> 32
    var c2 = stream & MASK32
    var c3 = UInt64(0)
    var k0 = seed & MASK32
    var k1 = (seed >> 32) & MASK32
    for _ in range(10):  # small-loop(the ten Philox rounds)
        var p0 = UInt64(0xD2511F53) * c0
        var p1 = UInt64(0xCD9E8D57) * c2
        var n0 = ((p1 >> 32) ^ c1 ^ k0) & MASK32
        var n1 = p1 & MASK32
        var n2 = ((p0 >> 32) ^ c3 ^ k1) & MASK32
        var n3 = p0 & MASK32
        c0 = n0
        c1 = n1
        c2 = n2
        c3 = n3
        k0 = (k0 + UInt64(0x9E3779B9)) & MASK32
        k1 = (k1 + UInt64(0xBB67AE85)) & MASK32
    var w = index % 4
    if w == 0:
        return c0
    if w == 1:
        return c1
    if w == 2:
        return c2
    return c3


def oracle_unit(seed: UInt64, stream: UInt64, index: Int) -> Float64:
    """(word >> 8) * 2^-24: a 24-bit integer times a power of two, exact."""
    return Float64(Int(philox_word(seed, stream, index) >> 8)) * (1.0 / 16777216.0)


def door_scale(p: Float32) -> Float32:
    """`_round_f32(1 / (1 - p))` with p already a float32."""
    return Float32(1.0 / (1.0 - Float64(p)))


# ---- The draw, through the column's entry ----
def draw(kind: Int, n: Int, offset: Int, seed: UInt64, stream: UInt64, a: Float32, b: Float32,
         inp: List[Float32]) raises -> List[Float32]:
    var out = List[Float32](length=n, fill=-7.0)
    var seed_lo = Int(seed & MASK32)
    var seed_hi = Int((seed >> 32) & MASK32)
    comptime if CPU_COLUMN_BUILD:
        _ = neural_rng_host_into(Int(out.unsafe_ptr()), Int(inp.unsafe_ptr()), n, offset, seed_lo, seed_hi,
                                 Int(stream), kind, a, b)
    else:
        var f: List[Float64] = [Float64(a), Float64(b)]
        _ = native_primitive_operation(6, [Int(out.unsafe_ptr()), Int(inp.unsafe_ptr())],
                                       [n, offset, seed_lo, seed_hi, Int(stream), kind], f)
    _ = inp
    return out^


def same_word(got: Float32, want: Float32, what: String, i: Int) raises:
    require(bitcast[DType.uint32](got) == bitcast[DType.uint32](want),
            what + ": word " + String(i) + " is " + String(got) + ", expected " + String(want))


def check_dropout(fwd: List[Float32], x: List[Float32], seed: UInt64, stream: UInt64, offset: Int,
                  p: Float32, scale: Float32, what: String) raises:
    for i in range(len(x)):  # small-loop(fixture words)
        var keep = Float32(oracle_unit(seed, stream, offset + i)) >= p
        var want = identical_mul(x[i], scale) if keep else Float32(0.0)
        same_word(fwd[i], want, what, i)


def placeholder() -> List[Float32]:
    var one: List[Float32] = [0.0]
    return one^


# ---- Fixtures ----
def uniform_fixture() raises:
    var seed = UInt64(0xFEDCBA9876543210)
    var got = draw(KIND_UNIFORM, 30, 0, seed, 0, -2.0, 8.0, placeholder())
    for i in range(30):  # small-loop(30 oracle words)
        same_word(got[i], Float32(oracle_unit(seed, 0, i) * 8.0 - 2.0), "uniform", i)


def dropout_fixture() raises:
    var seed = UInt64(13)
    var stream = UInt64(0x80000003)
    var p = Float32(0.3)
    var scale = door_scale(p)
    var x = List[Float32](length=24, fill=0.0)
    for i in range(24):  # small-loop(fixture fill)
        x[i] = Float32(i - 12) / 8.0
    var fwd = draw(KIND_DROPOUT_FORWARD, 24, 17, seed, stream, p, scale, x)
    check_dropout(fwd, x, seed, stream, 17, p, scale, "dropout forward")
    var dy = List[Float32](length=24, fill=0.125)
    var dx = draw(KIND_DROPOUT_BACKWARD, 24, 17, seed, stream, p, scale, dy)
    check_dropout(dx, dy, seed, stream, 17, p, scale, "dropout backward")


def chunk_fixture() raises:
    var seed = UInt64(71)
    var p = Float32(0.25)
    var scale = door_scale(p)
    var x = List[Float32](length=24, fill=0.0)
    for i in range(24):  # small-loop(fixture fill)
        x[i] = Float32(i) / 8.0
    var whole = draw(KIND_DROPOUT_FORWARD, 24, 0, seed, 0, p, scale, x)
    check_dropout(whole, x, seed, 0, 0, p, scale, "dropout whole")
    for row in range(4):  # small-loop(four row pieces)
        var piece = List[Float32](length=6, fill=0.0)
        for j in range(6):
            piece[j] = x[row * 6 + j]
        var got = draw(KIND_DROPOUT_FORWARD, 6, row * 6, seed, 0, p, scale, piece)
        for j in range(6):
            same_word(got[j], whole[row * 6 + j], "dropout piece " + String(row), j)
    var kept = draw(KIND_DROPOUT_FORWARD, 24, 0, seed, 1, 0.0, door_scale(0.0), x)
    for i in range(24):  # small-loop(fixture words)
        same_word(kept[i], x[i], "dropout p 0", i)


def last_stream_fixture() raises:
    var seed = UInt64(19)
    var last = UInt64(0xFFFFFFFF)
    var got = draw(KIND_UNIFORM, 11, 0, seed, last, 0.0, 1.0, placeholder())
    for i in range(11):  # small-loop(eleven oracle words)
        same_word(got[i], Float32(oracle_unit(seed, last, i)), "last stream uniform", i)
    var p = Float32(0.2)
    var scale = door_scale(p)
    var ones = List[Float32](length=3, fill=1.0)
    var fwd = draw(KIND_DROPOUT_FORWARD, 3, 0, seed, 5, p, scale, ones)
    check_dropout(fwd, ones, seed, 5, 0, p, scale, "explicit stream 5 dropout")


def run_fixture(unit: Int) raises:
    if unit == 0:
        uniform_fixture()
    elif unit == 1:
        dropout_fixture()
    elif unit == 2:
        chunk_fixture()
    else:
        last_stream_fixture()


# The host column runs the four independent fixtures across the host pool,
# each with its own buffers; a device runs them in order on the one shared
# training context, each launch keeping its own parallelism.


def _host_fixture(unit: Int) raises:
    run_fixture(unit)


def test_generator_draw() raises:
    comptime if CPU_COLUMN_BUILD:
        check_cases("generator_draw", GEN_DRAW_FIXTURES, _host_fixture)
    else:
        var failures = String("")
        for unit in range(GEN_DRAW_FIXTURES):  # driver-loop: over fixtures, in order
            try:
                run_fixture(unit)
            except error:
                failures += "\n      row " + String(unit) + ": " + String(error)
        if failures.byte_length() > 0:
            raise Error("generator_draw: device fixtures failed:" + failures)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
