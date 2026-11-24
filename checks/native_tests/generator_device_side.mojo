# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""A Generator's draws on the GPU side equal the CPU column, word for word
(lane vdf-sched, 2026-11-24: `Generator(device=...)`).

`Generator(device="gpu")` (and None on a GPU installation) draws through the
device training image's primitive operation 6
(`training/native_primitive_dispatch.mojo` `native_primitive_operation`,
core/philox_neural.mojo `neural_rng_host`: one thread per element of
`neural_rng_kernel`); `Generator(device="cpu")` beside a GPU set draws through
the host training image's operation 6 (`core/mojolearn_training_host_impl.mojo`
`neural_rng_host_into`, the CPU column). Both are run here, in the one GPU
build, on the same words the draw door passes (bindings/generator_draw_native_door.mojo
`_generator_emit`), and every output word must be the same bits:

  uniform (lo, span), normal (mean, sd) and both dropout arms (p, scale),
  at element counts 1, 255, 257 and 4099 (one partial block, one block plus
  one thread, many blocks of the 256-thread launch), nonzero offsets, the
  last stream id and a 64-bit seed with both halves set.

generator_draw.mojo checks each column against an independent Philox; this
check is the device side against the host column directly, including the
normal draw that check does not cover.
"""
from std.testing import TestSuite
from std.memory import bitcast
from checks.native_tests.testsuite_cases import CPU_COLUMN_BUILD, require
from core.mojolearn_training_host_impl import neural_rng_host_into
from training.native_primitive_dispatch import native_primitive_operation

comptime KIND_UNIFORM = 0
comptime KIND_NORMAL = 1
comptime KIND_DROPOUT_FORWARD = 2
comptime KIND_DROPOUT_BACKWARD = 3
comptime MASK32: UInt64 = 0xFFFFFFFF
comptime SIDE_CASES = 16


def _host_draw(kind: Int, n: Int, offset: Int, seed: UInt64, stream: UInt64, a: Float32, b: Float32,
               inp: List[Float32]) raises -> List[Float32]:
    var out = List[Float32](length=n, fill=-7.0)
    _ = neural_rng_host_into(Int(out.unsafe_ptr()), Int(inp.unsafe_ptr()), n, offset, Int(seed & MASK32),
                             Int((seed >> 32) & MASK32), Int(stream), kind, a, b)
    _ = inp
    return out^


def _device_draw(kind: Int, n: Int, offset: Int, seed: UInt64, stream: UInt64, a: Float32, b: Float32,
                 inp: List[Float32]) raises -> List[Float32]:
    var out = List[Float32](length=n, fill=-9.0)
    var f: List[Float64] = [Float64(a), Float64(b)]
    _ = native_primitive_operation(6, [Int(out.unsafe_ptr()), Int(inp.unsafe_ptr())],
                                   [n, offset, Int(seed & MASK32), Int((seed >> 32) & MASK32), Int(stream), kind], f)
    _ = inp
    return out^


def _case(unit: Int) raises:
    """Case `unit`: kind = unit % 4, size = [1, 255, 257, 4099][unit // 4]."""
    var kind = unit % 4
    var sizes: List[Int] = [1, 255, 257, 4099]
    var n = sizes[unit // 4]
    var seed = UInt64(0xFEDCBA9876543210) if unit % 2 == 0 else UInt64(13 + unit)
    var stream = UInt64(0xFFFFFFFF) if unit == 15 else UInt64(0x80000003 + unit)
    var offset = 0 if unit < 4 else 17 * unit
    var a = Float32(-2.0)
    var b = Float32(8.0)
    if kind == KIND_NORMAL:
        a = Float32(0.5)
        b = Float32(1.25)
    elif kind != KIND_UNIFORM:
        a = Float32(0.3)
        b = Float32(1.0 / (1.0 - Float64(Float32(0.3))))
    var inp = List[Float32](length=1, fill=0.0)
    if kind == KIND_DROPOUT_FORWARD or kind == KIND_DROPOUT_BACKWARD:
        inp = List[Float32](length=n, fill=0.0)
        for i in range(n):  # small-loop(fixture fill)
            inp[i] = Float32((i % 23) - 11) / 8.0
    var host = _host_draw(kind, n, offset, seed, stream, a, b, inp)
    var device = _device_draw(kind, n, offset, seed, stream, a, b, inp)
    for i in range(n):  # small-loop(fixture words)
        require(bitcast[DType.uint32](device[i]) == bitcast[DType.uint32](host[i]),
                "generator device side: case " + String(unit) + " kind " + String(kind) + " n " + String(n)
                + " word " + String(i) + " is " + String(device[i]) + " on the device, " + String(host[i])
                + " on the CPU column")


def test_generator_device_side() raises:
    comptime if CPU_COLUMN_BUILD:
        # the CPU column has no device side to compare; generator_draw.mojo covers it
        return
    else:
        var failures = String("")
        for unit in range(SIDE_CASES):  # driver-loop: over fixtures, in order on the one training context
            try:
                _case(unit)
            except error:
                failures += "\n      case " + String(unit) + ": " + String(error)
        if failures.byte_length() > 0:
            raise Error("generator_device_side: the GPU side differs from the CPU column:" + failures)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
