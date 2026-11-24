# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""`Generator.dropout` / `dropout_backward` on a TrainDeviceArray equal the
host-input path and the CPU column, word for word (lane vdf-tail-d, 2026-11-24).

The device-array form (training/neural_rng_dev_run.mojo `neural_rng_dev_run`,
operation OPT_NATIVE_NEURAL_RNG_DEV of the training native table) reads the
input handle in place and writes a second handle; the host-input form is the
device training image's primitive operation 6 (`native_primitive_operation`,
which uploads the host input and downloads the result); the CPU column is the
host training image's `neural_rng_host_into`. All three run here on the words
the draw door passes (bindings/generator_draw_native_door.mojo), at element
counts 1, 255, 257 and 4099, both dropout arms, nonzero offsets, the last
stream id and a 64-bit seed with both halves set.

Refusals: a non-finite input is refused by the device scan with the host
form's words and index; the same handle as input and output is refused; a
kind that reads no input is refused.
"""
from std.testing import TestSuite
from std.memory import bitcast
from checks.native_tests.testsuite_cases import CPU_COLUMN_BUILD, require
from core.hostptr import f32_ptr
from core.mojolearn_training_host_impl import neural_rng_host_into
from core.neural_context import neural_ctx
from core.train_dev_pool import train_dev_alloc, train_dev_free, train_dev_put, train_dev_get
from training.dev_tensors import IDN_TRAIN_DEV_TENSORS
from training.native_primitive_dispatch import native_primitive_operation
from training.neural_rng_dev_run import neural_rng_dev_run, _NRNG_DEV_CTX

comptime KIND_DROPOUT_FORWARD = 2
comptime KIND_DROPOUT_BACKWARD = 3
comptime MASK32: UInt64 = 0xFFFFFFFF
comptime ARRAY_CASES = 8


def _input(n: Int) -> List[Float32]:
    var values = List[Float32](length=n, fill=0.0)
    for i in range(n):  # small-loop(fixture fill)
        values[i] = Float32((i % 23) - 11) / 8.0
    return values^


def _host_column(kind: Int, n: Int, offset: Int, seed: UInt64, stream: UInt64, p: Float32, scale: Float32,
                 inp: List[Float32]) raises -> List[Float32]:
    var out = List[Float32](length=n, fill=-7.0)
    _ = neural_rng_host_into(Int(out.unsafe_ptr()), Int(inp.unsafe_ptr()), n, offset, Int(seed & MASK32),
                             Int((seed >> 32) & MASK32), Int(stream), kind, p, scale)
    _ = inp
    return out^


def _host_input_path(kind: Int, n: Int, offset: Int, seed: UInt64, stream: UInt64, p: Float32, scale: Float32,
                     inp: List[Float32]) raises -> List[Float32]:
    var out = List[Float32](length=n, fill=-9.0)
    var f: List[Float64] = [Float64(p), Float64(scale)]
    _ = native_primitive_operation(6, [Int(out.unsafe_ptr()), Int(inp.unsafe_ptr())],
                                   [n, offset, Int(seed & MASK32), Int((seed >> 32) & MASK32), Int(stream), kind], f)
    _ = inp
    return out^


def _device_array_path(kind: Int, n: Int, offset: Int, seed: UInt64, stream: UInt64, p: Float32, scale: Float32,
                       inp: List[Float32]) raises -> List[Float32]:
    var ctx = neural_ctx[_NRNG_DEV_CTX]()
    var src = train_dev_alloc(ctx, n)
    var dst = train_dev_alloc(ctx, n)
    var out = List[Float32](length=n, fill=-5.0)
    try:
        train_dev_put(ctx, src, f32_ptr(Int(inp.unsafe_ptr())), n)
        var written = neural_rng_dev_run(n, offset, Int(seed & MASK32), Int((seed >> 32) & MASK32), Int(stream),
                                         kind, dst, src, p, scale)
        require(written == n, "generator dropout device array: wrote " + String(written) + " of " + String(n))
        train_dev_get(ctx, dst, f32_ptr(Int(out.unsafe_ptr())), n)
    finally:
        train_dev_free(ctx, dst)
        train_dev_free(ctx, src)
    _ = inp
    return out^


def _case(unit: Int) raises:
    """Case `unit`: kind = 2 + unit % 2, size = [1, 255, 257, 4099][unit // 2]."""
    var kind = KIND_DROPOUT_FORWARD + unit % 2
    var sizes: List[Int] = [1, 255, 257, 4099]
    var n = sizes[unit // 2]
    var seed = UInt64(0xFEDCBA9876543210) if unit % 2 == 0 else UInt64(29 + unit)
    var stream = UInt64(0xFFFFFFFF) if unit == 7 else UInt64(0x80000005 + unit)
    var offset = 0 if unit < 2 else 19 * unit
    var p = Float32(0.3)
    var scale = Float32(1.0 / (1.0 - Float64(Float32(0.3))))
    var inp = _input(n)
    var column = _host_column(kind, n, offset, seed, stream, p, scale, inp)
    var host_input = _host_input_path(kind, n, offset, seed, stream, p, scale, inp)
    var device = _device_array_path(kind, n, offset, seed, stream, p, scale, inp)
    for i in range(n):  # small-loop(fixture words)
        var d = bitcast[DType.uint32](device[i])
        require(d == bitcast[DType.uint32](column[i]),
                "case " + String(unit) + " kind " + String(kind) + " n " + String(n) + " word " + String(i)
                + " is " + String(device[i]) + " from the device array, " + String(column[i]) + " on the CPU column")
        require(d == bitcast[DType.uint32](host_input[i]),
                "case " + String(unit) + " kind " + String(kind) + " n " + String(n) + " word " + String(i)
                + " is " + String(device[i]) + " from the device array, " + String(host_input[i])
                + " from the host-input path")


def _refusal(unit: Int) raises:
    """0: a NaN at index 300 of 4099; 1: input and output the same handle; 2: the uniform kind."""
    var n = 4099
    var ctx = neural_ctx[_NRNG_DEV_CTX]()
    var inp = _input(n)
    if unit == 0:
        inp[300] = Float32(0.0) / Float32(0.0)
    var src = train_dev_alloc(ctx, n)
    var dst = train_dev_alloc(ctx, n)
    var message = String("")
    try:
        train_dev_put(ctx, src, f32_ptr(Int(inp.unsafe_ptr())), n)
        var kind = 0 if unit == 2 else KIND_DROPOUT_FORWARD
        var target = src if unit == 1 else dst
        try:
            _ = neural_rng_dev_run(n, 0, 1, 2, 3, kind, target, src, Float32(0.25), Float32(1.0 / 0.75))
        except error:
            message = String(error)
    finally:
        train_dev_free(ctx, dst)
        train_dev_free(ctx, src)
    _ = inp
    var want = String("mojolearn neural rng: non-finite dropout input at 300") if unit == 0 else (
        String("mojolearn neural rng: the dropout output must be disjoint from its input") if unit == 1 else
        String("mojolearn neural rng: device arrays are read by the dropout arms only, got kind 0"))
    require(message == want, "refusal " + String(unit) + ": got '" + message + "', expected '" + want + "'")


def test_generator_dropout_device_array() raises:
    comptime if CPU_COLUMN_BUILD or not IDN_TRAIN_DEV_TENSORS:
        # no training device arrays in this build: the door refuses them before RUN
        return
    else:
        var failures = String("")
        for unit in range(ARRAY_CASES):  # driver-loop: over fixtures, in order on the one training context
            try:
                _case(unit)
            except error:
                failures += "\n      case " + String(unit) + ": " + String(error)
        for unit in range(3):  # driver-loop: over the three refusal fixtures
            try:
                _refusal(unit)
            except error:
                failures += "\n      refusal " + String(unit) + ": " + String(error)
        if failures.byte_length() > 0:
            raise Error("generator_dropout_device_array: the device-array path differs:" + failures)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
