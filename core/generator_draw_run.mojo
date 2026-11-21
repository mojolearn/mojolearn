# SPDX-License-Identifier: Apache-2.0
"""RUN of the training Generator: typed words only, no Python.

The door (`bindings/generator_draw_native_door.mojo`) turns one public draw
into a `GeneratorDrawRequest` at UNPACK and wraps the written Array at PACK.
The draw itself is ONE call of neural primitive operation 6 through the
training image's entry of the Samba table (`core/samba_entry_table.mojo`,
SAMBA_TABLE_TRAINING): `training/native_primitive_dispatch.mojo` on a device
image and `training/host/native_primitive_dispatch.mojo` on the host image.
Both run `neural_rng_host` over the same six integer and two float words the
former `neural_rng` Python entry of the same image unpacked from two lists,
in the same neural context, so the stream is the same function of
(seed, stream, element index) it was.

The initializer scalars are IEEE binary64 with +, *, / and the hardware
square root (correctly rounded, as `packaging/portable_math` `mojolearn_sqrt`
is on both wheel baselines). The one product that feeds an add is pinned so
no build contracts it: the bound is the value the binary64 sequence gives.

Nothing here loops over values: a draw is one launch whose kernel is parallel
over elements (`core/philox_neural.mojo`), so there is nothing to split.
"""
from std.math import sqrt
from checks.numerics import pinned_mul_f64
from core.entry_table import EntryRef, entry_ref_call
from core.samba_entry_table import SAMBA_PRIM_NEURAL_RNG

comptime GENERATOR_UNIFORM = 0
comptime GENERATOR_NORMAL = 1
comptime GENERATOR_DROPOUT = 2
comptime GENERATOR_DROPOUT_BACKWARD = 3
# One past the last stream id: the counter value that records exhaustion.
comptime GENERATOR_STREAMS = 4294967296


struct GeneratorDrawRequest(Copyable, Movable):
    """One draw: the resolved entry, the written and the read buffer, and the
    eight words of primitive operation 6."""

    var entry: EntryRef
    var target: Int
    var source: Int
    var count: Int
    var offset: Int
    var seed_low: Int
    var seed_high: Int
    var stream: Int
    var kind: Int
    var a: Float64
    var b: Float64

    def __init__(out self, entry: EntryRef, target: Int, source: Int, count: Int, offset: Int,
                 seed_low: Int, seed_high: Int, stream: Int, kind: Int, a: Float64, b: Float64):
        self.entry = entry
        self.target = target
        self.source = source
        self.count = count
        self.offset = offset
        self.seed_low = seed_low
        self.seed_high = seed_high
        self.stream = stream
        self.kind = kind
        self.a = a
        self.b = b


def generator_draw_run(request: GeneratorDrawRequest) raises -> Int:
    """The draw. The caller keeps both buffers alive past the call."""
    var addresses: Array[Int, 2] = [request.target, request.source]
    var ints: Array[Int, 6] = [request.count, request.offset, request.seed_low, request.seed_high,
                               request.stream, request.kind]
    var floats: Array[Float64, 2] = [request.a, request.b]
    return entry_ref_call(request.entry, SAMBA_PRIM_NEURAL_RNG, addresses, ints, floats)


def generator_stream_next(counter: Int) raises -> Int:
    """The stream id a draw consumes at `counter`; the caller stores
    `counter + 1`."""
    if counter >= GENERATOR_STREAMS:
        raise Error("RuntimeError: Generator: the stream counter is exhausted")
    if counter < 0:
        raise Error("Generator: counter must be nonnegative")
    return counter


def generator_sqrt(x: Float64) raises -> Float64:
    if x < 0:
        raise Error("Generator initializer: math domain error")
    return sqrt(x)


def generator_div(numerator: Float64, denominator: Float64) raises -> Float64:
    if denominator == 0:
        raise Error("ZeroDivisionError: float division by zero")
    return numerator / denominator


def generator_kaiming_uniform_bound(has_slope: Bool, slope: Float64, fan_in: Float64) raises -> Float64:
    """sqrt(2 / (1 + a^2)) * sqrt(3 / fan_in); a is sqrt(5) without a slope."""
    var a = slope if has_slope else generator_sqrt(5.0)
    var square = pinned_mul_f64(a, a)
    var gain = generator_sqrt(generator_div(2.0, Float64(1.0) + square))
    return gain * generator_sqrt(generator_div(3.0, fan_in))


def generator_xavier_scale(uniform: Bool, gain: Float64, fan: Float64) raises -> Float64:
    """gain * sqrt(6 / fan) for the uniform bound, gain * sqrt(2 / fan) for
    the normal deviation; fan is fan_in + fan_out."""
    var ratio = generator_div(6.0 if uniform else 2.0, fan)
    return gain * generator_sqrt(ratio)


def generator_kaiming_normal_std(has_gain: Bool, gain: Float64, fan_in: Float64) raises -> Float64:
    """gain / sqrt(fan_in); gain is sqrt(2) when none is given."""
    var numerator = gain if has_gain else generator_sqrt(2.0)
    return generator_div(numerator, generator_sqrt(fan_in))


def generator_uniform_span(low: Float32, high: Float32) raises -> Float32:
    var span = Float32(Float64(high) - Float64(low))
    if not (span >= 0):
        raise Error("Generator.uniform: high must be >= low")
    return span


def generator_dropout_scale(probability: Float32) raises -> Float32:
    if not (probability >= 0 and probability < 1):
        raise Error("Generator.dropout: p must be in [0, 1)")
    return Float32(Float64(1) / (Float64(1) - Float64(probability)))
