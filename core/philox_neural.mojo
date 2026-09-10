# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The neural-training random stream: Philox4x32-10 keyed by
(seed, stream id, element index), so every draw is a pure function of its
position and never of the launch geometry.

`core/philox.mojo` (the forest lane's, read-only here) supplies the block
function `philox4x32_10`; this file supplies the neural stream's LAYOUT:

    counter = (block_lo, block_hi, stream_id, 0)   key = (seed_lo, seed_hi)

    uniform  element i reads block i // 4, word i % 4
    normal   element i reads block i // 2, words 2*(i%2) and 2*(i%2)+1
    dropout  element i reads the UNIFORM word of element i (same layout)

INTEGER-TO-FLOAT MAPPING, fixed and exact on every vendor:

    unit(x)      = Float32(x >> 8) * 2^-24            in [0, 1), exact
    unit_open(x) = Float32((x >> 8) + 1) * 2^-24      in (0, 1], exact

    uniform(lo, hi):  ftz(fma(unit, hi - lo, lo))       ONE rounding
    normal(mean, sd): u1 = unit_open(w0), u2 = unit(w1)
                      r  = ftz(sqrt(ftz(-2 * log(u1))))
                      z  = ftz(r * cos(ftz(2*pi * u2)))
                      ftz(fma(sd, z, mean))
    dropout(p, s):    keep = unit >= p ; y = keep ? ftz(x * s) : +0.0
                      backward dx = keep ? ftz(dy * s) : +0.0

`log`, `sqrt` and `cos` go through the `identical_*` seams of
`checks/numerics.mojo`, so the IDENTICAL build uses the portable
implementations and FAST uses the vendor's. The scale `s = 1 / (1 - p)`
and the span `hi - lo` are HOST float32 scalars handed in by the caller.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.math import isfinite
from std.memory import bitcast
from max.gpu.host import DeviceContext

from core.philox import philox4x32_10
from checks.numerics import (
    ftz,
    identical_cos,
    identical_log,
    identical_mul,
    identical_mul_add,
    identical_sqrt,
)


comptime NRNG_TPB = 256

comptime NRNG_UNIFORM = 0
comptime NRNG_NORMAL = 1
comptime NRNG_DROPOUT_FORWARD = 2
comptime NRNG_DROPOUT_BACKWARD = 3

comptime NRNG_TWO_POW_M24_BITS: UInt32 = 0x33800000
comptime NRNG_TWO_PI_BITS: UInt32 = 0x40C90FDB


@always_inline
def neural_block(
    seed_lo: UInt32, seed_hi: UInt32, stream_id: UInt32, block_index: UInt64
) -> SIMD[DType.uint32, 4]:
    """The four words of one (seed, stream, block) cell."""
    var ctr = SIMD[DType.uint32, 4](
        UInt32(block_index & 0xFFFFFFFF),
        UInt32((block_index >> 32) & 0xFFFFFFFF),
        stream_id,
        UInt32(0),
    )
    var key = SIMD[DType.uint32, 2](seed_lo, seed_hi)
    return philox4x32_10(ctr, key)


@always_inline
def neural_unit(x: UInt32) -> Float32:
    """`(x >> 8) * 2^-24` in [0, 1), exact: a 24-bit integer times a power of two."""
    var scale = bitcast[DType.float32](NRNG_TWO_POW_M24_BITS)
    return Float32(Int(x >> 8)) * scale


@always_inline
def neural_unit_open(x: UInt32) -> Float32:
    """`((x >> 8) + 1) * 2^-24` in (0, 1], exact; the log argument."""
    var scale = bitcast[DType.float32](NRNG_TWO_POW_M24_BITS)
    return Float32(Int(x >> 8) + 1) * scale


@always_inline
def neural_uniform_at(
    seed_lo: UInt32, seed_hi: UInt32, stream_id: UInt32, index: Int,
    lo: Float32, span: Float32,
) -> Float32:
    var words = neural_block(seed_lo, seed_hi, stream_id, UInt64(index // 4))
    var w = index - (index // 4) * 4
    var x: UInt32
    if w == 1:
        x = words[1]
    elif w == 2:
        x = words[2]
    elif w == 3:
        x = words[3]
    else:
        x = words[0]
    return ftz(identical_mul_add(neural_unit(x), ftz(span), ftz(lo)))


@always_inline
def neural_unit_at(
    seed_lo: UInt32, seed_hi: UInt32, stream_id: UInt32, index: Int
) -> Float32:
    """The dropout coin of element `index`: the uniform word in [0, 1)."""
    var words = neural_block(seed_lo, seed_hi, stream_id, UInt64(index // 4))
    var w = index - (index // 4) * 4
    var x: UInt32
    if w == 1:
        x = words[1]
    elif w == 2:
        x = words[2]
    elif w == 3:
        x = words[3]
    else:
        x = words[0]
    return neural_unit(x)


@always_inline
def neural_normal_at(
    seed_lo: UInt32, seed_hi: UInt32, stream_id: UInt32, index: Int,
    mean: Float32, sd: Float32,
) -> Float32:
    var words = neural_block(seed_lo, seed_hi, stream_id, UInt64(index // 2))
    var w0 = (index - (index // 2) * 2) * 2
    var x0: UInt32
    var x1: UInt32
    if w0 == 2:
        x0 = words[2]
        x1 = words[3]
    else:
        x0 = words[0]
        x1 = words[1]
    var u1 = neural_unit_open(x0)
    var u2 = neural_unit(x1)
    var l = identical_log(u1)
    var t = ftz(identical_mul(Float32(-2.0), l))
    var r = ftz(identical_sqrt(t))
    var two_pi = bitcast[DType.float32](NRNG_TWO_PI_BITS)
    var c = ftz(identical_cos(ftz(identical_mul(two_pi, u2))))
    var z = ftz(identical_mul(r, c))
    return ftz(identical_mul_add(ftz(sd), z, ftz(mean)))


def neural_rng_kernel(
    out_buf: MutPointer[Float32, MutAnyOrigin],
    in_buf: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    offset_in: Int32,
    seed_lo_in: Int32,
    seed_hi_in: Int32,
    stream_in: Int32,
    kind_in: Int32,
    a: Float32,
    b: Float32,
):
    """One thread per element. `offset` is the element index of `out[0]`,
    so a slice of a stream equals the same elements of the whole stream.
    `a`, `b` are (lo, span) for UNIFORM, (mean, sd) for NORMAL and
    (p, scale) for the two DROPOUT arms."""
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    var index = Int(offset_in) + i
    var seed_lo = seed_lo_in.cast[DType.uint32]()
    var seed_hi = seed_hi_in.cast[DType.uint32]()
    var stream = stream_in.cast[DType.uint32]()
    var kind = Int(kind_in)
    if kind == NRNG_UNIFORM:
        out_buf.unsafe_store(
            i, neural_uniform_at(seed_lo, seed_hi, stream, index, a, b)
        )
        return
    if kind == NRNG_NORMAL:
        out_buf.unsafe_store(
            i, neural_normal_at(seed_lo, seed_hi, stream, index, a, b)
        )
        return
    var coin = neural_unit_at(seed_lo, seed_hi, stream, index)
    var value = Float32(0.0)
    if coin >= a:
        value = ftz(identical_mul(ftz(in_buf.unsafe_load(i)), ftz(b)))
    out_buf.unsafe_store(i, value)


def neural_rng_host(
    ctx: DeviceContext,
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    in_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n: Int,
    offset: Int,
    seed_lo: Int,
    seed_hi: Int,
    stream_id: Int,
    kind: Int,
    a: Float32,
    b: Float32,
) raises -> Int:
    """`n` draws of `kind` into `out_ptr`, elements `offset .. offset + n`
    of stream `(seed, stream_id)`. The DROPOUT arms read `n` floats from
    `in_ptr` (x for the forward, dy for the backward); the other two never
    touch it and it may be a one-float placeholder. Returns `n`."""
    if n < 1:
        raise Error("mojolearn neural rng: n must be at least 1, got " + String(n))
    if offset < 0:
        raise Error("mojolearn neural rng: offset must be >= 0, got " + String(offset))
    if kind < NRNG_UNIFORM or kind > NRNG_DROPOUT_BACKWARD:
        raise Error("mojolearn neural rng: unknown kind " + String(kind))
    if seed_lo < 0 or seed_lo > 0xFFFFFFFF or seed_hi < 0 or seed_hi > 0xFFFFFFFF:
        raise Error("mojolearn neural rng: seed halves must be 32-bit unsigned")
    if stream_id < 0 or stream_id > 0xFFFFFFFF:
        raise Error("mojolearn neural rng: stream id must be 32-bit unsigned")
    if not isfinite(a) or not isfinite(b):
        raise Error("mojolearn neural rng: non-finite scalar parameter")
    if kind == NRNG_UNIFORM and b < Float32(0.0):
        raise Error("mojolearn neural rng: uniform span must be >= 0")
    if kind == NRNG_NORMAL and b < Float32(0.0):
        raise Error("mojolearn neural rng: normal sd must be >= 0")
    var reads_input = kind == NRNG_DROPOUT_FORWARD or kind == NRNG_DROPOUT_BACKWARD
    if reads_input:
        if a < Float32(0.0) or a >= Float32(1.0):
            raise Error("mojolearn neural rng: dropout p must be in [0, 1)")
        if b <= Float32(0.0):
            raise Error("mojolearn neural rng: dropout scale must be > 0")
        for i in range(n):
            if not isfinite(in_ptr.unsafe_load(i)):
                raise Error(
                    "mojolearn neural rng: non-finite dropout input at " + String(i)
                )
    var in_n = n if reads_input else 1
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n)
    var in_buf = ctx.enqueue_create_buffer[DType.float32](in_n)
    if reads_input:
        ctx.enqueue_copy(dst_buf=in_buf, src_ptr=in_ptr)
    ctx.synchronize()
    ctx.enqueue_function[neural_rng_kernel](
        out_buf.unsafe_ptr(),
        in_buf.unsafe_ptr(),
        Int32(n),
        Int32(offset),
        UInt32(seed_lo).cast[DType.int32](),
        UInt32(seed_hi).cast[DType.int32](),
        UInt32(stream_id).cast[DType.int32](),
        Int32(kind),
        a,
        b,
        grid_dim=((n + NRNG_TPB - 1) // NRNG_TPB, 1, 1),
        block_dim=(NRNG_TPB, 1, 1),
    )
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=out_ptr, src_buf=out_buf)
    ctx.synchronize()
    _ = out_buf^
    _ = in_buf^
    return n
