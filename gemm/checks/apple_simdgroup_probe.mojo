# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-identical-neural (2026-09-26): what arithmetic Apple's fp32
simdgroup matrix multiply-accumulate performs, bit for bit.

The IDENTICAL GEMM step is `ftz(fma_rn(a, b, acc))`, one rounding per
product, `p` ascending within a leaf. On a window the block has proven
free of subnormal step results (`TUNED_WINDOW_ADMIT`) the flush is the
identity, so the step is the bare FMA. If `simdgroup_multiply_accumulate`
(8x8x8) returns, for every cell, exactly the ascending FMA chain seeded
with the incoming accumulator, then an admitted window can be computed on
the matrix path with the contract's bits.

One simdgroup per trial. A, B and C are generated from a hash of the trial
and position (no memory), with exponents spread so that the candidate
orders give different words. Every lane compares its two output elements
with four spellings of the same sum:
  bit 0  ascending FMA chain seeded C          (the contract's leaf step)
  bit 1  descending FMA chain seeded C
  bit 2  ascending unfused chain (round the product, then the add)
  bit 3  C + (ascending FMA chain seeded +0.0)
Kinds 4-6 are edge cases: infinities and NaNs, overflow, subnormal step
results. NaN words are compared as words (payload and sign included).
A zero count in column 0 over every kind is the claim; the other columns
show the probe can tell the orders apart.

    pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . \\
        gemm/checks/apple_simdgroup_probe.mojo -o <bin>
    python3 tools/mac_slot.py metal -- <bin>
"""

from std.ffi import external_call
from std.gpu import block_idx, thread_idx
from std.math import fma
from std.memory import bitcast
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import identical_mul_add
from max.gpu.host import DeviceContext
from transformer.impl.llama.modeling_llama import _download, _zeros

comptime _M64 = SIMD[DType.float32, 64]
comptime NKIND = 7


@always_inline
def _sg_mma(a: _M64, b: _M64, c: _M64) -> _M64:
    return external_call[
        "air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32",
        _M64,
    ](a, b, c)


@always_inline
def _mix(x_in: UInt32) -> UInt32:
    var x = x_in
    x ^= x >> 16
    x *= UInt32(0x7FEB352D)
    x ^= x >> 15
    x *= UInt32(0x846CA68B)
    x ^= x >> 16
    return x


@always_inline
def _val(kind: Int, trial: Int, which: Int, r: Int, c: Int) -> Float32:
    """Deterministic operand word. kind 0: exponents in [2^-12, 2^12);
    kind 1: exponents in [2^-2, 2^2) (heavy cancellation among equals);
    kind 2: kind 0 with a quarter of the words +0.0 or -0.0;
    kind 3: exponents in [2^-40, 2^40) (wide spread, big absorptions)."""
    var h = _mix(
        UInt32(trial) * UInt32(2654435761)
        ^ _mix(UInt32(which * 64 + r * 8 + c + 1) * UInt32(0x9E3779B9) ^ UInt32(kind))
    )
    var sign = h & UInt32(0x80000000)
    var mant = h & UInt32(0x007FFFFF)
    var h2 = _mix(h ^ UInt32(0x5BD1E995))
    var span = 24
    if kind == 1:
        span = 4
    elif kind == 3:
        span = 80
    var e = Int(h2 % UInt32(span)) - span // 2 + 127
    if kind == 2 and (h2 >> 24) % 4 == 0:
        return bitcast[DType.float32](sign)
    if kind == 4:
        # kind 0 with one word in 64 an infinity or a NaN (payload hashed).
        var sel = (h2 >> 20) % 128
        if sel == 0:
            return bitcast[DType.float32](sign | UInt32(0x7F800000))
        if sel == 1:
            return bitcast[DType.float32](sign | UInt32(0x7F800000) | (mant | UInt32(1)))
    if kind == 5:
        # exponents in [2^100, 2^127]: products and sums overflow to infinity.
        e = Int(h2 % UInt32(28)) + 100 + 127
        return bitcast[DType.float32](sign | (UInt32(e) << 23) | mant)
    if kind == 6:
        # exponents in [2^-80, 2^-50): products near and below 2^-126, so
        # step results are subnormal (Apple's FMA flushes before rounding).
        e = Int(h2 % UInt32(30)) - 80 + 127
        return bitcast[DType.float32](sign | (UInt32(e) << 23) | mant)
    return bitcast[DType.float32](sign | (UInt32(e) << 23) | mant)


def probe_kernel(res: UnsafePointer[UInt32, MutAnyOrigin], kind_in: Int32):
    var kind = Int(kind_in)
    var trial = Int(block_idx.x)
    var lane = Int(thread_idx.x)
    var qd = lane // 4
    var frow = (qd & 4) + ((lane // 2) % 4)
    var fcol = (qd & 2) * 2 + (lane % 2) * 2
    var a = _M64(0)
    var b = _M64(0)
    var c = _M64(0)
    comptime for e in range(2):
        a[e] = _val(kind, trial, 0, frow, fcol + e)
        b[e] = _val(kind, trial, 1, frow, fcol + e)
        c[e] = _val(kind, trial, 2, frow, fcol + e)
    var d = _sg_mma(a, b, c)
    var flags = UInt32(0)
    var opaque0 = UInt32(kind_in) >> UInt32(30)
    comptime for e in range(2):
        var i = frow
        var j = fcol + e
        var cij = _val(kind, trial, 2, i, j)
        var asc = cij
        var desc = cij
        var unf = cij
        var z = Float32(0.0)
        comptime for p in range(8):
            asc = identical_mul_add(_val(kind, trial, 0, i, p), _val(kind, trial, 1, p, j), asc)
            desc = fma(_val(kind, trial, 0, i, 7 - p), _val(kind, trial, 1, 7 - p, j), desc)
            # The product rounded on its own: a runtime-opaque integer
            # round trip (zero for every kind) keeps the compiler from
            # contracting the multiply into the add.
            var prod = _val(kind, trial, 0, i, p) * _val(kind, trial, 1, p, j)
            prod = bitcast[DType.float32](bitcast[DType.uint32](prod) ^ opaque0)
            unf = unf + prod
            z = fma(_val(kind, trial, 0, i, p), _val(kind, trial, 1, p, j), z)
        var got = bitcast[DType.uint32](d[e])
        if got != bitcast[DType.uint32](asc):
            flags |= UInt32(1) << UInt32(4 * e + 0)
        if got != bitcast[DType.uint32](desc):
            flags |= UInt32(1) << UInt32(4 * e + 1)
        if got != bitcast[DType.uint32](unf):
            flags |= UInt32(1) << UInt32(4 * e + 2)
        if got != bitcast[DType.uint32](cij + z):
            flags |= UInt32(1) << UInt32(4 * e + 3)
    res[trial * 32 + lane] = flags


def main() raises:
    comptime assert has_apple_gpu_accelerator(), "Apple GPU probe"
    var trials = 1 << 16
    var ctx = DeviceContext()
    for kind in range(NKIND):
        var ob = ctx.enqueue_create_buffer[DType.uint32](trials * 32)
        ctx.enqueue_function[probe_kernel](
            ob.unsafe_ptr(), Int32(kind), grid_dim=(trials, 1, 1), block_dim=(32, 1, 1)
        )
        ctx.synchronize()
        var h = List[UInt32](length=trials * 32, fill=0)
        ctx.enqueue_copy(h.unsafe_ptr(), ob)
        ctx.synchronize()
        var cnt = List[Int](length=4, fill=0)
        for t in range(trials * 32):
            for e in range(2):
                for bit in range(4):
                    if (h[t] >> UInt32(4 * e + bit)) & 1 == 1:
                        cnt[bit] += 1
        print(
            "SG_PROBE kind=" + String(kind) + " cells=" + String(trials * 64)
            + " vs_asc_fma=" + String(cnt[0]) + " vs_desc_fma=" + String(cnt[1])
            + " vs_asc_unfused=" + String(cnt[2]) + " vs_c_plus_chain=" + String(cnt[3])
        )
