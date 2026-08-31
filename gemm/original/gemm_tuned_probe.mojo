# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Does the TUNED kernel produce the PINNED kernel's exact bits?

A faster GEMM that moves one bit is worthless here, so this is the only
question worth asking first. Compares FNV-1a64 over the raw output bits of
`identical_gemm_into` and `tuned_gemm_into` at the llama8b shapes, with both
outputs POISONED first and the poison counted, so a kernel that never wrote
cannot agree by accident.
"""
from std.memory import bitcast
from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext

from bench.gemm_shapes import (
    GEMM_SHAPE_COUNT, gemm_shape_k, gemm_shape_m, gemm_shape_n,
    gemm_shape_name, gemm_shape_op,
)
from bench.gemm_shapes import OP_NN as TBL_NN
from bench.gemm_shapes import OP_NT as TBL_NT
from bench.gemm_shapes import OP_TN as TBL_TN
from gemm.original.gemm_identical import (
    identical_gemm_into, identical_gemm_workspace_max_floats,
)
from gemm.original.gemm_oracle import OP_NN, OP_NT, OP_TN
from gemm.original.gemm_identical_tuned import (
    tuned_gemm_banner, tuned_gemm_into, tuned_gemm_workspace_max_floats,
)

comptime POISON = Float32(-1.0e37)


def _oop(t: Int) -> Int:
    if t == TBL_NT: return OP_NT
    if t == TBL_TN: return OP_TN
    return OP_NN


def _mix(i: Int, salt: Int) -> UInt64:
    var z = UInt64(i) + UInt64(salt) + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def _val(i: Int, salt: Int) -> Float32:
    var h = _mix(i, salt)
    var b = UInt32(0x3F800000) | UInt32(h & UInt64(0x7FFFFF))
    if (h >> 23) & UInt64(1) == UInt64(1):
        b = b | UInt32(0x80000000)
    return bitcast[DType.float32](b)


def _fill(ctx: DeviceContext, mut d: DeviceBuffer[DType.float32], n: Int, salt: Int) raises:
    if n < 1: return
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    for i in range(n): h.unsafe_ptr().unsafe_store(i, _val(i, salt))
    ctx.enqueue_copy(dst_buf=d, src_ptr=h.unsafe_ptr()); ctx.synchronize(); _ = h


def _poison(ctx: DeviceContext, mut d: DeviceBuffer[DType.float32], n: Int) raises:
    var h = ctx.enqueue_create_host_buffer[DType.float32](n if n > 0 else 1)
    ctx.synchronize()
    for i in range(n): h.unsafe_ptr().unsafe_store(i, POISON)
    ctx.enqueue_copy(dst_buf=d, src_ptr=h.unsafe_ptr()); ctx.synchronize(); _ = h


def _digest(ctx: DeviceContext, mut d: DeviceBuffer[DType.float32], n: Int) raises -> Tuple[UInt64, Int]:
    var h = ctx.enqueue_create_host_buffer[DType.float32](n if n > 0 else 1)
    ctx.synchronize(); ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=d); ctx.synchronize()
    var acc = UInt64(0xCBF29CE484222325); var left = 0
    for i in range(n):
        var v = h.unsafe_ptr().unsafe_load(i)
        if bitcast[DType.uint32](v) == bitcast[DType.uint32](POISON): left += 1
        var b = UInt64(bitcast[DType.uint32](v))
        for s in range(4):
            acc = (acc ^ ((b >> UInt64(8 * s)) & UInt64(0xFF))) * UInt64(0x100000001B3)
    _ = h
    return (acc, left)


def main() raises:
    print("== tuned vs pinned, BITS ==")
    print("   " + tuned_gemm_banner())
    var same = 0; var moved = 0; var refused = 0
    for i in range(GEMM_SHAPE_COUNT):
        var k = gemm_shape_k(i)
        var op = _oop(gemm_shape_op(i))
        var m = gemm_shape_m(i); var n = gemm_shape_n(i)
        while m > 1 and m * n * k > 50_000_000_000: m = (m + 1) // 2
        while n > 1 and m * n * k > 50_000_000_000: n = (n + 1) // 2
        var mn = m * n
        var na = k * m if op == OP_TN else m * k
        var nb = n * k if op == OP_NT else k * n
        var w1 = identical_gemm_workspace_max_floats(m, n, k)
        var w2 = tuned_gemm_workspace_max_floats(m, n, k)
        var nw = w1 if w1 > w2 else w2
        var ctx = DeviceContext()
        var da = ctx.enqueue_create_buffer[DType.float32](na if na > 0 else 1)
        var db = ctx.enqueue_create_buffer[DType.float32](nb if nb > 0 else 1)
        var dc = ctx.enqueue_create_buffer[DType.float32](mn if mn > 0 else 1)
        var dw = ctx.enqueue_create_buffer[DType.float32](nw if nw > 0 else 1)
        ctx.synchronize()
        _fill(ctx, da, na, 11 + i); _fill(ctx, db, nb, 22 + i)
        _poison(ctx, dc, mn)
        identical_gemm_into(ctx, dc, da, db, dw, m, n, k, op); ctx.synchronize()
        var dp = _digest(ctx, dc, mn)
        _poison(ctx, dc, mn)
        tuned_gemm_into(ctx, dc, da, db, dw, m, n, k, op); ctx.synchronize()
        var dt = _digest(ctx, dc, mn)
        # Three timed pairs, ALTERNATING CALL BY CALL in one binary, after the
        # warm-ups above. Same discipline as gemm_unpinned_price.mojo: two
        # functions in one process, so the governor drift cancels at the arm
        # level rather than being averaged over rounds.
        var ns_p = 0
        var ns_t = 0
        for _ in range(3):
            var t0 = perf_counter_ns()
            identical_gemm_into(ctx, dc, da, db, dw, m, n, k, op); ctx.synchronize()
            ns_p += perf_counter_ns() - t0
            var t1 = perf_counter_ns()
            tuned_gemm_into(ctx, dc, da, db, dw, m, n, k, op); ctx.synchronize()
            ns_t += perf_counter_ns() - t1
        var ms_p = Float64(ns_p) / 3.0e6
        var ms_t = Float64(ns_t) / 3.0e6
        var nm = gemm_shape_name(i)
        if dp[1] != 0 or dt[1] != 0:
            print("   REFUSED " + nm + ": poison left pinned=" + String(dp[1]) + " tuned=" + String(dt[1]))
            refused += 1
        elif dp[0] == dt[0]:
            print("   OK  " + nm + "  m=" + String(m) + " n=" + String(n) + " k=" + String(k)
                  + "  BITS MATCH  pinned=" + String(ms_p) + "ms tuned=" + String(ms_t)
                  + "ms  speedup=" + String(ms_p / ms_t) + "x")
            same += 1
        else:
            print("   MOVED   " + nm + "  m=" + String(m) + " n=" + String(n) + " k=" + String(k)
                  + "  pinned=" + hex(dp[0]) + " tuned=" + hex(dt[0]))
            moved += 1
        _ = da^; _ = db^; _ = dc^; _ = dw^
    print()
    print("   " + String(same) + " match, " + String(moved) + " MOVED, " + String(refused) + " refused.")
    if moved != 0:
        raise Error("tuned_probe: the tuned kernel MOVED BITS on " + String(moved)
                    + " shapes. A faster kernel that changes the answer is not this profile.")
