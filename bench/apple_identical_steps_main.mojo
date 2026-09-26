# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Repeated IDENTICAL training steps, timed and hashed (lane apple-identical-steps).

Every arm is one per-iteration product a classical fit runs many times:

  gemv.N x D        `gemv_n`, logistic regression's linearFwd (QN, C == 1)
  xty.N x D         `xty_kernel`, logistic regression's linearBwd X^T dZ
  nt.N x C x D      `gemm_nt`, softmax linearFwd / k-means++ candidate cost
  gram.D x D x N    `gemm_tn`, the PCA/OLS Gram

Each arm prints `STEP <arm> <median ms> <min ms> hash=<fnv64 of the output
bits>`. The hash is the identity witness: a restructured kernel is admitted
only when every hash is unchanged. Run it on a build of the old kernel and of
the new one and diff the hash column.

    mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . \\
        bench/apple_identical_steps_main.mojo -o build/steps
    python3 tools/mac_slot.py --timeout 600 metal build/steps
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory import bitcast
from std.os import getenv
from std.time import perf_counter_ns

from core.gemm import gemm_nt, gemm_tn, gemv_n
from core.column_stats import STATS_TPB, xty_kernel
from checks.numerics import numeric_mode_name
from glm.impl.qn.glm_base import mean_kernel, sum_terms_kernel
from glm.impl.qn.glm_softmax import xtdz_multi_kernel
from core.xtdz_coalesced import (
    xtdz_coalesced,
    xtdz_coalesced_applies,
    xtdz_coalesced_workspace_floats,
)


def _mix(i: Int, salt: Int) -> UInt64:
    var z = UInt64(i + 1) * 0x9E3779B97F4A7C15 + UInt64(salt + 1) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _fill(ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int, salt: Int) raises:
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    var p = h.unsafe_ptr()
    for i in range(n):
        var v = Float32(Int(_mix(i, salt) % 2000001) - 1000000) * Float32(1.0e-6)
        # A sprinkle of tiny values so the flush and the Apple rtf repair are
        # reached, not just the normal path.
        if _mix(i, salt + 7) % 997 == 0:
            v = v * Float32(1.0e-36)
        p.unsafe_store(i, v)
    ctx.enqueue_copy(dst_buf=buf, src_ptr=h.unsafe_ptr())
    ctx.synchronize()


def _hash(ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int) raises -> UInt64:
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    var acc = UInt64(0xCBF29CE484222325)
    for i in range(n):
        var w = UInt64(bitcast[DType.uint32](h.unsafe_ptr().unsafe_load(i)))
        acc = (acc ^ w) * 0x100000001B3
    return acc


def _hex(v: UInt64) -> String:
    var digits = String("0123456789abcdef")
    var s = String("")
    for i in range(16):
        var nib = Int((v >> UInt64(60 - 4 * i)) & 15)
        s += digits[byte=nib]
    return s


def _reps() raises -> Int:
    var s = String(getenv("MOJOLEARN_STEPS_REPS"))
    return 7 if s == "" else Int(s)


def _batch() raises -> Int:
    """Calls enqueued per timed sample, one synchronize after them all: a
    Metal synchronize with pending work costs ~4 ms on the M4, which floors
    a one-call sample and hides the kernel."""
    var s = String(getenv("MOJOLEARN_STEPS_BATCH"))
    return 10 if s == "" else Int(s)


def _report(name: String, mut times: List[Float64], h: UInt64):
    # insertion sort, tiny list
    for i in range(1, len(times)):
        var j = i
        while j > 0 and times[j - 1] > times[j]:
            var t = times[j]
            times[j] = times[j - 1]
            times[j - 1] = t
            j -= 1
    print("STEP", numeric_mode_name(), name, times[len(times) // 2], times[0], "hash=" + _hex(h))


def _want(arm: String) -> Bool:
    var only = String(getenv("MOJOLEARN_STEPS_ONLY"))
    if only == "":
        return True
    return arm.startswith(only) or only in arm


def gemv_arm(ctx: DeviceContext, n: Int, d: Int) raises:
    var name = "gemv." + String(n) + "x" + String(d)
    if not _want(name):
        return
    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var w = ctx.enqueue_create_buffer[DType.float32](d)
    var z = ctx.enqueue_create_buffer[DType.float32](n)
    _fill(ctx, x, n * d, 1)
    _fill(ctx, w, d, 2)
    gemv_n(ctx, z, x, w, n, d)
    ctx.synchronize()
    var times = List[Float64]()
    for _ in range(_reps()):
        var t0 = perf_counter_ns()
        for _b in range(_batch()):
            gemv_n(ctx, z, x, w, n, d)
        ctx.synchronize()
        times.append(Float64(perf_counter_ns() - t0) / 1.0e6 / Float64(_batch()))
    _report(name, times, _hash(ctx, z, n))
    _ = x^
    _ = w^
    _ = z^


def xty_arm(ctx: DeviceContext, n: Int, d: Int) raises:
    var name = "xty." + String(n) + "x" + String(d)
    if not _want(name):
        return
    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var y = ctx.enqueue_create_buffer[DType.float32](n)
    var g = ctx.enqueue_create_buffer[DType.float32](d)
    _fill(ctx, x, n * d, 3)
    _fill(ctx, y, n, 4)
    ctx.enqueue_function[xty_kernel](
        g.unsafe_ptr(), x.unsafe_ptr(), y.unsafe_ptr(), Int32(n), Int32(d),
        grid_dim=(d, 1, 1), block_dim=(STATS_TPB, 1, 1),
    )
    ctx.synchronize()
    var times = List[Float64]()
    for _ in range(_reps()):
        var t0 = perf_counter_ns()
        for _b in range(_batch()):
            ctx.enqueue_function[xty_kernel](
                g.unsafe_ptr(), x.unsafe_ptr(), y.unsafe_ptr(), Int32(n), Int32(d),
                grid_dim=(d, 1, 1), block_dim=(STATS_TPB, 1, 1),
            )
        ctx.synchronize()
        times.append(Float64(perf_counter_ns() - t0) / 1.0e6 / Float64(_batch()))
    _report(name, times, _hash(ctx, g, d))
    _ = x^
    _ = y^
    _ = g^


def reduce_arm(ctx: DeviceContext, n: Int, which: Int) raises:
    var name = ("sum_terms." if which == 0 else "mean.") + String(n)
    if not _want(name):
        return
    var v = ctx.enqueue_create_buffer[DType.float32](n)
    var o = ctx.enqueue_create_buffer[DType.float32](1)
    _fill(ctx, v, n, 9)
    var times = List[Float64]()
    for rep in range(_reps() + 1):
        var t0 = perf_counter_ns()
        for _b in range(_batch()):
            if which == 0:
                ctx.enqueue_function[sum_terms_kernel](
                    o.unsafe_ptr(), v.unsafe_ptr(), Int32(n),
                    grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
                )
            else:
                ctx.enqueue_function[mean_kernel](
                    o.unsafe_ptr(), v.unsafe_ptr(), Int32(n),
                    grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
                )
        ctx.synchronize()
        if rep > 0:
            times.append(Float64(perf_counter_ns() - t0) / 1.0e6 / Float64(_batch()))
    _report(name, times, _hash(ctx, o, 1))
    _ = v^
    _ = o^


def xtdz_arm(ctx: DeviceContext, n: Int, d: Int, c: Int) raises:
    var name = "xtdz." + String(n) + "x" + String(d) + "x" + String(c)
    if not _want(name):
        return
    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var dz = ctx.enqueue_create_buffer[DType.float32](n * c)
    var g = ctx.enqueue_create_buffer[DType.float32](c * d)
    _fill(ctx, x, n * d, 10)
    _fill(ctx, dz, n * c, 11)
    var times = List[Float64]()
    for rep in range(_reps() + 1):
        var t0 = perf_counter_ns()
        for _b in range(_batch()):
            ctx.enqueue_function[xtdz_multi_kernel](
                g.unsafe_ptr(), x.unsafe_ptr(), dz.unsafe_ptr(),
                Int32(n), Int32(d), Int32(c),
                grid_dim=(c * d, 1, 1), block_dim=(STATS_TPB, 1, 1),
            )
        ctx.synchronize()
        if rep > 0:
            times.append(Float64(perf_counter_ns() - t0) / 1.0e6 / Float64(_batch()))
    _report(name, times, _hash(ctx, g, c * d))
    _ = x^
    _ = dz^
    _ = g^


def xtdz_co_arm(ctx: DeviceContext, n: Int, d: Int, c: Int, salt_x: Int, salt_z: Int) raises:
    """The coalesced two-pass `X^T dZ`; its hash must equal the matching
    `xty.` (c == 1, salts 3/4) or `xtdz.` (salts 10/11) arm."""
    var name = "co." + ("xty." if c == 1 else "xtdz.") + String(n) + "x" + String(d) + (
        "" if c == 1 else "x" + String(c)
    )
    if not _want(name):
        return
    if not xtdz_coalesced_applies(d, c):
        print("STEP", name, "not applicable in this build")
        return
    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var dz = ctx.enqueue_create_buffer[DType.float32](n * c)
    var g = ctx.enqueue_create_buffer[DType.float32](c * d)
    var ws = ctx.enqueue_create_buffer[DType.float32](xtdz_coalesced_workspace_floats(d, c))
    _fill(ctx, x, n * d, salt_x)
    _fill(ctx, dz, n * c, salt_z)
    var times = List[Float64]()
    for rep in range(_reps() + 1):
        var t0 = perf_counter_ns()
        for _b in range(_batch()):
            xtdz_coalesced(ctx, g, x, dz, ws, n, d, c)
        ctx.synchronize()
        if rep > 0:
            times.append(Float64(perf_counter_ns() - t0) / 1.0e6 / Float64(_batch()))
    _report(name, times, _hash(ctx, g, c * d))
    _ = x^
    _ = dz^
    _ = g^
    _ = ws^


def nt_arm(ctx: DeviceContext, m: Int, n: Int, k: Int) raises:
    var name = "nt." + String(m) + "x" + String(n) + "x" + String(k)
    if not _want(name):
        return
    var x = ctx.enqueue_create_buffer[DType.float32](m * k)
    var y = ctx.enqueue_create_buffer[DType.float32](n * k)
    var z = ctx.enqueue_create_buffer[DType.float32](m * n)
    _fill(ctx, x, m * k, 5)
    _fill(ctx, y, n * k, 6)
    gemm_nt(ctx, z, x, y, m, n, k)
    ctx.synchronize()
    var times = List[Float64]()
    for _ in range(_reps()):
        var t0 = perf_counter_ns()
        for _b in range(_batch()):
            gemm_nt(ctx, z, x, y, m, n, k)
        ctx.synchronize()
        times.append(Float64(perf_counter_ns() - t0) / 1.0e6 / Float64(_batch()))
    _report(name, times, _hash(ctx, z, m * n))
    _ = x^
    _ = y^
    _ = z^


def gram_arm(ctx: DeviceContext, d: Int, n: Int) raises:
    var name = "gram." + String(d) + "x" + String(d) + "x" + String(n)
    if not _want(name):
        return
    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var z = ctx.enqueue_create_buffer[DType.float32](d * d)
    var xt = ctx.enqueue_create_buffer[DType.float32](n * d)
    var xt2 = ctx.enqueue_create_buffer[DType.float32](n * d)
    _fill(ctx, x, n * d, 8)
    gemm_tn(ctx, z, x, xt, xt2, d, d, n)
    ctx.synchronize()
    var times = List[Float64]()
    for _ in range(_reps()):
        var t0 = perf_counter_ns()
        for _b in range(_batch()):
            gemm_tn(ctx, z, x, xt, xt2, d, d, n)
        ctx.synchronize()
        times.append(Float64(perf_counter_ns() - t0) / 1.0e6 / Float64(_batch()))
    _report(name, times, _hash(ctx, z, d * d))
    _ = x^
    _ = z^
    _ = xt^
    _ = xt2^


def main() raises:
    with DeviceContext() as ctx:
        print("STEPS mode", numeric_mode_name(), "device", ctx.name())
        # logistic regression per evaluation (taxi 11 columns, a wider 64)
        gemv_arm(ctx, 4_000_000, 11)
        gemv_arm(ctx, 1_000_000, 64)
        gemv_arm(ctx, 1_000_000, 220)
        gemv_arm(ctx, 1000, 1000)
        xty_arm(ctx, 4_000_000, 11)
        xty_arm(ctx, 1_000_000, 64)
        xty_arm(ctx, 1_000_000, 220)
        xtdz_co_arm(ctx, 4_000_000, 11, 1, 3, 4)
        xtdz_co_arm(ctx, 1_000_000, 64, 1, 3, 4)
        xtdz_co_arm(ctx, 1_000_000, 220, 1, 3, 4)
        xtdz_co_arm(ctx, 1_000_000, 11, 4, 10, 11)
        xtdz_co_arm(ctx, 500_000, 54, 7, 10, 11)
        xtdz_arm(ctx, 1_000_000, 11, 4)
        xtdz_arm(ctx, 500_000, 54, 7)
        reduce_arm(ctx, 4_000_000, 0)
        reduce_arm(ctx, 4_000_000, 1)
        # softmax linearFwd, k-means++ candidates, the k-NN-sized tile
        nt_arm(ctx, 1_000_000, 4, 11)
        nt_arm(ctx, 1_000_000, 10, 64)
        nt_arm(ctx, 65536, 64, 64)
        nt_arm(ctx, 4096, 512, 256)
        # Gram (PCA / OLS)
        gram_arm(ctx, 32, 1_000_000)
        gram_arm(ctx, 11, 4_000_000)
