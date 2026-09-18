# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Price of the Apple zero-result repair on the ONE-CELL-PER-THREAD pinned
GEMM kernels of core/gemm.mojo (lane `lane/apple-seam-repair`, 2026-09-18).

The classical estimators (k-means, PCA, GLM, lstsq, ridge, kNN brute force,
KDE, ...) reach `gemm_nt`, `gemv_n` and `gemm_nt_gram`, whose kernels now
take `rtf_mul_add` (the inline per-step repair on Apple, the unchanged
spelling elsewhere). Build twice in one directory, with and without
`-D MOJOLEARN_NO_ZERO_FMA_REPAIR=1`, and alternate the binaries. Prints one
`PINNED <call> median_ms=` line per call over `MOJOLEARN_RTF_PRICE_ROUNDS`
(default 7) host-synchronized rounds after two warmups, plus an FNV-1a hash
of each output so the arms can be checked for equal bits.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory import bitcast
from std.os import getenv
from std.time import perf_counter_ns
from checks.rtf_seam import RTF_REPAIR
from core.gemm import gemm_nt, gemm_nt_gram, gemv_n


def _fill(ctx: DeviceContext, n: Int, salt: UInt64) raises -> DeviceBuffer[DType.float32]:
    var b = ctx.enqueue_create_buffer[DType.float32](n)
    with b.map_to_host() as h:
        var s = salt * UInt64(0x9E3779B97F4A7C15) + UInt64(1)
        for i in range(n):
            s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            var u = Float32(Int((s >> UInt64(40)) & UInt64(0xFFFFFF))) / Float32(16777216.0)
            h[i] = u * Float32(2.0) - Float32(1.0)
    return b^


def _fnv(ctx: DeviceContext, b: DeviceBuffer[DType.float32], n: Int) raises -> String:
    var h = UInt64(0xCBF29CE484222325)
    with b.map_to_host() as v:
        for i in range(n):
            var w = bitcast[DType.uint32](v[i])
            for k in range(4):
                h = (h ^ UInt64((w >> UInt32(8 * k)) & UInt32(0xFF))) * UInt64(0x100000001B3)
    return hex(h)


def _median(mut xs: List[Float64]) -> Float64:
    sort(xs)
    return xs[len(xs) // 2]


def main() raises:
    var rounds = 7
    var rs = String(getenv("MOJOLEARN_RTF_PRICE_ROUNDS"))
    if rs != "":
        rounds = Int(rs)
    print("PINNED rtf_repair=" + String(RTF_REPAIR) + " rounds=" + String(rounds))
    var ctx = DeviceContext()
    # gemm_nt: z[m x n] = x[m x k] . y[n x k]^T
    var m = 1024
    var n = 1024
    var k = 512
    var x = _fill(ctx, m * k, 1)
    var y = _fill(ctx, n * k, 2)
    var z = ctx.enqueue_create_buffer[DType.float32](m * n)
    # gemv_n: 1M rows x 64
    var gm = 1 << 20
    var gk = 64
    var gx = _fill(ctx, gm * gk, 3)
    var gy = _fill(ctx, gk, 4)
    var gz = ctx.enqueue_create_buffer[DType.float32](gm)
    # gemm_nt_gram: 512 x 512 over 4096
    var qm = 512
    var qk = 4096
    var qx = _fill(ctx, qm * qk, 5)
    var qz = ctx.enqueue_create_buffer[DType.float32](qm * qm)
    ctx.synchronize()
    var names: List[String] = ["gemm_nt_1024x1024x512", "gemv_n_1Mx64", "gram_512x4096"]
    for c in range(3):
        var ts = List[Float64]()
        for r in range(rounds + 2):
            var t0 = perf_counter_ns()
            if c == 0:
                gemm_nt(ctx, z, x, y, m, n, k)
            elif c == 1:
                gemv_n(ctx, gz, gx, gy, gm, gk)
            else:
                gemm_nt_gram(ctx, qz, qx, qm, qm, qk)
            ctx.synchronize()
            var t1 = perf_counter_ns()
            if r >= 2:
                ts.append(Float64(t1 - t0) / 1.0e6)
        var h: String
        if c == 0:
            h = _fnv(ctx, z, m * n)
        elif c == 1:
            h = _fnv(ctx, gz, gm)
        else:
            h = _fnv(ctx, qz, qm * qm)
        print("PINNED " + names[c] + " median_ms=" + String(_median(ts)) + " out_fnv=" + h)
    _ = x
    _ = y
    _ = gx
    _ = gy
    _ = qx
