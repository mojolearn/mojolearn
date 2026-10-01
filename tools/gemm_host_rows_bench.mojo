# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""How the host GEMM scales on this box: `gemm_host_rows` at the byte LM host
step's projection shapes, one thread against the thread policy's count
(lane neural-pass9).

    MOJOLEARN_CPU_THREADS=1 pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . tools/gemm_host_rows_bench.mojo
    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . tools/gemm_host_rows_bench.mojo

Prints one line per shape: the best of three walls, GFLOP/s, and a digest of
the output (the same at both settings, since the task count moves no bit).
On the 64-core AMD host PR #14's block timing put the projections at 3 to 4
GFLOP/s a core; this says whether that is the kernel or the call."""
from std.time import perf_counter_ns

from gemm.host.gemm_host_rows import gemm_host_rows
from gemm.host.identical_gemm import OP_NN, OP_NT, OP_TN


def fill(n: Int, seed: Int) -> List[Float32]:
    var out = List[Float32](capacity=n)
    var st = UInt64(seed)
    for _ in range(n):
        st = st * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        out.append((Float32(Int((st >> 33) & UInt64(0xFFFFFF))) / Float32(16777216.0) - Float32(0.5)) * Float32(0.1))
    return out^


def bench(name: StaticString, m: Int, n: Int, k: Int, op: Int) raises:
    var a = fill(m * k, 1)
    var b = fill(n * k, 2)
    var best = Int(1 << 62)
    var h = UInt64(0)
    for _ in range(3):
        var t0 = perf_counter_ns()
        var c = gemm_host_rows(a, b, op, m, n, k)
        var dt = perf_counter_ns() - t0
        if dt < best:
            best = dt
        h = UInt64(0)
        for i in range(0, len(c), 97):
            h = h * UInt64(31) + UInt64(Int(c[i] * Float32(1e6)) & 0xFFFF)
    var flops = Float64(2 * m * n * k)
    print("gemm", String(name), "m", m, "n", n, "k", k, "ms", Float64(best) / 1e6,
          "gflops", flops / Float64(best), "digest", h)


def main() raises:
    bench("qkv_proj", 512, 384, 384, OP_NT)
    bench("gate_up_proj", 512, 1024, 384, OP_NT)
    bench("down_proj", 512, 384, 1024, OP_NT)
    bench("dW_gate (k = tokens)", 1024, 384, 512, OP_TN)
    bench("head_scores", 512, 512, 64, OP_NT)
    bench("lm_head", 512, 8192, 384, OP_NT)
    bench("mamba3_in_proj", 2048, 1728, 384, OP_NT)
