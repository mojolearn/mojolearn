# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/amd-step-time (2026-09-24): the twelve GEMM calls of ONE T3 shard
(batch 4, length 2048: M = 8192 token rows, d_model 768, FF 2048, vocabulary
50,257), each run through the SHIPPED entry points the byte LM step calls
(`identical_gemm_into` forward, `identical_gemm_backward_a_into` and
`_b_into` backward, all at `OP_NT`), on three operand kinds, printing a hash
of every output and a median time.

Built twice, once as the branch builds (the DETECT seam and the launch bound
on AMD) and once with `-D MOJOLEARN_GEMM_NO_DETECT_SEAM=1
-D MOJOLEARN_GEMM_NO_LAUNCH_BOUND=1` (the shipped kernels), the two
runs' HASH columns must be identical line for line: that is the bit proof of
the seam at the production shapes, and the ms columns are its price.

    pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_TARGET_COLUMN=amd \\
        -I . bench/gemm_excp_ab_main.mojo -o <bin>
    <bin> [calls comma list] [kinds comma list] [rounds]

Operand kinds (filled on the device from a hash of the index, the same on
every build):
  ordinary  sign and 23 mantissa bits hashed, exponent in [2^-8, 2^1)
  tiny      exponent in [2^-66, 2^-60): products near 2^-126, so bare FMA
            chains produce and consume subnormals and the exact recompute runs
            (by default only on proj_* and down_fwd: the recompute is slow)
  mixed     ordinary, with one word in 64 scaled into [2^-110, 2^-100)
Lines: `EXCP_AB call=... kind=... m= n= k= hash=<16 hex> ms=<median> samples=...`.
"""
from std.gpu import block_idx, block_dim, thread_idx
from std.memory import bitcast
from std.os import getenv
from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext
from gemm.checks.gemm_identical import (
    GEMM_DETECT_SEAM,
    GEMM_LAUNCH_BOUND,
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
    gemm_shipped_dispatch_name,
)
from gemm.checks.gemm_backward import (
    identical_gemm_backward_a_into,
    identical_gemm_backward_b_into,
    identical_gemm_backward_workspace_max_floats,
)
from gemm.checks.gemm_oracle import OP_NT
from checks.kernel_matrix import TARGET_COLUMN, column_name


@always_inline
def _mix(x: UInt32) -> UInt32:
    var h = x * UInt32(0x9E3779B1)
    h = h ^ (h >> UInt32(15))
    h = h * UInt32(0x85EBCA77)
    h = h ^ (h >> UInt32(13))
    h = h * UInt32(0xC2B2AE3D)
    return h ^ (h >> UInt32(16))


def fill_kernel(dst: MutPointer[Float32, MutAnyOrigin], n: Int64, seed: UInt32, kind: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n):
        return
    var h = _mix(UInt32(i) ^ _mix(seed))
    var h2 = _mix(h ^ UInt32(0x5BD1E995))
    var sign = h & UInt32(0x80000000)
    var mant = h & UInt32(0x007FFFFF)
    var e = UInt32(119) + (h2 % UInt32(9))  # 2^-8 .. 2^0
    if kind == 1:
        e = UInt32(61) + (h2 % UInt32(6))  # 2^-66 .. 2^-61
    elif kind == 2:
        if (h2 >> UInt32(26)) == UInt32(0):
            e = UInt32(17) + (h2 % UInt32(10))  # 2^-110 .. 2^-101
    dst.unsafe_store(i, bitcast[DType.float32](sign | (e << UInt32(23)) | mant))


def _fill(ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int, seed: UInt32, kind: Int) raises:
    ctx.enqueue_function[fill_kernel](
        buf.unsafe_ptr(), Int64(n), seed, Int32(kind),
        grid_dim=((n + 255) // 256, 1, 1), block_dim=(256, 1, 1),
    )


def _hash(ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int) raises -> UInt64:
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_buf=host, src_buf=buf.create_sub_buffer[DType.float32](0, n))
    ctx.synchronize()
    var p = host.unsafe_ptr()
    var h = UInt64(0xCBF29CE484222325)
    for i in range(n):
        h = (h ^ UInt64(bitcast[DType.uint32](p[i]))) * UInt64(0x100000001B3)
    return h


def _hex64(h: UInt64) -> String:
    var digits: List[String] = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f"
    ]
    var s = String("")
    for k in range(16):
        var nib = Int((h >> UInt64(60 - 4 * k)) & UInt64(0xF))
        s += digits[nib]
    return s


def _run(ctx: DeviceContext, call: String, mut out: DeviceBuffer[DType.float32],
         mut x: DeviceBuffer[DType.float32], mut y: DeviceBuffer[DType.float32],
         mut ws: DeviceBuffer[DType.float32], m: Int, n: Int, k: Int, which: Int) raises:
    if which == 0:
        identical_gemm_into(ctx, out, x, y, ws, m, n, k, OP_NT)
    elif which == 1:
        identical_gemm_backward_a_into(ctx, out, x, y, ws, m, n, k, OP_NT)
    else:
        identical_gemm_backward_b_into(ctx, out, x, y, ws, m, n, k, OP_NT)
    ctx.synchronize()


def main() raises:
    var M = 8192
    var names: List[String] = [
        "proj_fwd", "proj_dA", "proj_dB",
        "gateup_fwd", "gateup_dA", "gateup_dB",
        "down_fwd", "down_dA", "down_dB",
        "head_fwd", "head_dA", "head_dB",
    ]
    # forward (m, n, k) of each kind at OP_NT
    var fm: List[Int] = [M, M, M, M, M, M, M, M, M, M, M, M]
    var fnn: List[Int] = [768, 768, 768, 2048, 2048, 2048, 768, 768, 768, 50257, 50257, 50257]
    var fk: List[Int] = [768, 768, 768, 768, 768, 768, 2048, 2048, 2048, 768, 768, 768]
    var calls_env = String(getenv("MOJOLEARN_EXCP_AB_CALLS"))
    var kinds_env = String(getenv("MOJOLEARN_EXCP_AB_KINDS"))
    var rounds_env = String(getenv("MOJOLEARN_EXCP_AB_ROUNDS"))
    var rounds = 3 if rounds_env == "" else Int(rounds_env)
    var kind_names: List[String] = ["ordinary", "tiny", "mixed"]
    print("EXCP_AB_HEADER column=" + column_name(TARGET_COLUMN) + " detect_seam=" + String(GEMM_DETECT_SEAM) + " launch_bound=" + String(GEMM_LAUNCH_BOUND)
          + " rounds=" + String(rounds))
    var ctx = DeviceContext()
    for ci in range(len(names)):
        var call = names[ci]
        if calls_env != "" and not ("," + calls_env + ",").__contains__("," + call + ","):
            continue
        var m = fm[ci]
        var n = fnn[ci]
        var k = fk[ci]
        var which = ci % 3
        # operands of each call, row-major: forward A[m,k], B[n,k] -> C[m,n];
        # dA takes dC[m,n] and B[n,k] -> dA[m,k]; dB takes dC[m,n] and A[m,k] -> dB[n,k]
        var nx = m * k
        var ny = n * k
        var nout = m * n
        var ws_n = identical_gemm_workspace_max_floats(m, n, k)
        if which == 0:
            print("EXCP_AB_DISPATCH call=" + call + " " + gemm_shipped_dispatch_name(m, n, k))
        else:
            nx = m * n
            ny = n * k if which == 1 else m * k
            nout = m * k if which == 1 else n * k
            ws_n = identical_gemm_backward_workspace_max_floats(OP_NT, m, n, k, False)
        var x = ctx.enqueue_create_buffer[DType.float32](nx)
        var y = ctx.enqueue_create_buffer[DType.float32](ny)
        var out = ctx.enqueue_create_buffer[DType.float32](nout)
        var ws = ctx.enqueue_create_buffer[DType.float32](ws_n if ws_n > 0 else 1)
        for kd in range(3):
            if kinds_env != "" and not ("," + kinds_env + ",").__contains__("," + kind_names[kd] + ","):
                continue
            # The tiny kind makes (nearly) every wave take the exact recompute,
            # which is slow by design; by default it runs on the k = 768
            # projection calls and down_fwd only.
            if kd == 1 and kinds_env == "" and not (call.startswith("proj_") or call == "down_fwd"):
                continue
            _fill(ctx, x, nx, UInt32(1000 + 7 * ci), kd)
            _fill(ctx, y, ny, UInt32(2000 + 7 * ci), kd)
            ctx.synchronize()
            _run(ctx, call, out, x, y, ws, m, n, k, which)  # warmup and the hashed run
            var h = _hash(ctx, out, nout)
            var samples = List[Float64]()
            for _r in range(rounds):
                var t0 = perf_counter_ns()
                _run(ctx, call, out, x, y, ws, m, n, k, which)
                samples.append(Float64(perf_counter_ns() - t0) / 1.0e6)
            var h2 = _hash(ctx, out, nout)
            # median by insertion sort
            for i in range(1, len(samples)):
                var v = samples[i]
                var j = i - 1
                while j >= 0 and samples[j] > v:
                    samples[j + 1] = samples[j]
                    j -= 1
                samples[j + 1] = v
            var s = String("")
            for i in range(len(samples)):
                s += ("" if i == 0 else ",") + String(samples[i])
            print("EXCP_AB call=" + call + " kind=" + kind_names[kd] + " m=" + String(m) + " n=" + String(n)
                  + " k=" + String(k) + " hash=" + _hex64(h) + " rehash_equal=" + String(h == h2)
                  + " ms=" + String(samples[len(samples) // 2]) + " samples=" + s)
        _ = x^
        _ = y^
        _ = out^
        _ = ws^
    print("EXCP_AB_DONE")
