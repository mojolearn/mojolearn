# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-identical-neural (2026-09-26): the speed ceiling of an fp32
GEMM on the Apple simdgroup matrix path at the T3 shard shapes, with none
of the IDENTICAL machinery (no leaves, no fold, no admission). C = A B^T,
A [m, k] and B [n, k] row-major (the byte LM's OP_NT). Evidence only.

Block: SGM x SGN simdgroups, each owning FM x FN 8x8 fragments; the block
tile is (8 FM SGM) x (8 FN SGN). A is staged transposed in shared memory
(k-major) and B as stored, KB deep, so both fragments come from the same
transposed 8x8 load.

    pixi run mojo build -I . bench/apple_mma_gemm_ceiling_main.mojo -o <bin>
    python3 tools/mac_slot.py metal -- <bin>
"""

from std.ffi import external_call
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from std.sys.info import _accelerator_arch
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

comptime _M64 = SIMD[DType.float32, 64]
comptime _V2 = SIMD[DType.int64, 2]


@always_inline
def _sg_load_t(
    p: UnsafePointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED],
    stride: Int,
) -> _M64:
    """M[r][c] = p[c * stride + r] (see fast_mma_knn._sg_load_t)."""
    comptime arch = _accelerator_arch()
    comptime if "metal:1" in arch or "metal:2" in arch or "metal:3" in arch:
        return external_call["air.simdgroup_matrix_8x8_load.v64f32.p3f32", _M64](
            p, Int64(stride), _V2(0, 0), True
        )
    else:
        return external_call["air.simdgroup_matrix_8x8_load.v64f32.p3f32", _M64](
            p, _V2(Int64(stride), 8), _V2(Int64(stride), 1), _V2(0, 0)
        )


@always_inline
def _sg_mma(a: _M64, b: _M64, c: _M64) -> _M64:
    return external_call[
        "air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32",
        _M64,
    ](a, b, c)


def mma_gemm_kernel[SGM: Int, SGN: Int, FM: Int, FN: Int, KB: Int](
    c: UnsafePointer[Float32, MutAnyOrigin],
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    m: Int32, n: Int32, k: Int32,
):
    comptime NT = SGM * SGN * 32
    comptime BM = 8 * FM * SGM
    comptime BN = 8 * FN * SGN
    comptime AST = BM + 4
    comptime BST = KB + 4
    var mm = Int(m)
    var nn = Int(n)
    var kk = Int(k)
    var tid = Int(thread_idx.x)
    var sg = tid // 32
    var lane = tid % 32
    var sgm = sg // SGN
    var sgn = sg % SGN
    var nbn = (nn + BN - 1) // BN
    var m0 = (Int(block_idx.x) // nbn) * BM
    var n0 = (Int(block_idx.x) % nbn) * BN
    var at = stack_allocation[KB * AST, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var bt = stack_allocation[BN * BST, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var acc = InlineArray[_M64, FM * FN](fill=_M64(0))
    for k0 in range(0, kk, KB):
        # Stage A^T: at[p][i] = A[m0 + i][k0 + p]; B: bt[j][p] = B[n0 + j][k0 + p].
        comptime for s in range((BM * KB + NT - 1) // NT):
            var idx = s * NT + tid
            if idx < BM * KB:
                var i = idx // KB
                var p = idx % KB
                var gi = m0 + i
                var gp = k0 + p
                var v = Float32(0)
                if gi < mm and gp < kk:
                    v = a[gi * kk + gp]
                at[p * AST + i] = v
        comptime for s in range((BN * KB + NT - 1) // NT):
            var idx = s * NT + tid
            if idx < BN * KB:
                var j = idx // KB
                var p = idx % KB
                var gj = n0 + j
                var gp = k0 + p
                var v = Float32(0)
                if gj < nn and gp < kk:
                    v = b[gj * kk + gp]
                bt[j * BST + p] = v
        barrier()
        comptime for p8 in range(KB // 8):
            var af = InlineArray[_M64, FM](fill=_M64(0))
            var bf = InlineArray[_M64, FN](fill=_M64(0))
            comptime for fm in range(FM):
                # M[r][c] = at[(8 p8 + c) * AST + i0 + r] = A[i0 + r][k0 + 8 p8 + c]
                af[fm] = _sg_load_t(at + (8 * p8) * AST + (sgm * FM + fm) * 8, AST)
            comptime for fq in range(FN):
                # M[r][c] = bt[(j0 + c) * BST + 8 p8 + r] = B[j0 + c][k0 + 8 p8 + r]
                bf[fq] = _sg_load_t(bt + ((sgn * FN + fq) * 8) * BST + 8 * p8, BST)
            comptime for fm in range(FM):
                comptime for fq in range(FN):
                    acc[fm * FN + fq] = _sg_mma(af[fm], bf[fq], acc[fm * FN + fq])
        barrier()
    var qd = lane // 4
    var frow = (qd & 4) + ((lane // 2) % 4)
    var fcol = (qd & 2) * 2 + (lane % 2) * 2
    comptime for fm in range(FM):
        comptime for fq in range(FN):
            comptime for e in range(2):
                var i = m0 + (sgm * FM + fm) * 8 + frow
                var j = n0 + (sgn * FN + fq) * 8 + fcol + e
                if i < mm and j < nn:
                    c[i * nn + j] = acc[fm * FN + fq][e]


def run[SGM: Int, SGN: Int, FM: Int, FN: Int, KB: Int](
    ctx: DeviceContext, name: String, m: Int, n: Int, k: Int
) raises:
    comptime BM = 8 * FM * SGM
    comptime BN = 8 * FN * SGN
    var ab = ctx.enqueue_create_buffer[DType.float32](m * k)
    var bb = ctx.enqueue_create_buffer[DType.float32](n * k)
    var cb = ctx.enqueue_create_buffer[DType.float32](m * n)
    var ha = List[Float32](length=m * k, fill=0)
    var hb = List[Float32](length=n * k, fill=0)
    for i in range(m * k):
        ha[i] = Float32((i * 7 + 3) % 17) * 0.125 - 1.0
    for i in range(n * k):
        hb[i] = Float32((i * 5 + 1) % 13) * 0.25 - 1.5
    ctx.enqueue_copy(ab, ha.unsafe_ptr())
    ctx.enqueue_copy(bb, hb.unsafe_ptr())
    ctx.synchronize()
    var blocks = ((m + BM - 1) // BM) * ((n + BN - 1) // BN)
    comptime kern = mma_gemm_kernel[SGM, SGN, FM, FN, KB]
    var best = Float64(1e30)
    for r in range(4):
        var t0 = perf_counter_ns()
        ctx.enqueue_function[kern](
            cb.unsafe_ptr(), ab.unsafe_ptr(), bb.unsafe_ptr(), Int32(m), Int32(n), Int32(k),
            grid_dim=(blocks, 1, 1), block_dim=(SGM * SGN * 32, 1, 1),
        )
        ctx.synchronize()
        var ms = Float64(perf_counter_ns() - t0) / 1e6
        if r > 0 and ms < best:
            best = ms
    # spot check four cells against the host
    var hc = List[Float32](length=m * n, fill=0)
    ctx.enqueue_copy(hc.unsafe_ptr(), cb)
    ctx.synchronize()
    var maxerr = Float64(0)
    for t in range(4):
        var i = (t * 2654435761) % m
        var j = (t * 40503 + 7) % n
        var s = Float64(0)
        for p in range(k):
            s += Float64(ha[i * k + p]) * Float64(hb[j * k + p])
        var e = abs(s - Float64(hc[i * n + j]))
        if e > maxerr:
            maxerr = e
    var gf = 2.0 * Float64(m) * Float64(n) * Float64(k) / (best * 1e6)
    print("MMA_CEIL cfg=" + name + " m=" + String(m) + " n=" + String(n) + " k=" + String(k)
          + " ms=" + String(best) + " GF/s=" + String(Int(gf)) + " maxerr=" + String(maxerr))


def main() raises:
    var ctx = DeviceContext()
    run[2, 2, 4, 4, 32](ctx, "sg2x2_f4x4_kb32", 8192, 768, 768)
    run[2, 2, 4, 4, 16](ctx, "sg2x2_f4x4_kb16", 8192, 768, 768)
    run[2, 2, 2, 4, 32](ctx, "sg2x2_f2x4_kb32", 8192, 768, 768)
    run[4, 2, 2, 4, 32](ctx, "sg4x2_f2x4_kb32", 8192, 768, 768)
    run[2, 2, 4, 4, 32](ctx, "sg2x2_f4x4_kb32", 8192, 50257, 768)
    run[2, 2, 4, 4, 32](ctx, "sg2x2_f4x4_kb32", 8192, 2048, 768)
