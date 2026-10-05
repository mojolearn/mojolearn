# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST on Apple (lane/apple-fast-decomp-linalg, 2026-10-02): the decomp
kit's gemm as a threadgroup-tiled kernel. NOT an IDENTICAL path:
`x_decomp/device.mojo` `launch_gemm` reaches it only under FAST on the Apple
column, and only when built with `-D MOJOLEARN_DECOMP_FAST_GEMM_TILED`
(`DECOMP_FAST_GEMM_TILED` below; no env read). Recovered onto main by
lane/apple-fast-rec-decomp (2026-10-04).

Cause: `gemm_kernel` / `gemm_part_kernel` (x_decomp/device.mojo) are one
thread per output cell walking the whole k axis from device memory, so
every operand word is re-read once per output row (or column) it feeds:
at the kit's board shapes (1,000,000 x 220 by 220 x 8 in NMF's four
products an iteration, A V at 1,000,000 x 220 x 220 in svd / lstsq, the
Grams of ALS, PLS and FactorAnalysis) that is a factor m or n of traffic
over the operands' size. Here a block of FG_TPB threads owns an FG_T x
FG_T output tile and a FG_K-deep slab of each operand in threadgroup
memory, each thread a 2 x 2 micro-tile; the k axis keeps `launch_gemm`'s
FOLD_BLOCK partials (grid z = the partial, `fold_kernel` sums them), so
the small-output, long-k products (W^T M, A^T Q) fill the GPU the same way
they do today. Same products, a different sum order.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from x_decomp.cells import F32Ptr, FOLD_BLOCK

#: output tile side, k slab depth, threads per block (16 x 16, each 2 x 2)
comptime FG_T = 32
comptime FG_K = 16
comptime FG_TPB = 256


#: Recovered candidate (lane/apple-fast-rec-decomp, 2026-10-04), default OFF,
#: FAST + Apple only. Source lane/apple-fast-decomp-linalg@74d52352b
#: (50d12a950). What it does: the kit's `launch_gemm` (x_decomp/device.mojo)
#: runs this threadgroup-tiled kernel (32 x 32 output tiles, a 16-deep k slab
#: in threadgroup memory, 2 x 2 cells a thread, main's FOLD_BLOCK partials on
#: grid z and `fold_kernel`) ahead of main's DECOMP_FAST_GEMM_MMA route, so
#: the A/B is this kernel against the matrix-unit GEMM. Known: prior M3 B arm
#: (dlin-rsvd-tiled-istella, old head) randomized-svd istella 683 ms; main's
#: DECOMP_FAST_GEMM_MMA since measured 711 -> 533 ms on the same lane, so the
#: lead is negative; als / lstsq / nmf arms have no recorded result. Fixed
#: in the port: the comptime define (with the FAST + Apple guard) replaces a
#: runtime `fast_gemm_on()` read, and it now takes precedence over the MMA
#: route main added. Tile sizes are the kernel's (256 threads, 2 x 2 each);
#: no dimension window.
comptime DECOMP_FAST_GEMM_TILED = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_DECOMP_FAST_GEMM_TILED"]()
)


def fg_tiles(count: Int) -> Int:
    return (count + FG_T - 1) // FG_T if count > 0 else 1


def fg_gemm_tiled_kernel(
    a: F32Ptr, b: F32Ptr, dst: F32Ptr, m: Int32, k: Int32, n: Int32, ta: Int32, tb: Int32, nb: Int32
):
    """Partial `block_idx.z` of C = op(A) op(B) over p in [z * FOLD_BLOCK,
    min(k, (z + 1) * FOLD_BLOCK)), tile (block_idx.y, block_idx.x) of the
    m x n output, stored at dst[z * m * n + i * n + j] (dst = C itself when
    nb == 1). A is m x k (k x m when ta), B is k x n (n x k when tb); the
    slab loads walk the contiguous axis of each layout across consecutive
    threads."""
    var M = Int(m)
    var K = Int(k)
    var N = Int(n)
    var TA = Int(ta) != 0
    var TB = Int(tb) != 0
    var tid = Int(thread_idx.x)
    var tr = tid // 16
    var tc = tid - tr * 16
    var i0 = Int(block_idx.y) * FG_T
    var j0 = Int(block_idx.x) * FG_T
    var z = Int(block_idx.z)
    var p0 = z * FOLD_BLOCK
    var p1 = p0 + FOLD_BLOCK
    if p1 > K:
        p1 = K
    var As = stack_allocation[FG_T * FG_K, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var Bs = stack_allocation[FG_K * FG_T, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var c00 = Float32(0)
    var c01 = Float32(0)
    var c10 = Float32(0)
    var c11 = Float32(0)
    var ia = i0 + 2 * tr
    var jb = j0 + 2 * tc
    var ps = p0
    while ps < p1:
        # the A slab: FG_T rows i x FG_K cols p, As[ii * FG_K + pp]
        var q = tid
        while q < FG_T * FG_K:
            var ii = 0
            var pp = 0
            if TA:
                ii = q % FG_T
                pp = q // FG_T
            else:
                pp = q % FG_K
                ii = q // FG_K
            var i = i0 + ii
            var p = ps + pp
            var v = Float32(0)
            if i < M and p < p1:
                v = a.unsafe_load(p * M + i) if TA else a.unsafe_load(i * K + p)
            As[ii * FG_K + pp] = v
            q += FG_TPB
        # the B slab: FG_K rows p x FG_T cols j, Bs[pp * FG_T + jj]
        q = tid
        while q < FG_K * FG_T:
            var jj = 0
            var pp = 0
            if TB:
                pp = q % FG_K
                jj = q // FG_K
            else:
                jj = q % FG_T
                pp = q // FG_T
            var j = j0 + jj
            var p = ps + pp
            var v = Float32(0)
            if j < N and p < p1:
                v = b.unsafe_load(j * K + p) if TB else b.unsafe_load(p * N + j)
            Bs[pp * FG_T + jj] = v
            q += FG_TPB
        barrier()
        for pp in range(FG_K):
            var a0 = As[(2 * tr) * FG_K + pp]
            var a1 = As[(2 * tr + 1) * FG_K + pp]
            var b0 = Bs[pp * FG_T + 2 * tc]
            var b1 = Bs[pp * FG_T + 2 * tc + 1]
            c00 = a0 * b0 + c00
            c01 = a0 * b1 + c01
            c10 = a1 * b0 + c10
            c11 = a1 * b1 + c11
        barrier()
        ps += FG_K
    var base = z * M * N
    if ia < M:
        if jb < N:
            dst.unsafe_store(base + ia * N + jb, c00)
        if jb + 1 < N:
            dst.unsafe_store(base + ia * N + jb + 1, c01)
    if ia + 1 < M:
        if jb < N:
            dst.unsafe_store(base + (ia + 1) * N + jb, c10)
        if jb + 1 < N:
            dst.unsafe_store(base + (ia + 1) * N + jb + 1, c11)
