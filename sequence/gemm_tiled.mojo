# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`OP_GEMM` WITH SHARED-MEMORY STAGING (nr-small D1, 2026-10-04).

`op_gemm` (sequence/ops.mojo) runs one thread per output cell, and each
thread loads its own K-long row of A and column of B from global memory: the
recurrent input projections and the MLP fit products re-read every A row N
times and every B column M times. Here a block of GT_M x GT_N threads owns a
GT_M x GT_N tile of C; the block stages GT_K-wide slabs of A and B in
threadgroup memory (each word flushed by `ld`, as `gemm_dot` reads it), and
every thread then folds ITS cell's chain over the slab:

    acc = (i7 ? C[m, n] : +0.0); acc = fma3(A(m, k), B(k, n), acc), k ascending

which is `gemm_dot`'s chain term for term (same flushed operands, same fused
step, same order), so the stored word is the one-thread op's word on every
column and on the host. Only the loads are shared; no fold order changes.
The page is (GT_M + GT_N) * GT_K floats = 4 KB, under every column's
threadgroup limit (the fits gate is the comptime assert below).

IDENTICAL, NVIDIA and AMD (Apple keeps its measured paths).
-D MOJOLEARN_IDN_SEQ_GEMM_TILED_OFF (or MOJOLEARN_IDN_ALL_OFF) restores the
one-thread launch."""
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from sequence.ops import FP, fma3, ld, st

comptime GT_M = 16
comptime GT_N = 16
comptime GT_K = 32
comptime GT_TPB = GT_M * GT_N
#: threadgroup bytes of the two slabs (the smallest column limit is 32 KB)
comptime GT_PAGE_BYTES = (GT_M + GT_N) * GT_K * 4
#: the tiled launch pays off only when a tile is mostly live; below this
#: edge on either side of C the one-thread launch is kept (a generic rule,
#: not keyed to any data shape)
comptime GT_MIN_EDGE = 8

comptime SEQ_GEMM_TILED = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not has_apple_gpu_accelerator()
    and not (is_defined["MOJOLEARN_IDN_SEQ_GEMM_TILED_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
)


@always_inline
def seq_gemm_tiled_on(M: Int, N: Int) -> Bool:
    return M >= GT_MIN_EDGE and N >= GT_MIN_EDGE


@always_inline
def seq_gemm_tiled_blocks(M: Int, N: Int) -> Int:
    return ((M + GT_M - 1) // GT_M) * ((N + GT_N - 1) // GT_N)


def seq_gemm_tiled_kernel(
    pa: FP, pb: FP, pc: FP,
    m_rows: Int32, n_cols: Int32, k_len: Int32,
    sam: Int32, sak: Int32, sbk: Int32, sbn: Int32,
    accumulate: Int32, ldc: Int32,
):
    """C[m, n] (row stride ldc) = (accumulate ? C[m, n] : 0) + sum_k A(m, k) B(k, n),
    A(m, k) = pa[m sam + k sak], B(k, n) = pb[k sbk + n sbn]; op_gemm's
    i0..i8 in that order. One block per GT_M x GT_N tile of C."""
    comptime assert GT_PAGE_BYTES <= 32768, "seq_gemm_tiled: the slab page must fit every column"
    var M = Int(m_rows)
    var N = Int(n_cols)
    var K = Int(k_len)
    var ntn = (N + GT_N - 1) // GT_N
    var b = Int(block_idx.x)
    var tm = b // ntn
    var tn = b - tm * ntn
    var m0 = tm * GT_M
    var n0 = tn * GT_N
    var tid = Int(thread_idx.x)
    var r = tid // GT_N
    var c = tid - r * GT_N
    var m = m0 + r
    var n = n0 + c
    var live = m < M and n < N
    var ci = m * Int(ldc) + n
    var acc = Float32(0.0)
    if live and accumulate != 0:
        acc = ld(pc, ci)
    var a_s = stack_allocation[GT_M * GT_K, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var b_s = stack_allocation[GT_K * GT_N, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var k0 = 0
    while k0 < K:
        var cnt = min(GT_K, K - k0)
        # the A slab: GT_M rows of cnt terms (row-major [row, kk])
        var q = tid
        while q < GT_M * GT_K:
            var row = q // GT_K
            var kk = q - row * GT_K
            var v = Float32(0.0)
            if kk < cnt and m0 + row < M:
                v = ld(pa, (m0 + row) * Int(sam) + (k0 + kk) * Int(sak))
            a_s[q] = v
            q += GT_TPB
        # the B slab: cnt terms of GT_N columns (row-major [kk, col])
        q = tid
        while q < GT_K * GT_N:
            var kk = q // GT_N
            var col = q - kk * GT_N
            var v = Float32(0.0)
            if kk < cnt and n0 + col < N:
                v = ld(pb, (k0 + kk) * Int(sbk) + (n0 + col) * Int(sbn))
            b_s[q] = v
            q += GT_TPB
        barrier()
        if live:
            for i in range(cnt):
                acc = fma3(a_s[r * GT_K + i], b_s[i * GT_N + c], acc)
        barrier()
        k0 += GT_K
    if live:
        st(pc, ci, acc)
