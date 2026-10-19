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
The page is GT_PAGE_BYTES (16.6 KB with GT_V2, 4 KB without), under every
column's threadgroup limit (the fits gate is the comptime assert below).

GT_V2 (lane ag-pass3-impl, 2026-10-19, default on; PASS3_IDEAS idea 6):
GT_K 32 -> 128 (K = 512 runs 4 slab rounds instead of 16: 4x fewer
barriers and 4x more loads in flight a round), and each slab is loaded
along its operand's unit stride: A rows fastest when sam == 1 and sak != 1
(x^T in the weight gradients), B terms fastest when sbk == 1 and sbn != 1
(W^T in the input gradients), the old mapping otherwise; one padding word a
slab row keeps the transposed stores and the fold's reads free of bank
conflicts. The rule reads the operand strides, never a size. SAME BITS on
every column and against the hand path's one-thread op_gemm / op_gemm_epi
(the hand MLP and the recurrent GEMMs share this kernel through OP_GEMM):
a cell's chain is acc = fma3(A(m, k), B(k, n), acc) with the slabs
ascending and the terms ascending inside a slab, i.e. k ascending from 0 to
K - 1 whatever the slab width (the width only cuts the k range into
consecutive pieces; padding terms k >= K are never folded, i < cnt). The
load mapping and the padding only decide which thread fetches a word and
where it sits in threadgroup memory, never the word (each is ld(), flushed,
as gemm_dot reads it) or the position it is folded at.
-D MOJOLEARN_IDN_SEQ_GEMM_TILED_V2_OFF restores GT_K = 32 and the old mapping.

The epilogue kernel can also fold the column sums of B (lane ag-pass3-impl,
PASS3_IDEAS idea 2): `colsum != 0` makes the first tile row's 16 threads add
each staged B word of their column, k ascending from +0.0, and store
div(sum, f1) into p4[n]: `op_colsum` then `op_divs` (OP_COLSUM_DIV) word for
word, so the fused linear node's bias gradient rides in its dW launch.

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
from sequence.ops import FP, Args, add, fma3, ld, st
from sequence.mlp import div, gemm_epi_tail

comptime GT_V2 = not (
    is_defined["MOJOLEARN_IDN_SEQ_GEMM_TILED_V2_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime GT_M = 16
comptime GT_N = 16
comptime GT_K = 128 if GT_V2 else 32
#: slab row strides in threadgroup memory (one padding word a row under GT_V2)
comptime GT_AS = (GT_K + 1) if GT_V2 else GT_K
comptime GT_BS = (GT_N + 1) if GT_V2 else GT_N
comptime GT_TPB = GT_M * GT_N
#: threadgroup bytes of the two slabs (the smallest column limit is 32 KB)
comptime GT_PAGE_BYTES = (GT_M * GT_AS + GT_K * GT_BS) * 4
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
    var SAM = Int(sam)
    var SAK = Int(sak)
    var SBK = Int(sbk)
    var SBN = Int(sbn)
    # GT_V2: load each slab along its operand's unit stride (same words)
    var a_rows_fast = GT_V2 and SAM == 1 and SAK != 1
    var b_terms_fast = GT_V2 and SBK == 1 and SBN != 1
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
    var a_s = stack_allocation[GT_M * GT_AS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var b_s = stack_allocation[GT_K * GT_BS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var k0 = 0
    while k0 < K:
        var cnt = min(GT_K, K - k0)
        # the A slab: GT_M rows of cnt terms (a_s[row * GT_AS + kk])
        var q = tid
        while q < GT_M * GT_K:
            var row: Int
            var kk: Int
            if a_rows_fast:
                kk = q // GT_M
                row = q - kk * GT_M
            else:
                row = q // GT_K
                kk = q - row * GT_K
            var v = Float32(0.0)
            if kk < cnt and m0 + row < M:
                v = ld(pa, (m0 + row) * SAM + (k0 + kk) * SAK)
            a_s[row * GT_AS + kk] = v
            q += GT_TPB
        # the B slab: cnt terms of GT_N columns (b_s[kk * GT_BS + col])
        q = tid
        while q < GT_K * GT_N:
            var kk: Int
            var col: Int
            if b_terms_fast:
                col = q // GT_K
                kk = q - col * GT_K
            else:
                kk = q // GT_N
                col = q - kk * GT_N
            var v = Float32(0.0)
            if kk < cnt and n0 + col < N:
                v = ld(pb, (k0 + kk) * SBK + (n0 + col) * SBN)
            b_s[kk * GT_BS + col] = v
            q += GT_TPB
        barrier()
        if live:
            for i in range(cnt):
                acc = fma3(a_s[r * GT_AS + i], b_s[i * GT_BS + c], acc)
        barrier()
        k0 += GT_K
    if live:
        st(pc, ci, acc)


def seq_gemm_epi_tiled_kernel(
    pa: FP, pb: FP, pc: FP, p3: FP, p4: FP,
    m_rows: Int32, n_cols: Int32, k_len: Int32,
    sam: Int32, sak: Int32, sbk: Int32, sbn: Int32,
    epi: Int32, epi_arg: Int32, colsum: Int32, f0: Float32, f1: Float32,
):
    """OP_GEMM_EPI on the tile (lane ag-pass2-kernels, the tape's
    OP_AGK_GEMM_EPI): seq_gemm_tiled_kernel's chain into a dense C (row
    stride N, no accumulate), then on the word this thread just stored
    op_gemm_epi's follower verbatim (sequence/mlp.mojo gemm_epi_tail: i9 =
    epi, i10 = epi_arg, f0, f1, p3), indexed by the cell's dense index as
    op_gemm_epi indexes it by its thread. The same words as op_gemm_epi.
    colsum != 0 (i11, lane ag-pass3-impl): also p4[n] = div(sum_k B(k, n), f1),
    k ascending from +0.0, OP_COLSUM_DIV's word (B dense: sbk = N, sbn = 1)."""
    comptime assert GT_PAGE_BYTES <= 32768, "seq_gemm_tiled: the slab page must fit every column"
    var M = Int(m_rows)
    var N = Int(n_cols)
    var K = Int(k_len)
    var SAM = Int(sam)
    var SAK = Int(sak)
    var SBK = Int(sbk)
    var SBN = Int(sbn)
    var a_rows_fast = GT_V2 and SAM == 1 and SAK != 1
    var b_terms_fast = GT_V2 and SBK == 1 and SBN != 1
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
    var ci = m * N + n
    var acc = Float32(0.0)
    # the column sums: the first tile row's threads, one column each, so
    # every column is summed by exactly one thread (op_colsum's chain)
    var cs_on = colsum != 0 and tm == 0 and r == 0 and n < N
    var cs = Float32(0.0)
    var a_s = stack_allocation[GT_M * GT_AS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var b_s = stack_allocation[GT_K * GT_BS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var k0 = 0
    while k0 < K:
        var cnt = min(GT_K, K - k0)
        var q = tid
        while q < GT_M * GT_K:
            var row: Int
            var kk: Int
            if a_rows_fast:
                kk = q // GT_M
                row = q - kk * GT_M
            else:
                row = q // GT_K
                kk = q - row * GT_K
            var v = Float32(0.0)
            if kk < cnt and m0 + row < M:
                v = ld(pa, (m0 + row) * SAM + (k0 + kk) * SAK)
            a_s[row * GT_AS + kk] = v
            q += GT_TPB
        q = tid
        while q < GT_K * GT_N:
            var kk: Int
            var col: Int
            if b_terms_fast:
                col = q // GT_K
                kk = q - col * GT_K
            else:
                kk = q // GT_N
                col = q - kk * GT_N
            var v = Float32(0.0)
            if kk < cnt and n0 + col < N:
                v = ld(pb, (k0 + kk) * SBK + (n0 + col) * SBN)
            b_s[kk * GT_BS + col] = v
            q += GT_TPB
        barrier()
        if live:
            for i in range(cnt):
                acc = fma3(a_s[r * GT_AS + i], b_s[i * GT_BS + c], acc)
        if cs_on:
            # op_colsum: acc = add(acc, ld(X, r * i2 + c)), r ascending; the
            # staged word is that ld's word
            for i in range(cnt):
                cs = add(cs, b_s[i * GT_BS + c])
        barrier()
        k0 += GT_K
    if live:
        st(pc, ci, acc)
        var e = Args()
        e.p2 = pc
        e.p3 = p3
        e.i1 = N
        e.i8 = N
        e.i9 = Int(epi)
        e.i10 = Int(epi_arg)
        e.f0 = f0
        e.f1 = f1
        gemm_epi_tail(ci, e)
    if cs_on:
        # op_divs on op_colsum's stored word: div(ld(st(cs)), f0) = div(cs, f1)
        # (cs is already flushed by add)
        st(p4, n, div(cs, f1))
