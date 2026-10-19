# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# ---------------------------------------------------------------- GEMM ROUTES BY SHAPE (lane ag-pass4-gemm, 2026-10-19)
# Read from DeviceExec._launch (sequence/exec_device.mojo) and the kernels
# below; NVIDIA and AMD run the same routes (SEQ_GEMM_TILED: identical, not
# Apple). HIGGS MLP 28 -> 256 -> 1, batch 512: the five GEMMs a batch
# (M x N, K; operand strides sam sak / sbk sbn), the same arguments on every
# opcode (hand gemm_epi, tape linear_into, tape matmul VJPs). "cell" = one
# thread a cell (seq_kernel, TPB 128, gemm_dot: GEMM_STAGE 16 loads in flight,
# then ONE dependent load round trip per term of the K mod 16 tail);
# "tile16" = 16 x 16 cells, 256 threads, GT_K 128 slabs; "thinN" = 16 x 1
# cells, 256 threads loading, 256-term slabs; split-K and the coop cells are
# never taken here (coop: M N <= 1024 and K >= 32768 only).
#
# BEFORE (V1, -D MOJOLEARN_AUTOGRAD_GEMM_DISPATCH_V1): tile iff M >= 8 and N >= 8
# (seq_gemm_tiled_on), the same rule for OP_GEMM (1) and OP_AGK_GEMM_EPI
# (1401); OP_GEMM_EPI (60, hand) never tiled.
#  shape                      | OP_GEMM 1 (composed tape)      | 1401 (fused tape)               | 60 (hand)
#  Z1  512x256 K 28  28 1/256 1| tile16 512 blk, 1 slab, 2048+  | = OP_GEMM + bias/relu epilogue  | cell 1024 blk, 28-chain:
#                             | 2048 slab slots/blk for 448+448| in registers                   | 1 staged round + 12 tail
#                             | live (78% zero-filled)         |                                | round trips
#  Z2  512x1   K 256 256 1/1 1| cell 4 blk (512 thr = 8 wave64)| cell + bias/logistic            | cell (same)
#                             | 16 staged rounds; N = 1 < 8    |                                |
#  dW2 256x1   K 512 1 256/1 1| cell 2 blk (4 wave64), 32 rounds| cell + L2GRAD, THEN a nested   | cell; its colsum is the hand
#                             |                                | OP_COLSUM_DIV launch: 1 thread,| path's own launch
#                             |                                | a 512-row serial chain         |
#  dH  512x256 K 1   1 1/1 1  | tile16 512 blk, 1 slab of cnt 1| = OP_GEMM + act' epilogue       | cell 1024 blk, 1 term
#                             | 4096 slab slots/blk for 32 live|                                |
#                             | (127/128 zero-filled), 2 barr. |                                |
#                             | for 1 fma                      |                                |
#  dW1 28x256  K 512 1 28/256 1| tile16 2x16 = 32 blk (4/32 rows| = OP_GEMM + L2GRAD + colsum in  | cell 56 blk, 32 rounds, A
#                             | dead), 4 slabs; 0.1 blk a CU   | tm 0 blocks (16 thr, 512 adds) | strided by 28
#
# AFTER (default): seq_gemm_route below, the same rule for 1, 1401 and 60:
#  K < GT_MIN_K (4) -> cell; else both edges >= 8 -> tile16; else thinN
#  (N the short edge) or thinM; slab staging writes only the cnt live terms.
#  Z1  tile16 (2+2 slab slots a thread, was 8+8); 60 now tile16 too
#  Z2  thinN 32 blk, 1 slab: 16 A words a thread in flight, one 256 fma chain
#  dW2 thinN 16 blk, 2 slabs, db folded in the tile (no nested launch)
#  dH  cell 1024 blk, 1 term (no slab, no barrier); 1, 1401 and 60 alike
#  dW1 tile16 (unchanged); 60 now tile16 too
# Every route folds acc = fma3(A(m, k), B(k, n), acc), k ascending from
# (accumulate ? C : +0.0), then the same epilogue: the same words on every
# route, so no bit moves on any opcode, column or the host.
#
# WHY 1401 COST 2.3x OP_GEMM ON MI325X (AMD phase 2/3, AG_PROFILE=2):
#  (a) NOT a different tiling: 1 and 1401 take the same route on all five
#      shapes (exec_device.mojo, the two `seq_gemm_tiled_on` tests).
#  (b) 636 -> 744 ms (pass 3) is the nested OP_COLSUM_DIV after the
#      off-tile dW2 (exec_device.mojo, `comptime if OP == OP_AGK_GEMM_EPI:`
#      at the end of _launch): one thread, a 512-row chain, launched with
#      self.launch so under AG_PROFILE=2 its two syncs and its kernel sit
#      inside 1401's bracket: colsum_div 52 ms + 2 x 1954 syncs ~ 55 ms = 108.
#  (c) the other ~320 ms is per-launch kernel time on the same geometry:
#      the profile floor is ~14 us a launch (op_26 / op_28 / copy rows), so
#      forward 1401 ~33 us of kernel a launch vs OP_GEMM ~10 us, backward
#      ~63 vs ~22. The only code difference is the epilogue: a runtime
#      switch over EPI x activation that reaches identical_sigmoid,
#      identical_tanh and identical_div (checks/numerics.mojo, not
#      @always_inline: device calls), so on AMDGPU every 1401 kernel carries
#      a call ABI (callee-saved VGPR spills to scratch, conservative register
#      allocation) whatever the layer's activation; NVIDIA absorbs calls.
#      The new kernel takes the epilogue as a comptime kind (GK_*): the
#      plain, bias, bias + ReLU and act' kernels contain no call (L2GRAD
#      keeps identical_div; sigmoid / tanh keep theirs). Unverified
#      until profiled: AG_PROFILE=2 rows ag_gemm_epi vs gemm, both vendors.
#  (d) no per-thread Args record any more (pass 3 removed it), no host
#      round trip; p3 reads are p3[n] (broadcast) or p3[ci] (coalesced).
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
from sequence.ops import FP, add, fma3, ld, st
from sequence.mlp import div, gemm_epi_value, act_bwd_value, EPI_BIAS_ACT, EPI_L2GRAD, EPI_ACT_BWD, ACT_IDENTITY, ACT_RELU

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
    stride N, no accumulate), with op_gemm_epi's follower applied to the
    accumulator before the one store (sequence/mlp.mojo gemm_epi_value: i9 =
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
        # the follower applied to the accumulator, one store (lane
        # ag-pass3-impl; it used to store, build an Args record and reload
        # the word through gemm_epi_tail: the same words, a second pass)
        st(pc, ci, gemm_epi_value(acc, ci, n, Int(epi), Int(epi_arg), p3, f0, f1))
    if cs_on:
        # op_divs on op_colsum's stored word: div(ld(st(cs)), f0) = div(cs, f1)
        # (cs is already flushed by add)
        st(p4, n, div(cs, f1))


# ---------------------------------------------------------------- shape-aware routes (lane ag-pass4-gemm)
# The route rule (the table at the top of this file). Every number below is
# a cost boundary, never a data shape:
#  - GT_MIN_K: a cell's chain of K < 4 terms is at most 3 dependent load
#    round trips with no barrier and no shared staging; any tile pays one
#    global round trip, two barriers and the slab stores for the same cells,
#    so short-K products (an outer product, K = 1) stay one thread a cell.
#  - GT_MIN_EDGE (8, above): a 16 x 16 tile with an edge under 8 is mostly
#    dead cells; such products (a matrix-vector product, N = 1 or M = 1)
#    take a thin tile instead of one thread a cell: 16 cells along the long
#    edge, all 256 threads staging 256-term slabs (16 words a thread in
#    flight), so a K-long chain costs K / 256 global round trips instead of
#    K / 16 + K mod 16. The chain itself (K dependent fmas) is the floor on
#    every route.
# -D MOJOLEARN_AUTOGRAD_GEMM_DISPATCH_V1 restores the pass-3 routes (tile iff
# both edges >= 8 for OP_GEMM and OP_AGK_GEMM_EPI, OP_GEMM_EPI never tiled,
# the old kernels with full-slab staging): the A/B arm. Bits are the same on
# both arms.
comptime GEMM_DISPATCH_V1 = is_defined["MOJOLEARN_AUTOGRAD_GEMM_DISPATCH_V1"]()
comptime GT_MIN_K = 4
#: the thin tile: GT_THIN cells along the long edge, GT_THIN_K-term slabs
comptime GT_THIN = 16
comptime GT_THIN_K = 256

comptime GR_CELL = 0
comptime GR_TILE = 1
comptime GR_THIN_N = 2
comptime GR_THIN_M = 3

#: the epilogue kinds of seq_gemm_tile_kernel (comptime, so each kernel
#: carries only its own epilogue's code): GK_PLAIN op_gemm (accumulate, row
#: stride ldc); the others op_gemm_epi's follower on a dense C, as
#: gemm_epi_value computes it: GK_BIAS bias, no activation; GK_BIAS_RELU bias
#: then ReLU; GK_BIAS_ACT bias then the runtime activation (sigmoid, tanh,
#: anything else); GK_L2GRAD (acc + f0 p3[ci]) / f1; GK_ACT_BWD acc act'(p3[ci]).
comptime GK_PLAIN = 0
comptime GK_BIAS = 1
comptime GK_BIAS_RELU = 2
comptime GK_BIAS_ACT = 3
comptime GK_L2GRAD = 4
comptime GK_ACT_BWD = 5


@always_inline
def seq_gemm_route(M: Int, N: Int, K: Int) -> Int:
    """The kernel for an M x N x K product (GR_*), from the cost rule above."""
    comptime if GEMM_DISPATCH_V1:
        return GR_TILE if seq_gemm_tiled_on(M, N) else GR_CELL
    if K < GT_MIN_K:
        return GR_CELL
    if M >= GT_MIN_EDGE and N >= GT_MIN_EDGE:
        return GR_TILE
    # the thin tile runs along the long edge
    if N <= M:
        return GR_THIN_N
    return GR_THIN_M


@always_inline
def seq_gemm_epi_kind(epi: Int, act: Int) -> Int:
    """GK_* for op_gemm_epi's (i9, i10). EPI_BIAS_ACT with activation 0 or
    ACT_IDENTITY is a bias alone (gemm_epi_value applies no activation for 0;
    act_value leaves z for identity); ReLU has its own kind; every other
    activation code runs act_value at run time, as before. An unknown epi
    code is gemm_epi_value's plain accumulator."""
    if epi == EPI_BIAS_ACT:
        if act == 0 or act == ACT_IDENTITY:
            return GK_BIAS
        if act == ACT_RELU:
            return GK_BIAS_RELU
        return GK_BIAS_ACT
    if epi == EPI_L2GRAD:
        return GK_L2GRAD
    if epi == EPI_ACT_BWD:
        return GK_ACT_BWD
    return GK_PLAIN


@always_inline
def seq_gemm_tile_blocks(route: Int, M: Int, N: Int) -> Int:
    if route == GR_THIN_N:
        return ((M + GT_THIN - 1) // GT_THIN) * N
    if route == GR_THIN_M:
        return M * ((N + GT_THIN - 1) // GT_THIN)
    return seq_gemm_tiled_blocks(M, N)


@always_inline
def _tile_epi[EK: Int](
    acc: Float32, ci: Int, n: Int, epi_arg: Int, p3: FP, f0: Float32, f1: Float32,
) -> Float32:
    """gemm_epi_value's branch for kind EK, chosen at compile time (the same
    operations on the same operands, so the same word)."""
    var v = acc
    comptime if EK == GK_BIAS:
        v = add(acc, ld(p3, n))
    elif EK == GK_BIAS_RELU:
        # act_value's ACT_RELU branch on gemm_epi_value's z
        var z = add(acc, ld(p3, n))
        v = z if z > Float32(0.0) else Float32(0.0)
    elif EK == GK_BIAS_ACT:
        v = gemm_epi_value(acc, ci, n, EPI_BIAS_ACT, epi_arg, p3, f0, f1)
    elif EK == GK_L2GRAD:
        v = div(fma3(f0, ld(p3, ci), acc), f1)
    elif EK == GK_ACT_BWD:
        v = act_bwd_value(acc, ld(p3, ci), epi_arg)
    return v


def seq_gemm_tile_kernel[TM: Int, TN: Int, TK: Int, EK: Int](
    pa: FP, pb: FP, pc: FP, p3: FP, p4: FP,
    m_rows: Int32, n_cols: Int32, k_len: Int32,
    sam: Int32, sak: Int32, sbk: Int32, sbn: Int32,
    accumulate: Int32, ldc: Int32, epi_arg: Int32, colsum: Int32, f0: Float32, f1: Float32,
):
    """One block of GT_TPB threads per TM x TN tile of C (TM TN <= GT_TPB;
    the threads past TM TN only stage), TK-term slabs. Cell (m, n):
    acc = (accumulate ? C[m, n] : +0.0); acc = fma3(A(m, k), B(k, n), acc),
    k ascending (slabs ascending, terms ascending in a slab): gemm_dot's
    chain term for term, so op_gemm's word. EK != GK_PLAIN: dense C (ldc = N,
    no accumulate) and the epilogue _tile_epi[EK], op_gemm_epi's word.
    colsum != 0: also p4[n] = div(sum_k B(k, n), f1), k ascending from +0.0
    over the staged words, by the one thread of tile row 0 in column n
    (OP_COLSUM_DIV's word; B dense). Each slab stores only its cnt live
    terms (the fold reads no other slot), loaded along the operand's unit
    stride; neither changes a word or the order it is folded in."""
    comptime AS = TK + 1
    comptime BS = TN + 1
    comptime assert TM * TN <= GT_TPB, "seq_gemm_tile: more cells than threads"
    comptime assert (TM * AS + TK * BS) * 4 <= 32768, "seq_gemm_tile: the slab page must fit every column"
    var M = Int(m_rows)
    var N = Int(n_cols)
    var K = Int(k_len)
    var SAM = Int(sam)
    var SAK = Int(sak)
    var SBK = Int(sbk)
    var SBN = Int(sbn)
    var a_rows_fast = SAM == 1 and SAK != 1
    var b_terms_fast = SBK == 1 and SBN != 1
    var ntn = (N + TN - 1) // TN
    var b = Int(block_idx.x)
    var tm = b // ntn
    var tn = b - tm * ntn
    var m0 = tm * TM
    var n0 = tn * TN
    var tid = Int(thread_idx.x)
    var cell = tid < TM * TN
    var r = tid // TN
    var c = tid - r * TN
    var m = m0 + r
    var n = n0 + c
    var live = cell and m < M and n < N
    var ci = m * Int(ldc) + n
    var acc = Float32(0.0)
    comptime if EK == GK_PLAIN:
        if live and accumulate != 0:
            acc = ld(pc, ci)
    var cs_on = colsum != 0 and cell and tm == 0 and r == 0 and n < N
    var cs = Float32(0.0)
    var a_s = stack_allocation[TM * AS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var b_s = stack_allocation[TK * BS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var k0 = 0
    while k0 < K:
        var cnt = min(TK, K - k0)
        # the A slab: TM rows of cnt terms (a_s[row * AS + kk])
        var q = tid
        while q < TM * cnt:
            var row: Int
            var kk: Int
            if a_rows_fast:
                kk = q // TM
                row = q - kk * TM
            else:
                row = q // cnt
                kk = q - row * cnt
            var v = Float32(0.0)
            if m0 + row < M:
                v = ld(pa, (m0 + row) * SAM + (k0 + kk) * SAK)
            a_s[row * AS + kk] = v
            q += GT_TPB
        # the B slab: cnt terms of TN columns (b_s[kk * BS + col])
        q = tid
        while q < cnt * TN:
            var kk: Int
            var col: Int
            if b_terms_fast:
                col = q // cnt
                kk = q - col * cnt
            else:
                kk = q // TN
                col = q - kk * TN
            var v = Float32(0.0)
            if n0 + col < N:
                v = ld(pb, (k0 + kk) * SBK + (n0 + col) * SBN)
            b_s[kk * BS + col] = v
            q += GT_TPB
        barrier()
        if live:
            for i in range(cnt):
                acc = fma3(a_s[r * AS + i], b_s[i * BS + c], acc)
        if cs_on:
            # op_colsum: acc = add(acc, ld(X, r * i2 + c)), r ascending
            for i in range(cnt):
                cs = add(cs, b_s[i * BS + c])
        barrier()
        k0 += TK
    if live:
        comptime if EK == GK_PLAIN:
            st(pc, ci, acc)
        else:
            st(pc, ci, _tile_epi[EK](acc, ci, n, Int(epi_arg), p3, f0, f1))
    if cs_on:
        st(p4, n, div(cs, f1))
