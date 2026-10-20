# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The recurrent backward folds the tape calls: the blocked weight-gradient
and bias-gradient folds (wgrad_blocked, wgrad_block, colsum), moved unchanged
out of sequence/recurrent.mojo, which keeps the net, the forward and the
trainer. The cell backward scan is autograd/kernels/recurrent_scan.mojo."""

from sequence.exec_trait import Exec
from sequence.ops import OP_COLSUM, OP_GEMM_SPLITK
from sequence.recurrent import (
    NN43_WGRAD_FIXED128,
    SEQ_WGRAD_BLOCKED,
    WGRAD_IDN_SCRATCH,
    WGRAD_MIN_BLOCK,
    gemm,
)


def wgrad_block(K: Int) -> Int:
    """Rows per block: the smallest power of two R >= WGRAD_MIN_BLOCK with
    R * R >= K (so at most R blocks)."""
    comptime if NN43_WGRAD_FIXED128:
        # A single cell's leaf partials must fit the fixed scratch budget.
        # For exceptionally long K, double the leaf on ALL columns until
        # it fits. This resource rule covers every neighboring K equally.
        var leaf = 128
        while (K + leaf - 1) // leaf > WGRAD_IDN_SCRATCH:
            leaf *= 2
        return leaf
    var r = WGRAD_MIN_BLOCK
    while r * r < K:
        r *= 2
    return r


def wgrad_blocked[E: Exec](
    mut ex: E, A: FP, B: FP, C: FP, M: Int, N: Int, K: Int,
    sam: Int, sak: Int, sbk: Int, sbn: Int, ldc: Int, scratch: FP,
) raises:
    """SEQ_WGRAD_BLOCKED: C = A B (no accumulate) in the blocked K order;
    `gemm` when K is one block. scratch holds WGRAD_IDN_SCRATCH floats."""
    var KB = wgrad_block(K)
    var S = (K + KB - 1) // KB
    if S <= 1:
        gemm(ex, A, B, C, M, N, K, sam, sak, sbk, sbn, False, ldc)
        return
    # cell chunks of at most WGRAD_IDN_SCRATCH / S cells (S <= 2^16 since
    # K < 2^31, so a chunk holds at least 64 cells)
    var Nc = min(N, WGRAD_IDN_SCRATCH // S)
    var Mc = max(1, WGRAD_IDN_SCRATCH // (S * Nc))
    var m0 = 0
    while m0 < M:
        var mc = min(Mc, M - m0)
        var n0 = 0
        while n0 < N:
            var nc = min(Nc, N - n0)
            var a = Args()
            a.p0 = A + m0 * sam
            a.p1 = B + n0 * sbn
            a.p2 = C + m0 * ldc + n0
            a.p3 = scratch
            a.i0 = mc
            a.i1 = nc
            a.i2 = K
            a.i3 = sam
            a.i4 = sak
            a.i5 = sbk
            a.i6 = sbn
            a.i7 = 0
            a.i8 = ldc
            a.i9 = S
            a.i10 = KB
            a.i11 = 0
            ex.launch[OP_GEMM_SPLITK](a, S * mc * nc)
            a.i11 = 1
            ex.launch[OP_GEMM_SPLITK](a, mc * nc)
            n0 += nc
        m0 += mc


def colsum[E: Exec](mut ex: E, X: FP, dst: FP, R: Int, C: Int) raises:
    var a = Args()
    a.p0 = X
    a.p1 = dst
    a.i0 = R
    a.i1 = C
    a.i2 = C
    ex.launch[OP_COLSUM](a, C)
