# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The recurrent backward the tape calls: the one-launch cell backward scan
(op_cell_bwd_scan, cell_bwd_scan_kernel and its step, moved unchanged out of
sequence/recurrent_scan.mojo, which keeps the forward scan, the routes and the
shared argument packing) and the blocked weight-gradient and bias-gradient
folds (wgrad_blocked, wgrad_block, colsum, moved unchanged out of
sequence/recurrent.mojo, which keeps the net, the forward and the trainer)."""

from std.gpu import block_idx, thread_idx
from x_linear.team import team_barrier
from sequence.ops import FP, Args, gates_of, op_cell_bwd_h, st
from sequence.recurrent_scan import _scan_args
from sequence.exec_trait import Exec
from sequence.ops import OP_COLSUM, OP_GEMM_SPLITK
from sequence.recurrent import (
    NN43_WGRAD_FIXED128,
    SEQ_WGRAD_BLOCKED,
    WGRAD_IDN_SCRATCH,
    WGRAD_MIN_BLOCK,
    gemm,
)


@always_inline
def bwd_step[odd: Bool](a: Args, j: Int) -> Args:
    """`OP_CELL_BWD_H`'s arguments of iteration j (step s = T - 1 - j) from
    the scan's base arguments (s = 0 pointers; p5 / p10 the two dh buffers,
    p5 the recurrent dh of j = 0; i3 T, i4 B G H). `odd` is j's parity, a
    compile-time choice: the host chain swaps dh and dhn after every step, so
    an odd j reads p10 as its dh. A runtime parity would make the two slots
    a pointer select inside the kernel body, which Metal compiles to a
    pointer whose loads read zeros (autograd/ops.mojo _ld_right)."""
    var B = a.i1
    var H = a.i2
    var T = a.i3
    var GH = gates_of(a.i0) * H
    var s = T - 1 - j
    var r = a
    r.p0 = a.p0 + s * B * GH
    r.p1 = a.p1 + s * B * GH
    r.p2 = a.p2 + s * B * H
    r.p3 = a.p3 + s * B * H
    r.p4 = a.p4 + s * B * H
    r.p6 = a.p6 + s * B * H
    r.p8 = a.p8 + s * B * GH
    r.p9 = a.p9 + s * B * GH
    comptime if odd:
        r.p5 = a.p10
        r.p10 = a.p5
    r.i3 = 1 if s < T - 1 else 0
    return r


def op_cell_bwd_scan(row: Int, a: Args):
    """Row `row`'s whole backward recurrence: dh = dc = 0, then every step
    from T - 1 down."""
    var H = a.i2
    for u in range(H):
        st(a.p5, row * H + u, Float32(0.0))
        st(a.p7, row * H + u, Float32(0.0))
    var j = 0
    while j < a.i3:
        var se = bwd_step[False](a, j)
        for u in range(H):
            op_cell_bwd_h(row * H + u, se)
        if j + 1 < a.i3:
            var so = bwd_step[True](a, j + 1)
            for u in range(H):
                op_cell_bwd_h(row * H + u, so)
        j += 2


def cell_bwd_scan_kernel(
    p0: FP, p1: FP, p2: FP, p3: FP, p4: FP, p5: FP,
    p6: FP, p7: FP, p8: FP, p9: FP, p10: FP, p11: FP,
    i01: Int64, i23: Int64, i45: Int64, n: Int64,
):
    """One block per batch row: every step of the backward recurrence
    (`op_cell_bwd_h`, s from T - 1 down), a device-memory barrier between."""
    var row = Int(block_idx.x)
    if row >= Int(n):
        return
    var a = _scan_args(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, i01, i23, i45)
    var u = Int(thread_idx.x)
    var Hn = a.i2
    var t = row * Hn + u
    if u < Hn:
        st(a.p5, t, Float32(0.0))
        st(a.p7, t, Float32(0.0))
    # TOMBSTONE: MOJOLEARN_SEQ_FAST_LSTM_SCAN_WIDE (DROPPED-quality, rab10) deleted 2026-10-09 on lane/owed-deletions-D2;
    # code recoverable at b639a2bd2. Restore: git apply experiments/removed/MOJOLEARN_SEQ_FAST_LSTM_SCAN.patch
    # TOMBSTONE: MOJOLEARN_SEQ_FAST_LSTM_SCAN_SMEM (bundle DROPPED-quality, rab10 SCAN + SMEM does not train) deleted 2026-10-09
    # on lane/owed-deletions-D2: the threadgroup-staged arm; code recoverable at b639a2bd2.
    # Restore: git apply experiments/removed/MOJOLEARN_SEQ_FAST_LSTM_SCAN_SMEM.patch
    team_barrier()
    # Steps in pairs (even j, then odd j): each pair's dh / dhn slots are
    # fixed at compile time (bwd_step's parity), so no pointer is selected.
    # The bound tests are block-uniform (a.i3 is a kernel argument).
    var j = 0
    while j < a.i3:
        var se = bwd_step[False](a, j)
        if u < Hn:
            op_cell_bwd_h(t, se)
        team_barrier()
        if j + 1 < a.i3:
            var so = bwd_step[True](a, j + 1)
            if u < Hn:
                op_cell_bwd_h(t, so)
            team_barrier()
        j += 2


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
