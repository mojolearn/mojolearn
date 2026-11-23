# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The recurrent cell backward scan the tape calls: op_cell_bwd_scan,
cell_bwd_scan_kernel and its step, moved unchanged out of
sequence/recurrent_scan.mojo, which keeps the forward scan, the route rules
and the shared argument packing. Its own module so the device and host
executors that route OP_CELL_BWD_SCAN import nothing more than they did."""

from max.gpu import block_dim, block_idx, thread_idx
from x_linear.team import team_barrier
from sequence.ops import FP, Args, _cell_bwd_recurrent, add, gates_of, ld, op_cell_bwd, op_cell_bwd_masked, st
from sequence.gemm_split import gemm_split_partial
from sequence.recurrent_scan import _scan_args


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
    # the reverse direction (lane nn-recurrent): i4 (the offset of the next step's dGH) is negative and the base
    # pointers are step 0's (time T - 1), so step s walks back s row blocks and its mask position is T - 1 - s;
    # the forward direction (i4 > 0) computes the same addresses as before
    var rev = a.i4 < 0
    var k = -s if rev else s
    var r = a
    r.p0 = a.p0 + k * B * GH
    r.p1 = a.p1 + k * B * GH
    r.p2 = a.p2 + k * B * H
    r.p3 = a.p3 + k * B * H
    # c_t is c_{s}'s next row block (the previous one in reverse): the base p4 is free to carry the sequence
    # lengths (lane seq-blocks-3)
    r.p4 = r.p3 + (-B * H if rev else B * H)
    r.p6 = a.p6 + k * B * H
    r.p8 = a.p8 + k * B * GH
    r.p9 = a.p9 + k * B * GH
    comptime if odd:
        r.p5 = a.p10
        r.p10 = a.p5
    r.i3 = 1 if s < T - 1 else 0
    r.i8 = T - 1 - s if rev else s
    return r


@always_inline
def scan_bwd_recurrent(t: Int, a: Args, R: Int):
    """A backward step's recurrent part for element t = b H + u in the fold the per-step chain takes at the layer's
    shape (lane rnn-scan): R = 0, or the last step (i3 = 0, its dh / dc zeros), op_cell_bwd_h's own body
    (sequence/ops.mojo _cell_bwd_recurrent: gemm_dot's chain from the stored direct part, the fused step's and the
    unsplit OP_GEMM's word); R > 0 the split-K order of OP_GEMM's accumulate launch (op_gk_split: the stored dh,
    then the partials of blocks of R terms, sequence/gemm_split.mojo gemm_split_partial, added in ascending order).
    A(m, k) = dGH_{s+1} (p9 + i4, row stride G H), B(k, n) = W_hh (p11, row stride H)."""
    if R == 0 or a.i3 == 0:
        _cell_bwd_recurrent(t, a)
        return
    var H = a.i2
    var GH = gates_of(a.i0) * H
    var b = t // H
    var u = t - b * H
    var acc = ld(a.p5, t)
    for s in range((GH + R - 1) // R):
        acc = add(acc, gemm_split_partial(a.p9 + a.i4, GH, 1, a.p11, H, 1, GH, b, u, s, R))
    st(a.p5, t, acc)


@always_inline
def _bwd_cell(t: Int, a: Args, se: Args):
    """Step `se` of element t: the recurrent part (scan_bwd_recurrent, the base's i6 = R its fold), then
    `op_cell_bwd`, or with the scan's i5 = 1 its length-masked form with the rows' lengths in the base p4 (lane
    seq-blocks-3). i5 and i6 are kernel arguments, so the branches are block-uniform. With R = 0 this is
    op_cell_bwd_h / op_cell_bwd_h_lens, body for body."""
    scan_bwd_recurrent(t, se, a.i6)
    if a.i5 != 0:
        op_cell_bwd_masked(t, se, a.p4)
    else:
        op_cell_bwd(t, se)


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
            _bwd_cell(row * H + u, a, se)
        if j + 1 < a.i3:
            var so = bwd_step[True](a, j + 1)
            for u in range(H):
                _bwd_cell(row * H + u, a, so)
        j += 2


def cell_bwd_scan_kernel(
    p0: FP, p1: FP, p2: FP, p3: FP, p4: FP, p5: FP,
    p6: FP, p7: FP, p8: FP, p9: FP, p10: FP, p11: FP,
    i01: Int64, i23: Int64, i45: Int64, i67: Int64, n: Int64,
):
    """One block per batch row: every step of the backward recurrence
    (`_bwd_cell`, s from T - 1 down), a device-memory barrier between. Lanes stride by the block width (lane
    rnn-scan), one lane a unit."""
    var row = Int(block_idx.x)
    if row >= Int(n):
        return
    var a = _scan_args(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, i01, i23, i45, i67)
    var lane = Int(thread_idx.x)
    var W = Int(block_dim.x)
    var Hn = a.i2
    var u = lane
    while u < Hn:
        st(a.p5, row * Hn + u, Float32(0.0))
        st(a.p7, row * Hn + u, Float32(0.0))
        u += W
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
        u = lane
        while u < Hn:
            _bwd_cell(row * Hn + u, a, se)
            u += W
        team_barrier()
        if j + 1 < a.i3:
            var so = bwd_step[True](a, j + 1)
            u = lane
            while u < Hn:
                _bwd_cell(row * Hn + u, a, so)
                u += W
            team_barrier()
        j += 2
