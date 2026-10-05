# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""APPLE FAST RECURRENT SCAN (lane apple-fast-gap-lstm, 2026-10-03; default OFF).

The board's lstm rows (M3 FAST 1,880 ms against torch MPS 695 ms) train
564 optimizer steps (72,192 windows, batch 256, 2 epochs), and every step
launched the recurrence one kernel per time step: T = 24 `OP_CELL_FWD_H`
launches forward and 24 `OP_CELL_BWD_H` backward (sequence/recurrent.mojo
forward / backward), plus their zero fills. A batch row's recurrence never
reads another row, so one THREADGROUP PER ROW can walk all T steps in one
launch, a device-memory barrier between steps (x_linear/team.mojo
`team_barrier`): B blocks of H lanes, the same per-element bodies
(`op_cell_fwd_h`, `op_cell_bwd_h`) on the same words in the same order, so
the same bits. Parallel over rows and units; no host step.

Switches (FAST + Apple GPU only; IDENTICAL and other vendors compile main):
  -D MOJOLEARN_SEQ_FAST_LSTM_SCAN       the one-launch forward and backward
  -D MOJOLEARN_SEQ_FAST_LSTM_SCAN_SMEM  (with SCAN) h_prev / dGH_{s+1} staged
                                        in threadgroup memory per step, the
                                        same fold order (same bits)
  -D MOJOLEARN_SEQ_FAST_LSTM_WGRAD      the (time x batch)-long weight and bias
                                        gradient folds split over K
                                        (sequence/recurrent.mojo gemm/colsum;
                                        a different fold order, FAST only)

The host executor runs the scan ops as the per-step launches they replace
(row by row, step by step): the same cells in the same order.
"""
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.memory import AddressSpace

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL, ftz
from x_linear.team import team_barrier
from sequence.ops import FP, Args, add, fma3, gates_of, ld, op_cell_bwd, op_cell_bwd_h, op_cell_fwd, op_cell_fwd_h, st

comptime _APPLE_FAST = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
#: OUTCOME (M3 afc_ab_def, full board size, 1 run per arm, 2026-10-04, lane/
#: apple-fast-rec-ab2 @ 40027eb8e), the bundle SCAN + SCAN_SMEM + WGRAD:
#: lstm-clf 1877.4 -> 1315.5 ms but accuracy 0.9608 -> 0.5002, logloss 0.0954
#: -> 0.6931; lstm-reg 1876.9 -> 1312.2 ms but r2 0.9804 -> -0.1043. BROKEN, the
#: model does not train: DROPPED-quality, all three stay off. Symptom: logloss
#: = ln 2 and r2 ~ 0 are a constant predictor, so the recurrence contributes
#: nothing (h_T or the recurrent/weight gradients come out zero or unused).
#: The cause is not evident from reading the code (the step pointers in
#: fwd_step / bwd_step match the per-step launches, team_barrier orders device
#: memory on Apple); not fixed here. Next: an ID check of SCAN alone against the
#: per-step path (it claims the same bits), then SMEM, then WGRAD.
#: nr-small (review D2, 2026-10-04): CANDIDATE, default OFF. The one-launch
#: scan under IDENTICAL on every GPU, for the owed NV/AMD/Apple ID check
#: (the chain order is the T-launch path's; `team_barrier` orders device
#: memory on CUDA/HIP/Metal by the repo's lowering notes, unproven by a
#: run). -D MOJOLEARN_IDN_SEQ_LSTM_SCAN turns it on.
comptime SEQ_LSTM_SCAN_IDN = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_IDN_SEQ_LSTM_SCAN"]() and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
comptime SEQ_LSTM_SCAN = (_APPLE_FAST and is_defined["MOJOLEARN_SEQ_FAST_LSTM_SCAN"]()) or SEQ_LSTM_SCAN_IDN
comptime SEQ_LSTM_SCAN_SMEM = SEQ_LSTM_SCAN and is_defined["MOJOLEARN_SEQ_FAST_LSTM_SCAN_SMEM"]()
comptime SEQ_LSTM_WGRAD = _APPLE_FAST and is_defined["MOJOLEARN_SEQ_FAST_LSTM_WGRAD"]()

comptime OP_CELL_FWD_SCAN = 120
comptime OP_CELL_BWD_SCAN = 121

#: lanes per row block: H rounded up to a simdgroup multiple, at most this
comptime SCAN_MAX_H = 1024
#: threadgroup floats staged per step (SMEM): h_prev (H) or dGH_{s+1} (G H)
comptime SCAN_SMEM = 4096


def scan_applies(H: Int, G: Int) -> Bool:
    """The scan kernels' shape bound: one lane per unit, the staged row fits."""
    return H >= 1 and H <= SCAN_MAX_H and G * H <= SCAN_SMEM


def scan_tpb(H: Int) -> Int:
    return ((H + 31) // 32) * 32


# ------------------------------------------------------------------ per-step args
@always_inline
def fwd_step(a: Args, s: Int) -> Args:
    """`OP_CELL_FWD_H`'s arguments of step s from the scan's base arguments
    (step 0's pointers; i1 B, i2 H, i3 T)."""
    var B = a.i1
    var H = a.i2
    var GH = gates_of(a.i0) * H
    var r = a
    r.p0 = a.p0 + s * B * GH
    r.p1 = a.p1 + s * B * GH
    r.p2 = a.p2 + s * B * GH
    r.p3 = a.p3 + s * B * H
    r.p4 = a.p4 + s * B * H
    r.p5 = a.p5 + s * B * H
    r.p6 = a.p6 + s * B * H
    return r


@always_inline
def bwd_step(a: Args, j: Int) -> Args:
    """`OP_CELL_BWD_H`'s arguments of iteration j (step s = T - 1 - j) from
    the scan's base arguments (s = 0 pointers; p5 / p10 the two dh buffers,
    p5 the recurrent dh of j = 0; i3 T, i4 B G H)."""
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
    if j % 2 == 1:
        r.p5 = a.p10
        r.p10 = a.p5
    r.i3 = 1 if s < T - 1 else 0
    return r


# ------------------------------------------------------------------ host bodies
def op_cell_fwd_scan(row: Int, a: Args):
    """Row `row`'s whole forward recurrence, as the per-step launches ran it:
    h_0 = c_0 = 0, then every step's units in ascending order."""
    var H = a.i2
    for u in range(H):
        st(a.p3, row * H + u, Float32(0.0))
        st(a.p4, row * H + u, Float32(0.0))
    for s in range(a.i3):
        var sa = fwd_step(a, s)
        for u in range(H):
            op_cell_fwd_h(row * H + u, sa)


def op_cell_bwd_scan(row: Int, a: Args):
    """Row `row`'s whole backward recurrence: dh = dc = 0, then every step
    from T - 1 down."""
    var H = a.i2
    for u in range(H):
        st(a.p5, row * H + u, Float32(0.0))
        st(a.p7, row * H + u, Float32(0.0))
    for j in range(a.i3):
        var sa = bwd_step(a, j)
        for u in range(H):
            op_cell_bwd_h(row * H + u, sa)


# ------------------------------------------------------------------ device kernels
def _scan_args(
    p0: FP, p1: FP, p2: FP, p3: FP, p4: FP, p5: FP,
    p6: FP, p7: FP, p8: FP, p9: FP, p10: FP, p11: FP,
    cell: Int32, B: Int32, H: Int32, T: Int32, i4: Int32,
) -> Args:
    var a = Args()
    a.p0 = p0
    a.p1 = p1
    a.p2 = p2
    a.p3 = p3
    a.p4 = p4
    a.p5 = p5
    a.p6 = p6
    a.p7 = p7
    a.p8 = p8
    a.p9 = p9
    a.p10 = p10
    a.p11 = p11
    a.i0 = Int(cell)
    a.i1 = Int(B)
    a.i2 = Int(H)
    a.i3 = Int(T)
    a.i4 = Int(i4)
    return a


def cell_fwd_scan_kernel(
    p0: FP, p1: FP, p2: FP, p3: FP, p4: FP, p5: FP,
    p6: FP, p7: FP, p8: FP, p9: FP, p10: FP, p11: FP,
    cell: Int32, B: Int32, H: Int32, T: Int32, i4: Int32,
):
    """One block per batch row, lane u its unit: every step of the forward
    recurrence (`op_cell_fwd_h`), a device-memory barrier between steps.
    The early exit is block-uniform (block_idx only)."""
    var row = Int(block_idx.x)
    if row >= Int(B):
        return
    var a = _scan_args(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, cell, B, H, T, i4)
    var u = Int(thread_idx.x)
    var Hn = Int(H)
    var t = row * Hn + u
    if u < Hn:
        st(a.p3, t, Float32(0.0))
        st(a.p4, t, Float32(0.0))
    comptime if SEQ_LSTM_SCAN_SMEM:
        var hs = stack_allocation[SCAN_SMEM, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
        var G = gates_of(a.i0)
        var GH = G * Hn
        team_barrier()
        for s in range(a.i3):
            var sa = fwd_step(a, s)
            if u < Hn:
                hs[u] = ld(sa.p3, t)
            team_barrier()
            if u < Hn:
                # op_cell_fwd_h with h_prev read from the staged row: the
                # same fold (k ascending from 0, one fma per term), the
                # same bias add, then op_cell_fwd verbatim
                for g in range(G):
                    var n = g * Hn + u
                    var acc = Float32(0.0)
                    for k in range(Hn):
                        acc = fma3(hs[k], ld(sa.p7, n * Hn + k), acc)
                    st(sa.p1, row * GH + n, add(ftz(acc), ld(sa.p8, n)))
                op_cell_fwd(t, sa)
            team_barrier()
    else:
        team_barrier()
        for s in range(a.i3):
            var sa = fwd_step(a, s)
            if u < Hn:
                op_cell_fwd_h(t, sa)
            team_barrier()


def cell_bwd_scan_kernel(
    p0: FP, p1: FP, p2: FP, p3: FP, p4: FP, p5: FP,
    p6: FP, p7: FP, p8: FP, p9: FP, p10: FP, p11: FP,
    cell: Int32, B: Int32, H: Int32, T: Int32, i4: Int32,
):
    """One block per batch row: every step of the backward recurrence
    (`op_cell_bwd_h`, s from T - 1 down), a device-memory barrier between."""
    var row = Int(block_idx.x)
    if row >= Int(B):
        return
    var a = _scan_args(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, cell, B, H, T, i4)
    var u = Int(thread_idx.x)
    var Hn = Int(H)
    var t = row * Hn + u
    if u < Hn:
        st(a.p5, t, Float32(0.0))
        st(a.p7, t, Float32(0.0))
    comptime if SEQ_LSTM_SCAN_SMEM:
        var ds = stack_allocation[SCAN_SMEM, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
        var GH = gates_of(a.i0) * Hn
        team_barrier()
        for j in range(a.i3):
            var sa = bwd_step(a, j)
            if sa.i3 != 0:
                # stage dGH_{s+1}[row, :] (GH words) across the block's lanes
                var k = u
                while k < GH:
                    ds[k] = ld(sa.p9 + sa.i4, row * GH + k)
                    k += Int(scan_tpb(Hn))
            team_barrier()
            if u < Hn:
                if sa.i3 != 0:
                    # op_cell_bwd_h's fold from the staged row: k ascending
                    # from the direct part, one fma per term
                    var acc = ld(sa.p5, t)
                    for kk in range(GH):
                        acc = fma3(ds[kk], ld(sa.p11, kk * Hn + u), acc)
                    st(sa.p5, t, acc)
                op_cell_bwd(t, sa)
            team_barrier()
    else:
        team_barrier()
        for j in range(a.i3):
            var sa = bwd_step(a, j)
            if u < Hn:
                op_cell_bwd_h(t, sa)
            team_barrier()
