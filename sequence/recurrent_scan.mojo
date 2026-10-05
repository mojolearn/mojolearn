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
  -D MOJOLEARN_SEQ_FAST_LSTM_SCAN_WIDE  (with SCAN) G H lanes per row: one per
                                        gate column (see SEQ_LSTM_SCAN_WIDE)
  -D MOJOLEARN_SEQ_FAST_LSTM_WGRAD      the (time x batch)-long weight and bias
                                        gradient folds split over K
                                        (sequence/recurrent.mojo gemm/colsum;
                                        a different fold order, FAST only)

The host executor runs the scan ops as the per-step launches they replace
(row by row, step by step): the same cells in the same order.
"""
from std.gpu import block_dim, block_idx, thread_idx
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
#: memory on Apple). FIX CANDIDATE (lane apple-fast-s-seq, 2026-10-05): the
#: kernels' Args came from a non-inlined `_scan_args` that started from
#: `Args()`, whose pointer slots are integer-made (`dummy_ptr`): see the note
#: at `_scan_args`. Re-judge SCAN, SCAN + SMEM and the bundle on quality.
#: STILL BROKEN after the Args fix (8bb42b7de): M3 A/B rab10-scan (2026-10-05) lstm-clf synthetic accuracy 0.9608 -> 0.5002,
#: lstm-reg synthetic r2 0.9804 -> -0.1043 (constant prediction), 20% faster. SCAN_SMEM, WGRAD and SCAN_WIDE inherit it. DROPPED-quality;
#: the next step is a device-vs-host digest of SCAN alone, per step. Stays opt-in.
#: nr-small (review D2, 2026-10-04): CANDIDATE, default OFF. The one-launch
#: scan under IDENTICAL on every GPU, for the owed NV/AMD/Apple ID check
#: (the chain order is the T-launch path's; `team_barrier` orders device
#: memory on CUDA/HIP/Metal by the repo's lowering notes, unproven by a
#: run). -D MOJOLEARN_IDN_SEQ_LSTM_SCAN turns it on.
comptime SEQ_LSTM_SCAN_IDN = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_IDN_SEQ_LSTM_SCAN"]() and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
comptime SEQ_LSTM_SCAN = (_APPLE_FAST and is_defined["MOJOLEARN_SEQ_FAST_LSTM_SCAN"]()) or SEQ_LSTM_SCAN_IDN
comptime SEQ_LSTM_SCAN_SMEM = SEQ_LSTM_SCAN and is_defined["MOJOLEARN_SEQ_FAST_LSTM_SCAN_SMEM"]()
comptime SEQ_LSTM_WGRAD = _APPLE_FAST and is_defined["MOJOLEARN_SEQ_FAST_LSTM_WGRAD"]()
#: SEQ_FAST_LSTM_SCAN_WIDE (with SCAN; lane apple-fast-s-seq, 2026-10-05,
#: READY-AB): one lane per GATE COLUMN (G H lanes, 256 for the board's LSTM
#: H = 64) instead of one per unit. The scan's block of H lanes walks T steps
#: and each lane's step is G H-long dot chains (forward: G of length H;
#: backward: one of length G H), so a row's serial chain is G times what it
#: needs to be and the M3's cores hold 3 blocks of 64 lanes each (256 rows
#: over 80 cores): latency bound. Forward: lane n folds GH[row, n] alone
#: (k ascending from 0, one fma per term: `op_cell_fwd_h`'s fold, the same
#: bits), h_prev staged in threadgroup memory; then H lanes run the cell.
#: Backward: lane g H + u folds gate g's H terms of dh[u] from zero, then lane
#: u adds the G partials onto the direct part in g order: a different fold
#: from the one G H chain (FAST only). Shapes with G H > SCAN_MAX_H keep the
#: H-lane scan.
comptime SEQ_LSTM_SCAN_WIDE = SEQ_LSTM_SCAN and is_defined["MOJOLEARN_SEQ_FAST_LSTM_SCAN_WIDE"]()

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


def scan_block(cell: Int, H: Int) -> Int:
    """The scan launch's block width: G H lanes under SEQ_LSTM_SCAN_WIDE when
    they fit, else H. The kernels take the wide path only when the block holds
    G H lanes (block_dim), so the two sides cannot disagree."""
    comptime if SEQ_LSTM_SCAN_WIDE:
        var GH = gates_of(cell) * H
        if GH <= SCAN_MAX_H:
            return scan_tpb(GH)
    return scan_tpb(H)


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
@always_inline
def _scan_args(
    p0: FP, p1: FP, p2: FP, p3: FP, p4: FP, p5: FP,
    p6: FP, p7: FP, p8: FP, p9: FP, p10: FP, p11: FP,
    cell: Int32, B: Int32, H: Int32, T: Int32, i4: Int32,
) -> Args:
    #: THE BROKEN-BUNDLE FIX (lane apple-fast-s-seq, 2026-10-05). This built
    #: `Args()` and then overwrote its slots, and was not inlined. `Args()`
    #: fills every pointer slot with `dummy_ptr()`, a pointer made from the
    #: integer 64, and the struct then crossed a non-inlined call (returned
    #: through memory): the two Metal traps of
    #: memory/metal-no-int-pointers-inline-oct2 (PageRank all zeros, lane
    #: hr-graph). On Metal the kernel's loads and stores through those slots
    #: silently missed, so h_T stayed the zero fill and the dG buffers stayed
    #: zero: only the head bias trained, the constant predictor of the
    #: OUTCOME above (logloss = ln 2, r2 ~ 0). `seq_kernel`, which works,
    #: builds its Args with the fieldwise constructor in the kernel body
    #: (sequence/exec_device.mojo seq_kernel); this now does the same, inlined,
    #: with no integer-made pointer.
    return Args(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11,
                Int(cell), Int(B), Int(H), Int(T), Int(i4), 0, 0, 0, 0, 0, 0, 0,
                Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0),
                Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0))


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
    comptime if SEQ_LSTM_SCAN_WIDE:
        var Gw = gates_of(a.i0)
        var GHw = Gw * Hn
        if Int(block_dim.x) >= GHw:
            # SEQ_LSTM_SCAN_WIDE: lane n = g H + u folds GH[row, n]; block-
            # uniform branch (block_dim), so every lane meets every barrier
            var hw = stack_allocation[SCAN_MAX_H, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
            team_barrier()
            for s in range(a.i3):
                var sa = fwd_step(a, s)
                if u < Hn:
                    hw[u] = ld(sa.p3, t)
                team_barrier()
                if u < GHw:
                    # op_cell_fwd_h's fold for column n: k ascending from 0,
                    # one fma per term, then + b_hh (the same bits)
                    var acc = Float32(0.0)
                    for k in range(Hn):
                        acc = fma3(hw[k], ld(sa.p7, u * Hn + k), acc)
                    st(sa.p1, row * GHw + u, add(ftz(acc), ld(sa.p8, u)))
                team_barrier()
                if u < Hn:
                    op_cell_fwd(t, sa)
                team_barrier()
            return
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
    comptime if SEQ_LSTM_SCAN_WIDE:
        var Gw = gates_of(a.i0)
        var GHw = Gw * Hn
        if Int(block_dim.x) >= GHw:
            # SEQ_LSTM_SCAN_WIDE: lane n = g H + v folds gate g's part of
            # dh[v] = sum_{k in gate g} dGH_{s+1}[k] W_hh[k, v] from zero; lane
            # v then adds the G parts onto the direct part in g order (FAST
            # fold order). Block-uniform branch.
            var dw = stack_allocation[SCAN_MAX_H, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
            var pw = stack_allocation[SCAN_MAX_H, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
            var g = u // Hn if Hn > 0 else 0
            var v = u - g * Hn
            team_barrier()
            for j in range(a.i3):
                var sa = bwd_step(a, j)
                if sa.i3 != 0:
                    if u < GHw:
                        dw[u] = ld(sa.p9 + sa.i4, row * GHw + u)
                    team_barrier()
                    if u < GHw:
                        var acc = Float32(0.0)
                        var k0 = g * Hn
                        for k in range(k0, k0 + Hn):
                            acc = fma3(dw[k], ld(sa.p11, k * Hn + v), acc)
                        pw[u] = acc
                    team_barrier()
                    if u < Hn:
                        var dh = ld(sa.p5, t)
                        for gg in range(Gw):
                            dh = add(dh, pw[gg * Hn + u])
                        st(sa.p5, t, dh)
                if u < Hn:
                    op_cell_bwd(t, sa)
                team_barrier()
            return
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
                    k += Int(block_dim.x)
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
