# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RECURRENT SCAN (lane apple-fast-gap-lstm, 2026-10-03; the IDENTICAL default
since lane ag-pass5-recurrent, 2026-10-19: see SEQ_RNN_SCAN_IDN below).

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
  (-D MOJOLEARN_SEQ_FAST_LSTM_SCAN and its _SCAN_WIDE arm were deleted
   2026-10-09, DROPPED-quality; the kernels stay for the IDENTICAL
   -D MOJOLEARN_IDN_SEQ_LSTM_SCAN, see the TOMBSTONE below.)
  (-D MOJOLEARN_SEQ_FAST_LSTM_SCAN_SMEM, h_prev / dGH_{s+1} staged in
   threadgroup memory, was deleted 2026-10-09 with the same verdict.)
  (-D MOJOLEARN_SEQ_FAST_LSTM_WGRAD, the split-K weight-gradient folds,
   was deleted 2026-10-09 with the same verdict.)

The host executor runs the scan ops as the per-step launches they replace
(row by row, step by step): the same cells in the same order.
"""
from std.gpu import block_idx, thread_idx
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_linear.team import team_barrier
from sequence.ops import FP, Args, gates_of, ld, op_cell_bwd_h, op_cell_fwd_h, st

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
#: run). -D MOJOLEARN_IDN_SEQ_LSTM_SCAN turned it on (superseded: the scan
#: is the IDENTICAL default, -D MOJOLEARN_SEQ_RNN_SCAN_OFF is arm A).
# NI49 (2026-10-06) source-only disposition: NOT REPAIRED / NOT QUALIFIED.
# The failed 8bb42b7de Args change is not evidence that the remaining issue is
# numerical. No demonstrated source cause was established; preserve both
# constant-prediction failures above and keep SCAN default OFF. Forward and
# backward per-step device/host state digests and full training quality remain
# required before timing or promoting this candidate. The owner forbids those
# runs in this worktree; none were attempted. A faster broken arm is a loser.
# Lane ag-pass5-recurrent (2026-10-19, PASS5 idea 3): the scan is the
# IDENTICAL default for every recurrent cell (rnn, lstm, gru), in the hand
# trainer (sequence/recurrent.mojo forward / backward, HandStep) and the tape
# trainer (autograd/rules_recurrent.mojo rnn_layer / _layer_bwd,
# AutogradRnnStep) alike, so the two steps stay equal.
# -D MOJOLEARN_SEQ_RNN_SCAN_OFF (or MOJOLEARN_IDN_ALL_OFF) is arm A, the
# per-step chain (T launches of OP_CELL_FWD_H / OP_CELL_BWD_H plus 4 fills a
# layer). The older opt-in -D MOJOLEARN_IDN_SEQ_LSTM_SCAN is now a no-op.
# BITS: the same as the per-step chain, by construction. Each lane runs the
# per-step launch's element body (op_cell_fwd_h / op_cell_bwd_h) on the same
# Args the host built for that step (fwd_step / bwd_step reproduce the
# pointers, with the dh / dhn swap as a comptime parity, never a pointer
# select), on the same words: the gate folds are gemm_dot's k-ascending fma
# chains, the cell arithmetic is the same identical_* chain. A step's
# cross-lane reads (h_prev over all H units of the row forward, dGH_{s+1}
# over all G H columns of the row backward) are the row's own block's
# stores of the previous step, ordered by team_barrier (bar.sync on NVIDIA,
# s_barrier between workgroup fences on AMD, threadgroup_barrier(mem_device)
# on Metal; x_linear/team.mojo). The zero fills of h_0 / c_0 and dh / dc are
# stores of +0.0 to the same words. No fold order moves, so no column moves:
# the race digests (lstm 27500667, gru 8762faa2, rnn c81d55d6) must hold; the
# orchestrator's nv and amd ID checks decide it.
# Launches a layer step: forward T + 2 -> 1, backward T + 2 -> 1.
# APPLE (orchestrator compile-18, 2026-10-19): sequence/checks/rnn_scan_check
# PASSES on the AMD MI325X (12/12) and FAILS every case on Metal (M4): each
# scan output buffer is still the check's 7.0 poison, including the h_0 / dh
# zero stores that come before the first barrier, while the per-step chain on
# the same device equals the host word for word. On Metal the scan kernel
# stores nothing at all and no error is raised. The kernels now take seq_kernel's
# argument form (the integers packed two to an Int64, the element count last;
# see cell_fwd_scan_kernel); whether that is enough is unproven. Until the check
# passes on Metal, an Apple build routes every recurrent layer to the per-step
# chain, which gives the same bits (Apple's digests must still equal NVIDIA and
# AMD's, and the chain's do). -D MOJOLEARN_SEQ_RNN_SCAN_APPLE puts Apple on
# the scan for that check and its A/B. has_apple_gpu_accelerator is a
# property of the build host: an Apple build's host column takes the chain too,
# with the same bits.
comptime SEQ_RNN_SCAN_ON_THIS_TARGET = not has_apple_gpu_accelerator() or is_defined["MOJOLEARN_SEQ_RNN_SCAN_APPLE"]()
# THE DEFAULT, one line (orchestrator sets it after the NV numbers).
# A/B so far (one run per arm, bits equal scan == per-step on both vendors;
# rnn_scan_check 12/12 PASS on NV and AMD, 12/12 FAIL on the M4, see above):
#   AMD MI325X, s a fit, per-step -> scan: lstm 0.1206 -> 0.1388 (1.15x slower),
#   gru 0.1021 -> 0.1138 (1.11x slower), rnn 0.0672 -> 0.0591 (0.88x, faster).
#   NVIDIA L40S: owed.
# True: the scan is the IDENTICAL default and -D MOJOLEARN_SEQ_RNN_SCAN_OFF is
# arm A. False: the per-step chain is the default everywhere and
# -D MOJOLEARN_SEQ_RNN_SCAN opts in (the scan's arm B, and the Metal probe
# with MOJOLEARN_SEQ_RNN_SCAN_APPLE).
comptime SEQ_RNN_SCAN_DEFAULT = True
comptime SEQ_RNN_SCAN_REQUESTED = (
    is_defined["MOJOLEARN_SEQ_RNN_SCAN"]() or (SEQ_RNN_SCAN_DEFAULT and not is_defined["MOJOLEARN_SEQ_RNN_SCAN_OFF"]())
)
comptime SEQ_RNN_SCAN_IDN = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and SEQ_RNN_SCAN_ON_THIS_TARGET
    and SEQ_RNN_SCAN_REQUESTED
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime SEQ_LSTM_SCAN_IDN = SEQ_RNN_SCAN_IDN
# NI49's row-serial arm (MOJOLEARN_IDN_SEQ_ROW_SERIAL_SCAN: one GPU thread
# per batch row, serial over units and timesteps) was deleted by L11
# (2026-10-07): a per-row serial executor breaks the IDENTICAL GPU rule
# (parallel only). NN41 and NI49 are now the one switch above.
# TOMBSTONE: MOJOLEARN_SEQ_FAST_LSTM_SCAN (DROPPED-quality, BROKEN: M3 rab10-scan lstm-clf synthetic accuracy 0.9608 -> 0.5002,
# lstm-reg r2 0.9804 -> -0.1043, a constant predictor, 20% faster) deleted 2026-10-09 on lane/owed-deletions-D2: the FAST
# define only (the IDENTICAL MOJOLEARN_IDN_SEQ_LSTM_SCAN keeps these kernels, grid control seq_lstm_scan), plus
# MOJOLEARN_SEQ_FAST_LSTM_SCAN_WIDE (FAST-only arm on top of it, same DROPPED-quality rab10 row) and its kernel arms;
# code recoverable at b639a2bd2. Restore: git apply experiments/removed/MOJOLEARN_SEQ_FAST_LSTM_SCAN.patch
comptime SEQ_LSTM_SCAN = SEQ_LSTM_SCAN_IDN
# TOMBSTONE: MOJOLEARN_SEQ_FAST_LSTM_SCAN_SMEM (bundle DROPPED-quality, rab10 SCAN + SMEM does not train) deleted 2026-10-09
# on lane/owed-deletions-D2: the define and both kernels' staged arms; code recoverable at b639a2bd2.
# Restore: git apply experiments/removed/MOJOLEARN_SEQ_FAST_LSTM_SCAN_SMEM.patch
# TOMBSTONE: MOJOLEARN_SEQ_FAST_LSTM_WGRAD (DROPPED-quality with the SCAN bundle, 2026-10-04: lstm-clf accuracy 0.9608 ->
# 0.5002, lstm-reg r2 0.9804 -> -0.1043; never A/B-ed alone) deleted 2026-10-09 on lane/owed-deletions-D2: the FAST split-K
# weight-gradient folds (wgrad_gemm, WGRAD_SCRATCH); code recoverable at b639a2bd2.
# Restore: git apply experiments/removed/MOJOLEARN_SEQ_FAST_LSTM_WGRAD.patch
comptime OP_CELL_FWD_SCAN = 120
comptime OP_CELL_BWD_SCAN = 121

#: lanes per row block: H rounded up to a simdgroup multiple, at most this
comptime SCAN_MAX_H = 1024
#: the scan's shape bound on the staged row, G H <= this (kept from the
#: deleted SMEM arm's threadgroup buffer: `scan_applies` decides the path)
comptime SCAN_SMEM = 4096


def scan_applies(H: Int, G: Int) -> Bool:
    """The scan kernels' shape bound: one lane per unit, the staged row fits."""
    return H >= 1 and H <= SCAN_MAX_H and G * H <= SCAN_SMEM


def scan_tpb(H: Int) -> Int:
    return ((H + 31) // 32) * 32


def scan_block(cell: Int, H: Int) -> Int:
    """The scan launch's block width: H lanes (rounded to a simdgroup)."""
    # TOMBSTONE: MOJOLEARN_SEQ_FAST_LSTM_SCAN_WIDE (DROPPED-quality, rab10) deleted 2026-10-09 on lane/owed-deletions-D2:
    # the G H-lane width; code recoverable at b639a2bd2. Restore: git apply experiments/removed/MOJOLEARN_SEQ_FAST_LSTM_SCAN.patch
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


# ------------------------------------------------------------------ device kernels
@always_inline
def _w_lo(w: Int64) -> Int:
    return Int(Int32(w & 0xFFFFFFFF))


@always_inline
def _w_hi(w: Int64) -> Int:
    return Int(Int32((w >> 32) & 0xFFFFFFFF))


@always_inline
def _scan_args(
    p0: FP, p1: FP, p2: FP, p3: FP, p4: FP, p5: FP,
    p6: FP, p7: FP, p8: FP, p9: FP, p10: FP, p11: FP,
    i01: Int64, i23: Int64, i45: Int64,
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
    #: Since lane ag-pass5-recurrent the integers arrive packed as
    #: seq_kernel's (exec_device.mojo _pack_ii: i0 | i1 << 32, ...): the
    #: argument form every working Metal sequence kernel uses.
    return Args(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11,
                _w_lo(i01), _w_hi(i01), _w_lo(i23), _w_hi(i23), _w_lo(i45), _w_hi(i45), 0, 0, 0, 0, 0, 0,
                Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0),
                Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0))


def cell_fwd_scan_kernel(
    p0: FP, p1: FP, p2: FP, p3: FP, p4: FP, p5: FP,
    p6: FP, p7: FP, p8: FP, p9: FP, p10: FP, p11: FP,
    i01: Int64, i23: Int64, i45: Int64, n: Int64,
):
    """One block per batch row, lane u its unit: every step of the forward
    recurrence (`op_cell_fwd_h`), a device-memory barrier between steps.
    The early exit is block-uniform (block_idx only)."""
    var row = Int(block_idx.x)
    if row >= Int(n):
        return
    var a = _scan_args(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, i01, i23, i45)
    var u = Int(thread_idx.x)
    var Hn = a.i2
    var t = row * Hn + u
    if u < Hn:
        st(a.p3, t, Float32(0.0))
        st(a.p4, t, Float32(0.0))
    # TOMBSTONE: MOJOLEARN_SEQ_FAST_LSTM_SCAN_WIDE (DROPPED-quality, rab10) deleted 2026-10-09 on lane/owed-deletions-D2;
    # code recoverable at b639a2bd2. Restore: git apply experiments/removed/MOJOLEARN_SEQ_FAST_LSTM_SCAN.patch
    # TOMBSTONE: MOJOLEARN_SEQ_FAST_LSTM_SCAN_SMEM (bundle DROPPED-quality, rab10 SCAN + SMEM does not train) deleted 2026-10-09
    # on lane/owed-deletions-D2: the threadgroup-staged arm; code recoverable at b639a2bd2.
    # Restore: git apply experiments/removed/MOJOLEARN_SEQ_FAST_LSTM_SCAN_SMEM.patch
    team_barrier()
    for s in range(a.i3):
        var sa = fwd_step(a, s)
        if u < Hn:
            op_cell_fwd_h(t, sa)
        team_barrier()


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
