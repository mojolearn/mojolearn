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
from std.sys.compile import is_defined
from max.gpu import block_dim, block_idx, thread_idx
from core.accelerator_target import TARGET_HAS_APPLE_GPU

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_linear.team import team_barrier
from sequence.ops import FP, Args, add, cell_gh_chain, gates_of, ld, op_cell_fwd, st
from sequence.gemm_split import gemm_split_partial
from sequence.mlp import EPI_BIAS_ACT, gemm_epi_value

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
# Lane ag-pass5-recurrent (2026-10-19, PASS5 idea 3): under IDENTICAL the
# scan is the default for the cells the gate-count rule below picks (the
# one-gate rnn), in the hand trainer (sequence/recurrent.mojo forward /
# backward, HandStep) and the tape trainer (autograd/rules_recurrent.mojo
# rnn_layer / _layer_bwd, AutogradRnnStep) alike, both through scan_routes, so
# the two steps stay equal. The other route is the per-step chain (T launches
# of OP_CELL_FWD_H / OP_CELL_BWD_H plus 4 fills a layer). The older opt-in
# -D MOJOLEARN_IDN_SEQ_LSTM_SCAN is a no-op (MOJOLEARN_SEQ_RNN_SCAN replaces it).
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
# AMD's, and the chain's do). The -D MOJOLEARN_SEQ_RNN_SCAN_APPLE arm that put
# Apple on the scan for that check was deleted 2026-11-15. has_apple_gpu_accelerator is a
# property of the build host: an Apple build's host column takes the chain too,
# with the same bits.
# TOMBSTONE: MOJOLEARN_SEQ_RNN_SCAN_APPLE (opt-in arm putting Apple on the scan; EXP:2165, Metal scan unqualified) deleted 2026-11-15 by lane/expdel2-30-neural; code recoverable at e39daf79ad.
# Restore: git apply experiments/removed/MOJOLEARN_SEQ_RNN_SCAN_APPLE.patch
comptime SEQ_RNN_SCAN_ON_THIS_TARGET = not TARGET_HAS_APPLE_GPU
# THE ROUTE (orchestrator decision 2026-10-19, nv-04/NV_PASS5 + AMD pass 5):
# a cost rule on the cell's gate count G, the same for HandStep and
# AutogradRnnStep (both call scan_routes). The scan saves T + 1 launches and 2
# fills a layer direction, a fixed cost; but it runs one block per batch row,
# so each block walks its row's G gate folds (G x gemm_dot over K = H) and the
# cell step by step behind a barrier, with B blocks of H lanes where the chain
# spreads B H threads over the whole GPU each step. That serialized gate work
# grows with G and eats the launch saving: at G = 1 the scan wins on both
# vendors, at G = 3 and 4 it is a tie on the average (NVIDIA faster, AMD
# slower). One run per arm, s a fit, scan / chain, bits equal:
#   rnn  (G 1): NV L40S 0.83, AMD MI325X 0.88, average 0.86  -> scan
#   gru  (G 3): NV 0.87, AMD 1.11, average 0.99              -> chain (tie)
#   lstm (G 4): NV 0.88, AMD 1.15, average 1.015             -> chain (tie)
# SCAN_MAX_GATES is the one-line default: the scan for cells with
# G <= SCAN_MAX_GATES (0 = the chain everywhere, 4 = the scan everywhere).
# The force-scan / force-chain A/B arms (-D MOJOLEARN_SEQ_RNN_SCAN, _OFF) were
# deleted 2026-11-15. Apple stays on the chain (SEQ_RNN_SCAN_ON_THIS_TARGET above). Not a shape rule: G is the cell's structure,
# not a board dimension, and the same bits come out on either route.
#
# WHY THE RULE, AND THE GATED SCAN (lane rnn-scan, 2026-11-23). The rule above was measured against the FUSED chain
# (one OP_CELL_FWD_H launch a step, every thread folding its own G gate dots). Since 2026-11-19 the chain's forward
# step is the tiled GEMM plus the elementwise cell (autograd/rules_recurrent.mojo RNN_STEP_GEMM: fit gru 1.19x,
# lstm 1.28x, predict 2.6-4.5x faster than the fused step on the L40S), and the LSTM backward step is the GEMM
# (RNN_BPTT_LSTM_GEMM). So the old scan, whose lanes each walk G gate folds of K = H behind one barrier, would now
# race a much faster chain. The scan kernels below therefore (1) spread a step's projection over G H lanes of the
# row's block (one lane a gate unit, then a barrier, then one lane a unit for the cell; for G = 1 the same lane does
# both, so the tanh RNN keeps its one barrier a step and its words), lanes striding by the block width so any H
# fits one block; and (2) take the chain's fold of the step products (i6 = R, the split-K block the chain's GEMM
# launch takes at that shape, 0 = gemm_dot's chain; autograd/rules_recurrent.mojo scan_fwd_split /
# scan_bwd_split), so the scan's words are the chain's at every shape and cell: either route gives the same bits.
# GRU and LSTM stay on the chain: the arm that put them on the scan (SCAN_MAX_GATES 4) lost its A/B on the L40S
# (2026-11-24, nvc-ab-20261124T070727Z-69dac2: gru / lstm infer 4.2-4.5x slower, fit higgs 1.17-1.20; 18 cells,
# median 1.18). The scan kernels stay cell-generic (checks/native_tests/rnn_scan.mojo keeps them equal to the chain).
# TOMBSTONE: MOJOLEARN_SEQ_FAST_RNN_SCAN_GATED (LOSER: NV L40S 18 cells, median 1.18, gru/lstm infer 4.2-4.5x slower) deleted 2026-11-24 by lane/ab-outcome-1; code recoverable at d81b6ba708.
# Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_SEQ_FAST_RNN_SCAN_GATED.ab-outcome-1.patch
comptime SCAN_MAX_GATES = 1
# TOMBSTONE: MOJOLEARN_SEQ_RNN_SCAN (force-scan A/B arm; EXP:2166 TIE, the gate-count rule stays) deleted 2026-11-15 by lane/expdel2-30-neural; code recoverable at e39daf79ad.
# Restore: git apply experiments/removed/MOJOLEARN_SEQ_RNN_SCAN.patch
# TOMBSTONE: MOJOLEARN_SEQ_RNN_SCAN_OFF (force-chain A/B arm; EXP:2166, the gate-count rule stays) deleted 2026-11-15 by lane/expdel2-30-neural; code recoverable at e39daf79ad.
# Restore: git apply experiments/removed/MOJOLEARN_SEQ_RNN_SCAN_OFF.patch
# TOMBSTONE: MOJOLEARN_IDN_ALL_OFF (umbrella old-path switch, EXP:2438) read folded here (non-owner) 2026-11-15 by lane/expdel2-30-neural; code recoverable at e39daf79ad.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_ALL_OFF.expdel2-30-neural.patch
comptime SEQ_RNN_SCAN_IDN = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and SEQ_RNN_SCAN_ON_THIS_TARGET
    and SCAN_MAX_GATES > 0
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

#: the route's bound on H (scan_applies; once the lanes a row block, now a route rule only: blocks are SCAN_MAX_LANES)
comptime SCAN_MAX_H = 1024
#: the route's shape bound, G H <= this (kept from the deleted SMEM arm's threadgroup buffer). Since lane rnn-scan
#: the kernels stride their lanes, so the bound is a route rule only: it keeps every route as it was timed.
comptime SCAN_SMEM = 4096


def scan_applies(H: Int, G: Int) -> Bool:
    """The scan kernels' shape bound: one lane per unit, the staged row fits."""
    return H >= 1 and H <= SCAN_MAX_H and G * H <= SCAN_SMEM


def scan_routes(H: Int, G: Int) -> Bool:
    """Whether a layer of G gates and H units takes the scan (when
    SEQ_LSTM_SCAN compiles it in): inside the kernels' shape bound and
    G <= SCAN_MAX_GATES (the cost rule above). Every trainer route asks this one function."""
    if not scan_applies(H, G):
        return False
    return G <= SCAN_MAX_GATES


def scan_tpb(H: Int) -> Int:
    return ((H + 31) // 32) * 32


#: THE LAUNCH BOUND (lane rnn-scan-2, 2026-11-23): at most this many lanes a row block. The L40S refused the
#: forward scan's launch for a GRU layer of H 300 (block G H = 928 lanes; checks/native_tests/rnn_scan.mojo unit 20,
#: job nvc-measure-20261123T125806Z-285cae: a CUDA launch error at the enqueue, every unit of 320 lanes or fewer ran):
#: a block's threads share one register file (64 K 32-bit registers a block on NVIDIA), and the scan's step body
#: (gemm_dot's 16 + 16 staged loads, the split partials, the cell state) needs more than 64 K / 928 = 70 registers
#: a lane. 256 lanes fit at the hardware's 255-register maximum on NVIDIA, and in AMD's and Apple's limits, so no
#: shape can exceed the register file; wider rows stride (every lane's work and every word unchanged). The same
#: bound held the default tanh RNN scan at H above 256 (blocks up to 1024 lanes), which could fail the same way.
comptime SCAN_MAX_LANES = 256


def scan_fwd_block(cell: Int, H: Int) -> Int:
    """The forward scan's block width: one lane a gate unit, G H rounded to a simdgroup, at most SCAN_MAX_LANES
    (lanes stride past it)."""
    # TOMBSTONE: MOJOLEARN_SEQ_FAST_LSTM_SCAN_WIDE (DROPPED-quality, rab10) deleted 2026-10-09 on lane/owed-deletions-D2:
    # the G H-lane width; code recoverable at b639a2bd2. Restore: git apply experiments/removed/MOJOLEARN_SEQ_FAST_LSTM_SCAN.patch
    # (lane rnn-scan: the G H width returns for the forward projection only, in the IDENTICAL kernel, with every
    # gate unit's fold unchanged; the dropped arm's quality loss was the Metal Args defect noted at _scan_args.)
    return min(scan_tpb(gates_of(cell) * H), SCAN_MAX_LANES)


def scan_bwd_block(H: Int) -> Int:
    """The backward scan's block width: one lane a unit (its dh fold over K = G H is one chain), at most
    SCAN_MAX_LANES (lanes stride past it)."""
    return min(scan_tpb(H), SCAN_MAX_LANES)


# ------------------------------------------------------------------ step projection
@always_inline
def scan_gh_value(a: Args, b: Int, n: Int, R: Int) -> Float32:
    """GH[b, n] of a forward step in the fold the per-step chain takes at the layer's shape (lane rnn-scan): R = 0
    the fused step's and the unsplit GEMM's chain (sequence/ops.mojo cell_gh_chain); R > 0 the split-K order of
    blocks of R terms (sequence/gemm_split.mojo gemm_split_partial, the partials added in ascending order from +0.0,
    then op_gk_split's bias epilogue gemm_epi_value), the words of OP_GEMM_EPI's split launch. A(m, k) = h_prev
    (p3, row stride H), B(k, n) = W_hh[n, k] (p7), b_hh p8; i2 = H."""
    if R == 0:
        return cell_gh_chain(a, b, n)
    var H = a.i2
    var acc = Float32(0.0)
    for s in range((H + R - 1) // R):
        acc = add(acc, gemm_split_partial(a.p3, H, 1, a.p7, 1, H, H, b, n, s, R))
    return gemm_epi_value(acc, 0, n, EPI_BIAS_ACT, 0, a.p8, Float32(0.0), Float32(0.0))


# ------------------------------------------------------------------ per-step args
@always_inline
def fwd_step(a: Args, s: Int) -> Args:
    """`OP_CELL_FWD_H`'s arguments of step s from the scan's base arguments
    (step 0's pointers; i1 B, i2 H, i3 T).

    Sequence lengths (lane seq-blocks-3): i5 = 1 and p10 the rows' lengths
    make every step the masked cell (i9 = 1, i8 the step: op_cell_fwd copies a
    row's state forward past its length), as the per-step chain's launches.
    Without them i9 is 0 and the step is the unmasked cell, word for word.

    The reverse direction (i4 = 1; lane nn-recurrent): the base pointers are step 0's, which reads time T - 1,
    and every pointer walks back one row block a step, so step s reads time T - 1 - s, and i8 (the mask's
    position) is that time: a row's padded times come first and hold its zero state, then its own steps from
    its last valid one down to 0. The forward direction (i4 = 0) computes the same addresses as before."""
    var B = a.i1
    var H = a.i2
    var GH = gates_of(a.i0) * H
    var k = -s if a.i4 != 0 else s
    var r = a
    r.p0 = a.p0 + k * B * GH
    r.p1 = a.p1 + k * B * GH
    r.p2 = a.p2 + k * B * GH
    r.p3 = a.p3 + k * B * H
    r.p4 = a.p4 + k * B * H
    r.p5 = a.p5 + k * B * H
    r.p6 = a.p6 + k * B * H
    r.i8 = a.i3 - 1 - s if a.i4 != 0 else s
    r.i9 = a.i5
    return r


# ------------------------------------------------------------------ host bodies
def op_cell_fwd_scan(row: Int, a: Args):
    """Row `row`'s whole forward recurrence, as the per-step launches ran it:
    h_0 = c_0 = 0, then every step's projection (scan_gh_value, i6 = R) and its cells, units in ascending order.
    A step's projection reads only h_prev and stores only GH, which only that step's cells read, so this order
    gives the device kernel's words."""
    var H = a.i2
    var GH = gates_of(a.i0) * H
    for u in range(H):
        st(a.p3, row * H + u, Float32(0.0))
        st(a.p4, row * H + u, Float32(0.0))
    for s in range(a.i3):
        var sa = fwd_step(a, s)
        for n in range(GH):
            st(sa.p1, row * GH + n, scan_gh_value(sa, row, n, a.i6))
        for u in range(H):
            op_cell_fwd(row * H + u, sa)


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
    i01: Int64, i23: Int64, i45: Int64, i67: Int64,
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
                _w_lo(i01), _w_hi(i01), _w_lo(i23), _w_hi(i23), _w_lo(i45), _w_hi(i45),
                _w_lo(i67), _w_hi(i67), 0, 0, 0, 0,
                Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0),
                Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0))


def cell_fwd_scan_kernel(
    p0: FP, p1: FP, p2: FP, p3: FP, p4: FP, p5: FP,
    p6: FP, p7: FP, p8: FP, p9: FP, p10: FP, p11: FP,
    i01: Int64, i23: Int64, i45: Int64, i67: Int64, n: Int64,
):
    """One block per batch row: every step of the forward recurrence, a device-memory barrier between steps.
    A step is the projection (scan_gh_value, one lane a gate unit, i6 = R its fold) and, after a barrier, the
    cell (op_cell_fwd, one lane a unit); lanes stride by the block width. For a one-gate cell the lane that
    projects unit u is the lane that runs its cell, so no barrier sits between the two (the kernel's form before
    lane rnn-scan). The early exit and every branch on a barrier are block-uniform (block_idx and kernel
    arguments only)."""
    var row = Int(block_idx.x)
    if row >= Int(n):
        return
    var a = _scan_args(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, i01, i23, i45, i67)
    var lane = Int(thread_idx.x)
    var W = Int(block_dim.x)
    var Hn = a.i2
    var GH = gates_of(a.i0) * Hn
    var R = a.i6
    var u = lane
    while u < Hn:
        st(a.p3, row * Hn + u, Float32(0.0))
        st(a.p4, row * Hn + u, Float32(0.0))
        u += W
    # TOMBSTONE: MOJOLEARN_SEQ_FAST_LSTM_SCAN_WIDE (DROPPED-quality, rab10) deleted 2026-10-09 on lane/owed-deletions-D2;
    # code recoverable at b639a2bd2. Restore: git apply experiments/removed/MOJOLEARN_SEQ_FAST_LSTM_SCAN.patch
    # TOMBSTONE: MOJOLEARN_SEQ_FAST_LSTM_SCAN_SMEM (bundle DROPPED-quality, rab10 SCAN + SMEM does not train) deleted 2026-10-09
    # on lane/owed-deletions-D2: the threadgroup-staged arm; code recoverable at b639a2bd2.
    # Restore: git apply experiments/removed/MOJOLEARN_SEQ_FAST_LSTM_SCAN_SMEM.patch
    team_barrier()
    var one_gate = GH == Hn
    for s in range(a.i3):
        var sa = fwd_step(a, s)
        var g = lane
        while g < GH:
            st(sa.p1, row * GH + g, scan_gh_value(sa, row, g, R))
            g += W
        if not one_gate:
            team_barrier()
        u = lane
        while u < Hn:
            op_cell_fwd(row * Hn + u, sa)
            u += W
        team_barrier()
