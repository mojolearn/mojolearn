# SPDX-License-Identifier: Apache-2.0
# Recurrent trainer on the tape (rnn_fit's only backward since 2026-10-19).
#
# The hand-written recurrence kernels stay: OP_CELL_FWD_H / OP_CELL_BWD_H
# (sequence/ops.mojo) and the scan OP_CELL_FWD_SCAN / OP_CELL_BWD_SCAN
# (sequence/recurrent_scan.mojo). What moves to the tape is the model-level
# assembly of sequence/recurrent.mojo::backward: the head GEMM and column sum,
# the zero-padded h_T delta, the per-layer input-projection deltas and the
# hand-chained layer order. Three custom rules carry the existing launches:
#
#   RNN_LINEAR     Y = X W^T (+ b), W in the PyTorch [out, in] layout. The
#                  backward is the existing dW / db / dX launches, including
#                  the SEQ_WGRAD_BLOCKED order for the input projections.
#   RNN_LAYER      one scanned layer: gx [T B, G H] -> h_1..h_T [T B, H] with
#                  W_hh, b_hh. The backward is the existing cell (or scan)
#                  backward plus the dW_hh / db_hh launches.
#   RNN_LAST_STEP  the view h_T of a layer's output; its backward places the
#                  upstream rows at step T - 1 of a zeroed sequence gradient.
#                  (The trainer does not use it: its head is an RNN_LINEAR
#                  over the last B rows of the top layer's sequence, see
#                  rnn_head, which writes dh_T in place like the hand path.)
#
# Every launch, argument and fold order is the hand path's. No tensor is
# used twice, so every gradient is a first contribution written in place
# and no ACCUMULATE runs; parameter gradients land in Gr (variable_into),
# the loss derivative w.dy is the head's gradient storage (custom_into),
# and a rule's destination is zero-filled only where the hand path fills
# (zeroed_destination). The gradients in Gr carried the same bits as the
# deleted hand step (HandStep) on Apple, NVIDIA and AMD.
# Scratch comes from the trainer's Work buffers, not tape.reserve, so the
# tape allocates only node gradients and contributions.

from autograd.tape import Tape, Tensor, BackwardRules
from sequence.exec_trait import Exec
from sequence.ops import (
    FP,
    Args,
    CELL_LSTM,
    OP_CELL_BWD,
    OP_CELL_BWD_H,
    OP_CELL_BWD_H_MASK,
    OP_CELL_FWD,
    OP_CELL_FWD_H,
    OP_COPY,
    OP_GEMM,
    OP_GEMM_EPI,
    OP_SEQ_IOTA,
    OP_SEQ_OUT,
    OP_SOFTMAX,
    dummy_ptr,
    gates_of,
)
from sequence.recurrent_scan import OP_CELL_BWD_SCAN, OP_CELL_FWD_SCAN, SEQ_LSTM_SCAN, scan_routes
from sequence.recurrent import (
    Net, Work, RnnStep, SEQ_WGRAD_BLOCKED, TASK_CE, WGRAD_IDN_SCRATCH, bias_rows, fill, gather_rows, gather_seq, gemm,
)
from autograd.kernels.recurrent import colsum, wgrad_block, wgrad_blocked
from sequence.mlp import EPI_BIAS_ACT
from sequence.gemm_split import gemm_split_block, gemm_split_shape
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

# The recurrent trainer's backward runs on the tape (promoted 2026-10-19). One run per arm, HIGGS
# 1M rows, 16-step sequences, hidden 64, b512, 1 epoch, identical: NVIDIA L40S lstm 0.1419 -> 0.1397 s, gru 0.1256 ->
# 0.1129, rnn 0.0689 -> 0.0645; AMD MI325X lstm 0.1234 -> 0.1236, gru 0.1053 -> 0.1054, rnn 0.0709 -> 0.0718. Bits equal
# to the hand step on Apple, NVIDIA and AMD (same launches, same arguments).
# TOMBSTONE: MOJOLEARN_AUTOGRAD_RNN_OFF (hand glue, superseded by the tape) deleted 2026-10-19 by e88ae4e38; code recoverable at e88ae4e38^.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_RNN_OFF.patch
# MOJOLEARN_AUTOGRAD_RNN_OFF (the hand step) and the old opt-in spelling MOJOLEARN_AUTOGRAD_RNN are refused in
# core/six_lane_experiment_guards.mojo.

# Tags in a range of their own, clear of autograd/layers.mojo (LAYER_NORM = 1),
# so one NeuralRules can later dispatch both families.
comptime RNN_LINEAR = 16
comptime RNN_LAYER = 17
comptime RNN_LAST_STEP = 18

# The input projection adds b_ih to GX in its own launch unless the per-step
# kernel adds it (NN42 fused, per-step path only); the deleted hand forward
# (sequence/recurrent.mojo::forward, lane X3) used the same rule.
# (Sweep 2026-11-15: NN42_INPUT_BIAS_FUSED is gone, so the input bias is always in GX; the name stays for
# sequence/checks/seams_check.mojo.)
comptime RNN_INPUT_BIAS_IN_GX = True

# Lane fx-rnn-cnn-fit (2026-11-19), default path, same bits: a projection's bias is the GEMM launch's own epilogue
# (OP_GEMM_EPI with EPI_BIAS_ACT and no activation: op_gemm for the cell, then op_bias on the word it stored,
# sequence/mlp.mojo op_gemm_epi; the launch the hand MLP has used for every layer since 2026-09-28) where it was a
# GEMM launch and then an OP_BIAS launch over the same R x out words. The product takes the route and the fold order
# OP_GEMM took: DeviceExec and HostExec ask the same split-K rule and the same tile route for both opcodes; only the
# few-cells-over-a-long-K cooperative kernel is OP_GEMM's alone, so that shape keeps the two launches (_coop_shape).
# One launch and one pass over R x out words fewer per projection per step (the input projection of every layer and
# the head). IDENTICAL only: FAST's `gemm` has its own split for some shapes (sequence/recurrent.mojo gemm).
comptime RNN_BIAS_EPI = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL

# THE RECURRENT PRODUCTS ON THE LANE'S OWN GEMM (lane fx-rnn-cnn-fit, 2026-11-19), decided per site by A/B round 2
# (lane fx-verdicts4, 2026-11-19): the forward step is the default, the backward step and the weight gradients are a
# narrowed opt-in arm.
#
# The old form: the per-step chain (cells of more gates than the scan takes) runs each time step as ONE launch of one
# thread per (row, unit): the thread folds its own G gate dots of K = H terms straight from device memory, then the
# cell (OP_CELL_FWD_H; the backward one dot of K = G H, OP_CELL_BWD_H), a fusion made for Apple's launch cost
# (2026-09-28); and the two weight gradients of a layer are the blocked order of nr-small D3 (wgrad_blocked: two
# launches each, sqrt(K) blocks, one thread a partial).
#
# 1. FORWARD STEP, the IDENTICAL default on every column (RNN_STEP_GEMM; confirmed twice, its _OFF switch deleted
#    2026-11-20, round 3 below; the names MOJOLEARN_SEQ_FAST_RNN_STEP_GEMM and _OFF are refused,
#    core/experiment_guards_neural.mojo):
#      GH_s = h_{s-1} W_hh^T + b_hh   [B, G H] over K = H   (OP_GEMM_EPI, bias epilogue, on the route and in the fold
#      the lane's launch rule gives the shape: sequence/gemm_split.mojo, sequence/gemm_tiled.mojo), then the
#      elementwise cell OP_CELL_FWD.
#    Rule (rnn_step_gemm): at least RNN_STEP_GEMM_MIN_FMA multiply-adds a step. Cost reasoning, not a shape: the second
#    launch costs one launch floor (3-8 us on the L40S), which the fused fold spends on about 1 M multiply-adds; below
#    that the fused launch is the cheaper form on any hardware with a launch floor.
#    BITS against the fused launch: a step that is not a split-K shape (K = H <= 64, or enough tiles) SAME (gemm_dot's
#    chain, the fused body's); a split-K shape (K = H > 64 with few cells or tiles) CHANGE, to
#      R = gemm_split_block(K); p_s = +0.0; p_s = fma(A(m, k), B(k, n), p_s) over k = s R .. min(K, (s + 1) R) - 1;
#      acc = +0.0; acc = acc + p_s, s ascending; then + b_hh[n]
#    on every column together: the host column takes the same rule for the same opcodes (HostExec.launch), and fit and
#    predict share this forward, so they agree. The seams check runs device against host at such shapes (5504
#    step_gemm).
#    EVIDENCE, A/B round 2: NVIDIA L40S (host bd9c355c19c0), wheel 0c68d6cd27 and its B set, base and arm in ONE pod
#    job (tag nvc-ab-20261119T154232Z-0642a5), pairs whose two cells ran on one NUMA node only. Predict is the forward
#    alone (chunks of 4096 rows, T 24, H 64), base -> arm ms: gru-clf taxi 38.98 -> 14.81 (0.38), gru-reg 19.53 -> 6.89
#    and 19.52 -> 6.83 (0.35, 0.35), lstm-clf 50.97 -> 18.74 and 50.69 -> 18.80 (0.37, 0.37), lstm-reg 25.35 -> 9.10
#    and 25.44 -> 9.14 (0.36, 0.36); quality equal to 1e-8 (the arm as timed also changed the backward's bits). rnn
#    rows take the scan, not this chain: predict 1.01-1.04 (their pairs were on different nodes).
#    NOT TIMED ALONE: the forward step inside a fit (B 256), which the arm timed only together with sites 2 and 3
#    below. By subtraction on the board rows (the rnn rows change in site 3 alone: +0.38 ms a step for 64 x 64 cells;
#    gru 3 times the cells and +1.15 ms; lstm 4 times and +1.38 ms) sites 1 and 2 together are 0 to -0.14 ms of a
#    2 ms step. OWED: the _OFF arm on the fit and predict rows (owed-pre0867.json rnn-step-gemm-off).
#
# 2 and 3. BACKWARD STEP AND WEIGHT GRADIENTS, the former arm MOJOLEARN_SEQ_FAST_RNN_BPTT_GEMM (decided by A/B
#    round 3, below: site 2 is the default for the LSTM cell, the rest is deleted). As the arm had them:
#      backward step s   dh += dGH_{s+1} W_hh   [B, H] over K = G H   (OP_GEMM, accumulate), then the elementwise
#                        cell derivative OP_CELL_BWD (step T - 1 has no product: the fused launch); rule
#                        rnn_bptt_gemm, the forward step's cost rule
#      dW_hh, dW_ih      [G H, H] and [G H, D_in] over K = T B as ONE OP_GEMM over the stacked steps, ONLY for a
#                        product the launch rule does not split (not gemm_split_shape: the tile, with operand reuse);
#                        a split-K product keeps the blocked order (_wgrad)
#    WHY NARROWED. The arm as first written (lane fx-rnn-cnn-fit) took the GEMM for both weight gradients at every
#    shape and lost on every fit whose weight gradient is a split-K shape: fit, base -> arm ms, same-node pairs:
#    gru-clf taxi 467.2 -> 1115.7 (2.39), gru-reg 463.7 -> 1111.1 and 463.2 -> 1111.9 (2.40), lstm-clf 572.7 -> 1352.1
#    (2.36), lstm-reg 566.7 -> 1340.4 (2.37); neural gru-fit 120.0 -> 384.6 (3.2), lstm-fit 146.1 -> 490.0 (3.4),
#    rnn-fit 55.7 -> 154.3 (2.8); driver lstm T 48 H 64 879 -> 2575 (2.93), rnn H 64 137 -> 318 (2.32; the scan cell:
#    its ONLY change is the two weight gradients, so the loss is theirs). Cause: at H 64 the output [G H, H] is under 8
#    tiles of 64 x 64, the launch rule splits K = T B = 6144 into 96 blocks and gives every cell 128 threads of a
#    256-thread block (two cells a block; one at T 48), 66 GFLOP/s where the blocked order's sqrt(K) partials fill the
#    device. Where the gradient is NOT a split shape the same arm won: driver fit lstm H 128 1167 -> 720 (0.62), gru H
#    128 911 -> 594 (0.65), 2-layer gru H 96 on 12 features 1871 -> 1622 (0.87); and lstm H 16 at batch 64 (below the
#    step rule: weight gradients only, split) 254.5 -> 236.2 (0.93, one call).
#    The narrowed composition (forward and backward steps on the GEMM, tile-routed weight gradients on the GEMM,
#    split-K gradients blocked) was never timed as such: that is the owed arm, on the fit rows and the driver shapes.
#    BITS under the define: backward step, not a split-K shape SAME, split-K shape CHANGE (the fold above with c0 =
#    the stored dh); dW with T B <= 512 rows SAME (the default is already this GEMM); dW with T B > 512 rows and not
#    a split shape CHANGE (blocks of wgrad_block(T B) rows -> one fold, k ascending); split shapes SAME as the
#    default. The oracle follows under the same define (sequence/checks/oracle.mojo _o_bptt_block).
# TOMBSTONE: MOJOLEARN_SEQ_FAST_RNN_STEP_GEMM, its weight-gradient GEMM at split-K shapes (LOSS, numbers above) deleted
# 2026-11-19 by lane/fx-verdicts4; code recoverable at 8f3f639b13.
# Restore: git apply experiments/removed/MOJOLEARN_SEQ_FAST_RNN_STEP_GEMM.patch; record in docs/TOMBSTONES.md.
#
# A/B ROUND 3, 2026-11-20 (lane fx-verdicts5; NVIDIA L40S host 2ef132a6de04, wheel 28a780604c and its B sets, base and
# arm back to back in one pod job, tag nvc-ab-20261120T063324Z-c630ad, every algos worker on the GPU's NUMA node 0;
# the row's two bases in the job agree to 0.6 % on every fit). Base -> arm ms.
# rnn-step-gemm-off (B = the fused forward launch at every shape), digests and quality equal (H 64: the same-bits
#   class): fit gru-clf 391.7 -> 466.0 and 392.4 -> 466.1 (1.19), gru-reg 388.4 -> 462.5 and 388.3 -> 462.7 (1.19),
#   lstm-clf 446.8 -> 573.5 and 446.7 -> 570.9 (1.28), lstm-reg 442.2 -> 566.6 and 442.3 -> 566.9 (1.28), neural
#   gru-fit 91.32 -> 118.56 (1.30), lstm-fit 109.96 -> 145.12 (1.32); predict gru 3.37-4.39, lstm 3.69-4.54; rnn rows
#   (the scan) 0.99-1.02. Driver fit: lstm T 48 H 64 672.0 -> 876.5 (1.30), lstm H 128 728.6 -> 1163.4 (1.60), gru
#   H 128 570.1 -> 906.5 (1.59), 2-layer gru H 96 1451.0 -> 1829.8 (1.26); controls rnn H 64 and lstm H 16 1.00.
#   CONFIRMED a second time (round 2 was the predict rows; this round also times the forward step inside a fit):
#   the forward step's GEMM stands and its switch is deleted.
# TOMBSTONE: MOJOLEARN_SEQ_FAST_RNN_STEP_GEMM_OFF (old path's switch; the promotion confirmed by A/B rounds 2 and 3, numbers above) deleted 2026-11-20 by lane/fx-verdicts5; code recoverable at e25e35dab8.
# Restore: git apply experiments/removed/MOJOLEARN_SEQ_FAST_RNN_STEP_GEMM_OFF.patch; record in docs/TOMBSTONES.md.
#   What stays of the old path: OP_CELL_FWD_H itself, for a step below the rule's floor and for fast mode.
# rnn-bptt-gemm (B = sites 2 and 3 above), a bit version, quality equal to 1e-8 on every cell: fit lstm-clf 446.4 ->
#   428.9 and 446.6 -> 426.8 (0.961, 0.956), lstm-reg 441.6 -> 420.6 and 441.7 -> 422.1 (0.952, 0.956); gru-clf
#   393.7 -> 415.8 and 391.5 -> 417.8 (1.056, 1.067), gru-reg 387.9 -> 406.9 and 390.8 -> 409.5 (1.049, 1.048); rnn
#   rows 1.000-1.004; neural gru-fit 0.985, lstm-fit 1.002, rnn-fit 0.991; predicts 1.00 (two single-call outliers,
#   1.11 and 0.71, on rows whose bases differ as much). Driver fit: lstm T 48 H 64 672.0 -> 637.5 (0.949), lstm H 128
#   728.6 -> 714.8 (0.981), gru H 128 570.1 -> 592.5 (1.039), 2-layer gru H 96 1451.0 -> 1509.5 (1.040); controls
#   1.00. At H 64 both weight gradients are split shapes, so the board rows time SITE 2 ALONE: the backward step's
#   GEMM is faster for the LSTM cell at every LSTM shape timed and slower for the GRU cell at every GRU shape timed
#   (K = G H of 192, 288, 384 lose where the LSTM's 256 and 512 win, so it is the cell and not K; the fused GRU
#   backward's own work a thread was not profiled). The arm's rule (no gru / lstm fit slower) is not met as written.
#   DECIDED BY SITE AND CELL: site 2 is the default for the LSTM cell (RNN_BPTT_LSTM_GEMM below; its effect on the
#   LSTM rows is what this A/B timed, the GRU and tanh cells keep the fused launch and are untouched); site 3 (a
#   non-split weight gradient as one OP_GEMM) has no cell of its own in either round (active only at the two H 128 /
#   H 96 driver cases, 0.981 and 1.04 together with site 2) and is deleted with the arm's name.
#   BITS CHANGE, every column and the host column together (the same launches): an LSTM backward step that is a
#   split-K shape (the fold of site 1 with c0 = the stored dh), from the floor up. Weight gradients: unchanged.
# TOMBSTONE: MOJOLEARN_SEQ_FAST_RNN_BPTT_GEMM (SPLIT, A/B round 3 NVIDIA L40S, wheel 28a780604c, job nvc-ab-20261120T063324Z-c630ad: lstm fits 0.95-0.96, gru fits 1.05-1.07; the LSTM backward step promoted, the GRU backward step and the non-split weight-gradient GEMM deleted) deleted 2026-11-20 by lane/fx-verdicts5; code recoverable at e25e35dab8.
# Restore: git apply experiments/removed/MOJOLEARN_SEQ_FAST_RNN_BPTT_GEMM.patch; record in docs/TOMBSTONES.md.
comptime RNN_STEP_GEMM = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
#: The LSTM backward step on the lane's GEMM (site 2 above; default since 2026-11-20), in `identical` mode.
# TOMBSTONE: MOJOLEARN_SEQ_FAST_RNN_BPTT_LSTM_GEMM_OFF (removed: decided A/B round 3 NVIDIA L40S 2ef132a6de04, lstm fits
# 0.95-0.96, driver 0.949 / 0.981, quality equal; its old form, the fused backward launch for the LSTM cell too) deleted
# 2026-11-23 by lane/ab-prep-neural; code recoverable at 4d76a9e16e.
# Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_SEQ_FAST_RNN_BPTT_LSTM_GEMM_OFF.patch; record in docs/TOMBSTONES.md.
comptime RNN_BPTT_LSTM_GEMM = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
comptime RNN_STEP_GEMM_MIN_FMA = 1 << 20


@always_inline
def _coop_shape(M: Int, N: Int, K: Int) -> Bool:
    """The shape DeviceExec._launch gives OP_GEMM's cooperative kernel (few
    cells over a long K); the epilogue opcodes never take it."""
    return M * N <= 1024 and K >= 32768


@always_inline
def rnn_step_gemm(M: Int, N: Int, K: Int) -> Bool:
    """Whether a time step's M x N x K hidden projection of the forward
    runs as its own GEMM launch (RNN_STEP_GEMM, the rule above)."""
    comptime if not RNN_STEP_GEMM:
        return False
    return M * N * K >= RNN_STEP_GEMM_MIN_FMA


@always_inline
def rnn_bptt_gemm(cell: Int, M: Int, N: Int, K: Int) -> Bool:
    """Whether a backward time step's M x N x K product runs as its own GEMM
    launch: the LSTM cell, at the forward step's cost rule
    (RNN_BPTT_LSTM_GEMM, the round 3 note above)."""
    comptime if not RNN_BPTT_LSTM_GEMM:
        return False
    return cell == CELL_LSTM and M * N * K >= RNN_STEP_GEMM_MIN_FMA


@always_inline
def scan_fwd_split(B: Int, GH: Int, H: Int) -> Int:
    """The fold of the forward step projection [B, G H] over K = H as the per-step chain below takes it, for the
    scan (its i6, lane rnn-scan): the split-K block gemm_split_block(H) where the chain runs the step as its own
    GEMM launch (rnn_step_gemm) and the launch rule splits that product (gemm_split_shape, the rule DeviceExec and
    HostExec both ask for OP_GEMM_EPI over a dense C); else 0, gemm_dot's chain (the fused step's and the unsplit
    GEMM's). So the scan's GH words are the chain's at every shape, on every column."""
    if rnn_step_gemm(B, GH, H) and gemm_split_shape(B, GH, H):
        return gemm_split_block(H)
    return 0


@always_inline
def scan_bwd_split(cell: Int, B: Int, H: Int, GH: Int) -> Int:
    """The fold of a backward step's dh += dGH_{s+1} W_hh [B, H] over K = G H as the per-step chain takes it, for
    the scan (its i6, lane rnn-scan): gemm_split_block(G H) where the chain runs it as its own OP_GEMM launch
    (rnn_bptt_gemm: the LSTM cell above the step floor) and the launch rule splits it; else 0, gemm_dot's chain
    from the stored direct part (the fused step's and the unsplit GEMM's accumulate)."""
    if rnn_bptt_gemm(cell, B, H, GH) and gemm_split_shape(B, H, GH):
        return gemm_split_block(GH)
    return 0


def _gemm_bias[E: Exec](
    mut ex: E, A: FP, Bm: FP, bias: FP, C: FP, M: Int, N: Int, K: Int,
    sam: Int, sak: Int, sbk: Int, sbn: Int,
) raises:
    """C [M, N] dense = A B + bias[n] in one launch: `gemm` (no accumulate)
    with op_bias as its epilogue (sequence/mlp_fit.mojo gemm_epi's layout)."""
    var a = Args()
    a.p0 = A
    a.p1 = Bm
    a.p2 = C
    a.p3 = bias
    a.i0 = M
    a.i1 = N
    a.i2 = K
    a.i3 = sam
    a.i4 = sak
    a.i5 = sbk
    a.i6 = sbn
    a.i8 = N
    a.i9 = EPI_BIAS_ACT
    a.i10 = 0
    ex.launch[OP_GEMM_EPI](a, M * N)


def _copy[E: Exec](mut ex: E, dst: FP, src: FP, n: Int) raises:
    var a = Args()
    a.p0 = src
    a.p1 = dst
    ex.launch[OP_COPY](a, n)


# ------------------------------------------------------------------ forward records
def rnn_linear[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, weight: Tensor, bias: Tensor, output: FP,
    add_bias: Bool, blocked_wgrad: Bool, need_dx: Bool, w: Work,
) raises -> Tensor:
    """Y [R, out] = X [R, in] W^T + b into `output`, recorded as RNN_LINEAR.
    add_bias=False leaves b out of Y (a later kernel adds it) while db is
    still the column sum of dY. need_dx=False skips dX (X needs no gradient)."""
    var R = tape.rows(x)
    var saved = _linear_record(tape, ex, x, weight, bias, output, 0, R, add_bias, blocked_wgrad, need_dx, w)
    return tape.custom(ex, [x, weight, bias], output, R, tape.rows(weight), RNN_LINEAR, saved)


def _linear_record[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, weight: Tensor, bias: Tensor, output: FP,
    first_row: Int, R: Int, add_bias: Bool, blocked_wgrad: Bool, need_dx: Bool, w: Work,
) raises -> Args:
    # Y = X[first_row : first_row + R] W^T (+ b); returns the rule's Args.
    var din = tape.cols(x)
    var dout = tape.rows(weight)
    if tape.cols(weight) != din or tape.rows(bias) != 1 or tape.cols(bias) != dout:
        raise Error("recurrent linear: weight must be (out, in) and bias (1, out)")
    if first_row < 0 or R <= 0 or first_row + R > tape.rows(x):
        raise Error("recurrent linear: rows out of range")
    if not need_dx and tape.requires_grad(x):
        # the rule leaves dX unwritten (and unzeroed): only for an input
        # that needs no gradient
        raise Error("recurrent linear: need_dx=False needs an input without a gradient")
    var with_bias = False
    comptime if RNN_BIAS_EPI:
        if add_bias and not _coop_shape(R, dout, din):
            _gemm_bias(ex, tape.value(x) + first_row * din, tape.value(weight), tape.value(bias), output,
                       R, dout, din, din, 1, 1, din)
            with_bias = True
    if not with_bias:
        gemm(ex, tape.value(x) + first_row * din, tape.value(weight), output, R, dout, din, din, 1, 1, din, False, dout)
        if add_bias:
            bias_rows(ex, output, tape.value(bias), output, R, dout)
    var saved = Args()
    saved.p0 = w.wsplit
    saved.p1 = w.one
    saved.i0 = 1 if need_dx else 0
    saved.i1 = 1 if blocked_wgrad else 0
    saved.i2 = first_row
    saved.i3 = R
    # dX covers only rows [first_row, first_row + R) of x: the other rows
    # of x's gradient are zero, so that destination is zero-filled first.
    saved.i4 = 1 if need_dx and (first_row != 0 or R != tape.rows(x)) else 0
    return saved


def rnn_head[E: Exec](
    mut tape: Tape, mut ex: E, seq: Tensor, weight: Tensor, bias: Tensor, B: Int,
    output: FP, grad_destination: FP, w: Work,
) raises -> Tensor:
    """Y [B, out] = h_T W^T + b into `output`, h_T the last B rows of the
    time-major sequence `seq` [T B, H], recorded as RNN_LINEAR on `seq`
    itself; Y's gradient is `grad_destination` (the loss derivative, used
    as the backward seed in place). The backward zero-fills seq's gradient
    and writes dh_T into its last B rows (recurrent.backward's fill + GEMM),
    so no h_T view, gradient copy or seed copy runs."""
    var rows = tape.rows(seq)
    if B <= 0 or rows % B != 0:
        raise Error("recurrent head: rows must be a multiple of the batch")
    var saved = _linear_record(tape, ex, seq, weight, bias, output, rows - B, B, True, False, True, w)
    return tape.custom_into(ex, [seq, weight, bias], output, grad_destination, B, tape.rows(weight),
                            RNN_LINEAR, saved)


def layer_fwd_scan[E: Exec](
    mut ex: E, cell: Int, T: Int, B: Int, H: Int, gxp: FP, gh: FP, act: FP, hall: FP, call: FP,
    whh: FP, bhh: FP, b_ih: FP, lens: FP, masked: Bool, reverse: Bool,
) raises:
    """A layer's forward recurrence as ONE launch (OP_CELL_FWD_SCAN, one block a batch row): gx [T B, G H] -> the
    states in hall / call (h_0 = 0 first; reverse: after the states), GH in gh, the gate activations in act. i6 is
    the chain's fold of the step projection at this shape (scan_fwd_split), so the words are layer_fwd_chain's."""
    var GH = gates_of(cell) * H
    var sa = Args()
    # step 0's pointers: time 0 (forward), or time T - 1 (reverse: h_prev the zero rows after the states)
    var t0 = T - 1 if reverse else 0
    sa.p0 = gxp + t0 * B * GH
    sa.p1 = gh + t0 * B * GH
    sa.p2 = act + t0 * B * GH
    sa.p3 = hall + (T if reverse else 0) * B * H
    sa.p4 = call + (T if reverse else 0) * B * H
    sa.p5 = hall + (T - 1 if reverse else 1) * B * H
    sa.p6 = call + (T - 1 if reverse else 1) * B * H
    sa.p7 = whh
    sa.p8 = bhh
    # never read on this route (b_ih is in gx); bound to live
    # storage, not dummy_ptr (the Metal rule)
    sa.p9 = b_ih
    sa.p10 = lens if masked else gxp
    sa.p11 = whh
    sa.i0 = cell
    sa.i1 = B
    sa.i2 = H
    sa.i3 = T
    sa.i4 = 1 if reverse else 0
    sa.i5 = 1 if masked else 0
    # the chain's fold of the step projection at this shape (lane rnn-scan): the same GH words either route
    sa.i6 = scan_fwd_split(B, GH, H)
    ex.launch[OP_CELL_FWD_SCAN](sa, B)


def layer_fwd_chain[E: Exec](
    mut ex: E, cell: Int, T: Int, B: Int, H: Int, gxp: FP, gh: FP, act: FP, hall: FP, call: FP,
    whh: FP, bhh: FP, b_ih: FP, lens: FP, masked: Bool, reverse: Bool, h0_clean: Bool,
) raises:
    """The same forward as the per-step chain: per time step the projection (its own GEMM launch above the step
    rule, RNN_STEP_GEMM, else fused into the cell) and the cell. The buffers as layer_fwd_scan's."""
    var GH = gates_of(cell) * H
    # Lane fx-rnn-cnn-fit (2026-11-19): h_0 and c_0 are rows [0, B) of the two buffers, which nothing but
    # this fill (or the scan's own zero stores) writes while the batch size stays B: step s stores rows
    # [(s + 1) B, (s + 2) B). Only a forward of FEWER rows writes below B H, so the caller passes h0_clean when
    # the previous forward on this Work had at least B rows (AutogradRnnStep, rnn_predict): the two fills then
    # run once after an epoch's short last batch, not every step. The same zeros, so the same bits.
    if reverse:
        # the reverse direction's zero state is rows [T B, (T + 1) B) (lane nn-recurrent)
        fill(ex, hall + T * B * H, B * H, Float32(0.0))
        fill(ex, call + T * B * H, B * H, Float32(0.0))
    elif not h0_clean:
        fill(ex, hall, B * H, Float32(0.0))
        fill(ex, call, B * H, Float32(0.0))
    var step_gemm = False
    comptime if RNN_STEP_GEMM:
        step_gemm = rnn_step_gemm(B, GH, H)
    for s in range(T):  # driver-loop: over time steps (two launches per step: RNN_STEP_GEMM; one below its rule)
        # the time of step s, and the rows of its previous and its own state (reverse: lane nn-recurrent)
        var tm = T - 1 - s if reverse else s
        var prev = tm + 1 if reverse else s
        var own = tm if reverse else s + 1
        var a = Args()
        a.p0 = gxp + tm * B * GH
        a.p1 = gh + tm * B * GH
        a.p2 = act + tm * B * GH
        a.p3 = hall + prev * B * H
        a.p4 = call + prev * B * H
        a.p5 = hall + own * B * H
        a.p6 = call + own * B * H
        a.p7 = whh
        a.p8 = bhh
        a.p9 = b_ih
        a.i5 = 0
        a.i0 = cell
        a.i1 = B
        a.i2 = H
        if masked:
            # a row past its length copies its state forward (op_cell_fwd)
            a.p10 = lens
            a.i8 = tm
            a.i9 = 1
        if step_gemm:
            # GH_s = h_{s-1} W_hh^T + b_hh where the fused body stores it, then the cell alone
            _gemm_bias(ex, a.p3, whh, bhh, a.p1, B, GH, H, H, 1, 1, H)
            ex.launch[OP_CELL_FWD](a, B * H)
        else:
            ex.launch[OP_CELL_FWD_H](a, B * H)


def rnn_layer[E: Exec](
    mut tape: Tape, mut ex: E, gx: Tensor, w_hh: Tensor, b_hh: Tensor, b_ih: FP,
    cell: Int, T: Int, B: Int, w: Work, l: Int, h0_clean: Bool = False, reverse: Bool = False,
) raises -> Tensor:
    """Layer l's recurrence over gx [T B, G H]; returns h_1..h_T [T B, H]
    (w.hall[l] holds h_0 = 0 first). The launches are recurrent.forward's.

    `reverse` (lane nn-recurrent: Sequential's bidirectional layers): the same
    launches walking time from T - 1 down to 0 (every step pointer moves back
    one row block a step; the scan's i4 = 1). The states are stored in time
    order with the zero state AFTER them: w.hall[l] rows [t B, (t + 1) B) hold
    the state at time t and rows [T B, (T + 1) B) h_0 = 0, so the output is
    rows [0, T B), in time order, with no permutation. With lengths the mask
    position is the time, so a row's padded times (now first) hold its zero
    state and its own steps run from its last valid step down to 0.
    `h0_clean`: the caller knows rows [0, B) of w.hall[l] and w.call[l]
    already hold zeros, so the per-step chain does not fill them again."""
    var H = tape.cols(w_hh)
    var GH = tape.rows(w_hh)
    if GH != gates_of(cell) * H or tape.cols(gx) != GH or tape.rows(gx) != T * B:
        raise Error("recurrent layer: gx must be (T B, G H) and W_hh (G H, H)")
    if tape.rows(b_hh) != 1 or tape.cols(b_hh) != GH:
        raise Error("recurrent layer: b_hh must be (1, G H)")
    var gxp = tape.value(gx)
    var whh = tape.value(w_hh)
    var bhh = tape.value(b_hh)
    # The IDENTICAL default. Training and prediction (rnn_predict, the same
    # forward in inference mode) take this one route.
    # Sequence lengths (lane seq-blocks-2) mask every step (op_cell_fwd). Lane seq-blocks-3: the scan takes the
    # mask too (i5 = 1, p10 the lengths; sequence/recurrent_scan.mojo fwd_step), so with or without lengths a layer
    # takes the same route and the same launch count: the scan where scan_routes says so, else the chain.
    # Lane rnn-scan: the two routes are layer_fwd_scan / layer_fwd_chain, which give the same words at every shape
    # (checks/native_tests/rnn_scan.mojo runs both).
    var masked = w.has_lens
    var scanned = False
    comptime if SEQ_LSTM_SCAN:
        if scan_routes(H, gates_of(cell)):
            layer_fwd_scan(ex, cell, T, B, H, gxp, w.gh[l], w.act[l], w.hall[l], w.call[l], whh, bhh, b_ih, w.lens,
                           masked, reverse)
            scanned = True
    if not scanned:
        layer_fwd_chain(ex, cell, T, B, H, gxp, w.gh[l], w.act[l], w.hall[l], w.call[l], whh, bhh, b_ih, w.lens,
                        masked, reverse, h0_clean)
    var saved = Args()
    saved.p0 = w.act[l]
    saved.p1 = w.gh[l]
    saved.p2 = w.hall[l]
    saved.p3 = w.call[l]
    saved.p4 = w.dgh
    saved.p5 = w.dh
    saved.p6 = w.dhn
    saved.p7 = w.dc
    saved.p8 = w.wsplit
    saved.p9 = w.one
    saved.i0 = cell
    saved.i1 = B
    saved.i2 = T
    saved.p10 = w.lens
    saved.i3 = 1 if masked else 0
    saved.i4 = 1 if reverse else 0
    return tape.custom(ex, [gx, w_hh, b_hh], w.hall[l] + (0 if reverse else B * H), T * B, H, RNN_LAYER, saved)


# ------------------------------------------------------------------ backward rules
def _linear_bwd[E: Exec](
    mut ex: E, saved: Args, up: FP, inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
) raises:
    # recurrent.backward: dW = dY^T X (one GEMM over the R rows, or the
    # blocked order), db = column sums, dX = dY W. X is rows
    # [saved.i2, saved.i2 + R) of input 0 (all of it but for the head).
    var R = saved.i3
    var din = cols[0]
    var dout = rows[1]
    var xin = inputs[0] + saved.i2 * din
    var done = False
    comptime if SEQ_WGRAD_BLOCKED:
        if saved.i1 != 0:
            wgrad_blocked(ex, up, xin, gradients[1], dout, din, R, 1, dout, din, 1, din, saved.p0)
            if wgrad_block(R) < R:
                wgrad_blocked(ex, saved.p1, up, gradients[2], 1, dout, R, 0, 0, dout, 1, dout, saved.p0)
            else:
                colsum(ex, up, gradients[2], R, dout)
            done = True
    if not done:
        gemm(ex, up, xin, gradients[1], dout, din, R, 1, dout, din, 1, False, din)
        colsum(ex, up, gradients[2], R, dout)
    if saved.i0 != 0:
        gemm(ex, up, inputs[1], gradients[0] + saved.i2 * din, R, din, dout, dout, 1, din, 1, False, din)


def layer_bwd_scan[E: Exec](
    mut ex: E, cell: Int, T: Int, B: Int, H: Int, act: FP, gh: FP, hall: FP, call: FP, dgh: FP, dh_in: FP,
    dhn_in: FP, dc: FP, dgx: FP, whh: FP, up: FP, lens: FP, masked: Bool, reverse: Bool,
) raises:
    """A layer's backward through time as ONE launch (OP_CELL_BWD_SCAN, one block a batch row): dGX into dgx and
    dGH into dgh for every step, from the upstream state gradient `up`, with dh / dhn / dc as the scratch. i6 is
    the chain's fold of the recurrent product at this shape (scan_bwd_split), so the words are layer_bwd_chain's.
    The checkpoints are the forward's own stored states and gate activations (hall, call, act, gh): nothing is
    recomputed."""
    var GH = gates_of(cell) * H
    var sa = Args()
    # step 0's pointers (reverse: time T - 1, its previous state the zero rows after the states)
    var t0 = T - 1 if reverse else 0
    sa.p0 = act + t0 * B * GH
    sa.p1 = gh + t0 * B * GH
    sa.p2 = hall + (T if reverse else 0) * B * H
    sa.p3 = call + (T if reverse else 0) * B * H
    # c_t's base (call + B H), or with lengths the rows' lengths: the scan derives c_t from p3
    # (autograd/kernels/recurrent_scan.mojo bwd_step) and reads the lengths from here
    sa.p4 = lens if masked else sa.p3 + (-B * H if reverse else B * H)
    sa.p5 = dh_in
    sa.p6 = up + t0 * B * H
    sa.p7 = dc
    sa.p8 = dgx + t0 * B * GH
    sa.p9 = dgh + t0 * B * GH
    sa.p10 = dhn_in
    sa.p11 = whh
    sa.i0 = cell
    sa.i1 = B
    sa.i2 = H
    sa.i3 = T
    sa.i4 = -B * GH if reverse else B * GH
    sa.i5 = 1 if masked else 0
    # the chain's fold of the recurrent product at this shape (lane rnn-scan): the same dh words either route
    sa.i6 = scan_bwd_split(cell, B, H, GH)
    ex.launch[OP_CELL_BWD_SCAN](sa, B)


def layer_bwd_chain[E: Exec](
    mut ex: E, cell: Int, T: Int, B: Int, H: Int, act: FP, gh: FP, hall: FP, call: FP, dgh: FP, dh_in: FP,
    dhn_in: FP, dc: FP, dgx: FP, whh: FP, up: FP, lens: FP, masked: Bool, reverse: Bool,
) raises:
    """The same backward as the per-step chain: per time step, T - 1 down, the recurrent product (its own GEMM
    launch for the LSTM cell above the step rule, RNN_BPTT_LSTM_GEMM, else fused into the cell) and the cell.
    The buffers as layer_bwd_scan's."""
    var GH = gates_of(cell) * H
    var dh = dh_in
    var dhn = dhn_in
    # Lane fx-rnn-cnn-fit (2026-11-19): on the device the first backward step's kernel (s = T - 1, i3 = 0)
    # stores its own dh and dc zeros (sequence/ops.mojo op_cell_bwd_h), so the two fill launches are the
    # host's alone: HostExec runs OP_CELL_BWD_H as its GEMM and then op_cell_bwd, not that body
    # (sequence/exec.mojo). The same zeros in the same words on every column.
    # With lengths the fused step is OP_CELL_BWD_H_MASK (lane seq-blocks-3), which stores the same zeros at
    # its first step on the device; the GEMM-split step (RNN_BPTT_LSTM_GEMM's) fills them, as without lengths.
    var fills = False
    comptime if not E.IS_DEVICE:
        fills = True
    if fills:
        fill(ex, dh, B * H, Float32(0.0))
        fill(ex, dc, B * H, Float32(0.0))
    var step_gemm = False
    comptime if RNN_BPTT_LSTM_GEMM:
        step_gemm = rnn_bptt_gemm(cell, B, H, GH)
    var s = T - 1
    while s >= 0:  # driver-loop: over time steps, T - 1 down (one launch per step)
        # the time of step s and the rows of its previous and its own state (reverse: lane nn-recurrent)
        var tm = T - 1 - s if reverse else s
        var prev = tm + 1 if reverse else s
        var own = tm if reverse else s + 1
        var a = Args()
        a.p0 = act + tm * B * GH
        a.p1 = gh + tm * B * GH
        a.p2 = hall + prev * B * H
        a.p3 = call + prev * B * H
        a.p4 = call + own * B * H
        a.p5 = dh
        a.p6 = up + tm * B * H
        a.p7 = dc
        a.p8 = dgx + tm * B * GH
        a.p9 = dgh + tm * B * GH
        a.p10 = dhn
        a.p11 = whh
        a.i0 = cell
        a.i1 = B
        a.i2 = H
        a.i3 = 1 if s < T - 1 else 0
        a.i4 = -B * GH if reverse else B * GH
        if masked and not (step_gemm and s < T - 1):
            # lane seq-blocks-3: the fused step with the length mask, one launch (sequence/ops.mojo
            # op_cell_bwd_h_mask; p4 carries the lengths, c_t is p3 + B H; its first step stores the dh / dc zeros
            # as OP_CELL_BWD_H's): a step past a row's end passes dh and dc through and adds no gradient. It was
            # OP_GEMM + OP_CELL_BWD, the same words.
            a.p4 = lens
            a.i8 = tm
            ex.launch[OP_CELL_BWD_H_MASK](a, B * H)
        elif masked:
            # the GEMM-split route (RNN_BPTT_LSTM_GEMM) with the length mask: the product, then the masked cell
            var g = Args()
            g.p0 = a.p9 + a.i4
            g.p1 = whh
            g.p2 = dh
            g.i0 = B
            g.i1 = H
            g.i2 = GH
            g.i3 = GH
            g.i4 = 1
            g.i5 = H
            g.i6 = 1
            g.i7 = 1
            g.i8 = H
            ex.launch[OP_GEMM](g, B * H)
            a.p11 = lens
            a.i8 = tm
            a.i9 = 1
            ex.launch[OP_CELL_BWD](a, B * H)
        elif step_gemm and s < T - 1:
            # dh += dGH_{s+1} W_hh as the lane's GEMM (accumulate onto the direct part the later step stored),
            # then the cell alone. Step T - 1 has no product and keeps the fused launch (its dh / dc zeros).
            var g = Args()
            g.p0 = a.p9 + a.i4
            g.p1 = whh
            g.p2 = dh
            g.i0 = B
            g.i1 = H
            g.i2 = GH
            g.i3 = GH
            g.i4 = 1
            g.i5 = H
            g.i6 = 1
            g.i7 = 1
            g.i8 = H
            ex.launch[OP_GEMM](g, B * H)
            ex.launch[OP_CELL_BWD](a, B * H)
        else:
            ex.launch[OP_CELL_BWD_H](a, B * H)
        var t = dh
        dh = dhn
        dhn = t
        s -= 1


def _layer_bwd[E: Exec](
    mut ex: E, saved: Args, up: FP, inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
) raises:
    # recurrent.backward's per-layer body with cur = up (dH_out, read only)
    # and dgx = gradients[0]; dW_hh, db_hh into gradients[1], gradients[2].
    var cell = saved.i0
    var B = saved.i1
    var T = saved.i2
    var H = cols[1]
    var GH = rows[1]
    var act = saved.p0
    var gh = saved.p1
    var hall = saved.p2
    var call = saved.p3
    var dgh = saved.p4
    var dh = saved.p5
    var dhn = saved.p6
    var dc = saved.p7
    var dgx = gradients[0]
    var whh = inputs[1]
    var masked = saved.i3 != 0
    # the reverse direction (lane nn-recurrent, rnn_layer's `reverse`): step s is time T - 1 - s
    var reverse = saved.i4 != 0
    # Lane rnn-scan: the two routes are layer_bwd_scan / layer_bwd_chain, which give the same words at every shape
    # (checks/native_tests/rnn_scan.mojo runs both).
    var scanned = False
    comptime if SEQ_LSTM_SCAN:
        if scan_routes(H, gates_of(cell)):
            layer_bwd_scan(ex, cell, T, B, H, act, gh, hall, call, dgh, dh, dhn, dc, dgx, whh, up, saved.p10, masked,
                           reverse)
            scanned = True
    if not scanned:
        layer_bwd_chain(ex, cell, T, B, H, act, gh, hall, call, dgh, dh, dhn, dc, dgx, whh, up, saved.p10, masked,
                        reverse)
    var done = False
    # every step's previous state, in time order: rows [0, T B) forward, rows [B, (T + 1) B) reverse
    var hprev = hall + (B * H if reverse else 0)
    comptime if SEQ_WGRAD_BLOCKED:
        wgrad_blocked(ex, dgh, hprev, gradients[1], GH, H, T * B, 1, GH, H, 1, H, saved.p8)
        if wgrad_block(T * B) < T * B:
            wgrad_blocked(ex, saved.p9, dgh, gradients[2], 1, GH, T * B, 0, 0, GH, 1, GH, saved.p8)
        else:
            colsum(ex, dgh, gradients[2], T * B, GH)
        done = True
    if not done:
        gemm(ex, dgh, hprev, gradients[1], GH, H, T * B, 1, GH, H, 1, False, H)
        colsum(ex, dgh, gradients[2], T * B, GH)


@fieldwise_init
struct RecurrentRules(BackwardRules):
    """Native backward adapters for the recurrent trainer's custom ops."""

    def supports(self, tag: Int) -> Bool:
        return tag == RNN_LINEAR or tag == RNN_LAYER or tag == RNN_LAST_STEP

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        return False

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        # Fill only where recurrent.backward fills. RNN_LINEAR: dW (GEMM or
        # blocked wgrad) and db (column sum) overwrite every element; dX is
        # a full overwrite, or skipped for an input without a gradient
        # (_linear_record refuses otherwise), except the head's dh_T rows,
        # whose zeroed sequence gradient is the hand path's fill of cur.
        # RNN_LAYER: the cell (or scan) backward writes every step's dGX
        # rows, and dW_hh / db_hh overwrite; the hand path never zeroes
        # w.dgx or Gr either. RNN_LAST_STEP writes only step T - 1.
        if tag == RNN_LINEAR:
            return edge == 0 and saved.i4 != 0
        if tag == RNN_LAYER:
            return False
        return True

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        if tag == RNN_LINEAR:
            _linear_bwd(ex, saved, upstream, inputs, gradients, rows, cols)
        elif tag == RNN_LAYER:
            _layer_bwd(ex, saved, upstream, inputs, gradients, rows, cols)
        elif tag == RNN_LAST_STEP:
            # The tape zeroed the destination; only step T - 1 receives rows.
            _copy(ex, gradients[0] + saved.i0, upstream, saved.i1)
        else:
            raise Error("unknown recurrent backward rule")


# ------------------------------------------------------------------ the session op
#: Session operation code of one recurrent layer over a whole sequence (lane
#: fx-ops-session-blocks, 2026-11-19; autograd/ops_api.mojo `_rnn_scan`,
#: public `Session.lstm` / `Session.gru` / `Session.rnn`).
comptime OPS_RNN_SCAN = 94


def rnn_scan_op[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, w_ih: Tensor, w_hh: Tensor, b_ih: Tensor, b_hh: Tensor,
    cell: Int, T: Int, B: Int, lengths: Optional[FP] = None, reverse: Bool = False,
) raises -> Tensor:
    """One recurrent layer (cell CELL_LSTM, CELL_GRU or the tanh RNN) over a
    time-major sequence x [T B, D] from h_0 = c_0 = 0: returns h_1..h_T
    [T B, H]. PyTorch's layouts and gate order: w_ih (G H, D), w_hh
    (G H, H), b_ih and b_hh (1, G H).

    `lengths` (lane seq-blocks-3: Sequential's recurrent layers): B float32
    row lengths on the executor, each in [1, T]. A row's steps at or past its
    length copy its state forward (the cell's length mask, the estimators'
    packed semantics), so h_T is the state at its own last valid step and
    those steps add nothing to any gradient. The caller zeroes the padded
    inputs (x's rows of those steps), so their values never reach a product.

    It is the trainer's own layer, launch for launch: `rnn_linear` (the
    input projection with b_ih folded into gx, the blocked weight-gradient
    order) then `rnn_layer` (the scan route or the per-step chain), exactly
    as `rnn_forward_autograd` records layer 0, so the output words and, in
    a recording session, the gradients (RecurrentRules: RNN_LINEAR,
    RNN_LAYER) are the recurrent estimators' for the same weights. The only
    difference: dX is computed when x takes a gradient (the trainer's first
    layer reads data). The work buffers are tape storage, released at the
    tape's reset, of the sizes `Work` allocates for one layer.

    Serial by nature over time (h_t reads h_{t-1}); the rows of the batch
    are the scan launch's threads."""
    var H = tape.cols(w_hh)
    var GH = tape.rows(w_hh)
    var D = tape.cols(x)
    if T < 1 or B < 1:
        raise Error("ValueError: ops: a recurrent layer needs at least one step and one sequence")
    if GH != gates_of(cell) * H:
        raise Error("ValueError: ops: weight_hh must be (gates * hidden, hidden)")
    if tape.rows(x) != T * B:
        raise Error("ValueError: ops: x must hold steps * batch rows of input features")
    if tape.rows(w_ih) != GH or tape.cols(w_ih) != D:
        raise Error("ValueError: ops: weight_ih must be (gates * hidden, inputs)")
    if tape.rows(b_ih) != 1 or tape.cols(b_ih) != GH or tape.rows(b_hh) != 1 or tape.cols(b_hh) != GH:
        raise Error("ValueError: ops: bias_ih and bias_hh must each hold gates * hidden values")
    var train = tape.is_recording()
    var tb = T * B if train else 1
    var gxb = tape.reserve(ex, T * B * GH)
    var gh = tape.reserve(ex, T * B * GH)
    var act = tape.reserve(ex, T * B * GH)
    var hall = tape.reserve(ex, (T + 1) * B * H)
    var call = tape.reserve(ex, (T + 1) * B * H)
    var dgh = tape.reserve(ex, tb * GH)
    var dh = tape.reserve(ex, B * H if train else 1)
    var dhn = tape.reserve(ex, B * H if train else 1)
    var dc = tape.reserve(ex, B * H if train else 1)
    var one = tape.reserve(ex, 1)
    fill(ex, one, 1, Float32(1.0))
    var wsplit = one
    comptime if SEQ_WGRAD_BLOCKED:
        wsplit = tape.reserve(ex, WGRAD_IDN_SCRATCH if train else 1)
    var w = Work(gxb, gh, act, hall, call, dgh, dh, dhn, dc, one, wsplit)
    if lengths:
        w.lens = lengths.value()
        w.has_lens = True
    var gx = rnn_linear(tape, ex, x, w_ih, b_ih, gxb, RNN_INPUT_BIAS_IN_GX, True, tape.requires_grad(x), w)
    # `reverse` (lane nn-recurrent): time T - 1 down to 0, the same launches (rnn_layer); the output stays in
    # time order
    return rnn_layer(tape, ex, gx, w_hh, b_hh, tape.value(b_ih), cell, T, B, w, 0, False, reverse)


# ------------------------------------------------------------------ trainer step
def rnn_forward_autograd[E: Exec](
    mut tape: Tape, mut ex: E, net: Net, P: FP, Gr: FP, x: FP, T: Int, B: Int, w: Work, yhat: FP,
    h0_clean: Bool = False,
) raises -> Tensor:
    """Record input projections, layers and the head on the tape; the
    head's output lands in `yhat` (B x O; the trainer passes w.yhat) and its
    gradient is w.dy. Every parameter leaf's gradient is its slice of Gr
    (the `Net` layout). In inference mode (tape.no_grad(), rnn_predict) the
    same launches run and nothing is recorded. `h0_clean`: the previous
    forward on `w` had at least B rows (rnn_layer)."""
    tape.reset()
    var H = net.H
    var GH = net.G() * H
    var inp = tape.variable(ex, x, T * B, net.D, False)
    for l in range(net.L):  # driver-loop: over layers
        var din = net.din(l)
        var w_ih = tape.variable_into(ex, P + net.w_ih(l), Gr + net.w_ih(l), GH, din)
        var w_hh = tape.variable_into(ex, P + net.w_hh(l), Gr + net.w_hh(l), GH, H)
        var b_ih = tape.variable_into(ex, P + net.b_ih(l), Gr + net.b_ih(l), 1, GH)
        var b_hh = tape.variable_into(ex, P + net.b_hh(l), Gr + net.b_hh(l), 1, GH)
        var gx = rnn_linear(tape, ex, inp, w_ih, b_ih, w.gx[l], RNN_INPUT_BIAS_IN_GX, True, l > 0, w)
        inp = rnn_layer(tape, ex, gx, w_hh, b_hh, P + net.b_ih(l), net.cell, T, B, w, l, h0_clean)
    var w_o = tape.variable_into(ex, P + net.head(), Gr + net.head(), net.O, H)
    var b_o = tape.variable_into(ex, P + net.head() + net.O * H, Gr + net.head() + net.O * H, 1, net.O)
    return rnn_head(tape, ex, inp, w_o, b_o, B, yhat, w.dy, w)


def rnn_backward_autograd[E: Exec](mut tape: Tape, mut ex: E, net: Net, logits: Tensor, dy: FP, B: Int) raises:
    """Seed the existing loss derivative dy [B, O] (the head's gradient
    storage, so no seed copy) at the head output; parameter gradients land
    in Gr directly."""
    var seed = tape.variable(ex, dy, B, net.O, False)
    tape.backward_with_rules(ex, logits, seed, RecurrentRules())


struct AutogradRnnStep(RnnStep):
    """rnn_fit's step with the backward assembled by the tape."""

    # Launches per optimizer step, backward only (the forward launches were
    # recurrent.forward + head on both paths). L layers, T steps, per-step
    # cell path; W = one weight-gradient group (a GEMM, or wgrad_blocked's
    # launches), S = one bias sum (colsum or its blocked twin).
    #   HandStep (deleted 2026-10-19), for the record:
    #              head W + S + fill(dh_out) + GEMM(dh_T)                  4
    #              per layer: fill(dh) + fill(dc) + T cells + 2 W + 2 S
    #              (+ GEMM dX for l > 0)                    L (T + 6) + L - 1
    #   tape, before lane ag-rnn-fix: the same, plus a seed copy, three
    #              zero fills per custom node (6 L + 4), the h_T copy and
    #              4 L + 2 parameter-gradient copies into Gr:   10 L + 7 more
    #   tape now:  exactly the hand launches, same arguments; the only
    #              difference is order inside a layer (dW_hh, db_hh before
    #              dW_ih, db_ih, and the head fill before its dW), each
    #              writing its own buffer, so the bits are unchanged.
    var tape: Tape
    var logits: Tensor
    #: rows of the last forward this step ran on its trainer's one Work (0: none yet). A forward of B rows
    #: leaves rows [0, B) of every layer's h and c buffers zero and writes above them, so the next forward
    #: of at most that many rows needs no h_0 / c_0 fill (rnn_layer's h0_clean). rnn_fit hands one Work to
    #: one step for a whole fit; a step given a second Work must be a new step.
    var h0_rows: Int

    def __init__(out self):
        self.tape = Tape()
        self.logits = Tensor(0, 0, -1)
        self.h0_rows = 0

    def forward_head[E: Exec](
        mut self, mut ex: E, net: Net, P: FP, Gr: FP, x: FP, T: Int, B: Int, w: Work,
    ) raises:
        self.logits = rnn_forward_autograd(self.tape, ex, net, P, Gr, x, T, B, w, w.yhat, B <= self.h0_rows)
        self.h0_rows = B

    def backward[E: Exec](mut self, mut ex: E, net: Net, P: FP, Gr: FP, x: FP, T: Int, B: Int, w: Work) raises:
        rnn_backward_autograd(self.tape, ex, net, self.logits, w.dy, B)


# ------------------------------------------------------------------ predict
# Lane x3-mlp-rnn-predict-on-tape (plan EXECUTOR_CONSOLIDATION_AND_CLEANUP (git history) X3): prediction runs the
# trainer's recorded forward (rnn_forward_autograd) on a tape in inference mode.
# TOMBSTONE: MOJOLEARN_AUTOGRAD_PREDICT (folded in: the tape forward is the one predict path) deleted 2026-11-08 by 9a9a8e3148; code recoverable at c3b7dc42e6^.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_PREDICT.patch
# rnn_predict moved here from sequence/recurrent.mojo (which this module imports) with the hand forward and head
# it ran deleted. The launches are the hand chain's, argument for argument: the input projection GEMM (+ b_ih in
# its own launch exactly when recurrent.forward added it, RNN_INPUT_BIAS_IN_GX), the scan or the per-step
# OP_CELL_FWD_H chain with the same i5, then the head GEMM over h_T and its bias rows. Same bits by construction.
def rnn_predict[E: Exec](
    mut ex: E, net: Net, task: Int, X: FP, N: Int, T: Int, Pin: FP,
    dst: FP, seq: FP, want_seq: Bool, chunk: Int, resident: Bool = False, lengths: Optional[FP] = None,
) raises:
    """dst [N, O]: the head's output (task MSE) or its softmax (task CE);
    with want_seq, seq [N, T, H] is the top layer's h_1..h_T batch-first.
    Each chunk of `chunk` rows runs rnn_forward_autograd on a tape in
    inference mode (no graph, no gradient storage), the training forward's
    kernels in the training forward's order. `resident` (perf3-seq-neural):
    `Pin` is already this executor's memory (the model's resident fitted
    block, sequence/perf3_seq_resident.mojo), read in place: no per-call
    allocation and upload of the parameters.
    `lengths` (lane seq-blocks-2): N row lengths in [1, T] (float32 words,
    admitted by the caller); each row's output is read at its own last valid
    step and its padded steps never enter its state (rnn_fit's semantics).
    With want_seq, a row's steps past its length repeat its last state."""
    var np_ = net.n_params()
    var H = net.H
    var has_lens = Bool(lengths)
    var dL = ex.alloc(N if has_lens else 1)
    var dL_opt = Optional[FP](None)
    if has_lens:
        ex.upload(dL, lengths.value(), N)
        dL_opt = Optional[FP](dL)
    var bmax = chunk if chunk < N else N
    var dX = ex.alloc(N * T * net.D)
    ex.upload(dX, X, N * T * net.D)
    # perf3-seq-neural: the identity row order is built on the executor
    # (OP_SEQ_IOTA, the fit's unshuffled order: p0[t] = t, exact below 2^24,
    # N < 2^24 is admitted by rnn_predict_impl), not by a host loop over the
    # rows and an upload. The same float32 words.
    var didx = ex.alloc(N)
    var ia = Args()
    ia.p0 = didx
    ex.launch[OP_SEQ_IOTA](ia, N)
    var P: FP
    if resident:
        P = Pin
    else:
        P = ex.alloc(np_)
        ex.upload(P, Pin, np_)
    var dout = ex.alloc(N * net.O)
    var dseq = ex.alloc(N * T * H if want_seq else 1)
    var w = Work(ex, net, T, bmax, False, has_lens)
    var tape = Tape()
    tape.no_grad()
    # rows of the previous chunk's forward on w (rnn_layer's h0_clean: every chunk but a longer one skips the fills)
    var h0_rows = 0
    var b0 = 0
    while b0 < N:  # driver-loop: over prediction chunks
        var B = bmax if b0 + bmax <= N else N - b0
        gather_seq(ex, dX, didx, w.xb, T, B, net.D, b0, dL_opt)
        if has_lens:
            gather_rows(ex, dL, didx, w.lens, B, 1, b0)
        # Inference mode keeps no gradient pointer, so the gradient vector is
        # a placeholder; rnn_forward_autograd resets the tape, so every chunk
        # reuses the first chunk's storage. The head writes this chunk's rows
        # of dout directly.
        _ = rnn_forward_autograd(tape, ex, net, P, dummy_ptr(), w.xb, T, B, w, dout + b0 * net.O, B <= h0_rows)
        h0_rows = B
        if task == TASK_CE:
            var a = Args()
            a.p0 = dout + b0 * net.O
            a.p1 = dout + b0 * net.O
            a.i1 = net.O
            ex.launch[OP_SOFTMAX](a, B)
        if want_seq:
            var a = Args()
            a.p0 = w.hall[net.L - 1] + B * H
            a.p1 = dseq
            a.i0 = T
            a.i1 = B
            a.i2 = H
            a.i3 = b0
            ex.launch[OP_SEQ_OUT](a, B * T * H)
        b0 += B
    tape.release(ex)
    ex.sync()
    ex.download(dst, dout, N * net.O)
    if want_seq:
        ex.download(seq, dseq, N * T * H)
