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
from std.sys.compile import is_defined

from autograd.tape import Tape, Tensor, BackwardRules
from sequence.exec_trait import Exec
from sequence.ops import (
    FP,
    Args,
    OP_CELL_BWD,
    OP_CELL_BWD_H,
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
    Net, Work, RnnStep, SEQ_WGRAD_BLOCKED, TASK_CE, WGRAD_IDN_SCRATCH, bias_rows, fill, gather_seq, gemm,
)
from autograd.kernels.recurrent import colsum, wgrad_block, wgrad_blocked
from sequence.mlp import EPI_BIAS_ACT
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

# Experiment MOJOLEARN_SEQ_FAST_RNN_STEP_GEMM (lane fx-rnn-cnn-fit, 2026-11-19; default OFF, A/B owed on the NVIDIA
# L40S: owed-pre0867.json "rnn-step-gemm"). THE RECURRENT PRODUCTS ON THE LANE'S OWN GEMM. BIT VERSION under the
# define, every column together (host, Metal, NVIDIA, AMD run the same launches; nothing changes without it).
#
# Default: the per-step chain (cells of more gates than the scan takes) runs each time step as ONE launch of one
# thread per (row, unit): the thread folds its own G gate dots of K = H terms straight from device memory, then the
# cell (OP_CELL_FWD_H; the backward one dot of K = G H, OP_CELL_BWD_H), a fusion made for Apple's launch cost
# (2026-09-28); and the two weight gradients of a layer are the blocked order of nr-small D3 (wgrad_blocked: two
# launches each, sqrt(K) blocks, one thread a partial). Arithmetic a step at the board shape (T 24, B 256, H 64, G 4):
# forward and backward cells 48 x 4.2 M multiply-adds, dW_hh 100 M (one third of the step's arithmetic), all folded
# by one thread a cell or partial with no operand reuse; gru 0.80 and lstm 0.97 ms a step on the same launches is
# 0.17 ms a gate, about 0.9 TFLOP/s over the whole step, where the sequence GEMM's tile runs 3.6-4.0.
#
# Under the define every recurrent product is one launch of OP_GEMM / OP_GEMM_EPI, on the route and in the fold the
# lane's launch rule gives its shape (sequence/gemm_split.mojo, sequence/gemm_tiled.mojo), as every other GEMM of
# the lane:
#   forward step s    GH_s = h_{s-1} W_hh^T + b_hh   [B, G H] over K = H      (OP_GEMM_EPI, bias epilogue), then
#                     the elementwise cell OP_CELL_FWD
#   backward step s   dh += dGH_{s+1} W_hh           [B, H]   over K = G H    (OP_GEMM, accumulate), then the
#                     elementwise cell derivative OP_CELL_BWD (step T - 1 has no product: the fused launch)
#   dW_hh             dGH^T H_prev                   [G H, H] over K = T B    (ONE OP_GEMM over the stacked steps)
#   dW_ih             dGX^T X                        [G H, D_in] over K = T B (ONE OP_GEMM over the stacked steps)
# The weight gradients were already one product over the T B stacked rows each (never T products): what changes
# for them is two launches -> one and the fold. The per-step products cannot be stacked: step s reads step s + 1's
# cell derivative.
# THE FOLD, per output cell (m, n) of an M x N x K product, as a formula:
#   gemm_split_shape(M, N, K) false:  acc = c0; acc = fma(A(m, k), B(k, n), acc), k = 0 .. K - 1
#   gemm_split_shape(M, N, K) true:   R = gemm_split_block(K); p_s = +0.0; p_s = fma(A(m, k), B(k, n), p_s) over
#                                     k = s R .. min(K, (s + 1) R) - 1; acc = c0; acc = acc + p_s, s ascending
#   c0 = the stored dh for the backward step (accumulate), +0.0 otherwise; the forward then adds b_hh[n].
# BITS against the default, per shape class:
#   per-step forward, not a split-K shape (the board's H 64: K = 64): SAME (gemm_dot's chain, the fused body's)
#   per-step forward, split-K shape (K = H > 64 with few cells or tiles): CHANGE
#   per-step backward, not a split-K shape: SAME;  split-K shape (the board's: K = G H = 256, 4 tiles): CHANGE
#   dW_hh, dW_ih with T B <= 512 rows: SAME (the default is already this GEMM)
#   dW_hh, dW_ih with T B > 512 rows: CHANGE (blocks of wgrad_block(T B) rows -> the formula above)
#   the bias gradients, the head, the input projection, the scan route's own steps (one-gate cells): SAME
# The host column follows because HostExec.launch takes the same rule for the same opcodes; the oracle follows under
# the same define (sequence/checks/oracle.mojo o_bptt_dw) and the seams check adds device-against-host cases at an
# arm shape (sequence/checks/seams_check.mojo 5504 step_gemm). Fit and predict share the forward, so they agree.
# The rule for the per-step products (rnn_step_gemm): at least RNN_STEP_GEMM_MIN_FMA multiply-adds a step. Cost
# reasoning, not a shape: the second launch costs one launch floor (3-8 us on the L40S, solved from the board rows),
# which the fused fold spends on about 1 M multiply-adds; below that the fused launch is the cheaper form on any
# hardware with a launch floor. The weight gradients take the GEMM at every size (one launch instead of two).
# Launches a step on a GPU column (L 1, T 24): lstm / gru 68 -> 113 (48 cell launches become 95, the two weight
# gradients 4 -> 2); rnn (scan) 22 -> 20. The count goes UP on the chain: the bet is arithmetic rate, not launches.
# Neighbouring shapes to time: T 48, H 128; non-board model: a 2-layer GRU regressor on a 12-feature series.
comptime RNN_STEP_GEMM = (
    is_defined["MOJOLEARN_SEQ_FAST_RNN_STEP_GEMM"]() and GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
)
comptime RNN_STEP_GEMM_MIN_FMA = 1 << 20


@always_inline
def _coop_shape(M: Int, N: Int, K: Int) -> Bool:
    """The shape DeviceExec._launch gives OP_GEMM's cooperative kernel (few
    cells over a long K); the epilogue opcodes never take it."""
    return M * N <= 1024 and K >= 32768


@always_inline
def rnn_step_gemm(M: Int, N: Int, K: Int) -> Bool:
    """Whether a time step's M x N x K hidden projection runs as its own
    GEMM launch under MOJOLEARN_SEQ_FAST_RNN_STEP_GEMM (the rule above)."""
    comptime if not RNN_STEP_GEMM:
        return False
    return M * N * K >= RNN_STEP_GEMM_MIN_FMA


def _wgrad[E: Exec](
    mut ex: E, A: FP, B: FP, C: FP, M: Int, N: Int, K: Int,
    sam: Int, sak: Int, sbk: Int, sbn: Int, ldc: Int, scratch: FP,
) raises:
    """A recurrent weight gradient C [M, N] = A B over the K = T B stacked
    rows: the blocked order (wgrad_blocked), or under
    MOJOLEARN_SEQ_FAST_RNN_STEP_GEMM one OP_GEMM on the launch rule's route
    and fold (the formula above; `scratch` unused)."""
    comptime if RNN_STEP_GEMM:
        gemm(ex, A, B, C, M, N, K, sam, sak, sbk, sbn, False, ldc)
    else:
        wgrad_blocked(ex, A, B, C, M, N, K, sam, sak, sbk, sbn, ldc, scratch)


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


def rnn_layer[E: Exec](
    mut tape: Tape, mut ex: E, gx: Tensor, w_hh: Tensor, b_hh: Tensor, b_ih: FP,
    cell: Int, T: Int, B: Int, w: Work, l: Int, h0_clean: Bool = False,
) raises -> Tensor:
    """Layer l's recurrence over gx [T B, G H]; returns h_1..h_T [T B, H]
    (w.hall[l] holds h_0 = 0 first). The launches are recurrent.forward's.
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
    var scanned = False
    # The IDENTICAL default. Training and prediction (rnn_predict, the same
    # forward in inference mode) take this one route.
    comptime if SEQ_LSTM_SCAN:
        if scan_routes(H, gates_of(cell)):
            var sa = Args()
            sa.p0 = gxp
            sa.p1 = w.gh[l]
            sa.p2 = w.act[l]
            sa.p3 = w.hall[l]
            sa.p4 = w.call[l]
            sa.p5 = w.hall[l] + B * H
            sa.p6 = w.call[l] + B * H
            sa.p7 = whh
            sa.p8 = bhh
            # never read on this route (b_ih is in gx); bound to live
            # storage, not dummy_ptr (the Metal rule)
            sa.p9 = b_ih
            sa.p10 = gxp
            sa.p11 = whh
            sa.i0 = cell
            sa.i1 = B
            sa.i2 = H
            sa.i3 = T
            ex.launch[OP_CELL_FWD_SCAN](sa, B)
            scanned = True
    if not scanned:
        # Lane fx-rnn-cnn-fit (2026-11-19): h_0 and c_0 are rows [0, B) of the two buffers, which nothing but
        # this fill (or the scan's own zero stores) writes while the batch size stays B: step s stores rows
        # [(s + 1) B, (s + 2) B). Only a forward of FEWER rows writes below B H, so the caller passes h0_clean when
        # the previous forward on this Work had at least B rows (AutogradRnnStep, rnn_predict): the two fills then
        # run once after an epoch's short last batch, not every step. The same zeros, so the same bits.
        if not h0_clean:
            fill(ex, w.hall[l], B * H, Float32(0.0))
            fill(ex, w.call[l], B * H, Float32(0.0))
        var step_gemm = False
        comptime if RNN_STEP_GEMM:
            step_gemm = rnn_step_gemm(B, GH, H)
        for s in range(T):  # driver-loop: over time steps (one launch per step; two under RNN_STEP_GEMM)
            var a = Args()
            a.p0 = gxp + s * B * GH
            a.p1 = w.gh[l] + s * B * GH
            a.p2 = w.act[l] + s * B * GH
            a.p3 = w.hall[l] + s * B * H
            a.p4 = w.call[l] + s * B * H
            a.p5 = w.hall[l] + (s + 1) * B * H
            a.p6 = w.call[l] + (s + 1) * B * H
            a.p7 = whh
            a.p8 = bhh
            a.p9 = b_ih
            a.i5 = 0
            a.i0 = cell
            a.i1 = B
            a.i2 = H
            if step_gemm:
                # GH_s = h_{s-1} W_hh^T + b_hh where the fused body stores it, then the cell alone
                _gemm_bias(ex, a.p3, whh, bhh, a.p1, B, GH, H, H, 1, 1, H)
                ex.launch[OP_CELL_FWD](a, B * H)
            else:
                ex.launch[OP_CELL_FWD_H](a, B * H)
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
    return tape.custom(ex, [gx, w_hh, b_hh], w.hall[l] + B * H, T * B, H, RNN_LAYER, saved)


def rnn_last_step[E: Exec](mut tape: Tape, mut ex: E, seq: Tensor, B: Int) raises -> Tensor:
    """The last B rows of a time-major sequence [T B, H], as a view."""
    var rows = tape.rows(seq)
    var H = tape.cols(seq)
    if B <= 0 or rows % B != 0:
        raise Error("recurrent last step: rows must be a multiple of the batch")
    var saved = Args()
    saved.i0 = (rows - B) * H
    saved.i1 = B * H
    return tape.custom(ex, [seq], tape.value(seq) + (rows - B) * H, B, H, RNN_LAST_STEP, saved)


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
            _wgrad(ex, up, xin, gradients[1], dout, din, R, 1, dout, din, 1, din, saved.p0)
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
    var scanned = False
    comptime if SEQ_LSTM_SCAN:
        if scan_routes(H, gates_of(cell)):
            var sa = Args()
            sa.p0 = act
            sa.p1 = gh
            sa.p2 = hall
            sa.p3 = call
            sa.p4 = call + B * H
            sa.p5 = dh
            sa.p6 = up
            sa.p7 = dc
            sa.p8 = dgx
            sa.p9 = dgh
            sa.p10 = dhn
            sa.p11 = whh
            sa.i0 = cell
            sa.i1 = B
            sa.i2 = H
            sa.i3 = T
            sa.i4 = B * GH
            ex.launch[OP_CELL_BWD_SCAN](sa, B)
            scanned = True
    if not scanned:
        # Lane fx-rnn-cnn-fit (2026-11-19): on the device the first backward step's kernel (s = T - 1, i3 = 0)
        # stores its own dh and dc zeros (sequence/ops.mojo op_cell_bwd_h), so the two fill launches are the
        # host's alone: HostExec runs OP_CELL_BWD_H as its GEMM and then op_cell_bwd, not that body
        # (sequence/exec.mojo). The same zeros in the same words on every column.
        comptime if not E.IS_DEVICE:
            fill(ex, dh, B * H, Float32(0.0))
            fill(ex, dc, B * H, Float32(0.0))
    var step_gemm = False
    comptime if RNN_STEP_GEMM:
        step_gemm = (not scanned) and rnn_step_gemm(B, H, GH)
    var s = T - 1 if not scanned else -1
    while s >= 0:  # driver-loop: over time steps, T - 1 down (one launch per step)
        var a = Args()
        a.p0 = act + s * B * GH
        a.p1 = gh + s * B * GH
        a.p2 = hall + s * B * H
        a.p3 = call + s * B * H
        a.p4 = call + (s + 1) * B * H
        a.p5 = dh
        a.p6 = up + s * B * H
        a.p7 = dc
        a.p8 = dgx + s * B * GH
        a.p9 = dgh + s * B * GH
        a.p10 = dhn
        a.p11 = whh
        a.i0 = cell
        a.i1 = B
        a.i2 = H
        a.i3 = 1 if s < T - 1 else 0
        a.i4 = B * GH
        if step_gemm and s < T - 1:
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
    var done = False
    comptime if SEQ_WGRAD_BLOCKED:
        _wgrad(ex, dgh, hall, gradients[1], GH, H, T * B, 1, GH, H, 1, H, saved.p8)
        if wgrad_block(T * B) < T * B:
            wgrad_blocked(ex, saved.p9, dgh, gradients[2], 1, GH, T * B, 0, 0, GH, 1, GH, saved.p8)
        else:
            colsum(ex, dgh, gradients[2], T * B, GH)
        done = True
    if not done:
        gemm(ex, dgh, hall, gradients[1], GH, H, T * B, 1, GH, H, 1, False, H)
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
    cell: Int, T: Int, B: Int,
) raises -> Tensor:
    """One recurrent layer (cell CELL_LSTM, CELL_GRU or the tanh RNN) over a
    time-major sequence x [T B, D] from h_0 = c_0 = 0: returns h_1..h_T
    [T B, H]. PyTorch's layouts and gate order: w_ih (G H, D), w_hh
    (G H, H), b_ih and b_hh (1, G H).

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
    var gx = rnn_linear(tape, ex, x, w_ih, b_ih, gxb, RNN_INPUT_BIAS_IN_GX, True, tape.requires_grad(x), w)
    return rnn_layer(tape, ex, gx, w_hh, b_hh, tape.value(b_ih), cell, T, B, w, 0)


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
# Lane x3-mlp-rnn-predict-on-tape (docs/plans/EXECUTOR_CONSOLIDATION_AND_CLEANUP.md X3): prediction runs the
# trainer's recorded forward (rnn_forward_autograd) on a tape in inference mode.
# TOMBSTONE: MOJOLEARN_AUTOGRAD_PREDICT (folded in: the tape forward is the one predict path) deleted 2026-11-08 by 9a9a8e3148; code recoverable at c3b7dc42e6^.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_PREDICT.patch
# rnn_predict moved here from sequence/recurrent.mojo (which this module imports) with the hand forward and head
# it ran deleted. The launches are the hand chain's, argument for argument: the input projection GEMM (+ b_ih in
# its own launch exactly when recurrent.forward added it, RNN_INPUT_BIAS_IN_GX), the scan or the per-step
# OP_CELL_FWD_H chain with the same i5, then the head GEMM over h_T and its bias rows. Same bits by construction.
def rnn_predict[E: Exec](
    mut ex: E, net: Net, task: Int, X: FP, N: Int, T: Int, Pin: FP,
    dst: FP, seq: FP, want_seq: Bool, chunk: Int, resident: Bool = False,
) raises:
    """dst [N, O]: the head's output (task MSE) or its softmax (task CE);
    with want_seq, seq [N, T, H] is the top layer's h_1..h_T batch-first.
    Each chunk of `chunk` rows runs rnn_forward_autograd on a tape in
    inference mode (no graph, no gradient storage), the training forward's
    kernels in the training forward's order. `resident` (perf3-seq-neural):
    `Pin` is already this executor's memory (the model's resident fitted
    block, sequence/perf3_seq_resident.mojo), read in place: no per-call
    allocation and upload of the parameters."""
    var np_ = net.n_params()
    var H = net.H
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
    var w = Work(ex, net, T, bmax, False)
    var tape = Tape()
    tape.no_grad()
    # rows of the previous chunk's forward on w (rnn_layer's h0_clean: every chunk but a longer one skips the fills)
    var h0_rows = 0
    var b0 = 0
    while b0 < N:  # driver-loop: over prediction chunks
        var B = bmax if b0 + bmax <= N else N - b0
        gather_seq(ex, dX, didx, w.xb, T, B, net.D, b0)
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
