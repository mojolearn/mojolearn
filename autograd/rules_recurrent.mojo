# SPDX-License-Identifier: Apache-2.0
# Recurrent trainer on the tape (-D MOJOLEARN_AUTOGRAD_RNN, default off).
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
#
# Every launch, argument and fold order is the hand path's, and the tape only
# copies first contributions (no tensor is used twice, so no ACCUMULATE
# runs): the gradients in Gr are expected to carry the same bits as
# HandStep. The A/B is HandStep (arm A) against this step (arm B).
# Scratch comes from the trainer's Work buffers, not tape.reserve, so the
# tape allocates only node gradients and contributions.
from std.sys.compile import is_defined

from autograd.tape import Tape, Tensor, BackwardRules
from sequence.exec_trait import Exec
from sequence.ops import (
    FP,
    Args,
    NN42_INPUT_BIAS_FUSED,
    OP_CELL_BWD_H,
    OP_CELL_FWD_H,
    OP_COPY,
    gates_of,
)
from sequence.recurrent_scan import OP_CELL_BWD_SCAN, OP_CELL_FWD_SCAN, SEQ_LSTM_SCAN, scan_applies
from sequence.recurrent import (
    Net,
    Work,
    RnnStep,
    SEQ_WGRAD_BLOCKED,
    bias_rows,
    colsum,
    fill,
    gemm,
    wgrad_block,
    wgrad_blocked,
)

comptime AUTOGRAD_RNN = is_defined["MOJOLEARN_AUTOGRAD_RNN"]()

# Tags in a range of their own, clear of autograd/layers.mojo (LAYER_NORM = 1),
# so one NeuralRules can later dispatch both families.
comptime RNN_LINEAR = 16
comptime RNN_LAYER = 17
comptime RNN_LAST_STEP = 18

# sequence/recurrent.mojo::forward adds b_ih to GX in its own launch unless
# the per-step kernel adds it (NN42 fused, per-step path only).
comptime RNN_INPUT_BIAS_IN_GX = not NN42_INPUT_BIAS_FUSED or SEQ_LSTM_SCAN


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
    var din = tape.cols(x)
    var dout = tape.rows(weight)
    if tape.cols(weight) != din or tape.rows(bias) != 1 or tape.cols(bias) != dout:
        raise Error("recurrent linear: weight must be (out, in) and bias (1, out)")
    gemm(ex, tape.value(x), tape.value(weight), output, R, dout, din, din, 1, 1, din, False, dout)
    if add_bias:
        bias_rows(ex, output, tape.value(bias), output, R, dout)
    var saved = Args()
    saved.p0 = w.wsplit
    saved.p1 = w.one
    saved.i0 = 1 if need_dx else 0
    saved.i1 = 1 if blocked_wgrad else 0
    return tape.custom(ex, [x, weight, bias], output, R, dout, RNN_LINEAR, saved)


def rnn_layer[E: Exec](
    mut tape: Tape, mut ex: E, gx: Tensor, w_hh: Tensor, b_hh: Tensor, b_ih: FP,
    cell: Int, T: Int, B: Int, w: Work, l: Int,
) raises -> Tensor:
    """Layer l's recurrence over gx [T B, G H]; returns h_1..h_T [T B, H]
    (w.hall[l] holds h_0 = 0 first). The launches are recurrent.forward's."""
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
    comptime if SEQ_LSTM_SCAN:
        if scan_applies(H, gates_of(cell)):
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
            sa.i0 = cell
            sa.i1 = B
            sa.i2 = H
            sa.i3 = T
            ex.launch[OP_CELL_FWD_SCAN](sa, B)
            scanned = True
    if not scanned:
        fill(ex, w.hall[l], B * H, Float32(0.0))
        fill(ex, w.call[l], B * H, Float32(0.0))
        for s in range(T):  # driver-loop: over time steps (one launch per step)
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
            a.i5 = 0 if RNN_INPUT_BIAS_IN_GX else 1
            a.i0 = cell
            a.i1 = B
            a.i2 = H
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
    # blocked order), db = column sums, dX = dY W.
    var R = rows[0]
    var din = cols[0]
    var dout = rows[1]
    var done = False
    comptime if SEQ_WGRAD_BLOCKED:
        if saved.i1 != 0:
            wgrad_blocked(ex, up, inputs[0], gradients[1], dout, din, R, 1, dout, din, 1, din, saved.p0)
            if wgrad_block(R) < R:
                wgrad_blocked(ex, saved.p1, up, gradients[2], 1, dout, R, 0, 0, dout, 1, dout, saved.p0)
            else:
                colsum(ex, up, gradients[2], R, dout)
            done = True
    if not done:
        gemm(ex, up, inputs[0], gradients[1], dout, din, R, 1, dout, din, 1, False, din)
        colsum(ex, up, gradients[2], R, dout)
    if saved.i0 != 0:
        gemm(ex, up, inputs[1], gradients[0], R, din, dout, dout, 1, din, 1, False, din)


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
        if scan_applies(H, gates_of(cell)):
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
        fill(ex, dh, B * H, Float32(0.0))
        fill(ex, dc, B * H, Float32(0.0))
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
        ex.launch[OP_CELL_BWD_H](a, B * H)
        var t = dh
        dh = dhn
        dhn = t
        s -= 1
    var done = False
    comptime if SEQ_WGRAD_BLOCKED:
        wgrad_blocked(ex, dgh, hall, gradients[1], GH, H, T * B, 1, GH, H, 1, H, saved.p8)
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


# ------------------------------------------------------------------ trainer step
def rnn_forward_autograd[E: Exec](
    mut tape: Tape, mut ex: E, net: Net, P: FP, x: FP, T: Int, B: Int, w: Work,
    mut params: List[Tensor], mut offsets: List[Int],
) raises -> Tensor:
    """Record input projections, layers, h_T and the head on the tape; the
    head's output lands in w.yhat. params/offsets receive every parameter
    leaf and its offset in the `Net` layout."""
    tape.reset()
    params.clear()
    offsets.clear()
    var H = net.H
    var GH = net.G() * H
    var inp = tape.variable(ex, x, T * B, net.D, False)
    for l in range(net.L):  # driver-loop: over layers
        var din = net.din(l)
        var w_ih = tape.variable(ex, P + net.w_ih(l), GH, din)
        var w_hh = tape.variable(ex, P + net.w_hh(l), GH, H)
        var b_ih = tape.variable(ex, P + net.b_ih(l), 1, GH)
        var b_hh = tape.variable(ex, P + net.b_hh(l), 1, GH)
        params.append(w_ih)
        offsets.append(net.w_ih(l))
        params.append(w_hh)
        offsets.append(net.w_hh(l))
        params.append(b_ih)
        offsets.append(net.b_ih(l))
        params.append(b_hh)
        offsets.append(net.b_hh(l))
        var gx = rnn_linear(tape, ex, inp, w_ih, b_ih, w.gx[l], RNN_INPUT_BIAS_IN_GX, True, l > 0, w)
        inp = rnn_layer(tape, ex, gx, w_hh, b_hh, P + net.b_ih(l), net.cell, T, B, w, l)
    var hT = rnn_last_step(tape, ex, inp, B)
    var w_o = tape.variable(ex, P + net.head(), net.O, H)
    var b_o = tape.variable(ex, P + net.head() + net.O * H, 1, net.O)
    params.append(w_o)
    offsets.append(net.head())
    params.append(b_o)
    offsets.append(net.head() + net.O * H)
    return rnn_linear(tape, ex, hT, w_o, b_o, w.yhat, True, False, True, w)


def rnn_backward_autograd[E: Exec](
    mut tape: Tape, mut ex: E, net: Net, logits: Tensor,
    params: List[Tensor], offsets: List[Int], dy: FP, Gr: FP, B: Int,
) raises:
    """Seed the existing loss derivative dy [B, O] at the head output and
    copy each parameter gradient into Gr."""
    var seed = tape.variable(ex, dy, B, net.O, False)
    tape.backward_with_rules(ex, logits, seed, RecurrentRules())
    for i in range(len(params)):  # driver-loop: over parameter tensors
        _copy(ex, Gr + offsets[i], tape.gradient(params[i]), tape.rows(params[i]) * tape.cols(params[i]))


struct AutogradRnnStep(RnnStep):
    """rnn_fit's step with the backward assembled by the tape."""

    var tape: Tape
    var logits: Tensor
    var params: List[Tensor]
    var offsets: List[Int]

    def __init__(out self):
        self.tape = Tape()
        self.logits = Tensor(0, 0, -1)
        self.params = List[Tensor]()
        self.offsets = List[Int]()

    def forward_head[E: Exec](mut self, mut ex: E, net: Net, P: FP, x: FP, T: Int, B: Int, w: Work) raises:
        var params = List[Tensor]()
        var offsets = List[Int]()
        self.logits = rnn_forward_autograd(self.tape, ex, net, P, x, T, B, w, params, offsets)
        self.params = params^
        self.offsets = offsets^

    def backward[E: Exec](mut self, mut ex: E, net: Net, P: FP, Gr: FP, x: FP, T: Int, B: Int, w: Work) raises:
        var params = self.params.copy()
        var offsets = self.offsets.copy()
        rnn_backward_autograd(self.tape, ex, net, self.logits, params, offsets, w.dy, Gr, B)
