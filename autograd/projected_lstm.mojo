# SPDX-License-Identifier: Apache-2.0
"""Projected LSTM over a time-major input. The projection is inside the recurrence."""

from sequence.exec_trait import Exec
from sequence.lengths import check_recurrent_lengths
from sequence.ops import Args, FP, OP_FILL, OP_SEQ_IOTA
from autograd.tape import Tape, Tensor
from autograd.rankn import ShapeN
from autograd.rankn_api import rn_shape
from autograd.rankn_layout import rankn_concat, rankn_narrow
from autograd.import_api import imp_binary, imp_unary, imp_where
from autograd.import_ops import IMP_GE, IMP_LE, IMP_EQ, IMP_AND, IMP_FLOOR

comptime OPS_PROJECTED_LSTM = 190
comptime OPS_PROJECTED_LSTM_OUTPUT = 191


def projected_lstm_output[E: Exec](
    mut tape: Tape, mut ex: E, packed: Tensor, xs: Tensor, projection: Tensor, part: Int, reverse: Bool,
) raises -> Tensor:
    """Select y, h or c using the current input dimensions, including during dynamic Program replay."""
    var xs_shape = tape.shape_of(xs)
    var hr = tape.shape_of(projection)
    if xs_shape.ndim != 3 or hr.ndim != 2 or part < 0 or part > 2:
        raise Error("ValueError: invalid projected LSTM output description")
    var T = xs_shape.d[0]
    var B = xs_shape.d[1]
    var P = hr.d[0]
    var H = hr.d[1]
    var hidden_words = B * P
    var sequence_words = T * hidden_words
    if tape.cols(packed) != 1 or tape.rows(packed) != sequence_words + B * H:
        raise Error("ValueError: packed projected LSTM state does not match its sequence and weights")
    if part == 0:
        var y = rankn_narrow(tape, ex, packed, 0, 0, sequence_words)
        return tape.view_nd(ex, y, rn_shape([T, B, P], "projected LSTM sequence"))
    if part == 1:
        var start = 0 if reverse else (T - 1) * hidden_words
        var h = rankn_narrow(tape, ex, packed, 0, start, hidden_words)
        return tape.view_nd(ex, h, ShapeN.matrix(B, P))
    var c = rankn_narrow(tape, ex, packed, 0, sequence_words, B * H)
    return tape.view_nd(ex, c, ShapeN.matrix(B, H))


def _filled[E: Exec](mut tape: Tape, mut ex: E, rows: Int, cols: Int, value: Float32) raises -> Tensor:
    var p = tape.reserve(ex, rows * cols)
    var args = Args()
    args.p0 = p
    args.f0 = value
    ex.launch[OP_FILL](args, rows * cols)
    return tape.variable(ex, p, rows, cols, False)


def projected_lstm_check_lengths[E: Exec](mut tape: Tape, mut ex: E, values: Tensor, B: Int, T: Int) raises:
    """Admit device-resident Session lengths. Sequential's public fit/predict already admits its host input."""
    var dims = tape.shape_of(values)
    if dims.count() != B or dims.ndim > 2 or (dims.rows() != 1 and dims.cols() != 1):
        raise Error("ValueError: projected LSTM lengths must have one entry per batch row")
    var ptr = tape.value_of(ex, values)
    comptime if not E.IS_DEVICE:
        check_recurrent_lengths(ptr, B, T, "projected LSTM lengths")
    else:
        # A Session length tensor can be the result of earlier device operations.
        # Reduce its admission to one status word before entering the recurrence.
        var lengths = tape.variable(ex, ptr, B, 1, False)
        var low = imp_binary(tape, ex, lengths, _filled(tape, ex, 1, 1, 1), IMP_GE)
        var high = imp_binary(tape, ex, lengths, _filled(tape, ex, 1, 1, Float32(T)), IMP_LE)
        var whole = imp_binary(tape, ex, lengths, imp_unary(tape, ex, lengths, IMP_FLOOR), IMP_EQ)
        var valid = imp_binary(tape, ex, imp_binary(tape, ex, low, high, IMP_AND), whole, IMP_AND)
        var invalid = imp_binary(tape, ex, valid, _filled(tape, ex, 1, 1, 0), IMP_EQ)
        var count = tape.sum(ex, invalid)
        var result = List[Float32](length=1, fill=0)
        ex.download(FP(unsafe_from_address=Int(result.unsafe_ptr())), tape.value_of(ex, count), 1)
        if result[0] != 0:
            raise Error("ValueError: projected LSTM lengths must be finite integers between 1 and the number of steps")


def _bias[E: Exec](mut tape: Tape, mut ex: E, bias: Tensor, size: Int) raises -> Tensor:
    var shape = tape.shape_of(bias)
    if shape.count() != size or shape.ndim > 2 or (shape.rows() != 1 and shape.cols() != 1):
        raise Error("ValueError: each LSTM bias must be a vector with 4 * hidden_size values")
    return tape.view_nd(ex, bias, ShapeN.matrix(1, size))


def _state[E: Exec](mut tape: Tape, mut ex: E, initial: Optional[Tensor], B: Int, width: Int) raises -> Tensor:
    if not initial:
        return _filled(tape, ex, B, width, Float32(0))
    var shape = tape.shape_of(initial.value())
    if shape.ndim == 3 and shape.d[0] == 1 and shape.d[1] == B and shape.d[2] == width:
        return tape.view_nd(ex, initial.value(), ShapeN.matrix(B, width))
    if shape.ndim != 2 or shape.d[0] != B or shape.d[1] != width:
        raise Error("ValueError: projected LSTM initial state must have shape (" + String(B) + ", " + String(width) + ")")
    return initial.value()


def projected_lstm[E: Exec](
    mut tape: Tape, mut ex: E, xs: Tensor, weight_ih: Tensor, weight_hh: Tensor,
    bias_ih: Tensor, bias_hh: Tensor, weight_hr: Tensor, reverse: Bool,
    h0: Optional[Tensor] = None, c0: Optional[Tensor] = None, lengths: Optional[Tensor] = None,
) raises -> Tuple[Tensor, Tensor, Tensor]:
    """Return sequence, final hidden and final cell as differentiable tensors.

    Each step depends on its predecessor. Each matrix product and gate applies
    over the entire batch on the selected executor, with the same shapes in a
    single-step call as in a sequence. Padded inputs are selected out before
    the input product, including NaNs. A padded step carries both states.
    """
    var shape = tape.shape_of(xs)
    if shape.ndim != 3:
        raise Error("ValueError: projected LSTM expects (steps, batch, inputs)")
    var T = shape.d[0]
    var B = shape.d[1]
    var F = shape.d[2]
    var hr = tape.shape_of(weight_hr)
    if hr.ndim != 2:
        raise Error("ValueError: weight_hr must be (proj_size, hidden_size)")
    var P = hr.d[0]
    var H = hr.d[1]
    if T < 1 or B < 1 or F < 1 or P < 1 or P >= H:
        raise Error("ValueError: projected LSTM requires positive dimensions and 0 < proj_size < hidden_size")
    var wi = tape.shape_of(weight_ih)
    var wh = tape.shape_of(weight_hh)
    if wi.ndim != 2 or wi.d[0] != 4 * H or wi.d[1] != F:
        raise Error("ValueError: weight_ih must be (4 * hidden_size, inputs)")
    if wh.ndim != 2 or wh.d[0] != 4 * H or wh.d[1] != P:
        raise Error("ValueError: weight_hh must be (4 * hidden_size, proj_size)")
    var bi = _bias(tape, ex, bias_ih, 4 * H)
    var bh = _bias(tape, ex, bias_hh, 4 * H)
    var h = _state(tape, ex, h0, B, P)
    var c = _state(tape, ex, c0, B, H)
    var hw = tape.transpose(ex, weight_hh)
    var iw = tape.transpose(ex, weight_ih)
    var projection = tape.transpose(ex, weight_hr)
    var xflat = tape.view_nd(ex, xs, ShapeN.matrix(T * B, F))
    var output = List[Tensor](length=T, fill=h)
    var mask = Optional[Tensor](None)
    var zero = Optional[Tensor](None)
    if lengths:
        if tape.shape_of(lengths.value()).count() != B:
            raise Error("ValueError: projected LSTM lengths must have one entry per batch row")
        var positions = tape.reserve(ex, T)
        var cfg = Args()
        cfg.p0 = positions
        ex.launch[OP_SEQ_IOTA](cfg, T)
        var steps = tape.variable(ex, positions, T, 1, False)
        var lens = tape.view_nd(ex, lengths.value(), ShapeN.matrix(1, B))
        mask = imp_binary(tape, ex, steps, lens, IMP_GE)
        zero = _filled(tape, ex, 1, 1, Float32(0))
    for step in range(T):  # driver-loop: over recurrent time steps
        var t = T - 1 - step if reverse else step
        var x = rankn_narrow(tape, ex, xflat, 0, t * B, B)
        var padded = Optional[Tensor](None)
        if mask:
            var row = rankn_narrow(tape, ex, mask.value(), 0, t, 1)
            padded = tape.view_nd(ex, row, ShapeN.matrix(B, 1))
            x = imp_where(tape, ex, padded.value(), zero.value(), x)
        var gx = tape.linear(ex, x, iw, bi)
        var gh = tape.linear(ex, h, hw, bh)
        var gates = tape.add(ex, gx, gh)
        var i = tape.sigmoid(ex, rankn_narrow(tape, ex, gates, 1, 0, H))
        var f = tape.sigmoid(ex, rankn_narrow(tape, ex, gates, 1, H, H))
        var g = tape.tanh(ex, rankn_narrow(tape, ex, gates, 1, 2 * H, H))
        var o = tape.sigmoid(ex, rankn_narrow(tape, ex, gates, 1, 3 * H, H))
        var next_c = tape.add(ex, tape.multiply(ex, f, c), tape.multiply(ex, i, g))
        var unprojected = tape.multiply(ex, o, tape.tanh(ex, next_c))
        var next_h = tape.matmul(ex, unprojected, projection)
        if padded:
            h = imp_where(tape, ex, padded.value(), h, next_h)
            c = imp_where(tape, ex, padded.value(), c, next_c)
        else:
            h = next_h
            c = next_c
        output[t] = h
    var flat = h if T == 1 else rankn_concat(tape, ex, output, 0)
    var dims: List[Int] = [T, B, P]
    return (tape.view_nd(ex, flat, rn_shape(dims, "projected LSTM")), h, c)
