# SPDX-License-Identifier: Apache-2.0
# SEQUENTIAL'S CONVOLUTIONS AS ops.Session OPERATIONS (lane session-ops-conv, 2026-11-25).
#
# Session codes 130..134 (autograd/session_layer_rules.mojo OPS_LAYER_CONV_FIRST; OpsSession._layer_conv calls
# session_conv_forward below). Each code calls the forward function Sequential's layer calls, so it records the
# same tape node (same launches, same words, same backward rule):
#
#   130 conv1d             rankn_conv1d (Sequential's conv1d, layer 15): the tape convolution over (N, C, 1, L)
#   131 conv2d, dilated    _rankn_conv (the one convolution under Sequential's conv2d, layer 12) with the
#                          height and width steps, pads and tap spacings given apart; Session.conv2d's door
#                          records it ONLY when a call passes a dilation, a per-axis stride or padding, or a
#                          padding word; the default call keeps OPS_CONV2D (44) and its branch
#   132 conv3d             rankn_conv3d (Sequential's conv3d, layer 73)
#   133 conv_transpose1d   nn_conv_transpose1d (Sequential's conv_transpose1d, layer 56)
#   134 conv_transpose2d   nn_conv_transpose2d (Sequential's conv_transpose2d, layer 57), per-axis words
#
# INTEGER WORDS (resolved once at UNPACK by the ops door, recorded with the call, so a captured Program replays
# them): a convolution of n spatial axes takes [stride * n, padding * n, dilation * n, groups]; a transposed
# convolution [stride * n, padding * n, output_padding * n, dilation * n, groups]. Axis order is PyTorch's
# (depth, height, width). Operands: [x, weight] or [x, weight, bias]; the bias (any shape holding out_channels
# values) is viewed as (out_channels,), a free view, as Session.conv2d does.
#
# BACKWARD. 130..132 record the rank-N convolution tag (autograd/rules_conv2d.mojo RN_CONV2D), which every
# session's rules already own; 133..134 record NNC_TAG_CONV_TRANSPOSE, answered by LayerRules (NnConvRules) once
# the session's layer_nodes flag is set. No new kernel: the launches are the forward functions' own.
#
# conv_transpose3d is NOT exposed: nn_conv_transpose2d's host path is rank-4 only and its output body
# (autograd/nn_conv_ops.mojo op_nnc_ct_out) indexes the bias channel over H W, not D H W; a 3-D transposed
# convolution needs a depth-aware output body (docs/plans/SESSION_OPS_PARITY.md).
#
# Host work here is shape metadata and admission, once per recorded node: no loop over data.
from autograd.tape import Tape, Tensor
from autograd.rankn_api import rn_as_shape, rn_shape
from autograd.rankn_conv import rankn_conv1d, _rankn_conv
from autograd.nn_conv3d import rankn_conv3d, Dhw
from autograd.nn_conv import nn_conv_transpose1d, nn_conv_transpose2d
from sequence.exec_trait import Exec

comptime OPS_CONV1D = 130
comptime OPS_CONV2D_ND = 131
comptime OPS_CONV3D = 132
comptime OPS_CONV_TRANSPOSE1D = 133
comptime OPS_CONV_TRANSPOSE2D = 134


def session_conv_name(code: Int) -> String:
    """The public name of a session-ops-conv code."""
    if code == OPS_CONV1D:
        return "conv1d"
    if code == OPS_CONV2D_ND:
        return "conv2d"
    if code == OPS_CONV3D:
        return "conv3d"
    if code == OPS_CONV_TRANSPOSE1D:
        return "conv_transpose1d"
    if code == OPS_CONV_TRANSPOSE2D:
        return "conv_transpose2d"
    return "op_" + String(code)


def session_conv_axes(code: Int) -> Int:
    """The number of spatial axes of a code (0 for a code this file does not own)."""
    if code == OPS_CONV1D or code == OPS_CONV_TRANSPOSE1D:
        return 1
    if code == OPS_CONV2D_ND or code == OPS_CONV_TRANSPOSE2D:
        return 2
    if code == OPS_CONV3D:
        return 3
    return 0


def session_conv_transposed(code: Int) -> Bool:
    return code == OPS_CONV_TRANSPOSE1D or code == OPS_CONV_TRANSPOSE2D


def session_conv_word_count(code: Int) -> Int:
    """The integer words a code takes (the header's layout)."""
    var n = session_conv_axes(code)
    if session_conv_transposed(code):
        return 4 * n + 1
    return 3 * n + 1


def _bias[E: Exec](mut tape: Tape, mut ex: E, x: List[Tensor], oc: Int, what: String) raises -> Optional[Tensor]:
    """The optional third operand as an (out_channels,) view (free), refused unless it holds oc values."""
    if len(x) < 3:
        return None
    if tape.shape_of(x[2]).count() != oc:
        raise Error("ValueError: ops: " + what + ": the bias needs one value per output channel ("
                    + String(oc) + ")")
    var dims = List[Int](length=1, fill=oc)
    return Optional[Tensor](rn_as_shape(tape, ex, x[2], rn_shape(dims, "ops: " + what)))


def session_conv_forward[E: Exec](
    mut tape: Tape, mut ex: E, code: Int, ints: List[Int], x: List[Tensor],
) raises -> Tensor:
    """Record session-ops-conv operation `code` on the tape: the forward function Sequential's layer calls, with
    the integer words of the file header. Refusals name the operation and the argument."""
    var what = session_conv_name(code)
    var n = session_conv_axes(code)
    if n == 0:
        raise Error("ops: unknown operation code " + String(code))
    if len(x) != 2 and len(x) != 3:
        raise Error("ValueError: ops: " + what + " takes an input, a weight and an optional bias")
    if len(ints) != session_conv_word_count(code):
        raise Error("ValueError: ops: " + what + " takes " + String(session_conv_word_count(code))
                    + " integer arguments, got " + String(len(ints)))
    var rank = n + 2
    if tape.ndim(x[0]) != rank or tape.ndim(x[1]) != rank:
        var dims = String("(N, C, L)") if n == 1 else (String("(N, C, H, W)") if n == 2 else String("(N, C, D, H, W)"))
        raise Error("ValueError: ops: " + what + " takes x " + dims + " and a rank-" + String(rank) + " weight")
    var transposed = session_conv_transposed(code)
    var groups = ints[len(ints) - 1]
    if groups < 1:
        raise Error("ValueError: ops: " + what + ": groups must be at least 1, got " + String(groups))
    # per axis: stride >= 1, padding >= 0, dilation >= 1 (and output_padding >= 0)
    var dil_at = 3 * n if transposed else 2 * n
    for k in range(n):  # small-loop(n: at most three spatial axes)
        if ints[k] < 1:
            raise Error("ValueError: ops: " + what + ": stride must be at least 1, got " + String(ints[k]))
        if ints[n + k] < 0:
            raise Error("ValueError: ops: " + what + ": padding must be non-negative, got " + String(ints[n + k]))
        if ints[dil_at + k] < 1:
            raise Error("ValueError: ops: " + what + ": dilation must be at least 1, got " + String(ints[dil_at + k]))
        if transposed and ints[2 * n + k] < 0:
            raise Error("ValueError: ops: " + what + ": output_padding must be non-negative, got "
                        + String(ints[2 * n + k]))
    var c_in = tape.dim(x[0], 1)
    if transposed:
        # weight (C_in, C_out // groups, k...): PyTorch's layout
        if tape.dim(x[1], 0) != c_in or c_in % groups != 0:
            raise Error("ValueError: ops: " + what + ": the weight's first dimension must be x's channels ("
                        + String(c_in) + ") and groups must divide it")
        for k in range(n):  # small-loop(n: at most two spatial axes)
            if ints[2 * n + k] >= ints[k]:
                raise Error("ValueError: ops: " + what + ": output_padding must be smaller than stride on every axis "
                            + "(output_padding >= stride, allowed by PyTorch when dilation is larger, is not supported)")
        var bias = _bias(tape, ex, x, tape.dim(x[1], 1) * groups, what)
        if code == OPS_CONV_TRANSPOSE1D:
            return nn_conv_transpose1d(tape, ex, x[0], x[1], bias, ints[0], ints[1], ints[2], ints[3], groups)
        return nn_conv_transpose2d(tape, ex, x[0], x[1], bias, ints[0], ints[1], ints[2], ints[3], ints[4],
                                   ints[5], ints[6], ints[7], groups)
    # weight (out_channels, C // groups, k...)
    var oc = tape.dim(x[1], 0)
    if c_in % groups != 0 or oc % groups != 0 or tape.dim(x[1], 1) * groups != c_in:
        raise Error("ValueError: ops: " + what + ": the weight must be (out_channels, C // groups, ...) with groups ("
                    + String(groups) + ") dividing C (" + String(c_in) + ") and out_channels (" + String(oc) + ")")
    for k in range(n):  # small-loop(n: at most three spatial axes)
        var size = tape.dim(x[0], 2 + k)
        var span = ints[dil_at + k] * (tape.dim(x[1], 2 + k) - 1) + 1
        if size + 2 * ints[n + k] < span:
            raise Error("ValueError: ops: " + what + ": the dilated kernel (" + String(span)
                        + ") is larger than the padded input (" + String(size + 2 * ints[n + k]) + ") on spatial axis "
                        + String(k))
    var bias = _bias(tape, ex, x, oc, what)
    if code == OPS_CONV1D:
        return rankn_conv1d(tape, ex, x[0], x[1], bias, ints[0], ints[1], groups, ints[2])
    if code == OPS_CONV2D_ND:
        return _rankn_conv(tape, ex, x[0], x[1], bias, ints[0], ints[1], ints[2], ints[3], groups, ints[4], ints[5])
    return rankn_conv3d(tape, ex, x[0], x[1], bias, Dhw(ints[0], ints[1], ints[2]), Dhw(ints[3], ints[4], ints[5]),
                        Dhw(ints[6], ints[7], ints[8]), groups)
