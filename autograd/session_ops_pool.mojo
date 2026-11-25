# SPDX-License-Identifier: Apache-2.0
# SEQUENTIAL'S POOLS AS SESSION OPS (sub-lane session-ops-pool, 2026-11-25).
#
# Session codes 140..149 (autograd/session_layer_rules.mojo OPS_LAYER_POOL_FIRST), reached through
# OpsSession._layer_pool (autograd/ops_api.mojo) on the op chain's no-match path. Each code calls the
# forward function Sequential's layer calls, so it records the SAME tape nodes and takes the same bits:
#
#   code  Session method               forward function (the layer that uses it)
#   140   max_pool1d                   rankn_max_pool1d (Sequential 16); with dilation / ceil_mode the same
#                                      max pool node (_max_pool over (N, C, 1, L)) with those words
#   141   avg_pool1d                   rankn_avg_pool1d (Sequential 17); with ceil_mode / count_include_pad=False
#                                      the same avg pool node with those words
#   142   max_pool2d with options      _max_pool, the node of rankn_max_pool2d (Sequential 13) and of max_pool3d,
#                                      with per-axis windows, padding, dilation and ceil_mode in its PP_ words
#   143   avg_pool2d with options      _avg_pool likewise (padding, ceil_mode, count_include_pad,
#                                      divisor_override)
#   144   max_pool3d                   rankn_max_pool3d (Sequential 74)
#   145   avg_pool3d                   rankn_avg_pool3d (Sequential 75)
#   146   adaptive_avg_pool3d          rankn_adaptive_avg_pool3d (Sequential 76)
#   147   adaptive_avg_pool1d / 2d     1-D: Sequential 18's own dispatch (tape.mean for output 1, rankn_avg_pool1d
#                                      for an output that divides the length) and nn_adaptive_avg_pool1d for any
#                                      other length (PyTorch's windows); 2-D: nn_adaptive_avg_pool2d (Sequential 58)
#   148   adaptive_max_pool1d / 2d     nn_adaptive_max_pool1d / nn_adaptive_max_pool2d (Sequential 60, 59)
#   149   unused (a replay refuses it by name)
#
# A plain max_pool2d / avg_pool2d call (square window, no padding, no dilation, floor mode, the default
# count_include_pad and divisor) keeps its own code 45 / 46 and the branch it ran before: the ops door chooses 142
# / 143 at UNPACK only when the call uses an option the old code lacks.
#
# Backward: the rank-N pool nodes (RN_MAXPOOL2D, RN_AVGPOOL2D) differentiate with the rules every session already
# has (ImportRules / RankN); the adaptive nodes (NNC_TAG_ADAPT_AVG / _MAX) with NnConvRules through LayerRules
# (autograd/session_layer_rules.mojo). No new tag, no new kernel.
#
# LAUNCHES (each the layer's own): a max or avg pool 1 forward, 1 backward; adaptive avg 2 forward (blocked
# window fold, then totals and one division), 1 backward; adaptive max 1 and 1; the global mean (output 1 of
# adaptive_avg_pool1d / 3d) the tape's blocked fold. Host work here: shape metadata and the parameter block, once
# per recorded node.
#
# NOT EXPOSED (no kernel; listed in docs/plans/SESSION_OPS_PARITY.md): adaptive_max_pool3d; adaptive_avg_pool3d
# with an output size that neither is 1 nor divides the input; return_indices=True (the max pools' winners are held
# in PyTorch's order, but a Session operation returns one tensor).
from autograd.tape import Tape, Tensor
from autograd.rankn import ShapeN
from autograd.rankn_conv import (
    _max_pool, _avg_pool, _shape3, _shape4, rn_flat_depth, rankn_max_pool1d, rankn_avg_pool1d,
)
from autograd.nn_conv3d import Dhw, rankn_max_pool3d, rankn_avg_pool3d, rankn_adaptive_avg_pool3d
from autograd.nn_conv import (
    nn_adaptive_avg_pool1d, nn_adaptive_avg_pool2d, nn_adaptive_max_pool1d, nn_adaptive_max_pool2d,
)
from sequence.exec_trait import Exec
from x_cnn.ops import pool_params, PP_OW
from x_cnn.shape_words import _pool_out

comptime OPS_POOL_MAX1D = 140
comptime OPS_POOL_AVG1D = 141
comptime OPS_POOL_MAX2D = 142
comptime OPS_POOL_AVG2D = 143
comptime OPS_POOL_MAX3D = 144
comptime OPS_POOL_AVG3D = 145
comptime OPS_POOL_ADAPTIVE_AVG3D = 146
comptime OPS_POOL_ADAPTIVE_AVG = 147
comptime OPS_POOL_ADAPTIVE_MAX = 148
comptime OPS_POOL_LAST = OPS_POOL_ADAPTIVE_MAX  # 149 is unused


def pool_int_count(code: Int) -> Int:
    """The integer words a pool code takes (the ops door writes them; autograd/ops_api.mojo `_layer_pool` checks
    the count before reading them)."""
    if code == OPS_POOL_MAX1D or code == OPS_POOL_AVG1D:
        return 5    # [k, s, p, d, ceil] / [k, s, p, ceil, count_include_pad]
    if code == OPS_POOL_MAX2D or code == OPS_POOL_AVG2D:
        return 9    # [kh, kw, sh, sw, ph, pw, dh, dw, ceil] / [kh, kw, sh, sw, ph, pw, ceil, cip, divisor]
    if code == OPS_POOL_MAX3D:
        return 13   # [kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw, ceil]
    if code == OPS_POOL_AVG3D:
        return 12   # [kd, kh, kw, sd, sh, sw, pd, ph, pw, ceil, cip, divisor]
    if code == OPS_POOL_ADAPTIVE_AVG3D:
        return 3    # [od, oh, ow], 0 keeps the input's size
    return 3        # adaptive avg / max: [rank 1 or 2, o0, o1], 0 keeps the input's size


def pool_op_name(code: Int, ints: List[Int]) -> String:
    """The PyTorch name of a pool code (refusals name the call the user made)."""
    if code == OPS_POOL_MAX1D:
        return "max_pool1d"
    if code == OPS_POOL_AVG1D:
        return "avg_pool1d"
    if code == OPS_POOL_MAX2D:
        return "max_pool2d"
    if code == OPS_POOL_AVG2D:
        return "avg_pool2d"
    if code == OPS_POOL_MAX3D:
        return "max_pool3d"
    if code == OPS_POOL_AVG3D:
        return "avg_pool3d"
    if code == OPS_POOL_ADAPTIVE_AVG3D:
        return "adaptive_avg_pool3d"
    var two = len(ints) > 0 and ints[0] == 2
    if code == OPS_POOL_ADAPTIVE_AVG:
        return "adaptive_avg_pool2d" if two else "adaptive_avg_pool1d"
    if code == OPS_POOL_ADAPTIVE_MAX:
        return "adaptive_max_pool2d" if two else "adaptive_max_pool1d"
    return "pool code " + String(code)


def _rank(tape: Tape, x: Tensor, ndim: Int, what: String, layout: String) raises -> ShapeN:
    var s = tape.shape_of(x)
    if s.ndim != ndim:
        raise Error("ValueError: " + what + ": x must be " + layout + ", got a tensor of " + String(s.ndim)
                    + " dimensions")
    return s


def _check_axis(what: String, size: Int, k: Int, s: Int, p: Int, d: Int, ceil: Bool) raises:
    """PyTorch's admission of one pooled axis, in words, before the parameter block is built."""
    if k < 1 or s < 1 or d < 1:
        raise Error("ValueError: " + what + ": kernel_size, stride and dilation must be positive")
    if p < 0 or 2 * p > k:
        raise Error("ValueError: " + what + ": padding must be non-negative and at most half of kernel_size (padding "
                    + String(p) + ", kernel_size " + String(k) + ")")
    if _pool_out(size, k, p, s, d, ceil) < 1:
        raise Error("ValueError: " + what + ": the window does not fit the padded input (size " + String(size)
                    + ", kernel_size " + String(k) + ", padding " + String(p) + ", dilation " + String(d) + ")")


def _block2(sx: ShapeN, kh: Int, kw: Int, sh: Int, sw: Int, ph: Int, pw: Int, dh: Int, dw: Int, ceil: Bool,
            include_pad: Bool, divisor: Int) raises -> List[Int32]:
    """x_cnn's PP_ words of a pool over the (H, W) plane of sx (N, C, H, W): the words _pool_block_hw writes for the
    defaults, with the options in their slots."""
    # [N, C, H, W, KH, KW, SH, SW, PH, PW, DH, DW, OH, OW, count_include_pad, reverse, ceil_mode, divisor_override]
    var raw: List[Int] = [sx.d[0], sx.d[1], sx.d[2], sx.d[3], kh, kw, sh, sw, ph, pw, dh, dw, 0, 0,
                          1 if include_pad else 0, 0, 1 if ceil else 0, divisor]
    return pool_params(raw)


def _pool1d_words[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, s3: ShapeN, k: Int, s: Int, p: Int, d: Int, ceil: Bool, include_pad: Bool,
    is_max: Bool,
) raises -> Tensor:
    """The 1-D pool node with options: the same node rankn_max_pool1d / rankn_avg_pool1d record (the 2-D pool over
    the (N, C, 1, L) view), its words carrying dilation, ceil_mode and count_include_pad."""
    var sx = _shape4(s3.d[0], s3.d[1], 1, s3.d[2])
    var x4 = tape.view_nd(ex, x, sx)
    var block = _block2(sx, 1, k, 1, s, 0, p, 1, d, ceil, include_pad, 0)
    var y4: Tensor
    if is_max:
        y4 = _max_pool(tape, ex, x4, sx, block, rn_flat_depth())
    else:
        y4 = _avg_pool(tape, ex, x4, sx, block, rn_flat_depth())
    return tape.view_nd(ex, y4, _shape3(s3.d[0], s3.d[1], Int(block[PP_OW])))


def _keep(v: Int, size: Int) -> Int:
    """An adaptive output size: 0 (PyTorch's None) keeps the input's."""
    return size if v == 0 else v


def session_pool[E: Exec](mut tape: Tape, mut ex: E, code: Int, x: Tensor, ints: List[Int]) raises -> Tensor:
    """Session codes 140..148 (the table above). `ints` holds pool_int_count(code) words, checked by the caller."""
    var what = pool_op_name(code, ints)
    if code == OPS_POOL_MAX1D or code == OPS_POOL_AVG1D:
        var s3 = _rank(tape, x, 3, what, "(N, C, L)")
        var k = ints[0]
        var s = ints[1]
        var p = ints[2]
        var is_max = code == OPS_POOL_MAX1D
        var d = ints[3] if is_max else 1
        var ceil = (ints[4] if is_max else ints[3]) != 0
        var include_pad = True if is_max else ints[4] != 0
        _check_axis(what, s3.d[2], k, s, p, d, ceil)
        if d == 1 and not ceil and include_pad:
            # Sequential's own layer (max_pool1d 16 / avg_pool1d 17): its function, its words
            if is_max:
                return rankn_max_pool1d(tape, ex, x, k, s, p)
            return rankn_avg_pool1d(tape, ex, x, k, s, p)
        return _pool1d_words(tape, ex, x, s3, k, s, p, d, ceil, include_pad, is_max)
    if code == OPS_POOL_MAX2D or code == OPS_POOL_AVG2D:
        var sx = _rank(tape, x, 4, what, "(N, C, H, W)")
        var is_max = code == OPS_POOL_MAX2D
        var dh = ints[6] if is_max else 1
        var dw = ints[7] if is_max else 1
        var ceil = (ints[8] if is_max else ints[6]) != 0
        var include_pad = True if is_max else ints[7] != 0
        var divisor = 0 if is_max else ints[8]
        if divisor < 0:
            raise Error("ValueError: " + what + ": divisor_override must be a positive integer")
        _check_axis(what, sx.d[2], ints[0], ints[2], ints[4], dh, ceil)
        _check_axis(what, sx.d[3], ints[1], ints[3], ints[5], dw, ceil)
        var block = _block2(sx, ints[0], ints[1], ints[2], ints[3], ints[4], ints[5], dh, dw, ceil, include_pad,
                            divisor)
        if is_max:
            return _max_pool(tape, ex, x, sx, block, rn_flat_depth())
        return _avg_pool(tape, ex, x, sx, block, rn_flat_depth())
    if code == OPS_POOL_MAX3D:
        var sx = _rank(tape, x, 5, what, "(N, C, D, H, W)")
        var ceil = ints[12] != 0
        for a in range(3):  # small-loop(three pooled axes): admission words
            _check_axis(what, sx.d[2 + a], ints[a], ints[3 + a], ints[6 + a], ints[9 + a], ceil)
        return rankn_max_pool3d(tape, ex, x, Dhw(ints[0], ints[1], ints[2]), Dhw(ints[3], ints[4], ints[5]),
                                Dhw(ints[6], ints[7], ints[8]), Dhw(ints[9], ints[10], ints[11]), ceil)
    if code == OPS_POOL_AVG3D:
        var sx = _rank(tape, x, 5, what, "(N, C, D, H, W)")
        var ceil = ints[9] != 0
        if ints[11] < 0:
            raise Error("ValueError: " + what + ": divisor_override must be a positive integer")
        for a in range(3):  # small-loop(three pooled axes): admission words
            _check_axis(what, sx.d[2 + a], ints[a], ints[3 + a], ints[6 + a], 1, ceil)
        return rankn_avg_pool3d(tape, ex, x, Dhw(ints[0], ints[1], ints[2]), Dhw(ints[3], ints[4], ints[5]),
                                Dhw(ints[6], ints[7], ints[8]), ceil, ints[10] != 0, ints[11])
    for a in range(len(ints)):  # small-loop(at most three output sizes): admission words
        if (code == OPS_POOL_ADAPTIVE_AVG3D or a > 0) and ints[a] < 0:
            raise Error("ValueError: " + what + ": output_size must be positive (or None to keep the input's size)")
    if code == OPS_POOL_ADAPTIVE_AVG3D:
        var sx = _rank(tape, x, 5, what, "(N, C, D, H, W)")
        # rankn_adaptive_avg_pool3d refuses, in words, an output size that is neither 1 nor a divisor
        return rankn_adaptive_avg_pool3d(tape, ex, x, Dhw(_keep(ints[0], sx.d[2]), _keep(ints[1], sx.d[3]),
                                                          _keep(ints[2], sx.d[4])))
    if ints[0] != 1 and ints[0] != 2:
        raise Error("ValueError: " + what + ": the rank word must be 1 or 2, got " + String(ints[0]))
    if code == OPS_POOL_ADAPTIVE_AVG:
        if ints[0] == 1:
            var s3 = _rank(tape, x, 3, what, "(N, C, L)")
            var length = s3.d[2]
            var ol = _keep(ints[1], length)
            if ol == 1:
                # Sequential 18, output 1: the global mean of each (sample, channel) row, the tape's blocked fold
                # over the length ((N C, L) rows), shape (N, C, 1)
                return tape.mean(ex, x, 1)
            if length % ol == 0:
                # Sequential 18, an output that divides the length: equal windows, an average pool
                var win = length // ol
                return rankn_avg_pool1d(tape, ex, x, win, win, 0)
            # any other length: PyTorch's overlapping windows (x_cnn's adaptive average bodies)
            return nn_adaptive_avg_pool1d(tape, ex, x, ol)
        var s4 = _rank(tape, x, 4, what, "(N, C, H, W)")
        return nn_adaptive_avg_pool2d(tape, ex, x, _keep(ints[1], s4.d[2]), _keep(ints[2], s4.d[3]))
    if code == OPS_POOL_ADAPTIVE_MAX:
        if ints[0] == 1:
            var s3 = _rank(tape, x, 3, what, "(N, C, L)")
            return nn_adaptive_max_pool1d(tape, ex, x, _keep(ints[1], s3.d[2]))
        var s4 = _rank(tape, x, 4, what, "(N, C, H, W)")
        return nn_adaptive_max_pool2d(tape, ex, x, _keep(ints[1], s4.d[2]), _keep(ints[2], s4.d[3]))
    raise Error("ops: unknown operation code " + String(code))
