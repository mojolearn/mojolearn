# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""Sequential's convolutions as ops.Session operations (lane session-ops-conv, 2026-11-25):
autograd/session_ops_conv.mojo, session codes 130..134 (conv1d, conv2d with dilation or per-axis words, conv3d,
conv_transpose1d, conv_transpose2d), dispatched by OpsSession._layer_conv.

One test, test_session_ops_conv, runs independent units across host cores (one HostExec each):
  - units 0..4, one per operation on the smallest shapes that reach its edges (groups, a stride that skips
    input, padding wider than a tap, dilation on one axis only, output_padding, a bias):
      (a) the Session op in a recording OpsSession, loss = sum(y * seed), backward: y and the gradients of x,
          weight and bias, against
      (b) the forward function Sequential's layer calls (rankn_conv1d, _rankn_conv, rankn_conv3d,
          nn_conv_transpose1d / 2d) on a bare tape with the same values, seeded backward with ModelRules
          (Sequential's rules): the SAME words, value and gradient alike (the Session op is that layer's node,
          differentiated by the session's rules: ImportRules for the rank-N tag, LayerRules for the transposed
          tag);
      (c) a float64 oracle written here from PyTorch's definitions, and every gradient element against central
          differences of the float64 loss (the inputs are small dyadic numbers, so float32 holds every product
          and sum exactly and the tolerance is rounding only).
  - unit 5, conv2d with square words: code 131 (OPS_CONV2D_ND) and code 44 (OPS_CONV2D, the old branch) and
    rankn_conv2d (Sequential's conv2d) give the same words.
  - unit 6, conv1d with Sequential's own arguments (dilation 1): the Session op against rankn_conv1d as
    autograd/model.mojo calls it.
  - unit 7, the refusals in words: a dilated kernel wider than the padded input, groups not dividing the
    channels, output_padding not smaller than stride, a bias of the wrong length, a wrong-rank input.
  - one `SESSION_OPS_CONV <case> digest` line over the words of each unit.

Columns: the CPU column runs on HostExec. The device column runs each unit on DeviceExec AND HostExec and
requires the same words.
"""
from std.memory import bitcast
from std.testing import TestSuite
from autograd.tape import Tape, Tensor
from autograd.rankn import ShapeN, RN_MAX_RANK
from autograd.model_blocks import ModelRules
from autograd.ops_api import OpsSession, OPS_MUL, OPS_CONV2D
from autograd.rankn_conv import rankn_conv1d, rankn_conv2d, _rankn_conv
from autograd.nn_conv3d import rankn_conv3d, Dhw
from autograd.nn_conv import nn_conv_transpose1d, nn_conv_transpose2d
from autograd.session_ops_conv import (
    OPS_CONV1D, OPS_CONV2D_ND, OPS_CONV3D, OPS_CONV_TRANSPOSE1D, OPS_CONV_TRANSPOSE2D, session_conv_axes,
    session_conv_transposed,
)
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from checks.native_tests.rankn_cases import _upload, _binary, _gradient, _sum, _download
from checks.native_tests.testsuite_cases import CPU_COLUMN_BUILD
from core.host_parallel import host_parallelize

comptime _H = Float64(1.0) / 64.0
comptime _CASES = 5
comptime _UNITS = 8


def _fail(message: String) raises:
    print("FAIL SESSION_OPS_CONV", message)
    raise Error(message)


def _require(ok: Bool, message: String) raises:
    if not ok:
        _fail(message)


def _sh(dims: List[Int]) -> ShapeN:
    var d = Array[Int, RN_MAX_RANK](fill=1)
    for k in range(len(dims)):  # small-loop(len(dims): a tensor rank)
        d[k] = dims[k]
    return ShapeN(len(dims), d)


def _count(dims: List[Int]) -> Int:
    var n = 1
    for k in range(len(dims)):  # small-loop(len(dims): a tensor rank)
        n *= dims[k]
    return n


def _dyadic(n: Int, step: Int, modulus: Int, denom: Int) -> List[Float32]:
    var v = List[Float32](length=n, fill=0)
    for i in range(n):  # small-loop(n: check data)
        v[i] = Float32((i * step) % modulus - modulus // 2) / Float32(denom)
    return v^


def _mix(mut digest: UInt64, values: List[Float32]):
    for i in range(len(values)):  # small-loop(len(values): check digest)
        digest = (digest ^ UInt64(bitcast[DType.uint32](values[i]))) * UInt64(1099511628211)


def _close(name: String, got: List[Float32], want: List[Float64], tol: Float64) raises:
    if len(got) != len(want):
        _fail(name + ": " + String(len(got)) + " values, " + String(len(want)) + " expected")
    for i in range(len(got)):  # small-loop(len(got): check comparison)
        var d = abs(Float64(got[i]) - want[i])
        if not (d <= tol * (1.0 + abs(want[i]))):
            _fail(name + ": index " + String(i) + " got " + String(got[i]) + " want " + String(want[i]))


def _same_words(name: String, a: List[Float32], b: List[Float32]) raises:
    if len(a) != len(b):
        _fail(name + ": lengths differ " + String(len(a)) + " vs " + String(len(b)))
    for i in range(len(a)):  # small-loop(len(a): check comparison)
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            _fail(name + ": words differ at " + String(i) + " (" + String(a[i]) + " vs " + String(b[i]) + ")")


# ---------------------------------------------------------------- cases
struct _Case(Copyable, Movable):
    var name: String
    var code: Int
    var x: List[Int]      # (N, C, spatial...)
    var w: List[Int]      # conv (OC, C // g, k...); transposed (C_in, C_out // g, k...)
    var bias: Bool
    var ints: List[Int]   # the session words (autograd/session_ops_conv.mojo header)

    def __init__(out self, name: String, code: Int, x: List[Int], w: List[Int], bias: Bool, ints: List[Int]):
        self.name = name
        self.code = code
        self.x = x.copy()
        self.w = w.copy()
        self.bias = bias
        self.ints = ints.copy()


def _cases() -> List[_Case]:
    var out = List[_Case]()
    # conv1d: groups 2, stride 2 (skips input), padding 2, dilation 2; L_out = (7 + 4 - 4 - 1) // 2 + 1 = 4
    out.append(_Case("conv1d", OPS_CONV1D, [2, 4, 7], [4, 2, 3], True, [2, 2, 2, 2]))
    # conv2d dilated on height only, stride on width only, padding on height only: (1, 3, 7, 2)
    out.append(_Case("conv2d_dilated", OPS_CONV2D_ND, [1, 2, 5, 6], [3, 2, 2, 3], True, [1, 2, 2, 0, 2, 1, 1]))
    # conv3d: groups 2 (depthwise), stride (1, 2, 1), padding (1, 0, 1), dilation (1, 1, 2): (1, 2, 4, 2, 4)
    out.append(_Case("conv3d", OPS_CONV3D, [1, 2, 3, 4, 4], [2, 1, 2, 2, 2], True,
                     [1, 2, 1, 1, 0, 1, 1, 1, 2, 2]))
    # conv_transpose1d: stride 2, padding 1, output_padding 1: L_out = 3 * 2 - 2 + 2 + 1 + 1 = 8
    out.append(_Case("conv_transpose1d", OPS_CONV_TRANSPOSE1D, [1, 2, 4], [2, 3, 3], True, [2, 1, 1, 1, 1]))
    # conv_transpose2d: groups 2, stride (2, 1), padding (1, 0), output_padding (1, 0), dilation (1, 2): (1, 4, 6, 5)
    out.append(_Case("conv_transpose2d", OPS_CONV_TRANSPOSE2D, [1, 2, 3, 3], [2, 2, 3, 2], True,
                     [2, 1, 1, 0, 1, 0, 1, 2, 2]))
    return out^


def _axis3(c: _Case, block: Int, fill_value: Int) -> List[Int]:
    """Words block `block` (0 stride, 1 padding, 2 dilation or output_padding, 3 a transposed dilation), as three
    axes (D, H, W) with the missing leading axes at `fill_value`."""
    var n = session_conv_axes(c.code)
    var out = List[Int](length=3, fill=fill_value)
    for k in range(n):  # small-loop(n: at most three spatial axes)
        out[3 - n + k] = c.ints[block * n + k]
    return out^


def _spatial3(dims: List[Int]) -> List[Int]:
    var n = len(dims) - 2
    var out = List[Int](length=3, fill=1)
    for k in range(n):  # small-loop(n: at most three spatial axes)
        out[3 - n + k] = dims[2 + k]
    return out^


def _groups(c: _Case) -> Int:
    return c.ints[len(c.ints) - 1]


def _out_channels(c: _Case) -> Int:
    if session_conv_transposed(c.code):
        return c.w[1] * _groups(c)
    return c.w[0]


def _out_dims(c: _Case) -> List[Int]:
    var n = session_conv_axes(c.code)
    var X = _spatial3(c.x)
    var K = _spatial3(c.w)
    var S = _axis3(c, 0, 1)
    var P = _axis3(c, 1, 0)
    var out: List[Int] = [c.x[0], _out_channels(c)]
    for j in range(3 - n, 3):  # small-loop(n: at most three spatial axes)
        if session_conv_transposed(c.code):
            var OP = _axis3(c, 2, 0)
            var D = _axis3(c, 3, 1)
            out.append((X[j] - 1) * S[j] - 2 * P[j] + D[j] * (K[j] - 1) + OP[j] + 1)
        else:
            var D = _axis3(c, 2, 1)
            out.append((X[j] + 2 * P[j] - D[j] * (K[j] - 1) - 1) // S[j] + 1)
    return out^


def _inputs(c: _Case) -> List[List[Float32]]:
    var v = List[List[Float32]]()
    v.append(_dyadic(_count(c.x), 7, 19, 8))
    v.append(_dyadic(_count(c.w), 5, 13, 16))
    if c.bias:
        v.append(_dyadic(_out_channels(c), 3, 7, 4))
    return v^


def _wide(vals: List[List[Float32]]) -> List[List[Float64]]:
    var out = List[List[Float64]]()
    for a in range(len(vals)):  # small-loop(at most 3 inputs)
        var w = List[Float64](length=len(vals[a]), fill=0)
        for i in range(len(vals[a])):  # small-loop(len: check data)
            w[i] = Float64(vals[a][i])
        out.append(w^)
    return out^


# ---------------------------------------------------------------- the float64 oracle (PyTorch's definitions)
def _oracle(c: _Case, v: List[List[Float64]]) -> List[Float64]:
    var N = c.x[0]
    var C = c.x[1]
    var X = _spatial3(c.x)
    var K = _spatial3(c.w)
    var S = _axis3(c, 0, 1)
    var P = _axis3(c, 1, 0)
    var G = _groups(c)
    var od = _out_dims(c)
    var O = _spatial3(od)
    var OC = od[1]
    var y = List[Float64](length=_count(od), fill=0)
    if session_conv_transposed(c.code):
        # y[n, g Cog + co, o] += x[n, ci, i] w[ci, co, k] for o = i s - p + k d (PyTorch's conv_transpose)
        var D = _axis3(c, 3, 1)
        var Cog = c.w[1]
        var Cig = C // G
        for n in range(N):  # small-loop(check oracle)
            for ci in range(C):  # small-loop(check oracle)
                var g = ci // Cig
                for i0 in range(X[0]):  # small-loop(check oracle)
                    for i1 in range(X[1]):  # small-loop(check oracle)
                        for i2 in range(X[2]):  # small-loop(check oracle)
                            var xv = v[0][(((n * C + ci) * X[0] + i0) * X[1] + i1) * X[2] + i2]
                            for co in range(Cog):  # small-loop(check oracle)
                                for k0 in range(K[0]):  # small-loop(check oracle)
                                    for k1 in range(K[1]):  # small-loop(check oracle)
                                        for k2 in range(K[2]):  # small-loop(check oracle)
                                            var o0 = i0 * S[0] - P[0] + k0 * D[0]
                                            var o1 = i1 * S[1] - P[1] + k1 * D[1]
                                            var o2 = i2 * S[2] - P[2] + k2 * D[2]
                                            if o0 < 0 or o0 >= O[0] or o1 < 0 or o1 >= O[1] or o2 < 0 or o2 >= O[2]:
                                                continue
                                            var wv = v[1][(((ci * Cog + co) * K[0] + k0) * K[1] + k1) * K[2] + k2]
                                            var oc = g * Cog + co
                                            y[(((n * OC + oc) * O[0] + o0) * O[1] + o1) * O[2] + o2] += xv * wv
    else:
        # y[n, oc, o] = sum over the group's channels and taps of x[n, c, o s - p + k d] w[oc, c, k]
        var D = _axis3(c, 2, 1)
        var Cg = c.w[1]
        var OCg = OC // G
        for n in range(N):  # small-loop(check oracle)
            for oc in range(OC):  # small-loop(check oracle)
                var g = oc // OCg
                for o0 in range(O[0]):  # small-loop(check oracle)
                    for o1 in range(O[1]):  # small-loop(check oracle)
                        for o2 in range(O[2]):  # small-loop(check oracle)
                            var acc = Float64(0)
                            for ci in range(Cg):  # small-loop(check oracle)
                                for k0 in range(K[0]):  # small-loop(check oracle)
                                    for k1 in range(K[1]):  # small-loop(check oracle)
                                        for k2 in range(K[2]):  # small-loop(check oracle)
                                            var i0 = o0 * S[0] - P[0] + k0 * D[0]
                                            var i1 = o1 * S[1] - P[1] + k1 * D[1]
                                            var i2 = o2 * S[2] - P[2] + k2 * D[2]
                                            if i0 < 0 or i0 >= X[0] or i1 < 0 or i1 >= X[1] or i2 < 0 or i2 >= X[2]:
                                                continue
                                            var c_in = g * Cg + ci
                                            acc += (v[0][(((n * C + c_in) * X[0] + i0) * X[1] + i1) * X[2] + i2]
                                                    * v[1][(((oc * Cg + ci) * K[0] + k0) * K[1] + k1) * K[2] + k2])
                            y[(((n * OC + oc) * O[0] + o0) * O[1] + o1) * O[2] + o2] = acc
    if c.bias:
        var per = O[0] * O[1] * O[2]
        for t in range(len(y)):  # small-loop(check oracle)
            y[t] += v[2][(t // per) % OC]
    return y^


def _oracle_loss(c: _Case, v: List[List[Float64]], seed: List[Float32]) -> Float64:
    var y = _oracle(c, v)
    var total = Float64(0)
    for i in range(len(y)):  # small-loop(check oracle)
        total += y[i] * Float64(seed[i])
    return total


# ---------------------------------------------------------------- the two sides
def _session_side[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], code: Int, ints: List[Int], c: _Case, vals: List[List[Float32]], seed: List[Float32],
) raises -> List[Float32]:
    """The Session op: y, then the gradients of sum(y * seed) for x, weight[, bias]."""
    var hs = List[Int]()
    hs.append(_upload(s, vals[0], c.x, True))
    hs.append(_upload(s, vals[1], c.w, True))
    if c.bias:
        var bd = List[Int](length=1, fill=_out_channels(c))
        hs.append(_upload(s, vals[2], bd, True))
    var y = s.run_op(code, hs, ints, List[Float32]())
    var od = _out_dims(c)
    _require(s.shape_nd(y).same(_sh(od)), c.name + ": Session op output shape")
    var g = _upload(s, seed, od)
    var loss = _sum(s, _binary(s, OPS_MUL, y, g))
    s.backward(loss, False)
    var words = _download(s, y, _count(od))
    for a in range(len(hs)):  # small-loop(at most 3 inputs)
        var n = _count(c.x) if a == 0 else (_count(c.w) if a == 1 else _out_channels(c))
        var gr = _gradient(s, hs[a], n)
        for i in range(n):  # small-loop(n: check words)
            words.append(gr[i])
    return words^


def _direct[E: Exec](
    mut ex: E, code: Int, ints: List[Int], c: _Case, vals: List[List[Float32]], seed: List[Float32],
) raises -> List[Float32]:
    """The forward function Sequential's layer calls, on a bare tape, seeded backward with ModelRules (Sequential's
    rules). OPS_CONV2D (44) with words [stride, padding, groups] is rankn_conv2d as autograd/model.mojo calls it;
    OPS_CONV1D with dilation 1 is rankn_conv1d as autograd/model.mojo calls it (no dilation argument)."""
    var tape = Tape()
    var ins = List[Tensor]()
    var x = tape.tensor_nd(ex, vals[0], _sh(c.x), True)
    var w = tape.tensor_nd(ex, vals[1], _sh(c.w), True)
    ins.append(x)
    ins.append(w)
    var bias = Optional[Tensor]()
    if c.bias:
        var bd = List[Int](length=1, fill=_out_channels(c))
        var b = tape.tensor_nd(ex, vals[2], _sh(bd), True)
        ins.append(b)
        bias = Optional[Tensor](b)
    var i = ints.copy()
    var g = i[len(i) - 1]
    var y: Tensor
    if code == OPS_CONV2D:
        y = rankn_conv2d(tape, ex, x, w, bias, i[0], i[1], g)
    elif code == OPS_CONV1D and i[2] == 1:
        y = rankn_conv1d(tape, ex, x, w, bias, i[0], i[1], g)
    elif code == OPS_CONV1D:
        y = rankn_conv1d(tape, ex, x, w, bias, i[0], i[1], g, i[2])
    elif code == OPS_CONV2D_ND:
        y = _rankn_conv(tape, ex, x, w, bias, i[0], i[1], i[2], i[3], g, i[4], i[5])
    elif code == OPS_CONV3D:
        y = rankn_conv3d(tape, ex, x, w, bias, Dhw(i[0], i[1], i[2]), Dhw(i[3], i[4], i[5]), Dhw(i[6], i[7], i[8]), g)
    elif code == OPS_CONV_TRANSPOSE1D:
        y = nn_conv_transpose1d(tape, ex, x, w, bias, i[0], i[1], i[2], i[3], g)
    else:
        y = nn_conv_transpose2d(tape, ex, x, w, bias, i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], g)
    var words = tape.values(ex, y)
    var st = tape.tensor(ex, seed, tape.rows(y), tape.cols(y))
    tape.backward_with_rules(ex, y, st, ModelRules())
    for a in range(len(ins)):  # small-loop(at most 3 inputs)
        var gr = tape.gradients(ex, ins[a])
        for k in range(len(gr)):  # small-loop(len(gr): check words)
            words.append(gr[k])
    return words^


def _session_closed[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], code: Int, ints: List[Int], c: _Case, vals: List[List[Float32]], seed: List[Float32],
) raises -> List[Float32]:
    try:
        var words = _session_side(s, code, ints, c, vals, seed)
        s.close()
        return words^
    except error:
        s.close()
        raise error


def _session_words(
    device: Bool, code: Int, ints: List[Int], c: _Case, vals: List[List[Float32]], seed: List[Float32],
) raises -> List[Float32]:
    """The Session op's words in a fresh recording session on the device (device column) or the host."""
    comptime if not CPU_COLUMN_BUILD:
        if device:
            var d = OpsSession[DeviceExec](DeviceExec(), True)
            return _session_closed(d, code, ints, c, vals, seed)
    var h = OpsSession[HostExec](HostExec(), True)
    return _session_closed(h, code, ints, c, vals, seed)


def _direct_words(
    device: Bool, code: Int, ints: List[Int], c: _Case, vals: List[List[Float32]], seed: List[Float32],
) raises -> List[Float32]:
    comptime if not CPU_COLUMN_BUILD:
        if device:
            var dex = DeviceExec()
            return _direct(dex, code, ints, c, vals, seed)
    var host_ex = HostExec()
    return _direct(host_ex, code, ints, c, vals, seed)


def run_case(device: Bool, c: _Case) raises -> List[Float32]:
    """(a) == (b) word for word; (c) against the float64 oracle and central differences."""
    var vals = _inputs(c)
    var wide = _wide(vals)
    var od = _out_dims(c)
    var seed = _dyadic(_count(od), 3, 11, 16)
    var session = _session_words(device, c.code, c.ints, c, vals, seed)
    var direct = _direct_words(device, c.code, c.ints, c, vals, seed)
    _same_words(c.name + ": Session op vs Sequential's forward function", session, direct)
    var ny = _count(od)
    var yv = List[Float32](capacity=ny)
    for i in range(ny):  # small-loop(ny: check words)
        yv.append(session[i])
    _close(c.name + " forward", yv, _oracle(c, wide), Float64(1e-6))
    var at = ny
    for a in range(len(vals)):  # small-loop(at most 3 inputs)
        var n = len(vals[a])
        var gr = List[Float32](capacity=n)
        var fd = List[Float64](length=n, fill=0)
        for i in range(n):  # small-loop(n: every element probed)
            gr.append(session[at + i])
            var up = wide.copy()
            up[a][i] = wide[a][i] + _H
            var down = wide.copy()
            down[a][i] = wide[a][i] - _H
            fd[i] = (_oracle_loss(c, up, seed) - _oracle_loss(c, down, seed)) / (2.0 * _H)
        _close(c.name + " gradient " + String(a), gr, fd, Float64(1e-5))
        at += n
    var digest = UInt64(1469598103934665603)
    _mix(digest, session)
    print("SESSION_OPS_CONV", c.name, "digest", digest)
    return session^


def run_square(device: Bool) raises -> List[Float32]:
    """conv2d with square words: code 131, code 44 and rankn_conv2d (Sequential's conv2d) are the same words."""
    var c = _Case("conv2d_square", OPS_CONV2D_ND, [2, 2, 5, 5], [3, 2, 3, 3], True, [2, 2, 1, 1, 1, 1, 1])
    var vals = _inputs(c)
    var seed = _dyadic(_count(_out_dims(c)), 3, 11, 16)
    var nd = _session_words(device, OPS_CONV2D_ND, c.ints, c, vals, seed)
    var old_words: List[Int] = [2, 1, 1]
    var old = _session_words(device, OPS_CONV2D, old_words, c, vals, seed)
    _same_words("conv2d code 131 vs code 44", nd, old)
    var seq = _direct_words(device, OPS_CONV2D, old_words, c, vals, seed)
    _same_words("conv2d code 131 vs rankn_conv2d (Sequential's conv2d)", nd, seq)
    var digest = UInt64(1469598103934665603)
    _mix(digest, nd)
    print("SESSION_OPS_CONV conv2d_square digest", digest)
    return nd^


def run_sequential_conv1d(device: Bool) raises -> List[Float32]:
    """conv1d with Sequential's own arguments (no dilation): the Session op against rankn_conv1d as
    autograd/model.mojo calls it (stride, padding, groups; dilation left at its default)."""
    var c = _Case("conv1d_sequential", OPS_CONV1D, [2, 3, 9], [6, 1, 3], True, [2, 1, 1, 3])
    var vals = _inputs(c)
    var seed = _dyadic(_count(_out_dims(c)), 3, 11, 16)
    var session = _session_words(device, OPS_CONV1D, c.ints, c, vals, seed)
    var seq = _direct_words(device, OPS_CONV1D, c.ints, c, vals, seed)
    _same_words("conv1d Session op vs Sequential's rankn_conv1d call", session, seq)
    var digest = UInt64(1469598103934665603)
    _mix(digest, session)
    print("SESSION_OPS_CONV conv1d_sequential digest", digest)
    return session^


def _refused[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], code: Int, x: List[Int], w: List[Int], bias: Int, ints: List[Int], words: String,
) raises:
    var hs = List[Int]()
    hs.append(_upload(s, _dyadic(_count(x), 7, 19, 8), x))
    hs.append(_upload(s, _dyadic(_count(w), 5, 13, 16), w))
    if bias > 0:
        var bd = List[Int](length=1, fill=bias)
        hs.append(_upload(s, _dyadic(bias, 3, 7, 4), bd))
    var raised = False
    try:
        _ = s.run_op(code, hs, ints, List[Float32]())
    except error:
        raised = True
        _require(words in String(error), "refusal: expected '" + words + "', got '" + String(error) + "'")
    _require(raised, "refusal: no error for '" + words + "'")


def _refusals_on[E: Exec & Movable & Deinitable](mut s: OpsSession[E]) raises:
    _refused(s, OPS_CONV1D, [1, 2, 4], [2, 2, 3], 0, [1, 0, 2, 1],
             "the dilated kernel (5) is larger than the padded input (4)")
    _refused(s, OPS_CONV3D, [1, 3, 2, 2, 2], [2, 1, 1, 1, 1], 0, [1, 1, 1, 0, 0, 0, 1, 1, 1, 2],
             "with groups (2) dividing C (3)")
    _refused(s, OPS_CONV_TRANSPOSE1D, [1, 2, 4], [2, 2, 3], 0, [2, 0, 2, 1, 1],
             "output_padding must be smaller than stride")
    _refused(s, OPS_CONV2D_ND, [1, 2, 4, 4], [3, 2, 2, 2], 2, [1, 1, 0, 0, 1, 1, 1],
             "the bias needs one value per output channel (3)")
    _refused(s, OPS_CONV3D, [1, 2, 4, 4], [3, 2, 2, 2], 0, [1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
             "conv3d takes x (N, C, D, H, W)")
    _refused(s, OPS_CONV1D, [1, 2, 4], [2, 2, 3], 0, [0, 0, 1, 1], "stride must be at least 1")


def run_refusals(device: Bool) raises -> List[Float32]:
    comptime if not CPU_COLUMN_BUILD:
        if device:
            var d = OpsSession[DeviceExec](DeviceExec(), True)
            try:
                _refusals_on(d)
            except error:
                d.close()
                raise error
            d.close()
            return List[Float32]()
    var h = OpsSession[HostExec](HostExec(), True)
    try:
        _refusals_on(h)
    except error:
        h.close()
        raise error
    h.close()
    print("SESSION_OPS_CONV refusals 6")
    return List[Float32]()


def _unit(device: Bool, unit: Int) raises -> List[Float32]:
    if unit < _CASES:
        var cases = _cases()
        return run_case(device, cases[unit])
    if unit == _CASES:
        return run_square(device)
    if unit == _CASES + 1:
        return run_sequential_conv1d(device)
    return run_refusals(device)


def test_session_ops_conv() raises:
    # Independent units: the host column spreads them across host cores, one HostExec each.
    var failed = List[Int](length=_UNITS, fill=0)
    var host_words = List[List[Float32]](length=_UNITS, fill=List[Float32]())
    var fp = failed.unsafe_ptr()
    var wp = host_words.unsafe_ptr()
    def task(unit: Int) {imm fp, imm wp}:
        try:
            wp[unit] = _unit(False, unit)
        except error:
            print("FAIL SESSION_OPS_CONV unit", unit, error)
            fp[unit] = 1
    host_parallelize(task, _UNITS)
    var bad = 0
    for u in range(_UNITS):  # small-loop(_UNITS: failure flags)
        bad += failed[u]
    if bad != 0:
        _fail(String(bad) + " units failed on the host column")
    comptime if not CPU_COLUMN_BUILD:
        for k in range(_UNITS):  # driver-loop: over the units on the one device
            var dw = _unit(True, k)
            _same_words("unit " + String(k) + " device vs host", host_words[k], dw)
    _ = failed^
    _ = host_words^
    print("PASS SESSION_OPS_CONV units", _UNITS)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
