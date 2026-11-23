# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""The 3-D layers of Sequential (lane nn-conv3d, 2026-11-23).

Ops on the tape (autograd/nn_conv3d.mojo rankn_conv3d, rankn_max_pool3d, rankn_avg_pool3d,
rankn_adaptive_avg_pool3d: the tape's one conv and pool nodes with their depth words), each case:
  - the forward against a float64 oracle written here elementwise from PyTorch's definitions (output
    sizes, ceil_mode, the average pool's divisor with and without the padding, divisor_override);
  - every gradient (x, and weight and bias for conv3d) against central finite differences of the
    op's own forward loss sum(seed * y), every element probed. The fixtures are dyadic (multiples of
    1/16 and 1/8, h = 1/64; the max pool's input distinct multiples of 1/16, so no window ties within
    2h): the conv and the max pool are exact in binary32 and the differences equal the gradients
    (tolerance 1e-9); the average pools divide (1e-5 forward, 1e-4 gradients);
  - one `NN_CONV3D <case> digest` line over every compared word.
Baselines through the model interpreter (autograd/nn_conv3d_layers.mojo codes 73 .. 76): two tiny 3-D
CNNs, one with max_pool3d and the global adaptive_avg_pool3d, one with a strided anisotropic conv3d,
avg_pool3d and adaptive_avg_pool3d(2), each fitted on synthetic (1, 8, 8, 8) volumes holding a noisy
bar along the depth axis (class 1) or along the width axis (class 0) at a random place: the loss
falls, every loss, parameter and prediction is finite, and the held-out accuracy reaches 0.9.
Columns: the CPU column runs everything on HostExec, the independent units across host cores. The
device column runs every unit on DeviceExec AND HostExec and requires the same words (values,
gradients, fitted parameters, predictions).
Sabotage arms (autograd/rankn_conv_ops.mojo): MOJOLEARN_NN_CONV3D_SABOTAGE_CONV and _POOL each fail a
case here.
"""
from std.math import isfinite
from std.memory import bitcast
from std.testing import TestSuite
from autograd.tape import Tape, Tensor
from autograd.rankn import ShapeN, RN_MAX_RANK
from autograd.rankn_api import RankNRules
from autograd.nn_conv3d import (
    Dhw, dhw_pack, rankn_conv3d, rankn_max_pool3d, rankn_avg_pool3d, rankn_adaptive_avg_pool3d,
)
from autograd.model import LOSS_CROSS_ENTROPY, TARGET_CODES, OUT_ARGMAX
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from checks.native_tests.core_ops_common import normal
from checks.native_tests.training_sequential_common import SeqModel, codes_as_words, L_RELU, L_FLATTEN, L_LINEAR
from checks.native_tests.testsuite_cases import CPU_COLUMN_BUILD
from core.host_parallel import host_parallelize

comptime L_CONV3D = 73
comptime L_MAXPOOL3D = 74
comptime L_AVGPOOL3D = 75
comptime L_ADAPTIVE_AVGPOOL3D = 76

comptime _H = Float64(1.0) / Float64(64)

# ops
comptime OP_CONV = 0
comptime OP_MAX = 1
comptime OP_AVG = 2
comptime OP_ADAPT = 3


def _fail(message: String) raises:
    print("FAIL NN_CONV3D", message)
    raise Error(message)


def _sh(dims: List[Int]) -> ShapeN:
    var d = Array[Int, RN_MAX_RANK](fill=1)
    for k in range(len(dims)):  # small-loop(len(dims): a tensor rank)
        d[k] = dims[k]
    return ShapeN(len(dims), d)


def _dyadic(n: Int, step: Int, modulus: Int, denom: Int) -> List[Float32]:
    var v = List[Float32](length=n, fill=0)
    for i in range(n):  # small-loop(n: check data)
        v[i] = Float32((i * step) % modulus - modulus // 2) / Float32(denom)
    return v^


def _distinct(n: Int) -> List[Float32]:
    # distinct multiples of 1/16 (a permutation of 0..n-1 when gcd(37, n) = 1)
    var v = List[Float32](length=n, fill=0)
    for i in range(n):  # small-loop(n: check data)
        v[i] = Float32((i * 37) % n - n // 2) / Float32(16)
    return v^


def _mix(mut digest: UInt64, values: List[Float32]):
    for i in range(len(values)):  # small-loop(len(values): check digest)
        digest = (digest ^ UInt64(bitcast[DType.uint32](values[i]))) * UInt64(1099511628211)


def _close(name: String, got: List[Float32], want: List[Float64], tol: Float64) raises:
    if len(got) != len(want):
        _fail(name + ": " + String(len(got)) + " values, " + String(len(want)) + " expected")
    for i in range(len(got)):  # small-loop(len(got): check comparison)
        var d = abs(Float64(got[i]) - want[i])
        if d > tol * (1.0 + abs(want[i])):
            _fail(name + ": index " + String(i) + " got " + String(got[i]) + " want " + String(want[i]))


def _same_words(name: String, a: List[Float32], b: List[Float32]) raises:
    if len(a) != len(b):
        _fail(name + ": host and device lengths differ")
    for i in range(len(a)):  # small-loop(len(a): check comparison)
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            _fail(name + ": host and device words differ at " + String(i) + " (" + String(a[i]) + " vs "
                  + String(b[i]) + ")")


# ---------------------------------------------------------------- op cases
# x (N, C, D, H, W). g by op:
#   conv  [OC, kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw, groups, bias]
#   max   [kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw, ceil]
#   avg   [kd, kh, kw, sd, sh, sw, pd, ph, pw, ceil, count_include_pad, divisor_override]
#   adapt [od, oh, ow]
struct _Case(Copyable, Movable):
    var name: String
    var op: Int
    var x: List[Int]
    var g: List[Int]

    def __init__(out self, name: String, op: Int, x: List[Int], g: List[Int]):
        self.name = name
        self.op = op
        self.x = x.copy()
        self.g = g.copy()


def _cases() -> List[_Case]:
    var out = List[_Case]()
    out.append(_Case("conv3d k3 p1 bias", OP_CONV, [2, 2, 5, 5, 4], [3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    out.append(_Case("conv3d k(2,3,1) s(2,1,2) p(1,0,0) d(1,2,1)", OP_CONV, [2, 2, 5, 6, 5],
                     [2, 2, 3, 1, 2, 1, 2, 1, 0, 0, 1, 2, 1, 1, 0]))
    out.append(_Case("conv3d groups2 bias", OP_CONV, [1, 4, 4, 4, 4], [4, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1]))
    out.append(_Case("conv3d depthwise k(3,1,1) d2", OP_CONV, [2, 3, 6, 3, 3], [3, 3, 1, 1, 1, 1, 1, 2, 0, 0, 2, 1, 1, 3, 1]))
    out.append(_Case("max_pool3d k2 s2", OP_MAX, [2, 2, 4, 4, 4], [2, 2, 2, 2, 2, 2, 0, 0, 0, 1, 1, 1, 0]))
    out.append(_Case("max_pool3d k3 s(1,2,2) p1 d(1,1,2)", OP_MAX, [1, 2, 5, 5, 7], [3, 3, 3, 1, 2, 2, 1, 1, 1, 1, 1, 2, 0]))
    out.append(_Case("max_pool3d k(2,3,3) s2 p(0,1,1) ceil", OP_MAX, [1, 2, 5, 5, 5], [2, 3, 3, 2, 2, 2, 0, 1, 1, 1, 1, 1, 1]))
    out.append(_Case("avg_pool3d k2 s2", OP_AVG, [2, 2, 4, 4, 4], [2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 1, 0]))
    out.append(_Case("avg_pool3d k3 s2 p1", OP_AVG, [1, 2, 5, 5, 5], [3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 1, 0]))
    out.append(_Case("avg_pool3d k3 s2 p1 exclude pad ceil", OP_AVG, [1, 2, 6, 5, 6], [3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 0, 0]))
    out.append(_Case("avg_pool3d k(2,2,3) s1 divisor 5", OP_AVG, [1, 2, 4, 3, 5], [2, 2, 3, 1, 1, 1, 0, 0, 0, 0, 1, 5]))
    out.append(_Case("adaptive_avg_pool3d 1", OP_ADAPT, [2, 3, 3, 4, 5], [1, 1, 1]))
    out.append(_Case("adaptive_avg_pool3d (2,2,1)", OP_ADAPT, [2, 2, 4, 6, 3], [2, 2, 1]))
    return out^


def _count(dims: List[Int]) -> Int:
    var n = 1
    for k in range(len(dims)):  # small-loop(rank)
        n *= dims[k]
    return n


def _in_counts(c: _Case) -> List[Int]:
    """Element counts of the inputs: x, then (conv3d) weight and bias."""
    var out = List[Int]()
    out.append(_count(c.x))
    if c.op == OP_CONV:
        out.append(c.g[0] * (c.x[1] // c.g[13]) * c.g[1] * c.g[2] * c.g[3])
        if c.g[14] == 1:
            out.append(c.g[0])
    return out^


def _inputs(c: _Case) -> List[List[Float32]]:
    var counts = _in_counts(c)
    var out = List[List[Float32]]()
    if c.op == OP_MAX:
        out.append(_distinct(counts[0]))
    else:
        out.append(_dyadic(counts[0], 5, 17, 16))
    for k in range(1, len(counts)):  # small-loop(at most 3 inputs)
        out.append(_dyadic(counts[k], 7 + 2 * k, 13, 8))
    return out^


def _t(g: List[Int], at: Int) -> Dhw:
    return Dhw(g[at], g[at + 1], g[at + 2])


def _forward[E: Exec](mut ex: E, mut tape: Tape, c: _Case, vals: List[List[Float32]], grad: Bool,
                      mut ins: List[Tensor]) raises -> Tensor:
    ins = List[Tensor]()
    var x = tape.tensor_nd(ex, vals[0], _sh(c.x), grad)
    ins.append(x)
    if c.op == OP_CONV:
        var oc = c.g[0]
        var w = tape.tensor_nd(ex, vals[1], _sh([oc, c.x[1] // c.g[13], c.g[1], c.g[2], c.g[3]]), grad)
        ins.append(w)
        var bias = Optional[Tensor]()
        if c.g[14] == 1:
            var b = tape.tensor_nd(ex, vals[2], _sh([oc]), grad)
            ins.append(b)
            bias = Optional[Tensor](b)
        return rankn_conv3d(tape, ex, x, w, bias, _t(c.g, 4), _t(c.g, 7), _t(c.g, 10), c.g[13])
    if c.op == OP_MAX:
        return rankn_max_pool3d(tape, ex, x, _t(c.g, 0), _t(c.g, 3), _t(c.g, 6), _t(c.g, 9), c.g[12] == 1)
    if c.op == OP_AVG:
        return rankn_avg_pool3d(tape, ex, x, _t(c.g, 0), _t(c.g, 3), _t(c.g, 6), c.g[9] == 1, c.g[10] == 1, c.g[11])
    return rankn_adaptive_avg_pool3d(tape, ex, x, _t(c.g, 0))


def _pool_out(size: Int, k: Int, p: Int, s: Int, dil: Int, ceil: Bool) -> Int:
    """PyTorch's pooling_output_shape (aten/src/ATen/native/Pool.h), written again here."""
    var num = size + 2 * p - dil * (k - 1) - 1 + ((s - 1) if ceil else 0)
    var o = num // s + 1
    if ceil and (o - 1) * s >= size + p:
        o -= 1
    return o


def _oracle(c: _Case, vals: List[List[Float32]]) -> List[Float64]:
    """The forward in float64, elementwise."""
    var N = c.x[0]
    var C = c.x[1]
    var D = c.x[2]
    var H = c.x[3]
    var W = c.x[4]
    var x = vals[0].copy()
    var out = List[Float64]()
    var g = c.g.copy()
    if c.op == OP_CONV:
        var OC = g[0]
        var G = g[13]
        var Cg = C // G
        var OCg = OC // G
        var OD = (D + 2 * g[7] - g[10] * (g[1] - 1) - 1) // g[4] + 1
        var OH = (H + 2 * g[8] - g[11] * (g[2] - 1) - 1) // g[5] + 1
        var OW = (W + 2 * g[9] - g[12] * (g[3] - 1) - 1) // g[6] + 1
        for n in range(N):  # small-loop(check oracle)
            for oc in range(OC):
                for od in range(OD):
                    for oh in range(OH):
                        for ow in range(OW):
                            var acc = Float64(vals[2][oc]) if g[14] == 1 else Float64(0)
                            for cl in range(Cg):
                                var ch = (oc // OCg) * Cg + cl
                                for kd in range(g[1]):
                                    var z = od * g[4] - g[7] + kd * g[10]
                                    for kh in range(g[2]):
                                        var h = oh * g[5] - g[8] + kh * g[11]
                                        for kw in range(g[3]):
                                            var w = ow * g[6] - g[9] + kw * g[12]
                                            if z < 0 or z >= D or h < 0 or h >= H or w < 0 or w >= W:
                                                continue
                                            var wi = (((oc * Cg + cl) * g[1] + kd) * g[2] + kh) * g[3] + kw
                                            acc += Float64(x[(((n * C + ch) * D + z) * H + h) * W + w]) * Float64(vals[1][wi])
                            out.append(acc)
        return out^
    if c.op == OP_ADAPT:
        # PyTorch's windows [floor(o I / O), ceil((o + 1) I / O))
        var OD = g[0]
        var OH = g[1]
        var OW = g[2]
        for r in range(N * C):  # small-loop(check oracle)
            for od in range(OD):
                for oh in range(OH):
                    for ow in range(OW):
                        var d0 = (od * D) // OD
                        var d1 = ((od + 1) * D + OD - 1) // OD
                        var h0 = (oh * H) // OH
                        var h1 = ((oh + 1) * H + OH - 1) // OH
                        var w0 = (ow * W) // OW
                        var w1 = ((ow + 1) * W + OW - 1) // OW
                        var acc = Float64(0)
                        for z in range(d0, d1):
                            for h in range(h0, h1):
                                for w in range(w0, w1):
                                    acc += Float64(x[((r * D + z) * H + h) * W + w])
                        out.append(acc / Float64((d1 - d0) * (h1 - h0) * (w1 - w0)))
        return out^
    var is_max = c.op == OP_MAX
    var ceil = g[12] == 1 if is_max else g[9] == 1
    var dd = g[9] if is_max else 1
    var dh = g[10] if is_max else 1
    var dw = g[11] if is_max else 1
    var OD = _pool_out(D, g[0], g[6], g[3], dd, ceil)
    var OH = _pool_out(H, g[1], g[7], g[4], dh, ceil)
    var OW = _pool_out(W, g[2], g[8], g[5], dw, ceil)
    for r in range(N * C):  # small-loop(check oracle)
        for od in range(OD):
            for oh in range(OH):
                for ow in range(OW):
                    var best = Float64(0)
                    var first = True
                    var acc = Float64(0)
                    for kd in range(g[0]):
                        var z = od * g[3] - g[6] + kd * dd
                        for kh in range(g[1]):
                            var h = oh * g[4] - g[7] + kh * dh
                            for kw in range(g[2]):
                                var w = ow * g[5] - g[8] + kw * dw
                                if z < 0 or z >= D or h < 0 or h >= H or w < 0 or w >= W:
                                    continue
                                var v = Float64(x[((r * D + z) * H + h) * W + w])
                                if first or v > best:
                                    best = v
                                first = False
                                acc += v
                    if is_max:
                        out.append(best)
                    else:
                        # avg_pool3d's divisor (aten AvgPool3d): the window clipped to the padded input,
                        # then to the input unless count_include_pad
                        var ds = od * g[3] - g[6]
                        var hs = oh * g[4] - g[7]
                        var ws = ow * g[5] - g[8]
                        var de = min(ds + g[0], D + g[6])
                        var he = min(hs + g[1], H + g[7])
                        var we = min(ws + g[2], W + g[8])
                        var size = (de - ds) * (he - hs) * (we - ws)
                        if g[10] == 0:
                            size = (min(de, D) - max(ds, 0)) * (min(he, H) - max(hs, 0)) * (min(we, W) - max(ws, 0))
                        if g[11] > 0:
                            size = g[11]
                        out.append(acc / Float64(size))
    return out^


def _seed(count: Int) -> List[Float32]:
    return _dyadic(count, 3, 11, 16)


def _loss[E: Exec](mut ex: E, c: _Case, vals: List[List[Float32]], seed: List[Float32]) raises -> Float64:
    var tape = Tape()
    var ins = List[Tensor]()
    var y = _forward(ex, tape, c, vals, False, ins)
    var yv = tape.values(ex, y)
    var s = Float64(0)
    for i in range(len(yv)):  # small-loop(check loss)
        s += Float64(seed[i]) * Float64(yv[i])
    return s


def run_op_case[E: Exec](mut ex: E, c: _Case) raises -> List[Float32]:
    """Forward and every gradient checked; returns the words compared (values, then gradients)."""
    var vals = _inputs(c)
    var tape = Tape()
    var ins = List[Tensor]()
    var y = _forward(ex, tape, c, vals, True, ins)
    var yv = tape.values(ex, y)
    var divides = c.op == OP_AVG or c.op == OP_ADAPT
    _close(c.name + " forward", yv, _oracle(c, vals), Float64(1e-5) if divides else Float64(1e-9))
    var seed = _seed(len(yv))
    var st = tape.tensor(ex, seed, tape.rows(y), tape.cols(y))
    tape.backward_with_rules(ex, y, st, RankNRules())
    var words = yv.copy()
    for a in range(len(ins)):  # small-loop(at most 3 inputs)
        var gr = tape.gradients(ex, ins[a])
        var fd = List[Float64](length=len(gr), fill=0)
        for i in range(len(gr)):  # small-loop(len(gr): every element probed)
            var up = vals.copy()
            up[a][i] = Float32(Float64(vals[a][i]) + _H)
            var down = vals.copy()
            down[a][i] = Float32(Float64(vals[a][i]) - _H)
            fd[i] = (_loss(ex, c, up, seed) - _loss(ex, c, down, seed)) / (2.0 * _H)
        _close(c.name + " gradient " + String(a), gr, fd, Float64(1e-4) if divides else Float64(1e-9))
        for i in range(len(gr)):  # small-loop(len(gr))
            words.append(gr[i])
    var digest = UInt64(1469598103934665603)
    _mix(digest, words)
    print("NN_CONV3D", c.name, "digest", digest)
    return words^


# ---------------------------------------------------------------- baselines
comptime _S = 8
comptime _VOX = _S * _S * _S
comptime _TRAIN = 96
comptime _TEST = 64
comptime _EPOCHS = 8


struct _Vol(Movable):
    var x: List[Float32]
    var y: List[Float32]
    var codes: List[Int]

    def __init__(out self, rows: Int, salt: Int):
        """One channel of 8 x 8 x 8: 0.3 standard-normal noise everywhere, plus a bar of height 1.5 and
        length 5 along the depth axis (class 1) or the width axis (class 0) at a random place."""
        var noise = normal(rows * _VOX, salt)
        var place = normal(rows * 4, salt + 1)
        self.x = List[Float32](length=rows * _VOX, fill=0)
        self.codes = List[Int](length=rows, fill=0)
        for r in range(rows):  # small-loop(fixture data)
            var cls = (r * 7 + salt) % 2
            self.codes[r] = cls
            for v in range(_VOX):
                self.x[r * _VOX + v] = Float32(0.3) * noise[r * _VOX + v]
            var a = Int(abs(place[4 * r]) * 1000.0) % _S
            var b = Int(abs(place[4 * r + 1]) * 1000.0) % _S
            var s0 = Int(abs(place[4 * r + 2]) * 1000.0) % (_S - 4)
            for t in range(5):
                var z: Int
                var h: Int
                var w: Int
                if cls == 1:
                    z = s0 + t
                    h = a
                    w = b
                else:
                    z = a
                    h = b
                    w = s0 + t
                self.x[r * _VOX + (z * _S + h) * _S + w] += Float32(1.5)
        self.y = codes_as_words(self.codes)


def _net_max(mut m: SeqModel) raises:
    m.layer(L_CONV3D, 4, dhw_pack(3, 3, 3), dhw_pack(1, 1, 1), dhw_pack(1, 1, 1), 1 + 2, dhw_pack(1, 1, 1))
    m.layer(L_RELU)
    m.layer(L_MAXPOOL3D, dhw_pack(2, 2, 2), dhw_pack(2, 2, 2), 0, dhw_pack(1, 1, 1), 0)
    m.layer(L_CONV3D, 8, dhw_pack(3, 3, 3), dhw_pack(1, 1, 1), dhw_pack(1, 1, 1), 1 + 2, dhw_pack(1, 1, 1))
    m.layer(L_RELU)
    m.layer(L_ADAPTIVE_AVGPOOL3D, dhw_pack(1, 1, 1))
    m.layer(L_FLATTEN)
    m.layer(L_LINEAR, 2, 1)


def _net_avg(mut m: SeqModel) raises:
    # an anisotropic strided conv (k (3, 3, 1), padding (1, 1, 0)), avg_pool3d, a pooled grid of 2
    m.layer(L_CONV3D, 6, dhw_pack(3, 3, 1), dhw_pack(1, 1, 1), dhw_pack(1, 1, 0), 1 + 2, dhw_pack(1, 1, 1))
    m.layer(L_RELU)
    m.layer(L_AVGPOOL3D, dhw_pack(2, 2, 2), dhw_pack(2, 2, 2), 0, 1, 0, 0)
    m.layer(L_CONV3D, 8, dhw_pack(3, 1, 3), dhw_pack(1, 1, 1), dhw_pack(1, 0, 1), 0 + 2, dhw_pack(1, 1, 1))
    m.layer(L_RELU)
    m.layer(L_ADAPTIVE_AVGPOOL3D, dhw_pack(2, 2, 2))
    m.layer(L_FLATTEN)
    m.layer(L_LINEAR, 2, 1)


def _all_finite(name: String, values: List[Float32]) raises:
    for i in range(len(values)):  # small-loop(check)
        if not isfinite(values[i]):
            _fail(name + ": a non-finite value at " + String(i))


def run_baseline[E: Exec](mut ex: E, which: Int) raises -> List[Float32]:
    """Fit baseline `which` (0 max-pool net, 1 avg-pool net); returns params then predictions."""
    var train = _Vol(_TRAIN, 21)
    var test = _Vol(_TEST, 22)
    var m = SeqModel()
    m.sample = [1, _S, _S, _S]
    var name: String
    if which == 0:
        _net_max(m)
        name = "cnn3d-max"
    else:
        _net_avg(m)
        name = "cnn3d-avg"
    m.compile(LOSS_CROSS_ENTROPY, "adam", 0.01)
    _ = m.fit(ex, train.x, _TRAIN, _VOX, train.y, 1, TARGET_CODES, _EPOCHS, 32, -1, 3)
    var curve = m.losses()
    _all_finite(name + " losses", curve)
    _all_finite(name + " parameters", m.params)
    if not (curve[len(curve) - 1] < curve[0]):
        _fail(name + ": the loss did not fall (" + String(curve[0]) + " -> " + String(curve[len(curve) - 1]) + ")")
    var pred = m.predict(ex, test.x, _TEST, _VOX, OUT_ARGMAX)
    _all_finite(name + " predictions", pred)
    var right = 0
    for i in range(_TEST):  # small-loop(check accuracy)
        if Int(pred[i]) == test.codes[i]:
            right += 1
    var acc = Float64(right) / Float64(_TEST)
    print("NN_CONV3D baseline", name, "loss", curve[0], "->", curve[len(curve) - 1], "held-out accuracy", acc)
    if acc < 0.9:
        _fail(name + ": held-out accuracy " + String(acc) + " is below 0.9")
    var words = m.params.copy()
    for i in range(len(pred)):  # small-loop(predictions)
        words.append(pred[i])
    var digest = UInt64(1469598103934665603)
    _mix(digest, words)
    print("NN_CONV3D baseline", name, "digest", digest)
    _ = train^
    _ = test^
    return words^


# ---------------------------------------------------------------- tests
def test_nn_conv3d_ops() raises:
    # Independent cases: the host column spreads them across host cores (one HostExec each).
    var cases = _cases()
    var n = len(cases)
    var failed = List[Int](length=n, fill=0)
    var host_words = List[List[Float32]](length=n, fill=List[Float32]())
    var fp = failed.unsafe_ptr()
    var wp = host_words.unsafe_ptr()
    var cp = cases.unsafe_ptr()
    def task(unit: Int) {imm fp, imm wp, imm cp}:
        try:
            var host = HostExec()
            wp[unit] = run_op_case(host, cp[unit])
        except error:
            print("FAIL NN_CONV3D op", unit, error)
            fp[unit] = 1
    host_parallelize(task, n)
    for u in range(n):  # small-loop(n: failure flags)
        if failed[u] != 0:
            _fail("op case " + String(u) + " failed on the host column")
    comptime if not CPU_COLUMN_BUILD:
        for k in range(n):  # driver-loop: over the op cases on the one device
            var dev = DeviceExec()
            var dw = run_op_case(dev, cases[k])
            _same_words(cases[k].name, host_words[k], dw)
    _ = failed^
    _ = host_words^
    _ = cases^
    print("PASS NN_CONV3D ops", n)


def test_nn_conv3d_baselines() raises:
    comptime B = 2
    var failed = List[Int](length=B, fill=0)
    var host_words = List[List[Float32]](length=B, fill=List[Float32]())
    var fp = failed.unsafe_ptr()
    var wp = host_words.unsafe_ptr()
    def task(unit: Int) {imm fp, imm wp}:
        try:
            var host = HostExec()
            wp[unit] = run_baseline(host, unit)
        except error:
            print("FAIL NN_CONV3D baseline", unit, error)
            fp[unit] = 1
    host_parallelize(task, B)
    for u in range(B):  # small-loop(B: failure flags)
        if failed[u] != 0:
            _fail("baseline " + String(u) + " failed on the host column")
    comptime if not CPU_COLUMN_BUILD:
        for which in range(B):  # driver-loop: over the baselines on the one device
            var dev = DeviceExec()
            var dw = run_baseline(dev, which)
            _same_words("baseline " + String(which), host_words[which], dw)
    _ = failed^
    _ = host_words^
    print("PASS NN_CONV3D baselines", B)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
