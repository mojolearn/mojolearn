# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""The convolution and shape layers of Sequential (lane nn-conv, 2026-11-23).

Operations on the tape (autograd/nn_conv.mojo; layer codes 56..66 of autograd/model_blocks.mojo):
conv_transpose1d / 2d, adaptive_avg_pool2d, adaptive_max_pool2d / 1d, global_max_pool, constant /
zero, reflection and replication padding (1-D and 2-D), upsample (nearest and (bi)linear, size or
scale_factor, align_corners), pixel_shuffle / pixel_unshuffle. Each case:
  - the forward against a float64 oracle written here elementwise from PyTorch's definitions (no
    product helper; the upsampling scales are PyTorch's float32 scales, the rest float64);
  - every gradient (x; weight and bias of a transposed convolution) against central finite
    differences of the float64 oracle's loss sum(seed * y), every element probed (h = 1/64; the
    max-pool fixtures are distinct multiples of 1/16, so no window ties within 2h);
  - one `NN_CONV <case> digest` line over every compared word.
Models through the interpreter (autograd/model.mojo with the layer codes):
  - two models that use every layer code (a 2-D chain and a 1-D chain), three seeded epochs: finite
    loss, parameters and predictions, the loss falls, and a checkpoint saved and loaded predicts the
    same words;
  - a 1-D convolutional autoencoder (conv1d, conv1d, conv_transpose1d, conv_transpose1d) reconstructing
    two-tone signals: the reconstruction MSE falls below a fifth of the first epoch's and below 0.05;
  - a 2-D upsampling decoder (linear, reshape, nearest upsample, conv2d, bilinear upsample, conv2d)
    drawing a Gaussian blob at the latent's position: the MSE falls below a third of the first epoch's
    and below 0.01.
Columns: the CPU column runs everything on HostExec, the independent units across host cores. The
device column runs every unit on DeviceExec AND on HostExec and requires the same words (values,
gradients, fitted parameters, predictions).
"""
from std.math import sin, exp
from std.memory import bitcast
from std.testing import TestSuite
from autograd.tape import Tape, Tensor
from autograd.rankn import ShapeN, RN_MAX_RANK
from autograd.model import LOSS_MSE, TARGET_DENSE, OUT_RAW
from autograd.model_blocks import ModelRules
from autograd.nn_conv import (
    nn_conv_transpose1d, nn_conv_transpose2d, nn_adaptive_avg_pool2d, nn_adaptive_max_pool2d,
    nn_adaptive_max_pool1d, nn_global_max_pool, nn_pad, nn_upsample, nn_pixel_shuffle, up_scale,
    NNC_PAD_REFLECT, NNC_PAD_REPLICATE,
)
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from checks.native_tests.core_ops_common import normal
from checks.native_tests.training_sequential_common import (
    SeqModel, L_LINEAR, L_RELU, L_TANH, L_FLATTEN, L_RESHAPE, L_CONV2D,
)
from checks.native_tests.testsuite_cases import CPU_COLUMN_BUILD
from core.host_parallel import host_parallelize

comptime L_CONV1D = 15
comptime L_CT1 = 56
comptime L_CT2 = 57
comptime L_AAVG2 = 58
comptime L_AMAX2 = 59
comptime L_AMAX1 = 60
comptime L_GMAX = 61
comptime L_CPAD = 62
comptime L_RPAD = 63
comptime L_EPAD = 64
comptime L_UP = 65
comptime L_SHUF = 66

comptime _H = Float64(1.0) / Float64(64)

# case operations
comptime OP_CT2 = 0     # g: Cout, k, s, p, op, d, groups, bias
comptime OP_CT1 = 1     # g: the same over (N, C, L)
comptime OP_AAVG2 = 2   # g: OH, OW
comptime OP_AMAX2 = 3   # g: OH, OW
comptime OP_AMAX1 = 4   # g: OL
comptime OP_GMAX = 5
comptime OP_PAD = 6     # g: left, right, top, bottom, mode (0 constant, 1 reflect, 2 replicate); value
comptime OP_UP = 7      # g: OH, OW, bilinear, align (sizes 0: from value, the scale_factor); value
comptime OP_SHUF = 8    # g: r, unshuffle


def _fail(message: String) raises:
    print("FAIL NN_CONV", message)
    raise Error(message)


def _sh(dims: List[Int]) -> ShapeN:
    var d = Array[Int, RN_MAX_RANK](fill=1)
    for k in range(len(dims)):  # small-loop(len(dims): a tensor rank)
        d[k] = dims[k]
    return ShapeN(len(dims), d)


def _distinct(n: Int, salt: Int) -> List[Float32]:
    # distinct multiples of 1/16 (a permutation of 0..n-1 for odd step coprime with n)
    var v = List[Float32](length=n, fill=0)
    var step = 37 + 2 * salt
    while _gcd(step, n) != 1:
        step += 2
    for i in range(n):  # small-loop(n: check data)
        v[i] = Float32((i * step) % n - n // 2) / Float32(16)
    return v^


def _gcd(a: Int, b: Int) -> Int:
    var x = a
    var y = b
    while y != 0:
        var t = x % y
        x = y
        y = t
    return x


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
        _fail(name + ": host and device lengths differ")
    for i in range(len(a)):  # small-loop(len(a): check comparison)
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            _fail(name + ": host and device words differ at " + String(i) + " (" + String(a[i]) + " vs "
                  + String(b[i]) + ")")


def _finite(name: String, v: List[Float32]) raises:
    for i in range(len(v)):  # small-loop(len(v): check)
        if not (abs(Float64(v[i])) < 1e30):
            _fail(name + ": value " + String(i) + " is not finite (" + String(v[i]) + ")")


# ---------------------------------------------------------------- cases
struct _Case(Copyable, Movable):
    var name: String
    var op: Int
    var x: List[Int]     # (N, C, L) or (N, C, H, W)
    var g: List[Int]
    var value: Float64

    def __init__(out self, name: String, op: Int, x: List[Int], g: List[Int], value: Float64 = 0.0):
        self.name = name
        self.op = op
        self.x = x.copy()
        self.g = g.copy()
        self.value = value


def _cases() -> List[_Case]:
    var out = List[_Case]()
    out.append(_Case("conv_transpose2d s2 p1 op1 bias", OP_CT2, [2, 4, 3, 4], [6, 3, 2, 1, 1, 1, 1, 1]))
    out.append(_Case("conv_transpose2d groups2 dil2", OP_CT2, [2, 4, 3, 3], [6, 2, 1, 0, 0, 2, 2, 0]))
    out.append(_Case("conv_transpose1d s2 p1 op1 bias", OP_CT1, [2, 3, 5], [4, 3, 2, 1, 1, 1, 1, 1]))
    out.append(_Case("conv_transpose1d k4 s2 p1 groups3", OP_CT1, [2, 3, 6], [6, 4, 2, 1, 0, 1, 3, 1]))
    out.append(_Case("adaptive_avg_pool2d 5x7 to 3x4", OP_AAVG2, [2, 3, 5, 7], [3, 4]))
    out.append(_Case("adaptive_avg_pool2d 4x4 to 1x1", OP_AAVG2, [2, 3, 4, 4], [1, 1]))
    out.append(_Case("adaptive_max_pool2d 5x7 to 2x3", OP_AMAX2, [2, 3, 5, 7], [2, 3]))
    out.append(_Case("adaptive_max_pool1d 7 to 3", OP_AMAX1, [2, 3, 7], [3]))
    out.append(_Case("global_max_pool 2d", OP_GMAX, [2, 3, 4, 5], List[Int]()))
    out.append(_Case("global_max_pool 1d", OP_GMAX, [2, 3, 9], List[Int]()))
    out.append(_Case("constant_pad2d value 0.5", OP_PAD, [2, 2, 3, 4], [1, 2, 0, 1, 0], 0.5))
    out.append(_Case("zero_pad1d", OP_PAD, [2, 2, 5], [2, 1, 0, 0, 0], 0.0))
    out.append(_Case("constant_pad2d crop", OP_PAD, [2, 2, 4, 5], [-1, 1, 1, -1, 0], -0.25))
    out.append(_Case("reflection_pad2d", OP_PAD, [2, 2, 4, 5], [2, 1, 1, 3, 1]))
    out.append(_Case("reflection_pad1d", OP_PAD, [2, 2, 5], [3, 2, 0, 0, 1]))
    out.append(_Case("replication_pad2d", OP_PAD, [2, 2, 3, 4], [2, 3, 1, 2, 2]))
    out.append(_Case("replication_pad1d", OP_PAD, [2, 2, 5], [1, 4, 0, 0, 2]))
    out.append(_Case("upsample nearest x2", OP_UP, [2, 2, 3, 4], [0, 0, 0, 0], 2.0))
    out.append(_Case("upsample nearest size 5x7", OP_UP, [2, 2, 3, 4], [5, 7, 0, 0]))
    out.append(_Case("upsample nearest down 4x6 to 3x4", OP_UP, [2, 2, 4, 6], [3, 4, 0, 0]))
    out.append(_Case("upsample bilinear x2", OP_UP, [2, 2, 3, 4], [0, 0, 1, 0], 2.0))
    out.append(_Case("upsample bilinear x1.5", OP_UP, [2, 2, 3, 4], [0, 0, 1, 0], 1.5))
    out.append(_Case("upsample bilinear align size 5x7", OP_UP, [2, 2, 3, 4], [5, 7, 1, 1]))
    out.append(_Case("upsample bilinear size 4x9", OP_UP, [2, 2, 3, 4], [4, 9, 1, 0]))
    out.append(_Case("upsample linear 1d x2", OP_UP, [2, 2, 1, 5], [0, 0, 1, 0], 2.0))
    out.append(_Case("pixel_shuffle r2", OP_SHUF, [2, 8, 3, 2], [2, 0]))
    out.append(_Case("pixel_unshuffle r2", OP_SHUF, [2, 2, 4, 6], [2, 1]))
    return out^


def _count(dims: List[Int]) -> Int:
    var n = 1
    for d in dims:  # small-loop(rank)
        n *= d
    return n


def _four(c: _Case) -> List[Int]:
    """x as (N, C, H, W), H = 1 for a (N, C, L) case."""
    if len(c.x) == 4:
        return c.x.copy()
    var d: List[Int] = [c.x[0], c.x[1], 1, c.x[2]]
    return d^


def _in_counts(c: _Case) -> List[Int]:
    var out = List[Int]()
    out.append(_count(c.x))
    if c.op == OP_CT2 or c.op == OP_CT1:
        var taps = c.g[1] * c.g[1] if c.op == OP_CT2 else c.g[1]
        out.append(c.x[1] * (c.g[0] // c.g[6]) * taps)
        if c.g[7] == 1:
            out.append(c.g[0])
    return out^


def _inputs(c: _Case) -> List[List[Float32]]:
    var counts = _in_counts(c)
    var out = List[List[Float32]]()
    out.append(_distinct(counts[0], 0))
    for k in range(1, len(counts)):  # small-loop(at most 3 inputs)
        out.append(_dyadic(counts[k], 7 + 2 * k, 13, 8))
    return out^


def _up_sizes(c: _Case) -> Tuple[Int, Int, Float32, Float32]:
    """(OH, OW, scale_h, scale_w) through the product's up_scale (PyTorch's float32 scales)."""
    var f = _four(c)
    var sf = c.value
    var OH = c.g[0] if c.g[0] > 0 else Int(Float64(f[2]) * sf)
    var OW = c.g[1] if c.g[1] > 0 else Int(Float64(f[3]) * sf)
    if f[2] == 1:
        OH = 1
    var sfh = sf if (c.g[0] == 0 and f[2] > 1) else Float64(0)
    var sfw = sf if c.g[1] == 0 else Float64(0)
    var nearest = c.g[2] == 0
    var align = c.g[3] == 1
    return (OH, OW, up_scale(f[2], OH, sfh, align, nearest), up_scale(f[3], OW, sfw, align, nearest))


def _forward[E: Exec](mut ex: E, mut tape: Tape, c: _Case, vals: List[List[Float32]], grad: Bool,
                      mut ins: List[Tensor]) raises -> Tensor:
    ins = List[Tensor]()
    var x = tape.tensor_nd(ex, vals[0], _sh(c.x), grad)
    ins.append(x)
    if c.op == OP_CT2 or c.op == OP_CT1:
        var wd: List[Int] = [c.x[1], c.g[0] // c.g[6], c.g[1]]
        if c.op == OP_CT2:
            wd.append(c.g[1])
        var w = tape.tensor_nd(ex, vals[1], _sh(wd), grad)
        ins.append(w)
        var bias = Optional[Tensor]()
        if c.g[7] == 1:
            var bd: List[Int] = [c.g[0]]
            var b = tape.tensor_nd(ex, vals[2], _sh(bd), grad)
            ins.append(b)
            bias = Optional[Tensor](b)
        if c.op == OP_CT1:
            return nn_conv_transpose1d(tape, ex, x, w, bias, c.g[2], c.g[3], c.g[4], c.g[5], c.g[6])
        return nn_conv_transpose2d(tape, ex, x, w, bias, c.g[2], c.g[2], c.g[3], c.g[3], c.g[4], c.g[4],
                                   c.g[5], c.g[5], c.g[6])
    if c.op == OP_AAVG2:
        return nn_adaptive_avg_pool2d(tape, ex, x, c.g[0], c.g[1])
    if c.op == OP_AMAX2:
        return nn_adaptive_max_pool2d(tape, ex, x, c.g[0], c.g[1])
    if c.op == OP_AMAX1:
        return nn_adaptive_max_pool1d(tape, ex, x, c.g[0])
    if c.op == OP_GMAX:
        return nn_global_max_pool(tape, ex, x)
    if c.op == OP_PAD:
        var mode = 0 if c.g[4] == 0 else (NNC_PAD_REFLECT if c.g[4] == 1 else NNC_PAD_REPLICATE)
        return nn_pad(tape, ex, x, c.g[0], c.g[1], c.g[2], c.g[3], mode, Float32(c.value))
    if c.op == OP_UP:
        var u = _up_sizes(c)
        return nn_upsample(tape, ex, x, u[0], u[1], u[2], u[3], c.g[2] == 1, c.g[3] == 1)
    return nn_pixel_shuffle(tape, ex, x, c.g[0], c.g[1] == 1)


# ---------------------------------------------------------------- the float64 oracle
def _astart(a: Int, o: Int, i: Int) -> Int:
    return (a * i) // o


def _aend(a: Int, o: Int, i: Int) -> Int:
    return ((a + 1) * i + o - 1) // o


def _lin64(o: Int, isize: Int, scale: Float32, align: Bool) -> Tuple[Int, Int, Float64, Float64]:
    var s = Float64(scale)
    var src = s * Float64(o) if align else s * (Float64(o) + 0.5) - 0.5
    if src < 0:
        src = 0
    var i0 = Int(src)
    var i1 = i0 + 1 if i0 < isize - 1 else i0
    var l1 = src - Float64(i0)
    return (i0, i1, 1.0 - l1, l1)


def _oracle(c: _Case, v: List[List[Float64]]) -> List[Float64]:
    """The forward in float64, elementwise, from PyTorch's definitions."""
    var f = _four(c)
    var N = f[0]; var C = f[1]; var H = f[2]; var W = f[3]
    var x = v[0].copy()
    var out = List[Float64]()
    if c.op == OP_CT2 or c.op == OP_CT1:
        var Co = c.g[0]; var K = c.g[1]; var S = c.g[2]; var P = c.g[3]; var OP = c.g[4]; var D = c.g[5]
        var G = c.g[6]
        var KH = K if c.op == OP_CT2 else 1
        var SH = S if c.op == OP_CT2 else 1
        var PH = P if c.op == OP_CT2 else 0
        var OPH = OP if c.op == OP_CT2 else 0
        var DH = D if c.op == OP_CT2 else 1
        var OH = (H - 1) * SH - 2 * PH + DH * (KH - 1) + OPH + 1
        var OW = (W - 1) * S - 2 * P + D * (K - 1) + OP + 1
        var cig = C // G
        var cog = Co // G
        out = List[Float64](length=N * Co * OH * OW, fill=0)
        for n in range(N):  # small-loop(check oracle)
            for co in range(Co):
                for oh in range(OH):
                    for ow in range(OW):
                        out[((n * Co + co) * OH + oh) * OW + ow] = v[2][co] if c.g[7] == 1 else 0.0
        for n in range(N):  # small-loop(check oracle): the scatter definition
            for ci in range(C):
                var grp = ci // cig
                for ih in range(H):
                    for iw in range(W):
                        var xv = x[((n * C + ci) * H + ih) * W + iw]
                        for j in range(cog):
                            var co = grp * cog + j
                            for kh in range(KH):
                                var oh = ih * SH - PH + kh * DH
                                if oh < 0 or oh >= OH:
                                    continue
                                for kw in range(K):
                                    var ow = iw * S - P + kw * D
                                    if ow < 0 or ow >= OW:
                                        continue
                                    var wv = v[1][((ci * cog + j) * KH + kh) * K + kw]
                                    out[((n * Co + co) * OH + oh) * OW + ow] += xv * wv
        return out^
    if c.op == OP_AAVG2 or c.op == OP_AMAX2 or c.op == OP_AMAX1:
        var OH = c.g[0] if c.op != OP_AMAX1 else 1
        var OW = c.g[1] if c.op != OP_AMAX1 else c.g[0]
        for r in range(N * C):  # small-loop(check oracle)
            for oh in range(OH):
                for ow in range(OW):
                    var acc = Float64(0)
                    var first = True
                    var cnt = 0
                    for h in range(_astart(oh, OH, H), _aend(oh, OH, H)):
                        for w in range(_astart(ow, OW, W), _aend(ow, OW, W)):
                            var xv = x[(r * H + h) * W + w]
                            cnt += 1
                            if c.op == OP_AAVG2:
                                acc += xv
                            elif first or xv > acc:
                                acc = xv
                            first = False
                    out.append(acc / Float64(cnt) if c.op == OP_AAVG2 else acc)
        return out^
    if c.op == OP_GMAX:
        for r in range(N * C):  # small-loop(check oracle)
            var m = x[r * H * W]
            for k in range(H * W):
                if x[r * H * W + k] > m:
                    m = x[r * H * W + k]
            out.append(m)
        return out^
    if c.op == OP_PAD:
        var L = c.g[0]; var R = c.g[1]; var T = c.g[2]; var Bm = c.g[3]; var mode = c.g[4]
        var Hp = H + T + Bm
        var Wp = W + L + R
        for r in range(N * C):  # small-loop(check oracle)
            for hp in range(Hp):
                for wp in range(Wp):
                    var h = hp - T
                    var w = wp - L
                    if mode == 0:
                        if h >= 0 and h < H and w >= 0 and w < W:
                            out.append(x[(r * H + h) * W + w])
                        else:
                            out.append(c.value)
                        continue
                    if mode == 1:
                        h = -h if h < 0 else (2 * (H - 1) - h if h >= H else h)
                        w = -w if w < 0 else (2 * (W - 1) - w if w >= W else w)
                    else:
                        h = 0 if h < 0 else (H - 1 if h >= H else h)
                        w = 0 if w < 0 else (W - 1 if w >= W else w)
                    out.append(x[(r * H + h) * W + w])
        return out^
    if c.op == OP_UP:
        var u = _up_sizes(c)
        var OH = u[0]; var OW = u[1]
        var align = c.g[3] == 1
        for r in range(N * C):  # small-loop(check oracle)
            for oh in range(OH):
                for ow in range(OW):
                    if c.g[2] == 0:
                        var h = min(Int(Float64(oh) * Float64(u[2])), H - 1)
                        var w = min(Int(Float64(ow) * Float64(u[3])), W - 1)
                        out.append(x[(r * H + h) * W + w])
                        continue
                    var th = _lin64(oh, H, u[2], align)
                    var tw = _lin64(ow, W, u[3], align)
                    var b = r * H * W
                    out.append(th[2] * (tw[2] * x[b + th[0] * W + tw[0]] + tw[3] * x[b + th[0] * W + tw[1]])
                               + th[3] * (tw[2] * x[b + th[1] * W + tw[0]] + tw[3] * x[b + th[1] * W + tw[1]]))
        return out^
    # pixel shuffle / unshuffle
    var rr = c.g[0]
    if c.g[1] == 0:
        var Co = C // (rr * rr)
        for n in range(N):  # small-loop(check oracle)
            for co in range(Co):
                for oh in range(H * rr):
                    for ow in range(W * rr):
                        var ci = co * rr * rr + (oh % rr) * rr + (ow % rr)
                        out.append(x[((n * C + ci) * H + oh // rr) * W + ow // rr])
        return out^
    var OH = H // rr
    var OW = W // rr
    for n in range(N):  # small-loop(check oracle)
        for co in range(C * rr * rr):
            var ci = co // (rr * rr)
            var i = (co // rr) % rr
            var j = co % rr
            for oh in range(OH):
                for ow in range(OW):
                    out.append(x[((n * C + ci) * H + oh * rr + i) * W + ow * rr + j])
    return out^


def _wide(vals: List[List[Float32]]) -> List[List[Float64]]:
    var out = List[List[Float64]]()
    for a in range(len(vals)):  # small-loop(inputs)
        var w = List[Float64](length=len(vals[a]), fill=0)
        for i in range(len(vals[a])):  # small-loop(check data)
            w[i] = Float64(vals[a][i])
        out.append(w^)
    return out^


def _oracle_loss(c: _Case, v: List[List[Float64]], seed: List[Float32]) -> Float64:
    var y = _oracle(c, v)
    var s = Float64(0)
    for i in range(len(y)):  # small-loop(check loss)
        s += Float64(seed[i]) * y[i]
    return s


def run_case[E: Exec](mut ex: E, c: _Case) raises -> List[Float32]:
    """Forward and every gradient checked; returns the words compared (values, then gradients)."""
    var vals = _inputs(c)
    var wide = _wide(vals)
    var tape = Tape()
    var ins = List[Tensor]()
    var y = _forward(ex, tape, c, vals, True, ins)
    var yv = tape.values(ex, y)
    _close(c.name + " forward", yv, _oracle(c, wide), Float64(2e-6))
    var seed = _dyadic(len(yv), 3, 11, 16)
    var st = tape.tensor(ex, seed, tape.rows(y), tape.cols(y))
    tape.backward_with_rules(ex, y, st, ModelRules())
    var words = yv.copy()
    for a in range(len(ins)):  # small-loop(at most 3 inputs)
        var g = tape.gradients(ex, ins[a])
        var fd = List[Float64](length=len(g), fill=0)
        for i in range(len(g)):  # small-loop(len(g): every element probed)
            var up = wide.copy()
            up[a][i] = wide[a][i] + _H
            var down = wide.copy()
            down[a][i] = wide[a][i] - _H
            fd[i] = (_oracle_loss(c, up, seed) - _oracle_loss(c, down, seed)) / (2.0 * _H)
        _close(c.name + " gradient " + String(a), g, fd, Float64(2e-5))
        for i in range(len(g)):  # small-loop(len(g))
            words.append(g[i])
    var digest = UInt64(1469598103934665603)
    _mix(digest, words)
    print("NN_CONV", c.name, "digest", digest)
    return words^


# ---------------------------------------------------------------- models through the interpreter
comptime _MODEL_ROWS = 24


def _all2d(mut m: SeqModel):
    """Every 2-D layer code over (4, 8, 8) samples."""
    m.sample = [4, 8, 8]
    m.layer(L_CT2, 4, 3, 2, 1, 1, 2 * 1 + 1, 1.0)    # (4, 16, 16)
    m.layer(L_SHUF, 2, 1)                           # pixel_unshuffle: (16, 8, 8)
    m.layer(L_SHUF, 2, 0)                           # pixel_shuffle: (4, 16, 16)
    m.layer(L_RPAD, 1, 1, 1, 1, 2)                  # reflection: (4, 18, 18)
    m.layer(L_EPAD, 1, 0, 0, 1, 2)                  # replication: (4, 19, 19)
    m.layer(L_CPAD, 1, 0, 0, 1, 2, 0, 0.25)         # constant 0.25: (4, 20, 20)
    m.layer(L_AAVG2, 10, 7)                         # (4, 10, 7)
    m.layer(L_UP, 0, 0, 1, 0, 2, 0, 1.5)            # bilinear x1.5: (4, 15, 10)
    m.layer(L_UP, 6, 9, 0, 0, 2)                    # nearest to (4, 6, 9)
    m.layer(L_AMAX2, 3, 0)                          # (4, 3, 9)
    m.layer(L_RELU)
    m.layer(L_GMAX)                                 # (4,)
    m.layer(L_LINEAR, 1, 1)


def _all1d(mut m: SeqModel):
    """Every 1-D layer code over (3, 12) samples."""
    m.sample = [3, 12]
    m.layer(L_CT1, 4, 3, 1, 1, 0, 2 * 1 + 1, 1.0)    # (4, 12)
    m.layer(L_RPAD, 2, 2, 0, 0, 1)                  # (4, 16)
    m.layer(L_EPAD, 1, 0, 0, 0, 1)                  # (4, 17)
    m.layer(L_CPAD, 1, 2, 0, 0, 1, 0, 0.0)          # zero: (4, 20)
    m.layer(L_UP, 0, 30, 1, 1, 1)                   # linear, align, size 30
    m.layer(L_TANH)
    m.layer(L_AMAX1, 6)                             # (4, 6)
    m.layer(L_GMAX)                                 # (4,)
    m.layer(L_LINEAR, 1, 1)


def run_model[E: Exec](mut ex: E, which: Int) raises -> List[Float32]:
    """Fit the every-code model `which` (0: 2-D, 1: 1-D) for three seeded epochs; params, predictions."""
    var m = SeqModel()
    var name = String("all-codes-2d") if which == 0 else String("all-codes-1d")
    if which == 0:
        _all2d(m)
    else:
        _all1d(m)
    var cols = 1
    for d in m.sample:  # small-loop(rank)
        cols *= d
    var x = normal(_MODEL_ROWS * cols, 71 + which)
    var y = normal(_MODEL_ROWS, 73 + which)
    m.compile(LOSS_MSE, "adam", 0.01)
    _ = m.fit(ex, x, _MODEL_ROWS, cols, y, 1, TARGET_DENSE, 3, 8, 0, 5)
    var curve = m.losses()
    _finite(name + " loss", curve)
    _finite(name + " params", m.params)
    if not (curve[len(curve) - 1] < curve[0]):
        _fail(name + ": the loss did not fall (" + String(curve[0]) + " -> " + String(curve[len(curve) - 1]) + ")")
    var pred = m.predict(ex, x, _MODEL_ROWS, cols, OUT_RAW)
    _finite(name + " predictions", pred)
    # a checkpoint saved and loaded predicts the same words (the layer's tensors and names round trip)
    var file = String("nn_conv_ckpt_") + name + ".bin"
    _ = m.save(file, cols, False)
    var back = SeqModel()
    if which == 0:
        _all2d(back)
    else:
        _all1d(back)
    back.compile(LOSS_MSE, "adam", 0.01)
    back.load(file, cols)
    var again = back.predict(ex, x, _MODEL_ROWS, cols, OUT_RAW)
    _same_words(name + " checkpoint predictions", pred, again)
    print("NN_CONV model", name, "loss", curve[0], "->", curve[len(curve) - 1], "params", len(m.params))
    var words = m.params.copy()
    for i in range(len(pred)):  # small-loop(predictions)
        words.append(pred[i])
    var digest = UInt64(1469598103934665603)
    _mix(digest, words)
    print("NN_CONV model", name, "digest", digest)
    _ = x^
    _ = y^
    return words^


# ---------------------------------------------------------------- the two example fits
comptime _AE_LEN = 32
comptime _AE_ROWS = 128
comptime _AE_EPOCHS = 40
comptime _DEC_ROWS = 96
comptime _DEC_EPOCHS = 40
comptime _DEC_SIDE = 16


def _ae_data(rows: Int, salt: Int) -> List[Float32]:
    """Two tones per row (frequencies 1..3 and 4..6 cycles over the length, random phases), amplitude 0.5."""
    var ph = normal(2 * rows, salt)
    var x = List[Float32](length=rows * _AE_LEN, fill=0)
    for r in range(rows):  # small-loop(fixture data)
        var f1 = Float64(1 + (r * 5) % 3)
        var f2 = Float64(4 + (r * 7) % 3)
        for t in range(_AE_LEN):
            var a = 6.283185307179586 * Float64(t) / Float64(_AE_LEN)
            x[r * _AE_LEN + t] = Float32(0.5 * sin(f1 * a + Float64(ph[2 * r])) + 0.5 * sin(f2 * a + Float64(ph[2 * r + 1])))
    return x^


def run_autoencoder[E: Exec](mut ex: E) raises -> List[Float32]:
    """The 1-D convolutional autoencoder: (1, 32) -> conv1d(8, 5, stride 2, padding 2) tanh -> (8, 16)
    -> conv1d(8, 5, 2, 2) tanh -> (8, 8) -> conv_transpose1d(8, 4, 2, 1) tanh -> (8, 16) ->
    conv_transpose1d(1, 4, 2, 1) -> (1, 32), MSE against the input."""
    var m = SeqModel()
    m.sample = [1, _AE_LEN]
    m.layer(L_CONV1D, 8, 5, 2, 2, 1, 1)
    m.layer(L_TANH)
    m.layer(L_CONV1D, 8, 5, 2, 2, 1, 1)
    m.layer(L_TANH)
    m.layer(L_CT1, 8, 4, 2, 1, 0, 2 * 1 + 1, 1.0)
    m.layer(L_TANH)
    m.layer(L_CT1, 1, 4, 2, 1, 0, 2 * 1 + 1, 1.0)
    m.layer(L_FLATTEN)
    m.compile(LOSS_MSE, "adam", 0.01)
    var x = _ae_data(_AE_ROWS, 91)
    _ = m.fit(ex, x, _AE_ROWS, _AE_LEN, x, _AE_LEN, TARGET_DENSE, _AE_EPOCHS, 16, 1, 7)
    var curve = m.losses()
    _finite("autoencoder loss", curve)
    _finite("autoencoder params", m.params)
    var first = Float64(curve[0])
    var last = Float64(curve[len(curve) - 1])
    print("NN_CONV autoencoder reconstruction mse", first, "->", last)
    if not (last < 0.2 * first and last < 0.05):
        _fail("autoencoder: the reconstruction MSE " + String(last) + " is not below a fifth of the first epoch's ("
              + String(first) + ") and 0.05")
    var test = _ae_data(32, 93)
    var pred = m.predict(ex, test, 32, _AE_LEN, OUT_RAW)
    _finite("autoencoder predictions", pred)
    var words = m.params.copy()
    for i in range(len(pred)):  # small-loop(predictions)
        words.append(pred[i])
    var digest = UInt64(1469598103934665603)
    _mix(digest, words)
    print("NN_CONV autoencoder digest", digest)
    _ = x^
    _ = test^
    return words^


def _dec_data(rows: Int, salt: Int) -> Tuple[List[Float32], List[Float32]]:
    """Latents (4,) uniform-ish in [-1, 1]; images 16 x 16 of a Gaussian blob (width 2.5 pixels) centred
    at (8 + 4 z0, 8 + 4 z1), scaled by 0.75 + 0.25 z2."""
    var zr = normal(4 * rows, salt)
    var z = List[Float32](length=4 * rows, fill=0)
    var img = List[Float32](length=rows * _DEC_SIDE * _DEC_SIDE, fill=0)
    for r in range(rows):  # small-loop(fixture data)
        for k in range(4):
            var v = Float64(zr[4 * r + k]) / 2.0
            z[4 * r + k] = Float32(max(-1.0, min(1.0, v)))
        var cy = 7.5 + 4.0 * Float64(z[4 * r])
        var cx = 7.5 + 4.0 * Float64(z[4 * r + 1])
        var amp = 0.75 + 0.25 * Float64(z[4 * r + 2])
        for i in range(_DEC_SIDE):
            for j in range(_DEC_SIDE):
                var dy = Float64(i) - cy
                var dx = Float64(j) - cx
                var d2 = dy * dy + dx * dx
                img[(r * _DEC_SIDE + i) * _DEC_SIDE + j] = Float32(amp * exp(-d2 / (2.0 * 2.5 * 2.5)))
    return (z^, img^)


def run_decoder[E: Exec](mut ex: E) raises -> List[Float32]:
    """The 2-D upsampling decoder: (4,) -> linear(32) relu -> reshape (2, 4, 4) -> upsample nearest x2 ->
    conv2d(8, 3, padding 1) relu -> upsample bilinear x2 -> conv2d(1, 3, padding 1) -> (1, 16, 16),
    MSE against the blob image."""
    var m = SeqModel()
    m.sample = [4]
    m.layer(L_LINEAR, 32, 1)
    m.layer(L_RELU)
    m.layer(L_RESHAPE, 2, 4, 4)
    m.layer(L_UP, 0, 0, 0, 0, 2, 0, 2.0)
    m.layer(L_CONV2D, 8, 3, 1, 1, 1, 1)
    m.layer(L_RELU)
    m.layer(L_UP, 0, 0, 1, 0, 2, 0, 2.0)
    m.layer(L_CONV2D, 1, 3, 1, 1, 1, 1)
    m.layer(L_FLATTEN)
    m.compile(LOSS_MSE, "adam", 0.01)
    var data = _dec_data(_DEC_ROWS, 95)
    var px = _DEC_SIDE * _DEC_SIDE
    _ = m.fit(ex, data[0], _DEC_ROWS, 4, data[1], px, TARGET_DENSE, _DEC_EPOCHS, 16, 1, 9)
    var curve = m.losses()
    _finite("decoder loss", curve)
    _finite("decoder params", m.params)
    var first = Float64(curve[0])
    var last = Float64(curve[len(curve) - 1])
    print("NN_CONV decoder reconstruction mse", first, "->", last)
    if not (last < first / 3.0 and last < 0.01):
        _fail("decoder: the MSE " + String(last) + " is not below a third of the first epoch's (" + String(first)
              + ") and 0.01")
    var pred = m.predict(ex, data[0], _DEC_ROWS, 4, OUT_RAW)
    _finite("decoder predictions", pred)
    var words = m.params.copy()
    for i in range(len(pred)):  # small-loop(predictions)
        words.append(pred[i])
    var digest = UInt64(1469598103934665603)
    _mix(digest, words)
    print("NN_CONV decoder digest", digest)
    _ = data^
    return words^


# ---------------------------------------------------------------- tests
def _unit[E: Exec](mut ex: E, unit: Int) raises -> List[Float32]:
    var cases = _cases()
    var nc = len(cases)
    if unit < nc:
        return run_case(ex, cases[unit])
    if unit < nc + 2:
        return run_model(ex, unit - nc)
    if unit == nc + 2:
        return run_autoencoder(ex)
    return run_decoder(ex)


def test_nn_conv() raises:
    # Independent units (every case, the two every-code models, the two fits): the host column spreads
    # them across host cores, one HostExec each.
    var cases = _cases()
    var n = len(cases) + 4
    var failed = List[Int](length=n, fill=0)
    var host_words = List[List[Float32]](length=n, fill=List[Float32]())
    var fp = failed.unsafe_ptr()
    var wp = host_words.unsafe_ptr()
    def task(unit: Int) {imm fp, imm wp}:
        try:
            var host = HostExec()
            wp[unit] = _unit(host, unit)
        except error:
            print("FAIL NN_CONV unit", unit, error)
            fp[unit] = 1
    host_parallelize(task, n)
    var bad = 0
    for u in range(n):  # small-loop(n: failure flags)
        bad += failed[u]
    if bad != 0:
        _fail(String(bad) + " units failed on the host column")
    comptime if not CPU_COLUMN_BUILD:
        for k in range(n):  # driver-loop: over the units on the one device
            var dev = DeviceExec()
            var dw = _unit(dev, k)
            _same_words("unit " + String(k), host_words[k], dw)
    _ = failed^
    _ = host_words^
    _ = cases^
    print("PASS NN_CONV units", n)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
