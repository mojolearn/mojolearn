# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CNN LANE'S ELEMENT FUNCTIONS: ONE SOURCE FOR THE DEVICE AND THE HOST.

Every function here computes ONE output element `i` from raw pointers and an
Int32 parameter block `p`. `x_cnn/device.mojo` launches it one thread per
element (`elem_kernel`); `x_cnn/host/ops_host.mojo` calls it in a plain loop.
The arithmetic is therefore the same source on every column; what differs is
only the GEMM, and there the device calls `identical_gemm` and the host calls
`gemm_oracle`, the two halves of `mojolearn.identical.gemm.fp32.v1`.

References (semantics only, PyTorch 2.4):
  torch/nn/modules/conv.py            Conv1d / Conv2d (groups=1, zeros padding)
  aten/src/ATen/native/Unfold2d / im2col.h   the im2col column order (c, kh, kw)
  aten/src/ATen/native/cuda/DilatedMaxPool2d.cu   max pool: first max wins, NaN wins
  aten/src/ATen/native/cuda/AveragePool2d.cu      avg pool divisor rules

IDENTITY (pass 1): every reduction here is a sequential fold in a fixed
order, one thread per output, no atomics. The col2im scatter-add is written
as a GATHER per input pixel over (kh, kw) ascending (the one atomic hazard
of the backward pass, DEVIATION 5700). The weight-gradient reduction is the
pinned GEMM's (TN over the N*OH*OW rows, DEVIATION 5701).
"""
from core.philox import philox4x32_10
from checks.numerics import ftz, identical_div, identical_mul, identical_exp, identical_log, identical_rsqrt

comptime FP = MutPointer[Float32, MutAnyOrigin]
comptime IP = MutPointer[Int32, MutAnyOrigin]
comptime ElemFn = def(Int, FP, FP, FP, FP, IP, IP) thin -> None


# ---------------------------------------------------------------- conv params
# The Int32 block of a convolution (conv_params fills 13 and 14):
comptime CP_N = 0
comptime CP_C = 1
comptime CP_H = 2
comptime CP_W = 3
comptime CP_OC = 4
comptime CP_KH = 5
comptime CP_KW = 6
comptime CP_SH = 7
comptime CP_SW = 8
comptime CP_PH = 9
comptime CP_PW = 10
comptime CP_DH = 11
comptime CP_DW = 12
comptime CP_OH = 13
comptime CP_OW = 14
comptime CP_BIAS = 15
#: 1 reverses col2im's (kh) order: the host binding's sabotage build and a
#: device sabotage patch set it; production callers leave it 0.
comptime CP_REV = 16
comptime CP_LEN = 17


@always_inline
def _g(p: IP, k: Int) -> Int:
    return Int(p.unsafe_load(k))


def conv_params(raw: List[Int]) raises -> List[Int32]:
    """Validate the Python-side block and fill OH, OW."""
    if len(raw) < CP_BIAS + 1:
        raise Error("x_cnn: conv params need 16 entries")
    for k in range(CP_OH):
        if raw[k] < 0:
            raise Error("x_cnn: negative conv parameter")
    var positive: List[Int] = [CP_N, CP_C, CP_H, CP_W, CP_OC, CP_KH, CP_KW, CP_SH, CP_SW, CP_DH, CP_DW]
    for k in positive:
        if raw[k] <= 0:
            raise Error("x_cnn: conv dimensions, stride and dilation must be positive")
    var oh = (raw[CP_H] + 2 * raw[CP_PH] - raw[CP_DH] * (raw[CP_KH] - 1) - 1) // raw[CP_SH] + 1
    var ow = (raw[CP_W] + 2 * raw[CP_PW] - raw[CP_DW] * (raw[CP_KW] - 1) - 1) // raw[CP_SW] + 1
    if oh <= 0 or ow <= 0:
        raise Error("x_cnn: the kernel does not fit the padded input")
    var out = List[Int32](capacity=CP_LEN)
    for k in range(CP_LEN):
        out.append(Int32(raw[k]) if k < len(raw) else Int32(0))
    out[CP_OH] = Int32(oh)
    out[CP_OW] = Int32(ow)
    return out^


@always_inline
def im2col_at(i: Int, x: FP, cols: FP, f2: FP, f3: FP, q: IP, p: IP):
    """cols[r, qq], r = (n*OH + oh)*OW + ow, qq = (c*KH + kh)*KW + kw: the
    input pixel under that tap, or +0.0 in the zero padding. A copy, no
    arithmetic."""
    var C = _g(p, CP_C); var H = _g(p, CP_H); var W = _g(p, CP_W)
    var KH = _g(p, CP_KH); var KW = _g(p, CP_KW)
    var OH = _g(p, CP_OH); var OW = _g(p, CP_OW)
    var ckk = C * KH * KW
    var r = i // ckk
    var qq = i - r * ckk
    var n = r // (OH * OW)
    var rem = r - n * OH * OW
    var oh = rem // OW
    var ow = rem - oh * OW
    var c = qq // (KH * KW)
    var t = qq - c * KH * KW
    var kh = t // KW
    var kw = t - kh * KW
    var h = oh * _g(p, CP_SH) - _g(p, CP_PH) + kh * _g(p, CP_DH)
    var w = ow * _g(p, CP_SW) - _g(p, CP_PW) + kw * _g(p, CP_DW)
    var v = Float32(0)
    if h >= 0 and h < H and w >= 0 and w < W:
        v = ftz(x.unsafe_load(((n * C + c) * H + h) * W + w))
    cols.unsafe_store(i, v)


@always_inline
def conv_out_at(i: Int, y2: FP, bias: FP, dst: FP, f3: FP, q: IP, p: IP):
    """out[n, oc, oh, ow] (NCHW) = y2[r, oc] (+ bias[oc]); one add."""
    var OC = _g(p, CP_OC); var OH = _g(p, CP_OH); var OW = _g(p, CP_OW)
    var ow = i % OW
    var t = i // OW
    var oh = t % OH
    t = t // OH
    var oc = t % OC
    var n = t // OC
    var r = (n * OH + oh) * OW + ow
    var v = ftz(y2.unsafe_load(r * OC + oc))
    if _g(p, CP_BIAS) != 0:
        v = ftz(v + ftz(bias.unsafe_load(oc)))
    dst.unsafe_store(i, v)


@always_inline
def dout_rows_at(i: Int, dout: FP, g: FP, f2: FP, f3: FP, q: IP, p: IP):
    """g[r, oc] = dout[n, oc, oh, ow]: NCHW to the GEMM's row layout. A copy."""
    var OC = _g(p, CP_OC); var OH = _g(p, CP_OH); var OW = _g(p, CP_OW)
    var r = i // OC
    var oc = i - r * OC
    var n = r // (OH * OW)
    var rem = r - n * OH * OW
    g.unsafe_store(i, ftz(dout.unsafe_load(((n * OC + oc) * OH * OW) + rem)))


@always_inline
def col2im_at(i: Int, dcols: FP, dx: FP, f2: FP, f3: FP, q: IP, p: IP):
    """dx[n, c, h, w] = the sum of every dcols entry whose tap reads this
    pixel, GATHERED in (kh, kw) ascending order from +0.0 (DEVIATION 5700:
    the reference scatter-adds, whose order is the scheduler's)."""
    var C = _g(p, CP_C); var H = _g(p, CP_H); var W = _g(p, CP_W)
    var KH = _g(p, CP_KH); var KW = _g(p, CP_KW)
    var OH = _g(p, CP_OH); var OW = _g(p, CP_OW)
    var SH = _g(p, CP_SH); var SW = _g(p, CP_SW)
    var PH = _g(p, CP_PH); var PW = _g(p, CP_PW)
    var DH = _g(p, CP_DH); var DW = _g(p, CP_DW)
    var w = i % W
    var t = i // W
    var h = t % H
    t = t // H
    var c = t % C
    var n = t // C
    var ckk = C * KH * KW
    var acc = Float32(0)
    for a in range(KH):
        var kh = KH - 1 - a if _g(p, CP_REV) != 0 else a
        var th = h + PH - kh * DH
        if th < 0 or th % SH != 0:
            continue
        var oh = th // SH
        if oh >= OH:
            continue
        for kw in range(KW):
            var tw = w + PW - kw * DW
            if tw < 0 or tw % SW != 0:
                continue
            var ow = tw // SW
            if ow >= OW:
                continue
            var r = (n * OH + oh) * OW + ow
            acc = ftz(acc + ftz(dcols.unsafe_load(r * ckk + (c * KH + kh) * KW + kw)))
    dx.unsafe_store(i, acc)


@always_inline
def fill_one_at(i: Int, dst: FP, f1: FP, f2: FP, f3: FP, q: IP, p: IP):
    dst.unsafe_store(i, Float32(1))


# ---------------------------------------------------------------- pooling
# The Int32 block of a pooling layer (pool_params fills 12 and 13):
comptime PP_N = 0
comptime PP_C = 1
comptime PP_H = 2
comptime PP_W = 3
comptime PP_KH = 4
comptime PP_KW = 5
comptime PP_SH = 6
comptime PP_SW = 7
comptime PP_PH = 8
comptime PP_PW = 9
comptime PP_DH = 10
comptime PP_DW = 11
comptime PP_OH = 12
comptime PP_OW = 13
#: AvgPool's count_include_pad (PyTorch's default True).
comptime PP_INCPAD = 14
#: 1 reverses the backward gathers' (kh) order (sabotage only).
comptime PP_REV = 15
comptime PP_LEN = 16


def pool_params(raw: List[Int]) raises -> List[Int32]:
    """Validate and fill OH, OW (floor mode; ceil_mode is refused at the API)."""
    if len(raw) < PP_INCPAD + 1:
        raise Error("x_cnn: pool params need 15 entries")
    for k in range(PP_OH):
        if raw[k] < 0:
            raise Error("x_cnn: negative pool parameter")
    var positive: List[Int] = [PP_N, PP_C, PP_H, PP_W, PP_KH, PP_KW, PP_SH, PP_SW, PP_DH, PP_DW]
    for k in positive:
        if raw[k] <= 0:
            raise Error("x_cnn: pool dimensions, stride and dilation must be positive")
    if 2 * raw[PP_PH] > raw[PP_KH] or 2 * raw[PP_PW] > raw[PP_KW]:
        raise Error("x_cnn: pad should be at most half of the kernel size (PyTorch's rule)")
    var oh = (raw[PP_H] + 2 * raw[PP_PH] - raw[PP_DH] * (raw[PP_KH] - 1) - 1) // raw[PP_SH] + 1
    var ow = (raw[PP_W] + 2 * raw[PP_PW] - raw[PP_DW] * (raw[PP_KW] - 1) - 1) // raw[PP_SW] + 1
    if oh <= 0 or ow <= 0:
        raise Error("x_cnn: the pooling window does not fit the padded input")
    var out = List[Int32](capacity=PP_LEN)
    for k in range(PP_LEN):
        out.append(Int32(raw[k]) if k < len(raw) else Int32(0))
    out[PP_OH] = Int32(oh)
    out[PP_OW] = Int32(ow)
    return out^


@always_inline
def maxpool_fwd_at(i: Int, x: FP, dst: FP, f2: FP, f3: FP, idx: IP, p: IP):
    """out[n, c, oh, ow] = the max over the window, taps in (kh, kw)
    ascending order; the FIRST maximum wins a tie (strict >, so -0.0 and
    +0.0 keep whichever came first), a NaN wins (PyTorch's `val > max ||
    isnan(val)`). idx is the flat h*W + w of the winner."""
    var H = _g(p, PP_H); var W = _g(p, PP_W)
    var KH = _g(p, PP_KH); var KW = _g(p, PP_KW)
    var OH = _g(p, PP_OH); var OW = _g(p, PP_OW)
    var ow = i % OW
    var t = i // OW
    var oh = t % OH
    var nc = t // OH
    var base = nc * H * W
    var best = Float32(0)
    var bi = -1
    for kh in range(KH):
        var h = oh * _g(p, PP_SH) - _g(p, PP_PH) + kh * _g(p, PP_DH)
        if h < 0 or h >= H:
            continue
        for kw in range(KW):
            var w = ow * _g(p, PP_SW) - _g(p, PP_PW) + kw * _g(p, PP_DW)
            if w < 0 or w >= W:
                continue
            var v = ftz(x.unsafe_load(base + h * W + w))
            if bi < 0 or v > best or v != v:
                best = v
                bi = h * W + w
    dst.unsafe_store(i, best)
    idx.unsafe_store(i, Int32(bi))


@always_inline
def maxpool_bwd_at(i: Int, dout: FP, dx: FP, f2: FP, f3: FP, idx: IP, p: IP):
    """dx[n, c, h, w] = the sum of dout over every window whose winner is
    this pixel, gathered in (kh, kw) ascending order from +0.0."""
    var H = _g(p, PP_H); var W = _g(p, PP_W)
    var KH = _g(p, PP_KH); var KW = _g(p, PP_KW)
    var OH = _g(p, PP_OH); var OW = _g(p, PP_OW)
    var SH = _g(p, PP_SH); var SW = _g(p, PP_SW)
    var PH = _g(p, PP_PH); var PW = _g(p, PP_PW)
    var DH = _g(p, PP_DH); var DW = _g(p, PP_DW)
    var w = i % W
    var t = i // W
    var h = t % H
    var nc = t // H
    var me = Int32(h * W + w)
    var acc = Float32(0)
    for a in range(KH):
        var kh = KH - 1 - a if _g(p, PP_REV) != 0 else a
        var th = h + PH - kh * DH
        if th < 0 or th % SH != 0:
            continue
        var oh = th // SH
        if oh >= OH:
            continue
        for kw in range(KW):
            var tw = w + PW - kw * DW
            if tw < 0 or tw % SW != 0:
                continue
            var ow = tw // SW
            if ow >= OW:
                continue
            var o = (nc * OH + oh) * OW + ow
            if idx.unsafe_load(o) == me:
                acc = ftz(acc + ftz(dout.unsafe_load(o)))
    dx.unsafe_store(i, acc)


@always_inline
def _avg_divisor(oh: Int, ow: Int, p: IP) -> Int:
    """PyTorch AvgPool2d's divisor (floor mode): the window clipped to the
    padded input, or to the input itself when count_include_pad is 0."""
    var H = _g(p, PP_H); var W = _g(p, PP_W)
    var PH = _g(p, PP_PH); var PW = _g(p, PP_PW)
    var hs = oh * _g(p, PP_SH) - PH
    var ws = ow * _g(p, PP_SW) - PW
    var he = min(hs + _g(p, PP_KH), H + PH)
    var we = min(ws + _g(p, PP_KW), W + PW)
    if _g(p, PP_INCPAD) != 0:
        return (he - hs) * (we - ws)
    return (min(he, H) - max(hs, 0)) * (min(we, W) - max(ws, 0))


@always_inline
def avgpool_fwd_at(i: Int, x: FP, dst: FP, f2: FP, f3: FP, q: IP, p: IP):
    """out = (the window's sum in (kh, kw) ascending order) / divisor."""
    var H = _g(p, PP_H); var W = _g(p, PP_W)
    var KH = _g(p, PP_KH); var KW = _g(p, PP_KW)
    var OH = _g(p, PP_OH); var OW = _g(p, PP_OW)
    var ow = i % OW
    var t = i // OW
    var oh = t % OH
    var nc = t // OH
    var base = nc * H * W
    var acc = Float32(0)
    for kh in range(KH):
        var h = oh * _g(p, PP_SH) - _g(p, PP_PH) + kh
        if h < 0 or h >= H:
            continue
        for kw in range(KW):
            var w = ow * _g(p, PP_SW) - _g(p, PP_PW) + kw
            if w < 0 or w >= W:
                continue
            acc = ftz(acc + ftz(x.unsafe_load(base + h * W + w)))
    dst.unsafe_store(i, ftz(identical_div(acc, Float32(_avg_divisor(oh, ow, p)))))


@always_inline
def avgpool_bwd_at(i: Int, dout: FP, dx: FP, f2: FP, f3: FP, q: IP, p: IP):
    """dx[n, c, h, w] = the sum over every window holding this pixel of
    dout / divisor, gathered in (kh, kw) ascending order from +0.0."""
    var H = _g(p, PP_H); var W = _g(p, PP_W)
    var KH = _g(p, PP_KH); var KW = _g(p, PP_KW)
    var OH = _g(p, PP_OH); var OW = _g(p, PP_OW)
    var SH = _g(p, PP_SH); var SW = _g(p, PP_SW)
    var PH = _g(p, PP_PH); var PW = _g(p, PP_PW)
    var w = i % W
    var t = i // W
    var h = t % H
    var nc = t // H
    var acc = Float32(0)
    for a in range(KH):
        var kh = KH - 1 - a if _g(p, PP_REV) != 0 else a
        var th = h + PH - kh
        if th < 0 or th % SH != 0:
            continue
        var oh = th // SH
        if oh >= OH:
            continue
        for kw in range(KW):
            var tw = w + PW - kw
            if tw < 0 or tw % SW != 0:
                continue
            var ow = tw // SW
            if ow >= OW:
                continue
            var o = (nc * OH + oh) * OW + ow
            var g = ftz(identical_div(ftz(dout.unsafe_load(o)), Float32(_avg_divisor(oh, ow, p))))
            acc = ftz(acc + g)
    dx.unsafe_store(i, acc)


# ---------------------------------------------------------------- trainer ops
# Parameter blocks: bias rows / linear [rows, in, out]; softmax [n, k];
# sgd [count] with the floats [lr, momentum, weight_decay] in f3.


@always_inline
def relu_fwd_at(i: Int, x: FP, f1: FP, dst: FP, f3: FP, q: IP, p: IP):
    var v = ftz(x.unsafe_load(i))
    dst.unsafe_store(i, v if v > Float32(0) else Float32(0))


@always_inline
def relu_bwd_at(i: Int, x: FP, g: FP, dx: FP, f3: FP, q: IP, p: IP):
    """dx = g where x > 0, else +0.0 (PyTorch's threshold_backward)."""
    var v = ftz(x.unsafe_load(i))
    dx.unsafe_store(i, ftz(g.unsafe_load(i)) if v > Float32(0) else Float32(0))


@always_inline
def add_at(i: Int, a: FP, b: FP, dst: FP, f3: FP, q: IP, p: IP):
    dst.unsafe_store(i, ftz(ftz(a.unsafe_load(i)) + ftz(b.unsafe_load(i))))


@always_inline
def bias_rows_at(i: Int, y: FP, bias: FP, dst: FP, f3: FP, q: IP, p: IP):
    """dst[r, j] = y[r, j] + bias[j]; p[2] is the row width."""
    var cols = _g(p, 2)
    dst.unsafe_store(i, ftz(ftz(y.unsafe_load(i)) + ftz(bias.unsafe_load(i % cols))))


@always_inline
def softmax_xent_row_at(i: Int, logits: FP, grad: FP, proba: FP, rowloss: FP, labels: IP, p: IP):
    """Row i of softmax + mean cross entropy (PyTorch F.cross_entropy,
    reduction='mean'): proba = exp(l - max) / sum, in column order; grad =
    (proba - onehot) / n; rowloss = log(sum) - (l[y] - max). A label < 0
    writes proba only. The mean of rowloss is folded by the caller in row
    order (`seq_mean`)."""
    var n = _g(p, 0)
    var k = _g(p, 1)
    var base = i * k
    var mx = ftz(logits.unsafe_load(base))
    for j in range(1, k):
        var v = ftz(logits.unsafe_load(base + j))
        if v > mx:
            mx = v
    var s = Float32(0)
    for j in range(k):
        s = ftz(s + ftz(identical_exp(ftz(ftz(logits.unsafe_load(base + j)) - mx))))
    var y = Int(labels.unsafe_load(i))
    for j in range(k):
        var e = ftz(identical_exp(ftz(ftz(logits.unsafe_load(base + j)) - mx)))
        var pr = ftz(identical_div(e, s))
        proba.unsafe_store(base + j, pr)
        if y >= 0:
            var t = ftz(pr - Float32(1)) if j == y else pr
            grad.unsafe_store(base + j, ftz(identical_div(t, Float32(n))))
    if y >= 0:
        var ly = ftz(ftz(logits.unsafe_load(base + y)) - mx)
        rowloss.unsafe_store(i, ftz(ftz(identical_log(s)) - ly))


def seq_mean(values: List[Float32], n: Int) -> Float32:
    """The loss mean: a sequential fold in row order, then one division."""
    var acc = Float32(0)
    for i in range(n):
        acc = ftz(acc + ftz(values[i]))
    return ftz(identical_div(acc, Float32(n)))


@always_inline
def sgd_at(i: Int, w: FP, g: FP, v: FP, hyper: FP, q: IP, p: IP):
    """PyTorch SGD (dampening 0, nesterov False) with the momentum buffer
    starting at +0.0: d = g + wd*w; v = momentum*v + d; w = w - lr*v."""
    var lr = hyper.unsafe_load(0)
    var mom = hyper.unsafe_load(1)
    var wd = hyper.unsafe_load(2)
    var wi = ftz(w.unsafe_load(i))
    var d = ftz(g.unsafe_load(i))
    if wd != Float32(0):
        d = ftz(d + ftz(identical_mul(wd, wi)))
    var vi = d
    if mom != Float32(0):
        vi = ftz(ftz(identical_mul(mom, ftz(v.unsafe_load(i)))) + d)
        v.unsafe_store(i, vi)
    w.unsafe_store(i, ftz(wi - ftz(identical_mul(lr, vi))))



# ---------------------------------------------------------------- batch norm
# p = [N, C, HW]. One aux buffer per call: [eps, momentum] then seven
# per-channel arrays at 2 + k*C: mean, var (biased), invstd, sum_g, sum_gx,
# gamma, beta. Every per-channel reduction is ONE sequential fold over
# (n, hw) in that order, one thread per channel: its partial count is 1 for
# every shape and every core count (IDENTITY_PATHS row 7's rule; DEVIATION
# 5702). PyTorch's reference (aten/src/ATen/native/cuda/Normalization.cuh)
# uses Welford partials across a block, a schedule of its own.
comptime BN_MEAN = 0
comptime BN_VAR = 1
comptime BN_INVSTD = 2
comptime BN_SUMG = 3
comptime BN_SUMGX = 4
comptime BN_GAMMA = 5
comptime BN_BETA = 6
comptime BN_SLOTS = 7


@always_inline
def _bn(aux: FP, slot: Int, c: Int, C: Int) -> Int:
    return 2 + slot * C + c


@always_inline
def bn_stats_at(c: Int, x: FP, aux: FP, f2: FP, f3: FP, q: IP, p: IP):
    """Training statistics of channel c: mean, then the biased variance as a
    second fold of (x - mean)^2, invstd = rsqrt(var + eps)."""
    var N = _g(p, 0); var C = _g(p, 1); var HW = _g(p, 2)
    var count = Float32(N * HW)
    var acc = Float32(0)
    for n in range(N):
        var base = (n * C + c) * HW
        for k in range(HW):
            acc = ftz(acc + ftz(x.unsafe_load(base + k)))
    var mean = ftz(identical_div(acc, count))
    var sq = Float32(0)
    for n in range(N):
        var base = (n * C + c) * HW
        for k in range(HW):
            var d = ftz(ftz(x.unsafe_load(base + k)) - mean)
            sq = ftz(sq + ftz(identical_mul(d, d)))
    var var_b = ftz(identical_div(sq, count))
    aux.unsafe_store(_bn(aux, BN_MEAN, c, C), mean)
    aux.unsafe_store(_bn(aux, BN_VAR, c, C), var_b)
    aux.unsafe_store(_bn(aux, BN_INVSTD, c, C), ftz(identical_rsqrt(ftz(var_b + aux.unsafe_load(0)))))


@always_inline
def bn_eval_stats_at(c: Int, running: FP, aux: FP, f2: FP, f3: FP, q: IP, p: IP):
    """Eval statistics of channel c from running = [mean C | var C]."""
    var C = _g(p, 1)
    var rv = ftz(running.unsafe_load(C + c))
    aux.unsafe_store(_bn(aux, BN_MEAN, c, C), ftz(running.unsafe_load(c)))
    aux.unsafe_store(_bn(aux, BN_VAR, c, C), rv)
    aux.unsafe_store(_bn(aux, BN_INVSTD, c, C), ftz(identical_rsqrt(ftz(rv + aux.unsafe_load(0)))))


@always_inline
def bn_apply_at(i: Int, x: FP, aux: FP, dst: FP, f3: FP, q: IP, p: IP):
    """y = ((x - mean) * invstd) * gamma + beta, each product pinned."""
    var C = _g(p, 1); var HW = _g(p, 2)
    var c = (i // HW) % C
    var xhat = ftz(identical_mul(ftz(ftz(x.unsafe_load(i)) - aux.unsafe_load(_bn(aux, BN_MEAN, c, C))),
                                 aux.unsafe_load(_bn(aux, BN_INVSTD, c, C))))
    dst.unsafe_store(i, ftz(ftz(identical_mul(xhat, aux.unsafe_load(_bn(aux, BN_GAMMA, c, C))))
                            + aux.unsafe_load(_bn(aux, BN_BETA, c, C))))


@always_inline
def bn_running_at(c: Int, running: FP, aux: FP, f2: FP, f3: FP, q: IP, p: IP):
    """PyTorch's running update: r = (1 - m) r + m stat, the variance
    unbiased (var * count / (count - 1))."""
    var N = _g(p, 0); var C = _g(p, 1); var HW = _g(p, 2)
    var m = aux.unsafe_load(1)
    var keep = ftz(Float32(1) - m)
    var count = N * HW
    var unb = ftz(identical_div(ftz(identical_mul(aux.unsafe_load(_bn(aux, BN_VAR, c, C)), Float32(count))),
                                Float32(count - 1)))
    var rm = ftz(running.unsafe_load(c))
    var rv = ftz(running.unsafe_load(C + c))
    running.unsafe_store(c, ftz(ftz(identical_mul(keep, rm)) + ftz(identical_mul(m, aux.unsafe_load(_bn(aux, BN_MEAN, c, C))))))
    running.unsafe_store(C + c, ftz(ftz(identical_mul(keep, rv)) + ftz(identical_mul(m, unb))))


@always_inline
def bn_bwd_red_at(c: Int, x: FP, g: FP, aux: FP, f3: FP, q: IP, p: IP):
    """sum_g and sum_gx = sum(g * xhat) of channel c, one fold each in (n, hw) order."""
    var N = _g(p, 0); var C = _g(p, 1); var HW = _g(p, 2)
    var mean = aux.unsafe_load(_bn(aux, BN_MEAN, c, C))
    var invstd = aux.unsafe_load(_bn(aux, BN_INVSTD, c, C))
    var sg = Float32(0)
    var sgx = Float32(0)
    for n in range(N):
        var base = (n * C + c) * HW
        for k in range(HW):
            var gv = ftz(g.unsafe_load(base + k))
            var xhat = ftz(identical_mul(ftz(ftz(x.unsafe_load(base + k)) - mean), invstd))
            sg = ftz(sg + gv)
            sgx = ftz(sgx + ftz(identical_mul(gv, xhat)))
    aux.unsafe_store(_bn(aux, BN_SUMG, c, C), sg)
    aux.unsafe_store(_bn(aux, BN_SUMGX, c, C), sgx)


@always_inline
def bn_bwd_dx_at(i: Int, x: FP, g: FP, aux: FP, dst: FP, q: IP, p: IP):
    """Training-mode dx = (g - mean(g) - xhat * mean(g xhat)) * invstd * gamma
    (PyTorch batch_norm_backward's formula)."""
    var N = _g(p, 0); var C = _g(p, 1); var HW = _g(p, 2)
    var c = (i // HW) % C
    var count = Float32(N * HW)
    var invstd = aux.unsafe_load(_bn(aux, BN_INVSTD, c, C))
    var xhat = ftz(identical_mul(ftz(ftz(x.unsafe_load(i)) - aux.unsafe_load(_bn(aux, BN_MEAN, c, C))), invstd))
    var mg = ftz(identical_div(aux.unsafe_load(_bn(aux, BN_SUMG, c, C)), count))
    var mgx = ftz(identical_div(aux.unsafe_load(_bn(aux, BN_SUMGX, c, C)), count))
    var t = ftz(ftz(ftz(g.unsafe_load(i)) - mg) - ftz(identical_mul(xhat, mgx)))
    dst.unsafe_store(i, ftz(identical_mul(ftz(identical_mul(t, invstd)), aux.unsafe_load(_bn(aux, BN_GAMMA, c, C)))))


@always_inline
def bn_bwd_eval_dx_at(i: Int, x: FP, g: FP, aux: FP, dst: FP, q: IP, p: IP):
    """Eval-mode dx = g * invstd * gamma (the statistics are constants)."""
    var C = _g(p, 1); var HW = _g(p, 2)
    var c = (i // HW) % C
    dst.unsafe_store(i, ftz(identical_mul(ftz(identical_mul(ftz(g.unsafe_load(i)), aux.unsafe_load(_bn(aux, BN_INVSTD, c, C)))),
                                          aux.unsafe_load(_bn(aux, BN_GAMMA, c, C)))))



# ---------------------------------------------------------------- dropout2d
# p = [N, C, HW, seed_lo, seed_hi, thresh_hi, thresh_lo]; hyper f[0] = drop p.


@always_inline
def dropout2d_at(i: Int, x: FP, mask: FP, dst: FP, hyper: FP, q: IP, p: IP):
    """PyTorch nn.Dropout2d (training): every (n, c) channel is zeroed with
    probability p, the rest scaled by 1 / (1 - p). The draw is Philox4x32-10
    at counter (n*C + c, 0, 0, 0) under the 64-bit seed, word 0 compared as
    an INTEGER against thresh = round(p * 2^32): no float in the decision
    (DEVIATION 5703; torch's own stream is its generator's, not ours)."""
    var C = _g(p, 1); var HW = _g(p, 2)
    var nc = i // HW
    var key = SIMD[DType.uint32, 2](UInt32(p.unsafe_load(3)), UInt32(p.unsafe_load(4)))
    var ctr = SIMD[DType.uint32, 4](UInt32(nc), 0, 0, 0)
    var u = philox4x32_10(ctr, key)[0]
    var thresh = (UInt64(UInt32(p.unsafe_load(5))) << 16) | UInt64(UInt32(p.unsafe_load(6)))
    var m = Float32(0)
    if UInt64(u) >= thresh:
        m = ftz(identical_div(Float32(1), ftz(Float32(1) - hyper.unsafe_load(0))))
    mask.unsafe_store(i, m)
    dst.unsafe_store(i, ftz(identical_mul(ftz(x.unsafe_load(i)), m)))


@always_inline
def mul_at(i: Int, a: FP, b: FP, dst: FP, f3: FP, q: IP, p: IP):
    dst.unsafe_store(i, ftz(identical_mul(ftz(a.unsafe_load(i)), ftz(b.unsafe_load(i)))))
