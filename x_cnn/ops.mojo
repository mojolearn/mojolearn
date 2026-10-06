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
from std.math import fma  # only a sabotage arm (seam 5709) spells the fused form
from std.memory import bitcast
from std.sys.compile import is_defined
from core.philox import philox4x32_10
from checks.rtf_seam import rtf_mul_add
from gemm.experiments.neural_profile import NEURAL_LEAF, NEURAL_CHAINS, neural_partition, neural_cell, neural_merge_chains, neural_fold_push, neural_fold_drain
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL, ftz, identical_div, identical_mul, identical_exp, identical_log, identical_rsqrt, identical_sqrt

# NN45/46/47: source-only A/B arms; default OFF, no quality/identity/time
# claim. Shared element functions define every floating seam for host/GPU.
comptime NN45_CONV_RELU = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_NN45_CONV_RELU"]() and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
comptime NN46_GATHER_BOUNDS = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_NN46_GATHER_BOUNDS"]() and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
comptime NN47_APPLY_RUNNING = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_NN47_APPLY_RUNNING"]() and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()

comptime FP = MutPointer[Float32, MutAnyOrigin]
comptime IP = MutPointer[Int32, MutAnyOrigin]
comptime ElemFn = def(Int, FP, FP, FP, FP, IP, IP) thin -> None


# lane fam-neural (2026-10-04): three backward gathers visited every window
# (or every padded position) of the plane per pixel and kept the few that
# read it. Each now visits only the candidates, in the same ascending order,
# with the same membership test inside: the same terms folded in the same
# order from +0.0, so no bit moves on any column (this file is every
# column's element functions). IDENTICAL only; each has its own before arm.
#: AvgPool2d backward, windows that tile the input (kernel == stride, no
#: padding; global average pooling is this): the pixel's ONE window instead
#: of a KH x KW scan. `-D MOJOLEARN_IDN_AVGPOOL_BWD_TILE_OFF` is the before arm.
comptime IDN_AVGPOOL_BWD_TILE = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (is_defined["MOJOLEARN_IDN_AVGPOOL_BWD_TILE_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
#: pad backward: the pad margins and the pixel's own interior position
#: instead of the whole Hp x Wp plane. `-D MOJOLEARN_IDN_PAD_BWD_BOUNDED_OFF`.
comptime IDN_PAD_BWD_BOUNDED = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (is_defined["MOJOLEARN_IDN_PAD_BWD_BOUNDED_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
#: adaptive pool backward (avg and max): the output cells whose window can
#: hold the pixel instead of all OH x OW. `-D MOJOLEARN_IDN_ADAPT_BWD_BOUNDED_OFF`.
comptime IDN_ADAPT_BWD_BOUNDED = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (is_defined["MOJOLEARN_IDN_ADAPT_BWD_BOUNDED_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
#: nr-small D9 (2026-10-04): `softmax_xent_row_at` computes each exp once
#: and parks it in the proba row (same words, every column and the host).
#: -D MOJOLEARN_IDN_XCNN_SOFTMAX_ONE_EXP_OFF (or MOJOLEARN_IDN_ALL_OFF).
comptime XCNN_SOFTMAX_ONE_EXP = not (is_defined["MOJOLEARN_IDN_XCNN_SOFTMAX_ONE_EXP_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())


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
def canon(v: Float32) -> Float32:
    """Clause B: a NaN leaves this lane as ONE word, 0x7FC00000, never as a
    vendor's payload (NVIDIA writes 0x7FFFFFFF, x86 0xFFC00000). Applied to
    every value a trainer-path element function stores (DEVIATION 5705)."""
    if v != v:
        return bitcast[DType.float32](UInt32(0x7FC00000))
    return v


@always_inline
def _g(p: IP, k: Int) -> Int:
    return Int(p.unsafe_load(k))


# DEVIATION 5720 (lane/cnn-apple, 2026-09-28): the index decoding of the hot
# element functions divides in 32 bits. Mojo's Int is 64 bits and no GPU
# divides 64-bit integers in hardware (Apple emulates it in a long software
# routine); every operand here is a non-negative element index or dimension
# below 2^31 (the device launch passes the element count as Int32), so the
# unsigned 32-bit quotient and remainder equal the Int ones. Index arithmetic
# only: the same words are read and written, no float value changes.
@always_inline
def _ud(a: Int, b: Int) -> Int:
    return Int(UInt32(a) // UInt32(b))


@always_inline
def _um(a: Int, b: Int) -> Int:
    return Int(UInt32(a) % UInt32(b))


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
    var r = _ud(i, ckk)
    var qq = i - r * ckk
    var n = _ud(r, (OH * OW))
    var rem = r - n * OH * OW
    var oh = _ud(rem, OW)
    var ow = rem - oh * OW
    var c = _ud(qq, (KH * KW))
    var t = qq - c * KH * KW
    var kh = _ud(t, KW)
    var kw = t - kh * KW
    var h = oh * _g(p, CP_SH) - _g(p, CP_PH) + kh * _g(p, CP_DH)
    var w = ow * _g(p, CP_SW) - _g(p, CP_PW) + kw * _g(p, CP_DW)
    var v = Float32(0)
    if h >= 0 and h < H and w >= 0 and w < W:
        v = ftz(x.unsafe_load(((n * C + c) * H + h) * W + w))
    cols.unsafe_store(i, v)


@always_inline
def im2col_taps_at(i: Int, x: FP, cols: FP, f2: FP, f3: FP, q: IP, p: IP):
    """lane/cnn-apple2: `im2col_at` for the KH*KW taps of one (row r,
    channel c), i = r*C + c: the same words at the same addresses (each
    `im2col_at` element's copy), one index decode per KH*KW words instead
    of one per word."""
    var C = _g(p, CP_C); var H = _g(p, CP_H); var W = _g(p, CP_W)
    var KH = _g(p, CP_KH); var KW = _g(p, CP_KW)
    var OH = _g(p, CP_OH); var OW = _g(p, CP_OW)
    var r = _ud(i, C)
    var c = i - r * C
    var n = _ud(r, (OH * OW))
    var rem = r - n * OH * OW
    var oh = _ud(rem, OW)
    var ow = rem - oh * OW
    var h0 = oh * _g(p, CP_SH) - _g(p, CP_PH)
    var w0 = ow * _g(p, CP_SW) - _g(p, CP_PW)
    var DH = _g(p, CP_DH); var DW = _g(p, CP_DW)
    var xb = (n * C + c) * H
    var ob = r * (C * KH * KW) + c * KH * KW
    for kh in range(KH):
        var h = h0 + kh * DH
        for kw in range(KW):
            var w = w0 + kw * DW
            var v = Float32(0)
            if h >= 0 and h < H and w >= 0 and w < W:
                v = ftz(x.unsafe_load((xb + h) * W + w))
            cols.unsafe_store(ob + kh * KW + kw, v)


@always_inline
def conv_out_at(i: Int, y2: FP, bias: FP, dst: FP, f3: FP, q: IP, p: IP):
    """out[n, oc, oh, ow] (NCHW) = y2[r, oc] (+ bias[oc]); one add."""
    var OC = _g(p, CP_OC); var OH = _g(p, CP_OH); var OW = _g(p, CP_OW)
    var ow = _um(i, OW)
    var t = _ud(i, OW)
    var oh = _um(t, OH)
    t = _ud(t, OH)
    var oc = _um(t, OC)
    var n = _ud(t, OC)
    var r = (n * OH + oh) * OW + ow
    dst.unsafe_store(i, conv_out_val(y2.unsafe_load(r * OC + oc), bias, oc, p))


@always_inline
def conv_out_val(y: Float32, bias: FP, oc: Int, p: IP) -> Float32:
    """`conv_out_at`'s stored word for the GEMM value `y` of channel `oc`
    (lane/cnn-apple2: shared with the tiled layout kernel)."""
    var v = ftz(y)
    if _g(p, CP_BIAS) != 0:
        v = ftz(v + ftz(bias.unsafe_load(oc)))
    return canon(v)


@always_inline
def dout_rows_at(i: Int, dout: FP, g: FP, f2: FP, f3: FP, q: IP, p: IP):
    """g[r, oc] = dout[n, oc, oh, ow]: NCHW to the GEMM's row layout. A copy."""
    var OC = _g(p, CP_OC); var OH = _g(p, CP_OH); var OW = _g(p, CP_OW)
    var r = _ud(i, OC)
    var oc = i - r * OC
    var n = _ud(r, (OH * OW))
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
    var w = _um(i, W)
    var t = _ud(i, W)
    var h = _um(t, H)
    t = _ud(t, H)
    var c = _um(t, C)
    var n = _ud(t, C)
    var ckk = C * KH * KW
    var acc = Float32(0)
    # NN46: solve 0 <= h+PH-kh*DH <= (OH-1)*SH (and width)
    # before the incumbent divisibility tests. This only omits impossible
    # taps, never a floating term, for arbitrary stride/dilation/padding.
    var kh0 = 0
    var kh1 = KH
    var kw0 = 0
    var kw1 = KW
    comptime if NN46_GATHER_BOUNDS:
        if _g(p, CP_REV) == 0:
            var low_h = h + PH - (OH - 1) * SH
            var low_w = w + PW - (OW - 1) * SW
            kh0 = max(0, (max(0, low_h) + DH - 1) // DH)
            kh1 = min(KH, (h + PH) // DH + 1)
            kw0 = max(0, (max(0, low_w) + DW - 1) // DW)
            kw1 = min(KW, (w + PW) // DW + 1)
    for a in range(kh0, kh1):
        var kh = KH - 1 - a if _g(p, CP_REV) != 0 else a
        var th = h + PH - kh * DH
        if th < 0 or _um(th, SH) != 0:
            continue
        var oh = _ud(th, SH)
        if oh >= OH:
            continue
        for kw in range(kw0, kw1):
            var tw = w + PW - kw * DW
            if tw < 0 or _um(tw, SW) != 0:
                continue
            var ow = _ud(tw, SW)
            if ow >= OW:
                continue
            var r = (n * OH + oh) * OW + ow
            acc = ftz(acc + ftz(dcols.unsafe_load(r * ckk + (c * KH + kh) * KW + kw)))
    dx.unsafe_store(i, acc)


@always_inline
def fill_one_at(i: Int, dst: FP, f1: FP, f2: FP, f3: FP, q: IP, p: IP):
    dst.unsafe_store(i, Float32(1))


@always_inline
def gather_rows_at(i: Int, src: FP, dst: FP, f2: FP, f3: FP, q: IP, p: IP):
    """dst[i] = src[q[i // row] * row + i % row], row = p[0]: a 4-byte word
    copy (no float arithmetic touches it), the batch rows of a resident X."""
    var row = Int(p.unsafe_load(0))
    var r = _ud(i, row)
    var j = Int(q.unsafe_load(r)) * row + (i - r * row)
    dst.bitcast[Int32]().unsafe_store(i, src.bitcast[Int32]().unsafe_load(j))


def argmax_row_at(i: Int, proba: FP, f1: FP, f2: FP, f3: FP, q: IP, p: IP):
    """q[i] = the column of row i's first largest value (k = p[0] columns;
    a NaN wins at its first occurrence, as numpy's argmax): the classifier's
    predict on the device (lane pyglue-numeric: numpy's argmax on the host)."""
    var k = Int(p.unsafe_load(0))
    var base = i * k
    var best = 0
    var bv = proba.unsafe_load(base)
    for j in range(1, k):
        if bv != bv:
            break
        var v = proba.unsafe_load(base + j)
        if v != v or v > bv:
            best = j
            bv = v
    q.unsafe_store(i, Int32(best))


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
#: ceil_mode (PyTorch's output-size rule; the last window must start inside
#: the input or its left padding).
comptime PP_CEIL = 16
#: AvgPool2d divisor_override (0: none).
comptime PP_DIVOVR = 17
comptime PP_LEN = 18


def _pool_out(size: Int, k: Int, pad: Int, stride: Int, dil: Int, ceil: Bool) -> Int:
    """aten/src/ATen/native/Pool.h pooling_output_shape_pad_lr."""
    var num = size + 2 * pad - dil * (k - 1) - 1 + ((stride - 1) if ceil else 0)
    var out = num // stride + 1
    if ceil and (out - 1) * stride >= size + pad:
        out -= 1
    return out


def pool_params(raw: List[Int]) raises -> List[Int32]:
    """Validate and fill OH, OW (PyTorch's pooling_output_shape, floor or ceil mode)."""
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
    var ceil = len(raw) > PP_CEIL and raw[PP_CEIL] != 0
    var oh = _pool_out(raw[PP_H], raw[PP_KH], raw[PP_PH], raw[PP_SH], raw[PP_DH], ceil)
    var ow = _pool_out(raw[PP_W], raw[PP_KW], raw[PP_PW], raw[PP_SW], raw[PP_DW], ceil)
    if oh <= 0 or ow <= 0:
        raise Error("x_cnn: the pooling window does not fit the padded input")
    if len(raw) > PP_DIVOVR and raw[PP_DIVOVR] < 0:
        raise Error("x_cnn: divisor_override must be positive")
    var out = List[Int32](capacity=PP_LEN)
    for k in range(PP_LEN):
        out.append(Int32(raw[k]) if k < len(raw) else Int32(0))
    out[PP_OH] = Int32(oh)
    out[PP_OW] = Int32(ow)
    return out^


@always_inline
def maxpool_fwd_at(i: Int, x: FP, dst: FP, f2: FP, f3: FP, idx: IP, p: IP):
    _maxpool_fwd[False](i, x, dst, idx, p)


@always_inline
def relu_maxpool_fwd_at(i: Int, x: FP, dst: FP, f2: FP, f3: FP, idx: IP, p: IP):
    """maxpool(relu(x)) in one pass (DEVIATION 5720): each tap is the value
    `relu_fwd_at` would have stored (`relu_val`), so the max, the winner and
    the stored word are those of the two launches it replaces."""
    _maxpool_fwd[True](i, x, dst, idx, p)


@always_inline
def _maxpool_fwd[RELU: Bool](i: Int, x: FP, dst: FP, idx: IP, p: IP):
    """out[n, c, oh, ow] = the max over the window, taps in (kh, kw)
    ascending order; the FIRST maximum wins a tie (strict >, so -0.0 and
    +0.0 keep whichever came first), a NaN wins (PyTorch's `val > max ||
    isnan(val)`). idx is the flat h*W + w of the winner. DEVIATION 5706
    pins the tie (the reference's CUDA kernel agrees; a `>=` would take the
    last)."""
    var H = _g(p, PP_H); var W = _g(p, PP_W)
    var KH = _g(p, PP_KH); var KW = _g(p, PP_KW)
    var OH = _g(p, PP_OH); var OW = _g(p, PP_OW)
    var ow = _um(i, OW)
    var t = _ud(i, OW)
    var oh = _um(t, OH)
    var nc = _ud(t, OH)
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
            comptime if RELU:
                v = relu_val(v)
            if bi < 0 or v > best or v != v:
                best = v
                bi = h * W + w
    dst.unsafe_store(i, best)
    idx.unsafe_store(i, Int32(bi))


@always_inline
def maxpool_bwd_at(i: Int, dout: FP, dx: FP, f2: FP, f3: FP, idx: IP, p: IP):
    dx.unsafe_store(i, maxpool_bwd_val(i, dout, idx, p))


@always_inline
def maxpool_bwd_val(i: Int, dout: FP, idx: IP, p: IP) -> Float32:
    """dx[n, c, h, w] = the sum of dout over every window whose winner is
    this pixel, gathered in (kh, kw) ascending order from +0.0."""
    var H = _g(p, PP_H); var W = _g(p, PP_W)
    var KH = _g(p, PP_KH); var KW = _g(p, PP_KW)
    var OH = _g(p, PP_OH); var OW = _g(p, PP_OW)
    var SH = _g(p, PP_SH); var SW = _g(p, PP_SW)
    var PH = _g(p, PP_PH); var PW = _g(p, PP_PW)
    var DH = _g(p, PP_DH); var DW = _g(p, PP_DW)
    var w = _um(i, W)
    var t = _ud(i, W)
    var h = _um(t, H)
    var nc = _ud(t, H)
    var me = Int32(h * W + w)
    var acc = Float32(0)
    comptime if not is_defined["MOJOLEARN_XCNN_NO_POOL_BWD_TILE"]():
        # lane/cnn-apple2: windows that tile the input (kernel == stride, no
        # padding, no dilation, no reversed order asked) hold each pixel
        # exactly once, window (h // KH, w // KW): the loop below visits that
        # one window and no other, so this is its one step on the same words.
        if KH == SH and KW == SW and PH == 0 and PW == 0 and DH == 1 and DW == 1 and _g(p, PP_REV) == 0:
            var oh1 = _ud(h, KH)
            var ow1 = _ud(w, KW)
            if oh1 < OH and ow1 < OW:
                var o1 = (nc * OH + oh1) * OW + ow1
                if idx.unsafe_load(o1) == me:
                    acc = ftz(acc + ftz(dout.unsafe_load(o1)))
            return acc
    # NN46: solve 0 <= h+PH-kh*DH <= (OH-1)*SH (and width)
    # before the incumbent divisibility tests. This only omits impossible
    # taps, never a floating term, for arbitrary stride/dilation/padding.
    var kh0 = 0
    var kh1 = KH
    var kw0 = 0
    var kw1 = KW
    comptime if NN46_GATHER_BOUNDS:
        if _g(p, PP_REV) == 0:
            var low_h = h + PH - (OH - 1) * SH
            var low_w = w + PW - (OW - 1) * SW
            kh0 = max(0, (max(0, low_h) + DH - 1) // DH)
            kh1 = min(KH, (h + PH) // DH + 1)
            kw0 = max(0, (max(0, low_w) + DW - 1) // DW)
            kw1 = min(KW, (w + PW) // DW + 1)
    for a in range(kh0, kh1):
        var kh = KH - 1 - a if _g(p, PP_REV) != 0 else a
        var th = h + PH - kh * DH
        if th < 0 or _um(th, SH) != 0:
            continue
        var oh = _ud(th, SH)
        if oh >= OH:
            continue
        for kw in range(kw0, kw1):
            var tw = w + PW - kw * DW
            if tw < 0 or _um(tw, SW) != 0:
                continue
            var ow = _ud(tw, SW)
            if ow >= OW:
                continue
            var o = (nc * OH + oh) * OW + ow
            if idx.unsafe_load(o) == me:
                acc = ftz(acc + ftz(dout.unsafe_load(o)))
    return acc


@always_inline
def _avg_divisor(oh: Int, ow: Int, p: IP) -> Int:
    """PyTorch AvgPool2d's divisor (floor mode): the window clipped to the
    padded input, or to the input itself when count_include_pad is 0."""
    if _g(p, PP_DIVOVR) > 0:
        return _g(p, PP_DIVOVR)
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
    """out = (the window's sum in (kh, kw) ascending order) / divisor: one
    correctly rounded division, never a multiply by the reciprocal
    (DEVIATION 5707)."""
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
    comptime if IDN_AVGPOOL_BWD_TILE:
        # lane fam-neural: tiling windows hold each pixel exactly once, in
        # window (h // KH, w // KW) (the loop below reaches it at kh = h % KH,
        # kw = w % KW and no other), so this is its one step on the same words.
        if KH == SH and KW == SW and PH == 0 and PW == 0:
            var oh1 = _ud(h, KH)
            var ow1 = _ud(w, KW)
            if oh1 < OH and ow1 < OW:
                var o1 = (nc * OH + oh1) * OW + ow1
                var g1 = ftz(identical_div(ftz(dout.unsafe_load(o1)), Float32(_avg_divisor(oh1, ow1, p))))
                acc = ftz(acc + g1)
            dx.unsafe_store(i, acc)
            return
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
def relu_val(v: Float32) -> Float32:
    return v if v > Float32(0) else Float32(0)


@always_inline
def relu_fwd_at(i: Int, x: FP, f1: FP, dst: FP, f3: FP, q: IP, p: IP):
    dst.unsafe_store(i, relu_val(ftz(x.unsafe_load(i))))


@always_inline
def relu_bwd_val(x: Float32, g: Float32) -> Float32:
    """dx = g where x > 0, else +0.0 (PyTorch's threshold_backward); both ftz'd."""
    var v = ftz(x)
    return ftz(g) if v > Float32(0) else Float32(0)


@always_inline
def relu_bwd_at(i: Int, x: FP, g: FP, dx: FP, f3: FP, q: IP, p: IP):
    dx.unsafe_store(i, relu_bwd_val(x.unsafe_load(i), g.unsafe_load(i)))


@always_inline
def pool_relu_rows_bwd_at(i: Int, dpool: FP, yconv: FP, grow: FP, f3: FP, idx: IP, p: IP):
    """The conv block's backward from the pool's output gradient to the
    GEMM's rows in one pass (DEVIATION 5720): grow[r, oc] = dout_rows of
    relu_bwd(yconv, maxpool_bwd(dpool)), each value the one the three
    launches it replaces stored (`maxpool_bwd_val`, `relu_bwd_val`, and
    dout_rows_at's ftz of it). `p` is the conv block (CP_LEN words) followed
    by the pool block; the pool's input is the conv output (NCHW)."""
    var OC = _g(p, CP_OC); var OH = _g(p, CP_OH); var OW = _g(p, CP_OW)
    var r = _ud(i, OC)
    var oc = i - r * OC
    var n = _ud(r, (OH * OW))
    var rem = r - n * OH * OW
    var j = ((n * OC + oc) * OH * OW) + rem
    grow.unsafe_store(i, pool_relu_row_val(j, dpool, yconv, idx, p))


@always_inline
def pool_relu_row_val(j: Int, dpool: FP, yconv: FP, idx: IP, p: IP) -> Float32:
    """`pool_relu_rows_bwd_at`'s stored word for the conv output at NCHW
    index `j` (lane/cnn-apple2: shared with the tiled layout kernel)."""
    var gr = maxpool_bwd_val(j, dpool, idx, p + CP_LEN)
    return ftz(relu_bwd_val(yconv.unsafe_load(j), gr))


@always_inline
def add_at(i: Int, a: FP, b: FP, dst: FP, f3: FP, q: IP, p: IP):
    dst.unsafe_store(i, canon(ftz(ftz(a.unsafe_load(i)) + ftz(b.unsafe_load(i)))))


@always_inline
def bias_rows_at(i: Int, y: FP, bias: FP, dst: FP, f3: FP, q: IP, p: IP):
    """dst[r, j] = y[r, j] + bias[j]; p[2] is the row width."""
    var cols = _g(p, 2)
    dst.unsafe_store(i, canon(ftz(ftz(y.unsafe_load(i)) + ftz(bias.unsafe_load(i % cols)))))


@always_inline
def softmax_xent_row_at(i: Int, logits: FP, grad: FP, proba: FP, rowloss: FP, labels: IP, p: IP):
    """Row i of softmax + mean cross entropy (PyTorch F.cross_entropy,
    reduction='mean'): proba = exp(l - max) / sum, in column order; grad =
    (proba - onehot) / n; rowloss = log(sum) - (l[y] - max). A label < 0
    writes proba only. The mean of rowloss is folded by the caller in row
    order (`seq_mean`). DEVIATION 5708: the exp-sum is folded in column
    order.

    DEVIATION 5705 (Clause B): a row whose max is +inf is where the
    reference computes inf - inf = NaN. Here it is the limit: the +inf
    logits share the mass equally, the rest get +0.0; the loss is +0.0 when
    the label is one of them, else +inf. Any other NaN (a NaN logit) is
    stored as the canonical word."""
    var n = _g(p, 0)
    var k = _g(p, 1)
    var base = i * k
    var mx = ftz(logits.unsafe_load(base))
    for j in range(1, k):
        var v = ftz(logits.unsafe_load(base + j))
        if v > mx:
            mx = v
    var y = Int(labels.unsafe_load(i))
    var inf = bitcast[DType.float32](UInt32(0x7F800000))
    if mx == inf:
        var cnt = 0
        for j in range(k):
            if ftz(logits.unsafe_load(base + j)) == inf:
                cnt += 1
        var share = ftz(identical_div(Float32(1), Float32(cnt)))
        for j in range(k):
            var pr = share if ftz(logits.unsafe_load(base + j)) == inf else Float32(0)
            proba.unsafe_store(base + j, pr)
            if y >= 0:
                var t = ftz(pr - Float32(1)) if j == y else pr
                grad.unsafe_store(base + j, ftz(identical_div(t, Float32(n))))
        if y >= 0:
            rowloss.unsafe_store(i, Float32(0) if ftz(logits.unsafe_load(base + y)) == inf else inf)
        return
    var s = Float32(0)
    comptime if XCNN_SOFTMAX_ONE_EXP:
        # nr-small D9: each exp once, parked in the proba row (the same word
        # the second exp gave), the label's logit read first (proba may be
        # the logits buffer)
        var ly0 = Float32(0)
        if y >= 0:
            ly0 = ftz(ftz(logits.unsafe_load(base + y)) - mx)
        for j in range(k):
            var e = ftz(identical_exp(ftz(ftz(logits.unsafe_load(base + j)) - mx)))
            proba.unsafe_store(base + j, e)
            s = ftz(s + e)
        for j in range(k):
            var pr = canon(ftz(identical_div(proba.unsafe_load(base + j), s)))
            proba.unsafe_store(base + j, pr)
            if y >= 0:
                var t = ftz(pr - Float32(1)) if j == y else pr
                grad.unsafe_store(base + j, canon(ftz(identical_div(t, Float32(n)))))
        if y >= 0:
            rowloss.unsafe_store(i, canon(ftz(ftz(identical_log(s)) - ly0)))
        return
    for j in range(k):
        s = ftz(s + ftz(identical_exp(ftz(ftz(logits.unsafe_load(base + j)) - mx))))
    for j in range(k):
        var e = ftz(identical_exp(ftz(ftz(logits.unsafe_load(base + j)) - mx)))
        var pr = canon(ftz(identical_div(e, s)))
        proba.unsafe_store(base + j, pr)
        if y >= 0:
            var t = ftz(pr - Float32(1)) if j == y else pr
            grad.unsafe_store(base + j, canon(ftz(identical_div(t, Float32(n)))))
    if y >= 0:
        var ly = ftz(ftz(logits.unsafe_load(base + y)) - mx)
        rowloss.unsafe_store(i, canon(ftz(ftz(identical_log(s)) - ly)))


def seq_mean(values: List[Float32], n: Int) -> Float32:
    """The loss mean: a sequential fold in row order, then one division."""
    var acc = Float32(0)
    for i in range(n):
        acc = ftz(acc + ftz(values[i]))
    return canon(ftz(identical_div(acc, Float32(n))))


@always_inline
def sgd_at(i: Int, w: FP, g: FP, v: FP, hyper: FP, q: IP, p: IP):
    """PyTorch SGD (dampening 0, nesterov False) with the momentum buffer
    starting at +0.0: d = g + wd*w; v = momentum*v + d; w = w - lr*v.
    DEVIATION 5709: every product is `identical_mul`, so no backend fuses
    momentum*v + d (or lr*v) into an FMA."""
    var lr = hyper.unsafe_load(0)
    var mom = hyper.unsafe_load(1)
    var wd = hyper.unsafe_load(2)
    var damp = hyper.unsafe_load(3)
    var nesterov = hyper.unsafe_load(4) != Float32(0)
    var first = hyper.unsafe_load(5) != Float32(0)
    var wi = ftz(w.unsafe_load(i))
    var d = ftz(g.unsafe_load(i))
    if wd != Float32(0):
        d = ftz(d + ftz(identical_mul(wd, wi)))
    var vi = d
    if mom != Float32(0):
        if damp != Float32(0) and first:
            vi = d  # torch: the first step's buffer is a clone of d, undamped
        else:
            var dd = d if damp == Float32(0) else ftz(identical_mul(ftz(Float32(1) - damp), d))
            vi = ftz(ftz(identical_mul(mom, ftz(v.unsafe_load(i)))) + dd)
        v.unsafe_store(i, canon(vi))
        if nesterov:
            vi = ftz(d + ftz(identical_mul(mom, vi)))
    w.unsafe_store(i, canon(ftz(wi - ftz(identical_mul(lr, vi)))))


@always_inline
def adam_at(i: Int, w: FP, g: FP, mv: FP, hyper: FP, q: IP, p: IP):
    """torch.optim.Adam (amsgrad False, maximize False), one element; mv =
    [m (n) | v (n)], hyper = [step_size, 1 - beta1, beta2, 1 - beta2, eps,
    sqrt(bias_correction2), weight_decay, adamw] with the step's scalars
    computed by the caller in double as torch does:
      g += wd*w (Adam) or w *= hyper[8] = 1 - lr*wd (AdamW, in double)
      m = m + (1 - b1)(g - m)   (torch's lerp_)
      v = b2 v + (1 - b2) g g
      w = w - step_size * m / (sqrt(v) / sqrt(bc2) + eps)
    every product pinned (DEVIATION 5715)."""
    var n = _g(p, 0)
    var step = hyper.unsafe_load(0)
    var c1 = hyper.unsafe_load(1)
    var b2 = hyper.unsafe_load(2)
    var c2 = hyper.unsafe_load(3)
    var eps = hyper.unsafe_load(4)
    var bc2s = hyper.unsafe_load(5)
    var wd = hyper.unsafe_load(6)
    var decoupled = hyper.unsafe_load(7) != Float32(0)
    var wi = ftz(w.unsafe_load(i))
    var gi = ftz(g.unsafe_load(i))
    if wd != Float32(0):
        if decoupled:
            wi = ftz(identical_mul(wi, hyper.unsafe_load(8)))
        else:
            gi = ftz(gi + ftz(identical_mul(wd, wi)))
    var m = ftz(mv.unsafe_load(i))
    var v = ftz(mv.unsafe_load(n + i))
    m = ftz(m + ftz(identical_mul(c1, ftz(gi - m))))
    v = ftz(ftz(identical_mul(b2, v)) + ftz(identical_mul(c2, ftz(identical_mul(gi, gi)))))
    var denom = ftz(ftz(identical_div(ftz(identical_sqrt(v)), bc2s)) + eps)
    mv.unsafe_store(i, canon(m))
    mv.unsafe_store(n + i, canon(v))
    w.unsafe_store(i, canon(ftz(wi - ftz(identical_mul(step, ftz(identical_div(m, denom)))))))



# ---------------------------------------------------------------- batch norm
# p = [N, C, HW]. One aux buffer per call: [eps, momentum] then seven
# per-channel arrays at 2 + k*C: mean, var (biased), invstd, sum_g, sum_gx,
# gamma, beta. Every per-channel reduction is ONE sequential fold over
# (n, hw) in that order, one thread per channel: its partial count is 1 for
# every shape and every core count (IDENTITY_PATHS row 7's rule; DEVIATION
# 5702). PyTorch's reference (aten/src/ATen/native/cuda/Normalization.cuh)
# uses Welford partials across a block, a schedule of its own.
# Under IDENTICAL the reductions are the BLOCKED folds below (BN_FOLD_BLOCK):
# fixed blocks whose size depends on N * HW alone, on every column.
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
def bn_mean_row(a: Int, N: Int) -> Int:
    """The image the mean's fold visits a-th: ascending (seam 5702; the
    device's threadgroup form, x_cnn/device.mojo `bn_stats_block_kernel`,
    reads the same order through this)."""
    return a


@always_inline
def bn_stats_at(c: Int, x: FP, aux: FP, f2: FP, f3: FP, q: IP, p: IP):
    """Training statistics of channel c: mean, then the biased variance as a
    second fold of (x - mean)^2, invstd = rsqrt(var + eps)."""
    var N = _g(p, 0); var C = _g(p, 1); var HW = _g(p, 2)
    var count = Float32(N * HW)
    var acc = Float32(0)
    for a in range(N):
        var n = bn_mean_row(a, N)
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


# lane idn-loss-norm-folds (2026-10-04): the BLOCKED per-channel folds,
# IDENTICAL only, on every column (the device launches these one thread per
# element, the host calls the same functions). A channel's count = N * HW
# values, in the same (n, hw) order, are cut into consecutive blocks of
# bn_fold_block(count) values; each block is one ascending chain from +0.0
# (one thread per (channel, block)), and the channel's block partials are
# then added ascending from +0.0 (one thread per channel). The block size is
# a function of count alone, so the order is the same on every vendor, the
# host and every core count. With one block the result is the single
# chain's. It replaced one thread folding a whole channel.
# `-D MOJOLEARN_BN_FOLD_BLOCK_OFF` restores the single chain.
comptime BN_FOLD_BLOCK = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (is_defined["MOJOLEARN_BN_FOLD_BLOCK_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())


@always_inline
def bn_fold_block(count: Int) -> Int:
    """Values per block: the smallest power of two B >= 64 with B * B >= count."""
    var b = 64
    while b * b < count:
        b *= 2
    return b


@always_inline
def bn_fold_blocks(count: Int) -> Int:
    """Blocks per channel (at least 1)."""
    var b = bn_fold_block(count)
    var nb = (count + b - 1) // b
    return nb if nb > 0 else 1


@always_inline
def _bn_blk_fold[MODE: Int](t: Int, x: FP, g: FP, aux: FP, p: IP) -> Tuple[Float32, Float32]:
    """Block t = c * NB + b of channel c: the ascending chain over its
    values. MODE 0: (sum x, 0), images through bn_mean_row. MODE 1:
    (sum (x - mean)^2, 0). MODE 2: (sum g, sum g * xhat)."""
    var N = _g(p, 0); var C = _g(p, 1); var HW = _g(p, 2)
    var count = N * HW
    var B = bn_fold_block(count)
    var NB = bn_fold_blocks(count)
    var c = t // NB
    var j = (t - c * NB) * B
    var j1 = min(j + B, count)
    var a = j // HW
    var k = j - a * HW
    var mean = Float32(0)
    var invstd = Float32(0)
    comptime if MODE != 0:
        mean = aux.unsafe_load(_bn(aux, BN_MEAN, c, C))
    comptime if MODE == 2:
        invstd = aux.unsafe_load(_bn(aux, BN_INVSTD, c, C))
    var s0 = Float32(0)
    var s1 = Float32(0)
    while j < j1:
        var n = a
        comptime if MODE == 0:
            n = bn_mean_row(a, N)
        var base = (n * C + c) * HW
        var kend = min(HW, k + (j1 - j))
        var kk = k
        while kk < kend:
            comptime if MODE == 0:
                s0 = ftz(s0 + ftz(x.unsafe_load(base + kk)))
            elif MODE == 1:
                var d = ftz(ftz(x.unsafe_load(base + kk)) - mean)
                s0 = ftz(s0 + ftz(identical_mul(d, d)))
            else:
                var gv = ftz(g.unsafe_load(base + kk))
                var xhat = ftz(identical_mul(ftz(ftz(x.unsafe_load(base + kk)) - mean), invstd))
                s0 = ftz(s0 + gv)
                s1 = ftz(s1 + ftz(identical_mul(gv, xhat)))
            kk += 1
        j += kend - k
        k = 0
        a += 1
    return (s0, s1)


@always_inline
def _bn_blk_total(c: Int, part: FP, NB: Int) -> Float32:
    """The NB block partials of channel c added ascending from +0.0."""
    var acc = Float32(0)
    for b in range(NB):
        acc = ftz(acc + part.unsafe_load(c * NB + b))
    return acc


@always_inline
def bn_blk_sum_at(t: Int, x: FP, part: FP, f2: FP, f3: FP, q: IP, p: IP):
    """part[t] = block t's sum of x (part holds C * NB words)."""
    part.unsafe_store(t, _bn_blk_fold[0](t, x, x, x, p)[0])


@always_inline
def bn_blk_mean_at(c: Int, part: FP, aux: FP, f2: FP, f3: FP, q: IP, p: IP):
    """Channel c's mean from its block sums."""
    var N = _g(p, 0); var C = _g(p, 1); var HW = _g(p, 2)
    var acc = _bn_blk_total(c, part, bn_fold_blocks(N * HW))
    aux.unsafe_store(_bn(aux, BN_MEAN, c, C), ftz(identical_div(acc, Float32(N * HW))))


@always_inline
def bn_blk_sq_at(t: Int, x: FP, part: FP, aux: FP, f3: FP, q: IP, p: IP):
    """part[t] = block t's sum of (x - mean)^2."""
    part.unsafe_store(t, _bn_blk_fold[1](t, x, x, aux, p)[0])


@always_inline
def bn_blk_var_at(c: Int, part: FP, aux: FP, f2: FP, f3: FP, q: IP, p: IP):
    """Channel c's biased variance and invstd from its block sums."""
    var N = _g(p, 0); var C = _g(p, 1); var HW = _g(p, 2)
    var sq = _bn_blk_total(c, part, bn_fold_blocks(N * HW))
    var var_b = ftz(identical_div(sq, Float32(N * HW)))
    aux.unsafe_store(_bn(aux, BN_VAR, c, C), var_b)
    aux.unsafe_store(_bn(aux, BN_INVSTD, c, C), ftz(identical_rsqrt(ftz(var_b + aux.unsafe_load(0)))))


@always_inline
def bn_blk_red_at(t: Int, x: FP, g: FP, aux: FP, part: FP, q: IP, p: IP):
    """part[t] = block t's sum of g, part[C * NB + t] = its sum of g * xhat
    (part holds 2 * C * NB words)."""
    var N = _g(p, 0); var C = _g(p, 1); var HW = _g(p, 2)
    var r = _bn_blk_fold[2](t, x, g, aux, p)
    part.unsafe_store(t, r[0])
    part.unsafe_store(C * bn_fold_blocks(N * HW) + t, r[1])


@always_inline
def bn_blk_red_fin_at(c: Int, part: FP, aux: FP, f2: FP, f3: FP, q: IP, p: IP):
    """Channel c's sum_g and sum_gx from its block sums."""
    var N = _g(p, 0); var C = _g(p, 1); var HW = _g(p, 2)
    var NB = bn_fold_blocks(N * HW)
    aux.unsafe_store(_bn(aux, BN_SUMG, c, C), _bn_blk_total(c, part, NB))
    aux.unsafe_store(_bn(aux, BN_SUMGX, c, C), _bn_blk_total(C + c, part, NB))


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


# lane idn-loss-norm-folds (2026-10-04): the device draws a channel's mask
# value ONCE (dropout2d_chan_at, one thread per (n, c)) and the element pass
# reads it (dropout2d_apply_at), instead of ten Philox rounds and a division
# per element. Same draw, same threshold, same product: no bit moves.
# IDENTICAL only. `-D MOJOLEARN_DROPOUT2D_CH_MASK_OFF` restores the
# per-element draw. The host column keeps dropout2d_at (the same words).
comptime DROPOUT2D_CH_MASK = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (is_defined["MOJOLEARN_DROPOUT2D_CH_MASK_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())


@always_inline
def dropout2d_mask_val(nc: Int, hyper: FP, p: IP) -> Float32:
    """Channel nc = n*C + c's mask value: 0 or 1 / (1 - p)."""
    var key = SIMD[DType.uint32, 2](UInt32(p.unsafe_load(3)), UInt32(p.unsafe_load(4)))
    var ctr = SIMD[DType.uint32, 4](UInt32(nc), 0, 0, 0)
    var u = philox4x32_10(ctr, key)[0]
    var thresh = (UInt64(UInt32(p.unsafe_load(5))) << 16) | UInt64(UInt32(p.unsafe_load(6)))
    var m = Float32(0)
    if UInt64(u) >= thresh:
        m = ftz(identical_div(Float32(1), ftz(Float32(1) - hyper.unsafe_load(0))))
    return m


@always_inline
def dropout2d_at(i: Int, x: FP, mask: FP, dst: FP, hyper: FP, q: IP, p: IP):
    """PyTorch nn.Dropout2d (training): every (n, c) channel is zeroed with
    probability p, the rest scaled by 1 / (1 - p). The draw is Philox4x32-10
    at counter (n*C + c, 0, 0, 0) under the 64-bit seed, word 0 compared as
    an INTEGER against thresh = round(p * 2^32): no float in the decision
    (DEVIATION 5703; torch's own stream is its generator's, not ours)."""
    var HW = _g(p, 2)
    var m = dropout2d_mask_val(i // HW, hyper, p)
    mask.unsafe_store(i, m)
    dst.unsafe_store(i, ftz(identical_mul(ftz(x.unsafe_load(i)), m)))


@always_inline
def dropout2d_chan_at(nc: Int, tab: FP, hyper: FP, f2: FP, f3: FP, q: IP, p: IP):
    """tab[nc] = channel nc's mask value (tab holds N * C words)."""
    tab.unsafe_store(nc, dropout2d_mask_val(nc, hyper, p))


@always_inline
def dropout2d_apply_at(i: Int, x: FP, mask: FP, dst: FP, tab: FP, q: IP, p: IP):
    """dropout2d_at's two stores, the mask value read from tab."""
    var m = tab.unsafe_load(_ud(i, _g(p, 2)))
    mask.unsafe_store(i, m)
    dst.unsafe_store(i, ftz(identical_mul(ftz(x.unsafe_load(i)), m)))


@always_inline
def mul_at(i: Int, a: FP, b: FP, dst: FP, f3: FP, q: IP, p: IP):
    dst.unsafe_store(i, ftz(identical_mul(ftz(a.unsafe_load(i)), ftz(b.unsafe_load(i)))))



# ---------------------------------------------------------------- graph ops
# A CSR adjacency in one Int32 array q = [rowptr (n+1) | col (nnz) | row (nnz)]
# (row[e] is the row of entry e); p = [n, F, nnz, mode]. Rows are TARGET
# nodes for the forward propagation (PyG flow source_to_target) and SOURCE
# nodes for the transposed (backward) one; entries within a row are in
# ascending column order, so every fold below has one fixed order
# (DEVIATION 5704: PyG scatter-adds with atomics).


@always_inline
def spmm_at(i: Int, vals: FP, h: FP, dst: FP, f3: FP, q: IP, p: IP):
    """dst[r, f] = fold over row r's entries e, in order, of
    mode 0: vals[e] * h[col[e], f];  mode 1: h[col[e], f], then / count
    (PyG mean aggregation; an empty row is +0.0);  mode 2: h[col[e], f] / vals[e];
    mode 3 (no fold): h[r, f] / count, row r's entry count read from the
    offsets (the in-degree on the forward view; +0.0 for an empty row), the
    mean's backward scale before a mode-0 fold of the transposed view."""
    var n = _g(p, 0); var F = _g(p, 1); var mode = _g(p, 3)
    var r = i // F
    var f = i - r * F
    var lo = _g(q, r)
    var hi = _g(q, r + 1)
    if mode == 3:
        var d = Float32(0)
        if hi > lo:
            d = ftz(identical_div(ftz(h.unsafe_load(i)), Float32(hi - lo)))
        dst.unsafe_store(i, d)
        return
    var acc = Float32(0)
    for e in range(lo, hi):
        var c = _g(q, n + 1 + e)
        var v = ftz(h.unsafe_load(c * F + f))
        if mode == 0:
            v = ftz(identical_mul(ftz(vals.unsafe_load(e)), v))
        elif mode == 2:
            v = ftz(identical_div(v, ftz(vals.unsafe_load(e))))
        acc = ftz(acc + v)
    if mode == 1 and hi > lo:
        acc = ftz(identical_div(acc, Float32(hi - lo)))
    dst.unsafe_store(i, acc)


@always_inline
def gcn_deg_at(r: Int, w: FP, dis: FP, f2: FP, f3: FP, q: IP, p: IP):
    """PyG gcn_norm: deg[r] = the fold of row r's edge weights (self loop
    included by the caller); dis = deg^-1/2, +0.0 where deg is 0."""
    var lo = _g(q, r)
    var hi = _g(q, r + 1)
    var acc = Float32(0)
    for e in range(lo, hi):
        acc = ftz(acc + ftz(w.unsafe_load(e)))
    dis.unsafe_store(r, ftz(identical_rsqrt(acc)) if acc > Float32(0) else Float32(0))


@always_inline
def gcn_norm_at(e: Int, w: FP, dis: FP, vals: FP, f3: FP, q: IP, p: IP):
    """norm[e] = dis[src] * w[e] * dis[dst], left to right as PyG's
    `deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]` (DEVIATION 5710:
    that association, pinned)."""
    var n = _g(p, 0); var nnz = _g(p, 2)
    var src = _g(q, n + 1 + e)
    var dst = _g(q, n + 1 + nnz + e)
    var t = ftz(identical_mul(ftz(dis.unsafe_load(src)), ftz(w.unsafe_load(e))))
    vals.unsafe_store(e, ftz(identical_mul(t, ftz(dis.unsafe_load(dst)))))



# ---------------------------------------------------------------- padding
# p = [N, C, H, W, top, bottom, left, right, mode]; mode 0 zeros, 1 reflect,
# 2 replicate, 3 circular (torch.nn.functional.pad, 2-D).
comptime PAD_ZEROS = 0
comptime PAD_REFLECT = 1
comptime PAD_REPLICATE = 2
comptime PAD_CIRCULAR = 3


@always_inline
def _pad_src(hp: Int, before: Int, size: Int, mode: Int) -> Int:
    """The source index of padded position hp, or -1 (zeros outside)."""
    var h = hp - before
    if h >= 0 and h < size:
        return h
    if mode == PAD_ZEROS:
        return -1
    if mode == PAD_REFLECT:
        return -h if h < 0 else 2 * (size - 1) - h
    if mode == PAD_REPLICATE:
        return 0 if h < 0 else size - 1
    return (h % size + size) % size


@always_inline
def pad_fwd_at(i: Int, x: FP, f1: FP, dst: FP, f3: FP, q: IP, p: IP):
    var H = _g(p, 2); var W = _g(p, 3)
    var Hp = H + _g(p, 4) + _g(p, 5)
    var Wp = W + _g(p, 6) + _g(p, 7)
    var mode = _g(p, 8)
    var wp = i % Wp
    var t = i // Wp
    var hp = t % Hp
    var nc = t // Hp
    var h = _pad_src(hp, _g(p, 4), H, mode)
    var w = _pad_src(wp, _g(p, 6), W, mode)
    var v = Float32(0)
    if h >= 0 and w >= 0:
        v = ftz(x.unsafe_load((nc * H + h) * W + w))
    dst.unsafe_store(i, v)


@always_inline
def pad_bwd_at(i: Int, g: FP, f1: FP, dx: FP, f3: FP, q: IP, p: IP):
    """dx[n, c, h, w] = the sum of the padded gradient over every padded
    position that reads this pixel, gathered in (hp, wp) ascending order
    from +0.0 (DEVIATION 5711; the reference's reflect/replicate backward
    kernels scatter with atomics)."""
    var H = _g(p, 2); var W = _g(p, 3)
    var top = _g(p, 4); var left = _g(p, 6)
    var Hp = H + top + _g(p, 5)
    var Wp = W + left + _g(p, 7)
    var mode = _g(p, 8)
    var w = i % W
    var t = i // W
    var h = t % H
    var nc = t // H
    var acc = Float32(0)
    comptime if IDN_PAD_BWD_BOUNDED:
        # lane fam-neural: an interior padded position hp in [top, top + H)
        # reads source hp - top, so only hp = h + top can read row h; every
        # other reader is in a margin. Three ascending, disjoint ranges per
        # axis (top margin, that one position, bottom margin) with the same
        # test inside: the terms of the full scan in its order.
        for sh in range(3):
            var h0 = 0 if sh == 0 else (h + top if sh == 1 else top + H)
            var h1 = top if sh == 0 else (h + top + 1 if sh == 1 else Hp)
            for hp in range(h0, h1):
                if _pad_src(hp, top, H, mode) != h:
                    continue
                for sw in range(3):
                    var w0 = 0 if sw == 0 else (w + left if sw == 1 else left + W)
                    var w1 = left if sw == 0 else (w + left + 1 if sw == 1 else Wp)
                    for wp in range(w0, w1):
                        if _pad_src(wp, left, W, mode) != w:
                            continue
                        acc = ftz(acc + ftz(g.unsafe_load((nc * Hp + hp) * Wp + wp)))
        dx.unsafe_store(i, acc)
        return
    for hp in range(Hp):
        if _pad_src(hp, top, H, mode) != h:
            continue
        for wp in range(Wp):
            if _pad_src(wp, left, W, mode) != w:
                continue
            acc = ftz(acc + ftz(g.unsafe_load((nc * Hp + hp) * Wp + wp)))
    dx.unsafe_store(i, acc)



# ---------------------------------------------------------------- channel groups
# lane fam-neural (2026-10-04): a grouped convolution's channel slice and its
# inverse as device word copies (the host path slices with NumPy and uploads
# each group). p = [N, C, HW, cg, c0]: the group is channels [c0, c0 + cg) of
# an (N, C, HW) tensor; the part is (N, cg, HW). Words are copied, never
# converted, so no float value changes.


@always_inline
def chan_slice_at(i: Int, src: FP, f1: FP, dst: FP, f3: FP, q: IP, p: IP):
    """part[i] = full[n, c0 + c, s] for i = (n * cg + c) * HW + s."""
    var C = _g(p, 1); var HW = _g(p, 2); var cg = _g(p, 3); var c0 = _g(p, 4)
    var per = cg * HW
    var n = _ud(i, per)
    var rem = i - n * per
    dst.unsafe_store(i, src.unsafe_load((n * C + c0) * HW + rem))


@always_inline
def chan_place_at(i: Int, src: FP, f1: FP, dst: FP, f3: FP, q: IP, p: IP):
    """full[n, c0 + c, s] = part[i]: each part word has its own full word,
    so the launch's writes never collide."""
    var C = _g(p, 1); var HW = _g(p, 2); var cg = _g(p, 3); var c0 = _g(p, 4)
    var per = cg * HW
    var n = _ud(i, per)
    var rem = i - n * per
    dst.unsafe_store((n * C + c0) * HW + rem, src.unsafe_load(i))



# ---------------------------------------------------------------- adaptive pooling
# p = [N, C, H, W, OH, OW]; output cell (oh, ow) reads rows
# [floor(oh*H/OH), ceil((oh+1)*H/OH)) and the columns likewise
# (aten/src/ATen/native/AdaptivePooling.h start_index / end_index).


@always_inline
def _astart(a: Int, osize: Int, isize: Int) -> Int:
    return (a * isize) // osize


@always_inline
def _aend(a: Int, osize: Int, isize: Int) -> Int:
    return ((a + 1) * isize + osize - 1) // osize


@always_inline
def _acell_lo(a: Int, osize: Int, isize: Int) -> Int:
    """The first output cell whose window can hold input index `a`:
    _aend(o) > a  <=>  (o + 1) * isize > a * osize  <=>  o >= floor(a * osize / isize)."""
    comptime if IDN_ADAPT_BWD_BOUNDED:
        return (a * osize) // isize
    return 0


@always_inline
def _acell_hi(a: Int, osize: Int, isize: Int) -> Int:
    """One past the last such cell: _astart(o) <= a  <=>  o * isize <
    (a + 1) * osize  <=>  o < ceil((a + 1) * osize / isize) (at most osize)."""
    comptime if IDN_ADAPT_BWD_BOUNDED:
        return ((a + 1) * osize + isize - 1) // isize
    return osize


@always_inline
def adapt_avg_fwd_at(i: Int, x: FP, f1: FP, dst: FP, f3: FP, q: IP, p: IP):
    """The window's sum in (h, w) order, then one division by its size."""
    var H = _g(p, 2); var W = _g(p, 3); var OH = _g(p, 4); var OW = _g(p, 5)
    var ow = i % OW
    var t = i // OW
    var oh = t % OH
    var nc = t // OH
    var hs = _astart(oh, OH, H); var he = _aend(oh, OH, H)
    var ws = _astart(ow, OW, W); var we = _aend(ow, OW, W)
    var acc = Float32(0)
    for h in range(hs, he):
        for w in range(ws, we):
            acc = ftz(acc + ftz(x.unsafe_load((nc * H + h) * W + w)))
    dst.unsafe_store(i, ftz(identical_div(acc, Float32((he - hs) * (we - ws)))))


# lane fix-n1-lm-neural (2026-10-04, audit B9), IDENTICAL only, every column:
# the adaptive average forward (global average pooling is its 1 x 1 case)
# was one thread per output folding the whole window in (h, w) order. With
# IDN_GAP_BLOCK_FOLD the window's values, in the same (h, w) order, are cut
# into consecutive blocks of B = bn_fold_block(KH * KW) values (KH, KW the
# largest window extents of the shape, so B depends on the shape alone):
# one ascending chain from +0.0 per (output, block) (`adapt_avg_blk_at`),
# then the output's block partials added ascending from +0.0 and one
# division by the window size (`adapt_avg_fin_at`). A window of at most B
# values (B >= 64, so every window up to 8 x 8) is one block: the old chain's
# bits exactly; larger windows move bits on the devices and the host column
# together. `-D MOJOLEARN_IDN_GAP_BLOCK_FOLD_OFF` restores the single chain.
comptime IDN_GAP_BLOCK_FOLD = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (is_defined["MOJOLEARN_IDN_GAP_BLOCK_FOLD_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())


@always_inline
def adapt_max_extent(isize: Int, osize: Int) -> Int:
    """An upper bound of every adaptive window's extent along one axis."""
    var k = (isize + osize - 1) // osize + 1
    return k if k < isize else isize


@always_inline
def gap_fold_block(H: Int, W: Int, OH: Int, OW: Int) -> Int:
    """Values per block of the blocked adaptive average fold."""
    return bn_fold_block(adapt_max_extent(H, OH) * adapt_max_extent(W, OW))


@always_inline
def gap_fold_blocks(H: Int, W: Int, OH: Int, OW: Int) -> Int:
    """Block slots per output (at least 1); 1 means the single chain runs."""
    var c = adapt_max_extent(H, OH) * adapt_max_extent(W, OW)
    var b = bn_fold_block(c)
    var nb = (c + b - 1) // b
    return nb if nb > 0 else 1


@always_inline
def adapt_avg_blk_at(t: Int, x: FP, f1: FP, part: FP, f3: FP, q: IP, p: IP):
    """Slot t = i * NB + b: the ascending chain over block b of output i's
    window values in (h, w) order (+0.0 for a block past the window)."""
    var H = _g(p, 2); var W = _g(p, 3); var OH = _g(p, 4); var OW = _g(p, 5)
    var B = gap_fold_block(H, W, OH, OW)
    var NB = gap_fold_blocks(H, W, OH, OW)
    var i = t // NB
    var blk = t - i * NB
    var ow = i % OW
    var r = i // OW
    var oh = r % OH
    var nc = r // OH
    var hs = _astart(oh, OH, H); var he = _aend(oh, OH, H)
    var ws = _astart(ow, OW, W); var we = _aend(ow, OW, W)
    var ww = we - ws
    var cnt = (he - hs) * ww
    var j1 = min((blk + 1) * B, cnt)
    var acc = Float32(0)
    for k in range(blk * B, j1):
        var h = hs + k // ww
        var w = ws + k % ww
        acc = ftz(acc + ftz(x.unsafe_load((nc * H + h) * W + w)))
    part.unsafe_store(t, acc)


@always_inline
def adapt_avg_fin_at(i: Int, part: FP, f1: FP, dst: FP, f3: FP, q: IP, p: IP):
    """Output i: its block partials added ascending from +0.0, then one
    division by the window size."""
    var H = _g(p, 2); var W = _g(p, 3); var OH = _g(p, 4); var OW = _g(p, 5)
    var B = gap_fold_block(H, W, OH, OW)
    var NB = gap_fold_blocks(H, W, OH, OW)
    var ow = i % OW
    var oh = (i // OW) % OH
    var hs = _astart(oh, OH, H); var he = _aend(oh, OH, H)
    var ws = _astart(ow, OW, W); var we = _aend(ow, OW, W)
    var cnt = (he - hs) * (we - ws)
    var nb = (cnt + B - 1) // B
    var acc = Float32(0)
    for b in range(nb):
        acc = ftz(acc + part.unsafe_load(i * NB + b))
    dst.unsafe_store(i, ftz(identical_div(acc, Float32(cnt))))


@always_inline
def adapt_avg_bwd_at(i: Int, g: FP, f1: FP, dx: FP, f3: FP, q: IP, p: IP):
    """dx[n, c, h, w] = the sum over the windows holding the pixel of
    g / window size, gathered in (oh, ow) ascending order (DEVIATION 5712)."""
    var H = _g(p, 2); var W = _g(p, 3); var OH = _g(p, 4); var OW = _g(p, 5)
    var w = i % W
    var t = i // W
    var h = t % H
    var nc = t // H
    var acc = Float32(0)
    # lane fam-neural (IDN_ADAPT_BWD_BOUNDED): only the cells that can hold
    # the pixel, ascending, the membership test kept
    for oh in range(_acell_lo(h, OH, H), _acell_hi(h, OH, H)):
        var hs = _astart(oh, OH, H); var he = _aend(oh, OH, H)
        if h < hs or h >= he:
            continue
        for ow in range(_acell_lo(w, OW, W), _acell_hi(w, OW, W)):
            var ws = _astart(ow, OW, W); var we = _aend(ow, OW, W)
            if w < ws or w >= we:
                continue
            var gv = ftz(g.unsafe_load((nc * OH + oh) * OW + ow))
            acc = ftz(acc + ftz(identical_div(gv, Float32((he - hs) * (we - ws)))))
    dx.unsafe_store(i, acc)


@always_inline
def adapt_max_fwd_at(i: Int, x: FP, f1: FP, dst: FP, f3: FP, idx: IP, p: IP):
    """The first maximum in (h, w) order wins, a NaN wins (DEVIATION 5706's tie)."""
    var H = _g(p, 2); var W = _g(p, 3); var OH = _g(p, 4); var OW = _g(p, 5)
    var ow = i % OW
    var t = i // OW
    var oh = t % OH
    var nc = t // OH
    var best = Float32(0)
    var bi = -1
    for h in range(_astart(oh, OH, H), _aend(oh, OH, H)):
        for w in range(_astart(ow, OW, W), _aend(ow, OW, W)):
            var v = ftz(x.unsafe_load((nc * H + h) * W + w))
            if bi < 0 or v > best or v != v:
                best = v
                bi = h * W + w
    dst.unsafe_store(i, best)
    idx.unsafe_store(i, Int32(bi))


@always_inline
def adapt_max_bwd_at(i: Int, g: FP, f1: FP, dx: FP, f3: FP, idx: IP, p: IP):
    var H = _g(p, 2); var W = _g(p, 3); var OH = _g(p, 4); var OW = _g(p, 5)
    var w = i % W
    var t = i // W
    var h = t % H
    var nc = t // H
    var me = Int32(h * W + w)
    var acc = Float32(0)
    for oh in range(_acell_lo(h, OH, H), _acell_hi(h, OH, H)):
        if h < _astart(oh, OH, H) or h >= _aend(oh, OH, H):
            continue
        for ow in range(_acell_lo(w, OW, W), _acell_hi(w, OW, W)):
            if w < _astart(ow, OW, W) or w >= _aend(ow, OW, W):
                continue
            var o = (nc * OH + oh) * OW + ow
            if idx.unsafe_load(o) == me:
                acc = ftz(acc + ftz(g.unsafe_load(o)))
    dx.unsafe_store(i, acc)



# ---------------------------------------------------------------- SAGE max, L2 normalize
@always_inline
def sage_max_fwd_at(i: Int, h: FP, f1: FP, aux: FP, dst: FP, q: IP, p: IP):
    """PyG MaxAggregation (torch.scatter_reduce 'amax', include_self=False):
    the max over row r's entries in column order (a NaN wins), +0.0 for an
    empty row; aux[i] = the max, aux[n*F + i] = how many entries equal it
    (the backward splits the gradient evenly among them, as amax's does)."""
    var n = _g(p, 0); var F = _g(p, 1)
    var r = i // F
    var f = i - r * F
    var lo = _g(q, r)
    var hi = _g(q, r + 1)
    var mx = Float32(0)
    var cnt = 0
    for e in range(lo, hi):
        var v = ftz(h.unsafe_load(_g(q, n + 1 + e) * F + f))
        if e == lo or v > mx or v != v:
            mx = v
            cnt = 1 if v == v else 0
        elif v == mx:
            cnt += 1
    aux.unsafe_store(i, mx)
    aux.unsafe_store(n * F + i, Float32(cnt))
    dst.unsafe_store(i, mx)


@always_inline
def sage_max_bwd_at(i: Int, h: FP, g: FP, aux: FP, dst: FP, q: IP, p: IP):
    """dx[s, f] = the sum over s's outgoing entries (the TRANSPOSED CSR,
    targets ascending) whose value is the target's max of g[t, f] / count,
    gathered in order (DEVIATION 5713; the reference scatters)."""
    var n = _g(p, 0); var F = _g(p, 1)
    var r = i // F
    var f = i - r * F
    var v = ftz(h.unsafe_load(i))
    var acc = Float32(0)
    for e in range(_g(q, r), _g(q, r + 1)):
        var t = _g(q, n + 1 + e)
        var o = t * F + f
        if v == aux.unsafe_load(o) and aux.unsafe_load(n * F + o) > Float32(0):
            acc = ftz(acc + ftz(identical_div(ftz(g.unsafe_load(o)), aux.unsafe_load(n * F + o))))
    dst.unsafe_store(i, acc)


@always_inline
def l2norm_fwd_at(r: Int, x: FP, f1: FP, aux: FP, dst: FP, q: IP, p: IP):
    """torch.nn.functional.normalize(p=2, dim=-1, eps=1e-12) of row r: the
    squares folded in column order, one sqrt, the division by max(norm, eps).
    aux[r] = that denominator."""
    var F = _g(p, 1)
    var sq = Float32(0)
    for f in range(F):
        var v = ftz(x.unsafe_load(r * F + f))
        sq = ftz(sq + ftz(identical_mul(v, v)))
    var norm = ftz(identical_sqrt(sq))
    var den = norm if norm > Float32(1e-12) else Float32(1e-12)
    aux.unsafe_store(r, den)
    for f in range(F):
        dst.unsafe_store(r * F + f, ftz(identical_div(ftz(x.unsafe_load(r * F + f)), den)))


@always_inline
def l2norm_bwd_at(r: Int, y: FP, g: FP, aux: FP, dst: FP, q: IP, p: IP):
    """dx = (g - y * sum(g * y)) / den when the norm cleared eps, else g / eps
    (the clamp's derivative is zero); the dot folded in column order."""
    var F = _g(p, 1)
    var den = aux.unsafe_load(r)
    if den == Float32(1e-12):
        for f in range(F):
            dst.unsafe_store(r * F + f, ftz(identical_div(ftz(g.unsafe_load(r * F + f)), den)))
        return
    var dot = Float32(0)
    for f in range(F):
        dot = ftz(dot + ftz(identical_mul(ftz(g.unsafe_load(r * F + f)), ftz(y.unsafe_load(r * F + f)))))
    for f in range(F):
        var t = ftz(ftz(g.unsafe_load(r * F + f)) - ftz(identical_mul(ftz(y.unsafe_load(r * F + f)), dot)))
        dst.unsafe_store(r * F + f, ftz(identical_div(t, den)))


# ------------------------------------------------- lane fam2-neural (2026-10-04)
# CNNClassifier.fit's last host steps, as element functions every column
# runs (the device launches them; x_cnn/host/ops_host.mojo loops them):
#
# IDN_XENT_DEV_FOLD: the mean of the softmax row losses was n words read
#   back and folded on the host in row order (`seq_mean`). It is now a
#   BLOCKED fold on the device: blocks of LOSS_FOLD_BLOCK consecutive values
#   folded ascending, the block sums folded the same way, level by level,
#   and one division by n at the last level (`blk_fold_at`). The fold order
#   depends on n alone; the host column runs the same function, so the bits
#   move on all four columns together. `-D MOJOLEARN_IDN_XENT_DEV_FOLD_OFF`.
# IDN_CNN_EPOCH_DEV: (a) the epoch's row order was a Fisher-Yates walk on the
#   CPU (`epoch_order_i32`), uploaded per step. It is now a counter-based
#   permutation each thread computes for its own position (`epoch_rows_at`:
#   a six-round Feistel network over the smallest even-width domain holding
#   n, cycle-walked into [0, n); 32-bit integers only). (b) Adam's step
#   scalars were double arithmetic on the CPU (`adam_hyper_f64`). They are
#   now computed on the device per step (`adam_hyper_at`) in double-float32
#   arithmetic (error-free sums and Dekker products, every product pinned),
#   about 2^-45 relative before the final rounding to float32. Both change
#   bits, on every column together. `-D MOJOLEARN_IDN_CNN_EPOCH_DEV_OFF`.
comptime _FAM2_IDN = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
# cpu2-l11-neural (2026-10-04): FAST takes the same device forms on every
# vendor (no host epoch order, Adam scalars, loss fold or GCN loops on a GPU
# route). FAST promises quality, never bits, and has no `_OFF` arm here; the
# `_OFF` defines keep their meaning for the IDENTICAL A/B only.
comptime _FAM2_FAST = GLOBAL_NUMERIC_MODE == NUMERIC_FAST
comptime IDN_XENT_DEV_FOLD = _FAM2_FAST or (_FAM2_IDN and not is_defined["MOJOLEARN_IDN_XENT_DEV_FOLD_OFF"]())
comptime IDN_CNN_EPOCH_DEV = _FAM2_FAST or (IDN_XENT_DEV_FOLD and not is_defined["MOJOLEARN_IDN_CNN_EPOCH_DEV_OFF"]())
# IDN_GCN_LOOPS_DEV (lane fix-n1-lm-neural, audit F7 `gcn_self_loops`):
#   GCNConv's add_remaining_self_loops (drop every existing loop, append one
#   loop per node carrying the LAST existing loop's weight, else the fill)
#   was host NumPy over the edges (`_expansion_cnn.py` GCNConv._graph). It is
#   now `x_cnn_gcn_loops`: a stable radix sort of the edges by key
#   (0 for a non-loop, 1 + node for a loop) and two gather kernels on the
#   device (`gcn_loops_device`), the host twin `gcn_loops_host` on a CPU-only
#   install. Selection and copies only: no arithmetic, so no bit moves on any
#   column. Lane py-runtime round 2: the NumPy form is deleted; Python always
#   calls `x_cnn_gcn_loops` (this flag only reports bit 2 of `x_cnn_idn2_flags`).
comptime IDN_GCN_LOOPS_DEV = _FAM2_FAST or (_FAM2_IDN and not is_defined["MOJOLEARN_IDN_GCN_LOOPS_DEV_OFF"]())
#: CANDIDATE ARM (default OFF): `-D MOJOLEARN_IDN_XENT_FOLD_BLOCK_256` folds
#: blocks of 256 (one level up to 256 rows, two up to 65,536) instead of 32
#: (one level up to 32 rows, two up to 1,024): fewer launches per loss, a
#: longer one-thread chain per block. A different fold order, so different
#: bits, on every column together (the host column reads the same constant).
#: Dead under MOJOLEARN_IDN_ALL_OFF and reported as `x_cnn_idn2_flags` bit 3
#: (lane/review-fixes), so the glue can see which fold a binding was built with.
comptime IDN_XENT_FOLD_BLOCK_256 = is_defined["MOJOLEARN_IDN_XENT_FOLD_BLOCK_256"]() and not is_defined[
    "MOJOLEARN_IDN_ALL_OFF"
]()
comptime LOSS_FOLD_BLOCK = 256 if IDN_XENT_FOLD_BLOCK_256 else 32


@always_inline
def blk_fold_at(t: Int, src: FP, dst: FP, f2: FP, f3: FP, q: IP, p: IP):
    """dst[p[2] + t] = src[t*B : min((t+1)*B, p[0])] added ascending from
    +0.0 (B = LOSS_FOLD_BLOCK); when p[1] > 0 (the last level) the sum is
    divided by p[1]."""
    var count = _g(p, 0)
    var div = _g(p, 1)
    var lo = t * LOSS_FOLD_BLOCK
    var hi = lo + LOSS_FOLD_BLOCK
    if hi > count:
        hi = count
    var acc = Float32(0)
    for i in range(lo, hi):
        acc = ftz(acc + ftz(src.unsafe_load(i)))
    if div > 0:
        acc = ftz(identical_div(acc, Float32(div)))
    dst.unsafe_store(_g(p, 2) + t, canon(acc))


def fold_plan(count: Int, div: Int, index: Int) -> List[Int32]:
    """`blk_fold_at`'s parameter triples, one per level, for `count` values:
    [values at this level, 0, 0] until the level that leaves one block,
    which is [values, div, index]."""
    var prm = List[Int32]()
    var c = count
    while True:
        var nb = (c + LOSS_FOLD_BLOCK - 1) // LOSS_FOLD_BLOCK
        var last = nb <= 1
        prm.append(Int32(c))
        prm.append(Int32(div if last else 0))
        prm.append(Int32(index if last else 0))
        if last:
            break
        c = nb
    return prm^


# ---- the epoch order
comptime EP_N = 0  # rows
comptime EP_POS = 1  # the position of element 0 in the epoch
comptime EP_SHUFFLE = 2
comptime EP_HALF = 3  # the Feistel half width in bits
comptime EP_KEY = 4  # the 64-bit epoch key as four 16-bit words, low first
comptime EP_LEN = 8


@always_inline
def _perm_mix(x: UInt32) -> UInt32:
    """murmur3's 32-bit finalizer (a bijection of the 32-bit words)."""
    var z = x
    z = (z ^ (z >> 16)) * UInt32(0x85EBCA6B)
    z = (z ^ (z >> 13)) * UInt32(0xC2B2AE35)
    return z ^ (z >> 16)


def epoch_key(seed: UInt64, epoch: Int) -> UInt64:
    """Epoch `epoch`'s permutation key: splitmix64's output at state
    seed + (epoch + 1) * golden (the counter is the epoch)."""
    var z = seed + UInt64(epoch + 1) * UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def epoch_rows_prm(n: Int, pos: Int, shuffle: Bool, key: UInt64) -> List[Int32]:
    """`epoch_rows_at`'s parameter block for positions pos, pos + 1, ... of
    an epoch over n rows (1 <= n < 2^31)."""
    var bits = 1
    while (1 << bits) < n:
        bits += 1
    var prm = List[Int32]()
    prm.append(Int32(n))
    prm.append(Int32(pos))
    prm.append(Int32(1 if shuffle else 0))
    prm.append(Int32((bits + 1) // 2))
    for w in range(4):
        prm.append(Int32(Int((key >> UInt64(16 * w)) & UInt64(0xFFFF))))
    return prm^


@always_inline
def epoch_rows_at(i: Int, f0: FP, f1: FP, f2: FP, f3: FP, q: IP, p: IP):
    """q[i] = the row at position p[EP_POS] + i of the epoch's order: the
    position itself without shuffle, else its image under the epoch's
    permutation of [0, n): six Feistel rounds on the two EP_HALF-bit halves
    (round function `_perm_mix` of the half plus the round key, masked),
    repeated until the image is below n (cycle walking: the image of a
    permutation of the 2^(2 half) words restricted to [0, n) is a
    permutation of [0, n)). Integers only: the same row on every column."""
    var n = UInt32(_g(p, EP_N))
    var pos = _g(p, EP_POS) + i
    if _g(p, EP_SHUFFLE) == 0:
        q.unsafe_store(i, Int32(pos))
        return
    var h = UInt32(_g(p, EP_HALF))
    var mask = (UInt32(1) << h) - UInt32(1)
    var k0 = UInt32(_g(p, EP_KEY)) | (UInt32(_g(p, EP_KEY + 1)) << UInt32(16))
    var k1 = UInt32(_g(p, EP_KEY + 2)) | (UInt32(_g(p, EP_KEY + 3)) << UInt32(16))
    var x = UInt32(pos)
    while True:
        var l = (x >> h) & mask
        var r = x & mask
        for rd in range(6):
            var rk = _perm_mix(k0 + UInt32(rd) * UInt32(0x9E3779B9)) ^ k1
            var f = _perm_mix(r + rk) & mask
            var t = l ^ f
            l = r
            r = t
        x = (l << h) | r
        if x < n:
            break
    q.unsafe_store(i, Int32(Int(x)))


# ---- Adam's step scalars in double-float32
@always_inline
def _two_sum(a: Float32, b: Float32) -> Tuple[Float32, Float32]:
    """(s, e) with s = fl(a + b) and s + e = a + b exactly (Knuth)."""
    var s = ftz(a + b)
    var bb = ftz(s - a)
    var e = ftz(ftz(a - ftz(s - bb)) + ftz(b - bb))
    return (s, e)


@always_inline
def _two_prod(a: Float32, b: Float32) -> Tuple[Float32, Float32]:
    """(p, e) with p = fl(a b) and p + e = a b exactly (Dekker's split, no
    fused operation; every product pinned)."""
    var p = ftz(identical_mul(a, b))
    var ca = ftz(identical_mul(Float32(4097), a))
    var ah = ftz(ca - ftz(ca - a))
    var al = ftz(a - ah)
    var cb = ftz(identical_mul(Float32(4097), b))
    var bh = ftz(cb - ftz(cb - b))
    var bl = ftz(b - bh)
    var e = ftz(ftz(identical_mul(ah, bh)) - p)
    e = ftz(e + ftz(identical_mul(ah, bl)))
    e = ftz(e + ftz(identical_mul(al, bh)))
    e = ftz(e + ftz(identical_mul(al, bl)))
    return (p, e)


@always_inline
def _dd_mul(ah: Float32, al: Float32, bh: Float32, bl: Float32) -> Tuple[Float32, Float32]:
    """(ah + al)(bh + bl) as a normalized pair."""
    var pe = _two_prod(ah, bh)
    var e = ftz(pe[1] + ftz(ftz(identical_mul(ah, bl)) + ftz(identical_mul(al, bh))))
    return _two_sum(pe[0], e)


@always_inline
def _dd_powi(bh: Float32, bl: Float32, e: Int) -> Tuple[Float32, Float32]:
    """(bh + bl) ** e for e >= 0 by squaring, in one fixed order."""
    var rh = Float32(1)
    var rl = Float32(0)
    var xh = bh
    var xl = bl
    var k = e
    while k > 0:
        if (k & 1) != 0:
            var r = _dd_mul(rh, rl, xh, xl)
            rh = r[0]
            rl = r[1]
        var x = _dd_mul(xh, xl, xh, xl)
        xh = x[0]
        xl = x[1]
        k >>= 1
    return (rh, rl)


@always_inline
def _dd_one_minus(h: Float32, l: Float32) -> Tuple[Float32, Float32]:
    """1 - (h + l) as a normalized pair."""
    var se = _two_sum(Float32(1), -h)
    return _two_sum(se[0], ftz(se[1] - l))


comptime AH_BASE = 10  # [lr hi, lr lo, b1 hi, b1 lo, b2 hi, b2 lo, eps, wd hi, wd lo, decoupled]
comptime AH_ROW = 9


@always_inline
def adam_hyper_at(t: Int, base: FP, hy: FP, f2: FP, f3: FP, q: IP, p: IP):
    """hy[9 t : 9 t + 9] = `adam_at`'s hyper block of 1-based step p[0] + t:
    [lr / bc1, 1 - b1, b2, 1 - b2, eps, sqrt(bc2), wd, decoupled, 1 - lr wd]
    with bc = 1 - beta ** step, each value the float32 nearest a
    double-float32 result (the pairs in `base` are the float32 nearest the
    caller's double and the float32 nearest the remainder)."""
    var step = _g(p, 0) + t
    var lrh = base.unsafe_load(0)
    var lrl = base.unsafe_load(1)
    var b1h = base.unsafe_load(2)
    var b1l = base.unsafe_load(3)
    var b2h = base.unsafe_load(4)
    var b2l = base.unsafe_load(5)
    var wdh = base.unsafe_load(7)
    var wdl = base.unsafe_load(8)
    var p1 = _dd_powi(b1h, b1l, step)
    var c1 = _dd_one_minus(p1[0], p1[1])
    var p2 = _dd_powi(b2h, b2l, step)
    var c2 = _dd_one_minus(p2[0], p2[1])
    # lr / bc1: the float32 quotient plus its correction
    var q0 = ftz(identical_div(lrh, c1[0]))
    var step_size = q0
    if c1[0] != Float32(0):
        var pe = _two_prod(q0, c1[0])
        var rem = ftz(ftz(ftz(lrh - pe[0]) - pe[1]) + lrl)
        rem = ftz(rem - ftz(identical_mul(q0, c1[1])))
        step_size = ftz(q0 + ftz(identical_div(rem, c1[0])))
    # sqrt(bc2): the float32 root plus one Newton correction
    var s0 = ftz(identical_sqrt(c2[0]))
    var sq = s0
    if s0 > Float32(0):
        var ps = _two_prod(s0, s0)
        var rs = ftz(ftz(ftz(c2[0] - ps[0]) - ps[1]) + c2[1])
        sq = ftz(s0 + ftz(identical_div(rs, ftz(s0 + s0))))
    var o1 = _dd_one_minus(b1h, b1l)
    var o2 = _dd_one_minus(b2h, b2l)
    var lw = _dd_mul(lrh, lrl, wdh, wdl)
    var ow = _dd_one_minus(lw[0], lw[1])
    var row = t * AH_ROW
    hy.unsafe_store(row + 0, canon(step_size))
    hy.unsafe_store(row + 1, canon(o1[0]))
    hy.unsafe_store(row + 2, b2h)
    hy.unsafe_store(row + 3, canon(o2[0]))
    hy.unsafe_store(row + 4, base.unsafe_load(6))
    hy.unsafe_store(row + 5, canon(sq))
    hy.unsafe_store(row + 6, wdh)
    hy.unsafe_store(row + 7, base.unsafe_load(9))
    hy.unsafe_store(row + 8, canon(ow[0]))


def adam_hyper_base(lr: Float64, b1: Float64, b2: Float64, eps: Float64, wd: Float64, dec: Float64) -> List[Float32]:
    """`adam_hyper_at`'s base block: each double hyperparameter as the
    float32 nearest it and the float32 nearest the remainder (two casts and
    one exact subtraction; no step arithmetic)."""
    var out = List[Float32]()
    var lrh = Float32(lr)
    out.append(lrh)
    out.append(Float32(lr - Float64(lrh)))
    var b1h = Float32(b1)
    out.append(b1h)
    out.append(Float32(b1 - Float64(b1h)))
    var b2h = Float32(b2)
    out.append(b2h)
    out.append(Float32(b2 - Float64(b2h)))
    out.append(Float32(eps))
    var wdh = Float32(wd)
    out.append(wdh)
    out.append(Float32(wd - Float64(wdh)))
    out.append(Float32(dec))
    return out^


def idn2_flags() -> Int:
    """The lane fam2-neural switches this build has on (both bindings export
    it as `x_cnn_idn2_flags`): bit 0 IDN_XENT_DEV_FOLD, bit 1 IDN_CNN_EPOCH_DEV,
    bit 2 IDN_GCN_LOOPS_DEV (lane fix-n1-lm-neural); bit 3 the candidate
    IDN_XENT_FOLD_BLOCK_256 and bit 4 a `-D MOJOLEARN_IDN_ALL_OFF` build
    (lane/review-fixes: the glue reads the build's OFF arm from here, not
    only from the environment)."""
    var f = 0
    comptime if IDN_XENT_DEV_FOLD:
        f |= 1
    comptime if IDN_CNN_EPOCH_DEV:
        f |= 2
    comptime if IDN_GCN_LOOPS_DEV:
        f |= 4
    comptime if IDN_XENT_FOLD_BLOCK_256:
        f |= 8
    comptime if is_defined["MOJOLEARN_IDN_ALL_OFF"]():
        f |= 16
    return f


@always_inline
def nn45_conv_out_relu_at(i: Int, y2: FP, bias: FP, saved: FP, dst: FP, q: IP, p: IP):
    """Fused layout/bias/ReLU, with the exact preactivation retained for
    backward. conv_out_val and relu_val preserve the old rounded seams,
    NaNs and activation-zero policy; saved may alias dst when not retained."""
    var OC = _g(p, CP_OC)
    var S = _g(p, CP_OH) * _g(p, CP_OW)
    var nc = i // S
    var pos = i - nc * S
    var oc = nc % OC
    var n = nc // OC
    var v = conv_out_val(y2.unsafe_load((n * S + pos) * OC + oc), bias, oc, p)
    saved.unsafe_store(i, v)
    dst.unsafe_store(i, relu_val(ftz(v)))


@always_inline
def nn47_bn_apply_running_at(i: Int, x: FP, aux: FP, dst: FP, running: FP, q: IP, p: IP):
    """One normalization launch also updates each channel's running state
    exactly once. Statistics are already complete; apply never reads running.
    The first pixel of each channel of image zero owns its state update.
    Mean/variance folds, unbiased conversion, momentum and eps stay pinned."""
    bn_apply_at(i, x, aux, dst, dst, q, p)
    var C = _g(p, 1)
    var HW = _g(p, 2)
    if i < C * HW and i % HW == 0:
        bn_running_at(i // HW, running, aux, aux, aux, q, p)


# NN14: bounded forward/recompute and virtual-im2col backward. OFF/unmeasured.
# The budget is 8 MiB total cols+y2, except an indivisible single row.
# All reductions retain the complete K and original per-cell profile;
# absolute output row positions are restored before NCHW writes.
comptime NN14_BOUNDED_IM2COL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_NN14_BOUNDED_IM2COL"]() and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()


def nn14_conv_rows(rows: Int, k: Int, oc: Int) -> Int:
    return min(rows, max(1, (1 << 21) // max(1, k + oc)))


@always_inline
def nn14_im2col_value(i: Int, x: FP, p: IP) -> Float32:
    """cols[r, qq], r = (n*OH + oh)*OW + ow, qq = (c*KH + kh)*KW + kw: the
    input pixel under that tap, or +0.0 in the zero padding. A copy, no
    arithmetic."""
    var C = _g(p, CP_C); var H = _g(p, CP_H); var W = _g(p, CP_W)
    var KH = _g(p, CP_KH); var KW = _g(p, CP_KW)
    var OH = _g(p, CP_OH); var OW = _g(p, CP_OW)
    var ckk = C * KH * KW
    var r = _ud(i, ckk)
    var qq = i - r * ckk
    var n = _ud(r, (OH * OW))
    var rem = r - n * OH * OW
    var oh = _ud(rem, OW)
    var ow = rem - oh * OW
    var c = _ud(qq, (KH * KW))
    var t = qq - c * KH * KW
    var kh = _ud(t, KW)
    var kw = t - kh * KW
    var h = oh * _g(p, CP_SH) - _g(p, CP_PH) + kh * _g(p, CP_DH)
    var w = ow * _g(p, CP_SW) - _g(p, CP_PW) + kw * _g(p, CP_DW)
    var v = Float32(0)
    if h >= 0 and h < H and w >= 0 and w < W:
        v = ftz(x.unsafe_load(((n * C + c) * H + h) * W + w))
    return v


@always_inline
def nn14_im2col_slice_cell(local: Int, x: FP, cols: FP, p: IP, row0: Int):
    var width = _g(p, CP_C) * _g(p, CP_KH) * _g(p, CP_KW)
    cols.unsafe_store(local, nn14_im2col_value(local + row0 * width, x, p))



@always_inline
def nn14_conv_slice_cell(i: Int, y2: FP, bias: FP, dst: FP, p: IP, row0: Int):
    var OC = _g(p, CP_OC)
    var S = _g(p, CP_OH) * _g(p, CP_OW)
    var local_row = i // OC
    var oc = i % OC
    var row = row0 + local_row
    var n = row // S
    var pos = row % S
    dst.unsafe_store((n * OC + oc) * S + pos, conv_out_val(y2.unsafe_load(i), bias, oc, p))


@always_inline
def nn14_wgrad_at(i: Int, x: FP, grow: FP, dw: FP, f3: FP, q: IP, p: IP):
    """TN contraction against a virtual im2col operand. The selected GEMM
    leaf/chain/fold is complete over ALL rows; chunk boundaries never change
    this reduction. Scratch is fixed per output cell, independent of rows."""
    var width = _g(p, CP_C) * _g(p, CP_KH) * _g(p, CP_KW)
    var oc = i // width
    var qq = i % width
    var OC = _g(p, CP_OC)
    var rows = _g(p, CP_N) * _g(p, CP_OH) * _g(p, CP_OW)
    var part = neural_partition[NEURAL_LEAF](rows)
    var stack = SIMD[DType.float32, 16](0.0)
    var occupied = 0
    for leaf in range(part[1]):
        var begin = leaf * part[0]
        var accum = SIMD[DType.float32, NEURAL_CHAINS](0.0)
        for row in range(begin, min(rows, begin + part[0])):
            var chain = (row - begin) % NEURAL_CHAINS
            accum[chain] = rtf_mul_add(ftz(grow.unsafe_load(row * OC + oc)),
                nn14_im2col_value(row * width + qq, x, p), accum[chain])
        neural_fold_push[16](stack, occupied, neural_merge_chains[NEURAL_CHAINS](accum))
    dw.unsafe_store(i, neural_fold_drain[16](stack, occupied))


@always_inline
def nn14_input_grad_at(i: Int, grow: FP, weights: FP, dx: FP, f3: FP, q: IP, p: IP):
    """col2im's original tap gather, with each dcols cell generated by the
    selected complete OC contraction. No scatter/atomics or full dcols."""
    var C = _g(p, CP_C); var H = _g(p, CP_H); var W = _g(p, CP_W)
    var KH = _g(p, CP_KH); var KW = _g(p, CP_KW)
    var OH = _g(p, CP_OH); var OW = _g(p, CP_OW); var OC = _g(p, CP_OC)
    var SH = _g(p, CP_SH); var SW = _g(p, CP_SW)
    var PH = _g(p, CP_PH); var PW = _g(p, CP_PW)
    var DH = _g(p, CP_DH); var DW = _g(p, CP_DW)
    var w = i % W
    var h = (i // W) % H
    var c = (i // (W * H)) % C
    var n = i // (W * H * C)
    var width = C * KH * KW
    var part = neural_partition[NEURAL_LEAF](OC)
    var acc = Float32(0)
    var kh0 = 0; var kh1 = KH; var kw0 = 0; var kw1 = KW
    comptime if NN46_GATHER_BOUNDS:
        if _g(p, CP_REV) == 0:
            kh0 = max(0, (max(0, h + PH - (OH - 1) * SH) + DH - 1) // DH)
            kh1 = min(KH, (h + PH) // DH + 1)
            kw0 = max(0, (max(0, w + PW - (OW - 1) * SW) + DW - 1) // DW)
            kw1 = min(KW, (w + PW) // DW + 1)
    for a in range(kh0, kh1):
        var kh = KH - 1 - a if _g(p, CP_REV) != 0 else a
        var th = h + PH - kh * DH
        if th < 0 or th % SH != 0 or th // SH >= OH:
            continue
        for kw in range(kw0, kw1):
            var tw = w + PW - kw * DW
            if tw < 0 or tw % SW != 0 or tw // SW >= OW:
                continue
            var row = (n * OH + th // SH) * OW + tw // SW
            var qq = (c * KH + kh) * KW + kw
            var v = neural_cell[NEURAL_CHAINS](grow, weights, row, qq, OC,
                part[0], part[1], OC, 1, width, 1)
            acc = ftz(acc + ftz(v))
    dx.unsafe_store(i, acc)
