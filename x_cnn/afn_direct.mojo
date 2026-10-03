# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Apple FAST candidate for the CNN's convolution (lane afn-mlp, 2026-10-03):
`-D MOJOLEARN_AFN_CNN_DIRECT`, compiled only under FAST + Apple, default
OFF. IDENTICAL and every other vendor compile `x_cnn/device.mojo`'s paths
unchanged.

THE FORWARD main runs for a layer whose `k = C*KH*KW` is above the
one-leaf direct kernel's bound (`DIRECT_CONV`: k <= 32 and OC*k <= 2048,
the first 3x3 layer of a 3-channel image only): `im2col` (a launch writing
`rows x k` floats), the GEMM `cols . W^T` (its own launches and the
`rows x OC` product), and the NCHW + bias layout launch. Three launches and
two round trips through device memory for one convolution.

Here: ONE implicit-GEMM launch. A block owns 64 output positions x 32
output channels; the input taps of a 32-wide k chunk (`AT_TR x AT_KC`,
gathered straight from NCHW `x` with the zero padding applied) and the
matching weight tile (`AT_TO x AT_KC`) are staged in threadgroup memory,
each thread accumulates one position x 8 channels in registers, f32, the
k chunks in ascending order (the free fold FAST allows: a different
chunking from the pinned GEMM's leaves). The epilogue stores the NCHW word
with the bias (`conv_out_val`, the same word main's layout launch stores)
and, when the caller asks, `relu(.)` into a second output, so the conv
block's no-pool forward skips its ReLU launch. The im2col words are still
written to `cols` when the backward will read them (`save_cols`), from
the same staged values.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from x_cnn.ops import (
    CP_C, CP_DH, CP_DW, CP_H, CP_KH, CP_KW, CP_OH, CP_OW, CP_PH, CP_PW, CP_SH, CP_SW, CP_W,
    FP, IP, _g, _ud, conv_out_val, relu_val,
)

comptime AFN_CNN_DIRECT = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator() and not is_defined["MOJOLEARN_COLUMN_CPU"]()
    and is_defined["MOJOLEARN_AFN_CNN_DIRECT"]()
)
#: `x_cnn/device.mojo`'s `DC_MAXK` and `DC_MAXW`, restated (that file
#: imports this one): the one-leaf direct kernel keeps its shapes.
comptime AFN_DC_MAXK = 32
comptime AFN_DC_MAXW = 2048
comptime AT_TR = 64
comptime AT_TO = 32
comptime AT_KC = 32
comptime AT_TPB = 256
comptime AT_PER = AT_TO // (AT_TPB // AT_TR)  # 8 channels per thread


def afn_conv_direct_applies(ckk: Int, OC: Int) -> Bool:
    """The shapes main sends to im2col + GEMM (the one-leaf direct kernel
    keeps its own)."""
    return not (ckk <= AFN_DC_MAXK and OC * ckk <= AFN_DC_MAXW)


def afn_conv_tiled_kernel(
    x: FP, w: FP, bias: FP, cols: FP, yconv: FP, relu_out: FP, p: IP,
    rows_in: Int32, OC_in: Int32, save_cols: Int32, want_relu: Int32,
):
    var s_w = stack_allocation[AT_TO * AT_KC, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var s_a = stack_allocation[AT_TR * AT_KC, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var s_n = stack_allocation[AT_TR, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var s_rem = stack_allocation[AT_TR, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var s_h0 = stack_allocation[AT_TR, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var s_w0 = stack_allocation[AT_TR, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var rows = Int(rows_in)
    var OC = Int(OC_in)
    var C = _g(p, CP_C); var H = _g(p, CP_H); var W = _g(p, CP_W)
    var KH = _g(p, CP_KH); var KW = _g(p, CP_KW)
    var OH = _g(p, CP_OH); var OW = _g(p, CP_OW)
    var DH = _g(p, CP_DH); var DW = _g(p, CP_DW)
    var S = OH * OW
    var ckk = C * KH * KW
    var row0 = Int(block_idx.x) * AT_TR
    var oc0 = Int(block_idx.y) * AT_TO
    if tid < AT_TR:
        var r = row0 + tid
        var n = 0
        var rem = 0
        if r < rows:
            n = _ud(r, S)
            rem = r - n * S
        var oh = _ud(rem, OW)
        var ow = rem - oh * OW
        s_n[tid] = Int32(n)
        s_rem[tid] = Int32(rem)
        s_h0[tid] = Int32(oh * _g(p, CP_SH) - _g(p, CP_PH))
        s_w0[tid] = Int32(ow * _g(p, CP_SW) - _g(p, CP_PW))
    barrier()
    var rl = tid % AT_TR
    var grp = tid // AT_TR
    var kk = tid % AT_KC
    var acc = InlineArray[Float32, AT_PER](fill=Float32(0))
    for k0 in range(0, ckk, AT_KC):
        # the weight tile: AT_TO x AT_KC words, four per thread
        comptime for j in range(AT_TO * AT_KC // AT_TPB):
            var idx = tid + AT_TPB * j
            var o = idx // AT_KC
            var q = k0 + (idx - o * AT_KC)
            var oc = oc0 + o
            var wv = Float32(0)
            if oc < OC and q < ckk:
                wv = w.unsafe_load(oc * ckk + q)
            s_w[idx] = wv
        # the input tile: AT_TR x AT_KC words, eight per thread, the tap
        # (c, kh, kw) of column kk decoded once
        var q = k0 + kk
        var c = 0
        var kh = 0
        var kw = 0
        if q < ckk:
            c = _ud(q, KH * KW)
            var t = q - c * KH * KW
            kh = _ud(t, KW)
            kw = t - kh * KW
        comptime for j in range(AT_TR * AT_KC // AT_TPB):
            var rl2 = tid // AT_KC + (AT_TPB // AT_KC) * j
            var r = row0 + rl2
            var v = Float32(0)
            if r < rows and q < ckk:
                var h = Int(s_h0[rl2]) + kh * DH
                var ww = Int(s_w0[rl2]) + kw * DW
                if h >= 0 and h < H and ww >= 0 and ww < W:
                    v = x.unsafe_load(((Int(s_n[rl2]) * C + c) * H + h) * W + ww)
                if save_cols != Int32(0):
                    cols.unsafe_store(r * ckk + q, v)
            s_a[rl2 * AT_KC + kk] = v
        barrier()
        for k in range(AT_KC):
            var av = s_a[rl * AT_KC + k]
            comptime for o in range(AT_PER):
                acc[o] += av * s_w[(grp * AT_PER + o) * AT_KC + k]
        barrier()
    var r = row0 + rl
    if r >= rows:
        return
    var n = Int(s_n[rl])
    var rem = Int(s_rem[rl])
    comptime for o in range(AT_PER):
        var oc = oc0 + grp * AT_PER + o
        if oc < OC:
            var v = conv_out_val(acc[o], bias, oc, p)
            var idx = (n * OC + oc) * S + rem
            yconv.unsafe_store(idx, v)
            if want_relu != Int32(0):
                relu_out.unsafe_store(idx, relu_val(v))


def afn_conv_direct_launch(
    ctx: DeviceContext, x: FP, w: FP, bias: FP, cols: FP, yconv: FP, relu_out: FP, p: IP,
    rows: Int, OC: Int, save_cols: Bool, want_relu: Bool,
) raises:
    """One launch: `yconv` (and `relu_out` when `want_relu`) from `x`, `w`
    and `bias`; `cols` written when `save_cols`."""
    comptime if not AFN_CNN_DIRECT:
        raise Error("afn_conv_direct_launch: compiled without MOJOLEARN_AFN_CNN_DIRECT")
    else:
        if rows < 1 or OC < 1:
            return
        ctx.enqueue_function[afn_conv_tiled_kernel](
            x, w, bias, cols, yconv, relu_out, p, Int32(rows), Int32(OC),
            Int32(1 if save_cols else 0), Int32(1 if want_relu else 0),
            grid_dim=((rows + AT_TR - 1) // AT_TR, (OC + AT_TO - 1) // AT_TO, 1),
            block_dim=(AT_TPB, 1, 1),
        )
