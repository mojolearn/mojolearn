# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CNN lane's launch-word layouts and output-shape rules: integers only.

`conv_params` / `pool_params` validate a convolution's or a pooling layer's
Int32 parameter block and fill its output height and width. They moved here
from `x_cnn/ops.mojo` unchanged (ne3-c-cnn, 2026-11-07; ops.mojo imports and
re-exports every name, so `from x_cnn.ops import conv_params` still works)
so that the native layer owners (`bindings/xcnn_conv_native.mojo`,
`bindings/xcnn_pool_native.mojo`, `bindings/xcnn_classifier_native.mojo`)
compute an output shape with the same function instead of a binding
crossing (`x_cnn_conv_shape` / `x_cnn_pool_shape`), without importing the
element functions, the numerics switches or the device runtime.

This module imports nothing: no floating value, no fold order, no device.
"""


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


def conv_params(raw: List[Int]) raises -> List[Int32]:
    """Validate the Python-side block and fill OH, OW."""
    if len(raw) < CP_BIAS + 1:
        raise Error("x_cnn: conv params need 16 entries")
    for k in range(CP_OH):  # small-loop(the fixed launch-word block)
        if raw[k] < 0:
            raise Error("x_cnn: negative conv parameter")
    var positive: List[Int] = [CP_N, CP_C, CP_H, CP_W, CP_OC, CP_KH, CP_KW, CP_SH, CP_SW, CP_DH, CP_DW]
    for k in positive:  # small-loop(the fixed launch-word block)
        if raw[k] <= 0:
            raise Error("x_cnn: conv dimensions, stride and dilation must be positive")
    var oh = (raw[CP_H] + 2 * raw[CP_PH] - raw[CP_DH] * (raw[CP_KH] - 1) - 1) // raw[CP_SH] + 1
    var ow = (raw[CP_W] + 2 * raw[CP_PW] - raw[CP_DW] * (raw[CP_KW] - 1) - 1) // raw[CP_SW] + 1
    if oh <= 0 or ow <= 0:
        raise Error("x_cnn: the kernel does not fit the padded input")
    var out = List[Int32](capacity=CP_LEN)
    for k in range(CP_LEN):  # small-loop(the fixed launch-word block)
        out.append(Int32(raw[k]) if k < len(raw) else Int32(0))
    out[CP_OH] = Int32(oh)
    out[CP_OW] = Int32(ow)
    return out^


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
    for k in range(PP_OH):  # small-loop(the fixed launch-word block)
        if raw[k] < 0:
            raise Error("x_cnn: negative pool parameter")
    var positive: List[Int] = [PP_N, PP_C, PP_H, PP_W, PP_KH, PP_KW, PP_SH, PP_SW, PP_DH, PP_DW]
    for k in positive:  # small-loop(the fixed launch-word block)
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
    for k in range(PP_LEN):  # small-loop(the fixed launch-word block)
        out.append(Int32(raw[k]) if k < len(raw) else Int32(0))
    out[PP_OH] = Int32(oh)
    out[PP_OW] = Int32(ow)
    return out^
