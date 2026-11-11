# SPDX-License-Identifier: Apache-2.0
"""RUN half of Conv2d / Conv1d / ReLU / BasicBlock (`bindings/xcnn_conv_native.mojo`
is the UNPACK/PACK boundary; lane doorsplit-cnn-xseq, 2026-11-11).

Typed values only: the convolution configuration checks of `__init__`, the
launch words (`ConvGeom` -> `Conv2d._params`), the padded and output shapes,
the padding words, the residency-word prefix of the `_m` operations, the host
channel-group copies over raw float32 words, and the native padding,
convolution, channel-copy, map, ReLU and add entries. The boundary reads the
layer's fields into these words, owns every buffer (addresses and handles
arrive as words), releases the GIL around the entry calls here, and writes
the fields back.
"""
from std.math import sqrt
from std.memory import unsafe_memcpy
from core.xcnn_entry_table import (
    XcnnEntryKey, XC_PAD2D_FORWARD, XC_PAD2D_BACKWARD, XC_PAD2D_M, XC_CONV2D_FORWARD, XC_CONV2D_FORWARD_M,
    XC_CONV2D_BACKWARD, XC_CONV2D_BACKWARD_M, XC_CHAN_COPY_M, XC_MAP2_M, XC_RELU_FORWARD, XC_RELU_BACKWARD, XC_ADD,
)
from x_cnn.door_run import XcnnF32P, xcnn_run_call, xcnn_size
from x_cnn.shape_words import conv_params, CP_OH, CP_OW


@fieldwise_init
struct ConvGeom(Copyable, Movable):
    """The layer fields `Conv2d._params` reads (UNPACK, in the door's read
    order): groups, kernel, stride, the convolution's own padding
    (`_conv_pad`), dilation, out_channels and the bias switch."""
    var groups: Int
    var kh: Int
    var kw: Int
    var sh: Int
    var sw: Int
    var ph: Int
    var pw: Int
    var dh: Int
    var dw: Int
    var out_channels: Int
    var bias: Bool


# ---------------------------------------------------------------- configuration (`__init__`)

def conv_run_check_config(inc: Int, outc: Int, groups: Int, k: List[Int], s: List[Int], d: List[Int]) raises:
    """`__init__`'s channel, group, kernel, stride and dilation refusals."""
    if inc <= 0 or outc <= 0 or groups <= 0 or inc % groups or outc % groups:
        raise Error("mojolearn: positive in_channels and out_channels must be divisible by groups")
    if k[0] <= 0 or k[1] <= 0 or s[0] <= 0 or s[1] <= 0 or d[0] <= 0 or d[1] <= 0:
        raise Error("mojolearn: kernel_size, stride and dilation must be positive")


def conv_run_pad_code(mode: String) -> Int:
    """The padding-mode word of the pad entries: zeros 0, reflect 1,
    replicate 2, circular 3."""
    return 0 if mode == "zeros" else (1 if mode == "reflect" else (2 if mode == "replicate" else 3))


def conv_run_check_padding_mode(mode: String) raises:
    if mode != "zeros" and mode != "reflect" and mode != "replicate" and mode != "circular":
        raise Error("mojolearn: padding_mode must be zeros, reflect, replicate or circular")


def conv_run_same_pads(k: List[Int], s: List[Int], d: List[Int]) raises -> List[Int]:
    """`padding='same'`: (top, bottom, left, right), the extra word after."""
    if s[0] != 1 or s[1] != 1:
        raise Error("mojolearn: padding='same' is not supported for strided convolutions")
    var h = d[0] * (k[0] - 1)
    var w = d[1] * (k[1] - 1)
    return [h // 2, h - h // 2, w // 2, w - w // 2]


def conv_run_text_pads(text: String, k: List[Int], s: List[Int], d: List[Int]) raises -> List[Int]:
    """A padding string: 'same' pads, 'valid' none, anything else refused."""
    if text == "same":
        return conv_run_same_pads(k, s, d)
    if text != "valid":
        raise Error("mojolearn: invalid padding string; expected valid or same")
    return [0, 0, 0, 0]


def conv_run_size_pads(p: List[Int], one: Bool) raises -> List[Int]:
    """A padding size pair (Conv1d: its height word is zero): symmetric
    (top, bottom, left, right)."""
    var ph = 0 if one else p[0]
    if ph < 0 or p[1] < 0:
        raise Error("mojolearn: padding must be nonnegative")
    return [ph, ph, p[1], p[1]]


def conv_run_explicit(mode: String, pads: List[Int]) -> Bool:
    """Whether the input is padded explicitly before the convolution (a
    non-zero mode or asymmetric pads)."""
    return mode != "zeros" or pads[0] != pads[1] or pads[2] != pads[3]


def conv_run_conv_pad(explicit: Bool, pads: List[Int]) -> List[Int]:
    """`_conv_pad`: the convolution's own symmetric padding."""
    return [0 if explicit else pads[0], 0 if explicit else pads[2]]


def conv_run_fan(inc: Int, groups: Int, k: List[Int]) -> Int:
    return inc // groups * k[0] * k[1]


def conv_run_bound(fan: Int) -> Float64:
    """The uniform initialization bound 1 / sqrt(fan_in)."""
    return 1.0 / sqrt(Float64(fan))


# ---------------------------------------------------------------- shape and launch words

def conv_run_same_shape(left: List[Int], right: List[Int]) -> Bool:
    if len(left) != len(right):
        return False
    for i in range(len(left)):  # small-loop(tensor rank)
        if left[i] != right[i]:
            return False
    return True


def conv_run_words(g: ConvGeom, shape: List[Int]) -> List[Int]:
    """`Conv2d._params(shape)` as native words: one group's conv over the
    (padded) input (the launch block before `conv_params` fills OH, OW)."""
    return [shape[0], shape[1] // g.groups, shape[2], shape[3], g.out_channels // g.groups,
            g.kh, g.kw, g.sh, g.sw, g.ph, g.pw, g.dh, g.dw, 0, 0, 1 if g.bias else 0, 0]


def conv_run_padded(explicit: Bool, pads: List[Int], shape: List[Int]) -> List[Int]:
    if not explicit:
        return shape.copy()
    return [shape[0], shape[1], shape[2] + pads[0] + pads[1], shape[3] + pads[2] + pads[3]]


def conv_run_out(g: ConvGeom, padded: List[Int], shape: List[Int]) raises -> List[Int]:
    """`Conv2d._out_shape(shape)`: the integer rule of `conv_params`
    (x_cnn/shape_words.mojo, with its refusals) over the padded input."""
    var prm = conv_params(conv_run_words(g, padded))
    return [shape[0], g.out_channels, Int(prm[CP_OH]), Int(prm[CP_OW])]


def conv_run_check_rank(shape: List[Int]) raises:
    if len(shape) != 4:
        raise Error("mojolearn: Conv2d.forward takes (N, C, H, W)")


def conv_run_check_channels(c: Int, inc: Int) raises:
    if c != inc:
        raise Error("mojolearn: input has " + String(c) + " channels, the layer " + String(inc))


def conv_run_check_grad(given: List[Int], expected: List[Int]) raises:
    if not conv_run_same_shape(given, expected):
        raise Error("mojolearn: grad_out shape differs from the convolution output shape")


def conv_run_check_weight(ws: List[Int], outc: Int, cg: Int, k: List[Int]) raises:
    if not conv_run_same_shape(ws, [outc, cg, k[0], k[1]]):
        raise Error("mojolearn: weight_ shape differs from configured convolution")


def conv_run_pad_params(shape: List[Int], pads: List[Int], code: Int, last: Int) -> List[Int]:
    """`Conv2d._pad_params(shape) + [last]` (`last` < 0: none)."""
    var words: List[Int] = [shape[0], shape[1], shape[2], shape[3], pads[0], pads[1], pads[2], pads[3], code]
    if last >= 0:
        words.append(last)
    return words^


def conv_run_mw(dev: Int, words: List[Int]) -> List[Int]:
    """A `_m` operation's integers: the residency word, then the launch words
    (core/xcnn_entry_table.mojo)."""
    var prefixed: List[Int] = [dev]
    for i in range(len(words)):  # small-loop(the launch parameter words)
        prefixed.append(words[i])
    return prefixed^


def conv_run_group_prm(shape: List[Int], channels: Int, groups: Int, g: Int, direction: Int) -> List[Int]:
    """`x_cnn_chan_copy_m`'s integers with its residency word 3 (both resident)."""
    var per = channels // groups
    return [3, shape[0], channels, shape[2] * shape[3], per, g * per, direction]


def conv_run_row_width(shape: List[Int]) -> Int:
    """Words per leading row of a C-ordered array."""
    return xcnn_size(shape) // max(shape[0], 1)


def conv_run_tail_size(shape: List[Int]) -> Int:
    """Product of the shape words after the first (one sample's size)."""
    var size = 1
    for i in range(1, len(shape)):  # driver-loop: over shape words
        size *= shape[i]
    return size


def conv_run_spatial(shape: List[Int]) -> Int:
    """Product of the shape words after the first two (H x W)."""
    var hw = 1
    for i in range(2, len(shape)):  # driver-loop: over spatial shape words
        hw *= shape[i]
    return hw


def conv_run_check_ncl(shape: List[Int]) raises:
    if len(shape) != 3:
        raise Error("mojolearn: Conv1d.forward takes (N, C, L)")


def conv_run_check_ncl_grad(shape: List[Int]) raises:
    if len(shape) != 3:
        raise Error("mojolearn: expected (N, C, L)")


def conv_run_four(shape: List[Int]) -> List[Int]:
    """(N, C, L) as the (N, C, 1, L) image Conv1d runs on."""
    return [shape[0], shape[1], 1, shape[2]]


def conv_run_three(shape: List[Int]) -> List[Int]:
    """(N, C, 1, L) back to (N, C, L) (also (O, I, 1, K) to (O, I, K))."""
    return [shape[0], shape[1], shape[3]]


def conv_run_check_relu(shape: List[Int], grad: List[Int]) raises:
    if not conv_run_same_shape(shape, grad):
        raise Error("ReLU gradient shape differs from its input")


def conv_run_check_block_grad(given: List[Int], expected: List[Int]) raises:
    if not conv_run_same_shape(given, expected):
        raise Error("mojolearn: BasicBlock grad_out shape differs from forward output")


def conv_run_check_identity(size: Int, n2: Int) raises:
    if size != n2:
        raise Error("mojolearn: the identity's shape is not the block's output shape")


# ---------------------------------------------------------------- host words

def conv_run_take_channels(dst: Int, src: Int, shape: List[Int], start: Int, count: Int):
    """One NCHW channel group `[start, start + count)` of `src` (shape
    `shape`) into the contiguous (N, count, H, W) block at `dst`."""
    var s = XcnnF32P(unsafe_from_address=src)
    var d = XcnnF32P(unsafe_from_address=dst)
    var spatial = shape[2] * shape[3]
    for n in range(shape[0]):  # driver-loop: one native contiguous copy per image
        unsafe_memcpy(dest=d + n * count * spatial, src=s + (n * shape[1] + start) * spatial, count=count * spatial)


def conv_run_put_channels(dst: Int, src: Int, shape: List[Int], start: Int, count: Int):
    """The contiguous (N, count, H, W) block at `src` into channels
    `[start, start + count)` of `dst` (shape `shape`)."""
    var s = XcnnF32P(unsafe_from_address=src)
    var d = XcnnF32P(unsafe_from_address=dst)
    var spatial = shape[2] * shape[3]
    for n in range(shape[0]):  # driver-loop: one native contiguous copy per image
        unsafe_memcpy(dest=d + (n * shape[1] + start) * spatial, src=s + n * count * spatial, count=count * spatial)


def conv_run_copy_words(dst: Int, offset: Int, src: Int, count: Int):
    """`count` float32 words from `src` to word `offset` of `dst`."""
    if count:
        unsafe_memcpy(dest=XcnnF32P(unsafe_from_address=dst) + offset, src=XcnnF32P(unsafe_from_address=src),
                      count=count)


# ---------------------------------------------------------------- native entries (GIL released by the boundary)

def conv_run_pad_forward(key: XcnnEntryKey, x: Int, xp: Int, prm: List[Int]) raises:
    _ = xcnn_run_call(key, XC_PAD2D_FORWARD, [x, xp], prm)


def conv_run_pad_backward(key: XcnnEntryKey, dxp: Int, dx: Int, prm: List[Int]) raises:
    _ = xcnn_run_call(key, XC_PAD2D_BACKWARD, [dxp, dx], prm)


def conv_run_pad_m(key: XcnnEntryKey, src: Int, dst: Int, dev: Int, prm: List[Int]) raises:
    """The mixed-residency pad entry; `dev` is its residency word."""
    _ = xcnn_run_call(key, XC_PAD2D_M, [src, dst], conv_run_mw(dev, prm))


def conv_run_forward(key: XcnnEntryKey, x: Int, w: Int, b: Int, y: Int, prm: List[Int]) raises:
    _ = xcnn_run_call(key, XC_CONV2D_FORWARD, [x, w, b, y], prm)


def conv_run_forward_m(key: XcnnEntryKey, x: Int, w: Int, b: Int, y: Int, dev: Int, prm: List[Int]) raises:
    _ = xcnn_run_call(key, XC_CONV2D_FORWARD_M, [x, w, b, y], conv_run_mw(dev, prm))


def conv_run_backward(key: XcnnEntryKey, x: Int, w: Int, g: Int, dx: Int, dw: Int, db: Int,
                      prm: List[Int]) raises:
    _ = xcnn_run_call(key, XC_CONV2D_BACKWARD, [x, w, g, dx, dw, db], prm)


def conv_run_backward_m(key: XcnnEntryKey, x: Int, w: Int, g: Int, dx: Int, dw: Int, db: Int, dev: Int,
                        prm: List[Int]) raises:
    _ = xcnn_run_call(key, XC_CONV2D_BACKWARD_M, [x, w, g, dx, dw, db], conv_run_mw(dev, prm))


def conv_run_chan_copy_m(key: XcnnEntryKey, src: Int, dst: Int, shape: List[Int], channels: Int, groups: Int,
                         g: Int, direction: Int) raises:
    _ = xcnn_run_call(key, XC_CHAN_COPY_M, [src, dst], conv_run_group_prm(shape, channels, groups, g, direction))


def conv_run_map2_m(key: XcnnEntryKey, a: Int, b: Int, y: Int, dev: Int, op: Int, count: Int) raises:
    """`x_cnn_map2_m`: op 0 ReLU of a, 1 ReLU gradient (a, b), 2 a + b."""
    _ = xcnn_run_call(key, XC_MAP2_M, [a, b, y], [dev, op, count])


def conv_run_relu_forward(key: XcnnEntryKey, x: Int, y: Int, count: Int) raises:
    _ = xcnn_run_call(key, XC_RELU_FORWARD, [x, y], [count])


def conv_run_relu_backward(key: XcnnEntryKey, x: Int, g: Int, dx: Int, count: Int) raises:
    _ = xcnn_run_call(key, XC_RELU_BACKWARD, [x, g, dx], [count])


def conv_run_add(key: XcnnEntryKey, a: Int, b: Int, y: Int, count: Int) raises:
    _ = xcnn_run_call(key, XC_ADD, [a, b, y], [count])
