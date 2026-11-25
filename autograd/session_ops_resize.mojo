# SPDX-License-Identifier: Apache-2.0
# SEQUENTIAL'S RESAMPLING AND REARRANGING LAYERS AS SESSION OPS (lane session-ops-resize, 2026-11-25).
#
# Session codes 150..153 (autograd/session_layer_rules.mojo OPS_LAYER_RESIZE_FIRST; dispatched by
# OpsSession._layer_resize, autograd/ops_api.mojo). Each op calls the Sequential layer's OWN forward function
# (autograd/nn_conv.mojo nn_upsample, nn_pixel_shuffle, nn_pad), with the same arguments the Sequential layer
# derives (autograd/nn_conv_blocks.mojo, layer codes 63..66), so it records the same tape nodes and tags and
# its forward and backward are Sequential's bits by construction. No kernel is added: the launches are
# nn_conv.mojo's (upsample forward 1 / backward 1; reflection or replication pad forward 1 / backward 1; pixel
# (un)shuffle one view, one permute copy, one view, and the permute's backward). Host work here is shape
# metadata and argument words, once per call: no loop over data.
#
# Integer words recorded with each call (the door, providers/python/autograd_ops_frontend_pyapi_door.mojo,
# reads them at UNPACK; a captured Program replays the same words):
#   OPS_INTERPOLATE      [mode (0 nearest, 1 linear, 2 bilinear), align_corners (0, 1), recompute_scale_factor
#                        (0, 1), size kind, size0, size1, scale kind]; floats [scale0, scale1]. A kind is 0 for
#                        none, -1 for one value applied to every spatial axis, else the sequence's length (1, 2).
#   OPS_PIXEL_SHUFFLE    [upscale_factor]
#   OPS_PIXEL_UNSHUFFLE  [downscale_factor]
#   OPS_PAD_MODE         [mode (1 reflect, 2 replicate), then Session.pad's pads: every axis's count before,
#                        then every axis's count after (ONNX order)]
# Modes with no kernel (refused by name): interpolate trilinear / bicubic / area / nearest-exact and any 5-D
# input; pad circular and any 5-D reflection / replication.
from sequence.exec_trait import Exec
from autograd.tape import Tape, Tensor
from autograd.rankn_api import rn_shape
from autograd.nn_conv import nn_pad, nn_upsample, nn_pixel_shuffle, up_scale, NNC_PAD_REFLECT, NNC_PAD_REPLICATE
from autograd.nn_conv_blocks import _up_out
from autograd.session_ops_layer_sabotage import SOL_SABOTAGE_PIXEL_SHUFFLE, SOL_SABOTAGE_PIXEL_SHUFFLE_HOST

comptime OPS_INTERPOLATE = 150
comptime OPS_PIXEL_SHUFFLE = 151
comptime OPS_PIXEL_UNSHUFFLE = 152
comptime OPS_PAD_MODE = 153

# the interpolate mode words
comptime RSZ_NEAREST = 0
comptime RSZ_LINEAR = 1
comptime RSZ_BILINEAR = 2
# the pad mode words (recorded in Programs; mapped to x_cnn's words below, so a Program does not depend on them)
comptime RSZ_PAD_REFLECT = 1
comptime RSZ_PAD_REPLICATE = 2


def _axis_value(kind: Int, axis: Int, spatial: Int, v0: Int, v1: Int, what: String) raises -> Int:
    """The size word of spatial axis `axis` (0 height, 1 width; a 3-D input has only axis 1)."""
    if kind == -1:
        return v0
    if kind != spatial:
        raise Error("ValueError: interpolate: " + what + " must have " + String(spatial) + " value"
                    + (String("s") if spatial > 1 else String("")) + " for a " + String(spatial + 2) + "-D input, got "
                    + String(kind))
    if spatial == 1:
        return v0
    return v0 if axis == 0 else v1


def _axis_scale(kind: Int, axis: Int, spatial: Int, f0: Float32, f1: Float32) raises -> Float32:
    if kind == -1:
        return f0
    if kind != spatial:
        raise Error("ValueError: interpolate: scale_factor must have " + String(spatial) + " value"
                    + (String("s") if spatial > 1 else String("")) + " for a " + String(spatial + 2) + "-D input, got "
                    + String(kind))
    if spatial == 1:
        return f0
    return f0 if axis == 0 else f1


def session_interpolate[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, ints: List[Int], fs: List[Float32],
) raises -> Tensor:
    """torch.nn.functional.interpolate for (N, C, L) ('nearest', 'linear') and (N, C, H, W) ('nearest',
    'bilinear'): the Sequential upsample layer's forward (nn_conv_blocks.mojo, layer 65), the same output size
    floor(input * scale_factor) and the same per-axis source scales (up_scale)."""
    if len(ints) < 7 or len(fs) < 2:
        raise Error("ops: interpolate takes one tensor, 7 integers and 2 floats")
    var mode = ints[0]
    var align = ints[1] == 1
    var recompute = ints[2] == 1
    var size_kind = ints[3]
    var scale_kind = ints[6]
    if mode != RSZ_NEAREST and mode != RSZ_LINEAR and mode != RSZ_BILINEAR:
        raise Error("ValueError: interpolate: unknown mode word " + String(mode))
    if (size_kind == 0) == (scale_kind == 0):
        raise Error("ValueError: interpolate: give exactly one of size or scale_factor")
    if align and mode == RSZ_NEAREST:
        raise Error("ValueError: interpolate: align_corners can only be set with mode 'linear' or 'bilinear'")
    var s = tape.shape_of(x)
    if s.ndim != 3 and s.ndim != 4:
        raise Error("ValueError: interpolate: x must be (N, C, L) or (N, C, H, W), got a " + String(s.ndim)
                    + "-D tensor (5-D inputs and mode 'trilinear' have no kernel)")
    var two = s.ndim == 4
    if mode == RSZ_LINEAR and two:
        raise Error("ValueError: interpolate: mode 'linear' needs a 3-D input (N, C, L); use 'bilinear' for (N, C, H, W)")
    if mode == RSZ_BILINEAR and not two:
        raise Error("ValueError: interpolate: mode 'bilinear' needs a 4-D input (N, C, H, W); use 'linear' for (N, C, L)")
    var nearest = mode == RSZ_NEAREST
    var spatial = 2 if two else 1
    var H = s.d[2] if two else 1
    var W = s.d[s.ndim - 1]
    var OH = 1
    var OW: Int
    var sf_h = Float64(0)
    var sf_w = Float64(0)
    if size_kind != 0:
        OW = _axis_value(size_kind, 1, spatial, ints[4], ints[5], "size")
        if two:
            OH = _axis_value(size_kind, 0, spatial, ints[4], ints[5], "size")
        if OH < 1 or OW < 1:
            raise Error("ValueError: interpolate: size must be positive")
    else:
        var fw = _axis_scale(scale_kind, 1, spatial, fs[0], fs[1])
        var fh = _axis_scale(scale_kind, 0, spatial, fs[0], fs[1]) if two else Float32(1)
        if not (fw > Float32(0)) or not (fh > Float32(0)):
            raise Error("ValueError: interpolate: scale_factor must be positive")
        # PyTorch's floor(input_size * scale_factor) in float64, Sequential's own rule
        OW = _up_out(W, 0, fw)
        if two:
            OH = _up_out(H, 0, fh)
        if OH < 1 or OW < 1:
            raise Error("ValueError: interpolate: scale_factor gives an empty output for this input")
        if not recompute:
            sf_w = Float64(fw)
            sf_h = Float64(fh) if two else Float64(0)
    var sh = up_scale(H, OH, sf_h, align, nearest)
    var sw = up_scale(W, OW, sf_w, align, nearest)
    if two:
        return nn_upsample(tape, ex, x, OH, OW, sh, sw, not nearest, align)
    # a (N, C, L) input as Sequential runs it: a (N, C, 1, L) view, the 2-D kernel, a (N, C, OL) view
    var d4: List[Int] = [s.d[0], s.d[1], 1, W]
    var x4 = tape.view_nd(ex, x, rn_shape(d4, "interpolate"))
    var y = nn_upsample(tape, ex, x4, OH, OW, sh, sw, not nearest, align)
    var d3: List[Int] = [s.d[0], s.d[1], OW]
    return tape.view_nd(ex, y, rn_shape(d3, "interpolate"))


def session_pixel_shuffle[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, ints: List[Int], unshuffle: Bool,
) raises -> Tensor:
    """torch.nn.functional.pixel_shuffle / pixel_unshuffle on (N, C, H, W): the Sequential layer's forward
    (nn_pixel_shuffle, layer 66)."""
    var name = String("pixel_unshuffle") if unshuffle else String("pixel_shuffle")
    var factor = String("downscale_factor") if unshuffle else String("upscale_factor")
    if len(ints) < 1:
        raise Error("ops: " + name + " takes one tensor and its factor")
    if ints[0] < 1:
        raise Error("ValueError: " + name + ": " + factor + " must be a positive integer, got " + String(ints[0]))
    if tape.ndim(x) != 4:
        raise Error("ValueError: " + name + ": x must be (N, C, H, W), got a " + String(tape.ndim(x)) + "-D tensor")
    var r = ints[0]
    if not unshuffle and tape.dim(x, 1) % (r * r) != 0:
        raise Error("ValueError: pixel_shuffle: the channels (" + String(tape.dim(x, 1))
                    + ") must be divisible by upscale_factor squared (" + String(r * r) + ")")
    if unshuffle and (tape.dim(x, 2) % r != 0 or tape.dim(x, 3) % r != 0):
        raise Error("ValueError: pixel_unshuffle: the height and width must be divisible by downscale_factor ("
                    + String(r) + ")")
    comptime if (SOL_SABOTAGE_PIXEL_SHUFFLE and E.IS_DEVICE) or (SOL_SABOTAGE_PIXEL_SHUFFLE_HOST and not E.IS_DEVICE):
        return tape.scale(ex, nn_pixel_shuffle(tape, ex, x, r, unshuffle), Float32(2.0))  # SABOTAGE: twice the values
    return nn_pixel_shuffle(tape, ex, x, r, unshuffle)


def session_pad_mode[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, ints: List[Int],
) raises -> Tensor:
    """Session.pad with mode 'reflect' or 'replicate' on (N, C, L) or (N, C, H, W): the Sequential reflection /
    replication pad layer's forward (nn_pad, layers 63, 64). The pads are Session.pad's (ONNX order); the batch
    and channel axes take no padding."""
    if len(ints) < 1:
        raise Error("ops: pad takes one tensor, its mode and its pads")
    var mode = ints[0]
    if mode != RSZ_PAD_REFLECT and mode != RSZ_PAD_REPLICATE:
        raise Error("ValueError: pad: unknown mode word " + String(mode))
    var name = String("'reflect'") if mode == RSZ_PAD_REFLECT else String("'replicate'")
    var nd = tape.ndim(x)
    if nd != 3 and nd != 4:
        raise Error("ValueError: pad: mode " + name + " pads (N, C, L) or (N, C, H, W) inputs, got a "
                    + String(nd) + "-D tensor (5-D reflection and replication have no kernel)")
    if len(ints) != 1 + 2 * nd:
        raise Error("ValueError: pad: pads must list two counts for each of the tensor's " + String(nd) + " axes")
    # ints[1 + k]: before axis k; ints[1 + nd + k]: after axis k
    if ints[1] != 0 or ints[2] != 0 or ints[1 + nd] != 0 or ints[2 + nd] != 0:
        raise Error("ValueError: pad: mode " + name + " pads only the spatial axes; the batch and channel counts must be 0")
    var left = ints[nd]
    var right = ints[2 * nd]
    var top = ints[3] if nd == 4 else 0
    var bottom = ints[3 + nd] if nd == 4 else 0
    if left < 0 or right < 0 or top < 0 or bottom < 0:
        raise Error("ValueError: pad: mode " + name + " takes non-negative counts")
    if mode == RSZ_PAD_REFLECT:
        var W = tape.dim(x, nd - 1)
        var H = tape.dim(x, 2) if nd == 4 else 1
        if left >= W or right >= W or (nd == 4 and (top >= H or bottom >= H)):
            raise Error("ValueError: pad: a 'reflect' count must be smaller than the axis it pads")
    var kernel_mode = NNC_PAD_REFLECT if mode == RSZ_PAD_REFLECT else NNC_PAD_REPLICATE
    return nn_pad(tape, ex, x, left, right, top, bottom, kernel_mode, Float32(0))
