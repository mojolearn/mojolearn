# SPDX-License-Identifier: Apache-2.0
"""RUN half of BatchNorm2d / BatchNorm1d (`bindings/xcnn_bn_native.mojo` is
the UNPACK/PACK boundary; lane doorsplit-cnn-xseq, 2026-11-11).

Typed values only: the momentum factor and counter, the aux words, the shape
words of an activation, the channel checks in the layer's words, and the
host or `_m` batch-normalization entries with the residency flags the door
used. The boundary reads the layer's fields into `BnSettings`, owns every
buffer (addresses and handles arrive as words), releases the GIL around the
calls here, and writes the fields back.
"""
from core.xcnn_entry_table import (
    XcnnEntryKey, XC_BATCHNORM_FORWARD, XC_BATCHNORM_FORWARD_M, XC_BATCHNORM_BACKWARD, XC_BATCHNORM_BACKWARD_M,
)
from x_cnn.door_run import XcnnF32P, xcnn_run_call, xcnn_run_fill_f32, xcnn_size


@fieldwise_init
struct BnSettings(Copyable, Movable):
    """The layer fields one forward reads (UNPACK)."""
    var features: Int
    var eps: Float64
    var training: Bool
    var track: Bool
    var tracked: Int            # num_batches_tracked_ before this forward
    var has_momentum: Bool
    var momentum: Float64


@fieldwise_init
struct BnPrep(Copyable, Movable):
    """`_prep`'s typed answer: batch statistics or running ones, the
    momentum factor, and the counter the boundary stores back."""
    var batch_stats: Bool
    var factor: Float64
    var tracked: Int
    var counted: Bool           # the counter moved (training with tracking)


def bn_run_prep(settings: BnSettings) -> BnPrep:
    """`_prep(n, c, hw, need_running)`'s scalar part: the batch counter moves
    here, the factor is momentum or 1 / tracked."""
    var batch_stats = settings.training or not settings.track
    var factor = 0.0
    var tracked = settings.tracked
    var counted = settings.training and settings.track
    if counted:
        tracked += 1
        factor = settings.momentum if settings.has_momentum else 1.0 / Float64(tracked)
    return BnPrep(batch_stats, factor, tracked, counted)


def bn_run_aux_head(aux: Int, eps: Float64, factor: Float64):
    """The first two aux words: eps and the momentum factor, as float32."""
    var words = XcnnF32P(unsafe_from_address=aux)
    words[0] = Float32(eps)
    words[1] = Float32(factor)


def bn_run_aux_count(c: Int) -> Int:
    """Words of the aux block: eps, factor, then seven per channel."""
    return 2 + 7 * c


def bn_run_untracked_running(running: Int, c: Int):
    """The running block of an untracked layer: zero means (the zeroed host
    block) and unit variances."""
    xcnn_run_fill_f32(running + 4 * c, c, 1.0)


def bn_run_check_channels(c: Int, features: Int) raises:
    if c != features:
        raise Error("mojolearn: input has " + String(c) + " channels, the layer " + String(features))


def bn_run_host_dims(shape: List[Int], features: Int) raises -> List[Int]:
    """`_nchw(x)`: (N, C, HW) of a host input, with the layer's refusals."""
    if len(shape) < 2:
        raise Error("IndexError: tuple index out of range")
    var n = shape[0]
    var c = shape[1]
    bn_run_check_channels(c, features)
    var hw = xcnn_size(shape) // max(n * c, 1) if n * c != 0 else 0
    if len(shape) == 2:
        hw = 1
    return [n, c, hw]


def bn_run_check_saved(shape: List[Int], c: Int) raises:
    """The backward's saved rank-three activation metadata."""
    if len(shape) != 3:
        raise Error("ValueError: batch normalization requires saved rank-three activation metadata")
    if shape[1] != c:
        raise Error("ValueError: batch normalization configuration changed since forward")


def bn_run_check_nonneg(dim: Int) raises:
    if dim < 0:
        raise Error("ValueError: batch normalization saved shape must be nonnegative")


def bn_run_forward(key: XcnnEntryKey, x: Int, y: Int, running: Int, aux: Int, n: Int, c: Int, hw: Int,
                   batch_stats: Bool) raises:
    """The host batch-normalization forward entry."""
    _ = xcnn_run_call(key, XC_BATCHNORM_FORWARD, [x, y, running, aux], [n, c, hw, 1 if batch_stats else 0])


def bn_run_forward_m(key: XcnnEntryKey, x: Int, y: Int, running: Int, aux: Int, dev: Int, n: Int, c: Int,
                     hw: Int, batch_stats: Bool) raises:
    """The mixed-residency forward entry; `dev` is the residency word (1 x
    resident, 2 y resident, 4 running resident, 8 aux resident)."""
    _ = xcnn_run_call(key, XC_BATCHNORM_FORWARD_M, [x, y, running, aux], [dev, n, c, hw, 1 if batch_stats else 0])


def bn_run_backward(key: XcnnEntryKey, x: Int, g: Int, dx: Int, aux: Int, flags: Int, mixed_call: Bool,
                    shape: List[Int], c: Int, stats: Int) raises:
    """The backward entry: `_m` with residency `flags` when any operand is
    resident (`mixed_call`), the host entry otherwise."""
    if mixed_call:
        _ = xcnn_run_call(key, XC_BATCHNORM_BACKWARD_M, [x, g, dx, aux], [flags, shape[0], c, shape[2], stats])
    else:
        _ = xcnn_run_call(key, XC_BATCHNORM_BACKWARD, [x, g, dx, aux], [shape[0], c, shape[2], stats])


def bn_run_flags(dev_x: Bool, dev_g: Bool, dev_dx: Bool) -> Int:
    return (1 if dev_x else 0) | (2 if dev_g else 0) | (4 if dev_dx else 0)


def bn_run_dx_shape_check(given: Int, saved: Int) raises:
    if given != saved:
        raise Error("ValueError: batch normalization gradient must match the saved activation size")
