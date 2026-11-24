# SPDX-License-Identifier: Apache-2.0
"""The RMS-normalization and embedding layers on training device arrays (lane vdf-linalg2, 2026-11-24).

`training.rms_norm_forward/backward` and `embedding_forward/backward` took host buffers only: a training loop that
kept its activations in `TrainDeviceArray`s (training/dev_tensors.mojo) had to bring them down to call these and
upload the results again. These forms read and write device arrays in place, by handle, through the SAME device bodies
the host-address entries run (training/samba_ops.mojo `samba_*_on`), so a device-resident input gives the host input's
bits. The non-finite refusals are device scans with the host entries' words. Token ids are integers and the training
device arrays are float32: the embedding forms take the ids as host int32 words, uploaded once by view.
IDENTICAL GPU columns only (registered where `IDN_TRAIN_DEV_TENSORS`).
"""
from std.math import isfinite
from max.gpu.host import DeviceBuffer, DeviceContext
from core.device_scan import device_first_nonfinite
from training.dev_tensors import train_dev_view
from training.samba_ops import (
    samba_embedding_forward_on, samba_embedding_backward_on, samba_rms_norm_forward_on, samba_rms_norm_backward_on,
)

comptime _IP = MutPointer[Int32, MutUntrackedOrigin]


def _refuse(ctx: DeviceContext, name: String, mut buf: DeviceBuffer[DType.float32], n: Int) raises:
    """`samba_ops._upload_checked`'s refusal and words, on a resident array."""
    var idx = device_first_nonfinite(ctx, buf, n)
    if idx >= 0:
        raise Error("mojolearn samba ops: non-finite " + name + " at flat index " + String(idx))


def _ids(ctx: DeviceContext, ids: _IP, n: Int) raises -> DeviceBuffer[DType.int32]:
    var buf = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_buf=buf, src_ptr=ids)
    return buf^


def _rms_check(m: Int, dm: Int, eps: Float32) raises:
    if m < 1 or dm < 1:
        raise Error("mojolearn samba ops: rms_norm shape must be positive")
    if not isfinite(eps) or eps < Float32(0.0):
        raise Error("mojolearn samba ops: rms_norm eps must be finite and >= 0")


def train_rms_norm_forward_dev(ctx: DeviceContext, y_h: Int, x_h: Int, w_h: Int, m: Int, dm: Int,
                               eps: Float32) raises -> Int:
    """`samba_rms_norm_forward_host` on device arrays: y (y_h, written,
    m x dm) from x (x_h) and w (w_h, dm). Returns m * dm."""
    _rms_check(m, dm, eps)
    var cells = m * dm
    var x = train_dev_view(ctx, x_h, cells, "rms_norm input")
    var w = train_dev_view(ctx, w_h, dm, "rms_norm weight")
    var y = train_dev_view(ctx, y_h, cells, "rms_norm output")
    _refuse(ctx, "rms_norm input", x, cells)
    _refuse(ctx, "rms_norm weight", w, dm)
    samba_rms_norm_forward_on(ctx, y, x, w, m, dm, eps)
    ctx.synchronize()
    _ = x^
    _ = w^
    _ = y^
    return cells


def train_rms_norm_backward_dev(ctx: DeviceContext, dx_h: Int, dw_h: Int, dy_h: Int, x_h: Int, w_h: Int, m: Int,
                                dm: Int, eps: Float32) raises -> Int:
    """`samba_rms_norm_backward_host` on device arrays: dx (dx_h, written,
    m x dm) and dw (dw_h, written, dm). Returns m * dm."""
    _rms_check(m, dm, eps)
    var cells = m * dm
    var x = train_dev_view(ctx, x_h, cells, "rms_norm input")
    var w = train_dev_view(ctx, w_h, dm, "rms_norm weight")
    var dy = train_dev_view(ctx, dy_h, cells, "rms_norm upstream gradient")
    var dx = train_dev_view(ctx, dx_h, cells, "rms_norm input gradient")
    var dw = train_dev_view(ctx, dw_h, dm, "rms_norm weight gradient")
    _refuse(ctx, "rms_norm input", x, cells)
    _refuse(ctx, "rms_norm weight", w, dm)
    _refuse(ctx, "rms_norm upstream gradient", dy, cells)
    samba_rms_norm_backward_on(ctx, dx, dw, dy, x, w, m, dm, eps)
    ctx.synchronize()
    _ = x^
    _ = w^
    _ = dy^
    _ = dx^
    _ = dw^
    return cells


def train_embedding_forward_dev(ctx: DeviceContext, y_h: Int, w_h: Int, ids: _IP, n_positions: Int, vocab: Int,
                                width: Int) raises -> Int:
    """`samba_embedding_forward_host` with the weight and the output device
    arrays: y (y_h, written, n_positions x width) = w (w_h, vocab x width)
    [ids]. Returns n_positions * width."""
    if n_positions < 1 or vocab < 1 or width < 1:
        raise Error("mojolearn samba ops: embedding shape must be positive")
    var cells = n_positions * width
    var w = train_dev_view(ctx, w_h, vocab * width, "embedding weight")
    var y = train_dev_view(ctx, y_h, cells, "embedding output")
    _refuse(ctx, "embedding weight", w, vocab * width)
    var di = _ids(ctx, ids, n_positions)
    samba_embedding_forward_on(ctx, y, w, di, n_positions, vocab, width)
    ctx.synchronize()
    _ = w^
    _ = y^
    _ = di^
    return cells


def train_embedding_backward_dev(ctx: DeviceContext, dw_h: Int, dy_h: Int, ids: _IP, n_positions: Int, vocab: Int,
                                 width: Int) raises -> Int:
    """`samba_embedding_backward_host` on device arrays: dw (dw_h, written,
    vocab x width, a fresh gradient) from dy (dy_h). Returns vocab * width."""
    if n_positions < 1 or vocab < 1 or width < 1:
        raise Error("mojolearn samba ops: embedding shape must be positive")
    var cells = vocab * width
    var dy = train_dev_view(ctx, dy_h, n_positions * width, "embedding upstream gradient")
    var dw = train_dev_view(ctx, dw_h, cells, "embedding weight gradient")
    _refuse(ctx, "embedding upstream gradient", dy, n_positions * width)
    var di = _ids(ctx, ids, n_positions)
    samba_embedding_backward_on(ctx, dw, dy, di, n_positions, vocab, width)
    ctx.synchronize()
    _ = dy^
    _ = dw^
    _ = di^
    return cells
