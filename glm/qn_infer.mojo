# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The QN inference (LogisticRegression / LinearSVC decision, predict) over an
X already on the device (lane a8c-device-x-linear-prep-rfe, 2026-11-09; the
pattern of glm/qn_resident.mojo).

glm/estimator.mojo's host-address entries (`qn_decision_function_device`,
`qn_predict_binary_host`, `qn_predict_multiclass_host`) upload X and call
these; glm/qn_fold.mojo gathers a model-selection fold on the device and
calls the same ones, so a cross-validation score never brings the fold to the
host. This module imports only the QN decision launch, so the base binding
reaches it without the whole estimators module.

`qn_x_status` is the deferred finite test of the uploaded X
(core/admitted_finite_status.mojo contract): the predict doors no longer scan
X on the host; the entry tests the words it uploads and the door refuses on
the returned word.

Numerics: none moved. The same kernels on the same words in the same order
as before the split (the one synchronize between the uploads and the
decision launch is gone: the launch follows the copies on the same stream).
"""
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext
from core.admitted_finite_status import enqueue_finite_status
from glm.impl.qn.qn import qn_decision_function
from glm.impl.linear_model.qn import QN_LOSS_LOGISTIC, QN_LOSS_SOFTMAX, QNParams


def qn_x_status(
    ctx: DeviceContext, x: DeviceBuffer[DType.float32], count: Int, status: Int,
) raises -> DeviceBuffer[DType.int32]:
    """Enqueue the finite test of the `count` resident words of `x` and the
    copy of its word to the host int32 at `status` (0: the door tested X,
    nothing is enqueued). The caller keeps the returned buffer alive until
    its next synchronize, which delivers the word."""
    var dstat = ctx.enqueue_create_buffer[DType.int32](1)
    if status != 0:
        dstat.enqueue_fill(Int32(0))
        if count > 0:
            enqueue_finite_status(
                ctx, x.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin](), count,
                dstat.unsafe_ptr(), 0,
            )
        ctx.enqueue_copy(dst_ptr=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=status), src_buf=dstat)
    return dstat^


def qn_scores_resident(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    coef_ptr: MutPointer[Float32, MutUntrackedOrigin],
    mut scores: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_features: Int,
    fit_intercept: Bool,
    n_targets: Int,
) raises -> DeviceBuffer[DType.float32]:
    """Enqueue `scores = X w + b` (`qn_decision_function`, `n_targets` 1:
    the logistic shape; more: the softmax shape) over the resident `x`, the
    fitted `W` uploaded from `coef_ptr`. No synchronize: the returned weight
    buffer stays alive with the caller until its next one."""
    var n_param = (n_features + (1 if fit_intercept else 0)) * n_targets
    var w = ctx.enqueue_create_buffer[DType.float32](n_param)
    ctx.enqueue_copy(dst_buf=w, src_ptr=coef_ptr)
    var pams = QNParams.default()
    pams.loss = QN_LOSS_SOFTMAX if n_targets > 1 else QN_LOSS_LOGISTIC
    pams.fit_intercept = fit_intercept
    qn_decision_function(ctx, pams, x, n_rows, n_features, w, scores, n_targets)
    return w^


def qn_positive_code_kernel(
    scores: MutPointer[Float32, MutAnyOrigin],
    codes: MutPointer[Int64, MutAnyOrigin],
    n_in: Int32,
):
    """codes[i] = 1 if scores[i] > 0 else 0, decided BY BITS (Metal flushes
    compare operands): sign clear, magnitude nonzero and not a NaN. That is
    the host's `score > 0.0` for every float, subnormals included."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var b = bitcast[DType.uint32](scores.unsafe_load(i))
        var mag = b & UInt32(0x7FFFFFFF)
        var pos = (b >> 31) == UInt32(0) and mag != UInt32(0) and mag <= UInt32(0x7F800000)
        codes.unsafe_store(i, Int64(1) if pos else Int64(0))


@always_inline
def _argmax_key(b: UInt32) -> UInt32:
    """A float's bits as an unsigned key whose integer order is the float
    order of the non-NaN values, `-0.0` and `+0.0` sharing one key (so
    neither is greater than the other, as IEEE `>` says)."""
    var mag = b & UInt32(0x7FFFFFFF)
    if mag == UInt32(0):
        return UInt32(0x80000000)
    if (b >> 31) == UInt32(0):
        return b | UInt32(0x80000000)
    return ~b


def qn_row_argmax_code_kernel(
    scores: MutPointer[Float32, MutAnyOrigin],
    codes: MutPointer[Int64, MutAnyOrigin],
    n_in: Int32,
    n_cls: Int32,
):
    """codes[i] = the first maximum of scores[i*C .. i*C + C), decided BY
    BITS (Metal flushes compare operands): a later column replaces only
    when strictly greater, and a NaN on either side never replaces. That is
    `_labels.argmax_rows` (`argmax_rows_f32_binding`) for every float."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var c_n = Int(n_cls)
        var base = i * c_n
        var best = 0
        var bb = bitcast[DType.uint32](scores.unsafe_load(base))
        var best_nan = (bb & UInt32(0x7FFFFFFF)) > UInt32(0x7F800000)
        var best_key = _argmax_key(bb)
        for c in range(1, c_n):
            var b = bitcast[DType.uint32](scores.unsafe_load(base + c))
            var is_nan = (b & UInt32(0x7FFFFFFF)) > UInt32(0x7F800000)
            var key = _argmax_key(b)
            if (not is_nan) and (not best_nan) and key > best_key:
                best = c
                best_key = key
        codes.unsafe_store(i, Int64(best))


def qn_codes_resident(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    coef_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Int64, MutUntrackedOrigin],
    n_rows: Int,
    n_features: Int,
    fit_intercept: Bool,
    n_classes: Int,
) raises:
    """`qn_predict` over the resident `x`: the int64 class codes into the
    host `out_ptr`, the threshold at 0 (`n_classes < 3`) or the row argmax
    of the softmax scores (`n_classes > 2`) on the device. One synchronize,
    after the codes' download."""
    if n_rows <= 0:
        return
    var n_targets = n_classes if n_classes > 2 else 1
    var scores = ctx.enqueue_create_buffer[DType.float32](n_rows * n_targets)
    var w = qn_scores_resident(ctx, x, coef_ptr, scores, n_rows, n_features, fit_intercept, n_targets)
    var codes = ctx.enqueue_create_buffer[DType.int64](n_rows)
    if n_targets > 1:
        ctx.enqueue_function[qn_row_argmax_code_kernel](
            scores.unsafe_ptr(), codes.unsafe_ptr(), Int32(n_rows), Int32(n_classes),
            grid_dim=((n_rows + 255) // 256, 1, 1), block_dim=(256, 1, 1),
        )
    else:
        ctx.enqueue_function[qn_positive_code_kernel](
            scores.unsafe_ptr(), codes.unsafe_ptr(), Int32(n_rows),
            grid_dim=((n_rows + 255) // 256, 1, 1), block_dim=(256, 1, 1),
        )
    ctx.enqueue_copy(dst_ptr=out_ptr, src_buf=codes)
    ctx.synchronize()
    _ = codes^
    _ = scores^
    _ = w^
