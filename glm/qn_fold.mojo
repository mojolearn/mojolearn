# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The QN fit (LogisticRegression, QNRegressor) over a model-selection fold
resident on the device (lane a7c-device-x-linear-trees, 2026-11-09; the
pattern of decomposition/pca_fold.mojo).

`qn_fit_host` (glm/estimator.mojo) uploads X from host words. Under
cross-validation the fold's rows are already resident in model selection's
store (core/msel_device_x.mojo): `qn_fit_fold` gathers them on the device
straight into the solver's X buffer (no host gather, no upload of X), with
the finite and range tests in the same pass, and hands it to
`qn_fit_resident`, which uploads y and the weights, synchronises once,
reads the two status words and runs the same solver. Only GPU bindings
import this file (the store lives on the device).

Numerics: the same rows in the same order as the host gather, the same
kernels on the same words as the host-address path: no bit changes.
"""
from max.gpu.host import DeviceContext
from core.device_zero import enqueue_fill
from core.msel_device_x import msel_device_x_f32
from glm.qn_resident import qn_check_loss_args, qn_fit_resident
from glm.qn_infer import qn_scores_resident, qn_codes_resident
from core.estimators_impl import qn_link_from_device


def qn_fit_fold(
    ctx: DeviceContext,
    base_id: Int,
    base_rows: Int,
    index_id: Int,
    y_ptr: MutPointer[Float32, MutUntrackedOrigin],
    coef_ptr: MutPointer[Float32, MutUntrackedOrigin],
    info_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    n_features: Int,
    n_classes: Int,
    penalty_l1: Float64,
    penalty_l2: Float64,
    grad_tol: Float64,
    change_tol: Float64,
    max_iter: Int,
    linesearch_max_iter: Int,
    lbfgs_memory: Int,
    fit_intercept: Bool,
    penalty_normalized: Bool,
    has_sample_weight: Bool,
    sw_ptr: MutPointer[Float32, MutUntrackedOrigin],
    loss: Int,
    svr_eps: Float64,
    status_ptr: MutPointer[Int32, MutUntrackedOrigin],
) raises -> Int:
    """`qn_fit_host` over fold rows `index_id` of the resident base
    `base_id` (`base_rows` x `n_features`). `ctx` is
    `msel_device_x_context()`. `status_ptr` receives two host words (finite,
    range), both 0 when every gathered word is finite and every index in
    range; otherwise nothing is fitted (returns 0) and the door refuses.
    Returns num_iters."""
    qn_check_loss_args(loss, n_classes, svr_eps)
    var x = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
    var status = ctx.enqueue_create_buffer[DType.int32](2)
    enqueue_fill[DType.int32](ctx, status, Int32(0))
    msel_device_x_f32(ctx, base_id, base_rows, n_features, index_id, n_rows, x, status.unsafe_ptr(), 0)
    # The words come back under qn_fit_resident's one synchronisation.
    ctx.enqueue_copy(dst_ptr=status_ptr, src_buf=status)
    var iters = qn_fit_resident(
        ctx, x^, y_ptr, coef_ptr, info_ptr, n_rows, n_features, n_classes, penalty_l1, penalty_l2,
        grad_tol, change_tol, max_iter, linesearch_max_iter, lbfgs_memory, fit_intercept,
        penalty_normalized, has_sample_weight, sw_ptr, loss, svr_eps, Int(status_ptr), 2,
    )
    _ = status^
    return iters


#: `qn_infer_fold` modes: the scores (float32, `qn_decision_function`'s
#: layout), the int64 class codes (`qn_predict_binary` / `_multiclass`), or
#: the binary64 probabilities (`qn_proba`, optionally their log).
comptime QN_FOLD_DECISION = 0
comptime QN_FOLD_CODES = 1
comptime QN_FOLD_PROBA = 2


def qn_infer_fold(
    ctx: DeviceContext,
    base_id: Int,
    base_rows: Int,
    index_id: Int,
    coef_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_addr: Int,
    n_rows: Int,
    n_features: Int,
    fit_intercept: Bool,
    n_classes: Int,
    mode: Int,
    log_out: Bool,
) raises -> Tuple[Int32, Int32]:
    """LogisticRegression's decision_function / predict / predict_proba over
    fold rows `index_id` of the resident base `base_id` (`base_rows` x
    `n_features`), lane a8c-device-x-linear-prep-rfe: the rows are gathered
    on the device straight into the decision's X buffer (no host gather, no
    upload of X), the finite and range tests in the same pass, then the
    same kernels as the host-address entries (glm/qn_infer.mojo). `ctx` is
    `msel_device_x_context()`. `out_addr` is the host output (`mode`
    QN_FOLD_DECISION: n_rows * n_targets float32; QN_FOLD_CODES: n_rows
    int64; QN_FOLD_PROBA: n_rows * max(n_targets, 2) float64). Returns the
    (finite, range) words, both 0 when the fold was finite and in range
    (the door refuses otherwise; the output is then not read)."""
    if mode != QN_FOLD_DECISION and mode != QN_FOLD_CODES and mode != QN_FOLD_PROBA:
        raise Error("qn_infer_fold: mode must be decision (0), codes (1) or proba (2)")
    if out_addr <= 0:
        raise Error("qn_infer_fold: null output address")
    var n_targets = n_classes if n_classes > 2 else 1
    var x = ctx.enqueue_create_buffer[DType.float32](max(n_rows * n_features, 1))
    var status = ctx.enqueue_create_buffer[DType.int32](2)
    enqueue_fill[DType.int32](ctx, status, Int32(0))
    msel_device_x_f32(ctx, base_id, base_rows, n_features, index_id, n_rows, x, status.unsafe_ptr(), 0)
    var words = ctx.enqueue_create_host_buffer[DType.int32](2)
    # the words come back under the mode's one synchronize below
    ctx.enqueue_copy(dst_buf=words, src_buf=status)
    if mode == QN_FOLD_CODES:
        qn_codes_resident(ctx, x, coef_ptr, MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=out_addr),
                          n_rows, n_features, fit_intercept, n_classes)
        ctx.synchronize()
    else:
        var scores = ctx.enqueue_create_buffer[DType.float32](max(n_rows * n_targets, 1))
        var w = qn_scores_resident(ctx, x, coef_ptr, scores, n_rows, n_features, fit_intercept, n_targets)
        if mode == QN_FOLD_DECISION:
            if n_rows > 0:
                ctx.enqueue_copy(dst_ptr=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=out_addr),
                                 src_buf=scores.create_sub_buffer[DType.float32](0, n_rows * n_targets))
            ctx.synchronize()
        else:
            # qn_link_from_device synchronizes after its download (and
            # returns before any work when there are no rows)
            qn_link_from_device(ctx, scores, MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=out_addr),
                                n_rows, n_targets, log_out)
            ctx.synchronize()
        _ = w^
        _ = scores^
    var finite = words.unsafe_ptr().unsafe_load(0)
    var in_range = words.unsafe_ptr().unsafe_load(1)
    _ = words^
    _ = status^
    _ = x^
    return (finite, in_range)
