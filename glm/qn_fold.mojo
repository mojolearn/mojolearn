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
