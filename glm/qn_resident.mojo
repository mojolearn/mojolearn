# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The QN fit over an X already on the device, and the QN entry's loss
checks (lane a7c-device-x-linear-trees, 2026-11-09; moved out of
glm/estimator.mojo, which imports them back, so the fold entry
glm/qn_fold.mojo reaches the solver without the whole estimators module).

Numerics: none moved; the same kernels on the same words.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from core.identity_trace import IdentityTrace
from glm.impl.qn.qn import qn_fit_x
from glm.impl.linear_model.qn import (
    QN_LOSS_ABS,
    QN_LOSS_LOGISTIC,
    QN_LOSS_SOFTMAX,
    QN_LOSS_SQUARED,
    QN_LOSS_SVC_L1,
    QN_LOSS_SVC_L2,
    QN_LOSS_SVR_L1,
    QN_LOSS_SVR_L2,
    QNParams,
)


def qn_loss_name(loss: Int) -> String:
    """The identity card's header word for a loss id."""
    if loss == QN_LOSS_LOGISTIC:
        return "logistic"
    if loss == QN_LOSS_SOFTMAX:
        return "softmax"
    if loss == QN_LOSS_SQUARED:
        return "qn-squared"
    if loss == QN_LOSS_ABS:
        return "qn-absolute"
    if loss == QN_LOSS_SVC_L1:
        return "svc-l1"
    if loss == QN_LOSS_SVC_L2:
        return "svc-l2"
    if loss == QN_LOSS_SVR_L1:
        return "svr-l1"
    return "svr-l2"


def qn_check_loss_args(loss: Int, n_classes: Int, svr_eps: Float64) raises:
    """The entry's refusals: the eight routed ids, the `C` each one's
    `ASSERT` in `qn_fit_x` demands (`qn.cuh:121-173`), and `svr_eps` finite,
    non-negative and nonzero only where the loss reads it."""
    var is_svr = loss == QN_LOSS_SVR_L1 or loss == QN_LOSS_SVR_L2
    var is_regression = is_svr or loss == QN_LOSS_SQUARED or loss == QN_LOSS_ABS
    var is_binary = (
        loss == QN_LOSS_LOGISTIC or loss == QN_LOSS_SVC_L1 or loss == QN_LOSS_SVC_L2
    )
    if not (is_regression or is_binary or loss == QN_LOSS_SOFTMAX):
        raise Error(
            "qn_fit: loss " + String(loss) + " is not a qn_loss_type id;"
            " 0 to 7 are (glm/impl/linear_model/qn.mojo)"
        )
    if is_regression and n_classes != 1:
        raise Error(
            "qn_fit: loss " + String(loss) + " is a regression loss and needs"
            " n_classes == 1, got " + String(n_classes)
        )
    if is_binary and n_classes != 2:
        raise Error(
            "qn_fit: loss " + String(loss) + " needs n_classes == 2, got "
            + String(n_classes)
        )
    if loss == QN_LOSS_SOFTMAX and not (n_classes > 2):
        raise Error("qn_fit: the softmax loss needs n_classes > 2, got " + String(n_classes))
    if not (svr_eps >= 0.0) or svr_eps > 3.0e38:
        raise Error("qn_fit: svr_eps must be finite and non-negative, got " + String(svr_eps))
    if svr_eps != 0.0 and not is_svr:
        raise Error(
            "qn_fit: svr_eps is read by the two SVR losses only; loss "
            + String(loss) + " must be given 0"
        )


def qn_fit_resident(
    ctx: DeviceContext,
    var x: DeviceBuffer[DType.float32],
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
    status: Int,
    n_status: Int,
) raises -> Int:
    """`qn_fit_host` over an X already on the device (`x`, n_rows x
    n_features, C order: its upload, or a model-selection fold gathered by
    `qn_fit_fold`; lane a7c): uploads y and the weights, ONE synchronize,
    then `n_status` int32 status words at host address `status` (0: none)
    are read: any nonzero word fits nothing (returns 0, coef_ptr untouched).
    Otherwise the solver as before; the same kernels on the same words."""
    var n_targets = 1 if n_classes == 2 else n_classes
    var n_param = (n_features + (1 if fit_intercept else 0)) * n_targets
    var y = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var w = ctx.enqueue_create_buffer[DType.float32](n_param)
    ctx.enqueue_copy(dst_buf=y, src_ptr=y_ptr)
    ctx.enqueue_memset(w, Float32(0.0))
    var sw = Optional[DeviceBuffer[DType.float32]]()
    if has_sample_weight:
        var swb = ctx.enqueue_create_buffer[DType.float32](n_rows)
        ctx.enqueue_copy(dst_buf=swb, src_ptr=sw_ptr)
        sw = Optional[DeviceBuffer[DType.float32]](swb^)
    ctx.synchronize()
    if status != 0:
        var words = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=status)
        for k in range(n_status):  # small-loop(status words): at most two
            if words.unsafe_load(k) != 0:
                return 0
    var pams = QNParams.default()
    pams.loss = loss
    pams.penalty_l1 = penalty_l1
    pams.penalty_l2 = penalty_l2
    pams.grad_tol = grad_tol
    pams.change_tol = change_tol
    pams.max_iter = max_iter
    pams.linesearch_max_iter = linesearch_max_iter
    pams.lbfgs_memory = lbfgs_memory
    pams.fit_intercept = fit_intercept
    pams.penalty_normalized = penalty_normalized
    var trace = IdentityTrace()
    if trace.enabled:
        trace.header(
            qn_loss_name(loss) + " n=" + String(n_rows) + " d=" + String(n_features)
            + " loss=" + String(loss)
        )
    var fx = Float32(0.0)
    var iters = 0
    var ret = qn_fit_x(
        ctx, pams, x^, y^, n_rows, n_features, n_classes, w, fx, iters,
        has_sample_weight, trace, Float32(svr_eps), sw^,
    )
    var hw = ctx.enqueue_create_host_buffer[DType.float32](n_param)
    ctx.enqueue_copy(dst_ptr=hw.unsafe_ptr(), src_buf=w)
    ctx.synchronize()
    for i in range(n_param):
        coef_ptr.unsafe_store(i, hw.unsafe_ptr().unsafe_load(i))
    info_ptr.unsafe_store(0, fx)
    info_ptr.unsafe_store(1, Float32(ret))
    _ = hw^
    return iters
