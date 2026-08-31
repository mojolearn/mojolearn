# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`qn_fit`, `qn_fit_x`, `qnFit`, `qn_decision_function`, `qn_predict`.

PORT OF `cuml/cpp/src/glm/qn/qn.cuh` at cuML `00094f7`. Partial: the loss
switch (`qn_fit_x`, `qn.cuh:121-173`) ports all eight ids -- `QN_LOSS_
LOGISTIC`, `QN_LOSS_SOFTMAX` (`glm_softmax.mojo`, the multinomial arm),
`QN_LOSS_SQUARED` / `QN_LOSS_ABS` (`glm_linear.mojo`), the four SVM losses
(`glm_svm.mojo`) -- each with their `ASSERT` on `C` and their `n_targets =
is_classification && C == 2 ? 1 : C`, and RAISES by name on `sample_weight`
(`add_sample_weights`, the weighted arm of `getLossAndDZ`). `svr_eps` is
their trailing parameter (default 0). The sparse entries (`qnFitSparse` and
siblings) and `qn_predict`'s argmax arm are not ported. Do not improve.

WHAT `qn_fit` DOES WITH THE PENALTY, `qn.cuh:53-86`, copied:

    l1 = penalty_l1; l2 = penalty_l2          (double -> T)
    if penalty_normalized: l1 /= N; l2 /= N   (T divided by int-as-T)
    if l2 == 0:  GLMWithData<Loss>                   -> qn_minimize
    else:        GLMWithData<RegularizedGLM<Loss, Tikhonov(l2)>> -> qn_minimize

so `penalty='l2', C` from the Python door arrives as `l2 = (1/C) / N` and
the objective is `mean_i logloss_i + (l2 / 2) ||w||^2` -- sklearn's
objective divided by N, with the SAME minimizer. `glm_base.mojo`'s
`GLMWithData` carries both shapes keyed on `l2 == 0`.

`qn_predict` (`:262-283`) for a binary classification: `z > 0 ? 1 : 0` on
the decision function, which is `linearFwd` with the fitted `W`
(`qn_decision_function`, `:216-228`). `predict_proba` is NOT in this file
upstream -- cuML's Python layer computes `1 / (1 + exp(-z))` in cupy
(`logistic_regression.py:612-616`); ours is in `glm/estimator.mojo`.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from glm.derived.glm.qn.glm_base import GLMDims, GLMWithData, linear_fwd
from glm.derived.glm.qn.qn_solvers import qn_minimize
from glm.derived.glm.qn.qn_util import LBFGSParam
from glm.derived.linear_model.qn import (
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


def qn_is_classification(loss: Int) -> Bool:
    """`qn_is_classification(qn_loss_type)`, `qn_util.cuh:119-128`: the
    four classification ids. Belongs in `qn_util.mojo`; placed here under
    the QN-losses lane's file scope (HAND-OFF in `glm/README.md`)."""
    return (
        loss == QN_LOSS_LOGISTIC
        or loss == QN_LOSS_SOFTMAX
        or loss == QN_LOSS_SVC_L1
        or loss == QN_LOSS_SVC_L2
    )


def qn_fit(
    ctx: DeviceContext,
    pams: QNParams,
    mut loss: GLMWithData,
    mut w0: DeviceBuffer[DType.float32],
    mut fx: Float32,
    mut num_iters: Int,
    mut trace: IdentityTrace,
) raises -> Int:
    """`qn_fit`, `qn.cuh:38-87`. `loss.l2` must already be the normalized
    `l2` (this function is where it is computed upstream; `qn_fit_x` below
    builds the objective with it, which is the one reordering here, because
    `GLMWithData` owns the scalar)."""
    var opt_param = LBFGSParam.from_params(pams)
    var l1 = Float32(pams.penalty_l1)
    if pams.penalty_normalized:
        l1 = l1 / Float32(loss.n_rows)
    return qn_minimize(
        ctx, w0, fx, num_iters, loss, l1, opt_param, loss.dims.n_param, trace
    )


def qn_fit_x(
    ctx: DeviceContext,
    pams: QNParams,
    var x: DeviceBuffer[DType.float32],
    var y: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    n_classes: Int,
    mut w0: DeviceBuffer[DType.float32],
    mut fx: Float32,
    mut num_iters: Int,
    has_sample_weight: Bool,
    mut trace: IdentityTrace,
    svr_eps: Float32 = Float32(0.0),
) raises -> Int:
    """`qn_fit_x`, `qn.cuh:89-174`: the loss switch. Every arm is their
    `ASSERT` on `C`, the objective with `n_targets` (`:106`), `qn_fit`, and
    the three closing card stages (`qn.coef`, `qn.n_iter`, `qn.retcode`)."""
    if has_sample_weight:
        raise Error(
            "qn: sample_weight is NOT PORTED (GLMBase::add_sample_weights,"
            " glm_base.cuh:115-122, and the weighted arm of getLossAndDZ);"
            " refused by name. See glm/NOT_IMPLEMENTED.tsv"
        )
    # `qn.cuh:54-59` -- here rather than in `qn_fit` because the objective
    # struct carries it.
    var l2 = Float32(pams.penalty_l2)
    if pams.penalty_normalized:
        l2 = l2 / Float32(n_rows)
    # `qn.cuh:106`: `n_targets = qn_is_classification(loss) && C == 2 ? 1 : C`.
    var n_targets = 1 if (qn_is_classification(pams.loss) and n_classes == 2) else n_classes

    if pams.loss == QN_LOSS_LOGISTIC:
        if n_classes != 2:
            raise Error("qn.h: logistic loss invalid C")
        var dims = GLMDims.make(1, n_cols, pams.fit_intercept)
        var obj = GLMWithData(ctx, x^, y^, n_rows, dims, QN_LOSS_LOGISTIC, l2)
        var ret = qn_fit(ctx, pams, obj, w0, fx, num_iters, trace)
        trace.record_device[DType.float32](ctx, "qn.coef", w0, dims.n_param)
        var ints = List[Int32]()
        ints.append(Int32(num_iters))
        trace.record_list_i32("qn.n_iter", ints)
        var rc = List[Int32]()
        rc.append(Int32(ret))
        trace.record_list_i32("qn.retcode", rc)
        return ret
    elif pams.loss == QN_LOSS_SQUARED:
        if n_classes != 1:
            raise Error("qn.h: squared loss invalid C")
        return _qn_fit_loss(
            ctx, pams, x^, y^, n_rows, n_cols, n_targets, w0, fx, num_iters,
            trace, QN_LOSS_SQUARED, "QN_LOSS_SQUARED", l2, svr_eps,
        )
    elif pams.loss == QN_LOSS_SOFTMAX:
        if not (n_classes > 2):
            raise Error("qn.h: softmax invalid C")
        return _qn_fit_loss(
            ctx, pams, x^, y^, n_rows, n_cols, n_targets, w0, fx, num_iters,
            trace, QN_LOSS_SOFTMAX, "QN_LOSS_SOFTMAX", l2, svr_eps,
        )
    elif pams.loss == QN_LOSS_SVC_L1:
        if n_classes != 2:
            raise Error("qn.h: SVC-L1 loss invalid C")
        return _qn_fit_loss(
            ctx, pams, x^, y^, n_rows, n_cols, n_targets, w0, fx, num_iters,
            trace, QN_LOSS_SVC_L1, "QN_LOSS_SVC_L1", l2, svr_eps,
        )
    elif pams.loss == QN_LOSS_SVC_L2:
        if n_classes != 2:
            raise Error("qn.h: SVC-L2 loss invalid C")
        return _qn_fit_loss(
            ctx, pams, x^, y^, n_rows, n_cols, n_targets, w0, fx, num_iters,
            trace, QN_LOSS_SVC_L2, "QN_LOSS_SVC_L2", l2, svr_eps,
        )
    elif pams.loss == QN_LOSS_SVR_L1:
        if n_classes != 1:
            raise Error("qn.h: SVR-L1 loss invalid C")
        return _qn_fit_loss(
            ctx, pams, x^, y^, n_rows, n_cols, n_targets, w0, fx, num_iters,
            trace, QN_LOSS_SVR_L1, "QN_LOSS_SVR_L1", l2, svr_eps,
        )
    elif pams.loss == QN_LOSS_SVR_L2:
        if n_classes != 1:
            raise Error("qn.h: SVR-L2 loss invalid C")
        return _qn_fit_loss(
            ctx, pams, x^, y^, n_rows, n_cols, n_targets, w0, fx, num_iters,
            trace, QN_LOSS_SVR_L2, "QN_LOSS_SVR_L2", l2, svr_eps,
        )
    elif pams.loss == QN_LOSS_ABS:
        if n_classes != 1:
            raise Error("qn.h: abs loss (L1) invalid C")
        return _qn_fit_loss(
            ctx, pams, x^, y^, n_rows, n_cols, n_targets, w0, fx, num_iters,
            trace, QN_LOSS_ABS, "QN_LOSS_ABS", l2, svr_eps,
        )
    else:
        raise Error(
            "qn.h: unknown loss function type (id = " + String(pams.loss) + ")."
        )


def _qn_fit_loss(
    ctx: DeviceContext,
    pams: QNParams,
    var x: DeviceBuffer[DType.float32],
    var y: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    n_targets: Int,
    mut w0: DeviceBuffer[DType.float32],
    mut fx: Float32,
    mut num_iters: Int,
    mut trace: IdentityTrace,
    loss_id: Int,
    loss_name: StaticString,
    l2: Float32,
    svr_eps: Float32,
) raises -> Int:
    """One `case` of the switch for the seven non-logistic losses: the
    objective at `n_targets`, `qn_fit`, the closing card stages -- the
    logistic arm above character for character, parameterized. The `w0`
    length guard is ours: theirs takes a raw pointer and cannot check it;
    a `w0` sized for one target handed to the softmax arm would be read
    past its end."""
    var dims = GLMDims.make(n_targets, n_cols, pams.fit_intercept)
    if len(w0) < dims.n_param:
        raise Error(
            "qn: " + String(loss_name) + " needs w0 of n_param = C * dims = "
            + String(dims.n_param) + " floats (C = " + String(n_targets)
            + ", dims = " + String(dims.dims) + "), got " + String(len(w0))
        )
    var obj = GLMWithData(ctx, x^, y^, n_rows, dims, loss_id, l2, svr_eps)
    var ret = qn_fit(ctx, pams, obj, w0, fx, num_iters, trace)
    trace.record_device[DType.float32](ctx, "qn.coef", w0, dims.n_param)
    var ints = List[Int32]()
    ints.append(Int32(num_iters))
    trace.record_list_i32("qn.n_iter", ints)
    var rc = List[Int32]()
    rc.append(Int32(ret))
    trace.record_list_i32("qn.retcode", rc)
    return ret


def qn_decision_function(
    ctx: DeviceContext,
    pams: QNParams,
    mut x: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    mut params: DeviceBuffer[DType.float32],
    mut scores: DeviceBuffer[DType.float32],
    n_classes: Int = 1,
) raises:
    """`qn_decision_function`, `qn.cuh:216-228`: `scores = W X^T + b`, with
    `n_targets = qn_is_classification(loss) && C == 2 ? 1 : C` (`:213`),
    so `scores` is `n_targets x N` column-major (`scores[c + C*i]`). The
    default `n_classes = 1` is the one-target shape every existing caller
    (binary logistic, the regressions) wants; the softmax caller passes
    its `C`."""
    var n_targets = 1 if (qn_is_classification(pams.loss) and n_classes == 2) else n_classes
    var dims = GLMDims.make(n_targets, n_cols, pams.fit_intercept)
    var w_weights = ctx.enqueue_create_buffer[DType.float32](n_targets * n_cols)
    ctx.synchronize()
    linear_fwd(ctx, scores, x, params, w_weights, n_rows, dims)
    ctx.synchronize()
    _ = w_weights^
