# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Host-pointer surfaces for the GLM section: OLS, Ridge, logistic regression.

**THIS IS THE ENTRY THE PYTHON PACKAGE USES.** `bindings/
_mojolearn_estimators.mojo:198` calls `ols_fit_host` and
`python/mojolearn/linear_model.py` calls that, so everything below is what a
`mojolearn.LinearRegression().fit(X, y)` actually runs.

DEVIATION 527 -- THE GUARD WAS BYPASSED ON EXACTLY THIS PATH
-------------------------------------------------------------
`ols_fit_host` called `lstsq_eig` DIRECTLY. `glm/impl/ols.mojo` exists
because that is not safe: `ols.cuh:112-113` switches away from the
normal-equations solver when `n_cols > n_rows` or `n_cols == 1`, because
`A^T A` is singular by construction in the first case and cuML's own Python
layer refuses the second by name (`linear_regression.pyx:390`). That file's
docstring records the bypass as a defect that was found and closed --
**and it was closed only for the Mojo callers.** The host surface, the one
with a Python user on the other end, still went around it and returned a
plausible vector of garbage from a singular inverse, with no error.

It now goes through `ols_fit_traced` (the same guards and dispatch as
`ols_fit`, carrying the identity card -- DEVIATION 517 below), so the host
surface takes the same dispatch every other caller does.

CORRECTED 2026-09-01. The sentence that stood here said the host surface now
gets "the same REFUSAL every other caller already got" at both shapes. There
is no refusal at either shape any more: `n_cols == 1` takes `lstsq_eig`
(DEVIATION 551) and `n_cols > n_rows` takes `lstsq_min_norm` (DEVIATION 550),
and `glm/impl/ols.mojo`'s docstring carries why. What the gate
`check_ols_host_surface_takes_the_guard` asserts is therefore no longer that
those shapes raise; it is that the host surface takes the same DISPATCH --
that a wide fit through this door lands on the min-norm route and leaves the
min-norm card, which a bypass to `lstsq_eig` cannot do.
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext

from core.gemm import gemv_n
from core.identity_trace import IdentityTrace
from glm.impl.ols import (
    OLS_ALGO_EIG,
    ols_fit_traced,
    ols_fit_weighted_traced,
)
from glm.impl.qn.qn import qn_decision_function, qn_fit_x
from glm.impl.ridge import RIDGE_ALGO_EIG, ridge_fit_traced
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
from checks.numerics import ftz, identical_exp64


def _add_scalar_kernel(
    dst: MutPointer[Float32, MutAnyOrigin], n_in: Int32, value: Float32
):
    """`dst += value`, one thread per element. The intercept epilogue.

    IDENTITY_PATHS row 10, DEVIATION 527. `dst` is the prediction vector a
    caller reads and this add is the last operation performed on it, so it
    is a float SEAM in row 10's sense: the operand comes from `gemv_n` and
    the result leaves the device. A prediction near zero -- an ordinary
    thing for a centered regression -- plus a small intercept is exactly
    where the cancellation lands in the denormal range, and there CUDA
    keeps a number Metal has already flushed. Bitwise inert on an FTZ
    backend, which is why it costs nothing to have.

    Row 9 is NOT reachable here: there is no multiply, so there is no
    contraction to pin, and `identical_mul_add` is deliberately not called
    rather than called-and-inert.
    """
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var v = ftz(dst.unsafe_load(i))
        dst.unsafe_store(i, ftz(v + ftz(value)))


def ols_fit_host(
    ctx: DeviceContext,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    y_ptr: MutPointer[Float32, MutUntrackedOrigin],
    coef_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    n_features: Int,
) raises:
    var x = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
    var y = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var w = ctx.enqueue_create_buffer[DType.float32](n_features)
    var cov = ctx.enqueue_create_buffer[DType.float32](n_features * n_features)
    var q = ctx.enqueue_create_buffer[DType.float32](n_features * n_features)
    var qs = ctx.enqueue_create_buffer[DType.float32](n_features * n_features)
    var s = ctx.enqueue_create_buffer[DType.float32](n_features)
    var ab = ctx.enqueue_create_buffer[DType.float32](n_features)
    var inv = ctx.enqueue_create_buffer[DType.float32](n_features * n_features)
    var xa = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
    var xa2 = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
    ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)
    ctx.enqueue_copy(dst_buf=y, src_ptr=y_ptr)
    ctx.synchronize()
    # THROUGH `olsFit`'s DISPATCH (`ols.cuh:112`), NOT AROUND IT. See the
    # module docstring: this line used to call `lstsq_eig` and that is the
    # DEVIATION 527 defect.
    #
    # AND THROUGH THE TRACED ENTRY (DEVIATION 517, 2026-08-23). This is the
    # path `mojolearn.LinearRegression().fit` takes, and until now it called
    # `ols_fit`, whose trace is constructed DISABLED -- so the one OLS path
    # with a Python user on the other end was the one path that could not
    # leave an identity card, while `glm/ols_trace_main.mojo` carded a path
    # no user takes. `IdentityTrace()` reads `MOJOLEARN_IDENTITY_TRACE` and
    # is off unless it is set, so the shipping behaviour is unchanged; set,
    # `tools/e2u_matrix_fit.py` gets the same `ols.step*` stages the Mojo
    # driver does.
    var trace = IdentityTrace()
    if trace.enabled:
        trace.header(
            String("ols n=") + String(n_rows) + " d=" + String(n_features)
            + " algo=" + String(OLS_ALGO_EIG)
        )
    ols_fit_traced(
        ctx, x, y, w, cov, q, qs, s, ab, inv, xa, xa2,
        n_rows, n_features, trace, OLS_ALGO_EIG,
    )
    var hw = ctx.enqueue_create_host_buffer[DType.float32](n_features)
    ctx.enqueue_copy(dst_ptr=hw.unsafe_ptr(), src_buf=w)
    ctx.synchronize()
    for i in range(n_features):
        coef_ptr.unsafe_store(i, hw.unsafe_ptr().unsafe_load(i))


def ols_fit_weighted_host(
    ctx: DeviceContext,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    y_ptr: MutPointer[Float32, MutUntrackedOrigin],
    weight_ptr: MutPointer[Float32, MutUntrackedOrigin],
    coef_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    n_features: Int,
) raises:
    """`olsFit` with `sample_weight != nullptr` (`ols.cuh:99-110`, `:129-141`),
    through the same dispatch `ols_fit_host` takes.

    NOT YET REACHED FROM PYTHON, AND THAT IS RECORDED RATHER THAN HIDDEN.
    `bindings/_mojolearn_estimators.mojo::ols_fit_binding` takes a fixed
    four-argument shape with `len(params) == 2` and has no slot for a weight
    pointer, and `bindings/` is not this lane's file. Until a weight
    binding exists, `python/mojolearn/linear_model.py` applies the SAME two
    operations on the host in numpy -- `sqrt` and a row multiply, both
    single correctly-rounded float32 operations -- and calls the unweighted
    entry, which it documents. `check_ols_sample_weight_host_rescale_matches
    _device` in `glm/checks/ols_check.mojo` is the gate on those two being
    the same fit, so the Python route is checked against THIS one rather
    than asserted to match it.

    The weights are copied into a device buffer this function owns, so
    `olsFit`'s in-place mutation of them (their documented behaviour) is not
    visible to the caller through this surface.
    """
    var x = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
    var y = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var sw = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var w = ctx.enqueue_create_buffer[DType.float32](n_features)
    var cov = ctx.enqueue_create_buffer[DType.float32](n_features * n_features)
    var q = ctx.enqueue_create_buffer[DType.float32](n_features * n_features)
    var qs = ctx.enqueue_create_buffer[DType.float32](n_features * n_features)
    var s = ctx.enqueue_create_buffer[DType.float32](n_features)
    var ab = ctx.enqueue_create_buffer[DType.float32](n_features)
    var inv = ctx.enqueue_create_buffer[DType.float32](n_features * n_features)
    var xa = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
    var xa2 = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
    ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)
    ctx.enqueue_copy(dst_buf=y, src_ptr=y_ptr)
    ctx.enqueue_copy(dst_buf=sw, src_ptr=weight_ptr)
    ctx.synchronize()
    var trace = IdentityTrace()
    if trace.enabled:
        trace.header(
            String("ols n=") + String(n_rows) + " d=" + String(n_features)
            + " algo=" + String(OLS_ALGO_EIG) + " weighted"
        )
    ols_fit_weighted_traced(
        ctx, x, y, w, cov, q, qs, s, ab, inv, xa, xa2, sw, True,
        n_rows, n_features, trace, OLS_ALGO_EIG,
    )
    var hw = ctx.enqueue_create_host_buffer[DType.float32](n_features)
    ctx.enqueue_copy(dst_ptr=hw.unsafe_ptr(), src_buf=w)
    ctx.synchronize()
    for i in range(n_features):
        coef_ptr.unsafe_store(i, hw.unsafe_ptr().unsafe_load(i))
    _ = hw^
    _ = sw^


def ols_predict_host(
    ctx: DeviceContext,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    coef_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    n_features: Int,
    intercept: Float32,
) raises:
    var x = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
    var coef = ctx.enqueue_create_buffer[DType.float32](n_features)
    var out = ctx.enqueue_create_buffer[DType.float32](n_rows)
    ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)
    ctx.enqueue_copy(dst_buf=coef, src_ptr=coef_ptr)
    ctx.synchronize()
    gemv_n(ctx, out, x, coef, n_rows, n_features)
    # A HOST FLOAT COMPARISON DECIDING A LAUNCH, audited for DEVIATION 527
    # and left as it is. `intercept` is a value the caller hands in, not one
    # this repository computed on the device, so the compare is against a
    # host constant and is the same answer on every host. THE SENTENCE THAT
    # STOOD HERE -- "`ols_fit_host` refuses `fit_intercept`, so on the
    # fitted path it is always exactly 0.0" -- WAS FALSE at the surface
    # (corrected 2026-08-23, DEVIATION 517): the implemented `ols_fit` refuses
    # `fit_intercept`, but `python/mojolearn/linear_model.py` centers X and
    # y ON THE HOST before calling this file and hands a NON-ZERO intercept
    # back in, so `mojolearn.LinearRegression()`'s default takes this
    # branch on every fit. The intercept is a host float64 quantity
    # (exactly-rounded sums, no BLAS; see that file), so the compare is
    # still a function of the inputs alone. What the branch DOES change,
    # and it is the honest residue: `x + 0.0` is `x` for every `x` except
    # `-0.0`, which becomes `+0.0`. That is one sign bit on one value,
    # identical on every vendor, and taking the branch out would launch a
    # kernel over every prediction to achieve it.
    if intercept != Float32(0.0):
        ctx.enqueue_function[_add_scalar_kernel](
            out.unsafe_ptr(), Int32(n_rows), intercept,
            grid_dim=((n_rows + 255) // 256, 1, 1), block_dim=(256, 1, 1),
        )
    var hout = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    ctx.enqueue_copy(dst_ptr=hout.unsafe_ptr(), src_buf=out)
    ctx.synchronize()
    for i in range(n_rows):
        out_ptr.unsafe_store(i, hout.unsafe_ptr().unsafe_load(i))


# ===========================================================================
# RIDGE (DEVIATION 545) -- the entry `mojolearn.Ridge` reaches
# ===========================================================================


def ridge_fit_host(
    ctx: DeviceContext,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    y_ptr: MutPointer[Float32, MutUntrackedOrigin],
    coef_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    n_features: Int,
    alpha: Float32,
) raises:
    """`ridgeFit` through `ridge_fit_traced`'s guards and dispatch, with the
    identity card when `MOJOLEARN_IDENTITY_TRACE` is set (the same shape as
    `ols_fit_host`, DEVIATION 517). `fit_intercept` is the HOST centering
    `python/mojolearn/linear_model.py` does, exactly as for
    `LinearRegression`; the implemented `ridge_fit` sees a centered design and
    `fit_intercept=False`."""
    var x = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
    var y = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var w = ctx.enqueue_create_buffer[DType.float32](n_features)
    ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)
    ctx.enqueue_copy(dst_buf=y, src_ptr=y_ptr)
    ctx.synchronize()
    var trace = IdentityTrace()
    if trace.enabled:
        trace.header(
            String("ridge n=") + String(n_rows) + " d=" + String(n_features)
            + " algo=" + String(RIDGE_ALGO_EIG)
        )
    ridge_fit_traced(ctx, x, y, w, n_rows, n_features, alpha, trace, RIDGE_ALGO_EIG)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](n_features)
    ctx.enqueue_copy(dst_ptr=hw.unsafe_ptr(), src_buf=w)
    ctx.synchronize()
    for i in range(n_features):
        coef_ptr.unsafe_store(i, hw.unsafe_ptr().unsafe_load(i))
    _ = hw^


# ===========================================================================
# LOGISTIC REGRESSION (DEVIATIONS 546-549) -- the entry
# `mojolearn.LogisticRegression` reaches
# ===========================================================================


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


def qn_fit_host(
    ctx: DeviceContext,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
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
    loss: Int = QN_LOSS_LOGISTIC,
    svr_eps: Float64 = 0.0,
) raises -> Int:
    """`qnFit` (`qn.cuh:176-193`) for a dense row-major `X`, through
    `qn_fit_x`'s loss switch. `loss` is QN_LOSS_LOGISTIC (the default) with
    `n_classes == 2`, QN_LOSS_SOFTMAX with `n_classes > 2`, the two SVC
    losses with `n_classes == 2`, or one of the four regression losses
    (squared, absolute, the two SVR) with `n_classes == 1`
    (lane/expose-qn-objectives, 2026-09-20); `svr_eps` is the SVR
    sensitivity, non-negative, and must be 0 for every other loss. An
    unknown id or a mismatched `n_classes` is refused here by name before a
    buffer exists. `coef_ptr` holds `n_targets * (n_features
    + fit_intercept)` floats, cuML's column-major `W` (`w[c + C*j]`, the
    bias column last; `n_targets` is 1 for the logistic loss), zero-
    initialized here as `solvers/qn.pyx:552-554` does (no warm start).
    `info_ptr[0]` receives the final objective, `info_ptr[1]` the
    `OPT_RETCODE`; the return value is `num_iters`. Carries the identity
    card (`qn.*`); the logistic header line is the string it always was."""
    qn_check_loss_args(loss, n_classes, svr_eps)
    var n_targets = 1 if n_classes == 2 else n_classes
    var n_param = (n_features + (1 if fit_intercept else 0)) * n_targets
    var x = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
    var y = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var w = ctx.enqueue_create_buffer[DType.float32](n_param)
    ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)
    ctx.enqueue_copy(dst_buf=y, src_ptr=y_ptr)
    ctx.enqueue_memset(w, Float32(0.0))
    ctx.synchronize()
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
        has_sample_weight, trace, Float32(svr_eps),
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


def qn_decision_function_host(
    ctx: DeviceContext,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    coef_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    n_features: Int,
    fit_intercept: Bool,
    n_classes: Int = 1,
) raises:
    """`qnDecisionFunction` (`qn.cuh:231-243`): `scores = X w + b` on the
    device, the fitted `W` layout in. `n_classes` (lane/logistic-multiclass,
    2026-09-14) is 1 for the binary logistic shape, the default and the
    only value until that day: `n_targets = 1`, `scores` is `n_rows`
    floats, `predict` is `z > 0` and `predict_proba` the sigmoid, both in
    the Python layer. `n_classes > 2` is the softmax shape: `w` is the
    column-major `C x dims` block, `n_targets = C`, and `scores` is
    `n_rows * C` floats, `scores[i*C + c]` (row-major, which IS cuML's
    column-major `z[c + C*i]`, `glm_base.mojo::linear_fwd`); `predict` is
    the row argmax and `predict_proba` the softmax (`qn_softmax_host`)."""
    var n_targets = n_classes if n_classes > 2 else 1
    var n_param = (n_features + (1 if fit_intercept else 0)) * n_targets
    var x = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
    var w = ctx.enqueue_create_buffer[DType.float32](n_param)
    var scores = ctx.enqueue_create_buffer[DType.float32](n_rows * n_targets)
    ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)
    ctx.enqueue_copy(dst_buf=w, src_ptr=coef_ptr)
    ctx.synchronize()
    var pams = QNParams.default()
    pams.loss = QN_LOSS_SOFTMAX if n_targets > 1 else QN_LOSS_LOGISTIC
    pams.fit_intercept = fit_intercept
    qn_decision_function(ctx, pams, x, n_rows, n_features, w, scores, n_targets)
    var hs = ctx.enqueue_create_host_buffer[DType.float32](n_rows * n_targets)
    ctx.enqueue_copy(dst_ptr=hs.unsafe_ptr(), src_buf=scores)
    ctx.synchronize()
    for i in range(n_rows * n_targets):
        out_ptr.unsafe_store(i, hs.unsafe_ptr().unsafe_load(i))
    _ = hs^


def qn_predict_binary_host(
    ctx: DeviceContext,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    coef_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Int64, MutUntrackedOrigin],
    n_rows: Int,
    n_features: Int,
    fit_intercept: Bool,
) raises:
    """Binary `qn_predict`, retaining the score and threshold boundary in
    native code so Python never materializes scalar score/code objects."""
    var n_param = n_features + (1 if fit_intercept else 0)
    var x = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
    var w = ctx.enqueue_create_buffer[DType.float32](n_param)
    var scores = ctx.enqueue_create_buffer[DType.float32](n_rows)
    ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)
    ctx.enqueue_copy(dst_buf=w, src_ptr=coef_ptr)
    ctx.synchronize()
    var pams = QNParams.default()
    pams.loss = QN_LOSS_LOGISTIC
    pams.fit_intercept = fit_intercept
    qn_decision_function(ctx, pams, x, n_rows, n_features, w, scores)
    var hs = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    ctx.enqueue_copy(dst_ptr=hs.unsafe_ptr(), src_buf=scores)
    ctx.synchronize()
    for i in range(n_rows):
        out_ptr.unsafe_store(
            i, Int64(1) if hs.unsafe_ptr().unsafe_load(i) > Float32(0.0)
            else Int64(0),
        )
    _ = hs^


def qn_sigmoid_host(
    scores_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float64, MutUntrackedOrigin],
    n_rows: Int,
):
    """The binary `predict_proba` link, `p = 1 / (1 + exp(-z))`.

    DEVIATION 549. cuML's Python layer computes this in cupy on the device
    in float32 (`logistic_regression.py:612-616`, `cp.exp` on float32
    scores) and stores it into a float64 array. Here it is computed ON THE
    HOST in Float64 through `identical_exp64` -- `portable_exp64` under
    IDENTICAL, the repository's standing rule for a probability link (the
    GBDT lane's Logloss sigmoid, `checks/numerics.mojo`), because each
    host libm rounds double `exp` differently in the last bit and numpy's
    `np.exp` would carry the host's bit into the answer (E2 round 1's
    finding). The output dtype is float64, as cuML's and scikit-learn's
    are. `1 - p` for class 0 is one subtraction."""
    for i in range(n_rows):
        var z = Float64(scores_ptr.unsafe_load(i))
        var p = 1.0 / (1.0 + identical_exp64(-z))
        out_ptr.unsafe_store(2 * i, 1.0 - p)
        out_ptr.unsafe_store(2 * i + 1, p)


def qn_softmax_host(
    scores_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float64, MutUntrackedOrigin],
    n_rows: Int,
    n_classes: Int,
):
    """The multinomial `predict_proba` link (lane/logistic-multiclass,
    2026-09-14), the softmax of each row of the `(n_rows, n_classes)`
    float32 decision function, computed ON THE HOST in Float64 through
    `identical_exp64`, the standing rule for a probability link
    (`qn_sigmoid_host` above, DEVIATION 549; cuML's Python layer takes the
    softmax in cupy float32 on the device, `logistic_regression.py`). Per
    row: `m` is the first maximum under a strict `>` from the first entry
    (the positional rule of `softmax_row_max`, so a tie and a signed zero
    are decided by index, never by a hardware max); `s` is the serial
    ascending sum of `exp(z_c - m)`; `p_c = exp(z_c - m) / s`, one
    correctly rounded division per cell. No `log`, no fused multiply-add
    site. `core/classical_host_predict.mojo::host_qn_softmax` spells the
    same statements for the CPU binding and the gate holds the two to a
    bit."""
    for i in range(n_rows):
        var base = i * n_classes
        var m = Float64(scores_ptr.unsafe_load(base))
        for c in range(1, n_classes):
            var v = Float64(scores_ptr.unsafe_load(base + c))
            if v > m:
                m = v
        var s = 0.0
        for c in range(n_classes):
            var z = Float64(scores_ptr.unsafe_load(base + c))
            s = s + identical_exp64(z - m)
        for c in range(n_classes):
            var z = Float64(scores_ptr.unsafe_load(base + c))
            out_ptr.unsafe_store(base + c, identical_exp64(z - m) / s)
