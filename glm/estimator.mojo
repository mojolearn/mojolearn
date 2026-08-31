# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Host-pointer surfaces for the GLM section: OLS, Ridge, logistic regression.

**THIS IS THE ENTRY THE PYTHON PACKAGE USES.** `bindings/
_mojolearn_estimators.mojo:198` calls `ols_fit_host` and
`python/mojolearn/linear_model.py` calls that, so everything below is what a
`mojolearn.LinearRegression().fit(X, y)` actually runs.

DEVIATION 527 -- THE GUARD WAS BYPASSED ON EXACTLY THIS PATH
-------------------------------------------------------------
`ols_fit_host` called `lstsq_eig` DIRECTLY. `glm/ported/glm/ols.mojo` exists
because that is not safe: `ols.cuh:112-113` switches away from the
normal-equations solver when `n_cols > n_rows` or `n_cols == 1`, because
`A^T A` is singular by construction in the first case and cuML's own Python
layer refuses the second by name (`linear_regression.pyx:390`). That file's
docstring records the bypass as a defect that was found and closed --
**and it was closed only for the Mojo callers.** The host surface, the one
with a Python user on the other end, still went around it and returned a
plausible vector of garbage from a singular inverse, with no error.

It now goes through `ols_fit_traced` (the same guards and dispatch as
`ols_fit`, carrying the identity card -- DEVIATION 517 below). The refusal
is the same one every other caller already got, and
`check_ols_host_surface_takes_the_guard` asserts it at both shapes rather
than trusting this sentence.
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext

from core.gemm import gemv_n
from core.identity_trace import IdentityTrace
from glm.ported.glm.ols import OLS_ALGO_EIG, ols_fit_traced
from glm.ported.glm.qn.qn import qn_decision_function, qn_fit_x
from glm.ported.glm.ridge import RIDGE_ALGO_EIG, ridge_fit_traced
from glm.ported.linear_model.qn import QN_LOSS_LOGISTIC, QNParams
from mojo_only.numerics import ftz, identical_exp64


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
    # (corrected 2026-08-23, DEVIATION 517): the ported `ols_fit` refuses
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
    `LinearRegression`; the ported `ridge_fit` sees a centered design and
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
) raises -> Int:
    """`qnFit` (`qn.cuh:176-193`) for a dense row-major `X`, through
    `qn_fit_x`'s loss switch (QN_LOSS_LOGISTIC; everything else refused by
    name there). `coef_ptr` holds `n_features + fit_intercept` floats: the
    weights then the bias, cuML's `W` layout, zero-initialized here as
    `solvers/qn.pyx:552-554` does (no warm start). `info_ptr[0]` receives
    the final objective, `info_ptr[1]` the `OPT_RETCODE`; the return value
    is `num_iters`. Carries the identity card (`qn.*`)."""
    var n_param = n_features + (1 if fit_intercept else 0)
    var x = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
    var y = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var w = ctx.enqueue_create_buffer[DType.float32](n_param)
    ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)
    ctx.enqueue_copy(dst_buf=y, src_ptr=y_ptr)
    ctx.enqueue_memset(w, Float32(0.0))
    ctx.synchronize()
    var pams = QNParams.default()
    pams.loss = QN_LOSS_LOGISTIC
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
            String("logistic n=") + String(n_rows) + " d=" + String(n_features)
            + " loss=" + String(QN_LOSS_LOGISTIC)
        )
    var fx = Float32(0.0)
    var iters = 0
    var ret = qn_fit_x(
        ctx, pams, x^, y^, n_rows, n_features, n_classes, w, fx, iters,
        has_sample_weight, trace,
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
) raises:
    """`qnDecisionFunction` (`qn.cuh:231-243`): `scores = X w + b` on the
    device, the fitted `W` layout in. `predict` is `z > 0` and
    `predict_proba` is the sigmoid, both below / in the Python layer."""
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
        out_ptr.unsafe_store(i, hs.unsafe_ptr().unsafe_load(i))
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
    GBDT lane's Logloss sigmoid, `mojo_only/numerics.mojo`), because each
    host libm rounds double `exp` differently in the last bit and numpy's
    `np.exp` would carry the host's bit into the answer (E2 round 1's
    finding). The output dtype is float64, as cuML's and scikit-learn's
    are. `1 - p` for class 0 is one subtraction."""
    for i in range(n_rows):
        var z = Float64(scores_ptr.unsafe_load(i))
        var p = 1.0 / (1.0 + identical_exp64(-z))
        out_ptr.unsafe_store(2 * i, 1.0 - p)
        out_ptr.unsafe_store(2 * i + 1, p)
