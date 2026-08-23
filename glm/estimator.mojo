"""Host-pointer surface for normal-equations ordinary least squares.

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
from mojo_only.numerics import ftz


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
