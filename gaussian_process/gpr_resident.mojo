# ===----------------------------------------------------------------------=== #
# Gaussian process regression: fit and predict from the caller's memory to
# the caller's memory, no host list in between (lane fam2-kernel-gp,
# 2026-10-04).
# ===----------------------------------------------------------------------=== #
"""`gpr_fit` and `gpr_predict` without their host steps.

`gpr_fit_host` / `gpr_predict_host` (estimator.mojo) take and return host
lists: the binding copies X (and for predict the n_train^2 factor) into owned
lists, X and y are walked on one host thread for NaN / infinity, each is
uploaded through a staged copy with its own wait, the fit reads log|K| back
(`chol_logdet`) and then `y . alpha_` in a second wait, the factor and the
dual come back as lists that the binding copies again, and predict counts
the clamped variances in a host loop.

Here the same kernels run in the same order (so every output keeps its
bits; the host column is untouched) and:
  - X, y, the factor and the dual go to the device from the caller's
    memory, enqueued; X and y are scanned for NaN / infinity on the device
    (`core/device_scan.mojo`);
  - log|K| and `y . alpha_` stay on the device and come back in ONE wait
    with the dual;
  - the factor, the dual, the mean, the variance, the standard deviation and
    the clamp flags are copied from the device into the caller's memory;
  - the clamp count is a device kernel (an exact integer count).

No identity trace and no sabotage arm: the card stages and the negative
controls stay on `gpr_fit_host` / `gpr_predict_host`, which the checks call.

Switch (IDENTICAL only, ON by default, off under `MOJOLEARN_IDN_ALL_OFF`):
`-D MOJOLEARN_IDN_GPR_PTR_OFF` clears bit 1 of `gp_idn_caps` and the Python
glue calls `gpr_fit` / `gpr_predict`.
"""

from std.atomic import Atomic
from std.gpu import block_idx, thread_idx
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import GLOBAL_NUMERIC_MODE as _CTX_MODE, NUMERIC_IDENTICAL as _CTX_IDENTICAL
from cholesky.checks.potrf import (
    CHOL_ELEM_TPB,
    CHOL_NB_PINNED,
    CHOL_PANEL_TPB,
    add_jitter,
    chol_default_nb_hint,
    chol_nb_for,
    chol_validate_jitter,
    chol_workspace_floats,
    enqueue_logdet,
    potrf_lower,
)
from cholesky.checks.trsm import CHOL_SOLVE_TPB, cho_solve, trsm_lower
from cholesky.impl.matrix.detail.matrix import copy_vector_from_matrix_diagonal_kernel
from cholesky.logdet_fold import logdet_blocks
from core.device_scan import device_classify_nonfinite, device_first_nonfinite
from core.device_zero import enqueue_fill
from core.identity_trace import IdentityTrace
from gaussian_process.checks.gp_sabotage import GP_SAB_NONE
from gaussian_process.checks.kernels import (
    GP_ELEM_TPB,
    GPKernelSpec,
    gp_kernel_diag,
    gp_kernel_matrix,
    gp_kernel_stack_floats,
    gp_predictive_variance,
    gp_validate_kernel,
)
from gaussian_process.estimator import (
    GP_YDOT_TPB,
    _family_ctx,
    _gp_unnorm,
    _length_scale_table,
    _upload,
    gp_log_marginal_likelihood_value,
    gp_validate_alpha,
    gpr_ydot_fin_kernel,
    gpr_ydot_part_kernel,
)
from gaussian_process.gpc_items import gpc_fold_blocks
from gaussian_process.gpc_ovr import _gpc_upload_checked
from gaussian_process.unnorm import GP_UNNORM_MEAN, GP_UNNORM_STD
from gemm.checks.gemm_identical import (
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
)
from gemm.contract import OP_TN

#: fam2-kernel-gp (2026-10-04), IDENTICAL, ON by default
#: (`-D MOJOLEARN_IDN_GPR_PTR_OFF` restores `gpr_fit` / `gpr_predict`'s host
#: lists, host walks and per-scalar waits).
comptime GPR_IDN_PTR = _CTX_MODE == _CTX_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_GPR_PTR_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)

comptime _GP = MutPointer[Float32, MutAnyOrigin]
comptime _GI = MutPointer[Int32, MutAnyOrigin]


@always_inline
def _fp(buf: DeviceBuffer[DType.float32]) -> _GP:
    return _GP(unsafe_from_address=Int(buf.unsafe_ptr()))


@always_inline
def _ip(buf: DeviceBuffer[DType.int32]) -> _GI:
    return _GI(unsafe_from_address=Int(buf.unsafe_ptr()))


def gp_clamp_count_kernel(clamped: _GI, cnt: _GI, n: Int32):
    """How many predictive variances were clamped at zero (DEVIATION 1760):
    an exact integer count, one atomic add per flagged row (`cnt[0]` starts
    at 0). It was a host loop over the downloaded flags."""
    var t = Int(block_idx.x) * GP_YDOT_TPB + Int(thread_idx.x)
    if t < Int(n):
        if clamped.unsafe_load(t) != Int32(0):
            _ = Atomic[DType.int32].fetch_add(cnt, Int32(1))


def gpr_fit_ptr_device(
    x_addr: Int,
    y_addr: Int,
    n_train: Int,
    n_features: Int,
    kernel: GPKernelSpec,
    alpha: Float32,
    l_addr: Int,
    dual_addr: Int,
    scalars_addr: Int,
) raises -> Int:
    """`gpr_fit_host(x, n_train, n_features, y, kernel, alpha)` with its
    inputs read from and its outputs written to the caller's memory.
    `l_addr` receives the n_train^2 factor, `dual_addr` the n_train dual
    (zeros on a failed fit), `scalars_addr` five float64 (info, nb, logdet,
    ydotalpha, lml). Returns LAPACK's `info` (a RESULT, DEVIATION 1634)."""
    gp_validate_kernel(kernel, n_features)
    gp_validate_alpha(alpha)
    if l_addr == 0 or dual_addr == 0 or scalars_addr == 0:
        raise Error("gpr_fit: an output has a null address")
    if y_addr == 0:
        raise Error("gpr_fit: y has a null address")
    var n = n_train
    var ctx = _family_ctx()
    var dx = _gpc_upload_checked(ctx, x_addr, n_train, n_features, String("X"))
    var dy = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_buf=dy, src_ptr=_GP(unsafe_from_address=y_addr))
    var bad = device_first_nonfinite(ctx, dy, n)
    if bad >= 0:
        var is_nan = device_classify_nonfinite(ctx, dy, bad)
        raise Error(
            "gpr_fit_host: y contains "
            + ("NaN" if is_nan else "infinity")
            + " at index "
            + String(bad)
            + "; refused by name before any model launch (DEVIATION 1768)"
        )
    var dls = _upload(ctx, _length_scale_table(kernel))
    var dk = ctx.enqueue_create_buffer[DType.float32](n * n)
    var dstack = ctx.enqueue_create_buffer[DType.float32](gp_kernel_stack_floats(n, n))
    var trace = IdentityTrace()
    ctx.synchronize()
    gp_kernel_matrix(
        ctx, dk, dx, dx, dls, dstack, n, n, n_features, kernel, True, trace,
        "gp.kernel", GP_ELEM_TPB, GP_SAB_NONE,
    )
    # alpha IS the profile's jitter (DEVIATION 1751): `gpr_fit_host`'s calls.
    chol_validate_jitter(alpha)
    var nb_pin = chol_nb_for(n, CHOL_NB_PINNED)
    var ws = ctx.enqueue_create_buffer[DType.float32](chol_workspace_floats(n, nb_pin))
    var dwork = ctx.enqueue_create_buffer[DType.float32](n + 1)
    add_jitter(ctx, dk, n, alpha, CHOL_ELEM_TPB)
    var run = potrf_lower(ctx, dk, ws, n, trace, chol_default_nb_hint(), CHOL_PANEL_TPB, CHOL_ELEM_TPB)
    # the factor home, before the solve (which only reads it)
    ctx.enqueue_copy(dst_ptr=_GP(unsafe_from_address=l_addr), src_buf=dk)

    var ddual = ctx.enqueue_create_buffer[DType.float32](n)
    var dlparts = ctx.enqueue_create_buffer[DType.float32](2 * logdet_blocks(n))
    var nbf = gpc_fold_blocks(n)
    var dpart = ctx.enqueue_create_buffer[DType.float32](max(nbf, 1))
    var dyd = ctx.enqueue_create_buffer[DType.float32](1)
    var diag = dwork.create_sub_buffer[DType.float32](0, n)
    var scalar = dwork.create_sub_buffer[DType.float32](n, 1)
    var hld = ctx.enqueue_create_host_buffer[DType.float32](1)
    var hyd = ctx.enqueue_create_host_buffer[DType.float32](1)
    var logdet = Float32(0.0)
    var ydotalpha = Float32(0.0)
    var lml = Float32(0.0)
    if run.info == 0:
        # `chol_logdet`'s two launches, the scalar left on the device
        ctx.enqueue_function[copy_vector_from_matrix_diagonal_kernel](
            diag.unsafe_ptr(), dk.unsafe_ptr(), Int32(n), Int32(n),
            grid_dim=((n + CHOL_ELEM_TPB - 1) // CHOL_ELEM_TPB, 1, 1), block_dim=(CHOL_ELEM_TPB, 1, 1),
        )
        enqueue_logdet(ctx, _fp(diag), _fp(dlparts), _fp(scalar), n)
        ctx.enqueue_copy(dst_buf=ddual, src_buf=dy)
        cho_solve(ctx, dk, ddual, n, 1, trace, CHOL_SOLVE_TPB)
        # `_gpr_ydot_device`'s two launches
        if nbf > 0:
            ctx.enqueue_function[gpr_ydot_part_kernel](
                _fp(dy), _fp(ddual), Int32(n), _fp(dpart), Int32(nbf),
                grid_dim=(nbf + GP_YDOT_TPB - 1) // GP_YDOT_TPB, block_dim=GP_YDOT_TPB,
            )
        ctx.enqueue_function[gpr_ydot_fin_kernel](_fp(dpart), Int32(nbf), _fp(dyd), grid_dim=1, block_dim=1)
        ctx.enqueue_copy(dst_ptr=hld.unsafe_ptr(), src_buf=scalar)
        ctx.enqueue_copy(dst_ptr=hyd.unsafe_ptr(), src_buf=dyd)
    else:
        # a failed fit hands back a zero dual beside its nonzero info
        enqueue_fill(ctx, ddual, Float32(0.0))
    ctx.enqueue_copy(dst_ptr=_GP(unsafe_from_address=dual_addr), src_buf=ddual)
    ctx.synchronize()
    if run.info == 0:
        logdet = hld.unsafe_ptr().unsafe_load(0)
        ydotalpha = hyd.unsafe_ptr().unsafe_load(0)
        lml = gp_log_marginal_likelihood_value(ydotalpha, logdet, n_train)
    # info, nb, logdet, ydotalpha, lml; each float32 widens to float64 exactly.
    var sp = MutPointer[Float64, MutAnyOrigin](unsafe_from_address=scalars_addr)
    sp.unsafe_store(0, Float64(run.info))
    sp.unsafe_store(1, Float64(run.nb))
    sp.unsafe_store(2, Float64(logdet))
    sp.unsafe_store(3, Float64(ydotalpha))
    sp.unsafe_store(4, Float64(lml))
    var info = run.info
    _ = diag^
    _ = scalar^
    _ = hld^
    _ = hyd^
    _ = dx^
    _ = dy^
    _ = dls^
    _ = dk^
    _ = dstack^
    _ = ws^
    _ = dwork^
    _ = ddual^
    _ = dlparts^
    _ = dpart^
    _ = dyd^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return info


def gpr_predict_ptr_device(
    xt_addr: Int,
    l_addr: Int,
    dual_addr: Int,
    xs_addr: Int,
    n_train: Int,
    n_features: Int,
    n_star: Int,
    kernel: GPKernelSpec,
    return_std: Bool,
    info: Int,
    unnorm: Bool,
    y_std: Float32,
    y_mean: Float32,
    mean_addr: Int,
    var_addr: Int,
    std_addr: Int,
    clamped_addr: Int,
) raises -> Int:
    """`gpr_predict_host` with the model and the query read from, and the
    outputs written to, the caller's memory. The mean is always written;
    the variance, the standard deviation and the int32 clamp flags only with
    `return_std` (their addresses are not touched otherwise). The factor is
    uploaded only with `return_std`. Returns the clamp count."""
    if info != 0:
        raise Error(
            "gpr_predict_host: refusing to predict from a FAILED fit"
            " (info="
            + String(info)
            + "). The factor's columns from "
            + String(info - 1)
            + " onward are unfinished, and solving against them returns"
            " infinities that look like numbers. DEVIATION 1634"
        )
    if n_star <= 0:
        raise Error("gpr_predict_host: n_star must be positive, got " + String(n_star))
    if n_train <= 0:
        raise Error("gpr_predict_host: n_train must be positive, got " + String(n_train))
    if xt_addr == 0 or dual_addr == 0 or mean_addr == 0:
        raise Error("gpr_predict: a model or output address is null")
    if return_std and (l_addr == 0 or var_addr == 0 or std_addr == 0 or clamped_addr == 0):
        raise Error("gpr_predict: return_std needs the factor and the three variance outputs")
    gp_validate_kernel(kernel, n_features)
    var kss = gp_kernel_diag(kernel)

    var trace = IdentityTrace()
    var ctx = _family_ctx()
    var dxs = _gpc_upload_checked(ctx, xs_addr, n_star, n_features, String("X_star"))
    var dx = ctx.enqueue_create_buffer[DType.float32](n_train * n_features)
    ctx.enqueue_copy(dst_buf=dx, src_ptr=_GP(unsafe_from_address=xt_addr))
    var ddual = ctx.enqueue_create_buffer[DType.float32](n_train)
    ctx.enqueue_copy(dst_buf=ddual, src_ptr=_GP(unsafe_from_address=dual_addr))
    var dl = ctx.enqueue_create_buffer[DType.float32](n_train * n_train if return_std else 1)
    if return_std:
        ctx.enqueue_copy(dst_buf=dl, src_ptr=_GP(unsafe_from_address=l_addr))
    var dls = _upload(ctx, _length_scale_table(kernel))
    var dkcross = ctx.enqueue_create_buffer[DType.float32](n_train * n_star)
    var dstack = ctx.enqueue_create_buffer[DType.float32](gp_kernel_stack_floats(n_train, n_star))
    var dmean = ctx.enqueue_create_buffer[DType.float32](n_star)
    var dws = ctx.enqueue_create_buffer[DType.float32](
        identical_gemm_workspace_max_floats(n_star, 1, n_train)
    )
    var dvar = ctx.enqueue_create_buffer[DType.float32](n_star)
    var dstd = ctx.enqueue_create_buffer[DType.float32](n_star)
    var dclamp = ctx.enqueue_create_buffer[DType.int32](n_star)
    var dcnt = ctx.enqueue_create_buffer[DType.int32](1)
    var hcnt = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.synchronize()

    gp_kernel_matrix(
        ctx, dkcross, dx, dxs, dls, dstack, n_train, n_star, n_features, kernel, False, trace,
        "gp.kcross", GP_ELEM_TPB, GP_SAB_NONE,
    )
    identical_gemm_into(ctx, dmean, dkcross, ddual, dws, n_star, 1, n_train, OP_TN)
    if unnorm:
        _gp_unnorm(ctx, dmean, dmean, n_star, y_std, y_mean, GP_UNNORM_MEAN)
    ctx.enqueue_copy(dst_ptr=_GP(unsafe_from_address=mean_addr), src_buf=dmean)

    var n_clamped = 0
    if return_std:
        trsm_lower(ctx, dl, dkcross, n_train, n_star, trace, "gp.v", CHOL_SOLVE_TPB)
        gp_predictive_variance(
            ctx, dvar, dstd, dclamp, dkcross, n_train, n_star, kss, trace, GP_ELEM_TPB, GP_SAB_NONE
        )
        if unnorm:
            _gp_unnorm(ctx, dvar, dstd, n_star, y_std, y_mean, GP_UNNORM_STD)
        enqueue_fill(ctx, dcnt, Int32(0))
        ctx.enqueue_function[gp_clamp_count_kernel](
            _ip(dclamp), _ip(dcnt), Int32(n_star),
            grid_dim=(n_star + GP_YDOT_TPB - 1) // GP_YDOT_TPB, block_dim=GP_YDOT_TPB,
        )
        ctx.enqueue_copy(dst_ptr=_GP(unsafe_from_address=var_addr), src_buf=dvar)
        ctx.enqueue_copy(dst_ptr=_GP(unsafe_from_address=std_addr), src_buf=dstd)
        ctx.enqueue_copy(dst_ptr=_GI(unsafe_from_address=clamped_addr), src_buf=dclamp)
        ctx.enqueue_copy(dst_ptr=hcnt.unsafe_ptr(), src_buf=dcnt)
        ctx.synchronize()
        n_clamped = Int(hcnt.unsafe_ptr().unsafe_load(0))
    else:
        ctx.synchronize()

    _ = dx^
    _ = dxs^
    _ = dls^
    _ = ddual^
    _ = dl^
    _ = dkcross^
    _ = dstack^
    _ = dmean^
    _ = dws^
    _ = dvar^
    _ = dstd^
    _ = dclamp^
    _ = dcnt^
    _ = hcnt^
    # DEVIATION 1946: the context dies LAST.
    _ = ctx^
    return n_clamped
