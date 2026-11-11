# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Host-pointer surface for binary Gaussian process CLASSIFICATION on the
GPU: the Laplace approximation of scikit-learn 1.9.0 `_gpc.py`
(`_BinaryGaussianProcessClassifierLaplace`), what `bindings/_mojolearn_gp.mojo`
calls and `python/mojolearn/_gpc_impl.py` exposes as
`mojolearn.GaussianProcessClassifier`. One-vs-rest past two classes is the
surface's loop over this entry (DEVIATION 2833).

WHAT RUNS WHERE. The device runs what the regressor's device path runs, and
nothing vendor specific: the kernel matrix and the cross-covariance
(`gaussian_process/checks/kernels.mojo::gp_kernel_matrix`), the Cholesky
factor and solve of `B` (`cholesky/estimator.mojo`), the two
matrix-vector products per Newton iteration and the latent mean
(`gemm/checks/gemm_identical.mojo::identical_gemm_into` at `OP_TN`), and the
triangular solve for the latent variance (`cholesky/checks/trsm.mojo::
trsm_lower`). The per-row steps (the sigmoid, the weights, `B`'s cells, the
Newton right-hand side, the likelihood and the variance fold) are
`gaussian_process/gpc_items.mojo` kernels, the float64 probability and the
one-vs-rest combine are `gaussian_process/gpc_proba64.mojo` kernels in
software binary64, and the stop test is `gaussian_process/gpc_common.mojo`;
the CPU host oracle (`gaussian_process/host/gpc_steps.mojo`) compiles the
same statements. DEVIATIONS 2830 (the stop rule), 2831 (the pinned float32
orders) and 2832 (the float64 probability and its erf) are written there.

NO OPTIMIZER. As for the regressor (DEVIATION 1761), the kernel is the
kernel passed in; `optimizer`, `n_restarts_optimizer` and `warm_start` are
refused by name in Python before a binding is reached. DEVIATION 1766, the
refusal of classification itself, is closed by DEVIATION 2830.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu import block_idx, thread_idx
from checks.numerics import NUMERIC_FAST as _NUMERIC_FAST
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE as _CTX_MODE, NUMERIC_IDENTICAL as _CTX_IDENTICAL
from core.neural_context import neural_ctx
from std.time import perf_counter_ns
from std.memory import unsafe_memcpy
from std.os import getenv
from std.sys.compile import is_defined
# ONE PROCESS-LIFETIME DeviceContext per binding and tier (CURRENT DIRECTIVES;
# lane/neighbors-apple 2026-09-28): a new context per entry is a new Metal
# queue and a pipeline load per call. Same kernels, same launches, same order
# on one stream, and every entry still synchronizes before it returns, so no
# bit moves. This module is compiled into ONE GPU binding, so the slot name
# (per module and tier) is that binding's own.
comptime _FAMILY_CTX = "MojolearnGpContextIdenticalV3" if _CTX_MODE == _CTX_IDENTICAL else "MojolearnGpContextOtherV3"


def _family_ctx() raises -> DeviceContext:
    """The binding's process-lifetime context; `-D MOJOLEARN_FAMILY_CTX_PER_CALL`
    restores a new context per entry (the A/B arm)."""
    comptime if is_defined["MOJOLEARN_FAMILY_CTX_PER_CALL"]():
        return DeviceContext()
    return neural_ctx[_FAMILY_CTX]()

from cholesky.checks.trsm import CHOL_SOLVE_TPB, trsm_lower
from core.identity_trace import IdentityTrace
from gaussian_process.checks.gp_sabotage import GP_SAB_NONE
from gaussian_process.checks.kernels import (
    GP_ELEM_TPB,
    GPKernelSpec,
    gp_kernel_diag,
    gp_kernel_matrix,
    gp_kernel_stack_floats,
    gp_validate_kernel,
)
from gaussian_process.estimator import (
    _download,
    _length_scale_table,
    _upload,
    gp_validate_data,
)
# Keep device-module extension imports at module scope: importing the resident
# module inside a function duplicates DeviceContext enqueue overload candidates.
from gaussian_process.gpc_resident_k import _gpc_kernel_self_dev
from gaussian_process.gpc_device_var import (
    GPC_VAR_TPB,
    gpc_latent_var_launch,
    gpc_scale_rows_kernel,
)
from gaussian_process.gpc_common import (
    GPCBinaryFit,
    GPCLatent,
    gpc_validate_labels,
    gpc_validate_max_iter,
)
from gaussian_process.gpc_proba64 import gpc_ovr_combine_row, gpc_pi_star_sf64
from gaussian_process.gpc_items import gpc_residual_item
from gaussian_process.gpc_newton import gpc_newton_device
from gemm.checks.gemm_identical import (
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
)
from gemm.contract import OP_TN


#: lane/neighbors-apple (2026-09-28): the Laplace Newton loop keeps K, B and
#: the factor on the device -- B = I + W_sr K W_sr built by a kernel with
#: `gpc_b_matrix`'s arithmetic (the weights multiplied first, each product
#: `identical_mul` and flushed, the diagonal's `1 + cell`), factored, its
#: log-determinant taken and solved against where it lies -- instead of a
#: host B, an upload, a download of L and a second upload of L per step, and
#: K uploaded twice per step for the products. `cholesky_factor_host`'s
#: statements (jitter 0, `potrf_lower`, `chol_logdet`) and
#: `cholesky_solve_host`'s (`cho_solve`) in their order; its host
#: validation cannot refuse here (K is a validated finite kernel matrix and
#: B is symmetric by construction). The last factor is read back once.
#: The opt-in host round trips (`-D MOJOLEARN_GPC_HOST_NEWTON`) were removed
#: (hr-optin-flags). Lane n8-gp-krr-smo (2026-11-09): the loop is
#: gaussian_process/gpc_newton.mojo's bounded device loop over the executor's
#: Cholesky ops (GPC_NEWTON_TPB is this file's old GPC_B_TPB, 256).


#: lane neighbors-apple3 (2026-09-28), the default on every column since
#: cpu-gpu-cleanup c-gp-kernel (2026-10-02; was a FAST-on-Apple opt-in): the
#: latent variance stays on the device. `gpc_scale_rows` and `gpc_latent_var` as
#: kernels (their statements, the fold over i ascending, one thread per
#: column), so the n_train x n_star cross covariance is not read back,
#: scaled on the host, uploaded, read back again and folded on the host.
#: The Newton loop takes K from the kernel launch's own buffer
#: (`_gpc_kernel_self_dev`), not from a download and an upload.


# TOMBSTONE: the Newton loop's kernels (gpc_b_matrix_kernel, gpc_w_kernel, gpc_rhs_kernel, gpc_scale_nan_kernel, gpc_a_kernel, gpc_lml_part_kernel, gpc_lml_fin_kernel) deleted 2026-11-09 by lane/n8-gp-krr-smo; code recoverable at f59d913b0b.
# Restore: git apply experiments/removed/N8_GPC_NEWTON_ONE_LOOP.patch
# Their gated forms are gaussian_process/gpc_newton.mojo's.


comptime _GP = MutPointer[Float32, MutAnyOrigin]
comptime _GI = MutPointer[Int32, MutAnyOrigin]
comptime GPC_STEP_TPB = 256


@always_inline
def _gtid() -> Int:
    return Int(block_idx.x) * GPC_STEP_TPB + Int(thread_idx.x)


@always_inline
def _gblocks(count: Int) -> Int:
    return (count + GPC_STEP_TPB - 1) // GPC_STEP_TPB if count > 0 else 1


@always_inline
def _dp(buf: DeviceBuffer[DType.float32]) -> _GP:
    return _GP(unsafe_from_address=Int(buf.unsafe_ptr()))


def gpc_residual_kernel(y: _GP, pi: _GP, dst: _GP, n: Int32):
    var i = _gtid()
    if i < Int(n):
        gpc_residual_item(i, y, pi, dst)


comptime _GU = MutPointer[UInt64, MutAnyOrigin]


def gpc_proba_kernel(mean: _GP, variance: _GP, dst: _GU, n: Int32):
    """DEVIATION 2832's class-1 probability, one query row per thread, as
    the float64 word (`gpc_proba64.mojo::gpc_pi_star_sf64`, the host
    column's `gpc_pi_star` in software binary64)."""
    var t = _gtid()
    if t < Int(n):
        dst.unsafe_store(t, gpc_pi_star_sf64(mean.unsafe_load(t), variance.unsafe_load(t)))


def gpc_ovr_combine_kernel(cols: _GU, dst: _GU, codes: _GI, n: Int32, k: Int32):
    """DEVIATION 2833's one-vs-rest normalization and argmax, one query row
    per thread (`gpc_proba64.mojo::gpc_ovr_combine_row`)."""
    var t = _gtid()
    if t < Int(n):
        gpc_ovr_combine_row(cols, dst, codes, t, Int(n), Int(k))


def gpc_ovr_combine_host(col_addrs: List[Int], out_addr: Int, codes_addr: Int, n: Int) raises:
    """The one-vs-rest combine of k = len(col_addrs) class columns on the
    device. Column c's address holds the n float64 class-1 probabilities of
    class c (staged class-major, `cols[c * n + t]`); `out_addr` receives the
    normalized rows row-major (n * k float64) and `codes_addr` the n int32
    argmax codes."""
    var k = len(col_addrs)
    if n <= 0 or k <= 0:
        raise Error(
            "gpc_ovr_combine: n and k must be positive, got n="
            + String(n)
            + " k="
            + String(k)
        )
    var ctx = _family_ctx()
    var hin = ctx.enqueue_create_host_buffer[DType.uint64](n * k)
    for c in range(k):
        unsafe_memcpy(dest=hin.unsafe_ptr() + c * n, src=_GU(unsafe_from_address=col_addrs[c]), count=n)
    var din = ctx.enqueue_create_buffer[DType.uint64](n * k)
    ctx.enqueue_copy(dst_buf=din, src_ptr=hin.unsafe_ptr())
    var dout = ctx.enqueue_create_buffer[DType.uint64](n * k)
    var dcodes = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_function[gpc_ovr_combine_kernel](
        _GU(unsafe_from_address=Int(din.unsafe_ptr())),
        _GU(unsafe_from_address=Int(dout.unsafe_ptr())),
        _GI(unsafe_from_address=Int(dcodes.unsafe_ptr())),
        Int32(n),
        Int32(k),
        grid_dim=_gblocks(n),
        block_dim=GPC_STEP_TPB,
    )
    var hout = ctx.enqueue_create_host_buffer[DType.uint64](n * k)
    var hcodes = ctx.enqueue_create_host_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_ptr=hout.unsafe_ptr(), src_buf=dout)
    ctx.enqueue_copy(dst_ptr=hcodes.unsafe_ptr(), src_buf=dcodes)
    ctx.synchronize()
    unsafe_memcpy(dest=_GU(unsafe_from_address=out_addr), src=hout.unsafe_ptr(), count=n * k)
    unsafe_memcpy(dest=_GI(unsafe_from_address=codes_addr), src=hcodes.unsafe_ptr(), count=n)
    _ = hin^
    _ = din^
    _ = dout^
    _ = dcodes^
    _ = hout^
    _ = hcodes^


def _gpc_fit_binary_device(
    x: List[Float32],
    n_train: Int,
    n_features: Int,
    y: List[Float32],
    kernel: GPKernelSpec,
    max_iter_predict: Int,
) raises -> GPCBinaryFit:
    """The Laplace Newton loop with every step on the device: K from the
    kernel launch's own buffer, then `gpc_newton_device`
    (gaussian_process/gpc_newton.mojo, the one Newton loop the one-vs-rest
    fit runs too: a bounded device loop over the executor's Cholesky ops,
    the register block read once per GPC_NEWTON_CHECK iterations). The last
    factor, pi and W_sr are read back once."""
    var st_on = getenv("MOJOLEARN_STAGE_TIMES") == "1"
    var t_k0 = Int(perf_counter_ns())
    var n = n_train
    var ctx = _family_ctx()
    var dk = _gpc_kernel_self_dev(ctx, x, n_train, n_features, kernel)
    var t_kernel = Int(perf_counter_ns()) - t_k0
    var db = ctx.enqueue_create_buffer[DType.float32](n * n)
    var dy = _upload(ctx, y)
    var dpi = ctx.enqueue_create_buffer[DType.float32](n)
    var dwsr = ctx.enqueue_create_buffer[DType.float32](n)
    var dgw = ctx.enqueue_create_buffer[DType.float32](identical_gemm_workspace_max_floats(n, 1, n))
    var s0 = Int(perf_counter_ns())
    var run = gpc_newton_device(ctx, dk, dy, db, dgw, dpi, dwsr, n, max_iter_predict)
    var t_f = Int(perf_counter_ns()) - s0
    var t_l0 = Int(perf_counter_ns())
    var last_l = _download(ctx, db, n * n)
    var last_pi = _download(ctx, dpi, n)
    var last_wsr = _download(ctx, dwsr, n)
    if st_on:
        print("GPC_FIT_DEVICE_STAGES iters=" + String(run.n_iter) + " kernel_ms=" + String(t_kernel // 1000000)
              + " newton_ms=" + String(t_f // 1000000)
              + " read_ms=" + String((Int(perf_counter_ns()) - t_l0) // 1000000))
    _ = dk^
    _ = db^
    _ = dy^
    _ = dpi^
    _ = dwsr^
    _ = dgw^
    return GPCBinaryFit(last_l^, last_pi^, last_wsr^, run.lml, run.n_iter, run.nb)


def gpc_fit_binary_host(
    x: List[Float32],
    n_train: Int,
    n_features: Int,
    y: List[Float32],
    kernel: GPKernelSpec,
    max_iter_predict: Int,
) raises -> GPCBinaryFit:
    """`_BinaryGaussianProcessClassifierLaplace(kernel, optimizer=None).fit`
    with `y` already encoded to 0 and 1 (`_gpc.py:171-267`): the kernel
    matrix, then `_posterior_mode` from `f = 0` (`_gpc.py:461`, warm_start
    refused). Refuses by name: non-finite X, a kernel the constructors
    refuse, a target outside {0, 1} or a single class, max_iter_predict
    below 1, and a factorization of `B` that fails (it cannot for finite
    inputs, `B`'s eigenvalues are at least 1)."""
    gp_validate_data(x, n_train, n_features, String("X"))
    gpc_validate_labels(y, n_train)
    gp_validate_kernel(kernel, n_features)
    gpc_validate_max_iter(max_iter_predict)

    return _gpc_fit_binary_device(x, n_train, n_features, y, kernel, max_iter_predict)


def gpc_validate_model(
    y: List[Float32],
    pi: List[Float32],
    wsr: List[Float32],
    l: List[Float32],
    n_train: Int,
) raises:
    """The saved arrays a prediction reads, sized against `n_train`."""
    gpc_validate_labels(y, n_train)
    if len(pi) != n_train or len(wsr) != n_train or len(l) != n_train * n_train:
        raise Error(
            "gpc_predict_host: the fitted arrays hold pi "
            + String(len(pi))
            + ", W_sr "
            + String(len(wsr))
            + " and L "
            + String(len(l))
            + " values; n_train "
            + String(n_train)
            + " needs n_train, n_train and n_train^2"
        )


def gpc_predict_binary_host(
    x_train: List[Float32],
    y: List[Float32],
    pi: List[Float32],
    wsr: List[Float32],
    l: List[Float32],
    n_train: Int,
    n_features: Int,
    kernel: GPKernelSpec,
    x_star: List[Float32],
    n_star: Int,
    want_variance: Bool,
) raises -> GPCLatent:
    """`latent_mean_and_variance(X_star)` (`_gpc.py:434-441`); the mean
    alone when `want_variance` is False (`predict`, `_gpc.py:287-290`).

        K_star = kernel(X_train, X_star)          n_train x n_star
        mean   = K_star^T (y - pi)                gemm at OP_TN
        v      = solve(L, W_sr[:, None] K_star)   trsm_lower in place
        var    = kernel.diag(X_star) - sum_i v_i^2
    """
    if n_star <= 0:
        raise Error(
            "gpc_predict_host: n_star must be positive, got " + String(n_star)
        )
    gp_validate_data(x_train, n_train, n_features, String("X_train"))
    gp_validate_data(x_star, n_star, n_features, String("X_star"))
    gp_validate_kernel(kernel, n_features)
    gpc_validate_model(y, pi, wsr, l, n_train)
    var kss = gp_kernel_diag(kernel)

    var trace = IdentityTrace()
    var ctx = _family_ctx()
    var dx = _upload(ctx, x_train)
    var dxs = _upload(ctx, x_star)
    var dls = _upload(ctx, _length_scale_table(kernel))
    # y - pi on the device (gpc_items.gpc_residual_item, the host column's)
    var dyt = _upload(ctx, y)
    var dpit = _upload(ctx, pi)
    var dr = ctx.enqueue_create_buffer[DType.float32](n_train)
    ctx.enqueue_function[gpc_residual_kernel](
        _dp(dyt), _dp(dpit), _dp(dr), Int32(n_train), grid_dim=_gblocks(n_train), block_dim=GPC_STEP_TPB
    )
    var dkc = ctx.enqueue_create_buffer[DType.float32](n_train * n_star)
    var dstack = ctx.enqueue_create_buffer[DType.float32](
        gp_kernel_stack_floats(n_train, n_star)
    )
    var dmean = ctx.enqueue_create_buffer[DType.float32](n_star)
    var dws = ctx.enqueue_create_buffer[DType.float32](
        identical_gemm_workspace_max_floats(n_star, 1, n_train)
    )
    ctx.synchronize()
    gp_kernel_matrix(
        ctx,
        dkc,
        dx,
        dxs,
        dls,
        dstack,
        n_train,
        n_star,
        n_features,
        kernel,
        False,
        trace,
        "gpc.kcross",
        GP_ELEM_TPB,
        GP_SAB_NONE,
    )
    identical_gemm_into(ctx, dmean, dkc, dr, dws, n_star, 1, n_train, OP_TN)
    var mean = _download(ctx, dmean, n_star)
    var variance = List[Float32]()
    var proba = List[Float64]()
    var st_on = getenv("MOJOLEARN_STAGE_TIMES") == "1"
    var t_v0 = Int(perf_counter_ns())
    if want_variance:
        var dwv = _upload(ctx, wsr)
        var dv2 = ctx.enqueue_create_buffer[DType.float32](n_train * n_star)
        var dvar = ctx.enqueue_create_buffer[DType.float32](n_star)
        ctx.enqueue_function[gpc_scale_rows_kernel](
            dv2.unsafe_ptr(), dkc.unsafe_ptr(), dwv.unsafe_ptr(), Int32(n_train), Int32(n_star),
            grid_dim=((n_train * n_star + GPC_VAR_TPB - 1) // GPC_VAR_TPB, 1, 1),
            block_dim=(GPC_VAR_TPB, 1, 1),
        )
        var dl2 = _upload(ctx, l)
        trsm_lower(ctx, dl2, dv2, n_train, n_star, trace, "gpc.v", CHOL_SOLVE_TPB)
        gpc_latent_var_launch(ctx, dvar, dv2, n_train, n_star, kss)
        # DEVIATION 2832's probability from the resident mean and variance
        var dpr = ctx.enqueue_create_buffer[DType.uint64](n_star)
        ctx.enqueue_function[gpc_proba_kernel](
            _dp(dmean), _dp(dvar), _GU(unsafe_from_address=Int(dpr.unsafe_ptr())), Int32(n_star),
            grid_dim=_gblocks(n_star), block_dim=GPC_STEP_TPB,
        )
        variance = _download(ctx, dvar, n_star)
        var hpr = ctx.enqueue_create_host_buffer[DType.uint64](n_star)
        ctx.enqueue_copy(dst_ptr=hpr.unsafe_ptr(), src_buf=dpr)
        ctx.synchronize()
        proba = List[Float64](unsafe_uninit_length=n_star)
        unsafe_memcpy(
            dest=MutPointer[UInt64, MutAnyOrigin](unsafe_from_address=Int(proba.unsafe_ptr())),
            src=hpr.unsafe_ptr(),
            count=n_star,
        )
        _ = dpr^
        _ = hpr^
        _ = dwv^
        _ = dv2^
        _ = dvar^
        _ = dl2^
    if st_on:
        print("GPC_PREDICT_STAGES n_star=" + String(n_star) + " variance_ms="
              + String((Int(perf_counter_ns()) - t_v0) // 1000000))
    _ = dx^
    _ = dxs^
    _ = dls^
    _ = dr^
    _ = dyt^
    _ = dpit^
    _ = dkc^
    _ = dstack^
    _ = dmean^
    _ = dws^
    # DEVIATION 1946: the context dies LAST.
    _ = ctx^
    return GPCLatent(mean^, variance^, proba^)
