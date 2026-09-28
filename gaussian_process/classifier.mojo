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
Newton right-hand side, the likelihood and the stop test, the variance fold
and the float64 probability) are `gaussian_process/host/gpc_steps.mojo`, the
same source the CPU host oracle compiles. DEVIATIONS 2830 (the stop rule),
2831 (the pinned float32 orders) and 2832 (the float64 probability and its
erf) are written there.

NO OPTIMIZER. As for the regressor (DEVIATION 1761), the kernel is the
kernel passed in; `optimizer`, `n_restarts_optimizer` and `warm_start` are
refused by name in Python before a binding is reached. DEVIATION 1766, the
refusal of classification itself, is closed by DEVIATION 2830.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.host.dim import Dim
from std.gpu import block_idx, thread_idx
from checks.numerics import ftz, identical_mul
from checks.numerics import NUMERIC_FAST as _NUMERIC_FAST
from std.sys.info import has_apple_gpu_accelerator
from cholesky.checks.potrf import (
    CHOL_ELEM_TPB,
    CHOL_NB_PINNED,
    CHOL_PANEL_TPB,
    add_jitter,
    chol_default_nb_hint,
    chol_logdet,
    chol_nb_for,
    chol_workspace_floats,
    potrf_lower,
)
from cholesky.checks.trsm import cho_solve
from checks.numerics import GLOBAL_NUMERIC_MODE as _CTX_MODE, NUMERIC_IDENTICAL as _CTX_IDENTICAL
from core.neural_context import neural_ctx
from std.time import perf_counter_ns
from std.os import getenv
from std.sys.compile import is_defined
# ONE PROCESS-LIFETIME DeviceContext per binding and tier (CURRENT DIRECTIVES;
# lane/neighbors-apple 2026-09-28): a new context per entry is a new Metal
# queue and a pipeline load per call. Same kernels, same launches, same order
# on one stream, and every entry still synchronizes before it returns, so no
# bit moves. This module is compiled into ONE GPU binding, so the slot name
# (per module and tier) is that binding's own.
comptime _FAMILY_CTX = "MojoGpContextIdentical" if _CTX_MODE == _CTX_IDENTICAL else "MojoGpContextOther"


def _family_ctx() raises -> DeviceContext:
    """The binding's process-lifetime context; `-D MOJOLEARN_FAMILY_CTX_PER_CALL`
    restores a new context per entry (the A/B arm)."""
    comptime if is_defined["MOJOLEARN_FAMILY_CTX_PER_CALL"]():
        return DeviceContext()
    return neural_ctx[_FAMILY_CTX]()

from cholesky.checks.trsm import CHOL_SOLVE_TPB, trsm_lower
from cholesky.estimator import cholesky_factor_host, cholesky_solve_host
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
from gaussian_process.host.gpc_steps import (
    GPCBinaryFit,
    GPCLatent,
    gpc_a_vector,
    gpc_b_matrix,
    gpc_latent_var,
    gpc_lml,
    gpc_neg_inf32,
    gpc_newton_rhs,
    gpc_residual,
    gpc_scale,
    gpc_scale_rows,
    gpc_stop,
    gpc_validate_labels,
    gpc_validate_max_iter,
    gpc_weights,
)
from gemm.checks.gemm_identical import (
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
)
from gemm.checks.gemm_oracle import OP_TN


def _gpc_kernel_self(
    x: List[Float32], n_train: Int, n_features: Int, kernel: GPKernelSpec
) raises -> List[Float32]:
    """`K = kernel(X)` on the device (`_gpc.py:261`), is_self True, so a
    WhiteKernel adds its noise to the diagonal as in the regressor."""
    var trace = IdentityTrace()
    var ctx = _family_ctx()
    var dx = _upload(ctx, x)
    var dls = _upload(ctx, _length_scale_table(kernel))
    var dk = ctx.enqueue_create_buffer[DType.float32](n_train * n_train)
    var dstack = ctx.enqueue_create_buffer[DType.float32](
        gp_kernel_stack_floats(n_train, n_train)
    )
    ctx.synchronize()
    gp_kernel_matrix(
        ctx,
        dk,
        dx,
        dx,
        dls,
        dstack,
        n_train,
        n_train,
        n_features,
        kernel,
        True,
        trace,
        "gpc.kernel",
        GP_ELEM_TPB,
        GP_SAB_NONE,
    )
    var k_host = _download(ctx, dk, n_train * n_train)
    _ = dx^
    _ = dls^
    _ = dk^
    _ = dstack^
    # DEVIATION 1946: the context dies LAST.
    _ = ctx^
    return k_host^


def _gpc_matvec(k: List[Float32], v: List[Float32], n: Int) raises -> List[Float32]:
    """`K v` through the pinned gemm at `OP_TN` (`K` is symmetric by bits,
    so `K^T v` is `K v`), the host oracle's `gemm_oracle(k, v, OP_TN, n, 1,
    n)` on the device."""
    var ctx = _family_ctx()
    var dk = _upload(ctx, k)
    var dv = _upload(ctx, v)
    var dc = ctx.enqueue_create_buffer[DType.float32](n)
    var dws = ctx.enqueue_create_buffer[DType.float32](
        identical_gemm_workspace_max_floats(n, 1, n)
    )
    ctx.synchronize()
    identical_gemm_into(ctx, dc, dk, dv, dws, n, 1, n, OP_TN)
    var out = _download(ctx, dc, n)
    _ = dk^
    _ = dv^
    _ = dc^
    _ = dws^
    _ = ctx^
    return out^


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
#: `-D MOJOLEARN_GPC_HOST_NEWTON` keeps the host round trips.
comptime GPC_DEVICE_NEWTON = not is_defined["MOJOLEARN_GPC_HOST_NEWTON"]()
comptime GPC_B_TPB = 256


#: lane neighbors-apple3 (2026-09-28), FAST on Apple, OPT-IN until its A/B
#: and quality check pass (`-D MOJOLEARN_GPC_DEVICE_VAR`): the latent
#: variance stays on the device. `gpc_scale_rows` and `gpc_latent_var` as
#: kernels (their statements, the fold over i ascending, one thread per
#: column), so the n_train x n_star cross covariance is not read back,
#: scaled on the host, uploaded, read back again and folded on the host.
comptime GPC_DEVICE_VAR = (
    _CTX_MODE == _NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_GPC_DEVICE_VAR"]()
)
#: OPT-IN (`-D MOJOLEARN_GPC_RESIDENT_K`): the device Newton loop takes K
#: from the kernel launch's own buffer, not from a download and an upload.
comptime GPC_RESIDENT_K = (
    _CTX_MODE == _NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_GPC_RESIDENT_K"]()
)


def gpc_b_matrix_kernel(
    b: MutPointer[Float32, MutAnyOrigin],
    k: MutPointer[Float32, MutAnyOrigin],
    wsr: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    var n = Int(n_in)
    var e = Int(block_idx.x) * GPC_B_TPB + Int(thread_idx.x)
    if e >= n * n:
        return
    var i = e // n
    var j = e - i * n
    var ww = ftz(identical_mul(ftz(wsr[unsafe_offset = i]), ftz(wsr[unsafe_offset = j])))
    var cell = ftz(identical_mul(ww, ftz(k[unsafe_offset = e])))
    if i == j:
        cell = ftz(Float32(1.0) + cell)
    b[unsafe_offset = e] = cell


def _gpc_matvec_dev(
    ctx: DeviceContext, mut dk: DeviceBuffer[DType.float32], v: List[Float32], n: Int
) raises -> List[Float32]:
    """`_gpc_matvec` against the resident K."""
    var dv = _upload(ctx, v)
    var dc = ctx.enqueue_create_buffer[DType.float32](n)
    var dws = ctx.enqueue_create_buffer[DType.float32](
        identical_gemm_workspace_max_floats(n, 1, n)
    )
    ctx.synchronize()
    identical_gemm_into(ctx, dc, dk, dv, dws, n, 1, n, OP_TN)
    var out = _download(ctx, dc, n)
    _ = dv^
    _ = dc^
    _ = dws^
    return out^


def _gpc_fit_binary_device(
    x: List[Float32],
    n_train: Int,
    n_features: Int,
    y: List[Float32],
    kernel: GPKernelSpec,
    max_iter_predict: Int,
) raises -> GPCBinaryFit:
    # MOJOLEARN_STAGE_TIMES=1: wall per phase (every phase drains on its
    # own), printed once. Timing only.
    var st_on = getenv("MOJOLEARN_STAGE_TIMES") == "1"
    var t_k0 = Int(perf_counter_ns())
    var n = n_train
    var ctx = _family_ctx()
    var dk: DeviceBuffer[DType.float32]
    # Mojo imports belong to module or function scope, not a conditional block.
    from gaussian_process.gpc_resident_k import _gpc_kernel_self_dev

    comptime if GPC_RESIDENT_K:
        dk = _gpc_kernel_self_dev(ctx, x, n_train, n_features, kernel)
    else:
        var k = _gpc_kernel_self(x, n_train, n_features, kernel)
        dk = _upload(ctx, k)
    var t_kernel = Int(perf_counter_ns()) - t_k0
    var t_b = 0
    var t_f = 0
    var t_ld = 0
    var t_mv = 0
    var t_s = 0
    var db = ctx.enqueue_create_buffer[DType.float32](n * n)
    var dwsr = ctx.enqueue_create_buffer[DType.float32](n)
    var hwsr = ctx.enqueue_create_host_buffer[DType.float32](n)
    var nb_pin = chol_nb_for(n, CHOL_NB_PINNED)
    var ws = ctx.enqueue_create_buffer[DType.float32](chol_workspace_floats(n, nb_pin))
    var dwork = ctx.enqueue_create_buffer[DType.float32](n + 1)
    ctx.synchronize()
    var trace = IdentityTrace()
    var f = List[Float32](capacity=n)
    for _i in range(n):
        f.append(Float32(0.0))
    var previous = gpc_neg_inf32()
    var n_iter = 0
    var last_pi = List[Float32]()
    var last_wsr = List[Float32]()
    var nb = 0
    for it in range(max_iter_predict):
        var s0 = Int(perf_counter_ns())
        var wt = gpc_weights(f)
        for i in range(n):
            hwsr.unsafe_ptr().unsafe_store(i, wt.wsr[i])
        ctx.enqueue_copy(dst_buf=dwsr, src_ptr=hwsr.unsafe_ptr())
        ctx.enqueue_function[gpc_b_matrix_kernel](
            db.unsafe_ptr(), dk.unsafe_ptr(), dwsr.unsafe_ptr(), Int32(n),
            grid_dim=Dim((n * n + GPC_B_TPB - 1) // GPC_B_TPB, 1, 1),
            block_dim=Dim(GPC_B_TPB, 1, 1),
        )
        add_jitter(ctx, db, n, Float32(0.0), CHOL_ELEM_TPB)
        if st_on:
            ctx.synchronize()
        var s1 = Int(perf_counter_ns())
        var run = potrf_lower(
            ctx, db, ws, n, trace, chol_default_nb_hint(), CHOL_PANEL_TPB, CHOL_ELEM_TPB
        )
        var s2 = Int(perf_counter_ns())
        t_b += s1 - s0
        t_f += s2 - s1
        if run.info != 0:
            raise Error(
                "gpc_fit_host: the factorization of B = I + W_sr K W_sr failed"
                " at Newton iteration "
                + String(it + 1)
                + " (info="
                + String(run.info)
                + "). B's eigenvalues are at least 1 for any finite kernel"
                " matrix, so this means a non-finite latent value"
            )
        var logdet = chol_logdet(ctx, db, dwork, n, trace, CHOL_ELEM_TPB)
        var s3 = Int(perf_counter_ns())
        var bvec = gpc_newton_rhs(wt.w, f, y, wt.pi)
        var kb = _gpc_matvec_dev(ctx, dk, bvec, n)
        var s4 = Int(perf_counter_ns())
        t_ld += s3 - s2
        t_mv += s4 - s3
        var c = gpc_scale(wt.wsr, kb)
        for i in range(n):
            var v = c[i]
            if v != v:
                raise Error(
                    "cholesky_solve_host: the right-hand side contains NaN at"
                    " flat index "
                    + String(i)
                    + "; refused by name (DEVIATION 1638)"
                )
        var dc = _upload(ctx, c)
        cho_solve(ctx, db, dc, n, 1, trace, CHOL_SOLVE_TPB)
        ctx.synchronize()
        var xs = _download(ctx, dc, n)
        _ = dc^
        var s5 = Int(perf_counter_ns())
        var a = gpc_a_vector(bvec, wt.wsr, xs)
        f = _gpc_matvec_dev(ctx, dk, a, n)
        t_s += s5 - s4
        t_mv += Int(perf_counter_ns()) - s5
        var lml = gpc_lml(a, f, y, logdet)
        n_iter = it + 1
        last_pi = wt.pi.copy()
        last_wsr = wt.wsr.copy()
        nb = run.nb
        if gpc_stop(lml, previous):
            break
        previous = lml
    var t_l0 = Int(perf_counter_ns())
    var last_l = _download(ctx, db, n * n)
    if st_on:
        print("GPC_FIT_DEVICE_STAGES iters=" + String(n_iter) + " kernel_ms=" + String(t_kernel // 1000000)
              + " b_matrix_ms=" + String(t_b // 1000000) + " factor_ms=" + String(t_f // 1000000)
              + " logdet_ms=" + String(t_ld // 1000000) + " matvec_host_ms=" + String(t_mv // 1000000)
              + " solve_ms=" + String(t_s // 1000000)
              + " read_l_ms=" + String((Int(perf_counter_ns()) - t_l0) // 1000000))
    _ = dk^
    _ = db^
    _ = dwsr^
    _ = hwsr^
    _ = ws^
    _ = dwork^
    return GPCBinaryFit(last_l^, last_pi^, last_wsr^, previous, n_iter, nb)


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

    comptime if GPC_DEVICE_NEWTON:
        return _gpc_fit_binary_device(x, n_train, n_features, y, kernel, max_iter_predict)
    var k = _gpc_kernel_self(x, n_train, n_features, kernel)
    var f = List[Float32](capacity=n_train)
    for _i in range(n_train):
        f.append(Float32(0.0))
    var previous = gpc_neg_inf32()
    var n_iter = 0
    var last_pi = List[Float32]()
    var last_wsr = List[Float32]()
    var last_l = List[Float32]()
    var nb = 0
    # MOJOLEARN_STAGE_TIMES=1: wall per phase of the Newton loop (every
    # phase drains on its own), printed once. Timing only.
    var st_on = getenv("MOJOLEARN_STAGE_TIMES") == "1"
    var t_b = 0
    var t_f = 0
    var t_mv = 0
    var t_s = 0
    var t_h = 0
    for it in range(max_iter_predict):
        var t0 = Int(perf_counter_ns())
        var wt = gpc_weights(f)
        var bmat = gpc_b_matrix(k, wt.wsr, n_train)
        var t1 = Int(perf_counter_ns())
        var factor = cholesky_factor_host(bmat, n_train, Float32(0.0))
        var t2 = Int(perf_counter_ns())
        t_b += t1 - t0
        t_f += t2 - t1
        if factor.info != 0:
            raise Error(
                "gpc_fit_host: the factorization of B = I + W_sr K W_sr failed"
                " at Newton iteration "
                + String(it + 1)
                + " (info="
                + String(factor.info)
                + "). B's eigenvalues are at least 1 for any finite kernel"
                " matrix, so this means a non-finite latent value"
            )
        var bvec = gpc_newton_rhs(wt.w, f, y, wt.pi)
        var t3 = Int(perf_counter_ns())
        var kb = _gpc_matvec(k, bvec, n_train)
        var t4 = Int(perf_counter_ns())
        var c = gpc_scale(wt.wsr, kb)
        var xs = cholesky_solve_host(factor, c, 1)
        var t5 = Int(perf_counter_ns())
        var a = gpc_a_vector(bvec, wt.wsr, xs)
        var t6 = Int(perf_counter_ns())
        f = _gpc_matvec(k, a, n_train)
        var t7 = Int(perf_counter_ns())
        var lml = gpc_lml(a, f, y, factor.logdet)
        n_iter = it + 1
        last_pi = wt.pi.copy()
        last_wsr = wt.wsr.copy()
        last_l = factor.l.copy()
        nb = factor.nb
        var t8 = Int(perf_counter_ns())
        t_mv += (t4 - t3) + (t7 - t6)
        t_s += t5 - t4
        t_h += (t3 - t2) + (t6 - t5) + (t8 - t7)
        if gpc_stop(lml, previous):
            break
        previous = lml
    if st_on:
        print("GPC_FIT_STAGES iters=" + String(n_iter) + " b_matrix_ms=" + String(t_b // 1000000)
              + " factor_ms=" + String(t_f // 1000000) + " matvec_ms=" + String(t_mv // 1000000)
              + " solve_ms=" + String(t_s // 1000000) + " host_ms=" + String(t_h // 1000000))
    return GPCBinaryFit(last_l^, last_pi^, last_wsr^, previous, n_iter, nb)


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
    var r = gpc_residual(y, pi)

    var trace = IdentityTrace()
    var ctx = _family_ctx()
    var dx = _upload(ctx, x_train)
    var dxs = _upload(ctx, x_star)
    var dls = _upload(ctx, _length_scale_table(kernel))
    var dr = _upload(ctx, r)
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
    var st_on = getenv("MOJOLEARN_STAGE_TIMES") == "1"
    var t_v0 = Int(perf_counter_ns())
    from gaussian_process.gpc_device_var import (
        GPC_VAR_TPB,
        gpc_latent_var_kernel,
        gpc_scale_rows_kernel,
    )

    if want_variance:
        comptime if GPC_DEVICE_VAR:
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
            ctx.enqueue_function[gpc_latent_var_kernel](
                dvar.unsafe_ptr(), dv2.unsafe_ptr(), Int32(n_train), Int32(n_star), kss,
                grid_dim=((n_star + GPC_VAR_TPB - 1) // GPC_VAR_TPB, 1, 1),
                block_dim=(GPC_VAR_TPB, 1, 1),
            )
            variance = _download(ctx, dvar, n_star)
            _ = dwv^
            _ = dv2^
            _ = dvar^
            _ = dl2^
        else:
            var kc = _download(ctx, dkc, n_train * n_star)
            var scaled = gpc_scale_rows(kc, wsr, n_train, n_star)
            var dv = _upload(ctx, scaled)
            var dl = _upload(ctx, l)
            trsm_lower(ctx, dl, dv, n_train, n_star, trace, "gpc.v", CHOL_SOLVE_TPB)
            var v = _download(ctx, dv, n_train * n_star)
            variance = gpc_latent_var(v, n_train, n_star, kss)
            _ = dv^
            _ = dl^
    if st_on:
        print("GPC_PREDICT_STAGES n_star=" + String(n_star) + " variance_ms="
              + String((Int(perf_counter_ns()) - t_v0) // 1000000))
    _ = dx^
    _ = dxs^
    _ = dls^
    _ = dr^
    _ = dkc^
    _ = dstack^
    _ = dmean^
    _ = dws^
    # DEVIATION 1946: the context dies LAST.
    _ = ctx^
    return GPCLatent(mean^, variance^)
