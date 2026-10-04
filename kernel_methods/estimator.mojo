# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The three host-visible surfaces: KernelRidge, Nystroem, RBFSampler.

**NOT YET WIRED** into `bindings/_mojolearn_estimators.mojo` or
`python/mojolearn/` -- those directories are not this lane's. The README's
WHAT THE MAINTAINER MUST WIRE names the tasks; this file is the entry a
binding should reach, shaped like `cholesky/estimator.mojo::
cholesky_factor_host` and `kde/estimator.mojo::kde_score_samples_host`.

THE SHAPE IS AN ARGUMENT, NOT A CONVENIENCE, IN FOUR PLACES:

1. **EVERY MODEL CARRIES THE PARAMETERS THAT PRODUCED IT.** `KernelRidgeModel`
   holds its `KernelParams` and its `alpha`; `NystroemModel` holds its
   `KernelParams`, its `seed` and its basis row ids; `RBFSamplerModel` holds
   its `gamma`, its `seed` and the two derived constants. A model that does
   not carry them cannot be compared with another model, and `predict` /
   `transform` cannot be reproduced from it. This is `CholeskyFactor`'s
   argument about `nb` and `jitter`, applied to three more estimators.
2. **`KernelRidgeModel` CARRIES `info` AND THE FIT REFUSES A NON-ZERO ONE.**
   `info` is DATA-DEPENDENT (`cholesky/`'s DEVIATION 1634), and cuML's
   response to it is a silent least-squares fallback behind a
   `warnings.warn` (DEVIATION 1662). Ours raises, names `alpha` as the
   closure, and the field survives on the struct so a device-resident caller
   sweeping hyperparameters can read it.
3. **`NystroemModel` CARRIES ITS EIGENVALUES AND EIGENVECTORS, NOT ONLY THE
   NORMALIZATION.** scikit-learn keeps only `normalization_`. A single
   normalization matrix cannot distinguish an eigenvalue error from an
   ordering error from a sign error, and the whole identity story of this
   estimator is which of those three moved. The card records all three for
   the same reason.
4. **NOTHING HERE TAKES A BLOCK SIZE OR A JITTER.** The Cholesky profile's
   `nb` and its ridge are not this surface's to express: `alpha` IS the ridge
   (DEVIATION 1660) and the block size is pinned one layer down. What IS
   exposed is `elem_tpb` / `solve_tpb`, which are SCHEDULING and which the
   checks vary precisely to show that nothing moves with them.

A CALLER THAT KEEPS ITS DATA ON THE DEVICE across a hyperparameter sweep --
which a kernel-ridge cross-validation will -- should call
`kernel_methods/checks/kernel_matrix.mojo::km_kernel_matrix` and
`kernel_methods/impl/kernel_ridge/kernel_ridge.mojo::kernel_ridge_solve`
directly and keep its own `DeviceBuffer`s, exactly as cuML's `fit` keeps `X`
on the device. These entries are the one-shot form, which is what the gates
and the card use.
"""

# DEVIATION 2486: bulk host staging; stream/lifetime boundaries unchanged.
from bindings.hostptr import copy_f32
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE as _CTX_MODE, NUMERIC_IDENTICAL as _CTX_IDENTICAL
from core.neural_context import neural_ctx
from std.time import perf_counter_ns
from std.os import getenv
from std.sys.compile import is_defined
from std.memory import bitcast
# ONE PROCESS-LIFETIME DeviceContext per binding and tier (CURRENT DIRECTIVES;
# lane/neighbors-apple 2026-09-28): a new context per entry is a new Metal
# queue and a pipeline load per call. Same kernels, same launches, same order
# on one stream, and every entry still synchronizes before it returns, so no
# bit moves. This module is compiled into ONE GPU binding, so the slot name
# (per module and tier) is that binding's own.
comptime _FAMILY_CTX = "MojoKernelMethodsContextIdentical" if _CTX_MODE == _CTX_IDENTICAL else "MojoKernelMethodsContextOther"


def _family_ctx() raises -> DeviceContext:
    """The binding's process-lifetime context; `-D MOJOLEARN_FAMILY_CTX_PER_CALL`
    restores a new context per entry (the A/B arm)."""
    comptime if is_defined["MOJOLEARN_FAMILY_CTX_PER_CALL"]():
        return DeviceContext()
    return neural_ctx[_FAMILY_CTX]()

from cholesky.checks.potrf import (
    CHOL_ELEM_TPB,
    CHOL_PANEL_TPB,
)
from cholesky.checks.trsm import CHOL_SOLVE_TPB
from core.identity_trace import IdentityTrace
from decomposition.checks.jacobi_eigh_device import (
    JACOBI_SWEEPS,
    JACOBI_TOL,
    JACOBI_ROT_TPB,
    jacobi_eigh_kernel,
)
from decomposition.impl.linalg.detail.pca import (
    SIGNFLIP_TPB,
    sign_flip_kernel,
)
from gemm.checks.gemm_identical import (
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
)
from gemm.contract import OP_NN, OP_NT
from kernel_methods.checks.km_sabotage import (
    KMSAB_BASIS_FROM_LAUNCH,
    KMSAB_EIGEN_ORDER_ASCENDING,
    KMSAB_EIGEN_TIE_UNSTABLE,
    KMSAB_EMBED_OP_NN,
    KMSAB_NONE,
    KMSAB_NO_EIGEN_CLIP,
    KMSAB_NO_SIGN_FLIP,
)
from kernel_methods.checks.kernel_matrix import (
    KM_KERNEL_LINEAR,
    KM_KERNEL_PRECOMPUTED,
    KM_TPB,
    km_kernel_matrix,
    km_kernel_name,
    km_kernel_workspace_floats,
    km_validate_kernel_params,
    km_validate_matrix,
)
from kernel_methods.checks.random_features import (
    KM_RF_TPB,
    km_basis_indices,
    km_basis_indices_device,
    km_gather_rows_kernel,
    km_feature_map_epilogue,
    km_feature_scale,
    km_random_offsets,
    km_random_weights,
    km_weight_sigma,
)
from kernel_methods.impl.distance.kernel_matrices import KM_EPILOGUE_TPB
from kernel_methods.impl.kernel_ridge.kernel_ridge import (
    KRR_RIDGE_TPB,
    kernel_ridge_solve,
    kernel_ridge_workspace_floats,
)
from checks.numerics import ftz, identical_div, identical_sqrt
from checks.soft_f64 import sf64_sqrt, sf64_to_f32
from checks.numerics import NUMERIC_FAST as _NUMERIC_FAST
from core.device_scan import device_classify_nonfinite, device_first_nonfinite
from std.sys.info import has_apple_gpu_accelerator
from svm.impl.svm_parameter import KernelParams
from x_decomp.cells import F32Ptr
from x_decomp.rr import RR_EIGH_SWEEPS, RR_OFF_TPB, rr_converged, rr_fro_kept
from x_decomp.rr import rr_block, rr_cs, rr_vrow
from kernel_methods.rbf_fused import (
    RBF_FUSED_MAX_D,
    RBF_FUSED_TPB,
    rbf_fused_project_kernel,
    rbf_fused_transform_kernel,
)
from core.device_zero import enqueue_fill
from x_decomp.jacobi_par import (
    PJ_TPB,
    eigh_par_cs_kernel,
    eigh_par_off_fold_kernel,
    eigh_par_off_part_kernel,
    eigh_par_update_kernel,
    pj_identity_kernel,
)

#: The opt-in host Jacobi for Nystroem's eigenproblem (`-D MOJOLEARN_NYS_HOST_EIGH`)
#: was removed (hr-optin-flags): the eigendecomposition runs on the device.

#: lane neighbors-apple3 (2026-09-28), FAST on Apple, OPT-IN until its A/B
#: and quality check pass (`-D MOJOLEARN_RBF_FUSED`): RBFSampler.transform
#: as ONE kernel per cell, the projection (features ascending), the offset,
#: the cosine and the scale, when X has at most RBF_FUSED_MAX_D features.
#: The projection is 8 products a cell at the board's shape; forming it as
#: a matrix product first writes and reads the n x n_components matrix once
#: more (gemm 93 ms + epilogue 92 ms of the transform at 1M x 500, M4 Pro).
comptime RBF_FUSED = (
    _CTX_MODE == _NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_RBF_FUSED"]()
)


#: fam-kernel-gp (2026-10-04), IDENTICAL, ON by default
#: (`-D MOJOLEARN_IDN_RBF_FUSED_OFF` restores the GEMM on the device AND in
#: the host column, `km_host_oracle.mojo::KMH_RBF_FUSED` reads the same
#: define): RBFSampler.transform at n_features <= RBF_FUSED_MAX_D computes
#: the projection as ONE chain per cell over the features ascending
#: (`identical_mul_add`, rounded each step) and, when no trace and no
#: sabotage arm needs the projection as its own stage, the offset, cosine
#: and scale in the same kernel: one launch and one write of the
#: n x n_components matrix where the GEMM route made a workspace, the GEMM's
#: launches, and a second pass for the epilogue. BITS CHANGE at
#: n_features <= 64 (the projection's fold is the ascending chain, not the
#: GEMM profile's): NVIDIA, AMD and Apple run the same kernel and the host
#: column runs the same chain. A chain of at most 64 fused multiply-adds is
#: as accurate as the GEMM's fold at that length.
comptime RBF_IDN_FUSED = _CTX_MODE == _CTX_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_RBF_FUSED_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)


def _rbf_idn_fused_launch(
    ctx: DeviceContext,
    mut dp: DeviceBuffer[DType.float32],
    mut dx: DeviceBuffer[DType.float32],
    mut dw: DeviceBuffer[DType.float32],
    mut db: DeviceBuffer[DType.float32],
    n_rows: Int,
    d: Int,
    dd: Int,
    scale: Float32,
    whole: Bool,
) raises:
    """RBF_IDN_FUSED's launch: the whole transform (`whole`) or the
    projection alone, one thread per cell. ASYNCHRONOUS."""
    var grid = (n_rows * dd + RBF_FUSED_TPB - 1) // RBF_FUSED_TPB
    if whole:
        ctx.enqueue_function[rbf_fused_transform_kernel](
            dp.unsafe_ptr(), dx.unsafe_ptr(), dw.unsafe_ptr(), db.unsafe_ptr(),
            Int32(n_rows), Int32(d), Int32(dd), scale,
            grid_dim=(grid, 1, 1),
            block_dim=(RBF_FUSED_TPB, 1, 1),
        )
    else:
        ctx.enqueue_function[rbf_fused_project_kernel](
            dp.unsafe_ptr(), dx.unsafe_ptr(), dw.unsafe_ptr(),
            Int32(n_rows), Int32(d), Int32(dd),
            grid_dim=(grid, 1, 1),
            block_dim=(RBF_FUSED_TPB, 1, 1),
        )


# ===========================================================================
# Buffer plumbing. `cholesky/estimator.mojo`'s two helpers, character for
# character, including the `_ = host^` that keeps a host buffer alive past
# its `.unsafe_ptr()` (`[[mojo-buffer-freed-at-last-use]]`).
# ===========================================================================


def _upload(
    ctx: DeviceContext, values: List[Float32]
) raises -> DeviceBuffer[DType.float32]:
    var n = len(values)
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    copy_f32(values.unsafe_ptr(), host.unsafe_ptr(), n)
    ctx.enqueue_copy(dst_buf=buf, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return buf^


#: lane apple-fast-gap-kapprox2 (2026-10-03), the FAST + Apple default since
#: its M3 A/B kap2-km-ptrin-rbf-istella (rbf-sampler istella 121 -> 94 ms,
#: kernel_rel_error identical; `-D MOJOLEARN_KM_FAST_PTR_IN_OFF` reverts):
#: the transforms' X goes to the
#: device straight from the caller's memory (one raw host-pointer copy,
#: 1.6-2.4 ms per 64 MB on Apple) and its finiteness is scanned there
#: (`device_first_nonfinite`), instead of an owned host copy of X
#: (`read_f32`, fresh pages), a serial host finiteness walk and a second
#: copy into a fresh pinned stage (`_upload`). The same X words reach the
#: same kernels: bit-inert; the same refusal text.
#:
#: fam-kernel-gp (2026-10-04): IDENTICAL takes the same route on every
#: vendor, ON by default (`KM_IDN_PTR_IN`; `-D MOJOLEARN_IDN_KM_PTR_IN_OFF`
#: restores the owned host copy, the host walk and the staged upload). The
#: transforms' X is the one large operand (n_rows x n_features), and it
#: crossed host memory three times on one thread before the first launch.
comptime KM_IDN_PTR_IN = _CTX_MODE == _CTX_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_KM_PTR_IN_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime KM_FAST_PTR_IN = KM_IDN_PTR_IN or (
    _CTX_MODE == _NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_KM_FAST_PTR_IN_OFF"]()
)

#: fam-kernel-gp (2026-10-04), IDENTICAL, ON by default
#: (`-D MOJOLEARN_IDN_KM_BULK_DOWNLOAD_OFF` restores the appends):
#: `_download` fills a list of the final length 8 floats a step instead of n
#: appends into a growing one. The same words: no bit moves.
comptime KM_IDN_BULK_DOWNLOAD = _CTX_MODE == _CTX_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_KM_BULK_DOWNLOAD_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)


def _upload_checked(ctx: DeviceContext, xaddr: Int, n_rows: Int, n_cols: Int, what: String) raises -> DeviceBuffer[
    DType.float32
]:
    """`km_validate_matrix` + `_upload` for X at a caller's host address
    (KM_FAST_PTR_IN): shape refused on the host, NaN / infinity on the
    device, with `km_validate_matrix`'s messages."""
    if n_rows <= 0 or n_cols <= 0:
        raise Error(what + ": need positive dimensions, got " + String(n_rows) + " x " + String(n_cols))
    var n = n_rows * n_cols
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_buf=buf, src_ptr=MutPointer[Float32, MutAnyOrigin](unsafe_from_address=xaddr))
    var bad = device_first_nonfinite(ctx, buf, n)
    if bad >= 0:
        if device_classify_nonfinite(ctx, buf, bad):
            raise Error(what + ": NaN at flat index " + String(bad) + "; refused by name (DEVIATION 1686)")
        raise Error(what + ": infinity at flat index " + String(bad) + "; refused by name (DEVIATION 1686)")
    return buf^


def _download_into[out_origin: MutOrigin, //](
    ctx: DeviceContext,
    mut src: DeviceBuffer[DType.float32],
    output: MutPointer[Float32, out_origin],
    n: Int,
) raises:
    """A device result into the caller's memory: one device-to-host copy
    (cpu-gpu-cleanup c-gp-kernel: the staged copy over host threads and
    the opt-in mapped copy are gone)."""
    if n <= 0:
        return
    var sub = src.create_sub_buffer[DType.float32](0, n)
    ctx.enqueue_copy(dst_ptr=output, src_buf=sub)
    ctx.synchronize()
    _ = sub^


def _download(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int
) raises -> List[Float32]:
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    comptime if KM_IDN_BULK_DOWNLOAD:
        var packed = List[Float32](unsafe_uninit_length=n)
        var pdst = packed.unsafe_ptr()
        var psrc = h.unsafe_ptr()
        var pi = 0
        var pbody = n - n % 8
        while pi < pbody:
            pdst.unsafe_store[width=8](pi, psrc.unsafe_load[width=8](pi))
            pi += 8
        while pi < n:
            pdst.unsafe_store(pi, psrc.unsafe_load(pi))
            pi += 1
        _ = h^
        return packed^
    var out = List[Float32]()
    for i in range(n):
        out.append(h.unsafe_ptr().unsafe_load(i))
    _ = h^
    return out^


# ===========================================================================
# KernelRidge
# ===========================================================================


@fieldwise_init
struct KernelRidgeModel(Movable):
    """`sklearn.kernel_ridge.KernelRidge` and `cuml.kernel_ridge.KernelRidge`
    after `fit`, plus everything a caller must not have to recompute."""

    var dual_coef: List[Float32]
    """`dual_coef_`, `n_samples x n_targets` row-major."""

    var x_fit: List[Float32]
    """`X_fit_`, `n_samples x n_features` row-major. **PREDICT NEEDS IT**, and
    theirs keeps it for the same reason: a kernel method has no finite
    parameter vector, so the training data IS part of the model."""

    var n_samples: Int
    var n_features: Int
    var n_targets: Int

    var kernel: Int
    var degree: Int
    var gamma: Float64
    var coef0: Float64
    """The `KernelParams` fields, unpacked. Unpacked rather than held as a
    `KernelParams` so the struct stays `Movable` without depending on another
    lane's conformances, and so a caller reading `model.gamma` does not have
    to know which struct it came from."""

    var alpha: Float32
    """The ridge that was added, by value. DEVIATION 1660: this IS the ridge,
    and the Cholesky profile's jitter was `+0.0`."""

    var info: Int
    """LAPACK's `info` from the factorization. Always 0 on a model returned by
    `kernel_ridge_fit_host`, which refuses anything else; carried so that a
    device-resident caller can build one and inspect it."""


def kernel_ridge_params(model: KernelRidgeModel) -> KernelParams:
    """The model's kernel parameters, re-packed. One place that knows the
    field order, so `predict` and the checks cannot disagree with `fit`."""
    return KernelParams(
        model.kernel, model.degree, model.gamma, model.coef0
    )


#: SCHEDULING: one thread per kernel-matrix cell.
comptime KRR_WEIGHT_TPB = 256


def krr_weight_kernel(
    k_io: MutPointer[Float32, MutAnyOrigin],
    s: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`K *= outer(sw, sw)` (scikit-learn `_solve_cholesky_kernel`'s
    weighted arm), one thread per cell, in float32: `ftz(ftz(K_ij) *
    ftz(ftz(s_i) * ftz(s_j)))`, the pair product rounded first. DEVIATION
    1688: theirs forms the outer product and the scaled cell in float64 and
    rounds once; this rounds twice, on every vendor the same way
    (`kmh_weight_kernel` restates it on the host)."""
    var n = Int(n_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= n * n:
        return
    var i = idx // n
    var j = idx - i * n
    var w = ftz(ftz(s.unsafe_load(i)) * ftz(s.unsafe_load(j)))
    k_io.unsafe_store(idx, ftz(ftz(k_io.unsafe_load(idx)) * w))


def krr_scale_rows(v: List[Float32], sw: List[Float32], n: Int, t: Int) -> List[Float32]:
    """`y * sw[:, None]` and `dual_coef *= sw[:, None]`, on the host, one
    rounding per cell: `ftz(ftz(v) * ftz(s_i))` (`kmh_scale_rows` is the
    same line)."""
    var out = List[Float32](capacity=n * t)
    for i in range(n):
        var si = ftz(sw[i])
        for c in range(t):
            out.append(ftz(ftz(v[i * t + c]) * si))
    return out^


#: fam2-kernel-gp (2026-10-04), IDENTICAL, ON by default
#: (`-D MOJOLEARN_IDN_KRR_DEV_SCALE_OFF` restores the host loops): the
#: weighted fit's `y * sw[:, None]` and `dual_coef *= sw[:, None]` run as one
#: launch each on the resident buffers (`krr_scale_rows_kernel`, one thread
#: per cell, `krr_scale_rows`'s line) instead of two host loops over n x t
#: cells around the device solve. One multiplication per cell rounded once
#: either way: no bit moves, and `kmh_scale_rows` stays the host column.
#: cpu2-l6-bindings: ON on EVERY tier (FAST included); the host loops were
#: CPU work on a GPU route. `MOJOLEARN_IDN_ALL_OFF` still turns it off on
#: IDENTICAL builds only.
comptime KRR_IDN_DEV_SCALE = not (
    is_defined["MOJOLEARN_IDN_KRR_DEV_SCALE_OFF"]()
    or (_CTX_MODE == _CTX_IDENTICAL and is_defined["MOJOLEARN_IDN_ALL_OFF"]())
)

#: fam2-kernel-gp (2026-10-04), IDENTICAL, ON by default
#: (`-D MOJOLEARN_IDN_KRR_PTR_IN_OFF` unregisters the pointer bindings, so
#: `kernel_methods.py` takes the list route again): KernelRidge fit and
#: predict read X, y, X_fit and the dual from the caller's memory straight
#: to the device, X and y are scanned for NaN / infinity there
#: (`_upload_checked`) and the result is copied into the caller's output,
#: instead of an owned host copy of every operand (`read_f32`), a serial
#: host finiteness walk, a staged second copy (`_upload`), a host list of
#: the result and a model copy of X. The same words reach the same
#: kernels: no bit moves; the same refusal texts.
#:
#: cpu2-l6-bindings (2026-10-04): the pointer route is the default on EVERY
#: tier and vendor (FAST included), so no fast build stages X, y or the
#: sample-weight factors through host lists. Same kernels at the same
#: default scheduling as the list route: FAST bits do not move either.
comptime KRR_IDN_PTR_IN = not (
    is_defined["MOJOLEARN_IDN_KRR_PTR_IN_OFF"]()
    or (_CTX_MODE == _CTX_IDENTICAL and is_defined["MOJOLEARN_IDN_ALL_OFF"]())
)

#: SCHEDULING: one thread per target cell.
comptime KRR_SCALE_TPB = 256


def krr_scale_rows_kernel(
    v_io: MutPointer[Float32, MutAnyOrigin],
    s: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    t_in: Int32,
):
    """`krr_scale_rows` in place, one thread per cell (i, c):
    `ftz(ftz(v) * ftz(s_i))`."""
    var t = Int(t_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(n_in) * t:
        return
    var i = idx // t
    v_io.unsafe_store(idx, ftz(ftz(v_io.unsafe_load(idx)) * ftz(s.unsafe_load(i))))


#: SCHEDULING: one thread per row.
comptime KRR_SQRT_W_TPB = 256


def krr_sqrt_weights_kernel(
    w: MutPointer[UInt64, MutAnyOrigin],
    out: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """cpu2-l6-bindings: the per-row factor `sqrt(sample_weight)`, the
    binary64 square root of each weight rounded once to float32, on the
    device (`checks/soft_f64.mojo`: `sf64_sqrt` is correctly rounded and
    `sf64_to_f32` is round-to-nearest-even, the host loop's
    `Float32(sqrt(w))` word for word; the Apple GPU has no float64). A
    negative weight gives NaN and the caller's device scan refuses it."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    out.unsafe_store(i, sf64_to_f32(sf64_sqrt(w.unsafe_load(i))))


def _krr_sqrt_weights_dev(
    ctx: DeviceContext, waddr: Int, n: Int
) raises -> DeviceBuffer[DType.float32]:
    """The factors from the caller's float64 weights at `waddr`: one
    upload, one launch, and `krr_validate_weights`' refusal (finite and
    >= 0) as a device scan; one Int32 per block is read back, never the
    data."""
    if n <= 0 or n > 2147483647:
        raise Error("kernel_ridge_fit_host: sample weight count out of range")
    var dw = ctx.enqueue_create_buffer[DType.uint64](n)
    ctx.enqueue_copy(
        dst_buf=dw, src_ptr=MutPointer[UInt64, MutAnyOrigin](unsafe_from_address=waddr)
    )
    var dsw = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_function[krr_sqrt_weights_kernel](
        dw.unsafe_ptr(), dsw.unsafe_ptr(), Int32(n),
        grid_dim=((n + KRR_SQRT_W_TPB - 1) // KRR_SQRT_W_TPB, 1, 1),
        block_dim=(KRR_SQRT_W_TPB, 1, 1),
    )
    var bad = device_first_nonfinite(ctx, dsw, n)
    _ = dw^
    if bad >= 0:
        raise Error(
            "kernel_ridge_fit_host: sample weight factor " + String(bad)
            + " is negative or not finite; refused by name"
        )
    return dsw^


def _krr_scale_rows_dev(
    ctx: DeviceContext,
    mut v: DeviceBuffer[DType.float32],
    mut s: DeviceBuffer[DType.float32],
    n: Int,
    t: Int,
) raises:
    """Launch `krr_scale_rows_kernel` over the n x t cells of `v`. ASYNCHRONOUS."""
    if n <= 0 or t <= 0:
        return
    ctx.enqueue_function[krr_scale_rows_kernel](
        v.unsafe_ptr(), s.unsafe_ptr(), Int32(n), Int32(t),
        grid_dim=((n * t + KRR_SCALE_TPB - 1) // KRR_SCALE_TPB, 1, 1),
        block_dim=(KRR_SCALE_TPB, 1, 1),
    )


def _krr_not_pd_message(info: Int, kernel: Int, n_samples: Int) -> String:
    """DEVIATION 1662's refusal text (one copy for the list and pointer fits)."""
    return (
        "kernel_ridge_fit_host: the ridged kernel matrix K + alpha I is"
        " NOT positive definite (info="
        + String(info)
        + ", the leading minor of order "
        + String(info)
        + " failed). kernel="
        + km_kernel_name(kernel)
        + ", n_samples="
        + String(n_samples)
        + ". cuML catches this and silently returns a LEAST-SQUARES"
        " solution instead (kernel_ridge.py:26-44, behind a"
        " warnings.warn); this lane refuses, because a fit that returns"
        " a different estimator than the one it was asked for is a"
        " wrong answer with no error. THE CLOSURE IS alpha: raise it."
        " A float32 kernel matrix needs a larger ridge than cuML's"
        " float64 one at the same data (DEVIATION 1661). To close it"
        " properly, implementation an SVD-based least-squares arm -- there is one"
        " at solver/checks/lstsq.mojo -- and gate BOTH sides of the"
        " branch; kernel_methods/NOT_IMPLEMENTED.tsv carries the row"
    )


def _krr_validate_alpha(alpha: Float32) raises:
    """`kernel_ridge_fit_host`'s alpha refusals (NaN, negative)."""
    if alpha != alpha:
        raise Error("kernel_ridge_fit_host: alpha is NaN; refused by name")
    if alpha < Float32(0.0):
        raise Error(
            "kernel_ridge_fit_host: alpha must be non-negative, got a"
            " negative value. scikit-learn's own parameter constraint is"
            " Interval(Real, 0, None, closed='left') and a negative ridge"
            " SUBTRACTS from the diagonal, which can turn a positive"
            " definite kernel matrix indefinite and make the Cholesky fail"
            " on data that is perfectly well conditioned. DEVIATION 1686"
        )


def _upload_ptr(ctx: DeviceContext, addr: Int, n: Int) raises -> DeviceBuffer[DType.float32]:
    """`n` floats at a caller's host address onto the device, unchecked (a
    fitted array the list route did not validate either). The copy is
    waited on here: the caller's memory is read before this returns."""
    var buf = ctx.enqueue_create_buffer[DType.float32](max(n, 1))
    if n > 0:
        var sub = buf.create_sub_buffer[DType.float32](0, n)
        ctx.enqueue_copy(dst_buf=sub, src_ptr=MutPointer[Float32, MutAnyOrigin](unsafe_from_address=addr))
        ctx.synchronize()
        _ = sub^
    return buf^


def krr_validate_weights(sw: List[Float32], n: Int) raises:
    """The per-row sqrt(sample_weight) factors: n of them, finite, >= 0."""
    if len(sw) != n:
        raise Error(
            "kernel_ridge_fit_host: sample weight factors hold "
            + String(len(sw)) + " values, X has " + String(n) + " rows"
        )
    for i in range(n):
        var v = sw[i]
        if not (v >= Float32(0.0)) or v > Float32(3.4028234663852886e38):
            raise Error(
                "kernel_ridge_fit_host: sample weight factor " + String(i)
                + " is negative or not finite; refused by name"
            )


def kernel_ridge_fit_host(
    x: List[Float32],
    y: List[Float32],
    n_samples: Int,
    n_features: Int,
    n_targets: Int,
    kp: KernelParams,
    alpha: Float32,
    mut trace: IdentityTrace,
    elem_tpb: Int = KM_EPILOGUE_TPB,
    panel_tpb: Int = CHOL_PANEL_TPB,
    chol_elem_tpb: Int = CHOL_ELEM_TPB,
    solve_tpb: Int = CHOL_SOLVE_TPB,
    ridge_tpb: Int = KRR_RIDGE_TPB,
    sabotage: Int = KMSAB_NONE,
    sw: List[Float32] = List[Float32](),
) raises -> KernelRidgeModel:
    """`KernelRidge.fit(X, y)`: form `K`, ridge it, factor it, solve it.

    Their four lines (`kernel_ridge.py:305-313`), in their order:

        K = self._get_kernel(X)
        dual_coef = _solve_cholesky_kernel(K, y, alpha, sample_weight)
                        .astype(X.dtype, copy=False)
        self.X_fit_ = X
        self.dual_coef_ = dual_coef

    Refuses on the HOST, by name, before any upload (DEVIATION 1686):
    non-finite `X` or `y`, a bad shape, an unsupported kernel, a
    non-positive `gamma` where the kernel needs one, an out-of-range
    `degree`, and a NEGATIVE `alpha`.

    **A NEGATIVE `alpha` IS REFUSED AND scikit-learn's SCHEMA REFUSES IT
    TOO** (`Interval(Real, 0, None, closed="left")`), so this is a
    transcription of their constraint rather than a policy of ours. A ZERO
    `alpha` is ACCEPTED, because it is inside their interval and because the
    exact fixture this lane gates on needs it; what happens at `alpha = 0` on
    an ill-conditioned kernel is DEVIATION 1662's refusal, which names
    `alpha` as its closure.
    """
    km_validate_matrix(x, n_samples, n_features, "kernel_ridge X")
    km_validate_matrix(y, n_samples, n_targets, "kernel_ridge y")
    # kernel='precomputed': X IS the n x n kernel matrix (scikit-learn's
    # `pairwise_kernels(X, metric='precomputed')` returns it as given).
    var precomputed = kp.kernel == KM_KERNEL_PRECOMPUTED
    if precomputed:
        if n_features != n_samples:
            raise Error(
                "kernel_ridge_fit_host: kernel='precomputed' needs a square"
                " kernel matrix, got " + String(n_samples) + " x "
                + String(n_features)
            )
    else:
        km_validate_kernel_params(kp, "kernel_ridge")
    var weighted = len(sw) > 0
    if weighted:
        krr_validate_weights(sw, n_samples)
    if alpha != alpha:
        raise Error("kernel_ridge_fit_host: alpha is NaN; refused by name")
    if alpha < Float32(0.0):
        raise Error(
            "kernel_ridge_fit_host: alpha must be non-negative, got a"
            " negative value. scikit-learn's own parameter constraint is"
            " Interval(Real, 0, None, closed='left') and a negative ridge"
            " SUBTRACTS from the diagonal, which can turn a positive"
            " definite kernel matrix indefinite and make the Cholesky fail"
            " on data that is perfectly well conditioned. DEVIATION 1686"
        )

    var ctx = _family_ctx()

    # DEVIATION 2487: self-kernel operands share one uploaded allocation.
    var xa = _upload(ctx, x)
    # KRR_IDN_DEV_SCALE: the weighted targets are scaled where they lie.
    var dev_scale = weighted and KRR_IDN_DEV_SCALE
    var dy = _upload(ctx, krr_scale_rows(y, sw, n_samples, n_targets)) if (weighted and not dev_scale) else _upload(ctx, y)
    var dsw = _upload(ctx, sw) if dev_scale else ctx.enqueue_create_buffer[DType.float32](1)
    if dev_scale:
        _krr_scale_rows_dev(ctx, dy, dsw, n_samples, n_targets)
    trace.record_device(ctx, "krr.input", xa, n_samples * n_features)

    var dk = ctx.enqueue_create_buffer[DType.float32](n_samples * n_samples)
    var na = ctx.enqueue_create_buffer[DType.float32](n_samples)
    var nb = ctx.enqueue_create_buffer[DType.float32](n_samples)
    var kws = ctx.enqueue_create_buffer[DType.float32](
        km_kernel_workspace_floats(n_samples, n_samples, n_features)
    )
    ctx.synchronize()

    if precomputed:
        ctx.enqueue_copy(dst_buf=dk, src_buf=xa)
    else:
        km_kernel_matrix(
            ctx, kp, dk, xa, xa, n_samples, n_samples, n_features,
            na, nb, kws, elem_tpb, sabotage, True,
        )
    ctx.synchronize()
    trace.record_device(ctx, "krr.kernel", dk, n_samples * n_samples)
    if weighted:
        var ds = _upload(ctx, sw)
        var cells = n_samples * n_samples
        ctx.enqueue_function[krr_weight_kernel](
            dk.unsafe_ptr(), ds.unsafe_ptr(), Int32(n_samples),
            grid_dim=((cells + KRR_WEIGHT_TPB - 1) // KRR_WEIGHT_TPB, 1, 1),
            block_dim=(KRR_WEIGHT_TPB, 1, 1),
        )
        ctx.synchronize()
        trace.record_device(ctx, "krr.weighted", dk, cells)
        _ = ds^

    var cws = ctx.enqueue_create_buffer[DType.float32](
        kernel_ridge_workspace_floats(n_samples)
    )
    ctx.synchronize()
    var info = kernel_ridge_solve(
        ctx, dk, dy, cws, n_samples, n_targets, alpha, trace,
        panel_tpb, chol_elem_tpb, solve_tpb, ridge_tpb, sabotage,
    )
    ctx.synchronize()

    if info != 0:
        # DEVIATION 1662. cuML warns and switches estimator; we refuse and
        # name the closure.
        raise Error(
            "kernel_ridge_fit_host: the ridged kernel matrix K + alpha I is"
            " NOT positive definite (info="
            + String(info)
            + ", the leading minor of order "
            + String(info)
            + " failed). kernel="
            + km_kernel_name(kp.kernel)
            + ", n_samples="
            + String(n_samples)
            + ". cuML catches this and silently returns a LEAST-SQUARES"
            " solution instead (kernel_ridge.py:26-44, behind a"
            " warnings.warn); this lane refuses, because a fit that returns"
            " a different estimator than the one it was asked for is a"
            " wrong answer with no error. THE CLOSURE IS alpha: raise it."
            " A float32 kernel matrix needs a larger ridge than cuML's"
            " float64 one at the same data (DEVIATION 1661). To close it"
            " properly, implementation an SVD-based least-squares arm -- there is one"
            " at solver/checks/lstsq.mojo -- and gate BOTH sides of the"
            " branch; kernel_methods/NOT_IMPLEMENTED.tsv carries the row"
        )

    if dev_scale:
        _krr_scale_rows_dev(ctx, dy, dsw, n_samples, n_targets)
    var dual = _download(ctx, dy, n_samples * n_targets)
    if weighted and not dev_scale:
        dual = krr_scale_rows(dual, sw, n_samples, n_targets)
    _ = dsw^
    _ = xa^
    _ = dy^
    _ = dk^
    _ = na^
    _ = nb^
    _ = kws^
    _ = cws^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return KernelRidgeModel(
        dual^, x.copy(), n_samples, n_features, n_targets,
        kp.kernel, kp.degree, kp.gamma, kp.coef0, alpha, 0,
    )


def kernel_ridge_predict_host(
    model: KernelRidgeModel,
    x_new: List[Float32],
    n_query: Int,
    mut trace: IdentityTrace,
    elem_tpb: Int = KM_EPILOGUE_TPB,
    sabotage: Int = KMSAB_NONE,
) raises -> List[Float32]:
    """`KernelRidge.predict(X)` (`kernel_ridge.py:337-349`):

        K = self._get_kernel(X, self.X_fit_)
        return cp.dot(K, self.dual_coef_)

    `n_query x n_targets` row-major.

    DEVIATION 1680: `cp.dot` becomes `identical_gemm_into` at `OP_NN` under
    `mojolearn.identical.gemm.fp32.v1`. `linalg.matmul` is REFUSED -- a
    device-wide vendor GEMM's k-split is a per-vendor summation order that
    nothing in this repository can pin, read or check, and here the `k` axis
    is `n_samples`, which is the longest reduction in the whole estimator.
    """
    km_validate_matrix(x_new, n_query, model.n_features, "predict X")
    var kp = kernel_ridge_params(model)
    var n = model.n_samples
    var d = model.n_features
    var t = model.n_targets

    var ctx = _family_ctx()
    var dq = _upload(ctx, x_new)
    var dfit = _upload(ctx, model.x_fit)
    var ddual = _upload(ctx, model.dual_coef)
    var dk = ctx.enqueue_create_buffer[DType.float32](n_query * n)
    var na = ctx.enqueue_create_buffer[DType.float32](n_query)
    var nb = ctx.enqueue_create_buffer[DType.float32](n)
    var kws = ctx.enqueue_create_buffer[DType.float32](
        km_kernel_workspace_floats(n_query, n, d)
    )
    var dpred = ctx.enqueue_create_buffer[DType.float32](n_query * t)
    var gws = ctx.enqueue_create_buffer[DType.float32](
        identical_gemm_workspace_max_floats(n_query, t, n)
    )
    ctx.synchronize()

    if model.kernel == KM_KERNEL_PRECOMPUTED:
        # X IS the n_query x n_samples cross-kernel matrix.
        ctx.enqueue_copy(dst_buf=dk, src_buf=dq)
    else:
        km_kernel_matrix(
            ctx, kp, dk, dq, dfit, n_query, n, d, na, nb, kws, elem_tpb, sabotage
        )
    ctx.synchronize()
    trace.record_device(ctx, "krr.cross_kernel", dk, n_query * n)

    identical_gemm_into(ctx, dpred, dk, ddual, gws, n_query, t, n, OP_NN)
    ctx.synchronize()
    trace.record_device(ctx, "krr.predictions", dpred, n_query * t)

    var out = _download(ctx, dpred, n_query * t)
    _ = dq^
    _ = dfit^
    _ = ddual^
    _ = dk^
    _ = na^
    _ = nb^
    _ = kws^
    _ = dpred^
    _ = gws^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return out^


def kernel_ridge_fit_ptr_into[out_origin: MutOrigin, //](
    xaddr: Int,
    yaddr: Int,
    n_samples: Int,
    n_features: Int,
    n_targets: Int,
    kp: KernelParams,
    alpha: Float32,
    sw: List[Float32],
    dual_out: MutPointer[Float32, out_origin],
    mut trace: IdentityTrace,
    waddr: Int = 0,
) raises -> Int:
    """KRR_IDN_PTR_IN: `kernel_ridge_fit_host` with X and y read from the
    caller's addresses on the device (`_upload_checked`: shape refused on
    the host, NaN / infinity on the device, X before y as the list route
    walks them) and `dual_coef_` copied into `dual_out` (n_samples x
    n_targets). The same launches in the same order as the list route at
    its default scheduling and no sabotage; the weighted arm scales on the
    device (`krr_scale_rows_kernel`, the host loop's line). Returns `info`
    (0; a failed factorization raises DEVIATION 1662's refusal).

    `waddr` (cpu2-l6-bindings): nonzero is the caller's n float64 sample
    weights; the sqrt factors are formed and refused on the device
    (`_krr_sqrt_weights_dev`) and `sw` must then be empty."""
    if xaddr == 0 or yaddr == 0:
        raise Error("kernel_ridge_fit: null X or y address")
    var ctx = _family_ctx()
    var xa = _upload_checked(ctx, xaddr, n_samples, n_features, "kernel_ridge X")
    var dy = _upload_checked(ctx, yaddr, n_samples, n_targets, "kernel_ridge y")
    var precomputed = kp.kernel == KM_KERNEL_PRECOMPUTED
    if precomputed:
        if n_features != n_samples:
            raise Error(
                "kernel_ridge_fit_host: kernel='precomputed' needs a square"
                " kernel matrix, got " + String(n_samples) + " x "
                + String(n_features)
            )
    else:
        km_validate_kernel_params(kp, "kernel_ridge")
    if waddr != 0 and len(sw) > 0:
        raise Error("kernel_ridge_fit: weights given twice")
    var weighted = len(sw) > 0 or waddr != 0
    if len(sw) > 0:
        krr_validate_weights(sw, n_samples)
    var dsw: DeviceBuffer[DType.float32]
    if waddr != 0:
        dsw = _krr_sqrt_weights_dev(ctx, waddr, n_samples)
    elif weighted:
        dsw = _upload(ctx, sw)
    else:
        dsw = ctx.enqueue_create_buffer[DType.float32](1)
    _krr_validate_alpha(alpha)

    # lane/review-fixes: the pointer route honors KRR_IDN_DEV_SCALE's _OFF
    # arm too (the host loop on the downloaded y, the list route's old form).
    # cpu2-l6-bindings: that arm needs the factors on the host; with
    # `waddr` they were formed on the device, so it downloads them.
    var sw_h = List[Float32]()
    comptime if not KRR_IDN_DEV_SCALE:
        if waddr != 0:
            sw_h = _download(ctx, dsw, n_samples)
        else:
            sw_h = sw.copy()
    comptime if KRR_IDN_DEV_SCALE:
        if weighted:
            _krr_scale_rows_dev(ctx, dy, dsw, n_samples, n_targets)
    else:
        if weighted:
            dy = _upload(ctx, krr_scale_rows(_download(ctx, dy, n_samples * n_targets), sw_h, n_samples, n_targets))
    trace.record_device(ctx, "krr.input", xa, n_samples * n_features)

    var dk = ctx.enqueue_create_buffer[DType.float32](n_samples * n_samples)
    var na = ctx.enqueue_create_buffer[DType.float32](n_samples)
    var nb = ctx.enqueue_create_buffer[DType.float32](n_samples)
    var kws = ctx.enqueue_create_buffer[DType.float32](
        km_kernel_workspace_floats(n_samples, n_samples, n_features)
    )
    var cws = ctx.enqueue_create_buffer[DType.float32](
        kernel_ridge_workspace_floats(n_samples)
    )
    ctx.synchronize()

    if precomputed:
        ctx.enqueue_copy(dst_buf=dk, src_buf=xa)
    else:
        km_kernel_matrix(
            ctx, kp, dk, xa, xa, n_samples, n_samples, n_features,
            na, nb, kws, KM_EPILOGUE_TPB, KMSAB_NONE, True,
        )
    trace.record_device(ctx, "krr.kernel", dk, n_samples * n_samples)
    if weighted:
        var cells = n_samples * n_samples
        ctx.enqueue_function[krr_weight_kernel](
            dk.unsafe_ptr(), dsw.unsafe_ptr(), Int32(n_samples),
            grid_dim=((cells + KRR_WEIGHT_TPB - 1) // KRR_WEIGHT_TPB, 1, 1),
            block_dim=(KRR_WEIGHT_TPB, 1, 1),
        )
        trace.record_device(ctx, "krr.weighted", dk, cells)

    var info = kernel_ridge_solve(
        ctx, dk, dy, cws, n_samples, n_targets, alpha, trace,
        CHOL_PANEL_TPB, CHOL_ELEM_TPB, CHOL_SOLVE_TPB, KRR_RIDGE_TPB, KMSAB_NONE,
    )
    if info != 0:
        ctx.synchronize()
        raise Error(_krr_not_pd_message(info, kp.kernel, n_samples))
    comptime if KRR_IDN_DEV_SCALE:
        if weighted:
            _krr_scale_rows_dev(ctx, dy, dsw, n_samples, n_targets)
        _download_into(ctx, dy, dual_out, n_samples * n_targets)
    else:
        if weighted:
            var dual = krr_scale_rows(_download(ctx, dy, n_samples * n_targets), sw_h, n_samples, n_targets)
            for i in range(n_samples * n_targets):
                dual_out.unsafe_store(i, dual[i])
        else:
            _download_into(ctx, dy, dual_out, n_samples * n_targets)
    _ = dsw^
    _ = xa^
    _ = dy^
    _ = dk^
    _ = na^
    _ = nb^
    _ = kws^
    _ = cws^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return 0


def kernel_ridge_predict_ptr_into[out_origin: MutOrigin, //](
    xfit_addr: Int,
    dual_addr: Int,
    xnew_addr: Int,
    n: Int,
    d: Int,
    t: Int,
    kp: KernelParams,
    n_query: Int,
    output: MutPointer[Float32, out_origin],
    mut trace: IdentityTrace,
) raises:
    """KRR_IDN_PTR_IN: `kernel_ridge_predict_host` with the fit's X and dual
    and the query X read from the caller's addresses on the device (the
    query scanned there, `_upload_checked`; the fitted arrays unchecked as
    on the list route) and the n_query x t predictions copied into
    `output`. The same launches as the list route."""
    if xfit_addr == 0 or dual_addr == 0 or xnew_addr == 0:
        raise Error("kernel_ridge_predict: null X_fit, dual or X address")
    if n <= 0 or d <= 0 or t <= 0:
        raise Error(
            "kernel_ridge_predict: the model needs positive n, d and n_targets, got "
            + String(n) + ", " + String(d) + ", " + String(t)
        )
    var ctx = _family_ctx()
    var dq = _upload_checked(ctx, xnew_addr, n_query, d, "predict X")
    var dfit = _upload_ptr(ctx, xfit_addr, n * d)
    var ddual = _upload_ptr(ctx, dual_addr, n * t)
    var dk = ctx.enqueue_create_buffer[DType.float32](n_query * n)
    var na = ctx.enqueue_create_buffer[DType.float32](n_query)
    var nb = ctx.enqueue_create_buffer[DType.float32](n)
    var kws = ctx.enqueue_create_buffer[DType.float32](
        km_kernel_workspace_floats(n_query, n, d)
    )
    var dpred = ctx.enqueue_create_buffer[DType.float32](n_query * t)
    var gws = ctx.enqueue_create_buffer[DType.float32](
        identical_gemm_workspace_max_floats(n_query, t, n)
    )
    ctx.synchronize()

    if kp.kernel == KM_KERNEL_PRECOMPUTED:
        # X IS the n_query x n_samples cross-kernel matrix.
        ctx.enqueue_copy(dst_buf=dk, src_buf=dq)
    else:
        km_kernel_matrix(
            ctx, kp, dk, dq, dfit, n_query, n, d, na, nb, kws, KM_EPILOGUE_TPB, KMSAB_NONE
        )
    trace.record_device(ctx, "krr.cross_kernel", dk, n_query * n)
    identical_gemm_into(ctx, dpred, dk, ddual, gws, n_query, t, n, OP_NN)
    trace.record_device(ctx, "krr.predictions", dpred, n_query * t)
    _download_into(ctx, dpred, output, n_query * t)
    _ = dq^
    _ = dfit^
    _ = ddual^
    _ = dk^
    _ = na^
    _ = nb^
    _ = kws^
    _ = dpred^
    _ = gws^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^


def kernel_ridge_primal_weights(
    model: KernelRidgeModel, target: Int
) raises -> List[Float32]:
    """`w = X^T dual[:, target]`, the PRIMAL weight vector a LINEAR kernel's
    dual solution corresponds to. `n_features` of them.

    NOT AN ESTIMATOR METHOD -- scikit-learn does not expose it and neither
    does cuML -- and it is here because it is what
    `check_kernel_ridge_planted_linear` compares against a plant. HOST,
    float32, ascending, `identical_mul_add` at every seam, which makes it a
    small pinned reduction rather than a convenience.

    MEANINGLESS FOR A NON-LINEAR KERNEL and refused for one by name, because
    the correspondence is a property of the linear kernel and returning a
    number for an RBF fit would invite someone to interpret it.
    """
    if model.kernel != KM_KERNEL_LINEAR:
        raise Error(
            "kernel_ridge_primal_weights: only a LINEAR kernel has primal"
            " weights in the input space; this model's kernel is "
            + km_kernel_name(model.kernel)
            + ". A kernel method's parameter vector lives in the feature"
            " space, which for every other kernel here is not the input"
            " space and for the RBF kernel is infinite dimensional"
        )
    if target < 0 or target >= model.n_targets:
        raise Error(
            "kernel_ridge_primal_weights: target "
            + String(target)
            + " is outside [0, "
            + String(model.n_targets)
            + ")"
        )
    from checks.numerics import identical_mul_add

    var out = List[Float32]()
    for c in range(model.n_features):
        var acc = Float32(0.0)
        for i in range(model.n_samples):
            acc = ftz(
                identical_mul_add(
                    ftz(model.x_fit[i * model.n_features + c]),
                    ftz(model.dual_coef[i * model.n_targets + target]),
                    acc,
                )
            )
        out.append(ftz(acc))
    return out^


# ===========================================================================
# Nystroem
# ===========================================================================


def scale_columns_kernel(
    z_out: MutPointer[Float32, MutAnyOrigin],
    q_in: MutPointer[Float32, MutAnyOrigin],
    sqrt_s: MutPointer[Float32, MutAnyOrigin],
    q_dim_in: Int32,
):
    """`U / xp.sqrt(S)` (`kernel_approximation.py:1070`), one thread per cell.

    **DEVIATION 1689: A DIVIDE, NEVER A RECIPROCAL TIMES.** The obvious
    optimization is to precompute `1 / sqrt(s_k)` once per column and
    multiply, which is one divide instead of `q` per column. It is refused
    for the reason `cholesky/`'s DEVIATION 1643 refuses the same shape:
    a reciprocal-then-multiply is TWO roundings where a divide is ONE, so it
    is a different answer, and RAFT's own `getDiagonalInverseMatrix`
    (`matrix/detail/matrix.cuh:283-295`) is the exact construction that
    lane declined to put on an identity path. It is also what sklearn
    literally writes: `U / xp.sqrt(S)`, a division.

    `sqrt_s` is the CLIPPED square root, computed once on the host through
    `identical_sqrt` and uploaded, so no thread recomputes a square root.

    Eigenvector `k` is COLUMN `k` of a row-major `q x q` matrix, so cell
    `(i, k)` is at `i * q + k` and the divisor depends on `k` alone.
    """
    var q = Int(q_dim_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= q * q:
        return
    var k = t % q
    z_out.unsafe_store(
        t, ftz(identical_div(ftz(q_in.unsafe_load(t)), ftz(sqrt_s.unsafe_load(k))))
    )


@fieldwise_init
struct NystroemModel(Movable):
    """`sklearn.kernel_approximation.Nystroem` after `fit`, plus the two
    intermediate stages theirs discards."""

    var components: List[Float32]
    """`components_`, `n_components x n_features` row-major: the sampled
    training rows themselves."""

    var component_indices: List[Int32]
    """`component_indices_`, the row ids, IN RANK ORDER. Theirs keeps this
    too (`kernel_approximation.py:1073`) and it is what makes a fit
    reproducible from its own record."""

    var normalization: List[Float32]
    """`normalization_`, `n_components x n_components` row-major,
    `Q diag(s^{-1/2}) V` with `V = diag(sign(lambda)) Q^T`, which is
    `Q diag(s^{-1/2}) Q^T` when no eigenvalue is negative. **NOT bitwise
    symmetric** -- see DEVIATION 1674 and `nystroem_transform_host`."""

    var eigenvalues: List[Float32]
    """The SINGULAR VALUES `|lambda|`, DESCENDING, clipped at 1e-12
    (`_singular_value_f32`). Theirs discards these; see this file's header,
    point 3."""

    var eigenvectors: List[Float32]
    """`q x q` row-major, eigenvector `c` in COLUMN `c`, sign-flipped and
    permuted into the eigenvalue order. Theirs discards these too."""

    var n_components: Int
    var n_features: Int
    var kernel: Int
    var degree: Int
    var gamma: Float64
    var coef0: Float64
    var seed: UInt64
    var sweeps: Int
    """How many Jacobi sweeps the device solver executed. NOT a diagnostic:
    `decomposition/checks/jacobi_eigh_device.mojo`'s DEVIATION BLOCK 3
    measured that a one-ulp difference in the convergence quantity changes
    the SWEEP COUNT, and a different sweep count is a different matrix in
    the fifth decimal. Two runs that disagree here are not comparable below
    this stage at all, exactly as two Cholesky runs with different `nb` are
    not."""


def nystroem_params(model: NystroemModel) -> KernelParams:
    return KernelParams(model.kernel, model.degree, model.gamma, model.coef0)


#: lane/apple-fast-kernel (2026-10-02), FAST on Apple only, the default since
#: the M3 A/B kap2-km-nysrr-taxi (lane apple-fast-gap-kapprox2: nystroem taxi
#: 533 -> 190 ms, kernel_rel_error .04561 -> .04503; `-D
#: MOJOLEARN_KERNEL_FAST_NYS_RR_EIGH_OFF` reverts): it solves the q x q basis kernel's
#: eigenproblem with x_decomp/jacobi_par.mojo's round-robin Jacobi (every
#: round's q / 2 disjoint rotations across the grid, two launches a round,
#: the cyclic kernel's convergence test folded on the grid once a sweep,
#: three words read back, no host loop) in place
#: of `jacobi_eigh_kernel`: ONE block of 256 threads running the q (q - 1) / 2
#: rotations of a sweep one after the other behind two barriers each (at the
#: board's q = 256: 32,640 serial rotations a sweep, up to 15 sweeps) while
#: the rest of the GPU idles. A solve that does not converge in
#: NYS_RR_SWEEPS sweeps leaves `dk` untouched and the cyclic kernel runs.
comptime NYS_RR_EIGH = (_CTX_MODE != _CTX_IDENTICAL and has_apple_gpu_accelerator()
                        and not is_defined["MOJOLEARN_KERNEL_FAST_NYS_RR_EIGH_OFF"]())
comptime NYS_RR_SWEEPS = 30


def _pj_blocks(count: Int) -> Int:
    return (count + PJ_TPB - 1) // PJ_TPB if count > 0 else 1


def _nystroem_rr_eigh(
    ctx: DeviceContext,
    mut dk: DeviceBuffer[DType.float32],
    mut dvec: DeviceBuffer[DType.float32],
    q: Int,
    mut eig_diag: List[Float32],
    mut vecs: List[Float32],
) raises -> Int:
    """The round-robin eigh of the q x q matrix in `dk` (`DevExec._eigh_par`
    of x_decomp/device.mojo, on a copy). Converged: `dvec` holds the sign
    flipped eigenvectors (vector c in COLUMN c), `eig_diag` and `vecs` are
    filled, and the sweep count is returned; otherwise -1 and nothing is
    written but `dvec`."""
    var n = q
    var even_q = n + (n % 2)
    var h = even_q // 2
    var da = ctx.enqueue_create_buffer[DType.float32](n * n)
    var dcs = ctx.enqueue_create_buffer[DType.float32](2 * h)
    var doff = ctx.enqueue_create_buffer[DType.float32](3 * n)
    var hoff = ctx.enqueue_create_host_buffer[DType.float32](3 * n)
    # main's convergence test (x_decomp/device.mojo `_eigh_par_test`): the
    # block trees of the rows' off-diagonal squares and a_kk^2, then the tree
    # past the blocks; a_kk lands in doff[2 n, 3 n)
    var nb_off = max((n + RR_OFF_TPB - 1) // RR_OFF_TPB, 1)
    var dpart = ctx.enqueue_create_buffer[DType.float32](3 * nb_off)
    var dres = ctx.enqueue_create_buffer[DType.float32](3)
    var hres = ctx.enqueue_create_host_buffer[DType.float32](3)
    ctx.enqueue_copy(dst_buf=da, src_buf=dk)
    ctx.enqueue_function[pj_identity_kernel](
        dvec.unsafe_ptr(), Int32(n), grid_dim=_pj_blocks(n * n), block_dim=PJ_TPB
    )
    var tol2 = Float64(JACOBI_TOL) * Float64(JACOBI_TOL)
    var converged = False
    var executed = 0
    var fro_in = Float64(-1.0)
    var fro_now = Float64(0.0)
    for sweep in range(NYS_RR_SWEEPS + 1):
        # a sum of squares is never negative: -1 left in the readback is a
        # dispatch that did not run
        # a sum of squares is never negative: a -1 mark left in the fold is
        # a dispatch that did not run
        dpart.enqueue_fill(Float32(-1.0))
        dres.enqueue_fill(Float32(-1.0))
        ctx.enqueue_function[eigh_par_off_part_kernel](
            da.unsafe_ptr(), doff.unsafe_ptr(), dpart.unsafe_ptr(), Int32(n), grid_dim=nb_off, block_dim=RR_OFF_TPB
        )
        ctx.enqueue_function[eigh_par_off_fold_kernel](
            dpart.unsafe_ptr(), dres.unsafe_ptr(), Int32(nb_off), grid_dim=1, block_dim=RR_OFF_TPB
        )
        ctx.enqueue_copy(dst_ptr=hres.unsafe_ptr(), src_buf=dres)
        ctx.synchronize()
        var off = Float64(hres.unsafe_ptr().unsafe_load(0))
        var dg = Float64(hres.unsafe_ptr().unsafe_load(1))
        if hres.unsafe_ptr().unsafe_load(2) < Float32(0):
            break
        fro_now = off + dg
        if fro_in < 0.0:
            fro_in = fro_now
        if off <= tol2 * fro_now:
            converged = True
            break
        if sweep == NYS_RR_SWEEPS:
            break
        executed += 1
        for rd in range(even_q - 1):
            ctx.enqueue_function[eigh_par_cs_kernel](
                da.unsafe_ptr(), dcs.unsafe_ptr(), Int32(n), Int32(even_q), Int32(rd),
                grid_dim=_pj_blocks(h), block_dim=PJ_TPB,
            )
            ctx.enqueue_function[eigh_par_update_kernel](
                da.unsafe_ptr(), dvec.unsafe_ptr(), dcs.unsafe_ptr(), Int32(n), Int32(even_q), Int32(rd),
                grid_dim=_pj_blocks(h * h + n * h), block_dim=PJ_TPB,
            )
    # J^T A J keeps ||A||_F: a solve that moved it is not an answer
    if converged and not (abs(fro_now - fro_in) <= 1.0e-3 * fro_in):
        converged = False
    var sweeps = -1
    if converged:
        ctx.enqueue_function[sign_flip_kernel](
            dvec.unsafe_ptr(),
            Int32(n),
            grid_dim=(n, 1, 1),
            block_dim=(SIGNFLIP_TPB, 1, 1),
        )
        # the last test's words hold the diagonal of the converged A
        ctx.enqueue_copy(dst_ptr=hoff.unsafe_ptr(), src_buf=doff)
        ctx.synchronize()
        var got = _download(ctx, dvec, n * n)
        for c in range(n):
            eig_diag.append(hoff.unsafe_ptr().unsafe_load(2 * n + c))
        for i in range(n * n):
            vecs.append(got[i])
        sweeps = executed
    ctx.synchronize()
    _ = da^
    _ = dcs^
    _ = doff^
    _ = hoff^
    _ = dpart^
    _ = dres^
    _ = hres^
    return sweeps


#: fam-kernel-gp (2026-10-04), IDENTICAL, ON by default
#: (`-D MOJOLEARN_IDN_NYS_RR_EIGH_OFF` restores the cyclic one-block solver
#: on the device AND in the host column, `km_host_oracle.mojo::KMH_NYS_RR`
#: reads the same define): Nystroem's q x q eigendecomposition is the
#: two-sided Jacobi in the round-robin ordering (x_decomp's `eigh` order:
#: `eigh_par_cs_kernel` + `eigh_par_update_kernel` per round, the q / 2
#: disjoint rotations of a round across the grid, the convergence test
#: folded on the device before every sweep, three words read) in place of
#: `jacobi_eigh_kernel`, ONE block of 256 threads running the q (q - 1) / 2
#: rotations of a sweep one after the other (32,640 serial rotations a sweep
#: at q = 256). BITS CHANGE (another rotation order): NVIDIA, AMD and Apple
#: run these same kernels and the host column runs `x_decomp/rr.mojo::
#: host_eigh_rr`, the same rounds, test and sweep budget (RR_EIGH_SWEEPS).
#: Not converged in the budget raises on every column, as the cyclic one does.
comptime NYS_IDN_RR_EIGH = _CTX_MODE == _CTX_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_NYS_RR_EIGH_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
#: Rounds between waits on Apple (x_decomp/device.mojo `PJ_SYNC_ROUNDS`).
comptime NYS_IDN_RR_SYNC_ROUNDS = 512


def _nystroem_rr_eigh_idn(
    ctx: DeviceContext,
    mut dk: DeviceBuffer[DType.float32],
    mut dvec: DeviceBuffer[DType.float32],
    q: Int,
    sabotage: Int,
    mut eig_diag: List[Float32],
    mut vecs: List[Float32],
    want_host: Bool = True,
) raises -> Int:
    """NYS_IDN_RR_EIGH: the round-robin eigh of the q x q matrix in `dk`,
    consumed in place (x_decomp/device.mojo `_eigh_par_on`'s rounds, test and
    budget; `host_eigh_rr` is the host column's). `dvec` gets the sign
    flipped eigenvectors (vector c in COLUMN c); eigenvalue c is appended to
    `eig_diag` and the q x q vectors to `vecs`. Returns the sweeps run;
    raises when the budget does not converge."""
    var n = q
    var even_q = n + (n % 2)
    var h = even_q // 2
    var dcs = ctx.enqueue_create_buffer[DType.float32](2 * h)
    var doff = ctx.enqueue_create_buffer[DType.float32](3 * n)
    var hoff = ctx.enqueue_create_host_buffer[DType.float32](3 * n)
    var nb_off = max((n + RR_OFF_TPB - 1) // RR_OFF_TPB, 1)
    var dpart = ctx.enqueue_create_buffer[DType.float32](3 * nb_off)
    var dres = ctx.enqueue_create_buffer[DType.float32](3)
    var hres = ctx.enqueue_create_host_buffer[DType.float32](3)
    ctx.enqueue_function[pj_identity_kernel](
        dvec.unsafe_ptr(), Int32(n), grid_dim=_pj_blocks(n * n), block_dim=PJ_TPB
    )
    var tol = Float32(JACOBI_TOL)
    var converged = False
    var executed = 0
    var fro_in = Float32(-1.0)
    var fro_now = Float32(0.0)
    var off_last = Float32(0.0)
    for sweep in range(RR_EIGH_SWEEPS + 1):
        # a sum of squares is never negative: a -1 mark left in the fold is
        # a dispatch that did not run
        enqueue_fill(ctx, dpart, Float32(-1.0))
        enqueue_fill(ctx, dres, Float32(-1.0))
        ctx.enqueue_function[eigh_par_off_part_kernel](
            dk.unsafe_ptr(), doff.unsafe_ptr(), dpart.unsafe_ptr(), Int32(n), grid_dim=nb_off, block_dim=RR_OFF_TPB
        )
        ctx.enqueue_function[eigh_par_off_fold_kernel](
            dpart.unsafe_ptr(), dres.unsafe_ptr(), Int32(nb_off), grid_dim=1, block_dim=RR_OFF_TPB
        )
        ctx.enqueue_copy(dst_ptr=hres.unsafe_ptr(), src_buf=dres)
        ctx.synchronize()
        var off = hres.unsafe_ptr().unsafe_load(0)
        var dg = hres.unsafe_ptr().unsafe_load(1)
        if not (hres.unsafe_ptr().unsafe_load(2) >= Float32(0.0)):
            raise Error(
                "nystroem_fit_host: a block of the round-robin Jacobi's convergence test did not"
                " run (its mark is still -1): a launch failure, not a convergence failure"
            )
        off_last = off
        fro_now = ftz(off + dg)
        if fro_in < Float32(0.0):
            fro_in = fro_now
        if rr_converged(off, dg, tol):
            converged = True
            break
        if sweep == RR_EIGH_SWEEPS:
            break
        executed += 1
        for rd in range(even_q - 1):
            ctx.enqueue_function[eigh_par_cs_kernel](
                dk.unsafe_ptr(), dcs.unsafe_ptr(), Int32(n), Int32(even_q), Int32(rd),
                grid_dim=_pj_blocks(h), block_dim=PJ_TPB,
            )
            ctx.enqueue_function[eigh_par_update_kernel](
                dk.unsafe_ptr(), dvec.unsafe_ptr(), dcs.unsafe_ptr(), Int32(n), Int32(even_q), Int32(rd),
                grid_dim=_pj_blocks(h * h + n * h), block_dim=PJ_TPB,
            )
            comptime if has_apple_gpu_accelerator():
                if rd % NYS_IDN_RR_SYNC_ROUNDS == NYS_IDN_RR_SYNC_ROUNDS - 1:
                    ctx.synchronize()
    # J^T A J keeps ||A||_F: a solve that moved it is not an answer
    if converged and not rr_fro_kept(fro_in, fro_now):
        converged = False
    if not converged:
        raise Error(
            "nystroem_fit_host: the device Jacobi did not converge in "
            + String(RR_EIGH_SWEEPS)
            + " sweeps at n_components = "
            + String(q)
            + "; the off-diagonal mass is still "
            + String(off_last)
            + " of "
            + String(fro_now)
            + " against a tolerance of "
            + String(JACOBI_TOL)
            + ". An unconverged eigendecomposition returned as if it were"
            " one is a wrong answer with no error"
        )
    if sabotage != KMSAB_NO_SIGN_FLIP:
        ctx.enqueue_function[sign_flip_kernel](
            dvec.unsafe_ptr(),
            Int32(n),
            grid_dim=(n, 1, 1),
            block_dim=(SIGNFLIP_TPB, 1, 1),
        )
    # the last test's kernel left the diagonal of the converged A in
    # doff[2 n, 3 n)
    if want_host:
        ctx.enqueue_copy(dst_ptr=hoff.unsafe_ptr(), src_buf=doff)
        ctx.synchronize()
        var got = _download(ctx, dvec, n * n)
        for c in range(n):
            eig_diag.append(hoff.unsafe_ptr().unsafe_load(2 * n + c))
        for i in range(n * n):
            vecs.append(got[i])
    else:
        # NYS_IDN_DEV_ORDER: the caller orders the pairs where they lie
        # (`dk`'s diagonal, `dvec`); nothing is read back here.
        ctx.synchronize()
    _ = dcs^
    _ = doff^
    _ = hoff^
    _ = dpart^
    _ = dres^
    _ = hres^
    return executed


#: fam2-kernel-gp (2026-10-04), IDENTICAL, CANDIDATE ARM, OFF by default
#: (`-D MOJOLEARN_IDN_NYS_RR_DEV_STOP` turns it on; `MOJOLEARN_IDN_ALL_OFF`
#: turns it off): the round-robin eigh's convergence test is DECIDED on the
#: device. A one-thread gate kernel reads the folded sums after every test
#: and keeps the solve's state in five device words (done, the entry and
#: current Frobenius sums, the last off-diagonal sum, the sweeps run); the
#: rotation and update kernels of every later round return at once when the
#: state says done. The host reads the state once every NYS_RR_STOP_BATCH
#: sweeps instead of three words and a wait before every sweep. The
#: rotations that run are the default arm's, in its order: no bit moves and
#: the host column is unchanged. The cost is the empty launches of up to
#: NYS_RR_STOP_BATCH - 1 sweeps after convergence, which is why this is an
#: arm to time (`-D MOJOLEARN_IDN_NYS_RR_DEV_STOP_B4`: batches of 4, not 2).
comptime NYS_IDN_RR_DEV_STOP = (
    NYS_IDN_RR_EIGH
    and is_defined["MOJOLEARN_IDN_NYS_RR_DEV_STOP"]()
)
comptime NYS_RR_STOP_BATCH = 4 if is_defined["MOJOLEARN_IDN_NYS_RR_DEV_STOP_B4"]() else 2
#: `nys_rr_gate_kernel`'s state words.
comptime NYS_RR_ST_DONE = 0
comptime NYS_RR_ST_FRO_IN = 1
comptime NYS_RR_ST_FRO_NOW = 2
comptime NYS_RR_ST_OFF = 3
comptime NYS_RR_ST_SWEEPS = 4
comptime NYS_RR_ST_WORDS = 5


def nys_rr_gate_kernel(res: F32Ptr, st: F32Ptr, tol: Float32, last_in: Int32):
    """One thread: `_nystroem_rr_eigh_idn`'s host statements after a test.
    st[DONE]: 0 running, 1 converged, 2 the budget ran out, 3 a block of the
    test did not run. Does nothing once the solve is done."""
    if Int(block_idx.x) != 0 or Int(thread_idx.x) != 0:
        return
    if st.unsafe_load(NYS_RR_ST_DONE) != Float32(0.0):
        return
    var off = res.unsafe_load(0)
    var dg = res.unsafe_load(1)
    if not (res.unsafe_load(2) >= Float32(0.0)):
        st.unsafe_store(NYS_RR_ST_DONE, Float32(3.0))
        return
    st.unsafe_store(NYS_RR_ST_OFF, off)
    var fro = ftz(off + dg)
    st.unsafe_store(NYS_RR_ST_FRO_NOW, fro)
    if st.unsafe_load(NYS_RR_ST_FRO_IN) < Float32(0.0):
        st.unsafe_store(NYS_RR_ST_FRO_IN, fro)
    if rr_converged(off, dg, tol):
        st.unsafe_store(NYS_RR_ST_DONE, Float32(1.0))
        return
    if last_in != Int32(0):
        st.unsafe_store(NYS_RR_ST_DONE, Float32(2.0))
        return
    st.unsafe_store(NYS_RR_ST_SWEEPS, st.unsafe_load(NYS_RR_ST_SWEEPS) + Float32(1.0))


def nys_rr_cs_gated_kernel(a: F32Ptr, cs: F32Ptr, st: F32Ptr, n_in: Int32, m_in: Int32, round_in: Int32):
    """`eigh_par_cs_kernel` while the solve is running (st[DONE] == 0)."""
    if st.unsafe_load(NYS_RR_ST_DONE) != Float32(0.0):
        return
    var m = Int(m_in)
    var b = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if b < m // 2:
        var got = rr_cs(a, Int(n_in), m, Int(round_in), b)
        cs.unsafe_store(2 * b, got[0])
        cs.unsafe_store(2 * b + 1, got[1])


def nys_rr_update_gated_kernel(
    a: F32Ptr, v: F32Ptr, cs: F32Ptr, st: F32Ptr, n_in: Int32, m_in: Int32, round_in: Int32
):
    """`eigh_par_update_kernel` while the solve is running (st[DONE] == 0)."""
    if st.unsafe_load(NYS_RR_ST_DONE) != Float32(0.0):
        return
    var n = Int(n_in)
    var m = Int(m_in)
    var h = m // 2
    var r = Int(round_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < h * h:
        var i = t // h
        var j = t - i * h
        if i <= j:
            rr_block(a, cs, n, m, r, i, j)
    elif t < h * h + n * h:
        var u = t - h * h
        var k = u // h
        rr_vrow(v, cs, n, m, r, k, u - k * h)


def _nystroem_rr_eigh_idn_devstop(
    ctx: DeviceContext,
    mut dk: DeviceBuffer[DType.float32],
    mut dvec: DeviceBuffer[DType.float32],
    q: Int,
    sabotage: Int,
    mut eig_diag: List[Float32],
    mut vecs: List[Float32],
    want_host: Bool = True,
) raises -> Int:
    """NYS_IDN_RR_DEV_STOP: `_nystroem_rr_eigh_idn` with the stop decided on
    the device (see the define's comment). The same outputs."""
    var n = q
    var even_q = n + (n % 2)
    var h = even_q // 2
    var dcs = ctx.enqueue_create_buffer[DType.float32](2 * h)
    var doff = ctx.enqueue_create_buffer[DType.float32](3 * n)
    var hoff = ctx.enqueue_create_host_buffer[DType.float32](3 * n)
    var nb_off = max((n + RR_OFF_TPB - 1) // RR_OFF_TPB, 1)
    var dpart = ctx.enqueue_create_buffer[DType.float32](3 * nb_off)
    var dres = ctx.enqueue_create_buffer[DType.float32](3)
    var dst = ctx.enqueue_create_buffer[DType.float32](NYS_RR_ST_WORDS)
    var hst = ctx.enqueue_create_host_buffer[DType.float32](NYS_RR_ST_WORDS)
    # done = 0, fro_in = -1 (unset), fro_now, off, sweeps = 0
    hst.unsafe_ptr().unsafe_store(NYS_RR_ST_DONE, Float32(0.0))
    hst.unsafe_ptr().unsafe_store(NYS_RR_ST_FRO_IN, Float32(-1.0))
    hst.unsafe_ptr().unsafe_store(NYS_RR_ST_FRO_NOW, Float32(0.0))
    hst.unsafe_ptr().unsafe_store(NYS_RR_ST_OFF, Float32(0.0))
    hst.unsafe_ptr().unsafe_store(NYS_RR_ST_SWEEPS, Float32(0.0))
    ctx.enqueue_copy(dst_buf=dst, src_ptr=hst.unsafe_ptr())
    ctx.enqueue_function[pj_identity_kernel](
        dvec.unsafe_ptr(), Int32(n), grid_dim=_pj_blocks(n * n), block_dim=PJ_TPB
    )
    ctx.synchronize()
    var tol = Float32(JACOBI_TOL)
    var done = Float32(0.0)
    for sweep in range(RR_EIGH_SWEEPS + 1):
        enqueue_fill(ctx, dpart, Float32(-1.0))
        enqueue_fill(ctx, dres, Float32(-1.0))
        ctx.enqueue_function[eigh_par_off_part_kernel](
            dk.unsafe_ptr(), doff.unsafe_ptr(), dpart.unsafe_ptr(), Int32(n), grid_dim=nb_off, block_dim=RR_OFF_TPB
        )
        ctx.enqueue_function[eigh_par_off_fold_kernel](
            dpart.unsafe_ptr(), dres.unsafe_ptr(), Int32(nb_off), grid_dim=1, block_dim=RR_OFF_TPB
        )
        ctx.enqueue_function[nys_rr_gate_kernel](
            dres.unsafe_ptr(), dst.unsafe_ptr(), tol, Int32(1) if sweep == RR_EIGH_SWEEPS else Int32(0),
            grid_dim=1, block_dim=1,
        )
        if sweep % NYS_RR_STOP_BATCH == NYS_RR_STOP_BATCH - 1 or sweep == RR_EIGH_SWEEPS:
            ctx.enqueue_copy(dst_ptr=hst.unsafe_ptr(), src_buf=dst)
            ctx.synchronize()
            done = hst.unsafe_ptr().unsafe_load(NYS_RR_ST_DONE)
            if done != Float32(0.0):
                break
        if sweep == RR_EIGH_SWEEPS:
            break
        for rd in range(even_q - 1):
            ctx.enqueue_function[nys_rr_cs_gated_kernel](
                dk.unsafe_ptr(), dcs.unsafe_ptr(), dst.unsafe_ptr(), Int32(n), Int32(even_q), Int32(rd),
                grid_dim=_pj_blocks(h), block_dim=PJ_TPB,
            )
            ctx.enqueue_function[nys_rr_update_gated_kernel](
                dk.unsafe_ptr(), dvec.unsafe_ptr(), dcs.unsafe_ptr(), dst.unsafe_ptr(),
                Int32(n), Int32(even_q), Int32(rd),
                grid_dim=_pj_blocks(h * h + n * h), block_dim=PJ_TPB,
            )
            comptime if has_apple_gpu_accelerator():
                if rd % NYS_IDN_RR_SYNC_ROUNDS == NYS_IDN_RR_SYNC_ROUNDS - 1:
                    ctx.synchronize()
    var fro_in = hst.unsafe_ptr().unsafe_load(NYS_RR_ST_FRO_IN)
    var fro_now = hst.unsafe_ptr().unsafe_load(NYS_RR_ST_FRO_NOW)
    var off_last = hst.unsafe_ptr().unsafe_load(NYS_RR_ST_OFF)
    var executed = Int(hst.unsafe_ptr().unsafe_load(NYS_RR_ST_SWEEPS))
    if done == Float32(3.0):
        raise Error(
            "nystroem_fit_host: a block of the round-robin Jacobi's convergence test did not"
            " run (its mark is still -1): a launch failure, not a convergence failure"
        )
    var converged = done == Float32(1.0)
    # J^T A J keeps ||A||_F: a solve that moved it is not an answer
    if converged and not rr_fro_kept(fro_in, fro_now):
        converged = False
    if not converged:
        raise Error(
            "nystroem_fit_host: the device Jacobi did not converge in "
            + String(RR_EIGH_SWEEPS)
            + " sweeps at n_components = "
            + String(q)
            + "; the off-diagonal mass is still "
            + String(off_last)
            + " of "
            + String(fro_now)
            + " against a tolerance of "
            + String(JACOBI_TOL)
            + ". An unconverged eigendecomposition returned as if it were"
            " one is a wrong answer with no error"
        )
    if sabotage != KMSAB_NO_SIGN_FLIP:
        ctx.enqueue_function[sign_flip_kernel](
            dvec.unsafe_ptr(),
            Int32(n),
            grid_dim=(n, 1, 1),
            block_dim=(SIGNFLIP_TPB, 1, 1),
        )
    if want_host:
        # the last test's kernel left the diagonal of the converged A in
        # doff[2 n, 3 n)
        ctx.enqueue_copy(dst_ptr=hoff.unsafe_ptr(), src_buf=doff)
        ctx.synchronize()
        var got = _download(ctx, dvec, n * n)
        for c in range(n):
            eig_diag.append(hoff.unsafe_ptr().unsafe_load(2 * n + c))
        for i in range(n * n):
            vecs.append(got[i])
    else:
        ctx.synchronize()
    _ = dcs^
    _ = doff^
    _ = hoff^
    _ = dpart^
    _ = dres^
    _ = dst^
    _ = hst^
    return executed


def _nystroem_device_eigh(
    ctx: DeviceContext,
    mut dk: DeviceBuffer[DType.float32],
    mut dvec: DeviceBuffer[DType.float32],
    mut dinfo: DeviceBuffer[DType.float32],
    q: Int,
    sabotage: Int,
    mut trace: IdentityTrace,
    mut eig_diag: List[Float32],
    mut vecs: List[Float32],
    want_host: Bool = True,
) raises -> Int:
    """The eigendecomposition on the device, through decomposition/: `dk` is
    consumed (its diagonal becomes the eigenvalues), `dvec` gets the sign
    flipped eigenvectors. Appends eigenvalue c to `eig_diag` and the q x q
    eigenvectors (vector c in COLUMN c) to `vecs`; returns the sweeps."""
    comptime if NYS_RR_EIGH:
        if sabotage == KMSAB_NONE:
            var got = _nystroem_rr_eigh(ctx, dk, dvec, q, eig_diag, vecs)
            if got >= 0:
                trace.record_device(ctx, "nys.eigenvectors_flipped", dvec, q * q)
                return got
    comptime if NYS_IDN_RR_DEV_STOP:
        var ran_ds = _nystroem_rr_eigh_idn_devstop(ctx, dk, dvec, q, sabotage, eig_diag, vecs, want_host)
        trace.record_device(ctx, "nys.eigenvectors_flipped", dvec, q * q)
        return ran_ds
    comptime if NYS_IDN_RR_EIGH:
        var ran = _nystroem_rr_eigh_idn(ctx, dk, dvec, q, sabotage, eig_diag, vecs, want_host)
        trace.record_device(ctx, "nys.eigenvectors_flipped", dvec, q * q)
        return ran
    ctx.enqueue_function[jacobi_eigh_kernel[JACOBI_ROT_TPB]](
        dk.unsafe_ptr(),
        dvec.unsafe_ptr(),
        dinfo.unsafe_ptr(),
        Int32(q),
        Int32(JACOBI_SWEEPS),
        Float32(JACOBI_TOL),
        grid_dim=(1, 1, 1),
        block_dim=(JACOBI_ROT_TPB, 1, 1),
    )
    if sabotage != KMSAB_NO_SIGN_FLIP:
        ctx.enqueue_function[sign_flip_kernel](
            dvec.unsafe_ptr(),
            Int32(q),
            grid_dim=(q, 1, 1),
            block_dim=(SIGNFLIP_TPB, 1, 1),
        )
    ctx.synchronize()
    trace.record_device(ctx, "nys.eigenvectors_flipped", dvec, q * q)

    var info_h = _download(ctx, dinfo, 3)
    if info_h[0] == Float32(0.0):
        # `eig_and_truncate`'s refusal, for the same reason it gives: their
        # DEFAULT eigen arm (`eigDC` -> cuSOLVER `syevd`) aborts on a
        # non-zero `dev_info`, and their JACOBI arm silently does not
        # (`raft/linalg/detail/eig.cuh:310`, `executed_sweeps` fetched and
        # never read). We follow the default arm.
        raise Error(
            "nystroem_fit_host: the device Jacobi did not converge in "
            + String(JACOBI_SWEEPS)
            + " sweeps at n_components = "
            + String(q)
            + "; ||offdiag(A)||_F / ||A||_F is still "
            + String(info_h[1])
            + " against a tolerance of "
            + String(JACOBI_TOL)
            + ". An unconverged eigendecomposition returned as if it were"
            " one is a wrong answer with no error. The closure is a larger"
            " sweep budget, which is decomposition/'s parameter and not"
            " this lane's to change"
        )
    if want_host:
        var raw = _download(ctx, dk, q * q)
        var got = _download(ctx, dvec, q * q)
        for c in range(q):
            eig_diag.append(raw[c * q + c])
        for i in range(q * q):
            vecs.append(got[i])
    return Int(info_h[2])


#: fam2-kernel-gp (2026-10-04), IDENTICAL, ON by default
#: (`-D MOJOLEARN_IDN_NYS_DEV_ORDER_OFF` restores the host epilogue): after
#: the eigendecomposition the fit ordered the q eigenpairs with a host
#: selection sort (q^2 compares on one thread), clipped and square-rooted
#: them on the host, permuted the q x q eigenvectors on the host and
#: uploaded them again (plus a signed copy when an eigenvalue is negative).
#: Now `nys_order_kernel` ranks every eigenvalue by counting (one thread per
#: eigenvalue; the selection sort's total order: |lambda| descending, index
#: ascending on a tie) and writes the clipped value, its `identical_sqrt`
#: and its sign at its rank, and `nys_permute_kernel` writes the ordered and
#: the column-signed eigenvectors, all from the resident `dk` / `dvec`. The
#: eigenvectors are read back once, ordered, for the model. The same compare,
#: clip, square root and negation per value: no bit moves, and the host
#: column (`km_host_oracle.mojo`) is unchanged. A sabotage arm keeps the
#: host epilogue (its arms live there).
comptime NYS_IDN_DEV_ORDER = _CTX_MODE == _CTX_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_NYS_DEV_ORDER_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime NYS_ORDER_TPB = 256


@always_inline
def _nys_abs_bits(lam: Float32) -> Float32:
    """`_singular_value_f32`'s line, inlined for the kernels."""
    return bitcast[DType.float32](bitcast[DType.uint32](lam) & UInt32(0x7FFFFFFF))


def nys_order_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    rank: MutPointer[Int32, MutAnyOrigin],
    values: MutPointer[Float32, MutAnyOrigin],
    sqrt_s: MutPointer[Float32, MutAnyOrigin],
    signs: MutPointer[Float32, MutAnyOrigin],
    q_in: Int32,
    clip: Float32,
):
    """One thread per eigenvalue c (the diagonal of the q x q `a`): its rank
    r under `_eigen_order_f32`'s order (the count of eigenvalues whose
    magnitude is larger, or equal at a lower index), then values[r] = the
    clipped magnitude, sqrt_s[r] = ftz(identical_sqrt(values[r])) and
    signs[r] = -1 where the eigenvalue is negative, else 1. The ranks are a
    bijection (the order includes the index), so every slot has one writer."""
    var q = Int(q_in)
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if c >= q:
        return
    var lam = a.unsafe_load(c * q + c)
    var mc = _nys_abs_bits(lam)
    var r = 0
    for j in range(q):
        var mj = _nys_abs_bits(a.unsafe_load(j * q + j))
        if mj > mc or (mj == mc and j < c):
            r += 1
    rank.unsafe_store(c, Int32(r))
    var sv = mc
    if sv < clip:
        sv = clip
    values.unsafe_store(r, sv)
    sqrt_s.unsafe_store(r, ftz(identical_sqrt(sv)))
    signs.unsafe_store(r, Float32(-1.0) if lam < Float32(0.0) else Float32(1.0))


def nys_permute_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    vec: MutPointer[Float32, MutAnyOrigin],
    rank: MutPointer[Int32, MutAnyOrigin],
    q_out: MutPointer[Float32, MutAnyOrigin],
    vt_out: MutPointer[Float32, MutAnyOrigin],
    q_in: Int32,
):
    """One thread per cell (f, src) of the eigenvectors (vector src in
    COLUMN src): q_out[f, rank[src]] = the cell, vt_out[f, rank[src]] = the
    cell negated where eigenvalue src is negative (the host loop's lines)."""
    var q = Int(q_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= q * q:
        return
    var f = t // q
    var src = t - f * q
    var r = Int(rank.unsafe_load(src))
    var e = vec.unsafe_load(t)
    q_out.unsafe_store(f * q + r, e)
    vt_out.unsafe_store(f * q + r, -e if a.unsafe_load(src * q + src) < Float32(0.0) else e)


def nystroem_fit_host(
    x: List[Float32],
    n_samples: Int,
    n_features: Int,
    kp: KernelParams,
    n_components: Int,
    seed: UInt64,
    mut trace: IdentityTrace,
    elem_tpb: Int = KM_EPILOGUE_TPB,
    scale_tpb: Int = KM_TPB,
    sabotage: Int = KMSAB_NONE,
) raises -> NystroemModel:
    """`Nystroem.fit(X)` (`kernel_approximation.py:1032-1074`), step for step.

        inds = rnd.permutation(n_samples)              -> km_basis_indices
        basis = X[inds[:n_components]]
        basis_kernel = pairwise_kernels(basis, ...)    -> km_kernel_matrix
        U, S, V = xp.linalg.svd(basis_kernel)          -> jacobi_eigh_kernel
        S = xp.clip(S, 1e-12, None)                    -> DEVIATION 1670
        self.normalization_ = U / xp.sqrt(S) @ V       -> divide, then OP_NT
        self.components_ = basis
        self.component_indices_ = basis_inds

    DEVIATION 1667 records the SVD-to-eigendecomposition substitution and its
    argument: `decomposition/`'s Jacobi is the only symmetric eigensolver in
    this tree, and cuSOLVER's `gesvd` is closed. For a SYMMETRIC basis kernel
    the SVD is `U = Q`, `S = |lambda|`, `V = diag(sign(lambda)) Q^T`, which
    is `U = Q` and `V = Q^T` only when no eigenvalue is negative; a float32
    Jacobi on a rank deficient kernel returns negative ones, so the magnitude
    and the sign are both carried (`_singular_value_f32`, corrected
    2026-09-14).

    DEVIATION 1668: the eigenvector SIGN convention is
    `decomposition/impl/linalg/detail/pca.mojo::sign_flip_kernel`, RAFT's
    `signFlipKernel`, CALLED. It is NOT reinvented here, and the README's
    reuse table says why at length.
    """
    km_validate_matrix(x, n_samples, n_features, "nystroem X")
    km_validate_kernel_params(kp, "nystroem")

    var q = n_components
    var basis = km_basis_indices(seed, n_samples, q)
    if sabotage == KMSAB_BASIS_FROM_LAUNCH:
        # ARM: a LAUNCH-STRIDED slice instead of the position-mapped rank
        # prefix. Plausible, wrong, and dependent on a scheduling number.
        basis = List[Int32]()
        var stride = elem_tpb // 32
        if stride < 1:
            stride = 1
        for c in range(q):
            basis.append(Int32((c * stride) % n_samples))
    trace.record_list_i32("nys.basis_indices", basis)

    var comp = List[Float32]()
    for c in range(q):
        var srow = Int(basis[c])
        for f in range(n_features):
            comp.append(x[srow * n_features + f])

    var ctx = _family_ctx()
    var ca = _upload(ctx, comp)
    var model = _nystroem_fit_core(
        ctx, ca, comp^, basis^, n_features, kp, q, seed, trace, elem_tpb, scale_tpb, sabotage
    )
    _ = ca^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return model^


#: fam2-kernel-gp (2026-10-04), IDENTICAL, ON by default
#: (`-D MOJOLEARN_IDN_NYS_FIT_PTR_IN_OFF` unregisters the pointer binding, so
#: `kernel_methods.py` takes the list route again): Nystroem fit reads X
#: from the caller's memory straight to the device, scans it for NaN /
#: infinity there (`_upload_checked`) and gathers the basis rows there
#: (`km_gather_rows_kernel`), instead of an owned host copy of X
#: (`read_f32`), a serial host finiteness walk over n x d and a host gather.
#: The same component words reach the same kernels: no bit moves.
comptime NYS_IDN_FIT_PTR_IN = _CTX_MODE == _CTX_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_NYS_FIT_PTR_IN_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)

#: fam2-kernel-gp (2026-10-04), IDENTICAL, ON by default
#: (`-D MOJOLEARN_IDN_NYS_DEV_BASIS_OFF` restores the host draw and radix
#: sort): the pointer fit draws the basis rows on the device
#: (`km_basis_indices_device`: every row's key drawn there, the rows under a
#: threshold compacted and ranked there) instead of n_samples host draws and
#: a host radix sort inside the device fit. Integer only, the same total
#: order: the same rows in the same order, no bit moves. Applies on the
#: pointer route (NYS_IDN_FIT_PTR_IN), where X is resident.
comptime NYS_IDN_DEV_BASIS = _CTX_MODE == _CTX_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_NYS_DEV_BASIS_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)


def nystroem_fit_ptr(
    xaddr: Int,
    n_samples: Int,
    n_features: Int,
    kp: KernelParams,
    n_components: Int,
    seed: UInt64,
    mut trace: IdentityTrace,
) raises -> NystroemModel:
    """NYS_IDN_FIT_PTR_IN: `nystroem_fit_host` with X read from the caller's
    address on the device (`_upload_checked`), the basis rows gathered
    there, and (NYS_IDN_DEV_BASIS) drawn and ranked there. The fit proper is
    `_nystroem_fit_core`, the list route's."""
    if xaddr == 0:
        raise Error("nystroem_fit: null X address")
    var ctx = _family_ctx()
    var dx = _upload_checked(ctx, xaddr, n_samples, n_features, "nystroem X")
    km_validate_kernel_params(kp, "nystroem")
    var q = n_components
    var dbasis = ctx.enqueue_create_buffer[DType.int32](max(q, 1))
    var basis = List[Int32]()
    if NYS_IDN_DEV_BASIS:
        basis = km_basis_indices_device(ctx, seed, n_samples, q, dbasis)
    else:
        basis = km_basis_indices(seed, n_samples, q)
        var hb = ctx.enqueue_create_host_buffer[DType.int32](q)
        for c in range(q):
            hb.unsafe_ptr().unsafe_store(c, basis[c])
        ctx.enqueue_copy(dst_buf=dbasis, src_ptr=hb.unsafe_ptr())
        ctx.synchronize()
        _ = hb^
    trace.record_list_i32("nys.basis_indices", basis)
    var ca = ctx.enqueue_create_buffer[DType.float32](q * n_features)
    ctx.enqueue_function[km_gather_rows_kernel](
        ca.unsafe_ptr(), dx.unsafe_ptr(), dbasis.unsafe_ptr(), Int32(q), Int32(n_features),
        grid_dim=((q * n_features + KM_RF_TPB - 1) // KM_RF_TPB, 1, 1),
        block_dim=(KM_RF_TPB, 1, 1),
    )
    var comp = _download(ctx, ca, q * n_features)
    _ = dx^
    _ = dbasis^
    var model = _nystroem_fit_core(
        ctx, ca, comp^, basis^, n_features, kp, q, seed, trace, KM_EPILOGUE_TPB, KM_TPB, KMSAB_NONE
    )
    _ = ca^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return model^


def _nystroem_fit_core(
    ctx: DeviceContext,
    mut ca: DeviceBuffer[DType.float32],
    var comp: List[Float32],
    var basis: List[Int32],
    n_features: Int,
    kp: KernelParams,
    q: Int,
    seed: UInt64,
    mut trace: IdentityTrace,
    elem_tpb: Int,
    scale_tpb: Int,
    sabotage: Int,
) raises -> NystroemModel:
    """`nystroem_fit_host` from the uploaded components `ca` (q x n_features)
    on: the basis kernel, its eigendecomposition, the order and the
    normalization. `comp` and `basis` are the model's host copies."""
    var dk = ctx.enqueue_create_buffer[DType.float32](q * q)
    var na = ctx.enqueue_create_buffer[DType.float32](q)
    var nb = ctx.enqueue_create_buffer[DType.float32](q)
    var kws = ctx.enqueue_create_buffer[DType.float32](
        km_kernel_workspace_floats(q, q, n_features)
    )
    ctx.synchronize()
    km_kernel_matrix(
        ctx, kp, dk, ca, ca, q, q, n_features, na, nb, kws, elem_tpb, sabotage, True
    )
    ctx.synchronize()
    trace.record_device(ctx, "nys.basis_kernel", dk, q * q)

    # --- the eigendecomposition, on the device, through decomposition/ ---
    var dvec = ctx.enqueue_create_buffer[DType.float32](q * q)
    var dinfo = ctx.enqueue_create_buffer[DType.float32](3)
    ctx.synchronize()
    # ONE BLOCK OF EXACTLY `JACOBI_ROT_TPB` THREADS, THE WIDTH THE KERNEL WAS
    # INSTANTIATED AT. That file's header calls the block dim a CONTRACT
    # rather than a suggestion, and it still is -- what changed with
    # DEVIATION 2680 is WHICH constant it names. The fold slab is
    # `JACOBI_TPB` wide and only the first `JACOBI_TPB` lanes write it, so
    # the launch width is free; but the kernel indexes its rotation loop by
    # exactly `rot_tpb`, so a block of any other size would leave rows
    # unrotated (too few) or run lanes off the end (too many). Passing the
    # same constant in the parameter and in `block_dim` is what makes the
    # two impossible to drift apart. All of its other call sites do the same.
    var sweeps = 0
    var eig_diag = List[Float32]()
    var vecs = List[Float32]()
    # NYS_IDN_DEV_ORDER: the order, clip, square root and permutation run on
    # the device; a sabotage arm keeps the host epilogue.
    var dev_order = NYS_IDN_DEV_ORDER and sabotage == KMSAB_NONE
    sweeps = _nystroem_device_eigh(ctx, dk, dvec, dinfo, q, sabotage, trace, eig_diag, vecs, not dev_order)

    # --- the order and the clip, on the host (DEVIATIONS 1669, 1670, 1688) ---
    # THE SVD'S S, NOT THE EIGENVALUE. See `_singular_value_f32`: sklearn's
    # `S` is `|lambda|` with the sign carried by `V`, so the order, the clip
    # and the square root all read the MAGNITUDE, and a numerically negative
    # eigenvalue negates its column of the right operand below.
    var clip = _eigen_clip_f32()
    var values = List[Float32]()
    var sqrt_s = List[Float32]()
    var v_signs = List[Int32]()
    var any_negative = False
    var vecs_ord = List[Float32]()
    var vt_ord = List[Float32]()
    if not dev_order:
        var values_raw = List[Float32]()
        var mags = List[Float32]()
        for c in range(q):
            values_raw.append(eig_diag[c])
            mags.append(_singular_value_f32(eig_diag[c]))
        var order = _eigen_order_f32(mags, q, sabotage)
        for _ in range(q * q):
            vecs_ord.append(Float32(0.0))
            vt_ord.append(Float32(0.0))
        for c in range(q):
            var src = order[c]
            var s = mags[src]
            if sabotage != KMSAB_NO_EIGEN_CLIP and s < clip:
                s = clip
            values.append(s)
            sqrt_s.append(ftz(identical_sqrt(s)))
            var negative = values_raw[src] < Float32(0.0)
            if negative:
                any_negative = True
                v_signs.append(Int32(-1))
            else:
                v_signs.append(Int32(1))
            for f in range(q):
                var e = vecs[f * q + src]
                vecs_ord[f * q + c] = e
                vt_ord[f * q + c] = -e if negative else e
    # the ordered eigenvectors and the clipped square roots, on the device:
    # uploaded from the host epilogue, or written there (NYS_IDN_DEV_ORDER)
    var dq0 = ctx.enqueue_create_buffer[DType.float32](q * q) if dev_order else _upload(ctx, vecs_ord)
    var dsq = ctx.enqueue_create_buffer[DType.float32](q) if dev_order else _upload(ctx, sqrt_s)
    var dvt_dev = ctx.enqueue_create_buffer[DType.float32](q * q if dev_order else 1)
    if dev_order:
        var drank = ctx.enqueue_create_buffer[DType.int32](q)
        var dvals = ctx.enqueue_create_buffer[DType.float32](q)
        var dsg = ctx.enqueue_create_buffer[DType.float32](q)
        ctx.enqueue_function[nys_order_kernel](
            dk.unsafe_ptr(), drank.unsafe_ptr(), dvals.unsafe_ptr(), dsq.unsafe_ptr(), dsg.unsafe_ptr(),
            Int32(q), clip,
            grid_dim=((q + NYS_ORDER_TPB - 1) // NYS_ORDER_TPB, 1, 1),
            block_dim=(NYS_ORDER_TPB, 1, 1),
        )
        ctx.enqueue_function[nys_permute_kernel](
            dk.unsafe_ptr(), dvec.unsafe_ptr(), drank.unsafe_ptr(), dq0.unsafe_ptr(), dvt_dev.unsafe_ptr(),
            Int32(q),
            grid_dim=((q * q + NYS_ORDER_TPB - 1) // NYS_ORDER_TPB, 1, 1),
            block_dim=(NYS_ORDER_TPB, 1, 1),
        )
        # the model's copies (and the card's): copies, no arithmetic
        values = _download(ctx, dvals, q)
        vecs_ord = _download(ctx, dq0, q * q)
        if trace.enabled:
            sqrt_s = _download(ctx, dsq, q)
            var sg = _download(ctx, dsg, q)
            for c in range(q):
                v_signs.append(Int32(-1) if sg[c] < Float32(0.0) else Int32(1))
        _ = drank^
        _ = dvals^
        _ = dsg^
    trace.record_list_f32("nys.eigenvalues", values)
    trace.record_list_f32("nys.sqrt_eigenvalues", sqrt_s)
    trace.record_list_i32("nys.v_signs", v_signs)
    trace.record_list_f32("nys.eigenvectors", vecs_ord)

    # --- `U / sqrt(S) @ V`, on the device ---
    var dz = ctx.enqueue_create_buffer[DType.float32](q * q)
    var dnorm = ctx.enqueue_create_buffer[DType.float32](q * q)
    var gws = ctx.enqueue_create_buffer[DType.float32](
        identical_gemm_workspace_max_floats(q, q, q)
    )
    ctx.synchronize()
    ctx.enqueue_function[scale_columns_kernel](
        dz.unsafe_ptr(),
        dq0.unsafe_ptr(),
        dsq.unsafe_ptr(),
        Int32(q),
        grid_dim=((q * q + scale_tpb - 1) // scale_tpb, 1, 1),
        block_dim=(scale_tpb, 1, 1),
    )
    ctx.synchronize()
    trace.record_device(ctx, "nys.scaled", dz, q * q)
    # `Z . V`: cell `(i, j)` is `sum_k Z[i][k] V[k][j]`, and `V` is `Q^T`
    # with row `k` negated where `lambda_k < 0` (`_singular_value_f32`).
    # `Q` is stored with eigenvector `k` in COLUMN `k`, so this is `OP_NT`
    # with the column-signed `Q` as the right operand. When no eigenvalue is
    # negative that operand IS `Q`, so the same uploaded Q read by the
    # scaling kernel is reused and its output Z is a separate buffer
    # (DEVIATION 2487); only a basis kernel with a negative eigenvalue
    # stages the signed copy. Negation is exact, so each product term is
    # the unsigned term with its sign flipped and the fold order is the
    # same.
    if dev_order:
        # the column-signed Q written on the device; where no eigenvalue is
        # negative it is Q's own words, so the product is the one below
        identical_gemm_into(ctx, dnorm, dz, dvt_dev, gws, q, q, q, OP_NT)
        ctx.synchronize()
    elif any_negative:
        var dvt = _upload(ctx, vt_ord)
        identical_gemm_into(ctx, dnorm, dz, dvt, gws, q, q, q, OP_NT)
        ctx.synchronize()
        _ = dvt^
    else:
        identical_gemm_into(ctx, dnorm, dz, dq0, gws, q, q, q, OP_NT)
        ctx.synchronize()
    trace.record_device(ctx, "nys.normalization", dnorm, q * q)

    var norm = _download(ctx, dnorm, q * q)
    _ = dk^
    _ = na^
    _ = nb^
    _ = kws^
    _ = dvec^
    _ = dinfo^
    _ = dq0^
    _ = dsq^
    _ = dvt_dev^
    _ = dz^
    _ = dnorm^
    _ = gws^

    return NystroemModel(
        comp^, basis^, norm^, values^, vecs_ord^,
        q, n_features, kp.kernel, kp.degree, kp.gamma, kp.coef0,
        seed, sweeps,
    )


def _eigen_clip_f32() -> Float32:
    """sklearn's `S = xp.clip(S, 1e-12, None)`
    (`kernel_approximation.py:1069`), narrowed to float32.

    DEVIATION 1670. Copied BY VALUE from their line rather than chosen: a
    clip is a numerical policy, and a policy nobody wrote down is the thing
    `cholesky/`'s DEVIATION 1637 exists to forbid. The clip reads the
    SINGULAR VALUE `|lambda|` (`_singular_value_f32`), never the signed
    eigenvalue: their `S` comes out of an SVD and is never negative.
    """
    return Float32(1e-12)


def _singular_value_f32(lam: Float32) -> Float32:
    """`|lambda|` by bits: the singular value sklearn's `svd` returns for the
    eigenvalue `lambda` of a SYMMETRIC basis kernel.

    CORRECTED 2026-09-14, found by the first MI300X run of
    `test_kernel_methods_surface` ("NYS with every row a component, phi
    phi^T reproduces the linear kernel at 1e-2 -- 2.851021765361576"). The
    SVD of a symmetric `A = Q diag(lambda) Q^T` is `U = Q`,
    `S = |lambda|`, `V = diag(sign(lambda)) Q^T`. DEVIATION 1667's
    substitution was written for a positive semi-definite kernel, where the
    two are the same matrices, but a float32 Jacobi on a RANK DEFICIENT
    basis kernel (the surface test's linear kernel of 32 rows in 3
    features has 29 zero eigenvalues) returns numerically negative
    eigenvalues of order `eps32 * ||K||`. The earlier spelling clipped each
    of those to `1e-12` and so divided by `sqrt(1e-12)`: the embedding's
    Gram gains `lambda^2 / 1e-12` per such component, order one for a
    `lambda` near `-1e-6`, which is the 2.85 the run read. sklearn instead
    divides by `sqrt(|lambda|)` and carries the sign in `V`, so the same
    component adds `|lambda|` to the Gram. Ours now does what theirs does:
    the order, the clip and the square root read this magnitude, and the
    right operand of the normalization product carries the sign.

    ON EVERY KERNEL WITH NO NEGATIVE EIGENVALUE NO BIT MOVES: `|lambda|` is
    `lambda` (a `-0.0` compares equal to `+0.0` in the order and is clipped
    either way), the order is the same permutation and the right operand is
    `Q` itself. The sign test is `lambda < 0`, so `-0.0` keeps `V = Q^T`.
    """
    from std.memory import bitcast

    return bitcast[DType.float32](
        bitcast[DType.uint32](lam) & UInt32(0x7FFFFFFF)
    )


def _eigen_order_f32(
    values: List[Float32], q: Int, sabotage: Int
) -> List[Int]:
    """The PINNED order: singular value `|lambda|` DESCENDING, index
    ASCENDING on a tie (the caller passes `_singular_value_f32` of each
    eigenvalue; on a kernel with no negative eigenvalue that is the
    eigenvalue itself).

    DEVIATION 1669, and `km_oracle.mojo::km_eigen_order` is the float64
    mirror of exactly this loop. It is a SUMMATION ORDER, not a presentation
    choice: `scale_columns_kernel` and the `OP_NT` product below it sum over
    `k` in this order.

    A SELECTION SORT, not a comparison sort with an unspecified tie policy,
    so the result is a function of the values and the indices and of nothing
    else. `values[c] == values[best]` KEEPS `best`, which is the lower index
    because `c` walks ascending -- matching `sign_flip_kernel`'s tie rule,
    `cub::ArgMax`'s, `np.argmax`'s and cuML's thrust loop's, all four of
    which take the first occurrence.

    THE COMPARISON IS `>` ON FLOATS AND A NaN EIGENVALUE WOULD MAKE IT
    NON-TOTAL. It cannot arrive: `km_validate_matrix` refuses a non-finite
    input on the host, the kernel matrix of a finite input is finite for
    every kernel here, and the Jacobi's rotations are finite arithmetic on a
    finite matrix. Named rather than guarded, because a guard here would be a
    branch no fixture can reach and rule 8 says an unreachable branch is an
    unchecked one.
    """
    var used = List[Bool]()
    for _ in range(q):
        used.append(False)
    var order = List[Int]()
    for _ in range(q):
        var best = -1
        for c in range(q):
            if used[c]:
                continue
            if best < 0:
                best = c
                continue
            if sabotage == KMSAB_EIGEN_ORDER_ASCENDING:
                # ARM: same multiset, reversed order, so the `k` axis of the
                # normalization product is walked the other way.
                if values[c] < values[best]:
                    best = c
                continue
            if values[c] > values[best]:
                best = c
            elif sabotage == KMSAB_EIGEN_TIE_UNSTABLE and values[c] == values[
                best
            ]:
                # ARM: the tie break keeps the HIGHER index, so the order
                # stops being the total order the convention names. INERT on
                # any fixture without a repeated eigenvalue, which is why the
                # sweep is required.
                best = c
        used[best] = True
        order.append(best)
    return order^


def nystroem_transform_host(
    model: NystroemModel,
    x: List[Float32],
    n_rows: Int,
    mut trace: IdentityTrace,
    elem_tpb: Int = KM_EPILOGUE_TPB,
    sabotage: Int = KMSAB_NONE,
) raises -> List[Float32]:
    """`Nystroem.transform(X)` (`kernel_approximation.py:1094-1110`):

        embedded = pairwise_kernels(X, self.components_, ...)
        return embedded @ self.normalization_.T

    `n_rows x n_components` row-major.

    **DEVIATION 1674: THE TRANSPOSE IS NOT A FREE CHOICE.** `normalization_`
    is mathematically symmetric and is NOT bitwise symmetric -- cell `(i, j)`
    is `sum_k (Q[i][k] / sqrt(s_k)) Q[j][k]` and cell `(j, i)` is
    `sum_k (Q[j][k] / sqrt(s_k)) Q[i][k]`, and `fl(fl(a / w) b)` is not
    `fl(fl(b / w) a)`. So `@ normalization.T` and `@ normalization` are two
    different float32 answers, theirs is the transposed one, and the
    `KMSAB_EMBED_OP_NN` arm exists so a reader who "simplifies" it away is
    caught by a gate rather than by a downstream user.
    """
    km_validate_matrix(x, n_rows, model.n_features, "nystroem transform X")
    var kp = nystroem_params(model)
    var q = model.n_components
    var d = model.n_features

    var ctx = _family_ctx()
    var dx = _upload(ctx, x)
    var dc = _upload(ctx, model.components)
    var dnorm = _upload(ctx, model.normalization)
    var dk = ctx.enqueue_create_buffer[DType.float32](n_rows * q)
    var na = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var nb = ctx.enqueue_create_buffer[DType.float32](q)
    var kws = ctx.enqueue_create_buffer[DType.float32](
        km_kernel_workspace_floats(n_rows, q, d)
    )
    var demb = ctx.enqueue_create_buffer[DType.float32](n_rows * q)
    var gws = ctx.enqueue_create_buffer[DType.float32](
        identical_gemm_workspace_max_floats(n_rows, q, q)
    )
    ctx.synchronize()

    km_kernel_matrix(
        ctx, kp, dk, dx, dc, n_rows, q, d, na, nb, kws, elem_tpb, sabotage
    )
    ctx.synchronize()
    trace.record_device(ctx, "nys.cross_kernel", dk, n_rows * q)

    var op = OP_NT
    if sabotage == KMSAB_EMBED_OP_NN:
        op = OP_NN
    identical_gemm_into(ctx, demb, dk, dnorm, gws, n_rows, q, q, op)
    ctx.synchronize()
    trace.record_device(ctx, "nys.embedding", demb, n_rows * q)

    var out = _download(ctx, demb, n_rows * q)
    _ = dx^
    _ = dc^
    _ = dnorm^
    _ = dk^
    _ = na^
    _ = nb^
    _ = kws^
    _ = demb^
    _ = gws^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return out^


def nystroem_transform_host_into[out_origin: MutOrigin, //](
    model: NystroemModel,
    x: List[Float32],
    n_rows: Int,
    output: MutPointer[Float32, out_origin],
    mut trace: IdentityTrace,
    elem_tpb: Int = KM_EPILOGUE_TPB,
    sabotage: Int = KMSAB_NONE,
) raises:
    """The public transform written directly to caller-owned host storage."""
    km_validate_matrix(x, n_rows, model.n_features, "nystroem transform X")
    var t0 = Int(perf_counter_ns())
    var ctx = _family_ctx()
    var dx = _upload(ctx, x)
    _nystroem_transform_dev(ctx, dx, model, n_rows, output, trace, elem_tpb, sabotage, t0)
    _ = dx^
    _ = ctx^


def nystroem_transform_ptr_into[out_origin: MutOrigin, //](
    model: NystroemModel,
    xaddr: Int,
    n_rows: Int,
    output: MutPointer[Float32, out_origin],
    mut trace: IdentityTrace,
) raises:
    """KM_FAST_PTR_IN: `nystroem_transform_host_into` with X read from the
    caller's address on the device (`_upload_checked`)."""
    var t0 = Int(perf_counter_ns())
    var ctx = _family_ctx()
    var dx = _upload_checked(ctx, xaddr, n_rows, model.n_features, "nystroem transform X")
    _nystroem_transform_dev(ctx, dx, model, n_rows, output, trace, KM_EPILOGUE_TPB, KMSAB_NONE, t0)
    _ = dx^
    _ = ctx^


def _nystroem_transform_dev[out_origin: MutOrigin, //](
    ctx: DeviceContext,
    mut dx: DeviceBuffer[DType.float32],
    model: NystroemModel,
    n_rows: Int,
    output: MutPointer[Float32, out_origin],
    mut trace: IdentityTrace,
    elem_tpb: Int,
    sabotage: Int,
    t0: Int,
) raises:
    """The transform from X on the device (`dx`, n_rows x n_features)."""
    var kp = nystroem_params(model)
    var q = model.n_components
    var d = model.n_features
    # MOJOLEARN_STAGE_TIMES=1: wall per phase (each phase already drains).
    var st_on = getenv("MOJOLEARN_STAGE_TIMES") == "1"
    var dc = _upload(ctx, model.components)
    var dnorm = _upload(ctx, model.normalization)
    var dk = ctx.enqueue_create_buffer[DType.float32](n_rows * q)
    var na = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var nb = ctx.enqueue_create_buffer[DType.float32](q)
    var kws = ctx.enqueue_create_buffer[DType.float32](
        km_kernel_workspace_floats(n_rows, q, d)
    )
    var demb = ctx.enqueue_create_buffer[DType.float32](n_rows * q)
    var gws = ctx.enqueue_create_buffer[DType.float32](
        identical_gemm_workspace_max_floats(n_rows, q, q)
    )
    ctx.synchronize()
    var t1 = Int(perf_counter_ns())
    km_kernel_matrix(
        ctx, kp, dk, dx, dc, n_rows, q, d, na, nb, kws, elem_tpb, sabotage
    )
    ctx.synchronize()
    var t2 = Int(perf_counter_ns())
    trace.record_device(ctx, "nys.cross_kernel", dk, n_rows * q)
    var op = OP_NT
    if sabotage == KMSAB_EMBED_OP_NN:
        op = OP_NN
    identical_gemm_into(ctx, demb, dk, dnorm, gws, n_rows, q, q, op)
    ctx.synchronize()
    var t3 = Int(perf_counter_ns())
    trace.record_device(ctx, "nys.embedding", demb, n_rows * q)
    _download_into(ctx, demb, output, n_rows * q)
    if st_on:
        print("NYS_TRANSFORM_STAGES rows=" + String(n_rows) + " alloc_upload_ms=" + String((t1 - t0) // 1000000)
              + " kernel_ms=" + String((t2 - t1) // 1000000) + " gemm_ms=" + String((t3 - t2) // 1000000)
              + " copy_out_ms=" + String((Int(perf_counter_ns()) - t3) // 1000000))
    _ = dc^
    _ = dnorm^
    _ = dk^
    _ = na^
    _ = nb^
    _ = kws^
    _ = demb^
    _ = gws^


# ===========================================================================
# RBFSampler
# ===========================================================================


@fieldwise_init
struct RBFSamplerModel(Movable):
    """`sklearn.kernel_approximation.RBFSampler` after `fit`."""

    var random_weights: List[Float32]
    """`random_weights_`, `n_features x n_components` row-major."""

    var random_offset: List[Float32]
    """`random_offset_`, `n_components`."""

    var n_features: Int
    var n_components: Int
    var gamma: Float32
    var seed: UInt64

    var sigma: Float32
    """`sqrt(2 gamma)`, computed ONCE on the host. Carried rather than
    recomputed so `transform` and any reproduction of the fit use the same
    bits. DEVIATION 1678."""

    var scale: Float32
    """`sqrt(2 / n_components)`, same argument."""


def rbf_sampler_fit_host(
    n_features: Int,
    n_components: Int,
    gamma: Float32,
    seed: UInt64,
    mut trace: IdentityTrace,
    tpb: Int = KM_RF_TPB,
    sabotage: Int = KMSAB_NONE,
) raises -> RBFSamplerModel:
    """`RBFSampler.fit(X)` (`kernel_approximation.py:351-393`).

    **IT DOES NOT LOOK AT `X`, AND NEITHER DOES THEIRS.** Their `fit` reads
    `X.shape[1]` and, when `gamma == "scale"`, `X.var()`; the draws themselves
    depend on nothing but `n_features`, `n_components` and the random state.
    So this signature takes `n_features` rather than `X`, which makes the
    fact visible instead of implied.

    **`gamma="scale"` is resolved by the Python door, not here**: the EXACT
    variance of the float32 cells, the reciprocal rounded once
    (`python/mojolearn/_scale_gamma.py`), so no fold order
    sits in front of the draws and this entry still takes a gamma, not X.

    `n_components` is refused non-positive by name (DEVIATION 1686);
    scikit-learn's own constraint is `Interval(Integral, 1, None,
    closed="left")`.
    """
    if n_features <= 0:
        raise Error(
            "rbf_sampler_fit_host: n_features must be positive, got "
            + String(n_features)
        )
    if n_components <= 0:
        raise Error(
            "rbf_sampler_fit_host: n_components must be positive, got "
            + String(n_components)
            + ". scikit-learn's constraint is Interval(Integral, 1, None,"
            " closed='left'). DEVIATION 1686"
        )
    if gamma != gamma or not (gamma > Float32(0.0)):
        raise Error(
            "rbf_sampler_fit_host: gamma must be POSITIVE; got a value that"
            " is not greater than zero (spelled `not (gamma > 0)` so a NaN"
            " is refused by the same test). At gamma = 0 every weight is"
            " zero and the feature map is a constant; at gamma < 0 the"
            " square root in sqrt(2 gamma) is NaN. DEVIATION 1686"
        )

    var sigma = km_weight_sigma(gamma)
    var scale = km_feature_scale(n_components)

    var ctx = _family_ctx()
    var dw = ctx.enqueue_create_buffer[DType.float32](
        n_features * n_components
    )
    var db = ctx.enqueue_create_buffer[DType.float32](n_components)
    ctx.synchronize()
    km_random_weights(
        ctx, dw, seed, n_features, n_components, sigma, tpb, sabotage
    )
    km_random_offsets(ctx, db, seed, n_components, tpb)
    ctx.synchronize()
    trace.record_device(ctx, "rf.weights", dw, n_features * n_components)
    trace.record_device(ctx, "rf.offsets", db, n_components)
    trace.record_scalar_f32("rf.sigma", sigma)
    trace.record_scalar_f32("rf.scale", scale)

    var w = _download(ctx, dw, n_features * n_components)
    var b = _download(ctx, db, n_components)
    _ = dw^
    _ = db^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return RBFSamplerModel(
        w^, b^, n_features, n_components, gamma, seed, sigma, scale
    )


def rbf_sampler_transform_host(
    model: RBFSamplerModel,
    x: List[Float32],
    n_rows: Int,
    mut trace: IdentityTrace,
    tpb: Int = KM_RF_TPB,
    sabotage: Int = KMSAB_NONE,
) raises -> List[Float32]:
    """`RBFSampler.transform(X)` (`kernel_approximation.py:395-417`):

        projection = safe_sparse_dot(X, self.random_weights_)
        projection += self.random_offset_
        np.cos(projection, projection)
        projection *= (2.0 / self.n_components) ** 0.5

    `n_rows x n_components` row-major.

    The dot is `identical_gemm_into` at `OP_NN` (`X` is `n x d`,
    `random_weights_` is `d x D`), profile
    `mojolearn.identical.gemm.fp32.v1`; `linalg.matmul` is refused. The
    remaining three lines are `feature_map_epilogue_kernel`, in their order.

    **THIS IS THE ONE ARM OF THE LANE WHOSE ARITHMETIC INTENSITY MIGHT
    SURVIVE IDENTICAL.** See the README's WHAT THIS WILL COST, and note that
    the sentence there is a HYPOTHESIS: nothing in this lane has been timed.
    """
    km_validate_matrix(x, n_rows, model.n_features, "rbf_sampler transform X")
    var d = model.n_features
    var dd = model.n_components

    var ctx = _family_ctx()
    var dx = _upload(ctx, x)
    var dw = _upload(ctx, model.random_weights)
    var db = _upload(ctx, model.random_offset)
    var dp = ctx.enqueue_create_buffer[DType.float32](n_rows * dd)
    var gws = ctx.enqueue_create_buffer[DType.float32](
        identical_gemm_workspace_max_floats(n_rows, dd, d)
    )
    ctx.synchronize()

    # RBF_IDN_FUSED: the projection as the per-cell chain (and the whole
    # transform in that launch when no stage needs the projection alone)
    var chain = False
    var whole = False
    comptime if RBF_IDN_FUSED:
        chain = d <= RBF_FUSED_MAX_D and n_rows * dd > 0
        whole = chain and sabotage == KMSAB_NONE and not trace.enabled
        if chain:
            _rbf_idn_fused_launch(ctx, dp, dx, dw, db, n_rows, d, dd, model.scale, whole)
    if not chain:
        identical_gemm_into(ctx, dp, dx, dw, gws, n_rows, dd, d, OP_NN)
    ctx.synchronize()
    trace.record_device(ctx, "rf.projection", dp, n_rows * dd)

    if not whole:
        km_feature_map_epilogue(
            ctx, dp, db, n_rows, dd, model.scale, tpb, sabotage
        )
    ctx.synchronize()
    trace.record_device(ctx, "rf.feature_map", dp, n_rows * dd)

    var out = _download(ctx, dp, n_rows * dd)
    _ = dx^
    _ = dw^
    _ = db^
    _ = dp^
    _ = gws^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return out^


def rbf_sampler_transform_host_into[out_origin: MutOrigin, //](
    model: RBFSamplerModel,
    x: List[Float32],
    n_rows: Int,
    output: MutPointer[Float32, out_origin],
    mut trace: IdentityTrace,
    tpb: Int = KM_RF_TPB,
    sabotage: Int = KMSAB_NONE,
) raises:
    """The public transform written directly to caller-owned host storage."""
    km_validate_matrix(x, n_rows, model.n_features, "rbf_sampler transform X")
    var t0 = Int(perf_counter_ns())
    var ctx = _family_ctx()
    var dx = _upload(ctx, x)
    _rbf_transform_dev(ctx, dx, model, n_rows, output, trace, tpb, sabotage, t0)
    _ = dx^
    _ = ctx^


def rbf_sampler_transform_ptr_into[out_origin: MutOrigin, //](
    model: RBFSamplerModel,
    xaddr: Int,
    n_rows: Int,
    output: MutPointer[Float32, out_origin],
    mut trace: IdentityTrace,
) raises:
    """KM_FAST_PTR_IN: `rbf_sampler_transform_host_into` with X read from
    the caller's address on the device (`_upload_checked`)."""
    var t0 = Int(perf_counter_ns())
    var ctx = _family_ctx()
    var dx = _upload_checked(ctx, xaddr, n_rows, model.n_features, "rbf_sampler transform X")
    _rbf_transform_dev(ctx, dx, model, n_rows, output, trace, KM_RF_TPB, KMSAB_NONE, t0)
    _ = dx^
    _ = ctx^


def _rbf_transform_dev[out_origin: MutOrigin, //](
    ctx: DeviceContext,
    mut dx: DeviceBuffer[DType.float32],
    model: RBFSamplerModel,
    n_rows: Int,
    output: MutPointer[Float32, out_origin],
    mut trace: IdentityTrace,
    tpb: Int,
    sabotage: Int,
    t0: Int,
) raises:
    """The transform from X on the device (`dx`, n_rows x n_features)."""
    var d = model.n_features
    var dd = model.n_components
    var st_on = getenv("MOJOLEARN_STAGE_TIMES") == "1"
    var dw = _upload(ctx, model.random_weights)
    var db = _upload(ctx, model.random_offset)
    var dp = ctx.enqueue_create_buffer[DType.float32](n_rows * dd)
    var gws = ctx.enqueue_create_buffer[DType.float32](
        identical_gemm_workspace_max_floats(n_rows, dd, d)
    )
    ctx.synchronize()
    var t1 = Int(perf_counter_ns())
    var fused = False
    comptime if RBF_FUSED:
        fused = d <= RBF_FUSED_MAX_D and sabotage == KMSAB_NONE and n_rows * dd > 0
        if fused:
            ctx.enqueue_function[rbf_fused_transform_kernel](
                dp.unsafe_ptr(), dx.unsafe_ptr(), dw.unsafe_ptr(), db.unsafe_ptr(),
                Int32(n_rows), Int32(d), Int32(dd), model.scale,
                grid_dim=((n_rows * dd + RBF_FUSED_TPB - 1) // RBF_FUSED_TPB, 1, 1),
                block_dim=(RBF_FUSED_TPB, 1, 1),
            )
    # RBF_IDN_FUSED: the projection as the per-cell chain (and the whole
    # transform in that launch when no stage needs the projection alone)
    var chain = False
    comptime if RBF_IDN_FUSED:
        chain = d <= RBF_FUSED_MAX_D and n_rows * dd > 0
        fused = chain and sabotage == KMSAB_NONE and not trace.enabled
        if chain:
            _rbf_idn_fused_launch(ctx, dp, dx, dw, db, n_rows, d, dd, model.scale, fused)
    if not fused and not chain:
        identical_gemm_into(ctx, dp, dx, dw, gws, n_rows, dd, d, OP_NN)
    ctx.synchronize()
    var t2 = Int(perf_counter_ns())
    trace.record_device(ctx, "rf.projection", dp, n_rows * dd)
    if not fused:
        km_feature_map_epilogue(
            ctx, dp, db, n_rows, dd, model.scale, tpb, sabotage
        )
    ctx.synchronize()
    var t3 = Int(perf_counter_ns())
    trace.record_device(ctx, "rf.feature_map", dp, n_rows * dd)
    _download_into(ctx, dp, output, n_rows * dd)
    if st_on:
        print("RBF_TRANSFORM_STAGES rows=" + String(n_rows) + " alloc_upload_ms=" + String((t1 - t0) // 1000000)
              + " gemm_ms=" + String((t2 - t1) // 1000000) + " epilogue_ms=" + String((t3 - t2) // 1000000)
              + " copy_out_ms=" + String((Int(perf_counter_ns()) - t3) // 1000000))
    _ = dw^
    _ = db^
    _ = dp^
    _ = gws^
