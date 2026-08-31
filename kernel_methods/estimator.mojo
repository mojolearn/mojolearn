# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The three host-visible surfaces: KernelRidge, Nystroem, RBFSampler.

**NOT YET WIRED** into `bindings/_mojolearn_estimators.mojo` or
`python/mojolearn/` -- those directories are not this lane's. The README's
WHAT THE ORCHESTRATOR MUST WIRE names the tasks; this file is the entry a
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

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from cholesky.checks.potrf import (
    CHOL_ELEM_TPB,
    CHOL_PANEL_TPB,
)
from cholesky.checks.trsm import CHOL_SOLVE_TPB
from core.identity_trace import IdentityTrace
from decomposition.checks.jacobi_eigh_device import (
    JACOBI_SWEEPS,
    JACOBI_TOL,
    JACOBI_TPB,
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
from gemm.checks.gemm_oracle import OP_NN, OP_NT
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
from svm.impl.svm.svm_parameter import KernelParams


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
    for i in range(n):
        host.unsafe_ptr().unsafe_store(i, values[i])
    ctx.enqueue_copy(dst_buf=buf, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return buf^


def _download(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int
) raises -> List[Float32]:
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
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
    km_validate_kernel_params(kp, "kernel_ridge")
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

    var ctx = DeviceContext()

    # DEVIATION 1684. The training Gram is `K(X, X)` and `km_kernel_matrix`
    # takes its two operands as two MUTABLE buffers, which Mojo refuses to
    # satisfy from one allocation (PORTING.md 24, and
    # `decomposition/impl/linalg/detail/pca.mojo` carries the same note for
    # `X^T X`). So `X` is uploaded TWICE. The alternative -- a special
    # diagonal path -- would be a SECOND kernel-matrix code path reached only
    # when the two operands are the same, which is exactly the non-default
    # path `PORTING_RULES` rule 8 is about. One path, one extra copy of `X`.
    var xa = _upload(ctx, x)
    var xb = _upload(ctx, x)
    var dy = _upload(ctx, y)
    trace.record_device(ctx, "krr.input", xa, n_samples * n_features)

    var dk = ctx.enqueue_create_buffer[DType.float32](n_samples * n_samples)
    var na = ctx.enqueue_create_buffer[DType.float32](n_samples)
    var nb = ctx.enqueue_create_buffer[DType.float32](n_samples)
    var kws = ctx.enqueue_create_buffer[DType.float32](
        km_kernel_workspace_floats(n_samples, n_samples, n_features)
    )
    ctx.synchronize()

    km_kernel_matrix(
        ctx, kp, dk, xa, xb, n_samples, n_samples, n_features,
        na, nb, kws, elem_tpb, sabotage,
    )
    ctx.synchronize()
    trace.record_device(ctx, "krr.kernel", dk, n_samples * n_samples)

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
            " properly, port an SVD-based least-squares arm -- there is one"
            " at solver/checks/lstsq.mojo -- and gate BOTH sides of the"
            " branch; kernel_methods/NOT_IMPLEMENTED.tsv carries the row"
        )

    var dual = _download(ctx, dy, n_samples * n_targets)
    _ = xa^
    _ = xb^
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

    var ctx = DeviceContext()
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
    `Q diag(s^{-1/2}) Q^T`. **NOT bitwise symmetric** -- see DEVIATION 1674
    and `nystroem_transform_host`."""

    var eigenvalues: List[Float32]
    """DESCENDING, clipped. Theirs discards these; see this file's header,
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
    argument: for a symmetric positive semi-definite basis kernel the two
    coincide with `U = Q` and `V = Q^T`, `decomposition/`'s Jacobi is the
    only symmetric eigensolver in this tree, and cuSOLVER's `gesvd` is closed.

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

    var ctx = DeviceContext()
    var ca = _upload(ctx, comp)
    var cb = _upload(ctx, comp)
    var dk = ctx.enqueue_create_buffer[DType.float32](q * q)
    var na = ctx.enqueue_create_buffer[DType.float32](q)
    var nb = ctx.enqueue_create_buffer[DType.float32](q)
    var kws = ctx.enqueue_create_buffer[DType.float32](
        km_kernel_workspace_floats(q, q, n_features)
    )
    ctx.synchronize()
    km_kernel_matrix(
        ctx, kp, dk, ca, cb, q, q, n_features, na, nb, kws, elem_tpb, sabotage
    )
    ctx.synchronize()
    trace.record_device(ctx, "nys.basis_kernel", dk, q * q)

    # --- the eigendecomposition, on the device, through decomposition/ ---
    var dvec = ctx.enqueue_create_buffer[DType.float32](q * q)
    var dinfo = ctx.enqueue_create_buffer[DType.float32](3)
    ctx.synchronize()
    # ONE BLOCK OF EXACTLY `JACOBI_TPB` THREADS. That file's header calls the
    # block dim a CONTRACT rather than a suggestion: `pinned_block_sum`
    # writes one threadgroup slot per thread into a `JACOBI_TPB`-wide slab,
    # so a wider block writes past it and a narrower one folds a slot nobody
    # wrote. All of its other call sites pass the same constant.
    ctx.enqueue_function[jacobi_eigh_kernel](
        dk.unsafe_ptr(),
        dvec.unsafe_ptr(),
        dinfo.unsafe_ptr(),
        Int32(q),
        Int32(JACOBI_SWEEPS),
        Float32(JACOBI_TOL),
        grid_dim=(1, 1, 1),
        block_dim=(JACOBI_TPB, 1, 1),
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
    var sweeps = Int(info_h[2])

    var raw = _download(ctx, dk, q * q)
    var vecs = _download(ctx, dvec, q * q)

    # --- the order and the clip, on the host (DEVIATIONS 1669, 1670, 1688) ---
    var values_raw = List[Float32]()
    for c in range(q):
        values_raw.append(raw[c * q + c])
    var order = _eigen_order_f32(values_raw, q, sabotage)

    var clip = _eigen_clip_f32()
    var values = List[Float32]()
    var sqrt_s = List[Float32]()
    var vecs_ord = List[Float32]()
    for _ in range(q * q):
        vecs_ord.append(Float32(0.0))
    for c in range(q):
        var src = order[c]
        var s = values_raw[src]
        if sabotage != KMSAB_NO_EIGEN_CLIP and s < clip:
            s = clip
        values.append(s)
        sqrt_s.append(ftz(identical_sqrt(s)))
        for f in range(q):
            vecs_ord[f * q + c] = vecs[f * q + src]
    trace.record_list_f32("nys.eigenvalues", values)
    trace.record_list_f32("nys.sqrt_eigenvalues", sqrt_s)
    trace.record_list_f32("nys.eigenvectors", vecs_ord)

    # --- `U / sqrt(S) @ V`, on the device ---
    var dq0 = _upload(ctx, vecs_ord)
    var dq1 = _upload(ctx, vecs_ord)
    var dsq = _upload(ctx, sqrt_s)
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
    # `Z . Q^T`: cell `(i, j)` is `sum_k Z[i][k] Q[j][k]`, and `Q` is stored
    # with eigenvector `k` in COLUMN `k`, so this is `OP_NT` with `Q` as the
    # right operand. `dq1` is a SECOND upload of the same values because Mojo
    # refuses one buffer as two mutable kernel arguments (DEVIATION 1684).
    identical_gemm_into(ctx, dnorm, dz, dq1, gws, q, q, q, OP_NT)
    ctx.synchronize()
    trace.record_device(ctx, "nys.normalization", dnorm, q * q)

    var norm = _download(ctx, dnorm, q * q)
    _ = ca^
    _ = cb^
    _ = dk^
    _ = na^
    _ = nb^
    _ = kws^
    _ = dvec^
    _ = dinfo^
    _ = dq0^
    _ = dq1^
    _ = dsq^
    _ = dz^
    _ = dnorm^
    _ = gws^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^

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
    `cholesky/`'s DEVIATION 1637 exists to forbid. A NEGATIVE eigenvalue --
    which a float32 Jacobi produces on a numerically singular Gram matrix --
    is CLIPPED rather than refused, exactly as theirs does, because the
    Nystroem embedding of a rank-deficient basis is a legitimate thing to
    ask for and clipping is how they answer it.
    """
    return Float32(1e-12)


def _eigen_order_f32(
    values: List[Float32], q: Int, sabotage: Int
) -> List[Int]:
    """The PINNED order: eigenvalue DESCENDING, index ASCENDING on a tie.

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

    var ctx = DeviceContext()
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

    **`gamma="scale"` IS NOT PORTED** and there is nothing to port it into:
    it is a host reduction over the training data (`X.var()`), it would make
    `fit` data-dependent, and it would put a variance -- a fold -- on the
    identity path in front of every draw. `NOT_IMPLEMENTED.tsv` carries the row; a
    caller computes its own gamma and passes it.

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

    var ctx = DeviceContext()
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

    var ctx = DeviceContext()
    var dx = _upload(ctx, x)
    var dw = _upload(ctx, model.random_weights)
    var db = _upload(ctx, model.random_offset)
    var dp = ctx.enqueue_create_buffer[DType.float32](n_rows * dd)
    var gws = ctx.enqueue_create_buffer[DType.float32](
        identical_gemm_workspace_max_floats(n_rows, dd, d)
    )
    ctx.synchronize()

    identical_gemm_into(ctx, dp, dx, dw, gws, n_rows, dd, d, OP_NN)
    ctx.synchronize()
    trace.record_device(ctx, "rf.projection", dp, n_rows * dd)

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
