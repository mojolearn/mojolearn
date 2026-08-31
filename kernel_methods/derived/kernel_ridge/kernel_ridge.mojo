# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`_solve_cholesky_kernel` and `_safe_solve`: cuML's kernel-ridge solver.

PORT OF `cuml/python/cuml/cuml/kernel_ridge/kernel_ridge.py` at cuML
`265b9da` (`upstream/cuml-v26.08.00`), lines 26-88 and 285-349.

**AN UPSTREAM EXISTS FOR KERNEL RIDGE AND IT IS PURE PYTHON.** cuML ships no
`cpp/src/kernel_ridge/` at this pin; the estimator is a `Base` subclass whose
`fit` calls `pairwise_kernels` and then a module-level `_solve_cholesky_kernel`
that reaches `cupyx.lapack.posv` -- cuSOLVER's `potrf` followed by `potrs`.
So the ALGORITHM is fully readable and is transliterated here line for line;
only the two library calls underneath it are closed, and
`VENDOR_LIBS.md`'s surviving exception plus `cholesky/`'s pinned
`potrf_lower` / `cho_solve` stand in for them. `PORTING_RULES` rule 2 is why
this file exists at all: their control plane is on the host, so ours is.

THEIR SEQUENCE, `kernel_ridge.py:48-72`, transcribed branch for branch:

    n_samples = K.shape[0]
    K = cp.asarray(K, dtype=np.float64)            # :53   DEVIATION 1661
    alpha = cp.atleast_1d(alpha)
    one_alpha = alpha.size == 1
    has_sw = sample_weight is not None
    if has_sw:                                     # :59   DEVIATION 1682
        sw = cp.sqrt(cp.atleast_1d(sample_weight))
        y = y * sw[:, cp.newaxis]
        K *= cp.outer(sw, sw)
    if one_alpha:                                  # :65   the arm ported
        K.flat[:: n_samples + 1] += alpha[0]       # :67   DEVIATION 1660
        dual_coef = _safe_solve(K, y)              # :69
        if has_sw:
            dual_coef *= sw[:, cp.newaxis]
        return dual_coef
    else:                                          # :74   DEVIATION 1681
        ... one penalty per target, solved one at a time ...

and `_safe_solve`, `:26-44`:

    try:
        seterr(linalg="raise")
        dual_coef = lapack.posv(K, y)
        if cp.all(cp.isnan(dual_coef)):            # their cusolver workaround
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        warnings.warn("Singular matrix ... Using least-squares instead.")
        dual_coef = linalg.lstsq(K, y, rcond=None)[0]

`predict`, `:346-349`:

    K = self._get_kernel(X, self.X_fit_)
    return cp.dot(K, self.dual_coef_)

# =========================================================================
# DEVIATION 1660 -- `alpha` IS THE RIDGE, AND THE CHOLESKY LANE'S JITTER IS
# `+0.0`. THE TWO ARE NEVER BOTH APPLIED. THIS IS THE LANE'S HEADLINE.
#
# `cholesky/estimator.mojo::cholesky_factor_host` takes `jitter` with NO
# DEFAULT, on purpose, and its docstring says why: "Every caller of this in
# the three downstream lanes needs a ridge and every one of them needs to
# have decided about it; a default would let the decision be made by not
# making it." This file is one of those three callers and this block is the
# decision.
#
# WHAT IS DECIDED. `kernel_ridge_solve` passes `Float32(0.0)` -- the
# `CHOL_JITTER_NONE` arm of DEVIATION 1637 -- and adds `alpha` itself. Three
# reasons, in order of how much they matter:
#
#  1. **`alpha` IS ALREADY THE RIDGE.** Their line is
#     `K.flat[::n_samples + 1] += alpha[0]`: an ABSOLUTE addition to the
#     diagonal, which is character for character what `add_jitter` does. A
#     second ridge on top of it would make the factored matrix
#     `K + (alpha + 2^-20) I` while every docstring, every oracle and every
#     user says `K + alpha I`. That is not conservative, it is wrong by a
#     number nobody wrote down.
#  2. **THE TWO POLICIES COINCIDE ON THESE SHAPES ANYWAY, AND THAT IS A
#     MEASURED FINDING OF THE CHOLESKY LANE, CITED RATHER THAN RE-DERIVED.**
#     `cholesky/README.md`, "Two corrections the gate forced": every
#     correlation-shaped kernel matrix has `k(x, x) = exp(0) = 1`, so the
#     absolute and relative jitter policies are THE SAME NUMBER on a unit
#     diagonal, and `CHOL_SAB_JITTER_RELATIVE` was reached and provably inert
#     on `FIX_RBF` for exactly that reason. RBF, laplacian and (at
#     `coef0 = 1`, `degree` any) the correlation-shaped polynomial kernels
#     all land on that diagonal. There is nothing for a second knob to
#     decide.
#  3. **ONE KNOB IS CHECKABLE AND TWO ARE NOT.** `check_kernel_ridge_vs_oracle`
#     compares against a float64 host solve of `K + alpha I`. With two ridges
#     the oracle would have to know the second one, which means the profile's
#     constant would have leaked into the reference, which is the shape of
#     mistake `gemm_oracle.mojo` warns about at length.
#
# WHAT THIS COSTS AND WHO PAYS IT. A caller who passes `alpha = 0` gets NO
# ridge at all and an RBF Gram matrix in float32 at any interesting size is
# numerically singular, so the factorization will report `info != 0` and
# DEVIATION 1662 will refuse. That refusal names `alpha` as the closure. It
# is the right failure: silently ridging a matrix a user asked not to ridge
# is how a reproducibility mode stops being one.
#
# `kernel_ridge_alpha_is_the_ridge()` below exists so this decision appears
# in a caller's source as a NAME rather than as a bare `Float32(0.0)` that
# reads like an oversight.
# =========================================================================

# =========================================================================
# DEVIATION 1661 -- THEIR SOLVE IS FLOAT64 AND OURS CANNOT BE.
#
# `kernel_ridge.py:53` is `K = cp.asarray(K, dtype=np.float64)`,
# UNCONDITIONALLY, even when `X` is float32 -- the kernel matrix is formed in
# the input dtype and then WIDENED before `posv`, and `fit` narrows the
# answer back with `.astype(X.dtype, copy=False)` at `:311`. That is a
# deliberate conditioning choice: an RBF Gram matrix is ill-conditioned by
# construction and the extra 29 bits of significand are what makes a small
# `alpha` usable.
#
# There is no float64 on the Apple GPU (`mojolearn-hardware-limits`), and
# `cholesky/` is float32 by its own DEVIATION 1635. So the factorization here
# runs in float32 and the ONLY conditioning available is `alpha`. Stated
# rather than smoothed over, because it means a float32 kernel ridge needs a
# LARGER `alpha` than cuML's to solve the same problem, and
# `check_kernel_ridge_vs_oracle` measures the gap against a float64 HOST
# solve of the same system rather than asserting it away.
# =========================================================================

# =========================================================================
# DEVIATION 1662 -- THE LEAST-SQUARES FALLBACK IS NOT PORTED; A FAILED
# FACTORIZATION RAISES BY NAME.
#
# THEIRS catches `LinAlgError` from `posv`, warns, and returns
# `linalg.lstsq(K, y, rcond=None)[0]` -- a completely different estimator,
# silently, behind a `warnings.warn` that a notebook swallows. It also
# catches the case where cuSOLVER returns all-NaN without raising, which they
# document as "a workaround for cusolver issue to be fixed in a future CUDA
# version".
#
# OURS refuses. `cholesky_solve_host` already refuses a failed factor by name
# for its own reason ("dividing by those returns infinities that look like
# numbers"), and this file refuses one step earlier so the message can name
# `alpha` and the kernel. The closure is stated in the error text: raise
# `alpha`, or port an SVD-based least-squares arm. `solver/original/
# lstsq.mojo` has an eigendecomposition-based least squares in this tree, so
# the port is possible and is NOT done here -- wiring another lane's solver
# into this one without a gate that drives BOTH sides of the branch would
# create exactly the unchecked non-default path `PORTING_RULES` rule 8
# describes. `kernel_methods/NOT_IMPLEMENTED.tsv` carries the row.
# =========================================================================
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from cholesky.original.potrf import (
    CHOL_ELEM_TPB,
    CHOL_NB_PINNED,
    CHOL_PANEL_TPB,
    add_jitter,
    chol_jitter_pinned,
    chol_workspace_floats,
    potrf_lower,
)
from cholesky.original.trsm import CHOL_SOLVE_TPB, cho_solve
from core.identity_trace import IdentityTrace
from kernel_methods.original.km_sabotage import (
    KMSAB_NONE,
    KMSAB_RIDGE_PLUS_JITTER,
    KMSAB_RIDGE_RELATIVE,
    sabotage_ridge_diag_kernel,
)
from original.numerics import ftz

#: SCHEDULING: one thread per diagonal entry.
comptime KRR_RIDGE_TPB = 256


def kernel_ridge_alpha_is_the_ridge() -> Float32:
    """The jitter `kernel_ridge_solve` hands `potrf_lower`: `+0.0`.

    A FUNCTION rather than a literal at the call site, for the reason
    `cholesky/estimator.mojo::cholesky_profile_jitter` gives about its own
    constant: a decision that appears in source as a bare `0.0` reads like an
    oversight, and the next person to see an ill-conditioned Gram matrix will
    "fix" it. DEVIATION 1660 is what this returns.
    """
    return Float32(0.0)


def add_ridge_diag_kernel(
    k_io: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    alpha: Float32,
):
    """`K.flat[:: n_samples + 1] += alpha[0]` (`kernel_ridge.py:67`).

    ONE THREAD PER DIAGONAL ENTRY, ABSOLUTE, NEVER RELATIVE. The spelling is
    `ftz(d + alpha)` -- the same three tokens as `cholesky/original/
    potrf.mojo::jitter_diag_kernel`, deliberately, so that a reader diffing
    the two sees one arithmetic and not two.

    DEVIATION 1685 is why this kernel exists at all instead of calling
    `add_jitter`: `add_jitter` validates its argument against the Cholesky
    profile's two accepted values (DEVIATION 1637) and refuses everything
    else BY NAME, and `alpha` is a caller's free parameter that is none of
    them. Reusing it would mean either relaxing another lane's pin or
    lying to it about what is being added.
    """
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    var d = ftz(k_io.unsafe_load(i * n + i))
    k_io.unsafe_store(i * n + i, ftz(d + alpha))


def add_ridge_diag(
    ctx: DeviceContext,
    mut k: DeviceBuffer[DType.float32],
    n: Int,
    alpha: Float32,
    tpb: Int = KRR_RIDGE_TPB,
    sabotage: Int = KMSAB_NONE,
) raises:
    """`K += alpha I` in place. ASYNCHRONOUS.

    At `KMSAB_NONE` this launches the production kernel and the sabotage file
    is not reached; `KMSAB_RIDGE_RELATIVE` launches the COPY in
    `kernel_methods/original/km_sabotage.mojo` instead, so no production
    kernel here carries a sabotage branch (DEVIATION 1687)."""
    if n <= 0:
        return
    if sabotage == KMSAB_RIDGE_RELATIVE:
        ctx.enqueue_function[sabotage_ridge_diag_kernel](
            k.unsafe_ptr(),
            Int32(n),
            alpha,
            Int32(sabotage),
            grid_dim=((n + tpb - 1) // tpb, 1, 1),
            block_dim=(tpb, 1, 1),
        )
        return
    ctx.enqueue_function[add_ridge_diag_kernel](
        k.unsafe_ptr(),
        Int32(n),
        alpha,
        grid_dim=((n + tpb - 1) // tpb, 1, 1),
        block_dim=(tpb, 1, 1),
    )


def kernel_ridge_solve(
    ctx: DeviceContext,
    mut k: DeviceBuffer[DType.float32],
    mut y: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    n: Int,
    n_targets: Int,
    alpha: Float32,
    mut trace: IdentityTrace,
    panel_tpb: Int = CHOL_PANEL_TPB,
    elem_tpb: Int = CHOL_ELEM_TPB,
    solve_tpb: Int = CHOL_SOLVE_TPB,
    ridge_tpb: Int = KRR_RIDGE_TPB,
    sabotage: Int = KMSAB_NONE,
) raises -> Int:
    """`_solve_cholesky_kernel(K, y, alpha, sample_weight=None)`, the
    `one_alpha` arm, on the device.

    On entry `k` holds the `n x n` kernel matrix and `y` holds the
    `n x n_targets` targets, both row-major. On exit `k` holds `L` and `y`
    holds the DUAL COEFFICIENTS in place -- their `dual_coef`, `n x n_targets`
    row-major -- exactly as `posv` overwrites both of its arguments.

    Returns LAPACK's `info`. **A NON-ZERO `info` MEANS `y` IS NOT A SOLUTION**
    and this function does not raise on it; the host entry
    (`kernel_methods/estimator.mojo::kernel_ridge_fit_host`) is what refuses
    by name, so a caller keeping buffers on the device across a
    hyperparameter sweep can look at `info` and move on. That split copies
    `cholesky/estimator.mojo`'s: `cho_solve` trusts its caller, the host entry
    does not.

    `ws` must hold at least `kernel_ridge_workspace_floats(n)` floats.

    MULTI-TARGET FALLS OUT FREE (DEVIATION 1681). `cho_solve` already takes
    `nrhs` and solves every right-hand side in its own thread, so
    `n_targets > 1` is not a second code path here and is not deferred. What
    IS deferred is cuML's `else` arm at `:74`, one `alpha` PER TARGET, which
    re-adds and re-subtracts the penalty around a fresh factorization per
    column; that is a different algorithm and `NOT_IMPLEMENTED.tsv` carries it.
    """
    if n <= 0 or n_targets <= 0:
        raise Error(
            "kernel_ridge_solve: n and n_targets must be positive, got n="
            + String(n)
            + " n_targets="
            + String(n_targets)
        )

    # `:67`. Their line, before the factorization and after nothing.
    add_ridge_diag(ctx, k, n, alpha, ridge_tpb, sabotage)
    if sabotage == KMSAB_RIDGE_PLUS_JITTER:
        # ARM: the Cholesky profile's `2^-20` ridge applied IN ADDITION to
        # `alpha`. DEVIATION 1660's forbidden state, and the one a careful
        # reader reaches for by accident, because `cholesky/estimator.mojo`
        # deliberately refuses to default its `jitter` argument and the
        # obvious way to satisfy it is to pass the profile's value.
        add_jitter(ctx, k, n, chol_jitter_pinned(), elem_tpb)
    trace.record_device(ctx, "krr.ridged", k, n * n)

    # `_safe_solve` -> `lapack.posv` -> cusolverDnpotrf. DEVIATION 1660 hands
    # this the `+0.0` jitter: the ridge is already in the matrix.
    var run = potrf_lower(
        ctx,
        k,
        ws,
        n,
        trace,
        CHOL_NB_PINNED,
        panel_tpb,
        elem_tpb,
    )
    if run.info != 0:
        # No solve. `y` is left holding the targets, untouched, so a caller
        # that ignores `info` gets its own input back rather than a plausible
        # wrong answer. DEVIATION 1662.
        return run.info

    # `lapack.posv`'s second half, cusolverDnpotrs. `cho_solve` records
    # `chol.solve.forward` and `chol.solve.back` itself.
    cho_solve(ctx, k, y, n, n_targets, trace, solve_tpb)
    trace.record_device(ctx, "krr.dual_coef", y, n * n_targets)
    return 0


def kernel_ridge_workspace_floats(n: Int) -> Int:
    """What `kernel_ridge_solve` needs in `ws`: the Cholesky's workspace at
    the pinned block size, and nothing else. Named here so a caller never
    reaches into `cholesky/original/` for `chol_workspace_floats` and never
    guesses `CHOL_NB_PINNED`."""
    var w = chol_workspace_floats(n, CHOL_NB_PINNED)
    if w < 1:
        return 1
    return w
