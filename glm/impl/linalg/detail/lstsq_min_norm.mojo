# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Least squares when there are MORE COLUMNS THAN ROWS: the minimum-norm
solution, through the Gram of the ROWS and the same device Jacobi.

**NO UPSTREAM. This routine is ORIGINAL to mojolearn.** cuML does not have
it: `olsFit` sends `n_cols > n_rows` to `lstsqSvdJacobi`
(`cuml/cpp/src/glm/ols.cuh:112-113`), which is `cusolverDnGesvdj`, a
one-sided Jacobi SVD inside a closed vendor library. There is nothing to
transliterate, so `glm/DERIVATION_MAP.tsv` records this file with no
upstream file and this docstring carries the derivation instead.
DEVIATION 550.

WHY THE OLD REFUSAL WAS RIGHT ABOUT THE MATH AND WRONG ABOUT THE CONCLUSION
---------------------------------------------------------------------------
`glm/impl/glm/ols.mojo` used to raise here, and the reason it gave was
correct as far as it went:

    with `n_cols > n_rows` the Gram matrix is singular by construction

That is true of `A^T A`, which is `n_cols x n_cols` and can have rank at
most `n_rows`. `lstsq_eig` inverts exactly that matrix, so `lstsq_eig` is
genuinely the wrong program for this shape.

**But `A^T A` is not the only Gram matrix in the problem.** The
minimum-norm least-squares solution of an underdetermined system is

    w = A^T (A A^T)^+ b                                   [1]

and `A A^T` is `n_rows x n_rows`. For a design with more columns than rows
and full row rank -- the ordinary case, and the only case where the phrase
"more features than samples" describes real data -- `A A^T` is NONSINGULAR.
The singularity the refusal named is a property of the Gram of the COLUMNS.
It does not transfer to the Gram of the ROWS, and swapping which one is
formed removes it entirely.

Identity [1] is not restricted to full row rank; `A^+ = A^T (A A^T)^+` holds
for every `A`, and the `DivideByNonZero` guard that `lstsq_eig` already uses
supplies the `+` on `A A^T` exactly as it supplies it on `A^T A` there. So a
rank-deficient wide design lands on the same pseudo-inverse it would land on
by any other route, cut at the same threshold, with the same recorded
absolute-threshold deviation (`OLS_NONZERO_THRESH`, `glm/NOT_IMPLEMENTED.tsv`).

WHAT THIS COSTS IN DIGITS, STATED HONESTLY
-------------------------------------------
Forming `A A^T` squares the condition number of `A`, exactly as forming
`A^T A` does in `lstsq_eig`. This route is therefore NOT as accurate as a
one-sided Jacobi SVD of `A` would be, and on a badly conditioned wide design
it loses about twice the digits an SVD route would -- the same sentence
`lstsq_eig`'s docstring already carries about the tall case.

Two things follow and both are worth saying plainly:

1. **It is not WORSE than what this library already ships.** The
   overdetermined path a `mojolearn.LinearRegression` user takes every day
   is `lstsq_eig`, which pays exactly this cost on `cond(A)^2`. A user who
   accepts that solver for a 4,096 x 8 design is being handed the same
   arithmetic quality for an 8 x 4,096 one, not a downgrade.
2. **"Underdetermined" does not mean "ill-conditioned".** The two are
   routinely conflated and they are different properties. A wide random
   design has a perfectly well-conditioned `A A^T`; what makes it
   underdetermined is that the solution is not unique, and [1] answers that
   by picking the minimum-norm one, which is what an SVD pseudo-inverse
   picks too.

THE RIGHT ALGORITHM, IF THIS EVER NEEDS TO BE BETTER, is a rank-revealing
factorization of `A` that never forms a Gram: an LQ factorization (the
transpose of the QR route their `lstsqQR` takes, `algo = 2`), or a one-sided
Jacobi SVD of `A^T`. Either would halve the digits lost. Neither is here and
neither is refused as impossible; they are simply not written, and this file
is the cheap route that reuses machinery that already exists and is already
gated. That is an ENGINEERING position, not an attribution one.

THE SIX STEPS, AND EVERY ONE OF THEM ALREADY EXISTED
-----------------------------------------------------
    G   = A A^T          `core/gemm.mojo::gemm_nt_gram`   (n_rows^2 output)
    Q S Q^T = eig(G)     `decomposition/.../jacobi_eigh_kernel`
    S   <- diag(G)       `core/column_stats.mojo::diagonal_to_vector_kernel`
    QS  = Q invS         `divide_columns_by_nonzero_kernel`, the same guard
    inv = QS Q^T         `core/gemm.mojo::gemm_nt`
    z   = inv b          `core/gemm.mojo::gemv_n`
    w   = A^T z          `core/column_stats.mojo::xty_kernel`

Six of the seven are `lstsq_eig`'s own steps with `n_rows` where it writes
`n_cols`; the seventh, `w = A^T z`, is the one line that is new, and it is
the same kernel `lstsq_eig` uses for `A^T b`. No new arithmetic was written
for this file, which is why its identity story is inherited rather than
argued from scratch.

IDENTITY. Every step is a pinned one already: `gemm_nt_gram`'s IDENTICAL arm
is `pinned_gemm_nt_gram_kernel`, the Jacobi block width and fold are pinned
in `jacobi_eigh_device.mojo`, `xty_kernel` is row 29's pinned fold with
`identical_mul_add` per term, and the two elementwise kernels are one thread
per cell (SCHEDULING). There is no square root and no division on this path
outside `DivideByNonZero`, which is one IEEE divide. So this route is
bit-identical across Apple, NVIDIA and AMD under IDENTICAL for the same
reasons `lstsq_eig` is, and under FAST it moves for the same reasons.

BUFFERS. The caller hands the same scratch `lstsq_eig` takes. At this shape
`n_cols > n_rows`, so every `n_cols`-sized buffer is at least `n_rows` long
and every `n_cols x n_cols` buffer is at least `n_rows x n_rows`; nothing
needs to be reallocated and `ols_fit`'s signature does not change. The one
buffer used for a different thing is `ab`, which holds `z` (length `n_rows`)
rather than `A^T b` (length `n_cols`) -- named in the code where it happens.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.column_stats import (
    STATS_TPB,
    diagonal_to_vector_kernel,
    divide_columns_by_nonzero_kernel,
    xty_kernel,
)
from core.gemm import gemm_nt, gemm_nt_gram, gemv_n
from core.identity_trace import IdentityTrace
from decomposition.checks.jacobi_eigh_device import (
    JACOBI_SWEEPS,
    JACOBI_TOL,
    JACOBI_TPB,
    jacobi_eigh_kernel,
)
from glm.impl.linalg.detail.lstsq import OLS_ELEM_TPB, OLS_NONZERO_THRESH


def lstsq_min_norm(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut w: DeviceBuffer[DType.float32],
    mut gram: DeviceBuffer[DType.float32],
    mut q: DeviceBuffer[DType.float32],
    mut qs: DeviceBuffer[DType.float32],
    mut s_vec: DeviceBuffer[DType.float32],
    mut z: DeviceBuffer[DType.float32],
    mut inv: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    elem_tpb: Int = OLS_ELEM_TPB,
) raises:
    """`w = A^T (A A^T)^+ b`. The untraced entry.

    Constructs a DISABLED `IdentityTrace` and calls `lstsq_min_norm_traced`,
    so there is exactly one implementation of the seven steps and the traced
    runs certify the shipped path rather than a copy of it -- the shape
    `lstsq_eig` uses, for the reason recorded there (DEVIATION 527).
    """
    var off = IdentityTrace.disabled()
    lstsq_min_norm_traced(
        ctx, a, b, w, gram, q, qs, s_vec, z, inv, n_rows, n_cols, elem_tpb,
        off,
    )


def lstsq_min_norm_traced(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut w: DeviceBuffer[DType.float32],
    mut gram: DeviceBuffer[DType.float32],
    mut q: DeviceBuffer[DType.float32],
    mut qs: DeviceBuffer[DType.float32],
    mut s_vec: DeviceBuffer[DType.float32],
    mut z: DeviceBuffer[DType.float32],
    mut inv: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    elem_tpb: Int,
    mut trace: IdentityTrace,
) raises:
    """`w = A^T (A A^T)^+ b`, with the stage card.

    The card's tags are `ols.mn.*` and not `ols.step*`: a differ that saw
    `ols.step1.covA` from a wide fit would line it up against a tall fit's
    `A^T A`, which is a different matrix of a different size, and the two
    would "diverge at step 1" for no reason. A route that computes different
    objects names them differently (`core/identity_trace.mojo` rule 2: a tag
    names a POSITION IN THE ALGORITHM).
    """
    if n_rows <= 1:
        raise Error(
            "lstsq_min_norm: n_rows must be at least 2; A A^T at n_rows == 1"
            " is a 1 x 1 Gram and core/gemm.mojo::gemm_nt_gram refuses"
            " n == 1 by name."
        )
    if n_cols <= n_rows:
        raise Error(
            "lstsq_min_norm: this route is for n_cols > n_rows only (got"
            " n_rows " + String(n_rows) + ", n_cols " + String(n_cols)
            + "). At n_cols <= n_rows the Gram of the COLUMNS is the smaller"
            " and better-conditioned one and lstsq_eig is the route; taking"
            " this one there would form an n_rows x n_rows matrix that is"
            " singular by construction, i.e. the exact defect this file"
            " exists to avoid, mirrored."
        )

    # STEP 1. G <- A A^T, the Gram of the ROWS. `gemm_nt_gram` takes the one
    # operand once and builds both views from it (DEVIATION 1873), which is
    # what this shape needs: `gemm_nt`'s two `mut` parameters cannot be handed
    # the same buffer.
    gemm_nt_gram(ctx, gram, a, n_rows, n_rows, n_cols)
    ctx.synchronize()
    trace.record_device[DType.float32](
        ctx, "ols.mn.step1.AAt", gram, n_rows * n_rows
    )

    # STEP 2. Q S Q^T <- G. Jacobi CONSUMES `gram` in place and leaves the
    # eigenvalues on its diagonal, which is why step 1 is recorded above and
    # not after this launch.
    var info_buf = ctx.enqueue_create_buffer[DType.float32](3)
    ctx.synchronize()
    ctx.enqueue_function[jacobi_eigh_kernel](
        gram.unsafe_ptr(),
        q.unsafe_ptr(),
        info_buf.unsafe_ptr(),
        Int32(n_rows),
        Int32(JACOBI_SWEEPS),
        Float32(JACOBI_TOL),
        grid_dim=(1, 1, 1),
        block_dim=(JACOBI_TPB, 1, 1),
    )
    ctx.enqueue_function[diagonal_to_vector_kernel](
        s_vec.unsafe_ptr(),
        gram.unsafe_ptr(),
        Int32(n_rows),
        grid_dim=((n_rows + elem_tpb - 1) // elem_tpb, 1, 1),
        block_dim=(elem_tpb, 1, 1),
    )
    var h_info = ctx.enqueue_create_host_buffer[DType.float32](3)
    ctx.enqueue_copy(dst_ptr=h_info.unsafe_ptr(), src_buf=info_buf)
    ctx.synchronize()
    # `[[mojo-buffer-freed-at-last-use]]`: the reads below sit after the
    # synchronize, and this keeps the buffer alive across the copy even if
    # the branch is ever made conditional.
    _ = h_info
    if h_info.unsafe_ptr().unsafe_load(0) == Float32(0.0):
        raise Error(
            "lstsq_min_norm: the device Jacobi did not converge in "
            + String(JACOBI_SWEEPS)
            + " sweeps on the "
            + String(n_rows)
            + " x "
            + String(n_rows)
            + " row Gram A A^T; ||offdiag||_F / ||A A^T||_F is still "
            + String(h_info.unsafe_ptr().unsafe_load(1))
            + ". An unconverged eigendecomposition is a wrong answer with"
            " no error, which is why this is checked and not assumed."
        )
    trace.record_device[DType.float32](
        ctx, "ols.mn.step2.eigvals", s_vec, n_rows
    )
    trace.record_device[DType.float32](
        ctx, "ols.mn.step2.eigvecs", q, n_rows * n_rows
    )
    trace.record_device[DType.float32](ctx, "ols.mn.step2.info", info_buf, 3)
    _record_rank(ctx, trace, s_vec, n_rows)

    # STEP 3. QS <- Q invS, `DivideByNonZero` on the eigenvalues of A A^T.
    # This is where the pseudo-inverse in `w = A^T (A A^T)^+ b` comes from:
    # a direction the rows barely span is DROPPED rather than divided by.
    ctx.enqueue_function[divide_columns_by_nonzero_kernel](
        qs.unsafe_ptr(),
        q.unsafe_ptr(),
        s_vec.unsafe_ptr(),
        Int32(n_rows),
        OLS_NONZERO_THRESH,
        grid_dim=((n_rows * n_rows + elem_tpb - 1) // elem_tpb, 1, 1),
        block_dim=(elem_tpb, 1, 1),
    )
    trace.record_device[DType.float32](
        ctx, "ols.mn.step3.QS", qs, n_rows * n_rows
    )

    # STEP 4. inv <- QS Q^T == (A A^T)^+.
    gemm_nt(ctx, inv, qs, q, n_rows, n_rows, n_rows)
    ctx.synchronize()
    trace.record_device[DType.float32](
        ctx, "ols.mn.step4.inv", inv, n_rows * n_rows
    )

    # STEP 5. z <- inv b, an `n_rows` vector. `z` is the caller's `ab`
    # buffer, which on the tall route holds `A^T b` and is `n_cols` long;
    # here it holds a DIFFERENT quantity of a DIFFERENT length, and the
    # parameter is named `z` so a reader is not misled by the call site.
    gemv_n(ctx, z, inv, b, n_rows, n_rows)
    ctx.synchronize()
    trace.record_device[DType.float32](ctx, "ols.mn.step5.z", z, n_rows)

    # STEP 6. w <- A^T z. The same kernel `lstsq_eig` uses for `A^T b`, one
    # block per feature striding rows, `identical_mul_add` per term.
    ctx.enqueue_function[xty_kernel](
        w.unsafe_ptr(),
        a.unsafe_ptr(),
        z.unsafe_ptr(),
        Int32(n_rows),
        Int32(n_cols),
        grid_dim=(n_cols, 1, 1),
        block_dim=(STATS_TPB, 1, 1),
    )
    ctx.synchronize()
    trace.record_device[DType.float32](ctx, "ols.mn.step6.coef", w, n_cols)
    _ = info_buf


def _record_rank(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    mut s_vec: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    """The card's one INTEGER stage: how many directions survive step 3.

    `divide_columns_by_nonzero_kernel`'s predicate is
    `lam > thresh or lam < -thresh` and this recomputes exactly that on the
    host, character for character, so the record is the rank the kernel is
    about to use. A NaN eigenvalue fails both comparisons and is counted as
    DROPPED, which is what the kernel does too. Two vendors' cards that
    first differ HERE disagree about how many directions the row space has,
    which is a different model and not a different last bit.
    """
    if not trace.enabled:
        return
    var hs = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_ptr=hs.unsafe_ptr(), src_buf=s_vec)
    ctx.synchronize()
    var kept = 0
    for i in range(n):
        var lam = hs.unsafe_ptr().unsafe_load(i)
        if lam > OLS_NONZERO_THRESH or lam < -OLS_NONZERO_THRESH:
            kept += 1
    var one = List[Int32]()
    one.append(Int32(kept))
    trace.record_list_i32("ols.mn.step3.rank", one)
    # `[[mojo-buffer-freed-at-last-use]]`
    _ = hs^
