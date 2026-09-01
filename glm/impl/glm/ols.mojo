# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`olsFit`: the entry point, its guards, its SAMPLE WEIGHTS and its SOLVER
DISPATCH.

PORT OF `cuml/cpp/src/glm/ols.cuh::olsFit` at cuML `00094f7`. Partial.
Do not improve.

WHY THIS FILE EXISTS, AND IT IS NOT A WRAPPER
---------------------------------------------
`glm/impl/linalg/detail/lstsq.mojo` ports RAFT's `lstsqEig`, which is cuML's
`algo = 1`. It was reachable directly and nothing stood between a caller and
it. **That skipped their dispatch, and their dispatch contains a correctness
guard.** `ols.cuh:112-113`:

    int selectedAlgo = algo;
    if (n_cols > n_rows || n_cols == 1) selectedAlgo = 0;

so a 4,000 x 5,000 design, or a single column, must not reach `lstsqEig`.
Before that guard was ported a caller got a plausible-looking vector of
garbage from a singular inverse. **That is the same failure as the
eigensolver's old 32-feature cap: an ordinary input, silently wrong, no
error.**

WHAT CHANGED 2026-09-01, AND WHY THE TWO SHAPES NO LONGER REFUSE
-----------------------------------------------------------------
This file used to RAISE at both of those shapes, because `selectedAlgo = 0`
is `lstsqSvdJacobi` and `lstsqSvdJacobi` is `cusolverDnGesvdj`, a one-sided
Jacobi SVD inside a closed vendor library. **"It needs a closed library" is
not a reason to refuse in this repository** -- the tree hand-writes LU,
`potrf` and `trsm` for exactly this reason -- so the question is only
whether a portable route exists here. For both shapes it does, and neither
route is `lstsqSvdJacobi`:

    n_cols == 1      -> OLS_ALGO_EIG, the ported `lstsq_eig`. DEVIATION 551.
    n_cols > n_rows  -> OLS_ALGO_MIN_NORM_EIG, `lstsq_min_norm`, ORIGINAL
                        to this library. DEVIATION 550.

**`n_cols == 1` (DEVIATION 551).** Their switch here is an implementation
limit of THEIR eigensolver, and their own Python layer says so in as many
words: "Changing solver from 'eig' to 'svd' as eig solver does not support
training data with 1 column currently" (`linear_regression.pyx:390-394`).
Read that sentence carefully -- it is about what `eigDC` supports, not about
what the normal equations can express. And the numerical objection to the
normal equations does not apply at one column at all: `A^T A` is the 1 x 1
matrix `[sum a_i^2]`, whose condition number is 1, so "forming `A^T A`
squares the condition number" squares 1. The device Jacobi has no
one-column limit either: at `n = 1` the strict upper triangle is empty, the
off-diagonal norm is 0, the sweep loop exits at sweep 0 with `Q = [1]`, and
the answer is the exact scalar least squares `(A^T b) / (A^T A)`. So the
switch is theirs to need and ours not to. Recorded as a DEVIATION because
it is a place where this port deliberately does NOT follow their dispatch.

**`n_cols > n_rows` (DEVIATION 550).** The old refusal's reason -- "`A^T A`
is singular by construction" -- is true and is about the Gram of the
COLUMNS. It does not transfer to the Gram of the ROWS. `A A^T` is
`n_rows x n_rows`, it is nonsingular whenever the design has full row rank,
and `w = A^T (A A^T)^+ b` is the minimum-norm least-squares solution, which
is the same vector `lstsqSvdJacobi`'s SVD pseudo-inverse returns. That is
`glm/impl/linalg/detail/lstsq_min_norm.mojo`, seven steps of which six are
`lstsq_eig`'s own with `n_rows` where it writes `n_cols`. Its docstring
carries the honest accuracy statement: it squares the condition number
exactly as `lstsq_eig` does on the tall side, so it is not as accurate as a
true SVD route would be, and the algorithm that would be better is an LQ
factorization or a one-sided Jacobi SVD of `A^T`, neither of which is
written here.

WHAT IS STILL REFUSED HERE, AND WHY EACH ONE IS HONEST
-------------------------------------------------------
    algo = 0 asked for EXPLICITLY   `lstsqSvdJacobi` is a one-sided Jacobi
                                    SVD and this library does not have one.
                                    Not reachable from any Python door; the
                                    two shapes their dispatch FORCES to it
                                    are served above.
    algo = 2 (lstsqQR)              not written.
    algo = 3 (lstsqSvdQR)           not written.
    fit_intercept / normalize       `preProcessData` / `postProcessData`
                                    are not called from here; the intercept
                                    is the Python layer's host centering
                                    (DEVIATION 517). See below.

WHICH ALGO IS "THEIR DEFAULT" DEPENDS ON WHICH DOOR YOU COME IN
----------------------------------------------------------------
Both of these are true:

    C++    `olsFit(..., int algo = 0, ...)`          ols.cuh:67   -> SVD Jacobi
    Python `LinearRegression(algorithm='eig')`  linear_regression.pyx:309 -> algo 1

`_get_algorithm_int` (`:336-343`) maps `'eig'` to 1, which is `lstsqEig`,
which is what this repository ported. **A user calling cuML from Python gets
the SOLVER we have.** A user calling their C++ directly gets the one we do
not.

**BUT THE PYTHON DOOR ALSO DEFAULTS `fit_intercept=True`**
(`linear_regression.pyx:309`), and that is not a flag, it is two steps
wrapped around the solver. cuML's default Python fit is

    preProcessData  (center X by its column means, center y by its mean;
                     `preprocess.cuh:94-99`, `:110-117`)
      -> the algo-1 guard and lstsqEig
      -> postProcessData  (intercept = mean(y) - mu_X . coef; then UNDO the
                     centering of X and y in place)

This port refuses `fit_intercept` and defaults it to False, so what it
mirrors is the `fit_intercept=False` BRANCH of the Python default path --
which on their side skips both wrappers and sets `*intercept = 0`
(`ols.cuh:156`). It is a real arm of theirs; it is not the arm a Python user
lands on without asking. `python/mojolearn/linear_model.py` does the
centering on the host instead and says so (DEVIATION 517).

SAMPLE WEIGHTS ARE A RESCALE, NOT A SOLVER (`ols.cuh:99-110`, `:129-141`)
--------------------------------------------------------------------------
Ported here 2026-09-01. Their block, in their order, and every step of it is
outside the solve:

    w <- sqrt(w)                     raft::linalg::sqrt          `:100`
    A <- rowscale(A, w)              matrixVectorBinaryMult      `:101-102`
                                       <false, FALSE>: the vector is indexed
                                       by ROW; see glm/impl/matrix/math.mojo
    b <- b * w                       map_k(a * b)                `:103-110`
    -- the dispatch and the solve --
    A <- rowdiv_skipzero(A, w)       matrixVectorBinaryDivSkipZero `:130-131`
    b <- b / w                       map_k(a / b)                `:132-139`
    w <- pow(w, 2)                   powerScalar                 `:140`

The last three exist because `olsFit` mutates the caller's buffers and
documents that it does ("this vector is modified during the computation",
`ols.cuh:41-42`). Ours mutates the caller's `DeviceBuffer`s too, so they are
ported rather than dropped; they are not observable through
`glm/estimator.mojo`, which copies.

WHY A RESCALE IS THE WHOLE STORY. Minimising `sum_i w_i (b_i - a_i . x)^2`
is minimising `sum_i (sqrt(w_i) b_i - sqrt(w_i) a_i . x)^2`, so a weighted
least squares IS an unweighted least squares on rescaled rows. No solver
learns about weights, which is why this closes at the door and not in
`lstsq_eig`.

WEIGHTS AND `fit_intercept` TOGETHER ARE STILL REFUSED. Theirs computes
WEIGHTED column means in `preProcessData` (`raft::stats::weightedMean`,
`preprocess.cuh:95-97`, `:110-112`), which is a different centering from the
unweighted one, and `preProcessData` is not called from here at all. A
caller who asks for both gets the `fit_intercept` refusal, which is the
existing one.

IDENTITY. The six weight kernels are one thread per cell with no fold, so
their block width is SCHEDULING; the one sqrt is `identical_sqrt` and the
one `pow` is `identical_pow` (`glm/impl/matrix/math.mojo`). Nothing on this
path is a reduction, so the weighted fit is bit-identical across vendors
under IDENTICAL for exactly the reasons the unweighted one is.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from glm.impl.linalg.detail.lstsq import (
    OLS_ELEM_TPB,
    lstsq_eig_traced,
)
from glm.impl.linalg.detail.lstsq_min_norm import lstsq_min_norm_traced
from glm.impl.matrix.math import (
    MATRIX_ELEM_TPB,
    power_scalar_kernel,
    row_vector_binary_div_skip_zero_kernel,
    row_vector_binary_mult_kernel,
    sqrt_elementwise_kernel,
    vector_binary_div_kernel,
    vector_binary_mult_kernel,
)


# `ols.cuh:116-126`, the switch. Their ids, so an error message names the same solver
# theirs would.
comptime OLS_ALGO_SVD_JACOBI = 0
comptime OLS_ALGO_EIG = 1
comptime OLS_ALGO_QR = 2
comptime OLS_ALGO_SVD_QR = 3

#: OURS, NOT THEIRS. `lstsq_min_norm`, the minimum-norm route for
#: `n_cols > n_rows` (DEVIATION 550). It is numbered 100 and not 4 so that
#: nobody reads it as a cuML `algo` id that upstream might one day take:
#: their enumeration is 0..3 and this is not in it.
comptime OLS_ALGO_MIN_NORM_EIG = 100


def ols_fit(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut w: DeviceBuffer[DType.float32],
    mut cov_a: DeviceBuffer[DType.float32],
    mut q: DeviceBuffer[DType.float32],
    mut qs: DeviceBuffer[DType.float32],
    mut s_vec: DeviceBuffer[DType.float32],
    mut ab: DeviceBuffer[DType.float32],
    mut inv: DeviceBuffer[DType.float32],
    mut a_alias: DeviceBuffer[DType.float32],
    mut a_alias2: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    algo: Int = OLS_ALGO_EIG,
    fit_intercept: Bool = False,
    normalize: Bool = False,
    elem_tpb: Int = OLS_ELEM_TPB,
) raises:
    """`olsFit`, their guards and their dispatch, in their order, unweighted.

    `algo` defaults to `OLS_ALGO_EIG` and NOT to their `0`, because 0 selects
    `lstsqSvdJacobi`, a one-sided Jacobi SVD this library does not have.
    Defaulting to an arm that always raises would make the entry point
    useless; defaulting to the arm cuML's PYTHON layer defaults to is the
    closest honest choice. That is a DEVIATION and it is recorded in
    `glm/NOT_IMPLEMENTED.tsv`.

    The untraced entry. `ols_fit_traced` below is the same dispatch carrying
    a stage card; this one constructs a DISABLED trace so there is one
    implementation of the guards and not two (DEVIATION 527).
    """
    var off = IdentityTrace.disabled()
    ols_fit_traced(
        ctx, a, b, w, cov_a, q, qs, s_vec, ab, inv, a_alias, a_alias2,
        n_rows, n_cols, off, algo, fit_intercept, normalize, elem_tpb,
    )


def ols_fit_traced(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut w: DeviceBuffer[DType.float32],
    mut cov_a: DeviceBuffer[DType.float32],
    mut q: DeviceBuffer[DType.float32],
    mut qs: DeviceBuffer[DType.float32],
    mut s_vec: DeviceBuffer[DType.float32],
    mut ab: DeviceBuffer[DType.float32],
    mut inv: DeviceBuffer[DType.float32],
    mut a_alias: DeviceBuffer[DType.float32],
    mut a_alias2: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    mut trace: IdentityTrace,
    algo: Int = OLS_ALGO_EIG,
    fit_intercept: Bool = False,
    normalize: Bool = False,
    elem_tpb: Int = OLS_ELEM_TPB,
) raises:
    """`olsFit` with the DEVIATION 527 stage card, unweighted.

    A one-float dummy weight buffer and `has_sample_weight = False` into
    `ols_fit_weighted_traced`, so there is ONE implementation of the guards,
    the weight block and the dispatch rather than two that can drift. The
    dummy costs one allocation and no launch: every weight kernel is behind
    `if has_sample_weight`.
    """
    var no_weights = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.synchronize()
    ols_fit_weighted_traced(
        ctx, a, b, w, cov_a, q, qs, s_vec, ab, inv, a_alias, a_alias2,
        no_weights, False, n_rows, n_cols, trace, algo, fit_intercept,
        normalize, elem_tpb,
    )
    _ = no_weights^


def ols_fit_weighted(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut w: DeviceBuffer[DType.float32],
    mut cov_a: DeviceBuffer[DType.float32],
    mut q: DeviceBuffer[DType.float32],
    mut qs: DeviceBuffer[DType.float32],
    mut s_vec: DeviceBuffer[DType.float32],
    mut ab: DeviceBuffer[DType.float32],
    mut inv: DeviceBuffer[DType.float32],
    mut a_alias: DeviceBuffer[DType.float32],
    mut a_alias2: DeviceBuffer[DType.float32],
    mut sample_weight: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    algo: Int = OLS_ALGO_EIG,
    fit_intercept: Bool = False,
    normalize: Bool = False,
    elem_tpb: Int = OLS_ELEM_TPB,
) raises:
    """`olsFit` with `sample_weight != nullptr`. The untraced entry.

    `sample_weight` is `n_rows` long and IS MODIFIED, exactly as theirs is
    (`ols.cuh:41-42`): it comes back holding `pow(sqrt(w), 2)`, their own
    restore, which is `w` to within two roundings and not bit for bit.
    """
    var off = IdentityTrace.disabled()
    ols_fit_weighted_traced(
        ctx, a, b, w, cov_a, q, qs, s_vec, ab, inv, a_alias, a_alias2,
        sample_weight, True, n_rows, n_cols, off, algo, fit_intercept,
        normalize, elem_tpb,
    )


def ols_fit_weighted_traced(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut w: DeviceBuffer[DType.float32],
    mut cov_a: DeviceBuffer[DType.float32],
    mut q: DeviceBuffer[DType.float32],
    mut qs: DeviceBuffer[DType.float32],
    mut s_vec: DeviceBuffer[DType.float32],
    mut ab: DeviceBuffer[DType.float32],
    mut inv: DeviceBuffer[DType.float32],
    mut a_alias: DeviceBuffer[DType.float32],
    mut a_alias2: DeviceBuffer[DType.float32],
    mut sample_weight: DeviceBuffer[DType.float32],
    has_sample_weight: Bool,
    n_rows: Int,
    n_cols: Int,
    mut trace: IdentityTrace,
    algo: Int = OLS_ALGO_EIG,
    fit_intercept: Bool = False,
    normalize: Bool = False,
    elem_tpb: Int = OLS_ELEM_TPB,
) raises:
    """**THE ONE IMPLEMENTATION** of `olsFit`'s guards, weight block and
    dispatch. Every other entry in this file constructs arguments and calls
    it.

    THE GUARDS AND THE DISPATCH ARE INSIDE THE TRACED ENTRY DELIBERATELY.
    A card produced by a driver that called `lstsq_eig` directly would
    certify a path no user takes and would miss the very thing
    `ols.cuh:112-113` exists for -- and `glm/estimator.mojo` shipped
    exactly that bypass until DEVIATION 527 found it.
    """
    # `ols.cuh:74-75`, copied including the bounds.
    if n_cols <= 0:
        raise Error("olsFit: number of columns cannot be less than one")
    if n_rows <= 1:
        raise Error("olsFit: number of rows cannot be less than two")

    if fit_intercept:
        raise Error(
            "olsFit: fit_intercept is not ported. It needs preProcessData and"
            " postProcessData from cuml glm/preprocess.cuh; see"
            " glm/NOT_IMPLEMENTED.tsv. python/mojolearn/linear_model.py"
            " centers X and y on the HOST instead (DEVIATION 517)"
        )
    if normalize:
        raise Error(
            "olsFit: normalize is not ported, and theirs is only reachable"
            " with fit_intercept; see glm/NOT_IMPLEMENTED.tsv"
        )
    if has_sample_weight and len(sample_weight) < n_rows:
        raise Error(
            "olsFit: sample_weight must hold n_rows = " + String(n_rows)
            + " floats, got " + String(len(sample_weight))
        )

    # `ols.cuh:99-110`. THE SCALING, before the dispatch, exactly there.
    var cells = n_rows * n_cols
    if has_sample_weight:
        ctx.enqueue_function[sqrt_elementwise_kernel](
            sample_weight.unsafe_ptr(), Int32(n_rows),
            grid_dim=((n_rows + MATRIX_ELEM_TPB - 1) // MATRIX_ELEM_TPB, 1, 1),
            block_dim=(MATRIX_ELEM_TPB, 1, 1),
        )
        ctx.enqueue_function[row_vector_binary_mult_kernel](
            a.unsafe_ptr(), sample_weight.unsafe_ptr(),
            Int32(n_rows), Int32(n_cols),
            grid_dim=((cells + MATRIX_ELEM_TPB - 1) // MATRIX_ELEM_TPB, 1, 1),
            block_dim=(MATRIX_ELEM_TPB, 1, 1),
        )
        ctx.enqueue_function[vector_binary_mult_kernel](
            b.unsafe_ptr(), sample_weight.unsafe_ptr(), Int32(n_rows),
            grid_dim=((n_rows + MATRIX_ELEM_TPB - 1) // MATRIX_ELEM_TPB, 1, 1),
            block_dim=(MATRIX_ELEM_TPB, 1, 1),
        )
        ctx.synchronize()
        trace.record_device[DType.float32](
            ctx, "ols.weights.sqrt", sample_weight, n_rows
        )
        trace.record_device[DType.float32](ctx, "ols.weights.A", a, cells)
        trace.record_device[DType.float32](ctx, "ols.weights.b", b, n_rows)

    # THE DISPATCH. `ols.cuh:112-113` is
    #
    #     if (n_cols > n_rows || n_cols == 1) selectedAlgo = 0;
    #
    # and this is where this port deliberately parts from it. Both shapes
    # still OVERRIDE whatever the caller asked for, exactly as theirs does;
    # what changes is which solver they are overridden TO, because algo 0 is
    # a one-sided Jacobi SVD we do not have and both shapes have a portable
    # route here that theirs does not need. See the module docstring,
    # DEVIATIONS 550 and 551.
    var selected_algo = algo
    if n_cols > n_rows:
        selected_algo = OLS_ALGO_MIN_NORM_EIG
    elif n_cols == 1:
        selected_algo = OLS_ALGO_EIG

    if selected_algo == OLS_ALGO_EIG:
        lstsq_eig_traced(
            ctx, a, b, w, cov_a, q, qs, s_vec, ab, inv, a_alias, a_alias2,
            n_rows, n_cols, elem_tpb, trace,
        )
    elif selected_algo == OLS_ALGO_MIN_NORM_EIG:
        # Reached ONLY through the guard above: no caller can ask for it,
        # because it is not one of their algo ids and the Python door does
        # not expose an algorithm at all. `ab` carries `z`, an `n_rows`
        # vector, not `A^T b`; `cov_a` carries `A A^T`, not `A^T A`. Both
        # buffers are long enough at this shape because n_cols > n_rows.
        lstsq_min_norm_traced(
            ctx, a, b, w, cov_a, q, qs, s_vec, ab, inv,
            n_rows, n_cols, elem_tpb, trace,
        )
    elif selected_algo == OLS_ALGO_SVD_JACOBI:
        # Only an EXPLICIT `algo = 0` reaches here now. The two shapes the
        # guard used to force to this arm are served above, so this refusal
        # is no longer on any user's path -- it is the honest answer to a
        # caller who asked for a solver by name.
        raise Error(
            "olsFit: algo 0 is lstsqSvdJacobi, a ONE-SIDED JACOBI SVD of the"
            " design matrix, and this library does not implement one. It is"
            " cuML's C++ default and the more numerically stable route,"
            " because it never forms a Gram matrix and so never squares the"
            " condition number. The two shapes cuML's own dispatch forces to"
            " it (n_cols > n_rows, n_cols == 1) do NOT refuse here: they take"
            " lstsq_min_norm and lstsq_eig (DEVIATIONS 550, 551). See"
            " glm/NOT_IMPLEMENTED.tsv"
        )
    elif selected_algo == OLS_ALGO_QR:
        raise Error(
            "olsFit: algo 2 is lstsqQR, a Householder QR of the design"
            " matrix, which is not written here. Nothing forces a caller to"
            " it; algo 1 (lstsq_eig) is the ported solver"
        )
    elif selected_algo == OLS_ALGO_SVD_QR:
        raise Error(
            "olsFit: algo 3 is lstsqSvdQR, a full SVD of the design matrix,"
            " which is not written here. Nothing forces a caller to it;"
            " algo 1 (lstsq_eig) is the ported solver"
        )
    else:
        # `ols.cuh:123-125`, their default arm.
        raise Error(
            "olsFit: no algorithm with this id has been implemented"
        )

    # `ols.cuh:129-141`. THE RESTORE, after the solve, exactly there.
    if has_sample_weight:
        ctx.enqueue_function[row_vector_binary_div_skip_zero_kernel](
            a.unsafe_ptr(), sample_weight.unsafe_ptr(),
            Int32(n_rows), Int32(n_cols), Int32(0),
            grid_dim=((cells + MATRIX_ELEM_TPB - 1) // MATRIX_ELEM_TPB, 1, 1),
            block_dim=(MATRIX_ELEM_TPB, 1, 1),
        )
        ctx.enqueue_function[vector_binary_div_kernel](
            b.unsafe_ptr(), sample_weight.unsafe_ptr(), Int32(n_rows),
            grid_dim=((n_rows + MATRIX_ELEM_TPB - 1) // MATRIX_ELEM_TPB, 1, 1),
            block_dim=(MATRIX_ELEM_TPB, 1, 1),
        )
        ctx.enqueue_function[power_scalar_kernel](
            sample_weight.unsafe_ptr(), Int32(n_rows), Float32(2.0),
            grid_dim=((n_rows + MATRIX_ELEM_TPB - 1) // MATRIX_ELEM_TPB, 1, 1),
            block_dim=(MATRIX_ELEM_TPB, 1, 1),
        )
        ctx.synchronize()
