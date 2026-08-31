# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# This lane MIRRORS CatBoost (pinned commit 54a8143a). Per-file provenance is in gbdt/DERIVATION_MAP.tsv, in this file's own docstring, and in NOTICE; files with no CatBoost counterpart say so.
"""Solve a small symmetric positive-definite system, by Cholesky.

PORT OF `catboost/private/libs/lapack/linear_system.{h,cpp}` at CatBoost
`54a8143a` -- `SolveLinearSystemCholesky` (`:34-49`). Transliterated. Do not
improve.

Its ONE caller in this port's reach is the walker's blocked-Hessian arm,
`UpdateMoveDirectionBlockedHessian` (`descent_helpers.cpp:91-117`), which
solves `Hessian * direction = gradient` per LEAF for MultiClass. The matrix
is `(numClasses - 1) x (numClasses - 1)` -- six by six for a seven-class
problem -- so this is a small dense solve run once per leaf on the host, not
a device kernel.

`SolveLinearSystem` (`:12-32`), the PACKED-storage sibling that goes through
`dppsv_`, has no caller here: its users are the pairwise leaves calculation
(`algo_helpers/pairwise_leaves_calculation.cpp:47`) and a pairwise unit test,
and the pairwise oracle is not ported. It is in `gbdt/NOT_IMPLEMENTED.tsv`.

# =========================================================================
# DEVIATION 74: theirs is LAPACK's `dposv_` (`linear_system.cpp:46-47`),
# reached through the clapack vendored in `contrib/libs/clapack`.
#
# clapack is OPEN, so under PORTING_RULES 0b-i it is a port candidate rather
# than a call to make -- the "call the platform's equivalent" exception is
# for CLOSED libraries (cuBLAS, cuSOLVER) where there is nothing to read.
# And the shape rules it out anyway: this runs on the HOST, once per leaf,
# on a matrix small enough to sit in registers. A device linalg call would
# be a round trip per leaf to solve a 6x6.
#
# `dposv` is factor-then-solve: Cholesky (`dpotrf`) followed by two
# triangular solves (`dpotrs`). Both are transcribed below in their
# textbook form, which is what LAPACK's own reference implementation is.
#
# THEIR FAILURE BEHAVIOUR IS COPIED AND IT IS NOT WHAT IT LOOKS LIKE.
# `dposv` returns `info > 0` when the leading minor of order `info` is not
# positive definite; the factorization stops there and **the right-hand
# side is left UNMODIFIED**. Their check is
#
#     CB_ENSURE(info >= 0, "LAPACK dposv_ failed with status " << info);
#
# which PASSES for every positive `info`. So when the Hessian is not
# positive definite CatBoost does not raise and does not fall back -- it
# proceeds with `target` still holding the raw GRADIENT, and the walker
# steps along the gradient instead of the Newton direction for that leaf.
# This port does the same, and `solve_linear_system_cholesky` returns the
# `info` so a caller that wants to count it can.
#
# It is reachable in principle: the multinomial Hessian `diag(p) - p p^T`
# is positive SEMI-definite, and only the `+ lambda` on the diagonal
# (`pointwise_oracle.cpp:178`) makes it definite. At `l2_leaf_reg = 0` with
# a saturated probability it can go indefinite in float64.
# =========================================================================
"""

from std.math import sqrt


def solve_linear_system_cholesky(
    mut matrix: List[Float64], mut target: List[Float64]
) raises -> Int:
    """`SolveLinearSystemCholesky(&matrix, &target)` (`:34-49`).

    `matrix` is `n x n` row-major and SYMMETRIC; only one triangle is read.
    `target` is the right-hand side in, the solution out.

    Returns LAPACK's `info`: 0 on success, or `k > 0` when the leading
    minor of order `k` is not positive definite -- in which case `target`
    is UNTOUCHED, exactly as `dposv` leaves it. See DEVIATION 74 for why
    that is not an error here.

    Their `target->size() == 1` shortcut (`:35-38`) is kept: a one-by-one
    system is a division, and taking the general path for it would divide
    by `sqrt(a)` twice and land a different last bit.
    """
    var n = len(target)
    if n == 0:
        return 0
    if len(matrix) != n * n:
        raise Error(
            "solve_linear_system_cholesky: matrix is " + String(len(matrix))
            + " for a system of " + String(n)
        )

    # `if (target->size() == 1) { (*target)[0] /= (*matrix)[0]; return; }`
    if n == 1:
        target[0] = target[0] / matrix[0]
        return 0

    # ---- dpotrf: the Cholesky factorization, lower triangle -------------
    # `A = L L^T`, computed in place. Theirs asks for 'U' storage on a
    # COLUMN-major array, which for a symmetric matrix is the same
    # factorization as 'L' on a ROW-major one; the triangle that gets
    # written is the one that gets read back, and the solve below reads
    # what this writes.
    for j in range(n):
        var s = matrix[j * n + j]
        for k in range(j):
            var ljk = matrix[j * n + k]
            s -= ljk * ljk
        if s <= 0.0:
            # the leading minor of order j+1 is not positive definite:
            # `info = j + 1`, and `target` is left as it was
            return j + 1
        var ljj = sqrt(s)
        matrix[j * n + j] = ljj
        for i in range(j + 1, n):
            var t = matrix[i * n + j]
            for k in range(j):
                t -= matrix[i * n + k] * matrix[j * n + k]
            matrix[i * n + j] = t / ljj

    # ---- dpotrs: forward substitution, then back substitution -----------
    # solve `L y = b`
    for i in range(n):
        var t = target[i]
        for k in range(i):
            t -= matrix[i * n + k] * target[k]
        target[i] = t / matrix[i * n + i]
    # solve `L^T x = y`
    for ii in range(n):
        var i = n - 1 - ii
        var t = target[i]
        for k in range(i + 1, n):
            t -= matrix[k * n + i] * target[k]
        target[i] = t / matrix[i * n + i]

    return 0
