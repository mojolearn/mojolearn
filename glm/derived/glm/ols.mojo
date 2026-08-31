# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's DERIVATION_MAP.tsv and in this file's own docstring. See NOTICE.
"""`olsFit`: the entry point, its guards, and its SOLVER DISPATCH.

PORT OF `cuml/cpp/src/glm/ols.cuh::olsFit` at cuML `00094f7`. Partial.
Do not improve.

WHY THIS FILE EXISTS, AND IT IS NOT A WRAPPER
---------------------------------------------
`glm/gbdt/linalg/detail/lstsq.mojo` ports RAFT's `lstsqEig`, which is cuML's
`algo = 1`. It was reachable directly and nothing stood between a caller and
it. **That skipped their dispatch, and their dispatch contains a correctness
guard.** `ols.cuh:112-113`:

    int selectedAlgo = algo;
    if (n_cols > n_rows || n_cols == 1) selectedAlgo = 0;

An eigendecomposition of `A^T A` cannot solve either of those cases. With
`n_cols > n_rows` the Gram matrix is singular by construction, and cuML's
Python layer says so out loud when it forces the same switch
(`linear_regression.pyx:390-394`: "Changing solver from 'eig' to 'svd' as eig
solver does not support training data with 1 column currently").

So a caller handing this repository a 4,000 x 5,000 design, or a single
column, previously got a plausible-looking vector of garbage from a singular
inverse. **That is the same failure as the eigensolver's old 32-feature cap:
an ordinary input, silently wrong, no error.** Their dispatch is the thing
that prevented it and it was not ported.

`lstsqSvdJacobi` is not ported (`glm/NOT_IMPLEMENTED.tsv`), so this RAISES where
theirs would switch. Refusing by name is this repository's existing pattern
for an unported option, and it is the only honest option: the alternative is
running the solver their own code says cannot handle the shape.

WHICH ALGO IS "THEIR DEFAULT" DEPENDS ON WHICH DOOR YOU COME IN
----------------------------------------------------------------
Both of these are true and `glm/NOT_IMPLEMENTED.tsv` used to state only the first:

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
                     `preprocess.cuh:98`, `:100`, `:115`, `:117`)
      -> the algo-1 guard and lstsqEig
      -> postProcessData  (intercept = mean(y) - mu_X . coef,
                     `preprocess.cuh:148-161`; then UNDO the centering of X
                     and y in place, `:163-176`)

This port refuses `fit_intercept` and defaults it to False, so what it
mirrors is the `fit_intercept=False` BRANCH of the Python default path --
which on their side skips both wrappers and sets `*intercept = 0`
(`ols.cuh:156`). It is a real arm of theirs; it is not the arm a Python user
lands on without asking. That is the honest statement and the docstring used
to stop one sentence early.

NOT PORTED, and refused by name below: `fit_intercept`, `normalize` (both
need `preProcessData`/`postProcessData` from `cuml glm/preprocess.cuh`) and
`sample_weight` (`ols.cuh:99-110`, a sqrt-scaling of both operands and its
exact inverse afterwards). See `glm/NOT_IMPLEMENTED.tsv`.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from glm.derived.linalg.detail.lstsq import (
    OLS_ELEM_TPB,
    lstsq_eig_traced,
)


# `ols.cuh:116-126`, the switch. Their ids, so an error message names the same solver
# theirs would.
comptime OLS_ALGO_SVD_JACOBI = 0
comptime OLS_ALGO_EIG = 1
comptime OLS_ALGO_QR = 2
comptime OLS_ALGO_SVD_QR = 3


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
    """`olsFit`, their guards and their dispatch, in their order.

    `algo` defaults to `OLS_ALGO_EIG` and NOT to their `0`, because 0 selects
    `lstsqSvdJacobi`, which is not ported. Defaulting to an arm that always
    raises would make the entry point useless; defaulting to the arm cuML's
    PYTHON layer defaults to is the closest honest choice. That is a
    DEVIATION and it is recorded in `glm/NOT_IMPLEMENTED.tsv`.

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
    """`olsFit` with the DEVIATION 527 stage card. See `ols_fit`.

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
            " glm/NOT_IMPLEMENTED.tsv"
        )
    if normalize:
        raise Error(
            "olsFit: normalize is not ported, and theirs is only reachable"
            " with fit_intercept; see glm/NOT_IMPLEMENTED.tsv"
        )

    # THE GUARD. `ols.cuh:112-113`, copied exactly, including the fact that it
    # overrides whatever the caller asked for rather than erroring.
    var selected_algo = algo
    if n_cols > n_rows or n_cols == 1:
        selected_algo = OLS_ALGO_SVD_JACOBI

    if selected_algo == OLS_ALGO_EIG:
        lstsq_eig_traced(
            ctx, a, b, w, cov_a, q, qs, s_vec, ab, inv, a_alias, a_alias2,
            n_rows, n_cols, elem_tpb, trace,
        )
    elif selected_algo == OLS_ALGO_SVD_JACOBI:
        # Reached either because the caller asked for it, or -- far more
        # importantly -- because the guard above switched to it. Naming both
        # cases matters: the second one is not a user error, it is their
        # library telling us this shape needs the solver we do not have.
        if n_cols > n_rows:
            raise Error(
                "olsFit: n_cols > n_rows selects lstsqSvdJacobi"
                " (ols.cuh:113), which is NOT PORTED. The normal-equations"
                " solver cannot be used here: A^T A is singular by"
                " construction. See glm/NOT_IMPLEMENTED.tsv"
            )
        if n_cols == 1:
            raise Error(
                "olsFit: n_cols == 1 selects lstsqSvdJacobi (ols.cuh:113),"
                " which is NOT PORTED. cuML forces the same switch and says"
                " the eig solver does not support a single column"
                " (linear_regression.pyx:390). See glm/NOT_IMPLEMENTED.tsv"
            )
        raise Error(
            "olsFit: algo 0 is lstsqSvdJacobi, which is NOT PORTED. It is"
            " cuML's C++ default and the more numerically stable route; see"
            " glm/NOT_IMPLEMENTED.tsv"
        )
    elif selected_algo == OLS_ALGO_QR:
        raise Error("olsFit: algo 2 is lstsqQR, which is NOT PORTED")
    elif selected_algo == OLS_ALGO_SVD_QR:
        raise Error("olsFit: algo 3 is lstsqSvdQR, which is NOT PORTED")
    else:
        # `ols.cuh:123-125`, their default arm.
        raise Error(
            "olsFit: no algorithm with this id has been implemented"
        )
