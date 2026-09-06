#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""What the normal-equations route COSTS, priced in digits rather than time.

`bench_sklearn.py` reports two OLS arms because we and scikit-learn solve
least squares by different algorithms:

    LinearRegression              LAPACK gelsd, an SVD of the full design
    Ridge(alpha=0, "cholesky")    forms X^T X and solves it -- OUR route,
                                  the same class as `lstsq_eig`

A timing harness can only ever show the first one being slower. It cannot show
what that extra work BUYS, and the thing it buys is the whole reason cuML's
C++ `olsFit` defaults to SVD (`ols.cuh:67`) and scikit-learn offers nothing
else on `LinearRegression`.

**Forming X^T X squares the condition number.** A design with condition number
k is solved as if it had condition number k^2, so the normal-equations route
loses roughly twice as many digits. In float32 there are only about 7 to begin
with.

This script sweeps the conditioning of X and reports the coefficient error of
each route against a KNOWN answer, so the trade has a number on both sides.

It runs entirely in scikit-learn and numpy: it prices the ALGORITHM, not our
implementation. Our `lstsq_eig` should land near the cholesky column, and
where it does not, that is a defect of ours rather than of the method.
`DivideByNonZero` in `lstsq.cuh` means ours drops near-zero eigenvalues rather
than dividing by them, so ours may degrade more gracefully than cholesky at
the far end -- which is a claim to TEST, not to assume.

    python bench/ols_conditioning.py
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge


def design(n_rows, n_cols, cond, dtype, seed=0):
    """A matrix with a PRESCRIBED condition number.

    Built from an SVD with geometrically spaced singular values, so `cond` is
    exact rather than approximate: U diag(s) V^T with s spanning [1/cond, 1].
    """
    rng = np.random.default_rng(seed)
    u, _ = np.linalg.qr(rng.standard_normal((n_rows, n_cols)))
    v, _ = np.linalg.qr(rng.standard_normal((n_cols, n_cols)))
    s = np.geomspace(1.0, 1.0 / cond, n_cols)
    return np.ascontiguousarray((u * s) @ v.T, dtype=dtype)


def main():
    n_rows, n_cols = 20000, 16
    w_true = 1.0 + 0.1 * np.arange(n_cols)

    for dtype, label in ((np.float32, "float32"), (np.float64, "float64")):
        print(f"\n{label}, {n_rows} x {n_cols}, noiseless y = X w")
        print(f"{'cond(X)':>10}  {'gelsd (SVD)':>14}  {'cholesky (ours)':>16}  "
              f"{'ratio':>8}")
        for cond in (1e1, 1e2, 1e3, 1e4, 1e5, 1e6):
            x = design(n_rows, n_cols, cond, dtype)
            y = np.ascontiguousarray(x @ w_true, dtype=dtype)

            svd = LinearRegression(fit_intercept=False).fit(x, y).coef_
            chol = Ridge(
                alpha=0.0, solver="cholesky", fit_intercept=False
            ).fit(x, y).coef_

            e_svd = np.abs(svd - w_true).max()
            e_chol = np.abs(chol - w_true).max()
            ratio = e_chol / e_svd if e_svd > 0 else float("inf")

            # gelsd DISCARDS singular values below its rcond cutoff, which
            # is a deliberate pseudo-inverse truncation and not a failure.
            # Once it fires, "error against w_true" stops being the right
            # question for either method, so the row is marked rather than
            # silently compared.
            rank = np.linalg.matrix_rank(x)  # in X's OWN precision, not upcast
            mark = "" if rank == n_cols else f"   <- gelsd truncated to rank {rank}"
            print(f"{cond:10.0e}  {e_svd:14.3e}  {e_chol:16.3e}  "
                  f"{ratio:8.1f}x{mark}")

    print(
        "\nRead the ratio column, not the absolute errors. It is how many"
        "\ntimes worse the normal-equations answer is at that conditioning,"
        "\nand it is the price of the speed reported in bench_sklearn.py."
        "\n"
        "\nTHE RATIO INVERTS AT THE BOTTOM OF EACH TABLE AND THAT IS REAL."
        "\nOnce X is ill-conditioned enough, gelsd stops inverting the small"
        "\nsingular values and DROPS them instead. Its answer is then the"
        "\nminimum-norm solution of a truncated problem, which is deliberately"
        "\nnot w_true, so its 'error' grows while cholesky is still returning"
        "\na fully-determined -- and meaningless -- answer. Neither number is"
        "\nan accuracy result in that regime."
        "\n"
        "\nThat truncation is worth recognising: it is the same guard our own"
        "\nroute has. `lstsqEig`'s DivideByNonZero (raft lstsq.cuh) drops a"
        "\nnear-zero EIGENVALUE rather than dividing by it, turning the"
        "\ninverse into a pseudo-inverse. So ours is not plain cholesky at the"
        "\nfar end, and where our curve actually sits between these two"
        "\ncolumns is an OPEN QUESTION this script cannot answer -- it prices"
        "\nthe algorithm class, not our implementation. Running our lstsq_eig"
        "\non these exact designs is the missing measurement."
    )


if __name__ == "__main__":
    main()
