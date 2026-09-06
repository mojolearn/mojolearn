# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`householder_qr_solve`: the batched overdetermined least squares that
`estimate_x0` needs, written here because `cublasgelsBatched` is closed.

STANDS IN FOR `cuml/cpp/src_prims/linalg/batched/matrix.cuh::b_gels`
(`:644-669`) at cuML 265b9da6 (v26.08.00). Theirs is a nine-line wrapper
over `raft::linalg::detail::cublasgelsBatched`; there is no upstream
arithmetic to transliterate, only an upstream CONTRACT: non-transpose
mode, overdetermined only (`ASSERT(m > n)`), `A` copied so the caller's
matrix survives, the solution written into the first `n` rows of `C` in
place, one `devInfoArray` entry per problem.

=============================================================================
DEVIATION 678: cuBLAS gelsBatched IS CLOSED; THE LEAST SQUARES IS SPELLED HERE
=============================================================================
THEIRS. `b_gels` is `cublasgelsBatched`, and `_arma_least_squares` calls it
twice per series with `devInfoArray = nullptr`, so a rank-deficient design
matrix produces an unusable solution and NOTHING reads the failure code.
The routine's factorization, its association order, and what it does with a
tiny pivot are all unreadable.

OURS (PORTING_RULES 0b-i, the same rule DEVIATION 674 was taken under).
HOUSEHOLDER QR WITH BACK SUBSTITUTION, one thread per series, serial
ascending, every multiply-add `identical_mul_add`, every intermediate
through `ftz`.

WHY QR AND NOT THE NORMAL EQUATIONS, WHICH THIS TREE ALREADY HAS.
`cholesky/`'s `potrf_lower` + `trsm` and `glm/impl/linalg/detail/lstsq.mojo`'s
`lstsq_eig` were both assessed and both REJECTED, for two independent
reasons:

  1. ACCURACY. Forming `A' A` squares the condition number.
     `lstsq_eig`'s own header says so ("Forming `A^T A` SQUARES the
     condition number, so this route loses roughly twice the digits an SVD
     route would"). The design matrix here is a matrix of LAGS OF ONE
     SERIES, whose columns are near-collinear exactly when the series has a
     root near the unit circle -- which is the `ar2_unit` fixture, and which
     is the regime an ARIMA user is most often in. Float32 has 7.2 decimal
     digits and `1 / eps = 8.4e6`; a design with `kappa(A) = 3e3`, which is
     unremarkable for lagged columns, gives `kappa(A'A) = 9e6`, PAST the
     Float32 limit, and the Cholesky can meet a non-positive pivot on a
     matrix that is not remotely singular. The QR's sensitivity is
     `kappa(A)`, not `kappa(A)^2`: the same problem loses about 3.5 digits
     instead of all of them.
  2. THE UPSTREAM DESIGN. cuBLAS `gelsBatched` is itself a QR-based solver.
     `assume-our-code-is-broken` says theirs is right about DESIGN. Porting
     the design means QR; substituting a normal-equations route because the
     tree happens to own one is exactly the "improvement" this repository
     bans.

  And the cost argument does not survive contact with where the time goes.
  QR is ~`2mn^2` against ~`mn^2/2` for the Gram: a factor of four on a step
  that runs ONCE per fit, against the hundreds of batched Kalman passes the
  optimizer then spends. `n` is at most 17 and in practice at most 10.

WHAT IS CHOSEN HERE AND IS THEREFORE OURS TO GATE (DEVIATION 674's rule:
a choice a closed library made for them is a choice we make and write down):

  * THE REFLECTOR'S SIGN. `s = -sign(a_jj)` and `u1 = a_jj - s*normx`, so
    the two terms have OPPOSITE signs and `|u1| >= normx`. Choosing the
    other sign makes `u1` a DIFFERENCE OF NEAR-EQUAL NUMBERS whenever the
    column is already nearly axis-aligned, which is the classical
    cancellation this formulation exists to avoid. `SABOTAGES.md` arm (l)
    flips it, and the residual on `ar2_unit` must blow up.
  * THE FOLDS. Every inner product is serial ascending through
    `identical_mul_add`, as DEVIATION 674's substitutions are.
  * THE RANK TEST. After the factorization, `|R_jj| <= LS_RANK_TOL * max_j
    |R_jj|` marks the system numerically rank deficient. This is LAPACK
    `xGELSY`'s idea (a diagonal ratio as a cheap condition estimate) with a
    fixed threshold rather than an incremental estimator. `LS_RANK_TOL =
    1e-5` is DERIVED, not observed: a ratio below it means the back
    substitution divides by a diagonal carrying fewer than about two
    significant Float32 digits, so the solution's relative error exceeds one
    percent and the "estimate" would be noise. **A compile slot must replace
    it with what `check_x0_residual_is_small` measures.**
  * WHAT HAPPENS THEN. `info` is set and the CALLER takes cuML's OWN
    degenerate arm -- `ar = ma = mu = 0`, `sigma2 = 1` -- which is the
    fallback `_arma_least_squares` already has for the too-few-observations
    case (`batched_arima.cu:687-697`). We do not invent a policy and we do
    not return garbage; and unlike theirs the code is RECORDED, as the
    `x0.info_ls` decision stage, so a vendor that refused a series where
    another did not shows up in the card one stage before it shows up
    anywhere else.

THROUGHPUT, NAMED SO IT IS NOT DISCOVERED AS A SURPRISE. Theirs parallelizes
INSIDE each problem; ours is one thread per series and serial in `m`. At the
shipped shapes (`m` a few hundred, `n <= 10`) that is nothing. At `m = 1e5`
and `n = 16` it is about 5e7 flops in ONE thread and it will be visibly slow.
That is a throughput deviation, not a correctness one, and the fix if it ever
matters is a block-per-series arm, which is the same shape `batched_kalman.cu`
already has for `rd > 8` and which this lane also does not carry.
"""

from checks.numerics import ftz, identical_mul_add, identical_sqrt


#: The widest system `estimate_x0` can present. `p, q, P, Q <= 8` and
#: `k <= 1` bound the ARMA fit's columns at `p + q + k = 17`; the AR
#: pre-fit's `p_ar = max(p*s, 2*q*s)` is bounded by the same 17 once
#: `validate_order`'s `r <= 5` has run (it forces `p + s*P <= 5` and
#: `q + s*Q + 1 <= 5`, so the reachable maxima are `n = 10` and
#: `p_ar = 8`). Anything wider raises by name in the launcher.
comptime LS_MAX_COLS = 17

#: DERIVED, not observed. See the banner.
comptime LS_RANK_TOL = Float32(1.0e-5)


@always_inline
def householder_qr_solve(
    a: MutPointer[Float32, MutAnyOrigin],
    a_base: Int,
    m: Int,
    n: Int,
    b: MutPointer[Float32, MutAnyOrigin],
    b_base: Int,
) -> Int32:
    """Solve `min |A x - b|` for one column-major `m x n` system, `m > n`.

    DESTROYS `a[a_base ..]` (it becomes the packed reflectors and `R`, as
    LAPACK's `geqrf` does) and writes `x` into `b[b_base .. b_base + n)`,
    which is `b_gels`'s in-place contract. THE CALLER MUST PASS A COPY of
    the design matrix if it needs it afterwards, which is exactly what
    theirs says (`matrix.cuh:641`: "This function copies A to avoid
    modifying the original one").

    Returns `info`: 0, or `j + 1` for the first column that is numerically
    rank deficient (a zero column norm, or a diagonal below `LS_RANK_TOL`
    times the largest). On a non-zero return `b` is NOT a solution and the
    caller must not read it.
    """
    var rdiag = InlineArray[Float32, LS_MAX_COLS](fill=Float32(0.0))
    for j in range(n):
        # column norm of A[j:m, j], serial ascending
        var sigma = Float32(0.0)
        for i in range(j, m):
            var v = ftz(a.unsafe_load(a_base + i + j * m))
            sigma = ftz(identical_mul_add(v, v, sigma))
        if sigma == Float32(0.0):
            return Int32(j + 1)
        var normx = ftz(identical_sqrt(sigma))
        var ajj = ftz(a.unsafe_load(a_base + j + j * m))
        # s = -sign(a_jj), with sign(+-0) taken as +1 so s = -1 there. The
        # POINT of this sign is that `u1 = ajj - s*normx` adds two terms of
        # the same sign instead of subtracting near-equal ones.
        var s = Float32(-1.0) if ajj >= Float32(0.0) else Float32(1.0)
        var r_jj = ftz(s * normx)
        var u1 = ftz(ajj - r_jj)
        if u1 == Float32(0.0):
            return Int32(j + 1)
        # pack w into the subdiagonal (w_j = 1 implicitly), LAPACK's layout
        for i in range(j + 1, m):
            a.unsafe_store(
                a_base + i + j * m,
                ftz(ftz(a.unsafe_load(a_base + i + j * m)) / u1),
            )
        var tau = ftz(ftz(ftz(-s) * u1) / normx)
        rdiag[j] = r_jj
        # apply H = I - tau w w' to the trailing columns
        for c in range(j + 1, n):
            var acc = ftz(a.unsafe_load(a_base + j + c * m))
            for i in range(j + 1, m):
                var w = ftz(a.unsafe_load(a_base + i + j * m))
                var x = ftz(a.unsafe_load(a_base + i + c * m))
                acc = ftz(identical_mul_add(w, x, acc))
            var td = ftz(tau * acc)
            a.unsafe_store(
                a_base + j + c * m, ftz(a.unsafe_load(a_base + j + c * m) - td)
            )
            for i in range(j + 1, m):
                var w = ftz(a.unsafe_load(a_base + i + j * m))
                var cur = ftz(a.unsafe_load(a_base + i + c * m))
                a.unsafe_store(
                    a_base + i + c * m, ftz(identical_mul_add(-td, w, cur))
                )
        # and to the right-hand side
        var accb = ftz(b.unsafe_load(b_base + j))
        for i in range(j + 1, m):
            var w = ftz(a.unsafe_load(a_base + i + j * m))
            var x = ftz(b.unsafe_load(b_base + i))
            accb = ftz(identical_mul_add(w, x, accb))
        var tdb = ftz(tau * accb)
        b.unsafe_store(b_base + j, ftz(b.unsafe_load(b_base + j) - tdb))
        for i in range(j + 1, m):
            var w = ftz(a.unsafe_load(a_base + i + j * m))
            var cur = ftz(b.unsafe_load(b_base + i))
            b.unsafe_store(b_base + i, ftz(identical_mul_add(-tdb, w, cur)))
    # THE RANK TEST, before any division by a diagonal
    var rmax = Float32(0.0)
    for j in range(n):
        var v = abs(rdiag[j])
        if v > rmax:
            rmax = v
    if rmax == Float32(0.0):
        return Int32(1)
    for j in range(n):
        if abs(rdiag[j]) <= ftz(LS_RANK_TOL * rmax):
            return Int32(j + 1)
    # back substitution on R, in place on b, ascending in the inner fold
    var i = n - 1
    while i >= 0:
        var acc = ftz(b.unsafe_load(b_base + i))
        for c in range(i + 1, n):
            var u = ftz(a.unsafe_load(a_base + i + c * m))
            var xc = ftz(b.unsafe_load(b_base + c))
            acc = ftz(identical_mul_add(-u, xc, acc))
        b.unsafe_store(b_base + i, ftz(acc / rdiag[i]))
        i -= 1
    return Int32(0)
