"""The host symmetric eigensolver that stands where RAFT calls cuSOLVER.

NOT A PORT OF A FILE. `raft/sparse/solver/detail/lanczos.cuh:175`
(`lanczos_solve_ritz`) hands the `ncv x ncv` projected matrix to
`raft::linalg::eig_dc`, which is cuSOLVER `syevd` -- CLOSED, no source to
transliterate (PORTING_RULES 0b-i's one exception). What it returns is
eigenvalues ASCENDING and eigenvectors in COLUMNS with a sign the solver
chose. This file returns the same three things from one host routine whose
every floating-point operation is spelled through `mojo_only/numerics.mojo`,
so that the projected eigenproblem -- the only dense linear algebra in the
Lanczos loop -- is a pure function of the input bits on every vendor. THE
HOST IS PART OF THE NUMERICAL PLAN: `spectral/README.md` says so in those
words, and this file is why.

============ DEVIATION 770: THE EIGENVECTOR SIGN IS PINNED BY A RULE ======
THEIRS: `syevd` leaves each eigenvector's sign to the solver. The sign
propagates into the Ritz vectors (`V^T e`), into the restart (the Ritz
vectors become `V[0..k)` of the next pass), and into the embedding, so two
vendors' cuSOLVER builds -- or one vendor's two versions -- can return an
embedding that differs by a column sign and a different Lanczos trajectory
after the first restart.
OURS: after the solve and the ascending (value, index) sort, every column is
flipped so that ITS FIRST NONZERO COMPONENT IN INDEX ORDER IS POSITIVE.
"Nonzero" is `x != 0.0`, which is false for both `+0.0` and `-0.0`, so a
leading signed zero is skipped rather than consulted for its sign bit
(`check_tsolve_sign_pin_skips_signed_zero` plants one). A column that is
entirely zero cannot occur for an orthonormal basis and is left alone.
MEASURED: `check_spectral_device_equals_oracle` with the sabotage
`MOJOLEARN_SPECTRAL_SABOTAGE_SIGN_FLIP` (the device arm re-flips after the
shared solve) FAILS at the first Ritz-vector stage -- README, sabotage (b).
============ DEVIATION 771: cuSOLVER syevd -> HOST CYCLIC JACOBI ==========
THEIRS: divide-and-conquer on the device, closed.
OURS: cyclic Jacobi (Numerical Recipes `jacobi`, the classical rotation with
the `tresh` skip for the first three sweeps and the relative-size annihilation
after the fourth), on the host, in the dtype asked for. Under IDENTICAL the
Float32 arm routes every multiply-add through `identical_mul_add`, every
stored intermediate through `ftz`, every square root through
`identical_sqrt`; division is the host's IEEE `/`. The Float64 arm (the
oracle's reference) uses plain operations. Jacobi rather than the
Float64 host Jacobi in `decomposition/mojo_only/jacobi_eigh.mojo` because
that one is Float64-only and this solver runs INSIDE the fit, not beside it;
that one is what `check_tsolve_against_float64_jacobi` compares against.
WHY THE RESTART MATRIX IS NOT TRIDIAGONAL, which is why a general symmetric
solver is needed and not a `tql2`: after a restart `lanczos_solve_ritz`
writes `beta_k` into row `k` and column `k` (`lanczos.cuh:87-98`), so the
matrix is "arrowhead plus tridiagonal", and Jacobi does not care.
======================================================================
"""

from std.math import sqrt

from mojo_only.numerics import ftz, identical_mul_add, identical_sqrt


# ---------------------------------------------------------------------------
# dtype-generic seams: Float32 goes through the IDENTICAL helpers, Float64
# (the reference arm) through plain operations.
# ---------------------------------------------------------------------------


@always_inline
def hfma[dt: DType](a: Scalar[dt], b: Scalar[dt], c: Scalar[dt]) -> Scalar[dt]:
    """`a*b + c` at a seam: `identical_mul_add` for Float32, plain for the
    Float64 reference."""
    comptime if dt == DType.float32:
        return rebind[Scalar[dt]](
            identical_mul_add(
                rebind[Float32](a), rebind[Float32](b), rebind[Float32](c)
            )
        )
    else:
        return a * b + c


@always_inline
def hflush[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    """Row 10's seam: `ftz` for Float32, identity for Float64."""
    comptime if dt == DType.float32:
        return rebind[Scalar[dt]](ftz(rebind[Float32](x)))
    else:
        return x


@always_inline
def hsqrt[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    """`identical_sqrt` for Float32 (correctly rounded on every column), the
    host's `sqrt` for Float64."""
    comptime if dt == DType.float32:
        return rebind[Scalar[dt]](identical_sqrt(rebind[Float32](x)))
    else:
        return sqrt(x)


def _rotate[
    dt: DType
](
    mut a: List[Scalar[dt]],
    n: Int,
    i: Int,
    j: Int,
    k: Int,
    l: Int,
    s: Scalar[dt],
    tau: Scalar[dt],
):
    """NR's `ROTATE`: `g = a[i][j]; h = a[k][l]; a[i][j] = g - s*(h + g*tau);
    a[k][l] = h + s*(g - h*tau)`. Spelled as two fmas per entry, seams
    flushed."""
    var g = a[i * n + j]
    var h = a[k * n + l]
    var t1 = hflush[dt](hfma[dt](g, tau, h))  # h + g*tau
    a[i * n + j] = hflush[dt](hfma[dt](-s, t1, g))  # g - s*t1
    var t2 = hflush[dt](hfma[dt](-h, tau, g))  # g - h*tau
    a[k * n + l] = hflush[dt](hfma[dt](s, t2, h))  # h + s*t2


def symmetric_eig_host[
    dt: DType
](
    mut a: List[Scalar[dt]],
    n: Int,
    mut eigenvalues: List[Scalar[dt]],
    mut eigenvectors: List[Scalar[dt]],
    max_sweeps: Int = 60,
) raises -> Int:
    """Eigen-decompose the symmetric `n x n` row-major `a` (DESTROYED).

    On return `eigenvalues` holds `n` values ASCENDING by the total order
    (value, original index), and `eigenvectors` is `n x n` row-major with
    eigenvector `c` in COLUMN `c` -- cuSOLVER's convention, the one
    `lanczos_solve_ritz` reads `eigenvectors.data_handle() + (ncv - k) * ncv`
    from. Every column's sign is pinned by DEVIATION 770. Returns the number
    of sweeps used (an integer a card can carry).
    """
    if n <= 0:
        raise Error("symmetric_eig_host: n must be positive")
    var zero = Scalar[dt](0)
    var one = Scalar[dt](1)
    var v = List[Scalar[dt]]()
    var d = List[Scalar[dt]]()
    var b = List[Scalar[dt]]()
    var z = List[Scalar[dt]]()
    for i in range(n):
        for j in range(n):
            v.append(one if i == j else zero)
        d.append(a[i * n + i])
        b.append(a[i * n + i])
        z.append(zero)

    var sweeps_used = 0
    for sweep in range(1, max_sweeps + 1):
        sweeps_used = sweep
        var sm = zero
        for p in range(n - 1):
            for q in range(p + 1, n):
                sm = hflush[dt](sm + abs(a[p * n + q]))
        if sm == zero:
            break
        var tresh = zero
        if sweep < 4:
            # 0.2 * sm / (n*n): two roundings, as NR writes it.
            var t0 = hflush[dt](Scalar[dt](0.2) * sm)
            tresh = hflush[dt](t0 / Scalar[dt](n * n))
        for p in range(n - 1):
            for q in range(p + 1, n):
                var apq = a[p * n + q]
                var g = hflush[dt](Scalar[dt](100) * abs(apq))
                var dp_abs = abs(d[p])
                var dq_abs = abs(d[q])
                if (
                    sweep > 4
                    and hflush[dt](dp_abs + g) == dp_abs
                    and hflush[dt](dq_abs + g) == dq_abs
                ):
                    a[p * n + q] = zero
                elif abs(apq) > tresh:
                    var h = hflush[dt](d[q] - d[p])
                    var t: Scalar[dt]
                    var h_abs = abs(h)
                    if hflush[dt](h_abs + g) == h_abs:
                        t = hflush[dt](apq / h)
                    else:
                        var theta = hflush[dt](hflush[dt](Scalar[dt](0.5) * h) / apq)
                        var th2 = hflush[dt](hfma[dt](theta, theta, one))
                        var den = hflush[dt](abs(theta) + hsqrt[dt](th2))
                        t = hflush[dt](one / den)
                        if theta < zero:
                            t = -t
                    var t2 = hflush[dt](hfma[dt](t, t, one))
                    var c = hflush[dt](one / hsqrt[dt](t2))
                    var s = hflush[dt](t * c)
                    var tau = hflush[dt](s / hflush[dt](one + c))
                    var hh = hflush[dt](t * apq)
                    z[p] = hflush[dt](z[p] - hh)
                    z[q] = hflush[dt](z[q] + hh)
                    d[p] = hflush[dt](d[p] - hh)
                    d[q] = hflush[dt](d[q] + hh)
                    a[p * n + q] = zero
                    for j in range(0, p):
                        _rotate[dt](a, n, j, p, j, q, s, tau)
                    for j in range(p + 1, q):
                        _rotate[dt](a, n, p, j, j, q, s, tau)
                    for j in range(q + 1, n):
                        _rotate[dt](a, n, p, j, q, j, s, tau)
                    for j in range(0, n):
                        _rotate[dt](v, n, j, p, j, q, s, tau)
        for p in range(n):
            b[p] = hflush[dt](b[p] + z[p])
            d[p] = b[p]
            z[p] = zero

    # ---- ascending by the TOTAL ORDER (value, index): a `<` on the values,
    # ties (including +0.0 against -0.0, which compare equal) by index.
    var order = List[Int]()
    for i in range(n):
        order.append(i)
    for i in range(1, n):
        var key = order[i]
        var j = i - 1
        while j >= 0 and d[order[j]] > d[key]:
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = key
    eigenvalues.clear()
    eigenvectors.clear()
    for _ in range(n * n):
        eigenvectors.append(zero)
    for c in range(n):
        var src = order[c]
        eigenvalues.append(d[src])
        for r in range(n):
            eigenvectors[r * n + c] = v[r * n + src]
    pin_column_signs[dt](eigenvectors, n, n)
    return sweeps_used


def pin_column_signs[
    dt: DType
](mut m: List[Scalar[dt]], n_rows: Int, n_cols: Int):
    """DEVIATION 770's rule on a row-major `n_rows x n_cols` matrix: for
    each column, the first component (ascending row index) that is not a
    zero of either sign is made positive by negating the whole column when
    it is negative. An all-zero column is left untouched."""
    var zero = Scalar[dt](0)
    for c in range(n_cols):
        var r = 0
        while r < n_rows and m[r * n_cols + c] == zero:
            r += 1
        if r < n_rows and m[r * n_cols + c] < zero:
            for rr in range(n_rows):
                m[rr * n_cols + c] = -m[rr * n_cols + c]
