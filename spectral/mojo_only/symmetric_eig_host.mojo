# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
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
writes `beta_k` into row `k` and column `k` (`lanczos.cuh:86-98`), so the
matrix is "arrowhead plus tridiagonal", and Jacobi does not care.
============ DEVIATION 781: `ROTATE` IS TWO FMAS, NOT FOUR ROUNDINGS =====
NR's C is `a[i][j] = g - s*(h + g*tau)` and `a[k][l] = h + s*(g - h*tau)`,
four roundings per entry as written. A C compiler with contraction on
(nvcc's default, and clang's `-ffp-contract=fast`) gives two. This file
pins the FUSED two, `hfma(g, tau, h)` then `hfma(-s, t1, g)`, because a
fused seam has a portable spelling and "whatever the compiler did" does
not. CHOSEN, and it carries a sabotage arm
(`MOJOLEARN_SPECTRAL_SABOTAGE_ROTATE_UNFUSED`) that must fail the gate.
Contract section 5.3, seam J4.
============ THE SWEEP CAP IS CHOSEN, AND IT IS SILENT (DEVIATION 780) ====
This is one of the TWO clauses that survive DEVIATION 780's correction of
2026-08-23. Three of that deviation's original five were struck once cuVS
26.08 was checked out and turned out to spell them verbatim; this one
stands, because this solver is not a mirror of anything. It stands where
cuSOLVER `syevd` is called, and the cap is a number nobody upstream ever
had to pick.
`max_sweeps = 60`, where NR's `jacobi` uses 50 and calls `nrerror` when it
runs out. This routine RETURNS the unconverged basis and its sweep count
instead of raising. That is a bound this lane picked and a failure mode it
chose to make quiet, so it is CHOSEN, it is recorded here, and
`MOJOLEARN_SPECTRAL_SABOTAGE_SWEEP_CAP` drops the cap to 3 to prove the
cap is REACHED and not merely present. Contract section 4, C4, seam J6.
======================================================================
"""

from std.math import sqrt
from std.sys.compile import is_defined

from mojo_only.numerics import ftz, identical_mul_add, identical_sqrt

#: SABOTAGE. Drop the Jacobi sweep cap from 60 to 3, so the sweeps stop
#: before the off-diagonal is annihilated and the returned basis is not an
#: eigenbasis. CHOSEN bound C4 (DEVIATION 780).
#:
#: REACH, stated honestly and BEFORE it is run: this solver is SHARED by
#: the device arm (`lanczos_solve_ritz`) and the oracle
#: (`_host_solve_ritz`), so this arm moves BOTH and is expected to be
#: INERT against `check_spectral_device_equals_oracle`. Its reach is
#: against the INDEPENDENT references instead: `check_spectral_path_exact`
#: holds P_64's normalized Ritz values to `1e-4` of `1 - cos(pi j / 63)`,
#: and three sweeps on a 20x20 cannot get there. That is the gate this arm
#: must fail.
comptime SAB_SWEEP_CAP = is_defined["MOJOLEARN_SPECTRAL_SABOTAGE_SWEEP_CAP"]()

#: SABOTAGE. Spell `ROTATE` as NR's FOUR roundings instead of the pinned
#: TWO fmas (DEVIATION 781, seam J4).
#:
#: REACH, stated honestly and BEFORE it is run: shared solver again, so
#: this is expected to be INERT against device == oracle, AND the
#: perturbation is a last-bit one that every existing tolerance in this
#: lane (`1e-4` on the closed forms, `2e-6` on the Float64 Jacobi compare)
#: absorbs. **THIS ARM IS THEREFORE EXPECTED TO BE REACHED BUT INERT, AND
#: THAT IS A HOLE, NOT A PASS.** Seam J4 has no gate with teeth until this
#: lane records a CERTIFICATE -- an FNV hash of `symmetric_eig_host`'s
#: output on a pinned fixture, compared against a literal. That check is
#: OWED and is named in `spectral/README.md`.
comptime SAB_ROTATE_UNFUSED = is_defined[
    "MOJOLEARN_SPECTRAL_SABOTAGE_ROTATE_UNFUSED"
]()

#: SABOTAGE. Make the ascending sort ANTI-STABLE for ties (`>=` where the
#: pinned spelling has `>`), so two equal eigenvalues come back in the
#: reverse of their original index order. This is the arm for DEVIATION
#: 778's TIE RULE, and it exists because a convention with no arm behind it
#: is pinned in prose only.
#:
#: REACH: shared solver again, so it is EXPECTED INERT against
#: `check_spectral_device_equals_oracle`. Its instrument is
#: `check_tsolve_tie_order_is_stable`, a host check on an exactly
#: degenerate 4x4 that asserts WHICH original index each tied column came
#: from. That check is the whole reason DEVIATION 778 is falsifiable.
comptime SAB_TIE_REVERSE = is_defined[
    "MOJOLEARN_SPECTRAL_SABOTAGE_TIE_REVERSE"
]()


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
    comptime if SAB_ROTATE_UNFUSED:
        # NR's C, four roundings: g - s*(h + g*tau), h + s*(g - h*tau)
        var u1 = hflush[dt](h + hflush[dt](g * tau))
        a[i * n + j] = hflush[dt](g - hflush[dt](s * u1))
        var u2 = hflush[dt](g - hflush[dt](h * tau))
        a[k * n + l] = hflush[dt](h + hflush[dt](s * u2))
    else:
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

    var cap = max_sweeps
    comptime if SAB_SWEEP_CAP:
        cap = 3
    var sweeps_used = 0
    for sweep in range(1, cap + 1):
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
        while j >= 0:
            # STRICT `>` is the tie rule (DEVIATION 778): equal values keep
            # their ORIGINAL index order, because insertion sort with a
            # strict compare is stable. `>=` reverses ties, which is the
            # sabotage.
            var shift: Bool
            comptime if SAB_TIE_REVERSE:
                shift = d[order[j]] >= d[key]
            else:
                shift = d[order[j]] > d[key]
            if not shift:
                break
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
