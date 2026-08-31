# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host oracles: a float32 serial replay and a float64 reference.

NOT A PORT. cuML and cuVS check nothing about Cholesky at the bit level and
need not: `cuml/cpp/tests/sg/lars_test.cu:110-125` compares
`updateCholesky` against `cusolverDnpotrf` at a tolerance, which is the right
test for a library that ships one backend. We ship three from one source, so
the device arm is gated BIT FOR BIT under IDENTICAL against the float32 replay
below, and the replay is gated against the float64 reference to a tolerance.
Both are written FIRST and gated FIRST.

TWO ORACLES, TWO JOBS
---------------------
`oracle_potrf_lower`   float32, SERIAL, through the same helpers the device
                       uses (`identical_mul_add`, `identical_div`,
                       `identical_sqrt`, `ftz`), with every formula spelled
                       here a SECOND time rather than imported from
                       `cholesky/original/potrf.mojo` -- so the gate compares
                       two spellings of one arithmetic and not a function
                       against itself. It records the SAME card stages under
                       the SAME tags, so `first_divergence` names the first
                       stage that moved rather than just saying the answers
                       differ.
`reference_potrf_lower_f64`
                       float64, the textbook UNBLOCKED factorization through
                       `std.math`. Its job is tolerance sanity -- DEVIATION
                       1635's cost, measured per fixture instead of asserted
                       -- and the log-determinant's closed form.

THE ONE PLACE THIS ORACLE DOES NOT RE-SPELL THE ARITHMETIC, and it is
deliberate: the trailing update calls `gemm/original/gemm_oracle.mojo::
gemm_oracle`, the normative answer of profile
`mojolearn.identical.gemm.fp32.v1`, exactly as the device path calls that
profile's kernel. Re-spelling the balanced-tree fold here would put a SECOND
opinion about the gemm profile in the tree, which
`gemm/original/gemm_identical.mojo::contract_partition` is explicit about
refusing ("a second spelling of the leaf rule would be a second thing that
can be wrong, and the shape table already shipped one such re-spelling and
got it wrong"). The gemm lane's own gates certify that its kernel matches
that oracle at 62 shapes; this lane inherits the certificate rather than
re-earning it, and `cholesky/README.md` says so under WHAT THIS LANE REUSES
RATHER THAN REWRITES.

WHY THE BLOCKED STRUCTURE IS REPLAYED AND NOT SIMPLIFIED
---------------------------------------------------------
The obvious oracle is the textbook unblocked loop, and it would be WRONG here
-- not approximately wrong, categorically. DEVIATION 1630's whole argument is
that the panel partition changes the bracketing of the sum, so the unblocked
factorization and the `NB = 32` blocked one are two different float32
answers for `n > 32`. An oracle that ignored the blocking would fail against
a correct kernel and, worse, would pass against a kernel that had silently
stopped blocking. So this replays the panels, the panel solve and the
trailing update in the driver's order.

`reference_potrf_lower_f64` IS the unblocked loop, on purpose: in float64 at
these sizes the difference between the two bracketings is far below the
float32 tolerance the comparison uses, and an unblocked float64 reference is
the thing a reader can check against a textbook.
"""

from std.math import log, sqrt

from core.identity_trace import IdentityTrace
from cholesky.original.potrf import chol_panel_tag
from gemm.original.gemm_oracle import OP_NT, gemm_oracle
from original.numerics import (
    ftz,
    identical_div,
    identical_log,
    identical_mul,
    identical_mul_add,
    identical_sqrt,
)


@fieldwise_init
struct CholOracle(Movable):
    """One float32 replay: the factor, LAPACK's `info`, the panels walked."""

    var l: List[Float32]
    var info: Int
    var n_panels: Int
    var nb: Int


@fieldwise_init
struct CholOracleF64(Movable):
    """The float64 reference's factor and `info`."""

    var l: List[Float64]
    var info: Int


def oracle_add_jitter(
    a_in: List[Float32], n: Int, jitter: Float32
) -> List[Float32]:
    """`jitter_diag_kernel`, replayed. `ftz(d + jitter)` per diagonal cell and
    nothing else touched. DEVIATION 1637."""
    var a = a_in.copy()
    for i in range(n):
        a[i * n + i] = ftz(ftz(a[i * n + i]) + jitter)
    return a^


def oracle_potrf_lower(
    a_in: List[Float32],
    n: Int,
    nb: Int,
    mut trace: IdentityTrace,
) raises -> CholOracle:
    """The blocked right-looking factorization, float32, on the host.

    A second spelling of `cholesky/original/potrf.mojo`'s three kernels and
    its driver loop, in the driver's order, recording the driver's tags.

    THE PANEL'S EARLY EXIT IS REPLAYED EXACTLY. When the pivot at column `jc`
    fails, the device kernel has written columns `[j0, jc)` of the panel and
    NOTHING of column `jc` -- thread 0 stops before the diagonal store and
    every other thread returns at the next barrier without touching its row.
    The `break` below leaves the array in precisely that state, which is what
    makes `check_pivot_failure_is_identical` able to compare the PARTIAL
    factor rather than only the `info`.
    """
    var a = a_in.copy()
    var info = 0
    var p = 0
    var j0 = 0
    while j0 < n:
        var w = nb
        if j0 + w > n:
            w = n - j0
        var n_trail = n - j0 - w

        # ---- panel_factor_kernel ---------------------------------------
        for c in range(w):
            var jc = j0 + c
            var s = ftz(a[jc * n + jc])
            for k in range(j0, jc):
                var v = ftz(a[jc * n + k])
                s = ftz(identical_mul_add(-v, v, s))
            if not (s > Float32(0.0)):
                info = jc + 1
                break
            a[jc * n + jc] = ftz(identical_sqrt(s))
            var ljj = ftz(a[jc * n + jc])
            for i in range(c + 1, w):
                var r = j0 + i
                var t = ftz(a[r * n + jc])
                for k in range(j0, jc):
                    var lrk = ftz(a[r * n + k])
                    var lck = ftz(a[jc * n + k])
                    t = ftz(identical_mul_add(-lrk, lck, t))
                a[r * n + jc] = ftz(identical_div(t, ljj))
        _record_matrix(trace, chol_panel_tag("chol", p, "factored"), a, n * n)
        if info != 0:
            p += 1
            break

        if n_trail > 0:
            # ---- trsm_panel_kernel -------------------------------------
            for idx in range(n_trail):
                var r = j0 + w + idx
                for c in range(w):
                    var jc = j0 + c
                    var t = ftz(a[r * n + jc])
                    for k in range(j0, jc):
                        var lrk = ftz(a[r * n + k])
                        var lck = ftz(a[jc * n + k])
                        t = ftz(identical_mul_add(-lrk, lck, t))
                    var ljj = ftz(a[jc * n + jc])
                    a[r * n + jc] = ftz(identical_div(t, ljj))
            _record_matrix(trace, chol_panel_tag("chol", p, "solved"), a, n * n)

            # ---- pack_panel_kernel, identical_gemm_into, subtract -------
            var packed = List[Float32]()
            for i in range(n_trail):
                for c in range(w):
                    packed.append(a[(j0 + w + i) * n + j0 + c])
            # THE GEMM PROFILE'S OWN ANSWER, not a re-spelling of it.
            var g = gemm_oracle(
                packed, packed, OP_NT, n_trail, n_trail, w
            )
            for i in range(n_trail):
                for j in range(i + 1):
                    var cur = ftz(a[(j0 + w + i) * n + j0 + w + j])
                    var upd = ftz(g[i * n_trail + j])
                    a[(j0 + w + i) * n + j0 + w + j] = ftz(cur - upd)
            _record_matrix(trace, chol_panel_tag("chol", p, "trailing"), a, n * n)

        p += 1
        j0 += nb

    if info == 0:
        # zero_upper_kernel. DEVIATION 1640.
        for i in range(n):
            for j in range(i + 1, n):
                a[i * n + j] = Float32(0.0)
    _record_matrix(trace, "chol.factor", a, n * n)
    var nb_record = List[Int32]()
    nb_record.append(Int32(nb))
    nb_record.append(Int32(p))
    nb_record.append(Int32(info))
    trace.record_list_i32("chol.nb", nb_record)
    return CholOracle(a^, info, p, nb)


def _record_matrix(
    mut trace: IdentityTrace, tag: String, values: List[Float32], count: Int
) raises:
    """`record_host` over a mutable copy. `record_host` wants a
    mutable-origin pointer and a borrowed `List` yields an immutable one;
    `core/identity_trace.mojo::record_list_f32` makes the same copy for the
    same reason and says why a generic signature is not worth it."""
    var tmp = values.copy()
    trace.record_host(tag, tmp.unsafe_ptr(), count)
    _ = tmp^


def oracle_trsm_lower(
    l: List[Float32],
    b_in: List[Float32],
    n: Int,
    nrhs: Int,
    ld: Int,
) -> List[Float32]:
    """`trsm_lower_kernel`, replayed. `k` ASCENDING."""
    var b = b_in.copy()
    for j in range(nrhs):
        for i in range(n):
            var t = ftz(b[i * nrhs + j])
            for k in range(i):
                var lik = ftz(l[i * ld + k])
                var bk = ftz(b[k * nrhs + j])
                t = ftz(identical_mul_add(-lik, bk, t))
            var lii = ftz(l[i * ld + i])
            b[i * nrhs + j] = ftz(identical_div(t, lii))
    return b^


def oracle_trsm_upper(
    l: List[Float32],
    b_in: List[Float32],
    n: Int,
    nrhs: Int,
    ld: Int,
) -> List[Float32]:
    """`trsm_upper_kernel`, replayed. Rows descend, the inner sum ASCENDS."""
    var b = b_in.copy()
    for j in range(nrhs):
        for ii in range(n):
            var i = n - 1 - ii
            var t = ftz(b[i * nrhs + j])
            for k in range(i + 1, n):
                var lki = ftz(l[k * ld + i])
                var bk = ftz(b[k * nrhs + j])
                t = ftz(identical_mul_add(-lki, bk, t))
            var lii = ftz(l[i * ld + i])
            b[i * nrhs + j] = ftz(identical_div(t, lii))
    return b^


def oracle_cho_solve(
    l: List[Float32],
    b_in: List[Float32],
    n: Int,
    nrhs: Int,
    mut trace: IdentityTrace,
) raises -> List[Float32]:
    """`cho_solve`, replayed, recording the device's two stage tags."""
    var y = oracle_trsm_lower(l, b_in, n, nrhs, n)
    _record_matrix(trace, "chol.solve.forward", y, n * nrhs)
    var x = oracle_trsm_upper(l, y, n, nrhs, n)
    _record_matrix(trace, "chol.solve.back", x, n * nrhs)
    return x^


def oracle_logdet(
    l: List[Float32], n: Int, mut trace: IdentityTrace
) raises -> Float32:
    """`chol_logdet`, replayed: extract the diagonal, then the ascending fold
    of its logs. Records `chol.diag` and `chol.logdet`."""
    var diag = List[Float32]()
    for j in range(n):
        diag.append(l[j * n + j])
    _record_matrix(trace, "chol.diag", diag, n)
    var acc = Float32(0.0)
    for j in range(n):
        acc = ftz(acc + ftz(identical_log(ftz(diag[j]))))
    var out = ftz(identical_mul(Float32(2.0), acc))
    var one = List[Float32]()
    one.append(out)
    _record_matrix(trace, "chol.logdet", one, 1)
    return out


def oracle_rank1_update(
    l_in: List[Float32], n: Int, ld: Int
) raises -> List[Float32]:
    """`cholesky/derived/linalg/detail/cholesky_r1_update.mojo`, replayed.

    On entry row `n-1` holds the new row of `A`; on exit it holds the new row
    of `L` and the new diagonal. Raises by name on a non-positive new
    diagonal, exactly as the device form does with `eps < 0`.
    """
    var l = l_in.copy()
    var m = n - 1
    var s = Float32(0.0)
    if m > 0:
        var x = List[Float32]()
        for k in range(m):
            x.append(l[(n - 1) * ld + k])
        var y = oracle_trsm_lower(l, x, m, 1, ld)
        for k in range(m):
            l[(n - 1) * ld + k] = y[k]
        var acc = Float32(0.0)
        for k in range(m):
            var v = ftz(y[k])
            acc = ftz(identical_mul_add(v, v, acc))
        s = acc
    var a22 = ftz(l[(n - 1) * ld + n - 1])
    var v = ftz(a22 - ftz(s))
    if not (v > Float32(0.0)):
        raise Error(
            "oracle_rank1_update: the new diagonal of rank "
            + String(n)
            + " is not positive"
        )
    l[(n - 1) * ld + n - 1] = ftz(identical_sqrt(v))
    return l^


# ===========================================================================
# THE FLOAT64 REFERENCE
# ===========================================================================


def reference_potrf_lower_f64(
    a_in: List[Float32], n: Int
) -> CholOracleF64:
    """The textbook UNBLOCKED lower Cholesky in float64, `std.math.sqrt`.

    Deliberately unblocked; the file header says why. Deliberately float64;
    DEVIATION 1635 says why, and this function is the only float64 in the
    lane. Its `info` uses the same `not (s > 0)` spelling so a comparison of
    the two `info` values is a comparison of the same decision at two
    precisions rather than of two decisions.
    """
    var a = List[Float64]()
    for i in range(n * n):
        a.append(Float64(a_in[i]))
    var info = 0
    for j in range(n):
        var s = a[j * n + j]
        for k in range(j):
            var v = a[j * n + k]
            s = s - v * v
        if not (s > Float64(0.0)):
            info = j + 1
            break
        a[j * n + j] = sqrt(s)
        var ljj = a[j * n + j]
        for i in range(j + 1, n):
            var t = a[i * n + j]
            for k in range(j):
                t = t - a[i * n + k] * a[j * n + k]
            a[i * n + j] = t / ljj
    if info == 0:
        for i in range(n):
            for j in range(i + 1, n):
                a[i * n + j] = Float64(0.0)
    return CholOracleF64(a^, info)


def reference_logdet_f64(l: List[Float64], n: Int) -> Float64:
    """`2 * sum log(L[j][j])` in float64, ascending, `std.math.log`."""
    var acc = Float64(0.0)
    for j in range(n):
        acc = acc + log(l[j * n + j])
    return Float64(2.0) * acc


def reference_solve_f64(
    l: List[Float64], b: List[Float32], n: Int, nrhs: Int
) -> List[Float64]:
    """Forward then back substitution in float64. The residual check's other
    side, and the only place a `cho_solve` answer is compared against
    something that is not this repository's own float32 spelling."""
    var y = List[Float64]()
    for i in range(n * nrhs):
        y.append(Float64(b[i]))
    for j in range(nrhs):
        for i in range(n):
            var t = y[i * nrhs + j]
            for k in range(i):
                t = t - l[i * n + k] * y[k * nrhs + j]
            y[i * nrhs + j] = t / l[i * n + i]
        for ii in range(n):
            var i = n - 1 - ii
            var t = y[i * nrhs + j]
            for k in range(i + 1, n):
                t = t - l[k * n + i] * y[k * nrhs + j]
            y[i * nrhs + j] = t / l[i * n + i]
    return y^
