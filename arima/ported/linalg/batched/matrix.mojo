"""The batched small-matrix operations the Kalman filter's initialization
reaches: Kronecker product, LU inverse, GEMM, the direct Lyapunov solve.

PORT OF `cuml/cpp/src_prims/linalg/batched/matrix.cuh` at cuML 265b9da6
(v26.08.00), the part `_batched_kalman_filter` reaches for `r <= 5`:
`kronecker_product_kernel` (:506-526, via `b_kron` :814-842), `Matrix::inv`
(:380-390), `b_gemm` (:544-596), `_direct_lyapunov_helper` (:1852-1884).
The `Matrix<T>` class itself (an arena over cuBLAS handles), `b_gels`,
`b_lagged_mat`, `b_2dcopy`, the Hessenberg/Francis-QR/`trsyl` Schur path
of `b_lyapunov` (:1899-1948, `r > 5`) are not ported: `r > 5` is refused by
name (`arima_common.mojo::validate_order`). COPY, DO NOT IMPROVE -- except
where theirs is a CLOSED library call, which is the deviation below.

These are written as PER-SERIES device functions over raw pointers (one
thread per series, the shape their `thrust::for_each` lambdas already
have) and called from the initialization kernels in
`arima/ported/arima/batched_kalman.mojo`. The sizes are tiny (`r <= 5`, so
the Kronecker system is at most 25 x 25): the loops are serial and
ascending, and every multiply-add is `identical_mul_add`.

=============================================================================
DEVIATION 674: cuBLAS getrf/getri/gemm ARE CLOSED; THE SOLVE IS SPELLED HERE
=============================================================================
THEIRS. `Matrix::inv` is `cublasgetrfBatched` then `cublasgetriBatched`
(`matrix.cuh:383-389`): LU with partial pivoting, then the explicit inverse
(cuBLAS's getri inverts U and solves `inv(A) L = inv(U)`). `b_gemm` is
`cublasgemmStridedBatched`. Both are closed; their association order and
their pivot tie rule are not readable. The `info` array (singular pivot)
is computed and NEVER CHECKED by the caller (`batched_kalman.cu:1088-1089`,
`matrix.cuh:1876`), so a singular `I - T (x) T` produces an unusable `P0`
silently.
OURS (PORTING_RULES 0b-i: a closed call with no MAX equivalent for a
batched LU is written out). `lu_inverse` per series: column-major,
`getrf` semantics -- pivot = FIRST index of the largest `|a|` in the column
(cuBLAS's `isamax` rule, a selection over magnitudes so no signed zero can
be chosen over the other), row swap, `L` column = `a / pivot`, trailing
update `a = fma(-l, u, a)`; then the inverse column by column by
permuted forward (unit `L`) and backward (`U`) substitution, serial
ascending, `fma`. `b_gemm(alpha=1, beta=0)` is a serial ascending dot per
cell through `identical_mul_add`. A ZERO PIVOT sets `info[bid] = column + 1`
and the host RAISES BY NAME (`batched_kalman.mojo::_batched_kalman_filter`)
instead of filtering with a non-finite `P0` -- ADDENDUM 11 (no computed NaN
in a recorded stage) and `assume-our-code-is-broken`'s corollary that
theirs is right about design, not about dropping an error code.

MEASURED 2026-08-23, Apple M4, n_obs 24, batch 6, IDENTICAL
(`arima/mojo_only/arima_check.mojo::check_lyapunov_solves_the_equation`).
The residual `Ts P Ts' - P + RQRs` is evaluated in Float64 on the device's
own Float32 `P0` and scaled by `max|RQRs|`, so this is `P0` against the
EQUATION it is supposed to solve and not merely against an oracle that
shares its spelling. Worst relative residual, per order:

    arma11_k     7.3e-08     arima212     1.9e-07
    ar1          9.3e-08     ar2_unit     3.5e-06
    ma2          8.2e-08     sarima_full  4.3e-07
    arima111     7.3e-08     sarima_rd8   1.9e-07

`ar2_unit` is the loosest by an order of magnitude and that is expected: its
`phi_2` is pinned at the Jones clamp and then rewritten to -0.99 by the
unit-root guard, which is as near-singular as `I - T (x) T` gets here. The
gate's asserted bound is 5e-3, which is DERIVED and not observed; the
measured worst is 3.5e-6, roughly three orders of magnitude inside it. OWED:
tighten the bound to something the numbers actually justify.

The device `P0` is additionally bitwise equal to the host replay of this
exact spelling under IDENTICAL, 0 cells differing on all eight orders.
"""

from mojo_only.numerics import ftz, identical_mul_add


comptime LYAP_R_MAX = 5
comptime LYAP_R2_MAX = LYAP_R_MAX * LYAP_R_MAX


@always_inline
def lu_inverse(
    a: MutPointer[Float32, MutAnyOrigin],
    a_base: Int,
    inv: MutPointer[Float32, MutAnyOrigin],
    inv_base: Int,
    n: Int,
    piv: MutPointer[Int32, MutAnyOrigin],
    piv_base: Int,
) -> Int32:
    """`Matrix::inv(A, Ainv, P, info)` for one `n x n` column-major matrix at
    `a[a_base..]`, IN PLACE on `a` (getrf overwrites, as theirs), inverse to
    `inv[inv_base..]`, pivots to `piv[piv_base..piv_base+n)`. Returns
    `info`: 0, or `j + 1` for the first zero pivot (then `inv` is not
    written)."""
    # getrf: LU with partial pivoting, column by column
    for j in range(n):
        var best = j
        var best_mag = abs(ftz(a.unsafe_load(a_base + j + j * n)))
        for i in range(j + 1, n):
            var m = abs(ftz(a.unsafe_load(a_base + i + j * n)))
            if m > best_mag:
                best_mag = m
                best = i
        piv.unsafe_store(piv_base + j, Int32(best))
        if best != j:
            for c in range(n):
                var t0 = a.unsafe_load(a_base + j + c * n)
                a.unsafe_store(a_base + j + c * n, a.unsafe_load(a_base + best + c * n))
                a.unsafe_store(a_base + best + c * n, t0)
        var pivot = ftz(a.unsafe_load(a_base + j + j * n))
        if pivot == Float32(0.0):
            return Int32(j + 1)
        for i in range(j + 1, n):
            var l = ftz(ftz(a.unsafe_load(a_base + i + j * n)) / pivot)
            a.unsafe_store(a_base + i + j * n, l)
            for c in range(j + 1, n):
                var u = ftz(a.unsafe_load(a_base + j + c * n))
                var cur = ftz(a.unsafe_load(a_base + i + c * n))
                a.unsafe_store(a_base + i + c * n, ftz(identical_mul_add(-l, u, cur)))
    # getri: solve A X = I column by column through P, L, U
    for col in range(n):
        # b = P e_col: apply the row swaps to the unit vector, in order
        for i in range(n):
            inv.unsafe_store(inv_base + i + col * n, Float32(1.0) if i == col else Float32(0.0))
        for j in range(n):
            var pj = Int(piv.unsafe_load(piv_base + j))
            if pj != j:
                var t0 = inv.unsafe_load(inv_base + j + col * n)
                inv.unsafe_store(inv_base + j + col * n, inv.unsafe_load(inv_base + pj + col * n))
                inv.unsafe_store(inv_base + pj + col * n, t0)
        # forward: L y = b (unit diagonal)
        for i in range(n):
            var acc = ftz(inv.unsafe_load(inv_base + i + col * n))
            for k in range(i):
                var l = ftz(a.unsafe_load(a_base + i + k * n))
                var yk = ftz(inv.unsafe_load(inv_base + k + col * n))
                acc = ftz(identical_mul_add(-l, yk, acc))
            inv.unsafe_store(inv_base + i + col * n, acc)
        # backward: U x = y
        var i = n - 1
        while i >= 0:
            var acc = ftz(inv.unsafe_load(inv_base + i + col * n))
            for k in range(i + 1, n):
                var u = ftz(a.unsafe_load(a_base + i + k * n))
                var xk = ftz(inv.unsafe_load(inv_base + k + col * n))
                acc = ftz(identical_mul_add(-u, xk, acc))
            var d = ftz(a.unsafe_load(a_base + i + i * n))
            inv.unsafe_store(inv_base + i + col * n, ftz(acc / d))
            i -= 1
    return Int32(0)


@always_inline
def kron_minus_identity(
    t: MutPointer[Float32, MutAnyOrigin],
    t_base: Int,
    t_ld: Int,
    t_off: Int,
    r: Int,
    out_v: MutPointer[Float32, MutAnyOrigin],
    out_base: Int,
):
    """`b_kron(A, A, I_m_AxA, -1)` then `+1` on the diagonal
    (`matrix.cuh:1866-1875`): `out = I - A (x) A`, `r^2 x r^2` column-major,
    where `A = T[t_off.., t_off..]` read with leading dimension `t_ld`
    (their `b_2dcopy(Tb, Ts, n_diff, n_diff, r, r)` is folded into the read,
    a copy being no arithmetic). `kronecker_product_kernel:521`: `AkB[i_ab +
    j_ab*k_m] = (alpha * A[ia + ja*m]) * B[ib + jb*p]`, two roundings with
    `alpha = -1` exact."""
    var r2 = r * r
    for ia in range(r):
        for ja in range(r):
            var a_ia_ja = -ftz(t.unsafe_load(t_base + (ia + t_off) + (ja + t_off) * t_ld))
            for ib in range(r):
                for jb in range(r):
                    var i_ab = ia * r + ib
                    var j_ab = ja * r + jb
                    var b_val = ftz(t.unsafe_load(t_base + (ib + t_off) + (jb + t_off) * t_ld))
                    var v = ftz(a_ia_ja * b_val)
                    if i_ab == j_ab:
                        v = ftz(v + Float32(1.0))
                    out_v.unsafe_store(out_base + i_ab + j_ab * r2, v)


@always_inline
def matvec_serial(
    m: MutPointer[Float32, MutAnyOrigin],
    m_base: Int,
    n: Int,
    v: MutPointer[Float32, MutAnyOrigin],
    v_base: Int,
    out_v: MutPointer[Float32, MutAnyOrigin],
    out_base: Int,
):
    """`b_gemm(false, false, n, 1, n, 1, M, v, 0, out)`: `out = M v`,
    column-major `M`, serial ascending `fma` per row (DEVIATION 674)."""
    for i in range(n):
        var acc = Float32(0.0)
        for k in range(n):
            var a = ftz(m.unsafe_load(m_base + i + k * n))
            var x = ftz(v.unsafe_load(v_base + k))
            acc = ftz(identical_mul_add(a, x, acc))
        out_v.unsafe_store(out_base + i, acc)


# ---------------------------------------------------------------------------
# host replays over Lists (NOT ports: the oracles; spelled separately so a
# sabotage of the device helper cannot move both)
# ---------------------------------------------------------------------------


def lu_inverse_host(a_in: List[Float32], n: Int) raises -> List[Float32]:
    var a = a_in.copy()
    var piv = List[Int]()
    for j in range(n):
        var best = j
        var best_mag = abs(ftz(a[j + j * n]))
        for i in range(j + 1, n):
            var mg = abs(ftz(a[i + j * n]))
            if mg > best_mag:
                best_mag = mg
                best = i
        piv.append(best)
        if best != j:
            for c in range(n):
                var t0 = a[j + c * n]
                a[j + c * n] = a[best + c * n]
                a[best + c * n] = t0
        var pivot = ftz(a[j + j * n])
        if pivot == Float32(0.0):
            raise Error("lu_inverse_host: zero pivot at column " + String(j))
        for i in range(j + 1, n):
            var l = ftz(ftz(a[i + j * n]) / pivot)
            a[i + j * n] = l
            for c in range(j + 1, n):
                a[i + c * n] = ftz(identical_mul_add(-l, ftz(a[j + c * n]), ftz(a[i + c * n])))
    var inv = List[Float32]()
    for _ in range(n * n):
        inv.append(Float32(0.0))
    for col in range(n):
        for i in range(n):
            inv[i + col * n] = Float32(1.0) if i == col else Float32(0.0)
        for j in range(n):
            var pj = piv[j]
            if pj != j:
                var t0 = inv[j + col * n]
                inv[j + col * n] = inv[pj + col * n]
                inv[pj + col * n] = t0
        for i in range(n):
            var acc = ftz(inv[i + col * n])
            for k in range(i):
                acc = ftz(identical_mul_add(-ftz(a[i + k * n]), ftz(inv[k + col * n]), acc))
            inv[i + col * n] = acc
        var i = n - 1
        while i >= 0:
            var acc = ftz(inv[i + col * n])
            for k in range(i + 1, n):
                acc = ftz(identical_mul_add(-ftz(a[i + k * n]), ftz(inv[k + col * n]), acc))
            inv[i + col * n] = ftz(acc / ftz(a[i + i * n]))
            i -= 1
    return inv^


def kron_minus_identity_host(t: List[Float32], t_base: Int, t_ld: Int, t_off: Int, r: Int) -> List[Float32]:
    var r2 = r * r
    var out = List[Float32]()
    for _ in range(r2 * r2):
        out.append(Float32(0.0))
    for ia in range(r):
        for ja in range(r):
            var a_ia_ja = -ftz(t[t_base + (ia + t_off) + (ja + t_off) * t_ld])
            for ib in range(r):
                for jb in range(r):
                    var i_ab = ia * r + ib
                    var j_ab = ja * r + jb
                    var v = ftz(a_ia_ja * ftz(t[t_base + (ib + t_off) + (jb + t_off) * t_ld]))
                    if i_ab == j_ab:
                        v = ftz(v + Float32(1.0))
                    out[i_ab + j_ab * r2] = v
    return out^


def matvec_serial_host(m: List[Float32], n: Int, v: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        var acc = Float32(0.0)
        for k in range(n):
            acc = ftz(identical_mul_add(ftz(m[i + k * n]), ftz(v[k]), acc))
        out.append(acc)
    return out^
