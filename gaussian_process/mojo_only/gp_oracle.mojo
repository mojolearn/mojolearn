"""The host oracles: a float32 serial replay and a float64 reference.

NOT A PORT. scikit-learn checks nothing about a Gaussian process at the bit
level and need not: `sklearn/gaussian_process/tests/test_gpr.py` compares
against closed forms and against itself at `assert_almost_equal`, which is
the right test for a library that ships one backend. We ship three from one
source, so the device arm is gated BIT FOR BIT under IDENTICAL against the
float32 replay below, and the replay is gated against the float64 reference
to a tolerance the check prints. Both are written FIRST and gated FIRST.

TWO ORACLES, TWO JOBS
---------------------
`gp_oracle_kernel_matrix`   float32, SERIAL, through the same helpers the
                            device uses (`identical_div`, `identical_mul`,
                            `identical_mul_add`, `identical_exp`,
                            `identical_sqrt`, `ftz`), with every formula
                            spelled here a SECOND time rather than imported
                            from `kernels.mojo` -- so the gate compares two
                            spellings of one arithmetic and not a function
                            against itself.
`gp_reference_kernel_matrix_f64`
                            float64, scikit-learn's own expressions through
                            `std.math`. Its job is tolerance sanity,
                            measured per fixture instead of asserted.

**THE ORACLE MATERIALIZES `X / length_scale`, THE DEVICE FUSES IT.** That
is deliberate and it is the gate on DEVIATION 1753: sklearn writes
`cdist(X / length_scale, Y / length_scale, ...)`, which builds the scaled
copy first, and `gp_scaled_sqdist` divides inside the feature loop instead.
The two are bit-equal because each quotient is rounded to float32 before the
subtraction either way, and `check_kernels_vs_oracle` is what establishes
that rather than a paragraph asserting it. If the two ever disagree, the
fused form is wrong and the materialized one is right, because the
materialized one is what the semantics reference does.

THE THREE PLACES THIS ORACLE DOES NOT RE-SPELL THE ARITHMETIC, each because
re-spelling it would put a SECOND OPINION about someone else's certified
profile into this tree:

1. **The posterior mean** calls `gemm/mojo_only/gemm_oracle.mojo::
   gemm_oracle` at `OP_TN`, the normative answer of profile
   `mojolearn.identical.gemm.fp32.v1`, exactly as the device path calls
   that profile's kernel. `gemm_identical.mojo::contract_partition` is
   explicit that a second spelling of the leaf rule is a second thing that
   can be wrong, and records that the shape table already shipped one such
   re-spelling and got it wrong.
2. **The float64 factorization, log-determinant and solve** call
   `cholesky/mojo_only/cholesky_oracle.mojo::reference_potrf_lower_f64`,
   `reference_logdet_f64` and `reference_solve_f64`. That lane's gates
   certify them; this one inherits the certificate rather than re-earning
   it, and a textbook Cholesky written a second time here would be exactly
   the duplication this lane's README refuses.
3. **The ridge** calls that same file's `oracle_add_jitter`, which is
   `ftz(A_ii + jitter)` and nothing else.

WHY THE FLOAT64 REFERENCE FACTORS THE FLOAT32 KERNEL MATRIX, and not a
float64 one. There are two independent error sources in a float32 Gaussian
process -- the kernel matrix's own rounding, and the factorization's -- and
an end-to-end float64 reference would report their SUM as one number, from
which neither can be recovered. So they are separated: the kernel matrix is
compared against `gp_reference_kernel_matrix_f64` on its own, and the
factorization, solve and log-determinant are compared against a float64
replay OF THE SAME float32 matrix the device factored. Each check then names
one thing.
"""

from std.math import exp, log, sqrt

from cholesky.mojo_only.cholesky_oracle import (
    oracle_add_jitter,
    reference_logdet_f64,
    reference_potrf_lower_f64,
    reference_solve_f64,
)
from gaussian_process.mojo_only.kernels import (
    GP_K_CONST,
    GP_K_MATERN,
    GP_K_PROD,
    GP_K_RBF,
    GP_K_SUM,
    GP_K_WHITE,
    GPKernelSpec,
    gp_matern_nu_selector,
    gp_sqrt3,
    gp_sqrt5,
    gp_validate_kernel,
)
from gemm.mojo_only.gemm_oracle import OP_TN, gemm_oracle
from mojo_only.numerics import (
    ftz,
    identical_div,
    identical_exp,
    identical_mul,
    identical_mul_add,
    identical_sqrt,
)


@fieldwise_init
struct GPOracleVariance(Movable):
    """The replayed predictive variance, its root, and the clamp flags."""

    var variance: List[Float32]
    var std: List[Float32]
    var clamped: List[Int32]
    var n_clamped: Int


@fieldwise_init
struct GPReferenceFit(Movable):
    """The float64 reference's view of one fit."""

    var logdet: Float64
    var dual: List[Float64]
    var ydotalpha: Float64
    var lml: Float64
    var info: Int


# ===========================================================================
# THE FLOAT32 REPLAY
# ===========================================================================


def _scaled_copy(
    x: List[Float32], rows: Int, d: Int, ls: List[Float32], ls_len: Int
) -> List[Float32]:
    """`X / length_scale`, MATERIALIZED, scikit-learn `kernels.py:1568`.

    The device fuses this into the feature loop (DEVIATION 1753); this is
    the spelling the fused one is gated against.
    """
    var out = List[Float32]()
    for i in range(rows):
        for f in range(d):
            var li = f
            if ls_len == 1:
                li = 0
            out.append(
                ftz(identical_div(ftz(x[i * d + f]), ftz(ls[li])))
            )
    return out^


def _sqdist_from_scaled(
    xs: List[Float32], ys: List[Float32], i: Int, j: Int, d: Int
) -> Float32:
    """`cdist(..., "sqeuclidean")` over ALREADY SCALED coordinates, `f`
    ascending. A second spelling of `l2_unexp_core`'s loop, written out
    rather than imported so the gate compares two spellings."""
    var acc = Float32(0.0)
    for f in range(d):
        var diff = ftz(ftz(xs[i * d + f]) - ftz(ys[j * d + f]))
        acc = ftz(identical_mul_add(diff, diff, acc))
    return ftz(acc)


def gp_oracle_kernel_matrix(
    x: List[Float32],
    m: Int,
    y: List[Float32],
    n: Int,
    d: Int,
    spec: GPKernelSpec,
    is_self: Bool,
) raises -> List[Float32]:
    """`k(x[m x d], y[n x d])`, float32, on the host, row-major.

    The postfix list is walked in the SAME order `gp_kernel_matrix` walks
    it, with a host stack of matrices standing in for the device's stack of
    buffers, so a divergence between the two is a divergence of ARITHMETIC
    and never of evaluation order.
    """
    gp_validate_kernel(spec, d)
    var cells = m * n
    var stack = List[List[Float32]]()
    var sqrt3 = gp_sqrt3()
    var sqrt5 = gp_sqrt5()

    for t in range(len(spec.kinds)):
        var kind = Int(spec.kinds[t])
        if kind == GP_K_SUM or kind == GP_K_PROD:
            var rhs = stack.pop()
            var lhs = stack.pop()
            var merged = List[Float32]()
            for c in range(cells):
                var av = ftz(lhs[c])
                var bv = ftz(rhs[c])
                if kind == GP_K_PROD:
                    merged.append(ftz(identical_mul(av, bv)))
                else:
                    merged.append(ftz(av + bv))
            stack.append(merged^)
            continue

        var slot = List[Float32]()
        if kind == GP_K_CONST:
            for _c in range(cells):
                slot.append(ftz(spec.params[t]))
        elif kind == GP_K_WHITE:
            # `kernels.py:1406-1419`: the STRUCTURAL test, not a coordinate
            # one. DEVIATION 1762.
            for i in range(m):
                for j in range(n):
                    if is_self and i == j:
                        slot.append(ftz(spec.params[t]))
                    else:
                        slot.append(Float32(0.0))
        else:
            var ls_len = Int(spec.ls_len[t])
            var ls = List[Float32]()
            for q in range(ls_len):
                ls.append(spec.length_scales[Int(spec.ls_off[t]) + q])
            var xs = _scaled_copy(x, m, d, ls, ls_len)
            var ys = _scaled_copy(y, n, d, ls, ls_len)
            if kind == GP_K_RBF:
                # `kernels.py:1569-1570`: K = exp(-0.5 * sqeuclidean)
                for i in range(m):
                    for j in range(n):
                        var d2 = _sqdist_from_scaled(xs, ys, i, j, d)
                        var e = ftz(identical_mul(Float32(-0.5), d2))
                        slot.append(ftz(identical_exp(e)))
            else:
                # `kernels.py:1722-1730`: the three closed forms, in their
                # order, over the EUCLIDEAN distance.
                var nu_sel = gp_matern_nu_selector(spec.params[t])
                for i in range(m):
                    for j in range(n):
                        var d2 = _sqdist_from_scaled(xs, ys, i, j, d)
                        var dist = ftz(identical_sqrt(d2))
                        if nu_sel == 0:
                            slot.append(ftz(identical_exp(-dist)))
                        elif nu_sel == 1:
                            var s = ftz(identical_mul(dist, sqrt3))
                            var pre = ftz(Float32(1.0) + s)
                            slot.append(
                                ftz(
                                    identical_mul(
                                        pre, ftz(identical_exp(-s))
                                    )
                                )
                            )
                        else:
                            var s5 = ftz(identical_mul(dist, sqrt5))
                            var ss = ftz(identical_mul(s5, s5))
                            var third = ftz(
                                identical_div(ss, Float32(3.0))
                            )
                            var pre5 = ftz(
                                ftz(Float32(1.0) + s5) + third
                            )
                            slot.append(
                                ftz(
                                    identical_mul(
                                        pre5, ftz(identical_exp(-s5))
                                    )
                                )
                            )
        stack.append(slot^)

    return stack[0].copy()


def gp_oracle_mean(
    kcross: List[Float32], dual: List[Float32], n_train: Int, n_star: Int
) -> List[Float32]:
    """`y_mean = K_trans @ alpha_`, scikit-learn `_gpr.py:447`.

    **CALLS `gemm_oracle` AT `OP_TN` AND SPELLS NOTHING ITSELF.** `kcross`
    is `K_trans^T`, stored `n_train x n_star` (DEVIATION 1758), so with
    `A = kcross` (`k x m`), `B = dual` (`k x n`), `m = n_star`, `n = 1` and
    `k = n_train`, `OP_TN`'s contract `C[m x n] = A[k x m]^T . B[k x n]`
    (`gemm_oracle.mojo:198`) is exactly the product wanted.
    """
    return gemm_oracle(kcross, dual, OP_TN, n_star, 1, n_train)


def gp_oracle_ydotalpha(
    y: List[Float32], dual: List[Float32], n: Int
) -> Float32:
    """`einsum("ik,ik->k", y_train, alpha)`, `_gpr.py:613`, ascending.

    A second spelling of `gaussian_process/estimator.mojo::_y_dot_alpha`.
    Both are host folds, so this comparison establishes that the two
    spellings agree and not that a device matches a host; the value of it
    is that the estimator's loop could have been written descending by
    accident, and `GP_SAB_YALPHA_DESCENDING` is the arm that shows this
    comparison would see it.
    """
    var acc = Float32(0.0)
    for i in range(n):
        acc = ftz(identical_mul_add(ftz(y[i]), ftz(dual[i]), acc))
    return ftz(acc)


def gp_oracle_variance(
    v: List[Float32], n_train: Int, n_star: Int, kss: Float32
) -> GPOracleVariance:
    """`gp_variance_kernel`, replayed: `kss - sum_i v[i][t]^2`, `i`
    ascending, clamped at zero, with the clamp recorded as a per-test-point
    FLAG set from a BIT COMPARISON. `_gpr.py:480-491` and `:500`.

    `v` is `n_train x n_star` row-major, the orientation `trsm_lower`
    writes.
    """
    var variance = List[Float32]()
    var std = List[Float32]()
    var clamped = List[Int32]()
    var n_clamped = 0
    for t in range(n_star):
        var acc = Float32(0.0)
        for i in range(n_train):
            var vv = ftz(v[i * n_star + t])
            acc = ftz(identical_mul_add(vv, vv, acc))
        var raw = ftz(ftz(kss) - acc)
        var outv = raw
        if not (raw > Float32(0.0)):
            outv = Float32(0.0)
        var moved = _bits(outv) != _bits(raw)
        variance.append(outv)
        std.append(ftz(identical_sqrt(outv)))
        if moved:
            clamped.append(Int32(1))
            n_clamped += 1
        else:
            clamped.append(Int32(0))
    return GPOracleVariance(variance^, std^, clamped^, n_clamped)


def _bits(v: Float32) -> UInt32:
    from std.memory import bitcast

    return bitcast[DType.uint32](v)


def gp_oracle_forward_solve(
    l: List[Float32], b: List[Float32], n: Int, nrhs: Int
) -> List[Float32]:
    """`trsm_lower`, replayed: `L V = B`, forward substitution, `k`
    ascending. A second spelling of `cholesky/mojo_only/trsm.mojo::
    trsm_lower_kernel`, needed because `predict`'s `v` stage has to be
    gated against something and the Cholesky lane's own oracle exposes
    only the two-sided `cho_solve`.

    Written here rather than reached for in that lane's oracle because
    `oracle_trsm_lower` there is not exported for a single-sided use at an
    arbitrary right-hand side count; `gaussian_process/README.md` names the
    one-line merge under WHAT IS OWED.
    """
    var out = b.copy()
    for j in range(nrhs):
        for i in range(n):
            var t = ftz(out[i * nrhs + j])
            for k in range(i):
                var lik = ftz(l[i * n + k])
                var bk = ftz(out[k * nrhs + j])
                t = ftz(identical_mul_add(-lik, bk, t))
            var lii = ftz(l[i * n + i])
            out[i * nrhs + j] = ftz(identical_div(t, lii))
    return out^


# ===========================================================================
# THE FLOAT64 REFERENCE
# ===========================================================================


def _f64_scaled(
    x: List[Float32], rows: Int, d: Int, ls: List[Float32], ls_len: Int
) -> List[Float64]:
    var out = List[Float64]()
    for i in range(rows):
        for f in range(d):
            var li = f
            if ls_len == 1:
                li = 0
            out.append(Float64(x[i * d + f]) / Float64(ls[li]))
    return out^


def gp_reference_kernel_matrix_f64(
    x: List[Float32],
    m: Int,
    y: List[Float32],
    n: Int,
    d: Int,
    spec: GPKernelSpec,
    is_self: Bool,
) raises -> List[Float64]:
    """scikit-learn's expressions in float64 through `std.math`, the only
    float64 in this lane and the only `std.math` transcendental in it.

    `math.sqrt(3)` and `math.sqrt(5)` are computed here in float64 exactly
    as sklearn computes them, rather than widened from the pinned float32
    constants: this is the reference, so it has to be independent of the
    thing it is checking. `check_gp_constants` is what ties the two
    together, by asserting that the pinned float32 bits are the correctly
    rounded float32 of these float64 values.
    """
    gp_validate_kernel(spec, d)
    var cells = m * n
    var stack = List[List[Float64]]()
    var s3 = sqrt(Float64(3.0))
    var s5 = sqrt(Float64(5.0))

    for t in range(len(spec.kinds)):
        var kind = Int(spec.kinds[t])
        if kind == GP_K_SUM or kind == GP_K_PROD:
            var rhs = stack.pop()
            var lhs = stack.pop()
            var merged = List[Float64]()
            for c in range(cells):
                if kind == GP_K_PROD:
                    merged.append(lhs[c] * rhs[c])
                else:
                    merged.append(lhs[c] + rhs[c])
            stack.append(merged^)
            continue

        var slot = List[Float64]()
        if kind == GP_K_CONST:
            for _c in range(cells):
                slot.append(Float64(spec.params[t]))
        elif kind == GP_K_WHITE:
            for i in range(m):
                for j in range(n):
                    if is_self and i == j:
                        slot.append(Float64(spec.params[t]))
                    else:
                        slot.append(Float64(0.0))
        else:
            var ls_len = Int(spec.ls_len[t])
            var ls = List[Float32]()
            for q in range(ls_len):
                ls.append(spec.length_scales[Int(spec.ls_off[t]) + q])
            var xs = _f64_scaled(x, m, d, ls, ls_len)
            var ys = _f64_scaled(y, n, d, ls, ls_len)
            var nu_sel = 0
            if kind == GP_K_MATERN:
                nu_sel = gp_matern_nu_selector(spec.params[t])
            for i in range(m):
                for j in range(n):
                    var acc = Float64(0.0)
                    for f in range(d):
                        var diff = xs[i * d + f] - ys[j * d + f]
                        acc = acc + diff * diff
                    if kind == GP_K_RBF:
                        slot.append(exp(Float64(-0.5) * acc))
                    else:
                        var dist = sqrt(acc)
                        if nu_sel == 0:
                            slot.append(exp(-dist))
                        elif nu_sel == 1:
                            var s = dist * s3
                            slot.append((Float64(1.0) + s) * exp(-s))
                        else:
                            var s = dist * s5
                            slot.append(
                                (
                                    Float64(1.0)
                                    + s
                                    + s * s / Float64(3.0)
                                )
                                * exp(-s)
                            )
        stack.append(slot^)

    return stack[0].copy()


def gp_reference_fit_f64(
    k: List[Float32], y: List[Float32], n: Int, alpha: Float32
) raises -> GPReferenceFit:
    """The float64 reference for `fit`, over the float32 kernel matrix the
    device actually built. This file's header says why it is not a fully
    float64 chain.

    Every step is a CALL into the Cholesky lane's float64 reference:

        oracle_add_jitter          K[diag] += alpha       _gpr.py:350
        reference_potrf_lower_f64  L = cholesky(K)        _gpr.py:352
        reference_solve_f64        alpha_ = cho_solve     _gpr.py:363
        reference_logdet_f64       2 sum log diag L       (twice
                                   _gpr.py:614's term)

    and the only arithmetic written here is the three-term marginal
    likelihood, in scikit-learn's own order (`_gpr.py:613-617`).
    """
    var ridged = oracle_add_jitter(k, n, alpha)
    var fac = reference_potrf_lower_f64(ridged, n)
    if fac.info != 0:
        var empty = List[Float64]()
        return GPReferenceFit(
            Float64(0.0), empty^, Float64(0.0), Float64(0.0), fac.info
        )
    var logdet = reference_logdet_f64(fac.l, n)
    var dual = reference_solve_f64(fac.l, y, n, 1)
    var ydot = Float64(0.0)
    for i in range(n):
        ydot = ydot + Float64(y[i]) * dual[i]
    var lml = Float64(-0.5) * ydot
    lml = lml - Float64(0.5) * logdet
    lml = lml - Float64(n) / Float64(2.0) * log(
        Float64(2.0) * Float64(3.141592653589793)
    )
    return GPReferenceFit(logdet, dual^, ydot, lml, 0)


def gp_reference_log_2pi_f64() -> Float64:
    """`np.log(2 * np.pi)` in float64. `check_gp_constants` asserts that
    `GP_LOG_2PI_BITS` is the correctly rounded float32 of this."""
    return log(Float64(2.0) * Float64(3.141592653589793))


def gp_reference_lml_terms_f64(
    ydotalpha: Float64, logdet: Float64, n: Int
) -> Float64:
    """`_gpr.py:613-617` in float64, for a caller that already has the two
    scalars. Used by the hand-worked check, whose two scalars are stated in
    `gp_fixture.mojo::gp_handworked_notes` rather than computed."""
    var lml = Float64(-0.5) * ydotalpha
    lml = lml - Float64(0.5) * logdet
    lml = lml - Float64(n) / Float64(2.0) * gp_reference_log_2pi_f64()
    return lml


def gp_reference_sqrt3_f64() -> Float64:
    return sqrt(Float64(3.0))


def gp_reference_sqrt5_f64() -> Float64:
    return sqrt(Float64(5.0))
