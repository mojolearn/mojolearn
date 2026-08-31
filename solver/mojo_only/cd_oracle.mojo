# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The HOST ORACLE for `cdFit`: serial, one step at a time, through the same
helpers the device kernels use, plus a Float64 reference and the fixtures.

NOT A PORT. cuML ships one backend and needs no oracle. This file is what
`solver/ported/solver/cd.mojo` is gated against BIT FOR BIT under IDENTICAL
(`check_cd_device_equals_oracle`), stage by stage: the column norms, the
means, and after EVERY epoch the coefficient vector, the residual and the
`ConvState` triple. Under FAST the same comparison is a REPORT.

THE ARITHMETIC IS THE DEVICE'S, NOT A SECOND OPINION ABOUT IT:

- every row-length reduction is `gemm_oracle_cell` at the contract's leaf
  size (`solver/mojo_only/profile_dot.mojo::profile_dot_host`), IMPORTED
  from the gemm lane, so the fold tree is the device's by construction;
- the axpys are `ftz(identical_mul_add(ftz(a), ftz(x), ftz(y)))`, character
  for character `axpy_device_alpha_kernel`;
- the update is `cdUpdateCoefKernel` transcribed once more (the device
  file has the citations), with the flushed quotient, diff and |r|;
- the stopping test is the host's own Float32 `coefMax < tol ||
  diffMax / coefMax < tol`, as in both.

`cd_oracle_fit(..., profile=True)` is the normative oracle. `profile=False`
replaces every reduction with the WHOLE-ROW serial ascending chain
(`serial_dot_host`, the one-leaf case): at `n_rows <= 128` the two are the
same bits (gemm contract section 6), and at `n_rows > 128` they differ in
the last bits -- the check REPORTS how many cells of `coef` and `residual`
move between the two, which is the witness that the balanced tree is
reached and that a serial spelling is a different answer.

`cd_reference_f64` is the Float64 reference: the same algorithm, plain
arithmetic, serial dots, no flush. It exists for TOLERANCE sanity only
(the bitwise gates are against the Float32 oracle); `check_cd_recovers_the
_planted_support` holds the device within a tolerance of it and requires
the recovered support to be the planted one.

SABOTAGE SWITCH (a no-op unless named): `MOJOLEARN_CD_SABOTAGE_NO_FTZ_RESID`
drops the `ftz` at the oracle's RESIDUAL seam -- the residual as STORED by
each axpy and as RE-READ by the next one (the dot's loads keep the gemm
contract's 5b flush, so what moves is the residual alone). Found while
building it: dropping only the store-side flush is invisible at n_cols > 1,
because the next coordinate's axpy flushes the denormal residual on load
and the sweep-end snapshot is all zeros either way; the seam is the pair.
On Apple the device flush is
HARDWARE (`mojo_only/numerics.mojo::ftz` is bit-inert there), so a device-
side drop could not be seen on this column; the host honors denormals, so
dropping the oracle's flush is how the reach of that seam is demonstrated
on this machine -- `check_cd_device_equals_oracle` on the denormal fixture
MUST FAIL with it, at the first epoch's `resid` stage. On a denormal-
honoring backend the same define on the DEVICE side would be the witness.

FIXTURES, all HASHED (no uniform or constant data; IDENTITY_PATHS' "uniform
test data hides permutation"):

    fixture_planted_sparse(n, d, seed)  X hashed in [-0.5, 0.5); w* with
                                        ceil(d/4) nonzeros (hashed magnitude
                                        in [1, 3), hashed sign); y = X w* +
                                        0.1 * hashed noise. Lasso at alpha =
                                        0.01 must recover exactly w*'s support.
    fixture_denormal_residual(n, d)     X hashed; y = X[:, 0] * 1e-37, so the
                                        first coordinate's update drives the
                                        residual to the denormal range (and a
                                        quarter of y is denormal on input).
                                        alpha = 0.
    fixture_large(n, d)                 planted_sparse at n = 20,000: P = 157
                                        leaves, so the SPLITK leaf kernel
                                        spans two blocks and the gemm lane's
                                        LEAF_ROTATE sabotage can bite.
    fixture_signed_zero(n, d, negate)   X, y hashed; with alpha 1e36 and
                                        l1_ratio 0 every quotient flushes to
                                        a zero SIGNED by its dot; `negate`
                                        flips every sign (row 39, both orders)
    fixture_nonfinite_labels(n, d, k)   y with one +inf (k = 0: a COMPUTED
                                        NaN in the residual) or one payload
                                        NaN 0x7FC0BEEF (k = 1: a PROPAGATED
                                        NaN); DEVIATION 612's gate
"""

from std.memory import bitcast
from std.sys.compile import is_defined

from gemm.mojo_only.gemm_oracle import contract_leaf_size
from mojo_only.numerics import ftz, identical_mul_add
from solver.mojo_only.profile_dot import (
    column_as_list,
    profile_dot_host,
    serial_dot_host,
)

comptime SAB_NO_FTZ_RESID = is_defined["MOJOLEARN_CD_SABOTAGE_NO_FTZ_RESID"]()

#: `cd.cuh:62`, `math_t(1e-5)` -- the same constant `cd.mojo` carries.
comptime ORACLE_SQUARED_GUARD = Float32(1.0e-5)


@always_inline
def _ftz_resid(v: Float32) -> Float32:
    comptime if SAB_NO_FTZ_RESID:
        return v
    return ftz(v)


def oracle_sabotage_name() -> String:
    comptime if SAB_NO_FTZ_RESID:
        return String("NO_FTZ_RESID")
    return String("none")


# ===========================================================================
# HASHED FIXTURES
# ===========================================================================


def _u01(row: Int, k: Int, salt: Int) -> Float64:
    """A hashed uniform in [0, 1): splitmix-style mixing of (row, k, salt)."""
    var z = (
        UInt64(row) * 0x9E3779B97F4A7C15
        + UInt64(k + 1) * 0xBF58476D1CE4E5B9
        + UInt64(salt + 1) * 0x94D049BB133111EB
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return Float64(z >> 11) / Float64(1 << 53)


def planted_w(d: Int, seed: Int) -> List[Float32]:
    """`w*`: nonzero at `k % 4 == 1`, hashed magnitude in [1, 3), hashed sign."""
    var w = List[Float32]()
    for k in range(d):
        if k % 4 == 1:
            var mag = 1.0 + 2.0 * _u01(k, 7, seed)
            var sgn = 1.0 if _u01(k, 11, seed) < 0.5 else -1.0
            w.append(Float32(sgn * mag))
        else:
            w.append(Float32(0.0))
    return w^


def fixture_planted_sparse(
    n: Int, d: Int, seed: Int
) -> Tuple[List[Float32], List[Float32], List[Float32]]:
    """`(X column-major n x d, y, w*)`. `y` is assembled in Float64 and
    rounded once, so the fixture itself performs no Float32 contraction
    (glm/README.md's note on a host `target += v * w` chain)."""
    var w = planted_w(d, seed)
    var x = List[Float32]()
    x.reserve(n * d)
    for k in range(d):
        for i in range(n):
            x.append(Float32(_u01(i, k, seed) - 0.5))
    var y = List[Float32]()
    y.reserve(n)
    for i in range(n):
        var t = 0.0
        for k in range(d):
            t += Float64(x[k * n + i]) * Float64(w[k])
        t += 0.1 * (_u01(i, 991, seed + 17) - 0.5)
        y.append(Float32(t))
    return (x^, y^, w^)


def fixture_denormal_residual(
    n: Int, d: Int
) -> Tuple[List[Float32], List[Float32]]:
    """`y = X[:, 0] * 1e-37`: after coordinate 0's first update the residual
    is rounding noise at ~1e-44, i.e. denormal or zero, and the seam is
    reached. About a quarter of `y` is itself denormal (|x| < 0.1175)."""
    var x = List[Float32]()
    x.reserve(n * d)
    for k in range(d):
        for i in range(n):
            x.append(Float32(_u01(i, k, 4242) - 0.5))
    var y = List[Float32]()
    y.reserve(n)
    for i in range(n):
        y.append(Float32(Float64(x[i]) * 1.0e-37))
    return (x^, y^)


def fixture_signed_zero(
    n: Int, d: Int, negate: Bool
) -> Tuple[List[Float32], List[Float32]]:
    """IDENTITY_PATHS row 39's fixture: `X` hashed in [-0.5, 0.5), `y`
    hashed in [-0.5, 0.5) and NEGATED when `negate`. Meant to be fit with
    `fit_intercept = False, alpha = 1e36, l1_ratio = 0, tol = 0`: `squared`
    is `colnorm + 1e36 * n` (~2.56e38 at n = 256) and every coordinate's
    quotient `dot / squared` lies below the normal floor, so `r` flushes to
    a zero CARRYING THE SIGN OF THE DOT -- a -0.0 coefficient wherever
    `dot(X[:, j], y) < 0`, a +0.0 wherever it is positive, and the negated
    fixture flips every one (negation is exact, `fma(x, -y, -acc)` is
    `-fma(x, y, acc)`). Both zeros reach `cdUpdateCoefKernel`'s `coefMax` /
    `diffMax` folds, in both orders across the two fixtures, and the
    coefficient array on the card carries them."""
    var x = List[Float32]()
    x.reserve(n * d)
    for k in range(d):
        for i in range(n):
            x.append(Float32(_u01(i, k, 3939) - 0.5))
    var y = List[Float32]()
    y.reserve(n)
    for i in range(n):
        var v = _u01(i, 0, 3940) - 0.5
        if negate:
            v = -v
        y.append(Float32(v))
    return (x^, y^)


def fixture_nonfinite_labels(
    n: Int, d: Int, kind: Int
) -> Tuple[List[Float32], List[Float32]]:
    """IDENTITY_PATHS row 39 FACT 2's fixture (DEVIATION 612): `X` hashed;
    `y` hashed with ONE non-finite label. `kind = 0`: `y[5] = +inf`, so the
    first coordinate's dot is `+-inf`, its coefficient `+-inf`, and the
    second axpy writes `(-inf) * x + inf` -- a COMPUTED NaN with the
    vendor's payload -- into `residual[5]`; the next epoch's first axpy
    makes every residual `inf - inf`. `kind = 1`: `y[9]` is the quiet NaN
    `0x7FC0BEEF`, a payload-carrying NaN that PROPAGATES (`fma(-0, x, NaN)`)
    into the residual and the dot. Fit with `fit_intercept = False`."""
    var x = List[Float32]()
    x.reserve(n * d)
    for k in range(d):
        for i in range(n):
            x.append(Float32(_u01(i, k, 6121) - 0.5))
    var y = List[Float32]()
    y.reserve(n)
    for i in range(n):
        y.append(Float32(_u01(i, 0, 6122) - 0.5))
    if kind == 0:
        y[5] = bitcast[DType.float32](UInt32(0x7F800000))  # +inf
    else:
        y[9] = bitcast[DType.float32](UInt32(0x7FC0BEEF))  # quiet NaN, payload
    return (x^, y^)


# ===========================================================================
# THE ORACLE (Float32, the device's arithmetic, serial)
# ===========================================================================


struct CdOracleResult(Movable):
    var x_input: List[Float32]
    var y_input: List[Float32]
    var coef: List[Float32]
    var residual: List[Float32]
    var x_after: List[Float32]
    var y_after: List[Float32]
    var intercept: Float32
    var n_iter: Int
    var l1_alpha: Float32
    var l2_alpha: Float32
    var colnorm: List[Float32]
    var squared: List[Float32]
    var mu_input: List[Float32]
    var mu_labels: Float32
    var coef_sweeps: List[List[Float32]]
    var resid_sweeps: List[List[Float32]]
    var conv_sweeps: List[List[Float32]]

    def __init__(out self):
        self.x_input = List[Float32]()
        self.y_input = List[Float32]()
        self.coef = List[Float32]()
        self.residual = List[Float32]()
        self.x_after = List[Float32]()
        self.y_after = List[Float32]()
        self.intercept = Float32(0.0)
        self.n_iter = 0
        self.l1_alpha = Float32(0.0)
        self.l2_alpha = Float32(0.0)
        self.colnorm = List[Float32]()
        self.squared = List[Float32]()
        self.mu_input = List[Float32]()
        self.mu_labels = Float32(0.0)
        self.coef_sweeps = List[List[Float32]]()
        self.resid_sweeps = List[List[Float32]]()
        self.conv_sweeps = List[List[Float32]]()


def _dot(a: List[Float32], b: List[Float32], k: Int, profile: Bool) -> Float32:
    if profile:
        return profile_dot_host(a, b, k)
    return serial_dot_host(a, b, k)


def cd_oracle_fit(
    x_in: List[Float32],
    y_in: List[Float32],
    n: Int,
    d: Int,
    fit_intercept: Bool,
    epochs: Int,
    alpha: Float32,
    l1_ratio: Float32,
    tol: Float32,
    profile: Bool = True,
) -> CdOracleResult:
    """`cdFit` on the host, stage for stage. `coef` starts at zero (cuML's
    Python passes `cp.zeros`). `profile` selects the normative fold (True)
    or the whole-row serial chain (False, diagnostic)."""
    var out = CdOracleResult()
    out.x_input = x_in.copy()
    out.y_input = y_in.copy()
    var x = x_in.copy()
    var y = y_in.copy()
    var ones = List[Float32]()
    ones.reserve(n)
    for _ in range(n):
        ones.append(Float32(1.0))
    var ratio = Float32(1.0) / Float32(n)

    # preProcessData
    var mu_input = List[Float32]()
    var mu_labels = Float32(0.0)
    if fit_intercept:
        for j in range(d):
            var col = column_as_list(x, j * n, n)
            var s = _dot(col, ones, n, profile)
            mu_input.append(ftz(s * ratio))
        for j in range(d):
            var m = ftz(mu_input[j])
            for i in range(n):
                x[j * n + i] = ftz(ftz(x[j * n + i]) + Float32(-1.0) * m)
        var sy = _dot(y, ones, n, profile)
        mu_labels = ftz(sy * ratio)
        var ml = ftz(mu_labels)
        for i in range(n):
            y[i] = ftz(ftz(y[i]) + Float32(-1.0) * ml)
    out.mu_input = mu_input.copy()
    out.mu_labels = mu_labels

    # cd.cuh:168-169
    var one_minus = Float32(1.0) - l1_ratio
    var l2_a = one_minus * alpha
    var l2_alpha = l2_a * Float32(n)
    var l1_a = l1_ratio * alpha
    var l1_alpha = l1_a * Float32(n)
    out.l1_alpha = l1_alpha
    out.l2_alpha = l2_alpha

    # colNorm, + l2_alpha
    var squared = List[Float32]()
    for j in range(d):
        var col = column_as_list(x, j * n, n)
        var nn = _dot(col, col, n, profile)
        out.colnorm.append(nn)
        squared.append(ftz(nn + l2_alpha))
    out.squared = squared.copy()

    var residual = y.copy()
    var coef = List[Float32]()
    for _ in range(d):
        coef.append(Float32(0.0))

    var n_iter = 0
    while n_iter < epochs:
        var conv_coef = Float32(0.0)
        var coef_max = Float32(0.0)
        var diff_max = Float32(0.0)
        for ci in range(d):
            # remember current coef
            conv_coef = coef[ci]
            # residual += coef[ci] * X[:, ci]
            var a1 = ftz(coef[ci])
            for i in range(n):
                var xi = ftz(x[ci * n + i])
                var ri = _ftz_resid(residual[i])
                residual[i] = _ftz_resid(identical_mul_add(a1, xi, ri))
            # coef[ci] = dot(X[:, ci], residual)
            var col = column_as_list(x, ci * n, n)
            coef[ci] = _dot(col, residual, n, profile)
            # cdUpdateCoefKernel
            var c = ftz(coef[ci])
            var r: Float32
            if c > l1_alpha:
                r = c - l1_alpha
            elif c < -l1_alpha:
                r = c + l1_alpha
            else:
                r = Float32(0.0)
            var sq = ftz(squared[ci])
            if sq > ORACLE_SQUARED_GUARD:
                r = r / sq
            else:
                r = Float32(0.0)
            r = ftz(r)
            # IDENTITY_PATHS row 39: the same spelling as the device's
            # `cd_update_coef_kernel` -- `abs()` candidates (never -0.0),
            # +0.0 seeds, STRICT `<` (the earlier value survives a tie by
            # position; a NaN never enters). `r` itself may be -0.0 and is
            # kept as such, like the device's.
            var diff = ftz(abs(ftz(conv_coef) - r))
            if diff_max < diff:
                diff_max = diff
            var absv = abs(r)
            if coef_max < absv:
                coef_max = absv
            conv_coef = -r
            coef[ci] = r
            # residual += conv.coef * X[:, ci]
            var a2 = ftz(conv_coef)
            for i in range(n):
                var xi = ftz(x[ci * n + i])
                var ri = _ftz_resid(residual[i])
                residual[i] = _ftz_resid(identical_mul_add(a2, xi, ri))
        n_iter += 1
        out.coef_sweeps.append(coef.copy())
        out.resid_sweeps.append(residual.copy())
        var cv = List[Float32]()
        cv.append(conv_coef)
        cv.append(coef_max)
        cv.append(diff_max)
        out.conv_sweeps.append(cv^)
        if coef_max < tol or (diff_max / coef_max) < tol:
            break

    var intercept = Float32(0.0)
    if fit_intercept:
        var dot_mu = _dot(mu_input, coef, d, profile)
        intercept = ftz(ftz(mu_labels) - ftz(dot_mu))
        for j in range(d):
            var m = ftz(mu_input[j])
            for i in range(n):
                x[j * n + i] = ftz(ftz(x[j * n + i]) + Float32(1.0) * m)
        var ml = ftz(mu_labels)
        for i in range(n):
            y[i] = ftz(ftz(y[i]) + Float32(1.0) * ml)

    out.coef = coef^
    out.residual = residual^
    out.x_after = x^
    out.y_after = y^
    out.intercept = intercept
    out.n_iter = n_iter
    return out^


# ===========================================================================
# THE FLOAT64 REFERENCE (tolerance sanity only)
# ===========================================================================


def cd_reference_f64(
    x_in: List[Float32],
    y_in: List[Float32],
    n: Int,
    d: Int,
    fit_intercept: Bool,
    epochs: Int,
    alpha: Float64,
    l1_ratio: Float64,
    tol: Float64,
) -> Tuple[List[Float64], Float64, Int]:
    """The same algorithm in Float64 with serial dots and no flush. Returns
    `(coef, intercept, n_iter)`."""
    var x = List[Float64]()
    x.reserve(n * d)
    for v in x_in:
        x.append(Float64(v))
    var y = List[Float64]()
    y.reserve(n)
    for v in y_in:
        y.append(Float64(v))
    var mu = List[Float64]()
    var mu_y = 0.0
    if fit_intercept:
        for j in range(d):
            var s = 0.0
            for i in range(n):
                s += x[j * n + i]
            var m = s / Float64(n)
            mu.append(m)
            for i in range(n):
                x[j * n + i] -= m
        var sy = 0.0
        for i in range(n):
            sy += y[i]
        mu_y = sy / Float64(n)
        for i in range(n):
            y[i] -= mu_y
    var l2_alpha = (1.0 - l1_ratio) * alpha * Float64(n)
    var l1_alpha = l1_ratio * alpha * Float64(n)
    var squared = List[Float64]()
    for j in range(d):
        var s = 0.0
        for i in range(n):
            s += x[j * n + i] * x[j * n + i]
        squared.append(s + l2_alpha)
    var residual = y.copy()
    var coef = List[Float64]()
    for _ in range(d):
        coef.append(0.0)
    var n_iter = 0
    while n_iter < epochs:
        var coef_max = 0.0
        var diff_max = 0.0
        for ci in range(d):
            var old = coef[ci]
            for i in range(n):
                residual[i] += old * x[ci * n + i]
            var dot = 0.0
            for i in range(n):
                dot += x[ci * n + i] * residual[i]
            var r: Float64
            if dot > l1_alpha:
                r = dot - l1_alpha
            elif dot < -l1_alpha:
                r = dot + l1_alpha
            else:
                r = 0.0
            if squared[ci] > 1.0e-5:
                r = r / squared[ci]
            else:
                r = 0.0
            var diff = abs(old - r)
            if diff_max < diff:
                diff_max = diff
            if coef_max < abs(r):
                coef_max = abs(r)
            coef[ci] = r
            for i in range(n):
                residual[i] -= r * x[ci * n + i]
        n_iter += 1
        if coef_max < tol or (diff_max / coef_max) < tol:
            break
    var intercept = 0.0
    if fit_intercept:
        var s = 0.0
        for j in range(d):
            s += mu[j] * coef[j]
        intercept = mu_y - s
    return (coef^, intercept, n_iter)
