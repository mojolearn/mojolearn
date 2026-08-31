# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Five fixtures, and every bit of every one of them is accounted for.

Each exists to make ONE claim checkable, and none of them is a random data
set with a tolerance draped over it.

    GP_FIX_PLANTED      a planted function at WELL-SEPARATED points, where
                        the posterior mean at a training point must come
                        back as the observation. Also the fixture that
                        drives the predictive-variance CLAMP, because it
                        predicts AT the training points and the true
                        variance there is exactly zero
    GP_FIX_DUPLICATE    EXACT duplicate inputs. The kernel matrix is
                        exactly singular without the ridge and the pivot
                        refusal fires at a HAND-DERIVABLE column
    GP_FIX_HANDWORKED   one dimension, two points, a coordinate-free
                        kernel, and a log marginal likelihood worked out
                        by hand in this file's comments with every
                        intermediate EXACT in float32 except two
                        logarithms
    GP_FIX_ARD          three dimensions of which ONE IS IRRELEVANT, and
                        irrelevant EXACTLY rather than approximately
    GP_FIX_SIGNED_ZERO  a coordinate carrying both zero signs.
                        IDENTITY_PATHS row 39

**NO NEW HASH FUNCTION AND NO NEW VALUE CONSTRUCTOR.**
`cholesky/checks/cholesky_fixture.mojo` already carries `chol_mix64`
(splitmix64 over three integers), `bits_value` (a float32 assembled from
hashed bits, never from host arithmetic), `exact_offdiag` (a quarter-integer
below one in magnitude) and `rbf_gram`, and its own README records that four
copies of one splitmix64 already exist in this tree and that a fifth would
be a debt. So this file IMPORTS them. `rbf_gram`'s docstring is explicit
that it was put in the Cholesky lane rather than in this one "so that the
callers cannot disagree about the fixture", and
`check_posterior_recovers_training` uses it for exactly that.

THE HOST FLOATING-POINT BUDGET, stated because a fixture that rounds is a
fixture two hosts can disagree about. Every value below comes from one of
three sources and no other:

1. `bits_value` and `exact_offdiag`, which assemble a float32 from integer
   arithmetic and at most one exact division by four;
2. an exact integer converted with `Float32(i)`, which is exact for every
   `|i| < 2^24` and every fixture here stays under 64;
3. a literal that is a small dyadic rational (`0.0`, `-0.0`, `0.25`, `0.5`,
   `1.0`, `1.5`, `1.75`, `2.0`, `2.5`, `5.0`, `7.0`, `10.0`, `15.0`), each
   of which is exactly representable and parses to the same bits on any
   conforming compiler.

There is no host `exp`, no host `sqrt` and no host division outside
`exact_offdiag`'s divide by four anywhere in this file.
"""

from std.memory import bitcast

from cholesky.checks.cholesky_fixture import (
    bits_value,
    chol_mix64,
    exact_offdiag,
)
from cholesky.checks.potrf import chol_jitter_pinned
from gaussian_process.checks.kernels import (
    GPKernelSpec,
    gp_kernel_const,
    gp_kernel_matern,
    gp_kernel_prod,
    gp_kernel_rbf,
    gp_kernel_sum,
    gp_kernel_white,
)


comptime GP_FIX_PLANTED = 0
comptime GP_FIX_DUPLICATE = 1
comptime GP_FIX_HANDWORKED = 2
comptime GP_FIX_ARD = 3
comptime GP_FIX_SIGNED_ZERO = 4
comptime GP_FIXTURE_COUNT = 5

#: `GP_FIX_PLANTED`'s point spacing. Integers, so `Float32(4 * i)` is exact,
#: and 4 length scales apart so the nearest off-diagonal kernel value is
#: `exp(-8) = 3.35e-4` and the next `exp(-32) = 1.3e-14`. **The matrix is
#: therefore numerically near-identity and WELL CONDITIONED**, which is the
#: point: `cholesky/README.md`'s second finding records that an RBF Gram
#: matrix in float32 at 64 points is numerically close to singular, and a
#: fixture that is meant to test posterior RECOVERY must not also be testing
#: conditioning. The near-singular case has its own fixture (`rbf_gram`, in
#: the Cholesky lane) and its own check.
comptime GP_PLANTED_SPACING = 4

#: `GP_FIX_DUPLICATE`'s duplicated pair. Rows 0 and 1 are the SAME point, so
#: the first two rows and columns of the kernel matrix are exactly
#: `[[1, 1], [1, 1]]` for any RBF: the diagonal is `exp(-0) = 1` exactly and
#: the off-diagonal is `exp(-0) = 1` exactly too, because the two rows have
#: identical bits. That makes the failure HAND-DERIVABLE rather than
#: probable -- see `gp_duplicate_expected_info`.
comptime GP_DUP_ROW_A = 0
comptime GP_DUP_ROW_B = 1

#: `GP_FIX_SIGNED_ZERO`'s planted zeros: row 0 carries `-0.0` in feature 0
#: and row 1 carries `+0.0` in the same feature. No ordinary input mixes the
#: two, so they are PLANTED (the Cholesky lane's `FIX_SIGNED_ZERO` makes the
#: same argument).
comptime GP_SZ_NEG_ROW = 0
comptime GP_SZ_POS_ROW = 1
comptime GP_SZ_COL = 0

#: `GP_FIX_ARD`'s irrelevant feature, and the constant every row carries in
#: it. Because the value is the SAME in every row, `x[i,2]/l - x[j,2]/l` is
#: exactly `+0.0` for every pair at every length scale, so feature 2
#: contributes exactly nothing to any cell. **The irrelevance is EXACT, not
#: approximate**, which is what lets `check_kernels_vs_oracle` gate it by
#: bits instead of by a tolerance.
comptime GP_ARD_DEAD_FEATURE = 2


def gp_fixture_name(which: Int) -> String:
    if which == GP_FIX_PLANTED:
        return String("planted")
    if which == GP_FIX_DUPLICATE:
        return String("duplicate")
    if which == GP_FIX_HANDWORKED:
        return String("handworked")
    if which == GP_FIX_ARD:
        return String("ard")
    if which == GP_FIX_SIGNED_ZERO:
        return String("signed_zero")
    return String("fixture?")


def gp_fixture_n(which: Int) -> Int:
    """Training rows."""
    if which == GP_FIX_PLANTED:
        return 16
    if which == GP_FIX_DUPLICATE:
        return 4
    if which == GP_FIX_HANDWORKED:
        return 2
    if which == GP_FIX_ARD:
        return 12
    return 8


def gp_fixture_d(which: Int) -> Int:
    """Features."""
    if which == GP_FIX_PLANTED:
        return 1
    if which == GP_FIX_DUPLICATE:
        return 2
    if which == GP_FIX_HANDWORKED:
        return 1
    if which == GP_FIX_ARD:
        return 3
    return 2


def gp_fixture_n_star(which: Int) -> Int:
    """Test rows."""
    if which == GP_FIX_PLANTED:
        return 16
    if which == GP_FIX_HANDWORKED:
        return 2
    if which == GP_FIX_ARD:
        return 6
    return 4


def gp_fixture_predicts_at_training_points(which: Int) -> Bool:
    """True where `X_star` IS `X_train`, so the true predictive variance is
    exactly zero at every test point WHEN THE FIT CARRIES NO RIDGE and no
    white noise, and the float32 computation of it then lands a few ulps
    either side. `GP_FIX_PLANTED` is that case and is where the clamp is
    expected to fire; `GP_FIX_HANDWORKED` also predicts at its training
    points but carries a `WhiteKernel(10)`, whose noise is in `k**` and not
    in the cross-covariance, so its predictive variance is about 13.75 and
    nothing clamps. `check_variance_is_nonnegative_and_clamps_are_counted`
    raises if the clamp fires on NO fixture at all, because a clamp counter
    that has never counted anything is not gated."""
    return which == GP_FIX_PLANTED or which == GP_FIX_HANDWORKED


def gp_fixture_x(which: Int, salt: Int) raises -> List[Float32]:
    """`n_train x n_features` row-major training inputs."""
    var n = gp_fixture_n(which)
    var d = gp_fixture_d(which)
    var x = List[Float32]()

    if which == GP_FIX_PLANTED:
        # x_i = 4 i, exactly. Integers under 64, so Float32(i) is exact.
        for i in range(n):
            x.append(Float32(GP_PLANTED_SPACING * i))
        return x^

    if which == GP_FIX_HANDWORKED:
        # The kernel is coordinate-free (Constant + White), so the two
        # coordinates only have to be finite and distinct. See
        # `gp_handworked_notes` for the whole derivation.
        x.append(Float32(0.0))
        x.append(Float32(1.0))
        return x^

    if which == GP_FIX_DUPLICATE:
        for i in range(n):
            for f in range(d):
                var src = i
                if i == GP_DUP_ROW_B:
                    # THE DUPLICATE. Row B is row A's bits, exactly.
                    src = GP_DUP_ROW_A
                x.append(
                    bits_value(chol_mix64(src, f, salt + 11), True)
                )
        return x^

    if which == GP_FIX_ARD:
        for i in range(n):
            for f in range(d):
                if f == GP_ARD_DEAD_FEATURE:
                    # THE IRRELEVANT FEATURE: the same value in every row,
                    # so its contribution to every pair is exactly +0.0.
                    # Hashed on the feature index alone, never on the row.
                    x.append(
                        bits_value(chol_mix64(0, f, salt + 23), True)
                    )
                else:
                    x.append(
                        bits_value(chol_mix64(i, f, salt + 23), True)
                    )
        return x^

    # GP_FIX_SIGNED_ZERO
    for i in range(n):
        for f in range(d):
            x.append(bits_value(chol_mix64(i, f, salt + 37), True))
    x[GP_SZ_NEG_ROW * d + GP_SZ_COL] = bitcast[DType.float32](
        UInt32(0x80000000)
    )
    x[GP_SZ_POS_ROW * d + GP_SZ_COL] = bitcast[DType.float32](UInt32(0))
    return x^


def gp_fixture_x_positive_zero(
    which: Int, salt: Int
) raises -> List[Float32]:
    """`GP_FIX_SIGNED_ZERO`'s inputs with the planted `-0.0` replaced by
    `+0.0` and NOTHING else changed.

    The pair is what `check_kernels_vs_oracle`'s row-39 arm compares: a
    covariance function is blind to the sign of a zero coordinate
    (`-0.0 - +0.0` is `-0.0`, and `fma(-0.0, -0.0, +0.0)` is `+0.0`), so the
    two kernel matrices must be equal BIT FOR BIT. That is a claim, and it
    is asserted rather than assumed.
    """
    var x = gp_fixture_x(which, salt)
    var d = gp_fixture_d(which)
    x[GP_SZ_NEG_ROW * d + GP_SZ_COL] = bitcast[DType.float32](UInt32(0))
    return x^


def gp_fixture_x_star(which: Int, salt: Int) raises -> List[Float32]:
    """`n_star x n_features` row-major test inputs."""
    if gp_fixture_predicts_at_training_points(which):
        return gp_fixture_x(which, salt)
    var ns = gp_fixture_n_star(which)
    var d = gp_fixture_d(which)
    var xs = List[Float32]()
    if which == GP_FIX_ARD:
        for i in range(ns):
            for f in range(d):
                if f == GP_ARD_DEAD_FEATURE:
                    # The same constant the training rows carry, so the
                    # feature stays exactly irrelevant in the
                    # cross-covariance too.
                    xs.append(
                        bits_value(chol_mix64(0, f, salt + 23), True)
                    )
                else:
                    xs.append(
                        bits_value(chol_mix64(i + 512, f, salt + 23), True)
                    )
        return xs^
    for i in range(ns):
        for f in range(d):
            xs.append(bits_value(chol_mix64(i + 512, f, salt + 71), True))
    return xs^


def gp_fixture_y(which: Int, salt: Int) raises -> List[Float32]:
    """`n_train` targets.

    `exact_offdiag` everywhere except the hand-worked fixture, so every
    target is a quarter-integer below one in magnitude: exactly
    representable, and exactly representable after multiplication by another
    such value, which is what makes the recovery check's residual a
    statement about the SOLVE rather than about the data.
    """
    var n = gp_fixture_n(which)
    if which == GP_FIX_HANDWORKED:
        # See `gp_handworked_notes`. y = (5, -5) is what makes every
        # intermediate of the solve exact.
        var y = List[Float32]()
        y.append(Float32(5.0))
        y.append(Float32(-5.0))
        return y^
    var y = List[Float32]()
    for i in range(n):
        y.append(exact_offdiag(chol_mix64(i, 0, salt + 101)))
    return y^


def gp_fixture_alpha(which: Int) raises -> Float32:
    """The ridge each fixture is fitted with.

    `GP_FIX_HANDWORKED` takes NO ridge, because its whole point is that
    every intermediate is exact and a ridge of `2^-20` on a diagonal of 25
    would round.

    `GP_FIX_PLANTED` takes NO ridge either, and for a reason worth stating:
    it is the fixture that has to DRIVE THE PREDICTIVE-VARIANCE CLAMP. Its
    test points are its training points, so the true predictive variance at
    each of them is `k** - k^T K^-1 k`, which with no ridge is exactly zero
    in exact arithmetic and lands a few ulps either side of zero in float32
    -- so roughly half the test points come back negative and the clamp
    fires. WITH a ridge the same quantity is about `alpha` rather than zero
    (the ridge is what the posterior does not explain), which at `2^-20 =
    9.5e-7` is comfortably above the float32 noise floor of about `1.2e-7`
    and the clamp would never fire at all. The fixture is well conditioned
    enough to factor without a ridge: its points are four length scales
    apart, so the largest off-diagonal is `exp(-8) = 3.4e-4` and the matrix
    is numerically near-identity.

    Everything else takes the profile's pinned ridge, which is the Cholesky
    lane's jitter and the only one there is (DEVIATION 1751). Both pinned
    values are therefore exercised by a real fit.
    """
    if which == GP_FIX_HANDWORKED or which == GP_FIX_PLANTED:
        return Float32(0.0)
    return chol_jitter_pinned()


def gp_fixture_kernel(which: Int) raises -> GPKernelSpec:
    """The covariance function each fixture is fitted with.

    Chosen so that the five fixtures between them exercise all four leaves
    and both operators at least once: RBF isotropic (planted, duplicate),
    RBF ARD (ard), Constant and White and Sum (handworked), Matern
    (signed_zero). The per-kernel bit-level gate is
    `check_kernels_vs_oracle`, which sweeps `gp_kernel_case` instead and
    covers every kernel and every `nu`; these are the ones a FIT runs.
    """
    if which == GP_FIX_HANDWORKED:
        # ConstantKernel(15) + WhiteKernel(10). See `gp_handworked_notes`.
        return gp_kernel_sum(
            gp_kernel_const(Float32(15.0)), gp_kernel_white(Float32(10.0))
        )
    if which == GP_FIX_ARD:
        # 0.5 and 1.0 rather than 1.0 and 2.0: `bits_value` puts every
        # coordinate in [0.5, 2) in magnitude, so dividing by 0.5 SPREADS
        # the points instead of compressing them, and a spread point set is
        # what keeps a 12-point RBF Gram matrix out of the near-singular
        # regime `cholesky/README.md`'s second finding describes. Both are
        # powers of two, so the scaling divide is exact and an ARD indexing
        # defect shows up as a WRONG VALUE rather than as a rounding
        # difference a reader could argue about.
        var ls = List[Float32]()
        ls.append(Float32(0.5))
        ls.append(Float32(1.0))
        ls.append(Float32(7.0))
        return gp_kernel_rbf(ls)
    if which == GP_FIX_SIGNED_ZERO:
        var ls = List[Float32]()
        ls.append(Float32(0.5))
        return gp_kernel_matern(ls, Float32(1.5))
    var one = List[Float32]()
    one.append(Float32(1.0))
    return gp_kernel_rbf(one)


def gp_fixture_ard_reduced_kernel() raises -> GPKernelSpec:
    """`GP_FIX_ARD`'s kernel with the irrelevant feature's length scale
    dropped: `RBF([1.0, 2.0])` over the first two features only.

    `check_kernels_vs_oracle`'s ARD arm requires the three-feature matrix
    and the two-feature matrix to be equal BIT FOR BIT. Two things fail that
    at once if either is wrong: an implementation that indexed the length
    scales with `ls[0]` at every feature (so feature 1 would be scaled by
    0.5 instead of 1.0), and one that let a constant feature contribute
    anything at all.
    """
    var ls = List[Float32]()
    ls.append(Float32(0.5))
    ls.append(Float32(1.0))
    return gp_kernel_rbf(ls)


def gp_fixture_ard_reduced_x(which: Int, salt: Int) raises -> List[Float32]:
    """`GP_FIX_ARD`'s training inputs with the irrelevant feature removed:
    `n x 2` instead of `n x 3`."""
    var x = gp_fixture_x(which, salt)
    var n = gp_fixture_n(which)
    var d = gp_fixture_d(which)
    var out = List[Float32]()
    for i in range(n):
        for f in range(d):
            if f != GP_ARD_DEAD_FEATURE:
                out.append(x[i * d + f])
    return out^


def gp_duplicate_expected_info() -> Int:
    """`GP_FIX_DUPLICATE` without a ridge stops at `info = 2`, and the
    arithmetic is short enough to write out.

    Rows 0 and 1 hold IDENTICAL BITS, so for any RBF the leading 2x2 block
    of the kernel matrix is exactly

        K[0][0] = exp(-0) = 1        K[0][1] = exp(-0) = 1
        K[1][0] = exp(-0) = 1        K[1][1] = exp(-0) = 1

    -- exactly `1.0` at all four cells, because the squared distance between
    two identical rows is a chain of `fma(+0.0, +0.0, +0.0)` and
    `portable_expf(-0.0)` is exactly `1.0`. The factorization then computes

        L[0][0] = sqrt(1)        = 1     exactly
        L[1][0] = 1 / 1          = 1     exactly
        s_1     = 1 - 1 * 1      = 0     exactly

    and the pivot test is `not (s > 0.0)` (DEVIATION 1634), which `0.0`
    fails. LAPACK's `info` is the ONE-BASED order of the leading minor that
    was not positive definite, so `info = 2`.

    **Nothing about that is probabilistic**, which is the whole reason the
    duplicated pair is rows 0 and 1 rather than an interior pair: no
    rounding from any earlier column can reach it.

    With the pinned ridge the same three lines give `L[0][0] = sqrt(1+j)`,
    `L[1][0] = 1/sqrt(1+j)` and `s_1 = (1+j) - 1/(1+j)`, which is about `2j`
    and strictly positive. That is the whole content of
    `check_duplicate_inputs_need_the_ridge`.
    """
    return 2


def gp_handworked_notes() -> String:
    """`GP_FIX_HANDWORKED` worked out by hand, in one string so that the
    check can PRINT the derivation beside the numbers it got.

    Two points, one feature, `ConstantKernel(15) + WhiteKernel(10)`, no
    ridge, `y = (5, -5)`. The kernel ignores the coordinates entirely
    (`ConstantKernel` is constant and `WhiteKernel` is structural), so

        K = [[15+10, 15], [15, 15+10]] = [[25, 15], [15, 25]]

    and every step below is EXACT in float32 -- every value is a small
    integer or a half-integer, and 25, 15, 5, 3, 16 and 4 all have exact
    square roots or exact quotients where one is taken:

        L[0][0] = sqrt(25)          = 5
        L[1][0] = 15 / 5            = 3
        s_1     = 25 - 3*3          = 16
        L[1][1] = sqrt(16)          = 4
        forward L z = y:  z_0 = 5/5 = 1 ;  z_1 = (-5 - 3*1)/4 = -2
        back   L^T a = z: a_1 = -2/4 = -0.5 ;  a_0 = (1 - 3*(-0.5))/5 = 0.5
        y^T a   = 5*0.5 + (-5)*(-0.5)               = 5
        |K|     = 25*25 - 15*15                     = 400
        log|K|  = 2*(log 5 + log 4) = log 400

    Confirmed independently by the closed form for a 2x2 exchangeable
    matrix: `K^-1 y = (1/400) [[25, -15], [-15, 25]] (5, -5)^T =
    (1/400)(200, -200)^T = (0.5, -0.5)^T`.

        lml = -0.5 * 5 - 0.5 * log(400) - (2/2) * log(2 pi)
            = -2.5 - 2.995732273553991 - 1.8378770664093453
            = -7.333609339963336

    **`dual_coef` and `y^T alpha` are asserted BIT FOR BIT**; `log|K|` and
    `lml` are compared to the float64 values above at a tolerance the check
    prints, because two logarithms are the only inexact operations in the
    whole fixture.
    """
    return String(
        "K=[[25,15],[15,25]] L=[[5,0],[3,4]] dual=(0.5,-0.5)"
        " y^T.dual=5 log|K|=log(400)=5.991464547107982"
        " lml=-7.333609339963336"
    )


comptime GP_HANDWORKED_LOGDET_F64 = Float64(5.991464547107982)
comptime GP_HANDWORKED_YDOTALPHA = Float32(5.0)
comptime GP_HANDWORKED_LML_F64 = Float64(-7.333609339963336)


# ===========================================================================
# THE KERNEL CASES: every kernel and every nu, for the per-cell gate
# ===========================================================================

comptime GP_KCASE_CONST = 0
comptime GP_KCASE_WHITE = 1
comptime GP_KCASE_RBF_ISO = 2
comptime GP_KCASE_RBF_ARD = 3
comptime GP_KCASE_MATERN_05 = 4
comptime GP_KCASE_MATERN_15 = 5
comptime GP_KCASE_MATERN_25 = 6
comptime GP_KCASE_SUM = 7
comptime GP_KCASE_PROD = 8
comptime GP_KCASE_CONST_TIMES_RBF = 9
comptime GP_KCASE_NESTED = 10
comptime GP_KCASE_COUNT = 11


def gp_kernel_case_name(which: Int) -> String:
    if which == GP_KCASE_CONST:
        return String("ConstantKernel(1.75)")
    if which == GP_KCASE_WHITE:
        return String("WhiteKernel(0.25)")
    if which == GP_KCASE_RBF_ISO:
        return String("RBF(1.0)")
    if which == GP_KCASE_RBF_ARD:
        return String("RBF([1.0, 2.0])")
    if which == GP_KCASE_MATERN_05:
        return String("Matern(1.0, nu=0.5)")
    if which == GP_KCASE_MATERN_15:
        return String("Matern(1.0, nu=1.5)")
    if which == GP_KCASE_MATERN_25:
        return String("Matern([1.0, 2.0], nu=2.5)")
    if which == GP_KCASE_SUM:
        return String("RBF(1.0) + WhiteKernel(0.25)")
    if which == GP_KCASE_PROD:
        return String("RBF(1.0) * Matern(1.0, nu=1.5)")
    if which == GP_KCASE_CONST_TIMES_RBF:
        return String("ConstantKernel(1.0) * RBF(1.0)")
    if which == GP_KCASE_NESTED:
        return String("(ConstantKernel(1.75) * RBF(1.0)) + WhiteKernel(0.25)")
    return String("kcase?")


def gp_kernel_case(which: Int, d: Int) raises -> GPKernelSpec:
    """The eleven kernels `check_kernels_vs_oracle` and
    `check_kernel_algebra` sweep. `d` is the feature count the ARD cases
    size themselves to; the isotropic cases ignore it.

    `GP_KCASE_CONST_TIMES_RBF` is `ConstantKernel(1.0) * RBF(1.0)` and it is
    here for one specific claim: it must equal the bare `RBF(1.0)` **BIT FOR
    BIT**, because `identical_mul(1.0, k)` is exactly `k` at every input
    including both zero signs. If it does not, either the Product node is
    not elementwise or the constant leaf is not writing what it says.
    """
    var iso = List[Float32]()
    iso.append(Float32(1.0))
    if which == GP_KCASE_CONST:
        return gp_kernel_const(Float32(1.75))
    if which == GP_KCASE_WHITE:
        return gp_kernel_white(Float32(0.25))
    if which == GP_KCASE_RBF_ISO:
        return gp_kernel_rbf(iso)
    if which == GP_KCASE_RBF_ARD:
        return gp_kernel_rbf(_ard_scales(d))
    if which == GP_KCASE_MATERN_05:
        return gp_kernel_matern(iso, Float32(0.5))
    if which == GP_KCASE_MATERN_15:
        return gp_kernel_matern(iso, Float32(1.5))
    if which == GP_KCASE_MATERN_25:
        return gp_kernel_matern(_ard_scales(d), Float32(2.5))
    if which == GP_KCASE_SUM:
        return gp_kernel_sum(
            gp_kernel_rbf(iso.copy()), gp_kernel_white(Float32(0.25))
        )
    if which == GP_KCASE_PROD:
        return gp_kernel_prod(
            gp_kernel_rbf(iso.copy()),
            gp_kernel_matern(iso.copy(), Float32(1.5)),
        )
    if which == GP_KCASE_CONST_TIMES_RBF:
        return gp_kernel_prod(
            gp_kernel_const(Float32(1.0)), gp_kernel_rbf(iso.copy())
        )
    if which == GP_KCASE_NESTED:
        return gp_kernel_sum(
            gp_kernel_prod(
                gp_kernel_const(Float32(1.75)), gp_kernel_rbf(iso.copy())
            ),
            gp_kernel_white(Float32(0.25)),
        )
    raise Error(
        "gp_kernel_case: unknown case " + String(which)
    )


def _ard_scales(d: Int) -> List[Float32]:
    """`[1.0, 2.0, 1.0, 2.0, ...]`, `d` long. Powers of two, so the scaling
    divide is exact and an ARD failure shows up as a WRONG INDEX rather than
    as a rounding difference that a reader could argue about."""
    var ls = List[Float32]()
    for f in range(d):
        if f % 2 == 0:
            ls.append(Float32(1.0))
        else:
            ls.append(Float32(2.0))
    return ls^
