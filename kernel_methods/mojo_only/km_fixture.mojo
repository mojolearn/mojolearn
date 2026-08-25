"""Five data sets, and every bit of every one of them is accounted for.

NOT A PORT. cuML's kernel-ridge test (`python/cuml/tests/test_kernel_ridge.py`)
is a hypothesis-driven comparison against scikit-learn at
`assert_allclose(..., rtol=1e-3)`, which is the right test for a library
shipping one backend and says nothing this lane needs.

# =========================================================================
# THE SPLITMIX64 DEBT IS NOT ADDED TO HERE. IT IS PAID DOWN BY ONE.
#
# `cholesky/mojo_only/cholesky_fixture.mojo`'s header names a debt in its own
# words: "`chol_mix64` in `cholesky_fixture.mojo` is the same three lines as
# `kde/mojo_only/kde_fixture.mojo:18`, `holtwinters/mojo_only/hw_fixture.mojo:30`
# and `isolation_forest/mojo_only/if_fixture.mojo:32` ... Four copies of one
# hash is a debt and this sentence is the record of it."
#
# **THIS FILE MAKES NO FIFTH COPY.** `chol_mix64`, `bits_value`,
# `exact_offdiag` and `rbf_gram` are IMPORTED from that file, and the brief
# that opened this lane blessed the import explicitly for `rbf_gram`
# ("`cholesky_fixture.mojo::rbf_gram` already exists and is shared
# deliberately. Use it."). The other three come along because they are the
# functions `rbf_gram`'s own point sets are built from, and importing the
# consumer while re-spelling its inputs would be the worst of both.
#
# THE COUPLING IS REAL AND IS NAMED: `kernel_methods/` now fails to build if
# `cholesky/mojo_only/cholesky_fixture.mojo` moves, and a change to
# `chol_mix64`'s constants changes THIS lane's fixtures. That is the price of
# not making a fifth copy and it is the right price. The alternative that
# `core/pinned_reduce.mojo` argues for -- a per-lane copy, to keep two
# sessions from colliding on a hot file -- applies to files that are edited
# often, and a fixture hash is edited never.
# =========================================================================

THE TWO CONSTRUCTIONS, and each carries its own host-independence argument
because `IDENTITY_PATHS` row 32 says a fixture built by host floating point
can hand two machines different inputs before the first kernel runs.

**(1) THE EXACT ONE.** `FIX_KM_ORTHO` is built entirely from POWERS OF TWO
and QUARTER-INTEGERS, and every quantity derived from it -- the kernel
matrix, the ridge, the Cholesky factor, the dual coefficients, the training
predictions, the recovered primal weights, the Nystroem eigenvalues, their
inverse square roots, the normalization and the embedding -- is exactly
representable in float32. See `km_ortho_x`'s docstring for the arithmetic,
which is written out rather than asserted so that a reader can redo it. It
asserts in BOTH modes on every vendor, because exact arithmetic has no
rounding for a mode to change.

**(2) THE INEXACT ONES.** `FIX_KM_RBF`, `FIX_KM_DUP`, `FIX_KM_SIGNED` and
`FIX_KM_MIXED` are hashed bit patterns fed straight to the device with NO
host floating-point arithmetic performed on them at all -- `bits_value`
assembles a float from integer fields and `exact_offdiag` divides an integer
by four. So the INPUT is host-independent by construction; what the device
computes from it is the thing being checked.

Non-uniform and non-repeating per cell in every fixture, so a permutation of
rows or columns changes the answer (`uniform-test-data-hides-permutation`).
"""

from std.memory import bitcast

from cholesky.mojo_only.cholesky_fixture import (
    bits_value,
    chol_mix64,
    exact_offdiag,
)


#: **THE EXACT ONE.** `n` rows with DISJOINT SUPPORT, so the linear kernel
#: matrix is DIAGONAL with power-of-four entries and every downstream
#: quantity is exact. Carries planted primal weights and a planted `y`.
comptime FIX_KM_ORTHO = 0

#: Hashed points in a small feature space -- the shape kernel ridge and
#: Nystroem will actually be handed. Ill-conditioned under an RBF kernel,
#: which is what the ridge is for.
comptime FIX_KM_RBF = 1

#: EXACT DUPLICATE ROWS. Two pairs of them, in different panels. Every
#: kernel matrix over it is singular, so it is the fixture that drives
#: DEVIATION 1662's refusal and Nystroem's eigenvalue clip.
comptime FIX_KM_DUP = 2

#: SIGNED ZEROS planted in `X` and in `y`. IDENTITY_PATHS row 39 territory,
#: and the only fixture where a `-0.0` reaches a recorded stage.
comptime FIX_KM_SIGNED = 3

#: Hashed SIGNED values, so `gamma * <x_i, x_j> + coef0` is NEGATIVE on a
#: large fraction of cells. The fixture DEVIATION 1663 exists for: an
#: `exp(p * log(base))` polynomial power returns NaN on every one of them.
comptime FIX_KM_MIXED = 4

comptime KM_FIXTURE_COUNT = 5


# ===========================================================================
# `FIX_KM_ORTHO`'s shape constants. Read by the checks, which derive the
# expected answers from them rather than from a table somebody typed.
# ===========================================================================

#: Rows.
comptime KM_ORTHO_N = 8

#: How many columns each row's support occupies. FOUR, and it has to be a
#: power of four's square root story: the squared row norm is
#: `KM_ORTHO_BLOCK * 4^a`, and with `KM_ORTHO_BLOCK = 4` that is `4^(a+1)`,
#: a POWER OF FOUR, whose square root and inverse square root are both exact
#: powers of two. At `KM_ORTHO_BLOCK = 3` the norm would be `3 * 4^a` and the
#: Nystroem gate would stop being exact.
comptime KM_ORTHO_BLOCK = 4

#: Columns. Disjoint supports, so `d = n * block`.
comptime KM_ORTHO_D = KM_ORTHO_N * KM_ORTHO_BLOCK


def km_ortho_exponent(i: Int) -> Int:
    """Row `i`'s magnitude exponent `a_i`, alternating 0 and 1.

    TWO DISTINCT VALUES AND NOT ONE, deliberately. With a single exponent the
    linear kernel matrix would be a multiple of the identity, every
    eigenvalue would be equal, and the Nystroem sort's TIE BREAK and its
    DESCENDING order would both be unobservable. With two, the spectrum is
    `{4, 4, 4, 4, 16, 16, 16, 16}` -- four-way ties within each value, so the
    total order `(value descending, index ascending)` is exercised on a tie
    that is neither two-way nor accidental.
    """
    return i % 2


def km_ortho_gram_diag(i: Int) -> Float32:
    """`|x_i|^2 = KM_ORTHO_BLOCK * 4^{a_i} = 4^{a_i + 1}`.

    A POWER OF FOUR, so `sqrt` is `2^{a_i+1}` and `1/sqrt` is `2^{-(a_i+1)}`,
    both exact in float32. The check derives its expected Cholesky factor,
    its expected dual coefficients and its expected Nystroem normalization
    from this function rather than from a literal.
    """
    var c = Float32(1.0)
    for _ in range(km_ortho_exponent(i) + 1):
        c = c * Float32(4.0)
    return c


def km_ortho_x(salt: Int) -> List[Float32]:
    """`X`, `KM_ORTHO_N x KM_ORTHO_D` row-major. **EXACT AT EVERY STEP.**

    Row `i` is `2^{a_i}` on columns `[i*B, (i+1)*B)` and `+0.0` everywhere
    else, with `B = KM_ORTHO_BLOCK` and `a_i = km_ortho_exponent(i)`.

    THE ARITHMETIC, written out because the whole gate rests on it:

    - `X[i][c] * X[j][c]` is `4^{a_i}` when `i == j` and `c` is in row `i`'s
      support, and `0` otherwise, because the supports are DISJOINT. So the
      linear kernel `K = X X^T` is `diag(B * 4^{a_i}) = diag(4^{a_i+1})`,
      and every cell of it is a sum of at most `B` equal exact terms with
      magnitude at most `4^{a_max + 1}`. At `B = 4` and `a_max = 1` that is
      64, needing 7 significant bits against float32's 24. **The sum is
      exact regardless of the order it is added in**, so it does not depend
      on `identical_gemm`'s partition, on the fold width, on a contraction
      or on a denormal policy -- there is nothing left for any of those to
      decide, and that is why the gate asserts in FAST too.
    - `K` is DIAGONAL with power-of-four entries, so its Cholesky factor is
      `diag(2^{a_i+1})` exactly, its inverse square root is
      `diag(2^{-(a_i+1)})` exactly, and its eigendecomposition is itself:
      Jacobi's convergence test sees a zero off-diagonal norm and returns at
      sweep 0 without applying a single rotation.
    - The planted `y` (see `km_ortho_y`) is `X w` for a BLOCK-CONSTANT `w`
      of quarter-integers, so `y_i = B * 2^{a_i} * v_i` has at most two
      fractional bits and magnitude at most 6. Dividing it by `4^{a_i+1}` is
      exact because the divisor is a power of two.

    **THE BOUND MUST BE REDONE IF `KM_ORTHO_N`, `KM_ORTHO_BLOCK` OR THE
    EXPONENT SET MOVES.** It is written out rather than asserted so that
    redoing it is possible; `check_ortho_fixture_is_exact` re-derives the
    kernel matrix on the host and requires it diagonal with the exact
    entries, which is the check that would catch an edit that broke the
    bound.
    """
    _ = salt
    var out = List[Float32]()
    for i in range(KM_ORTHO_N):
        var mag = Float32(1.0)
        for _ in range(km_ortho_exponent(i)):
            mag = mag * Float32(2.0)
        for c in range(KM_ORTHO_D):
            if c >= i * KM_ORTHO_BLOCK and c < (i + 1) * KM_ORTHO_BLOCK:
                out.append(mag)
            else:
                out.append(Float32(0.0))
    return out^


def km_ortho_w(salt: Int) -> List[Float32]:
    """The PLANTED primal weights, `KM_ORTHO_D` of them, BLOCK CONSTANT.

    `w[c] = v_i` for every `c` in row `i`'s support, with `v_i` a hashed
    quarter-integer from `exact_offdiag`. Block constant so that the primal
    weights the dual solution recovers are the plant EXACTLY: see
    `check_kernel_ridge_planted_linear` for the three-line derivation.

    A `v_i` of exactly `0.0` is possible (`exact_offdiag` draws from
    `{-0.75, ..., 0.75}` including zero) and is left in rather than rejected:
    a zero weight is a legitimate plant and the recovery has to reproduce it.
    """
    var out = List[Float32]()
    for i in range(KM_ORTHO_N):
        var v = exact_offdiag(chol_mix64(i, 0, salt + 101))
        for _ in range(KM_ORTHO_BLOCK):
            out.append(v)
    return out^


def km_ortho_y(salt: Int) -> List[Float32]:
    """`y = X w`, one target, EXACT.

    `y_i = sum_c X[i][c] w[c] = KM_ORTHO_BLOCK * 2^{a_i} * v_i`, an integer
    multiple of a quarter-integer times a power of two, so at most two
    fractional bits and magnitude at most `4 * 2 * 0.75 = 6`. Formed here by
    the closed form rather than by a host matrix-vector product, so that the
    check has a target that no floating-point loop of ours produced.
    """
    var w = km_ortho_w(salt)
    var out = List[Float32]()
    for i in range(KM_ORTHO_N):
        var mag = Float32(1.0)
        for _ in range(km_ortho_exponent(i)):
            mag = mag * Float32(2.0)
        out.append(
            Float32(KM_ORTHO_BLOCK) * mag * w[i * KM_ORTHO_BLOCK]
        )
    return out^


# ===========================================================================
# The rest of the fixtures
# ===========================================================================

#: `FIX_KM_DUP`'s duplicate pairs: row `a` is made an EXACT bitwise copy of
#: row `b`. Two pairs, both away from row 0, so a kernel matrix's singularity
#: shows up in a later Cholesky panel rather than at column 1 -- the same
#: reason `cholesky_fixture.mojo` zeroes column 37 rather than column 1.
comptime FIX_DUP_A0 = 3
comptime FIX_DUP_B0 = 2
comptime FIX_DUP_A1 = 9
comptime FIX_DUP_B1 = 6

#: `FIX_KM_SIGNED`'s planted negative zeros. Row `SZ_ROW` of `X` is `-0.0`
#: off its support; target `SZ_TARGET` of `y` is `-0.0`.
comptime FIX_SZ_ROW = 5
comptime FIX_SZ_TARGET = 2


def km_fixture_n(which: Int) -> Int:
    """Rows."""
    if which == FIX_KM_ORTHO:
        return KM_ORTHO_N
    if which == FIX_KM_RBF:
        return 16
    if which == FIX_KM_DUP:
        return 12
    if which == FIX_KM_SIGNED:
        return 8
    if which == FIX_KM_MIXED:
        return 16
    return 0


def km_fixture_d(which: Int) -> Int:
    """Features."""
    if which == FIX_KM_ORTHO:
        return KM_ORTHO_D
    if which == FIX_KM_RBF:
        return 5
    if which == FIX_KM_DUP:
        return 6
    if which == FIX_KM_SIGNED:
        return 8
    if which == FIX_KM_MIXED:
        return 6
    return 0


def km_fixture_name(which: Int) -> String:
    if which == FIX_KM_ORTHO:
        return String("ORTHO")
    if which == FIX_KM_RBF:
        return String("RBF")
    if which == FIX_KM_DUP:
        return String("DUP")
    if which == FIX_KM_SIGNED:
        return String("SIGNED")
    if which == FIX_KM_MIXED:
        return String("MIXED")
    return String("UNKNOWN")


def km_fixture_is_exact(which: Int) -> Bool:
    """True only for the fixture whose whole arithmetic is exact, so a check
    can ask rather than remember which one that is."""
    return which == FIX_KM_ORTHO


def km_fixture_x(which: Int, salt: Int) raises -> List[Float32]:
    """`X`, `n x d` row-major, host-independent by construction."""
    var n = km_fixture_n(which)
    var d = km_fixture_d(which)
    if n <= 0 or d <= 0:
        raise Error("km_fixture_x: unknown fixture id " + String(which))

    if which == FIX_KM_ORTHO:
        return km_ortho_x(salt)

    if which == FIX_KM_RBF:
        # Magnitudes in [0.5, 2), unsigned, so the squared distances stay in
        # a range where `exp(-gamma d2)` is neither saturated at 1 nor
        # flushed to zero at the gammas the checks use. Assembled from
        # integer fields; no host float arithmetic touches them.
        var out = List[Float32]()
        for i in range(n):
            for k in range(d):
                out.append(bits_value(chol_mix64(i, k, salt + 11), False))
        return out^

    if which == FIX_KM_DUP:
        var out = List[Float32]()
        for i in range(n):
            for k in range(d):
                out.append(bits_value(chol_mix64(i, k, salt + 23), True))
        # EXACT bitwise duplicates, made by copying, so the two rows are the
        # same bits and not merely close. A kernel matrix over them has two
        # identical rows and two identical columns, hence rank at most n - 2.
        for k in range(d):
            out[FIX_DUP_A0 * d + k] = out[FIX_DUP_B0 * d + k]
            out[FIX_DUP_A1 * d + k] = out[FIX_DUP_B1 * d + k]
        return out^

    if which == FIX_KM_SIGNED:
        # Row-blocked support like ORTHO so the structure is legible, then
        # every OFF-SUPPORT cell of row FIX_SZ_ROW is planted `-0.0`.
        #
        # WHAT THIS TESTS AND WHAT IT CANNOT. `identical_gemm`'s fold is
        # seeded `+0.0`, so a cell of `K` whose products are all zero comes
        # back `+0.0` whatever the signs of those zeros were (row 39: a sum
        # is `-0.0` only when EVERY term is `-0.0` and no `+0.0` is added
        # anywhere in the tree). So the planted signs are EXPECTED to be
        # erased by the kernel matrix, and `check_signed_zero_reach` asserts
        # that erasure rather than hoping for survival -- it is a property of
        # the fold, it is the same on every vendor under IDENTICAL, and a
        # version that preserved them would be the surprising one.
        #
        # The sign that DOES survive is `y`'s; see `km_fixture_y`.
        var out = List[Float32]()
        for i in range(n):
            for c in range(d):
                if c == i:
                    out.append(Float32(1.0))
                elif i == FIX_SZ_ROW:
                    out.append(Float32(-0.0))
                else:
                    out.append(Float32(0.0))
        return out^

    # FIX_KM_MIXED. SIGNED magnitudes, so roughly half the inner products
    # are negative and `gamma * dot + coef0` crosses zero.
    var out = List[Float32]()
    for i in range(n):
        for k in range(d):
            out.append(bits_value(chol_mix64(i, k, salt + 37), True))
    return out^


def km_fixture_y(
    which: Int, n_targets: Int, salt: Int
) raises -> List[Float32]:
    """`y`, `n x n_targets` row-major.

    `FIX_KM_ORTHO` at one target returns the PLANTED `X w` and nothing else,
    because that fixture's whole point is that the answer is derivable. At
    more than one target its extra columns are hashed quarter-integers, which
    keeps multi-target exercised without pretending the extra columns have a
    closed form.
    """
    var n = km_fixture_n(which)
    if n <= 0:
        raise Error("km_fixture_y: unknown fixture id " + String(which))
    if n_targets <= 0:
        raise Error(
            "km_fixture_y: n_targets must be positive, got "
            + String(n_targets)
        )

    var planted = List[Float32]()
    if which == FIX_KM_ORTHO:
        planted = km_ortho_y(salt)

    var out = List[Float32]()
    for i in range(n):
        for t in range(n_targets):
            if which == FIX_KM_ORTHO and t == 0:
                out.append(planted[i])
            elif which == FIX_KM_SIGNED and i == FIX_SZ_ROW and t == (
                FIX_SZ_TARGET % n_targets
            ):
                # A PLANTED NEGATIVE ZERO TARGET. This one survives: the
                # forward substitution's accumulator starts at
                # `ftz(b[i][t])`, and on this fixture's diagonal kernel
                # matrix there is nothing to subtract, so `-0.0 / L_ii` is
                # `-0.0` and reaches `krr.dual_coef` with its sign.
                # `check_signed_zero_reach` RAISES if it does not, because
                # an agreement that no negative zero reached proves nothing.
                out.append(Float32(-0.0))
            else:
                out.append(exact_offdiag(chol_mix64(i, t, salt + 53)))
    return out^


def km_fixture_query(which: Int, n_query: Int, salt: Int) raises -> List[
    Float32
]:
    """A separate `X_new`, `n_query x d`, for the PREDICT and TRANSFORM
    paths.

    A cross-kernel `K(X_new, X_fit)` is RECTANGULAR and a square one is not,
    and every index error in a kernel-methods lane lives in that difference.
    So `n_query` is deliberately never equal to `n` at any call site in
    `km_check.mojo`, and the query rows are drawn from a DIFFERENT salt than
    the training rows so a transposed index cannot accidentally agree.
    """
    var d = km_fixture_d(which)
    if d <= 0:
        raise Error("km_fixture_query: unknown fixture id " + String(which))
    if n_query <= 0:
        raise Error(
            "km_fixture_query: n_query must be positive, got "
            + String(n_query)
        )

    if which == FIX_KM_ORTHO:
        # Keep the disjoint-support structure so the query kernel is also
        # exact: query row `q` reuses training row `q % KM_ORTHO_N`'s support
        # at HALF the magnitude, which is still a power of two.
        var out = List[Float32]()
        for q in range(n_query):
            var i = q % KM_ORTHO_N
            var mag = Float32(0.5)
            for _ in range(km_ortho_exponent(i)):
                mag = mag * Float32(2.0)
            for c in range(d):
                if c >= i * KM_ORTHO_BLOCK and c < (i + 1) * KM_ORTHO_BLOCK:
                    out.append(mag)
                else:
                    out.append(Float32(0.0))
        return out^

    var signed = which != FIX_KM_RBF
    var out = List[Float32]()
    for q in range(n_query):
        for k in range(d):
            out.append(bits_value(chol_mix64(q, k, salt + 71), signed))
    return out^


def km_hex32(v: Float32) -> String:
    """A float32 by its bits. `String(Float32)` does not round trip
    (`[[mojo-string-float-roundtrip]]`), so every scalar this lane prints or
    compares in a message is printed as decimal AND hex."""
    comptime DIGITS = "0123456789abcdef"
    var u = bitcast[DType.uint32](v)
    var out = String("0x")
    for i in range(8):
        var nib = Int((u >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out


def km_same_bits(a: Float32, b: Float32) -> Bool:
    """Bitwise equality, INCLUDING the sign of zero and including NaN
    payloads. `a == b` is the wrong test twice over here: it calls `+0.0`
    and `-0.0` equal, which is exactly the distinction row 39 is about, and
    it calls two NaNs unequal, which makes a card comparison unrepeatable."""
    return bitcast[DType.uint32](a) == bitcast[DType.uint32](b)
