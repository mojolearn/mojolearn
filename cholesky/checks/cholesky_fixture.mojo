# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Six matrices, and every bit of every one of them is accounted for.

NOT A PORT. cuML and cuVS have no Cholesky and therefore no Cholesky
fixtures; the nearest thing in the checkouts is `cuml/cpp/tests/sg/
lars_test.cu:110-125`, which factors a small Gram matrix with `cusolverDnpotrf`
and compares against `updateCholesky` -- an oracle-vs-implementation test with
no bit claim, which is the right test for a library that ships one backend.

`kde/checks/kde_fixture.mojo`'s rule applies here verbatim and then runs
into a wall this lane cannot walk around: **a fixture built by host
floating-point arithmetic can hand two machines different inputs before the
first kernel runs** (IDENTITY_PATHS row 32). A Cholesky fixture, unlike a
KDE one, cannot be assembled entirely from hashed bits, because the input has
to be SYMMETRIC POSITIVE DEFINITE and no bit pattern is that by construction.

So there are exactly two constructions here and each carries its own argument:

**(1) THE EXACT ONES.** `planted_lower` draws `L` from a set of values whose
products and sums are EXACTLY representable in float32 at this `n`, and
`gram_from_lower` forms `A = L L^T` by plain host arithmetic. Every operation
in it is exact, so the result is a pure function of the plant and does not
depend on the host's rounding, its FMA contraction, its denormal policy or
its libm -- there is nothing for any of those to decide. The bound is proved
in `gram_from_lower`'s docstring and `check_potrf_vs_oracle` asserts the
factor equals the PLANT bit for bit, in BOTH modes, which is only possible
because the whole path is exact.

**(2) THE INEXACT ONE.** `FIX_RBF` and `FIX_ILL` are built through
`identical_mul_add`, `identical_exp` and `ftz`, which under IDENTICAL are one
float32 arithmetic made of correctly-rounded basic operations and are
therefore the same bits on any IEEE host. Under FAST they are not, and no
cross-vendor claim is made under FAST anyway. Both are also hashed as
`chol.input` by the driver, so a host difference lands on stage 0 rather than
propagating silently.

Non-uniform and non-repeating per cell in every fixture, so a permutation of
rows or columns changes the answer (`uniform-test-data-hides-permutation`).

THE HASH IS A PER-LANE COPY AND THAT IS THE CONVENTION, NOT A NEW IDEA.
`chol_mix64` is the same three lines as `kde/checks/kde_fixture.mojo:18`,
`holtwinters/checks/hw_fixture.mojo:30` and `isolation_forest/checks/
if_fixture.mojo:32`, with the same splitmix64 constants, for the reason
`core/pinned_reduce.mojo` gives about its own duplication: the alternative is
a cross-lane import of another lane's fixture file, and cross-lane
dependencies on hot files are how two sessions collide. Four copies of one
hash is a debt and it is named here rather than hidden.
"""

from std.memory import bitcast

from checks.numerics import ftz, identical_exp, identical_mul_add


#: `A = L L^T` for a hand-checkable exact `L`. The factor is recoverable BIT
#: FOR BIT and equals the plant, in both modes, on every vendor.
comptime FIX_PLANTED = 0
#: An RBF Gram matrix over a small hashed point set -- the shape the Gaussian
#: process and kernel-ridge lanes will actually feed this.
comptime FIX_RBF = 1
#: Rank deficient by three: `B B^T` with `B` of `n - 3` columns. The fixture
#: that needs a ridge.
comptime FIX_ILL = 2
#: Exactly singular, and singular in a LATER PANEL: `L` with column 37 zeroed
#: entirely, so the pivot at 37 is exactly `+0.0` and `info` is exactly 38.
comptime FIX_SINGULAR = 3
#: Signed zeros in the strict lower triangle and one subnormal off-diagonal.
#: IDENTITY_PATHS row 39 territory.
comptime FIX_SIGNED_ZERO = 4
#: A SUBNORMAL diagonal entry, positive definite in exact arithmetic and
#: refused on every column here because the pivot is flushed first.
comptime FIX_DENORMAL_PIVOT = 5
comptime CHOL_FIXTURE_COUNT = 6


#: `2^-140` as a float32 SUBNORMAL, by its bits: mantissa `2^9 = 0x200`, all
#: exponent bits zero. Written this way and never as a decimal, because
#: `[[mojo-string-float-roundtrip]]` says a decimal cannot be trusted to come
#: back, and this particular value's whole job is to be a subnormal.
comptime CHOL_SUBNORMAL_BITS: UInt32 = 0x00000200

#: `2^-70`, the planted diagonal that squares to `CHOL_SUBNORMAL_BITS`.
#: Exponent field `127 - 70 = 57 = 0x39`, mantissa zero.
comptime CHOL_TINY_DIAG_BITS: UInt32 = 0x1C800000


def chol_mix64(a: Int, b: Int, salt: Int) -> UInt64:
    """splitmix64 over three integers. The lane's only source of randomness,
    and it is a pure function of its arguments on every machine."""
    var z = (
        UInt64(a + 1) * 0x9E3779B97F4A7C15
        + UInt64(b + 1) * 0xBF58476D1CE4E5B9
        + UInt64(salt + 1) * 0x94D049BB133111EB
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return z


def bits_value(z: UInt64, signed: Bool) -> Float32:
    """A float32 from bits: hashed mantissa, exponent 126 or 127 so `|v|` is
    in `[0.5, 2)`, sign from the hash when `signed`. `kde_fixture.mojo:30`'s
    function, and the RBF and rank-deficient fixtures' only value source."""
    var mant = UInt32(z & 0x7FFFFF)
    var expo = UInt32(126 + Int((z >> 23) & 1))
    var sign = UInt32(0)
    if signed and ((z >> 24) & 1) == 1:
        sign = UInt32(1)
    return bitcast[DType.float32]((sign << 31) | (expo << 23) | mant)


def exact_offdiag(z: UInt64) -> Float32:
    """One of `{-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75}`, hashed.

    **TWO FRACTIONAL BITS AND A MAGNITUDE BELOW ONE.** That pair is what
    makes `gram_from_lower` exact; see its docstring for the arithmetic. The
    value is built by integer arithmetic and one exact division by four, so
    no rounding happens here either.
    """
    var q = Int(z % 7) - 3
    return Float32(q) / Float32(4.0)


def exact_diag(z: UInt64) -> Float32:
    """`1.0` or `2.0`, hashed. Powers of two, so the panel's `sqrt(s)` (of
    `1.0` or `4.0`) and its `t / L[j][j]` are both exact."""
    if (z & 1) == 1:
        return Float32(2.0)
    return Float32(1.0)


def planted_lower(n: Int, salt: Int) -> List[Float32]:
    """A hand-checkable lower-triangular `L`, `n x n` row-major, strict upper
    triangle `+0.0`.

    Diagonal from `exact_diag` (a power of two), strict lower from
    `exact_offdiag` (a quarter-integer below one in magnitude). Every entry
    is a dyadic rational with at most two fractional bits and magnitude at
    most 2.
    """
    var out = List[Float32]()
    for i in range(n):
        for j in range(n):
            if j > i:
                out.append(Float32(0.0))
            elif j == i:
                out.append(exact_diag(chol_mix64(i, i, salt + 5)))
            else:
                out.append(exact_offdiag(chol_mix64(i, j, salt)))
    return out^


def gram_from_lower(l: List[Float32], n: Int) -> List[Float32]:
    """`A = L L^T`, host, ascending. **EXACT AT EVERY STEP, and here is why.**

    With `L` from `planted_lower`: every entry has at most 2 fractional bits
    and magnitude at most 2. A product `L[i][k] * L[j][k]` therefore has at
    most 4 fractional bits and magnitude at most 4, and is exact. The sum
    over at most `n` such products has at most 4 fractional bits and
    magnitude at most `4n`, so it needs `4 + ceil(log2(4n))` significant bits
    and float32 has 24. At `n = 48` that is `4 + 8 = 12`. Every partial sum
    is exact, so the result does not depend on the order they are added in,
    on whether the host contracts `a*b+c`, on its rounding mode or on its
    denormal policy -- there is nothing left for any of those to decide.

    **`n` MUST STAY UNDER 2^19 FOR THAT ARGUMENT TO HOLD**, which every
    fixture here does by four orders of magnitude. A caller who plants a
    bigger matrix has to redo the bound; the argument is written out rather
    than asserted so that redoing it is possible.

    No `ftz` and no `identical_mul_add` here, deliberately: an exact
    computation has no seam to flush and no contraction to pin, and spelling
    it with them would suggest the exactness came from the helpers rather
    than from the plant.

    The result is symmetric BY CONSTRUCTION and exactly so: cell `(i, j)` and
    cell `(j, i)` are the same sum over the same `k` with the two operands of
    each product exchanged, and float multiplication is exactly commutative.
    """
    var a = List[Float32]()
    for i in range(n):
        for j in range(n):
            var lim = i
            if j < lim:
                lim = j
            var acc = Float32(0.0)
            for k in range(lim + 1):
                acc = acc + l[i * n + k] * l[j * n + k]
            a.append(acc)
    return a^


def gram_from_rectangular(b: List[Float32], rows: Int, cols: Int) -> List[Float32]:
    """`A = B B^T` for a rectangular `B` (`rows x cols`, row-major).

    Construction (2): `identical_mul_add` and `ftz` at every step, ascending
    in `k`. Under IDENTICAL that is one float32 arithmetic made of
    correctly-rounded operations, so the same bits on any IEEE host; under
    FAST it is the host's own contraction and denormal policy and the fixture
    is not host-independent, which is stated rather than worked around.

    Exactly symmetric for the same commutativity reason `gram_from_lower`
    gives, and `cols < rows` makes it rank deficient by `rows - cols`.
    """
    var a = List[Float32]()
    for i in range(rows):
        for j in range(rows):
            var acc = Float32(0.0)
            for k in range(cols):
                acc = ftz(identical_mul_add(b[i * cols + k], b[j * cols + k], acc))
            a.append(acc)
    return a^


def rbf_gram(points: List[Float32], rows: Int, d: Int) -> List[Float32]:
    """`A[i][j] = exp(-|x_i - x_j|^2 / 2)` over a hashed point set.

    The shape the Gaussian-process and kernel-ridge lanes will feed this, so
    it is here rather than in those lanes: a fixture the callers share is a
    fixture the callers cannot disagree about.

    Lengthscale `1` and the squared distance halved by a multiply by `0.5`
    (exact, a power of two). `identical_exp` because a device `exp` is a
    VENDOR CHOICE in its last bit (IDENTITY_PATHS row 12) and the host is no
    better; `identical_mul_add` and `ftz` at every seam.

    The diagonal is `exp(-0) = exp(+0.0)`, which `portable_expf` returns as
    exactly `1.0`. So the matrix has a unit diagonal, is symmetric by the
    commutativity argument, and is positive definite for distinct points --
    but only MATHEMATICALLY: an RBF Gram matrix in float32 at 64 points is
    numerically close to singular, which is the entire reason DEVIATION
    1637's ridge exists and why the check jitters this fixture.
    """
    var a = List[Float32]()
    for i in range(rows):
        for j in range(rows):
            var acc = Float32(0.0)
            for k in range(d):
                var diff = ftz(
                    ftz(points[i * d + k]) - ftz(points[j * d + k])
                )
                acc = ftz(identical_mul_add(diff, diff, acc))
            var half = ftz(acc * Float32(0.5))
            a.append(ftz(identical_exp(-half)))
    return a^


def chol_fixture_n(which: Int) -> Int:
    """The dimension of each fixture. `FIX_PLANTED` and `FIX_SINGULAR` are 48
    so that at `CHOL_NB_PINNED = 32` the factorization walks TWO panels and
    performs one real trailing update -- a one-panel fixture cannot see
    DEVIATION 1630 at all."""
    if which == FIX_PLANTED:
        return 48
    if which == FIX_RBF:
        return 64
    if which == FIX_ILL:
        return 32
    if which == FIX_SINGULAR:
        return 48
    if which == FIX_SIGNED_ZERO:
        return 16
    if which == FIX_DENORMAL_PIVOT:
        return 40
    return 0


def chol_fixture_name(which: Int) -> String:
    if which == FIX_PLANTED:
        return String("PLANTED")
    if which == FIX_RBF:
        return String("RBF")
    if which == FIX_ILL:
        return String("ILL")
    if which == FIX_SINGULAR:
        return String("SINGULAR")
    if which == FIX_SIGNED_ZERO:
        return String("SIGNED_ZERO")
    if which == FIX_DENORMAL_PIVOT:
        return String("DENORMAL_PIVOT")
    return String("UNKNOWN")


#: The column `FIX_SINGULAR` zeroes. In panel 1 at `NB = 32`, so the pivot
#: fails AFTER a real trailing update rather than inside the first panel --
#: a fixture that fails at column 1 says nothing about the blocked path.
comptime FIX_SINGULAR_COL = 37

#: The row `FIX_DENORMAL_PIVOT` gives a subnormal diagonal to. Also in a
#: later panel.
comptime FIX_DENORMAL_ROW = 35

#: The two identity rows `FIX_SIGNED_ZERO` uses. Row `SZ_ZERO_ROW`'s
#: off-diagonal cells are the planted signed zeros; row `SZ_SUB_ROW` carries
#: the subnormal off-diagonal.
comptime FIX_SZ_ZERO_ROW = 9
comptime FIX_SZ_SUB_ROW = 3
comptime FIX_SZ_SUB_COL = 1


def chol_fixture(which: Int, salt: Int) raises -> List[Float32]:
    """The matrix, `n x n` row-major, symmetric, host-independent under
    IDENTICAL. `n` is `chol_fixture_n(which)`."""
    var n = chol_fixture_n(which)
    if n <= 0:
        raise Error(
            "chol_fixture: unknown fixture id " + String(which)
        )

    if which == FIX_PLANTED:
        var l = planted_lower(n, salt)
        return gram_from_lower(l, n)

    if which == FIX_RBF:
        var d = 5
        var pts = List[Float32]()
        for i in range(n):
            for k in range(d):
                pts.append(bits_value(chol_mix64(i, k, salt + 11), True))
        return rbf_gram(pts, n, d)

    if which == FIX_ILL:
        # Rank deficient by three. Mathematically singular, so the last
        # pivots are zero in exact arithmetic and whatever float32 rounding
        # makes of zero in practice -- which is the point, and which is why
        # the check asserts device == oracle rather than asserting a
        # particular `info`.
        var cols = n - 3
        var b = List[Float32]()
        for i in range(n):
            for k in range(cols):
                b.append(bits_value(chol_mix64(i, k, salt + 23), True))
        return gram_from_rectangular(b, n, cols)

    if which == FIX_SINGULAR:
        # Column FIX_SINGULAR_COL of L is entirely zero, DIAGONAL INCLUDED.
        # So row FIX_SINGULAR_COL of A is zero, the pivot there is exactly
        # `+0.0` (an exact sum of exact zeros minus an exact sum of exact
        # zeros), and `info` is exactly FIX_SINGULAR_COL + 1 = 38 on every
        # column and in both modes. Nothing about that number is a
        # measurement; it is arithmetic.
        var l = planted_lower(n, salt)
        for j in range(n):
            l[FIX_SINGULAR_COL * n + j] = Float32(0.0)
        for i in range(n):
            l[i * n + FIX_SINGULAR_COL] = Float32(0.0)
        return gram_from_lower(l, n)

    if which == FIX_DENORMAL_PIVOT:
        # Row FIX_DENORMAL_ROW of L is an IDENTITY ROW carrying `2^-70`, so
        # A's row and column there are zero off the diagonal and the
        # diagonal itself would be `2^-140`. The square is PLANTED BY BITS
        # rather than computed, because computing it through this file's
        # `ftz` seams would flush it to `+0.0` and the fixture would then
        # test the wrong thing -- the whole point is that the SUBNORMAL
        # reaches the device.
        var l = planted_lower(n, salt)
        for j in range(n):
            l[FIX_DENORMAL_ROW * n + j] = Float32(0.0)
        for i in range(n):
            l[i * n + FIX_DENORMAL_ROW] = Float32(0.0)
        var a = gram_from_lower(l, n)
        a[FIX_DENORMAL_ROW * n + FIX_DENORMAL_ROW] = bitcast[DType.float32](
            CHOL_SUBNORMAL_BITS
        )
        return a^

    # FIX_SIGNED_ZERO. Two identity rows, then the planted zeros.
    var l = planted_lower(n, salt)
    for j in range(n):
        if j != FIX_SZ_ZERO_ROW:
            l[FIX_SZ_ZERO_ROW * n + j] = Float32(0.0)
        if j != FIX_SZ_SUB_ROW:
            l[FIX_SZ_SUB_ROW * n + j] = Float32(0.0)
    for i in range(n):
        if i != FIX_SZ_ZERO_ROW:
            l[i * n + FIX_SZ_ZERO_ROW] = Float32(0.0)
        if i != FIX_SZ_SUB_ROW:
            l[i * n + FIX_SZ_SUB_ROW] = Float32(0.0)
    l[FIX_SZ_ZERO_ROW * n + FIX_SZ_ZERO_ROW] = Float32(1.0)
    l[FIX_SZ_SUB_ROW * n + FIX_SZ_SUB_ROW] = Float32(1.0)
    var a = gram_from_lower(l, n)
    # Row FIX_SZ_ZERO_ROW's off-diagonal cells alternate `-0.0` and `+0.0`,
    # planted in BOTH triangles so the symmetry check passes exactly. Their
    # signs reach the factor: the panel computes `t = ftz(A[r][c])` and then
    # adds products that are all zero, so `t` keeps whatever zero it started
    # from, modified only by the SIGNS of those zero products -- which is
    # deterministic, is the same on every column, and is exactly what
    # `check_signed_zero_and_denormal` compares device against oracle.
    for k in range(n):
        if k == FIX_SZ_ZERO_ROW:
            continue
        var z = Float32(0.0)
        if k % 2 == 0:
            z = Float32(-0.0)
        a[FIX_SZ_ZERO_ROW * n + k] = z
        a[k * n + FIX_SZ_ZERO_ROW] = z
    # One SUBNORMAL off-diagonal, in both triangles. It is flushed by the
    # first `ftz` the panel applies to it, on every column, which is the
    # property being planted.
    var sub = bitcast[DType.float32](CHOL_SUBNORMAL_BITS)
    a[FIX_SZ_SUB_ROW * n + FIX_SZ_SUB_COL] = sub
    a[FIX_SZ_SUB_COL * n + FIX_SZ_SUB_ROW] = sub
    return a^


def chol_rhs_fixture(n: Int, nrhs: Int, salt: Int) -> List[Float32]:
    """A planted SOLUTION `X`, `n x nrhs` row-major, from `exact_offdiag`.

    `check_cho_solve_residual` forms `B = A X` with `matvec_exact` and then
    requires `cho_solve(L, B)` to return `X` BIT FOR BIT. That is only a
    legitimate demand because both the forming and the solving are exact on
    the planted fixture -- see `matvec_exact` for the bound -- and it is a
    far stronger claim than a residual norm, which would pass on a solver
    that is merely close.
    """
    var out = List[Float32]()
    for i in range(n):
        for j in range(nrhs):
            out.append(exact_offdiag(chol_mix64(i, j, salt + 31)))
    return out^


def matvec_exact(
    a: List[Float32], x: List[Float32], n: Int, nrhs: Int
) -> List[Float32]:
    """`B = A X`, host, ascending, and EXACT on the planted fixture.

    The bound, continuing `gram_from_lower`'s: `A`'s entries there have at
    most 4 fractional bits and magnitude at most `4n`; `X`'s have 2
    fractional bits and magnitude below 1. A product has at most 6
    fractional bits and magnitude below `4n`, and a sum of `n` of them has at
    most 6 fractional bits and magnitude below `4n^2`, needing
    `6 + ceil(log2(4 n^2))` bits. At `n = 48` that is `6 + 14 = 20`, inside
    float32's 24.

    **THIS IS EXACT ONLY FOR THE PLANTED FIXTURE.** Called on `FIX_RBF` or
    `FIX_ILL` it is ordinary rounded arithmetic and the bit claim above does
    not hold; `check_cho_solve_residual` therefore makes its bitwise demand
    on `FIX_PLANTED` and a tolerance report elsewhere.
    """
    var b = List[Float32]()
    for i in range(n):
        for j in range(nrhs):
            var acc = Float32(0.0)
            for k in range(n):
                acc = acc + a[i * n + k] * x[k * nrhs + j]
            b.append(acc)
    return b^
