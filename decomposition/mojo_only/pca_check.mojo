"""Launch PCA, and test it with INVARIANTS rather than only with a fixture.

NO RAFT COUNTERPART. Same discipline as the other two sections.

THE FIXTURE HAS AN ANALYTIC ANSWER
----------------------------------
Latent columns `Z` are independent with KNOWN variances `[100, 50, 20, 5]`,
and the data is `X = Z A` for a known orthogonal `A` built from two Givens
rotations, in the (0,1) plane and the (2,3) plane.

Then `cov(X) = A^T diag(v) A` exactly, so the eigenvalues ARE the latent
variances and the eigenvectors ARE the rows of `A`. Both the values and the
directions are known in closed form, so the check tests PLACEMENT and not
just a total: a component in the wrong slot, or a correct spectrum attached
to wrong directions, both fail.

The rotation matters. Without it the principal axes would be the coordinate
axes, and an implementation that never rotated anything would still look
right.

TWO INVARIANTS, PREDICTING DIFFERENT THINGS
-------------------------------------------
Same paired-prediction design as the k-means and k-NN reach checks, except
these are properties of PCA rather than corruptions, which makes them
stronger: they hold for the true algorithm and fail for most broken ones.

1. **Scale the data by 3.** Covariance scales by 9, so the explained
   variances must scale by 9 and the components, being directions, must NOT
   move at all.
2. **Shift one column by +1000.** PCA centers, so NOTHING may change. Not the
   variances, not the directions.

Invariant 2 is the reach test for the centering path specifically. If
`column_mean_kernel` were a no-op then `mu` is zero; if the centered read
were a no-op -- the FUSED `x - mu[col]` tile load on the split-K arm
(DEVIATION 42), `shift_columns_kernel` on the fallback arm -- then the
centering never happens; and in either case a column with a 1000 offset
dominates the covariance and the first component swings onto it. There is
no way to pass invariant 2 without the mean and the centered read both
live on whichever arm this shape dispatches to.

Invariant 1 fails if the `n_rows - 1` normalization is wrong or missing.
"""

from std.math import cos, sin, sqrt
from max.gpu.host import DeviceBuffer, DeviceContext

from decomposition.ported.linalg.detail.pca import (
    PCAResult,
    compute_covariance,
    pca_fit,
)
from decomposition.ported.linalg.detail.tsvd import tsvd_fit


comptime PCA_ROWS = 8192
comptime PCA_COLS = 4
comptime PCA_COMPONENTS = 4
comptime THETA_A = 0.6
comptime THETA_B = 1.1


def _latent_sd(k: Int) -> Float64:
    """sqrt of the planted variance for latent column k."""
    if k == 0:
        return sqrt(100.0)
    if k == 1:
        return sqrt(50.0)
    if k == 2:
        return sqrt(20.0)
    return sqrt(5.0)


def _planted_var(k: Int) -> Float64:
    return _latent_sd(k) * _latent_sd(k)


def _rotation(k: Int, j: Int) -> Float64:
    """Row k, column j of the known orthogonal A."""
    var ca = cos(Float64(THETA_A))
    var sa = sin(Float64(THETA_A))
    var cb = cos(Float64(THETA_B))
    var sb = sin(Float64(THETA_B))
    if k == 0:
        if j == 0:
            return ca
        if j == 1:
            return sa
        return 0.0
    if k == 1:
        if j == 0:
            return -sa
        if j == 1:
            return ca
        return 0.0
    if k == 2:
        if j == 2:
            return cb
        if j == 3:
            return sb
        return 0.0
    if j == 2:
        return -sb
    if j == 3:
        return cb
    return 0.0


def _latent(row: Int, k: Int) -> Float64:
    """Uniform on [-a, a] with a chosen so the variance is the planted one.

    Uniform on [-a, a] has variance a^2 / 3, so a = sqrt(3 * v).
    """
    # splitmix64. The first version used `(row * C + k * D) % prime`, which
    # makes every latent column an ARITHMETIC PROGRESSION differing only by a
    # constant offset, so the four columns were near-perfectly dependent, the
    # covariance was not `diag(v)`, and the planted eigenvalues were not the
    # true ones. The check failed at 145 against a planted 100 and the port
    # was right. A fixture needs a real mixer, not a modular stride.
    var z = (
        UInt64(row) * 0x9E3779B97F4A7C15
        + UInt64(k + 1) * 0xBF58476D1CE4E5B9
        + 0x94D049BB133111EB
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    var u = Float64(z >> 11) * (1.0 / 9007199254740992.0) - 0.5
    var a = sqrt(3.0 * _planted_var(k))
    return 2.0 * a * u


def _fill(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    scale: Float64,
    shift_col0: Float64,
) raises:
    var hx = ctx.enqueue_create_host_buffer[DType.float32](
        PCA_ROWS * PCA_COLS
    )
    for i in range(PCA_ROWS):
        for j in range(PCA_COLS):
            var v = 0.0
            for k in range(PCA_COLS):
                v += _latent(i, k) * _rotation(k, j)
            v *= scale
            if j == 0:
                v += shift_col0
            hx.unsafe_ptr().unsafe_store(i * PCA_COLS + j, Float32(v))
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.synchronize()


def _fit(
    ctx: DeviceContext, scale: Float64, shift_col0: Float64
) raises -> PCAResult:
    var x = ctx.enqueue_create_buffer[DType.float32](PCA_ROWS * PCA_COLS)
    var mu = ctx.enqueue_create_buffer[DType.float32](PCA_COLS)
    var cov = ctx.enqueue_create_buffer[DType.float32](PCA_COLS * PCA_COLS)
    var x_alias = ctx.enqueue_create_buffer[DType.float32](PCA_ROWS * PCA_COLS)
    var x_alias2 = ctx.enqueue_create_buffer[DType.float32](PCA_ROWS * PCA_COLS)
    ctx.synchronize()
    _fill(ctx, x, scale, shift_col0)
    return pca_fit(
        ctx, x, x_alias, x_alias2, mu, cov, PCA_ROWS, PCA_COLS, PCA_COMPONENTS
    )


def check_pca_fit() raises:
    var ctx = DeviceContext()
    var r = _fit(ctx, 1.0, 0.0)

    # --- eigenvalues are the planted latent variances --------------------
    for c in range(PCA_COMPONENTS):
        var want = _planted_var(c)
        var rel = abs(r.explained_var[c] - want) / want
        if rel > 0.06:
            raise Error(
                "explained_var[" + String(c) + "] = "
                + String(r.explained_var[c]) + ", planted " + String(want)
                + ", relative " + String(rel)
            )

    # --- eigenvectors are the rows of A, in the SAME order ---------------
    # Checked as a full dot-product matrix so a permutation cannot pass.
    for c in range(PCA_COMPONENTS):
        for k in range(PCA_COLS):
            var dot = 0.0
            for j in range(PCA_COLS):
                dot += r.components[c * PCA_COLS + j] * _rotation(k, j)
            if c == k:
                if abs(dot) < 0.99:
                    raise Error(
                        "component " + String(c) + " is not aligned with"
                        " planted axis " + String(k) + ", |dot| = "
                        + String(abs(dot))
                    )
            else:
                if abs(dot) > 0.06:
                    raise Error(
                        "component " + String(c) + " leaks onto planted axis"
                        " " + String(k) + ", |dot| = " + String(abs(dot))
                    )

    # --- the sign convention is actually applied -------------------------
    for c in range(PCA_COMPONENTS):
        var biggest = 0.0
        for j in range(PCA_COLS):
            var v = r.components[c * PCA_COLS + j]
            if abs(v) > abs(biggest):
                biggest = v
        if biggest < 0.0:
            raise Error(
                "component " + String(c) + " has its largest-magnitude entry"
                " negative; sign_flip was not applied, so the output is not"
                " reproducible"
            )

    var ratio_sum = 0.0
    for c in range(PCA_COMPONENTS):
        ratio_sum += r.explained_var_ratio[c]
    if abs(ratio_sum - 1.0) > 1e-3:
        raise Error(
            "explained_var_ratio over all components sums to "
            + String(ratio_sum) + ", expected 1"
        )

    print(
        "check_pca_fit OK: 4/4 eigenvalues within 6% of planted"
        " [100, 50, 20, 5], 4/4 components aligned with the planted rotation"
        " and orthogonal to the others, sign convention applied, ratios sum"
        " to " + String(ratio_sum)
    )


def check_pca_invariants() raises:
    var ctx = DeviceContext()
    var base = _fit(ctx, 1.0, 0.0)

    # --- INVARIANT 1: scale by 3 -> variances x9, directions unchanged ---
    var scaled = _fit(ctx, 3.0, 0.0)
    for c in range(PCA_COMPONENTS):
        var want = base.explained_var[c] * 9.0
        var rel = abs(scaled.explained_var[c] - want) / want
        if rel > 0.02:
            raise Error(
                "INVARIANT 1 FAILED: scaling the data by 3 changed"
                " explained_var[" + String(c) + "] by a factor other than 9"
                " (relative error " + String(rel) + "). The n_rows - 1"
                " normalization is wrong."
            )
        var dot = 0.0
        for j in range(PCA_COLS):
            dot += (
                scaled.components[c * PCA_COLS + j]
                * base.components[c * PCA_COLS + j]
            )
        if abs(dot) < 0.999:
            raise Error(
                "INVARIANT 1 FAILED: scaling moved component " + String(c)
                + ", |dot| = " + String(abs(dot)) + ". A direction cannot"
                " depend on the units."
            )

    # --- INVARIANT 2: shift column 0 by 1000 -> NOTHING changes ----------
    var shifted = _fit(ctx, 1.0, 1000.0)
    for c in range(PCA_COMPONENTS):
        var rel = abs(
            shifted.explained_var[c] - base.explained_var[c]
        ) / base.explained_var[c]
        if rel > 0.02:
            raise Error(
                "INVARIANT 2 FAILED: shifting column 0 by 1000 changed"
                " explained_var[" + String(c) + "] (relative "
                + String(rel) + "). The data is NOT being centered, so"
                " column_mean_kernel or shift_columns_kernel is not reached."
            )
        var dot = 0.0
        for j in range(PCA_COLS):
            dot += (
                shifted.components[c * PCA_COLS + j]
                * base.components[c * PCA_COLS + j]
            )
        if abs(dot) < 0.999:
            raise Error(
                "INVARIANT 2 FAILED: shifting column 0 moved component "
                + String(c) + ", |dot| = " + String(abs(dot))
                + ". The centering path is not reached."
            )

    print(
        "check_pca_invariants OK: x3 scaling moved variances by exactly 9 and"
        " moved no direction; a +1000 column shift moved nothing at all,"
        " which is the reach evidence for the centering path"
    )


def check_input_restored() raises:
    """The caller's matrix must end the fit unchanged, on EVERY arm.

    On the fallback arm `pca_fit` centers the caller's matrix in place and
    must add the mean back (`pca.cuh:138`); a fit that forgets leaves the
    caller holding centered data, and nothing inside the fit will ever
    reveal it. On the split-K arm (which this 4-column shape takes) the
    centering is fused into the Gram read and x is never written at all
    (DEVIATION 42), so the worst element move here is exactly 0 rather
    than rounding-sized. `check_covariance_fused_and_fallback_restore`
    holds the fused arm to that bitwise standard and exercises the
    fallback arm's center + restore pair at a 129-column shape.
    """
    var ctx = DeviceContext()
    var x = ctx.enqueue_create_buffer[DType.float32](PCA_ROWS * PCA_COLS)
    var mu = ctx.enqueue_create_buffer[DType.float32](PCA_COLS)
    var cov = ctx.enqueue_create_buffer[DType.float32](PCA_COLS * PCA_COLS)
    var x_alias = ctx.enqueue_create_buffer[DType.float32](PCA_ROWS * PCA_COLS)
    var x_alias2 = ctx.enqueue_create_buffer[DType.float32](PCA_ROWS * PCA_COLS)
    ctx.synchronize()
    _fill(ctx, x, 1.0, 0.0)

    var before = ctx.enqueue_create_host_buffer[DType.float32](
        PCA_ROWS * PCA_COLS
    )
    ctx.enqueue_copy(dst_ptr=before.unsafe_ptr(), src_buf=x)
    ctx.synchronize()

    _ = pca_fit(ctx, x, x_alias, x_alias2, mu, cov, PCA_ROWS, PCA_COLS, PCA_COMPONENTS)

    var after = ctx.enqueue_create_host_buffer[DType.float32](
        PCA_ROWS * PCA_COLS
    )
    ctx.enqueue_copy(dst_ptr=after.unsafe_ptr(), src_buf=x)
    ctx.synchronize()

    var worst = Float32(0.0)
    for i in range(PCA_ROWS * PCA_COLS):
        var d = abs(
            after.unsafe_ptr().unsafe_load(i)
            - before.unsafe_ptr().unsafe_load(i)
        )
        if d > worst:
            worst = d
    if worst > Float32(0.05):
        raise Error(
            "the input was NOT restored: worst element moved by "
            + String(worst)
        )
    print(
        "check_input_restored OK: worst element moved " + String(worst)
        + " after a full fit"
    )


def check_covariance_fused_and_fallback_restore() raises:
    """Sentinels for BOTH `compute_covariance` arms, split by the SAME
    predicate the code uses (`gram_splitk_applies`), so this check cannot
    drift from the dispatch it watches.

    FUSED ARM (4 columns): x must be BIT-IDENTICAL after the call -- the
    arm's contract is a read-only X, so `!=` on every element, no
    tolerance, with `restore_input` at both values (the flag is dead
    there).

    FALLBACK ARM (129 columns, one past `GRAM_MAX_COLS`): with
    `restore_input=False` x MUST MOVE (the reach sentinel: the in-place
    center pass really ran -- a no-op center would leave x untouched and
    this check red), and with `restore_input=True` it must return to the
    original within rounding. Together with the move sentinel, "returns to
    original" is evidence of a real center + restore pair rather than of
    two no-ops.
    """
    from core.gram_splitk import gram_splitk_applies

    if not gram_splitk_applies(PCA_COLS, PCA_COLS, PCA_ROWS):
        raise Error(
            "fixture assumption broke: the 4-column shape no longer"
            " dispatches to split-K, so this check tests nothing"
        )
    var ctx = DeviceContext()
    var x = ctx.enqueue_create_buffer[DType.float32](PCA_ROWS * PCA_COLS)
    var mu = ctx.enqueue_create_buffer[DType.float32](PCA_COLS)
    var cov = ctx.enqueue_create_buffer[DType.float32](PCA_COLS * PCA_COLS)
    var x_alias = ctx.enqueue_create_buffer[DType.float32](
        PCA_ROWS * PCA_COLS
    )
    var x_alias2 = ctx.enqueue_create_buffer[DType.float32](
        PCA_ROWS * PCA_COLS
    )
    ctx.synchronize()
    _fill(ctx, x, 1.0, 7.5)

    var before = ctx.enqueue_create_host_buffer[DType.float32](
        PCA_ROWS * PCA_COLS
    )
    var after = ctx.enqueue_create_host_buffer[DType.float32](
        PCA_ROWS * PCA_COLS
    )
    ctx.enqueue_copy(dst_ptr=before.unsafe_ptr(), src_buf=x)
    ctx.synchronize()
    for pass_idx in range(2):
        compute_covariance(
            ctx, x, x_alias, x_alias2, mu, cov, PCA_ROWS, PCA_COLS,
            restore_input=(pass_idx == 1),
        )
        ctx.enqueue_copy(dst_ptr=after.unsafe_ptr(), src_buf=x)
        ctx.synchronize()
        for i in range(PCA_ROWS * PCA_COLS):
            if (
                after.unsafe_ptr().unsafe_load(i)
                != before.unsafe_ptr().unsafe_load(i)
            ):
                raise Error(
                    "FUSED arm modified x at element " + String(i)
                    + " (restore_input=" + String(pass_idx == 1)
                    + "); its contract is a read-only X, bitwise"
                )

    # FALLBACK ARM: one column past the split-K width cap.
    comptime FB_ROWS = 512
    comptime FB_COLS = 129
    if gram_splitk_applies(FB_COLS, FB_COLS, FB_ROWS):
        raise Error(
            "fixture assumption broke: 129 columns dispatches to split-K"
        )
    var fx = ctx.enqueue_create_buffer[DType.float32](FB_ROWS * FB_COLS)
    var fmu = ctx.enqueue_create_buffer[DType.float32](FB_COLS)
    var fcov = ctx.enqueue_create_buffer[DType.float32](FB_COLS * FB_COLS)
    var fa = ctx.enqueue_create_buffer[DType.float32](FB_ROWS * FB_COLS)
    var fa2 = ctx.enqueue_create_buffer[DType.float32](FB_ROWS * FB_COLS)
    var fh = ctx.enqueue_create_host_buffer[DType.float32](
        FB_ROWS * FB_COLS
    )
    var fh_after = ctx.enqueue_create_host_buffer[DType.float32](
        FB_ROWS * FB_COLS
    )
    ctx.synchronize()
    for i in range(FB_ROWS * FB_COLS):
        # Hashed, with a nonzero mean so the center pass has work to do.
        var z = (UInt64(i + 1) * 0x9E3779B97F4A7C15) >> 33
        fh.unsafe_ptr().unsafe_store(
            i, Float32(Int(z % 1000)) * Float32(0.01) + Float32(3.0)
        )
    ctx.enqueue_copy(dst_buf=fx, src_ptr=fh.unsafe_ptr())
    ctx.synchronize()

    compute_covariance(
        ctx, fx, fa, fa2, fmu, fcov, FB_ROWS, FB_COLS, restore_input=False
    )
    ctx.enqueue_copy(dst_ptr=fh_after.unsafe_ptr(), src_buf=fx)
    ctx.synchronize()
    var moved = 0
    for i in range(FB_ROWS * FB_COLS):
        if (
            fh_after.unsafe_ptr().unsafe_load(i)
            != fh.unsafe_ptr().unsafe_load(i)
        ):
            moved += 1
    if moved == 0:
        raise Error(
            "FALLBACK arm with restore_input=False left x untouched: the"
            " in-place center pass did not run, so the restore sentinel"
            " below would be vacuous"
        )

    # Restore by hand (x is currently centered), then run the restoring
    # variant end to end and demand it comes back within rounding.
    ctx.enqueue_copy(dst_buf=fx, src_ptr=fh.unsafe_ptr())
    ctx.synchronize()
    compute_covariance(
        ctx, fx, fa, fa2, fmu, fcov, FB_ROWS, FB_COLS, restore_input=True
    )
    ctx.enqueue_copy(dst_ptr=fh_after.unsafe_ptr(), src_buf=fx)
    ctx.synchronize()
    var worst = Float32(0.0)
    for i in range(FB_ROWS * FB_COLS):
        var d = abs(
            fh_after.unsafe_ptr().unsafe_load(i)
            - fh.unsafe_ptr().unsafe_load(i)
        )
        if d > worst:
            worst = d
    if worst > Float32(0.01):
        raise Error(
            "FALLBACK arm restore failed: worst element still off by "
            + String(worst)
        )
    print(
        "check_covariance_fused_and_fallback_restore OK: fused arm left x"
        " bit-identical under both restore_input values; fallback arm's"
        " center moved " + String(moved) + "/" + String(FB_ROWS * FB_COLS)
        + " elements and its restore brought the worst back to "
        + String(worst)
    )


def check_tsvd_against_pca() raises:
    """The two differ by CENTERING and by nothing else, so assert exactly that.

    On data whose columns already have zero mean, `X^T X = (n_rows - 1) * cov`,
    so truncated SVD and PCA must find the SAME directions. Shift a column and
    they must part company: PCA is unchanged, truncated SVD is not, because
    nothing centered it.

    Asserting both halves in one check is the point. Either alone would pass
    for an implementation that centered in the wrong place.
    """
    var ctx = DeviceContext()

    var x = ctx.enqueue_create_buffer[DType.float32](PCA_ROWS * PCA_COLS)
    var mu = ctx.enqueue_create_buffer[DType.float32](PCA_COLS)
    var cov = ctx.enqueue_create_buffer[DType.float32](PCA_COLS * PCA_COLS)
    var x_alias = ctx.enqueue_create_buffer[DType.float32](PCA_ROWS * PCA_COLS)
    var x_alias2 = ctx.enqueue_create_buffer[DType.float32](PCA_ROWS * PCA_COLS)
    var gram = ctx.enqueue_create_buffer[DType.float32](PCA_COLS * PCA_COLS)
    ctx.synchronize()

    # The latent columns are symmetric about zero, so the raw data is already
    # very nearly centered and the two must agree.
    _fill(ctx, x, 1.0, 0.0)
    var t0 = tsvd_fit(ctx, x, gram, x_alias, x_alias2, PCA_ROWS, PCA_COLS, PCA_COMPONENTS)
    _fill(ctx, x, 1.0, 0.0)
    var p0 = pca_fit(ctx, x, x_alias, x_alias2, mu, cov, PCA_ROWS, PCA_COLS, PCA_COMPONENTS)

    for c in range(PCA_COMPONENTS):
        var dot = 0.0
        for j in range(PCA_COLS):
            dot += t0.components[c * PCA_COLS + j] * p0.components[c * PCA_COLS + j]
        if abs(dot) < 0.99:
            raise Error(
                "on centered data tsvd and pca disagree on component "
                + String(c) + ", |dot| = " + String(abs(dot))
                + ". They share cal_eig, so they can only differ by the"
                " centering, and there is nothing to center here."
            )

    # Now shift. PCA must not move; tsvd must.
    _fill(ctx, x, 1.0, 1000.0)
    var t1 = tsvd_fit(ctx, x, gram, x_alias, x_alias2, PCA_ROWS, PCA_COLS, PCA_COMPONENTS)

    var moved = 0.0
    for j in range(PCA_COLS):
        moved += t1.components[j] * t0.components[j]
    if abs(moved) > 0.9:
        raise Error(
            "shifting a column by 1000 did NOT move the first truncated-SVD"
            " component (|dot| = " + String(abs(moved)) + "). tsvd_fit is"
            " centering, which is the one thing that would make it PCA."
        )

    print(
        "check_tsvd_against_pca OK: identical directions on centered data,"
        " and a +1000 shift moved tsvd's first component to |dot| = "
        + String(abs(moved)) + " while PCA's was unmoved"
    )


def check_covariance_is_symmetric() raises:
    """`X^T X` must be EXACTLY symmetric, and this is a tripwire not a formality.

    `PORTING.md 23` records what a transposed contraction looks like here: it
    does not produce an obviously wrong number, it produces a plausible and
    NON-SYMMETRIC matrix, and the symptom surfaced two files away as the
    Jacobi eigensolver running to its sweep limit and raising. That cost a
    debugging session, and a three-line assertion on `cov[i][j] ==
    cov[j][i]` would have found it in one run.

    Symmetry here is STRUCTURAL, not approximate. `(i, j)` and `(j, i)` are
    accumulated over the same rows in the same order by mirrored blocks, and
    float multiply is exactly commutative, so the two must agree BITWISE.
    Anything else means the tile indexing or the row split disagrees between
    the two halves. The test is `!=`, deliberately, with no tolerance.
    """
    var ctx = DeviceContext()
    var x = ctx.enqueue_create_buffer[DType.float32](PCA_ROWS * PCA_COLS)
    var x_alias = ctx.enqueue_create_buffer[DType.float32](PCA_ROWS * PCA_COLS)
    var x_alias2 = ctx.enqueue_create_buffer[DType.float32](
        PCA_ROWS * PCA_COLS
    )
    var mu = ctx.enqueue_create_buffer[DType.float32](PCA_COLS)
    var cov = ctx.enqueue_create_buffer[DType.float32](PCA_COLS * PCA_COLS)
    ctx.synchronize()
    _fill(ctx, x, 1.0, 0.0)

    compute_covariance(
        ctx, x, x_alias, x_alias2, mu, cov, PCA_ROWS, PCA_COLS
    )

    var h = ctx.enqueue_create_host_buffer[DType.float32](PCA_COLS * PCA_COLS)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=cov)
    ctx.synchronize()

    var asym = 0
    for i in range(PCA_COLS):
        for j in range(i + 1, PCA_COLS):
            var a = h.unsafe_ptr().unsafe_load(i * PCA_COLS + j)
            var b = h.unsafe_ptr().unsafe_load(j * PCA_COLS + i)
            if a != b:
                asym += 1
    if asym != 0:
        raise Error(
            String(asym) + " of " + String(PCA_COLS * (PCA_COLS - 1) // 2)
            + " off-diagonal pairs are not bitwise symmetric. A transposed"
            " contraction or a disagreeing row split produces exactly this,"
            " and its next symptom is the eigensolver failing to converge."
        )
    print(
        "check_covariance_is_symmetric OK: all "
        + String(PCA_COLS * (PCA_COLS - 1) // 2)
        + " off-diagonal pairs bitwise equal"
    )


# ---------------------------------------------------------------------------
# WIDE PCA: 64 and 128 features, which is where the section shipped a bug
# ---------------------------------------------------------------------------
#
# Every check above runs at `PCA_COLS = 4`. The device eigensolver used to cap
# `n_cols` at 32 and return a non-answer above it, and NOTHING HERE COULD SEE
# THAT, because nothing here ever asked for more than four features. A check
# suite whose widest case is four columns does not test a PCA, it tests a
# 4 x 4 PCA.
#
# So this part runs the real fit at 64 and at 128, and prints the result in a
# form `bench/pca_wide_sklearn.py` re-derives from the SAME fixture with
# `sklearn.decomposition.PCA`. sklearn is the ORACLE here and not a design
# source: the design is cuML's `pcaFit` (`cuml/cpp/src/pca/pca.cuh:104`), and
# sklearn only says whether the numbers are right.
#
# THE FIXTURE IS LOW RANK PLUS NOISE, ON PURPOSE. A matrix of independent
# columns has a DIAGONAL covariance, so its eigenvectors are the coordinate
# axes and an implementation that never rotated anything would pass. Here
# eight dense hashed loadings are mixed into every column, so the leading
# eigenvectors are dense rotations that only a working solver produces, and
# the trailing spectrum is deliberately degenerate at the noise floor, which
# is where the sign and permutation caveats in the comparison come from.

comptime WIDE_ROWS = 4096
comptime WIDE_RANK = 8
comptime WIDE_TOP = 8


def _wide_u01(row: Int, k: Int, salt: Int) -> Float64:
    """splitmix64 -> [0, 1). Mirrored exactly in `bench/pca_wide_sklearn.py`."""
    var z = (
        UInt64(row) * 0x9E3779B97F4A7C15
        + UInt64(k + 1) * 0xBF58476D1CE4E5B9
        + UInt64(salt + 1) * 0x94D049BB133111EB
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return Float64(z >> 11) * (1.0 / 9007199254740992.0)


def _wide_latent_sd(k: Int) -> Float64:
    """Eight well-separated signal strengths, then the noise floor."""
    var v = 200.0 - Float64(k) * 24.0
    return sqrt(v)


def _fill_wide(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
) raises:
    var hx = ctx.enqueue_create_host_buffer[DType.float32](n_rows * n_cols)
    for i in range(n_rows):
        for j in range(n_cols):
            var v = 0.0
            for k in range(WIDE_RANK):
                var z = (_wide_u01(i, k, 1) - 0.5) * 2.0 * sqrt(3.0)
                var b = (_wide_u01(k, j, 2) - 0.5) * 2.0
                v += z * _wide_latent_sd(k) * b
            v += (_wide_u01(i, j + 1000, 3) - 0.5) * 2.0 * sqrt(3.0)
            hx.unsafe_ptr().unsafe_store(i * n_cols + j, Float32(v))
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.synchronize()


def _fit_wide(
    ctx: DeviceContext, n_rows: Int, n_cols: Int, n_components: Int
) raises -> PCAResult:
    var x = ctx.enqueue_create_buffer[DType.float32](n_rows * n_cols)
    var mu = ctx.enqueue_create_buffer[DType.float32](n_cols)
    var cov = ctx.enqueue_create_buffer[DType.float32](n_cols * n_cols)
    var x_alias = ctx.enqueue_create_buffer[DType.float32](n_rows * n_cols)
    var x_alias2 = ctx.enqueue_create_buffer[DType.float32](n_rows * n_cols)
    ctx.synchronize()
    _fill_wide(ctx, x, n_rows, n_cols)
    return pca_fit(
        ctx, x, x_alias, x_alias2, mu, cov, n_rows, n_cols, n_components
    )


def _emit(name: String, xs: List[Float64], count: Int):
    var line = name
    for i in range(count):
        line += " " + String(xs[i])
    print(line)


def check_pca_wide() raises:
    """Run the real fit at 64 and 128 features and assert the properties that
    hold for ANY correct PCA, then print for the sklearn oracle.

    Asserted here, in Mojo, without needing Python:

    - the spectrum is DESCENDING (a permutation of the truth is a different
      answer, not a worse one, and the sort is the step cuSOLVER did for cuML);
    - `explained_variance_ratio_` sums to 1 over all components;
    - the eight planted signal variances are recovered to within 10 percent,
      so a solver that returned the noise floor everywhere fails;
    - every component is a UNIT vector, and the top eight are mutually
      orthogonal. This is the property that dies above a size cap.
    """
    var ctx = DeviceContext()
    var widths = List[Int]()
    widths.append(64)
    widths.append(128)

    for w in range(len(widths)):
        var n_cols = widths[w]
        var r = _fit_wide(ctx, WIDE_ROWS, n_cols, n_cols)

        for c in range(1, n_cols):
            if r.explained_var[c] > r.explained_var[c - 1] + 1e-6:
                raise Error(
                    "n_cols = " + String(n_cols) + ": explained_var is not"
                    " descending at " + String(c) + " ("
                    + String(r.explained_var[c - 1]) + " then "
                    + String(r.explained_var[c]) + ")"
                )

        var ratio_sum = 0.0
        for c in range(n_cols):
            ratio_sum += r.explained_var_ratio[c]
        if abs(ratio_sum - 1.0) > 1e-3:
            raise Error(
                "n_cols = " + String(n_cols) + ": explained_var_ratio sums to "
                + String(ratio_sum) + ", expected 1"
            )

        # Unit norm and mutual orthogonality of the leading components. Under
        # the old 32-feature cap the entries past 32 were never rotated, so a
        # component was not a unit vector at all.
        for c in range(WIDE_TOP):
            var nrm = 0.0
            for j in range(n_cols):
                nrm += r.components[c * n_cols + j] * r.components[c * n_cols + j]
            if abs(nrm - 1.0) > 1e-4:
                raise Error(
                    "n_cols = " + String(n_cols) + ": component " + String(c)
                    + " has squared norm " + String(nrm) + ", not 1"
                )
            for d in range(c + 1, WIDE_TOP):
                var dot = 0.0
                for j in range(n_cols):
                    dot += (
                        r.components[c * n_cols + j]
                        * r.components[d * n_cols + j]
                    )
                if abs(dot) > 1e-4:
                    raise Error(
                        "n_cols = " + String(n_cols) + ": components "
                        + String(c) + " and " + String(d) + " are not"
                        " orthogonal, dot = " + String(dot)
                    )

        # THE SPECTRUM HAS THE PLANTED SHAPE: a rank-8 signal standing well
        # clear of an isotropic noise floor of variance 1.
        #
        # This is asserted as the GAP and as the floor, not as eight absolute
        # numbers. The eight loading vectors are hashed and therefore NOT
        # mutually orthogonal, so the signal eigenvalues are the spectrum of a
        # Gram matrix of non-orthogonal directions and have no closed form.
        # An earlier version of this check predicted `sd[k]^2 * n_cols / 3`,
        # which is what they would be if the loadings WERE orthogonal, and it
        # failed at 787 against a predicted 1194. The port was right and the
        # prediction was wrong.
        if r.explained_var[WIDE_RANK - 1] < 20.0 * r.explained_var[WIDE_RANK]:
            raise Error(
                "n_cols = " + String(n_cols) + ": no rank-"
                + String(WIDE_RANK) + " gap in the spectrum. Component "
                + String(WIDE_RANK - 1) + " has variance "
                + String(r.explained_var[WIDE_RANK - 1]) + " and component "
                + String(WIDE_RANK) + " has "
                + String(r.explained_var[WIDE_RANK])
                + ". The planted signal is rank 8 over a noise floor of 1, so"
                " a solver that found it shows a cliff here and a solver that"
                " returned garbage does not."
            )
        var floor = r.explained_var[n_cols - 1]
        if floor < 0.5 or floor > 2.0:
            raise Error(
                "n_cols = " + String(n_cols) + ": the smallest eigenvalue is "
                + String(floor) + ", and the planted noise floor is variance"
                " 1. Anything else means the trailing components are not"
                " being solved."
            )

        print("WIDE n_cols=" + String(n_cols) + " n_rows=" + String(WIDE_ROWS))
        _emit("EV", r.explained_var, n_cols)
        _emit("EVR", r.explained_var_ratio, n_cols)
        _emit("SV", r.singular_vals, n_cols)
        for c in range(WIDE_TOP):
            var comp = List[Float64]()
            for j in range(n_cols):
                comp.append(r.components[c * n_cols + j])
            _emit("COMP " + String(c), comp, n_cols)
        print("NOISE " + String(r.noise_var))
        print(
            "  check_pca_wide n_cols = " + String(n_cols) + " OK: spectrum"
            " descending, ratios sum to " + String(ratio_sum) + ", top "
            + String(WIDE_TOP) + " components orthonormal, rank-"
            + String(WIDE_RANK) + " cliff present, noise floor "
            + String(r.explained_var[n_cols - 1])
        )

    print("check_pca_wide OK at 64 and 128 features")


def check_pca_truncation() raises:
    """Truncation and `noise_vars`, which nothing else in this file reaches.

    `truncCompExpVars` (`cuml/cpp/src/pca/pca.cuh:39`) computes the ratios
    over ALL `n_cols` eigenvalues and only then keeps the first
    `n_components`, so a truncated fit's `explained_variance_ratio_` sums to
    LESS than one and its entries must be identical to the corresponding
    entries of an untruncated fit. `noise_vars` is the MEAN OF THE DISCARDED
    eigenvalues, guarded by `n_components < n_cols && n_components < n_rows`.

    Both are checked against a full fit on the same fixture, so the assertion
    is on the DIFFERENCE and not on numbers that a broken truncation could
    also produce. This runs at 128 features, past the old cap.
    """
    var ctx = DeviceContext()
    var n_cols = 128
    var keep = 10
    var full = _fit_wide(ctx, WIDE_ROWS, n_cols, n_cols)
    var trunc = _fit_wide(ctx, WIDE_ROWS, n_cols, keep)

    for c in range(keep):
        var rel = abs(trunc.explained_var[c] - full.explained_var[c]) / full.explained_var[c]
        if rel > 1e-6:
            raise Error(
                "truncating to " + String(keep) + " changed explained_var["
                + String(c) + "] by " + String(rel) + " relative. Truncation"
                " selects, it does not recompute."
            )
        var dr = abs(trunc.explained_var_ratio[c] - full.explained_var_ratio[c])
        if dr > 1e-9:
            raise Error(
                "truncating changed explained_var_ratio[" + String(c) + "] by "
                + String(dr) + ". Theirs divides by the total over ALL"
                " n_cols eigenvalues before truncating, so it cannot move."
            )

    var want_noise = 0.0
    for c in range(keep, n_cols):
        want_noise += full.explained_var[c]
    want_noise /= Float64(n_cols - keep)
    var nrel = abs(trunc.noise_var - want_noise) / want_noise
    if nrel > 1e-5:
        raise Error(
            "noise_var came back " + String(trunc.noise_var) + ", and the"
            " mean of the " + String(n_cols - keep) + " discarded"
            " eigenvalues is " + String(want_noise) + " (relative "
            + String(nrel) + ")"
        )
    if full.noise_var != 0.0:
        raise Error(
            "an untruncated fit must report noise_var 0 and reported "
            + String(full.noise_var)
        )

    var s = 0.0
    for c in range(keep):
        s += trunc.explained_var_ratio[c]
    print(
        "check_pca_truncation OK at 128 features: top " + String(keep)
        + " variances and ratios identical to the full fit, ratios sum to "
        + String(s) + " (not 1, which is the point), noise_var "
        + String(trunc.noise_var) + " = mean of the " + String(n_cols - keep)
        + " discarded eigenvalues, and 0 when nothing is discarded"
    )
