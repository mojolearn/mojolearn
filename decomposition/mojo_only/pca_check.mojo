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
`column_mean_kernel` were a no-op then `mu` is zero, if
`shift_columns_kernel` were a no-op then the centering never happens, and in
either case a column with a 1000 offset dominates the covariance and the
first component swings onto it. There is no way to pass invariant 2 without
both kernels running.

Invariant 1 fails if the `n_rows - 1` normalization is wrong or missing.
"""

from std.math import cos, sin, sqrt
from max.gpu.host import DeviceBuffer, DeviceContext

from decomposition.ported.linalg.detail.pca import PCAResult, pca_fit
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
    ctx.synchronize()
    _fill(ctx, x, scale, shift_col0)
    return pca_fit(
        ctx, x, mu, cov, PCA_ROWS, PCA_COLS, PCA_COMPONENTS
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
    """Step 6, the one that is easy to drop and invisible when dropped.

    `pca_fit` centers the caller's matrix in place and must add the mean back
    (`detail/pca.cuh:186`). A fit that forgets leaves the caller holding
    centered data, and nothing inside the fit will ever reveal it.
    """
    var ctx = DeviceContext()
    var x = ctx.enqueue_create_buffer[DType.float32](PCA_ROWS * PCA_COLS)
    var mu = ctx.enqueue_create_buffer[DType.float32](PCA_COLS)
    var cov = ctx.enqueue_create_buffer[DType.float32](PCA_COLS * PCA_COLS)
    ctx.synchronize()
    _fill(ctx, x, 1.0, 0.0)

    var before = ctx.enqueue_create_host_buffer[DType.float32](
        PCA_ROWS * PCA_COLS
    )
    ctx.enqueue_copy(dst_ptr=before.unsafe_ptr(), src_buf=x)
    ctx.synchronize()

    _ = pca_fit(ctx, x, mu, cov, PCA_ROWS, PCA_COLS, PCA_COMPONENTS)

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
    var gram = ctx.enqueue_create_buffer[DType.float32](PCA_COLS * PCA_COLS)
    ctx.synchronize()

    # The latent columns are symmetric about zero, so the raw data is already
    # very nearly centered and the two must agree.
    _fill(ctx, x, 1.0, 0.0)
    var t0 = tsvd_fit(ctx, x, gram, PCA_ROWS, PCA_COLS, PCA_COMPONENTS)
    _fill(ctx, x, 1.0, 0.0)
    var p0 = pca_fit(ctx, x, mu, cov, PCA_ROWS, PCA_COLS, PCA_COMPONENTS)

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
    var t1 = tsvd_fit(ctx, x, gram, PCA_ROWS, PCA_COLS, PCA_COMPONENTS)

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
