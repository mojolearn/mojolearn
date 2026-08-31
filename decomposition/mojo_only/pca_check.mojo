# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
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

from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

from decomposition.ported.linalg.detail.pca import (
    PCAResult,
    compute_covariance,
    pca_fit,
)
from decomposition.ported.linalg.detail.tsvd import tsvd_fit


comptime _IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL

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

    # THE FUSED ARM IS NOT EVERY COLUMN'S ARM UNDER FAST, and this guard used
    # to treat that as a broken fixture (DEVIATION 528). It is not broken: by
    # IDENTITY_PATHS row 27 the split-K kernel is the APPLE column's FAST arm
    # and the BIT-IDENTICAL column's arm in both modes, so a FAST build
    # against the NVIDIA or AMD column correctly sends this shape to
    # `linalg.matmul` -- and then there is no fused centered read to compare,
    # because that arm centers in place through `shift_columns_kernel`.
    # Raising there reported a defect where the answer is "this column takes
    # the other arm", which is the same over-reading `check_gram_arm_is_pinned`
    # was rewritten for. Found by compiling this file with
    # `-D MOJOLEARN_COLUMN_AMD=1`, which is the cheapest cross-vendor
    # instrument in the tree.
    from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

    if not gram_splitk_applies(PCA_COLS, PCA_COLS, PCA_ROWS):
        comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
            raise Error(
                "check_covariance_fused_and_fallback_restore: under"
                " IDENTICAL the split-K kernel is the arm on EVERY column"
                " (DEVIATION 521), and this shape did not take it. That is"
                " a real dispatch defect, not a fixture assumption."
            )
        print(
            "check_covariance_fused_and_fallback_restore SKIPPED (FAST):"
            " this column sends the Gram shape to linalg.matmul, so there"
            " is no fused centered read on it to compare. Not a failure --"
            " IDENTITY_PATHS row 27. Under IDENTICAL every column takes the"
            " split-K arm and this check asserts."
        )
        return
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

    # THE FALLBACK ARM DOES NOT EXIST UNDER `IDENTICAL`, AND THAT IS ANOTHER
    # LANE'S PIN, NOT THIS ONE'S. `gemm_tn` refuses an over-capacity Gram
    # shape by name under IDENTICAL (DEVIATION 521): its only other arm is
    # `linalg.matmul`, whose k-split is a per-vendor summation order, so
    # serving 129 columns there would return a non-identical model under a
    # mode that promises one. Asserted rather than skipped -- a check that
    # quietly does nothing in one mode reads, in a log, exactly like a check
    # that passed.
    comptime if _IDENTICAL:
        var refusal = String("")
        var raised = False
        try:
            compute_covariance(
                ctx, fx, fa, fa2, fmu, fcov, FB_ROWS, FB_COLS,
                restore_input=False,
            )
        except e:
            raised = True
            refusal = String(e)
        # `raised` is a separate Bool ON PURPOSE and the obvious
        # `if refusal == ""` is not used. The first version of this check
        # tested the string, and in the IDENTICAL build it reported the call
        # as having COMPLETED while a debug print in the same block showed
        # `refusal` holding gemm_tn's full refusal text. Not reduced to a
        # minimal case (two standalone repros of the same shape behave), so
        # it is recorded here rather than named as a trap -- but a check that
        # reports the wrong branch is the one kind of check that is worse
        # than no check, so this one does not depend on that comparison.
        if not raised:
            raise Error(
                "the 129-column FALLBACK arm COMPLETED under IDENTICAL. It"
                " cannot have run on the split-K kernel at that width, so it"
                " ran on the vendor matmul and returned a model this mode"
                " promises is vendor-independent and is not."
            )
        if refusal.find("IDENTITY_PATHS row") < 0:
            raise Error(
                "the 129-column arm refused under IDENTICAL, but the message"
                " does not cite a ledger row. A refusal a user cannot trace"
                " to its reason is a crash. Got: " + refusal
            )
        print(
            "check_covariance_fused_and_fallback_restore OK [IDENTICAL]:"
            " fused arm left x bit-identical under both restore_input"
            " values; the 129-column fallback arm RAISED by name rather"
            " than falling through to the vendor matmul"
        )
        return

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

        # THE SIGN CONVENTION, ON EVERY COMPONENT (DEVIATION 525). Asserted
        # here as well as in `check_sign_flip_reaches_the_fit` because this
        # is the fit whose components go to `pca_wide_sklearn.py`, and
        # sklearn's `_fit_full` applies the SAME rule
        # (`svd_flip(U, Vt, u_based_decision=False)`, first-occurrence
        # argmax): where an eigenvalue is separated, our component and
        # sklearn's should agree in SIGN and not merely up to sign. Nothing
        # in the oracle currently asks that -- it compares |dot| -- so this
        # is the half of the claim that lives on this side.
        var sign_violations = 0
        for c in range(n_cols):
            var biggest = 0.0
            for j in range(n_cols):
                var m = abs(r.components[c * n_cols + j])
                if m > biggest:
                    biggest = m
            if biggest == 0.0:
                continue
            var first = 0
            for j in range(n_cols):
                if abs(r.components[c * n_cols + j]) == biggest:
                    first = j
                    break
            if r.components[c * n_cols + first] < 0.0:
                sign_violations += 1
        if sign_violations != 0:
            raise Error(
                "n_cols = " + String(n_cols) + ": " + String(sign_violations)
                + " components come back with their lowest-index"
                " largest-magnitude entry NEGATIVE. sign_flip_kernel is not"
                " reached, so every component's sign is the eigensolver's"
                " rounding and a vendor's last bit flips a whole component."
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
            + ", sign convention satisfied by all " + String(n_cols)
            + " components"
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


# ---------------------------------------------------------------------------
# THE SIGN CONVENTION (DEVIATION 525, IDENTITY_PATHS row 30)
# ---------------------------------------------------------------------------
#
# An eigenvector is defined up to sign, so SOMETHING has to pick one or the
# fit is not reproducible. `sign_flip_kernel` picks it, and everything below
# exists to hold that pick to the standard the ledger asks of a pinned
# pathway: the rule must be a pure function of the component's VALUES, its
# tie case must be TOTAL, and the fixture must actually contain the tie.
#
# WHY NOT CHECK THIS THROUGH `explained_variance_` OR A RECONSTRUCTION ERROR.
# Both are sign-INVARIANT. `X ~ scores @ components` is unchanged when a
# component and its score column both flip, and an eigenvalue does not know
# the sign of its eigenvector at all. A PCA suite can be entirely green on
# variance and reconstruction while every component's sign is decided by the
# eigensolver's last bit. So these checks assert on component VALUES, and on
# the bits of them.
#
# THREE THINGS ARE SEPARATED HERE, BECAUSE THEY FAIL DIFFERENTLY:
#
#   `check_sign_flip_rule_and_ties`      the RULE, on planted matrices where
#                                        the tie exists by construction
#   `check_sign_flip_matches_host_rule`  the DEVICE answer against a host
#                                        scan with no fold in it, bitwise
#   `check_sign_flip_reaches_the_fit`    the rule is applied on the path a
#                                        `pca_fit` caller actually takes

from std.memory import bitcast

from decomposition.ported.linalg.detail.pca import (
    SIGNFLIP_TPB,
    sign_flip_kernel,
)


comptime _NEG_ZERO_BITS = UInt32(0x80000000)
comptime _POS_ZERO_BITS = UInt32(0x00000000)
comptime _QNAN_BITS = UInt32(0x7FC00001)


def _bits_of(v: Float32) -> UInt32:
    return bitcast[DType.uint32](v)


def _float_of(b: UInt32) -> Float32:
    return bitcast[DType.float32](b)


def _hex8(b: UInt32) -> String:
    """`String(Float32)` does not round trip (PORTING trap), so every float
    printed by a failure below is printed as its BITS."""
    var digits = String("0123456789abcdef")
    var out = String("0x")
    for i in range(8):
        var nib = Int((b >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += digits[byte=nib]
    return out


def _sign_flip_roundtrip(
    ctx: DeviceContext, n: Int, bits: List[UInt32]
) raises -> List[UInt32]:
    """Launch `sign_flip_kernel` on the `n x n` matrix given by its float32
    BIT PATTERNS and return the bits that come back.

    Layout is the one the kernel documents and `jacobi_eigh_kernel` writes:
    entry `f` of component `c` lives at `f * n + c`.
    """
    var buf = ctx.enqueue_create_buffer[DType.float32](n * n)
    var h_in = ctx.enqueue_create_host_buffer[DType.float32](n * n)
    var h_out = ctx.enqueue_create_host_buffer[DType.float32](n * n)
    ctx.synchronize()

    var p_in = h_in.unsafe_ptr().unsafe_bitcast[UInt32]()
    for i in range(n * n):
        p_in.unsafe_store(i, bits[i])
    ctx.enqueue_copy(dst_buf=buf, src_ptr=h_in.unsafe_ptr())
    ctx.synchronize()

    ctx.enqueue_function[sign_flip_kernel](
        buf.unsafe_ptr(),
        Int32(n),
        grid_dim=(n, 1, 1),
        block_dim=(SIGNFLIP_TPB, 1, 1),
    )
    ctx.synchronize()

    # `buf` is READ here, after the launch and after the drain. Taking
    # `.unsafe_ptr()` above would otherwise have been its last use and Mojo
    # frees a buffer at its last use (PORTING trap), which on a bad day means
    # the kernel writes into freed memory.
    ctx.enqueue_copy(dst_ptr=h_out.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()

    var p_out = h_out.unsafe_ptr().unsafe_bitcast[UInt32]()
    var out = List[UInt32]()
    for i in range(n * n):
        out.append(p_out.unsafe_load(i))
    return out^


def _sign_flip_host_oracle(n: Int, bits: List[UInt32]) -> List[UInt32]:
    """The SAME rule, written as a sequential host scan with no reduction in
    it at all.

    This is the row-20 pattern: the device answer is required to equal a host
    answer whose control flow contains no cross-lane fold, so agreement is
    evidence that no fold decided anything. Deliberately spelled out rather
    than shared with the kernel -- an oracle that calls the code under test
    proves nothing.
    """
    var out = List[UInt32]()
    for i in range(n * n):
        out.append(bits[i])
    for c in range(n):
        # Pass 1, exactly the kernel's: `local` starts at +0.0 and only a
        # STRICTLY larger magnitude replaces it, so a NaN (which loses every
        # compare) can never become the maximum.
        var biggest = Float32(0.0)
        for f in range(n):
            var m = abs(_float_of(bits[f * n + c]))
            if m > biggest:
                biggest = m
        # Pass 2, the tie-break: the LOWEST index attaining it.
        var first = n
        for f in range(n):
            if abs(_float_of(bits[f * n + c])) == biggest:
                first = f
                break
        # Pass 3. `first == n` is the all-NaN column, where the maximum is the
        # +0.0 sentinel and no real entry attains it; the kernel leaves
        # `need_sign_flip` false there and so does this.
        var flip = False
        if first < n:
            if _float_of(bits[first * n + c]) < Float32(0.0):
                flip = True
        if flip:
            for f in range(n):
                out[f * n + c] = _bits_of(-_float_of(bits[f * n + c]))
    return out^


def _tie_multiplicity(n: Int, bits: List[UInt32], col: Int) -> Int:
    """How many entries of `col` attain the largest magnitude.

    A tie fixture that does not tie tests nothing, so every tie case below
    asserts this is at least 2 BEFORE asserting what the tie-break did.
    """
    var biggest = Float32(0.0)
    for f in range(n):
        var m = abs(_float_of(bits[f * n + col]))
        if m > biggest:
            biggest = m
    var hits = 0
    for f in range(n):
        if abs(_float_of(bits[f * n + col])) == biggest:
            hits += 1
    return hits


def _want_cell(
    cells: List[UInt32], n: Int, f: Int, c: Int, v: Float32
) raises:
    var got = cells[f * n + c]
    if got != _bits_of(v):
        raise Error(
            "sign rule: n = " + String(n) + " entry [" + String(f) + "]["
            + String(c) + "] came back " + _hex8(got) + ", expected "
            + _hex8(_bits_of(v))
        )


def _col_flipped(
    n: Int, before: List[UInt32], after: List[UInt32], c: Int
) raises -> Bool:
    """Did the whole component come back negated, BITWISE?

    Deliberately three-valued rather than two: a kernel that negated only
    part of a component is not a sign convention with a bug in it, it is a
    different kind of failure, and it raises here instead of being reported
    as "not flipped".
    """
    var neg = 0
    var same = 0
    for f in range(n):
        if after[f * n + c] == before[f * n + c]:
            same += 1
        elif after[f * n + c] == _bits_of(-_float_of(before[f * n + c])):
            neg += 1
    if same == n:
        return False
    if neg + same == n and neg > 0:
        return True
    raise Error(
        "component " + String(c) + " came back as neither itself nor its"
        " negation (" + String(same) + " unchanged, " + String(neg)
        + " negated, of " + String(n) + " entries). The kernel is not"
        " applying a SIGN to the whole component."
    )


def _put(mut bits: List[UInt32], n: Int, f: Int, c: Int, v: Float32):
    bits[f * n + c] = _bits_of(v)


def check_sign_flip_rule_and_ties() raises:
    """The rule, and the tie the rule has to have an answer for.

    scikit-learn's `svd_flip(U, Vt, u_based_decision=False)` -- the
    convention this port adopts, and the one cuML's `pcaFit` has none of --
    is "make the entry of largest ABSOLUTE value positive". That rule is
    INCOMPLETE on its own: two entries can share the largest magnitude with
    opposite signs, and then "the" largest entry does not exist. Resolved by
    iteration order, the sign of a whole component becomes a function of
    which lane got there first, which is the class of defect IDENTITY_PATHS
    exists to close.

    The completion, and it is part of the convention rather than an
    implementation detail: **the LOWEST INDEX attaining the maximum wins.**
    That is `cub::ArgMax`'s tie rule, it is cuML's own thrust loop
    (`tsvd.cuh:160`, `if (val > max)` -- first strict improvement), it is
    `np.argmax`'s, and therefore it is sklearn's. Four independent
    implementations of the same tie-break, which is why it is the one to pin.

    FIXTURES, and what each one separates:

      n = 4, col 0   plain negative maximum          -> flips
      n = 4, col 1   plain positive maximum          -> untouched, BITWISE
      n = 4, col 2   TIE at index 0 (-2) and 2 (+2)  -> lowest index is
                     negative, so the component flips. A highest-index
                     tie-break leaves it alone: the two rules return
                     component vectors that differ in EVERY sign.
      n = 4, col 3   the same tie with the signs swapped -> must NOT flip,
                     so the separation is two-sided and a rule that simply
                     always flipped a tied column would fail here.

      n = 64, col 0  tie at indices 3 and 4, which are DIFFERENT THREADS at
                     `SIGNFLIP_TPB = 32`. This is the one a cross-lane fold
                     could decide.
      n = 64, col 1  tie at indices 3 and 35 -- the SAME thread, two
                     iterations apart, so the strided loop's own order
                     decides it rather than the fold's.
      n = 64, col 2  that tie with the signs swapped.
      n = 64, col 3  a THREE-way tie (1, 33, 40), because a tie-break that
                     is total for two candidates can still be undefined for
                     three.
      n = 64, col 4  every entry zero, one of them -0.0.
      n = 64, col 5  every entry -0.0.
      n = 64, col 6  every entry NaN.
      n = 64, col 7  a single nonzero at index 63, negative.

    ON `-0.0`, WHICH IS IDENTITY_PATHS ROW 13'S HAZARD AND IS RULED ON HERE.
    Row 13 is the ET range fold, where `-0.0` and `+0.0` compare EQUAL so
    which one survived a float min/max was decided by fold ORDER, and the
    surviving sign then reached the model through a guard. The same two
    values reach this kernel's pass-1 maximum. **They cannot decide anything
    here**, and the reason is structural rather than lucky:

      1. pass 1 folds `abs(...)`, so its maximum is `+0.0` or `-0.0` and the
         two are consumed only by pass 2's `==`, which cannot tell them
         apart. Whichever one the fold happens to keep, pass 2 selects the
         same index.
      2. pass 3's test is `entry < 0.0`, and `-0.0 < 0.0` is FALSE. So a
         component whose largest-magnitude entry is a zero of EITHER sign is
         never flipped, and an all-zero component comes back bit for bit as
         it went in.

    That second point is a real bit-level statement and not a formality: the
    flip is a negation, and negating `+0.0` yields `-0.0`. Had the rule
    flipped on `-0.0`, an all-zero component would come back as a row of
    `0x80000000` instead of `0x00000000` -- different bits, from a fold
    order. Column 4 and column 5 assert exactly that, on the bits.

    Column 7 is the other side of the same coin, asserted rather than
    avoided: that component IS flipped, so its zeros DO become `-0.0`. The
    sign of a zero in a flipped component is decided by the rule, not by a
    race, and the check pins which one it is.

    ON NaN. A NaN loses every comparison, so `m > local` never admits one and
    `abs(NaN) == biggest` never matches: an all-NaN component has no defined
    maximum, the kernel's `need_sign_flip` stays false, and the column comes
    back untouched. Deterministic, and stated because "the maximum is
    attained by at least one real entry" is the one sentence about this
    kernel that is NOT true.
    """
    var ctx = DeviceContext()

    # ---------------- n = 4 ------------------------------------------------
    var n4 = 4
    var b4 = List[UInt32]()
    for _ in range(n4 * n4):
        b4.append(_POS_ZERO_BITS)
    # col 0: max |3| at index 1, negative -> flip
    _put(b4, n4, 0, 0, Float32(0.5))
    _put(b4, n4, 1, 0, Float32(-3.0))
    _put(b4, n4, 2, 0, Float32(1.0))
    _put(b4, n4, 3, 0, Float32(2.0))
    # col 1: same magnitudes, max positive -> untouched
    _put(b4, n4, 0, 1, Float32(0.5))
    _put(b4, n4, 1, 1, Float32(3.0))
    _put(b4, n4, 2, 1, Float32(1.0))
    _put(b4, n4, 3, 1, Float32(2.0))
    # col 2: TIE at 0 and 2, lowest index negative -> flip
    _put(b4, n4, 0, 2, Float32(-2.0))
    _put(b4, n4, 1, 2, Float32(1.0))
    _put(b4, n4, 2, 2, Float32(2.0))
    _put(b4, n4, 3, 2, Float32(0.5))
    # col 3: TIE at 0 and 2, lowest index positive -> no flip
    _put(b4, n4, 0, 3, Float32(2.0))
    _put(b4, n4, 1, 3, Float32(1.0))
    _put(b4, n4, 2, 3, Float32(-2.0))
    _put(b4, n4, 3, 3, Float32(0.5))

    for c in range(2, 4):
        var hits = _tie_multiplicity(n4, b4, c)
        if hits < 2:
            raise Error(
                "FIXTURE IS VACUOUS: n = 4 column " + String(c) + " has"
                " only " + String(hits) + " entry at the largest magnitude,"
                " so the tie this check is about does not occur in it"
            )

    var out4 = _sign_flip_roundtrip(ctx, n4, b4)

    _want_cell(out4, n4, 0, 0, Float32(-0.5))
    _want_cell(out4, n4, 1, 0, Float32(3.0))
    _want_cell(out4, n4, 2, 0, Float32(-1.0))
    _want_cell(out4, n4, 3, 0, Float32(-2.0))
    for f in range(n4):
        if out4[f * n4 + 1] != b4[f * n4 + 1]:
            raise Error(
                "sign rule: a component whose largest entry is already"
                " positive was MODIFIED at index " + String(f) + ": "
                + _hex8(b4[f * n4 + 1]) + " -> " + _hex8(out4[f * n4 + 1])
            )
    # The tie, both ways round. This pair is the separator: swap the
    # tie-break to highest-index and BOTH of these invert.
    _want_cell(out4, n4, 0, 2, Float32(2.0))
    _want_cell(out4, n4, 1, 2, Float32(-1.0))
    _want_cell(out4, n4, 2, 2, Float32(-2.0))
    _want_cell(out4, n4, 3, 2, Float32(-0.5))
    for f in range(n4):
        if out4[f * n4 + 3] != b4[f * n4 + 3]:
            raise Error(
                "sign rule: the tied component whose LOWEST-index maximum is"
                " positive was flipped anyway at index " + String(f)
                + ". The tie-break is not lowest-index."
            )

    # ---------------- n = 64 ----------------------------------------------
    var n = 64
    var b = List[UInt32]()
    for _ in range(n * n):
        b.append(_POS_ZERO_BITS)

    # Background small values so the planted maxima really are maxima.
    for c in range(8, n):
        for f in range(n):
            var z = (UInt64(f + 1) * 0x9E3779B97F4A7C15
                     + UInt64(c + 1) * 0xBF58476D1CE4E5B9)
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) * 0x94D049BB133111EB
            z = z ^ (z >> 31)
            var u = Float64(z >> 11) * (1.0 / 9007199254740992.0) - 0.5
            _put(b, n, f, c, Float32(u * 4.0))

    for c in range(0, 4):
        for f in range(n):
            _put(b, n, f, c, Float32(0.25))
    # col 0: tie at 3 and 4 -- DIFFERENT threads at stride 32.
    _put(b, n, 3, 0, Float32(-5.0))
    _put(b, n, 4, 0, Float32(5.0))
    # col 1: tie at 3 and 35 -- the SAME thread, two iterations apart.
    _put(b, n, 3, 1, Float32(-5.0))
    _put(b, n, 35, 1, Float32(5.0))
    # col 2: that tie with the signs swapped.
    _put(b, n, 3, 2, Float32(5.0))
    _put(b, n, 35, 2, Float32(-5.0))
    # col 3: a THREE-way tie, lowest index positive.
    _put(b, n, 1, 3, Float32(7.0))
    _put(b, n, 33, 3, Float32(-7.0))
    _put(b, n, 40, 3, Float32(-7.0))
    # col 4: all zero, one of them -0.0.
    for f in range(n):
        b[f * n + 4] = _POS_ZERO_BITS
    b[2 * n + 4] = _NEG_ZERO_BITS
    # col 5: all -0.0.
    for f in range(n):
        b[f * n + 5] = _NEG_ZERO_BITS
    # col 6: all NaN.
    for f in range(n):
        b[f * n + 6] = _QNAN_BITS
    # col 7: one nonzero, negative, at the far end of the strided loop.
    for f in range(n):
        b[f * n + 7] = _POS_ZERO_BITS
    _put(b, n, 63, 7, Float32(-1.0))

    for c in range(0, 4):
        var hits = _tie_multiplicity(n, b, c)
        var want_hits = 3 if c == 3 else 2
        if hits != want_hits:
            raise Error(
                "FIXTURE IS VACUOUS: n = 64 column " + String(c) + " has "
                + String(hits) + " entries at the largest magnitude and the"
                " fixture plants " + String(want_hits) + ". The tie this"
                " column exists to exercise is not there."
            )

    var out = _sign_flip_roundtrip(ctx, n, b)

    if not _col_flipped(n, b, out, 0):
        raise Error(
            "TIE ACROSS THREADS: n = 64 column 0 ties at indices 3 (-5) and"
            " 4 (+5), which land on different lanes at SIGNFLIP_TPB = 32."
            " Lowest index wins, index 3 is negative, so the component must"
            " flip. It did not, which means the tie was resolved by whichever"
            " lane reached it first."
        )
    if not _col_flipped(n, b, out, 1):
        raise Error(
            "TIE WITHIN ONE THREAD: n = 64 column 1 ties at 3 (-5) and 35"
            " (+5), both owned by lane 3. Lowest index wins, so it must flip."
        )
    if _col_flipped(n, b, out, 2):
        raise Error(
            "n = 64 column 2 ties at 3 (+5) and 35 (-5) and must NOT flip."
            " The tie-break is picking the HIGHEST index, or the last writer."
        )
    if _col_flipped(n, b, out, 3):
        raise Error(
            "n = 64 column 3 is a THREE-way tie at 1 (+7), 33 (-7), 40 (-7)."
            " The lowest index is positive, so it must not flip. A tie-break"
            " that is total for two candidates but not three fails here."
        )

    # -0.0 / +0.0, on the bits.
    for f in range(n):
        var want = b[f * n + 4]
        if out[f * n + 4] != want:
            raise Error(
                "ZERO COLUMN: entry " + String(f) + " of the all-zero"
                " component came back " + _hex8(out[f * n + 4])
                + ", expected " + _hex8(want) + ". A component whose"
                " largest-magnitude entry is a zero must not be flipped:"
                " -0.0 < 0.0 is FALSE and the whole column is therefore"
                " untouched, bit for bit. IDENTITY_PATHS row 13 is the"
                " precedent -- there a surviving -0.0 reached the model."
            )
    for f in range(n):
        if out[f * n + 5] != _NEG_ZERO_BITS:
            raise Error(
                "ALL -0.0 COLUMN: entry " + String(f) + " came back "
                + _hex8(out[f * n + 5]) + ", expected 0x80000000. Flipping"
                " this column would turn every -0.0 into +0.0, which is a"
                " BIT change decided by nothing but the sign of a zero."
            )
    for f in range(n):
        if out[f * n + 6] != _QNAN_BITS:
            raise Error(
                "ALL-NaN COLUMN: entry " + String(f) + " came back "
                + _hex8(out[f * n + 6]) + ", expected " + _hex8(_QNAN_BITS)
                + ". A NaN loses every compare, so no entry attains the"
                " maximum and the column has no defined sign; it must be"
                " left alone rather than flipped on a sentinel."
            )
    if out[63 * n + 7] != _bits_of(Float32(1.0)):
        raise Error(
            "the single nonzero entry at index 63 (past the first pass of"
            " the strided loop) was not flipped: "
            + _hex8(out[63 * n + 7])
        )
    for f in range(63):
        if out[f * n + 7] != _NEG_ZERO_BITS:
            raise Error(
                "component 7 WAS flipped, so its +0.0 entries must come back"
                " as -0.0 (negation is a bit operation on a zero). Entry "
                + String(f) + " came back " + _hex8(out[f * n + 7])
            )

    # IDEMPOTENCE. Applying a sign convention twice must equal applying it
    # once, bitwise. That is the mathematical characterisation of "the sign
    # is a function of the values", and it is the property that dies first if
    # anything in the kernel reads state rather than the component.
    var out2 = _sign_flip_roundtrip(ctx, n, out)
    for i in range(n * n):
        if out2[i] != out[i] and not (
            out[i] == _QNAN_BITS and out2[i] == _QNAN_BITS
        ):
            raise Error(
                "NOT IDEMPOTENT at cell " + String(i) + ": " + _hex8(out[i])
                + " -> " + _hex8(out2[i]) + ". The sign is not a pure"
                " function of the component's values."
            )

    print(
        "check_sign_flip_rule_and_ties OK: largest-magnitude entry positive,"
        " ties broken by LOWEST INDEX and separated four ways (across lanes,"
        " within one lane, sign-swapped, three-way); a zero maximum never"
        " flips so an all-zero or all--0.0 component is bit-identical after;"
        " an all-NaN component has no maximum and is left alone; a flipped"
        " component's zeros become -0.0 and that is pinned too; applying the"
        " rule twice is bitwise applying it once"
    )


def check_sign_flip_matches_host_rule() raises:
    """The DEVICE answer against a host scan with no reduction in it.

    `sign_flip_kernel` reaches its answer through two block collectives --
    `block_max` over the magnitudes and `block_min` over the indices that
    attain it. IDENTITY_PATHS row 20 is about exactly this shape of call and
    records why it is dangerous: `max.gpu.primitives.block.sum` folds across
    lanes at the HARDWARE width, 32 on Apple and NVIDIA and 64 on AMD's
    wavefront, so the fold's SHAPE is a per-vendor number.

    **A fold shape cannot move a min or a max, and that is the whole
    argument for why this pathway needs no `IDENTICAL` arm.** A sum is
    non-associative in floating point, which is why row 20 had to replace
    one. A max is a SELECTION over a total order: it is associative and
    commutative exactly, on any tree, at any width, so 32-wide and 64-wide
    folds return the same bits. The two escapes from that total order are
    `-0.0` vs `+0.0` (equal under `<`, so a fold picks between them) and
    NaN (unordered, so a fold's answer depends on which side it is on), and
    both are neutralised in this kernel rather than avoided: the magnitude
    fold sees only `abs(...)` so its zeros are consumed by an `==` that
    cannot tell them apart, and a NaN never enters `local` because
    `m > local` rejects it.

    An argument is not evidence, so this check demands the device equal a
    host scan that has no fold at all, BITWISE, over five widths and 8256
    cells -- including the widths where the strided loop's iteration count
    changes (n < TPB, n == TPB, n == TPB + 1) and columns with planted ties.
    On a 64-lane column the same check becomes a genuine cross-vendor
    comparison with no edit, because the host side does not move.
    """
    var ctx = DeviceContext()
    var widths = List[Int]()
    widths.append(4)
    widths.append(31)
    widths.append(32)
    widths.append(33)
    widths.append(64)

    var cells = 0
    var flipped_total = 0
    var tied_cols = 0
    for w in range(len(widths)):
        var n = widths[w]
        var bits = List[UInt32]()
        for i in range(n * n):
            var z = (UInt64(i + 1) * 0x9E3779B97F4A7C15
                     + UInt64(n + 7) * 0xBF58476D1CE4E5B9
                     + 0x94D049BB133111EB)
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) * 0x94D049BB133111EB
            z = z ^ (z >> 31)
            var u = Float64(z >> 11) * (1.0 / 9007199254740992.0) - 0.5
            bits.append(_bits_of(Float32(u * 2.0)))
        # Plant an exact tie in every third column, at two indices chosen so
        # that some ties straddle a lane boundary and some do not. Hashed
        # data alone essentially never ties, so without this the widths below
        # would exercise the ordinary path only.
        var c = 0
        while c < n:
            var i0 = c % n
            var i1 = (c + 1 + (c % 5)) % n
            if i1 != i0:
                var mag = Float32(9.0 + Float32(c))
                bits[i0 * n + c] = _bits_of(
                    -mag if (c % 2) == 0 else mag
                )
                bits[i1 * n + c] = _bits_of(
                    mag if (c % 2) == 0 else -mag
                )
                if _tie_multiplicity(n, bits, c) >= 2:
                    tied_cols += 1
            c += 3

        var got = _sign_flip_roundtrip(ctx, n, bits)
        var want = _sign_flip_host_oracle(n, bits)
        for i in range(n * n):
            if got[i] != want[i]:
                raise Error(
                    "n = " + String(n) + " cell " + String(i) + ": device "
                    + _hex8(got[i]) + ", host scan " + _hex8(want[i])
                    + ". The block fold decided something the value-only"
                    " rule does not."
                )
        for cc in range(n):
            if want[cc] != bits[cc] or want[n + cc] != bits[n + cc]:
                flipped_total += 1
        cells += n * n

    if tied_cols == 0:
        raise Error(
            "FIXTURE IS VACUOUS: not one column carries an exact tie, so this"
            " check never exercised the tie-break it is here to hold"
        )
    if flipped_total == 0:
        raise Error(
            "FIXTURE IS VACUOUS: not one component was flipped across "
            + String(cells) + " cells, so agreeing with the host oracle"
            " proves only that both did nothing"
        )
    print(
        "check_sign_flip_matches_host_rule OK: device == a fold-free host"
        " scan, BITWISE, on " + String(cells) + " cells at n = 4/31/32/33/64"
        " (across, at and past the 32-wide stride), with " + String(tied_cols)
        + " columns carrying an exact tie and at least " + String(flipped_total)
        + " components actually flipped. A min or a max is a selection, so"
        " the fold's WIDTH -- 32 on Apple and NVIDIA, 64 on AMD -- cannot"
        " move this answer the way IDENTITY_PATHS row 20's sums move"
    )


comptime SIGN_FIT_ROWS = 1024
comptime SIGN_FIT_COLS = 48


def check_sign_flip_reaches_the_fit() raises:
    """The rule is applied on the path `pca_fit`'s caller actually takes.

    The two checks above hold `sign_flip_kernel`. This one holds `pca_fit`,
    and it is a different question: a kernel that is correct and not launched
    is IDENTITY_PATHS' "reached but inert" class, and `eig_and_truncate` is
    shared, so this covers `tsvd_fit` at the same time.

    WHY 48 COMPONENTS AND NOT 4. Every component's sign is decided
    independently, so a fixture with `k` components separates from a
    no-flip build with probability `1 - 2^-k` -- at four components that is
    one run in sixteen coming back green with the flip removed, which is not
    a reach test, it is a coin. At 48 it is one in 2.8e14. The 4-column
    fixture in `check_pca_fit` asserts the same convention and is kept, but
    THIS is the one that separates.

    It asserts the RULE, re-derived from the returned values, and never an
    expected component. The eigenvector bits themselves move when the
    eigensolver's folds are pinned, and a check that hardcoded them would
    break on a change that is not about signs at all.
    """
    var ctx = DeviceContext()
    var r = _fit_wide(ctx, SIGN_FIT_ROWS, SIGN_FIT_COLS, SIGN_FIT_COLS)

    var violations = 0
    var first_bad = -1
    var nonzero_components = 0
    for c in range(SIGN_FIT_COLS):
        var biggest = 0.0
        for j in range(SIGN_FIT_COLS):
            var m = abs(r.components[c * SIGN_FIT_COLS + j])
            if m > biggest:
                biggest = m
        if biggest == 0.0:
            continue
        nonzero_components += 1
        var first = -1
        for j in range(SIGN_FIT_COLS):
            if abs(r.components[c * SIGN_FIT_COLS + j]) == biggest:
                first = j
                break
        if r.components[c * SIGN_FIT_COLS + first] < 0.0:
            violations += 1
            if first_bad < 0:
                first_bad = c

    if nonzero_components != SIGN_FIT_COLS:
        raise Error(
            "FIXTURE IS VACUOUS: only " + String(nonzero_components) + " of "
            + String(SIGN_FIT_COLS) + " components are nonzero, so the sign"
            " rule had nothing to decide for the rest"
        )
    if violations != 0:
        raise Error(
            String(violations) + " of " + String(SIGN_FIT_COLS)
            + " components come back from pca_fit with their"
            " lowest-index largest-magnitude entry NEGATIVE (first at"
            " component " + String(first_bad) + "). sign_flip_kernel is not"
            " reached on the fit path, so every component's sign is whatever"
            " the eigensolver's rounding produced -- and a last bit that"
            " differs between two vendors flips a whole component."
        )
    print(
        "check_sign_flip_reaches_the_fit OK: all " + String(SIGN_FIT_COLS)
        + " components out of a real pca_fit at " + String(SIGN_FIT_COLS)
        + " features satisfy the convention (lowest-index"
        " largest-magnitude entry positive). Removing the flip separates"
        " with probability 1 - 2^-" + String(SIGN_FIT_COLS)
    )


def main() raises:
    """Everything in this file, so it runs as one process in one numeric mode.

    `GLOBAL_NUMERIC_MODE` is comptime, so one process is one mode; the wide
    and truncation checks keep their own entry point in
    `decomposition/pca_wide_main.mojo` because their output feeds the sklearn
    oracle and is parsed there.
    """
    check_covariance_is_symmetric()
    check_covariance_fused_and_fallback_restore()
    check_pca_fit()
    check_pca_invariants()
    check_input_restored()
    check_tsvd_against_pca()
    check_sign_flip_rule_and_ties()
    check_sign_flip_matches_host_rule()
    check_sign_flip_reaches_the_fit()
