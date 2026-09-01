# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Launch OLS against a planted linear model.

THE FIXTURE HAS AN EXACT ANSWER
-------------------------------
`y = X w*` with a known `w*` and NO noise, so the least-squares solution is
`w*` exactly, up to conditioning. `X` is uniform random in 8 dimensions, so
`A^T A` is well conditioned and the normal-equations route has nothing to
struggle with.

A second fixture adds noise. There the recovered `w` must be CLOSE to `w*`
but not equal, and the residual must be smaller than the residual of `w*`
itself: least squares fits the noise slightly better than the truth does, by
construction. That second assertion is the one that catches a solver which
merely returns something plausible.

THE REACH TEST IS AN INVARIANT
------------------------------
Scale `y` by 5. Every coefficient must scale by exactly 5 and nothing else
may change, because least squares is linear in the target. A solver that
ignored `b` entirely, or that normalized it away, fails this and a
fixed-fixture check would not notice.

AND NONE OF THAT IS AN IDENTITY CHECK
-------------------------------------
The three checks above are tolerance checks -- 1%, 1%, and a residual
comparison at 1.0001 -- and every one of them passes on a build whose
summation order is a different number on every vendor. A tolerance cannot
see a summation order at all. DEVIATION 527's checks, in the second half of
this file, are bitwise or discrete instead, and each asserts something
different in `NUMERIC_FAST` and in `NUMERIC_IDENTICAL`. See the banner
above `check_ols_arms_are_pinned`.

THE 2026-09-01 CHECKS: THE TWO SPECIAL SHAPES AND SAMPLE WEIGHTS
-----------------------------------------------------------------
`ols.cuh:112-113`'s two shapes used to REFUSE and the check here used to
assert the refusal. Both are solved now (DEVIATIONS 550 and 551) and
`sample_weight` is ported (`ols.cuh:99-110`, `:129-141`), so five checks
were added and every one of them is a PROPERTY that shares no spelling with
the implementation:

    check_ols_dispatch_routes_special_shapes    shapes only: both complete,
                                       an ordinary shape is not diverted,
                                       an EXPLICIT algo 0 still refuses
    check_ols_wide_is_the_minimum_norm_solution
                                       6 x 16: it interpolates, its norm is
                                       below the planted solution's, and it
                                       agrees with a float64 Gaussian
                                       elimination oracle
    check_ols_single_column_matches_the_closed_form
                                       n x 1 against sum(ab)/sum(aa) in
                                       float64 -- a closed form, not an
                                       approximation of one
    check_ols_normal_equation_residual_is_zero
                                       the tall route is a STATIONARY POINT:
                                       A^T (A w - b) ~ 0 in float64
    check_ols_duplicating_a_row_equals_doubling_its_weight
                                       the weight property: one objective,
                                       two spellings of it
    check_ols_sample_weight_restores_its_operands
                                       A, b and w come back (`:129-141`)
    check_ols_sample_weight_host_rescale_matches_device
                                       the gate the PYTHON surface rests on

**EVERY ONE OF THEM CARRIES ITS OWN NEGATIVE CONTROL**, run in the same
process, and the check FAILS if the control passes. A perturbed coefficient
must break the residual bound; 1.5x the fitted vector must break the norm
bound; the wrong closed form must be distinguishable; the UNWEIGHTED fit
must not match the row-duplicated one; a different weight vector must move
the coefficient bytes; an overdetermined fit must not interpolate. A gate
that has never been shown capable of failing does not count, and a control
that runs every time is stronger than a sabotage somebody performed once
and reverted.

SABOTAGES FOR THE ORCHESTRATOR TO PERFORM BY HAND, each reverted, on the
seams the in-process controls cannot reach (they are inside kernels):

    (a) `row_vector_binary_mult_kernel` indexing `idx % n_cols` instead of
        `idx // n_cols`, i.e. the COLUMN instantiation of raft's template
        where `olsFit` calls the ROW one. Must fail
        `check_ols_duplicating_a_row_equals_doubling_its_weight`: the
        weights land on features instead of samples, so the fit is a
        rescaling of the design and not a reweighting of the rows.
    (b) `sqrt_elementwise_kernel` storing `w` instead of `sqrt(w)`. Must
        fail the same check: rows get weight `w^2`, so a weight of 2 counts
        four times.
    (c) the restore block (`ols.cuh:129-141`) deleted. Must fail
        `check_ols_sample_weight_restores_its_operands` on all three
        buffers, and must NOT fail the duplicate-row check, which is the
        evidence the two checks separate the solve from the restore.
    (d) `lstsq_min_norm`'s step 6 using `gemv_n(w, a, z)` instead of
        `xty_kernel` -- `A z` where `A^T z` is wanted. Must fail
        `check_ols_wide_is_the_minimum_norm_solution` on the residual, and
        (at n != d) fail to compile or write the wrong length, which is why
        the check reads `len(w)` first.
    (e) `lstsq_min_norm` forming `A^T A` (`gemm_tn`) instead of `A A^T`.
        Must fail the interpolation property: that is precisely the
        singular Gram the old refusal was about.
"""

from max.gpu.host import DeviceContext

from std.math import sqrt

from glm.impl.glm.ols import OLS_ALGO_EIG, ols_fit, ols_fit_weighted
from glm.impl.linalg.detail.lstsq import lstsq_eig


comptime OLS_ROWS = 4096
comptime OLS_COLS = 8


def _u01(row: Int, k: Int, salt: Int) -> Float64:
    var z = (
        UInt64(row) * 0x9E3779B97F4A7C15
        + UInt64(k + 1) * 0xBF58476D1CE4E5B9
        + UInt64(salt + 1) * 0x94D049BB133111EB
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return Float64(z >> 11) * (1.0 / 9007199254740992.0)


def _true_w(k: Int) -> Float64:
    """Distinct, mixed sign, none of them near zero or near each other."""
    return 3.0 - 0.7 * Float64(k) + (1.0 if k % 2 == 0 else -1.0)


def _solve(
    ctx: DeviceContext, y_scale: Float64, noise: Float64
) raises -> List[Float64]:
    var n = OLS_ROWS
    var d = OLS_COLS

    var a = ctx.enqueue_create_buffer[DType.float32](n * d)
    var b = ctx.enqueue_create_buffer[DType.float32](n)
    var w = ctx.enqueue_create_buffer[DType.float32](d)
    var cov_a = ctx.enqueue_create_buffer[DType.float32](d * d)
    var q = ctx.enqueue_create_buffer[DType.float32](d * d)
    var qs = ctx.enqueue_create_buffer[DType.float32](d * d)
    var s_vec = ctx.enqueue_create_buffer[DType.float32](d)
    var ab = ctx.enqueue_create_buffer[DType.float32](d)
    var inv = ctx.enqueue_create_buffer[DType.float32](d * d)
    var a_alias = ctx.enqueue_create_buffer[DType.float32](n * d)
    var a_alias2 = ctx.enqueue_create_buffer[DType.float32](n * d)
    ctx.synchronize()

    var ha = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var hb = ctx.enqueue_create_host_buffer[DType.float32](n)
    for i in range(n):
        var target = 0.0
        for k in range(d):
            var v = _u01(i, k, 0) - 0.5
            ha.unsafe_ptr().unsafe_store(i * d + k, Float32(v))
            target += v * _true_w(k)
        if noise != 0.0:
            target += noise * (_u01(i, 99, 3) - 0.5)
        hb.unsafe_ptr().unsafe_store(i, Float32(target * y_scale))
    ctx.enqueue_copy(dst_buf=a, src_ptr=ha.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=b, src_ptr=hb.unsafe_ptr())
    ctx.synchronize()

    # Through `olsFit`'s dispatch (`ols.cuh:112`), not around it. Calling
    # `lstsq_eig` directly is what let a singular shape reach the solver.
    ols_fit(
        ctx, a, b, w, cov_a, q, qs, s_vec, ab, inv, a_alias, a_alias2,
        n, d, OLS_ALGO_EIG,
    )

    var hw = ctx.enqueue_create_host_buffer[DType.float32](d)
    ctx.enqueue_copy(dst_ptr=hw.unsafe_ptr(), src_buf=w)
    ctx.synchronize()

    var result = List[Float64]()
    for k in range(d):
        result.append(Float64(hw.unsafe_ptr().unsafe_load(k)))
    return result^


def check_ols_exact() raises:
    var ctx = DeviceContext()
    var w = _solve(ctx, 1.0, 0.0)
    # 1% on a noiseless planted model is the fp32 answer. Under FAST on a
    # column whose vendor matmul is TF32/fp19 (IDENTITY_PATHS row 33,
    # DEVIATIONS 529/540) the Gram and X^T y carry a 10-bit-mantissa
    # product and the smallest coefficient moves by more: measured on the
    # H100 2026-08-23, coefficient 3 = -0.09893 for planted -0.1 (1.07%).
    # That is the shipped FAST arm's accuracy on that vendor, recorded
    # with its label; IDENTICAL never calls the vendor matmul and keeps 1%.
    var lossy = (
        GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL
        and vendor_fp32_matmul_is_lossy(TARGET_COLUMN, ctx.compute_capability())
    )
    var tol = 0.05 if lossy else 0.01
    var worst = 0.0
    for k in range(OLS_COLS):
        var want = _true_w(k)
        var rel = abs(w[k] - want) / abs(want)
        if rel > worst:
            worst = rel
        if rel > tol:
            raise Error(
                "coefficient " + String(k) + " = " + String(w[k])
                + ", planted " + String(want) + ", relative " + String(rel)
                + " (tolerance " + String(tol) + ", vendor product "
                + vendor_fp32_matmul_precision_name(
                    TARGET_COLUMN, ctx.compute_capability()
                ) + ")"
            )
    print(
        "check_ols_exact OK: all " + String(OLS_COLS)
        + " coefficients recovered within " + String(tol * 100.0)
        + "% from a noiseless planted model (worst relative "
        + String(worst) + "; vendor fp32 product on this column: "
        + vendor_fp32_matmul_precision_name(
            TARGET_COLUMN, ctx.compute_capability()
        ) + (", FAST arm -- RECORDED at the TF32/fp19 tolerance" if lossy else "")
        + ")"
    )


def check_ols_scale_invariant() raises:
    """The reach test: scale the target by 5, every coefficient scales by 5."""
    var ctx = DeviceContext()
    var base = _solve(ctx, 1.0, 0.0)
    var scaled = _solve(ctx, 5.0, 0.0)
    for k in range(OLS_COLS):
        var want = base[k] * 5.0
        var rel = abs(scaled[k] - want) / abs(want)
        if rel > 0.01:
            raise Error(
                "scaling the target by 5 did not scale coefficient "
                + String(k) + " by 5 (relative " + String(rel)
                + "). The solver is not reading b the way least squares"
                " says it should."
            )
    print(
        "check_ols_scale_invariant OK: y x5 scaled every coefficient by"
        " exactly 5, which is the reach evidence for xty_kernel"
    )


def check_ols_beats_truth_on_noise() raises:
    """With noise, least squares must fit the SAMPLE better than the truth.

    That is what least squares is: it minimizes the residual on the data in
    front of it, so its residual is at most the residual of the true
    coefficients, always, on any sample. A solver returning something merely
    plausible fails this.
    """
    var ctx = DeviceContext()
    var w = _solve(ctx, 1.0, 0.5)

    var n = OLS_ROWS
    var d = OLS_COLS
    var res_fit = 0.0
    var res_true = 0.0
    for i in range(n):
        var target = 0.0
        var pred_fit = 0.0
        var pred_true = 0.0
        for k in range(d):
            var v = _u01(i, k, 0) - 0.5
            target += v * _true_w(k)
            pred_fit += v * w[k]
            pred_true += v * _true_w(k)
        target += 0.5 * (_u01(i, 99, 3) - 0.5)
        var e1 = target - pred_fit
        var e2 = target - pred_true
        res_fit += e1 * e1
        res_true += e2 * e2

    if res_fit > res_true * 1.0001:
        raise Error(
            "the fitted coefficients have a LARGER residual ("
            + String(res_fit) + ") than the true ones (" + String(res_true)
            + "). Least squares minimizes exactly this, so it cannot lose."
        )
    print(
        "check_ols_beats_truth_on_noise OK: fitted residual "
        + String(res_fit) + " against the true model's " + String(res_true)
    )


def check_ols_dispatch_routes_special_shapes() raises:
    """`ols.cuh:112-113`'s two shapes are SOLVED, and by the right route.

    WHAT THIS CHECK USED TO BE, AND WHY IT CHANGED. It was
    `check_ols_dispatch_guard` and it asserted that both shapes RAISED. That
    was right while the only alternative was `lstsqSvdJacobi`
    (`cusolverDnGesvdj`); it is wrong now that both have a portable route
    (DEVIATIONS 550 and 551, `glm/impl/glm/ols.mojo`). The half of it that
    still matters is unchanged and is asserted below: an ordinary shape must
    not be diverted, or the other two assertions would prove nothing.

    THIS ONE IS A SHAPE CHECK ONLY. It says the calls COMPLETE. Whether what
    they return is the right vector is
    `check_ols_wide_is_the_minimum_norm_solution` and
    `check_ols_single_column_matches_the_closed_form`, which are oracles;
    "it did not raise" is not an answer about arithmetic and is not treated
    as one here.
    """
    var ctx = DeviceContext()

    # n_cols > n_rows. Theirs switches to lstsqSvdJacobi; ours takes
    # lstsq_min_norm.
    var wide = _solve_shaped(ctx, 6, 16)
    if len(wide) != 16:
        raise Error(
            "check_ols_dispatch_routes_special_shapes: the 6 x 16 fit"
            " returned " + String(len(wide)) + " coefficients, not 16"
        )

    # n_cols == 1, the case cuML's Python layer warns about by name.
    var single = _solve_shaped(ctx, 64, 1)
    if len(single) != 1:
        raise Error(
            "check_ols_dispatch_routes_special_shapes: the 64 x 1 fit"
            " returned " + String(len(single)) + " coefficients, not 1"
        )

    # And an ordinary shape must still go through untouched.
    var fine = _solve_shaped(ctx, 256, 4)
    if len(fine) != 4:
        raise Error(
            "check_ols_dispatch_routes_special_shapes: the ordinary 256 x 4"
            " fit returned " + String(len(fine)) + " coefficients, not 4"
        )

    # AN EXPLICIT algo = 0 IS STILL REFUSED, and that is the one refusal
    # left on this dispatch. If it stopped raising, something would be
    # silently substituting a solver for the one a caller named.
    var jacobi_raised = False
    var msg = String("")
    try:
        var j = _solve_shaped_algo(ctx, 256, 4, 0)
        _ = len(j)
    except e:
        jacobi_raised = True
        msg = String(e)
    if not jacobi_raised:
        raise Error(
            "check_ols_dispatch_routes_special_shapes: an EXPLICIT algo = 0"
            " completed. lstsqSvdJacobi is a one-sided Jacobi SVD and this"
            " library does not have one, so something answered for it."
        )
    if msg.find("ONE-SIDED JACOBI SVD") < 0:
        raise Error(
            "check_ols_dispatch_routes_special_shapes: algo 0 raised but the"
            " message does not name what is missing. Got: " + msg
        )

    print(
        "check_ols_dispatch_routes_special_shapes OK: 6x16 -> min-norm,"
        " 64x1 -> eig, 256x4 untouched, explicit algo 0 refused by name"
    )


def _solve_shaped(ctx: DeviceContext, n: Int, d: Int) raises -> List[Float64]:
    """A solve at an arbitrary shape, so the guard can be exercised."""
    var a = ctx.enqueue_create_buffer[DType.float32](n * d)
    var b = ctx.enqueue_create_buffer[DType.float32](n)
    var w = ctx.enqueue_create_buffer[DType.float32](d)
    var cov_a = ctx.enqueue_create_buffer[DType.float32](d * d)
    var q = ctx.enqueue_create_buffer[DType.float32](d * d)
    var qs = ctx.enqueue_create_buffer[DType.float32](d * d)
    var s_vec = ctx.enqueue_create_buffer[DType.float32](d)
    var ab = ctx.enqueue_create_buffer[DType.float32](d)
    var inv = ctx.enqueue_create_buffer[DType.float32](d * d)
    var a_alias = ctx.enqueue_create_buffer[DType.float32](n * d)
    var a_alias2 = ctx.enqueue_create_buffer[DType.float32](n * d)
    ctx.synchronize()

    var ha = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var hb = ctx.enqueue_create_host_buffer[DType.float32](n)
    for i in range(n):
        var target = 0.0
        for k in range(d):
            var v = _u01(i, k, 0) - 0.5
            ha.unsafe_ptr().unsafe_store(i * d + k, Float32(v))
            target += v * _true_w(k)
        hb.unsafe_ptr().unsafe_store(i, Float32(target))
    ctx.enqueue_copy(dst_buf=a, src_ptr=ha.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=b, src_ptr=hb.unsafe_ptr())
    ctx.synchronize()

    ols_fit(
        ctx, a, b, w, cov_a, q, qs, s_vec, ab, inv, a_alias, a_alias2,
        n, d, OLS_ALGO_EIG,
    )

    var hw = ctx.enqueue_create_host_buffer[DType.float32](d)
    ctx.enqueue_copy(dst_ptr=hw.unsafe_ptr(), src_buf=w)
    ctx.synchronize()
    var result = List[Float64]()
    for k in range(d):
        result.append(Float64(hw.unsafe_ptr().unsafe_load(k)))
    return result^


def _solve_shaped_algo(
    ctx: DeviceContext, n: Int, d: Int, algo: Int
) raises -> List[Float64]:
    """`_solve_shaped` with the solver id named explicitly, so a check can
    ask for an arm the dispatch would never choose on its own -- the only
    way to reach `olsFit`'s remaining refusals (algo 0, 2, 3) now that no
    SHAPE forces them."""
    var a = ctx.enqueue_create_buffer[DType.float32](n * d)
    var b = ctx.enqueue_create_buffer[DType.float32](n)
    var w = ctx.enqueue_create_buffer[DType.float32](d)
    var cov_a = ctx.enqueue_create_buffer[DType.float32](d * d)
    var q = ctx.enqueue_create_buffer[DType.float32](d * d)
    var qs = ctx.enqueue_create_buffer[DType.float32](d * d)
    var s_vec = ctx.enqueue_create_buffer[DType.float32](d)
    var ab = ctx.enqueue_create_buffer[DType.float32](d)
    var inv = ctx.enqueue_create_buffer[DType.float32](d * d)
    var a_alias = ctx.enqueue_create_buffer[DType.float32](n * d)
    var a_alias2 = ctx.enqueue_create_buffer[DType.float32](n * d)
    ctx.synchronize()
    var ha = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var hb = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    for i in range(n * d):
        ha.unsafe_ptr().unsafe_store(i, Float32(_u01(i, 0, 11) - 0.5))
    for i in range(n):
        hb.unsafe_ptr().unsafe_store(i, Float32(_u01(i, 1, 12) - 0.5))
    ctx.enqueue_copy(dst_buf=a, src_ptr=ha.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=b, src_ptr=hb.unsafe_ptr())
    ctx.synchronize()
    ols_fit(
        ctx, a, b, w, cov_a, q, qs, s_vec, ab, inv, a_alias, a_alias2,
        n, d, algo,
    )
    var hw = ctx.enqueue_create_host_buffer[DType.float32](d)
    ctx.enqueue_copy(dst_ptr=hw.unsafe_ptr(), src_buf=w)
    ctx.synchronize()
    var result = List[Float64]()
    for k in range(d):
        result.append(Float64(hw.unsafe_ptr().unsafe_load(k)))
    _ = ha^
    _ = hb^
    _ = hw^
    return result^


# ===========================================================================
# DEVIATION 527 -- THE IDENTITY PROPERTIES
# ===========================================================================
#
# WHY THE FOUR CHECKS ABOVE PROVE NOTHING ABOUT IDENTITY, SAID PLAINLY.
# `check_ols_exact`, `check_ols_scale_invariant` and
# `check_ols_beats_truth_on_noise` are TOLERANCE checks -- 1%, 1%, and a
# residual comparison at 1.0001. Every one of them passes on a build whose
# summation order is a different number on every vendor, because a
# tolerance cannot see a summation order at all. Their being green says the
# solver is a least-squares solver. It says nothing about whether it is the
# SAME least-squares solver on Metal, CUDA and HIP.
# `check_ols_dispatch_guard` is a REFUSAL check and is exact, but what it
# gates is a shape, not a bit.
#
# The checks below are bitwise or discrete, and each asserts something
# DIFFERENT in the two modes, because `GLOBAL_NUMERIC_MODE` is comptime and
# one process is one mode:
#
#   FAST       the shipped behaviour does not move; where the upstream is
#              not launch-invariant the check REPORTS rather than asserts,
#              and that report is what prices the pin.
#   IDENTICAL  the pinned arms are REACHED, the fitted coefficients are
#              bit-identical across launch geometries that must not matter,
#              and the one shape past the pinned Gram kernel's capacity
#              RAISES by name.
#
# CORRECTED 2026-09-01. The last clause used to read "and the one shape
# whose only other arm is a closed vendor library RAISES by name". No shape
# refuses for that reason any more: `n_cols > n_rows` and `n_cols == 1` are
# solved (DEVIATIONS 550, 551). The refusal that survives under IDENTICAL is
# `check_ols_refuses_over_capacity`'s, which is about the pinned Gram
# kernel's capacity and not about anybody's library.

from std.memory import bitcast

from core.gemm import gemm_nt, gemm_tn, gemv_n
from core.gram_splitk import (
    GRAM_MAX_COLS,
    GRAM_SPLITK_RESOLVED_COLUMN,
    gram_splitk_applies,
)
from core.identity_trace import IdentityTrace, first_divergence, fnv1a64_bytes
from glm.estimator import ols_fit_host, ols_fit_weighted_host
from glm.checks.ols_trace import (
    OLS_CARD_COLS,
    OLS_CARD_ROWS,
    _hash_f32,
    emit_ols_card,
)
from glm.impl.linalg.detail.lstsq import OLS_ELEM_TPB, OLS_NONZERO_THRESH
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    identical_sqrt,
    numeric_mode_name,
)
from checks.kernel_matrix import (
    COLUMN_BIT_IDENTICAL,
    TARGET_COLUMN,
    column_name,
    vendor_fp32_matmul_is_lossy,
    vendor_fp32_matmul_precision_name,
)


comptime IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL


def _mode_name() -> String:
    """The build's tier, from the ONE definition of it.

    Delegates to `numeric_mode_name()` since 2026-08-29; see the note
    on that function. A local two-way IDENTICAL-or-FAST answers "FAST"
    for a DETERMINISTIC build, which mislabels every line the driver
    prints.
    """
    return numeric_mode_name()


# ===========================================================================
# 1. THE PINNED ARM IS THE ARM OLS TAKES
# ===========================================================================


def check_ols_arms_are_pinned() raises:
    """Step 1's Gram product enters the PINNED kernel at OLS's own shapes.

    `core/gemm_identity_check.mojo` proves the arm resolution against the
    vendor table. This check asks the narrower question that belongs to this
    estimator: **at the shapes ordinary least squares actually calls with**,
    does the predicate answer True? A pin on a kernel a caller's shape never
    reaches is a pin on nothing, and OLS's shape is `m = n = n_features`,
    `k = n_rows`, which is a different corner of that predicate's domain
    from the one the gemm check exercises.

    Under FAST this is a REPORT: which arm the shipped build takes at these
    shapes is a measurement, and on the Apple column it happens to be the
    same kernel, which is exactly why the IDENTICAL assertion cannot be
    replaced by running the FAST build and looking.
    """
    comptime expected = COLUMN_BIT_IDENTICAL if IDENTICAL else TARGET_COLUMN
    if GRAM_SPLITK_RESOLVED_COLUMN != expected:
        raise Error(
            "check_ols_arms_are_pinned: step 1's arm compiles against column "
            + column_name(GRAM_SPLITK_RESOLVED_COLUMN)
            + " but this build requires "
            + column_name(expected)
            + ". Under IDENTICAL, reading the DEVICE's column here sends"
            " NVIDIA and AMD to linalg.matmul, whose k-split is a"
            " per-vendor summation order. IDENTITY_PATHS row 27."
        )

    # The shipped OLS shapes: the check fixture, the card fixture, and the
    # widest feature count the split-K kernel can serve.
    var shapes_ok = True
    if not gram_splitk_applies(OLS_COLS, OLS_COLS, OLS_ROWS):
        shapes_ok = False
    if not gram_splitk_applies(OLS_CARD_COLS, OLS_CARD_COLS, OLS_CARD_ROWS):
        shapes_ok = False
    if not gram_splitk_applies(GRAM_MAX_COLS, GRAM_MAX_COLS, 100_003):
        shapes_ok = False

    comptime if IDENTICAL:
        if not shapes_ok:
            raise Error(
                "check_ols_arms_are_pinned: one of OLS's own Gram shapes"
                " does NOT take the split-K arm under IDENTICAL, so this"
                " estimator's step 1 falls through to the refusal or to a"
                " closed library. IDENTITY_PATHS row 27."
            )
        print(
            "check_ols_arms_are_pinned OK [IDENTICAL]: step 1 compiles"
            " against column "
            + column_name(GRAM_SPLITK_RESOLVED_COLUMN)
            + " and the pinned Gram kernel serves "
            + String(OLS_COLS)
            + ", "
            + String(OLS_CARD_COLS)
            + " and "
            + String(GRAM_MAX_COLS)
            + " features"
        )
    else:
        print(
            "check_ols_arms_are_pinned REPORT [FAST]: step 1 compiles"
            " against column "
            + column_name(GRAM_SPLITK_RESOLVED_COLUMN)
            + " (the device's), split-K covers OLS's shapes: "
            + String(shapes_ok)
        )


# ===========================================================================
# 2. THE REFUSAL, AT OLS'S OWN ENTRY POINT
# ===========================================================================


def _fit_at_width(ctx: DeviceContext, n: Int, d: Int) raises -> String:
    """Fit an `n x d` design through `ols_fit`. Returns the error text, or
    "" if it completed."""
    try:
        var w = _solve_shaped(ctx, n, d)
        _ = len(w)
    except e:
        return String(e)
    return String("")


def check_ols_refuses_over_capacity() raises:
    """A feature count past the pinned Gram kernel's capacity RAISES under
    IDENTICAL and RUNS under FAST.

    This is not a copy of `check_gemm_tn_refuses_over_capacity`: that one
    calls `gemm_tn` directly, and what is asserted here is that the refusal
    SURVIVES THE WHOLE ESTIMATOR -- that it is not swallowed by a `try` in
    `lstsq_eig`, not turned into a warning by `ols_fit`, and not preceded by
    some other guard that fires first and hides it. A refusal a caller never
    sees is a fall-through with extra steps.

    `n_cols = GRAM_MAX_COLS + 2` with `n_rows` comfortably larger, so the
    `ols.cuh:112-113` guard (`n_cols > n_rows`) does NOT fire and the shape
    reaches `lstsq_eig`'s step 1, which is the Gram product this is about.
    """
    var d = GRAM_MAX_COLS + 2
    var n = 512
    if d >= n:
        raise Error(
            "check_ols_refuses_over_capacity: the fixture is n_cols >="
            " n_rows, so ols.cuh:113 sends it to lstsq_min_norm, whose Gram"
            " is n_rows x n_rows and never reaches the capacity this check"
            " is about. The check would be measuring the wrong route."
        )
    with DeviceContext() as ctx:
        var err = _fit_at_width(ctx, n, d)
        comptime if IDENTICAL:
            if err == "":
                raise Error(
                    "check_ols_refuses_over_capacity: a "
                    + String(n)
                    + " x "
                    + String(d)
                    + " design COMPLETED under IDENTICAL. It cannot have"
                    " used the pinned Gram kernel (n_cols > "
                    + String(GRAM_MAX_COLS)
                    + "), so step 1 ran on linalg.matmul and this fit is a"
                    " model the mode promises is vendor-independent and is"
                    " not."
                )
            if err.find("IDENTITY_PATHS row 27") < 0:
                raise Error(
                    "check_ols_refuses_over_capacity: it refused, but the"
                    " message does not cite the ledger row, so a user"
                    " cannot trace it. Got: "
                    + err
                )
            print(
                "check_ols_refuses_over_capacity OK [IDENTICAL]: "
                + String(n)
                + " x "
                + String(d)
                + " raised by name through the whole estimator rather than"
                " falling through to the vendor matmul"
            )
        else:
            if err != "":
                raise Error(
                    "check_ols_refuses_over_capacity: the FAST build must"
                    " still fit "
                    + String(n)
                    + " x "
                    + String(d)
                    + " through the transpose+matmul arm, but it raised: "
                    + err
                )
            print(
                "check_ols_refuses_over_capacity OK [FAST]: "
                + String(n)
                + " x "
                + String(d)
                + " still fits on the transpose+matmul arm"
            )


# ===========================================================================
# 3. THE HEADLINE -- THE COEFFICIENTS DO NOT MOVE WITH THE LAUNCH
# ===========================================================================


def _fit_bits(
    ctx: DeviceContext,
    n: Int,
    d: Int,
    elem_tpb: Int,
    pad: Int,
    poison: Float32,
    salt: Int,
) raises -> List[UInt32]:
    """One fit of the card fixture, returning the coefficient BIT PATTERNS.

    Three things are varied here and all three are SCHEDULING, argued rather
    than assumed:

    - `elem_tpb`: the block width of `divide_columns_by_nonzero_kernel` and
      `diagonal_to_vector_kernel`. Each thread owns one output cell
      (`idx = block_idx.x * block_dim.x + thread_idx.x`, one store per
      `idx`), so this moves WHICH thread computes a cell and never what any
      cell is computed from. It is NOT threaded into the Gram fold, the
      `xty` fold or the Jacobi block, because those three block widths ARE
      fold widths.
    - `pad`: extra trailing capacity on every device buffer. No kernel reads
      a length -- every one takes its extents as arguments -- so this
      changes only the allocator's answers and, with them, the base
      ALIGNMENT of every pointer. That is the class IDENTITY_PATHS row 22
      names by hand (`fused_veclen_for` reads pointer alignment, "which is
      an allocator's business"); a product whose rounding depended on
      alignment would fail here.
    - `poison`: what the scratch and output buffers contained BEFORE the
      fit. A kernel that leaves an output cell unwritten, or folds a
      partials slot nobody filled, reads this. `-987654.0` and `+13.5` are
      both far from anything the fit produces, so a survivor is visible.

    None of the three changes one operand of one arithmetic operation. Under
    IDENTICAL the coefficient bytes must therefore be equal.
    """
    var a = ctx.enqueue_create_buffer[DType.float32](n * d + pad)
    var b = ctx.enqueue_create_buffer[DType.float32](n + pad)
    var w = ctx.enqueue_create_buffer[DType.float32](d + pad)
    var cov_a = ctx.enqueue_create_buffer[DType.float32](d * d + pad)
    var q = ctx.enqueue_create_buffer[DType.float32](d * d + pad)
    var qs = ctx.enqueue_create_buffer[DType.float32](d * d + pad)
    var s_vec = ctx.enqueue_create_buffer[DType.float32](d + pad)
    var ab = ctx.enqueue_create_buffer[DType.float32](d + pad)
    var inv = ctx.enqueue_create_buffer[DType.float32](d * d + pad)
    var a_alias = ctx.enqueue_create_buffer[DType.float32](n * d + pad)
    var a_alias2 = ctx.enqueue_create_buffer[DType.float32](n * d + pad)
    ctx.synchronize()

    # POISON EVERY BUFFER THE FIT WRITES, including the scratch it reuses.
    var big = ctx.enqueue_create_host_buffer[DType.float32](n * d + pad)
    ctx.synchronize()
    for i in range(n * d + pad):
        big.unsafe_ptr().unsafe_store(i, poison)
    ctx.enqueue_copy(dst_buf=w, src_ptr=big.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=cov_a, src_ptr=big.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=q, src_ptr=big.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=qs, src_ptr=big.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=s_vec, src_ptr=big.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=ab, src_ptr=big.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=inv, src_ptr=big.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=a_alias, src_ptr=big.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=a_alias2, src_ptr=big.unsafe_ptr())
    ctx.synchronize()

    # THE FIXTURE IS THE CARD'S: bit-assembled, no host float arithmetic, so
    # the two arms cannot differ because their inputs did. `salt` is the
    # same for both arms; it is a parameter only so a caller can prove the
    # comparator is not vacuous by handing the two arms different data.
    #
    # ONE STAGING BUFFER PER COPY. This block used to fill `big`, enqueue the
    # copy into `a`, then OVERWRITE THE FIRST n FLOATS OF `big` and enqueue
    # the copy into `b`, with no synchronize between them. `enqueue_copy` is
    # a promise, not a read. On a discrete GPU the A upload is a real
    # asynchronous DMA of 98,304 floats and the host rewrite of the first
    # 8,192 of them races it, so `a` held a design matrix whose first ~683
    # rows were whichever side won each cache line. That is upstream of
    # `ols_fit` and therefore MODE-INDEPENDENT, which is exactly why
    # `check_ols_is_launch_invariant` failed on the H100 and the MI325X in
    # BOTH numeric modes and passed on Apple, where unified memory leaves no
    # DMA to race. It is the harness, not a kernel. The comment above was
    # false in the one way that mattered: the arms DID differ because their
    # inputs did.
    #
    # `emit_ols_card`'s fixture (`ols_trace.mojo`) always used two host
    # buffers, which is why the traced repro never reproduced and every leg
    # reported that the card fixture's two fits agreed. PORTING.md item 12
    # states the rule this violated, and records that the last time it cost
    # an hour and "presented as a broken kernel".
    var big_b = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    for i in range(n * d):
        big.unsafe_ptr().unsafe_store(i, _hash_f32(i, salt))
    for i in range(n):
        big_b.unsafe_ptr().unsafe_store(i, _hash_f32(i, salt + 1))
    ctx.enqueue_copy(dst_buf=a, src_ptr=big.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=b, src_ptr=big_b.unsafe_ptr())
    ctx.synchronize()
    _ = big_b^

    ols_fit(
        ctx, a, b, w, cov_a, q, qs, s_vec, ab, inv, a_alias, a_alias2,
        n, d, OLS_ALGO_EIG, False, False, elem_tpb,
    )

    var hw = ctx.enqueue_create_host_buffer[DType.float32](d)
    ctx.enqueue_copy(dst_ptr=hw.unsafe_ptr(), src_buf=w)
    ctx.synchronize()
    var out = List[UInt32]()
    for k in range(d):
        var v = hw.unsafe_ptr().unsafe_load(k)
        if v == poison:
            raise Error(
                "check_ols_is_launch_invariant: the poison survived at"
                " coefficient "
                + String(k)
                + " -- that coefficient was never written, so nothing"
                " below is a comparison of fits."
            )
        out.append(bitcast[DType.uint32](v))
    _ = a
    _ = b
    _ = w
    _ = cov_a
    _ = q
    _ = qs
    _ = s_vec
    _ = ab
    _ = inv
    _ = a_alias
    _ = a_alias2
    _ = big^
    _ = hw^
    return out^


def check_ols_is_launch_invariant() raises:
    """**THE HEADLINE.** The fitted coefficients are bit-identical across
    launch geometries that must not matter.

    Three fits of one design matrix:

        A   elem_tpb = 256, no padding,  poison -987654.0
        B   elem_tpb =  64, 37 floats of padding, poison +13.5
        C   elem_tpb = 256, no padding,  poison -987654.0   (A repeated)

    A vs B varies the elementwise block width, every buffer's base
    alignment, and every scratch buffer's prior contents. A vs C varies
    nothing at all and is the run-to-run control, which is what makes an
    A == B result mean something: without it, two fits agreeing could simply
    be a device that returns the same wrong thing twice.

    UNDER IDENTICAL BOTH ARE ASSERTIONS. Under FAST, A vs C is still an
    assertion -- a fit that is not repeatable in one process on one device
    is a defect in either mode, and nothing on this path uses a float atomic
    -- while A vs B is a REPORT, because the FAST arms include closed vendor
    kernels that are entitled to tile by shape and whether they do is the
    measurement that prices the pins.

    The comparison is on BIT PATTERNS. A tolerance here would pass on every
    build this file exists to distinguish.
    """
    comptime N = 8192
    comptime D = 12
    var a_bits: List[UInt32]
    var b_bits: List[UInt32]
    var c_bits: List[UInt32]
    with DeviceContext() as ctx:
        a_bits = _fit_bits(ctx, N, D, OLS_ELEM_TPB, 0, Float32(-987654.0), 527)
        b_bits = _fit_bits(ctx, N, D, 64, 37, Float32(13.5), 527)
        c_bits = _fit_bits(ctx, N, D, OLS_ELEM_TPB, 0, Float32(-987654.0), 527)

    var ab_same = True
    var ab_first = -1
    for k in range(D):
        if a_bits[k] != b_bits[k]:
            ab_same = False
            if ab_first < 0:
                ab_first = k
    var ac_same = True
    var ac_first = -1
    for k in range(D):
        if a_bits[k] != c_bits[k]:
            ac_same = False
            if ac_first < 0:
                ac_first = k

    if not ac_same:
        # LOCALIZE before raising (2026-08-23): the H100 leg raised here in
        # BOTH modes (coefficient 0: 0xbbc76fa8 vs 0xbbc6fa1b, 0.2% apart)
        # and the message named the class but not the stage. Two traced
        # fits of the card fixture at THIS shape, in this process, and the
        # first record they disagree on -- the same certificate machinery
        # the cross-vendor legs use, pointed at one process.
        var stage = String("<no traced repro: the card fixture's two fits agreed>")
        var pa = String("/tmp/mojolearn_ols_launch_inv_a.card")
        var pb = String("/tmp/mojolearn_ols_launch_inv_b.card")
        with DeviceContext() as ctx2:
            var ta = IdentityTrace.to_path(pa, "", True)
            _ = emit_ols_card(ctx2, ta, N, D, OLS_ELEM_TPB, 527)
            var tb = IdentityTrace.to_path(pb, "", True)
            _ = emit_ols_card(ctx2, tb, N, D, OLS_ELEM_TPB, 527)
        var d2 = first_divergence(pa, pb)
        if d2 != "":
            stage = d2
        raise Error(
            "check_ols_is_launch_invariant: TWO IDENTICAL FITS in one"
            " process disagreed at coefficient "
            + String(ac_first)
            + " ("
            + hex(a_bits[ac_first])
            + " vs "
            + hex(c_bits[ac_first])
            + "). Nothing on this path uses a float atomic, so this is not"
            " the ordering hazard `numerics.mojo` warns about -- it is an"
            " uninitialized read or a race, and it is a defect in BOTH"
            " modes. This ran in mode ["
            + _mode_name()
            + "]. FIRST DIVERGING STAGE of two traced fits at the same"
            " shape: "
            + stage
        )

    comptime if IDENTICAL:
        if not ab_same:
            raise Error(
                "check_ols_is_launch_invariant: the fitted coefficients"
                " MOVED with the launch geometry. Coefficient "
                + String(ab_first)
                + " is "
                + hex(a_bits[ab_first])
                + " at elem_tpb=256/no padding/poison -987654 and "
                + hex(b_bits[ab_first])
                + " at elem_tpb=64/37 floats of padding/poison +13.5."
                " None of those three is an operand of any arithmetic in"
                " this estimator, so under IDENTICAL they cannot reach the"
                " answer. This is the cross-vendor failure measured on ONE"
                " device, before any second GPU."
            )
        print(
            "check_ols_is_launch_invariant OK [IDENTICAL]: all "
            + String(D)
            + " coefficients bit-identical across elem_tpb 256/64, 0/37"
            " floats of buffer padding and two different scratch poisons"
            " (coef[0] = "
            + hex(a_bits[0])
            + "), and the A-vs-A control agrees"
        )
    else:
        print(
            "check_ols_is_launch_invariant REPORT [FAST]: run-to-run"
            " control AGREES (asserted); geometry invariance is "
            + String(ab_same)
            + " -- coef[0] "
            + hex(a_bits[0])
            + " vs "
            + hex(b_bits[0])
            + ". A False here is not a bug, it is the measurement that"
            " prices the pins."
        )


# ===========================================================================
# 4. THE HOST SURFACE TAKES THE DISPATCH GUARD (the DEVIATION 527 fix)
# ===========================================================================


def check_ols_host_surface_takes_the_guard() raises:
    """`ols_fit_host` goes THROUGH `olsFit`'s dispatch, not around it.

    THE DEFECT THIS GATES, FOUND BY DEVIATION 527's AUDIT AND NOT BY ANY
    GREEN CHECK. `glm/impl/glm/ols.mojo` exists because reaching `lstsq_eig`
    directly skips `ols.cuh:112-113`, which switches away from the
    normal-equations solver at `n_cols > n_rows` (where `A^T A` is singular
    by construction) and at `n_cols == 1`. That file's docstring records the
    bypass as closed. **It was closed for the Mojo callers only.**
    `ols_fit_host` -- the host-pointer surface, the one with
    `mojolearn.LinearRegression` on the other end of it -- still called
    `lstsq_eig` directly, so a Python user handing in a wide design got a
    plausible vector out of a singular inverse and no error.

    HOW THIS CHECK CHANGED WITH DEVIATIONS 550 AND 551. It used to assert
    that the host surface RAISED at 4 x 8 and 64 x 1, because raising was
    what the dispatch did there. Both shapes are now solved, so a refusal is
    no longer available as evidence and something stronger replaces it: the
    host surface's 6 x 16 fit must INTERPOLATE, `max_i |(A w - b)_i|` at the
    float32 noise floor. A bypass straight to `lstsq_eig` cannot do that --
    `A^T A` is 16 x 16 of rank at most 6, `DivideByNonZero` drops ten
    directions, and what comes back does not solve the system. So the
    property is exactly a test of WHICH ROUTE the host surface took, and it
    is stronger than the refusal it replaces, which only ever said that some
    branch fired.

    Mode-independent: a residual at the noise floor is not a bitwise claim.
    """
    var n = 6
    var d = 16
    var fx = _wide_fixture(n, d)
    var a64 = fx[0].copy()
    var b64 = fx[2].copy()
    var x = List[Float32]()
    for i in range(n * d):
        x.append(Float32(a64[i]))
    var y = List[Float32]()
    for i in range(n):
        y.append(Float32(b64[i]))
    var w = _host_fit_coefs(x, y, n, d)

    var bmax = 0.0
    for i in range(n):
        if abs(b64[i]) > bmax:
            bmax = abs(b64[i])
    var res = _residual_inf(a64, w, b64, n, d)
    var bound = 1.0e-4 * (bmax if bmax > 1.0 else 1.0)
    if not (res <= bound):
        raise Error(
            "check_ols_host_surface_takes_the_guard: the 6 x 16 fit through"
            " ols_fit_host has residual " + String(res) + " > "
            + String(bound) + ". A wide design has exact solutions and the"
            " min-norm route finds one; this did not, so the host surface is"
            " NOT going through olsFit's dispatch -- it is on lstsq_eig with"
            " a singular A^T A. See glm/estimator.mojo."
        )

    # THE NEGATIVE CONTROL. The same fixture solved by the WRONG route: a
    # tall-shaped call at 16 x 6, which is a different (overdetermined)
    # problem on the same numbers and must NOT interpolate. Without it,
    # "the residual is small" could be a property of the fixture.
    var xt = List[Float32]()
    for i in range(d * n):
        xt.append(Float32(a64[i]))
    var yt = List[Float32]()
    for i in range(d):
        yt.append(Float32(_hash_f32(i, 7601)))
    var wt = _host_fit_coefs(xt, yt, d, n)
    var at = List[Float64]()
    for i in range(d * n):
        at.append(a64[i])
    var bt = List[Float64]()
    for i in range(d):
        bt.append(Float64(yt[i]))
    var res_t = _residual_inf(at, wt, bt, d, n)
    if res_t <= bound:
        raise Error(
            "check_ols_host_surface_takes_the_guard: the NEGATIVE CONTROL"
            " interpolated. A 16 x 6 overdetermined fit on hashed targets"
            " has no exact solution, so a residual of " + String(res_t)
            + " below " + String(bound) + " means the bound is meaningless"
            " and the assertion above proves nothing."
        )

    # And an ordinary shape must still come back with the right count.
    var ord_x = List[Float32]()
    for i in range(256 * 4):
        ord_x.append(_hash_f32(i, 7602))
    var ord_y = List[Float32]()
    for i in range(256):
        ord_y.append(_hash_f32(i, 7603))
    var ord_w = _host_fit_coefs(ord_x, ord_y, 256, 4)
    if len(ord_w) != 4:
        raise Error(
            "check_ols_host_surface_takes_the_guard: the ordinary 256 x 4"
            " fit came back with " + String(len(ord_w)) + " coefficients"
        )

    print(
        "check_ols_host_surface_takes_the_guard OK ["
        + _mode_name()
        + "]: the Python-facing entry interpolates a 6 x 16 design"
        " (residual " + String(res) + "), i.e. it takes the min-norm route"
        " and not lstsq_eig; the 16 x 6 control does not (" + String(res_t)
        + ")"
    )


def _host_fit_coefs(
    x: List[Float32], y: List[Float32], n: Int, d: Int
) raises -> List[Float64]:
    """`ols_fit_host` through raw addresses -- the entry
    `bindings/_mojolearn_estimators.mojo` and therefore
    `mojolearn.LinearRegression` reach -- returning the coefficients.

    The addresses are built the same way `_f32_ptr` builds them from numpy,
    which is what makes this a check of the PYTHON-FACING entry rather than
    of a Mojo-shaped imitation of it."""
    var xs = x.copy()
    var ys = y.copy()
    var coef = List[Float32]()
    for _ in range(d):
        coef.append(Float32(0.0))
    var xp = MutPointer[Float32, MutUntrackedOrigin](
        unsafe_from_address=Int(xs.unsafe_ptr())
    )
    var yp = MutPointer[Float32, MutUntrackedOrigin](
        unsafe_from_address=Int(ys.unsafe_ptr())
    )
    var wp = MutPointer[Float32, MutUntrackedOrigin](
        unsafe_from_address=Int(coef.unsafe_ptr())
    )
    with DeviceContext() as ctx:
        ols_fit_host(ctx, xp, yp, wp, n, d)
        var out = List[Float64]()
        for k in range(d):
            out.append(Float64(coef[k]))
        _ = xs^
        _ = ys^
        _ = coef^
        # DEVIATION 1946: the context dies LAST, after every value built on
        # it. Mojo frees at LAST USE, so without this the buffer releases
        # above run against a context that is already gone. On sm_89 the
        # next GPU call in the process then never returns; Apple and AMD do
        # not show it, which is how it stayed latent here.
        _ = ctx^
        return out^


# ===========================================================================
# 5. THE RANK GUARD IS A FLOAT COMPARISON WITH A DISCRETE OUTPUT
# ===========================================================================


def _rank_at_scale(ctx: DeviceContext, n: Int, d: Int, e: Int) raises -> Int:
    """Fit a design scaled by `2^e` and return how many directions survive
    `DivideByNonZero`.

    The scale is a POWER OF TWO applied by exponent surgery on the bits, so
    it is exact: every mantissa is unchanged and the design is the same
    design in different units, which is the only way to ask this question
    without the answer being contaminated by the rescaling's own rounding.
    """
    var a = ctx.enqueue_create_buffer[DType.float32](n * d)
    var b = ctx.enqueue_create_buffer[DType.float32](n)
    var w = ctx.enqueue_create_buffer[DType.float32](d)
    var cov_a = ctx.enqueue_create_buffer[DType.float32](d * d)
    var q = ctx.enqueue_create_buffer[DType.float32](d * d)
    var qs = ctx.enqueue_create_buffer[DType.float32](d * d)
    var s_vec = ctx.enqueue_create_buffer[DType.float32](d)
    var ab = ctx.enqueue_create_buffer[DType.float32](d)
    var inv = ctx.enqueue_create_buffer[DType.float32](d * d)
    var a_alias = ctx.enqueue_create_buffer[DType.float32](n * d)
    var a_alias2 = ctx.enqueue_create_buffer[DType.float32](n * d)
    ctx.synchronize()
    var ha = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var hb = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    for i in range(n * d):
        var v = _hash_f32(i, 4242)
        var bits = bitcast[DType.uint32](v)
        var ex = Int((bits >> 23) & UInt32(0xFF)) + e
        var nb = (bits & UInt32(0x807FFFFF)) | (UInt32(ex) << 23)
        ha.unsafe_ptr().unsafe_store(i, bitcast[DType.float32](nb))
    for i in range(n):
        hb.unsafe_ptr().unsafe_store(i, _hash_f32(i, 4243))
    ctx.enqueue_copy(dst_buf=a, src_ptr=ha.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=b, src_ptr=hb.unsafe_ptr())
    ctx.synchronize()
    ols_fit(
        ctx, a, b, w, cov_a, q, qs, s_vec, ab, inv, a_alias, a_alias2,
        n, d, OLS_ALGO_EIG,
    )
    var hs = ctx.enqueue_create_host_buffer[DType.float32](d)
    ctx.enqueue_copy(dst_ptr=hs.unsafe_ptr(), src_buf=s_vec)
    ctx.synchronize()
    var kept = 0
    for i in range(d):
        var lam = hs.unsafe_ptr().unsafe_load(i)
        if lam > OLS_NONZERO_THRESH or lam < -OLS_NONZERO_THRESH:
            kept += 1
    _ = a
    _ = b
    _ = w
    _ = cov_a
    _ = q
    _ = qs
    _ = s_vec
    _ = ab
    _ = inv
    _ = a_alias
    _ = a_alias2
    _ = ha^
    _ = hb^
    _ = hs^
    return kept


def check_ols_rank_guard_is_absolute() raises:
    """`DivideByNonZero`'s threshold is ABSOLUTE, and the same design in
    different units therefore has a different RANK.

    THE CLASS THIS BELONGS TO. Every other pathway in this estimator is a
    rounding: two vendors disagree in the last bits and the model drifts.
    This one is not. `divide_columns_by_nonzero_kernel` compares a float
    against `OLS_NONZERO_THRESH` and ZEROES a whole eigen-direction on the
    strength of it, so what comes out of the comparison is an INTEGER --
    the rank of the pseudo-inverse -- and a last-bit move at the boundary
    changes the model's rank rather than its fifth decimal. IDENTITY_PATHS
    has no more dangerous shape.

    WHAT IS ASSERTED, and it is a measurement rather than an opinion: the
    same design matrix, multiplied by an exact power of two, is fitted at
    two scales, and the two ranks are DIFFERENT. That is not a vendor
    difference and it is not a bug this lane introduced -- it is the
    consequence of an absolute threshold on a quantity (an eigenvalue of
    `A^T A`) that scales with the SQUARE of the data. It is the identical
    defect DEVIATION BLOCK 1 of `jacobi_eigh_device.mojo` fixed one step
    upstream, where the convergence test was `off <= 1e-10` on a quantity
    that also scales with the square of the data, and where the fix was to
    make the test relative.

    It is NOT fixed here. `glm/impl/` is COPY-DO-NOT-IMPROVE and changing
    the threshold moves shipped `NUMERIC_FAST` bits on every design near
    the boundary. The finding is recorded, gated, and handed up.

    Mode-independent: the threshold is a comptime constant in both modes.
    """
    comptime N = 4096
    comptime D = 8
    var full = 0
    var shrunk = 0
    with DeviceContext() as ctx:
        full = _rank_at_scale(ctx, N, D, 0)
        # 2^-24 per entry -> eigenvalues of A^T A down by 2^-48, which puts
        # them under 1e-10 for this fixture. Every mantissa is unchanged.
        shrunk = _rank_at_scale(ctx, N, D, -24)
    if full != D:
        raise Error(
            "check_ols_rank_guard_is_absolute: the unscaled design already"
            " lost directions (rank "
            + String(full)
            + " of "
            + String(D)
            + "), so the fixture is not the well-conditioned one this"
            " check needs and the comparison below is not about units."
        )
    if shrunk >= full:
        raise Error(
            "check_ols_rank_guard_is_absolute: scaling the design by 2^-24"
            " did NOT change the rank ("
            + String(shrunk)
            + " vs "
            + String(full)
            + "). Either the threshold has been made relative -- in which"
            " case delete this check and the OLS_NONZERO_THRESH docstring"
            " that calls it absolute -- or the fixture's eigenvalues no"
            " longer straddle "
            + String(OLS_NONZERO_THRESH)
            + " at that scale."
        )
    print(
        "check_ols_rank_guard_is_absolute OK ["
        + _mode_name()
        + "]: the SAME design has rank "
        + String(full)
        + " at scale 1 and rank "
        + String(shrunk)
        + " at scale 2^-24. DivideByNonZero's threshold is absolute"
        " (OLS_NONZERO_THRESH), the quantity it tests scales with the"
        " square of the data, and what the comparison outputs is a RANK."
    )


# ===========================================================================
# 6. THE CERTIFICATE HASHES BYTES, AND DECIMAL TEXT WOULD NOT DO
# ===========================================================================


def check_ols_card_hashes_raw_bytes() raises:
    """A card built from printed floats cannot see a one-ulp move. MEASURED.

    `core/identity_trace.mojo`'s first rule is BIT PATTERNS, NEVER DECIMAL
    TEXT, and the reason given is that `String(Float32)` does not round trip
    in this toolchain. That is an argument. This is the number: over the
    200,000 adjacent Float32 pairs starting at `0x3F000000`, **390 pairs
    render to the SAME decimal string** -- 0.195%, the first of them
    `0x3f00004b` and `0x3f00004c`, both printing `0.5000045`. A card built
    on `String()` is therefore blind to about one adjacent-pair move in five
    hundred, silently, and would report AGREEMENT across a real divergence,
    which is the worst failure an instrument of this kind can have.

    The check: find such a pair, then require that the card's own hash
    function separates the two while the decimal rendering does not. It
    fails if `String` starts round-tripping (in which case the rule can be
    revisited on evidence) and it fails if `fnv1a64_bytes` stops
    distinguishing adjacent floats (in which case the certificate is
    worthless).

    Mode-independent: this is a property of the instrument, not of the
    arithmetic.
    """
    var found = False
    var lo = UInt32(0)
    for i in range(200_000):
        var bits = UInt32(0x3F000000) + UInt32(i)
        var x = bitcast[DType.float32](bits)
        var y = bitcast[DType.float32](bits + UInt32(1))
        if String(x) == String(y):
            found = True
            lo = bits
            break
    if not found:
        raise Error(
            "check_ols_card_hashes_raw_bytes: no adjacent Float32 pair in"
            " 200,000 rendered to the same decimal text. Either String()"
            " now round trips -- which would be news, and"
            " core/identity_trace.mojo's rule 1 should be re-derived on the"
            " new evidence rather than kept on the old -- or this search"
            " window is wrong."
        )

    var one = List[Float32]()
    one.append(bitcast[DType.float32](lo))
    var two = List[Float32]()
    two.append(bitcast[DType.float32](lo + UInt32(1)))
    var h1 = fnv1a64_bytes(
        UInt64(0xCBF29CE484222325), one.unsafe_ptr().bitcast[UInt8](), 4
    )
    var h2 = fnv1a64_bytes(
        UInt64(0xCBF29CE484222325), two.unsafe_ptr().bitcast[UInt8](), 4
    )
    if h1 == h2:
        raise Error(
            "check_ols_card_hashes_raw_bytes: the card's hash function gave"
            " the SAME value to two Float32 values one ulp apart. The"
            " certificate cannot detect a last-bit divergence and every"
            " agreement it has ever reported is unsupported."
        )
    print(
        "check_ols_card_hashes_raw_bytes OK ["
        + _mode_name()
        + "]: "
        + hex(lo)
        + " and "
        + hex(lo + UInt32(1))
        + " both print "
        + String(one[0])
        + " -- decimal text cannot separate them, FNV-1a64 over raw bytes"
        " gives "
        + hex(h1)
        + " vs "
        + hex(h2)
    )
    _ = one^
    _ = two^


# ===========================================================================
# 7. THE CARD ITSELF RUNS, AND ITS STAGES ARE THE ONES CLAIMED
# ===========================================================================


def check_ols_card_is_emitted() raises:
    """The certificate emits eleven named stages and matches its control.

    `glm/ols_trace_main.mojo` is the driver a cross-vendor run uses; this is
    the gate that it still works, taken in process so a broken card is a red
    check rather than an empty file discovered on the other machine. The
    fixture is the card's own, shrunk so the check stays cheap -- the STAGE
    LIST is what is under test here, not the arithmetic.

    Two fits, two files, compared with `first_divergence`. Under IDENTICAL a
    mismatch is a failure; under FAST it is reported, because the shipped
    mode makes no run-to-run promise.
    """
    var p1 = String("/tmp/mojolearn_ols_card_check_a.card")
    var p2 = String("/tmp/mojolearn_ols_card_check_b.card")
    var n1 = 0
    with DeviceContext() as ctx:
        var t1 = IdentityTrace.to_path(p1, "", True)
        var c1 = emit_ols_card(ctx, t1, 2048, 8)
        _ = len(c1)
        n1 = t1.seq
        var t2 = IdentityTrace.to_path(p2, "", True)
        var c2 = emit_ols_card(ctx, t2, 2048, 8)
        _ = len(c2)
        if t2.seq != n1:
            raise Error(
                "check_ols_card_is_emitted: two fits of the same fixture"
                " emitted "
                + String(n1)
                + " and "
                + String(t2.seq)
                + " records. A card whose STAGE COUNT is not a function of"
                " the fixture cannot be aligned against another machine's."
            )
    if n1 != 11:
        raise Error(
            "check_ols_card_is_emitted: expected 11 stages (2 inputs, 6"
            " steps, the eigensolver info and the rank, with step 3 in"
            " three parts), got "
            + String(n1)
            + ". If a stage was added or removed on purpose, update this"
            " number AND glm/checks/ols_trace.mojo's stage list, which"
            " is what a reader of a card will go to."
        )
    var diff = first_divergence(p1, p2)
    comptime if IDENTICAL:
        if diff != "":
            raise Error(
                "check_ols_card_is_emitted: the card does not match its own"
                " run-to-run control under IDENTICAL. First divergence:\n  "
                + diff
            )
        print(
            "check_ols_card_is_emitted OK [IDENTICAL]: 11 stages, and two"
            " fits of one fixture agree on every one of them"
        )
    else:
        if diff == "":
            print(
                "check_ols_card_is_emitted OK [FAST]: 11 stages, and the"
                " run-to-run control happens to agree on all of them"
            )
        else:
            print(
                "check_ols_card_is_emitted REPORT [FAST]: 11 stages; the"
                " run-to-run control DIFFERS, first at: "
                + diff
            )


# ---------------------------------------------------------------------------
# THE TWO SHAPES, ORACLED (DEVIATIONS 550 and 551)
# ---------------------------------------------------------------------------


def _wide_fixture(n: Int, d: Int) -> Tuple[List[Float64], List[Float64], List[Float64]]:
    """An `n x d` design with `n < d`, a planted `w*`, and `b = A w*`.

    `b` is built in Float64 from the SAME float32 cells the device sees, so
    `w*` is an exact solution of `A w = b` up to one narrowing per entry.
    That matters for the minimum-norm assertion below: `w*` has to be a
    genuine solution before "the fit's norm is smaller than `w*`'s" means
    anything.
    """
    var a = List[Float64]()
    for i in range(n * d):
        a.append(Float64(_hash_f32(i, 7001)))
    var wstar = List[Float64]()
    for k in range(d):
        wstar.append(Float64(_hash_f32(k, 7002)))
    var b = List[Float64]()
    for i in range(n):
        var acc = 0.0
        for k in range(d):
            acc += a[i * d + k] * wstar[k]
        b.append(Float64(Float32(acc)))
    return (a^, wstar^, b^)


def _fit_from_host64(
    ctx: DeviceContext, a: List[Float64], b: List[Float64], n: Int, d: Int
) raises -> List[Float64]:
    """Push a host fixture through `ols_fit` at whatever shape it is and
    return the coefficients. The dispatch picks the route; this helper does
    not know which one ran, on purpose."""
    var big = max(n, d)
    var da = ctx.enqueue_create_buffer[DType.float32](n * d)
    var db = ctx.enqueue_create_buffer[DType.float32](n)
    var dw = ctx.enqueue_create_buffer[DType.float32](d)
    var cov = ctx.enqueue_create_buffer[DType.float32](big * big)
    var q = ctx.enqueue_create_buffer[DType.float32](big * big)
    var qs = ctx.enqueue_create_buffer[DType.float32](big * big)
    var sv = ctx.enqueue_create_buffer[DType.float32](big)
    var ab = ctx.enqueue_create_buffer[DType.float32](big)
    var inv = ctx.enqueue_create_buffer[DType.float32](big * big)
    var xa = ctx.enqueue_create_buffer[DType.float32](n * d)
    var xa2 = ctx.enqueue_create_buffer[DType.float32](n * d)
    ctx.synchronize()
    var ha = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var hb = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    for i in range(n * d):
        ha.unsafe_ptr().unsafe_store(i, Float32(a[i]))
    for i in range(n):
        hb.unsafe_ptr().unsafe_store(i, Float32(b[i]))
    ctx.enqueue_copy(dst_buf=da, src_ptr=ha.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=db, src_ptr=hb.unsafe_ptr())
    ctx.synchronize()
    ols_fit(
        ctx, da, db, dw, cov, q, qs, sv, ab, inv, xa, xa2, n, d, OLS_ALGO_EIG,
    )
    var hw = ctx.enqueue_create_host_buffer[DType.float32](d)
    ctx.enqueue_copy(dst_ptr=hw.unsafe_ptr(), src_buf=dw)
    ctx.synchronize()
    var out = List[Float64]()
    for k in range(d):
        out.append(Float64(hw.unsafe_ptr().unsafe_load(k)))
    _ = da
    _ = db
    _ = dw
    _ = cov
    _ = q
    _ = qs
    _ = sv
    _ = ab
    _ = inv
    _ = xa
    _ = xa2
    _ = ha^
    _ = hb^
    _ = hw^
    return out^


def _gauss_solve64(
    var m: List[Float64], var rhs: List[Float64], n: Int
) raises -> List[Float64]:
    """Solve `M z = rhs` for an `n x n` row-major `M` by Gaussian
    elimination with partial pivoting, in Float64.

    THE ORACLE'S WHOLE POINT IS THAT IT IS A DIFFERENT PROGRAM. The device
    route inverts `A A^T` through a Jacobi eigendecomposition and a
    `DivideByNonZero` pseudo-inverse; this one does row reduction and shares
    no line, no fold and no threshold with it. A bug common to both would
    have to be a bug in linear algebra.
    """
    var idx = List[Int]()
    for i in range(n):
        idx.append(i)
    for col in range(n):
        var piv = col
        var best = abs(m[idx[col] * n + col])
        for r in range(col + 1, n):
            var v = abs(m[idx[r] * n + col])
            if v > best:
                best = v
                piv = r
        if best == 0.0:
            raise Error(
                "_gauss_solve64: the oracle's own matrix is singular at"
                " column " + String(col) + ", so the fixture is degenerate"
                " and the comparison below would be meaningless"
            )
        var t = idx[col]
        idx[col] = idx[piv]
        idx[piv] = t
        for r in range(col + 1, n):
            var f = m[idx[r] * n + col] / m[idx[col] * n + col]
            for c in range(col, n):
                m[idx[r] * n + c] -= f * m[idx[col] * n + c]
            rhs[idx[r]] -= f * rhs[idx[col]]
    var z = List[Float64]()
    for _ in range(n):
        z.append(0.0)
    for step in range(n):
        var r = n - 1 - step
        var acc = rhs[idx[r]]
        for c in range(r + 1, n):
            acc -= m[idx[r] * n + c] * z[c]
        z[r] = acc / m[idx[r] * n + r]
    return z^


def _residual_inf(
    a: List[Float64], w: List[Float64], b: List[Float64], n: Int, d: Int
) -> Float64:
    """`max_i |(A w - b)_i|`, in Float64."""
    var worst = 0.0
    for i in range(n):
        var acc = 0.0
        for k in range(d):
            acc += a[i * d + k] * w[k]
        var r = abs(acc - b[i])
        if r > worst:
            worst = r
    return worst


def _norm2(w: List[Float64], d: Int) -> Float64:
    var s = 0.0
    for k in range(d):
        s += w[k] * w[k]
    return sqrt(s)


def check_ols_wide_is_the_minimum_norm_solution() raises:
    """`n_cols > n_rows` returns `A^T (A A^T)^+ b`, oracled three ways.

    DEVIATION 550. Three properties, and NONE of them shares a line with
    `lstsq_min_norm.mojo`:

    1. **IT INTERPOLATES.** An underdetermined system with full row rank has
       EXACT solutions, so `max_i |(A w - b)_i|` must be at the float32
       noise floor. A least-squares solver that merely minimised something
       would not hit that.
    2. **IT IS THE MINIMUM-NORM ONE.** The planted `w*` is itself an exact
       solution of `A w = b` by construction, so `||w_fit|| <= ||w*||` must
       hold -- and it holds STRICTLY here, because `w*` is a hashed vector
       with no reason to be the shortest of the infinitely many solutions.
       This is the property that separates the right answer from any other
       interpolant, and it is the one an SVD pseudo-inverse also has, which
       is why the route is a legitimate stand-in for `lstsqSvdJacobi`.
    3. **IT AGREES WITH A FLOAT64 ORACLE.** `_gauss_solve64` solves
       `(A A^T) c = b` by row reduction and forms `A^T c` -- a different
       program, in a different precision, with no eigendecomposition in it.

    THE NEGATIVE CONTROL IS IN THE CHECK. Properties 1 and 2 are re-run on a
    DELIBERATELY PERTURBED coefficient vector and must both FAIL on it. A
    gate that has not been shown capable of failing does not count, and
    running the control every time is stronger than a sabotage somebody
    performed once and reverted.
    """
    var n = 6
    var d = 16
    var fx = _wide_fixture(n, d)
    var a = fx[0].copy()
    var wstar = fx[1].copy()
    var b = fx[2].copy()
    var ctx = DeviceContext()
    var w = _fit_from_host64(ctx, a, b, n, d)

    # scale references, so the tolerances are relative to the fixture
    var bmax = 0.0
    for i in range(n):
        if abs(b[i]) > bmax:
            bmax = abs(b[i])

    # 1. it interpolates
    var res = _residual_inf(a, w, b, n, d)
    var res_bound = 1.0e-4 * (bmax if bmax > 1.0 else 1.0)
    if not (res <= res_bound):
        raise Error(
            "check_ols_wide_is_the_minimum_norm_solution: ||A w - b||_inf "
            + String(res) + " > " + String(res_bound) + ". A wide system"
            " with full row rank has exact solutions; this one is not"
            " interpolating, so the route is not solving the system."
        )

    # 2. it is the SHORTEST of them
    var nfit = _norm2(w, d)
    var nstar = _norm2(wstar, d)
    if not (nfit <= nstar * 1.0001):
        raise Error(
            "check_ols_wide_is_the_minimum_norm_solution: ||w_fit|| "
            + String(nfit) + " exceeds ||w*|| " + String(nstar)
            + ", and w* is an exact solution by construction. The returned"
            " vector interpolates but is not the minimum-norm solution,"
            " which is the vector lstsqSvdJacobi's pseudo-inverse returns"
            " and the one this route claims."
        )

    # 3. the float64 oracle
    var gram = List[Float64]()
    for i in range(n):
        for j in range(n):
            var acc = 0.0
            for k in range(d):
                acc += a[i * d + k] * a[j * d + k]
            gram.append(acc)
    var rhs = List[Float64]()
    for i in range(n):
        rhs.append(b[i])
    var c = _gauss_solve64(gram^, rhs^, n)
    var worst = 0.0
    var worst_k = 0
    for k in range(d):
        var expect = 0.0
        for i in range(n):
            expect += a[i * d + k] * c[i]
        var e = abs(expect - w[k])
        if e > worst:
            worst = e
            worst_k = k
    var oracle_bound = 1.0e-3 * (nstar if nstar > 1.0 else 1.0)
    if not (worst <= oracle_bound):
        raise Error(
            "check_ols_wide_is_the_minimum_norm_solution: coefficient "
            + String(worst_k) + " is " + String(worst) + " away from the"
            " float64 Gaussian-elimination oracle A^T (A A^T)^-1 b (bound "
            + String(oracle_bound) + ")"
        )

    # THE NEGATIVE CONTROL. Perturb one coefficient by 1% of the solution
    # norm and require BOTH properties to reject it.
    var bad = w.copy()
    bad[0] = bad[0] + 0.01 * (nstar if nstar > 1.0 else 1.0)
    var bad_res = _residual_inf(a, bad, b, n, d)
    if bad_res <= res_bound:
        raise Error(
            "check_ols_wide_is_the_minimum_norm_solution: the NEGATIVE"
            " CONTROL passed the interpolation test (residual "
            + String(bad_res) + " <= " + String(res_bound) + "). The bound"
            " is too loose to see a wrong answer, so property 1 above"
            " asserts nothing."
        )
    var longer = w.copy()
    for k in range(d):
        longer[k] = longer[k] * 1.5
    if _norm2(longer, d) <= nstar * 1.0001:
        raise Error(
            "check_ols_wide_is_the_minimum_norm_solution: the NEGATIVE"
            " CONTROL for the norm test passed -- 1.5x the fitted vector is"
            " still shorter than ||w*||, so property 2 has no room to fail"
            " and the fixture is the wrong shape for it."
        )

    print(
        "check_ols_wide_is_the_minimum_norm_solution OK ["
        + _mode_name() + "]: 6 x 16, residual " + String(res)
        + " <= " + String(res_bound) + "; ||w_fit|| " + String(nfit)
        + " < ||w*|| " + String(nstar) + "; float64 oracle within "
        + String(worst) + "; both negative controls rejected"
    )


def check_ols_single_column_matches_the_closed_form() raises:
    """`n_cols == 1` returns `sum(a_i b_i) / sum(a_i^2)`, in Float64.

    DEVIATION 551. A single-column least squares has a closed form with no
    solver in it at all, so the oracle here is not an approximation of the
    right answer, it IS the right answer. The reason this shape is worth its
    own check is that cuML refuses it (`linear_regression.pyx:390-394`,
    "eig solver does not support training data with 1 column currently") and
    this port does not: what is being gated is a place where the two
    libraries deliberately differ.

    THE NEGATIVE CONTROL is a plausible WRONG closed form -- dividing by
    `n` instead of by `sum(a_i^2)`, which is the mistake a reader of
    `A^T b` would make -- and the check asserts the device does NOT match
    it. Without that, a fixture whose `sum(a_i^2)` happened to be near `n`
    would pass either way.
    """
    var n = 512
    var a = List[Float64]()
    for i in range(n):
        a.append(Float64(_hash_f32(i, 7101)))
    var b = List[Float64]()
    for i in range(n):
        b.append(Float64(Float32(a[i] * 3.25 + 0.05 * Float64(_hash_f32(i, 7102)))))
    var ctx = DeviceContext()
    var w = _fit_from_host64(ctx, a, b, n, 1)

    var num = 0.0
    var den = 0.0
    for i in range(n):
        num += a[i] * b[i]
        den += a[i] * a[i]
    var closed = num / den
    var err = abs(w[0] - closed)
    var bound = 1.0e-4 * (abs(closed) if abs(closed) > 1.0 else 1.0)
    if not (err <= bound):
        raise Error(
            "check_ols_single_column_matches_the_closed_form: device "
            + String(w[0]) + " vs float64 closed form " + String(closed)
            + ", |delta| " + String(err) + " > " + String(bound)
        )
    var wrong = num / Float64(n)
    if abs(wrong - closed) <= bound:
        raise Error(
            "check_ols_single_column_matches_the_closed_form: the NEGATIVE"
            " CONTROL (dividing A^T b by n instead of by A^T A) is within"
            " tolerance of the right answer on this fixture, so passing the"
            " test above proves nothing. sum(a^2) = " + String(den)
            + " is too close to n = " + String(n) + "; change the fixture."
        )
    print(
        "check_ols_single_column_matches_the_closed_form OK ["
        + _mode_name() + "]: " + String(w[0]) + " vs " + String(closed)
        + " (|delta| " + String(err) + "); the wrong closed form "
        + String(wrong) + " is distinguishable"
    )


def check_ols_normal_equation_residual_is_zero() raises:
    """The TALL route satisfies the normal equations: `A^T (A w - b) ~ 0`.

    A property, in Float64, that shares nothing with the solver's spelling:
    least squares is DEFINED by `A^T A w = A^T b`, so `A^T (A w - b)` is
    zero at the solution and at no other point when `A^T A` is nonsingular.
    `check_ols_beats_truth_on_noise` compares two residuals and can be
    passed by a vector that is merely good; this one can only be passed by
    the stationary point.

    THE NEGATIVE CONTROL perturbs one coefficient and requires the gradient
    to move past the bound.
    """
    var n = 512
    var d = 6
    var a = List[Float64]()
    for i in range(n * d):
        a.append(Float64(_hash_f32(i, 7201)))
    var b = List[Float64]()
    for i in range(n):
        var acc = 0.0
        for k in range(d):
            acc += a[i * d + k] * (1.0 + 0.5 * Float64(k))
        b.append(Float64(Float32(acc + 0.1 * Float64(_hash_f32(i, 7202)))))
    var ctx = DeviceContext()
    var w = _fit_from_host64(ctx, a, b, n, d)

    var scale = 0.0
    for i in range(n * d):
        if abs(a[i]) > scale:
            scale = abs(a[i])
    var gmax = _normal_grad_inf(a, w, b, n, d)
    # `A^T (A w - b)` has n terms of size ~ |a| * |residual|, so the bound
    # scales with n and with the design, not with 1.
    var bound = 1.0e-3 * Float64(n) * scale * scale
    if not (gmax <= bound):
        raise Error(
            "check_ols_normal_equation_residual_is_zero: ||A^T (A w - b)||_inf "
            + String(gmax) + " > " + String(bound) + ". The returned vector"
            " is not a stationary point of the least-squares objective."
        )
    var bad = w.copy()
    bad[0] = bad[0] + 0.05
    var bad_g = _normal_grad_inf(a, bad, b, n, d)
    if bad_g <= bound:
        raise Error(
            "check_ols_normal_equation_residual_is_zero: the NEGATIVE"
            " CONTROL (coefficient 0 moved by 0.05) still satisfies the"
            " bound " + String(bound) + " at " + String(bad_g)
            + ", so the test above cannot fail."
        )
    print(
        "check_ols_normal_equation_residual_is_zero OK [" + _mode_name()
        + "]: ||A^T (A w - b)||_inf " + String(gmax) + " <= "
        + String(bound) + "; the perturbed vector gives " + String(bad_g)
    )


def _normal_grad_inf(
    a: List[Float64], w: List[Float64], b: List[Float64], n: Int, d: Int
) -> Float64:
    """`max_k |(A^T (A w - b))_k|`, Float64."""
    var resid = List[Float64]()
    for i in range(n):
        var acc = 0.0
        for k in range(d):
            acc += a[i * d + k] * w[k]
        resid.append(acc - b[i])
    var worst = 0.0
    for k in range(d):
        var g = 0.0
        for i in range(n):
            g += a[i * d + k] * resid[i]
        if abs(g) > worst:
            worst = abs(g)
    return worst


# ---------------------------------------------------------------------------
# SAMPLE WEIGHTS (`ols.cuh:99-110`, `:129-141`)
# ---------------------------------------------------------------------------


def _fit_weighted64(
    ctx: DeviceContext,
    a: List[Float64],
    b: List[Float64],
    sw: List[Float64],
    n: Int,
    d: Int,
    mut a_back: List[Float64],
    mut b_back: List[Float64],
    mut sw_back: List[Float64],
) raises -> List[Float64]:
    """`ols_fit_weighted` on a host fixture, returning the coefficients AND
    reading back the three buffers `olsFit` mutates, so the restore
    (`ols.cuh:129-141`) can be checked rather than assumed."""
    var da = ctx.enqueue_create_buffer[DType.float32](n * d)
    var db = ctx.enqueue_create_buffer[DType.float32](n)
    var dsw = ctx.enqueue_create_buffer[DType.float32](n)
    var dw = ctx.enqueue_create_buffer[DType.float32](d)
    var cov = ctx.enqueue_create_buffer[DType.float32](d * d)
    var q = ctx.enqueue_create_buffer[DType.float32](d * d)
    var qs = ctx.enqueue_create_buffer[DType.float32](d * d)
    var sv = ctx.enqueue_create_buffer[DType.float32](d)
    var ab = ctx.enqueue_create_buffer[DType.float32](d)
    var inv = ctx.enqueue_create_buffer[DType.float32](d * d)
    var xa = ctx.enqueue_create_buffer[DType.float32](n * d)
    var xa2 = ctx.enqueue_create_buffer[DType.float32](n * d)
    ctx.synchronize()
    var ha = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var hb = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hs = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    for i in range(n * d):
        ha.unsafe_ptr().unsafe_store(i, Float32(a[i]))
    for i in range(n):
        hb.unsafe_ptr().unsafe_store(i, Float32(b[i]))
        hs.unsafe_ptr().unsafe_store(i, Float32(sw[i]))
    ctx.enqueue_copy(dst_buf=da, src_ptr=ha.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=db, src_ptr=hb.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dsw, src_ptr=hs.unsafe_ptr())
    ctx.synchronize()
    ols_fit_weighted(
        ctx, da, db, dw, cov, q, qs, sv, ab, inv, xa, xa2, dsw, n, d,
        OLS_ALGO_EIG,
    )
    var hw = ctx.enqueue_create_host_buffer[DType.float32](d)
    ctx.enqueue_copy(dst_ptr=hw.unsafe_ptr(), src_buf=dw)
    ctx.enqueue_copy(dst_ptr=ha.unsafe_ptr(), src_buf=da)
    ctx.enqueue_copy(dst_ptr=hb.unsafe_ptr(), src_buf=db)
    ctx.enqueue_copy(dst_ptr=hs.unsafe_ptr(), src_buf=dsw)
    ctx.synchronize()
    var out = List[Float64]()
    for k in range(d):
        out.append(Float64(hw.unsafe_ptr().unsafe_load(k)))
    a_back.clear()
    for i in range(n * d):
        a_back.append(Float64(ha.unsafe_ptr().unsafe_load(i)))
    b_back.clear()
    sw_back.clear()
    for i in range(n):
        b_back.append(Float64(hb.unsafe_ptr().unsafe_load(i)))
        sw_back.append(Float64(hs.unsafe_ptr().unsafe_load(i)))
    _ = da
    _ = db
    _ = dsw
    _ = dw
    _ = cov
    _ = q
    _ = qs
    _ = sv
    _ = ab
    _ = inv
    _ = xa
    _ = xa2
    _ = ha^
    _ = hb^
    _ = hs^
    _ = hw^
    return out^


def check_ols_duplicating_a_row_equals_doubling_its_weight() raises:
    """**THE SAMPLE-WEIGHT PROPERTY**, and it shares no spelling with the
    implementation.

    `sum_i w_i (b_i - a_i . x)^2` with `w_0 = 2` is the same objective as
    the unweighted sum over a design where row 0 appears TWICE. So:

        fit(A, b, w = [2, 1, 1, ...])  ==  fit([a_0; A], [b_0; b], no w)

    to float32 noise. Nothing in `glm/impl/glm/ols.mojo` knows that; it does
    a `sqrt` and a row multiply. A weight applied to the wrong axis, applied
    once instead of as a square root, or applied to `A` and not to `b`, all
    fail this and none of them fails a "did it run" check.

    THE NEGATIVE CONTROL is the UNWEIGHTED fit of the same 64 rows, which
    must NOT match the duplicated fit. Without it, an implementation that
    ignored the weight vector entirely would pass whenever the duplicated
    row barely moved the answer.
    """
    var n = 64
    var d = 4
    var a = List[Float64]()
    for i in range(n * d):
        a.append(Float64(_hash_f32(i, 7301)))
    var b = List[Float64]()
    for i in range(n):
        var acc = 0.0
        for k in range(d):
            acc += a[i * d + k] * (2.0 - 0.6 * Float64(k))
        b.append(Float64(Float32(acc + 0.3 * Float64(_hash_f32(i, 7302)))))

    # ROW 0 IS MADE INFLUENTIAL ON PURPOSE, and the negative control is why.
    # With 64 well-conditioned rows over 4 columns, duplicating one ordinary
    # row moves the least-squares solution by far less than the bound, so the
    # control could not tell the duplicated fit from the plain one and the
    # gate refused itself as VACUOUS on 2026-09-01. That refusal was correct:
    # without a distinguishable control, an implementation that ignored the
    # weight vector entirely would have passed. Giving row 0 a large residual
    # makes it a high-leverage point, so counting it twice actually moves the
    # answer and the control has something to detect.
    b[0] = b[0] + 25.0

    # weighted: row 0 counts twice
    var sw = List[Float64]()
    sw.append(2.0)
    for _ in range(n - 1):
        sw.append(1.0)
    var a_back = List[Float64]()
    var b_back = List[Float64]()
    var sw_back = List[Float64]()
    var ctx = DeviceContext()
    var w_weighted = _fit_weighted64(
        ctx, a, b, sw, n, d, a_back, b_back, sw_back
    )

    # duplicated: row 0 is physically present twice, no weights
    var a_dup = List[Float64]()
    for k in range(d):
        a_dup.append(a[k])
    for i in range(n * d):
        a_dup.append(a[i])
    var b_dup = List[Float64]()
    b_dup.append(b[0])
    for i in range(n):
        b_dup.append(b[i])
    var w_dup = _fit_from_host64(ctx, a_dup, b_dup, n + 1, d)

    # and the unweighted 64-row fit, the negative control
    var w_plain = _fit_from_host64(ctx, a, b, n, d)

    var scale = _norm2(w_dup, d)
    if scale < 1.0:
        scale = 1.0
    var worst = 0.0
    var worst_k = 0
    for k in range(d):
        var e = abs(w_weighted[k] - w_dup[k])
        if e > worst:
            worst = e
            worst_k = k
    var bound = 2.0e-3 * scale
    if not (worst <= bound):
        raise Error(
            "check_ols_duplicating_a_row_equals_doubling_its_weight:"
            " coefficient " + String(worst_k) + " differs by "
            + String(worst) + " between the weighted fit and the"
            " row-duplicated fit (bound " + String(bound) + "). The two are"
            " the same objective, so the weights are being applied wrongly."
        )
    var control = 0.0
    for k in range(d):
        var e = abs(w_plain[k] - w_dup[k])
        if e > control:
            control = e
    if control <= bound:
        raise Error(
            "check_ols_duplicating_a_row_equals_doubling_its_weight: the"
            " NEGATIVE CONTROL passed -- the UNWEIGHTED 64-row fit is"
            " already within " + String(bound) + " of the duplicated fit"
            " (" + String(control) + "), so the agreement above would hold"
            " even if sample_weight were ignored entirely. The fixture needs"
            " a row that matters more."
        )
    print(
        "check_ols_duplicating_a_row_equals_doubling_its_weight OK ["
        + _mode_name() + "]: worst |delta| " + String(worst) + " <= "
        + String(bound) + "; the unweighted control is " + String(control)
        + " away, so the weights demonstrably reached the fit"
    )


def check_ols_sample_weight_restores_its_operands() raises:
    """`ols.cuh:129-141`: `A`, `b` and `sample_weight` come back.

    `olsFit` documents that it modifies the caller's weight vector, and it
    undoes the scaling of `A` and `b` after the solve. A Mojo caller passes
    `DeviceBuffer`s and sees exactly that, so it is ported and gated rather
    than dropped as invisible.

    THE NEGATIVE CONTROL is the SCALED design: the check computes what `A`
    looked like DURING the solve and requires it to be far from what came
    back, so "restored" cannot be satisfied by a no-op that never scaled
    anything in the first place.
    """
    var n = 32
    var d = 3
    var a = List[Float64]()
    for i in range(n * d):
        a.append(Float64(_hash_f32(i, 7401)))
    var b = List[Float64]()
    for i in range(n):
        b.append(Float64(_hash_f32(i, 7402)))
    var sw = List[Float64]()
    for i in range(n):
        # 4, 9, 4, 9, ... : perfect squares, so sqrt and the square back are
        # exact and any drift is the implementation's, not the fixture's.
        sw.append(4.0 if i % 2 == 0 else 9.0)
    var a_back = List[Float64]()
    var b_back = List[Float64]()
    var sw_back = List[Float64]()
    var ctx = DeviceContext()
    var w = _fit_weighted64(ctx, a, b, sw, n, d, a_back, b_back, sw_back)
    _ = len(w)

    var worst_a = 0.0
    for i in range(n * d):
        var e = abs(a_back[i] - a[i])
        if e > worst_a:
            worst_a = e
    var worst_b = 0.0
    var worst_s = 0.0
    for i in range(n):
        var eb = abs(b_back[i] - b[i])
        if eb > worst_b:
            worst_b = eb
        var es = abs(sw_back[i] - sw[i])
        if es > worst_s:
            worst_s = es
    if not (worst_a <= 1.0e-5 and worst_b <= 1.0e-5):
        raise Error(
            "check_ols_sample_weight_restores_its_operands: A drifted by "
            + String(worst_a) + " and b by " + String(worst_b)
            + " across the fit; ols.cuh:129-139 restores both."
        )
    if not (worst_s <= 1.0e-5):
        raise Error(
            "check_ols_sample_weight_restores_its_operands: sample_weight"
            " came back " + String(worst_s) + " away from its input, and"
            " the fixture is perfect squares, so pow(sqrt(w), 2) is exact."
            " ols.cuh:140 is not running."
        )
    # NEGATIVE CONTROL: how far the operands moved DURING the solve.
    var moved = 0.0
    for i in range(n * d):
        var scaled = a[i] * sqrt(sw[i // d])
        var e = abs(scaled - a[i])
        if e > moved:
            moved = e
    if moved <= 1.0e-5:
        raise Error(
            "check_ols_sample_weight_restores_its_operands: the NEGATIVE"
            " CONTROL says the scaling itself would have moved A by only "
            + String(moved) + ", so 'A came back unchanged' is satisfied by"
            " doing nothing. The fixture's weights are too close to 1."
        )
    print(
        "check_ols_sample_weight_restores_its_operands OK [" + _mode_name()
        + "]: A within " + String(worst_a) + ", b within " + String(worst_b)
        + ", w within " + String(worst_s) + "; the scaling itself moves A by "
        + String(moved)
    )


def check_ols_sample_weight_host_rescale_matches_device() raises:
    """**THE GATE THE PYTHON SURFACE RESTS ON.**

    `python/mojolearn/linear_model.py` applies `sqrt(w)` and the row
    multiply in numpy and calls the UNWEIGHTED binding, because
    `bindings/_mojolearn_estimators.mojo` has no weight pointer yet. That is
    only honest if the two routes are the same fit, and this asserts it
    rather than the docstring claiming it:

        ols_fit_weighted_host(X, y, w)  ==  ols_fit_host(X * sqrt(w), y * sqrt(w))

    Under IDENTICAL the coefficient BYTES must be equal: `identical_sqrt` is
    `portable_sqrtf`, correctly rounded, which is what numpy's float32
    `sqrt` is, and the multiply is one rounding on both sides. Under FAST
    the device `sqrt` is the vendor's and the two can differ in the last
    bit, so the check REPORTS the distance instead of asserting it -- the
    same two-mode shape every other identity check in this file has.

    THE NEGATIVE CONTROL fits a DIFFERENT weight vector and requires the
    bytes to move. Without it, a comparator that always compared a buffer
    against itself would pass.
    """
    var n = 128
    var d = 5
    var x = List[Float32]()
    for i in range(n * d):
        x.append(_hash_f32(i, 7501))
    var y = List[Float32]()
    for i in range(n):
        y.append(_hash_f32(i, 7502))
    var wv = List[Float32]()
    for i in range(n):
        wv.append(Float32(0.25) + abs(_hash_f32(i, 7503)))
    var wv2 = List[Float32]()
    for i in range(n):
        wv2.append(Float32(0.25) + abs(_hash_f32(i, 7504)))

    var device = _weighted_host_bits(x, y, wv, n, d)
    var host = _rescaled_host_bits(x, y, wv, n, d)
    var other = _weighted_host_bits(x, y, wv2, n, d)

    var differ = 0
    for k in range(d):
        if device[k] != host[k]:
            differ += 1
    var control = 0
    for k in range(d):
        if device[k] != other[k]:
            control += 1
    if control == 0:
        raise Error(
            "check_ols_sample_weight_host_rescale_matches_device: the"
            " NEGATIVE CONTROL failed -- a DIFFERENT weight vector produced"
            " byte-identical coefficients, so the comparison above is"
            " comparing something to itself."
        )
    comptime if IDENTICAL:
        if differ != 0:
            raise Error(
                "check_ols_sample_weight_host_rescale_matches_device: "
                + String(differ) + " of " + String(d) + " coefficients"
                " differ between the DEVICE weight block and the HOST"
                " rescale under IDENTICAL. python/mojolearn/linear_model.py"
                " does the host rescale and claims they are the same fit;"
                " they are not."
            )
        print(
            "check_ols_sample_weight_host_rescale_matches_device OK"
            " [IDENTICAL]: all " + String(d) + " coefficients byte-identical"
            " between ols_fit_weighted_host and the host rescale; a"
            " different weight vector moves " + String(control) + " of them"
        )
    else:
        print(
            "check_ols_sample_weight_host_rescale_matches_device REPORT"
            " [FAST]: " + String(differ) + " of " + String(d)
            + " coefficients differ between the device weight block and the"
            " host rescale (the vendor sqrt is not pinned under FAST); a"
            " different weight vector moves " + String(control) + " of them"
        )


def _weighted_host_bits(
    x: List[Float32], y: List[Float32], wv: List[Float32], n: Int, d: Int
) raises -> List[UInt32]:
    """`ols_fit_weighted_host` through raw addresses, coefficients as bits."""
    var xs = x.copy()
    var ys = y.copy()
    var ws = wv.copy()
    var coef = List[Float32]()
    for _ in range(d):
        coef.append(Float32(0.0))
    var xp = MutPointer[Float32, MutUntrackedOrigin](
        unsafe_from_address=Int(xs.unsafe_ptr())
    )
    var yp = MutPointer[Float32, MutUntrackedOrigin](
        unsafe_from_address=Int(ys.unsafe_ptr())
    )
    var wp = MutPointer[Float32, MutUntrackedOrigin](
        unsafe_from_address=Int(ws.unsafe_ptr())
    )
    var cp = MutPointer[Float32, MutUntrackedOrigin](
        unsafe_from_address=Int(coef.unsafe_ptr())
    )
    with DeviceContext() as ctx:
        ols_fit_weighted_host(ctx, xp, yp, wp, cp, n, d)
        var out = List[UInt32]()
        for k in range(d):
            out.append(bitcast[DType.uint32](coef[k]))
        _ = xs^
        _ = ys^
        _ = ws^
        _ = coef^
        # DEVIATION 1946: the context dies last.
        _ = ctx^
        return out^


def _rescaled_host_bits(
    x: List[Float32], y: List[Float32], wv: List[Float32], n: Int, d: Int
) raises -> List[UInt32]:
    """The HOST rescale then `ols_fit_host`, i.e. what
    `python/mojolearn/linear_model.py` does, spelled with `identical_sqrt`
    so that under IDENTICAL it is numpy's correctly-rounded float32 sqrt."""
    var xs = List[Float32]()
    var ys = List[Float32]()
    for i in range(n):
        var root = identical_sqrt(wv[i])
        for k in range(d):
            xs.append(x[i * d + k] * root)
        ys.append(y[i] * root)
    var coef = List[Float32]()
    for _ in range(d):
        coef.append(Float32(0.0))
    var xp = MutPointer[Float32, MutUntrackedOrigin](
        unsafe_from_address=Int(xs.unsafe_ptr())
    )
    var yp = MutPointer[Float32, MutUntrackedOrigin](
        unsafe_from_address=Int(ys.unsafe_ptr())
    )
    var cp = MutPointer[Float32, MutUntrackedOrigin](
        unsafe_from_address=Int(coef.unsafe_ptr())
    )
    with DeviceContext() as ctx:
        ols_fit_host(ctx, xp, yp, cp, n, d)
        var out = List[UInt32]()
        for k in range(d):
            out.append(bitcast[DType.uint32](coef[k]))
        _ = xs^
        _ = ys^
        _ = coef^
        _ = ctx^
        return out^


def main() raises:
    print("== glm/checks/ols_check.mojo [" + _mode_name() + "] ==")
    check_ols_exact()
    check_ols_scale_invariant()
    check_ols_beats_truth_on_noise()
    check_ols_dispatch_routes_special_shapes()
    check_ols_wide_is_the_minimum_norm_solution()
    check_ols_single_column_matches_the_closed_form()
    check_ols_normal_equation_residual_is_zero()
    check_ols_duplicating_a_row_equals_doubling_its_weight()
    check_ols_sample_weight_restores_its_operands()
    check_ols_sample_weight_host_rescale_matches_device()
    check_ols_arms_are_pinned()
    check_ols_refuses_over_capacity()
    check_ols_is_launch_invariant()
    check_ols_host_surface_takes_the_guard()
    check_ols_rank_guard_is_absolute()
    check_ols_card_hashes_raw_bytes()
    check_ols_card_is_emitted()
