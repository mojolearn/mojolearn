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
"""

from max.gpu.host import DeviceContext

from glm.ported.glm.ols import OLS_ALGO_EIG, ols_fit
from glm.ported.linalg.detail.lstsq import lstsq_eig


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


def check_ols_dispatch_guard() raises:
    """`ols.cuh:112-113` REFUSES the two shapes eig cannot solve.

    Before this guard was ported, both of these reached the normal-equations
    solver and returned a plausible vector from a singular `A^T A`. Neither
    is an exotic input: `n_cols > n_rows` is every wide design, and
    `n_cols == 1` is a simple regression.

    The assertion is that the call RAISES. A check that merely ran them and
    looked at the numbers would have passed before the fix, because garbage
    from a singular inverse is finite and plausibly scaled.
    """
    var ctx = DeviceContext()

    # n_cols > n_rows. Their guard switches to lstsqSvdJacobi; ours refuses.
    var wide_raised = False
    try:
        var wide = _solve_shaped(ctx, 4, 8)
        _ = len(wide)
    except e:
        wide_raised = True
    if not wide_raised:
        raise Error(
            "check_ols_dispatch_guard: n_cols > n_rows did NOT raise. The"
            " ols.cuh:113 guard is not reached, and a singular A^T A is"
            " being inverted."
        )

    # n_cols == 1, the case cuML's Python layer warns about by name.
    var single_raised = False
    try:
        var single = _solve_shaped(ctx, 64, 1)
        _ = len(single)
    except e:
        single_raised = True
    if not single_raised:
        raise Error(
            "check_ols_dispatch_guard: n_cols == 1 did NOT raise. The"
            " ols.cuh:113 guard is not reached."
        )

    # And the guard must not fire on an ordinary shape, or it would be
    # refusing everything and the two assertions above would be vacuous.
    var ok_raised = False
    try:
        var fine = _solve_shaped(ctx, 256, 4)
        _ = len(fine)
    except e:
        ok_raised = True
    if ok_raised:
        raise Error(
            "check_ols_dispatch_guard: an ordinary 256 x 4 design raised."
            " The guard is over-firing and the other two assertions prove"
            " nothing."
        )

    print(
        "check_ols_dispatch_guard OK: n_cols>n_rows and n_cols==1 both"
        " refused, 256x4 accepted"
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
#              and the one shape whose only other arm is a closed vendor
#              library RAISES by name.

from std.memory import bitcast

from core.gemm import gemm_nt, gemm_tn, gemv_n
from core.gram_splitk import (
    GRAM_MAX_COLS,
    GRAM_SPLITK_RESOLVED_COLUMN,
    gram_splitk_applies,
)
from core.identity_trace import IdentityTrace, first_divergence, fnv1a64_bytes
from glm.estimator import ols_fit_host
from glm.mojo_only.ols_trace import (
    OLS_CARD_COLS,
    OLS_CARD_ROWS,
    _hash_f32,
    emit_ols_card,
)
from glm.ported.linalg.detail.lstsq import OLS_ELEM_TPB, OLS_NONZERO_THRESH
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name
from mojo_only.kernel_matrix import (
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
    reaches step 1.
    """
    var d = GRAM_MAX_COLS + 2
    var n = 512
    if d >= n:
        raise Error(
            "check_ols_refuses_over_capacity: the fixture is n_cols >="
            " n_rows, so ols.cuh:113 refuses it before step 1 and this"
            " check would be measuring the wrong guard."
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
    for i in range(n * d):
        big.unsafe_ptr().unsafe_store(i, _hash_f32(i, salt))
    ctx.enqueue_copy(dst_buf=a, src_ptr=big.unsafe_ptr())
    for i in range(n):
        big.unsafe_ptr().unsafe_store(i, _hash_f32(i, salt + 1))
    ctx.enqueue_copy(dst_buf=b, src_ptr=big.unsafe_ptr())
    ctx.synchronize()

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


def _host_fit_raises(n: Int, d: Int) raises -> Bool:
    """Run `ols_fit_host` -- the entry `bindings/_mojolearn_estimators.mojo`
    and therefore `mojolearn.LinearRegression` reach -- and say whether it
    raised."""
    var x = List[Float32]()
    for i in range(n * d):
        x.append(_hash_f32(i, 909))
    var y = List[Float32]()
    for i in range(n):
        y.append(_hash_f32(i, 910))
    var coef = List[Float32]()
    for _ in range(d):
        coef.append(Float32(0.0))
    # The host surface takes raw addresses, exactly as
    # `bindings/_mojolearn_estimators.mojo::_f32_ptr` hands them over from
    # numpy. Building them the same way here is what makes this a check of
    # the PYTHON-FACING entry rather than of a Mojo-shaped imitation of it.
    var xp = MutPointer[Float32, MutUntrackedOrigin](
        unsafe_from_address=Int(x.unsafe_ptr())
    )
    var yp = MutPointer[Float32, MutUntrackedOrigin](
        unsafe_from_address=Int(y.unsafe_ptr())
    )
    var wp = MutPointer[Float32, MutUntrackedOrigin](
        unsafe_from_address=Int(coef.unsafe_ptr())
    )
    var raised = False
    with DeviceContext() as ctx:
        try:
            ols_fit_host(ctx, xp, yp, wp, n, d)
        except e:
            raised = True
    _ = x^
    _ = y^
    _ = coef^
    return raised


def check_ols_host_surface_takes_the_guard() raises:
    """`ols_fit_host` goes THROUGH `olsFit`'s dispatch, not around it.

    THE DEFECT THIS GATES, FOUND BY DEVIATION 527's AUDIT AND NOT BY ANY
    GREEN CHECK. `glm/ported/glm/ols.mojo` exists because reaching
    `lstsq_eig` directly skips `ols.cuh:112-113`, which switches away from
    the normal-equations solver at `n_cols > n_rows` (where `A^T A` is
    singular by construction) and at `n_cols == 1` (which cuML's own Python
    layer refuses by name). That file's docstring records the bypass as
    closed. **It was closed for the Mojo callers only.** `ols_fit_host` --
    the host-pointer surface, the one with `mojolearn.LinearRegression` on
    the other end of it -- still called `lstsq_eig` directly, so a Python
    user handing in a wide design or a single column got a plausible vector
    out of a singular inverse and no error.

    `check_ols_dispatch_guard` above did not catch it because it exercises
    `ols_fit`, which was never the bypassed entry. A guard is only reached
    from the doors that were checked.

    Mode-independent: this is a shape refusal, not a numeric one, and it
    must hold in both.
    """
    if not _host_fit_raises(4, 8):
        raise Error(
            "check_ols_host_surface_takes_the_guard: ols_fit_host accepted"
            " a 4 x 8 design (n_cols > n_rows). A^T A is singular by"
            " construction there and the coefficients it returned came out"
            " of a singular inverse. The host surface is bypassing"
            " ols.cuh:113 -- see glm/estimator.mojo."
        )
    if not _host_fit_raises(64, 1):
        raise Error(
            "check_ols_host_surface_takes_the_guard: ols_fit_host accepted"
            " a single-column design. cuML forces the same switch and says"
            " the eig solver does not support one column"
            " (linear_regression.pyx:390)."
        )
    if _host_fit_raises(256, 4):
        raise Error(
            "check_ols_host_surface_takes_the_guard: an ordinary 256 x 4"
            " design raised through the host surface, so the guard is"
            " over-firing and the two assertions above prove nothing."
        )
    print(
        "check_ols_host_surface_takes_the_guard OK ["
        + _mode_name()
        + "]: the Python-facing entry refuses 4x8 and 64x1 and accepts"
        " 256x4, i.e. it goes through olsFit's dispatch"
    )


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

    It is NOT fixed here. `glm/ported/` is COPY-DO-NOT-IMPROVE and changing
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
            " number AND glm/mojo_only/ols_trace.mojo's stage list, which"
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


def main() raises:
    print("== glm/mojo_only/ols_check.mojo [" + _mode_name() + "] ==")
    check_ols_exact()
    check_ols_scale_invariant()
    check_ols_beats_truth_on_noise()
    check_ols_dispatch_guard()
    check_ols_arms_are_pinned()
    check_ols_refuses_over_capacity()
    check_ols_is_launch_invariant()
    check_ols_host_surface_takes_the_guard()
    check_ols_rank_guard_is_absolute()
    check_ols_card_hashes_raw_bytes()
    check_ols_card_is_emitted()
